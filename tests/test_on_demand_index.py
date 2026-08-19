"""On-demand indexing and honest pending states (AAA).

The AI panel's contract when a file has no derived rows yet:
  - per-file endpoints distinguish "the worker has not reached this file"
    (pending) from "scanned, nothing found" and from "no stage will ever
    process this type";
  - POST /index front-queues the file with the running worker (whose
    thread owns every loaded model -- nothing runs inline then, avoiding
    cross-thread model sharing) and wakes it; without a worker it runs
    the fast stages inline on backends leased from the process registry,
    recording scans the same way the worker would;
  - /status carries gallery-wide backlog totals for progress display;
  - the worker wakes on a priority request instead of sleeping out its
    poll interval, and its cycle log shows progress only when work happened.

Model-free: stub backends and real tiny PNGs only.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from flask import Flask
from PIL import Image

from smartgallery_ai import RUBRIC_VERSION, SPACE_SEMANTIC, SPACE_VISUAL, AIConfig, backends, hashing, vectors
from smartgallery_ai import worker as W
from smartgallery_ai.embedders import StubSemanticEmbedder, StubVisualEmbedder
from smartgallery_ai.faces import (
    FaceDetection,
    StubFaceBackend,
    cluster_faces,
    get_face_backend,
    replace_faces_for_file,
)
from smartgallery_ai.review import Finding, ReviewResult, store_review
from smartgallery_ai.schema import init_schema
from smartgallery_ai.service import _index_one_file, create_ai_blueprint, set_worker
from smartgallery_ai.worker import AIWorker, app_git_ref, indexing_totals, record_scan

_PREFIX = "/aidam"


def _make_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE files (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            mtime REAL NOT NULL,
            name TEXT NOT NULL,
            type TEXT,
            size INTEGER DEFAULT 0,
            workflow_prompt TEXT DEFAULT ''
        )
        """
    )
    init_schema(conn)
    return conn


def _add_image_file(conn, tmp_path, file_id: str, mtime: float = 1000.0) -> str:
    path = str(tmp_path / f"{file_id}.png")
    Image.new("RGB", (16, 16), (40, 90, 200)).save(path)
    conn.execute(
        "INSERT INTO files (id, path, mtime, name, type) VALUES (?, ?, ?, ?, 'image')",
        (file_id, path, mtime, file_id),
    )
    conn.commit()
    return path


def _add_typed_file(conn, file_id: str, file_type: str) -> None:
    conn.execute(
        "INSERT INTO files (id, path, mtime, name, type) VALUES (?, ?, 1000.0, ?, ?)",
        (file_id, f"/gallery/{file_id}", file_id, file_type),
    )
    conn.commit()


def _stub_resolver(monkeypatch, kind: str, resolve):
    """Point one backend-registry kind at `resolve` for this test.

    The rest of the kind's spec -- which AIConfig fields select a distinct
    instance, whether "stub" bypasses the cache -- stays exactly as
    production declares it, so a test cannot accidentally exercise keying
    the real code does not use.
    """
    monkeypatch.setitem(backends._KINDS, kind, backends._KINDS[kind]._replace(resolve=resolve))


def _cfg(tmp_path, **overrides) -> AIConfig:
    base = {
        "enabled": True,
        "base_path": str(tmp_path),
        "db_path": str(tmp_path / "gallery.sqlite"),
        "models_dir": str(tmp_path / "models"),
        "cache_dir": str(tmp_path / "cache"),
        "ephemeral_index": True,
        "semantic_backend": "stub",
        "visual_backend": "stub",
        "face_backend": "none",
        "critic_backend": "none",
        "segmenter_backend": "none",
    }
    base.update(overrides)
    return AIConfig(**base)


# --- worker: priority queue and wake ------------------------------------------


def test_request_priority_index_dedupes_bounds_and_wakes(tmp_path):
    """Duplicate requests collapse to one queue slot, a full queue refuses
    (returns False), and any accepted or duplicate request sets the wake
    event so the sleeping loop starts a cycle immediately."""
    cfg = _cfg(tmp_path)
    _make_db(cfg.db_path).close()
    worker = AIWorker(cfg, cfg.db_path, poll_interval=999.0, batch_size=0)
    worker._priority_max = 2

    assert worker.request_priority_index("f1") is True
    assert worker.request_priority_index("f1") is True
    assert worker._priority_ids == ["f1"]

    assert worker.request_priority_index("f2") is True
    assert worker.request_priority_index("f3") is False
    assert worker._priority_ids == ["f1", "f2"]
    assert worker._wake_event.is_set()


def test_priority_file_fully_indexed_outside_the_cycle_budget(tmp_path):
    """With a zero per-cycle budget the backlog is untouched, but a
    priority-requested file still gets hashed and embedded in both spaces
    in the same cycle: user requests never wait behind the crawl budget."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    _add_image_file(conn, tmp_path, "backlog_file")
    _add_image_file(conn, tmp_path, "urgent_file")
    conn.close()

    worker = AIWorker(cfg, cfg.db_path, poll_interval=999.0, batch_size=0)
    worker.request_priority_index("urgent_file")
    worker._run_cycle()

    conn = sqlite3.connect(cfg.db_path)
    try:
        hashed = {r[0] for r in conn.execute("SELECT file_id FROM ai_file_hashes")}
        embedded = {(r[0], r[1]) for r in conn.execute("SELECT file_id, space FROM ai_embeddings")}
    finally:
        conn.close()
    assert hashed == {"urgent_file"}
    assert ("urgent_file", SPACE_SEMANTIC) in embedded
    assert ("urgent_file", SPACE_VISUAL) in embedded
    assert not any(fid == "backlog_file" for fid, _ in embedded)


def test_priority_request_wakes_a_sleeping_worker(tmp_path):
    """A priority request breaks the between-cycle sleep: with a 30s poll
    interval the requested file's rows appear within a couple of seconds,
    which is only possible when the wake event interrupts the wait."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    path = _add_image_file(conn, tmp_path, "wake_file")
    conn.close()
    assert path

    worker = AIWorker(cfg, cfg.db_path, poll_interval=30.0, batch_size=0)
    worker.start()
    try:
        deadline = time.time() + 10.0
        while worker.stats["cycles"] < 1 and time.time() < deadline:
            time.sleep(0.05)
        assert worker.stats["cycles"] >= 1, "first cycle never ran"

        worker.request_priority_index("wake_file")
        deadline = time.time() + 5.0
        hashed = False
        while time.time() < deadline:
            conn = sqlite3.connect(cfg.db_path)
            try:
                hashed = conn.execute("SELECT 1 FROM ai_file_hashes WHERE file_id = 'wake_file'").fetchone() is not None
            finally:
                conn.close()
            if hashed:
                break
            time.sleep(0.1)
        assert hashed, "priority request did not wake the worker within 5s"
    finally:
        worker.stop(timeout=5.0)


def test_stop_interrupts_the_between_cycle_sleep(tmp_path):
    """stop() returns promptly even mid-sleep on a long poll interval."""
    cfg = _cfg(tmp_path)
    _make_db(cfg.db_path).close()
    worker = AIWorker(cfg, cfg.db_path, poll_interval=60.0, batch_size=0)
    worker.start()
    deadline = time.time() + 10.0
    while worker.stats["cycles"] < 1 and time.time() < deadline:
        time.sleep(0.05)

    started = time.time()
    worker.stop(timeout=5.0)
    assert not worker.is_running
    assert time.time() - started < 5.0


# --- shared scan marker --------------------------------------------------------


def test_sync_index_faces_scan_suppresses_worker_rescan(tmp_path):
    """_index_one_file records the faces scan exactly like a worker stage
    would, so the worker's candidate query no longer offers the file."""
    cfg = _cfg(tmp_path, face_backend="stub")
    conn = _make_db(cfg.db_path)
    _add_image_file(conn, tmp_path, "sync_file")

    file_row = conn.execute("SELECT * FROM files WHERE id = 'sync_file'").fetchone()
    result = _index_one_file(conn, cfg, file_row, force=False)
    assert result["faces"] is True

    worker = AIWorker(cfg, cfg.db_path, poll_interval=999.0, batch_size=10)
    remaining = worker._scan_candidates(conn, "faces", StubFaceBackend(lambda _img: []), 10)
    conn.close()
    assert [r["id"] for r in remaining] == []


def test_sync_index_loads_each_backend_once_across_requests(tmp_path, monkeypatch):
    """The inline path leases from the process registry, so N requests load
    each backend ONCE.

    Resolving per call is what this pins against: with the real backends that
    is an open_clip and a dinov2 torch model loaded on every HTTP request,
    none of which is ever freed.
    """
    cfg = _cfg(tmp_path, semantic_backend="auto", visual_backend="auto")
    conn = _make_db(cfg.db_path)
    _add_image_file(conn, tmp_path, "first")
    _add_image_file(conn, tmp_path, "second")

    resolved = {"semantic": 0, "visual": 0}

    def _counting(kind, make):
        def resolve(_config):
            resolved[kind] += 1
            return make()

        return resolve

    _stub_resolver(monkeypatch, "semantic", _counting("semantic", StubSemanticEmbedder))
    _stub_resolver(monkeypatch, "visual", _counting("visual", StubVisualEmbedder))
    backends.reset()

    for file_id in ("first", "second"):
        row = conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()
        assert _index_one_file(conn, cfg, row, force=False)["embedded"] == [SPACE_SEMANTIC, SPACE_VISUAL]
    conn.close()

    assert resolved == {"semantic": 1, "visual": 1}


def test_lease_excludes_concurrent_callers_of_an_unsafe_backend(tmp_path, monkeypatch):
    """A backend that does not declare `thread_safe` is leased exclusively:
    the second thread waits rather than calling a stateful detector that is
    already mid-inference on another thread."""
    cfg = _cfg(tmp_path, face_backend="auto")

    class _Unsafe(StubFaceBackend):
        thread_safe = False

    _stub_resolver(monkeypatch, "faces", lambda _config: _Unsafe(lambda _img: []))
    backends.reset()

    inside = threading.Semaphore(0)
    release = threading.Event()
    overlapped = []

    def hold():
        with backends.lease("faces", cfg):
            inside.release()
            release.wait(5.0)

    def probe():
        with backends.lease("faces", cfg):
            overlapped.append(release.is_set())

    holder = threading.Thread(target=hold)
    holder.start()
    assert inside.acquire(timeout=5.0), "first lease never entered"

    prober = threading.Thread(target=probe)
    prober.start()
    prober.join(timeout=0.5)
    assert prober.is_alive(), "second caller entered while the backend was leased"

    release.set()
    holder.join(timeout=5.0)
    prober.join(timeout=5.0)
    assert overlapped == [True]  # it got in only after the holder let go


def test_shared_refuses_a_backend_that_is_not_thread_safe(tmp_path, monkeypatch):
    """`shared` hands out an instance callers may keep past the block only
    when it is safe unguarded; anything else answers None rather than
    escaping without its lock."""
    cfg = _cfg(tmp_path, semantic_backend="auto", visual_backend="auto")

    class _Unsafe(StubVisualEmbedder):
        thread_safe = False

    _stub_resolver(monkeypatch, "visual", lambda _config: _Unsafe())
    _stub_resolver(monkeypatch, "semantic", lambda _config: StubSemanticEmbedder())
    backends.reset()

    assert backends.shared("visual", cfg) is None
    assert isinstance(backends.shared("semantic", cfg), StubSemanticEmbedder)


def test_forget_unavailable_reresolves_only_the_missing_backends(tmp_path, monkeypatch):
    """Weights landing mid-process must be able to activate a backend that
    previously answered None, without discarding the instances already
    loaded (reloading those costs the weights again)."""
    cfg = _cfg(tmp_path, semantic_backend="auto", visual_backend="auto")

    available = {"visual": False}
    resolved = {"semantic": 0, "visual": 0}

    def _semantic(_config):
        resolved["semantic"] += 1
        return StubSemanticEmbedder()

    def _visual(_config):
        resolved["visual"] += 1
        return StubVisualEmbedder() if available["visual"] else None

    _stub_resolver(monkeypatch, "semantic", _semantic)
    _stub_resolver(monkeypatch, "visual", _visual)
    backends.reset()

    assert backends.shared("semantic", cfg) is not None
    assert backends.shared("visual", cfg) is None

    available["visual"] = True
    backends.forget_unavailable()

    assert backends.shared("visual", cfg) is not None
    assert resolved == {"semantic": 1, "visual": 2}  # the loaded one was kept


# --- indexing totals -----------------------------------------------------------


def test_indexing_totals_counts_files_and_stage_coverage(tmp_path):
    """Totals reflect the files table and each stage's covered set;
    non-renderable types count in files_total but not visual_files_total."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    _add_image_file(conn, tmp_path, "img1")
    _add_image_file(conn, tmp_path, "img2")
    _add_typed_file(conn, "song", "music")

    now = time.time()
    hashing.upsert_hashes(
        conn,
        "img1",
        hashing.HashResult(sha256="a" * 64, phash64=0, dhash64=0),
        1000.0,
        "algo-v1",
        now,
    )
    backend = StubSemanticEmbedder()
    record_scan(conn, "img1", "faces", backend, 1000.0, now, 0)
    record_scan(conn, "img1", "review", backend, 1000.0, now, -1)

    totals = indexing_totals(conn)
    conn.close()
    assert totals == {
        "files_total": 3,
        "visual_files_total": 2,
        "hashed": 1,
        "embeddings_semantic": 0,
        "embeddings_visual": 0,
        "faces_scanned": 1,
        "reviews_scanned": 1,
    }


# --- cycle progress log --------------------------------------------------------


def test_cycle_logs_progress_only_when_work_happened(tmp_path, caplog):
    """A cycle that indexed something emits one '[AIWorker] indexed:' INFO
    line with backlog totals; an idle cycle stays silent."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    _add_image_file(conn, tmp_path, "logged_file")
    conn.close()

    worker = AIWorker(cfg, cfg.db_path, poll_interval=999.0, batch_size=10)
    with caplog.at_level(logging.INFO, logger="smartgallery_ai.worker"):
        worker._run_cycle()
    progress_lines = [r for r in caplog.records if "indexed:" in r.getMessage()]
    assert len(progress_lines) == 1
    assert "1/1 hashed" in progress_lines[0].getMessage()

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="smartgallery_ai.worker"):
        worker._run_cycle()
    assert not [r for r in caplog.records if "indexed:" in r.getMessage()]


# --- service: pending flags and the /index kick --------------------------------


@pytest.fixture
def api(tmp_path):
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    _add_image_file(conn, tmp_path, "fresh_img")
    _add_typed_file(conn, "song", "music")
    conn.close()

    app = Flask(__name__)
    app.register_blueprint(create_ai_blueprint(cfg), url_prefix=_PREFIX)
    return SimpleNamespace(cfg=cfg, client=app.test_client(), tmp_path=tmp_path)


def test_similar_pending_only_for_renderable_unembedded_files(api):
    """An unembedded image reports pending=True ('not indexed yet'); a
    music file reports pending=False (no stage will ever embed it)."""
    img = api.client.get(f"{_PREFIX}/similar/fresh_img").get_json()
    assert img["pending"] is True
    assert img["note"] == "not indexed yet"

    song = api.client.get(f"{_PREFIX}/similar/song").get_json()
    assert song["pending"] is False
    assert song["note"] == "no embedding for this file"


def test_duplicates_pending_until_hashed(api):
    """pending flips False as soon as the anchor file has hash rows."""
    before = api.client.get(f"{_PREFIX}/duplicates/fresh_img").get_json()
    assert before["pending"] is True

    conn = sqlite3.connect(api.cfg.db_path)
    conn.row_factory = sqlite3.Row
    hashing.upsert_hashes(
        conn,
        "fresh_img",
        hashing.HashResult(sha256="d" * 64, phash64=0, dhash64=0),
        1000.0,
        "algo-v1",
        time.time(),
    )
    conn.close()

    after = api.client.get(f"{_PREFIX}/duplicates/fresh_img").get_json()
    assert after["pending"] is False
    assert after["exact"] == []
    assert after["near"] == []


def test_faces_pending_vs_scanned_zero_faces(api):
    """No scan row -> pending; a recorded zero-face scan -> a definitive
    empty (pending False), which is a different UI state."""
    before = api.client.get(f"{_PREFIX}/faces/fresh_img").get_json()
    assert before["pending"] is True

    conn = sqlite3.connect(api.cfg.db_path)
    conn.row_factory = sqlite3.Row
    record_scan(conn, "fresh_img", "faces", StubSemanticEmbedder(), 1000.0, time.time(), 0)
    conn.close()

    after = api.client.get(f"{_PREFIX}/faces/fresh_img").get_json()
    assert after["pending"] is False
    assert after["faces"] == []


def test_review_pending_vs_recorded_failure(api):
    """No scan row -> pending; a result_count=-1 scan row -> scan_failed
    (the one attempt failed; it is NOT still pending)."""
    before = api.client.get(f"{_PREFIX}/review/fresh_img").get_json()
    assert before["pending"] is True
    assert before["scan_failed"] is False

    conn = sqlite3.connect(api.cfg.db_path)
    conn.row_factory = sqlite3.Row
    record_scan(conn, "fresh_img", "review", StubSemanticEmbedder(), 1000.0, time.time(), -1)
    conn.close()

    after = api.client.get(f"{_PREFIX}/review/fresh_img").get_json()
    assert after["pending"] is False
    assert after["scan_failed"] is True


def test_status_carries_indexing_backlog_totals(api):
    """/status exposes the same totals indexing_totals computes, so the
    panel can render gallery-wide progress."""
    status = api.client.get(f"{_PREFIX}/status").get_json()
    assert status["indexing"]["files_total"] == 2
    assert status["indexing"]["visual_files_total"] == 1
    assert set(status["indexing"]) == {
        "files_total",
        "visual_files_total",
        "hashed",
        "embeddings_semantic",
        "embeddings_visual",
        "faces_scanned",
        "reviews_scanned",
    }


def test_index_endpoint_defers_entirely_to_a_running_worker(api):
    """With a running worker POST /index front-queues the file and runs
    NOTHING inline: the worker thread owns every loaded model, and sharing
    its live instances with the request thread would be a data race."""
    queued = []
    fake_worker = SimpleNamespace(
        is_running=True,
        request_priority_index=lambda fid: queued.append(fid) or True,
    )
    set_worker(fake_worker)
    try:
        data = api.client.post(f"{_PREFIX}/index/fresh_img", json={}).get_json()
    finally:
        set_worker(None)
    assert data["worker_queued"] is True
    assert queued == ["fresh_img"]
    assert "hashed" not in data  # nothing ran inline

    conn = sqlite3.connect(api.cfg.db_path)
    try:
        inline_rows = conn.execute("SELECT COUNT(*) FROM ai_file_hashes").fetchone()[0]
    finally:
        conn.close()
    assert inline_rows == 0


def test_index_endpoint_force_reschedules_review_before_queueing(api):
    """force=true clears the file's review scan-log row so the worker's
    priority pass re-reviews it."""
    conn = sqlite3.connect(api.cfg.db_path)
    conn.row_factory = sqlite3.Row
    record_scan(conn, "fresh_img", "review", StubSemanticEmbedder(), 1000.0, time.time(), 2)
    conn.close()

    fake_worker = SimpleNamespace(is_running=True, request_priority_index=lambda _fid: True)
    set_worker(fake_worker)
    try:
        data = api.client.post(f"{_PREFIX}/index/fresh_img", json={"force": True}).get_json()
    finally:
        set_worker(None)
    assert data["review_rescheduled"] is True

    conn = sqlite3.connect(api.cfg.db_path)
    try:
        remaining = conn.execute(
            "SELECT COUNT(*) FROM ai_scan_log WHERE file_id = 'fresh_img' AND kind = 'review'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert remaining == 0


def test_index_endpoint_reports_worker_not_queued_when_absent(api):
    """Without a running worker the inline stages still run but the
    response says the background stages were not queued."""
    set_worker(None)
    data = api.client.post(f"{_PREFIX}/index/fresh_img", json={}).get_json()
    assert data["hashed"] is True
    assert data["worker_queued"] is False


def test_status_reports_backend_devices_key(api):
    """/status carries a devices map (torch device per probed backend;
    null for backends without a device concept, like the stubs here)."""
    status = api.client.get(f"{_PREFIX}/status").get_json()
    assert set(status["devices"]) == {"semantic", "visual", "face", "critic", "segmenter"}
    assert all(device is None for device in status["devices"].values())


def test_sync_index_faces_queue_a_recluster_for_the_worker(tmp_path):
    """Faces stored by the sync /index path set the worker's
    cluster-pending marker, and the worker's next faces stage runs
    clustering (clearing the marker) even with zero new scan candidates --
    without this, sync-indexed faces stay unclustered indefinitely."""
    cfg = _cfg(tmp_path, face_backend="stub")
    conn = _make_db(cfg.db_path)
    _add_image_file(conn, tmp_path, "clustered_file")

    file_row = conn.execute("SELECT * FROM files WHERE id = 'clustered_file'").fetchone()
    _index_one_file(conn, cfg, file_row, force=False)

    markers = [r[0] for r in conn.execute("SELECT key FROM ai_dam_state WHERE key LIKE 'faces_cluster_pending:%'")]
    assert len(markers) == 1

    worker = AIWorker(cfg, cfg.db_path, poll_interval=999.0, batch_size=10)
    worker._process_faces(conn, get_face_backend(cfg), 10)
    remaining = conn.execute("SELECT key FROM ai_dam_state WHERE key LIKE 'faces_cluster_pending:%'").fetchall()
    conn.close()
    assert remaining == []


def test_faces_modified_file_reads_as_pending_again(api):
    """A scan recorded at an older mtime does NOT count as scanned: the
    worker will rescan the modified file, so the panel must say pending,
    not 'scanned — nothing found'."""
    conn = sqlite3.connect(api.cfg.db_path)
    conn.row_factory = sqlite3.Row
    record_scan(conn, "fresh_img", "faces", StubSemanticEmbedder(), 1000.0, time.time(), 0)
    conn.close()
    scanned = api.client.get(f"{_PREFIX}/faces/fresh_img").get_json()
    assert scanned["pending"] is False

    conn = sqlite3.connect(api.cfg.db_path)
    conn.execute("UPDATE files SET mtime = 2000.0 WHERE id = 'fresh_img'")
    conn.commit()
    conn.close()
    modified = api.client.get(f"{_PREFIX}/faces/fresh_img").get_json()
    assert modified["pending"] is True


# --- crawl order + throughput --------------------------------------------------


def test_backlog_processes_newest_files_first(tmp_path):
    """Indexing is strictly newest-first (mtime descending): with a budget
    of 2 and three files, the two most recent get hashed AND embedded in
    the first cycle; the oldest waits."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    _add_image_file(conn, tmp_path, "oldest", mtime=1000.0)
    _add_image_file(conn, tmp_path, "yesterday", mtime=2000.0)
    _add_image_file(conn, tmp_path, "today", mtime=3000.0)
    conn.close()

    worker = AIWorker(cfg, cfg.db_path, poll_interval=999.0, batch_size=2)
    worker._run_cycle()

    conn = sqlite3.connect(cfg.db_path)
    try:
        hashed = {r[0] for r in conn.execute("SELECT file_id FROM ai_file_hashes")}
        embedded = {r[0] for r in conn.execute("SELECT DISTINCT file_id FROM ai_embeddings")}
    finally:
        conn.close()
    assert hashed == {"today", "yesterday"}
    # The even budget split gives each embedding space one slot this
    # cycle; both spend it on the NEWEST file.
    assert embedded == {"today"}


def test_hashing_no_longer_starves_the_model_stages(tmp_path):
    """Hashing runs against its own budget: even with a hash backlog at or
    beyond batch_size, embeddings still happen in the same cycle (before
    this fix a 42k-file gallery hashed for hours with zero embeddings)."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    for i in range(4):
        _add_image_file(conn, tmp_path, f"file{i}", mtime=1000.0 + i)
    conn.close()

    worker = AIWorker(cfg, cfg.db_path, poll_interval=999.0, batch_size=2)
    worker._run_cycle()

    conn = sqlite3.connect(cfg.db_path)
    try:
        hashed = conn.execute("SELECT COUNT(*) FROM ai_file_hashes").fetchone()[0]
        embedded = conn.execute("SELECT COUNT(*) FROM ai_embeddings").fetchone()[0]
    finally:
        conn.close()
    assert hashed == 2
    assert embedded == 2  # one slot per embedding space, same cycle


def test_loop_skips_sleep_while_backlog_remains(tmp_path):
    """With batch_size=1 and a 60s poll interval, four files all get
    hashed within seconds -- only possible when full-budget cycles skip
    the between-cycle sleep and continue the crawl immediately."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    for i in range(4):
        _add_image_file(conn, tmp_path, f"burst{i}", mtime=1000.0 + i)
    conn.close()

    worker = AIWorker(cfg, cfg.db_path, poll_interval=60.0, batch_size=1)
    worker.start()
    try:
        deadline = time.time() + 10.0
        count = 0
        while time.time() < deadline:
            conn = sqlite3.connect(cfg.db_path)
            try:
                count = conn.execute("SELECT COUNT(*) FROM ai_file_hashes").fetchone()[0]
            finally:
                conn.close()
            if count >= 4:
                break
            time.sleep(0.1)
        assert count >= 4, f"only {count}/4 hashed: loop slept between batches"
    finally:
        worker.stop(timeout=5.0)


def test_priority_request_served_mid_stage(tmp_path):
    """A priority request that arrives while a crawl batch is mid-flight
    is served between items of the SAME cycle, not after it: the panel
    never waits behind a long batch."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    for i in range(3):
        _add_image_file(conn, tmp_path, f"crawl{i}", mtime=2000.0 + i)
    _add_image_file(conn, tmp_path, "urgent", mtime=1000.0)  # oldest: crawled last
    conn.close()

    worker = AIWorker(cfg, cfg.db_path, poll_interval=999.0, batch_size=10)

    order = []
    original_stage = worker._process_embedding_space

    def tracking_stage(conn_, backend, space, limit, only_file_id=None):
        if only_file_id is not None:
            order.append(("priority", only_file_id, space))
        return original_stage(conn_, backend, space, limit, only_file_id=only_file_id)

    worker._process_embedding_space = tracking_stage

    fired = []
    original_load = worker._process_hashes

    def hashes_then_request(conn_, limit, only_file_id=None):
        consumed = original_load(conn_, limit, only_file_id=only_file_id)
        # Simulate the panel POSTing /index while the cycle is running.
        if not fired and only_file_id is None:
            fired.append(True)
            worker.request_priority_index("urgent")
        return consumed

    worker._process_hashes = hashes_then_request
    worker._run_cycle()

    assert ("priority", "urgent", SPACE_SEMANTIC) in order, (
        "the mid-cycle priority request was not served during the cycle"
    )


# --- status page data + CUDA-swap hold -----------------------------------------


def test_note_error_feeds_recent_errors_once_per_key(tmp_path):
    """recent_errors records each distinct error key once (with timestamp
    and message); repeats bump the counter only."""
    cfg = _cfg(tmp_path)
    _make_db(cfg.db_path).close()
    worker = AIWorker(cfg, cfg.db_path, poll_interval=999.0, batch_size=0)

    worker._note_error("hash:f1", "hash: could not read /x/f1")
    worker._note_error("hash:f1", "hash: could not read /x/f1")
    worker._note_error("embed:semantic:f2", "embed failed for /x/f2")

    assert worker.stats["errors"] == 3
    messages = [entry["message"] for entry in worker.recent_errors]
    assert messages == ["hash: could not read /x/f1", "embed failed for /x/f2"]
    assert all(entry["at"] > 0 for entry in worker.recent_errors)


def test_status_exposes_priority_queue_depth_and_recent_errors(api):
    """/status carries the live worker signals the status tab renders."""
    fake_worker = SimpleNamespace(
        is_running=True,
        stats={"cycles": 3, "errors": 1},
        provision_state={"state": "done", "groups": []},
        _priority_ids=["a", "b"],
        recent_errors=[{"at": 123.0, "message": "boom"}],
    )
    set_worker(fake_worker)
    try:
        status = api.client.get(f"{_PREFIX}/status").get_json()
    finally:
        set_worker(None)
    assert status["worker"]["priority_queued"] == 2
    assert status["worker"]["recent_errors"] == [{"at": 123.0, "message": "boom"}]

    bare = api.client.get(f"{_PREFIX}/status").get_json()
    assert bare["worker"]["priority_queued"] == 0
    assert bare["worker"]["recent_errors"] == []


# --- fair scheduling across model stages ---------------------------------------


def test_model_stages_share_the_budget_evenly(tmp_path):
    """With both embedding spaces active and a backlog bigger than the
    budget, ONE cycle advances both spaces instead of letting the first
    stage starve the second (observed live: semantic 1,841 vs visual 3)."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    for i in range(6):
        _add_image_file(conn, tmp_path, f"even{i}", mtime=1000.0 + i)
    conn.close()

    worker = AIWorker(cfg, cfg.db_path, poll_interval=999.0, batch_size=4)
    worker._run_cycle()

    conn = sqlite3.connect(cfg.db_path)
    try:
        by_space = dict(conn.execute("SELECT space, COUNT(*) FROM ai_embeddings GROUP BY space").fetchall())
    finally:
        conn.close()
    assert by_space.get(SPACE_SEMANTIC, 0) == 2
    assert by_space.get(SPACE_VISUAL, 0) == 2


def test_backlog_reviews_ride_along_never_starved(tmp_path, monkeypatch):
    """While embeddings/faces still have backlog, reviews are paced, not
    held: exactly one review rides along per eligible cycle (measured
    backoff may stretch the interval; it never reaches zero). Reviews keep
    running after the fast stages go idle."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    for i in range(3):
        _add_image_file(conn, tmp_path, f"rev{i}", mtime=1000.0 + i)
    conn.close()

    worker = AIWorker(cfg, cfg.db_path, poll_interval=999.0, batch_size=2)
    review_calls = []
    monkeypatch.setattr(
        worker,
        "_backend",
        lambda key, resolver, _orig=worker._backend: object() if key == "critic" else _orig(key, resolver),
    )

    def _fake_process_reviews(_self, _conn, _backend, limit, only_file_id=None):
        del only_file_id  # accepted only for _process_reviews' call-signature compatibility
        review_calls.append(limit)
        return 0

    monkeypatch.setattr(AIWorker, "_process_reviews", _fake_process_reviews)

    worker._run_cycle()  # embedding backlog present -> one ride-along review
    assert review_calls == [1]

    while True:  # drain the fast-stage backlog
        before = dict(worker.stats)
        worker._run_cycle()
        if worker.stats["embedded"] == before["embedded"]:
            break
    assert len(review_calls) > 1, "reviews stopped running during/after the crawl"
    assert all(n >= 1 for n in review_calls)


def test_failed_provisioning_retries_after_cooldown(tmp_path, monkeypatch):
    """A failed provisioning run re-attempts after the cooldown (bounded
    at three retries) instead of staying dead until restart."""

    cfg = _cfg(tmp_path, semantic_backend="auto", auto_provision=True)
    _make_db(cfg.db_path).close()
    worker = AIWorker(cfg, cfg.db_path, poll_interval=999.0, batch_size=0)
    worker.provision_state = {"state": "failed: connection reset", "groups": ["semantic"]}
    worker._provision_started_at = time.monotonic() - 601.0

    monkeypatch.setattr(W, "provision_groups_for", lambda _config: ["semantic"])
    monkeypatch.setattr(W.provisioning, "provision", lambda *_a, **_k: {"downloaded": [], "skipped": []})

    worker._maybe_retry_provision()
    worker._provision_thread.join(timeout=10)
    assert worker.provision_state["state"] == "done"
    assert worker._provision_attempts == 1

    # A healthy state is never retried.
    worker._maybe_retry_provision()
    assert worker._provision_attempts == 1


def test_cycle_log_names_why_stages_are_waiting(tmp_path, caplog):
    """When a configured stage produced nothing, the cycle log says why --
    e.g. a failed provisioning run -- instead of a bare '+0 embedded'."""
    cfg = _cfg(tmp_path, semantic_backend="auto", visual_backend="auto")
    conn = _make_db(cfg.db_path)
    _add_image_file(conn, tmp_path, "why_file")
    conn.close()

    worker = AIWorker(cfg, cfg.db_path, poll_interval=999.0, batch_size=5)
    worker.provision_state = {"state": "failed: connection reset", "groups": ["semantic", "visual"]}
    worker._provision_started_at = time.monotonic() - 120.0

    with caplog.at_level(logging.INFO, logger="smartgallery_ai.worker"):
        worker._run_cycle()
    lines = [r.getMessage() for r in caplog.records if "indexed:" in r.getMessage()]
    assert lines
    assert "provisioning failed" in lines[0]
    assert "semantic" in lines[0]
    assert "visual" in lines[0]


def test_app_git_ref_reads_branch_and_short_sha(tmp_path):
    """Reads branch@shortsha from a .git dir (loose ref, packed-refs, and
    detached HEAD all supported); non-checkouts return None."""

    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/feature-x\n")
    (git / "refs" / "heads" / "feature-x").write_text("abc123def456789\n")
    assert app_git_ref(str(tmp_path)) == "feature-x@abc123def"

    (git / "refs" / "heads" / "feature-x").unlink()
    (git / "packed-refs").write_text("# pack-refs\nfeedbeef12345678 refs/heads/feature-x\n")
    assert app_git_ref(str(tmp_path)) == "feature-x@feedbeef1"

    (git / "HEAD").write_text("0123456789abcdef\n")
    assert app_git_ref(str(tmp_path)) == "012345678"

    assert app_git_ref(str(tmp_path / "not-a-checkout")) is None

    assert app_git_ref() is not None  # this test runs inside the repo checkout


# --- gallery-wide surfacing endpoints ------------------------------------------


@pytest.fixture
def surf(tmp_path):
    """Fixture for the global surfaces: three visible embedded files (two
    of them exact duplicates with sizes), one policy-hidden embedded file,
    two reviews with distinct quality, and one 2-face cluster."""

    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    now = time.time()
    for fid in ("vis_a", "vis_b", "vis_c", "hidden1"):
        _add_image_file(conn, tmp_path, fid)

    stub = StubSemanticEmbedder()
    store = vectors.VectorStore(cache_dir=cfg.cache_dir, ephemeral=True)
    for fid in ("vis_a", "vis_b", "hidden1"):
        store.add(conn, fid, SPACE_SEMANTIC, stub.model_id, stub.model_version, stub.embed_text(fid), 1000.0)

    store_review(
        conn,
        "vis_a",
        ReviewResult(
            quality_score=0.2,
            prompt_alignment_score=None,
            summary="rough",
            findings=[
                Finding(type="artifact", severity="high", confidence=0.9, localizable=False, description="bad hands")
            ],
        ),
        "stub-critic",
        "stub-v1",
        RUBRIC_VERSION,
        None,
        1000.0,
        now,
    )
    store_review(
        conn,
        "vis_b",
        ReviewResult(quality_score=0.8, prompt_alignment_score=0.5, summary="clean", findings=[]),
        "stub-critic",
        "stub-v1",
        RUBRIC_VERSION,
        None,
        1000.0,
        now,
    )

    shared = "e" * 64
    for fid, size in (("vis_a", 100), ("vis_b", 40), ("hidden1", 0)):
        conn.execute("UPDATE files SET size = ? WHERE id = ?", (size, fid))
    hashing.upsert_hashes(
        conn, "vis_a", hashing.HashResult(sha256=shared, phash64=0, dhash64=0), 1000.0, "algo-v1", now
    )
    hashing.upsert_hashes(
        conn, "vis_b", hashing.HashResult(sha256=shared, phash64=-1, dhash64=0), 1000.0, "algo-v1", now
    )

    rng = np.random.default_rng(3)
    base = rng.standard_normal(16).astype(np.float32)
    base /= np.linalg.norm(base)
    for fid, _seed in (("vis_a", 1), ("vis_b", 2)):
        jitter = base + rng.standard_normal(16).astype(np.float32) * 0.01
        replace_faces_for_file(
            conn,
            fid,
            [
                FaceDetection(
                    bbox=(0.1, 0.1, 0.2, 0.2), landmarks=[], det_score=0.9, embedding=jitter.astype(np.float32)
                )
            ],
            StubFaceBackend.model_id,
            StubFaceBackend.model_version,
            1000.0,
            now,
        )
    cluster_ids = cluster_faces(
        conn, StubFaceBackend.model_id, StubFaceBackend.model_version, threshold=0.9, min_cluster_size=2
    )
    conn.close()

    app = Flask(__name__)
    app.register_blueprint(create_ai_blueprint(cfg, file_access_check=lambda fid: fid != "hidden1"), url_prefix=_PREFIX)
    return SimpleNamespace(cfg=cfg, client=app.test_client(), cluster_id=cluster_ids[0])


def test_semantic_search_filters_hidden_files(surf):
    """Free-text search embeds the query with the stub text tower and
    returns nearest embedded files, minus policy-hidden ids; a missing
    query is a 400."""
    data = surf.client.get(f"{_PREFIX}/search/semantic?q=anything").get_json()
    returned = {r["file_id"] for r in data["results"]}
    assert "vis_a" in returned
    assert "vis_b" in returned
    assert "hidden1" not in returned

    assert surf.client.get(f"{_PREFIX}/search/semantic").status_code == 400


def test_reviews_browser_sorts_by_quality_and_counts_findings(surf):
    """quality_asc puts the rough review first with its finding count;
    quality_desc flips it; an unknown sort is a 400."""
    worst = surf.client.get(f"{_PREFIX}/reviews?sort=quality_asc").get_json()
    assert worst["total"] == 2
    assert [r["file_id"] for r in worst["reviews"]] == ["vis_a", "vis_b"]
    assert worst["reviews"][0]["finding_count"] == 1

    best = surf.client.get(f"{_PREFIX}/reviews?sort=quality_desc").get_json()
    assert [r["file_id"] for r in best["reviews"]] == ["vis_b", "vis_a"]

    assert surf.client.get(f"{_PREFIX}/reviews?sort=sneaky").status_code == 400


def test_duplicates_overview_reports_reclaimable_bytes(surf):
    """The sweep lists each exact-duplicate group with the bytes saved by
    keeping the largest copy (100+40 -> keep 100, reclaim 40)."""
    data = surf.client.get(f"{_PREFIX}/duplicates").get_json()
    assert data["group_count"] == 1
    assert data["redundant_files"] == 1
    assert data["total_bytes_reclaimable"] == 40
    assert set(data["groups"][0]["file_ids"]) == {"vis_a", "vis_b"}


def test_cluster_label_roundtrip_and_unknown_404(surf):
    """POST sets a cluster's label (visible in the listing), empty clears
    it, and an unknown cluster id is a 404."""
    cid = surf.cluster_id
    res = surf.client.post(f"{_PREFIX}/faces/clusters/{cid}/label", json={"label": "  Sarah  "})
    assert res.get_json()["label"] == "Sarah"

    listing = surf.client.get(f"{_PREFIX}/faces/clusters").get_json()
    labels = {c["cluster_id"]: c["label"] for c in listing["clusters"]}
    assert labels[cid] == "Sarah"

    cleared = surf.client.post(f"{_PREFIX}/faces/clusters/{cid}/label", json={"label": ""})
    assert cleared.get_json()["label"] is None

    assert surf.client.post(f"{_PREFIX}/faces/clusters/99999/label", json={"label": "x"}).status_code == 404


def test_semantic_embedder_for_search_only_lends_thread_safe_instances(tmp_path):
    """The worker lends its semantic embedder to request threads ONLY when the
    instance declares `thread_safe`; anything that does not -- including a
    backend that simply forgot to answer the question -- is never shared."""
    cfg = _cfg(tmp_path)
    _make_db(cfg.db_path).close()
    worker = AIWorker(cfg, cfg.db_path, poll_interval=999.0, batch_size=0)

    assert worker.semantic_embedder_for_search() is None  # nothing cached

    class _Unsafe(StubSemanticEmbedder):
        thread_safe = False

    worker._backend_cache["semantic"] = _Unsafe()
    assert worker.semantic_embedder_for_search() is None

    class _Undeclared:
        """Declares nothing; the unsafe answer is the default."""

    worker._backend_cache["semantic"] = _Undeclared()
    assert worker.semantic_embedder_for_search() is None

    safe = StubSemanticEmbedder()  # thread_safe: pure function of its argument
    worker._backend_cache["semantic"] = safe
    assert worker.semantic_embedder_for_search() is safe
