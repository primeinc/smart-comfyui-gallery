"""Tests for smartgallery_ai.worker.AIWorker: hashing + embedding rows appear
for real on-disk PNGs, staleness recompute on mtime change, clean stop(),
and resilience to an unreadable file (never crashes, logs once, counts an
error). Only 'stub' backends are used -- never needle2/llama/real weights.
"""

from __future__ import annotations

import sqlite3
import time

import numpy as np
import pytest
from PIL import Image

from smartgallery_ai import AIConfig, SPACE_SEMANTIC, SPACE_VISUAL
from smartgallery_ai.faces import FaceDetection
from smartgallery_ai.schema import init_schema
from smartgallery_ai.worker import AIWorker


def _make_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE files (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            mtime REAL NOT NULL,
            name TEXT NOT NULL,
            type TEXT,
            workflow_prompt TEXT DEFAULT ''
        )
        """
    )
    init_schema(conn)
    conn.commit()
    conn.close()


def _add_file(db_path: str, file_id: str, path: str, mtime: float, file_type: str = "image") -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO files (id, path, mtime, name, type) VALUES (?, ?, ?, ?, ?)",
        (file_id, path, mtime, file_id, file_type),
    )
    conn.commit()
    conn.close()


def _set_mtime(db_path: str, file_id: str, mtime: float) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE files SET mtime = ? WHERE id = ?", (mtime, file_id))
    conn.commit()
    conn.close()


def _query_one(db_path: str, sql: str, params: tuple = ()):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql, params).fetchone()
    finally:
        conn.close()


def _face_stub_source(_img):
    return [
        FaceDetection(
            bbox=(0.1, 0.1, 0.2, 0.2), landmarks=[], det_score=0.9,
            embedding=np.ones(8, dtype=np.float32),
        )
    ]


def _wait_until(predicate, timeout: float = 3.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


@pytest.fixture()
def worker_env(tmp_path):
    db_path = str(tmp_path / "gallery.sqlite")
    _make_db(db_path)

    file_ids = []
    for i, color in enumerate([(200, 10, 10), (10, 200, 10), (10, 10, 200)]):
        img_path = str(tmp_path / f"img{i}.png")
        Image.new("RGB", (16, 16), color).save(img_path)
        file_id = f"f{i}"
        _add_file(db_path, file_id, img_path, mtime=1000.0)
        file_ids.append(file_id)

    # A row pointing at a file that is never created on disk.
    missing_path = str(tmp_path / "does_not_exist.png")
    _add_file(db_path, "f_missing", missing_path, mtime=1000.0)

    config = AIConfig(
        enabled=True,
        base_path=str(tmp_path),
        db_path=db_path,
        models_dir=str(tmp_path / "models"),
        cache_dir=str(tmp_path / "cache"),
        ephemeral_index=True,
        semantic_backend="stub",
        visual_backend="stub",
        face_backend="stub",
        critic_backend="none",
        extra={"face_stub_source": _face_stub_source},
    )
    worker = AIWorker(config, db_path, poll_interval=0.05, batch_size=50)
    yield db_path, config, worker, file_ids
    worker.stop(timeout=2.0)


def test_worker_hashes_and_embeds_real_files(worker_env):
    db_path, config, worker, file_ids = worker_env
    worker.start()
    try:
        assert _wait_until(lambda: worker.stats["hashed"] >= 3, timeout=3.0)
        assert _wait_until(lambda: worker.stats["embedded"] >= 6, timeout=3.0)
    finally:
        worker.stop(timeout=2.0)

    hash_count = _query_one(db_path, "SELECT COUNT(*) FROM ai_file_hashes")[0]
    assert hash_count == 3
    for fid in file_ids:
        row = _query_one(db_path, "SELECT sha256 FROM ai_file_hashes WHERE file_id = ?", (fid,))
        assert row is not None

    sem_count = _query_one(
        db_path, "SELECT COUNT(*) FROM ai_embeddings WHERE space = ?", (SPACE_SEMANTIC,)
    )[0]
    vis_count = _query_one(
        db_path, "SELECT COUNT(*) FROM ai_embeddings WHERE space = ?", (SPACE_VISUAL,)
    )[0]
    assert sem_count == 3
    assert vis_count == 3

    face_count = _query_one(db_path, "SELECT COUNT(*) FROM ai_face_instances")[0]
    assert face_count == 3  # one stub-sourced face per real image

    # f_missing must never produce a hash/embedding/face row.
    assert _query_one(db_path, "SELECT 1 FROM ai_file_hashes WHERE file_id = 'f_missing'") is None
    assert worker.stats["errors"] >= 1


def test_worker_recomputes_on_mtime_staleness(worker_env):
    db_path, config, worker, file_ids = worker_env
    worker.start()
    try:
        assert _wait_until(lambda: worker.stats["hashed"] >= 3, timeout=3.0)
        original_row = _query_one(
            db_path, "SELECT source_mtime FROM ai_file_hashes WHERE file_id = ?", ("f0",)
        )
        assert original_row[0] == pytest.approx(1000.0)

        _set_mtime(db_path, "f0", 5000.0)

        assert _wait_until(
            lambda: (_query_one(
                db_path, "SELECT source_mtime FROM ai_file_hashes WHERE file_id = ?", ("f0",)
            ) or [None])[0] == pytest.approx(5000.0),
            timeout=3.0,
        )
    finally:
        worker.stop(timeout=2.0)

    updated_row = _query_one(
        db_path, "SELECT source_mtime FROM ai_file_hashes WHERE file_id = ?", ("f0",)
    )
    assert updated_row[0] == pytest.approx(5000.0)


def test_worker_stop_joins_cleanly(worker_env):
    db_path, config, worker, file_ids = worker_env
    worker.start()
    assert _wait_until(lambda: worker.stats["cycles"] >= 1, timeout=3.0)
    assert worker.is_running

    worker.stop(timeout=2.0)
    assert not worker.is_running


def test_worker_never_raises_on_unreadable_path(worker_env):
    db_path, config, worker, file_ids = worker_env
    worker.start()
    try:
        assert _wait_until(lambda: worker.stats["cycles"] >= 2, timeout=3.0)
    finally:
        worker.stop(timeout=2.0)
    # The thread must still be a clean, joinable stop -- i.e. it never died
    # from an uncaught exception while processing f_missing.
    assert not worker.is_running
    assert worker.stats["errors"] >= 1


def test_worker_start_is_idempotent_and_default_state(tmp_path):
    db_path = str(tmp_path / "gallery.sqlite")
    _make_db(db_path)
    config = AIConfig(enabled=True, base_path=str(tmp_path), db_path=db_path,
                       cache_dir=str(tmp_path / "cache"))
    worker = AIWorker(config, db_path, poll_interval=0.05)

    assert not worker.is_running
    assert worker.stats == {
        "cycles": 0, "hashed": 0, "embedded": 0, "faces_indexed": 0, "reviewed": 0, "errors": 0,
    }

    worker.start()
    worker.start()  # no-op, must not spawn a second thread
    assert worker.is_running
    worker.stop(timeout=2.0)
    assert not worker.is_running


def test_worker_zero_result_scan_not_repeated(tmp_path):
    """A file with zero faces gets ONE scan per (model, mtime), recorded in
    ai_scan_log with result_count 0 — not re-detected every cycle."""
    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    img_path = str(tmp_path / "nofaces.png")
    Image.new("RGB", (16, 16), (5, 5, 5)).save(img_path)
    _add_file(db_path, "nf1", img_path, mtime=1000.0)

    calls = {"n": 0}

    def counting_empty_source(_img):
        calls["n"] += 1
        return []

    config = AIConfig(
        enabled=True, base_path=str(tmp_path), db_path=db_path,
        models_dir=str(tmp_path / "models"), cache_dir=str(tmp_path / "cache"),
        ephemeral_index=True, semantic_backend="none", visual_backend="none",
        face_backend="stub", critic_backend="none",
        extra={"face_stub_source": counting_empty_source},
    )
    worker = AIWorker(config, db_path, poll_interval=0.03, batch_size=50)
    worker.start()
    try:
        assert _wait_until(lambda: worker.stats["cycles"] >= 4, timeout=5.0)
    finally:
        worker.stop(timeout=2.0)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM ai_scan_log WHERE file_id = 'nf1' AND kind = 'faces'"
    ).fetchone()
    conn.close()
    assert row is not None
    assert row["result_count"] == 0
    assert calls["n"] == 1, f"zero-face file re-scanned {calls['n']} times"


def test_worker_rescans_after_mtime_change(tmp_path):
    """The scan log keys on source mtime: touching the file re-scans it."""
    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    img_path = str(tmp_path / "face.png")
    Image.new("RGB", (16, 16), (50, 50, 50)).save(img_path)
    _add_file(db_path, "fx", img_path, mtime=1000.0)

    calls = {"n": 0}

    def counting_source(img):
        calls["n"] += 1
        return _face_stub_source(img)

    config = AIConfig(
        enabled=True, base_path=str(tmp_path), db_path=db_path,
        models_dir=str(tmp_path / "models"), cache_dir=str(tmp_path / "cache"),
        ephemeral_index=True, semantic_backend="none", visual_backend="none",
        face_backend="stub", critic_backend="none",
        extra={"face_stub_source": counting_source},
    )
    worker = AIWorker(config, db_path, poll_interval=0.03, batch_size=50)
    worker.start()
    try:
        assert _wait_until(lambda: calls["n"] >= 1, timeout=5.0)
        first = calls["n"]
        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE files SET mtime = 2000.0 WHERE id = 'fx'")
        conn.commit()
        conn.close()
        assert _wait_until(lambda: calls["n"] > first, timeout=5.0)
    finally:
        worker.stop(timeout=2.0)


def test_worker_masks_generated_when_segmenter_arrives_late(tmp_path):
    """Oracle-confirmed fix: a review stored while NO segmenter was
    available must still get masks once a segmenter is provisioned — the
    'masks' scan-log unit is independent of the review row."""
    from smartgallery_ai import RUBRIC_VERSION, review as R

    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    img_path = str(tmp_path / "img.png")
    Image.new("RGB", (64, 64), (90, 90, 90)).save(img_path)
    _add_file(db_path, "mf1", img_path, mtime=1000.0)

    # Store a review with one localizable finding directly (as if the
    # critic ran while segmenter_backend was 'none').
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    result = R.validate_review_payload({
        "quality_score": 5.0, "prompt_alignment_score": None, "summary": "s",
        "findings": [{"type": "artifact", "severity": "low", "confidence": 0.9,
                      "localizable": True, "description": "spot",
                      "bbox": [0.25, 0.25, 0.5, 0.5]}]})
    R.store_review(conn, "mf1", result, "critic-x", "v1", RUBRIC_VERSION,
                   "{}", 1000.0, 1.0)
    # Mark the review scan as done (so only the mask stage has work).
    conn.execute(
        "INSERT INTO ai_scan_log VALUES ('mf1', 'review', 'critic-x', 'v1',"
        " 1000.0, 1.0, 1)")
    conn.commit()
    assert conn.execute("SELECT mask_path FROM ai_review_findings").fetchone()[0] is None
    conn.close()

    config = AIConfig(
        enabled=True, base_path=str(tmp_path), db_path=db_path,
        models_dir=str(tmp_path / "models"), cache_dir=str(tmp_path / "cache"),
        ephemeral_index=True, semantic_backend="none", visual_backend="none",
        face_backend="none", critic_backend="none", segmenter_backend="stub",
    )
    worker = AIWorker(config, db_path, poll_interval=0.03, batch_size=50)
    worker.start()
    try:
        def has_mask():
            c = sqlite3.connect(db_path)
            mp = c.execute("SELECT mask_path FROM ai_review_findings").fetchone()[0]
            c.close()
            return mp is not None
        assert _wait_until(has_mask, timeout=5.0), "late segmenter never masked"
    finally:
        worker.stop(timeout=2.0)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    log = conn.execute(
        "SELECT * FROM ai_scan_log WHERE file_id='mf1' AND kind='masks'").fetchone()
    mask_path = conn.execute("SELECT mask_path FROM ai_review_findings").fetchone()[0]
    conn.close()
    import os as _os
    assert log is not None and log["result_count"] == 1
    assert _os.path.isfile(mask_path)


def test_scan_log_check_migration_admits_masks(tmp_path):
    """Old databases carry a CHECK without kind 'masks'; init_schema
    rebuilds the table in place, preserving rows."""
    from smartgallery_ai import schema as S

    db_path = str(tmp_path / "old.sqlite")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE files (id TEXT PRIMARY KEY, path TEXT, mtime REAL)")
    conn.execute("INSERT INTO files VALUES ('f1', '/f1', 0)")
    # Simulate the pre-'masks' table shape.
    conn.execute("""
        CREATE TABLE ai_scan_log (
            file_id TEXT NOT NULL REFERENCES files(id)
                ON DELETE CASCADE ON UPDATE CASCADE,
            kind TEXT NOT NULL CHECK (kind IN ('faces', 'review')),
            model_id TEXT NOT NULL, model_version TEXT NOT NULL,
            source_mtime REAL NOT NULL, scanned_at REAL NOT NULL,
            result_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (file_id, kind))""")
    conn.execute("INSERT INTO ai_scan_log VALUES ('f1','faces','m','v',0,0,2)")
    conn.commit()

    S.init_schema(conn)
    # Old row preserved, new kind admitted.
    rows = conn.execute("SELECT kind, result_count FROM ai_scan_log").fetchall()
    assert ("faces", 2) in rows
    conn.execute("INSERT INTO ai_scan_log VALUES ('f1','masks','m','v',0,0,1)")
    conn.commit()


def test_worker_sweeps_orphaned_mask_dirs(tmp_path):
    """Deleting a files row cascades findings rows away but not mask PNGs;
    the worker's sweep removes mask directories for vanished file ids."""
    import os as _os

    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    img_path = str(tmp_path / "img.png")
    Image.new("RGB", (16, 16), (10, 10, 10)).save(img_path)
    _add_file(db_path, "keep1", img_path, mtime=1000.0)

    cache = tmp_path / "cache"
    (cache / "masks" / "keep1").mkdir(parents=True)
    (cache / "masks" / "keep1" / "1.png").write_bytes(b"x")
    (cache / "masks" / "ghost").mkdir(parents=True)
    (cache / "masks" / "ghost" / "9.png").write_bytes(b"x")

    config = AIConfig(
        enabled=True, base_path=str(tmp_path), db_path=db_path,
        models_dir=str(tmp_path / "models"), cache_dir=str(cache),
        ephemeral_index=True, semantic_backend="none", visual_backend="none",
        face_backend="none", critic_backend="none",
    )
    worker = AIWorker(config, db_path, poll_interval=0.03, batch_size=50)
    worker.start()
    try:
        assert _wait_until(
            lambda: not _os.path.isdir(str(cache / "masks" / "ghost")), timeout=5.0)
    finally:
        worker.stop(timeout=2.0)
    assert _os.path.isfile(str(cache / "masks" / "keep1" / "1.png"))


def test_backend_resolver_exception_cached_none_no_deadlock(tmp_path):
    """A raising resolver must record the error, cache None, and return —
    without re-acquiring the worker lock from inside the locked section."""
    import threading

    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    config = AIConfig(
        enabled=True, base_path=str(tmp_path), db_path=db_path,
        models_dir=str(tmp_path / "models"), cache_dir=str(tmp_path / "cache"),
        ephemeral_index=True, semantic_backend="none", visual_backend="none",
        face_backend="none", critic_backend="none",
    )
    worker = AIWorker(config, db_path, poll_interval=0.05, batch_size=10)

    calls = []

    def bad_resolver(_cfg):
        calls.append(1)
        raise RuntimeError("resolver exploded")

    out = {}

    def run():
        out["first"] = worker._backend("boom", bad_resolver)
        out["second"] = worker._backend("boom", bad_resolver)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), "worker._backend deadlocked on a raising resolver"
    assert out["first"] is None and out["second"] is None
    assert len(calls) == 1  # the failure is cached, not re-probed


class _ExplodingSegmenter:
    model_id = "boom-segmenter"
    model_version = "boom-v1"

    def segment(self, _img, bbox=None, points=None):
        del bbox, points  # accepted only for segment()'s call-signature compatibility (kwarg call)
        raise RuntimeError("segmentation failed")


def test_failed_mask_generation_is_retried_not_logged_complete(tmp_path):
    """A cycle where every mask attempt fails must not record a 'masks'
    scan row; the file stays selectable and succeeds once the segmenter
    recovers."""
    from smartgallery_ai import RUBRIC_VERSION, review as R
    from smartgallery_ai.review import StubSegmenter

    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    img_path = str(tmp_path / "img.png")
    Image.new("RGB", (64, 64), (90, 90, 90)).save(img_path)
    _add_file(db_path, "mf1", img_path, mtime=1000.0)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    result = R.validate_review_payload({
        "quality_score": 5.0, "prompt_alignment_score": None, "summary": "s",
        "findings": [{"type": "artifact", "severity": "low", "confidence": 0.9,
                      "localizable": True, "description": "spot",
                      "bbox": [0.25, 0.25, 0.5, 0.5]}]})
    R.store_review(conn, "mf1", result, "critic-x", "v1", RUBRIC_VERSION,
                   "{}", 1000.0, 1.0)
    conn.execute(
        "INSERT INTO ai_scan_log VALUES ('mf1', 'review', 'critic-x', 'v1',"
        " 1000.0, 1.0, 1)")
    conn.commit()

    config = AIConfig(
        enabled=True, base_path=str(tmp_path), db_path=db_path,
        models_dir=str(tmp_path / "models"), cache_dir=str(tmp_path / "cache"),
        ephemeral_index=True, semantic_backend="none", visual_backend="none",
        face_backend="none", critic_backend="none", segmenter_backend="stub",
    )
    worker = AIWorker(config, db_path, poll_interval=0.05, batch_size=50)

    processed = worker._process_masks(conn, _ExplodingSegmenter(), 10)
    assert processed == 1  # the candidate was attempted...
    assert conn.execute(
        "SELECT COUNT(*) FROM ai_scan_log WHERE file_id='mf1' AND kind='masks'"
    ).fetchone()[0] == 0  # ...but not recorded as complete
    assert conn.execute(
        "SELECT mask_path FROM ai_review_findings").fetchone()[0] is None

    worker._process_masks(conn, StubSegmenter(), 10)
    mask_path = conn.execute("SELECT mask_path FROM ai_review_findings").fetchone()[0]
    log = conn.execute(
        "SELECT result_count FROM ai_scan_log WHERE file_id='mf1' AND kind='masks'"
    ).fetchone()
    conn.close()
    import os as _os
    assert mask_path is not None and _os.path.isfile(mask_path)
    assert log is not None and log[0] == 1


def test_worker_clusters_faces_after_indexing(worker_env):
    """Indexing new face instances triggers clustering in the same cycle;
    the identical stub embeddings across files form one cluster."""
    db_path, config, worker, file_ids = worker_env
    worker.start()
    assert _wait_until(
        lambda: (_query_one(db_path, "SELECT COUNT(*) FROM ai_face_clusters") or (0,))[0] > 0,
        timeout=5.0,
    ), "worker never clustered the indexed faces"
    assigned = _query_one(
        db_path,
        "SELECT COUNT(*) FROM ai_face_instances WHERE cluster_id IS NOT NULL")[0]
    assert assigned >= 2


def test_unavailable_backend_reprobed_after_retry_window(tmp_path):
    """An unavailable backend must not be cached for the worker's lifetime:
    provisioning weights later has to activate it once the retry window
    elapses — the advertised late-provisioning path for masks depends on
    the standalone stage actually receiving a segmenter."""
    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    config = AIConfig(
        enabled=True, base_path=str(tmp_path), db_path=db_path,
        models_dir=str(tmp_path / "models"), cache_dir=str(tmp_path / "cache"),
        ephemeral_index=True, semantic_backend="none", visual_backend="none",
        face_backend="none", critic_backend="none",
    )
    worker = AIWorker(config, db_path, poll_interval=0.05, batch_size=10)

    provisioned = object()
    attempts = []

    def resolver(_cfg):
        attempts.append(1)
        return None if len(attempts) == 1 else provisioned

    # Within the retry window the None result is served from cache.
    assert worker._backend("seg", resolver) is None
    assert worker._backend("seg", resolver) is None
    assert len(attempts) == 1

    # After the window the backend is re-probed and the instance sticks.
    worker._backend_retry_seconds = 0.0
    assert worker._backend("seg", resolver) is provisioned
    assert len(attempts) == 2
    worker._backend_retry_seconds = 300.0
    assert worker._backend("seg", resolver) is provisioned
    assert len(attempts) == 2  # success is cached for the lifetime


def test_record_scan_is_model_scoped(tmp_path):
    """Each pipeline keeps its own scan-log row: a second model's scan adds
    a row instead of overwriting the first model's last-run bookkeeping;
    re-scanning with the same model upserts in place."""
    from smartgallery_ai.worker import record_scan

    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    _add_file(db_path, "f1", str(tmp_path / "f1.png"), mtime=1000.0)

    class _M1:
        model_id, model_version = "m1", "v1"

    class _M2:
        model_id, model_version = "m2", "v1"

    conn = sqlite3.connect(db_path)
    record_scan(conn, "f1", "faces", _M1, 1000.0, 2000.0, 3)
    record_scan(conn, "f1", "faces", _M2, 1000.0, 2001.0, 1)
    rows = conn.execute(
        "SELECT model_id, result_count, scanned_at FROM ai_scan_log "
        "WHERE file_id = 'f1' ORDER BY model_id").fetchall()
    assert [tuple(r) for r in rows] == [("m1", 3, 2000.0), ("m2", 1, 2001.0)]

    record_scan(conn, "f1", "faces", _M1, 1000.0, 2002.0, 2)
    rows = conn.execute(
        "SELECT model_id, result_count, scanned_at FROM ai_scan_log "
        "WHERE file_id = 'f1' ORDER BY model_id").fetchall()
    assert [tuple(r) for r in rows] == [("m1", 2, 2002.0), ("m2", 1, 2001.0)]
    conn.close()


def test_scan_log_pk_migration_preserves_rows(tmp_path):
    """A database whose ai_scan_log predates the model-scoped primary key
    is rebuilt in place by init_schema with every row preserved, after
    which two models' rows for one (file, kind) coexist."""
    from smartgallery_ai.schema import init_schema
    from smartgallery_ai.worker import record_scan

    db_path = str(tmp_path / "g.sqlite")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE files (id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, "
        "mtime REAL NOT NULL, name TEXT NOT NULL, type TEXT, "
        "workflow_prompt TEXT DEFAULT '')")
    conn.execute(
        """
        CREATE TABLE ai_scan_log (
            file_id TEXT NOT NULL REFERENCES files(id)
                ON DELETE CASCADE ON UPDATE CASCADE,
            kind TEXT NOT NULL CHECK (kind IN ('faces', 'review', 'masks')),
            model_id TEXT NOT NULL,
            model_version TEXT NOT NULL,
            source_mtime REAL NOT NULL,
            scanned_at REAL NOT NULL,
            result_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (file_id, kind)
        )
        """)
    conn.execute(
        "INSERT INTO files (id, path, mtime, name, type) "
        "VALUES ('f1', 'p1', 1000.0, 'f1', 'image')")
    conn.execute(
        "INSERT INTO ai_scan_log VALUES ('f1', 'faces', 'm1', 'v1', 1000.0, 2000.0, 3)")
    init_schema(conn)

    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='ai_scan_log'"
    ).fetchone()
    assert "kind, model_id, model_version" in row[0]
    assert conn.execute("SELECT COUNT(*) FROM ai_scan_log").fetchone()[0] == 1

    class _M2:
        model_id, model_version = "m2", "v1"

    record_scan(conn, "f1", "faces", _M2, 1000.0, 2001.0, 1)
    assert conn.execute(
        "SELECT COUNT(*) FROM ai_scan_log WHERE file_id = 'f1' AND kind = 'faces'"
    ).fetchone()[0] == 2
    conn.close()


def test_faces_attribute_backfill_requeues_only_null_attribute_rows(tmp_path):
    """Face rows written before attribute persistence (attributes NULL under
    the current model) are re-queued exactly once — by deleting their scan
    log rows, never by a version bump — while files whose rows already
    carry attributes are left alone."""
    from smartgallery_ai.faces import StubFaceBackend

    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    for i in (1, 2):
        img_path = str(tmp_path / f"img{i}.png")
        Image.new("RGB", (16, 16), (10 * i, 10, 10)).save(img_path)
        _add_file(db_path, f"bf{i}", img_path, mtime=1000.0)

    def source_with_attrs(_img):
        return [FaceDetection(
            bbox=(0.1, 0.1, 0.2, 0.2), landmarks=[], det_score=0.9,
            embedding=np.ones(8, dtype=np.float32),
            attributes={"age": 30, "landmark_2d_106": [[0.1, 0.2]]})]

    config = AIConfig(
        enabled=True, base_path=str(tmp_path), db_path=db_path,
        models_dir=str(tmp_path / "models"), cache_dir=str(tmp_path / "cache"),
        ephemeral_index=True, semantic_backend="none", visual_backend="none",
        face_backend="stub", critic_backend="none",
        extra={"face_stub_source": source_with_attrs},
    )
    worker = AIWorker(config, db_path, poll_interval=0.05, batch_size=10)
    backend = StubFaceBackend(source=source_with_attrs)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    assert worker._process_faces(conn, backend, 10) == 2

    # Simulate rows written before attribute persistence on ONE file, and
    # clear the marker so the backfill re-examines the table.
    conn.execute(
        "UPDATE ai_face_instances SET attributes = NULL, age = NULL "
        "WHERE file_id = 'bf1'")
    conn.execute(
        "DELETE FROM ai_dam_state WHERE key LIKE 'faces_attr_backfill:%'")
    conn.commit()

    # Only bf1 re-enters the queue; re-processing restores its attributes.
    assert worker._process_faces(conn, backend, 10) == 1
    row = conn.execute(
        "SELECT attributes, age FROM ai_face_instances WHERE file_id = 'bf1'"
    ).fetchone()
    assert row["attributes"] is not None
    assert row["age"] == 30

    # Marker set: a further cycle re-queues nothing.
    assert worker._process_faces(conn, backend, 10) == 0
    conn.close()


def test_face_clustering_retried_after_failure(tmp_path, monkeypatch):
    """Face scans commit before clustering runs, so a clustering failure
    must leave persistent pending state: the next cycle (with zero new
    face candidates) has to retry and succeed."""
    from smartgallery_ai import faces as F
    from smartgallery_ai.faces import StubFaceBackend

    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    # Two files, identical stub embeddings: enough to form one cluster
    # (min_cluster_size = 2).
    for i in (1, 2):
        img_path = str(tmp_path / f"img{i}.png")
        Image.new("RGB", (16, 16), (10 * i, 10, 10)).save(img_path)
        _add_file(db_path, f"cf{i}", img_path, mtime=1000.0)

    config = AIConfig(
        enabled=True, base_path=str(tmp_path), db_path=db_path,
        models_dir=str(tmp_path / "models"), cache_dir=str(tmp_path / "cache"),
        ephemeral_index=True, semantic_backend="none", visual_backend="none",
        face_backend="stub", critic_backend="none",
        extra={"face_stub_source": _face_stub_source},
    )
    worker = AIWorker(config, db_path, poll_interval=0.05, batch_size=10)
    backend = StubFaceBackend(source=_face_stub_source)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    real_cluster = F.cluster_faces
    calls = []

    def failing_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("clustering exploded")
        return real_cluster(*args, **kwargs)

    monkeypatch.setattr(F, "cluster_faces", failing_once)

    # Cycle 1: faces indexed, clustering fails — scan rows are committed,
    # so without pending state this would never retry.
    assert worker._process_faces(conn, backend, 10) == 2
    assert conn.execute("SELECT COUNT(*) FROM ai_face_clusters").fetchone()[0] == 0

    # Cycle 2: no face candidates remain, yet clustering retries and lands.
    assert worker._process_faces(conn, backend, 10) == 0
    assert len(calls) == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM ai_face_instances WHERE cluster_id IS NOT NULL"
    ).fetchone()[0] >= 1

    # Cycle 3: nothing pending — clustering is not re-run.
    assert worker._process_faces(conn, backend, 10) == 0
    assert len(calls) == 2
    conn.close()
