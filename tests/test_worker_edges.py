"""Edge-behavior tests for smartgallery_ai.worker: _fetch_candidates ordering/
chunking/type-filter contract, per-cycle budget exhaustion between stages,
run-loop resilience to a failing cycle, stop() mid-cycle, scan-log staleness
selection, review-failure logging (result_count -1, no retry), the
missing-workflow_prompt-column degradation, per-stage failure paths, the mask
stage's no-work skip, video frame loading, and the ai_dam_state round-trip.
Only stub/fake backends -- never real weights.
"""

from __future__ import annotations

import os
import sqlite3
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from smartgallery_ai import AIConfig, RUBRIC_VERSION, SPACE_SEMANTIC, SPACE_VISUAL
from smartgallery_ai.review import CriticBackend, StubSegmenter
from smartgallery_ai.schema import init_schema
from smartgallery_ai.worker import (
    AIWorker,
    _fetch_candidates,
    _has_column,
    load_source_image,
)


# -- helpers (mirroring tests/test_worker.py conventions) ---------------------

def _make_db(db_path: str, with_prompt_column: bool = True) -> None:
    conn = sqlite3.connect(db_path)
    prompt_col = ",\n            workflow_prompt TEXT DEFAULT ''" if with_prompt_column else ""
    conn.execute(
        f"""
        CREATE TABLE files (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            mtime REAL NOT NULL,
            name TEXT NOT NULL,
            type TEXT{prompt_col}
        )
        """
    )
    init_schema(conn)
    conn.commit()
    conn.close()


def _add_file(db_path: str, file_id: str, path: str, mtime: float,
              file_type: str = "image") -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO files (id, path, mtime, name, type) VALUES (?, ?, ?, ?, ?)",
        (file_id, path, mtime, file_id, file_type),
    )
    conn.commit()
    conn.close()


def _open(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _query_one(db_path: str, sql: str, params: tuple = ()):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql, params).fetchone()
    finally:
        conn.close()


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _config(tmp_path, db_path, **overrides) -> AIConfig:
    kwargs = dict(
        enabled=True, base_path=str(tmp_path), db_path=db_path,
        models_dir=str(tmp_path / "models"), cache_dir=str(tmp_path / "cache"),
        ephemeral_index=True, semantic_backend="none", visual_backend="none",
        face_backend="none", critic_backend="none", segmenter_backend="none",
    )
    kwargs.update(overrides)
    return AIConfig(**kwargs)


def _save_png(tmp_path, name: str, color=(120, 120, 120), size=(16, 16)) -> str:
    path = str(tmp_path / name)
    Image.new("RGB", size, color).save(path)
    return path


class _RaisingCritic(CriticBackend):
    model_id = "raising-critic"
    model_version = "boom-v1"

    def __init__(self):
        self.calls = 0

    def review(self, _img, _prompt_text, _rubric_version, negative_text=None):
        self.calls += 1
        raise RuntimeError("VLM exploded")


class _RecordingCritic(CriticBackend):
    """Returns a fixed valid payload and records the prompt it was given."""

    model_id = "recording-critic"
    model_version = "rec-v1"

    def __init__(self):
        self.prompts = []

    def review(self, _img, prompt_text, _rubric_version, negative_text=None):
        self.prompts.append(prompt_text)
        return {"quality_score": 7.0, "prompt_alignment_score": None,
                "summary": "recorded", "findings": []}


# -- _fetch_candidates --------------------------------------------------------

def test_fetch_candidates_orders_and_caps_across_chunks():
    """With >CHUNK (500) ids, results are still globally mtime-DESC/id-ASC
    ordered and capped at limit, not per-chunk artifacts."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE files (id TEXT PRIMARY KEY, path TEXT, "
                 "mtime REAL, type TEXT)")
    ids = [f"f{i:04d}" for i in range(1100)]
    mtimes = {fid: float((i * 7) % 41) for i, fid in enumerate(ids)}
    conn.executemany(
        "INSERT INTO files VALUES (?, ?, ?, 'image')",
        [(fid, f"/x/{fid}", mtimes[fid]) for fid in ids])

    # Scrambled id order: chunk membership must not affect the result.
    rows = _fetch_candidates(conn, list(reversed(ids)), limit=9)

    expected = sorted(((fid, mtimes[fid]) for fid in ids),
                      key=lambda t: (-t[1], t[0]))[:9]
    assert [(r["id"], r["mtime"]) for r in rows] == expected
    conn.close()


def test_fetch_candidates_allowed_types_filters_nonvisual():
    """allowed_types restricts results to visual types; audio/document rows
    never come back even when their ids were requested."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE files (id TEXT PRIMARY KEY, path TEXT, "
                 "mtime REAL, type TEXT)")
    data = [("a", 5.0, "image"), ("b", 4.0, "audio"), ("c", 3.0, "video"),
            ("d", 2.0, "document"), ("e", 1.0, "animated_image")]
    conn.executemany("INSERT INTO files VALUES (?, ?, ?, ?)",
                     [(i, f"/x/{i}", m, t) for i, m, t in data])

    from smartgallery_ai.worker import _VISUAL_TYPES
    rows = _fetch_candidates(conn, ["a", "b", "c", "d", "e"], limit=10,
                             allowed_types=_VISUAL_TYPES)
    assert [r["id"] for r in rows] == ["a", "c", "e"]

    # Without a type filter every requested row comes back.
    rows_all = _fetch_candidates(conn, ["a", "b", "c", "d", "e"], limit=10)
    assert [r["id"] for r in rows_all] == ["a", "b", "c", "d", "e"]
    conn.close()


def test_fetch_candidates_empty_ids_or_nonpositive_limit():
    """No ids or a non-positive limit yields [] without touching the DB."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE files (id TEXT PRIMARY KEY, path TEXT, "
                 "mtime REAL, type TEXT)")
    conn.execute("INSERT INTO files VALUES ('a', '/a', 1.0, 'image')")
    assert _fetch_candidates(conn, [], limit=5) == []
    assert _fetch_candidates(conn, ["a"], limit=0) == []
    assert _fetch_candidates(conn, ["a"], limit=-3) == []
    conn.close()


# -- per-cycle budget ---------------------------------------------------------

def test_hashing_has_its_own_budget_and_model_stages_split_theirs(tmp_path):
    """Hashing never starves the model stages (its per-file cost is
    milliseconds; on a large gallery a shared budget would delay the first
    embedding by hours), and the model stages split their budget EVENLY --
    both embedding spaces advance every cycle instead of the first one
    hogging the whole allocation."""
    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    for i in range(2):
        _add_file(db_path, f"f{i}", _save_png(tmp_path, f"img{i}.png",
                                              (40 * i + 30, 10, 10)), 1000.0)
    config = _config(tmp_path, db_path, semantic_backend="stub",
                     visual_backend="stub")
    worker = AIWorker(config, db_path, batch_size=2)

    worker._run_cycle()  # cycle 1: hashes (own budget) + one slot per space
    assert worker.stats["hashed"] == 2
    assert worker.stats["embedded"] == 2
    assert _query_one(db_path, "SELECT COUNT(*) FROM ai_embeddings "
                      "WHERE space = ?", (SPACE_SEMANTIC,))[0] == 1
    assert _query_one(db_path, "SELECT COUNT(*) FROM ai_embeddings "
                      "WHERE space = ?", (SPACE_VISUAL,))[0] == 1

    worker._run_cycle()  # cycle 2: the remaining file in both spaces
    assert worker.stats["embedded"] == 4
    assert _query_one(db_path, "SELECT COUNT(*) FROM ai_embeddings "
                      "WHERE space = ?", (SPACE_VISUAL,))[0] == 2


def test_all_stages_return_zero_on_nonpositive_limit(tmp_path):
    """Every stage is a no-op returning 0 when its budget share is <= 0."""
    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    _add_file(db_path, "f0", _save_png(tmp_path, "img.png"), 1000.0)
    worker = AIWorker(_config(tmp_path, db_path), db_path)
    conn = _open(db_path)
    dummy = SimpleNamespace(model_id="m", model_version="v")
    try:
        assert worker._process_hashes(conn, 0) == 0
        assert worker._process_embedding_space(conn, dummy, SPACE_SEMANTIC, 0) == 0
        assert worker._process_faces(conn, dummy, 0) == 0
        assert worker._process_reviews(conn, dummy, -1) == 0
        assert worker._process_masks(conn, dummy, 0) == 0
        # Nothing was written by any of the no-ops.
        assert conn.execute("SELECT COUNT(*) FROM ai_file_hashes").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM ai_scan_log").fetchone()[0] == 0
    finally:
        conn.close()


# -- loop / lifecycle resilience ---------------------------------------------

def test_run_loop_survives_failing_cycles(tmp_path):
    """A cycle that raises (unopenable DB) is counted as an error and the
    worker thread keeps running instead of dying."""
    bad_db = str(tmp_path / "no_such_dir" / "g.sqlite")  # connect() raises
    config = _config(tmp_path, bad_db)
    worker = AIWorker(config, bad_db, poll_interval=0.02)
    worker.start()
    try:
        assert _wait_until(lambda: worker.stats["errors"] >= 2, timeout=5.0), \
            "cycle errors were not counted"
        assert worker.is_running, "worker thread died on a failing cycle"
        assert worker.stats["cycles"] == 0  # no cycle ever completed
    finally:
        worker.stop(timeout=2.0)
    assert not worker.is_running


def test_stop_during_active_cycle_joins_cleanly(tmp_path):
    """stop() called while a cycle is mid-stage still joins the thread."""
    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    for i in range(4):
        _add_file(db_path, f"s{i}", _save_png(tmp_path, f"s{i}.png",
                                              (i * 20, 30, 30)), 1000.0)

    detect_started = []

    def slow_source(_img):
        detect_started.append(1)
        time.sleep(0.15)
        return []

    config = _config(tmp_path, db_path, face_backend="stub",
                     extra={"face_stub_source": slow_source})
    worker = AIWorker(config, db_path, poll_interval=30.0, batch_size=50)
    worker.start()
    assert _wait_until(lambda: len(detect_started) >= 1, timeout=5.0), \
        "cycle never reached the slow face stage"
    worker.stop(timeout=10.0)  # issued mid-cycle
    assert not worker.is_running


# -- scan-log staleness selection --------------------------------------------

def test_scan_candidates_staleness_selection(tmp_path):
    """A file is (re-)selected when unlogged, when its mtime changes, or
    when the model version changes -- and skipped when its log matches."""
    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    _add_file(db_path, "fx", "/nowhere/fx.png", 1000.0)
    worker = AIWorker(_config(tmp_path, db_path), db_path)
    conn = _open(db_path)
    backend_v1 = SimpleNamespace(model_id="m", model_version="v1")
    backend_v2 = SimpleNamespace(model_id="m", model_version="v2")
    try:
        # Unlogged -> selected.
        assert [r["id"] for r in worker._scan_candidates(conn, "faces", backend_v1, 10)] == ["fx"]

        # Logged at current (model, mtime) -> skipped.
        AIWorker._log_scan(conn, "fx", "faces", backend_v1, 1000.0, 1.0, 0)
        assert worker._scan_candidates(conn, "faces", backend_v1, 10) == []

        # mtime change -> selected again.
        conn.execute("UPDATE files SET mtime = 2000.0 WHERE id = 'fx'")
        conn.commit()
        assert [r["id"] for r in worker._scan_candidates(conn, "faces", backend_v1, 10)] == ["fx"]

        # Re-logged at the new mtime -> skipped; model bump -> selected.
        AIWorker._log_scan(conn, "fx", "faces", backend_v1, 2000.0, 2.0, 0)
        assert worker._scan_candidates(conn, "faces", backend_v1, 10) == []
        assert [r["id"] for r in worker._scan_candidates(conn, "faces", backend_v2, 10)] == ["fx"]
    finally:
        conn.close()


# -- review stage -------------------------------------------------------------

def test_review_failure_logged_minus_one_and_not_retried(tmp_path):
    """A critic that raises writes a 'review' scan-log row with
    result_count -1 and the file is NOT re-attempted next cycle."""
    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    _add_file(db_path, "rv1", _save_png(tmp_path, "rv1.png"), 1000.0)
    worker = AIWorker(_config(tmp_path, db_path), db_path)
    critic = _RaisingCritic()
    conn = _open(db_path)
    try:
        assert worker._process_reviews(conn, critic, 10) == 1  # attempted once

        log = conn.execute(
            "SELECT model_id, model_version, result_count FROM ai_scan_log "
            "WHERE file_id = 'rv1' AND kind = 'review'").fetchone()
        assert log is not None
        assert (log["model_id"], log["model_version"]) == ("raising-critic", "boom-v1")
        assert log["result_count"] == -1
        assert conn.execute("SELECT COUNT(*) FROM ai_reviews").fetchone()[0] == 0
        assert worker.stats["reviewed"] == 0
        assert worker.stats["errors"] >= 1

        # Next cycle: no candidates, critic not re-invoked.
        assert worker._process_reviews(conn, critic, 10) == 0
        assert critic.calls == 1
    finally:
        conn.close()


def test_review_failed_file_requeued_on_mtime_change(tmp_path):
    """The -1 failure log keys on mtime: touching the file re-enters it
    into the review queue (normal staleness, not infinite retry)."""
    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    _add_file(db_path, "rv1", _save_png(tmp_path, "rv1.png"), 1000.0)
    worker = AIWorker(_config(tmp_path, db_path), db_path)
    critic = _RaisingCritic()
    conn = _open(db_path)
    try:
        assert worker._process_reviews(conn, critic, 10) == 1
        assert worker._process_reviews(conn, critic, 10) == 0
        conn.execute("UPDATE files SET mtime = 2000.0 WHERE id = 'rv1'")
        conn.commit()
        assert worker._process_reviews(conn, critic, 10) == 1
        assert critic.calls == 2
    finally:
        conn.close()


def test_review_success_stores_scores_findings_and_masks(tmp_path):
    """A successful stub review persists the review row (with prompt
    alignment from workflow_prompt), its localizable finding, a mask PNG,
    and 'review' + 'masks' scan-log rows."""
    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    img = Image.new("RGB", (64, 64), (200, 200, 200))
    for x in range(16, 48):
        for y in range(16, 48):
            img.putpixel((x, y), (255, 0, 0))  # StubCritic's red artifact
    img_path = str(tmp_path / "red.png")
    img.save(img_path)
    _add_file(db_path, "ok1", img_path, 1000.0)
    conn = _open(db_path)
    conn.execute("UPDATE files SET workflow_prompt = 'a castle' WHERE id = 'ok1'")
    conn.commit()

    from smartgallery_ai.review import StubCritic
    worker = AIWorker(_config(tmp_path, db_path, critic_backend="stub",
                              segmenter_backend="stub"), db_path)
    try:
        assert worker._process_reviews(conn, StubCritic(), 10) == 1

        review_row = conn.execute(
            "SELECT quality_score, prompt_alignment_score, rubric_version "
            "FROM ai_reviews WHERE file_id = 'ok1'").fetchone()
        assert review_row is not None
        assert review_row["quality_score"] == pytest.approx(8.0)  # one finding
        assert review_row["prompt_alignment_score"] == pytest.approx(1.0)
        assert review_row["rubric_version"] == RUBRIC_VERSION

        finding = conn.execute(
            "SELECT localizable, mask_path FROM ai_review_findings "
            "WHERE file_id = 'ok1'").fetchone()
        assert finding["localizable"] == 1
        assert finding["mask_path"] is not None
        assert os.path.isfile(finding["mask_path"])

        review_log = conn.execute(
            "SELECT result_count FROM ai_scan_log WHERE file_id='ok1' "
            "AND kind='review'").fetchone()
        masks_log = conn.execute(
            "SELECT result_count FROM ai_scan_log WHERE file_id='ok1' "
            "AND kind='masks'").fetchone()
        assert review_log["result_count"] == 1
        assert masks_log["result_count"] == 1
        assert worker.stats["reviewed"] == 1
    finally:
        conn.close()


def test_review_without_workflow_prompt_column_uses_null_prompt(tmp_path):
    """A files table lacking workflow_prompt still reviews: the critic gets
    a NULL prompt and the review row lands with NULL alignment."""
    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path, with_prompt_column=False)
    _add_file(db_path, "np1", _save_png(tmp_path, "np1.png"), 1000.0)
    worker = AIWorker(_config(tmp_path, db_path), db_path)
    critic = _RecordingCritic()
    conn = _open(db_path)
    try:
        assert _has_column(conn, "files", "workflow_prompt") is False

        assert worker._process_reviews(conn, critic, 10) == 1
        assert critic.prompts == [None]

        row = conn.execute(
            "SELECT prompt_alignment_score, quality_score FROM ai_reviews "
            "WHERE file_id = 'np1'").fetchone()
        assert row is not None
        assert row["prompt_alignment_score"] is None
        assert row["quality_score"] == pytest.approx(7.0)
        log = conn.execute(
            "SELECT result_count FROM ai_scan_log WHERE file_id='np1' "
            "AND kind='review'").fetchone()
        assert log["result_count"] == 0
        assert worker.stats["reviewed"] == 1
    finally:
        conn.close()


# -- embedding / face failure paths ------------------------------------------

class _RaisingEmbedder:
    model_id = "raising-embedder"
    model_version = "v1"

    def embed_image(self, _img):
        raise RuntimeError("embed exploded")


def test_embedding_backend_failure_counts_error_writes_nothing(tmp_path):
    """A backend.embed_image failure is counted, writes no embedding row,
    and still consumes the candidate's slice of the cycle budget."""
    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    _add_file(db_path, "e1", _save_png(tmp_path, "e1.png"), 1000.0)
    worker = AIWorker(_config(tmp_path, db_path), db_path)
    conn = _open(db_path)
    try:
        used = worker._process_embedding_space(conn, _RaisingEmbedder(),
                                               SPACE_SEMANTIC, 10)
        assert used == 1  # counted against the budget despite failing
        assert conn.execute("SELECT COUNT(*) FROM ai_embeddings").fetchone()[0] == 0
        assert worker.stats["embedded"] == 0
        assert worker.stats["errors"] == 1
    finally:
        conn.close()


class _RaisingFaceBackend:
    model_id = "raising-face"
    model_version = "v1"

    def __init__(self):
        self.calls = 0

    def detect(self, _img):
        self.calls += 1
        raise RuntimeError("detect exploded")


def test_face_detect_failure_leaves_file_retryable(tmp_path):
    """Unlike reviews, a face-detect failure writes NO scan-log row, so
    the file is retried the next cycle (and no face rows appear)."""
    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    _add_file(db_path, "ff1", _save_png(tmp_path, "ff1.png"), 1000.0)
    worker = AIWorker(_config(tmp_path, db_path), db_path)
    backend = _RaisingFaceBackend()
    conn = _open(db_path)
    try:
        assert worker._process_faces(conn, backend, 10) == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM ai_scan_log WHERE kind='faces'"
        ).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM ai_face_instances").fetchone()[0] == 0
        assert worker.stats["faces_indexed"] == 0
        assert worker.stats["errors"] >= 1

        # Still a candidate: the failure did not freeze it.
        assert worker._process_faces(conn, backend, 10) == 1
        assert backend.calls == 2
    finally:
        conn.close()


# -- mask stage ---------------------------------------------------------------

def test_mask_stage_skips_when_no_findings_lack_masks(tmp_path):
    """With only global (non-localizable) findings, the standalone mask
    stage selects nothing and writes no 'masks' scan-log row."""
    from smartgallery_ai import review as R

    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    _add_file(db_path, "g1", _save_png(tmp_path, "g1.png"), 1000.0)
    conn = _open(db_path)
    result = R.validate_review_payload({
        "quality_score": 6.0, "prompt_alignment_score": None, "summary": "s",
        "findings": [{"type": "lighting", "severity": "low", "confidence": 0.7,
                      "localizable": False, "description": "flat light"}]})
    R.store_review(conn, "g1", result, "critic-x", "v1", RUBRIC_VERSION,
                   "{}", 1000.0, 1.0)

    worker = AIWorker(_config(tmp_path, db_path), db_path)
    try:
        assert worker._process_masks(conn, StubSegmenter(), 10) == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM ai_scan_log WHERE kind='masks'"
        ).fetchone()[0] == 0
        assert worker.stats["errors"] == 0
    finally:
        conn.close()


def test_mask_stage_unreadable_file_counts_error_stays_retryable(tmp_path):
    """A mask candidate whose source file is unreadable is counted as an
    error and left unlogged, so it stays selectable for retry."""
    from smartgallery_ai import review as R

    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    _add_file(db_path, "mm1", str(tmp_path / "vanished.png"), 1000.0)
    conn = _open(db_path)
    result = R.validate_review_payload({
        "quality_score": 5.0, "prompt_alignment_score": None, "summary": "s",
        "findings": [{"type": "artifact", "severity": "low", "confidence": 0.9,
                      "localizable": True, "description": "spot",
                      "bbox": [0.25, 0.25, 0.5, 0.5]}]})
    R.store_review(conn, "mm1", result, "critic-x", "v1", RUBRIC_VERSION,
                   "{}", 1000.0, 1.0)

    worker = AIWorker(_config(tmp_path, db_path), db_path)
    try:
        assert worker._process_masks(conn, StubSegmenter(), 10) == 1
        assert worker.stats["errors"] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM ai_scan_log WHERE kind='masks'"
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT mask_path FROM ai_review_findings").fetchone()[0] is None
        # Still a candidate next cycle.
        assert worker._process_masks(conn, StubSegmenter(), 10) == 1
    finally:
        conn.close()


def test_cycle_runs_review_stage_with_stub_critic(tmp_path):
    """The review stage is wired into _run_cycle: one full cycle with a
    stub critic produces the ai_reviews row itself."""
    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    _add_file(db_path, "cy1", _save_png(tmp_path, "cy1.png"), 1000.0)
    worker = AIWorker(_config(tmp_path, db_path, critic_backend="stub"),
                      db_path, batch_size=50)

    worker._run_cycle()

    assert worker.stats["reviewed"] == 1
    row = _query_one(db_path, "SELECT model_id, quality_score FROM ai_reviews "
                     "WHERE file_id = 'cy1'")
    assert row is not None
    assert row[0] == "stub-critic"
    assert 0.0 <= row[1] <= 10.0
    log = _query_one(db_path, "SELECT result_count FROM ai_scan_log "
                     "WHERE file_id='cy1' AND kind='review'")
    assert log is not None and log[0] >= 0


# -- orphaned-mask sweep ------------------------------------------------------

def test_sweep_unremovable_entry_counts_error_keeps_rest_working(tmp_path):
    """An unremovable masks-cache entry (a plain file) is counted as an
    error and left in place while real orphan dirs are still swept."""
    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    _add_file(db_path, "keep1", _save_png(tmp_path, "k.png"), 1000.0)

    cache = tmp_path / "cache"
    (cache / "masks" / "keep1").mkdir(parents=True)
    (cache / "masks" / "ghost").mkdir(parents=True)
    (cache / "masks" / "stray").write_bytes(b"not a directory")  # rmtree fails

    worker = AIWorker(_config(tmp_path, db_path), db_path)
    conn = _open(db_path)
    try:
        worker._sweep_orphaned_masks(conn)
    finally:
        conn.close()

    assert not (cache / "masks" / "ghost").exists()  # orphan removed
    assert (cache / "masks" / "keep1").is_dir()      # live id kept
    assert (cache / "masks" / "stray").is_file()     # failure left in place
    assert worker.stats["errors"] == 1


# -- video / source loading ---------------------------------------------------

def test_load_source_image_video_first_frame(tmp_path):
    """load_source_image on a real video returns specifically the FIRST
    frame (red; later frames are blue) as an RGB PIL image of the video's
    dimensions."""
    video_path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"),
                             5, (32, 24))
    assert writer.isOpened(), "environment cannot write MJPG avi"
    red = np.zeros((24, 32, 3), dtype=np.uint8)
    red[:, :, 2] = 255   # BGR: pure red -- frame 0 only
    blue = np.zeros((24, 32, 3), dtype=np.uint8)
    blue[:, :, 0] = 255  # BGR: pure blue -- frames 1..2
    writer.write(red)
    writer.write(blue)
    writer.write(blue)
    writer.release()

    img = load_source_image(video_path, "video")
    assert img is not None
    assert img.size == (32, 24)
    r, g, b = img.convert("RGB").getpixel((16, 12))
    # Red proves BOTH the BGR->RGB conversion AND that it was frame 0:
    # any later frame would read blue.
    assert r > 200 and g < 60 and b < 60


def test_load_source_image_returns_none_for_unreadable_and_nonvisual(tmp_path):
    """Garbage video bytes, missing images, and non-visual types all load
    as None -- never an exception."""
    garbage = str(tmp_path / "junk.mp4")
    with open(garbage, "wb") as fh:
        fh.write(b"this is not a video")
    assert load_source_image(garbage, "video") is None

    assert load_source_image(str(tmp_path / "missing.png"), "image") is None

    real_png = _save_png(tmp_path, "real.png")
    assert load_source_image(real_png, "audio") is None
    assert load_source_image(real_png, "document") is None


# -- ai_dam_state -------------------------------------------------------------

def test_state_helpers_round_trip(tmp_path):
    """_set_state upserts, _get_state reads back, _clear_state deletes;
    reads of missing keys are None and clearing twice is safe."""
    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    conn = _open(db_path)
    try:
        assert AIWorker._get_state(conn, "k") is None
        AIWorker._set_state(conn, "k", "one")
        assert AIWorker._get_state(conn, "k") == "one"
        AIWorker._set_state(conn, "k", "two")  # upsert, not duplicate
        assert AIWorker._get_state(conn, "k") == "two"
        assert conn.execute(
            "SELECT COUNT(*) FROM ai_dam_state WHERE key='k'").fetchone()[0] == 1
        AIWorker._clear_state(conn, "k")
        assert AIWorker._get_state(conn, "k") is None
        AIWorker._clear_state(conn, "k")  # idempotent
    finally:
        conn.close()
