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


def _face_stub_source(img):
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
