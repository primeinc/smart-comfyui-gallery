"""Tests for smartgallery_ai.invalidation: staleness rules, missing-row
scans, and the ai_dam_state key/value store."""

import sqlite3
import time

import pytest

from smartgallery_ai.invalidation import (
    active_versions,
    find_missing,
    find_stale_embeddings,
    find_stale_hashes,
    is_stale,
    set_active_version,
)
from smartgallery_ai.schema import init_schema


def make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE files (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            mtime REAL NOT NULL,
            name TEXT NOT NULL,
            type TEXT
        )
        """
    )
    init_schema(conn)
    return conn


def add_file(conn, file_id, mtime=1000.0):
    conn.execute(
        "INSERT INTO files (id, path, mtime, name, type) VALUES (?, ?, ?, ?, ?)",
        (file_id, f"/gallery/{file_id}.png", mtime, file_id, "image"),
    )
    conn.commit()


def add_hash_row(conn, file_id, source_mtime, algo_version):
    conn.execute(
        """
        INSERT INTO ai_file_hashes (file_id, sha256, phash64, dhash64, algo_version, source_mtime, computed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (file_id, "sha", None, None, algo_version, source_mtime, time.time()),
    )
    conn.commit()


def add_embedding_row(conn, file_id, space, model_id, model_version, source_mtime):
    conn.execute(
        """
        INSERT INTO ai_embeddings (file_id, space, model_id, model_version, dim, vector, source_mtime, computed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (file_id, space, model_id, model_version, 4, b"\x00" * 16, source_mtime, time.time()),
    )
    conn.commit()


# --- is_stale ------------------------------------------------------------


def test_is_stale_false_when_mtime_and_version_match():
    assert is_stale(1000.0, 1000.0, "v1", "v1") is False


def test_is_stale_true_on_mtime_bump():
    assert is_stale(1000.0, 1000.5, "v1", "v1") is True


def test_is_stale_true_on_version_bump():
    assert is_stale(1000.0, 1000.0, "v1", "v2") is True


def test_is_stale_tolerates_float_epsilon_jitter():
    assert is_stale(1000.0, 1000.0 + 1e-9, "v1", "v1") is False


def test_is_stale_version_check_is_exact_string_match():
    assert is_stale(1000.0, 1000.0, "v1", "v1.0") is True


# --- find_stale_hashes -----------------------------------------------------


def test_find_stale_hashes_detects_mtime_bump():
    conn = make_conn()
    add_file(conn, "f1", mtime=1000.0)
    add_hash_row(conn, "f1", source_mtime=1000.0, algo_version="algo-v1")
    assert find_stale_hashes(conn, "algo-v1") == []

    conn.execute("UPDATE files SET mtime = ? WHERE id = ?", (2000.0, "f1"))
    conn.commit()
    assert find_stale_hashes(conn, "algo-v1") == ["f1"]


def test_find_stale_hashes_detects_version_bump():
    conn = make_conn()
    add_file(conn, "f1", mtime=1000.0)
    add_hash_row(conn, "f1", source_mtime=1000.0, algo_version="algo-v1")
    assert find_stale_hashes(conn, "algo-v2") == ["f1"]


def test_find_stale_hashes_fresh_row_excluded():
    conn = make_conn()
    add_file(conn, "f1", mtime=1000.0)
    add_file(conn, "f2", mtime=2000.0)
    add_hash_row(conn, "f1", source_mtime=1000.0, algo_version="algo-v1")
    add_hash_row(conn, "f2", source_mtime=1999.0, algo_version="algo-v1")  # stale mtime
    assert find_stale_hashes(conn, "algo-v1") == ["f2"]


def test_find_stale_hashes_deterministic_order():
    conn = make_conn()
    for fid in ("z", "a", "m"):
        add_file(conn, fid, mtime=1.0)
        add_hash_row(conn, fid, source_mtime=1.0, algo_version="old")
    assert find_stale_hashes(conn, "new") == ["a", "m", "z"]


# --- find_stale_embeddings --------------------------------------------------


def test_find_stale_embeddings_detects_mtime_bump():
    conn = make_conn()
    add_file(conn, "f1", mtime=1000.0)
    add_embedding_row(conn, "f1", "semantic", "modelA", "v1", source_mtime=1000.0)
    assert find_stale_embeddings(conn, "semantic", "modelA", "v1") == []

    conn.execute("UPDATE files SET mtime = ? WHERE id = ?", (2000.0, "f1"))
    conn.commit()
    assert find_stale_embeddings(conn, "semantic", "modelA", "v1") == ["f1"]


def test_find_stale_embeddings_detects_model_version_bump():
    conn = make_conn()
    add_file(conn, "f1", mtime=1000.0)
    add_embedding_row(conn, "f1", "semantic", "modelA", "v1", source_mtime=1000.0)
    assert find_stale_embeddings(conn, "semantic", "modelA", "v2") == ["f1"]


def test_find_stale_embeddings_detects_model_id_bump():
    conn = make_conn()
    add_file(conn, "f1", mtime=1000.0)
    add_embedding_row(conn, "f1", "semantic", "modelA", "v1", source_mtime=1000.0)
    assert find_stale_embeddings(conn, "semantic", "modelB", "v1") == ["f1"]


def test_find_stale_embeddings_scoped_to_space():
    conn = make_conn()
    add_file(conn, "f1", mtime=1000.0)
    add_embedding_row(conn, "f1", "semantic", "modelA", "v1", source_mtime=1000.0)
    add_embedding_row(conn, "f1", "visual", "modelB", "v9", source_mtime=1000.0)
    # 'visual' row is version-mismatched against the semantic active version,
    # but find_stale_embeddings(space='semantic', ...) must not see it.
    assert find_stale_embeddings(conn, "semantic", "modelA", "v1") == []


# --- find_missing ------------------------------------------------------------


def test_find_missing_hashes():
    conn = make_conn()
    add_file(conn, "f1")
    add_file(conn, "f2")
    add_hash_row(conn, "f1", source_mtime=1000.0, algo_version="v1")
    assert find_missing(conn, "ai_file_hashes") == ["f2"]


def test_find_missing_embeddings_requires_space():
    conn = make_conn()
    add_file(conn, "f1")
    with pytest.raises(ValueError, match="space is required"):
        find_missing(conn, "ai_embeddings")


def test_find_missing_embeddings_scoped_to_space():
    conn = make_conn()
    add_file(conn, "f1")
    add_file(conn, "f2")
    add_embedding_row(conn, "f1", "semantic", "modelA", "v1", source_mtime=1000.0)
    assert find_missing(conn, "ai_embeddings", space="semantic") == ["f2"]
    # f1 has no 'visual' row at all, so it's missing there too.
    assert find_missing(conn, "ai_embeddings", space="visual") == ["f1", "f2"]


def test_find_missing_unsupported_table_raises():
    conn = make_conn()
    with pytest.raises(ValueError, match="unsupported table for find_missing"):
        find_missing(conn, "not_a_real_table")


def test_find_missing_deterministic_order():
    conn = make_conn()
    for fid in ("z", "a", "m"):
        add_file(conn, fid)
    assert find_missing(conn, "ai_file_hashes") == ["a", "m", "z"]


# --- active_versions / set_active_version -----------------------------------


def test_set_and_get_active_version():
    conn = make_conn()
    set_active_version(conn, "semantic_model_version", "clip-v1")
    assert active_versions(conn) == {"semantic_model_version": "clip-v1"}


def test_set_active_version_upserts():
    conn = make_conn()
    set_active_version(conn, "hash_algo_version", "v1")
    set_active_version(conn, "hash_algo_version", "v2")
    assert active_versions(conn) == {"hash_algo_version": "v2"}


def test_active_versions_multiple_keys_sorted():
    conn = make_conn()
    set_active_version(conn, "visual_model_version", "dinov2-v1")
    set_active_version(conn, "semantic_model_version", "clip-v1")
    assert list(active_versions(conn).keys()) == ["semantic_model_version", "visual_model_version"]


def test_end_to_end_active_version_drives_staleness():
    conn = make_conn()
    add_file(conn, "f1", mtime=1000.0)
    add_hash_row(conn, "f1", source_mtime=1000.0, algo_version="algo-v1")
    set_active_version(conn, "hash_algo_version", "algo-v1")

    versions = active_versions(conn)
    assert find_stale_hashes(conn, versions["hash_algo_version"]) == []

    set_active_version(conn, "hash_algo_version", "algo-v2")
    versions = active_versions(conn)
    assert find_stale_hashes(conn, versions["hash_algo_version"]) == ["f1"]
