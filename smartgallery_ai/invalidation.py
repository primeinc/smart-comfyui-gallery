"""Deterministic staleness rules for derived AI DAM state.

A derived row is stale iff its recorded source_mtime differs (beyond float
jitter) from the file's current mtime, or its algo/model version differs
from the active version for that derived kind. No heuristics: version
comparison is an exact string match, mtime comparison allows a small
epsilon for floating-point round-tripping through SQLite REAL columns.
"""

from __future__ import annotations

import time
from typing import Optional

_MTIME_EPSILON = 1e-6  # seconds; covers float64 round-tripping through REAL columns


def is_stale(
    row_source_mtime: float,
    file_mtime: float,
    row_version: str,
    active_version: str,
) -> bool:
    """True when a derived row must be recomputed: version strings differ
    exactly, or the recorded mtime drifts from the file's beyond epsilon."""
    if row_version != active_version:
        return True
    return abs(row_source_mtime - file_mtime) > _MTIME_EPSILON


def find_stale_hashes(conn, algo_version: str) -> list[str]:
    """file_ids whose `ai_file_hashes` row is stale vs `algo_version`."""
    rows = conn.execute(
        """
        SELECT h.file_id, h.source_mtime, h.algo_version, f.mtime
        FROM ai_file_hashes h
        JOIN files f ON f.id = h.file_id
        ORDER BY h.file_id
        """
    ).fetchall()
    return [
        file_id
        for file_id, source_mtime, row_version, file_mtime in rows
        if is_stale(source_mtime, file_mtime, row_version, algo_version)
    ]


def find_stale_embeddings(
    conn, space: str, model_id: str, model_version: str
) -> list[str]:
    """file_ids whose `ai_embeddings` row in `space` is stale vs the active model."""
    rows = conn.execute(
        """
        SELECT e.file_id, e.source_mtime, e.model_id, e.model_version, f.mtime
        FROM ai_embeddings e
        JOIN files f ON f.id = e.file_id
        WHERE e.space = ?
        ORDER BY e.file_id
        """,
        (space,),
    ).fetchall()
    active_key = f"{model_id}::{model_version}"
    stale = []
    for file_id, source_mtime, row_model_id, row_model_version, file_mtime in rows:
        row_key = f"{row_model_id}::{row_model_version}"
        if is_stale(source_mtime, file_mtime, row_key, active_key):
            stale.append(file_id)
    return stale


def find_missing(conn, table: str, space: Optional[str] = None) -> list[str]:
    """file_ids with NO derived row at all in `table` (never computed, vs stale).

    `table` must be 'ai_file_hashes' or 'ai_embeddings'; the latter requires
    `space` since its primary key is (file_id, space).
    """
    if table == "ai_file_hashes":
        query = """
            SELECT f.id FROM files f
            WHERE NOT EXISTS (
                SELECT 1 FROM ai_file_hashes h WHERE h.file_id = f.id
            )
            ORDER BY f.id
        """
        params: tuple = ()
    elif table == "ai_embeddings":
        if space is None:
            raise ValueError("space is required when table='ai_embeddings'")
        query = """
            SELECT f.id FROM files f
            WHERE NOT EXISTS (
                SELECT 1 FROM ai_embeddings e
                WHERE e.file_id = f.id AND e.space = ?
            )
            ORDER BY f.id
        """
        params = (space,)
    else:
        raise ValueError(f"unsupported table for find_missing: {table!r}")
    return [row[0] for row in conn.execute(query, params).fetchall()]


def set_active_version(conn, key: str, value: str) -> None:
    """Persist an active version/threshold in `ai_dam_state` (upsert)."""
    conn.execute(
        """
        INSERT INTO ai_dam_state (key, value, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        (key, value, time.time()),
    )
    conn.commit()


def active_versions(conn) -> dict[str, str]:
    """All key/value pairs currently stored in `ai_dam_state`."""
    rows = conn.execute("SELECT key, value FROM ai_dam_state ORDER BY key").fetchall()
    return {key: value for key, value in rows}
