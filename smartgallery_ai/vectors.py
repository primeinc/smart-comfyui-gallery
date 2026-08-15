"""Local vector index: one implementation hosting multiple named,
model-versioned similarity spaces (semantic / visual / face / ...).

SQLite (`ai_embeddings`) is the authoritative derived record. The in-memory
matrix per space is a cache lazily built from SQLite and, unless
`ephemeral=True`, mirrored to `cache_dir/vectors/{space}__{model_version}.npz`
alongside a cheap staleness stamp (row count, max computed_at). The cache is
always safe to delete: the next query rebuilds it from SQLite.

Spaces never mix: a matrix built for one (space, model_version) only ever
contains rows for that exact pair, so `topk('semantic', ...)` can never see
`'visual'` rows for the same file, and stale rows left behind by a
model-version migration are naturally excluded until re-indexed.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np

_VECTOR_DTYPE = "<f4"  # little-endian float32, per schema.py contract


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


@dataclass
class _SpaceMatrix:
    model_version: str
    ids: list
    matrix: np.ndarray  # (n, dim), L2-normalized rows, float32
    row_count: int
    max_computed_at: float


class VectorStore:
    """Cosine top-k index over `ai_embeddings`, cached per (space, model_version)."""

    def __init__(
        self,
        db: Union[str, Callable[[], sqlite3.Connection], None] = None,
        cache_dir: str = "",
        ephemeral: bool = False,
    ):
        if db is None:
            self._conn_factory: Optional[Callable[[], sqlite3.Connection]] = None
        elif callable(db):
            self._conn_factory = db
        else:
            db_path = db
            self._conn_factory = lambda: sqlite3.connect(db_path)
        self.cache_dir = cache_dir
        self.ephemeral = ephemeral
        self._memory: dict[str, _SpaceMatrix] = {}

    # -- connection handling -------------------------------------------------

    def _resolve_conn(self, conn: Optional[sqlite3.Connection]):
        """Returns (conn, owns_it). Falls back to the configured factory."""
        if conn is not None:
            return conn, False
        if self._conn_factory is None:
            raise ValueError("no connection provided and no db factory configured")
        return self._conn_factory(), True

    # -- writes ----------------------------------------------------------------

    def add(
        self,
        conn: Optional[sqlite3.Connection],
        file_id: str,
        space: str,
        model_id: str,
        model_version: str,
        vec: np.ndarray,
        source_mtime: float,
    ) -> None:
        """Upsert one (file_id, space) embedding row.

        Validates that `vec`'s dimensionality matches any other row already
        stored under the same (space, model_version) -- different spaces, or
        different model_versions within a space (e.g. mid-migration), are
        allowed to carry different dims since they're never compared directly.
        """
        conn, owns_conn = self._resolve_conn(conn)
        try:
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            dim = int(arr.shape[0])
            existing = conn.execute(
                "SELECT dim FROM ai_embeddings WHERE space = ? AND model_version = ? LIMIT 1",
                (space, model_version),
            ).fetchone()
            if existing is not None and existing[0] != dim:
                raise ValueError(
                    f"dim mismatch for space={space!r} model_version={model_version!r}: "
                    f"existing dim={existing[0]}, new dim={dim}"
                )
            blob = arr.astype(_VECTOR_DTYPE).tobytes()
            conn.execute(
                """
                INSERT INTO ai_embeddings
                    (file_id, space, model_id, model_version, dim, vector, source_mtime, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_id, space) DO UPDATE SET
                    model_id=excluded.model_id,
                    model_version=excluded.model_version,
                    dim=excluded.dim,
                    vector=excluded.vector,
                    source_mtime=excluded.source_mtime,
                    computed_at=excluded.computed_at
                """,
                (file_id, space, model_id, model_version, dim, blob, source_mtime, time.time()),
            )
            conn.commit()
        finally:
            if owns_conn:
                conn.close()
        self._memory.pop(space, None)

    # -- reads -----------------------------------------------------------------

    def topk(
        self,
        conn: Optional[sqlite3.Connection],
        space: str,
        query_vec: np.ndarray,
        k: int,
        exclude: Sequence[str] = (),
        model_version: Optional[str] = None,
    ) -> list[tuple[str, float]]:
        """Top-k (file_id, cosine_similarity) neighbors of `query_vec` in `space`.

        Ties break on ascending file_id for determinism. Never mixes spaces:
        the matrix used here only ever holds rows for this exact space.

        `model_version` pins the candidate matrix to that version. Callers
        whose query vector comes from a stored row MUST pass the row's own
        model_version: during a model migration the "active" (most recent)
        version can differ from the row's, and comparing vectors across
        versions is meaningless (or a dim-mismatch error).
        """
        conn, owns_conn = self._resolve_conn(conn)
        try:
            sm = self._get_matrix(conn, space, model_version)
        finally:
            if owns_conn:
                conn.close()
        if sm is None or sm.row_count == 0:
            return []

        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        if q.shape[0] != sm.matrix.shape[1]:
            raise ValueError(
                f"query dim {q.shape[0]} does not match space {space!r} dim {sm.matrix.shape[1]}"
            )
        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        q = q / norm

        excluded = set(exclude)
        sims = sm.matrix @ q
        candidates = [i for i in range(len(sm.ids)) if sm.ids[i] not in excluded]
        candidates.sort(key=lambda i: (-float(sims[i]), sm.ids[i]))
        return [(sm.ids[i], float(sims[i])) for i in candidates[:k]]

    def invalidate(self, space: str) -> None:
        """Drop the in-memory and on-disk cache for `space` (SQLite untouched)."""
        self._memory.pop(space, None)
        vectors_dir = os.path.join(self.cache_dir, "vectors")
        if not os.path.isdir(vectors_dir):
            return
        prefix = f"{space}__"
        for name in os.listdir(vectors_dir):
            if name.startswith(prefix) and name.endswith(".npz"):
                os.remove(os.path.join(vectors_dir, name))

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _active_model_version(conn: sqlite3.Connection, space: str) -> Optional[str]:
        """The model_version of the most recently computed row in this space."""
        row = conn.execute(
            "SELECT model_version FROM ai_embeddings WHERE space = ? "
            "ORDER BY computed_at DESC, model_version DESC LIMIT 1",
            (space,),
        ).fetchone()
        return row[0] if row else None

    @staticmethod
    def _db_stamp(conn: sqlite3.Connection, space: str, model_version: str) -> tuple[int, float]:
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(computed_at), 0.0) FROM ai_embeddings "
            "WHERE space = ? AND model_version = ?",
            (space, model_version),
        ).fetchone()
        return int(row[0]), float(row[1])

    def _cache_path(self, space: str, model_version: str) -> str:
        safe_version = model_version.replace(os.sep, "_").replace("/", "_")
        return os.path.join(self.cache_dir, "vectors", f"{space}__{safe_version}.npz")

    def _get_matrix(self, conn: sqlite3.Connection, space: str,
                    model_version: Optional[str] = None) -> Optional[_SpaceMatrix]:
        if model_version is None:
            model_version = self._active_model_version(conn, space)
        if model_version is None:
            self._memory.pop(space, None)
            return None

        row_count, max_computed_at = self._db_stamp(conn, space, model_version)
        cached = self._memory.get(space)
        if (
            cached is not None
            and cached.model_version == model_version
            and cached.row_count == row_count
            and cached.max_computed_at == max_computed_at
        ):
            return cached

        if not self.ephemeral:
            disk = self._load_disk_cache(space, model_version)
            if disk is not None and disk.row_count == row_count and disk.max_computed_at == max_computed_at:
                self._memory[space] = disk
                return disk

        fresh = self._load_from_sqlite(conn, space, model_version)
        self._memory[space] = fresh
        if fresh.row_count > 0 and not self.ephemeral:
            self._save_disk_cache(space, fresh)
        return fresh

    @staticmethod
    def _load_from_sqlite(conn: sqlite3.Connection, space: str, model_version: str) -> _SpaceMatrix:
        rows = conn.execute(
            "SELECT file_id, vector, dim, computed_at FROM ai_embeddings "
            "WHERE space = ? AND model_version = ? ORDER BY file_id",
            (space, model_version),
        ).fetchall()
        if not rows:
            return _SpaceMatrix(model_version, [], np.zeros((0, 0), dtype=np.float32), 0, 0.0)

        ids = [r[0] for r in rows]
        dim = rows[0][2]
        raw = np.zeros((len(rows), dim), dtype=np.float32)
        max_computed_at = 0.0
        for i, (_, blob, row_dim, computed_at) in enumerate(rows):
            if row_dim != dim:
                raise ValueError(
                    f"inconsistent dim within space={space!r} model_version={model_version!r}: "
                    f"{dim} vs {row_dim}"
                )
            raw[i] = np.frombuffer(blob, dtype=_VECTOR_DTYPE)
            max_computed_at = max(max_computed_at, computed_at)
        matrix = _l2_normalize_rows(raw)
        return _SpaceMatrix(model_version, ids, matrix, len(rows), max_computed_at)

    def _save_disk_cache(self, space: str, sm: _SpaceMatrix) -> None:
        path = self._cache_path(space, sm.model_version)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # np.savez appends ".npz" itself only when the name lacks that suffix,
        # so give the temp file the suffix up front and rename atomically.
        # The temp name is unique per writer: concurrent request threads can
        # each hold their own VectorStore, and a shared temp path would let
        # one writer os.replace the file out from under another mid-save.
        tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}.npz"
        try:
            np.savez(
                tmp_path,
                ids=np.array(sm.ids),
                matrix=sm.matrix,
                row_count=np.array(sm.row_count),
                max_computed_at=np.array(sm.max_computed_at),
            )
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _load_disk_cache(self, space: str, model_version: str) -> Optional[_SpaceMatrix]:
        path = self._cache_path(space, model_version)
        if not os.path.exists(path):
            return None
        try:
            with np.load(path) as data:
                ids = [str(x) for x in data["ids"]]
                matrix = data["matrix"].astype(np.float32)
                row_count = int(data["row_count"])
                max_computed_at = float(data["max_computed_at"])
        except Exception:
            return None
        return _SpaceMatrix(model_version, ids, matrix, row_count, max_computed_at)
