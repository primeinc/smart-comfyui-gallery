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

import contextlib
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from smartgallery_ai import schema

_VECTOR_DTYPE = "<f4"  # little-endian float32, per schema.py contract

# Process-wide generation registry: one immutable _SpaceMatrix per space,
# shared by every VectorStore instance (service blueprint and worker hold
# separate stores over the same DB). Faiss's own concurrency contract
# (wiki Threads-and-asynchronous-calls) allows concurrent reads on an
# immutable index but requires external exclusion for mutation — so
# mutation here is a whole-generation swap: the single writer (the
# ingest worker) rebuilds off the request path and swaps the pointer
# atomically; searches only ever see complete generations.
_GEN_LOCK = threading.Lock()
_GENERATIONS: dict = {}  # space -> _SpaceMatrix
_WRITER_ACTIVE = threading.Event()

# GPU faiss state: StandardGpuResources is created once per process, and
# every GPU-index search runs under this lock — faiss GPU indexes are not
# thread-safe even for concurrent reads (faiss wiki Threads-and-
# asynchronous-calls), and topk serves multiple request threads.
_FAISS_GPU_SEARCH_LOCK = threading.Lock()
_faiss_gpu_res: list = []  # memoized [StandardGpuResources]


def _vector_gpu_wanted() -> bool:
    """AI_DAM_VECTOR_GPU=0 keeps topk on the CPU index even when the
    loaded faiss build has GPUs."""
    return os.environ.get("AI_DAM_VECTOR_GPU", "1") == "1"


def _faiss_gpu_resources(faiss):
    if not _faiss_gpu_res:
        _faiss_gpu_res.append(faiss.StandardGpuResources())
    return _faiss_gpu_res[0]


def set_writer_active(active: bool) -> None:
    """Declare a single-writer (the ingest worker) present in this process.

    With a writer active, searches serve the current generation without
    checking SQLite staleness — the writer refreshes generations after
    ingest batches (bounded staleness, no request-path rebuilds). Without
    one (tests, CLI, no-worker deployments), searches keep the strict
    stamp check and rebuild inline when stale.
    """
    if active:
        _WRITER_ACTIVE.set()
    else:
        _WRITER_ACTIVE.clear()


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization; all-zero rows pass through unchanged (their
    norm is substituted with 1 to avoid division by zero)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


@dataclass
class _SpaceMatrix:
    """In-memory candidate matrix for one (space, model_version) pair.

    `row_count` and `max_computed_at` mirror the SQLite stamp so staleness
    checks are a cheap comparison rather than a reload.
    """

    model_version: str
    ids: list  # file_ids aligned with matrix rows, ascending
    matrix: np.ndarray  # (n, dim), L2-normalized rows, float32
    row_count: int
    max_computed_at: float
    faiss_index: object | None = None  # lazy IndexFlatIP over `matrix`; rebuilt with it
    faiss_gpu: bool = False  # True when faiss_index lives on a GPU (search needs the lock)
    id_to_row: dict | None = None  # lazy file_id -> row index; rebuilt with matrix


class VectorStore:
    """Cosine top-k index over `ai_embeddings`, cached per (space, model_version)."""

    def __init__(
        self,
        db: str | Callable[[], sqlite3.Connection] | None = None,
        cache_dir: str = "",
        ephemeral: bool = False,
    ):
        """`db` is a sqlite path, a zero-arg connection factory, or None (each
        call must then supply its own connection). `ephemeral=True` keeps the
        cache purely in memory -- nothing is written under `cache_dir`."""
        if db is None:
            self._conn_factory: Callable[[], sqlite3.Connection] | None = None
        elif callable(db):
            self._conn_factory = db
        else:
            db_path = db
            self._conn_factory = lambda: schema.connect(db_path,
                                                         row_factory=None)
        self.cache_dir = cache_dir
        self.ephemeral = ephemeral

    # -- connection handling -------------------------------------------------

    def _resolve_conn(self, conn: sqlite3.Connection | None):
        """Returns (conn, owns_it). Falls back to the configured factory."""
        if conn is not None:
            return conn, False
        if self._conn_factory is None:
            raise ValueError("no connection provided and no db factory configured")
        return self._conn_factory(), True

    # -- writes ----------------------------------------------------------------

    def add(
        self,
        conn: sqlite3.Connection | None,
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
        # No cache pop: generations are immutable and replaced wholesale by
        # the writer's refresh() (or the strict stamp check when no writer
        # is active); popping here made every search during ingest rebuild
        # the full matrix on the request path.

    # -- reads -----------------------------------------------------------------

    def topk(
        self,
        conn: sqlite3.Connection | None,
        space: str,
        query_vec: np.ndarray,
        k: int,
        exclude: Sequence[str] = (),
        model_version: str | None = None,
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
        try:
            from smartgallery_ai.faiss_runtime import import_faiss
            faiss = import_faiss()
        except ImportError:
            faiss = None
        if faiss is not None:
            # Exact cosine via inner product on the already-normalized rows
            # (facebookresearch/faiss README: cosine similarity is a dot
            # product on normalized vectors; IndexFlatIP is exact search).
            if sm.faiss_index is None:
                index = faiss.IndexFlatIP(int(sm.matrix.shape[1]))
                index.add(sm.matrix)
                sm.faiss_gpu = False
                if _vector_gpu_wanted() and faiss.get_num_gpus() > 0:
                    # Same exact search on the GPU. Falls back to the CPU
                    # index when the copy fails (VRAM pressure, driver).
                    try:
                        index = faiss.index_cpu_to_gpu(
                            _faiss_gpu_resources(faiss), 0, index)
                        sm.faiss_gpu = True
                    except Exception:
                        pass
                sm.faiss_index = index
            if excluded and sm.id_to_row is None:
                sm.id_to_row = {fid: i for i, fid in enumerate(sm.ids)}
            rows = (np.array(
                sorted(sm.id_to_row[f] for f in excluded if f in sm.id_to_row),
                dtype=np.int64) if excluded else np.empty(0, np.int64))
            if sm.faiss_gpu:
                # GPU indexes support neither SearchParameters/IDSelector
                # nor k > 2048 (faiss wiki), and are not thread-safe even
                # for reads: over-fetch by the exclusion count under the
                # module lock, filter after.
                fetch = min(sm.row_count, k + len(rows), 2048)
                with _FAISS_GPU_SEARCH_LOCK:
                    sims_f, ids_f = sm.faiss_index.search(
                        q[None, :].astype(np.float32), fetch)
                pairs = [
                    (sm.ids[int(i)], float(s))
                    for s, i in zip(sims_f[0], ids_f[0], strict=False)
                    if int(i) >= 0 and sm.ids[int(i)] not in excluded
                ]
            else:
                params = None
                if len(rows):
                    # First-party exclusion: excluded rows are skipped inside
                    # the scan itself (faiss wiki "Setting search parameters
                    # for one query"; tests/test_search_params.py), instead
                    # of over-fetching and filtering here.
                    params = faiss.SearchParameters(
                        sel=faiss.IDSelectorNot(faiss.IDSelectorBatch(rows))
                    )
                sims_f, ids_f = sm.faiss_index.search(
                    q[None, :].astype(np.float32), min(sm.row_count, k),
                    params=params
                )
                pairs = [
                    (sm.ids[int(i)], float(s))
                    for s, i in zip(sims_f[0], ids_f[0], strict=False)
                    if int(i) >= 0
                ]
            pairs.sort(key=lambda t: (-t[1], t[0]))
            return pairs[:k]

        sims = sm.matrix @ q
        candidates = [i for i in range(len(sm.ids)) if sm.ids[i] not in excluded]
        candidates.sort(key=lambda i: (-float(sims[i]), sm.ids[i]))
        return [(sm.ids[i], float(sims[i])) for i in candidates[:k]]

    def invalidate(self, space: str) -> None:
        """Drop the current generation and on-disk cache for `space`
        (SQLite untouched)."""
        with _GEN_LOCK:
            _GENERATIONS.pop(space, None)
        vectors_dir = os.path.join(self.cache_dir, "vectors")
        if not os.path.isdir(vectors_dir):
            return
        prefix = f"{space}__"
        for name in os.listdir(vectors_dir):
            if name.startswith(prefix) and name.endswith(".npz"):
                os.remove(os.path.join(vectors_dir, name))

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _active_model_version(conn: sqlite3.Connection, space: str) -> str | None:
        """The model_version of the most recently computed row in this space."""
        row = conn.execute(
            "SELECT model_version FROM ai_embeddings WHERE space = ? "
            "ORDER BY computed_at DESC, model_version DESC LIMIT 1",
            (space,),
        ).fetchone()
        return row[0] if row else None

    @staticmethod
    def _db_stamp(conn: sqlite3.Connection, space: str, model_version: str) -> tuple[int, float]:
        """(row_count, max computed_at) for the pair -- the cheap staleness
        stamp compared against memory and disk caches."""
        row = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(computed_at), 0.0) FROM ai_embeddings "
            "WHERE space = ? AND model_version = ?",
            (space, model_version),
        ).fetchone()
        return int(row[0]), float(row[1])

    def _cache_path(self, space: str, model_version: str) -> str:
        """On-disk mirror path; path separators in the version are flattened
        so the pair always maps to a single filename."""
        safe_version = model_version.replace(os.sep, "_").replace("/", "_")
        return os.path.join(self.cache_dir, "vectors", f"{space}__{safe_version}.npz")

    def refresh(self, conn: sqlite3.Connection, space: str) -> None:
        """Single-writer generation rebuild: load the space fresh from SQLite
        and atomically swap it into the process-wide registry. Called by the
        ingest worker after embedding batches; searches never rebuild while
        a writer is active."""
        model_version = self._active_model_version(conn, space)
        if model_version is None:
            with _GEN_LOCK:
                _GENERATIONS.pop(space, None)
            return
        fresh = self._load_from_sqlite(conn, space, model_version)
        with _GEN_LOCK:
            _GENERATIONS[space] = fresh
        if fresh.row_count > 0 and not self.ephemeral:
            self._save_disk_cache(space, fresh)

    def _get_matrix(self, conn: sqlite3.Connection, space: str,
                    model_version: str | None = None) -> _SpaceMatrix | None:
        """Current generation for `space`: registry first, then disk mirror,
        then a rebuild from SQLite. `model_version` defaults to the most
        recently computed one; None means the space holds no rows.

        With an active single writer, a registry generation of the right
        model_version is served as-is (bounded staleness; the writer swaps
        in fresh generations off the request path). Without a writer the
        strict SQLite stamp check decides, rebuilding inline when stale."""
        if model_version is None:
            model_version = self._active_model_version(conn, space)
        if model_version is None:
            with _GEN_LOCK:
                _GENERATIONS.pop(space, None)
            return None

        with _GEN_LOCK:
            cached = _GENERATIONS.get(space)
        if cached is not None and cached.model_version == model_version:
            if _WRITER_ACTIVE.is_set():
                return cached
            row_count, max_computed_at = self._db_stamp(conn, space, model_version)
            if cached.row_count == row_count and cached.max_computed_at == max_computed_at:
                return cached

        row_count, max_computed_at = self._db_stamp(conn, space, model_version)
        if not self.ephemeral:
            disk = self._load_disk_cache(space, model_version)
            if disk is not None and disk.row_count == row_count and disk.max_computed_at == max_computed_at:
                with _GEN_LOCK:
                    _GENERATIONS[space] = disk
                return disk

        fresh = self._load_from_sqlite(conn, space, model_version)
        with _GEN_LOCK:
            _GENERATIONS[space] = fresh
        if fresh.row_count > 0 and not self.ephemeral:
            self._save_disk_cache(space, fresh)
        return fresh

    @staticmethod
    def _load_from_sqlite(conn: sqlite3.Connection, space: str, model_version: str) -> _SpaceMatrix:
        """Build the matrix straight from `ai_embeddings`; raises ValueError
        if stored dims disagree within the (space, model_version) pair."""
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
        """Write the space's .npz mirror atomically (temp file + rename), so a
        reader never observes a partially written cache."""
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
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)

    def _load_disk_cache(self, space: str, model_version: str) -> _SpaceMatrix | None:
        """Read the .npz mirror; None when absent or unreadable -- any failure
        means fall back to SQLite, never raise."""
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
