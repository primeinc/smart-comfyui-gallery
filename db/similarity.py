"""Similarity as this application means it, on top of the FAISS layer.

vision/faiss_index.py owns index lifecycle and execution; this module
owns what the results MEAN: which spaces exist, how a pair graph is cut
from a range search, how id-keyed edges become the positional CSR the
grouping algorithms eat, and how a space is kept aligned with the rows
SQLite holds. No FAISS index is constructed anywhere but the manager.

The numpy oracle at the bottom is exactly that -- an oracle. It computes
every pairwise cosine with a blocked matrix product, exists so tests can
hold the FAISS paths to an independent exact answer, and is reachable
from tests and diagnostics only. It is not an engine and nothing in
production falls back to it.
"""

from __future__ import annotations

import threading

from vision.faiss_index import IndexManager, SpaceSpec


def _imagehash_version() -> str:
    """The producer version of every stored phash64: a hash algorithm
    change across library versions silently un-matches every stored
    hash, so the version is part of the space's identity and a snapshot
    from another version is refused."""
    import importlib.metadata

    try:
        return importlib.metadata.version("ImageHash")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


#: The perceptual-hash space: 64 hamming bits per picture, produced by
#: imagehash.phash (vision/dupes.py perceptual) and consumed by dupe
#: grouping.
PHASH = SpaceSpec(
    key="perceptual.phash64",
    representation="binary",
    dimensions=64,
    metric="hamming",
    producer="imagehash.phash",
    producer_version=_imagehash_version(),
)


def face_space(model_id: str, model_version: str, dimensions: int) -> SpaceSpec:
    """One space per recognition model+version: embeddings from different
    models never share an index, because their cosines are not comparable."""
    return SpaceSpec(
        key=f"face.{model_id}.{model_version}",
        representation="float32",
        dimensions=int(dimensions),
        metric="cosine",
        producer=model_id,
        producer_version=model_version,
    )


#: One manager per snapshot directory per process -- residency IS the
#: point, and two managers over one directory would fight for the files.
_SHARED: dict[str, IndexManager] = {}
_LOCK = threading.Lock()


def manager_for(conn) -> IndexManager:
    """The process's manager for this connection's database: snapshots in
    an `indexes/` directory beside the database file, device policy from
    the `faiss_gpu` setting read once at construction.

    An in-memory database gets a fresh RAM-only manager per call: it has
    no path to share on and no life beyond its connection (which sqlite3
    refuses to weak-reference), so residency there is scoped to the call
    chain that asked -- one job aligns, searches and drops it. The real
    application database is a file, where the manager and its spaces
    live for the process.
    """
    import pathlib

    from . import settings

    row = next((r for r in conn.execute("PRAGMA database_list") if r[1] == "main"), None)
    file = row[2] if row else ""
    if not file:
        return IndexManager(None, gpu=settings.flag(conn, "faiss_gpu"))
    where = str(pathlib.Path(file).parent / "indexes")
    with _LOCK:
        if where not in _SHARED:
            _SHARED[where] = IndexManager(where, gpu=settings.flag(conn, "faiss_gpu"))
        return _SHARED[where]


def align(conn, manager: IndexManager, spec: SpaceSpec, ids, fetch, now: float) -> None:
    """Make `spec.key` resident and holding exactly these rows, and record
    the space's durable identity in `derived_similarity_space`.

    The cheap tiers first: an already-resident space is diffed against
    the wanted rows and mutated -- adds are searchable immediately,
    strangers leave -- and a cold process tries the snapshot before
    paying for a full build. Any mutation is checkpointed, so the next
    boot restores instead of rebuilding.

    Align digests COMMITTED truth only. A producer must never push its
    own uncommitted rows into a live index: the runner rolls a failed
    item's writes back (db/runner.py, `conn.rollback()` on
    ITEM_FAILURES) and an index cannot ride that rollback, so producers
    write rows and this function reconciles the index with what commits
    actually kept.

    What "same row" means differs by representation, and each gets the
    check its hazard demands:

    - binary: the id is a FILE, whose hash changes when its bytes do --
      so every held row's value is compared and a changed one is
      re-indexed. The values are 8 bytes each and the caller already
      holds them; the comparison is free.
    - float32: the id is never reused -- derived_face_instance is
      AUTOINCREMENT precisely so a deleted face's id cannot come back
      wearing a different embedding (db/schema.sql) -- and embeddings
      are written once at detection, so the id diff IS the content
      diff, and gigabytes of blobs stay unread on the restore path.
    """
    import numpy as np

    from vision.faiss_index import _signed_to_packed

    wanted = [int(v) for v in ids]
    if not manager.has(spec.key):
        manager.restore(spec)
    changed = False
    if not manager.has(spec.key):
        manager.load(spec, wanted, fetch(wanted))
        changed = True
    else:
        held = set(manager.ids(spec.key).tolist())
        strangers = sorted(held - set(wanted))
        missing = [v for v in wanted if v not in held]
        if strangers:
            manager.remove(spec.key, strangers)
        stale: list[int] = []
        if spec.representation == "binary":
            keeping = [v for v in wanted if v in held and v not in set(missing)]
            if keeping:
                values = fetch(keeping)
                stored = dict(zip(manager.ids(spec.key).tolist(), manager.vectors(spec.key), strict=True))
                packed = _signed_to_packed(values, spec.dimensions)
                stale = [v for at, v in enumerate(keeping) if not np.array_equal(packed[at], stored[v])]
                if stale:
                    manager.remove(spec.key, stale)
        renewed = missing + stale
        if renewed:
            manager.add(spec.key, renewed, fetch(renewed))
        changed = bool(strangers or renewed)
    if changed:
        manager.checkpoint(spec.key)
    conn.execute(
        "INSERT INTO derived_similarity_space(key, representation, dimensions, metric,"
        " producer, producer_version, aligned_at) VALUES(?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(key) DO UPDATE SET representation = excluded.representation,"
        " dimensions = excluded.dimensions, metric = excluded.metric,"
        " producer = excluded.producer, producer_version = excluded.producer_version,"
        " aligned_at = excluded.aligned_at",
        (spec.key, spec.representation, spec.dimensions, spec.metric, spec.producer, spec.producer_version, now),
    )


def pair_graph(manager: IndexManager, key: str, threshold):
    """Every pair of stored rows within `threshold`, as (ids, ids, weights).

    Both directions of each pair are present and self-pairs are dropped
    -- the shape union-find and label propagation both consume. For
    binary spaces `threshold` is hamming bits and weights are distances;
    for float spaces it is the cosine floor and weights are similarities.
    """
    import numpy as np

    lims, neighbours, weights = manager.range(key, threshold)
    rows = np.repeat(manager.ids(key), np.diff(np.asarray(lims)).astype(np.int64))
    keep = rows != neighbours
    return rows[keep], neighbours[keep], weights[keep]


def normalise(vectors):
    """Unit-length rows, so an inner product IS the cosine."""
    import numpy as np

    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return np.ascontiguousarray(matrix / lengths, dtype=np.float32)


def as_csr(n: int, rows, cols, weights):
    """Edge arrays into the positional CSR shape `grouping.group` eats.

    For callers holding id-keyed edges from `pair_graph` who need the
    propagation algorithms' positional form back.
    """
    import numpy as np

    return _csr(
        n,
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
        np,
    )


def _csr(n, rows, cols, weights, np):
    """(indptr, cols, weights): row i's neighbours are cols[indptr[i]:indptr[i+1]]."""
    order = np.argsort(rows, kind="stable")
    rows, cols, weights = rows[order], cols[order], weights[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    return indptr, cols, weights.astype(np.float32)


# -- the exact oracle, for tests and diagnostics only -----------------------

#: Rows per block for the numpy oracle. An n x n float32 matrix at 100,000
#: vectors is 40 GB; a 2,048-row block is 800 MB.
BLOCK = 2_048


def numpy_graph(vectors, threshold: float):
    """Every pair at or above `threshold`, positional CSR, exactly.

    `>=` on the raw float comparison -- the inclusive semantics every
    radius in this application documents. The FAISS layer absorbs its
    strict comparisons to agree with this function, not the reverse.
    """
    import numpy as np

    unit = normalise(vectors)
    if unit.shape[0] == 0:
        return np.zeros(1, "int64"), np.zeros(0, "int64"), np.zeros(0, "float32")
    rows, cols, weights = [], [], []
    for start in range(0, unit.shape[0], BLOCK):
        block = unit[start : start + BLOCK]
        sims = block @ unit.T
        keep = sims >= threshold
        here = np.arange(block.shape[0])
        keep[here, here + start] = False
        row, col = np.nonzero(keep)
        rows.append(row.astype(np.int64) + start)
        cols.append(col.astype(np.int64))
        weights.append(sims[keep])
    empty_i, empty_f = np.zeros(0, "int64"), np.zeros(0, "float32")
    return _csr(
        unit.shape[0],
        np.concatenate(rows) if rows else empty_i,
        np.concatenate(cols) if cols else empty_i,
        np.concatenate(weights) if weights else empty_f,
        np,
    )
