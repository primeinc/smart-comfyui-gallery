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

#: The perceptual-hash space: 64 hamming bits per picture, produced by
#: the phash job and consumed by dupe grouping.
PHASH = SpaceSpec(key="perceptual.phash64", representation="binary", dimensions=64, metric="hamming")


def face_space(model_id: str, model_version: str, dimensions: int) -> SpaceSpec:
    """One space per recognition model+version: embeddings from different
    models never share an index, because their cosines are not comparable."""
    return SpaceSpec(
        key=f"face.{model_id}.{model_version}",
        representation="float32",
        dimensions=int(dimensions),
        metric="cosine",
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


def align(manager: IndexManager, spec: SpaceSpec, ids, fetch) -> None:
    """Make `spec.key` resident and holding exactly `ids`.

    The cheap tiers first: an already-resident space is diffed against
    the wanted ids and mutated -- adds are searchable immediately,
    strangers leave -- reading representations (`fetch(missing_ids)`)
    only for rows the index does not already hold. A cold process tries
    the snapshot before paying for a full build. Any mutation is
    checkpointed, so the next boot restores instead of rebuilding.

    The id diff is sufficient because every producer writes derived rows
    fresh under new ids or upserts through the index (db/derived.py
    record_hash) -- a row's representation never changes behind an id
    this module would keep.
    """
    wanted = [int(v) for v in ids]
    if not manager.has(spec.key):
        manager.restore(spec)
    if not manager.has(spec.key):
        manager.load(spec, wanted, fetch(wanted))
        manager.checkpoint(spec.key)
        return
    held = set(manager.ids(spec.key).tolist())
    strangers = sorted(held - set(wanted))
    missing = [v for v in wanted if v not in held]
    if strangers:
        manager.remove(spec.key, strangers)
    if missing:
        manager.add(spec.key, missing, fetch(missing))
    if strangers or missing:
        manager.checkpoint(spec.key)


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
