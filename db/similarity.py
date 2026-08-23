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
#: imagehash.phash (vision/dupes.py perceptual) over the repo's oriented
#: frame -- for a video, the poster frame (db/runner.py _perceptual_item).
#: The preprocess token is part of the representation's identity: the
#: same hash algorithm over a differently chosen or differently oriented
#: frame is a different number, and tests pin what "v1" means. Bump it
#: when the frame policy meaningfully changes.
PHASH = SpaceSpec(
    key="perceptual.phash64",
    representation="binary",
    dimensions=64,
    metric="hamming",
    producer="imagehash.phash",
    producer_version=_imagehash_version(),
    preprocess="smartgallery.perceptual-frame",
    preprocess_version="v1",
)

#: The difference-hash space: a DIFFERENT algorithm over the same frame,
#: so a different space -- two values sharing one provenance row is how
#: dHash bits got labeled as pHash output. Recorded for future retrieval
#: work; nothing searches it yet.
DHASH = SpaceSpec(
    key="perceptual.dhash64",
    representation="binary",
    dimensions=64,
    metric="hamming",
    producer="imagehash.dhash",
    producer_version=_imagehash_version(),
    preprocess="smartgallery.perceptual-frame",
    preprocess_version="v1",
)


def face_space(model_id: str, model_version: str, dimensions: int) -> SpaceSpec:
    """One space per recognition model+version: embeddings from different
    models never share an index, because their cosines are not comparable.
    The preprocess token covers the repo's side of the pipeline -- the
    oriented frame handed to the detector (db/oriented.py)."""
    return SpaceSpec(
        key=f"face.{model_id}.{model_version}",
        representation="float32",
        dimensions=int(dimensions),
        metric="cosine",
        producer=model_id,
        producer_version=model_version,
        preprocess="smartgallery.oriented-face",
        preprocess_version="v1",
    )


#: Every field that carries meaning. The spec hash -- and so a space's
#: immutable identity -- is exactly these, in this order.
_MEANING = (
    "key",
    "representation",
    "dimensions",
    "metric",
    "producer",
    "producer_version",
    "preprocess",
    "preprocess_version",
)


def spec_hash(spec: SpaceSpec) -> str:
    """Canonical JSON, then SHA-256: a delimiter join lets two different
    specs collide the moment a field contains the delimiter, and identity
    hashes are the wrong place for folklore."""
    import hashlib
    import json

    canon = json.dumps([getattr(spec, field) for field in _MEANING], separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode()).hexdigest()


def space_id(conn, spec: SpaceSpec, now: float) -> int:
    """The immutable `similarity_space` row for this spec, minted once.

    Rows are keyed by the hash of every meaning-bearing field and never
    updated (the schema enforces it with a trigger): an upgraded
    producer or preprocess mints a NEW space, and rows written under the
    old one keep saying -- forever -- what actually computed them."""
    digest = spec_hash(spec)
    row = conn.execute("SELECT id FROM similarity_space WHERE spec_hash = ?", (digest,)).fetchone()
    if row is not None:
        return int(row[0])
    cursor = conn.execute(
        "INSERT INTO similarity_space(key, representation, dimensions, metric,"
        " producer, producer_version, preprocess, preprocess_version, spec_hash, created_at)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (*(getattr(spec, field) for field in _MEANING), digest, now),
    )
    return int(cursor.lastrowid or 0)


def semantic_space(model: str, checkpoint: str, dimensions: int) -> SpaceSpec:
    """The joint image/text space one OpenCLIP checkpoint defines.

    ONE space for both modalities -- that is the entire trick: image
    vectors are the durable rows, a typed phrase becomes an ephemeral
    query vector in the same space, and they are comparable because one
    model produced both. The identity itself belongs to the adapter
    (vision/semantic/openclip.py space()); this is its name here for
    the callers that speak spaces rather than providers.
    """
    from vision.semantic import openclip

    return openclip.space(model, checkpoint, dimensions)


def _current_space_of(conn, probe: SpaceSpec) -> tuple[int, int] | None:
    """The registry row matching every current-identity field of `probe`
    except dimensions (which the registry itself supplies), newest first
    -- spaces are immutable, so newest is the one the current producer
    writes into. Returns (space id, dimensions) or None."""
    row = conn.execute(
        "SELECT id, dimensions FROM similarity_space"
        " WHERE key = ? AND representation = ? AND metric = ?"
        " AND producer = ? AND producer_version = ? AND preprocess = ? AND preprocess_version = ?"
        " ORDER BY id DESC LIMIT 1",
        (
            probe.key,
            probe.representation,
            probe.metric,
            probe.producer,
            probe.producer_version,
            probe.preprocess,
            probe.preprocess_version,
        ),
    ).fetchone()
    if row is None:
        return None
    return int(row[0]), int(row[1])


def face_space_of(conn, model_id: str, model_version: str) -> tuple[int, SpaceSpec] | None:
    """The CURRENT face space for this model, from the registry -- id and
    full spec, dimensions included -- or None when the model has minted
    nothing yet. Clustering keys on the returned space id; nothing
    reconstructs a space's meaning from the duplicated model columns."""
    found = _current_space_of(conn, face_space(model_id, model_version, 1))
    if found is None:
        return None
    sid, dimensions = found
    return sid, face_space(model_id, model_version, dimensions)


def keyed(spec: SpaceSpec, sid: int, lane: str = "") -> SpaceSpec:
    """The spec as the index layer sees it: the resident key and every
    snapshot filename carry the immutable space id, so an upgraded spec
    can never restore -- or answer for -- an older space's vectors.

    `lane` names a second CORPUS in the same space: prompt vectors live
    in the provider's joint space beside its media vectors (same
    coordinates, comparable cosines) but their ids are another table's,
    so they get their own resident index -- `<key>@<sid>+<lane>`."""
    import dataclasses

    return dataclasses.replace(spec, key=f"{spec.key}@{sid}" + (f"+{lane}" if lane else ""))


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


def align(conn, manager: IndexManager, spec: SpaceSpec, ids, fetch, now: float, *, lane: str = "") -> str:
    """Make this spec's space resident and holding exactly these rows.

    Returns the resident key -- `<spec.key>@<space id>` -- which is the
    only name the index layer knows the space by. Keying residency and
    snapshots on the immutable space id is what makes provenance
    laundering structurally impossible: an upgraded producer resolves to
    a NEW space id, so the old snapshot is a different file it never
    opens and the old rows are a different space it never loads.

    The cheap tiers first: an already-resident space is diffed against
    the wanted rows and mutated; a cold process tries the snapshot
    before paying for a full build. Any mutation is checkpointed, so
    the next boot restores instead of rebuilding.

    Align is the REPAIR path and digests committed truth only -- it
    reconciles the resident tier after boot, crash, or drift. The live
    path is the runner's post-commit sync (`apply_pending`): producers
    note their writes, the runner applies them only after the commit
    that made them durable succeeded (db/runner.py), and a rollback
    discards them unapplied. The invariant both paths hold together:
    the resident index may lag committed SQLite, it may never lead it.

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

    named = keyed(spec, space_id(conn, spec, now), lane)
    wanted = [int(v) for v in ids]
    if not manager.has(named.key):
        manager.restore(named)
    changed = False
    if not manager.has(named.key):
        manager.load(named, wanted, fetch(wanted))
        changed = True
    else:
        held = set(manager.ids(named.key).tolist())
        strangers = sorted(held - set(wanted))
        missing = [v for v in wanted if v not in held]
        if strangers:
            manager.remove(named.key, strangers)
        stale: list[int] = []
        if named.representation == "binary":
            keeping = [v for v in wanted if v in held and v not in set(missing)]
            if keeping:
                values = fetch(keeping)
                stored = dict(zip(manager.ids(named.key).tolist(), manager.vectors(named.key), strict=True))
                packed = _signed_to_packed(values, named.dimensions)
                stale = [v for at, v in enumerate(keeping) if not np.array_equal(packed[at], stored[v])]
                if stale:
                    manager.remove(named.key, stale)
        renewed = missing + stale
        if renewed:
            manager.add(named.key, renewed, fetch(renewed))
        changed = bool(strangers or renewed)
    if changed:
        manager.checkpoint(named.key)
    return named.key


# -- the live path: producers note, the runner applies after commit ---------

#: Pending index mutations live ON the connection (db/connect.py
#: Connection.pending), applied only after the commit that made their
#: rows durable: every runner turn ends in exactly one of
#: apply_pending/discard_pending, close() discards whatever is left, and
#: a connection that dies any other way takes its notes with it.


def pending(conn) -> list:
    return conn.pending


def note(conn, spec: SpaceSpec, subject_id: int, value, now: float, *, lane: str = "") -> None:
    """A producer wrote (or deleted, value=None) one representation row.

    Nothing touches the resident index here -- the write may yet roll
    back. The runner applies the note after its commit succeeds."""
    named = keyed(spec, space_id(conn, spec, now), lane)
    pending(conn).append((named.key, int(subject_id), value))


def note_gone(conn, sid: int, subject_id: int, *, lane: str = "") -> None:
    """A producer deleted a row that may belong to an OLDER space than the
    current spec resolves to -- the resident key is reconstructed from
    the immutable row the deleted data pointed at."""
    row = conn.execute("SELECT key FROM similarity_space WHERE id = ?", (sid,)).fetchone()
    if row is not None:
        key = f"{row[0]}@{sid}" + (f"+{lane}" if lane else "")
        pending(conn).append((key, int(subject_id), None))


def apply_pending(conn, manager: IndexManager | None = None) -> None:
    """The runner's half of the sync, called strictly AFTER conn.commit().

    Notes are batched per space with the last write per subject winning,
    then handed to the manager's own mutation primitives -- membership is
    the space's bookkeeping, so applying a commit never scans the index
    it updates. Only already-resident spaces mutate; a cold space is
    built by align from committed rows.

    Failure marks the space unservable rather than pretending: a space
    that took half a batch could answer with stale rows, so it is
    invalidated on the spot -- resident and snapshot both -- and the
    next align rebuilds it from committed truth. SQLite stays
    authoritative through every branch; the index may lag it, never
    lead it, and never quietly diverge from it."""
    import logging

    resolved = manager_for(conn) if manager is None else manager
    final: dict[str, dict[int, object]] = {}
    noted, conn.pending = conn.pending, []
    for key, subject, value in noted:
        final.setdefault(key, {})[subject] = value
    for key, changes in final.items():
        if not resolved.has(key):
            continue
        try:
            gone = [subject for subject, value in changes.items() if value is None]
            kept = {subject: value for subject, value in changes.items() if value is not None}
            if gone:
                resolved.remove_present(key, gone)
            if kept:
                resolved.upsert(key, list(kept), list(kept.values()))
        except Exception:
            resolved.invalidate(key)
            logging.getLogger(__name__).exception(
                "post-commit sync failed for %s; the space is invalidated and the next align rebuilds it", key
            )


def discard_pending(conn) -> None:
    """The rollback half: the rows never became durable, so their notes
    must never reach an index."""
    conn.pending = []


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
