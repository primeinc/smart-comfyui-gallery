"""Resident FAISS spaces over stable SQLite ids -- lifecycle and execution.

This layer owns indexes the way the reference services do
(../refs/facebookresearch/distributed-faiss server.py holds a
`dict[index_id -> Index]` its README compares to "fully separate tables
in an SQL database"; ../refs/neuml/txtai keeps one ANN per config beside
its database): named spaces, resident for the process lifetime, mutated
in place, snapshotted to disk, restored at boot. What a vector MEANS is
the caller's `SpaceSpec`; graph semantics, thresholds and grouping live
in db/similarity.py -- this module executes.

Three tiers, and this manager owns the traffic between them:

- SQLite is the durable truth. Losing every other tier loses nothing.
- Disk snapshots (`write_index` / `write_index_binary` + a JSON sidecar)
  are a startup accelerator. A snapshot that does not match -- wrong
  spec, wrong count, unreadable file -- is refused, because FAISS itself
  does not validate what it loads (faiss.wiki Index-IO).
- RAM/VRAM holds the live index that answers. Indexes are REBUILDABLE,
  never routinely rebuilt: `add` rows are searchable immediately,
  `remove` takes them out, and every other id survives both.

Index classes, from upstream's own guidance (faiss.wiki
Guidelines-to-choose-an-index: exact results -> "Flat"; flat indexes
need no training, and re-training is not a FAISS concept -- FAQ: "it is
simpler to just construct a new one"):

- float32/cosine: `IndexIDMap2(IndexFlatIP)`, rows L2-normalised here so
  inner product IS the cosine (faiss.wiki MetricType-and-distances).
  IDMap2 stores ids explicitly, so removal keeps every other id
  (faiss/IndexIDMap.h; faiss.wiki Special-operations-on-indexes). txtai
  deploys the same shape at this scale ("IDMap,Flat" for small exact
  indexes, txtai ann/dense/faiss.py `configure`).
- binary/hamming: `IndexBinaryIDMap2(IndexBinaryFlat)`, popcount
  exhaustive search (faiss.wiki Binary-Indexes).

Device policy is CONFIGURATION, decided when the manager is built --
never an argument on a search. With `gpu=True` a float space keeps one
resident device clone of its inner flat index, per upstream's residency
doctrine: "it is best to copy an index once to a GPU and keep it there"
(faiss.wiki Comparing-GPU-vs-CPU). The clone is invalidated by mutation
and rebuilt on the next search. GPU range search is emulated exactly --
k nearest on the device, a CPU range pass for any query whose k-th
neighbour still cleared the radius (faiss/contrib/exhaustive_search.py:
60-116). A build without GPU support serves the same answers from the
CPU canonical. Binary flat search has no GPU implementation in any
faiss build, so binary spaces serve from CPU always. The GPU never
serialises; snapshots are the CPU canonical form (faiss.wiki
Faiss-on-the-GPU: "a GPU index should be converted to CPU ... before
storing it").

Locking is this layer's job because it is nobody else's: "There is no
locking mechanism in place ... the calling code should maintain a lock"
(faiss.wiki FAQ), and one `StandardGpuResources` may serve several
indexes only if they never issue concurrent queries (faiss.wiki
Running-on-GPUs). One lock per space around anything touching its
index; one manager-wide lock around all GPU work.

Radius arguments are INCLUSIVE, the way the settings that feed them are
documented; FAISS's strict comparisons are absorbed here, at the one
boundary that knows about them.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
import threading
from dataclasses import dataclass
from typing import Any

_logger = logging.getLogger(__name__)

#: Neighbours asked of the GPU before the CPU is consulted for a query
#: that had more. Upstream's own default. A GPU index caps k at 2048 and
#: selection cost climbs above about 512 (faiss.wiki Faiss-on-the-GPU,
#: "Limitations"), so this already sits near the useful ceiling.
GPU_K = 1024

#: The hard device ceiling on k-selection: past it the GPU index refuses
#: outright ("GPU index only supports min/max-K selection up to 2048",
#: faiss/gpu/impl/IndexUtils.cu validateKSelect). A deeper search --
#: retrieval asking for a whole scoped ranking -- serves from the CPU
#: canonical, which computes the same exact flat answer with no ceiling.
GPU_MAX_K = 2048

#: The ways a faiss capability fails to exist on a machine: no module, a
#: DLL that will not load, an API the build lacks, no GPU, a SWIG-level
#: refusal. Named so the device probe catches what "not available"
#: actually raises and nothing more -- a genuine bug propagates instead
#: of reading as absence.
FALLIBLE = (ImportError, OSError, RuntimeError, AttributeError, ValueError, TypeError)


@dataclass(frozen=True)
class SpaceSpec:
    """What the vectors in one space mean -- never how they are searched.

    `producer` and `producer_version` name what computed the vectors: an
    index of ArcFace embeddings answered with SFace queries is garbage
    with a valid shape. `preprocess` and `preprocess_version` name what
    fed the computation -- the same hash algorithm over a differently
    oriented frame is a different representation. All of it is part of
    the spec, rides the snapshot sidecar, and a snapshot from an
    obsolete producer or preprocess is refused the same as one with the
    wrong dimensions."""

    key: str
    representation: str  # 'binary' | 'float32'
    dimensions: int  # bits for binary, floats for float32
    metric: str  # 'hamming' for binary, 'cosine' for float32
    producer: str = ""
    producer_version: str = ""
    preprocess: str = ""
    preprocess_version: str = ""


def _signed_to_packed(values, bits: int):
    """Schema-form signed integers to the uint8 rows FAISS wants.

    SQLite INTEGER is signed 64-bit, so the stored hash is the unsigned
    value folded into the signed range; big-endian byte order, though any
    order works as long as every row uses the same one -- XOR popcount
    does not care."""
    import numpy as np

    unsigned = np.array([v & 0xFFFFFFFFFFFFFFFF for v in values], dtype=np.uint64)
    return np.ascontiguousarray(unsigned.astype(">u8").view(np.uint8).reshape(-1, bits // 8))


def _unit(vectors):
    """Unit-length float32 rows, so inner product IS the cosine (faiss.wiki
    MetricType-and-distances). Zero rows stay zero instead of dividing by
    zero, which `faiss.normalize_L2` would."""
    import numpy as np

    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        return matrix
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return np.ascontiguousarray(matrix / lengths, dtype=np.float32)


def _inclusive(threshold: float) -> float:
    """One float32 step below the threshold.

    FAISS keeps `similarity > radius` for an inner-product metric where
    this application's thresholds are documented at-or-above. `nextafter`
    steps the radius down by one representable value, making the strict
    comparison mean the inclusive one."""
    import numpy as np

    return float(np.nextafter(np.float32(threshold), np.float32("-inf")))


class _Space:
    """One resident space: its meaning, its live index, its book-keeping."""

    def __init__(self, spec: SpaceSpec, index, known: set[int]):
        self.spec = spec
        self.index = index
        self.known = known
        self.dirty = True
        self.gpu_clone: Any = None
        self.lock = threading.RLock()


class IndexManager:
    """Named resident spaces: load or restore, mutate, search, checkpoint.

    `gpu` is the device policy for every float space this manager holds,
    fixed at construction the way the reference services fix it in
    configuration. Without a `snapshot_dir` the manager is RAM-only --
    checkpoint and restore politely do nothing, which is what an
    in-memory database wants.
    """

    def __init__(self, snapshot_dir: str | pathlib.Path | None = None, *, gpu: bool = True):
        self._snapshots = pathlib.Path(snapshot_dir) if snapshot_dir is not None else None
        self._gpu_wanted = gpu
        self._spaces: dict[str, _Space] = {}
        self._served: dict[str, str] = {}
        self._lock = threading.RLock()  # guards the _spaces dict itself
        self._gpu_lock = threading.RLock()  # one resources object, never queried concurrently
        self._resources = None

    # -- building -----------------------------------------------------------

    def load(self, spec: SpaceSpec, ids, vectors) -> None:
        """Replace `spec.key`'s rows with these ids and representations."""
        import numpy as np

        if spec.representation not in ("binary", "float32"):
            raise ValueError(f"unknown representation {spec.representation!r}")
        wants = {"binary": "hamming", "float32": "cosine"}[spec.representation]
        if spec.metric != wants:
            raise ValueError(f"{spec.representation} spaces take the {wants} metric, not {spec.metric!r}")
        with self._lock:
            held = self._spaces.get(spec.key)
            if held is not None and held.spec != spec:
                raise ValueError(f"{spec.key!r} is already loaded under a different spec")

        keys = np.asarray(list(ids), dtype=np.int64)
        if len(set(keys.tolist())) != keys.shape[0]:
            raise ValueError(f"{spec.key}: every id may appear once")

        faiss = self._faiss()
        if spec.representation == "binary":
            if spec.dimensions % 8:
                raise ValueError(f"binary dimensions must be whole bytes, not {spec.dimensions}")
            packed = _signed_to_packed(vectors, spec.dimensions)
            if keys.shape[0] != packed.shape[0]:
                raise ValueError(f"{keys.shape[0]} ids for {packed.shape[0]} rows")
            index = faiss.IndexBinaryIDMap2(faiss.IndexBinaryFlat(spec.dimensions))
            if keys.shape[0]:
                index.add_with_ids(packed, keys)
        else:
            unit = _unit(vectors)
            if unit.ndim != 2 or unit.shape[1] != spec.dimensions:
                raise ValueError(f"{spec.key} declares dimensions {spec.dimensions}, rows have {unit.shape}")
            if keys.shape[0] != unit.shape[0]:
                raise ValueError(f"{keys.shape[0]} ids for {unit.shape[0]} rows")
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(spec.dimensions))
            if keys.shape[0]:
                index.add_with_ids(unit, keys)
        with self._lock:
            self._spaces[spec.key] = _Space(spec, index, set(keys.tolist()))

    def restore(self, spec: SpaceSpec) -> bool:
        """Snapshot to resident, or False. Every mismatch is a refusal:
        the sidecar must claim exactly this spec and the index file must
        open and hold the counted rows -- the same internal-consistency
        check distributed-faiss makes (index.py from_storage_dir). A
        False costs the caller one rebuild; a wrong True costs wrong
        neighbours."""
        if self._snapshots is None:
            return False
        index_path, sidecar_path = self._files(spec.key)
        if not index_path.exists() or not sidecar_path.exists():
            return False
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as why:
            _logger.warning("snapshot %s refused: sidecar unreadable: %s: %s", spec.key, type(why).__name__, why)
            return False
        fields = (
            "key",
            "representation",
            "dimensions",
            "metric",
            "producer",
            "producer_version",
            "preprocess",
            "preprocess_version",
        )
        if tuple(sidecar.get(field) for field in fields) != tuple(getattr(spec, field) for field in fields):
            _logger.warning(
                "snapshot %s refused: sidecar claims %r, this build wants %r",
                spec.key,
                {field: sidecar.get(field) for field in fields},
                {field: getattr(spec, field) for field in fields},
            )
            return False
        faiss = self._faiss()
        try:
            if spec.representation == "binary":
                index = faiss.read_index_binary(str(index_path))
            else:
                index = faiss.read_index(str(index_path))
        except (RuntimeError, OSError) as why:
            _logger.warning("snapshot %s refused: %s unreadable: %s: %s", spec.key, index_path, type(why).__name__, why)
            return False
        # read_index hands back the concrete class (faiss's out-typemap
        # downcasts every Index*); this proxy OWNS the object, and a
        # downcast_index() on it would be a second, non-owning proxy left
        # pointing at freed memory once this one is collected. A snapshot
        # that is not an id-mapped index is not this manager's snapshot,
        # whatever the sidecar says.
        if not isinstance(index, (faiss.IndexIDMap, faiss.IndexBinaryIDMap)):
            _logger.warning(
                "snapshot %s refused: %s holds a %s, not an id-mapped index", spec.key, index_path, type(index).__name__
            )
            return False
        if int(index.ntotal) != sidecar.get("vectors"):
            _logger.warning(
                "snapshot %s refused: %s holds %d vectors, sidecar counts %r",
                spec.key,
                index_path,
                int(index.ntotal),
                sidecar.get("vectors"),
            )
            return False
        space = _Space(spec, index, set(faiss.vector_to_array(index.id_map).tolist()))
        space.dirty = False
        with self._lock:
            self._spaces[spec.key] = space
        return True

    # -- mutating -----------------------------------------------------------

    def add(self, key: str, ids, vectors) -> None:
        """New rows, searchable the moment this returns."""
        import numpy as np

        space = self._space(key)
        keys = np.asarray(list(ids), dtype=np.int64)
        taken = [k for k in keys.tolist() if k in space.known]
        if taken:
            raise ValueError(f"{key}: ids {taken} are already in the space")
        with space.lock:
            if space.spec.representation == "binary":
                space.index.add_with_ids(_signed_to_packed(vectors, space.spec.dimensions), keys)
            else:
                space.index.add_with_ids(_unit(vectors), keys)
            space.known.update(keys.tolist())
            space.dirty = True
            space.gpu_clone = None

    def remove(self, key: str, ids) -> None:
        import numpy as np

        space = self._space(key)
        keys = np.asarray(list(ids), dtype=np.int64)
        strangers = [k for k in keys.tolist() if k not in space.known]
        if strangers:
            raise ValueError(f"{key}: ids {strangers} are not in the space")
        faiss = self._faiss()
        with space.lock:
            space.index.remove_ids(faiss.IDSelectorBatch(keys))
            space.known.difference_update(keys.tolist())
            space.dirty = True
            space.gpu_clone = None

    def upsert(self, key: str, ids, vectors) -> None:
        """Replace-or-add these rows, searchable the moment this returns.

        The representation mutation primitive the post-commit sync uses:
        membership is answered from the space's own bookkeeping, never by
        reading every stored id back out -- an incremental commit must
        not cost a scan of the index it is updating."""
        space = self._space(key)
        with space.lock:
            present = [int(v) for v in ids if int(v) in space.known]
            if present:
                self.remove(key, present)
            self.add(key, ids, vectors)

    def remove_present(self, key: str, ids) -> None:
        """Remove whichever of these rows the space holds; strangers are
        no-ops. For deletions noted before the deleting commit -- the
        row is gone from SQLite whether or not it was ever resident."""
        space = self._space(key)
        with space.lock:
            present = [int(v) for v in ids if int(v) in space.known]
            if present:
                self.remove(key, present)

    def invalidate(self, key: str) -> None:
        """Drop the space everywhere -- resident AND snapshot. The rows
        it described changed meaning; a tier that outlives that is a
        wrong answer waiting for a boot."""
        with self._lock:
            self._spaces.pop(key, None)
            self._served.pop(key, None)
        for path in self._files(key):
            path.unlink(missing_ok=True)

    # -- persisting ---------------------------------------------------------

    def checkpoint(self, key: str) -> pathlib.Path | None:
        """The CPU canonical form to disk, with the sidecar a restore
        judges it by. RAM-only managers skip."""
        if self._snapshots is None:
            return None
        space = self._space(key)
        faiss = self._faiss()
        self._snapshots.mkdir(parents=True, exist_ok=True)
        index_path, sidecar_path = self._files(key)
        with space.lock:
            if space.spec.representation == "binary":
                faiss.write_index_binary(space.index, str(index_path))
            else:
                faiss.write_index(space.index, str(index_path))
            sidecar = {
                "key": space.spec.key,
                "representation": space.spec.representation,
                "dimensions": space.spec.dimensions,
                "metric": space.spec.metric,
                "producer": space.spec.producer,
                "producer_version": space.spec.producer_version,
                "preprocess": space.spec.preprocess,
                "preprocess_version": space.spec.preprocess_version,
                "normalization": "l2" if space.spec.representation == "float32" else None,
                "vectors": int(space.index.ntotal),
                "faiss": getattr(faiss, "__version__", None),
            }
            sidecar_path.write_text(json.dumps(sidecar, indent=1), encoding="utf-8")
            space.dirty = False
        return index_path

    def checkpoint_all(self) -> list[pathlib.Path]:
        """Every space that changed since its last write -- the shutdown
        sweep, cheap when nothing moved."""
        with self._lock:
            keys = [key for key, space in self._spaces.items() if space.dirty]
        return [written for key in keys if (written := self.checkpoint(key)) is not None]

    # -- answering ----------------------------------------------------------

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._spaces

    def count(self, key: str) -> int:
        space = self._space(key)
        with space.lock:
            return int(space.index.ntotal)

    def ids(self, key: str):
        """The stored ids in index order, as int64."""
        faiss = self._faiss()
        space = self._space(key)
        with space.lock:
            return faiss.vector_to_array(space.index.id_map)

    def vectors(self, key: str):
        """The stored rows in the same order as `ids` -- unit float32 for
        float spaces, packed uint8 for binary. Read back from the index
        itself: the index IS the store."""
        space = self._space(key)
        with space.lock:
            return self._matrix(space)

    def served_by(self, key: str) -> str | None:
        """Which execution answered this space's last search -- provenance
        for run rows, because a timing nobody can attribute to a machine
        is not a measurement."""
        return self._served.get(key)

    def search(self, key: str, queries, k: int):
        """The `k` nearest stored rows per query: (ids, scores), ids -1
        where fewer than `k` rows exist. Queries arrive raw; this layer
        applies the space's preprocessing."""
        space = self._space(key)
        with space.lock:
            if space.spec.representation == "binary":
                packed = _signed_to_packed(queries, space.spec.dimensions)
                self._served[key] = "faiss-cpu"
                distances, labels = space.index.search(packed, int(k))
                return labels, distances
            unit = _unit(queries)
            device = self._device_for(space) if int(k) <= GPU_MAX_K else None
            if device is not None:
                with self._gpu_lock:
                    scores, positions = device.search(unit, int(k))
                held = self.ids(key)
                labels = held[positions.clip(min=0)]
                labels[positions < 0] = -1
                self._served[key] = "faiss-gpu"
                return labels, scores
            self._served[key] = "faiss-cpu"
            scores, labels = space.index.search(unit, int(k))
            return labels, scores

    def range(self, key: str, radius, queries=None):
        """Every stored row within `radius` (INCLUSIVE) of each query, as
        FAISS range triplets (lims, ids, distances): query i's neighbours
        are ids[lims[i]:lims[i+1]]. `queries=None` searches the space
        against its own rows in id order -- the self-join the pair graph
        is built from, without copying the store out.

        For binary spaces `radius` is hamming bits and distances are
        bits; for float spaces it is the cosine similarity floor and
        distances are similarities.
        """
        import numpy as np

        space = self._space(key)
        with space.lock:
            if space.spec.representation == "binary":
                if int(space.index.ntotal) == 0:
                    return np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32)
                data = self._matrix(space) if queries is None else _signed_to_packed(queries, space.spec.dimensions)
                # IndexBinary.range_search keeps distance < radius ("only
                # distances < radius (strict comparison)"), so +1 makes
                # the argument inclusive. Labels are the stored ids.
                lims, distances, labels = space.index.range_search(data, int(radius) + 1)
                self._served[key] = "faiss-cpu"
                return lims, labels.astype(np.int64), distances

            if int(space.index.ntotal) == 0:
                return np.zeros(1, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
            unit = self._matrix(space) if queries is None else _unit(queries)
            device = self._device_for(space)
            if device is not None:
                from faiss.contrib.exhaustive_search import range_search_gpu

                inner = space.index.index
                with self._gpu_lock:
                    # `inner` as the CPU fallback rather than None: contrib
                    # runs a CPU range pass for queries whose k-th neighbour
                    # still cleared the radius, which is what makes this
                    # exact instead of a top-k approximation.
                    lims, distances, positions = range_search_gpu(unit, _inclusive(radius), device, inner, gpu_k=GPU_K)
                held = self.ids(key)
                self._served[key] = "faiss-gpu"
                return np.asarray(lims), held[positions], np.asarray(distances, dtype=np.float32)
            lims, distances, labels = space.index.range_search(unit, _inclusive(radius))
            self._served[key] = "faiss-cpu"
            return lims, labels.astype(np.int64), np.asarray(distances, dtype=np.float32)

    # -- plumbing -----------------------------------------------------------

    def _space(self, key: str) -> _Space:
        with self._lock:
            return self._spaces[key]

    def _matrix(self, space: _Space):
        """The stored rows in index order, read back from the index itself
        -- the index IS the store, a parallel copy would drift on the
        first remove_ids compaction."""
        if space.spec.representation == "binary":
            packed = space.index.index
            # uint8 rows of d/8 bytes: the codes, in index order
            return packed.reconstruct_n(0, int(packed.ntotal))
        inner = space.index.index
        return inner.reconstruct_n(0, int(inner.ntotal))

    def _device_for(self, space: _Space):
        """The space's resident GPU clone, built once and kept ("copy an
        index once to a GPU and keep it there"), or None when policy or
        the build says CPU. Callers already hold the space lock."""
        if not self._gpu_wanted or space.spec.representation == "binary":
            return None
        if space.gpu_clone is not None:
            return space.gpu_clone
        try:
            faiss = self._faiss()
            if not hasattr(faiss, "StandardGpuResources") or faiss.get_num_gpus() < 1:
                return None
            with self._gpu_lock:
                if self._resources is None:
                    resources = faiss.StandardGpuResources()
                    # Both from the wiki's brute-force page: FAISS orders its
                    # work on a non-default CUDA stream, so results are read
                    # before kernels finish without this; and the default
                    # scratch reservation is sized for indexed search.
                    resources.setDefaultNullStreamAllDevices()
                    resources.setTempMemory(64 * 1024 * 1024)
                    self._resources = resources
                inner = space.index.index
                space.gpu_clone = faiss.index_cpu_to_gpu(self._resources, 0, inner)
        except FALLIBLE as why:
            _logger.warning(
                "space %s serves from the CPU: GPU clone failed: %s: %s", space.spec.key, type(why).__name__, why
            )
            return None
        return space.gpu_clone

    def _files(self, key: str) -> tuple[pathlib.Path, pathlib.Path]:
        base = self._snapshots if self._snapshots is not None else pathlib.Path(".")
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", key)
        return base / f"{safe}.faiss", base / f"{safe}.json"

    def _faiss(self, gpu: bool | None = None):
        """Through the repo's loader, not a bare `import faiss`: a vendored
        CUDA build sits under vendor/faiss-gpu-win64 and needs its DLL
        directories registered before the import."""
        from vision.faiss_runtime import import_faiss

        return import_faiss(gpu=self._gpu_wanted if gpu is None else gpu)
