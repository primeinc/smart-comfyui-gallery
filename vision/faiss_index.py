"""Similarity search over stable SQLite ids -- the one place indexes live.

Before this, every FAISS consumer was bespoke: the dupes job packed
hashes and read positional results in db/runner.py while faces went
through db/similarity.py, and each new representation would have meant a
third copy. Consumers now hand this manager the rows they own -- ids and
representations -- and get neighbours back keyed by those same ids. What
a vector MEANS is the `SpaceSpec`; how the search executes is this
module's business and nobody else's.

Binary spaces ride `IndexBinaryIDMap(IndexBinaryFlat)` so FAISS itself
translates results to the stored ids (facebookresearch/faiss@v1.15.0
faiss/IndexIDMap.h:68-89 -- `range_search` is overridden on the
template, and `add` without ids is a refusal). Float spaces run on
db/similarity.py's engine -- it carries the GPU-exact range search and
the backend fallback story -- and this layer translates its positional
CSR to ids: the contrib GPU helper works on raw indexes, so the
translation is done here rather than by wrapping what it cannot wrap.

Radius and threshold are INCLUSIVE, the way the settings that feed them
are documented; FAISS's strict comparisons are absorbed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: How a binary space's execution is named in provenance columns: binary
#: flat search is CPU in every faiss build -- there is no GPU path for
#: IndexBinaryFlat -- whichever package the loader imported.
BINARY_BACKEND = "faiss-cpu"


@dataclass(frozen=True)
class SpaceSpec:
    """What the vectors in one space mean -- never how they are searched."""

    key: str
    representation: str  # 'binary' | 'float32'
    dimensions: int  # bits for binary, floats for float32
    metric: str  # 'hamming' for binary, 'cosine' for float32


def _signed_to_packed(values, bits: int):
    """Schema-form signed integers to the uint8 rows FAISS wants.

    SQLite INTEGER is signed 64-bit, so the stored hash is the unsigned
    value folded into the signed range; big-endian byte order, though any
    order works as long as every row uses the same one -- XOR popcount
    does not care."""
    import numpy as np

    unsigned = np.array([v & 0xFFFFFFFFFFFFFFFF for v in values], dtype=np.uint64)
    return np.ascontiguousarray(unsigned.astype(">u8").view(np.uint8).reshape(-1, bits // 8))


class IndexManager:
    """Per-space indexes, replaced wholesale when a space reloads.

    Indexes are disposable caches over rows the caller owns in SQLite;
    dropping every one of them changes nothing authoritative.
    """

    def __init__(self):
        self._spaces: dict[str, tuple[SpaceSpec, Any, Any]] = {}

    def load(self, spec: SpaceSpec, ids, vectors, *, gpu: bool = True) -> None:
        """Replace `spec.key`'s rows with these ids and representations."""
        import numpy as np

        if spec.representation not in ("binary", "float32"):
            raise ValueError(f"unknown representation {spec.representation!r}")
        wants = {"binary": "hamming", "float32": "cosine"}[spec.representation]
        if spec.metric != wants:
            raise ValueError(f"{spec.representation} spaces take the {wants} metric, not {spec.metric!r}")
        held = self._spaces.get(spec.key)
        if held is not None and held[0] != spec:
            raise ValueError(f"{spec.key!r} is already loaded under a different spec")

        keys = np.asarray(list(ids), dtype=np.int64)

        def counted(rows: int) -> None:
            if keys.shape[0] != rows:
                raise ValueError(f"{keys.shape[0]} ids for {rows} rows")

        if spec.representation == "binary":
            if spec.dimensions % 8:
                raise ValueError(f"binary dimensions must be whole bytes, not {spec.dimensions}")
            packed = _signed_to_packed(vectors, spec.dimensions)
            counted(packed.shape[0])
            from vision.faiss_runtime import import_faiss

            faiss = import_faiss(gpu=gpu)
            index = faiss.IndexBinaryIDMap(faiss.IndexBinaryFlat(spec.dimensions))
            if keys.shape[0]:
                index.add_with_ids(packed, keys)
            self._spaces[spec.key] = (spec, keys, (index, packed))
        else:
            floats = np.ascontiguousarray(vectors, dtype=np.float32)
            if floats.ndim != 2 or floats.shape[1] != spec.dimensions:
                raise ValueError(f"{spec.key} declares dimensions {spec.dimensions}, rows have {floats.shape}")
            counted(floats.shape[0])
            # The float engine builds per call inside db/similarity.py --
            # what is cached here is the space's rows, which is what makes
            # a reload a replacement instead of an accumulation.
            self._spaces[spec.key] = (spec, keys, floats)

    def invalidate(self, key: str) -> None:
        self._spaces.pop(key, None)

    def graph(self, key: str, radius, *, backend: str | None = None, gpu: bool = True):
        """Every pair within `radius` (inclusive), as (ids, ids, distances).

        Self-pairs are dropped; both directions of a pair are present, the
        shape union-find and label propagation both consume. For float
        spaces `radius` is the cosine similarity floor and the third array
        is similarity; for binary it is hamming bits and distance.
        """
        import numpy as np

        spec, keys, held = self._spaces[key]
        if spec.representation == "binary":
            index, data = held
            if keys.shape[0] < 2:
                empty = np.zeros(0, dtype=np.int64)
                return empty, empty, np.zeros(0, dtype=np.int32)
            # range_search keeps distance < radius (faiss/IndexBinary.h:
            # "only distances < radius (strict comparison)"), so +1 makes
            # the argument inclusive. Labels are the stored ids already.
            lims, distances, neighbours = index.range_search(data, int(radius) + 1)
            a = np.repeat(keys, np.diff(lims).astype(np.int64))
            b = neighbours.astype(np.int64)
            keep = a != b
            return a[keep], b[keep], distances[keep]

        from db import similarity

        indptr, cols, weights = similarity.graph(held, float(radius), backend=backend, gpu=gpu)[0]
        a = np.repeat(keys, np.diff(indptr).astype(np.int64))
        return a, keys[cols], weights
