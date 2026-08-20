"""Which vectors are near which -- the expensive half of clustering.

Grouping faces is cheap; finding the pairs to group is not. It is every
vector against every other, and a nested Python loop over it is correct and
useless: measured here, 557 faces took 3.8 seconds, so 100,000 faces would
take about a day. FAISS does the same work in two seconds.

**GPU FAISS has no range search, and that is not a problem.** Asking a GPU
index for one raises, which reads like a missing feature and is not:
`faiss.contrib.exhaustive_search.range_search_gpu` says outright that "GPU
does not support range search, so we emulate it with knn search + fallback
to CPU index" (refs/facebookresearch/faiss/contrib/exhaustive_search.py:
60-67), and does it exactly -- k nearest on the GPU, then a CPU range search
for any query whose k-th neighbour still cleared the threshold, the two
merged by `CombinerRangeKNN` (:110-116). The edge set is identical, which is
measured rather than assumed: at 100,000 vectors both paths returned the
same count, the GPU in 2.03s against the CPU's 19.80s.

Two things the GPU path needs that nothing would guess, both from the wiki's
brute-force page: FAISS orders its work on a non-default CUDA stream, so
`setDefaultNullStreamAllDevices()` is required or results are read before
the kernels finish; and its default scratch reservation is sized for indexed
search rather than brute force, so `setTempMemory` is turned down. Getting
the first wrong is occasionally-wrong numbers, which is worse than slow.

Whichever backend ran comes back with the result. A measurement that cannot
say what produced it cannot be compared with another, and "faiss is GPU
here" and "faiss is a CPU wheel here" are both true on different machines --
including two Python environments on the same one.
"""

from __future__ import annotations

import os

#: Neighbours asked of the GPU before the CPU is consulted for a query that
#: had more. Upstream's own default. A GPU index caps k at 2048 for every
#: index type and selection cost climbs above about 512
#: (refs/facebookresearch/faiss.wiki, Faiss-on-the-GPU.md, "Limitations"),
#: so this already sits near the useful ceiling.
GPU_K = 1024

#: Rows per block for the numpy path. An n x n float32 matrix at 100,000
#: vectors is 40 GB; a 2,048-row block is 800 MB.
BLOCK = 2_048

#: Name a backend to force it. A named backend that cannot run raises rather
#: than falling back -- a benchmark that quietly ran on something else is a
#: measurement of the wrong thing.
ENV_VAR = "SG_SIMILARITY_BACKEND"


def normalise(vectors):
    """Unit-length rows, so an inner product IS the cosine."""
    import numpy as np

    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    lengths = np.linalg.norm(matrix, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return np.ascontiguousarray(matrix / lengths, dtype=np.float32)


def _faiss():
    """The faiss this project should use.

    Through the repo's loader, not a bare `import faiss`: a vendored CUDA
    build sits under vendor/faiss-gpu-win64 and needs its DLL directories
    registered before the import. Importing the module directly gets the pip
    wheel, which here is CPU-only -- so a probe written the obvious way
    reports "no GPU" on a machine with two of them, which is exactly what
    this first did.
    """
    from vision.faiss_runtime import import_faiss

    return import_faiss()


def _inclusive(threshold: float) -> float:
    """One float32 step below the threshold.

    FAISS keeps `similarity > radius` for an inner-product metric where the
    numpy path here keeps `>= threshold`, so a vector sitting exactly on the
    line is dropped by one and kept by the other -- the same library, the
    same number, two different sets of people. `nextafter` returns "the next
    floating-point value after x1 towards x2"
    (refs/numpy/numpy/numpy/_core/code_generators/ufunc_docstrings.py:
    3914-3930), so stepping the radius down by one representable value makes
    the strict comparison mean the inclusive one.
    """
    import numpy as np

    return float(np.nextafter(np.float32(threshold), np.float32("-inf")))


def _gpu(unit, threshold):
    import numpy as np

    faiss = _faiss()
    if not hasattr(faiss, "StandardGpuResources") or faiss.get_num_gpus() < 1:
        raise RuntimeError("this faiss build has no GPU support")
    from faiss.contrib.exhaustive_search import range_search_gpu

    resources = faiss.StandardGpuResources()
    resources.setDefaultNullStreamAllDevices()
    resources.setTempMemory(64 * 1024 * 1024)
    index = faiss.index_cpu_to_gpu(resources, 0, faiss.IndexFlatIP(int(unit.shape[1])))
    index.add(unit)
    # `unit` as the CPU fallback rather than None: contrib builds a flat
    # index from it for the queries whose k-th neighbour still cleared the
    # threshold, and that is what makes this exact instead of a top-k
    # approximation. Passing None caps every face at GPU_K neighbours.
    limits, sims, ids = range_search_gpu(unit, _inclusive(threshold), index, unit, gpu_k=GPU_K)
    return _from_ranges(unit.shape[0], limits, sims, ids, np)


def _cpu(unit, threshold):
    import numpy as np

    faiss = _faiss()
    index = faiss.IndexFlatIP(int(unit.shape[1]))
    index.add(unit)
    limits, sims, ids = index.range_search(unit, _inclusive(threshold))
    return _from_ranges(unit.shape[0], limits, sims, ids, np)


def _numpy(unit, threshold):
    """Always available, memory bounded at one block by n."""
    import numpy as np

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


def _from_ranges(n, limits, sims, ids, np):
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(limits).astype(np.int64))
    ids = ids.astype(np.int64)
    keep = ids != rows
    return _csr(n, rows[keep], ids[keep], sims[keep], np)


def _csr(n, rows, cols, weights, np):
    """(indptr, cols, weights): row i's neighbours are cols[indptr[i]:indptr[i+1]]."""
    order = np.argsort(rows, kind="stable")
    rows, cols, weights = rows[order], cols[order], weights[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    return indptr, cols, weights.astype(np.float32)


#: In the order they are tried. Every one is reached by asking the machine,
#: never by assuming what is installed.
BACKENDS = (("faiss-gpu", _gpu), ("faiss-cpu", _cpu), ("numpy", _numpy))

#: The ways a backend fails to exist here: no module, a DLL that will not
#: load, an API the build lacks, no GPU, a SWIG-level refusal. Named so the
#: auto path catches what "not installed" actually raises and nothing more --
#: a genuine bug inside a backend propagates instead of reading as absence.
FALLIBLE = (ImportError, OSError, RuntimeError, AttributeError, ValueError, TypeError)


def graph(vectors, threshold: float, *, backend: str | None = None):
    """Every pair at or above `threshold`, and the name of what computed it."""
    import numpy as np

    unit = normalise(vectors)
    if unit.shape[0] == 0:
        return (np.zeros(1, "int64"), np.zeros(0, "int64"), np.zeros(0, "float32")), "numpy"

    wanted = (backend or os.environ.get(ENV_VAR, "auto")).strip().lower()
    known = dict(BACKENDS)
    if wanted != "auto":
        if wanted not in known:
            raise ValueError(f"{wanted!r} is not one of {', '.join(known)}")
        return known[wanted](unit, threshold), wanted

    refused: dict[str, str] = {}
    for name, build in BACKENDS:
        try:
            result = build(unit, threshold), name
        except FALLIBLE as why:
            refused[name] = f"{type(why).__name__}: {why}"
            continue
        if refused and name == "numpy":
            # FAISS is the engine this library is built around; numpy is the
            # correctness fallback. Landing here means every faiss path
            # failed, and doing that silently is how a machine with two GPUs
            # ran a day-long numpy loop with nobody told. The work still
            # happens -- degraded and SAID, never degraded and quiet.
            import warnings

            warnings.warn(
                "similarity graph fell back to numpy; faiss unavailable: "
                + "; ".join(f"{k} ({v})" for k, v in refused.items()),
                RuntimeWarning,
                stacklevel=2,
            )
        return result
    raise RuntimeError(
        "no similarity backend could run, including numpy: " + "; ".join(f"{k} ({v})" for k, v in refused.items())
    )


def available() -> list[str]:
    """Which backends this machine can actually use.

    For saying so in a report rather than guessing. Two Python environments
    on one box answer differently, and only the environment can say.
    """
    import numpy as np

    probe = normalise(np.eye(4, 8, dtype=np.float32))
    return [name for name, build in BACKENDS if _runs(build, probe)]


def _runs(build, probe) -> bool:
    try:
        build(probe, 0.5)
    except FALLIBLE:
        return False
    return True
