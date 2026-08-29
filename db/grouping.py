"""Ways of turning faces into people, and a way of combining them.

There is no settled answer to "which faces are one person". A threshold that
welds strangers together on one library splits one person into four on
another, and the only way to find out is to run more than one and look at
both -- which is why a clustering RUN is (embedder, version, method,
threshold) in the schema, and why a method is a name you add to rather than
a branch inside one function.

Every method here uses something the project already has. FAISS is already a
dependency, with a vendored CUDA build, and it ships k-means; the graph
methods need only the CSR graph `db.similarity` produces. Nothing here
reaches for a new library to say something a present one can say.

Finding the pairs and deciding what to do with them are kept apart on
purpose: the first is the expensive part and belongs on a GPU, the second is
cheap and belongs somewhere a person can read it.
"""

from __future__ import annotations


def chinese_whispers(graph, vectors=None, *, sweeps: int = 20, **_):
    """Label propagation. Each node adopts what its neighbours agree on most.

    From the canonical implementation rather than a description of it: a
    node's neighbours vote with their edge WEIGHTS summed per label, not
    with a count (davisking/dlib@f28ef50 dlib/clustering/chinese_whispers.h:
    48-53), and the winner is picked with a strict `>` over a label-ordered
    map, so a tie goes to the lowest label id (:57-66).

    dlib picks nodes at random for `n * num_iterations` steps (:42-45). This
    sweeps in index order and stops when a sweep changes nothing, making the
    result a pure function of the graph -- a library that reclusters twice
    gets the same people both times.
    """
    indptr, cols, weights = graph
    n = len(indptr) - 1
    labels = list(range(n))
    for _ in range(sweeps):
        moved = 0
        for node in range(n):
            start, end = int(indptr[node]), int(indptr[node + 1])
            if start == end:
                continue
            tally: dict[int, float] = {}
            for edge in range(start, end):
                label = labels[int(cols[edge])]
                tally[label] = tally.get(label, 0.0) + float(weights[edge])
            best = min(tally, key=lambda label: (-tally[label], label))
            if best != labels[node]:
                labels[node] = best
                moved += 1
        if not moved:
            break
    return labels


def connected_components(graph, vectors=None, **_):
    """Single linkage: everything reachable is one group.

    Kept as a real option and labelled honestly rather than left out. It is
    the obvious method and it chains -- the previous pipeline documented
    that "transitive chaining merges dense look-alike sets into one
    cluster" (git history) -- which over 834 real faces made one group of
    123 spanning 53 different
    photographs. That is not a person, it is a chain of people who each
    slightly resemble the next.

    It is here so that can be re-measured on a real library instead of taken
    on trust, and because at a tight threshold, where chaining cannot reach,
    it is both faster and exactly right.
    """
    indptr, cols, _ = graph
    n = len(indptr) - 1
    parent = list(range(n))

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for node in range(n):
        for edge in range(int(indptr[node]), int(indptr[node + 1])):
            a, b = root(node), root(int(cols[edge]))
            if a != b:
                parent[max(a, b)] = min(a, b)
    return [root(node) for node in range(n)]


def spherical_kmeans(
    graph, vectors=None, *, people: int | None = None, iterations: int = 20, redo: int = 1, gpu: bool = True, **_
):
    """FAISS k-means on the unit sphere.

    Nothing new is installed for this: FAISS is already here, ships k-means,
    and runs it on every GPU in the machine with `gpu=True`
    (facebookresearch/faiss.wiki@1354fdb Faiss-building-blocks:-clustering,-PCA,-quantization.md:20).

    `spherical=True` L2-normalises the centroids after each iteration, which
    is the version that means anything for face embeddings -- the vectors
    are unit length and compared by angle, so a centroid allowed to drift
    off the sphere is not a direction any face points in.

    It differs from the graph methods in what it needs from you: a number of
    people, decided before it runs. That is a real disadvantage on a library
    nobody has counted, and a real advantage when you already know -- and
    the reason to keep both rather than pick one.
    """
    import numpy as np

    from . import similarity

    unit = similarity.normalise(vectors)
    n = unit.shape[0]
    if people is None:
        # No count given: take the one the graph implies, so k-means answers
        # the same question the graph methods were asked rather than a
        # different one nobody chose.
        #
        # Groups of two or more, NOT every distinct label. A face nothing
        # matched is a singleton, and counting those made k almost equal to
        # the number of faces -- 732 centroids for 834 points, which FAISS
        # warns about because it is not clustering, it is renaming.
        from collections import Counter

        sizes = Counter(connected_components(graph))
        people = max(1, sum(1 for count in sizes.values() if count >= 2))
    people = max(1, min(people, n))

    faiss = _faiss()
    # The dataset being clustered IS the training set, which upstream calls
    # out as the case where its training-size guards do not apply: below
    # `min_points_per_centroid * k` it warns ("This may still be ok if the
    # dataset to index is as small as the training set"), and above
    # `max_points_per_centroid * k` -- 256 by default -- it silently
    # subsamples the training data, which here would mean clustering on part
    # of the library. Both knobs are ClusteringParameters fields settable
    # through this constructor. (facebookresearch/faiss.wiki@1354fdb FAQ.md,
    # "Can I ignore WARNING clustering XXX points to YYY centroids?";
    # Faiss-building-blocks page, "Additional options".)
    kmeans = faiss.Kmeans(
        int(unit.shape[1]),
        people,
        niter=iterations,
        nredo=redo,
        spherical=True,
        gpu=gpu and faiss.get_num_gpus() > 0,
        min_points_per_centroid=1,
        max_points_per_centroid=10_000_000,
    )
    kmeans.train(unit)
    _, assignment = kmeans.index.search(unit, 1)
    return [int(v) for v in np.asarray(assignment).reshape(-1)]


def consensus(
    graph, vectors=None, *, methods=("chinese-whispers", "connected-components"), agree: int | None = None, **options
):
    """Keep only what several methods agree on.

    Two clusterings that disagree about a pair are telling you the pair is
    marginal. Running them and keeping the edges a majority put together
    drops exactly those, which is the point: the cost of a wrong merge in a
    face library is somebody else's photograph on your page, and the cost of
    a wrong split is a person appearing twice -- one of those a human can
    fix in a click and the other they have to notice first.

    Co-association over the graph's EDGES rather than over all pairs. Every
    pair that could possibly be joined is already an edge, so this is linear
    in edges and not quadratic in faces.
    """
    indptr, cols, weights = graph
    n = len(indptr) - 1
    votes = [group(graph, vectors, name, **options) for name in methods]
    needed = agree if agree is not None else len(votes) // 2 + 1

    import numpy as np

    keep_rows, keep_cols, keep_weights = [], [], []
    for node in range(n):
        for edge in range(int(indptr[node]), int(indptr[node + 1])):
            other = int(cols[edge])
            together = sum(1 for labels in votes if labels[node] == labels[other])
            if together >= needed:
                keep_rows.append(node)
                keep_cols.append(other)
                keep_weights.append(float(weights[edge]))
    agreed = _csr(n, keep_rows, keep_cols, keep_weights, np)
    return chinese_whispers(agreed)


def _csr(n, rows, cols, weights, np):
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    order = np.argsort(rows, kind="stable")
    rows, cols, weights = rows[order], cols[order], weights[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    return indptr, cols, weights


def _faiss():
    from vision.faiss_runtime import import_faiss

    return import_faiss()


#: The methods a run may name. A new one is an entry here and a token in the
#: database, not a schema change: `derived_face_run.method` is deliberately
#: not a CHECK list.
METHODS = {
    "chinese-whispers": chinese_whispers,
    "connected-components": connected_components,
    "spherical-kmeans": spherical_kmeans,
    "consensus": consensus,
}


def group(graph, vectors, method: str, **options):
    """Labels for one graph by one named method."""
    if method not in METHODS:
        raise ValueError(f"{method!r} is not one of {', '.join(sorted(METHODS))}")
    return METHODS[method](graph, vectors, **options)
