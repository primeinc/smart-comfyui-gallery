"""The one FAISS layer: stable ids in, neighbours by those ids out.

pHash grouping and face clustering each built their own FAISS consumer --
the runner packed hashes and read positional results, db/similarity.py
owned the float path -- so every result was keyed by insertion position
and every new representation meant another bespoke consumer. This layer
is where indexes are built, and its contract is the invariant the two
paths could not share: results name SQLite ids, never positions.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision.faiss_index import IndexManager, SpaceSpec

PHASH = SpaceSpec(key="perceptual.phash64", representation="binary", dimensions=64, metric="hamming")
FACE = SpaceSpec(key="face.test", representation="float32", dimensions=8, metric="cosine")


def test_binary_neighbours_come_back_as_the_ids_that_went_in():
    """Non-contiguous, unsorted ids -- the shape a real file table has.
    A positional index would answer 0, 1, 2 here; the contract is 700, 3,
    12, and getting this wrong groups the wrong pictures."""
    manager = IndexManager()
    twin = 0x0123456789ABCDEF
    manager.load(PHASH, [700, 3, 12], [twin, twin ^ 0b1, -1])
    a, b, distance = manager.graph(PHASH.key, 4)
    pairs = {(int(x), int(y), int(d)) for x, y, d in zip(a, b, distance, strict=True)}
    assert (700, 3, 1) in pairs
    assert (3, 700, 1) in pairs
    assert all({x, y} != {12} and 12 not in (x, y) for x, y, _ in pairs), "the far hash joined a pair"


def test_the_radius_is_inclusive_and_self_pairs_are_absent():
    """`4` means "within 4 bits", the way the dupe_threshold setting is
    documented -- FAISS's strict `< radius` is this layer's problem. And
    every vector is 0 bits from itself; an edge saying so is noise every
    caller would have to strip."""
    manager = IndexManager()
    base = 0x00FF00FF00FF00FF
    manager.load(PHASH, [1, 2], [base, base ^ 0b1111])
    a, b, distance = manager.graph(PHASH.key, 4)
    assert {(int(x), int(y)) for x, y in zip(a, b, strict=True)} == {(1, 2), (2, 1)}
    assert [int(d) for d in distance] == [4, 4]
    a, b, _ = manager.graph(PHASH.key, 3)
    assert len(a) == 0 == len(b), "a pair past the radius survived"


def test_float_edges_match_the_similarity_engine_with_ids_translated():
    """The float path is db/similarity.py's engine underneath -- same
    edges, same weights -- with positions translated to the caller's ids.
    Parity is the refactor's proof: faces moving onto this layer must not
    change who clusters with whom."""
    from db import similarity

    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(20, 8)).astype(np.float32)
    ids = [n * 10 + 5 for n in range(20)]

    manager = IndexManager()
    manager.load(FACE, ids, vectors)
    a, b, weight = manager.graph(FACE.key, 0.5)

    (indptr, cols, weights), _backend = similarity.graph(vectors, 0.5)
    expected = set()
    for row in range(20):
        for at in range(int(indptr[row]), int(indptr[row + 1])):
            expected.add((ids[row], ids[int(cols[at])], round(float(weights[at]), 5)))
    assert {(int(x), int(y), round(float(w), 5)) for x, y, w in zip(a, b, weight, strict=True)} == expected


def test_a_space_refuses_what_it_did_not_declare():
    manager = IndexManager()
    with pytest.raises(ValueError, match="dimensions"):
        manager.load(FACE, [1], np.zeros((1, 9), dtype=np.float32))
    with pytest.raises(ValueError, match="metric"):
        manager.load(SpaceSpec("bad", "binary", 64, "cosine"), [1], [0])
    with pytest.raises(ValueError, match="metric"):
        manager.load(SpaceSpec("worse", "float32", 8, "hamming"), [1], np.zeros((1, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="representation"):
        manager.load(SpaceSpec("weird", "float64", 8, "cosine"), [1], np.zeros((1, 8)))
    with pytest.raises(ValueError, match="ids"):
        manager.load(PHASH, [1, 2], [0])


def test_one_key_cannot_quietly_change_meaning():
    """Reloading a key with the same spec replaces its rows; reloading it
    as a different spec is refused -- a key whose meaning drifts is how
    incompatible vectors end up in one index."""
    manager = IndexManager()
    manager.load(PHASH, [1], [0])
    manager.load(PHASH, [2], [5])
    a, b, _ = manager.graph(PHASH.key, 64 - 1)
    assert 1 not in set(map(int, a)) | set(map(int, b)), "the replaced row is still answering"
    with pytest.raises(ValueError, match="spec"):
        manager.load(SpaceSpec(PHASH.key, "float32", 8, "cosine"), [3], np.zeros((1, 8), dtype=np.float32))


def test_an_invalidated_space_stops_answering():
    manager = IndexManager()
    manager.load(PHASH, [1, 2], [0, 1])
    manager.invalidate(PHASH.key)
    with pytest.raises(KeyError):
        manager.graph(PHASH.key, 4)
