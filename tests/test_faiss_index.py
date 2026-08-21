"""The one FAISS layer: named resident spaces, stable ids, full lifecycle.

pHash grouping and face clustering each built their own FAISS consumer
once -- the runner packed hashes and read positional results, the old
db/similarity.py rebuilt a float index per call -- so every result was
keyed by insertion position and every new representation meant another
bespoke consumer. This layer is where indexes live, and its contract is
what the two paths could not share: results name SQLite ids, never
positions; spaces stay resident and mutate in place; snapshots restore
or are refused. The numpy oracle (db/similarity.numpy_graph) appears
here as the independent exact answer the engine is held to -- it is a
test instrument, not a backend.
"""

from __future__ import annotations

import numpy as np
import pytest

from db import similarity
from vision.faiss_index import IndexManager, SpaceSpec

PHASH = SpaceSpec(key="perceptual.phash64", representation="binary", dimensions=64, metric="hamming")
FACE = SpaceSpec(key="face.test", representation="float32", dimensions=8, metric="cosine")


def pairs_of(manager: IndexManager, key: str, radius) -> set[tuple[int, int]]:
    a, b, _ = similarity.pair_graph(manager, key, radius)
    return {(int(x), int(y)) for x, y in zip(a, b, strict=True)}


def test_binary_neighbours_come_back_as_the_ids_that_went_in():
    """Non-contiguous, unsorted ids -- the shape a real file table has.
    A positional index would answer 0, 1, 2 here; the contract is 700, 3,
    12, and getting this wrong groups the wrong pictures."""
    manager = IndexManager()
    twin = 0x0123456789ABCDEF
    manager.load(PHASH, [700, 3, 12], [twin, twin ^ 0b1, -1])
    a, b, distance = similarity.pair_graph(manager, PHASH.key, 4)
    pairs = {(int(x), int(y), int(d)) for x, y, d in zip(a, b, distance, strict=True)}
    assert (700, 3, 1) in pairs
    assert (3, 700, 1) in pairs
    assert all(12 not in (x, y) for x, y, _ in pairs), "the far hash joined a pair"


def test_the_radius_is_inclusive_and_self_pairs_are_absent():
    """`4` means "within 4 bits", the way the dupe_threshold setting is
    documented -- FAISS's strict `< radius` is this layer's problem. And
    every vector is 0 bits from itself; an edge saying so is noise every
    caller would have to strip."""
    manager = IndexManager()
    base = 0x00FF00FF00FF00FF
    manager.load(PHASH, [1, 2], [base, base ^ 0b1111])
    a, b, distance = similarity.pair_graph(manager, PHASH.key, 4)
    assert {(int(x), int(y)) for x, y in zip(a, b, strict=True)} == {(1, 2), (2, 1)}
    assert [int(d) for d in distance] == [4, 4]
    assert pairs_of(manager, PHASH.key, 3) == set(), "a pair past the radius survived"


def test_float_edges_match_the_exact_numpy_oracle_with_ids_translated():
    """The engine is held to an independent exact answer: every edge and
    weight the blocked numpy sweep finds, keyed by the caller's ids."""
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(20, 8)).astype(np.float32)
    ids = [n * 10 + 5 for n in range(20)]

    manager = IndexManager()
    manager.load(FACE, ids, vectors)
    a, b, weight = similarity.pair_graph(manager, FACE.key, 0.5)

    indptr, cols, weights = similarity.numpy_graph(vectors, 0.5)
    expected = set()
    for row in range(20):
        for at in range(int(indptr[row]), int(indptr[row + 1])):
            expected.add((ids[row], ids[int(cols[at])], round(float(weights[at]), 5)))
    assert {(int(x), int(y), round(float(w), 5)) for x, y, w in zip(a, b, weight, strict=True)} == expected


def test_cpu_and_gpu_policies_agree_on_the_edges():
    """Device policy is the manager's configuration. Whichever way it is
    configured, the answers are the same edges -- the GPU path is exact
    by construction (knn on device + CPU range pass for overflow)."""
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(30, 16)).astype(np.float32)
    ids = list(range(0, 300, 10))
    spec = SpaceSpec("face.device", "float32", 16, "cosine")

    plain = IndexManager(gpu=False)
    plain.load(spec, ids, vectors)
    told = pairs_of(plain, spec.key, 0.4)
    assert plain.served_by(spec.key) == "faiss-cpu"

    quick = IndexManager(gpu=True)
    quick.load(spec, ids, vectors)
    assert pairs_of(quick, spec.key, 0.4) == told
    if quick.served_by(spec.key) == "faiss-gpu":
        assert quick._spaces[spec.key].gpu_clone is not None, "the device clone was not kept resident"
        assert pairs_of(quick, spec.key, 0.4) == told, "the resident clone answered differently"


def test_a_pair_sitting_exactly_on_the_threshold_is_kept():
    """FAISS range_search keeps strictly-above for inner product where
    this application's thresholds are at-or-above. The nextafter step
    absorbs that, and a pair at similarity exactly 1.0 against threshold
    1.0 is the sharpest case."""
    vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    manager = IndexManager(gpu=False)
    manager.load(SpaceSpec("face.exact", "float32", 2, "cosine"), [5, 6, 7], vectors)
    assert pairs_of(manager, "face.exact", 1.0) == {(5, 6), (6, 5)}


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
    assert 1 not in {v for pair in pairs_of(manager, PHASH.key, 63) for v in pair}
    with pytest.raises(ValueError, match="spec"):
        manager.load(SpaceSpec(PHASH.key, "float32", 8, "cosine"), [3], np.zeros((1, 8), dtype=np.float32))


def test_an_invalidated_space_stops_answering_and_loses_its_snapshot(tmp_path):
    manager = IndexManager(tmp_path)
    manager.load(PHASH, [1, 2], [0, 1])
    written = manager.checkpoint(PHASH.key)
    manager.invalidate(PHASH.key)
    with pytest.raises(KeyError):
        manager.range(PHASH.key, 4)
    assert not written.exists(), "an invalidated space's snapshot survived to poison a boot"


# --- lifecycle: mutate, persist, restore -----------------------------------


def test_added_rows_answer_immediately_and_removed_rows_stop():
    manager = IndexManager()
    manager.load(PHASH, [1], [0])
    manager.add(PHASH.key, [2], [0b11])
    assert pairs_of(manager, PHASH.key, 4) == {(1, 2), (2, 1)}
    manager.remove(PHASH.key, [1])
    assert pairs_of(manager, PHASH.key, 63) == set(), "a removed row is still answering"

    rng = np.random.default_rng(3)
    base = rng.normal(size=(2, 8)).astype(np.float32)
    manager.load(FACE, [10, 20], base)
    manager.add(FACE.key, [30], base[:1] + 0.001)
    assert (10, 30) in pairs_of(manager, FACE.key, 0.99)


def test_duplicate_and_unknown_ids_are_refused():
    manager = IndexManager()
    with pytest.raises(ValueError, match="once"):
        manager.load(PHASH, [1, 1], [0, 1])
    manager.load(PHASH, [1, 2], [0, 1])
    with pytest.raises(ValueError, match="already"):
        manager.add(PHASH.key, [2], [5])
    with pytest.raises(ValueError, match="not in"):
        manager.remove(PHASH.key, [9])


def test_a_checkpoint_restores_into_a_fresh_manager(tmp_path):
    """The snapshot tier: a fresh process restores the space from disk
    and answers identically, without touching the rows that built it."""
    twin = 0x0123456789ABCDEF
    first = IndexManager(tmp_path)
    first.load(PHASH, [700, 3, 12], [twin, twin ^ 0b1, -1])
    first.checkpoint(PHASH.key)

    second = IndexManager(tmp_path)
    assert second.restore(PHASH)
    assert (700, 3) in pairs_of(second, PHASH.key, 4)

    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(20, 8)).astype(np.float32)
    ids = [n * 10 + 5 for n in range(20)]
    first.load(FACE, ids, vectors)
    told = pairs_of(first, FACE.key, 0.5)
    first.checkpoint(FACE.key)
    assert second.restore(FACE)
    assert pairs_of(second, FACE.key, 0.5) == told


def test_a_wrong_or_broken_snapshot_refuses_to_restore(tmp_path):
    first = IndexManager(tmp_path)
    first.load(PHASH, [1, 2], [0, 1])
    written = first.checkpoint(PHASH.key)
    assert written is not None

    second = IndexManager(tmp_path)
    fatter = SpaceSpec(PHASH.key, "binary", 128, "hamming")
    assert not second.restore(fatter), "another spec's vectors answered"
    written.write_bytes(b"not a faiss file")
    assert not second.restore(PHASH), "a corrupt snapshot answered"
    assert not second.has(PHASH.key)


def test_checkpoint_all_writes_only_what_changed(tmp_path):
    manager = IndexManager(tmp_path)
    manager.load(PHASH, [1, 2], [0, 1])
    assert len(manager.checkpoint_all()) == 1
    assert manager.checkpoint_all() == [], "a clean space was re-written"
    manager.add(PHASH.key, [3], [7])
    assert len(manager.checkpoint_all()) == 1


# --- align: the one path a consumer keeps a space current by ---------------


def test_align_builds_restores_and_diffs_without_rereading(tmp_path):
    fetched: list[list[int]] = []

    def rows(wanted):
        fetched.append(list(wanted))
        return [{1: 0, 2: 0b111, 3: 0b101}[v] for v in wanted]

    first = IndexManager(tmp_path)
    similarity.align(first, PHASH, [1, 2], rows)
    assert fetched == [[1, 2]]
    assert first.count(PHASH.key) == 2

    similarity.align(first, PHASH, [1, 2], rows)
    assert fetched == [[1, 2]], "an aligned space re-read its rows"

    second = IndexManager(tmp_path)
    similarity.align(second, PHASH, [1, 2], rows)
    assert fetched == [[1, 2]], "a valid snapshot re-read its rows"

    similarity.align(second, PHASH, [2, 3], rows)
    assert fetched == [[1, 2], [3]], "a diff re-read rows the index already held"
    assert set(second.ids(PHASH.key).tolist()) == {2, 3}


def test_search_answers_topk_by_stable_id():
    manager = IndexManager(gpu=False)
    rng = np.random.default_rng(9)
    vectors = rng.normal(size=(6, 8)).astype(np.float32)
    ids = [11, 22, 33, 44, 55, 66]
    manager.load(FACE, ids, vectors)
    labels, scores = manager.search(FACE.key, vectors[:2], 1)
    assert labels[:, 0].tolist() == [11, 22]
    assert scores[0][0] == pytest.approx(1.0, abs=1e-5)
