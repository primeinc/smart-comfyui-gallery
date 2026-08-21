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

import pathlib
import sqlite3

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
    by construction (knn on device + CPU range pass for overflow).

    The CUDA build is imported FIRST, deliberately: the process's faiss
    is decided by the first import (vision/faiss_runtime.py), and a
    gpu=False manager constructed before any gpu=True one would pin the
    whole process to the CPU wheel -- this test would then skip on a
    machine with two GPUs, which is exactly the silent degradation it
    exists to catch."""
    from vision.faiss_runtime import import_faiss

    import_faiss(gpu=True)
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
    from vision.faiss_runtime import import_faiss

    faiss = import_faiss(gpu=True)
    if not (hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0):
        pytest.skip("no GPU in this environment; the CPU-policy half of the parity ran")
    # A GPU exists, so the gpu=True manager MUST have answered on it --
    # a silent CPU answer here is the fallback theater this test bans.
    assert quick.served_by(spec.key) == "faiss-gpu"
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


SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def test_align_keeps_a_binary_space_content_exact(db, tmp_path):
    """A file's id is stable while its bytes -- and so its hash -- change.
    Align compares every held value, so the same id carrying a new hash
    is re-indexed instead of silently serving the old picture. And every
    align records the space's durable identity."""
    table = {1: 0, 2: 0b111}
    manager = IndexManager(tmp_path)
    similarity.align(db, manager, similarity.PHASH, [1, 2], lambda w: [table[v] for v in w], 5.0)
    assert pairs_of(manager, similarity.PHASH.key, 3) == {(1, 2), (2, 1)}

    table[2] = 0xFFFFFFFFFFFFFFF0
    similarity.align(db, manager, similarity.PHASH, [1, 2], lambda w: [table[v] for v in w], 6.0)
    assert pairs_of(manager, similarity.PHASH.key, 3) == set(), "the changed hash kept its old vector"
    assert pairs_of(manager, similarity.PHASH.key, 63) == {(1, 2), (2, 1)}

    key, producer, version, when = db.execute(
        "SELECT key, producer, producer_version, aligned_at FROM derived_similarity_space"
    ).fetchone()
    assert (key, producer) == ("perceptual.phash64", "imagehash.phash")
    assert version
    assert when == 6.0


def test_align_restores_a_float_space_without_rereading_embeddings(db, tmp_path):
    """The restore path's whole point: a warm boot must not read every
    embedding blob back just to prove the snapshot right. Ids never
    reuse (derived_face_instance is AUTOINCREMENT), so the id diff reads
    only genuinely new rows."""
    rng = np.random.default_rng(7)
    vectors = {n: rng.normal(size=8).astype(np.float32) for n in (10, 20, 30)}
    fetched: list[list[int]] = []

    def rows(wanted):
        fetched.append(list(wanted))
        return np.vstack([vectors[v] for v in wanted])

    spec = SpaceSpec("face.aligned", "float32", 8, "cosine", producer="fake", producer_version="1")
    first = IndexManager(tmp_path)
    similarity.align(db, first, spec, [10, 20], rows, 0.0)
    assert fetched == [[10, 20]]

    second = IndexManager(tmp_path)
    similarity.align(db, second, spec, [10, 20], rows, 1.0)
    assert fetched == [[10, 20]], "a valid snapshot re-read its embeddings"

    similarity.align(db, second, spec, [20, 30], rows, 2.0)
    assert fetched == [[10, 20], [30]], "a diff re-read embeddings the index already held"
    assert set(second.ids(spec.key).tolist()) == {20, 30}


def test_a_rolled_back_hash_never_reaches_the_live_index(db, tmp_path):
    """The runner rolls a failed item's writes back; an index that took
    the write anyway serves a hash the database never kept -- and a
    shutdown checkpoint makes the lie durable. So producers write rows
    only, and align digests what commits actually kept."""
    from db import derived, scan

    db.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'C:/lib','library',0)")
    folder = scan.mint(db, "folder", "lib")
    db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(?,1,NULL,'lib',0)", (folder,))
    file_id = scan.mint(db, "file", "p")
    db.execute(
        "INSERT INTO file(id,folder_id,name,kind,size,mtime,content_sha256,first_seen_at,last_seen_at)"
        " VALUES(?,?,?,'image',1,0,'aa',0,0)",
        (file_id, folder, "p.png"),
    )
    derived.record_hash(db, file_id, "aa", 0.0, phash64=0b1)
    db.commit()

    def committed(wanted):
        held = dict(db.execute("SELECT file_id, phash64 FROM derived_file_hash"))
        return [held[v] for v in wanted]

    manager = IndexManager(tmp_path)
    similarity.align(db, manager, similarity.PHASH, [file_id], committed, 1.0)
    db.commit()

    derived.record_hash(db, file_id, "bb", 2.0, phash64=0b11110000)
    db.rollback()
    similarity.align(db, manager, similarity.PHASH, [file_id], committed, 3.0)
    _labels, distances = manager.search(similarity.PHASH.key, [0b1], 1)
    assert int(distances[0][0]) == 0, "the index is ahead of the database it serves"

    derived.record_hash(db, file_id, "bb", 4.0, phash64=0b11110000)
    db.commit()
    similarity.align(db, manager, similarity.PHASH, [file_id], committed, 5.0)
    _labels, distances = manager.search(similarity.PHASH.key, [0b11110000], 1)
    assert int(distances[0][0]) == 0, "the committed replacement never reached the index"


def test_a_snapshot_from_another_process_restores_and_answers(tmp_path):
    """Process A checkpoints and dies; process B restores from its files
    and answers -- the boot the snapshot tier exists for, across a real
    process boundary."""
    import subprocess
    import sys

    spec = SpaceSpec("perceptual.phash64", "binary", 64, "hamming", producer="imagehash.phash", producer_version="x")
    fields = (spec.key, spec.representation, spec.dimensions, spec.metric, spec.producer, spec.producer_version)
    script = (
        "from vision.faiss_index import IndexManager, SpaceSpec\n"
        f"spec = SpaceSpec(*{fields!r})\n"
        f"manager = IndexManager({str(tmp_path)!r})\n"
        "manager.load(spec, [700, 3], [7, 5])\n"
        "manager.checkpoint(spec.key)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script],
        cwd=pathlib.Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr

    manager = IndexManager(tmp_path)
    assert manager.restore(spec), "process B refused process A's snapshot"
    assert (700, 3) in pairs_of(manager, spec.key, 2)

    obsolete = SpaceSpec(spec.key, spec.representation, spec.dimensions, spec.metric, spec.producer, "y")
    fresh = IndexManager(tmp_path)
    assert not fresh.restore(obsolete), "a snapshot from an obsolete producer version answered"


def test_search_answers_topk_by_stable_id():
    manager = IndexManager(gpu=False)
    rng = np.random.default_rng(9)
    vectors = rng.normal(size=(6, 8)).astype(np.float32)
    ids = [11, 22, 33, 44, 55, 66]
    manager.load(FACE, ids, vectors)
    labels, scores = manager.search(FACE.key, vectors[:2], 1)
    assert labels[:, 0].tolist() == [11, 22]
    assert scores[0][0] == pytest.approx(1.0, abs=1e-5)
