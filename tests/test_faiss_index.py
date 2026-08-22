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

import numpy as np
import pytest

from db import similarity
from tests.staging import fresh_schema
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
    assert written is not None
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
    return fresh_schema()


def test_align_keeps_a_binary_space_content_exact(db, tmp_path):
    """A file's id is stable while its bytes -- and so its hash -- change.
    Align compares every held value, so the same id carrying a new hash
    is re-indexed instead of silently serving the old picture. And every
    align records the space's durable identity."""
    table = {1: 0, 2: 0b111}
    manager = IndexManager(tmp_path)
    key = similarity.align(db, manager, similarity.PHASH, [1, 2], lambda w: [table[v] for v in w], 5.0)
    assert pairs_of(manager, key, 3) == {(1, 2), (2, 1)}

    table[2] = 0xFFFFFFFFFFFFFFF0
    assert key == similarity.align(db, manager, similarity.PHASH, [1, 2], lambda w: [table[v] for v in w], 6.0)
    assert pairs_of(manager, key, 3) == set(), "the changed hash kept its old vector"
    assert pairs_of(manager, key, 63) == {(1, 2), (2, 1)}

    sid, spec_key, producer, version, preprocess = db.execute(
        "SELECT id, key, producer, producer_version, preprocess FROM similarity_space"
    ).fetchone()
    assert key == f"{spec_key}@{sid}"
    told = ("perceptual.phash64", "imagehash.phash", "smartgallery.perceptual-frame")
    assert (spec_key, producer, preprocess) == told
    assert version


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
    key = similarity.align(db, first, spec, [10, 20], rows, 0.0)
    assert fetched == [[10, 20]]

    second = IndexManager(tmp_path)
    assert key == similarity.align(db, second, spec, [10, 20], rows, 1.0)
    assert fetched == [[10, 20]], "a valid snapshot re-read its embeddings"

    similarity.align(db, second, spec, [20, 30], rows, 2.0)
    assert fetched == [[10, 20], [30]], "a diff re-read embeddings the index already held"
    assert set(second.ids(key).tolist()) == {20, 30}


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
        sid = similarity.space_id(db, similarity.PHASH, 0.0)
        held = dict(db.execute("SELECT file_id, value FROM derived_file_hash WHERE space_id = ?", (sid,)))
        return [held[v] for v in wanted]

    manager = IndexManager(tmp_path)
    key = similarity.align(db, manager, similarity.PHASH, [file_id], committed, 1.0)
    db.commit()

    derived.record_hash(db, file_id, "bb", 2.0, phash64=0b11110000)
    db.rollback()
    similarity.discard_pending(db)
    assert key == similarity.align(db, manager, similarity.PHASH, [file_id], committed, 3.0)
    _labels, distances = manager.search(key, [0b1], 1)
    assert int(distances[0][0]) == 0, "the index is ahead of the database it serves"

    derived.record_hash(db, file_id, "bb", 4.0, phash64=0b11110000)
    db.commit()
    similarity.apply_pending(db, manager)
    _labels, distances = manager.search(key, [0b11110000], 1)
    assert int(distances[0][0]) == 0, "the committed replacement never reached the index"


def test_a_snapshot_from_another_manager_restores_and_answers(tmp_path):
    """Manager A checkpoints and is gone; manager B restores from the
    files alone and answers -- the boot the snapshot tier exists for.
    The boundary is the files on disk: nothing of A survives in memory."""
    spec = SpaceSpec("perceptual.phash64", "binary", 64, "hamming", producer="imagehash.phash", producer_version="x")
    first = IndexManager(tmp_path)
    first.load(spec, [700, 3], [7, 5])
    first.checkpoint(spec.key)
    del first

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


def test_an_upgrade_cannot_relabel_old_hashes_as_new(db, tmp_path, monkeypatch):
    """The laundering case: hashes produced under producer v1, then the
    software upgrades to v2. The old rows must be rejected as input --
    recomputed, never reindexed under the new identity -- and the old
    space row must keep saying v1 forever."""
    import dataclasses

    from db import derived, runner, scan

    db.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'C:/lib','library',0)")
    folder = scan.mint(db, "folder", "lib")
    db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(?,1,NULL,'lib',0)", (folder,))
    files = []
    for n in range(2):
        file_id = scan.mint(db, "file", f"p{n}")
        db.execute(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,content_sha256,first_seen_at,last_seen_at)"
            " VALUES(?,?,?,'image',1,0,'aa',0,0)",
            (file_id, folder, f"p{n}.png"),
        )
        files.append(file_id)

    monkeypatch.setattr(similarity, "PHASH", dataclasses.replace(similarity.PHASH, producer_version="v1"))
    for file_id in files:
        derived.record_hash(db, file_id, "aa", 1.0, phash64=0b1)
    db.commit()
    manager = IndexManager(tmp_path)
    v1_key = similarity.align(db, manager, similarity.PHASH, files, lambda w: [0b1 for _ in w], 2.0)
    manager.checkpoint(v1_key)
    assert manager.count(v1_key) == 2

    # The upgrade. A fresh process boots with producer v2.
    monkeypatch.setattr(similarity, "PHASH", dataclasses.replace(similarity.PHASH, producer_version="v2"))
    fresh = IndexManager(tmp_path)
    runner.warm_similarity(db, 3.0)
    v2_key = similarity.keyed(similarity.PHASH, similarity.space_id(db, similarity.PHASH, 3.0))
    assert v2_key.key != v1_key, "the upgrade did not mint a new space"
    assert not fresh.restore(v2_key), "a v1 snapshot answered for v2"
    assert not fresh.has(v2_key.key) or fresh.count(v2_key.key) == 0, "v1 rows were reindexed as v2"

    old = db.execute(
        "SELECT s.producer_version, count(h.file_id) FROM similarity_space s"
        " JOIN derived_file_hash h ON h.space_id = s.id WHERE s.producer_version = 'v1'"
    ).fetchone()
    assert old == ("v1", 2), "the v1 rows lost their identity"

    # Recompute under v2: new rows under the new space; v1 rows intact.
    for file_id in files:
        derived.record_hash(db, file_id, "aa", 4.0, phash64=0b111)
    db.commit()
    told = {
        (version, count)
        for version, count in db.execute(
            "SELECT s.producer_version, count(h.file_id) FROM similarity_space s"
            " JOIN derived_file_hash h ON h.space_id = s.id GROUP BY s.id ORDER BY s.id"
        )
    }
    assert told == {("v1", 2), ("v2", 2)}, "recompute overwrote history instead of adding to it"


def test_spec_hash_is_canonical_not_delimited(db):
    """A delimiter join lets two different specs collide the moment a
    field contains the delimiter; canonical serialization cannot."""
    left = SpaceSpec("s", "binary", 64, "hamming", producer="a|b", producer_version="c")
    right = SpaceSpec("s", "binary", 64, "hamming", producer="a", producer_version="b|c")
    assert similarity.spec_hash(left) != similarity.spec_hash(right)
    assert similarity.space_id(db, left, 0.0) != similarity.space_id(db, right, 0.0)


def test_each_fingerprint_carries_its_own_producer(db):
    """pHash and dHash are different algorithms, so they are different
    spaces -- two values sharing one provenance row is how dHash bits
    got labeled as pHash output."""
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
    derived.record_hash(db, file_id, "aa", 1.0, phash64=0b1, dhash64=0b10)
    told = dict(
        db.execute("SELECT s.producer, h.value FROM derived_file_hash h JOIN similarity_space s ON s.id = h.space_id")
    )
    assert told == {"imagehash.phash": 0b1, "imagehash.dhash": 0b10}


def test_a_failed_post_commit_sync_invalidates_the_space(db, tmp_path):
    """A space that took half a batch could answer with stale rows, so a
    failed application marks it unservable -- resident and snapshot both
    -- and the next align rebuilds it from committed truth."""
    manager = IndexManager(tmp_path)
    key = similarity.align(db, manager, similarity.PHASH, [1], lambda w: [0b1 for _ in w], 1.0)
    similarity.note(db, similarity.PHASH, 2, 0b11, 2.0)

    def broken(*args, **kwargs):
        raise RuntimeError("the device fell over mid-batch")

    manager.upsert = broken
    similarity.apply_pending(db, manager)
    assert not manager.has(key), "a half-applied space is still answering"
    restored = IndexManager(tmp_path)
    assert not restored.restore(similarity.keyed(similarity.PHASH, similarity.space_id(db, similarity.PHASH, 3.0))), (
        "the stale snapshot survived the invalidation"
    )
    key = similarity.align(db, manager, similarity.PHASH, [1, 2], lambda w: [{1: 0b1, 2: 0b11}[v] for v in w], 4.0)
    assert manager.count(key) == 2, "align did not repair the invalidated space"


def test_a_face_row_cannot_claim_a_space_another_model_produced(db):
    """The duplicated model columns are conveniences; the space is the
    identity, and the schema refuses a row where they disagree."""
    import sqlite3 as sqlite_module

    from db import scan

    db.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'C:/lib','library',0)")
    folder = scan.mint(db, "folder", "lib")
    db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(?,1,NULL,'lib',0)", (folder,))
    file_id = scan.mint(db, "file", "p")
    db.execute(
        "INSERT INTO file(id,folder_id,name,kind,size,mtime,content_sha256,first_seen_at,last_seen_at)"
        " VALUES(?,?,?,'image',1,0,'aa',0,0)",
        (file_id, folder, "p.png"),
    )
    db.execute("INSERT INTO region(id, x, y, w, h) VALUES(1, 0.1, 0.1, 0.2, 0.2)")
    other = similarity.space_id(db, similarity.face_space("other-model", "9", 1), 0.0)
    with pytest.raises(sqlite_module.IntegrityError, match="another"):
        db.execute(
            "INSERT INTO derived_face_instance(file_id, region_id, model_id, model_version,"
            " det_score, embedding, dim, space_id, source_sha256, computed_at)"
            " VALUES(?, 1, 'm', '1', 0.9, x'00000000', 1, ?, 'aa', 0)",
            (file_id, other),
        )


def test_rank_fusion_consumes_positions_never_magnitudes():
    """RRF's entire contract: two rankings whose scores live on absurdly
    different scales fuse identically to the same rankings with any
    other scales, because scores never enter. Agreement accumulates;
    either list can still surface what the other missed."""
    from db import retrieval

    fused = retrieval.rrf([[7, 3, 9], [3, 7, 100]])
    assert fused[3] == fused[7], "symmetric agreement must tie"
    assert fused[3] > fused[9], "two mid ranks outweigh one high-only rank"
    assert fused[100] > 0, "a single space's find still surfaces"
    assert set(fused) == {3, 7, 9, 100}


def test_a_crash_between_commit_and_sync_cannot_revive_an_old_embedding(db, tmp_path):
    """The verdict's hostile case: vector A resident and checkpointed;
    vector B committed for the SAME file; the process dies before the
    post-commit sync. On restart the snapshot still holds A -- but under
    an embedding id that no longer exists, because a replacement mints a
    new immutable id instead of reusing the file's. Alignment sees an id
    disappear and another appear, and the space answers B, never A."""
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
    spec = similarity.semantic_space("ViT-B-32", "test", 4)
    a_vector = np.array([1, 0, 0, 0], dtype=np.float32)
    b_vector = np.array([0, 1, 0, 0], dtype=np.float32)
    first_id = derived.record_embedding(db, file_id, spec, a_vector, "aa", 0.0)
    db.commit()
    similarity.discard_pending(db)

    from db import retrieval

    manager = IndexManager(tmp_path)
    found = retrieval._space_of(db, "openclip", "ViT-B-32", "test")
    assert found is not None
    sid, full = found
    rows = retrieval.current_rows(db, sid)
    key = similarity.align(db, manager, full, [e for e, _ in rows], lambda w: retrieval._vectors(db, w), 1.0)
    manager.checkpoint(key)

    # The re-embed commits; the process dies before apply_pending.
    second_id = derived.record_embedding(db, file_id, spec, b_vector, "aa", 2.0)
    db.commit()
    similarity.discard_pending(db)  # the crash: notes lost, sync never ran
    assert second_id != first_id, "a replacement must mint a new immutable id"

    # Restart: a fresh manager restores the stale snapshot, then aligns.
    reborn = IndexManager(tmp_path)
    rows = retrieval.current_rows(db, sid)
    key = similarity.align(db, reborn, full, [e for e, _ in rows], lambda w: retrieval._vectors(db, w), 3.0)
    labels, scores = reborn.search(key, [b_vector], 1)
    assert int(labels[0][0]) == second_id, "the index answered something other than the committed vector"
    assert float(scores[0][0]) > 0.99, "the committed vector B is not what the space serves"
    labels, scores = reborn.search(key, [a_vector], 1)
    assert float(scores[0][0]) < 0.5, "the crashed-out vector A is still resident"


def test_a_search_deeper_than_the_gpu_ceiling_still_answers_exactly():
    """The GPU k-select kernel refuses k > 2048 outright -- a constant of
    the selection algorithm (GPU_MAX_SELECTION_K, faiss/gpu/utils/
    DeviceDefs.cuh; the refusal in gpu/impl/IndexUtils.cu
    validateKSelect), the same on every card. The one caller that asks
    deeper is retrieval materializing a whole ranking, once per
    projection; past the ceiling the CPU canonical answers -- the same
    exact flat computation with no ceiling. Found live: a 2,995-file
    library 500'd /g?q=... on this exact refusal."""
    from vision.faiss_index import GPU_MAX_K
    from vision.faiss_runtime import import_faiss

    import_faiss(gpu=True)
    rng = np.random.default_rng(23)
    count = GPU_MAX_K + 300
    vectors = rng.normal(size=(count, 8)).astype(np.float32)
    ids = list(range(1, count + 1))
    spec = SpaceSpec("face.deep", "float32", 8, "cosine")
    manager = IndexManager(gpu=True)
    manager.load(spec, ids, vectors)

    shallow, _ = manager.search(spec.key, vectors[:1], 5)
    deep, _ = manager.search(spec.key, vectors[:1], count)
    assert manager.served_by(spec.key) == "faiss-cpu", "past the ceiling the CPU canonical must answer"
    assert list(deep[0][:5]) == list(shallow[0][:5]), "depth must not change the front of the answer"
    assert set(deep[0].tolist()) == set(ids), "the deep ask is the whole ranking"
