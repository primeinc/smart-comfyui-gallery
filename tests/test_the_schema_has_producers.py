"""Every table is written by something, and the writes honour their contracts.

A schema whose tables nothing fills is a design document. These exercise the
producers against the promises the DDL makes but cannot enforce: that an
address survives a rename, that dropping every derived table and re-indexing
leaves the library intact, that a job cannot report success with work
outstanding, and that a worker which lost its lease cannot still write.
"""

import json
import pathlib
import sqlite3

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import authored, collections, connect, derived, ingest, jobs, library, lineage, naming, probe, sample, scan
from db import similarity as similarity_module
from tests.staging import fresh_schema
from vision import decode

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"
NOW = 1_700_000_000.0


@pytest.fixture
def db():
    conn = fresh_schema()
    yield conn
    conn.close()


@pytest.fixture
def a_library(db, tmp_path):
    """A root, a folder, and one file to hang everything on."""
    root = tmp_path / "lib"
    root.mkdir()
    root_id = library.add_root(db, root, "library", NOW)
    folder_id = scan.ensure_folder(db, root_id, None, "lib")
    file_id = scan.mint(db, "file", "dusk")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256,"
        " first_seen_at, last_seen_at) VALUES(?, ?, 'dusk.png', 'image', 10, 0, 'aa', ?, ?)",
        (file_id, folder_id, NOW, NOW),
    )
    user_id = authored.add_user(db, "will", "hash", "ADMIN", NOW)
    return {"root": root_id, "path": root, "folder": folder_id, "file": file_id, "user": user_id}


# --- addressing ------------------------------------------------------------


def claimed(conn, owner, now):
    """jobs.claim with "something was runnable" asserted once."""
    taken = jobs.claim(conn, owner, now)
    assert taken is not None, "nothing was runnable"
    return taken


def slug_of(conn, entity_id):
    addressed = naming.entity_slug(conn, entity_id)
    assert addressed is not None, f"entity {entity_id} has no address"
    return addressed[1]


def test_a_rename_keeps_the_old_address_working(db, a_library):
    person = authored.person(db, None, NOW)
    first = slug_of(db, person)

    authored.name_person(db, person, "Ilse", NOW)
    current = slug_of(db, person)
    assert current == "ilse"

    assert naming.resolve(db, "person", "ilse") == (person, True)
    assert naming.resolve(db, "person", first) == (person, False), (
        "the address someone wrote down last year must still resolve"
    )


def test_a_slug_can_be_retired_more_than_once(db, a_library):
    """The history key includes the retirement time for exactly this."""
    person = authored.person(db, "Ilse", NOW)
    authored.name_person(db, person, "Rook", NOW + 1)
    authored.name_person(db, person, "Ilse", NOW + 2)
    authored.name_person(db, person, "Marguerite", NOW + 3)
    retired = db.execute(
        "SELECT slug, retired_at FROM slug_history WHERE entity_id = ? ORDER BY retired_at",
        (person,),
    ).fetchall()
    assert [r[0] for r in retired] == ["ilse", "rook", "ilse"]


def test_a_live_slug_beats_a_retired_one(db, a_library):
    """Otherwise renaming A frees a slug that B takes, and the old link to A
    starts answering with B."""
    first = authored.person(db, "Ilse", NOW)
    authored.name_person(db, first, "Rook", NOW + 1)
    second = authored.person(db, "Ilse", NOW + 2)

    assert naming.resolve(db, "person", "ilse") == (second, True)


def test_renaming_to_a_taken_name_does_not_steal_the_address(db, a_library):
    first = authored.person(db, "Ilse", NOW)
    second = authored.person(db, "Rook", NOW)
    authored.name_person(db, second, "Ilse", NOW + 1)

    assert slug_of(db, first) == "ilse"
    assert slug_of(db, second) == "ilse-2"


def test_a_rename_that_changes_nothing_writes_no_history(db, a_library):
    """A slug in history that is also live makes resolution depend on which
    table is consulted first."""
    person = authored.person(db, "Ilse", NOW)
    authored.name_person(db, person, "Ilse", NOW + 1)
    assert db.execute("SELECT count(*) FROM slug_history").fetchone()[0] == 0


def test_a_smart_collection_refuses_stored_members(db, a_library):
    """A smart collection's children are what its rule says, freshly,
    every time. A stored member row would give it a second, disagreeing
    answer -- refused at the schema, and by the writer before that."""
    import sqlite3 as sqlite_module

    file_id = a_library["file"]
    album = collections.collection(db, "Keepers", NOW)
    flag = collections.collection(db, "Flagged", NOW, kind="flag")
    smart = collections.collection(db, "Big seeds", NOW, kind="smart")

    collections.set_membership(db, album, file_id, True, NOW)
    collections.set_membership(db, flag, file_id, True, NOW)
    with pytest.raises(ValueError, match="smart"):
        collections.set_membership(db, smart, file_id, True, NOW)
    with pytest.raises(sqlite_module.IntegrityError):
        db.execute(
            "INSERT INTO collection_file(collection_id, file_id, added_at) VALUES(?, ?, ?)",
            (smart, file_id, NOW),
        )

    # Nor may a filled collection quietly BECOME smart: its rows would
    # instantly disagree with whatever rule it was given.
    with pytest.raises(sqlite_module.IntegrityError):
        db.execute("UPDATE collection SET kind = 'smart' WHERE id = ?", (album,))
    db.execute("DELETE FROM collection_file WHERE collection_id = ?", (album,))
    db.execute("UPDATE collection SET kind = 'smart' WHERE id = ?", (album,))


@pytest.mark.slow
def test_perceptual_hashes_come_from_pixels_not_literals(db, a_library, tmp_path):
    """The runtime producer for derived_file_hash, fed real pixels.

    The columns sat for a generation with only test literals behind them
    -- a storage function fed constants satisfies every gate while nothing
    in the running system computes a hash. These run the real path: the
    same picture re-encoded stays within a few bits; a different picture
    is far away; the signed-64 storage convention round-trips."""
    from PIL import Image

    from vision import dupes

    source = tmp_path / "castle.png"
    rng_pixels = Image.effect_noise((64, 64), 40).convert("RGB")
    rng_pixels.save(source)
    copy = tmp_path / "castle_half.jpg"
    rng_pixels.resize((32, 32)).save(copy, quality=80)
    other = tmp_path / "meadow.png"
    Image.effect_noise((64, 64), 90).convert("RGB").save(other)

    with decode.open_still(source) as img:
        p_source, d_source = dupes.perceptual(img)
    with decode.open_still(copy) as img:
        p_copy, _ = dupes.perceptual(img)
    with decode.open_still(other) as img:
        p_other, _ = dupes.perceptual(img)

    signed64 = range(-(1 << 63), 1 << 63)
    assert p_source in signed64
    assert d_source in signed64
    assert dupes.hamming(p_source, p_copy) <= 6, "a resized re-encode is the same picture"
    assert dupes.hamming(p_source, p_other) > 10, "different pictures must stay apart"

    derived.record_hash(db, a_library["file"], "aa", NOW, phash64=p_source, dhash64=d_source)
    stored = dict(
        db.execute(
            "SELECT s.producer, h.value FROM derived_file_hash h JOIN similarity_space s ON s.id = h.space_id"
            " WHERE h.file_id = ?",
            (a_library["file"],),
        )
    )
    assert stored == {"imagehash.phash": p_source, "imagehash.dhash": d_source}, (
        "the signed storage convention did not round-trip, or a fingerprint lost its producer"
    )


def test_detection_records_perceptual_hashes_as_a_byproduct(db, a_library, tmp_path):
    """The decoded frame is in hand; hashing it later means decoding twice.
    Same doctrine as the thumbnail byproduct, same guard: only the whole
    file's frame, never a video sample's."""
    from PIL import Image

    from db import detect

    path = tmp_path / "dusk.png"
    Image.effect_noise((64, 64), 55).convert("RGB").save(path)

    class NothingFound:
        model_id = "test/none"
        model_version = "0"

        def detect(self, image):
            return []

    detect.harvest(db, NothingFound(), a_library["file"], path, NOW)
    told = db.execute(
        "SELECT count(*) FROM derived_file_hash WHERE file_id = ? AND value IS NOT NULL", (a_library["file"],)
    ).fetchone()[0]
    assert told == 2, "detection decoded the frame and did not record both fingerprints"


def test_a_byproduct_on_an_unhashed_file_is_not_born_stale(db, a_library, tmp_path):
    """harvest computes the sha its derived rows key their staleness on
    when the file row has none -- and then threw the computation away:
    every byproduct recorded before scan hashed the file read as
    permanently stale (`source_sha256 IS NOT content_sha256` against
    NULL), recomputed by every sweep until an unrelated scan happened to
    write the sha. The computed sha is a fact about the file; it lands on
    the file row in the same transaction."""
    from PIL import Image

    from db import detect

    path = tmp_path / "early.png"
    Image.effect_noise((64, 64), 35).convert("RGB").save(path)

    class NothingFound:
        model_id = "test/none"
        model_version = "0"

        def detect(self, image):
            return []

    db.execute("UPDATE file SET content_sha256 = NULL WHERE id = ?", (a_library["file"],))
    detect.harvest(db, NothingFound(), a_library["file"], path, NOW)
    assert derived.stale(db, "derived_file_hash") == [], "the byproduct hash was born stale"
    (persisted,) = db.execute("SELECT content_sha256 FROM file WHERE id = ?", (a_library["file"],)).fetchone()
    assert persisted is not None, "the computed sha was thrown away instead of recorded"


# --- the rebuild contract --------------------------------------------------


def test_dropping_every_derived_table_leaves_the_library_standing(db, a_library):
    """The whole reason the derived namespace is segregated by name.

    Name a person, let a model infer them into files, drop the entire
    derived namespace, re-index -- and both the name and the attribution
    come back, the second from what a person asserted rather than from a
    similarity heuristic.
    """
    file_id, user_id = a_library["file"], a_library["user"]
    person = authored.person(db, "Ilse", NOW)
    authored.rate(db, file_id, user_id, 5, NOW)
    authored.assert_person(db, person, file_id, user_id, NOW)

    run = derived.run_for(db, "insightface", "v1", "given", None, NOW)
    cluster = derived.recluster(db, "insightface", "v1", NOW, [{"person_id": person}])[0]
    box = derived.region(db, 0.3, 0.2, 0.2, 0.3)
    faces = derived.record_faces(db, file_id, "insightface", "v1", "aa", NOW, [{"region": box}])
    derived.assign_cluster(db, faces[0], cluster)
    derived.attribute(db, file_id, person, run, "insightface", "v1")
    derived.annotate(db, file_id, "caption", "a brass diving helmet", "qwen-vl", "2.5", "aa", NOW)
    verdict = authored.feedback(db, "person", "right", NOW, file_id=file_id, person_id=person, user_id=user_id)

    dropped = derived.drop_all(db)
    # named, not counted: a count passes just as well when a table is missed
    assert set(dropped) == {
        "derived_annotation",
        "derived_context_state",
        "derived_dupe_group",
        "derived_embedding",
        "derived_face_cluster",
        "derived_face_instance",
        "derived_face_membership",
        "derived_face_scan",
        "derived_face_run",
        "derived_event",
        "derived_event_file",
        "derived_event_run",
        "derived_file_hash",
        "derived_file_person",
        "derived_prompt_embedding",
        "derived_prompt_section",
        "derived_media_context",
        "derived_media_occurrence",
        "derived_media_sample",
    }, dropped
    assert db.execute("SELECT count(*) FROM annotation_fts").fetchone()[0] == 0, (
        "the caption index outlived the captions"
    )

    # the authored side is untouched
    assert db.execute("SELECT name FROM person WHERE id = ?", (person,)).fetchone()[0] == "Ilse"
    assert db.execute("SELECT rating FROM rating WHERE file_id = ?", (file_id,)).fetchone()[0] == 5
    assert db.execute("SELECT count(*) FROM person_assertion WHERE person_id = ?", (person,)).fetchone()[0] == 1
    assert db.execute("SELECT verdict FROM feedback WHERE id = ?", (verdict,)).fetchone()[0] == "right"

    # re-index with a newer model, and the naming re-attaches from the record
    rebuilt = derived.recluster(db, "insightface", "v2", NOW + 10, [{}])[0]
    rebuilt_run = derived.run_for(db, "insightface", "v2", "given", None, NOW + 10)
    box_again = derived.region(db, 0.3, 0.2, 0.2, 0.3)
    again = derived.record_faces(
        db,
        file_id,
        "insightface",
        "v2",
        "aa",
        NOW + 10,
        [{"region": box_again}],
    )
    derived.assign_cluster(db, again[0], rebuilt)
    named = derived.seed_clusters_from_assertions(db, rebuilt_run)

    assert named == 1
    assert db.execute("SELECT person_id FROM derived_face_cluster WHERE id = ?", (rebuilt,)).fetchone()[0] == person
    assert db.execute("SELECT count(*) FROM derived_file_person WHERE person_id = ?", (person,)).fetchone()[0] == 1


def test_a_video_naming_survives_the_rebuild(db, a_library):
    """The video form of the rebuild contract. A face on a video belongs
    to a MOMENT (`sample_id`), and the assertion names that moment. If the
    rebuild erases which moment -- samples dropped, `sample_id` nulled --
    the seeder's cross-moment guard goes blind: in a video of two people
    the same box exists at both moments, every cluster collects both
    votes, and both names are lost."""
    folder = a_library["folder"]
    clip = scan.mint(db, "file", "clip")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256,"
        " first_seen_at, last_seen_at) VALUES(?, ?, 'clip.mp4', 'video', 10, 0, 'cc', ?, ?)",
        (clip, folder, NOW, NOW),
    )
    alice = authored.person(db, "Alice", NOW)
    bob = authored.person(db, "Bob", NOW)

    def indexed(version, when):
        """One detection pass: two moments, one face each, the same box."""
        run = derived.run_for(db, "test/emb", version, "given", None, when)
        faces = {}
        for name, offset in (("alice", 1000), ("bob", 9000)):
            moment = derived.add_sample(db, clip, "frame", "poster", offset_ms=offset)
            box = derived.region(db, 0.4, 0.4, 0.2, 0.2)
            (face,) = derived.record_faces(
                db, clip, "test/emb", version, "cc", when, [{"region": box}], sample_id=moment
            )
            faces[name] = (face, moment, box)
        groups = derived.recluster(db, "test/emb", version, when, [{}, {}])
        derived.assign_cluster(db, faces["alice"][0], groups[0])
        derived.assign_cluster(db, faces["bob"][0], groups[1])
        return run, faces, groups

    _, faces, _groups = indexed("v1", NOW)
    for person, name in ((alice, "alice"), (bob, "bob")):
        _face, moment, box = faces[name]
        authored.assert_person(db, person, clip, None, NOW, sample_id=moment, region_id=box)

    derived.drop_all(db)
    run2, _, groups2 = indexed("v2", NOW + 10)
    named = derived.seed_clusters_from_assertions(db, run2)

    assert named == 2, "both moments' names must come back after the rebuild"
    owners = dict(db.execute("SELECT id, person_id FROM derived_face_cluster WHERE run_id = ?", (run2,)))
    assert owners[groups2[0]] == alice
    assert owners[groups2[1]] == bob


def test_a_video_naming_survives_a_changed_sampling_policy(db, a_library):
    """The policy token is expected to evolve without a schema change --
    the schema says so on `derived_media_sample.policy`. The assertion
    names a MOMENT; a rebuild that samples the same moment under a new
    token must still re-attach the name, because the human's claim is
    about the frame, never about the token that chose it."""
    folder = a_library["folder"]
    clip = scan.mint(db, "file", "clip2")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256,"
        " first_seen_at, last_seen_at) VALUES(?, ?, 'clip2.mp4', 'video', 10, 0, 'dd', ?, ?)",
        (clip, folder, NOW, NOW),
    )
    alice = authored.person(db, "Alice", NOW)
    bob = authored.person(db, "Bob", NOW)

    def indexed(version, when, policy):
        run = derived.run_for(db, "test/emb", version, "given", None, when)
        faces = {}
        for name, offset in (("alice", 1000), ("bob", 9000)):
            moment = derived.add_sample(db, clip, "frame", policy, offset_ms=offset)
            box = derived.region(db, 0.4, 0.4, 0.2, 0.2)
            (face,) = derived.record_faces(
                db, clip, "test/emb", version, "dd", when, [{"region": box}], sample_id=moment
            )
            faces[name] = (face, moment, box)
        groups = derived.recluster(db, "test/emb", version, when, [{}, {}])
        derived.assign_cluster(db, faces["alice"][0], groups[0])
        derived.assign_cluster(db, faces["bob"][0], groups[1])
        return run, faces, groups

    _, faces, _groups = indexed("v1", NOW, "every-2s")
    for person, name in ((alice, "alice"), (bob, "bob")):
        _face, moment, box = faces[name]
        authored.assert_person(db, person, clip, None, NOW, sample_id=moment, region_id=box)

    derived.drop_all(db)
    run2, _, groups2 = indexed("v2", NOW + 10, "every-2000ms")
    named = derived.seed_clusters_from_assertions(db, run2)

    assert named == 2, "the same moments under a new policy token must still carry the names"
    owners = dict(db.execute("SELECT id, person_id FROM derived_face_cluster WHERE run_id = ?", (run2,)))
    assert owners[groups2[0]] == alice
    assert owners[groups2[1]] == bob


def test_the_fallback_naming_run_is_the_earliest_carrying_run(db, a_library):
    """When the primary run does not carry the person, the assertion base
    is the EARLIEST-STAMPED run that does, run id as the final tiebreak:
    a deterministic function of the recorded rows, never of a query plan.
    Preferring the newest run wrote whichever model ran last into the
    authored record. Minting itself is not something the rows record --
    re-stamping a run legitimately moves the pick, and the second half of
    this test pins that as the chosen rule rather than an accident."""
    folder, file_a = a_library["folder"], a_library["file"]
    file_b = scan.mint(db, "file", "later")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256,"
        " first_seen_at, last_seen_at) VALUES(?, ?, 'later.png', 'image', 10, 0, 'ee', ?, ?)",
        (file_b, folder, NOW, NOW),
    )
    person = authored.person(db, None, NOW)

    def carried(version, when, file_id):
        run = derived.run_for(db, "test/emb", version, "given", None, when)
        (group,) = derived.recluster(db, "test/emb", version, when, [{"person_id": person}])
        (face,) = derived.record_faces(
            db, file_id, "test/emb", version, "aa", when, [{"region": derived.region(db, 0.2, 0.2, 0.2, 0.2)}]
        )
        derived.assign_cluster(db, face, group)
        return run

    minting = carried("v1", NOW, file_a)
    carried("v2", NOW + 10, file_b)  # a newer run carries them somewhere else

    asserted = authored.assert_named_cluster(db, person, None, NOW + 20)
    held = [row[0] for row in db.execute("SELECT file_id FROM person_assertion WHERE person_id = ?", (person,))]
    assert asserted == 1
    assert held == [file_a], f"the assertion base must be the earliest run {minting}'s files, not the newest run's"

    # Re-stamping the earliest run moves the pick to the other one: the
    # rule reads the stamps as they stand. Recording minting would need a
    # column the schema does not carry; until somebody needs it, the
    # deterministic-in-the-rows rule is the contract.
    db.execute("DELETE FROM person_assertion WHERE person_id = ?", (person,))
    derived.run_for(db, "test/emb", "v1", "given", None, NOW + 30)
    authored.assert_named_cluster(db, person, None, NOW + 40)
    moved = [row[0] for row in db.execute("SELECT file_id FROM person_assertion WHERE person_id = ?", (person,))]
    assert moved == [file_b], "after a re-stamp the earliest-stamped run is the other one"


def _two_people_in_one_photo(db, a_library, tmp_path):
    """A group shot and a solo shot, with a human's claim on each face."""
    group = a_library["file"]
    solo = scan.mint(db, "file", "solo")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256,"
        " first_seen_at, last_seen_at) VALUES(?, ?, 'solo.png', 'image', 10, 0, 'bb', ?, ?)",
        (solo, a_library["folder"], NOW, NOW),
    )
    alice = authored.person(db, "Alice", NOW)
    bob = authored.person(db, "Bob", NOW)
    # left half of the frame is Alice, right half is Bob
    alice_box = derived.region(db, 0.05, 0.1, 0.35, 0.5)
    bob_box = derived.region(db, 0.55, 0.1, 0.35, 0.5)
    authored.assert_person(db, alice, group, a_library["user"], NOW, region_id=alice_box)
    authored.assert_person(db, bob, group, a_library["user"], NOW, region_id=bob_box)
    authored.assert_person(db, bob, solo, a_library["user"], NOW)
    return {"group": group, "solo": solo, "alice": alice, "bob": bob}


def test_a_photograph_of_two_people_re_attaches_both_names(db, a_library, tmp_path):
    """Matched by where in the picture, not by which file.

    Joining an assertion to a cluster on file_id and breaking the tie with
    `count(*) DESC LIMIT 1` gave both clusters the same arbitrary winner:
    Bob was attached to the face the detector put on Alice, and Alice's name
    left the library. Every test in the suite used one person and one file,
    where that cannot happen.
    """
    cast = _two_people_in_one_photo(db, a_library, tmp_path)
    hers, his, his_again = (
        derived.region(db, 0.06, 0.12, 0.33, 0.47),  # detector, on Alice
        derived.region(db, 0.56, 0.12, 0.33, 0.47),  # detector, on Bob
        derived.region(db, 0.30, 0.20, 0.40, 0.40),  # Bob again, in the solo
    )
    left, right = derived.recluster(db, "insightface", "v2", NOW, [{}, {}])
    pair = derived.record_faces(
        db,
        cast["group"],
        "insightface",
        "v2",
        "aa",
        NOW,
        [{"region": hers}, {"region": his}],
    )
    derived.assign_cluster(db, pair[0], left)
    derived.assign_cluster(db, pair[1], right)
    solo_face = derived.record_faces(
        db,
        cast["solo"],
        "insightface",
        "v2",
        "bb",
        NOW,
        [{"region": his_again}],
    )
    derived.assign_cluster(db, solo_face[0], right)

    run = derived.run_for(db, "insightface", "v2", "given", None, NOW)
    assert derived.seed_clusters_from_assertions(db, run) == 2
    named = dict(db.execute("SELECT c.id, p.name FROM derived_face_cluster c JOIN person p ON p.id = c.person_id"))
    assert named == {left: "Alice", right: "Bob"}


def test_a_claim_with_no_box_does_not_name_a_face_in_a_group(db, a_library, tmp_path):
    """ "She is in this picture" is true and does not say which face.

    Naming a cluster from it anyway is how a person's name lands on somebody
    else. An unnamed cluster is a question for the People page; a wrongly
    named one is a lie the user has to find.
    """
    cast = _two_people_in_one_photo(db, a_library, tmp_path)
    db.execute("UPDATE person_assertion SET region_id = NULL WHERE file_id = ?", (cast["group"],))
    hers = derived.region(db, 0.06, 0.12, 0.33, 0.47)
    his = derived.region(db, 0.56, 0.12, 0.33, 0.47)
    left, right = derived.recluster(db, "insightface", "v2", NOW, [{}, {}])
    pair = derived.record_faces(
        db,
        cast["group"],
        "insightface",
        "v2",
        "aa",
        NOW,
        [{"region": hers}, {"region": his}],
    )
    derived.assign_cluster(db, pair[0], left)
    derived.assign_cluster(db, pair[1], right)

    run = derived.run_for(db, "insightface", "v2", "given", None, NOW)
    assert derived.seed_clusters_from_assertions(db, run) == 0
    assert db.execute("SELECT count(*) FROM derived_face_cluster WHERE person_id IS NOT NULL").fetchone()[0] == 0


def test_running_a_detector_twice_does_not_double_the_faces(db, a_library):
    """The re-run is the case the derived namespace exists for, and the only
    test that covered "recomputing is not an append" used a table whose
    composite primary key made it true for free."""
    file_id = a_library["file"]
    for _ in range(3):
        derived.record_faces(
            db,
            file_id,
            "insightface",
            "v1",
            "aa",
            NOW,
            [
                {"region": derived.region(db, 0.1, 0.1, 0.2, 0.2), "det_score": 0.9},
                {"region": derived.region(db, 0.6, 0.1, 0.2, 0.2), "det_score": 0.8},
            ],
        )
    assert db.execute("SELECT count(*) FROM derived_face_instance").fetchone()[0] == 2
    assert db.execute("SELECT count(*) FROM region").fetchone()[0] == 2, (
        "the boxes of the replaced detections were left behind"
    )

    # a better version finds fewer faces, and must be able to say so
    derived.record_faces(
        db,
        file_id,
        "insightface",
        "v1",
        "aa",
        NOW,
        [{"region": derived.region(db, 0.1, 0.1, 0.2, 0.2)}],
    )
    assert db.execute("SELECT count(*) FROM derived_face_instance").fetchone()[0] == 1


def test_sampling_the_same_frame_twice_returns_the_same_row(db, a_library):
    """A frame job that was interrupted and resumed raised on the first frame
    it had already taken -- the one case resumption exists for."""
    first = derived.add_sample(db, a_library["file"], "frame", "every-2s", offset_ms=4000)
    again = derived.add_sample(db, a_library["file"], "frame", "every-2s", offset_ms=4000)
    assert first == again
    assert db.execute("SELECT count(*) FROM derived_media_sample").fetchone()[0] == 1


def test_a_region_is_a_fraction_of_the_frame_not_a_pixel_count(db, a_library):
    """A box in pixels is a box against one rendering: the same numbers on a
    thumbnail or a re-encoded proxy point somewhere else."""
    box = derived.region_from_pixels(db, (256, 128, 512, 384), 1024, 768)
    stored = db.execute("SELECT x, y, w, h FROM region WHERE id = ?", (box,)).fetchone()
    assert stored == (0.25, pytest.approx(1 / 6), 0.5, 0.5)

    with pytest.raises(ValueError, match="mostly outside"):
        derived.region(db, 0.9, 0.1, 0.5, 0.1)  # four fifths of it is not there
    with pytest.raises(sqlite3.IntegrityError):
        derived.region(db, 0.1, 0.1, 0.0, 0.1)  # zero width locates nothing


def test_a_face_at_the_edge_of_the_frame_is_kept(db, a_library):
    """A detector reports the whole head's extent, so a face at the side of
    the picture comes back overhanging it. Measured on 423 real YuNet
    detections, one did -- by 1% of the frame -- and the CHECK refused it,
    losing a real face over a rounding of reality. It is trimmed to the
    frame, because the region says where in the picture something is."""
    box = derived.region(db, 0.843, 0.295, 0.167, 0.394)
    assert db.execute("SELECT round(x, 3), round(x + w, 3) FROM region WHERE id = ?", (box,)).fetchone() == (0.843, 1.0)

    # and a box that never needed trimming is stored exactly as given
    exact = derived.region(db, 0.6, 0.1, 0.3, 0.1)
    assert db.execute("SELECT x, w FROM region WHERE id = ?", (exact,)).fetchone() == (0.6, 0.3), (
        "a coordinate made a round trip it never asked for"
    )


def test_a_mask_is_bytes_not_a_path(db, a_library):
    """A path is identity derived from location, which is the defect this
    schema exists to delete. Moving a cache directory must not void a mask."""
    box = derived.region(db, 0.1, 0.1, 0.4, 0.4, mask=b"\x89PNG\r\n\x1a\n-mask-bytes")
    row = db.execute(
        "SELECT b.payload_bin, b.byte_len FROM region r JOIN blob b ON b.hash = r.mask_hash WHERE r.id = ?",
        (box,),
    ).fetchone()
    assert row[0] == b"\x89PNG\r\n\x1a\n-mask-bytes"
    assert row[1] == len(b"\x89PNG\r\n\x1a\n-mask-bytes")


def test_a_caption_is_found_by_its_words(db, a_library):
    """A description nobody can search for is the same as not having one."""
    file_id = a_library["file"]
    derived.annotate(db, file_id, "caption", "a brass diving helmet at dusk", "qwen-vl", "2.5", "aa", NOW)
    derived.annotate(
        db,
        file_id,
        "description",
        "A weathered brass helmet rests on a jetty as the light fails.",
        "qwen-vl",
        "2.5",
        "aa",
        NOW,
    )
    hits = derived.search_annotations(db, "brass")
    assert {hit["kind"] for hit in hits} == {"caption", "description"}
    assert derived.search_annotations(db, "helicopter") == []


def test_two_models_may_describe_one_picture(db, a_library):
    """They are compared, not merged -- which is the point of running both."""
    file_id = a_library["file"]
    derived.annotate(db, file_id, "caption", "a diving helmet", "qwen-vl", "2.5", "aa", NOW)
    derived.annotate(db, file_id, "caption", "an old brass hat", "florence", "2", "aa", NOW)
    captions = derived.said_about(db, file_id, kind="caption")
    assert {c["model_id"] for c in captions} == {"qwen-vl", "florence"}


def test_rerunning_one_model_replaces_its_own_answer(db, a_library):
    """Otherwise a re-parse accumulates versions of the same claim."""
    file_id = a_library["file"]
    derived.annotate(db, file_id, "caption", "first attempt", "qwen-vl", "2.5", "aa", NOW)
    derived.annotate(db, file_id, "caption", "better attempt", "qwen-vl", "2.5", "aa", NOW + 5)
    captions = derived.said_about(db, file_id, kind="caption")
    assert [c["text"] for c in captions] == ["better attempt"]
    assert db.execute("SELECT count(*) FROM annotation_fts").fetchone()[0] == 1, (
        "the search index kept the superseded text"
    )


def test_an_annotation_may_point_at_part_of_the_picture(db, a_library):
    """Text read out of an image sits somewhere; a tag may be about one
    object. NULL means it is about the whole frame."""
    file_id = a_library["file"]
    box = derived.region(db, 0.6, 0.1, 0.3, 0.1)
    derived.annotate(db, file_id, "ocr", "CLOSED", "paddle", "3", "aa", NOW, region_id=box)
    row = db.execute(
        "SELECT a.text, r.x, r.w FROM derived_annotation a JOIN region r ON r.id = a.region_id WHERE a.file_id = ?",
        (file_id,),
    ).fetchone()
    assert row == ("CLOSED", 0.6, 0.3)


def test_an_annotation_cannot_cite_another_files_frame(db, a_library):
    """Otherwise a caption quotes a moment from a different video and the
    evidence link still reads as sound."""
    other = scan.mint(db, "file", "clip")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'clip.mp4', 'video', 10, 0, ?, ?)",
        (other, a_library["folder"], NOW, NOW),
    )
    frame = derived.add_sample(db, other, "frame", "every-2s", offset_ms=4000)
    with pytest.raises(sqlite3.IntegrityError):
        derived.annotate(
            db,
            a_library["file"],
            "caption",
            "wrong film",
            "qwen-vl",
            "2.5",
            "aa",
            NOW,
            sample_id=frame,
        )


def test_a_verdict_on_a_caption_survives_the_caption(db, a_library):
    """The annotation is derived and the next rebuild deletes it; "the
    caption for this file was wrong" has to still mean something after."""
    file_id, user_id = a_library["file"], a_library["user"]
    derived.annotate(db, file_id, "caption", "a diving helmet", "qwen-vl", "2.5", "aa", NOW)
    verdict = authored.feedback(
        db,
        "annotation",
        "wrong",
        NOW,
        file_id=file_id,
        annotation_kind="caption",
        user_id=user_id,
    )
    derived.drop_all(db)
    assert db.execute(
        "SELECT target_kind, annotation_kind, verdict FROM feedback WHERE id = ?", (verdict,)
    ).fetchone() == ("annotation", "caption", "wrong")


def test_feedback_must_say_which_description_it_judged(db, a_library):
    """ "The model was wrong about this file" is not actionable when the model
    said four different things about it."""
    with pytest.raises(sqlite3.IntegrityError):
        authored.feedback(db, "annotation", "wrong", NOW, file_id=a_library["file"])


def test_feedback_outlives_the_thing_it_judged(db, a_library):
    """Its pointers are SET NULL, not CASCADE: a verdict is authored, and
    dropping derived state must not delete it."""
    file_id, user_id = a_library["file"], a_library["user"]
    person = authored.person(db, "Ilse", NOW)
    verdict = authored.feedback(db, "person", "wrong", NOW, file_id=file_id, person_id=person, user_id=user_id)
    db.execute("DELETE FROM person WHERE id = ?", (person,))
    row = db.execute("SELECT verdict, person_id FROM feedback WHERE id = ?", (verdict,)).fetchone()
    assert row == ("wrong", None)


def test_staleness_follows_the_bytes_not_the_clock(db, a_library):
    """A restore or a sync client rewrites mtime without changing a pixel."""
    file_id = a_library["file"]
    derived.record_hash(db, file_id, "aa", NOW)
    db.execute("UPDATE file SET mtime = ? WHERE id = ?", (NOW + 9999, file_id))
    assert derived.stale(db, "derived_file_hash") == []

    db.execute("UPDATE file SET content_sha256 = 'bb' WHERE id = ?", (file_id,))
    assert derived.stale(db, "derived_file_hash") == [file_id]


def test_an_unhashed_file_never_reads_as_current(db, a_library):
    """`<>` is NULL-blind, and NULL is the normal state before hashing."""
    file_id = a_library["file"]
    derived.record_hash(db, file_id, "v1", NOW)
    db.execute("UPDATE file SET content_sha256 = NULL WHERE id = ?", (file_id,))
    assert derived.stale(db, "derived_file_hash") == [file_id]


# --- jobs ------------------------------------------------------------------


def test_a_job_reports_its_own_progress_from_the_row(db, a_library):
    job = jobs.submit(db, "scan", NOW, items=[1, 2, 3])
    claimed = jobs.claim(db, "worker-a", NOW)
    assert claimed is not None
    job_id, fence = claimed
    assert job_id == job

    jobs.finish_item(db, job_id, fence, 1)
    state = jobs.finish_item(db, job_id, fence, 2)
    assert (state.done, state.total, state.state) == (2, 3, "running")
    assert state.fraction == pytest.approx(2 / 3)

    # a client arriving now renders from the row, not from messages it missed
    assert jobs.snapshot(db, job_id)["done_count"] == 2


def test_a_job_may_not_report_success_with_work_outstanding(db, a_library):
    jobs.submit(db, "scan", NOW, items=[1, 2])
    job_id, fence = claimed(db, "worker-a", NOW)
    jobs.finish_item(db, job_id, fence, 1)

    with pytest.raises(ValueError, match="unfinished"):
        jobs.settle(db, job_id, fence, "done", NOW + 1)

    jobs.finish_item(db, job_id, fence, 2)
    jobs.settle(db, job_id, fence, "done", NOW + 2)
    assert jobs.progress(db, job_id).state == "done"


def test_cancelling_asks_and_the_runner_answers(db, a_library):
    """Flipping the state from outside would mark work finished that is
    still running."""
    jobs.submit(db, "scan", NOW, items=[1, 2])
    job_id, fence = claimed(db, "worker-a", NOW)
    jobs.cancel(db, job_id)

    assert jobs.cancelled(db, job_id)
    assert jobs.progress(db, job_id).state == "running", "a request is not a state"

    jobs.settle(db, job_id, fence, "cancelled", NOW + 1)
    assert jobs.progress(db, job_id).state == "cancelled"


def test_a_resumed_job_repeats_nothing(db, a_library):
    jobs.submit(db, "scan", NOW, items=[1, 2, 3, 4])
    job_id, fence = claimed(db, "worker-a", NOW)
    jobs.finish_item(db, job_id, fence, 1)
    jobs.finish_item(db, job_id, fence, 2)

    # the worker dies; its lease expires and another takes over
    later = NOW + jobs.LEASE_SECONDS + 1
    resumed = jobs.claim(db, "worker-b", later)
    assert resumed is not None
    assert resumed[0] == job_id
    assert jobs.pending(db, job_id) == [3, 4]


def test_an_evicted_worker_cannot_still_write(db, a_library):
    """A lease nobody can prove is not a lease: the reclaiming worker must
    fence the one it replaced."""
    jobs.submit(db, "scan", NOW, items=[1, 2])
    job_id, first_fence = claimed(db, "worker-a", NOW)

    later = NOW + jobs.LEASE_SECONDS + 1
    job_again, second_fence = claimed(db, "worker-b", later)
    assert job_again == job_id
    assert second_fence != first_fence

    with pytest.raises(jobs.LeaseLost):
        jobs.finish_item(db, job_id, first_fence, 1)
    with pytest.raises(jobs.LeaseLost):
        jobs.settle(db, job_id, first_fence, "done", later)

    jobs.finish_item(db, job_id, second_fence, 1)
    assert jobs.progress(db, job_id).done == 1


@pytest.mark.slow
def test_two_workers_racing_for_one_job_cannot_both_get_it(tmp_path):
    """The claim is a single write, and this is why.

    Every eviction test in the suite claimed twice in sequence on one
    connection, where a SELECT-then-UPDATE claim is correct. Run two
    connections at it and both selected the same queued row, both incremented
    the fence, both read it back as the same number, and `_held` passed for
    both -- two workers running one job, each believing it held the lease.
    """
    path = tmp_path / "jobs.db"
    setup = connect.connect(path)
    setup.executescript(SCHEMA.read_text(encoding="utf-8"))
    setup.commit()
    job_id = jobs.submit(setup, "scan", NOW, items=[1, 2])
    setup.commit()
    setup.close()

    a = connect.connect(path, autocommit=True)
    b = connect.connect(path, autocommit=True)
    try:
        a.execute("PRAGMA busy_timeout=5000")
        b.execute("PRAGMA busy_timeout=5000")
        claims = [jobs.claim(a, "worker-a", NOW), jobs.claim(b, "worker-b", NOW)]
    finally:
        a.close()
        b.close()

    won = [c for c in claims if c is not None]
    assert len(won) == 1, f"both workers claimed job {job_id}: {claims}"


def test_a_live_lease_is_not_stolen(db, a_library):
    """The control: without it the test above would pass on a claim that
    always succeeds."""
    jobs.submit(db, "scan", NOW, items=[1])
    jobs.claim(db, "worker-a", NOW)
    assert jobs.claim(db, "worker-b", NOW + 1) is None


def test_a_heartbeat_holds_the_lease(db, a_library):
    jobs.submit(db, "scan", NOW, items=[1])
    job_id, fence = claimed(db, "worker-a", NOW)
    jobs.heartbeat(db, job_id, fence, NOW + jobs.LEASE_SECONDS - 1)
    assert jobs.claim(db, "worker-b", NOW + jobs.LEASE_SECONDS + 1) is None, (
        "a worker that is still reporting must keep its work"
    )


def test_work_without_units_resumes_from_a_checkpoint(db, a_library):
    """A scan cannot enumerate its units up front -- it is discovering them."""
    jobs.submit(db, "scan", NOW)
    job_id, fence = claimed(db, "worker-a", NOW)
    jobs.checkpoint(db, job_id, fence, {"after": "portraits/2026"}, done=140, at=NOW)
    stored = db.execute("SELECT checkpoint, done_count FROM job WHERE id = ?", (job_id,)).fetchone()
    assert json.loads(stored[0]) == {"after": "portraits/2026"}
    assert stored[1] == 140


def test_watching_a_folder_starts_nothing(db, a_library):
    """A watch says which folders a scan should cover. It is not a thread."""
    jobs.watch_folder(db, a_library["folder"], NOW)
    assert [row[0] for row in jobs.watched(db)] == [a_library["folder"]]
    assert jobs.active(db) == [], "recording a watch must not queue work"


# --- lineage ---------------------------------------------------------------


def test_a_derivation_is_recorded_when_it_is_asked_for(db, a_library):
    """The edge is knowable at submit and unrecoverable afterwards: the
    output arrives looking like any other new file."""
    parent = a_library["file"]
    child = scan.mint(db, "file", "child")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'child.png', 'image', 10, 0, ?, ?)",
        (child, a_library["folder"], NOW, NOW),
    )
    lineage.intend(db, parent, "remix", "comfy-job-9f2", NOW)
    assert len(lineage.open_intents(db)) == 1

    edge = lineage.resolve(db, "comfy-job-9f2", child, NOW + 30)
    assert edge is not None
    assert lineage.open_intents(db) == []
    assert db.execute("SELECT parent_id, child_id, kind FROM file_derivation WHERE id = ?", (edge,)).fetchone() == (
        parent,
        child,
        "remix",
    )


def test_submitting_twice_does_not_make_two_intents(db, a_library):
    first = lineage.intend(db, a_library["file"], "remix", "comfy-job-9f2", NOW)
    second = lineage.intend(db, a_library["file"], "remix", "comfy-job-9f2", NOW + 1)
    assert first == second


def test_a_file_cannot_derive_from_itself(db, a_library):
    """Every lineage walk from a self-edge is a cycle."""
    file_id = a_library["file"]
    lineage.intend(db, file_id, "upscale", "comfy-job-self", NOW)
    assert lineage.resolve(db, "comfy-job-self", file_id, NOW) is None
    assert lineage.link(db, file_id, file_id, "remix", NOW) is None
    assert db.execute("SELECT count(*) FROM file_derivation").fetchone()[0] == 0


def test_a_companion_is_found_from_either_side(db, a_library):
    raw_file = scan.mint(db, "file", "raw")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'dusk.dng', 'image', 10, 0, ?, ?)",
        (raw_file, a_library["folder"], NOW, NOW),
    )
    lineage.relate(db, a_library["file"], raw_file, "raw_pair", NOW)
    for left, right in ((a_library["file"], raw_file), (raw_file, a_library["file"])):
        assert (
            db.execute(
                "SELECT count(*) FROM file_relation WHERE file_id = ? AND related_id = ?",
                (left, right),
            ).fetchone()[0]
            == 1
        )


def test_a_proxy_does_not_claim_the_video_as_its_own_proxy(db, a_library):
    """`raw_pair` is symmetric and every other kind is not.

    Writing both directions for all of them asserted something false: after
    relating a video to its proxy, asking for the proxy of that file returned
    the video. The test above could not see it, because it used the one kind
    where both directions are true.
    """
    video = a_library["file"]
    proxy = scan.mint(db, "file", "proxy")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'dusk-proxy.mp4', 'video', 10, 0, ?, ?)",
        (proxy, a_library["folder"], NOW, NOW),
    )
    lineage.relate(db, video, proxy, "proxy", NOW)

    assert lineage.related(db, video, kind="proxy") == [(proxy, "proxy", "has")]
    assert lineage.related(db, proxy, kind="proxy") == [(video, "proxy", "belongs_to")]
    assert (
        db.execute(
            "SELECT count(*) FROM file_relation WHERE file_id = ? AND related_id = ?",
            (proxy, video),
        ).fetchone()[0]
        == 0
    ), "the proxy was recorded as having the video as its proxy"


# --- ingest ----------------------------------------------------------------


A1111 = (
    "a brass diving helmet at dusk <lora:filmGrain:0.35>\n"
    "Negative prompt: blurry\n"
    "Steps: 28, Sampler: Euler a, CFG scale: 7, Seed: 4242, Size: 832x1216, "
    "Model: dreamshaper_8, Version: v1.10.1"
)


@pytest.fixture
def a_generated_file(db, a_library, tmp_path):
    info = PngInfo()
    info.add_text("parameters", A1111)
    path = tmp_path / "gen.png"
    Image.new("RGB", (16, 16), (30, 40, 60)).save(path, pnginfo=info)
    return path


def test_a_generated_file_becomes_rows(db, a_library, a_generated_file):
    out = ingest.one(db, a_library["file"], a_generated_file, NOW)
    assert out.tool == "A1111 / Forge"

    row = db.execute(
        "SELECT g.seed, g.steps, g.cfg, g.sampler, g.width, g.height, p.text, n.text"
        "  FROM generation g"
        "  LEFT JOIN generation_prompt gp ON gp.file_id = g.file_id AND gp.role = 'effective'"
        "  LEFT JOIN prompt p ON p.id = gp.prompt_id"
        "  LEFT JOIN generation_prompt gn ON gn.file_id = g.file_id AND gn.role = 'negative'"
        "  LEFT JOIN prompt n ON n.id = gn.prompt_id WHERE g.file_id = ?",
        (a_library["file"],),
    ).fetchone()
    assert row[:6] == (4242, 28, 7.0, "Euler a", 832, 1216)
    assert row[6].startswith("a brass diving helmet")
    assert row[7] == "blurry"

    weights = db.execute(
        "SELECT a.kind, a.name FROM file_artifact fa JOIN artifact a ON a.id = fa.artifact_id"
        " WHERE fa.file_id = ? ORDER BY a.kind",
        (a_library["file"],),
    ).fetchall()
    assert weights == [("checkpoint", "dreamshaper_8"), ("lora", "filmGrain")]


def test_no_field_is_written_as_a_document(db, a_library, a_generated_file):
    """A structure stored as JSON is a field nothing can search."""
    ingest.one(db, a_library["file"], a_generated_file, NOW)
    values = [
        value
        for (value,) in db.execute(
            "SELECT value_text FROM file_param WHERE file_id = ? AND value_text IS NOT NULL",
            (a_library["file"],),
        )
    ]
    assert values, "a file full of metadata produced no fields"
    for value in values:
        stripped = value.strip()
        assert not (stripped.startswith("{") and stripped.endswith("}")), value
        assert not (stripped.startswith("[") and stripped.endswith("]")), value


def test_a_nested_value_becomes_one_field_per_leaf(db, a_library):
    """Flattened under dotted keys, so each leaf is its own facet."""
    ingest._param(
        db,
        a_library["file"],
        "sidecar",
        "capture",
        {"lens": {"model": "XF35mmF1.4", "serial": "12ab"}, "tags": ["dusk", "brass"]},
    )
    stored = dict(
        db.execute(
            "SELECT key, value_text FROM file_param WHERE file_id = ? AND source = 'sidecar'",
            (a_library["file"],),
        )
    )
    assert stored == {
        "capture.lens.model": "XF35mmF1.4",
        "capture.lens.serial": "12ab",
        "capture.tags.0": "dusk",
        "capture.tags.1": "brass",
    }


def test_the_carrier_is_kept_and_says_whether_it_was_understood(db, a_library, a_generated_file):
    ingest.one(db, a_library["file"], a_generated_file, NOW)
    rows = db.execute(
        "SELECT fb.slot, fb.parsed_by, b.byte_len FROM file_blob fb"
        " JOIN blob b ON b.hash = fb.blob_hash WHERE fb.file_id = ?",
        (a_library["file"],),
    ).fetchall()
    assert rows, "nothing kept the payload it parsed"
    assert all(length > 0 for _, _, length in rows)
    # The half the name promises and the assertions above never checked:
    # deleting the parsed_by logic from ingest._carrier left this test green.
    # A carrier is kept whether or not anything understood it, and which of
    # the two it was is the whole point -- it turns unparsed metadata into a
    # backlog you can query instead of a silent loss.
    claimed = {slot: parser for slot, parser, _ in rows}
    assert claimed.get("parameters") == "metaparse/A1111 / Forge", claimed


def test_a_carrier_nothing_understood_says_so(db, a_library, tmp_path):
    """The other half of `parsed_by`, and the half that makes it a backlog.

    It only means anything if some carriers are NULL and others are not.
    Marking a fixed list of slot names claimed every ComfyUI graph and no
    A1111 infotext, so "what does nothing understand yet" answered with the
    files that parsed best while a genuinely unrecognised chunk was
    indistinguishable from them.
    """
    info = PngInfo()
    info.add_text("parameters", A1111)
    info.add_text("SomeToolNobodyWrote", '{"knobs": [1, 2, 3]}')
    path = tmp_path / "mixed.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(path, pnginfo=info)
    ingest.one(db, a_library["file"], path, NOW)

    claimed = dict(db.execute("SELECT slot, parsed_by FROM file_blob WHERE file_id = ?", (a_library["file"],)))
    assert claimed["parameters"] is not None, "the chunk the recipe was read from"
    assert claimed["SomeToolNobodyWrote"] is None, "a chunk nothing read"
    assert db.execute("SELECT count(*) FROM file_blob WHERE parsed_by IS NULL").fetchone()[0] == 1, (
        "the backlog is not queryable"
    )


def test_the_registry_learns_what_the_file_contained(db, a_library, a_generated_file):
    ingest.one(db, a_library["file"], a_generated_file, NOW)
    # Keyed on (source, key), which is param_key's actual primary key. Keyed
    # on `key` alone the dict silently kept one row per name while the GROUP
    # BY summed every source, and the comparison meant nothing the moment one
    # key appeared under two sources -- `Width` already does.
    learned = {(s, k): n for s, k, n in db.execute("SELECT source, key, occurrences FROM param_key")}
    counted = {(s, k): n for s, k, n in db.execute("SELECT source, key, count(*) FROM file_param GROUP BY source, key")}
    assert learned == counted
    assert ("container", "Format") in learned, "container facts are metadata too"


def test_two_files_naming_one_model_share_its_row(db, a_library, a_generated_file, tmp_path):
    """Otherwise a model page counts spellings instead of pictures."""
    second = scan.mint(db, "file", "gen2")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'gen2.png', 'image', 10, 0, ?, ?)",
        (second, a_library["folder"], NOW, NOW),
    )
    ingest.one(db, a_library["file"], a_generated_file, NOW)
    ingest.one(db, second, a_generated_file, NOW)

    assert db.execute("SELECT count(*) FROM artifact WHERE kind = 'checkpoint'").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM file_artifact WHERE role = 'checkpoint'").fetchone()[0] == 2
    assert db.execute("SELECT count(*) FROM prompt").fetchone()[0] == 2


# --- roots -----------------------------------------------------------------


def test_an_unreachable_root_is_marked_offline_not_emptied(db, a_library, tmp_path):
    """Unplugged and emptied look identical from a listing, and only one of
    them is recoverable.

    A plain library, because this IS the case `root.kind = 'mount'` was
    reaching for and never carried: "not always attached" is `online`,
    per-root and set by probing, and it works the same for a folder that
    has never been anywhere near a removable drive.
    """
    missing = library.add_root(db, tmp_path / "not-here", "library", NOW)
    checked = {row[0]: row[2] for row in library.check_roots(db)}
    assert checked[a_library["root"]] is True
    assert checked[missing] is False
    assert db.execute("SELECT count(*) FROM file WHERE missing_since IS NOT NULL").fetchone()[0] == 0, (
        "checking a root must never touch a file"
    )


def test_a_setting_keeps_its_type(db):
    library.put(db, "thumbnail_size", 512)
    library.put(db, "watch", True)
    library.put(db, "roots", ["a", "b"])
    assert library.get(db, "thumbnail_size") == 512
    assert library.get(db, "watch") is True
    assert library.get(db, "roots") == ["a", "b"]
    assert library.get(db, "absent", "fallback") == "fallback"


# --- containers: the media Pillow cannot open -------------------------------


def _write_clip(path, *, seconds: int, rate: int = 15, size=(320, 180), rotation: int = 0) -> None:
    """A real H.264 clip, encoded in-process through the same libraries the
    reader uses. `rotation` writes the container display matrix the way a
    phone does (VideoStream.set_display_rotation; the write->decode round
    trip is pinned upstream in PyAV-Org/PyAV@040da79 tests/test_display_matrix.py)."""
    import av
    import numpy as np

    width, height = size
    with av.open(str(path), "w") as container:
        stream = container.add_stream("h264", rate=rate)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        if rotation:
            stream.set_display_rotation(rotation)
        for n in range(seconds * rate):
            frame = av.VideoFrame.from_ndarray(
                np.full((height, width, 3), (n * 8) % 256, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture(scope="module")
def a_clip(tmp_path_factory):
    """Three seconds of real video, and the same shape marked to be turned
    -- encoded once; the tests only read them."""
    tmp_path = tmp_path_factory.mktemp("clips")
    landscape, portrait = tmp_path / "clip.mp4", tmp_path / "portrait.mp4"
    _write_clip(landscape, seconds=3)
    _write_clip(portrait, seconds=3, rotation=90)
    return landscape, portrait


def test_a_video_knows_its_own_length_and_size(db, a_library, a_clip):
    """`file.duration` had no producer anywhere in the package, and the DDL
    said so rather than fixing it. A gallery whose plan says image and video
    are equal citizens cannot have one of them unable to state its length."""
    landscape, _ = a_clip
    file_id = scan.mint(db, "file", "clip")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'clip.mp4', 'video', 10, 0, ?, ?)",
        (file_id, a_library["folder"], NOW, NOW),
    )
    result = ingest.one(db, file_id, landscape, NOW)

    assert result.probed, result.unreadable
    assert db.execute("SELECT width, height, duration FROM file WHERE id = ?", (file_id,)).fetchone() == (
        320,
        180,
        pytest.approx(3.0, abs=0.2),
    )
    fields = dict(
        db.execute(
            "SELECT key, value_text FROM file_param WHERE file_id = ? AND source='container'",
            (file_id,),
        )
    )
    assert fields.get("VideoCodec") == "h264"
    assert "FrameRate" in fields, "a video that cannot say its frame rate is not searchable by it"


def test_a_portrait_video_is_not_filed_as_landscape(db, a_library, a_clip):
    """A phone records landscape and writes a display matrix saying to turn
    it. ffprobe reports the stored size and the rotation separately, so a
    reader taking width and height at face value files every portrait video
    in the library the wrong way round -- the EXIF orientation defect, one
    medium over."""
    _, portrait = a_clip
    file_id = scan.mint(db, "file", "portrait")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'portrait.mp4', 'video', 10, 0, ?, ?)",
        (file_id, a_library["folder"], NOW, NOW),
    )
    ingest.one(db, file_id, portrait, NOW)

    assert db.execute("SELECT width, height FROM file WHERE id = ?", (file_id,)).fetchone() == (180, 320), (
        "the stored size was taken for the displayed one"
    )
    assert (
        db.execute("SELECT value_num FROM file_param WHERE file_id=? AND key='Rotation'", (file_id,)).fetchone()[0]
        == 90
    )


def test_a_file_the_prober_cannot_read_costs_only_that_file(db, a_library, tmp_path):
    """One damaged video must not end the scan around it, and 'nothing could
    be read' has to be distinguishable from 'it had nothing to say'."""

    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video at all")
    file_id = scan.mint(db, "file", "broken")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'broken.mp4', 'video', 10, 0, ?, ?)",
        (file_id, a_library["folder"], NOW, NOW),
    )
    result = ingest.one(db, file_id, broken, NOW)

    assert not result.probed
    assert result.unreadable, "a file nothing could read said nothing about why"
    assert db.execute("SELECT width, height, duration FROM file WHERE id = ?", (file_id,)).fetchone() == (
        None,
        None,
        None,
    )
    assert probe.read(broken).is_empty


# --- ComfyUI writes a graph, not a line of text -----------------------------


def a_graph(**changes):
    """A workflow of the shape ComfyUI actually emits as its `prompt` chunk."""
    nodes = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "dreamshaper_8.safetensors"}},
        "10": {
            "class_type": "LoraLoader",
            "inputs": {
                "model": ["4", 0],
                "clip": ["4", 1],
                "lora_name": "filmGrain.safetensors",
                "strength_model": 0.4,
                "strength_clip": 0.35,
            },
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["10", 1], "text": "a brass diving helmet at dusk"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["10", 1], "text": "blurry"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1216, "batch_size": 1}},
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["10", 0],
                "seed": 4242,
                "steps": 28,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0]}},
    }
    nodes.update(changes)
    return nodes


def a_comfy_file(path, nodes):
    info = PngInfo()
    info.add_text("prompt", json.dumps(nodes))
    Image.new("RGB", (832, 1216), (30, 40, 60)).save(path, pnginfo=info)
    return path


def _ingest_comfy(db, a_library, path):
    file_id = scan.mint(db, "file", path.stem)
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, ?, 'image', 10, 0, ?, ?)",
        (file_id, a_library["folder"], path.name, NOW, NOW),
    )
    ingest.one(db, file_id, path, NOW)
    return file_id


def test_a_comfyui_picture_reports_its_whole_recipe(db, a_library, tmp_path):
    """This is a ComfyUI gallery and its recipe axis was empty for ComfyUI.

    Every other tool writes its settings as text and metaparse reads them.
    ComfyUI writes a node graph, and metaparse's own adapter says it "only
    identifies the tool" (metaparse/adapters.py:474-479). So a ComfyUI
    picture arrived with tool='ComfyUI' and NULL seed, steps, cfg, sampler,
    model and prompt -- no checkpoint row, no LoRA rows, nothing on the model
    page and nothing for LoRA synergy to join. The graph was in the file the
    whole time.
    """
    file_id = _ingest_comfy(db, a_library, a_comfy_file(tmp_path / "comfy.png", a_graph()))

    assert db.execute(
        "SELECT tool, detection, seed, steps, cfg, sampler, scheduler, width, height"
        "  FROM generation WHERE file_id = ?",
        (file_id,),
    ).fetchone() == ("ComfyUI", "graph", 4242, 28, 7.0, "euler", "normal", 832, 1216)
    assert (
        db.execute(
            "SELECT a.name FROM artifact a JOIN file_artifact fa ON fa.artifact_id = a.id"
            " WHERE fa.file_id = ? AND fa.role = 'checkpoint'",
            (file_id,),
        ).fetchone()[0]
        == "dreamshaper_8.safetensors"
    )
    assert db.execute(
        "SELECT a.name, fa.model_weight, fa.clip_weight FROM artifact a"
        "  JOIN file_artifact fa ON fa.artifact_id = a.id"
        " WHERE fa.file_id = ? AND fa.role = 'lora'",
        (file_id,),
    ).fetchall() == [("filmGrain.safetensors", 0.4, 0.35)]
    assert (
        db.execute(
            "SELECT p.text FROM prompt p JOIN generation_prompt g ON g.prompt_id = p.id"
            " WHERE g.file_id = ? AND g.role = 'effective'",
            (file_id,),
        ).fetchone()[0]
        == "a brass diving helmet at dusk"
    )


def test_a_refiner_pass_does_not_report_the_pass_that_was_thrown_away(db, a_library, tmp_path):
    """A workflow routinely holds several samplers. The one that made the
    file is the one whose latent reaches the node that saved it; taking the
    first found describes a pass whose output was discarded, which is worse
    than reporting nothing because it looks like an answer."""
    nodes = a_graph()
    nodes["20"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["10", 0],
            "seed": 1,
            "steps": 4,
            "cfg": 1.0,
            "sampler_name": "lcm",
            "scheduler": "sgm_uniform",
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        },
    }
    nodes["3"]["inputs"] = dict(nodes["3"]["inputs"], latent_image=["20", 0], seed=999)

    file_id = _ingest_comfy(db, a_library, a_comfy_file(tmp_path / "refined.png", nodes))
    assert db.execute("SELECT seed, steps, sampler FROM generation WHERE file_id = ?", (file_id,)).fetchone() == (
        999,
        28,
        "euler",
    ), "it read the pass whose output was discarded"


def test_a_swarmui_refiner_is_checkpoint_weights_in_the_refiner_role(db, a_library, tmp_path):
    """Using a model as a refiner is a different fact from using it as the
    base -- the ROLE says which -- but the artifact row says what the thing
    IS: checkpoint weights. Ingest passed the role straight through as the
    artifact kind, which the kind CHECK refuses, so every SwarmUI render
    carrying `refinermodel` died as an IntegrityError blamed on the file.
    The schema's own role-match trigger states the mapping (schema.sql:
    role 'refiner' attaches kind 'checkpoint')."""
    payload = json.dumps(
        {
            "sui_image_params": {
                "prompt": "a harbour at dusk",
                "model": "sd_xl_base_1.0",
                "refinermodel": "sd_xl_refiner_1.0",
                "seed": 7,
                "steps": 20,
                "cfgscale": 7.0,
                "width": 1024,
                "height": 1024,
                "swarm_version": "0.9.8.1",
            },
            "sui_models": [
                {"name": "sd_xl_base_1.0.safetensors", "param": "model", "hash": "0xaa"},
                {"name": "sd_xl_refiner_1.0.safetensors", "param": "refinermodel", "hash": "0xbb"},
            ],
        }
    )
    info = PngInfo()
    info.add_text("parameters", payload)
    path = tmp_path / "refined.png"
    Image.new("RGB", (32, 32), (30, 40, 60)).save(path, pnginfo=info)

    file_id = _ingest_comfy(db, a_library, path)
    rows = db.execute(
        "SELECT fa.role, a.kind, a.name FROM file_artifact fa JOIN artifact a ON a.id = fa.artifact_id"
        " WHERE fa.file_id = ? ORDER BY fa.role",
        (file_id,),
    ).fetchall()
    assert ("refiner", "checkpoint", "sd_xl_refiner_1.0.safetensors") in rows, rows
    assert ("checkpoint", "checkpoint", "sd_xl_base_1.0.safetensors") in rows, rows


def test_a_probe_complaint_on_an_animated_image_survives_the_capture_read(db, a_library, tmp_path, monkeypatch):
    """An animated image is read twice -- the container probe for duration
    and frame facts, Pillow for capture facts -- and the second read's
    silence overwrote the first's complaint: `unreadable` went back to
    None while duration stayed NULL, and nothing said why. The first
    complaint stands; absence never overwrites presence."""
    from db import probe as probe_module

    path = tmp_path / "flip.gif"
    frames = [Image.new("RGB", (16, 16), shade) for shade in ((10, 10, 10), (200, 200, 200))]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=100)

    monkeypatch.setattr(
        probe_module, "read", lambda _p: probe_module.Probed(unreadable="the container reader choked on this")
    )
    file_id = scan.mint(db, "file", path.stem)
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, ?, 'animated_image', 10, 0, ?, ?)",
        (file_id, a_library["folder"], path.name, NOW, NOW),
    )
    result = ingest.one(db, file_id, path, NOW)
    assert result.unreadable == "the container reader choked on this", (
        f"the capture read silenced the probe's complaint: {result.unreadable!r}"
    )


def test_a_prompt_routed_through_another_node_is_still_found(db, a_library, tmp_path):
    """A workflow that runs its prompt through a primitive, a concat or a
    wildcard node keeps the words a node or two upstream. Reading only the
    literal reports an empty prompt for the workflows most likely to have an
    interesting one."""
    nodes = a_graph()
    nodes["30"] = {"class_type": "PrimitiveString", "inputs": {"value": "a castle assembled from wildcards"}}
    nodes["6"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["10", 1], "text": ["30", 0]}}

    file_id = _ingest_comfy(db, a_library, a_comfy_file(tmp_path / "routed.png", nodes))
    assert (
        db.execute(
            "SELECT p.text FROM prompt p JOIN generation_prompt g ON g.prompt_id = p.id"
            " WHERE g.file_id = ? AND g.role = 'effective'",
            (file_id,),
        ).fetchone()[0]
        == "a castle assembled from wildcards"
    )


def test_a_graph_that_refers_to_itself_ends_the_walk(db):
    """A graph is meant to be acyclic. A malformed one is still a file
    somebody has in their library, and it must cost that file rather than
    hanging the scan."""
    from db import graph as graph_module

    recipe = graph_module.read(
        {
            "1": {"class_type": "KSampler", "inputs": {"model": ["2", 0], "seed": 7, "positive": ["1", 0]}},
            "2": {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "lora_name": "loop.safetensors"}},
        }
    )
    assert recipe is not None
    assert recipe.seed == 7


@pytest.mark.parametrize(
    "payload",
    ["not json at all", "{}", '{"nodes": [], "links": []}', '{"a": 1}', ""],
)
def test_something_that_is_not_a_graph_is_not_read_as_one(payload):
    """The control. Without it a reader that returns a Recipe for anything
    would pass every test above."""
    from db import graph as graph_module

    assert graph_module.read(payload) is None


# --- choosing the moments of a video ----------------------------------------


def test_a_video_has_its_moments_chosen(db, a_library, tmp_path):
    """`derived_media_sample` had a producer and no caller.

    A face on a video is a face on a frame, and without a sample row "Ilse is
    in this video" cannot be checked, cropped or corrected, and a re-run
    cannot tell it has already done this part.
    """
    clip = tmp_path / "eleven.mp4"
    _write_clip(clip, seconds=11, rate=10, size=(160, 90))
    file_id = scan.mint(db, "file", "eleven")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'eleven.mp4', 'video', 10, 0, ?, ?)",
        (file_id, a_library["folder"], NOW, NOW),
    )

    chosen = sample.frames(db, file_id, clip)
    assert [offset for _, offset, _ in sample.taken(db, file_id)] == [
        0,
        2000,
        4000,
        6000,
        8000,
        10000,
    ]

    # an interrupted job resumes rather than raising on a frame it already took
    assert sample.frames(db, file_id, clip) == chosen
    assert db.execute("SELECT count(*) FROM derived_media_sample").fetchone()[0] == 6

    # a different cadence is a different set, side by side: a job that sampled
    # every two seconds and one that sampled every five did not look at the
    # same video
    sample.frames(db, file_id, clip, every_ms=5000)
    assert {p for _, _, p in sample.taken(db, file_id)} == {"every-2s", "every-5s"}


def test_a_long_film_is_sampled_across_its_length_not_truncated():
    """Truncating at a cap would sample the first hour of a three-hour film
    and call the rest unexamined. Widening samples all of it, less finely,
    and the policy token says which it was."""
    offsets, spacing = sample.moments(3 * 3600.0)
    assert len(offsets) <= sample.MOST
    assert offsets[-1] / 1000 > 3 * 3600 * 0.99, "the end of the film was never looked at"
    assert spacing > sample.EVERY_MS
    assert sample.cadence(spacing) == f"every-{spacing}ms"


def test_a_still_picture_has_no_moments():
    """The control: a sampler that returns something for everything would
    grow a frame row per photograph."""
    assert sample.moments(None)[0] == []
    assert sample.moments(0)[0] == []


# --- what a real detector actually hands back -------------------------------


def test_a_detectors_own_numbers_can_be_stored(db, a_library):
    """Every face in every other test was placed here by hand, as a Python
    literal. A detector does not return Python literals.

    sqlite3 binds an object it does not recognise through the buffer
    protocol, which a numpy scalar supports, so `np.float32(0.98)` arrives as
    a BLOB -- an IntegrityError against a STRICT table on the very first
    face, and against a lax one it would be stored as bytes and read back as
    garbage. It works until it doesn't: `np.float64` subclasses Python's
    float and `np.int32` does not, so a detector reporting doubles stored
    fine and the same code reporting float32 -- which is what every ONNX face
    model returns -- could not write a single row.
    """
    numpy = pytest.importorskip("numpy")
    file_id = a_library["file"]

    box = numpy.array([256, 128, 512, 384], dtype=numpy.int32)
    region_id = derived.region_from_pixels(db, box, numpy.int32(1024), numpy.int32(768))
    derived.record_faces(
        db,
        file_id,
        "yunet",
        "2023mar",
        "aa",
        NOW,
        [
            {
                "region": region_id,
                "det_score": numpy.float32(0.987),
                # `dim` is not passed: it describes the vector, so it is taken
                # from it. Passing one without an embedding used to be accepted
                # and describes nothing.
                "embedding": numpy.random.rand(128).astype(numpy.float32).tobytes(),
                "age": numpy.int32(34),
                "landmarks": numpy.array([[1.5, 2.5]], dtype=numpy.float32).tobytes(),
                "pose": (numpy.float32(1.5), numpy.float32(-2.0), numpy.float32(0.25)),
            }
        ],
    )
    derived.record_embedding(
        db,
        file_id,
        similarity_module.semantic_space("clip", "v1", 8),
        numpy.random.rand(8).astype(numpy.float32),
        "aa",
        NOW,
    )
    derived.record_hash(db, file_id, "aa", NOW, phash64=numpy.int64(-42))
    derived.annotate(
        db,
        file_id,
        "caption",
        "a brass helmet",
        "qwen-vl",
        "2.5",
        "aa",
        NOW,
        confidence=numpy.float32(0.75),
    )
    derived.add_sample(db, file_id, "frame", "every-2s", offset_ms=numpy.int64(4000))

    score, dim, age, yaw = db.execute("SELECT det_score, dim, age, pose_yaw FROM derived_face_instance").fetchone()
    assert score == pytest.approx(0.987, abs=1e-6)
    assert (dim, age, yaw) == (128, 34, 1.5)
    assert db.execute("SELECT length(vector) / 4 FROM derived_embedding").fetchone()[0] == 8
    assert db.execute("SELECT value FROM derived_file_hash").fetchone()[0] == -42
    assert db.execute("SELECT confidence FROM derived_annotation").fetchone()[0] == 0.75
    assert db.execute("SELECT offset_ms FROM derived_media_sample").fetchone()[0] == 4000

    # every one of those is a real number in the column, not bytes
    kinds = db.execute(
        "SELECT typeof(det_score), typeof(dim), typeof(age), typeof(pose_yaw)  FROM derived_face_instance"
    ).fetchone()
    assert kinds == ("real", "integer", "integer", "real"), kinds


def test_the_coercion_leaves_ordinary_values_alone(db, a_library):
    """The control: a coercion that mangled Python's own types would be worse
    than the problem, and one that quietly swallowed a real error would hide
    the next defect at this seam."""
    assert derived.plain(None) is None
    assert derived.plain(5) == 5
    assert isinstance(derived.plain(5), int)
    assert derived.plain(1.5) == 1.5
    assert derived.plain("euler") == "euler"
    assert derived.plain(b"\x00\x01") == b"\x00\x01"
    # something with no .item() passes through untouched, so a genuinely
    # unstorable value still fails loudly at the bind rather than here
    marker = object()
    assert derived.plain(marker) is marker


# --- documents --------------------------------------------------------------


def a_document(path, pages=3, **meta):
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(612, 792)
    if meta:
        writer.add_metadata(meta)
    with open(path, "wb") as handle:
        writer.write(handle)
    return path


def _ingest_as(db, a_library, path, kind):
    file_id = scan.mint(db, "file", path.stem)
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, ?, ?, 10, 0, ?, ?)",
        (file_id, a_library["folder"], path.name, kind, NOW, NOW),
    )
    return file_id, ingest.one(db, file_id, path, NOW)


def test_a_document_knows_how_long_it_is(db, a_library, tmp_path):
    """The last kind that could not state its own length.

    `.pdf` is a kind the scanner recognises, so a document that cannot say
    how many pages it has is exactly the hole a video had before ffprobe was
    wired in -- and 'page' was a value in derived_media_sample's CHECK that
    nothing could ever write.
    """
    path = a_document(tmp_path / "manual.pdf", pages=3, **{"/Title": "The Diving Manual", "/Author": "Ilse"})
    file_id, result = _ingest_as(db, a_library, path, "document")

    assert result.probed, result.unreadable
    assert result.unreadable is None
    assert db.execute("SELECT width, height FROM file WHERE id = ?", (file_id,)).fetchone() == (612, 792)
    fields = dict(
        db.execute(
            "SELECT key, value_text FROM file_param WHERE file_id = ? AND source = 'container'",
            (file_id,),
        )
    )
    assert fields["Pages"] == "3"
    assert fields["Title"] == "The Diving Manual"
    assert fields["Author"] == "Ilse"


def test_every_page_of_a_document_is_somewhere_to_point(db, a_library, tmp_path):
    """OCR from page nine is not OCR of the file. A page sample is where a
    caption or a piece of read text attaches, exactly as a frame is for a
    video -- and re-ingesting must resume rather than raise on page one."""
    path = a_document(tmp_path / "manual.pdf", pages=4)
    file_id, _ = _ingest_as(db, a_library, path, "document")

    assert [
        r[0]
        for r in db.execute(
            "SELECT page_index FROM derived_media_sample WHERE file_id = ? AND kind = 'page' ORDER BY page_index",
            (file_id,),
        )
    ] == [0, 1, 2, 3]

    ingest.one(db, file_id, path, NOW)
    assert db.execute("SELECT count(*) FROM derived_media_sample WHERE kind = 'page'").fetchone()[0] == 4


def test_a_document_nothing_can_open_costs_only_that_document(db, a_library, tmp_path):
    """A library is full of files nobody validated. One truncated PDF must
    report why rather than ending the scan around it."""
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 and then nothing at all")
    file_id, result = _ingest_as(db, a_library, path, "document")

    assert not result.probed
    assert result.unreadable, "a document nothing could read said nothing about why"
    assert db.execute("SELECT width, height FROM file WHERE id = ?", (file_id,)).fetchone() == (None, None)
    assert db.execute("SELECT count(*) FROM derived_media_sample").fetchone()[0] == 0


def test_a_pdf_is_not_read_for_camera_tags(db, a_library, tmp_path):
    """Running the camera reader over everything that is not a video meant
    `unreadable` said "Pillow cannot open this" for every PDF in the library
    -- true, uninteresting, and it buried the message from the reader that
    could open it."""
    path = a_document(tmp_path / "quiet.pdf", pages=1)
    file_id, result = _ingest_as(db, a_library, path, "document")

    assert result.unreadable is None
    assert db.execute("SELECT count(*) FROM capture WHERE file_id = ?", (file_id,)).fetchone()[0] == 0


# --- a real detector, all the way through -----------------------------------


def a_detectable_face(path, size=200):
    """An image OpenCV's own cascade actually fires on.

    Drawn rather than committed: a binary fixture is a thing nobody can read
    in a diff, and the point here is not this face but that a real detector's
    output survives the trip. The proportions are what the cascade wants --
    a light ground, a darker oval, two dark eyes above a nose shadow.
    """
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")

    img = numpy.full((size, size), 200, numpy.uint8)
    middle = size // 2
    cv2.ellipse(img, (middle, middle), (size // 3, int(size * 0.42)), 0, 0, 360, 170, -1)
    cv2.ellipse(img, (middle - size // 8, middle - size // 10), (size // 14, size // 22), 0, 0, 360, 60, -1)
    cv2.ellipse(img, (middle + size // 8, middle - size // 10), (size // 14, size // 22), 0, 0, 360, 60, -1)
    cv2.ellipse(img, (middle, middle + size // 12), (size // 20, size // 14), 0, 0, 360, 140, -1)
    cv2.ellipse(img, (middle, middle + size // 4), (size // 8, size // 30), 0, 0, 180, 90, -1)
    cv2.ellipse(img, (middle, middle - int(size * 0.36)), (size // 3, size // 8), 0, 180, 360, 90, -1)
    cv2.imwrite(str(path), img)
    return path


def detect(path):
    """What a real detector hands back: pixel boxes as numpy int32."""
    cv2 = pytest.importorskip("cv2")

    grey = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    boxes, _, weights = cascade.detectMultiScale3(
        grey, scaleFactor=1.05, minNeighbors=1, minSize=(20, 20), outputRejectLevels=True
    )
    return grey.shape[1], grey.shape[0], boxes, weights


def test_a_real_detectors_output_reaches_the_people_page(db, a_library, tmp_path):
    """The pipeline end to end, with nothing placed by hand.

    Every other face in this suite is a Python literal I wrote. This one is
    whatever OpenCV's cascade says, in the types it says it in, and it has to
    survive being stored, clustered, named and read back off a page. The
    detector does not matter -- a bundled Haar cascade is not the one this
    app will ship -- what matters is that no step of the journey was ever
    made with a real one.
    """
    picture = a_detectable_face(tmp_path / "portrait.png")
    width, height, boxes, weights = detect(picture)
    assert len(boxes), "the cascade found nothing, so this proves nothing"

    file_id = scan.mint(db, "file", "portrait")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256,"
        " first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'portrait.png', 'image', 10, 0, 'aa', ?, ?)",
        (file_id, a_library["folder"], NOW, NOW),
    )
    written = derived.record_faces(
        db,
        file_id,
        "opencv/haar",
        "frontalface_default",
        "aa",
        NOW,
        [
            {"region": derived.region_from_pixels(db, box, width, height), "det_score": min(1.0, float(weight) / 10.0)}
            for box, weight in zip(boxes, weights, strict=True)
        ],
    )
    assert len(written) == len(boxes)

    # stored as numbers, not as the BLOBs a numpy scalar binds to by default
    assert db.execute("SELECT DISTINCT typeof(det_score), typeof(region_id) FROM derived_face_instance").fetchall() == [
        ("real", "integer")
    ]
    # and inside the frame, which the region CHECK would have refused otherwise
    assert db.execute("SELECT count(*) FROM region").fetchone()[0] == len(boxes)

    person = authored.person(db, "Ilse", NOW)
    run = derived.run_for(db, "opencv/haar", "frontalface_default", "given", None, NOW)
    cluster = derived.recluster(db, "opencv/haar", "frontalface_default", NOW, [{"person_id": person}])[0]
    for face_id in written:
        derived.assign_cluster(db, face_id, cluster)
    derived.attribute(
        db,
        file_id,
        person,
        run,
        "opencv/haar",
        "frontalface_default",
        face_count=len(written),
    )
    derived.make_primary(db, run)

    from db import pages

    assert pages.people_by_most(db) == [("Ilse", "ilse", 1)]
    assert pages.person_files(db, person) == [("portrait", "portrait.png")]


def test_the_rebuild_contract_holds_on_real_detections(db, a_library, tmp_path):
    """Drop every derived table, run the detector again, and the human's
    name comes back -- on boxes a model chose rather than ones I placed
    where the assertion already was."""
    picture = a_detectable_face(tmp_path / "portrait.png")
    width, height, boxes, _ = detect(picture)
    file_id = scan.mint(db, "file", "portrait")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256,"
        " first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'portrait.png', 'image', 10, 0, 'aa', ?, ?)",
        (file_id, a_library["folder"], NOW, NOW),
    )
    person = authored.person(db, "Ilse", NOW)
    authored.assert_person(
        db,
        person,
        file_id,
        a_library["user"],
        NOW,
        region_id=derived.region_from_pixels(db, boxes[0], width, height),
    )

    derived.drop_all(db)
    rebuilt = derived.recluster(db, "opencv/haar", "v2", NOW + 1, [{}])[0]
    derived.record_faces(
        db,
        file_id,
        "opencv/haar",
        "v2",
        "aa",
        NOW + 1,
        [{"region": derived.region_from_pixels(db, box, width, height)} for box in boxes],
    )
    for face_id in db.execute("SELECT id FROM derived_face_instance WHERE model_version = 'v2'").fetchall():
        derived.assign_cluster(db, face_id[0], rebuilt)

    run = derived.run_for(db, "opencv/haar", "v2", "given", None, NOW + 1)
    assert derived.seed_clusters_from_assertions(db, run) == 1
    assert (
        db.execute("SELECT p.name FROM derived_face_cluster c JOIN person p ON p.id = c.person_id").fetchone()[0]
        == "Ilse"
    )


def test_an_embedding_that_does_not_fit_its_space_is_refused(db, a_library):
    """The vector column carries no dim of its own -- the space row owns
    the dimensions, and the schema holds every vector to them."""
    import numpy as np

    spec = similarity_module.semantic_space("clip", "v1", 8)
    with pytest.raises(sqlite3.IntegrityError, match="dimensions"):
        derived.record_embedding(db, a_library["file"], spec, np.zeros(4, dtype=np.float32), "aa", NOW)


def test_a_root_is_a_library_or_the_trash_and_nothing_else(db, tmp_path):
    """One media kind, and `trash`.

    There was a `mount` beside `library` and nothing anywhere branched on
    the difference: every read that cared spelled
    `kind IN ('library','mount')`. It reached the person as a dropdown on
    the add-a-folder form -- a choice that changed nothing, offered to
    somebody with no way to know that, at the moment they were trying to
    add their photographs.

    What it was reaching for is the test above this one: `online`, which
    is per-root, set by probing, and what the whole deletion doctrine
    rests on.
    """
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError, match="kind"):
        library.add_root(db, tmp_path / "elsewhere", "mount", NOW)


def test_the_form_does_not_ask_which_kind(tmp_path):
    """And the choice is gone from where it was asked."""
    from litestar.testing import TestClient

    from sg_web.app import build_app

    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        page = client.get("/operations", headers={"accept": "text/html"}).text
        assert "data-add-root" in page, "the control: the form is on the page"
        assert '<option value="mount">' not in page
        assert '<select name="kind"' not in page, "it still asks a question with one answer"
