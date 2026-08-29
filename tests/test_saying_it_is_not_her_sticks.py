"""A correction that survives the next clustering run.

The positive claim was the best thing in this schema: a person says who
is in a picture, and `seed_clusters_from_assertions` re-applies it after
every rebuild rather than re-guessing by centroid similarity. The
negative claim had nothing at all.

`retract_person` DELETES, and deleting means "I take that back" -- so
the next run is free to decide the same thing again, because nothing
recorded that it was wrong. A false merge stayed a chore somebody
repeated after every re-run, and this is what makes correcting it a
thing you do once.

Three distinct acts, and keeping them apart is the whole design:

    assert     she is in this picture      -- votes for a cluster
    deny       that is not her             -- REFUSES one, durably
    retract    I never said either way     -- leaves no record at all

The third is not the second. That is the difference the feature turns
on, and the test that pins it is the one that re-runs the clustering.
"""

from __future__ import annotations

import contextlib
import uuid

import numpy as np
import pytest

from db import authored, derived
from tests.staging import NOW, fresh_schema, hosting

pytestmark = pytest.mark.slow

MODEL = ("m", "1")


@pytest.fixture(scope="module")
def _world(tmp_path_factory):
    with hosting(tmp_path_factory, "test_saying_it_is_not_her_sticks") as stage:
        yield stage


@pytest.fixture
def served(_world):
    """One application for every served claim in this file. Each test
    writes its own pictures and registers them as root 1 over a restored
    empty home."""
    _world.restore()
    return _world.client


@pytest.fixture
def library():
    """Two pictures, one face each, embedded close enough to cluster."""
    conn = fresh_schema()
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'C:/x','library',0)")
    conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(1,?,'folder','x')", (uuid.uuid4().bytes,))
    conn.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(1,1,NULL,'x',0)")
    conn.execute("INSERT INTO user(id,username,password_hash,role,created_at) VALUES(1,'me','x','USER',0)")
    for at in (2, 3):
        conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(?,?,'file',?)", (at, uuid.uuid4().bytes, f"f{at}"))
        conn.execute(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,content_sha256,first_seen_at,last_seen_at)"
            " VALUES(?,1,?,'image',1,0,?,0,0)",
            (at, f"f{at}.png", f"{at:064d}"),
        )
        derived.record_faces(
            conn,
            at,
            MODEL[0],
            MODEL[1],
            f"{at:064d}",
            NOW,
            [{"region": derived.region(conn, 0.1, 0.1, 0.3, 0.3), "embedding": np.ones(4, np.float32).tobytes()}],
        )
    conn.commit()
    yield conn
    conn.close()


def _cluster(conn) -> int:
    """Run the clustering the way the job does, and return its run."""
    pinned = derived.threshold_for(MODEL[0])
    derived.cluster(conn, MODEL[0], MODEL[1], NOW, method=derived.DEFAULT_METHOD, threshold=pinned)
    run_id = derived.run_for(conn, MODEL[0], MODEL[1], derived.DEFAULT_METHOD, pinned, NOW)
    conn.execute("DELETE FROM derived_file_person WHERE run_id = ?", (run_id,))
    derived.seed_clusters_from_assertions(conn, run_id)
    conn.execute(
        "INSERT OR IGNORE INTO derived_file_person(file_id, person_id, run_id, model_id, model_version)"
        " SELECT fi.file_id, c.person_id, c.run_id, c.model_id, c.model_version"
        "   FROM derived_face_membership m"
        "   JOIN derived_face_instance fi ON fi.id = m.face_id"
        "   JOIN derived_face_cluster c ON c.id = m.cluster_id"
        "  WHERE c.person_id IS NOT NULL AND c.run_id = ?"
        "    AND NOT EXISTS (SELECT 1 FROM person_assertion pa"
        "                     WHERE pa.person_id = c.person_id AND pa.file_id = fi.file_id"
        "                       AND pa.stance = 'is_not')",
        (run_id,),
    )
    conn.commit()
    return run_id


def _named_clusters(conn, run_id) -> set[int]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT person_id FROM derived_face_cluster WHERE run_id = ? AND person_id IS NOT NULL", (run_id,)
        )
    }


def test_an_assertion_names_the_cluster_as_it_always_did(library):
    """The control. Without it the denial tests below prove nothing --
    a cluster that was never going to be named is not evidence that
    something refused it."""
    who = authored.person(library, "Hannah", NOW)
    authored.assert_person(library, who, 2, 1, NOW)
    library.commit()
    run_id = _cluster(library)
    assert _named_clusters(library, run_id) == {who}


def test_a_denial_refuses_the_name(library):
    who = authored.person(library, "Hannah", NOW)
    authored.assert_person(library, who, 2, 1, NOW)
    authored.deny_person(library, who, 3, 1, NOW)
    library.commit()
    run_id = _cluster(library)
    held = library.execute(
        "SELECT file_id FROM derived_file_person WHERE person_id = ? AND run_id = ?", (who, run_id)
    ).fetchall()
    assert [row[0] for row in held] == [2], "the denied picture kept the name"


def test_the_denial_survives_the_rebuild_and_a_retraction_does_not(library):
    """The distinction the whole feature turns on.

    Both start from the same place -- a name on a picture somebody says
    is wrong -- and they differ in what the NEXT run is allowed to do. A
    retraction leaves no record, so the clustering decides it again; a
    denial is a record that stops it.
    """
    who = authored.person(library, "Hannah", NOW)
    authored.assert_person(library, who, 2, 1, NOW)
    authored.assert_person(library, who, 3, 1, NOW)
    library.commit()
    assert _named_clusters(library, _cluster(library)) == {who}

    # retract: the claim is gone, and the clustering is free to decide
    # the same thing again -- which it does, because the faces are alike
    authored.retract_person(library, who, 3)
    library.commit()
    run_id = _cluster(library)
    again = [
        row[0]
        for row in library.execute(
            "SELECT file_id FROM derived_file_person WHERE person_id = ? AND run_id = ?", (who, run_id)
        )
    ]
    assert 3 in again, "a retraction is not a denial; the run put it back, which is correct"

    # deny: a record, and it holds through the same re-run
    authored.deny_person(library, who, 3, 1, NOW)
    library.commit()
    run_id = _cluster(library)
    held = [
        row[0]
        for row in library.execute(
            "SELECT file_id FROM derived_file_person WHERE person_id = ? AND run_id = ?", (who, run_id)
        )
    ]
    assert held == [2], "the denial did not survive the rebuild"


def test_denying_takes_the_name_off_the_picture_now(library):
    """Not at the next clustering run, which may never come. The claim
    constrains the run; `derived_file_person` is what the page reads."""
    who = authored.person(library, "Hannah", NOW)
    authored.assert_person(library, who, 2, 1, NOW)
    authored.assert_person(library, who, 3, 1, NOW)
    library.commit()
    _cluster(library)
    assert library.execute("SELECT count(*) FROM derived_file_person WHERE file_id = 3").fetchone()[0] >= 1

    authored.deny_person(library, who, 3, 1, NOW)
    library.commit()
    assert (
        library.execute(
            "SELECT count(*) FROM derived_file_person WHERE file_id = 3 AND person_id = ?", (who,)
        ).fetchone()[0]
        == 0
    )


def test_saying_the_opposite_withdraws_what_was_said(library):
    """A person cannot both be and not be in one picture, so one row per
    pair and the newer statement wins."""
    who = authored.person(library, "Hannah", NOW)
    authored.assert_person(library, who, 2, 1, NOW)
    authored.deny_person(library, who, 2, 1, NOW)
    library.commit()
    held = library.execute("SELECT stance, count(*) FROM person_assertion GROUP BY stance").fetchall()
    assert held == [("is_not", 1)]

    authored.assert_person(library, who, 2, 1, NOW)
    library.commit()
    assert library.execute("SELECT stance FROM person_assertion").fetchall() == [("is",)]


def test_a_denial_is_not_evidence_that_the_clustering_agreed(library):
    """`agreement` measures the clustering against what people said. A
    denial says two faces are NOT one person, so counting it there would
    make the measure improve every time somebody corrected the thing it
    measures."""
    who = authored.person(library, "Hannah", NOW)
    authored.assert_person(library, who, 2, 1, NOW)
    library.commit()
    run_id = _cluster(library)
    before = derived.agreement(library, run_id)

    authored.deny_person(library, who, 3, 1, NOW)
    library.commit()
    assert derived.agreement(library, run_id)["held_together"] == before["held_together"]


def test_an_unspellable_stance_is_refused(library):
    who = authored.person(library, "Hannah", NOW)
    with pytest.raises(ValueError, match="'is' or 'is_not'"):
        authored.assert_person(library, who, 2, 1, NOW, stance="maybe")


def test_denying_one_picture_does_not_unname_the_others(library):
    """The mistake the broad veto made, kept out by a test.

    Both faces sit in ONE cluster, so refusing the cluster because one
    of its pictures was denied takes the name off the picture the denial
    was not about. "She is not in this picture" is a claim about a FILE;
    it acts through the attribution, on exactly the file it named.
    """
    who = authored.person(library, "Hannah", NOW)
    authored.assert_person(library, who, 2, 1, NOW)
    library.commit()
    assert _named_clusters(library, _cluster(library)) == {who}

    authored.deny_person(library, who, 3, 1, NOW)
    library.commit()
    run_id = _cluster(library)
    assert _named_clusters(library, run_id) == {who}, "one denied picture unnamed the cluster"
    held = [
        row[0]
        for row in library.execute(
            "SELECT file_id FROM derived_file_person WHERE person_id = ? AND run_id = ?", (who, run_id)
        )
    ]
    assert held == [2], "and the denied picture keeps the name off"


def test_denying_a_FACE_refuses_the_cluster_that_holds_it(library):
    """The other sentence, and the reason the two are told apart. "Not
    her, THAT one" is a claim about a face, and a cluster is what
    collects faces -- so it refuses the name for the whole group, which
    is what a false merge needs."""
    who = authored.person(library, "Hannah", NOW)
    authored.assert_person(library, who, 2, 1, NOW)
    library.commit()
    _cluster(library)

    box = library.execute("SELECT region_id FROM derived_face_instance WHERE file_id = 3").fetchone()[0]
    authored.deny_person(library, who, 3, 1, NOW, region_id=box)
    library.commit()
    assert _named_clusters(library, _cluster(library)) == set(), "the face was refused; the cluster is a question now"


# --- and it is sayable from the page ----------------------------------------


def test_the_page_offers_the_correction_beside_the_name(tmp_path, served):
    """The button, and where it is. This is where somebody is already
    looking at "Hannah" over a picture that is not Hannah -- the value
    of a denial is that it is said once and holds, so it must not be a
    page you have to go and find."""
    import time as clock

    from PIL import Image

    from db import connect, naming

    root = tmp_path / "pics"
    root.mkdir()
    Image.new("RGB", (16, 12), (30, 90, 140)).save(root / "one.png")
    with contextlib.nullcontext(served) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        conn = connect.connect(client.app.state.db_path)
        try:
            file_id = conn.execute("SELECT id FROM file").fetchone()[0]
            sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()[0]
            who = authored.person(conn, "Hannah", clock.time())
            derived.record_faces(
                conn,
                file_id,
                MODEL[0],
                MODEL[1],
                sha,
                clock.time(),
                [
                    {
                        "region": derived.region(conn, 0.1, 0.1, 0.3, 0.3),
                        "embedding": np.ones(4, np.float32).tobytes(),
                    }
                ],
            )
            run_id = derived.run_for(conn, MODEL[0], MODEL[1], derived.DEFAULT_METHOD, 0.5, clock.time())
            derived.attribute(conn, file_id, who, run_id, MODEL[0], MODEL[1], face_count=1)
            # The page shows the PRIMARY run's people and nothing else
            # (db/pages.py MEDIA_PEOPLE), so a run nobody chose renders an
            # empty "who" that reads as a missing button.
            derived.make_primary(conn, run_id)
            conn.commit()
            named = naming.entity_slug(conn, file_id)
            person_slug = naming.entity_slug(conn, who)
            assert named is not None
            assert person_slug is not None
        finally:
            connect.close(conn)

        page = client.get(f"/i/{named[1]}", headers={"accept": "text/html"}).text
        assert f'data-person-deny="{person_slug[1]}"' in page, "no way to say it from the page"

        told = client.post(f"/i/{named[1]}/people/{person_slug[1]}/deny", json={"value": True})
        assert told.status_code in (200, 201), told.text
        # answers with who the picture NOW holds, read from the database
        assert told.json()["people"] == []

        again = client.get(f"/i/{named[1]}", headers={"accept": "text/html"}).text
        assert f'data-person-deny="{person_slug[1]}"' not in again, "the name is still on the picture"


def test_the_persons_own_page_offers_it_over_every_picture(tmp_path, served):
    """And it is sayable from the page where the wrong picture is SEEN.

    The claim has been makeable since it existed, and reachable from
    exactly one place: the inspector of one picture, one picture at a
    time. But nobody notices a wrong face by opening pictures one by one.
    They notice it looking at a wall of somebody's photographs -- which
    is this page, and this page could only send them somewhere else to
    say so.

    So every cell here carries it, over the thumbnail, addressed by the
    picture's slug and the person whose page it is.
    """
    import time as clock

    from PIL import Image

    from db import connect, naming

    root = tmp_path / "pics"
    root.mkdir()
    for i in range(3):
        Image.new("RGB", (16, 12), (30 * i, 90, 140)).save(root / f"p{i}.png")
    with contextlib.nullcontext(served) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        conn = connect.connect(client.app.state.db_path)
        try:
            who = authored.person(conn, "Hannah", clock.time())
            run_id = derived.run_for(conn, MODEL[0], MODEL[1], derived.DEFAULT_METHOD, 0.5, clock.time())
            slugs = []
            for file_id, sha in conn.execute("SELECT id, content_sha256 FROM file ORDER BY id").fetchall():
                derived.record_faces(
                    conn,
                    file_id,
                    MODEL[0],
                    MODEL[1],
                    sha,
                    clock.time(),
                    [
                        {
                            "region": derived.region(conn, 0.1, 0.1, 0.3, 0.3),
                            "embedding": np.ones(4, np.float32).tobytes(),
                        }
                    ],
                )
                derived.attribute(conn, file_id, who, run_id, MODEL[0], MODEL[1], face_count=1)
                named = naming.entity_slug(conn, file_id)
                assert named is not None
                slugs.append(named[1])
            derived.make_primary(conn, run_id)
            conn.commit()
            person_slug = naming.entity_slug(conn, who)
            assert person_slug is not None
        finally:
            connect.close(conn)

        page = client.get(f"/p/{person_slug[1]}", headers={"accept": "text/html"}).text
        assert f'data-person-pictures="{person_slug[1]}"' in page, "the grid does not say whose it is"
        for slug in slugs:
            assert f'data-person-not-here="{slug}"' in page, f"no way to say it over {slug}"
            assert f'data-person-picture="{slug}"' in page

        # and saying it takes that picture off the person, leaving the others
        told = client.post(f"/i/{slugs[0]}/people/{person_slug[1]}/deny", json={"value": True})
        assert told.status_code in (200, 201), told.text
        again = client.get(f"/p/{person_slug[1]}", headers={"accept": "text/html"}).text
        assert f'data-person-picture="{slugs[0]}"' not in again, "the picture is still under this person"
        for slug in slugs[1:]:
            assert f'data-person-picture="{slug}"' in again, "denying one picture took the others too"


def test_withdrawing_does_not_put_the_name_back_by_itself(tmp_path, served):
    """What undo undoes, exactly -- because the page says so and the page
    must be right.

    Retracting deletes the record that the attribution was wrong, so the
    next clustering run is free to decide it again. It does NOT restore
    the attribution: no run has said so since, and inventing the row
    would be the browser making up derived state. The picture comes back
    to this page when clustering next names them in it, which is what
    the withdrawn cell says in those words.
    """
    import time as clock

    from PIL import Image

    from db import connect, naming

    root = tmp_path / "pics"
    root.mkdir()
    Image.new("RGB", (16, 12), (30, 90, 140)).save(root / "one.png")
    with contextlib.nullcontext(served) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        conn = connect.connect(client.app.state.db_path)
        try:
            file_id, sha = conn.execute("SELECT id, content_sha256 FROM file").fetchone()
            who = authored.person(conn, "Hannah", clock.time())
            derived.record_faces(
                conn,
                file_id,
                MODEL[0],
                MODEL[1],
                sha,
                clock.time(),
                [{"region": derived.region(conn, 0.1, 0.1, 0.3, 0.3), "embedding": np.ones(4, np.float32).tobytes()}],
            )
            run_id = derived.run_for(conn, MODEL[0], MODEL[1], derived.DEFAULT_METHOD, 0.5, clock.time())
            derived.attribute(conn, file_id, who, run_id, MODEL[0], MODEL[1], face_count=1)
            derived.make_primary(conn, run_id)
            conn.commit()
            named = naming.entity_slug(conn, file_id)
            person_slug = naming.entity_slug(conn, who)
            assert named is not None
            assert person_slug is not None
        finally:
            connect.close(conn)

        client.post(f"/i/{named[1]}/people/{person_slug[1]}/deny", json={"value": True})
        client.post(f"/i/{named[1]}/people/{person_slug[1]}/deny", json={"value": False})

        conn = connect.connect(client.app.state.db_path)
        try:
            assert authored.denials(conn) == [], "the claim was not withdrawn"
            still = conn.execute("SELECT count(*) FROM derived_file_person").fetchone()[0]
            assert still == 0, "withdrawing invented an attribution no clustering run made"
        finally:
            connect.close(conn)

        page = client.get(f"/p/{person_slug[1]}", headers={"accept": "text/html"}).text
        assert f'data-person-picture="{named[1]}"' not in page, (
            "the page shows a picture under this person that no run attributes to them"
        )


# --- and the correction is evidence about the model that made it -------------


def _corrections(conn) -> list[tuple[str, str, int, int]]:
    from db import verdicts

    return [(one.model_id, one.model_version, one.corrections, one.people) for one in verdicts.corrections(conn)]


def test_putting_a_face_right_is_recorded_against_the_model_that_got_it_wrong(library):
    """The whole point of collecting verdicts, for free.

    Somebody correcting a face is already doing the work; until now the
    only thing that learned from it was the next clustering run. The
    thing that would tell them their face model is bad -- the verdict
    aggregate -- never heard about the forty corrections they made.

    Named by the producer that was corrected, copied off the attribution
    before it is withdrawn, because after the delete nothing records
    which model's output this was about.
    """
    who = authored.person(library, "Hannah", NOW)
    run_id = _cluster(library)
    derived.attribute(library, 2, who, run_id, MODEL[0], MODEL[1], face_count=1)
    library.commit()

    assert _corrections(library) == [], "nothing corrected yet"
    authored.deny_person(library, who, 2, 1, NOW)
    library.commit()
    assert _corrections(library) == [(MODEL[0], MODEL[1], 1, 1)]


def test_denying_a_person_no_run_ever_named_judges_nothing(library):
    """The guard, and it is the one that keeps the number honest.

    Denying is a claim about the picture; it is only a JUDGEMENT when a
    model actually said the thing being denied. A denial over a file
    nothing attributed would otherwise put a correction against a
    producer that never spoke -- and the producer it happened to name
    would be whichever one ran last.
    """
    who = authored.person(library, "Hannah", NOW)
    library.commit()
    authored.deny_person(library, who, 2, 1, NOW)
    library.commit()
    assert _corrections(library) == []
    assert authored.denials(library) != [], "the claim itself is still recorded"


def test_taking_the_denial_back_takes_the_correction_back(library):
    """A verdict recorded because somebody denied a person is that
    denial's evidence. Retracting the denial and leaving the verdict
    would keep counting a mistake nobody says was one any more."""
    who = authored.person(library, "Hannah", NOW)
    run_id = _cluster(library)
    derived.attribute(library, 2, who, run_id, MODEL[0], MODEL[1], face_count=1)
    library.commit()
    authored.deny_person(library, who, 2, 1, NOW)
    library.commit()
    assert _corrections(library) == [(MODEL[0], MODEL[1], 1, 1)]

    authored.retract_person(library, who, 2)
    library.commit()
    assert _corrections(library) == []


def test_denying_the_same_picture_twice_is_one_correction(library):
    """One person has one opinion about one claim -- the rule
    `retract_feedback` already holds for a caption."""
    who = authored.person(library, "Hannah", NOW)
    run_id = _cluster(library)
    derived.attribute(library, 2, who, run_id, MODEL[0], MODEL[1], face_count=1)
    library.commit()
    authored.deny_person(library, who, 2, 1, NOW)
    library.commit()
    # the attribution is gone now, so the second denial judges nothing
    # and must not add to a count it did not earn
    authored.deny_person(library, who, 2, 1, NOW)
    library.commit()
    assert _corrections(library) == [(MODEL[0], MODEL[1], 1, 1)]


def test_the_correction_count_is_never_offered_as_a_rate(library):
    """The refusal, pinned. These verdicts are 100% `wrong` by
    construction -- nobody clicks "yes that is her" on a face that is
    simply right -- so there is no denominator and no share. A `Judged`
    would compute one; a `Corrected` cannot be asked."""
    from db import verdicts

    who = authored.person(library, "Hannah", NOW)
    run_id = _cluster(library)
    derived.attribute(library, 2, who, run_id, MODEL[0], MODEL[1], face_count=1)
    library.commit()
    authored.deny_person(library, who, 2, 1, NOW)
    library.commit()

    one = verdicts.corrections(library)[0]
    assert not hasattr(one, "wrong_share")
    assert not hasattr(one, "judged")
    # and it stays out of the rated table, where a reader would compare
    # a tally against a percentage
    assert verdicts.by_producer(library) == []


# --- and two people can be told they were always one -------------------------


def test_merging_moves_the_durable_claims(library):
    """The other correction, and the one the durable model was missing.

    Denying says "not them, in this picture". A clustering run splitting
    somebody into four is the ordinary failure, and a threshold cannot
    fix it without trading away somebody else's correct grouping. Said
    here it is local, permanent, and re-applied after every future run.
    """
    keep = authored.person(library, "Hannah", NOW)
    folded = authored.person(library, "Hanna", NOW)
    authored.assert_person(library, folded, 2, 1, NOW)
    authored.assert_person(library, folded, 3, 1, NOW)
    library.commit()

    told = authored.merge_people(library, keep, folded, 1, NOW)
    library.commit()

    assert told["assertions"] == 2
    held = {
        (person_id, file_id)
        for person_id, file_id in library.execute("SELECT person_id, file_id FROM person_assertion")
    }
    assert held == {(keep, 2), (keep, 3)}
    assert library.execute("SELECT count(*) FROM person WHERE id = ?", (folded,)).fetchone()[0] == 0


def test_a_merge_never_overrules_what_was_said_about_the_survivor(library):
    """A person who has said something about the one being kept said it
    about the one being kept, and a merge is not the moment to overrule
    them."""
    keep = authored.person(library, "Hannah", NOW)
    folded = authored.person(library, "Hanna", NOW)
    authored.deny_person(library, keep, 2, 1, NOW)
    authored.assert_person(library, folded, 2, 1, NOW)
    library.commit()

    authored.merge_people(library, keep, folded, 1, NOW)
    library.commit()

    stance = library.execute(
        "SELECT stance FROM person_assertion WHERE person_id = ? AND file_id = 2", (keep,)
    ).fetchone()[0]
    assert stance == "is_not", "the merge overwrote a denial somebody had made about the survivor"


def test_the_folded_address_still_answers(library):
    """A bookmark, a shared link or an exported document keeps working:
    `slug_history` already answers a retired slug with the entity that
    holds it now, so a merge is one more kind of retirement rather than
    a new sort of hole."""
    from db import naming

    keep = authored.person(library, "Hannah", NOW)
    folded = authored.person(library, "Hanna", NOW)
    authored.assert_person(library, folded, 2, 1, NOW)
    gone = naming.entity_slug(library, folded)
    assert gone is not None
    library.commit()

    authored.merge_people(library, keep, folded, 1, NOW)
    library.commit()

    found = naming.resolve(library, "person", gone[1])
    assert found is not None, "the folded person's address answers nothing"
    assert found[0] == keep
    assert found[1] is False, "it should redirect rather than serve"


def test_a_correction_survives_the_merge_it_was_made_before(library):
    """A verdict recorded against the folded person still counts against
    the model that earned it."""
    who = authored.person(library, "Hannah", NOW)
    folded = authored.person(library, "Hanna", NOW)
    run_id = _cluster(library)
    derived.attribute(library, 2, folded, run_id, MODEL[0], MODEL[1], face_count=1)
    library.commit()
    authored.deny_person(library, folded, 2, 1, NOW)
    library.commit()
    assert _corrections(library) == [(MODEL[0], MODEL[1], 1, 1)]

    authored.merge_people(library, who, folded, 1, NOW)
    library.commit()
    assert _corrections(library) == [(MODEL[0], MODEL[1], 1, 1)], "the correction was lost with the person"


def test_the_pictures_move_now_rather_than_after_a_rerun(library):
    """The same reason denying withdraws an attribution instead of only
    recording a claim: the page reads `derived_file_person`, and a merge
    that waited for a re-run would show a person still split."""
    keep = authored.person(library, "Hannah", NOW)
    folded = authored.person(library, "Hanna", NOW)
    run_id = _cluster(library)
    derived.attribute(library, 3, folded, run_id, MODEL[0], MODEL[1], face_count=1)
    library.commit()

    authored.merge_people(library, keep, folded, 1, NOW)
    library.commit()

    held = {row[0] for row in library.execute("SELECT person_id FROM derived_file_person")}
    assert held == {keep}


def test_a_person_cannot_be_merged_into_themselves(library):
    who = authored.person(library, "Hannah", NOW)
    library.commit()
    with pytest.raises(ValueError, match="themselves"):
        authored.merge_people(library, who, who, 1, NOW)


def test_the_person_page_offers_it(tmp_path, served):
    """Where the split is NOTICED: standing on one of them, looking at
    half their pictures."""
    import time as clock

    from db import connect, naming

    with contextlib.nullcontext(served) as client:
        conn = connect.connect(client.app.state.db_path)
        try:
            who = authored.person(conn, "Hannah", clock.time())
            conn.commit()
            slug = naming.entity_slug(conn, who)
            assert slug is not None
        finally:
            connect.close(conn)

        page = client.get(f"/p/{slug[1]}", headers={"accept": "text/html"}).text
        assert f'data-same-as="{slug[1]}"' in page, "no way to say two people are one"


# --- and the work a fresh clustering leaves behind --------------------------


def _served(client, tmp_path, howmany: int = 2):
    """A library with `howmany` pictures in it, under the module's
    application."""
    from PIL import Image

    root = tmp_path / "lib"
    root.mkdir()
    for i in range(howmany):
        Image.new("RGB", (16, 12), (10 * i, 90, 140)).save(root / f"p{i}.png")
    return client, root


def test_the_people_page_puts_the_unnamed_first_and_names_them_in_place(tmp_path, served):
    """Who is this?

    A run mints one placeholder person per group nobody has named, and
    they sorted into the same grid as everybody else -- so the people
    somebody HAD named were scattered through the ones they had not, and
    naming the rest meant finding them first.

    Named in place, because twelve people should cost twelve names and
    not twelve page loads.
    """
    import time as clock

    from db import connect, naming

    client, root = _served(served, tmp_path)
    with contextlib.nullcontext(client):
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        conn = connect.connect(client.app.state.db_path)
        try:
            known = authored.person(conn, "Hannah", clock.time())
            stranger = authored.person(conn, None, clock.time())
            run_id = derived.run_for(conn, MODEL[0], MODEL[1], derived.DEFAULT_METHOD, 0.5, clock.time())
            files = [one for (one,) in conn.execute("SELECT id FROM file ORDER BY id")]
            derived.attribute(conn, files[0], known, run_id, MODEL[0], MODEL[1], face_count=1)
            derived.attribute(conn, files[1], stranger, run_id, MODEL[0], MODEL[1], face_count=1)
            derived.make_primary(conn, run_id)
            conn.commit()
            named = naming.entity_slug(conn, known)
            unnamed = naming.entity_slug(conn, stranger)
            assert named is not None
            assert unnamed is not None
        finally:
            connect.close(conn)

        told = client.get("/people", headers={"accept": "application/json"}).json()
        held = {one["slug"]: one for one in told}
        assert held[unnamed[1]]["name"] is None, "a placeholder is named '(unnamed)' rather than being unnamed"
        assert held[named[1]]["name"] == "Hannah"

        page = client.get("/people", headers={"accept": "text/html"}).text
        assert f'data-unknown="{unnamed[1]}"' in page, "the unnamed group is not in the queue"
        assert f'data-unknown="{named[1]}"' not in page, "somebody already named is in the queue"
        # and the queue offers the name box beside the face
        assert f'data-person-rename="{unnamed[1]}"' in page


def test_a_person_with_no_clustered_face_is_not_pointed_at(tmp_path, served):
    """`/avatar/<slug>` answers 404 for somebody no run found a face
    for, which is a normal state -- so the index asks before it points
    rather than drawing a broken image on every card."""
    import time as clock

    from db import connect

    client, root = _served(served, tmp_path, 1)
    with contextlib.nullcontext(client):
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        conn = connect.connect(client.app.state.db_path)
        try:
            who = authored.person(conn, "Hannah", clock.time())
            run_id = derived.run_for(conn, MODEL[0], MODEL[1], derived.DEFAULT_METHOD, 0.5, clock.time())
            one = conn.execute("SELECT id FROM file").fetchone()[0]
            # attributed, but no FACE was clustered -- the avatar route
            # crops a detection and there is none
            derived.attribute(conn, one, who, run_id, MODEL[0], MODEL[1], face_count=1)
            derived.make_primary(conn, run_id)
            conn.commit()
        finally:
            connect.close(conn)

        told = client.get("/people", headers={"accept": "application/json"}).json()
        assert told, "the control: the person is on the index"
        assert told[0]["avatar"] is None, "the index points at an avatar that answers 404"
        page = client.get("/people", headers={"accept": "text/html"}).text
        assert "/avatar/" not in page, "the page points at one anyway"


# --- and whose face stands for them ------------------------------------------


def _a_picture(conn, at: int, slug: str) -> None:
    """A file nothing has looked at: no faces, no derived rows."""
    conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(?,?,'file',?)", (at, bytes([at]) * 16, slug))
    conn.execute(
        "INSERT INTO file(id,folder_id,name,kind,size,mtime,content_sha256,first_seen_at,last_seen_at)"
        " VALUES(?,1,?,'image',1,0,?,0,0)",
        (at, f"{slug}.png", f"{at:064d}"),
    )


def test_choosing_a_face_names_a_picture_not_a_face(library):
    """The durability the other three corrections have, and the reason
    this one stores what it does.

    The avatar is cropped from a `derived_face_instance`, and every one
    of those is deleted by `derived.drop_all` and minted afresh by the
    next detection. Remembering the FACE would remember something the
    next re-detect destroys; remembering the PICTURE survives it, which
    is the same trade `person_assertion` makes by naming a file and a
    region rather than a cluster.
    """
    who = authored.person(library, "Hannah", NOW)
    authored.choose_face(library, who, 3)
    library.commit()

    held = library.execute("SELECT exemplar_file_id FROM person WHERE id = ?", (who,)).fetchone()[0]
    assert held == 3
    columns = {row[1] for row in library.execute("PRAGMA table_info(person)")}
    assert "exemplar_face_id" not in columns, "a face id would dangle at the next re-detect"


def test_the_avatar_comes_from_the_chosen_picture(library):
    """What it is for. Two faces of one person; the chosen picture wins
    even when the other is the more confident detection."""
    from sg_web import media

    who = authored.person(library, "Hannah", NOW)
    # Asserted, then clustered: `exemplar_face` crops a real detection,
    # so the cluster has to hold faces rather than the file merely being
    # attributed.
    authored.assert_person(library, who, 2, 1, NOW)
    library.commit()
    run_id = _cluster(library)
    derived.make_primary(library, run_id)
    library.commit()

    automatic = media.exemplar_face(library, who)
    assert automatic is not None, "the control: there is a face to take"

    every = {
        one
        for (one,) in library.execute(
            "SELECT DISTINCT fi.file_id FROM derived_face_membership m"
            " JOIN derived_face_instance fi ON fi.id = m.face_id"
            " JOIN derived_face_cluster c ON c.id = m.cluster_id"
            " WHERE c.person_id = ?",
            (who,),
        )
    }
    other = next(one for one in sorted(every) if one != automatic[1])
    authored.choose_face(library, who, other)
    library.commit()
    chosen = media.exemplar_face(library, who)
    assert chosen is not None
    assert chosen[1] == other, "the chosen picture did not win"


def test_a_chosen_picture_that_holds_no_face_falls_back(library):
    """Rather than failing. A picture re-detected, cropped or replaced
    may no longer hold a face of theirs, and somebody with no avatar
    BECAUSE they expressed a preference is worse off than somebody who
    never expressed one."""
    from sg_web import media

    who = authored.person(library, "Hannah", NOW)
    authored.assert_person(library, who, 2, 1, NOW)
    library.commit()
    run_id = _cluster(library)
    derived.make_primary(library, run_id)
    library.commit()
    automatic = media.exemplar_face(library, who)
    assert automatic is not None

    _a_picture(library, 90, "elsewhere")
    authored.choose_face(library, who, 90)
    library.commit()
    held = media.exemplar_face(library, who)
    assert held is not None, "expressing a preference took their face away"
    assert held[1] == automatic[1]


def test_clearing_it_is_the_automatic_choice_not_no_choice(library):
    from sg_web import media

    who = authored.person(library, "Hannah", NOW)
    authored.assert_person(library, who, 2, 1, NOW)
    library.commit()
    run_id = _cluster(library)
    derived.make_primary(library, run_id)
    authored.choose_face(library, who, 2)
    library.commit()

    authored.choose_face(library, who, None)
    library.commit()
    assert media.exemplar_face(library, who) is not None


def test_deleting_the_picture_leaves_the_person(library):
    """SET NULL: deleting a picture is not a statement about a person,
    so they go back to the automatic choice rather than disappearing
    with it."""
    who = authored.person(library, "Hannah", NOW)
    # A picture with nothing else pointing at it, so this is about the
    # person's column and not about somebody else's cascade.
    _a_picture(library, 91, "going")
    authored.choose_face(library, who, 91)
    library.commit()

    library.execute("PRAGMA foreign_keys = ON")
    library.execute("DELETE FROM file WHERE id = 91")
    library.commit()

    held = library.execute("SELECT exemplar_file_id FROM person WHERE id = ?", (who,)).fetchone()
    assert held is not None, "the person went with the picture"
    assert held[0] is None
