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

import uuid

import numpy as np
import pytest

from db import authored, derived
from tests.staging import fresh_schema

pytestmark = pytest.mark.slow

NOW = 1_700_000_000.0
MODEL = ("m", "1")


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
