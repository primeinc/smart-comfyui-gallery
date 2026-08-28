"""You can say a caption is wrong, and the saying survives the rebuild.

The `feedback` table was designed for this and nothing but a test ever
wrote it: judge an annotation, a similarity, a duplicate or a person,
right / wrong / unsure, with pointers that are ON DELETE SET NULL on
purpose so dropping the whole derived namespace leaves the judgement
standing. It is the one authored table whose subject is disposable.

Two decisions make it useful rather than decorative.

The thumb sits WHERE THE CLAIM IS SHOWN. A review queue is a chore
nobody does; the inspector is where somebody is already reading the
sentence and already knows whether it is right. One click, no dialog,
and clicking the lit one takes it back.

And a verdict names the PRODUCER it judged, copied rather than
referenced. The table already recorded `annotation_kind` rather than the
annotation's row, so the judgement outlives a rebuild -- but it could
not say WHICH model wrote the thing judged, and "this caption model gets
12% of my library wrong" is the reason to collect verdicts at all. A
foreign key would be wrong in both directions: CASCADE deletes the
human's words with the machine's, SET NULL erases the only thing that
makes the verdict aggregable afterwards.
"""

from __future__ import annotations

import uuid

import pytest

from db import authored, derived
from tests.staging import NOW, fresh_schema

pytestmark = pytest.mark.slow

MODEL = ("Salesforce/blip-image-captioning-base", "main")


@pytest.fixture
def library():
    conn = fresh_schema()
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'C:/x','library',0)")
    conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(1,?,'folder','x')", (uuid.uuid4().bytes,))
    conn.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(1,1,NULL,'x',0)")
    conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(2,?,'file','pic')", (uuid.uuid4().bytes,))
    conn.execute(
        "INSERT INTO file(id,folder_id,name,kind,size,mtime,content_sha256,first_seen_at,last_seen_at)"
        " VALUES(2,1,'pic.png','image',1,0,'abc',0,0)"
    )
    conn.execute("INSERT INTO user(id,username,password_hash,role,created_at) VALUES(1,'me','x','USER',0)")
    derived.annotate(conn, 2, "caption", "a dog on a beach", MODEL[0], MODEL[1], "abc", NOW)
    conn.commit()
    yield conn
    conn.close()


def _verdict(conn, actor=1):
    return authored.standing_verdict(conn, 2, "caption", MODEL[0], MODEL[1], actor)


def test_a_verdict_is_recorded_against_the_producer_that_earned_it(library):
    authored.feedback(
        library,
        "annotation",
        "wrong",
        NOW,
        file_id=2,
        annotation_kind="caption",
        user_id=1,
        model_id=MODEL[0],
        model_version=MODEL[1],
    )
    library.commit()
    held = library.execute("SELECT model_id, model_version, annotation_kind FROM feedback").fetchone()
    assert held == (MODEL[0], MODEL[1], "caption")
    assert _verdict(library) == "wrong"


def test_it_survives_the_derived_layer_being_rebuilt(library):
    """The whole reason it is a copy. A re-run deletes every annotation
    and mints new ones; the judgement is about what a MODEL said, and it
    still names the model afterwards."""
    authored.feedback(
        library,
        "annotation",
        "wrong",
        NOW,
        file_id=2,
        annotation_kind="caption",
        user_id=1,
        model_id=MODEL[0],
        model_version=MODEL[1],
    )
    library.commit()

    library.execute("DELETE FROM derived_annotation")
    library.commit()
    assert library.execute("SELECT count(*) FROM derived_annotation").fetchone()[0] == 0

    held = library.execute("SELECT verdict, model_id, annotation_kind, file_id FROM feedback").fetchone()
    assert held == ("wrong", MODEL[0], "caption", 2), "the judgement went with the thing it judged"
    assert _verdict(library) == "wrong", "and is still findable by what it was about"


def test_changing_your_mind_leaves_one_standing_opinion(library):
    for said in ("right", "wrong", "unsure"):
        authored.retract_feedback(library, 2, "caption", MODEL[0], MODEL[1], 1)
        authored.feedback(
            library,
            "annotation",
            said,
            NOW,
            file_id=2,
            annotation_kind="caption",
            user_id=1,
            model_id=MODEL[0],
            model_version=MODEL[1],
        )
    library.commit()
    assert _verdict(library) == "unsure"
    assert library.execute("SELECT count(*) FROM feedback").fetchone()[0] == 1


def test_taking_it_back_leaves_no_row(library):
    """Clicking the lit thumb means "I take that back", and the honest
    record of that is no row -- a verdict of "none" would be a third
    opinion nobody expressed."""
    authored.feedback(
        library,
        "annotation",
        "right",
        NOW,
        file_id=2,
        annotation_kind="caption",
        user_id=1,
        model_id=MODEL[0],
        model_version=MODEL[1],
    )
    library.commit()
    assert authored.retract_feedback(library, 2, "caption", MODEL[0], MODEL[1], 1) == 1
    library.commit()
    assert _verdict(library) is None
    assert library.execute("SELECT count(*) FROM feedback").fetchone()[0] == 0


def test_two_producers_are_judged_apart(library):
    """The comparison this exists to support: two caption models over
    one library, judged over the same pictures."""
    derived.annotate(library, 2, "caption", "a beach", "other/model", "v2", "abc", NOW)
    authored.feedback(
        library,
        "annotation",
        "wrong",
        NOW,
        file_id=2,
        annotation_kind="caption",
        user_id=1,
        model_id=MODEL[0],
        model_version=MODEL[1],
    )
    authored.feedback(
        library,
        "annotation",
        "right",
        NOW,
        file_id=2,
        annotation_kind="caption",
        user_id=1,
        model_id="other/model",
        model_version="v2",
    )
    library.commit()
    assert _verdict(library) == "wrong"
    assert authored.standing_verdict(library, 2, "caption", "other/model", "v2", 1) == "right"

    # and the aggregate the whole thing is for is one GROUP BY away
    told = dict(
        library.execute("SELECT model_id, count(*) FROM feedback WHERE verdict = 'wrong' GROUP BY model_id").fetchall()
    )
    assert told == {MODEL[0]: 1}


def test_a_verdict_from_before_the_producer_columns_is_still_found(library):
    """`IS`, not `=`. A row written before v34 holds NULL there, and
    `= NULL` is never true -- so an old judgement would be invisible and
    the control would open blank over one that exists."""
    authored.feedback(library, "annotation", "wrong", NOW, file_id=2, annotation_kind="caption", user_id=1)
    library.commit()
    assert authored.standing_verdict(library, 2, "caption", None, None, 1) == "wrong"


def test_one_person_verdict_is_not_another_person(library):
    """It is one actor's judgement, like a rating."""
    library.execute("INSERT INTO user(id,username,password_hash,role,created_at) VALUES(2,'you','x','USER',0)")
    authored.feedback(
        library,
        "annotation",
        "wrong",
        NOW,
        file_id=2,
        annotation_kind="caption",
        user_id=1,
        model_id=MODEL[0],
        model_version=MODEL[1],
    )
    library.commit()
    assert _verdict(library, actor=1) == "wrong"
    assert _verdict(library, actor=2) is None


# --- through the seam a person actually uses --------------------------------


def test_the_route_records_retracts_and_says_what_it_now_holds(tmp_path):
    """One click, and the answer is what the SERVER holds afterwards --
    never an echo of what was clicked. The two differ on a retraction,
    and a control drawn from the click would then lie about its own
    state."""
    import time as clock

    from litestar.testing import TestClient
    from PIL import Image

    from db import connect, naming
    from sg_web.app import build_app

    root = tmp_path / "pics"
    root.mkdir()
    Image.new("RGB", (16, 12), (30, 90, 140)).save(root / "one.png")
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        conn = connect.connect(client.app.state.db_path)
        try:
            file_id = conn.execute("SELECT id FROM file").fetchone()[0]
            sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()[0]
            derived.annotate(conn, file_id, "caption", "a blue square", MODEL[0], MODEL[1], sha, clock.time())
            conn.commit()
            named = naming.entity_slug(conn, file_id)
            assert named is not None
            slug = named[1]
        finally:
            connect.close(conn)

        body = {"kind": "caption", "model_id": MODEL[0], "model_version": MODEL[1]}
        told = client.post(f"/i/{slug}/said/verdict", json={**body, "verdict": "wrong"})
        assert told.status_code in (200, 201), told.text
        assert told.json()["verdict"] == "wrong"

        # and the page opens showing it, rather than blank over a
        # judgement already made
        page = client.get(f"/i/{slug}", headers={"accept": "text/html"}).text
        assert 'data-said-verdict="wrong"' in page
        assert 'data-said-verdict-set="wrong" aria-pressed="true"' in page

        # the lit thumb again is a retraction, and the answer says so
        again = client.post(f"/i/{slug}/said/verdict", json={**body, "verdict": None})
        assert again.json()["verdict"] is None
        assert "data-said-verdict=" not in client.get(f"/i/{slug}", headers={"accept": "text/html"}).text

        held = client.get(f"/i/{slug}", headers={"accept": "application/json"}).json()
        assert held["said"][0]["verdict"] is None


def test_the_verdict_is_not_the_models_confidence(tmp_path):
    """Two different facts that a single field would average together: a
    model's certainty about its own sentence, and a person's judgement
    of whether it is true."""
    from sg_web.media_view import Said

    fields = Said.model_fields
    assert "confidence" in fields
    assert "verdict" in fields
