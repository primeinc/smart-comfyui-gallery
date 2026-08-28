"""What you told it about its models, portable, with none of your library.

Verdicts are the cheapest valuable thing this application accumulates
and the easiest to share safely. "This model got these 41 wrong" is
worth more than a leaderboard somebody else ran on somebody else's
pictures -- and it is a producer identity, a kind of claim, a verdict, a
content hash and a time. Nothing in that has to carry a photograph.

So the default is the privacy-forward shape and the richer one is asked
for BY NAME. The note especially: it is free text a person typed into a
box, it can hold anything at all, and an export that swept it up by
default would decide on their behalf that it was shareable.

What is tested here is mostly what is ABSENT, because absence is the
whole feature and absence is what quietly regresses -- a column added to
`feedback` later is a column an export built from `SELECT *` would start
handing out.
"""

from __future__ import annotations

import time

import pytest
from litestar.testing import TestClient
from PIL import Image

from db import authored, connect, verdicts
from sg_web.app import build_app
from tests.staging import Stage, staged

pytestmark = pytest.mark.slow


def _holiday(root):
    """The pictures sit under a folder named after somebody. That name is
    one of the things the export must not carry, so it has to be real."""
    shots = root / "Ana's holiday"
    shots.mkdir()
    for i in range(2):
        Image.new("RGB", (16, 12), (10 * i, 90, 140)).save(shots / f"beach_{i}.png")


def _judgements(stage: Stage) -> None:
    conn = stage.conn()
    try:
        ids = [one for (one,) in conn.execute("SELECT id FROM file ORDER BY id")]
        who = authored.person(conn, "Sarah", 0.0)
        authored.feedback(
            conn,
            "annotation",
            "wrong",
            time.time(),
            file_id=ids[0],
            annotation_kind="caption",
            note="it says a dog and it is a cat",
            model_id="Salesforce/blip",
            model_version="base",
        )
        authored.feedback(
            conn,
            "duplicate",
            "wrong",
            time.time(),
            file_id=ids[0],
            other_file_id=ids[1],
            model_id="perceptual",
            model_version="phash64",
        )
        authored.feedback(conn, "person", "wrong", time.time(), file_id=ids[1], person_id=who)
        conn.commit()
    finally:
        connect.close(conn)


@pytest.fixture(scope="module")
def _world(tmp_path_factory):
    with staged(tmp_path_factory, "test_a_verdict_leaves_without_the_picture", _holiday, _judgements) as stage:
        yield stage


@pytest.fixture
def judged(_world):
    """The judged library, restored: one world for the module, and the
    one test that deletes a file gets it back."""
    _world.restore()
    conn = _world.conn()
    ids = [one for (one,) in conn.execute("SELECT id FROM file ORDER BY id")]
    yield _world.client, conn, ids
    connect.close(conn)


def test_a_verdict_carries_its_producer_and_the_bytes_it_was_about(judged):
    """The point of the file: an eval set somebody else can act on. The
    producer is what makes "try another model" possible, and the hash is
    what makes a row checkable -- an eval set nobody can verify is not
    one."""
    client, _conn, _ids = judged
    told = client.get("/operations/export/verdicts.json")
    assert told.status_code == 200, told.text
    assert told.headers["content-disposition"] == 'attachment; filename="verdicts.json"'

    rows = told.json()
    assert len(rows) == 3
    caption = next(row for row in rows if row["judged"] == "annotation")
    assert caption["model_id"] == "Salesforce/blip"
    assert caption["model_version"] == "base"
    assert caption["annotation_kind"] == "caption"
    assert caption["verdict"] == "wrong"
    assert len(caption["sha256"]) == 64
    assert isinstance(caption["at"], float)

    # a pair verdict needs both sides or it says nothing
    pair = next(row for row in rows if row["judged"] == "duplicate")
    assert len(pair["other_sha256"]) == 64
    assert pair["sha256"] != pair["other_sha256"]


def test_no_picture_no_path_and_no_name_leaves_with_it(judged):
    """The half that matters, and the half that regresses quietly.

    Checked against the WHOLE serialised body rather than key by key: a
    column added to `feedback` next year is a column that would start
    riding along, and a test that only names today's keys would not
    notice.
    """
    client, _conn, _ids = judged
    body = client.get("/operations/export/verdicts.json").text

    for secret in ("beach_0.png", "beach_1.png", "Ana's holiday", "Sarah", "it says a dog"):
        assert secret not in body, f"the export carried {secret!r} off the machine"

    for row in client.get("/operations/export/verdicts.json").json():
        # the SHAPE is fixed -- a route's answer has to describe itself
        # -- so what is withheld is the note's VALUE, not its key
        assert set(row) == {*verdicts.EXPORTED, *verdicts.BY_REQUEST}, sorted(row)
        assert row["note"] is None, "the note's text left without being asked for"
        assert "user_id" not in row
        assert "person_id" not in row
        assert "file_id" not in row


def test_the_note_goes_only_when_it_is_asked_for_by_name(judged):
    """It is free text somebody typed. It can hold anything, so nothing
    decides on their behalf that it is shareable."""
    client, _conn, _ids = judged
    assert "it says a dog" not in client.get("/operations/export/verdicts.json").text

    asked = client.get("/operations/export/verdicts.json", params={"include": "note"})
    assert asked.status_code == 200, asked.text
    assert "it says a dog and it is a cat" in asked.text
    assert all("note" in row for row in asked.json())
    assert any(row["note"] for row in asked.json()), "asked for by name and still withheld"


def test_asking_for_anything_else_is_refused_and_says_so(judged):
    """A field quietly ignored would hand somebody a file they believe
    holds something it does not -- which is worse than refusing, because
    they would go on to share it believing the opposite."""
    client, _conn, _ids = judged
    for asked in ("path", "name", "person", "embedding", "note,path"):
        refused = client.get("/operations/export/verdicts.json", params={"include": asked})
        assert refused.status_code == 400, (asked, refused.status_code, refused.text)
        assert "note" in refused.text, "the refusal does not say what it WILL add"


def test_a_verdict_outlives_the_thing_it_judged(judged):
    """`feedback`'s pointers are ON DELETE SET NULL on purpose: dropping
    the derived namespace leaves the judgement standing. So the export
    has to answer for a row whose file is gone -- without a hash, but
    still saying a model got something wrong."""
    client, conn, ids = judged
    conn.execute("DELETE FROM file WHERE id = ?", (ids[0],))
    conn.commit()

    rows = client.get("/operations/export/verdicts.json").json()
    assert len(rows) == 3, "a judgement was lost with the picture it was about"
    orphaned = [row for row in rows if row["sha256"] is None]
    assert orphaned, "the row kept a hash for bytes that are gone"
    assert orphaned[0]["model_id"] in ("Salesforce/blip", "perceptual")
    assert orphaned[0]["verdict"] == "wrong"


def test_an_empty_library_exports_an_empty_list_not_an_error(tmp_path):
    """Nothing judged yet is a real state and the commonest one."""
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        told = client.get("/operations/export/verdicts.json")
        assert told.status_code == 200
        assert told.json() == []


def test_it_is_offered_where_the_verdicts_are(judged):
    """An export nobody can find is nearly an export that does not
    exist. It sits under the panel that adds the verdicts up."""
    client, _conn, _ids = judged
    page = client.get("/operations", headers={"accept": "text/html"}).text
    assert "data-export-verdicts" in page
    assert 'href="/operations/export/verdicts.json"' in page
    assert "data-export-verdicts-note" in page, "the opt-in is not offered beside it"
