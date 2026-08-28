"""Deleting the application must not delete the understanding.

An application whose thesis is custody of your own data cannot be the
only place that data can exist. Names, ratings, places, albums and who
is in what live in `gallery.db` and nowhere else, so this is the way
out: everything a person told this library about their own pictures, as
one document.

The opposite shape from the verdict export beside it, and deliberately.
That one is for SHARING, so it carries no name and no path. This one is
for CUSTODY -- it is yours, it is about your pictures, and an export
that withheld the names would be withholding them from their owner.

Keyed by `content_sha256` and never by a row id, which is the whole
difference between an export and a dump: an id belongs to one database
file, a hash names the same photograph in any library that holds it.

This is the OUT half. Reading one back in is a different problem with
its own conflicts to resolve, and XMP sidecars -- what another DAM could
read -- are untouched.
"""

from __future__ import annotations

import pytest
from PIL import Image

from db import authored, collections, connect, derived
from tests.staging import hosting, staged

pytestmark = pytest.mark.slow


def _library(root) -> None:
    for i in range(4):
        Image.new("RGB", (16, 12), (10 * i, 90, 140)).save(root / f"p{i}.png")


def _what_was_said(stage) -> None:
    """Everything a person told this library, written once."""
    conn = connect.connect(stage.client.app.state.db_path)
    try:
        ids = [one for (one,) in conn.execute("SELECT id FROM file ORDER BY name")]
        who = int(
            conn.execute(
                "INSERT INTO user(username, password_hash, role, created_at) VALUES('ana', 'x', 'USER', 0) RETURNING id"
            ).fetchone()[0]
        )
        sarah = authored.person(conn, "Sarah", 0.0)
        where = derived.region(conn, 0.1, 0.2, 0.3, 0.4)
        # p0: rated, favourited, filed, and Sarah is in it, with a box
        conn.execute(
            "INSERT INTO person_assertion(person_id, file_id, region_id, user_id, created_at, stance)"
            " VALUES(?, ?, ?, ?, 0, 'is')",
            (sarah, ids[0], where, who),
        )
        conn.execute("INSERT INTO rating VALUES(?, ?, 5, 0)", (ids[0], who))
        conn.execute("INSERT INTO favorite VALUES(?, ?, 0)", (ids[0], who))
        trips = collections.collection(conn, "Trips", 0.0)
        iowa = collections.collection(conn, "Iowa 2019", 0.0, parent_id=trips)
        conn.execute("INSERT INTO collection_file VALUES(?, ?, 0)", (iowa, ids[0]))
        # p1: Sarah is expressly NOT in it
        conn.execute(
            "INSERT INTO person_assertion(person_id, file_id, user_id, created_at, stance)"
            " VALUES(?, ?, ?, 0, 'is_not')",
            (sarah, ids[1], who),
        )
        # p2 and p3: nobody has said anything at all
        conn.commit()
    finally:
        connect.close(conn)


@pytest.fixture(scope="module")
def _said_stage(tmp_path_factory):
    with staged(tmp_path_factory, "what_you_said_leaves_with_you", _library, _what_was_said) as stage:
        yield stage


@pytest.fixture(scope="module")
def _bare_stage(tmp_path_factory):
    """A library nobody has said anything about -- no root, no scan.

    A second world rather than a variation of the first: what these two
    tests are about is the state BEFORE anything exists, and a stage
    that had been given a library and then emptied is not that state.
    """
    with hosting(tmp_path_factory, "what_you_said_untouched") as stage:
        yield stage


@pytest.fixture
def untouched(_bare_stage):
    _bare_stage.restore()
    return _bare_stage.client


@pytest.fixture
def said(_said_stage):
    """The world with everything already said, restored between tests.

    Four pictures, an application, a scan and a page of INSERTs cost a
    fifth of a second per test to answer questions that cost a
    hundredth. The snapshot holds the same rows and gives each test a
    clean one.
    """
    _said_stage.restore()
    client = _said_stage.client
    conn = connect.connect(client.app.state.db_path)
    ids = [one for (one,) in conn.execute("SELECT id FROM file ORDER BY name")]
    yield client, conn, ids
    connect.close(conn)


def _slug(client, file_id: int) -> str:
    conn = connect.connect(client.app.state.db_path, read_only=True)
    try:
        return str(conn.execute("SELECT slug FROM entity WHERE id = ?", (file_id,)).fetchone()[0])
    finally:
        connect.close(conn)


def _exported(client) -> dict:
    told = client.get("/operations/export/authored.json")
    assert told.status_code == 200, told.text
    return told.json()


def test_it_carries_what_a_person_said_about_a_picture(said):
    """The whole document, on one photograph: the stars, the flag, the
    keyword, the album it is filed in and who is in it."""
    client, _conn, ids = said
    client.post(f"/i/{_slug(client, ids[0])}/tags", json={"name": "New York"})
    told = _exported(client)

    assert told["people"] == [{"slug": "sarah", "name": "Sarah"}]
    picture = next(one for one in told["pictures"] if one["name"] == "p0.png")
    assert picture["rating"] == 5
    assert picture["favorite"] is True
    assert picture["collections"] == ["iowa-2019"]
    # Both spellings, and the space intact: a keyword holds spaces, so it
    # cannot ride the space-joined slug list the collections use.
    assert picture["tags"] == [{"tag": "new york", "label": "New York"}]
    assert picture["people"] == [
        {"person": "sarah", "stance": "is", "region": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}}
    ]
    assert len(picture["sha256"]) == 64


def test_a_picture_is_named_by_its_bytes_and_not_by_a_row_number(said):
    """The difference between an export and a dump. An id belongs to one
    database file; a hash names the same photograph in any library that
    holds it, which is what lets this be read back at all."""
    client, conn, ids = said
    told = _exported(client)
    shas = {one["sha256"] for one in told["pictures"]}
    assert shas == {
        sha
        for (sha,) in conn.execute(
            "SELECT content_sha256 FROM file WHERE id IN (?, ?)",
            (ids[0], ids[1]),
        )
    }
    for one in told["pictures"]:
        assert "id" not in one
        assert "file_id" not in one


def test_pictures_nobody_has_said_anything_about_are_not_listed(said):
    """A library is mostly pictures nobody has touched. A hundred
    thousand rows of `rating: null` would bury the few hundred that are
    the understanding."""
    client, _conn, _ids = said
    told = _exported(client)
    names = {one["name"] for one in told["pictures"]}
    assert names == {"p0.png", "p1.png"}, sorted(names)


def test_a_negative_claim_leaves_with_the_rest(said):
    """ "Not her" is a CLAIM here rather than the absence of one: it
    survives a rebuild and constrains the next clustering. An export
    that dropped it would hand back a library that starts making the
    same mistake again."""
    client, _conn, _ids = said
    told = _exported(client)
    denied = next(one for one in told["pictures"] if one["name"] == "p1.png")
    assert denied["people"] == [{"person": "sarah", "stance": "is_not", "region": None}]


def test_the_shelf_rebuilds_from_slugs_alone(said):
    """A collection's parent is named the way a picture names its
    collection -- by slug -- so the nesting is in the document rather
    than in the ids it came from."""
    client, _conn, _ids = said
    told = _exported(client)
    by_slug = {one["slug"]: one for one in told["collections"]}
    assert by_slug["iowa-2019"]["parent"] == "trips"
    assert by_slug["trips"]["parent"] is None
    # and every parent named is a collection the document also carries
    for one in told["collections"]:
        assert one["parent"] is None or one["parent"] in by_slug


def test_every_person_a_picture_names_is_in_the_document(said):
    """A slug with nothing to resolve it against is not portable."""
    client, _conn, _ids = said
    told = _exported(client)
    known = {one["slug"] for one in told["people"]}
    for picture in told["pictures"]:
        for who in picture["people"]:
            assert who["person"] in known, f"{who['person']} is named and never introduced"
        for album in picture["collections"]:
            assert album in {one["slug"] for one in told["collections"]}


def test_it_says_the_names_because_they_are_yours(said):
    """The deliberate difference from the verdict export beside it. That
    one shares and so carries no name; this one is custody, and
    withholding a person's own names from them would be the defect."""
    client, _conn, _ids = said
    body = client.get("/operations/export/authored.json").text
    assert "Sarah" in body
    assert "Iowa 2019" in body
    assert "p0.png" in body


def test_an_untouched_library_exports_an_empty_document(untouched):
    """Nothing said yet is a real state, and the answer is a document
    with nothing in it rather than a refusal."""
    told = untouched.get("/operations/export/authored.json")
    assert told.status_code == 200
    assert told.json() == {"people": [], "collections": [], "pictures": []}


def test_both_exports_are_offered_before_anything_has_been_judged(untouched):
    """The defect this test found when it was written.

    Both links first went inside the verdict panel, which only renders
    once somebody has judged something -- so on a fresh library there
    was no way out at all, which is the case where a person is most
    likely to be deciding whether they trust this with their pictures.
    They live in their own section now.
    """
    page = untouched.get("/operations", headers={"accept": "text/html"}).text
    assert "data-operations-export" in page
    assert "data-export-authored" in page
    assert "data-export-verdicts" in page, "the verdict export was hidden until a verdict existed"


def test_it_is_offered_beside_the_one_that_shares(said):
    """Two exports, opposite shapes, next to each other -- which is how
    somebody sees that they are different things."""
    client, _conn, _ids = said
    page = client.get("/operations", headers={"accept": "text/html"}).text
    assert "data-export-authored" in page
    assert 'href="/operations/export/authored.json"' in page
    assert "data-export-verdicts" in page
