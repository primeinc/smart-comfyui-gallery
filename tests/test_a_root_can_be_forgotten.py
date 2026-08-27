"""A directory can be un-watched, and it says what that costs first.

Roots could be added and never removed. Whatever you pointed the
application at, it was pointed at for ever -- which is a restriction
nothing in the schema required, only an absent route.

Removing one is not a small act, though, and the reason is not obvious
from the outside: `folder.root_id` cascades to folders, folders cascade
to files, and files cascade to every rating, comment, favourite, name,
place and collection membership somebody attached to them. A person
deleting "just a directory" would find that out afterwards.

So the shape follows the doctrine the rest of the application already
holds -- observe, preview, prove, act:

    GET    /roots/{id}/removal          what would go
    DELETE /roots/{id}                  refused, and says what would go
    DELETE /roots/{id}?confirm=<path>   done, and says what went

And the thing that must be true whatever happens: NOTHING ON DISK IS
TOUCHED. This forgets; it does not delete. Re-adding the directory finds
every file again -- what does not come back is the knowledge attached to
them, which is the half no rescan can recompute.
"""

from __future__ import annotations

import pathlib

import pytest
from litestar.testing import TestClient
from PIL import Image

from db import connect
from sg_web.app import build_app


@pytest.fixture
def served(tmp_path):
    """Two roots, one of them carrying a person's work: a rating, a
    favourite, a comment, a place, an album membership."""
    keep = tmp_path / "keep"
    drop = tmp_path / "drop"
    # DISTINCT pixels across the two roots, not merely distinct names.
    # `resolve_scan` reconciles by content across the whole library on
    # purpose -- a file dragged from one drive to another is a move --
    # so two roots holding byte-identical pictures are one set of files
    # that appears to migrate, and the second scan re-attributed every
    # row from the first root. That is the scanner being right; a
    # fixture that trips it is measuring the wrong thing.
    made = 0
    for where, count in ((keep, 2), (drop, 3)):
        where.mkdir()
        for i in range(count):
            made += 1
            Image.new("RGB", (24, 18), (made * 17 % 251, made * 53 % 251, made * 97 % 251)).save(
                where / f"{where.name}{i}.png"
            )

    app = build_app(str(tmp_path / "run"))
    with TestClient(app=app) as client:
        made = {}
        for where in (keep, drop):
            root = client.post("/roots", json={"path": str(where)}).json()
            client.post(f"/roots/{root['id']}/scan")
            made[where.name] = root["id"]

        slugs = client.get("/g", params={"size": 100}).text
        import re

        named = dict(re.findall(r'data-slug="([^"]+)"[^>]*>\s*<img[^>]*alt="([^"]*)"', slugs))
        target = next(slug for slug, name in named.items() if name.startswith("drop"))

        assert client.post(f"/i/{target}/rating", json={"value": 5}).status_code < 300
        assert client.post(f"/i/{target}/favorite", json={"value": True}).status_code < 300
        album = client.post("/albums", json={"name": "Keepers"}).json()
        assert client.post(f"/t/{album['slug']}/add", json={"file": target}).status_code == 201

        yield client, made, keep, drop, target


#: What `library.add_root` writes into a directory to give it a durable
#: identity, so a moved root is recognised rather than re-indexed. It is
#: the one file this application puts in somebody's media directory, and
#: forgetting a root deliberately leaves it: removing it would be a disk
#: WRITE, which is exactly what forgetting promises not to do -- and
#: re-adding the directory then reuses the identity it already had.
MARKER = ".smartgallery-root"


def _pictures_on_disk(where: pathlib.Path) -> set[str]:
    return {one.name for one in where.iterdir() if one.is_file() and one.name != MARKER}


# --- looking before deciding ------------------------------------------------


def test_the_cost_is_answerable_before_anything_is_removed(served):
    client, made, _keep, drop, _target = served
    told = client.get(f"/roots/{made['drop']}/removal")
    assert told.status_code == 200, told.text
    cost = told.json()
    assert cost["path"] == str(drop)
    assert cost["files"] == 3
    assert cost["folders"] >= 1
    assert cost["ratings"] == 1
    assert cost["favorites"] == 1
    assert cost["in_collections"] == 1
    # and asking did not remove anything
    assert client.get(f"/roots/{made['drop']}/removal").json()["files"] == 3


def test_a_root_that_is_not_there_is_a_404(served):
    client, _made, _keep, _drop, _target = served
    assert client.get("/roots/9999/removal").status_code == 404
    assert client.delete("/roots/9999").status_code == 404


# --- the refusal names what it is protecting --------------------------------


def test_removing_without_the_path_is_refused_and_says_why(served):
    client, made, _keep, drop, _target = served
    told = client.delete(f"/roots/{made['drop']}")
    assert told.status_code == 400, told.text
    # `detail`, not the raw body: a Windows path arrives JSON-escaped,
    # and comparing against the encoding rather than the value is how a
    # true assertion fails
    said = told.json()["detail"]
    assert "3 file(s)" in said, said
    assert "1 rating(s)" in said, said
    assert "Nothing on disk is touched" in said, said
    assert str(drop) in said, "the refusal must say what to repeat back"
    # nothing happened
    assert client.get(f"/roots/{made['drop']}/removal").json()["files"] == 3


def test_the_wrong_path_is_refused_too(served):
    client, made, keep, _drop, _target = served
    told = client.delete(f"/roots/{made['drop']}", params={"confirm": str(keep)})
    assert told.status_code == 400, told.text
    assert client.get(f"/roots/{made['drop']}/removal").json()["files"] == 3


# --- and then it goes -------------------------------------------------------


def test_the_root_and_what_it_indexed_are_gone(served):
    client, made, _keep, drop, target = served
    before = _pictures_on_disk(drop)

    told = client.delete(f"/roots/{made['drop']}", params={"confirm": str(drop)})
    assert told.status_code == 200, told.text
    went = told.json()["forgot"]
    assert went["files"] == 3
    assert went["ratings"] == 1

    assert client.get(f"/roots/{made['drop']}/removal").status_code == 404
    assert [one["id"] for one in client.get("/roots").json()] == [made["keep"]]
    assert client.get(f"/i/{target}", headers={"accept": "application/json"}).status_code == 404

    assert _pictures_on_disk(drop) == before, "forgetting a root deleted files from the disk"
    assert (drop / MARKER).is_file(), "forgetting a root wrote to the directory it was forgetting"


def test_the_other_root_is_untouched(served):
    client, made, keep, drop, _target = served
    client.delete(f"/roots/{made['drop']}", params={"confirm": str(drop)})
    left = client.get(f"/roots/{made['keep']}/removal").json()
    assert left["files"] == 2, "removing one root took another's files with it"
    assert _pictures_on_disk(keep) == {"keep0.png", "keep1.png"}


def test_re_adding_finds_the_files_again_and_not_the_knowledge(served):
    """The honest half. The bytes come back because they never left; the
    rating does not, because forgetting is what was asked for."""
    client, made, _keep, drop, _target = served
    client.delete(f"/roots/{made['drop']}", params={"confirm": str(drop)})

    again = client.post("/roots", json={"path": str(drop)}).json()
    client.post(f"/roots/{again['id']}/scan")

    cost = client.get(f"/roots/{again['id']}/removal").json()
    assert cost["files"] == 3, "a rescan did not find the files that were always there"
    assert cost["ratings"] == 0, "a rating survived being forgotten"
    assert cost["in_collections"] == 0


def test_the_album_survives_losing_its_member(served):
    """A collection is a thing somebody made. Its member going does not
    take it with them."""
    client, made, _keep, drop, _target = served
    client.delete(f"/roots/{made['drop']}", params={"confirm": str(drop)})
    shelf = client.get("/albums", headers={"accept": "application/json"}).json()
    assert [one["name"] for one in shelf] == ["Keepers"]


# --- the cascade is real, so the count has to be ----------------------------


def test_the_counts_are_what_actually_goes(served, tmp_path):
    """The cost is a promise about a cascade nobody can see. Counted
    against the database itself, before and after."""
    client, made, _keep, drop, _target = served
    db = tmp_path / "run" / "gallery.db"
    conn = connect.connect(str(db), read_only=True)
    try:
        was = {
            name: conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            for name in ("file", "folder", "rating", "favorite", "collection_file")
        }
    finally:
        connect.close(conn)

    cost = client.get(f"/roots/{made['drop']}/removal").json()
    client.delete(f"/roots/{made['drop']}", params={"confirm": str(drop)})

    conn = connect.connect(str(db), read_only=True)
    try:
        now = {
            name: conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
            for name in ("file", "folder", "rating", "favorite", "collection_file")
        }
    finally:
        connect.close(conn)

    assert was["file"] - now["file"] == cost["files"]
    assert was["folder"] - now["folder"] == cost["folders"]
    assert was["rating"] - now["rating"] == cost["ratings"]
    assert was["favorite"] - now["favorite"] == cost["favorites"]
    assert was["collection_file"] - now["collection_file"] == cost["in_collections"]
