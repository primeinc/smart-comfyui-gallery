"""Favorite, rating and album membership: one vertical through MediaView.

`/i/{slug}` has one authored state per actor, assembled inside the same
snapshot as everything else the address shows; the write routes state
DESIRED FACTS so retries are harmless; every adapter -- browser JSON,
the legacy /t membership routes -- runs the same db/authored.py
implementation. And the ResultSet's answer identity keeps the two
generations honest: a favorite moves the library's currency without
moving any answer, while unfiling the walked item moves the answer.
"""

from __future__ import annotations

import os

import pytest
from litestar.testing import TestClient
from PIL import Image

from db import authored, connect
from sg_web.app import build_app

AS_BROWSER = {"accept": "text/html,application/xhtml+xml"}
AS_MACHINE = {"accept": "application/json"}
AS_OVERLAY = {"hx-request": "true"}


def _library(tmp) -> tuple:
    root = tmp / "lib"
    root.mkdir()
    stamped = 1_700_000_000
    for i in range(4):
        path = root / f"pic_{i}.png"
        Image.new("RGB", (12, 12), (60 + i * 20, 90, 140)).save(path)
        os.utime(path, (stamped + i * 60, stamped + i * 60))
    return tmp / "run", root


@pytest.fixture(scope="module")
def kept(tmp_path_factory):
    burrow, root = _library(tmp_path_factory.mktemp("authored"))
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        assert client.post(f"/roots/{made['id']}/scan").json()["added"] == 4
        assert client.post("/albums", json={"name": "Keep"}).json()["slug"] == "keep"
        conn = connect.connect(client.app.state.db_path)
        authored.collection(conn, "Rules", 3.0, kind="smart", nl_text="only the good ones")
        conn.commit()
        connect.close(conn)
        yield client


def test_the_three_faces_report_one_authored_state(kept):
    assert kept.post("/i/pic-1/favorite", json={"value": True}).json()["authored"]["favorite"] is True
    assert kept.post("/i/pic-1/rating", json={"value": 4}).json()["authored"]["rating"] == 4
    told = kept.post("/i/pic-1/collections/keep", json={"value": True}).json()["authored"]
    assert told == {"favorite": True, "rating": 4, "collections": [{"slug": "keep", "name": "Keep"}]}

    import re

    body = kept.get("/i/pic-1", headers=AS_MACHINE).json()
    assert body["authored"] == told
    page = kept.get("/i/pic-1", headers=AS_BROWSER).text
    part = kept.get("/i/pic-1", headers=AS_OVERLAY).text
    for face in (page, part):
        assert re.search(r'data-fav\s+aria-pressed="true"', face), "the favorite must render pressed"
        assert 'data-rating="4"' in face
        assert 'href="/t/keep"' in face, "the strip must show the membership in every presentation"


def test_desired_state_retries_are_idempotent(kept):
    for _ in range(2):
        assert kept.post("/i/pic-2/favorite", json={"value": True}).json()["authored"]["favorite"] is True
    for _ in range(2):
        assert kept.post("/i/pic-2/rating", json={"value": 5}).json()["authored"]["rating"] == 5
    for _ in range(2):
        told = kept.post("/i/pic-2/collections/keep", json={"value": True}).json()["authored"]
        assert [held["slug"] for held in told["collections"]] == ["keep"]
    count = kept.get("/t/keep", headers=AS_MACHINE).json()["count"]
    for _ in range(2):
        told = kept.post("/i/pic-2/collections/keep", json={"value": False}).json()["authored"]
        assert told["collections"] == []
    assert kept.get("/t/keep", headers=AS_MACHINE).json()["count"] == count - 1
    assert kept.post("/i/pic-2/rating", json={"value": None}).json()["authored"]["rating"] is None
    assert kept.post("/i/pic-2/rating", json={"value": 9}).status_code == 400
    assert kept.post("/i/pic-2/favorite", json={"value": False}).json()["authored"]["favorite"] is False


def test_membership_reflects_immediately_everywhere(kept):
    kept.post("/i/pic-3/collections/keep", json={"value": True})
    assert "pic-3" in [row["slug"] for row in kept.get("/t/keep", headers=AS_MACHINE).json()["gallery"]["items"]]
    assert 'data-slug="pic-3"' in kept.get("/g", params={"album": "keep"}).text
    kept.post("/i/pic-3/collections/keep", json={"value": False})
    assert "pic-3" not in [row["slug"] for row in kept.get("/t/keep", headers=AS_MACHINE).json()["gallery"]["items"]]


def test_a_smart_collection_refuses_through_every_adapter(kept):
    for value in (True, False):
        refused = kept.post("/i/pic-1/collections/rules", json={"value": value})
        assert refused.status_code == 400
        assert "rule" in refused.json()["detail"]
    choices = kept.get("/i/pic-1/collection-choices").json()
    assert "rules" not in [one["slug"] for one in choices], "a rule-derived kind has no membership to offer"
    assert [one["slug"] for one in choices] == ["keep"]


def test_authored_state_is_per_actor_and_survives_restart(tmp_path):
    burrow, root = _library(tmp_path)
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        actor = client.app.state.actor_id
        client.post("/i/pic-0/favorite", json={"value": True})
        client.post("/i/pic-0/rating", json={"value": 3})
        conn = connect.connect(client.app.state.db_path)
        other = authored.add_user(conn, "guest", "!", "USER", 0.0)
        file_id = conn.execute("SELECT id FROM file WHERE name = 'pic_0.png'").fetchone()[0]
        authored.set_rating(conn, file_id, other, 1, 0.0)
        conn.commit()
        # Each signature holds its own facts.
        assert authored.media_state(conn, file_id, actor).rating == 3
        assert authored.media_state(conn, file_id, other).rating == 1
        assert authored.media_state(conn, file_id, other).favorite is False
        connect.close(conn)

    with TestClient(app=build_app(str(burrow), worker=False)) as reopened:
        assert reopened.app.state.actor_id == actor, "the local actor must resolve to the same identity"
        told = reopened.get("/i/pic-0", headers=AS_MACHINE).json()["authored"]
        assert (told["favorite"], told["rating"]) == (True, 3)


def test_a_favorite_moves_the_currency_but_not_the_answer(kept):
    """THE answer-identity contract: data_version bumps on every commit,
    so a favorite invalidates the projection -- but the re-answered
    question orders the same files, and the answer hash says so. A
    membership write against the walked album is the opposite case."""
    before = kept.get("/g", params={"album": "keep"})
    import re

    held = dict(re.findall(r'data-(currency|answer)="([^"]*)"', before.text))
    kept.post("/i/pic-1/favorite", json={"value": False})
    located = kept.get("/g/locate/pic-1", params={"album": "keep"}).json()
    assert located["currency"] != held["currency"], "a commit must move the library generation"
    assert located["answer"] == held["answer"], "a favorite must not move the album's ordered answer"

    kept.post("/i/pic-1/collections/keep", json={"value": False})
    gone = kept.get("/g/locate/pic-1", params={"album": "keep"}).json()
    assert gone == {"in_answer": False}, "the unfiled item must leave the walked answer"
    emptied = dict(re.findall(r'data-(currency|answer)="([^"]*)"', kept.get("/g", params={"album": "keep"}).text))
    assert emptied["answer"] != located["answer"], "membership changed the album's answer identity"
    kept.post("/i/pic-1/collections/keep", json={"value": True})
    back = kept.get("/g/locate/pic-1", params={"album": "keep"}).json()
    assert back["answer"] == located["answer"], "the same ordered answer must carry the same identity"
    assert back["currency"] != located["currency"]


def test_the_write_routes_own_no_semantics():
    """Every adapter shares db/authored.py: the media routes run no SQL
    of their own, and the legacy /t membership routes delegate to the
    same set_collection_membership implementation."""
    import ast
    import pathlib

    web = pathlib.Path(__file__).resolve().parent.parent / "sg_web"
    tree = ast.parse((web / "media_authored.py").read_text(encoding="utf-8"))
    called = {
        node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called & {"execute", "executemany", "executescript", "cursor"}, (
        "a write route ran its own statement; the rules live in db/authored.py"
    )
    assert {"set_favorite", "set_rating", "set_collection_membership", "media_state"} <= called
    spoken = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "db":
            spoken |= {alias.name for alias in node.names}
    assert spoken <= {"authored", "connect", "naming", "pages"}, f"unexpected db vocabulary {sorted(spoken)}"

    legacy = (web / "app.py").read_text(encoding="utf-8")
    assert "set_collection_membership" in legacy, "the /t routes stopped delegating to the shared implementation"
    assert "add_to_collection" not in legacy, "a second membership write path came back"
    assert "remove_from_collection" not in legacy, "a second membership write path came back"
