"""Folders and collections as entity addresses over the one ResultSet.

`/f/{slug}` and `/t/{slug}` describe ENTITIES -- identity, hierarchy,
presence, authored facts -- and never own the media answer: the
rendered grid is one ResultSet page of the folder- or album-faceted
GalleryQuery, the same membership `/g` serves, links in context.

Two meanings pinned hard: `folder=` is the folder ITSELF (direct
children, never the subtree), and a smart collection's membership is
UNEVALUATED, not empty -- the ResultSet refuses the scope rather than
answering zero rows as if the rule had run.
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


def _library(tmp) -> tuple:
    """lib/ holds two stills directly and three more in lib/deep/ --
    the geometry that tells 'the folder' apart from 'the subtree'."""
    root = tmp / "lib"
    (root / "deep").mkdir(parents=True)
    stamped = 1_700_000_000
    for i, name in enumerate(("shore_1.png", "shore_2.png")):
        path = root / name
        Image.new("RGB", (12, 12), (40, 90, 140)).save(path)
        os.utime(path, (stamped + i * 60, stamped + i * 60))
    for i, name in enumerate(("deep_1.png", "deep_2.png", "deep_3.png")):
        path = root / "deep" / name
        Image.new("RGB", (12, 12), (140, 90, 40)).save(path)
        os.utime(path, (stamped + 600 + i * 60, stamped + 600 + i * 60))
    return tmp / "run", root


@pytest.fixture(scope="module")
def placed_on_disk(tmp_path_factory):
    burrow, root = _library(tmp_path_factory.mktemp("place"))
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        assert client.post(f"/roots/{made['id']}/scan").json()["added"] == 5
        assert client.post("/albums", json={"name": "Keep"}).json()["slug"] == "keep"
        for slug in ("shore-1", "deep-1"):
            client.post("/t/keep/add", json={"file": slug})
        conn = connect.connect(client.app.state.db_path)
        authored.collection(conn, "Rules", 3.0, kind="smart", nl_text="only the good ones")
        conn.commit()
        connect.close(conn)
        yield client, root


@pytest.fixture(scope="module")
def placed(placed_on_disk):
    client, _ = placed_on_disk
    return client


def test_the_folder_address_keeps_its_json_and_gains_the_bounded_answer(placed_on_disk):
    placed, root = placed_on_disk
    told = placed.get("/f/lib", headers=AS_MACHINE)
    assert told.status_code == 200
    assert told.headers["vary"] == "Accept, HX-Request"
    body = told.json()
    # The historical machine shape survives, wholly.
    assert sorted(f["name"] for f in body["files"]) == ["shore_1.png", "shore_2.png"]
    assert body["breadcrumb"][-1]["name"] == "lib"
    # And the entity gained its bounded ResultSet answer, in context.
    assert body["count"] == 2, "folder= means the folder itself, never the subtree"
    assert body["gallery"]["qs"] == "folder=lib"
    assert body["state"] == "present"
    assert body["folders"] == [{"slug": "deep", "name": "deep", "pictures": 3}]
    # Durable identity is the slug and the parent chain; the disk path is
    # server-side state and never part of any answer.
    for answer in (told.text, placed.get("/f/lib", headers=AS_BROWSER).text):
        assert str(root) not in answer
        assert str(root).replace("\\", "/") not in answer
    assert placed.get("/f/nowhere").status_code == 404


def test_the_folder_preview_is_exactly_the_resultset_page(placed):
    from db import resultset

    body = placed.get("/f/deep", headers=AS_MACHINE).json()
    conn = connect.connect(placed.app.state.db_path)
    try:
        answer = resultset.page(conn, "", resultset.parse(folder="deep"), 1, 0.0)
    finally:
        connect.close(conn)
    assert [row["slug"] for row in body["gallery"]["items"]] == [row["slug"] for row in answer["items"]]
    assert body["count"] == answer["total"] == 3


def test_the_folder_page_renders_bounded_with_context_links(placed):
    page = placed.get("/f/lib", headers=AS_BROWSER)
    assert page.status_code == 200
    assert '<link rel="canonical" href="/f/lib">' in page.text
    assert page.text.count("?folder=lib") >= 2, "preview media links must carry the folder context"
    assert 'href="/f/deep"' in page.text, "child folders are entities to navigate into"
    deeper = placed.get("/f/deep", headers=AS_BROWSER).text
    assert 'href="/f/lib"' in deeper, "the breadcrumb walks parents as addresses"
    assert "?folder=deep" in deeper


def test_the_folder_browser_path_never_enumerates_the_directory(placed, monkeypatch):
    from db import pages

    walked: list[int] = []
    real = pages.folder_files

    def counted(conn, folder_id, limit=120):
        walked.append(folder_id)
        return real(conn, folder_id, limit)

    monkeypatch.setattr(pages, "folder_files", counted)
    assert placed.get("/f/lib", headers=AS_BROWSER).status_code == 200
    assert walked == [], "a rendered folder page enumerated the directory instead of asking the ResultSet"
    assert placed.get("/f/lib", headers=AS_MACHINE).status_code == 200
    assert len(walked) == 1, "the machine Adapter still carries the legacy list"


def test_the_album_address_keeps_its_json_and_gains_the_bounded_answer(placed):
    told = placed.get("/t/keep", headers=AS_MACHINE)
    assert told.status_code == 200
    assert told.headers["vary"] == "Accept, HX-Request"
    body = told.json()
    assert sorted(f["name"] for f in body["files"]) == ["deep_1.png", "shore_1.png"]
    assert body["count"] == 2
    assert body["gallery"]["qs"] == "album=keep"
    page = placed.get("/t/keep", headers=AS_BROWSER)
    assert '<link rel="canonical" href="/t/keep">' in page.text
    assert page.text.count("?album=keep") >= 2, "preview media links must carry the album context"


def test_the_album_preview_is_exactly_the_resultset_page(placed):
    from db import resultset

    body = placed.get("/t/keep", headers=AS_MACHINE).json()
    conn = connect.connect(placed.app.state.db_path)
    try:
        answer = resultset.page(conn, "", resultset.parse(album="keep"), 1, 0.0)
    finally:
        connect.close(conn)
    assert [row["slug"] for row in body["gallery"]["items"]] == [row["slug"] for row in answer["items"]]


def test_a_smart_collection_is_refused_not_emptied(placed):
    """THE load-bearing distinction: a rule nobody has evaluated is not
    an empty album. The ResultSet refuses the scope; the entity page
    shows the rule and says the media answer does not exist yet."""
    from db import resultset

    refused = placed.get("/g", params={"album": "rules"})
    assert refused.status_code == 400
    assert "not evaluated" in refused.json()["detail"]
    assert placed.get("/g/peek", params={"album": "rules", "page": 1}).status_code == 400

    conn = connect.connect(placed.app.state.db_path)
    try:
        with pytest.raises(ValueError, match="not evaluated"):
            resultset.describe(conn, "", resultset.parse(album="rules"), 0.0)
    finally:
        connect.close(conn)

    body = placed.get("/t/rules", headers=AS_MACHINE).json()
    assert body["rule"] == {"sql": None, "nl": "only the good ones"}
    assert body["gallery"] is None, "unevaluated must never be presented as an empty grid"
    assert body["count"] is None
    assert body["files"] == []
    page = placed.get("/t/rules", headers=AS_BROWSER)
    assert page.status_code == 200
    assert "not evaluated yet" in page.text
    assert 'class="cell"' not in page.text


def test_the_albums_index_renders_for_a_browser_and_stays_json_for_machines(placed):
    index = placed.get("/albums")
    assert index.headers["content-type"].startswith("application/json")
    listed = {row["slug"]: row for row in index.json()}
    assert listed["keep"]["pictures"] == 2
    assert listed["rules"]["kind"] == "smart"
    page = placed.get("/albums", headers=AS_BROWSER)
    assert page.status_code == 200
    assert 'data-album="keep"' in page.text
    assert "rule-defined" in page.text, "a smart collection is never shown as '0 pictures'"


def test_a_missing_folder_is_a_state_not_a_404_and_not_merely_empty(tmp_path):
    """Presence has three values and the address outlives them all: a
    reachable folder is present, an unplugged root is offline, a
    directory gone from where it was last seen is missing -- and only
    an address nothing lives at is a 404."""
    import shutil

    burrow, root = _library(tmp_path)
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        assert client.get("/f/deep", headers=AS_MACHINE).json()["state"] == "present"

        shutil.rmtree(root)
        assert client.get("/f/deep", headers=AS_MACHINE).json()["state"] == "offline", (
            "an unreachable root is not a missing folder and not an empty one"
        )

        root.mkdir()
        client.post(f"/roots/{made['id']}/scan")
        told = client.get("/f/deep", headers=AS_MACHINE)
        assert told.status_code == 200, "a missing directory keeps its address and its history"
        body = told.json()
        assert body["state"] == "missing"
        assert client.get("/f/never-was").status_code == 404


def test_the_place_views_own_no_sql():
    import ast
    import pathlib

    web = pathlib.Path(__file__).resolve().parent.parent / "sg_web"
    for module in ("folder_view.py", "collection_view.py"):
        tree = ast.parse((web / module).read_text(encoding="utf-8"))
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert not called & {"execute", "executemany", "executescript", "cursor"}, (
            f"{module} ran its own statement; queries live in db/pages.py and db/resultset.py"
        )
        spoken = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "db":
                spoken |= {alias.name for alias in node.names}
        assert spoken <= {"connect", "naming", "pages", "resultset", "settings"}, (
            f"{module}: unexpected db vocabulary {sorted(spoken)}"
        )
