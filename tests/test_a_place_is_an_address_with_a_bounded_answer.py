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

import contextlib
import os

import pytest
from litestar.testing import TestClient
from PIL import Image

from db import collection_rules, collections, connect
from sg_web.app import build_app
from tests.staging import hosting


@pytest.fixture(scope="module")
def _world(tmp_path_factory):
    with hosting(tmp_path_factory, "bounded") as stage:
        yield stage


@pytest.fixture
def served(_world):
    """One application per module for the tests that mutate their own
    worlds; the restore hands each a virgin library."""
    _world.restore()
    return _world.client


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
        rules = collections.collection(conn, "Rules", 3.0, kind="smart")
        collection_rules.keep_prose(conn, rules, nl="only the good ones", now=3.0)
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
    assert body["folders"] == [{"slug": "deep", "name": "deep", "pictures": 3, "below": 3}]
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
        # TYPED, not a message string: a view decides "show the rule
        # instead" by catching this, and the route seams still see a
        # ValueError to answer 400 with.
        with pytest.raises(resultset.UnevaluatedCollection, match="not evaluated"):
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


def test_the_folders_index_enters_by_entity_never_by_path(placed_on_disk):
    """Physical navigation's front link: each registered root as a
    shelf -- kind, reachability, and its depth-0 folder ENTITIES. No
    root ids and no host paths anywhere in the browsing surface; the
    operational /roots route keeps the management shape."""
    placed, root = placed_on_disk
    told = placed.get("/folders", headers=AS_MACHINE)
    assert told.status_code == 200
    assert told.headers["vary"] == "Accept, HX-Request"

    # `cover` is a content address, which is the whole reason it is
    # allowed on this surface: the assertions below prove no host path
    # reaches a browsing answer, and /thumbs/<sha> carries none. Lifted
    # out of BOTH readings rather than popped out of one -- the trash
    # check further down compares a fresh answer against this body.
    def without_covers(rows):
        return [
            {**shelf, "folders": [{k: v for k, v in one.items() if k != "cover"} for one in shelf["folders"]]}
            for shelf in rows
        ]

    body = without_covers(told.json())
    covers = [one["cover"] for shelf in told.json() for one in shelf["folders"]]
    assert all(c is None or c.startswith("/thumbs/") for c in covers), covers
    assert body == [
        {
            "kind": "library",
            "online": True,
            "folders": [
                {"slug": "lib", "name": "lib", "pictures": 2, "below": 5, "first_seen": None, "last_seen": None}
            ],
        }
    ], "2 here, 5 in the subtree; nothing interpreted, so no span"
    page = placed.get("/folders", headers=AS_BROWSER)
    assert page.status_code == 200
    assert 'data-folder="lib"' in page.text
    assert ">online<" in page.text
    for answer in (told.text, page.text):
        assert str(root) not in answer
        assert str(root).replace("\\", "/") not in answer
    # The operational route still says what an operator needs.
    assert str(root) in [row["path"] for row in placed.get("/roots").json()]

    # Trash is a storage location, never a shelf: registering one must
    # not add a browsable "trash" section to the navigation surface.
    bin_dir = root.parent / "bin"
    bin_dir.mkdir()
    assert placed.post("/roots", json={"path": str(bin_dir), "kind": "trash"}).status_code < 300
    assert without_covers(placed.get("/folders", headers=AS_MACHINE).json()) == body, (
        "a trash root changed the navigation"
    )
    assert "trash" not in placed.get("/folders", headers=AS_BROWSER).text


def test_the_albums_index_shows_the_hierarchy_as_authored(placed_on_disk):
    """The browser's /albums is the collection tree: a child renders
    INSIDE its parent's branch and still opens its own /t address; a
    rule-defined node wears its badge instead of a count nothing
    computed. Machines keep the historical flat list."""
    import re

    placed, _ = placed_on_disk
    conn = connect.connect(placed.app.state.db_path)
    keep = conn.execute("SELECT id FROM collection WHERE name = 'Keep'").fetchone()[0]
    collections.collection(conn, "Inner", 4.0, parent_id=keep)
    conn.commit()
    connect.close(conn)

    page = placed.get("/albums", headers=AS_BROWSER).text
    assert re.search(r'data-album="keep".*?<ul>.*?data-album="inner"', page, re.DOTALL), (
        "the child must render inside its parent's branch"
    )
    assert re.search(r'data-album="rules".*?rule-defined', page, re.DOTALL)
    flat = placed.get("/albums", headers=AS_MACHINE).json()
    assert {row["slug"] for row in flat} == {"keep", "inner", "rules"}, "the machine list stays flat and complete"


def test_the_albums_tree_is_one_statement_and_one_snapshot(monkeypatch, served):
    """The whole shelf is ONE SELECT, nested in Python: no query per
    node (the N+1 the review caught), and single-statement atomicity is
    what makes the rendered tree one generation -- a reparent committed
    mid-render can never show a collection twice or lose it."""
    import re

    from db import pages

    with contextlib.nullcontext(served) as client:
        for name in ("Keep", "Drift"):
            client.post("/albums", json={"name": name})
        conn = connect.connect(client.app.state.db_path)
        keep, drift = (
            conn.execute("SELECT id FROM collection WHERE name = ?", (name,)).fetchone()[0]
            for name in ("Keep", "Drift")
        )
        connect.close(conn)

        # The shape: one shelf read, zero per-node walks.
        shelf_calls: list[int] = []
        child_calls: list[int] = []
        real_shelf = pages.collection_shelf
        real_children = pages.collection_children
        monkeypatch.setattr(pages, "collection_shelf", lambda c: (shelf_calls.append(1), real_shelf(c))[1])
        monkeypatch.setattr(pages, "collection_children", lambda c, p: (child_calls.append(1), real_children(c, p))[1])
        assert client.get("/albums", headers=AS_BROWSER).status_code == 200
        assert len(shelf_calls) == 1, "the tree must be one statement"
        assert child_calls == [], "the tree ran a query per node"

        # The snapshot: the reparent commits AFTER the one read -- the
        # response is wholly before, exactly once, and the next request
        # is wholly after, exactly once.
        def fetch_then_reparent(conn_):
            rows = real_shelf(conn_)
            writer = connect.connect(client.app.state.db_path)
            writer.execute("UPDATE collection SET parent_id = ? WHERE id = ?", (keep, drift))
            writer.commit()
            connect.close(writer)
            return rows

        monkeypatch.setattr(pages, "collection_shelf", fetch_then_reparent)
        nested = r'data-album="keep".*?<ul>.*?data-album="drift"'
        before = client.get("/albums", headers=AS_BROWSER).text
        assert before.count('data-album="drift"') == 1, "the reparent forked the rendered tree"
        assert not re.search(nested, before, re.DOTALL), "one response mixed two generations"
        monkeypatch.setattr(pages, "collection_shelf", real_shelf)
        after = client.get("/albums", headers=AS_BROWSER).text
        assert after.count('data-album="drift"') == 1
        assert re.search(nested, after, re.DOTALL), "the NEXT response must see the commit"


@pytest.mark.slow
def test_a_browsing_get_records_nothing_and_the_operational_one_commits(tmp_path, served):
    """/folders observes; /roots records. After the disk changes, the
    browsing GET must answer with fresh reachability while writing
    nothing -- and the operational GET must persist what it saw, not
    hold the writer lane for a rollback."""
    import shutil

    _, root = _library(tmp_path)
    with contextlib.nullcontext(served) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")

        def recorded() -> int:
            conn = connect.connect(client.app.state.db_path, read_only=True)
            try:
                return conn.execute("SELECT online FROM root WHERE id = ?", (made["id"],)).fetchone()[0]
            finally:
                connect.close(conn)

        assert recorded() == 1
        shutil.rmtree(root)
        assert client.get("/folders", headers=AS_MACHINE).json()[0]["online"] is False
        assert recorded() == 1, "a browsing GET wrote to the database"
        assert client.get("/roots").json()[0]["online"] is False
        assert recorded() == 0, "the operational route observed offline but did not persist it"


def test_a_kind_converted_mid_assembly_cannot_mix_the_answer(tmp_path, monkeypatch, served):
    """The CollectionView invariant under fire: kind is NOT immutable --
    an empty collection legally converts to smart -- so the static/smart
    decision must be made under the SAME snapshot the card is read from.
    The writer converts exactly before the ResultSet's first read: the
    response must be wholly one generation (here, wholly smart), never a
    static header over a refused body, and never a 500."""
    from db import resultset

    _, root = _library(tmp_path)
    with contextlib.nullcontext(served) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        assert client.post("/albums", json={"name": "Turncoat"}).json()["slug"] == "turncoat"
        conn = connect.connect(client.app.state.db_path)
        turncoat = conn.execute("SELECT id FROM collection WHERE name = 'Turncoat'").fetchone()[0]
        connect.close(conn)

        real = resultset.page

        def convert_then_ask(conn_, models_dir, query, page_number, now):
            writer = connect.connect(client.app.state.db_path)
            writer.execute("UPDATE collection SET kind = 'smart' WHERE id = ?", (turncoat,))
            writer.commit()
            connect.close(writer)
            return real(conn_, models_dir, query, page_number, now)

        monkeypatch.setattr(resultset, "page", convert_then_ask)
        told = client.get("/t/turncoat", headers=AS_MACHINE)
        assert told.status_code == 200, "a mid-request conversion must never be a 500"
        body = told.json()
        assert body["kind"] == "smart"
        assert body["gallery"] is None
        assert body["state"] == "unevaluated", "a fresh conversion has no typed rule yet"
        assert body["rule"] is None
        monkeypatch.setattr(resultset, "page", real)
        after = client.get("/t/turncoat", headers=AS_MACHINE).json()
        assert (after["kind"], after["gallery"]) == ("smart", None)


def test_a_missing_folder_is_a_state_not_a_404_and_not_merely_empty(tmp_path, served):
    """Presence has three values and the address outlives them all: a
    reachable folder is present, an unplugged root is offline, a
    directory gone from where it was last seen is missing -- and only
    an address nothing lives at is a 404."""
    import shutil

    _, root = _library(tmp_path)
    with contextlib.nullcontext(served) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        assert client.get("/f/deep", headers=AS_MACHINE).json()["state"] == "present"

        shutil.rmtree(root)
        assert client.get("/f/deep", headers=AS_MACHINE).json()["state"] == "offline", (
            "an unreachable root is not a missing folder and not an empty one"
        )
        assert client.get("/folders", headers=AS_MACHINE).json()[0]["online"] is False, (
            "the folders index must say the shelf is unreachable"
        )

        root.mkdir()
        client.post(f"/roots/{made['id']}/scan")
        told = client.get("/f/deep", headers=AS_MACHINE)
        assert told.status_code == 200, "a missing directory keeps its address and its history"
        body = told.json()
        assert body["state"] == "missing"
        assert client.get("/f/never-was").status_code == 404
