"""One media address, three representations, zero second resources.

`/i/{slug}` answers as the MediaView (JSON) for machines, as a lightbox
fragment for the mounted gallery, and as a complete page for a browser
-- all from one assembly, negotiated deterministically and declared
with Vary. The query string is browsing context: previous/next mean the
ResultSet being walked, never a folder walk, and the context survives a
rename's 301. WI-36's acceptance, pinned.
"""

from __future__ import annotations

import re

import pytest
from litestar.testing import TestClient
from PIL import Image

from sg_web.app import build_app

AS_BROWSER = {"accept": "text/html,application/xhtml+xml"}
AS_MACHINE = {"accept": "application/json"}
AS_OVERLAY = {"hx-request": "true"}


@pytest.fixture(scope="module")
def address(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("media")
    root = tmp / "lib"
    root.mkdir()
    import os

    for i in range(6):
        path = root / f"m_{i}.png"
        Image.new("RGB", (12, 12), (40 * i, 90, 120)).save(path)
        os.utime(path, (1_700_000_000 + i * 60, 1_700_000_000 + i * 60))

    with TestClient(app=build_app(str(tmp / "run"), worker=False)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        assert client.post(f"/roots/{made['id']}/scan").json()["added"] == 6
        yield client


def test_the_three_faces_come_from_one_view_and_declare_vary(address):
    told = address.get("/i/m-3", headers=AS_MACHINE)
    page = address.get("/i/m-3", headers=AS_BROWSER)
    part = address.get("/i/m-3", headers=AS_OVERLAY)
    for answer in (told, page, part):
        assert answer.status_code == 200
        assert answer.headers["vary"] == "Accept, HX-Request"
    body = told.json()
    assert body["name"] == "m_3.png"
    assert body["stage"]["kind"] == "image"
    assert '<link rel="canonical" href="/i/m-3">' in page.text, "the canonical address is the BARE entity URL"
    assert "<html" in page.text
    assert "<html" not in part.text, "a fragment mounts into a page, never a page into a page"
    assert "data-lightbox" in part.text
    # One assembly: the fragment and the page name the same neighbours.
    fragment_navs = re.findall(r'data-nav="\w+"[^>]*href="([^"]+)"', part.text)
    assert all(href.startswith("/i/") for href in fragment_navs)


def test_a_machine_with_no_browser_accept_still_gets_json(address):
    # httpx's default Accept is */* -- the historical machine caller.
    told = address.get("/i/m-3")
    assert told.headers["content-type"].startswith("application/json")
    assert told.json()["slug"] == "m-3"
    explicit = address.get("/i/m-3", headers=AS_MACHINE).json()
    assert told.json() == explicit, "the wildcard fallback IS the JSON representation, not a third shape"


def test_previous_and_next_walk_the_resultset_the_url_names(address):
    # Default context: whole library, newest first -- m_5 leads.
    bare = address.get("/i/m-3").json()
    assert (bare["context"]["previous"], bare["context"]["next"]) == ("m-4", "m-2")
    assert bare["context"]["in_answer"] is True
    assert bare["context"]["qs"] == ""
    assert bare["context"]["return_url"] == "/g"
    # The SAME address under the reversed walk swaps the arrows.
    oldest = address.get("/i/m-3", params={"sort": "oldest"}).json()
    assert (oldest["context"]["previous"], oldest["context"]["next"]) == ("m-2", "m-4")
    assert oldest["context"]["qs"] == "sort=oldest"
    # A paging context computes the return page.
    paged = address.get("/i/m-0", params={"sort": "oldest", "size": 2}).json()
    assert paged["context"]["ordinal"] == 1
    assert paged["context"]["return_url"] == "/g?sort=oldest&size=2"
    deep = address.get("/i/m-4", params={"sort": "oldest", "size": 2}).json()
    assert deep["context"]["page"] == 3
    assert deep["context"]["return_url"] == "/g?sort=oldest&size=2&page=3"


def test_a_scope_the_item_is_outside_says_so_instead_of_inventing_arrows(address):
    from db import collections, connect

    conn = connect.connect(address.app.state.db_path)
    album = collections.collection(conn, "Two", 1_700_100_000.0)
    for file_id in [row[0] for row in conn.execute("SELECT id FROM file ORDER BY id LIMIT 2")]:
        collections.set_membership(conn, album, file_id, True, 1_700_100_000.0)
    conn.commit()
    conn.close()
    outside = address.get("/i/m-5", params={"album": "two"}).json()
    assert outside["context"]["in_answer"] is False
    assert outside["context"]["previous"] is None
    assert outside["context"]["next"] is None


def test_a_retired_slug_301s_with_its_context_intact(address):
    from db import connect, naming

    conn = connect.connect(address.app.state.db_path)
    found = naming.resolve(conn, "file", "m-1")
    assert found is not None
    naming.rename(conn, found[0], "renamed one", 1_700_200_000.0)
    conn.commit()
    conn.close()
    moved = address.get("/i/m-1", params={"sort": "oldest", "size": 2}, follow_redirects=False)
    assert moved.status_code == 301
    assert moved.headers["location"] == "/i/renamed-one?sort=oldest&size=2"


def test_a_superseded_currency_is_refused_not_mixed(address):
    current = address.get("/i/m-3").json()["context"]["currency"]
    fresh = address.get("/i/m-3", headers={"x-sg-expect": current})
    assert fresh.status_code == 200
    stale = address.get("/i/m-3", headers={"x-sg-expect": "v0-long-gone"})
    assert stale.status_code == 409


def test_a_commit_landing_mid_request_cannot_cross_generations(tmp_path, monkeypatch):
    """The last WI-36 race: the expectation check must compare the
    currency the view was ACTUALLY located in, after assembly -- a
    pre-assembly check passes at v10, a worker commits, and the arrows
    would silently belong to v11 under the mounted v10 gallery. The
    writer here commits exactly inside that window."""
    import os

    from litestar.testing import TestClient as Client

    from db import connect
    from sg_web import media_view
    from sg_web.app import build_app

    root = tmp_path / "lib"
    root.mkdir()
    for i in range(3):
        path = root / f"r_{i}.png"
        Image.new("RGB", (12, 12), (60 * i, 90, 120)).save(path)
        os.utime(path, (1_700_000_000 + i * 60, 1_700_000_000 + i * 60))

    with Client(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        assert client.post(f"/roots/{made['id']}/scan").json()["added"] == 3
        expected = client.get("/i/r-1").json()["context"]["currency"]

        real = media_view.view

        def commit_then_assemble(conn, models_dir, file_id, slug, query, now, actor_id):
            writer = connect.connect(client.app.state.db_path)
            writer.execute("UPDATE file SET mtime = mtime + 1 WHERE name = 'r_0.png'")
            writer.commit()
            connect.close(writer)
            return real(conn, models_dir, file_id, slug, query, now, actor_id)

        monkeypatch.setattr(media_view, "view", commit_then_assemble)
        raced = client.get("/i/r-1", headers={"x-sg-expect": expected})
        assert raced.status_code == 409, (
            "a commit inside the request window must be refused, not answered under the old expectation"
        )
        monkeypatch.setattr(media_view, "view", real)
        fresh = client.get("/i/r-1").json()["context"]["currency"]
        assert fresh != expected
        assert client.get("/i/r-1", headers={"x-sg-expect": fresh}).status_code == 200
