"""Seeing what is duplicated, and what removing a copy would cost.

Detection has shipped since `/jobs/dupes`. Seeing the result had no
surface at all -- `/dupes` answered JSON and no page rendered it, so a
person could not look at their own duplicates in their own library.

The review is deliberately read-only, and the reason is the naive
resolution being the one to avoid: byte identity and organisational
identity are different things. Three copies of one photograph filed
under `Iowa 2019`, `Family` and `Old Backup` are ONE content and THREE
placements, and a deduper that reports "2 duplicates removed" has
silently turned two complete collections into incomplete ones. So the
page shows where each copy is filed, and offers no button.

The other half it must not get wrong: these groups are PERCEPTUAL. A
re-encode, a resize and a different crop all land in one. Copies that
are byte-identical can become one stored payload losing nothing; copies
that merely look alike cannot, and a page that called both "duplicates"
without distinguishing them would be recommending data loss.
"""

from __future__ import annotations

import pathlib
import time as clock

import pytest
from litestar.testing import TestClient
from PIL import Image

from db import connect, naming

pytestmark = pytest.mark.slow

NOW = 1_700_000_000.0


def _library(root: pathlib.Path) -> None:
    # Same bytes, three placements: what the entry is about.
    same = Image.new("RGB", (32, 24), (20, 90, 160))
    for name in ("iowa.png", "family.png", "backup.png"):
        same.save(root / name)
    # And one that merely looks alike, saved differently.
    Image.new("RGB", (32, 24), (21, 91, 161)).save(root / "resized.png")


def _grouped(conn, best: int, others: list[int], *, distance: int = 0) -> None:
    """A dupe group as the sweep writes one: a best, and its members."""
    conn.execute(
        "INSERT INTO derived_dupe_group(file_id, group_id, distance, threshold, is_best, verified, computed_at)"
        " VALUES(?, ?, 0, 4, 1, 1, ?)",
        (best, best, NOW),
    )
    for one in others:
        conn.execute(
            "INSERT INTO derived_dupe_group(file_id, group_id, distance, threshold, is_best, verified, computed_at)"
            " VALUES(?, ?, ?, 4, 0, 1, ?)",
            (one, best, distance, NOW),
        )


@pytest.fixture
def reviewed(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    _library(root)
    with TestClient(app=build_app_for(tmp_path)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        conn = connect.connect(client.app.state.db_path)
        try:
            by_name = {
                name: file_id for file_id, name in conn.execute("SELECT id, name FROM file ORDER BY name").fetchall()
            }
            # the three that really are one payload
            _grouped(conn, by_name["backup.png"], [by_name["family.png"], by_name["iowa.png"]])
            conn.commit()
            yield client, conn, by_name
        finally:
            connect.close(conn)


def build_app_for(tmp_path):
    from sg_web.app import build_app

    return build_app(str(tmp_path / "run"), worker=False)


def _filed(conn, file_id: int, name: str) -> None:
    from db import collections

    album = collections.collection(conn, name, clock.time())
    conn.execute("INSERT INTO collection_file VALUES(?, ?, ?)", (album, file_id, clock.time()))


def test_the_page_shows_every_copy_and_where_it_is_filed(reviewed):
    """The whole reason it is a review. A copy's placements are what
    would be lost if that copy went, so they are beside it."""
    client, conn, by_name = reviewed
    _filed(conn, by_name["iowa.png"], "Iowa 2019")
    _filed(conn, by_name["family.png"], "Family")
    conn.commit()

    page = client.get("/dupes", headers={"accept": "text/html"}).text
    for name in ("iowa.png", "family.png", "backup.png"):
        slug = naming.entity_slug(conn, by_name[name])
        assert slug is not None
        assert f'data-dupe-member="{slug[1]}"' in page, f"{name} is not shown"
    assert "Iowa 2019" in page, "a placement that would be lost is not shown"
    assert "Family" in page
    assert page.count("data-dupe-filed") == 2
    assert "data-dupe-unfiled" in page, "the copy in no collection says so rather than showing nothing"


def test_identical_bytes_say_what_consolidating_would_leave(reviewed):
    """The arithmetic, and only where it is true: three placements, one
    payload, every collection still complete."""
    client, _conn, _by_name = reviewed
    page = client.get("/dupes", headers={"accept": "text/html"}).text
    assert 'data-dupe-kind="identical"' in page
    assert "data-dupe-identical" in page
    assert "3 placements, 1 payload" in page


def test_copies_that_are_only_alike_are_never_called_the_same(tmp_path):
    """The distinction that decides whether consolidating is safe at all.

    These groups are perceptual. A group whose members hold different
    bytes cannot become one file without losing whichever is not kept,
    and the page must not offer the sentence that says it can.
    """
    root = tmp_path / "lib"
    root.mkdir()
    _library(root)
    with TestClient(app=build_app_for(tmp_path)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        conn = connect.connect(client.app.state.db_path)
        try:
            by_name = {
                name: file_id for file_id, name in conn.execute("SELECT id, name FROM file ORDER BY name").fetchall()
            }
            _grouped(conn, by_name["backup.png"], [by_name["resized.png"]], distance=3)
            shas = conn.execute(
                "SELECT count(DISTINCT content_sha256) FROM file WHERE name IN ('backup.png','resized.png')"
            ).fetchone()[0]
            assert shas == 2, "the fixture must hold two different payloads for this to mean anything"
            conn.commit()
        finally:
            connect.close(conn)

        page = client.get("/dupes", headers={"accept": "text/html"}).text
        assert 'data-dupe-kind="alike"' in page
        assert "data-dupe-alike" in page
        assert "not the same bytes" in page
        assert "placements, 1 payload" not in page, "it offered to collapse pictures that are not the same file"


def test_the_page_removes_nothing_and_offers_no_way_to(reviewed):
    """Read-only, and provably. The preview is the half that has to be
    right before anything is allowed to touch a file."""
    client, conn, _by_name = reviewed
    before = conn.execute("SELECT count(*) FROM file").fetchone()[0]
    page = client.get("/dupes", headers={"accept": "text/html"}).text
    assert "<form" not in page, "a review surface grew a form"
    for word in ("delete", "remove", "resolve", "merge"):
        assert f">{word}<" not in page.lower(), f"the page offers to {word}"
    assert conn.execute("SELECT count(*) FROM file").fetchone()[0] == before


def test_a_machine_still_gets_the_historical_list(reviewed):
    """Same address, two audiences -- the Adapter rule `/people` follows.
    The JSON shape predates the page and does not change because a page
    arrived."""
    client, _conn, _by_name = reviewed
    told = client.get("/dupes", headers={"accept": "application/json"}).json()
    assert isinstance(told, list)
    assert told, "the machine list is empty where the page shows a group"
    assert set(told[0]) == {"slug", "name", "copies"}
    assert told[0]["copies"] == 3


def test_it_is_reachable_without_knowing_the_address(reviewed):
    """A page nothing links to is a page nobody finds."""
    client, _conn, _by_name = reviewed
    assert '<a href="/dupes"' in client.get("/g", headers={"accept": "text/html"}).text
