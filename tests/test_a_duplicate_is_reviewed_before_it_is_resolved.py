"""Seeing what is duplicated, and what removing a copy would cost.

Detection has shipped since `/jobs/dupes`. Seeing the result had no
surface at all -- `/dupes` answered JSON and no page rendered it, so a
person could not look at their own duplicates in their own library.

The review is deliberately read-only ABOUT THE FILES -- it can disagree
with a grouping, which is a guess, and it cannot touch a byte. The
reason is the naive resolution being the one to avoid: byte identity and organisational
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
from tests.staging import NOW, staged

pytestmark = pytest.mark.slow


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


def _setup(stage):
    """The dupe group written once; the snapshot carries it to every test."""
    conn = stage.conn()
    try:
        by_name = {
            name: file_id for file_id, name in conn.execute("SELECT id, name FROM file ORDER BY name").fetchall()
        }
        # the three that really are one payload
        _grouped(conn, by_name["backup.png"], [by_name["family.png"], by_name["iowa.png"]])
        conn.commit()
    finally:
        connect.close(conn)


@pytest.fixture(scope="module")
def _world(tmp_path_factory):
    with staged(tmp_path_factory, "dupes", _library, _setup) as stage:
        yield stage


@pytest.fixture
def reviewed(_world):
    """One served world per module instead of one boot per test: 17
    setups at ~230ms each were the whole cost of this file."""
    _world.restore()
    conn = _world.conn()
    try:
        by_name = {
            name: file_id for file_id, name in conn.execute("SELECT id, name FROM file ORDER BY name").fetchall()
        }
        yield _world.client, conn, by_name
    finally:
        connect.close(conn)


def build_app_for(tmp_path):
    from sg_web.app import build_app

    return build_app(str(tmp_path / "run"), worker=False)


def _album(conn, name: str, *file_ids: int) -> int:
    """One album, made once, holding all of these.

    `collections.collection` is a CONSTRUCTOR and not a get-or-create --
    two albums may share a name here, because the address is the slug --
    so calling it per picture makes one album per picture and every
    count in this file reads 1/1.
    """
    from db import collections

    album = collections.collection(conn, name, clock.time())
    for file_id in file_ids:
        conn.execute("INSERT INTO collection_file VALUES(?, ?, ?)", (album, file_id, clock.time()))
    return album


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
    # three placements, one payload -- the count the page is allowed to
    # offer only because the bytes match.
    assert 'data-dupe-consolidates="3"' in page
    assert "The same file, saved 3 times" in page, "the arithmetic is on the element and not in the page"


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
        assert 'data-dupe-payloads="2"' in page, "two payloads are not counted where the page says they differ"
        assert "not interchangeable" in page
        assert "data-dupe-consolidates" not in page, "it offered to collapse pictures that are not the same file"


def test_the_page_says_how_whole_each_album_would_be_left(reviewed):
    """The other half of the preview.

    "every collection still complete" is a claim about somebody's own
    albums, and until now the page asserted it with nothing beside it.
    Each collection a copy is filed under is named here with how whole
    it is: 6 of 6 present, which is the number the naive operation would
    move and this one does not.
    """
    client, conn, by_name = reviewed
    # albums with more in them than the copy, so the count means something
    _album(conn, "Iowa 2019", by_name["iowa.png"], by_name["resized.png"])
    _album(conn, "Family", by_name["family.png"])
    conn.commit()

    page = client.get("/dupes", headers={"accept": "text/html"}).text
    assert "data-dupe-placements" in page, "the page claims completeness and shows none"
    assert 'data-dupe-placement="Iowa 2019"' in page
    assert 'data-dupe-placement="Family"' in page
    assert 'data-placement-whole="2/2"' in page, "Iowa 2019 holds two pictures, both present"
    assert 'data-placement-whole="1/1"' in page, "Family holds one"


def test_an_album_already_short_of_its_own_says_so(reviewed):
    """A file whose bytes are gone keeps its placement -- `missing_since`
    and never a delete. So an album can be short of itself before
    anybody consolidates anything, and the preview must show that
    rather than rounding it up to complete: the whole use of the number
    is that somebody can watch whether it moves.
    """
    client, conn, by_name = reviewed
    _album(conn, "Iowa 2019", by_name["iowa.png"], by_name["resized.png"])
    conn.execute("UPDATE file SET missing_since = ? WHERE id = ?", (NOW, by_name["resized.png"]))
    conn.commit()

    page = client.get("/dupes", headers={"accept": "text/html"}).text
    assert 'data-placement-whole="1/2"' in page, "an album short of its own members read as complete"


def test_a_copy_in_no_collection_adds_no_album(reviewed):
    """Nothing is invented for a picture nobody filed."""
    client, _conn, _by_name = reviewed
    page = client.get("/dupes", headers={"accept": "text/html"}).text
    assert "data-dupe-placements" not in page, "an album appeared for pictures in none"


def test_two_copies_in_one_album_do_not_count_it_twice(reviewed):
    """The arithmetic trap. Reaching the collection's other members by
    JOINing them to the group's members multiplies the two: two copies
    filed in one eight-picture album count sixteen. Measured on this
    tree before the query was written that way -- 16 against 8."""
    client, conn, by_name = reviewed
    _album(conn, "One Shoot", *(by_name[n] for n in ("iowa.png", "family.png", "backup.png", "resized.png")))
    conn.commit()

    page = client.get("/dupes", headers={"accept": "text/html"}).text
    assert 'data-placement-whole="4/4"' in page, "the album's own members were counted once per copy"
    assert 'data-placement-whole="12/12"' not in page


def test_the_machine_list_did_not_widen(reviewed):
    """A route's answer shape only ever shrinks. The placements are for
    the page; the historical JSON list is three keys and stays three."""
    client, _conn, _by_name = reviewed
    told = client.get("/dupes", headers={"accept": "application/json"}).json()
    assert set(told[0]) == {"slug", "name", "copies"}


def test_the_page_removes_nothing_and_offers_no_way_to(reviewed):
    """Read-only about the FILES, and provably.

    It is not read-only about the GROUPING -- a group is a guess and the
    page can disagree with one, which is the correction below. What it
    cannot do is touch a byte: the preview is the half that has to be
    right before anything is allowed to.
    """
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


# --- and the page can disagree with the grouping -----------------------------


def test_saying_two_are_not_one_takes_them_apart_now(reviewed):
    """A perceptual group is a GUESS: pHash sees global composition, so
    two photographs of one scene a second apart are close in it. The
    page that showed them had no way to say they are two pictures."""
    client, conn, by_name = reviewed
    best = naming.entity_slug(conn, by_name["backup.png"])
    other = naming.entity_slug(conn, by_name["iowa.png"])
    assert best is not None
    assert other is not None

    told = client.post(f"/dupes/{other[1]}/not-a-duplicate", json={"other": best[1]})
    assert told.status_code == 204, told.text

    page = client.get("/dupes", headers={"accept": "text/html"}).text
    assert f'data-dupe-member="{other[1]}"' not in page, "the picture is still in the group"
    assert f'data-dupe-member="{best[1]}"' in page, "the rest of the group went with it"


def test_the_correction_survives_the_next_sweep(reviewed):
    """The half that makes it a correction rather than a chore. A
    grouping the sweep rebuilds every run would put them back together,
    so the sweep reads the verdicts back before it writes a group."""
    from db import authored

    client, conn, by_name = reviewed
    low, high = sorted((by_name["iowa.png"], by_name["backup.png"]))
    one = naming.entity_slug(conn, high)
    two = naming.entity_slug(conn, low)
    assert one is not None
    assert two is not None

    client.post(f"/dupes/{one[1]}/not-a-duplicate", json={"other": two[1]})
    assert (low, high) in authored.rejected_pairs(conn), "nothing recorded that they are not one picture"


def test_it_counts_against_the_producer_that_grouped_them(reviewed):
    """The same thing correcting a face does: somebody is already doing
    the work, and the aggregate that would tell them a fingerprinting
    threshold is costing them time should hear about it."""
    from db import verdicts

    client, conn, by_name = reviewed
    one = naming.entity_slug(conn, by_name["iowa.png"])
    two = naming.entity_slug(conn, by_name["backup.png"])
    assert one is not None
    assert two is not None

    client.post(f"/dupes/{one[1]}/not-a-duplicate", json={"other": two[1]})
    held = conn.execute(
        "SELECT target_kind, verdict, model_id FROM feedback WHERE target_kind = 'duplicate'"
    ).fetchall()
    assert held == [("duplicate", "wrong", "perceptual")]
    # and it does not leak into the RATED table, which is about captions
    assert verdicts.by_producer(conn) == []


def test_saying_it_twice_is_one_correction(reviewed):
    """One person has one opinion about one pair."""
    from db import authored

    client, conn, by_name = reviewed
    one = naming.entity_slug(conn, by_name["iowa.png"])
    two = naming.entity_slug(conn, by_name["backup.png"])
    assert one is not None
    assert two is not None

    client.post(f"/dupes/{one[1]}/not-a-duplicate", json={"other": two[1]})
    client.post(f"/dupes/{one[1]}/not-a-duplicate", json={"other": two[1]})
    assert len(authored.rejected_pairs(conn)) == 1


def test_the_order_of_the_pair_does_not_matter(reviewed):
    """ "A is not B" and "B is not A" are one statement, so the pair is
    stored lowest id first and read back the same way."""
    from db import authored

    client, conn, by_name = reviewed
    one = naming.entity_slug(conn, by_name["iowa.png"])
    two = naming.entity_slug(conn, by_name["backup.png"])
    assert one is not None
    assert two is not None

    client.post(f"/dupes/{one[1]}/not-a-duplicate", json={"other": two[1]})
    client.post(f"/dupes/{two[1]}/not-a-duplicate", json={"other": one[1]})
    assert len(authored.rejected_pairs(conn)) == 1


def test_a_picture_is_not_a_duplicate_of_itself(reviewed):
    client, conn, by_name = reviewed
    one = naming.entity_slug(conn, by_name["iowa.png"])
    assert one is not None
    told = client.post(f"/dupes/{one[1]}/not-a-duplicate", json={"other": one[1]})
    assert told.status_code == 400, told.text
