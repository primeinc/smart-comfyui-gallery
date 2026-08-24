"""The filmstrip, in a library big enough for it to be wrong.

Two claims cannot be seen in a walk of two pictures, and they are the two
that decide whether this is a neighbourhood or a second gallery hiding
under the first:

    the current picture is where the eye expects it, not at one end;
    the window is a slice of the ANSWER, so a page boundary in the middle
    of it is not an event.

`size=5` puts a boundary every five, and the picture opened here sits on
one. A strip that thought in pages would start there, or stitch two.
"""

from __future__ import annotations

import time

import pytest
from PIL import Image
from playwright.sync_api import Page

from tests.conftest import Live

pytestmark = pytest.mark.slow

MANY = 23


def write_library(root) -> None:
    import os

    for i in range(MANY):
        path = root / f"n_{i:02d}.png"
        Image.new("RGB", (40, 30), (10 * i, 90, 160)).save(path)
        os.utime(path, (1_700_000_000 + i * 60, 1_700_000_000 + i * 60))


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    swept = api.post(f"/roots/{made['id']}/scan").json()
    assert swept["added"] == MANY
    if swept["precache"] is not None:
        deadline = time.monotonic() + 60
        while api.get(f"/jobs/{swept['precache']}").json()["state"] not in ("done", "failed", "cancelled"):
            assert time.monotonic() < deadline
            time.sleep(0.05)


#: The walk this module asks for: oldest first, five to a page.
WALK = {"sort": "oldest", "size": 5}


def _ordinals(page: Page) -> list[int]:
    return page.evaluate(
        "() => [...document.querySelectorAll('[data-filmstrip-item]')].map(a => Number(a.dataset.ordinal))"
    )


def _at(api, ordinal: int) -> str:
    """The slug at one whole-answer ordinal, by the server's own count."""
    listed = api.get("/g/peek", params={**WALK, "page": (ordinal - 1) // 5 + 1, "count": 5}).json()["items"]
    for row in listed:
        if row["ordinal"] == ordinal:
            return row["slug"]
    raise AssertionError(f"no member at ordinal {ordinal}: {[r['ordinal'] for r in listed]}")


def _open(page: Page, live: Live, ordinal: int) -> None:
    page.goto(f"/i/{_at(live.api, ordinal)}?sort=oldest&size=5")
    page.wait_for_selector("[data-filmstrip-item][aria-current='true']", timeout=15_000)


def test_the_window_is_a_slice_of_the_answer_not_of_a_page(page: Page, live: Live, unbroken):
    """Ordinal 11 is the FIRST member of page 3 under size=5.

    The strip around it must run 4..18 -- straight through the boundaries
    at 10 and 15 as though they were not there, because to this window
    they are not.
    """
    _open(page, live, 11)
    held = _ordinals(page)
    assert held == list(range(4, 19)), f"a fifteen-wide window centred on 11: {held}"
    assert (
        page.evaluate(
            "() => Number(document.querySelector(\"[data-filmstrip-item][aria-current='true']\").dataset.ordinal)"
        )
        == 11
    )


def test_the_current_picture_is_where_the_eye_expects_it(page: Page, live: Live, unbroken):
    """Centred on mount -- not scrolled to, not left at an end.

    Measured against the track's own box rather than the window's: the
    strip is what scrolls, and "centred" means centred in it.

    Narrow on purpose. Fifteen 64px cells fit inside a 1280px viewport
    with room to spare, and a strip that does not overflow has nothing to
    centre -- the assertion would pass for a viewer that never scrolled
    at all.
    """
    page.set_viewport_size({"width": 620, "height": 800})
    _open(page, live, 11)
    offset = page.evaluate(
        "() => { const t = document.querySelector('[data-filmstrip-track]');"
        " const a = t.querySelector(\"[data-filmstrip-item][aria-current='true']\");"
        " const track = t.getBoundingClientRect(); const mine = a.getBoundingClientRect();"
        " return {middle: Math.abs((mine.x + mine.width / 2) - (track.x + track.width / 2)),"
        "         scroll: t.scrollWidth, client: t.clientWidth, cells: t.children.length,"
        "         cell: mine.width}; }"
    )
    assert offset["scroll"] > offset["client"], (
        f"fifteen thumbnails in this viewport must overflow, or centring means nothing: {offset}"
    )
    assert offset["middle"] < 40, f"the current picture sits in the middle of the strip: {offset}"


def test_an_edge_of_the_answer_slides_the_window_rather_than_padding_it(page: Page, live: Live, unbroken):
    """Near the start the strip stays FULL and the current item sits near
    the left. Fifteen cells are fifteen pictures, never seven blanks."""
    _open(page, live, 1)
    assert _ordinals(page) == list(range(1, 16))
    assert page.get_attribute("[data-filmstrip-item][aria-current='true']", "data-ordinal") == "1"

    _open(page, live, MANY)
    assert _ordinals(page) == list(range(MANY - 14, MANY + 1))
    assert page.get_attribute("[data-filmstrip-item][aria-current='true']", "data-ordinal") == str(MANY)


def test_walking_moves_the_window_with_you(page: Page, live: Live, unbroken):
    """The strip is the neighbourhood of wherever you now are, so an arrow
    press re-centres it rather than leaving a stale row behind."""
    _open(page, live, 11)
    assert _ordinals(page) == list(range(4, 19))
    page.keyboard.press("ArrowRight")
    page.wait_for_function(
        "() => document.querySelector(\"[data-filmstrip-item][aria-current='true']\").dataset.ordinal === '12'",
        timeout=15_000,
    )
    assert _ordinals(page) == list(range(5, 20))
