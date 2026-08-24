"""Browsing, in a browser: the next page arrives because you kept going.

The gallery had two ways forward, `next` and `previous`, which is a fine
way to READ a result set and a poor way to browse one. Looking for a
picture you would recognise means going through a few hundred, and
clicking `next` every sixty is a decision you have to make five times a
minute about something you are not thinking about.

Every claim here needs a real browser, a real scroll and a real history
stack:

  * that more arrives, and that it is the SERVER'S next page rather than
    a second paging engine in the client
  * that the document stays BOUNDED -- appending for ever is how a tab
    dies on a library of eighty thousand
  * that the URL follows by REPLACING, so reload lands where you were
    reading and Back is still the question before this one rather than
    one press per sixty pictures
  * that the pager survives, because it is the sentinel, the way to
    jump, and the no-JavaScript path

The library is 150 pictures at a page size of 20, so there are eight
pages and the window (six) is genuinely exceeded.
"""

from __future__ import annotations

import time

import pytest
from PIL import Image
from playwright.sync_api import Page

from tests.conftest import Live

pytestmark = pytest.mark.slow

#: More than WINDOW * SIZE, so cells really are dropped off the top.
MANY = 150
#: Small pages, so eight of them fit in a test that is not slow.
SIZE = 20


def write_library(root) -> None:
    for i in range(MANY):
        Image.new("RGB", (48, 36), (20 + (i * 7) % 200, 60, 90)).save(root / f"p{i:03d}.png")


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    swept = api.post(f"/roots/{made['id']}/scan").json()
    assert swept["added"] == MANY
    api.post("/jobs/ingest")
    _drained(api)


def _drained(api, timeout=180.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        running = [j["id"] for j in api.get("/jobs").json() if j["state"] in ("queued", "running")]
        if not running:
            return
        assert time.monotonic() < deadline, f"jobs still running: {running}"
        time.sleep(0.05)


def _open(page: Page, extra: str = "") -> None:
    page.goto(f"/g?size={SIZE}{extra}")
    page.wait_for_selector("[data-grid] a.cell", timeout=20_000)


def _cells(page: Page) -> int:
    return page.evaluate("() => document.querySelectorAll('[data-cells] > *').length")


def _pages_held(page: Page) -> list[int]:
    return page.evaluate(
        "() => [...new Set([...document.querySelectorAll('[data-cells] > *')].map(c => Number(c.dataset.page)))]"
        ".sort((a, b) => a - b)"
    )


def _to_bottom(page: Page) -> None:
    page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")


def _grew_to(page: Page, atLeast: int, timeout: int = 15_000) -> None:
    page.wait_for_function(
        "n => document.querySelectorAll('[data-cells] > *').length >= n", arg=atLeast, timeout=timeout
    )


# --- more arrives -----------------------------------------------------------


def test_scrolling_to_the_end_brings_the_next_page(page: Page, live: Live, unbroken):
    _open(page)
    # The first screen may already have pulled a page or two: the trigger
    # is "is the end within reach", and on a wide window a page of twenty
    # cells does not fill it. What matters is that it starts at page 1 and
    # that more arrives when there is more.
    assert _pages_held(page)[0] == 1
    was = _cells(page)

    _to_bottom(page)
    _grew_to(page, was + SIZE)
    held = _pages_held(page)
    assert held == list(range(1, held[-1] + 1)), f"the server's pages, in order, with no gap: {held}"

    # and the cells really are the next ones in the answer, not a repeat
    names = page.evaluate("() => [...document.querySelectorAll('[data-cells] img')].map(i => i.alt)")
    assert len(names) == len(set(names)), "a page was appended twice"


def test_it_keeps_going_for_as_long_as_there_is_more(page: Page, live: Live, unbroken):
    _open(page)
    for _ in range(4):
        _to_bottom(page)
        page.wait_for_timeout(250)
    _grew_to(page, SIZE * 4)
    held = _pages_held(page)
    assert held[-1] >= 4, held
    assert held == list(range(held[0], held[-1] + 1)), f"no gap in what is held: {held}"


def test_the_end_of_the_answer_is_the_end(page: Page, live: Live, unbroken):
    """Nothing loops, and nothing invents a page nine."""
    _open(page)
    for _ in range(12):
        _to_bottom(page)
        page.wait_for_timeout(150)
    held = _pages_held(page)
    assert held[-1] <= (MANY + SIZE - 1) // SIZE, f"held a page past the end: {held}"


# --- the document stays bounded ---------------------------------------------


def test_the_document_does_not_grow_for_ever(page: Page, live: Live, unbroken):
    """Appending without bound is how a tab dies on a real library."""
    _open(page)
    for _ in range(10):
        _to_bottom(page)
        page.wait_for_timeout(150)
    held = _pages_held(page)
    assert len(held) <= 6, f"{len(held)} pages of cells are in the document at once: {held}"
    assert _cells(page) <= 6 * SIZE


def test_dropping_from_the_top_holds_its_space_open(page: Page, live: Live, unbroken):
    """A drop that reflowed the page under the reader's eyes would be
    worse than the unbounded document it prevents."""
    _open(page)
    for _ in range(10):
        _to_bottom(page)
        page.wait_for_timeout(150)
    held = _pages_held(page)
    assert held[0] > 1, f"nothing was dropped, so there is nothing to check: {held}"
    kept = page.evaluate("() => parseFloat(getComputedStyle(document.querySelector('[data-cells]')).paddingTop)")
    assert kept > 0, "the space the dropped pages took is held open"


def test_scrolling_back_up_brings_them_back(page: Page, live: Live, unbroken):
    _open(page)
    for _ in range(10):
        _to_bottom(page)
        page.wait_for_timeout(150)
    dropped = _pages_held(page)[0]
    assert dropped > 1

    page.evaluate("() => window.scrollTo(0, 0)")
    page.wait_for_function(
        "was => Number([...document.querySelectorAll('[data-cells] > *')][0].dataset.page) < was",
        arg=dropped,
        timeout=15_000,
    )
    assert _pages_held(page)[0] < dropped


# --- the URL follows --------------------------------------------------------


def test_the_url_follows_and_a_reload_lands_where_you_were(page: Page, live: Live, unbroken):
    _open(page)
    for _ in range(4):
        _to_bottom(page)
        page.wait_for_timeout(200)
    page.wait_for_function("() => new URLSearchParams(location.search).get('page') !== null", timeout=15_000)
    at = int(page.evaluate("() => new URLSearchParams(location.search).get('page')"))
    assert at > 1, page.url

    page.reload()
    page.wait_for_selector("[data-grid] a.cell", timeout=20_000)
    assert page.get_attribute("[data-grid]", "data-page") == str(at), "reload lands where you were reading"


def test_back_is_still_the_question_before_this_one(page: Page, live: Live, unbroken):
    """Pushing a history entry per screen would make leaving a gallery
    somebody scrolled through twenty presses of Back."""
    # NOT "/" -- the front link redirects into the gallery (sg_web/app.py),
    # so it is the same surface and proves nothing about leaving it.
    page.goto("/operations")
    page.wait_for_load_state()
    _open(page)
    for _ in range(4):
        _to_bottom(page)
        page.wait_for_timeout(200)
    assert _cells(page) > SIZE

    page.go_back()
    page.wait_for_load_state()
    assert page.url.endswith("/operations"), f"one Back left the gallery, and landed on {page.url}"


# --- nothing was taken away -------------------------------------------------


def test_the_pager_is_still_there_and_still_works(page: Page, live: Live, unbroken):
    """It is the sentinel, the way to jump, and the no-JavaScript path."""
    _open(page)
    pager = page.locator("[data-pager]")
    assert pager.count() == 1
    assert "page 1 of" in pager.inner_text()

    # jumping still renders the whole answer from the server
    page.goto(f"/g?size={SIZE}&page=5")
    page.wait_for_selector("[data-grid] a.cell", timeout=20_000)
    assert page.get_attribute("[data-grid]", "data-page") == "5"
    held = _pages_held(page)
    # 5, and then whatever it took to fill the screen -- never page 1.
    # A jump starts a new window AT the page jumped to.
    assert held[0] == 5, held
    assert held == list(range(5, held[-1] + 1)), held
