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
pages and the window (six) is genuinely exceeded. The page cannot be
smaller: the eight-pages-against-a-window-of-six ratio is not the whole
requirement, because a page also has to be big enough that the first
screen does NOT reach the end. Tried at 80 files and a page of ten,
which keeps that ratio -- the loader pulled all eight pages before the
first assertion and the window had already dropped 1 and 2, leaving
`[3, 4, 5, 6, 7, 8]` where the test wants to watch more ARRIVE.
"""

from __future__ import annotations

import itertools
import time

import pytest
from PIL import Image
from playwright.sync_api import Page, expect

from tests.conftest import POLL, Live

pytestmark = pytest.mark.slow

#: More than WINDOW * SIZE, so cells really are dropped off the top.
MANY = 150
#: Small pages, so eight of them fit in a test that is not slow. Not
#: smaller, for the reason the module docstring records.
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
        time.sleep(POLL)


def _open(page: Page, extra: str = "") -> None:
    """Opened, and the loader mounted.

    The grid's cells are server-rendered, so they are on the page before
    the script that watches the scroll has run -- and a scroll in that
    window is one nothing is listening for. `data-endless` is the
    loader's own attribute, so waiting for it is waiting for the thing
    every test here then talks to.

    Through `expect`, which retries: a read that answers once "won't wait
    a single second, it will just check the locator is there and return
    immediately" (microsoft/playwright docs/src/best-practices-js.md:174).
    """
    page.goto(f"/g?size={SIZE}{extra}")
    expect(page.locator("[data-grid] a.cell").first).to_be_visible()
    expect(page.locator("[data-grid][data-endless]")).to_have_count(1)


def _cells(page: Page) -> int:
    """How many cells the document is holding.

    THE PAGE MAY RELOAD ITSELF UNDER A READ: the authored surface settles by
    asking `/g/locate/{slug}` and calls `window.location.reload()` if that
    answers with an error (frontend/src/authored.ts:113-121).

    `expect(...)` retries across that reload and `page.evaluate` does not, so
    the first read after `_open` is an `expect` and the rest depend on the
    ingest in `prepare` making `/g/locate` answer.
    """
    return page.evaluate("() => document.querySelectorAll('[data-cells] > *').length")


def _pages_held(page: Page) -> list[int]:
    return page.evaluate(
        "() => [...new Set([...document.querySelectorAll('[data-cells] > *')].map(c => Number(c.dataset.page)))]"
        ".sort((a, b) => a - b)"
    )


def _to_bottom(page: Page) -> None:
    page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")


def _ended(page: Page) -> bool:
    """Is the last page of the answer already held?

    The server spells how many there are on the grid
    (sg_web/templates/_grid.html:4 `data-pages`), so "there is no more to
    fetch" is a FACT the page states, not something to conclude by
    waiting for nothing to happen. That distinction is the whole reason
    the settle below is affordable.

    Through locators rather than `evaluate`, for the same reason
    `_settled` polls: this page reloads itself when `/g/locate/{slug}`
    errors (frontend/src/authored.ts:113-121), and a one-shot `evaluate`
    arriving during that reload dies with "Execution context was
    destroyed". A locator re-resolves against whatever document is there.
    """
    last = page.locator("[data-grid]").get_attribute("data-pages")
    return page.locator(f'[data-cells] > *[data-page="{last}"]').count() > 0


#: Consecutive animation frames of `idle` that end a round of work: `pump`
#: re-arms itself a frame after finishing when it appended with more in reach
#: (endless.ts:216), and below five the settle returns mid-work.
STILL_FOR = 5

#: Frames to allow work to BEGIN before giving up on this scroll, and only
#: ever paid when there is more to fetch. `_settled` records why.
BEGIN_WITHIN = 30


def _keep_going(page: Page, times: int) -> None:
    """Scroll to the end `times` over, waiting for the loader each time."""
    for _ in range(times):
        if _ended(page):
            return
        _to_bottom(page)
        _settled(page)


#: Which call of `_settled` this is, so the poll can tell its own
#: counters from the ones the previous call left behind.
_ROUNDS = itertools.count()


def _settled(page: Page, timeout: int = 15_000) -> None:
    """Wait for a round of loading to have STARTED and finished.

    Waiting on a count races the very fetch it triggered, because a scroll
    arriving mid-fetch can be dropped and the count never moves. The loader
    says what it is doing (`data-endless`), which is a fact about the loader
    rather than a guess at how long it needs -- but "idle" is ambiguous
    between not-yet-started and finished, so both edges are watched.

    The counters are primed INSIDE the poll, keyed on the round number,
    rather than by an `evaluate` before it. `wait_for_function` re-runs
    after a navigation; a one-shot `evaluate` dies on one with "Execution
    context was destroyed", and this page reloads itself when
    `/g/locate/{slug}` errors (frontend/src/authored.ts:113-121) -- which
    is exactly what a priming `evaluate` here met. Priming inside the
    poll also means a reload RESTARTS the round, which is what a reload
    does to the loading it interrupted.

    `wait_for_function` polls on animation frames, so these count
    frames, which is the unit the loader's own wake-ups are in.

    `BEGIN_WITHIN` is how long the round is given to START. `idle` is also
    the state BEFORE anything starts, so a settle that accepts it returns at
    once and the round fetches nothing -- and the next `_to_bottom` is then a
    scroll to where the page already is, which fires no scroll event at all
    (endless.ts:309-315). The remaining wake-up is the IntersectionObserver
    on the pager (endless.ts:257), and this is how long it is given.

    That wait is only ever paid when there IS more to fetch, because `_ended`
    answers from `data-pages` first. Waiting it out on every exhausted scroll
    is what took the module from 15s to 23s when it was tried without the
    check.
    """
    page.wait_for_function(
        "([round, still, begin]) => { const g = document.querySelector('[data-grid]');"
        " if (!g) return false;"
        " let w = window.__round;"
        " if (!w || w.id !== round) { w = window.__round = {id: round, still: 0, began: false, waited: 0}; }"
        " if (g.dataset.endless !== 'idle') { w.began = true; w.still = 0; return false; }"
        " w.still += 1;"
        " w.waited += 1;"
        " if (w.began) return w.still >= still;"
        " return w.waited >= begin; }",
        arg=[next(_ROUNDS), STILL_FOR, BEGIN_WITHIN],
        timeout=timeout,
    )


def _reached_page(page: Page, atLeast: int, timeout: int = 15_000) -> None:
    """More arrived, measured in PAGES.

    Not in cells, which is what this waited on and why it failed one run
    in four. The loader holds a WINDOW of pages and drops what falls off
    the top, so the cell count stops rising once the window is full --
    and `was + SIZE` was then a number that could never arrive. Whether
    it did depended on how much the first screen had loaded before `was`
    was read, which is a race with the loader rather than a fact about
    it.

    A page number cannot be capped by the window. It only goes up, and
    "the next page came" is what this test is about.
    """
    for _ in range(6):
        _settled(page, timeout)
        held = _pages_held(page)
        if held and held[-1] >= atLeast:
            return
        _to_bottom(page)
        page.wait_for_timeout(80)
    page.wait_for_function(
        "n => [...document.querySelectorAll('[data-cells] > *')].some(one => Number(one.dataset.page) >= n)",
        arg=atLeast,
        timeout=timeout,
    )


# --- more arrives -----------------------------------------------------------


def test_scrolling_to_the_end_brings_the_next_page(page: Page, live: Live, unbroken):
    _open(page)
    # The first screen may already have pulled a page or two: the trigger is
    # "is the end within reach", and on a wide window a page of twenty cells
    # does not fill it. Read through `expect`, which retries -- see `_cells`.
    expect(page.locator('[data-cells] > *[data-page="1"]').first).to_be_attached()
    # SETTLED first. Read while the loader is still filling the first
    # screen, `was` is a number that keeps moving, and the target built
    # from it is a race rather than an expectation.
    _settled(page)
    was = _pages_held(page)[-1]

    _to_bottom(page)
    _reached_page(page, was + 1)
    held = _pages_held(page)
    assert held == list(range(1, held[-1] + 1)), f"the server's pages, in order, with no gap: {held}"

    # and the cells really are the next ones in the answer, not a repeat
    names = page.evaluate("() => [...document.querySelectorAll('[data-cells] img')].map(i => i.alt)")
    assert len(names) == len(set(names)), "a page was appended twice"


def test_it_keeps_going_for_as_long_as_there_is_more(page: Page, live: Live, unbroken):
    _open(page)
    _keep_going(page, 4)
    _reached_page(page, 4)
    held = _pages_held(page)
    assert held[-1] >= 4, held
    assert held == list(range(held[0], held[-1] + 1)), f"no gap in what is held: {held}"


def test_the_end_of_the_answer_is_the_end(page: Page, live: Live, unbroken):
    """Nothing loops, and nothing invents a page nine."""
    _open(page)
    _keep_going(page, 12)
    held = _pages_held(page)
    assert held[-1] <= (MANY + SIZE - 1) // SIZE, f"held a page past the end: {held}"


# --- the document stays bounded ---------------------------------------------


def test_the_document_does_not_grow_for_ever(page: Page, live: Live, unbroken):
    """Appending without bound is how a tab dies on a real library."""
    _open(page)
    _keep_going(page, 10)
    held = _pages_held(page)
    assert len(held) <= 6, f"{len(held)} pages of cells are in the document at once: {held}"
    assert _cells(page) <= 6 * SIZE


def test_dropping_from_the_top_holds_its_space_open(page: Page, live: Live, unbroken):
    """A drop that reflowed the page under the reader's eyes would be
    worse than the unbounded document it prevents."""
    _open(page)
    _keep_going(page, 10)
    held = _pages_held(page)
    assert held[0] > 1, f"nothing was dropped, so there is nothing to check: {held}"
    kept = page.evaluate("() => parseFloat(getComputedStyle(document.querySelector('[data-cells]')).paddingTop)")
    assert kept > 0, "the space the dropped pages took is held open"


def test_scrolling_back_up_brings_them_back(page: Page, live: Live, unbroken):
    _open(page)
    _keep_going(page, 10)
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
    _keep_going(page, 4)
    page.wait_for_function("() => new URLSearchParams(location.search).get('page') !== null", timeout=15_000)
    at = int(page.evaluate("() => new URLSearchParams(location.search).get('page')"))
    assert at > 1, page.url

    page.reload()
    expect(page.locator("[data-grid]")).to_have_attribute("data-page", str(at))


def test_back_is_still_the_question_before_this_one(page: Page, live: Live, unbroken):
    """Pushing a history entry per screen would make leaving a gallery
    somebody scrolled through twenty presses of Back."""
    # NOT "/" -- the front link redirects into the gallery (sg_web/app.py),
    # so it is the same surface and proves nothing about leaving it.
    page.goto("/operations")
    page.wait_for_load_state()
    _open(page)
    _keep_going(page, 4)
    assert _cells(page) > SIZE

    page.go_back()
    page.wait_for_load_state()
    assert page.url.endswith("/operations"), f"one Back left the gallery, and landed on {page.url}"


# --- nothing was taken away -------------------------------------------------


def test_the_pager_is_still_there_and_still_works(page: Page, live: Live, unbroken):
    """It is the sentinel, the way to jump, and the no-JavaScript path."""
    _open(page)
    pager = page.locator("[data-pager]")
    expect(pager).to_have_count(1)
    expect(pager).to_contain_text("page 1 of")

    # jumping still renders the whole answer from the server
    _open(page, "&page=5")
    expect(page.locator("[data-grid]")).to_have_attribute("data-page", "5")
    held = _pages_held(page)
    # 5, and then whatever it took to fill the screen -- never page 1.
    # A jump starts a new window AT the page jumped to.
    assert held[0] == 5, held
    assert held == list(range(5, held[-1] + 1)), held
