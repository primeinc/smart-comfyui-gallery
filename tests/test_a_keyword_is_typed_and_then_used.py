"""Tagging, in a browser: the gesture and the round trip.

The store, the fold and the filter are proved without a browser
(test_a_word_you_wrote_on_a_picture.py). What needs one is the part no
unit test can see: whether a person can actually put a word on a
picture, and whether the word then takes them somewhere.

A keyword nobody can type is a schema change. A keyword you can type and
not use is a decoration. So both halves are here, in that order, and the
second one is a real navigation to a real filtered answer rather than an
assertion about an href.

The suggestion list matters more than it looks. A vocabulary is only
worth having if one picture-idea gets one word every time, and nobody
remembers whether they wrote "seaside" or "sea side" last year -- so what
the library already calls things is offered as the box is entered.
"""

from __future__ import annotations

import time

import pytest
from PIL import Image
from playwright.sync_api import Error, Page, expect

from tests.conftest import Live

pytestmark = pytest.mark.slow


FILES = 4


def write_library(root) -> None:
    for i in range(FILES):
        Image.new("RGB", (64, 48), (20 + i * 30, 90, 140)).save(root / f"p{i:02d}.png")


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    swept = api.post(f"/roots/{made['id']}/scan").json()
    assert swept["added"] == FILES


# One served run for the module, so the library these tests write on is
# SHARED. Every assertion below is about a word this test typed, never
# about how many keywords exist -- an order-dependent count here would be
# a test that passes alone and fails in the suite.
#: The pictures' own addresses, asked once for the whole module.
#:
#: Every test here reached a picture by loading the grid and clicking a
#: cell -- TWO page loads to arrive at one picture, eight times over. The
#: addresses are a fact about the library this module wrote, and it does
#: not change between tests, so the first test discovers them and the
#: rest go straight there.
_ADDRESSES: list[str] = []


def _cell_addresses(page: Page, timeout: float = 15.0) -> list[str]:
    """Every cell's address, read across a reload the page may do.

    This surface reloads itself -- `/g/locate/{slug}` erroring, or the
    walked answer having really moved, both end in `location.reload()`
    (frontend/src/authored.ts:113-127). Both are the product being
    right; the test simply cannot assume the document it is reading
    outlives the read.

    The retry has to be on the READ. An `expect` in front of it retries
    across a navigation, so it proves a cell was attached at SOME instant --
    never that it was still there at the instant of the one-shot
    `evaluate_all` after it, which is where a reload destroys the context.
    """
    ended = time.monotonic() + timeout
    while True:
        expect(page.locator("[data-grid] a.cell").first).to_be_attached()
        try:
            return page.locator("[data-grid] a.cell").evaluate_all(
                "cells => cells.map(one => one.getAttribute('href'))"
            )
        except Error as destroyed:
            if "destroyed" not in str(destroyed) or time.monotonic() > ended:
                raise


def _open(page: Page, nth: int = 0) -> None:
    if not _ADDRESSES:
        page.goto("/g?sort=oldest")
        _ADDRESSES.extend(_cell_addresses(page))
    page.goto(_ADDRESSES[nth])
    expect(page.locator("[data-authored] [data-tag-input]").last).to_be_visible()


def _open_first(page: Page) -> None:
    _open(page, 0)


def _type(page: Page, word: str) -> None:
    box = page.locator("[data-authored] [data-tag-input]").last
    box.fill(word)
    box.press("Enter")


def test_a_word_typed_on_a_picture_appears_on_it(live: Live, page: Page):
    """The gesture. Type, press Enter, and the picture wears the word --
    without a reload, because the write answers with the authoritative
    state and the strip redraws from it."""
    _open_first(page)
    _type(page, "Sunset")
    expect(page.locator('[data-authored] [data-tag="sunset"]').last).to_contain_text("Sunset")


def test_the_word_survives_a_reload_because_it_was_written_down(live: Live, page: Page):
    """The difference between a chip drawn by script and a fact. The
    strip is rendered from `item.authored` server-side, so a reload
    proves the row exists rather than that the browser remembers."""
    _open_first(page)
    _type(page, "Harbour")
    page.reload()
    expect(page.locator('[data-authored] [data-tag="harbour"]').last).to_contain_text("Harbour")


def test_the_box_empties_so_the_next_word_can_be_typed(live: Live, page: Page):
    """Keywords arrive in threes. A box that clears only when the
    response lands drops whatever was typed in between."""
    _open_first(page)
    _type(page, "one")
    box = page.locator("[data-authored] [data-tag-input]").last
    expect(box).to_have_value("")
    _type(page, "two")
    expect(page.locator('[data-authored] [data-tag="one"]').last).to_be_visible()
    expect(page.locator('[data-authored] [data-tag="two"]').last).to_be_visible()


def test_case_does_not_make_a_second_keyword(live: Live, page: Page):
    """The fold, seen from the outside: two spellings, one chip."""
    _open_first(page)
    _type(page, "Beach")
    _type(page, "BEACH")
    expect(page.locator('[data-authored] [data-tag="beach"]')).to_have_count(1)


def test_the_cross_takes_it_off_again(live: Live, page: Page):
    """A word put on by mistake has to come off in one click, where it
    is -- not in a settings page somewhere."""
    _open_first(page)
    _type(page, "wrong")
    chip = page.locator('[data-authored] [data-tag="wrong"]').last
    expect(chip).to_be_visible()
    chip.locator("[data-untag]").click()
    expect(page.locator('[data-authored] [data-tag="wrong"]')).to_have_count(0)


def test_the_keyword_is_a_way_into_the_library(live: Live, page: Page):
    """The half that makes it a keyword rather than a label. Clicking it
    asks the gallery the question, through the same `f=tag:eq:` spelling
    the filter drawer writes -- so arriving from a picture and arriving
    from the drawer are one question."""
    _open_first(page)
    _type(page, "Iowa")
    page.locator('[data-authored] [data-tag="iowa"] a').last.click()
    page.wait_for_url("**/g?f=tag*")
    expect(page.locator("[data-grid] a.cell")).to_have_count(1)
    # and the question says itself in words, not as a raw key
    expect(page.locator("[data-chips]")).to_contain_text("keyword")


def test_what_the_library_already_calls_things_is_offered(live: Live, page: Page):
    """Typo prevention is the whole keyword problem. Entering the box
    fills the datalist from the same /g/options the filter drawer
    reads."""
    _open_first(page)
    _type(page, "Cormorant")
    _open(page, 1)
    page.locator("[data-authored] [data-tag-input]").last.focus()
    # By VALUE and not by count: the run is shared, so what else the
    # library has been called by now is another test's business.
    expect(page.locator('[data-keyword-list] option[value="Cormorant"]')).to_have_count(1)


def test_an_empty_box_asks_for_nothing(live: Live, page: Page):
    """Enter on an empty box is what a person does while thinking. It
    must not become a request, let alone a refusal on screen."""
    _open_first(page)
    before = page.locator("[data-authored] [data-tag]").count()
    _type(page, "   ")
    expect(page.locator("[data-authored] [data-tag]")).to_have_count(before)
