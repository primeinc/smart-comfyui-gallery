"""Open the app knowing nothing of the schema, and find the field anyway.

The backlog stated this as a browser interaction on purpose, because it
cannot be shown any other way: a unit test for `param.is` passes with the
old text box still in place, and the old text box is the defect. Its
placeholder was `key=value`. That is the application asking a person to
recall its internal spelling -- the one thing it exists to remember for
them -- and it is what makes the long tail unreachable to anybody who
did not write the parser.

So this drives the whole path: type a word, see what the library has,
choose it, land on a control that already knows what to do with it.

The two halves of the vocabulary stay apart INSIDE and arrive together
here. A curated dimension has a section waiting with its own operators
and its own counted values; a discovered key is asked through the
long-tail door with its spelling already filled in. Nothing in the list
tells a person which is which, because that distinction is ours.
"""

from __future__ import annotations

import time

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from playwright.sync_api import Page

from tests.conftest import Live

pytestmark = pytest.mark.slow

#: Four with one recipe and two with another, so a wrong count is
#: visible -- with one of each, every wrong answer is still 1.
MADE = [
    *[("dreamshaper_8", "Euler a")] * 4,
    *[("juggernautXL", "DPM++ 2M")] * 2,
]
WHOLE = len(MADE) + 1


def _recipe(checkpoint: str, sampler: str) -> str:
    return (
        "a brass diving helmet at dusk\n"
        "Negative prompt: blurry\n"
        f"Steps: 28, Sampler: {sampler}, CFG scale: 7, Seed: 4242, Size: 832x1216, Model: {checkpoint}"
    )


def write_library(root) -> None:
    for i, (checkpoint, sampler) in enumerate(MADE):
        info = PngInfo()
        info.add_text("parameters", _recipe(checkpoint, sampler))
        Image.new("RGB", (64, 48), (20 + i * 7, 60, 90)).save(root / f"made_{i:02d}.png", pnginfo=info)
    Image.new("RGB", (64, 48), (10, 120, 10)).save(root / "taken.png")


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    swept = api.post(f"/roots/{made['id']}/scan").json()
    assert swept["added"] == WHOLE
    api.post("/jobs/ingest")
    deadline = time.monotonic() + 90
    while True:
        running = [one["id"] for one in api.get("/jobs").json() if one["state"] in ("queued", "running")]
        if not running:
            return
        assert time.monotonic() < deadline, f"jobs still running: {running}"
        time.sleep(0.1)


def _open_filters(page: Page) -> None:
    if page.get_attribute("[data-filters-open]", "aria-expanded") != "true":
        page.click("[data-filters-open]")
    page.wait_for_selector("[data-filters-panel]:not([hidden])", timeout=5_000)


def _find(page: Page, typed: str) -> None:
    """Type into the Add-filter box and wait for the list it draws."""
    _open_filters(page)
    box = page.locator("[data-filter-find-input]")
    box.click()
    box.fill(typed)
    page.wait_for_selector("[data-filter-found]:not([hidden]) [data-field]", timeout=10_000)


def _rows(page: Page) -> list[dict]:
    return page.evaluate(
        "() => [...document.querySelectorAll('[data-filter-found] [data-field]')]"
        ".map(r => ({key: r.dataset.field, param: r.dataset.param ?? null,"
        " label: r.querySelector('.filter-found-label').textContent}))"
    )


def test_an_empty_box_answers_what_is_worth_asking(page: Page, live: Live, unbroken):
    """Somebody with an empty box has the question "what CAN I filter
    by", and the honest answer is a list, not a placeholder."""
    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=10_000)
    _open_filters(page)
    page.click("[data-filter-find-input]")
    page.wait_for_selector("[data-filter-found]:not([hidden]) [data-field]", timeout=10_000)
    assert len(_rows(page)) > 5, _rows(page)


def test_typing_a_word_finds_a_field_nobody_told_you_about(page: Page, live: Live, unbroken):
    """The acceptance path.

    `SniffedFormat` is a container key this library really carries --
    written by the reader, named by nothing on screen, and in the
    `container` source that the catalog deliberately RANKS DOWN. So it
    is both the hardest case and the honest one: a person who half
    remembers the word gets there, and a person who never heard of it is
    not made to scroll past it.
    """
    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=10_000)
    _find(page, "sniff")
    found = _rows(page)
    assert any(one["param"] == "SniffedFormat" for one in found), found


def test_the_two_halves_of_the_vocabulary_are_one_list(page: Page, live: Live, unbroken):
    """A fact we named and a string some tool wrote arrive the same way.
    Nothing in a row says which it is -- that distinction is ours, and
    it decides which control opens, not what a person reads."""
    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=10_000)
    # A broad letter, on purpose: this is where the tail used to lose.
    # There are forty-one curated dimensions and thirty rows, so ranked
    # purely by match quality the list filled with names and the long
    # tail was unreachable again -- inside the thing built to reach it.
    _find(page, "o")
    found = _rows(page)
    assert any(one["param"] is None for one in found), f"no curated dimension offered: {found}"
    assert any(one["param"] is not None for one in found), f"no discovered key offered: {found}"


def test_choosing_a_named_field_lands_on_its_own_control(page: Page, live: Live, unbroken):
    """Choosing does not apply a filter -- it takes you to the field's
    own control, which already knows its operators and counts its own
    values. The alternative is a second, poorer copy of the drawer."""
    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=10_000)
    _find(page, "rating")
    page.click('[data-filter-found] [data-field="rating_min"]')
    page.wait_for_function(
        "() => document.querySelector('[data-filter=\"rating_min\"]')?.open === true", timeout=10_000
    )
    # and it FILLED: a section that opens and never counts reads as a
    # control that does nothing
    page.wait_for_function(
        "() => document.querySelector('[data-filter=\"rating_min\"] [data-filter-body]')?.dataset.state === 'ready'",
        timeout=10_000,
    )
    assert page.locator("[data-filter-found]").is_hidden(), "the list stayed open over the control it opened"


def test_choosing_a_discovered_key_fills_in_the_spelling(page: Page, live: Live, unbroken):
    """The whole defect, closed. The long-tail control is a text box
    because there is no curated list of these -- so the key goes in for
    them, and the caret lands after the `=` so the next keystroke is the
    VALUE they actually came to type."""
    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=10_000)
    _find(page, "sniff")
    page.click('[data-filter-found] [data-field][data-param="SniffedFormat"]')
    page.wait_for_function("() => document.querySelector('[data-filter=\"param.is\"]')?.open === true", timeout=10_000)
    box = page.locator('[data-filter="param.is"] input[type="text"]')
    assert box.input_value() == "SniffedFormat=", box.input_value()
    # and the caret is past it, so typing continues the value
    assert page.evaluate(
        '() => { const i = document.querySelector(\'[data-filter="param.is"] input[type="text"]\');'
        " return i.selectionStart === i.value.length; }"
    )


def test_the_field_found_by_typing_actually_filters(page: Page, live: Live, unbroken):
    """End to end and the only claim that matters: a person who knew
    nothing of the schema arrives at the right media."""
    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=10_000)
    assert int(page.get_attribute("[data-grid]", "data-total") or 0) == WHOLE

    _find(page, "sniff")
    page.click('[data-filter-found] [data-field][data-param="SniffedFormat"]')
    box = page.locator('[data-filter="param.is"] input[type="text"]')
    box.fill("SniffedFormat=png")
    page.keyboard.press("Enter")
    page.wait_for_function(
        "() => new URLSearchParams(location.search).getAll('f').some(one => one.startsWith('param.is:'))",
        timeout=10_000,
    )
    page.wait_for_selector("[data-grid]", timeout=10_000)
    assert int(page.get_attribute("[data-grid]", "data-total") or 0) == WHOLE, (
        "every picture here is a png, so asking for pngs must answer with all of them"
    )


def test_the_list_is_walkable_from_the_keyboard(page: Page, live: Live, unbroken):
    """It is a search box. Arrows move, Enter takes, Escape shuts the
    list -- and Escape shuts the LIST, not the drawer: it means undo the
    smallest thing I am doing."""
    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=10_000)
    _find(page, "rating")
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Escape")
    page.wait_for_selector("[data-filter-found]", state="hidden", timeout=10_000)
    assert page.locator("[data-filters-panel]").is_visible(), "Escape closed the drawer as well as the list"


def test_a_word_nothing_answers_to_says_so(page: Page, live: Live, unbroken):
    """An empty list that says nothing reads as a broken box."""
    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=10_000)
    _open_filters(page)
    box = page.locator("[data-filter-find-input]")
    box.click()
    box.fill("qzqzqzqz")
    page.wait_for_function(
        "() => { const l = document.querySelector('[data-filter-found]');"
        " return l && !l.hidden && l.textContent.includes('nothing here answers'); }",
        timeout=10_000,
    )
