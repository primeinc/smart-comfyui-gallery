"""Asking, in a browser: the filter surface as a person meets it.

The vocabulary and the counting are proved without a browser
(test_the_query_vocabulary_is_one_module.py). What needs one is
everything that decides whether a person can actually USE them:

  * that the door is VISIBLE. A filter surface reachable only by knowing
    a keyboard shortcut is a filter surface for whoever wrote it, and no
    unit test can tell the difference.
  * that the URL stays the question. Reload, a pasted link and Back are
    claims about the browser, and they are the difference between a
    shared link and a screenshot.
  * that filtering is ONE editing session. "Back goes to what I was
    looking at" cannot be asserted anywhere but a real history stack.
  * that what is REMEMBERED is the furniture and never the question --
    the drawer stays open, the filters do not follow you home.

The library is mixed on purpose: generated stills of two recipes, a
photograph, and a real video. A filter surface proved only on generated
images is a surface that will surprise somebody the first time they ask
about a clip.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from playwright.sync_api import Page, expect

from tests.conftest import POLL, Live

pytestmark = pytest.mark.slow

#: Four of one recipe and two of another, so a wrong count is visible.
#: With one of each, every wrong answer is still 1.
MADE = [
    *[("dreamshaper_8", "filmGrain", "Euler a")] * 4,
    *[("juggernautXL", "detailTweaker", "DPM++ 2M")] * 2,
]


def _recipe(checkpoint: str, lora: str, sampler: str) -> str:
    return (
        f"a brass diving helmet at dusk <lora:{lora}:0.35>\n"
        "Negative prompt: blurry\n"
        f"Steps: 28, Sampler: {sampler}, CFG scale: 7, Seed: 4242, Size: 832x1216, "
        f"Model: {checkpoint}"
    )


def write_library(root) -> None:
    for i, (checkpoint, lora, sampler) in enumerate(MADE):
        info = PngInfo()
        info.add_text("parameters", _recipe(checkpoint, lora, sampler))
        Image.new("RGB", (64, 48), (20 + i * 7, 60, 90)).save(root / f"made_{i:02d}.png", pnginfo=info)
    Image.new("RGB", (64, 48), (10, 120, 10)).save(root / "taken.png")

    import av

    with av.open(str(root / "clip.mp4"), "w") as container:
        stream = container.add_stream("h264", rate=5)
        stream.width, stream.height = 320, 180
        stream.pix_fmt = "yuv420p"
        for _ in range(5):
            frame = av.VideoFrame.from_ndarray(np.full((180, 320, 3), (0, 0, 255), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


#: Everything the library holds: the recipes, the photograph, the clip.
WHOLE = len(MADE) + 2


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    swept = api.post(f"/roots/{made['id']}/scan").json()
    assert swept["added"] == WHOLE
    api.post("/jobs/ingest")
    _drained(api)


def _drained(api, timeout=90.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        running = [j["id"] for j in api.get("/jobs").json() if j["state"] in ("queued", "running")]
        if not running:
            return
        assert time.monotonic() < deadline, f"jobs still running: {running}"
        time.sleep(POLL)


# --- reading the surface ----------------------------------------------------


def _open_gallery(page: Page) -> None:
    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=15_000)


def _open_filters(page: Page) -> None:
    if page.get_attribute("[data-filters-open]", "aria-expanded") != "true":
        page.click("[data-filters-open]")
    page.wait_for_selector("[data-filters-panel]:not([hidden])", timeout=5_000)


def _open_dimension(page: Page, key: str) -> None:
    """Disclose one filter and wait for its counted values."""
    _open_filters(page)
    section = page.locator(f'[data-filter="{key}"]')
    if not section.evaluate("s => s.open"):
        section.locator("summary").click()
    # on `state`, not on the absence of the word "counting": right after
    # the click the body is still EMPTY, which also contains no such word,
    # and waiting on that returned before the fetch had begun
    page.wait_for_function(
        "key => document.querySelector(`[data-filter=\"${key}\"] [data-filter-body]`)?.dataset.state === 'ready'",
        arg=key,
        timeout=10_000,
    )


def _values(page: Page, key: str) -> dict[str, int]:
    return page.evaluate(
        'key => Object.fromEntries([...document.querySelectorAll(`[data-filter="${key}"] [data-option]`)]'
        ".map(row => [row.dataset.label, Number(row.querySelector('.filter-option-count').textContent"
        ".replace(/[^0-9]/g, ''))]))",
        key,
    )


def _cells(page: Page) -> int:
    return page.evaluate("() => document.querySelectorAll('[data-grid] a.cell').length")


def _chips(page: Page) -> list[str]:
    return page.evaluate("() => [...document.querySelectorAll('[data-chip-edit]')].map(c => c.textContent.trim())")


def _pick(page: Page, key: str, label: str) -> None:
    _open_dimension(page, key)
    page.click(
        f'[data-filter="{key}"] [data-option="{label}"] .filter-option, '
        f'[data-filter="{key}"] [data-label="{label}"] .filter-option'
    )
    # Not a no-op, though `[data-grid]` is there before the click as well:
    # choosing a value REPLACES the grid, so this waits for the new one.
    # Removing it fails `test_a_dimensions_own_list_still_offers_what_it
    # _would_give`, which then reads the grid it just replaced.
    page.wait_for_selector("[data-grid]", timeout=15_000)


# --- the door is visible ----------------------------------------------------


def test_the_filters_door_is_always_there_and_says_how_many(page: Page, live: Live, unbroken):
    """Discoverability first, shortcut second. The control is visible on
    a page nobody has configured, and it carries the count."""
    _open_gallery(page)
    door = page.locator("[data-filters-open]")
    assert door.is_visible(), "the way into filtering must not be a keyboard secret"
    assert door.get_attribute("aria-expanded") == "false"
    assert page.locator("[data-filters-held]").count() == 0, "no filters held, so no badge"

    _open_filters(page)
    assert page.locator("[data-filters-panel]").is_visible()
    # the sections come from the vocabulary, not from the template
    groups = page.evaluate(
        "() => [...document.querySelectorAll('[data-filter-group]')].map(g => g.dataset.filterGroup)"
    )
    assert {"mine", "media", "generation", "camera", "time"} <= set(groups), groups


def test_the_surface_offers_far_more_than_the_header_ever_did(page: Page, live: Live, unbroken):
    """The header carried three questions out of a registry of thirty.
    This is the number that changed."""
    _open_gallery(page)
    _open_filters(page)
    offered = page.evaluate("() => [...document.querySelectorAll('[data-filter]')].map(d => d.dataset.filter)")
    assert len(offered) >= 20, f"only {len(offered)} filters reached the surface"
    for key in ("has.generation", "generation.checkpoint", "generation.lora", "media.kind", "capture.iso"):
        assert key in offered, f"{key} is answerable and was not offered"
    # and the machine's own links are described but never listed
    assert "context.moment" not in offered
    assert "event.id" not in offered


# --- counts, and what they mean ---------------------------------------------


def test_the_values_are_counted_and_the_counts_are_real(page: Page, live: Live, unbroken):
    _open_gallery(page)
    assert _cells(page) == WHOLE
    assert _values(page, "generation.checkpoint") == {}  # not opened yet
    _open_dimension(page, "generation.checkpoint")
    assert _values(page, "generation.checkpoint") == {"dreamshaper_8": 4, "juggernautXL": 2}

    _open_dimension(page, "media.kind")
    assert _values(page, "media.kind") == {"image": WHOLE - 1, "video": 1}


def test_choosing_a_value_narrows_the_answer_and_says_so(page: Page, live: Live, unbroken):
    _open_gallery(page)
    _pick(page, "generation.checkpoint", "dreamshaper_8")

    assert _cells(page) == 4
    assert _chips(page) == ["checkpoint dreamshaper_8"], "the chip reads as words, not as a key"
    assert "f=generation.checkpoint" in page.url, page.url
    assert "4 results" in page.inner_text("[data-total-count]")
    assert page.inner_text("[data-filters-held]").strip() == "1"


def test_a_checkpoint_and_a_lora_compose(page: Page, live: Live, unbroken):
    """The `artifact` scope holds exactly one, so "this checkpoint with
    that LoRA" was unaskable from any surface. Two chips, one answer."""
    _open_gallery(page)
    _pick(page, "generation.checkpoint", "dreamshaper_8")
    _pick(page, "generation.lora", "filmGrain")

    assert sorted(_chips(page)) == ["LoRA filmGrain", "checkpoint dreamshaper_8"]
    assert _cells(page) == 4
    assert page.inner_text("[data-filters-held]").strip() == "2"


def test_a_dimensions_own_list_still_offers_what_it_would_give(page: Page, live: Live, unbroken):
    """Disjunctive faceting, where a person can see it.

    Counted against the whole question, choosing dreamshaper would leave
    juggernaut reading 0 and the list a person opened to change their
    mind could only agree with them.
    """
    _open_gallery(page)
    _pick(page, "generation.checkpoint", "dreamshaper_8")
    assert _values(page, "generation.checkpoint") == {"dreamshaper_8": 4, "juggernautXL": 2}, (
        "the other checkpoint must still say what it would give"
    )
    # and a different dimension IS narrowed by the choice
    _open_dimension(page, "generation.sampler")
    assert _values(page, "generation.sampler") == {"Euler a": 4}


def test_choosing_the_held_value_again_takes_it_off(page: Page, live: Live, unbroken):
    _open_gallery(page)
    _pick(page, "generation.checkpoint", "dreamshaper_8")
    assert _cells(page) == 4
    _pick(page, "generation.checkpoint", "dreamshaper_8")
    assert _cells(page) == WHOLE
    assert _chips(page) == []


# --- cross-media ------------------------------------------------------------


def test_the_surface_is_not_a_feature_for_one_file_type(page: Page, live: Live, unbroken):
    """A video is as filterable as a picture, and the sections offered
    follow the medium being asked about."""
    _open_gallery(page)
    _pick(page, "media.kind", "video")
    assert _cells(page) == 1
    assert _chips(page) == ["kind video"]

    _open_filters(page)
    offered = page.evaluate("() => [...document.querySelectorAll('[data-filter]')].map(d => d.dataset.filter)")
    assert "media.duration" in offered, "a clip has a length and it is askable"

    page.goto("/g?kind=image")
    page.wait_for_selector("[data-grid]", timeout=15_000)
    _open_filters(page)
    stills = page.evaluate("() => [...document.querySelectorAll('[data-filter]')].map(d => d.dataset.filter)")
    assert "media.duration" not in stills, "a still picture has no length; offering it offers an empty answer"
    assert "media.width" in stills


def test_ai_generated_is_asked_of_the_fact_not_the_interpretation(page: Page, live: Live, unbroken):
    """`has.generation` answers before any context job has run, which is
    the state this library is in and the state a fresh one is in."""
    _open_gallery(page)
    _pick(page, "has.generation", "yes")
    assert _cells(page) == len(MADE)
    assert _chips(page) == ["AI generated yes"]

    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=15_000)
    _pick(page, "has.generation", "no")
    assert _cells(page) == 2, "the photograph and the clip"


# --- the URL is the question ------------------------------------------------


def test_the_question_survives_a_reload_and_a_fresh_browser(page: Page, live: Live, unbroken):
    _open_gallery(page)
    _pick(page, "generation.checkpoint", "dreamshaper_8")
    _pick(page, "generation.lora", "filmGrain")
    asked, cells, chips = page.url, _cells(page), sorted(_chips(page))

    page.reload()
    # The same retrying read, for the same reason: a reload the test asked
    # for can be followed by one the surface asks for itself.
    expect(page.locator("[data-grid] a.cell")).to_have_count(cells)
    assert sorted(_chips(page)) == chips

    # a different browsing context entirely: the link is the question
    fresh = page.context.browser.new_context(base_url=live.url) if page.context.browser else None
    assert fresh is not None
    try:
        other = fresh.new_page()
        other.goto(asked)
        # Through `expect`, which RETRIES across a navigation. This surface
        # settles by asking `/g/locate/{slug}` and reloads itself when that
        # answers with an error (frontend/src/authored.ts:113-121), so the
        # first read after a `goto` can meet a document that is being
        # replaced -- and a one-shot `evaluate` dies on it with "Execution
        # context was destroyed" while a locator re-reads whatever replaced
        # it. Measured: this line, one full-suite run in two or three.
        expect(other.locator("[data-grid] a.cell")).to_have_count(cells)
        # The chips read as a WAIT, not as an evaluate-then-assert. One
        # `expect` above absorbs one reload; this surface can ask for a
        # second, and the raw read below it died on exactly that
        # (measured, line 318, under six workers). `wait_for_function` is
        # the same claim -- these chips, sorted -- made of the page
        # instead of of an instant, and it re-runs after a navigation.
        other.wait_for_function(
            "(want) => [...document.querySelectorAll('[data-chip-edit]')]"
            ".map(c => c.textContent.trim()).sort().join('|') === want",
            arg="|".join(chips),
            timeout=15_000,
        )
    finally:
        fresh.close()


def test_removing_a_chip_widens_the_answer(page: Page, live: Live, unbroken):
    _open_gallery(page)
    _pick(page, "generation.checkpoint", "dreamshaper_8")
    _pick(page, "generation.lora", "filmGrain")
    assert _cells(page) == 4

    page.click('.chip:has([data-chip-edit="generation.lora"]) [data-chip-remove]')
    page.wait_for_selector("[data-grid]", timeout=15_000)
    assert _cells(page) >= 4
    assert len(_chips(page)) == 1


def test_a_chip_opens_the_filter_that_made_it(page: Page, live: Live, unbroken):
    """The relationship between a chip and its filter is something a
    person can see, rather than something they have to be told."""
    _open_gallery(page)
    _pick(page, "generation.checkpoint", "dreamshaper_8")
    page.click("[data-filters-close]")
    page.wait_for_selector("[data-filters-panel]", state="hidden", timeout=5_000)

    page.click('[data-chip-edit="generation.checkpoint"]')
    page.wait_for_selector("[data-filters-panel]:not([hidden])", timeout=5_000)
    assert page.locator('[data-filter="generation.checkpoint"]').evaluate("s => s.open")


# --- what is remembered, and what is not ------------------------------------


def test_the_drawer_is_remembered_and_the_filters_are_not(page: Page, live: Live, unbroken):
    """The furniture is workspace state. The question is the URL's, and
    a filter that outlived its URL would mean one link answering
    differently for two people."""
    _open_gallery(page)
    _open_filters(page)
    _open_dimension(page, "generation.checkpoint")

    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=15_000)
    assert page.locator("[data-filters-panel]").is_visible(), "the drawer stays how it was left"
    assert page.locator('[data-filter="generation.checkpoint"]').evaluate("s => s.open"), (
        "and so does the section that was disclosed"
    )
    assert _chips(page) == [], "but no filter followed the person to a bare /g"
    assert _cells(page) == WHOLE

    page.click("[data-filters-close]")
    page.goto("/g")
    page.wait_for_selector("[data-grid]", timeout=15_000)
    assert not page.locator("[data-filters-panel]").is_visible()


def test_filtering_is_one_editing_session_not_fourteen(page: Page, live: Live, unbroken):
    """Back means "the question I had before I started filtering", not
    six presses of undo-one-clause."""
    page.goto("/g?kind=image")
    page.wait_for_selector("[data-grid]", timeout=15_000)
    was = page.url

    _pick(page, "generation.checkpoint", "dreamshaper_8")
    _pick(page, "generation.lora", "filmGrain")
    _pick(page, "generation.sampler", "Euler a")
    assert _cells(page) == 4

    page.go_back()
    page.wait_for_selector("[data-grid]", timeout=15_000)
    assert page.url.endswith("/g?kind=image"), f"Back landed on {page.url}, not {was}"
    assert _cells(page) == WHOLE - 1, "every still picture, which is where the filtering began"


def test_slash_puts_the_caret_in_the_search_box(page: Page, live: Live, unbroken):
    """What `/` does everywhere. The BUTTON is the way into filtering and
    it is always visible; this is a convenience on top of it, never
    instead of it -- and deliberately not a letter, because `f` has been
    favourite since authored.ts claimed it and one keystroke has one
    meaning."""
    _open_gallery(page)
    page.keyboard.press("/")
    focused = page.evaluate("() => document.activeElement?.getAttribute('type')")
    assert focused == "search", focused
    # and typing into it does not fire commands: a search for "focus"
    # must not turn the lights out on the way past the l
    page.keyboard.type("lantern")
    assert page.input_value('[data-ask] input[type="search"]') == "lantern"


# --- saying more than one thing about one dimension -------------------------


def test_choosing_two_kinds_means_either_and_reads_as_one_chip(page: Page, live: Live, unbroken):
    """The most ordinary multi-select there is, and it was unaskable: a
    scope holds one value, and repeated facets ANDed -- which for a
    dimension a file has exactly one of answers nothing, every time."""
    _open_gallery(page)
    _pick(page, "media.kind", "image")
    assert _cells(page) == WHOLE - 1
    _pick(page, "media.kind", "video")
    assert _cells(page) == WHOLE, "either kind, not both at once"

    # ONE chip, because an OR group is one thing the question says.
    # Two chips would read exactly like two ANDed clauses.
    assert _chips(page) == ["kind image or video"], _chips(page)


def test_any_and_all_are_offered_only_where_both_are_real(page: Page, live: Live, unbroken):
    """A file has one kind, so "all of these kinds" is a question that
    answers nothing by construction and is not offered. A picture
    carries several LoRAs, so both readings are real."""
    _open_gallery(page)
    _open_dimension(page, "media.kind")
    assert page.locator('[data-filter-choice="media.kind"]').count() == 0

    _open_dimension(page, "generation.lora")
    assert page.locator('[data-filter-choice="generation.lora"]').count() == 1


def test_switching_to_all_respells_the_clauses_already_held(page: Page, live: Live, unbroken):
    """The switch changes the QUESTION, not merely the next click -- or
    the control would sit there disagreeing with the chips above it."""
    _open_gallery(page)
    _pick(page, "generation.lora", "filmGrain")
    _pick(page, "generation.lora", "detailTweaker")
    assert _cells(page) == len(MADE), "either LoRA"
    assert _chips(page) == ["LoRA filmGrain or detailTweaker"], _chips(page)

    _open_dimension(page, "generation.lora")
    page.click('[data-filter-choice="generation.lora"] [data-mode="all"]')
    page.wait_for_selector("[data-grid]", timeout=15_000)
    assert _cells(page) == 0, "no picture in this library used both"
    assert _chips(page) == ["LoRA all of filmGrain, detailTweaker"], _chips(page)


def test_the_advanced_door_is_there_and_asks_the_long_tail(page: Page, live: Live, unbroken):
    """The schema records every key any tool emitted and registers it,
    and `param_key`'s own comment says the registry is what the facet UI
    is generated from. Nothing generated one."""
    _open_gallery(page)
    _open_filters(page)
    groups = page.evaluate(
        "() => [...document.querySelectorAll('[data-filter-group]')].map(g => g.dataset.filterGroup)"
    )
    assert "advanced" in groups, groups
    assert groups[-1] == "advanced", "last, so four hundred discovered keys do not bury the curated twenty"

    _open_dimension(page, "param.has")
    offered = _values(page, "param.has")
    assert offered, "the registry has keys and they are listed with counts"
    # Discovered, not curated: these are whatever the tools that made
    # this library happened to write, which is the whole point of the
    # door. Asserting a PARTICULAR key would be asserting the parsers'
    # current output, which is not what this surface promises.
    picked = next(iter(offered))
    _pick(page, "param.has", picked)
    assert _cells(page) == offered[picked], f"choosing {picked!r} leaves exactly what it was counted at"
    assert _chips(page) == [f"carries the field {picked}"], _chips(page)


def test_folder_and_album_offer_their_values_now(page: Page, live: Live, unbroken):
    """They held a slug and offered no list, so the drawer showed a
    heading with nothing under it -- a filter you can only use if you
    already know the answer."""
    _open_gallery(page)
    _open_dimension(page, "folder")
    assert _values(page, "folder"), "a folder list, counted"
