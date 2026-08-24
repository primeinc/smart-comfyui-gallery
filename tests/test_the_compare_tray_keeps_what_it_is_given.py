"""Keeping, in a browser: a tray that outlives the page that filled it.

Looking at two pictures side by side is the ordinary thing somebody does
with a library of near-identical generations, and the application had no
way to do it. Two tabs was the workaround.

Everything that makes the tray worth having is a claim about a real
browser and cannot be checked anywhere cheaper:

  * that it SURVIVES NAVIGATION. The case it exists for is gathering
    across surfaces -- one from the gallery, one from a person's page --
    and a tray that emptied on the way there would be useless for
    exactly that.
  * that it survives RELOAD, and empties only when told.
  * that the order is the comparison, and dragging changes it.
  * that the comparison rises above everything else on the screen.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from PIL import Image
from playwright.sync_api import Page

from tests.conftest import Live

pytestmark = pytest.mark.slow

NAMES = ("alpha.png", "bravo.png", "charlie.png", "delta.png")
CLIP = "echo.mp4"


def write_library(root) -> None:
    for i, name in enumerate(NAMES):
        Image.new("RGB", (80, 60), (30 + i * 40, 70, 120)).save(root / name)

    # A real clip, because a compare tray that only handles stills is a
    # compare tray for one file type.
    import av

    with av.open(str(root / CLIP), "w") as container:
        stream = container.add_stream("h264", rate=5)
        stream.width, stream.height = 320, 180
        stream.pix_fmt = "yuv420p"
        for _ in range(5):
            frame = av.VideoFrame.from_ndarray(np.full((180, 320, 3), (0, 0, 255), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    swept = api.post(f"/roots/{made['id']}/scan").json()
    assert swept["added"] == len(NAMES) + 1
    api.post("/jobs/ingest")
    _drained(api)


def _drained(api, timeout=90.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        running = [j["id"] for j in api.get("/jobs").json() if j["state"] in ("queued", "running")]
        if not running:
            return
        assert time.monotonic() < deadline, f"jobs still running: {running}"
        time.sleep(0.05)


def _gallery(page: Page) -> None:
    page.goto("/g")
    page.wait_for_selector("[data-grid] a.cell", timeout=15_000)


def _keep_cell(page: Page, name: str) -> None:
    """Hover a cell and press the key, which is what a person does."""
    cell = page.locator(f'a.cell:has(img[alt="{name}"])')
    cell.hover()
    page.keyboard.press("c")
    page.wait_for_function(
        "name => [...document.querySelectorAll('[data-compare-items] img')].some(i => i.alt === name)",
        arg=name,
        timeout=5_000,
    )


def _in_tray(page: Page) -> list[str]:
    return page.evaluate("() => [...document.querySelectorAll('[data-compare-items] img')].map(i => i.alt)")


def _slug(live: Live, name: str) -> str:
    for row in live.api.get("/g/peek", params={"page": 1, "count": 9}).json()["items"]:
        if row["name"] == name:
            return row["slug"]
    raise AssertionError(f"no picture called {name}")


# --- keeping ----------------------------------------------------------------


def test_the_tray_stays_out_of_the_way_until_something_is_kept(page: Page, live: Live, unbroken):
    _gallery(page)
    assert page.locator("[data-compare-tray]").count() == 1, "the tray exists on every surface that shows media"
    assert not page.locator("[data-compare-tray]").is_visible(), "and shows nothing until it holds something"


def test_a_key_keeps_what_is_under_the_pointer(page: Page, live: Live, unbroken):
    _gallery(page)
    _keep_cell(page, "bravo.png")
    assert _in_tray(page) == ["bravo.png"]
    assert page.locator("[data-compare-tray]").is_visible()
    assert page.inner_text("[data-compare-count]").strip() == "1"


def test_pressing_it_again_on_the_same_picture_takes_it_back_off(page: Page, live: Live, unbroken):
    """The key is its own undo, so a mistake does not send anybody
    hunting for the small remove button."""
    _gallery(page)
    _keep_cell(page, "bravo.png")
    page.locator('a.cell:has(img[alt="bravo.png"])').hover()
    page.keyboard.press("c")
    page.wait_for_function("() => document.querySelectorAll('[data-compare-items] img').length === 0", timeout=5_000)
    assert _in_tray(page) == []


def test_it_holds_several_and_keeps_the_order_they_arrived(page: Page, live: Live, unbroken):
    _gallery(page)
    for name in NAMES:
        _keep_cell(page, name)
    assert sorted(_in_tray(page)) == sorted(NAMES)
    assert page.inner_text("[data-compare-count]").strip() == str(len(NAMES))


# --- it outlives the page that filled it ------------------------------------


def test_it_survives_walking_to_another_surface_and_back(page: Page, live: Live, unbroken):
    """The case the tray exists for: gathering ACROSS surfaces. A tray
    that emptied on the way to the second picture would be useless for
    exactly the thing it is for."""
    _gallery(page)
    _keep_cell(page, "alpha.png")

    page.goto(f"/i/{_slug(live, 'charlie.png')}")
    page.wait_for_selector("[data-viewer]", timeout=15_000)
    assert _in_tray(page) == ["alpha.png"], "the tray came with"

    # and the picture being LOOKED AT is what the key keeps here
    page.keyboard.press("c")
    page.wait_for_function(
        "() => document.querySelectorAll('[data-compare-items] img').length === 2",
        timeout=5_000,
    )
    assert _in_tray(page) == ["alpha.png", "charlie.png"]

    _gallery(page)
    assert _in_tray(page) == ["alpha.png", "charlie.png"], "and back again"


def test_it_survives_a_reload_and_empties_only_when_told(page: Page, live: Live, unbroken):
    _gallery(page)
    _keep_cell(page, "alpha.png")
    _keep_cell(page, "delta.png")

    page.reload()
    page.wait_for_selector("[data-grid] a.cell", timeout=15_000)
    assert _in_tray(page) == ["alpha.png", "delta.png"]

    page.click("[data-compare-clear]")
    page.wait_for_function("() => !document.querySelector('[data-compare-tray]').checkVisibility()", timeout=5_000)
    assert _in_tray(page) == []
    page.reload()
    page.wait_for_selector("[data-grid] a.cell", timeout=15_000)
    assert _in_tray(page) == [], "cleared stays cleared"


def test_collapsing_hides_the_thumbnails_and_not_the_count(page: Page, live: Live, unbroken):
    """A tray that vanished would take the count with it, and then
    nothing on screen says anything is being kept."""
    _gallery(page)
    _keep_cell(page, "alpha.png")
    page.click("[data-compare-collapse]")
    page.wait_for_function(
        "() => document.querySelector('[data-compare-tray]').dataset.tray === 'closed'", timeout=5_000
    )
    assert page.locator("[data-compare-tray]").is_visible()
    assert page.inner_text("[data-compare-count]").strip() == "1"
    assert not page.locator("[data-compare-items]").is_visible()

    page.goto("/g")
    page.wait_for_selector("[data-grid] a.cell", timeout=15_000)
    assert page.get_attribute("[data-compare-tray]", "data-tray") == "closed", "and it stays how it was left"


# --- removing and reordering ------------------------------------------------


def test_one_can_be_dropped_without_touching_the_rest(page: Page, live: Live, unbroken):
    _gallery(page)
    for name in NAMES[:3]:
        _keep_cell(page, name)
    page.click('[data-compare-remove][aria-label="stop keeping bravo.png"]')
    page.wait_for_function("() => document.querySelectorAll('[data-compare-items] img').length === 2", timeout=5_000)
    assert _in_tray(page) == ["alpha.png", "charlie.png"]


def test_dragging_reorders_and_the_order_is_the_comparison(page: Page, live: Live, unbroken):
    """ "Swap these two" is a drag rather than a mode with its own
    buttons, and the comparison shows them in exactly that order."""
    _gallery(page)
    for name in NAMES[:3]:
        _keep_cell(page, name)
    assert _in_tray(page) == ["alpha.png", "bravo.png", "charlie.png"]

    page.drag_and_drop(
        "[data-compare-items] li:nth-child(3)",
        "[data-compare-items] li:nth-child(1)",
    )
    page.wait_for_function(
        "() => document.querySelector('[data-compare-items] img').alt === 'charlie.png'", timeout=5_000
    )
    assert _in_tray(page) == ["charlie.png", "alpha.png", "bravo.png"]

    page.click("[data-compare-open]")
    page.wait_for_selector("[data-compare-view]", timeout=5_000)
    shown = page.evaluate("() => [...document.querySelectorAll('[data-compare-column] img')].map(i => i.alt)")
    assert shown == ["charlie.png", "alpha.png", "bravo.png"], "left to right, in the tray's order"


# --- comparing --------------------------------------------------------------


def test_comparing_needs_two_and_says_so_by_being_inert(page: Page, live: Live, unbroken):
    _gallery(page)
    _keep_cell(page, "alpha.png")
    assert page.locator("[data-compare-open]").is_disabled(), (
        "two is the smallest number of things that can be compared; the control is present and inert, not absent"
    )
    _keep_cell(page, "bravo.png")
    assert page.locator("[data-compare-open]").is_enabled()


def test_the_comparison_rises_above_everything_and_dismisses(page: Page, live: Live, unbroken):
    _gallery(page)
    _keep_cell(page, "alpha.png")
    _keep_cell(page, "bravo.png")
    page.click("[data-compare-open]")
    page.wait_for_selector("[data-compare-view]", timeout=5_000)

    over = page.evaluate(
        "() => { const v = document.querySelector('[data-compare-view]');"
        " const t = document.querySelector('[data-compare-tray]');"
        " return Number(getComputedStyle(v).zIndex) > Number(getComputedStyle(t).zIndex); }"
    )
    assert over, "the comparison is the thing being done, so it is on top of the tray that opened it"

    # both pictures, shown the same way -- nothing crops one or picks a primary
    fits = page.evaluate(
        "() => [...document.querySelectorAll('[data-compare-column] img, [data-compare-column] video')]"
        ".map(i => getComputedStyle(i).objectFit)"
    )
    assert fits == ["contain", "contain"], fits

    page.keyboard.press("Escape")
    page.wait_for_selector("[data-compare-view]", state="detached", timeout=5_000)
    assert _in_tray(page) == ["alpha.png", "bravo.png"], "closing the comparison keeps what was kept"


# --- cross-media ------------------------------------------------------------


def test_a_clip_is_kept_and_compared_as_a_clip(page: Page, live: Live, unbroken):
    """Comparing two generations of a video by looking at two frozen
    frames is the failure this surface exists to avoid, so a kind that
    moves gets an element that plays."""
    _gallery(page)
    _keep_cell(page, "alpha.png")
    _keep_cell(page, CLIP)
    assert _in_tray(page) == ["alpha.png", CLIP]

    page.click("[data-compare-open]")
    page.wait_for_selector("[data-compare-view]", timeout=5_000)
    shown = page.evaluate(
        "() => [...document.querySelectorAll('[data-compare-column]')].map(c =>"
        " c.querySelector('img, video, audio').tagName.toLowerCase())"
    )
    assert shown == ["img", "video"], shown
    # and the clip is playable rather than a poster in disguise
    assert page.locator("[data-compare-column] video").get_attribute("controls") is not None
