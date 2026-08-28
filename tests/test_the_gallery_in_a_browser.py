"""The basics, witnessed in a browser: the gallery shows pictures, a
click opens one in the lightbox, the arrow walks to the next, Escape
closes, the picture page opens on its own, and the rail's preview sits
on screen and on top wherever the pointer rests."""

from __future__ import annotations

import time

import pytest
from PIL import Image
from playwright.sync_api import Page, expect

from tests.conftest import POLL, Live

pytestmark = pytest.mark.slow

FILES = 5


def write_library(root) -> None:
    for i in range(FILES):
        Image.new("RGB", (64, 48), (40 * i, 90, 160)).save(root / f"g_{i:02d}.png")


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    swept = api.post(f"/roots/{made['id']}/scan").json()
    assert swept["added"] == FILES
    if swept["precache"] is not None:
        _settled(api, swept["precache"])


def _settled(api, job_id, timeout=60.0) -> str:
    deadline = time.monotonic() + timeout
    while True:
        state = api.get(f"/jobs/{job_id}").json()["state"]
        if state in ("done", "failed", "cancelled"):
            return state
        assert time.monotonic() < deadline, f"job {job_id} still {state}"
        time.sleep(POLL)


def test_a_picture_can_be_clicked_walked_and_closed(page: Page, live: Live, unbroken):
    # `unbroken` watches every first-party answer, not only 500s: a script
    # that 404s is a page which renders and does nothing, which is how a
    # dead viewer sat behind a green suite (tests/conftest.py).

    page.goto("/g")
    page.wait_for_selector("[data-grid] a.cell img", timeout=10_000)
    # `to_have_count`, not `count() ==`: the first cell having an image
    # does not mean the last cell exists yet, and `count()` answers about
    # the instant it is asked rather than about the page. A web-first
    # assertion retries. Measured: this line, under four workers.
    expect(page.locator("[data-grid] a.cell")).to_have_count(FILES)
    # every thumbnail really loaded: natural size, not a broken image
    page.wait_for_function(
        "() => Array.from(document.querySelectorAll('[data-grid] a.cell img')).every(i => i.complete)",
        timeout=10_000,
    )
    loaded = page.evaluate(
        "() => Array.from(document.querySelectorAll('[data-grid] a.cell img')).map(i => i.naturalWidth > 0)"
    )
    assert loaded == [True] * FILES, loaded

    first_href = page.get_attribute("[data-grid] a.cell", "href")
    assert first_href is not None
    page.click("[data-grid] a.cell")
    page.wait_for_selector("[data-lightbox-root]:not([hidden]) [data-lightbox]", timeout=10_000)
    assert first_href.split("?")[0] in page.url
    page.wait_for_function(
        "() => { const i = document.querySelector('[data-lightbox] [data-stage] img[data-stage-media]');"
        " return i && i.complete && i.naturalWidth > 0; }",
        timeout=10_000,
    )
    opened = page.get_attribute("[data-lightbox]", "data-slug")

    page.keyboard.press("ArrowRight")
    page.wait_for_function(
        "(was) => { const l = document.querySelector('[data-lightbox]'); return l && l.dataset.slug !== was; }",
        arg=opened,
        timeout=10_000,
    )
    walked = page.get_attribute("[data-lightbox]", "data-slug")
    assert walked != opened
    assert f"/i/{walked}" in page.url

    page.keyboard.press("Escape")
    # wait_for_selector defaults to state="visible"; a hidden root is attached, not visible
    page.wait_for_selector("[data-lightbox-root][hidden]", state="attached", timeout=10_000)
    assert page.url.rstrip("/").endswith("/g") or "/g?" in page.url

    page.goto(first_href)
    # No wait for `main` first: the poll below runs until a picture has
    # loaded, which cannot happen before the page it is on exists.
    page.wait_for_function(
        "() => Array.from(document.images).some(i => i.complete && i.naturalWidth > 0)", timeout=10_000
    )
    # what the browser found is asserted by `unbroken` as the test ends


INSIDE = (
    "() => { const p = document.querySelector('[data-rail-pop]'); const r = p.getBoundingClientRect();"
    " const bar = document.querySelector('header.bar').getBoundingClientRect();"
    " p.style.pointerEvents = 'auto';"  # pointer-events:none by design; elementFromPoint skips such boxes
    " const mid = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);"
    " p.style.pointerEvents = '';"
    " return { top: r.top, bottom: r.bottom, left: r.left, right: r.right, height: r.height,"
    " header: bar.bottom, vh: innerHeight, vw: innerWidth, onTop: p.contains(mid) }; }"
)


@pytest.mark.browser_context_args(viewport={"width": 900, "height": 500})
def test_the_rail_preview_stays_on_screen_and_on_top(page: Page, live: Live):
    """Hover the rail at its very top and its very bottom: the preview is
    whole inside the viewport, below the sticky header, and the element
    under its centre is the preview itself -- nothing covers it."""
    page.goto("/g?size=2")  # three pages: the rail has somewhere to go
    page.wait_for_selector("[data-grid] a.cell img", timeout=10_000)
    rail = page.locator("[data-rail]").bounding_box()
    assert rail is not None
    x = rail["x"] + rail["width"] / 2
    for y in (rail["y"] + 1, rail["y"] + rail["height"] - 1, rail["y"] + rail["height"] / 2):
        page.mouse.move(x, y)
        page.wait_for_selector("[data-rail-pop]:not([hidden]) [data-rail-pop-grid] img", timeout=10_000)
        page.wait_for_function(
            "() => Array.from(document.querySelectorAll('[data-rail-pop-grid] img')).every(i => i.complete)",
            timeout=10_000,
        )
        told = page.evaluate(INSIDE)
        assert told["top"] >= told["header"], told
        assert told["bottom"] <= told["vh"], told
        assert told["left"] >= 0, told
        assert told["right"] <= told["vw"], told
        assert told["onTop"] is True, told
        page.mouse.move(10, 10)
        page.wait_for_selector("[data-rail-pop][hidden]", state="attached", timeout=10_000)
