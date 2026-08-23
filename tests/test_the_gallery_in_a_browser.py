"""The basics, witnessed in a browser: the gallery shows pictures, a
click opens one in the lightbox, the arrow walks to the next, Escape
closes, and the picture page opens on its own."""

from __future__ import annotations

import socket
import threading
import time

import pytest
from PIL import Image

pytestmark = pytest.mark.slow

FILES = 5


def _free_port() -> int:
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        return held.getsockname()[1]


def _settled(api, job_id, timeout=60.0) -> str:
    deadline = time.monotonic() + timeout
    while True:
        state = api.get(f"/jobs/{job_id}").json()["state"]
        if state in ("done", "failed", "cancelled"):
            return state
        assert time.monotonic() < deadline, f"job {job_id} still {state}"
        time.sleep(0.05)


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    import httpx
    import uvicorn

    from sg_web.app import build_app

    tmp = tmp_path_factory.mktemp("gallery-browser")
    root = tmp / "lib"
    root.mkdir()
    for i in range(FILES):
        Image.new("RGB", (64, 48), (40 * i, 90, 160)).save(root / f"g_{i:02d}.png")
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            build_app(str(tmp / "run"), worker=True),
            host="127.0.0.1",
            port=port,
            log_level="warning",
            loop="tests.staging:selector_loop",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 20
    while True:
        try:
            if httpx.get(base + "/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        assert time.monotonic() < deadline
        time.sleep(0.1)
    with httpx.Client(base_url=base, timeout=10) as api:
        made = api.post("/roots", json={"path": str(root)}).json()
        swept = api.post(f"/roots/{made['id']}/scan").json()
        assert swept["added"] == FILES
        if swept["precache"] is not None:
            _settled(api, swept["precache"])
        yield base
    server.should_exit = True
    thread.join(timeout=10)


def test_a_picture_can_be_clicked_walked_and_closed(served):
    from playwright.sync_api import sync_playwright

    base = served
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        broken: list[str] = []
        page.on("response", lambda r: broken.append(f"{r.status} {r.url}") if r.status >= 500 else None)
        page.on("pageerror", lambda e: broken.append(f"pageerror {e}"))

        page.goto(base + "/g")
        page.wait_for_selector("[data-grid] a.cell img", timeout=10_000)
        assert page.locator("[data-grid] a.cell").count() == FILES
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
        assert page.url.endswith(first_href) or first_href.split("?")[0] in page.url
        page.wait_for_function(
            "() => { const i = document.querySelector('[data-lightbox] .lightbox-media img');"
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

        page.goto(base + first_href)
        page.wait_for_selector("main", timeout=10_000)
        page.wait_for_function(
            "() => Array.from(document.images).some(i => i.complete && i.naturalWidth > 0)", timeout=10_000
        )
        assert broken == [], broken
        browser.close()


INSIDE = (
    "() => { const p = document.querySelector('[data-rail-pop]'); const r = p.getBoundingClientRect();"
    " const bar = document.querySelector('header.bar').getBoundingClientRect();"
    " p.style.pointerEvents = 'auto';"  # the preview is pointer-events:none by design; elementFromPoint skips such boxes
    " const mid = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);"
    " p.style.pointerEvents = '';"
    " return { top: r.top, bottom: r.bottom, left: r.left, right: r.right, height: r.height,"
    " header: bar.bottom, vh: innerHeight, vw: innerWidth, onTop: p.contains(mid) }; }"
)


def test_the_rail_preview_stays_on_screen_and_on_top(served):
    """Hover the rail at its very top and its very bottom: the preview is
    whole inside the viewport, below the sticky header, and the element
    under its centre is the preview itself -- nothing covers it."""
    from playwright.sync_api import sync_playwright

    base = served
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 900, "height": 500})
        page.goto(base + "/g?size=2")  # three pages: the rail has somewhere to go
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
        browser.close()
