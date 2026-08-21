"""The gallery in a real browser: WI-35's acceptance walk, on screen.

One Chromium drives the whole promised path: a sorted question, normal
paging, a far rail jump that walks no intermediate page, a rail preview
holding real members of the exact destination page, opening an item,
and browser-back restoring the state the URL owns. DOM stays bounded:
one page of cells at a time, never the library.
"""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import time

import pytest
from PIL import Image

pytestmark = pytest.mark.spawns


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def driven(tmp_path_factory):
    """150 stills behind the real server, and one Chromium."""
    import os

    import httpx
    from playwright.sync_api import sync_playwright

    tmp = tmp_path_factory.mktemp("grid")
    root = tmp / "lib"
    root.mkdir()
    for i in range(150):
        path = root / f"pic_{i:03d}.png"
        Image.new("RGB", (12, 12), (i % 256, 80, 120)).save(path)
        os.utime(path, (1_700_000_000 + i * 60, 1_700_000_000 + i * 60))

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    server_log = (tmp / "server.log").open("wb")
    server = None
    try:
        server = subprocess.Popen(
            [sys.executable, "-m", "sg_web", "--home", str(tmp / "run"), "--port", str(port)],
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        with httpx.Client(base_url=base, timeout=5.0) as web:
            deadline = time.time() + 30
            while True:
                try:
                    if web.get("/health").text == "ok":
                        break
                except httpx.TransportError:
                    if time.time() > deadline:
                        raise
                    time.sleep(0.2)
            made = web.post("/roots", json={"path": str(root)}).json()
            assert web.post(f"/roots/{made['id']}/scan").json()["added"] == 150

        with sync_playwright() as p:
            browser = p.chromium.launch()
            yield browser, base
            browser.close()
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=15)
        server_log.close()


def test_the_promised_walk_holds_on_screen(driven):
    browser, base = driven
    page = browser.new_page()
    fetched: list[str] = []
    page.on("request", lambda request: fetched.append(request.url))
    try:
        # A sorted question, from the URL alone. size=30 makes five pages,
        # so the rail jump below is a real skip over the middle.
        page.goto(f"{base}/g?sort=oldest&size=30")
        page.wait_for_selector("[data-grid]")
        assert page.locator(".cell").count() == 30, "the DOM holds one page, not the library"
        assert page.get_attribute("[data-grid]", "data-pages") == "5"
        first = page.locator(".cell").first.get_attribute("data-slug")
        assert first is not None
        assert "pic-000" in first, "sort=oldest must lead with the oldest"

        # Normal paging: the htmx swap moves the grid AND the address bar.
        page.click('[data-pager] a[rel="next"]')
        page.wait_for_function("() => document.querySelector('[data-grid]').dataset.page === '2'")
        assert "page=2" in page.url
        ordinal = page.locator(".cell").first.get_attribute("data-ordinal")
        assert ordinal == "31"

        # The rail preview: hover near the bottom shows real members of
        # the page a click there would land on.
        rail = page.locator("[data-rail]")
        box = rail.bounding_box()
        assert box is not None
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] - 2)
        page.wait_for_selector("[data-rail-pop]:not([hidden]) [data-rail-pop-grid] img")
        label = page.text_content("[data-rail-pop-label]")
        assert label is not None
        assert "page 5 of 5" in label
        shown = page.eval_on_selector_all("[data-rail-pop-grid] img", "els => els.map(e => new URL(e.src).pathname)")
        import httpx

        with httpx.Client(base_url=base, timeout=5.0) as web:
            told = web.get("/g/peek", params={"sort": "oldest", "size": 30, "page": 5, "count": 9}).json()
        assert shown == [f"/thumb/{item['slug']}" for item in told["items"]], (
            "the preview must be the exact destination page's members"
        )

        # The far jump: one navigation to page 5, no walk through 3 and 4.
        fetched.clear()
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] - 2)
        page.wait_for_function("() => document.querySelector('[data-grid]').dataset.page === '5'")
        assert "page=5" in page.url
        walked = [url for url in fetched if re.search(r"page=[34]\b", url)]
        assert walked == [], f"random access walked prior pages: {walked}"
        last = page.locator(".cell").last.get_attribute("data-slug")
        assert last is not None
        assert "pic-149" in last

        # Opening an item is a navigation to its own address.
        opened = page.locator(".cell").first.get_attribute("href")
        page.click(".cell")
        page.wait_for_url(f"**{opened}")

        # And browser-back restores the state the URL was carrying.
        page.go_back()
        page.wait_for_selector("[data-grid]")
        assert "page=5" in page.url
        assert page.get_attribute("[data-grid]", "data-page") == "5"
        assert page.locator(".cell").count() == 30
    finally:
        page.close()
