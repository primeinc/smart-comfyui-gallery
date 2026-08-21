"""The person drawer in a real browser: the second addressable overlay.

Same history contract the lightbox proved -- open pushes /p/{slug} over
the mounted People index, Escape/Back leave in one step, Forward
re-opens what the URL names, a direct visit renders the full profile --
plus the page's primary action: naming a person from the drawer's form
lands the browser on the freshly minted address.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.spawns


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def peopled(tmp_path_factory):
    import httpx
    from playwright.sync_api import sync_playwright
    from test_a_person_is_an_address_with_two_looks import _clustered_library

    tmp = tmp_path_factory.mktemp("drawer")
    burrow, _ = _clustered_library(tmp)

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    server_log = (tmp / "server.log").open("wb")
    server = None
    try:
        server = subprocess.Popen(
            [sys.executable, "-m", "sg_web", "--home", str(burrow), "--port", str(port)],
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


def test_the_drawer_is_an_address_and_naming_moves_it(peopled):
    browser, base = peopled
    page = browser.new_page()
    try:
        page.goto(f"{base}/people")
        page.wait_for_selector('[data-person="ana"]')

        # Open: one PUSH, the index still mounted underneath.
        page.click('[data-person="ana"]')
        page.wait_for_selector("[data-drawer]")
        assert page.url.endswith("/p/ana")
        assert page.locator("[data-people]").count() == 1, "the People index must stay mounted behind the drawer"
        assert page.locator("[data-drawer] h2").inner_text() == "Ana"

        # Escape is Back: one step, drawer gone, index intact.
        page.keyboard.press("Escape")
        page.wait_for_function("() => window.location.pathname === '/people'")
        page.wait_for_function("() => document.querySelector('[data-drawer-root]').hidden === true")

        # Forward re-opens what the URL names.
        page.go_forward()
        page.wait_for_selector("[data-drawer]")
        assert page.url.endswith("/p/ana")

        # Naming from the drawer mints the new address and lands on it
        # by REPLACEMENT -- the retired slug is not a history stop.
        page.fill("[data-drawer] [data-rename] input", "Ana Torres")
        page.click("[data-drawer] [data-rename] button")
        page.wait_for_function("() => window.location.pathname === '/p/ana-torres'")
        page.wait_for_selector(".person-hero")
        assert page.locator("h1").inner_text() == "Ana Torres"

        # ONE Back from the renamed profile is /people -- never a bounce
        # through the retired address's 301.
        page.go_back()
        page.wait_for_function("() => window.location.pathname === '/people'")
        page.go_forward()
        page.wait_for_function("() => window.location.pathname === '/p/ana-torres'")

        # A direct visit is the complete profile, and Escape goes to the
        # People index, never blindly off-site.
        page.goto(f"{base}/p/ana-torres")
        page.wait_for_selector(".person-hero")
        assert page.locator('link[rel="canonical"][href="/p/ana-torres"]').count() == 1
        page.keyboard.press("Escape")
        page.wait_for_function("() => window.location.pathname === '/people'")

        # The retired address still finds them.
        page.goto(f"{base}/p/ana")
        page.wait_for_function("() => window.location.pathname === '/p/ana-torres'")
    finally:
        page.close()


def test_clicking_the_backdrop_dismisses_the_drawer_like_back(peopled):
    browser, base = peopled
    page = browser.new_page()
    try:
        page.goto(f"{base}/people")
        page.wait_for_selector("[data-person]")
        page.click("[data-person]")
        page.wait_for_selector("[data-drawer]")

        # A click ON the drawer must not dismiss it.
        page.click("[data-drawer] h2")
        assert page.locator("[data-drawer]").count() == 1

        size = page.viewport_size
        assert size is not None
        page.mouse.click(4, size["height"] // 2)
        page.wait_for_function("() => window.location.pathname === '/people'")
        page.wait_for_function("() => document.querySelector('[data-drawer-root]').hidden === true")
    finally:
        page.close()
