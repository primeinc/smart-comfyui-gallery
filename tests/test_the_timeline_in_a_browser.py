"""The timeline page, witnessed in a browser: the URL owns the zoom, a
bar zooms and back returns, the strip carries thumbnails, a session card
opens its pictures."""

from __future__ import annotations

import socket
import threading
import time

import pytest
from PIL import Image

pytestmark = pytest.mark.slow

FILES = 6


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
    import os

    import httpx
    import uvicorn

    from sg_web.app import build_app

    tmp = tmp_path_factory.mktemp("timeline-browser")
    root = tmp / "lib"
    root.mkdir()
    base_at = 1_686_355_200.0 + 14 * 3600  # 2023-06-10 14:00, stamped names every five minutes
    for i in range(FILES):
        path = root / f"Screenshot 2023-06-10 at 14.{i * 5:02d}.0{i}.png"
        Image.new("RGB", (8, 8), (30 * i, 70, 130)).save(path)
        os.utime(path, (base_at + i * 300, base_at + i * 300))
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
        if swept["precache"] is not None:
            _settled(api, swept["precache"])
        assert _settled(api, api.post("/jobs/ingest").json()["id"]) == "done"
        assert _settled(api, api.post("/jobs/context").json()["id"]) == "done"
        assert _settled(api, api.post("/jobs/events").json()["id"]) == "done"
        yield base, api
    server.should_exit = True
    thread.join(timeout=10)


def test_the_url_owns_the_window_and_the_surface_carries_pictures(served):
    from playwright.sync_api import sync_playwright

    base, _api = served
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        # a day-wide window: one bar per hour; a bar opens its hour
        day = 1_686_355_200
        page.goto(base + f"/timeline?start={day}&end={day + 86400}")
        page.wait_for_selector("[data-strip] .bin", timeout=10_000)
        assert page.get_attribute("[data-surface]", "data-window-start") == f"{day}.0"
        page.wait_for_selector("[data-samples] .surface-sample img", timeout=10_000)
        page.wait_for_selector("[data-sessions] .session [data-session-open]", timeout=10_000)
        assert page.locator("[data-sessions] .session-strip img").count() >= 1
        bar_at = page.get_attribute("[data-strip] [data-bin-window]", "data-bin-at")
        assert bar_at is not None
        page.click("[data-strip] [data-bin-window]")
        page.wait_for_function(
            "(at) => new URLSearchParams(location.search).get('start') === at", arg=bar_at, timeout=10_000
        )
        page.wait_for_selector("[data-zoom] a", timeout=10_000)
        assert page.get_attribute("[data-surface]", "data-window-start") == f"{bar_at}.0"
        page.go_back()
        page.wait_for_function(
            "(d) => new URLSearchParams(location.search).get('start') === d", arg=str(day), timeout=10_000
        )
        page.wait_for_function(
            "(d) => document.querySelector('[data-surface]').dataset.windowStart === d", arg=f"{day}.0", timeout=10_000
        )
        page.reload()
        page.wait_for_selector("[data-strip] .bin", timeout=10_000)
        assert page.get_attribute("[data-surface]", "data-window-start") == f"{day}.0"
        href = page.get_attribute("[data-sessions] .session [data-session-open]", "href")
        assert href is not None
        assert "event.id%3Aeq%3A" in href
        page.goto(base + href)
        page.wait_for_selector("[data-chips] [data-chip]", timeout=10_000)
        total = page.get_attribute("[data-grid]", "data-total")
        assert total is not None
        assert int(total) == FILES
        # tell the story: freeze, plan (a job the real worker drains), render, read
        page.goto(base + "/timeline")
        page.wait_for_selector("[data-sessions] .session [data-session-tell]", timeout=10_000)
        page.click("[data-sessions] .session [data-session-tell]")
        page.wait_for_url("**/stories/renders/*", timeout=60_000)
        page.wait_for_selector(".story-heroes img", timeout=10_000)
        assert page.locator(".story-members img").count() >= FILES
        first = page.url
        page.click('[data-story-profile-ask="technical"]')
        page.wait_for_function("(was) => location.href !== was", arg=first, timeout=20_000)
        page.wait_for_selector('[data-story-profile="technical"]', timeout=10_000)
        page.click("[data-story-evolution]")
        page.wait_for_url("**/evolution", timeout=10_000)
        page.wait_for_selector("[data-evolution-story]", timeout=10_000)
        page.goto(base + "/timeline")
        page.wait_for_selector("[data-sessions] .session [data-session-story]", timeout=10_000)
        browser.close()


def test_the_save_view_button_keeps_every_facet(served):
    """A two-facet door saved from the gallery is a two-facet rule: the
    button sends every `f` the mounted answer carries, and the saved
    collection's words name both."""
    from playwright.sync_api import sync_playwright

    base, api = served
    asked = "/g?f=context.local_day%3Aeq%3A2023-06-10&f=context.origin%3Aeq%3Aimported"
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.on("dialog", lambda dialog: dialog.accept("Two facets"))
        page.goto(base + asked)
        page.wait_for_selector("[data-grid]", timeout=10_000)
        assert int(page.get_attribute("[data-grid]", "data-total") or 0) == FILES
        page.click("[data-save-smart]")
        page.wait_for_url("**/t/*", timeout=20_000)
        slug = page.url.rsplit("/t/", 1)[1].split("?", 1)[0]
        browser.close()
    told = api.get(f"/t/{slug}", headers={"accept": "application/json"}).json()
    assert told["rule"] is not None
    assert "context.local_day" in told["rule"]["nl"]
    assert "context.origin" in told["rule"]["nl"], "the second facet was dropped on the way to the rule"
    inside = api.get(f"/g?album={slug}", headers={"accept": "text/html"}).text
    assert f'data-total="{FILES}"' in inside


WIDE_FILES = 9


@pytest.fixture(scope="module")
def served_wide(tmp_path_factory):
    """A library forty days wide, one picture every five days: room for a
    month-wide opening window that is NOT the whole library, and for a
    brush to move."""
    import os

    import httpx
    import uvicorn

    from sg_web.app import build_app

    tmp = tmp_path_factory.mktemp("timeline-browser-wide")
    root = tmp / "lib"
    root.mkdir()
    base_at = 1_686_355_200.0 + 14 * 3600
    for i in range(WIDE_FILES):
        at = base_at + i * 5 * 86400
        day = time.strftime("%Y-%m-%d", time.gmtime(at))
        path = root / f"Screenshot {day} at 14.00.0{i}.png"
        Image.new("RGB", (8, 8), (20 * i, 90, 140)).save(path)
        os.utime(path, (at, at))
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
        if swept["precache"] is not None:
            _settled(api, swept["precache"])
        assert _settled(api, api.post("/jobs/ingest").json()["id"]) == "done"
        assert _settled(api, api.post("/jobs/context").json()["id"]) == "done"
        yield base, api
    server.should_exit = True
    thread.join(timeout=10)


def test_the_window_opens_on_the_last_month_and_the_brush_moves_it(served_wide):
    """A first visit is the last month that holds pictures, never the
    whole library; the overview's brush drags the window and the presets
    set it, each move swapping the one fragment and writing the URL; no
    bar is ever "too many"."""
    from playwright.sync_api import sync_playwright

    base, api = served_wide
    extent = api.get("/timeline/density", params={"bin": "week", "lean": "true"}).json()["extent"]
    whole_end = extent["end"] + 1
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 800})
        page.goto(base + "/timeline")
        page.wait_for_selector("[data-strip] .bin", timeout=10_000)
        start = float(page.get_attribute("[data-surface]", "data-window-start") or "nan")
        end = float(page.get_attribute("[data-surface]", "data-window-end") or "nan")
        assert end == whole_end, "the window ends at the newest picture"
        assert start == end - 30 * 86400, "and opens a month wide"
        assert start > extent["start"], "which is not the whole forty-day library"
        page.wait_for_selector("[data-samples] .surface-sample img", timeout=10_000)
        assert "too many" not in page.inner_text("[data-surface]")
        assert page.locator("[data-overview] [data-brush]").count() == 1
        assert page.locator("[data-zoom] a[data-preset]").count() == 5

        # the brush: a new window drawn across the left half of the overview
        box = page.locator("[data-overview]").bounding_box()
        assert box is not None
        y = box["y"] + box["height"] / 2
        page.mouse.move(box["x"] + 2, y)
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] * 0.5, y, steps=8)
        page.mouse.up()
        page.wait_for_function("() => new URLSearchParams(location.search).has('start')", timeout=10_000)
        page.wait_for_function(
            "(was) => document.querySelector('[data-surface]').dataset.windowEnd !== was", arg=str(end), timeout=10_000
        )
        moved_start = float(page.get_attribute("[data-surface]", "data-window-start") or "nan")
        moved_end = float(page.get_attribute("[data-surface]", "data-window-end") or "nan")
        whole = whole_end - extent["start"]
        assert moved_start < extent["start"] + 0.02 * whole, "the drag began by the overview's left edge"
        assert abs(moved_end - (extent["start"] + whole / 2)) < 0.02 * whole, "and ended at its middle"

        # a preset: the whole library
        page.click('[data-zoom] a[data-preset="all"]')
        page.wait_for_function(
            "(whole) => document.querySelector('[data-surface]').dataset.windowEnd === whole",
            arg=str(float(whole_end)),  # the surface spells its window as the float it holds
            timeout=10_000,
        )
        page.wait_for_selector("[data-strip] .bin", timeout=10_000)
        assert page.locator('[data-zoom] a[data-preset="all"][data-current]').count() == 1
        assert page.locator("[data-samples] .surface-sample img").count() >= 1, "thumbnails at every window"
        browser.close()
