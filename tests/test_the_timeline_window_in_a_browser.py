"""The timeline's window, witnessed in a browser over a forty-day library:
a first visit is the last month that holds pictures; the brush moves the
window and the stage moves WHILE it is dragged; a preset sets it; a job
that groups pictures lands on the page by itself."""

from __future__ import annotations

import os
import time

import pytest
from PIL import Image
from playwright.sync_api import Page

from tests.conftest import Live

pytestmark = [pytest.mark.slow, pytest.mark.browser_context_args(viewport={"width": 1200, "height": 800})]

DAYS = 9


def write_library(root) -> None:
    """Nine days five days apart, a pair of pictures a minute apart on
    each (enough for a group to form); the name carries the clock
    (14:0j:0i) and the file's mtime says the same moment."""
    base_at = 1_686_355_200.0 + 14 * 3600
    for i in range(DAYS):
        for j in range(2):
            at = base_at + i * 5 * 86400 + j * 60 + i
            day = time.strftime("%Y-%m-%d", time.gmtime(at))
            path = root / f"Screenshot {day} at 14.0{j}.0{i}.png"
            Image.new("RGB", (8, 8), (20 * i, 90 + 40 * j, 140)).save(path)
            os.utime(path, (at, at))


def moments() -> list[float]:
    """Every moment `write_library` wrote, from the same numbers."""
    base_at = 1_686_355_200.0 + 14 * 3600
    return [base_at + i * 5 * 86400 + j * 60 + i for i in range(DAYS) for j in range(2)]


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    swept = api.post(f"/roots/{made['id']}/scan").json()
    if swept["precache"] is not None:
        _settled(api, swept["precache"])
    assert _settled(api, api.post("/jobs/ingest").json()["id"]) == "done"
    assert _settled(api, api.post("/jobs/context").json()["id"]) == "done"


def _settled(api, job_id, timeout=60.0) -> str:
    deadline = time.monotonic() + timeout
    while True:
        state = api.get(f"/jobs/{job_id}").json()["state"]
        if state in ("done", "failed", "cancelled"):
            return state
        assert time.monotonic() < deadline, f"job {job_id} still {state}"
        time.sleep(0.05)


def _window(page: Page) -> tuple[float, float]:
    return (
        float(page.get_attribute("[data-surface]", "data-window-start") or "nan"),
        float(page.get_attribute("[data-surface]", "data-window-end") or "nan"),
    )


def test_the_window_opens_on_the_last_month_and_the_brush_moves_it(page: Page, live: Live):
    extent = live.api.get("/timeline/density", params={"bin": "week", "lean": "true"}).json()["extent"]
    whole_end = extent["end"] + 1
    assert extent["end"] == max(moments()), "the newest picture is the one the fixture wrote last"
    in_the_last_month = [at for at in moments() if at >= whole_end - 30 * 86400]

    page.goto("/timeline")
    page.wait_for_selector("[data-strip] .bin", timeout=10_000)
    start, end = _window(page)
    assert end == whole_end, "the window ends at the newest picture"
    assert start == min(in_the_last_month), "and opens on the earliest picture of the last month, exactly"
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
    moved_start, moved_end = _window(page)
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


def test_the_surface_moves_while_the_hand_moves_and_refreshes_itself(page: Page, live: Live):
    """Dynamic, in both senses: the stage re-renders WHILE the brush is
    dragged, not only on release; and when a job that groups pictures
    settles, the surface fetches itself again -- the group cards appear
    with no reload."""
    page.goto("/timeline")
    page.wait_for_selector("[data-strip] .bin", timeout=10_000)
    was = page.get_attribute("[data-surface]", "data-window-start")
    box = page.locator("[data-overview]").bounding_box()
    assert box is not None
    y = box["y"] + box["height"] / 2
    page.mouse.move(box["x"] + 2, y)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * 0.3, y, steps=12)
    page.wait_for_function(
        "(was) => document.querySelector('[data-surface]').dataset.windowStart !== was", arg=was, timeout=10_000
    )
    assert page.get_attribute("[data-surface]", "data-window-start") != was, "the surface moved before the release"
    page.mouse.up()
    page.wait_for_function("() => new URLSearchParams(location.search).has('start')", timeout=10_000)

    # no groups yet: the job that makes them lands on the page by itself
    page.click('[data-zoom] a[data-preset="all"]')
    page.wait_for_function(
        "() => document.querySelector('[data-zoom] a[data-preset=\"all\"][data-current]') !== null", timeout=10_000
    )
    assert page.locator("[data-sessions] .session").count() == 0
    assert _settled(live.api, live.api.post("/jobs/events").json()["id"]) == "done"
    page.wait_for_selector("[data-sessions] .session", timeout=15_000)
    assert page.url.endswith(page.evaluate("() => location.pathname + location.search")), "no reload, no redirect"
    assert "too many" not in page.inner_text("[data-surface]")
