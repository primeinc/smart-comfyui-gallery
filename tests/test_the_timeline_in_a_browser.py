"""The timeline page, witnessed in a browser: the URL owns the window,
a bar opens its hour and back returns, the strip carries thumbnails, a
group card opens its pictures, tells its story, and carries it."""

from __future__ import annotations

import os
import time

import pytest
from PIL import Image
from playwright.sync_api import Page

from tests.conftest import Live

pytestmark = pytest.mark.slow

FILES = 6
DAY = 1_686_355_200  # 2023-06-10


def write_library(root) -> None:
    base_at = DAY + 14 * 3600  # 14:00, stamped names every five minutes
    for i in range(FILES):
        path = root / f"Screenshot 2023-06-10 at 14.{i * 5:02d}.0{i}.png"
        Image.new("RGB", (8, 8), (30 * i, 70, 130)).save(path)
        os.utime(path, (base_at + i * 300, base_at + i * 300))


def prepare(api, root) -> None:
    made = api.post("/roots", json={"path": str(root)}).json()
    swept = api.post(f"/roots/{made['id']}/scan").json()
    if swept["precache"] is not None:
        _settled(api, swept["precache"])
    assert _settled(api, api.post("/jobs/ingest").json()["id"]) == "done"
    assert _settled(api, api.post("/jobs/context").json()["id"]) == "done"
    assert _settled(api, api.post("/jobs/events").json()["id"]) == "done"


def _settled(api, job_id, timeout=60.0) -> str:
    deadline = time.monotonic() + timeout
    while True:
        state = api.get(f"/jobs/{job_id}").json()["state"]
        if state in ("done", "failed", "cancelled"):
            return state
        assert time.monotonic() < deadline, f"job {job_id} still {state}"
        time.sleep(0.05)


def test_the_url_owns_the_window_and_the_surface_carries_pictures(page: Page, live: Live):
    # a day-wide window: one bar per hour; a bar opens its hour
    page.goto(f"/timeline?start={DAY}&end={DAY + 86400}")
    page.wait_for_selector("[data-strip] .bin", timeout=10_000)
    assert page.get_attribute("[data-surface]", "data-window-start") == f"{DAY}.0"
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
        "(d) => new URLSearchParams(location.search).get('start') === d", arg=str(DAY), timeout=10_000
    )
    page.wait_for_function(
        "(d) => document.querySelector('[data-surface]').dataset.windowStart === d", arg=f"{DAY}.0", timeout=10_000
    )
    page.reload()
    page.wait_for_selector("[data-strip] .bin", timeout=10_000)
    assert page.get_attribute("[data-surface]", "data-window-start") == f"{DAY}.0"
    href = page.get_attribute("[data-sessions] .session [data-session-open]", "href")
    assert href is not None
    assert "event.id%3Aeq%3A" in href
    page.goto(href)
    page.wait_for_selector("[data-chips] [data-chip]", timeout=10_000)
    total = page.get_attribute("[data-grid]", "data-total")
    assert total is not None
    assert int(total) == FILES


def test_a_group_tells_its_story_and_carries_it(page: Page, live: Live):
    """Tell the story from the card: freeze, plan (a job the real worker
    drains), render, read; back on the timeline the story rides the card
    -- title, heroes, the door to the whole -- and the evolution view is
    a door away."""
    page.goto("/timeline")
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
    page.goto("/timeline")
    page.wait_for_selector("[data-sessions] .session [data-session-story]", timeout=10_000)
    assert page.inner_text("[data-sessions] .session [data-session-story-title]").strip()
    assert page.locator("[data-sessions] .session .session-story-hero").count() >= 1
    assert page.locator("[data-sessions] .session [data-session-story-read]").count() == 1


def test_the_save_view_button_keeps_every_facet(page: Page, live: Live):
    """A two-facet door saved from the gallery is a two-facet rule: the
    button sends every `f` the mounted answer carries, and the saved
    collection's words name both."""
    asked = "/g?f=context.local_day%3Aeq%3A2023-06-10&f=context.origin%3Aeq%3Aimported"
    page.on("dialog", lambda dialog: dialog.accept("Two facets"))
    page.goto(asked)
    page.wait_for_selector("[data-grid]", timeout=10_000)
    assert int(page.get_attribute("[data-grid]", "data-total") or 0) == FILES
    page.click("[data-save-smart]")
    page.wait_for_url("**/t/*", timeout=20_000)
    slug = page.url.rsplit("/t/", 1)[1].split("?", 1)[0]
    told = live.api.get(f"/t/{slug}", headers={"accept": "application/json"}).json()
    assert told["rule"] is not None
    assert "context.local_day" in told["rule"]["nl"]
    assert "context.origin" in told["rule"]["nl"], "the second facet was dropped on the way to the rule"
    inside = live.api.get(f"/g?album={slug}", headers={"accept": "text/html"}).text
    assert f'data-total="{FILES}"' in inside
