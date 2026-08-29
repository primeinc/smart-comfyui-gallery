"""The timeline page, witnessed in a browser: the URL owns the window,
a month on the scrubber opens its window and back returns, the page
carries the window's pictures, a session's words open its pictures;
opening an untold session tells its story, and the story then rides
the session as its title."""

from __future__ import annotations

import json
import os
import time

import pytest
from PIL import Image
from playwright.sync_api import Page, expect

from tests.conftest import POLL, Live
from tests.staging import JUNE_10 as _JUNE_10

pytestmark = pytest.mark.slow

FILES = 6
# int on purpose: the URL below spells whole seconds, and the surface
# echoes them back as the float the assertions append '.0' to.
JUNE_10 = int(_JUNE_10)


def write_library(root) -> None:
    base_at = JUNE_10 + 14 * 3600  # 14:00, stamped names every five minutes
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
        time.sleep(POLL)


def test_the_url_owns_the_window_and_the_surface_carries_pictures(page: Page, live: Live):
    # a day-wide window; the scrubber's month opens the month
    page.goto(f"/timeline?start={JUNE_10}&end={JUNE_10 + 86400}")
    page.wait_for_selector("[data-strip] .bin", timeout=10_000)
    assert page.get_attribute("[data-surface]", "data-window-start") == f"{JUNE_10}.0"
    page.wait_for_selector("[data-samples] .surface-sample img", timeout=10_000)
    page.wait_for_selector("[data-sessions] .session [data-session-open]", timeout=10_000)
    # The strip's pictures arrive after the session it belongs to, so a
    # `count()` taken the moment the session appears reads zero -- measured,
    # `assert 0 >= 1`, on a full-suite run. `not_to_have_count(0)` retries,
    # which is the same claim made of the page rather than of an instant.
    expect(page.locator("[data-sessions] .session-strip img")).not_to_have_count(0)
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
        "(d) => new URLSearchParams(location.search).get('start') === d", arg=str(JUNE_10), timeout=10_000
    )
    page.wait_for_function(
        "(d) => document.querySelector('[data-surface]').dataset.windowStart === d", arg=f"{JUNE_10}.0", timeout=10_000
    )
    page.reload()
    page.wait_for_selector("[data-strip] .bin", timeout=10_000)
    assert page.get_attribute("[data-surface]", "data-window-start") == f"{JUNE_10}.0"
    href = page.get_attribute("[data-sessions] .session [data-session-open]", "data-session-pictures")
    assert href is not None
    assert "event.id%3Aeq%3A" in href
    page.goto(href)
    page.wait_for_selector("[data-chips] [data-chip]", timeout=10_000)
    total = page.get_attribute("[data-grid]", "data-total")
    assert total is not None
    assert int(total) == FILES


def test_opening_a_session_tells_its_story_and_the_story_rides_the_session(page: Page, live: Live):
    """Opening an untold session IS telling it: the session's words are
    the link; the route freezes, asks for the plan (a job the real worker
    drains -- the page comes back to the timeline and refreshes itself
    when it settles), renders, and shows the story; back on the
    timeline the story is the session's title, and the evolution view
    is a link away."""
    page.goto("/timeline")
    page.wait_for_selector("[data-sessions] .session [data-session-open]", timeout=10_000)
    assert page.locator("[data-sessions] .session [data-session-story-title]").count() == 0, "untold"
    page.click("[data-sessions] .session [data-session-open]")
    page.wait_for_url(lambda url: "/stories/renders/" in url or "/timeline" in url, timeout=60_000)
    if "/timeline" in page.url:
        # the plan was durable work: the real worker drains it, the page
        # refreshes itself, and the same link then opens the story
        page.wait_for_selector("[data-sessions] .session [data-session-open]", timeout=10_000)
        deadline = time.monotonic() + 60
        while any(job["kind"] == "story_plan" for job in live.api.get("/jobs").json()):
            assert time.monotonic() < deadline, "the plan job never settled"
            time.sleep(0.1)
        page.click("[data-sessions] .session [data-session-open]")
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
    # The crumb above is server-rendered and proves only that the shell
    # arrived. The members are not in the HTML at all: they exist once the
    # explorer has asked the same route for its document and drawn it, so
    # this is the clause that holds the fetch.
    page.wait_for_selector("[data-main] [data-ref]", timeout=10_000)
    page.goto("/timeline")
    page.wait_for_selector("[data-sessions] .session[data-session-story]", timeout=10_000)
    assert page.inner_text("[data-sessions] .session [data-session-story-title]").strip()
    href = page.get_attribute("[data-sessions] .session [data-session-open]", "href")
    assert href is not None
    assert "/stories/renders/" in href, "a told session's link is its story"


def test_the_save_view_button_keeps_every_facet(page: Page, live: Live):
    """A two-facet link saved from the gallery is a two-facet rule: the
    button sends every `f` the mounted answer carries, and the saved
    collection's words name both."""
    asked = "/g?f=context.local_day%3Aeq%3A2023-06-10&f=context.origin%3Aeq%3Aimported"
    page.goto(asked)
    page.wait_for_selector("[data-grid]", timeout=10_000)
    assert int(page.get_attribute("[data-grid]", "data-total") or 0) == FILES
    page.click("[data-save-smart]")
    # The application's own dialog, not the browser's: an autofocused
    # field and Enter (tests/test_the_application_asks_in_its_own_words.py).
    page.wait_for_selector("dialog.ask-box[open]", timeout=10_000)
    page.keyboard.type("Two facets")
    page.keyboard.press("Enter")
    page.wait_for_url("**/t/*", timeout=20_000)
    slug = page.url.rsplit("/t/", 1)[1].split("?", 1)[0]
    told = live.api.get(f"/t/{slug}", headers={"accept": "application/json"}).json()
    assert told["rule"] is not None
    assert "context.local_day" in told["rule"]["nl"]
    assert "context.origin" in told["rule"]["nl"], "the second facet was dropped on the way to the rule"
    inside = live.api.get(f"/g?album={slug}", headers={"accept": "text/html"}).text
    assert f'data-total="{FILES}"' in inside


def test_a_page_of_dates_stays_alive_under_people_js(page: Page, live: Live):
    """The People bundle spells every <time data-epoch> and watches the
    document for new ones. Spelling is itself a mutation: a watcher that
    re-spelt what it had just spelt looped the main thread forever, and
    every person page -- the pages with dated sessions -- hung the tab.
    The page must answer after load, with every date spelled once. The
    script loaded here is the one every page loads."""
    page.goto("/people")
    page.set_content(
        '<body><time data-epoch="1686355200">1686355200</time><time data-epoch="1686441600">x</time>'
        f'<script src="{live.url}/static/build/app.js"></script></body>'
    )
    # Waited for the spelling, not for a guess at how long spelling takes.
    # The bug this catches -- a watcher that re-spells what it just spelled
    # -- pins the main thread, so it still fails here: a pinned thread
    # never satisfies a retrying read either. It just costs milliseconds
    # when the page is well instead of the 700ms two fixed pauses cost.
    expect(page.locator("time")).to_have_text(["2023-06-10", "2023-06-11"])
    page.evaluate(
        "const t = document.createElement('time'); t.dataset.epoch = '1686528000'; document.body.appendChild(t)"
    )
    expect(page.locator("time").last).to_have_text("2023-06-12")


def test_a_missed_terminal_delta_is_recovered_by_the_next_snapshot(page: Page, live: Live):
    """The job feed's snapshot is a resynchronisation boundary.

    A delta says what just happened; a SNAPSHOT says only which jobs are
    still unsettled. So a job that settled while the page was not
    listening -- a dropped socket, a lost frame -- appears in neither: not
    in the delta the page never got, and not in the snapshot, because by
    then it is finished. A page that ignores snapshots and waits for a
    terminal delta that already happened waits forever.

    Here the first socket is made deaf and dropped at the exact moment the
    relevant job settles, so the page is never told. The second socket's
    snapshot is the only thing that arrives -- and it must be enough to
    make the page read the rows again.
    """
    api = live.api
    sockets: dict = {"n": 0, "lost": None}

    def route(ws) -> None:
        sockets["n"] += 1
        mine = sockets["n"]
        server = ws.connect_to_server()

        def from_server(message) -> None:
            if mine > 1:
                ws.send(message)
                return
            # deaf: the page is told nothing on this connection, and the
            # connection is dropped the moment the job it cares about
            # settles -- so that terminal delta is lost for good
            held = json.loads(message) if isinstance(message, str) else {}
            settled = held.get("type") == "delta" and held.get("state") in ("done", "failed", "cancelled")
            if settled and held.get("kind") == "events":
                sockets["lost"] = held["job"]
                ws.close()

        server.on_message(from_server)

    page.route_web_socket("**/ws/jobs", route)
    # The page's own clock, so the reconnect is OBSERVED rather than waited
    # out: `feed.onclose` re-opens after RECONNECT_MS (timeline.ts). What is
    # under test is the second connection's SNAPSHOT, not the gap before it.
    page.clock.install()
    page.goto(f"/timeline?start={JUNE_10}&end={JUNE_10 + 86400}")
    page.wait_for_selector("[data-strip] .bin", timeout=10_000)
    # mark the surface on screen; a re-read replaces the node with a fresh
    # one, and nothing else on this page removes the mark
    page.evaluate("() => { document.querySelector('[data-surface]').dataset.stale = 'yes'; }")

    job_id = api.post("/jobs/events").json()["id"]
    assert _settled(api, job_id) == "done"

    # The socket must be SHUT before the clock moves: `run_for` fires only the
    # timers armed when it runs, and `onclose` arms the reconnect. Waited
    # through Playwright, whose dispatcher only turns inside a Playwright call.
    ended = time.monotonic() + 15
    while sockets["lost"] is None:
        assert time.monotonic() < ended, "the terminal delta never arrived to be withheld"
        page.wait_for_timeout(20)
    # Past the backoff, in one step. The socket then opens for real and
    # its snapshot arrives over the network, which no clock touches.
    page.clock.run_for(2100)  # milliseconds; `run_for` takes a number or "mm:ss"
    # `expect` and not `wait_for_function`: the installed clock stubs
    # `requestAnimationFrame`, which is what an in-page poll runs on, so
    # a `wait_for_function` here would be waiting on a clock this test
    # has stopped. `expect` retries from the driver instead.
    expect(page.locator("[data-surface][data-stale]")).to_have_count(0, timeout=30_000)
    assert sockets["lost"] == job_id, "the terminal delta was never withheld; the test proved nothing"
    assert sockets["n"] >= 2, "the page never reconnected"
    page.wait_for_selector("[data-strip] .bin", timeout=10_000)
