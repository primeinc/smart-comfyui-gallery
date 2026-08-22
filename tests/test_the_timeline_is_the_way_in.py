"""The timeline is a primary way into the library, not a probe.

Every shelf and session is a door spelled by the Facet Interface; the
doors open galleries ordered by the human moment; the gallery says
which question it is answering and lets one chip go; a picture's page
says when it happened, on what evidence, and which sessions it is in;
the surface carries pictures, its weeks start on Monday, and it says how
much of the library it can see.
"""

from __future__ import annotations

import datetime
import re

import pytest

from db import facets, pages
from tests.staging import staged
from tests.test_the_timeline_is_a_surface import DAY, JUNE_10, _interpreted, _library


@pytest.fixture(scope="module")
def _stage(tmp_path_factory):
    with staged(tmp_path_factory, "timeline-doors", _library, _interpreted) as stage:
        yield stage


@pytest.fixture
def doors(_stage):
    _stage.restore()
    return _stage.client


def _total(client, qs: str) -> int:
    opened = client.get(f"/g?{qs}")
    assert opened.status_code == 200, opened.text
    found = re.search(r'data-total="(\d+)"', opened.text)
    assert found is not None
    return int(found.group(1))


def _density(client, **params):
    told = client.get("/timeline/density", params=params, headers={"accept": "application/json"})
    assert told.status_code == 200, told.text
    return told.json()


def test_a_session_is_a_door_that_opens_exactly_its_members(doors):
    whole = _density(doors, bin="day")
    for s in whole["sessions"]:
        assert "event.id%3Aeq%3A" in s["qs"]
        assert s["qs"].endswith("sort=moment")
        assert _total(doors, s["qs"]) == s["pictures"], s
    assert facets.facet("event.id", "eq", "1").key == "event.id"
    with pytest.raises(ValueError, match="allows eq"):
        facets.facet("event.id", "gte", "1")


def test_a_stale_event_opens_nothing(doors):
    """The facet answers only for runs proven over the current
    interpretation: after the contexts move on, yesterday's session id
    is a door onto an empty room, never onto whatever the id now means."""
    whole = _density(doors, bin="day")
    held = next(s for s in whole["sessions"] if s["domain"] == "wall")
    assert _total(doors, held["qs"]) == held["pictures"]
    doors.post("/jobs/context")  # re-interpreting advances the generation; runs are stale until regrouped
    from tests.test_the_timeline_is_a_surface import _drain

    _drain(doors)
    assert _total(doors, held["qs"]) == 0


def test_a_door_orders_by_the_moment_it_opened_on(doors):
    qs = "f=context.local_day%3Aeq%3A2023-06-10&sort=moment"
    page = doors.get(f"/g?{qs}").text
    names = re.findall(r'alt="([^"]+)"', page)
    assert names[0] == "scan-001.png", "09:00 (the folder's day claim) comes before the afternoon's screenshots"
    assert names[1:4] == [
        "Screenshot 2023-06-10 at 14.00.00.png",
        "Screenshot 2023-06-10 at 14.05.01.png",
        "Screenshot 2023-06-10 at 14.12.02.png",
    ]
    newest = re.findall(r'alt="([^"]+)"', doors.get(f"/g?{qs.replace('moment', 'moment-newest')}").text)
    assert newest == list(reversed(names))
    assert doors.get("/g?sort=sideways").status_code == 400


def test_the_gallery_shows_its_facets_as_chips_that_can_go(doors):
    page = doors.get("/g?f=context.local_day%3Aeq%3A2023-06-10&f=context.origin%3Aeq%3Aimported&sort=moment").text
    assert 'data-chip="context.local_day:eq:2023-06-10"' in page
    assert "day 2023-06-10" in page
    assert "origin imported" in page
    removes = re.findall(r'data-chip="([^"]+)">[^<]*<a href="([^"]+)"', page)
    assert removes == [
        ("context.local_day:eq:2023-06-10", "/g?f=context.origin%3Aeq%3Aimported&amp;sort=moment"),
        ("context.origin:eq:imported", "/g?f=context.local_day%3Aeq%3A2023-06-10&amp;sort=moment"),
    ]
    assert "data-chips" not in doors.get("/g").text


def test_months_and_events_are_doors_too(doors):
    body = doors.get("/timeline", headers={"accept": "application/json"}).json()
    for month in body["months"]:
        assert "context.local_day%3Agte%3A" in month["qs"]
        assert "context.local_day%3Alte%3A" in month["qs"]
        assert _total(doors, month["qs"]) == month["pictures"], month
    for event in body["events"]:
        assert _total(doors, event["qs"]) == event["pictures"]
        assert event["domain"] in ("wall", "instant")
        assert event["start"] is not None
    page = doors.get("/timeline", headers={"accept": "text/html"}).text
    for marker in ("data-timeline-month=", "data-timeline-event=", "data-samples"):
        assert marker in page, marker
    coverage = body["coverage"]
    assert (coverage["interpreted"], coverage["present"], coverage["complete"]) == (8, 8, True)
    assert coverage["policy_version"] == pages.context.POLICY_VERSION
    # the downloads' btime disputes nothing on a copy but may on a fresh
    # write: the count is the filesystem's, so only its shape is pinned
    assert isinstance(coverage["contested"], int)
    assert "8 of 8 files interpreted" in page


def test_the_surface_carries_pictures_origins_and_its_coverage(doors):
    view = _density(doors, bin="hour", start=JUNE_10, end=JUNE_10 + DAY)
    assert view["sampled"] is True
    for b in view["bins"]:
        assert 1 <= len(b["samples"]) <= pages.SAMPLES_PER_BIN
        assert sum(b["origin"].values()) == b["pictures"]
        assert b["origin"]["imported"] == b["pictures"], "screenshots carry neither camera nor generator"
    assert view["coverage"]["complete"] is True
    for s in view["sessions"]:
        assert 1 <= len(s["samples"]) <= pages.SAMPLES_PER_SESSION
        assert s["tellable"] is True
        assert s["planner"] == "file_history"


def test_a_week_starts_on_a_monday(doors):
    week = _density(doors, bin="week", start=JUNE_10 - 3 * DAY, end=JUNE_10 + 4 * DAY)
    assert week["bins"], "the June pictures fall in a week"
    for b in week["bins"]:
        assert (b["at"] - pages.MONDAY) % 604_800 == 0
        assert datetime.datetime.fromtimestamp(b["at"], datetime.UTC).weekday() == 0
    assert week["bin_seconds"] == 604_800


def test_a_picture_says_when_and_in_which_sessions(doors):
    page = doors.get("/g?f=context.local_day%3Aeq%3A2023-06-10&sort=moment").text
    slugs = re.findall(r'data-slug="([^"]+)"', page)
    item = doors.get(f"/i/{slugs[1]}", headers={"accept": "application/json"}).json()
    when = item["when"]
    assert (when["domain"], when["basis"], when["precision"]) == ("wall", "filename", "second")
    assert when["local_day"] == "2023-06-10"
    assert when["day_qs"].startswith("f=context.local_day%3Aeq%3A2023-06-10")
    assert when["timeline"].startswith("/timeline?bin=hour&start=")
    assert [s["kind"] for s in when["sessions"]] == ["file_session"]
    assert _total(doors, when["sessions"][0]["qs"]) == 5
    html = doors.get(f"/i/{slugs[1]}", headers={"accept": "text/html"}).text
    for marker in ('data-when data-domain="wall"', "data-when-sessions", "data-when-day"):
        assert marker in html, marker
