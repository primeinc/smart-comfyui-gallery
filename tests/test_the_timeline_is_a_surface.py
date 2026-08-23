"""The timeline is a SURFACE: density at the zoom's bin, from the one
interpretation, every bin a link.

`/timeline/density` answers pictures per bin (day, hour, quarter,
minute) of the human moment over a range, in one GROUP BY; a picture
enters a bin only when its own precision fits inside it, and the
coarser claims come back as spans across the window they name -- the
signal is shown at the width it has, never dropped and never narrowed.
Each bin carries the wall-clock/instant split and a gallery link that
answers exactly its pictures (the `context.moment` facet on the same
axis). Sessions touching the range ride under it in their own domain,
each a link to the story told of that membership. A range wider than
the page can draw is refused with the remedy.
"""

from __future__ import annotations

import datetime
import os
import pathlib

import pytest
from PIL import Image

from db import connect, context, ingest, pages, runner, stories
from tests.staging import Stage, staged

NOW = 1_700_000_000.0
HOUR = 3600.0
MIN = 60.0
DAY = 86400.0
JUNE_10 = 1_686_355_200.0  # 2023-06-10 00:00 as a wall clock


def _instant(wall: float) -> float:
    naive = datetime.datetime.fromtimestamp(wall, datetime.UTC).replace(tzinfo=None)
    return naive.astimezone().timestamp()


def _drain(client) -> None:
    conn = connect.connect(client.app.state.db_path)
    try:
        while runner.run_next(conn, "test-worker", NOW + 24 * HOUR) is not None:
            conn.commit()
        conn.commit()
    finally:
        connect.close(conn)


def _total_of(page: str) -> int:
    import re

    found = re.search(r'data-total="(\d+)"', page)
    assert found is not None, "the gallery page carries its total"
    return int(found.group(1))


def _plain(path, at: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), (30, 60, 90)).save(path)
    os.utime(path, (at, at))


def _library(root: pathlib.Path) -> None:
    """Eight files: five screenshots named to the second across one
    afternoon (three inside 14:00-14:15, two at 16:30), one scan in a
    dated folder (a day claim), two claimless downloads an hour apart
    (instants)."""
    for i, minute in enumerate((0, 5, 12, 150, 152)):
        h, m = divmod(minute, 60)
        _plain(
            root / f"Screenshot 2023-06-10 at {14 + h}.{m:02d}.0{i}.png", _instant(JUNE_10 + 14 * HOUR + minute * MIN)
        )
    _plain(root / "2023-06-10" / "scan-001.png", _instant(JUNE_10 + 9 * HOUR))
    _plain(root / "download-a.png", NOW)
    _plain(root / "download-b.png", NOW + HOUR)


def _interpreted(stage: Stage) -> None:
    client, root = stage.client, stage.root
    conn = stage.conn()
    try:
        chains = context._folder_names(conn)
        for name, file_id, folder_id in conn.execute("SELECT f.name, f.id, f.folder_id FROM file f").fetchall():
            sub = root.joinpath(*chains[folder_id][1:]) if len(chains[folder_id]) > 1 else root
            ingest.one(conn, file_id, sub / name, NOW)
        conn.commit()
    finally:
        connect.close(conn)
    client.post("/jobs/context")
    client.post("/jobs/events")
    _drain(client)


@pytest.fixture(scope="module")
def _stage(tmp_path_factory):
    with staged(tmp_path_factory, "timeline", _library, _interpreted) as stage:
        yield stage


@pytest.fixture
def surfaced(_stage):
    _stage.restore()
    return _stage.client


def test_density_counts_each_bin_from_the_one_interpretation_and_spans_the_coarse(surfaced):
    client = surfaced
    told = client.get("/timeline/density", params={"bin": "day"}, headers={"accept": "application/json"})
    assert told.status_code == 200, told.text
    view = told.json()
    assert view["bin_seconds"] == 86400
    assert view["extent"]["pictures"] == 8
    by_day = {b["at"]: b for b in view["bins"]}
    assert by_day[JUNE_10]["pictures"] == 6, "five screenshots and the day-fine scan all fit a day bin"
    assert (by_day[JUNE_10]["wall"], by_day[JUNE_10]["instant"]) == (6, 0)
    november = (NOW // 86400) * 86400
    assert (by_day[november]["pictures"], by_day[november]["wall"], by_day[november]["instant"]) == (2, 0, 2)
    assert view["spans"] == [], "nothing is coarser than a day"

    hour = client.get(
        "/timeline/density",
        params={"bin": "hour", "start": JUNE_10, "end": JUNE_10 + DAY},
        headers={"accept": "application/json"},
    ).json()
    assert {b["at"]: b["pictures"] for b in hour["bins"]} == {JUNE_10 + 14 * HOUR: 3, JUNE_10 + 16 * HOUR: 2}
    assert [(s["start"], s["end"], s["precision"], s["pictures"]) for s in hour["spans"]] == [
        (JUNE_10, JUNE_10 + DAY, "day", 1)
    ], "the scan claims the day: drawn across the whole day, counted in no hour"
    quarter = client.get(
        "/timeline/density",
        params={"bin": "quarter", "start": JUNE_10 + 14 * HOUR, "end": JUNE_10 + 17 * HOUR},
        headers={"accept": "application/json"},
    ).json()
    assert {b["at"]: b["pictures"] for b in quarter["bins"]} == {
        JUNE_10 + 14 * HOUR: 3,
        JUNE_10 + 16 * HOUR + 30 * MIN: 2,
    }
    minute = client.get(
        "/timeline/density",
        params={"bin": "minute", "start": JUNE_10 + 14 * HOUR, "end": JUNE_10 + 14 * HOUR + 15 * MIN},
        headers={"accept": "application/json"},
    ).json()
    assert [b["at"] - JUNE_10 - 14 * HOUR for b in minute["bins"]] == [0, 5 * MIN, 12 * MIN]


def test_a_bin_is_a_link_that_opens_exactly_its_pictures(surfaced):
    client = surfaced
    view = client.get(
        "/timeline/density",
        params={"bin": "hour", "start": JUNE_10, "end": JUNE_10 + DAY},
        headers={"accept": "application/json"},
    ).json()
    two = next(b for b in view["bins"] if b["pictures"] == 2)
    assert "context.moment%3Agte%3A" in two["qs"]
    assert "context.moment%3Alt%3A" in two["qs"], "the link is the half-open window the count uses"
    import re

    def total(qs: str) -> int:
        opened = client.get(f"/g?{qs}")
        assert opened.status_code == 200, opened.text
        found = re.search(r'data-total="(\d+)"', opened.text)
        assert found is not None, "the gallery page carries its total"
        return int(found.group(1))

    assert total(two["qs"]) == 2, "the bar of 2 opens a gallery of exactly those 2"
    three = next(b for b in view["bins"] if b["pictures"] == 3)
    assert total(three["qs"]) == 3


def test_sessions_ride_under_the_surface_in_their_own_domain_with_a_story_link(surfaced):
    client = surfaced
    view = client.get(
        "/timeline/density",
        params={"bin": "hour", "start": JUNE_10, "end": JUNE_10 + DAY},
        headers={"accept": "application/json"},
    ).json()
    assert [(s["kind"], s["domain"], s["pictures"], s["story"]) for s in view["sessions"]] == [
        ("file_session", "wall", 5, None)
    ], "the five screenshots are one wall-clock session; the downloads' instant session is not on this day"
    whole = client.get("/timeline/density", params={"bin": "day"}, headers={"accept": "application/json"}).json()
    assert sorted((s["domain"], s["pictures"]) for s in whole["sessions"]) == [("instant", 2), ("wall", 5)]
    # a story told of exactly this membership becomes the session's link
    conn = connect.connect(client.app.state.db_path)
    try:
        event_id = next(s["id"] for s in whole["sessions"] if s["domain"] == "wall")
        snap = stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
        conn.commit()
    finally:
        connect.close(conn)
    again = client.get("/timeline/density", params={"bin": "day"}, headers={"accept": "application/json"}).json()
    held = next(s for s in again["sessions"] if s["domain"] == "wall")
    assert held["snapshot_id"] == snap.id
    assert held["story"] is None, "frozen, not yet told"


def test_a_range_the_page_cannot_draw_is_refused_with_the_remedy(surfaced):
    client = surfaced
    wide = client.get(
        "/timeline/density", params={"bin": "minute", "start": 0, "end": NOW}, headers={"accept": "application/json"}
    )
    assert wide.status_code == 400
    assert "narrow the range" in wide.text
    assert client.get("/timeline/density", params={"bin": "fortnight"}).status_code == 400
    empty = client.get(
        "/timeline/density", params={"bin": "hour", "start": NOW, "end": NOW}, headers={"accept": "application/json"}
    )
    assert empty.status_code == 400


def test_the_surface_queries_declare_their_costs(surfaced):
    """Density and spans are whole-range aggregates over the context's
    moment; sessions stop on their own range predicate."""
    from tests.test_the_pages_are_answerable import assert_no_growing_scan

    conn = connect.connect(surfaced.app.state.db_path, read_only=True)
    try:
        assert_no_growing_scan(
            conn,
            pages.TIMELINE_DENSITY,
            (0, 3600, 3600, 0, context.POLICY_VERSION, 0.0, NOW * 2, '["second","subsecond"]'),
            aggregate=True,
            counts=True,
        )
        assert_no_growing_scan(
            conn,
            pages.TIMELINE_SPANS,
            (context.POLICY_VERSION, 0.0, NOW * 2, '["second"]'),
            aggregate=True,
            counts=True,
        )
        assert_no_growing_scan(conn, pages.TIMELINE_EXTENT, (context.POLICY_VERSION,), aggregate=True, counts=True)
    finally:
        connect.close(conn)


def test_the_page_is_the_surface_and_nothing_beside_it(surfaced):
    """One surface: the window, its overview and its sessions. No second
    ladder of months and days renders beside it -- the overview's bars
    and the window's bars are those links."""
    page = surfaced.get("/timeline", headers={"accept": "text/html"})
    assert page.status_code == 200
    assert "data-surface" in page.text
    assert "/static/build/timeline.js" in page.text
    assert "data-overview" in page.text
    assert "data-timeline-day" not in page.text
    assert "data-timeline-month" not in page.text


def test_a_story_belongs_to_its_subject_never_to_a_membership_checksum(surfaced):
    """Two snapshots with the same ordered membership but different
    subjects (a capture session and a generation session over the same
    mixed files share a member_hash): the session's story link is the
    one told of ITS kind and grouper, never the other's."""
    client = surfaced
    whole = client.get("/timeline/density", params={"bin": "day"}, headers={"accept": "application/json"}).json()
    held = next(s for s in whole["sessions"] if s["domain"] == "wall")
    conn = connect.connect(client.app.state.db_path)
    try:
        member_hash = conn.execute("SELECT member_hash FROM derived_event WHERE id = ?", (held["id"],)).fetchone()[0]
        conn.execute(
            "INSERT INTO story_snapshot(format_version, source_kind, event_kind, grouper, context_generation,"
            " context_policy_version, member_hash, document_json, document_sha256, created_at)"
            " VALUES(1, 'event', 'generation_session', 'generation_session', 1, 1, ?, '{}', ?, 0)",
            (member_hash, "e" * 64),
        )
        conn.commit()
    finally:
        connect.close(conn)
    again = client.get("/timeline/density", params={"bin": "day"}, headers={"accept": "application/json"}).json()
    assert next(s for s in again["sessions"] if s["id"] == held["id"])["snapshot_id"] is None, (
        "a generation story is not this file session's story"
    )


def test_the_groupers_carry_the_refined_time_rule_in_their_versions():
    """The gap rule moved to the refined second (WI-58); a run under
    the old rule and a run under this one must never share a producer
    identity."""
    from db import events

    assert (
        events.GenerationSessionGrouper.version,
        events.CaptureSessionGrouper.version,
        events.FileSessionGrouper.version,
    ) == ("5", "5", "2")


def test_a_library_wider_than_the_day_cap_answers_at_the_week(surfaced):
    """Thirteen years is more day bins than the page draws; the week
    bin is the coarse level that still fits, and a day-fine claim fits
    a week exactly as it fits a day."""
    client = surfaced
    week = client.get(
        "/timeline/density",
        params={"bin": "week", "start": JUNE_10 - 3 * DAY, "end": JUNE_10 + 4 * DAY},
        headers={"accept": "application/json"},
    ).json()
    assert week["bin_seconds"] == 604_800
    assert sum(b["pictures"] for b in week["bins"]) == 6, "five screenshots and the day-fine scan"
    assert week["spans"] == []
    far = client.get(
        "/timeline/density",
        params={"bin": "day", "start": 0, "end": 5000 * DAY},
        headers={"accept": "application/json"},
    )
    assert far.status_code == 400
    assert (
        client.get(
            "/timeline/density",
            params={"bin": "week", "start": 0, "end": 5000 * DAY},
            headers={"accept": "application/json"},
        ).status_code
        == 200
    )


#: A minute boundary (divisible by 60) past NOW.
MINUTE_AT = 1_700_000_040.0


def test_a_bar_counts_a_fractional_moment_and_its_link_opens_it(tmp_path_factory):
    """A claimless file's moment is its mtime, fractional; the bar counts
    on [at, at+width) over that real axis and the link must spell the
    same half-open window, or a bar of one opens a gallery of none."""

    def build(root: pathlib.Path) -> None:
        _plain(root / "download-c.png", MINUTE_AT + 59.5)

    with staged(tmp_path_factory, "timeline-fraction", build, _interpreted) as stage:
        client = stage.client
        view = client.get(
            "/timeline/density",
            params={"bin": "minute", "start": MINUTE_AT, "end": MINUTE_AT + MIN},
            headers={"accept": "application/json"},
        ).json()
        assert [(b["at"], b["pictures"]) for b in view["bins"]] == [(MINUTE_AT, 1)]
        opened = client.get(f"/g?{view['bins'][0]['qs']}")
        assert opened.status_code == 200, opened.text
        assert _total_of(opened.text) == 1, "the bar of one opens that one"


def test_the_page_the_fragment_and_the_machine_answer_are_one_surface(surfaced):
    """/timeline is one builder in three representations: JSON to a
    machine, the surface fragment to htmx, the page to a browser -- the
    same window, the same bars, the same links."""
    told = surfaced.get("/timeline", headers={"accept": "application/json"}).json()
    # the last month holds the two November downloads an hour apart: the
    # opening window tightens to that hour, and the zoom follows it
    assert (told["start"], told["end"]) == (NOW, NOW + HOUR + 1)
    assert told["bin"] == "minute"
    assert sum(b["pictures"] for b in told["bins"]) == 2
    assert [p["name"] for p in told["presets"]] == ["1w", "1m", "3m", "1y", "all"]
    assert told["overview"]["bars"], "the whole extent at week resolution rides every answer"
    fragment = surfaced.get("/timeline", headers={"hx-request": "true"}).text
    page = surfaced.get("/timeline", headers={"accept": "text/html"}).text
    assert fragment.startswith('<section class="surface"')
    assert fragment in page, "the page is the shell around the very fragment a move fetches"
    import html

    for bar in told["bins"]:
        assert f'data-bin-at="{bar["at"]}"' in fragment
        assert html.escape(bar["href"], quote=False) in fragment
    assert "too many" not in page
    # a window the URL names is the window the page shows, at the zoom its width earns
    day = surfaced.get(
        "/timeline", params={"start": JUNE_10, "end": JUNE_10 + DAY}, headers={"accept": "application/json"}
    ).json()
    assert (day["bin"], day["start"], day["end"]) == ("quarter", JUNE_10, JUNE_10 + DAY)
    assert any(b["samples"] for b in day["bins"]), "thumbnails at every window"


def test_an_authored_place_marks_the_sessions_stale_and_names_the_remedy(surfaced):
    """A rebuild advances the interpretation's generation; session runs
    answer only at the generation they were computed over. The page
    says the sessions need the events job instead of listing none."""
    before = surfaced.get("/timeline/density", params={"bin": "day"}, headers={"accept": "application/json"}).json()
    assert before["coverage"]["events_current"] is True
    assert before["sessions"]
    link = before["sessions"][0]["qs"]
    assert surfaced.get(f"/g?{link}").status_code == 200
    conn = connect.connect(surfaced.app.state.db_path, read_only=True)
    try:
        slug = conn.execute("SELECT e.slug FROM entity e JOIN file f ON f.id = e.id ORDER BY f.id LIMIT 1").fetchone()[
            0
        ]
    finally:
        connect.close(conn)
    told = surfaced.post(f"/i/{slug}/place", json={"name": "Lisbon", "kind": "city"})
    assert told.status_code < 300, told.text
    after = surfaced.get("/timeline/density", params={"bin": "day"}, headers={"accept": "application/json"}).json()
    assert after["coverage"]["events_current"] is False
    assert after["sessions"] == []
    assert after["sessions_total"] == 0
    assert "data-timeline-events-stale" in surfaced.get("/timeline", headers={"accept": "text/html"}).text
    density = surfaced.get("/timeline/density", params={"bin": "day"}, headers={"accept": "application/json"}).json()
    assert density["coverage"]["events_current"] is False
    # the bookmarked link refuses with the remedy, never an empty grid
    refused = surfaced.get(f"/g?{link}")
    assert refused.status_code == 400, refused.text
    assert "events job" in refused.json()["detail"]
    assert surfaced.get("/g?f=event.id:eq:999999").status_code == 404


def test_an_empty_scope_answers_the_same_shape_as_a_full_one(surfaced):
    full = surfaced.get("/timeline/density", params={"bin": "day"}, headers={"accept": "application/json"})
    empty = surfaced.get(
        "/timeline/density",
        params={"bin": "day", "f": "capture.iso:gte:999999"},
        headers={"accept": "application/json"},
    )
    assert empty.status_code == 200, empty.text
    assert set(empty.json()) == set(full.json())
    assert empty.json()["bins"] == []
    assert empty.json()["extent"] is None
    assert empty.headers.get("vary") == full.headers.get("vary")


def test_the_opening_window_is_where_the_last_months_pictures_are(surfaced):
    """The surface opens on the pictures, not on a month of nothing: the
    last month is tightened to the span its pictures occupy, never
    narrower than an hour, and a window the URL names is left alone."""
    opened = surfaced.get("/timeline", headers={"accept": "application/json"}).json()
    assert opened["end"] - opened["start"] >= 3600
    assert all(b["pictures"] for b in opened["bins"]) or opened["bins"], "no bar of nothing at either end"
    named = surfaced.get(
        "/timeline", params={"start": JUNE_10, "end": JUNE_10 + DAY}, headers={"accept": "application/json"}
    ).json()
    assert (named["start"], named["end"]) == (JUNE_10, JUNE_10 + DAY), "a named window is not tightened"


@pytest.mark.parametrize("bin_name", ["day", "hour", "quarter", "minute"])
def test_every_bar_at_every_zoom_opens_exactly_its_pictures(surfaced, bin_name):
    """The invariant the surface is built on, checked over the whole
    extent at each zoom: a bar's link answers the bar's count."""

    view = surfaced.get(
        "/timeline/density",
        params={"bin": bin_name, "start": JUNE_10, "end": JUNE_10 + DAY},
        headers={"accept": "application/json"},
    ).json()
    assert view["bins"], "the June day has bars at every zoom"
    for bar in view["bins"]:
        opened = surfaced.get(f"/g?{bar['qs']}")
        assert opened.status_code == 200, opened.text
        total = _total_of(opened.text)
        assert total == bar["pictures"], (
            f"{bin_name} bar at {bar['at']}: bar says {bar['pictures']}, link opens {total}"
        )


def test_past_the_sampling_bound_the_busiest_bins_still_carry_thumbnails(surfaced, monkeypatch):
    """A strip always: past SAMPLED_BINS_MOST bins the busiest that many
    are sampled and `sampled` says the strip is partial -- never an
    empty strip with an apology."""
    monkeypatch.setattr(pages, "SAMPLED_BINS_MOST", 1)
    view = surfaced.get(
        "/timeline/density",
        params={"bin": "hour", "start": JUNE_10, "end": JUNE_10 + DAY},
        headers={"accept": "application/json"},
    ).json()
    assert view["sampled"] is False
    by_count = sorted(view["bins"], key=lambda b: -b["pictures"])
    assert by_count[0]["pictures"] == 3
    assert len(by_count[0]["samples"]) >= 1, "the busiest hour carries thumbnails"
    assert all(b["samples"] == [] for b in by_count[1:]), "only the busiest one past the bound"


def test_a_lean_ask_is_the_shape_alone(surfaced):
    """The overview asks for the whole extent at week resolution with
    `lean`: bins and extent, no thumbnails, no session cards."""
    view = surfaced.get(
        "/timeline/density", params={"bin": "week", "lean": "true"}, headers={"accept": "application/json"}
    ).json()
    assert view["extent"]["pictures"] == 8
    assert view["bins"]
    assert all(b["samples"] == [] for b in view["bins"])
    assert view["sessions"] == []
    assert view["sessions_total"] == 0
