"""The timeline stops paying rent on nothing.

A surface that spends pixels in proportion to ELAPSED time spends them
on nothing at all. A picture in July and a week's work in August draws
one bar and three weeks of blank. A scanned photograph from 2002 draws
twenty-four years of blank to reach this month. It is one defect at two
zooms, and the fix has to be one rule or it will be argued about again
at the next zoom.

The blank is not even honest. Empty pixels are ambiguous between "no
pictures here", "nothing dated this" and "the render broke" -- a reader
cannot tell which. A band saying `3 weeks · nothing` says exactly which,
and costs a fortieth of the space.

Three things this must not break, and they are what most of these tests
are about:

**A dense window must not move.** If nothing is worth collapsing the
axis is the arithmetic it replaced, to the pixel. Otherwise every
existing surface shifts the day this module lands.

**A click must still mean a time.** The browser turns clicks, drags and
pans back into moments, so the map has to invert. A one-way projection
would put every gesture in the wrong year the moment a gap appeared.

**Kept time stays linear.** A gap between two days of one week is
information at that zoom. Only runs that were paying no rent go.
"""

from __future__ import annotations

import datetime
import os
import time

import pytest

from db import when
from sg_web import projecting
from tests.staging import staged

pytestmark = pytest.mark.slow

#: `db/when.py`'s units, not a fourth spelling of them. This said
#: `YEAR = 365 * DAY` -- the common year, 31_536_000 -- while the
#: surface under test drew against 31_556_952 and
#: `tests/test_the_timeline_draws_any_span.py` built its fixtures
#: against 31_557_600.
DAY = when.DAY
YEAR = when.YEAR


def test_a_window_with_no_gap_is_the_arithmetic_it_replaced():
    """The regression guard for every surface that already draws. With
    nothing worth collapsing this must be indistinguishable from
    `((t - lo) / span) * W`, or the day this lands every bar moves."""
    lo, hi = 0.0, 10 * DAY
    axis = projecting.projected(lo, hi, [(0.0, 10 * DAY)])
    assert axis.collapsed == ()
    for t in (0.0, DAY, 2.5 * DAY, 9.99 * DAY, hi):
        assert axis.x(t) == pytest.approx(((t - lo) / (hi - lo)) * projecting.WIDTH, abs=0.01)


def test_a_run_of_nothing_stops_taking_the_axis():
    """The screenshot: one day in July, a week in August, and three
    weeks of blank between them taking most of the page."""
    lo, hi = 0.0, 30 * DAY
    axis = projecting.projected(lo, hi, [(0.0, DAY), (27 * DAY, 30 * DAY)])
    assert len(axis.collapsed) == 1
    gap = axis.collapsed[0]
    assert (gap.t0, gap.t1) == (DAY, 27 * DAY)
    # 26 of 30 days was 87% of the axis; it is now a band
    assert (gap.x1 - gap.x0) == pytest.approx(projecting.COLLAPSED, abs=0.01)
    # and the pictures got what it was holding
    assert axis.x(DAY) > projecting.WIDTH * 0.2, "the July day is still a sliver"


def test_the_same_rule_holds_at_twenty_four_years():
    """The scrubber: one scanned photograph in 2002 and everything else
    this year. The rule is a SHARE of the axis, so it needs no second
    threshold for a quarter-century -- which is the whole reason the
    complaint was that this is pervasive."""
    lo, hi = 0.0, 24 * YEAR
    axis = projecting.projected(lo, hi, [(0.0, DAY), (24 * YEAR - 30 * DAY, 24 * YEAR)])
    assert len(axis.collapsed) == 1
    assert axis.collapsed[0].seconds > 23 * YEAR

    # Against the linear axis, not against a number somebody picked. The
    # 2002 photograph was 0.011% of the width -- a hairline nothing can
    # be aimed at. It is a day beside thirty days of content, so it gets
    # a thirty-first of the drawn time and no more: `x` stays linear
    # INSIDE kept time, and a rule that gave it a fixed share would be
    # distorting the content to flatter one end of it.
    was = (DAY / (24 * YEAR)) * projecting.WIDTH
    assert was < 0.2, f"a ninth of one pixel, and that was the whole of 2002: {was}"
    assert axis.x(DAY) > was * 200, "collapsing the empty years bought the old day nothing"
    assert axis.x(DAY) < projecting.WIDTH * 0.05, "it took more than a day beside thirty days is owed"


def test_a_gap_too_small_to_be_worth_it_is_left_alone():
    """Kept time stays linear. A day off in the middle of a week is
    information at that zoom, and a band saying `1 day · nothing` would
    be worse than the day."""
    lo, hi = 0.0, 30 * DAY
    axis = projecting.projected(lo, hi, [(0.0, 14 * DAY), (15 * DAY, 30 * DAY)])
    assert axis.collapsed == ()
    assert axis.x(15 * DAY) == pytest.approx(projecting.WIDTH * 0.5, abs=0.01)


def test_nothing_at_the_start_is_a_gap_like_any_other():
    """The leading case, and the one the scrubber actually has: the
    library's extent begins at its oldest picture, so the emptiness sits
    BEFORE the content rather than between two runs of it."""
    lo, hi = 0.0, 20 * YEAR
    axis = projecting.projected(lo, hi, [(19 * YEAR, 20 * YEAR)])
    assert len(axis.collapsed) == 1
    assert axis.collapsed[0].t0 == lo
    assert axis.x(19 * YEAR) == pytest.approx(projecting.COLLAPSED, abs=0.01)


def test_a_click_still_means_a_time():
    """The half a server-side fix would miss. The browser inverts x back
    to a moment for clicks, drags and pans, so a piecewise axis has to
    invert or every gesture lands in the wrong year."""
    lo, hi = 0.0, 30 * DAY
    axis = projecting.projected(lo, hi, [(0.0, DAY), (27 * DAY, 30 * DAY)])
    for t in (0.0, 0.5 * DAY, DAY, 27 * DAY, 28.5 * DAY, hi):
        assert axis.t(axis.x(t)) == pytest.approx(t, abs=1.0), f"{t} did not survive the round trip"


def test_a_click_to_the_right_of_a_gap_lands_after_it():
    """The gesture that would break first: reaching for August must not
    put you in the middle of the three weeks that are not drawn."""
    lo, hi = 0.0, 30 * DAY
    axis = projecting.projected(lo, hi, [(0.0, DAY), (27 * DAY, 30 * DAY)])
    gap = axis.collapsed[0]
    assert axis.t(gap.x1 + 1.0) >= 27 * DAY


def test_the_axis_ends_where_the_axis_ends():
    """Rounding must not leave the last bar a pixel short of the edge,
    which is how a right-hand column goes missing."""
    for occupied in ([(0.0, DAY), (27 * DAY, 30 * DAY)], [(0.0, 30 * DAY)], [(5 * DAY, 6 * DAY)]):
        axis = projecting.projected(0.0, 30 * DAY, occupied)
        assert axis.x(30 * DAY) == projecting.WIDTH
        assert axis.segments[-1].x1 == projecting.WIDTH


def test_two_bins_that_touch_are_not_a_gap():
    """A session lying across several bins arrives as several rows, and
    cutting it into pieces by the bins it happens to cross would invent
    gaps inside content."""
    held = projecting.merged([(0.0, DAY), (DAY, 2 * DAY), (2 * DAY, 3 * DAY)], 0.0, 10 * DAY)
    assert held == [(0.0, 3 * DAY)]


def test_an_empty_window_is_linear_rather_than_a_refusal():
    """A scope that holds nothing is an ordinary state -- a person
    filtered to a keyword nobody has used yet."""
    axis = projecting.projected(0.0, 30 * DAY, [])
    assert axis.collapsed == ()
    assert axis.x(15 * DAY) == pytest.approx(projecting.WIDTH * 0.5, abs=0.01)


def test_the_pictures_never_end_up_with_less_than_half_the_axis():
    """Collapsing must not become its own disease. A window riddled with
    worthwhile gaps would otherwise spend the page on bands saying
    nothing, which is the defect again wearing a label."""
    occupied = [(n * 10 * DAY, n * 10 * DAY + DAY) for n in range(20)]
    axis = projecting.projected(0.0, 200 * DAY, occupied)
    drawn = sum(one.x1 - one.x0 for one in axis.segments if not one.skipped)
    assert drawn >= projecting.WIDTH * 0.5


def test_the_segments_travel_to_the_browser():
    """The client inverts through the same function or it inverts
    through a different one, and a different one is wrong."""
    axis = projecting.projected(0.0, 30 * DAY, [(0.0, DAY), (27 * DAY, 30 * DAY)])
    told = axis.told()
    assert [one["skipped"] for one in told] == [False, True, False]
    assert told[0]["x0"] == 0.0
    assert told[-1]["x1"] == projecting.WIDTH
    for one in told:
        assert one["t1"] >= one["t0"]
        assert one["x1"] >= one["x0"]


def test_a_window_riddled_with_gaps_does_not_run_off_the_end():
    """The bound is load-bearing rather than tidy.

    Fifty worthwhile gaps want fifty bands, and fifty bands are eleven
    hundred units of a thousand-unit axis -- the marks would be drawn
    past the end of the thing they are drawn on. The longest runs are
    collapsed and the rest stay to scale, which is also the honest
    answer for a short one.
    """
    occupied = [(n * 4 * DAY, n * 4 * DAY + 3600.0) for n in range(60)]
    axis = projecting.projected(0.0, 240 * DAY, occupied)
    assert axis.x(240 * DAY) == projecting.WIDTH
    for one in axis.segments:
        assert 0.0 <= one.x0 <= projecting.WIDTH
        assert 0.0 <= one.x1 <= projecting.WIDTH
    assert axis.segments[-1].x1 == projecting.WIDTH
    drawn = sum(one.x1 - one.x0 for one in axis.segments if not one.skipped)
    assert drawn >= projecting.WIDTH * projecting.LEAST_DRAWN
    assert len(axis.collapsed) * projecting.COLLAPSED <= projecting.WIDTH * (1 - projecting.LEAST_DRAWN)


def test_the_longest_runs_are_the_ones_collapsed():
    """When not every gap can have a band, the ones that buy the most
    space get them."""
    occupied = [(0.0, DAY), (2 * DAY, 3 * DAY), (100 * DAY, 101 * DAY), (300 * DAY, 301 * DAY)]
    axis = projecting.projected(0.0, 400 * DAY, occupied, width=60.0)
    assert axis.collapsed, "nothing was collapsed at all"
    longest = max(one.seconds for one in axis.collapsed)
    assert longest > 90 * DAY, "a short gap took a band a long one needed"


# --- through the surface a person actually gets -------------------------------


def _dated(root, name: str, when: float) -> None:
    from PIL import Image

    path = root / name
    Image.new("RGB", (48, 36), (30, 90, 140)).save(path)
    os.utime(path, (when, when))


def _drained(client) -> None:
    from db import connect, runner

    conn = connect.connect(client.app.state.db_path)
    try:
        while runner.run_next(conn, "test-worker", time.time() + 86_400) is not None:
            conn.commit()
        conn.commit()
    finally:
        connect.close(conn)


def _sparse_library(root) -> None:
    """One scanned photograph in 2004 and an afternoon's work this year
    -- the library shape that made the timeline spend nine tenths of
    itself on nothing."""
    _dated(root, "scanned.png", datetime.datetime(2004, 3, 1, 12, tzinfo=datetime.UTC).timestamp())
    for i in range(6):
        _dated(root, f"today_{i:02d}.png", datetime.datetime(2026, 8, 20, 10 + i, tzinfo=datetime.UTC).timestamp())


def _interpreted(stage) -> None:
    _drained(stage.client)
    for job in ("/jobs/ingest", "/jobs/context"):
        stage.client.post(job)
        _drained(stage.client)


@pytest.fixture(scope="module")
def _world(tmp_path_factory):
    with staged(tmp_path_factory, "test_empty_time_does_not_get_the_pixels", _sparse_library, _interpreted) as stage:
        yield stage


@pytest.fixture
def sparse(_world):
    """The sparse library, built once: the five claims over it only read."""
    _world.restore()
    return _world.client


WHOLE = "start=1078099200&end=1787270400"  # 2004-03-01 to 2026-08-21


def test_the_whole_library_stops_being_two_decades_of_nothing(sparse):
    """The complaint, end to end. Twenty-two years separate one
    photograph from the rest, and drawing them to scale left both ends
    as hairlines on a strip of blank."""
    told = sparse.get(f"/timeline?{WHOLE}", headers={"accept": "application/json"}).json()
    assert [one["lasted"] for one in told["skipped"]] == ["22 years"]
    gap = told["skipped"][0]
    assert gap["w"] < 30, "the collapsed run is still taking the axis"

    # both ends are now something a person can see and aim at
    bars = told["bins"]
    assert len(bars) == 2
    assert all(one["w"] > 20 for one in bars), [one["w"] for one in bars]


def test_the_band_says_how_long_rather_than_being_blank(sparse):
    """Blank pixels are ambiguous between "no pictures", "nothing dated
    this" and "the render broke". The band is not."""
    page = sparse.get(f"/timeline?{WHOLE}", headers={"accept": "text/html"}).text
    assert "data-axis-skipped" in page
    assert "22 years" in page
    assert "with no pictures" in page, "the band does not say what it is"


def test_the_band_is_a_way_in_rather_than_a_wall(sparse):
    """A run with no pictures is still a range somebody may want to look
    at -- to see that it really is empty, or to find what was never
    dated. The band opens it."""
    told = sparse.get(f"/timeline?{WHOLE}", headers={"accept": "application/json"}).json()
    href = told["skipped"][0]["href"]
    assert href, "the band goes nowhere"
    assert "start=" in href
    assert "end=" in href
    opened = sparse.get(href, headers={"accept": "application/json"})
    assert opened.status_code == 200, opened.text


def test_the_browser_is_given_the_axis_it_must_invert(sparse):
    """A click, a drag and a pan all turn x back into a moment. Without
    the segments the browser would invert through a linear axis the bars
    were not drawn on, and every gesture would land in the wrong year --
    silently, because a wrong moment is still a plausible one."""
    page = sparse.get(f"/timeline?{WHOLE}", headers={"accept": "text/html"}).text
    assert "data-axis=" in page
    told = sparse.get(f"/timeline?{WHOLE}", headers={"accept": "application/json"}).json()
    assert [one["skipped"] for one in told["segments"]] == [False, True, False]
    assert told["segments"][-1]["x1"] == pytest.approx(projecting.WIDTH)


def test_the_scrubber_collapses_on_the_same_rule(sparse):
    """The strip under the chart is the twenty-four-year one, and it had
    the same defect for the same reason. The vertical rail beside it has
    always run empty bins together (`_scrubber`); the horizontal strips
    never did."""
    told = sparse.get(f"/timeline?{WHOLE}", headers={"accept": "application/json"}).json()
    overview = told["overview"]
    assert overview is not None
    assert [one["lasted"] for one in overview["skipped"]] == ["22 years"]
    # the 2004 photograph is no longer a hairline at the far left
    assert overview["bars"][0]["w"] >= 3.0
    assert overview["skipped"][0]["w"] < 30
