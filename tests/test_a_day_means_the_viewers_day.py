"""A day in the date filter must be the day the pictures are labelled with.

The date on a card is drawn by the browser:

    new Date(file.mtime * 1000).toLocaleString(...)

The filter boundary was worked out by the server:

    datetime.strptime(start_date, '%Y-%m-%d').timestamp()   # naive: local

Those are two different clocks whenever the machine showing the gallery
is not the machine running it, which is the ordinary case: the shipped
compose file leaves `/etc/localtime` commented out, so the container runs
in UTC while the person looking at it does not.

Filtering a single day, against one picture per hour, on a UTC server:

    the person looking at it   of their 24 hours   also shown, from
                               the filter returns   another day
    New Zealand    (UTC+13)          11                 13
    US west coast  (UTC-7)           17                  7
    India          (UTC+5:30)        18                  5
    Germany        (UTC+2)           22                  2

Nothing reports this. The pictures are all still there, they are simply
not in the answer, and the day they are labelled with is the day that was
asked for.

The browser is the only thing that knows which instants its own day
covers, so the page now works out the two boundaries and sends them, and
they are used when they arrive. A link with only the dates in it -- a
bookmark, a typed URL -- cannot say whose day it means, so that still
reads as the server's day, exactly as before.

Two more things this settles.

The end of a day was the start plus 86399 seconds, which is a day only
when the day is 24 hours long:

    2026-11-01   local day 25.0 hours   +86399 ends 22:59:59  (-3600s)
    2026-03-08   local day 23.0 hours   +86399 ends 00:59:59  (+3600s, next day)

So on the day the clocks go back the last hour was missing, and on the
day they go forward an hour of the next day was included.

And the count on the Filters button never included a date range. The
increment was there, but `active_filters_count` does not exist yet at
that point in gallery_view, so it raised UnboundLocalError into a bare
`except: pass` on every single request that filtered by date. The
condition had already been added, so the filtering worked and only the
counting was lost -- which is exactly why nobody noticed.
"""

from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timedelta

import pytest

import smartgallery


DAY = "2026-08-15"


def server_midnight(day=DAY):
    """Midnight on this machine, the way the app reads a typed date."""
    return datetime.strptime(day, "%Y-%m-%d").timestamp()


@pytest.fixture
def a_library_by_the_hour(smartgallery_app, tmp_path, monkeypatch):
    """One picture at each of a set of offsets from the server's midnight."""
    sg = smartgallery_app
    root = tmp_path / "dated_root"
    root.mkdir()
    monkeypatch.setattr(sg, "BASE_OUTPUT_PATH", str(root))

    base = server_midnight()
    hours = [2, 11, 30]
    ids = {}
    with sg.get_db_connection() as conn:
        conn.execute("DELETE FROM files")
        for hour in hours:
            path = str(root / ("h%02d.png" % hour))
            with open(path, "wb") as handle:
                handle.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
            fid = hashlib.md5(path.encode("utf-8")).hexdigest()
            conn.execute(
                "INSERT OR REPLACE INTO files (id, path, mtime, name, type, "
                "has_workflow, size, last_scanned) VALUES (?,?,?,?,?,?,?,?)",
                (fid, path, base + hour * 3600, "h%02d.png" % hour, "image",
                 0, 64, 1.0))
            ids[hour] = fid
        conn.commit()

    client = sg.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "ADMIN"

    def shown(**args):
        page = client.get("/galleryout/view/_root_", query_string=dict(
            args, scope="global", recursive="true"), follow_redirects=True)
        assert page.status_code == 200, page.status_code
        body = page.get_data(as_text=True)
        return {hour for hour, fid in ids.items() if fid in body}

    yield shown

    with smartgallery.get_db_connection() as conn:
        conn.execute("DELETE FROM files")
        conn.commit()


def test_the_date_filter_narrows_at_all(a_library_by_the_hour):
    """Control. Without this a filter that does nothing passes everything
    below by showing the whole library."""
    shown = a_library_by_the_hour

    assert shown() == {2, 11, 30}, "no filter should show every picture"
    assert shown(start_date=DAY, end_date=DAY) == {2, 11}, \
        "the server's own day is hours 0..24, so 30 is the next day"


def test_a_bookmarked_link_still_reads_as_the_servers_day(a_library_by_the_hour):
    """Control, and a deliberate limit: a URL carrying only `2026-08-15`
    does not say whose day it means, so it keeps meaning what it meant."""
    shown = a_library_by_the_hour

    assert shown(start_date=DAY, end_date=DAY) == {2, 11}


def test_the_two_clocks_cannot_agree_by_luck():
    """The premise, in arithmetic, on any machine in any timezone.

    Someone ten hours east of the server has a day covering +10h to +34h
    of the server's, so a picture at +30h is on their chosen day and
    outside the server's window for it. If this ever stopped being true
    there would be nothing here to fix."""
    base = server_midnight()
    servers_day = (base, base + 86399)
    theirs = (base + 10 * 3600, base + 34 * 3600 - 1)

    picture = base + 30 * 3600
    assert theirs[0] <= picture <= theirs[1], "on the day they asked for"
    assert not servers_day[0] <= picture <= servers_day[1], \
        "and outside the day the server would have filtered"


def test_the_boundaries_the_page_sends_are_the_ones_used(a_library_by_the_hour):
    """The defect. The browser's day here runs from +10h to +34h of the
    server's -- some ten hours away, as New Zealand or California is --
    and it is that day the pictures are labelled with."""
    shown = a_library_by_the_hour
    base = server_midnight()

    theirs = shown(start_date=DAY, end_date=DAY,
                   start_ts=str(base + 10 * 3600),
                   end_ts=str(base + 34 * 3600 - 1))

    assert 30 in theirs, (
        "a picture the viewer sees dated on the chosen day was left out, "
        "because the server's clock put it on the next one")
    assert 11 in theirs
    assert 2 not in theirs, (
        "a picture from before the viewer's day began was included")


def test_only_one_end_can_be_given(a_library_by_the_hour):
    """Over-reach guard: the two ends are independent, as they were."""
    shown = a_library_by_the_hour
    base = server_midnight()

    assert shown(start_date=DAY, start_ts=str(base + 10 * 3600)) == {11, 30}
    assert shown(end_date=DAY, end_ts=str(base + 10 * 3600)) == {2}


def test_something_that_is_not_a_number_is_ignored(a_library_by_the_hour):
    """Over-reach guard: anything can arrive in a URL. A boundary that is
    not a number falls back to the date rather than dropping the filter
    or raising."""
    shown = a_library_by_the_hour

    # NaN and inf both survive float(); a NaN boundary compares false
    # against every row, so the gallery empties while the filter still
    # shows the date that was asked for.
    for rubbish in ("", "abc", "2026-08-15", "NaN", "nan", "inf", "-inf", "[]"):
        assert shown(start_date=DAY, end_date=DAY,
                     start_ts=rubbish, end_ts=rubbish) == {2, 11}, rubbish


class TestTheEndOfADay:
    """The end of a day is the next midnight, not 86399 seconds later."""

    @pytest.mark.parametrize("day", [
        "2026-01-15", "2026-03-08", "2026-06-21", "2026-08-15",
        "2026-10-25", "2026-11-01", "2026-12-31", "2026-02-28",
    ])
    def test_it_is_the_last_second_before_the_next_one(self, day):
        """True in every timezone: where there is no change of clocks this
        is the same as +86399, and where there is, it is not."""
        next_midnight = (datetime.strptime(day, "%Y-%m-%d")
                         + timedelta(days=1)).timestamp()

        assert smartgallery.day_bounds(day, None, True) == next_midnight - 1

    def test_the_start_is_that_midnight(self):
        assert smartgallery.day_bounds(DAY, None, False) == server_midnight()

    def test_a_day_whose_length_is_not_24_hours(self):
        """The days this actually changes. Skipped where the machine's own
        timezone never changes its clocks -- the check above still holds
        the rule there."""
        odd = []
        for day in ("2026-03-08", "2026-03-29", "2026-10-25", "2026-11-01",
                    "2026-04-05", "2026-09-27"):
            midnight = datetime.strptime(day, "%Y-%m-%d").timestamp()
            following = (datetime.strptime(day, "%Y-%m-%d")
                         + timedelta(days=1)).timestamp()
            if following - midnight != 86400:
                odd.append((day, midnight, following))

        if not odd:
            pytest.skip("this machine's timezone has no daylight saving; "
                        "test_it_is_the_last_second_before_the_next_one "
                        "holds the rule regardless")

        for day, midnight, following in odd:
            was = midnight + 86399
            now = smartgallery.day_bounds(day, None, True)
            assert now == following - 1
            assert now != was, day

    def test_nothing_asked_for_is_no_boundary(self):
        assert smartgallery.day_bounds("", None, False) is None
        assert smartgallery.day_bounds(None, None, True) is None
        assert smartgallery.day_bounds("not a date", None, False) is None


def test_the_filters_button_counts_a_date_range(smartgallery_app):
    """The count the bare `except: pass` was eating."""
    sg = smartgallery_app
    client = sg.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "ADMIN"

    def counted(**args):
        page = client.get("/galleryout/view/_root_", query_string=args,
                          follow_redirects=True)
        assert page.status_code == 200, page.status_code
        found = re.search(r"const activeFiltersCount = (\d+);",
                          page.get_data(as_text=True))
        assert found, "the page no longer states how many filters are on"
        return int(found.group(1))

    none_at_all = counted()
    assert counted(start_date=DAY) == none_at_all + 1, \
        "a start date is a filter and was not counted"
    assert counted(end_date=DAY) == none_at_all + 1, \
        "an end date is a filter and was not counted"
    assert counted(start_date=DAY, end_date=DAY) == none_at_all + 2


class TestThePageSendsThem:
    """The other half of the fix lives in the template."""

    def _page(self):
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "templates", "index.html"),
                  encoding="utf-8") as handle:
            return handle.read()

    def test_the_two_boundaries_are_in_the_filter_form(self):
        page = self._page()

        assert 'name="start_ts"' in page
        assert 'name="end_ts"' in page

    def test_they_are_worked_out_from_the_chosen_dates(self):
        """The computation has to be the browser's own Date, because that
        is what knows this timezone and its changes of clock."""
        page = self._page()

        assert "localDayEdge" in page, \
            "nothing works out the boundaries of the chosen day"
        assert "new Date(Number(parts[1]), Number(parts[2]) - 1," in page

    def test_they_are_not_carried_back_from_the_url(self):
        """A boundary is only true for the browser that computed it, so
        the field must start empty and be filled in on submit."""
        page = self._page()

        for field in ("start_ts", "end_ts"):
            spot = page.index('name="%s"' % field)
            line = page[spot:page.index(">", spot)]
            assert "request.args" not in line, \
                f"{field} is refilled from the URL: {line}"
            assert 'value=""' in line, line
