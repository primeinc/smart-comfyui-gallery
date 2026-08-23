"""The timeline: a VIEW over the one interpretation, never a feature
database.

Months and days come from derived_media_context (the local wall clock
when one was claimed, the knowable instant otherwise) and the overlay
comes from the latest event runs. Every month, day, bin and session is
a DOOR into the gallery -- spelled by the Facet Interface, so the
ResultSet answers the media and the timeline never grows a second
membership engine. Nothing here writes, groups, geocodes or interprets:
POST /jobs/context and POST /jobs/events are where the interpretation
is refreshed, and this page renders whatever they last produced, saying
how much of the library that is. An empty timeline names its own remedy
instead of pretending an unindexed library has no past.
"""

from __future__ import annotations

import dataclasses
import datetime
import pathlib
import time
from typing import Annotated

from litestar import Request, get
from litestar.datastructures import State
from litestar.exceptions import ClientException, NotFoundException
from litestar.params import FromQuery, QueryParameter
from litestar.response import Response, Template

from db import connect, context, facets, pages, planning, rendering, resultset, settings
from sg_web import home
from sg_web.asking import gallery_query as _asked
from sg_web.presenting import VARIES, presented

#: The surface's scope is a gallery question (db/resultset.py scope_of):
#: its scopes and facets in the live spelling, unsorted, unpaged. The
#: timeline never invents its own spelling of a door -- every door is
#: that question plus the facets the door adds, ordered by moment.
WHOLE = resultset.GalleryQuery()


def _door(question: resultset.GalleryQuery, *held: facets.Facet) -> str:
    asked = dataclasses.replace(
        question, facets=tuple(sorted({*question.facets, *held}, key=facets.spell)), sort="moment", text=None
    )
    return resultset.canonical(asked)


def _bin_door(at: float, width: int, question: resultset.GalleryQuery = WHOLE) -> str:
    """The bar's pictures, exactly: the window, and the precision the
    count applied -- a day-precision claim sitting at midnight inside an
    hour's window was not counted in that bar and must not open from it."""
    low = facets.facet("context.moment", "gte", str(int(at)))
    high = facets.facet("context.moment", "lt", str(int(at) + width))
    fine = facets.facet("context.granule", "lte", str(int(width)))
    return _door(question, low, high, fine)


def _event_door(event_id: int, question: resultset.GalleryQuery = WHOLE) -> str:
    return _door(question, facets.facet("event.id", "eq", str(event_id)))


def _question(folder, album, person, artifact, kind, favorite, rating_min, f) -> resultset.GalleryQuery:
    """The surface's scope as the gallery's own question, parsed by the
    one seam that owns query semantics; a bad spelling is refused with
    the vocabulary. A session is a door, not a scope: `event.id` names
    one session, and a timeline of one session is the gallery's job."""
    try:
        asked = _asked(
            folder,
            album,
            kind,
            None,
            None,
            None,
            person=person,
            artifact=artifact,
            favorite=favorite,
            rating_min=rating_min,
            facets=f,
        )
    except ValueError as refused:
        raise ClientException(str(refused)) from refused
    if any(one.key == "event.id" for one in asked.facets):
        raise ClientException("a session is a door, not a scope; open it in the gallery")
    return asked


def _scope(conn, state: State, asked: resultset.GalleryQuery) -> tuple[tuple[str, list], resultset.GalleryQuery]:
    """The question bound: (the conjunct and its values, the question in
    its live spelling). A slug nothing lives at is a 404; a rule-defined
    collection that cannot be answered right now is refused with why."""
    try:
        sql, values, live = resultset.scope_of(
            conn,
            asked,
            state.actor_id,
            models_dir=str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir"))),
            now=time.time(),
        )
    except LookupError as missing:
        raise NotFoundException(str(missing)) from missing
    except ValueError as refused:
        raise ClientException(str(refused)) from refused
    return (sql, values), live


def _scope_told(question: resultset.GalleryQuery) -> dict | None:
    """What the page says it is scoped to, or None for the whole library:
    the canonical spelling and its parts, one per scope and facet."""
    parts = [
        {"key": key, "value": value}
        for key, value in (
            ("folder", question.folder),
            ("album", question.album),
            ("person", question.person),
            ("artifact", question.artifact),
            ("kind", question.kind),
            ("favorite", question.favorite),
            ("rating_min", question.rating_min),
        )
        if value is not None
    ] + [{"key": one.key, "value": one.value, "spelled": facets.spell(one)} for one in question.facets]
    if not parts:
        return None
    return {"qs": resultset.canonical(question), "parts": parts}


#: Which planner tells which kind of session's story; a kind with none
#: is offered no button and told why.
PLANNER_FOR = {
    "generation_session": "generation_history",
    "capture_session": "capture_history",
    "file_session": "file_history",
}

_SPAN = {"day": 86_400, "hour": 3_600, "minute": 60}

#: Sessions one answer lists -- a whole library's extent can touch
#: thousands; the page lists the most recent this many, says how many
#: more there are, and the person narrows the window. Never a silent
#: cut. Every listed session carries its thumbnails.
SESSIONS_MOST = 200
SESSIONS_SAMPLED_MOST = SESSIONS_MOST
#: How much time a first visit shows: the last month that holds pictures,
#: clipped to the library -- never the whole library at once.
OPENING = 30 * 86_400
#: The presets beside the window, each ending at the newest picture.
PRESETS = (("1w", 7 * 86_400), ("1m", 30 * 86_400), ("3m", 91 * 86_400), ("1y", 365 * 86_400), ("all", None))
#: The zoom follows the window's width: enough bars to see the shape,
#: never more than the strip samples thumbnails for.
_ZOOM = (("minute", 6 * 3_600), ("quarter", 2 * 86_400), ("hour", 14 * 86_400), ("day", 183 * 86_400))
#: The narrowest window the surface opens on.
NARROWEST = 3_600
#: The drawing's width in its own units (the SVG viewBox).
_W = 1000


def _height(pictures: int, most: int, full: float) -> float:
    """A bar's height on a square-root scale: one week holding most of
    the library still leaves the other bursts visible as bars, not
    hairlines."""
    return (pictures / most) ** 0.5 * full if pictures else 0.0


def _coverage(conn, scope: tuple[str, list] = ("", []), question: resultset.GalleryQuery = WHOLE) -> dict:
    have, present, contested = pages.timeline_coverage(conn, scope)
    return {
        "interpreted": have,
        "present": present,
        "contested": contested,
        #: a session run answers only at the current interpretation
        #: (db/pages.py TIMELINE_EVENTS); one authored place moves the
        #: generation and every session door goes dark until the events
        #: job runs again -- the page names that remedy, never an empty list
        "events_current": pages.timeline_events_current(conn),
        "contested_qs": _door(question, facets.facet("context.disputed", "eq", "1")),
        "policy_version": context.POLICY_VERSION,
        "complete": have == present,
    }


def _session(conn, row, *, samples: bool, scope: resultset.GalleryQuery = WHOLE) -> dict:
    (
        event_id,
        kind,
        local_start,
        local_end,
        instant_start,
        instant_end,
        pictures,
        snapshot_id,
        render_id,
        place_id,
        place_name,
        place_slug,
        here,
    ) = row
    planner = PLANNER_FOR.get(kind)
    return {
        #: where the session happened: the one place its placed members
        #: agree on (db/events.py _shared_place), with the gallery door
        "place": (
            {
                "id": place_id,
                "name": place_name,
                "slug": place_slug,
                "qs": _door(scope, facets.facet("place.id", "eq", str(place_id))),
            }
            if place_id is not None
            else None
        ),
        "id": event_id,
        "kind": kind,
        "domain": "wall" if local_start is not None else "instant",
        "start": local_start if local_start is not None else instant_start,
        "end": local_end if local_end is not None else instant_end,
        "pictures": pictures,
        #: of those, how many the surface's scope holds (all, unscoped)
        "in_scope": int(here),
        "snapshot_id": snapshot_id,
        #: the story told of this session, on its card: title, dek, heroes
        "story": rendering.story_card(conn, render_id) if render_id is not None else None,
        "qs": _event_door(event_id, scope),
        "planner": planner,
        "tellable": planner in planning.PLANNERS,
        "samples": pages.session_samples(conn, event_id) if samples else [],
        "people": [
            {"slug": slug, "name": name, "href": f"/p/{slug}", "pictures": int(count)}
            for slug, name, count in pages.session_people(conn, event_id)
        ],
        "people_total": pages.session_people_total(conn, event_id),
    }


def _bin_for(width: float) -> str:
    for name, most in _ZOOM:
        if width <= most:
            return name
    return "week"


def _day(d: datetime.datetime) -> str:
    return f"{d.day} {d.strftime('%b %Y')}"


def _spell(epoch: float, bin_name: str) -> str:
    """A moment as a person reads it: the day at the day and week zooms
    ("10 Jun 2023", "week of 5 Jun 2023"), the day and the clock below."""
    d = datetime.datetime.fromtimestamp(epoch, datetime.UTC)
    if bin_name == "week":
        return f"week of {_day(d)}"
    if bin_name == "day":
        return _day(d)
    return f"{_day(d)}, {d.strftime('%H:%M')}"


def _span(lo: float, hi: float) -> str:
    """A range as a person reads it: "22 Jul – 21 Aug 2026" across days,
    "21 Aug 2026, 01:33 – 01:39" within one."""
    a = datetime.datetime.fromtimestamp(lo, datetime.UTC)
    b = datetime.datetime.fromtimestamp(max(lo, hi - 1), datetime.UTC)
    if a.date() == b.date():
        return f"{_day(a)}, {a.strftime('%H:%M')} – {b.strftime('%H:%M')}"
    if a.year == b.year:
        return f"{a.day} {a.strftime('%b')} – {_day(b)}"
    return f"{_day(a)} – {_day(b)}"


#: What a session is, said from the person's side: what happened to the
#: pictures, not which grouper found them.
HAPPENED = {
    "capture_session": "photos taken",
    "generation_session": "pictures generated",
    "file_session": "files added",
}
#: A bar's unit, as a person says it.
UNIT = {"week": "week", "day": "day", "hour": "hour", "quarter": "quarter hour", "minute": "minute"}


#: Pictures one surface draws in place. Past it the page says how many
#: more there were; the person narrows the window.
PICTURES_MOST = 2_000
#: At the week zoom the body is month sheets; finer, it is the river.
CALENDAR_AT = "week"
_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_TICK_STEPS = (
    (3_600, 300),
    (6 * 3_600, 1_800),
    (2 * 86_400, 3 * 3_600),
    (14 * 86_400, 86_400),
    (float("inf"), 7 * 86_400),
)


def _utc(epoch: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(epoch, datetime.UTC)


def _month_start(d: datetime.datetime) -> float:
    return datetime.datetime(d.year, d.month, 1, tzinfo=datetime.UTC).timestamp()


def _next_month(d: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(d.year + (d.month == 12), d.month % 12 + 1, 1, tzinfo=datetime.UTC)


def _ticks(lo: float, hi: float) -> list[dict]:
    """The axis's furniture: a tick per step of the zoom with its label,
    major at the calendar boundary above it (midnight, the 1st, January)."""
    span = hi - lo

    def x(t: float) -> float:
        return round(((t - lo) / span) * _W, 2)

    out: list[dict] = []
    if span > 183 * 86_400:
        # months up to three years; past that only the years, else the
        # labels pile into one another
        years_only = span > 3 * 366 * 86_400
        d = _utc(lo)
        d = datetime.datetime(d.year, d.month, 1, tzinfo=datetime.UTC)
        while d.timestamp() < hi:
            t = d.timestamp()
            if t >= lo and (d.month == 1 or not years_only):
                out.append(
                    {
                        "x": x(t),
                        "label": str(d.year) if d.month == 1 else d.strftime("%b").upper(),
                        "major": d.month == 1,
                    }
                )
            d = _next_month(d)
        return out
    step = next(s for most, s in _TICK_STEPS if span <= most)
    t = -(-int(lo) // step) * step
    while t < hi:
        d = _utc(t)
        midnight = t % 86_400 == 0
        day_label = f"{d.day} {d.strftime('%b').upper()}"
        if step >= 86_400:
            out.append({"x": x(t), "label": day_label, "major": d.day <= step // 86_400})
        else:
            out.append({"x": x(t), "label": day_label if midnight else d.strftime("%H:%M"), "major": midnight})
        t += step
    return out


def _picture(row, qs: str) -> dict:
    slug, name, kind, width, height, moment, precision, origin, wall, sessions = row
    return {
        "slug": slug,
        "name": name,
        "kind": kind,
        "width": width,
        "height": height,
        "ratio": round(width / height, 4) if width and height else 1.0,
        "moment": moment,
        "precision": precision,
        "origin": origin,
        "domain": "wall" if wall else "instant",
        "sessions": [int(one) for one in sessions.split(",")] if sessions else [],
        "href": f"/i/{slug}?{qs}",
        "clock": _utc(moment).strftime("%H:%M"),
    }


def _grouped(pictures: list[dict], sessions: list[dict], bins: list[dict], width: int, window_qs: str) -> list[dict]:
    """The window's pictures in groups, oldest first: each listed session
    holds its members; the rest gather by the bin they fall in. A
    picture in two listed sessions is drawn in the first."""
    by_session = {
        s["id"]: {"t": s["start"], "end": s["end"], "session": s, "bin": None, "qs": s["qs"], "pictures": []}
        for s in sessions
    }
    by_bin = {b["at"]: b for b in bins}
    loose: dict[int, dict] = {}
    for p in pictures:
        sid = next((one for one in p["sessions"] if one in by_session), None)
        if sid is not None:
            by_session[sid]["pictures"].append(p)
            continue
        at = int(p["moment"] // width) * width
        held = loose.get(at)
        if held is None:
            b = by_bin.get(at)
            held = loose[at] = {
                "t": at,
                "end": at + width,
                "session": None,
                "bin": b,
                "qs": b["qs"] if b else window_qs,
                "pictures": [],
            }
        held["pictures"].append(p)
    groups = [g for g in (*by_session.values(), *loose.values()) if g["pictures"]]
    groups.sort(key=lambda g: g["t"])
    for g in groups:
        g["clock"] = f"{_utc(g['t']).strftime('%H:%M')}–{_utc(g['end']).strftime('%H:%M')}"
        g["lasted"] = _lasted(g["end"] - g["t"])
    return groups


def _lasted(seconds: float) -> str:
    if seconds < 90:
        return "a minute"
    if seconds < 3_600:
        return f"{round(seconds / 60)} min"
    if seconds < 2 * 86_400:
        return f"{seconds / 3_600:.1f} h"
    return f"{round(seconds / 86_400)} days"


def _river(groups: list[dict]) -> list[dict]:
    """Days, oldest first, each with its groups; a day carries the month
    cap when it opens one, and the count of empty days before it."""
    days: list[dict] = []
    by_day: dict[str, list] = {}
    for g in groups:
        by_day.setdefault(_utc(g["t"]).strftime("%Y-%m-%d"), []).append(g)
    prev_day, prev_month = None, None
    for key in sorted(by_day):
        d = datetime.datetime.strptime(key, "%Y-%m-%d").replace(tzinfo=datetime.UTC)
        month = (d.year, d.month)
        gap = 0 if prev_day is None or month != prev_month else (d - prev_day).days - 1
        days.append(
            {
                "key": key,
                "day": d.day,
                "weekday": _WEEKDAYS[d.weekday()],
                "weekend": d.weekday() >= 5,
                "month_cap": {"year": d.year, "month": _MONTHS[d.month - 1]} if month != prev_month else None,
                "gap_before": max(0, gap),
                "groups": by_day[key],
            }
        )
        prev_day, prev_month = d, month
    return days


def _calendar(conn, lo: float, hi: float, scope, question: resultset.GalleryQuery) -> list[dict]:
    """Month sheets over the window, newest first: every day a cell with
    its count and first picture; day bins from the one density query."""
    first = _month_start(_utc(lo))
    last = _next_month(_utc(max(lo, hi - 1))).timestamp()
    _, day_bins, _ = pages.timeline_density(conn, "day", first, last, scope)
    samples = pages.timeline_samples(conn, "day", first, last, None, scope)
    counts = {int(at): pictures for at, pictures, *_ in day_bins}
    today = _utc(time.time()).strftime("%Y-%m-%d")
    months = []
    d = _utc(first)
    while d.timestamp() < last:
        nxt = _next_month(d)
        days, total = [], 0
        t = d.timestamp()
        while t < nxt.timestamp():
            n = counts.get(int(t), 0)
            total += n
            days.append(
                {
                    "n": _utc(t).day,
                    "pictures": n,
                    "hero": (samples.get(int(t)) or [None])[0],
                    "qs": _bin_door(t, 86_400, question) if n else None,
                    "spelled": _spell(t, "day"),
                    "today": _utc(t).strftime("%Y-%m-%d") == today,
                }
            )
            t += 86_400
        months.append(
            {"year": d.year, "month": _MONTHS[d.month - 1], "lead": d.weekday(), "pictures": total, "days": days}
        )
        d = nxt
    months.reverse()
    return months


#: The widest window that draws month sheets; wider, the body is years.
SHEETS_WIDEST = 2 * 366 * 86_400


def _years(told_bins: list[dict], lo: float, hi: float, question: resultset.GalleryQuery) -> list[dict]:
    """Year rows, newest first, each twelve month cells with the count
    and first picture -- from the window's week bins, a week counted in
    the month it starts in."""
    months: dict[tuple[int, int], dict] = {}
    for b in told_bins:
        d = _utc(b["at"])
        held = months.setdefault((d.year, d.month), {"pictures": 0, "hero": None})
        held["pictures"] += b["pictures"]
        if held["hero"] is None and b["samples"]:
            held["hero"] = b["samples"][0]
    years = []
    for y in range(_utc(lo).year, _utc(max(lo, hi - 1)).year + 1):
        cells = []
        for m in range(1, 13):
            start = datetime.datetime(y, m, 1, tzinfo=datetime.UTC)
            end = _next_month(start)
            held = months.get((y, m), {"pictures": 0, "hero": None})
            door = _door(
                question,
                facets.facet("context.moment", "gte", str(int(start.timestamp()))),
                facets.facet("context.moment", "lt", str(int(end.timestamp()))),
            )
            cells.append(
                {
                    "month": _MONTHS[m - 1][:3].upper(),
                    "pictures": held["pictures"],
                    "hero": held["hero"],
                    "qs": door if held["pictures"] else None,
                    "href": _window_url(question, start.timestamp(), end.timestamp()),
                    "outside": end.timestamp() <= lo or start.timestamp() >= hi,
                }
            )
        years.append({"year": y, "pictures": sum(c["pictures"] for c in cells), "months": cells})
    years.reverse()
    return years


def _window_url(question: resultset.GalleryQuery, start: float, end: float) -> str:
    qs = resultset.canonical(question)
    return f"/timeline?{qs + '&' if qs else ''}start={int(start)}&end={int(end)}"


def _surface(conn, state: State, asked: resultset.GalleryQuery, start, end, *, bin_name=None, lean=False) -> dict:
    """The surface at one window: pictures per bin of the human moment
    over [start, end) -- the last month that holds pictures when no
    window is asked -- split by clock domain and by origin, with a
    thumbnail sample per bin (the busiest past SAMPLED_BINS_MOST); the
    claims too coarse for the bin as spans; the sessions touching the
    window in their own domain, each a door to its pictures and to its
    story; the whole extent at week resolution as the overview the
    brush rides. The zoom follows the window's width unless `bin_name`
    asks for one. Every bin is a door into the gallery, carrying the
    scope. `lean` is the shape alone: no thumbnails, no session cards.
    A window wider than the page can draw is refused with the remedy."""
    scope, held = _scope(conn, state, asked)
    coverage = _coverage(conn, scope, held)
    extent = pages.timeline_extent(conn, scope)
    scope_told = _scope_told(held)
    if extent is None or extent[0] is None:
        return {
            "bin": bin_name or "day",
            "bin_seconds": pages.BINS.get(bin_name or "day"),
            "start": None,
            "end": None,
            "start_spelled": "",
            "end_spelled": "",
            "window_spelled": "",
            "unit": "day",
            "scope": scope_told,
            "extent": None,
            "overview": None,
            "presets": [],
            "coverage": coverage,
            "sampled": True,
            "bins": [],
            "spans": [],
            "note": "",
            "sessions": [],
            "sessions_total": 0,
            "sessions_sampled": True,
            "composition": "river",
            "ticks": [],
            "now_x": None,
            "pictures_total": 0,
            "pictures_drawn": 0,
            "groups": [],
            "river": [],
            "calendar": [],
            "years": [],
            "listed": [],
        }
    whole_lo, whole_hi = float(extent[0]), float(extent[1]) + 1.0
    if start is None and end is None:
        # the opening window: the last month, tightened to where its
        # pictures actually sit -- a month whose pictures all fall on one
        # day opens on that day, not on one bar in thirty days of nothing
        lo, hi = max(whole_lo, whole_hi - OPENING), whole_hi
        first, last, _ = pages.timeline_span(conn, lo, hi, scope)
        if first is not None:
            lo, hi = max(lo, float(first)), min(hi, float(last) + 1.0)
            if hi - lo < NARROWEST:
                lo = max(whole_lo, hi - NARROWEST)
    else:
        lo = float(start) if start is not None else whole_lo
        hi = float(end) if end is not None else whole_hi
    bin_name = bin_name or _bin_for(hi - lo)
    try:
        width, bins, spans = pages.timeline_density(conn, bin_name, lo, hi, scope)
        overview_width, overview_bins, _ = pages.timeline_density(conn, "week", whole_lo, whole_hi, scope)
    except ValueError as refused:
        raise ClientException(str(refused)) from refused
    every = len(bins) <= pages.SAMPLED_BINS_MOST
    busiest = (
        None if every else [at for at, pictures, *_ in sorted(bins, key=lambda b: -b[1])[: pages.SAMPLED_BINS_MOST]]
    )
    samples = {} if lean else pages.timeline_samples(conn, bin_name, lo, hi, busiest, scope)
    rows = [] if lean else pages.timeline_sessions(conn, lo, hi, scope)
    # rows are oldest first (db/pages.py _TIMELINE_SESSIONS_TAIL); the tail is the latest
    listed = rows[-SESSIONS_MOST:] if len(rows) > SESSIONS_MOST else rows
    sessions = [_session(conn, row, samples=len(listed) <= SESSIONS_SAMPLED_MOST, scope=held) for row in listed]
    span = max(1.0, hi - lo)
    for one in sessions:
        one["when"] = _span(one["start"], one["end"] + 1)
        one["happened"] = HAPPENED.get(one["kind"], one["kind"].replace("_", " "))
        one["title"] = one["story"]["title"] if one["story"] else f"{one['pictures']:,} {one['happened']}"
        named = [p for p in one["people"] if p["name"]]
        others = one["people_total"] - len(named)
        one["with"] = {"named": named, "others": others}
        one["lasted"] = _lasted(one["end"] - one["start"])
        # the session's frame on the axis, clipped to the window
        x0 = max(0.0, ((one["start"] - lo) / span) * _W)
        x1 = min(float(_W), ((one["end"] + 1 - lo) / span) * _W)
        one["x"], one["w"] = round(x0, 2), round(max(2.0, x1 - x0), 2)
    window_qs = _door(
        held, facets.facet("context.moment", "gte", str(int(lo))), facets.facet("context.moment", "lt", str(int(hi)))
    )
    picture_rows, pictures_total = ([], 0) if lean else pages.timeline_pictures(conn, lo, hi, PICTURES_MOST, scope)
    drawn_rows = [_picture(row, window_qs) for row in picture_rows]
    most = max([1, *(pictures for _, pictures, *_ in bins)])
    bar_w = max(1.0, (width / span) * _W - 0.5)
    finest = bin_name == "minute"
    told_bins = []
    for at, pictures, wall, instant, captured, generated, mixed, imported in bins:
        h = _height(pictures, most, 100)
        told_bins.append(
            {
                "at": at,
                "pictures": pictures,
                "wall": wall,
                "instant": instant,
                "origin": {"captured": captured, "generated": generated, "mixed": mixed, "imported": imported},
                "samples": samples.get(int(at), []),
                "qs": _bin_door(at, width, held),
                "spelled": _spell(at, bin_name),
                "finest": finest,
                "href": f"/g?{_bin_door(at, width, held)}" if finest else _window_url(held, at, at + width),
                "x": round(((at - lo) / span) * _W, 2),
                "w": round(bar_w, 2),
                "h": round(h, 2),
                "wall_h": round((wall / pictures) * h, 2) if pictures else 0,
            }
        )
    whole_span = max(1.0, whole_hi - whole_lo)
    overview_most = max([1, *(pictures for _, pictures, *_ in overview_bins)])
    overview = {
        "start": whole_lo,
        "end": whole_hi,
        "bin_seconds": overview_width,
        "bars": [
            {
                "at": at,
                "pictures": pictures,
                "spelled": _spell(at, "week"),
                "x": round(((at - whole_lo) / whole_span) * _W, 2),
                # a week of a decade-long library is a sliver; a burst must still read as a mark
                "w": round(max(3.0, (overview_width / whole_span) * _W), 2),
                "h": round(max(1.0, _height(pictures, overview_most, 36)), 2),
            }
            for at, pictures, *_ in overview_bins
        ],
        "brush": {
            "x": round(((lo - whole_lo) / whole_span) * _W, 2),
            "w": round(max(2.0, ((hi - lo) / whole_span) * _W), 2),
        },
    }
    presets = []
    for name, wide in PRESETS:
        p_start = whole_lo if wide is None else max(whole_lo, whole_hi - wide)
        current = (
            (lo <= whole_lo and hi >= whole_hi) if wide is None else (abs((hi - lo) - wide) < 1 and hi == whole_hi)
        )
        presets.append(
            {
                "name": name,
                "start": p_start,
                "end": whole_hi,
                "href": _window_url(held, p_start, whole_hi),
                "current": current,
            }
        )
    fine = sum(b["pictures"] for b in told_bins)
    coarse = sum(pictures for _, _, pictures in spans)
    unit = UNIT[bin_name]
    note = f"{fine:,} pictures · each bar is a {unit}"
    if coarse:
        note += f" · {coarse:,} are dated only to the day, shown as bands"
    if not every:
        note += f" · previews for the busiest {sum(1 for b in told_bins if b['samples'])} {unit}s"
    if not coverage["complete"]:
        note += f" · {coverage['present'] - coverage['interpreted']:,} pictures not dated yet"
    if coverage["present"] and not coverage["events_current"]:
        note += " · the groups are out of date"
    if pictures_total > len(drawn_rows):
        note += (
            f" · the first {len(drawn_rows):,} of {pictures_total:,} pictures drawn — show less time to see them all"
        )
    groups = _grouped(drawn_rows, sessions, told_bins, width, window_qs)
    composition = "river" if bin_name != CALENDAR_AT else ("calendar" if hi - lo <= SHEETS_WIDEST else "years")
    drawn = {g["session"]["id"]: g["pictures"] for g in groups if g["session"]}
    for one in sessions:
        # the members the window holds; a session touching the window
        # whose members all sit outside it (or past the cap) shows its samples
        one["drawn_pictures"] = drawn.get(one["id"], [])
    overview["years"] = [
        {
            "year": y,
            "x": round(((datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp() - whole_lo) / whole_span) * _W, 2),
        }
        for y in range(_utc(whole_lo).year + 1, _utc(whole_hi).year + 1)
    ]
    return {
        "composition": composition,
        "ticks": _ticks(lo, hi),
        "now_x": round(((time.time() - lo) / span) * _W, 2) if lo <= time.time() < hi else None,
        "pictures_total": pictures_total,
        "pictures_drawn": len(drawn_rows),
        "groups": groups,
        "river": _river(groups) if composition == "river" else [],
        "calendar": _calendar(conn, lo, hi, scope, held) if composition == "calendar" and not lean else [],
        "years": _years(told_bins, lo, hi, held) if composition == "years" else [],
        #: the cards the body lists on its own: every session under the
        #: sheets; in the river only those no day of it placed
        "listed": sessions if composition != "river" else [one for one in sessions if not one["drawn_pictures"]],
        "bin": bin_name,
        "bin_seconds": width,
        "start": lo,
        "end": hi,
        "start_spelled": _spell(lo, "hour"),
        "end_spelled": _spell(hi, "hour"),
        "window_spelled": _span(lo, hi),
        "unit": unit,
        "scope": scope_told,
        "extent": {"start": extent[0], "end": extent[1], "pictures": extent[2]},
        "overview": overview,
        "presets": presets,
        "coverage": coverage,
        #: every bin carries thumbnails; False when only the busiest do
        "sampled": every,
        "bins": told_bins,
        "spans": [
            {
                "start": s,
                "end": s + _SPAN[precision],
                "precision": precision,
                "pictures": pictures,
                "spelled": _spell(s, "day" if precision == "day" else "hour"),
                "x": round(((s - lo) / span) * _W, 2),
                "w": round(max(1.0, (_SPAN[precision] / span) * _W), 2),
            }
            for s, precision, pictures in spans
        ],
        "note": note,
        "sessions": sessions,
        "sessions_total": len(rows),
        "sessions_sampled": len(listed) <= SESSIONS_SAMPLED_MOST,
    }


@get("/timeline/density", sync_to_thread=True)
def density(
    state: State,
    bin_name: Annotated[str | None, QueryParameter(name="bin")] = None,
    start: FromQuery[float | None] = None,
    end: FromQuery[float | None] = None,
    folder: FromQuery[str | None] = None,
    album: FromQuery[str | None] = None,
    person: FromQuery[str | None] = None,
    artifact: FromQuery[str | None] = None,
    kind: FromQuery[str | None] = None,
    favorite: FromQuery[str | None] = None,
    rating_min: FromQuery[int | None] = None,
    f: FromQuery[list[str] | None] = None,
    lean: FromQuery[bool] = False,
) -> Response:
    """The surface as JSON (`_surface`): the same answer `/timeline`
    gives a machine, with `bin` as an explicit zoom when asked and the
    whole extent when no window is -- the machine's spelling."""
    asked = _question(folder, album, person, artifact, kind, favorite, rating_min, f)
    if bin_name is not None and bin_name not in pages.BINS:
        raise ClientException(f"no bin named {bin_name!r}; one of {', '.join(pages.BINS)}")
    conn = connect.connect(state.db_path, read_only=True)
    try:
        if start is None and end is None:
            extent = pages.timeline_extent(conn, _scope(conn, state, asked)[0])
            if extent is not None and extent[0] is not None:
                start, end = float(extent[0]), float(extent[1]) + 1.0
        told = _surface(conn, state, asked, start, end, bin_name=bin_name, lean=lean)
    finally:
        connect.close(conn)
    return Response(told, headers=VARIES)


@get("/timeline/pictures", sync_to_thread=True)
def pictures(
    state: State,
    start: FromQuery[float],
    end: FromQuery[float],
    folder: FromQuery[str | None] = None,
    album: FromQuery[str | None] = None,
    person: FromQuery[str | None] = None,
    artifact: FromQuery[str | None] = None,
    kind: FromQuery[str | None] = None,
    favorite: FromQuery[str | None] = None,
    rating_min: FromQuery[int | None] = None,
    f: FromQuery[list[str] | None] = None,
    limit: FromQuery[int] = PICTURES_MOST,
) -> Response:
    """Every picture of [start, end) in the scope, in moment order, each
    with its shape, its moment and precision, and the sessions it is in:
    what a surface needs to draw pictures ON time rather than beside
    it. Bounded by `limit` (at most PICTURES_MOST); `total` says how
    many the window holds."""
    if end <= start:
        raise ClientException("the range is empty")
    asked = _question(folder, album, person, artifact, kind, favorite, rating_min, f)
    conn = connect.connect(state.db_path, read_only=True)
    try:
        scope, held = _scope(conn, state, asked)
        rows, total = pages.timeline_pictures(conn, start, end, max(1, min(limit, PICTURES_MOST)), scope)
    finally:
        connect.close(conn)
    qs = resultset.canonical(dataclasses.replace(held, sort="moment"))
    return Response(
        {
            "start": start,
            "end": end,
            "scope": _scope_told(held),
            "qs": qs,
            "total": total,
            "pictures": [_picture(row, qs) for row in rows],
        },
        headers=VARIES,
    )


@get("/timeline", sync_to_thread=True)
def timeline(
    state: State,
    request: Request,
    start: FromQuery[float | None] = None,
    end: FromQuery[float | None] = None,
    bin_name: Annotated[str | None, QueryParameter(name="bin")] = None,
    folder: FromQuery[str | None] = None,
    album: FromQuery[str | None] = None,
    person: FromQuery[str | None] = None,
    artifact: FromQuery[str | None] = None,
    kind: FromQuery[str | None] = None,
    favorite: FromQuery[str | None] = None,
    rating_min: FromQuery[int | None] = None,
    f: FromQuery[list[str] | None] = None,
) -> Template | Response:
    """The timeline at one window: JSON to a machine, the surface fragment
    to htmx (what a brush move fetches), the page to a browser -- one
    builder, one renderer. `bin` is accepted from older doors and
    ignored: the zoom follows the window."""
    asked = _question(folder, album, person, artifact, kind, favorite, rating_min, f)
    conn = connect.connect(state.db_path, read_only=True)
    try:
        told = _surface(conn, state, asked, start, end)
    finally:
        connect.close(conn)
    return presented(request, told, page="timeline.html", fragment="_timeline_surface.html", name="surface")
