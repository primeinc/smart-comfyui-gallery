"""The timeline: a VIEW over the one interpretation, never a feature
database.

Months and days come from derived_media_context (the local wall clock
when one was claimed, the knowable instant otherwise) and the overlay
comes from the latest event runs. Every month, day, bin and session is
a LINK into the gallery -- spelled by the Facet Interface, so the
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
import urllib.parse
from typing import Annotated, Literal, NotRequired, TypedDict

from litestar import MediaType, Request, get
from litestar.datastructures import State
from litestar.exceptions import ClientException, NotFoundException
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import FromQuery, QueryParameter
from litestar.response import Redirect, Response, Template

from db import connect, context, facets, pages, planning, rendering, resultset, settings, when
from sg_web import home, projecting
from sg_web.asking import gallery_query as _asked
from sg_web.presenting import VARIES, wants_json
from sg_web.wire import Wire
from story_renderers import formatting
from vision import thumbs

#: The surface's scope is a gallery question (db/resultset.py scope_of):
#: its scopes and facets in the live spelling, unsorted, unpaged. The
#: timeline never invents its own spelling of a link -- every link is
#: that question plus the facets the link adds, ordered by moment.
WHOLE = resultset.GalleryQuery()


def _link(question: resultset.GalleryQuery, *held: facets.Facet) -> str:
    asked = dataclasses.replace(
        question, facets=tuple(sorted({*question.facets, *held}, key=facets.spell)), sort="moment", text=None
    )
    return resultset.canonical(asked)


def _bin_link(at: float, width: int, question: resultset.GalleryQuery = WHOLE) -> str:
    """The bar's pictures, exactly: the window, and the precision the
    count applied -- a day-precision claim sitting at midnight inside an
    hour's window was not counted in that bar and must not open from it."""
    low = facets.facet("context.moment", "gte", str(int(at)))
    high = facets.facet("context.moment", "lt", str(int(at) + width))
    fine = facets.facet("context.granule", "lte", str(width))
    return _link(question, low, high, fine)


def _event_link(event_id: int, question: resultset.GalleryQuery = WHOLE) -> str:
    return _link(question, facets.facet("event.id", "eq", str(event_id)))


def _question(folder, album, person, artifact, kind, favorite, rating_min, f) -> resultset.GalleryQuery:
    """The surface's filters as the gallery's own query, parsed by the
    one seam that owns query semantics; a bad filter is refused with
    the vocabulary. A session filter is not a timeline filter (the
    timeline shows every session of a range): `timeline` turns it into
    the session's range before this runs; the other routes refuse it."""
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
        raise ClientException("the timeline cannot filter by session; open the session's range instead")
    return asked


def _session_range(conn, f: list[str] | None) -> tuple[list[str], int, int] | None:
    """The session a filter list names, as the filters without it and
    the hour range that holds the session; None when no filter is a
    session. A session nobody has is a 404."""
    if not f:
        return None
    rest, named = [], None
    for one in f:
        held = facets.parse_spelling(one)
        if held.key == "event.id":
            named = int(held.value)
        else:
            rest.append(one)
    if named is None:
        return None
    span = pages.session_span(conn, named)
    if span is None:
        raise NotFoundException(f"no session {named}")
    return (rest, *pages.binned("hour", span[0], span[1]))


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


#: The comparison each scope states. Every one is equality except the
#: rating floor, which is the only scope that names a bound rather than a
#: value -- `rating from 4`, not `rating 4`.
_SCOPE_OPS = {"rating_min": "gte"}


def _scope_told(question: resultset.GalleryQuery) -> dict | None:
    """What the page says it is scoped to, or None for the whole library:
    the canonical spelling and its parts, one per scope and facet.

    Every part carries `spelled`. The template's fallback was
    `key ~ "=" ~ value`, which printed the query string on the page: a
    bool read `favorite=False`, and a facet -- which did carry a spelling
    -- spelled itself `tag:eq:beach`, the URL rather than the sentence.
    `facets.said` is the sentence.
    """
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


def _chips(conn, question: resultset.GalleryQuery) -> list[str]:
    """The same scope, as sentences for the page to print.

    NOT on the wire model, and that is the point. `spelled` there is the
    ADDRESS -- a facet spells itself `tag:eq:beach`, which is the URL --
    and the model is the machine's contract, asserted key-for-key by
    tests/test_a_picture_has_a_place.py and
    tests/test_the_timeline_is_the_way_in.py.

    The page needs the sentence: `kind image`, not `kind=image`, and
    `favorite yes`, not `favorite=True`. Same facts, rendered for a reader,
    built from the same question the model was built from -- so the two
    still cannot describe different surfaces.

    Takes the CONNECTION because an id-valued clause has to be resolved to
    a name while there is one: `place #11 (gone)` says the row was deleted
    about a place that is sitting in the table.
    """
    return facets.chips(conn, question, _SCOPE_OPS)


#: Which planner tells which kind of session's story; a kind with none
#: is offered no button and told why.
PLANNER_FOR = {
    "generation_session": "generation_history",
    "capture_session": "capture_history",
    "file_session": "file_history",
}

#: How wide a claim at each precision is: `db/when.py SPAN`, NOT a copy of it.
#: This is indexed by a precision read out of the database, and a lookup keyed
#: by a vocabulary somebody else owns has to be that vocabulary or it raises
#: `KeyError` on a precision the schema allows.
_SPAN = when.SPAN

#: Sessions one answer lists -- a whole library's extent can touch thousands,
#: so the page lists the most recent this many, says how many more there are,
#: and lets the person narrow the window. Never a silent cut, and every listed
#: session carries its thumbnails.
#: The number is the query's own bound (db/pages.py TIMELINE_EVENTS_MOST).
SESSIONS_MOST = pages.TIMELINE_EVENTS_MOST
SESSIONS_SAMPLED_MOST = SESSIONS_MOST
#: How much time a first visit shows: the last month that holds pictures,
#: clipped to the library -- never the whole library at once.
OPENING = 30 * when.DAY
#: The presets beside the window, each ending at the newest picture.
#: `1m` and `1y` are the mean month and year the bars themselves are
#: drawn from (db/pages.py BINS): they were 30 and 365 days, so the
#: preset labelled `1y` selected a window narrower than the year bin.
PRESETS = (("1w", 7 * when.DAY), ("1m", when.MONTH), ("3m", 3 * when.MONTH), ("1y", when.YEAR), ("all", None))
#: The zoom follows the window's width: enough bars to see the shape,
#: never more than the strip samples thumbnails for.
#: `when.YEAR` is the Gregorian mean (365.2425 days). 31_557_600 -- the
#: JULIAN year, 365.25 days -- was typed here twice, so the bars were
#: drawn against one length of year and the zoom boundaries decided by
#: another.
#: Half a year: the span above which the day zoom ends and the axis
#: labels months instead of days. It was `183 * 86_400` at both sites --
#: one number doing two jobs, derived from nothing.
MONTH_LABELS_ABOVE = when.YEAR / 2
_ZOOM = (
    ("minute", 6 * when.HOUR),
    ("quarter", 2 * when.DAY),
    ("hour", 14 * when.DAY),
    ("day", MONTH_LABELS_ABOVE),
    ("week", 12 * when.YEAR),
    ("month", 120 * when.YEAR),
)
#: The narrowest window the surface opens on.
NARROWEST = int(when.HOUR)
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
        #: generation and every session link goes dark until the events
        #: job runs again -- the page names that remedy, never an empty list
        "events_current": pages.timeline_events_current(conn),
        "contested_qs": _link(question, facets.facet("context.disputed", "eq", "1")),
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
        #: agree on (db/events.py _shared_place), with the gallery link
        "place": (
            {
                "id": place_id,
                "name": place_name,
                "slug": place_slug,
                "qs": _link(scope, facets.facet("place.id", "eq", str(place_id))),
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
        "qs": _event_link(event_id, scope),
        "planner": planner,
        "tellable": planner in planning.PLANNERS,
        "samples": _drawn(pages.session_samples(conn, event_id)) if samples else [],
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
    return "year"


def _day(d: datetime.datetime) -> str:
    return f"{d.day} {d.strftime('%b %Y')}"


def _spell(epoch: float, bin_name: str) -> str:
    """A moment as a person reads it: the day at the day and week zooms
    ("10 Jun 2023", "week of 5 Jun 2023"), the day and the clock below."""
    d = _utc(epoch)
    if bin_name in ("year", "month"):
        return f"{bin_name} from {_day(d)}"
    if bin_name == "week":
        return f"week of {_day(d)}"
    if bin_name == "day":
        return _day(d)
    return f"{_day(d)}, {d.strftime('%H:%M')}"


def _span(lo: float, hi: float) -> str:
    """A range as a person reads it: "22 Jul – 21 Aug 2026" across days,
    "21 Aug 2026, 01:33 – 01:39" within one."""
    a = _utc(lo)
    b = _utc(max(lo, hi - 1))
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
UNIT = {
    "year": "year",
    "month": "month",
    "week": "week",
    "day": "day",
    "hour": "hour",
    "quarter": "quarter hour",
    "minute": "minute",
}


def _drawn(pictures) -> list[str]:
    """The thumbnail addresses of `(slug, sha, kind)` rows, in order.

    Every strip, cell, frame and segment on this surface draws pictures,
    and until now each spelled `/thumb/<slug>` into its own markup -- a
    route with a lookup behind it, once per picture, on the page that
    draws the most of them. A content-addressed asset costs no
    connection at all (vision/thumbs.py `asset_url`).

    A file with no picture to take -- a sound, a document -- has no
    address and is DROPPED here rather than drawn as a broken image.
    That is a real difference from the grid, which keeps the cell and
    says the kind in it: a scrubber segment is a row of forty-pixel
    tiles with no room to say anything, and a strip is a sample of what
    is there rather than a claim to be all of it.
    """
    from vision import thumbs

    return [one for slug, sha, kind in pictures if (one := thumbs.asset_url(sha, slug, medium=kind)) is not None]


#: Pictures one surface draws in place. Past it the page says how many
#: more there were; the person narrows the window.
PICTURES_MOST = 2_000
#: At the week zoom the body is month sheets; finer, it is the river.
CALENDAR_AT = "week"
#: `story_renderers/formatting.py MONTHS`, whose docstring promises the
#: month spelling is decided once -- this module was the second chooser.
_MONTHS = formatting.MONTHS
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
#: Axis tick furniture: up to each window width (left), a tick every
#: step (right). In `db/when.py`'s units -- five rows of raw seconds
#: carried no derivation in a module whose other ladders all do.
_TICK_STEPS = (
    (when.HOUR, 5 * when.MINUTE),
    (6 * when.HOUR, 30 * when.MINUTE),
    (2 * when.DAY, 3 * when.HOUR),
    (14 * when.DAY, when.DAY),
    (float("inf"), 7 * when.DAY),
)


_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)


def _utc(epoch: float) -> datetime.datetime:
    """A moment as a UTC datetime for ANY year the calendar has: not
    `fromtimestamp`, which is the platform's gmtime and on Windows
    refuses everything before 1970 (Doc/library/datetime.rst) -- a
    scanned 1965 photograph is a moment like any other."""
    return _EPOCH + datetime.timedelta(seconds=epoch)


def _month_start(d: datetime.datetime) -> float:
    return datetime.datetime(d.year, d.month, 1, tzinfo=datetime.UTC).timestamp()


def _next_month(d: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(d.year + (d.month == 12), d.month % 12 + 1, 1, tzinfo=datetime.UTC)


def _ticks(lo: float, hi: float, axis: projecting.Projection | None = None) -> list[dict]:
    """The axis's furniture: a tick per step of the zoom with its label,
    major at the calendar boundary above it (midnight, the 1st, January).

    Placed THROUGH the projection, so a tick inside a run of collapsed
    time lands inside its band rather than where elapsed time would have
    put it -- furniture that disagreed with the bars would be worse than
    no furniture.
    """
    span = hi - lo
    held = axis or projecting.linear(lo, hi, float(_W))

    def x(t: float) -> float:
        return round(held.x(t), 2)

    out: list[dict] = []
    if span > MONTH_LABELS_ABOVE:
        # months up to three years; past that only the years, else the
        # labels pile into one another
        years_only = span > 3 * when.YEAR
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
        return _thin(out, TICK_LABEL_WIDE, "named")
    step = next(s for most, s in _TICK_STEPS if span <= most)
    t = -(-int(lo) // step) * step
    while t < hi:
        d = _utc(t)
        midnight = t % when.DAY == 0
        day_label = f"{d.day} {d.strftime('%b').upper()}"
        if step >= when.DAY:
            out.append({"x": x(t), "label": day_label, "major": d.day <= step // when.DAY})
        else:
            out.append({"x": x(t), "label": day_label if midnight else d.strftime("%H:%M"), "major": midnight})
        t += step
    return _thin(out, TICK_LABEL_WIDE, "named")


def _picture(row, qs: str) -> dict:
    slug, name, kind, width, height, moment, precision, origin, wall, sha, faces, sessions = row
    return {
        "slug": slug,
        #: Where to draw it. Content-addressed from the hash this row
        #: already carries, so a river of a thousand pictures costs no
        #: connections; None for a medium with no picture to take, and
        #: the cell says the kind instead.
        "thumb": thumbs.asset_url(sha, slug, medium=kind),
        "name": name,
        "kind": kind,
        "width": width,
        "height": height,
        "faces": int(faces or 0),
        "ratio": round(width / height, 4) if width and height else 1.0,
        "moment": moment,
        "precision": precision,
        "origin": origin,
        "domain": "wall" if wall else "instant",
        "sessions": [int(one) for one in sessions.split(",")] if sessions else [],
        "href": f"/i/{slug}?{qs}",
        "clock": _utc(moment).strftime("%H:%M"),
    }


class _Segment(TypedDict):
    """One band of the scrubber: a stretch of time and how it is drawn."""

    at: int
    end: int
    #: the link's window, clipped to the library, spelled as the URL spells it
    window_start: int
    year: int
    label: str
    pictures: int
    face: str | None
    strip: list[str]
    y: float
    h: float
    year_label: bool
    href: str


class _Cell(TypedDict):
    """One month of the year grid."""

    month: str
    pictures: int
    hero: dict | None
    qs: str | None
    href: str
    outside: bool


class _Bin(TypedDict):
    """One bar of the histogram."""

    at: int
    pictures: int
    wall: int
    instant: int
    origin: dict
    samples: list[str]
    qs: str
    spelled: str
    finest: bool
    href: str
    x: float
    w: float
    h: float
    wall_h: float


class _Group(TypedDict):
    """One row of the window: a listed session, or the pictures that fell
    in one bin.

    Spelled out because a dict literal holding an int, a dict, a None, a
    str and a list infers as the UNION of those for every key, so
    `group["pictures"].append(...)` is `.append` on `int | None | ...`
    and nothing downstream can be checked at all. The four keys added
    after it is built are `NotRequired` -- which is what they are.
    """

    t: int
    end: int
    session: dict | None
    bin: _Bin | None
    qs: str
    pictures: list[dict]
    clock: NotRequired[str]
    lasted: NotRequired[str]
    lead: NotRequired[dict]
    leads: NotRequired[list[dict]]


def _grouped(pictures: list[dict], sessions: list[dict], bins: list[_Bin], width: int, window_qs: str) -> list[_Group]:
    """The window's pictures in groups, oldest first: each listed session
    holds its members; the rest gather by the bin they fall in. A
    picture in two listed sessions is drawn in the first."""
    by_session: dict[int, _Group] = {
        s["id"]: _Group(t=s["start"], end=s["end"], session=s, bin=None, qs=s["qs"], pictures=[]) for s in sessions
    }
    by_bin = {b["at"]: b for b in bins}
    loose: dict[int, _Group] = {}
    for p in pictures:
        sid = next((one for one in p["sessions"] if one in by_session), None)
        if sid is not None:
            by_session[sid]["pictures"].append(p)
            continue
        at = int(p["moment"] // width) * width
        held = loose.get(at)
        if held is None:
            b = by_bin.get(at)
            held = loose[at] = _Group(
                t=at,
                end=at + width,
                session=None,
                bin=b,
                qs=b["qs"] if b else window_qs,
                pictures=[],
            )
        held["pictures"].append(p)
    groups = [g for g in (*by_session.values(), *loose.values()) if g["pictures"]]
    groups.sort(key=lambda g: -g["t"])
    for g in groups:
        g["pictures"].reverse()
        g["clock"] = f"{_utc(g['t']).strftime('%H:%M')}–{_utc(g['end']).strftime('%H:%M')}"
        g["lasted"] = _lasted(g["end"] - g["t"])
        g["lead"] = _lead(g["pictures"])
        g["leads"] = _leads(g["pictures"])
    return groups


def _lead(pictures: list[dict]) -> dict:
    """The picture a group is shown by: the one with the most faces, the
    largest among equals -- never simply the first, which in a burst is
    the frame before anyone was ready."""
    return max(pictures, key=lambda p: (p["faces"], (p["width"] or 0) * (p["height"] or 0)))


#: The hero row's width in units of its height: a card 2.8 times wider
#: than it is tall. One landscape fills it; portraits come two or three
#: across, each whole at its own ratio.
HERO_ROW = 2.8


def _leads(pictures: list[dict]) -> list[dict]:
    """The pictures a group's hero row is made of: the best first (faces,
    then size), taken until their widths at one height fill the row."""
    ranked = sorted(pictures, key=lambda p: (p["faces"], (p["width"] or 0) * (p["height"] or 0)), reverse=True)
    chosen: list[dict] = []
    filled = 0.0
    for p in ranked:
        if chosen and filled + p["ratio"] > HERO_ROW:
            break
        chosen.append(p)
        filled += p["ratio"]
        if filled >= HERO_ROW:
            break
    chosen.sort(key=lambda p: p["moment"])
    return chosen


def _lasted(seconds: float) -> str:
    if seconds < 90:
        return "a minute"
    if seconds < when.HOUR:
        return f"{round(seconds / when.MINUTE)} min"
    if seconds < 2 * when.DAY:
        return f"{seconds / when.HOUR:.1f} h"
    return f"{round(seconds / when.DAY)} days"


def _nothing_for(seconds: float) -> str:
    """How long a collapsed run was, in the unit a person would use.

    Not `_lasted`, which tops out at days because a SESSION lasting days
    is the longest a session gets. A gap is the other extreme: the run
    this exists for is twenty-two years, and "8203 days" is a number
    nobody converts.
    """
    days = seconds / when.DAY
    # Lengths in days, from `db/when.py` -- 30.44 and 365.25 were typed
    # here, and 365.25 is the Julian year, which is not the year the bars
    # beside this text are drawn against.
    for most, per, unit in (
        (2.0, when.HOUR / when.DAY, "hour"),
        (14.0, 1.0, "day"),
        (70.0, 7.0, "week"),
        (365.0, when.MONTH / when.DAY, "month"),
        (float("inf"), when.YEAR / when.DAY, "year"),
    ):
        if days < most:
            n = max(1, round(days / per))
            return f"{n} {unit}" if n == 1 else f"{n} {unit}s"
    return ""


def _river(groups: list[_Group]) -> list[dict]:
    """Days, newest first, each with its groups; a day carries the month
    cap when it opens one, and the silence above it -- the days of
    nothing between it and the newer day -- as the height the page draws
    them at."""
    days: list[dict] = []
    by_day: dict[str, list] = {}
    for g in groups:
        by_day.setdefault(_utc(g["t"]).strftime("%Y-%m-%d"), []).append(g)
    newer, prev_month = None, None
    for key in sorted(by_day, reverse=True):
        d = datetime.datetime.strptime(key, "%Y-%m-%d").replace(tzinfo=datetime.UTC)
        month = (d.year, d.month)
        silent = 0 if newer is None else (newer - d).days - 1
        days.append(
            {
                "key": key,
                "day": d.day,
                "title": f"{_WEEKDAYS[d.weekday()]}, {d.day} {d.strftime('%b %Y')}",
                "weekday": _WEEKDAYS[d.weekday()],
                "weekend": d.weekday() >= 5,
                "month_cap": {"year": d.year, "month": _MONTHS[d.month - 1]} if month != prev_month else None,
                "pictures": sum(len(g["pictures"]) for g in by_day[key]),
                "silent_days": max(0, silent),
                "silence": _silence(max(0, silent)),
                "groups": by_day[key],
            }
        )
        newer, prev_month = d, month
    return days


def _silence(days: int) -> int:
    """How tall a gap draws, in pixels: nothing for a day, then the log
    of the days -- a week is felt, a year is fallen through, a decade
    is not a wall."""
    import math

    return 0 if days < 1 else int(min(320, 28 + 52 * math.log2(days + 1)))


#: The most segments with pictures a scrubber carries: the unit is the
#: finest that keeps to this (_scrubber_unit).
SEGMENTS_MOST = 40
#: The scrubber's height in its own units; the least a segment with
#: pictures takes of it (a thumb can land on 2.5% of a screen); what a
#: run of empty bins takes, however long. A segment's pictures are not
#: counted here: the page asks /timeline/spread for as many as the
#: segment's own pixels can show, and /timeline/at for the one under
#: the pointer.
_H = 1000
SEGMENT_LEAST = 25.0
GAP_H = 10.0
#: The most pictures one spread answers.
SPREAD_MOST = 400


def _scrubber_unit(conn, scope, whole_lo: float, whole_hi: float) -> tuple[str, int, list]:
    """The unit the library's own spread earns: the finest bin at which
    the extent holds at most SEGMENTS_MOST bins with pictures and can be
    drawn at all (db/pages.py MAX_BINS). Five minutes of pictures scrub
    by the minute, five centuries by the year, and nothing here presumes
    which. (name, width, bins)."""
    chosen = None
    for name in ("minute", "quarter", "hour", "day", "week", "month", "year"):
        try:
            width, bins, _ = pages.timeline_density(conn, name, whole_lo, whole_hi, scope)
        except ValueError:
            continue
        chosen = (name, width, bins)
        if sum(1 for b in bins if b[1]) <= SEGMENTS_MOST:
            break
    if chosen is None:
        raise ClientException("the library spans more than the page can draw")
    return chosen


def _scrubber(conn, scope, question, whole_lo: float, whole_hi: float, lo: float, hi: float, *, lean: bool) -> dict:
    """The library top to bottom, newest first, in the unit its spread
    earns (_scrubber_unit). A segment per bin that holds pictures, its
    height its share of the pictures and never less than a thumb can
    grab, carrying a strip of its pictures -- as many as that height
    shows -- and its count; a run of empty bins is ONE short segment
    saying how long it was, not a hairline each. The year is named where
    it changes, far enough from the last name; the window is marked
    across the segments it touches. Every segment is a link to its own
    window. Strips come from the same bins the counts do."""
    name, width, bins = _scrubber_unit(conn, scope, whole_lo, whole_hi)
    faces = {} if lean else pages.timeline_samples(conn, name, whole_lo, whole_hi, None, scope)
    counts = {int(at): int(n) for at, n, *_ in bins}
    anchor = pages._ANCHOR.get(name, 0)
    first = int((whole_lo - anchor) // width) * width + anchor
    told: list[dict] = []
    at = first
    # bins up to the one holding the last picture (whole_hi is that
    # picture's moment + 1): a bin past it would be a gap after the
    # library, its range clipped to nothing
    while at <= whole_hi - 1.0:
        told.append({"at": at, "end": at + width, "pictures": counts.get(at, 0), "units": 1})
        at += width
    told.reverse()
    # bins with pictures stand alone; empty bins run together
    runs: list[dict] = []
    for u in told:
        if u["pictures"] or not runs or runs[-1]["pictures"]:
            runs.append(dict(u))
        else:
            runs[-1]["at"] = u["at"]
            runs[-1]["units"] += 1
    total = max(1, sum(u["pictures"] for u in runs))
    gaps = sum(1 for u in runs if not u["pictures"])
    held = len(runs) - gaps
    least = min(SEGMENT_LEAST, _H / max(1, len(runs)))
    gap_h = min(GAP_H, least)
    room = max(0.0, _H - least * held - gap_h * gaps)
    y = 0.0
    segments: list[_Segment] = []
    last_year = None
    labelled_at = -1e9
    unit = UNIT[name]
    for u in runs:
        at, end, n = u["at"], u["end"], u["pictures"]
        h = gap_h if not n else least + (n / total) * room
        year = _utc(at).year
        named = n > 0 and year != last_year and (y + h) - labelled_at >= 18
        if named:
            labelled_at = y + h
        strip = _drawn(faces.get(int(at)) or [])
        label = _spell(at, name) if n else f"{u['units']} {unit}{'s' if u['units'] > 1 else ''} without pictures"
        segment: _Segment = {
            "at": at,
            "end": end,
            #: the link's window, clipped to the library, spelled as the URL spells it
            "window_start": int(max(whole_lo, at)),
            "year": year,
            "label": label,
            "pictures": n,
            "face": strip[0] if strip else None,
            "strip": strip,
            "y": round(y, 2),
            "h": round(h, 2),
            "year_label": named,
            "href": _window_url(question, max(whole_lo, at), min(whole_hi, end)),
        }
        segments.append(segment)
        if n:
            last_year = year
        y += h

    def y_of(t: float, at_end: bool) -> float:
        for seg in segments:
            if seg["at"] <= t < seg["end"]:
                f = (t - seg["at"]) / max(1.0, seg["end"] - seg["at"])
                return seg["y"] + seg["h"] * (1.0 - f)
        return 0.0 if at_end else float(_H)

    top = y_of(min(hi, whole_hi - 1), True)
    bottom = y_of(max(lo, whole_lo), False)
    return {
        "unit": name,
        "bin_seconds": width,
        "segments": segments,
        "brush": {"y": round(min(top, bottom), 2), "h": round(max(4.0, abs(bottom - top)), 2)},
    }


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
                    "hero": next(iter(_drawn(samples.get(int(t)) or [])), None),
                    "qs": _bin_link(t, pages.BINS["day"], question) if n else None,
                    "spelled": _spell(t, "day"),
                    "today": _utc(t).strftime("%Y-%m-%d") == today,
                }
            )
            t += pages.BINS["day"]
        months.append(
            {"year": d.year, "month": _MONTHS[d.month - 1], "lead": d.weekday(), "pictures": total, "days": days}
        )
        d = nxt
    months.reverse()
    return months


#: Room a label needs, in the 1000 units both axes are drawn in. They
#: render around 1150px wide, so a unit is about 1.15px. A year measures
#: 25px and a clock time 36px.
YEAR_LABEL_WIDE = 28.0
TICK_LABEL_WIDE = 34.0


def _thin(marks: list[dict], apart: float, into: str) -> list[dict]:
    """Set `into` on the marks with room to be named.

    Both axes collapse runs that hold nothing, so marks chosen by elapsed
    time can land a couple of pixels apart: a library with a gap between
    2013 and 2024 packs eleven years into twenty pixels, and an hour tick
    either side of a collapsed run does the same. Every mark keeps its
    line — only the ones far enough from the last name get named, or they
    are drawn on top of each other and none is legible.

    Walked newest first, so the end holding the pictures is always named.
    """
    last: float | None = None
    for one in reversed(marks):
        one[into] = last is None or last - one["x"] >= apart
        if one[into]:
            last = one["x"]
    return marks


#: The widest window that draws month sheets; wider, the body is years.
#: In the same year the bins are drawn from -- it was a hand-typed leap
#: year, `2 * 366 * 86_400`.
SHEETS_WIDEST = 2 * when.YEAR


def _years(told_bins: list[_Bin], lo: float, hi: float, question: resultset.GalleryQuery) -> list[dict]:
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
        cells: list[_Cell] = []
        for m in range(1, 13):
            start = datetime.datetime(y, m, 1, tzinfo=datetime.UTC)
            end = _next_month(start)
            held = months.get((y, m), {"pictures": 0, "hero": None})
            link = _link(
                question,
                facets.facet("context.moment", "gte", str(int(start.timestamp()))),
                facets.facet("context.moment", "lt", str(int(end.timestamp()))),
            )
            cell: _Cell = {
                "month": _MONTHS[m - 1][:3].upper(),
                "pictures": held["pictures"],
                "hero": held["hero"],
                "qs": link if held["pictures"] else None,
                "href": _window_url(question, start.timestamp(), end.timestamp()),
                "outside": end.timestamp() <= lo or start.timestamp() >= hi,
            }
            cells.append(cell)
        years.append({"year": y, "pictures": sum(c["pictures"] for c in cells), "months": cells})
    years.reverse()
    return years


def _window_url(question: resultset.GalleryQuery, start: float, end: float) -> str:
    qs = resultset.canonical(question)
    return f"/timeline?{qs + '&' if qs else ''}start={int(start)}&end={int(end)}"


def _surface(
    conn, state: State, asked: resultset.GalleryQuery, start, end, *, bin_name=None, lean=False, snap=False
) -> dict:
    """The surface at one window: pictures per bin of the human moment
    over [start, end) -- the last month that holds pictures when no
    window is asked -- split by clock domain and by origin, with a
    thumbnail sample per bin (the busiest past SAMPLED_BINS_MOST); the
    claims too coarse for the bin as spans; the sessions touching the
    window in their own domain, each a link to its pictures and to its
    story; the whole extent at week resolution as the overview the
    brush rides. The zoom follows the window's width unless `bin_name`
    asks for one. Every bin is a link into the gallery, carrying the
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
            "segments": [],
            "skipped": [],
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
            "scrubber": None,
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
        if snap:
            # a hand on the scrubber lands in time, not on pictures: carry
            # the window back to the newest picture at or before its end,
            # keeping its width -- a window is never empty by a hair
            last = pages.timeline_last_before(conn, hi, scope)
            if last is not None and last + 1.0 < hi and pages.timeline_span(conn, lo, hi, scope)[0] is None:
                width_asked = hi - lo
                hi = min(whole_hi, last + 1.0)
                lo = max(whole_lo, hi - width_asked)
                # the width survives the library's start: a window carried
                # onto the first picture is still as wide as the hand drew it
                hi = min(whole_hi, lo + width_asked)
    bin_name = bin_name or _bin_for(hi - lo)
    try:
        width, bins, spans = pages.timeline_density(conn, bin_name, lo, hi, scope)
    except ValueError as refused:
        raise ClientException(str(refused)) from refused
    # the overview at the finest bin the whole extent can be drawn at
    for overview_bin in ("week", "month", "year"):
        try:
            overview_width, overview_bins, _ = pages.timeline_density(conn, overview_bin, whole_lo, whole_hi, scope)
            break
        except ValueError:
            continue
    else:
        raise ClientException("the library spans more than the page can draw")
    every = len(bins) <= pages.SAMPLED_BINS_MOST
    busiest = None if every else [at for at, *_ in sorted(bins, key=lambda b: -b[1])[: pages.SAMPLED_BINS_MOST]]
    samples = {} if lean else pages.timeline_samples(conn, bin_name, lo, hi, busiest, scope)
    rows = [] if lean else pages.timeline_sessions(conn, lo, hi, scope)
    # rows are oldest first (db/pages.py _TIMELINE_SESSIONS_TAIL); the tail is the latest
    listed = rows[-SESSIONS_MOST:] if len(rows) > SESSIONS_MOST else rows
    sessions = [_session(conn, row, samples=len(listed) <= SESSIONS_SAMPLED_MOST, scope=held) for row in listed]
    # The axis. Built from the bins that came back non-empty and the
    # spans too coarse to have landed in one -- everything that holds a
    # picture -- so the runs holding nothing can stop taking the page.
    axis = projecting.projected(
        lo,
        hi,
        [(float(at), float(at) + width) for at, *_ in bins]
        # a claim too coarse for the bin still HOLDS its granule: a
        # picture known only to the day occupies that day, and a gap
        # invented across it would collapse content
        + [(float(s), float(s) + _SPAN[precision]) for s, precision, _ in spans],
        width=float(_W),
    )
    for one in sessions:
        one["when"] = _span(one["start"], one["end"] + 1)
        one["happened"] = HAPPENED.get(one["kind"], one["kind"].replace("_", " "))
        one["title"] = one["story"]["title"] if one["story"] else f"{one['pictures']:,} {one['happened']}"
        named = [p for p in one["people"] if p["name"]]
        others = one["people_total"] - len(named)
        one["company"] = {"named": named, "others": others}
        one["lasted"] = _lasted(one["end"] - one["start"])
        # the session's frame on the axis, clipped to the window
        x0 = axis.x(one["start"])
        x1 = axis.x(one["end"] + 1)
        one["x"], one["w"] = round(x0, 2), round(max(2.0, x1 - x0), 2)
    window_qs = _link(
        held, facets.facet("context.moment", "gte", str(int(lo))), facets.facet("context.moment", "lt", str(int(hi)))
    )
    picture_rows, pictures_total = ([], 0) if lean else pages.timeline_pictures(conn, lo, hi, PICTURES_MOST, scope)
    drawn_rows = [_picture(row, window_qs) for row in picture_rows]
    most = max([1, *(pictures for _, pictures, *_ in bins)])
    finest = bin_name == "minute"
    told_bins: list[_Bin] = []
    for at, pictures, wall, instant, *by_origin in bins:
        h = _height(pictures, most, 100)
        told: _Bin = {
            "at": at,
            "pictures": pictures,
            "wall": wall,
            "instant": instant,
            # the counting columns come back in ORIGINS order because the
            # statement is BUILT over the same tuple (db/pages.py
            # _TIMELINE_DENSITY_HEAD); this dict retyped all four members
            "origin": dict(zip(context.ORIGINS, by_origin, strict=True)),
            "samples": _drawn(samples.get(int(at), [])),
            "qs": _bin_link(at, width, held),
            "spelled": _spell(at, bin_name),
            "finest": finest,
            "href": f"/g?{_bin_link(at, width, held)}" if finest else _window_url(held, at, at + width),
            "x": round(axis.x(at), 2),
            "w": round(max(1.0, axis.x(at + width) - axis.x(at) - 0.5), 2),
            "h": round(h, 2),
            "wall_h": round((wall / pictures) * h, 2) if pictures else 0,
        }
        told_bins.append(told)
    overview_most = max([1, *(pictures for _, pictures, *_ in overview_bins)])
    # The scrubber's axis, collapsed on the same rule. This is the
    # twenty-four-year case: a library holding one scanned photograph
    # from 2002 spends nine tenths of its navigation control on years
    # that hold nothing, and the handle for THIS month ends a hairline
    # nobody can take hold of.
    whole_axis = projecting.projected(
        whole_lo,
        whole_hi,
        [(float(at), float(at) + overview_width) for at, *_ in overview_bins],
        width=float(_W),
    )
    overview = {
        "start": whole_lo,
        "end": whole_hi,
        "segments": whole_axis.told(),
        "skipped": [
            {
                "x": round(one.x0, 2),
                "w": round(one.x1 - one.x0, 2),
                "lasted": _nothing_for(one.seconds),
                "start": one.t0,
                "end": one.t1,
            }
            for one in whole_axis.collapsed
        ],
        "bin_seconds": overview_width,
        "bars": [
            {
                "at": at,
                "pictures": pictures,
                "spelled": _spell(at, "week"),
                "x": round(whole_axis.x(at), 2),
                # a week of a decade-long library is a sliver; a burst must still read as a mark
                "w": round(max(3.0, whole_axis.x(at + overview_width) - whole_axis.x(at)), 2),
                "h": round(max(1.0, _height(pictures, overview_most, 36)), 2),
            }
            for at, pictures, *_ in overview_bins
        ],
        "brush": {
            "x": round(whole_axis.x(lo), 2),
            "w": round(max(2.0, whole_axis.x(hi) - whole_axis.x(lo)), 2),
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
    # The Y SCALE, which the surface drew and never stated: "each bar is
    # a day" names the x unit and nothing named the other one, so a bar
    # could be five pictures or five hundred and the only way to find out
    # was to hover it.
    note = f"{fine:,} pictures · each bar is a {unit} · tallest {most:,}"
    if axis.collapsed:
        # Said in the sentence as well as drawn as a band: somebody
        # reading the note is owed the fact that the axis is not to
        # scale, not only somebody looking at the picture of it.
        note += (
            f" · {len(axis.collapsed)} empty "
            f"{'run' if len(axis.collapsed) == 1 else 'runs'} collapsed "
            f"({', '.join(_nothing_for(one.seconds) for one in axis.collapsed)})"
        )
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
        one["drawn_lead"] = _lead(one["drawn_pictures"]) if one["drawn_pictures"] else None
        one["drawn_leads"] = _leads(one["drawn_pictures"]) if one["drawn_pictures"] else []
    overview["years"] = _thin(
        [
            {
                "year": y,
                "x": round(whole_axis.x(datetime.datetime(y, 1, 1, tzinfo=datetime.UTC).timestamp()), 2),
            }
            for y in range(_utc(whole_lo).year + 1, _utc(whole_hi).year + 1)
        ],
        YEAR_LABEL_WIDE,
        "label",
    )
    return {
        "composition": composition,
        "ticks": _ticks(lo, hi, axis),
        "now_x": round(axis.x(time.time()), 2) if lo <= time.time() < hi else None,
        #: The window axis as the browser must invert it: a click, a drag
        #: and a pan all turn x back into a moment, and a piecewise axis
        #: the server kept to itself would put every gesture in the wrong
        #: year (frontend/src/timeline.ts).
        "segments": axis.told(),
        #: The runs that hold nothing and are drawn as a band saying so.
        #: Blank pixels are ambiguous between "no pictures", "nothing
        #: dated" and "the render broke"; a label is not.
        "skipped": [
            {
                "x": round(one.x0, 2),
                "w": round(one.x1 - one.x0, 2),
                "lasted": _nothing_for(one.seconds),
                "start": one.t0,
                "end": one.t1,
                "href": _window_url(held, one.t0, one.t1),
            }
            for one in axis.collapsed
        ],
        "pictures_total": pictures_total,
        "pictures_drawn": len(drawn_rows),
        "groups": groups,
        "river": _river(groups) if composition == "river" else [],
        "calendar": _calendar(conn, lo, hi, scope, held) if composition == "calendar" and not lean else [],
        "years": _years(told_bins, lo, hi, held) if composition == "years" else [],
        #: the cards the body lists on its own: every session under the
        #: sheets; in the river only those no day of it placed
        "listed": sessions if composition != "river" else [one for one in sessions if not one["drawn_pictures"]],
        "scrubber": _scrubber(conn, scope, held, whole_lo, whole_hi, lo, hi, lean=lean),
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
                "x": round(axis.x(s), 2),
                "w": round(max(1.0, axis.x(s + _SPAN[precision]) - axis.x(s)), 2),
            }
            for s, precision, pictures in spans
        ],
        "note": note,
        "sessions": sessions,
        "sessions_total": len(rows),
        "sessions_sampled": len(listed) <= SESSIONS_SAMPLED_MOST,
    }


# --- the timeline's contract -------------------------------------------------
#
# These do not restate `_surface`; they DESCRIBE it, and the description is
# executable. `TimelineSurface(**told)` validates the whole tree at the seam:
# `extra="forbid"` means a key the builder grows and this does not name is a
# refusal here rather than an undescribed field on the wire, and strict mode
# means a count that turned into a string, or an int where the contract
# promised a boolean, is caught in the same call. A dict is accepted for a
# model field and a list of dicts for a list of models, so the builder goes on
# building dicts and this stays the one place that says what they are.
#
# Both what a person reads (`spelled`, `label`, `title`, `note`) and what a
# browser draws with (`x`, `w`, `h`, `y`) are here on purpose: the server
# decides how the surface reads AND how it is laid out, and the browser puts
# that on screen rather than recomputing it.


class TimelineScopePart(Wire):
    """One scope or facet the surface is narrowed by, said out loud."""

    key: str
    value: object
    #: only a facet is spelled; a scope's value is its own address
    spelled: str | None = None


class TimelineScope(Wire):
    """What the page says it is scoped to. Null for the whole library."""

    qs: str
    parts: list[TimelineScopePart]


class TimelineCoverage(Wire):
    """How much of the library the timeline can actually place, and the
    remedy when that is not all of it -- never an empty surface with no
    reason given."""

    interpreted: int
    present: int
    contested: int
    #: a session run answers only at the current interpretation; false
    #: means the groups are stale and the page says so
    events_current: bool
    contested_qs: str
    policy_version: int
    complete: bool


class TimelinePicture(Wire):
    """One picture ON the axis: its shape, its moment and how precisely
    that is known, which clock domain it is in, and the sessions holding
    it."""

    slug: str
    name: str
    kind: str
    #: where to draw it: content-addressed, so a river of a thousand
    #: pictures costs no connections. None for a medium with no picture
    #: to take, and the cell says the kind instead.
    thumb: str | None
    width: int | None
    height: int | None
    faces: int
    ratio: float
    moment: float
    precision: str
    origin: str | None
    domain: Literal["wall", "instant"]
    sessions: list[int]
    href: str
    clock: str


class TimelineHero(Wire):
    """A picture a story names, addressed as the story page addresses it.
    `thumbnail` is null once the file has left the library."""

    name: str
    thumbnail: str | None


class TimelineStoryCard(Wire):
    """What a card says of one story. Null on a session whose render no
    longer verifies: a card shows nothing it cannot prove."""

    id: int
    href: str
    evolution: str
    title: str
    dek: str
    heroes: list[TimelineHero]


class TimelinePlace(Wire):
    """Where a session happened: the one place its placed members agree
    on, with its gallery link."""

    id: int
    name: str | None
    slug: str | None
    qs: str


class TimelinePerson(Wire):
    """One person in a session, and how many of its pictures hold them."""

    slug: str
    name: str | None
    href: str
    pictures: int


class TimelineCompany(Wire):
    """Who was there: those with names, and how many more there were."""

    named: list[TimelinePerson]
    others: int


class TimelineSession(Wire):
    """One session touching the window, in its own clock domain.

    `pictures` is how many it holds and `in_scope` how many of those the
    surface's scope keeps, so a scoped page never claims the whole
    session. `x` and `w` are its frame on the axis, clipped to the window.
    """

    id: int
    kind: str
    domain: Literal["wall", "instant"]
    start: float
    end: float
    pictures: int
    in_scope: int
    snapshot_id: int | None
    story: TimelineStoryCard | None
    place: TimelinePlace | None
    qs: str
    #: which planner tells this kind of session's story, and whether the
    #: application actually has it
    planner: str | None
    tellable: bool
    #: THUMBNAIL ADDRESSES, not slugs. Content-addressed, so drawing
    #: them costs no database connection; a file with no picture to take
    #: is absent rather than a broken image (`_drawn`).
    samples: list[str]
    people: list[TimelinePerson]
    people_total: int
    when: str
    happened: str
    title: str
    lasted: str
    company: TimelineCompany
    x: float
    w: float
    #: the members the window holds; a session whose members all sit
    #: outside it shows its samples instead
    drawn_pictures: list[TimelinePicture]
    drawn_lead: TimelinePicture | None
    drawn_leads: list[TimelinePicture]


class TimelineTick(Wire):
    """One step of the axis's furniture, major at the calendar boundary
    above it (midnight, the 1st, January)."""

    x: float
    label: str
    major: bool
    #: Whether to print the label. The axis collapses runs holding
    #: nothing, so ticks either side of one land a few pixels apart; every
    #: tick keeps its line, only those with room are named.
    named: bool


class TimelineOrigin(Wire):
    """A bin's pictures by where they came from."""

    captured: int
    generated: int
    mixed: int
    imported: int


class TimelineBin(Wire):
    """One bar: its pictures split by clock domain and by origin, a
    thumbnail sample, and the link into the gallery it carries."""

    at: float
    pictures: int
    #: of those, how many are claimed on a wall clock rather than an
    #: instant -- `wall_h` is that share of the bar's height
    wall: int
    instant: int
    origin: TimelineOrigin
    #: THUMBNAIL ADDRESSES, not slugs. Content-addressed, so drawing
    #: them costs no database connection; a file with no picture to take
    #: is absent rather than a broken image (`_drawn`).
    samples: list[str]
    qs: str
    spelled: str
    #: at the finest zoom a bar IS its pictures, so it links to them;
    #: otherwise it links to its own narrower window
    finest: bool
    href: str
    x: float
    w: float
    h: float
    wall_h: float


class TimelineSpan(Wire):
    """A claim too coarse for the bin, drawn as a band over the time it
    could be rather than a bar at a moment it is not."""

    start: float
    end: float
    precision: str
    pictures: int
    spelled: str
    x: float
    w: float


class TimelineGroup(Wire):
    """The window's pictures in one group: a listed session's members, or
    the pictures of one bin. `lead` is the picture it is shown by -- the
    one with the most faces, never simply the first, which in a burst is
    the frame before anyone was ready."""

    t: float
    end: float
    session: TimelineSession | None
    bin: TimelineBin | None
    qs: str
    pictures: list[TimelinePicture]
    clock: str
    lasted: str
    lead: TimelinePicture
    leads: list[TimelinePicture]


class TimelineMonthCap(Wire):
    """The month a day opens, named once at the top of it."""

    year: int
    month: str


class TimelineDay(Wire):
    """One day of the river, with the silence above it: the days of
    nothing between it and the newer day, and the height that draws at."""

    key: str
    day: int
    title: str
    weekday: str
    weekend: bool
    month_cap: TimelineMonthCap | None
    pictures: int
    silent_days: int
    silence: int
    groups: list[TimelineGroup]


class TimelineCalendarDay(Wire):
    """One cell of a month sheet. `qs` is null on a day holding nothing:
    an empty day is not a link to an empty gallery."""

    n: int
    pictures: int
    #: THUMBNAIL ADDRESSES, not slugs. Content-addressed, so drawing
    #: them costs no database connection; a file with no picture to take
    #: is absent rather than a broken image (`_drawn`).
    hero: str | None
    qs: str | None
    spelled: str
    today: bool


class TimelineMonthSheet(Wire):
    """One month sheet. `lead` is the weekday the 1st falls on, so the
    cells start in the right column."""

    year: int
    month: str
    lead: int
    pictures: int
    days: list[TimelineCalendarDay]


class TimelineYearCell(Wire):
    """One month of a year row. `outside` marks a month the window does
    not reach: drawn, so the year keeps its shape, but not lit."""

    month: str
    pictures: int
    #: THUMBNAIL ADDRESSES, not slugs. Content-addressed, so drawing
    #: them costs no database connection; a file with no picture to take
    #: is absent rather than a broken image (`_drawn`).
    hero: str | None
    qs: str | None
    href: str
    outside: bool


class TimelineYearRow(Wire):
    """One year, twelve cells."""

    year: int
    pictures: int
    months: list[TimelineYearCell]


class AxisSegment(Wire):
    """One run of a horizontal axis: `[t0, t1)` drawn from `x0` to `x1`.

    The browser turns a click, a drag and a pan back into a moment, so a
    piecewise axis has to travel or every gesture lands in the wrong
    year. `skipped` is a run that holds nothing and was collapsed --
    which the vertical rail beside this has always done ("empty bins run
    together", `_scrubber`) and the horizontal surfaces never did.
    """

    t0: float
    t1: float
    x0: float
    x1: float
    skipped: bool


class AxisSkipped(Wire):
    """A collapsed run, as the band that replaces it says it.

    Blank pixels are ambiguous between "no pictures", "nothing dated
    this" and "the render broke". `lasted` is the sentence that is not.
    """

    x: float
    w: float
    lasted: str
    start: float
    end: float
    href: str | None = None


class TimelineOverviewBar(Wire):
    """One week of the whole extent. A week of a decade-long library is a
    sliver, so a burst is drawn at a floor width rather than vanishing."""

    at: float
    pictures: int
    spelled: str
    x: float
    w: float
    h: float


class TimelineBrush(Wire):
    """The window's frame on the overview."""

    x: float
    w: float


class TimelineOverviewYear(Wire):
    """Where a year begins along the overview strip, so a decade of
    library reads as years rather than as one undivided smear."""

    year: int
    x: float
    #: Whether to draw the year's number. The axis collapses runs holding
    #: nothing, so years can land a couple of pixels apart; every one
    #: keeps its line, only those with room are named.
    label: bool


class TimelineOverview(Wire):
    """The whole extent at week resolution -- the strip the brush rides."""

    start: float
    end: float
    bin_seconds: int
    #: the strip's own axis, collapsed on the same rule as the window's
    segments: list[AxisSegment]
    skipped: list[AxisSkipped]
    bars: list[TimelineOverviewBar]
    brush: TimelineBrush
    years: list[TimelineOverviewYear]


class TimelineSegment(Wire):
    """One band of the scrubber: a bin that holds pictures, or a run of
    empty ones as ONE short band saying how long it was.

    A band's height is its share of the library's pictures and never less
    than a thumb can grab. `strip` is as many of its pictures as that
    height shows; the picture under the pointer comes from /timeline/nth,
    by rank, so a burst spreads over the whole band.
    """

    at: float
    end: float
    #: the link's window, clipped to the library
    window_start: int
    year: int
    label: str
    pictures: int
    #: THUMBNAIL ADDRESSES, not slugs. Content-addressed, so drawing
    #: them costs no database connection; a file with no picture to take
    #: is absent rather than a broken image (`_drawn`).
    face: str | None
    strip: list[str]
    y: float
    h: float
    #: the year is named where it changes, far enough from the last name
    year_label: bool
    href: str


class TimelineScrubberBrush(Wire):
    """The window's mark down the scrubber."""

    y: float
    h: float


class TimelineScrubber(Wire):
    """The library top to bottom, newest first, in the unit its own
    spread earns: five minutes of pictures scrub by the minute, five
    centuries by the year."""

    unit: str
    bin_seconds: int
    segments: list[TimelineSegment]
    brush: TimelineScrubberBrush


class TimelineExtent(Wire):
    """Everything the scope holds, however far outside the window."""

    start: float
    end: float
    pictures: int


class TimelinePreset(Wire):
    """One window offered beside the surface, each ending at the newest
    picture."""

    name: str
    start: float
    end: float
    href: str
    current: bool


class TimelineSurface(Wire):
    """The timeline at one window.

    Null `start`, `end`, `extent`, `overview` and `scrubber` together mean
    the scope holds nothing placeable -- and `coverage` then says why and
    what to run about it, which is the whole reason an empty surface is
    still an answer.
    """

    #: the zoom, and how wide one bar is in seconds
    bin: str
    bin_seconds: int | None
    start: float | None
    end: float | None
    start_spelled: str
    end_spelled: str
    window_spelled: str
    #: the bar's unit as a person says it
    unit: str
    scope: TimelineScope | None
    extent: TimelineExtent | None
    overview: TimelineOverview | None
    presets: list[TimelinePreset]
    coverage: TimelineCoverage
    #: every bin carries thumbnails; false when only the busiest do
    sampled: bool
    bins: list[TimelineBin]
    spans: list[TimelineSpan]
    #: the window's axis, so the browser inverts x through the same
    #: piecewise function the bars were drawn with
    segments: list[AxisSegment]
    #: the runs it collapsed, each drawn as a band that says how long
    skipped: list[AxisSkipped]
    #: what the surface says of itself, in one line
    note: str
    sessions: list[TimelineSession]
    sessions_total: int
    sessions_sampled: bool
    #: which body the window earns: the river of days, month sheets, or
    #: year rows
    composition: Literal["river", "calendar", "years"]
    ticks: list[TimelineTick]
    now_x: float | None
    pictures_total: int
    pictures_drawn: int
    groups: list[TimelineGroup]
    river: list[TimelineDay]
    calendar: list[TimelineMonthSheet]
    years: list[TimelineYearRow]
    #: the cards the body lists on its own: every session under the
    #: sheets; in the river only those no day of it placed
    listed: list[TimelineSession]
    scrubber: TimelineScrubber | None


class TimelineMoment(Wire):
    """A picture and when it is: the smallest thing the timeline answers."""

    slug: str
    moment: float
    #: where to draw it: content-addressed, so a strip of forty costs no
    #: connection. None when the medium has no picture to take.
    thumb: str | None


class TimelineSpread(Wire):
    """Pictures of a range, spread evenly through it."""

    pictures: list[TimelineMoment]


class TimelineAt(Wire):
    """The picture a hand over a moment is pointing at."""

    slug: str
    moment: float
    spelled: str


class TimelineNth(Wire):
    """The k-th picture of a range in moment order, of the `of` it holds
    -- what a hand k/n of the way along a segment points at."""

    slug: str
    moment: float
    #: where to draw it: content-addressed, so a strip of forty costs no
    #: connection. None when the medium has no picture to take.
    thumb: str | None
    k: int
    of: int
    spelled: str


class TimelinePictures(Wire):
    """Every picture of a window in moment order, bounded by the request's
    limit. `total` says how many the window holds, so a page that drew
    fewer knows it and can say so."""

    start: float
    end: float
    scope: TimelineScope | None
    qs: str
    total: int
    pictures: list[TimelinePicture]


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
) -> Response[TimelineSurface]:
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
    return Response(TimelineSurface.model_validate(told), headers=VARIES)


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
) -> Response[TimelinePictures]:
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
        TimelinePictures.model_validate(
            {
                "start": start,
                "end": end,
                "scope": _scope_told(held),
                "qs": qs,
                "total": total,
                "pictures": [_picture(row, qs) for row in rows],
            }
        ),
        headers=VARIES,
    )


@get("/timeline/spread", sync_to_thread=True)
def spread(
    state: State,
    start: FromQuery[float],
    end: FromQuery[float],
    n: FromQuery[int],
    folder: FromQuery[str | None] = None,
    album: FromQuery[str | None] = None,
    person: FromQuery[str | None] = None,
    artifact: FromQuery[str | None] = None,
    kind: FromQuery[str | None] = None,
    favorite: FromQuery[str | None] = None,
    rating_min: FromQuery[int | None] = None,
    f: FromQuery[list[str] | None] = None,
) -> Response[TimelineSpread]:
    """Up to `n` pictures of [start, end) spread evenly through it: what
    a surface that can show n pictures of a range asks for, whatever
    the range holds."""
    if end <= start:
        raise ClientException("the range is empty")
    asked = _question(folder, album, person, artifact, kind, favorite, rating_min, f)
    conn = connect.connect(state.db_path, read_only=True)
    try:
        scope, _ = _scope(conn, state, asked)
        rows = pages.timeline_spread(conn, start, end, max(1, min(n, SPREAD_MOST)), scope)
    finally:
        connect.close(conn)
    return Response(
        TimelineSpread(
            pictures=[
                TimelineMoment(slug=slug, moment=moment, thumb=thumbs.asset_url(sha, slug, medium=kind))
                for slug, moment, sha, kind in rows
            ]
        ),
        headers=VARIES,
    )


@get("/timeline/nth", sync_to_thread=True)
def nth(
    state: State,
    start: FromQuery[float],
    end: FromQuery[float],
    k: FromQuery[int],
    folder: FromQuery[str | None] = None,
    album: FromQuery[str | None] = None,
    person: FromQuery[str | None] = None,
    artifact: FromQuery[str | None] = None,
    kind: FromQuery[str | None] = None,
    favorite: FromQuery[str | None] = None,
    rating_min: FromQuery[int | None] = None,
    f: FromQuery[list[str] | None] = None,
) -> Response[TimelineNth]:
    """The k-th picture of [start, end) in moment order, of the n it
    holds -- what a hand k/n of the way along a segment points at. By
    rank, so a burst spreads across the segment's whole height. 404
    when the range holds none."""
    if end <= start:
        raise ClientException("the range is empty")
    asked = _question(folder, album, person, artifact, kind, favorite, rating_min, f)
    conn = connect.connect(state.db_path, read_only=True)
    try:
        scope, _ = _scope(conn, state, asked)
        found, n = pages.timeline_nth(conn, start, end, k, scope)
    finally:
        connect.close(conn)
    if found is None:
        raise NotFoundException("no picture in this range")
    slug, moment, sha, medium = found
    return Response(
        TimelineNth(
            slug=slug,
            moment=moment,
            thumb=thumbs.asset_url(sha, slug, medium=medium),
            k=min(max(0, k), n - 1),
            of=n,
            spelled=_spell(moment, "minute"),
        ),
        headers=VARIES,
    )


@get("/timeline/at", sync_to_thread=True)
def at(
    state: State,
    t: FromQuery[float],
    folder: FromQuery[str | None] = None,
    album: FromQuery[str | None] = None,
    person: FromQuery[str | None] = None,
    artifact: FromQuery[str | None] = None,
    kind: FromQuery[str | None] = None,
    favorite: FromQuery[str | None] = None,
    rating_min: FromQuery[int | None] = None,
    f: FromQuery[list[str] | None] = None,
) -> Response[TimelineAt]:
    """The picture a hand over moment `t` is pointing at: the nearest in
    time, either side; 404 when the scope holds none."""
    asked = _question(folder, album, person, artifact, kind, favorite, rating_min, f)
    conn = connect.connect(state.db_path, read_only=True)
    try:
        scope, _ = _scope(conn, state, asked)
        found = pages.timeline_at(conn, t, scope)
    finally:
        connect.close(conn)
    if found is None:
        raise NotFoundException("no pictures match these filters in this range")
    slug, moment = found
    return Response(TimelineAt(slug=slug, moment=moment, spelled=_spell(moment, "minute")), headers=VARIES)


@get(
    "/timeline",
    # The route negotiates three ways, and a union that mixes a page with a
    # JSON answer reaches OpenAPI as the empty schema however precisely the
    # arms are written (litestar v2.24.0). The JSON answer is declared here.
    responses={
        200: ResponseSpec(
            data_container=TimelineSurface,
            description="The timeline at one window: the same surface the page and the fragment render",
            media_type=MediaType.JSON,
            generate_examples=False,
        )
    },
    sync_to_thread=True,
)
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
    snap: FromQuery[bool] = False,
) -> Template | Response[TimelineSurface] | Redirect:
    """The timeline at one window: JSON to a machine, the surface fragment
    to htmx (what a brush or scrubber move fetches), the page to a
    browser -- one builder, one renderer. `bin` is accepted from older
    links and ignored: the zoom follows the window. `snap` is the
    scrubber's ask: a window landing in empty time is carried back onto
    pictures."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        found = _session_range(conn, f)
        if found is not None:
            # a session filter from the gallery lands on the session's
            # hour range, the other filters kept, the session dropped
            rest, lo, hi = found
            query = [(k, v) for k, v in request.query_params.items() if k not in ("f", "start", "end", "bin")]
            query += [("f", one) for one in rest] + [("start", str(lo)), ("end", str(hi))]
            return Redirect(path="/timeline?" + urllib.parse.urlencode(query), status_code=303)
        asked = _question(folder, album, person, artifact, kind, favorite, rating_min, f)
        told = _surface(conn, state, asked, start, end, snap=snap)
        # While the connection is open: an id-valued clause needs a name,
        # and there is nothing to resolve it with after the finally below.
        chips = _chips(conn, asked)
    finally:
        connect.close(conn)
    # The same answer either way, stated once: the machine gets the model
    # and the templates render from what it validated, so the page and the
    # contract cannot describe different surfaces.
    surface = TimelineSurface.model_validate(told)
    if wants_json(request):
        return Response(surface, headers=VARIES)
    drawn = surface.model_dump(mode="json")
    template = "_timeline_surface.html" if request.headers.get("hx-request") == "true" else "timeline.html"
    return Template(
        media_type=MediaType.HTML,
        template_name=template,
        context={"surface": drawn, "chips": chips},
        headers=VARIES,
    )
