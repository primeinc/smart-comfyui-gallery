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

from db import connect, context, facets, pages, planning, resultset, settings
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
        "story": f"/stories/renders/{render_id}" if render_id is not None else None,
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


def _spell(epoch: float, bin_name: str, domain: str | None = None) -> str:
    """The moment as the page says it: the UTC day, and the clock past
    the day zoom; `Z` for an instant, `wall` for a wall clock."""
    d = datetime.datetime.fromtimestamp(epoch, datetime.UTC)
    suffix = "Z" if domain == "instant" else " wall" if domain == "wall" else ""
    if bin_name in ("day", "week"):
        return d.strftime("%Y-%m-%d") + suffix
    return d.strftime("%Y-%m-%d %H:%M") + suffix


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
    listed = rows[:SESSIONS_MOST]
    sessions = [_session(conn, row, samples=len(listed) <= SESSIONS_SAMPLED_MOST, scope=held) for row in listed]
    for one in sessions:
        one["start_spelled"] = _spell(one["start"], "hour", one["domain"])
        one["end_spelled"] = _spell(one["end"], "hour", one["domain"])

    span = max(1.0, hi - lo)
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
                "w": round(max(1.0, (overview_width / whole_span) * _W), 2),
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
    note = f"{fine} pictures in this window, one bar per {bin_name}"
    if coarse:
        note += f"; {coarse} claim only a coarser window, drawn as spans"
    if not every:
        note += f" · thumbnails for the {sum(1 for b in told_bins if b['samples'])} busiest {bin_name}s"
    if not coverage["complete"]:
        note += f" · {coverage['present'] - coverage['interpreted']} files not yet interpreted"
    if coverage["present"] and not coverage["events_current"]:
        note += " · sessions need the events job: the interpretation moved since they were grouped"
    return {
        "bin": bin_name,
        "bin_seconds": width,
        "start": lo,
        "end": hi,
        "start_spelled": _spell(lo, "hour"),
        "end_spelled": _spell(hi, "hour"),
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
                "spelled": _spell(s, "hour"),
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
