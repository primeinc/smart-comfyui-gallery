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

import calendar
import urllib.parse

from litestar import Request, get
from litestar.datastructures import State
from litestar.exceptions import ClientException
from litestar.params import Parameter
from litestar.response import Response, Template

from db import connect, context, facets, pages, planning
from sg_web.presenting import VARIES, presented_page


def _door(*held: facets.Facet) -> str:
    """A gallery question, spelled by the Facet Interface -- the timeline
    never invents its own spelling. Every door carries the surface's own
    scope, so what it opens is exactly what it counted."""
    return urllib.parse.urlencode([("f", facets.spell(one)) for one in held] + [("sort", "moment")])


def _day_door(day: str, scope: tuple = ()) -> str:
    return _door(facets.facet("context.local_day", "eq", day), *scope)


def _month_door(month: str, scope: tuple = ()) -> str:
    year, mo = (int(part) for part in month.split("-"))
    last = calendar.monthrange(year, mo)[1]
    return _door(
        facets.facet("context.local_day", "gte", f"{month}-01"),
        facets.facet("context.local_day", "lte", f"{month}-{last:02d}"),
        *scope,
    )


def _bin_door(at: float, width: int, scope: tuple = ()) -> str:
    low = facets.facet("context.moment", "gte", str(int(at)))
    high = facets.facet("context.moment", "lte", str(int(at) + width - 1))
    return _door(low, high, *scope)


def _event_door(event_id: int, scope: tuple = ()) -> str:
    return _door(facets.facet("event.id", "eq", str(event_id)), *scope)


def _scoped(f) -> tuple:
    """The surface's scope: the gallery's facet spellings, normalized by
    the Facet Interface; a bad one is refused with the vocabulary. A
    session's door inside a scope is refused too -- `event.id` names
    one session, and a scope of one session is the gallery's job."""
    try:
        held = facets.normalized(f)
    except ValueError as refused:
        raise ClientException(str(refused)) from refused
    if any(one.key == "event.id" for one in held):
        raise ClientException("a session is a door, not a scope; open it in the gallery")
    return held


def _scope_told(held: tuple) -> list[dict]:
    return [{"spelled": facets.spell(one), "key": one.key, "value": one.value} for one in held]


#: Which planner tells which kind of session's story; a kind with none
#: is offered no button and told why.
PLANNER_FOR = {
    "generation_session": "generation_history",
    "capture_session": "capture_history",
    "file_session": "file_history",
}

_SPAN = {"day": 86_400, "hour": 3_600, "minute": 60}

#: Sessions one answer lists, and how many of those carry thumbnails --
#: a whole library's extent at the day zoom can touch thousands of
#: sessions; the page lists a bounded head, says how many more there
#: are, and the person zooms in. Never a silent cut.
SESSIONS_MOST = 200
SESSIONS_SAMPLED_MOST = 60


def _coverage(conn) -> dict:
    have, present, contested = pages.timeline_coverage(conn)
    return {
        "interpreted": have,
        "present": present,
        "contested": contested,
        "contested_qs": _door(facets.facet("context.disputed", "eq", "1")),
        "policy_version": context.POLICY_VERSION,
        "complete": have == present,
    }


def _session(conn, row, *, samples: bool, scope: tuple = ()) -> dict:
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
                "qs": _door(facets.facet("place.id", "eq", str(place_id))),
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
    }


@get("/timeline/density", sync_to_thread=True)
def density(
    state: State,
    bin_name: str = Parameter(query="bin", default="day"),
    start: float | None = None,
    end: float | None = None,
    f: list[str] | None = None,
) -> Response:
    """The surface at one zoom: pictures per bin of the human moment
    over [start, end) -- the whole interpreted extent when no range is
    asked -- split by clock domain and by origin, with a thumbnail
    sample per bin when the bins are few; the claims too coarse for
    the bin as spans; the sessions touching the range in their own
    domain, each a door to its pictures and to its story. Every bin is
    a door into the gallery. `f` scopes the surface by the gallery's
    own facets -- a place, an origin, a day -- and every door carries
    the scope. A range wider than the page can draw is refused with the
    remedy."""
    held = _scoped(f)
    scope = facets.conjunction(held)
    conn = connect.connect(state.db_path, read_only=True)
    try:
        coverage = _coverage(conn)
        extent = pages.timeline_extent(conn, scope)
        if extent is None or extent[0] is None:
            return Response(
                {
                    "bin": bin_name,
                    "bin_seconds": pages.BINS.get(bin_name),
                    "scope": _scope_told(held),
                    "bins": [],
                    "spans": [],
                    "sessions": [],
                    "sessions_total": 0,
                    "sessions_sampled": True,
                    "coverage": coverage,
                }
            )
        lo = float(start) if start is not None else float(extent[0])
        hi = float(end) if end is not None else float(extent[1]) + 1.0
        try:
            width, bins, spans = pages.timeline_density(conn, bin_name, lo, hi, scope)
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        samples = pages.timeline_samples(conn, bin_name, lo, hi, len(bins), scope)
        rows = pages.timeline_sessions(conn, lo, hi, scope)
        listed = rows[:SESSIONS_MOST]
        sessions = [_session(conn, row, samples=len(listed) <= SESSIONS_SAMPLED_MOST, scope=held) for row in listed]
    finally:
        connect.close(conn)
    return Response(
        {
            "bin": bin_name,
            "bin_seconds": width,
            "start": lo,
            "end": hi,
            "scope": _scope_told(held),
            "extent": {"start": extent[0], "end": extent[1], "pictures": extent[2]},
            "coverage": coverage,
            "sampled": bool(samples) or not bins,
            "bins": [
                {
                    "at": at,
                    "pictures": pictures,
                    "wall": wall,
                    "instant": instant,
                    "origin": {"captured": captured, "generated": generated, "mixed": mixed, "imported": imported},
                    "samples": samples.get(int(at), []),
                    "qs": _bin_door(at, width, held),
                }
                for at, pictures, wall, instant, captured, generated, mixed, imported in bins
            ],
            "spans": [
                {"start": s, "end": s + _SPAN[precision], "precision": precision, "pictures": pictures}
                for s, precision, pictures in spans
            ],
            "sessions": sessions,
            "sessions_total": len(rows),
            "sessions_sampled": len(listed) <= SESSIONS_SAMPLED_MOST,
        },
        headers=VARIES,
    )


@get("/timeline", sync_to_thread=True)
def timeline(state: State, request: Request, f: list[str] | None = None) -> Template | Response:
    held = _scoped(f)
    scope = facets.conjunction(held)
    conn = connect.connect(state.db_path, read_only=True)
    try:
        coverage = _coverage(conn)
        months = [
            {"month": month, "pictures": pictures, "qs": _month_door(month, held)}
            for month, pictures in pages.timeline_months(conn, scope)
        ]
        days = [
            {"day": day, "pictures": pictures, "qs": _day_door(day, held)}
            for day, pictures in pages.timeline_days(conn, scope=scope)
        ]
        happenings = [
            {
                "id": event_id,
                "grouper": grouper,
                "kind": kind,
                "local_start": local_start,
                "local_end": local_end,
                "instant_start": instant_start,
                "instant_end": instant_end,
                "start": local_start if local_start is not None else instant_start,
                "domain": "wall" if local_start is not None else "instant",
                "confidence": confidence,
                "member_hash": member_hash,
                "pictures": pictures,
                "qs": _event_door(event_id, held),
            }
            for (
                event_id,
                grouper,
                kind,
                local_start,
                local_end,
                instant_start,
                instant_end,
                confidence,
                member_hash,
                pictures,
            ) in pages.timeline_events(conn, scope=scope)
        ]
    finally:
        connect.close(conn)
    told = {
        "months": months,
        "days": days,
        "events": happenings,
        "coverage": coverage,
        "scope": _scope_told(held),
        "scope_qs": urllib.parse.urlencode([("f", facets.spell(one)) for one in held]),
    }
    return presented_page(request, told, page="timeline.html")
