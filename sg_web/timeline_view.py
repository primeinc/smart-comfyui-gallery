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
    never invents its own spelling."""
    return urllib.parse.urlencode([("f", facets.spell(one)) for one in held] + [("sort", "moment")])


def _day_door(day: str) -> str:
    return _door(facets.facet("context.local_day", "eq", day))


def _month_door(month: str) -> str:
    year, mo = (int(part) for part in month.split("-"))
    last = calendar.monthrange(year, mo)[1]
    return _door(
        facets.facet("context.local_day", "gte", f"{month}-01"),
        facets.facet("context.local_day", "lte", f"{month}-{last:02d}"),
    )


def _bin_door(at: float, width: int) -> str:
    low = facets.facet("context.moment", "gte", str(int(at)))
    high = facets.facet("context.moment", "lte", str(int(at) + width - 1))
    return _door(low, high)


def _event_door(event_id: int) -> str:
    return _door(facets.facet("event.id", "eq", str(event_id)))


#: Which planner tells which kind of session's story; a kind with none
#: is offered no button and told why.
PLANNER_FOR = {
    "generation_session": "generation_history",
    "capture_session": "capture_history",
    "file_session": "file_history",
}

_SPAN = {"day": 86_400, "hour": 3_600, "minute": 60}


def _coverage(conn) -> dict:
    have, present, contested = pages.timeline_coverage(conn)
    return {
        "interpreted": have,
        "present": present,
        "contested": contested,
        "policy_version": context.POLICY_VERSION,
        "complete": have == present,
    }


def _session(conn, row, *, samples: bool) -> dict:
    event_id, kind, local_start, local_end, instant_start, instant_end, pictures, snapshot_id, render_id = row
    planner = PLANNER_FOR.get(kind)
    return {
        "id": event_id,
        "kind": kind,
        "domain": "wall" if local_start is not None else "instant",
        "start": local_start if local_start is not None else instant_start,
        "end": local_end if local_end is not None else instant_end,
        "pictures": pictures,
        "snapshot_id": snapshot_id,
        "story": f"/stories/renders/{render_id}" if render_id is not None else None,
        "qs": _event_door(event_id),
        "planner": planner,
        "tellable": planner in planning.PLANNERS,
        "samples": pages.session_samples(conn, event_id) if samples else [],
    }


@get("/timeline/density", sync_to_thread=True)
def density(
    state: State,
    bin_name: str = Parameter(query="bin", default="day"),
    start: float | None = None,
    end: float | None = None,
) -> Response:
    """The surface at one zoom: pictures per bin of the human moment
    over [start, end) -- the whole interpreted extent when no range is
    asked -- split by clock domain and by origin, with a thumbnail
    sample per bin when the bins are few; the claims too coarse for
    the bin as spans; the sessions touching the range in their own
    domain, each a door to its pictures and to its story. Every bin is
    a door into the gallery. A range wider than the page can draw is
    refused with the remedy."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        coverage = _coverage(conn)
        extent = pages.timeline_extent(conn)
        if extent is None or extent[0] is None:
            return Response(
                {
                    "bin": bin_name,
                    "bin_seconds": pages.BINS.get(bin_name),
                    "bins": [],
                    "spans": [],
                    "sessions": [],
                    "coverage": coverage,
                }
            )
        lo = float(start) if start is not None else float(extent[0])
        hi = float(end) if end is not None else float(extent[1]) + 1.0
        try:
            width, bins, spans = pages.timeline_density(conn, bin_name, lo, hi)
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        samples = pages.timeline_samples(conn, bin_name, lo, hi, len(bins))
        sessions = [_session(conn, row, samples=True) for row in pages.timeline_sessions(conn, lo, hi)]
    finally:
        connect.close(conn)
    return Response(
        {
            "bin": bin_name,
            "bin_seconds": width,
            "start": lo,
            "end": hi,
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
                    "qs": _bin_door(at, width),
                }
                for at, pictures, wall, instant, captured, generated, mixed, imported in bins
            ],
            "spans": [
                {"start": s, "end": s + _SPAN[precision], "precision": precision, "pictures": pictures}
                for s, precision, pictures in spans
            ],
            "sessions": sessions,
        },
        headers=VARIES,
    )


@get("/timeline", sync_to_thread=True)
def timeline(state: State, request: Request) -> Template | Response:
    conn = connect.connect(state.db_path, read_only=True)
    try:
        coverage = _coverage(conn)
        months = [
            {"month": month, "pictures": pictures, "qs": _month_door(month)}
            for month, pictures in pages.timeline_months(conn)
        ]
        days = [{"day": day, "pictures": pictures, "qs": _day_door(day)} for day, pictures in pages.timeline_days(conn)]
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
                "qs": _event_door(event_id),
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
            ) in pages.timeline_events(conn)
        ]
    finally:
        connect.close(conn)
    told = {"months": months, "days": days, "events": happenings, "coverage": coverage}
    return presented_page(request, told, page="timeline.html")
