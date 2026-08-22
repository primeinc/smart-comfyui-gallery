"""The timeline: a VIEW over the one interpretation, never a feature
database.

Months and days come from derived_media_context (the local wall clock
when one was claimed, the knowable instant otherwise) and the overlay
comes from the latest event runs. Every day is a DOOR into the gallery
-- `/g?f=context.local_day:eq:...` -- so the ResultSet answers the
media and the timeline never grows a second membership engine. Nothing
here writes, groups, geocodes or interprets: POST /jobs/context and
POST /jobs/events are where the interpretation is refreshed, and this
page renders whatever they last produced. An empty timeline names its
own remedy instead of pretending an unindexed library has no past.
"""

from __future__ import annotations

from litestar import Request, get
from litestar.datastructures import State
from litestar.exceptions import ClientException
from litestar.params import Parameter
from litestar.response import Response, Template

from db import connect, facets, pages
from sg_web.presenting import VARIES, wants_json


def _day_door(day: str) -> str:
    """The day's link into the gallery, spelled by the Facet Interface
    itself -- the timeline never invents its own spelling of a
    question."""
    import urllib.parse

    return urllib.parse.urlencode([("f", facets.spell(facets.facet("context.local_day", "eq", day)))])


def _bin_door(at: float, width: int) -> str:
    """The gallery question a bin answers: every picture whose human
    moment lies inside it, spelled by the Facet Interface."""
    import urllib.parse

    low = facets.facet("context.moment", "gte", str(int(at)))
    high = facets.facet("context.moment", "lte", str(int(at) + width - 1))
    return urllib.parse.urlencode([("f", facets.spell(low)), ("f", facets.spell(high))])


_SPAN = {"day": 86_400, "hour": 3_600, "minute": 60}


@get("/timeline/density", sync_to_thread=True)
def density(
    state: State,
    bin_name: str = Parameter(query="bin", default="day"),
    start: float | None = None,
    end: float | None = None,
) -> Response:
    """The surface at one zoom: pictures per bin of the human moment
    over [start, end) -- the whole interpreted extent when no range is
    asked -- the claims too coarse for the bin as spans, and the
    sessions touching the range in their own domain, each a door to
    its story when one has been told. Every bin is a door into the
    gallery. A range wider than the page can draw is refused with the
    remedy."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        extent = pages.timeline_extent(conn)
        if extent is None or extent[0] is None:
            return Response(
                {"bin": bin_name, "bin_seconds": pages.BINS.get(bin_name), "bins": [], "spans": [], "sessions": []}
            )
        lo = float(start) if start is not None else float(extent[0])
        hi = float(end) if end is not None else float(extent[1]) + 1.0
        try:
            width, bins, spans = pages.timeline_density(conn, bin_name, lo, hi)
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        sessions = pages.timeline_sessions(conn, lo, hi)
    finally:
        connect.close(conn)
    return Response(
        {
            "bin": bin_name,
            "bin_seconds": width,
            "start": lo,
            "end": hi,
            "extent": {"start": extent[0], "end": extent[1], "pictures": extent[2]},
            "bins": [
                {"at": at, "pictures": pictures, "wall": wall, "instant": instant, "qs": _bin_door(at, width)}
                for at, pictures, wall, instant in bins
            ],
            "spans": [
                {"start": s, "end": s + _SPAN[precision], "precision": precision, "pictures": pictures}
                for s, precision, pictures in spans
            ],
            "sessions": [
                {
                    "id": event_id,
                    "kind": kind,
                    "domain": "wall" if local_start is not None else "instant",
                    "start": local_start if local_start is not None else instant_start,
                    "end": local_end if local_end is not None else instant_end,
                    "pictures": pictures,
                    "snapshot_id": snapshot_id,
                    "story": f"/stories/renders/{render_id}" if render_id is not None else None,
                }
                for (
                    event_id,
                    kind,
                    local_start,
                    local_end,
                    instant_start,
                    instant_end,
                    pictures,
                    snapshot_id,
                    render_id,
                ) in sessions
            ],
        },
        headers=VARIES,
    )


@get("/timeline", sync_to_thread=True)
def timeline(state: State, request: Request) -> Template | Response:
    conn = connect.connect(state.db_path, read_only=True)
    try:
        months = [{"month": month, "pictures": pictures} for month, pictures in pages.timeline_months(conn)]
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
                "confidence": confidence,
                "member_hash": member_hash,
                "pictures": pictures,
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
    told = {"months": months, "days": days, "events": happenings}
    if wants_json(request):
        return Response(told, headers=VARIES)
    return Template(template_name="timeline.html", context=told, headers=VARIES)
