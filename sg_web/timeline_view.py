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
from litestar.response import Response, Template

from db import connect, facets, pages
from sg_web.presenting import VARIES, wants_json


def _day_door(day: str) -> str:
    """The day's link into the gallery, spelled by the Facet Interface
    itself -- the timeline never invents its own spelling of a
    question."""
    import urllib.parse

    return urllib.parse.urlencode([("f", facets.spell(facets.facet("context.local_day", "eq", day)))])


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
