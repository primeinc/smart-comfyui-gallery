"""The places shelf: everywhere a person has said a picture happened.

Read-only over what the context holds (db/pages.py PLACES_SHELF); a
place's door is the gallery's `place.id` facet. Saying where happens on
a picture's page or over a selection, never here.
"""

from __future__ import annotations

import urllib.parse

from litestar import Request, get
from litestar.datastructures import State
from litestar.response import Response, Template

from db import connect, facets, pages
from sg_web.presenting import presented_page


@get("/places", sync_to_thread=True)
def places_index(state: State, request: Request) -> Template | Response:
    """Every place named, most pictures first -- rendered for a browser,
    a JSON list for everything else. A place nobody's pictures are in
    any more is still listed: it was named, and it is an entity."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        told = [
            {
                "id": place_id,
                "slug": slug,
                "name": name,
                "kind": kind,
                "pictures": int(pictures),
                "first_seen": first,
                "last_seen": last,
                "qs": urllib.parse.urlencode([("f", facets.spell(facets.facet("place.id", "eq", str(place_id))))]),
                "timeline": "/timeline?"
                + urllib.parse.urlencode([("f", facets.spell(facets.facet("place.id", "eq", str(place_id))))]),
            }
            for place_id, slug, name, kind, pictures, first, last in pages.places_shelf(conn)
        ]
    finally:
        connect.close(conn)
    return presented_page(request, told, page="places.html", context={"places": told})
