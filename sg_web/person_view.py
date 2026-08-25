"""One person, one address, three ways of looking at them.

`/p/{slug}` is the person's ONLY address, with the rename/301 contract
it has always carried. A person is an entity with a collection of
media, not a piece of media -- so the contextual presentation is a
DRAWER over the mounted People index, never the photographic lightbox.
Same architecture as sg_web/media_view.py, different Adapter; the
shared overlay mechanics get extracted only now that both concrete
Adapters exist to show what is actually common.

The presentations, negotiated exactly as the media address is (and
declared with `Vary: Accept, HX-Request`):

    Accept names application/json      -> the PersonView itself
    else HX-Request: true              -> the drawer fragment
    else Accept names text/html        -> the full person page
    else (wildcard, machine default)   -> the PersonView itself

`/people` follows the same rule: the historical JSON index for
machines, a rendered card grid for a browser -- the page the drawer
mounts over.
"""

from __future__ import annotations

import time
import urllib.parse

from litestar import Request, get, post
from litestar.datastructures import State
from litestar.exceptions import ClientException, NotFoundException
from litestar.params import FromPath
from litestar.response import Redirect, Response, Template

from db import authored, connect, facets, naming, pages, resultset, settings
from sg_web import home
from sg_web.presenting import presented, presented_page, wants_json
from sg_web.wire import Wire


def _wall(conn, event_id: int) -> bool:
    """Whether the session knows a wall clock (db/pages.py spells the
    start in whichever domain it has; the page says which)."""
    row = pages.event_domain(conn, event_id)
    return bool(row and row[0] is not None)


def view(conn, models_dir: str, person_id: int, slug: str, now: float, *, legacy: bool) -> dict:
    """The PersonView: everything every presentation shows, assembled
    once, inside ONE database snapshot -- a clustering or naming commit
    landing between the reads must not hand back pictures from one
    generation under the name and folder counts of another. Same
    invariant MediaView carries.

    `gallery` is the bounded grid: one ResultSet page of this person's
    pictures under the person-faceted GalleryQuery, whose canonical
    form every rendered media link carries so the lightbox arrows
    walk THIS person, not the library. `count` is that same
    membership's total. The unbounded legacy `pictures` list exists for
    the historical JSON shape ONLY -- `legacy=True` is the machine
    Adapter's ask, and the browser and drawer paths never enumerate a
    person's whole photographic existence to render sixty cells."""
    query = resultset.parse(person=slug)
    with resultset.snapshot(conn):
        grid = resultset.page(conn, models_dir, query, 1, now)
        told = {
            "slug": slug,
            "name": pages.person_name(conn, person_id),
            "count": grid["total"],
            "sessions": [
                {
                    "id": event_id,
                    "kind": kind,
                    "start": start,
                    "end": end,
                    "domain": "wall" if start is not None and _wall(conn, event_id) else "instant",
                    "theirs": theirs,
                    "pictures": pictures,
                    "qs": resultset.canonical(
                        resultset.parse(person=slug, facets=[f"event.id:eq:{event_id}"], sort="moment")
                    ),
                    "story": f"/stories/renders/{render_id}" if render_id is not None else None,
                    # the session's hour window on the timeline, where its
                    # story is told; None while the session has no time
                    "timeline": (
                        "/timeline?"
                        + urllib.parse.urlencode(
                            {"bin": "hour", "start": int(start // 3600) * 3600, "end": int(end // 3600) * 3600 + 3600}
                        )
                        if start is not None and end is not None
                        else None
                    ),
                }
                for event_id, kind, start, end, theirs, pictures, render_id in pages.person_sessions(conn, person_id)
            ],
            "timeline": "/timeline?" + urllib.parse.urlencode([("person", slug)]),
            "places": [
                {
                    "id": place_id,
                    "slug": place_slug,
                    "name": name,
                    "kind": kind,
                    "pictures": int(pictures),
                    "qs": resultset.canonical(
                        resultset.parse(
                            person=slug, facets=[facets.spell(facets.facet("place.id", "eq", str(place_id)))]
                        )
                    ),
                }
                for place_id, place_slug, name, kind, pictures in pages.person_places(conn, person_id)
            ],
            "across_folders": [
                {"folder": f, "folder_slug": fs, "pictures": p}
                for f, fs, p in pages.person_across_folders(conn, person_id)
            ],
            "gallery": {
                "items": grid["items"],
                "total": grid["total"],
                "pages": grid["pages"],
                "qs": grid["qs"],
            },
        }
        if legacy:
            told["pictures"] = [{"slug": s, "name": n} for s, n in pages.person_files(conn, person_id)]
        return told


@get("/people", sync_to_thread=True)
def people_index(state: State, request: Request) -> Template | Response:
    """Everyone, most pictures first -- rendered for a browser, the
    historical JSON list for everything else."""
    conn = connect.connect(state.db_path)
    try:
        spans = pages.people_spans(conn)
        ids = {slug: person_id for person_id, slug in pages.people_ids(conn)}
        told = [
            {
                "name": name,
                "slug": slug,
                "pictures": pictures,
                "first_seen": spans.get(ids.get(slug, -1), (None, None))[0],
                "last_seen": spans.get(ids.get(slug, -1), (None, None))[1],
            }
            for name, slug, pictures in pages.people_by_most(conn)
        ]
        # An empty page says WHY: runs that exist but are not the default
        # each carry their standing, so "nobody clustered yet" is only
        # said when nothing ran.
        runs = pages.standings(conn) if not told else []
    finally:
        connect.close(conn)
    return presented_page(request, told, page="people.html", context={"people": told, "runs": runs})


@get("/p/{slug:str}", sync_to_thread=True)
def person_page(state: State, request: Request, slug: FromPath[str]) -> Template | Response | Redirect:
    """One person at their address, presented for whoever is asking. A
    retired slug redirects to the live one, so one person never has two
    addresses serving content."""
    import pathlib
    import time

    conn = connect.connect(state.db_path)
    try:
        found = naming.resolve(conn, "person", slug)
        if found is None:
            raise NotFoundException(f"no person at /p/{slug}")
        person_id, is_current = found
        if not is_current:
            live = naming.entity_slug(conn, person_id)
            if live is not None:
                return Redirect(path=f"/p/{live[1]}", status_code=301)
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        told = view(conn, weights, person_id, slug, time.time(), legacy=wants_json(request))
    finally:
        connect.close(conn)
    return presented(request, told, page="person.html", fragment="_person_drawer.html", name="person")


class NamedPerson(Wire):
    """What naming a person returns.

    The name became the ADDRESS, so `slug` is where the person lives now
    and the browser replaces its location with it -- the old slug retires
    into history and returns 301, and leaving it as a history stop would
    put a redirect between Back and the index.
    """

    slug: str
    name: str
    #: how many of the person's faces the name was written down against.
    #: A name with none would be lost by the next re-cluster, so the write
    #: is refused before the commit rather than accepted and forgotten.
    asserted: int


class NewName(Wire):
    """The body of POST /p/{slug}/name."""

    name: str


@post("/p/{slug:str}/name", sync_to_thread=True)
def name_person(state: State, slug: FromPath[str], data: NewName) -> NamedPerson:
    """Name a person -- the People page's primary action.

    The name becomes the address: a new slug is minted and the old one
    retires into history, so the URL somebody saved before the naming
    still returns a 301 (db/naming.py). And the naming is written
    down as assertions against the person's files -- the durable record a
    re-cluster or a full derived rebuild re-applies the name from, so the
    application cannot destroy what it accepted (db/authored.py)."""
    conn = connect.connect(state.db_path)
    try:
        found = naming.resolve(conn, "person", slug)
        if found is None:
            raise NotFoundException(f"no person at /p/{slug}")
        person_id, _ = found
        cleaned = data.name.strip()
        if not cleaned:
            raise ClientException("a name needs letters in it")
        now = time.time()
        fresh = authored.name_person(conn, person_id, cleaned, now)
        asserted = authored.assert_named_cluster(conn, person_id, None, now)
        if pages.person_assertions(conn, person_id) == 0:
            # Refused BEFORE the commit, so nothing above persists: a name
            # with no face to assert it against would be silently lost by
            # the next re-cluster, and the application must not accept
            # what it cannot keep.
            raise ClientException(f"/p/{slug} has no clustered face to keep the name by; nothing was renamed")
        conn.commit()
        return NamedPerson(slug=fresh, name=cleaned, asserted=asserted)
    finally:
        connect.close(conn)
