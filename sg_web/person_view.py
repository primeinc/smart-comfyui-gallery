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

import dataclasses
import time

from litestar import Request, get, post
from litestar.datastructures import State
from litestar.exceptions import ClientException, NotFoundException
from litestar.response import Redirect, Response, Template

from db import authored, connect, naming, pages, resultset, settings
from sg_web import home
from sg_web.gallery import canonical

VARIES = {"vary": "Accept, HX-Request"}


def view(conn, models_dir: str, person_id: int, slug: str, now: float) -> dict:
    """The PersonView: everything every presentation shows, assembled
    once, inside ONE database snapshot -- a clustering or naming commit
    landing between the reads must not hand back pictures from one
    generation under the name and folder counts of another. Same
    invariant MediaView carries. Keys carried by the old JSON page keep
    their names; `gallery` is the ADDITIVE bounded grid: one ResultSet
    page of this person's pictures under the person-scoped
    GalleryQuery, whose canonical spelling every rendered media link
    carries so the lightbox arrows walk THIS person, not the library.
    `pages.person_files` stays only for the legacy JSON shape -- the
    rendered grid orders and pages through the ResultSet alone."""
    query = resultset.parse(person=slug)
    with resultset.snapshot(conn):
        grid = resultset.page(conn, models_dir, query, 1, now)
        pictures = [{"slug": s, "name": n} for s, n in pages.person_files(conn, person_id)]
        return {
            "slug": slug,
            "name": pages.person_name(conn, person_id),
            "count": len(pictures),
            "pictures": pictures,
            "across_folders": [
                {"folder": f, "folder_slug": fs, "pictures": p}
                for f, fs, p in pages.person_across_folders(conn, person_id)
            ],
            "gallery": {
                "items": grid["items"],
                "total": grid["total"],
                "pages": grid["pages"],
                "qs": canonical(query),
            },
        }


def _presented(request: Request, told: dict, page_template: str, fragment_template: str) -> Template | Response:
    accept = request.headers.get("accept", "")
    if "application/json" in accept:
        return Response(told, headers=VARIES)
    if request.headers.get("hx-request") == "true":
        return Template(template_name=fragment_template, context={"person": told}, headers=VARIES)
    if "text/html" in accept:
        return Template(template_name=page_template, context={"person": told}, headers=VARIES)
    return Response(told, headers=VARIES)


@get("/people", sync_to_thread=True)
def people_index(state: State, request: Request) -> Template | Response:
    """Everyone, most pictures first -- rendered for a browser, the
    historical JSON list for everything else."""
    conn = connect.connect(state.db_path)
    try:
        told = [
            {"name": name, "slug": slug, "pictures": pictures} for name, slug, pictures in pages.people_by_most(conn)
        ]
    finally:
        connect.close(conn)
    accept = request.headers.get("accept", "")
    if "text/html" in accept and "application/json" not in accept:
        return Template(template_name="people.html", context={"people": told}, headers=VARIES)
    return Response(told, headers=VARIES)


@get("/p/{slug:str}", sync_to_thread=True)
def person_page(state: State, request: Request, slug: str) -> Template | Response | Redirect:
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
        told = view(conn, weights, person_id, slug, time.time())
    finally:
        connect.close(conn)
    return _presented(request, told, "person.html", "_person_drawer.html")


@dataclasses.dataclass
class NewName:
    """The body of POST /p/{slug}/name. Typed so a nameless request is a
    400 from the signature model."""

    name: str


@post("/p/{slug:str}/name", sync_to_thread=True)
def name_person(state: State, slug: str, data: NewName) -> dict:
    """Name a person -- the People page's primary action.

    The name becomes the address: a new slug is minted and the old one
    retires into history, so the URL somebody saved before the naming
    still answers with a 301 (db/naming.py). And the naming is written
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
        return {"slug": fresh, "name": cleaned, "asserted": asserted}
    finally:
        connect.close(conn)
