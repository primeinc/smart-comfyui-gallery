"""One collection, one address: the authored axis rendered as an entity.

`/t/{slug}` is the collection's only address, with the 301 contract
every entity address carries. The CollectionView owns AUTHORED facts --
name, kind, color, description, the parent/child hierarchy, and the
rule when the kind is rule-defined -- and never the media answer: for
`album` and `flag` the members are ONE ResultSet page of the
album-faceted GalleryQuery, the same membership `/g?album=` serves.

A `smart` collection's membership is UNEVALUATED, not empty: the
ResultSet refuses the scope outright (db/resultset.py bind), and this
view shows the rule and says the media answer does not exist yet --
`gallery` is None, never an empty grid pretending the rule ran.

`/albums` follows the negotiation the other indexes carry: the
historical JSON list for machines, a rendered card grid for a browser.
"""

from __future__ import annotations

import pathlib
import time

from litestar import Request, get
from litestar.datastructures import State
from litestar.exceptions import NotFoundException
from litestar.response import Redirect, Response, Template

from db import connect, naming, pages, resultset, settings
from sg_web import home
from sg_web.presenting import VARIES, wants_json


def view(conn, models_dir: str, collection_id: int, slug: str, now: float, *, legacy: bool) -> dict:
    """The CollectionView, assembled inside ONE database snapshot. The
    ResultSet page is the FIRST read inside it -- its currency read
    precedes the snapshot pin -- and a rule-defined collection is the
    ResultSet's own typed refusal, decided under the same snapshot the
    card is then read from: a kind converted mid-request answers wholly
    as one generation, never a static header over a smart body. (An
    empty collection CAN legally convert; the schema only refuses
    converting one that holds filed members.)

    The unbounded legacy `files` list is the machine Adapter's shape
    only, exactly as on the person and folder addresses."""
    with resultset.snapshot(conn):
        try:
            grid = resultset.page(conn, models_dir, resultset.parse(album=slug), 1, now)
        except resultset.UnevaluatedCollection:
            grid = None
        name, kind, color, description, sql_text, nl_text, parent_id = pages.collection_card(conn, collection_id)
        parent = None
        if parent_id is not None:
            addressed = naming.entity_slug(conn, parent_id)
            if addressed is not None:
                parent = addressed[1]
        told = {
            "slug": slug,
            "name": name,
            "kind": kind,
            "color": color,
            "description": description,
            "parent": parent,
            "collections": [
                {"slug": s, "name": n, "kind": k, "pictures": p}
                for _, s, n, k, p in pages.collection_children(conn, collection_id)
            ],
        }
        if grid is None:  # smart: unevaluated, refused by the ResultSet -- never "empty"
            told["rule"] = {"sql": sql_text, "nl": nl_text}
            told["count"] = None
            told["gallery"] = None
        else:
            told["count"] = grid["total"]
            told["gallery"] = {
                "items": grid["items"],
                "total": grid["total"],
                "pages": grid["pages"],
                "qs": grid["qs"],
            }
        if legacy:
            told["files"] = [{"slug": s, "name": n} for s, n in pages.album_files(conn, collection_id)]
        return told


def _albums_listed(db_path: str) -> list[dict]:
    conn = connect.connect(db_path)
    try:
        return [
            {"name": name, "slug": slug, "kind": kind, "pictures": pictures}
            for name, slug, kind, pictures in pages.albums(conn)
        ]
    finally:
        connect.close(conn)


def _branch(conn, parent_id: int | None) -> list[dict]:
    """One level of the authored hierarchy, recursively: each level is
    its own index search (db/pages.py COLLECTION_CHILDREN), so the tree
    never runs the whole-shelf scan-and-sort the plan gate forbids. The
    schema's cycle guard is what makes the recursion finite."""
    return [
        {"slug": slug, "name": name, "kind": kind, "pictures": pictures, "collections": _branch(conn, child_id)}
        for child_id, slug, name, kind, pictures in pages.collection_children(conn, parent_id)
    ]


def _albums_nested(db_path: str) -> list[dict]:
    """The collection hierarchy as it was authored: every node still
    opens its own /t/{slug}, and a rule-defined node shows its badge
    rather than a member count nothing computed."""
    conn = connect.connect(db_path)
    try:
        return _branch(conn, None)
    finally:
        connect.close(conn)


@get("/albums")
async def albums_index(state: State, request: Request) -> Template | Response:
    """Every collection, alphabetically -- rendered for a browser, the
    historical JSON list for everything else.

    Async on purpose: POST /albums shares this path, and a SYNC handler
    that returns a Response object 500s when a second handler sits on
    its path ("'coroutine' object has no attribute to_asgi_response",
    reproduced on litestar-org/litestar@64cd7da with a 12-line pair; its
    own static_files pairs same-path handlers only as async ones,
    litestar/static_files.py:115-133). The sqlite read crosses to a
    thread because this coroutine shares the event loop."""
    from anyio import to_thread

    accept = request.headers.get("accept", "")
    if "text/html" in accept and "application/json" not in accept:
        # The browser gets the hierarchy as authored; the flat list
        # stays the machines' historical shape.
        tree = await to_thread.run_sync(_albums_nested, state.db_path)
        return Template(template_name="albums.html", context={"albums": tree}, headers=VARIES)
    told = await to_thread.run_sync(_albums_listed, state.db_path)
    return Response(told, headers=VARIES)


@get("/t/{slug:str}", sync_to_thread=True)
def album_page(state: State, request: Request, slug: str) -> Template | Response | Redirect:
    """One collection at its address, presented for whoever is asking. A
    retired slug redirects to the live one."""
    conn = connect.connect(state.db_path)
    try:
        found = naming.resolve(conn, "collection", slug)
        if found is None:
            raise NotFoundException(f"no collection at /t/{slug}")
        collection_id, is_current = found
        if not is_current:
            live = naming.entity_slug(conn, collection_id)
            if live is not None:
                return Redirect(path=f"/t/{live[1]}", status_code=301)
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        told = view(conn, weights, collection_id, slug, time.time(), legacy=wants_json(request))
    finally:
        connect.close(conn)
    if wants_json(request):
        return Response(told, headers=VARIES)
    return Template(template_name="album.html", context={"album": told}, headers=VARIES)
