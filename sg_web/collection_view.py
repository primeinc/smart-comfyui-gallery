"""One collection, one address: the authored axis rendered as an entity.

`/t/{slug}` is the collection's only address, with the 301 contract
every entity address carries. The CollectionView owns AUTHORED facts --
name, kind, color, description, the parent/child hierarchy, and the
rule when the kind is rule-defined -- and never the media answer: for
`album` and `flag` the members are ONE ResultSet page of the
album-faceted GalleryQuery, the same membership `/g?album=` serves.

A `smart` collection with a typed rule (db/collection_rules.py) is a
real gallery: the ResultSet evaluates the rule to a membership set and
orders it like any other scope. The other states stay LOUD and
distinct -- never an empty grid pretending the rule ran:

    no typed rule (migrated prose, or nothing)  -> unevaluated
    rule references a deleted entity            -> broken
    semantic rule nothing can answer right now  -> unavailable

`/albums` follows the negotiation the other indexes carry: the
historical JSON list for machines, a rendered card grid for a browser
-- and it NEVER evaluates smart rules just to show counts.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time

from litestar import Request, get, post
from litestar.datastructures import State
from litestar.exceptions import ClientException, NotFoundException
from litestar.response import Redirect, Response, Template

from db import authored, collection_rules, connect, naming, pages, resultset, settings
from db.resultset import canonical
from sg_web import home
from sg_web.asking import gallery_query as _asked
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
        grid = None
        state, reason = "evaluated", None
        try:
            grid = resultset.page(conn, models_dir, resultset.parse(album=slug), 1, now)
        except collection_rules.BrokenCollectionRule as why:
            state, reason = "broken", str(why)
        except collection_rules.UnavailableCollectionRule as why:
            state, reason = "unavailable", str(why)
        except resultset.UnevaluatedCollection:
            state = "unevaluated"
        name, kind, color, description, parent_id = pages.collection_card(conn, collection_id)
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
        if kind == "smart":
            held = collection_rules.provenance(conn, collection_id)
            told["rule"] = None if held is None else {"sql": held["sql"], "nl": held["nl"]}
            told["state"] = state
            if reason is not None:
                told["reason"] = reason
        if grid is None:
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


def _albums_nested(db_path: str) -> list[dict]:
    """The collection hierarchy as it was authored: every node still
    opens its own /t/{slug}, and a rule-defined node shows its badge
    rather than a member count nothing computed.

    ONE statement (db/pages.py COLLECTION_SHELF), nested here: a single
    SELECT is a single snapshot, so a reparent committed mid-render
    cannot show a collection twice or lose it -- the failure a query per
    node invited, on top of costing N+1 round trips for a page that
    promises every collection anyway. Rows arrive (parent_id, name)-
    ordered from the index: parents-first is NOT guaranteed, so nodes
    are made whole before any child is attached."""
    conn = connect.connect(db_path, read_only=True)
    try:
        rows = pages.collection_shelf(conn)
    finally:
        connect.close(conn)
    nodes = {
        cid: {"slug": slug, "name": name, "kind": kind, "pictures": pictures, "collections": []}
        for cid, _, slug, name, kind, pictures in rows
    }
    top: list[dict] = []
    for cid, parent_id, *_ in rows:  # row order IS name order within each parent
        if parent_id in nodes:
            nodes[parent_id]["collections"].append(nodes[cid])
        else:
            top.append(nodes[cid])
    return top


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


@dataclasses.dataclass
class NewSmart:
    """The body of POST /albums/smart: a name, an optional cutoff, and
    the canonical spelling of the question being saved. The server
    reconstructs the typed rule through the same seams that own query
    semantics -- the browser never defines a rule shape."""

    name: str
    take: int | None = None
    folder: str | None = None
    person: str | None = None
    kind: str | None = None
    favorite: str | None = None
    rating_min: int | None = None
    q: str | None = None
    sort: str | None = None


@post("/albums/smart", sync_to_thread=True)
def make_smart(state: State, data: NewSmart) -> dict:
    """Save the current view as a smart collection: one entity, one
    typed rule, one commit. The rule pins the creating actor for its
    authored facets and stores entity references by uuid
    (db/collection_rules.py owns every conversion)."""
    cleaned = data.name.strip()
    if not cleaned:
        raise ClientException("a smart collection needs a name")
    query = _asked(
        data.folder,
        None,
        data.kind,
        data.q,
        data.sort,
        None,
        person=data.person,
        favorite=data.favorite,
        rating_min=data.rating_min,
    )
    conn = connect.connect(state.db_path)
    try:
        try:
            rule = collection_rules.from_gallery_query(conn, query, actor_id=state.actor_id, take=data.take)
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        now = time.time()
        collection_id = authored.collection(conn, cleaned, now, kind="smart")
        spelled = canonical(query)
        collection_rules.save(conn, collection_id, rule, source_text=spelled or "the whole library", now=now)
        conn.commit()
        addressed = naming.entity_slug(conn, collection_id)
        return {"name": cleaned, "slug": addressed[1] if addressed else None, "kind": "smart"}
    finally:
        connect.close(conn)
