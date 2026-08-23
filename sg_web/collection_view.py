"""One collection, one address: the authored axis rendered as an entity.

`/t/{slug}` is the collection's only address, with the 301 contract
every entity address carries. The CollectionView owns AUTHORED facts --
name, kind, color, description, the parent/child hierarchy, lifecycle
(active or archived, the definition revision, who last defined it) and
the rule when the kind is rule-defined -- and never the media answer:
for `album` and `flag` the members are ONE ResultSet page of the
album-faceted GalleryQuery, the same membership `/g?album=` serves.

A `smart` collection with a typed rule (db/collection_rules.py) is a
real gallery: the ResultSet evaluates the rule to a membership set and
orders it like any other scope. The other states stay LOUD and
distinct -- never an empty grid pretending the rule ran:

    no typed rule (migrated prose, or nothing)  -> unevaluated
    rule references a deleted entity            -> broken
    semantic rule nothing can answer right now  -> unavailable

Those are conditions of the RULE, orthogonal to lifecycle: an archived
collection can be evaluated and an active one broken.

`/albums` follows the negotiation the other indexes carry: the
historical JSON list for machines, a rendered card grid for a browser
-- and it NEVER evaluates smart rules just to show counts. It shows the
ACTIVE tree; an active child of an archived parent surfaces at the top
level rather than vanishing with its organizer, which falls out of the
one-statement shelf structurally: the archived parent simply is not
among the nodes. `?state=archived` is the management shelf.
"""

from __future__ import annotations

import pathlib
import time
import urllib.parse
from typing import Annotated

from litestar import Request, get
from litestar.datastructures import State
from litestar.exceptions import NotFoundException
from litestar.params import FromPath, QueryParameter
from litestar.response import Redirect, Response, Template

from db import collection_rules, collections, connect, facets, naming, pages, resultset, settings
from sg_web import home
from sg_web.presenting import presented_page, wants_json


def view(
    conn, models_dir: str, collection_id: int, slug: str, now: float, *, legacy: bool, manage: bool = False
) -> dict:
    """The CollectionView, assembled inside ONE database snapshot. The
    ResultSet page is the FIRST read inside it -- its currency read
    precedes the snapshot pin -- and a rule-defined collection is the
    ResultSet's own typed refusal, decided under the same snapshot the
    card is then read from: a kind converted mid-request answers wholly
    as one generation, never a static header over a smart body. (An
    empty collection CAN legally convert; the schema only refuses
    converting one that holds filed members.)

    `manage` adds the parent picker's choices -- every active collection
    this one may legally move under -- so the browser never offers a
    move the database will refuse.

    The unbounded legacy `files` list is the machine Adapter's shape
    only, exactly as on the person and folder addresses."""
    with resultset.snapshot(conn):
        grid = None
        rule_state, reason = "evaluated", None
        try:
            grid = resultset.page(conn, models_dir, resultset.parse(album=slug), 1, now)
        except collection_rules.BrokenCollectionRule as why:
            rule_state, reason = "broken", str(why)
        except collection_rules.UnavailableCollectionRule as why:
            rule_state, reason = "unavailable", str(why)
        except resultset.UnevaluatedCollection:
            rule_state = "unevaluated"
        name, kind, color, description, parent_id, archived_at, definition_rev, updated_at, updated_by = (
            pages.collection_card(conn, collection_id)
        )
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
            "archived": archived_at is not None,
            "definition_rev": definition_rev,
            "updated_at": updated_at,
            "updated_by": updated_by,
            "collections": [
                {"slug": s, "name": n, "kind": k, "pictures": p}
                for _, s, n, k, p in pages.collection_children(conn, collection_id)
            ],
        }
        if kind == "smart":
            held = collection_rules.provenance(conn, collection_id)
            told["rule"] = None if held is None else {"sql": held["sql"], "nl": held["nl"]}
            told["state"] = rule_state
            if reason is not None:
                told["reason"] = reason
        if grid is None:
            told["count"] = None
            told["gallery"] = None
        else:
            told["count"] = grid["total"]
            told["first_seen"], told["last_seen"] = pages.collection_spans(conn).get(slug, (None, None))
            told["timeline"] = "/timeline?" + urllib.parse.urlencode([("album", slug)])
            told["places"] = [
                {
                    "id": place_id,
                    "slug": place_slug,
                    "name": name,
                    "kind": kind,
                    "pictures": int(pictures),
                    "qs": urllib.parse.urlencode(
                        [("album", slug), ("f", facets.spell(facets.facet("place.id", "eq", str(place_id))))]
                    ),
                }
                for place_id, place_slug, name, kind, pictures in pages.collection_places(conn, collection_id)
            ]
            told["gallery"] = {
                "items": grid["items"],
                "total": grid["total"],
                "pages": grid["pages"],
                "qs": grid["qs"],
            }
        if manage:
            allowed = collections.eligible_parents(conn, collection_id)
            told["parents"] = [
                {"slug": s, "name": n, "archived": bool(a)} for s, n, a in pages.collections_named(conn, allowed)
            ]
        if legacy:
            told["files"] = [{"slug": s, "name": n} for s, n in pages.album_files(conn, collection_id)]
        return told


def _albums_listed(db_path: str) -> list[dict]:
    conn = connect.connect(db_path)
    try:
        spans = pages.collection_spans(conn)
        return [
            {
                "name": name,
                "slug": slug,
                "kind": kind,
                "pictures": pictures,
                "first_seen": spans.get(slug, (None, None))[0],
                "last_seen": spans.get(slug, (None, None))[1],
            }
            for name, slug, kind, pictures in pages.albums(conn)
        ]
    finally:
        connect.close(conn)


def _albums_archived(db_path: str) -> list[dict]:
    conn = connect.connect(db_path, read_only=True)
    try:
        return [
            {"name": name, "slug": slug, "kind": kind, "pictures": pictures}
            for name, slug, kind, pictures in pages.archived_albums(conn)
        ]
    finally:
        connect.close(conn)


def _albums_nested(db_path: str) -> tuple[list[dict], int]:
    """The ACTIVE collection hierarchy as it was authored: every node
    still opens its own /t/{slug}, and a rule-defined node shows its
    badge rather than a member count nothing computed.

    ONE statement (db/pages.py COLLECTION_SHELF), nested here: a single
    SELECT is a single snapshot, so a reparent committed mid-render
    cannot show a collection twice or lose it -- the failure a query per
    node invited, on top of costing N+1 round trips for a page that
    promises every collection anyway. Rows arrive (parent_id, name)-
    ordered from the index: parents-first is NOT guaranteed, so nodes
    are made whole before any child is attached. An active child whose
    parent is archived finds no parent among the nodes and lands at the
    top level -- promotion is the data structure, not a special case."""
    conn = connect.connect(db_path, read_only=True)
    try:
        rows = pages.collection_shelf(conn)
        retired = pages.archived_count(conn)
        spans = pages.collection_spans(conn)
    finally:
        connect.close(conn)
    nodes = {
        cid: {
            "slug": slug,
            "name": name,
            "kind": kind,
            "pictures": pictures,
            "first_seen": spans.get(slug, (None, None))[0],
            "last_seen": spans.get(slug, (None, None))[1],
            "collections": [],
        }
        for cid, _, slug, name, kind, pictures in rows
    }
    top: list[dict] = []
    for cid, parent_id, *_ in rows:  # row order IS name order within each parent
        if parent_id in nodes:
            nodes[parent_id]["collections"].append(nodes[cid])
        else:
            top.append(nodes[cid])
    return top, retired


@get("/albums")
async def albums_index(
    state: State,
    request: Request,
    shown: Annotated[str | None, QueryParameter(name="state")] = None,
) -> Template | Response:
    """Every active collection, alphabetically -- rendered for a
    browser, the historical JSON list for everything else.
    `?state=archived` is the management shelf: the same negotiation,
    over what was retired.

    Async on purpose: POST /albums shares this path, and a SYNC handler
    that returns a Response object 500s when a second handler sits on
    its path ("'coroutine' object has no attribute to_asgi_response",
    reproduced on litestar-org/litestar@64cd7da with a 12-line pair; its
    own static_files pairs same-path handlers only as async ones,
    litestar/static_files.py:115-133). The sqlite read crosses to a
    thread because this coroutine shares the event loop."""
    from anyio import to_thread

    if shown == "archived":
        held = await to_thread.run_sync(_albums_archived, state.db_path)
        return presented_page(
            request,
            held,
            page="albums.html",
            context={"albums": [], "archived": held, "archived_count": len(held), "showing_archived": True},
        )
    if wants_json(request):
        return presented_page(request, await to_thread.run_sync(_albums_listed, state.db_path), page="albums.html")
    # The browser gets the hierarchy as authored; the flat list stays the
    # machines' historical shape -- assembled only for the one who asked.
    tree, retired = await to_thread.run_sync(_albums_nested, state.db_path)
    return presented_page(
        request,
        tree,
        page="albums.html",
        context={"albums": tree, "archived": [], "archived_count": retired, "showing_archived": False},
    )


def _album_page(state: State, request: Request, slug: str) -> Template | Response | Redirect:
    json_wanted = wants_json(request)
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
        told = view(conn, weights, collection_id, slug, time.time(), legacy=json_wanted, manage=not json_wanted)
    finally:
        connect.close(conn)
    return presented_page(request, told, page="album.html", context={"album": told})


@get("/t/{slug:str}")
async def album_page(state: State, request: Request, slug: FromPath[str]) -> Template | Response | Redirect:
    """One collection at its address, presented for whoever is asking. A
    retired slug redirects to the live one. The definition's concurrency
    token is `definition_rev` in the body -- deliberately NOT an ETag:
    this page's bytes change when membership does while the definition
    revision stands still, so no honest representation validator can
    double as the definition's.

    Async on purpose: PATCH /t/{slug} shares this path, and same-path
    handlers survive only as async ones (the albums_index note). The
    sqlite read crosses to a thread."""
    from anyio import to_thread

    return await to_thread.run_sync(_album_page, state, request, slug)
