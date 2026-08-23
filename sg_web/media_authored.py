"""Authored facts about one media item, written as DESIRED STATE.

Every route states the final fact -- favorite = true, rating = 4,
member of album X = false -- so a retry after a network hiccup lands
where the person already put it, where a toggle retried would land on
the opposite. The rules live in db/authored.py (idempotence, rating
bounds, smart-collection refusal, per-actor keys); these routes only
resolve addresses, call that one Implementation, commit exactly once,
and answer with the authoritative post-commit state.

The answer carries the LIVE file slug and the actor's whole
MediaAuthoredState, so the strip that asked can redraw itself from the
response instead of trusting its own click. Deciding whether the
MOUNTED gallery is still current after the commit is the client's
coherence check against /g/locate's (currency, answer) pair -- not
these routes' business.
"""

from __future__ import annotations

import dataclasses
import time

from litestar import get, post
from litestar.datastructures import State
from litestar.exceptions import ClientException, NotFoundException
from litestar.response import Response

from db import authored, collections, connect, context, naming, pages, places
from sg_web import media_view
from sg_web.presenting import VARIES


def _resolved(conn, kind: str, slug: str, where: str) -> int:
    """The entity id for an address, retired spellings included -- a
    write through an old bookmark still means the same entity."""
    found = naming.resolve(conn, kind, slug)
    if found is None:
        raise NotFoundException(f"no {kind} at {where}/{slug}")
    return found[0]


def _answered(conn, file_id: int, actor_id: int) -> Response:
    live = naming.entity_slug(conn, file_id)
    told = {
        "slug": live[1] if live else None,
        "authored": dataclasses.asdict(authored.media_state(conn, file_id, actor_id)),
    }
    return Response(told, headers=VARIES)


@dataclasses.dataclass
class DesiredFlag:
    """The body of the boolean desired-state routes."""

    value: bool


@dataclasses.dataclass
class DesiredRating:
    """The body of POST /i/{slug}/rating: 1..5, or null to clear."""

    value: int | None = None


@post("/i/{slug:str}/favorite", sync_to_thread=True)
def set_favorite(state: State, slug: str, data: DesiredFlag) -> Response:
    conn = connect.connect(state.db_path)
    try:
        file_id = _resolved(conn, "file", slug, "/i")
        authored.set_favorite(conn, file_id, state.actor_id, data.value, time.time())
        conn.commit()
        return _answered(conn, file_id, state.actor_id)
    finally:
        connect.close(conn)


@dataclasses.dataclass
class DesiredPlace:
    """The body of POST /i/{slug}/place: a place by name and kind, or a
    null name to withdraw the claim."""

    name: str | None = None
    kind: str = "locality"


@post("/i/{slug:str}/place", sync_to_thread=True)
def set_place(state: State, slug: str, data: DesiredPlace) -> Response:
    """Say where this picture happened. The place is found or minted by
    name and kind, the claim is authored desired state, and the file's
    context is re-interpreted at once so every page reads it."""
    conn = connect.connect(state.db_path)
    try:
        file_id = _resolved(conn, "file", slug, "/i")
        now = time.time()
        try:
            place_id = places.named(conn, data.name, data.kind, now) if data.name is not None else None
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        authored.set_place(conn, file_id, state.actor_id, place_id, now)
        context.rebuild_one(conn, file_id, now)
        conn.commit()
        live = naming.entity_slug(conn, file_id)
        return Response(
            {"slug": live[1] if live else None, "where": media_view.where_of(conn, file_id)}, headers=VARIES
        )
    finally:
        connect.close(conn)


@post("/i/{slug:str}/rating", sync_to_thread=True)
def set_rating(state: State, slug: str, data: DesiredRating) -> Response:
    conn = connect.connect(state.db_path)
    try:
        file_id = _resolved(conn, "file", slug, "/i")
        try:
            authored.set_rating(conn, file_id, state.actor_id, data.value, time.time())
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        return _answered(conn, file_id, state.actor_id)
    finally:
        connect.close(conn)


@post("/i/{slug:str}/collections/{collection:str}", sync_to_thread=True)
def set_membership(state: State, slug: str, collection: str, data: DesiredFlag) -> Response:
    conn = connect.connect(state.db_path)
    try:
        file_id = _resolved(conn, "file", slug, "/i")
        collection_id = _resolved(conn, "collection", collection, "/t")
        try:
            collections.set_membership(conn, collection_id, file_id, data.value, time.time())
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        return _answered(conn, file_id, state.actor_id)
    finally:
        connect.close(conn)


@get("/i/{slug:str}/collection-choices", sync_to_thread=True)
def collection_choices(state: State, slug: str) -> Response:
    """The album picker's menu: every LISTED collection and whether this
    file is filed in each. Lazily fetched on click -- the MediaView
    itself carries only current memberships -- and not an entity: no
    canonical URL, no history, no overlay."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        file_id = _resolved(conn, "file", slug, "/i")
        told = [
            {"slug": s, "name": n, "kind": k, "filed": bool(filed)}
            for s, n, k, filed in pages.collection_choices(conn, file_id)
        ]
    finally:
        connect.close(conn)
    return Response(told, headers=VARIES)
