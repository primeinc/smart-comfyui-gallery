"""Authored facts about one media item, written as DESIRED STATE.

Every route states the final fact -- favorite = true, rating = 4,
member of album X = false -- so a retry after a network hiccup lands
where the person already put it, where a toggle retried would land on
the opposite. The rules live in db/authored.py (idempotence, rating
bounds, smart-collection refusal, per-actor keys); these routes only
resolve addresses, call that one Implementation, commit exactly once,
and return the authoritative post-commit state.

The response carries the LIVE file slug and the actor's whole
MediaAuthoredState, so the strip that asked can redraw itself from the
response instead of trusting its own click. Deciding whether the
MOUNTED gallery is still current after the commit is the client's
coherence check against /g/locate's (data version, result-set identity)
pair -- not these routes' business.
"""

from __future__ import annotations

import time

from litestar import get, post
from litestar.datastructures import State
from litestar.exceptions import ClientException, NotFoundException
from litestar.params import FromPath
from litestar.response import Response

from db import authored, collections, connect, context, naming, pages, places
from sg_web import media_view
from sg_web.media_view import AuthoredState, CollectionSummary
from sg_web.presenting import VARIES
from sg_web.wire import Wire


def _resolved(conn, kind: str, slug: str, where: str) -> int:
    """The entity id for an address, retired slugs included -- a
    write through an old bookmark still means the same entity."""
    found = naming.resolve(conn, kind, slug)
    if found is None:
        raise NotFoundException(f"no {kind} at {where}/{slug}")
    return found[0]


class AuthoredAnswer(Wire):
    """The authoritative post-commit state: the LIVE slug and the whole
    authored state, so the strip that asked redraws from the response
    instead of trusting its own click."""

    slug: str | None
    authored: AuthoredState


class CollectionChoice(Wire):
    """One row of the album picker's menu."""

    slug: str
    name: str
    kind: str
    filed: bool


def _answered(conn, file_id: int, actor_id: int) -> Response[AuthoredAnswer]:
    """The response every desired-state route gives.

    Built field by field rather than by asdict: the translation from the
    database's value to the browser's contract is the seam's work, and
    writing it out is what keeps `collections` from crossing as bare dicts.
    """
    live = naming.entity_slug(conn, file_id)
    state = authored.media_state(conn, file_id, actor_id)
    return Response(
        AuthoredAnswer(
            slug=live[1] if live else None,
            authored=AuthoredState(
                favorite=state.favorite,
                rating=state.rating,
                collections=[CollectionSummary(slug=one["slug"], name=one["name"]) for one in state.collections],
            ),
        ),
        headers=VARIES,
    )


class DesiredFlag(Wire):
    """The body of the boolean desired-state routes."""

    value: bool


class DesiredRating(Wire):
    """The body of POST /i/{slug}/rating: 1..5, or null to clear."""

    value: int | None = None


@post("/i/{slug:str}/favorite", sync_to_thread=True)
def set_favorite(state: State, slug: FromPath[str], data: DesiredFlag) -> Response[AuthoredAnswer]:
    conn = connect.connect(state.db_path)
    try:
        file_id = _resolved(conn, "file", slug, "/i")
        authored.set_favorite(conn, file_id, state.actor_id, data.value, time.time())
        conn.commit()
        return _answered(conn, file_id, state.actor_id)
    finally:
        connect.close(conn)


class DesiredPlace(Wire):
    """The body of POST /i/{slug}/place: a place by name and kind, or a
    null name to withdraw the claim."""

    name: str | None = None
    kind: media_view.PlaceKind = "locality"
    #: the place this one is within, named the same way; optional
    within: str | None = None
    within_kind: media_view.PlaceKind = "country"


class PlaceAnswer(Wire):
    """The authoritative post-commit response of POST /i/{slug}/place: the
    LIVE slug and where the picture now says it happened, or null where
    the claim was withdrawn."""

    slug: str | None
    where: media_view.Where | None


@post("/i/{slug:str}/place", sync_to_thread=True)
def set_place(state: State, slug: FromPath[str], data: DesiredPlace) -> Response[PlaceAnswer]:
    """Say where this picture happened. The place is found or minted by
    name and kind, the claim is authored desired state, and the file's
    context is re-interpreted at once so every page reads it."""
    conn = connect.connect(state.db_path)
    try:
        file_id = _resolved(conn, "file", slug, "/i")
        now = time.time()
        try:
            place_id = None
            if data.name is not None:
                parent = places.named(conn, data.within, data.within_kind, now) if data.within else None
                place_id = places.named(conn, data.name, data.kind, now, within=parent)
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        authored.set_place(conn, file_id, state.actor_id, place_id, now)
        context.rebuild_one(conn, file_id, now)
        conn.commit()
        live = naming.entity_slug(conn, file_id)
        return Response(
            PlaceAnswer(slug=live[1] if live else None, where=media_view.where_of(conn, file_id)), headers=VARIES
        )
    finally:
        connect.close(conn)


@post("/i/{slug:str}/rating", sync_to_thread=True)
def set_rating(state: State, slug: FromPath[str], data: DesiredRating) -> Response[AuthoredAnswer]:
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
def set_membership(
    state: State, slug: FromPath[str], collection: FromPath[str], data: DesiredFlag
) -> Response[AuthoredAnswer]:
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
def collection_choices(state: State, slug: FromPath[str]) -> Response[list[CollectionChoice]]:
    """The album picker's menu: every LISTED collection and whether this
    file is filed in each. Lazily fetched on click -- the MediaView
    itself carries only current memberships -- and not an entity: no
    canonical URL, no history, no overlay."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        file_id = _resolved(conn, "file", slug, "/i")
        told = [
            CollectionChoice(slug=s, name=n, kind=k, filed=bool(filed))
            for s, n, k, filed in pages.collection_choices(conn, file_id)
        ]
    finally:
        connect.close(conn)
    return Response(told, headers=VARIES)
