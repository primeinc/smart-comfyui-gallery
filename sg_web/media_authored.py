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

import time
from typing import Literal

from litestar import get, post
from litestar.datastructures import State
from litestar.exceptions import ClientException, NotFoundException
from litestar.params import FromPath
from litestar.response import Response

from db import authored, collections, connect, context, naming, pages, places
from sg_web import media_view
from sg_web.media_view import AuthoredState, CollectionSummary, Faces, TagSummary
from sg_web.presenting import VARIES
from sg_web.wire import Wire


def _resolved(conn, kind: str, slug: str, where: str) -> int:
    """The entity id for an address, retired spellings included -- a
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
    """The answer every desired-state route gives.

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
                tags=[TagSummary(tag=one["tag"], label=one["label"]) for one in state.tags],
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
    """The authoritative post-commit answer of POST /i/{slug}/place: the
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


class DesiredTag(Wire):
    """The body of POST /i/{slug}/tags: a keyword and whether it is on.

    The word is in the BODY and not the path, unlike the album route
    beside it. An album is addressed by a slug this application minted;
    a keyword is whatever somebody typed, spaces and all, and a free
    sentence squeezed into a path segment is an encoding argument nobody
    asked to have.
    """

    name: str
    value: bool = True


@post("/i/{slug:str}/tags", sync_to_thread=True)
def set_tag(state: State, slug: FromPath[str], data: DesiredTag) -> Response[AuthoredAnswer]:
    """Write a word on a picture, or take it off.

    Desired state like everything else here, so the same word posted
    twice is one keyword rather than a toggle that lands wherever the
    retry left it. The keyword is minted on first use and disappears
    when its last picture lets go of it (db/authored.py set_tag_many).
    """
    conn = connect.connect(state.db_path)
    try:
        file_id = _resolved(conn, "file", slug, "/i")
        try:
            authored.set_tag(conn, file_id, state.actor_id, data.name, data.value, time.time())
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


class Judged(Wire):
    """The body of POST /i/{slug}/said/verdict: a thumb on one caption.

    The claim names the kind of annotation and the producer that made it,
    not the annotation's row id. A re-run replaces that row, and the
    verdict has to survive it; a row id would point at what the re-run
    deleted.
    """

    #: which kind of thing the model said (db/schema.sql derived_annotation)
    kind: media_view.SaidKind
    model_id: str
    model_version: str
    #: `null` retracts: clicking the lit thumb means "I take that back",
    #: and the honest record of taking something back is no row -- a
    #: verdict of "none" would be a third opinion nobody expressed.
    verdict: Literal["right", "wrong", "unsure"] | None = None
    note: str | None = None


class Verdict(Wire):
    """What this actor now says about that claim, after the write."""

    kind: str
    model_id: str
    model_version: str
    verdict: str | None


@post("/i/{slug:str}/said/verdict", sync_to_thread=True)
def judge_said(state: State, slug: FromPath[str], data: Judged) -> Response[Verdict]:
    """Say whether a caption is right, in one click, where it is shown.

    A review queue is a chore nobody does; the inspector is where
    somebody is already looking at the sentence and already knows the
    answer. So the gesture costs nothing to ignore and one click to use,
    and clicking the lit thumb takes it back.

    Only the ANNOTATION arm is built. `feedback` also models verdicts on
    a similarity, a duplicate and a person, and shipping a general
    endpoint whose other three arms nothing exercises would be three
    contracts nobody has tested.
    """
    conn = connect.connect(state.db_path)
    try:
        file_id = _resolved(conn, "file", slug, "/i")
        # Retract first either way: a person changing yes to no has one
        # standing opinion, not two rows fighting over which is newest.
        authored.retract_feedback(conn, file_id, data.kind, data.model_id, data.model_version, state.actor_id)
        if data.verdict is not None:
            authored.feedback(
                conn,
                "annotation",
                data.verdict,
                time.time(),
                file_id=file_id,
                annotation_kind=data.kind,
                note=data.note,
                user_id=state.actor_id,
                model_id=data.model_id,
                model_version=data.model_version,
            )
        conn.commit()
        held = authored.standing_verdict(conn, file_id, data.kind, data.model_id, data.model_version, state.actor_id)
        return Response(
            Verdict(kind=data.kind, model_id=data.model_id, model_version=data.model_version, verdict=held),
            headers=VARIES,
        )
    finally:
        connect.close(conn)


class DeniedPerson(Wire):
    """The body of POST /i/{slug}/people/{person}/deny.

    `value` true denies, false withdraws the denial. Withdrawing is NOT
    asserting: it leaves no claim at all, so the next clustering run is
    free to decide again -- which is the difference the whole feature
    turns on.
    """

    value: bool = True


@post("/i/{slug:str}/people/{person:str}/deny", sync_to_thread=True)
def deny_person(state: State, slug: FromPath[str], person: FromPath[str], data: DeniedPerson) -> Response[Faces]:
    """Say this person is not in this picture, and have it stick.

    The thing there was no way to say. `retract` deletes a claim, which
    means "I take that back" -- and the next clustering run is then free
    to decide the same thing again, because nothing recorded that it was
    wrong. A denial is a claim, survives the rebuild, and refuses the
    name (db/derived.py `seed_clusters_from_assertions`).

    Answers with who the picture now holds, from the same read the page
    uses: the browser never computes the resulting state.
    """
    conn = connect.connect(state.db_path)
    try:
        file_id = _resolved(conn, "file", slug, "/i")
        person_id = _resolved(conn, "person", person, "/p")
        if data.value:
            authored.deny_person(conn, person_id, file_id, state.actor_id, time.time())
        else:
            authored.retract_person(conn, person_id, file_id)
        conn.commit()
        return Response(media_view.faces_of(conn, file_id), headers=VARIES)
    finally:
        connect.close(conn)
