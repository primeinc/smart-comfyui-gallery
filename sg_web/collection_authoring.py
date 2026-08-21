"""One write adapter for the collection lifecycle.

Every route here parses an address and a body, hands the desired state
to db/collections.py, commits exactly once, and answers with the
authoritative CollectionView -- the browser never invents the resulting
state, it renders what the server read back after the commit. Rules
come from the same GalleryQuery-shaped inputs the gallery itself takes;
db/collection_rules.py owns every conversion and the browser never
constructs rule JSON.

Definition writes name the revision they edited: `expected_rev` in the
body, or a standard `If-Match` carrying the ETag the GET handed out. A
stale revision is a 409 with zero mutation -- the editor re-reads and
decides again. The PATCH body is read as a plain mapping on purpose:
absent means unchanged and null means clear, a distinction a typed
default would flatten (db/collections.py UNSET).
"""

from __future__ import annotations

import dataclasses
import pathlib
import re
import time

from litestar import Request, patch, post, put
from litestar.datastructures import State
from litestar.exceptions import ClientException, HTTPException, NotFoundException
from litestar.response import Response

from db import collection_rules, collections, connect, naming, settings
from db.resultset import canonical
from sg_web import collection_view, home
from sg_web.asking import gallery_query as _asked
from sg_web.presenting import VARIES

#: The revision inside an ETag this application minted: W/"{slug}-r{N}".
_ETAG_REV = re.compile(r'-r(\d+)"\s*$')


def _collection_at(conn, slug: str) -> int:
    """The entity id for an address, retired spellings included -- a
    write through an old bookmark still means the same entity."""
    found = naming.resolve(conn, "collection", slug)
    if found is None:
        raise NotFoundException(f"no collection at /t/{slug}")
    return found[0]


def _revision_named(data: dict, request: Request):
    """expected_rev from the body, or the If-Match validator. Absence is
    a 428-shaped refusal spelled as a 400: a definition write that names
    no revision cannot be checked against anything."""
    if "expected_rev" in data:
        return data["expected_rev"]
    held = request.headers.get("if-match")
    if held is not None:
        matched = _ETAG_REV.search(held)
        if matched is not None:
            return int(matched.group(1))
    raise ClientException("a definition write names the revision it edited: expected_rev, or If-Match with the ETag")


def _written(state: State, work) -> Response:
    """Refusal mapping, one commit, and the authoritative after-state:
    every lifecycle write answers with the same CollectionView a GET
    serves, ETag included."""
    conn = connect.connect(state.db_path)
    try:
        try:
            collection_id = work(conn)
        except collections.CollectionChanged as moved:
            raise HTTPException(status_code=409, detail=str(moved)) from moved
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        live = naming.entity_slug(conn, collection_id)
        told = collection_view.view(
            conn, weights, collection_id, live[1] if live else "", time.time(), legacy=False, manage=True
        )
        return Response(told, headers={**VARIES, **collection_view.etag_of(told)})
    finally:
        connect.close(conn)


@dataclasses.dataclass
class NewCollection:
    """The body of POST /albums: the listed kinds, born active at
    revision 1, optionally already placed and decorated."""

    name: str
    kind: str = "album"
    parent: str | None = None
    color: str | None = None
    description: str | None = None


@post("/albums", sync_to_thread=True)
def make_album(state: State, data: NewCollection) -> Response:
    def work(conn):
        parent_id = _collection_at(conn, data.parent) if data.parent is not None else None
        return collections.create_listed(
            conn,
            data.name,
            time.time(),
            kind=data.kind,
            parent_id=parent_id,
            color=data.color,
            description=data.description,
            actor_id=state.actor_id,
        )

    return _written(state, work)


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
    parent: str | None = None
    color: str | None = None
    description: str | None = None


@post("/albums/smart", sync_to_thread=True)
def make_smart(state: State, data: NewSmart) -> Response:
    """Save the current view as a smart collection: one entity, one
    typed rule, one commit -- when the rule refuses, no collection
    remains. The rule pins the creating actor for its authored facets
    and stores entity references by uuid."""

    def work(conn):
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
        rule = collection_rules.from_gallery_query(conn, query, actor_id=state.actor_id, take=data.take)
        parent_id = _collection_at(conn, data.parent) if data.parent is not None else None
        return collections.create_smart(
            conn,
            data.name,
            rule,
            canonical(query) or "the whole library",
            time.time(),
            parent_id=parent_id,
            color=data.color,
            description=data.description,
            actor_id=state.actor_id,
        )

    return _written(state, work)


#: What a definition patch may say. Anything else is refused by name --
#: a misspelled field silently ignored would report success for an edit
#: that never happened.
_PATCHABLE = {"name", "color", "description", "parent", "archived", "expected_rev"}


def _edit_definition(state: State, expected_rev, slug: str, data: dict) -> Response:
    def work(conn):
        strange = set(data) - _PATCHABLE
        if strange:
            raise ValueError(f"the patch names facts a definition does not have: {', '.join(sorted(strange))}")
        collection_id = _collection_at(conn, slug)
        held: dict = {}
        for field in ("name", "color", "description", "archived"):
            if field in data:
                held[field] = data[field]
        if "parent" in data:
            if data["parent"] is None:
                held["parent_id"] = None
            elif isinstance(data["parent"], str):
                found = naming.resolve(conn, "collection", data["parent"])
                if found is None:
                    raise ValueError(f"no collection at /t/{data['parent']} to move under")
                held["parent_id"] = found[0]
            else:
                raise ValueError("parent names a collection by slug, or null for the top")
        collections.update_definition(
            conn, collection_id, collections.CollectionPatch(**held), state.actor_id, expected_rev, time.time()
        )
        return collection_id

    return _written(state, work)


@patch("/t/{slug:str}")
async def edit_definition(state: State, request: Request, slug: str, data: dict) -> Response:
    """The whole definition edit as one desired-state patch under one
    revision claim. Kind is deliberately not patchable -- changing how
    membership is decided is a transition, not a field.

    Async on purpose: GET /t/{slug} shares this path, and same-path
    handlers survive only as async ones (the albums_index note); the
    sqlite work crosses to a thread."""
    from anyio import to_thread

    expected_rev = _revision_named(data, request)
    return await to_thread.run_sync(_edit_definition, state, expected_rev, slug, data)


@put("/t/{slug:str}/rule", sync_to_thread=True)
def replace_rule(state: State, request: Request, slug: str, data: dict) -> Response:
    """This exact rule is now the collection's meaning: whole desired
    state, never predicate edits, under the same revision claim as any
    definition write. The body carries the same GalleryQuery-shaped
    inputs the save-view flow sends (`kind` here is the media kind)."""

    def work(conn):
        expected_rev = _revision_named(data, request)
        collection_id = _collection_at(conn, slug)
        query = _asked(
            data.get("folder"),
            None,
            data.get("kind"),
            data.get("q"),
            data.get("sort"),
            None,
            person=data.get("person"),
            favorite=data.get("favorite"),
            rating_min=data.get("rating_min"),
        )
        rule = collection_rules.from_gallery_query(conn, query, actor_id=state.actor_id, take=data.get("take"))
        spelled = canonical(query) or "the whole library"
        collections.replace_rule(conn, collection_id, rule, spelled, state.actor_id, expected_rev, time.time())
        return collection_id

    return _written(state, work)


@post("/t/{slug:str}/convert", sync_to_thread=True)
def convert_collection(state: State, request: Request, slug: str, data: dict) -> Response:
    """An explicit definition-mode transition. album<->flag moves
    freely; becoming smart requires an empty membership and a valid rule
    in this same operation; leaving smart requires the rule's discard
    said out loud, because the rule is authored state."""

    def work(conn):
        expected_rev = _revision_named(data, request)
        collection_id = _collection_at(conn, slug)
        wanted = data.get("kind")
        if wanted == "smart":
            query = _asked(
                data.get("folder"),
                None,
                data.get("media_kind"),
                data.get("q"),
                data.get("sort"),
                None,
                person=data.get("person"),
                favorite=data.get("favorite"),
                rating_min=data.get("rating_min"),
            )
            rule = collection_rules.from_gallery_query(conn, query, actor_id=state.actor_id, take=data.get("take"))
            collections.convert_to_smart(
                conn,
                collection_id,
                rule,
                canonical(query) or "the whole library",
                state.actor_id,
                expected_rev,
                time.time(),
            )
        elif wanted in collections.LISTED:
            collections.convert_to_listed(
                conn,
                collection_id,
                wanted,
                state.actor_id,
                expected_rev,
                time.time(),
                discard_rule=data.get("discard_rule", False),
            )
        else:
            raise ValueError("convert names the target kind: album, flag or smart")
        return collection_id

    return _written(state, work)
