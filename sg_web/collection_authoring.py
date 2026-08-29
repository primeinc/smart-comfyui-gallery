"""One write adapter for the collection lifecycle.

Every route here parses an address and a body, hands the desired state
to db/collections.py, commits exactly once, and answers where the facts
landed: the slug and the definition's next revision. Rules come from the
same GalleryQuery-shaped inputs the gallery itself takes;
db/collection_rules.py owns every conversion and the browser never
constructs rule JSON.

Definition writes name the revision they edited: `expected_rev` in the
body, always -- deliberately not If-Match, because the page's ETag could
only honestly validate the whole representation (which changes with
membership) while the thing being claimed is the definition revision,
and a header token would also arrive unbound to the target in the URL.
A stale revision is a 409 with zero mutation -- the editor re-reads and
decides again.
"""

from __future__ import annotations

import time
from typing import Annotated, Literal

from litestar import patch, post, put
from litestar.datastructures import State
from litestar.exceptions import ClientException, HTTPException, NotFoundException
from litestar.params import FromPath
from litestar.response import Response
from pydantic import Field, RootModel

from db import collection_rules, collections, connect, naming
from db.resultset import canonical
from sg_web import collection_view
from sg_web.asking import gallery_query as _asked
from sg_web.presenting import VARIES
from sg_web.wire import Wire


def _collection_at(conn, slug: str) -> int:
    """The entity id for an address, retired spellings included -- a
    write through an old bookmark still means the same entity."""
    found = naming.resolve(conn, "collection", slug)
    if found is None:
        raise NotFoundException(f"no collection at /t/{slug}")
    return found[0]


def _written(state: State, work) -> Response[collection_view.CollectionWriteAnswer]:
    """Refusal mapping, one commit, and the small answer a write owes.

    Where to go and the definition's next concurrency token, and nothing
    more: answering with the whole management view -- the ResultSet page
    evaluated, the spans, the places and every legal parent move -- would
    re-run the collection's rule on every rename to build a body the
    browser reads one field out of.
    """
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
        return Response(collection_view.write_answer(conn, collection_id), headers=VARIES)
    finally:
        connect.close(conn)


class NewCollection(Wire):
    """The body of POST /albums: the listed kinds, born active at
    revision 1, optionally already placed and decorated."""

    name: str
    kind: collections.ListedKind = "album"
    parent: str | None = None
    color: str | None = None
    description: str | None = None


@post("/albums", sync_to_thread=True)
def make_album(state: State, data: NewCollection) -> Response[collection_view.CollectionWriteAnswer]:
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


class NewSmart(Wire):
    """The body of POST /albums/smart: a name, an optional cutoff, and
    the canonical spelling of the question being saved. The server
    reconstructs the typed rule through the same seams that own query
    semantics -- the browser never defines a rule shape."""

    name: str
    take: int | None = None
    folder: str | None = None
    person: str | None = None
    artifact: str | None = None
    kind: str | None = None
    favorite: str | None = None
    rating_min: int | None = None
    q: str | None = None
    sort: str | None = None
    parent: str | None = None
    color: str | None = None
    description: str | None = None
    #: the facets, one spelling or a list of them -- `f` repeats in the
    #: canonical question, and a view with two must save with two
    f: str | list[str] | None = None


@post("/albums/smart", sync_to_thread=True)
def make_smart(state: State, data: NewSmart) -> Response[collection_view.CollectionWriteAnswer]:
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
            artifact=data.artifact,
            favorite=data.favorite,
            rating_min=data.rating_min,
            facets=data.f,
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


class EditCollection(Wire):
    """A partial definition: only the facts the request names change.

    Absent, explicitly null and given are three DIFFERENT instructions --
    leave it, clear it, set it -- and pydantic keeps them apart without
    help: `model_fields_set` is the set of fields the request actually
    provided (pydantic docs/concepts/models.md:175). A default of None
    means "not said" only because the name is missing from that set, never
    because the value is None.

    `name` and `archived` accept null on the wire and are refused below it:
    a collection cannot be nameless, and a lifecycle is on or off. Refusing
    them here would answer 400 without saying which fact was impossible.
    """

    expected_rev: int
    name: str | None = None
    color: str | None = None
    description: str | None = None
    parent: str | None = None
    archived: bool | None = None


def _edit_definition(state: State, slug: str, data: EditCollection) -> Response[collection_view.CollectionWriteAnswer]:
    def work(conn):
        collection_id = _collection_at(conn, slug)
        said = data.model_fields_set
        held: dict = {
            field: getattr(data, field) for field in ("name", "color", "description", "archived") if field in said
        }
        if "parent" in said:
            if data.parent is None:
                held["parent_id"] = None
            else:
                found = naming.resolve(conn, "collection", data.parent)
                if found is None:
                    raise ValueError(f"no collection at /t/{data.parent} to move under")
                held["parent_id"] = found[0]
        collections.update_definition(
            conn, collection_id, collections.CollectionPatch(**held), state.actor_id, data.expected_rev, time.time()
        )
        return collection_id

    return _written(state, work)


@patch("/t/{slug:str}")
async def edit_definition(
    state: State, slug: FromPath[str], data: EditCollection
) -> Response[collection_view.CollectionWriteAnswer]:
    """The whole definition edit as one desired-state patch under one
    revision claim. Kind is deliberately not patchable -- changing how
    membership is decided is a transition, not a field.

    Async on purpose: GET /t/{slug} shares this path, and same-path
    handlers survive only as async ones (the albums_index note); the
    sqlite work crosses to a thread."""
    from anyio import to_thread

    return await to_thread.run_sync(_edit_definition, state, slug, data)


class _Question(Wire):
    """The GalleryQuery-shaped inputs a rule is minted from.

    The same spellings the save-view flow sends. The server reconstructs
    the typed rule through the seams that own query semantics; a browser
    never constructs rule JSON.
    """

    take: int | None = None
    folder: str | None = None
    person: str | None = None
    artifact: str | None = None
    q: str | None = None
    sort: str | None = None
    favorite: str | None = None
    rating_min: int | None = None
    #: the facets, one spelling or a list of them
    f: str | list[str] | None = None


class ReplaceRule(_Question):
    """The body of PUT /t/{slug}/rule. `kind` here is the MEDIA kind, the
    same one a gallery question carries -- a collection's own kind is not
    patchable, it is a conversion."""

    expected_rev: int
    kind: str | None = None


@put("/t/{slug:str}/rule", sync_to_thread=True)
def replace_rule(
    state: State, slug: FromPath[str], data: ReplaceRule
) -> Response[collection_view.CollectionWriteAnswer]:
    """This exact rule is now the collection's meaning: whole desired
    state, never predicate edits, under the same revision claim as any
    definition write."""

    def work(conn):
        collection_id = _collection_at(conn, slug)
        query = _asked(
            data.folder,
            None,
            data.kind,
            data.q,
            data.sort,
            None,
            person=data.person,
            artifact=data.artifact,
            favorite=data.favorite,
            rating_min=data.rating_min,
            facets=data.f,
        )
        rule = collection_rules.from_gallery_query(conn, query, actor_id=state.actor_id, take=data.take)
        spelled = canonical(query) or "the whole library"
        collections.replace_rule(conn, collection_id, rule, spelled, state.actor_id, data.expected_rev, time.time())
        return collection_id

    return _written(state, work)


class ConvertToListed(Wire):
    """Become an album or a flag.

    `discard_rule` is the rule's destruction said out loud, and it is only
    meaningful leaving smart; album <-> flag ignores it because there is
    no rule to lose.
    """

    kind: collections.ListedKind
    expected_rev: int
    discard_rule: bool = False


class ConvertToSmart(_Question):
    """Become smart, with the rule that makes it so in the same act.

    `media_kind`, not `kind`, carries the rule's media kind, because
    `kind` already names the collection kind being asked for.
    """

    kind: Literal["smart"]
    expected_rev: int
    media_kind: str | None = None


class ConvertCollection(RootModel[Annotated[ConvertToListed | ConvertToSmart, Field(discriminator="kind")]]):
    """The body of POST /t/{slug}/convert: which transition, and its terms.

    A discriminated union rather than one model holding both sets of
    fields, so a field the target kind cannot mean is refused by name
    instead of accepted and ignored -- `{"kind": "album", "q": "cat"}`
    answers 400 saying `album.q`, and a smart conversion carrying
    `discard_rule` says `smart.discard_rule`.

    A RootModel because litestar cannot take a bare union body: measured,
    a handler annotated `data: A | B` answers 500 to every request,
    including valid ones, since a body reaches pydantic only when the
    annotation IS a BaseModel subclass
    (litestar/plugins/pydantic/plugins/init.py is_pydantic_v2_model_class).
    A RootModel is one, so the union travels inside a class.
    """


@post("/t/{slug:str}/convert", sync_to_thread=True)
def convert_collection(
    state: State, slug: FromPath[str], data: ConvertCollection
) -> Response[collection_view.CollectionWriteAnswer]:
    """An explicit definition-mode transition. album<->flag moves
    freely; becoming smart requires an empty membership and a valid rule
    in this same operation; leaving smart requires the rule's discard
    said out loud, because the rule is authored state."""

    def work(conn):
        collection_id = _collection_at(conn, slug)
        wanted = data.root
        if isinstance(wanted, ConvertToSmart):
            query = _asked(
                wanted.folder,
                None,
                wanted.media_kind,
                wanted.q,
                wanted.sort,
                None,
                person=wanted.person,
                artifact=wanted.artifact,
                favorite=wanted.favorite,
                rating_min=wanted.rating_min,
                facets=wanted.f,
            )
            rule = collection_rules.from_gallery_query(conn, query, actor_id=state.actor_id, take=wanted.take)
            collections.convert_to_smart(
                conn,
                collection_id,
                rule,
                canonical(query) or "the whole library",
                state.actor_id,
                wanted.expected_rev,
                time.time(),
            )
        else:
            collections.convert_to_listed(
                conn,
                collection_id,
                wanted.kind,
                state.actor_id,
                wanted.expected_rev,
                time.time(),
                discard_rule=wanted.discard_rule,
            )
        return collection_id

    return _written(state, work)
