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

import dataclasses
import pathlib
import time
import urllib.parse
from typing import Annotated, Literal, TypedDict

from litestar import MediaType, Request, get
from litestar.datastructures import State
from litestar.exceptions import NotFoundException
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import FromPath, QueryParameter
from litestar.response import Redirect, Response, Template

from db import collection_rules, collections, connect, facets, naming, pages, resultset, settings
from sg_web import gallery, home, media_view
from sg_web.presenting import presented_page, wants_json
from sg_web.wire import Wire


class RuleView(Wire):
    """What was written down about a rule, kept as provenance whether or
    not it ever ran (db/collection_rules.py provenance). Both halves are
    nullable columns: a rule minted from a question carries no prose, and
    preserved prose carries no typed rule."""

    sql: str | None
    nl: str | None


class ChildCollection(Wire):
    """One level down the authored hierarchy."""

    slug: str
    name: str
    kind: collections.CollectionKind
    pictures: int


class FiledPicture(Wire):
    """A member as the legacy machine adapter names it."""

    slug: str
    name: str


class PlaceInCollection(Wire):
    """Where this collection's pictures happened, with the question that
    narrows the collection to that place."""

    id: int
    slug: str
    name: str
    kind: media_view.PlaceKind
    pictures: int
    qs: str


class GalleryPage(Wire):
    """One ResultSet page of the collection's membership."""

    items: list[gallery.ResultItem]
    total: int
    pages: int
    qs: str


class _Collection(Wire):
    """What every collection says about itself, whatever its kind and
    whatever its rule did. Never served: the five documents below are the
    shapes that cross."""

    slug: str
    name: str
    color: str | None
    description: str | None
    parent: str | None
    archived: bool
    definition_rev: int
    updated_at: float
    updated_by: str | None
    collections: list[ChildCollection]
    #: the legacy adapter's flat member list, present whether or not a
    #: rule ever ran -- these are the rows FILED in the collection
    files: list[FiledPicture]


class _Answered(_Collection):
    """A membership that produced an answer, so there are facts about it."""

    count: int
    first_seen: float | None
    last_seen: float | None
    timeline: str
    places: list[PlaceInCollection]
    gallery: GalleryPage


class _Unanswered(_Collection):
    """A rule that produced no answer.

    `count` and `gallery` are null rather than absent: a client asking how
    many members there are gets an answer, and the answer is "no number",
    not a missing key. The span, the timeline link and the places describe
    an answer, and there is none, so the variants below do not declare
    them at all -- reaching for `timeline` on a broken rule is a type
    error in the browser rather than undefined at runtime.
    """

    count: None = None
    gallery: None = None


class ListedCollection(_Answered):
    """An album or a flag: members are filed by hand, so the membership
    always evaluates and there is no rule to have a state."""

    kind: Literal["album", "flag"]


class SmartEvaluated(_Answered):
    """A typed rule that ran: a listed collection plus the rule and its
    condition, and nothing to explain."""

    kind: Literal["smart"]
    state: Literal["evaluated"]
    rule: RuleView | None


class SmartUnevaluated(_Unanswered):
    """Preserved prose, or nothing -- never run, so nothing to explain."""

    kind: Literal["smart"]
    state: Literal["unevaluated"]
    rule: RuleView | None


class SmartBroken(_Unanswered):
    """A rule naming an entity that is gone. It says which."""

    kind: Literal["smart"]
    state: Literal["broken"]
    rule: RuleView | None
    reason: str


class SmartUnavailable(_Unanswered):
    """A semantic rule nothing can answer right now. It says why."""

    kind: Literal["smart"]
    state: Literal["unavailable"]
    rule: RuleView | None
    reason: str


#: One collection at its address.
#:
#: A plain union, not `Field(discriminator=...)`: litestar builds a union's
#: schema itself and never asks pydantic for it
#: (litestar/_openapi/schema_generation/schema.py for_union_field), so the
#: OpenAPI `discriminator` object would be dropped and the annotation would
#: only be lying about what the document says. It is not needed anyway --
#: every variant states `kind`, and the smart four state `state`, as
#: single-valued enums, which is what the browser narrows on.
CollectionDocument = ListedCollection | SmartEvaluated | SmartUnevaluated | SmartBroken | SmartUnavailable


class CollectionWriteAnswer(Wire):
    """What a lifecycle write answers.

    The address to go to and the definition's next concurrency token, and
    that is the whole contract. Answering with the management view -- a
    ResultSet page, the spans, the places and every legal parent move --
    would assemble a body the browser reads `slug` out of and nothing else.
    """

    slug: str
    definition_rev: int


#: What a rule's evaluation came to. The same four words the documents
#: above spell, named once so Facts cannot hold a fifth.
RuleState = Literal["evaluated", "unevaluated", "broken", "unavailable"]


@dataclasses.dataclass(frozen=True)
class Answer:
    """The facts a membership that evaluated has.

    Internal, and not the `_Answered` wire base: that one carries every
    common field too, and building it here would mean assembling the whole
    document twice -- once to hold, once to splat into the variant.
    """

    count: int
    first_seen: float | None
    last_seen: float | None
    timeline: str
    places: list[PlaceInCollection]
    gallery: GalleryPage


@dataclasses.dataclass(frozen=True)
class Facts:
    """Everything one snapshot read about a collection.

    Internal, and deliberately not a Wire: it is the union of what the two
    audiences need and neither is told all of it. `files` and `parents`
    each cost a statement, so the caller says which it wants rather than
    paying for both.
    """

    slug: str
    name: str
    kind: collections.CollectionKind
    color: str | None
    description: str | None
    parent: str | None
    archived: bool
    definition_rev: int
    updated_at: float
    updated_by: str | None
    children: list[ChildCollection]
    rule: RuleView | None
    state: RuleState
    #: why the rule could not answer, when that is a thing to say
    reason: str | None
    answer: Answer | None
    files: list[FiledPicture]
    parents: list[dict]


def document_at(conn, models_dir: str, collection_id: int, slug: str, now: float) -> CollectionDocument:
    """The collection as a machine is told it, read and shaped.

    A caller asks for the representation it wants and nothing else: which
    statements that costs -- the filed members here, the parent picker for
    the page -- is this module's business, not the route's.
    """
    return document(_facts(conn, models_dir, collection_id, slug, now, files=True, parents=False))


def context_at(conn, models_dir: str, collection_id: int, slug: str, now: float) -> dict:
    """The collection as album.html is given it."""
    return context(_facts(conn, models_dir, collection_id, slug, now, files=False, parents=True))


def _facts(conn, models_dir: str, collection_id: int, slug: str, now: float, *, files: bool, parents: bool) -> Facts:
    """Read one collection inside ONE database snapshot.

    The ResultSet page is the FIRST read inside it -- its currency read
    precedes the snapshot pin -- and a rule-defined collection is the
    ResultSet's own typed refusal, decided under the same snapshot the card
    is then read from: a kind converted mid-request answers wholly as one
    generation, never a static header over a smart body. (An empty
    collection CAN legally convert; the schema only refuses converting one
    that holds filed members.)

    `parents` reads the parent picker's choices -- every active collection
    this one may legally move under -- so the browser never offers a move
    the database will refuse. `files` reads the legacy adapter's flat
    member list, the machine document's shape only, exactly as on the
    person and folder addresses.
    """
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
        rule = None
        if kind == "smart":
            held = collection_rules.provenance(conn, collection_id)
            rule = None if held is None else RuleView(sql=held["sql"], nl=held["nl"])
        filed = [FiledPicture(slug=s, name=n) for s, n in pages.album_files(conn, collection_id)] if files else []
        children = [
            ChildCollection(slug=s, name=n, kind=k, pictures=p)
            for _, s, n, k, p in pages.collection_children(conn, collection_id)
        ]
        answer = None
        if grid is not None:
            first_seen, last_seen = pages.collection_spans(conn).get(slug, (None, None))
            answer = Answer(
                count=grid["total"],
                first_seen=first_seen,
                last_seen=last_seen,
                timeline="/timeline?" + urllib.parse.urlencode([("album", slug)]),
                places=[
                    PlaceInCollection(
                        id=place_id,
                        slug=place_slug,
                        name=place_name,
                        kind=place_kind,
                        pictures=int(pictures),
                        qs=urllib.parse.urlencode(
                            [("album", slug), ("f", facets.spell(facets.facet("place.id", "eq", str(place_id))))]
                        ),
                    )
                    for place_id, place_slug, place_name, place_kind, pictures in pages.collection_places(
                        conn, collection_id
                    )
                ],
                gallery=GalleryPage(
                    items=gallery.result_items(grid["items"]),
                    total=grid["total"],
                    pages=grid["pages"],
                    qs=grid["qs"],
                ),
            )
        offered: list[dict] = []
        if parents:
            allowed = collections.eligible_parents(conn, collection_id)
            offered = [
                {"slug": s, "name": n, "archived": bool(a)} for s, n, a in pages.collections_named(conn, allowed)
            ]
        return Facts(
            slug=slug,
            name=name,
            kind=kind,
            color=color,
            description=description,
            parent=parent,
            archived=archived_at is not None,
            definition_rev=definition_rev,
            updated_at=updated_at,
            updated_by=updated_by,
            children=children,
            rule=rule,
            state=rule_state,
            reason=reason,
            answer=answer,
            files=filed,
            parents=offered,
        )


def document(held: Facts) -> CollectionDocument:
    """What a machine is told: the variant its kind and its rule's state
    make it, carrying no field that variant cannot have.

    Every field is named at every variant. A dict splatted into these
    constructors would read shorter and check nothing -- no checker can
    follow a heterogeneous mapping into a typed signature, and
    this function's whole job is to be the place where the facts become a
    contract.

    A state that cannot happen raises rather than picking something. The
    first draft answered `reason or ""` for a broken rule that had not said
    why, and called an unrecognised kind an album -- which would have shown
    a person an empty explanation and filed a stranger under the wrong
    kind, both silently. There is no honest default for either.
    """
    answer = held.answer
    if answer is None and held.state == "evaluated":
        raise AssertionError(f"/t/{held.slug} evaluated its rule and produced no answer")
    if answer is not None and held.state != "evaluated":
        raise AssertionError(f"/t/{held.slug} answered while its rule was {held.state}")
    if answer is not None and held.kind == "smart":
        return SmartEvaluated(
            kind="smart",
            state="evaluated",
            rule=held.rule,
            slug=held.slug,
            name=held.name,
            color=held.color,
            description=held.description,
            parent=held.parent,
            archived=held.archived,
            definition_rev=held.definition_rev,
            updated_at=held.updated_at,
            updated_by=held.updated_by,
            collections=held.children,
            files=held.files,
            count=answer.count,
            first_seen=answer.first_seen,
            last_seen=answer.last_seen,
            timeline=answer.timeline,
            places=answer.places,
            gallery=answer.gallery,
        )
    if answer is not None:
        if held.kind == "smart":  # unreachable: the branch above took it
            raise AssertionError(f"/t/{held.slug} is smart and was not answered as one")
        return ListedCollection(
            kind=held.kind,
            slug=held.slug,
            name=held.name,
            color=held.color,
            description=held.description,
            parent=held.parent,
            archived=held.archived,
            definition_rev=held.definition_rev,
            updated_at=held.updated_at,
            updated_by=held.updated_by,
            collections=held.children,
            files=held.files,
            count=answer.count,
            first_seen=answer.first_seen,
            last_seen=answer.last_seen,
            timeline=answer.timeline,
            places=answer.places,
            gallery=answer.gallery,
        )
    if held.state == "broken":
        reason = held.reason
        if not reason:
            raise AssertionError(f"/t/{held.slug} is broken and did not say why")
        return SmartBroken(
            kind="smart",
            state="broken",
            rule=held.rule,
            reason=reason,
            slug=held.slug,
            name=held.name,
            color=held.color,
            description=held.description,
            parent=held.parent,
            archived=held.archived,
            definition_rev=held.definition_rev,
            updated_at=held.updated_at,
            updated_by=held.updated_by,
            collections=held.children,
            files=held.files,
        )
    if held.state == "unavailable":
        reason = held.reason
        if not reason:
            raise AssertionError(f"/t/{held.slug} is unavailable and did not say why")
        return SmartUnavailable(
            kind="smart",
            state="unavailable",
            rule=held.rule,
            reason=reason,
            slug=held.slug,
            name=held.name,
            color=held.color,
            description=held.description,
            parent=held.parent,
            archived=held.archived,
            definition_rev=held.definition_rev,
            updated_at=held.updated_at,
            updated_by=held.updated_by,
            collections=held.children,
            files=held.files,
        )
    return SmartUnevaluated(
        kind="smart",
        state="unevaluated",
        rule=held.rule,
        slug=held.slug,
        name=held.name,
        color=held.color,
        description=held.description,
        parent=held.parent,
        archived=held.archived,
        definition_rev=held.definition_rev,
        updated_at=held.updated_at,
        updated_by=held.updated_by,
        collections=held.children,
        files=held.files,
    )


def write_answer(conn, collection_id: int) -> CollectionWriteAnswer:
    """What a lifecycle write hands back, read after the commit.

    Here rather than in the authoring adapter because this module owns the
    collection's representations, and all three of them: an adapter that
    assembled one itself would be reading db.pages to do it.
    """
    live = naming.entity_slug(conn, collection_id)
    definition_rev = pages.collection_card(conn, collection_id)[6]
    return CollectionWriteAnswer(slug=live[1] if live else "", definition_rev=definition_rev)


def context(held: Facts) -> dict:
    """What album.html is given.

    The document a machine gets, plus the parent picker's choices -- the
    one audience whose shape is not a wire contract, because a template
    reads what it reads and no client is typed against it.
    """
    told = document(held).model_dump(mode="json")
    told["parents"] = held.parents
    return told


class CollectionListed(Wire):
    """One collection in the flat list `/albums` serves to machines.

    `first_seen` and `last_seen` are the span of the pictures filed in it.
    The archived shelf computes no spans, so it answers null for both --
    NOT by omitting the keys, which would be a difference a client has to
    discover. Both lists are one representation; a shelf row carries the
    span keys because a listed collection has them.
    """

    name: str
    slug: str
    kind: str
    pictures: int
    first_seen: float | None = None
    last_seen: float | None = None
    #: The newest member's thumbnail. This is a library of PHOTOGRAPHS and
    #: the shelf was a list of words; a collection should show what is in
    #: it. None for a rule-defined collection, which holds a question
    #: rather than files, and for one whose members are all missing.
    cover: str | None = None


def _albums_listed(db_path: str) -> list[CollectionListed]:
    from vision import thumbs

    conn = connect.connect(db_path)
    try:
        spans = pages.collection_spans(conn)
        covers = pages.collection_covers(conn)
        return [
            CollectionListed(
                name=name,
                slug=slug,
                kind=kind,
                pictures=pictures,
                first_seen=spans.get(slug, (None, None))[0],
                last_seen=spans.get(slug, (None, None))[1],
                cover=_cover_url(thumbs, covers.get(slug)),
            )
            for name, slug, kind, pictures in pages.albums(conn)
        ]
    finally:
        connect.close(conn)


def _cover_url(thumbs, held: tuple[str | None, str, str] | None) -> str | None:
    """Resolved ONCE per collection, here, the way the grid resolves its
    cells: the content-addressed asset when the bytes have been hashed,
    the slug route when they have not."""
    if held is None:
        return None
    sha, file_slug, kind = held
    return thumbs.asset_url(sha, file_slug, medium=kind)


def _albums_archived(db_path: str) -> list[CollectionListed]:
    conn = connect.connect(db_path, read_only=True)
    try:
        return [
            CollectionListed(name=name, slug=slug, kind=kind, pictures=pictures)
            for name, slug, kind, pictures in pages.archived_albums(conn)
        ]
    finally:
        connect.close(conn)


class _Shelved(TypedDict):
    """One collection on the shelf, and the collections under it.

    Spelled out because a literal holding a str, an int, a None and a
    list infers every key as the union of those, and the tree is built
    by `nodes[parent]["collections"].append(...)` -- `.append` on
    `str | int | None | list`.
    """

    slug: str
    name: str
    kind: str
    pictures: int
    first_seen: float | None
    last_seen: float | None
    collections: list[_Shelved]
    #: The newest member's thumbnail; None for a rule-defined node, which
    #: holds a question rather than files. A shelf in a library of
    #: photographs shows photographs.
    cover: str | None


def _albums_nested(db_path: str) -> tuple[list[_Shelved], int]:
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
        covers = pages.collection_covers(conn)
    finally:
        connect.close(conn)
    from vision import thumbs

    nodes: dict[int, _Shelved] = {
        cid: {
            "slug": slug,
            "name": name,
            "kind": kind,
            "pictures": pictures,
            "first_seen": spans.get(slug, (None, None))[0],
            "last_seen": spans.get(slug, (None, None))[1],
            "collections": [],
            "cover": _cover_url(thumbs, covers.get(slug)),
        }
        for cid, _, slug, name, kind, pictures in rows
    }
    top: list[_Shelved] = []
    for cid, parent_id, *_ in rows:  # row order IS name order within each parent
        if parent_id in nodes:
            nodes[parent_id]["collections"].append(nodes[cid])
        else:
            top.append(nodes[cid])
    return top, retired


@get(
    "/albums",
    # The route negotiates: a browser gets a page, a machine gets this list.
    # The return annotation can only say `Template | Response`, which tells
    # OpenAPI nothing, so the JSON body is declared here instead -- otherwise
    # the one shape a client actually parses would be the one shape the
    # contract did not describe.
    responses={
        200: ResponseSpec(
            data_container=list[CollectionListed],
            description="Every active collection, or the archived shelf under ?state=archived",
            media_type=MediaType.JSON,
            # ResponseSpec invents examples by default. They are deterministic,
            # so the drift gate does not flap, but they are made-up values in a
            # committed contract -- a reader should not have to work out that
            # `"kind": "VvHsoVSEeCtLViFvDEMp"` was never a kind.
            generate_examples=False,
        )
    },
)
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
        now = time.time()
        if json_wanted:
            told = document_at(conn, weights, collection_id, slug, now)
        else:
            told = context_at(conn, weights, collection_id, slug, now)
    finally:
        connect.close(conn)
    return presented_page(request, told, page="album.html", context={"album": told})


@get(
    "/t/{slug:str}",
    responses={
        200: ResponseSpec(
            data_container=CollectionDocument,
            description="The collection: a listed one, or a smart one in the state its rule reached",
            media_type=MediaType.JSON,
            generate_examples=False,
        )
    },
)
async def album_page(
    state: State, request: Request, slug: FromPath[str]
) -> Template | Response[CollectionDocument] | Redirect:
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
