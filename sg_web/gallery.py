"""The gallery grid and its rail: presentation adapters over db/resultset.py.

Every route here maps request-shaped inputs onto ONE GalleryQuery and
reads the module's materialized answer. No SQL, no sorting, no
membership opinions -- a route that grew any of those would be the
second notion of truth the ResultSet contract exists to forbid.

The URL owns canonical query state: `/g?q=...&folder=...&sort=...&page=N`
renders complete from cold, so reload, back, forward and bookmarks all
work without client state. The fragments and JSON exist for the parts a
running page swaps live -- the grid on a page change, the rail popover
on hover -- and answer from the same query the shell was drawn from.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
from typing import Literal

from litestar import MediaType, get
from litestar.datastructures import State
from litestar.exceptions import ClientException, HTTPException, NotFoundException
from litestar.params import FromPath, FromQuery
from litestar.response import Template

from db import analysis, catalog, connect, discovery, naming, pages, places, resultset, settings, vocabulary
from db import facets as facets_module
from db import views as views_module
from sg_web import home
from sg_web.asking import gallery_query as _asked
from sg_web.wire import Wire

#: The presentations one answer has. `view` is NOT part of the question:
#: it never reaches the GalleryQuery, never moves the fingerprint, and
#: switching between them must leave the membership and the total exactly
#: where they were. It rides the URL so a link to an analysis is a link.
VIEWS = ("gallery", "table", "analyze")


#: The table's sortable columns: the heading, and which way the FIRST
#: click orders it.
#:
#: Text ascending, numbers descending, because they mean different
#: things. "Sort by name" means start at A; "sort by size" almost always
#: means show me what is huge, and opening on the smallest file in the
#: library is a click somebody has to undo every time. The second click
#: reverses either way and the heading says which way it is.
#:
#: Every column the table draws, including the LEFT-JOINed ones. A sort
#: by a column most files lack does not narrow the answer -- they order
#: last and say so by position -- because narrowing would change what
#: the answer holds with no chip on screen admitting it.
TABLE_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("name", "name", False),
    ("kind", "kind", False),
    ("size", "size", True),
    ("pixels", "pixels", True),
    ("length", "length", True),
    # the recipe
    ("checkpoint", "checkpoint", False),
    ("sampler", "sampler", False),
    ("steps", "steps", True),
    ("cfg", "cfg", True),
    ("seed", "seed", True),
    # the camera
    ("camera", "camera", False),
    ("iso", "iso", True),
    ("f_number", "f", True),
    ("focal_length", "mm", True),
    # and what a person said, which is theirs: "sort by rating" means
    # sort by MINE, so the actor binds into the ordering statement.
    ("rating", "&starf;", True),
)


def _column_sorts(query: resultset.GalleryQuery, view: str) -> dict[str, dict]:
    """Per sortable column: where its heading points, and which way this
    answer is currently ordered by it.

    Spelled by the server because the sort is part of the QUESTION --
    it changes which pictures are on page one, so a click that reordered
    the rows already fetched would be a table disagreeing with its own
    pager. Reload, Back and a shared link all land on the same order for
    the same reason every other filter does.
    """
    made: dict[str, dict] = {}
    for name, label, biggest_first in TABLE_COLUMNS:
        down = f"{name}-desc"
        # Which way it is ordered NOW, or None when this is not the sort.
        held = "asc" if query.sort == name else "desc" if query.sort == down else None
        # Clicking reverses what is held, and otherwise opens the way
        # this column is usually wanted.
        wanted = name if held == "desc" else down if held == "asc" else (down if biggest_first else name)
        spelled = resultset.canonical(dataclasses.replace(query, sort=wanted))
        if view != "gallery":
            spelled = f"{spelled}&view={view}" if spelled else f"view={view}"
        made[name] = {"label": label, "href": f"/g?{spelled}" if spelled else "/g", "held": held}
    return made


def _with_clause(query: resultset.GalleryQuery, key: str, value: str, view: str) -> str:
    """The question with one more clause, canonically spelled.

    What makes an analysis navigation rather than a report. A count that
    cannot be clicked back into the query is a dashboard, and a dashboard
    is where data goes to be looked at instead of used.
    """
    one = vocabulary.dimension(key)
    if one is None:
        return resultset.canonical(query)
    if one.carried == "scope":
        held = resultset.with_scope(query, key, value)
    else:
        made = facets_module.facet(key, one.ops[0], value)
        held = dataclasses.replace(
            query,
            facets=facets_module.normalized(
                [*[facets_module.spell(f) for f in query.facets], facets_module.spell(made)]
            ),
        )
    spelled = resultset.canonical(held)
    if view != "gallery":
        spelled = f"{spelled}&view={view}" if spelled else f"view={view}"
    return spelled


def _split_out(query: resultset.GalleryQuery, slug: str, rest: str, view: str) -> str:
    """The same question, asked as a person and a phrase.

    The phrase keeps its ranking when there is one left to rank; with
    nothing left, "Sarah" alone is a person's pictures and ordering them
    by similarity to an empty phrase is not a thing.
    """
    import dataclasses as _dataclasses

    held = _dataclasses.replace(query, person=slug, text=rest or None)
    if not rest:
        held = _dataclasses.replace(held, sort="newest")
    spelled = resultset.canonical(held)
    if view != "gallery":
        spelled = f"{spelled}&view={view}" if spelled else f"view={view}"
    return f"/g?{spelled}" if spelled else "/g"


def _with_text(query: resultset.GalleryQuery, said: str, view: str) -> str:
    """The question this answer would become, searched for one term.

    The phrase REPLACES whatever text the question carried rather than
    joining it: two phrases would be a narrower question than either,
    and a person clicking a term in a breakdown means "this one", not
    "this one as well as the last one I clicked".
    """
    import dataclasses as _dataclasses

    # `similarity`, which is what a ranked text answer is called here --
    # and it is the only sort that REQUIRES a phrase, so it can only be
    # asked for alongside one. The question keeps whatever else it said.
    spelled = resultset.canonical(_dataclasses.replace(query, text=said, sort="similarity"))
    if view != "gallery":
        spelled = f"{spelled}&view={view}" if spelled else f"view={view}"
    return f"/g?{spelled}" if spelled else "/g"


def _analysis(conn, query: resultset.GalleryQuery, total: int, weights: str, view: str) -> dict:
    """The answer, described -- and every row carrying the question it
    would make."""
    told = analysis.analyze(conn, query, total, models_dir=weights, now=time.time())
    return {
        "breakdowns": [
            {
                "key": one.key,
                "label": one.label,
                "covered": one.covered,
                "more": one.more,
                "rows": [
                    {
                        "label": row.label,
                        "count": row.count,
                        "share": row.share,
                        "chosen": row.chosen,
                        "qs": _with_clause(query, one.key, row.value, view),
                    }
                    for row in one.rows
                ],
            }
            for one in told.breakdowns
        ],
        "prompts": [{"id": one.id, "text": one.text, "uses": one.uses, "role": one.role} for one in told.prompts],
        "more_prompts": told.more_prompts,
        # Each term carries the question it would make: clicking one
        # narrows to the files whose prompt says it. `q` rather than a
        # facet, because a term is text inside a prompt and the text
        # search is what reads inside prompts.
        "terms": [
            {"term": one.term, "files": one.files, "qs": _with_text(query, one.term, view)} for one in told.terms
        ],
        "more_terms": told.more_terms,
        "loras": [
            {
                "name": one.name,
                "uses": one.uses,
                "typical": one.typical,
                "lowest": one.lowest,
                "highest": one.highest,
                "qs": _with_clause(query, "generation.lora", str(one.id), view),
            }
            for one in told.loras
        ],
    }


def _grid_context(state: State, query: resultset.GalleryQuery, page: int, view: str = "gallery") -> dict:
    from vision import thumbs

    conn = connect.connect(state.db_path)
    try:
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        try:
            shape = resultset.page(conn, weights, query, page, time.time(), actor_id=state.actor_id)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()  # a semantic answer may have minted registry rows
        # an id-valued chip says the name, not the number
        named = discovery.labels(conn, query)
        # only when asked: an analysis is a dozen aggregates, and
        # nobody drawing a grid is waiting for them
        described = _analysis(conn, query, shape["total"], weights, view) if view == "analyze" else None
        # The same members the grid would show, as facts rather than
        # pictures. One read over the ids the ResultSet already
        # returned -- never a second query about which files those are.
        listed = (
            pages.table_of(conn, [item["id"] for item in shape["items"]], state.actor_id) if view == "table" else None
        )
        if listed:
            for row in listed:
                row["thumb"] = thumbs.asset_url(row.get("sha"), row["slug"], medium=row["kind"])
        # Read INSIDE the connection, like everything else here: the
        # context dict below is assembled after `close`, so anything
        # asked for down there is asked of a closed database.
        remembered = views_module.all_of(conn)
        # "Sarah at the beach" is what somebody types, and the ranking
        # is over image embeddings: the text encoder has never heard of
        # Sarah and never will. The answer is not a better caption, it
        # is the question splitting into a filter the vocabulary already
        # has plus the phrase that is left.
        #
        # Offered, never applied. Rewriting a typed question silently is
        # how somebody stops trusting what the box does, and this
        # application says what a question is with visible chips.
        splitting = None
        if query.text and not query.person:
            found = discovery.person_in(conn, query.text)
            if found is not None:
                name, slug, rest = found
                splitting = {
                    "name": name,
                    "slug": slug,
                    "rest": rest,
                    "qs": _split_out(query, slug, rest, view),
                }
    finally:
        connect.close(conn)
    provenance = shape["provenance"] or {}
    # a caption that mentions no word of the phrase is the ordinary
    # outcome of a word match (retrieval's `unmatched`), said quietly
    unmatched = (provenance.get("unmatched") or {}).get("captions")
    # Where each cell points, resolved ONCE for the page. The ResultSet
    # already carried the content hash out of its own row read, so this
    # is arithmetic rather than a query -- and it is what stops sixty
    # cells being sixty connections (vision/thumbs.py `asset_url`).
    thumbs.address(shape["items"])
    return {
        "items": shape["items"],
        "page": shape["page"],
        "pages": shape["pages"],
        "total": shape["total"],
        "size": query.size,
        "sort": query.sort,
        "currency": shape["currency"],
        "answer": shape["answer"],
        "missing_spaces": provenance.get("missing") or {},
        "captions_unmatched": unmatched,
        "answered_by": provenance.get("contributors") or [],
        # What the cut cost. A phrase ranks every file a space holds and
        # `head` keeps the ones standing above the middle of what it
        # said (db/retrieval.py); without these three the page looks
        # like a small library rather than an answer that ended.
        "ranked": provenance.get("ranked"),
        "answering": provenance.get("answering"),
        "depth": query.depth if query.text else None,
        # The same question at the other depth, both spelled here: `qs`
        # is whichever one is being looked at, so a link back to the
        # other cannot be built from it.
        "whole_qs": resultset.canonical(dataclasses.replace(query, depth="all")) if query.text else None,
        "head_qs": resultset.canonical(dataclasses.replace(query, depth="head")) if query.text else None,
        "q": query.text or "",
        "folder": query.folder or "",
        "album": query.album or "",
        "person": query.person or "",
        "artifact": query.artifact or "",
        "kind": query.kind or "",
        "favorite": "" if query.favorite is None else ("1" if query.favorite else "0"),
        "rating_min": query.rating_min or "",
        "facets": [facets_module.spell(held) for held in query.facets],
        "place_kinds": list(places.KINDS),
        "chips": _chips(query, named),
        "qs": shape["qs"],
        "view": view,
        "views": VIEWS,
        # The questions somebody asked to be reminded of, carried on the
        # first paint: a remembered question is a way IN, and a list that
        # arrives after the answer is a list nobody used to get to it.
        "remembered": remembered,
        #: A person's name spotted in the phrase, and the question it
        #: would become. None when the phrase names nobody, or when the
        #: question already says who.
        "splitting": splitting,
        "analysis": described,
        "table": listed,
        "columns": _column_sorts(query, view),
        "kinds": resultset.KINDS,
        "sorts": resultset.SORTS,
        # the filter surface, drawn from the one vocabulary: the
        # sections and their dimensions, and how many clauses the
        # question already carries per dimension. The VALUES are not
        # here -- counting thirty dimensions to draw a closed drawer
        # is thirty queries nobody asked for; each section fetches its
        # own from /g/options when somebody opens it.
        "filter_groups": [
            {"name": name, "label": label, "dimensions": held}
            # `asked_kind`, not `query.kind`: the question can say which
            # medium in two places now -- the scope every bookmark
            # carries, and the facet the drawer writes so kinds can be
            # OR'd -- and which dimensions apply must follow both.
            for name, label, held in vocabulary.grouped(discovery.asked_kind(query))
        ],
        "filter_counts": discovery.counts(query),
        "filters_held": sum(discovery.counts(query).values()),
    }


def _chips(query: resultset.GalleryQuery, named: dict[str, dict[int, str]] | None = None) -> list[dict]:
    """Every clause the question carries, as a chip that reads as words
    and carries the question that remains when it is removed.

    Both halves come from elsewhere on purpose. What a clause is CALLED
    is db/vocabulary.py's -- this module used to hold a private dict of
    labels beside the registry that held the predicates, and the two
    drifted, which is how a chip ends up printing a key. What "the
    question without this" MEANS is db/discovery.py's `without`, the
    same function that decides what a dimension's own option counts are
    taken against, so a remove link and a count cannot disagree.
    """
    made = []
    for one in vocabulary.DIMENSIONS:
        held = discovery.chosen_values(query, one.key)
        if not held:
            continue
        if one.carried == "scope":
            rest = discovery.without(query, one.key)
            value = getattr(query, one.key)
            op = one.ops[0]
            made.append(
                {
                    "key": one.key,
                    "spelled": f"{one.key}={held[0]}",
                    "label": vocabulary.chip(one, op, value, named and named.get(one.key)),
                    "remove_qs": resultset.canonical(rest),
                }
            )
            continue
        held = [facet for facet in query.facets if facet.key == one.key]
        # An OR GROUP is one thing the question says, so it is one chip
        # that removes as one. Rendering `kind image` beside `kind video`
        # would read exactly like two ANDed clauses -- the opposite
        # question, and one that answers nothing.
        ored = [facet for facet in held if facet.op == facets_module.ANY]
        if ored:
            rest = dataclasses.replace(query, facets=tuple(other for other in query.facets if other not in ored))
            made.append(
                {
                    "key": one.key,
                    "spelled": facets_module.spell(ored[0]),
                    "label": vocabulary.chip_any(one, [f.value for f in ored], named and named.get(one.key)),
                    "remove_qs": resultset.canonical(rest),
                }
            )
        # Repeated `eq` on a dimension a file can hold several of means
        # ALL of them, which is also one thing said once -- and saying it
        # as separate chips reads identically to an OR.
        anded = [facet for facet in held if facet.op != facets_module.ANY]
        if one.multi == "both" and len(anded) > 1:
            rest = dataclasses.replace(query, facets=tuple(other for other in query.facets if other not in anded))
            made.append(
                {
                    "key": one.key,
                    "spelled": facets_module.spell(anded[0]),
                    "label": vocabulary.chip_all(one, [f.value for f in anded], named and named.get(one.key)),
                    "remove_qs": resultset.canonical(rest),
                }
            )
            continue
        for facet in anded:
            rest = dataclasses.replace(query, facets=tuple(other for other in query.facets if other != facet))
            made.append(
                {
                    "key": one.key,
                    "spelled": facets_module.spell(facet),
                    "label": vocabulary.chip(one, facet.op, facet.value, named and named.get(one.key)),
                    "remove_qs": resultset.canonical(rest),
                }
            )
    return made


@get("/g", sync_to_thread=True)
def gallery(
    state: State,
    folder: FromQuery[str | None] = None,
    album: FromQuery[str | None] = None,
    person: FromQuery[str | None] = None,
    artifact: FromQuery[str | None] = None,
    kind: FromQuery[str | None] = None,
    favorite: FromQuery[str | None] = None,
    rating_min: FromQuery[int | None] = None,
    q: FromQuery[str | None] = None,
    f: FromQuery[list[str] | None] = None,
    sort: FromQuery[str | None] = None,
    depth: FromQuery[str | None] = None,
    size: FromQuery[int | None] = None,
    page: FromQuery[int] = 1,
    view: FromQuery[str] = "gallery",
) -> Template:
    """One answer, in whichever presentation was asked for, from nothing
    but the URL.

    `view` is presentation, never the question: it does not reach the
    GalleryQuery and does not move the fingerprint, so switching from the
    grid to the analysis and back leaves membership and total exactly
    where they were. That is the whole contract between them.
    """
    if view not in VIEWS:
        raise ClientException(f"view is one of {', '.join(VIEWS)}, not {view!r}")
    query = _asked(
        folder,
        album,
        kind,
        q,
        sort,
        size,
        person=person,
        artifact=artifact,
        favorite=favorite,
        rating_min=rating_min,
        facets=f,
        depth=depth,
    )
    return Template(
        media_type=MediaType.HTML, template_name="gallery.html", context=_grid_context(state, query, page, view)
    )


@get("/g/grid", sync_to_thread=True)
def grid_fragment(
    state: State,
    folder: FromQuery[str | None] = None,
    album: FromQuery[str | None] = None,
    person: FromQuery[str | None] = None,
    artifact: FromQuery[str | None] = None,
    kind: FromQuery[str | None] = None,
    favorite: FromQuery[str | None] = None,
    rating_min: FromQuery[int | None] = None,
    q: FromQuery[str | None] = None,
    f: FromQuery[list[str] | None] = None,
    sort: FromQuery[str | None] = None,
    depth: FromQuery[str | None] = None,
    size: FromQuery[int | None] = None,
    page: FromQuery[int] = 1,
) -> Template:
    """One page of cells, for the running page to swap in place."""
    query = _asked(
        folder,
        album,
        kind,
        q,
        sort,
        size,
        person=person,
        artifact=artifact,
        favorite=favorite,
        rating_min=rating_min,
        facets=f,
        depth=depth,
    )
    return Template(media_type=MediaType.HTML, template_name="_grid.html", context=_grid_context(state, query, page))


class FilterOption(Wire):
    """One value a dimension could take, and what it would leave."""

    #: as the URL spells it
    value: str
    #: as a person reads it
    label: str
    #: how many media it would leave, FROM THE REST OF THIS QUESTION --
    #: this dimension's own clauses removed first, so the list can
    #: broaden the question and not only narrow it
    count: int
    #: whether the question already carries it
    chosen: bool


class FilterOptions(Wire):
    """One dimension's list, ready to draw."""

    key: str
    label: str
    #: what it means, where the label alone would mislead
    note: str
    value_kind: str
    ops: list[str]
    #: How choosing SEVERAL reads: "" one at a time, "any" OR'd, "both"
    #: OR'd or ANDed at the person's choosing. A fact about the
    #: dimension, not a preference (db/vocabulary.py `multi`), so the
    #: surface is told rather than deciding.
    multi: str
    options: list[FilterOption]
    #: how many values were not returned. Never silently zero: a
    #: truncated list that does not say so reads as a complete one, and
    #: then a model that IS in the library looks absent.
    more: int


@get("/g/options", sync_to_thread=True)
def filter_options(
    state: State,
    key: FromQuery[str],
    folder: FromQuery[str | None] = None,
    album: FromQuery[str | None] = None,
    person: FromQuery[str | None] = None,
    artifact: FromQuery[str | None] = None,
    kind: FromQuery[str | None] = None,
    favorite: FromQuery[str | None] = None,
    rating_min: FromQuery[int | None] = None,
    q: FromQuery[str | None] = None,
    f: FromQuery[list[str] | None] = None,
    sort: FromQuery[str | None] = None,
    depth: FromQuery[str | None] = None,
    size: FromQuery[int | None] = None,
    search: FromQuery[str | None] = None,
) -> FilterOptions:
    """What one dimension could be narrowed to from here, counted.

    Counted through db/resultset.py `scope_of`, so this and the grid
    cannot come to disagree about which media the question holds; and
    counted with THIS dimension's own clauses removed, so opening the
    list a person came to change their mind with can widen it.
    """
    query = _asked(
        folder,
        album,
        kind,
        q,
        sort,
        size,
        person=person,
        artifact=artifact,
        favorite=favorite,
        rating_min=rating_min,
        facets=f,
        depth=depth,
    )
    one = vocabulary.dimension(key)
    if one is None:
        raise NotFoundException(f"there is no filter named {key!r}")
    conn = connect.connect(state.db_path)
    try:
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        try:
            told = discovery.options(
                conn, query, key, actor_id=state.actor_id, models_dir=weights, now=time.time(), search=search
            )
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
    finally:
        connect.close(conn)
    return FilterOptions(
        key=told.key,
        label=told.label,
        note=one.note,
        value_kind=one.value_kind,
        ops=list(one.ops),
        multi=one.multi,
        options=[
            FilterOption(value=each.value, label=each.label, count=each.count, chosen=each.chosen)
            for each in told.options
        ],
        more=told.more,
    )


class CatalogField(Wire):
    """One filterable fact, as the Add-filter list shows it."""

    #: what the URL carries: a dimension's own key, or `param.is`
    key: str
    #: for a discovered field, the raw metadata key its clause names
    param: str | None
    label: str
    #: the section it belongs to, or the source that wrote it
    group: str
    value_kind: str
    ops: list[str]
    multi: str
    note: str
    #: whether this application understands the fact or merely recorded
    #: that some tool wrote it. The surface builds a different control
    #: for each; the PERSON is never shown the distinction.
    curated: bool
    #: media in this answer carrying it, and how many values they hold
    #: between them -- the two numbers the ranking is made of, sent so a
    #: surface can say "412 files, 6 values" rather than only ordering.
    covered: int
    values: int
    #: how many positional members collapsed into this one
    repeats: int


class Catalog(Wire):
    """What the Add-filter box answers with."""

    fields: list[CatalogField]
    #: how many more matched. Never silently zero: a truncated list that
    #: does not say so reads as a complete one, and then a field that IS
    #: in the library looks absent.
    more: int


@get("/g/fields", sync_to_thread=True)
def filter_catalog(
    state: State,
    folder: FromQuery[str | None] = None,
    album: FromQuery[str | None] = None,
    person: FromQuery[str | None] = None,
    artifact: FromQuery[str | None] = None,
    kind: FromQuery[str | None] = None,
    favorite: FromQuery[str | None] = None,
    rating_min: FromQuery[int | None] = None,
    q: FromQuery[str | None] = None,
    f: FromQuery[list[str] | None] = None,
    sort: FromQuery[str | None] = None,
    depth: FromQuery[str | None] = None,
    size: FromQuery[int | None] = None,
    search: FromQuery[str | None] = None,
) -> Catalog:
    """Every fact this answer can be asked about, best first.

    The whole question rides the query string, exactly as `/g/options`
    takes it, because the ranking is counted WITHIN the answer being
    looked at -- "what could I ask next about these" is a different list
    from "what does this library contain", and only the first is worth
    reading.
    """
    query = _asked(
        folder,
        album,
        kind,
        q,
        sort,
        size,
        person=person,
        artifact=artifact,
        favorite=favorite,
        rating_min=rating_min,
        facets=f,
        depth=depth,
    )
    conn = connect.connect(state.db_path)
    try:
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        found, more = catalog.catalog(
            conn,
            query,
            search=search or "",
            actor_id=state.actor_id,
            models_dir=weights,
            now=time.time(),
        )
        conn.commit()
    finally:
        connect.close(conn)
    return Catalog(
        fields=[
            CatalogField(
                key=one.key,
                param=one.param,
                label=one.label,
                group=one.group,
                value_kind=one.value_kind,
                ops=list(one.ops),
                multi=one.multi,
                note=one.note,
                curated=one.curated,
                covered=one.covered,
                values=one.values,
                repeats=one.repeats,
            )
            for one in found
        ],
        more=more,
    )


class FieldValues(Wire):
    """What one discovered key holds across this answer."""

    param: str
    #: value and how many media here carry it, most used first
    options: list[FilterOption]
    #: how many values were not returned -- never silently zero
    more: int


@get("/g/fields/values", sync_to_thread=True)
def field_values(
    state: State,
    param: FromQuery[str],
    folder: FromQuery[str | None] = None,
    album: FromQuery[str | None] = None,
    person: FromQuery[str | None] = None,
    artifact: FromQuery[str | None] = None,
    kind: FromQuery[str | None] = None,
    favorite: FromQuery[str | None] = None,
    rating_min: FromQuery[int | None] = None,
    q: FromQuery[str | None] = None,
    f: FromQuery[list[str] | None] = None,
    sort: FromQuery[str | None] = None,
    depth: FromQuery[str | None] = None,
    size: FromQuery[int | None] = None,
    search: FromQuery[str | None] = None,
) -> FieldValues:
    """The values a key nothing here named actually takes.

    `/g/options` answers this for a CURATED dimension, from the
    statement the vocabulary carries per dimension. The long tail has no
    statement per key -- it has one statement for every key, and this is
    the route over it. The two are deliberately separate addresses: one
    takes a dimension, the other takes a metadata key, and collapsing
    them would mean a `key` parameter that means two things.
    """
    query = _asked(
        folder,
        album,
        kind,
        q,
        sort,
        size,
        person=person,
        artifact=artifact,
        favorite=favorite,
        rating_min=rating_min,
        facets=f,
        depth=depth,
    )
    held = {one.value for one in query.facets if one.key == "param.is"}
    conn = connect.connect(state.db_path)
    try:
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        found, more = catalog.values(
            conn,
            query,
            param,
            actor_id=state.actor_id,
            models_dir=weights,
            now=time.time(),
            search=search or "",
        )
        conn.commit()
    finally:
        connect.close(conn)
    return FieldValues(
        param=param,
        options=[
            FilterOption(value=value, label=value, count=count, chosen=f"{param}={value}" in held)
            for value, count in found
        ],
        more=more,
    )


@get("/g/peek", sync_to_thread=True)
def rail_peek(
    state: State,
    page: FromQuery[int],
    folder: FromQuery[str | None] = None,
    album: FromQuery[str | None] = None,
    person: FromQuery[str | None] = None,
    artifact: FromQuery[str | None] = None,
    kind: FromQuery[str | None] = None,
    favorite: FromQuery[str | None] = None,
    rating_min: FromQuery[int | None] = None,
    q: FromQuery[str | None] = None,
    f: FromQuery[list[str] | None] = None,
    sort: FromQuery[str | None] = None,
    depth: FromQuery[str | None] = None,
    size: FromQuery[int | None] = None,
    count: FromQuery[int] = resultset.PEEK_MOST,
    expect: FromQuery[str | None] = None,
) -> PeekView:
    """The rail popover: real members of exactly the page a jump would
    land on, never a guess from scroll height.

    `expect` is the currency of the grid the rail is drawn beside. When
    the library has moved on, a preview from the NEW ordering shown
    beside the OLD grid would present two generations as one answer --
    the response is 409 and the client redraws instead of pretending.
    """
    query = _asked(
        folder,
        album,
        kind,
        q,
        sort,
        size,
        person=person,
        artifact=artifact,
        favorite=favorite,
        rating_min=rating_min,
        facets=f,
        depth=depth,
    )
    conn = connect.connect(state.db_path)
    try:
        if expect is not None and resultset.currency(conn) != expect:
            raise HTTPException(status_code=409, detail="the result set has changed; redraw the gallery")
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        try:
            told = resultset.peek(conn, weights, query, page, time.time(), count=count, actor_id=state.actor_id)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        return PeekView(
            page=told["page"],
            pages=told["pages"],
            total=told["total"],
            first_ordinal=told["first_ordinal"],
            last_ordinal=told["last_ordinal"],
            currency=told["currency"],
            answer=told["answer"],
            items=result_items(told["items"]),
        )
    finally:
        connect.close(conn)


class ResultItem(Wire):
    """One picture as an ordered answer names it.

    The ResultSet's own vocabulary, stated here because this module owns
    that Implementation's HTTP presentation: db/resultset.py builds rows,
    and the conversion to a contract happens at this seam rather than in
    the module that reads the database.
    """

    id: int
    slug: str
    name: str
    kind: str
    #: the entity's uuid, hex-spelled
    uuid: str
    #: the caption the configured model gave it, when it has one
    said: str | None
    #: position in the whole answer, 1-based -- not within the page
    ordinal: int
    #: Where to point an `<img>`: the content-addressed asset when the
    #: bytes have been hashed, the slug route when they have not, and
    #: None when the kind has no picture to take -- audio, documents.
    #: RESOLVED ONCE, here, for the whole page -- which is the entire
    #: point (vision/thumbs.py `asset_url`).
    thumb: str | None
    #: The picture's own proportion, for a grid that justifies rows
    #: instead of cropping every file to a square. None for a kind with
    #: no picture, and for a file nothing has read the dimensions of yet
    #: -- the cell falls back to a square there rather than collapsing.
    width: int | None = None
    height: int | None = None
    #: How many files the dupe job put in this one's group, itself
    #: included; None when no group holds it. A library of generation
    #: sweeps draws the same picture forty times and the grid showed
    #: forty peers. Marked, never collapsed -- the total, the ordinals
    #: and the rail's map are all about MEMBERS.
    copies: int | None = None


def result_items(rows: list[dict]) -> list[ResultItem]:
    from vision import thumbs

    return [
        ResultItem(
            id=row["id"],
            slug=row["slug"],
            name=row["name"],
            kind=row["kind"],
            uuid=row["uuid"],
            said=row["said"],
            ordinal=row["ordinal"],
            thumb=thumbs.asset_url(row.get("sha"), row["slug"], medium=row["kind"]),
            width=row.get("width"),
            height=row.get("height"),
            copies=row.get("copies"),
        )
        for row in rows
    ]


class PeekView(Wire):
    """The rail popover's preview: real members of exactly the page a jump
    would land on, with the ordinals to say which part of the answer they
    are, and the generation evidence to prove they belong beside the grid
    they float over."""

    page: int
    pages: int
    total: int
    first_ordinal: int
    last_ordinal: int
    currency: str
    answer: str
    items: list[ResultItem]


class NotLocated(Wire):
    """The file is not in this answer's membership at all."""

    in_answer: Literal[False]


class Located(Wire):
    """Where one picture sits in this answer. `previous` and `next` are
    addresses in ANSWER order, which is what the arrows mean while a
    result set is being walked."""

    in_answer: Literal[True]
    ordinal: int
    page: int
    total: int
    #: the library generation this answer was computed at
    currency: str
    #: the identity of the ordering itself; the same answer means the
    #: same ordering, whatever the currency has done since
    answer: str
    qs: str
    previous: str | None
    next: str | None


@get("/g/locate/{slug:str}", sync_to_thread=True)
def locate_in_answer(
    state: State,
    slug: FromPath[str],
    folder: FromQuery[str | None] = None,
    album: FromQuery[str | None] = None,
    person: FromQuery[str | None] = None,
    artifact: FromQuery[str | None] = None,
    kind: FromQuery[str | None] = None,
    favorite: FromQuery[str | None] = None,
    rating_min: FromQuery[int | None] = None,
    q: FromQuery[str | None] = None,
    f: FromQuery[list[str] | None] = None,
    sort: FromQuery[str | None] = None,
    depth: FromQuery[str | None] = None,
    size: FromQuery[int | None] = None,
) -> Located | NotLocated:
    """Where one picture sits in this answer -- ordinal, page, and its
    previous/next in ANSWER order, which is what the arrows mean while
    a result set is being walked.

    Two shapes discriminated by `in_answer`, so a client that checks it
    has the rest of the fields and one that does not cannot reach them."""
    query = _asked(
        folder,
        album,
        kind,
        q,
        sort,
        size,
        person=person,
        artifact=artifact,
        favorite=favorite,
        rating_min=rating_min,
        facets=f,
        depth=depth,
    )
    conn = connect.connect(state.db_path)
    try:
        found = naming.resolve(conn, "file", slug)
        if found is None:
            raise NotFoundException(f"no file at /i/{slug}")
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        try:
            told = resultset.locate(conn, weights, query, found[0], time.time(), actor_id=state.actor_id)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        conn.commit()
        if told is None:
            return NotLocated(in_answer=False)
        return Located(
            in_answer=True,
            ordinal=told["ordinal"],
            page=told["page"],
            total=told["total"],
            currency=told["currency"],
            answer=told["answer"],
            qs=told["qs"],
            previous=told["previous"],
            next=told["next"],
        )
    finally:
        connect.close(conn)
