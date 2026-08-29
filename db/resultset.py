"""One authoritative answer to "what is the user looking at".

A GalleryQuery names the question -- scope, filter, phrase, order, page
size -- and every presentation surface reads the SAME materialized
answer: the grid page, the total and page count, the rail's geometry,
the hover peek, locate, previous and next. The browser chooses how
results look, never what they are; no route, template or script owns a
second opinion about membership or order.

Behind the interface sits a disposable ordered projection: the full
list of file ids in answer order, built once and paged by slicing. Two
rules govern it, and both are contracts rather than implementation
choices:

- **Validity is (query fingerprint, data currency), never fingerprint
  alone.** The fingerprint identifies the QUESTION; the currency
  identifies the LIBRARY STATE it was answered against. A scan lands,
  an embed job finishes, a rating changes -- a projection keyed only by
  the question would keep answering from the world before the change,
  which is precisely the divergence this module exists to prevent.
  Currency comes from `PRAGMA data_version` read on ONE long-lived
  read-only monitor connection per database file: the counter is
  per-connection (sqlite/sqlite@b09c88c14 src/pager.c:669), bumps when a
  read begins after any other connection's commit (src/pager.c:3306
  -> pager_reset -> :1784), and reading the pragma opens that read
  (src/btree.c:10443 asserts an open transaction) -- so a per-request
  connection could never carry the key, and the monitor always can.

- **Semantic order makes the projection MANDATORY, not an
  optimization.** db/retrieval.py returns a rank fusion, and rank
  fusion is not incrementally pageable: page 138 of a fused ranking
  depends on every space's full candidate list. The fused ordering is
  materialized once per (fingerprint, currency) and every page, peek
  and locate reads it; nothing reruns FAISS per page. Do not "optimize"
  the projection away -- keyset paging is an alternative only for the
  time-ordered sorts, and adopting it would silently break semantic
  paging.

Materializing walks the whole membership once -- an ordered index walk
for the time sorts, the fused retrieval for similarity. That cost is
amortized over every page, peek and locate until the library changes,
which is a different contract from db/pages.py's per-request queries
and why these statements live here rather than there.

Grouping is deliberately absent: group-aware page breaks change the
page count, every anchor, and the meaning of an ordinal. It arrives
when its effect on page boundaries is defined, not before.

`_SEEN` carries the rewind half of the currency key. `answer_generation`
lives in the database file, so restoring a snapshot over a running
process rewinds it, and `PRAGMA data_version` cannot see the rewind: it
counts a connection's own observations, which only ever go up. The count
of observed rewinds goes into the projection cache key, which is
otherwise process-lifetime, so every key minted after a restore is new
and no old key can match an answer computed from data that is gone.

`NAMED` selects the names one page of cells needs. `content_sha256` is
in it because the thumbnail's identity resolves once for the whole page:
the derivative cache is content-addressed (vision/thumbs.py `path_for`),
so a cell's `src` points straight at an immutable file rather than at a
route that resolves the slug again, and width/height ride along because
the justified grid needs the proportion before the image loads. `copies`
counts the files the dupe job put in one file's group, itself included,
and is NULL for a file no group holds; it marks rather than collapses,
because the answer's total, its ordinals and the rail's map are all
statements about members. `moment` is the interpretation -- the wall
clock when one was claimed, the knowable instant otherwise -- falling
back to `file.mtime`, which is NOT NULL, so the column never is, while
`dated` says which of the two it is, because mtime is when the file was
written and not when the photograph happened. It is not `first_seen_at`:
a bulk import stamps forty thousand pictures with one afternoon, which
is no position on an axis of when things happened.

`COLUMN_ORDERS` holds the expression each table heading orders on, one
row per column rather than one branch per column. Every expression may
be NULL, and that is the point: a sound has no pixels and a photograph
has no length, so those files sort last and say so by position rather
than being dropped or called zero. A sort never narrows the answer --
every join in `COLUMN_JOINS` is a LEFT join, so asking for "by sampler"
over a library of photographs orders them and keeps them.
"""

from __future__ import annotations

import atexit
import contextlib
import dataclasses
import json
import sqlite3
import threading
import typing

from . import naming

#: The column a table heading orders by, and the expression it orders on; a
#: sort is a closed vocabulary with one implementation each. NULL sorts last
#: and every join is a LEFT join, both covered in the module docstring.
COLUMN_ORDERS: dict[str, str] = {
    "name": "f.name COLLATE NOCASE",
    "kind": "f.kind",
    "size": "f.size",
    #: The pixels ON DISK, so a file missing either dimension has no
    #: area rather than an area of nothing.
    "pixels": "f.width * f.height",
    "length": "f.duration",
    # The recipe's numbers.
    "seed": "g.seed",
    "steps": "g.steps",
    "cfg": "g.cfg",
    "sampler": "g.sampler",
    "checkpoint": "ck.name COLLATE NOCASE",
    # The camera's.
    "camera": "cam.name COLLATE NOCASE",
    "iso": "cap.iso",
    "f_number": "cap.f_number",
    "focal_length": "cap.focal_length",
    # And what a person said. Bound to the ACTOR, which is why this one
    # takes an argument in the ORDER BY as well as the WHERE: a rating
    # is somebody's, and "sort by rating" means sort by MINE.
    "rating": "r.rating",
}

#: What each sortable column needs joined to be reachable, keyed by the alias
#: the expression above uses; written once and shared, so two columns off one
#: table do not join it twice. `?` is the actor, and the only argument here.
COLUMN_JOINS: dict[str, str] = {
    "g": " LEFT JOIN generation g ON g.file_id = f.id",
    "ck": (
        " LEFT JOIN file_artifact fck ON fck.file_id = f.id AND fck.role = 'checkpoint' AND fck.ordinal = 0"
        " LEFT JOIN artifact ck ON ck.id = fck.artifact_id"
    ),
    "cam": (
        " LEFT JOIN file_artifact fcam ON fcam.file_id = f.id AND fcam.role = 'captured_with' AND fcam.ordinal = 0"
        " LEFT JOIN artifact cam ON cam.id = fcam.artifact_id"
    ),
    "cap": " LEFT JOIN capture cap ON cap.file_id = f.id",
    "r": " LEFT JOIN rating r ON r.file_id = f.id AND r.user_id = ?",
}

#: Which alias each column's expression reaches through, and how many arguments
#: that join binds. Derived from the expression rather than restated, so a
#: column cannot name a join it does not use.
COLUMN_JOIN_OF: dict[str, str] = {
    name: expression.split(".", 1)[0].split()[-1]
    for name, expression in COLUMN_ORDERS.items()
    if expression.split(".", 1)[0].split()[-1] in ("g", "ck", "cam", "cap", "r")
}

#: Each column, ascending and descending. A second click on a heading means
#: "the other way round", so both directions are real sorts with their own
#: spelling; the question lives in the URL and every answer has to be spellable.
COLUMN_SORTS = tuple(one for name in COLUMN_ORDERS for one in (name, f"{name}-desc"))

#: The orders a query may ask for: "similarity" needs a phrase, the time sorts
#: follow the file table's own indexes, the column sorts a table heading. The
#: moment sorts follow db/context.py HUMAN_MOMENT, uninterpreted files last, not dropped.
SORTS = ("newest", "oldest", "moment", "moment-newest", "similarity", *COLUMN_SORTS)

#: How much of a ranking is the answer: `head` keeps the files that stand above
#: their space's own middle (db/retrieval.py `head`), `all` keeps the whole
#: ranked library. Only a phrase ranks anything, so only a phrase carries a depth.
DEPTHS = ("head", "all")

#: What a file can be, as the one Literal. db/schema.sql constrains `file.kind`
#: to exactly this list, sglint SG709 holds the declaration against that CHECK
#: (sglint/policy.py WIRE_VOCABULARIES), and sg_web/media_view.py imports it.
MediaKind = typing.Literal["image", "animated_image", "video", "audio", "document"]

#: The file kinds a query may filter to -- DERIVED: a bare Literal's
#: arguments come back in declared order (python/cpython
#: Doc/library/typing.rst:3595-3613 get_args).
KINDS = typing.get_args(MediaKind)

DEFAULT_PAGE_SIZE = 60
MAX_PAGE_SIZE = 400


class StaleSession(ValueError):
    """A session link (`event.id`) names a run grouped over an older
    interpretation: its members are a hypothesis nobody has re-proved,
    so the question is refused with the remedy -- never an empty grid
    wearing the bookmark's URL."""


class UnevaluatedCollection(ValueError):
    """A rule-defined collection was asked for members no evaluator has
    produced: unevaluated is not empty, so the question is refused. A
    ValueError, so every route seam already answers it as a bad question;
    typed, so a view can decide "show the rule instead" without matching
    message strings."""


class AnswerChanged(Exception):
    """The caller's expectation names an answer this question no longer
    has -- a selection made against one generation must never mutate
    another. Routes answer it as 409, and nothing was written."""


#: How many previews a peek may carry -- the rail popover shows 6..9.
PEEK_MOST = 9

#: How many projections stay resident per process. Each is one int per
#: file plus an ordinal map -- a handful of concurrent questions, not a
#: history.
KEEP = 8


@dataclasses.dataclass(frozen=True)
class GalleryQuery:
    """The question, whole. Frozen because the fingerprint is derived
    from it; build one through `parse`, which refuses what the module
    cannot answer instead of guessing."""

    folder: str | None = None  # scope: one folder, by slug
    album: str | None = None  # scope: one album (collection), by slug
    person: str | None = None  # facet: composes with any scope, kind and phrase
    #: One artifact entity, by slug -- checkpoint, LoRA, workflow, lens:
    #: the artifact's OWN kind decides its canonical relation, invisible
    #: here. Never model=/lora=/role= -- one entity, one facet.
    artifact: str | None = None
    kind: str | None = None  # filter: one file kind
    #: Authored facets: the asking actor's own judgement, composable like any
    #: predicate. The actor never rides the URL; it binds at answer time and
    #: lives in the projection identity, so one spelling is never one answer.
    favorite: bool | None = None  # True: favorited; False: NOT favorited
    rating_min: int | None = None  # at least this many stars from the actor, 1..5
    text: str | None = None  # the semantic phrase; implies sort=similarity
    #: Registered metadata predicates (db/context.py owns the vocabulary): each
    #: composes like any other facet, timed and semantic alike. Canonically
    #: ordered, so two spellings of one conjunction are one question.
    facets: tuple = ()
    sort: str = "newest"
    #: Semantic only: `head` answers with the files that stand above the middle
    #: of what each space said, `all` with the whole ranked library. Two depths
    #: are two questions, never two views of one, and the fingerprint carries it.
    depth: str = "head"
    size: int = DEFAULT_PAGE_SIZE


def parse(
    *,
    folder: str | None = None,
    album: str | None = None,
    person: str | None = None,
    artifact: str | None = None,
    kind: str | None = None,
    favorite: str | None = None,
    rating_min: int | None = None,
    text: str | None = None,
    facets=None,
    sort: str | None = None,
    depth: str | None = None,
    size: int | None = None,
) -> GalleryQuery:
    """A validated GalleryQuery from request-shaped inputs.

    Refusals are loud and name the rule: an unanswerable question must
    fail where it is asked, never become an empty page that looks like
    an answer.
    """
    folder = (folder or "").strip() or None
    album = (album or "").strip() or None
    person = (person or "").strip() or None
    artifact = (artifact or "").strip() or None
    kind = (kind or "").strip() or None
    text = (text or "").strip() or None
    if sort is None or not sort.strip():
        sort = "similarity" if text else "newest"
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {', '.join(SORTS)}, not {sort!r}")
    if sort == "similarity" and text is None:
        raise ValueError("sort=similarity needs a phrase to rank by")
    if text is not None and sort != "similarity":
        # A phrase used as a filter under a time sort is a real feature
        # with its own membership rule; until that rule exists, refusing
        # beats silently ignoring the phrase.
        raise ValueError("a phrase orders by similarity; other sorts do not consume it")
    chosen_depth = (depth or "").strip() or None
    if chosen_depth is not None and chosen_depth not in DEPTHS:
        raise ValueError(f"depth must be one of {', '.join(DEPTHS)}, not {chosen_depth!r}")
    if chosen_depth is not None and text is None:
        # Nothing was ranked, so there is no head to keep and no whole
        # ranking to ask back for. Silently ignoring it would let a
        # bookmarked `depth=all` mean nothing on half the surfaces.
        raise ValueError("depth describes how much of a RANKING answers; only a phrase ranks anything")
    if folder is not None and album is not None:
        raise ValueError("choose a folder or an album, not both")
    # `person` deliberately COMPOSES with either -- and with kind and a
    # phrase: eligibility is an intersection of predicates, and a
    # person's beach pictures in one album is a real question.
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}, not {kind!r}")
    liked = (favorite or "").strip() or None
    if liked is not None and liked not in ("1", "0"):
        # Tri-state, two spellings: 1 favorited, 0 not favorited, and
        # dropping the parameter stops constraining. Nothing else.
        raise ValueError(f"favorite is 1 (favorited) or 0 (not favorited), not favorite={liked!r}")
    if rating_min is not None and not 1 <= rating_min <= 5:
        raise ValueError(f"rating_min names the minimum stars, 1..5, not {rating_min!r}")
    chosen = DEFAULT_PAGE_SIZE if size is None else size
    if not 1 <= chosen <= MAX_PAGE_SIZE:
        raise ValueError(f"page size must be 1..{MAX_PAGE_SIZE}, not {chosen}")
    from . import facets as facets_module

    held = facets_module.normalized(facets)
    return GalleryQuery(
        folder=folder,
        album=album,
        person=person,
        artifact=artifact,
        kind=kind,
        favorite=None if liked is None else liked == "1",
        rating_min=None if rating_min is None else rating_min,
        text=text,
        facets=held,
        sort=sort,
        depth=chosen_depth or "head",
        size=chosen,
    )


def with_scope(query: GalleryQuery, key: str, value: str | None) -> GalleryQuery:
    """`query` with one scope dimension set from its URL spelling, or
    cleared when `value` is None.

    Scope dimensions are the ones a GalleryQuery carries as a field of
    its own rather than as a facet (db/vocabulary.py, `carried ==
    "scope"`), and they are NOT all strings: `favorite` holds
    `bool | None` and `rating_min` holds `int | None`. So the obvious
    `dataclasses.replace(query, **{key: value})` writes a URL string into
    two fields that do not hold one -- `favorite="1"` instead of
    `favorite=True` -- and nothing catches it, because a `**` spread of
    `dict[str, str]` says nothing about which field it lands in. Written
    out, every branch is checked against the field it assigns, and the
    two conversions are the ones `parse` already makes for the same
    spellings.
    """
    match key:
        case "folder":
            return dataclasses.replace(query, folder=value)
        case "album":
            return dataclasses.replace(query, album=value)
        case "person":
            return dataclasses.replace(query, person=value)
        case "artifact":
            return dataclasses.replace(query, artifact=value)
        case "kind":
            return dataclasses.replace(query, kind=value)
        case "favorite":
            return dataclasses.replace(query, favorite=None if value is None else value == "1")
        case "rating_min":
            return dataclasses.replace(query, rating_min=None if value is None else int(value))
    raise ValueError(f"{key!r} is not a scope dimension of a gallery query")


def fingerprint(query: GalleryQuery) -> str:
    """The identity of the question AS SPELLED: canonical JSON over
    every field, hashed. Page size is part of it because ordinal->page
    arithmetic is. The projection cache does NOT key on this -- it keys
    on `_bound_fingerprint`, over stable entity ids, so a renamed
    person's two spellings stay one cached question."""
    told = json.dumps(dataclasses.asdict(query), sort_keys=True, separators=(",", ":"))
    return naming.short_hash(told)


# --- data currency ----------------------------------------------------------

#: One read-only monitor connection per database file, held for the process.
#: `PRAGMA data_version` is only comparable to itself on the same connection,
#: so the connection that answers it is never a per-request one.
_MONITORS: dict[str, sqlite3.Connection] = {}
_MONITOR_LOCK = threading.Lock()

#: Per database file, the highest generation this process has seen and how many
#: times that number has gone backwards. The rewind count keys the projection
#: cache; the module docstring says why `PRAGMA data_version` cannot supply it.
_SEEN: dict[str, tuple[int, int]] = {}


def close_monitors() -> int:
    """Close every monitor and forget it; returns how many there were.

    The other half of a process-lifetime cache. Nothing may close a
    monitor per request -- `data_version` is comparable only across reads
    on the SAME connection, so a replaced monitor silently restarts the
    numbering the projection cache is keyed on -- but a handle held for
    the life of the process still has to be given up when the process
    ends. Left to the interpreter, the dict's globals are torn down and
    every monitor is deleted without close(), which is a
    `ResourceWarning: unclosed database` per open database file
    (python/cpython Doc/library/sqlite3.rst: Connection warns if close()
    was not called before it is deleted).

    A caller that reads `currency` again after this simply gets a fresh
    monitor: the cache is a cache.
    """
    with _MONITOR_LOCK:
        held = list(_MONITORS.values())
        _MONITORS.clear()
    for monitor in held:
        monitor.close()
    return len(held)


# `_SEEN` is deliberately not cleared above: it is the rewind epoch, and the
# projection cache it protects outlives any monitor, so forgetting the highest
# generation seen would let every key minted before a restore match again.


# The process owns these, not an application instance: `_MONITORS` is module
# state, so two Litestar apps in one process share it and a shutdown hook on
# either would close the other's monitors. atexit matches that lifetime.
atexit.register(close_monitors)


def _database_file(conn) -> str:
    row = next((r for r in conn.execute("PRAGMA database_list") if r[1] == "main"), None)
    return row[2] if row else ""


def currency(conn) -> str:
    """The library-state half of the projection key.

    A file database answers from the monitor connection, which sees
    every OTHER connection's commit -- and every writer in this
    application is another connection, per-request or worker. An
    in-memory database is reachable only through the one connection
    that holds it, so its own `total_changes` (monotonic per DML row,
    python/cpython Doc/library/sqlite3.rst Connection.total_changes)
    carries the same meaning.

    `answer_generation`, not `PRAGMA data_version`. The pragma is the
    obvious answer and is what FTS5 uses for its own structure cache
    (sqlite/sqlite ext/fts5/fts5_index.c fts5IndexDataVersion), but it
    means "somebody committed something", and what is cached here is the
    WHOLE ORDERED ANSWER. Jobs commit per item, so at 80,000 files a
    page costs 0.179 ms at rest and 37.93 ms while a job runs, 211.8x
    (benchmarks/results/answer_currency.json) -- and the job that runs
    for hours writes nothing but the ledger.

    The counter moves for every table except `job`, `job_item` and
    `job_event` (db/schema.sql), so a ledger commit no longer discards
    an answer and every other commit still does.
    """
    from . import connect

    where = _database_file(conn)
    if not where:
        # One connection, so its own row counter is the whole story, and it
        # counts ledger rows the file path ignores. An in-memory database is a
        # test or a probe, never a run somebody browses while a job writes.
        return f"mem{id(conn)}.{conn.total_changes}"
    with _MONITOR_LOCK:
        monitor = _MONITORS.get(where)
        if monitor is None:
            monitor = _MONITORS[where] = connect.connect(where, read_only=True, cross_thread=True)
        held = int(monitor.execute("SELECT value FROM answer_generation").fetchone()[0])
        highest, epoch = _SEEN.get(where, (held, 0))
        if held < highest:
            epoch += 1
            highest = held
        _SEEN[where] = (max(highest, held), epoch)
        return f"g{epoch}.{held}"


# --- the projection ---------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Projection:
    """One materialized ordered answer and the identities that key it.

    ``answer`` is the identity of the ordered answer itself, as against
    ``currency``, which is the identity of the library generation it was
    computed from. ``data_version`` moves on every commit -- a favorite, a
    rating -- while most commits leave most answers untouched, so a client
    holding (currency, answer) can tell "the library moved but this answer did
    not", and adopt the new currency in place, from "the answer really
    changed", and redraw, without anyone teaching it which tables affect which
    queries.
    """

    fingerprint: str
    currency: str
    #: The identity of the ordered answer itself; see the class docstring.
    answer: str
    ids: tuple[int, ...]  # file ids, answer order
    ordinal: dict[int, int]  # file id -> 0-based position
    provenance: dict | None  # similarity only: participants/contributors/missing
    #: similarity only: file id -> how far it stands from the middle of what a
    #: space said toward that space's best, 0..1. Empty for a timed answer,
    #: where no space was asked anything and no such quantity exists.
    relevance: dict[int, float] = dataclasses.field(default_factory=dict)


#: (database, fingerprint, currency) -> Projection, oldest evicted first.
_PROJECTIONS: dict[tuple[str, str, str], Projection] = {}
_PROJECTION_LOCK = threading.Lock()

#: The names one page of cells needs, id-keyed; order is restored from the
#: projection slice, so this query carries none. The module docstring says what
#: each column that is not self-evident is for.
NAMED = (
    "SELECT f.id, e.slug, f.name, f.kind, e.uuid, f.content_sha256, f.width, f.height,"
    " (SELECT count(*) FROM derived_dupe_group m WHERE m.group_id = dg.group_id) AS copies,"
    " COALESCE({moment}, f.mtime) AS moment,"
    " ({moment} IS NOT NULL) AS dated"
    " FROM file f JOIN entity e ON e.id = f.id"
    " LEFT JOIN derived_dupe_group dg ON dg.file_id = f.id"
    " LEFT JOIN derived_media_context mc ON mc.file_id = f.id AND mc.policy_version = {policy}"
    " WHERE f.id IN ({marks})"
)


def canonical(query: GalleryQuery, page: int | None = None) -> str:
    """The query's one spelling in a URL -- owned HERE, beside what the
    question means, because the spelling is entity-aware: answers carry
    the canonical string rebuilt from the BOUND query, so a context
    written with a since-retired slug heals to the live one as it is
    navigated. Defaults are omitted so two ways of asking the same
    question share an address; `page` rides at the end so the rail can
    append its jumps."""
    import urllib.parse

    pairs: list[tuple[str, str]] = []
    if query.text:
        pairs.append(("q", query.text))
    if query.folder:
        pairs.append(("folder", query.folder))
    if query.album:
        pairs.append(("album", query.album))
    if query.person:
        pairs.append(("person", query.person))
    if query.artifact:
        pairs.append(("artifact", query.artifact))
    if query.kind:
        pairs.append(("kind", query.kind))
    if query.favorite is not None:
        pairs.append(("favorite", "1" if query.favorite else "0"))
    if query.rating_min is not None:
        pairs.append(("rating_min", str(query.rating_min)))
    if query.facets:
        from . import facets as facets_module

        pairs.extend(("f", facets_module.spell(held)) for held in query.facets)
    if query.sort != ("similarity" if query.text else "newest"):
        pairs.append(("sort", query.sort))
    if query.text and query.depth != "head":
        pairs.append(("depth", query.depth))
    if query.size != DEFAULT_PAGE_SIZE:
        pairs.append(("size", str(query.size)))
    if page is not None and page > 1:
        pairs.append(("page", str(page)))
    return urllib.parse.urlencode(pairs)


@dataclasses.dataclass(frozen=True)
class _Bound:
    """The question bound to STABLE identities.

    The public interface speaks slugs because URLs do; the
    implementation works in entity ids, because slugs are the one part
    of an address that moves -- naming a person is the People page's
    primary action, and a projection keyed on the slug string would
    fork the cache for one human and answer an old bookmark as a
    different question. `query` carries the LIVE spelling, which is
    what every emitted URL uses; `face_run_id` pins WHICH clustering's
    attribution person membership means, so switching the primary run
    is visibly a different question."""

    query: GalleryQuery  # live-slug spelling
    folder_id: int | None
    collection_id: int | None
    person_id: int | None
    face_run_id: int | None
    #: The artifact facet, bound: the stable entity id plus the artifact's own
    #: kind, which privately decides whether membership means file_artifact or
    #: generation.workflow_id. Nothing above this module sees that split.
    artifact_id: int | None
    artifact_kind: str | None
    #: WHOSE judgement the authored facets mean. Set only when the query
    #: carries one, so questions without an authored facet stay one
    #: cached projection however many actors ask them.
    actor_id: int | None
    #: A smart collection scope, bound: the rule's own question as an inner
    #: _Bound plus its take. The rule owns membership and the outer question the
    #: ordered answer; `collection_id` stays None, collection_file holding no row.
    rule: tuple[_Bound, int | None] | None = None


def bind(conn, query: GalleryQuery, actor_id: int | None = None) -> _Bound:
    """Resolve every slug to its entity -- retired spellings included,
    refusing an address nothing lives at: an empty page at a misspelled
    folder would look exactly like an empty folder.

    `actor_id` is required exactly when the query carries an authored facet
    (favorite, rating): those predicates are one person's judgement, and
    answering them for nobody would be answering a different question
    while wearing this one's URL."""
    from . import naming

    held: dict[str, int | None] = {"folder": None, "album": None, "person": None, "artifact": None}
    live: dict[str, str] = {}
    for field, entity_kind in (
        ("folder", "folder"),
        ("album", "collection"),
        ("person", "person"),
        ("artifact", "artifact"),
    ):
        slug = getattr(query, field)
        if slug is None:
            continue
        found = naming.resolve(conn, entity_kind, slug)
        if found is None:
            raise LookupError(f"no {entity_kind} at {slug!r}")
        held[field] = found[0]
        if not found[1]:
            fresh = naming.entity_slug(conn, found[0])
            if fresh is not None:
                live[field] = fresh[1]
    rule = None
    if held["album"] is not None:
        kind = conn.execute("SELECT kind FROM collection WHERE id = ?", (held["album"],)).fetchone()[0]
        if kind == "smart":
            # A smart collection's membership is its rule's answer. A
            # collection with no typed rule is unevaluated and one whose
            # references rot is broken, neither empty (db/collection_rules).
            from . import collection_rules

            spelled = live.get("album") or query.album or str(held["album"])
            told = collection_rules.load(conn, held["album"])
            if told is None:
                raise UnevaluatedCollection(
                    f"collection {spelled!r} is rule-defined; smart membership is not evaluated yet"
                )
            rule = (_bind_rule(conn, told, spelled), told.take)
            held["album"] = None  # membership comes from the rule set, not collection_file
    for one in query.facets:
        if one.key != "event.id":
            continue
        # the facet's own predicate answers nothing for a stale run; say why
        from .context import POLICY_VERSION

        row = conn.execute(
            "SELECT r.context_generation = (SELECT generation FROM derived_context_state)"
            "   AND r.context_policy_version = ?"
            "  FROM derived_event ev JOIN derived_event_run r ON r.id = ev.run_id WHERE ev.id = ?",
            (int(POLICY_VERSION), int(one.value)),
        ).fetchone()
        if row is None:
            raise LookupError(f"no session {one.value}")
        if not row[0]:
            raise StaleSession(
                f"session {one.value} was grouped over an older interpretation; run the events job to group again"
            )
    run = None
    if held["person"] is not None:
        row = conn.execute("SELECT id FROM derived_face_run WHERE is_primary = 1").fetchone()
        run = row[0] if row else None
    artifact_kind = None
    if held["artifact"] is not None:
        artifact_kind = conn.execute("SELECT kind FROM artifact WHERE id = ?", (held["artifact"],)).fetchone()[0]
    asks_authored = query.favorite is not None or query.rating_min is not None
    if asks_authored and actor_id is None:
        raise ValueError("an authored facet (favorite, rating) is one actor's judgement; no actor was bound")
    # Retired slugs answered with their current spelling, one field at a
    # time: `live` only ever carries the four slug fields resolved above.
    spelled_now = query
    for field, slug in live.items():
        spelled_now = with_scope(spelled_now, field, slug)
    return _Bound(
        query=spelled_now,
        folder_id=held["folder"],
        collection_id=held["album"],
        person_id=held["person"],
        face_run_id=run,
        artifact_id=held["artifact"],
        artifact_kind=artifact_kind,
        actor_id=actor_id if asks_authored else None,
        rule=rule,
    )


def _bind_rule(conn, rule, spelled: str) -> _Bound:
    """The rule's own question, bound like any other -- uuid references
    to live entity ids, the CURRENT primary run for its person, and the
    actor PINNED AT CREATION for its authored facets, never the viewer."""
    from .collection_rules import BrokenCollectionRule

    held: dict[str, int | None] = {"folder": None, "person": None, "artifact": None}
    for field, uuid in (
        ("folder", rule.folder_uuid),
        ("person", rule.person_uuid),
        ("artifact", rule.artifact_uuid),
    ):
        if uuid is None:
            continue
        row = conn.execute("SELECT id FROM entity WHERE uuid = ? AND kind = ?", (uuid, field)).fetchone()
        if row is None:
            raise BrokenCollectionRule(f"collection {spelled!r}'s rule references a {field} that no longer exists")
        held[field] = row[0]
    run = None
    if held["person"] is not None:
        row = conn.execute("SELECT id FROM derived_face_run WHERE is_primary = 1").fetchone()
        run = row[0] if row else None
    artifact_kind = None
    if held["artifact"] is not None:
        artifact_kind = conn.execute("SELECT kind FROM artifact WHERE id = ?", (held["artifact"],)).fetchone()[0]
    inner = GalleryQuery(
        kind=rule.kind,
        favorite=rule.favorite,
        rating_min=rule.rating_min,
        text=rule.text,
        sort=rule.sort or ("similarity" if rule.text else "newest"),
        facets=tuple(rule.facets),
    )
    return _Bound(
        query=inner,
        folder_id=held["folder"],
        collection_id=None,
        person_id=held["person"],
        face_run_id=run,
        artifact_id=held["artifact"],
        artifact_kind=artifact_kind,
        actor_id=rule.actor_id,
    )


def _bound_fingerprint(bound: _Bound) -> str:
    """The projection key's question half, over bound identities: two
    spellings of one entity are one question, and one spelling across a
    primary-run switch is two."""
    query = bound.query
    told = json.dumps(
        {
            "folder": bound.folder_id,
            "album": bound.collection_id,
            "person": bound.person_id,
            "artifact": bound.artifact_id,
            "run": bound.face_run_id,
            "actor": bound.actor_id,
            "favorite": query.favorite,
            "rating_min": query.rating_min,
            "kind": query.kind,
            "facets": [dataclasses.astuple(held) for held in query.facets],
            "text": query.text,
            "sort": query.sort,
            # Two depths are two ORDERED ANSWERS over one library
            # generation; sharing a projection key would serve whichever
            # was computed first under the other's address.
            "depth": query.depth,
            "size": query.size,
            # A smart scope's identity is its bound rule (recursively
            # fingerprinted -- ids, pinned actor, run) plus its take, so two
            # collections with one rule are one membership question.
            "rule": None if bound.rule is None else [_bound_fingerprint(bound.rule[0]), bound.rule[1]],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return naming.short_hash(told)


def scope_of(
    conn, query: GalleryQuery, actor_id: int | None = None, *, models_dir: str | None = None, now: float | None = None
) -> tuple[str, list, GalleryQuery]:
    """A gallery question as a scope another surface appends to its own
    statements: the membership conjunct over the file alias `f` (the
    same predicates the gallery walks), its bound values, and the
    question in its LIVE spelling for the links that surface offers.
    A rule-defined collection's membership is a materialized set
    (`_rule_members`, the one engine), appended as `f.id IN (...)`; it
    needs the models directory, because a semantic rule ranks through
    its encoder, and a rule that cannot be answered right now refuses
    (UnavailableCollectionRule) rather than scoping to nothing."""
    import json

    bound = bind(conn, query, actor_id)
    where, args, _ = _eligibility(bound)
    conjunct = "".join(" AND " + part for part in where[1:])  # where[0] is presence
    if bound.rule is not None:
        if models_dir is None or now is None:
            raise ValueError("a rule-defined collection scopes only with a models directory and a clock to evaluate it")
        members = _rule_members(conn, models_dir, bound, now)
        conjunct += " AND f.id IN (SELECT value FROM json_each(?))"
        args = [*args, json.dumps(sorted(members))]
    return (conjunct, args, bound.query)


def _eligibility(bound: _Bound) -> tuple[list[str], list[object], bool]:
    """The membership predicates of a bound question, constructed ONCE
    -- eligibility is an INTERSECTION of predicates, not a choice
    between fixed statements, and this single construction feeds BOTH
    the ordered timed walk and semantic retrieval's allowed set. A facet
    added here is automatically part of both; the alternative was two
    manually-synchronized definitions of "is this question faceted?",
    whose drift would have made a timed artifact search correct while
    the same facet under a phrase silently ranked the whole library.

    Returns (predicates, values, constrained) -- `constrained` says
    whether anything beyond presence narrows the membership."""
    query = bound.query
    where = ["f.missing_since IS NULL"]
    args: list[object] = []
    if bound.folder_id is not None:
        where.append("f.folder_id = ?")
        args.append(bound.folder_id)
    if bound.collection_id is not None:
        where.append("EXISTS (SELECT 1 FROM collection_file cf WHERE cf.file_id = f.id AND cf.collection_id = ?)")
        args.append(bound.collection_id)
    if bound.person_id is not None:
        if bound.face_run_id is None:
            # The person exists; no primary clustering attributes anyone
            # anything. An honest empty, distinct from an unknown slug.
            where.append("0")
        else:
            where.append(
                "EXISTS (SELECT 1 FROM derived_file_person fp"
                " WHERE fp.file_id = f.id AND fp.person_id = ? AND fp.run_id = ?)"
            )
            args.extend((bound.person_id, bound.face_run_id))
    if bound.artifact_id is not None:
        # The artifact's canonical relation, decided by its own kind: a workflow
        # attaches through generation, weights and equipment through
        # file_artifact, where EXISTS makes a twice-stacked LoRA one member.
        if bound.artifact_kind == "workflow":
            where.append("EXISTS (SELECT 1 FROM generation g WHERE g.file_id = f.id AND g.workflow_id = ?)")
        else:
            where.append("EXISTS (SELECT 1 FROM file_artifact fa WHERE fa.file_id = f.id AND fa.artifact_id = ?)")
        args.append(bound.artifact_id)
    if query.kind is not None:
        where.append("f.kind = ?")
        args.append(query.kind)
    if query.favorite is True:
        where.append("EXISTS (SELECT 1 FROM favorite fav WHERE fav.file_id = f.id AND fav.user_id = ?)")
        args.append(bound.actor_id)
    elif query.favorite is False:
        where.append("NOT EXISTS (SELECT 1 FROM favorite fav WHERE fav.file_id = f.id AND fav.user_id = ?)")
        args.append(bound.actor_id)
    if query.rating_min is not None:
        where.append("EXISTS (SELECT 1 FROM rating r WHERE r.file_id = f.id AND r.user_id = ? AND r.rating >= ?)")
        args.extend((bound.actor_id, query.rating_min))
    if query.facets:
        from . import facets as facets_module

        # Through `clauses`, never `predicate` in a loop: repeating a key with
        # `any` means OR and arrives here as one clause. Appending each
        # predicate separately would AND them, so "image or video" answers none.
        for sql, values in facets_module.clauses(query.facets):
            where.append(sql)
            args.extend(values)
    return where, args, len(where) > 1


def _timed_ids(conn, bound: _Bound) -> list[int]:
    """The ordered walk over `_eligibility`'s membership. The walk stays
    on the file table's own time index; the whole statement runs once
    per library change, never once per page."""
    # The ordering contract: (mtime, id) both in the sort's direction, as
    # file_in_folder_by_time carries it, so global and folder-scoped questions tie
    # identically. The indexes implement it (file_recent is (mtime DESC, id DESC), schema v6).
    where, args, _ = _eligibility(bound)
    if bound.query.sort in ("moment", "moment-newest"):
        # The human moment, not the filesystem's: what a timeline link means by
        # "this day" is the order its pictures come back in. LEFT JOIN keeps
        # membership identical; an uninterpreted file sorts last by position.
        from .context import HUMAN_MOMENT, POLICY_VERSION

        order = "ASC" if bound.query.sort == "moment" else "DESC"
        sql = (
            f"SELECT f.id FROM file f LEFT JOIN derived_media_context mc"
            f" ON mc.file_id = f.id AND mc.policy_version = {int(POLICY_VERSION)}"
            f" WHERE {' AND '.join(where)}"
            f" ORDER BY {HUMAN_MOMENT} IS NULL, {HUMAN_MOMENT} {order}, f.id {order}"
        )
        return [row[0] for row in conn.execute(sql, args)]
    if bound.query.sort in COLUMN_SORTS:
        # A table heading, under the ordering contract the time sorts keep: the
        # column, then `f.id` in the same direction, so the order is total. Two
        # files of equal size never swap between two reads of one answer.
        name, _, backwards = bound.query.sort.partition("-")
        order = "DESC" if backwards else "ASC"
        column = COLUMN_ORDERS[name]
        # A LEFT join, so ordering by a column most files do not have keeps
        # them: a photograph has no sampler, sorts last, and says so by
        # position. Narrowing here would change what the answer holds.
        alias = COLUMN_JOIN_OF.get(name)
        joined = COLUMN_JOINS[alias] if alias else ""
        # The actor binds in the JOIN, which comes before the WHERE, so
        # its argument goes first. Bound after, every other placeholder
        # in the statement would be reading one position along.
        args = ([bound.actor_id] if alias == "r" else []) + list(args)
        sql = (
            f"SELECT f.id FROM file f{joined} WHERE {' AND '.join(where)}"
            f" ORDER BY ({column}) IS NULL, {column} {order}, f.id {order}"
        )
        return [row[0] for row in conn.execute(sql, args)]
    order = "ASC" if bound.query.sort == "oldest" else "DESC"
    sql = f"SELECT f.id FROM file f WHERE {' AND '.join(where)} ORDER BY f.mtime {order}, f.id {order}"
    return [row[0] for row in conn.execute(sql, args)]


def _rule_members(conn, models_dir: str, bound: _Bound, now: float) -> frozenset[int]:
    """A smart rule evaluated to its MEMBERSHIP SET, through the same
    machinery every question uses -- never a second engine. `take` cuts
    the rule's own ordering (fused for a semantic rule, timed
    otherwise) down to a set; the outer question then orders whatever
    of that set it keeps."""
    from .collection_rules import UnavailableCollectionRule

    inner, take = bound.rule or (None, None)
    if inner is None:
        return frozenset()
    if any(one.key.startswith("context.") or one.key == "place.id" for one in inner.query.facets):
        # A facet over the interpretation answers only for interpreted
        # files: after a policy bump, or before the context job ran, the
        # rule would evaluate to a smaller set that looks like an answer.
        from . import pages

        have, present, _ = pages.timeline_coverage(conn)
        if have < present:
            raise UnavailableCollectionRule(
                f"the collection's rule reads the interpretation and {present - have} of {present} files have none"
                " under the running policy; run the context job"
            )
    if inner.query.text:
        try:
            ids, _, _ = _fused_ids(conn, models_dir, inner, now)
        except (ValueError, LookupError) as silent:
            raise UnavailableCollectionRule(
                f"the collection's semantic rule cannot be answered right now: {silent}"
            ) from silent
    else:
        ids = _timed_ids(conn, inner)
    if take is not None:
        ids = ids[:take]
    return frozenset(ids)


def _fused_ids(
    conn, models_dir: str, bound: _Bound, now: float, members: frozenset[int] | None = None
) -> tuple[list[int], dict | None, dict[int, float]]:
    """The whole fused ordering, once.

    A scope or filter is handed to retrieval as the ALLOWED set, never
    applied to the fused answer afterwards: RRF consumes rank positions,
    so each space's ranking must be constrained and renumbered BEFORE
    the fusion (db/retrieval.py owns that arithmetic) -- filtering a
    global fusion keeps global ranks, and two spaces whose out-of-scope
    candidates sit at different depths compress differently and can
    flip the fused order. This module owns WHICH files are eligible;
    retrieval owns how constrained rankings fuse.
    """
    from . import retrieval

    query = bound.query
    allowed = None
    where, args, constrained = _eligibility(bound)
    if constrained:
        allowed = {row[0] for row in conn.execute(f"SELECT f.id FROM file f WHERE {' AND '.join(where)}", args)}
    if members is not None:
        # A smart scope's membership intersects the outer eligibility, evaluated
        # to a set so a rule's person and the viewer's person stay a conjunction
        # instead of one field overwriting the other.
        allowed = members if allowed is None else allowed & members
    if allowed is not None and not allowed:
        # An empty scope needs no encoder and has no honest
        # provenance -- nothing was asked of any space.
        return [], None, {}
    if query.text is None:
        # no phrase, no semantic ordering: nothing was asked of any space
        return [], None, {}
    depth = len(allowed) if allowed is not None else _present(conn)
    found = retrieval.query(
        conn,
        models_dir,
        query.text,
        max(depth, 1),
        now,
        offline=True,
        allowed=None if allowed is None else set(allowed),
    )
    ranked = found["results"]
    # `head` keeps what stands above the middle of what each space said, `all`
    # keeps the ranking whole. Retrieval decides which files answer, holding the
    # per-space distributions this cannot see; the depth decides how many.
    kept = ranked if query.depth == "all" else [row for row in ranked if row["answers"]]
    fused = [row["file_id"] for row in kept]
    provenance = {key: found[key] for key in ("participants", "contributors", "missing")}
    provenance["unmatched"] = found.get("unmatched") or {}
    #: What the cut cost, so a page can say it rather than looking like
    #: a small library.
    provenance["ranked"] = len(ranked)
    provenance["answering"] = found["answering"]
    provenance["depth"] = query.depth
    relevance = {row["file_id"]: row["relevance"] for row in kept}
    return fused, provenance, relevance


def _present(conn) -> int:
    return conn.execute("SELECT count(*) FROM file WHERE missing_since IS NULL").fetchone()[0]


def _current(conn, models_dir: str, query: GalleryQuery, now: float, actor_id: int | None) -> tuple[_Bound, Projection]:
    """The projection for this question over the library as it stands --
    a stale one is never reused, it is replaced. Currency is read BEFORE
    binding: bind's resolves are this connection's first data reads and
    pin the snapshot, so a commit in the gap builds fresh data under an
    obsolete key -- wasted work the next request replaces, never stale
    data cached under a fresh key."""
    database = _database_file(conn) or f"mem{id(conn)}"
    told = currency(conn)
    bound = bind(conn, query, actor_id)
    key = (database, _bound_fingerprint(bound), told)
    with _PROJECTION_LOCK:
        held = _PROJECTIONS.get(key)
    if held is not None:
        return bound, held
    members = _rule_members(conn, models_dir, bound, now) if bound.rule is not None else None
    if query.sort == "similarity":
        ids, provenance, relevance = _fused_ids(conn, models_dir, bound, now, members=members)
    else:
        ids, provenance, relevance = _timed_ids(conn, bound), None, {}
        if members is not None:
            # The rule owns membership; the OUTER walk keeps its order.
            ids = [file_id for file_id in ids if file_id in members]
    made = Projection(
        fingerprint=key[1],
        currency=key[2],
        answer=naming.short_hash(",".join(str(file_id) for file_id in ids)),
        ids=tuple(ids),
        ordinal={file_id: position for position, file_id in enumerate(ids)},
        provenance=provenance,
        relevance=relevance,
    )
    with _PROJECTION_LOCK:
        _PROJECTIONS[key] = made
        while len(_PROJECTIONS) > KEEP:
            _PROJECTIONS.pop(next(iter(_PROJECTIONS)))
    return bound, made


@contextlib.contextmanager
def snapshot(conn):
    """One public operation reads one SQLite snapshot.

    Counting `_current` takes proved an operation cannot MIX projections;
    this proves the projection itself is not a mixture: construction and
    item hydration span several reads (membership, per-space rows,
    hydration), and in autocommit each is its own read transaction --
    a worker's commit between two of them hands back items from one
    generation of the library described by another's counts. A DEFERRED
    read transaction pins the connection's snapshot at its FIRST data
    read -- which lands after the currency read, because currency comes
    from the monitor connection (file libraries) or an attribute
    (:memory:), so a racing commit can only produce fresh data under an
    already-obsolete key: wasted work the next request replaces, never
    stale data cached under a fresh key.

    Registry minting on a space's first scoped query still writes inside
    the snapshot; that upgrade is the same write the operation always
    performed and `busy_timeout` governs it as before. A caller already
    holding a transaction keeps it -- their snapshot is theirs.
    """
    if conn.in_transaction:
        yield
        return
    conn.execute("BEGIN")
    try:
        yield
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


# --- the interface ----------------------------------------------------------


def _shape(bound: _Bound, held: Projection) -> dict:
    """The answer's shape, computed from ONE projection snapshot. Every
    public operation takes `_current` exactly once and derives all of
    its counts, pages, items and currency from that same Projection --
    two takes could straddle another connection's commit and hand back
    items from one generation under the totals of another. One
    response describes one answer. `qs` is the canonical spelling of
    the BOUND question -- live slugs, so a stale contextual name heals
    as it is navigated."""
    query = bound.query
    total = len(held.ids)
    return {
        "total": total,
        "pages": max(1, -(-total // query.size)),
        "size": query.size,
        "sort": query.sort,
        "fingerprint": held.fingerprint,
        "currency": held.currency,
        "answer": held.answer,
        "provenance": held.provenance,
        "qs": canonical(query),
    }


def describe(conn, models_dir: str, query: GalleryQuery, now: float, *, actor_id: int | None = None) -> dict:
    """The result set's shape: what the rail is drawn from and what the
    grid's pager believes. `currency` rides along so a client can tell
    a redrawn answer from the one it is holding."""
    with snapshot(conn):
        bound, held = _current(conn, models_dir, query, now, actor_id)
        return _shape(bound, held)


def page(conn, models_dir: str, query: GalleryQuery, number: int, now: float, *, actor_id: int | None = None) -> dict:
    """One page of the answer, by number. A number past the end answers
    with the last page that exists -- the library may have shrunk since
    the rail was drawn, and the honest response is the page that IS,
    named as itself."""
    with snapshot(conn):
        bound, held = _current(conn, models_dir, query, now, actor_id)
        shape = _shape(bound, held)
        number = min(max(1, number), shape["pages"])
        start = (number - 1) * bound.query.size
        shape["page"] = number
        shape["items"] = _named(conn, held.ids[start : start + bound.query.size], start, held.relevance)
        return shape


def peek(
    conn,
    models_dir: str,
    query: GalleryQuery,
    number: int,
    now: float,
    count: int = PEEK_MOST,
    *,
    actor_id: int | None = None,
) -> dict:
    """The rail popover's preview: the first few members of EXACTLY the
    page a jump would land on -- by construction a prefix of what
    `page` answers, and the test suite holds the two to it."""
    with snapshot(conn):
        bound, held = _current(conn, models_dir, query, now, actor_id)
        shape = _shape(bound, held)
        number = min(max(1, number), shape["pages"])
        start = (number - 1) * bound.query.size
        take = min(max(1, count), PEEK_MOST, bound.query.size)
        return {
            "page": number,
            "pages": shape["pages"],
            "total": shape["total"],
            "first_ordinal": min(start + 1, max(shape["total"], 1)),
            "last_ordinal": min(start + bound.query.size, shape["total"]),
            "currency": held.currency,
            "answer": held.answer,
            "qs": shape["qs"],
            "items": _named(conn, held.ids[start : start + take], start, held.relevance),
        }


#: How many moments one shape answer carries, however many the answer
#: holds. A profile of a forty-thousand-picture library and one of a
#: four-hundred-picture library are the same size on the wire.
SHAPE_SAMPLES = 2000

#: The most pictures one window answer names. A window is what is on
#: screen; nothing that fits on a screen needs more than this.
WINDOW_MOST = 900


def _moments(conn, ids) -> dict[int, float]:
    """When each of these files happened, keyed by file id.

    Chunked, because an answer can hold tens of thousands of ids and one
    statement binds a bounded number of parameters. The bound is ASKED
    FOR rather than assumed: SQLite's own default for
    `SQLITE_MAX_VARIABLE_NUMBER` is 32766
    (../refs/sqlite/sqlite/src/sqliteLimit.h:189-191), it was 999 before
    3.32, and any build may be compiled with its own. `getlimit` reports
    what THIS build will actually accept, so the chunk is right on a
    system nobody tested on. Measured here: 32766 (sqlite 3.47.1).

    Half the limit, not all of it, so a caller that adds a parameter of
    its own to the same statement later does not silently cross it.

    Each chunk is a primary-key lookup, so the walk is linear in the
    answer and never touches a file the answer does not hold.
    """
    import sqlite3

    from .context import HUMAN_MOMENT, POLICY_VERSION

    try:
        room = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER) // 2
    except (AttributeError, sqlite3.NotSupportedError):
        # A connection object that cannot report its limits: fall back to
        # the pre-3.32 default, which every build accepts.
        room = 499
    step = max(1, room)
    found: dict[int, float] = {}
    ids = list(ids)
    for at in range(0, len(ids), step):
        chunk = ids[at : at + step]
        marks = ",".join("?" for _ in chunk)
        sql = (
            f"SELECT f.id, COALESCE({HUMAN_MOMENT}, f.mtime) FROM file f"
            f" LEFT JOIN derived_media_context mc"
            f"   ON mc.file_id = f.id AND mc.policy_version = {int(POLICY_VERSION)}"
            f" WHERE f.id IN ({marks})"
        )
        found.update({one: when for one, when in conn.execute(sql, chunk) if when is not None})
    return found


def over_time(conn, models_dir: str, query: GalleryQuery, now: float, *, actor_id: int | None = None) -> dict:
    """The answer's SHAPE in time, at a fixed cost whatever its size.

    The coarse level of detail. A surface that draws the whole answer as
    a place -- where the bursts are, and how long the library went
    quiet -- needs the distribution, not the pictures. Forty thousand
    slugs and thumbnail addresses would be megabytes to say something
    that fits in a few kilobytes, and no screen can show forty thousand
    pictures at once anyway.

    So this answers with the moments themselves, DOWNSAMPLED BY RANK:
    every nth one from the sorted walk, plus both ends. Rank sampling
    rather than fixed-width bins because it survives any distribution --
    a library of one wedding and a library of fifteen years both come
    back as `SHAPE_SAMPLES` numbers, and the caller reads density off
    the spacing between them. A wide jump between two neighbours IS a
    stretch with nothing in it, so the gaps arrive intact rather than
    being averaged away by a bin that straddles them.

    `stride` is how many members each sample stands for, which is what
    lets the caller turn spacing back into a count.
    """
    with snapshot(conn):
        _bound, held = _current(conn, models_dir, query, now, actor_id)
        stamps = sorted(_moments(conn, held.ids).values())
        stride = max(1, -(-len(stamps) // SHAPE_SAMPLES))
        samples = stamps[::stride]
        # Both ends always, so the axis a caller draws covers exactly
        # what the answer covers rather than stopping short of it.
        if stamps and samples[-1:] != stamps[-1:]:
            samples.append(stamps[-1])
        return {
            "total": len(held.ids),
            "dated": len(stamps),
            "currency": held.currency,
            "answer": held.answer,
            "stride": stride,
            "samples": samples,
        }


def against(
    conn,
    models_dir: str,
    left: GalleryQuery,
    right: GalleryQuery,
    now: float,
    *,
    most: int = 12,
    actor_id: int | None = None,
) -> dict:
    """What two questions have in common, and what only one of them holds.

    Set arithmetic over two answers' MEMBERSHIPS -- exactly, on the ids
    the projections already hold, rather than by fetching both sets of
    pictures and comparing them in a browser. Two answers of forty
    thousand each cost two projections and three set operations here;
    fetching them would cost eighty thousand rows over the wire to
    compute a number.

    The two are taken under ONE snapshot, so `both + only_left` is
    always `left` and the three numbers cannot come from two different
    generations of the library. Two separate reads could straddle a
    commit and report a comparison that was never true at any instant.

    `shared` and the two `only` lists carry a few members each, so a
    surface can SHOW what it is talking about rather than only counting
    it -- a difference of six that you cannot look at is a number, not
    an answer.
    """
    with snapshot(conn):
        _bl, held_left = _current(conn, models_dir, left, now, actor_id)
        _br, held_right = _current(conn, models_dir, right, now, actor_id)
        a, b = set(held_left.ids), set(held_right.ids)
        both = a & b
        # In the LEFT answer's own order, so a comparison reads the way
        # the answer it came from reads.
        ordered = [one for one in held_left.ids if one in both]
        only_a = [one for one in held_left.ids if one not in b]
        only_b = [one for one in held_right.ids if one not in a]
        return {
            "left": len(a),
            "right": len(b),
            "both": len(both),
            "only_left": len(only_a),
            "only_right": len(only_b),
            "currency": held_left.currency,
            "shared": _named(conn, ordered[:most], 0, None, said=False),
            "left_only": _named(conn, only_a[:most], 0, None, said=False),
            "right_only": _named(conn, only_b[:most], 0, None, said=False),
        }


def window(
    conn,
    models_dir: str,
    query: GalleryQuery,
    now: float,
    after: float,
    before: float,
    *,
    most: int = WINDOW_MOST,
    actor_id: int | None = None,
) -> dict:
    """The members of this answer that happened between two moments.

    The fine level of detail, and the reason there is no cap on how big
    an answer this surface can draw: a window is what is on screen, and
    what is on screen is bounded by the screen. Zooming in narrows the
    window, so the cost of looking closely goes DOWN.

    `more` is how many fell inside and were not named. Never silently
    dropped: a window that quietly returned the first nine hundred of
    four thousand would be a lie about what that stretch of time holds,
    and the caller has to be able to say "denser than this shows".
    """
    with snapshot(conn):
        _bound, held = _current(conn, models_dir, query, now, actor_id)
        ids = list(held.ids)
        # Cut to the window, IN THE ANSWER'S OWN ORDER: the ordering
        # contract is the answer's, and a window that reordered would
        # hand back a different walk than the one being looked at.
        stamps = _moments(conn, ids)
        inside = [one for one in ids if one in stamps and after <= stamps[one] <= before]
        take = min(max(1, most), WINDOW_MOST)
        named = _named(conn, inside[:take], 0, held.relevance, said=False)
        for row in named:
            row["moment"] = stamps.get(row["id"])
        return {
            "currency": held.currency,
            "answer": held.answer,
            "held": len(inside),
            "more": max(0, len(inside) - len(named)),
            "items": named,
        }


def _located(conn, bound: _Bound, held: Projection, position: int) -> dict:
    """Where a position sits in an answer, from one projection.

    Shared rather than repeated: `locate` and `neighborhood` answer the
    same question about the same walk, and two spellings of "previous"
    are two chances for the arrows and the strip beneath them to
    disagree about what comes next.
    """
    neighbours = [held.ids[at] if 0 <= at < len(held.ids) else None for at in (position - 1, position + 1)]
    # The two ends come along, because a walk that wraps needs an address for
    # them and this is the only read that holds the whole ordered answer;
    # `held.ids` is already in hand and the four ids ride the one `_named`.
    ends = [held.ids[0], held.ids[-1]] if held.ids else []
    # Whether wrapping happens is not decided here: it is how a person arranged
    # their viewer (frontend/src/workspace.ts), it changes no membership and
    # belongs in no fingerprint, and the server's part is spelling where the ends are.
    wanted = [one for one in [*neighbours, *ends] if one is not None]
    named = {row["id"]: row["slug"] for row in _named(conn, wanted, 0)}
    return {
        "ordinal": position + 1,
        "page": position // bound.query.size + 1,
        "total": len(held.ids),
        "currency": held.currency,
        "answer": held.answer,
        "qs": canonical(bound.query),
        "previous": named.get(neighbours[0]),
        "next": named.get(neighbours[1]),
        "first": named.get(ends[0]) if ends else None,
        "last": named.get(ends[1]) if ends else None,
    }


def locate(
    conn, models_dir: str, query: GalleryQuery, file_id: int, now: float, *, actor_id: int | None = None
) -> dict | None:
    """Where one file sits in the answer -- its ordinal, its page, and
    its neighbours in ANSWER order, which is what previous/next mean
    while a result set is being walked. None when the file is not in
    the membership at all."""
    with snapshot(conn):
        bound, held = _current(conn, models_dir, query, now, actor_id)
        position = held.ordinal.get(file_id)
        return None if position is None else _located(conn, bound, held, position)


#: The widest neighborhood a caller may ask for, clamped rather than refused:
#: a strip is a convenience and an absurd `count` still has an obvious best
#: answer, the same reasoning as a collection's `take` (db/collection_rules.py).
NEIGHBORHOOD_MOST = 51
#: The filmstrip a viewer opens with when it does not ask: seven each side of
#: the member fills the strip on ordinary windows without paying the ceiling.
NEIGHBORHOOD_DEFAULT = 15


def neighborhood(
    conn,
    models_dir: str,
    query: GalleryQuery,
    file_id: int,
    now: float,
    count: int = NEIGHBORHOOD_DEFAULT,
    *,
    actor_id: int | None = None,
) -> dict | None:
    """What SURROUNDS one member of this answer, in answer order.

    A different question from `peek`, which answers "the first few
    members of page N" for the rail's long-distance jumps. This one
    knows nothing about pages: it is a window around a POSITION, so a
    window that happens to straddle a page boundary is not a special
    case and is not stitched from two reads. `page` in the result is
    only the located file's own, carried for the return-to-results URL.

    The window slides at the edges rather than being padded: an item
    near the start of the answer sits near the left of a FULL window,
    not centred with blanks beside it. Everything comes from the one
    `_current` inside one snapshot, so the ordinal, the arrows and the
    window are the same answer at the same generation -- three reads
    would be three chances to describe two.

    None when the file is not in this answer at all. The query defines
    the walk; there is no fallback neighborhood to invent.
    """
    with snapshot(conn):
        bound, held = _current(conn, models_dir, query, now, actor_id)
        position = held.ordinal.get(file_id)
        if position is None:
            return None
        told = _located(conn, bound, held, position)
        # Zero and negatives clamp up to one: a window of nothing is not a window.
        take = min(max(1, count), NEIGHBORHOOD_MOST)
        start = max(0, min(position - take // 2, max(0, len(held.ids) - take)))
        # No caption: `FilmstripItem` renders a 64-pixel square and says in its
        # own docstring that it carries neither a uuid nor a model caption.
        # Hydrating one to throw it away is a join per walk step.
        items = _named(conn, held.ids[start : start + take], start, held.relevance, said=False)
        return {
            **told,
            "first_ordinal": start + 1 if items else 0,
            "last_ordinal": start + len(items),
            "items": items,
        }


#: The most entities one explicit selection may name -- a bound, so an
#: absurd payload is refused instead of exercised.
SUBSET_MOST = 5_000

#: The one uuid spelling rule (db/naming.py, where the fullmatch lesson
#: is recorded once instead of twice).
_HEX_UUID = naming.UUID_HEX


@dataclasses.dataclass(frozen=True)
class SelectionProof:
    """A completed membership proof: these file ids belonged to this
    answer under this library generation. Immutable, so a writer can
    revalidate it with ONE cheap currency comparison instead of holding
    the writer lane through the proof's own work."""

    currency: str
    answer: str
    ids: tuple[int, ...]


def prove_subset(
    conn,
    models_dir: str,
    query: GalleryQuery,
    now: float,
    *,
    actor_id: int | None = None,
    expect_answer: str,
    entity_uuids: list[str],
) -> SelectionProof:
    """Prove a selection against THIS question's current answer.

    Returns a SelectionProof iff the question's answer identity is
    exactly `expect_answer`, every uuid resolves to a live file entity,
    and every file belongs to the answer. Anything else is
    AnswerChanged -- a selection made against one generation must never
    mutate another -- or ValueError for a payload that was never a
    selection at all.

    Nothing here trusts the browser: uuids are exactly 32 hex
    characters by FULLMATCH -- bytes.fromhex skips whitespace, so
    neither a padded spelling nor spaces hiding inside a 32-character
    one may reach it -- the count is bounded, and membership is checked
    against the ONE projection, never a locate per item.

    Deliberately NOT the write transaction: proving may materialize a
    projection -- a whole membership walk, a smart-rule evaluation, a
    semantic encode-FAISS-RRF round -- and none of that may hold
    sqlite's one writer lane. A writer takes the proof, claims the
    lane, compares currency (one monitor read), and mutates only when
    the world the proof described is still the world -- re-proving
    OUTSIDE the lane when it is not.
    """
    if type(expect_answer) is not str or not expect_answer:
        raise ValueError("a selection names the answer it was made against")
    if not isinstance(entity_uuids, (list, tuple)) or not entity_uuids:
        raise ValueError("a selection names at least one entity")
    if len(entity_uuids) > SUBSET_MOST:
        raise ValueError(f"a selection names at most {SUBSET_MOST} entities, not {len(entity_uuids)}")
    keys: list[bytes] = []
    for one in entity_uuids:
        if type(one) is not str or _HEX_UUID.fullmatch(one) is None:
            raise ValueError(f"a selection key is exactly 32 hex characters, not {one!r}")
        keys.append(bytes.fromhex(one))
    keys = list(dict.fromkeys(keys))  # idempotent: naming a file twice is naming it once

    with snapshot(conn):
        _bound, held = _current(conn, models_dir, query, now, actor_id)
        if held.answer != expect_answer:
            raise AnswerChanged("the result set has changed; redraw the gallery and reselect")
        marks = ",".join("?" for _ in keys)
        resolved = {
            row[0]: row[1]
            for row in conn.execute(
                f"SELECT e.uuid, e.id FROM entity e WHERE e.kind = 'file' AND e.uuid IN ({marks})", keys
            )
        }
        missing = [key.hex() for key in keys if key not in resolved]
        if missing:
            raise AnswerChanged(f"{len(missing)} selected file(s) no longer exist; redraw and reselect")
        ids = [resolved[key] for key in keys]
        strays = [file_id for file_id in ids if file_id not in held.ordinal]
        if strays:
            raise AnswerChanged(f"{len(strays)} selected file(s) are not part of this answer; redraw and reselect")
        return SelectionProof(currency=held.currency, answer=held.answer, ids=tuple(ids))


def _named(conn, ids, start: int, relevance: dict[int, float] | None = None, *, said: bool = True) -> list[dict]:
    """One page of members, named.

    `said` off skips the caption read for callers that do not render
    one. The filmstrip is fifteen 64-pixel squares and `FilmstripItem`
    carries neither a uuid nor a caption by design; running
    `derived.said_first` for it is a join over the annotation tables
    whose entire result is discarded. The uuid comes from the same row
    as the slug and costs nothing extra, so only the caption is
    optional.
    """
    if not ids:
        return []
    marks = ",".join("?" for _ in ids)
    from . import derived, settings
    from .context import HUMAN_MOMENT, POLICY_VERSION

    # `dict[str, object]`, stated: a member row is a heterogeneous record of
    # ids, names, a hex uuid, a caption and a float relevance. Inferred from the
    # literal, `said` takes the type `None` and the caption pass cannot write it.
    held: dict[int, dict[str, object]] = {
        row[0]: {
            "id": row[0],
            "slug": row[1],
            "name": row[2],
            "kind": row[3],
            "uuid": row[4].hex(),
            # None until ingest has hashed it; a surface then falls back
            # to the slug route, which can still answer.
            "sha": row[5],
            # The picture's own proportion, for the justified grid. None
            # for a kind with no picture and for a file ingest has not
            # measured yet; the cell draws a square rather than nothing.
            "width": row[6],
            "height": row[7],
            # How many files the dupe job put in this one's group, this
            # one included. None when no group holds it.
            "copies": row[8],
            # Where this picture sits in time, epoch seconds, always a
            # number: the interpretation when there is one, the file's
            # own mtime otherwise. `dated` is which.
            "moment": row[9],
            "dated": bool(row[10]),
            "said": None,
            # How far this file stood above the middle of what a
            # space said, 0..1 -- None when nothing was asked of any
            # space, which is not the same claim as "nothing matched".
            "relevance": None if relevance is None else relevance.get(row[0]),
        }
        for row in conn.execute(NAMED.format(marks=marks, moment=HUMAN_MOMENT, policy=int(POLICY_VERSION)), list(ids))
    }
    if said:
        for file_id, text in derived.said_first(conn, held, prefer=settings.value(conn, "caption_model")).items():
            held[file_id]["said"] = text
    told = []
    for offset, file_id in enumerate(ids):
        row = held.get(file_id)
        if row is not None:
            row["ordinal"] = start + offset + 1
            told.append(row)
    return told
