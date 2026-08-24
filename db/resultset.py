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
"""

from __future__ import annotations

import atexit
import contextlib
import dataclasses
import hashlib
import json
import re
import sqlite3
import threading

#: The orders a query may ask for. "similarity" requires a phrase; the
#: time sorts follow the file table's own indexes; the moment sorts
#: follow the human timeline (db/context.py HUMAN_MOMENT) -- the axis a
#: timeline link opened -- with uninterpreted files last, never dropped.
SORTS = ("newest", "oldest", "moment", "moment-newest", "similarity")

#: The file kinds a query may filter to -- the vocabulary of file.kind.
KINDS = ("image", "animated_image", "video", "audio", "document")

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
    #: Authored facets: the asking ACTOR's own judgement, composable like
    #: any predicate. The actor never rides the URL -- it binds at answer
    #: time and lives in the projection identity, so two people asking
    #: "favorite=1" share a spelling and never an answer.
    favorite: bool | None = None  # True: favorited; False: NOT favorited
    rating_min: int | None = None  # at least this many stars from the actor, 1..5
    text: str | None = None  # the semantic phrase; implies sort=similarity
    #: Registered metadata predicates (db/context.py owns the
    #: vocabulary): each composes like any other facet, timed and
    #: semantic alike. Canonically ordered, so two spellings of one
    #: conjunction are one question.
    facets: tuple = ()
    sort: str = "newest"
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
    if rating_min is not None and not 1 <= int(rating_min) <= 5:
        raise ValueError(f"rating_min names the minimum stars, 1..5, not {rating_min!r}")
    chosen = DEFAULT_PAGE_SIZE if size is None else int(size)
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
        rating_min=None if rating_min is None else int(rating_min),
        text=text,
        facets=held,
        sort=sort,
        size=chosen,
    )


def fingerprint(query: GalleryQuery) -> str:
    """The identity of the question AS SPELLED: canonical JSON over
    every field, hashed. Page size is part of it because ordinal->page
    arithmetic is. The projection cache does NOT key on this -- it keys
    on `_bound_fingerprint`, over stable entity ids, so a renamed
    person's two spellings stay one cached question."""
    told = json.dumps(dataclasses.asdict(query), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(told.encode()).hexdigest()[:16]


# --- data currency ----------------------------------------------------------

#: One read-only monitor connection per database file, held for the
#: process. `PRAGMA data_version` is only comparable to itself on the
#: same connection, so the connection that answers it must never be a
#: per-request one.
_MONITORS: dict[str, sqlite3.Connection] = {}
_MONITOR_LOCK = threading.Lock()


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


# The PROCESS owns these, not an application instance: `_MONITORS` is
# module state, so two Litestar apps in one process share it and a
# shutdown hook on either would close the other's monitors. atexit is the
# lifetime that actually matches.
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
    """
    from . import connect

    where = _database_file(conn)
    if not where:
        return f"mem{id(conn)}.{conn.total_changes}"
    with _MONITOR_LOCK:
        monitor = _MONITORS.get(where)
        if monitor is None:
            monitor = _MONITORS[where] = connect.connect(where, read_only=True, cross_thread=True)
        return f"v{monitor.execute('PRAGMA data_version').fetchone()[0]}"


# --- the projection ---------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Projection:
    fingerprint: str
    currency: str
    #: The identity of the ORDERED ANSWER itself, as against `currency`,
    #: which is the identity of the library generation it was computed
    #: from. `data_version` moves on EVERY commit -- a favorite, a
    #: rating -- while most commits leave most answers untouched; a
    #: client holding (currency, answer) can tell "the library moved but
    #: this answer did not" (adopt the new currency in place) from "the
    #: answer really changed" (redraw), without anyone teaching it which
    #: tables affect which queries.
    answer: str
    ids: tuple[int, ...]  # file ids, answer order
    ordinal: dict[int, int]  # file id -> 0-based position
    provenance: dict | None  # similarity only: participants/contributors/missing


#: (database, fingerprint, currency) -> Projection, oldest evicted first.
_PROJECTIONS: dict[tuple[str, str, str], Projection] = {}
_PROJECTION_LOCK = threading.Lock()

#: The names one page of cells needs, id-keyed; order is restored from
#: the projection slice, so this query carries none.
#: One row per member, and it carries `content_sha256` because THE
#: THUMBNAIL'S IDENTITY IS RESOLVED HERE, ONCE, for the whole page.
#:
#: The derivative cache is content-addressed (vision/thumbs.py
#: `path_for`), so the hash IS the asset's address. Carrying it means a
#: cell's `src` can point straight at an immutable file instead of at a
#: route that opens a connection, resolves the slug, reads the kind and
#: the hash back out of the database, and only then knows which file to
#: send. Sixty cells were sixty of those. This is the same shape
#: PhotoPrism and Immich settled on: resolve once, serve statically.
NAMED = (
    "SELECT f.id, e.slug, f.name, f.kind, e.uuid, f.content_sha256"
    " FROM file f JOIN entity e ON e.id = f.id WHERE f.id IN ({marks})"
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
    #: The artifact facet, bound: the stable entity id plus the
    #: artifact's OWN kind, which privately decides whether membership
    #: means file_artifact or generation.workflow_id. Nothing above this
    #: module ever sees that split.
    artifact_id: int | None
    artifact_kind: str | None
    #: WHOSE judgement the authored facets mean. Set only when the query
    #: carries one, so questions without an authored facet stay one
    #: cached projection however many actors ask them.
    actor_id: int | None
    #: A SMART collection scope, bound: the rule's own question as an
    #: inner _Bound plus its take. The rule owns MEMBERSHIP (evaluated to
    #: a set); the outer question still owns the ordered answer --
    #: `collection_id` stays None for smart, because collection_file
    #: holds nothing to EXISTS against.
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
            # A smart collection's membership is its RULE's answer. No
            # typed rule yet -- migrated prose, or nothing -- stays an
            # UNEVALUATED collection, never an empty one; a rule whose
            # references rot is BROKEN, never empty (db/collection_rules).
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
    return _Bound(
        query=dataclasses.replace(query, **live) if live else query,
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
            "size": query.size,
            # A smart scope's identity is its BOUND rule (recursively
            # fingerprinted -- ids, pinned actor, run) plus its take: two
            # collections with one rule are one membership question, and
            # an edited rule is a different one.
            "rule": None if bound.rule is None else [_bound_fingerprint(bound.rule[0]), bound.rule[1]],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(told.encode()).hexdigest()[:16]


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
        # The artifact's canonical relation, decided by ITS kind: a
        # workflow attaches through generation, weights and equipment
        # through file_artifact -- role-blind, and EXISTS makes a LoRA
        # stacked at two ordinals one media member.
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

        # Through `clauses`, never `predicate` in a loop: repeating a key
        # with `any` means OR and has to arrive here as ONE clause. A
        # loop that appended each predicate separately would AND them,
        # and "image or video" would silently answer nothing.
        for sql, values in facets_module.clauses(query.facets):
            where.append(sql)
            args.extend(values)
    return where, args, len(where) > 1


def _timed_ids(conn, bound: _Bound) -> list[int]:
    """The ordered walk over `_eligibility`'s membership. The walk stays
    on the file table's own time index; the whole statement runs once
    per library change, never once per page."""
    # The ORDERING CONTRACT: (mtime, id) both in the sort's direction --
    # the same contract file_in_folder_by_time carries, so global and
    # folder-scoped questions tie identically. The indexes implement
    # this spelling (file_recent is (mtime DESC, id DESC), schema v6),
    # never the reverse: bending the tiebreak to fit an index once
    # silently changed real answer identities and ordinals.
    where, args, _ = _eligibility(bound)
    if bound.query.sort in ("moment", "moment-newest"):
        # The human moment, not the filesystem's: what a timeline link
        # means by "this day" is the order its pictures come back in.
        # LEFT JOIN keeps membership identical to the other sorts; a
        # file with no interpretation sorts last and says so by position.
        from .context import HUMAN_MOMENT, POLICY_VERSION

        order = "ASC" if bound.query.sort == "moment" else "DESC"
        sql = (
            f"SELECT f.id FROM file f LEFT JOIN derived_media_context mc"
            f" ON mc.file_id = f.id AND mc.policy_version = {int(POLICY_VERSION)}"
            f" WHERE {' AND '.join(where)}"
            f" ORDER BY {HUMAN_MOMENT} IS NULL, {HUMAN_MOMENT} {order}, f.id {order}"
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
            ids, _ = _fused_ids(conn, models_dir, inner, now)
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
) -> tuple[list[int], dict | None]:
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
        # A smart scope's membership INTERSECTS the outer eligibility --
        # evaluated to a set precisely so a rule's person and the
        # viewer's person stay a conjunction instead of one field
        # overwriting the other.
        allowed = members if allowed is None else allowed & members
    if allowed is not None and not allowed:
        # An empty scope needs no encoder and has no honest
        # provenance -- nothing was asked of any space.
        return [], None
    if query.text is None:
        # no phrase, no semantic ordering: nothing was asked of any space
        return [], None
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
    fused = [row["file_id"] for row in found["results"]]
    provenance = {key: found[key] for key in ("participants", "contributors", "missing")}
    provenance["unmatched"] = found.get("unmatched") or {}
    return fused, provenance


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
        ids, provenance = _fused_ids(conn, models_dir, bound, now, members=members)
    else:
        ids, provenance = _timed_ids(conn, bound), None
        if members is not None:
            # The rule owns membership; the OUTER walk keeps its order.
            ids = [file_id for file_id in ids if file_id in members]
    made = Projection(
        fingerprint=key[1],
        currency=key[2],
        answer=hashlib.sha256(",".join(str(file_id) for file_id in ids).encode()).hexdigest()[:16],
        ids=tuple(ids),
        ordinal={file_id: position for position, file_id in enumerate(ids)},
        provenance=provenance,
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
        number = min(max(1, int(number)), shape["pages"])
        start = (number - 1) * bound.query.size
        shape["page"] = number
        shape["items"] = _named(conn, held.ids[start : start + bound.query.size], start)
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
        number = min(max(1, int(number)), shape["pages"])
        start = (number - 1) * bound.query.size
        take = min(max(1, int(count)), PEEK_MOST, bound.query.size)
        return {
            "page": number,
            "pages": shape["pages"],
            "total": shape["total"],
            "first_ordinal": min(start + 1, max(shape["total"], 1)),
            "last_ordinal": min(start + bound.query.size, shape["total"]),
            "currency": held.currency,
            "answer": held.answer,
            "qs": shape["qs"],
            "items": _named(conn, held.ids[start : start + take], start),
        }


def _located(conn, bound: _Bound, held: Projection, position: int) -> dict:
    """Where a position sits in an answer, from one projection.

    Shared rather than repeated: `locate` and `neighborhood` answer the
    same question about the same walk, and two spellings of "previous"
    are two chances for the arrows and the strip beneath them to
    disagree about what comes next.
    """
    neighbours = [held.ids[at] if 0 <= at < len(held.ids) else None for at in (position - 1, position + 1)]
    named = {row["id"]: row["slug"] for row in _named(conn, [n for n in neighbours if n is not None], 0)}
    return {
        "ordinal": position + 1,
        "page": position // bound.query.size + 1,
        "total": len(held.ids),
        "currency": held.currency,
        "answer": held.answer,
        "qs": canonical(bound.query),
        "previous": named.get(neighbours[0]),
        "next": named.get(neighbours[1]),
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
        position = held.ordinal.get(int(file_id))
        return None if position is None else _located(conn, bound, held, position)


#: The widest neighborhood a caller may ask for. A bound, so an absurd
#: `count` is refused rather than answered with a page of thumbnails.
NEIGHBORHOOD_MOST = 51


def neighborhood(
    conn,
    models_dir: str,
    query: GalleryQuery,
    file_id: int,
    now: float,
    count: int = 15,
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
        position = held.ordinal.get(int(file_id))
        if position is None:
            return None
        told = _located(conn, bound, held, position)
        take = min(max(1, int(count)), NEIGHBORHOOD_MOST)
        start = max(0, min(position - take // 2, max(0, len(held.ids) - take)))
        items = _named(conn, held.ids[start : start + take], start)
        return {
            **told,
            "first_ordinal": start + 1 if items else 0,
            "last_ordinal": start + len(items),
            "items": items,
        }


#: The most entities one explicit selection may name -- a bound, so an
#: absurd payload is refused instead of exercised.
SUBSET_MOST = 5_000

#: Exactly 32 hex characters -- a fullmatch, because bytes.fromhex
#: skips whitespace and a raw-length check alone lets two spaces hide
#: INSIDE a 32-character spelling and decode to 15 bytes.
_HEX_UUID = re.compile(r"[0-9a-fA-F]{32}")


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


def _named(conn, ids, start: int) -> list[dict]:
    if not ids:
        return []
    marks = ",".join("?" for _ in ids)
    from . import derived, settings

    held = {
        row[0]: {
            "id": row[0],
            "slug": row[1],
            "name": row[2],
            "kind": row[3],
            "uuid": row[4].hex(),
            # None until ingest has hashed it; a surface then falls back
            # to the slug route, which can still answer.
            "sha": row[5],
            "said": None,
        }
        for row in conn.execute(NAMED.format(marks=marks), list(ids))
    }
    for file_id, text in derived.said_first(conn, held, prefer=settings.value(conn, "caption_model")).items():
        held[file_id]["said"] = text
    told = []
    for offset, file_id in enumerate(ids):
        row = held.get(file_id)
        if row is not None:
            row["ordinal"] = start + offset + 1
            told.append(row)
    return told
