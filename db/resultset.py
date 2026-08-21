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

import contextlib
import dataclasses
import hashlib
import json
import sqlite3
import threading

#: The orders a query may ask for. "similarity" requires a phrase; the
#: time sorts follow the file table's own indexes.
SORTS = ("newest", "oldest", "similarity")

#: The file kinds a query may filter to -- the vocabulary of file.kind.
KINDS = ("image", "animated_image", "video", "audio", "document")

DEFAULT_PAGE_SIZE = 60
MAX_PAGE_SIZE = 400

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
    kind: str | None = None  # filter: one file kind
    text: str | None = None  # the semantic phrase; implies sort=similarity
    sort: str = "newest"
    size: int = DEFAULT_PAGE_SIZE


def parse(
    *,
    folder: str | None = None,
    album: str | None = None,
    person: str | None = None,
    kind: str | None = None,
    text: str | None = None,
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
        raise ValueError("one scope at a time: folder or album, not both")
    # `person` deliberately COMPOSES with either -- and with kind and a
    # phrase: eligibility is an intersection of predicates, and a
    # person's beach pictures in one album is a real question.
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}, not {kind!r}")
    chosen = DEFAULT_PAGE_SIZE if size is None else int(size)
    if not 1 <= chosen <= MAX_PAGE_SIZE:
        raise ValueError(f"page size must be 1..{MAX_PAGE_SIZE}, not {chosen}")
    return GalleryQuery(folder=folder, album=album, person=person, kind=kind, text=text, sort=sort, size=chosen)


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
    ids: tuple[int, ...]  # file ids, answer order
    ordinal: dict[int, int]  # file id -> 0-based position
    provenance: dict | None  # similarity only: participants/contributors/missing


#: (database, fingerprint, currency) -> Projection, oldest evicted first.
_PROJECTIONS: dict[tuple[str, str, str], Projection] = {}
_PROJECTION_LOCK = threading.Lock()

#: The names one page of cells needs, id-keyed; order is restored from
#: the projection slice, so this query carries none.
NAMED = "SELECT f.id, e.slug, f.name, f.kind FROM file f JOIN entity e ON e.id = f.id WHERE f.id IN ({marks})"


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
    if query.kind:
        pairs.append(("kind", query.kind))
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


def bind(conn, query: GalleryQuery) -> _Bound:
    """Resolve every slug to its entity -- retired spellings included,
    refusing an address nothing lives at: an empty page at a misspelled
    folder would look exactly like an empty folder."""
    from . import naming

    held: dict[str, int | None] = {"folder": None, "album": None, "person": None}
    live: dict[str, str] = {}
    for field, entity_kind in (("folder", "folder"), ("album", "collection"), ("person", "person")):
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
    run = None
    if held["person"] is not None:
        row = conn.execute("SELECT id FROM derived_face_run WHERE is_primary = 1").fetchone()
        run = row[0] if row else None
    return _Bound(
        query=dataclasses.replace(query, **live) if live else query,
        folder_id=held["folder"],
        collection_id=held["album"],
        person_id=held["person"],
        face_run_id=run,
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
            "run": bound.face_run_id,
            "kind": query.kind,
            "text": query.text,
            "sort": query.sort,
            "size": query.size,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(told.encode()).hexdigest()[:16]


def _timed_ids(conn, bound: _Bound) -> list[int]:
    """Eligibility is an INTERSECTION of predicates, constructed -- not
    a choice between fixed statements. `person` composes with a folder,
    an album, a kind, and (via the allowed set) a phrase: a file
    satisfies person=jane iff the bound primary run attributes it to
    her, exactly what /people and the profile already mean. The walk
    stays ordered by the file table's own time index; the whole
    statement runs once per library change, never once per page."""
    query = bound.query
    order = "ASC" if query.sort == "oldest" else "DESC"
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
            return []
        where.append(
            "EXISTS (SELECT 1 FROM derived_file_person fp"
            " WHERE fp.file_id = f.id AND fp.person_id = ? AND fp.run_id = ?)"
        )
        args.extend((bound.person_id, bound.face_run_id))
    if query.kind is not None:
        where.append("f.kind = ?")
        args.append(query.kind)
    sql = f"SELECT f.id FROM file f WHERE {' AND '.join(where)} ORDER BY f.mtime {order}, f.id {order}"
    return [row[0] for row in conn.execute(sql, args)]


def _fused_ids(conn, models_dir: str, bound: _Bound, now: float) -> tuple[list[int], dict | None]:
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
    if any(x is not None for x in (bound.folder_id, bound.collection_id, bound.person_id, query.kind)):
        eligible = dataclasses.replace(bound, query=dataclasses.replace(query, text=None, sort="newest"))
        allowed = set(_timed_ids(conn, eligible))
        if not allowed:
            # An empty scope needs no encoder and has no honest
            # provenance -- nothing was asked of any space.
            return [], None
    depth = len(allowed) if allowed is not None else _present(conn)
    found = retrieval.query(conn, models_dir, query.text, max(depth, 1), now, offline=True, allowed=allowed)
    fused = [row["file_id"] for row in found["results"]]
    provenance = {key: found[key] for key in ("participants", "contributors", "missing")}
    return fused, provenance


def _present(conn) -> int:
    return conn.execute("SELECT count(*) FROM file WHERE missing_since IS NULL").fetchone()[0]


def _current(conn, models_dir: str, query: GalleryQuery, now: float) -> tuple[_Bound, Projection]:
    """The projection for this question over the library as it stands --
    a stale one is never reused, it is replaced. Currency is read BEFORE
    binding: bind's resolves are this connection's first data reads and
    pin the snapshot, so a commit in the gap builds fresh data under an
    obsolete key -- wasted work the next request replaces, never stale
    data cached under a fresh key."""
    database = _database_file(conn) or f"mem{id(conn)}"
    told = currency(conn)
    bound = bind(conn, query)
    key = (database, _bound_fingerprint(bound), told)
    with _PROJECTION_LOCK:
        held = _PROJECTIONS.get(key)
    if held is not None:
        return bound, held
    if query.sort == "similarity":
        ids, provenance = _fused_ids(conn, models_dir, bound, now)
    else:
        ids, provenance = _timed_ids(conn, bound), None
    made = Projection(
        fingerprint=key[1],
        currency=key[2],
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
        "provenance": held.provenance,
        "qs": canonical(query),
    }


def describe(conn, models_dir: str, query: GalleryQuery, now: float) -> dict:
    """The result set's shape: what the rail is drawn from and what the
    grid's pager believes. `currency` rides along so a client can tell
    a redrawn answer from the one it is holding."""
    with snapshot(conn):
        bound, held = _current(conn, models_dir, query, now)
        return _shape(bound, held)


def page(conn, models_dir: str, query: GalleryQuery, number: int, now: float) -> dict:
    """One page of the answer, by number. A number past the end answers
    with the last page that exists -- the library may have shrunk since
    the rail was drawn, and the honest response is the page that IS,
    named as itself."""
    with snapshot(conn):
        bound, held = _current(conn, models_dir, query, now)
        shape = _shape(bound, held)
        number = min(max(1, int(number)), shape["pages"])
        start = (number - 1) * bound.query.size
        shape["page"] = number
        shape["items"] = _named(conn, held.ids[start : start + bound.query.size], start)
        return shape


def peek(conn, models_dir: str, query: GalleryQuery, number: int, now: float, count: int = PEEK_MOST) -> dict:
    """The rail popover's preview: the first few members of EXACTLY the
    page a jump would land on -- by construction a prefix of what
    `page` answers, and the test suite holds the two to it."""
    with snapshot(conn):
        bound, held = _current(conn, models_dir, query, now)
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
            "qs": shape["qs"],
            "items": _named(conn, held.ids[start : start + take], start),
        }


def locate(conn, models_dir: str, query: GalleryQuery, file_id: int, now: float) -> dict | None:
    """Where one file sits in the answer -- its ordinal, its page, and
    its neighbours in ANSWER order, which is what previous/next mean
    while a result set is being walked. None when the file is not in
    the membership at all."""
    with snapshot(conn):
        bound, held = _current(conn, models_dir, query, now)
        position = held.ordinal.get(int(file_id))
        if position is None:
            return None
        neighbours = [held.ids[at] if 0 <= at < len(held.ids) else None for at in (position - 1, position + 1)]
        named = {row["id"]: row["slug"] for row in _named(conn, [n for n in neighbours if n is not None], 0)}
        return {
            "ordinal": position + 1,
            "page": position // bound.query.size + 1,
            "total": len(held.ids),
            "currency": held.currency,
            "qs": canonical(bound.query),
            "previous": named.get(neighbours[0]),
            "next": named.get(neighbours[1]),
        }


def _named(conn, ids, start: int) -> list[dict]:
    if not ids:
        return []
    marks = ",".join("?" for _ in ids)
    held = {
        row[0]: {"id": row[0], "slug": row[1], "name": row[2], "kind": row[3]}
        for row in conn.execute(NAMED.format(marks=marks), list(ids))
    }
    told = []
    for offset, file_id in enumerate(ids):
        row = held.get(file_id)
        if row is not None:
            row["ordinal"] = start + offset + 1
            told.append(row)
    return told
