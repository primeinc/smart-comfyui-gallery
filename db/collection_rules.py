"""What a smart collection MEANS, durably: a typed membership rule.

A GalleryQuery answers "what am I looking at, and in what order?"; a
collection answers "which files belong to me?". Almost the same
question -- so a rule is deliberately SMALLER than a GalleryQuery: no
page, no page size, no viewer, no album reference (a smart collection
inside a smart collection is a dependency graph nobody asked for yet),
and never any executable text.

Three conversions happen exactly once, in `from_gallery_query`:

- entity references are stored by `entity.uuid`, never by slug -- a
  slug is an address spelling that can be renamed and eventually
  reused, and a saved rule must mean the entity that was selected;
- authored facets (favorite, rating_min) PIN the creating actor into
  the rule -- "Will's favorites" means Will's whoever is looking;
- a semantic phrase REQUIRES `take`: ranking the library by similarity
  is not membership until a cutoff makes it a set, and without `take`
  a time sort is irrelevant to membership and is normalized away.

The ResultSet evaluates rules (db/resultset.py); this module never
runs a membership query of its own -- two membership engines is the
fork the whole design exists to avoid.
"""

from __future__ import annotations

import dataclasses
import json

RULE_VERSION = 1


class BrokenCollectionRule(ValueError):
    """The rule references an entity that no longer exists, or its
    stored form cannot be read. Never presented as an empty collection."""


class UnavailableCollectionRule(ValueError):
    """The rule is sound but cannot be answered RIGHT NOW -- a semantic
    rule with no space able to answer. Never presented as empty."""


@dataclasses.dataclass(frozen=True)
class CollectionRule:
    """One smart collection's membership question, durable form."""

    version: int
    folder_uuid: bytes | None
    person_uuid: bytes | None
    kind: str | None
    favorite: bool | None
    rating_min: int | None
    text: str | None
    sort: str | None
    take: int | None
    actor_id: int | None


#: The sort vocabulary a rule may carry -- the ResultSet's own words.
_TIMED_SORTS = ("newest", "oldest")


def validate(rule: CollectionRule, refuse: type[Exception]) -> CollectionRule:
    """The ONE semantic gate, for both directions: a rule being authored
    and a rule read back from storage. `json_valid` is syntax; this is
    meaning -- and a stored rule that fails it is refused (as
    BrokenCollectionRule on load, ValueError on authoring), because a
    semantically rotten rule evaluating to zero rows would masquerade
    as an evaluated empty collection, the exact lie this design bans."""
    from .resultset import KINDS

    if rule.version != RULE_VERSION:
        raise refuse(f"rule version {rule.version!r} is not one this build understands")
    for name, uuid in (("folder", rule.folder_uuid), ("person", rule.person_uuid)):
        if uuid is not None and (not isinstance(uuid, bytes) or len(uuid) != 16):
            raise refuse(f"the rule's {name} reference is not a 16-byte entity uuid")
    if rule.kind is not None and rule.kind not in KINDS:
        raise refuse(f"kind must be one of {', '.join(KINDS)}, not {rule.kind!r}")
    if rule.favorite is not None and not isinstance(rule.favorite, bool):
        raise refuse(f"favorite is true, false or absent, not {rule.favorite!r}")
    if rule.rating_min is not None and (not isinstance(rule.rating_min, int) or not 1 <= rule.rating_min <= 5):
        raise refuse(f"rating_min names the minimum stars, 1..5, not {rule.rating_min!r}")
    if rule.take is not None and (not isinstance(rule.take, int) or not 1 <= rule.take <= 10_000):
        raise refuse(f"take must be 1..10000, not {rule.take!r}")
    if rule.text is not None:
        if not isinstance(rule.text, str) or not rule.text.strip():
            raise refuse("a semantic rule's phrase must be a non-empty string")
        if rule.take is None:
            raise refuse("a semantic rule needs `take`: similarity ranks the library, a cutoff makes a set")
        if rule.sort != "similarity":
            raise refuse(f"a semantic rule orders by similarity, not {rule.sort!r}")
    elif rule.take is not None:
        if rule.sort not in _TIMED_SORTS:
            raise refuse(f"a take-bounded rule orders by one of {', '.join(_TIMED_SORTS)}, not {rule.sort!r}")
    elif rule.sort is not None:
        raise refuse("without a cutoff, order is presentation, not membership; the rule carries no sort")
    asks_authored = rule.favorite is not None or rule.rating_min is not None
    if asks_authored and rule.actor_id is None:
        raise refuse("an authored facet (favorite, rating_min) needs its pinned actor")
    if not asks_authored and rule.actor_id is not None:
        raise refuse("an actor is pinned only when an authored facet needs one")
    return rule


def _entity_uuid(conn, kind: str, slug: str) -> bytes:
    row = conn.execute("SELECT e.uuid FROM entity e WHERE e.kind = ? AND e.slug = ?", (kind, slug)).fetchone()
    if row is None:
        raise ValueError(f"no {kind} at {slug!r} to save into a rule")
    return row[0]


def from_gallery_query(conn, query, *, actor_id: int | None, take: int | None) -> CollectionRule:
    """The one place a spelled question becomes a durable rule.

    Refusals are loud: an album scope (no smart-in-smart), a semantic
    phrase without `take`, an authored facet with no actor to pin.
    Page geometry never enters; a time sort without `take` does not
    affect membership and is normalized out.
    """
    if query.album is not None:
        raise ValueError("a rule cannot reference a collection; smart-in-smart is not a v1 question")
    asks_authored = query.favorite is not None or query.rating_min is not None
    sort = query.sort
    if take is None and query.text is None:
        sort = None  # without a cutoff, order is presentation, not membership
    made = CollectionRule(
        version=RULE_VERSION,
        folder_uuid=_entity_uuid(conn, "folder", query.folder) if query.folder else None,
        person_uuid=_entity_uuid(conn, "person", query.person) if query.person else None,
        kind=query.kind,
        favorite=query.favorite,
        rating_min=query.rating_min,
        text=query.text,
        sort=sort,
        take=None if take is None else int(take),
        actor_id=actor_id if asks_authored else None,
    )
    return validate(made, ValueError)


def _smart_only(conn, collection_id: int) -> None:
    row = conn.execute("SELECT kind FROM collection WHERE id = ?", (collection_id,)).fetchone()
    if row is None or row[0] != "smart":
        raise ValueError("only a smart collection carries a rule -- a listed collection's membership is its filed rows")


def save(conn, collection_id: int, rule: CollectionRule, *, source_text: str | None, now: float) -> None:
    """The whole rule, as desired state: one row, one new version of the
    collection's meaning -- never predicate-by-predicate edits."""
    _smart_only(conn, collection_id)
    told = json.dumps(
        {
            "v": rule.version,
            "where": {
                "folder": rule.folder_uuid.hex() if rule.folder_uuid else None,
                "person": rule.person_uuid.hex() if rule.person_uuid else None,
                "kind": rule.kind,
                "favorite": rule.favorite,
                "rating_min": rule.rating_min,
            },
            "select": {"sort": rule.sort, "text": rule.text, "take": rule.take},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    conn.execute(
        "INSERT INTO collection_rule(collection_id, rule_version, rule_json, actor_id,"
        " source_text, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(collection_id) DO UPDATE SET rule_version = excluded.rule_version,"
        " rule_json = excluded.rule_json, actor_id = excluded.actor_id,"
        " source_text = excluded.source_text, updated_at = excluded.updated_at",
        (collection_id, rule.version, told, rule.actor_id, source_text, now, now),
    )


def load(conn, collection_id: int) -> CollectionRule | None:
    """The rule, or None when the collection is UNEVALUATED -- no row,
    or a legacy row holding only preserved prose."""
    row = conn.execute(
        "SELECT rule_version, rule_json, actor_id FROM collection_rule WHERE collection_id = ?",
        (collection_id,),
    ).fetchone()
    if row is None or row[1] is None:
        return None
    version, told, actor_id = row
    try:
        held = json.loads(told)
        where, select = held["where"], held["select"]
        made = CollectionRule(
            version=int(version),
            folder_uuid=bytes.fromhex(where["folder"]) if where["folder"] else None,
            person_uuid=bytes.fromhex(where["person"]) if where["person"] else None,
            kind=where["kind"],
            favorite=where["favorite"],
            rating_min=where["rating_min"],
            text=select["text"],
            sort=select["sort"],
            take=select["take"],
            actor_id=actor_id,
        )
    except (KeyError, TypeError, ValueError) as rotten:
        raise BrokenCollectionRule(f"collection {collection_id}'s stored rule cannot be read: {rotten}") from rotten
    if held.get("v") != version:
        raise BrokenCollectionRule(
            f"collection {collection_id}'s stored form says v{held.get('v')!r} under a v{version!r} column"
        )
    return validate(made, BrokenCollectionRule)


def keep_prose(conn, collection_id: int, *, nl: str | None = None, sql: str | None = None, now: float) -> None:
    """A rule's human prose WITHOUT a typed rule -- the state migration
    leaves legacy smart collections in, and the explicit UNEVALUATED
    form. Nothing here will ever run."""
    _smart_only(conn, collection_id)
    conn.execute(
        "INSERT INTO collection_rule(collection_id, source_text, legacy_sql_text, created_at, updated_at)"
        " VALUES(?, ?, ?, ?, ?)"
        " ON CONFLICT(collection_id) DO UPDATE SET source_text = excluded.source_text,"
        " legacy_sql_text = excluded.legacy_sql_text, updated_at = excluded.updated_at",
        (collection_id, nl, sql, now, now),
    )


def provenance(conn, collection_id: int) -> dict | None:
    """The human-readable half: what was written down about this rule --
    shown when the rule is unevaluated, kept as provenance always."""
    row = conn.execute(
        "SELECT source_text, legacy_sql_text, rule_json IS NOT NULL FROM collection_rule WHERE collection_id = ?",
        (collection_id,),
    ).fetchone()
    if row is None:
        return None
    return {"nl": row[0], "sql": row[1], "evaluated": bool(row[2])}
