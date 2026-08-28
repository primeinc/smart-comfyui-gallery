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

from . import facets as facets_module
from . import naming

#: What this build AUTHORS. Reading is wider: `_KNOWN_VERSIONS` -- a
#: stored v1 rule keeps meaning exactly what it meant, and "versioned"
#: means the reader dispatches on the version instead of quietly
#: reinterpreting old rows under a new shape.
RULE_VERSION = 3

_KNOWN_VERSIONS = (1, 2, 3)


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
    #: v2: one artifact entity by uuid -- a checkpoint, LoRA or workflow
    #: facet saved as membership. Always None in a v1 rule.
    artifact_uuid: bytes | None
    kind: str | None
    favorite: bool | None
    rating_min: int | None
    text: str | None
    sort: str | None
    take: int | None
    actor_id: int | None
    #: v3: registered metadata predicates (db/facets.py), the facets a
    #: gallery question carries, saved as membership. Always empty in a
    #: v1 or v2 rule. Never `event.id`: a session is a run's hypothesis
    #: over one interpretation, and a rule holding one would answer
    #: nothing the day the runs regroup -- an empty collection wearing a
    #: saved view's clothes.
    facets: tuple = ()


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

    # Exact-type integer checks throughout (`type(x) is int`), because
    # Python's bool IS an int and JSON true would otherwise pass as 1 --
    # the truthiness corner a "semantic gate" exists to close.
    if type(rule.version) is not int or rule.version not in _KNOWN_VERSIONS:
        raise refuse(f"rule version {rule.version!r} is not one this build understands")
    if rule.version == 1 and rule.artifact_uuid is not None:
        raise refuse("a v1 rule has no artifact reference; that shape arrived in v2")
    if rule.version < 3 and rule.facets:
        raise refuse(f"a v{rule.version} rule carries no metadata facets; that shape arrived in v3")
    if not isinstance(rule.facets, tuple) or any(not isinstance(one, facets_module.Facet) for one in rule.facets):
        raise refuse("a rule's facets are a tuple of registered Facet predicates")
    for one in rule.facets:
        if one.key == "event.id":
            raise refuse(
                "a session is a hypothesis, not a durable membership; save its day (context.local_day)"
                " or its moment window (context.moment) instead"
            )
    for name, uuid in (
        ("folder", rule.folder_uuid),
        ("person", rule.person_uuid),
        ("artifact", rule.artifact_uuid),
    ):
        if uuid is not None and (not isinstance(uuid, bytes) or len(uuid) != 16):
            raise refuse(f"the rule's {name} reference is not a 16-byte entity uuid")
    if rule.kind is not None and rule.kind not in KINDS:
        raise refuse(f"kind must be one of {', '.join(KINDS)}, not {rule.kind!r}")
    if rule.favorite is not None and not isinstance(rule.favorite, bool):
        raise refuse(f"favorite is true, false or absent, not {rule.favorite!r}")
    if rule.rating_min is not None and (type(rule.rating_min) is not int or not 1 <= rule.rating_min <= 5):
        raise refuse(f"rating_min names the minimum stars, 1..5, not {rule.rating_min!r}")
    if rule.take is not None and (type(rule.take) is not int or rule.take < 1):
        raise refuse(f"take is a rank to stop at, at least 1, not {rule.take!r}")
    # No ceiling. `take` names the LAST RANK that belongs, and the rule
    # is applied by slicing an ordered list (db/resultset.py
    # `_rule_members`), so a number larger than the library costs
    # nothing and means the only thing it can mean: all of them. The
    # old 1..10000 bound refused questions it could have answered --
    # against a 3,748-file library every number above 2,995 named the
    # same set -- and it announced itself only by rejecting whatever a
    # person had already typed.
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


#: The one uuid spelling rule (db/naming.py, where the fullmatch lesson
#: is recorded once instead of twice).
_UUID_HEX = naming.UUID_HEX

#: The durable vocabulary, EXACT per version: a stored rule carrying a
#: key this build does not understand means something this build cannot
#: evaluate -- BROKEN, never "evaluated after quietly throwing that
#: meaning away". Fail-closed is the whole point of a typed rule.
_TOP_KEYS = frozenset({"v", "where", "select"})
_WHERE_KEYS = {
    1: frozenset({"folder", "person", "kind", "favorite", "rating_min"}),
    2: frozenset({"folder", "person", "artifact", "kind", "favorite", "rating_min"}),
    3: frozenset({"folder", "person", "artifact", "kind", "favorite", "rating_min", "facets"}),
}
_SELECT_KEYS = frozenset({"sort", "text", "take"})


def _versioned_shape(version: int, held) -> tuple[dict, dict]:
    """The stored form's key sets, held exactly -- unknown fields are
    refusals, missing fields are refusals, per version."""
    if not isinstance(held, dict) or set(held) != _TOP_KEYS:
        raise ValueError("the stored form does not have the versioned top-level shape")
    where, select = held["where"], held["select"]
    expected = _WHERE_KEYS.get(version)
    if expected is None:
        raise ValueError(f"v{version!r} is not a shape this build reads")
    if not isinstance(where, dict) or set(where) != expected:
        raise ValueError(f"a v{version} rule's predicates are exactly {', '.join(sorted(expected))}")
    if not isinstance(select, dict) or set(select) != _SELECT_KEYS:
        raise ValueError(f"a rule's selection clause is exactly {', '.join(sorted(_SELECT_KEYS))}")
    return where, select


def _stored_uuid(value, field: str) -> bytes | None:
    """Only actual JSON null means unconstrained: an empty string, a
    false, or any other falsy value is CORRUPTION, not the absence of a
    reference -- and the spelling is exactly 32 hex characters BEFORE
    the decoder (which forgives whitespace) ever sees it."""
    if value is None:
        return None
    if type(value) is not str or _UUID_HEX.fullmatch(value) is None:
        raise ValueError(f"the rule's {field} reference is not a 32-hex entity uuid")
    return bytes.fromhex(value)


def _stored_facets(spelled) -> tuple:
    """A list of spellings, each re-validated against the CURRENT
    registry: a key this build no longer knows is a rule this build
    cannot evaluate -- refused, never quietly dropped."""
    if not isinstance(spelled, list) or any(type(one) is not str for one in spelled):
        raise ValueError("the rule's filters are a list of key:op:value strings")
    held = facets_module.normalized(spelled)
    if len(held) != len(spelled):
        raise ValueError("the rule's filters hold a duplicate or an empty one")
    return held


def _entity_uuid(conn, kind: str, slug: str) -> bytes:
    """Any spelling ResultSet recognizes is also legal when the question
    becomes durable meaning: retired slugs resolve to the same entity,
    and the uuid -- the identity itself -- is what gets stored. A save
    that refused the spelling whose answer is on screen would be
    healing's opposite."""
    from .naming import resolve

    found = resolve(conn, kind, slug)
    if found is None:
        raise ValueError(f"no {kind} at {slug!r} to save into a rule")
    return conn.execute("SELECT uuid FROM entity WHERE id = ?", (found[0],)).fetchone()[0]


def from_gallery_query(conn, query, *, actor_id: int | None, take: int | None) -> CollectionRule:
    """The one place a spelled question becomes a durable rule.

    Refusals are loud: an album scope (no smart-in-smart), a semantic
    phrase without `take`, an authored facet with no actor to pin.
    Page geometry never enters; a time sort without `take` does not
    affect membership and is normalized out.
    """
    if query.album is not None:
        raise ValueError("a smart album cannot be built from another album")
    if take is not None and type(take) is not int:
        # Exact-integer BEFORE any coercion: int(True) is 1, and a
        # boolean quietly becoming a one-item cutoff is the truthiness
        # species this module exists to refuse.
        raise ValueError(f"take is an exact integer, not {take!r}")
    asks_authored = query.favorite is not None or query.rating_min is not None
    sort = query.sort
    if take is None and query.text is None:
        sort = None  # without a cutoff, order is presentation, not membership
    made = CollectionRule(
        version=RULE_VERSION,
        folder_uuid=_entity_uuid(conn, "folder", query.folder) if query.folder else None,
        person_uuid=_entity_uuid(conn, "person", query.person) if query.person else None,
        artifact_uuid=_entity_uuid(conn, "artifact", query.artifact) if query.artifact else None,
        kind=query.kind,
        favorite=query.favorite,
        rating_min=query.rating_min,
        text=query.text,
        sort=sort,
        take=take,
        actor_id=actor_id if asks_authored else None,
        facets=tuple(query.facets),
    )
    return validate(made, ValueError)


def _smart_only(conn, collection_id: int) -> None:
    row = conn.execute("SELECT kind FROM collection WHERE id = ?", (collection_id,)).fetchone()
    if row is None or row[0] != "smart":
        raise ValueError("only a smart collection carries a rule -- a listed collection's membership is its filed rows")


def save(conn, collection_id: int, rule: CollectionRule, *, source_text: str | None, now: float) -> None:
    """The whole rule, as desired state: one row, one new version of the
    collection's meaning -- never predicate-by-predicate edits.

    Validated HERE, not only in from_gallery_query: the persistence
    interface owns its invariant, or correctness depends on every
    future caller remembering which constructor blessed the object."""
    _smart_only(conn, collection_id)
    rule = validate(rule, ValueError)
    if rule.version != RULE_VERSION:
        raise ValueError(f"this build authors v{RULE_VERSION} rules; v{rule.version} is read-only")
    told = json.dumps(
        {
            "v": rule.version,
            "where": {
                "folder": rule.folder_uuid.hex() if rule.folder_uuid else None,
                "person": rule.person_uuid.hex() if rule.person_uuid else None,
                "artifact": rule.artifact_uuid.hex() if rule.artifact_uuid else None,
                "kind": rule.kind,
                "favorite": rule.favorite,
                "rating_min": rule.rating_min,
                "facets": [facets_module.spell(one) for one in rule.facets],
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
        # The reader is per-version and the shape is EXACT: v1 has no
        # artifact key (one appearing there is corruption, not an
        # upgrade), v2 requires it, and a key from any future this build
        # does not understand refuses instead of evaluating without it.
        where, select = _versioned_shape(int(version), held)
        artifact_stored = None if int(version) == 1 else _stored_uuid(where["artifact"], "artifact")
        facets_stored = _stored_facets(where["facets"]) if int(version) >= 3 else ()
        made = CollectionRule(
            version=int(version),
            folder_uuid=_stored_uuid(where["folder"], "folder"),
            person_uuid=_stored_uuid(where["person"], "person"),
            artifact_uuid=artifact_stored,
            kind=where["kind"],
            favorite=where["favorite"],
            rating_min=where["rating_min"],
            text=select["text"],
            sort=select["sort"],
            take=select["take"],
            actor_id=actor_id,
            facets=facets_stored,
        )
    except (KeyError, TypeError, ValueError) as rotten:
        raise BrokenCollectionRule(f"collection {collection_id}'s stored rule cannot be read: {rotten}") from rotten
    stored_v = held.get("v")
    if type(stored_v) is not int or stored_v != version:
        # Exact JSON integer: true == 1 and 1.0 == 1 in Python, and a
        # version stamp that "equals" its column by coercion is corrupt.
        raise BrokenCollectionRule(
            f"collection {collection_id}'s stored form says v{stored_v!r} under a v{version!r} column"
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
