"""What a collection IS, over its whole life.

An album, a flag or a smart collection is one authored entity with four
separable facts: identity (its entity row and slug history), definition
(name, kind, color, description, parent), membership definition (filed
rows for the listed kinds, a typed rule for `smart` -- meaning owned by
db/collection_rules.py), and lifecycle (active or archived). This module
owns the definition and the lifecycle, and the ONE listed-membership
implementation every adapter files through. `db/authored.py` keeps the
judgements about pictures -- ratings, favourites, comments, people,
feedback -- which have none of this structure.

Every write is DESIRED STATE: "name is X", "parent is Y", "archived is
true", "the rule is exactly this" -- never a toggle or a move-ish
command, so a retried request lands where the first one did. Definition
writes carry optimistic concurrency: each names the `definition_rev` it
edited, the UPDATE claims that revision or `CollectionChanged` refuses,
and a stale editor can never silently overwrite newer authored state.

A refusal this module raises -- ValueError, LookupError,
CollectionChanged -- leaves the caller's transaction EXACTLY as it
found it: every domain check runs before the first mutation, and the
revision claim is the first mutation of every multi-step transition, so
a caller that catches the refusal and commits persists nothing partial.
Only an unexpected SQLite failure still needs the caller's rollback.

Membership never bumps the revision -- filing a
picture does not invalidate an open description editor; membership
coherence already belongs to the ResultSet's data version and result
identity.

Archive is the user-facing end of life, not deletion: hard-deleting a
collection would take its entity and its slug history with it, and a
retired address that can someday resolve to a DIFFERENT entity breaks
the addressability doctrine every page is built on. An archived
collection keeps its members, children, rule and address; it leaves the
indexes and the pickers, and restore reverses exactly.

Kind transitions are explicit operations, never patch fields: changing
a description and changing HOW MEMBERSHIP IS DECIDED are different
classes of act, and the listed<->smart crossings carry preconditions
(zero filed members and a valid rule to become smart, an explicit rule
discard to stop being smart) the schema's own triggers backstop.
"""

from __future__ import annotations

import dataclasses
import re
import typing

from . import collection_rules
from .naming import rename
from .scan import mint


class CollectionChanged(Exception):
    """The definition moved since this editor read it: the named
    `definition_rev` is no longer current. Nothing was written."""


class _Unset:
    def __repr__(self) -> str:  # in refusal messages
        return "UNSET"


#: Absent-from-the-patch, as a real value: `None` means "clear this
#: fact" and cannot double as "leave it alone".
UNSET = _Unset()


@dataclasses.dataclass(frozen=True)
class CollectionPatch:
    """One definition edit, whole: a field left UNSET is unchanged, a
    field set to None is deliberately cleared. `kind` is absent on
    purpose -- see the transition operations."""

    name: object = UNSET
    color: object = UNSET
    description: object = UNSET
    parent_id: object = UNSET
    archived: object = UNSET


_COLOR = re.compile(r"#[0-9a-fA-F]{6}")

#: The kinds whose membership is filed rows.
#: Every kind of collection, per db/schema.sql collection.kind.
CollectionKind = typing.Literal["album", "flag", "smart"]

#: The kinds a person fills by listing files. A `smart` collection is
#: born from a rule instead, through its own seam. The runtime tuple is
#: the type's own members, so the check and the type cannot drift; that
#: a listed kind is a collection kind is proved where create_listed
#: hands one to collection().
ListedKind = typing.Literal["album", "flag"]
LISTED: tuple[ListedKind, ...] = typing.get_args(ListedKind)


def _cleaned_name(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("a collection's name is a non-empty string")
    return value.strip()


def _stored_color(value) -> str | None:
    """One stored encoding: `#rrggbb` lowercase, or NULL. Arbitrary CSS
    is refused now, before something renders it into a style context."""
    if value is None:
        return None
    if not isinstance(value, str) or _COLOR.fullmatch(value) is None:
        raise ValueError(f"a color is written #rrggbb, not {value!r}")
    return value.lower()


def _stored_description(value) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError("a description is text, or null to clear it")
    return value.strip() or None


def _named_rev(value) -> int:
    """Exact-integer, like every gate in this schema: JSON true is not
    revision 1."""
    if type(value) is not int or value < 1:
        raise ValueError(f"a definition write names the revision it edited, a positive integer, not {value!r}")
    return value


def _definition(conn, collection_id: int):
    row = conn.execute("SELECT kind, parent_id, archived_at FROM collection WHERE id = ?", (collection_id,)).fetchone()
    if row is None:
        raise LookupError(f"no collection {collection_id}")
    return row


def _parent_allowed(conn, parent_id: int, *, collection_id: int | None = None, current_parent_id=None) -> None:
    """ONE parent-admissibility doctrine for every adapter -- creation
    (no collection exists yet, so no self/descendant to check) and moves
    alike. An active collection is a legal destination; the archived
    parent a collection ALREADY has may be restated, or a patch naming
    its own current state would be impossible to spell; an archived
    collection is never a NEW destination -- from a move or from a
    creation, because "new children" does not care how the child came to
    exist. The collection_no_cycle trigger stays the raw-write backstop."""
    if collection_id is not None and parent_id == collection_id:
        raise ValueError("a collection cannot be its own parent")
    held = conn.execute("SELECT archived_at FROM collection WHERE id = ?", (parent_id,)).fetchone()
    if held is None:
        raise ValueError("the named parent is not a collection")
    if held[0] is not None and parent_id != current_parent_id:
        raise ValueError("an archived collection does not take new children; restore it first")
    if collection_id is None:
        return
    descended = conn.execute(
        "WITH RECURSIVE down(id) AS ("
        " SELECT id FROM collection WHERE parent_id = ?"
        " UNION SELECT c.id FROM collection c JOIN down d ON c.parent_id = d.id)"
        " SELECT 1 FROM down WHERE id = ?",
        (collection_id, parent_id),
    ).fetchone()
    if descended is not None:
        raise ValueError("a collection cannot move under its own descendant")


def eligible_parents(conn, collection_id: int) -> list[int]:
    """Everything the parent picker may offer: every ACTIVE collection
    that is not this one and not inside it, PLUS the current parent even
    when archived -- the UI must not offer a choice the module will
    refuse, and it must always be able to spell the state that already
    holds. A form that cannot represent "keep the archived parent"
    falls back to its first option and silently reparents on an
    unrelated edit; that is the bug this shape exists to prevent."""
    rows = conn.execute(
        "WITH RECURSIVE down(id) AS ("
        " SELECT id FROM collection WHERE parent_id = ?"
        " UNION SELECT c.id FROM collection c JOIN down d ON c.parent_id = d.id)"
        " SELECT id FROM collection WHERE id <> ?"
        " AND (archived_at IS NULL OR id = (SELECT parent_id FROM collection WHERE id = ?))"
        " AND id NOT IN (SELECT id FROM down)",
        (collection_id, collection_id, collection_id),
    ).fetchall()
    return [row[0] for row in rows]


# --- creation --------------------------------------------------------------


def collection(
    conn,
    name: str,
    now: float,
    *,
    kind: CollectionKind = "album",
    parent_id=None,
    color=None,
    description=None,
    actor_id=None,
) -> int:
    """The row itself: entity minted, definition at revision 1. The
    create_* operations validate on the way in; this is also the
    fixture-level constructor tests reach for, including the legal
    rule-less `smart` (an UNEVALUATED collection, the migrated-legacy
    state)."""
    collection_id = mint(conn, "collection", name)
    conn.execute(
        "INSERT INTO collection(id, parent_id, name, kind, color, description,"
        " created_at, updated_at, created_by, updated_by)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (collection_id, parent_id, name, kind, color, description, now, now, actor_id, actor_id),
    )
    return collection_id


def create_listed(
    conn,
    name: str,
    now: float,
    *,
    kind: ListedKind = "album",
    parent_id=None,
    color=None,
    description=None,
    actor_id=None,
) -> int:
    if kind not in LISTED:
        raise ValueError("kind must be album or flag; a smart collection needs a rule")
    cleaned = _cleaned_name(name)
    if parent_id is not None:
        _parent_allowed(conn, parent_id)
    return collection(
        conn,
        cleaned,
        now,
        kind=kind,
        parent_id=parent_id,
        color=_stored_color(color),
        description=_stored_description(description),
        actor_id=actor_id,
    )


def create_smart(
    conn,
    name: str,
    rule,
    source_text: str | None,
    now: float,
    *,
    parent_id=None,
    color=None,
    description=None,
    actor_id=None,
) -> int:
    """One atomic authored operation: entity, definition and typed rule
    together -- and the rule validates BEFORE the entity exists, so a
    refused rule leaves nothing in the caller's transaction, not even
    uncommitted rows a caller could mistakenly commit."""
    collection_rules.validate(rule, ValueError)
    cleaned = _cleaned_name(name)
    if parent_id is not None:
        _parent_allowed(conn, parent_id)
    collection_id = collection(
        conn,
        cleaned,
        now,
        kind="smart",
        parent_id=parent_id,
        color=_stored_color(color),
        description=_stored_description(description),
        actor_id=actor_id,
    )
    collection_rules.save(conn, collection_id, rule, source_text=source_text, now=now)
    return collection_id


# --- the definition, as one patch ------------------------------------------


def _claim_revision(conn, collection_id: int, expected_rev: int, sets: str, values: tuple, now, actor_id) -> None:
    """One guarded UPDATE: the named revision is claimed and bumped, or
    nothing at all happens."""
    claimed = conn.execute(
        f"UPDATE collection SET {sets} updated_at = ?, updated_by = ?,"
        " definition_rev = definition_rev + 1"
        " WHERE id = ? AND definition_rev = ?",
        (*values, now, actor_id, collection_id, expected_rev),
    )
    if claimed.rowcount == 0:
        _definition(conn, collection_id)  # LookupError when it never existed
        raise CollectionChanged(f"the definition is no longer at revision {expected_rev}; read it again before editing")


def update_definition(conn, collection_id: int, patch: CollectionPatch, actor_id, expected_rev, now: float) -> str:
    """The whole edit under one revision claim. Returns the live slug --
    the authoritative address after a rename, unchanged otherwise."""
    expected_rev = _named_rev(expected_rev)
    _, current_parent_id, _ = _definition(conn, collection_id)
    sets: list[str] = []
    values: list = []
    renamed: str | None = None
    if patch.name is not UNSET:
        renamed = _cleaned_name(patch.name)
        sets.append("name = ?")
        values.append(renamed)
    if patch.color is not UNSET:
        sets.append("color = ?")
        values.append(_stored_color(patch.color))
    if patch.description is not UNSET:
        sets.append("description = ?")
        values.append(_stored_description(patch.description))
    if patch.parent_id is not UNSET:
        if patch.parent_id is not None:
            if type(patch.parent_id) is not int:
                raise ValueError("parent names a collection, or null for the top")
            _parent_allowed(conn, patch.parent_id, collection_id=collection_id, current_parent_id=current_parent_id)
        sets.append("parent_id = ?")
        values.append(patch.parent_id)
    if patch.archived is not UNSET:
        if not isinstance(patch.archived, bool):
            raise ValueError(f"archived is true or false, not {patch.archived!r}")
        # Desired state: archiving what is archived keeps the original
        # archived_at rather than restamping the fact.
        sets.append("archived_at = CASE WHEN ? THEN COALESCE(archived_at, ?) ELSE NULL END")
        values.extend([patch.archived, now])
    if not sets:
        raise ValueError("the patch names no definition fact")
    _claim_revision(conn, collection_id, expected_rev, ", ".join(sets) + ",", tuple(values), now, actor_id)
    if renamed is not None:
        return rename(conn, collection_id, renamed, now)
    row = conn.execute("SELECT slug FROM entity WHERE id = ?", (collection_id,)).fetchone()
    return row[0]


# --- the rule, and the definition-mode crossings ---------------------------


def replace_rule(conn, collection_id: int, rule, source_text, actor_id, expected_rev, now: float) -> None:
    """The whole rule as the collection's new meaning, under the same
    revision claim as any definition edit."""
    expected_rev = _named_rev(expected_rev)
    kind, _, _ = _definition(conn, collection_id)
    if kind != "smart":
        raise ValueError("only a smart collection carries a rule; convert it first")
    collection_rules.validate(rule, ValueError)  # every refusal precedes the first mutation
    _claim_revision(conn, collection_id, expected_rev, "", (), now, actor_id)
    collection_rules.save(conn, collection_id, rule, source_text=source_text, now=now)


def convert_to_smart(conn, collection_id: int, rule, source_text, actor_id, expected_rev, now: float) -> None:
    """listed -> smart, atomically WITH its rule: an empty membership and
    a valid rule in the same operation, or nothing. The
    collection_with_members_stays_listed trigger is the raw-write
    backstop; the refusal here is the one with a caller to hear it."""
    expected_rev = _named_rev(expected_rev)
    kind, _, _ = _definition(conn, collection_id)
    if kind == "smart":
        raise ValueError("this collection is already smart; replace its rule instead")
    filed = conn.execute("SELECT count(*) FROM collection_file WHERE collection_id = ?", (collection_id,)).fetchone()[0]
    if filed:
        raise ValueError(f"this collection holds {filed} filed member(s); empty it before making it smart")
    collection_rules.validate(rule, ValueError)  # every refusal precedes the first mutation
    _claim_revision(conn, collection_id, expected_rev, "kind = 'smart',", (), now, actor_id)
    collection_rules.save(conn, collection_id, rule, source_text=source_text, now=now)


def convert_to_listed(
    conn, collection_id: int, kind: ListedKind, actor_id, expected_rev, now: float, *, discard_rule=False
) -> None:
    """smart -> album/flag only with the rule's discard said out loud --
    the rule is authored state -- and album <-> flag freely: both listed
    kinds mean the same filed rows.

    The revision claim comes FIRST: a stale editor must be refused
    before the authored rule is touched, or catching CollectionChanged
    and committing would persist a deleted rule. The kind change comes
    LAST because the collection_with_rule_stays_smart trigger rightly
    refuses it while the rule still exists."""
    expected_rev = _named_rev(expected_rev)
    if kind not in LISTED:
        raise ValueError("kind must be album or flag")
    current, _, _ = _definition(conn, collection_id)
    if current == "smart" and discard_rule is not True:
        raise ValueError("converting a smart collection discards its authored rule; say discard_rule true to mean it")
    _claim_revision(conn, collection_id, expected_rev, "", (), now, actor_id)
    if current == "smart":
        conn.execute("DELETE FROM collection_rule WHERE collection_id = ?", (collection_id,))
    conn.execute("UPDATE collection SET kind = ? WHERE id = ?", (kind, collection_id))


# --- listed membership: the ONE implementation -----------------------------

_FILE_INTO = "INSERT OR IGNORE INTO collection_file(collection_id, file_id, added_at) VALUES(?, ?, ?)"
_FILE_OUT_OF = "DELETE FROM collection_file WHERE collection_id = ? AND file_id = ?"


def _takes_filings(conn, collection_id: int, *, removing: bool) -> None:
    """A smart collection is refused by name here, and by trigger
    beneath: its members are its rule's result set, and a stored row would
    be a second, disagreeing one -- and pretending to remove one would
    be acting under a membership model the kind does not have."""
    kind = conn.execute("SELECT kind FROM collection WHERE id = ?", (collection_id,)).fetchone()
    if kind is not None and kind[0] == "smart":
        what = "to remove" if removing else "into it"
        raise ValueError(f"a smart collection derives its members from its rule; nothing is filed {what}")


def set_membership_many(conn, collection_id: int, file_ids, value: bool, now: float) -> None:
    """One membership write for every adapter. The smart refusal runs
    ONCE, before any row -- all or nothing is the transaction's job, but
    not even the first row of a doomed batch should be attempted.
    Membership is not definition: `definition_rev` does not move."""
    _takes_filings(conn, collection_id, removing=not value)
    if value:
        conn.executemany(_FILE_INTO, [(collection_id, file_id, now) for file_id in file_ids])
    else:
        conn.executemany(_FILE_OUT_OF, [(collection_id, file_id) for file_id in file_ids])


def set_membership(conn, collection_id: int, file_id: int, value: bool, now: float) -> None:
    set_membership_many(conn, collection_id, (file_id,), value, now)
