"""Place identity: where media happened, as an entity.

"Hawaii", "HI" and "Hawai'i" as strings are three unrelated spellings;
a place is an entity with an address and a hierarchy, so a query for
the island naturally includes the beach. Rows are minted by explicit
enrichment or authoring -- never by a GET, and never automatically from
raw GPS: coordinates without a resolver stay coordinates on the media
context, and a future reverse-geocoding job (cached by geographic cell,
so one beach is one lookup) assigns real identity here. The schema's
kind-agreement, hierarchy-cycle and name-search triggers make a place
the same full entity citizen a person or a collection is.
"""

from __future__ import annotations

from . import connect
from .scan import mint

KINDS = ("country", "region", "island", "county", "city", "locality", "neighborhood", "poi")


def label(conn, place_id: int) -> str | None:
    """The place's name, or None when no such place: what a chip says
    instead of an id."""
    row = conn.execute("SELECT name FROM place WHERE id = ?", (place_id,)).fetchone()
    return str(row[0]) if row else None


def named(conn, name: str, kind: str, now: float, *, within: int | None = None) -> int:
    """The place called `name` of this kind, minted on first mention:
    two pictures said to be in Lisbon are in ONE Lisbon, by name
    (case-insensitive) and kind, never two rows. `within` is the parent
    place: a bare Lisbon later said to be within Portugal gains that
    parent rather than a twin; a Lisbon already within somewhere else
    is a different Lisbon."""
    if kind not in KINDS:
        raise ValueError(f"a place kind is one of {', '.join(KINDS)}, not {kind!r}")
    spelled = (name or "").strip()
    if not spelled:
        raise ValueError("a place's name is a non-empty string")
    # Find-or-mint under the writer lane: two people naming Lisbon at once
    # must not mint two (the `place_identity` index refuses the second
    # outright; the lane makes the first win cleanly).
    connect.claim_lane(conn)
    rows = conn.execute(
        "SELECT id, parent_id FROM place WHERE name = ? COLLATE NOCASE AND kind = ? ORDER BY id", (spelled, kind)
    ).fetchall()
    if within is None and len(rows) > 1 and all(parent_id is not None for _, parent_id in rows):
        # two real places of this name, each within somewhere: a bare
        # mention cannot pick one by insertion order
        parents = ", ".join(label(conn, parent_id) or "?" for _, parent_id in rows)
        raise ValueError(f"{spelled} is ambiguous: one within each of {parents}; say which")
    for place_id, parent_id in rows:
        if within is None or parent_id == within:
            return int(place_id)
    for place_id, parent_id in rows:
        if parent_id is None:
            # a bare one, and nobody of that name is within `within` yet --
            # unless `within` is this very place or something inside it,
            # which the no-cycle trigger would refuse as a 500
            if within == place_id or place_id in {held["id"] for held in chain(conn, within)}:
                raise ValueError(f"{spelled} cannot be within itself or within a place inside it")
            conn.execute("UPDATE place SET parent_id = ? WHERE id = ?", (within, place_id))
            return int(place_id)
    return place(conn, spelled, kind, now, parent_id=within)


def chain(conn, place_id: int | None) -> list[dict]:
    """The place and every ancestor, leaf first: id, slug, kind, name.
    Empty for None. Cycle-proof, so a corrupt parent never loops."""
    held: list[dict] = []
    seen: set[int] = set()
    cursor = place_id
    while cursor is not None and cursor not in seen:
        seen.add(cursor)
        row = conn.execute(
            "SELECT p.parent_id, p.kind, p.name, e.slug FROM place p JOIN entity e ON e.id = p.id WHERE p.id = ?",
            (cursor,),
        ).fetchone()
        if row is None:
            break
        held.append({"id": cursor, "slug": row[3], "kind": row[1], "name": row[2]})
        cursor = row[0]
    return held


def place(
    conn,
    name: str,
    kind: str,
    now: float,
    *,
    parent_id: int | None = None,
    centroid_lat: float | None = None,
    centroid_lon: float | None = None,
    country_code: str | None = None,
    provider: str | None = None,
    provider_key: str | None = None,
) -> int:
    if kind not in KINDS:
        raise ValueError(f"a place kind is one of {', '.join(KINDS)}, not {kind!r}")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("a place's name is a non-empty string")
    if parent_id is not None and conn.execute("SELECT 1 FROM place WHERE id = ?", (parent_id,)).fetchone() is None:
        # BEFORE the mint: a refusal must leave the caller's transaction
        # exactly as it found it, or a caught failure plus a commit
        # strands an entity with no subtype -- the lesson the collection
        # lifecycle already paid for.
        raise ValueError("the named parent is not a place")
    place_id = mint(conn, "place", name.strip())
    conn.execute(
        "INSERT INTO place(id, parent_id, kind, name, centroid_lat, centroid_lon,"
        " country_code, provider, provider_key, created_at)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            place_id,
            parent_id,
            kind,
            name.strip(),
            centroid_lat,
            centroid_lon,
            country_code,
            provider,
            provider_key,
            now,
        ),
    )
    return place_id
