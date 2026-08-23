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

from .scan import mint

KINDS = ("country", "region", "island", "county", "city", "locality", "neighborhood", "poi")


def label(conn, place_id: int) -> str | None:
    """The place's name, or None when no such place: what a chip says
    instead of an id."""
    row = conn.execute("SELECT name FROM place WHERE id = ?", (place_id,)).fetchone()
    return str(row[0]) if row else None


def named(conn, name: str, kind: str, now: float) -> int:
    """The place called `name` of this kind, minted on first mention:
    two pictures said to be in Lisbon are in ONE Lisbon, by name
    (case-insensitive) and kind, never two rows."""
    if kind not in KINDS:
        raise ValueError(f"a place kind is one of {', '.join(KINDS)}, not {kind!r}")
    spelled = (name or "").strip()
    if not spelled:
        raise ValueError("a place's name is a non-empty string")
    row = conn.execute("SELECT id FROM place WHERE name = ? COLLATE NOCASE AND kind = ?", (spelled, kind)).fetchone()
    return int(row[0]) if row else place(conn, spelled, kind, now)


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
