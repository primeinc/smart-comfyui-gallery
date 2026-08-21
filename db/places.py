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
