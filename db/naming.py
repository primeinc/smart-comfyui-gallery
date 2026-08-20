"""Minting, renaming and resolving the addresses entities are reached by.

A slug is seeded from a name once and then belongs to the entity. Renaming
mints a new one and retires the old into `slug_history`, so an address
written down last year still resolves rather than 404ing.

Resolution order is fixed and not negotiable: a live `entity.slug` always
wins, and history answers only on a miss, most recent retirement first. The
other order would let a retired slug shadow a live entity that has since
taken that name.
"""

from __future__ import annotations

import math

from .scan import mint, slugify


class SlugTaken(Exception):
    """The requested slug belongs to a different live entity."""


def entity_slug(conn, entity_id: int) -> tuple[str, str] | None:
    row = conn.execute("SELECT kind, slug FROM entity WHERE id = ?", (entity_id,)).fetchone()
    return (row[0], row[1]) if row else None


def resolve(conn, kind: str, slug: str) -> tuple[int, bool] | None:
    """`(entity_id, is_current)` for an address, or None.

    `is_current` False means the caller should redirect to the live slug
    rather than serve the page, so an old link keeps working without two
    addresses answering for one thing.
    """
    row = conn.execute("SELECT id FROM entity WHERE kind = ? AND slug = ?", (kind, slug)).fetchone()
    if row:
        return row[0], True
    row = conn.execute(
        "SELECT entity_id FROM slug_history WHERE kind = ? AND slug = ? ORDER BY retired_at DESC LIMIT 1",
        (kind, slug),
    ).fetchone()
    return (row[0], False) if row else None


def rename(conn, entity_id: int, new_name: str, now: float) -> str:
    """Give an entity a new address, keeping the old one working.

    Returns the new slug. A rename that produces the same slug is a no-op
    rather than an entry in history: recording it would put a slug in
    `slug_history` that is also live, and resolution would then depend on
    which table was consulted first.
    """
    current = entity_slug(conn, entity_id)
    if current is None:
        raise LookupError(f"no entity {entity_id}")
    kind, old = current

    base = slugify(new_name) or f"{kind}-{entity_id:x}"
    if base == old:
        return old

    slug, suffix = base, 1
    while True:
        clash = conn.execute("SELECT id FROM entity WHERE kind = ? AND slug = ?", (kind, slug)).fetchone()
        if clash is None or clash[0] == entity_id:
            break
        suffix += 1
        slug = f"{base}-{suffix}"

    # `retired_at` is in the primary key, and `now` is whatever the caller
    # passed -- a batch rename computes it once, so two entities of one kind
    # releasing the same slug in one pass collided and the second raised. The
    # key needs to be distinct, not the timestamp to be exact: nudge forward
    # to the first instant this (kind, slug) is free.
    while conn.execute(
        "SELECT 1 FROM slug_history WHERE kind = ? AND slug = ? AND retired_at = ?",
        (kind, old, now),
    ).fetchone():
        now = math.nextafter(now, math.inf)
    conn.execute(
        "INSERT INTO slug_history(kind, slug, entity_id, retired_at) VALUES(?, ?, ?, ?)",
        (kind, old, entity_id, now),
    )
    conn.execute("UPDATE entity SET slug = ? WHERE id = ?", (slug, entity_id))
    return slug


def claim(conn, kind: str, seed: str) -> int:
    """Mint a new entity, for callers that do not go through the scanner."""
    return mint(conn, kind, seed)
