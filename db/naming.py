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

import hashlib
import math
import re

from .scan import mint, slugify

#: Exactly 32 hex characters by FULLMATCH: bytes.fromhex skips ASCII whitespace,
#: so a length check after decoding lets spaces hide inside a 32-character
#: spelling and decode to 15 bytes.
UUID_HEX = re.compile(r"[0-9a-fA-F]{32}")

#: A sha256's spelling. The bare pattern exists beside the compiled rule
#: because sg_web/app.py embeds it inside a larger asset-name grammar;
#: three modules stated the spelling independently.
SHA256_HEX_PATTERN = "[0-9a-f]{64}"
SHA256_HEX = re.compile(f"^{SHA256_HEX_PATTERN}$")

#: How many hex characters a short identity hash keeps, 64 bits' worth. Two
#: other widths exist on purpose and are named at their own sites
#: (vision/semantic, db/ingest).
SHORT_HASH_HEX = 16

#: How many characters of a content hash a MESSAGE shows -- enough to
#: find the file, short enough to read. Four surfaces cut this by hand.
SHOWN_SHA_HEX = 12


def short_hash(text: str) -> str:
    """The short stable identity of a text: sha256 over its utf-8 bytes,
    cut to SHORT_HASH_HEX characters."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:SHORT_HASH_HEX]


def short_sha(sha: str) -> str:
    """A content hash as a message shows it."""
    return sha[:SHOWN_SHA_HEX]


class SlugTaken(Exception):
    """The requested slug belongs to a different live entity."""


def entity_slug(conn, entity_id: int) -> tuple[str, str] | None:
    row = conn.execute("SELECT kind, slug FROM entity WHERE id = ?", (entity_id,)).fetchone()
    return (row[0], row[1]) if row else None


def by_uuid(conn, uuid_hex: str) -> tuple[str, str] | None:
    """The CURRENT address of a portable identity: (kind, slug), or None
    when nothing holds that uuid any more. Address resolution only -- a
    frozen record keeps its own facts and asks here for a link."""
    raw = bytes.fromhex(uuid_hex) if UUID_HEX.fullmatch(uuid_hex) else None
    if raw is None:
        return None
    row = conn.execute("SELECT kind, slug FROM entity WHERE uuid = ?", (raw,)).fetchone()
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

    # `retired_at` is in the primary key and a batch rename computes `now` once,
    # so two entities of one kind releasing the same slug in one pass would
    # collide. The key needs distinctness, not an exact timestamp.
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
