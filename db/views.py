"""A question worth asking again.

People mean three different things by "save this", and only two of them
had somewhere to go.

An **album** is what somebody deliberately put together. A **smart
collection** is a dynamic grouping that behaves like one: it has
members, an address, a place on the shelf, things filed under it. A
**saved view** is neither -- it is "that was a useful question, remember
it", and it has no members, no colour, no parent and nothing filed under
it. Making one a collection put five things that are not albums into
somebody's album list, one per good question they had.

They share a GalleryQuery underneath without being one product object,
which is the whole design: this stores the canonical SPELLING
(db/resultset.py `canonical`) rather than a typed rule, because the
spelling is entity-aware and heals a retired slug to the live one as it
is navigated. A view saved before a rename still answers afterwards.
"""

from __future__ import annotations


def all_of(conn) -> list[dict]:
    """Every remembered question, most recently used first.

    Used, not created. A list of questions is only useful if the one
    somebody keeps coming back to is near the top, and the order they
    happened to be invented in says nothing about that. Never-opened
    ones sort by when they were made, which is the best guess available
    for a question nobody has returned to yet.
    """
    cursor = conn.execute(
        "SELECT id, name, qs, created_at, last_used_at FROM saved_view"
        " ORDER BY COALESCE(last_used_at, created_at) DESC, name"
    )
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor]


def remember(conn, name: str, qs: str, now: float) -> int:
    """Remember this question under this name, or replace what that name held.

    Replacing rather than refusing: somebody typing a name they have
    used before is refining the question, not colliding with themselves,
    and a refusal at that moment costs them the question they had just
    composed.
    """
    said = name.strip()
    if not said:
        raise ValueError("a remembered question needs a name")
    # Without a page. A remembered question opens at its beginning, never
    # at page 7 of an answer that has since changed length.
    spelling = "&".join(one for one in qs.lstrip("?").split("&") if one and not one.startswith("page="))
    row = conn.execute(
        "INSERT INTO saved_view(name, qs, created_at) VALUES(?, ?, ?)"
        " ON CONFLICT(name) DO UPDATE SET qs = excluded.qs RETURNING id",
        (said, spelling, now),
    ).fetchone()
    return int(row[0])


def opened(conn, view_id: int, now: float) -> None:
    """Somebody went back to this one."""
    conn.execute("UPDATE saved_view SET last_used_at = ? WHERE id = ?", (now, view_id))


def forget(conn, view_id: int) -> int:
    """Stop remembering it; returns how many there were."""
    return int(conn.execute("DELETE FROM saved_view WHERE id = ?", (view_id,)).rowcount or 0)
