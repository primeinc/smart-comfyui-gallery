"""Where a file came from, recorded when it is known rather than guessed later.

A generator hands back a job reference at submit time and the output file
appears seconds or minutes later, somewhere a scan will eventually find. The
edge between them is knowable exactly once -- at submit -- and is
unrecoverable afterwards, because the child arrives looking like any other
new file.

So the intent is written first, keyed on whatever the generator calls the
job, and `resolve` closes it when the output is identified. An intent that is
never resolved stays visible as an open row rather than disappearing.
"""

from __future__ import annotations


def intend(
    conn, parent_id: int, kind: str, external_ref: str, now: float, *, job_id=None
) -> int:
    """Record that a derivation was asked for, before its output exists.

    `external_ref` is the generator's own job id and is UNIQUE, so a retry or
    a duplicate submit reuses the intent instead of creating a second one.
    """
    row = conn.execute(
        "SELECT id FROM derivation_intent WHERE external_ref = ?", (external_ref,)
    ).fetchone()
    if row:
        return row[0]
    cursor = conn.execute(
        "INSERT INTO derivation_intent(parent_id, kind, external_ref, job_id, created_at)"
        " VALUES(?, ?, ?, ?, ?)",
        (parent_id, kind, external_ref, job_id, now),
    )
    return int(cursor.lastrowid or 0)


def resolve(conn, external_ref: str, child_id: int, now: float) -> int | None:
    """Attach the output to the intent that asked for it.

    Returns the edge id, or None when nothing asked for this file -- which is
    the ordinary case for anything the user made outside the app.
    """
    row = conn.execute(
        "SELECT id, parent_id, kind FROM derivation_intent WHERE external_ref = ?",
        (external_ref,),
    ).fetchone()
    if row is None:
        return None
    intent_id, parent_id, kind = row
    if parent_id == child_id:
        # A generator that hands back the input as its output would otherwise
        # write a self-edge, and every lineage walk from here is a cycle.
        return None
    conn.execute(
        "INSERT OR IGNORE INTO file_derivation(intent_id, parent_id, child_id, kind,"
        " created_at) VALUES(?, ?, ?, ?, ?)",
        (intent_id, parent_id, child_id, kind, now),
    )
    edge = conn.execute(
        "SELECT id FROM file_derivation WHERE parent_id = ? AND child_id = ? AND kind = ?",
        (parent_id, child_id, kind),
    ).fetchone()
    return edge[0] if edge else None


def link(conn, parent_id: int, child_id: int, kind: str, now: float) -> int | None:
    """An edge with no intent behind it, for a lineage learned after the fact."""
    if parent_id == child_id:
        return None
    conn.execute(
        "INSERT OR IGNORE INTO file_derivation(parent_id, child_id, kind, created_at)"
        " VALUES(?, ?, ?, ?)",
        (parent_id, child_id, kind, now),
    )
    row = conn.execute(
        "SELECT id FROM file_derivation WHERE parent_id = ? AND child_id = ? AND kind = ?",
        (parent_id, child_id, kind),
    ).fetchone()
    return row[0] if row else None


def open_intents(conn) -> list[tuple]:
    """Submitted, never resolved. A queue, not a leak."""
    return conn.execute(
        "SELECT i.id, i.parent_id, i.kind, i.external_ref, i.created_at"
        "  FROM derivation_intent i"
        "  LEFT JOIN file_derivation d ON d.intent_id = i.id"
        " WHERE d.id IS NULL ORDER BY i.created_at"
    ).fetchall()


def relate(conn, file_id: int, related_id: int, kind: str, now: float) -> None:
    """A non-derivation relationship: a RAW pair, a sidecar, a proxy.

    Symmetric by nature, so both directions are written -- a query for "what
    belongs with this file" should not have to know which of the two was
    discovered first.
    """
    if file_id == related_id:
        return
    for left, right in ((file_id, related_id), (related_id, file_id)):
        conn.execute(
            "INSERT OR IGNORE INTO file_relation(file_id, related_id, kind, created_at)"
            " VALUES(?, ?, ?, ?)",
            (left, right, kind, now),
        )
