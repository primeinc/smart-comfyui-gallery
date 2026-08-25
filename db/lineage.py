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


def intend(conn, parent_id: int, kind: str, external_ref: str, now: float, *, job_id=None) -> int:
    """Record that a derivation was asked for, before its output exists.

    `external_ref` is the generator's own job id and is UNIQUE, so a retry or
    a duplicate submit reuses the intent instead of creating a second one.
    """
    row = conn.execute("SELECT id FROM derivation_intent WHERE external_ref = ?", (external_ref,)).fetchone()
    if row:
        return row[0]
    cursor = conn.execute(
        "INSERT INTO derivation_intent(parent_id, kind, external_ref, job_id, created_at) VALUES(?, ?, ?, ?, ?)",
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
        "INSERT OR IGNORE INTO file_derivation(intent_id, parent_id, child_id, kind, created_at) VALUES(?, ?, ?, ?, ?)",
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
        "INSERT OR IGNORE INTO file_derivation(parent_id, child_id, kind, created_at) VALUES(?, ?, ?, ?)",
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


#: Relationships where the two files are interchangeable, so both directions
#: are the same statement. Everything else in `file_relation.kind` reads one
#: way round -- a video HAS a proxy, a photograph HAS a sidecar -- and writing
#: the reverse as well asserted something false: after relating a video to its
#: proxy, "give me the proxy for this file" returned the video.
_SYMMETRIC = frozenset({"raw_pair"})


def relate(conn, file_id: int, related_id: int, kind: str, now: float) -> None:
    """A non-derivation relationship: a RAW pair, a sidecar, a proxy.

    `file_id` is the subject and `related_id` is what it has, except for the
    symmetric kinds where the distinction does not exist. Reading is by
    `related`, which looks at both sides and says which way each row points,
    so a caller still never has to know which of the two was discovered first.
    """
    if file_id == related_id:
        return
    pairs = [(file_id, related_id)]
    if kind in _SYMMETRIC:
        pairs.append((related_id, file_id))
    for left, right in pairs:
        conn.execute(
            "INSERT OR IGNORE INTO file_relation(file_id, related_id, kind, created_at) VALUES(?, ?, ?, ?)",
            (left, right, kind, now),
        )


def related(conn, file_id: int, *, kind: str | None = None) -> list[tuple[int, str, str]]:
    """Everything attached to this file, from either side.

    Returns `(other_id, kind, direction)` where direction is `has` when this
    file is the subject and `belongs_to` when it is the object -- so a video
    reports `(proxy_id, 'proxy', 'has')` and the proxy reports
    `(video_id, 'proxy', 'belongs_to')`.
    """
    sql = (
        "SELECT related_id, kind, 'has' FROM file_relation WHERE file_id = ?"
        " UNION ALL"
        " SELECT file_id, kind, 'belongs_to' FROM file_relation WHERE related_id = ?"
    )
    args: list = [file_id, file_id]
    if kind:
        sql = (
            "SELECT related_id, kind, 'has' FROM file_relation"
            "  WHERE file_id = ? AND kind = ?"
            " UNION ALL"
            " SELECT file_id, kind, 'belongs_to' FROM file_relation"
            "  WHERE related_id = ? AND kind = ?"
        )
        args = [file_id, kind, file_id, kind]
    return conn.execute(sql, args).fetchall()
