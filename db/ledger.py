"""The operational ledger: what happened to a job, in order, forever.

Three things hold the same subject and must never be confused. The job
row (db/jobs.py) is CURRENT TRUTH -- what a page renders from cold. This
ledger is HISTORICAL OBSERVATION -- one append-only row per operationally
meaningful transition, typed, with a monotonic id. The channel
(sg_web/app.py) is TRANSPORT -- it carries rows after they commit and
stores nothing. A websocket is not a database and a browser is not an
archive: a client that missed events resumes from the last id it holds.

Nothing here samples, compacts or ages out. A 22,000-file sweep leaves
44,000 rows; a reader pages them (`since`, `for_job`). Rendering may
virtualize; storage may paginate; neither may drop an event.

The vocabulary is spelled once, here, as a `Literal` beside the table
that stores it: the wire restates it by importing it, never by copying
it, and sglint SG709 holds it equal to the CHECK on `job_event.type` in
db/schema.sql. So an event the schema refuses cannot be typed, and a
test holds the other end -- an event the schema allows cannot lack a
renderer (sg_web/console.py).
"""

from __future__ import annotations

import json
import re
import typing

#: What a ledger event can be. Equal to the CHECK on job_event.type.
EventType = typing.Literal[
    "job.submitted",
    "job.claimed",
    "job.reclaimed",
    "job.paused",
    "job.cancel_requested",
    "job.cancelled",
    "job.done",
    "job.failed",
    "item.started",
    "item.done",
    "item.failed",
    "item.observed",
    "phase.started",
    "phase.progress",
    "phase.finished",
    "checkpoint.changed",
    "worker.turn_failed",
]

#: How loudly. Equal to the CHECK on job_event.severity.
Severity = typing.Literal["info", "warning", "error"]

#: The same two vocabularies as values, for the runtime refusals below.
#: Derived, never restated: a member can only be added in one place.
TYPES: tuple[EventType, ...] = typing.get_args(EventType)
SEVERITIES: tuple[Severity, ...] = typing.get_args(Severity)

#: Most rows one read returns. A caller wanting more pages.
PAGE_MOST = 2_000
#: The page a reader gets when it does not ask -- the ceiling is named
#: above, and the default was unnamed at three signatures.
PAGE_DEFAULT = 500

_COLUMNS = "id, job_id, at, type, item_id, phase, severity, message, data"

#: A payload or data key that names a credential. Matched on the key, not
#: the value: a value is never inspected, so a secret cannot leak by being
#: shaped like something harmless.
_SECRET_KEY = re.compile(
    r"(token|secret|passw(or)?d|credential|cookie|api[_-]?key|apikey|authorization|bearer)", re.IGNORECASE
)
REDACTED = "•••"


def redacted(value):
    """`value` with every secret-named key's value replaced, at any depth.
    Lists and scalars pass through; only mappings carry names to judge."""
    if isinstance(value, dict):
        return {k: (REDACTED if _SECRET_KEY.search(str(k)) else redacted(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redacted(v) for v in value]
    return value


def _row(row) -> dict:
    told = dict(zip(_COLUMNS.split(", "), row, strict=True))
    told["data"] = json.loads(told["data"]) if told["data"] else None
    return told


def record(
    conn,
    job_id: int,
    type_: str,
    at: float,
    *,
    item_id: int | None = None,
    phase: str | None = None,
    severity: str = "info",
    message: str | None = None,
    data: dict | None = None,
) -> dict:
    """Append one event and return it as the row it became, id included.
    Does not commit: the event rides whatever transaction the transition
    it describes rides, so the ledger can never say what the rows do not."""
    if type_ not in TYPES:
        raise ValueError(f"{type_!r} is not an event type; one of {', '.join(TYPES)}")
    if severity not in SEVERITIES:
        raise ValueError(f"{severity!r} is not a severity; one of {', '.join(SEVERITIES)}")
    cursor = conn.execute(
        "INSERT INTO job_event(job_id, at, type, item_id, phase, severity, message, data)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
        (job_id, at, type_, item_id, phase, severity, message, json.dumps(data) if data is not None else None),
    )
    return {
        "id": int(cursor.lastrowid or 0),
        "job_id": job_id,
        "at": at,
        "type": type_,
        "item_id": item_id,
        "phase": phase,
        "severity": severity,
        "message": message,
        "data": data,
    }


def last_id(conn) -> int:
    """The newest event's id, 0 on an empty ledger -- what a subscriber
    names to resume from."""
    row = conn.execute("SELECT max(id) FROM job_event").fetchone()
    return int(row[0] or 0)


def since(conn, after: int, *, limit: int = PAGE_MOST) -> list[dict]:
    """Every event with id > `after`, ascending, at most `limit`: the
    backlog a reconnecting client asks for. One ordered index walk."""
    rows = conn.execute(
        "SELECT " + _COLUMNS + " FROM job_event WHERE id > ? ORDER BY id LIMIT ?",
        (after, min(limit, PAGE_MOST)),
    )
    return [_row(row) for row in rows]


def latest(conn, *, limit: int = PAGE_DEFAULT) -> list[dict]:
    """The newest `limit` events, ascending -- the cold tape."""
    rows = conn.execute(
        "SELECT " + _COLUMNS + " FROM job_event ORDER BY id DESC LIMIT ?", (min(limit, PAGE_MOST),)
    ).fetchall()
    return [_row(row) for row in reversed(rows)]


def for_job(conn, job_id: int, *, after: int = 0, limit: int = PAGE_MOST) -> list[dict]:
    """One job's events with id > `after`, ascending; rides job_event_job."""
    rows = conn.execute(
        "SELECT " + _COLUMNS + " FROM job_event WHERE job_id = ? AND id > ? ORDER BY id LIMIT ?",
        (job_id, after, min(limit, PAGE_MOST)),
    )
    return [_row(row) for row in rows]


def before(conn, before_id: int, *, job_id: int | None = None, limit: int = PAGE_MOST) -> list[dict]:
    """The `limit` events with id < `before_id`, ascending -- the page
    above a held one, the whole ledger's or one job's."""
    most = min(limit, PAGE_MOST)
    if job_id is None:
        rows = conn.execute(
            "SELECT " + _COLUMNS + " FROM job_event WHERE id < ? ORDER BY id DESC LIMIT ?", (before_id, most)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT " + _COLUMNS + " FROM job_event WHERE job_id = ? AND id < ? ORDER BY id DESC LIMIT ?",
            (job_id, before_id, most),
        ).fetchall()
    return [_row(row) for row in reversed(rows)]


def latest_for_job(conn, job_id: int) -> dict | None:
    row = conn.execute(
        "SELECT " + _COLUMNS + " FROM job_event WHERE job_id = ? ORDER BY id DESC LIMIT 1", (job_id,)
    ).fetchone()
    return _row(row) if row else None


def defects_for_item(conn, job_id: int, item_id: int) -> list[dict]:
    """The turns of this job that have already died on this one item,
    oldest first.

    A defect expires the lease and the job is reclaimed, which is right
    for a transient fault and a livelock for a deterministic one: the
    same item is picked up, crashes the same way, and the job never
    advances. The runner counts these to tell the two apart, and quotes
    the last one so the item's failure names the defect rather than the
    counting."""
    return [
        _row(row)
        for row in conn.execute(
            "SELECT " + _COLUMNS + " FROM job_event"
            " WHERE job_id = ? AND item_id = ? AND type = 'worker.turn_failed' ORDER BY id",
            (job_id, item_id),
        )
    ]


def count_for_job(conn, job_id: int) -> int:
    return int(conn.execute("SELECT count(*) FROM job_event WHERE job_id = ?", (job_id,)).fetchone()[0])


def count(conn) -> int:
    return int(conn.execute("SELECT count(*) FROM job_event").fetchone()[0])


def one(conn, event_id: int) -> dict | None:
    row = conn.execute("SELECT " + _COLUMNS + " FROM job_event WHERE id = ?", (event_id,)).fetchone()
    return _row(row) if row else None
