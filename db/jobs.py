"""Long work: what was asked for, what has been done, and who is doing it.

Nothing expensive starts by itself. Every sweep is a row here, created by an
explicit request, and the row is the truth about it -- not a message that was
broadcast, not a counter in a worker's memory. A page reload, a dropped
socket and a killed process all recover by reading the job back.

That is also what makes live progress cheap rather than fragile: a subscriber
is sent the current row first and deltas afterwards, so arriving late is
indistinguishable from having been there. A channel is transport, never
storage, and never authorization.

Three semantics the columns imply and code has to honour.

**Cancellation is cooperative.** Asking sets `cancel_requested`; only the
runner moves a job to `cancelled`, at an item boundary, once it has stopped.
A setter that flipped the state directly would mark work finished that is
still running.

**Resumption is per item.** `job_item` holds one row per unit, so a resumed
job skips what is already done instead of repeating it. Work with no
enumerable units uses `checkpoint` instead.

**A lease can be lost.** `lease_until` lets a killed process's work be
reclaimed instead of stranding it as `running` forever, and `fence` is what
makes that safe: reclaiming increments it, and every write by a worker is
conditional on still holding the fence it was given. The evicted worker's
writes then fail rather than corrupting the job it no longer owns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

#: A job whose lease has expired by this much is reclaimable. Generous: a
#: worker paused by a slow disk must not lose its job to a false positive.
LEASE_SECONDS = 60.0


class LeaseLost(Exception):
    """This worker no longer owns the job it was writing to."""


@dataclass
class Progress:
    done: int
    total: int | None
    state: str

    @property
    def fraction(self) -> float | None:
        if not self.total:
            return None
        return min(1.0, self.done / self.total)


def submit(conn, kind: str, now: float, *, target_id=None, payload=None, items=None) -> int:
    """Ask for work. Nothing runs until a worker claims it."""
    cursor = conn.execute(
        "INSERT INTO job(kind, target_id, state, payload, total, created_at)"
        " VALUES(?, ?, 'queued', ?, ?, ?)",
        (
            kind, target_id,
            json.dumps(payload) if payload is not None else None,
            len(items) if items is not None else None,
            now,
        ),
    )
    job_id = int(cursor.lastrowid or 0)
    if items:
        conn.executemany(
            "INSERT INTO job_item(job_id, item_id, state) VALUES(?, ?, 'pending')",
            [(job_id, item) for item in items],
        )
    return job_id


def claim(conn, owner: str, now: float, *, kinds=None) -> tuple[int, int] | None:
    """Take the next runnable job, returning `(job_id, fence)`.

    Runnable means queued, or running under a lease that has expired. The
    fence is incremented on every claim, so a previous owner that wakes up
    still holding the old value can no longer write.
    """
    where = "(j.state = 'queued' OR (j.state = 'running' AND j.lease_until < ?))"
    args: list = [now]
    if kinds:
        where += " AND j.kind IN (%s)" % ",".join("?" * len(kinds))
        args.extend(kinds)
    row = conn.execute(
        f"SELECT j.id FROM job j WHERE {where} ORDER BY j.created_at LIMIT 1", args
    ).fetchone()
    if row is None:
        return None
    job_id = row[0]
    conn.execute(
        "UPDATE job SET state = 'running', owner = ?, fence = fence + 1,"
        " attempt = attempt + 1, lease_until = ?, heartbeat_at = ?,"
        " started_at = COALESCE(started_at, ?) WHERE id = ?",
        (owner, now + LEASE_SECONDS, now, now, job_id),
    )
    fence = conn.execute("SELECT fence FROM job WHERE id = ?", (job_id,)).fetchone()[0]
    return job_id, fence


def _held(conn, job_id: int, fence: int) -> None:
    row = conn.execute("SELECT fence FROM job WHERE id = ?", (job_id,)).fetchone()
    if row is None or row[0] != fence:
        raise LeaseLost(f"job {job_id} was reclaimed")


def heartbeat(conn, job_id: int, fence: int, now: float) -> None:
    """Say the worker is still alive, and extend the lease."""
    _held(conn, job_id, fence)
    conn.execute(
        "UPDATE job SET heartbeat_at = ?, lease_until = ? WHERE id = ? AND fence = ?",
        (now, now + LEASE_SECONDS, job_id, fence),
    )


def pending(conn, job_id: int) -> list[int]:
    """The units still to do, so a resumed job repeats nothing."""
    return [
        row[0]
        for row in conn.execute(
            "SELECT item_id FROM job_item WHERE job_id = ? AND state = 'pending'"
            " ORDER BY item_id",
            (job_id,),
        )
    ]


def finish_item(conn, job_id: int, fence: int, item_id: int, *, error=None) -> Progress:
    """Settle one unit and report where the job now stands."""
    _held(conn, job_id, fence)
    conn.execute(
        "UPDATE job_item SET state = ?, error = ? WHERE job_id = ? AND item_id = ?",
        ("failed" if error else "done", error, job_id, item_id),
    )
    conn.execute(
        "UPDATE job SET done_count = (SELECT count(*) FROM job_item"
        " WHERE job_id = ? AND state <> 'pending') WHERE id = ? AND fence = ?",
        (job_id, job_id, fence),
    )
    return progress(conn, job_id)


def checkpoint(conn, job_id: int, fence: int, marker, done: int | None = None) -> None:
    """Where to resume from, for work with no enumerable units."""
    _held(conn, job_id, fence)
    if done is None:
        conn.execute(
            "UPDATE job SET checkpoint = ? WHERE id = ? AND fence = ?",
            (json.dumps(marker), job_id, fence),
        )
    else:
        conn.execute(
            "UPDATE job SET checkpoint = ?, done_count = ? WHERE id = ? AND fence = ?",
            (json.dumps(marker), done, job_id, fence),
        )


def cancel(conn, job_id: int) -> None:
    """Ask a job to stop. It stops itself, at a boundary, and says so."""
    conn.execute(
        "UPDATE job SET cancel_requested = 1 WHERE id = ? AND state IN ('queued','running')",
        (job_id,),
    )


def cancelled(conn, job_id: int) -> bool:
    """What a runner checks between units."""
    row = conn.execute(
        "SELECT cancel_requested FROM job WHERE id = ?", (job_id,)
    ).fetchone()
    return bool(row and row[0])


def settle(conn, job_id: int, fence: int, state: str, now: float, *, error=None) -> None:
    """Reach a terminal state.

    `done` is refused while any unit is still pending: a job that reports
    success with work outstanding is worse than one that reports failure,
    because nothing will ever come back for the remainder.
    """
    _held(conn, job_id, fence)
    if state not in ("done", "failed", "cancelled"):
        raise ValueError(f"{state!r} is not terminal")
    if state == "done":
        outstanding = conn.execute(
            "SELECT count(*) FROM job_item WHERE job_id = ? AND state = 'pending'",
            (job_id,),
        ).fetchone()[0]
        if outstanding:
            raise ValueError(f"job {job_id} has {outstanding} unfinished items")
    conn.execute(
        "UPDATE job SET state = ?, error = ?, finished_at = ?, lease_until = NULL,"
        " owner = NULL WHERE id = ? AND fence = ?",
        (state, error, now, job_id, fence),
    )


def progress(conn, job_id: int) -> Progress:
    """The snapshot a subscriber is sent before any delta reaches it."""
    row = conn.execute(
        "SELECT done_count, total, state FROM job WHERE id = ?", (job_id,)
    ).fetchone()
    if row is None:
        raise LookupError(f"no job {job_id}")
    return Progress(done=row[0], total=row[1], state=row[2])


def snapshot(conn, job_id: int) -> dict:
    """Everything a client needs to render the job from cold."""
    cursor = conn.execute(
        "SELECT id, kind, state, cancel_requested, total, done_count, attempt,"
        " error, created_at, started_at, finished_at FROM job WHERE id = ?",
        (job_id,),
    )
    row = cursor.fetchone()
    if row is None:
        raise LookupError(f"no job {job_id}")
    return dict(zip([c[0] for c in cursor.description], row))


def active(conn) -> list[dict]:
    cursor = conn.execute(
        "SELECT id, kind, state, total, done_count, created_at FROM job"
        " WHERE state IN ('queued','running') ORDER BY created_at"
    )
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row)) for row in cursor]


def watch_folder(conn, folder_id: int, now: float, *, recursive: bool = True) -> None:
    """Mark a folder as one to re-examine when work is asked for.

    A watch is a recorded intention, not a running thread: it says which
    folders a scan job should cover, so "check for new pictures" is a job
    somebody starts rather than a poller grinding through the library.
    """
    conn.execute(
        "INSERT INTO watched_folder(folder_id, recursive, added_at) VALUES(?, ?, ?)"
        " ON CONFLICT(folder_id) DO UPDATE SET recursive = excluded.recursive",
        (folder_id, 1 if recursive else 0, now),
    )


def unwatch_folder(conn, folder_id: int) -> None:
    conn.execute("DELETE FROM watched_folder WHERE folder_id = ?", (folder_id,))


def watched(conn) -> list[tuple]:
    return conn.execute(
        "SELECT w.folder_id, w.recursive, f.name FROM watched_folder w"
        " JOIN folder f ON f.id = w.folder_id ORDER BY f.name"
    ).fetchall()
