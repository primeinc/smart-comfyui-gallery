"""What runs without being asked.

A schedule points at a COLLECTION and never at a kind. That is the whole
reason collections exist before this could: naming kinds would mean
re-deriving the order at 3am -- which of scan, ingest, embed,
detect_faces, cluster_faces, and in which sequence -- and that order is
what `job.after_id` exists so nobody has to carry.

It is also why this is worth having only now. A catch-up that could not
walk derived forever over a library it never noticed growing, so a
nightly one would have been busy and blind. With the walk at its head
and every file-reading step counting its units on claim, "every night,
catch up" means what it says.

Two refusals, and both matter more than the scheduling:

**Never two at once.** A collection with an unsettled job in it is
already running, and queueing a second is how a nightly job on a slow
library becomes seven overlapping ones by Sunday. The guard is a query,
not a lock -- there is nothing to hold across a night.

**Measured from the START.** A three-hour catch-up on a nightly
schedule runs once a night; measured from the finish it would slip three
hours later every day and be running at noon by the end of the week.
"""

from __future__ import annotations

#: The collections a schedule may name. Not free text: a typo would be a
#: schedule that never runs anything and looks exactly like one that
#: does, and the runner is the only thing that knows what a collection
#: is made of.
RUNNABLE = ("catch up",)


def all_of(conn) -> list[dict]:
    """Every schedule, whether enabled or not."""
    cursor = conn.execute(
        "SELECT id, collection, every_hours, enabled, last_started_at, created_at FROM schedule ORDER BY collection"
    )
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor]


def put(conn, collection: str, every_hours: float, now: float, *, enabled: bool = True) -> None:
    """Set the schedule for one collection, or change it.

    Refused rather than stored for a collection nothing can run: a
    schedule that names something the runner has never heard of is a row
    that looks like it works and never does.
    """
    if collection not in RUNNABLE:
        raise ValueError(f"nothing runs a collection called {collection!r}; try one of {', '.join(RUNNABLE)}")
    if not every_hours > 0:
        raise ValueError(f"a schedule repeats every some-hours, not every {every_hours}")
    conn.execute(
        "INSERT INTO schedule(collection, every_hours, enabled, created_at) VALUES(?, ?, ?, ?)"
        " ON CONFLICT(collection) DO UPDATE SET every_hours = excluded.every_hours, enabled = excluded.enabled",
        (collection, every_hours, int(enabled), now),
    )


def forget(conn, collection: str) -> int:
    """Remove a schedule; returns how many there were."""
    return int(conn.execute("DELETE FROM schedule WHERE collection = ?", (collection,)).rowcount or 0)


def next_due(row: dict) -> float | None:
    """When this schedule is next due, or None if it never runs.

    None for a disabled one, and 0 for one that has never run: a
    schedule somebody just turned on should not wait a full interval to
    prove it works.
    """
    if not row["enabled"]:
        return None
    if row["last_started_at"] is None:
        return 0.0
    return float(row["last_started_at"]) + float(row["every_hours"]) * 3600.0


def running(conn, collection: str) -> bool:
    """Is this collection already going?

    An unsettled job in it is one that has not finished, including a
    step still queued behind another. That is what makes the guard
    correct for a chain: the collection is not done until its last step
    is, and "the running one" is not always the one anybody can see
    working.
    """
    return (
        conn.execute(
            "SELECT 1 FROM job WHERE collection = ? AND state IN ('queued','running') LIMIT 1",
            (collection,),
        ).fetchone()
        is not None
    )


def due(conn, now: float) -> list[dict]:
    """The schedules that should start something, right now.

    Not the ones merely past their time: a collection already running is
    not due, however long ago it last started. A nightly job on a library
    that takes thirty hours to catch up runs continuously and never
    stacks, which is the honest behaviour -- the alternative is seven
    overlapping catch-ups by Sunday, each slowing the others.
    """
    told = []
    for row in all_of(conn):
        when = next_due(row)
        if when is None or when > now:
            continue
        if running(conn, row["collection"]):
            continue
        told.append(row)
    return told


def started(conn, collection: str, now: float) -> None:
    """Stamp a schedule as having started its collection."""
    conn.execute("UPDATE schedule SET last_started_at = ? WHERE collection = ?", (now, collection))
