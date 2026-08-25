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
import typing
from dataclasses import dataclass

from . import ledger

#: What a job is doing and how it is going, per db/schema.sql job.kind and
#: job.state. Here rather than at a web seam because the table owns them:
#: a value outside either is already impossible in the database, and the
#: browser gets the closed set as a union instead of `string`.
JobState = typing.Literal["queued", "running", "done", "failed", "cancelled"]
JobKind = typing.Literal[
    # the directory walk itself -- 'scan' is the metadata read that
    # follows it (db/runner.py _ingest_item), which is why the walk
    # needed a word of its own rather than borrowing that one
    "walk",
    "scan",
    "hash",
    "embed",
    "detect_faces",
    "cluster_faces",
    "sample_frames",
    "annotate",
    "remix",
    "zip",
    "context",
    "events",
    "story_plan",
    "embed_prompts",
]

#: The kinds whose `job_item.item_id` IS a file id.
#:
#: Beside `JobKind` because it is a fact ABOUT the vocabulary, and it
#: was written down twice -- once in db/runner.py so the ledger could
#: name an item, once in db/inspecting.py so the console could link one
#: -- which is two places to forget when a kind is added.
#:
#: Guessing is unsafe rather than merely imprecise: `cluster_faces` and
#: `events` number their items 0, 1, 2 as indices into a payload, so a
#: lookup that assumed every item id were a file would put a real
#: picture's name beside item 2 of a clustering run.
#:
#: `hash` is here with one exception its readers apply: `derive=groups`
#: runs a single item that is not a file either.
FILE_ITEMS = frozenset({"scan", "hash", "embed", "detect_faces", "context", "annotate"})

#: Where one unit of a job stands, per db/schema.sql job_item.state.
ItemState = typing.Literal["pending", "done", "failed"]

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


def submit(
    conn, kind: str, now: float, *, target_id=None, payload=None, items=None, collection=None, after_id=None
) -> int:
    """Ask for work. Nothing runs until a worker claims it.

    `after_id` gates it: `claim` will not take this job until that one has
    settled `done`. `collection` is the name it shares with its siblings,
    which is what the console groups on and what a schedule can point at.
    Both default to None, which is a job that stands alone -- what every
    job was before steps existed.
    """
    cursor = conn.execute(
        "INSERT INTO job(kind, target_id, state, payload, total, created_at, collection, after_id)"
        " VALUES(?, ?, 'queued', ?, ?, ?, ?, ?)",
        (
            kind,
            target_id,
            json.dumps(payload) if payload is not None else None,
            len(items) if items is not None else None,
            now,
            collection,
            after_id,
        ),
    )
    job_id = int(cursor.lastrowid or 0)
    if items:
        conn.executemany(
            "INSERT INTO job_item(job_id, item_id, state) VALUES(?, ?, 'pending')",
            [(job_id, item) for item in items],
        )
    ledger.record(
        conn,
        job_id,
        "job.submitted",
        now,
        message=f"{kind} queued" + (f" with {len(items)} items" if items is not None else ""),
        data={"kind": kind, "total": len(items) if items is not None else None, "target_id": target_id},
    )
    return job_id


def begin(conn, kind: str, owner: str, now: float, *, payload=None) -> tuple[int, int]:
    """A job that is already running, owned by whoever asked for it.

    For work a REQUEST does itself rather than handing to the worker --
    the directory walk, which has to answer with its own counts. Submit
    and then claim looks equivalent and is not: between the two the row
    sits queued, the worker polls for any runnable kind, and it took the
    walk, found no handler for it and failed it while the request that
    created it was still reaching for it. Inserted running and owned,
    the row is never claimable, because `claim` only takes queued ones.

    Returns `(job_id, fence)` -- the same pair `claim` returns, so the
    caller drives it with `checkpoint` and `settle` exactly as a worker
    would, and the row obeys the same invariants either way.
    """
    lease = now + LEASE_SECONDS
    cursor = conn.execute(
        "INSERT INTO job(kind, state, payload, owner, fence, lease_until, heartbeat_at, attempt,"
        " created_at, started_at) VALUES(?, 'running', ?, ?, 1, ?, ?, 1, ?, ?)",
        (kind, json.dumps(payload) if payload is not None else None, owner, lease, now, now, now),
    )
    job_id = int(cursor.lastrowid or 0)
    ledger.record(
        conn,
        job_id,
        "job.submitted",
        now,
        message=f"{kind} started by {owner}",
        data={"kind": kind, "owner": owner, "fence": 1},
    )
    return job_id, 1


def claim(conn, owner: str, now: float, *, kinds=None, gate=None) -> tuple[int, int] | None:
    """Take the next runnable job, returning `(job_id, fence)`.

    Runnable means queued, or running under a lease that has expired. The
    fence is incremented on every claim, so a previous owner that wakes up
    still holding the old value can no longer write.

    `gate` is `(setting key, default)`: the claim then succeeds only while
    that setting reads 'on', evaluated INSIDE the claim's single UPDATE.
    A caller that checks the switch and then claims holds a stale answer
    for the gap between the two -- observed live once per-request
    connection setup widened it -- and an off-switch committed before the
    claim must never lose to that gap.
    """
    # A step whose predecessor has not finished is not runnable, however
    # queued it looks. `done` specifically, not merely settled: a step
    # after a failed one must not run, which is what makes a failure stop
    # its dependents and nothing else.
    #
    # Inside `runnable` rather than beside it, because the predicate is
    # evaluated twice -- once to pick the row and once, under the write
    # lock, to re-check it. A gate applied in only one of those places is
    # a gate two workers can race through.
    runnable = (
        "(state = 'queued' OR (state = 'running' AND lease_until < ?))"
        " AND (after_id IS NULL OR EXISTS"
        "      (SELECT 1 FROM job before WHERE before.id = job.after_id AND before.state = 'done'))"
    )
    kind_filter = ""
    kind_args: list = []
    if kinds:
        kind_filter = f" AND kind IN ({','.join('?' * len(kinds))})"
        kind_args = list(kinds)
    gate_filter = ""
    gate_args: list = []
    if gate is not None:
        gate_filter = " AND COALESCE((SELECT value FROM setting WHERE key = ?), ?) = 'on'"
        gate_args = list(gate)

    # One statement. As a SELECT then an UPDATE this was not a claim at all:
    # two workers both read the same queued row, both incremented the fence,
    # both read it back as the same number, and `_held` passed for both --
    # verified with two connections on a real file. The fence only excludes
    # anybody if the row is taken and stamped in a single write.
    #
    # The predicate is repeated inside the UPDATE rather than trusting the
    # subquery: whichever writer gets the lock second re-evaluates it against
    # what the first committed, and sees a job that is no longer runnable.
    row = conn.execute(
        "UPDATE job SET state = 'running', owner = ?, fence = fence + 1,"
        " attempt = attempt + 1, lease_until = ?, heartbeat_at = ?,"
        " started_at = COALESCE(started_at, ?)"
        f" WHERE id = (SELECT id FROM job WHERE {runnable}{kind_filter}"
        "             ORDER BY created_at LIMIT 1)"
        f"   AND {runnable}{gate_filter}"
        " RETURNING id, fence",
        [owner, now + LEASE_SECONDS, now, now, now, *kind_args, now, *gate_args],
    ).fetchone()
    if row is None:
        return None
    return int(row[0]), int(row[1])


def _held(conn, job_id: int, fence: int) -> None:
    row = conn.execute("SELECT fence FROM job WHERE id = ?", (job_id,)).fetchone()
    if row is None or row[0] != fence:
        raise LeaseLost(f"job {job_id} was reclaimed")


def _wrote(cursor, job_id: int, fence: int) -> None:
    """A fenced write that changed nothing means the fence has moved.

    Every fenced statement here needs this. `WHERE ... AND fence = ?` on its
    own turns an evicted worker's write into a silent no-op, which reads to
    the worker as success and leaves the job describing work that did not
    happen.
    """
    if not cursor.rowcount:
        raise LeaseLost(f"job {job_id} was reclaimed; fence {fence} no longer holds it")


def heartbeat(conn, job_id: int, fence: int, now: float) -> None:
    """Say the worker is still alive, and extend the lease."""
    _held(conn, job_id, fence)
    _wrote(
        conn.execute(
            "UPDATE job SET heartbeat_at = ?, lease_until = ? WHERE id = ? AND fence = ?",
            (now, now + LEASE_SECONDS, job_id, fence),
        ),
        job_id,
        fence,
    )


def pause(conn, job_id: int, fence: int, now: float) -> None:
    """Stop working a job on purpose, leaving it immediately claimable.

    A worker that stops at a budget is not dead, so making the next turn
    wait out a liveness lease punishes exactly the caller who did the
    polite thing. Expiring the lease on the spot keeps `claim` the single
    way in -- the job stays `running` with its items, and whichever worker
    turns up next (this process or another) takes it over under a new
    fence.
    """
    _held(conn, job_id, fence)
    _wrote(
        conn.execute(
            "UPDATE job SET lease_until = ?, heartbeat_at = ? WHERE id = ? AND fence = ?",
            (now, now, job_id, fence),
        ),
        job_id,
        fence,
    )


def pending(conn, job_id: int) -> list[int]:
    """The units still to do, so a resumed job repeats nothing."""
    return [
        row[0]
        for row in conn.execute(
            "SELECT item_id FROM job_item WHERE job_id = ? AND state = 'pending' ORDER BY item_id",
            (job_id,),
        )
    ]


def finish_item(conn, job_id: int, fence: int, item_id: int, *, error=None) -> Progress:
    """Settle one unit and report where the job now stands.

    Both writes carry the fence. Unfenced, the item write landed for a worker
    that had already lost the job while the fenced `done_count` write did
    not, so the row said "done" and the counter disagreed -- and the worker
    that took the job over then skipped an item nobody had performed.
    """
    _held(conn, job_id, fence)
    settled = conn.execute(
        "UPDATE job_item SET state = ?, error = ? WHERE job_id = ? AND item_id = ?"
        " AND EXISTS (SELECT 1 FROM job WHERE id = job_item.job_id AND fence = ?)",
        ("failed" if error else "done", error, job_id, item_id, fence),
    )
    if not settled.rowcount:
        # Nothing was written. Either the item is not on this job, or the
        # fence moved between the check above and the write.
        _held(conn, job_id, fence)
        raise LookupError(f"job {job_id} has no item {item_id}")
    _wrote(
        conn.execute(
            "UPDATE job SET done_count = (SELECT count(*) FROM job_item"
            " WHERE job_id = ? AND state <> 'pending') WHERE id = ? AND fence = ?",
            (job_id, job_id, fence),
        ),
        job_id,
        fence,
    )
    return progress(conn, job_id)


def checkpoint(conn, job_id: int, fence: int, marker, done: int | None = None, *, at: float) -> None:
    """Where to resume from, for work with no enumerable units. The change
    is an event: a resume that starts from the wrong place is explained by
    the last marker the ledger saw."""
    _held(conn, job_id, fence)
    ledger.record(
        conn,
        job_id,
        "checkpoint.changed",
        at,
        message="checkpoint moved",
        data={"checkpoint": marker, "done": done, "fence": fence},
    )
    if done is None:
        _wrote(
            conn.execute(
                "UPDATE job SET checkpoint = ? WHERE id = ? AND fence = ?",
                (json.dumps(marker), job_id, fence),
            ),
            job_id,
            fence,
        )
    else:
        _wrote(
            conn.execute(
                "UPDATE job SET checkpoint = ?, done_count = ? WHERE id = ? AND fence = ?",
                (json.dumps(marker), done, job_id, fence),
            ),
            job_id,
            fence,
        )


def cancel(conn, job_id: int, now: float | None = None) -> None:
    """Ask a job to stop. It stops itself, at a boundary, and says so. The
    ask is an event only when it changed the row: a second press on a job
    already asked, or on one already settled, records nothing."""
    asked = conn.execute(
        "UPDATE job SET cancel_requested = 1 WHERE id = ? AND state IN ('queued','running') AND cancel_requested = 0",
        (job_id,),
    )
    if asked.rowcount:
        ledger.record(
            conn,
            job_id,
            "job.cancel_requested",
            now if now is not None else 0.0,
            severity="warning",
            message="cancel asked; the runner stops at the next item boundary",
        )


def cancelled(conn, job_id: int) -> bool:
    """What a runner checks between units."""
    row = conn.execute("SELECT cancel_requested FROM job WHERE id = ?", (job_id,)).fetchone()
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
    _wrote(
        conn.execute(
            "UPDATE job SET state = ?, error = ?, finished_at = ?, lease_until = NULL,"
            " owner = NULL WHERE id = ? AND fence = ?",
            (state, error, now, job_id, fence),
        ),
        job_id,
        fence,
    )
    if state != "done":
        stop_dependents(conn, job_id, now)


def enlist(conn, job_id: int, collection: str, after_id: int | None) -> None:
    """Make an already-submitted job a step of `collection`.

    Every submitter in db/runner.py builds its own job -- with its own
    payload, its own item list and its own reasons to queue nothing at
    all -- and threading two more arguments through eleven of them would
    put the recipe's concern inside each of them. This stamps the row
    instead, in the same transaction, before anything can claim it.
    """
    conn.execute("UPDATE job SET collection = ?, after_id = ? WHERE id = ?", (collection, after_id, job_id))


def stop_dependents(conn, job_id: int, now: float) -> list[int]:
    """Cancel every step that was waiting on this one; returns which.

    The product decision this feature turned on, and it is not the
    obvious one. A partial catch-up is normal and useful -- one
    unreadable file must not abandon the other four thousand -- so a
    failed step does NOT fail its collection. It stops exactly the steps
    that depended on it, transitively, and every unrelated step in the
    same collection runs to completion.

    Cancelled rather than left queued. A step whose predecessor failed
    can never become claimable (`claim` gates on `done`), so leaving it
    queued would be a row that waits for ever and reads as pending work
    -- the console would show a collection permanently one step short of
    finished, with nothing to explain why.

    Only steps that have not STARTED. A dependent already running was
    claimed before its predecessor settled, which the gate makes rare but
    not impossible under an expired lease; killing work in flight is the
    runner's business, not a bookkeeping cascade's.
    """
    stopped: list[int] = []
    frontier = [job_id]
    while frontier:
        parent = frontier.pop()
        rows = conn.execute(
            "UPDATE job SET state = 'cancelled', finished_at = ?,"
            "               error = COALESCE(error, 'the step before it did not finish')"
            " WHERE after_id = ? AND state = 'queued' RETURNING id",
            (now, parent),
        ).fetchall()
        for (one,) in rows:
            stopped.append(int(one))
            frontier.append(int(one))
    return stopped


def progress(conn, job_id: int) -> Progress:
    """The snapshot a subscriber is sent before any delta reaches it."""
    row = conn.execute("SELECT done_count, total, state FROM job WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise LookupError(f"no job {job_id}")
    return Progress(done=row[0], total=row[1], state=row[2])


def snapshot(conn, job_id: int) -> dict:
    """Everything a client needs to render the job from cold.

    `failed_count` is here because "done, with three files unreadable" and
    "done" are different outcomes a page must show without a worker's turn
    summary to read -- the turn is not addressable, the row is.
    """
    cursor = conn.execute(
        "SELECT id, kind, state, cancel_requested, total, done_count, attempt,"
        " error, created_at, started_at, finished_at, json_extract(payload, '$.derive') AS derive"
        " FROM job WHERE id = ?",
        (job_id,),
    )
    row = cursor.fetchone()
    if row is None:
        raise LookupError(f"no job {job_id}")
    told = dict(zip([c[0] for c in cursor.description], row, strict=True))
    told["failed_count"] = conn.execute(
        "SELECT count(*) FROM job_item WHERE job_id = ? AND state = 'failed'", (job_id,)
    ).fetchone()[0]
    return told


#: What a live list shows of a job, active or settled: one column list so
#: `active` and `recent` rows are the same shape to a renderer.
_LISTED = (
    "id, kind, state, cancel_requested, total, done_count, created_at, finished_at,"
    " json_extract(payload, '$.derive') AS derive,"
    # A walk has no enumerable items -- finding them is the job -- so
    # WHERE it is looking is the only specific thing it can say, and a
    # console that cannot say it shows the same line for every root.
    " json_extract(payload, '$.path') AS path"
)


def active(conn) -> list[dict]:
    """Every job still owed work, oldest first: what a subscriber is sent
    before any delta, and what a page renders from cold."""
    cursor = conn.execute(f"SELECT {_LISTED} FROM job WHERE state IN ('queued','running') ORDER BY created_at")
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor]


def recent(conn, limit: int = 12) -> list[dict]:
    """The last `limit` settled jobs, newest first -- so a page opened
    after a sweep finished shows the same rows a page that watched it
    does. Walks the primary key backwards and stops at `limit`: no sort
    of the table, however long the history grows."""
    cursor = conn.execute(
        f"SELECT {_LISTED} FROM job WHERE state IN ('done','failed','cancelled') ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor]


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
