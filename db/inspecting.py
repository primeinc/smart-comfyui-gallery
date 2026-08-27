"""The operations read model: everything the runtime knows about a job,
for the console that diagnoses it.

Three depths of the same rows, on purpose. `jobs.active` is the shell's
list -- tiny, on every page. `jobs.snapshot` is the ordinary client
projection. This module is the expert inspection: the whole job row,
its items, its ledger and the numbers derived from them, because an
operator asking "why is this job sitting under an expiring lease" needs
`fence`, `attempt`, `lease_until` and the traceback, not a progress bar.

The rule this module is held to: a fact the backend knows and this view
does not expose is a fact the implementer has to justify. Every column
of `job` is here (payload REDACTED, db/ledger.py redacted). The one
deliberate omission is the full item list inlined on the detail -- a
22,000-item job is paged through `items`, never folded into one answer.

Derived numbers say what they are derived FROM. `rate` is items settled
per second since `started_at`; `eta` is pending items at that rate --
both None when the job has not run long enough to say, never 0 dressed
as an answer. A phase inside a running item is known only to the live
feed (db/runner.py Report): `current.phase` here is what the LEDGER
holds, which is the last phase the last SETTLED item reached.
"""

from __future__ import annotations

import json

from . import ingest, jobs, ledger, settings


def _json(text):
    if text is None:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return {"unparsed": text}


def _named_item(conn, kind: str, payload: dict | None, item_id: int | None) -> dict | None:
    """`{id, name, href}` for one item: a file's name and address when the
    kind's items are files and the row still exists; the bare id otherwise."""
    if item_id is None:
        return None
    told: dict = {"id": item_id, "name": None, "href": None}
    if kind in jobs.FILE_ITEMS and not (kind == "hash" and (payload or {}).get("derive") == "groups"):
        row = conn.execute(
            "SELECT f.name, e.slug FROM file f JOIN entity e ON e.id = f.id WHERE f.id = ?", (item_id,)
        ).fetchone()
        if row is not None:
            told["name"], told["href"] = row[0], f"/i/{row[1]}"
    return told


def _target(conn, target_id: int | None) -> dict | None:
    if target_id is None:
        return None
    row = conn.execute("SELECT kind, slug FROM entity WHERE id = ?", (target_id,)).fetchone()
    if row is None:
        return {"id": target_id, "kind": None, "slug": None}
    return {"id": target_id, "kind": row[0], "slug": row[1]}


#: What each missing-only sweep would still queue, counted the way the
#: sweep counts (db/runner.py submit_*): present files with no record for
#: their current bytes. Embed is per configured space at the checkpoint
#: the cache pins, so it needs the models directory; without one it is
#: not counted rather than guessed.
_PRESENT = "SELECT count(*) FROM file f WHERE f.missing_since IS NULL"
_PICTURE = " AND f.kind IN ('image', 'animated_image', 'video')"
_MISSING = {
    # Stale by BYTES or by READER, the same rule the sweep queues on
    # (db/runner.py submit_ingest). Counted differently from what is
    # queued would make the console say "0 missing" beside a sweep that
    # is about to read the whole library.
    "ingest": _PRESENT + " AND (f.ingested_sha256 IS NULL OR f.ingested_sha256 IS NOT f.content_sha256"
    "        OR f.ingested_by IS NOT ?)",
    "faces": _PRESENT + _PICTURE + " AND NOT EXISTS (SELECT 1 FROM derived_face_scan s WHERE s.file_id = f.id"
    "   AND s.source_sha256 = f.content_sha256)",
    "annotate": _PRESENT
    + _PICTURE
    + " AND NOT EXISTS (SELECT 1 FROM derived_annotation a WHERE a.file_id = f.id AND a.kind = 'caption'"
    "   AND a.model_id = ? AND a.source_sha256 = f.content_sha256)",
    "context": _PRESENT
    + " AND NOT EXISTS (SELECT 1 FROM derived_media_context c WHERE c.file_id = f.id AND c.policy_version = ?)",
}
_MISSING_PHASH = (
    _PRESENT + _PICTURE + " AND NOT EXISTS (SELECT 1 FROM derived_file_hash h WHERE h.file_id = f.id AND h.space_id = ?"
    "   AND h.source_sha256 = f.content_sha256)"
)


_MISSING_EMBED = (
    _PRESENT + _PICTURE + " AND NOT EXISTS (SELECT 1 FROM derived_embedding e WHERE e.file_id = f.id AND e.space_id = ?"
    "   AND e.source_sha256 = f.content_sha256)"
)


def _embed_missing(conn, models_dir: str) -> dict[str, int]:
    """Per configured space, what the embed sweep would still queue; a
    space nothing has minted yet has every picture missing. A refused
    `semantic_model` setting is reported as nothing, not guessed at."""
    from vision import semantic

    from . import retrieval

    try:
        choices = retrieval.choices(conn)
    except ValueError:
        return {}
    held: dict[str, int] = {}
    for provider, model, configured in choices:
        checkpoint = semantic.pin(provider, models_dir, model, configured)
        key = semantic.space(provider, model, checkpoint, 1).key
        found = retrieval._space_of(conn, provider, model, checkpoint)
        if found is None:
            held[key] = int(conn.execute(_PRESENT + _PICTURE).fetchone()[0])
        else:
            held[key] = int(conn.execute(_MISSING_EMBED, (found[0],)).fetchone()[0])
    return held


def coverage(conn, models_dir: str | None = None) -> dict:
    """Present files, and how many each missing-only sweep still has to
    do -- the console's answer to "is the library done". `embed` is the
    most any one space still lacks; `embed_spaces` says which."""
    from . import context, similarity

    held = {
        "ingest": conn.execute(_MISSING["ingest"], (ingest.READER,)).fetchone()[0],
        "faces": conn.execute(_MISSING["faces"]).fetchone()[0],
        "context": conn.execute(_MISSING["context"], (context.POLICY_VERSION,)).fetchone()[0],
    }
    caption_model = settings.value(conn, "caption_model")
    if "/" in caption_model:  # else the sweep refuses the setting; a count beside a refusal would be a lie
        held["annotate"] = conn.execute(_MISSING["annotate"], (caption_model,)).fetchone()[0]
    space = similarity._current_space_of(conn, similarity.PHASH)
    held["phash"] = (
        conn.execute(_MISSING_PHASH, (space[0],)).fetchone()[0]
        if space is not None
        else conn.execute(_PRESENT + _PICTURE).fetchone()[0]
    )
    told = {"files": conn.execute(_PRESENT).fetchone()[0], "missing": {k: int(v) for k, v in sorted(held.items())}}
    if models_dir is not None:
        spaces = _embed_missing(conn, models_dir)
        told["embed_spaces"] = spaces
        if spaces:
            told["missing"]["embed"] = max(spaces.values())
    return told


def overview(conn, now: float, *, models_dir: str | None = None) -> dict:
    """The system health strip: worker, queue, ledger head, and what
    each sweep still has to do."""
    running = conn.execute(
        "SELECT count(*), max(heartbeat_at), min(created_at) FROM job WHERE state = 'running'"
    ).fetchone()
    queued = conn.execute("SELECT count(*), min(created_at) FROM job WHERE state = 'queued'").fetchone()
    owners = [
        row[0] for row in conn.execute("SELECT DISTINCT owner FROM job WHERE state = 'running' AND owner IS NOT NULL")
    ]
    last_heartbeat = running[1]
    settled_24h = conn.execute(
        "SELECT state, count(*) FROM job WHERE finished_at >= ? GROUP BY state", (now - 86_400,)
    ).fetchall()
    return {
        "now": now,
        "coverage": coverage(conn, models_dir),
        "worker": {
            "enabled": settings.flag(conn, "worker"),
            "owners": owners,
            "working": bool(running[0]) and last_heartbeat is not None and now - last_heartbeat < jobs.LEASE_SECONDS,
            "last_heartbeat": last_heartbeat,
            "heartbeat_age": (now - last_heartbeat) if last_heartbeat is not None else None,
            "lease_seconds": jobs.LEASE_SECONDS,
        },
        "queue": {
            "queued": int(queued[0]),
            "running": int(running[0]),
            "oldest_queued_age": (now - queued[1]) if queued[1] is not None else None,
            "oldest_running_age": (now - running[2]) if running[2] is not None else None,
            "settled_24h": {state: int(n) for state, n in settled_24h},
        },
        "ledger": {"last_id": ledger.last_id(conn), "events": ledger.count(conn)},
    }


_MATRIX = (
    "SELECT j.id, j.kind, j.state, j.cancel_requested, j.total, j.done_count, j.attempt, j.owner, j.fence,"
    " j.heartbeat_at, j.lease_until, j.created_at, j.started_at, j.finished_at, j.error,"
    " j.collection, j.after_id,"
    " json_extract(j.payload, '$.derive') AS derive,"
    " (SELECT count(*) FROM job_item i WHERE i.job_id = j.id AND i.state = 'failed') AS failed_count"
    " FROM job j"
)


def _lifecycle(row: dict, now: float) -> dict:
    started, finished, created = row["started_at"], row["finished_at"], row["created_at"]
    end = finished if finished is not None else now
    elapsed = (end - started) if started is not None else None
    total = row["total"]
    done = row["done_count"] or 0
    failed = row.get("failed_count") or 0
    pending = (total - done) if total is not None else None
    rate = (done / elapsed) if (elapsed and elapsed > 0 and done) else None
    eta = (pending / rate) if (rate and pending is not None and row["state"] == "running") else None
    if row["state"] == "cancelled":
        cancellation = "cancelled"
    elif row["cancel_requested"] and row["state"] in ("queued", "running"):
        cancellation = "requested"
    else:
        cancellation = "not_requested"
    heartbeat = row["heartbeat_at"]
    lease = row["lease_until"]
    live = row["state"] == "running"
    return {
        "elapsed": elapsed,
        "queue_wait": ((started if started is not None else now) - created),
        "fraction": (min(1.0, done / total) if total else None),
        "pending": pending,
        "succeeded": done - failed,
        "rate": rate,
        "eta": eta,
        "cancellation": cancellation,
        "heartbeat_age": (now - heartbeat) if (live and heartbeat is not None) else None,
        "lease_remaining": (lease - now) if (live and lease is not None) else None,
        "lease_expired": bool(live and lease is not None and lease < now),
    }


def matrix(conn, now: float, *, recent: int = 30) -> list[dict]:
    """Every active job then the settled tail, each with the numbers the
    matrix shows. Two bounded statements, no sort of the table."""
    cursor = conn.execute(_MATRIX + " WHERE j.state IN ('queued','running') ORDER BY j.created_at")
    columns = [c[0] for c in cursor.description]
    rows = [dict(zip(columns, row, strict=True)) for row in cursor]
    cursor = conn.execute(_MATRIX + f" WHERE j.state IN {jobs.TERMINAL_SQL} ORDER BY j.id DESC LIMIT ?", (recent,))
    rows += [dict(zip(columns, row, strict=True)) for row in cursor]
    return [{**row, "derived": _lifecycle(row, now), "settled": row["state"] in jobs.TERMINAL} for row in rows]


def job_detail(conn, job_id: int, now: float, *, recent_events: int = 50) -> dict:
    """One job, whole. LookupError when there is no such job."""
    cursor = conn.execute(
        "SELECT id, kind, target_id, state, cancel_requested, payload, total, done_count, checkpoint, attempt,"
        " owner, fence, lease_until, heartbeat_at, error, created_at, started_at, finished_at FROM job WHERE id = ?",
        (job_id,),
    )
    row = cursor.fetchone()
    if row is None:
        raise LookupError(f"no job {job_id}")
    told = dict(zip([c[0] for c in cursor.description], row, strict=True))
    payload = _json(told.pop("payload"))
    told["payload"] = ledger.redacted(payload)
    told["checkpoint"] = _json(told["checkpoint"])
    told["target"] = _target(conn, told.pop("target_id"))
    counts = conn.execute(
        "SELECT sum(state = 'failed'), sum(state = 'pending'), sum(state = 'done'), count(*)"
        " FROM job_item WHERE job_id = ?",
        (job_id,),
    ).fetchone()
    told["failed_count"] = int(counts[0] or 0)
    told["pending_count"] = int(counts[1] or 0)
    told["succeeded_count"] = int(counts[2] or 0)
    told["item_count"] = int(counts[3] or 0)
    told["derived"] = _lifecycle(told, now)
    told["settled"] = told["state"] in jobs.TERMINAL

    failures = conn.execute(
        "SELECT item_id, error FROM job_item WHERE job_id = ? AND state = 'failed' ORDER BY item_id LIMIT 200",
        (job_id,),
    ).fetchall()
    told["failures"] = [
        {**(_named_item(conn, told["kind"], payload, item) or {"id": item}), "error": error} for item, error in failures
    ]

    history = ledger.for_job(conn, job_id)
    told["event_count"] = len(history)
    told["last_event_id"] = history[-1]["id"] if history else 0
    told["attempts"] = [
        {"at": e["at"], "type": e["type"], **(e["data"] or {})}
        for e in history
        if e["type"] in ("job.claimed", "job.reclaimed", "job.paused")
    ]
    told["defects"] = [
        {
            "at": e["at"],
            "id": e["id"],
            "item": _named_item(conn, told["kind"], payload, e["item_id"]),
            **(e["data"] or {}),
        }
        for e in history
        if e["type"] == "worker.turn_failed"
    ]
    current = None
    last_phase = None
    for e in reversed(history):
        if e["type"] in (
            "item.done",
            "item.failed",
            "worker.turn_failed",
            "job.done",
            "job.failed",
            "job.cancelled",
            "job.paused",
        ):
            break
        if e["type"] == "item.started":
            current = {
                "item": _named_item(conn, told["kind"], payload, e["item_id"]),
                "since": e["at"],
                "event_id": e["id"],
            }
            break
    for e in reversed(history):
        if e["type"] in ("phase.started", "phase.progress", "phase.finished", "item.observed"):
            last_phase = {
                "phase": e["phase"],
                "type": e["type"],
                "message": e["message"],
                "at": e["at"],
                "data": e["data"],
            }
            break
    told["current"] = {
        "item": current["item"] if current else None,
        "since": current["since"] if current else None,
        "phase": None,
        "last_settled_phase": last_phase,
    }
    told["recent_events"] = history[-recent_events:]
    return told


def events(conn, *, job_id: int | None = None, after: int = 0, limit: int = 500) -> dict:
    """A page of the ledger, ascending by id: the whole ledger or one
    job's. `next_after` is the cursor for the following page, None when
    this page reached the head."""
    limit = max(1, min(int(limit), ledger.PAGE_MOST))
    page = (
        ledger.for_job(conn, job_id, after=after, limit=limit)
        if job_id is not None
        else ledger.since(conn, after, limit=limit)
    )
    return {
        "events": page,
        "after": after,
        "next_after": page[-1]["id"] if len(page) == limit else None,
        "last_id": ledger.last_id(conn),
    }


def events_before(conn, before: int, *, job_id: int | None = None, limit: int = 500) -> list[dict]:
    """The `limit` events with id < `before`, ascending: the page above
    the one a reader holds. Walks the index backwards and stops."""
    return ledger.before(conn, before, job_id=job_id, limit=limit)


def items(conn, job_id: int, *, state: str | None = None, after: int = 0, limit: int = 200) -> dict:
    """A page of one job's items by state, in item order; each named and
    linked when the kind's items are files."""
    if state is not None and state not in ("pending", "done", "failed"):
        raise ValueError(f"{state!r} is not an item state")
    row = conn.execute("SELECT kind, payload FROM job WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        raise LookupError(f"no job {job_id}")
    kind, payload = row[0], _json(row[1])
    limit = max(1, min(int(limit), ledger.PAGE_MOST))
    if state is None:
        rows = conn.execute(
            "SELECT item_id, state, error FROM job_item WHERE job_id = ? AND item_id > ? ORDER BY item_id LIMIT ?",
            (job_id, after, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT item_id, state, error FROM job_item WHERE job_id = ? AND state = ? AND item_id > ?"
            " ORDER BY item_id LIMIT ?",
            (job_id, state, after, limit),
        ).fetchall()
    return {
        "items": [
            {**(_named_item(conn, kind, payload, item) or {"id": item}), "state": st, "error": error}
            for item, st, error in rows
        ],
        "next_after": rows[-1][0] if len(rows) == limit else None,
    }
