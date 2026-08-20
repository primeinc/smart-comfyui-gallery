"""Run what was asked for: one claimed job, item by item, off the row.

The row is the truth the whole time. Progress is `done_count` moving,
cancellation is a flag the loop reads between items, resumption is
`job_item` rows still pending, and a killed process leaves nothing to
clean up -- its lease expires and the next `run_next` picks the job up
where the items say it stopped.

Failure is split on purpose. The failures work is EXPECTED to produce --
an unreadable file, a corrupt image, a row that vanished mid-job -- are
recorded on the item and the job carries on. Anything else propagates and
takes the worker turn down with it: the job stays `running`, the lease
runs out, and the work is reclaimed rather than marked broken by a bug in
the code that judged it. A runner that converted every exception into an
item error would turn its own defects into permanent verdicts about files.
"""

from __future__ import annotations

import json
import sqlite3

from . import jobs

#: What a handler is allowed to fail with, per item. Everything else is a
#: defect in the handler, not a fact about the item.
ITEM_FAILURES = (OSError, ValueError, RuntimeError, LookupError, sqlite3.Error)


def submit_verify(conn, now: float) -> int:
    """An integrity sweep: is every present file still the bytes we recorded?

    One item per file that has a recorded hash. Finds silent corruption and
    out-of-band edits; writes nothing, so a mismatch is a finding on the
    item, never a mutation of the library.
    """
    items = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM file WHERE missing_since IS NULL AND content_sha256 IS NOT NULL ORDER BY id"
        )
    ]
    return jobs.submit(conn, "hash", now, items=items)


def _verify_item(conn, file_id: int, payload: dict, now: float) -> None:
    from . import detect, scan

    stored = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()
    if stored is None:
        raise LookupError(f"file {file_id} left the library mid-job")
    actual = scan.sha256_of(detect.path_of(conn, file_id))
    if actual != stored[0]:
        raise ValueError(f"bytes changed behind the library's back: recorded {stored[0][:12]}, found {actual[:12]}")


def submit_faces(conn, now: float, *, models_dir: str) -> int:
    """Face detection over every present image, as an explicit job."""
    items = [
        row[0] for row in conn.execute("SELECT id FROM file WHERE kind = 'image' AND missing_since IS NULL ORDER BY id")
    ]
    return jobs.submit(conn, "detect_faces", now, payload={"models_dir": models_dir}, items=items)


_BACKENDS: dict = {}


def _face_item(conn, file_id: int, payload: dict, now: float) -> None:
    from smartgallery_ai.faces import OpenCVFaceBackend

    from . import detect

    models_dir = payload["models_dir"]
    backend = _BACKENDS.get(models_dir)
    if backend is None:
        backend = _BACKENDS[models_dir] = OpenCVFaceBackend(models_dir)
    detect.harvest(conn, backend, file_id, detect.path_of(conn, file_id), now)


#: kind -> handler(conn, item_id, payload, now). The names are the schema's:
#: `job.kind` is CHECK-constrained (db/schema.sql:493-495) so a typo is an
#: IntegrityError at submit, never a job that queues and waits forever.
HANDLERS = {
    "hash": _verify_item,
    "detect_faces": _face_item,
}


def run_next(conn, owner: str, now: float, *, handlers=None, kinds=None, budget: int | None = None) -> dict | None:
    """One worker turn: claim the next runnable job and work it.

    Returns None when nothing is runnable, otherwise a summary of what this
    turn did. `budget` bounds how many items the turn performs -- the job
    stays `running` under its lease and the next turn (this process or any
    other) continues from the items still pending, which is the resumption
    contract stated on db/jobs.py.
    """
    handlers = HANDLERS if handlers is None else handlers
    claimed = jobs.claim(conn, owner, now, kinds=kinds)
    if claimed is None:
        return None
    job_id, fence = claimed

    kind, raw = conn.execute("SELECT kind, payload FROM job WHERE id = ?", (job_id,)).fetchone()
    handler = handlers.get(kind)
    if handler is None:
        jobs.settle(conn, job_id, fence, "failed", now, error=f"no handler for kind {kind!r}")
        return {"job": job_id, "state": "failed", "did": 0}
    payload = json.loads(raw) if raw else {}

    did = failed = 0
    for item in jobs.pending(conn, job_id):
        if budget is not None and did >= budget:
            # A deliberate stop, not a death: the lease is expired on the
            # spot so the very next turn -- any process -- resumes the job
            # instead of waiting out a liveness timeout meant for crashes.
            jobs.pause(conn, job_id, fence, now)
            return {"job": job_id, "state": "running", "did": did, "failed": failed}
        if jobs.cancelled(conn, job_id):
            jobs.settle(conn, job_id, fence, "cancelled", now)
            return {"job": job_id, "state": "cancelled", "did": did, "failed": failed}
        try:
            handler(conn, item, payload, now)
        except ITEM_FAILURES as why:
            jobs.finish_item(conn, job_id, fence, item, error=str(why))
            failed += 1
        else:
            jobs.finish_item(conn, job_id, fence, item)
        did += 1
        jobs.heartbeat(conn, job_id, fence, now)

    jobs.settle(conn, job_id, fence, "done", now)
    return {"job": job_id, "state": "done", "did": did, "failed": failed}
