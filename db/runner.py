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


def submit_faces(conn, now: float, *, models_dir: str, thumbs_dir: str | None = None) -> int:
    """Face detection over every present picture and video, as one job.

    A face on a video is a face on a sampled frame; the handler routes by
    the file kind, and the schema already keys every face to the moment it
    was seen (`derived_face_instance.sample_id`).

    `thumbs_dir` rides in the payload when thumbnail precaching is on:
    detection decodes every frame anyway, and the cache takes the decoded
    pixels on the way past instead of re-decoding later on first view.
    """
    items = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM file WHERE missing_since IS NULL"
            " AND kind IN ('image', 'animated_image', 'video') ORDER BY id"
        )
    ]
    payload: dict = {"models_dir": models_dir}
    if thumbs_dir is not None:
        payload["thumbs_dir"] = thumbs_dir
    return jobs.submit(conn, "detect_faces", now, payload=payload, items=items)


def submit_cluster(conn, now: float) -> int:
    """Group every embedding space's faces into people, as one job.

    One item per (model_id, model_version) that holds embedded faces. The
    spaces ride in the payload -- `job_item` holds integers, so each item
    is an index into that list, fixed at submit time. No spaces means a
    job with nothing to do, which settles `done` honestly rather than
    being refused: "cluster an unindexed library" is an answer, not an
    error.
    """
    spaces = conn.execute(
        "SELECT DISTINCT model_id, model_version FROM derived_face_instance"
        " WHERE embedding IS NOT NULL ORDER BY model_id, model_version"
    ).fetchall()
    return jobs.submit(
        conn,
        "cluster_faces",
        now,
        payload={"spaces": [list(space) for space in spaces]},
        items=list(range(len(spaces))),
    )


def _cluster_item(conn, index: int, payload: dict, now: float) -> None:
    """Cluster one embedding space and give every group a person.

    The run's whole answer is replaced -- clusters, inferred appearances,
    and the placeholder people minted for groups nobody has named. Names
    are never carried across by similarity: `seed_clusters_from_assertions`
    re-applies them from what a human wrote down, and only the groups
    still unnamed after that get a fresh unnamed person, addressable at
    `/p/person-<short-id>` until somebody names them.
    """
    from . import derived, naming

    model_id, model_version = payload["spaces"][index]
    # Method and threshold pinned once and passed to BOTH calls: recomputing
    # the run identity from separately-spelled defaults is how a drift makes
    # the DELETE below clear a different run's attributions.
    pinned = derived.threshold_for(model_id)
    derived.cluster(conn, model_id, model_version, now, method=derived.DEFAULT_METHOD, threshold=pinned)
    run_id = derived.run_for(conn, model_id, model_version, derived.DEFAULT_METHOD, pinned, now)
    conn.execute("DELETE FROM derived_file_person WHERE run_id = ?", (run_id,))
    derived.seed_clusters_from_assertions(conn, run_id)
    for (cluster_id,) in conn.execute(
        "SELECT id FROM derived_face_cluster WHERE run_id = ? AND person_id IS NULL ORDER BY id",
        (run_id,),
    ).fetchall():
        person_id = naming.claim(conn, "person", "")
        conn.execute("INSERT INTO person(id, name, created_at) VALUES(?, NULL, ?)", (person_id, now))
        conn.execute("UPDATE derived_face_cluster SET person_id = ? WHERE id = ?", (person_id, cluster_id))
        for file_id, faces in conn.execute(
            "SELECT fi.file_id, count(*) FROM derived_face_membership m"
            " JOIN derived_face_instance fi ON fi.id = m.face_id"
            " WHERE m.cluster_id = ? GROUP BY fi.file_id",
            (cluster_id,),
        ).fetchall():
            derived.attribute(conn, file_id, person_id, run_id, model_id, model_version, face_count=faces)
    # A re-run dissolves groups, and a placeholder person whose group
    # dissolved is an address about nothing. Only the unnamed go: a name
    # is a human's word and keeps its entity regardless.
    conn.execute(
        "DELETE FROM entity WHERE kind = 'person' AND id IN ("
        " SELECT p.id FROM person p WHERE p.name IS NULL"
        " AND NOT EXISTS (SELECT 1 FROM person_assertion pa WHERE pa.person_id = p.id)"
        " AND NOT EXISTS (SELECT 1 FROM derived_face_cluster c WHERE c.person_id = p.id)"
        " AND NOT EXISTS (SELECT 1 FROM derived_file_person fp WHERE fp.person_id = p.id)"
        " AND NOT EXISTS (SELECT 1 FROM feedback fb WHERE fb.person_id = p.id))"
    )


_BACKENDS: dict = {}


def _face_item(conn, file_id: int, payload: dict, now: float) -> None:
    from vision.faces import OpenCVFaceBackend

    from . import detect

    models_dir = payload["models_dir"]
    thumbs_dir = payload.get("thumbs_dir")
    backend = _BACKENDS.get(models_dir)
    if backend is None:
        backend = _BACKENDS[models_dir] = OpenCVFaceBackend(models_dir)
    kind = conn.execute("SELECT kind FROM file WHERE id = ?", (file_id,)).fetchone()[0]
    path = detect.path_of(conn, file_id)
    if kind == "video":
        detect.harvest_video(conn, backend, file_id, path, now, thumbs_dir=thumbs_dir)
    else:
        detect.harvest(conn, backend, file_id, path, now, thumbs_dir=thumbs_dir)


#: kind -> handler(conn, item_id, payload, now). The names are the schema's:
#: `job.kind` is CHECK-constrained (db/schema.sql:493-495) so a typo is an
#: IntegrityError at submit, never a job that queues and waits forever.
HANDLERS = {
    "hash": _verify_item,
    "detect_faces": _face_item,
    "cluster_faces": _cluster_item,
}


def run_next(
    conn,
    owner: str,
    now: float,
    *,
    handlers=None,
    kinds=None,
    gate=None,
    budget: int | None = None,
    clock=None,
    on_progress=None,
    should_stop=None,
) -> dict | None:
    """One worker turn: claim the next runnable job and work it.

    Returns None when nothing is runnable, otherwise a summary of what this
    turn did. `budget` bounds how many items the turn performs -- the job
    stays `running` under its lease and the next turn (this process or any
    other) continues from the items still pending, which is the resumption
    contract stated on db/jobs.py.

    `clock` supplies the time for heartbeats and settlement; without one,
    `now` stands for the whole turn. A long-lived worker MUST pass one: a
    heartbeat stamped with claim time extends the lease to claim + 60s
    forever, so any single job outliving the lease was reclaimed mid-run
    by the next worker while still being worked.

    `on_progress` hears every observable change -- claim, each finished
    item, the terminal state -- as `{job, kind, state, done, total}`. This
    is the delta feed the live channel publishes; the row stays the truth
    a subscriber renders from cold. Every state change COMMITS BEFORE it
    is spoken, terminal states included: a delta describing an uncommitted
    write invites the subscriber to read the row and find it behind what
    the wire just said -- caught live by a client whose snapshot read
    'running' after its socket said 'done'.
    """
    handlers = HANDLERS if handlers is None else handlers
    tick = clock if clock is not None else (lambda: now)
    tell = on_progress if on_progress is not None else (lambda delta: None)
    claimed = jobs.claim(conn, owner, now, kinds=kinds, gate=gate)
    if claimed is None:
        return None
    job_id, fence = claimed
    conn.commit()

    kind, raw = conn.execute("SELECT kind, payload FROM job WHERE id = ?", (job_id,)).fetchone()

    def spoke(state: str, moved) -> None:
        tell({"job": job_id, "kind": kind, "state": state, "done": moved.done, "total": moved.total})

    spoke("running", jobs.progress(conn, job_id))
    handler = handlers.get(kind)
    if handler is None:
        jobs.settle(conn, job_id, fence, "failed", tick(), error=f"no handler for kind {kind!r}")
        conn.commit()
        spoke("failed", jobs.progress(conn, job_id))
        return {"job": job_id, "state": "failed", "did": 0}
    payload = json.loads(raw) if raw else {}

    did = failed = 0
    for item in jobs.pending(conn, job_id):
        if (budget is not None and did >= budget) or (should_stop is not None and should_stop()):
            # A deliberate stop, not a death: the lease is expired on the
            # spot so the very next turn -- any process -- resumes the job
            # instead of waiting out a liveness timeout meant for crashes.
            # `should_stop` is how a shutting-down worker leaves a long
            # job at an item boundary instead of holding the exit hostage.
            jobs.pause(conn, job_id, fence, tick())
            conn.commit()
            return {"job": job_id, "state": "running", "did": did, "failed": failed}
        if jobs.cancelled(conn, job_id):
            jobs.settle(conn, job_id, fence, "cancelled", tick())
            conn.commit()
            spoke("cancelled", jobs.progress(conn, job_id))
            return {"job": job_id, "state": "cancelled", "did": did, "failed": failed}
        try:
            handler(conn, item, payload, now)
        except ITEM_FAILURES as why:
            # The dead handler's half-finished writes ride the same open
            # transaction as the failure record about to be committed --
            # dropped first, so the record carries the verdict and nothing
            # the handler did not live to finish.
            conn.rollback()
            moved = jobs.finish_item(conn, job_id, fence, item, error=str(why))
            failed += 1
        else:
            moved = jobs.finish_item(conn, job_id, fence, item)
        did += 1
        jobs.heartbeat(conn, job_id, fence, tick())
        conn.commit()
        spoke("running", moved)

    jobs.settle(conn, job_id, fence, "done", tick())
    conn.commit()
    spoke("done", jobs.progress(conn, job_id))
    return {"job": job_id, "state": "done", "did": did, "failed": failed}
