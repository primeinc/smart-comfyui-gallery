"""What a request does to a job row after its commit: speak, then wake.

The worker publishes every change it makes (db/runner.py run_next), but
two changes are made by requests, not workers: a job is born `queued` by
a submit, and `cancel_requested` is set by a cancel. Without this, both
were invisible on the feed until a worker next touched the job -- a
subscriber watching the activity surface saw nothing happen when it
pressed the button, and a worker switched off hid the job entirely until
a reload. The delta carries the row's own state, read back after the
commit, so the wire never says something the rows do not.

The same two changes are ledger rows (db/jobs.py submit and cancel
record them), spoken on the events channel here for the same reason
and under the same rule: after the commit, read back, never invented.
"""

from __future__ import annotations

from litestar.datastructures import State

from db import jobs, ledger


def nudge(state: State) -> None:
    """Tell the worker there is work, so pickup is immediate rather than
    on its idle cadence. No worker in this process (build_app worker=False)
    means nothing to wake."""
    wake = getattr(state, "worker_wake", None)
    if wake is not None:
        wake.set()


def announce(state: State, conn, job_id: int, *, event_type: str | None = None) -> dict:
    """The committed job's snapshot, spoken on the feed as a delta of the
    same shape the worker publishes; and, when `event_type` names the
    ledger row this request wrote, that row spoken on the events channel.
    Call after `conn.commit()`."""
    told = jobs.snapshot(conn, job_id)
    publish = getattr(state, "publish", None)
    if publish is not None:
        publish(
            {
                "job": told["id"],
                "kind": told["kind"],
                "state": told["state"],
                "done": told["done_count"],
                "total": told["total"],
                "cancel_requested": told["cancel_requested"],
            }
        )
    publish_event = getattr(state, "publish_event", None)
    if publish_event is not None and event_type is not None:
        latest = ledger.latest_for_job(conn, job_id)
        if latest is not None and latest["type"] == event_type:
            publish_event(latest)
    return told


def submitted(state: State, conn, job_id: int) -> dict:
    """A freshly queued job: announced, and the worker woken for it."""
    told = announce(state, conn, job_id, event_type="job.submitted")
    nudge(state)
    return told
