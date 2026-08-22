"""What every job submit does after its row commits: speak, then wake.

The worker publishes every change it makes (db/runner.py run_next), but
a job is born `queued` by a request, before any worker has touched it.
Without this, a queued job was invisible on the feed until a claim --
a subscriber watching the activity surface saw nothing happen when it
pressed the button, and a worker switched off hid the job entirely
until a reload. The queued delta carries the row's own state, read back
after the commit, so the wire never says something the rows do not.
"""

from __future__ import annotations

from litestar.datastructures import State

from db import jobs


def nudge(state: State) -> None:
    """Tell the worker there is work, so pickup is immediate rather than
    on its idle cadence. No worker in this process (build_app worker=False)
    means nothing to wake."""
    wake = getattr(state, "worker_wake", None)
    if wake is not None:
        wake.set()


def submitted(state: State, conn, job_id: int) -> dict:
    """The committed job's snapshot, announced on the feed, the worker
    woken. Call after `conn.commit()`; the delta describes committed rows."""
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
            }
        )
    nudge(state)
    return told
