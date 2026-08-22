"""The activity surface: persisted jobs and their deltas, as HTML.

A presentation Adapter and nothing more. `db/jobs.py` owns the rows, the
worker owns the deltas, the channel owns transport; this module only
projects them into `_activity.html` / `_job.html` so the shell can mount
the list on every page and htmx can swap one `<li id="job-N">` per delta
(bigskysoftware/htmx-extensions@1358232 src/ws/ws.js: every child of a
message is an out-of-band swap keyed by id; bigskysoftware/htmx@v2.0.7
www/content/attributes/hx-swap-oob.md for the `beforeend:#selector` form).

Two shapes, one source:

    cold load    `active_jobs` -- a Jinja global the shell calls for the
                 rows, then includes `_activity.html` over them
    live         `render_list` / `render_delta` -- what /ws/jobs?as=html
                 sends: the whole list first, then one <li> per delta,
                 appended when the connection has never seen the job,
                 replaced otherwise

The per-connection `seen` set is what makes append-vs-replace decidable
without a DB read per delta: the snapshot names every job that exists,
so a delta for an unnamed id is a job born after it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jinja2 import pass_context

from db import connect, jobs

TERMINAL = ("done", "failed", "cancelled")


def row_view(row: Mapping[str, Any]) -> dict:
    """One `jobs.active` row as the template sees it."""
    return {
        "id": row["id"],
        "kind": row["kind"],
        "state": row["state"],
        "done": row["done_count"],
        "total": row["total"],
    }


def delta_view(delta: Mapping[str, Any]) -> dict:
    """One published delta (`{job, kind, state, done, total}`, db/runner.py
    run_next and sg_web/submitting.py) as the template sees it -- the
    same shape as a row."""
    return {
        "id": delta["job"],
        "kind": delta["kind"],
        "state": delta["state"],
        "done": delta["done"],
        "total": delta["total"],
    }


def active_rows(db_path: str) -> list[dict]:
    conn = connect.connect(db_path, read_only=True)
    try:
        return [row_view(row) for row in jobs.active(conn)]
    finally:
        connect.close(conn)


def render_list(engine, rows: list[dict]) -> str:
    """The whole list, as an out-of-band replacement of #activity-jobs."""
    return engine.get_template("_activity_list.html").render(jobs=rows)


def render_delta(engine, delta: Mapping[str, Any], seen: set[int]) -> str:
    """One delta as the fragment htmx swaps: a replacement for a job the
    connection has already shown, an append for one it has not."""
    view = delta_view(delta)
    if view["id"] in seen:
        return engine.get_template("_job.html").render(job=view, oob=True)
    seen.add(view["id"])
    return engine.get_template("_job_append.html").render(job=view)


@pass_context
def active_jobs(context: Mapping[str, Any]) -> list[dict]:
    """The shell's `activity_jobs()`: the active rows for the page being
    served, read through the request's app state so no handler passes
    jobs around. Rows only -- the shell includes the template over them,
    so nothing here produces markup. Without a request (a render outside
    the application) there are no rows to show."""
    request = context.get("request")
    if request is None:
        return []
    return active_rows(request.app.state.db_path)


def register(engine) -> None:
    """TemplateConfig.engine_callback (litestar-org/litestar@v2.24.0
    litestar/template/config.py:46-51): install `activity_jobs` as a
    global on the Jinja environment the application renders with."""
    engine.engine.globals["activity_jobs"] = active_jobs
