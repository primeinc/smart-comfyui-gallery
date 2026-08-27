"""The activity surface: persisted jobs and their deltas, as HTML.

A presentation Adapter and nothing more. `db/jobs.py` owns the rows, the
worker and the request seam (sg_web/submitting.py) own the deltas, the
channel owns transport; this module only projects them into
`_activity_list.html` / `_job.html` so the shell can mount the list on
every page and htmx can swap one `<li id="job-N">` per delta
(bigskysoftware/htmx-extensions@1358232 src/ws/ws.js: every child of a
message is an out-of-band swap keyed by id; bigskysoftware/htmx@v2.0.7
www/content/attributes/hx-swap-oob.md for the `beforeend:#selector` form).

One list, two moments:

    cold load    `rows` -- the active jobs then the recently settled ones,
                 so a page opened after a sweep shows what a page that
                 watched it shows
    live         `render_list` / `render_delta` -- what /ws/jobs?as=html
                 sends: that same list first, then one <li> per delta,
                 appended when the connection has never seen the job,
                 replaced otherwise

The per-connection `seen` set is what makes append-vs-replace decidable
without a DB read per delta: the list names every job that exists, so a
delta for an unnamed id is a job born after it. A settled job leaves the
set -- nothing is published about a job after its terminal state -- so
the set is bounded by the jobs alive during the connection.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jinja2 import pass_context

from db import connect, jobs
from sg_web import console

#: How many settled jobs the cold list carries -- db/jobs.py RECENT,
#: which sits beside the read it bounds.
RECENT = jobs.RECENT


def _view(
    id_: int, kind: str, state: str, done: int, total: int | None, cancel_requested: int, derive=None, where=None
) -> dict:
    return {
        "id": id_,
        "kind": kind,
        "what": console.describe_kind(kind, derive, total, where),
        "state": state,
        "done": done,
        "total": total,
        "cancelling": bool(cancel_requested) and state not in jobs.TERMINAL,
        "settled": state in jobs.TERMINAL,
    }


def row_view(row: Mapping[str, Any]) -> dict:
    """One `jobs.active` / `jobs.recent` row as the template sees it."""
    return _view(
        row["id"],
        row["kind"],
        row["state"],
        row["done_count"],
        row["total"],
        row["cancel_requested"],
        row.get("derive"),
        row.get("path"),
    )


def delta_view(delta: Mapping[str, Any]) -> dict:
    """One published delta (`{job, kind, state, done, total,
    cancel_requested}`, db/runner.py run_next and sg_web/submitting.py)
    as the template sees it -- the same shape as a row."""
    return _view(
        delta["job"],
        delta["kind"],
        delta["state"],
        delta["done"],
        delta["total"],
        delta.get("cancel_requested", 0),
        delta.get("derive"),
        delta.get("path"),
    )


def rows(db_path: str) -> list[dict]:
    """The cold list: every active job, then the settled tail. One
    read-only connection, two bounded statements."""
    conn = connect.connect(db_path, read_only=True)
    try:
        return [row_view(row) for row in jobs.active(conn)] + [row_view(row) for row in jobs.recent(conn, RECENT)]
    finally:
        connect.close(conn)


def render_list(engine, listed: list[dict]) -> str:
    """The whole list, as an out-of-band replacement of #activity-jobs."""
    return engine.get_template("_activity_list.html").render(jobs=listed, oob=True)


def render_delta(engine, delta: Mapping[str, Any], seen: set[int]) -> str:
    """One delta as the fragment htmx swaps: a replacement for a job the
    connection has already shown, an append for one it has not. A
    terminal delta retires the id from `seen`."""
    view = delta_view(delta)
    known = view["id"] in seen
    if view["settled"]:
        seen.discard(view["id"])
    else:
        seen.add(view["id"])
    if known:
        return engine.get_template("_job.html").render(job=view, oob=True)
    return engine.get_template("_job_append.html").render(job=view)


@pass_context
def active_jobs(context: Mapping[str, Any]) -> list[dict]:
    """The shell's `activity_jobs()`: the cold list for the page being
    served, read through the request's app state so no handler passes
    jobs around. Rows only -- the shell includes the template over them,
    so nothing here produces markup. Without a request (a render outside
    the application) there are no rows to show."""
    request = context.get("request")
    if request is None:
        return []
    return rows(request.app.state.db_path)


def register(engine) -> None:
    """Install `activity_jobs` as a global on the Jinja environment the
    application renders with. Called before any template loads: Jinja
    forbids changing environment globals afterwards (pallets/jinja@3.1.6
    docs/api.rst "The Global Namespace")."""
    engine.engine.globals["activity_jobs"] = active_jobs
