"""The application over the schema: every page a query, every sweep a job.

The skeleton of plan Phase 2, held to its two rules. Addresses are entity
slugs resolved through `db.pages.resolve` -- never paths, never raw ids --
and nothing expensive starts by itself: a sweep is a `job` row somebody
POSTs into existence, worked by explicit worker turns, cancellable and
resumable because the row is the truth.

Handlers are synchronous on purpose: sqlite is synchronous, and Litestar
runs sync handlers on its thread pool when told so
(refs/litestar-org/litestar/docs/topics/sync-vs-async.rst). Each request
opens its own connection, which is what makes that safe -- sqlite3
connections refuse cross-thread use, and the pool gives no thread pinning.
"""

from __future__ import annotations

import pathlib
import sqlite3
import time

from litestar import Litestar, get, post
from litestar.datastructures import State
from litestar.exceptions import NotFoundException
from litestar.response import Redirect

from db import connect, derived, jobs, library, naming, pages, runner, scan
from sg_web import home


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _rows(cursor_rows, columns) -> list[dict]:
    return [dict(zip(columns, row, strict=True)) for row in cursor_rows]


@get("/health", sync_to_thread=False)
def health() -> str:
    return "ok"


@get("/people", sync_to_thread=True)
def people(state: State) -> list[dict]:
    """Everyone, most pictures first -- the People index."""
    conn = _connect(state.db_path)
    try:
        return _rows(pages.people_by_most(conn), ("name", "slug", "pictures"))
    finally:
        conn.close()


@get("/p/{slug:str}", sync_to_thread=True)
def person(state: State, slug: str) -> dict | Redirect:
    """One person: their pictures, and where those live on disk.

    A retired slug redirects to the live one rather than answering, so one
    person never has two addresses serving content -- the rename contract
    the naming module carries `is_current` for."""
    conn = _connect(state.db_path)
    try:
        found = naming.resolve(conn, "person", slug)
        if found is None:
            raise NotFoundException(f"no person at /p/{slug}")
        person_id, is_current = found
        if not is_current:
            live = naming.entity_slug(conn, person_id)
            if live is not None:
                return Redirect(path=f"/p/{live[1]}", status_code=301)
        name = conn.execute("SELECT name FROM person WHERE id = ?", (person_id,)).fetchone()
        return {
            "slug": slug,
            "name": name[0] if name else None,
            "pictures": _rows(pages.person_files(conn, person_id), ("slug", "name")),
            "across_folders": _rows(
                pages.person_across_folders(conn, person_id),
                ("folder", "folder_slug", "pictures"),
            ),
        }
    finally:
        conn.close()


@get("/clusterings", sync_to_thread=True)
def clusterings(state: State) -> list[dict]:
    """Every clustering run held side by side, primary first."""
    conn = _connect(state.db_path)
    try:
        return pages.clusterings(conn)
    finally:
        conn.close()


@get("/ways", sync_to_thread=True)
def ways(state: State) -> list[dict]:
    """What the library can be searched by, generated from what it holds."""
    conn = _connect(state.db_path)
    try:
        return _rows(pages.ways(conn), ("source", "key", "value_kind", "occurrences"))
    finally:
        conn.close()


@get("/jobs", sync_to_thread=True)
def active_jobs(state: State) -> list[dict]:
    conn = _connect(state.db_path)
    try:
        return jobs.active(conn)
    finally:
        conn.close()


@get("/jobs/{job_id:int}", sync_to_thread=True)
def job_snapshot(state: State, job_id: int) -> dict:
    """The persisted snapshot -- what a client renders from cold. A page
    reload or a dropped socket recovers by reading this, never a replay."""
    conn = _connect(state.db_path)
    try:
        try:
            return jobs.snapshot(conn, job_id)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
    finally:
        conn.close()


@post("/jobs/verify", sync_to_thread=True)
def submit_verify(state: State) -> dict:
    """Ask for an integrity sweep. Nothing runs until a worker turn."""
    conn = _connect(state.db_path)
    try:
        job_id = runner.submit_verify(conn, time.time())
        conn.commit()
        return jobs.snapshot(conn, job_id)
    finally:
        conn.close()


@post("/jobs/faces", sync_to_thread=True)
def submit_faces(state: State, data: dict) -> dict:
    """Ask for face detection over the library, with the models named."""
    conn = _connect(state.db_path)
    try:
        weights = data.get("models_dir") or str(home.models_dir())
        job_id = runner.submit_faces(conn, time.time(), models_dir=weights)
        conn.commit()
        return jobs.snapshot(conn, job_id)
    finally:
        conn.close()


@post("/jobs/{job_id:int}/cancel", sync_to_thread=True)
def cancel_job(state: State, job_id: int) -> dict:
    """Ask a job to stop. The runner stops it, at an item boundary."""
    conn = _connect(state.db_path)
    try:
        jobs.cancel(conn, job_id)
        conn.commit()
        return jobs.snapshot(conn, job_id)
    finally:
        conn.close()


@post("/worker/turn", sync_to_thread=True)
def worker_turn(state: State, data: dict | None = None) -> dict:
    """One explicit worker turn. `budget` bounds the items it performs;
    a bounded turn leaves the job running and resumable, which is what
    makes progress observable over plain requests."""
    conn = _connect(state.db_path)
    try:
        budget = (data or {}).get("budget")
        turn = runner.run_next(
            conn,
            owner="web-worker",
            now=time.time(),
            budget=int(budget) if budget is not None else None,
        )
        conn.commit()
        return turn if turn is not None else {"state": "idle"}
    finally:
        conn.close()


@get("/roots", sync_to_thread=True)
def roots(state: State) -> list[dict]:
    """Every media directory this library reads, and whether it is
    reachable right now. Media roots are rows, not configuration: any
    number of directories, anywhere, and they travel with the database."""
    conn = _connect(state.db_path)
    try:
        return [{"id": root_id, "path": path, "online": online} for root_id, path, online in library.check_roots(conn)]
    finally:
        conn.close()


@post("/roots", sync_to_thread=True)
def add_root(state: State, data: dict) -> dict:
    """Register a media directory. Nothing is read until a scan is asked
    for -- registering is a statement of intent, not a sweep."""
    conn = _connect(state.db_path)
    try:
        root_id = library.add_root(conn, data["path"], data.get("kind", "library"), time.time())
        conn.commit()
        return {"id": root_id, "path": data["path"]}
    finally:
        conn.close()


@post("/roots/{root_id:int}/scan", sync_to_thread=True)
def scan_root(state: State, root_id: int) -> dict:
    """Walk one root and reconcile the library with what is on disk."""
    conn = _connect(state.db_path)
    try:
        row = conn.execute("SELECT path FROM root WHERE id = ?", (root_id,)).fetchone()
        if row is None:
            raise NotFoundException(f"no root {root_id}")
        result = scan.scan(conn, root_id, row[0], time.time())
        conn.commit()
        return {
            "root": root_id,
            "added": result.added,
            "matched": result.matched,
            "replaced": result.replaced,
            "ambiguous": result.ambiguous,
            "missing": result.missing,
            "hashed": result.hashed,
        }
    finally:
        conn.close()


@post("/clusterings/choose", sync_to_thread=True)
def choose_primary(state: State) -> dict:
    """Re-rank every run and set the default the People page shows."""
    conn = _connect(state.db_path)
    try:
        chosen = derived.choose_primary(conn)
        conn.commit()
        return {"primary_run": chosen}
    finally:
        conn.close()


def build_app(db_path: str | None = None) -> Litestar:
    """The application, bound to one database file.

    With no path, the run lives in its home directory (sg_web/home.py) and
    a database that does not exist yet is created from the schema -- a
    first run needs nothing but the command that starts it.
    """
    where = pathlib.Path(db_path) if db_path else home.db_path()
    if not where.exists():
        fresh = connect.connect(where)
        fresh.executescript(connect.schema_sql())
        fresh.commit()
        fresh.close()
    app = Litestar(
        route_handlers=[
            health,
            people,
            person,
            clusterings,
            ways,
            roots,
            add_root,
            scan_root,
            active_jobs,
            job_snapshot,
            submit_verify,
            submit_faces,
            cancel_job,
            worker_turn,
            choose_primary,
        ],
    )
    app.state.db_path = str(where)
    return app
