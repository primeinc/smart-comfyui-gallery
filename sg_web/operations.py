"""The runtime's own surface: the console, roots, sweeps, the worker,
clustering.

Everything here already exists as a machine route in sg_web/app.py --
POST /roots, POST /roots/{id}/scan, POST /jobs/*, POST /settings/{key},
POST /clusterings/choose. Those keep their JSON shape for machines. This
module is the BROWSER's way in: one page, forms that post url-encoded
(what an htmx form sends without help), fragments back in place. It is
registered as one Litestar Router under /operations, so every operational
page and form shares that prefix and whatever policy the layer grows
(litestar-org/litestar@v2.24.0 docs/usage/routing/overview.rst "Routers";
litestar/router.py Router.__init__ for the layered kwargs).

The console is the expert depth (db/inspecting.py): the health strip,
the job matrix, one job's inspector, the ledger as a tape. It reads the
operations read model and the ledger, never `jobs.active` widened -- the
shell's list stays tiny. Live delivery is /ws/events (sg_web/app.py);
the routes here are what a cold load and a gap-fill read.

Nothing operational is offered anywhere else: the gallery header asks
questions about media, this page runs the library.
"""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import time
from collections.abc import Callable

from litestar import Request, Router, get, post
from litestar.datastructures import State
from litestar.exceptions import ClientException, HTTPException, NotFoundException
from litestar.params import FromPath, URLEncodedBody
from litestar.response import Response, Template
from litestar.status_codes import HTTP_500_INTERNAL_SERVER_ERROR

from db import connect, derived, inspecting, ledger, library, pages, prompts, runner, scan, settings
from sg_web import console, home
from sg_web.presenting import VARIES, wants_json
from sg_web.submitting import submitted

_logger = logging.getLogger(__name__)

Launcher = Callable[[State, object], list[int]]


def _weights(state: State, conn) -> str:
    return str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))


def _ingest(state: State, conn) -> list[int]:
    return [runner.submit_ingest(conn, time.time())]


def _verify(state: State, conn) -> list[int]:
    return [runner.submit_verify(conn, time.time())]


def _phash(state: State, conn) -> list[int]:
    job_id = runner.submit_phash(conn, time.time())
    return [] if job_id is None else [job_id]


def _dupes(state: State, conn) -> list[int]:
    return [runner.submit_dupes(conn, time.time())]


def _thumbs(state: State, conn) -> list[int]:
    job_id = runner.submit_thumbs(conn, time.time(), thumbs_dir=str(home.thumbs_dir(pathlib.Path(state.home))))
    return [] if job_id is None else [job_id]


def _embed(state: State, conn) -> list[int]:
    return runner.submit_embed(conn, time.time(), models_dir=_weights(state, conn))


def _embed_prompts(state: State, conn) -> list[int]:
    return prompts.submit_embed(conn, time.time(), models_dir=_weights(state, conn))


def _faces(state: State, conn) -> list[int]:
    cache = str(home.thumbs_dir(pathlib.Path(state.home))) if settings.flag(conn, "thumbnail_precache") else None
    return [runner.submit_faces(conn, time.time(), models_dir=_weights(state, conn), thumbs_dir=cache)]


def _annotate(state: State, conn) -> list[int]:
    job_id = runner.submit_annotate(conn, time.time(), models_dir=_weights(state, conn))
    return [] if job_id is None else [job_id]


def _cluster(state: State, conn) -> list[int]:
    return [runner.submit_cluster(conn, time.time())]


def _context(state: State, conn) -> list[int]:
    job_id = runner.submit_context(conn, time.time())
    return [] if job_id is None else [job_id]


def _events(state: State, conn) -> list[int]:
    return [runner.submit_events(conn, time.time())]


#: What the page can start, in the order a library is usually built:
#: find files, read them, fingerprint, thumbnail, group copies, embed, detect faces,
#: cluster, interpret time, group events. Each launcher returns the job
#: ids it queued -- the same db/runner.py entry points the JSON routes use.
LAUNCHERS: dict[str, tuple[str, Launcher]] = {
    "ingest": ("read every file's metadata", _ingest),
    "verify": ("verify every file's bytes", _verify),
    "phash": ("fingerprint every picture not yet fingerprinted", _phash),
    "thumbs": ("render every missing thumbnail", _thumbs),
    "dupes": ("group perceptual copies", _dupes),
    "embed": ("embed every picture not yet embedded", _embed),
    "embed_prompts": ("embed every prompt", _embed_prompts),
    "faces": ("detect faces", _faces),
    "cluster": ("cluster faces into people", _cluster),
    "annotate": ("caption every picture not yet captioned", _annotate),
    "context": ("interpret every file not yet interpreted", _context),
    "events": ("propose events", _events),
}


def _roots(conn) -> list[dict]:
    """Every root with its live reachability -- the probe only, no write
    (db/library.py probe_roots); the JSON /roots route is what records."""
    return [{"id": root_id, "path": path, "online": online} for root_id, path, online in library.probe_roots(conn)]


#: How many of the newest ledger rows the cold page carries; the tape
#: pages earlier ones on demand and the feed appends the rest.
TAPE_COLD = 500


def _page_context(state: State) -> dict:
    now = time.time()
    conn = connect.connect(state.db_path, read_only=True)
    try:
        told = {
            "roots": _roots(conn),
            "settings": settings.snapshot(conn),
            "clusterings": pages.clusterings(conn),
            "launchers": [{"kind": kind, "label": label} for kind, (label, _) in LAUNCHERS.items()],
            "notice": None,
            "overview": inspecting.overview(conn, now),
            "matrix": inspecting.matrix(conn, now),
            "tape": [console.envelope(event) for event in ledger.latest(conn, limit=TAPE_COLD)],
            "last_event_id": ledger.last_id(conn),
            "now": now,
        }
        _with_live(state, told)
        return told
    finally:
        connect.close(conn)


@get("/", sync_to_thread=True)
def operations_page(state: State) -> Template:
    return Template(template_name="operations.html", context=_page_context(state), headers=VARIES)


@get("/overview", sync_to_thread=True)
def overview(state: State) -> dict:
    """The health strip and the matrix, from the rows: what the console
    re-reads after a reconnect, and what a machine asks."""
    now = time.time()
    conn = connect.connect(state.db_path, read_only=True)
    try:
        told = {"overview": inspecting.overview(conn, now), "matrix": inspecting.matrix(conn, now)}
    finally:
        connect.close(conn)
    _with_live(state, told)
    return told


def _with_live(state: State, told: dict) -> None:
    """What only the running process knows, beside the rows: whether
    its worker thread is alive, and the latest report inside each
    running job's item (sg_web/app.py live_reports)."""
    thread = getattr(state, "worker_thread", None)
    told["overview"]["worker"]["thread_alive"] = bool(thread is not None and thread.is_alive())
    told["overview"]["worker"]["thread"] = getattr(thread, "name", None)
    live = getattr(state, "live_reports", {})
    for job in told["matrix"]:
        job["what"] = console.describe_kind(job["kind"], job.get("derive"))
        held = live.get(job["id"]) if job["state"] == "running" else None
        job["live"] = (
            {"phase": held.get("phase"), "type": held["type"], "text": held["text"], "item_id": held.get("item_id")}
            if held
            else None
        )


@get("/job/{job_id:int}", sync_to_thread=True)
def job_inspector(state: State, request: Request, job_id: FromPath[int]) -> Template | Response:
    """One job, whole (db/inspecting.py job_detail): JSON to a machine,
    the inspector fragment to the console. Every column of the row is in
    it; the payload is redacted at this seam."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        try:
            told = inspecting.job_detail(conn, job_id, time.time())
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
    finally:
        connect.close(conn)
    told["recent_events"] = [console.envelope(event) for event in told["recent_events"]]
    told["what"] = console.describe_kind(told["kind"], (told.get("payload") or {}).get("derive"))
    # the phase inside the running item lives in process memory until the
    # item settles (sg_web/app.py live_reports); the row cannot hold it yet
    live = getattr(state, "live_reports", {}).get(job_id)
    if live is not None and told["state"] == "running":
        told["current"]["phase"] = {
            "phase": live.get("phase"),
            "type": live["type"],
            "message": live.get("message"),
            "text": live["text"],
            "at": live["at"],
            "data": live.get("data"),
            "live": True,
        }
    if wants_json(request):
        return Response(told, headers=VARIES)
    return Template(template_name="_operations_job.html", context={"job": told}, headers=VARIES)


@get("/job/{job_id:int}/items", sync_to_thread=True)
def job_items(
    state: State,
    request: Request,
    job_id: FromPath[int],
    state_filter: str | None = None,
    after: int = 0,
    limit: int = 200,
) -> Response | Template:
    """A page of one job's items by state: `?state_filter=failed&after=N`
    -- JSON to a machine, a fragment the inspector swaps in otherwise.
    Paged, never folded into the detail: a 22,000-item job is read a
    page at a time."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        try:
            told = inspecting.items(conn, job_id, state=state_filter, after=after, limit=limit)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
    finally:
        connect.close(conn)
    if wants_json(request):
        return Response(told, headers=VARIES)
    return Template(
        template_name="_operations_items.html",
        context={
            "items": told["items"],
            "next_after": told["next_after"],
            "job_id": job_id,
            "state_filter": state_filter,
        },
        headers=VARIES,
    )


@get("/events", sync_to_thread=True)
def events(state: State, after: int = 0, job: int | None = None, limit: int = 500) -> dict:
    """A page of the ledger, ascending from `after`, the whole ledger or
    one job's: the gap-fill and the "earlier" read. Every row, never a
    sample; `next_after` pages the rest."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        told = inspecting.events(conn, job_id=job, after=after, limit=limit)
    finally:
        connect.close(conn)
    told["events"] = [console.envelope(event) for event in told["events"]]
    return told


@get("/events/before", sync_to_thread=True)
def events_before(state: State, before: int = 0, job: int | None = None, limit: int = 500) -> dict:
    """The `limit` events with id < `before`, ascending: the tape's
    "earlier" button. Bounded; walks the index backwards and stops.
    No `before` is nothing earlier than the beginning: an empty page."""
    limit = max(1, min(int(limit), ledger.PAGE_MOST))
    conn = connect.connect(state.db_path, read_only=True)
    try:
        page = inspecting.events_before(conn, before, job_id=job, limit=limit)
    finally:
        connect.close(conn)
    return {"events": [console.envelope(event) for event in page], "before": before}


@post("/jobs/{kind:str}", sync_to_thread=True)
def launch(state: State, kind: FromPath[str]) -> Template:
    """Start one sweep from its button. The answer is the notice fragment;
    the job itself arrives on the activity surface through the feed, as
    every job does, so the page never grows a second list of jobs."""
    found = LAUNCHERS.get(kind)
    if found is None:
        raise NotFoundException(f"/operations/jobs/{kind}: nothing to start by that name")
    label, launcher = found
    conn = connect.connect(state.db_path)
    try:
        try:
            job_ids = launcher(state, conn)
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        told = [submitted(state, conn, job_id) for job_id in job_ids]
    finally:
        connect.close(conn)
    queued = ", ".join(f"#{job['id']}" for job in told)
    notice = f"{label}: queued {queued}" if told else f"{label}: nothing to do"
    return Template(template_name="_operations_notice.html", context={"notice": notice, "error": None}, headers=VARIES)


@dataclasses.dataclass
class RootForm:
    path: str
    kind: str = "library"


@post("/roots", sync_to_thread=True)
def add_root(state: State, data: URLEncodedBody[RootForm]) -> Template:
    """Register a media directory from the form; the roots section comes
    back re-read. Registering reads nothing -- scanning is its own button."""
    cleaned = data.path.strip()
    if not cleaned:
        raise ClientException("a root needs a path")
    conn = connect.connect(state.db_path)
    try:
        library.add_root(conn, cleaned, data.kind, time.time())
        conn.commit()
        roots = _roots(conn)
    finally:
        connect.close(conn)
    return Template(
        template_name="_operations_roots.html",
        context={"roots": roots, "notice": f"registered {cleaned}"},
        headers=VARIES,
    )


@post("/roots/{root_id:int}/scan", sync_to_thread=True)
def scan_root(state: State, root_id: FromPath[int]) -> Template:
    """Walk one root now. Synchronous like the JSON route: a walk is
    cheap, and its counts are the answer the person pressed for."""
    conn = connect.connect(state.db_path)
    try:
        row = conn.execute("SELECT path FROM root WHERE id = ?", (root_id,)).fetchone()
        if row is None:
            raise NotFoundException(f"no root {root_id}")
        result = scan.scan(conn, root_id, row[0], time.time())
        conn.commit()
        cache = str(home.thumbs_dir(pathlib.Path(state.home)))
        precache = runner.precache_after_scan(conn, time.time(), result, thumbs_dir=cache)
        if precache is not None:
            conn.commit()
            submitted(state, conn, precache)
        roots = _roots(conn)
    finally:
        connect.close(conn)
    notice = (
        f"scanned {row[0]}: {result.added} added, {result.matched} matched, {result.replaced} replaced,"
        f" {result.missing} missing, {result.ambiguous} ambiguous"
        + (f"; thumbnails queued as job #{precache}" if precache is not None else "")
    )
    return Template(template_name="_operations_roots.html", context={"roots": roots, "notice": notice}, headers=VARIES)


@dataclasses.dataclass
class SettingForm:
    value: str


@post("/settings/{key:str}", sync_to_thread=True)
def change_setting(state: State, key: FromPath[str], data: URLEncodedBody[SettingForm]) -> Template:
    """One setting, changed live, the whole vocabulary re-read. Refusals
    are the registry's (db/settings.py put)."""
    conn = connect.connect(state.db_path)
    try:
        try:
            settings.put(conn, key, data.value)
        except (KeyError, ValueError) as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        rows = settings.snapshot(conn)
    finally:
        connect.close(conn)
    return Template(
        template_name="_operations_settings.html",
        context={"settings": rows, "notice": f"{key} = {data.value}"},
        headers=VARIES,
    )


@post("/clusterings/choose", sync_to_thread=True)
def choose_primary(state: State) -> Template:
    conn = connect.connect(state.db_path)
    try:
        chosen = derived.choose_primary(conn)
        conn.commit()
        runs = pages.clusterings(conn)
    finally:
        connect.close(conn)
    return Template(
        template_name="_operations_clusterings.html",
        context={"clusterings": runs, "notice": f"primary run: {chosen}"},
        headers=VARIES,
    )


def refused(request: Request, exc: HTTPException) -> Template:
    """A refusal, rendered where the person is looking: the shell notice,
    carrying the refusal's own status. htmx swaps 4xx/5xx into
    #shell-notice by the shell's response-handling config
    (templates/base.html), so the reason lands on screen instead of in
    the console. The status is the exception's -- a 400 stays a 400
    (litestar-org/litestar@v2.24.0 litestar/router.py:96,135
    `exception_handlers`; docs/usage/exceptions.rst:95
    per_exception_handlers.py: a Router-level handler overrides the
    app's JSON one for this layer only)."""
    return Template(
        template_name="_operations_notice.html",
        context={"error": exc.detail, "notice": None},
        status_code=exc.status_code,
        headers=VARIES,
    )


def failed(request: Request, exc: Exception) -> Template:
    """An unexpected error on an operations form: logged whole, shown as
    the 500 it is, never swallowed into a blank notice."""
    _logger.error("operations request %s %s failed", request.method, request.url.path, exc_info=exc)
    return Template(
        template_name="_operations_notice.html",
        context={"error": f"internal error: {exc}", "notice": None},
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        headers=VARIES,
    )


router = Router(
    path="/operations",
    route_handlers=[
        operations_page,
        overview,
        job_inspector,
        job_items,
        events,
        events_before,
        launch,
        add_root,
        scan_root,
        change_setting,
        choose_primary,
    ],
    exception_handlers={HTTPException: refused, Exception: failed},
)
