"""The runtime's own surface: roots, sweeps, the worker, clustering.

Everything here already exists as a machine route in sg_web/app.py --
POST /roots, POST /roots/{id}/scan, POST /jobs/*, POST /settings/{key},
POST /clusterings/choose. Those keep their JSON shape for machines. This
module is the BROWSER's way in: one page, forms that post url-encoded
(what an htmx form sends without help), fragments back in place. It is
registered as one Litestar Router under /operations, so every operational
page and form shares that prefix and whatever policy the layer grows
(litestar-org/litestar@v2.24.0 docs/usage/routing/overview.rst "Routers";
litestar/router.py Router.__init__ for the layered kwargs).

Nothing operational is offered anywhere else: the gallery header asks
questions about media, this page runs the library.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
from collections.abc import Callable

from litestar import Router, get, post
from litestar.datastructures import State
from litestar.exceptions import ClientException, NotFoundException
from litestar.params import FromPath, URLEncodedBody
from litestar.response import Template

from db import connect, derived, library, pages, prompts, runner, scan, settings
from sg_web import home
from sg_web.presenting import VARIES
from sg_web.submitting import submitted

Launcher = Callable[[State, object], list[int]]


def _weights(state: State, conn) -> str:
    return str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))


def _ingest(state: State, conn) -> list[int]:
    return [runner.submit_ingest(conn, time.time())]


def _verify(state: State, conn) -> list[int]:
    return [runner.submit_verify(conn, time.time())]


def _phash(state: State, conn) -> list[int]:
    return [runner.submit_phash(conn, time.time())]


def _dupes(state: State, conn) -> list[int]:
    return [runner.submit_dupes(conn, time.time())]


def _embed(state: State, conn) -> list[int]:
    return runner.submit_embed(conn, time.time(), models_dir=_weights(state, conn))


def _embed_prompts(state: State, conn) -> list[int]:
    return prompts.submit_embed(conn, time.time(), models_dir=_weights(state, conn))


def _faces(state: State, conn) -> list[int]:
    cache = str(home.thumbs_dir(pathlib.Path(state.home))) if settings.flag(conn, "thumbnail_precache") else None
    return [runner.submit_faces(conn, time.time(), models_dir=_weights(state, conn), thumbs_dir=cache)]


def _cluster(state: State, conn) -> list[int]:
    return [runner.submit_cluster(conn, time.time())]


def _context(state: State, conn) -> list[int]:
    return [runner.submit_context(conn, time.time())]


def _events(state: State, conn) -> list[int]:
    return [runner.submit_events(conn, time.time())]


#: What the page can start, in the order a library is usually built:
#: find files, read them, fingerprint, group copies, embed, detect faces,
#: cluster, interpret time, group events. Each launcher returns the job
#: ids it queued -- the same db/runner.py entry points the JSON routes use.
LAUNCHERS: dict[str, tuple[str, Launcher]] = {
    "ingest": ("read every file's metadata", _ingest),
    "verify": ("verify every file's bytes", _verify),
    "phash": ("fingerprint every picture", _phash),
    "dupes": ("group perceptual copies", _dupes),
    "embed": ("embed every picture for search", _embed),
    "embed_prompts": ("embed every prompt", _embed_prompts),
    "faces": ("detect faces", _faces),
    "cluster": ("cluster faces into people", _cluster),
    "context": ("interpret every file's time and place", _context),
    "events": ("propose events", _events),
}


def _roots(conn) -> list[dict]:
    """Every root with its live reachability -- the probe only, no write
    (db/library.py probe_roots); the JSON /roots route is what records."""
    return [{"id": root_id, "path": path, "online": online} for root_id, path, online in library.probe_roots(conn)]


def _page_context(state: State) -> dict:
    conn = connect.connect(state.db_path, read_only=True)
    try:
        return {
            "roots": _roots(conn),
            "settings": settings.snapshot(conn),
            "clusterings": pages.clusterings(conn),
            "launchers": [{"kind": kind, "label": label} for kind, (label, _) in LAUNCHERS.items()],
            "notice": None,
        }
    finally:
        connect.close(conn)


@get("/", sync_to_thread=True)
def operations_page(state: State) -> Template:
    return Template(template_name="operations.html", context=_page_context(state), headers=VARIES)


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
    return Template(template_name="_operations_notice.html", context={"notice": notice}, headers=VARIES)


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
        roots = _roots(conn)
    finally:
        connect.close(conn)
    notice = (
        f"scanned {row[0]}: {result.added} added, {result.matched} matched, {result.replaced} replaced,"
        f" {result.missing} missing, {result.ambiguous} ambiguous"
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


router = Router(
    path="/operations",
    route_handlers=[operations_page, launch, add_root, scan_root, change_setting, choose_primary],
)
