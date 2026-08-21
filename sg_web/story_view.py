"""The story snapshot adapters: freeze one current event, read one
frozen document.

Freezing is synchronous on purpose -- "freeze the event I am looking
at right now" cannot be queued, because the event could change before
a worker reached it; it is database work measured in milliseconds and
proves its own currentness (db/stories.py). Model work never happens
here: planning and writing are later durable jobs over the frozen
input. Reading a snapshot consults history only.
"""

from __future__ import annotations

import dataclasses
import functools
import pathlib
import time

from litestar import Request, get, post
from litestar.datastructures import State
from litestar.exceptions import ClientException, NotFoundException
from litestar.response import Response

from db import connect, evolution, naming, planning, rendering, settings, stories
from sg_web import home
from sg_web.presenting import VARIES, wants_json


@dataclasses.dataclass
class FreezeRequest:
    """The body of POST /stories/snapshots: which current event."""

    event_id: int


@post("/stories/snapshots", sync_to_thread=True)
def freeze_snapshot(state: State, data: FreezeRequest) -> Response:
    conn = connect.connect(state.db_path)
    try:
        try:
            made = stories.snapshot_event(conn, data.event_id, time.time())
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
    finally:
        connect.close(conn)
    return Response(
        {"id": made.id, "sha256": made.sha256, "reused": made.reused},
        status_code=200 if made.reused else 201,
    )


@get("/stories/snapshots/{snapshot_id:int}", sync_to_thread=True)
def snapshot_document(state: State, snapshot_id: int) -> dict:
    conn = connect.connect(state.db_path, read_only=True)
    try:
        try:
            return stories.load_snapshot(conn, snapshot_id)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as corrupt:
            raise ClientException(str(corrupt), status_code=409) from corrupt
    finally:
        connect.close(conn)


@dataclasses.dataclass
class PlanRequest:
    """The body of POST /stories/plans: which frozen snapshot, under
    which planner, read by which similarity engine -- named EXACTLY
    (`lexical`, or a configured semantic provider such as `openclip` or
    `qwen`; never a default that might mean something else)."""

    snapshot_id: int
    planner: str = "generation_history"
    similarity: str = "lexical"
    settings: dict | None = None


@post("/stories/plans", sync_to_thread=True)
def plan_snapshot(state: State, data: PlanRequest) -> Response:
    """Ask for a plan. The service records the request's identity,
    reuses a finished plan or a live job for the same request, or
    queues durable work -- no weights load on this thread. 200 with a
    plan id when the answer already exists, 202 with the job otherwise."""
    import pathlib

    from db import jobs, settings
    from sg_web import home

    conn = connect.connect(state.db_path)
    try:
        try:
            weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
            engine = planning.engine_for(conn, data.similarity, weights)
            asked = planning.request_plan(conn, data.snapshot_id, data.planner, engine, data.settings, time.time())
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except stories.Corrupt as corrupt:
            raise ClientException(str(corrupt), status_code=409) from corrupt
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        told = {"request_sha256": asked.request_sha256, "plan_id": asked.plan_id, "job": None}
        if asked.job_id is not None:
            told["job"] = jobs.snapshot(conn, asked.job_id)
    finally:
        connect.close(conn)
    if asked.job_id is not None:
        _nudge(state)
    return Response(told, status_code=200 if asked.plan_id is not None else 202)


def _nudge(state: State) -> None:
    """Wake the worker if this process runs one (sg_web/app.py)."""
    from sg_web import app as web

    nudge = getattr(web, "_nudge", None)
    if nudge is not None:
        nudge(state)


@get("/stories/plans/{plan_id:int}", sync_to_thread=True)
def plan_document(state: State, plan_id: int) -> dict:
    conn = connect.connect(state.db_path, read_only=True)
    try:
        try:
            return planning.load_plan(conn, plan_id)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as corrupt:
            raise ClientException(str(corrupt), status_code=409) from corrupt
    finally:
        connect.close(conn)


@dataclasses.dataclass
class RenderRequest:
    """The body of POST /stories/renders: which plan, under which
    profile and locale. Rendering is pure code and synchronous."""

    plan_id: int
    profile: str = "memory"
    locale: str = "en"


@post("/stories/renders", sync_to_thread=True)
def render_plan(state: State, data: RenderRequest) -> Response:
    conn = connect.connect(state.db_path)
    try:
        try:
            narrator = rendering.TemplateStoryRenderer(data.profile, data.locale)
            made = rendering.render_plan(conn, data.plan_id, narrator, time.time())
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except stories.Corrupt as corrupt:
            raise ClientException(str(corrupt), status_code=409) from corrupt
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
    finally:
        connect.close(conn)
    return Response(
        {"id": made.id, "sha256": made.sha256, "reused": made.reused},
        status_code=200 if made.reused else 201,
    )


@functools.cache
def _story_env():
    """The story page's OWN template environment: StrictUndefined, so a
    missing field explodes instead of rendering "You introduced ."; and
    autoescape, because frozen evidence (file names, prompt text) is
    evidence, not trusted markup. Bundled templates are trusted, so the
    sandbox is not needed (jinja docs/api.rst: Undefined Types)."""
    import pathlib

    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    return Environment(
        loader=FileSystemLoader(str(pathlib.Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        autoescape=True,
    )


@get("/stories/renders/{render_id:int}", sync_to_thread=True)
def render_document(state: State, render_id: int, request: Request) -> Response:
    """The verified render, as JSON or laid out as HTML by Accept. The
    page interprets no claim, joins no table, calls no model: every
    hero is the FROZEN name, linked only through address resolution."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        try:
            story, members = rendering.load_render_with_members(conn, render_id)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as corrupt:
            raise ClientException(str(corrupt), status_code=409) from corrupt
        if wants_json(request):
            return Response(story, headers=VARIES)
        heroes = {}
        for section in story["sections"]:
            for ref in section["hero_refs"]:
                held = naming.by_uuid(conn, members[ref]["file_uuid"])
                heroes[ref] = {"name": members[ref]["name"], "slug": held[1] if held and held[0] == "file" else None}
    finally:
        connect.close(conn)
    page = _story_env().get_template("story.html").render(story=story, heroes=heroes, render_id=render_id)
    return Response(page, media_type="text/html", headers=VARIES)


@get("/stories/plans/{plan_id:int}/evolution", sync_to_thread=True)
def plan_evolution(state: State, plan_id: int, request: Request, space: str | None = None) -> Response:
    """The Generation Evolution Explorer: a read-only view of one plan
    (db/evolution.py) -- JSON, or the page by Accept. `space` names
    the provider whose joint space and query policy every metric is
    measured in; the first configured provider otherwise. No writes,
    no model loads, no embedding computation."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        try:
            view = evolution.load(conn, plan_id, provider=space, models_dir=weights)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except stories.Corrupt as corrupt:
            raise ClientException(str(corrupt), status_code=409) from corrupt
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
    finally:
        connect.close(conn)
    if wants_json(request):
        return Response(view, headers=VARIES)
    page = _story_env().get_template("evolution.html").render(view=view, plan_id=plan_id)
    return Response(page, media_type="text/html", headers=VARIES)
