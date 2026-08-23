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
import json
import pathlib
import time
import urllib.parse

from litestar import Request, get, post
from litestar.datastructures import State
from litestar.exceptions import ClientException, NotFoundException
from litestar.params import FromPath, FromQuery
from litestar.response import Redirect, Response, Template

from db import connect, derived, evolution, facets, naming, pages, planning, rendering, settings, stories
from sg_web import home, submitting
from sg_web.presenting import VARIES, presented_page, wants_json


def _window(subject: dict) -> str | None:
    """The timeline's hour window around the frozen session, in the
    domain the evidence claims it in; None when the subject holds no
    interval."""
    when = subject.get("time") or {}
    held = when.get("local") or when.get("instant")
    if not held:
        return None
    start = int(held[0] // 3600) * 3600
    end = int(held[-1] // 3600) * 3600 + 3600
    return "/timeline?" + urllib.parse.urlencode({"bin": "hour", "start": start, "end": end})


@get("/stories", sync_to_thread=True)
def stories_index(state: State, request: Request, kind: FromQuery[str | None] = None) -> Template | Response:
    """Every story told, newest first -- the shelf the timeline's
    buttons fill -- or, with `?kind=`, those of one session kind. Each
    entry is its title and dek as rendered, its heroes, its subject
    kind, its profile, and doors to the render and the plan's
    evolution."""
    if kind is not None and kind not in stories.EVENT_KINDS:
        raise ClientException(f"kind is one of {', '.join(stories.EVENT_KINDS)}, not {kind!r}")
    conn = connect.connect(state.db_path, read_only=True)
    try:
        kinds = pages.story_kinds(conn)
        told = []
        for row in pages.stories(conn, kind=kind):
            render_id, plan_id, profile, created_at, document, planner, event_kind, snapshot_id, frozen = row
            words = json.loads(document)
            told.append(
                {
                    "id": render_id,
                    "plan_id": plan_id,
                    "snapshot_id": snapshot_id,
                    "profile": profile,
                    "planner": planner,
                    "kind": event_kind,
                    "told_at": created_at,
                    "title": words.get("title", ""),
                    "dek": words.get("dek", ""),
                    "members": len((words.get("support") or {}).get("member_refs") or []),
                    "heroes": rendering.heroes(conn, words, json.loads(frozen)),
                    "href": f"/stories/renders/{render_id}",
                    "evolution": f"/stories/plans/{plan_id}/evolution",
                }
            )
    finally:
        connect.close(conn)
    return presented_page(request, told, page="stories.html", context={"stories": told, "kind": kind, "kinds": kinds})


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
def snapshot_document(state: State, snapshot_id: FromPath[int]) -> dict:
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
            # Announced and woken like every other submit: a plan job is
            # a job on the feed from the moment it is queued.
            told["job"] = submitting.submitted(state, conn, asked.job_id)
    finally:
        connect.close(conn)
    return Response(told, status_code=200 if asked.plan_id is not None else 202)


@get("/stories/plans/{plan_id:int}", sync_to_thread=True)
def plan_document(state: State, request: Request, plan_id: FromPath[int]) -> dict | Redirect:
    """The plan document to a machine; a browser is sent to the plan's
    page, which is its evolution view -- a person never lands on JSON."""
    if not wants_json(request):
        return Redirect(path=f"/stories/plans/{plan_id}/evolution", status_code=302)
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


@get("/stories/renders/{render_id:int}", sync_to_thread=True)
def render_document(state: State, render_id: FromPath[int], request: Request) -> Response | Template:
    """The verified render, as JSON or laid out as HTML by Accept. The
    page interprets no claim, joins no table, calls no model: every
    hero and every member is the FROZEN name, linked to its picture only
    through address resolution -- a member whose file has since left the
    library keeps its name and loses its link."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        try:
            story, members, plan_id, subject = rendering.load_render_with_members(conn, render_id)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as corrupt:
            raise ClientException(str(corrupt), status_code=409) from corrupt
        if wants_json(request):
            return Response(story, headers=VARIES)
        addressed = {}
        ids: dict[str, int] = {}
        for ref, member in members.items():
            held = naming.by_uuid(conn, member["file_uuid"])
            slug = held[1] if held and held[0] == "file" else None
            addressed[ref] = {
                "name": member["name"],
                "slug": slug,
                "page": f"/i/{slug}" if slug else None,
                "thumbnail": f"/thumb/{slug}" if slug else None,
                "kind": member.get("media_kind"),
                "said": None,
            }
            found = naming.resolve(conn, "file", slug) if slug else None
            if found is not None:
                ids[ref] = found[0]
        # What a model says about a hero TODAY -- live, by address, never
        # part of the frozen story: the evidence the snapshot froze is the
        # story's; this is a courtesy line the page labels as its own.
        said = derived.said_first(conn, ids.values(), prefer=settings.value(conn, "caption_model"))
        for ref, file_id in ids.items():
            addressed[ref]["said"] = said.get(file_id)
    finally:
        connect.close(conn)
    # Rendered by the application's one engine (sg_web/app.py
    # _template_engine: StrictUndefined, so a missing frozen field explodes
    # instead of rendering "You introduced ."; autoescape, because frozen
    # evidence is evidence, not trusted markup).
    return Template(
        template_name="story.html",
        context={
            "story": story,
            "members": addressed,
            "render_id": render_id,
            "plan_id": plan_id,
            "profiles": rendering.PROFILES,
            "session_window": _window(subject),
        },
        headers=VARIES,
    )


@get("/stories/plans/{plan_id:int}/evolution", sync_to_thread=True)
def plan_evolution(
    state: State, plan_id: FromPath[int], request: Request, space: FromQuery[str | None] = None
) -> Response | Template:
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
        render_id = rendering.latest_render_id(conn, plan_id)
    finally:
        connect.close(conn)
    _addressed(view)
    view["doors"]["story"] = f"/stories/renders/{render_id}" if render_id is not None else None
    if wants_json(request):
        return Response(view, headers=VARIES)
    return Template(template_name="evolution.html", context={"view": view, "plan_id": plan_id}, headers=VARIES)


def _addressed(view: dict) -> None:
    """Identities into addresses -- the web adapter's job, never the
    database module's: a member's slug becomes its thumbnail and page,
    the session's day the gallery's day-facet door, a prompt row its
    neighbours route."""
    for member in view["members"]:
        slug = member["media"].get("slug")
        member["media"]["thumbnail"] = f"/thumb/{slug}" if slug else None
        member["media"]["page"] = f"/i/{slug}" if slug else None
    day = view["identities"].get("local_day")
    view["doors"] = {
        "gallery_day": (
            "/g?" + urllib.parse.urlencode([("f", facets.spell(facets.facet("context.local_day", "eq", day)))])
            if day
            else None
        ),
        "search": "/search?q=",
        "neighbours": "/prompts/{id}/neighbours",
    }
