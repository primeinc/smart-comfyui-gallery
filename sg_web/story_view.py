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
import time

from litestar import get, post
from litestar.datastructures import State
from litestar.exceptions import ClientException, NotFoundException
from litestar.response import Response

from db import connect, planning, stories


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
