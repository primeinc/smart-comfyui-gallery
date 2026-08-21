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
    finally:
        connect.close(conn)


@dataclasses.dataclass
class PlanRequest:
    """The body of POST /stories/plans: which frozen snapshot, under
    which planner and which similarity engine."""

    snapshot_id: int
    planner: str = "generation_history"
    similarity: str = "openclip"
    settings: dict | None = None


def _similarity(conn, state: State, named: str):
    """The similarity Adapter by name. `openclip` is the configured
    semantic text encoder, loaded OFFLINE -- a plan request never begins
    downloading weights; `lexical` is the model-free deterministic
    engine. The planner receives the engine, never the connection."""
    if named == "lexical":
        return planning.LexicalPromptSimilarity()
    if named == "openclip":
        import pathlib

        from db import retrieval, settings
        from sg_web import home
        from vision import semantic
        from vision.semantic import openclip

        provider, model, configured = retrieval.choices(conn)[0]
        models_dir = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        checkpoint = semantic.pin(provider, models_dir, model, configured)
        encoder = semantic.encoder(provider, models_dir, model, checkpoint, offline=True)
        return planning.ClipPromptSimilarity(encoder, provider, checkpoint, openclip.openclip_version())
    raise ValueError(f"no similarity engine named {named!r}; one of lexical, openclip")


def _planner_for(named: str):
    maker = planning.PLANNERS.get(named)
    if maker is None:
        raise ValueError(f"no planner named {named!r}; one of {', '.join(sorted(planning.PLANNERS))}")
    return maker


@post("/stories/plans", sync_to_thread=True)
def plan_snapshot(state: State, data: PlanRequest) -> Response:
    conn = connect.connect(state.db_path)
    try:
        try:
            planner = _planner_for(data.planner)(_similarity(conn, state, data.similarity), data.settings)
            made = planning.plan_snapshot(conn, data.snapshot_id, planner, time.time())
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


@get("/stories/plans/{plan_id:int}", sync_to_thread=True)
def plan_document(state: State, plan_id: int) -> dict:
    conn = connect.connect(state.db_path, read_only=True)
    try:
        try:
            return planning.load_plan(conn, plan_id)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
    finally:
        connect.close(conn)
