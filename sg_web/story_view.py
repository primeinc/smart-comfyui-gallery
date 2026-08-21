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

from db import connect, stories


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
