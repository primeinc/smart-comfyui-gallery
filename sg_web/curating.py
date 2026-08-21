"""Bulk curation: one desired fact over an explicit, proven selection.

A bulk operation applies ONE desired state to an explicit set of media
entities selected from ONE known ResultSet answer, atomically. The
browser names the answer it selected against and the entity uuids it
selected; the server proves both against the authoritative projection
and writes all or nothing -- no partial batches, no per-file HTTP loop,
no 207-shaped ambiguity about which three of two thousand files are
now lying.

Ordering is load-bearing: the WRITE TRANSACTION opens BEFORE the proof
(BEGIN IMMEDIATE claims sqlite's one writer lane), so no other writer
can slip between "this selection belongs to this answer" and the
mutation -- validation and write see one library generation. The
routes themselves stay boring: parse the question, claim the lane,
prove the subset (db/resultset.py), call the one authored
implementation (db/authored.py *_many), commit once, and answer with
the after-state so the client can settle by answer identity exactly as
single writes do.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time

from litestar import post
from litestar.datastructures import State
from litestar.exceptions import ClientException, HTTPException, NotFoundException
from litestar.response import Response

from db import authored, connect, naming, resultset, settings
from sg_web import home
from sg_web.asking import gallery_query as _asked
from sg_web.presenting import VARIES


@dataclasses.dataclass
class BulkFlag:
    """The body of the boolean bulk routes: the answer the selection was
    made against, the selected entity uuids, and the desired fact."""

    answer: str
    items: list[str]
    value: bool


@dataclasses.dataclass
class BulkRating:
    """POST /g/selection/rating: 1..5 sets everyone, null clears."""

    answer: str
    items: list[str]
    value: int | None = None


def _applied(state: State, query, data, write) -> Response:
    """Claim the writer lane, prove the selection, write, commit once --
    then describe the SAME question again so the client settles on the
    (currency, answer) pair."""
    conn = connect.connect(state.db_path)
    try:
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        conn.execute("BEGIN IMMEDIATE")
        try:
            ids = resultset.subset(
                conn,
                weights,
                query,
                time.time(),
                actor_id=state.actor_id,
                expect_answer=data.answer,
                entity_uuids=data.items,
            )
            write(conn, ids)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        after = resultset.describe(conn, weights, query, time.time(), actor_id=state.actor_id)
        return Response(
            {
                "changed": len(ids),
                "after": {"answer": after["answer"], "currency": after["currency"], "total": after["total"]},
            },
            headers=VARIES,
        )
    except resultset.AnswerChanged as moved:
        raise HTTPException(status_code=409, detail=str(moved)) from moved
    except resultset.UnevaluatedCollection as refused:
        raise ClientException(str(refused)) from refused
    except LookupError as missing:
        raise NotFoundException(str(missing)) from missing
    except ValueError as refused:
        raise ClientException(str(refused)) from refused
    finally:
        connect.close(conn)


@post("/g/selection/favorite", sync_to_thread=True)
def bulk_favorite(
    state: State,
    data: BulkFlag,
    folder: str | None = None,
    album: str | None = None,
    person: str | None = None,
    kind: str | None = None,
    favorite: str | None = None,
    rating_min: int | None = None,
    q: str | None = None,
    sort: str | None = None,
    size: int | None = None,
) -> Response:
    query = _asked(folder, album, kind, q, sort, size, person=person, favorite=favorite, rating_min=rating_min)
    return _applied(
        state,
        query,
        data,
        lambda conn, ids: authored.set_favorite_many(conn, ids, state.actor_id, data.value, time.time()),
    )


@post("/g/selection/rating", sync_to_thread=True)
def bulk_rating(
    state: State,
    data: BulkRating,
    folder: str | None = None,
    album: str | None = None,
    person: str | None = None,
    kind: str | None = None,
    favorite: str | None = None,
    rating_min: int | None = None,
    q: str | None = None,
    sort: str | None = None,
    size: int | None = None,
) -> Response:
    query = _asked(folder, album, kind, q, sort, size, person=person, favorite=favorite, rating_min=rating_min)
    return _applied(
        state,
        query,
        data,
        lambda conn, ids: authored.set_rating_many(conn, ids, state.actor_id, data.value, time.time()),
    )


@post("/g/selection/collections/{collection:str}", sync_to_thread=True)
def bulk_membership(
    state: State,
    collection: str,
    data: BulkFlag,
    folder: str | None = None,
    album: str | None = None,
    person: str | None = None,
    kind: str | None = None,
    favorite: str | None = None,
    rating_min: int | None = None,
    q: str | None = None,
    sort: str | None = None,
    size: int | None = None,
) -> Response:
    query = _asked(folder, album, kind, q, sort, size, person=person, favorite=favorite, rating_min=rating_min)

    def write(conn, ids):
        found = naming.resolve(conn, "collection", collection)
        if found is None:
            raise LookupError(f"no collection at /t/{collection}")
        authored.set_collection_membership_many(conn, found[0], ids, data.value, time.time())

    return _applied(state, query, data, write)
