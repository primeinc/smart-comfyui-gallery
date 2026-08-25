"""The application over the schema: every page a query, every sweep a job.

Addresses are entity slugs -- never paths, never raw ids -- and nothing
expensive starts by itself: a sweep is a `job` row somebody POSTs into
existence, drained by the in-process worker (sg_web/worker.py),
cancellable and resumable because the row is the truth.

Realtime first: progress is pushed, not polled. The worker publishes
every observable change onto the "jobs" channel and /ws/jobs streams it;
the snapshot routes exist for rendering from cold, and no client has a
reason to poll them in a loop.

Handlers are synchronous on purpose: sqlite is synchronous, and Litestar
runs sync handlers on its thread pool when told so
(litestar-org/litestar@64cd7da docs/topics/sync-vs-async.rst). Each request
opens its own connection, which is what makes that safe -- sqlite3
connections refuse cross-thread use, and the pool gives no thread pinning.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import sqlite3
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any, Literal

from litestar import Litestar, Request, delete, get, post, route, websocket
from litestar.channels import ChannelsPlugin
from litestar.channels.backends.memory import MemoryChannelsBackend
from litestar.connection import WebSocket
from litestar.datastructures import State
from litestar.di import NamedDependency
from litestar.exceptions import ClientException, HTTPException, NotFoundException
from litestar.exceptions.responses import create_debug_response, create_exception_response
from litestar.logging import LoggingConfig
from litestar.params import FromPath, FromQuery
from litestar.plugins import InitPlugin
from litestar.plugins.jinja import JinjaTemplateEngine
from litestar.response import File, Redirect, Response, Stream, Template
from litestar.static_files import create_static_files_router
from litestar.template import TemplateConfig

from db import (
    authored,
    collections,
    connect,
    derived,
    detect,
    jobs,
    ledger,
    library,
    migrate,
    naming,
    oriented,
    pages,
    prompts,
    runner,
    sample,
    scan,
    settings,
    views,
)
from sg_web import (
    activity,
    artifact_view,
    collection_authoring,
    collection_view,
    console,
    curating,
    folder_view,
    gallery,
    home,
    media,
    media_authored,
    media_view,
    operations,
    person_view,
    place_view,
    story_view,
    timeline_view,
    wire,
)
from sg_web import worker as worker_module
from sg_web.presenting import VARIES, presented_page, wants_json
from sg_web.submitting import announce as _announce
from sg_web.submitting import nudge as _nudge
from sg_web.submitting import submitted as _submitted
from sg_web.wire import Wire

_logger = logging.getLogger(__name__)


def _connect(db_path: str) -> sqlite3.Connection:
    """Through db/connect.py, like every consumer: foreign keys, IMMEDIATE
    writers, busy_timeout and the cache are per-connection settings a raw
    sqlite3.connect silently runs without."""
    return connect.connect(db_path)


def _rows(cursor_rows, columns) -> list[dict]:
    return [dict(zip(columns, row, strict=True)) for row in cursor_rows]


@get("/health", sync_to_thread=False)
def health() -> str:
    return "ok"


def _resolved(conn, kind: str, slug: str, where: str) -> tuple[int, str | None]:
    """`(entity_id, live_slug_when_retired)` for an address, 404ing what
    does not resolve. The caller shapes its own 301 from the live slug so
    each route redirects within its own prefix."""
    found = naming.resolve(conn, kind, slug)
    if found is None:
        raise NotFoundException(f"no {kind} at {where}/{slug}")
    entity_id, is_current = found
    if not is_current:
        live = naming.entity_slug(conn, entity_id)
        if live is not None:
            return entity_id, live[1]
    return entity_id, None


@get("/", sync_to_thread=True)
def front(state: State, request: Request) -> Response | Redirect:
    """The front link. A browser lands in the gallery -- /g owns the
    canonical question state, and an entrance pointing at JSON was the
    one page of this application still shaped for its developers. A
    machine gets the compact library summary with a newest strip; the
    media answers themselves are the ResultSet's."""
    if not wants_json(request):
        return Redirect(path="/g", status_code=302)
    conn = _connect(state.db_path)
    try:
        files, folders, people, collections_held, artifacts = pages.library_summary(conn)
        return Response(
            {
                "files": files,
                "folders": folders,
                "people": people,
                "collections": collections_held,
                "artifacts": artifacts,
                "newest": _rows(pages.newest(conn, 12), ("slug", "name", "mtime")),
            },
            headers=VARIES,
        )
    finally:
        connect.close(conn)


# The media page lives in sg_web/media_view.py, the folder page in
# sg_web/folder_view.py, the artifact pages in sg_web/artifact_view.py:
# one address each, negotiated per caller. The shelf indexes below are
# aggregates -- "which artifacts are commonly used?" -- not media
# answers; every media answer is the ResultSet's.


# The album index and page live in sg_web/collection_view.py, and every
# lifecycle write in sg_web/collection_authoring.py: one address per
# collection, one write adapter over db/collections.py. The legacy
# membership routes below stay as compatibility adapters.


class AlbumEntry(Wire):
    """The body of the album membership routes: a file, by its address."""

    file: str


def _album_membership(state: State, slug: str, data: AlbumEntry, *, adding: bool) -> dict:
    conn = _connect(state.db_path)
    try:
        collection_id, live_album = _resolved(conn, "collection", slug, "/t")
        # The file resolves at its own address: a 404 that says
        # "no file at /t/keepers/add/nope" names a place nothing lives at.
        file_id, live_file = _resolved(conn, "file", data.file, "/i")
        try:
            # The SAME membership Implementation the /i desired-state
            # routes use -- these stay as compatibility adapters, never a
            # second write path.
            collections.set_membership(conn, collection_id, file_id, adding, time.time())
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        # Present members only -- the same number GET /albums answers with,
        # so the two routes cannot drift apart; and the LIVE slugs, so a
        # caller holding a retired address learns the current one.
        return {
            "slug": live_album or slug,
            "file": live_file or data.file,
            "pictures": pages.album_present(conn, collection_id),
        }
    finally:
        connect.close(conn)


@post("/t/{slug:str}/add", sync_to_thread=True)
def album_add(state: State, slug: FromPath[str], data: AlbumEntry) -> dict:
    return _album_membership(state, slug, data, adding=True)


@post("/t/{slug:str}/remove", sync_to_thread=True)
def album_remove(state: State, slug: FromPath[str], data: AlbumEntry) -> dict:
    return _album_membership(state, slug, data, adding=False)


# The People index, person page/drawer and naming live in
# sg_web/person_view.py: one address per person, presented as the full
# profile, the drawer over the mounted index, or the PersonView itself.


@get("/clusterings", sync_to_thread=True)
def clusterings(state: State) -> list[dict]:
    """Every clustering run held side by side, primary first."""
    conn = _connect(state.db_path)
    try:
        return pages.clusterings(conn)
    finally:
        connect.close(conn)


@get("/ways", sync_to_thread=True)
def ways(state: State) -> list[dict]:
    """What the library can be searched by, generated from what it holds."""
    conn = _connect(state.db_path)
    try:
        return _rows(pages.ways(conn), ("source", "key", "value_kind", "occurrences"))
    finally:
        connect.close(conn)


#: The job vocabularies live with the table that owns them (db/jobs.py),
#: so the two seams that spell them -- this one and the operations
#: console -- cannot come to different conclusions about what a job is.
JobState = jobs.JobState
JobKind = jobs.JobKind


class JobListed(Wire):
    """One job as a list carries it -- db/jobs.py `active` and `recent`,
    which share a column list so both are this shape."""

    id: int
    kind: JobKind
    state: JobState
    cancel_requested: bool
    total: int | None
    done_count: int
    created_at: float
    finished_at: float | None
    derive: str | None


class JobSnapshot(JobListed):
    """Everything a client renders one job from cold: the listed columns
    and the rest of the row. `failed_count` is here because "done, with
    three files unreadable" and "done" are different outcomes a page must
    show without a worker's turn summary to read."""

    attempt: int
    error: str | None
    started_at: float | None
    failed_count: int


class JobsSnapshotFrame(Wire):
    """Every unsettled job, read from the rows, sent first on connect.

    A client renders from this and applies deltas onto it, so it can never
    show a state the rows did not hold. The channel stores nothing: a
    reconnect starts from the rows again.
    """

    type: Literal["snapshot"]
    jobs: list[JobListed]


class JobDeltaFrame(Wire):
    """One observable change to one job, as the worker and the request
    routes publish it (db/runner.py spoke, sg_web/submitting.py announce).

    `cancel_requested` rides every delta so a subscriber that saw the
    cancel asked for keeps seeing it asked for until the job settles. The
    row stores it as 0 or 1 and the contract promises a boolean; the
    translating happens where the frame is built, once, for both
    publishers.
    """

    type: Literal["delta"]
    job: int
    kind: JobKind
    state: JobState
    done: int
    total: int | None
    cancel_requested: bool
    derive: str | None


#: What arrives on /ws/jobs. Discriminated on `type` -- not `frame`,
#: which is what /ws/events uses: there an event row already HAS a `type`
#: column and the discriminant needed another name. A job delta does not,
#: so the plain word is free here. Narrowing works the same, and a browser
#: narrows to the arm it is handling and cannot read a job list off a
#: delta.
#:
#: Not an OpenAPI path -- a socket has none -- so this is carried into the
#: contract by job_frames() below. The browser's type is generated from
#: that, never written twice.
JobFrame = JobsSnapshotFrame | JobDeltaFrame


@get("/ws/jobs/frames", sync_to_thread=False)
def job_frames() -> JobFrame:
    """Every frame /ws/jobs sends, carried into the document.

    The same seam as console.socket_frames(): a route is the only way a
    shape with no path reaches `components.schemas`, because the document
    assigns that dict from what the routes generated
    (litestar-org/litestar@v2.24.0 litestar/_openapi/plugin.py:90). It
    answers an empty snapshot rather than raising -- a route a generator
    reads is still a route somebody can request.
    """
    return JobsSnapshotFrame(type="snapshot", jobs=[])


def _job_delta(told: Mapping[str, Any]) -> JobDeltaFrame:
    """One channel payload as the contract states it."""
    return JobDeltaFrame(
        type="delta",
        job=told["job"],
        kind=told["kind"],
        state=told["state"],
        done=told["done"],
        total=told["total"],
        cancel_requested=bool(told["cancel_requested"]),
        derive=told.get("derive"),
    )


def _job_listed(row: Mapping[str, Any]) -> JobListed:
    """A `job` row as the wire carries it. `cancel_requested` is the one
    translation: SQLite stores the flag as 0 or 1, the contract promises a
    boolean, and strict mode will not pretend those are the same."""
    return JobListed(
        id=row["id"],
        kind=row["kind"],
        state=row["state"],
        cancel_requested=bool(row["cancel_requested"]),
        total=row["total"],
        done_count=row["done_count"],
        created_at=row["created_at"],
        finished_at=row["finished_at"],
        derive=row["derive"],
    )


@get("/jobs", sync_to_thread=True)
def active_jobs(state: State) -> list[JobListed]:
    conn = _connect(state.db_path)
    try:
        return [_job_listed(row) for row in jobs.active(conn)]
    finally:
        connect.close(conn)


@get("/jobs/{job_id:int}", sync_to_thread=True)
def job_snapshot(state: State, job_id: FromPath[int]) -> JobSnapshot:
    """The persisted snapshot -- what a client renders from cold. A page
    reload or a dropped socket recovers by reading this, never a replay."""
    conn = _connect(state.db_path)
    try:
        try:
            row = jobs.snapshot(conn, job_id)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        return JobSnapshot(
            **_job_listed(row).model_dump(),
            attempt=row["attempt"],
            error=row["error"],
            started_at=row["started_at"],
            failed_count=row["failed_count"],
        )
    finally:
        connect.close(conn)


@post("/jobs/verify", sync_to_thread=True)
def submit_verify(state: State) -> dict:
    """Ask for an integrity sweep. The row queues it; the worker drains it."""
    conn = _connect(state.db_path)
    try:
        job_id = runner.submit_verify(conn, time.time())
        conn.commit()
        return _submitted(state, conn, job_id)
    finally:
        connect.close(conn)


class Everything(Wire):
    """The body of a missing-only sweep's route: redo all of it, or not.
    Nothing else -- where weights live is the `models_dir` setting, never
    a request's word."""

    everything: bool = False


@post("/jobs/faces", sync_to_thread=True)
def submit_faces(state: State, data: Everything | None = None) -> dict | Response:
    """Ask for face detection over every picture no detector has looked
    at for its current bytes -- `{"everything": true}` for all of them
    again -- with the models named by the settings. 204 when nothing is
    left."""
    conn = _connect(state.db_path)
    try:
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        cache = str(home.thumbs_dir(pathlib.Path(state.home))) if settings.flag(conn, "thumbnail_precache") else None
        job_id = runner.submit_faces(
            conn, time.time(), models_dir=weights, thumbs_dir=cache, everything=bool(data and data.everything)
        )
        if job_id is None:
            return Response(content=None, status_code=204)
        conn.commit()
        return _submitted(state, conn, job_id)
    finally:
        connect.close(conn)


@post("/jobs/annotate", sync_to_thread=True)
def submit_annotate(state: State, data: Everything | None = None) -> dict | Response:
    """Ask for a caption on every picture that lacks one from the
    configured model -- `{"everything": true}` for all of them again.
    204 when nothing is left to caption."""
    conn = _connect(state.db_path)
    try:
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        try:
            job_id = runner.submit_annotate(
                conn, time.time(), models_dir=weights, everything=bool(data and data.everything)
            )
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        if job_id is None:
            return Response(content=None, status_code=204)
        conn.commit()
        return _submitted(state, conn, job_id)
    finally:
        connect.close(conn)


class CatchUpQueued(Wire):
    """What one ask queued, and in what order.

    Named rather than answered as a bare dict, because the browser is
    typed against this: `steps` is the order the runner will take them
    in, which is the whole point of the collection and the one thing a
    caller cannot re-derive from the job rows alone until they settle.
    """

    #: the name every step shares
    collection: str
    #: job ids, in the order each gates the next
    steps: list[int]
    #: the first step, snapshotted -- what a single submit would answer.
    #: None only when `steps` is empty, which needs every submitter to
    #: decline at once.
    first: JobSnapshot | None = None


class RememberedView(Wire):
    """One question somebody asked to be reminded of."""

    id: int
    name: str
    #: the canonical query string, without a page -- open it at /g?<qs>
    qs: str
    created_at: float
    last_used_at: float | None


class AskedView(Wire):
    """The body of POST /views: a name, and the question's own spelling.

    The spelling, never a rule. A saved view is not a collection and has
    no membership to define -- it is the address of a question, and the
    address is what heals a retired slug to the live one as it is
    navigated (db/resultset.py `canonical`).
    """

    name: str
    qs: str


@get("/views", sync_to_thread=True)
def saved_views(state: State) -> list[RememberedView]:
    """Every remembered question, most recently USED first."""
    conn = _connect(state.db_path)
    try:
        return [RememberedView(**row) for row in views.all_of(conn)]
    finally:
        connect.close(conn)


@post("/views", sync_to_thread=True)
def remember_view(state: State, data: AskedView) -> Response[RememberedView]:
    """Remember this question under this name.

    The third thing people mean by "save this", and the one that had
    nowhere to go: an album is what somebody put together, a smart
    collection is a dynamic grouping that behaves like one, and this has
    no members, no colour and nothing filed under it. Making one a
    collection put a thing that is not an album into somebody's album
    list, once per good question they had.
    """
    conn = _connect(state.db_path)
    try:
        try:
            made = views.remember(conn, data.name, data.qs, time.time())
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        held = next(one for one in views.all_of(conn) if one["id"] == made)
    finally:
        connect.close(conn)
    return Response(RememberedView(**held), headers=VARIES)


@post("/views/{view_id:int}/opened", sync_to_thread=True)
def view_opened(state: State, view_id: FromPath[int]) -> Response[None]:
    """Somebody went back to this one, so it sorts higher next time."""
    conn = _connect(state.db_path)
    try:
        views.opened(conn, view_id, time.time())
        conn.commit()
    finally:
        connect.close(conn)
    return Response(content=None, status_code=204)


@post("/views/{view_id:int}/forget", sync_to_thread=True)
def forget_view(state: State, view_id: FromPath[int]) -> Response[None]:
    """Stop remembering it. 404 when there was nothing to forget, so a
    second press is a refusal rather than a quiet success."""
    conn = _connect(state.db_path)
    try:
        gone = views.forget(conn, view_id)
        conn.commit()
    finally:
        connect.close(conn)
    if not gone:
        raise NotFoundException(f"no saved view {view_id}")
    return Response(content=None, status_code=204)


@post("/jobs/catch-up", sync_to_thread=True)
def submit_catch_up(state: State) -> CatchUpQueued:
    """Bring the library up to date, in one ask, in the right order.

    The eight buttons pressed in the sequence only this application knew
    -- and the sequence is not advice: `cluster_faces` over an unembedded
    library settles `done` having clustered nothing, so pressing them out
    of order does not look like a mistake, it looks like a library with
    no people in it.

    Every step is gated on the one before it, so all of them queue now
    and the runner takes them in turn.

    `steps` is always a list, never a 204 a caller has to special-case.
    It is rarely empty even over an empty library: some steps cannot know
    in advance that they have nothing to do -- `events` and
    `cluster_faces` reach that conclusion by running -- so they queue and
    settle `done` having done nothing. The steps that CAN tell (ingest,
    embed, annotate, faces) are simply absent, and the chain closes over
    the hole.
    """
    conn = _connect(state.db_path)
    try:
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        cache = str(home.thumbs_dir(pathlib.Path(state.home))) if settings.flag(conn, "thumbnail_precache") else None
        try:
            queued = runner.catch_up(conn, time.time(), models_dir=weights, thumbs_dir=cache)
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        if not queued:
            return CatchUpQueued(collection=runner.CATCH_UP, steps=[])
        conn.commit()
        first = _submitted(state, conn, queued[0])
        # `cancel_requested` is stored as 0/1 -- STRICT has no boolean --
        # and every other submit route hands the raw dict back, so nothing
        # ever validated it against the model that says `bool`. Naming the
        # answer is what made the disagreement visible.
        return CatchUpQueued(
            collection=runner.CATCH_UP,
            steps=queued,
            first=JobSnapshot(**(first | {"cancel_requested": bool(first["cancel_requested"])})),
        )
    finally:
        connect.close(conn)


@post("/jobs/thumbs", sync_to_thread=True)
def submit_thumbs(state: State) -> dict | Response:
    """Ask for every missing grid thumb and lightbox preview to be
    rendered ahead of a view (db/runner.py submit_thumbs). 204 when the
    cache already holds them all."""
    conn = _connect(state.db_path)
    try:
        job_id = runner.submit_thumbs(conn, time.time(), thumbs_dir=str(home.thumbs_dir(pathlib.Path(state.home))))
        if job_id is None:
            return Response(content=None, status_code=204)
        conn.commit()
        return _submitted(state, conn, job_id)
    finally:
        connect.close(conn)


@post("/jobs/phash", sync_to_thread=True)
def submit_phash(state: State, everything: FromQuery[bool] = False) -> dict | Response:
    """Ask for the perceptual fingerprint of every present picture still
    without one -- `?everything=true` for all of them again -- the
    identity that survives copies of copies (db/runner.py submit_phash).
    204 when every picture is fingerprinted."""
    conn = _connect(state.db_path)
    try:
        job_id = runner.submit_phash(conn, time.time(), everything=everything)
        if job_id is None:
            return Response(content=None, status_code=204)
        conn.commit()
        return _submitted(state, conn, job_id)
    finally:
        connect.close(conn)


@post("/jobs/embed", sync_to_thread=True)
def submit_embed(state: State, everything: FromQuery[bool] = False) -> list[dict]:
    """Ask for the joint image/text embedding of every present picture
    still without a current vector -- `?everything=true` for all of them
    again -- the representation /search answers from (db/runner.py
    submit_embed). One job per participating space, so one model's
    failure never costs another's progress; the response carries one
    snapshot per job and is empty when every space is current. The
    first run downloads the model weights into the run's models
    directory; a bad `semantic_model` setting is refused here, not
    queued."""
    conn = _connect(state.db_path)
    try:
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        try:
            job_ids = runner.submit_embed(conn, time.time(), models_dir=weights, everything=everything)
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        return [_submitted(state, conn, job_id) for job_id in job_ids]
    finally:
        connect.close(conn)


@post("/jobs/embed_prompts", sync_to_thread=True)
def submit_embed_prompts(state: State) -> list[dict]:
    """Ask for every role-playing prompt's vector under every participating
    space (db/prompts.py submit_embed) -- the reusable substrate story
    planning, prompt neighbours and prompt clustering read from. One job
    per space; already-current prompts are not queued."""
    conn = _connect(state.db_path)
    try:
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        try:
            job_ids = prompts.submit_embed(conn, time.time(), models_dir=weights)
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        return [_submitted(state, conn, job_id) for job_id in job_ids]
    finally:
        connect.close(conn)


@get("/prompts/{prompt_id:int}/neighbours", sync_to_thread=True)
def prompt_neighbours(
    state: State,
    request: Request,
    prompt_id: FromPath[int],
    space: FromQuery[str],
    k: FromQuery[int] = 10,
    role: FromQuery[str | None] = None,
) -> Template | Response:
    """Prompts nearest to one prompt in ONE chosen space (`space` names
    the provider) under its current query policy, by that space's own
    cosine; no model loads. `role` constrains the candidates before
    ranking. Scores from different spaces are never merged
    (db/prompts.py neighbours)."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        try:
            told = prompts.neighbours(conn, prompt_id, space, weights, k, time.time(), role=role)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
    finally:
        connect.close(conn)
    return presented_page(request, told, page="prompt_neighbours.html", context={"told": told})


@get("/search", sync_to_thread=True)
def search(state: State, q: FromQuery[str], k: FromQuery[int] = 60) -> dict:
    """Pictures by what they LOOK like: the phrase becomes a query vector
    in every participating joint space, each resident index answers with
    its nearest pictures, and the rankings fuse (db/retrieval.py) -- no
    tags, no captions, no metadata anywhere in the loop.

    The fused RRF score orders `results`; each space's own rank and raw
    cosine ride along as `sources`, because cross-model scores are not
    comparable and knowing which model found what is the evidence the
    next model choice is made on. `participants`, `contributors` and
    `missing` say which configured spaces actually answered -- a page
    that hides a silently absent model reports agreement that never
    happened. No model weights are ever downloaded on this path --
    provisioning belongs to /jobs/embed, and a request NOTHING can
    answer is refused.
    """
    from db import retrieval

    conn = _connect(state.db_path)
    try:
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        try:
            found = retrieval.query(conn, weights, q, int(k), time.time(), offline=True)
        except (ValueError, LookupError) as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()  # align may have minted registry rows on the way
        results = found["results"]
        told = []
        named = pages.files_named(conn, [row["file_id"] for row in results])
        for row in results:
            if row["file_id"] in named:
                slug, name = named[row["file_id"]]
                told.append({"slug": slug, "name": name, "score": row["score"], "sources": row["sources"]})
        return {
            "results": told,
            "participants": found["participants"],
            "contributors": found["contributors"],
            "missing": found["missing"],
            "unmatched": found["unmatched"],
        }
    finally:
        connect.close(conn)


@post("/jobs/dupes", sync_to_thread=True)
def submit_dupes(state: State) -> dict:
    """Ask for the perceptual copies to be grouped, using what /jobs/phash
    (or detection's byproduct) recorded. The dupe_threshold setting is
    the hamming radius; a bad value is refused here, not queued."""
    conn = _connect(state.db_path)
    try:
        try:
            job_id = runner.submit_dupes(conn, time.time())
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        return _submitted(state, conn, job_id)
    finally:
        connect.close(conn)


#: What a group is called when its members are byte-identical, and when
#: they are only alike. The distinction decides whether consolidating is
#: safe, so it is a fact the page states rather than a word it picks.
IDENTICAL = "identical"
ALIKE = "alike"


def _dupe_group(conn, best_slug: str, best_name: str, copies: int) -> dict:
    """One group as the review page needs it: the copies, where each
    LIVES, and whether they are the same bytes.

    The placements are why this is a review and not a delete button.
    Three copies of one photograph filed under `Iowa 2019`, `Family` and
    `Old Backup` are one content and three placements; a page that says
    "3 copies" and nothing else invites somebody to remove two and leave
    two collections quietly incomplete.

    `payloads` is how many DISTINCT sets of bytes the group holds. These
    groups are perceptual, so a re-encode, a resize and a crop can all
    land in one: at 1 the copies really are one payload and consolidating
    loses nothing, and above 1 they are alike but not the same, and
    nothing here should suggest otherwise.
    """
    from vision import thumbs

    found = naming.resolve(conn, "file", best_slug)
    group_id = None if found is None else pages.dupe_group_of(conn, found[0])
    members = [] if group_id is None else pages.dupe_members(conn, group_id)
    payloads = {one["content_sha256"] for one in members if one["content_sha256"]}
    return {
        "slug": best_slug,
        "name": best_name,
        "copies": copies,
        "payloads": len(payloads),
        "kind": IDENTICAL if len(payloads) == 1 else ALIKE,
        # The arithmetic said plainly, and only where it is true. Bytes
        # that differ cannot be consolidated to one payload, so a group
        # of merely-similar pictures is offered no such sentence.
        "consolidates_to": len(members) if len(payloads) == 1 else None,
        "members": [
            {
                "slug": one["slug"],
                "name": one["name"],
                "kind": one["kind"],
                "folder": one["folder"],
                "collections": one["collections"],
                "distance": one["distance"],
                "is_best": bool(one["is_best"]),
                "verified": bool(one["verified"]),
                "size": one["size"],
                "thumb": thumbs.asset_url(one["content_sha256"], one["slug"], medium=one["kind"]),
            }
            for one in members
        ],
    }


class NotADuplicate(Wire):
    """The body of POST /dupes/{slug}/not-a-duplicate: the other one."""

    other: str


@post("/dupes/{slug:str}/not-a-duplicate", sync_to_thread=True)
def not_a_duplicate(state: State, slug: FromPath[str], data: NotADuplicate) -> Response[None]:
    """These two are not the same picture.

    The same doctrine denying a person follows, because it is the same
    problem. A perceptual group is a GUESS -- pHash sees composition, so
    two photographs of one scene a second apart are close in it -- and
    the page had no way to disagree with one.

    Said here it takes them apart now AND survives the next sweep, which
    reads the verdicts back before it writes a group. A correction that
    lasted only until the next run would be a chore repeated for ever.
    """
    conn = _connect(state.db_path)
    try:
        one = naming.resolve(conn, "file", slug)
        other = naming.resolve(conn, "file", data.other.strip().removeprefix("/i/"))
        if one is None or other is None:
            raise NotFoundException("no such picture")
        if one[0] == other[0]:
            raise ClientException("a picture is not a duplicate of itself")
        authored.reject_duplicate(conn, one[0], other[0], state.actor_id, time.time())
        conn.commit()
    finally:
        connect.close(conn)
    return Response(content=None, status_code=204)


@get("/dupes", sync_to_thread=True)
def dupes(state: State, request: Request) -> Template | Response:
    """Every group of perceptual copies -- rendered for a browser, the
    historical JSON list for everything else.

    Detection has shipped since `/jobs/dupes`; seeing the result had no
    surface at all. This one is deliberately READ-ONLY: it shows what is
    duplicated, where each copy lives, and whether the copies are the
    same bytes. Nothing here removes anything, because the operation
    worth building is not "delete duplicates" but "consolidate redundant
    storage while preserving every logical placement", and the preview is
    the half that has to be right first.
    """
    conn = _connect(state.db_path)
    try:
        groups = pages.dupe_groups(conn)
        if wants_json(request):
            return Response(_rows(groups, ("slug", "name", "copies")), headers=VARIES)
        told = [_dupe_group(conn, slug, name, copies) for slug, name, copies in groups]
    finally:
        connect.close(conn)
    return presented_page(request, told, page="dupes.html", context={"groups": told})


@post("/jobs/context", sync_to_thread=True)
def submit_context(state: State, everything: FromQuery[bool] = False) -> dict | Response:
    """Ask for every present file still without a current interpretation
    to get one from its sources' claims -- `?everything=true` for all of
    them again -- one item per file (db/runner.py submit_context). 204
    when every file is interpreted. Nothing expensive runs on a GET."""
    conn = _connect(state.db_path)
    try:
        job_id = runner.submit_context(conn, time.time(), everything=everything)
        if job_id is None:
            return Response(content=None, status_code=204)
        conn.commit()
        return _submitted(state, conn, job_id)
    finally:
        connect.close(conn)


@post("/jobs/events", sync_to_thread=True)
def submit_events(state: State) -> dict:
    """Ask for the grouping hypotheses to be re-proposed over the
    current contexts -- one item per Grouper (db/runner.py
    submit_events)."""
    conn = _connect(state.db_path)
    try:
        job_id = runner.submit_events(conn, time.time())
        conn.commit()
        return _submitted(state, conn, job_id)
    finally:
        connect.close(conn)


@post("/jobs/ingest", sync_to_thread=True)
def submit_ingest(
    state: State, everything: FromQuery[bool] = False, folder: FromQuery[str | None] = None
) -> dict | Response:
    """Ask for the metadata of every present file not yet read for its
    current bytes -- `?everything=true` for all of them again -- the
    expensive half of scanning, as a job (db/runner.py submit_ingest).
    204 when every file is read.

    `?folder=<slug>` bounds it to that folder and everything under it,
    which is what makes `everything` usable on a real library. Re-reading
    is how this application corrects itself -- improving a parser is a
    re-parse -- and "re-read all eighty thousand files" is a price nobody
    pays to fix one folder.
    """
    conn = _connect(state.db_path)
    try:
        # `_resolved` answers (id, live_slug_when_retired); a submit takes
        # the id and lets the retired spelling be, because a job is not
        # an address somebody bookmarks.
        folder_id = None if folder is None else _resolved(conn, "folder", folder, "/f")[0]
        job_id = runner.submit_ingest(conn, time.time(), everything=everything, folder_id=folder_id)
        if job_id is None:
            return Response(content=None, status_code=204)
        conn.commit()
        return _submitted(state, conn, job_id)
    finally:
        connect.close(conn)


@post("/jobs/cluster", sync_to_thread=True)
def submit_cluster(state: State) -> dict:
    """Ask for the faces to be grouped into people.

    The step the People page is downstream of, offered by the application
    itself: every embedding space is re-clustered, names re-applied from
    assertions, and each still-unnamed group minted an addressable person
    (db/runner.py submit_cluster)."""
    conn = _connect(state.db_path)
    try:
        job_id = runner.submit_cluster(conn, time.time())
        conn.commit()
        return _submitted(state, conn, job_id)
    finally:
        connect.close(conn)


def _file_at(conn, slug: str, where: str) -> tuple[int, str] | str:
    """Resolve a file slug to `(file_id, disk path)`, refusing what is not
    there to serve. A retired slug comes back as the LIVE slug (a str) so
    each caller can shape its own 301 -- a HEAD handler may not return a
    Redirect, whose annotation implies a body (litestar-org/litestar@
    64cd7da litestar/handlers/http_handlers/decorators.py:588-601)."""
    found = naming.resolve(conn, "file", slug)
    if found is None:
        raise NotFoundException(f"no file at {where}/{slug}")
    file_id, is_current = found
    if not is_current:
        live = naming.entity_slug(conn, file_id)
        if live is not None:
            return live[1]
    if pages.file_present(conn, file_id) is not True:
        raise NotFoundException(f"{where}/{slug} is not on disk right now")
    path = detect.path_of(conn, file_id)
    if not os.path.isfile(path):
        raise NotFoundException(f"{where}/{slug} is not on disk right now")
    return file_id, path


@route("/media/{slug:str}", http_method=["GET", "HEAD"], sync_to_thread=True)
def media_bytes(state: State, slug: FromPath[str], request: Request) -> Stream | Redirect | Response[bytes]:
    """The original bytes, typed by what they are and seekable by range.

    Content-Type comes from the sniff, never the suffix -- the route
    exists to feed decoders and `<video>` elements, and feeding them a
    lie about an MP4 wearing .jpg is how players break. Range semantics
    live in sg_web/media.py.

    HEAD answers here too, with the same headers and no body (RFC 9110:
    a resource that answers GET answers HEAD) -- one mixed-method handler
    rather than a separate `@head` sibling, because registering a second
    handler on a sync handler's path breaks the sync wrapper upstream
    (GET answers 500 "coroutine has no attribute to_asgi_response";
    reproduced on litestar-org/litestar@64cd7da with a 15-line pair, while
    its own static_files pairs @get with @head only as async handlers,
    litestar/static_files.py:115-133). The explicit content-length
    survives the empty body because the response base only setdefaults it
    (litestar/response/base.py:112-113).
    """
    conn = _connect(state.db_path)
    try:
        resolved = _file_at(conn, slug, "/media")
        if isinstance(resolved, str):
            return Redirect(path=f"/media/{resolved}", status_code=301)
        _, path = resolved
    finally:
        connect.close(conn)

    from vision import sniff as sniff_module

    size = os.path.getsize(path)
    ctype = sniff_module.content_type(sniff_module.sniff_path(path))
    if request.method == "HEAD":
        # b"", not None: render() refuses None under a non-text media type
        # ("unsupported media_type image/png for content None"). The empty
        # body computes length 0, and the true length survives because the
        # base only setdefaults content-length (response/base.py:112-113).
        return Response(
            content=b"",
            media_type=ctype,
            headers={"content-length": str(size), "accept-ranges": "bytes"},
        )
    try:
        wanted = media.parse_range(request.headers.get("range"), size)
    except media.Unsatisfiable:
        return Response(content=b"", status_code=416, headers={"content-range": f"bytes */{size}"})
    if wanted is None:
        return Stream(
            media.chunks(path, 0, size),
            media_type=ctype,
            headers={"content-length": str(size), "accept-ranges": "bytes"},
        )
    first, last = wanted
    return Stream(
        media.chunks(path, first, last - first + 1),
        status_code=206,
        media_type=ctype,
        headers={
            "content-length": str(last - first + 1),
            "content-range": f"bytes {first}-{last}/{size}",
            "accept-ranges": "bytes",
        },
    )


def _variant_bytes(state: State, slug: str, variant: str, where: str) -> Response[bytes] | Redirect:
    """Serve one cached raster variant, rendering it on first request.

    The byproduct path (detection jobs) usually got here first; this is
    the fallback for files no job has touched. Kinds with no picture to
    take -- audio, documents -- are told so rather than given a favicon.
    """
    from vision import derive, thumbs

    conn = _connect(state.db_path)
    try:
        resolved = _file_at(conn, slug, where)
        if isinstance(resolved, str):
            return Redirect(path=f"{where}/{resolved}", status_code=301)
        file_id, path = resolved
        held = pages.file_bytes(conn, file_id)
        if held is None:
            raise NotFoundException(f"no file at {where}/{slug}")
        kind, sha = held
        if kind not in thumbs.PICTURED:
            raise NotFoundException(f"a {kind} has no {variant}")
        if sha is None:
            sha = scan.sha256_of(path)
        cache = home.thumbs_dir(pathlib.Path(state.home))
        target = thumbs.path_for(cache, sha, variant)
        if not target.exists():
            # A browser asking for one variant needs only that one's
            # pixels; the precache job is the caller that renders both.
            derive.put_one(cache, sha, pathlib.Path(path), kind, oriented.orientation_of(conn, file_id), variant)
    finally:
        connect.close(conn)
    return Response(content=target.read_bytes(), media_type="image/webp")


#: How long a content-addressed asset may be kept. A year, and
#: `immutable`, because the URL names the BYTES: `<sha>.webp` cannot come
#: to mean different pixels, so a browser that has it never needs to ask
#: again. Immich says the same thing about its assets in
#: refs/immich-app/immich/server/src/utils/file.ts:41 -- `private`
#: because a library is somebody's, cacheable because the bytes are
#: fixed.
ASSET_CACHE = "private, max-age=31536000, immutable, no-transform"

#: The variants an asset URL may name, and the on-disk suffix of each.
#: A closed vocabulary, so a request cannot ask for a path.
ASSET_VARIANTS = {"thumb": "", "preview": ".preview"}
#: The same vocabulary, read the other way, for a URL that arrives.
ASSET_VARIANTS_BY_SUFFIX = {suffix: name for name, suffix in ASSET_VARIANTS.items()}

_ASSET_NAME = re.compile(r"^([0-9a-f]{64})(\.preview)?\.webp$")


@get("/thumbs/{shard:str}/{name:str}", sync_to_thread=True, name="asset")
def asset_bytes(state: State, shard: FromPath[str], name: FromPath[str]) -> File:
    """One derivative, by the hash of the bytes it was made from.

    NO DATABASE. Not a connection, not a slug to resolve, not a kind to
    look up -- the URL already carries the only fact needed, because the
    cache is keyed on `content_sha256` and always was. Sixty cells used
    to be sixty connections; this is the whole reason the hash now rides
    the ResultSet's rows.

    The name is matched against a pattern rather than trusted: sixty-four
    hex characters and one of two known suffixes, so nothing that is not
    a cache entry can be spelled, and `..` cannot appear at all.
    """
    found = _ASSET_NAME.match(name)
    if found is None or shard != name[:2]:
        raise NotFoundException(f"/thumbs/{shard}/{name} is not the name of a derivative")
    sha, suffix = found.group(1), found.group(2) or ""
    if suffix not in ASSET_VARIANTS.values():
        raise NotFoundException(f"/thumbs/{shard}/{name} names no variant")
    target = home.thumbs_dir(pathlib.Path(state.home)) / shard / name
    if not target.is_file():
        # A MISS RENDERS, exactly as the slug route always did.
        #
        # The surface emits this URL for anything ingest has hashed,
        # which is not the same set as "anything the thumbs job has
        # rendered" -- so 404ing here would give a fresh library a grid
        # of broken pictures where it used to give a slow one. This is
        # the ONLY path that opens a connection, and after the precache
        # job it is never taken.
        #
        # By CONTENT, not by slug: the cache is keyed on the bytes, so
        # any present file carrying them will do, which is the whole
        # reason it is content-addressed.
        try:
            _render_asset(state, sha, ASSET_VARIANTS_BY_SUFFIX[suffix], target)
        except ValueError as unrenderable:
            # A file with no decodable frame has no thumbnail, and that
            # is a 404 rather than a defect: the request asked for a
            # picture of something that does not have one.
            #
            # It reached here as an uncaught 500 with a traceback, once
            # per cell, for a folder of album tracks -- an .m4a is
            # ISO-BMFF, so the sniffer called it video/mp4 (fixed in
            # vision/sniff.py) and the grid minted it an address. The
            # sniff was the cause; this is the reason one bad row cost a
            # page of stack traces instead of one grey cell.
            raise NotFoundException(str(unrenderable)) from unrenderable
    return File(
        path=target,
        media_type="image/webp",
        content_disposition_type="inline",
        headers={"cache-control": ASSET_CACHE},
    )


def _render_asset(state: State, sha: str, variant: str, target: pathlib.Path) -> None:
    """Render one missing derivative from any file with those bytes."""
    from vision import derive, thumbs

    conn = _connect(state.db_path)
    try:
        found = pages.file_of_content(conn, sha)
        if found is None:
            raise NotFoundException(f"nothing present carries the bytes {sha[:12]}")
        file_id, kind = found
        if kind not in thumbs.PICTURED:
            raise NotFoundException(f"a {kind} has no {variant}")
        path = detect.path_of(conn, file_id)
        if not os.path.isfile(path):
            raise NotFoundException(f"the bytes behind {sha[:12]} are offline")
        derive.put_one(
            home.thumbs_dir(pathlib.Path(state.home)),
            sha,
            pathlib.Path(path),
            kind,
            oriented.orientation_of(conn, file_id),
            variant,
        )
    finally:
        connect.close(conn)
    if not target.is_file():
        raise NotFoundException(f"the {variant} of {sha[:12]} could not be rendered")


@get("/thumb/{slug:str}", sync_to_thread=True)
def thumb_bytes(state: State, slug: FromPath[str]) -> Response[bytes] | Redirect:
    """The grid cell: longest side 512, upright, aspect kept."""
    return _variant_bytes(state, slug, "thumb", "/thumb")


@get("/preview/{slug:str}", sync_to_thread=True)
def preview_bytes(state: State, slug: FromPath[str]) -> Response[bytes] | Redirect:
    """The lightbox image: longest side 1440, upright, aspect kept."""
    return _variant_bytes(state, slug, "preview", "/preview")


@get("/avatar/{slug:str}", sync_to_thread=True)
def avatar_bytes(state: State, slug: FromPath[str]) -> Response[bytes] | Redirect:
    """A person's face, squared: their highest-confidence detection in the
    primary run, cropped with context (vision/thumbs.py). A video face is
    cropped from the sampled frame the detection actually looked at."""
    from vision import decode, thumbs

    conn = _connect(state.db_path)
    try:
        found = naming.resolve(conn, "person", slug)
        if found is None:
            raise NotFoundException(f"no person at /avatar/{slug}")
        person_id, is_current = found
        if not is_current:
            live = naming.entity_slug(conn, person_id)
            if live is not None:
                return Redirect(path=f"/avatar/{live[1]}", status_code=301)
        face = media.exemplar_face(conn, person_id)
        if face is None:
            raise NotFoundException(f"/avatar/{slug}: no clustered face to show")
        face_id, file_id, sample_id, x, y, w, h = face
        cache = home.thumbs_dir(pathlib.Path(state.home))
        target = thumbs.avatar_path(cache, face_id)
        if not target.exists():
            path = detect.path_of(conn, file_id)
            if not os.path.isfile(path):
                raise NotFoundException(f"/avatar/{slug}: the picture behind the face is offline")
            if sample_id is not None:
                offset = sample.offset_of(conn, sample_id)
                frame = next((image for _, image in decode.frames_at(path, [offset])), None)
            else:
                frame = oriented.for_model(conn, file_id, path)
            if frame is None:
                raise NotFoundException(f"/avatar/{slug}: the face's frame no longer decodes")
            thumbs.put_avatar(cache, face_id, frame, (x, y, w, h))
    finally:
        connect.close(conn)
    return Response(content=target.read_bytes(), media_type="image/webp")


@get("/settings", sync_to_thread=True)
def all_settings(state: State) -> list[dict]:
    """Every setting, its value, default and choices -- the whole vocabulary."""
    conn = _connect(state.db_path)
    try:
        return settings.snapshot(conn)
    finally:
        connect.close(conn)


class SettingChange(Wire):
    """The body of POST /settings/{key}. A setting's value is stored as
    text and read back through its own vocabulary, so the wire carries
    whichever JSON scalar the setting is spelled with rather than
    pretending every setting is a string."""

    value: str | int | float | bool


@post("/settings/{key:str}", sync_to_thread=True)
def change_setting(state: State, key: FromPath[str], data: SettingChange) -> dict:
    """Change one setting while the application runs. Unknown keys and
    out-of-vocabulary values are refused, so the table only ever holds
    configuration something reads."""
    conn = _connect(state.db_path)
    try:
        try:
            settings.put(conn, key, str(data.value))
        except (KeyError, ValueError) as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        return {"key": key, "value": settings.value(conn, key)}
    finally:
        connect.close(conn)


@post("/jobs/{job_id:int}/cancel", sync_to_thread=True)
def cancel_job(state: State, job_id: FromPath[int]) -> dict:
    """Ask a job to stop. The runner stops it, at an item boundary; a
    still-queued job needs a claim to settle, hence the nudge."""
    conn = _connect(state.db_path)
    try:
        jobs.cancel(conn, job_id, time.time())
        conn.commit()
        # The request changed the row (cancel_requested), so the request
        # speaks: a subscriber sees "cancelling" now, not at the worker's
        # next item -- which never comes while the worker is off.
        told = _announce(state, conn, job_id, event_type="job.cancel_requested")
        _nudge(state)
        return told
    finally:
        connect.close(conn)


def _jobs_snapshot_of(db_path: str) -> JobsSnapshotFrame:
    """Every unsettled job as one frame -- what the feed opens with."""
    conn = _connect(db_path)
    try:
        return JobsSnapshotFrame(type="snapshot", jobs=[_job_listed(row) for row in jobs.active(conn)])
    finally:
        connect.close(conn)


@websocket("/ws/jobs")
async def jobs_feed(socket: WebSocket, channels: NamedDependency[ChannelsPlugin], state: State) -> None:
    """Live job progress: the persisted snapshot first, then every delta.

    The subscription opens BEFORE the snapshot is read, so a delta landing
    between the two is queued behind the snapshot instead of lost; a
    client applies deltas onto the snapshot and can never render a state
    the rows did not hold. The channel is transport, never storage --
    reconnection starts from the rows again (db/jobs.py). The snapshot
    read crosses to a thread (anyio.to_thread.run_sync, agronholm/anyio
    src/anyio/to_thread.py:27-52) because sqlite blocks and this handler
    shares the event loop with every open socket.

    Two representations of the same feed, chosen by `?as=`: JSON (the
    machine default, snapshot then raw deltas) and `html` -- the list and
    each delta rendered as out-of-band fragments (sg_web/activity.py) for
    the shell's activity surface, which the htmx ws extension swaps in by
    id. The query string is the only negotiation a browser WebSocket can
    carry: the extension opens `new WebSocket(url, [])` with no headers
    (bigskysoftware/htmx-extensions@1358232 src/ws/ws.js createWebSocket).
    Same subscribe-then-snapshot order either way.
    """
    from anyio import to_thread

    as_html = socket.query_params.get("as") == "html"
    await socket.accept()
    async with channels.start_subscription("jobs") as subscriber:
        if as_html:
            engine = socket.app.template_engine
            listed = await to_thread.run_sync(activity.rows, state.db_path)
            seen = {int(row["id"]) for row in listed if not row["settled"]}
            await socket.send_text(activity.render_list(engine, listed))

            async def relay(raw: bytes) -> None:
                """One delta rendered and sent. A render that fails is a
                defect in the fragment, not in the feed: it is logged
                whole, the socket is closed 1011 so the extension
                reconnects and re-reads the rows (bigskysoftware/
                htmx-extensions@1358232 src/ws/ws.js:256 -- close codes
                1006/1011/1012/1013 retry), and the error propagates --
                never a silent dead task."""
                try:
                    frame = activity.render_delta(engine, json.loads(raw), seen)
                except Exception:
                    _logger.exception("activity fragment failed to render for delta %r", raw)
                    await socket.close(code=1011, reason="activity render failed")
                    raise
                await socket.send_text(frame)

            deliver = relay
        else:
            snapshot = await to_thread.run_sync(_jobs_snapshot_of, state.db_path)
            await socket.send_json(snapshot.model_dump(mode="json"))

            async def send_delta(raw: bytes) -> None:
                """One channel payload as the frame the contract describes.

                The channel carries what the publishers put on it -- a row's
                own columns, `cancel_requested` among them as the 0 or 1
                SQLite holds. Building the frame HERE is what makes the
                translating happen once for both publishers, and what stops
                the socket from being the one surface that forwards storage
                straight to a browser.
                """
                await socket.send_json(_job_delta(json.loads(raw)).model_dump(mode="json"))

            deliver = send_delta
        async with subscriber.run_in_background(deliver):
            while (await socket.receive())["type"] != "websocket.disconnect":
                continue


def _backlog_of(db_path: str, after: int) -> console.BacklogFrame:
    """The ledger since `after` as one frame, read at a head the frame
    names -- one read-only connection, one ordered index walk."""
    conn = connect.connect(db_path, read_only=True)
    try:
        return console.BacklogFrame(
            frame="backlog",
            events=[console.envelope(event) for event in ledger.since(conn, after)],
            after=after,
            last_id=ledger.last_id(conn),
        )
    finally:
        connect.close(conn)


@websocket("/ws/events")
async def events_feed(socket: WebSocket, channels: NamedDependency[ChannelsPlugin], state: State) -> None:
    """The ledger, live: `?after=N` names the last event id the client
    holds; everything newer is sent first as `backlog` frames read from
    the rows, then every committed row as it is published (`event`) and
    every handler report between commits (`pending`, no id -- see
    db/runner.py Report).

    Subscribe-then-backlog, the order /ws/jobs uses: a row committed
    while the backlog is being read is queued behind it, never lost, and
    a row that lands in both is the same id twice -- the client keeps
    one. Ids are the order; a client whose ids skip knows exactly what
    it is missing and asks GET /operations/events for it. The channel
    stores nothing: a reconnect resumes from the rows.
    """
    from anyio import to_thread

    raw_after = socket.query_params.get("after", "0")
    after = int(raw_after) if str(raw_after).isdigit() else 0
    await socket.accept()
    async with channels.start_subscription("events") as subscriber:
        while True:
            page = await to_thread.run_sync(_backlog_of, state.db_path, after)
            await socket.send_json(page.model_dump(mode="json"))
            if len(page.events) < ledger.PAGE_MOST:
                break
            after = page.events[-1].id
        async with subscriber.run_in_background(socket.send_text):
            while (await socket.receive())["type"] != "websocket.disconnect":
                continue


@get("/roots", sync_to_thread=True)
def roots(state: State) -> list[dict]:
    """Every media directory this library reads, and whether it is
    reachable right now. Media roots are rows, not configuration: any
    number of directories, anywhere, and they travel with the database.

    The OPERATIONAL surface: check_roots records `online`, and the
    commit here is what makes the record real -- the browsing /folders
    route observes without writing (db/library.py probe_roots)."""
    conn = _connect(state.db_path)
    try:
        seen = library.check_roots(conn)
        conn.commit()
        return [{"id": root_id, "path": path, "online": online} for root_id, path, online in seen]
    finally:
        connect.close(conn)


#: What a watched directory is to the library, per db/schema.sql root.kind.
#: What a root IS. 'mount' was here and nothing branched on it -- the
#: distinction it reached for, "not always attached", is `root.online`,
#: which is per-root and set by probing.
RootKind = Literal["library", "trash"]


class NewRoot(Wire):
    """The body of POST /roots: a directory and what it is to us."""

    path: str
    kind: RootKind = "library"


@post("/roots", sync_to_thread=True)
def add_root(state: State, data: NewRoot) -> dict:
    """Register a media directory. Nothing is read until a scan is asked
    for -- registering is a statement of intent, not a sweep."""
    conn = _connect(state.db_path)
    try:
        root_id = library.add_root(conn, data.path, data.kind, time.time())
        conn.commit()
        return {"id": root_id, "path": data.path}
    finally:
        connect.close(conn)


class RootRemoval(Wire):
    """What removing one root would cost.

    Every count is of ROWS. Nothing on disk is counted because nothing
    on disk is touched -- the numbers that matter are the ones a rescan
    cannot bring back.
    """

    root: int
    path: str
    folders: int
    files: int
    ratings: int
    favorites: int
    comments: int
    people_named: int
    places: int
    in_collections: int


class RootForgotten(Wire):
    """What removing one root did cost, counted before it happened."""

    forgot: RootRemoval


@get("/roots/{root_id:int}/removal", sync_to_thread=True)
def removal_cost(state: State, root_id: FromPath[int]) -> RootRemoval:
    """What removing this root would cost, before anything is removed.

    Its own address because a person is entitled to look before they
    decide, and because the answer is not obvious: `folder.root_id`
    cascades to folders, folders to files, files to every rating,
    comment, favourite, name, place and collection membership on them.
    Nothing on disk is counted, because nothing on disk is touched.
    """
    conn = _connect(state.db_path)
    try:
        told = library.removal_cost(conn, root_id)
        if told["path"] is None:
            raise NotFoundException(f"no root {root_id}")
        return RootRemoval(**told)
    finally:
        connect.close(conn)


@delete("/roots/{root_id:int}", status_code=200, sync_to_thread=True)
def forget_root(state: State, root_id: FromPath[int], confirm: FromQuery[str] = "") -> RootForgotten:
    """Stop indexing a directory and drop what was indexed from it.

    Nothing on disk is touched. Re-adding the directory finds every file
    again; what does not come back is the knowledge attached to them,
    which is the half no rescan can recompute.

    `confirm` must be the root's own path. Not ceremony: this is the
    only route in the application that can destroy authored state in
    bulk, and the doctrine everywhere else is that a destructive act
    proves what it is acting on first. Ask `GET /roots/{id}/removal` to
    see the cost, then repeat the path back.
    """
    conn = _connect(state.db_path)
    try:
        told = library.removal_cost(conn, root_id)
        if told["path"] is None:
            raise NotFoundException(f"no root {root_id}")
        if confirm != told["path"]:
            raise ClientException(
                f"removing this root drops {told['files']} file(s) and what is attached to them"
                f" -- {told['ratings']} rating(s), {told['favorites']} favourite(s),"
                f" {told['comments']} comment(s), {told['people_named']} named person(s),"
                f" {told['places']} place(s), {told['in_collections']} collection membership(s)."
                f" Nothing on disk is touched. Repeat the path to confirm:"
                f" DELETE /roots/{root_id}?confirm={told['path']}"
            )
        removed = library.forget_root(conn, root_id)
        conn.commit()
        return RootForgotten(forgot=RootRemoval(**removed))
    finally:
        connect.close(conn)


#: The cadence, from where the walk lives (db/scan.py). Imported rather
#: than restated: two callers reporting at two rates would tell somebody
#: about their library differently depending on which one they asked.
WALK_EVERY = scan.WALK_EVERY


@post("/roots/{root_id:int}/scan", sync_to_thread=True)
def scan_root(state: State, root_id: FromPath[int]) -> dict:
    """Walk one root and reconcile the library with what is on disk.

    The walk is a JOB, and until now it was the one expensive thing here
    that was not. Everything cheaper that follows it -- hashing,
    thumbnails, embeddings -- reported itself into the operations
    console, while this read every byte of every changed file and showed
    nothing at all: the request hung, and on a large root it hung for
    minutes.

    Still synchronous, so the answer is still the counts. What is new is
    that somebody watching has something to watch: a `walk` row, its
    phases, and a file count that moves.
    """
    conn = _connect(state.db_path)
    try:
        path = library.root_path(conn, root_id)
        if path is None:
            raise NotFoundException(f"no root {root_id}")

        # This request IS the worker for its own job -- the same
        # checkpoint/settle the runner uses, so the row obeys the same
        # invariants (a fence, a lease, one owner) rather than being a
        # second kind of job nothing else understands.
        #
        # `begin`, not submit-then-claim: between those two the row sits
        # QUEUED, and the background worker polls for any runnable kind.
        # It took the walk, found no handler for it, failed it, and the
        # request that had just created it then could not claim its own
        # work. Inserted running and owned, it is never claimable.
        #
        # The root goes in the PAYLOAD: `target_id` references `entity`,
        # and a root is not one -- its top folder is, and on a first scan
        # that folder does not exist until this walk makes it.
        owner = f"scan-{os.getpid()}"
        walking, fence = jobs.begin(conn, "walk", owner, time.time(), payload={"root": root_id, "path": path})
        conn.commit()
        _announce(state, conn, walking, event_type="job.submitted")

        # The last seen counts, kept so the FINAL report is the true one.
        # Throttling alone left the job settled at 750 of 790 -- the tail
        # below the cadence was never spoken, so the count a person was
        # watching stopped short of the number the same request returned.
        seen = {"folders": 0, "files": 0, "hashed": 0, "spoken": 0}

        def say(folders: int, files: int, hashed: int) -> None:
            jobs.checkpoint(conn, walking, fence, {"folders": folders, "hashed": hashed}, files, at=time.time())
            conn.commit()
            _announce(state, conn, walking)

        def watch(folders: int, files: int, hashed: int) -> None:
            seen.update(folders=folders, files=files, hashed=hashed)
            if files - seen["spoken"] < WALK_EVERY:
                return
            seen["spoken"] = files
            say(folders, files, hashed)

        try:
            result = scan.scan(conn, root_id, path, time.time(), watch)
        except Exception as broke:
            jobs.settle(conn, walking, fence, "failed", time.time(), error=f"{type(broke).__name__}: {broke}")
            ledger.record(
                conn,
                walking,
                "job.failed",
                time.time(),
                severity="error",
                message=f"{type(broke).__name__}: {broke}",
            )
            conn.commit()
            _announce(state, conn, walking)
            raise
        if seen["files"] != seen["spoken"]:
            say(seen["folders"], seen["files"], seen["hashed"])
        jobs.settle(conn, walking, fence, "done", time.time())
        ledger.record(
            conn,
            walking,
            "job.done",
            time.time(),
            message=(
                f"walked {result.added + result.matched + result.replaced} file(s),"
                f" hashed {result.hashed}, {result.missing} now missing"
            ),
            data={"added": result.added, "matched": result.matched, "hashed": result.hashed},
        )
        conn.commit()
        _announce(state, conn, walking)
        cache = str(home.thumbs_dir(pathlib.Path(state.home)))
        precache = runner.precache_after_scan(conn, time.time(), result, thumbs_dir=cache)
        if precache is not None:
            conn.commit()
            _submitted(state, conn, precache)
        return {
            "root": root_id,
            "added": result.added,
            "matched": result.matched,
            "replaced": result.replaced,
            "ambiguous": result.ambiguous,
            "missing": result.missing,
            "hashed": result.hashed,
            "precache": precache,
        }
    finally:
        connect.close(conn)


@post("/clusterings/choose", sync_to_thread=True)
def choose_primary(state: State) -> dict:
    """Re-rank every run and set the default the People page shows."""
    conn = _connect(state.db_path)
    try:
        chosen = derived.choose_primary(conn)
        conn.commit()
        return {"primary_run": chosen}
    finally:
        connect.close(conn)


def _told_whole(request: Request, exc: Exception) -> Response:
    """What broke and where, to whoever asked. An HTTPException keeps its
    own status and detail; anything else is a 500 that carries its
    traceback -- a page for a browser, `details` for a machine."""
    if isinstance(exc, HTTPException):
        return create_exception_response(request, exc)
    return create_debug_response(request, exc)


def _template_engine() -> JinjaTemplateEngine:
    """The ONE Jinja environment every page renders with.

    StrictUndefined: a template that names a field the view did not
    supply explodes at render, instead of printing an empty string and
    shipping "You introduced ." to a screen. Autoescape: every value a
    template prints is evidence (file names, prompt text), never trusted
    markup. Litestar's engine wraps the environment
    (litestar-org/litestar@v2.24.0 litestar/plugins/jinja.py:106-115
    `from_environment` -> `cls(directory=None, engine_instance=...)`);
    passed as `TemplateConfig(instance=...)` the callback path is skipped
    (litestar/template/config.py:58-61 `engine_instance`), so the activity
    Module's global is registered here, before any template loads
    (pallets/jinja@3.1.6 docs/api.rst "The Global Namespace").
    """
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    environment = Environment(
        loader=FileSystemLoader(str(pathlib.Path(__file__).resolve().parent / "templates")),
        undefined=StrictUndefined,
        autoescape=True,
    )
    # every stylesheet and script URL carries the newest mtime under
    # static/: a browser that cached yesterday's timeline.js fetches
    # today's, and a deploy that changed nothing keeps every cache warm
    environment.globals["static_v"] = _static_version()
    engine = JinjaTemplateEngine.from_environment(environment)
    activity.register(engine)
    return engine


def _static_version() -> str:
    """The newest mtime of anything served under `static/`, in millis.

    `rglob`, not `iterdir`: the value is stamped onto
    `/static/build/*.js` as well as onto the stylesheets beside it, and
    a walk one directory deep could not see the bundles. Editing only
    TypeScript therefore left the cache-buster exactly where it was, so
    a browser holding yesterday's `gallery.js` was told the URL had not
    changed and went on holding it -- the one failure this value exists
    to prevent, on the files most likely to change.
    """
    static = pathlib.Path(__file__).resolve().parent / "static"
    newest = max((p.stat().st_mtime_ns for p in static.rglob("*") if p.is_file()), default=0)
    return str(newest // 1_000_000)


def build_app(home_dir: str | None = None, *, worker: bool = True) -> Litestar:
    """The application, bound to one home directory (sg_web/home.py).

    With no argument the run lives in `~/.smartgallery`. A database that
    does not exist yet is created from the schema -- a first run needs
    nothing but the command that starts it.

    `worker=True` -- the runtime truth -- starts the draining thread with
    the app and stops it with the app; the `worker` setting row idles it
    live without a restart. `worker=False` is for embedding the routes
    over a database whose jobs something else is stepping.
    """
    base = home.home(home_dir)
    where = home.db_path(base)
    if not where.exists():
        connect.create(where)
    else:
        # A database an older build wrote is brought forward HERE, one
        # version per transaction with a `.vN.backup` beside it
        # (db/migrate.py) -- never opened as-is to 500 on the first column
        # it lacks. A newer build's file, or one this build has no step
        # for, is refused at boot with the reason, not per request.
        try:
            applied = migrate.migrate(where)
        except (migrate.Downgrade, migrate.StepMissing, migrate.NotOurDatabase) as refused:
            raise SystemExit(f"{where}: {refused}") from refused
        if applied:
            _logger.info("%s: brought forward to v%d (steps %s)", where, applied[-1], applied)

    # The one local authored identity, resolved ONCE into application
    # state: every rating and favorite is per-user by schema, and this is
    # the single place the local-first deployment answers "who is
    # writing" (db/authored.py local_actor). A future session layer
    # replaces this resolution, not the authored signatures.
    opening = connect.connect(where)
    try:
        actor_id = authored.local_actor(opening, time.time())
        opening.commit()
    finally:
        connect.close(opening)

    channels = ChannelsPlugin(MemoryChannelsBackend(), channels=["jobs", "events"])

    @asynccontextmanager
    async def working(app: Litestar):
        """The worker's life is strictly inside the channel's life. The
        loop is captured here because `ChannelsPlugin.publish` must be
        entered from the loop's own thread (call_soon_threadsafe is the
        bridge), and the join on the way out -- before the channel tears
        down, see _WorkerPlugin -- is what makes ctrl-C leave no thread
        mid-write. The join happens on a worker thread while the loop
        keeps running, so publishes the worker scheduled before stopping
        still land on a live channel."""
        import asyncio
        import threading

        from anyio import to_thread

        loop = asyncio.get_running_loop()
        stop, wake = threading.Event(), threading.Event()
        app.state.worker_wake = wake

        def publish(delta: dict) -> None:
            loop.call_soon_threadsafe(channels.publish, delta, "jobs")

        # The latest report INSIDE the item each job is working on -- what
        # the ledger cannot hold yet (db/runner.py Report: it lands at the
        # item boundary). Process memory, never storage: a restart loses
        # it exactly as it loses the item. A reconnecting console reads it
        # through the inspector instead of waiting for the next report.
        live_reports: dict[int, dict] = {}
        app.state.live_reports = live_reports

        def publish_event(event: dict) -> None:
            """A ledger row (or a pending report) onto the events channel,
            with its words and condition (sg_web/console.py) -- the
            presentation seam, so the worker never learns the vocabulary."""
            frame: console.Frame
            if event.get("pending"):
                # No id: a report from inside an item is not a row yet
                # (db/runner.py Report lands at the item boundary), which
                # is why PendingFrame carries no id to send.
                frame = console.pending_frame(event)
                live_reports[int(event["job_id"])] = frame.model_dump(mode="json")
            else:
                frame = console.event_frame(event)
                if not console.inside_item(event["type"]):
                    live_reports.pop(int(event["job_id"]), None)
            loop.call_soon_threadsafe(channels.publish, frame.model_dump(mode="json"), "events")

        # Request handlers run on the thread pool too, so a job's `queued`
        # delta (sg_web/submitting.py) crosses the same bridge the worker's
        # deltas do. Set whether or not the worker thread starts: a
        # submit is an observable change in either case.
        app.state.publish = publish
        app.state.publish_event = publish_event

        thread = threading.Thread(
            target=worker_module.run,
            args=(str(where), publish, stop, wake, publish_event),
            name="sg-worker",
            daemon=True,
        )
        app.state.worker_thread = thread
        if worker:
            thread.start()
        try:
            yield
        finally:
            stop.set()
            wake.set()
            if thread.is_alive():
                await to_thread.run_sync(thread.join)

    class _WorkerPlugin(InitPlugin):
        """Registers `working` AFTER ChannelsPlugin has registered itself.

        Ordering is the point, not convenience: lifespan managers exit in
        reverse (litestar-org/litestar@64cd7da litestar/app.py:598-608,
        AsyncExitStack), and ChannelsPlugin appends its own manager in
        `on_app_init` (channels/plugin.py:123). Passed via `lifespan=[...]`
        the worker preceded the channel, so on shutdown the channel nulled
        its queue first and a draining worker's publish crashed with
        "Plugin not yet initialized". Appended here, plugin order puts the
        worker last -- first to exit, stopped and joined while the channel
        it publishes to is still alive."""

        def on_app_init(self, app_config):
            app_config.lifespan.append(working)
            return app_config

    app = Litestar(
        route_handlers=[
            health,
            front,
            # The sockets' frames, declared so the generated types carry
            # them: a WebSocket has no path OpenAPI can describe.
            console.socket_frames,
            job_frames,
            media_view.media_page,
            media_authored.set_favorite,
            media_authored.judge_said,
            media_authored.deny_person,
            media_authored.set_rating,
            media_authored.set_place,
            media_authored.set_membership,
            media_authored.collection_choices,
            folder_view.folders_index,
            folder_view.folder_page,
            artifact_view.models_index,
            artifact_view.loras_index,
            artifact_view.workflows_index,
            artifact_view.model_page,
            artifact_view.lora_page,
            artifact_view.workflow_page,
            collection_view.albums_index,
            collection_authoring.make_album,
            collection_authoring.make_smart,
            saved_views,
            not_a_duplicate,
            remember_view,
            view_opened,
            forget_view,
            collection_authoring.edit_definition,
            collection_authoring.replace_rule,
            collection_authoring.convert_collection,
            collection_view.album_page,
            album_add,
            album_remove,
            submit_context,
            submit_events,
            submit_ingest,
            submit_phash,
            submit_thumbs,
            submit_catch_up,
            submit_embed,
            submit_embed_prompts,
            prompt_neighbours,
            search,
            submit_dupes,
            dupes,
            timeline_view.timeline,
            timeline_view.density,
            timeline_view.pictures,
            timeline_view.spread,
            timeline_view.at,
            timeline_view.nth,
            story_view.stories_index,
            story_view.freeze_snapshot,
            story_view.snapshot_document,
            story_view.plan_snapshot,
            story_view.plan_document,
            story_view.render_plan,
            story_view.render_document,
            story_view.session_story,
            story_view.plan_evolution,
            person_view.people_index,
            place_view.places_index,
            person_view.person_page,
            clusterings,
            ways,
            roots,
            add_root,
            removal_cost,
            forget_root,
            scan_root,
            active_jobs,
            job_snapshot,
            submit_verify,
            submit_faces,
            submit_annotate,
            submit_cluster,
            person_view.name_person,
            person_view.same_person,
            person_view.choose_face,
            cancel_job,
            jobs_feed,
            events_feed,
            choose_primary,
            all_settings,
            change_setting,
            media_bytes,
            asset_bytes,
            thumb_bytes,
            preview_bytes,
            avatar_bytes,
            gallery.gallery,
            gallery.grid_fragment,
            gallery.filter_options,
            gallery.filter_catalog,
            gallery.field_values,
            gallery.rail_peek,
            gallery.locate_in_answer,
            curating.bulk_favorite,
            curating.bulk_rating,
            curating.bulk_place,
            curating.bulk_membership,
            # The runtime's own surface, under one Router seam (litestar-org/
            # litestar@v2.24.0 docs/usage/routing/overview.rst "Routers"):
            # every operational page and form shares the /operations prefix
            # and whatever policy that layer grows later.
            operations.router,
            create_static_files_router(
                # Absolute on purpose: the docs interpret relative
                # directories against the process working directory
                # (litestar-org/litestar docs/usage/static-files.rst),
                # and this application is started from anywhere.
                path="/static",
                directories=[str(pathlib.Path(__file__).resolve().parent / "static")],
            ),
        ],
        plugins=[*wire.plugins(), channels, _WorkerPlugin()],
        template_config=TemplateConfig(instance=_template_engine()),
        # a 500 says what broke and where: the traceback page for a
        # browser, {"details": <traceback>} for JSON (litestar-org/litestar
        # litestar/exceptions/responses/_debug_response.py:175-195), and
        # the same traceback in the log (litestar/logging/config.py:247)
        exception_handlers={Exception: _told_whole},
        logging_config=LoggingConfig(log_exceptions="always"),
    )
    app.state.home = str(base)
    app.state.db_path = str(where)
    app.state.actor_id = actor_id
    return app
