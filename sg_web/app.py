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

import dataclasses
import json
import logging
import os
import pathlib
import sqlite3
import time
from contextlib import asynccontextmanager

from litestar import Litestar, Request, get, post, route, websocket
from litestar.channels import ChannelsPlugin
from litestar.channels.backends.memory import MemoryChannelsBackend
from litestar.connection import WebSocket
from litestar.datastructures import State
from litestar.di import NamedDependency
from litestar.exceptions import ClientException, NotFoundException
from litestar.params import FromPath, FromQuery
from litestar.plugins import InitPlugin
from litestar.plugins.jinja import JinjaTemplateEngine
from litestar.response import Redirect, Response, Stream, Template
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
)
from sg_web import worker as worker_module
from sg_web.presenting import VARIES, presented_page, wants_json
from sg_web.submitting import announce as _announce
from sg_web.submitting import nudge as _nudge
from sg_web.submitting import submitted as _submitted

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
    """The front door. A browser lands in the gallery -- /g owns the
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


@dataclasses.dataclass
class AlbumEntry:
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


@get("/jobs", sync_to_thread=True)
def active_jobs(state: State) -> list[dict]:
    conn = _connect(state.db_path)
    try:
        return jobs.active(conn)
    finally:
        connect.close(conn)


@get("/jobs/{job_id:int}", sync_to_thread=True)
def job_snapshot(state: State, job_id: FromPath[int]) -> dict:
    """The persisted snapshot -- what a client renders from cold. A page
    reload or a dropped socket recovers by reading this, never a replay."""
    conn = _connect(state.db_path)
    try:
        try:
            return jobs.snapshot(conn, job_id)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
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


@dataclasses.dataclass
class Everything:
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


@get("/dupes", sync_to_thread=True)
def dupes(state: State) -> list[dict]:
    """Every group of perceptual copies, best face forward with a count --
    the page that collapses copies-of-copies into pictures."""
    conn = _connect(state.db_path)
    try:
        return _rows(pages.dupe_groups(conn), ("slug", "name", "copies"))
    finally:
        connect.close(conn)


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
def submit_ingest(state: State, everything: FromQuery[bool] = False) -> dict | Response:
    """Ask for the metadata of every present file not yet read for its
    current bytes -- `?everything=true` for all of them again -- the
    expensive half of scanning, as a job (db/runner.py submit_ingest).
    204 when every file is read."""
    conn = _connect(state.db_path)
    try:
        job_id = runner.submit_ingest(conn, time.time(), everything=everything)
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
def media_bytes(state: State, slug: FromPath[str], request: Request) -> Stream | Redirect | Response:
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


def _variant_bytes(state: State, slug: str, variant: str, where: str) -> Response | Redirect:
    """Serve one cached raster variant, rendering it on first request.

    The byproduct path (detection jobs) usually got here first; this is
    the fallback for files no job has touched. Kinds with no picture to
    take -- audio, documents -- are told so rather than given a favicon.
    """
    from vision import decode, thumbs

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
        if kind not in ("image", "animated_image", "video"):
            raise NotFoundException(f"a {kind} has no {variant}")
        if sha is None:
            sha = scan.sha256_of(path)
        cache = home.thumbs_dir(pathlib.Path(state.home))
        target = thumbs.path_for(cache, sha, variant)
        if not target.exists():
            frame = decode.poster(path) if kind == "video" else oriented.for_model(conn, file_id, path)
            if frame is None:
                raise NotFoundException(f"{where}/{slug} has no decodable frame")
            thumbs.put(cache, sha, frame, variant)
    finally:
        connect.close(conn)
    return Response(content=target.read_bytes(), media_type="image/webp")


@get("/thumb/{slug:str}", sync_to_thread=True)
def thumb_bytes(state: State, slug: FromPath[str]) -> Response | Redirect:
    """The grid cell: longest side 512, upright, aspect kept."""
    return _variant_bytes(state, slug, "thumb", "/thumb")


@get("/preview/{slug:str}", sync_to_thread=True)
def preview_bytes(state: State, slug: FromPath[str]) -> Response | Redirect:
    """The lightbox image: longest side 1440, upright, aspect kept."""
    return _variant_bytes(state, slug, "preview", "/preview")


@get("/avatar/{slug:str}", sync_to_thread=True)
def avatar_bytes(state: State, slug: FromPath[str]) -> Response | Redirect:
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


@post("/settings/{key:str}", sync_to_thread=True)
def change_setting(state: State, key: FromPath[str], data: dict) -> dict:
    """Change one setting while the application runs. Unknown keys and
    out-of-vocabulary values are refused, so the table only ever holds
    configuration something reads."""
    conn = _connect(state.db_path)
    try:
        try:
            settings.put(conn, key, str(data["value"]))
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


def _active_jobs_of(db_path: str) -> list[dict]:
    conn = _connect(db_path)
    try:
        return jobs.active(conn)
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
            rows = await to_thread.run_sync(_active_jobs_of, state.db_path)
            await socket.send_json({"type": "snapshot", "jobs": rows})
            deliver = socket.send_text
        async with subscriber.run_in_background(deliver):
            while (await socket.receive())["type"] != "websocket.disconnect":
                continue


def _backlog_of(db_path: str, after: int) -> tuple[list[dict], int]:
    """The ledger since `after`, as envelopes, and the head id the page
    was read at -- one read-only connection, one ordered index walk."""
    conn = connect.connect(db_path, read_only=True)
    try:
        return [console.envelope(event) for event in ledger.since(conn, after)], ledger.last_id(conn)
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
            page, head = await to_thread.run_sync(_backlog_of, state.db_path, after)
            await socket.send_json({"frame": "backlog", "events": page, "after": after, "last_id": head})
            if len(page) < ledger.PAGE_MOST:
                break
            after = page[-1]["id"]
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


@dataclasses.dataclass
class NewRoot:
    """The body of POST /roots. Typed so a request without a path is a
    400 from the signature model, never a KeyError a handler forgot --
    the shape every write route's body should take."""

    path: str
    kind: str = "library"


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


@post("/roots/{root_id:int}/scan", sync_to_thread=True)
def scan_root(state: State, root_id: FromPath[int]) -> dict:
    """Walk one root and reconcile the library with what is on disk."""
    conn = _connect(state.db_path)
    try:
        path = library.root_path(conn, root_id)
        if path is None:
            raise NotFoundException(f"no root {root_id}")
        result = scan.scan(conn, root_id, path, time.time())
        conn.commit()
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
    engine = JinjaTemplateEngine.from_environment(environment)
    activity.register(engine)
    return engine


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
            told = console.envelope(event)
            if event.get("pending"):
                live_reports[int(event["job_id"])] = told
                frame = "pending"
            else:
                frame = "event"
                if not console.inside_item(event["type"]):
                    live_reports.pop(int(event["job_id"]), None)
            loop.call_soon_threadsafe(channels.publish, {"frame": frame, **told}, "events")

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
            media_view.media_page,
            media_authored.set_favorite,
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
            submit_embed,
            submit_embed_prompts,
            prompt_neighbours,
            search,
            submit_dupes,
            dupes,
            timeline_view.timeline,
            timeline_view.density,
            story_view.stories_index,
            story_view.freeze_snapshot,
            story_view.snapshot_document,
            story_view.plan_snapshot,
            story_view.plan_document,
            story_view.render_plan,
            story_view.render_document,
            story_view.plan_evolution,
            person_view.people_index,
            place_view.places_index,
            person_view.person_page,
            clusterings,
            ways,
            roots,
            add_root,
            scan_root,
            active_jobs,
            job_snapshot,
            submit_verify,
            submit_faces,
            submit_annotate,
            submit_cluster,
            person_view.name_person,
            cancel_job,
            jobs_feed,
            events_feed,
            choose_primary,
            all_settings,
            change_setting,
            media_bytes,
            thumb_bytes,
            preview_bytes,
            avatar_bytes,
            gallery.gallery,
            gallery.grid_fragment,
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
        plugins=[channels, _WorkerPlugin()],
        template_config=TemplateConfig(instance=_template_engine()),
    )
    app.state.home = str(base)
    app.state.db_path = str(where)
    app.state.actor_id = actor_id
    return app
