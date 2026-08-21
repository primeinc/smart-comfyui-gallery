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
from litestar.exceptions import ClientException, NotFoundException
from litestar.plugins import InitPlugin
from litestar.response import Redirect, Response, Stream

from db import authored, connect, derived, detect, jobs, library, naming, oriented, pages, runner, scan, settings
from sg_web import home, media
from sg_web import worker as worker_module


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
def front(state: State) -> list[dict]:
    """The front page: the library, newest first, addressed by slug."""
    conn = _connect(state.db_path)
    try:
        return _rows(pages.newest(conn), ("slug", "name", "mtime"))
    finally:
        connect.close(conn)


@get("/i/{slug:str}", sync_to_thread=True)
def picture_page(state: State, slug: str) -> dict | Redirect:
    """One picture: its recipe, its parsed fields row by row, its folder
    neighbours, and its place in the derivation graph -- each from the
    query the page owns in db/pages.py, never a grab-bag."""
    conn = _connect(state.db_path)
    try:
        file_id, live = _resolved(conn, "file", slug, "/i")
        if live is not None:
            return Redirect(path=f"/i/{live}", status_code=301)
        told = pages.picture(conn, file_id)
        if told is None:
            raise NotFoundException(f"/i/{slug} has no file row")
        name, folder, width, height, duration, asked_w, checkpoint, _loras_csv, prompt, seed, fields = told
        away = conn.execute("SELECT missing_since FROM file WHERE id = ?", (file_id,)).fetchone()
        return {
            "slug": slug,
            "name": name,
            "folder": folder,
            "present": away is not None and away[0] is None,
            "width": width,
            "height": height,
            "duration": duration,
            "asked_for_width": asked_w,
            "checkpoint": checkpoint,
            "loras": pages.file_loras(conn, file_id),
            "prompt": prompt,
            "seed": seed,
            "fields": fields,
            "params": _rows(pages.fields_of(conn, file_id), ("source", "key", "value")),
            "previous": pages.neighbour(conn, file_id, previous=True),
            "next": pages.neighbour(conn, file_id, previous=False),
            "parents": _rows(pages.parents(conn, file_id), ("slug", "name", "kind")),
            "children": _rows(pages.children(conn, file_id), ("slug", "name", "kind")),
        }
    finally:
        connect.close(conn)


@get("/f/{slug:str}", sync_to_thread=True)
def folder_page(state: State, slug: str) -> dict | Redirect:
    """One folder: its files and its breadcrumb, walked by parent --
    never by splitting a path, because a path is presentation."""
    conn = _connect(state.db_path)
    try:
        folder_id, live = _resolved(conn, "folder", slug, "/f")
        if live is not None:
            return Redirect(path=f"/f/{live}", status_code=301)
        crumbs = []
        for crumb_id, crumb_name in pages.breadcrumb(conn, folder_id):
            addressed = naming.entity_slug(conn, crumb_id)
            crumbs.append({"name": crumb_name, "slug": addressed[1] if addressed else None})
        return {
            "slug": slug,
            "breadcrumb": crumbs,
            "files": _rows(pages.folder_files(conn, folder_id), ("slug", "name")),
        }
    finally:
        connect.close(conn)


#: Which shelf each addressable artifact kind lives on. Kinds outside this
#: map have rows and identity but no page yet.
_SHELVES = {"checkpoint": "/m", "lora": "/l", "workflow": "/w"}


def _shelf_index(state: State, kind: str) -> list[dict]:
    conn = _connect(state.db_path)
    try:
        return _rows(pages.artifacts_by_use(conn, kind), ("name", "slug", "pictures"))
    finally:
        connect.close(conn)


def _artifact_page(state: State, slug: str, shelf: str) -> dict | Redirect:
    """One artifact, on the shelf its kind belongs to. The wrong shelf
    301s to the right one -- an address written down keeps working, and
    one artifact never answers at two."""
    conn = _connect(state.db_path)
    try:
        artifact_id, live = _resolved(conn, "artifact", slug, shelf)
        if live is not None:
            return Redirect(path=f"{shelf}/{live}", status_code=301)
        kind, name = conn.execute("SELECT kind, name FROM artifact WHERE id = ?", (artifact_id,)).fetchone()
        home_shelf = _SHELVES.get(kind)
        if home_shelf is None:
            raise NotFoundException(f"a {kind} has no page yet")
        if home_shelf != shelf:
            return Redirect(path=f"{home_shelf}/{slug}", status_code=301)
        held = (
            pages.workflow_files(conn, artifact_id) if kind == "workflow" else pages.artifact_files(conn, artifact_id)
        )
        page = {
            "slug": slug,
            "name": name,
            "kind": kind,
            "pictures": _rows(held, ("slug", "name")),
        }
        if kind == "lora":
            page["used_with"] = _rows(pages.lora_synergy(conn, artifact_id), ("name", "slug", "together"))
        return page
    finally:
        connect.close(conn)


@get("/models", sync_to_thread=True)
def models(state: State) -> list[dict]:
    """Checkpoints, by how many pictures used them -- a real join, never
    a substring over a blob."""
    return _shelf_index(state, "checkpoint")


@get("/loras", sync_to_thread=True)
def loras(state: State) -> list[dict]:
    return _shelf_index(state, "lora")


@get("/workflows", sync_to_thread=True)
def workflows(state: State) -> list[dict]:
    """Workflows attach through `generation`, so their shelf has its own
    join (db/pages.py WORKFLOWS_BY_USE)."""
    conn = _connect(state.db_path)
    try:
        return _rows(pages.workflows_by_use(conn), ("name", "slug", "pictures"))
    finally:
        connect.close(conn)


@get("/m/{slug:str}", sync_to_thread=True)
def model_page(state: State, slug: str) -> dict | Redirect:
    return _artifact_page(state, slug, "/m")


@get("/l/{slug:str}", sync_to_thread=True)
def lora_page(state: State, slug: str) -> dict | Redirect:
    return _artifact_page(state, slug, "/l")


@get("/w/{slug:str}", sync_to_thread=True)
def workflow_page(state: State, slug: str) -> dict | Redirect:
    return _artifact_page(state, slug, "/w")


@get("/albums", sync_to_thread=True)
def albums(state: State) -> list[dict]:
    conn = _connect(state.db_path)
    try:
        return _rows(pages.albums(conn), ("name", "slug", "kind", "pictures"))
    finally:
        connect.close(conn)


@dataclasses.dataclass
class NewAlbum:
    """The body of POST /albums. A smart collection needs a rule and will
    need its own route; this one makes the listed kinds."""

    name: str
    kind: str = "album"


@post("/albums", sync_to_thread=True)
def make_album(state: State, data: NewAlbum) -> dict:
    conn = _connect(state.db_path)
    try:
        cleaned = data.name.strip()
        if not cleaned:
            raise ClientException("an album needs a name")
        if data.kind not in ("album", "flag"):
            raise ClientException("kind must be album or flag; a smart collection needs a rule")
        collection_id = authored.collection(conn, cleaned, time.time(), kind=data.kind)
        conn.commit()
        addressed = naming.entity_slug(conn, collection_id)
        return {"name": cleaned, "slug": addressed[1] if addressed else None, "kind": data.kind}
    finally:
        connect.close(conn)


@get("/t/{slug:str}", sync_to_thread=True)
def album_page(state: State, slug: str) -> dict | Redirect:
    conn = _connect(state.db_path)
    try:
        collection_id, live = _resolved(conn, "collection", slug, "/t")
        if live is not None:
            return Redirect(path=f"/t/{live}", status_code=301)
        name, kind, sql_text, nl_text = conn.execute(
            "SELECT name, kind, sql_text, nl_text FROM collection WHERE id = ?", (collection_id,)
        ).fetchone()
        page = {
            "slug": slug,
            "name": name,
            "kind": kind,
            "files": _rows(pages.album_files(conn, collection_id), ("slug", "name")),
        }
        if kind == "smart":
            # Served as the rule it is. EVALUATING the rule is deferred
            # work, said here rather than implied: the guards guarantee
            # `files` cannot hold stored members to lie with meanwhile.
            page["rule"] = {"sql": sql_text, "nl": nl_text}
        return page
    finally:
        connect.close(conn)


@dataclasses.dataclass
class AlbumEntry:
    """The body of the album membership routes: a file, by its address."""

    file: str


def _album_membership(state: State, slug: str, data: AlbumEntry, *, adding: bool) -> dict:
    route = f"/t/{slug}/{'add' if adding else 'remove'}"
    conn = _connect(state.db_path)
    try:
        collection_id, live_album = _resolved(conn, "collection", slug, "/t")
        file_id, live_file = _resolved(conn, "file", data.file, route)
        try:
            if adding:
                authored.add_to_collection(conn, collection_id, file_id, time.time())
            else:
                authored.remove_from_collection(conn, collection_id, file_id)
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
def album_add(state: State, slug: str, data: AlbumEntry) -> dict:
    return _album_membership(state, slug, data, adding=True)


@post("/t/{slug:str}/remove", sync_to_thread=True)
def album_remove(state: State, slug: str, data: AlbumEntry) -> dict:
    return _album_membership(state, slug, data, adding=False)


@get("/people", sync_to_thread=True)
def people(state: State) -> list[dict]:
    """Everyone, most pictures first -- the People index."""
    conn = _connect(state.db_path)
    try:
        return _rows(pages.people_by_most(conn), ("name", "slug", "pictures"))
    finally:
        connect.close(conn)


@get("/p/{slug:str}", sync_to_thread=True)
def person(state: State, slug: str) -> dict | Redirect:
    """One person: their pictures, and where those live on disk.

    A retired slug redirects to the live one rather than answering, so one
    person never has two addresses serving content -- the rename contract
    the naming module carries `is_current` for."""
    conn = _connect(state.db_path)
    try:
        found = naming.resolve(conn, "person", slug)
        if found is None:
            raise NotFoundException(f"no person at /p/{slug}")
        person_id, is_current = found
        if not is_current:
            live = naming.entity_slug(conn, person_id)
            if live is not None:
                return Redirect(path=f"/p/{live[1]}", status_code=301)
        name = conn.execute("SELECT name FROM person WHERE id = ?", (person_id,)).fetchone()
        return {
            "slug": slug,
            "name": name[0] if name else None,
            "pictures": _rows(pages.person_files(conn, person_id), ("slug", "name")),
            "across_folders": _rows(
                pages.person_across_folders(conn, person_id),
                ("folder", "folder_slug", "pictures"),
            ),
        }
    finally:
        connect.close(conn)


@dataclasses.dataclass
class NewName:
    """The body of POST /p/{slug}/name. Typed for the same reason NewRoot
    is: a nameless request is a 400 from the signature model."""

    name: str


@post("/p/{slug:str}/name", sync_to_thread=True)
def name_person(state: State, slug: str, data: NewName) -> dict:
    """Name a person -- the People page's primary action.

    The name becomes the address: a new slug is minted and the old one
    retires into history, so the URL somebody saved before the naming
    still answers with a 301 (db/naming.py). And the naming is written
    down as assertions against the person's files -- the durable record a
    re-cluster or a full derived rebuild re-applies the name from, so the
    application cannot destroy what it accepted (db/authored.py)."""
    conn = _connect(state.db_path)
    try:
        found = naming.resolve(conn, "person", slug)
        if found is None:
            raise NotFoundException(f"no person at /p/{slug}")
        person_id, _ = found
        cleaned = data.name.strip()
        if not cleaned:
            raise ClientException("a name needs letters in it")
        now = time.time()
        fresh = authored.name_person(conn, person_id, cleaned, now)
        asserted = authored.assert_named_cluster(conn, person_id, None, now)
        held = conn.execute("SELECT count(*) FROM person_assertion WHERE person_id = ?", (person_id,)).fetchone()[0]
        if held == 0:
            # Refused BEFORE the commit, so nothing above persists: a name
            # with no face to assert it against would be silently lost by
            # the next re-cluster, and the application must not accept
            # what it cannot keep.
            raise ClientException(f"/p/{slug} has no clustered face to keep the name by; nothing was renamed")
        conn.commit()
        return {"slug": fresh, "name": cleaned, "asserted": asserted}
    finally:
        connect.close(conn)


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
def job_snapshot(state: State, job_id: int) -> dict:
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


def _nudge(state: State) -> None:
    """Tell the worker there is work, so pickup is immediate rather than
    on its idle cadence."""
    wake = getattr(state, "worker_wake", None)
    if wake is not None:
        wake.set()


@post("/jobs/verify", sync_to_thread=True)
def submit_verify(state: State) -> dict:
    """Ask for an integrity sweep. The row queues it; the worker drains it."""
    conn = _connect(state.db_path)
    try:
        job_id = runner.submit_verify(conn, time.time())
        conn.commit()
        _nudge(state)
        return jobs.snapshot(conn, job_id)
    finally:
        connect.close(conn)


@post("/jobs/faces", sync_to_thread=True)
def submit_faces(state: State, data: dict) -> dict:
    """Ask for face detection over the library, with the models named."""
    conn = _connect(state.db_path)
    try:
        weights = data.get("models_dir") or str(
            home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir"))
        )
        cache = str(home.thumbs_dir(pathlib.Path(state.home))) if settings.flag(conn, "thumbnail_precache") else None
        job_id = runner.submit_faces(conn, time.time(), models_dir=weights, thumbs_dir=cache)
        conn.commit()
        _nudge(state)
        return jobs.snapshot(conn, job_id)
    finally:
        connect.close(conn)


@post("/jobs/phash", sync_to_thread=True)
def submit_phash(state: State) -> dict:
    """Ask for every present picture's perceptual fingerprint -- the
    identity that survives copies of copies (db/runner.py submit_phash)."""
    conn = _connect(state.db_path)
    try:
        job_id = runner.submit_phash(conn, time.time())
        conn.commit()
        _nudge(state)
        return jobs.snapshot(conn, job_id)
    finally:
        connect.close(conn)


@post("/jobs/ingest", sync_to_thread=True)
def submit_ingest(state: State) -> dict:
    """Ask for every present file's metadata to be read into entities --
    the expensive half of scanning, as a job (db/runner.py submit_ingest)."""
    conn = _connect(state.db_path)
    try:
        job_id = runner.submit_ingest(conn, time.time())
        conn.commit()
        _nudge(state)
        return jobs.snapshot(conn, job_id)
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
        _nudge(state)
        return jobs.snapshot(conn, job_id)
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
    row = conn.execute("SELECT missing_since FROM file WHERE id = ?", (file_id,)).fetchone()
    if row is None or row[0] is not None:
        raise NotFoundException(f"{where}/{slug} is not on disk right now")
    path = detect.path_of(conn, file_id)
    if not os.path.isfile(path):
        raise NotFoundException(f"{where}/{slug} is not on disk right now")
    return file_id, path


@route("/media/{slug:str}", http_method=["GET", "HEAD"], sync_to_thread=True)
def media_bytes(state: State, slug: str, request: Request) -> Stream | Redirect | Response:
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
        kind, sha = conn.execute("SELECT kind, content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()
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
def thumb_bytes(state: State, slug: str) -> Response | Redirect:
    """The grid cell: longest side 512, upright, aspect kept."""
    return _variant_bytes(state, slug, "thumb", "/thumb")


@get("/preview/{slug:str}", sync_to_thread=True)
def preview_bytes(state: State, slug: str) -> Response | Redirect:
    """The lightbox image: longest side 1440, upright, aspect kept."""
    return _variant_bytes(state, slug, "preview", "/preview")


@get("/avatar/{slug:str}", sync_to_thread=True)
def avatar_bytes(state: State, slug: str) -> Response | Redirect:
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
                offset = conn.execute(
                    "SELECT offset_ms FROM derived_media_sample WHERE id = ?", (sample_id,)
                ).fetchone()[0]
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
def change_setting(state: State, key: str, data: dict) -> dict:
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
def cancel_job(state: State, job_id: int) -> dict:
    """Ask a job to stop. The runner stops it, at an item boundary; a
    still-queued job needs a claim to settle, hence the nudge."""
    conn = _connect(state.db_path)
    try:
        jobs.cancel(conn, job_id)
        conn.commit()
        _nudge(state)
        return jobs.snapshot(conn, job_id)
    finally:
        connect.close(conn)


def _active_jobs_of(db_path: str) -> list[dict]:
    conn = _connect(db_path)
    try:
        return jobs.active(conn)
    finally:
        connect.close(conn)


@websocket("/ws/jobs")
async def jobs_feed(socket: WebSocket, channels: ChannelsPlugin, state: State) -> None:
    """Live job progress: the persisted snapshot first, then every delta.

    The subscription opens BEFORE the snapshot is read, so a delta landing
    between the two is queued behind the snapshot instead of lost; a
    client applies deltas onto the snapshot and can never render a state
    the rows did not hold. The channel is transport, never storage --
    reconnection starts from the rows again (db/jobs.py). The snapshot
    read crosses to a thread (anyio.to_thread.run_sync, agronholm/anyio
    src/anyio/to_thread.py:27-52) because sqlite blocks and this handler
    shares the event loop with every open socket.
    """
    from anyio import to_thread

    await socket.accept()
    async with channels.start_subscription("jobs") as subscriber:
        rows = await to_thread.run_sync(_active_jobs_of, state.db_path)
        await socket.send_json({"type": "snapshot", "jobs": rows})
        async with subscriber.run_in_background(socket.send_text):
            while (await socket.receive())["type"] != "websocket.disconnect":
                continue


@get("/roots", sync_to_thread=True)
def roots(state: State) -> list[dict]:
    """Every media directory this library reads, and whether it is
    reachable right now. Media roots are rows, not configuration: any
    number of directories, anywhere, and they travel with the database."""
    conn = _connect(state.db_path)
    try:
        return [{"id": root_id, "path": path, "online": online} for root_id, path, online in library.check_roots(conn)]
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
def scan_root(state: State, root_id: int) -> dict:
    """Walk one root and reconcile the library with what is on disk."""
    conn = _connect(state.db_path)
    try:
        row = conn.execute("SELECT path FROM root WHERE id = ?", (root_id,)).fetchone()
        if row is None:
            raise NotFoundException(f"no root {root_id}")
        result = scan.scan(conn, root_id, row[0], time.time())
        conn.commit()
        return {
            "root": root_id,
            "added": result.added,
            "matched": result.matched,
            "replaced": result.replaced,
            "ambiguous": result.ambiguous,
            "missing": result.missing,
            "hashed": result.hashed,
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
        fresh = connect.connect(where)
        fresh.executescript(connect.schema_sql())
        fresh.commit()
        connect.close(fresh)

    channels = ChannelsPlugin(MemoryChannelsBackend(), channels=["jobs"])

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

        thread = threading.Thread(
            target=worker_module.run, args=(str(where), publish, stop, wake), name="sg-worker", daemon=True
        )
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
            picture_page,
            folder_page,
            models,
            loras,
            workflows,
            model_page,
            lora_page,
            workflow_page,
            albums,
            make_album,
            album_page,
            album_add,
            album_remove,
            submit_ingest,
            submit_phash,
            people,
            person,
            clusterings,
            ways,
            roots,
            add_root,
            scan_root,
            active_jobs,
            job_snapshot,
            submit_verify,
            submit_faces,
            submit_cluster,
            name_person,
            cancel_job,
            jobs_feed,
            choose_primary,
            all_settings,
            change_setting,
            media_bytes,
            thumb_bytes,
            preview_bytes,
            avatar_bytes,
        ],
        plugins=[channels, _WorkerPlugin()],
    )
    app.state.home = str(base)
    app.state.db_path = str(where)
    return app
