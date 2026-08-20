"""The application over the schema: every page a query, every sweep a job.

The skeleton of plan Phase 2, held to its two rules. Addresses are entity
slugs resolved through `db.pages.resolve` -- never paths, never raw ids --
and nothing expensive starts by itself: a sweep is a `job` row somebody
POSTs into existence, worked by explicit worker turns, cancellable and
resumable because the row is the truth.

Handlers are synchronous on purpose: sqlite is synchronous, and Litestar
runs sync handlers on its thread pool when told so
(litestar-org/litestar@64cd7da docs/topics/sync-vs-async.rst). Each request
opens its own connection, which is what makes that safe -- sqlite3
connections refuse cross-thread use, and the pool gives no thread pinning.
"""

from __future__ import annotations

import os
import pathlib
import sqlite3
import time

from litestar import Litestar, Request, get, post
from litestar.datastructures import State
from litestar.exceptions import ClientException, NotFoundException
from litestar.response import Redirect, Response, Stream

from db import connect, derived, detect, jobs, library, naming, oriented, pages, runner, scan, settings
from sg_web import home, media


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _rows(cursor_rows, columns) -> list[dict]:
    return [dict(zip(columns, row, strict=True)) for row in cursor_rows]


@get("/health", sync_to_thread=False)
def health() -> str:
    return "ok"


@get("/people", sync_to_thread=True)
def people(state: State) -> list[dict]:
    """Everyone, most pictures first -- the People index."""
    conn = _connect(state.db_path)
    try:
        return _rows(pages.people_by_most(conn), ("name", "slug", "pictures"))
    finally:
        conn.close()


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
        conn.close()


@get("/clusterings", sync_to_thread=True)
def clusterings(state: State) -> list[dict]:
    """Every clustering run held side by side, primary first."""
    conn = _connect(state.db_path)
    try:
        return pages.clusterings(conn)
    finally:
        conn.close()


@get("/ways", sync_to_thread=True)
def ways(state: State) -> list[dict]:
    """What the library can be searched by, generated from what it holds."""
    conn = _connect(state.db_path)
    try:
        return _rows(pages.ways(conn), ("source", "key", "value_kind", "occurrences"))
    finally:
        conn.close()


@get("/jobs", sync_to_thread=True)
def active_jobs(state: State) -> list[dict]:
    conn = _connect(state.db_path)
    try:
        return jobs.active(conn)
    finally:
        conn.close()


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
        conn.close()


@post("/jobs/verify", sync_to_thread=True)
def submit_verify(state: State) -> dict:
    """Ask for an integrity sweep. Nothing runs until a worker turn."""
    conn = _connect(state.db_path)
    try:
        job_id = runner.submit_verify(conn, time.time())
        conn.commit()
        return jobs.snapshot(conn, job_id)
    finally:
        conn.close()


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
        return jobs.snapshot(conn, job_id)
    finally:
        conn.close()


def _file_at(conn, slug: str, where: str) -> tuple[int, str] | Redirect:
    """Resolve a file slug to `(file_id, disk path)`, 301ing retired slugs
    and refusing what is not there to serve."""
    found = naming.resolve(conn, "file", slug)
    if found is None:
        raise NotFoundException(f"no file at {where}/{slug}")
    file_id, is_current = found
    if not is_current:
        live = naming.entity_slug(conn, file_id)
        if live is not None:
            return Redirect(path=f"{where}/{live[1]}", status_code=301)
    row = conn.execute("SELECT missing_since FROM file WHERE id = ?", (file_id,)).fetchone()
    if row is None or row[0] is not None:
        raise NotFoundException(f"{where}/{slug} is not on disk right now")
    path = detect.path_of(conn, file_id)
    if not os.path.isfile(path):
        raise NotFoundException(f"{where}/{slug} is not on disk right now")
    return file_id, path


@get("/media/{slug:str}", sync_to_thread=True)
def media_bytes(state: State, slug: str, request: Request) -> Stream | Redirect | Response:
    """The original bytes, typed by what they are and seekable by range.

    Content-Type comes from the sniff, never the suffix -- the route
    exists to feed decoders and `<video>` elements, and feeding them a
    lie about an MP4 wearing .jpg is how players break. Range semantics
    live in sg_web/media.py.
    """
    conn = _connect(state.db_path)
    try:
        resolved = _file_at(conn, slug, "/media")
        if isinstance(resolved, Redirect):
            return resolved
        _, path = resolved
    finally:
        conn.close()

    from vision import sniff as sniff_module

    size = os.path.getsize(path)
    ctype = sniff_module.content_type(sniff_module.sniff_path(path))
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
        if isinstance(resolved, Redirect):
            return resolved
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
        conn.close()
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
        conn.close()
    return Response(content=target.read_bytes(), media_type="image/webp")


@get("/settings", sync_to_thread=True)
def all_settings(state: State) -> list[dict]:
    """Every setting, its value, default and choices -- the whole vocabulary."""
    conn = _connect(state.db_path)
    try:
        return settings.snapshot(conn)
    finally:
        conn.close()


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
        conn.close()


@post("/jobs/{job_id:int}/cancel", sync_to_thread=True)
def cancel_job(state: State, job_id: int) -> dict:
    """Ask a job to stop. The runner stops it, at an item boundary."""
    conn = _connect(state.db_path)
    try:
        jobs.cancel(conn, job_id)
        conn.commit()
        return jobs.snapshot(conn, job_id)
    finally:
        conn.close()


@post("/worker/turn", sync_to_thread=True)
def worker_turn(state: State, data: dict | None = None) -> dict:
    """One explicit worker turn. `budget` bounds the items it performs;
    a bounded turn leaves the job running and resumable, which is what
    makes progress observable over plain requests."""
    conn = _connect(state.db_path)
    try:
        budget = (data or {}).get("budget")
        turn = runner.run_next(
            conn,
            owner="web-worker",
            now=time.time(),
            budget=int(budget) if budget is not None else None,
        )
        conn.commit()
        return turn if turn is not None else {"state": "idle"}
    finally:
        conn.close()


@get("/roots", sync_to_thread=True)
def roots(state: State) -> list[dict]:
    """Every media directory this library reads, and whether it is
    reachable right now. Media roots are rows, not configuration: any
    number of directories, anywhere, and they travel with the database."""
    conn = _connect(state.db_path)
    try:
        return [{"id": root_id, "path": path, "online": online} for root_id, path, online in library.check_roots(conn)]
    finally:
        conn.close()


@post("/roots", sync_to_thread=True)
def add_root(state: State, data: dict) -> dict:
    """Register a media directory. Nothing is read until a scan is asked
    for -- registering is a statement of intent, not a sweep."""
    conn = _connect(state.db_path)
    try:
        root_id = library.add_root(conn, data["path"], data.get("kind", "library"), time.time())
        conn.commit()
        return {"id": root_id, "path": data["path"]}
    finally:
        conn.close()


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
        conn.close()


@post("/clusterings/choose", sync_to_thread=True)
def choose_primary(state: State) -> dict:
    """Re-rank every run and set the default the People page shows."""
    conn = _connect(state.db_path)
    try:
        chosen = derived.choose_primary(conn)
        conn.commit()
        return {"primary_run": chosen}
    finally:
        conn.close()


def build_app(home_dir: str | None = None) -> Litestar:
    """The application, bound to one home directory (sg_web/home.py).

    With no argument the run lives in `~/.smartgallery`. A database that
    does not exist yet is created from the schema -- a first run needs
    nothing but the command that starts it.
    """
    base = home.home(home_dir)
    where = home.db_path(base)
    if not where.exists():
        fresh = connect.connect(where)
        fresh.executescript(connect.schema_sql())
        fresh.commit()
        fresh.close()
    app = Litestar(
        route_handlers=[
            health,
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
            cancel_job,
            worker_turn,
            choose_primary,
            all_settings,
            change_setting,
            media_bytes,
            thumb_bytes,
            preview_bytes,
            avatar_bytes,
        ],
    )
    app.state.home = str(base)
    app.state.db_path = str(where)
    return app
