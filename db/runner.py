"""Run what was asked for: one claimed job, item by item, off the row.

The row is the truth the whole time. Progress is `done_count` moving,
cancellation is a flag the loop reads between items, resumption is
`job_item` rows still pending, and a killed process leaves nothing to
clean up -- its lease expires and the next `run_next` picks the job up
where the items say it stopped.

Failure is split on purpose. The failures work is EXPECTED to produce --
an unreadable file, a corrupt image, a row that vanished mid-job -- are
recorded on the item and the job carries on. Anything else propagates and
takes the worker turn down with it: the job stays `running`, the lease
runs out, and the work is reclaimed rather than marked broken by a bug in
the code that judged it. A runner that converted every exception into an
item error would turn its own defects into permanent verdicts about files.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sqlite3
import time
import traceback
from collections.abc import Callable
from concurrent import futures

from . import connect, jobs, ledger

_logger = logging.getLogger(__name__)

#: What a handler is allowed to fail with, per item. Everything else is a
#: defect in the handler, not a fact about the item.
#:
#: EOFError is raised where a decoder ran out of input: a fact about the
#: file, never a defect in the code reading it. pillow_heif 1.1.0 raises the
#: builtin out of `Image.load()` ("Decoder plugin generated an error:
#: Unexpected end of file") at db/oriented.py:137 -- outside vision/decode.py,
#: so there is no decoder boundary to translate it at. PyAV's
#: `EOFError(FFmpegError, builtins.EOFError)` (PyAV-Org/PyAV@040da79
#: av/error.pyi:59) is not an OSError either.
#:
#: Measured over ../sg-corpus at 7cf254e: one truncated HEIC ended a scan of
#: 901 items -- the outcome the module docstring above forbids. Three
#: decoders (rawpy, PyAV, pillow_heif) raised outside this tuple; see
#: docs/CORPUS_FINDINGS.md.
ITEM_FAILURES = (OSError, ValueError, RuntimeError, LookupError, EOFError, sqlite3.Error)

#: SQLite saying somebody else is writing. By NAME, not by message: the
#: driver carries the result code as `sqlite_errorname` (Python 3.11+),
#: and the strings behind these are "database is locked" and "database
#: table is locked" (sqlite/sqlite src/main.c:1667-1668) -- which a
#: future release is free to word differently and a matcher on prose
#: would then silently stop recognising.
BUSY = frozenset({"SQLITE_BUSY", "SQLITE_LOCKED"})


# --- the reporting seam ----------------------------------------------------


class Report:
    """What a handler says about the inside of one item: phases, progress,
    observations. Knows nothing of Jinja, Litestar or a socket.

    Two fates for every report. It is SPOKEN at once -- `speak` carries it
    to whoever listens, marked `pending`, with no id, because it describes
    work inside a transaction that may yet roll back -- so a console shows
    "decoding frame 48 of 220" while it is true. And it is KEPT, to be
    written to the ledger at the item boundary in the same commit as the
    item's outcome: a phase an item failed in survives the rollback of the
    item's writes, because the ledger rows are recorded after it. The
    pending message is presentation; the ledger row is history; a client
    that holds both keeps the row.
    """

    def __init__(self, job_id: int, item_id: int | None, clock, speak) -> None:
        self.job_id = job_id
        self.item_id = item_id
        self._clock = clock
        self._speak = speak
        self.kept: list[dict] = []
        self.phase_now: str | None = None
        self.phase_opened = time.perf_counter()

    def _note(self, type_: str, *, phase: str | None, message: str | None, data: dict | None) -> None:
        told = {
            "job_id": self.job_id,
            "at": self._clock(),
            "type": type_,
            "item_id": self.item_id,
            "phase": phase,
            "severity": "info",
            "message": message,
            "data": data,
        }
        self.kept.append(told)
        self._speak({**told, "pending": True})

    def phase(self, name: str, **data) -> None:
        """A named stretch of work begins; the previous one, if any, ends."""
        self._finish()
        self.phase_now = name
        self.phase_opened = time.perf_counter()
        self._note("phase.started", phase=name, message=f"{name} started", data=data or None)

    def _finish(self) -> None:
        """End the open phase, saying how long it took.

        The duration is the point. `phase.finished` used to carry nothing,
        so anything that wanted to know where a job's time went had to
        pair the events up itself and subtract their `at` stamps -- which
        made "what is slow" a question you could only answer by writing a
        program, and every consumer wrote a different one.

        Measured with `perf_counter` rather than the ledger's clock. That
        clock is whatever the turn was given: `time.time` when a worker
        runs, a fixed number in a test that pins it. Subtracting stamps
        from a pinned clock reports every phase as instant, which is a
        plausible-looking zero rather than an obvious absence.
        """
        if self.phase_now is None:
            return
        took = round((time.perf_counter() - self.phase_opened) * 1000, 1)
        self._note(
            "phase.finished",
            phase=self.phase_now,
            message=f"{self.phase_now} finished in {took:g} ms",
            data={"elapsed_ms": took},
        )

    def progress(self, unit: str, done: int, total: int | None = None) -> None:
        """How far the current phase is, in its own unit."""
        spelled = f"{done} {unit}" if total is None else f"{done} / {total} {unit}"
        self._note(
            "phase.progress",
            phase=self.phase_now,
            message=spelled,
            data={"unit": unit, "done": done, "total": total},
        )

    def observe(self, name: str, **data) -> None:
        """A fact the handler found on the way: faces seen, frames sampled."""
        self._note("item.observed", phase=self.phase_now, message=name, data={"name": name, **data})

    def close(self) -> None:
        self._finish()
        self.phase_now = None


class _Silent(Report):
    """The report outside any runner turn: everything said is dropped."""

    def __init__(self) -> None:
        super().__init__(0, None, lambda: 0.0, lambda told: None)

    def _note(self, type_: str, *, phase, message, data) -> None:
        return


_REPORT: contextvars.ContextVar[Report | None] = contextvars.ContextVar("sg_report", default=None)


def report() -> Report:
    """The report for the item being worked on this thread. A handler
    calls `report().phase("decoding")`; outside a turn it gets a report
    that listens and says nothing, so handlers never branch on it."""
    held = _REPORT.get()
    return held if held is not None else _Silent()


def submit_verify(conn, now: float) -> int:
    """An integrity sweep: is every present file still the bytes we recorded?

    One item per file that has a recorded hash. Finds silent corruption and
    out-of-band edits; writes nothing, so a mismatch is a finding on the
    item, never a mutation of the library.
    """
    items = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM file WHERE missing_since IS NULL AND content_sha256 IS NOT NULL ORDER BY id"
        )
    ]
    return jobs.submit(conn, "hash", now, items=items)


def _verify_item(conn, file_id: int, payload: dict, now: float) -> None:
    from . import detect, scan

    told = report()
    told.phase("reading-recorded-hash")
    stored = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()
    if stored is None:
        raise LookupError(f"file {file_id} left the library mid-job")
    if stored[0] is None:
        raise LookupError(f"file {file_id} has no recorded hash to verify against")
    told.phase("hashing-bytes")
    actual = scan.sha256_of(detect.path_of(conn, file_id))
    if actual != stored[0]:
        raise ValueError(f"bytes changed behind the library's back: recorded {stored[0][:12]}, found {actual[:12]}")


def submit_faces(
    conn, now: float, *, models_dir: str, thumbs_dir: str | None = None, everything: bool = False
) -> int | None:
    """Face detection over every present picture and video no detector
    has looked at for its current bytes (derived_face_scan), as one job
    -- or, with `everything`, over all of them again; None when nothing
    is left. ANY backend's pass counts: switching backends and wanting
    the other's answer everywhere is `everything`, said so.

    A face on a video is a face on a sampled frame; the handler routes by
    the file kind, and the schema already keys every face to the moment it
    was seen (`derived_face_instance.sample_id`).

    `thumbs_dir` rides in the payload when thumbnail precaching is on:
    detection decodes every frame anyway, and the cache takes the decoded
    pixels on the way past instead of re-decoding later on first view.

    The `face_backend` and `ort_providers` settings are read HERE, once,
    into the payload: every item of one job runs the same pipeline on the
    same device, whatever the settings say by the time it drains.
    """
    from . import settings as settings_module

    items = face_items(conn, everything=everything)
    if not items:
        return None
    payload: dict = {
        "models_dir": models_dir,
        "backend": settings_module.value(conn, "face_backend"),
        "providers": settings_module.value(conn, "ort_providers"),
    }
    if thumbs_dir is not None:
        payload["thumbs_dir"] = thumbs_dir
    return jobs.submit(conn, "detect_faces", now, payload=payload, items=items)


#: The files a sweep would take, asked as a question rather than
#: answered at submit. A step in a chain has to ask it when it RUNS: the
#: step before it has not gone yet, so the files it will find do not
#: exist and the derivations it will make are not there to be missing.
def face_items(conn, *, everything: bool = False) -> list[int]:
    """Every present picture and video no detector has looked at for its
    current bytes -- or all of them, said so."""
    sql = "SELECT f.id FROM file f WHERE f.missing_since IS NULL AND f.kind IN ('image', 'animated_image', 'video')"
    if not everything:
        sql += (
            " AND NOT EXISTS (SELECT 1 FROM derived_face_scan s WHERE s.file_id = f.id"
            "   AND s.source_sha256 = f.content_sha256)"
        )
    return [row[0] for row in conn.execute(sql + " ORDER BY f.id")]


def caption_items(conn, model: str, *, everything: bool = False) -> list[int]:
    """Every present picture and video this model has not captioned for
    its current bytes -- or all of them, said so."""
    sql = "SELECT f.id FROM file f WHERE f.missing_since IS NULL AND f.kind IN ('image', 'animated_image', 'video')"
    args: tuple = ()
    if not everything:
        sql += (
            " AND NOT EXISTS (SELECT 1 FROM derived_annotation a WHERE a.file_id = f.id AND a.kind = 'caption'"
            "   AND a.model_id = ? AND a.source_sha256 = f.content_sha256)"
        )
        args = (model,)
    return [row[0] for row in conn.execute(sql + " ORDER BY f.id", args)]


def submit_phash(conn, now: float, *, everything: bool = False) -> int | None:
    """Perceptual hashes for every present picture still without one
    taken from its current bytes under the current phash space -- or,
    with `everything`, for all of them again -- as one job; None when
    nothing is left.

    The backfill for a library that never ran detection -- and for every
    video even when it did: detection records hashes as a byproduct only
    for whole still frames (a video's frames are samples, and a sample
    hash is not a file hash), so videos are fingerprinted here or not at
    all. A row recorded against the same bytes counts even when no value
    came of it: that outcome was recorded on purpose. Rides the schema's
    'hash' kind with a payload the handler dispatches on: both jobs are
    about what a file's content IS, one verifying bytes, one
    fingerprinting pixels.
    """
    from . import similarity

    sql = "SELECT f.id FROM file f WHERE f.missing_since IS NULL AND f.kind IN ('image', 'animated_image', 'video')"
    args: tuple = ()
    found = None if everything else similarity._current_space_of(conn, similarity.PHASH)
    if found is not None:
        sql += (
            " AND NOT EXISTS (SELECT 1 FROM derived_file_hash h WHERE h.file_id = f.id AND h.space_id = ?"
            "   AND h.source_sha256 = f.content_sha256)"
        )
        args = (found[0],)
    items = [row[0] for row in conn.execute(sql + " ORDER BY f.id", args)]
    if not items:
        return None
    return jobs.submit(conn, "hash", now, payload={"derive": "perceptual"}, items=items)


def _perceptual_item(conn, file_id: int, payload: dict, now: float) -> None:
    from vision import decode, dupes

    from . import derived, detect, oriented, scan

    told = report()
    kind, sha = conn.execute("SELECT kind, content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()
    path = detect.path_of(conn, file_id)
    told.phase("decoding", kind=kind)
    frame = decode.poster(path) if kind == "video" else oriented.for_model(conn, file_id, path)
    if frame is None:
        raise ValueError(f"file {file_id} has no decodable frame to fingerprint")
    if sha is None:
        told.phase("hashing-bytes")
        sha = scan.sha256_of(path)
    told.phase("fingerprinting")
    phash64, dhash64 = dupes.perceptual(frame)
    told.phase("recording")
    derived.record_hash(conn, file_id, sha, now, phash64=phash64, dhash64=dhash64)


def submit_thumbs(conn, now: float, *, thumbs_dir: str) -> int | None:
    """Render the grid thumb and lightbox preview for every present
    picture and video the cache does not hold yet, as one job.

    The serving layer renders a missing variant on first request: one
    full decode of the original per picture, nine at once under the
    rail's popover -- 3.7s for a page of 22-megapixel PNGs. This job
    pays that decode once, in the background, as files arrive
    (`precache_after_scan`), so a view is a stat and a read. Rides the
    'hash' kind with a payload the handler dispatches on, like the
    perceptual job: it is about what a file's pixels are.

    None when the cache already holds everything: an empty job on the
    feed would announce work that does not exist.
    """
    import pathlib

    from vision import thumbs

    cache = pathlib.Path(thumbs_dir)
    items = [
        file_id
        for file_id, sha in conn.execute(
            "SELECT id, content_sha256 FROM file WHERE missing_since IS NULL"
            " AND kind IN ('image', 'animated_image', 'video') ORDER BY id DESC"
        )
        if sha is None or any(not thumbs.path_for(cache, sha, kind).exists() for kind in thumbs.EDGES)
    ]
    if not items:
        return None
    return jobs.submit(conn, "hash", now, payload={"derive": "thumbs", "thumbs_dir": thumbs_dir}, items=items)


def precache_after_scan(conn, now: float, result, *, thumbs_dir: str) -> int | None:
    """The walk found new bytes: queue their thumbnails, when the
    `thumbnail_precache` setting says the cache is filled ahead of views.
    `result` is the walk's ScanResult (db/scan.py)."""
    from . import settings as settings_module

    if not (result.added or result.replaced) or not settings_module.flag(conn, "thumbnail_precache"):
        return None
    return submit_thumbs(conn, now, thumbs_dir=thumbs_dir)


def _thumbs_alongside(conn, job_id: int, file_id: int, cache) -> list[tuple]:
    """Upcoming items of this job worth rendering beside this one.

    The same bargain `_Ahead` makes for vectors, and a cheaper one. The
    runner still works one item at a time -- started, committed, worked
    and settled on its own -- and what moves is only WHEN the pixels are
    computed. Where a vector has to be held in memory because writing a
    row ahead would not be safe, a thumbnail's result IS a file in a
    content-addressed cache: rendering one early is exactly what the job
    would have produced, and an item whose turn never comes has simply
    got its thumbnail sooner. So nothing is held and nothing is undone
    by a cancel.

    Resolved HERE, on the connection's own thread. sqlite refuses
    cross-thread use, so the pool is handed paths, hashes and
    orientation tags, never a cursor.

    Bounded by megapixels as well as by count for the reason the encoder
    is: cancellation is checked BETWEEN items, so an unbounded group is
    an unbounded wait for somebody who asked the job to stop.

    Skipped: anything with no recorded hash, because the cache is keyed
    on it and hashing ahead is reading a whole file to speculate; and
    video, which is a seek and a decode of a different shape and never
    joins a batch here for the same reason it never joins one there.
    """
    import pathlib

    from vision import thumbs

    from . import detect, oriented

    budget = BATCH_MEGAPIXELS
    found: list[tuple] = []
    room = thumbs_in_flight() - 1
    for item in jobs.pending(conn, job_id):
        if len(found) >= room:
            break
        if item == file_id:
            continue
        row = conn.execute("SELECT kind, content_sha256, width, height FROM file WHERE id = ?", (item,)).fetchone()
        if row is None or row[0] == "video" or row[1] is None:
            continue
        kind, sha = row[0], row[1]
        if kind not in thumbs.PICTURED:
            continue
        if all(thumbs.path_for(cache, sha, variant).exists() for variant in thumbs.EDGES):
            continue
        budget -= _megapixels(row[2:])
        if budget < 0:
            break
        path = detect.path_of(conn, item)
        found.append((sha, pathlib.Path(path), kind, oriented.orientation_of(conn, item)))
    return found


def _thumbs_item(conn, file_id: int, payload: dict, now: float) -> None:
    import pathlib

    from vision import derive, thumbs

    from . import detect, oriented, scan

    told = report()
    cache = pathlib.Path(payload["thumbs_dir"])
    kind, sha = conn.execute("SELECT kind, content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()
    path = detect.path_of(conn, file_id)
    if sha is None:
        told.phase("hashing-bytes")
        sha = scan.sha256_of(path)
    if all(thumbs.path_for(cache, sha, variant).exists() for variant in thumbs.EDGES):
        told.observe("already-cached")
        return
    # derive picks the decoder: libvips for what it reads, Pillow for the
    # rest. Bounded by the largest variant either way, because every
    # smaller one is taken off it rather than off the source again.
    mine = (sha, pathlib.Path(path), kind, oriented.orientation_of(conn, file_id))
    alongside = _thumbs_alongside(conn, told.job_id, file_id, cache) if kind != "video" else []
    if not alongside:
        told.phase("rendering-thumbnails", kind=kind, variants=len(thumbs.EDGES))
        derive.stand_aside()
        derive.put_all(cache, *mine)
        return

    # Reported ONCE, against the item that triggered it, and named so it
    # cannot be read as this item's own cost -- the same accounting
    # `_Ahead` states for batched encoding. The others render no phase of
    # their own, which is true: by their turn the file is already there
    # and they take the `already-cached` return above.
    told.phase("rendering-thumbnails-together", pictures=len(alongside) + 1, variants=len(thumbs.EDGES))

    def rendered(one):
        # A person blocked on a cell outranks a queue guessing at what
        # they will want next, and this fills every core while guessing.
        derive.stand_aside()
        return derive.put_all(cache, *one)

    with futures.ThreadPoolExecutor(max_workers=thumbs_in_flight()) as pool:
        running = {pool.submit(rendered, one): one for one in [mine, *alongside]}
        for done in futures.as_completed(running):
            if running[done] is mine:
                done.result()  # this item's own failure is this item's verdict
                continue
            why = done.exception()
            if why is not None:
                # A picture rendered AHEAD is speculative. Its failure
                # says nothing about the item being worked, and reporting
                # it here would blame the wrong file -- it meets its own
                # failure, attributed to itself, when its turn comes.
                _logger.info("job #%d: rendering ahead skipped a picture (%s)", told.job_id, why)


def submit_embed(conn, now: float, *, models_dir: str, everything: bool = False) -> list[int]:
    """The joint image/text embedding for every present picture still
    without a CURRENT vector in a space -- or, with `everything`, for
    all of them again. ONE JOB PER participating space: each provider's
    items commit in their own transactions, so a model that fails to
    load or encode costs its own space's progress and nobody else's; a
    space with nothing left to embed gets no job. A bad `semantic_model`
    setting is refused here, not queued.

    Current means what retrieval means by it (db/retrieval.py
    current_rows): a vector computed from the file's present bytes. A
    space nothing has minted yet -- weights never loaded -- has every
    picture as an item; the space it resolves to is looked up at the
    checkpoint the shared cache pins today, the way retrieval looks it
    up.
    """
    from vision import semantic

    from . import retrieval

    present = "SELECT f.id FROM file f WHERE f.missing_since IS NULL AND f.kind IN ('image', 'animated_image', 'video')"
    made = []
    for provider, model, configured in retrieval.choices(conn):
        found = None
        if not everything:
            checkpoint = semantic.pin(provider, models_dir, model, configured)
            found = retrieval._space_of(conn, provider, model, checkpoint)
        if found is None:
            items = [row[0] for row in conn.execute(present + " ORDER BY f.id")]
        else:
            items = [
                row[0]
                for row in conn.execute(
                    present + " AND NOT EXISTS (SELECT 1 FROM derived_embedding e WHERE e.file_id = f.id"
                    "   AND e.space_id = ? AND e.source_sha256 = f.content_sha256) ORDER BY f.id",
                    (found[0],),
                )
            ]
        if not items:
            continue
        payload = {"models_dir": models_dir, "choice": [provider, model, configured]}
        made.append(jobs.submit(conn, "embed", now, payload=payload, items=items))
    return made


#: How many thumbnails to render at once, and why this number.
#:
#: libvips already uses every core to calculate ONE image
#: (../refs/libvips/libvips/doc/using-threads.md, "Threads"), so the
#: obvious reading is that a second file in flight can only fight the
#: first. Measured, that is wrong: one thumbnail is not enough work to
#: fill sixteen cores, and the win is across files rather than inside
#: one. On 32 pictures of 4000x3000, per file:
#:
#:      1 in flight    4.70 files/sec   1.0x
#:      2              9.72             2.1x
#:      4             17.99             3.8x
#:      8             28.20             6.0x   <- the knee
#:     12             27.74             5.9x
#:     16             25.47             5.4x
#:
#: Past the knee the two thread pools oversubscribe and it gets slower,
#: so this is half the cores rather than all of them -- the measured
#: best on a 16-core machine -- and never fewer than two, because one
#: would silently turn the whole thing back off. libvips is documented
#: thread-safe for this: images are immutable and shareable, and only
#: the drawing operators and Regions are not.
def thumbs_in_flight() -> int:
    return max(2, min(8, (os.cpu_count() or 2) // 2))


#: How many source megapixels one batch may decode. The count alone is
#: not a bound: 64 pictures is 66 megapixels of generated PNG and 1400 of
#: camera raw, and only one of those fits in memory or in a reasonable
#: wait. Set so a batch of ordinary 1-2 MP generated images fills on
#: count while a batch of raws fills on size, and the item leading either
#: one stays responsive to a cancel.
BATCH_MEGAPIXELS = 160.0


class _Ahead:
    """Vectors computed for items the job has not reached yet.

    The runner works one item at a time and must keep doing so: an item
    is started, committed, worked, and settled on its own, which is what
    makes a job resumable, cancellable at a boundary, and able to fail
    one picture without losing the rest. None of that changes here.

    What changes is WHEN the arithmetic happens. An encoder given one
    picture at a time was measured leaving the GPU idle 74% of a real
    job while paying a kernel launch and a copy back per picture. So
    when an item finds no vector waiting, it takes the next `BATCH` of
    the job's own pending items, encodes them together, and keeps the
    rest here. The item it was asked about commits exactly as before;
    the others are simply already done when their turn comes.

    Nothing durable is written ahead. A cancelled or paused job discards
    whatever is held, which costs the encode and nothing else -- vectors
    are recomputable by definition, which is why this is safe to do
    speculatively and a database write would not be.

    Keyed by space as well as job because one library can embed into
    several spaces, and a vector from one is not a vector for another.
    """

    def __init__(self) -> None:
        self._held: dict[tuple[int, str], dict[int, object]] = {}

    def take(self, job_id: int, space, file_id: int):
        """The vector held for this file, by THIS job, in THIS space.

        Keyed exactly. An earlier version searched every held job for a
        matching space and file, which made the lookup contract
        `space + file` while the storage key and the docstring both said
        `job + space + file`. Two jobs over overlapping files in one
        space would have crossed, and the reason it would usually have
        looked fine -- the same file in the same space encodes to nearly
        the same vector -- is exactly what would have kept it hidden.
        """
        return self._held.get((job_id, space.key), {}).pop(file_id, None)

    def forget(self, job_id: int) -> None:
        """Drop everything held for a job that has stopped."""
        for key in [key for key in self._held if key[0] == job_id]:
            del self._held[key]

    def fill(self, conn, told, encoder, space, file_id: int, media):
        """Encode this item, and as many upcoming ones as fit in a batch.

        Falls back to the single-picture path whenever a batch cannot be
        formed or an adapter has no batched entry point. It also falls
        back when a batch RAISES: a failure inside a group of pictures
        says nothing about which one was bad, and the runner's contract
        is that a failure is a verdict on the item it was reported for.
        Encoding this one alone either reproduces the failure -- and it
        is then honestly this item's -- or succeeds, and the bad picture
        is met on its own turn.
        """
        from . import detect, oriented

        batch = getattr(encoder, "encode_many", None)
        together = []
        # Video's representative frame is a seek and a decode of a
        # different shape, so a video item never leads a batch and never
        # joins one; it stays on the single path.
        if batch is not None and media.kind != "video":
            # Resolved HERE, on the connection's own thread. sqlite
            # refuses cross-thread use, so the workers are handed paths
            # and orientation tags and never a cursor. Including the
            # TRIGGERING item's: handing `media.frame` to a worker looks
            # right and is not, because that closure reads `capture` for
            # the orientation tag.
            upcoming = [item for item in jobs.pending(conn, told.job_id) if item != file_id]
            # The item that LEADS the batch is charged first. It is in the
            # batch and it decodes like any other member, so leaving it
            # out made the stated bound a bound on the followers: a 100
            # megapixel leader and 150 of followers formed a 250
            # megapixel batch under a limit of 160.
            mine = conn.execute("SELECT width, height FROM file WHERE id = ?", (file_id,)).fetchone()
            budget = BATCH_MEGAPIXELS - _megapixels(mine)
            for item in upcoming[: openclip_batch() - 1]:
                row = conn.execute("SELECT kind, width, height FROM file WHERE id = ?", (item,)).fetchone()
                if row is None or row[0] == "video":
                    continue
                # Bounded by PIXELS as well as by count. Sixty-four
                # 22-megapixel raws is a gigabyte and a half of decode
                # inside one item, and the item that leads a batch pays
                # for all of it: cancellation is checked BETWEEN items,
                # so an unbounded batch is also an unbounded wait for
                # somebody who asked the job to stop.
                #
                # A leader already over the whole budget still encodes,
                # alone: one picture is the smallest batch there is, and
                # refusing it would be refusing to embed a large file.
                budget -= _megapixels(row[1:])
                if budget < 0:
                    break
                together.append((item, detect.path_of(conn, item), oriented.orientation_of(conn, item)))

        if batch is None or not together:
            told.phase("encoding", kind=media.kind)
            return encoder.encode_media(media)

        def framer_for(path, orientation):
            return lambda: oriented.open_upright(path, orientation)

        mine = framer_for(media.path, oriented.orientation_of(conn, file_id))
        framers = [mine, *[framer_for(path, tag) for _item, path, tag in together]]
        # Reported ONCE, against the item that triggered it, and named so
        # it cannot be read as that item's own cost: `pictures` says how
        # many it covered. The other items in the batch report no encode
        # phase at all, which is true -- they performed none.
        #
        # The alternative, writing the batch's duration against all 64
        # item ledgers, would attribute 64 x 50 ms to a kernel that took
        # 50 ms. This project has already produced one 184%-of-wall-clock
        # table by double-counting; a second would be a choice.
        told.phase("batch-encoding", pictures=len(framers))
        try:
            vectors = batch(framers)
        except (OSError, ValueError) as why:
            # Narrower than ITEM_FAILURES on purpose. What a batch may
            # legitimately fail with is what a bad PICTURE raises -- an
            # unreadable file, a corrupt image -- and the honest response
            # is to encode this one alone so the failure is attributed to
            # whichever item actually owns it.
            #
            # ITEM_FAILURES includes sqlite3.Error, and catching that
            # here turned a threading DEFECT into a silent fallback:
            # every batch raised ProgrammingError, every batch was
            # discarded, every item was then encoded alone, and the job
            # ran four times slower than before with nothing in the log
            # above INFO. A defect must propagate and take the turn down,
            # which is the rule this module states at the top.
            _logger.warning(
                "job #%d: a batch of %d failed (%s); encoding this item alone",
                told.job_id,
                len(framers),
                why,
            )
            told.phase("encoding", kind=media.kind)
            return encoder.encode_media(media)

        held = self._held.setdefault((told.job_id, space.key), {})
        for (item, _path, _tag), vector in zip(together, vectors[1:], strict=True):
            held[item] = vector
        return vectors[0]


def _megapixels(size) -> float:
    """A file row's `(width, height)` in megapixels.

    One megapixel when the row does not say, which is the case for a file
    scanned but not yet ingested. Counting an unknown as nothing would
    let a batch of them past a bound meant to hold it.
    """
    if not size or not size[0] or not size[1]:
        return 1.0
    return size[0] * size[1] / 1e6


def openclip_batch() -> int:
    from vision.semantic import openclip

    return openclip.BATCH


class _Said:
    """Captions computed for items the job has not reached yet.

    The same bargain `_Ahead` makes for vectors, for the same reason: the
    runner still works one item at a time -- started, committed, worked
    and settled on its own -- and what changes is only WHEN the model
    runs. An item that finds no caption waiting takes the next few of the
    job's own pending items, captions them together, and keeps the rest
    here.

    Nothing durable is written ahead. A cancelled job discards what is
    held, which costs the forward pass and nothing else: a caption is
    regenerable by definition, which is what makes this safe to do
    speculatively and a database write not.

    Keyed by job alone, unlike vectors: one job carries one
    `caption_model` in its payload (db/runner.py `submit_annotate` reads
    the setting once), so there is no second axis for two answers about
    one file to cross on.
    """

    def __init__(self) -> None:
        self._held: dict[int, dict[int, str]] = {}

    def take(self, job_id: int, file_id: int) -> str | None:
        return self._held.get(job_id, {}).pop(file_id, None)

    def keep(self, job_id: int, said: dict[int, str]) -> None:
        self._held.setdefault(job_id, {}).update(said)

    def forget(self, job_id: int) -> None:
        self._held.pop(job_id, None)


def caption_batch() -> int:
    from vision import captions

    return captions.BATCH


#: Per process. A job's held vectors are dropped when its turn ends.
_ahead = _Ahead()
#: The same, for captions.
_said = _Said()


def _embed_item(conn, file_id: int, payload: dict, now: float) -> None:
    import functools

    from vision import decode, semantic

    from . import derived, detect, oriented, scan

    kind, sha = conn.execute("SELECT kind, content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()
    path = detect.path_of(conn, file_id)
    if sha is None:
        # The staleness contract keys on the file's recorded bytes, so the
        # hash computed here is persisted the way detection persists its
        # own (db/detect.py) -- an embedding of bytes the file row cannot
        # vouch for would be excluded from retrieval as unverifiable.
        sha = scan.sha256_of(path)
        conn.execute("UPDATE file SET content_sha256 = ? WHERE id = ?", (sha, file_id))

    @functools.cache
    def representative_frame():
        # Its own phase, because it is lazy and therefore lands wherever
        # the adapter happens to ask for it. Without this the decode was
        # billed to "encoding" -- 92% of the job under a name that made
        # it look like the model's fault, when the model is a fifth of it.
        # Named from inside so an adapter that never asks for a frame
        # never reports a decode it did not do.
        told = report()
        resuming = told.phase_now
        told.phase("decoding", kind=kind)
        try:
            frame = decode.poster(path) if kind == "video" else oriented.for_model(conn, file_id, path)
        finally:
            if resuming is not None:
                told.phase(resuming)
        if frame is None:
            raise ValueError(f"file {file_id} has no decodable frame to embed")
        return frame

    told = report()
    # The adapter names its own stages through this. Handed in rather
    # than reached for: the reporter lives here and vision must not
    # import db (db/oriented.py already imports vision/decode).
    media = semantic.MediaRef(path=str(path), kind=kind, frame=representative_frame, phase=told.phase)
    provider, model, checkpoint = payload["choice"]
    told.phase("loading-encoder", provider=provider, model=model)
    encoder = semantic.encoder(provider, payload["models_dir"], model, checkpoint)
    space = encoder.space()

    vector = _ahead.take(told.job_id, space, file_id)
    if vector is None:
        vector = _ahead.fill(conn, told, encoder, space, file_id, media)
    told.phase("recording", space=str(space))
    derived.record_embedding(conn, file_id, space, vector, sha, now)


def submit_dupes(conn, now: float) -> int:
    """Group perceptually identical pictures, as one job of one unit.

    Reads what the phash job (or detection's byproduct) recorded and
    writes derived_dupe_group wholesale -- the first space of the unified
    FAISS index: perceptual, 64 binary bits, hamming. The threshold is
    the `dupe_threshold` setting, validated here so a bad value is a
    refused submit, never a job that fails later.
    """
    from . import settings as settings_module

    raw = settings_module.value(conn, "dupe_threshold")
    try:
        threshold = int(raw)
    except ValueError as bad:
        raise ValueError(f"dupe_threshold must be a number of bits, not {raw!r}") from bad
    # 31, not 64: two unrelated 64-bit hashes disagree on 32 bits on
    # average, so radius 32 admits the average random pair -- range_search
    # would materialize the O(n^2) all-pairs result and wedge the job on a
    # MemoryError no item failure catches. The dial stops before the cliff.
    if not 0 <= threshold <= 31:
        raise ValueError(f"dupe_threshold must be 0..31 bits, not {threshold}: at 32 random pairs match")
    verify_raw = settings_module.value(conn, "dupe_dhash_verify")
    if verify_raw.strip().lower() == "off":
        verify = None
    else:
        try:
            verify = int(verify_raw)
        except ValueError as bad:
            raise ValueError(f"dupe_dhash_verify must be a number of bits or 'off', not {verify_raw!r}") from bad
        if not 0 <= verify <= 63:
            raise ValueError(f"dupe_dhash_verify must be 0..63 bits or 'off', not {verify}")
    return jobs.submit(
        conn, "hash", now, payload={"derive": "groups", "threshold": threshold, "dhash_verify": verify}, items=[0]
    )


def warm_similarity(conn, now: float) -> None:
    """Boot: make the hot spaces resident -- restore from snapshots when
    they match, rebuild once when they do not. After this, jobs mutate
    and query live indexes without a build step.

    Only rows belonging to each CURRENT space are loaded: after a
    producer or preprocess upgrade the current spec resolves to a new
    immutable space id and old rows stop being input -- they are never
    reindexed under the new identity."""
    from . import similarity

    manager = similarity.manager_for(conn)
    sid = similarity.space_id(conn, similarity.PHASH, now)
    rows = dict(
        conn.execute(
            "SELECT h.file_id, h.value FROM derived_file_hash h"
            " JOIN file f ON f.id = h.file_id AND f.missing_since IS NULL"
            " WHERE h.value IS NOT NULL AND h.space_id = ?",
            (sid,),
        )
    )
    if rows:
        similarity.align(conn, manager, similarity.PHASH, sorted(rows), lambda wanted: [rows[v] for v in wanted], now)
    for model_id, model_version in conn.execute(
        "SELECT DISTINCT model_id, model_version FROM derived_face_instance WHERE embedding IS NOT NULL"
    ):
        current = similarity.face_space_of(conn, model_id, model_version)
        if current is None:
            continue
        face_sid, space = current
        ids = [
            row[0]
            for row in conn.execute(
                "SELECT id FROM derived_face_instance WHERE space_id = ? ORDER BY id",
                (face_sid,),
            )
        ]
        if ids:
            similarity.align(conn, manager, space, ids, lambda wanted: _face_vectors(conn, wanted), now)
    from . import retrieval

    for provider, model, checkpoint in retrieval.choices(conn):
        # Probed with the checkpoint as configured. A provider whose
        # checkpoint is a mutable ref mints its spaces under the PINNED
        # commit (vision/semantic seam `pin`), which this boot-time warm
        # cannot resolve without a models_dir -- those spaces warm on
        # the first query instead, the same "slower first use, never a
        # refusal" contract a failed warm already has.
        found = retrieval._space_of(conn, provider, model, checkpoint)
        if found is None:
            continue
        sem_sid, space = found
        rows = retrieval.current_rows(conn, sem_sid)
        if rows:
            ids = [embedding_id for embedding_id, _ in rows]
            similarity.align(conn, manager, space, ids, lambda wanted: retrieval._vectors(conn, wanted), now)


def _face_vectors(conn, wanted):
    """Embedding blobs for exactly these face ids, in their order."""
    import numpy as np

    held = {}
    batch = [int(v) for v in wanted]
    for start in range(0, len(batch), connect.PARAM_BATCH):
        piece = batch[start : start + connect.PARAM_BATCH]
        marks = ",".join("?" for _ in piece)
        for face_id, blob in conn.execute(
            f"SELECT id, embedding FROM derived_face_instance WHERE id IN ({marks})", piece
        ):
            held[face_id] = np.frombuffer(blob, dtype=np.float32)
    return np.vstack([held[v] for v in batch])


def _dupe_groups_item(conn, item: int, payload: dict, now: float) -> None:
    """One global pass: align the perceptual space with the rows SQLite
    holds, cut its hamming pair graph, verify each candidate pair with
    the independent dHash space, union-find over the survivors, groups
    of two or more written with the policy-picked best member -- most
    pixels, then most bytes, then the earliest identity.

    The verification is a second opinion, not a second vote: pHash sees
    global low-frequency composition and proposes; dHash sees local
    gradient structure and vetoes a pair whose structure disagrees by
    more than the payload's `dhash_verify` bits. A pair either file
    cannot be verified for (no dHash row in the current space) passes
    unverified -- the verifier narrows, it never invents absence."""
    from vision import dupes

    from . import similarity

    def _pixels(conn, row, file_id: int) -> int:
        """The member's resolution, the primary fidelity axis. The file
        row knows it once ingest has run; until then the decoder
        answers from the media's own headers, whatever the kind -- byte
        size must NEVER stand in for it, because bytes measure
        compression: a 48px JPEG outweighs a lossless original of the
        same picture at four times the pixels."""
        from vision import decode

        from . import detect

        width, height = row[2], row[3]
        if width and height:
            return int(width) * int(height)
        found = decode.dimensions(detect.path_of(conn, file_id), row[5])
        return found[0] * found[1] if found is not None else 0

    told = report()
    threshold = int(payload["threshold"])
    verify = payload.get("dhash_verify")
    told.phase("reading-fingerprints", threshold=threshold, dhash_verify=verify)
    rows = conn.execute(
        "SELECT h.file_id, h.value, f.width, f.height, f.size, f.kind FROM derived_file_hash h"
        " JOIN file f ON f.id = h.file_id AND f.missing_since IS NULL"
        " WHERE h.value IS NOT NULL AND h.space_id = ? ORDER BY h.file_id",
        (similarity.space_id(conn, similarity.PHASH, now),),
    ).fetchall()
    conn.execute("DELETE FROM derived_dupe_group")
    if len(rows) < 2:
        return

    by_id = {row[0]: row for row in rows}
    told.phase("aligning-space", fingerprints=len(rows))
    manager = similarity.manager_for(conn)
    key = similarity.align(
        conn, manager, similarity.PHASH, sorted(by_id), lambda wanted: [by_id[v][1] for v in wanted], now
    )
    told.phase("cutting-pair-graph")
    twins_a, twins_b, _distances = similarity.pair_graph(manager, key, threshold)
    told.observe("candidate-pairs", count=len(twins_a))

    structure: dict[int, int] = {}
    if verify is not None:
        structure = dict(
            conn.execute(
                "SELECT file_id, value FROM derived_file_hash WHERE value IS NOT NULL AND space_id = ?",
                (similarity.space_id(conn, similarity.DHASH, now),),
            )
        )

    def agreed(a: int, b: int) -> bool:
        if verify is None or a not in structure or b not in structure:
            return True
        return dupes.hamming(structure[a], structure[b]) <= verify

    parent = {file_id: file_id for file_id in by_id}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(twins_a, twins_b, strict=True):
        if not agreed(int(a), int(b)):
            continue
        rooted_a, rooted_b = find(int(a)), find(int(b))
        if rooted_a != rooted_b:
            parent[rooted_b] = rooted_a

    from . import authored

    # Read once for the sweep: a verdict per candidate pair would be a
    # query per pair over a library where the pairs are the expensive part.
    rejected = authored.rejected_pairs(conn)
    grouped: dict[int, list[int]] = {}
    for file_id in by_id:
        grouped.setdefault(find(file_id), []).append(file_id)
    told.phase("writing-groups", candidates=sum(1 for m in grouped.values() if len(m) >= 2))
    for members in grouped.values():
        if len(members) < 2:
            continue
        # A group means "every member is a duplicate of the best" -- so
        # every member is checked against the BEST, not just against
        # whichever neighbour union-find walked in through. Chains are
        # real: A~B and B~C within threshold with A and C far apart put
        # two admittedly-different pictures in one "duplicate" group.
        # A member the canonical checks reject is dropped -- related,
        # perhaps, but not a duplicate this pass can claim.
        best = max(members, key=lambda m: (_pixels(conn, by_id[m], m), by_id[m][4], -m))
        kept = [
            member
            for member in members
            if member == best
            or (
                dupes.hamming(by_id[member][1], by_id[best][1]) <= threshold
                and agreed(member, best)
                # And not a pair somebody has already said is not one
                # picture. A group is a GUESS -- pHash sees composition,
                # so two photographs of one scene a second apart are
                # close in it -- and a correction that survived only
                # until the next sweep would be a chore repeated for
                # ever (db/authored.py `reject_duplicate`).
                and (min(member, best), max(member, best)) not in rejected
            )
        ]
        if len(kept) < 2:
            continue
        seed = min(kept)
        for member in kept:
            conn.execute(
                "INSERT INTO derived_dupe_group(file_id, group_id, distance, threshold, is_best, verified,"
                " computed_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (
                    member,
                    seed,
                    dupes.hamming(by_id[member][1], by_id[best][1]),
                    threshold,
                    1 if member == best else 0,
                    1 if (verify is not None and member in structure and best in structure) else 0,
                    now,
                ),
            )


def _hash_item(conn, file_id: int, payload: dict, now: float) -> None:
    """The 'hash' kind's modes, told apart by payload: a bare job is
    the integrity sweep it always was, so jobs queued before the payload
    existed keep meaning what they meant."""
    derive = payload.get("derive")
    if derive == "perceptual":
        _perceptual_item(conn, file_id, payload, now)
    elif derive == "thumbs":
        _thumbs_item(conn, file_id, payload, now)
    elif derive == "groups":
        _dupe_groups_item(conn, file_id, payload, now)
    elif derive is None:
        _verify_item(conn, file_id, payload, now)
    else:
        raise ValueError(f"unknown derive {derive!r} -- refusing to guess which job this is")


#: One folder and everything under it. The same walk db/pages.py uses
#: for a folder's span and places, so "this folder" means one thing.
_UNDER = (
    "WITH RECURSIVE sub(id) AS (SELECT ? UNION ALL SELECT c.id FROM folder c JOIN sub ON c.parent_id = sub.id"
    "  WHERE c.missing_since IS NULL)"
)


def submit_ingest(conn, now: float, *, everything: bool = False, folder_id: int | None = None) -> int | None:
    """Read every present file's own story, as one job.

    The walk (POST /roots/{id}/scan) finds files cheaply; this is the
    expensive half of scanning that turns each file's metadata into
    entities -- models, LoRAs, prompts, generation settings, capture
    facts, learned param keys. The schema's job kind for it is 'scan'.

    For what is missing: a file whose last read was of its current bytes
    (`file.ingested_sha256`) is not an item again; `everything` reads all
    of them -- the way to catch bytes that rotted behind the scanner's
    back. None when nothing is left.

    `folder_id` bounds it to one folder AND EVERYTHING UNDER IT, which
    is what makes `everything` usable at all on a real library.
    Re-reading is how this application corrects itself -- improving a
    parser is a re-parse, and the sniffer that decides a file's KIND is
    the part most likely to improve -- but "re-read all eighty thousand
    files" is a cost nobody pays to fix one folder of album tracks. A
    correction that is too expensive to apply is not a correction.
    """
    items = ingest_items(conn, everything=everything, folder_id=folder_id)
    if not items:
        return None
    return jobs.submit(conn, "scan", now, items=items)


def ingest_items(conn, *, everything: bool = False, folder_id: int | None = None) -> list[int]:
    """The files an ingest sweep would read, asked when somebody asks."""
    from .ingest import READER

    where = "WHERE missing_since IS NULL"
    args: list = []
    if not everything:
        # Stale by BYTES or by READER. The second is what makes a fixed
        # parser repair the library on its own: every file read by an
        # older reader is due, and the ordinary sweep -- the one a worker
        # already runs for what is missing -- picks them up with nobody
        # asked to do anything.
        where += " AND (ingested_sha256 IS NULL OR ingested_sha256 IS NOT content_sha256 OR ingested_by IS NOT ?)"
        args.append(READER)
    if folder_id is None:
        sql = f"SELECT id FROM file {where} ORDER BY id"
    else:
        # The folder is bound BEFORE the freshness arguments, because the
        # CTE that names the subtree comes first in the statement.
        args = [folder_id, *args]
        # The subtree, not the one folder: somebody pointing at `music`
        # means the albums inside it, and a scope that stopped at the
        # top level would silently do a fraction of what was asked.
        sql = f"{_UNDER} SELECT f.id FROM sub JOIN file f ON f.folder_id = sub.id {where} ORDER BY f.id"
    return [row[0] for row in conn.execute(sql, args)]


def _ingest_item(conn, file_id: int, payload: dict, now: float) -> None:
    from . import detect, ingest

    report().phase("reading-metadata")
    out = ingest.one(conn, file_id, detect.path_of(conn, file_id), now)
    if out.unreadable is not None:
        # Ingest retracts what it wrote last time BEFORE it re-reads, in
        # this same transaction -- so a file whose bytes cannot be opened
        # must RAISE here: the runner's rollback then restores the recipe
        # a dead read could not replace, and the failure lands on the
        # item instead of committing the destruction as success.
        raise ValueError(out.unreadable)


def chosen_threshold(conn) -> float | None:
    """The operating point somebody asked for, or None for the measured one.

    Validated HERE, at submit, so a bad value is a refused submit rather
    than a job that fails on its third item -- the rule `dupe_threshold`
    follows above.

    The bounds are wide on purpose. This is somebody's own library and
    the whole reason to expose the knob is to let them find out what a
    different operating point does; the reason it is safe to let them is
    that a new threshold writes a NEW run beside the old one
    (schema.sql derived_face_run_identity), so nothing they had is
    overwritten by finding out. What is refused is the range where the
    answer is not interesting but broken: at 0 every face is everyone,
    and at 1 nobody is anybody.
    """
    from . import settings as settings_module

    raw = settings_module.value(conn, "face_cluster_threshold").strip().lower()
    if raw in ("", "auto"):
        return None
    try:
        threshold = float(raw)
    except ValueError as bad:
        raise ValueError(f"face_cluster_threshold must be a cosine similarity or 'auto', not {raw!r}") from bad
    if not 0.0 < threshold < 1.0:
        raise ValueError(
            f"face_cluster_threshold must be between 0 and 1 exclusive, not {threshold}: "
            "at 0 every face is the same person and at 1 no two faces ever are"
        )
    return threshold


def submit_cluster(conn, now: float) -> int:
    """Group every embedding space's faces into people, as one job.

    ONE item, and the spaces are found when it RUNS rather than now.

    They used to be enumerated here, one item each. That is correct for
    a job somebody presses on its own and wrong for a step in a chain,
    which is the shape this now has to work in: queued behind
    `detect_faces`, the spaces do not exist yet, so the enumeration
    found none, the job queued zero items, and it settled `done` having
    clustered nothing.

    That is the exact failure the ordering exists to prevent -- a
    library with no people in it and no row that looks wrong -- and
    putting the steps in the right order does not fix it if the later
    step decided what it was going to do before the earlier one ran.

    The operating point is still pinned HERE, at submit. It is a setting
    a person changes, not a fact about the library: reading it per space
    would let a change mid-job give two spaces two answers inside one
    run whose row records a single threshold for both.
    """
    return jobs.submit(conn, "cluster_faces", now, payload={"threshold": chosen_threshold(conn)}, items=[0])


def _cluster_item(conn, index: int, payload: dict, now: float) -> None:
    """Cluster every embedding space, and give every group a person.

    The spaces are read HERE, when the item runs: a step queued behind
    face detection cannot know at submit time which spaces will exist by
    the time it is claimed. `payload["spaces"]` is still honoured, so a
    job queued by an older build keeps meaning what it meant.

    The run's whole answer is replaced -- clusters, inferred appearances,
    and the placeholder people minted for groups nobody has named. Names
    are never carried across by similarity: `seed_clusters_from_assertions`
    re-applies them from what a human wrote down, and only the groups
    still unnamed after that get a fresh unnamed person, addressable at
    `/p/person-<short-id>` until somebody names them.
    """
    named = payload.get("spaces")
    if named is None:
        named = [
            [str(model_id), str(model_version)]
            for model_id, model_version in conn.execute(
                "SELECT DISTINCT model_id, model_version FROM derived_face_instance"
                " WHERE embedding IS NOT NULL ORDER BY model_id, model_version"
            )
        ]
        if not named:
            # An answer, not an error: "cluster a library nothing has
            # looked at" is a thing somebody can ask for.
            report().phase("clustering", spaces=0)
            return
    for space in named:
        _cluster_space(conn, space, payload, now)


def _cluster_space(conn, space, payload: dict, now: float) -> None:
    from . import derived, naming

    told = report()
    model_id, model_version = space
    # Method and threshold pinned once and passed to BOTH calls: recomputing
    # the run identity from separately-spelled defaults is how a drift makes
    # the DELETE below clear a different run's attributions.
    # What somebody asked for, else what was measured for this embedder.
    # `.get`, because a job queued before this setting existed has no
    # such key and must still cluster at the measured point.
    asked = payload.get("threshold")
    pinned = derived.threshold_for(model_id) if asked is None else float(asked)
    told.phase("clustering", model_id=model_id, model_version=model_version, threshold=pinned)
    derived.cluster(conn, model_id, model_version, now, method=derived.DEFAULT_METHOD, threshold=pinned)
    run_id = derived.run_for(conn, model_id, model_version, derived.DEFAULT_METHOD, pinned, now)
    told.phase("naming-groups", run_id=run_id)
    conn.execute("DELETE FROM derived_file_person WHERE run_id = ?", (run_id,))
    derived.seed_clusters_from_assertions(conn, run_id)
    unnamed = conn.execute(
        "SELECT id FROM derived_face_cluster WHERE run_id = ? AND person_id IS NULL ORDER BY id",
        (run_id,),
    ).fetchall()
    faces, clusters = conn.execute("SELECT faces, clusters FROM derived_face_run WHERE id = ?", (run_id,)).fetchone()
    told.observe("clusters", faces=faces, clusters=clusters, named=clusters - len(unnamed), unnamed=len(unnamed))
    _logger.info(
        "cluster %s %s: run #%d, threshold %.2f, %d faces -> %d groups (%d named by a human, %d minted unnamed)",
        model_id,
        model_version,
        run_id,
        pinned,
        faces,
        clusters,
        clusters - len(unnamed),
        len(unnamed),
    )
    _logger.info(
        "cluster %s %s: run #%d is %s",
        model_id,
        model_version,
        run_id,
        derived.standing(conn, run_id, model_id, pinned),
    )
    for (cluster_id,) in unnamed:
        person_id = naming.claim(conn, "person", "")
        conn.execute("INSERT INTO person(id, name, created_at) VALUES(?, NULL, ?)", (person_id, now))
        conn.execute("UPDATE derived_face_cluster SET person_id = ? WHERE id = ?", (person_id, cluster_id))
        for file_id, faces in conn.execute(
            "SELECT fi.file_id, count(*) FROM derived_face_membership m"
            " JOIN derived_face_instance fi ON fi.id = m.face_id"
            " WHERE m.cluster_id = ? GROUP BY fi.file_id",
            (cluster_id,),
        ).fetchall():
            derived.attribute(conn, file_id, person_id, run_id, model_id, model_version, face_count=faces)
    # A re-run dissolves groups, and a placeholder person whose group
    # dissolved is an address about nothing. Only the unnamed go: a name
    # is a human's word and keeps its entity regardless.
    conn.execute(
        "DELETE FROM entity WHERE kind = 'person' AND id IN ("
        " SELECT p.id FROM person p WHERE p.name IS NULL"
        " AND NOT EXISTS (SELECT 1 FROM person_assertion pa WHERE pa.person_id = p.id)"
        " AND NOT EXISTS (SELECT 1 FROM derived_face_cluster c WHERE c.person_id = p.id)"
        " AND NOT EXISTS (SELECT 1 FROM derived_file_person fp WHERE fp.person_id = p.id)"
        " AND NOT EXISTS (SELECT 1 FROM feedback fb WHERE fb.person_id = p.id))"
    )


_BACKENDS: dict = {}


def _face_item(conn, file_id: int, payload: dict, now: float) -> None:
    """One file through the backend the job's payload names. The job is
    the one place that may provision weights (docs/AI_MODELS.md)."""
    from vision import faces as faces_module

    from . import detect

    models_dir = payload["models_dir"]
    thumbs_dir = payload.get("thumbs_dir")
    key = (models_dir, payload.get("backend", "auto"), payload.get("providers", "auto"))
    told = report()
    backend = _BACKENDS.get(key)
    if backend is None:
        told.phase("loading-backend", face_backend=key[1], ort_providers=key[2])
        try:
            backend = faces_module.backend_for(models_dir, choice=key[1], providers=key[2], provision=True)
        except faces_module.BackendUnavailable as why:
            # Held, so every item of this job fails by the same name at
            # once instead of re-attempting a download per picture.
            _BACKENDS[key] = why
            _logger.exception("faces: no backend for face_backend=%s", key[1])
            raise
        _BACKENDS[key] = backend
        _logger.info(
            "faces: backend %s %s (face_backend=%s, ort_providers=%s, models_dir=%s)",
            backend.model_id,
            backend.model_version,
            key[1],
            key[2],
            models_dir,
        )
    if isinstance(backend, faces_module.BackendUnavailable):
        raise backend
    kind = conn.execute("SELECT kind FROM file WHERE id = ?", (file_id,)).fetchone()[0]
    path = detect.path_of(conn, file_id)
    told.phase("detecting", kind=kind, backend=str(getattr(backend, "model_id", "")))
    if kind == "video":
        detect.harvest_video(conn, backend, file_id, path, now, thumbs_dir=thumbs_dir)
    else:
        detect.harvest(conn, backend, file_id, path, now, thumbs_dir=thumbs_dir)
    faces = conn.execute("SELECT count(*) FROM derived_face_instance WHERE file_id = ?", (file_id,)).fetchone()[0]
    told.observe("faces-found", count=int(faces))


#: (models_dir, caption_model) -> the loaded captioner, or the LookupError
#: that refused it, held across items the way _BACKENDS holds face backends.
_CAPTIONERS: dict = {}


def submit_annotate(conn, now: float, *, models_dir: str, everything: bool = False) -> int | None:
    """A caption for every present picture and video that lacks one from
    the configured model for its CURRENT bytes, as one job -- or, with
    `everything`, for all of them again. None when nothing is left to
    caption. The `caption_model` setting is read HERE, once, into the
    payload: every item of one job runs the same model."""
    from . import settings as settings_module

    model = settings_module.value(conn, "caption_model")
    if not model or "/" not in model:
        # refused here, not queued: every item of a job under a name that
        # is no repository would fail by the same sentence
        raise ValueError(
            f"caption_model must be a Hub repository id like Salesforce/blip-image-captioning-base, not {model!r}"
        )
    items = caption_items(conn, model, everything=everything)
    if not items:
        return None
    payload = {"models_dir": models_dir, "model": model, "kind": "caption"}
    return jobs.submit(conn, "annotate", now, payload=payload, items=items)


def _caption_with_lookahead(conn, told, captioner, file_id: int, kind: str, frame) -> str:
    """This picture's caption, and as many of the job's next as fit.

    The same shape the embed job uses (`_Ahead`), for the same measured
    reason: one picture per `generate()` was 3.62 pictures/sec where a
    batch of sixteen is 21.28. The item asked about is captioned and
    returned exactly as before; the others are simply already done when
    their turn comes.

    A VIDEO leads no batch. Its poster is only half its work -- the
    sampled moments are captioned per clip afterwards -- so batching it
    with stills would mix two shapes of work for the smaller half.
    """
    from . import detect, oriented

    if kind == "video":
        told.phase("captioning", model=captioner.model_id)
        return captioner.describe(frame)

    together: list[tuple[int, object]] = []
    # The LEADER is charged first: it is in the batch and it decodes like
    # any other member, so leaving it out would make the stated bound a
    # bound on the followers.
    mine = conn.execute("SELECT width, height FROM file WHERE id = ?", (file_id,)).fetchone()
    budget = BATCH_MEGAPIXELS - _megapixels(mine)
    for item in [one for one in jobs.pending(conn, told.job_id) if one != file_id][: caption_batch() - 1]:
        row = conn.execute("SELECT kind, width, height FROM file WHERE id = ?", (item,)).fetchone()
        if row is None or row[0] == "video":
            continue
        # Bounded by PIXELS as well as by count, because cancellation is
        # checked BETWEEN items and a batch runs inside one: an unbounded
        # batch is an unbounded wait for somebody who asked it to stop.
        budget -= _megapixels(row[1:])
        if budget < 0:
            break
        together.append((item, detect.path_of(conn, item)))

    if not together:
        told.phase("captioning", model=captioner.model_id)
        return captioner.describe(frame)

    # Decoded HERE, on this thread, from paths and orientations read
    # here. Handing a connection to a worker is what once turned the
    # embed batch into a silent four-times-slower fallback.
    frames = [frame]
    kept: list[int] = []
    for item, path in together:
        try:
            held = oriented.for_model(conn, item, path)
        except (OSError, ValueError):
            # Its own turn will fail it, by its own name, with its own
            # error. Dropping it from the batch is not deciding anything
            # about it.
            continue
        frames.append(held)
        kept.append(item)

    # Reported ONCE, against the item that triggered it, and named so it
    # cannot be read as that item's own cost: `pictures` says how many it
    # covered. The others report no captioning phase at all, which is
    # true -- they performed none.
    told.phase("batch-captioning", model=captioner.model_id, pictures=len(frames))
    try:
        said = captioner.describe_many(frames)
    except (OSError, ValueError) as why:
        # Narrower than ITEM_FAILURES on purpose. What a batch may
        # legitimately fail with is what a bad PICTURE raises; catching
        # sqlite3.Error here once turned a threading defect into a silent
        # fallback that ran four times slower with nothing in the log.
        _logger.warning(
            "job #%d: a caption batch of %d failed (%s); captioning this item alone", told.job_id, len(frames), why
        )
        told.phase("captioning", model=captioner.model_id)
        return captioner.describe(frame)

    _said.keep(told.job_id, {item: text for item, text in zip(kept, said[1:], strict=True) if text})
    return said[0]


def _annotate_item(conn, file_id: int, payload: dict, now: float) -> None:
    """One file through the captioner the payload names; a video is
    captioned by its poster frame, the same frame search embeds. The
    job is the one place that may provision weights (docs/AI_MODELS.md)."""
    from vision import captions as captions_module
    from vision import decode

    from . import derived, detect, oriented, scan

    key = (payload["models_dir"], payload["model"])
    told = report()
    captioner = _CAPTIONERS.get(key)
    if captioner is None:
        told.phase("loading-captioner", model=key[1])
        try:
            captioner = captions_module.captioner_for(key[0], key[1], provision=True)
        except (LookupError, OSError, ValueError) as why:
            # Held, so every item of this job fails by the same name at
            # once instead of re-attempting a download per picture --
            # OSError is what from_pretrained raises for a repository that
            # is not there (transformers utils/hub.py), an item failure too.
            _CAPTIONERS[key] = why
            _logger.exception("annotate: no captioner for caption_model=%s", key[1])
            raise
        _CAPTIONERS[key] = captioner
        _logger.info("annotate: captioner %s %s (models_dir=%s)", captioner.model_id, captioner.model_version, key[0])
    if isinstance(captioner, Exception):
        raise captioner
    kind, sha = conn.execute("SELECT kind, content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()
    path = detect.path_of(conn, file_id)
    if sha is None:
        sha = scan.sha256_of(path)
        conn.execute("UPDATE file SET content_sha256 = ? WHERE id = ?", (sha, file_id))
    # Captioned already, with the batch some earlier item led.
    text = _said.take(told.job_id, file_id) or ""
    if not text:
        told.phase("decoding", kind=kind)
        frame = decode.poster(path) if kind == "video" else oriented.for_model(conn, file_id, path)
        if frame is None:
            raise ValueError(f"file {file_id} has no decodable frame to caption")
        text = _caption_with_lookahead(conn, told, captioner, file_id, kind, frame).strip()
    if not text:
        raise ValueError(f"{captioner.model_id} said nothing about file {file_id}")
    told.phase("recording")
    derived.annotate(
        conn, file_id, payload.get("kind", "caption"), text, captioner.model_id, captioner.model_version, sha, now
    )
    moments = 0
    if kind == "video":
        # a clip is also captioned at its sampled moments -- the same
        # persisted rows detection looks at (db/sample.py), so a caption
        # says which second it describes and a re-run finds its own work
        from . import sample

        sample.frames(conn, file_id, path)
        by_offset = {offset: sample_id for sample_id, offset, _ in sample.taken(conn, file_id)}
        told.phase("captioning-moments", model=captioner.model_id, moments=len(by_offset))
        for offset_ms, image in decode.frames_at(path, sorted(by_offset)):
            said = captioner.describe(image).strip()
            if not said:
                continue
            derived.annotate(
                conn,
                file_id,
                payload.get("kind", "caption"),
                said,
                captioner.model_id,
                captioner.model_version,
                sha,
                now,
                sample_id=by_offset[offset_ms],
            )
            moments += 1
    told.observe("caption", words=len(text.split()), moments=moments)


#: kind -> handler(conn, item_id, payload, now). The names are the schema's:
#: `job.kind` is CHECK-constrained (db/schema.sql:493-495) so a typo is an
#: IntegrityError at submit, never a job that queues and waits forever.
def submit_context(conn, now: float, *, everything: bool = False) -> int | None:
    """Interpret every present file whose interpretation is missing --
    never made, staled by a source change (db/context.py stale), or made
    under an older policy -- or, with `everything`, every present file
    again. None when nothing is left. One item per file, so
    cancellation, resume and failures land at file boundaries
    (db/context.py rebuild_one). The schema's job kind is 'context'."""
    from . import context as context_module

    sql = "SELECT f.id FROM file f WHERE f.missing_since IS NULL"
    args: tuple = ()
    if not everything:
        sql += " AND NOT EXISTS (SELECT 1 FROM derived_media_context c WHERE c.file_id = f.id AND c.policy_version = ?)"
        args = (context_module.POLICY_VERSION,)
    items = [row[0] for row in conn.execute(sql + " ORDER BY f.id", args)]
    if not items:
        return None
    return jobs.submit(conn, "context", now, items=items)


def _context_item(conn, file_id: int, payload: dict, now: float) -> None:
    from . import context as context_module

    report().phase("interpreting")
    context_module.rebuild_one(conn, file_id, now)


def submit_events(conn, now: float) -> int:
    """Re-propose every grouping hypothesis over the current contexts:
    one item per Grouper adapter, so a smarter grouper failing never
    costs the others their run. A separate job from 'context' on
    purpose -- per-file rebuild and global regroup have different
    invalidation and retry semantics, so they are two durable facts."""
    from . import events as events_module

    return jobs.submit(conn, "events", now, items=list(range(len(events_module.GROUPERS))))


def _events_item(conn, index: int, payload: dict, now: float) -> None:
    from . import events as events_module

    grouper = events_module.GROUPERS[index]
    report().phase("regrouping", grouper=grouper.name, version=grouper.version)
    events_module.regroup_one(conn, grouper, now)


def _story_plan_item(conn, item: int, payload: dict, now: float) -> None:
    """Durable planning: the engine the request named is loaded HERE,
    off the request thread (db/planning.py plan_item)."""
    from . import planning

    report().phase("planning", planner=payload.get("planner"), similarity=payload.get("similarity"))
    planning.plan_item(conn, item, payload, now)


def _embed_prompts_item(conn, prompt_id: int, payload: dict, now: float) -> None:
    """One prompt's vector under one text space (db/prompts.py)."""
    from . import prompts

    report().phase("embedding-prompt")
    prompts.embed_item(conn, prompt_id, payload, now)


#: What an item is called, for the ledger. One indexed lookup on a
#: connection that is already open, next to per-item work measured in
#: tens of milliseconds -- and it is the difference between a console
#: that says "item 41 started" a hundred thousand times and one that
#: says which picture it is on.
_ITEM_NAME = "SELECT name FROM file WHERE id = ?"


def _item_named(conn, kind: str, item: int) -> str | None:
    if kind not in jobs.FILE_ITEMS:
        return None
    row = conn.execute(_ITEM_NAME, (item,)).fetchone()
    return None if row is None else str(row[0])


def _walk_item(conn, index: int, payload: dict, now: float) -> None:
    """One root, walked -- the same `scan.scan` the request runs.

    The walk was the one expensive thing here that could not be QUEUED.
    `POST /roots/{id}/scan` does it inline and is its own worker, which
    is right for somebody who just pressed scan and is watching: the
    answer is the counts, and they arrive when the walk is done.

    Nothing unattended could ask for one. A nightly catch-up that cannot
    walk derives forever over a library it never notices growing, which
    is the most useless kind of scheduled job -- busy, and blind.

    Same function, same reconciliation, same offline veto. `RootOffline`
    is left to propagate: a root that cannot be read is a failed item,
    and the alternative -- treating an unplugged drive as an empty
    library -- is what `scan.scan` refuses at the top for the same reason.
    """
    from . import scan as scan_module

    told = report()
    root_id = int(payload["roots"][index])
    path = conn.execute("SELECT path FROM root WHERE id = ?", (root_id,)).fetchone()
    if path is None:
        raise ValueError(f"no root {root_id} to walk")
    told.phase("walking", root=root_id, path=path[0])
    seen = {"spoken": 0}

    def watch(folders: int, files: int, hashed: int) -> None:
        # Phase reports, not checkpoints: a checkpoint is the ITEM
        # boundary and this whole walk is one item. Throttled the way the
        # request path throttles, so a large root does not spend its time
        # writing about itself.
        if files - seen["spoken"] < scan_module.WALK_EVERY:
            return
        seen["spoken"] = files
        told.phase("walking", root=root_id, folders=folders, files=files, hashed=hashed)

    result = scan_module.scan(conn, root_id, path[0], now, watch)
    told.phase("walked", root=root_id, added=result.added, replaced=result.replaced, missing=result.missing)


def submit_walk(conn, now: float, *, roots: list[int] | None = None) -> int | None:
    """Walk every online root, as one job of one item each.

    `roots` names them explicitly; without it, every root that is online
    -- an offline one cannot be read, and `scan.scan` refuses to act on
    that reading rather than marking a whole library missing.

    None when there is nothing to walk, which is a library with no roots
    registered.
    """
    named = (
        roots
        if roots is not None
        else [one for (one,) in conn.execute("SELECT id FROM root WHERE online = 1 ORDER BY id")]
    )
    if not named:
        return None
    return jobs.submit(conn, "walk", now, payload={"roots": named}, items=list(range(len(named))))


HANDLERS = {
    "walk": _walk_item,
    "story_plan": _story_plan_item,
    "embed_prompts": _embed_prompts_item,
    "context": _context_item,
    "events": _events_item,
    "scan": _ingest_item,
    "hash": _hash_item,
    "embed": _embed_item,
    "detect_faces": _face_item,
    "cluster_faces": _cluster_item,
    "annotate": _annotate_item,
}


def run_next(
    conn,
    owner: str,
    now: float,
    *,
    handlers=None,
    kinds=None,
    gate=None,
    budget: int | None = None,
    clock=None,
    on_progress=None,
    on_event=None,
    should_stop=None,
) -> dict | None:
    """One worker turn: claim the next runnable job and work it.

    Returns None when nothing is runnable, otherwise a summary of what this
    turn did. `budget` bounds how many items the turn performs -- the job
    stays `running` under its lease and the next turn (this process or any
    other) continues from the items still pending, which is the resumption
    contract stated on db/jobs.py.

    `clock` supplies the time for heartbeats and settlement; without one,
    `now` stands for the whole turn. A long-lived worker MUST pass one: a
    heartbeat stamped with claim time extends the lease to claim + 60s
    forever, so any single job outliving the lease was reclaimed mid-run
    by the next worker while still being worked.

    `on_progress` hears every observable change -- claim, each finished
    item, the terminal state -- as `{job, kind, state, done, total}`. This
    is the delta feed the live channel publishes; the row stays the truth
    a subscriber renders from cold. Every state change COMMITS BEFORE it
    is spoken, terminal states included: a delta describing an uncommitted
    write invites the subscriber to read the row and find it behind what
    the wire just said -- caught live by a client whose snapshot read
    'running' after its socket said 'done'.

    `on_event` hears the ledger (db/ledger.py): every row this turn
    appends, spoken after the commit that made it durable and carrying
    its id -- and, between those, each handler report marked `pending`
    (see Report), which is presentation until the item settles.

    A handler that raises anything outside ITEM_FAILURES is a DEFECT, not
    a verdict about the item: its writes are rolled back, a
    `worker.turn_failed` event carrying the traceback is committed so the
    console can explain why the job sits under an expiring lease, and the
    exception propagates to take the turn down exactly as before.
    """
    from . import similarity as similarity_module

    handlers = HANDLERS if handlers is None else handlers
    tick = clock if clock is not None else (lambda: now)
    tell = on_progress if on_progress is not None else (lambda delta: None)
    tell_event = on_event if on_event is not None else (lambda event: None)
    try:
        claimed = jobs.claim(conn, owner, now, kinds=kinds, gate=gate)
    except sqlite3.OperationalError as busy:
        if getattr(busy, "sqlite_errorname", "") not in BUSY:
            raise
        # Another writer holds the lane. SQLite has ONE, and a long
        # write -- a scan of a new root walks and commits once -- holds
        # it well past `busy_timeout`. Nothing has gone wrong: there is
        # no turn to take right now, which is what None already means
        # here and what the worker already waits on.
        #
        # Raising instead produced a traceback every few seconds saying
        # "a worker turn died; the job's lease will be reclaimed" --
        # both halves false, because the claim is what failed, so no job
        # was claimed and no lease exists. A log that reports healthy
        # backpressure as a crash teaches people to ignore it.
        _logger.info("the database is busy; no turn this pass (%s)", busy)
        return None
    if claimed is None:
        return None
    job_id, fence = claimed

    # Anything a previous turn of this job computed ahead is dropped: the
    # items it covered may have been worked by another worker since, and
    # a vector held across a lease lapse is a guess about a file nobody
    # has looked at recently. Recomputing is the cheap half of this.
    #
    # Captions the same, and for the same reason: a sentence held across
    # a lease lapse is a claim about bytes that may have changed, and a
    # caption records the `source_sha256` it was made from.
    _ahead.forget(job_id)
    _said.forget(job_id)

    kind, raw, attempt, lease_until = conn.execute(
        "SELECT kind, payload, attempt, lease_until FROM job WHERE id = ?", (job_id,)
    ).fetchone()
    unspoken: list[dict] = []

    def note(type_: str, **kw) -> dict:
        """One ledger row in the open transaction; spoken after commit."""
        told = ledger.record(conn, job_id, type_, tick(), **kw)
        unspoken.append(told)
        return told

    def committed() -> None:
        conn.commit()
        for told in unspoken:
            tell_event(told)
        unspoken.clear()

    # A second attempt is a RECLAIM -- the last owner's lease lapsed --
    # unless the ledger says the last turn paused on purpose, which is a
    # resume under a new fence, not a recovery.
    last = ledger.latest_for_job(conn, job_id)
    resumed = attempt > 1 and last is not None and last["type"] == "job.paused"
    reclaimed = attempt > 1 and not resumed
    note(
        "job.reclaimed" if reclaimed else "job.claimed",
        message=(
            f"{owner} reclaimed the job (attempt {attempt}; the last lease lapsed)"
            if reclaimed
            else f"{owner} resumed the job (attempt {attempt})"
            if resumed
            else f"{owner} took the job"
        ),
        severity="warning" if reclaimed else "info",
        data={"owner": owner, "attempt": attempt, "fence": fence, "lease_until": lease_until, "resumed": resumed},
    )
    committed()

    def spoke(state: str, moved) -> None:
        # `cancel_requested` rides every delta so a subscriber that saw the
        # cancel asked for keeps seeing it asked for until the job settles;
        # the row is read, never remembered.
        tell(
            {
                "job": job_id,
                "kind": kind,
                "state": state,
                "done": moved.done,
                "total": moved.total,
                "cancel_requested": 1 if jobs.cancelled(conn, job_id) else 0,
                # the hash kind's mode, so the activity surface words it
                "derive": (json.loads(raw) if raw else {}).get("derive"),
            }
        )

    opened = jobs.progress(conn, job_id)
    started = tick()
    total, done = opened.total or 0, opened.done or 0
    _logger.info("job #%d %s: claimed, %d of %d items pending", job_id, kind, total - done, total)
    spoke("running", opened)
    handler = handlers.get(kind)
    if handler is None:
        why = f"no handler for kind {kind!r}"
        jobs.settle(conn, job_id, fence, "failed", tick(), error=why)
        note("job.failed", severity="error", message=why, data={"error": why})
        committed()
        _logger.error("job #%d %s: failed, no handler for that kind", job_id, kind)
        spoke("failed", jobs.progress(conn, job_id))
        return {"job": job_id, "state": "failed", "did": 0}
    payload = json.loads(raw) if raw else {}

    # A lazy step works out its units NOW, holding the lease. Submitted
    # with no items and no total, it could not know them: the step before
    # it had not run, so the files it will read did not exist yet.
    #
    # `total IS NULL` is the marker rather than an empty item list,
    # because "nothing to do" and "not decided yet" have to be different
    # rows -- a job that really has no work settles `done` over zero
    # items and must not be re-counted every time it is claimed.
    counter = COUNTERS.get(payload.get("count_when_claimed", ""))
    if counter is not None and jobs.not_yet_counted(conn, job_id):
        found = jobs.count_now(conn, job_id, fence, counter(conn, payload))
        committed()
        _logger.info("job #%d %s: %d to do, counted when claimed", job_id, kind, found)

    did = failed = 0
    for item in jobs.pending(conn, job_id):
        if (budget is not None and did >= budget) or (should_stop is not None and should_stop()):
            # A deliberate stop, not a death: the lease is expired on the
            # spot so the very next turn -- any process -- resumes the job
            # instead of waiting out a liveness timeout meant for crashes.
            # `should_stop` is how a shutting-down worker leaves a long
            # job at an item boundary instead of holding the exit hostage.
            jobs.pause(conn, job_id, fence, tick())
            note(
                "job.paused",
                message=f"paused after {did} items; the next turn resumes it",
                data={
                    "did": did,
                    "failed": failed,
                    "why": "budget" if budget is not None and did >= budget else "stop",
                },
            )
            committed()
            _logger.info(
                "job #%d %s: paused after %d items (%d failed); the next turn resumes it", job_id, kind, did, failed
            )
            return {"job": job_id, "state": "running", "did": did, "failed": failed}
        if jobs.cancelled(conn, job_id):
            jobs.settle(conn, job_id, fence, "cancelled", tick())
            note(
                "job.cancelled",
                severity="warning",
                message=f"stopped at the item boundary after {did} items this turn",
                data={"did": did, "failed": failed},
            )
            committed()
            _logger.info("job #%d %s: cancelled after %d items (%d failed)", job_id, kind, did, failed)
            spoke("cancelled", jobs.progress(conn, job_id))
            return {"job": job_id, "state": "cancelled", "did": did, "failed": failed}
        # The start is committed BEFORE the handler runs -- a 47-second
        # decode is then a console row saying so, not a frozen bar.
        named = _item_named(conn, kind, item)
        note(
            "item.started",
            item_id=item,
            message=f"item {item} started",
            data={"item_name": named} if named else None,
        )
        committed()
        told = Report(job_id, item, tick, tell_event)
        token = _REPORT.set(told)
        try:
            handler(conn, item, payload, now)
        except ITEM_FAILURES as why:
            # The dead handler's half-finished writes ride the same open
            # transaction as the failure record about to be committed --
            # dropped first, so the record carries the verdict and nothing
            # the handler did not live to finish. Its noted index
            # mutations die with it: rows that never became durable must
            # never reach a live index.
            conn.rollback()
            similarity_module.discard_pending(conn)
            told.close()
            _keep(conn, told, unspoken)
            moved = jobs.finish_item(conn, job_id, fence, item, error=str(why))
            failed += 1
            note(
                "item.failed",
                item_id=item,
                severity="warning",
                message=str(why),
                data={
                    "error": str(why),
                    "exception": type(why).__name__,
                    "job_continues": True,
                    # The one a person most wants named. "item 41 failed:
                    # cannot identify image file" is a defect report with
                    # the subject removed.
                    **({"item_name": named} if named else {}),
                },
            )
            # The name only where there IS one: "item 2 (?) failed" is
            # a question mark standing in for a fact that does not exist
            # for this kind of item, which reads as a lookup that broke.
            _logger.warning("job #%d %s: item %r%s failed: %s", job_id, kind, item, f" ({named})" if named else "", why)
        except Exception as defect:
            conn.rollback()
            similarity_module.discard_pending(conn)
            told.close()
            _keep(conn, told, unspoken)
            lease = conn.execute("SELECT lease_until FROM job WHERE id = ?", (job_id,)).fetchone()
            note(
                "worker.turn_failed",
                item_id=item,
                severity="error",
                message=f"{type(defect).__name__}: {defect}",
                data={
                    "exception": type(defect).__name__,
                    "error": str(defect),
                    "traceback": traceback.format_exc(),
                    "attempt": attempt,
                    "fence": fence,
                    "owner": owner,
                    "lease_until": lease[0] if lease else None,
                    "job_continues": False,
                    "reclaimable": True,
                },
            )
            committed()
            raise
        else:
            told.close()
            _keep(conn, told, unspoken)
            moved = jobs.finish_item(conn, job_id, fence, item)
            # `named` is the one read at the top of this item, reused:
            # the row cannot have changed under a transaction that has
            # been open the whole time.
            note(
                "item.done",
                item_id=item,
                message=f"item {item} done",
                data={"item_name": named} if named else None,
            )
        finally:
            _REPORT.reset(token)
        did += 1
        jobs.heartbeat(conn, job_id, fence, tick())
        committed()
        # The commit succeeded, so the item's representation writes are
        # durable -- NOW they may reach the resident indexes. A crash in
        # the gap is safe: the index lags committed truth until the next
        # align repairs it, which is the one direction the invariant
        # permits (it may lag SQLite, never lead it).
        similarity_module.apply_pending(conn)
        spoke("running", moved)

    jobs.settle(conn, job_id, fence, "done", tick())
    note(
        "job.done",
        message=f"done: {did} items this turn, {failed} failed, {tick() - started:.1f}s",
        data={"did": did, "failed": failed, "seconds": tick() - started},
    )
    committed()
    _logger.info("job #%d %s: done, %d items, %d failed, %.1fs", job_id, kind, did, failed, tick() - started)
    spoke("done", jobs.progress(conn, job_id))
    return {"job": job_id, "state": "done", "did": did, "failed": failed}


def _keep(conn, told: Report, unspoken: list[dict]) -> None:
    """The item's kept reports become ledger rows, in the order they were
    said, inside the transaction that settles the item."""
    unspoken.extend(
        ledger.record(
            conn,
            said["job_id"],
            said["type"],
            said["at"],
            item_id=said["item_id"],
            phase=said["phase"],
            severity=said["severity"],
            message=said["message"],
            data=said["data"],
        )
        for said in told.kept
    )


#: How a lazy step works out its units when it is claimed.
#:
#: Keyed by a name in the payload rather than by kind, because a kind
#: does not decide this: `scan` submitted on its own knows its files at
#: submit and should, so a person pressing the button is told "nothing
#: to do" then and not four jobs later.
COUNTERS: dict[str, Callable[..., list[int]]] = {
    "ingest": lambda conn, payload: ingest_items(conn, everything=bool(payload.get("everything"))),
    "faces": lambda conn, payload: face_items(conn, everything=bool(payload.get("everything"))),
    "captions": lambda conn, payload: caption_items(
        conn, str(payload.get("model", "")), everything=bool(payload.get("everything"))
    ),
}


#: The name the catch-up's steps share. A schedule points at this; the
#: console groups on it.
CATCH_UP = "catch up"


def catch_up(conn, now: float, *, models_dir: str, thumbs_dir: str | None = None) -> list[int]:
    """Bring the library up to date, in the order the work actually has.

    Eight buttons in a sequence only this application knew. The order is
    real and the failure it causes is quiet: `cluster_faces` over an
    unembedded library settles `done` having clustered nothing, so
    pressing them out of order does not look like a mistake -- it looks
    like a library with no people in it.

    Every step is gated on the one before with `after_id`, so this queues
    all of them at once and the runner takes them in order. A step that
    has nothing to do returns no job and is simply absent: the next step
    is then gated on the last one that DID queue, never on a hole.

    Returns the job ids in order. Empty means the library was already up
    to date, which is an answer and not a failure.

    The walk leads, and every step that reads files is LAZY -- submitted
    with no items, counting its units when a worker claims it
    (`COUNTERS`). That pairing is the whole correctness of this: a chain
    is only as ordered as its least lazy step, and an eager step decides
    what to do before the step in front of it has run. Led by a walk,
    eager steps would derive nothing over everything the walk found and
    look like they had worked.

    They keep their per-file items, which is what gives a sweep of
    eighty thousand files its progress and its per-file failure
    isolation. What moved is WHEN the list is made, not what is in it.

    Pressing one of these sweeps on its own still counts at submit, and
    should: somebody who presses "read the metadata of every file not
    yet read" is owed "nothing to do" now, not four jobs later.
    """
    from . import jobs

    queued: list[int] = []
    after: int | None = None

    def step(made) -> None:
        """One link. `made` is whatever a submitter returned: an id, None
        for nothing to do, or a list (embed queues one job per space)."""
        nonlocal after
        for one in made if isinstance(made, list) else [made]:
            if one is None:
                continue
            jobs.enlist(conn, one, CATCH_UP, after)
            queued.append(one)
            after = one

    # What is on disk, before anything that reads what is on disk.
    step(submit_walk(conn, now))
    # Metadata before anything derived from it: an embedding of a file
    # ingest has not read is an embedding of bytes nothing has described.
    step(jobs.submit(conn, "scan", now, payload={"count_when_claimed": "ingest"}))
    # Interpretation, then the sessions built out of it.
    step(submit_context(conn, now))
    step(submit_events(conn, now))
    # Semantic vectors, then faces, then the grouping OF those faces --
    # which is the pair the order exists for.
    step(submit_embed(conn, now, models_dir=models_dir))
    step(_lazy_faces(conn, now, models_dir=models_dir, thumbs_dir=thumbs_dir))
    step(submit_cluster(conn, now))
    # Captions last: the most expensive per file, and nothing waits on it.
    step(_lazy_captions(conn, now, models_dir=models_dir))
    return queued


def run_schedules(conn, now: float, *, models_dir: str, thumbs_dir: str | None = None) -> list[str]:
    """Start whatever is due, and say what was started.

    Called on the worker's own turn rather than by a timer of its own: a
    second scheduler is a second thing that can be running when nobody
    thinks anything is, and the runner is already the only thing that
    runs jobs. A worker that is off starts nothing, which is what "off"
    should mean.

    The guard against starting a collection twice lives in
    `scheduling.due` and is the load-bearing part -- a nightly catch-up
    over a library that takes thirty hours would otherwise be seven
    overlapping ones by Sunday.
    """
    from . import scheduling

    started = []
    for row in scheduling.due(conn, now):
        if row["collection"] != CATCH_UP:
            # `scheduling.RUNNABLE` refuses the others on write; this is
            # the same refusal at the other end, because a row can also
            # arrive from a restored backup written by a later build.
            _logger.warning("schedule names %r, which this build cannot run", row["collection"])
            continue
        queued = catch_up(conn, now, models_dir=models_dir, thumbs_dir=thumbs_dir)
        scheduling.started(conn, row["collection"], now)
        started.append(row["collection"])
        _logger.info("schedule: started %s (%d steps)", row["collection"], len(queued))
    return started


def _lazy_faces(conn, now: float, *, models_dir: str, thumbs_dir: str | None) -> int:
    """Face detection as a STEP: same payload, units counted on claim.

    The settings still ride in the payload, read once here, because every
    item of one job must run the same pipeline on the same device
    whatever the settings say by the time it drains. That is a fact about
    the job; the file list is a fact about the library, and only the
    second one has to wait.
    """
    from . import settings as settings_module

    payload: dict = {
        "models_dir": models_dir,
        "backend": settings_module.value(conn, "face_backend"),
        "providers": settings_module.value(conn, "ort_providers"),
        "count_when_claimed": "faces",
    }
    if thumbs_dir is not None:
        payload["thumbs_dir"] = thumbs_dir
    return jobs.submit(conn, "detect_faces", now, payload=payload)


def _lazy_captions(conn, now: float, *, models_dir: str) -> int:
    """Captioning as a STEP. The model is refused here if it is not a
    repository id: every item of a job under a bad name fails by the same
    sentence, and a chain should not queue a step that cannot work."""
    from . import settings as settings_module

    model = settings_module.value(conn, "caption_model").strip()
    if not model or "/" not in model:
        raise ValueError(
            f"caption_model must be a Hub repository id like Salesforce/blip-image-captioning-base, not {model!r}"
        )
    payload = {"models_dir": models_dir, "model": model, "kind": "caption", "count_when_claimed": "captions"}
    return jobs.submit(conn, "annotate", now, payload=payload)
