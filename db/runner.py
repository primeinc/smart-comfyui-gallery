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
import sqlite3
import traceback

from . import jobs, ledger

_logger = logging.getLogger(__name__)

#: What a handler is allowed to fail with, per item. Everything else is a
#: defect in the handler, not a fact about the item.
ITEM_FAILURES = (OSError, ValueError, RuntimeError, LookupError, sqlite3.Error)


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
        if self.phase_now is not None:
            self._note("phase.finished", phase=self.phase_now, message=f"{self.phase_now} finished", data=None)
        self.phase_now = name
        self._note("phase.started", phase=name, message=f"{name} started", data=data or None)

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
        if self.phase_now is not None:
            self._note("phase.finished", phase=self.phase_now, message=f"{self.phase_now} finished", data=None)
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

    sql = "SELECT f.id FROM file f WHERE f.missing_since IS NULL AND f.kind IN ('image', 'animated_image', 'video')"
    if not everything:
        sql += (
            " AND NOT EXISTS (SELECT 1 FROM derived_face_scan s WHERE s.file_id = f.id"
            "   AND s.source_sha256 = f.content_sha256)"
        )
    items = [row[0] for row in conn.execute(sql + " ORDER BY f.id")]
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


def _thumbs_item(conn, file_id: int, payload: dict, now: float) -> None:
    import pathlib

    from vision import decode, thumbs

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
    told.phase("decoding", kind=kind)
    frame = decode.poster(path) if kind == "video" else oriented.for_model(conn, file_id, path)
    if frame is None:
        raise ValueError(f"file {file_id} has no decodable frame to thumbnail")
    told.phase("rendering-thumbnails", variants=len(thumbs.EDGES))
    thumbs.put_all(cache, sha, frame)


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
        frame = decode.poster(path) if kind == "video" else oriented.for_model(conn, file_id, path)
        if frame is None:
            raise ValueError(f"file {file_id} has no decodable frame to embed")
        return frame

    media = semantic.MediaRef(path=str(path), kind=kind, frame=representative_frame)
    provider, model, checkpoint = payload["choice"]
    told = report()
    told.phase("loading-encoder", provider=provider, model=model)
    encoder = semantic.encoder(provider, payload["models_dir"], model, checkpoint)
    told.phase("encoding", kind=kind)
    vector = encoder.encode_media(media)
    told.phase("recording", space=str(encoder.space()))
    derived.record_embedding(conn, file_id, encoder.space(), vector, sha, now)


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
    for start in range(0, len(batch), 500):
        piece = batch[start : start + 500]
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
        row knows it once ingest has run; until then the decoder door
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
            if member == best or (dupes.hamming(by_id[member][1], by_id[best][1]) <= threshold and agreed(member, best))
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


def submit_ingest(conn, now: float) -> int:
    """Read every present file's own story, as one job.

    The walk (POST /roots/{id}/scan) finds files cheaply; this is the
    expensive half of scanning that turns each file's metadata into
    entities -- models, LoRAs, prompts, generation settings, capture
    facts, learned param keys. The schema's job kind for it is 'scan'.
    """
    items = [row[0] for row in conn.execute("SELECT id FROM file WHERE missing_since IS NULL ORDER BY id")]
    return jobs.submit(conn, "scan", now, items=items)


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


def submit_cluster(conn, now: float) -> int:
    """Group every embedding space's faces into people, as one job.

    One item per (model_id, model_version) that holds embedded faces. The
    spaces ride in the payload -- `job_item` holds integers, so each item
    is an index into that list, fixed at submit time. No spaces means a
    job with nothing to do, which settles `done` honestly rather than
    being refused: "cluster an unindexed library" is an answer, not an
    error.
    """
    spaces = conn.execute(
        "SELECT DISTINCT model_id, model_version FROM derived_face_instance"
        " WHERE embedding IS NOT NULL ORDER BY model_id, model_version"
    ).fetchall()
    return jobs.submit(
        conn,
        "cluster_faces",
        now,
        payload={"spaces": [list(space) for space in spaces]},
        items=list(range(len(spaces))),
    )


def _cluster_item(conn, index: int, payload: dict, now: float) -> None:
    """Cluster one embedding space and give every group a person.

    The run's whole answer is replaced -- clusters, inferred appearances,
    and the placeholder people minted for groups nobody has named. Names
    are never carried across by similarity: `seed_clusters_from_assertions`
    re-applies them from what a human wrote down, and only the groups
    still unnamed after that get a fresh unnamed person, addressable at
    `/p/person-<short-id>` until somebody names them.
    """
    from . import derived, naming

    told = report()
    model_id, model_version = payload["spaces"][index]
    # Method and threshold pinned once and passed to BOTH calls: recomputing
    # the run identity from separately-spelled defaults is how a drift makes
    # the DELETE below clear a different run's attributions.
    pinned = derived.threshold_for(model_id)
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
    sql = "SELECT f.id FROM file f WHERE f.missing_since IS NULL AND f.kind IN ('image', 'animated_image', 'video')"
    args: tuple = ()
    if not everything:
        sql += (
            " AND NOT EXISTS (SELECT 1 FROM derived_annotation a WHERE a.file_id = f.id AND a.kind = 'caption'"
            "   AND a.model_id = ? AND a.source_sha256 = f.content_sha256)"
        )
        args = (model,)
    items = [row[0] for row in conn.execute(sql + " ORDER BY f.id", args)]
    if not items:
        return None
    payload = {"models_dir": models_dir, "model": model, "kind": "caption"}
    return jobs.submit(conn, "annotate", now, payload=payload, items=items)


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
        except LookupError as why:
            # Held, so every item of this job fails by the same name at
            # once instead of re-attempting a download per picture.
            _CAPTIONERS[key] = why
            _logger.exception("annotate: no captioner for caption_model=%s", key[1])
            raise
        _CAPTIONERS[key] = captioner
        _logger.info("annotate: captioner %s %s (models_dir=%s)", captioner.model_id, captioner.model_version, key[0])
    if isinstance(captioner, LookupError):
        raise captioner
    kind, sha = conn.execute("SELECT kind, content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()
    path = detect.path_of(conn, file_id)
    if sha is None:
        sha = scan.sha256_of(path)
        conn.execute("UPDATE file SET content_sha256 = ? WHERE id = ?", (sha, file_id))
    told.phase("decoding", kind=kind)
    frame = decode.poster(path) if kind == "video" else oriented.for_model(conn, file_id, path)
    if frame is None:
        raise ValueError(f"file {file_id} has no decodable frame to caption")
    told.phase("captioning", model=captioner.model_id)
    text = captioner.describe(frame).strip()
    if not text:
        raise ValueError(f"{captioner.model_id} said nothing about file {file_id}")
    told.phase("recording")
    derived.annotate(
        conn, file_id, payload.get("kind", "caption"), text, captioner.model_id, captioner.model_version, sha, now
    )
    told.observe("caption", words=len(text.split()))


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


HANDLERS = {
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
    claimed = jobs.claim(conn, owner, now, kinds=kinds, gate=gate)
    if claimed is None:
        return None
    job_id, fence = claimed

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
        note("item.started", item_id=item, message=f"item {item} started")
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
                data={"error": str(why), "exception": type(why).__name__, "job_continues": True},
            )
            _logger.warning("job #%d %s: item %r failed: %s", job_id, kind, item, why)
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
            note("item.done", item_id=item, message=f"item {item} done")
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
