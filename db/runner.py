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

import json
import sqlite3

from . import jobs

#: What a handler is allowed to fail with, per item. Everything else is a
#: defect in the handler, not a fact about the item.
ITEM_FAILURES = (OSError, ValueError, RuntimeError, LookupError, sqlite3.Error)


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

    stored = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()
    if stored is None:
        raise LookupError(f"file {file_id} left the library mid-job")
    if stored[0] is None:
        raise LookupError(f"file {file_id} has no recorded hash to verify against")
    actual = scan.sha256_of(detect.path_of(conn, file_id))
    if actual != stored[0]:
        raise ValueError(f"bytes changed behind the library's back: recorded {stored[0][:12]}, found {actual[:12]}")


def submit_faces(conn, now: float, *, models_dir: str, thumbs_dir: str | None = None) -> int:
    """Face detection over every present picture and video, as one job.

    A face on a video is a face on a sampled frame; the handler routes by
    the file kind, and the schema already keys every face to the moment it
    was seen (`derived_face_instance.sample_id`).

    `thumbs_dir` rides in the payload when thumbnail precaching is on:
    detection decodes every frame anyway, and the cache takes the decoded
    pixels on the way past instead of re-decoding later on first view.
    """
    items = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM file WHERE missing_since IS NULL"
            " AND kind IN ('image', 'animated_image', 'video') ORDER BY id"
        )
    ]
    payload: dict = {"models_dir": models_dir}
    if thumbs_dir is not None:
        payload["thumbs_dir"] = thumbs_dir
    return jobs.submit(conn, "detect_faces", now, payload=payload, items=items)


def submit_phash(conn, now: float) -> int:
    """Perceptual hashes for every present picture, as one job.

    The backfill for a library that never ran detection -- and for every
    video even when it did: detection records hashes as a byproduct only
    for whole still frames (a video's frames are samples, and a sample
    hash is not a file hash), so videos are fingerprinted here or not at
    all. Rides the schema's 'hash' kind with a payload the handler
    dispatches on: both jobs are about what a file's content IS, one
    verifying bytes, one fingerprinting pixels.
    """
    items = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM file WHERE missing_since IS NULL"
            " AND kind IN ('image', 'animated_image', 'video') ORDER BY id"
        )
    ]
    return jobs.submit(conn, "hash", now, payload={"derive": "perceptual"}, items=items)


def _perceptual_item(conn, file_id: int, payload: dict, now: float) -> None:
    from vision import decode, dupes

    from . import derived, detect, oriented, scan

    kind, sha = conn.execute("SELECT kind, content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()
    path = detect.path_of(conn, file_id)
    frame = decode.poster(path) if kind == "video" else oriented.for_model(conn, file_id, path)
    if frame is None:
        raise ValueError(f"file {file_id} has no decodable frame to fingerprint")
    if sha is None:
        sha = scan.sha256_of(path)
    phash64, dhash64 = dupes.perceptual(frame)
    derived.record_hash(conn, file_id, sha, now, phash64=phash64, dhash64=dhash64)


def submit_embed(conn, now: float, *, models_dir: str) -> list[int]:
    """The joint image/text embedding for every present picture -- the
    representations `/search` answers from. ONE JOB PER participating
    space: each provider's items commit in their own transactions, so a
    model that fails to load or encode costs its own space's progress
    and nobody else's. A bad `semantic_model` setting is refused here,
    not queued.
    """
    from . import retrieval

    told = retrieval.choices(conn)
    items = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM file WHERE missing_since IS NULL"
            " AND kind IN ('image', 'animated_image', 'video') ORDER BY id"
        )
    ]
    return [
        jobs.submit(conn, "embed", now, payload={"models_dir": models_dir, "choice": list(choice)}, items=list(items))
        for choice in told
    ]


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
    encoder = semantic.encoder(provider, payload["models_dir"], model, checkpoint)
    derived.record_embedding(conn, file_id, encoder.space(), encoder.encode_media(media), sha, now)


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

    threshold = int(payload["threshold"])
    verify = payload.get("dhash_verify")
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
    manager = similarity.manager_for(conn)
    key = similarity.align(
        conn, manager, similarity.PHASH, sorted(by_id), lambda wanted: [by_id[v][1] for v in wanted], now
    )
    twins_a, twins_b, _distances = similarity.pair_graph(manager, key, threshold)

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
    """The 'hash' kind's two modes, told apart by payload: a bare job is
    the integrity sweep it always was, so jobs queued before the payload
    existed keep meaning what they meant."""
    derive = payload.get("derive")
    if derive == "perceptual":
        _perceptual_item(conn, file_id, payload, now)
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

    model_id, model_version = payload["spaces"][index]
    # Method and threshold pinned once and passed to BOTH calls: recomputing
    # the run identity from separately-spelled defaults is how a drift makes
    # the DELETE below clear a different run's attributions.
    pinned = derived.threshold_for(model_id)
    derived.cluster(conn, model_id, model_version, now, method=derived.DEFAULT_METHOD, threshold=pinned)
    run_id = derived.run_for(conn, model_id, model_version, derived.DEFAULT_METHOD, pinned, now)
    conn.execute("DELETE FROM derived_file_person WHERE run_id = ?", (run_id,))
    derived.seed_clusters_from_assertions(conn, run_id)
    for (cluster_id,) in conn.execute(
        "SELECT id FROM derived_face_cluster WHERE run_id = ? AND person_id IS NULL ORDER BY id",
        (run_id,),
    ).fetchall():
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
    from vision.faces import OpenCVFaceBackend

    from . import detect

    models_dir = payload["models_dir"]
    thumbs_dir = payload.get("thumbs_dir")
    backend = _BACKENDS.get(models_dir)
    if backend is None:
        backend = _BACKENDS[models_dir] = OpenCVFaceBackend(models_dir)
    kind = conn.execute("SELECT kind FROM file WHERE id = ?", (file_id,)).fetchone()[0]
    path = detect.path_of(conn, file_id)
    if kind == "video":
        detect.harvest_video(conn, backend, file_id, path, now, thumbs_dir=thumbs_dir)
    else:
        detect.harvest(conn, backend, file_id, path, now, thumbs_dir=thumbs_dir)


#: kind -> handler(conn, item_id, payload, now). The names are the schema's:
#: `job.kind` is CHECK-constrained (db/schema.sql:493-495) so a typo is an
#: IntegrityError at submit, never a job that queues and waits forever.
def submit_context(conn, now: float) -> int:
    """Reinterpret every present file: one item per file, so
    cancellation, resume and failures land at file boundaries
    (db/context.py rebuild_one). The schema's job kind is 'context'."""
    items = [row[0] for row in conn.execute("SELECT id FROM file WHERE missing_since IS NULL ORDER BY id")]
    return jobs.submit(conn, "context", now, items=items)


def _context_item(conn, file_id: int, payload: dict, now: float) -> None:
    from . import context as context_module

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

    events_module.regroup_one(conn, events_module.GROUPERS[index], now)


HANDLERS = {
    "context": _context_item,
    "events": _events_item,
    "scan": _ingest_item,
    "hash": _hash_item,
    "embed": _embed_item,
    "detect_faces": _face_item,
    "cluster_faces": _cluster_item,
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
    """
    from . import similarity as similarity_module

    handlers = HANDLERS if handlers is None else handlers
    tick = clock if clock is not None else (lambda: now)
    tell = on_progress if on_progress is not None else (lambda delta: None)
    claimed = jobs.claim(conn, owner, now, kinds=kinds, gate=gate)
    if claimed is None:
        return None
    job_id, fence = claimed
    conn.commit()

    kind, raw = conn.execute("SELECT kind, payload FROM job WHERE id = ?", (job_id,)).fetchone()

    def spoke(state: str, moved) -> None:
        tell({"job": job_id, "kind": kind, "state": state, "done": moved.done, "total": moved.total})

    spoke("running", jobs.progress(conn, job_id))
    handler = handlers.get(kind)
    if handler is None:
        jobs.settle(conn, job_id, fence, "failed", tick(), error=f"no handler for kind {kind!r}")
        conn.commit()
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
            conn.commit()
            return {"job": job_id, "state": "running", "did": did, "failed": failed}
        if jobs.cancelled(conn, job_id):
            jobs.settle(conn, job_id, fence, "cancelled", tick())
            conn.commit()
            spoke("cancelled", jobs.progress(conn, job_id))
            return {"job": job_id, "state": "cancelled", "did": did, "failed": failed}
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
            moved = jobs.finish_item(conn, job_id, fence, item, error=str(why))
            failed += 1
        else:
            moved = jobs.finish_item(conn, job_id, fence, item)
        did += 1
        jobs.heartbeat(conn, job_id, fence, tick())
        conn.commit()
        # The commit succeeded, so the item's representation writes are
        # durable -- NOW they may reach the resident indexes. A crash in
        # the gap is safe: the index lags committed truth until the next
        # align repairs it, which is the one direction the invariant
        # permits (it may lag SQLite, never lead it).
        similarity_module.apply_pending(conn)
        spoke("running", moved)

    jobs.settle(conn, job_id, fence, "done", tick())
    conn.commit()
    spoke("done", jobs.progress(conn, job_id))
    return {"job": job_id, "state": "done", "did": did, "failed": failed}
