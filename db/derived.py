"""Model output: what producers computed over the library.

Every row here can be recomputed from the source files, so no row here is
the only copy of anything a person wrote. Authored rows live in
`authored.py` and outlive any recomputation.

Recomputing is recovery or schema migration, not routine. A face pass reads
every file and runs a detector and an embedder on every face; the cost is
GPU time proportional to the library. Callers read stored values instead of
recomputing them.

Staleness is keyed on `source_sha256`, not on a timestamp. A backup restore,
a sync client or a copy rewrites mtime without changing a byte. Equal bytes
mean the derivation still holds.

`model_id` and `model_version` are on every row so upgrading one producer
invalidates that producer's output and nothing else. An upgrade is a
targeted re-run.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import SupportsFloat


def plain(value):
    """A model's number as one SQLite can store.

    Detectors return numpy, and sqlite3 binds an object it does not recognise
    through the buffer protocol -- which a numpy scalar supports -- so
    `np.float32(0.98)` arrives as a BLOB. Against a STRICT table that is an
    IntegrityError on the very first face; against a lax one it would be
    stored as raw bytes and read back as garbage.

    The trap is that it works until it doesn't: `np.float64` subclasses
    Python's float and `np.int64` does not, so a detector reporting doubles
    stores fine and the same code reporting float32 -- which is what every
    ONNX face model returns -- fails. Nothing caught it because every face in
    every test was placed here by hand, as Python literals.

    Duck-typed rather than importing numpy: torch scalars answer `.item()`
    too, and the schema layer should not take a dependency on either.
    """
    if value is None or isinstance(value, (int, float, str, bytes)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):
            return value
    return value


def drop_all(conn) -> list[str]:
    """Delete the derived tables, keeping the rows human assertions point at.

    For recovery and schema migration. Every deleted row must be recomputed
    to replace it. To invalidate one producer, delete by `model_id` and
    `model_version` instead of calling this.

    The `derived_` prefix is the table list, so no separate list is kept in
    step. Two exceptions are retained: a `person_assertion` locates its
    claim by a region and, on video, by a sampled moment. Deleting either
    leaves the assertion without its discriminant and the FK nulls the
    pointer.
    """
    names = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'derived\\_%' ESCAPE '\\' ORDER BY name"
        )
    ]
    for name in reversed(names):
        if name == "derived_media_sample":
            # A sample an assertion points at is the MOMENT the human's claim
            # is about; deleting it nulls the assertion's sample_id and the
            # seeder's cross-moment guard goes blind. Same rule as regions below.
            conn.execute(
                "DELETE FROM derived_media_sample WHERE id NOT IN"
                " (SELECT sample_id FROM person_assertion WHERE sample_id IS NOT NULL)"
            )
            continue
        conn.execute(f"DELETE FROM {name}")
    # Regions exist only to locate derived findings and human assertions. The
    # ones an assertion still points at are kept; the rest went with the rows
    # that cited them.
    conn.execute(
        "DELETE FROM region WHERE id NOT IN (SELECT region_id FROM person_assertion WHERE region_id IS NOT NULL)"
    )
    return names


# --- where in the picture --------------------------------------------------


def region(
    conn,
    x: SupportsFloat,
    y: SupportsFloat,
    w: SupportsFloat,
    h: SupportsFloat,
    *,
    mask: bytes | None = None,
) -> int:
    """A rectangle, in fractions of the frame.

    Normalized because a box in pixels is a box against one particular
    rendering: the same coordinates on a thumbnail or a re-encoded proxy
    point somewhere else. A mask goes to the blob store rather than to a
    path, so moving a cache directory cannot void it.

    `SupportsFloat`, not `float`, because that is what the callers hand
    over: `region_from_pixels` divides a detector's box by a frame size,
    and a detector reports numpy. `np.float64` subclasses Python's float
    and `np.float32` does not -- the same asymmetry `plain` above was
    written for -- so a `float` here would be a claim the caller cannot
    keep. The conversion below is the one place it is made.
    """
    x, y, w, h = float(x), float(y), float(w), float(h)
    # A detector's box can run off the edge, so an overhang is trimmed to the
    # frame; a box more than half outside is pixel coordinates passed as
    # fractions, not a face at the edge, and is refused rather than trimmed.
    if x < 0 or y < 0 or x + w > 1 or y + h > 1:
        # Guarded, not unconditional: clamping every box puts floating-point
        # error into ones already inside, where 0.6 + 0.3 - 0.6 is
        # 0.29999999999999993.
        left, top = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        right = min(max(x + w, 0.0), 1.0)
        bottom = min(max(y + h, 0.0), 1.0)
        kept = (right - left) * (bottom - top)
        asked = w * h
        if asked > 0 and kept < asked / 2:
            raise ValueError(
                f"the box ({x}, {y}, {w}, {h}) is mostly outside the frame. "
                f"Regions are fractions of the frame, 0..1 -- pixels go through "
                f"region_from_pixels."
            )
        x, y, w, h = left, top, right - left, bottom - top

    mask_hash = None
    if mask:
        mask_hash = hashlib.sha256(mask).hexdigest()
        conn.execute(
            "INSERT OR IGNORE INTO blob(hash, payload_bin, byte_len) VALUES(?, ?, ?)",
            (mask_hash, mask, len(mask)),
        )
    cursor = conn.execute(
        "INSERT INTO region(x, y, w, h, mask_hash) VALUES(?, ?, ?, ?, ?)",
        tuple(plain(value) for value in (x, y, w, h, mask_hash)),
    )
    return int(cursor.lastrowid or 0)


def region_from_pixels(
    conn, box: Iterable[SupportsFloat], width: SupportsFloat, height: SupportsFloat, **kwargs
) -> int:
    """The same, given pixels and the size they were measured against.

    Offered so a caller with pixel coordinates converts once, here, rather
    than each detector inventing its own convention.

    This IS the foreign boundary: a detector hands over a numpy box and a
    numpy frame size, which is what
    `test_a_detectors_own_numbers_can_be_stored` passes. The parameters
    say so -- `SupportsFloat` is the `__float__` protocol, which numpy
    and torch scalars answer and `float` does not describe -- and the
    division below is on real floats.

    A zero dimension is what a truncated decode reports, and dividing by it
    raised ZeroDivisionError out of whatever job was running. It is refused
    by name instead: there is no rectangle inside a frame with no area.
    """
    across, down = float(width), float(height)
    if across <= 0 or down <= 0:
        raise ValueError(f"a {width}x{height} frame has nowhere to put a box")
    x, y, w, h = (float(one) for one in box)
    return region(conn, x / across, y / down, w / across, h / down, **kwargs)


# --- content hashes --------------------------------------------------------


def record_hash(conn, file_id: int, sha: str, now: float, *, phash64=None, dhash64=None) -> None:
    """Perceptual hashes, keyed on the content hash they were taken from
    and on the immutable space that computed them.

    The space id makes provenance belong to the ROW: after an ImageHash
    or frame-policy upgrade this writes a new row under the new space,
    and the old row keeps naming the implementation that actually
    produced it -- without the column, an upgrade relabeled every old
    hash as new by doing nothing.

    This writes the row and NOTES the index mutation; it never touches
    the live index itself. The runner rolls a failed item's writes back
    (db/runner.py, `conn.rollback()` on ITEM_FAILURES), and a FAISS
    index cannot ride that rollback -- so the note is applied by the
    runner strictly after the commit that made this row durable, and
    discarded on rollback (db/similarity.py apply_pending/discard_pending).
    """
    from . import similarity

    given = ((similarity.PHASH, phash64), (similarity.DHASH, dhash64))
    told = [(spec, value) for spec, value in given if value is not None]
    # A call with no fingerprint still records the source sha it was taken
    # from -- the staleness contract rides the row, and "hashing happened
    # against bytes X" is a fact even when no value came of it.
    for spec, value in told or [(similarity.PHASH, None)]:
        sid = similarity.space_id(conn, spec, now)
        conn.execute(
            "INSERT INTO derived_file_hash(file_id, space_id, value, source_sha256, computed_at)"
            " VALUES(?, ?, ?, ?, ?)"
            " ON CONFLICT(file_id, space_id) DO UPDATE SET value = excluded.value,"
            " source_sha256 = excluded.source_sha256, computed_at = excluded.computed_at",
            tuple(plain(v) for v in (file_id, sid, value, sha, now)),
        )
    if phash64 is not None:
        similarity.note(conn, similarity.PHASH, file_id, phash64, now)


def stale(conn, table: str) -> list[int]:
    """Rows whose source bytes have changed since they were computed.

    `IS NOT` rather than `<>`, because a file that has never been hashed has
    a NULL sha and `<>` is NULL-blind: with `<>` every derivation attached to
    an unhashed file reads as current forever.

    The table name is checked against the database rather than interpolated
    on trust. It is the one place in this package where a caller's string
    reaches the parser unbound, and the rest of the repo already refuses that
    shape -- see sglint (SG101).
    """
    known = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'derived\\_%' ESCAPE '\\'"
        )
    }
    if table not in known:
        raise ValueError(f"{table!r} is not a derived table")
    return [
        row[0]
        for row in conn.execute(
            f"SELECT DISTINCT d.file_id FROM {table} d JOIN file f ON f.id = d.file_id"
            " WHERE d.source_sha256 IS NOT f.content_sha256"
        )
    ]


# --- samples ---------------------------------------------------------------


def add_sample(conn, file_id: int, kind: str, policy: str, *, offset_ms=None, page_index=None) -> int:
    """A frame out of a video, or a page out of a document.

    Faces and captions on video attach to these rather than to the file, so
    a claim can say which moment it was looking at. `policy` records how the
    sample was chosen, because a result is only reproducible if the sampling
    is.

    Asking twice for the same moment returns the same row. As a plain INSERT
    this raised against `derived_media_sample_pos`, so a sampling job that
    was interrupted and resumed crashed on the first frame it had already
    taken -- the one case resumption exists for.
    """
    conn.execute(
        "INSERT INTO derived_media_sample(file_id, kind, offset_ms, page_index, policy)"
        " VALUES(?, ?, ?, ?, ?)"
        " ON CONFLICT(file_id, kind, IFNULL(offset_ms,-1), IFNULL(page_index,-1), policy)"
        " DO NOTHING",
        tuple(plain(value) for value in (file_id, kind, offset_ms, page_index, policy)),
    )
    row = conn.execute(
        "SELECT id FROM derived_media_sample WHERE file_id = ? AND kind = ?"
        " AND IFNULL(offset_ms,-1) = IFNULL(?,-1)"
        " AND IFNULL(page_index,-1) = IFNULL(?,-1) AND policy = ?",
        tuple(plain(value) for value in (file_id, kind, offset_ms, page_index, policy)),
    ).fetchone()
    return int(row[0]) if row else 0


# --- faces -----------------------------------------------------------------


def _insert_face(
    conn,
    file_id: int,
    region_id: int,
    model_id: str,
    model_version: str,
    sha: str,
    now: float,
    *,
    sample_id=None,
    det_score=None,
    landmarks=None,
    embedding=None,
    dim=None,
    age=None,
    sex=None,
    pose=None,
    native=None,
) -> int:
    """One detected face. The region is required: a detection with no
    location cannot be shown, cropped, checked, or asserted against.

    Private because it appends. `record_faces` is the way in -- a detector
    run has to replace what it said last time, and a public row-at-a-time
    insert is how "re-running a detector doubles every face" comes back.

    `pose` is a mapping keyed yaw/pitch/roll, never a triple. InsightFace's
    array is [pitch, yaw, roll] and these columns are yaw-first, so a
    positional unpack would write pitch into pose_yaw with no CHECK able to
    see it: three REAL columns holding plausible degrees either way.

    `native` is the producer's complete output as a `vision/facestore.py`
    envelope -- the canonical record. Every other value in this signature
    is a promotion out of it for the values a facet filters on or a page
    renders; none of them decides what survives.
    """
    if pose is None:
        yaw = pitch = roll = None
    elif hasattr(pose, "get"):
        yaw, pitch, roll = (pose.get(axis) for axis in ("yaw", "pitch", "roll"))
    else:
        # Refused by name rather than unpacked: a triple is ambiguous between the two
        # orders in play -- InsightFace emits [pitch, yaw, roll] (deepinsight/insightface
        # model_zoo/landmark.py:111) and these columns are yaw-first. No CHECK can see it.
        raise TypeError(
            f"face pose arrived as {type(pose).__name__}, which has no axis names: "
            f"pass a mapping keyed yaw/pitch/roll. A triple is ambiguous -- the "
            f"detector's array is [pitch, yaw, roll] and these columns are yaw-first, "
            f"so the swap it invites is invisible once written."
        )
    # `dim` describes `embedding`, so it is taken from it rather than trusted from a
    # caller, and the schema checks the two agree. The space id travels with the
    # embedding for the same reason: a vector of unknown space compares with nothing.
    space_id = None
    if embedding is not None:
        from . import similarity

        embedding = bytes(embedding)
        dim = len(embedding) // 4
        space_id = similarity.space_id(conn, similarity.face_space(model_id, model_version, dim), now)
    cursor = conn.execute(
        "INSERT INTO derived_face_instance(file_id, sample_id, region_id,"
        " landmarks, embedding, det_score, dim, age, sex, pose_yaw, pose_pitch,"
        " pose_roll, native, model_id, model_version, space_id, source_sha256,"
        " computed_at)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        tuple(
            plain(value)
            for value in (
                file_id,
                sample_id,
                region_id,
                landmarks,
                embedding,
                det_score,
                dim,
                age,
                sex,
                yaw,
                pitch,
                roll,
                native,
                model_id,
                model_version,
                space_id,
                sha,
                now,
            )
        ),
    )
    face_id = int(cursor.lastrowid or 0)
    if embedding is not None:
        import numpy as np

        from . import similarity

        similarity.note(
            conn,
            similarity.face_space(model_id, model_version, len(embedding) // 4),
            face_id,
            np.frombuffer(embedding, dtype=np.float32),
            now,
        )
    return face_id


def run_for(conn, model_id: str, model_version: str, method: str, threshold, now: float) -> int:
    """The row identifying one clustering run, created once.

    A run is (embedder, version, method, threshold) -- all four, because all
    four decide who ends up in a cluster. Asking twice for the same four
    returns the same run, so re-clustering at the same settings replaces its
    own answer and leaves everybody else's alone.
    """
    conn.execute(
        "INSERT INTO derived_face_run(model_id, model_version, method, threshold,"
        " computed_at) VALUES(?, ?, ?, ?, ?)"
        " ON CONFLICT(model_id, model_version, method, IFNULL(threshold, -1))"
        " DO UPDATE SET computed_at = excluded.computed_at",
        tuple(plain(v) for v in (model_id, model_version, method, threshold, now)),
    )
    return conn.execute(
        "SELECT id FROM derived_face_run WHERE model_id = ? AND model_version = ?"
        " AND method = ? AND IFNULL(threshold, -1) = IFNULL(?, -1)",
        tuple(plain(v) for v in (model_id, model_version, method, threshold)),
    ).fetchone()[0]


def runs(conn) -> list[dict]:
    """Every clustering the library holds, so two can be compared."""
    cursor = conn.execute(
        "SELECT id, model_id, model_version, method, threshold, is_primary,"
        " faces, clusters, computed_at FROM derived_face_run"
        " ORDER BY is_primary DESC, computed_at DESC"
    )
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor]


def make_primary(conn, run_id: int) -> None:
    """Choose the run the People page shows when nobody asked for one.

    Cleared first, then set: the partial unique index allows one, so setting
    a second without clearing the first is an IntegrityError rather than a
    silent second default.
    """
    conn.execute("UPDATE derived_face_run SET is_primary = 0 WHERE is_primary = 1")
    conn.execute("UPDATE derived_face_run SET is_primary = 1 WHERE id = ?", (run_id,))


def primary_run(conn) -> int | None:
    row = conn.execute("SELECT id FROM derived_face_run WHERE is_primary = 1").fetchone()
    return row[0] if row else None


def _insert_cluster(
    conn,
    run_id: int,
    model_id: str,
    model_version: str,
    now: float,
    *,
    person_id=None,
    centroid=None,
    dim=None,
) -> int:
    """Private for the same reason as `_insert_face`. `recluster` is the API."""
    cursor = conn.execute(
        "INSERT INTO derived_face_cluster(run_id, person_id, centroid, dim,"
        " model_id, model_version, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
        tuple(plain(value) for value in (run_id, person_id, centroid, dim, model_id, model_version, now)),
    )
    return int(cursor.lastrowid or 0)


def _reclaim_regions(conn, region_ids) -> None:
    """Drop the boxes the deleted rows were the only owner of.

    A region exists to locate something. Once nothing points at it, it is a
    rectangle about nothing, and leaving it behind means a re-run grows the
    table by every face it re-detected.
    """
    for region_id in {r for r in region_ids if r is not None}:
        conn.execute(
            "DELETE FROM region WHERE id = ?"
            " AND NOT EXISTS (SELECT 1 FROM derived_face_instance WHERE region_id = region.id)"
            " AND NOT EXISTS (SELECT 1 FROM derived_annotation    WHERE region_id = region.id)"
            " AND NOT EXISTS (SELECT 1 FROM person_assertion      WHERE region_id = region.id)",
            (region_id,),
        )


def record_face_scan(conn, file_id: int, model_id: str, model_version: str, sha: str, now: float, faces: int) -> None:
    """That this detector looked at this file's current bytes and found
    `faces` -- zero included. The faces sweep reads it to leave looked-at
    files alone (db/runner.py submit_faces)."""
    conn.execute(
        "INSERT INTO derived_face_scan(file_id, model_id, model_version, source_sha256, faces, computed_at)"
        " VALUES(?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(file_id, model_id, model_version) DO UPDATE SET source_sha256 = excluded.source_sha256,"
        " faces = excluded.faces, computed_at = excluded.computed_at",
        tuple(plain(v) for v in (file_id, model_id, model_version, sha, faces, now)),
    )


def record_faces(
    conn,
    file_id: int,
    model_id: str,
    model_version: str,
    sha: str,
    now: float,
    faces,
    *,
    sample_id=None,
) -> list[int]:
    """Every face this model found in this file, replacing what it found before.

    Scoped replacement rather than an upsert, because a detector's answer is
    the whole set and not a row: a version that finds two faces where the
    last one found three has to be able to say so. There is no natural key to
    upsert on either -- a face is located by a `region`, and a re-run mints a
    new region row, so keying on it would append forever: running the detector
    twice over one photograph would leave two copies of every face.

    `faces` is a sequence of mappings, one per detection: `region` (an id
    from `region()`) is required; `det_score`, `landmarks`, `dim`, `age`,
    `sex`, `pose` and `native` are optional. `native` is the producer's
    complete output (a `vision/facestore.py` envelope) and the others are
    promotions out of it -- `db/detect.py` fills both, and the promotions
    exist because a facet filters on them, not because they are the part
    worth keeping.

    A score outside 0..1 is refused here, by name, rather than left to the
    CHECK. Run over sixty real photographs, OpenCV's cascade reported reject
    levels from -0.714 to 11.97 -- not a confidence at all -- and the
    constraint did its job, with an IntegrityError from inside a loop that
    named neither the file, nor which of its faces, nor the value. The
    schema was right and unhelpful. Converting a model's raw output to a
    confidence is the caller's job, and finding out which caller got it
    wrong should not be an afternoon.
    """
    for index, face in enumerate(faces):
        score = plain(face.get("det_score"))
        if isinstance(score, (int, float)) and not 0.0 <= float(score) <= 1.0:
            raise ValueError(
                f"face {index} of file {file_id} reports det_score {score!r}: "
                f"scores are 0..1, never a model's raw output "
                f"({model_id} {model_version})"
            )
    from . import similarity

    doomed = []
    for face_id, region_id, sid in conn.execute(
        "SELECT id, region_id, space_id FROM derived_face_instance WHERE file_id = ?"
        " AND IFNULL(sample_id, 0) = IFNULL(?, 0)"
        " AND model_id = ? AND model_version = ?",
        (file_id, sample_id, model_id, model_version),
    ):
        doomed.append(region_id)
        # The replaced faces leave their space's live index too -- noted
        # here, applied by the runner only after the commit that made the
        # deletion durable.
        if sid is not None:
            similarity.note_gone(conn, sid, face_id)
    conn.execute(
        "DELETE FROM derived_face_instance WHERE file_id = ?"
        " AND IFNULL(sample_id, 0) = IFNULL(?, 0)"
        " AND model_id = ? AND model_version = ?",
        (file_id, sample_id, model_id, model_version),
    )
    written = [
        _insert_face(
            conn,
            file_id,
            face["region"],
            model_id,
            model_version,
            sha,
            now,
            sample_id=sample_id,
            det_score=face.get("det_score"),
            landmarks=face.get("landmarks"),
            embedding=face.get("embedding"),
            dim=face.get("dim"),
            age=face.get("age"),
            sex=face.get("sex"),
            pose=face.get("pose"),
            native=face.get("native"),
        )
        for face in faces
    ]
    _reclaim_regions(conn, doomed)
    return written


def recluster(
    conn, model_id: str, model_version: str, now: float, clusters, *, method: str = "given", threshold=None
) -> list[int]:
    """Every cluster one clustering RUN produced, replacing that run's last.

    A run is (model, version, method, threshold) -- all four, because all
    four decide who a cluster contains. Replacing on the model alone meant a
    second method could not coexist with the first, so the only way to
    compare two was to lose one.

    Memberships go with the clusters: `derived_face_membership` cascades, so
    the faces come out of this run unassigned and stay in every other run
    they belong to.

    `clusters` is a sequence of mappings with optional `centroid`, `dim` and
    `person_id`. Names are not carried across by hand -- run
    `seed_clusters_from_assertions` afterwards, which re-applies them from
    what people wrote down.
    """
    run_id = run_for(conn, model_id, model_version, method, threshold, now)
    conn.execute("DELETE FROM derived_face_cluster WHERE run_id = ?", (run_id,))
    made = [
        _insert_cluster(
            conn,
            run_id,
            model_id,
            model_version,
            now,
            person_id=cluster.get("person_id"),
            centroid=cluster.get("centroid"),
            dim=cluster.get("dim"),
        )
        for cluster in clusters
    ]
    conn.execute("UPDATE derived_face_run SET clusters = ? WHERE id = ?", (len(made), run_id))
    return made


#: The clustering the application runs when nobody asked for a different
#: one. Named once so submitters and the runner cannot drift apart on it.
DEFAULT_METHOD = "chinese-whispers"

#: Cosine similarity at which two vectors are taken to be the same face, per embedding
#: space; the spaces are not comparable. docs/FACE_CLUSTERING.md:42-45 states these
#: values and :71-74 what a mismatched threshold does to top-cluster share.
SAME_PERSON = {
    "opencv/yunet+arcface": 0.48,
    "opencv/yunet+sface": 0.55,
    "insightface": 0.40,
}
#: For an embedder nobody has measured here. Deliberately tight: a threshold
#: too high leaves people in several clusters, which somebody can merge; one
#: too low welds strangers together, which nobody can unpick.
UNMEASURED = 0.55


def threshold_for(model_id: str) -> float:
    """The measured threshold for this embedder, or a cautious default."""
    for known, value in SAME_PERSON.items():
        if model_id.startswith(known):
            return value
    return UNMEASURED


def cluster(
    conn,
    model_id: str,
    model_version: str,
    now: float,
    *,
    method: str = DEFAULT_METHOD,
    threshold: float | None = None,
    smallest: int = 2,
    **options,
) -> list[int]:
    """Group this model's faces by their vectors, and write the clusters.

    The step the People page is downstream of, and the one nothing could do:
    a face's embedding had no column to live in, so every test in this suite
    formed clusters by assigning `cluster_id` by hand -- which is not
    clustering, it is stating the answer.

    **Label propagation, not connected components.** Each face repeatedly
    adopts whichever label its neighbours agree on most strongly. Single
    linkage was the obvious thing and is the wrong thing -- the previous
    pipeline documented that "transitive chaining merges dense look-alike
    sets into one cluster" (git history) -- and it did: over 834 real faces it
    made one cluster of 123 spanning 53 different photographs, which is not
    a person, it is a chain of people who each slightly resemble the next.

    Chinese whispers, from the canonical implementation rather than from a
    description of it: a node's neighbours vote with their edge WEIGHTS
    summed per label, not with a count
    (davisking/dlib@f28ef50 dlib/clustering/chinese_whispers.h:48-53), and the
    winner is found with a strict `>` over a label-ordered map, so a tie
    goes to the lowest label id (:57-66).

    dlib picks nodes at random for `n * num_iterations` steps (:42-45). This
    sweeps them in index order instead and stops when a sweep changes
    nothing -- the deviation this repo already made and documented, for the
    same reason: the result becomes a pure function of the graph, and a
    library that reclusters twice gets the same people both times.

    Groups smaller than `smallest` are left unclustered rather than becoming
    one-face people: a singleton is not somebody you would recognise, it is
    a detection.

    Re-running replaces this model's clusters. Names are not carried across
    by similarity -- `seed_clusters_from_assertions` re-applies them from
    what a human wrote down, which is the whole reason the assertion exists.
    """
    import numpy as np

    from . import grouping, similarity

    if threshold is None:
        threshold = threshold_for(model_id)
    run_id = run_for(conn, model_id, model_version, method, threshold, now)
    # The space id is the clustering input's identity: rows are selected by which
    # immutable space produced them, never by reading the duplicated model columns. A
    # producer or preprocess upgrade mints a new space id, so old rows stop being input.
    current = similarity.face_space_of(conn, model_id, model_version)
    rows = []
    if current is not None:
        rows = conn.execute(
            "SELECT id, embedding FROM derived_face_instance WHERE space_id = ? ORDER BY id",
            (current[0],),
        ).fetchall()
    conn.execute("DELETE FROM derived_face_cluster WHERE run_id = ?", (run_id,))
    conn.execute(
        "UPDATE derived_face_run SET faces = ?, clusters = 0 WHERE id = ?",
        (len(rows), run_id),
    )
    if current is None or not rows:
        return []
    space = current[1]

    vectors = np.vstack([np.frombuffer(raw, dtype=np.float32) for _, raw in rows])
    face_ids = [int(row[0]) for row in rows]
    # Through the shared index layer -- the same resident manager the dupes job
    # searches -- so face similarity is not its own FAISS consumer. Align mutates the
    # live space to exactly these rows; device policy is the manager's configuration.
    manager = similarity.manager_for(conn)
    at = {face_id: position for position, face_id in enumerate(face_ids)}
    key = similarity.align(conn, manager, space, face_ids, lambda wanted: vectors[[at[int(v)] for v in wanted]], now)
    edges_a, edges_b, weights = similarity.pair_graph(manager, key, threshold)
    backend = manager.served_by(key)
    # grouping.group consumes the positional CSR shape; positions here are
    # the sorted face_ids the rows arrived in.
    graph = similarity.as_csr(
        len(face_ids),
        np.array([at[int(v)] for v in edges_a], dtype=np.int64),
        np.array([at[int(v)] for v in edges_b], dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )
    labels = grouping.group(graph, vectors, method, **options)

    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)

    made = []
    for members in groups.values():
        if len(members) < smallest:
            for index in members:
                conn.execute(
                    "DELETE FROM derived_face_membership WHERE face_id = ?"
                    " AND cluster_id IN (SELECT id FROM derived_face_cluster"
                    "   WHERE run_id = ?)",
                    (int(rows[index][0]), run_id),
                )
            continue
        centre = similarity.normalise(vectors)[members].mean(axis=0)
        scale = float(np.linalg.norm(centre)) or 1.0
        cluster_id = _insert_cluster(
            conn,
            run_id,
            model_id,
            model_version,
            now,
            centroid=(centre / scale).astype(np.float32).tobytes(),
            dim=int(vectors.shape[1]),
        )
        for index in members:
            conn.execute(
                "INSERT OR IGNORE INTO derived_face_membership(cluster_id, face_id) VALUES(?, ?)",
                (cluster_id, int(rows[index][0])),
            )
        made.append(cluster_id)
    conn.execute(
        "UPDATE derived_face_run SET clusters = ?, backend = ? WHERE id = ?",
        (len(made), backend, run_id),
    )
    if made:
        _adopt_if_better(conn, run_id, model_id, threshold)
    return made


#: A cluster holding more than this share of every face in the library has chained:
#: everybody who resembles somebody who resembles somebody, not a person. At a threshold
#: a tenth too loose the top cluster held 96% (docs/FACE_CLUSTERING.md, Chaining).
CHAINED = 0.5

#: And the other end: a run where nearly everything is alone has not grouped
#: anything, it has renamed faces.
ALL_ALONE = 0.95

#: Faces measured when judging a run, capping the cost regardless of library size. The
#: silhouette is a mean over faces, so a random sample estimates it, and each sampled
#: face is still measured against every grouped face as the definition requires.
SILHOUETTE_SAMPLE = 20_000
#: Rows per block when the silhouette matrices are walked; the peak allocation is
#: block x faces.
_BLOCK = 4096
#: One page of annotation search hits -- the same size as a grid page
#: (db/resultset.py DEFAULT_PAGE_SIZE), stated here because this module
#: keeps its imports lazy and a default argument cannot.
_ANNOTATIONS_PAGE = 60

#: Below this, the groups are not meaningfully apart -- a face sits about as
#: close to somebody else's centre as to its own -- and a run that scores it
#: should not become what the library shows without somebody saying so.
GOOD_ENOUGH = 0.10

#: Faces before the statistical gates apply at all: under this they misfire by construction,
#: since two photographs are most of a three-face library ("chained") and the silhouette is
#: defined only between two clusters and n-1 (Rousseeuw). Below it, any run that grouped is eligible.
JUDGEABLE = 20


def disqualification(reading: dict) -> str | None:
    """Why a run's shape bars it from becoming the default unasked, in
    words a person can act on -- or None when nothing does."""
    if reading["clusters"] == 0:
        return "it grouped nothing"
    if reading["faces"] < JUDGEABLE:
        return None
    if reading["largest_share"] > CHAINED:
        return (
            f"it chained: one group holds {reading['largest_share']:.0%} of every face"
            f" (more than {CHAINED:.0%}), which is not a person"
        )
    if reading["alone_share"] > ALL_ALONE:
        return f"{reading['alone_share']:.0%} of faces are alone (more than {ALL_ALONE:.0%}); it grouped nothing much"
    if reading["clusters"] > 1 and reading["silhouette"] < GOOD_ENOUGH:
        return (
            f"its groups are not apart: silhouette {reading['silhouette']:.2f} is under {GOOD_ENOUGH:.2f},"
            " a face sits about as close to somebody else's group as to its own"
        )
    return None


def _disqualified(reading: dict) -> bool:
    """Whether a run's shape bars it from becoming the default unasked."""
    return disqualification(reading) is not None


def standing(conn, run_id: int, model_id: str, threshold) -> str:
    """One sentence on where a run stands with the People page: the
    default, or why not -- the verdict `_adopt_if_better` reaches in
    silence, said out loud for the log and the page."""
    if conn.execute("SELECT is_primary FROM derived_face_run WHERE id = ?", (run_id,)).fetchone()[0]:
        return "the People page's default"
    why = disqualification(health(conn, run_id))
    if why is None and threshold is not None and abs(float(threshold) - threshold_for(model_id)) > 1e-9:
        why = f"threshold {float(threshold):.2f} is not this embedder's measured {threshold_for(model_id):.2f}"
    chosen = primary_run(conn)
    held = f"the default is run #{chosen}" if chosen is not None else "no run is the default, so /people is empty"
    if why is None:
        return f"sound but not adopted unasked; {held} -- POST /clusterings/choose to choose"
    return f"not adopted: {why}; {held}"


def health(conn, run_id: int) -> dict:
    """What can be measured about a clustering without knowing any answers.

    Nothing here needs a label, which is the point: a library nobody has
    named anybody in still has to be able to tell a run that chained from a
    run that worked.

    `silhouette` is the Rousseeuw silhouette coefficient, by its actual
    definition rather than a centroid shortcut wearing the name: for each
    face, `a` is its mean distance to the rest of its own cluster (over
    n-1), `b` is the smallest mean distance to any other cluster's members,
    and the score is `(b - a) / max(a, b)`; a singleton scores 0, and the
    number is only defined with at least two clusters
    (scikit-learn/scikit-learn@bb9d35b sklearn/metrics/cluster/
    _unsupervised.py:149-199, 211-230, 311-323). Distance is cosine
    distance, `1 - <x, y>` on unit vectors. 1 is dense and separated, 0 is
    an arbitrary split, negative is faces in the wrong groups.

    An earlier version measured distance-to-centroid instead and called it a
    silhouette. Measured on 103 labelled faces it also could not rank sound
    runs -- its best run was not the labels' best run -- so whichever number
    sits here is a gate against degenerate runs, never the judge of good
    ones. `agreement` against `person_assertion` is the judge.

    The distribution comes back with it, because a mean over cluster sizes
    hides the case that matters. One group of 400 and forty of 2 has a
    respectable mean and is a chained library; the median and the largest
    together say so where the mean does not.
    """
    import math

    import numpy as np

    faces = conn.execute(
        "SELECT count(*) FROM derived_face_instance fi"
        " WHERE fi.embedding IS NOT NULL AND fi.model_id ="
        "   (SELECT model_id FROM derived_face_run WHERE id = ?)"
        " AND fi.model_version ="
        "   (SELECT model_version FROM derived_face_run WHERE id = ?)",
        (run_id, run_id),
    ).fetchone()[0]

    rows = conn.execute(
        "SELECT m.cluster_id, fi.embedding FROM derived_face_membership m"
        "  JOIN derived_face_instance fi ON fi.id = m.face_id"
        "  JOIN derived_face_cluster c ON c.id = m.cluster_id"
        " WHERE c.run_id = ? AND fi.embedding IS NOT NULL",
        (run_id,),
    ).fetchall()

    reading = {
        "faces": faces,
        "clusters": 0,
        "grouped": 0,
        "largest": 0,
        "median": 0.0,
        "mean": 0.0,
        "largest_share": 0.0,
        "alone_share": 1.0 if faces else 0.0,
        "cohesion": 0.0,
        "separation": 0.0,
        "silhouette": 0.0,
        "outliers": 0,
    }
    if not rows:
        return reading

    from . import similarity

    labels = np.array([r[0] for r in rows], dtype=np.int64)
    unit = similarity.normalise(np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows]))
    ids, index = np.unique(labels, return_inverse=True)
    counts = np.bincount(index)
    sizes = np.sort(counts)[::-1]
    grouped = int(counts.sum())
    median = float(np.median(sizes))

    # Centroids as one scatter-add rather than a loop per cluster: at a
    # thousand people that loop is a thousand round trips through Python for
    # a single matrix operation.
    centres = np.zeros((len(ids), unit.shape[1]), dtype=np.float32)
    np.add.at(centres, index, unit)
    centres /= np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-12)

    # Sampled above a limit: the coefficient is a mean over faces, so a random sample
    # estimates it. Each sampled face is still measured against every grouped face,
    # because the definition needs the mean distance to whole clusters, not to centroids.
    picked = np.arange(len(labels))
    if len(picked) > SILHOUETTE_SAMPLE:
        picked = np.random.default_rng(0).choice(len(labels), SILHOUETTE_SAMPLE, replace=False)
    sample, sample_index = unit[picked], index[picked]

    own = np.einsum("ij,ij->i", sample, centres[sample_index])
    if len(ids) > 1:
        nearest = np.empty(len(sample), dtype=np.float32)
        for start in range(0, len(sample), _BLOCK):
            block = slice(start, start + _BLOCK)
            against = sample[block] @ centres.T
            against[np.arange(against.shape[0]), sample_index[block]] = -1.0
            nearest[block] = against.max(axis=1)
    else:
        nearest = np.zeros(len(sample), dtype=np.float32)

    silhouette = 0.0
    if len(ids) > 1:
        # Sum of cosine distances from each sampled face to each cluster, accumulated in
        # column blocks so the sample-by-faces matrix is never held whole. Grouped faces
        # are sorted by cluster first, so a block's per-cluster sums fall out of `reduceat`.
        by_cluster = np.argsort(index, kind="stable")
        sorted_unit, sorted_index = unit[by_cluster], index[by_cluster]
        sums = np.zeros((len(sample), len(ids)), dtype=np.float64)
        for start in range(0, grouped, _BLOCK):
            block = slice(start, start + _BLOCK)
            dist = 1.0 - sample @ sorted_unit[block].T
            here = sorted_index[block]
            bounds = np.flatnonzero(np.diff(here, prepend=here[0] - 1))
            sums[:, here[bounds]] += np.add.reduceat(dist, bounds, axis=1)
        mine = (np.arange(len(sample)), sample_index)
        # a: mean distance to the REST of the own cluster. The face's own
        # zero self-distance is in the sum, so dividing by n-1 removes it --
        # the same correction upstream applies (:315-317).
        a = sums[mine] / np.maximum(counts[sample_index] - 1, 1)
        others = sums / counts
        others[mine] = np.inf
        b = others.min(axis=1)
        scores = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
        silhouette = float(np.nan_to_num(scores).mean())

    reading.update(
        {
            "clusters": len(ids),
            "grouped": grouped,
            "largest": int(sizes[0]),
            "median": median,
            "mean": grouped / len(ids),
            "largest_share": int(sizes[0]) / faces if faces else 0.0,
            "alone_share": (faces - grouped) / faces if faces else 0.0,
            "sampled": len(picked),
            # How tightly a face sits to its own group's centre.
            "cohesion": float(own.mean()),
            # How close it sits to the nearest OTHER group's centre.
            "separation": float(nearest.mean()),
            # Rousseeuw silhouette, defined above. A gate, not a judge.
            "silhouette": silhouette,
            # Groups far larger than the middle of the distribution -- the shape
            # chaining makes, and invisible to a mean.
            "outliers": sum(1 for n in sizes if median and n > 4 * median),
        }
    )
    if math.isnan(reading["silhouette"]):
        reading["silhouette"] = 0.0
    return reading


def agreement(conn, run_id: int) -> dict:
    """How a run stands against what people actually said.

    `person_assertion` is the library's own ground truth -- somebody looked
    at a picture and named who is in it -- and nothing was reading it back to
    ask whether a clustering agrees. Two faces a person put under one name
    should be in one cluster; two faces they put under different names should
    not.

    Returns counts rather than a verdict. A library with three assertions
    cannot judge a clustering and should not pretend to; the caller can see
    how much evidence there was.
    """
    rows = conn.execute(
        "SELECT pa.person_id, m.cluster_id FROM person_assertion pa"
        "  JOIN derived_face_instance fi ON fi.file_id = pa.file_id"
        "  JOIN derived_face_membership m ON m.face_id = fi.id"
        "  JOIN derived_face_cluster c ON c.id = m.cluster_id AND c.run_id = ?"
        # Positive claims only: a denial says two faces are not the same person, so
        # counting it here would read as evidence that they are, and the measure would
        # improve every time somebody corrected the thing it measures.
        " WHERE pa.stance = 'is'",
        (run_id,),
    ).fetchall()
    together = apart = mixed = 0
    by_person: dict[int, set[int]] = {}
    by_cluster: dict[int, set[int]] = {}
    for person_id, cluster_id in rows:
        by_person.setdefault(person_id, set()).add(cluster_id)
        by_cluster.setdefault(cluster_id, set()).add(person_id)
    for clusters in by_person.values():
        together += 1 if len(clusters) == 1 else 0
        apart += 1 if len(clusters) > 1 else 0
    for people in by_cluster.values():
        mixed += 1 if len(people) > 1 else 0
    return {
        "asserted_people": len(by_person),
        "held_together": together,
        "split_apart": apart,
        "clusters_mixing_people": mixed,
    }


def _adopt_if_better(conn, run_id: int, model_id: str, threshold) -> None:
    """Choose the run the People page shows, on evidence rather than order.

    The order runs happen in is an accident; what they produced is not. Adopting
    by order alone lets a loop that tries 0.55 before 0.48 show three people
    where there are two.

    Never adopts a run that chained or grouped nothing -- a page showing one
    person called Everybody is worse than a page showing none -- and never
    overrides a choice somebody made: `make_primary` is how a person decides,
    and this only fills the gap before they have.
    """
    if conn.execute("SELECT count(*) FROM derived_face_run WHERE is_primary = 1").fetchone()[0]:
        return
    if _disqualified(health(conn, run_id)):
        return
    # Among sound runs, prefer the one at the threshold this embedder was measured at;
    # a run at another threshold is one somebody asked for to compare against, not a
    # default. A better-scoring run already at that threshold is left alone.
    if threshold is not None and abs(float(threshold) - threshold_for(model_id)) > 1e-9:
        return
    make_primary(conn, run_id)


def choose_primary(conn) -> int | None:
    """Re-rank every run and set the best one primary. Returns its id.

    The deliberate version of `_adopt_if_better`, for after several runs
    exist. The ranking is by what people said: measured on 103 labelled
    faces, the silhouette's best run was not the labels' best run, so no
    label-free statistic judges here -- it only disqualifies. Runs that
    chained, grouped nothing, or sit below `GOOD_ENOUGH` are out; among the
    sound ones:

    - with assertions on file, the run that keeps the most asserted people
      in one cluster each, without mixing two people into one, wins;
    - with none, the run at its embedder's measured threshold wins, because
      that number came from labelled data and the others came from a sweep.

    Calling this IS choosing, so it overwrites the current primary. The
    passive path never does.
    """
    sound = [run for run in runs(conn) if not _disqualified(health(conn, run["id"]))]
    if not sound:
        return None

    asserted = conn.execute("SELECT count(*) FROM person_assertion WHERE stance = 'is'").fetchone()[0]
    if asserted:

        def by_agreement(run):
            said = agreement(conn, run["id"])
            return (
                said["held_together"] - said["split_apart"] - said["clusters_mixing_people"],
                # Same agreement: fall through to the measured threshold.
                -abs(float(run["threshold"] or 0) - threshold_for(run["model_id"])),
            )

        best = max(sound, key=by_agreement)
    else:
        best = min(
            sound,
            key=lambda run: abs(float(run["threshold"] or 0) - threshold_for(run["model_id"])),
        )
    make_primary(conn, best["id"])
    return best["id"]


def assign_cluster(conn, face_id: int, cluster_id: int) -> None:
    """Put one detected face in one cluster.

    A face is in one cluster per RUN and in as many runs as have grouped it,
    so this adds a membership rather than overwriting a column -- which is
    what made a second clustering destroy the first one's answer.
    """
    conn.execute(
        "INSERT OR IGNORE INTO derived_face_membership(cluster_id, face_id) VALUES(?, ?)",
        (cluster_id, face_id),
    )


def unassign_cluster(conn, face_id: int, cluster_id: int) -> None:
    conn.execute(
        "DELETE FROM derived_face_membership WHERE face_id = ? AND cluster_id = ?",
        (face_id, cluster_id),
    )


def attribute(
    conn, file_id: int, person_id: int, run_id: int, model_id: str, model_version: str, *, face_count: int = 1
) -> None:
    """One clustering run's inference that this person appears in this file.

    Keyed on the run. Two runs disagree about who is in a picture -- that
    disagreement is the reason for running both -- and keyed on the embedder
    alone the second overwrote the first, so the cluster tables held two
    answers while this one held whichever wrote last.
    """
    conn.execute(
        "INSERT INTO derived_file_person(file_id, person_id, run_id, model_id,"
        " model_version, face_count) VALUES(?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(file_id, person_id, run_id)"
        " DO UPDATE SET face_count = excluded.face_count",
        tuple(plain(value) for value in (file_id, person_id, run_id, model_id, model_version, face_count)),
    )


#: How much of a box two boxes must share before they are taken to be the
#: same face. A human drawing a box round somebody and a detector finding
#: them do not agree to the pixel, and they do not have to.
_SAME_FACE = 0.3


def _overlap(a, b) -> float:
    """Intersection over union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    wide = min(ax + aw, bx + bw) - max(ax, bx)
    tall = min(ay + ah, by + bh) - max(ay, by)
    if wide <= 0 or tall <= 0:
        return 0.0
    inner = wide * tall
    return inner / (aw * ah + bw * bh - inner)


def seed_clusters_from_assertions(conn, run_id: int) -> int:
    """Re-attach names after a rebuild, from what people said rather than
    from what the previous clustering happened to decide.

    Matched by where in the picture, not by which file. Joining an assertion
    to a cluster on `file_id` alone and breaking the tie with
    `count(*) DESC LIMIT 1` mislabels every photograph of two people: with
    Alice and Bob both asserted into one frame, the arbitrary winner was
    written onto *both* clusters, so Bob was attached to the face the
    detector put on Alice and Alice's name left the library entirely. A
    photograph of two people is not an edge case.

    So an assertion carrying a `region` votes only for the cluster holding a
    face that overlaps it. An assertion with no region -- "she is in this
    picture", no box -- votes for every cluster with a face in that file,
    but only where that file names one person; in a group photo it says
    nothing, because it does not know which face it meant.

    A cluster whose votes name two different people is left unnamed. That is
    the point of naming from a record: where the record does not say, this
    does not invent, and an unnamed cluster is a question the People page can
    put to somebody who knows the answer.

    Returns the number of clusters named.
    """
    boxes = {row[0]: row[1:] for row in conn.execute("SELECT id, x, y, w, h FROM region")}
    # The file is part of the moment's identity: an assertion whose sample row belongs
    # to another file never votes, however its offset reads. A writer that produced
    # that shape is held to a silent skip rather than a cross-file mislabel.
    moments = {
        row[0]: (row[1], row[2], row[3], row[4])
        for row in conn.execute("SELECT id, file_id, kind, offset_ms, page_index FROM derived_media_sample")
    }
    # Positive claims vote, negative ones veto: one proposes a name for a cluster and
    # the other refuses one. Read apart, because a denial that merely failed to vote
    # would be indistinguishable from never having been said.
    assertions: dict[int, list[tuple[int, int | None, int | None]]] = {}
    denied: dict[int, list[tuple[int, int | None]]] = {}
    for person_id, file_id, sample_id, region_id, stance in conn.execute(
        "SELECT person_id, file_id, sample_id, region_id, stance FROM person_assertion"
    ):
        if stance == "is_not":
            denied.setdefault(file_id, []).append((person_id, region_id))
        else:
            assertions.setdefault(file_id, []).append((person_id, sample_id, region_id))

    votes: dict[int, set[int]] = {}
    vetoes: dict[int, set[int]] = {}
    for cluster_id, file_id, sample_id, region_id in conn.execute(
        "SELECT m.cluster_id, fi.file_id, fi.sample_id, fi.region_id"
        "  FROM derived_face_membership m"
        "  JOIN derived_face_instance fi ON fi.id = m.face_id"
        "  JOIN derived_face_cluster c ON c.id = m.cluster_id"
        " WHERE c.person_id IS NULL AND c.run_id = ?",
        (run_id,),
    ):
        # A denial naming a REGION refuses that person for the cluster holding
        # the face it overlaps. A denial naming no region is about the FILE and
        # acts through the attribution filter, never by vetoing the cluster.
        for person, box in denied.get(file_id, ()):
            if box is not None and _overlap(boxes[region_id], boxes[box]) >= _SAME_FACE:
                vetoes.setdefault(cluster_id, set()).add(person)

        claims = assertions.get(file_id, ())
        for person, on_sample, box in claims:
            # A claim about one frame says nothing about another: two frames of a video
            # can hold a face in the same part of the picture. Compared as moments, not
            # row ids, because a rebuild mints fresh sample rows for the same frame.
            if on_sample is not None and moments.get(on_sample) != moments.get(sample_id, ()):
                continue
            if box is not None:
                if _overlap(boxes[region_id], boxes[box]) >= _SAME_FACE:
                    votes.setdefault(cluster_id, set()).add(person)
                continue
            # No box. It can speak only where it is the sole claim over the
            # same ground, or it would name whichever face came first. The
            # same ground is the same MOMENT, by the same identity as above.
            alone = [
                other
                for other, other_sample, _ in claims
                if on_sample is None or other_sample is None or moments.get(other_sample) == moments.get(on_sample)
            ]
            if len(set(alone)) == 1 and not any(b is not None for _, _, b in claims):
                votes.setdefault(cluster_id, set()).add(person)

    named = 0
    for cluster_id, people in votes.items():
        # A vetoed name is not a name, applied after the vote rather than by filtering
        # the claims, so a cluster whose only proposal was refused ends up unnamed
        # rather than named by whatever came second.
        allowed = people - vetoes.get(cluster_id, set())
        if len(allowed) != 1:
            continue
        conn.execute(
            "UPDATE derived_face_cluster SET person_id = ? WHERE id = ?",
            (allowed.pop(), cluster_id),
        )
        named += 1
    conn.execute(
        "INSERT OR IGNORE INTO derived_file_person(file_id, person_id, run_id,"
        " model_id, model_version)"
        " SELECT fi.file_id, c.person_id, c.run_id, c.model_id, c.model_version"
        "   FROM derived_face_membership m"
        "   JOIN derived_face_instance fi ON fi.id = m.face_id"
        "   JOIN derived_face_cluster c ON c.id = m.cluster_id"
        "  WHERE c.person_id IS NOT NULL AND c.run_id = ?"
        # A denial stops the attribution too, not only the naming. A correctly named
        # cluster can still hold one face from a picture that person is not in, and
        # without this the name returns on that picture through the file attribution.
        "    AND NOT EXISTS (SELECT 1 FROM person_assertion pa"
        "                     WHERE pa.person_id = c.person_id AND pa.file_id = fi.file_id"
        "                       AND pa.stance = 'is_not')",
        (run_id,),
    )
    return named


def attributing_producers(conn, person_id: int, file_id: int) -> list[tuple[str, str]]:
    """Which producers put this person on this file, most recent run first.

    Read BEFORE withdrawing, because withdrawing is what makes the
    answer interesting: a correction judges the model whose output was
    corrected, and after the delete there is nothing left saying which
    one that was.

    Usually one. More than one means several runs agreed, and each of
    them was told the same thing by the same correction.
    """
    return [
        (str(model_id), str(model_version))
        for model_id, model_version in conn.execute(
            "SELECT DISTINCT model_id, model_version FROM derived_file_person"
            " WHERE person_id = ? AND file_id = ? ORDER BY run_id DESC",
            (person_id, file_id),
        )
    ]


def withdraw_attribution(conn, person_id: int, file_id: int) -> int:
    """Take a person off a file in the INFERRED layer; returns rows gone.

    The consequence of a denial, and it has to be immediate. The claim
    constrains the next clustering run (`seed_clusters_from_assertions`),
    but `derived_file_person` is what the page reads, so leaving it
    standing would show the picture contradicting the thing somebody
    just said until the next re-run -- which may be never.
    """
    return int(
        conn.execute(
            "DELETE FROM derived_file_person WHERE file_id = ? AND person_id = ?", (file_id, person_id)
        ).rowcount
        or 0
    )


# --- embeddings ------------------------------------------------------------


def record_embedding(conn, file_id: int, spec, vector, sha: str, now: float) -> int:
    """One whole-file embedding under its own IMMUTABLE row id, replacing
    what this space held for the file before.

    The row id -- not the file id -- is what the resident index stores:
    a file's embedding legitimately changes (re-embed after replaced
    bytes), so a replacement DELETES the old row and mints a new
    AUTOINCREMENT id. Index alignment then sees an id disappear and a
    new one appear, which is a diff it handles exactly; the old shape
    (same file id, new vector) was a divergence a crash between commit
    and index sync could make permanent, because the float path trusts
    matching ids to mean matching vectors.

    Both moves are noted for the post-commit sync: the old id leaves the
    live index and the new one enters it only after the commit that made
    this row durable.

    `vector` is float32, numpy or bytes, already normalised by the
    encoder that produced it. Returns the new embedding id.
    """
    import numpy as np

    from . import similarity

    unit = np.asarray(vector, dtype=np.float32) if not isinstance(vector, bytes) else np.frombuffer(vector, np.float32)
    sid = similarity.space_id(conn, spec, now)
    old = conn.execute(
        "SELECT id FROM derived_embedding WHERE file_id = ? AND space_id = ?", (plain(file_id), sid)
    ).fetchone()
    if old is not None:
        similarity.note_gone(conn, sid, int(old[0]))
        conn.execute("DELETE FROM derived_embedding WHERE id = ?", (int(old[0]),))
    cursor = conn.execute(
        "INSERT INTO derived_embedding(file_id, space_id, vector, source_sha256, computed_at) VALUES(?, ?, ?, ?, ?)",
        tuple(plain(value) for value in (file_id, sid, unit.tobytes(), sha, now)),
    )
    embedding_id = int(cursor.lastrowid or 0)
    similarity.note(conn, spec, embedding_id, unit, now)
    return embedding_id


def record_prompt_embedding(conn, prompt_id: int, spec, policy: str, vector, text_hash: str, now: float) -> int:
    """One prompt TEXT's vector under the provider's joint space and one
    query policy, keyed by the text hash it was computed from -- the
    same immutable-id discipline as record_embedding: a replacement is
    a new AUTOINCREMENT id, the old one leaves the index, both moves
    noted for the post-commit sync of the PROMPT lane of that space
    (db/prompts.py lane), never the media index."""
    import numpy as np

    from . import prompts, similarity

    unit = np.asarray(vector, dtype=np.float32) if not isinstance(vector, bytes) else np.frombuffer(vector, np.float32)
    sid = similarity.space_id(conn, spec, now)
    where = prompts.lane(policy)
    old = conn.execute(
        "SELECT id FROM derived_prompt_embedding WHERE prompt_id = ? AND space_id = ? AND policy_hash = ?",
        (plain(prompt_id), sid, policy),
    ).fetchone()
    if old is not None:
        similarity.note_gone(conn, sid, int(old[0]), lane=where)
        conn.execute("DELETE FROM derived_prompt_embedding WHERE id = ?", (int(old[0]),))
    cursor = conn.execute(
        "INSERT INTO derived_prompt_embedding(prompt_id, space_id, policy_hash, vector, source_text_hash, computed_at)"
        " VALUES(?, ?, ?, ?, ?, ?)",
        tuple(plain(value) for value in (prompt_id, sid, policy, unit.tobytes(), text_hash, now)),
    )
    embedding_id = int(cursor.lastrowid or 0)
    similarity.note(conn, spec, embedding_id, unit, now, lane=where)
    return embedding_id


# --- what a model said about the picture -----------------------------------


def annotate(
    conn,
    file_id: int,
    kind: str,
    text: str,
    model_id: str,
    model_version: str,
    sha: str,
    now: float,
    *,
    sample_id=None,
    region_id=None,
    confidence=None,
) -> int:
    """A caption, a description, a tag, text read out of the image.

    Re-running the same model on the same picture replaces its own answer
    rather than accumulating; two *different* models are kept side by side on
    purpose, because comparing them is the point of running both.
    """
    conn.execute(
        "INSERT INTO derived_annotation(file_id, sample_id, region_id, kind, text,"
        " confidence, model_id, model_version, source_sha256, computed_at)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(file_id, kind, model_id, model_version,"
        "             IFNULL(region_id, 0), IFNULL(sample_id, 0))"
        " DO UPDATE SET text = excluded.text, confidence = excluded.confidence,"
        " source_sha256 = excluded.source_sha256, computed_at = excluded.computed_at",
        tuple(
            plain(value)
            for value in (
                file_id,
                sample_id,
                region_id,
                kind,
                text,
                confidence,
                model_id,
                model_version,
                sha,
                now,
            )
        ),
    )
    row = conn.execute(
        "SELECT id FROM derived_annotation WHERE file_id = ? AND kind = ?"
        " AND model_id = ? AND model_version = ?"
        " AND IFNULL(region_id, 0) = IFNULL(?, 0) AND IFNULL(sample_id, 0) = IFNULL(?, 0)",
        (file_id, kind, model_id, model_version, region_id, sample_id),
    ).fetchone()
    return row[0] if row else 0


def said_about(conn, file_id: int, *, kind=None) -> list[dict]:
    sql = (
        "SELECT a.id, a.kind, a.text, a.confidence, a.model_id, a.model_version, a.region_id, a.sample_id,"
        " s.offset_ms AS offset_ms, (a.source_sha256 IS NOT f.content_sha256) AS stale"
        "  FROM derived_annotation a JOIN file f ON f.id = a.file_id"
        "  LEFT JOIN derived_media_sample s ON s.id = a.sample_id WHERE a.file_id = ?"
    )
    args: list = [file_id]
    if kind:
        sql += " AND a.kind = ?"
        args.append(kind)
    cursor = conn.execute(sql + " ORDER BY a.kind, a.model_id, s.offset_ms NULLS FIRST, a.id", args)
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor]


#: The lexical channel's name in a retrieval answer's provenance.
CAPTIONS = "captions"


def any_annotations(conn) -> bool:
    return conn.execute("SELECT 1 FROM derived_annotation LIMIT 1").fetchone() is not None


def rank_by_annotation(conn, phrase: str, limit: int, allowed=None) -> list[tuple[int, float]]:
    """Present files whose annotations mention any word of the phrase,
    best first by bm25 -- one row per file, its best annotation. Score
    is bm25 negated so higher is better, the way a cosine reads.
    `allowed` discards before enumeration: the caller fuses by rank."""
    words = phrase.split()
    if not words:
        return []
    match = " OR ".join('"' + word.replace('"', '""') + '"' for word in words)
    rows = conn.execute(
        "SELECT a.file_id, bm25(annotation_fts) FROM annotation_fts"
        "  JOIN derived_annotation a ON a.id = annotation_fts.rowid"
        "  JOIN file f ON f.id = a.file_id AND a.source_sha256 = f.content_sha256"
        " WHERE annotation_fts MATCH ? AND f.missing_since IS NULL"
        " ORDER BY bm25(annotation_fts), a.file_id",
        (match,),
    )
    best: dict[int, float] = {}
    for file_id, score in rows:
        if allowed is not None and file_id not in allowed:
            continue
        if file_id not in best:
            best[file_id] = -float(score)
        if len(best) >= limit:
            break
    return list(best.items())


def said_first(conn, file_ids, *, prefer: str | None = None) -> dict[int, str]:
    """One caption per file for a page of them -- the configured model's
    (`prefer`, the `caption_model` setting) when it has spoken, else the
    first by name -- or no entry. What a grid cell can say on hover."""
    ids = list(file_ids)
    if not ids:
        return {}
    told: dict[int, str] = {}
    for file_id, text in conn.execute(
        "SELECT a.file_id, a.text FROM derived_annotation a JOIN file f ON f.id = a.file_id"
        " WHERE a.kind = 'caption' AND a.sample_id IS NULL AND a.source_sha256 = f.content_sha256 AND a.file_id IN ("
        + ",".join("?" for _ in ids)
        + ") ORDER BY a.file_id, (a.model_id = ?) DESC, a.model_id, a.model_version",
        [*ids, prefer or ""],
    ):
        told.setdefault(file_id, text)
    return told


def search_annotations(conn, text: str, limit: int = _ANNOTATIONS_PAGE) -> list[dict]:
    """Find a picture by what a model said about it."""
    quoted = '"' + text.replace('"', '""') + '"'
    cursor = conn.execute(
        "SELECT a.file_id, a.kind, a.text, a.model_id FROM annotation_fts"
        "  JOIN derived_annotation a ON a.id = annotation_fts.rowid"
        " WHERE annotation_fts MATCH ? LIMIT ?",
        (quoted, limit),
    )
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor]


# --- taking the expensive thing with you -------------------------------------

#: A person's faces in the primary run, each with the provenance that makes the numbers
#: mean something: a naked vector without producer, preprocessing and dimensions cannot be
#: compared, reproduced or checked. `similarity_space` is that identity, immutable by trigger.

#: `content_sha256` is the join back to a photograph, and there is no path: the bytes
#: identify a picture in any library that holds it, where a path is a fact about this
#: machine.
FACES_OF = (
    "SELECT s.key AS space, s.representation, s.dimensions, s.metric,"
    "       s.producer, s.producer_version, s.preprocess, s.preprocess_version,"
    "       s.spec_hash, c.centroid, c.dim AS centroid_dim,"
    "       f.content_sha256 AS sha256, fi.det_score, fi.dim, fi.embedding,"
    "       r.x, r.y, r.w, r.h, cap.captured_at"
    "  FROM derived_face_cluster c"
    "  JOIN derived_face_membership m ON m.cluster_id = c.id"
    "  JOIN derived_face_instance fi ON fi.id = m.face_id"
    "  JOIN derived_face_run run ON run.id = c.run_id AND run.is_primary = 1"
    "  JOIN file f ON f.id = fi.file_id AND f.content_sha256 IS NOT NULL"
    "  JOIN region r ON r.id = fi.region_id"
    "  JOIN similarity_space s ON s.id = fi.space_id"
    "  LEFT JOIN capture cap ON cap.file_id = f.id"
    " WHERE c.person_id = ? AND fi.embedding IS NOT NULL"
    "   AND (? IS NULL OR cap.captured_at >= ?)"
    "   AND (? IS NULL OR cap.captured_at <= ?)"
    # Dated first, in time order, and the undated after them: SQLite
    # sorts NULL FIRST, which would have led the file with the pictures
    # whose camera never said when -- the least locatable ones.
    " ORDER BY cap.captured_at IS NULL, cap.captured_at, f.content_sha256"
)


def _floats(raw) -> list[float]:
    import numpy as np

    return [] if raw is None else [float(x) for x in np.frombuffer(raw, dtype=np.float32)]


def person_faces(conn, slug: str, *, since: float | None = None, until: float | None = None) -> dict | None:
    """`{person, name, spaces}` for an address, or None for no such
    person.

    The slug is resolved HERE, through `naming`, so a retired address
    still answers -- somebody exporting from a bookmark should not be
    told the person does not exist because they were renamed.
    """
    from . import naming

    found = naming.resolve(conn, "person", slug)
    if found is None:
        return None
    person_id, _live = found
    row = conn.execute("SELECT name FROM person WHERE id = ?", (person_id,)).fetchone()
    if row is None:
        return None
    return {
        "person": slug,
        "name": row[0],
        "spaces": faces_exported(conn, person_id, since=since, until=until),
    }


def faces_exported(conn, person_id: int, *, since: float | None = None, until: float | None = None) -> list[dict]:
    """A person's face vectors, grouped by the space that gives them
    meaning, each group with its centroid.

    Grouped rather than flat because a vector is only comparable to
    another from the SAME space: a library that has re-detected under a
    new model holds two representations of one person, and flattening
    them into one list would invite a comparison that means nothing.

    A date range is over CAPTURE time, so a picture whose camera never
    said when excludes itself the moment a range is given. That is the
    honest reading of "faces from 2019" and the surface says it.
    """
    cursor = conn.execute(FACES_OF, (person_id, since, since, until, until))
    columns = [c[0] for c in cursor.description]
    spaces: dict[str, dict] = {}
    for row in (dict(zip(columns, one, strict=True)) for one in cursor):
        held = spaces.setdefault(
            row["space"],
            {
                "space": row["space"],
                "representation": row["representation"],
                "dimensions": row["dimensions"],
                "metric": row["metric"],
                "producer": row["producer"],
                "producer_version": row["producer_version"],
                "preprocess": row["preprocess"],
                "preprocess_version": row["preprocess_version"],
                "spec_hash": row["spec_hash"],
                "centroid": _floats(row["centroid"]),
                "faces": [],
            },
        )
        held["faces"].append(
            {
                "sha256": row["sha256"],
                "captured_at": row["captured_at"],
                "det_score": row["det_score"],
                "dim": row["dim"],
                "region": {"x": row["x"], "y": row["y"], "w": row["w"], "h": row["h"]},
                "embedding": _floats(row["embedding"]),
            }
        )
    return list(spaces.values())
