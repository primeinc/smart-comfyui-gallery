"""Everything a model produced, and the contract that it can be thrown away.

One rule holds this group together: **drop every `derived_*` table, re-index,
and the library is unchanged.** Nothing here may be the only copy of anything
a person wrote. That is why the name lives on `person`, why the human claim
lives in `person_assertion`, and why `feedback` keeps its verdict with a
nulled pointer when its subject is deleted.

Staleness is keyed on `source_sha256`, never on a timestamp. A backup
restore, a sync client, or a copy rewrites mtime without changing a pixel,
and a library that re-ran every model on that would be unusable. If the bytes
are the same, the derivation still holds.

`model_id` and `model_version` are on every row for the same reason: upgrading
a detector invalidates its own output and nothing else, so an upgrade is a
targeted re-run rather than a full rebuild.
"""

from __future__ import annotations

import hashlib


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
    """Delete the derived namespace, minus what human assertions pin.

    Segregating these tables by name is what makes the rebuild contract a
    mechanical operation instead of a careful one -- there is no list to keep
    in step, because the prefix *is* the list. Two pinned exceptions, both
    for the same reason: a `person_assertion` locates its claim by a region
    and, on video, a sampled moment, and deleting either would corrupt the
    claim's discriminant while the FK quietly nulls the pointer.
    """
    names = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'derived\\_%' ESCAPE '\\' ORDER BY name"
        )
    ]
    for name in reversed(names):
        if name == "derived_media_sample":
            # A sample an assertion points at is the MOMENT the human's
            # claim is about. Deleted, the FK nulls the assertion's
            # sample_id and the seeder's cross-moment guard goes blind: a
            # video of two people holds the same box at two moments, every
            # rebuilt cluster collects both votes, and both names are
            # lost. Same rule as regions below. Re-detection reclaims the
            # kept rows through add_sample's upsert while the policy token
            # is unchanged; under a new token the kept row persists as a
            # moment-alias the seeder matches by moment, bounded by the
            # number of assertions.
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


def region(conn, x: float, y: float, w: float, h: float, *, mask: bytes | None = None) -> int:
    """A rectangle, in fractions of the frame.

    Normalized because a box in pixels is a box against one particular
    rendering: the same coordinates on a thumbnail or a re-encoded proxy
    point somewhere else. A mask goes to the blob store rather than to a
    path, so moving a cache directory cannot void it.
    """
    # A detector's box can run off the edge: a face at the side of the frame
    # is reported with the whole head's extent, and part of that is not in
    # the picture. Measured on 423 real YuNet detections, one overhung, by
    # 1% of the frame -- and the CHECK refused it, losing a real face over a
    # rounding of reality.
    #
    # So an overhang is trimmed to the frame, because the region says where
    # in the picture something is and the rest is not in the picture. Only an
    # overhang: a box more than half outside is not a face at the edge, it is
    # pixel coordinates being passed as fractions, and silently turning that
    # into a full-frame box would attach every face in the library to the
    # same rectangle.
    # Only a box that actually overhangs is rewritten. Clamping every box
    # unconditionally put floating-point error into ones that were already
    # inside -- 0.6 + 0.3 - 0.6 is 0.29999999999999993 -- so a coordinate
    # made a round trip it never asked for and came back different.
    if float(x) < 0 or float(y) < 0 or float(x) + float(w) > 1 or float(y) + float(h) > 1:
        left, top = min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0)
        right = min(max(float(x) + float(w), 0.0), 1.0)
        bottom = min(max(float(y) + float(h), 0.0), 1.0)
        kept = (right - left) * (bottom - top)
        asked = float(w) * float(h)
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


def region_from_pixels(conn, box, width: int, height: int, **kwargs) -> int:
    """The same, given pixels and the size they were measured against.

    Offered so a caller with pixel coordinates converts once, here, rather
    than each detector inventing its own convention.

    A zero dimension is what a truncated decode reports, and dividing by it
    raised ZeroDivisionError out of whatever job was running. It is refused
    by name instead: there is no rectangle inside a frame with no area.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"a {width}x{height} frame has nowhere to put a box")
    x, y, w, h = box
    return region(conn, x / width, y / height, w / width, h / height, **kwargs)


# --- content hashes --------------------------------------------------------


def record_hash(conn, file_id: int, sha: str, now: float, *, phash64=None, dhash64=None) -> None:
    """Perceptual hashes, keyed on the content hash they were taken from."""
    conn.execute(
        "INSERT INTO derived_file_hash(file_id, phash64, dhash64, source_sha256, computed_at)"
        " VALUES(?, ?, ?, ?, ?)"
        " ON CONFLICT(file_id) DO UPDATE SET phash64 = excluded.phash64,"
        " dhash64 = excluded.dhash64, source_sha256 = excluded.source_sha256,"
        " computed_at = excluded.computed_at",
        tuple(plain(value) for value in (file_id, phash64, dhash64, sha, now)),
    )


def stale(conn, table: str) -> list[int]:
    """Rows whose source bytes have changed since they were computed.

    `IS NOT` rather than `<>`, because a file that has never been hashed has
    a NULL sha and `<>` is NULL-blind: with `<>` every derivation attached to
    an unhashed file reads as current forever.

    The table name is checked against the database rather than interpolated
    on trust. It is the one place in this package where a caller's string
    reaches the parser unbound, and the rest of the repo already refuses that
    shape -- see tests/test_sql_is_built_from_structure_only.py.
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
) -> int:
    """One detected face. The region is required: a detection with no
    location cannot be shown, cropped, checked, or asserted against.

    Private because it appends. `record_faces` is the way in -- a detector
    run has to replace what it said last time, and a public row-at-a-time
    insert is how "re-running a detector doubles every face" comes back.
    """
    yaw, pitch, roll = pose or (None, None, None)
    # `dim` describes `embedding`, so it is taken from it rather than trusted
    # from a caller. The schema checks the two agree; deriving it here means
    # nobody has to be told twice.
    if embedding is not None:
        embedding = bytes(embedding)
        dim = len(embedding) // 4
    cursor = conn.execute(
        "INSERT INTO derived_face_instance(file_id, sample_id, region_id,"
        " landmarks, embedding, det_score, dim, age, sex, pose_yaw, pose_pitch,"
        " pose_roll, model_id, model_version, source_sha256, computed_at)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                model_id,
                model_version,
                sha,
                now,
            )
        ),
    )
    return int(cursor.lastrowid or 0)


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
    new region row, so keying on it would append forever. That is what this
    used to do: running the detector twice over one photograph left two
    copies of every face, and the only test in the suite ran it once.

    `faces` is a sequence of mappings, one per detection: `region` (an id
    from `region()`) is required; `det_score`, `landmarks`, `dim`, `age`,
    `sex` and `pose` are optional.

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
    doomed = [
        row[0]
        for row in conn.execute(
            "SELECT region_id FROM derived_face_instance WHERE file_id = ?"
            " AND IFNULL(sample_id, 0) = IFNULL(?, 0)"
            " AND model_id = ? AND model_version = ?",
            (file_id, sample_id, model_id, model_version),
        )
    ]
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

#: Cosine similarity at which two vectors are taken to be the same face, per
#: embedding space. The spaces are not comparable and a single number for all
#: of them is wrong for all but one: docs/FACE_CLUSTERING.md:63-65 gives the
#: shipped per-pipeline defaults, measured on labelled data.
#:
#: Getting this wrong is not a small error. At 0.363 -- SFace's
#: same-identity point, applied to ArcFace by mistake -- that document
#: measures a top-cluster share of 0.963: essentially the whole library in
#: one person. At 0.45 it is 0.462, at 0.6 it is 0.036 (:128-133).
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
    if threshold is None:
        threshold = threshold_for(model_id)
    run_id = run_for(conn, model_id, model_version, method, threshold, now)
    rows = conn.execute(
        "SELECT id, embedding FROM derived_face_instance"
        " WHERE model_id = ? AND model_version = ? AND embedding IS NOT NULL"
        " ORDER BY id",
        (model_id, model_version),
    ).fetchall()
    conn.execute("DELETE FROM derived_face_cluster WHERE run_id = ?", (run_id,))
    conn.execute(
        "UPDATE derived_face_run SET faces = ?, clusters = 0 WHERE id = ?",
        (len(rows), run_id),
    )
    if not rows:
        return []

    import numpy as np

    from . import grouping, settings, similarity

    vectors = np.vstack([np.frombuffer(raw, dtype=np.float32) for _, raw in rows])
    graph, backend = similarity.graph(
        vectors,
        threshold,
        backend=settings.value(conn, "similarity_backend"),
        gpu=settings.flag(conn, "faiss_gpu"),
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


#: A cluster holding more than this share of every face in the library has
#: chained: it is not a person, it is everybody who resembles somebody who
#: resembles somebody. The repo's own measurements show the shape -- at a
#: threshold a tenth too loose the top cluster held 96% of the library
#: (docs/FACE_CLUSTERING.md:128-133).
CHAINED = 0.5

#: And the other end: a run where nearly everything is alone has not grouped
#: anything, it has renamed faces.
ALL_ALONE = 0.95

#: Faces measured when judging a run. The silhouette is a mean over faces, so
#: a random sample of faces estimates it; each sampled face is still measured
#: against every grouped face, which is what the definition requires. This
#: caps the cost of judging a run regardless of library size.
SILHOUETTE_SAMPLE = 20_000

#: Below this, the groups are not meaningfully apart -- a face sits about as
#: close to somebody else's centre as to its own -- and a run that scores it
#: should not become what the library shows without somebody saying so.
GOOD_ENOUGH = 0.10

#: Faces before the statistical gates apply at all. Under this, they misfire
#: by construction: one person's two photographs are most of a three-face
#: library ("chained"), and a single cluster has no silhouette -- the number
#: is only defined between two clusters and n-1 (Rousseeuw). A library this
#: small is judged by the person looking at it, so any run that grouped
#: something is eligible and the usual ranking picks among them.
JUDGEABLE = 20


def _disqualified(reading: dict) -> bool:
    """Whether a run's shape bars it from becoming the default unasked."""
    if reading["clusters"] == 0:
        return True
    if reading["faces"] < JUDGEABLE:
        return False
    return (
        reading["largest_share"] > CHAINED
        or reading["alone_share"] > ALL_ALONE
        or (reading["clusters"] > 1 and reading["silhouette"] < GOOD_ENOUGH)
    )


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

    # SAMPLED above a limit: the coefficient is a mean over faces, so a
    # random sample of faces estimates it. Each sampled face is still
    # measured against EVERY grouped face -- the definition needs the mean
    # distance to whole clusters, and swapping in centroids is how the
    # previous, wrong version of this number was made.
    picked = np.arange(len(labels))
    if len(picked) > SILHOUETTE_SAMPLE:
        picked = np.random.default_rng(0).choice(len(labels), SILHOUETTE_SAMPLE, replace=False)
    sample, sample_index = unit[picked], index[picked]

    own = np.einsum("ij,ij->i", sample, centres[sample_index])
    if len(ids) > 1:
        nearest = np.empty(len(sample), dtype=np.float32)
        for start in range(0, len(sample), 4096):
            block = slice(start, start + 4096)
            against = sample[block] @ centres.T
            against[np.arange(against.shape[0]), sample_index[block]] = -1.0
            nearest[block] = against.max(axis=1)
    else:
        nearest = np.zeros(len(sample), dtype=np.float32)

    silhouette = 0.0
    if len(ids) > 1:
        # Sum of cosine distances from each sampled face to each cluster,
        # accumulated in column blocks so the sample-by-faces matrix is
        # never held whole. Grouped faces are sorted by cluster first so a
        # block's per-cluster sums fall out of one `reduceat` instead of a
        # Python loop over clusters.
        by_cluster = np.argsort(index, kind="stable")
        sorted_unit, sorted_index = unit[by_cluster], index[by_cluster]
        sums = np.zeros((len(sample), len(ids)), dtype=np.float64)
        for start in range(0, grouped, 4096):
            block = slice(start, start + 4096)
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
            "outliers": int(sum(1 for n in sizes if median and n > 4 * median)),
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
        "  JOIN derived_face_cluster c ON c.id = m.cluster_id AND c.run_id = ?",
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

    It used to be "whichever ran first", which meant a loop that happened to
    try 0.55 before 0.48 showed three people where there are two. The order
    runs happen in is an accident; what they produced is not.

    Never adopts a run that chained or grouped nothing -- a page showing one
    person called Everybody is worse than a page showing none -- and never
    overrides a choice somebody made: `make_primary` is how a person decides,
    and this only fills the gap before they have.
    """
    if conn.execute("SELECT count(*) FROM derived_face_run WHERE is_primary = 1").fetchone()[0]:
        return
    if _disqualified(health(conn, run_id)):
        return
    # Among runs that are sound, prefer the one at the threshold this embedder
    # was actually measured at; a run at another threshold is one somebody
    # asked for to compare against, not a default. If a better-scoring run
    # exists at that threshold already, leave it alone.
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

    asserted = conn.execute("SELECT count(*) FROM person_assertion").fetchone()[0]
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
    # The file is part of the moment's identity: an assertion whose sample
    # row belongs to ANOTHER file must never vote, however its offset reads
    # -- nothing shipped writes that shape, and this keeps a future buggy
    # writer at a silent skip instead of a cross-file mislabel.
    moments = {
        row[0]: (row[1], row[2], row[3], row[4])
        for row in conn.execute("SELECT id, file_id, kind, offset_ms, page_index FROM derived_media_sample")
    }
    assertions: dict[int, list[tuple[int, int | None, int | None]]] = {}
    for person_id, file_id, sample_id, region_id in conn.execute(
        "SELECT person_id, file_id, sample_id, region_id FROM person_assertion"
    ):
        assertions.setdefault(file_id, []).append((person_id, sample_id, region_id))

    votes: dict[int, set[int]] = {}
    for cluster_id, file_id, sample_id, region_id in conn.execute(
        "SELECT m.cluster_id, fi.file_id, fi.sample_id, fi.region_id"
        "  FROM derived_face_membership m"
        "  JOIN derived_face_instance fi ON fi.id = m.face_id"
        "  JOIN derived_face_cluster c ON c.id = m.cluster_id"
        " WHERE c.person_id IS NULL AND c.run_id = ?",
        (run_id,),
    ):
        claims = assertions.get(file_id, ())
        for person, on_sample, box in claims:
            # A claim about one frame says nothing about another. Two frames
            # of a video can hold a face in the same part of the picture, so
            # without this the box match reaches across moments and a video of
            # two people mislabels the same way a photograph of two people did.
            # Compared as MOMENTS, not row ids: a rebuild under a new policy
            # token mints fresh sample rows for the same frame, and the claim
            # is about the frame, never about the token that chose it.
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
        if len(people) != 1:
            continue
        conn.execute(
            "UPDATE derived_face_cluster SET person_id = ? WHERE id = ?",
            (people.pop(), cluster_id),
        )
        named += 1
    conn.execute(
        "INSERT OR IGNORE INTO derived_file_person(file_id, person_id, run_id,"
        " model_id, model_version)"
        " SELECT fi.file_id, c.person_id, c.run_id, c.model_id, c.model_version"
        "   FROM derived_face_membership m"
        "   JOIN derived_face_instance fi ON fi.id = m.face_id"
        "   JOIN derived_face_cluster c ON c.id = m.cluster_id"
        "  WHERE c.person_id IS NOT NULL AND c.run_id = ?",
        (run_id,),
    )
    return named


# --- embeddings ------------------------------------------------------------


def add_embedding(
    conn,
    file_id: int,
    space: str,
    model_id: str,
    model_version: str,
    vector: bytes,
    dim: int,
    sha: str,
    now: float,
) -> None:
    conn.execute(
        "INSERT INTO derived_embedding(file_id, space, vector, dim, model_id,"
        " model_version, source_sha256, computed_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(file_id, space, model_id, model_version) DO UPDATE SET"
        " vector = excluded.vector, dim = excluded.dim,"
        " source_sha256 = excluded.source_sha256, computed_at = excluded.computed_at",
        tuple(plain(value) for value in (file_id, space, vector, dim, model_id, model_version, sha, now)),
    )


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
        "SELECT id, kind, text, confidence, model_id, model_version, region_id, sample_id"
        "  FROM derived_annotation WHERE file_id = ?"
    )
    args: list = [file_id]
    if kind:
        sql += " AND kind = ?"
        args.append(kind)
    cursor = conn.execute(sql + " ORDER BY kind, model_id", args)
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor]


def search_annotations(conn, text: str, limit: int = 60) -> list[dict]:
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
