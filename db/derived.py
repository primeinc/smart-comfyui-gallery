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


def drop_all(conn) -> list[str]:
    """Delete the whole derived namespace.

    Segregating these tables by name is what makes the rebuild contract a
    mechanical operation instead of a careful one -- there is no list to keep
    in step, because the prefix *is* the list.
    """
    names = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
            " AND name LIKE 'derived\\_%' ESCAPE '\\' ORDER BY name"
        )
    ]
    for name in reversed(names):
        conn.execute(f"DELETE FROM {name}")
    # Regions exist only to locate derived findings and human assertions. The
    # ones an assertion still points at are kept; the rest went with the rows
    # that cited them.
    conn.execute(
        "DELETE FROM region WHERE id NOT IN"
        " (SELECT region_id FROM person_assertion WHERE region_id IS NOT NULL)"
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
    mask_hash = None
    if mask:
        mask_hash = hashlib.sha256(mask).hexdigest()
        conn.execute(
            "INSERT OR IGNORE INTO blob(hash, payload_bin, byte_len) VALUES(?, ?, ?)",
            (mask_hash, mask, len(mask)),
        )
    cursor = conn.execute(
        "INSERT INTO region(x, y, w, h, mask_hash) VALUES(?, ?, ?, ?, ?)",
        (x, y, w, h, mask_hash),
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
        (file_id, phash64, dhash64, sha, now),
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
            "SELECT name FROM sqlite_master WHERE type = 'table'"
            " AND name LIKE 'derived\\_%' ESCAPE '\\'"
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


def add_sample(
    conn, file_id: int, kind: str, policy: str, *, offset_ms=None, page_index=None
) -> int:
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
        (file_id, kind, offset_ms, page_index, policy),
    )
    row = conn.execute(
        "SELECT id FROM derived_media_sample WHERE file_id = ? AND kind = ?"
        " AND IFNULL(offset_ms,-1) = IFNULL(?,-1)"
        " AND IFNULL(page_index,-1) = IFNULL(?,-1) AND policy = ?",
        (file_id, kind, offset_ms, page_index, policy),
    ).fetchone()
    return int(row[0]) if row else 0


# --- faces -----------------------------------------------------------------


def _insert_face(
    conn, file_id: int, region_id: int, model_id: str, model_version: str,
    sha: str, now: float, *, sample_id=None, cluster_id=None, det_score=None,
    landmarks=None, dim=None, age=None, sex=None, pose=None,
) -> int:
    """One detected face. The region is required: a detection with no
    location cannot be shown, cropped, checked, or asserted against.

    Private because it appends. `record_faces` is the way in -- a detector
    run has to replace what it said last time, and a public row-at-a-time
    insert is how "re-running a detector doubles every face" comes back.
    """
    yaw, pitch, roll = pose if pose else (None, None, None)
    cursor = conn.execute(
        "INSERT INTO derived_face_instance(file_id, sample_id, cluster_id, region_id,"
        " landmarks, det_score, dim, age, sex, pose_yaw, pose_pitch, pose_roll,"
        " model_id, model_version, source_sha256, computed_at)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            file_id, sample_id, cluster_id, region_id, landmarks, det_score, dim,
            age, sex, yaw, pitch, roll, model_id, model_version, sha, now,
        ),
    )
    return int(cursor.lastrowid or 0)


def _insert_cluster(
    conn, model_id: str, model_version: str, now: float, *, person_id=None,
    centroid=None, dim=None,
) -> int:
    """Private for the same reason as `_insert_face`. `recluster` is the API."""
    cursor = conn.execute(
        "INSERT INTO derived_face_cluster(person_id, centroid, dim, model_id,"
        " model_version, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
        (person_id, centroid, dim, model_id, model_version, now),
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
    conn, file_id: int, model_id: str, model_version: str, sha: str, now: float,
    faces, *, sample_id=None,
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
    """
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
            conn, file_id, face["region"], model_id, model_version, sha, now,
            sample_id=sample_id, cluster_id=face.get("cluster_id"),
            det_score=face.get("det_score"), landmarks=face.get("landmarks"),
            dim=face.get("dim"), age=face.get("age"), sex=face.get("sex"),
            pose=face.get("pose"),
        )
        for face in faces
    ]
    _reclaim_regions(conn, doomed)
    return written


def recluster(conn, model_id: str, model_version: str, now: float, clusters) -> list[int]:
    """Every cluster this model's clustering produced, replacing the last run.

    Clustering is a whole-library answer, so a re-run replaces the whole set
    for this (model, version). The instances survive: `cluster_id` is
    `ON DELETE SET NULL`, so they come out unassigned and the caller assigns
    them to the new ids.

    `clusters` is a sequence of mappings with optional `centroid`, `dim` and
    `person_id`. Names are not carried across by hand -- run
    `seed_clusters_from_assertions` afterwards, which re-applies them from
    what people wrote down.
    """
    conn.execute(
        "DELETE FROM derived_face_cluster WHERE model_id = ? AND model_version = ?",
        (model_id, model_version),
    )
    return [
        _insert_cluster(
            conn, model_id, model_version, now,
            person_id=cluster.get("person_id"), centroid=cluster.get("centroid"),
            dim=cluster.get("dim"),
        )
        for cluster in clusters
    ]


def assign_cluster(conn, face_id: int, cluster_id: int | None) -> None:
    """Put one detected face in a cluster, or take it out of one."""
    conn.execute(
        "UPDATE derived_face_instance SET cluster_id = ? WHERE id = ?",
        (cluster_id, face_id),
    )


def attribute(conn, file_id: int, person_id: int, model_id: str, model_version: str,
              *, face_count: int = 1) -> None:
    """A model's inference that this person appears in this file."""
    conn.execute(
        "INSERT INTO derived_file_person(file_id, person_id, model_id, model_version,"
        " face_count) VALUES(?, ?, ?, ?, ?)"
        " ON CONFLICT(file_id, person_id, model_id, model_version)"
        " DO UPDATE SET face_count = excluded.face_count",
        (file_id, person_id, model_id, model_version, face_count),
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


def seed_clusters_from_assertions(conn, model_id: str, model_version: str) -> int:
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
    boxes = {
        row[0]: row[1:]
        for row in conn.execute("SELECT id, x, y, w, h FROM region")
    }
    assertions: dict[int, list[tuple[int, int | None, int | None]]] = {}
    for person_id, file_id, sample_id, region_id in conn.execute(
        "SELECT person_id, file_id, sample_id, region_id FROM person_assertion"
    ):
        assertions.setdefault(file_id, []).append((person_id, sample_id, region_id))

    votes: dict[int, set[int]] = {}
    for cluster_id, file_id, sample_id, region_id in conn.execute(
        "SELECT fi.cluster_id, fi.file_id, fi.sample_id, fi.region_id"
        "  FROM derived_face_instance fi"
        "  JOIN derived_face_cluster c ON c.id = fi.cluster_id"
        " WHERE c.person_id IS NULL AND c.model_id = ? AND c.model_version = ?",
        (model_id, model_version),
    ):
        claims = assertions.get(file_id, ())
        for person, on_sample, box in claims:
            # A claim about one frame says nothing about another. Two frames
            # of a video can hold a face in the same part of the picture, so
            # without this the box match reaches across moments and a video of
            # two people mislabels the same way a photograph of two people did.
            if on_sample is not None and on_sample != sample_id:
                continue
            if box is not None:
                if _overlap(boxes[region_id], boxes[box]) >= _SAME_FACE:
                    votes.setdefault(cluster_id, set()).add(person)
                continue
            # No box. It can speak only where it is the sole claim over the
            # same ground, or it would name whichever face came first.
            alone = [
                other for other, other_sample, _ in claims
                if on_sample is None or other_sample in (None, on_sample)
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
        "INSERT OR IGNORE INTO derived_file_person(file_id, person_id, model_id, model_version)"
        " SELECT fi.file_id, c.person_id, c.model_id, c.model_version"
        "   FROM derived_face_instance fi JOIN derived_face_cluster c ON c.id = fi.cluster_id"
        "  WHERE c.person_id IS NOT NULL AND c.model_id = ? AND c.model_version = ?",
        (model_id, model_version),
    )
    return named


# --- embeddings ------------------------------------------------------------


def add_embedding(
    conn, file_id: int, space: str, model_id: str, model_version: str,
    vector: bytes, dim: int, sha: str, now: float,
) -> None:
    conn.execute(
        "INSERT INTO derived_embedding(file_id, space, vector, dim, model_id,"
        " model_version, source_sha256, computed_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(file_id, space, model_id, model_version) DO UPDATE SET"
        " vector = excluded.vector, dim = excluded.dim,"
        " source_sha256 = excluded.source_sha256, computed_at = excluded.computed_at",
        (file_id, space, vector, dim, model_id, model_version, sha, now),
    )


# --- what a model said about the picture -----------------------------------


def annotate(
    conn, file_id: int, kind: str, text: str, model_id: str, model_version: str,
    sha: str, now: float, *, sample_id=None, region_id=None, confidence=None,
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
        (
            file_id, sample_id, region_id, kind, text, confidence,
            model_id, model_version, sha, now,
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
    return [dict(zip(columns, row)) for row in cursor]


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
    return [dict(zip(columns, row)) for row in cursor]
