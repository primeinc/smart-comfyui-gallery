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
    """
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
    """
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
    """
    cursor = conn.execute(
        "INSERT INTO derived_media_sample(file_id, kind, offset_ms, page_index, policy)"
        " VALUES(?, ?, ?, ?, ?)",
        (file_id, kind, offset_ms, page_index, policy),
    )
    return int(cursor.lastrowid or 0)


# --- faces -----------------------------------------------------------------


def add_face(
    conn, file_id: int, region_id: int, model_id: str, model_version: str,
    sha: str, now: float, *, sample_id=None, cluster_id=None, det_score=None,
    landmarks=None, dim=None, age=None, sex=None, pose=None,
) -> int:
    """One detected face. The region is required: a detection with no
    location cannot be shown, cropped, checked, or asserted against."""
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


def add_cluster(
    conn, model_id: str, model_version: str, now: float, *, person_id=None,
    centroid=None, dim=None,
) -> int:
    cursor = conn.execute(
        "INSERT INTO derived_face_cluster(person_id, centroid, dim, model_id,"
        " model_version, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
        (person_id, centroid, dim, model_id, model_version, now),
    )
    return int(cursor.lastrowid or 0)


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


def seed_clusters_from_assertions(conn, model_id: str, model_version: str) -> int:
    """Re-attach names after a rebuild, from what people said rather than
    from what the previous clustering happened to decide.

    A cluster containing a face in a file somebody asserted a person into
    inherits that person. Re-deriving the naming by centroid similarity
    instead would be a guess that usually works and silently does not when
    it fails -- and the thing being guessed at is the one part of the face
    pipeline a human actually authored.

    Returns the number of clusters named.
    """
    conn.execute(
        "UPDATE derived_face_cluster SET person_id = ("
        "  SELECT pa.person_id FROM derived_face_instance fi"
        "    JOIN person_assertion pa ON pa.file_id = fi.file_id"
        "   WHERE fi.cluster_id = derived_face_cluster.id"
        "   GROUP BY pa.person_id ORDER BY count(*) DESC LIMIT 1)"
        " WHERE person_id IS NULL AND model_id = ? AND model_version = ?",
        (model_id, model_version),
    )
    named = conn.execute(
        "SELECT count(*) FROM derived_face_cluster WHERE person_id IS NOT NULL"
        " AND model_id = ? AND model_version = ?",
        (model_id, model_version),
    ).fetchone()[0]
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
