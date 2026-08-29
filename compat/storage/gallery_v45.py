"""The gallery database at user_version 45, as one storage candidate.

Write path: `vision/faces.py:442` converts, `db/detect.py:96-131` records,
`db/derived.py:304-410` inserts.

Read path: none in the application. Classified 2026-08-28, same tool, scope
and cwd, positive control first:

    rg "SELECT[^\"']*embedding"  *.py, excl compat  -> 13 files   MATCH
    rg "SELECT[^\"']*landmarks"  *.py, excl compat  -> rc=1       VALIDATED_EMPTY
    rg "SELECT \\*[^\"']*derived_face_instance"      -> rc=1       VALIDATED_EMPTY

`load` below is therefore the suite's own reader. Each conversion is the
inverse of a cited line. A key with no total inverse is omitted, not
approximated.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from compat.storage.contract import Observation

#: Fixed instant. A clock inside evidence differs between identical runs.
NOW: Final[float] = 1_700_000_000.0


class _Detected:
    """Supplies `FaceAnalysis.get`, the only app method
    `InsightFaceBackend.detect` calls (vision/faces.py:456)."""

    def __init__(self, found: list[Any]) -> None:
        self._found = found

    def get(self, bgr: Any) -> list[Any]:
        del bgr  # the detection is fixed
        return self._found


def replaying(found: list[Any]) -> Any:
    """`InsightFaceBackend` with `_app` pinned, `detect` inherited unchanged.

    `detect` reads only `_app`, `_min_det_score` and `_min_face_px`
    (vision/faces.py:442-499). `__init__` does not call up: the parent loads a
    143 MB pack whose output is already in hand. Holding the detection fixed
    leaves persistence as the only variable.
    """
    from vision import faces

    class Replaying(faces.InsightFaceBackend):
        def __init__(self) -> None:
            self._app = _Detected(found)
            self._min_det_score = faces.DEFAULT_MIN_DET_SCORE
            self._min_face_px = faces.DEFAULT_MIN_FACE_PX

    return Replaying()


def _genders() -> dict[str, int]:
    """Inverse of the stored `sex` word, composed from the two mappings.

    `Face.sex` is `'M' if gender == 1 else 'F'`
    (deepinsight/insightface@7fadd420c2351d0ffa8cac403421c1a3ed733365
    python-package/insightface/app/common.py:45-48); `sex_word`
    (vision/faces.py:400) turns the code into the word. Words that more than
    one gender maps to are dropped, so a non-injective mapping omits the key
    instead of inventing a value.
    """
    from vision.faces import sex_word

    seen: dict[str, list[int]] = {}
    for gender in (0, 1):
        seen.setdefault(sex_word("M" if gender == 1 else "F"), []).append(gender)
    return {word: found[0] for word, found in seen.items() if len(found) == 1}


def _library(conn: sqlite3.Connection, folder: Path, name: str, sha: str, size: tuple[int, int]) -> int:
    """A root, a folder and one file, as `db/scan.py` mints them.

    `width` and `height` are required by the read path: every stored
    coordinate is a fraction of the frame (vision/faces.py:462-465).
    """
    from db import authored, library, scan

    root_id = library.add_root(conn, folder, "library", NOW)
    folder_id = scan.ensure_folder(conn, root_id, None, folder.name)
    file_id = scan.mint(conn, "file", name)
    conn.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256,"
        " width, height, first_seen_at, last_seen_at)"
        " VALUES(?, ?, ?, 'image', 1, 0, ?, ?, ?, ?, ?)",
        (file_id, folder_id, name, sha, size[0], size[1], NOW, NOW),
    )
    authored.add_user(conn, "compat", "hash", "ADMIN", NOW)
    return file_id


def _pixels(points: Any, width: int, height: int) -> npt.NDArray[np.float32]:
    """Fractions to source pixels; inverse of vision/faces.py:462-465.

    float64 multiply, cast once at the end: a float32 multiply would add an
    error to the one being measured.
    """
    # `np.array`, not `np.asarray`: this multiplies IN PLACE, and asarray
    # returns its input unchanged when the dtype already matches. The blob
    # used to be float32, so the float64 request always forced a conversion
    # and produced a writable array by accident; at v46 it is already float64
    # and `np.frombuffer` hands back a READ-ONLY view of the sqlite buffer.
    # Every one of the 36 storage cases then failed with "output array is
    # read-only" and the lane reported 0 FAIL, exit 0 -- a population that
    # vanished and read as clean.
    values = np.array(points, dtype=np.float64, copy=True)
    values[..., 0] *= width
    values[..., 1] *= height
    return values.astype(np.float32)


def load(conn: sqlite3.Connection, file_id: int) -> Observation:
    """The stored row in the producer's vocabulary. Omits keys with no total
    inverse; the comparison reports the absence as a shape divergence."""
    row = conn.execute(
        "SELECT r.x, r.y, r.w, r.h, f.landmarks, f.embedding, f.det_score,"
        " f.age, f.sex, f.pose_yaw, f.pose_pitch, f.pose_roll, f.attributes"
        " FROM derived_face_instance f JOIN region r ON r.id = f.region_id"
        " WHERE f.file_id = ? ORDER BY f.id",
        (file_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"file {file_id} has no stored face: the write path recorded nothing")
    x, y, w, h, landmarks, embedding, det_score, age, sex, yaw, pitch, roll, attributes = row
    (width, height) = conn.execute("SELECT width, height FROM file WHERE id = ?", (file_id,)).fetchone()

    held: dict[str, npt.NDArray[np.generic]] = {}

    # region (x, y, w, h) -> the (x1, y1, x2, y2) the producer emits.
    corners = np.array([[x, y], [x + w, y + h]], dtype=np.float64)
    held["bbox"] = _pixels(corners, width, height).reshape(-1)

    if landmarks:
        # float64, matching what `db/detect.py` now writes. The blob held
        # normalized coordinates at float32, which cost 2.4e-4 source pixels
        # on the largest corpus frame -- measured by this lane, and the reason
        # the schema moved to v46.
        held["kps"] = _pixels(np.frombuffer(landmarks, dtype=np.float64).reshape(-1, 2), width, height)
    if embedding:
        held["embedding"] = np.frombuffer(embedding, dtype=np.float32).copy()
    if det_score is not None:
        # NARROWED back to the producer's width. SQLite REAL is 8-byte IEEE
        # and returns float64 whatever was written, so a float32 measurement
        # comes back wider than it went in -- the VALUE is exact either way
        # (this lane measured the difference at 0.0), but the dtype is part of
        # what a consumer receives: ReActor's `save_face_model` builds
        # `torch.tensor(face["det_score"])`, and a float64 there produces a
        # different tensor from upstream's.
        #
        # Leaving it wide was a choice this file made and then described as a
        # storage divergence. It is not one: the store knows the column holds
        # a float32 measurement, so returning it as one is what reading the
        # value back means.
        held["det_score"] = np.asarray(det_score, dtype=np.float32)
    if age is not None:
        held["age"] = np.asarray(int(age), dtype=np.int64)
    if sex is not None:
        gender = _genders().get(str(sex))
        if gender is not None:
            held["gender"] = np.asarray(gender, dtype=np.int64)

    # Columns are yaw-first; the producer's array is [pitch, yaw, roll]
    # (deepinsight/insightface@7fadd420c2351d0ffa8cac403421c1a3ed733365
    # python-package/insightface/model_zoo/landmark.py:111).
    if None not in (yaw, pitch, roll):
        held["pose"] = np.array([pitch, yaw, roll], dtype=np.float32)

    record: dict[str, Any] = json.loads(attributes) if attributes else {}
    if "landmark_2d_106" in record:
        held["landmark_2d_106"] = _pixels(np.asarray(record["landmark_2d_106"]), width, height)
    if "landmark_3d_68" in record:
        # z carries no image dimension and is stored unnormalised
        # (vision/faces.py:479-485).
        dense = np.asarray(record["landmark_3d_68"], dtype=np.float64)
        dense[:, 0] *= width
        dense[:, 1] *= height
        held["landmark_3d_68"] = dense.astype(np.float32)

    return Observation(held)


class GalleryV45:
    """`db/detect.py` and `db/derived.py` at user_version 45."""

    name = "gallery_v45"
    described = "derived_face_instance + region, via db/detect.py harvest"

    def round_trip(self, face: Any, frame: npt.NDArray[np.uint8], sha: str) -> Observation:
        """Write through `detect.harvest`, read through `load`.

        A file rather than `:memory:`: `db/connect.py:79` converts the journal
        to WAL, which a memory database cannot hold.
        """
        import cv2
        from PIL import Image

        from db import connect, detect

        height, width = frame.shape[:2]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with tempfile.TemporaryDirectory(prefix="compat-gallery-v45-") as scratch:
            holding = Path(scratch)
            path = holding / "library.sgly"
            connect.create(path)
            conn = connect.connect(path)
            try:
                folder = holding / "lib"
                folder.mkdir()
                file_id = _library(conn, folder, "corpus.jpg", sha, (width, height))
                written = detect.harvest(conn, replaying([face]), file_id, None, NOW, image=image)
                if not written:
                    raise ValueError("harvest recorded no face: the write path stored nothing")
                return load(conn, file_id)
            finally:
                connect.close(conn)


def candidates() -> tuple[GalleryV45, ...]:
    """Every storage contract with evidence. A tuple: the storage answer is a
    comparison between candidates."""
    return (GalleryV45(),)
