from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from compat.storage.contract import Observation

NOW: Final[float] = 1_700_000_000.0


class _Detected:
    def __init__(self, found: list[Any]) -> None:
        self._found = found

    def get(self, bgr: Any) -> list[Any]:
        del bgr
        return self._found


def replaying(found: list[Any]) -> Any:
    from vision import faces

    class Replaying(faces.InsightFaceBackend):
        def __init__(self) -> None:
            self._app = _Detected(found)
            self._min_det_score = faces.DEFAULT_MIN_DET_SCORE
            self._min_face_px = faces.DEFAULT_MIN_FACE_PX

    return Replaying()


def _library(conn: sqlite3.Connection, folder: Path, name: str, sha: str, size: tuple[int, int]) -> int:
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


def native_round_trip(face: Any, frame: npt.NDArray[np.uint8], sha: str) -> Any:
    import cv2
    from PIL import Image

    from db import connect, detect, faces_native

    height, width = frame.shape[:2]
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    with tempfile.TemporaryDirectory(prefix="compat-gallery-native-") as scratch:
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
            return faces_native.native_of(conn, written[0])
        finally:
            connect.close(conn)


class GalleryNative:
    name = "gallery_native"
    described = "derived_face_instance.native, via db/detect.py harvest and db/faces_native.native_of"

    def round_trip(self, face: Any, frame: npt.NDArray[np.uint8], sha: str) -> Observation:
        native = native_round_trip(face, frame, sha)
        values: dict[str, npt.NDArray[np.generic]] = {
            str(key): np.asarray(value) for key, value in native.record.items() if value is not None
        }
        return Observation(values)


def candidates() -> tuple[GalleryNative, ...]:
    return (GalleryNative(),)
