"""What a detector produced is what the row holds.

A face pass is the expensive thing in this application: antelopev2 loads a
143 MB 1k3d68 session to derive head pose and two dense landmark sets, and
keeping all of it costs a few kilobytes per face. Anything dropped between
the producer and the row can only come back by reading the whole library off
disk again.

These hold the storage contract: the canonical thing persisted is `native` --
the producer's complete record, frozen by `vision/facestore.py` -- and the
promoted columns are promotions OUT of it rather than a filter in front of
it, with pose landing on the axis it names.
"""

import numpy as np
import pytest
from PIL import Image

from db import authored, derived, detect, faces_native, library, scan
from tests.staging import NOW, fresh_schema
from vision import facestore

#: The promoted per-face values, keyed the way `vision/faces.py` promotes
#: them. Pose is keyed because upstream returns [pitch, yaw, roll] as an
#: array and the columns are yaw-first (insightface model_zoo/landmark.py:111).
TRAITS = {
    "age": 27,
    "sex": "male",
    "pose": {"pitch": -3.5, "yaw": 12.25, "roll": 1.0},
}

#: A producer's record as the InsightFace backend captures it: the
#: producer's own dtypes and shapes, keys iterated rather than named.
RECORD = {
    "bbox": np.array([12.5, -3.0, 200.25, 240.0], dtype=np.float32),
    "kps": np.arange(10, dtype=np.float32).reshape(5, 2),
    "det_score": np.float32(0.99),
    "landmark_2d_106": np.linspace(0.1, 300.7, 212, dtype=np.float32).reshape(106, 2),
    "landmark_3d_68": np.linspace(-40.0, 250.3, 204, dtype=np.float32).reshape(68, 3),
    "pose": np.array([-3.5, 12.25, 1.0], dtype=np.float32),
    "embedding": np.linspace(-1.0, 1.0, 512, dtype=np.float32),
    "gender": np.int64(1),
    "age": np.int64(27),
}


def frozen(record) -> bytes:
    return facestore.freeze(
        dict(record),
        producer="insightface/antelopev2",
        producer_version="scrfd10g+glintr100-v1",
        container="insightface.app.common.Face",
    )


@pytest.fixture
def db():
    conn = fresh_schema()
    yield conn
    conn.close()


@pytest.fixture
def a_library(db, tmp_path):
    """A root, a folder, and one file to hang everything on."""
    root = tmp_path / "lib"
    root.mkdir()
    root_id = library.add_root(db, root, "library", NOW)
    folder_id = scan.ensure_folder(db, root_id, None, "lib")
    file_id = scan.mint(db, "file", "dusk")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256,"
        " first_seen_at, last_seen_at) VALUES(?, ?, 'dusk.png', 'image', 10, 0, 'aa', ?, ?)",
        (file_id, folder_id, NOW, NOW),
    )
    authored.add_user(db, "will", "hash", "ADMIN", NOW)
    return {"root": root_id, "path": root, "folder": folder_id, "file": file_id}


def _record(db, file_id, traits, native=None):
    """One face carrying `traits`, written the way `db/detect.py` writes it.

    `dict[str, object]`, because that is what a detection record is: an id,
    a score, packed bytes, a keyed pose and the producer's frozen output. The
    value type is not a column type -- the record is what crosses into
    `record_faces`, and `_insert_face` is where it fans out into columns.
    """
    record: dict[str, object] = {
        "region": derived.region(db, 0.3, 0.2, 0.2, 0.3),
        "det_score": 0.99,
        "embedding": np.ones(512, dtype=np.float32).tobytes(),
    }
    if native is not None:
        record["native"] = native
    if traits:
        if "age" in traits:
            record["age"] = int(traits["age"])
        if "sex" in traits:
            record["sex"] = str(traits["sex"])
        pose = traits.get("pose")
        if isinstance(pose, dict):
            record["pose"] = {axis: float(pose[axis]) for axis in ("yaw", "pitch", "roll") if axis in pose}
    (face_id,) = derived.record_faces(
        db, file_id, "insightface/antelopev2", "scrfd10g+glintr100-v1", "aa", NOW, [record]
    )
    return face_id


def test_every_value_the_detector_produced_reaches_the_row(db, a_library):
    """The dense sets are the point: they are what the 143 MB session is
    for, and the whole reason the record is kept whole. The row hands back
    every key in the producer's own dtype and shape, bit for bit."""
    face_id = _record(db, a_library["file"], TRAITS, native=frozen(RECORD))

    native = faces_native.native_of(db, face_id)

    assert native.producer == "insightface/antelopev2"
    assert list(native.record) == list(RECORD), "the row holds a different set of values than the detector produced"
    for key, want in RECORD.items():
        held = native.record[key]
        assert np.asarray(held).dtype == np.asarray(want).dtype, key
        assert np.asarray(held).shape == np.asarray(want).shape, key
        assert np.asarray(held).tobytes() == np.asarray(want).tobytes(), f"{key} did not survive bit-for-bit"


def test_a_value_nothing_was_written_for_survives(db, a_library):
    """The storage boundary is not an allowlist.

    An allowlist is what produced the original defect: named keys copied,
    the rest dropped. A backend emitting something nobody thought to name --
    with a dtype and shape no column mentions -- has to keep it anyway,
    because the alternative is reading the library again.
    """
    big_endian = np.array([0.4, 0.1, 0.9], dtype=">f8")
    unheard_of = dict(
        RECORD,
        expression_coefficients=big_endian,
        mask={"kind": "occlusion", "score": np.float16(0.02), "cells": np.arange(6, dtype=np.uint8).reshape(2, 3)},
    )
    face_id = _record(db, a_library["file"], TRAITS, native=frozen(unheard_of))

    held = faces_native.native_of(db, face_id).record

    coeffs = held["expression_coefficients"]
    assert coeffs.dtype.str == ">f8", "the unheard-of key lost its byte order"
    assert coeffs.tobytes() == big_endian.tobytes()
    mask = held["mask"]
    assert mask["kind"] == "occlusion"
    assert type(mask["score"]) is np.float16
    assert mask["score"].tobytes() == np.float16(0.02).tobytes()
    assert mask["cells"].shape == (2, 3)
    assert mask["cells"].dtype == np.uint8


def test_pose_lands_on_the_axis_it_names(db, a_library):
    """The trap this cost, stated as a test.

    InsightFace hands back [pitch, yaw, roll]; the columns are yaw, pitch,
    roll. A positional copy writes pitch into `pose_yaw` and no CHECK can
    see it -- three REAL columns holding plausible degrees either way. The
    values here are distinct and differently signed so a swap cannot pass.
    """
    face_id = _record(db, a_library["file"], TRAITS)

    yaw, pitch, roll = db.execute(
        "SELECT pose_yaw, pose_pitch, pose_roll FROM derived_face_instance WHERE id = ?", (face_id,)
    ).fetchone()

    assert yaw == pytest.approx(12.25), "pose_yaw is not the yaw the detector reported"
    assert pitch == pytest.approx(-3.5), "pose_pitch is not the pitch the detector reported"
    assert roll == pytest.approx(1.0)


def test_a_pose_with_no_axis_names_is_refused(db, a_library):
    """The negative control for the test above.

    A triple is the shape that carries the bug: it is ambiguous between the
    detector's [pitch, yaw, roll] and these columns' yaw-first order, and
    once written the two are indistinguishable -- three REAL columns holding
    plausible degrees. There is no value to range-check and no constraint
    that can fire, so the only defence is refusing the anonymous shape at
    the door.
    """
    record = {
        "region": derived.region(db, 0.3, 0.2, 0.2, 0.3),
        "det_score": 0.99,
        "pose": (12.25, -3.5, 1.0),
    }

    with pytest.raises(TypeError, match="no axis names"):
        derived.record_faces(
            db, a_library["file"], "insightface/antelopev2", "scrfd10g+glintr100-v1", "aa", NOW, [record]
        )


def test_the_promoted_columns_agree_with_the_record_they_came_from(db, a_library):
    """`age`, `sex` and the three pose columns exist because a facet filters
    on them. They are a second spelling of values the record already holds,
    so a row where the two disagree is a row whose facet lies about it."""
    face_id = _record(db, a_library["file"], TRAITS, native=frozen(RECORD))

    age, sex, yaw, pitch, roll = db.execute(
        "SELECT age, sex, pose_yaw, pose_pitch, pose_roll FROM derived_face_instance WHERE id = ?",
        (face_id,),
    ).fetchone()
    held = faces_native.native_of(db, face_id).record

    assert age == int(held["age"])
    assert sex == ("male" if int(held["gender"]) == 1 else "female")
    # The record's pose is upstream's [pitch, yaw, roll] array; the columns
    # are yaw-first.
    assert pitch == pytest.approx(float(held["pose"][0]))
    assert yaw == pytest.approx(float(held["pose"][1]))
    assert roll == pytest.approx(float(held["pose"][2]))


def test_a_backend_that_says_nothing_writes_no_record(db, a_library):
    """The stub backend hands over no native record at all -- NULL here means
    the backend had nothing to say, and replaying such a row refuses by name
    instead of serving an empty record as if it were the producer's."""
    face_id = _record(db, a_library["file"], {})

    (stored,) = db.execute("SELECT native FROM derived_face_instance WHERE id = ?", (face_id,)).fetchone()
    assert stored is None

    with pytest.raises(faces_native.NativeMissing, match="no native record"):
        faces_native.native_of(db, face_id)


def test_harvest_persists_the_record_and_promotes_out_of_it(db, a_library):
    """The producer's own contract, not the fixture's imitation of it.

    `_record` above mirrors what `db/detect.py` does; this asserts the two
    have not drifted, by driving the real thing with a stub backend.
    """
    envelope = frozen(RECORD)

    class OneFace:
        model_id = "insightface/antelopev2"
        model_version = "scrfd10g+glintr100-v1"

        def detect(self, image):
            from vision.faces import FaceDetection

            return [
                FaceDetection(
                    bbox=(0.3, 0.2, 0.2, 0.3),
                    landmarks=[(0.1, 0.1)] * 5,
                    det_score=0.99,
                    embedding=np.ones(512, dtype=np.float32),
                    attributes=dict(TRAITS),
                    native=envelope,
                )
            ]

    written = detect.harvest(db, OneFace(), a_library["file"], None, NOW, image=Image.new("RGB", (64, 64)))

    (face_id,) = written
    age, yaw, pitch, roll = db.execute(
        "SELECT age, pose_yaw, pose_pitch, pose_roll FROM derived_face_instance WHERE id = ?",
        (face_id,),
    ).fetchone()
    held = faces_native.native_of(db, face_id)

    assert set(held.record) == set(RECORD), "harvest filtered the detector's output"
    assert age == 27
    assert yaw == pytest.approx(12.25)
    assert pitch == pytest.approx(-3.5)
    assert roll == pytest.approx(1.0)
