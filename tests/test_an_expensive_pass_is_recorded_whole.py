"""What a detector produced is what the row holds.

A face pass is the expensive thing in this application: antelopev2 loads a
143 MB 1k3d68 session to derive head pose and two dense landmark sets, and
keeping all of it costs a few kilobytes per face. Anything `db/detect.py`
drops on the line that reads the attribute dict can only come back by
reading the whole library off disk again.

These hold the storage contract: the record is complete
and unfiltered, the promoted columns are promotions OUT of it rather than a
filter in front of it, and pose lands on the axis it names.
"""

import json

import numpy as np
import pytest
from PIL import Image

from db import authored, derived, detect, library, scan
from tests.staging import NOW, fresh_schema

#: One InsightFace detection, as `vision/faces.py` builds it. Pose is keyed here
#: because upstream returns [pitch, yaw, roll] as an array and these columns are
#: yaw-first (deepinsight/insightface model_zoo/landmark.py:111).
TRAITS = {
    "age": 27,
    "sex": "male",
    "pose": {"pitch": -3.5, "yaw": 12.25, "roll": 1.0},
    "landmark_2d_106": [[0.10, 0.20]] * 106,
    "landmark_3d_68": [[0.10, 0.20, 4.5]] * 68,
}


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


def _record(db, file_id, traits):
    """One face carrying `traits`, written the way `db/detect.py` writes it.

    `dict[str, object]`, because that is what a detection record is: an id,
    a score, packed bytes, a keyed pose and the producer's whole output. The
    value type is not a column type -- the record is what crosses into
    `record_faces`, and `_insert_face` is where it fans out into columns.
    """
    record: dict[str, object] = {
        "region": derived.region(db, 0.3, 0.2, 0.2, 0.3),
        "det_score": 0.99,
        "embedding": np.ones(512, dtype=np.float32).tobytes(),
    }
    if traits:
        record["attributes"] = traits
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
    """The dense sets are the point: they are what the 143 MB session is for,
    and they are what had nowhere to land."""
    face_id = _record(db, a_library["file"], TRAITS)

    (stored,) = db.execute("SELECT attributes FROM derived_face_instance WHERE id = ?", (face_id,)).fetchone()
    held = json.loads(stored)

    assert set(held) == set(TRAITS), "the row holds a different set of values than the detector produced"
    assert len(held["landmark_2d_106"]) == 106
    assert len(held["landmark_3d_68"]) == 68
    assert held["landmark_3d_68"][0] == [0.10, 0.20, 4.5], "the z component is not carried"


def test_a_value_nothing_was_written_for_survives(db, a_library):
    """The storage boundary is not an allowlist.

    An allowlist is what produced the defect: two named keys copied, three
    values dropped. A second allowlist would only move the next re-detect
    further out -- a backend emitting something nobody thought to name has
    to keep it anyway, because the alternative is reading the library again.
    """
    unheard_of = dict(TRAITS, expression_coefficients=[0.4, 0.1, 0.9], mask_score=0.02)
    face_id = _record(db, a_library["file"], unheard_of)

    (stored,) = db.execute("SELECT attributes FROM derived_face_instance WHERE id = ?", (face_id,)).fetchone()
    held = json.loads(stored)

    assert held["expression_coefficients"] == [0.4, 0.1, 0.9]
    assert held["mask_score"] == 0.02


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
    face_id = _record(db, a_library["file"], TRAITS)

    age, sex, yaw, pitch, roll, stored = db.execute(
        "SELECT age, sex, pose_yaw, pose_pitch, pose_roll, attributes FROM derived_face_instance WHERE id = ?",
        (face_id,),
    ).fetchone()
    held = json.loads(stored)

    assert age == held["age"]
    assert sex == held["sex"]
    assert yaw == pytest.approx(held["pose"]["yaw"])
    assert pitch == pytest.approx(held["pose"]["pitch"])
    assert roll == pytest.approx(held["pose"]["roll"])


def test_a_backend_that_says_nothing_writes_no_record(db, a_library):
    """The OpenCV backends attach no attributes at all -- `vision/faces.py`
    assigns them only inside the InsightFace block -- so NULL here means the
    detector had nothing to say, not that a value was dropped. Writing "{}"
    instead would make those two states one."""
    face_id = _record(db, a_library["file"], {})

    (stored,) = db.execute("SELECT attributes FROM derived_face_instance WHERE id = ?", (face_id,)).fetchone()

    assert stored is None


def test_harvest_promotes_out_of_the_record_it_keeps(db, a_library):
    """The producer's own contract, not the fixture's imitation of it.

    `_record` above mirrors what `db/detect.py` does; this asserts the two
    have not drifted, by driving the real thing with a stub backend.
    """

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
                )
            ]

    written = detect.harvest(db, OneFace(), a_library["file"], None, NOW, image=Image.new("RGB", (64, 64)))

    (face_id,) = written
    age, yaw, pitch, roll, stored = db.execute(
        "SELECT age, pose_yaw, pose_pitch, pose_roll, attributes FROM derived_face_instance WHERE id = ?",
        (face_id,),
    ).fetchone()
    held = json.loads(stored)

    assert set(held) == set(TRAITS), "harvest filtered the detector's output"
    assert age == 27
    assert yaw == pytest.approx(12.25)
    assert pitch == pytest.approx(-3.5)
    assert roll == pytest.approx(1.0)
