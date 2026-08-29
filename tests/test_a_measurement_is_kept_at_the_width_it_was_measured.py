"""What the detector measured is what the row gives back.

Three lossy transforms sat between antelopev2 and `derived_face_instance`,
each deliberate, none of them paid for:

    landmarks           NORMALIZED coordinates in a float32 blob. Reading one
                        back multiplies by the frame size, and a float32 value
                        in [0, 1] carries a half-ulp of 2**-24 -- on a 6528 px
                        photograph, 3.9e-4 source pixels discarded before
                        anything asked for the value.
    landmark_2d_106     `round(x, 5)` on the normalized coordinate: half of one
                        unit in the 5th decimal is 5e-6 of the frame, 0.033 px
                        at 6528.
    landmark_3d_68      the same, plus `round(z, 2)` on the depth.
    pose                `round(deg, 2)` on pitch, yaw and roll.

`compat/consumers/gallery_storage.py` measured every one against the producer
at rtol=0, atol=0 and found 19 keys that did not survive the round trip. The
fix is v46: `vision/faces.py` stops rounding, `db/detect.py` writes the blob at
float64, and `db/migrate.py` widens what is already stored.

REAL WEIGHTS, REAL PHOTOGRAPHS. A stub backend cannot hold this contract. The
claim is about what an actual detector produces and what an actual row returns,
and a fake on both sides of that comparison only shows the fake agrees with
itself. These run antelopev2 over the sample-dataset photographs the production
face benchmark uses, and skip only when the weights or the corpus are genuinely
not on this machine.
"""

import hashlib
import os
import pathlib

import numpy as np
import pytest

from db import connect, derived, library, migrate, scan
from tests.staging import NOW, fresh_schema

#: Where the sample datasets live. The production face benchmark names the
#: same root, and `compat/corpus/index.py` reads this same environment
#: override -- one location, not three.
DATASETS = pathlib.Path(os.environ.get("COMPAT_DATASETS", "C:/ComfyUI/output/sample-datasets"))
KYC = DATASETS / "caucasian-people-kyc-photo-dataset" / "files"

#: `models_dir` as `sg_web/home.py:49` means it -- the directory the app is
#: pointed at, NOT the insightface root inside it. `vision/weights.py:112`
#: joins `INSIGHTFACE_SUBDIR` onto this, and insightface's own FaceAnalysis
#: then looks for `<that>/models/<pack>`; naming the inner directory here
#: makes the backend look one level too deep and report the pack absent when
#: it is present.
MODELS = pathlib.Path(os.environ.get("SG_MODELS_DIR", "C:/ComfyUI/output/.AImodels"))

#: What must exist for the real detector to run, spelled the way the loader
#: spells it rather than guessed at.
PACK = MODELS / "insightface" / "models" / "antelopev2"


def photographs(limit: int = 2) -> list[pathlib.Path]:
    """Real faces, chosen BY CONTENT so the selection does not move.

    Sorted by digest rather than by directory order: a listing changes when
    the filesystem feels like it, and a test whose input moves is a test whose
    failure cannot be reproduced.
    """
    if not KYC.is_dir():
        return []
    found: list[tuple[str, pathlib.Path]] = []
    for folder in sorted(KYC.iterdir(), key=lambda one: one.name):
        if not folder.is_dir():
            continue
        for image in sorted(folder.iterdir(), key=lambda one: one.name):
            if image.is_file():
                found.append((hashlib.sha256(image.read_bytes()).hexdigest(), image))
                break
    return [path for _, path in sorted(found)[:limit]]


CORPUS = photographs()

#: The SAME narrow filter `vision/faces.py:55-57` declares, restated where
#: pytest can see it. `pytest.ini` sets `filterwarnings = error`, and pytest
#: wraps each test in `catch_warnings`, which resets the module-level filter
#: the application installs at import -- so a warning the app has already
#: reasoned about and silenced at its source comes back as an error here.
#:
#: Module and message, never a bare `ignore`: insightface 1.0.1 aligns through
#: skimage's pre-2.2 `estimate()` API and skimage 0.26 deprecates it with a
#: FutureWarning on EVERY alignment. Every other warning stays fatal.
pytestmark = pytest.mark.filterwarnings(
    "ignore:.*`estimate` is deprecated.*:FutureWarning:insightface.utils.face_align"
)

#: Skips only when the real thing is ABSENT, never when it is merely slow.
#: `refs/pytest-dev/pytest/src/_pytest/skipping.py:171` iterates these marks
#: per item, and the condition is evaluated once at collection.
needs_the_real_thing = pytest.mark.skipif(
    not CORPUS or not PACK.is_dir(),
    reason=f"needs the antelopev2 pack at {PACK} and the KYC corpus under {KYC}",
)


@pytest.fixture
def db():
    conn = fresh_schema()
    yield conn
    conn.close()


@pytest.fixture
def a_library(db, tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    root_id = library.add_root(db, root, "library", NOW)
    folder_id = scan.ensure_folder(db, root_id, None, "lib")
    return {"root": root_id, "folder": folder_id, "path": root}


def detections(path: pathlib.Path):
    """One real photograph through the real detector, as the app runs it.

    `vision.decode.open_still` rather than a bare open: it is the surface the
    application uses, it registers the plugins, and it returns the picture as
    stored.
    """
    from vision import decode, faces

    with decode.open_still(path) as opened:
        opened.load()
        size = opened.size
        backend = faces.InsightFaceBackend(str(MODELS))
        found = backend.detect(opened if opened.mode == "RGB" else opened.convert("RGB"))
    assert found, f"antelopev2 found no face in {path.name}; the corpus image is the input, not the claim"
    return found, size


# ---------------------------------------------------------------------------
# The write path no longer narrows what the detector produced.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@needs_the_real_thing
@pytest.mark.parametrize("path", CORPUS, ids=lambda one: one.parent.name)
def test_a_dense_landmark_keeps_every_digit_the_detector_gave_it(path):
    """`round(x, 5)` is gone from the dense landmark sets.

    Asserted on the VALUE, never on the absence of a call. A coordinate that
    happens to be exact at five places proves nothing either way, so this
    requires at least one that is not -- and says so if the photograph cannot
    separate a rounded write from an unrounded one.
    """
    found, _ = detections(path)
    dense = [
        one.attributes["landmark_2d_106"] for one in found if one.attributes and "landmark_2d_106" in one.attributes
    ]
    assert dense, f"{path.name}: no detection carried landmark_2d_106"

    values = np.asarray(dense[0], dtype=np.float64).ravel()
    moved = values[values != np.round(values, 5)]
    assert moved.size, (
        f"{path.name}: every stored coordinate is already exact at 5 decimal places, so this "
        f"photograph cannot tell a rounded write from an unrounded one"
    )


@pytest.mark.slow
@needs_the_real_thing
@pytest.mark.parametrize("path", CORPUS, ids=lambda one: one.parent.name)
def test_a_pose_keeps_every_digit_the_detector_gave_it(path):
    """`round(deg, 2)` is gone from pitch, yaw and roll."""
    found, _ = detections(path)
    poses = [one.attributes["pose"] for one in found if one.attributes and "pose" in one.attributes]
    assert poses, f"{path.name}: no detection carried a pose"

    angles = np.asarray([poses[0][axis] for axis in ("pitch", "yaw", "roll")], dtype=np.float64)
    moved = angles[angles != np.round(angles, 2)]
    assert moved.size, f"{path.name}: every pose angle is exact at 2 places; this fixture cannot separate them"


@pytest.mark.slow
@needs_the_real_thing
def test_a_stored_keypoint_returns_the_pixel_the_detector_named(db, a_library):
    """Through `db.detect.harvest`, which is the write path this claim is about.

    An earlier version of this test built the record by hand and wrote the
    blob at float64 itself. It passed with the float32 defect reinstated,
    because it never executed `db/detect.py` at all -- it asserted that the
    bytes it had just written were the bytes it had just written. `harvest`
    is the function that decides the width, so `harvest` is what runs here.
    """
    from db import detect
    from vision import faces

    path = CORPUS[0]
    stored = a_library["path"] / path.name
    stored.write_bytes(path.read_bytes())

    file_id = scan.mint(db, "file", "kyc")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256,"
        " first_seen_at, last_seen_at) VALUES(?, ?, ?, 'image', ?, 0, 'aa', ?, ?)",
        (file_id, a_library["folder"], path.name, stored.stat().st_size, NOW, NOW),
    )

    backend = faces.InsightFaceBackend(str(MODELS))
    written = detect.harvest(db, backend, file_id, stored, NOW)
    assert written, f"harvest recorded no face for {path.name}"

    (blob,) = db.execute("SELECT landmarks FROM derived_face_instance WHERE id = ?", (written[0],)).fetchone()
    held = bytes(blob)

    # Five (x, y) pairs. At float32 that is 40 bytes and the value has been
    # narrowed; at float64 it is 80 and it has not.
    assert len(held) == 5 * 2 * 8, (
        f"the write path stored {len(held)} bytes for five keypoints; float64 is {5 * 2 * 8}, "
        f"float32 is {5 * 2 * 4} and loses source pixels on every read"
    )

    read_back = np.frombuffer(held, dtype=np.float64).reshape(-1, 2)
    assert ((read_back >= 0.0) & (read_back <= 1.0)).all(), "the stored coordinates are not normalized"


@pytest.mark.slow
@needs_the_real_thing
def test_the_stored_keypoint_is_the_one_the_detector_produced(db, a_library):
    """`harvest` writes the detector's own value, not a narrowed copy of it.

    The width alone is not the claim -- a float64 blob holding
    `float32(x)` is eight bytes of a value that was already thrown away.
    This runs the detector separately, over the same file, and requires the
    stored coordinate to be bit-identical to what it produced.
    """
    from db import detect
    from vision import faces

    path = CORPUS[0]
    stored = a_library["path"] / path.name
    stored.write_bytes(path.read_bytes())

    file_id = scan.mint(db, "file", "kyc-exact")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256,"
        " first_seen_at, last_seen_at) VALUES(?, ?, ?, 'image', ?, 0, 'bb', ?, ?)",
        (file_id, a_library["folder"], path.name, stored.stat().st_size, NOW, NOW),
    )

    backend = faces.InsightFaceBackend(str(MODELS))
    written = detect.harvest(db, backend, file_id, stored, NOW)
    assert written, f"harvest recorded no face for {path.name}"

    produced, (width, height) = detections(path)
    (blob,) = db.execute("SELECT landmarks FROM derived_face_instance WHERE id = ?", (written[0],)).fetchone()
    held = bytes(blob)
    # Checked BEFORE the reshape. A float32 blob is 40 bytes, which reshapes
    # to (5,) and raises `cannot reshape array of size 5 into shape (2)` --
    # a crash three lines from the fact, naming neither the width nor the
    # value. The test must say what is wrong with the row, not fall over it.
    assert len(held) % (2 * 8) == 0, (
        f"the stored blob is {len(held)} bytes, which is not whole float64 pairs; "
        f"a float32 write path stores half this and loses source pixels on every read"
    )
    read_back = np.frombuffer(held, dtype=np.float64).reshape(-1, 2)

    # `harvest` keeps the faces it recorded in its own order; match on the
    # stored coordinates rather than assuming which detection came first.
    scale = np.array([width, height], dtype=np.float64)
    wanted = [np.asarray(one.landmarks, dtype=np.float64) for one in produced]
    closest = min(wanted, key=lambda one: float(np.max(np.abs(one - read_back))))

    lost = float(np.max(np.abs(closest * scale - read_back * scale)))
    assert lost == 0.0, f"{lost} source pixels lost between the detector and the row"


# ---------------------------------------------------------------------------
# The migration converts what is already stored.
# ---------------------------------------------------------------------------


def _one_face(db, tmp_path, name: str, landmarks: bytes | None):
    root_id = library.add_root(db, tmp_path / name, "library", NOW)
    folder_id = scan.ensure_folder(db, root_id, None, "lib")
    file_id = scan.mint(db, "file", name)
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256,"
        " first_seen_at, last_seen_at) VALUES(?, ?, ?, 'image', 10, 0, 'aa', ?, ?)",
        (file_id, folder_id, f"{name}.png", NOW, NOW),
    )
    record: dict[str, object] = {
        "region": derived.region(db, 0.3, 0.2, 0.2, 0.3),
        "det_score": 0.99,
        "embedding": np.ones(512, dtype=np.float32).tobytes(),
    }
    if landmarks is not None:
        record["landmarks"] = landmarks
    (face_id,) = derived.record_faces(
        db, file_id, "insightface/antelopev2", "scrfd10g+glintr100-v1", "aa", NOW, [record]
    )
    return face_id


@pytest.mark.slow
def test_the_v46_step_widens_a_stored_keypoint_without_moving_it(db, tmp_path):
    """A v45 float32 blob becomes float64 and lands on the same value.

    This step can silently destroy every stored keypoint: reading 40 bytes of
    float32 as float64 yields five plausible doubles that are garbage. The
    local database it was first applied to held ZERO landmark rows, so running
    it there demonstrated nothing whatsoever.
    """
    original = np.array(
        [[0.3125, 0.5], [0.61328125, 0.5], [0.5, 0.6875], [0.375, 0.8125], [0.5625, 0.8125]],
        dtype=np.float32,
    )
    face_id = _one_face(db, tmp_path, "widened", original.tobytes())

    (before,) = db.execute("SELECT landmarks FROM derived_face_instance WHERE id = ?", (face_id,)).fetchone()
    assert len(bytes(before)) == original.size * 4, "the fixture is not a float32 blob"

    migrate.STEPS[45](db)

    (after,) = db.execute("SELECT landmarks FROM derived_face_instance WHERE id = ?", (face_id,)).fetchone()
    assert len(bytes(after)) == original.size * 8, "the blob was not widened"
    widened = np.frombuffer(bytes(after), dtype=np.float64).reshape(-1, 2)
    assert np.array_equal(widened, original.astype(np.float64)), "widening moved the value"


@pytest.mark.slow
def test_the_v46_step_leaves_a_row_with_no_keypoints_alone(db, tmp_path):
    """A face with no landmarks is not given any."""
    face_id = _one_face(db, tmp_path, "bare", None)
    migrate.STEPS[45](db)
    (held,) = db.execute("SELECT landmarks FROM derived_face_instance WHERE id = ?", (face_id,)).fetchone()
    assert held is None


def test_the_version_this_build_writes_has_a_step_off_the_one_before_it():
    """v46 is reachable from v45. A bump with no step bricks every database."""
    assert connect.USER_VERSION == 46
    assert 45 in migrate.STEPS, "v45 has no step off it; a v45 database cannot be opened by this build"
