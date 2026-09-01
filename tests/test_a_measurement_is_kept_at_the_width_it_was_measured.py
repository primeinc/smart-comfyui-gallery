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

#: `models_dir` as `sg_web/home.py:49` means it: the directory the app is
#: pointed at, NOT the insightface root inside it. `vision/weights.py:112`
#: joins `INSIGHTFACE_SUBDIR` onto this.
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
#: pytest can see it: `catch_warnings` resets the filter the application
#: installs at import. Module and message, so other warnings stay fatal.
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
    """The captured record carries 2d106det's output in the producer's own
    dtype and shape, with no rounding anywhere between the pass and it.

    Asserted on the VALUE, never on the absence of a call. A coordinate that
    happens to be exact at five places proves nothing either way, so this
    requires at least one that is not -- and says so if the photograph cannot
    separate a rounded write from an unrounded one.
    """
    from vision import facestore

    found, _ = detections(path)
    records = [facestore.thaw(one.native).record for one in found if one.native is not None]
    dense = [one["landmark_2d_106"] for one in records if "landmark_2d_106" in one]
    assert dense, f"{path.name}: no captured record carried landmark_2d_106"

    held = dense[0]
    assert held.dtype == np.float32, f"{path.name}: the record narrowed {held.dtype} from the producer's float32"
    assert held.shape == (106, 2), f"{path.name}: landmark_2d_106 is {held.shape}, expected (106, 2)"

    values = held.astype(np.float64).ravel()
    moved = values[values != np.round(values, 5)]
    assert moved.size, (
        f"{path.name}: every stored coordinate is already exact at 5 decimal places, so this "
        f"photograph cannot tell a rounded write from an unrounded one"
    )


@pytest.mark.slow
@needs_the_real_thing
@pytest.mark.parametrize("path", CORPUS, ids=lambda one: one.parent.name)
def test_a_depth_landmark_keeps_every_digit_the_detector_gave_it(path):
    """1k3d68's x/y and depth reach the record unrounded, at float32.

    Both halves, because the removed defect was two separate roundings at two
    separate widths and a test covering only x/y passes with the depth one
    reinstated. Depth is asserted on its own: z is in the model's
    pixel-scaled units, so it is the one coordinate whose magnitude makes 2
    decimal places a real loss.
    """
    from vision import facestore

    found, _ = detections(path)
    records = [facestore.thaw(one.native).record for one in found if one.native is not None]
    dense = [one["landmark_3d_68"] for one in records if "landmark_3d_68" in one]
    assert dense, f"{path.name}: no captured record carried landmark_3d_68"

    held = dense[0]
    assert held.dtype == np.float32, f"{path.name}: the record narrowed {held.dtype} from the producer's float32"
    assert held.shape == (68, 3), f"{path.name}: landmark_3d_68 is {held.shape}, expected (68, 3)"
    points = held.astype(np.float64)

    flat = points[:, :2].ravel()
    moved = flat[flat != np.round(flat, 5)]
    assert moved.size, (
        f"{path.name}: every stored x/y is already exact at 5 decimal places, so this "
        f"photograph cannot tell a rounded write from an unrounded one"
    )

    depth = points[:, 2]
    deep = depth[depth != np.round(depth, 2)]
    assert deep.size, (
        f"{path.name}: every stored depth is already exact at 2 decimal places, so this "
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
    # Checked BEFORE the reshape: a float32 blob reshapes to (5,) and raises
    # `cannot reshape array of size 5 into shape (2)`, naming neither the width
    # nor the value.
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
def test_the_v46_step_leaves_an_already_widened_row_alone(db, tmp_path):
    """A float64 blob is already the v46 shape. Skip it, never reinterpret it.

    Reading one as float32 does not raise: 80 bytes become 20 values spanning
    -5.0e-05 to 1.828125, written back as 160. And the step must not RAISE
    either -- `migrate` rolls back and re-raises, leaving the file at v45,
    which `db/connect.py` then refuses with no down step. One bad row would
    brick the database.
    """
    already_wide = np.array(
        [[0.3125, 0.5], [0.61, 0.5], [0.5, 0.6875], [0.375, 0.8125], [0.5625, 0.8125]],
        dtype=np.float64,
    )
    face_id = _one_face(db, tmp_path, "already_wide", already_wide.tobytes())

    migrate.STEPS[45](db)

    (after,) = db.execute("SELECT landmarks FROM derived_face_instance WHERE id = ?", (face_id,)).fetchone()
    assert bytes(after) == already_wide.tobytes(), "an already-float64 row was rewritten"


@pytest.mark.slow
def test_the_v46_step_leaves_a_ragged_blob_alone(db, tmp_path):
    """A blob that is not whole float32 PAIRS is skipped, not converted.

    44 bytes is eleven floats: divisible by 4 and unreshapeable to (N, 2). A
    `% 4` guard admits it and writes back 88 bytes of malformed float64.
    """
    face_id = _one_face(db, tmp_path, "ragged", bytes(44))

    migrate.STEPS[45](db)

    (held,) = db.execute("SELECT landmarks FROM derived_face_instance WHERE id = ?", (face_id,)).fetchone()
    assert len(bytes(held)) == 44, f"a ragged blob was rewritten to {len(bytes(held))} bytes"

    # Both lengths are caught by `len(held) % 8`. 37 also proves the check is
    # load-bearing: `np.frombuffer` raises on a non-multiple-of-4, and a raise
    # rolls the migration back and leaves the database unopenable at v45.
    other = _one_face(db, tmp_path, "unaligned", bytes(37))

    migrate.STEPS[45](db)

    (after,) = db.execute("SELECT landmarks FROM derived_face_instance WHERE id = ?", (other,)).fetchone()
    assert len(bytes(after)) == 37, f"an unaligned blob was rewritten to {len(bytes(after))} bytes"


@pytest.mark.slow
def test_the_v46_step_refuses_an_infinite_coordinate(db, tmp_path):
    """NaN is admitted; an infinity is not.

    `vision/faces.py` `_clamp01` returns nan for nan, so a legitimate row can
    carry one. It cannot carry an infinity, and a check that filters on
    `np.isfinite` drops both before testing the range -- so an all-infinity
    blob and one mixing inf with real coordinates both pass.
    """
    both = np.array([[0.5, float("inf")], [0.25, 0.75]], dtype=np.float32)
    face_id = _one_face(db, tmp_path, "infinite", both.tobytes())

    migrate.STEPS[45](db)

    (after,) = db.execute("SELECT landmarks FROM derived_face_instance WHERE id = ?", (face_id,)).fetchone()
    assert bytes(after) == both.tobytes(), "a row carrying an infinity was widened"


@pytest.mark.slow
def test_the_v46_step_widens_a_row_carrying_a_nan(db, tmp_path):
    """A NaN coordinate is the producer's own output and must not block the row."""
    held = np.array([[0.5, float("nan")], [0.25, 0.75]], dtype=np.float32)
    face_id = _one_face(db, tmp_path, "with_nan", held.tobytes())

    migrate.STEPS[45](db)

    (after,) = db.execute("SELECT landmarks FROM derived_face_instance WHERE id = ?", (face_id,)).fetchone()
    assert len(bytes(after)) == held.size * 8, "a row carrying a NaN was left at float32"


@pytest.mark.slow
def test_a_small_float64_blob_is_widened_again(db, tmp_path):
    """The limit of reinterpretation, pinned as behaviour rather than prose.

    `user_version` is what says a blob is float32, so the step reads it that
    way first. A float64 blob whose magnitudes fall in [2**-16, 2**-8) has
    high words that read as float32 in [0, 1), so it passes `plausible` and is
    widened again, doubling the point count. Reaching this needs float64 at
    v45, which one build wrote before the version bump landed.

    Asserted so the gap is discoverable and so closing it fails here.
    """
    small = np.array([[2.0**-16, 2.0**-10], [2.0**-9, 2.0**-12]], dtype=np.float64)
    face_id = _one_face(db, tmp_path, "small_wide", small.tobytes())

    migrate.STEPS[45](db)

    (after,) = db.execute("SELECT landmarks FROM derived_face_instance WHERE id = ?", (face_id,)).fetchone()
    assert len(bytes(after)) == len(small.tobytes()) * 2, (
        "the step no longer widens a small float64 blob; the gap this pins is closed and "
        "this test should assert the row is left alone"
    )


@pytest.mark.slow
def test_the_v46_step_is_idempotent(db, tmp_path):
    """Running it twice leaves the same bytes.

    It cannot raise, so an operator who re-runs a partially applied upgrade
    must not get a second widening on top of the first.
    """
    original = np.array([[0.25, 0.5], [0.75, 0.125]], dtype=np.float32)
    face_id = _one_face(db, tmp_path, "twice", original.tobytes())

    migrate.STEPS[45](db)
    (once,) = db.execute("SELECT landmarks FROM derived_face_instance WHERE id = ?", (face_id,)).fetchone()
    migrate.STEPS[45](db)
    (twice,) = db.execute("SELECT landmarks FROM derived_face_instance WHERE id = ?", (face_id,)).fetchone()

    assert bytes(once) == bytes(twice), "a second run moved the value"
    assert np.array_equal(np.frombuffer(bytes(twice), dtype=np.float64).reshape(-1, 2), original.astype(np.float64))


@pytest.mark.slow
def test_the_v46_step_leaves_a_row_with_no_keypoints_alone(db, tmp_path):
    """A face with no landmarks is not given any."""
    face_id = _one_face(db, tmp_path, "bare", None)
    migrate.STEPS[45](db)
    (held,) = db.execute("SELECT landmarks FROM derived_face_instance WHERE id = ?", (face_id,)).fetchone()
    assert held is None


def test_the_version_this_build_writes_has_a_step_off_the_one_before_it():
    """Whatever version this build writes, the one before it must have a step.
    A bump with no step bricks every database sitting at the older version.

    Written against USER_VERSION rather than against a literal. A pinned number
    tests the number, not the rule: it goes red on every bump including the
    correct ones, which teaches a reader to edit the line rather than to read
    it -- and the bump itself is already covered by the drift check and the
    migration suite.
    """
    before = connect.USER_VERSION - 1
    assert before in migrate.STEPS, f"v{before} has no step off it; a v{before} database cannot be opened by this build"
    # The half that proves the membership test discriminates: there is no step
    # off the version this build WRITES, because nothing has moved past it yet.
    # Without this, `in STEPS` passing would say nothing.
    assert connect.USER_VERSION not in migrate.STEPS, (
        f"a step claims to move a database off v{connect.USER_VERSION}, the version this build writes"
    )
