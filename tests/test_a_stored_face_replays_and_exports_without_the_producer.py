"""A stored face serves its consumers with the producer gone.

The claim under test is the whole lane: a value enters at the PRODUCER
boundary -- an insightface `Face`-shaped record handed to the real
`InsightFaceBackend.detect` -- crosses `db/detect.py` into the row, and comes
back through `db/faces_native.py` bit-identical, with the producer unable to
run, the original result object garbage, and no source pixels on disk.

Adversarial on purpose: the record carries keys no line of persistence code
names, in shapes JSON cannot hold -- a big-endian array, a float16 scalar, a
nested mapping, raw bytes, a tuple -- because the boundary is generic
iteration or it is another allowlist. And a value the envelope genuinely
cannot carry must fail AT CAPTURE, loudly, not vanish.

The export half writes ReActor's face model from the stored replay. Byte
equivalence against the pinned upstream writer is compat's job
(`compat/consumers/reactor_face_model.py`, which runs upstream's own code);
here the assertions are the ones the main suite can hold without the pinned
clone: the construction, the key order the container records, the dtypes,
and the refusal when a key upstream subscripts is absent.
"""

import json
import sys

import numpy as np
import pytest
from PIL import Image

from db import authored, detect, faces_native, library, scan
from tests.staging import NOW, fresh_schema
from vision import faces, facestore


class FakeFace(dict):
    """insightface's `Face` shape: a dict subclass whose attribute reads
    fall back to its keys (None when absent) and whose `sex` derives from
    `gender` (deepinsight/insightface app/common.py)."""

    def __getattr__(self, name):
        if name == "sex":
            gender = self.get("gender")
            return None if gender is None else ("M" if int(gender) == 1 else "F")
        return self.get(name)


def replaying(found):
    """`InsightFaceBackend` with `_app` pinned, `detect` inherited unchanged.

    `detect` reads only `_app`, `_min_det_score` and `_min_face_px`;
    `__init__` does not call up, because the parent loads a 143 MB pack whose
    output is already in hand. The double supplies `get` and nothing else, so
    `first_hit_descending` passes straight through.
    """

    class Detected:
        def get(self, bgr):
            return found

    class Replaying(faces.InsightFaceBackend):
        def __init__(self):
            self._app = Detected()
            self._min_det_score = faces.DEFAULT_MIN_DET_SCORE
            self._min_face_px = faces.DEFAULT_MIN_FACE_PX

    return Replaying()


def producer_record():
    """Everything the antelopev2 pass emits, in its dtypes, plus values no
    persistence code has ever heard of. Coordinates are detect-input pixels,
    the way upstream hands them over."""
    return FakeFace(
        bbox=np.array([100.25, 120.5, 420.75, 460.0], dtype=np.float32),
        kps=np.linspace(110.0, 450.0, 10, dtype=np.float32).reshape(5, 2),
        det_score=np.float32(0.9),
        landmark_2d_106=np.linspace(101.0, 459.0, 212, dtype=np.float32).reshape(106, 2),
        landmark_3d_68=np.linspace(-40.0, 250.0, 204, dtype=np.float32).reshape(68, 3),
        pose=np.array([-3.5, 12.25, 1.0], dtype=np.float32),
        embedding=np.linspace(-1.0, 1.0, 512, dtype=np.float32),
        gender=np.int64(1),
        age=np.int64(27),
        # The adversarial half: unnamed anywhere, un-JSON-able as they stand.
        attention_map=np.arange(12, dtype=">f4").reshape(3, 4),
        occlusion={"kind": "left", "score": np.float16(0.02), "cells": np.arange(6, dtype=np.uint8).reshape(2, 3)},
        codec_state=b"\x00\x01\xfe\xff",
        tags=("synthetic", 3, 2.5),
        quality=np.float64(0.12345678901234567),
        spectrum=np.array([1.5 + 2.25j, -3.0 - 0.5j], dtype=np.complex128),
        phase=1.5 - 2.25j,
    )


@pytest.fixture
def db():
    conn = fresh_schema()
    yield conn
    conn.close()


@pytest.fixture
def a_file(db, tmp_path):
    """A root, a folder, and one file row. No pixels ever touch the disk:
    `harvest` is handed the frame in memory, so a later replay has no source
    image to lean on even by accident."""
    root = tmp_path / "lib"
    root.mkdir()
    root_id = library.add_root(db, root, "library", NOW)
    folder_id = scan.ensure_folder(db, root_id, None, "lib")
    file_id = scan.mint(db, "file", "cold")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256, width, height,"
        " first_seen_at, last_seen_at) VALUES(?, ?, 'cold.png', 'image', 10, 0, 'aa', 640, 480, ?, ?)",
        (file_id, folder_id, NOW, NOW),
    )
    authored.add_user(db, "will", "hash", "ADMIN", NOW)
    return file_id


def harvested(db, a_file, face) -> int:
    written = detect.harvest(db, replaying([face]), a_file, None, NOW, image=Image.new("RGB", (640, 480)))
    assert written, "harvest recorded no face; the write path stored nothing"
    return written[0]


def test_a_key_no_persistence_code_names_survives_the_whole_lane(db, a_file):
    """Producer boundary -> capture -> row -> reload -> reconstructed record.

    Every value compares bit-for-bit in its original dtype and shape,
    including the ones that exist nowhere but this test -- which is the
    proof the boundary iterates rather than selects.
    """
    face = producer_record()
    face_id = harvested(db, a_file, face)

    native = faces_native.native_of(db, face_id)

    assert native.producer == faces.InsightFaceBackend.model_id
    assert native.producer_version == faces.InsightFaceBackend.model_version
    assert native.container.endswith("FakeFace"), "the envelope must record the producer's own container"
    assert list(native.record) == [str(key) for key in face], "keys or their order changed"

    held = native.record
    for key in ("bbox", "kps", "landmark_2d_106", "landmark_3d_68", "pose", "embedding", "attention_map", "spectrum"):
        assert held[key].dtype == face[key].dtype, key
        assert held[key].shape == face[key].shape, key
        assert held[key].tobytes() == face[key].tobytes(), f"{key} did not survive bit-for-bit"
    assert held["attention_map"].dtype.str == ">f4", "byte order was normalised away"
    for key in ("det_score", "gender", "age", "quality"):
        assert type(held[key]) is type(face[key]), key
        assert held[key].tobytes() == face[key].tobytes(), key
    assert held["codec_state"] == face["codec_state"]
    assert held["phase"] == face["phase"]
    assert type(held["phase"]) is complex
    assert held["tags"] == face["tags"]
    assert type(held["tags"]) is tuple
    inner = held["occlusion"]
    assert inner["kind"] == "left"
    assert type(inner["score"]) is np.float16
    assert inner["score"].tobytes() == np.float16(0.02).tobytes()
    assert inner["cells"].dtype == np.uint8
    assert inner["cells"].shape == (2, 3)


def test_replay_and_export_run_with_the_producer_gone(db, a_file, monkeypatch):
    """Cold replay: the model cannot load, the live result is garbage, and
    there are no pixels to re-read.

    `sys.modules[name] = None` makes any `import name` raise ImportError, so
    a replay path that quietly reaches for the producer fails here rather
    than passing on a machine that happens to have the runtime.
    """
    face = producer_record()
    face_id = harvested(db, a_file, face)
    warm = faces_native.reactor_face_model_bytes(faces_native.native_of(db, face_id))

    face.clear()  # the live producer object is gone, not merely out of scope
    monkeypatch.setitem(sys.modules, "insightface", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    native = faces_native.native_of(db, face_id)
    cold = faces_native.reactor_face_model_bytes(native)

    assert cold == warm, "the export changed when the producer disappeared, so something was leaning on it"
    assert native.record["embedding"].tobytes() == np.linspace(-1.0, 1.0, 512, dtype=np.float32).tobytes()


def test_the_exported_face_model_is_upstreams_construction(db, a_file):
    """The container holds exactly ReActor's nine tensors at the dtypes the
    serializer derives from the record. Byte identity against upstream's own
    writer is proven in compat, where the pinned clone runs; this holds the
    construction in the suite that runs on every push."""
    face = producer_record()
    face_id = harvested(db, a_file, face)
    native = faces_native.native_of(db, face_id)

    blob = faces_native.reactor_face_model_bytes(native)

    header_len = int.from_bytes(blob[:8], "little")
    header = json.loads(blob[8 : 8 + header_len].decode("utf-8"))
    tensor_names = [key for key in header if key != "__metadata__"]
    # The serializer canonicalises layout itself (int64 first, then
    # alphabetical), so the SET is upstream's contract and the order is the
    # container's own -- identical for upstream's writer, same serializer.
    assert sorted(tensor_names) == sorted(faces_native.REACTOR_KEYS), "not exactly ReActor's nine tensors"
    assert header["embedding"]["dtype"] == "F32"
    assert header["embedding"]["shape"] == [512]
    assert header["gender"]["dtype"] == "I64"
    assert header["gender"]["shape"] == []
    assert header["landmark_3d_68"]["shape"] == [68, 3]

    from safetensors.numpy import load

    loaded = load(blob)
    for key in faces_native.REACTOR_KEYS:
        assert loaded[key].tobytes() == np.asarray(native.record[key]).tobytes(), key


def test_an_export_missing_an_upstream_key_refuses_by_name(db, a_file):
    """Upstream's writer swallows the KeyError into a print and leaves no
    file; ours must say which key a record cannot supply."""
    face = producer_record()
    del face["landmark_3d_68"]
    face_id = harvested(db, a_file, face)

    with pytest.raises(KeyError, match="landmark_3d_68"):
        faces_native.reactor_face_model_bytes(faces_native.native_of(db, face_id))


def test_a_value_the_envelope_cannot_carry_fails_at_capture(db, a_file):
    """Loud, at the pass that can still re-run -- never a silent omission
    discovered by a replay years later. The path and runtime type are in the
    message because the fix starts with knowing which producer value it was."""
    face = producer_record()
    face["novel"] = {"weights": {1, 2, 3}}

    with pytest.raises(facestore.Unpreservable, match=r"novel.*set"):
        replaying([face]).detect(Image.new("RGB", (640, 480)))

    count = db.execute("SELECT count(*) FROM derived_face_instance WHERE file_id = ?", (a_file,)).fetchone()[0]
    assert count == 0, "a record that failed capture must not half-persist"


def test_precision_the_projection_discards_survives_in_the_record(db, a_file):
    """The promoted columns clamp coordinates to the frame; the record must
    not. A bbox corner past the edge and ulp-adjacent embedding values are
    exactly what a narrowed or clamped copy destroys."""
    face = producer_record()
    face["bbox"] = np.array([-7.25, -3.5, 655.0, 490.0], dtype=np.float32)
    tight = np.float32(0.1)
    apart = np.nextafter(tight, np.float32(1.0), dtype=np.float32)
    face["embedding"] = np.array([tight, apart] * 256, dtype=np.float32)
    face_id = harvested(db, a_file, face)

    held = faces_native.native_of(db, face_id).record

    assert held["bbox"].tobytes() == face["bbox"].tobytes(), "an out-of-frame corner was clamped or moved"
    assert held["embedding"][0] != held["embedding"][1], "ulp-adjacent values collapsed; the width narrowed"
    assert held["embedding"].tobytes() == face["embedding"].tobytes()
