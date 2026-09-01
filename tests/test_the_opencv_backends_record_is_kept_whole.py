"""The OpenCV backend's native record is its producer's complete output.

This producer is a composition -- YuNet detection plus an ArcFace or SFace
recognizer -- and its complete output per face is the detector row (box,
five landmarks and confidence, in detect-input pixels) and the recognizer's
feature exactly as it came back. The record holds both, bit for bit, and
every `FaceDetection` field is a projection out of them: the normalized
bbox clamps to the frame, the flat embedding reshapes the feature, and
neither narrowing reaches the record.

REAL WEIGHTS, REAL PHOTOGRAPH, same grounds as
`test_a_measurement_is_kept_at_the_width_it_was_measured`: a stub on both
sides of a conservation claim only shows the stub agrees with itself.
"""

import hashlib
import os
import pathlib

import numpy as np
import pytest

from vision import facestore

DATASETS = pathlib.Path(os.environ.get("COMPAT_DATASETS", "C:/ComfyUI/output/sample-datasets"))
KYC = DATASETS / "caucasian-people-kyc-photo-dataset" / "files"

MODELS = pathlib.Path(os.environ.get("SG_MODELS_DIR", "C:/ComfyUI/output/.AImodels"))


def a_photograph() -> pathlib.Path | None:
    """One real face, chosen by content so the selection does not move."""
    if not KYC.is_dir():
        return None
    found: list[tuple[str, pathlib.Path]] = []
    for folder in sorted(KYC.iterdir(), key=lambda one: one.name):
        if not folder.is_dir():
            continue
        for image in sorted(folder.iterdir(), key=lambda one: one.name):
            if image.is_file():
                found.append((hashlib.sha256(image.read_bytes()).hexdigest(), image))
                break
    return min(found)[1] if found else None


CHOSEN = a_photograph()

needs_the_real_thing = pytest.mark.skipif(
    CHOSEN is None, reason=f"needs the KYC corpus under {KYC} and OpenCV face weights under {MODELS}"
)


@pytest.mark.slow
@needs_the_real_thing
def test_the_row_and_the_feature_survive_bit_for_bit(tmp_path):
    """Producer -> capture -> real database row -> replay, for the OpenCV
    backend -- the same whole lane the InsightFace shape gets, because a
    second producer that is only spot-checked in memory is a second chance
    for the row to lie."""
    from db import authored, connect, detect, faces_native, library, scan
    from vision import decode, faces

    try:
        backend = faces.OpenCVFaceBackend(str(MODELS))
    except faces.BackendUnavailable as why:
        pytest.skip(f"OpenCV face weights are not on this machine: {why}")

    assert CHOSEN is not None
    with decode.open_still(CHOSEN) as opened:
        opened.load()
        image = opened if opened.mode == "RGB" else opened.convert("RGB")
        found = backend.detect(image)
        assert found, f"YuNet found no face in {CHOSEN.name}; the corpus image is the input, not the claim"

        where = tmp_path / "library.sgly"
        connect.create(where)
        conn = connect.connect(where)
        try:
            root = tmp_path / "lib"
            root.mkdir()
            root_id = library.add_root(conn, root, "library", 0.0)
            folder_id = scan.ensure_folder(conn, root_id, None, "lib")
            file_id = scan.mint(conn, "file", "yunet")
            conn.execute(
                "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256, width, height,"
                " first_seen_at, last_seen_at) VALUES(?, ?, 'yunet.png', 'image', 10, 0, 'aa', ?, ?, 0, 0)",
                (file_id, folder_id, image.size[0], image.size[1]),
            )
            authored.add_user(conn, "will", "hash", "ADMIN", 0.0)
            written = detect.harvest(conn, backend, file_id, None, 0.0, image=image)
            assert written, "harvest recorded no OpenCV face"
            stored = [faces_native.native_of(conn, one) for one in written]
        finally:
            connect.close(conn)

    kept = [one for one in found if one.det_score >= detect.FLOOR]
    for one, native in zip(kept, stored, strict=True):
        assert native.producer == backend.model_id
        assert native.producer_version == backend.model_version
        assert sorted(native.record) == ["feature", "row"], "the record is not this producer's complete output"

        row = native.record["row"]
        assert row.dtype == np.float32
        assert row.shape == (15,), "YuNet's row is box, five landmark pairs and confidence"
        feature = native.record["feature"]
        assert feature.dtype == np.float32

        # The projections agree with the record they came from, and the
        # record keeps what they narrowed: the row's confidence is the
        # det_score, and the feature reshapes to the flat embedding.
        assert one.det_score == float(row[14])
        assert one.embedding is not None
        assert np.asarray(one.embedding).tobytes() == feature.reshape(-1).tobytes()
        assert one.native is not None
        assert facestore.thaw(one.native).record["row"].tobytes() == row.tobytes()
