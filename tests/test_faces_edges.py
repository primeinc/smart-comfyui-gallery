"""Edge-case tests for smartgallery_ai.faces: OpenCV backend availability
contracts (missing/corrupt weights, missing cv2 APIs), get_face_backend
resolution rules, image_key stability, replace_faces_for_file transactional
rollback and cross-model replacement, and cluster_faces edge behavior
(transitive chaining, label handling on dim change and cluster merge,
inconsistent-dim failure rollback, params provenance, empty recluster).
"""

import json
import sqlite3

import cv2
import numpy as np
import pytest
from PIL import Image

from smartgallery_ai import AIConfig
from smartgallery_ai.embedders import BackendUnavailable
from smartgallery_ai.faces import (
    FaceDetection,
    OpenCVFaceBackend,
    StubFaceBackend,
    _clamp01,
    _pil_to_bgr,
    cluster_faces,
    get_face_backend,
    image_key,
    replace_faces_for_file,
)
from smartgallery_ai.schema import init_schema

# --- fixtures / helpers (mirrors tests/test_faces.py) ------------------------


def make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE files (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            mtime REAL NOT NULL,
            name TEXT NOT NULL,
            type TEXT
        )
        """
    )
    init_schema(conn)
    return conn


def add_files(conn, file_ids, mtime=1000.0):
    for fid in file_ids:
        conn.execute(
            "INSERT INTO files (id, path, mtime, name, type) VALUES (?, ?, ?, ?, ?)",
            (fid, f"/gallery/{fid}.png", mtime, fid, "image"),
        )
    conn.commit()


def detection(seed: int, dim: int = 16, embedding: np.ndarray = None) -> FaceDetection:
    rng = np.random.default_rng(seed)
    if embedding is None:
        embedding = rng.standard_normal(dim).astype(np.float32)
    return FaceDetection(
        bbox=(0.1, 0.1, 0.2, 0.2),
        landmarks=[(0.15, 0.15), (0.25, 0.15)],
        det_score=0.9,
        embedding=embedding,
    )


def _insert_instances(conn, file_id, vectors, model_id="m1", model_version="v1"):
    dets = [FaceDetection(bbox=(0.0, 0.0, 0.1, 0.1), landmarks=[], det_score=0.9, embedding=v) for v in vectors]
    return replace_faces_for_file(conn, file_id, dets, model_id, model_version, 1000.0, 2000.0)


def tight_group(seed: int, n: int, dim: int = 16, spread: float = 0.02) -> list:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(dim).astype(np.float32)
    base /= np.linalg.norm(base)
    vecs = []
    for _ in range(n):
        noise = rng.standard_normal(dim).astype(np.float32) * spread
        vecs.append((base + noise).astype(np.float32))
    return vecs


# --- small pure helpers -------------------------------------------------------


def test_clamp01_boundaries():
    """_clamp01 clamps below-zero to 0.0, above-one to 1.0, passes through in-range."""
    assert _clamp01(-0.5) == 0.0
    assert _clamp01(1.5) == 1.0
    assert _clamp01(0.0) == 0.0
    assert _clamp01(1.0) == 1.0
    assert _clamp01(0.25) == 0.25


def test_pil_to_bgr_swaps_channels_and_converts_mode():
    """_pil_to_bgr yields OpenCV BGR channel order and coerces non-RGB modes via RGB."""
    img = Image.new("RGB", (2, 1), color=(10, 20, 30))
    bgr = _pil_to_bgr(img)
    assert bgr.shape == (1, 2, 3)
    assert list(bgr[0, 0]) == [30, 20, 10]

    gray = Image.new("L", (2, 2), color=77)
    bgr_gray = _pil_to_bgr(gray)
    assert bgr_gray.shape == (2, 2, 3)
    assert list(bgr_gray[0, 0]) == [77, 77, 77]


# --- image_key ----------------------------------------------------------------


def test_image_key_stable_across_objects_and_sensitive_to_content():
    """image_key is identical for equal pixel content and differs by pixels, size, and mode."""
    a1 = Image.new("RGB", (8, 8), color=(1, 2, 3))
    a2 = Image.new("RGB", (8, 8), color=(1, 2, 3))
    assert image_key(a1) == image_key(a2)

    different_pixels = Image.new("RGB", (8, 8), color=(4, 5, 6))
    assert image_key(a1) != image_key(different_pixels)

    different_size = Image.new("RGB", (8, 4), color=(1, 2, 3))
    assert image_key(a1) != image_key(different_size)

    different_mode = Image.new("RGBA", (8, 8), color=(1, 2, 3, 255))
    assert image_key(a1) != image_key(different_mode)


# --- OpenCVFaceBackend availability contract ---------------------------------


def test_opencv_backend_empty_models_dir_names_missing_yunet_file(tmp_path):
    """OpenCVFaceBackend raises BackendUnavailable naming the missing YuNet onnx path."""
    with pytest.raises(BackendUnavailable) as excinfo:
        OpenCVFaceBackend(str(tmp_path))
    msg = str(excinfo.value)
    assert "face_detection_yunet_2023mar.onnx" in msg
    assert str(tmp_path) in msg


def test_opencv_backend_missing_sface_names_missing_sface_file(tmp_path):
    """With only YuNet present, OpenCVFaceBackend names the missing SFace onnx path."""
    (tmp_path / "face_detection_yunet_2023mar.onnx").write_bytes(b"placeholder")
    with pytest.raises(BackendUnavailable) as excinfo:
        OpenCVFaceBackend(str(tmp_path))
    msg = str(excinfo.value)
    assert "face_recognition_sface_2021dec.onnx" in msg


def test_opencv_backend_corrupt_model_files_wrap_as_backend_unavailable(tmp_path):
    """Unloadable weight files raise BackendUnavailable ('failed to load'), not cv2.error."""
    (tmp_path / "face_detection_yunet_2023mar.onnx").write_bytes(b"not an onnx model")
    (tmp_path / "face_recognition_sface_2021dec.onnx").write_bytes(b"not an onnx model")
    with pytest.raises(BackendUnavailable) as excinfo:
        OpenCVFaceBackend(str(tmp_path))
    assert "failed to load face models" in str(excinfo.value)


def test_opencv_backend_missing_cv2_api_raises_backend_unavailable(tmp_path, monkeypatch):
    """A cv2 build without FaceDetectorYN fails closed with BackendUnavailable."""
    monkeypatch.delattr(cv2, "FaceDetectorYN")
    with pytest.raises(BackendUnavailable) as excinfo:
        OpenCVFaceBackend(str(tmp_path))
    assert "FaceDetectorYN" in str(excinfo.value)


# --- get_face_backend ---------------------------------------------------------


def test_get_face_backend_explicit_opencv_without_weights_raises(tmp_path):
    """Explicit 'opencv' propagates BackendUnavailable instead of degrading to None."""
    config = AIConfig(face_backend="opencv", models_dir=str(tmp_path))
    with pytest.raises(BackendUnavailable):
        get_face_backend(config)


def test_get_face_backend_stub_default_source_detects_no_faces():
    """'stub' with no face_stub_source in extra defaults to an always-empty detector."""
    config = AIConfig(face_backend="stub")
    backend = get_face_backend(config)
    assert isinstance(backend, StubFaceBackend)
    img = Image.new("RGB", (16, 16), color=(9, 9, 9))
    assert backend.detect(img) == []


def test_get_face_backend_stub_uses_extra_source_mapping():
    """'stub' wires config.extra['face_stub_source'] through as the detection source."""
    img = Image.new("RGB", (16, 16), color=(50, 60, 70))
    other = Image.new("RGB", (16, 16), color=(0, 0, 0))
    dets = [detection(seed=3)]
    config = AIConfig(face_backend="stub", extra={"face_stub_source": {image_key(img): dets}})
    backend = get_face_backend(config)
    assert backend.detect(img) == dets
    assert backend.detect(other) == []


# --- replace_faces_for_file ---------------------------------------------------


def test_replace_faces_keeps_other_models_rows():
    """Replacement is scoped to (file, model_id, model_version): another
    model's rows survive a different pipeline's run, and a version bump of
    the SAME model replaces only that model's rows."""
    conn = make_conn()
    add_files(conn, ["f1"])
    replace_faces_for_file(conn, "f1", [detection(seed=1)], "old-model", "v0", 1000.0, 2000.0)

    replace_faces_for_file(conn, "f1", [detection(seed=2)], "new-model", "v1", 1000.0, 2001.0)
    rows = conn.execute(
        "SELECT model_id, model_version FROM ai_face_instances WHERE file_id = ? ORDER BY model_id", ("f1",)
    ).fetchall()
    assert rows == [("new-model", "v1"), ("old-model", "v0")]

    replace_faces_for_file(conn, "f1", [detection(seed=3)], "old-model", "v1", 1000.0, 2002.0)
    rows = conn.execute(
        "SELECT model_id, model_version FROM ai_face_instances WHERE file_id = ? ORDER BY model_id, model_version",
        ("f1",),
    ).fetchall()
    assert rows == [("new-model", "v1"), ("old-model", "v0"), ("old-model", "v1")]


def test_replace_faces_failure_mid_insert_rolls_back_prior_rows():
    """A failure during replacement rolls back, leaving the previously committed rows intact."""
    conn = make_conn()
    add_files(conn, ["f1"])
    original_ids = replace_faces_for_file(
        conn, "f1", [detection(seed=1), detection(seed=2)], "m1", "v1", 1000.0, 2000.0
    )

    good = detection(seed=3)
    broken = detection(seed=4)
    broken.bbox = (0.1, 0.2, 0.3)  # too short: det.bbox[3] raises IndexError mid-loop

    with pytest.raises(IndexError):
        replace_faces_for_file(conn, "f1", [good, broken], "m1", "v1", 1000.0, 2001.0)

    rows = conn.execute("SELECT face_id FROM ai_face_instances WHERE file_id = ? ORDER BY face_id", ("f1",)).fetchall()
    assert [r[0] for r in rows] == original_ids


# --- cluster_faces ------------------------------------------------------------


def test_cluster_faces_transitive_chain_merges_subthreshold_pair():
    """A bridge face merges two faces into one cluster even when they are mutually below threshold."""
    conn = make_conn()
    e1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    e2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    bridge = ((e1 + e2) / np.sqrt(2.0)).astype(np.float32)  # cos 0.707 to each
    outlier = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    add_files(conn, ["f_out", "f_e1", "f_e2", "f_bridge"])
    _insert_instances(conn, "f_out", [outlier])
    _insert_instances(conn, "f_e1", [e1])
    _insert_instances(conn, "f_e2", [e2])
    _insert_instances(conn, "f_bridge", [bridge])

    new_cluster_ids = cluster_faces(conn, "m1", "v1", threshold=0.7, min_cluster_size=2)
    assert len(new_cluster_ids) == 1

    size = conn.execute("SELECT size FROM ai_face_clusters WHERE cluster_id = ?", (new_cluster_ids[0],)).fetchone()[0]
    assert size == 3

    for fid in ("f_e1", "f_e2", "f_bridge"):
        cid = conn.execute("SELECT cluster_id FROM ai_face_instances WHERE file_id = ?", (fid,)).fetchone()[0]
        assert cid == new_cluster_ids[0]
    assert conn.execute("SELECT cluster_id FROM ai_face_instances WHERE file_id = ?", ("f_out",)).fetchone()[0] is None


def test_cluster_faces_merged_clusters_keep_single_best_label():
    """When two labeled clusters merge on recluster, only the closest label survives (1:1 greedy)."""
    conn = make_conn()
    base = np.zeros(16, dtype=np.float32)
    base[0] = 1.0
    base2 = np.zeros(16, dtype=np.float32)
    base2[0], base2[1] = 1.0, 0.25
    base2 /= np.linalg.norm(base2)  # cos(base, base2) ~ 0.970

    a_files = ["a0", "a1", "a2"]
    b_files = ["b0", "b1"]
    add_files(conn, a_files + b_files)
    for fid in a_files:
        _insert_instances(conn, fid, [base.copy()])
    for fid in b_files:
        _insert_instances(conn, fid, [base2.copy()])

    # Run 1: high threshold keeps the two groups separate; label both.
    cluster_faces(conn, "m1", "v1", threshold=0.99, min_cluster_size=2)
    a_cluster = conn.execute("SELECT cluster_id FROM ai_face_instances WHERE file_id = ?", ("a0",)).fetchone()[0]
    b_cluster = conn.execute("SELECT cluster_id FROM ai_face_instances WHERE file_id = ?", ("b0",)).fetchone()[0]
    assert a_cluster is not None
    assert b_cluster is not None
    assert a_cluster != b_cluster
    conn.execute("UPDATE ai_face_clusters SET label = ? WHERE cluster_id = ?", ("Alice", a_cluster))
    conn.execute("UPDATE ai_face_clusters SET label = ? WHERE cluster_id = ?", ("Bob", b_cluster))
    conn.commit()

    # Run 2: lower threshold merges everything into one cluster. Its centroid
    # is closer to group A's old centroid (3 members vs 2), so "Alice" wins
    # and "Bob" is dropped rather than duplicated onto the same cluster.
    new_cluster_ids = cluster_faces(conn, "m1", "v1", threshold=0.95, min_cluster_size=2)
    assert len(new_cluster_ids) == 1

    labels = [r[0] for r in conn.execute("SELECT label FROM ai_face_clusters").fetchall()]
    assert labels == ["Alice"]


def test_cluster_faces_dim_change_drops_label_preservation_without_error():
    """Label carry-over is silently skipped when embedding dim changed since the labeled run."""
    conn = make_conn()
    vecs16 = tight_group(seed=30, n=2, dim=16)
    add_files(conn, ["g0", "g1"])
    _insert_instances(conn, "g0", [vecs16[0]])
    _insert_instances(conn, "g1", [vecs16[1]])
    ids_run1 = cluster_faces(conn, "m1", "v1", threshold=0.9, min_cluster_size=2)
    assert len(ids_run1) == 1
    conn.execute("UPDATE ai_face_clusters SET label = ? WHERE cluster_id = ?", ("Alice", ids_run1[0]))
    conn.commit()

    # Same model/version now produces dim-8 embeddings for every instance.
    vecs8 = tight_group(seed=31, n=2, dim=8)
    _insert_instances(conn, "g0", [vecs8[0]])
    _insert_instances(conn, "g1", [vecs8[1]])

    ids_run2 = cluster_faces(conn, "m1", "v1", threshold=0.9, min_cluster_size=2)
    assert len(ids_run2) == 1
    labels = [r[0] for r in conn.execute("SELECT label FROM ai_face_clusters").fetchall()]
    assert labels == [None]  # "Alice" not carried across the dim change


def test_cluster_faces_inconsistent_dims_raises_and_rolls_back():
    """Mixed embedding dims for one model/version raise ValueError and leave prior clusters intact."""
    conn = make_conn()
    vecs = tight_group(seed=40, n=2, dim=16)
    add_files(conn, ["g0", "g1", "odd"])
    _insert_instances(conn, "g0", [vecs[0]])
    _insert_instances(conn, "g1", [vecs[1]])
    ids_run1 = cluster_faces(conn, "m1", "v1", threshold=0.9, min_cluster_size=2)
    assert len(ids_run1) == 1

    _insert_instances(conn, "odd", [np.ones(8, dtype=np.float32)])  # wrong dim

    with pytest.raises(ValueError, match="inconsistent embedding dims"):
        cluster_faces(conn, "m1", "v1", threshold=0.9, min_cluster_size=2)

    # Rollback restored both the cluster row and the instances' cluster_id.
    remaining = [r[0] for r in conn.execute("SELECT cluster_id FROM ai_face_clusters").fetchall()]
    assert remaining == ids_run1
    for fid in ("g0", "g1"):
        cid = conn.execute("SELECT cluster_id FROM ai_face_instances WHERE file_id = ?", (fid,)).fetchone()[0]
        assert cid == ids_run1[0]


def test_cluster_faces_params_note_recorded_in_cluster_provenance():
    """Cluster rows record threshold/algo/min_cluster_size and the optional params_note."""
    conn = make_conn()
    vecs = tight_group(seed=50, n=2)
    add_files(conn, ["p0", "p1"])
    _insert_instances(conn, "p0", [vecs[0]])
    _insert_instances(conn, "p1", [vecs[1]])

    ids = cluster_faces(conn, "m1", "v1", threshold=0.9, min_cluster_size=2, params_note="tuned-v2")
    assert len(ids) == 1

    params_json = conn.execute("SELECT params FROM ai_face_clusters WHERE cluster_id = ?", (ids[0],)).fetchone()[0]
    params = json.loads(params_json)
    assert params.pop("graph_backend") in {"torch-cuda", "faiss-cpu", "numpy"}
    assert params == {
        "threshold": 0.9,
        "algo": "cosine-chinese-whispers",
        "min_cluster_size": 2,
        "note": "tuned-v2",
    }


def test_cluster_faces_recluster_after_all_instances_removed_clears_clusters():
    """Reclustering after every instance is gone deletes stale clusters and returns []."""
    conn = make_conn()
    vecs = tight_group(seed=60, n=2)
    add_files(conn, ["r0", "r1"])
    _insert_instances(conn, "r0", [vecs[0]])
    _insert_instances(conn, "r1", [vecs[1]])
    assert len(cluster_faces(conn, "m1", "v1", threshold=0.9, min_cluster_size=2)) == 1

    # Re-index both files with zero detections (faces disappeared upstream).
    replace_faces_for_file(conn, "r0", [], "m1", "v1", 1000.0, 3000.0)
    replace_faces_for_file(conn, "r1", [], "m1", "v1", 1000.0, 3000.0)

    result = cluster_faces(conn, "m1", "v1", threshold=0.9, min_cluster_size=2)
    assert result == []
    assert conn.execute("SELECT COUNT(*) FROM ai_face_clusters").fetchone()[0] == 0
