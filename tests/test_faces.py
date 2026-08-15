"""Tests for smartgallery_ai.faces: replace_faces_for_file round-trip,
cosine-threshold clustering (union-find), cluster-label preservation across
re-clustering, and multi-face-per-file cluster cardinality."""

import sqlite3

import numpy as np
import pytest

from smartgallery_ai import AIConfig
from smartgallery_ai.faces import (
    FaceDetection,
    StubFaceBackend,
    cluster_faces,
    get_face_backend,
    image_key,
    replace_faces_for_file,
)
from smartgallery_ai.schema import init_schema


# --- fixtures / helpers -----------------------------------------------------


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


def tight_group(seed: int, n: int, dim: int = 16, spread: float = 0.02) -> list:
    """n embeddings clustered tightly around one random unit direction
    (cosine ~0.99+ between members)."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(dim).astype(np.float32)
    base /= np.linalg.norm(base)
    vecs = []
    for _ in range(n):
        noise = rng.standard_normal(dim).astype(np.float32) * spread
        vecs.append((base + noise).astype(np.float32))
    return vecs


# --- FaceDetection -----------------------------------------------------------


def test_face_detection_derives_dim_from_embedding():
    det = FaceDetection(
        bbox=[0.0, 0.0, 0.5, 0.5],
        landmarks=[],
        det_score=0.7,
        embedding=np.arange(8, dtype=np.float64),
    )
    assert det.dim == 8
    assert det.embedding.dtype == np.float32
    assert det.bbox == (0.0, 0.0, 0.5, 0.5)


def test_face_detection_none_embedding_keeps_dim_none():
    det = FaceDetection(bbox=(0, 0, 1, 1), landmarks=[], det_score=0.5, embedding=None)
    assert det.embedding is None
    assert det.dim is None


# --- replace_faces_for_file --------------------------------------------------


def test_replace_faces_for_file_three_detections_round_trip():
    conn = make_conn()
    add_files(conn, ["f1"])
    dets = [detection(seed=i) for i in range(3)]

    face_ids = replace_faces_for_file(conn, "f1", dets, "m1", "v1", source_mtime=1000.0, now=2000.0)
    assert len(face_ids) == 3

    rows = conn.execute(
        "SELECT face_id, bbox_x, bbox_y, bbox_w, bbox_h, det_score, embedding, dim, "
        "model_id, model_version, source_mtime, computed_at FROM ai_face_instances "
        "WHERE file_id = ? ORDER BY face_id",
        ("f1",),
    ).fetchall()
    assert len(rows) == 3

    for row, det in zip(rows, dets):
        (
            _face_id, bbox_x, bbox_y, bbox_w, bbox_h, det_score, embedding_blob, dim,
            model_id, model_version, source_mtime, computed_at,
        ) = row
        assert (bbox_x, bbox_y, bbox_w, bbox_h) == pytest.approx(det.bbox)
        assert det_score == pytest.approx(det.det_score)
        assert dim == det.dim
        got_embedding = np.frombuffer(embedding_blob, dtype="<f4")
        np.testing.assert_allclose(got_embedding, det.embedding)
        assert model_id == "m1"
        assert model_version == "v1"
        assert source_mtime == 1000.0
        assert computed_at == 2000.0


def test_replace_faces_for_file_rerun_with_fewer_detections_leaves_exact_count():
    conn = make_conn()
    add_files(conn, ["f1"])
    replace_faces_for_file(
        conn, "f1", [detection(seed=i) for i in range(3)], "m1", "v1", 1000.0, 2000.0
    )
    replace_faces_for_file(
        conn, "f1", [detection(seed=i) for i in range(2)], "m1", "v1", 1000.0, 2001.0
    )
    rows = conn.execute(
        "SELECT face_id FROM ai_face_instances WHERE file_id = ?", ("f1",)
    ).fetchall()
    assert len(rows) == 2


def test_replace_faces_for_file_multi_face_asset_multiple_rows():
    conn = make_conn()
    add_files(conn, ["group_photo"])
    dets = [detection(seed=i) for i in range(4)]
    face_ids = replace_faces_for_file(conn, "group_photo", dets, "m1", "v1", 1000.0, 2000.0)
    assert len(set(face_ids)) == 4
    count = conn.execute(
        "SELECT COUNT(*) FROM ai_face_instances WHERE file_id = ?", ("group_photo",)
    ).fetchone()[0]
    assert count == 4


def test_replace_faces_for_file_no_embedding_stores_null_blob():
    conn = make_conn()
    add_files(conn, ["f1"])
    det = FaceDetection(bbox=(0, 0, 1, 1), landmarks=[], det_score=0.4, embedding=None)
    replace_faces_for_file(conn, "f1", [det], "m1", "v1", 1000.0, 2000.0)
    embedding_blob, dim = conn.execute(
        "SELECT embedding, dim FROM ai_face_instances WHERE file_id = ?", ("f1",)
    ).fetchone()
    assert embedding_blob is None
    assert dim is None


# --- StubFaceBackend ----------------------------------------------------------


def test_stub_face_backend_callable_is_deterministic():
    from PIL import Image

    img = Image.new("RGB", (32, 32), color=(1, 2, 3))
    dets = [detection(seed=0)]
    backend = StubFaceBackend(lambda _im: dets)
    assert backend.detect(img) == dets
    assert backend.detect(img) == dets


def test_stub_face_backend_mapping_keyed_by_image_content():
    from PIL import Image

    img_a = Image.new("RGB", (16, 16), color=(10, 10, 10))
    img_b = Image.new("RGB", (16, 16), color=(200, 0, 0))
    dets_a = [detection(seed=1)]
    backend = StubFaceBackend({image_key(img_a): dets_a})
    assert backend.detect(img_a) == dets_a
    assert backend.detect(img_b) == []  # unmapped image -> no faces


def test_get_face_backend_none_and_stub():
    config = AIConfig(face_backend="none")
    assert get_face_backend(config) is None

    config = AIConfig(face_backend="stub")
    backend = get_face_backend(config)
    assert isinstance(backend, StubFaceBackend)


def test_get_face_backend_auto_without_models_returns_none(tmp_path):
    config = AIConfig(face_backend="auto", models_dir=str(tmp_path))
    assert get_face_backend(config) is None


def test_get_face_backend_unknown_name_raises():
    config = AIConfig(face_backend="not-a-real-backend")
    with pytest.raises(ValueError):
        get_face_backend(config)


# --- cluster_faces ------------------------------------------------------------


def _insert_instances(conn, file_id, vectors, model_id="m1", model_version="v1"):
    dets = [
        FaceDetection(bbox=(0.0, 0.0, 0.1, 0.1), landmarks=[], det_score=0.9, embedding=v)
        for v in vectors
    ]
    return replace_faces_for_file(conn, file_id, dets, model_id, model_version, 1000.0, 2000.0)


def test_cluster_faces_two_tight_groups_plus_outliers():
    conn = make_conn()
    group_a = tight_group(seed=1, n=5)
    group_b = tight_group(seed=2, n=5)
    rng = np.random.default_rng(99)
    outliers = [rng.standard_normal(16).astype(np.float32) * 5.0 for _ in range(3)]

    all_vectors = group_a + group_b + outliers
    file_ids = [f"f{i}" for i in range(len(all_vectors))]
    add_files(conn, file_ids)
    # one face per file, in one call each so every embedding is its own instance
    for fid, vec in zip(file_ids, all_vectors):
        _insert_instances(conn, fid, [vec])

    # sanity: intra-group cosine really is ~0.99+
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    intra_a = [cos(group_a[i], group_a[j]) for i in range(5) for j in range(i + 1, 5)]
    assert min(intra_a) > 0.98

    new_cluster_ids = cluster_faces(conn, "m1", "v1", threshold=0.9, min_cluster_size=2)
    assert len(new_cluster_ids) == 2

    clusters = conn.execute(
        "SELECT cluster_id, size, centroid, dim FROM ai_face_clusters ORDER BY cluster_id"
    ).fetchall()
    assert len(clusters) == 2
    sizes = sorted(c[1] for c in clusters)
    assert sizes == [5, 5]

    for cluster_id, size, centroid_blob, dim in clusters:
        centroid = np.frombuffer(centroid_blob, dtype="<f4")
        assert dim == 16
        assert centroid.shape == (16,)
        assert np.linalg.norm(centroid) == pytest.approx(1.0, abs=1e-5)

    # outliers must have NULL cluster_id
    outlier_ids = file_ids[10:13]
    for fid in outlier_ids:
        cluster_id = conn.execute(
            "SELECT cluster_id FROM ai_face_instances WHERE file_id = ?", (fid,)
        ).fetchone()[0]
        assert cluster_id is None

    # every member of the tight groups got assigned to some cluster
    for fid in file_ids[:10]:
        cluster_id = conn.execute(
            "SELECT cluster_id FROM ai_face_instances WHERE file_id = ?", (fid,)
        ).fetchone()[0]
        assert cluster_id is not None


def test_cluster_faces_label_preserved_across_recluster():
    conn = make_conn()
    group_a = tight_group(seed=5, n=4)
    group_b = tight_group(seed=6, n=4)
    file_ids_a = [f"a{i}" for i in range(4)]
    file_ids_b = [f"b{i}" for i in range(4)]
    add_files(conn, file_ids_a + file_ids_b)
    for fid, vec in zip(file_ids_a, group_a):
        _insert_instances(conn, fid, [vec])
    for fid, vec in zip(file_ids_b, group_b):
        _insert_instances(conn, fid, [vec])

    cluster_faces(conn, "m1", "v1", threshold=0.9, min_cluster_size=2)
    clusters = conn.execute(
        "SELECT cluster_id, centroid FROM ai_face_clusters ORDER BY cluster_id"
    ).fetchall()
    assert len(clusters) == 2

    # label the cluster containing group_a's members
    a_face_id = conn.execute(
        "SELECT face_id FROM ai_face_instances WHERE file_id = ?", (file_ids_a[0],)
    ).fetchone()[0]
    a_cluster_id = conn.execute(
        "SELECT cluster_id FROM ai_face_instances WHERE face_id = ?", (a_face_id,)
    ).fetchone()[0]
    conn.execute(
        "UPDATE ai_face_clusters SET label = ? WHERE cluster_id = ?", ("Alice", a_cluster_id)
    )
    conn.commit()

    # add one more member to group_a and re-cluster; label must carry over
    # to whichever *new* cluster_id now represents that same group.
    extra = (group_a[0] + np.random.default_rng(7).standard_normal(16).astype(np.float32) * 0.02)
    add_files(conn, ["a4"])
    _insert_instances(conn, "a4", [extra.astype(np.float32)])

    new_cluster_ids = cluster_faces(conn, "m1", "v1", threshold=0.9, min_cluster_size=2)
    assert len(new_cluster_ids) == 2

    labels = {
        cid: label
        for cid, label in conn.execute(
            "SELECT cluster_id, label FROM ai_face_clusters"
        ).fetchall()
    }
    assert "Alice" in labels.values()
    labeled_cluster_id = [cid for cid, label in labels.items() if label == "Alice"][0]

    # every original group_a member (plus the new one) should now be in the
    # relabeled cluster
    for fid in file_ids_a + ["a4"]:
        cid = conn.execute(
            "SELECT cluster_id FROM ai_face_instances WHERE file_id = ?", (fid,)
        ).fetchone()[0]
        assert cid == labeled_cluster_id

    # clusters were fully replaced (AUTOINCREMENT never reuses an id), yet
    # the label survived onto whichever new cluster now represents group_a
    old_cluster_ids = {c[0] for c in clusters}
    assert old_cluster_ids.isdisjoint(set(labels.keys()))


def test_cluster_faces_multi_face_file_appears_in_two_different_clusters():
    conn = make_conn()
    group_a = tight_group(seed=11, n=3)
    group_b = tight_group(seed=12, n=3)
    add_files(conn, ["shared", "a0", "a1", "b0", "b1"])

    # "shared" has two faces: one belonging to group_a, one to group_b
    _insert_instances(conn, "shared", [group_a[0], group_b[0]])
    _insert_instances(conn, "a0", [group_a[1]])
    _insert_instances(conn, "a1", [group_a[2]])
    _insert_instances(conn, "b0", [group_b[1]])
    _insert_instances(conn, "b1", [group_b[2]])

    cluster_faces(conn, "m1", "v1", threshold=0.9, min_cluster_size=2)

    shared_rows = conn.execute(
        "SELECT face_id, cluster_id FROM ai_face_instances WHERE file_id = ? ORDER BY face_id",
        ("shared",),
    ).fetchall()
    assert len(shared_rows) == 2
    cluster_ids = {row[1] for row in shared_rows}
    assert None not in cluster_ids
    assert len(cluster_ids) == 2  # the two faces of "shared" belong to two different clusters


def test_cluster_faces_empty_is_idempotent_noop():
    conn = make_conn()
    result = cluster_faces(conn, "m1", "v1", threshold=0.9)
    assert result == []
    assert conn.execute("SELECT COUNT(*) FROM ai_face_clusters").fetchone()[0] == 0


def test_cluster_faces_below_min_size_stays_unclustered():
    conn = make_conn()
    vecs = tight_group(seed=20, n=2)
    add_files(conn, ["only1", "only2"])
    _insert_instances(conn, "only1", [vecs[0]])
    _insert_instances(conn, "only2", [vecs[1]])

    result = cluster_faces(conn, "m1", "v1", threshold=0.9, min_cluster_size=3)
    assert result == []
    for fid in ("only1", "only2"):
        cid = conn.execute(
            "SELECT cluster_id FROM ai_face_instances WHERE file_id = ?", (fid,)
        ).fetchone()[0]
        assert cid is None
