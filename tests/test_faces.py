"""Tests for smartgallery_ai.faces: replace_faces_for_file round-trip,
cosine-threshold clustering (chinese whispers), cluster-label preservation across
re-clustering, and multi-face-per-file cluster cardinality."""

import os
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


def test_cluster_faces_bridge_does_not_chain_cliques_together():
    """Regression for the production mega-cluster (97% of 22k faces in one
    cluster): single-linkage merged any two groups connected by ONE bridge
    face. Chinese whispers must keep two dense cliques separate even when a
    bridge face is similar to members of both."""
    conn = make_conn()
    rng = np.random.default_rng(5)
    base_a = np.zeros(16, dtype=np.float32); base_a[0] = 1.0
    base_b = np.zeros(16, dtype=np.float32); base_b[1] = 1.0
    clique_a = [base_a + rng.standard_normal(16).astype(np.float32) * 0.02 for _ in range(4)]
    clique_b = [base_b + rng.standard_normal(16).astype(np.float32) * 0.02 for _ in range(4)]
    bridge = (base_a + base_b) / np.linalg.norm(base_a + base_b)  # ~0.71 cos to both cliques

    all_vectors = clique_a + clique_b + [bridge.astype(np.float32)]
    file_ids = [f"br{i}" for i in range(len(all_vectors))]
    add_files(conn, file_ids)
    for fid, vec in zip(file_ids, all_vectors):
        _insert_instances(conn, fid, [vec])

    new_cluster_ids = cluster_faces(conn, "m1", "v1", threshold=0.6, min_cluster_size=2)

    # Single-linkage would produce ONE 9-member cluster via the bridge.
    assert len(new_cluster_ids) == 2
    sizes = sorted(r[0] for r in conn.execute("SELECT size FROM ai_face_clusters").fetchall())
    assert max(sizes) <= 5  # each clique (+ possibly the bridge) stays separate


def test_cluster_faces_is_deterministic():
    conn = make_conn()
    rng = np.random.default_rng(11)
    vectors = [rng.standard_normal(16).astype(np.float32) for _ in range(30)]
    file_ids = [f"det{i}" for i in range(len(vectors))]
    add_files(conn, file_ids)
    for fid, vec in zip(file_ids, vectors):
        _insert_instances(conn, fid, [vec])

    def snapshot():
        cluster_faces(conn, "m1", "v1", threshold=0.2, min_cluster_size=2)
        return conn.execute(
            "SELECT file_id, cluster_id IS NULL, "
            "(SELECT size FROM ai_face_clusters c WHERE c.cluster_id = ai_face_instances.cluster_id) "
            "FROM ai_face_instances ORDER BY file_id"
        ).fetchall()

    first = snapshot()
    second = snapshot()
    assert [r[1:] for r in first] == [r[1:] for r in second]


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


# --- neighbor-graph backend contract ---------------------------------------
# All backends must produce the identical exhaustive edge set: same (i, j)
# pairs, same similarities (IEEE float32 in every backend; the torch backend
# pins allow_tf32=False so Ampere+ tensor cores cannot skew the boundary).

def _edge_set(graph):
    indptr, cols, weights = graph
    rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
    return {
        (int(i), int(j), round(float(w), 4))
        for i, j, w in zip(rows, cols, weights)
    }


def _backend_fixture():
    """Two tight cliques + spread noise, all sims kept >= 1e-3 away from the
    0.6 threshold except the deliberately exact boundary pair below."""
    rng = np.random.default_rng(7)
    base_a = rng.standard_normal(64).astype(np.float32)
    base_b = rng.standard_normal(64).astype(np.float32)
    rows = []
    for base in (base_a, base_b):
        for _ in range(6):
            v = base + 0.05 * rng.standard_normal(64).astype(np.float32)
            rows.append(v)
    rows.extend(rng.standard_normal((8, 64)).astype(np.float32))
    m = np.stack(rows)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m.astype(np.float32)


def test_neighbor_graph_faiss_matches_numpy():
    faiss = pytest.importorskip("faiss")
    assert faiss is not None
    from smartgallery_ai.faces import _neighbor_graph_faiss, _neighbor_graph_numpy

    m = _backend_fixture()
    assert _edge_set(_neighbor_graph_faiss(m, 0.6)) == _edge_set(
        _neighbor_graph_numpy(m, 0.6)
    )


def test_neighbor_graph_torch_cuda_matches_numpy():
    """Runs in a SUBPROCESS: importing torch in this process would poison
    sys.modules and break test_normal_browsing_never_imports_torch's
    process-level lazy-import guard."""
    import subprocess
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = (
        "import importlib.util, sys\n"
        f"sys.path.insert(0, {repo_root!r})\n"
        "if importlib.util.find_spec('torch') is None:\n"
        "    print('SKIP: torch not installed'); raise SystemExit(0)\n"
        "import torch\n"
        "if not torch.cuda.is_available():\n"
        "    print('SKIP: no CUDA device'); raise SystemExit(0)\n"
        "from tests.test_faces import _backend_fixture, _edge_set\n"
        "from smartgallery_ai.faces import _neighbor_graph_numpy, _neighbor_graph_torch_cuda\n"
        "m = _backend_fixture()\n"
        "assert _edge_set(_neighbor_graph_torch_cuda(m, 0.6)) == _edge_set(\n"
        "    _neighbor_graph_numpy(m, 0.6)\n"
        ")\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if result.stdout.startswith("SKIP"):
        pytest.skip(result.stdout.strip())
    assert "OK" in result.stdout


def test_neighbor_graph_threshold_boundary_inclusive():
    """sim == threshold exactly must be an edge (>= contract). d=4 with a
    single nonzero product term makes the float32 result exact in every
    backend: dot((1,0,0,0), (0.6, 0.8, 0, 0)) is the float32 literal 0.6."""
    from smartgallery_ai.faces import _neighbor_graph_numpy

    m = np.array([[1, 0, 0, 0], [0.6, 0.8, 0, 0]], dtype=np.float32)
    thr = float(np.float32(0.6))
    at_indptr, at_cols, _ = _neighbor_graph_numpy(m, thr)
    assert list(at_indptr) == [0, 1, 2] and list(at_cols) == [1, 0]
    above_indptr, _, _ = _neighbor_graph_numpy(
        m, float(np.nextafter(np.float32(0.6), np.float32(1)))
    )
    assert list(above_indptr) == [0, 0, 0]


def test_neighbor_graph_unknown_backend_request_raises(monkeypatch):
    """An explicit backend request that cannot be honored must fail loud,
    never silently fall back."""
    from smartgallery_ai.faces import _neighbor_graph

    monkeypatch.setenv("AI_DAM_FACE_GRAPH_BACKEND", "quantum")
    with pytest.raises(ValueError):
        _neighbor_graph(_backend_fixture(), 0.6)


def test_neighbor_graph_reports_backend_that_ran(monkeypatch):
    from smartgallery_ai.faces import _neighbor_graph

    monkeypatch.setenv("AI_DAM_FACE_GRAPH_BACKEND", "numpy")
    _, backend = _neighbor_graph(_backend_fixture(), 0.6)
    assert backend == "numpy"


def test_aiconfig_face_min_px_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_DAM_FACE_MIN_PX", "32")
    cfg = AIConfig.from_env(str(tmp_path), str(tmp_path / "db.sqlite"))
    assert cfg.face_min_px == 32


def test_aiconfig_face_min_px_default(tmp_path):
    cfg = AIConfig.from_env(str(tmp_path), str(tmp_path / "db.sqlite"))
    assert cfg.face_min_px == 24


def test_aiconfig_face_embedder_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_DAM_FACE_EMBEDDER", "sface")
    cfg = AIConfig.from_env(str(tmp_path), str(tmp_path / "db.sqlite"))
    assert cfg.face_embedder == "sface"


def test_aiconfig_face_embedder_default(tmp_path):
    cfg = AIConfig.from_env(str(tmp_path), str(tmp_path / "db.sqlite"))
    assert cfg.face_embedder == "auto"


# --- ArcFace alignment (insightface norm_crop contract) -----------------------


def test_umeyama_matches_skimage_fixture():
    """The numpy Umeyama estimator must reproduce
    skimage.transform.SimilarityTransform.estimate — insightface's
    estimator — on a captured fixture (skimage 0.26.0, verified to ~1e-6
    over 200 random landmark sets)."""
    from smartgallery_ai.faces import _ARCFACE_DST, _umeyama_similarity

    lmk = np.array([[210.5, 180.25], [312.75, 178.0], [260.0, 240.5],
                    [222.25, 300.75], [305.5, 298.0]])
    expected = np.array([
        [3.41340034e-01, -5.51772644e-03, -3.21517002e+01],
        [5.51772644e-03, 3.41340034e-01, -1.12969063e+01]])
    m = _umeyama_similarity(lmk, _ARCFACE_DST)
    # fixture repr carries 8 significant digits; translation terms are
    # O(30), so equality holds to ~2e-6 absolute
    assert np.abs(m - expected).max() < 5e-6


def test_umeyama_identity_when_landmarks_match_template():
    from smartgallery_ai.faces import _ARCFACE_DST, _umeyama_similarity

    m = _umeyama_similarity(_ARCFACE_DST, _ARCFACE_DST)
    assert np.abs(m - np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])).max() < 1e-6


def test_arcface_norm_crop_shape_and_identity_warp():
    """Landmarks already on the template warp the image onto itself."""
    from smartgallery_ai.faces import _ARCFACE_DST, _arcface_norm_crop

    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (112, 112, 3), dtype=np.uint8)
    warped = _arcface_norm_crop(img, _ARCFACE_DST.copy())
    assert warped.shape == (112, 112, 3)
    assert np.abs(warped.astype(int) - img.astype(int)).mean() < 1.0


# --- embedder selection ------------------------------------------------------


def test_opencv_backend_rejects_unknown_embedder(tmp_path):
    from smartgallery_ai.faces import OpenCVFaceBackend, _YUNET_FILENAME

    (tmp_path / _YUNET_FILENAME).write_bytes(b"")
    with pytest.raises(ValueError, match="unknown face embedder"):
        OpenCVFaceBackend(str(tmp_path), embedder="bogus")


def test_opencv_backend_forced_arcface_missing_raises(tmp_path):
    from smartgallery_ai.embedders import BackendUnavailable
    from smartgallery_ai.faces import OpenCVFaceBackend, _YUNET_FILENAME

    (tmp_path / _YUNET_FILENAME).write_bytes(b"")
    with pytest.raises(BackendUnavailable, match="ArcFace model not found"):
        OpenCVFaceBackend(str(tmp_path), embedder="arcface")


def test_opencv_backend_auto_falls_back_to_sface_version(tmp_path):
    """auto with only sface weights resolves to the sface identity; the
    load then fails on the empty file, proving resolution happened first."""
    from smartgallery_ai.embedders import BackendUnavailable
    from smartgallery_ai.faces import (OpenCVFaceBackend, _SFACE_FILENAME,
                                       _YUNET_FILENAME)

    (tmp_path / _YUNET_FILENAME).write_bytes(b"")
    (tmp_path / _SFACE_FILENAME).write_bytes(b"")
    with pytest.raises(BackendUnavailable, match="failed to load face models"):
        OpenCVFaceBackend(str(tmp_path), embedder="auto")


def test_resolve_cluster_threshold_explicit_config_wins(tmp_path):
    from smartgallery_ai.faces import StubFaceBackend, resolve_cluster_threshold

    cfg = AIConfig(face_cluster_threshold=0.7)
    assert resolve_cluster_threshold(cfg, StubFaceBackend({})) == 0.7


def test_resolve_cluster_threshold_backend_default_when_unset():
    from smartgallery_ai.faces import StubFaceBackend, resolve_cluster_threshold

    cfg = AIConfig()
    backend = StubFaceBackend({})
    assert resolve_cluster_threshold(cfg, backend) == 0.55
    backend.default_cluster_threshold = 0.40
    assert resolve_cluster_threshold(cfg, backend) == 0.40


def test_aiconfig_cluster_threshold_env_and_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("AI_DAM_FACE_CLUSTER_THRESHOLD", raising=False)
    cfg = AIConfig.from_env(str(tmp_path), str(tmp_path / "db.sqlite"))
    assert cfg.face_cluster_threshold is None
    monkeypatch.setenv("AI_DAM_FACE_CLUSTER_THRESHOLD", "0.62")
    cfg = AIConfig.from_env(str(tmp_path), str(tmp_path / "db.sqlite"))
    assert cfg.face_cluster_threshold == 0.62


# --- InsightFaceBackend selection / inventory / compare -----------------------


def test_get_face_backend_auto_prefers_insightface(tmp_path, monkeypatch):
    import smartgallery_ai.faces as F

    class _FakeInsight(F.FaceBackend):
        model_id = "insightface/antelopev2"
        model_version = "scrfd10g+glintr100-v1"

        def __init__(self, models_dir, min_det_score, min_face_px):
            pass

        def detect(self, img):
            return []

    monkeypatch.setattr(F, "InsightFaceBackend", _FakeInsight)
    cfg = AIConfig(face_backend="auto", models_dir=str(tmp_path))
    assert isinstance(get_face_backend(cfg), _FakeInsight)


def test_get_face_backend_auto_falls_back_when_insightface_missing(tmp_path):
    """No antelopev2 pack and no opencv weights: auto resolves to None
    (insightface raises internally, opencv raises internally, no crash)."""
    cfg = AIConfig(face_backend="auto", models_dir=str(tmp_path))
    assert get_face_backend(cfg) is None


def test_get_face_backend_forced_insightface_missing_raises(tmp_path):
    from smartgallery_ai.embedders import BackendUnavailable

    cfg = AIConfig(face_backend="insightface", models_dir=str(tmp_path))
    with pytest.raises(BackendUnavailable, match="antelopev2 pack not found"):
        get_face_backend(cfg)


def test_installed_pipelines_inventory(tmp_path):
    """Three pipelines, weight presence per file, nothing active when no
    weights exist."""
    from smartgallery_ai.faces import installed_pipelines

    cfg = AIConfig(face_backend="auto", models_dir=str(tmp_path))
    inv = installed_pipelines(cfg)
    assert [p["name"] for p in inv] == [
        "scrfd+glintr100", "yunet+arcface", "yunet+sface"]
    assert all(p["weights_present"] is False for p in inv)
    assert all(p["active"] is False for p in inv)
    versions = {p["model_version"] for p in inv}
    assert len(versions) == 3  # distinct model_versions never mix outputs


def test_compare_detectors_reports_every_lane(tmp_path, monkeypatch):
    """Both lanes answer; an unavailable lane carries its error while the
    other still reports detections, and the inventory rides along."""
    import smartgallery_ai.faces as F
    from PIL import Image

    det = FaceDetection(bbox=(0.1, 0.1, 0.2, 0.2),
                        landmarks=[(0.15, 0.15)], det_score=0.9,
                        embedding=np.ones(4, dtype=np.float32))

    class _FakeCv(F.FaceBackend):
        model_id = "opencv/yunet+sface"
        model_version = "v-cv"

        def __init__(self, *a, **k):
            pass

        def detect(self, img):
            return [det]

    def _boom(*a, **k):
        raise F.BackendUnavailable("antelopev2 pack not found")

    monkeypatch.setattr(F, "OpenCVFaceBackend", _FakeCv)
    monkeypatch.setattr(F, "InsightFaceBackend", _boom)
    cfg = AIConfig(face_backend="auto", models_dir=str(tmp_path))
    out = F.compare_detectors(Image.new("RGB", (10, 10)), cfg)
    assert out["lanes"]["yunet"]["faces"][0]["det_score"] == 0.9
    assert out["lanes"]["yunet"]["faces"][0]["landmarks"] == [(0.15, 0.15)]
    assert "antelopev2" in out["lanes"]["scrfd"]["error"]
    assert len(out["installed"]) == 3
