"""End-to-end tests for smartgallery_ai.service: a tiny Flask app registers
create_ai_blueprint() against a temp DB seeded with fixture rows (planted
via the real wave-1 APIs, not raw INSERTs), and every route is exercised
through the Flask test client. Also covers create_ai_resolvers() end-to-end
against omniquery.engine.OmniQueryEngine.
"""

from __future__ import annotations

import json
import sqlite3
import time
from types import SimpleNamespace

import numpy as np
import pytest
from flask import Flask
from PIL import Image

from omniquery.engine import OmniQueryEngine
from omniquery.validation import AuthContext
from smartgallery_ai import (
    HASH_ALGO_VERSION,
    RUBRIC_VERSION,
    SPACE_SEMANTIC,
    SPACE_VISUAL,
    AIConfig,
    backends,
    hashing,
    vectors,
)
from smartgallery_ai import faces as F
from smartgallery_ai.faces import FaceDetection, StubFaceBackend, cluster_faces, replace_faces_for_file
from smartgallery_ai.review import (
    AlignmentElement,
    Finding,
    ReviewResult,
    StubSegmenter,
    generate_alignment_mask,
    generate_finding_mask,
    store_review,
)
from smartgallery_ai.schema import init_schema
from smartgallery_ai.service import create_ai_blueprint, create_ai_resolvers

# --- fixture construction ----------------------------------------------------


def _make_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE files (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            mtime REAL NOT NULL,
            name TEXT NOT NULL,
            type TEXT,
            workflow_prompt TEXT DEFAULT ''
        )
        """
    )
    init_schema(conn)
    return conn


def _add_file(conn, file_id, mtime=1000.0, file_type="image", path=None):
    path = path or f"/gallery/{file_id}.png"
    conn.execute(
        "INSERT INTO files (id, path, mtime, name, type) VALUES (?, ?, ?, ?, ?)",
        (file_id, path, mtime, file_id, file_type),
    )
    conn.commit()


def _tight_vector(base: np.ndarray, seed: int, spread: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (base + rng.standard_normal(base.shape[0]).astype(np.float32) * spread).astype(np.float32)


@pytest.fixture
def fixture(tmp_path):
    db_path = str(tmp_path / "gallery.sqlite")
    cache_dir = str(tmp_path / "cache")
    conn = _make_conn(db_path)
    now = time.time()

    # --- exact + near duplicates -------------------------------------------
    _add_file(conn, "dup_target")
    _add_file(conn, "dup_twin")
    _add_file(conn, "dup_near")
    _add_file(conn, "dup_far")

    shared_sha = "a" * 64
    hashing.upsert_hashes(
        conn,
        "dup_target",
        hashing.HashResult(sha256=shared_sha, phash64=0, dhash64=0),
        1000.0,
        HASH_ALGO_VERSION,
        now,
    )
    hashing.upsert_hashes(
        conn,
        "dup_twin",
        hashing.HashResult(sha256=shared_sha, phash64=-1, dhash64=0),
        1000.0,
        HASH_ALGO_VERSION,
        now,
    )
    hashing.upsert_hashes(
        conn,
        "dup_near",
        hashing.HashResult(sha256="b" * 64, phash64=0b111, dhash64=0),
        1000.0,
        HASH_ALGO_VERSION,
        now,
    )
    hashing.upsert_hashes(
        conn,
        "dup_far",
        hashing.HashResult(sha256="c" * 64, phash64=-1, dhash64=0),
        1000.0,
        HASH_ALGO_VERSION,
        now,
    )

    # --- semantic vs. visual similarity (spaces must genuinely differ) -----
    _add_file(conn, "sim_target")
    _add_file(conn, "sim_sem_near")
    _add_file(conn, "sim_vis_near")

    store = vectors.VectorStore(cache_dir=cache_dir, ephemeral=True)
    store.add(conn, "sim_target", SPACE_SEMANTIC, "m-sem", "v1", np.array([1, 0, 0, 0], dtype=np.float32), 1000.0)
    store.add(
        conn, "sim_sem_near", SPACE_SEMANTIC, "m-sem", "v1", np.array([0.95, 0.05, 0, 0], dtype=np.float32), 1000.0
    )
    store.add(conn, "sim_vis_near", SPACE_SEMANTIC, "m-sem", "v1", np.array([0, 1, 0, 0], dtype=np.float32), 1000.0)
    store.add(conn, "sim_target", SPACE_VISUAL, "m-vis", "v1", np.array([0, 0, 1, 0], dtype=np.float32), 1000.0)
    store.add(conn, "sim_sem_near", SPACE_VISUAL, "m-vis", "v1", np.array([0, 0, 0, 1], dtype=np.float32), 1000.0)
    store.add(conn, "sim_vis_near", SPACE_VISUAL, "m-vis", "v1", np.array([0, 0, 0.95, 0.05], dtype=np.float32), 1000.0)

    # --- faces: two tight groups plus one multi-face file spanning both ----
    for fid in ("multi_face", "grp_a1", "grp_a2", "grp_b1", "grp_b2"):
        _add_file(conn, fid)

    rng = np.random.default_rng(7)
    base_a = rng.standard_normal(16).astype(np.float32)
    base_a /= np.linalg.norm(base_a)
    base_b = rng.standard_normal(16).astype(np.float32)
    base_b /= np.linalg.norm(base_b)

    face_model_id = StubFaceBackend.model_id
    face_model_version = StubFaceBackend.model_version

    replace_faces_for_file(
        conn,
        "grp_a1",
        [FaceDetection(bbox=(0.1, 0.1, 0.2, 0.2), landmarks=[], det_score=0.9, embedding=_tight_vector(base_a, 10))],
        face_model_id,
        face_model_version,
        1000.0,
        now,
    )
    replace_faces_for_file(
        conn,
        "grp_a2",
        [FaceDetection(bbox=(0.1, 0.1, 0.2, 0.2), landmarks=[], det_score=0.9, embedding=_tight_vector(base_a, 11))],
        face_model_id,
        face_model_version,
        1000.0,
        now,
    )
    replace_faces_for_file(
        conn,
        "grp_b1",
        [FaceDetection(bbox=(0.3, 0.3, 0.2, 0.2), landmarks=[], det_score=0.85, embedding=_tight_vector(base_b, 20))],
        face_model_id,
        face_model_version,
        1000.0,
        now,
    )
    replace_faces_for_file(
        conn,
        "grp_b2",
        [FaceDetection(bbox=(0.3, 0.3, 0.2, 0.2), landmarks=[], det_score=0.85, embedding=_tight_vector(base_b, 21))],
        face_model_id,
        face_model_version,
        1000.0,
        now,
    )
    replace_faces_for_file(
        conn,
        "multi_face",
        [
            FaceDetection(
                bbox=(0.0, 0.0, 0.15, 0.15),
                landmarks=[(0.05, 0.05)],
                det_score=0.8,
                embedding=_tight_vector(base_a, 30),
            ),
            FaceDetection(
                bbox=(0.5, 0.5, 0.15, 0.15),
                landmarks=[(0.55, 0.55)],
                det_score=0.75,
                embedding=_tight_vector(base_b, 31),
            ),
        ],
        face_model_id,
        face_model_version,
        1000.0,
        now,
    )

    new_cluster_ids = cluster_faces(conn, face_model_id, face_model_version, threshold=0.9, min_cluster_size=2)
    assert len(new_cluster_ids) == 2

    # --- review: one global finding, one localizable finding + mask --------
    _add_file(conn, "review_file")
    review_result = ReviewResult(
        quality_score=6.0,
        prompt_alignment_score=2 / 3,
        summary="stub review",
        findings=[
            Finding(type="lighting", severity="medium", confidence=0.8, localizable=False, description="too dark"),
            Finding(
                type="artifact",
                severity="high",
                confidence=0.9,
                localizable=True,
                description="red box",
                bbox=(0.1, 0.1, 0.2, 0.2),
            ),
        ],
        alignment=[
            # located match -> gets a mask and a highlight
            AlignmentElement(ordinal=0, text="a red cube", satisfied=True, confidence=0.9, bbox=(0.1, 0.1, 0.2, 0.2)),
            # satisfied but whole-image (style) -> no locus, no mask
            AlignmentElement(ordinal=1, text="cinematic lighting", satisfied=True, confidence=0.6),
            AlignmentElement(ordinal=2, text="a blue sphere", satisfied=False, confidence=0.8),
        ],
    )
    review_id = store_review(
        conn,
        "review_file",
        review_result,
        "stub-critic",
        "stub-v1",
        RUBRIC_VERSION,
        None,
        1000.0,
        now,
    )
    finding_rows = conn.execute(
        "SELECT finding_id, localizable FROM ai_review_findings WHERE review_id = ? ORDER BY finding_id",
        (review_id,),
    ).fetchall()
    global_finding_id = next(r["finding_id"] for r in finding_rows if not r["localizable"])
    local_finding_id = next(r["finding_id"] for r in finding_rows if r["localizable"])

    mask_img = Image.new("RGB", (64, 64), (5, 5, 5))
    generate_finding_mask(conn, cache_dir, mask_img, "review_file", local_finding_id, StubSegmenter())

    element_rows = conn.execute(
        "SELECT element_id, ordinal FROM ai_review_alignment WHERE review_id = ? ORDER BY ordinal",
        (review_id,),
    ).fetchall()
    located_element_id = element_rows[0]["element_id"]  # ordinal 0, has a bbox
    global_element_id = element_rows[1]["element_id"]  # ordinal 1, whole-image
    absent_element_id = element_rows[2]["element_id"]  # ordinal 2, not present
    generate_alignment_mask(conn, cache_dir, mask_img, "review_file", located_element_id, StubSegmenter())

    # --- a finding whose mask_path was tampered to point outside cache_dir -
    _add_file(conn, "review_file2")
    outside_path = str(tmp_path / "outside.png")
    Image.new("L", (4, 4), 0).save(outside_path)
    review_result2 = ReviewResult(
        quality_score=3.0,
        prompt_alignment_score=None,
        summary="second",
        findings=[
            Finding(
                type="artifact",
                severity="low",
                confidence=0.5,
                localizable=True,
                description="x",
                bbox=(0.0, 0.0, 0.1, 0.1),
            )
        ],
    )
    review_id2 = store_review(
        conn,
        "review_file2",
        review_result2,
        "stub-critic",
        "stub-v1",
        RUBRIC_VERSION,
        None,
        1000.0,
        now,
    )
    outside_finding_id = conn.execute(
        "SELECT finding_id FROM ai_review_findings WHERE review_id = ?", (review_id2,)
    ).fetchone()["finding_id"]
    conn.execute(
        "UPDATE ai_review_findings SET mask_path = ? WHERE finding_id = ?",
        (outside_path, outside_finding_id),
    )
    conn.commit()

    # --- a real on-disk image for the synchronous /index endpoint ----------
    index_img_path = str(tmp_path / "index_target.png")
    Image.new("RGB", (32, 32), (7, 7, 7)).save(index_img_path)
    _add_file(conn, "index_target", path=index_img_path)

    conn.close()

    config = AIConfig(
        enabled=True,
        base_path=str(tmp_path),
        db_path=db_path,
        models_dir=str(tmp_path / "models"),
        cache_dir=cache_dir,
        ephemeral_index=True,
        face_backend="stub",
        near_dup_max_distance=8,
        similar_default_k=24,
    )
    app = Flask(__name__)
    app.register_blueprint(create_ai_blueprint(config), url_prefix="/galleryout/api/aidam")

    return SimpleNamespace(
        config=config,
        client=app.test_client(),
        global_finding_id=global_finding_id,
        local_finding_id=local_finding_id,
        outside_finding_id=outside_finding_id,
        located_element_id=located_element_id,
        global_element_id=global_element_id,
        absent_element_id=absent_element_id,
    )


_PREFIX = "/galleryout/api/aidam"


# --- /status ------------------------------------------------------------------


def test_status_reports_backends_and_counts(fixture):
    resp = fixture.client.get(f"{_PREFIX}/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["enabled"] is True
    assert data["backends"] == {
        "semantic": False,
        "visual": False,
        "face": True,
        "critic": False,
        "segmenter": False,
    }
    counts = data["counts"]
    assert counts["hashed"] == 4
    assert counts["embeddings_semantic"] == 3
    assert counts["embeddings_visual"] == 3
    assert counts["face_instances"] == 6
    assert counts["face_clusters"] == 2
    assert counts["reviews"] == 2
    assert data["worker"] == {
        "running": False,
        "stats": {},
        "provisioning": {},
        "priority_queued": 0,
        "recent_errors": [],
        "review_seconds": None,
        "stage_pace": {},
    }


# --- /duplicates ----------------------------------------------------------------


def test_duplicates_finds_exact_twin_and_near_within_distance(fixture):
    data = fixture.client.get(f"{_PREFIX}/duplicates/dup_target").get_json()
    assert data["enabled"] is True
    assert data["exact"] == ["dup_twin"]
    near_ids = {n["file_id"] for n in data["near"]}
    assert near_ids == {"dup_near"}
    assert all(n["distance"] <= 8 for n in data["near"])


def test_duplicates_respects_max_distance_param(fixture):
    data = fixture.client.get(f"{_PREFIX}/duplicates/dup_target?max_distance=2").get_json()
    assert data["near"] == []


def test_duplicates_unknown_file_is_empty(fixture):
    """Unknown ids yield empty results and are NOT 'pending': nothing will
    ever hash a file that does not exist."""
    data = fixture.client.get(f"{_PREFIX}/duplicates/does-not-exist").get_json()
    assert data == {"enabled": True, "exact": [], "near": [], "pending": False}


# --- /similar -------------------------------------------------------------------


def test_similar_semantic_and_visual_spaces_differ(fixture):
    sem = fixture.client.get(f"{_PREFIX}/similar/sim_target?space=semantic&k=5").get_json()
    vis = fixture.client.get(f"{_PREFIX}/similar/sim_target?space=visual&k=5").get_json()
    assert sem["enabled"] is True
    assert sem["space"] == "semantic"
    assert vis["space"] == "visual"
    assert sem["neighbors"][0]["file_id"] == "sim_sem_near"
    assert vis["neighbors"][0]["file_id"] == "sim_vis_near"
    assert sem["neighbors"][0]["file_id"] != vis["neighbors"][0]["file_id"]
    assert all(n["file_id"] != "sim_target" for n in sem["neighbors"])


def test_similar_invalid_space_400(fixture):
    resp = fixture.client.get(f"{_PREFIX}/similar/sim_target?space=bogus")
    assert resp.status_code == 400


def test_similar_no_embedding_returns_empty_with_note(fixture):
    data = fixture.client.get(f"{_PREFIX}/similar/dup_target?space=semantic").get_json()
    assert data["neighbors"] == []
    assert "note" in data


# --- /faces ---------------------------------------------------------------------


def test_faces_for_file_returns_two_instances(fixture):
    data = fixture.client.get(f"{_PREFIX}/faces/multi_face").get_json()
    assert data["enabled"] is True
    assert len(data["faces"]) == 2
    cluster_ids = {f["cluster_id"] for f in data["faces"]}
    assert None not in cluster_ids
    assert len(cluster_ids) == 2  # each face belongs to a different cluster


def test_faces_clusters_listing(fixture):
    data = fixture.client.get(f"{_PREFIX}/faces/clusters").get_json()
    assert data["enabled"] is True
    assert len(data["clusters"]) == 2
    for cluster in data["clusters"]:
        assert cluster["size"] == 3
        assert 1 <= len(cluster["sample_file_ids"]) <= 4


def test_faces_cluster_detail_matches_listing_size(fixture):
    clusters = fixture.client.get(f"{_PREFIX}/faces/clusters").get_json()["clusters"]
    cluster_id = clusters[0]["cluster_id"]
    data = fixture.client.get(f"{_PREFIX}/faces/clusters/{cluster_id}").get_json()
    assert data["enabled"] is True
    assert data["cluster"]["cluster_id"] == cluster_id
    assert len(data["members"]) == data["cluster"]["size"]


def test_faces_recluster_reproduces_cluster_count(fixture):
    resp = fixture.client.post(f"{_PREFIX}/faces/recluster")
    data = resp.get_json()
    assert data["enabled"] is True
    assert data["clusters"] == 2


# --- /review --------------------------------------------------------------------


def test_review_for_file_findings_mask_url_only_on_localizable(fixture):
    data = fixture.client.get(f"{_PREFIX}/review/review_file").get_json()
    assert data["enabled"] is True
    assert data["review"]["summary"] == "stub review"
    assert data["review"]["scores"]["quality"] == pytest.approx(6.0)
    findings_by_localizable = {f["localizable"]: f for f in data["findings"]}
    assert len(data["findings"]) == 2

    global_finding = findings_by_localizable[False]
    assert global_finding["bbox"] is None
    assert "mask_url" not in global_finding

    local_finding = findings_by_localizable[True]
    assert local_finding["bbox"] == pytest.approx([0.1, 0.1, 0.2, 0.2])
    assert "mask_url" in local_finding
    assert local_finding["mask_url"] == f"{_PREFIX}/review/mask/{fixture.local_finding_id}"


def test_review_for_file_missing_review_returns_none(fixture):
    """Unknown ids yield the empty review shape, not-pending and not-failed:
    no critic will ever reach a file that does not exist."""
    data = fixture.client.get(f"{_PREFIX}/review/no-such-file").get_json()
    assert data == {"enabled": True, "review": None, "findings": [], "pending": False, "scan_failed": False}


def test_review_alignment_reports_every_element_in_prompt_order(fixture):
    """Both sides of prompt-following reach the panel -- what landed AND what
    did not -- with the score as a 0..1 fraction of the elements."""
    data = fixture.client.get(f"{_PREFIX}/review/review_file").get_json()
    alignment = data["review"]["alignment"]
    assert [(e["ordinal"], e["text"], e["satisfied"]) for e in alignment] == [
        (0, "a red cube", True),
        (1, "cinematic lighting", True),
        (2, "a blue sphere", False),
    ]
    assert data["review"]["scores"]["prompt_alignment"] == pytest.approx(2 / 3)


def test_review_alignment_mask_url_only_on_located_matches(fixture):
    """A highlight needs a locus: the whole-image match and the absent
    element carry no bbox and no mask, so nothing is drawn for them."""
    alignment = {
        e["ordinal"]: e for e in fixture.client.get(f"{_PREFIX}/review/review_file").get_json()["review"]["alignment"]
    }

    located = alignment[0]
    assert located["bbox"] == pytest.approx([0.1, 0.1, 0.2, 0.2])
    assert located["mask_url"] == (f"{_PREFIX}/review/alignment/mask/{fixture.located_element_id}")

    for ordinal in (1, 2):
        assert alignment[ordinal]["bbox"] is None
        assert "mask_url" not in alignment[ordinal]


def test_review_alignment_mask_serves_png(fixture):
    resp = fixture.client.get(f"{_PREFIX}/review/alignment/mask/{fixture.located_element_id}")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"


def test_review_alignment_mask_404s_without_a_mask(fixture):
    """An unlocalized match and an absent element have no mask to serve."""
    for element_id in (fixture.global_element_id, fixture.absent_element_id):
        assert fixture.client.get(f"{_PREFIX}/review/alignment/mask/{element_id}").status_code == 404
    assert fixture.client.get(f"{_PREFIX}/review/alignment/mask/999999").status_code == 404


def test_review_mask_serves_png(fixture):
    resp = fixture.client.get(f"{_PREFIX}/review/mask/{fixture.local_finding_id}")
    assert resp.status_code == 200
    assert resp.mimetype == "image/png"


def test_review_mask_404_for_non_localizable_finding(fixture):
    resp = fixture.client.get(f"{_PREFIX}/review/mask/{fixture.global_finding_id}")
    assert resp.status_code == 404


def test_review_mask_404_for_unknown_finding(fixture):
    resp = fixture.client.get(f"{_PREFIX}/review/mask/999999")
    assert resp.status_code == 404


def test_review_mask_rejects_path_outside_cache_dir(fixture):
    # This finding's mask_path was tampered to point outside cache_dir; even
    # though the file genuinely exists on disk, it must never be served.
    resp = fixture.client.get(f"{_PREFIX}/review/mask/{fixture.outside_finding_id}")
    assert resp.status_code == 404


# --- /review/feedback -------------------------------------------------------------


def test_feedback_post_and_export_round_trip(fixture):
    resp = fixture.client.post(
        f"{_PREFIX}/review/feedback",
        json={
            "target_kind": "finding",
            "target_id": str(fixture.local_finding_id),
            "verdict": "accept",
            "file_id": "review_file",
            "note": "looks right",
        },
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["enabled"] is True
    feedback_id = body["feedback_id"]

    export_resp = fixture.client.get(f"{_PREFIX}/review/feedback/export")
    assert export_resp.status_code == 200
    rows = [json.loads(line) for line in export_resp.get_data(as_text=True).splitlines() if line]
    row = next(r for r in rows if r["feedback_id"] == feedback_id)
    assert row["verdict"] == "accept"
    assert row["target_kind"] == "finding"
    assert row["exported_at"] is not None


def test_feedback_post_invalid_verdict_returns_400(fixture):
    resp = fixture.client.post(
        f"{_PREFIX}/review/feedback",
        json={
            "target_kind": "finding",
            "target_id": "1",
            "verdict": "not-a-real-verdict",
        },
    )
    assert resp.status_code == 400


# --- /index ---------------------------------------------------------------------


def test_index_file_computes_hash_synchronously(fixture):
    resp = fixture.client.post(f"{_PREFIX}/index/index_target", json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["enabled"] is True
    assert data["hashed"] is True

    status = fixture.client.get(f"{_PREFIX}/status").get_json()
    assert status["counts"]["hashed"] == 5


def test_index_unknown_file_404(fixture):
    resp = fixture.client.post(f"{_PREFIX}/index/does-not-exist", json={})
    assert resp.status_code == 404


# --- feature-disabled short-circuit (within an otherwise-enabled config) --------


def test_disabled_blueprint_short_circuits_every_route(tmp_path):
    db_path = str(tmp_path / "disabled.sqlite")
    conn = _make_conn(db_path)
    conn.close()
    config = AIConfig(enabled=False, base_path=str(tmp_path), db_path=db_path, cache_dir=str(tmp_path / "cache"))
    app = Flask(__name__)
    app.register_blueprint(create_ai_blueprint(config), url_prefix=_PREFIX)
    client = app.test_client()

    # /status always reports, even when disabled.
    status = client.get(f"{_PREFIX}/status").get_json()
    assert status["enabled"] is False
    assert "backends" in status
    assert "counts" in status

    for method, path in [
        ("get", "/duplicates/f1"),
        ("get", "/similar/f1"),
        ("get", "/faces/f1"),
        ("get", "/faces/clusters"),
        ("get", "/faces/clusters/1"),
        ("post", "/faces/recluster"),
        ("get", "/review/f1"),
        ("get", "/review/mask/1"),
        ("post", "/review/feedback"),
        ("get", "/review/feedback/export"),
        ("post", "/index/f1"),
    ]:
        resp = getattr(client, method)(f"{_PREFIX}{path}")
        assert resp.status_code == 200
        assert resp.get_json() == {"enabled": False}


# --- create_ai_resolvers() against the real OmniQuery engine --------------------


def test_ai_resolvers_end_to_end_with_omniquery_engine(fixture):
    resolvers = create_ai_resolvers(fixture.config)
    engine = OmniQueryEngine(
        db_path=fixture.config.db_path,
        base_path=fixture.config.base_path,
        ai_resolvers=resolvers,
    )
    ctx = AuthContext(role="STAFF", user_id="3", client_uuid="3", ai_enabled=True)

    semantic_outcome = engine.run(
        {
            "where": {"field": "similar_to_semantic", "op": "eq", "value": {"file_id": "sim_target", "k": 5}},
            "limit": 50,
        },
        ctx,
    )
    assert semantic_outcome.ok
    assert "sim_sem_near" in semantic_outcome.ids
    assert "sim_target" not in semantic_outcome.ids

    near_dup_outcome = engine.run(
        {"where": {"field": "near_dup_of", "op": "eq", "value": "dup_target"}, "limit": 50},
        ctx,
    )
    assert near_dup_outcome.ok
    assert set(near_dup_outcome.ids) == {"dup_near"}

    unknown_file_outcome = engine.run(
        {"where": {"field": "similar_to_visual", "op": "eq", "value": "no-such-file"}, "limit": 50},
        ctx,
    )
    assert unknown_file_outcome.ok
    assert unknown_file_outcome.ids == []


# --- per-file authorization (file_access_check) -------------------------------


def test_file_access_check_scopes_per_file_routes(tmp_path):
    """Every per-file read route (including mask fetches) must 404 for a
    file the host's access policy denies, and still serve visible files."""
    db_path = str(tmp_path / "authz.sqlite")
    cache_dir = str(tmp_path / "cache")
    conn = _make_conn(db_path)

    _add_file(conn, "vis")
    _add_file(conn, "hid")
    for fid in ("vis", "hid"):
        result = ReviewResult(
            quality_score=5.0,
            prompt_alignment_score=None,
            summary="s",
            findings=[
                Finding(
                    type="artifact",
                    severity="low",
                    confidence=0.9,
                    localizable=True,
                    description="spot",
                    bbox=(0.25, 0.25, 0.5, 0.5),
                )
            ],
        )
        store_review(conn, fid, result, "critic-x", "v1", RUBRIC_VERSION, None, 1000.0, 2000.0)
    hid_finding = conn.execute("SELECT finding_id FROM ai_review_findings WHERE file_id='hid'").fetchone()[0]
    img = Image.new("RGB", (32, 32), (40, 40, 40))
    generate_finding_mask(conn, cache_dir, img, "hid", hid_finding, StubSegmenter())
    conn.close()

    config = AIConfig(enabled=True, base_path=str(tmp_path), db_path=db_path, cache_dir=cache_dir, ephemeral_index=True)
    app = Flask(__name__)
    app.register_blueprint(create_ai_blueprint(config, file_access_check=lambda fid: fid == "vis"), url_prefix=_PREFIX)
    client = app.test_client()

    assert client.get(f"{_PREFIX}/review/vis").status_code == 200
    for path in ("/review/hid", "/duplicates/hid", "/similar/hid", "/faces/hid", f"/review/mask/{hid_finding}"):
        assert client.get(f"{_PREFIX}{path}").status_code == 404, path


def test_file_access_check_filters_returned_neighbors(tmp_path):
    """The visibility policy applies to every RETURNED file id: a visible
    anchor must not reveal hidden exact/near duplicates or hidden vector
    neighbors."""
    db_path = str(tmp_path / "authz2.sqlite")
    cache_dir = str(tmp_path / "cache")
    conn = _make_conn(db_path)
    now = time.time()

    for fid in ("vis_anchor", "vis_twin", "hid_twin", "hid_near"):
        _add_file(conn, fid)
    shared_sha = "c" * 64
    hashing.upsert_hashes(
        conn, "vis_anchor", hashing.HashResult(sha256=shared_sha, phash64=0, dhash64=0), 1000.0, HASH_ALGO_VERSION, now
    )
    hashing.upsert_hashes(
        conn, "vis_twin", hashing.HashResult(sha256=shared_sha, phash64=0, dhash64=0), 1000.0, HASH_ALGO_VERSION, now
    )
    hashing.upsert_hashes(
        conn, "hid_twin", hashing.HashResult(sha256=shared_sha, phash64=0, dhash64=0), 1000.0, HASH_ALGO_VERSION, now
    )
    hashing.upsert_hashes(
        conn, "hid_near", hashing.HashResult(sha256="d" * 64, phash64=0b11, dhash64=0), 1000.0, HASH_ALGO_VERSION, now
    )

    store = vectors.VectorStore(cache_dir=cache_dir, ephemeral=True)
    base = np.ones(8, dtype=np.float32)
    for i, fid in enumerate(("vis_anchor", "vis_twin", "hid_twin")):
        store.add(conn, fid, SPACE_SEMANTIC, "m", "v1", _tight_vector(base, seed=i), 1000.0)

    conn.close()

    config = AIConfig(enabled=True, base_path=str(tmp_path), db_path=db_path, cache_dir=cache_dir, ephemeral_index=True)
    app = Flask(__name__)
    app.register_blueprint(
        create_ai_blueprint(config, file_access_check=lambda fid: fid.startswith("vis")), url_prefix=_PREFIX
    )
    client = app.test_client()

    dup = client.get(f"{_PREFIX}/duplicates/vis_anchor").get_json()
    assert dup["exact"] == ["vis_twin"]
    assert all(n["file_id"].startswith("vis") for n in dup["near"])

    sim = client.get(f"{_PREFIX}/similar/vis_anchor?space=semantic").get_json()
    neighbor_ids = [n["file_id"] for n in sim["neighbors"]]
    assert "vis_twin" in neighbor_ids
    assert not any(fid.startswith("hid") for fid in neighbor_ids)


def test_review_reports_prompt_availability(fixture):
    """The review response says whether the file carries a generation
    prompt: a null alignment score with prompt_available=False means
    'nothing to align against', not 'scoring failed'."""
    before = fixture.client.get(f"{_PREFIX}/review/review_file").get_json()
    assert before["prompt_available"] is False

    conn = sqlite3.connect(fixture.config.db_path)
    conn.execute("UPDATE files SET workflow_prompt = 'a red fox in snow' WHERE id = 'review_file'")
    conn.commit()
    conn.close()

    after = fixture.client.get(f"{_PREFIX}/review/review_file").get_json()
    assert after["prompt_available"] is True


# --- /faces/compare/<file_id> -------------------------------------------------


def test_faces_compare_reports_lanes_and_inventory(fixture, monkeypatch):
    """The endpoint renders the file and answers every lane plus the
    installed-pipeline inventory; a lane that cannot load reports its
    error in place instead of failing the request."""

    def fake_compare(img, config, registry):
        assert img.size == (32, 32)
        assert registry is backends  # the route passes the process registry in
        return {
            "lanes": {
                "yunet": {"model": "opencv/yunet+sface (v)", "elapsed_ms": 1.2, "faces": []},
                "scrfd": {"model": "scrfd", "error": "antelopev2 pack not found"},
            },
            "installed": [{"name": "yunet+sface", "weights_present": False, "active": False}],
        }

    monkeypatch.setattr(F, "compare_detectors", fake_compare)
    res = fixture.client.get(f"{_PREFIX}/faces/compare/index_target")
    assert res.status_code == 200
    data = res.get_json()
    assert data["lanes"]["yunet"]["faces"] == []
    assert "error" in data["lanes"]["scrfd"]
    assert data["installed"][0]["name"] == "yunet+sface"


def test_faces_compare_unknown_file_404(fixture):
    assert fixture.client.get(f"{_PREFIX}/faces/compare/nope").status_code == 404


def test_faces_compare_nonrenderable_422(fixture):
    """A file whose path cannot be rendered answers 422, not a crash."""
    res = fixture.client.get(f"{_PREFIX}/faces/compare/dup_target")
    assert res.status_code == 422
    assert "renderable" in res.get_json()["error"]


# --- /faces/recent + cluster attribute aggregates ------------------------------


def test_faces_recent_lists_scanned_files(fixture):
    """Recent-faces picker: files with face rows, newest scan first."""
    res = fixture.client.get(f"{_PREFIX}/faces/recent")
    assert res.status_code == 200
    data = res.get_json()
    assert data["enabled"] is True
    assert isinstance(data["files"], list)
    for entry in data["files"]:
        assert set(entry) == {"file_id", "faces"}
        assert entry["faces"] >= 1


def test_faces_cluster_detail_reports_attribute_aggregates(fixture):
    """Cluster detail carries typed attribute aggregates and per-member
    age/sex/pose so the dashboard can describe the bucket."""
    clusters = fixture.client.get(f"{_PREFIX}/faces/clusters").get_json()["clusters"]
    if not clusters:
        pytest.skip("fixture has no clusters")
    detail = fixture.client.get(f"{_PREFIX}/faces/clusters/{clusters[0]['cluster_id']}").get_json()
    assert "attributes" in detail
    agg = detail["attributes"]
    assert {"with_age", "age_min", "age_max", "age_avg", "male", "female", "yaw_abs_avg"} <= set(agg)
    for member in detail["members"]:
        assert {"age", "sex", "pose_yaw"} <= set(member)
