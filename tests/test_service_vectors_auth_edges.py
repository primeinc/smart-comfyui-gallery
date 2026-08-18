"""Edge-contract tests for smartgallery_ai.service, smartgallery_ai.vectors,
and sg_auth: worker registration surfaced via /status, resolver degradation,
fail-closed mask serving, /index force semantics, VectorStore connection
factories / cache-corruption fallback, and password-hash failure contracts.
"""

from __future__ import annotations

import os
import sqlite3
import time
from types import SimpleNamespace

import numpy as np
import pytest
from argon2 import PasswordHasher
from flask import Flask
from PIL import Image

import sg_auth
from smartgallery_ai import RUBRIC_VERSION, SPACE_SEMANTIC, SPACE_VISUAL, AIConfig
from smartgallery_ai import vectors as vectors_mod
from smartgallery_ai.review import Finding, ReviewResult, store_review
from smartgallery_ai.schema import init_schema
from smartgallery_ai.service import (
    create_ai_blueprint,
    create_ai_resolvers,
    get_worker,
    set_worker,
)
from smartgallery_ai.vectors import VectorStore

_PREFIX = "/galleryout/api/aidam"


# --- fixture helpers ----------------------------------------------------------


def _make_conn(db_path: str = ":memory:") -> sqlite3.Connection:
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


def _make_config(tmp_path, **overrides) -> AIConfig:
    """Enabled config with every backend off unless a test opts in, so no
    'auto' probe can wander toward a real model runtime."""
    defaults = {
        "enabled": True,
        "base_path": str(tmp_path),
        "db_path": str(tmp_path / "gallery.sqlite"),
        "models_dir": str(tmp_path / "models"),
        "cache_dir": str(tmp_path / "cache"),
        "ephemeral_index": True,
        "semantic_backend": "none",
        "visual_backend": "none",
        "face_backend": "none",
        "critic_backend": "none",
        "segmenter_backend": "none",
    }
    defaults.update(overrides)
    return AIConfig(**defaults)


def _client(config: AIConfig):
    app = Flask(__name__)
    app.register_blueprint(create_ai_blueprint(config), url_prefix=_PREFIX)
    return app.test_client()


# --- service: /status worker registration ------------------------------------


def test_status_reports_registered_worker_and_reset(tmp_path):
    """/status mirrors the worker registered via set_worker (running flag and
    stats) and reverts to not-running once the worker is unregistered."""
    _make_conn(str(tmp_path / "gallery.sqlite")).close()
    client = _client(_make_config(tmp_path))
    fake_worker = SimpleNamespace(is_running=True, stats={"scanned": 3})

    set_worker(fake_worker)
    try:
        assert get_worker() is fake_worker
        data = client.get(f"{_PREFIX}/status").get_json()
        assert data["worker"] == {
            "running": True,
            "stats": {"scanned": 3},
            "provisioning": {},
            "priority_queued": 0,
            "recent_errors": [],
            "review_seconds": None,
            "stage_pace": {},
        }
    finally:
        set_worker(None)

    data = client.get(f"{_PREFIX}/status").get_json()
    assert data["worker"] == {
        "running": False,
        "stats": {},
        "provisioning": {},
        "priority_queued": 0,
        "recent_errors": [],
        "review_seconds": None,
        "stage_pace": {},
    }


def test_status_invalid_segmenter_selector_degrades_to_unavailable(tmp_path):
    """A bogus segmenter_backend name makes /status report the segmenter as
    unavailable instead of erroring (the availability probe never raises)."""
    _make_conn(str(tmp_path / "gallery.sqlite")).close()
    client = _client(_make_config(tmp_path, segmenter_backend="not-a-backend"))

    resp = client.get(f"{_PREFIX}/status")

    assert resp.status_code == 200
    assert resp.get_json()["backends"]["segmenter"] is False


# --- service: resolver degradation --------------------------------------------


@pytest.mark.parametrize("field", ["near_dup_of", "similar_to_semantic", "similar_to_visual"])
@pytest.mark.parametrize(
    "value", [None, "", {}, {"file_id": None}, 123], ids=["none", "empty", "dict", "null-id", "int"]
)
def test_resolvers_return_empty_for_unresolvable_values(tmp_path, field, value):
    """Every AI resolver answers [] (never an error) for a value carrying no
    usable file_id — None, empty string, id-less dicts, or a non-string."""
    _make_conn(str(tmp_path / "gallery.sqlite")).close()
    resolvers = create_ai_resolvers(_make_config(tmp_path))

    assert resolvers[field](value) == []


# --- service: face cluster / recluster edges ----------------------------------


def test_faces_cluster_detail_unknown_cluster_404(tmp_path):
    """An unknown cluster id answers 404 with an explicit empty body, not an
    empty 200."""
    _make_conn(str(tmp_path / "gallery.sqlite")).close()
    client = _client(_make_config(tmp_path))

    resp = client.get(f"{_PREFIX}/faces/clusters/424242")

    assert resp.status_code == 404
    assert resp.get_json() == {"enabled": True, "cluster": None, "members": []}


def test_faces_recluster_without_backend_notes_absence(tmp_path):
    """POST /faces/recluster with no face backend reports zero clusters plus
    an explanatory note instead of failing."""
    _make_conn(str(tmp_path / "gallery.sqlite")).close()
    client = _client(_make_config(tmp_path, face_backend="none"))

    resp = client.post(f"{_PREFIX}/faces/recluster")

    assert resp.status_code == 200
    assert resp.get_json() == {
        "enabled": True,
        "clusters": 0,
        "note": "no face backend configured",
    }


# --- service: review findings with points / mask fail-closed paths -----------


def test_review_points_only_finding_serializes_points(tmp_path):
    """A localizable finding grounded by points (no bbox) serializes its
    points list in /review/<file_id> with bbox null and no mask_url."""
    conn = _make_conn(str(tmp_path / "gallery.sqlite"))
    _add_file(conn, "pf")
    result = ReviewResult(
        quality_score=7.0,
        prompt_alignment_score=None,
        summary="pts",
        findings=[
            Finding(
                type="anatomy",
                severity="low",
                confidence=0.6,
                localizable=True,
                description="extra finger",
                points=[(0.1, 0.2), (0.3, 0.4)],
            )
        ],
    )
    store_review(conn, "pf", result, "critic-x", "v1", RUBRIC_VERSION, None, 1000.0, time.time())
    conn.close()
    client = _client(_make_config(tmp_path))

    data = client.get(f"{_PREFIX}/review/pf").get_json()

    (finding,) = data["findings"]
    assert finding["points"] == [[0.1, 0.2], [0.3, 0.4]]
    assert finding["bbox"] is None
    assert "mask_url" not in finding


def _store_single_localizable_finding(conn, file_id: str) -> int:
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
    review_id = store_review(conn, file_id, result, "critic-x", "v1", RUBRIC_VERSION, None, 1000.0, time.time())
    return conn.execute("SELECT finding_id FROM ai_review_findings WHERE review_id = ?", (review_id,)).fetchone()[0]


def test_review_mask_404_before_mask_generation(tmp_path):
    """A localizable finding whose mask has not been generated yet
    (mask_path NULL) answers 404 on the mask route."""
    conn = _make_conn(str(tmp_path / "gallery.sqlite"))
    _add_file(conn, "pending")
    finding_id = _store_single_localizable_finding(conn, "pending")
    assert (
        conn.execute("SELECT mask_path FROM ai_review_findings WHERE finding_id = ?", (finding_id,)).fetchone()[0]
        is None
    )
    conn.close()
    client = _client(_make_config(tmp_path))

    resp = client.get(f"{_PREFIX}/review/mask/{finding_id}")

    assert resp.status_code == 404


def test_review_mask_404_when_mask_file_missing_on_disk(tmp_path):
    """A mask_path correctly contained under cache_dir/masks but pointing at
    a file that no longer exists answers 404 (fail closed, no 500)."""
    conn = _make_conn(str(tmp_path / "gallery.sqlite"))
    _add_file(conn, "ghost")
    finding_id = _store_single_localizable_finding(conn, "ghost")
    config = _make_config(tmp_path)
    missing = os.path.join(config.cache_dir, "masks", "ghost", "gone.png")
    conn.execute("UPDATE ai_review_findings SET mask_path = ? WHERE finding_id = ?", (missing, finding_id))
    conn.commit()
    conn.close()
    client = _client(config)

    resp = client.get(f"{_PREFIX}/review/mask/{finding_id}")

    assert resp.status_code == 404


# --- service: /index force + stub-backend embedding ---------------------------


def _seed_indexable_image(conn, tmp_path, file_id: str) -> str:
    img_path = str(tmp_path / f"{file_id}.png")
    Image.new("RGB", (32, 32), (7, 7, 7)).save(img_path)
    _add_file(conn, file_id, path=img_path)
    return img_path


def test_index_force_clears_only_review_scan_log_entry(tmp_path):
    """POST /index with force=true reports review_rescheduled and deletes the
    file's 'review' scan-log row while leaving other kinds untouched."""
    db_path = str(tmp_path / "gallery.sqlite")
    conn = _make_conn(db_path)
    _seed_indexable_image(conn, tmp_path, "forced")
    for kind in ("review", "faces"):
        conn.execute(
            "INSERT INTO ai_scan_log (file_id, kind, model_id, model_version, "
            "source_mtime, scanned_at, result_count) VALUES (?, ?, 'm', 'v1', 1000.0, 2000.0, 1)",
            ("forced", kind),
        )
    conn.commit()
    conn.close()
    client = _client(_make_config(tmp_path))

    resp = client.post(f"{_PREFIX}/index/forced", json={"force": True})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["review_rescheduled"] is True
    assert data["hashed"] is True
    check = sqlite3.connect(db_path)
    kinds = [r[0] for r in check.execute("SELECT kind FROM ai_scan_log WHERE file_id = 'forced'")]
    check.close()
    assert kinds == ["faces"]


def test_index_with_stub_backends_embeds_both_spaces(tmp_path):
    """POST /index with stub embedders writes one embedding row per space
    (dim 64, stub model ids) and reports them plus the face pass."""
    db_path = str(tmp_path / "gallery.sqlite")
    conn = _make_conn(db_path)
    _seed_indexable_image(conn, tmp_path, "embed_me")
    conn.close()
    config = _make_config(tmp_path, semantic_backend="stub", visual_backend="stub", face_backend="stub")
    client = _client(config)

    resp = client.post(f"{_PREFIX}/index/embed_me", json={})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["embedded"] == [SPACE_SEMANTIC, SPACE_VISUAL]
    assert data["faces"] is True
    assert "review_rescheduled" not in data  # force not requested
    check = sqlite3.connect(db_path)
    rows = check.execute(
        "SELECT space, model_id, dim FROM ai_embeddings WHERE file_id = 'embed_me' ORDER BY space"
    ).fetchall()
    check.close()
    assert rows == [(SPACE_SEMANTIC, "stub-semantic", 64), (SPACE_VISUAL, "stub-visual", 64)]


def test_index_skips_fresh_embeddings_on_reindex(tmp_path):
    """Re-POSTing /index without force skips hash and embedding work that is
    already up to date for the same mtime and model version."""
    db_path = str(tmp_path / "gallery.sqlite")
    conn = _make_conn(db_path)
    _seed_indexable_image(conn, tmp_path, "twice")
    conn.close()
    config = _make_config(tmp_path, semantic_backend="stub", visual_backend="stub")
    client = _client(config)
    first = client.post(f"{_PREFIX}/index/twice", json={}).get_json()
    assert first["hashed"] is True
    assert first["embedded"] == [SPACE_SEMANTIC, SPACE_VISUAL]

    second = client.post(f"{_PREFIX}/index/twice", json={}).get_json()

    assert second["hashed"] is False
    assert second["embedded"] == []


# --- vectors: connection factories --------------------------------------------


def test_vectorstore_db_path_string_factory_round_trip(tmp_path):
    """VectorStore(db=<path>) opens its own connections: add()/topk() with
    conn=None persist to and read from that SQLite file."""
    db_path = str(tmp_path / "vec.sqlite")
    conn = _make_conn(db_path)
    _add_file(conn, "f1")
    _add_file(conn, "f2")
    conn.close()
    store = VectorStore(db=db_path, ephemeral=True)
    store.add(None, "f1", "semantic", "m", "v1", np.array([1.0, 0.0], dtype=np.float32), 1000.0)
    store.add(None, "f2", "semantic", "m", "v1", np.array([0.0, 1.0], dtype=np.float32), 1000.0)

    got = store.topk(None, "semantic", np.array([1.0, 0.1], dtype=np.float32), k=2)

    assert [fid for fid, _ in got] == ["f1", "f2"]
    check = sqlite3.connect(db_path)
    count = check.execute("SELECT COUNT(*) FROM ai_embeddings").fetchone()[0]
    check.close()
    assert count == 2


def test_vectorstore_callable_factory_is_used(tmp_path):
    """VectorStore(db=<zero-arg factory>) resolves every conn=None call
    through the factory."""
    db_path = str(tmp_path / "vec.sqlite")
    conn = _make_conn(db_path)
    _add_file(conn, "f1")
    conn.close()
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return sqlite3.connect(db_path)

    store = VectorStore(db=factory, ephemeral=True)
    store.add(None, "f1", "semantic", "m", "v1", np.array([1.0, 0.0], dtype=np.float32), 1000.0)

    got = store.topk(None, "semantic", np.array([1.0, 0.0], dtype=np.float32), k=1)

    assert got == [("f1", pytest.approx(1.0))]
    assert calls["n"] >= 2  # one connection per add/topk call


def test_vectorstore_without_db_requires_explicit_connection():
    """With no db configured, conn=None raises ValueError naming the missing
    factory for both reads and writes."""
    store = VectorStore(ephemeral=True)

    with pytest.raises(ValueError, match="no connection provided and no db factory configured"):
        store.topk(None, "semantic", np.ones(4, dtype=np.float32), k=1)
    with pytest.raises(ValueError, match="no connection provided and no db factory configured"):
        store.add(None, "f1", "semantic", "m", "v1", np.ones(4, dtype=np.float32), 1000.0)


# --- vectors: add()/topk() dimension contracts ---------------------------------


def test_add_dim_mismatch_error_names_both_dims():
    """The dim-mismatch error identifies the space, model_version, and both
    conflicting dimensions."""
    conn = _make_conn()
    _add_file(conn, "f1")
    _add_file(conn, "f2")
    store = VectorStore(ephemeral=True)
    store.add(conn, "f1", "semantic", "m", "v1", np.zeros(8, dtype=np.float32), 1000.0)

    with pytest.raises(
        ValueError,
        match=r"dim mismatch for space='semantic' model_version='v1': existing dim=8, new dim=16",
    ):
        store.add(conn, "f2", "semantic", "m", "v1", np.zeros(16, dtype=np.float32), 1000.0)


def test_add_allows_dim_change_across_model_versions():
    """Different model_versions within one space may carry different dims
    (mid-migration), and topk pinned to each version sees only its rows."""
    conn = _make_conn()
    _add_file(conn, "old")
    _add_file(conn, "new")
    store = VectorStore(ephemeral=True)
    store.add(conn, "old", "semantic", "m", "v1", np.ones(8, dtype=np.float32), 1000.0)

    store.add(conn, "new", "semantic", "m", "v2", np.ones(16, dtype=np.float32), 1000.0)

    pinned_v1 = store.topk(conn, "semantic", np.ones(8, dtype=np.float32), k=5, model_version="v1")
    pinned_v2 = store.topk(conn, "semantic", np.ones(16, dtype=np.float32), k=5, model_version="v2")
    assert [fid for fid, _ in pinned_v1] == ["old"]
    assert [fid for fid, _ in pinned_v2] == ["new"]


def test_topk_zero_norm_query_returns_empty():
    """An all-zero query vector yields [] instead of NaN scores or an error."""
    conn = _make_conn()
    _add_file(conn, "f1")
    store = VectorStore(ephemeral=True)
    store.add(conn, "f1", "semantic", "m", "v1", np.array([1.0, 2.0], dtype=np.float32), 1000.0)

    assert store.topk(conn, "semantic", np.zeros(2, dtype=np.float32), k=3) == []


def test_topk_query_dim_mismatch_raises_naming_dims():
    """A query whose dim differs from the space's matrix raises ValueError
    naming both the query dim and the space dim."""
    conn = _make_conn()
    _add_file(conn, "f1")
    store = VectorStore(ephemeral=True)
    store.add(conn, "f1", "semantic", "m", "v1", np.zeros(8, dtype=np.float32), 1000.0)

    with pytest.raises(ValueError, match=r"query dim 4 does not match space 'semantic' dim 8"):
        store.topk(conn, "semantic", np.ones(4, dtype=np.float32), k=1)


# --- vectors: disk-cache corruption and invalidation ---------------------------


def test_corrupt_disk_cache_falls_back_to_sqlite_and_repairs(tmp_path):
    """Garbage bytes in the .npz mirror are ignored: a fresh store rebuilds
    from SQLite, returns identical results, and rewrites a valid cache."""
    conn = _make_conn()
    file_ids = [f"f{i}" for i in range(4)]
    for fid in file_ids:
        _add_file(conn, fid)
    store1 = VectorStore(cache_dir=str(tmp_path), ephemeral=False)
    rng = np.random.default_rng(42)
    for fid in file_ids:
        store1.add(conn, fid, "semantic", "m", "v1", rng.standard_normal(6), 1000.0)
    query = rng.standard_normal(6).astype(np.float32)
    expected = store1.topk(conn, "semantic", query, k=4)
    cache_file = tmp_path / "vectors" / "semantic__v1.npz"
    assert cache_file.exists()
    cache_file.write_bytes(b"this is definitely not a zip archive")

    # Simulate a fresh process: generations are process-global, so a new
    # store instance alone no longer implies a cold cache.

    with vectors_mod._GEN_LOCK:
        vectors_mod._GENERATIONS.clear()

    store2 = VectorStore(cache_dir=str(tmp_path), ephemeral=False)
    got = store2.topk(conn, "semantic", query, k=4)

    assert got == expected
    with np.load(cache_file) as repaired:  # cache was re-mirrored, valid again
        assert int(repaired["row_count"]) == 4


def test_invalidate_removes_only_matching_space_cache_files(tmp_path):
    """invalidate('semantic') deletes that space's .npz mirror and leaves
    other spaces' mirrors on disk."""
    conn = _make_conn()
    _add_file(conn, "f1")
    store = VectorStore(cache_dir=str(tmp_path), ephemeral=False)
    store.add(conn, "f1", "semantic", "m", "v1", np.array([1.0, 0.0], dtype=np.float32), 1000.0)
    store.add(conn, "f1", "visual", "m", "v1", np.array([0.0, 1.0], dtype=np.float32), 1000.0)
    query = np.array([1.0, 0.0], dtype=np.float32)
    store.topk(conn, "semantic", query, k=1)
    store.topk(conn, "visual", query, k=1)
    sem_cache = tmp_path / "vectors" / "semantic__v1.npz"
    vis_cache = tmp_path / "vectors" / "visual__v1.npz"
    assert sem_cache.exists()
    assert vis_cache.exists()

    store.invalidate("semantic")

    assert not sem_cache.exists()
    assert vis_cache.exists()


def test_invalidate_before_any_cache_leaves_filesystem_untouched(tmp_path):
    """invalidate() on a store that never wrote a cache neither raises nor
    creates the vectors directory."""
    cache_dir = tmp_path / "never_written"
    store = VectorStore(cache_dir=str(cache_dir), ephemeral=False)

    store.invalidate("semantic")

    assert not cache_dir.exists()


# --- sg_auth: verify_password failure and rehash contracts ---------------------


def test_verify_password_malformed_argon2_prefixed_hashes_fail_closed():
    """$argon2-prefixed strings that are not well-formed hashes fail
    verification with (False, False) instead of raising."""
    malformed = [
        "$argon2",
        "$argon2id$",
        "$argon2id$v=19$m=65536,t=3,p=4$brokensalt",
        sg_auth.hash_password("real-password")[:-15],  # truncated real hash
    ]
    for stored in malformed:
        assert sg_auth.verify_password(stored, "real-password") == (False, False)


def test_verify_password_reports_needs_rehash_for_outdated_parameters():
    """A valid hash produced with parameters differing from the module's
    hasher verifies as (True, True), signaling a transparent rehash."""
    outdated = PasswordHasher(time_cost=sg_auth.ph.time_cost + 1).hash("pw-123")

    assert sg_auth.verify_password(outdated, "pw-123") == (True, True)


def test_constant_time_equals_bytes_and_hostile_objects():
    """constant_time_equals compares bytes operands directly, normalizes
    str-vs-bytes, and returns False for objects whose str() raises."""

    class Unstringable:
        def __str__(self):
            raise RuntimeError("no string for you")

    assert sg_auth.constant_time_equals(b"abc", b"abc") is True
    assert sg_auth.constant_time_equals(b"abc", "abc") is True
    assert sg_auth.constant_time_equals(b"abc", b"abd") is False
    assert sg_auth.constant_time_equals(b"abc", b"abcd") is False
    assert sg_auth.constant_time_equals(Unstringable(), "x") is False
    assert sg_auth.constant_time_equals("x", Unstringable()) is False


# --- sg_auth: migration key-file failure path ----------------------------------


def test_migrate_legacy_passwords_unreadable_key_path_fails_closed(tmp_path):
    """A key path that exists but cannot be read as a file (a directory)
    marks every legacy row unusable and never reports the key deleted."""
    conn = sqlite3.connect(str(tmp_path / "m.sqlite"))
    conn.execute("CREATE TABLE users (user_id INTEGER PRIMARY KEY, password TEXT)")
    conn.execute("INSERT INTO users (password) VALUES ('gAAAAlegacy-one')")
    conn.execute("INSERT INTO users (password) VALUES ('gAAAAlegacy-two')")
    conn.commit()
    key_dir = tmp_path / "keydir"
    key_dir.mkdir()

    report = sg_auth.migrate_legacy_passwords(conn, str(key_dir))

    assert report == {"migrated": 0, "failed": 2, "skipped": 0, "key_deleted": False}
    passwords = [r[0] for r in conn.execute("SELECT password FROM users")]
    assert passwords == [sg_auth.LOGIN_DISABLED, sg_auth.LOGIN_DISABLED]
    assert key_dir.exists()  # os.remove on the directory failed, key retained
