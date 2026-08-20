"""Tests for smartgallery_ai.vectors.VectorStore: cosine top-k, space
independence, on-disk cache persistence/staleness, and ephemeral mode."""

import os
import sqlite3
import sys

import numpy as np
import pytest

from smartgallery_ai import vectors as V
from smartgallery_ai import vectors as vectors_mod
from smartgallery_ai.schema import init_schema
from smartgallery_ai.vectors import VectorStore

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


def numpy_reference_topk(vectors: dict, query: np.ndarray, k: int, exclude=()):
    """Brute-force cosine top-k, independent of VectorStore, for comparison."""
    q = query / np.linalg.norm(query)
    scored = []
    for fid, vec in vectors.items():
        if fid in exclude:
            continue
        v = vec / np.linalg.norm(vec)
        scored.append((fid, float(np.dot(v, q))))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored[:k]


# --- cosine correctness -------------------------------------------------------


def test_topk_matches_numpy_reference(tmp_path):
    conn = make_conn()
    file_ids = [f"f{i}" for i in range(6)]
    add_files(conn, file_ids)
    store = VectorStore(cache_dir=str(tmp_path), ephemeral=False)

    rng = np.random.default_rng(7)
    vectors = {fid: rng.standard_normal(16).astype(np.float32) for fid in file_ids}
    for fid, vec in vectors.items():
        store.add(conn, fid, "semantic", "modelA", "v1", vec, source_mtime=1000.0)

    query = rng.standard_normal(16).astype(np.float32)
    got = store.topk(conn, "semantic", query, k=4)
    expected = numpy_reference_topk(vectors, query, k=4)

    assert [fid for fid, _ in got] == [fid for fid, _ in expected]
    for (_, got_sim), (_, exp_sim) in zip(got, expected, strict=False):
        assert got_sim == pytest.approx(exp_sim, abs=1e-5)


def test_topk_respects_k_and_exclude(tmp_path):
    conn = make_conn()
    file_ids = [f"f{i}" for i in range(6)]
    add_files(conn, file_ids)
    store = VectorStore(cache_dir=str(tmp_path), ephemeral=False)

    rng = np.random.default_rng(11)
    for fid in file_ids:
        store.add(conn, fid, "semantic", "modelA", "v1", rng.standard_normal(8), 1000.0)

    full = store.topk(conn, "semantic", np.ones(8, dtype=np.float32), k=6)
    assert len(full) == 6

    limited = store.topk(conn, "semantic", np.ones(8, dtype=np.float32), k=2)
    assert len(limited) == 2
    assert limited == full[:2]

    excluded = store.topk(conn, "semantic", np.ones(8, dtype=np.float32), k=6, exclude=("f0", "f1"))
    assert "f0" not in [fid for fid, _ in excluded]
    assert "f1" not in [fid for fid, _ in excluded]
    assert len(excluded) == 4


def test_topk_deterministic_tie_order(tmp_path):
    conn = make_conn()
    add_files(conn, ["c", "a", "b"])
    store = VectorStore(cache_dir=str(tmp_path), ephemeral=False)
    # identical vectors -> identical cosine similarity to any query -> tie
    same_vec = np.array([1.0, 0.0], dtype=np.float32)
    for fid in ("c", "a", "b"):
        store.add(conn, fid, "semantic", "modelA", "v1", same_vec, 1000.0)

    results = store.topk(conn, "semantic", np.array([1.0, 0.0], dtype=np.float32), k=3)
    assert [fid for fid, _ in results] == ["a", "b", "c"]


def test_topk_empty_space_returns_empty_list(tmp_path):
    conn = make_conn()
    add_files(conn, ["f1"])
    store = VectorStore(cache_dir=str(tmp_path), ephemeral=False)
    assert store.topk(conn, "semantic", np.array([1.0, 0.0], dtype=np.float32), k=5) == []


def test_add_rejects_dim_mismatch_within_space_and_version(tmp_path):
    conn = make_conn()
    add_files(conn, ["f1", "f2"])
    store = VectorStore(cache_dir=str(tmp_path), ephemeral=False)
    store.add(conn, "f1", "semantic", "modelA", "v1", np.zeros(8, dtype=np.float32), 1000.0)
    with pytest.raises(ValueError, match="dim mismatch"):
        store.add(conn, "f2", "semantic", "modelA", "v1", np.zeros(16, dtype=np.float32), 1000.0)


# --- space independence -------------------------------------------------------


def test_spaces_never_mix_results():
    conn = make_conn()
    file_ids = [f"f{i}" for i in range(6)]
    add_files(conn, file_ids)
    store = VectorStore(ephemeral=True)

    rng = np.random.default_rng(3)
    semantic_vectors = {fid: rng.standard_normal(8).astype(np.float32) for fid in file_ids}
    visual_vectors = {fid: rng.standard_normal(8).astype(np.float32) for fid in file_ids}
    for fid in file_ids:
        store.add(conn, fid, "semantic", "modelA", "v1", semantic_vectors[fid], 1000.0)
        store.add(conn, fid, "visual", "modelB", "v1", visual_vectors[fid], 1000.0)

    query = rng.standard_normal(8).astype(np.float32)
    semantic_order = [fid for fid, _ in store.topk(conn, "semantic", query, k=6)]
    visual_order = [fid for fid, _ in store.topk(conn, "visual", query, k=6)]

    assert semantic_order != visual_order
    assert semantic_order == [fid for fid, _ in numpy_reference_topk(semantic_vectors, query, 6)]
    assert visual_order == [fid for fid, _ in numpy_reference_topk(visual_vectors, query, 6)]


def test_adding_to_one_space_never_changes_another():
    conn = make_conn()
    file_ids = [f"f{i}" for i in range(6)]
    add_files(conn, file_ids)
    store = VectorStore(ephemeral=True)

    rng = np.random.default_rng(5)
    for fid in file_ids:
        store.add(conn, fid, "semantic", "modelA", "v1", rng.standard_normal(6), 1000.0)
        store.add(conn, fid, "visual", "modelB", "v1", rng.standard_normal(6), 1000.0)

    query = np.ones(6, dtype=np.float32)
    visual_before = store.topk(conn, "visual", query, k=6)

    add_files(conn, ["f6"])
    store.add(conn, "f6", "semantic", "modelA", "v1", rng.standard_normal(6), 1000.0)

    visual_after = store.topk(conn, "visual", query, k=6)
    assert visual_before == visual_after
    assert "f6" not in [fid for fid, _ in visual_after]


# --- persistence / cache -----------------------------------------------------


def test_ephemeral_mode_writes_no_files(tmp_path):
    conn = make_conn()
    add_files(conn, ["f1", "f2"])
    store = VectorStore(cache_dir=str(tmp_path), ephemeral=True)
    store.add(conn, "f1", "semantic", "modelA", "v1", np.array([1.0, 0.0], dtype=np.float32), 1000.0)
    store.add(conn, "f2", "semantic", "modelA", "v1", np.array([0.0, 1.0], dtype=np.float32), 1000.0)
    store.topk(conn, "semantic", np.array([1.0, 0.0], dtype=np.float32), k=2)

    vectors_dir = tmp_path / "vectors"
    assert not vectors_dir.exists() or list(vectors_dir.iterdir()) == []


def test_persistence_round_trip_via_disk_cache(tmp_path):
    conn = make_conn()
    file_ids = [f"f{i}" for i in range(6)]
    add_files(conn, file_ids)
    store1 = VectorStore(cache_dir=str(tmp_path), ephemeral=False)

    rng = np.random.default_rng(13)
    vectors = {fid: rng.standard_normal(10).astype(np.float32) for fid in file_ids}
    for fid, vec in vectors.items():
        store1.add(conn, fid, "semantic", "modelA", "v1", vec, 1000.0)

    query = rng.standard_normal(10).astype(np.float32)
    expected = store1.topk(conn, "semantic", query, k=6)

    cache_file = tmp_path / "vectors" / "semantic__v1.npz"
    assert cache_file.exists()

    # Fresh store instance: empty in-memory cache, same on-disk cache_dir.
    store2 = VectorStore(cache_dir=str(tmp_path), ephemeral=False)

    def _forbidden_rebuild(*_args, **_kwargs):
        raise AssertionError("should have loaded from the on-disk cache, not SQLite")

    store2._load_from_sqlite = _forbidden_rebuild
    got = store2.topk(conn, "semantic", query, k=6)
    assert got == expected


def test_stale_stamp_triggers_automatic_rebuild_on_add(tmp_path):
    conn = make_conn()
    file_ids = [f"f{i}" for i in range(3)]
    add_files(conn, file_ids)
    store = VectorStore(cache_dir=str(tmp_path), ephemeral=False)

    for i, fid in enumerate(file_ids):
        store.add(conn, fid, "semantic", "modelA", "v1", np.eye(3, dtype=np.float32)[i], 1000.0)

    query = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # f2 = [0,0,1] matches exactly (sim=1); f0,f1 tie at sim=0, broken by id.
    before = store.topk(conn, "semantic", query, k=5)
    assert [fid for fid, _ in before] == ["f2", "f0", "f1"]

    # Add a new row directly (a fresh file), without calling store.invalidate().
    add_files(conn, ["f3"])
    store.add(conn, "f3", "semantic", "modelA", "v1", np.array([0.0, 0.0, 1.0], dtype=np.float32), 1000.0)

    # f3 also matches exactly; ties break by ascending id, so f2 stays first.
    after = store.topk(conn, "semantic", query, k=5)
    assert [fid for fid, _ in after] == ["f2", "f3", "f0", "f1"]


def test_stale_stamp_triggers_rebuild_even_on_a_fresh_store_instance(tmp_path):
    """Deleting the npz + starting from empty memory still reproduces topk."""
    conn = make_conn()
    file_ids = [f"f{i}" for i in range(4)]
    add_files(conn, file_ids)
    store1 = VectorStore(cache_dir=str(tmp_path), ephemeral=False)
    rng = np.random.default_rng(21)
    vectors = {fid: rng.standard_normal(5).astype(np.float32) for fid in file_ids}
    for fid, vec in vectors.items():
        store1.add(conn, fid, "semantic", "modelA", "v1", vec, 1000.0)

    query = rng.standard_normal(5).astype(np.float32)
    expected = store1.topk(conn, "semantic", query, k=4)

    cache_file = tmp_path / "vectors" / "semantic__v1.npz"
    assert cache_file.exists()
    os.remove(cache_file)

    # Simulate a fresh process: generations are process-global, so a new
    # store instance alone no longer implies a cold cache.

    with vectors_mod._GEN_LOCK:
        vectors_mod._GENERATIONS.clear()

    store2 = VectorStore(cache_dir=str(tmp_path), ephemeral=False)
    rebuilt = store2.topk(conn, "semantic", query, k=4)
    assert rebuilt == expected
    assert cache_file.exists()  # rebuild re-persists the cache


def test_invalidate_drops_memory_and_disk_cache(tmp_path):
    conn = make_conn()
    add_files(conn, ["f1", "f2"])
    store = VectorStore(cache_dir=str(tmp_path), ephemeral=False)
    store.add(conn, "f1", "semantic", "modelA", "v1", np.array([1.0, 0.0], dtype=np.float32), 1000.0)
    store.add(conn, "f2", "semantic", "modelA", "v1", np.array([0.0, 1.0], dtype=np.float32), 1000.0)
    store.topk(conn, "semantic", np.array([1.0, 0.0], dtype=np.float32), k=2)

    cache_file = tmp_path / "vectors" / "semantic__v1.npz"
    assert cache_file.exists()

    with vectors_mod._GEN_LOCK:
        assert "semantic" in vectors_mod._GENERATIONS

    store.invalidate("semantic")
    assert not cache_file.exists()
    with vectors_mod._GEN_LOCK:
        assert "semantic" not in vectors_mod._GENERATIONS


def test_topk_pins_model_version_during_migration():
    """A query vector taken from a stored row must be compared only against
    rows of that row's own model_version — even when a newer version has
    become the space's active one."""
    conn = make_conn()
    add_files(conn, ["old_a", "old_b", "new_a", "new_b"])
    store = VectorStore(ephemeral=True)

    va = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vb = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    store.add(conn, "old_a", "semantic", "m", "v1", va, 1000.0)
    store.add(conn, "old_b", "semantic", "m", "v1", vb, 1000.0)
    store.add(conn, "new_a", "semantic", "m", "v2", va, 1000.0)
    store.add(conn, "new_b", "semantic", "m", "v2", vb, 1000.0)

    pinned = store.topk(conn, "semantic", va, k=10, model_version="v1")
    assert [fid for fid, _ in pinned] == ["old_a", "old_b"]

    # Unpinned falls back to the active (most recently computed) version.
    active = store.topk(conn, "semantic", va, k=10)
    assert [fid for fid, _ in active] == ["new_a", "new_b"]


def test_topk_faiss_and_numpy_paths_agree_with_exclusions(tmp_path, monkeypatch):
    """The faiss IDSelector exclusion path and the numpy fallback must return
    the same ids and similarities. sys.modules[name]=None makes the inline
    `import faiss` raise ImportError, forcing the fallback."""

    pytest.importorskip("faiss")
    conn = make_conn()
    file_ids = [f"f{i}" for i in range(12)]
    add_files(conn, file_ids)
    store = VectorStore(cache_dir=str(tmp_path), ephemeral=False)
    rng = np.random.default_rng(23)
    for fid in file_ids:
        store.add(conn, fid, "semantic", "modelA", "v1", rng.standard_normal(8), 1000.0)
    query = rng.standard_normal(8).astype(np.float32)

    via_faiss = store.topk(conn, "semantic", query, k=5, exclude=("f3", "f7", "ghost"))
    monkeypatch.setitem(sys.modules, "faiss", None)
    store.invalidate("semantic")
    via_numpy = store.topk(conn, "semantic", query, k=5, exclude=("f3", "f7", "ghost"))

    assert [fid for fid, _ in via_faiss] == [fid for fid, _ in via_numpy]
    for (_, a), (_, b) in zip(via_faiss, via_numpy, strict=False):
        assert a == pytest.approx(b, abs=1e-5)


def test_topk_gpu_and_cpu_paths_agree(monkeypatch):
    """With a GPU faiss build the GPU index must return exactly the CPU
    path's neighbors (exact search either way; exclusions honored via
    over-fetch+filter on GPU, IDSelector on CPU). On faiss-cpu builds
    both runs take the CPU path and the test still holds."""

    conn = make_conn()
    ids = [f"g{i:03d}" for i in range(200)]
    add_files(conn, ids)
    store = VectorStore(ephemeral=True)
    rng = np.random.default_rng(9)
    for fid in ids:
        v = rng.standard_normal(32).astype(np.float32)
        store.add(conn, fid, "semantic", "m", "v", v / np.linalg.norm(v), 1.0)
    q = rng.standard_normal(32).astype(np.float32)

    monkeypatch.setenv("AI_DAM_VECTOR_GPU", "1")
    first = store.topk(conn, "semantic", q, 10, exclude=["g001", "g002"])
    V._GENERATIONS.clear()
    monkeypatch.setenv("AI_DAM_VECTOR_GPU", "0")
    second = store.topk(conn, "semantic", q, 10, exclude=["g001", "g002"])

    assert [f for f, _ in first] == [f for f, _ in second]
    assert max(abs(a - b) for (_, a), (_, b) in zip(first, second, strict=False)) < 1e-5
    assert all(f not in ("g001", "g002") for f, _ in first)
