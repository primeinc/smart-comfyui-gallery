"""The faces job runs the backend the setting names, on the device the
setting names, over weights it may fetch from their own registry.

Lookup is read-only and two-deep -- the run's models_dir, then the
machine's shared cache -- and only a job may download. Nothing here
touches the network: the registries' clients are replaced at the seam
and what reaches them is asserted.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from db import runner, settings
from tests.staging import fresh_schema
from vision import faces, weights
from vision.faces import FaceDetection, StubFaceBackend


def _hub_layout(cache: pathlib.Path, repo_id: str, filename: str, revision: str = "abc123") -> pathlib.Path:
    """One file in huggingface_hub's cache layout: refs/main -> a
    snapshot directory holding the file (file_download.py:1540-1564)."""
    repo = cache / f"models--{repo_id.replace('/', '--')}"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(revision, encoding="utf-8")
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True)
    path = snapshot / filename
    path.write_bytes(b"onnx")
    return path


def _pack_at(root: pathlib.Path) -> pathlib.Path:
    pack = root / "models" / weights.PACK
    pack.mkdir(parents=True)
    (pack / "glintr100.onnx").write_bytes(b"onnx")
    return pack


@pytest.fixture
def models_dir(tmp_path) -> str:
    where = tmp_path / "models"
    where.mkdir()
    return str(where)


@pytest.fixture
def shared_hub(tmp_path, monkeypatch) -> pathlib.Path:
    """The machine's HF_HUB_CACHE, pointed at a scratch directory."""
    from huggingface_hub import constants

    cache = tmp_path / "shared-hub"
    cache.mkdir()
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    return cache


@pytest.fixture
def shared_insightface(tmp_path, monkeypatch) -> pathlib.Path:
    """The machine's ~/.insightface, pointed at a scratch directory."""
    home = tmp_path / "dot-insightface"
    monkeypatch.setattr(weights, "INSIGHTFACE_HOME", str(home))
    return home


# --- hub files: models_dir, then the shared cache, then (a job) the Hub ------


def test_a_hub_file_under_models_dir_is_found_there(models_dir, shared_hub):
    own = _hub_layout(pathlib.Path(models_dir), weights.YUNET_REPO, weights.YUNET_FILE)

    assert weights.hub_cached(weights.YUNET_REPO, weights.YUNET_FILE, models_dir) == str(own)


def test_a_hub_file_in_the_shared_cache_is_used_where_it_lies(models_dir, shared_hub):
    shared = _hub_layout(shared_hub, weights.YUNET_REPO, weights.YUNET_FILE)

    assert weights.hub_cached(weights.YUNET_REPO, weights.YUNET_FILE, models_dir) == str(shared)


def test_models_dir_wins_over_the_shared_cache(models_dir, shared_hub):
    own = _hub_layout(pathlib.Path(models_dir), weights.YUNET_REPO, weights.YUNET_FILE)
    _hub_layout(shared_hub, weights.YUNET_REPO, weights.YUNET_FILE)

    assert weights.hub_cached(weights.YUNET_REPO, weights.YUNET_FILE, models_dir) == str(own)


def test_an_absent_hub_file_is_none_never_a_download(models_dir, shared_hub, monkeypatch):
    from huggingface_hub import file_download

    monkeypatch.setattr(file_download, "hf_hub_download", lambda *a, **k: pytest.fail("a lookup downloaded"))

    assert weights.hub_cached(weights.YUNET_REPO, weights.YUNET_FILE, models_dir) is None


def test_without_provision_a_missing_hub_file_is_refused_by_name(models_dir, shared_hub):
    with pytest.raises(weights.Unprovisioned, match=weights.YUNET_FILE):
        weights.hub_file(weights.YUNET_REPO, weights.YUNET_FILE, models_dir, provision=False)


def test_with_provision_a_missing_hub_file_is_fetched_into_models_dir(models_dir, shared_hub, monkeypatch):
    import huggingface_hub

    asked: list = []

    def fetched(repo_id, filename, *, cache_dir=None, revision=None, **kw):
        asked.append((repo_id, filename, cache_dir, revision))
        return os.path.join(cache_dir, filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetched)

    got = weights.hub_file(weights.SFACE_REPO, weights.SFACE_FILE, models_dir, provision=True)

    assert asked == [(weights.SFACE_REPO, weights.SFACE_FILE, models_dir, None)], "the download lands in models_dir"
    assert got == os.path.join(models_dir, weights.SFACE_FILE)


# --- the insightface pack: own copy, shared copy, insightface's own fetch ------


def test_the_runs_own_pack_is_the_root(models_dir, shared_insightface):
    own = pathlib.Path(models_dir) / weights.INSIGHTFACE_SUBDIR
    _pack_at(own)

    assert weights.insightface_root(models_dir, provision=False) == str(own)


def test_the_shared_pack_is_used_where_it_lies(models_dir, shared_insightface):
    _pack_at(shared_insightface)

    assert weights.insightface_root(models_dir, provision=False) == str(shared_insightface)


def test_an_absent_pack_is_none_without_provision(models_dir, shared_insightface):
    assert weights.insightface_root(models_dir, provision=False) is None


def test_with_provision_insightface_fetches_the_pack_into_the_runs_copy(models_dir, shared_insightface, monkeypatch):
    import insightface.utils

    asked: list = []

    def fetched(sub_dir, name, root):
        asked.append((sub_dir, name, root))
        pack = pathlib.Path(root) / sub_dir / name
        pack.mkdir(parents=True)
        (pack / "glintr100.onnx").write_bytes(b"onnx")
        return str(pack)

    monkeypatch.setattr(insightface.utils, "ensure_available", fetched)
    own = os.path.join(models_dir, weights.INSIGHTFACE_SUBDIR)

    assert weights.insightface_root(models_dir, provision=True) == own
    assert asked == [("models", weights.PACK, own)], "insightface's own client, into the run's copy"


def test_a_pack_unzipped_one_level_too_deep_is_lifted_to_where_faceanalysis_looks(
    models_dir, shared_insightface, monkeypatch
):
    """The release zip repeats the pack name as a directory inside the
    pack; FaceAnalysis globs only the top level."""
    import insightface.utils

    def fetched(sub_dir, name, root):
        nested = pathlib.Path(root) / sub_dir / name / name
        nested.mkdir(parents=True)
        (nested / "glintr100.onnx").write_bytes(b"onnx")
        (nested / "scrfd_10g_bnkps.onnx").write_bytes(b"onnx")
        return str(nested.parent)

    monkeypatch.setattr(insightface.utils, "ensure_available", fetched)

    root = weights.insightface_root(models_dir, provision=True)

    pack = pathlib.Path(root) / "models" / weights.PACK
    assert sorted(p.name for p in pack.iterdir()) == ["glintr100.onnx", "scrfd_10g_bnkps.onnx"]


# --- the OpenCV stack's files ---------------------------------------------------


def test_flat_files_under_models_dir_are_honoured_first(models_dir, shared_hub, shared_insightface):
    yunet = pathlib.Path(models_dir) / weights.YUNET_FILE
    sface = pathlib.Path(models_dir) / weights.SFACE_FILE
    yunet.write_bytes(b"onnx")
    sface.write_bytes(b"onnx")

    held = weights.opencv_weights(models_dir, provision=False)

    assert held == weights.OpenCVWeights(yunet=str(yunet), sface=str(sface), arcface=None)


def test_the_pack_when_present_supplies_arcface_and_sface_is_not_fetched(
    models_dir, shared_hub, shared_insightface, monkeypatch
):
    import huggingface_hub

    _hub_layout(shared_hub, weights.YUNET_REPO, weights.YUNET_FILE)
    pack = _pack_at(shared_insightface)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: pytest.fail("sface was fetched"))

    held = weights.opencv_weights(models_dir, provision=True)

    assert held.arcface == str(pack / "glintr100.onnx")
    assert held.sface is None


def test_without_the_pack_provisioning_fetches_yunet_and_sface(models_dir, shared_hub, shared_insightface, monkeypatch):
    import huggingface_hub

    asked: list = []

    def fetched(repo_id, filename, *, cache_dir=None, revision=None, **kw):
        asked.append((repo_id, filename))
        return os.path.join(cache_dir, filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fetched)

    held = weights.opencv_weights(models_dir, provision=True)

    assert asked == [(weights.YUNET_REPO, weights.YUNET_FILE), (weights.SFACE_REPO, weights.SFACE_FILE)]
    assert held.arcface is None
    assert held.sface is not None


# --- the backend the setting names ----------------------------------------------


@pytest.fixture
def recorded_backends(monkeypatch) -> dict:
    """Both constructors replaced: what each was asked is recorded."""
    made: dict = {"opencv": [], "insightface": []}

    class OpenCV:
        def __init__(self, models_dir, **kw):
            made["opencv"].append((models_dir, kw))

    class Insight:
        def __init__(self, models_dir, **kw):
            made["insightface"].append((models_dir, kw))

    monkeypatch.setattr(faces, "OpenCVFaceBackend", OpenCV)
    monkeypatch.setattr(faces, "InsightFaceBackend", Insight)
    return made


def test_opencv_is_exactly_the_opencv_stack(recorded_backends):
    faces.backend_for("M", choice="opencv", providers="cpu", provision=True)

    assert recorded_backends == {"opencv": [("M", {"provision": True})], "insightface": []}


def test_insightface_carries_the_providers_setting(recorded_backends):
    faces.backend_for("M", choice="insightface", providers="CUDAExecutionProvider", provision=True)

    assert recorded_backends["insightface"] == [("M", {"providers": "CUDAExecutionProvider", "provision": True})]
    assert recorded_backends["opencv"] == []


def test_auto_is_insightface_when_its_runtime_is_installed(recorded_backends, monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

    faces.backend_for("M", choice="auto", providers="auto")

    assert recorded_backends["insightface"] == [("M", {"providers": "auto", "provision": False})]


def test_auto_takes_the_opencv_stack_only_without_the_insightface_runtime(recorded_backends, monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    faces.backend_for("M", choice="auto")

    assert recorded_backends == {"opencv": [("M", {"provision": False})], "insightface": []}


def test_an_unknown_backend_is_refused_by_name(recorded_backends):
    with pytest.raises(ValueError, match="face_backend"):
        faces.backend_for("M", choice="dlib")


# --- the job --------------------------------------------------------------------


def _one_picture(conn, root: pathlib.Path) -> int:
    from PIL import Image

    from db import library, scan

    root.mkdir()
    Image.new("RGB", (16, 16), (200, 90, 40)).save(root / "one.png")
    root_id = library.add_root(conn, str(root), "library", 0.0)
    scan.scan(conn, root_id, str(root), 0.0)
    return conn.execute("SELECT id FROM file").fetchone()[0]


def test_submit_reads_the_backend_and_providers_settings_into_the_payload(tmp_path):
    conn = fresh_schema()
    _one_picture(conn, tmp_path / "lib")
    settings.put(conn, "face_backend", "opencv")
    settings.put(conn, "ort_providers", "cpu")

    job_id = runner.submit_faces(conn, 0.0, models_dir="M")

    payload = conn.execute("SELECT payload FROM job WHERE id = ?", (job_id,)).fetchone()[0]
    import json

    assert json.loads(payload) == {"models_dir": "M", "backend": "opencv", "providers": "cpu"}


def test_the_job_item_asks_for_the_payloads_backend_with_provisioning(tmp_path, monkeypatch):
    conn = fresh_schema()
    file_id = _one_picture(conn, tmp_path / "lib")
    asked: list = []

    def chosen(models_dir, *, choice, providers, provision):
        asked.append((models_dir, choice, providers, provision))
        return StubFaceBackend(lambda img: [FaceDetection((0.1, 0.1, 0.2, 0.2), [], 0.9, [1.0, 0.0])])

    monkeypatch.setattr(faces, "backend_for", chosen)
    monkeypatch.setattr(runner, "_BACKENDS", {})

    runner._face_item(conn, file_id, {"models_dir": "M", "backend": "insightface", "providers": "cpu"}, 0.0)

    assert asked == [("M", "insightface", "cpu", True)], "the job is the one caller that may provision"
    assert conn.execute("SELECT count(*) FROM derived_face_instance").fetchone()[0] == 1


def test_one_backend_per_payload_is_held_across_items(tmp_path, monkeypatch):
    conn = fresh_schema()
    file_id = _one_picture(conn, tmp_path / "lib")
    built: list = []

    def chosen(models_dir, **kw):
        built.append(kw)
        return StubFaceBackend(lambda img: [])

    monkeypatch.setattr(faces, "backend_for", chosen)
    monkeypatch.setattr(runner, "_BACKENDS", {})
    payload = {"models_dir": "M", "backend": "opencv", "providers": "auto"}

    runner._face_item(conn, file_id, payload, 0.0)
    runner._face_item(conn, file_id, payload, 0.0)
    runner._face_item(conn, file_id, {**payload, "providers": "cpu"}, 0.0)

    assert len(built) == 2, "same payload, same backend; a different device is a different backend"


def test_the_face_backend_setting_is_a_row_with_a_closed_vocabulary():
    conn = fresh_schema()

    assert settings.value(conn, "face_backend") == "auto"
    settings.put(conn, "face_backend", "insightface")
    assert settings.value(conn, "face_backend") == "insightface"
    with pytest.raises(ValueError, match="face_backend"):
        settings.put(conn, "face_backend", "dlib")
    conn.close()
