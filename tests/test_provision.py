"""Tests for smartgallery_ai.provision and the worker's async
auto-provisioning: group resolution, plan rendering, idempotent skip,
hash verification, downloader dispatch, the backend-key -> group mapping,
and the worker thread's success/failure/disabled behavior. Entirely
network-free: every downloader is injected or monkeypatched."""

from __future__ import annotations

import os
import sqlite3
import time

import pytest

from smartgallery_ai import AIConfig
from smartgallery_ai import provision as P
from smartgallery_ai.schema import init_schema
from smartgallery_ai.worker import AIWorker, provision_groups_for


# --- group resolution / plan --------------------------------------------------


def test_resolve_groups_all_and_named_and_unknown():
    """'all' (and no names) expands to every group; unknown names raise
    ValueError listing the valid ones."""
    assert [g.name for g in P.resolve_groups(["all"])] == [g.name for g in P.GROUPS]
    assert [g.name for g in P.resolve_groups([])] == [g.name for g in P.GROUPS]
    assert [g.name for g in P.resolve_groups(["faces", "semantic"])] == ["faces", "semantic"]
    with pytest.raises(ValueError, match="bogus"):
        P.resolve_groups(["faces", "bogus"])


def test_registry_dests_match_backend_expectations():
    """The registry's target paths are exactly where the backends look;
    drifting apart would download weights nobody loads."""
    dests = {a.dest for g in P.GROUPS for a in g.artifacts}
    assert "face_detection_yunet_2023mar.onnx" in dests
    assert "face_recognition_sface_2021dec.onnx" in dests
    assert "open_clip/ViT-B-32_laion2b_s34b_b79k.bin" in dests
    assert "dinov2-small" in dests
    assert "mobile_sam.pt" in dests
    assert "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" in dests
    assert "mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf" in dests


def test_format_plan_reports_missing_and_present(tmp_path):
    """The plan marks artifacts MISSING/present according to the models
    dir contents."""
    (tmp_path / "face_detection_yunet_2023mar.onnx").write_bytes(b"x")
    plan = P.format_plan(str(tmp_path), ["faces"])
    assert "present  face_detection_yunet_2023mar.onnx" in plan
    assert "MISSING  face_recognition_sface_2021dec.onnx" in plan


# --- provision(): dispatch, idempotence, verification -------------------------


def _fake_downloaders(written: dict, content: bytes = b"weights"):
    """Downloader trio that records calls and writes `content` at dest."""

    def url(u, dest):
        written[dest] = ("url", u)
        with open(dest, "wb") as fh:
            fh.write(content)

    def hf_file(repo, filename, dest):
        written[dest] = ("hf_file", repo, filename)
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(content)

    def hf_snapshot(repo, dest):
        written[dest] = ("hf_snapshot", repo)
        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, "model.safetensors"), "wb") as fh:
            fh.write(content)

    return {"url": url, "hf_file": hf_file, "hf_snapshot": hf_snapshot}


def test_provision_dispatches_by_source_kind_and_skips_present(tmp_path):
    """URL artifacts use the url downloader, single HF files the hf_file
    downloader, snapshot dirs the snapshot downloader; artifacts already
    on disk are skipped, not re-downloaded."""
    written: dict = {}
    # visual has no sha256 pin; faces artifacts do (fake bytes would fail
    # verification), so pre-place them to exercise the skip path instead.
    (tmp_path / "face_detection_yunet_2023mar.onnx").write_bytes(b"x")
    (tmp_path / "face_recognition_sface_2021dec.onnx").write_bytes(b"x")

    result = P.provision(str(tmp_path), ["faces", "visual"], log=lambda m: None,
                         downloaders=_fake_downloaders(written))

    assert result["skipped"] == ["face_detection_yunet_2023mar.onnx",
                                 "face_recognition_sface_2021dec.onnx"]
    assert result["downloaded"] == ["dinov2-small"]
    dest = str(tmp_path / "dinov2-small")
    assert written[dest] == ("hf_snapshot", "facebook/dinov2-small")
    assert os.path.isfile(os.path.join(dest, "model.safetensors"))


def test_provision_hash_mismatch_deletes_and_raises(tmp_path):
    """A pinned artifact whose downloaded bytes hash differently is
    removed and the run fails loudly, never leaving a poisoned file."""
    written: dict = {}
    with pytest.raises(P.ProvisionError, match="SHA-256 mismatch"):
        P.provision(str(tmp_path), ["faces"], log=lambda m: None,
                    downloaders=_fake_downloaders(written, content=b"tampered"))
    assert not os.path.exists(tmp_path / "face_detection_yunet_2023mar.onnx")


def test_provision_download_failure_names_artifact(tmp_path):
    """A downloader exception surfaces as ProvisionError naming the dest."""

    def boom(u, dest):
        raise OSError("connection refused")

    with pytest.raises(P.ProvisionError, match="face_detection_yunet_2023mar.onnx"):
        P.provision(str(tmp_path), ["faces"], log=lambda m: None,
                    downloaders={"url": boom})


# --- worker mapping + async auto-provisioning ---------------------------------


def _cfg(tmp_path, **overrides) -> AIConfig:
    defaults = dict(
        enabled=True, base_path=str(tmp_path), db_path=str(tmp_path / "g.sqlite"),
        models_dir=str(tmp_path / "models"), cache_dir=str(tmp_path / "cache"),
        ephemeral_index=True,
    )
    defaults.update(overrides)
    return AIConfig(**defaults)


def _make_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE files (id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE,"
                 " mtime REAL NOT NULL, name TEXT NOT NULL, type TEXT)")
    init_schema(conn)
    conn.commit()
    conn.close()


def test_provision_groups_for_maps_backends_and_critic_pulls_semantic(tmp_path):
    """Each real-backend selector maps to its group; qwen-vl critic adds
    the semantic (grounding-gate) weights; 'none'/'stub' map to nothing."""
    os.makedirs(tmp_path / "models", exist_ok=True)
    cfg = _cfg(tmp_path, semantic_backend="none", visual_backend="none",
               face_backend="none", segmenter_backend="none", critic_backend="qwen-vl")
    assert provision_groups_for(cfg) == ["critic", "semantic"]

    cfg2 = _cfg(tmp_path, semantic_backend="stub", visual_backend="stub",
                face_backend="stub", segmenter_backend="stub", critic_backend="stub")
    assert provision_groups_for(cfg2) == []

    cfg3 = _cfg(tmp_path, semantic_backend="none", visual_backend="auto",
                face_backend="auto", segmenter_backend="none", critic_backend="none")
    assert provision_groups_for(cfg3) == ["visual", "faces"]


def test_provision_groups_for_omits_groups_already_on_disk(tmp_path):
    """A group whose artifacts all exist is not re-provisioned."""
    models = tmp_path / "models"
    os.makedirs(models)
    (models / "face_detection_yunet_2023mar.onnx").write_bytes(b"x")
    (models / "face_recognition_sface_2021dec.onnx").write_bytes(b"x")
    cfg = _cfg(tmp_path, semantic_backend="none", visual_backend="none",
               face_backend="auto", segmenter_backend="none", critic_backend="none")
    assert provision_groups_for(cfg) == []


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_worker_auto_provisions_missing_groups_async(tmp_path, monkeypatch):
    """start() kicks off ONE background provisioning attempt for exactly
    the missing groups, without blocking the worker; on success the cached
    backend misses are dropped so the next cycle re-probes immediately."""
    from smartgallery_ai import worker as W

    _make_db(str(tmp_path / "g.sqlite"))
    os.makedirs(tmp_path / "models", exist_ok=True)
    cfg = _cfg(tmp_path, semantic_backend="none", visual_backend="none",
               face_backend="auto", segmenter_backend="none", critic_backend="none")
    worker = AIWorker(cfg, cfg.db_path, poll_interval=0.05, batch_size=10)

    calls = []

    def fake_provision(models_dir, groups, force=False, log=print, downloaders=None):
        calls.append((models_dir, list(groups)))
        return {"downloaded": list(groups), "skipped": []}

    monkeypatch.setattr(W.provisioning, "provision", fake_provision)
    worker._backend_cache["face"] = None  # a cached miss that must be dropped
    worker._backend_failed_at["face"] = 1e18

    worker.start()
    try:
        assert _wait_until(lambda: worker.provision_state["state"] == "done")
        assert calls == [(cfg.models_dir, ["faces"])]
        assert "face" not in worker._backend_cache  # miss dropped -> re-probe
        assert worker._backend_failed_at == {}
        assert _wait_until(lambda: worker.stats["cycles"] > 0)  # never blocked
    finally:
        worker.stop(timeout=2.0)


def test_worker_auto_provision_failure_degrades_and_worker_survives(tmp_path, monkeypatch):
    """A failing download (egress-denied host) leaves state 'failed: ...',
    counts one error, and the worker keeps cycling normally."""
    from smartgallery_ai import worker as W

    _make_db(str(tmp_path / "g.sqlite"))
    os.makedirs(tmp_path / "models", exist_ok=True)
    cfg = _cfg(tmp_path, semantic_backend="none", visual_backend="none",
               face_backend="auto", segmenter_backend="none", critic_backend="none")
    worker = AIWorker(cfg, cfg.db_path, poll_interval=0.05, batch_size=10)

    def refuse(*a, **k):
        raise P.ProvisionError("connection refused")

    monkeypatch.setattr(W.provisioning, "provision", refuse)
    worker.start()
    try:
        assert _wait_until(
            lambda: str(worker.provision_state["state"]).startswith("failed"))
        assert _wait_until(lambda: worker.stats["cycles"] > 0)
        assert worker.stats["errors"] >= 1
    finally:
        worker.stop(timeout=2.0)


def test_worker_auto_provision_disabled_never_downloads(tmp_path, monkeypatch):
    """AI_DAM_AUTO_PROVISION=false (config auto_provision=False) is the
    strict no-egress mode: no provisioning attempt is ever made."""
    from smartgallery_ai import worker as W

    _make_db(str(tmp_path / "g.sqlite"))
    os.makedirs(tmp_path / "models", exist_ok=True)
    cfg = _cfg(tmp_path, auto_provision=False, semantic_backend="auto",
               visual_backend="auto", face_backend="auto",
               segmenter_backend="auto", critic_backend="auto")
    worker = AIWorker(cfg, cfg.db_path, poll_interval=0.05, batch_size=10)

    def must_not_run(*a, **k):
        raise AssertionError("provisioning ran despite auto_provision=False")

    monkeypatch.setattr(W.provisioning, "provision", must_not_run)
    worker.start()
    try:
        assert worker.provision_state == {"state": "disabled", "groups": []}
        assert _wait_until(lambda: worker.stats["cycles"] > 0)
    finally:
        worker.stop(timeout=2.0)


def test_worker_provision_state_exposed_by_status_endpoint(tmp_path, monkeypatch):
    """/status reports the worker's provisioning state so the UI can show
    download progress instead of a bare empty tab."""
    from flask import Flask

    from smartgallery_ai import service as S
    from smartgallery_ai import worker as W
    from smartgallery_ai.service import create_ai_blueprint, set_worker

    _make_db(str(tmp_path / "g.sqlite"))
    os.makedirs(tmp_path / "models", exist_ok=True)
    cfg = _cfg(tmp_path, semantic_backend="none", visual_backend="none",
               face_backend="none", segmenter_backend="none", critic_backend="none")
    worker = AIWorker(cfg, cfg.db_path, poll_interval=0.05, batch_size=10)
    worker.provision_state = {"state": "downloading", "groups": ["semantic"]}
    set_worker(worker)
    try:
        app = Flask(__name__)
        app.register_blueprint(create_ai_blueprint(cfg), url_prefix="/aidam")
        status = app.test_client().get("/aidam/status").get_json()
        assert status["worker"]["provisioning"] == {
            "state": "downloading", "groups": ["semantic"]}
    finally:
        set_worker(None)


def test_invalidate_backend_probe_cache_clears_registered_caches(tmp_path):
    """The worker's post-provision hook empties every blueprint's probe
    cache so /status re-probes instead of serving a stale False."""
    from smartgallery_ai import service as S

    cache_before = list(S._PROBE_CACHES)
    try:
        S._PROBE_CACHES.append({"semantic": False})
        S.invalidate_backend_probe_cache()
        assert S._PROBE_CACHES[-1] == {}
    finally:
        S._PROBE_CACHES[:] = cache_before


# --- CLI ----------------------------------------------------------------------


def test_cli_provision_list_prints_plan_without_downloading(tmp_path, capsys):
    """`provision --list` prints the plan and exits 0 with no downloads."""
    from smartgallery_ai.__main__ import main

    rc = main(["provision", "faces", "--models-dir", str(tmp_path), "--list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "MISSING" in out and "face_detection_yunet_2023mar.onnx" in out
    assert not os.listdir(tmp_path)


def test_cli_provision_unknown_group_exits_nonzero(tmp_path, capsys):
    from smartgallery_ai.__main__ import main

    rc = main(["provision", "bogus", "--models-dir", str(tmp_path), "--list"])
    assert rc == 2
    assert "unknown group" in capsys.readouterr().out


# --- runtime package installation ---------------------------------------------


def test_runtime_missing_reports_only_unimportable(monkeypatch):
    """Probes that import are never reinstalled; only genuinely missing
    modules produce install work."""
    import importlib.util as IU

    group = next(g for g in P.GROUPS if g.name == "semantic")
    real_find = IU.find_spec

    monkeypatch.setattr(P.importlib.util, "find_spec",
                        lambda name: None if name == "open_clip" else real_find(name))
    assert [req for _, req in P.runtime_missing(group)] == ["open_clip_torch"]


def test_ensure_runtime_installs_missing_via_pip_runner(monkeypatch):
    """ensure_runtime pip-installs exactly the missing requirements and
    refreshes import caches afterwards."""
    group = next(g for g in P.GROUPS if g.name == "visual")
    monkeypatch.setattr(P.importlib.util, "find_spec", lambda name: None)
    invalidated = []
    monkeypatch.setattr(P.importlib, "invalidate_caches", lambda: invalidated.append(1))

    calls = []
    installed = P.ensure_runtime(group, log=lambda m: None,
                                 pip_runner=lambda args: calls.append(args))
    assert installed == ["torch", "transformers"]
    assert calls[1] == ["transformers"]
    assert invalidated == [1]


def test_pip_args_steer_torch_by_hardware(monkeypatch):
    """torch wheel choice: NVIDIA present -> CUDA-capable (PyPI on Linux,
    cu-index on Windows); absent -> CPU index; AI_DAM_DEVICE=cpu forces
    CPU even with hardware; non-torch requirements pass through."""
    monkeypatch.delenv("AI_DAM_DEVICE", raising=False)
    monkeypatch.setattr(P.sys, "platform", "linux")

    monkeypatch.setattr(P, "cuda_hardware_present", lambda: False)
    assert P._pip_args_for("torch") == ["torch", "--index-url", P._TORCH_CPU_INDEX]

    monkeypatch.setattr(P, "cuda_hardware_present", lambda: True)
    assert P._pip_args_for("torch") == ["torch"]

    monkeypatch.setattr(P.sys, "platform", "win32")
    assert P._pip_args_for("torch") == [
        "torch", "--index-url", P._TORCH_CUDA_WINDOWS_INDEX]

    monkeypatch.setenv("AI_DAM_DEVICE", "cpu")
    assert P._pip_args_for("torch") == ["torch", "--index-url", P._TORCH_CPU_INDEX]

    assert P._pip_args_for("timm") == ["timm"]


def test_provision_installs_runtime_before_weights(tmp_path, monkeypatch):
    """provision() makes the group loadable end to end: missing runtime
    packages install first, then weights download."""
    events = []
    monkeypatch.setattr(P.importlib.util, "find_spec",
                        lambda name: None if name in ("torch", "transformers") else object())

    def runner(args):
        events.append(("pip", tuple(args)))

    downloaders = _fake_downloaders({})

    def snapshot(repo, dest):
        events.append(("weights", repo))
        downloaders["hf_snapshot"](repo, dest)

    result = P.provision(str(tmp_path), ["visual"], log=lambda m: None,
                         downloaders={**downloaders, "hf_snapshot": snapshot},
                         pip_runner=runner)
    assert result["installed"] == ["torch", "transformers"]
    assert [e[0] for e in events] == ["pip", "pip", "weights"]


def test_provision_install_packages_false_skips_pip(tmp_path, monkeypatch):
    """The opt-out: install_packages=False downloads weights only."""
    monkeypatch.setattr(P.importlib.util, "find_spec", lambda name: None)

    def must_not_run(args):
        raise AssertionError(f"pip ran: {args}")

    result = P.provision(str(tmp_path), ["visual"], log=lambda m: None,
                         downloaders=_fake_downloaders({}),
                         install_packages=False, pip_runner=must_not_run)
    assert result["installed"] == []
    assert result["downloaded"] == ["dinov2-small"]


def test_provision_groups_for_includes_runtime_missing_groups(tmp_path, monkeypatch):
    """A group whose weights exist but whose runtime cannot import is still
    auto-provisioned (the runtime is installable)."""
    models = tmp_path / "models"
    os.makedirs(models / "dinov2-small", exist_ok=True)
    (models / "dinov2-small" / "model.safetensors").write_bytes(b"x")

    from smartgallery_ai import worker as W

    monkeypatch.setattr(W.provisioning, "runtime_missing",
                        lambda g: [("torch", "torch")] if g.name == "visual" else [])
    cfg = _cfg(tmp_path, semantic_backend="none", visual_backend="auto",
               face_backend="none", segmenter_backend="none", critic_backend="none")
    assert provision_groups_for(cfg) == ["visual"]


def test_format_plan_lists_runtime_rows(tmp_path, monkeypatch):
    """The plan shows runtime requirements with present/MISSING state."""
    monkeypatch.setattr(P.importlib.util, "find_spec",
                        lambda name: None if name == "open_clip" else object())
    plan = P.format_plan(str(tmp_path), ["semantic"])
    assert "present  runtime torch" in plan
    assert "MISSING  runtime open_clip_torch" in plan


# --- opt-out defaults ---------------------------------------------------------


def test_ai_layer_enabled_by_default_from_env(monkeypatch):
    """The layer is opt-OUT: a clean environment enables it; only an
    explicit false disables."""
    for var in ("ENABLE_AI_DAM", "AI_DAM_AUTO_PROVISION"):
        monkeypatch.delenv(var, raising=False)
    cfg = AIConfig.from_env("/tmp/base", "/tmp/db.sqlite")
    assert cfg.enabled is True
    assert cfg.auto_provision is True

    monkeypatch.setenv("ENABLE_AI_DAM", "false")
    assert AIConfig.from_env("/tmp/base", "/tmp/db.sqlite").enabled is False
    monkeypatch.setenv("ENABLE_AI_DAM", "true")
    monkeypatch.setenv("AI_DAM_AUTO_PROVISION", "false")
    assert AIConfig.from_env("/tmp/base", "/tmp/db.sqlite").auto_provision is False


def test_pick_torch_device_prefers_cuda_then_cpu(monkeypatch):
    """Device selection: forced env wins, else CUDA when available, else
    CPU (MPS covered on mac hardware)."""
    from types import SimpleNamespace

    from smartgallery_ai.embedders import pick_torch_device

    fake_cuda = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace())
    fake_cpu = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace())

    monkeypatch.delenv("AI_DAM_DEVICE", raising=False)
    assert pick_torch_device(fake_cuda) == "cuda"
    assert pick_torch_device(fake_cpu) == "cpu"
    monkeypatch.setenv("AI_DAM_DEVICE", "cpu")
    assert pick_torch_device(fake_cuda) == "cpu"
