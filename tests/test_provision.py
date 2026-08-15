"""Tests for smartgallery_ai.provision and the worker's async
auto-provisioning: group resolution, plan rendering, idempotent skip,
hash verification, downloader dispatch, the backend-key -> group mapping,
and the worker thread's success/failure/disabled behavior. Entirely
network-free: every downloader is injected or monkeypatched."""

from __future__ import annotations

import contextlib
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
    cfg = _cfg(tmp_path, auto_provision=True, semantic_backend="none",
               visual_backend="none", face_backend="auto",
               segmenter_backend="none", critic_backend="none")
    worker = AIWorker(cfg, cfg.db_path, poll_interval=0.05, batch_size=10)

    calls = []

    def fake_provision(models_dir, groups, force=False, log=print,
                       downloaders=None, progress=None):
        calls.append((models_dir, list(groups)))
        return {"downloaded": list(groups), "skipped": [], "installed": []}

    monkeypatch.setattr(W.provisioning, "provision", fake_provision)
    worker._backend_cache["face"] = None  # a cached miss that must be dropped
    worker._backend_failed_at["face"] = 1e18

    # Drive the provisioning path directly (no cycle loop): the loop would
    # legitimately re-probe and re-cache a miss after the clear, racing the
    # assertions below. Loop liveness during provisioning is pinned by
    # test_worker_auto_provision_failure_degrades_and_worker_survives.
    worker._maybe_start_auto_provision()
    assert worker._provision_thread is not None
    worker._provision_thread.join(timeout=5.0)
    assert worker.provision_state["state"] == "done"
    assert calls == [(cfg.models_dir, ["faces"])]
    assert "face" not in worker._backend_cache  # miss dropped -> re-probe
    assert worker._backend_failed_at == {}


def test_worker_auto_provision_failure_degrades_and_worker_survives(tmp_path, monkeypatch):
    """A failing download (egress-denied host) leaves state 'failed: ...',
    counts one error, and the worker keeps cycling normally."""
    from smartgallery_ai import worker as W

    _make_db(str(tmp_path / "g.sqlite"))
    os.makedirs(tmp_path / "models", exist_ok=True)
    cfg = _cfg(tmp_path, auto_provision=True, semantic_backend="none",
               visual_backend="none", face_backend="auto",
               segmenter_backend="none", critic_backend="none")
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
    assert installed == ["torch", "torchvision", "transformers"]
    assert calls[2] == ["transformers"]
    assert invalidated == [1]


@pytest.mark.parametrize("requirement", ["torch", "torchvision"])
def test_pip_args_steer_torch_by_hardware(monkeypatch, requirement):
    """torch AND torchvision wheel choice: NVIDIA present -> CUDA-capable
    (PyPI on Linux, cu-index on Windows); absent -> CPU index;
    AI_DAM_DEVICE=cpu forces CPU even with hardware. Both must get the
    SAME steering or torchvision's compiled ops fail to register against
    the installed torch; other requirements pass through."""
    monkeypatch.delenv("AI_DAM_DEVICE", raising=False)
    monkeypatch.setattr(P.sys, "platform", "linux")

    monkeypatch.setattr(P, "cuda_hardware_present", lambda: False)
    assert P._pip_args_for(requirement) == [
        requirement, "--index-url", P._TORCH_CPU_INDEX]

    monkeypatch.setattr(P, "cuda_hardware_present", lambda: True)
    assert P._pip_args_for(requirement) == [requirement]

    monkeypatch.setattr(P.sys, "platform", "win32")
    monkeypatch.setattr(P, "torch_cuda_index",
                        lambda: "https://download.pytorch.org/whl/cuTEST")
    assert P._pip_args_for(requirement) == [
        requirement, "--index-url", "https://download.pytorch.org/whl/cuTEST"]

    monkeypatch.setenv("AI_DAM_DEVICE", "cpu")
    assert P._pip_args_for(requirement) == [
        requirement, "--index-url", P._TORCH_CPU_INDEX]

    assert P._pip_args_for("timm") == ["timm"]


def test_every_torch_group_also_declares_torchvision():
    """Registry contract: any group whose runtime needs torch must install
    torchvision alongside it, and BEFORE packages that transitively depend
    on torchvision (open_clip_torch/timm/mobile_sam). Installs run
    sequentially, so a steered torchvision must already be present or pip
    resolves the transitive dependency from PyPI -- pairing a CPU-index
    torch with a PyPI torchvision, and every import dies with
    'operator torchvision::nms does not exist'."""
    torchvision_dependents = {"open_clip", "timm", "mobile_sam"}
    for group in P.GROUPS:
        probes = [probe for probe, _ in group.runtime]
        if "torch" in probes:
            assert "torchvision" in probes, (
                f"group {group.name!r} installs torch without torchvision")
            for dependent in torchvision_dependents & set(probes):
                assert probes.index("torchvision") < probes.index(dependent), (
                    f"group {group.name!r} must install torchvision before "
                    f"{dependent!r} or pip pulls an unpaired PyPI build")


def test_hub_bars_silenced_disables_bars_then_restores():
    """Inside the context the hub's console progress bars are off; on exit
    the previous state comes back."""
    hub_utils = pytest.importorskip("huggingface_hub.utils")
    if hub_utils.are_progress_bars_disabled():
        pytest.skip("progress bars globally disabled in this environment")
    with P._hub_bars_silenced():
        assert hub_utils.are_progress_bars_disabled()
    assert not hub_utils.are_progress_bars_disabled()


def test_hub_bars_silenced_keeps_pre_disabled_state():
    """When bars were already disabled the context changes nothing and does
    not re-enable them on exit."""
    hub_utils = pytest.importorskip("huggingface_hub.utils")
    if hub_utils.are_progress_bars_disabled():
        pytest.skip("progress bars globally disabled in this environment")
    hub_utils.disable_progress_bars()
    try:
        with P._hub_bars_silenced():
            assert hub_utils.are_progress_bars_disabled()
        assert hub_utils.are_progress_bars_disabled()
    finally:
        hub_utils.enable_progress_bars()


def test_provision_silences_hub_bars_only_in_structured_progress_mode(tmp_path, monkeypatch):
    """With a structured progress callback the default HF downloaders run
    inside the hub-bar silencer (the caller owns console rendering); the
    bar-rendering CLI path -- no callback -- leaves the hub's bars on."""
    from types import SimpleNamespace
    monkeypatch.setattr(P.importlib.util, "find_spec",
                        lambda name: SimpleNamespace(origin="stub.py"))
    entered = []

    @contextlib.contextmanager
    def recording_silencer():
        entered.append(True)
        yield

    def fake_hf_snapshot(repo, dest):
        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, "model.safetensors"), "wb") as fh:
            fh.write(b"weights")

    monkeypatch.setattr(P, "_hub_bars_silenced", recording_silencer)
    monkeypatch.setattr(P, "_download_hf_snapshot", fake_hf_snapshot)

    P.provision(str(tmp_path / "with_progress"), ["visual"],
                log=lambda m: None, progress=lambda e: None)
    assert entered == [True]

    entered.clear()
    P.provision(str(tmp_path / "no_progress"), ["visual"], log=lambda m: None)
    assert entered == []


def test_provision_installs_runtime_before_weights(tmp_path, monkeypatch):
    """provision() makes the group loadable end to end: missing runtime
    packages install first, then weights download."""
    events = []
    from types import SimpleNamespace
    monkeypatch.setattr(
        P.importlib.util, "find_spec",
        lambda name: None if name in ("torch", "torchvision", "transformers")
        else SimpleNamespace(origin="stub.py"))

    def runner(args):
        events.append(("pip", tuple(args)))

    downloaders = _fake_downloaders({})

    def snapshot(repo, dest):
        events.append(("weights", repo))
        downloaders["hf_snapshot"](repo, dest)

    result = P.provision(str(tmp_path), ["visual"], log=lambda m: None,
                         downloaders={**downloaders, "hf_snapshot": snapshot},
                         pip_runner=runner)
    assert result["installed"] == ["torch", "torchvision", "transformers"]
    assert [e[0] for e in events] == ["pip", "pip", "pip", "weights"]


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
    from types import SimpleNamespace
    monkeypatch.setattr(
        P.importlib.util, "find_spec",
        lambda name: None if name == "open_clip"
        else SimpleNamespace(origin="stub.py"))
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


# --- progress reporting -------------------------------------------------------


def test_copy_with_progress_reports_running_totals():
    """Every chunk reports cumulative bytes against the total."""
    import io

    src = io.BytesIO(b"x" * 2500)
    dst = io.BytesIO()
    seen = []
    P._copy_with_progress(src, dst, 2500, lambda done, total: seen.append((done, total)),
                          chunk_size=1000)
    assert dst.getvalue() == b"x" * 2500
    assert seen == [(1000, 2500), (2000, 2500), (2500, 2500)]


def test_provision_emits_structured_progress_events(tmp_path, monkeypatch):
    """provision() narrates its work: runtime install start/done, then
    artifact start/done, in execution order."""
    from types import SimpleNamespace
    monkeypatch.setattr(
        P.importlib.util, "find_spec",
        lambda name: None if name == "transformers"
        else SimpleNamespace(origin="stub.py"))
    events = []
    P.provision(str(tmp_path), ["visual"], log=lambda m: None,
                downloaders=_fake_downloaders({}),
                pip_runner=lambda args: None,
                progress=events.append)
    assert [(e["kind"], e["phase"], e["item"]) for e in events] == [
        ("runtime", "start", "transformers"),
        ("runtime", "done", "transformers"),
        ("artifact", "start", "dinov2-small"),
        ("artifact", "done", "dinov2-small"),
    ]
    assert events[2]["size"] == "90 MB"


def test_worker_folds_progress_events_into_served_state(tmp_path):
    """The worker's event handler keeps /status-visible state current —
    item, human-readable byte detail, completed list — and swaps the dict
    instead of mutating it (another thread snapshots it)."""
    _make_db(str(tmp_path / "g.sqlite"))
    cfg = _cfg(tmp_path, semantic_backend="none", visual_backend="none",
               face_backend="none", segmenter_backend="none", critic_backend="none")
    worker = AIWorker(cfg, cfg.db_path, poll_interval=0.05, batch_size=10)
    worker.provision_state = {"state": "downloading", "groups": ["semantic"]}
    snapshot = worker.provision_state

    worker._on_provision_event({"kind": "artifact", "phase": "start",
                                "item": "open_clip/ViT-B-32_laion2b_s34b_b79k.bin",
                                "size": "605 MB"})
    assert worker.provision_state["current"].endswith(".bin")
    assert "605 MB" in worker.provision_state["detail"]

    worker._on_provision_event({"kind": "artifact", "phase": "bytes",
                                "item": "open_clip/ViT-B-32_laion2b_s34b_b79k.bin",
                                "bytes_done": 302_500_000, "bytes_total": 605_000_000})
    assert "(50%)" in worker.provision_state["detail"]

    worker._on_provision_event({"kind": "artifact", "phase": "done",
                                "item": "open_clip/ViT-B-32_laion2b_s34b_b79k.bin"})
    assert worker.provision_state["done"] == ["open_clip/ViT-B-32_laion2b_s34b_b79k.bin"]
    assert worker.provision_state["current"] is None
    assert snapshot == {"state": "downloading", "groups": ["semantic"]}  # never mutated


def test_worker_start_makes_info_logging_visible(tmp_path):
    """start() attaches a handler so provisioning progress reaches the
    console even though the host app never configures logging."""
    import logging

    _make_db(str(tmp_path / "g.sqlite"))
    cfg = _cfg(tmp_path, auto_provision=False, semantic_backend="none",
               visual_backend="none", face_backend="none",
               segmenter_backend="none", critic_backend="none")
    worker = AIWorker(cfg, cfg.db_path, poll_interval=0.05, batch_size=10)

    # Simulate production: no root handlers (pytest installs its own, which
    # start() correctly treats as "logging already configured" and defers to).
    root = logging.getLogger()
    pkg = logging.getLogger("smartgallery_ai")
    saved_root, saved_pkg = root.handlers[:], pkg.handlers[:]
    saved_level = pkg.level
    root.handlers, pkg.handlers = [], []
    try:
        worker.start()
        assert pkg.handlers, "start() must attach a console handler"
        assert logging.getLogger("smartgallery_ai.worker").isEnabledFor(logging.INFO)
    finally:
        worker.stop(timeout=2.0)
        root.handlers, pkg.handlers = saved_root, saved_pkg
        pkg.setLevel(saved_level)


def test_namespace_package_shadow_counts_as_missing():
    """A bare directory on sys.path materializes as a namespace package
    (spec.origin is None); the runtime probe must treat that as NOT
    installed, or a stray folder suppresses the install and the backend
    fails later."""
    from types import SimpleNamespace

    group = next(g for g in P.GROUPS if g.name == "semantic")
    real = P.importlib.util.find_spec

    def shadowed(name):
        if name == "open_clip":
            return SimpleNamespace(origin=None, submodule_search_locations=["/repo/open_clip"])
        return real(name)

    import unittest.mock as mock
    with mock.patch.object(P.importlib.util, "find_spec", shadowed):
        assert ("open_clip", "open_clip_torch") in P.runtime_missing(group)


def test_explicitly_constructed_config_never_auto_provisions(tmp_path):
    """AIConfig() is inert: auto_provision defaults False on the dataclass
    (from_env flips it on), so tests and embedders constructing configs by
    hand can never reach the network by accident."""
    assert AIConfig().auto_provision is False
    assert AIConfig(enabled=True).auto_provision is False


def test_provision_refuses_empty_models_dir():
    """An empty models_dir would scatter weights relative to the working
    directory; provision() refuses it outright."""
    with pytest.raises(P.ProvisionError, match="models_dir is required"):
        P.provision("", ["faces"], log=lambda m: None)


# --- GPU self-heal: CPU-build torch on CUDA hardware ---------------------------


def test_torch_cuda_reinstall_needed_matrix(monkeypatch):
    """The swap triggers only for a +cpu torch build on non-mac CUDA
    hardware without the AI_DAM_DEVICE=cpu opt-out; absent torch or a
    CUDA/plain build never triggers it."""
    monkeypatch.delenv("AI_DAM_DEVICE", raising=False)
    monkeypatch.setattr(P.sys, "platform", "linux")
    monkeypatch.setattr(P, "cuda_hardware_present", lambda: True)
    monkeypatch.setattr(P.importlib.metadata, "version", lambda name: "2.13.0+cpu")
    assert P.torch_cuda_reinstall_needed() is True

    monkeypatch.setattr(P.importlib.metadata, "version", lambda name: "2.13.0+cu126")
    assert P.torch_cuda_reinstall_needed() is False
    monkeypatch.setattr(P.importlib.metadata, "version", lambda name: "2.13.0")
    assert P.torch_cuda_reinstall_needed() is False

    monkeypatch.setattr(P.importlib.metadata, "version", lambda name: "2.13.0+cpu")
    monkeypatch.setenv("AI_DAM_DEVICE", "cpu")
    assert P.torch_cuda_reinstall_needed() is False
    monkeypatch.delenv("AI_DAM_DEVICE")

    monkeypatch.setattr(P, "cuda_hardware_present", lambda: False)
    assert P.torch_cuda_reinstall_needed() is False
    monkeypatch.setattr(P, "cuda_hardware_present", lambda: True)

    monkeypatch.setattr(P.sys, "platform", "darwin")
    assert P.torch_cuda_reinstall_needed() is False  # macOS torch has no CUDA variant

    monkeypatch.setattr(P.sys, "platform", "linux")

    def _missing(name):
        raise P.importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(P.importlib.metadata, "version", _missing)
    assert P.torch_cuda_reinstall_needed() is False


def test_provision_swaps_cpu_torch_for_cuda_when_unimported(tmp_path, monkeypatch):
    """With a +cpu torch, CUDA hardware, and torch not yet imported,
    provision() uninstalls the pair and the missing-package loop reinstalls
    both with hardware steering."""
    from types import SimpleNamespace
    monkeypatch.setattr(P, "torch_cuda_reinstall_needed", lambda: True)
    monkeypatch.setattr(P, "cuda_hardware_present", lambda: True)
    monkeypatch.setattr(P.sys, "platform", "linux")
    monkeypatch.delitem(P.sys.modules, "torch", raising=False)

    uninstalled = []
    installs = []

    def fake_find_spec(name):
        # torch/torchvision read as missing once the uninstall happened
        if name in ("torch", "torchvision") and uninstalled:
            return None
        return SimpleNamespace(origin="stub.py")

    monkeypatch.setattr(P.importlib.util, "find_spec", fake_find_spec)
    result = P.provision(
        str(tmp_path), ["visual"], log=lambda m: None,
        downloaders=_fake_downloaders({}),
        pip_runner=lambda args: installs.append(args),
        pip_uninstaller=lambda pkgs: uninstalled.append(pkgs),
    )
    assert uninstalled == [["torch", "torchvision"]]
    assert ["torch"] in installs and ["torchvision"] in installs
    assert set(result["installed"]) >= {"torch", "torchvision"}


def test_provision_only_advises_when_torch_already_imported(tmp_path, monkeypatch):
    """A loaded torch pins its files (locked DLLs on Windows), so the swap
    must not uninstall underneath it: provision() logs a restart advisory
    and leaves the packages alone."""
    from types import SimpleNamespace
    monkeypatch.setattr(P, "torch_cuda_reinstall_needed", lambda: True)
    monkeypatch.setitem(P.sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(P.importlib.util, "find_spec",
                        lambda name: SimpleNamespace(origin="stub.py"))

    lines = []
    result = P.provision(
        str(tmp_path), ["visual"], log=lines.append,
        downloaders=_fake_downloaders({}),
        pip_runner=lambda args: (_ for _ in ()).throw(AssertionError("no installs expected")),
        pip_uninstaller=lambda pkgs: (_ for _ in ()).throw(AssertionError("must not uninstall")),
    )
    assert any("restart the app to switch to CUDA" in line for line in lines)
    assert result["installed"] == []


def test_provision_groups_for_includes_cuda_swap_groups(tmp_path, monkeypatch):
    """A fully-provisioned torch group still auto-provisions when the
    installed torch is a CPU build on CUDA hardware -- that is how the
    swap reaches machines with nothing else missing."""
    from types import SimpleNamespace
    _make_db(str(tmp_path / "g.sqlite"))
    weights_dir = tmp_path / "models" / "dinov2-small"
    weights_dir.mkdir(parents=True)
    (weights_dir / "model.safetensors").write_bytes(b"w")
    cfg = _cfg(tmp_path, semantic_backend="none", visual_backend="auto",
               face_backend="none", segmenter_backend="none", critic_backend="none")

    monkeypatch.setattr(P.importlib.util, "find_spec",
                        lambda name: SimpleNamespace(origin="stub.py"))
    monkeypatch.setattr(P, "torch_cuda_reinstall_needed", lambda: False)
    assert provision_groups_for(cfg) == []

    monkeypatch.setattr(P, "torch_cuda_reinstall_needed", lambda: True)
    assert provision_groups_for(cfg) == ["visual"]


# --- pip operations in pip-less (uv) environments ------------------------------


def _pip_proc(returncode=0, stderr=""):
    from types import SimpleNamespace
    return SimpleNamespace(returncode=returncode, stderr=stderr, stdout="")


def test_pip_runner_uses_python_m_pip_when_available(monkeypatch):
    """The normal environment needs exactly one subprocess call."""
    calls = []
    monkeypatch.setattr(P.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or _pip_proc())
    P._default_pip_runner(["timm"])
    assert len(calls) == 1
    assert calls[0][1:] == ["-m", "pip", "install", "--quiet", "timm"]


def test_pip_runner_falls_back_to_uv_pip_in_pipless_venv(monkeypatch):
    """A uv-created venv has no pip module; the runner retries the same
    operation through `uv pip --python <this python>` (the user's exact
    'No module named pip' failure)."""
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[1:3] == ["-m", "pip"]:
            return _pip_proc(1, "python.exe: No module named pip")
        return _pip_proc()

    monkeypatch.setattr(P.subprocess, "run", fake_run)
    monkeypatch.setattr(P.shutil, "which",
                        lambda name: "/usr/bin/uv" if name == "uv" else None)
    P._default_pip_uninstaller(["torch", "torchvision"])
    assert calls[1][0] == "/usr/bin/uv"
    assert calls[1][1:4] == ["pip", "uninstall", "--quiet"]
    assert "--python" in calls[1]


def test_pip_runner_bootstraps_ensurepip_without_uv(monkeypatch):
    """No uv on PATH: bootstrap pip via ensurepip once, then retry."""
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[1:3] == ["-m", "pip"] and len(calls) == 1:
            return _pip_proc(1, "No module named pip")
        return _pip_proc()

    monkeypatch.setattr(P.subprocess, "run", fake_run)
    monkeypatch.setattr(P.shutil, "which", lambda name: None)
    P._default_pip_runner(["timm"])
    assert calls[1][1:3] == ["-m", "ensurepip"]
    assert calls[2][1:3] == ["-m", "pip"]


def test_pip_runner_raises_when_every_fallback_fails(monkeypatch):
    """All routes exhausted -> ProvisionError carrying stderr."""
    monkeypatch.setattr(P.subprocess, "run",
                        lambda cmd, **kw: _pip_proc(1, "No module named pip"))
    monkeypatch.setattr(P.shutil, "which", lambda name: None)
    with pytest.raises(P.ProvisionError, match="No module named pip"):
        P._default_pip_runner(["timm"])


# --- CUDA wheel index by GPU generation ----------------------------------------


def test_torch_cuda_index_picks_by_compute_cap_and_driver(monkeypatch):
    """Pre-Blackwell cards keep cu126; Blackwell (cc >= 10) gets the
    newest sm_120 build the driver supports; unknown driver falls back to
    cu130; AI_DAM_CUDA_INDEX overrides everything."""
    monkeypatch.delenv("AI_DAM_CUDA_INDEX", raising=False)

    monkeypatch.setattr(P, "_cuda_compute_capability", lambda: 8.9)
    assert P.torch_cuda_index().endswith("/cu126")
    monkeypatch.setattr(P, "_cuda_compute_capability", lambda: None)
    assert P.torch_cuda_index().endswith("/cu126")

    monkeypatch.setattr(P, "_cuda_compute_capability", lambda: 12.0)
    monkeypatch.setattr(P, "_driver_cuda_version", lambda: 13.5)
    assert P.torch_cuda_index().endswith("/cu132")
    monkeypatch.setattr(P, "_driver_cuda_version", lambda: 13.0)
    assert P.torch_cuda_index().endswith("/cu130")
    monkeypatch.setattr(P, "_driver_cuda_version", lambda: 12.9)
    assert P.torch_cuda_index().endswith("/cu129")
    monkeypatch.setattr(P, "_driver_cuda_version", lambda: None)
    assert P.torch_cuda_index().endswith("/cu130")

    monkeypatch.setenv("AI_DAM_CUDA_INDEX", "https://example.test/whl/custom")
    assert P.torch_cuda_index() == "https://example.test/whl/custom"


def test_reinstall_needed_for_wrong_generation_cuda_build(monkeypatch):
    """A CUDA build from the wrong generation's index (kernels missing for
    this GPU: cudaErrorNoKernelImageForDevice) counts as swap-needed on
    Windows; the matching build does not; Linux CUDA builds are left
    alone (they came from PyPI or the user's own choice)."""
    monkeypatch.delenv("AI_DAM_DEVICE", raising=False)
    monkeypatch.setattr(P, "cuda_hardware_present", lambda: True)
    monkeypatch.setattr(P.sys, "platform", "win32")
    monkeypatch.setattr(P, "torch_cuda_index",
                        lambda: "https://download.pytorch.org/whl/cu130")

    monkeypatch.setattr(P.importlib.metadata, "version",
                        lambda name: "2.13.0+cu126")
    assert P.torch_cuda_reinstall_needed() is True

    monkeypatch.setattr(P.importlib.metadata, "version",
                        lambda name: "2.13.0+cu130")
    assert P.torch_cuda_reinstall_needed() is False

    monkeypatch.setattr(P.sys, "platform", "linux")
    monkeypatch.setattr(P.importlib.metadata, "version",
                        lambda name: "2.13.0+cu126")
    assert P.torch_cuda_reinstall_needed() is False


def test_cuda_summary_absent_without_nvidia_driver(monkeypatch):
    """No nvidia-smi on PATH -> no GPU inventory (the boot log then says
    'no NVIDIA GPU detected')."""
    monkeypatch.setattr(P, "cuda_hardware_present", lambda: False)
    assert P.cuda_summary() is None
