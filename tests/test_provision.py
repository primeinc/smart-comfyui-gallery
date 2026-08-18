"""Tests for smartgallery_ai.provision and the worker's async
auto-provisioning: group resolution, plan rendering, idempotent skip,
hash verification, downloader dispatch, the backend-key -> group mapping,
and the worker thread's success/failure/disabled behavior. Entirely
network-free: every downloader is injected or monkeypatched."""

from __future__ import annotations

import contextlib
import importlib.util as IU
import io
import logging
import os
import shutil as _sh
import sqlite3
import time
import zipfile
from types import SimpleNamespace
from unittest import mock

import click
import pytest
from flask import Flask

from smartgallery_ai import AIConfig
from smartgallery_ai import provision as P
from smartgallery_ai import service as S
from smartgallery_ai import worker as W
from smartgallery_ai.__main__ import main
from smartgallery_ai.embedders import pick_torch_device
from smartgallery_ai.provision import _download_zip_member
from smartgallery_ai.schema import init_schema
from smartgallery_ai.service import create_ai_blueprint, set_worker
from smartgallery_ai.worker import AIWorker, _ClickConsoleHandler, provision_groups_for


@pytest.fixture(autouse=True)
def _plentiful_disk(monkeypatch):
    """provision()'s disk preflight reads the real volume; the suite must
    not depend on this machine's free space. Preflight tests re-patch."""
    monkeypatch.setattr(P.shutil, "disk_usage", lambda _p: type("U", (), {"free": 200 * 1024**3})())


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
    # Checkpoint DIRECTORIES, not single files: transformers loads a
    # snapshot, and smartgallery_ai.models resolves a model ref to one of
    # these directory names under the models dir.
    assert "Qwen3-VL-2B-Instruct" in dests
    assert "distil-qwen3-4b-text2sql" in dests


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
        if dest.endswith(".zip"):
            # unzip_member artifacts fetch a zip and keep one member; serve
            # a real zip holding every member any registry artifact names
            import zipfile

            with zipfile.ZipFile(dest, "w") as zf:
                for member in {a.unzip_member for g in P.GROUPS for a in g.artifacts if a.unzip_member}:
                    zf.writestr(member, content)
            return
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
    pack = tmp_path / "insightface" / "models" / "antelopev2"
    pack.mkdir(parents=True)
    (pack / "glintr100.onnx").write_bytes(b"x")

    result = P.provision(
        str(tmp_path), ["faces", "visual"], log=lambda _m: None, downloaders=_fake_downloaders(written)
    )

    assert result["skipped"] == [
        "face_detection_yunet_2023mar.onnx",
        "face_recognition_sface_2021dec.onnx",
        "insightface/models/antelopev2",
    ]
    assert result["downloaded"] == ["dinov2-small"]
    dest = str(tmp_path / "dinov2-small")
    assert written[dest] == ("hf_snapshot", "facebook/dinov2-small")
    assert os.path.isfile(os.path.join(dest, "model.safetensors"))


def test_provision_hash_mismatch_deletes_and_raises(tmp_path):
    """A pinned artifact whose downloaded bytes hash differently is
    removed and the run fails loudly, never leaving a poisoned file."""
    written: dict = {}
    with pytest.raises(P.ProvisionError, match="SHA-256 mismatch"):
        P.provision(
            str(tmp_path), ["faces"], log=lambda _m: None, downloaders=_fake_downloaders(written, content=b"tampered")
        )
    assert not os.path.exists(tmp_path / "face_detection_yunet_2023mar.onnx")


def test_provision_download_failure_names_artifact(tmp_path):
    """A downloader exception surfaces as ProvisionError naming the dest."""

    def boom(_u, _dest):
        raise OSError("connection refused")

    with pytest.raises(P.ProvisionError, match="face_detection_yunet_2023mar.onnx"):
        P.provision(str(tmp_path), ["faces"], log=lambda _m: None, downloaders={"url": boom})


# --- worker mapping + async auto-provisioning ---------------------------------


def _cfg(tmp_path, **overrides) -> AIConfig:
    defaults = {
        "enabled": True,
        "base_path": str(tmp_path),
        "db_path": str(tmp_path / "g.sqlite"),
        "models_dir": str(tmp_path / "models"),
        "cache_dir": str(tmp_path / "cache"),
        "ephemeral_index": True,
    }
    defaults.update(overrides)
    return AIConfig(**defaults)


def _make_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE files (id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE,"
        " mtime REAL NOT NULL, name TEXT NOT NULL, type TEXT)"
    )
    init_schema(conn)
    conn.commit()
    conn.close()


def test_provision_groups_for_maps_backends_and_critic_pulls_semantic(tmp_path):
    """Each real-backend selector maps to its group; qwen-vl critic adds
    the semantic (grounding-gate) weights; 'none'/'stub' map to nothing."""
    os.makedirs(tmp_path / "models", exist_ok=True)
    cfg = _cfg(
        tmp_path,
        semantic_backend="none",
        visual_backend="none",
        face_backend="none",
        segmenter_backend="none",
        critic_backend="vlm",
    )
    assert provision_groups_for(cfg) == ["critic", "semantic"]

    cfg2 = _cfg(
        tmp_path,
        semantic_backend="stub",
        visual_backend="stub",
        face_backend="stub",
        segmenter_backend="stub",
        critic_backend="stub",
    )
    assert provision_groups_for(cfg2) == []

    cfg3 = _cfg(
        tmp_path,
        semantic_backend="none",
        visual_backend="auto",
        face_backend="auto",
        segmenter_backend="none",
        critic_backend="none",
    )
    assert provision_groups_for(cfg3) == ["visual", "faces"]


def test_provision_groups_for_omits_groups_already_on_disk(tmp_path):
    """A group whose artifacts all exist is not re-provisioned."""
    models = tmp_path / "models"
    os.makedirs(models)
    (models / "face_detection_yunet_2023mar.onnx").write_bytes(b"x")
    (models / "face_recognition_sface_2021dec.onnx").write_bytes(b"x")
    pack = models / "insightface" / "models" / "antelopev2"
    pack.mkdir(parents=True)
    (pack / "glintr100.onnx").write_bytes(b"x")
    cfg = _cfg(
        tmp_path,
        semantic_backend="none",
        visual_backend="none",
        face_backend="auto",
        segmenter_backend="none",
        critic_backend="none",
    )
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

    _make_db(str(tmp_path / "g.sqlite"))
    os.makedirs(tmp_path / "models", exist_ok=True)
    cfg = _cfg(
        tmp_path,
        auto_provision=True,
        semantic_backend="none",
        visual_backend="none",
        face_backend="auto",
        segmenter_backend="none",
        critic_backend="none",
    )
    worker = AIWorker(cfg, cfg.db_path, poll_interval=0.05, batch_size=10)

    calls = []

    def fake_provision(models_dir, groups, force=False, log=print, downloaders=None, progress=None):
        # force/log/downloaders/progress accepted (log, progress kept named
        # since the real call site passes them by keyword) only for
        # provision()'s call-signature compatibility; this stub ignores them.
        del force, log, downloaders, progress
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

    _make_db(str(tmp_path / "g.sqlite"))
    os.makedirs(tmp_path / "models", exist_ok=True)
    cfg = _cfg(
        tmp_path,
        auto_provision=True,
        semantic_backend="none",
        visual_backend="none",
        face_backend="auto",
        segmenter_backend="none",
        critic_backend="none",
    )
    worker = AIWorker(cfg, cfg.db_path, poll_interval=0.05, batch_size=10)

    def refuse(*_a, **_k):
        raise P.ProvisionError("connection refused")

    monkeypatch.setattr(W.provisioning, "provision", refuse)
    worker.start()
    try:
        assert _wait_until(lambda: str(worker.provision_state["state"]).startswith("failed"))
        assert _wait_until(lambda: worker.stats["cycles"] > 0)
        assert worker.stats["errors"] >= 1
    finally:
        worker.stop(timeout=2.0)


def test_worker_auto_provision_disabled_never_downloads(tmp_path, monkeypatch):
    """AI_DAM_AUTO_PROVISION=false (config auto_provision=False) is the
    strict no-egress mode: no provisioning attempt is ever made."""

    _make_db(str(tmp_path / "g.sqlite"))
    os.makedirs(tmp_path / "models", exist_ok=True)
    cfg = _cfg(
        tmp_path,
        auto_provision=False,
        semantic_backend="auto",
        visual_backend="auto",
        face_backend="auto",
        segmenter_backend="auto",
        critic_backend="auto",
    )
    worker = AIWorker(cfg, cfg.db_path, poll_interval=0.05, batch_size=10)

    def must_not_run(*_a, **_k):
        raise AssertionError("provisioning ran despite auto_provision=False")

    monkeypatch.setattr(W.provisioning, "provision", must_not_run)
    worker.start()
    try:
        assert worker.provision_state == {"state": "disabled", "groups": []}
        assert _wait_until(lambda: worker.stats["cycles"] > 0)
    finally:
        worker.stop(timeout=2.0)


def test_worker_provision_state_exposed_by_status_endpoint(tmp_path):
    """/status reports the worker's provisioning state so the UI can show
    download progress instead of a bare empty tab."""

    _make_db(str(tmp_path / "g.sqlite"))
    os.makedirs(tmp_path / "models", exist_ok=True)
    cfg = _cfg(
        tmp_path,
        semantic_backend="none",
        visual_backend="none",
        face_backend="none",
        segmenter_backend="none",
        critic_backend="none",
    )
    worker = AIWorker(cfg, cfg.db_path, poll_interval=0.05, batch_size=10)
    worker.provision_state = {"state": "downloading", "groups": ["semantic"]}
    set_worker(worker)
    try:
        app = Flask(__name__)
        app.register_blueprint(create_ai_blueprint(cfg), url_prefix="/aidam")
        status = app.test_client().get("/aidam/status").get_json()
        assert status["worker"]["provisioning"] == {"state": "downloading", "groups": ["semantic"]}
    finally:
        set_worker(None)


def test_invalidate_backend_probe_cache_clears_registered_caches():
    """The worker's post-provision hook empties every blueprint's probe
    cache so /status re-probes instead of serving a stale False."""

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

    rc = main(["provision", "faces", "--models-dir", str(tmp_path), "--list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "MISSING" in out
    assert "face_detection_yunet_2023mar.onnx" in out
    assert not os.listdir(tmp_path)


def test_cli_provision_unknown_group_exits_nonzero(tmp_path, capsys):

    rc = main(["provision", "bogus", "--models-dir", str(tmp_path), "--list"])
    assert rc == 2
    assert "unknown group" in capsys.readouterr().out


# --- runtime package installation ---------------------------------------------


def test_runtime_missing_reports_only_unimportable(monkeypatch):
    """Probes that import are never reinstalled; only genuinely missing
    modules produce install work."""

    group = next(g for g in P.GROUPS if g.name == "semantic")
    real_find = IU.find_spec

    monkeypatch.setattr(P.importlib.util, "find_spec", lambda name: None if name == "open_clip" else real_find(name))
    assert [req for _, req in P.runtime_missing(group)] == ["open_clip_torch"]


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
            assert "torchvision" in probes, f"group {group.name!r} installs torch without torchvision"
            for dependent in torchvision_dependents & set(probes):
                assert probes.index("torchvision") < probes.index(dependent), (
                    f"group {group.name!r} must install torchvision before "
                    f"{dependent!r} or pip pulls an unpaired PyPI build"
                )


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
    monkeypatch.setattr(P.importlib.util, "find_spec", lambda _name: SimpleNamespace(origin="stub.py"))
    entered = []

    @contextlib.contextmanager
    def recording_silencer():
        entered.append(True)
        yield

    def fake_hf_snapshot(_repo, dest):
        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, "model.safetensors"), "wb") as fh:
            fh.write(b"weights")

    monkeypatch.setattr(P, "_hub_bars_silenced", recording_silencer)
    monkeypatch.setattr(P, "_download_hf_snapshot", fake_hf_snapshot)

    P.provision(str(tmp_path / "with_progress"), ["visual"], log=lambda _m: None, progress=lambda _e: None)
    assert entered == [True]

    entered.clear()
    P.provision(str(tmp_path / "no_progress"), ["visual"], log=lambda _m: None)
    assert entered == []


def test_provision_groups_for_includes_runtime_missing_groups(tmp_path, monkeypatch):
    """A group whose weights exist but whose runtime cannot import is still
    auto-provisioned (the runtime is installable)."""
    models = tmp_path / "models"
    os.makedirs(models / "dinov2-small", exist_ok=True)
    (models / "dinov2-small" / "model.safetensors").write_bytes(b"x")

    monkeypatch.setattr(W.provisioning, "runtime_missing", lambda g: [("torch", "torch")] if g.name == "visual" else [])
    cfg = _cfg(
        tmp_path,
        semantic_backend="none",
        visual_backend="auto",
        face_backend="none",
        segmenter_backend="none",
        critic_backend="none",
    )
    assert provision_groups_for(cfg) == ["visual"]


def test_format_plan_lists_runtime_rows(tmp_path, monkeypatch):
    """The plan shows runtime requirements with present/MISSING state."""
    monkeypatch.setattr(
        P.importlib.util, "find_spec", lambda name: None if name == "open_clip" else SimpleNamespace(origin="stub.py")
    )
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

    fake_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True), backends=SimpleNamespace())
    fake_cpu = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), backends=SimpleNamespace())

    monkeypatch.delenv("AI_DAM_DEVICE", raising=False)
    assert pick_torch_device(fake_cuda) == "cuda"
    assert pick_torch_device(fake_cpu) == "cpu"
    monkeypatch.setenv("AI_DAM_DEVICE", "cpu")
    assert pick_torch_device(fake_cuda) == "cpu"


# --- progress reporting -------------------------------------------------------


def test_copy_with_progress_reports_running_totals():
    """Every chunk reports cumulative bytes against the total."""

    src = io.BytesIO(b"x" * 2500)
    dst = io.BytesIO()
    seen = []
    P._copy_with_progress(src, dst, 2500, lambda done, total: seen.append((done, total)), chunk_size=1000)
    assert dst.getvalue() == b"x" * 2500
    assert seen == [(1000, 2500), (2000, 2500), (2500, 2500)]


def test_provision_emits_structured_progress_events(tmp_path):
    """provision() narrates its work: artifact start/done, in execution
    order. There are no runtime events -- it installs nothing. uv owns
    dependency management; this owns model weights."""
    events = []
    P.provision(
        str(tmp_path), ["visual"], log=lambda _m: None, downloaders=_fake_downloaders({}), progress=events.append
    )
    assert [(e["kind"], e["phase"], e["item"]) for e in events] == [
        ("artifact", "start", "dinov2-small"),
        ("artifact", "done", "dinov2-small"),
    ]
    assert events[0]["size"] == "90 MB"
    assert not any(e["kind"] == "runtime" for e in events)


def test_worker_folds_progress_events_into_served_state(tmp_path):
    """The worker's event handler keeps /status-visible state current —
    item, human-readable byte detail, completed list — and swaps the dict
    instead of mutating it (another thread snapshots it)."""
    _make_db(str(tmp_path / "g.sqlite"))
    cfg = _cfg(
        tmp_path,
        semantic_backend="none",
        visual_backend="none",
        face_backend="none",
        segmenter_backend="none",
        critic_backend="none",
    )
    worker = AIWorker(cfg, cfg.db_path, poll_interval=0.05, batch_size=10)
    worker.provision_state = {"state": "downloading", "groups": ["semantic"]}
    snapshot = worker.provision_state

    worker._on_provision_event(
        {"kind": "artifact", "phase": "start", "item": "open_clip/ViT-B-32_laion2b_s34b_b79k.bin", "size": "605 MB"}
    )
    assert worker.provision_state["current"].endswith(".bin")
    assert "605 MB" in worker.provision_state["detail"]

    worker._on_provision_event(
        {
            "kind": "artifact",
            "phase": "bytes",
            "item": "open_clip/ViT-B-32_laion2b_s34b_b79k.bin",
            "bytes_done": 302_500_000,
            "bytes_total": 605_000_000,
        }
    )
    assert "(50%)" in worker.provision_state["detail"]

    worker._on_provision_event(
        {"kind": "artifact", "phase": "done", "item": "open_clip/ViT-B-32_laion2b_s34b_b79k.bin"}
    )
    assert worker.provision_state["done"] == ["open_clip/ViT-B-32_laion2b_s34b_b79k.bin"]
    assert worker.provision_state["current"] is None
    assert snapshot == {"state": "downloading", "groups": ["semantic"]}  # never mutated


def test_worker_start_makes_info_logging_visible(tmp_path):
    """start() attaches a handler so provisioning progress reaches the
    console even though the host app never configures logging."""

    _make_db(str(tmp_path / "g.sqlite"))
    cfg = _cfg(
        tmp_path,
        auto_provision=False,
        semantic_backend="none",
        visual_backend="none",
        face_backend="none",
        segmenter_backend="none",
        critic_backend="none",
    )
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


def test_console_handler_prefixes_a_timestamp(capsys):
    """Every console line carries an HH:MM:SS timestamp so long indexing
    runs are readable; the message text follows unchanged."""

    handler = _ClickConsoleHandler()
    record = logging.LogRecord(
        name="smartgallery_ai.worker",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="[AIWorker] indexed: +50 hashed",
        args=(),
        exc_info=None,
    )
    handler.emit(record)

    out = capsys.readouterr().out
    assert "[AIWorker] indexed: +50 hashed" in out
    stamp = time.strftime("%H:%M", time.localtime(record.created))
    assert stamp in out  # HH:MM of the record's own timestamp leads the line


def test_namespace_package_shadow_counts_as_missing():
    """A bare directory on sys.path materializes as a namespace package
    (spec.origin is None); the runtime probe must treat that as NOT
    installed, or a stray folder suppresses the install and the backend
    fails later."""

    group = next(g for g in P.GROUPS if g.name == "semantic")
    real = P.importlib.util.find_spec

    def shadowed(name):
        if name == "open_clip":
            return SimpleNamespace(origin=None, submodule_search_locations=["/repo/open_clip"])
        return real(name)

    with mock.patch.object(P.importlib.util, "find_spec", shadowed):
        assert ("open_clip", "open_clip_torch") in P.runtime_missing(group)


def test_explicitly_constructed_config_never_auto_provisions():
    """AIConfig() is inert: auto_provision defaults False on the dataclass
    (from_env flips it on), so tests and embedders constructing configs by
    hand can never reach the network by accident."""
    assert AIConfig().auto_provision is False
    assert AIConfig(enabled=True).auto_provision is False


def test_provision_refuses_empty_models_dir():
    """An empty models_dir would scatter weights relative to the working
    directory; provision() refuses it outright."""
    with pytest.raises(P.ProvisionError, match="models_dir is required"):
        P.provision("", ["faces"], log=lambda _m: None)


# --- GPU self-heal: CPU-build torch on CUDA hardware ---------------------------


def test_cuda_summary_absent_without_nvidia_driver(monkeypatch):
    """No nvidia-smi on PATH -> no GPU inventory (the boot log then says
    'no NVIDIA GPU detected')."""
    monkeypatch.setattr(P, "cuda_hardware_present", lambda: False)
    assert P.cuda_summary() is None


def test_cuda_summary_lists_every_gpu_separately(monkeypatch):
    """A mixed-generation machine reports each card with its OWN name,
    compute capability, and VRAM -- never one card's name stitched to
    another card's capability."""
    monkeypatch.setattr(P, "cuda_hardware_present", lambda: True)
    monkeypatch.setattr(P, "_driver_cuda_version", lambda: 13.1)
    monkeypatch.setattr(P, "_cuda_compute_capability", lambda: 12.0)

    def fake_run(cmd, **_kw):
        assert ("--query-gpu=name,driver_version,compute_cap,memory.total,memory.used") in cmd
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=(
                "NVIDIA GeForce RTX 3070 Ti, 591.86, 8.6, 8192 MiB, 512 MiB\n"
                "NVIDIA GeForce RTX 5060 Ti, 591.86, 12.0, 16384 MiB, 15020 MiB\n"
            ),
        )

    monkeypatch.setattr(P.subprocess, "run", fake_run)
    summary = P.cuda_summary()
    assert [g["name"] for g in summary["gpus"]] == ["NVIDIA GeForce RTX 3070 Ti", "NVIDIA GeForce RTX 5060 Ti"]
    assert summary["gpus"][0]["compute_capability"] == 8.6
    assert summary["gpus"][1]["vram"] == "16384 MiB"
    assert summary["gpus"][1]["vram_used"] == "15020 MiB"
    assert summary["driver"] == "591.86"


def test_console_handler_falls_back_to_plain_after_console_failure(monkeypatch, capsys):
    """A broken Windows console handle (click raising OSError) permanently
    drops the handler to plain stderr writes: the line still lands, no
    handleError traceback, and click is not retried per line."""

    attempts = []

    def broken_echo(*_args, **_kwargs):
        attempts.append(1)
        raise OSError("Windows error: 6")

    monkeypatch.setattr(click, "echo", broken_echo)
    handler = _ClickConsoleHandler()

    def rec(msg):
        return logging.LogRecord(
            name="smartgallery_ai.worker",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )

    handler.emit(rec("[AIWorker] line one"))
    handler.emit(rec("[AIWorker] line two"))

    err = capsys.readouterr().err
    assert "[AIWorker] line one" in err
    assert "[AIWorker] line two" in err
    assert "Traceback" not in err
    assert attempts == [1]  # click tried once, then permanently plain


def test_worker_start_disables_propagation_to_a_late_root_logger(tmp_path):
    """After start() attaches its own console handler, a root logger
    configured LATER must not double-print every line (propagate off)."""

    _make_db(str(tmp_path / "g.sqlite"))
    cfg = _cfg(
        tmp_path,
        auto_provision=False,
        semantic_backend="none",
        visual_backend="none",
        face_backend="none",
        segmenter_backend="none",
        critic_backend="none",
    )
    worker = AIWorker(cfg, cfg.db_path, poll_interval=0.05, batch_size=10)

    root = logging.getLogger()
    pkg = logging.getLogger("smartgallery_ai")
    saved_root, saved_pkg = root.handlers[:], pkg.handlers[:]
    saved_level, saved_prop = pkg.level, pkg.propagate
    root.handlers, pkg.handlers = [], []
    try:
        worker.start()
        assert pkg.propagate is False
    finally:
        worker.stop(timeout=2.0)
        root.handlers, pkg.handlers = saved_root, saved_pkg
        pkg.setLevel(saved_level)
        pkg.propagate = saved_prop


def test_download_zip_member_extracts_one_file(tmp_path):
    """unzip_member artifacts: the zip is fetched, exactly the named
    member lands at dest, and the zip is removed."""

    src_zip = tmp_path / "pack.zip"
    with zipfile.ZipFile(src_zip, "w") as zf:
        zf.writestr("keep.onnx", b"weights-bytes")
        zf.writestr("drop.onnx", b"other")
    dest = tmp_path / "out" / "keep.onnx"
    dest.parent.mkdir()

    def fake_dl(url, path):
        assert url == "https://example.test/pack.zip"
        _sh.copyfile(src_zip, path)

    _download_zip_member(fake_dl, "https://example.test/pack.zip", "keep.onnx", str(dest))
    assert dest.read_bytes() == b"weights-bytes"
    assert not (tmp_path / "out" / "keep.onnx.zip").exists()
    assert not (tmp_path / "out" / "keep.onnx.part").exists()


# ---------------------------------------------------------------------------
# Disk-space preflight: downloads that cannot fit are refused up front
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected_mb"),
    [
        ("232 KB", 0),  # rounds below 1 MB; still nonzero bytes
        ("37 MB", 37),
        ("2.5 GB", 2560),
        ("407 MB (344 MB zip)", 751),  # archive + extraction coexist on disk
        ("see notes", 0),  # unparseable -> 0 (no false refusal)
    ],
)
def test_approx_bytes_parses_declared_sizes(text, expected_mb):
    assert P._approx_bytes(text) // (1024**2) == expected_mb


def _fake_artifact(size_text):
    return P.Artifact(dest="x.bin", approx_size=size_text, license="MIT", url="https://example.invalid/x.bin")


def test_disk_preflight_refuses_when_weights_cannot_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(P.shutil, "disk_usage", lambda _p: type("U", (), {"free": 500 * 1024**2})())
    with pytest.raises(P.ProvisionError) as exc:
        P._check_disk_space(str(tmp_path), [_fake_artifact("2.5 GB")])
    assert "not enough disk space" in str(exc.value)
    assert "AI_DAM_MODELS_DIR" in str(exc.value)


def test_disk_preflight_passes_with_room_and_skips_when_nothing_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(P.shutil, "disk_usage", lambda _p: type("U", (), {"free": 200 * 1024**3})())
    P._check_disk_space(str(tmp_path), [_fake_artifact("2.5 GB")])  # no raise
    # Nothing missing -> never even stats the volume.
    monkeypatch.setattr(P.shutil, "disk_usage", lambda _p: (_ for _ in ()).throw(AssertionError))
    P._check_disk_space(str(tmp_path), [])


def test_provision_surfaces_disk_refusal_before_any_download(tmp_path, monkeypatch):
    """The full provision() path refuses before its download loop runs."""
    monkeypatch.setattr(P.shutil, "disk_usage", lambda _p: type("U", (), {"free": 10 * 1024**2})())
    calls = []
    dl = {k: (lambda *a, **k2: calls.append(k)) for k in ("url", "hf_file", "hf_snapshot")}
    with pytest.raises(P.ProvisionError, match="not enough disk space"):
        P.provision(str(tmp_path / "models"), ["faces"], downloaders=dl)
    assert calls == []
