"""Lazy acquisition of everything the AI layer needs to run.

Makes a capability group fully loadable in two steps: pip-install its
missing runtime packages into the current environment (CPU-only torch
wheels; import caches refreshed so the running process picks them up),
then download its model weights into `models_dir` (default `.AImodels/`)
from their official Hugging Face repositories (via `huggingface_hub`'s
cached, resumable downloader) or, for the two OpenCV Zoo ONNX files,
their upstream Git-LFS media URLs (raw.githubusercontent serves LFS
pointer stubs, not content). Every artifact lands at the exact path the
backends load from, and small/medium artifacts are verified against
pinned SHA-256 digests.

This module is the ONLY download path in the system, and it runs in
exactly two ways: explicitly via the CLI below, or once, asynchronously,
by the worker's auto-provisioning thread (AI_DAM_AUTO_PROVISION, default
on). Backends and request handlers never fetch anything -- they load
local files or report BackendUnavailable -- so an egress-denied host
degrades gracefully and AI_DAM_AUTO_PROVISION=false restores a strictly
download-free process. Invoke:

    python -m smartgallery_ai provision --list
    python -m smartgallery_ai provision faces semantic
    python -m smartgallery_ai provision all --yes
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

# torch wheel selection: an NVIDIA GPU (detected via nvidia-smi on PATH)
# gets CUDA-capable wheels — PyPI's Linux torch bundles CUDA, Windows CUDA
# builds live only on the cu-index. Without a GPU the CPU index avoids the
# multi-GB CUDA payload. macOS torch on PyPI is already CPU/MPS.
# AI_DAM_DEVICE=cpu forces the CPU wheel regardless of hardware.
_TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
_TORCH_CUDA_WINDOWS_INDEX = "https://download.pytorch.org/whl/cu126"


def cuda_hardware_present() -> bool:
    """Whether an NVIDIA driver is installed (nvidia-smi on PATH) — the
    pre-torch signal for choosing CUDA-capable wheels."""
    return shutil.which("nvidia-smi") is not None


@dataclass(frozen=True)
class Artifact:
    """One downloadable weight file (or snapshot directory).

    Exactly one of (`hf_repo` + `hf_filename`), (`hf_repo` + snapshot), or
    `url` describes the source; `dest` is the models_dir-relative path the
    backends expect. `sha256` pins content for single-file artifacts (None
    for multi-GB files, where huggingface_hub's own integrity checking
    applies, and for snapshots)."""

    dest: str  # models_dir-relative target path (file, or dir for snapshots)
    approx_size: str  # human-readable size, shown in the plan
    license: str  # SPDX-ish license of the weights, shown in the plan
    hf_repo: Optional[str] = None
    hf_filename: Optional[str] = None  # None with hf_repo => full snapshot
    url: Optional[str] = None
    sha256: Optional[str] = None


@dataclass(frozen=True)
class Group:
    """A provisionable capability: the runtime packages and weight
    artifacts one backend needs. `runtime` pairs an importable probe module
    with the pip requirement that provides it; probes that already import
    are never reinstalled."""

    name: str
    enables: str  # what lights up in the UI/worker once provisioned
    artifacts: tuple
    runtime: tuple = ()  # ((probe_module, pip_requirement), ...)


# Registry of everything the AI layer can load. dest paths mirror the
# constants in embedders.py / faces.py / critic_qwen.py /
# segmenter_mobilesam.py -- those modules are the source of truth.
GROUPS = (
    Group(
        name="faces",
        enables="Faces tab (detection + clustering); ~3 MB",
        runtime=(),  # OpenCV ships with the core app
        artifacts=(
            Artifact(
                dest="face_detection_yunet_2023mar.onnx",
                approx_size="232 KB", license="MIT",
                url=("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
                     "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
                sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
            ),
            Artifact(
                dest="face_recognition_sface_2021dec.onnx",
                approx_size="37 MB", license="Apache-2.0",
                url=("https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
                     "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
                sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
            ),
        ),
    ),
    Group(
        name="semantic",
        enables="Similar tab (semantic space), critic grounding gate, prompt alignment",
        runtime=(("torch", "torch"), ("open_clip", "open_clip_torch")),
        artifacts=(
            Artifact(
                dest="open_clip/ViT-B-32_laion2b_s34b_b79k.bin",
                approx_size="605 MB", license="MIT",
                hf_repo="laion/CLIP-ViT-B-32-laion2B-s34b-b79k",
                hf_filename="open_clip_pytorch_model.bin",
                sha256="1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad",
            ),
        ),
    ),
    Group(
        name="visual",
        enables="Similar tab (visual space)",
        runtime=(("torch", "torch"), ("transformers", "transformers")),
        artifacts=(
            Artifact(
                dest="dinov2-small",  # snapshot directory
                approx_size="90 MB", license="Apache-2.0",
                hf_repo="facebook/dinov2-small",
            ),
        ),
    ),
    Group(
        name="segmenter",
        enables="Defect masks for localizable review findings",
        runtime=(("torch", "torch"), ("timm", "timm"),
                 ("mobile_sam",
                  "mobile-sam @ git+https://github.com/ChaoningZhang/MobileSAM.git")),
        artifacts=(
            Artifact(
                dest="mobile_sam.pt",
                approx_size="40 MB", license="Apache-2.0",
                hf_repo="dhkim2810/MobileSAM",
                hf_filename="mobile_sam.pt",
                sha256="6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
            ),
        ),
    ),
    Group(
        name="critic",
        enables="Review tab (quality/alignment scores + typed findings); needs 'semantic' too",
        runtime=(("llama_cpp", "llama-cpp-python>=0.3.0"),
                 ("torch", "torch"), ("open_clip", "open_clip_torch")),
        artifacts=(
            Artifact(
                dest="Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                approx_size="4.7 GB", license="Apache-2.0",
                hf_repo="ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
                hf_filename="Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
            ),
            Artifact(
                dest="mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
                approx_size="845 MB", license="Apache-2.0",
                hf_repo="ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
                hf_filename="mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
            ),
        ),
    ),
)

_GROUPS_BY_NAME = {g.name: g for g in GROUPS}


class ProvisionError(RuntimeError):
    """A download or verification failed; the message says which artifact."""


def _sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: str, expected: Optional[str], label: str) -> None:
    """Check a downloaded file against its pinned digest (no-op when the
    artifact carries none)."""
    if expected is None:
        return
    actual = _sha256_of(path)
    if actual != expected:
        raise ProvisionError(
            f"{label}: SHA-256 mismatch (expected {expected[:16]}..., got "
            f"{actual[:16]}...); refusing to keep the file")


def _download_url(url: str, dest_path: str) -> None:
    """Stream one direct URL to dest_path via a temp file."""
    tmp = dest_path + ".part"
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out)
    os.replace(tmp, dest_path)


def _download_hf_file(repo: str, filename: str, dest_path: str) -> None:
    """Fetch one file from a Hugging Face repo into dest_path, using the
    hub's cache/resume machinery."""
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(repo_id=repo, filename=filename)
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    shutil.copyfile(cached, dest_path)


def _download_hf_snapshot(repo: str, dest_dir: str) -> None:
    """Materialize a full Hugging Face repo snapshot at dest_dir."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, local_dir=dest_dir)


def _default_pip_runner(args: list) -> None:
    """Run one `pip install` into the current interpreter's environment;
    failure raises ProvisionError carrying the tail of pip's stderr."""
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        raise ProvisionError(
            f"pip install {' '.join(args)} failed: {proc.stderr.strip()[-400:]}")


def _pip_args_for(requirement: str) -> list:
    """pip arguments for one requirement. torch picks the wheel matching
    the hardware: CUDA-capable with an NVIDIA driver present (unless
    AI_DAM_DEVICE=cpu), CPU-index otherwise; macOS always uses PyPI."""
    if requirement != "torch":
        return [requirement]
    if sys.platform == "darwin":
        return ["torch"]
    force_cpu = os.environ.get("AI_DAM_DEVICE", "").lower() == "cpu"
    if not force_cpu and cuda_hardware_present():
        if sys.platform == "win32":
            return ["torch", "--index-url", _TORCH_CUDA_WINDOWS_INDEX]
        return ["torch"]  # PyPI Linux wheels bundle CUDA
    return ["torch", "--index-url", _TORCH_CPU_INDEX]


def runtime_missing(group: Group) -> list:
    """The group's runtime requirements whose probe modules cannot be
    imported right now, as (probe_module, pip_requirement) pairs."""
    return [(probe, req) for probe, req in group.runtime
            if importlib.util.find_spec(probe) is None]


def ensure_runtime(group: Group, log: Callable[[str], None] = print,
                   pip_runner: Optional[Callable[[list], None]] = None) -> list:
    """Install the group's missing runtime packages into the running
    environment and refresh import caches so they are importable in this
    process immediately. Returns the pip requirements installed."""
    runner = pip_runner or _default_pip_runner
    installed = []
    for probe, requirement in runtime_missing(group):
        log(f"  + runtime {requirement} (provides '{probe}')")
        runner(_pip_args_for(requirement))
        installed.append(requirement)
    if installed:
        importlib.invalidate_caches()
    return installed


def _ensure_hub(needed: bool, log: Callable[[str], None],
                pip_runner: Optional[Callable[[list], None]]) -> None:
    """Bootstrap huggingface_hub (the downloader itself) when any Hugging
    Face artifact is about to be fetched and the hub is not importable."""
    if not needed or importlib.util.find_spec("huggingface_hub") is not None:
        return
    log("  + runtime huggingface_hub (downloader)")
    (pip_runner or _default_pip_runner)(["huggingface_hub"])
    importlib.invalidate_caches()


def artifact_present(models_dir: str, artifact: Artifact) -> bool:
    """Whether the artifact already exists at its expected location
    (snapshot directories count as present when non-empty)."""
    path = os.path.join(models_dir, artifact.dest)
    if artifact.hf_repo is not None and artifact.hf_filename is None:
        return os.path.isdir(path) and bool(os.listdir(path))
    return os.path.isfile(path)


def resolve_groups(names) -> list:
    """Expand CLI group names ('all' included) into Group objects; unknown
    names raise ValueError listing the valid ones."""
    if not names or "all" in names:
        return list(GROUPS)
    unknown = [n for n in names if n not in _GROUPS_BY_NAME]
    if unknown:
        raise ValueError(
            f"unknown group(s) {unknown}; valid: "
            f"{[g.name for g in GROUPS] + ['all']}")
    return [_GROUPS_BY_NAME[n] for n in names]


def provision(
    models_dir: str,
    group_names,
    force: bool = False,
    log: Callable[[str], None] = print,
    downloaders: Optional[dict] = None,
    install_packages: bool = True,
    pip_runner: Optional[Callable[[list], None]] = None,
) -> dict:
    """Make the requested groups fully loadable: install their missing
    runtime packages (unless `install_packages` is False), then download
    every missing weight artifact into `models_dir`. Returns
    {'downloaded': [...], 'skipped': [...], 'installed': [...]}. `force`
    re-downloads artifacts that already exist. `downloaders` overrides the
    three fetch functions (keys 'url', 'hf_file', 'hf_snapshot') and
    `pip_runner` the pip invocation -- the seams tests use to stay
    network-free."""
    dl = {
        "url": _download_url,
        "hf_file": _download_hf_file,
        "hf_snapshot": _download_hf_snapshot,
    }
    if downloaders:
        dl.update(downloaders)

    groups = resolve_groups(group_names)
    installed: list = []
    if install_packages:
        _ensure_hub(
            any(a.hf_repo is not None for g in groups for a in g.artifacts
                if force or not artifact_present(models_dir, a)),
            log, pip_runner)
        for group in groups:
            installed.extend(ensure_runtime(group, log=log, pip_runner=pip_runner))

    downloaded: list = []
    skipped: list = []
    for group in groups:
        for artifact in group.artifacts:
            dest_path = os.path.join(models_dir, artifact.dest)
            if not force and artifact_present(models_dir, artifact):
                skipped.append(artifact.dest)
                log(f"  = {artifact.dest} (already present)")
                continue
            os.makedirs(os.path.dirname(dest_path) or models_dir, exist_ok=True)
            log(f"  + {artifact.dest} ({artifact.approx_size}, {artifact.license})")
            try:
                if artifact.url is not None:
                    dl["url"](artifact.url, dest_path)
                elif artifact.hf_filename is not None:
                    dl["hf_file"](artifact.hf_repo, artifact.hf_filename, dest_path)
                else:
                    dl["hf_snapshot"](artifact.hf_repo, dest_path)
            except ProvisionError:
                raise
            except Exception as exc:
                raise ProvisionError(f"{artifact.dest}: download failed: {exc}") from exc
            if artifact.hf_repo is None or artifact.hf_filename is not None:
                try:
                    _verify(dest_path, artifact.sha256, artifact.dest)
                except ProvisionError:
                    os.unlink(dest_path)
                    raise
            downloaded.append(artifact.dest)
    return {"downloaded": downloaded, "skipped": skipped, "installed": installed}


def format_plan(models_dir: str, group_names) -> str:
    """Human-readable table of what the requested groups would install and
    fetch, and what is already in place."""
    lines = [f"models_dir: {models_dir}"]
    for group in resolve_groups(group_names):
        lines.append(f"\n[{group.name}] {group.enables}")
        for probe, requirement in group.runtime:
            state = ("present" if importlib.util.find_spec(probe) is not None
                     else "MISSING")
            lines.append(f"  {state:8s} runtime {requirement}")
        for artifact in group.artifacts:
            state = "present" if artifact_present(models_dir, artifact) else "MISSING"
            lines.append(f"  {state:8s} {artifact.dest} "
                         f"({artifact.approx_size}, {artifact.license})")
    return "\n".join(lines)
