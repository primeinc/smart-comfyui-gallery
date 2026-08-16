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

import contextlib
import glob
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

# torch wheel selection: an NVIDIA GPU (detected via nvidia-smi on PATH)
# gets CUDA-capable wheels — PyPI's Linux torch bundles CUDA, Windows CUDA
# builds live only on a cu-index. Without a GPU the CPU index avoids the
# multi-GB CUDA payload. macOS torch on PyPI is already CPU/MPS.
# AI_DAM_DEVICE=cpu forces the CPU wheel regardless of hardware.
#
# The cu-index must match the GPU GENERATION: cu126 wheels carry no
# kernels for Blackwell-and-newer cards (compute capability >= 10) — the
# install "succeeds" and then every kernel launch dies with
# cudaErrorNoKernelImageForDevice. AI_DAM_CUDA_INDEX overrides the choice.
_TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
_TORCH_CUDA_INDEX_DEFAULT = "https://download.pytorch.org/whl/cu126"
# Blackwell-and-newer (sm_120+) kernel builds of current torch, newest
# first with the minimum driver-side CUDA version each requires (torch's
# own unsupported-GPU warning names exactly these three for 2.13).
_TORCH_CUDA_BLACKWELL_CHOICES = (
    (13.2, "https://download.pytorch.org/whl/cu132"),
    (13.0, "https://download.pytorch.org/whl/cu130"),
    (12.9, "https://download.pytorch.org/whl/cu129"),
)
_TORCH_CUDA_INDEX_BLACKWELL_FALLBACK = "https://download.pytorch.org/whl/cu130"


def cuda_hardware_present() -> bool:
    """Whether an NVIDIA driver is installed (nvidia-smi on PATH) — the
    pre-torch signal for choosing CUDA-capable wheels."""
    return shutil.which("nvidia-smi") is not None


_compute_cap_cache: list = []  # memoized [value-or-None]; nvidia-smi costs ~100ms
_driver_cuda_cache: list = []  # memoized [value-or-None]


def _cuda_compute_capability():
    """Highest GPU compute capability nvidia-smi reports (e.g. 12.0 for a
    Blackwell consumer card), or None when undetectable. Memoized."""
    if _compute_cap_cache:
        return _compute_cap_cache[0]
    cap = None
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        caps = [float(line.strip()) for line in proc.stdout.splitlines()
                if line.strip()]
        cap = max(caps) if caps else None
    except (OSError, subprocess.SubprocessError, ValueError):
        cap = None
    _compute_cap_cache.append(cap)
    return cap


def _driver_cuda_version():
    """The maximum CUDA version the installed driver supports ("CUDA
    Version" in nvidia-smi's header), or None when undetectable. Memoized."""
    if _driver_cuda_cache:
        return _driver_cuda_cache[0]
    value = None
    try:
        proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                              timeout=10)
        match = re.search(r"CUDA Version:\s*([\d.]+)", proc.stdout or "")
        if match:
            value = float(match.group(1))
    except (OSError, subprocess.SubprocessError, ValueError):
        value = None
    _driver_cuda_cache.append(value)
    return value


def torch_cuda_index() -> str:
    """The CUDA wheel index matching this machine's GPU generation and
    driver: pre-Blackwell cards keep cu126; Blackwell-and-newer (compute
    capability >= 10) get the newest sm_120 build the driver can run.
    AI_DAM_CUDA_INDEX overrides for hardware this table doesn't know.
    (Mixed rigs pair the newest card's index — it still carries kernels
    for every generation back to Turing.)"""
    override = os.environ.get("AI_DAM_CUDA_INDEX", "").strip()
    if override:
        return override
    cap = _cuda_compute_capability()
    if cap is None or cap < 10.0:
        return _TORCH_CUDA_INDEX_DEFAULT
    driver_cuda = _driver_cuda_version()
    if driver_cuda is not None:
        for minimum, index in _TORCH_CUDA_BLACKWELL_CHOICES:
            if driver_cuda >= minimum:
                return index
    return _TORCH_CUDA_INDEX_BLACKWELL_FALLBACK


# Official prebuilt CUDA wheels for llama-cpp-python (the critic's
# runtime). Coverage is narrower than torch's: CPython 3.10-3.12 only as
# of 2026-08 — the swap is attempted best-effort and the CPU build stays
# when no wheel matches. AI_DAM_LLAMA_CUDA_INDEX overrides.
_LLAMA_CUDA_INDEX_DEFAULT = "https://abetlen.github.io/llama-cpp-python/whl/cu124"
_llama_gpu_cache: list = []  # memoized [bool-or-None]


def llama_cuda_index() -> str:
    """The prebuilt-CUDA wheel index for llama-cpp-python."""
    return (os.environ.get("AI_DAM_LLAMA_CUDA_INDEX", "").strip()
            or _LLAMA_CUDA_INDEX_DEFAULT)


def _llama_supports_gpu():
    """Whether the installed llama-cpp-python build can offload to GPU,
    probed in a CHILD process — importing it here would load (and on
    Windows lock) the DLL a swap needs to replace. None when the package
    is absent or the probe fails. Memoized."""
    if _llama_gpu_cache:
        return _llama_gpu_cache[0]
    value = None
    if importlib.util.find_spec("llama_cpp") is not None:
        try:
            proc = subprocess.run(
                [sys.executable, "-c",
                 # Inline PATH bootstrap (mirrors smartgallery_ai.llama_runtime,
                 # dependency-free for the child): the prebuilt CUDA wheel loads
                 # its DLLs via the legacy PATH search, so the pip-installed
                 # NVIDIA runtime bins must be prepended first.
                 "import glob, os, sysconfig; "
                 "[os.environ.__setitem__('PATH', d + os.pathsep + os.environ['PATH']) "
                 "for d in glob.glob(os.path.join(sysconfig.get_paths()['purelib'], 'nvidia', '*', 'bin'))]; "
                 "import llama_cpp; print(int(llama_cpp.llama_supports_gpu_offload()))"],
                capture_output=True, text=True, timeout=120)
            if proc.returncode == 0:
                value = proc.stdout.strip() == "1"
        except (OSError, subprocess.SubprocessError, ValueError):
            value = None
    _llama_gpu_cache.append(value)
    return value


def llama_cuda_reinstall_needed() -> bool:
    """Whether the installed llama-cpp-python is a CPU-only build on CUDA
    hardware (the critic then runs its 30s-per-image vision encoder on
    CPU while the GPU idles). AI_DAM_DEVICE=cpu opts out."""
    if sys.platform == "darwin" or not cuda_hardware_present():
        return False
    if os.environ.get("AI_DAM_DEVICE", "").lower() == "cpu":
        return False
    return _llama_supports_gpu() is False


def cuda_summary():
    """One-shot GPU inventory for boot logging and /status: EVERY card
    with its own name/compute capability/VRAM (a machine can mix
    generations), the driver, and the wheel index this machine would use
    — or None when no NVIDIA driver is present."""
    if not cuda_hardware_present():
        return None
    gpus = []
    driver = None
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,compute_cap,memory.total,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        for line in proc.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4:
                driver = parts[1] or driver
                try:
                    cap = float(parts[2])
                except ValueError:
                    cap = None
                gpus.append({"name": parts[0], "compute_capability": cap,
                             "vram": parts[3],
                             "vram_used": parts[4] if len(parts) >= 5 else None})
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return {
        "gpus": gpus,
        "driver": driver,
        "driver_cuda": _driver_cuda_version(),
        "compute_capability": _cuda_compute_capability(),
        "torch_index": torch_cuda_index(),
    }


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
    unzip_member: Optional[str] = None  # with url: extract this member of the downloaded zip as dest
    unzip_all: bool = False  # with url: extract the whole zip under dest (a directory)


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
        enables="Faces tab (detection + clustering + detector compare)",
        # OpenCV ships with the core app; faiss accelerates the clustering
        # similarity graph (exact IndexFlatIP; NumPy fallback exists).
        # insightface + onnxruntime power the SCRFD pipeline used by the
        # detector-compare endpoint (and FaceAnalysis generally).
        runtime=(("faiss", "faiss-cpu"),
                 ("insightface", "insightface"),
                 ("onnxruntime", "onnxruntime")),
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
            # insightface antelopev2 pack (official release asset), laid
            # out for FaceAnalysis(name='antelopev2', root=<models>/insightface):
            # SCRFD-10GF detector, glintr100 recognizer (ResNet100@Glint360K,
            # 512-d — best of the labeled A/B in
            # benchmarks/results/face_embedder_ab.json), landmark and
            # attribute heads. glintr100 also feeds the cv2 arcface embedder
            # directly from this pack. License is non-commercial research
            # per deepinsight/insightface README.
            Artifact(
                dest="insightface/models/antelopev2",
                approx_size="407 MB (344 MB zip)",
                license="non-commercial research (insightface)",
                url=("https://github.com/deepinsight/insightface/releases/"
                     "download/v0.7/antelopev2.zip"),
                unzip_all=True,
            ),
        ),
    ),
    Group(
        name="semantic",
        enables="Similar tab (semantic space), critic grounding gate, prompt alignment",
        runtime=(("torch", "torch"), ("torchvision", "torchvision"),
                 ("open_clip", "open_clip_torch")),
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
        runtime=(("torch", "torch"), ("torchvision", "torchvision"),
                 ("transformers", "transformers")),
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
        runtime=(("torch", "torch"), ("torchvision", "torchvision"),
                 ("timm", "timm"),
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
                 ("torch", "torch"), ("torchvision", "torchvision"),
                 ("open_clip", "open_clip_torch")),
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


def _copy_with_progress(reader, writer, total: Optional[int],
                        progress: Optional[Callable[[int, Optional[int]], None]],
                        chunk_size: int = 1 << 20) -> None:
    """Chunked stream copy that reports (bytes_done, bytes_total) after
    every chunk; total may be None when the server sent no length."""
    done = 0
    while True:
        chunk = reader.read(chunk_size)
        if not chunk:
            break
        writer.write(chunk)
        done += len(chunk)
        if progress is not None:
            progress(done, total)


def _download_url(url: str, dest_path: str,
                  progress: Optional[Callable[[int, Optional[int]], None]] = None) -> None:
    """Stream one direct URL to dest_path via a temp file, reporting byte
    progress as it goes."""
    tmp = dest_path + ".part"
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        length = resp.headers.get("Content-Length")
        _copy_with_progress(resp, out, int(length) if length else None, progress)
    os.replace(tmp, dest_path)


def _download_zip_member(url_dl, url: str, member: str, dest_path: str) -> None:
    """Download a zip via `url_dl` and keep exactly one member as
    dest_path; the zip itself is always removed."""
    import zipfile

    zip_tmp = dest_path + ".zip"
    url_dl(url, zip_tmp)
    try:
        with zipfile.ZipFile(zip_tmp) as zf, \
                zf.open(member) as src, \
                open(dest_path + ".part", "wb") as out:
            shutil.copyfileobj(src, out)
        os.replace(dest_path + ".part", dest_path)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(zip_tmp)


def _download_zip_all(url_dl, url: str, dest_dir: str) -> None:
    """Download a zip via `url_dl` and extract every member under
    dest_dir, stripping one shared top-level directory when the zip has
    one (release packs nest a single '<name>/' folder); the zip itself is
    always removed."""
    import zipfile

    os.makedirs(dest_dir, exist_ok=True)
    zip_tmp = os.path.join(dest_dir, "_download.zip")
    url_dl(url, zip_tmp)
    try:
        with zipfile.ZipFile(zip_tmp) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            roots = {n.split("/", 1)[0] for n in names}
            strip = (len(roots) == 1 and all("/" in n for n in names))
            for name in names:
                rel = name.split("/", 1)[1] if strip else name
                target = os.path.join(dest_dir, *rel.split("/"))
                if not os.path.abspath(target).startswith(
                        os.path.abspath(dest_dir) + os.sep):
                    raise ProvisionError(f"zip member escapes dest: {name}")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(name) as src, open(target + ".part", "wb") as out:
                    shutil.copyfileobj(src, out)
                os.replace(target + ".part", target)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(zip_tmp)


@contextlib.contextmanager
def _hub_bars_silenced():
    """Silence huggingface_hub's own console progress bars (including the
    hf_xet 'downloading bytes'/'reconstructing file' ones) for the duration.
    Structured-progress callers render their own output; the hub's
    carriage-return bars interleave with it and garble the console."""
    try:
        from huggingface_hub import utils as hub_utils
    except Exception:  # no hub, nothing to silence
        yield
        return
    was_disabled = hub_utils.are_progress_bars_disabled()
    if not was_disabled:
        hub_utils.disable_progress_bars()
    try:
        yield
    finally:
        if not was_disabled:
            hub_utils.enable_progress_bars()


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


def _run_pip_operation(pip_args: list, uv_args: list, timeout: int) -> None:
    """Run one pip operation in THIS interpreter's environment, tolerating
    environments that ship without pip (uv-created venvs do): try
    `python -m pip` first, then `uv pip ... --python <this python>` when
    uv is on PATH, else bootstrap pip once via ensurepip and retry.
    Failure raises ProvisionError carrying the tail of stderr."""
    proc = subprocess.run([sys.executable, "-m", "pip", *pip_args],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode == 0:
        return
    err = proc.stderr or ""
    if "No module named pip" in err:
        uv = shutil.which("uv")
        if uv is not None:
            proc = subprocess.run(
                [uv, "pip", *uv_args, "--python", sys.executable],
                capture_output=True, text=True, timeout=timeout)
            if proc.returncode == 0:
                return
            err = proc.stderr or err
        else:
            boot = subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"],
                                  capture_output=True, text=True, timeout=600)
            if boot.returncode == 0:
                proc = subprocess.run([sys.executable, "-m", "pip", *pip_args],
                                      capture_output=True, text=True, timeout=timeout)
                if proc.returncode == 0:
                    return
                err = proc.stderr or err
            else:
                err = boot.stderr or err
    raise ProvisionError(
        f"pip {' '.join(pip_args)} failed: {err.strip()[-400:]}")


_ort_gpu_cache: list = []  # memoized [bool-or-None]


def _ort_cuda_reinstall_needed():
    """Whether the installed onnxruntime build lacks the CUDA execution
    provider, probed in a CHILD process (importing it here would lock the
    DLLs the swap needs to replace). False when the package is absent or
    the probe fails. Memoized."""
    if _ort_gpu_cache:
        return _ort_gpu_cache[0]
    value = False
    if importlib.util.find_spec("onnxruntime") is not None:
        try:
            proc = subprocess.run(
                [sys.executable, "-c",
                 "import onnxruntime; print(int('CUDAExecutionProvider' in "
                 "onnxruntime.get_available_providers()))"],
                capture_output=True, text=True, timeout=120)
            if proc.returncode == 0:
                value = proc.stdout.strip() == "0"
        except (OSError, subprocess.SubprocessError, ValueError):
            value = False
    _ort_gpu_cache.append(value)
    return value


def _default_pip_runner(args: list) -> None:
    """Install one requirement into the current interpreter's environment
    (works in pip-less uv venvs too; see _run_pip_operation). uv spells
    pip's --force-reinstall as --reinstall; translate for its fallback."""
    uv_args = ["--reinstall" if arg == "--force-reinstall" else arg
               for arg in args]
    _run_pip_operation(["install", "--quiet", *args],
                       ["install", "--quiet", *uv_args], timeout=3600)


def _default_pip_uninstaller(packages: list) -> None:
    """Uninstall packages from the current interpreter's environment
    (works in pip-less uv venvs too; see _run_pip_operation)."""
    _run_pip_operation(["uninstall", "--quiet", "-y", *packages],
                       ["uninstall", "--quiet", *packages], timeout=600)


def torch_cuda_reinstall_needed() -> bool:
    """Whether the installed torch build cannot use this machine's GPU:
    a CPU-index build on CUDA hardware, or (Windows) a CUDA build from
    the WRONG generation's index — wrong-generation kernels install fine
    and then fail at launch with cudaErrorNoKernelImageForDevice. The
    static installers pin an index blind because package resolution
    cannot see GPUs; only the running app can, so it swaps the pair at
    startup. Metadata-only -- never imports torch. AI_DAM_DEVICE=cpu
    opts out."""
    if sys.platform == "darwin" or not cuda_hardware_present():
        return False
    if os.environ.get("AI_DAM_DEVICE", "").lower() == "cpu":
        return False
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return False
    if "+cpu" in version:
        return True
    if sys.platform == "win32" and "+cu" in version:
        installed_tag = version.split("+", 1)[1]
        expected_tag = torch_cuda_index().rstrip("/").rsplit("/", 1)[-1]
        return installed_tag != expected_tag
    return False


def _pip_args_for(requirement: str) -> list:
    """pip arguments for one requirement. torch AND torchvision pick the
    wheel index matching the hardware — they must come from the SAME index
    or torchvision's compiled ops fail to register against the installed
    torch (RuntimeError: operator torchvision::nms does not exist).
    CUDA-capable with an NVIDIA driver present (unless AI_DAM_DEVICE=cpu),
    CPU-index otherwise; macOS always uses PyPI."""
    if requirement not in ("torch", "torchvision"):
        return [requirement]
    if sys.platform == "darwin":
        return [requirement]
    force_cpu = os.environ.get("AI_DAM_DEVICE", "").lower() == "cpu"
    if not force_cpu and cuda_hardware_present():
        if sys.platform == "win32":
            return [requirement, "--index-url", torch_cuda_index()]
        return [requirement]  # PyPI Linux wheels bundle CUDA
    return [requirement, "--index-url", _TORCH_CPU_INDEX]


def _module_installed(probe: str) -> bool:
    """Whether `probe` resolves to a real installed package. A bare
    directory on sys.path (e.g. in the working directory) materializes as
    a namespace package with no origin — that must count as MISSING, or a
    stray folder silently suppresses the runtime install and the backend
    fails later with a confusing error."""
    try:
        spec = importlib.util.find_spec(probe)
    except (ImportError, ValueError):
        return False
    return spec is not None and spec.origin is not None


def runtime_missing(group: Group) -> list:
    """The group's runtime requirements whose probe modules cannot be
    imported right now, as (probe_module, pip_requirement) pairs."""
    return [(probe, req) for probe, req in group.runtime
            if not _module_installed(probe)]


def ensure_runtime(group: Group, log: Callable[[str], None] = print,
                   pip_runner: Optional[Callable[[list], None]] = None) -> list:
    """Install the group's missing runtime packages into the running
    environment and refresh import caches so they are importable in this
    process immediately. Returns the pip requirements installed."""
    runner = pip_runner or _default_pip_runner
    installed = []
    for probe, requirement in runtime_missing(group):
        log(f"  + runtime {requirement} (provides '{probe}')")
        started = time.time()
        runner(_pip_args_for(requirement))
        log(f"    {requirement} installed in {time.time() - started:.0f}s")
        installed.append(requirement)
    if installed:
        importlib.invalidate_caches()
    return installed


def _ensure_hub(needed: bool, log: Callable[[str], None],
                pip_runner: Optional[Callable[[list], None]]) -> None:
    """Bootstrap huggingface_hub (the downloader itself) when any Hugging
    Face artifact is about to be fetched and the hub is not importable."""
    if not needed or _module_installed("huggingface_hub"):
        return
    log("  + runtime huggingface_hub (downloader)")
    (pip_runner or _default_pip_runner)(["huggingface_hub"])
    importlib.invalidate_caches()


def artifact_present(models_dir: str, artifact: Artifact) -> bool:
    """Whether the artifact already exists at its expected location
    (snapshot directories count as present when non-empty)."""
    path = os.path.join(models_dir, artifact.dest)
    if (artifact.hf_repo is not None and artifact.hf_filename is None) or artifact.unzip_all:
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
    progress: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Make the requested groups fully loadable: install their missing
    runtime packages (unless `install_packages` is False), then download
    every missing weight artifact into `models_dir`. Returns
    {'downloaded': [...], 'skipped': [...], 'installed': [...]}. `force`
    re-downloads artifacts that already exist. `downloaders` overrides the
    three fetch functions (keys 'url', 'hf_file', 'hf_snapshot') and
    `pip_runner` the pip invocation -- the seams tests use to stay
    network-free.

    `progress`, when given, receives structured events as work happens:
    {'kind': 'runtime'|'artifact', 'phase': 'start'|'bytes'|'done',
     'item': <requirement or dest>, ...} with 'bytes_done'/'bytes_total'
    on byte events (direct-URL downloads only; Hugging Face transfers
    report start/done)."""
    if not models_dir:
        raise ProvisionError(
            "models_dir is required (an empty value would scatter weight "
            "files relative to the working directory)")
    dl = {
        "url": _download_url,
        "hf_file": _download_hf_file,
        "hf_snapshot": _download_hf_snapshot,
    }
    if downloaders:
        dl.update(downloaders)

    def emit(event: dict) -> None:
        if progress is not None:
            progress(event)

    groups = resolve_groups(group_names)
    installed: list = []
    if install_packages:
        # GPU self-heal: a torch build that cannot use this machine's GPU
        # is upgraded IN PLACE to the right wheels (PEP 440: +cu130 >
        # +cu126 > +cpu, so --upgrade against the target index replaces
        # it without an uninstall — a failed or interrupted download
        # leaves the previous working build installed). Only safe while
        # torch is unimported: an in-process torch pins its files (and
        # locks DLLs on Windows), so then we can only advise.
        if (any(req == "torch" for g in groups for _, req in g.runtime)
                and torch_cuda_reinstall_needed()):
            if "torch" in sys.modules:
                log("  ! torch is loaded in this process but its build cannot "
                    "use the GPU; restart the app to switch wheels "
                    "(AI_DAM_DEVICE=cpu opts out)")
            else:
                steering = _pip_args_for("torch")[1:]
                tag = (steering[-1].rstrip("/").rsplit("/", 1)[-1]
                       if steering else "PyPI")
                log(f"  ~ upgrading torch/torchvision in place to {tag} wheels")
                emit({"kind": "runtime", "phase": "start",
                      "item": f"torch ({tag} swap)"})
                (pip_runner or _default_pip_runner)(
                    ["--upgrade", "torch", "torchvision", *steering])
                importlib.invalidate_caches()
                installed.extend(["torch", "torchvision"])
                emit({"kind": "runtime", "phase": "done",
                      "item": f"torch ({tag} swap)"})
        # Critic GPU self-heal, BEST-EFFORT: official prebuilt CUDA wheels
        # for llama-cpp-python cover fewer Python versions than torch's,
        # so a failed attempt keeps the working CPU build and says exactly
        # what would unlock GPU reviews.
        if (any(req.startswith("llama-cpp-python")
                for g in groups for _, req in g.runtime)
                and llama_cuda_reinstall_needed()):
            if "llama_cpp" in sys.modules:
                log("  ! llama-cpp-python is loaded in this process but is a "
                    "CPU-only build; restart the app to attempt the CUDA swap")
            else:
                index = llama_cuda_index()
                cuda_tag = index.rstrip("/").rsplit("/", 1)[-1]
                log(f"  ~ trying prebuilt CUDA llama-cpp-python ({cuda_tag}) "
                    "for GPU reviews")
                try:
                    (pip_runner or _default_pip_runner)(
                        ["--force-reinstall", "--no-deps", "llama-cpp-python",
                         "--index-url", index])
                    if sys.platform == "win32":
                        # The prebuilt cu12x wheel links cudart64_12/cublas64_12,
                        # which nothing else provides on a torch-cu13 box —
                        # without these the swap ships a build that cannot even
                        # import (observed live: critic dead post-swap).
                        (pip_runner or _default_pip_runner)(
                            ["nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"])
                    importlib.invalidate_caches()
                    _llama_gpu_cache.clear()
                    installed.append("llama-cpp-python (CUDA)")
                    log("  + llama-cpp-python CUDA build installed — the "
                        "critic will offload to GPU")
                except ProvisionError as exc:
                    log("  ! no prebuilt CUDA llama-cpp-python wheel matches "
                        "this Python/OS (official wheels cover CPython "
                        "3.10-3.12); the critic stays on CPU (~90s/review). "
                        "Fastest fix: recreate the venv on Python 3.12 "
                        "(uv venv -p 3.12) and restart — everything "
                        f"reinstalls itself. Details: {str(exc)[-160:]}")
        # ORT GPU self-heal, BEST-EFFORT: the faces group ships CPU
        # onnxruntime; on an NVIDIA box swap it for onnxruntime-gpu so
        # the insightface pipeline's sessions get CUDAExecutionProvider
        # (faces._ort_providers picks it up; ORT falls back to CPU
        # per-session if the EP cannot initialize). The two packages
        # provide the same module, so the CPU build is removed first.
        if (cuda_hardware_present()
                and any(req == "onnxruntime"
                        for g in groups for _, req in g.runtime)
                and _ort_cuda_reinstall_needed()):
            if "onnxruntime" in sys.modules:
                log("  ! onnxruntime is loaded in this process but is a "
                    "CPU-only build; restart the app to attempt the CUDA swap")
            else:
                log("  ~ swapping onnxruntime for onnxruntime-gpu "
                    "(insightface pipeline on GPU)")
                emit({"kind": "runtime", "phase": "start",
                      "item": "onnxruntime-gpu swap"})
                try:
                    _default_pip_uninstaller(["onnxruntime"])
                    (pip_runner or _default_pip_runner)(["onnxruntime-gpu"])
                    importlib.invalidate_caches()
                    _ort_gpu_cache.clear()
                    installed.append("onnxruntime-gpu")
                    log("  + onnxruntime-gpu installed")
                except ProvisionError as exc:
                    log("  ! onnxruntime-gpu swap failed; restoring the CPU "
                        f"build. Details: {str(exc)[-160:]}")
                    (pip_runner or _default_pip_runner)(["onnxruntime"])
                emit({"kind": "runtime", "phase": "done",
                      "item": "onnxruntime-gpu swap"})
        # Vendored GPU faiss (vendor/faiss-gpu-win64) links the CUDA-13
        # runtime DLLs, which are not vendored (GitHub's 100MB file cap);
        # the nvidia wheels carry them (nvidia/<pkg>/bin/x86_64/). Install
        # them whenever the vendored build is usable on this box so
        # faiss_runtime.import_faiss() can select it; faiss-cpu stays the
        # installed fallback either way.
        if (sys.platform == "win32" and cuda_hardware_present()
                and any(req.startswith("faiss")
                        for g in groups for _, req in g.runtime)):
            from smartgallery_ai.faiss_runtime import vendored_faiss_dir
            purelib = sysconfig.get_paths().get("purelib") or ""
            cu13_present = bool(glob.glob(os.path.join(
                purelib, "nvidia", "*", "bin", "x86_64", "cublasLt64_13.dll")))
            if os.path.isdir(vendored_faiss_dir()) and not cu13_present:
                log("  ~ installing CUDA runtime wheels for the vendored "
                    "GPU faiss build")
                emit({"kind": "runtime", "phase": "start",
                      "item": "nvidia CUDA runtime (faiss GPU)"})
                (pip_runner or _default_pip_runner)(
                    ["nvidia-cublas~=13.0", "nvidia-cuda-runtime~=13.0",
                     "nvidia-nvjitlink~=13.0"])
                installed.append("nvidia CUDA runtime (faiss GPU)")
                emit({"kind": "runtime", "phase": "done",
                      "item": "nvidia CUDA runtime (faiss GPU)"})
        _ensure_hub(
            any(a.hf_repo is not None for g in groups for a in g.artifacts
                if force or not artifact_present(models_dir, a)),
            log, pip_runner)
        for group in groups:
            for _, requirement in runtime_missing(group):
                emit({"kind": "runtime", "phase": "start", "item": requirement})
            for requirement in ensure_runtime(group, log=log, pip_runner=pip_runner):
                installed.append(requirement)
                emit({"kind": "runtime", "phase": "done", "item": requirement})

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
            emit({"kind": "artifact", "phase": "start", "item": artifact.dest,
                  "size": artifact.approx_size})
            url_dl = dl["url"]
            if url_dl is _download_url and progress is not None:
                def url_dl(u, d, _dest=artifact.dest):
                    _download_url(u, d, progress=lambda done, total: emit(
                        {"kind": "artifact", "phase": "bytes", "item": _dest,
                         "bytes_done": done, "bytes_total": total}))
            # Structured-progress mode owns the console: the hub's own
            # carriage-return bars would interleave with the caller's log
            # lines, so silence them around the default hub downloaders.
            def hf_quiet(fn, default):
                return (_hub_bars_silenced if progress is not None
                        and fn is default else contextlib.nullcontext)
            try:
                if artifact.url is not None and artifact.unzip_all:
                    _download_zip_all(url_dl, artifact.url, dest_path)
                elif artifact.url is not None and artifact.unzip_member is not None:
                    _download_zip_member(
                        url_dl, artifact.url, artifact.unzip_member, dest_path)
                elif artifact.url is not None:
                    url_dl(artifact.url, dest_path)
                elif artifact.hf_filename is not None:
                    with hf_quiet(dl["hf_file"], _download_hf_file)():
                        dl["hf_file"](artifact.hf_repo, artifact.hf_filename, dest_path)
                else:
                    with hf_quiet(dl["hf_snapshot"], _download_hf_snapshot)():
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
            emit({"kind": "artifact", "phase": "done", "item": artifact.dest})
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
