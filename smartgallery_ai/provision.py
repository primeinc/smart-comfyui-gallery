"""Lazy acquisition of everything the AI layer needs to run.

Downloads a capability group's model weights into `models_dir` (default
`.AImodels/`) from their official Hugging Face repositories (via
`huggingface_hub`'s
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
import hashlib
import importlib
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass

from huggingface_hub import hf_hub_download, snapshot_download

_logger = logging.getLogger(__name__)


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
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        caps = [float(line.strip()) for line in proc.stdout.splitlines() if line.strip()]
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
        proc = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10, check=False)
        match = re.search(r"CUDA Version:\s*([\d.]+)", proc.stdout or "")
        if match:
            value = float(match.group(1))
    except (OSError, subprocess.SubprocessError, ValueError):
        value = None
    _driver_cuda_cache.append(value)
    return value


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
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap,memory.total,memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        for line in proc.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 4:
                driver = parts[1] or driver
                try:
                    cap = float(parts[2])
                except ValueError:
                    cap = None
                gpus.append(
                    {
                        "name": parts[0],
                        "compute_capability": cap,
                        "vram": parts[3],
                        "vram_used": parts[4] if len(parts) >= 5 else None,
                    }
                )
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return {
        "gpus": gpus,
        "driver": driver,
        "driver_cuda": _driver_cuda_version(),
        "compute_capability": _cuda_compute_capability(),
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
    hf_repo: str | None = None
    hf_filename: str | None = None  # None with hf_repo => full snapshot
    url: str | None = None
    sha256: str | None = None
    unzip_member: str | None = None  # with url: extract this member of the downloaded zip as dest
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
# constants in embedders.py / faces.py / reviewer.py /
# segmenter_mobilesam.py -- those modules are the source of truth.
GROUPS = (
    Group(
        name="faces",
        enables="Faces tab (detection + clustering + detector compare)",
        # OpenCV ships with the core app; faiss accelerates the clustering
        # similarity graph (exact IndexFlatIP; NumPy fallback exists).
        # insightface + onnxruntime power the SCRFD pipeline used by the
        # detector-compare endpoint (and FaceAnalysis generally).
        runtime=(("faiss", "faiss-cpu"), ("insightface", "insightface"), ("onnxruntime", "onnxruntime")),
        artifacts=(
            Artifact(
                dest="face_detection_yunet_2023mar.onnx",
                approx_size="232 KB",
                license="MIT",
                url=(
                    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
                    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
                ),
                sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
            ),
            Artifact(
                dest="face_recognition_sface_2021dec.onnx",
                approx_size="37 MB",
                license="Apache-2.0",
                url=(
                    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
                    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
                ),
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
                url=("https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"),
                unzip_all=True,
            ),
        ),
    ),
    Group(
        name="semantic",
        enables="Similar tab (semantic space), critic grounding gate, prompt alignment",
        runtime=(("torch", "torch"), ("torchvision", "torchvision"), ("open_clip", "open_clip_torch")),
        artifacts=(
            Artifact(
                dest="open_clip/ViT-B-32_laion2b_s34b_b79k.bin",
                approx_size="605 MB",
                license="MIT",
                hf_repo="laion/CLIP-ViT-B-32-laion2B-s34b-b79k",
                hf_filename="open_clip_pytorch_model.bin",
                sha256="1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad",
            ),
        ),
    ),
    Group(
        name="visual",
        enables="Similar tab (visual space)",
        runtime=(("torch", "torch"), ("torchvision", "torchvision"), ("transformers", "transformers")),
        artifacts=(
            Artifact(
                dest="dinov2-small",  # snapshot directory
                approx_size="90 MB",
                license="Apache-2.0",
                hf_repo="facebook/dinov2-small",
            ),
        ),
    ),
    Group(
        name="segmenter",
        enables="Defect masks for localizable review findings",
        runtime=(
            ("torch", "torch"),
            ("torchvision", "torchvision"),
            ("timm", "timm"),
            ("mobile_sam", "mobile-sam @ git+https://github.com/ChaoningZhang/MobileSAM.git"),
        ),
        artifacts=(
            Artifact(
                dest="mobile_sam.pt",
                approx_size="40 MB",
                license="Apache-2.0",
                hf_repo="dhkim2810/MobileSAM",
                hf_filename="mobile_sam.pt",
                sha256="6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
            ),
        ),
    ),
    Group(
        name="omniquery",
        enables="Search palette AI answerer (free-language questions via read-only nl2sql)",
        # torchvision even though this checkpoint is text-only:
        # smartgallery_ai.models imports it before transformers so the
        # availability flag is set for the whole process, and a torch group
        # without a paired torchvision resolves an unmatched PyPI build.
        runtime=(("torch", "torch"), ("torchvision", "torchvision"), ("transformers", "transformers")),
        artifacts=(
            # The safetensors source the previously-shipped 4-bit GGUF was
            # quantized from, so the prompt contract and the 98-entry
            # corpus measurement (43.4% standalone execution match,
            # 2026-08-16) still describe this checkpoint. Consulted only
            # when the deterministic nlq parse flags structural leftovers.
            Artifact(
                dest="distil-qwen3-4b-text2sql",
                approx_size="8.1 GB",
                license="Apache-2.0",
                hf_repo="distil-labs/distil-qwen3-4b-text2sql",
            ),
        ),
    ),
    Group(
        name="critic",
        enables="Review tab (quality/alignment scores + typed findings); needs 'semantic' too",
        runtime=(
            ("torch", "torch"),
            ("torchvision", "torchvision"),
            ("transformers", "transformers"),
            ("open_clip", "open_clip_torch"),
        ),
        artifacts=(
            # A full snapshot, not a single file: transformers loads a
            # checkpoint directory. The reviewer runs ANY image-text-to-text
            # checkpoint (AI_DAM_CRITIC_MODEL), so this is the provisioned
            # default rather than a hard-coded dependency.
            Artifact(
                dest="Qwen3-VL-2B-Instruct",
                approx_size="4.4 GB",
                license="Apache-2.0",
                hf_repo="Qwen/Qwen3-VL-2B-Instruct",
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


def _verify(path: str, expected: str | None, label: str) -> None:
    """Check a downloaded file against its pinned digest (no-op when the
    artifact carries none)."""
    if expected is None:
        return
    actual = _sha256_of(path)
    if actual != expected:
        raise ProvisionError(
            f"{label}: SHA-256 mismatch (expected {expected[:16]}..., got {actual[:16]}...); refusing to keep the file"
        )


def _copy_with_progress(
    reader, writer, total: int | None, progress: Callable[[int, int | None], None] | None, chunk_size: int = 1 << 20
) -> int:
    """Chunked stream copy that reports (bytes_done, bytes_total) after
    every chunk; total may be None when the server sent no length.

    Returns the number of bytes copied, which is the only way the caller
    can tell a finished download from an abandoned one -- see
    _download_url.
    """
    done = 0
    while True:
        chunk = reader.read(chunk_size)
        if not chunk:
            break
        writer.write(chunk)
        done += len(chunk)
        if progress is not None:
            progress(done, total)
    return done


# Seconds a download may go without receiving a single byte. This is a
# per-read timeout, not a budget for the whole transfer, so a slow link
# still finishes a multi-gigabyte model as long as something keeps
# arriving. Without it a connection that is accepted and then goes silent
# -- a stalled mirror, a captive portal, a firewall that black-holes
# instead of refusing -- blocks the provisioning thread for ever: the AI
# layer never arrives, and nothing reports an error because nothing failed.
DOWNLOAD_STALL_TIMEOUT = 60


def _download_url(url: str, dest_path: str, progress: Callable[[int, int | None], None] | None = None) -> None:
    """Stream one direct URL to dest_path via a temp file, reporting byte
    progress as it goes.

    A download that stops early does not raise. CPython says so in
    http/client.py, where readinto returns 0 at a short body rather than
    reporting it:

        n = self.fp.readinto(b)
        if not n and b:
            # Ideally, we would raise IncompleteRead if the content-length
            # wasn't satisfied, but it might break compatibility.
            self._close_conn()

    So a dropped connection ends the copy loop the same way a finished one
    does, and what lands is however much arrived. These are weights of a
    few hundred megabytes upwards, on whatever connection somebody has.
    When the server said how many bytes to expect, that many have to have
    arrived.

    Every artifact fetched this way today happens to be a zip, so
    truncation currently surfaces further along as BadZipFile -- loud, but
    describing the wrong thing: the file is a zip, it is a piece of one.
    Saying how much is missing is the difference between retrying and
    going to look for a bad URL.

    Nothing is left behind either way. The .part file is the download in
    progress, and on any failure it was simply abandoned in the models
    directory, where nothing looks for it again.
    """
    tmp = dest_path + ".part"
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_STALL_TIMEOUT) as resp, open(tmp, "wb") as out:
            length = resp.headers.get("Content-Length")
            expected = int(length) if length else None
            written = _copy_with_progress(resp, out, expected, progress)

        if expected is not None and written != expected:
            raise ProvisionError(
                f"{os.path.basename(dest_path)}: the download stopped early "
                f"-- {written:,} of {expected:,} bytes arrived from {url}. "
                f"Nothing was kept; run provisioning again to retry."
            )
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise

    os.replace(tmp, dest_path)


def _download_zip_member(url_dl, url: str, member: str, dest_path: str) -> None:
    """Download a zip via `url_dl` and keep exactly one member as
    dest_path; the zip itself is always removed."""

    zip_tmp = dest_path + ".zip"
    url_dl(url, zip_tmp)
    try:
        with zipfile.ZipFile(zip_tmp) as zf, zf.open(member) as src, open(dest_path + ".part", "wb") as out:
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

    os.makedirs(dest_dir, exist_ok=True)
    zip_tmp = os.path.join(dest_dir, "_download.zip")
    url_dl(url, zip_tmp)
    try:
        with zipfile.ZipFile(zip_tmp) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            roots = {n.split("/", 1)[0] for n in names}
            strip = len(roots) == 1 and all("/" in n for n in names)
            for name in names:
                rel = name.split("/", 1)[1] if strip else name
                target = os.path.join(dest_dir, *rel.split("/"))
                if not os.path.abspath(target).startswith(os.path.abspath(dest_dir) + os.sep):
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
        _logger.debug("handled a failure in _hub_bars_silenced", exc_info=True)
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

    cached = hf_hub_download(repo_id=repo, filename=filename)
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    shutil.copyfile(cached, dest_path)


def _download_hf_snapshot(repo: str, dest_dir: str) -> None:
    """Materialize a full Hugging Face repo snapshot at dest_dir."""

    snapshot_download(repo_id=repo, local_dir=dest_dir)


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
    return [(probe, req) for probe, req in group.runtime if not _module_installed(probe)]


def artifact_present(models_dir: str, artifact: Artifact) -> bool:
    """Whether the artifact already exists at its expected location
    (snapshot directories count as present when non-empty)."""
    path = os.path.join(models_dir, artifact.dest)
    if (artifact.hf_repo is not None and artifact.hf_filename is None) or artifact.unzip_all:
        return os.path.isdir(path) and bool(os.listdir(path))
    return os.path.isfile(path)


# Conservative parse of Artifact.approx_size for the disk preflight: every
# figure in the string is summed ("407 MB (344 MB zip)" needs the archive
# and the extraction on disk at the same time).
_SIZE_TOKEN_RE = re.compile(r"([\d.]+)\s*(KB|MB|GB)", re.IGNORECASE)
_SIZE_UNITS = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}
# Headroom the preflight insists on beyond the artifact bytes: caches,
# thumbnails, and SQLite growth share the volume, and filling a system
# drive to zero takes the whole machine down, not just this app.
_DISK_HEADROOM_BYTES = 1024**3


def _approx_bytes(size_text: str) -> int:
    """Byte estimate of an approx_size string; 0 when nothing parses."""
    return sum(int(float(num) * _SIZE_UNITS[unit.upper()]) for num, unit in _SIZE_TOKEN_RE.findall(size_text))


def _check_disk_space(models_dir: str, artifacts) -> None:
    """Refuse to start downloads that cannot fit: the missing artifacts'
    declared sizes plus headroom vs the target volume's free bytes. An
    unprobeable volume falls through to the download attempt."""
    needed = sum(_approx_bytes(a.approx_size) for a in artifacts)
    if not needed:
        return
    probe = os.path.abspath(models_dir)
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return
    if free < needed + _DISK_HEADROOM_BYTES:
        raise ProvisionError(
            f"not enough disk space for the requested weights: "
            f"~{needed // 1024**2} MB to download plus "
            f"{_DISK_HEADROOM_BYTES // 1024**2} MB headroom, but only "
            f"{free // 1024**2} MB free on the volume of {probe}. Free up "
            "space or point AI_DAM_MODELS_DIR at a roomier drive."
        )


def resolve_groups(names) -> list:
    """Expand CLI group names ('all' included) into Group objects; unknown
    names raise ValueError listing the valid ones."""
    if not names or "all" in names:
        return list(GROUPS)
    unknown = [n for n in names if n not in _GROUPS_BY_NAME]
    if unknown:
        raise ValueError(f"unknown group(s) {unknown}; valid: {[g.name for g in GROUPS] + ['all']}")
    return [_GROUPS_BY_NAME[n] for n in names]


def provision(
    models_dir: str,
    group_names,
    force: bool = False,
    log: Callable[[str], None] = print,
    downloaders: dict | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Download every missing weight artifact for the requested groups
    into `models_dir`. Returns {'downloaded': [...], 'skipped': [...]}.
    `force` re-downloads artifacts that already exist; `downloaders`
    overrides the three fetch functions (keys 'url', 'hf_file',
    'hf_snapshot') -- the seam tests use to stay network-free.

    This function does NOT install packages. uv owns dependency
    management: the environment comes from `uv sync` against the
    manifest, which is also where the torch CUDA index is pinned. A
    second installer here meant two owners of the same decision, and the
    one that guessed from a driver table kept overwriting the one that
    was declared.

    `progress`, when given, receives structured events as work happens:
    {'kind': 'artifact', 'phase': 'start'|'bytes'|'done',
     'item': <dest>, ...} with 'bytes_done'/'bytes_total' on byte events
    (direct-URL downloads only; Hugging Face transfers report
    start/done)."""
    if not models_dir:
        raise ProvisionError(
            "models_dir is required (an empty value would scatter weight files relative to the working directory)"
        )
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
    _check_disk_space(
        models_dir, [a for g in groups for a in g.artifacts if force or not artifact_present(models_dir, a)]
    )

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
            emit({"kind": "artifact", "phase": "start", "item": artifact.dest, "size": artifact.approx_size})
            url_dl = dl["url"]
            if url_dl is _download_url and progress is not None:

                def url_dl(u, d, _dest=artifact.dest):
                    _download_url(
                        u,
                        d,
                        progress=lambda done, total: emit(
                            {
                                "kind": "artifact",
                                "phase": "bytes",
                                "item": _dest,
                                "bytes_done": done,
                                "bytes_total": total,
                            }
                        ),
                    )

            # Structured-progress mode owns the console: the hub's own
            # carriage-return bars would interleave with the caller's log
            # lines, so silence them around the default hub downloaders.
            def hf_quiet(fn, default):
                return _hub_bars_silenced if progress is not None and fn is default else contextlib.nullcontext

            try:
                if artifact.url is not None and artifact.unzip_all:
                    _download_zip_all(url_dl, artifact.url, dest_path)
                elif artifact.url is not None and artifact.unzip_member is not None:
                    _download_zip_member(url_dl, artifact.url, artifact.unzip_member, dest_path)
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
    return {"downloaded": downloaded, "skipped": skipped}


def format_plan(models_dir: str, group_names) -> str:
    """Human-readable table of what the requested groups would install and
    fetch, and what is already in place."""
    lines = [f"models_dir: {models_dir}"]
    for group in resolve_groups(group_names):
        lines.append(f"\n[{group.name}] {group.enables}")
        for probe, requirement in group.runtime:
            state = "present" if importlib.util.find_spec(probe) is not None else "MISSING"
            lines.append(f"  {state:8s} runtime {requirement}")
        for artifact in group.artifacts:
            state = "present" if artifact_present(models_dir, artifact) else "MISSING"
            lines.append(f"  {state:8s} {artifact.dest} ({artifact.approx_size}, {artifact.license})")
    return "\n".join(lines)
