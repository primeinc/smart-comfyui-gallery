"""Where model weights are, and who may fetch them.

Every weight file has a registry somebody else maintains, and this module
asks that registry -- never a URL, never a hash of its own:

    OpenCV YuNet / SFace   huggingface_hub, repos published by the opencv org
    insightface antelopev2 insightface.utils.ensure_available, its own release
    semantic checkpoints   huggingface_hub, through each adapter

Lookup is read-only and checks two places, in order: the run's
`models_dir`, then the machine's shared cache (`HF_HUB_CACHE` for Hub
files, `~/.insightface` for the pack). A weight found in either is used
where it lies. Downloading lands in `models_dir` only, and only when the
caller says `provision=True` -- a job. A serving path never provisions:
it resolves or refuses with the fix named (docs/AI_MODELS.md).
"""

from __future__ import annotations

import glob
import os
import shutil
from dataclasses import dataclass

#: The opencv org's own Hub repos for the two OpenCV face models.
YUNET_REPO, YUNET_FILE = "opencv/face_detection_yunet", "face_detection_yunet_2023mar.onnx"
SFACE_REPO, SFACE_FILE = "opencv/face_recognition_sface", "face_recognition_sface_2021dec.onnx"

#: insightface's pack name and the layout FaceAnalysis reads:
#: <root>/models/<pack>/*.onnx. `INSIGHTFACE_HOME` is upstream's default
#: root -- the shared, machine-wide copy.
PACK = "antelopev2"
INSIGHTFACE_HOME = "~/.insightface"
INSIGHTFACE_SUBDIR = "insightface"  # models_dir-relative root for the run's own copy


class Unprovisioned(LookupError):
    """A weight is in neither the run's models_dir nor the shared cache,
    and the caller may not download."""


def hub_cached(repo_id: str, filename: str, models_dir: str, revision: str | None = None) -> str | None:
    """The local path of one Hub file, from `models_dir` first and the
    shared HF_HUB_CACHE second, or None. Disk only: `try_to_load_from_cache`
    "will not raise any exception if the file is not cached"
    (huggingface_hub file_download.py:1485)."""
    from huggingface_hub import try_to_load_from_cache

    for cache_dir in (models_dir, None):
        found = try_to_load_from_cache(repo_id, filename, cache_dir=cache_dir, revision=revision)
        if isinstance(found, str):
            return found
    return None


def hub_file(repo_id: str, filename: str, models_dir: str, *, provision: bool, revision: str | None = None) -> str:
    """`hub_cached`, or -- with `provision` -- the file downloaded into
    `models_dir` by huggingface_hub (its own integrity checks, its own
    resume). Without `provision`, a miss is `Unprovisioned`."""
    found = hub_cached(repo_id, filename, models_dir, revision)
    if found is not None:
        return found
    if not provision:
        raise Unprovisioned(f"{repo_id}/{filename} is not under {models_dir} or the shared HF cache")
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename, cache_dir=models_dir, revision=revision)


@dataclass(frozen=True)
class OpenCVWeights:
    yunet: str
    sface: str | None
    arcface: str | None  # glintr100 from the insightface pack, when that pack is present


def opencv_weights(models_dir: str, *, provision: bool) -> OpenCVWeights:
    """The OpenCV stack's files. Flat files directly under `models_dir`
    (a hand-provisioned layout) are honoured first; otherwise the Hub
    cache, then a download when provisioning. SFace is only fetched when
    glintr100 is not already here -- the arcface embedder is preferred
    and needs no second download."""
    flat_yunet = os.path.join(models_dir, YUNET_FILE)
    flat_sface = os.path.join(models_dir, SFACE_FILE)
    yunet = (
        flat_yunet if os.path.isfile(flat_yunet) else hub_file(YUNET_REPO, YUNET_FILE, models_dir, provision=provision)
    )
    arcface = None
    root = insightface_root(models_dir, provision=False)
    if root is not None:
        candidate = os.path.join(root, "models", PACK, "glintr100.onnx")
        arcface = candidate if os.path.isfile(candidate) else None
    if os.path.isfile(flat_sface):
        sface: str | None = flat_sface
    elif arcface is None:
        sface = hub_file(SFACE_REPO, SFACE_FILE, models_dir, provision=provision)
    else:
        sface = hub_cached(SFACE_REPO, SFACE_FILE, models_dir)
    return OpenCVWeights(yunet=yunet, sface=sface, arcface=arcface)


def _pack_is_whole(root: str) -> bool:
    return bool(glob.glob(os.path.join(root, "models", PACK, "*.onnx")))


def insightface_root(models_dir: str, *, provision: bool) -> str | None:
    """The FaceAnalysis root whose models/antelopev2 holds the pack: the
    run's own copy under `models_dir`, else the shared `~/.insightface`,
    else -- when provisioning -- upstream's own `ensure_available`
    fetching the pack into the run's copy. None when absent and not
    provisioning."""
    own = os.path.join(models_dir, INSIGHTFACE_SUBDIR)
    if _pack_is_whole(own):
        return own
    shared = os.path.expanduser(INSIGHTFACE_HOME)
    if _pack_is_whole(shared):
        return shared
    if not provision:
        return None
    from insightface.utils import ensure_available

    pack_dir = ensure_available("models", PACK, root=own)
    _flatten(pack_dir)
    if not _pack_is_whole(own):
        raise Unprovisioned(f"insightface fetched {PACK} into {pack_dir} but no .onnx is there")
    return own


def _flatten(pack_dir: str) -> None:
    """The release zip unpacks with the pack's name repeated one level
    down; FaceAnalysis globs only the top. Lift the files up once."""
    if glob.glob(os.path.join(pack_dir, "*.onnx")):
        return
    nested = os.path.join(pack_dir, PACK)
    if not os.path.isdir(nested):
        return
    for name in os.listdir(nested):
        shutil.move(os.path.join(nested, name), os.path.join(pack_dir, name))
    os.rmdir(nested)
