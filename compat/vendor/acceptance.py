"""VENDOR ACCEPTANCE: the pinned upstream reproducing its own example.

The layer under everything else. `compat/consumers/` shows our adapter is
self-consistent; `compat/vendor/conformance.py` shows it agrees with itself on
the vendor's data. Neither shows the VENDOR runs. Until upstream reproduces
its own example on its own input with its own checkpoints, "our adapter
matches" is a comparison against an unvalidated reference.

WHAT IS AND IS NOT AVAILABLE HERE
---------------------------------
Most of the population conditions a diffusion model -- FLUX, SDXL -- whose
checkpoints are not on this machine. Verified rather than assumed:
`C:/ComfyUI/output/.AImodels` holds insightface, YuNet, SFace, mobile_sam,
CLIP-ViT-B-32, DINOv2-small and Qwen VL, and no base model. Those consumers
are UNSUPPORTED -- a fact about this machine -- and NOT
VENDOR_BASELINE_UNAVAILABLE, which is a fact about the upstream.

A diffusion model is not needed to run a vendor's ID side. Eight vendors here
compute their conditioning entirely before the first sampling step, and each
of those halves runs on this box as upstream wrote it:

    consisid          process_face_embeddings_infer   id_cond   [1, 1280]
    pulid_upstream    PuLIDPipeline.get_id_embedding  id_cond   [2, 10, 2048]
    infiniteyou       extract_id_embedding            id_cond   [1, 8, 4096]
    ipadapter_upstream get_image_embeds               id_cond   [1, 4, 768]
    uniportrait       get_single_faceid_embeds        id_cond   [1, 16, 768]
    photomaker_v2     analyze_faces + stack           id_embeds [4, 512]
    reactor           save_face_model/load_face_model 9 keys, 4392 B

ConsisID is the widest of them: its entrypoint is committed in diffusers,
which the manifest pins and `provenance.py` verifies byte-for-byte against
that commit, and every weight it needs is published by the vendor in one tree
(BestWishYsh/ConsisID-preview face_encoder/). It hardcodes
`CUDAExecutionProvider` and `ctx_id=0` (consisid_utils.py:325-327, 347-349);
this box reports
`['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`,
so it runs unmodified rather than under a CPU substitution that would be a
different arithmetic path.

THE BOUNDARIES ARE NOT ONE BOUNDARY
-----------------------------------
There is no shared shape and no shared primitive. Two vendors reading the SAME
buffalo_l pack disagree about what they take off it: IP-Adapter FaceID takes
`normed_embedding` and PhotoMaker takes the raw `embedding`, so one stored
vector cannot serve both. Selection disagrees too -- ReActor, IP-Adapter and
PhotoMaker index `faces[0]`, while UniPortrait, InfiniteYou and PuLID sort by
bbox area first. And UniPortrait aligns at 224 where every antelopev2 consumer
aligns at 112.

Only ReActor persists anything, and what it persists is upstream's own answer
to this suite's question rather than a reading of one.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
import time
from collections.abc import Callable, Generator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

import proc
from compat.harness import provenance
from compat.producers import insightface_pass as producer

ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Every git call here is bounded. `pinned_source.py:44` and
#: `provenance.py:44` already were, with the reason: a hang turns a red
#: gate into a run that never finishes, which reports nothing at all.

#: The vendor's own checkpoint tree, downloaded from the repo its README names
#: and kept outside this repository.
CONSISID_MODELS: Final[Path] = ROOT.parent.parent / "sg-vendor-fixtures" / "consisid"

#: PuLID's own ID adapter, from guozinan/PuLID -- the repo its README Model Zoo
#: names. `get_id_embedding` runs the adapter (pipeline.py:210-214).
#:
#: v1, NOT v1.1: `pulid/pipeline.py` builds `IDEncoder` (encoders.py) with
#: `id_adapter.body.*` keys while `pulid/pipeline_v1_1.py` builds `IDFormer`
#: (encoders_transformer.py) with `id_adapter.id_embedding_mapping.*`, so v1.1
#: weights fail on every key.
#: v1 is also the path with no substitutions: it takes `img2tensor` from
#: pulid/utils.py, while v1.1 takes it from basicsr, which is not installed
#: here.
PULID_WEIGHT: Final[Path] = ROOT.parent.parent / "sg-vendor-fixtures" / "pulid" / "pulid_v1.bin"

#: Everything downloaded from a vendor's own published tree, kept outside this
#: repository.
VENDOR: Final[Path] = ROOT.parent.parent / "sg-vendor-fixtures"

#: insightface's buffalo_l, and the root a `FaceAnalysis` is given -- it looks
#: under `<root>/models/<name>`. ReActor names the pack outright
#: (scripts/reactor_swapper.py:100-102), IP-Adapter's own faceid notebook names
#: it (visualization_attnmap_faceid.ipynb:72), and PhotoMaker passes no `name=`
#: at all (inference_scripts/inference_pmv2.py:13), which is insightface's
#: default and is also buffalo_l. antelopev2 -- what ConsisID, PuLID and
#: InfiniteYou use -- is a DIFFERENT recognition space: substituting it would
#: change every embedding below without failing anything.
#: The same root `producers/insightface_pass.py` resolves, not a second
#: copy of the literal: three modules held it and only one was overridable.
BUFFALO_ROOT: Final[Path] = producer.MODELS_ROOT
BUFFALO: Final[tuple[str, ...]] = (
    "models/buffalo_l/det_10g.onnx",
    "models/buffalo_l/w600k_r50.onnx",
    "models/buffalo_l/1k3d68.onnx",
    "models/buffalo_l/2d106det.onnx",
    "models/buffalo_l/genderage.onnx",
)

#: Files `prepare_face_models` opens, relative to `CONSISID_MODELS`.
REQUIRED: Final[tuple[str, ...]] = (
    "face_encoder/EVA02_CLIP_L_336_psz14_s6B.pt",
    "face_encoder/detection_Resnet50_Final.pth",
    "face_encoder/parsing_bisenet.pth",
    "face_encoder/models/antelopev2/glintr100.onnx",
    "face_encoder/models/antelopev2/scrfd_10g_bnkps.onnx",
)


@dataclass
class Acceptance:
    """One vendor's own example, run as upstream specifies it."""

    consumer_id: str
    entrypoint: str
    repo: str
    commit: str
    fixture_path: str
    fixture_sha256: str
    ran: bool
    #: Where the input came from. `vendor_commit` is a file the vendor commits
    #: at the pin; anything else names what stood in and why, so a corpus
    #: photograph is never read as the vendor's own example.
    fixture_origin: str = "vendor_commit"
    reason: str = ""
    boundary: dict[str, Any] = field(default_factory=dict)
    weights: list[dict[str, str]] = field(default_factory=list)
    seconds: float = 0.0


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def _digest_array(values: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(values.dtype).encode("ascii"))
    hasher.update(repr(values.shape).encode("ascii"))
    hasher.update(np.ascontiguousarray(values).tobytes())
    return hasher.hexdigest()


def missing_weights() -> list[str]:
    return [one for one in REQUIRED if not (CONSISID_MODELS / one).is_file()]


def consisid_fixture() -> Path | None:
    """The vendor's own example image, extracted from its pinned commit."""

    manifest = provenance.load_manifest()
    refs_root = (ROOT.parent / manifest["refs_root"]).resolve()
    row = next((one for one in manifest["consumers"] if one["id"] == "consisid"), None)
    if row is None:
        return None
    clone = provenance.clone_dir(refs_root, row["repo"])
    where = CONSISID_MODELS / "fixtures" / "example_1.png"
    where.parent.mkdir(parents=True, exist_ok=True)
    if not where.is_file():
        argv: list[str] = [
            "git",
            "-C",
            str(clone),
            "cat-file",
            "blob",
            f"{row['commit']}:asserts/example_images/1.png",
        ]
        code, out, _ = proc.run(argv, timeout=proc.LOCAL_SECONDS)
        if code != 0:
            return None
        where.write_bytes(out)
    return where


def run_consisid() -> Acceptance:
    """ConsisID's own entrypoint, its own checkpoints, its own example image."""
    import torch
    from diffusers.pipelines.consisid.consisid_utils import (
        prepare_face_models,
        process_face_embeddings_infer,
    )

    manifest = provenance.load_manifest()
    row = next(one for one in manifest["consumers"] if one["id"] == "consisid")
    upstream = manifest["upstreams"]["diffusers"]

    fixture = consisid_fixture()
    held = Acceptance(
        consumer_id="consisid",
        # The entrypoint is committed in diffusers, which is what the manifest
        # pins for it and what provenance verifies the installed bytes against.
        entrypoint="diffusers/pipelines/consisid/consisid_utils.py::process_face_embeddings_infer",
        repo=upstream["repo"],
        commit=upstream["commit"],
        fixture_path=str(fixture) if fixture else "",
        fixture_sha256=_sha256(fixture) if fixture else "",
        ran=False,
        weights=[
            {"file": one, "sha256": _sha256(CONSISID_MODELS / one)}
            for one in REQUIRED
            if (CONSISID_MODELS / one).is_file()
        ],
    )
    if fixture is None:
        held.reason = f"consisid fixture absent: asserts/example_images/1.png at {row['commit'][:12]}"
        return held
    absent = missing_weights()
    if absent:
        held.reason = f"UNSUPPORTED on this machine: {', '.join(absent)} not under {CONSISID_MODELS}"
        return held

    began = time.perf_counter()
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        models = prepare_face_models(str(CONSISID_MODELS), device=device, dtype=dtype)
        face_helper_1, face_helper_2, face_clip_model, face_main_model, mean, std = models
        id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(
            face_helper_1,
            face_clip_model,
            face_helper_2,
            mean,
            std,
            face_main_model,
            device,
            dtype,
            str(fixture),
            is_align_face=True,
        )
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError) as problem:
        held.reason = f"{type(problem).__name__}: {problem}"
        held.seconds = time.perf_counter() - began
        return held

    cond = np.asarray(id_cond.detach().float().cpu().numpy())
    kps = np.asarray(face_kps)
    held.ran = True
    held.seconds = time.perf_counter() - began
    held.boundary = {
        "device": device,
        "dtype": str(dtype),
        "id_cond": {"shape": list(cond.shape), "dtype": str(cond.dtype), "sha256": _digest_array(cond)},
        "id_vit_hidden": [list(np.asarray(one.detach().float().cpu().numpy()).shape) for one in id_vit_hidden],
        "face_kps": {"shape": list(kps.shape), "sha256": _digest_array(kps)},
        "align_crop_size": list(np.asarray(image).shape),
    }
    return held


def run_pulid() -> Acceptance:
    """PuLID's own `get_id_embedding`, from its pinned blob, on its own image.

    `PuLIDPipeline.__init__` builds the SDXL UNet and the ID adapter
    (pulid/pipeline.py:33-91), neither of which this box has. But
    `get_id_embedding` (:144-205) is entirely deterministic and closes over
    only four things, ALL of which are here:

        self.app            insightface antelopev2 FaceAnalysis
        self.handler_ante   glintr100, used only if insightface finds no face
        self.face_helper    facexlib FaceRestoreHelper + its bisenet parser
        self.clip_vision_model  EVA02-CLIP-L-14-336

    So the method is loaded out of the pinned commit and given a stand-in
    carrying those four. That is upstream's own bytes over upstream's own
    weights -- not a reimplementation -- and the boundary is `id_cond`, which
    precedes every sampling step.

    Worth contrasting with ConsisID: both concatenate an antelopev2 embedding
    with an EVA-CLIP vision embedding, but PuLID greys and face-parses the
    aligned crop first (:183-192) and L2-normalises the vision half (:200-201).
    Same shape, different construction.
    """
    import cv2
    import numpy as np
    import torch

    from compat.harness import pinned_source

    manifest = provenance.load_manifest()
    row = next(one for one in manifest["consumers"] if one["id"] == "pulid_upstream")
    refs_root = (ROOT.parent / manifest["refs_root"]).resolve()
    clone = provenance.clone_dir(refs_root, row["repo"])

    fixture = _vendor_blob(clone, row["commit"], "example_inputs/lecun.jpg", "pulid", "lecun.jpg")
    held = Acceptance(
        consumer_id="pulid_upstream",
        entrypoint="pulid/pipeline.py::PuLIDPipeline.get_id_embedding",
        repo=row["repo"],
        commit=row["commit"],
        fixture_path=str(fixture) if fixture else "",
        fixture_sha256=_sha256(fixture) if fixture else "",
        ran=False,
        weights=[
            {"file": one, "sha256": _sha256(CONSISID_MODELS / one)}
            for one in REQUIRED
            if (CONSISID_MODELS / one).is_file()
        ],
    )
    if fixture is None:
        held.reason = "pulid fixture absent: example_inputs/lecun.jpg"
        return held
    absent = [*missing_weights(), *([] if PULID_WEIGHT.is_file() else [str(PULID_WEIGHT)])]
    if absent:
        held.reason = f"UNSUPPORTED on this machine: {', '.join(absent)}"
        return held

    began = time.perf_counter()
    try:
        from diffusers.pipelines.consisid.consisid_utils import prepare_face_models

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # float32, NOT the fp16 ConsisID runs at. PuLID's own `img2tensor`
        # (pulid/utils.py) yields float32 and the method feeds it straight to
        # the vision model, so an fp16 model raises "Input type (float) and
        # bias type (struct c10::Half) should be the same". Casting the input
        # would change upstream's arithmetic; running the model at the input's
        # own width does not.
        dtype = torch.float32
        # ConsisID's loader builds exactly the four models PuLID's method
        # closes over, from the SAME published weights, so they are borrowed
        # rather than re-instantiated a second way.
        helper, ante, clip_vision, app, mean, std = prepare_face_models(str(CONSISID_MODELS), device, dtype)

        # `img2tensor` and `tensor2img` come from PuLID's OWN pulid/utils.py
        # (pipeline.py:23), not from basicsr, so they are loaded from the same
        # commit rather than substituted with facexlib's equivalents. The
        # torchvision names are upstream's own imports (pipeline.py:17-18).
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import normalize, resize

        helpers: dict[str, Any] = {}
        for name in ("img2tensor", "tensor2img"):
            fn, _ = pinned_source.load_symbol(
                clone, row["commit"], "pulid/utils.py", name, {"torch": torch, "np": np, "cv2": cv2}
            )
            helpers[name] = fn

        gray_method, _ = pinned_source.load_symbol(
            clone, row["commit"], "pulid/pipeline.py", "PuLIDPipeline.to_gray", {"torch": torch}
        )

        method, proof = pinned_source.load_symbol(
            clone,
            row["commit"],
            "pulid/pipeline.py",
            "PuLIDPipeline.get_id_embedding",
            {
                "torch": torch,
                "cv2": cv2,
                "np": np,
                "normalize": normalize,
                "resize": resize,
                "InterpolationMode": InterpolationMode,
                **helpers,
            },
        )

        # `get_id_embedding` does NOT stop at id_cond: pipeline.py:210-214
        # feeds it and the ViT hidden states through the ID adapter and
        # returns cat(uncond, cond). So the adapter is part of the boundary,
        # and it is PuLID's own IDEncoder carrying PuLID's own weights.
        encoder_cls, _ = pinned_source.load_symbol(
            clone, row["commit"], "pulid/encoders.py", "IDEncoder", {"torch": torch, "nn": torch.nn}
        )
        adapter = encoder_cls().to(device)
        # load_pretrain (pipeline.py:124-137) groups the checkpoint by the
        # first key component and loads each group into the attribute of that
        # name; only `id_adapter` is needed here.
        held_state = torch.load(str(PULID_WEIGHT), map_location="cpu")
        prefix = "id_adapter."
        adapter.load_state_dict(
            {k[len(prefix) :]: v for k, v in held_state.items() if k.startswith(prefix)}, strict=True
        )
        adapter.eval()

        class Stand:
            """Exactly the attributes `get_id_embedding` reads, and no others."""

            def __init__(self) -> None:
                self.app = app
                self.handler_ante = ante
                self.face_helper = helper
                self.clip_vision_model = clip_vision
                self.eva_transform_mean = mean
                self.eva_transform_std = std
                self.id_adapter = adapter
                self.device = device
                self.debug_img_list: list[Any] = []

            # `to_gray` is loaded from the pin like every other method here.
            # It was hand-copied arithmetic under a comment citing
            # `pulid/pipeline.py:41-44`, which is the SDXL loader; the method
            # is at :139-142. A transcribed constant with a wrong line number
            # is the exact thing this module says it does not do.
            to_gray = gray_method

        # RGB: get_id_embedding's docstring says "numpy rgb image, range
        # [0, 255]" (pipeline.py:146-148) and its first act is RGB2BGR.
        rgb = _decode_rgb(fixture)
        out = method(Stand(), rgb)
        id_cond = out[0] if isinstance(out, tuple) else out
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError, IndexError) as problem:
        held.reason = f"{type(problem).__name__}: {problem}"
        held.seconds = time.perf_counter() - began
        return held

    cond = np.asarray(id_cond.detach().float().cpu().numpy())
    held.ran = True
    held.seconds = time.perf_counter() - began
    held.boundary = {
        "device": device,
        "dtype": str(dtype),
        "symbol_sha256": proof.blob_sha256,
        "id_cond": {"shape": list(cond.shape), "dtype": str(cond.dtype), "sha256": _digest_array(cond)},
    }
    return held


def _largest_face(found: list[Any], where: str) -> Any:
    """The maximum-area face, or a named failure.

    Its own function so the `raise` is not inside the `try` that records a
    run's reason: an undetected face is a fixture problem, not a vendor whose
    entrypoint failed to reproduce, and the evidence must not read them alike.

    Largest by bbox area is upstream's own rule -- InfiniteYou nodes.py:198,
    PuLID pipeline.py:153-155, InstantID README:167 all sort by
    (x2-x1)*(y2-y1) and take the last.
    """
    if not found:
        raise ValueError(f"no face detected in {where}")
    return max(found, key=lambda one: (one.bbox[2] - one.bbox[0]) * (one.bbox[3] - one.bbox[1]))


def _first_face(found: list[Any], where: str) -> Any:
    """The face at index 0, or a named failure.

    Its own function for the same reason `_largest_face` is: the `raise` must
    sit outside the `try` that records a run's reason, so an undetected face
    reads as a fixture problem rather than as a vendor that failed to
    reproduce.

    First rather than largest is upstream's own rule here -- ReActor
    nodes.py:761, IP-Adapter visualization_attnmap_faceid.ipynb:177 and
    PhotoMaker inference_pmv2.py:75 all index [0] with no sort.
    """
    if not found:
        raise ValueError(f"no face detected in {where}")
    return found[0]


def _placed(model: Any, device: str, dtype: Any) -> Any:
    """One transformers model on one device at one width, in eval mode.

    `PreTrainedModel.to` is declared `@wraps(torch.nn.Module.to)`
    (transformers modeling_utils.py:3687-3688). `functools.wraps` copies the
    UNBOUND signature onto the wrapper, so a static reader sees `self` as the
    first positional and reports the device argument as a model. The call is
    `torch.nn.Module.to`'s documented one; the parameter is untyped here
    because the third-party declaration cannot describe it, not to silence a
    check -- a wrong device or dtype still fails at the next forward pass.
    """
    return model.to(device, dtype).eval()


def _nonempty(values: list[Any], where: str) -> list[Any]:
    """A list with something in it, or a named failure.

    Its own function so the `raise` sits outside the `try` that records a
    run's reason, for the same reason `_largest_face` is.
    """
    if not values:
        raise ValueError(f"no face detected in {where}")
    return values


def _written(target: Path) -> Path:
    """The file a vendor's writer claims to have produced, or a named failure.

    `save_face_model` (reactor_utils.py:184-200) swallows every exception into
    a print, so a failed write returns normally. This is what turns that back
    into a failure.
    """
    if not target.is_file():
        raise ValueError(f"the vendor's writer produced no file at {target}")
    return target


def _decode_rgb(where: Path) -> np.ndarray:
    """One image file as an RGB array, or a named failure.

    Separate from its caller so the `raise` is not inside the `try` that
    records a run's reason: a decode that fails is a broken fixture, not a
    vendor whose entrypoint did not reproduce, and the two must not read the
    same in the evidence.
    """
    import cv2

    bgr = cv2.imdecode(np.frombuffer(where.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cv2 could not decode {where}")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def _blob_bytes(clone: Path, commit: str, path: str) -> bytes:
    """One file's bytes at one commit, without touching the working tree."""
    code, out, _ = proc.run(
        ["git", "-C", str(clone), "cat-file", "blob", f"{commit}:{path}"], timeout=proc.LOCAL_SECONDS
    )
    if code != 0:
        raise LookupError(f"{path} is not at {commit[:12]} in {clone.name}")
    return out


def _tree_against_pin(clone: Path, commit: str, paths: tuple[str, ...]) -> dict[str, str]:
    """Working-tree files that differ from their blob at `commit`.

    Needed wherever a lane imports from a clone rather than loading pinned
    bytes. `provenance` proves the clone is AT a commit; it does not prove the
    files are unmodified, and an editable install or a stray patch is exactly
    the case where a recorded digest and the executed code part company.

    RAISES on any difference. Returning the drift and letting the caller
    decide would put a `raise` inside the recording `try`, where it would be
    caught and filed as "this vendor did not run here" -- an absent-checkpoint
    reason, for a provenance failure.
    """
    drift: dict[str, str] = {}
    for one in paths:
        where = clone / one
        if not where.is_file():
            drift[one] = "absent from the working tree"
            continue
        if where.read_bytes() != _blob_bytes(clone, commit, one):
            drift[one] = "working tree differs from the pinned blob"
    if drift:
        raise RuntimeError(
            f"{clone.name} working tree does not match {commit[:12]}: {drift}. "
            f"The recorded source digests would describe bytes that did not run."
        )
    return drift


@contextlib.contextmanager
def _importable(clone: Path) -> Generator[None]:
    """`clone` on sys.path, bytecode off, both undone on the way out.

    Bytecode off because the mirrors under `../refs` are read-only: importing
    from one wrote `__pycache__` into three of UniPortrait's directories, which
    is this project writing to a tree it only ever reads.

    sys.path restored because it was not: an entry left behind shadows every
    later import in the process, and this module runs eight vendors back to
    back in one interpreter.
    """
    held = str(clone)
    added = held not in sys.path
    was = sys.dont_write_bytecode
    if added:
        sys.path.insert(0, held)
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = was
        if added and held in sys.path:
            sys.path.remove(held)


def _vendor_blob(clone: Path, commit: str, path: str, into: str, name: str) -> Path | None:
    """One committed vendor file, extracted to the cache outside the repo."""
    where = ROOT.parent.parent / "sg-vendor-fixtures" / into / name
    where.parent.mkdir(parents=True, exist_ok=True)
    code, blob, _ = proc.run(
        ["git", "-C", str(clone), "cat-file", "blob", f"{commit}:{path}"], timeout=proc.LOCAL_SECONDS
    )
    if code != 0:
        return None
    # The cache is checked AGAINST the pin, not instead of it: a hit that
    # returned early would publish a file edited, truncated or left over from
    # another commit as the vendor's committed bytes. The blob read is cheap; the
    # claim it supports is not.
    if where.is_file() and where.read_bytes() == blob:
        return where
    where.write_bytes(blob)
    return where


def run_infiniteyou() -> Acceptance:
    """InfiniteYou's own ID-embedding path, from its pinned blobs.

    `ExtractIDEmbedding.extract_id_embedding` (nodes.py:189-207) needs three
    things and no FLUX:

        face_detector      insightface antelopev2
        arcface_model      facexlib init_recognition_model('arcface')
        image_proj_model   Resampler carrying infu_flux_v1.0/aes_stage2/
                           image_proj_model.bin

    All three are here, so the ID half runs exactly as upstream wrote it. The
    boundary is the projected embedding -- the last deterministic artifact
    before the ControlNet, and the one every InfiniteYou case in this suite is
    quoted against.

    `Resampler`'s constructor arguments are upstream's own (nodes.py:117-126):
    dim 1280, depth 4, dim_head 64, heads 20, embedding_dim 512,
    output_dim 4096, ff_mult 4. Copied as numbers because they are numbers;
    the CLASS is loaded from the pinned commit rather than reimplemented.
    """
    import math

    import torch

    from compat.harness import pinned_source

    manifest = provenance.load_manifest()
    row = next(one for one in manifest["consumers"] if one["id"] == "infiniteyou")
    refs_root = (ROOT.parent / manifest["refs_root"]).resolve()
    clone = provenance.clone_dir(refs_root, row["repo"])
    main = manifest["upstreams"]["infiniteyou_main"]
    main_clone = provenance.clone_dir(refs_root, main["repo"])

    proj = (
        ROOT.parent.parent
        / "sg-vendor-fixtures"
        / "infiniteyou"
        / "infu_flux_v1.0"
        / "aes_stage2"
        / "image_proj_model.bin"
    )
    fixture = _vendor_blob(main_clone, main["commit"], "assets/examples/man.jpg", "infiniteyou", "man.jpg")

    held = Acceptance(
        consumer_id="infiniteyou",
        entrypoint="nodes.py::ExtractIDEmbedding.extract_id_embedding",
        repo=row["repo"],
        commit=row["commit"],
        fixture_path=str(fixture) if fixture else "",
        fixture_sha256=_sha256(fixture) if fixture else "",
        ran=False,
        weights=[{"file": proj.name, "sha256": _sha256(proj)}] if proj.is_file() else [],
    )
    if fixture is None:
        held.reason = "infiniteyou fixture absent: assets/examples/man.jpg"
        return held
    if not proj.is_file():
        held.reason = f"UNSUPPORTED on this machine: {proj} absent"
        return held

    began = time.perf_counter()
    try:
        from facexlib.recognition import init_recognition_model
        from insightface.app import FaceAnalysis
        from insightface.utils import face_align

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # `Resampler` closes over three names defined beside it in the same
        # module -- FeedForward, reshape_tensor and PerceiverAttention
        # (resampler.py:15, 25, 36). `load_symbol` extracts ONE symbol, so its
        # dependencies are loaded first, in definition order, and bound into
        # its namespace. Every one is upstream's own bytes at the pin; none is
        # reimplemented.
        namespace: dict[str, Any] = {"torch": torch, "nn": torch.nn, "math": math}
        proof = None
        for name in ("FeedForward", "reshape_tensor", "PerceiverAttention", "Resampler"):
            loaded, proof = pinned_source.load_symbol(clone, row["commit"], "resampler.py", name, dict(namespace))
            namespace[name] = loaded
        resampler_cls = namespace["Resampler"]
        model = resampler_cls(
            dim=1280, depth=4, dim_head=64, heads=20, num_queries=8, embedding_dim=512, output_dim=4096, ff_mult=4
        )
        model.load_state_dict(torch.load(str(proj), map_location="cpu")["image_proj"])
        model.to(device, torch.bfloat16).eval()

        app = FaceAnalysis(
            name="antelopev2",
            root=str(CONSISID_MODELS / "face_encoder"),
            providers=["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
        arcface = init_recognition_model("arcface", device=device)

        bgr = _decode_rgb(fixture)[:, :, ::-1].copy()
        best = _largest_face(app.get(bgr), "InfiniteYou's own example")
        seen_pack = observed_pack(app)

        # utils.py:22-29, upstream's own arithmetic: norm_crop@112, /255,
        # 2x-1, then the model returns an already-normalised [512].
        crop = face_align.norm_crop(bgr, landmark=np.array(best.kps), image_size=112)
        arc = torch.from_numpy(crop).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        arc = (2 * arc - 1).to(device).contiguous()
        with torch.no_grad():
            embed = arcface(arc)[0]
            staged = embed.clone().unsqueeze(0).float().to(device).reshape([1, -1, 512]).to(torch.bfloat16)
            projected = model(staged)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError, KeyError) as problem:
        held.reason = f"{type(problem).__name__}: {problem}"
        held.seconds = time.perf_counter() - began
        return held

    out = np.asarray(projected.detach().float().cpu().numpy())
    held.ran = True
    held.seconds = time.perf_counter() - began
    held.boundary = {
        "device": device,
        "symbol_sha256": proof.blob_sha256 if proof else "",
        "observed_pack": seen_pack,
        "id_cond": {"shape": list(out.shape), "dtype": str(out.dtype), "sha256": _digest_array(out)},
    }
    return held


def _weight_rows(root: Path, names: tuple[str, ...]) -> list[dict[str, str]]:
    return [{"file": one, "sha256": _sha256(root / one)} for one in names if (root / one).is_file()]


def _corpus_selfie() -> tuple[str, str] | None:
    """One licensed photograph of one real person, by path and digest.

    ReActor commits no image at its pin: 152 `.py` blobs and zero
    `.jpg`/`.png` from the same `git ls-tree` at 6ad6b35a4df2, the `.py` count
    being the positive control. Its README's images live in Gourieff/Assets
    and are UI screenshots, not portraits. So its acceptance runs on the
    corpus the rest of this suite validates against, and says so.

    Nothing is copied, resized or emitted: the path and the sha256 are
    recorded and the file is read where it lies.
    """
    from compat.corpus import index as corpus_index

    files = corpus_index.KYC / "files"
    if not files.is_dir():
        return None
    for folder in sorted(one for one in files.iterdir() if one.is_dir()):
        for image in sorted(one for one in folder.iterdir() if one.is_file()):
            if corpus_index.role_of(image.name) != "selfie":
                continue
            sha, _ = corpus_index.digest_file(image)
            return str(image), sha
    return None


def observed_embedding_kind(values: np.ndarray) -> dict[str, Any]:
    """Whether a recorded vector is L2-normalised, measured rather than declared.

    `normed_embedding` and `embedding` come off the same insightface Face and
    differ only by a division. The manifest declares which one a consumer
    takes; the vector itself answers it, because a normed one has norm 1.

    Recorded with the norm so a reader can see how far from 1 it landed
    rather than trusting a boolean.
    """
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(flat))
    return {
        "l2_norm": norm,
        "kind": "normed" if abs(norm - 1.0) < 1e-3 else "raw",
        "dims": flat.size,
    }


def observed_crop_size(crop: np.ndarray) -> int:
    """The side length of a crop the vendor's own code produced.

    A number read off the array rather than copied out of a call site, so a
    consumer whose alignment size changes upstream contradicts the manifest
    instead of silently disagreeing with it.
    """
    return int(np.asarray(crop).shape[0])


def observed_pack(app: Any) -> dict[str, Any]:
    """Which model files an analyser ACTUALLY opened, read off the live object.

    The point of this function is that it is not a restatement. Every other
    `vendor_setup` field in the manifest is a value somebody typed after
    reading upstream, and this session found four of them wrong -- each with
    a citation that resolved to a line which really did construct a
    `FaceAnalysis`. A citation check cannot catch that, and neither can the
    substitution ablation: with the wrong pack it expects no break and
    observes none, so the case is green either way.

    `app.models[task].model_file` is the path onnxruntime was handed. The
    pack directory and the per-file digest therefore come from the run, and a
    manifest naming a different pack is CONTRADICTED by them.
    """
    files: dict[str, dict[str, str]] = {}
    packs: set[str] = set()
    for task, model in sorted(getattr(app, "models", {}).items()):
        where = getattr(model, "model_file", None) or getattr(model, "onnx_file", None)
        if not where:
            continue
        path = Path(str(where))
        packs.add(path.parent.name)
        files[task] = {"file": path.name, "sha256": _sha256(path) if path.is_file() else "ABSENT"}
    return {
        # One pack is the normal case. More than one means the analyser was
        # assembled from several directories, which is a fact worth failing on
        # rather than collapsing to whichever sorted first.
        "pack": next(iter(packs)) if len(packs) == 1 else f"MIXED:{sorted(packs)}",
        "modules": sorted(files),
        "files": files,
    }


def _buffalo(device: str) -> Any:
    """insightface buffalo_l, prepared as these three vendors prepare it."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        root=str(BUFFALO_ROOT),
        providers=["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
    return app


def reactor_core(clone: Path, commit: str) -> tuple[Any, dict[str, str]]:
    """ReActor's OWN face analyser, materialised as a package from the pin.

    At this pin ReActor does not use insightface. Its README states it
    outright -- version 0.7.0_alpha2 (README.md:5), and 0.7.0 ALPHA1's note
    (README.md:55) reads "New ReActor Core! No `Insightface` required!" and
    "a swap result is slightly different now ... the accuracy is actually a
    little higher than with Insightface".

    `reactor_core/analyzer.py` bears that out: it loads the SAME buffalo_l
    onnx files (:22-26) but runs them through ReActor's own SCRFD,
    ArcFaceONNX, Attribute and Landmark (:5, :50-58), not insightface's
    model_zoo. So a Face built by insightface is NOT the Face this vendor
    builds, and an acceptance that used one could not claim to be the
    vendor's own path.

    The package is written out and imported rather than pulled apart symbol
    by symbol, because `reactor_core` uses relative imports and running it
    through Python's own import machinery is what keeps them upstream's.

    Two modules are STUBBED, and both are only reachable on paths this run
    does not take: `reactor_utils.download` (used solely when the weights are
    missing, and they are present) and `scripts.reactor_logger` (status
    printing). Recorded in the returned proofs rather than hidden.
    """
    import importlib
    import sys

    scratch = ROOT.parent.parent / "sg-vendor-fixtures" / "reactor" / f"core_{commit[:12]}"
    package = scratch / "reactor_core"
    package.mkdir(parents=True, exist_ok=True)
    (scratch / "scripts").mkdir(exist_ok=True)

    blobs: dict[str, str] = {}
    for name in ("__init__.py", "face_objects.py", "meanshape_68.py", "inswap.py", "analyzer.py"):
        try:
            raw = _blob_bytes(clone, commit, f"reactor_core/{name}")
        except LookupError:
            if name != "__init__.py":
                raise
            raw = b""
        (package / name).write_bytes(raw)
        blobs[f"reactor_core/{name}"] = hashlib.sha256(raw).hexdigest()

    # The two stubs. `download` raises rather than fetching: if this run ever
    # reaches it, the weights were absent and the row must fail loudly instead
    # of quietly pulling a file the manifest never declared.
    (scratch / "reactor_utils.py").write_text(
        "def download(url, path, name=None):\n"
        "    raise RuntimeError(f'weights absent; refusing to fetch {name or url}')\n",
        encoding="utf-8",
    )
    (scratch / "scripts" / "__init__.py").write_text("", encoding="utf-8")
    (scratch / "scripts" / "reactor_logger.py").write_text(
        "class _Logger:\n"
        "    def status(self, *a, **k): pass\n"
        "    def error(self, *a, **k): pass\n"
        "    def info(self, *a, **k): pass\n"
        "logger = _Logger()\n",
        encoding="utf-8",
    )

    # `sys.path` and the evicted modules are both put back. The entry was
    # inserted and left, and `scripts` -- a name nothing here owns -- was
    # deleted from `sys.modules` for the rest of the process, so any unrelated
    # package by that name was shadowed for the seven vendors that follow.
    added = str(scratch) not in sys.path
    if added:
        sys.path.insert(0, str(scratch))
    evicted = {
        one: sys.modules[one]
        for one in list(sys.modules)
        if one.startswith(("reactor_core", "reactor_utils", "scripts"))
    }
    for one in evicted:
        del sys.modules[one]
    try:
        return importlib.import_module("reactor_core.analyzer"), blobs
    finally:
        for one, module in evicted.items():
            sys.modules.setdefault(one, module)
        if added and str(scratch) in sys.path:
            sys.path.remove(str(scratch))


def run_reactor() -> Acceptance:
    """ReActor's own persisted face format, written and read back by its own code.

    The only vendor in this population that ships a durable face record, which
    makes it the one vendor whose answer to this suite's question is written
    down rather than inferred. `save_face_model` (reactor_utils.py:184-200)
    names nine keys and `load_face_model` (:203-208) rebuilds a `Face` from
    whatever the file holds. Both are loaded from the pin and run here, and
    the nine keys are read out of upstream's own subscripts rather than
    transcribed from them.

    `blend_faces` (nodes.py:816-827) is why the gap matters: it averages ONLY
    the embedding and copies bbox, kps, det_score, landmark_3d_68, pose,
    landmark_2d_106, gender and age from `faces[0]`, so a blended model's
    geometry belongs to whichever reference came first.
    """
    import os

    import onnxruntime as ort
    import torch
    from safetensors.torch import safe_open, save_file

    from compat.harness import pinned_source
    from compat.storage.contract import Observation

    manifest = provenance.load_manifest()
    row = next(one for one in manifest["consumers"] if one["id"] == "reactor")
    refs_root = (ROOT.parent / manifest["refs_root"]).resolve()
    clone = provenance.clone_dir(refs_root, row["repo"])
    commit = row["commit"]

    sample = _corpus_selfie()
    held = Acceptance(
        consumer_id="reactor",
        entrypoint="reactor_utils.py::save_face_model + load_face_model",
        repo=row["repo"],
        commit=commit,
        fixture_path=sample[0] if sample else "",
        fixture_sha256=sample[1] if sample else "",
        fixture_origin="corpus:caucasian-people-kyc-photo-dataset (no image committed at this pin)",
        ran=False,
        weights=_weight_rows(BUFFALO_ROOT, BUFFALO),
    )
    if sample is None:
        held.reason = "corpus absent: no selfie under the KYC dataset root"
        return held
    if len(held.weights) != len(BUFFALO):
        held.reason = f"UNSUPPORTED on this machine: buffalo_l incomplete under {BUFFALO_ROOT}"
        return held

    began = time.perf_counter()
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        persists = pinned_source.subscript_keys(clone, commit, "reactor_utils.py", "save_face_model", "face")

        face_cls, _ = pinned_source.load_symbol(
            clone, commit, "reactor_core/face_objects.py", "Face", {"np": np, "ort": ort, "os": os}
        )
        # `Face` is in the signature's annotation and reactor_utils.py has no
        # `from __future__ import annotations`, so it is evaluated at def time.
        save_fn, proof = pinned_source.load_symbol(
            clone,
            commit,
            "reactor_utils.py",
            "save_face_model",
            {"torch": torch, "save_file": save_file, "Face": face_cls},
        )
        load_fn, _ = pinned_source.load_symbol(
            clone, commit, "reactor_utils.py", "load_face_model", {"safe_open": safe_open, "Face": face_cls}
        )

        # ReActor's OWN analyser, not insightface. At this pin they are not
        # the same code over the same weights, and the vendor says the results
        # differ (README.md:55).
        analyzer, core_blobs = reactor_core(clone, commit)
        app = analyzer.ReActorFaceAnalysis(
            name="buffalo_l",
            root=str(BUFFALO_ROOT),
            providers=["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

        bgr = _decode_rgb(Path(sample[0]))[:, :, ::-1].copy()
        # `build_face_model` returns `face_model[0]` (nodes.py:761): the
        # first, NOT the largest. `ReActorFaceAnalysis.get` does not sort
        # (analyzer.py:70-103), so index 0 is detector order. Upstream's rule.
        face = _first_face(app.get(bgr), sample[0])
        seen_pack = observed_pack(app)
        observed = Observation.of(face)

        # The same photograph through insightface, to measure what the vendor
        # asserts: that its own core gives a different answer. Recorded, never
        # substituted for the vendor's.
        insight = _first_face(_buffalo(device).get(bgr), sample[0])

        target = VENDOR / "reactor" / "acceptance_face.safetensors"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        save_fn(face, str(target))
        restored = load_fn(str(_written(target)))
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError, KeyError) as problem:
        held.reason = f"{type(problem).__name__}: {problem}"
        held.seconds = time.perf_counter() - began
        return held

    round_trip: dict[str, Any] = {}
    every = True
    for key in persists:
        before = np.asarray(face[key])
        after = np.asarray(restored[key])
        # Byte identity, not `array_equal`: the question is whether the same
        # bytes came back, and `array_equal` reports a NaN as unequal to
        # itself -- which would read as storage loss where there was none.
        same = (
            before.shape == after.shape
            and before.dtype == after.dtype
            and np.ascontiguousarray(before).tobytes() == np.ascontiguousarray(after).tobytes()
        )
        every = every and same
        round_trip[key] = {
            "shape": list(after.shape),
            "dtype": str(after.dtype),
            "sha256": _digest_array(after),
            "survives_round_trip": same,
        }

    # What the vendor CLAIMS in README.md:55 -- that its own core answers
    # differently from insightface over the same weights -- measured per key
    # rather than taken on trust.
    against_insightface: dict[str, Any] = {}
    for key in persists:
        mine = np.asarray(face[key])
        theirs = np.asarray(insight[key]) if key in set(insight) else None
        if theirs is None:
            against_insightface[key] = "absent from the insightface Face"
            continue
        same_shape = mine.shape == theirs.shape
        against_insightface[key] = {
            "identical": bool(
                same_shape and np.ascontiguousarray(mine).tobytes() == np.ascontiguousarray(theirs).tobytes()
            ),
            "max_abs_delta": float(np.max(np.abs(mine.astype(np.float64) - theirs.astype(np.float64))))
            if same_shape and mine.dtype != np.dtype("O")
            else None,
        }

    held.ran = True
    held.seconds = time.perf_counter() - began
    held.boundary = {
        "device": device,
        "symbol_sha256": proof.blob_sha256,
        "analyzer": "reactor_core.analyzer.ReActorFaceAnalysis",
        "observed_pack": seen_pack,
        "analyzer_blobs": core_blobs,
        "analyzer_stubs": ["reactor_utils.download", "scripts.reactor_logger.logger"],
        "format": "safetensors",
        "bytes": target.stat().st_size,
        "persists": list(persists),
        "round_trip": round_trip,
        "every_key_survives": every,
        # The producer emitted these and the vendor's own format has no slot
        # for them. Named because that gap is the question this suite asks.
        "producer_keys_not_persisted": sorted(set(observed) - set(persists)),
        # The substitution this suite must never make silently.
        "vs_insightface": against_insightface,
    }
    return held


def run_ipadapter_faceid() -> Acceptance:
    """IP-Adapter FaceID's own projection, on its own committed image.

    `IPAdapterFaceID.get_image_embeds` (ip_adapter/ip_adapter_faceid.py:182-188)
    reads exactly three attributes -- `device`, `torch_dtype` and
    `image_proj_model` -- and no UNet, so the whole ID half runs here. The
    projector is upstream's own `MLPProjModel` (:64-84) carrying
    `ip-adapter-faceid_sd15.bin`'s `image_proj` group.

    Its input is built the way upstream's own notebook builds it
    (visualization_attnmap_faceid.ipynb:176-178): buffalo_l, `faces[0]`, then
    `normed_embedding`. FIRST, not largest -- unlike every antelopev2 consumer
    in this population -- and the L2-normalised vector, not the raw one, which
    is a different stored primitive from the one PhotoMaker takes off the same
    pack.

    Run at the class default `torch_dtype=torch.float16` (:121) on CUDA rather
    than a widened float32: the dtype is upstream's own arithmetic and a
    boundary digest taken at another width would not be theirs.
    """
    import torch

    from compat.harness import pinned_source

    manifest = provenance.load_manifest()
    row = next(one for one in manifest["consumers"] if one["id"] == "ipadapter_upstream")
    refs_root = (ROOT.parent / manifest["refs_root"]).resolve()
    clone = provenance.clone_dir(refs_root, row["repo"])
    commit = row["commit"]

    weight = VENDOR / "ipadapter" / "ip-adapter-faceid_sd15.bin"
    fixture = _vendor_blob(clone, commit, "assets/images/woman.png", "ipadapter", "woman.png")
    held = Acceptance(
        consumer_id="ipadapter_upstream",
        entrypoint="ip_adapter/ip_adapter_faceid.py::IPAdapterFaceID.get_image_embeds",
        repo=row["repo"],
        commit=commit,
        fixture_path=str(fixture) if fixture else "",
        fixture_sha256=_sha256(fixture) if fixture else "",
        ran=False,
        weights=[
            *([{"file": weight.name, "sha256": _sha256(weight)}] if weight.is_file() else []),
            *_weight_rows(BUFFALO_ROOT, BUFFALO),
        ],
    )
    if fixture is None:
        held.reason = "ipadapter fixture absent: assets/images/woman.png"
        return held
    if not weight.is_file():
        held.reason = f"UNSUPPORTED on this machine: {weight} absent"
        return held

    began = time.perf_counter()
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        proj_cls, proof = pinned_source.load_symbol(
            clone, commit, "ip_adapter/ip_adapter_faceid.py", "MLPProjModel", {"torch": torch}
        )
        # SD1.5 cross-attention is 768, arcface is 512, and `num_tokens=4` is
        # the class default (:121). Numbers copied because they are numbers;
        # the CLASS is upstream's own bytes.
        model = proj_cls(cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4)
        model.load_state_dict(torch.load(str(weight), map_location="cpu")["image_proj"])
        model.to(device, dtype).eval()

        method, _ = pinned_source.load_symbol(
            clone, commit, "ip_adapter/ip_adapter_faceid.py", "IPAdapterFaceID.get_image_embeds", {"torch": torch}
        )

        class Stand:
            """Exactly the attributes `get_image_embeds` reads, and no others."""

            def __init__(self) -> None:
                self.device = device
                self.torch_dtype = dtype
                self.image_proj_model = model

        # The notebook's `cv2.imread` yields BGR and hands it straight to
        # `app.get`, which expects BGR.
        bgr = _decode_rgb(fixture)[:, :, ::-1].copy()
        app = _buffalo(device)
        best = _first_face(app.get(bgr), "IP-Adapter's own example")
        seen_pack = observed_pack(app)
        embeds = torch.from_numpy(best.normed_embedding).unsqueeze(0)
        seen_embedding = observed_embedding_kind(best.normed_embedding)
        cond, uncond = method(Stand(), embeds)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError, KeyError) as problem:
        held.reason = f"{type(problem).__name__}: {problem}"
        held.seconds = time.perf_counter() - began
        return held

    out = np.asarray(cond.detach().float().cpu().numpy())
    null = np.asarray(uncond.detach().float().cpu().numpy())
    held.ran = True
    held.seconds = time.perf_counter() - began
    held.boundary = {
        "device": device,
        "dtype": str(dtype),
        "symbol_sha256": proof.blob_sha256,
        "selection": "faces[0]",
        "observed_pack": seen_pack,
        "observed_embedding": seen_embedding,
        "retained_primitive": "normed_embedding",
        "id_cond": {"shape": list(out.shape), "dtype": str(out.dtype), "sha256": _digest_array(out)},
        "uncond": {"shape": list(null.shape), "sha256": _digest_array(null)},
    }
    return held


def run_uniportrait() -> Acceptance:
    """UniPortrait's own `get_single_faceid_embeds`, on its own committed image.

    `UniPortraitPipeline.__init__` builds an SD1.5 UNet this box does not
    have, but `get_single_faceid_embeds` (uniportrait_pipeline.py:176-217)
    reads only six attributes and none of them is the pipeline:

        clip_image_processor    CLIPImageProcessor, upstream's own settings
        clip_image_encoder      CLIP ViT-H image encoder, h94/IP-Adapter
        facerecog_model         IR_101 + glint360k_curricular_face_r101
        faceid_proj_model       UniPortraitFaceIDResampler + faceid_proj
        device, torch_dtype

    `dim` is 768 because the constructor's own default is 768
    (resampler.py:96) and the method's own comment records the result as
    `[b, 16, 768]` (:210).

    The input is what upstream's own demo builds: largest face by bbox area,
    then `norm_crop` at 224 (gradio_app.py:98-101) -- a 224 crop, where every
    antelopev2 consumer in this population aligns at 112. `face_structure_scale`
    is 0.0, the demo's own default (gradio_app.py:144).

    `curricular_face` is a four-module package with relative imports, so it is
    imported from the verified clone rather than extracted symbol by symbol;
    each module's blob digest at the pin is recorded beside the weights.
    """
    import importlib
    import math

    import torch

    from compat.harness import pinned_source

    manifest = provenance.load_manifest()
    row = next(one for one in manifest["consumers"] if one["id"] == "uniportrait")
    refs_root = (ROOT.parent / manifest["refs_root"]).resolve()
    clone = provenance.clone_dir(refs_root, row["repo"])
    commit = row["commit"]

    backbone = VENDOR / "uniportrait" / "glint360k_curricular_face_r101_backbone.bin"
    faceid = VENDOR / "uniportrait" / "uniportrait-faceid_sd15.bin"
    encoder = VENDOR / "uniportrait" / "models" / "image_encoder"
    fixture = _vendor_blob(clone, commit, "assets/examples/1-newton.jpg", "uniportrait", "1-newton.jpg")

    held = Acceptance(
        consumer_id="uniportrait",
        entrypoint="uniportrait/uniportrait_pipeline.py::UniPortraitPipeline.get_single_faceid_embeds",
        repo=row["repo"],
        commit=commit,
        fixture_path=str(fixture) if fixture else "",
        fixture_sha256=_sha256(fixture) if fixture else "",
        ran=False,
        weights=[
            *[{"file": one.name, "sha256": _sha256(one)} for one in (backbone, faceid) if one.is_file()],
            *(
                [{"file": "image_encoder/model.safetensors", "sha256": _sha256(encoder / "model.safetensors")}]
                if (encoder / "model.safetensors").is_file()
                else []
            ),
        ],
    )
    if fixture is None:
        held.reason = "uniportrait fixture absent: assets/examples/1-newton.jpg"
        return held
    needed = (backbone, faceid, encoder / "model.safetensors", encoder / "config.json")
    absent = [str(one) for one in needed if not one.is_file()]
    if absent:
        held.reason = f"UNSUPPORTED on this machine: {', '.join(absent)}"
        return held

    began = time.perf_counter()
    try:
        from insightface.utils import face_align
        from PIL import Image
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        curricular = (
            "uniportrait/curricular_face/__init__.py",
            "uniportrait/curricular_face/backbone/__init__.py",
            "uniportrait/curricular_face/backbone/common.py",
            "uniportrait/curricular_face/backbone/model_irse.py",
        )
        sources = {one: hashlib.sha256(_blob_bytes(clone, commit, one)).hexdigest() for one in curricular}
        # The digests above describe the pinned blobs; the import below runs
        # the WORKING TREE. Checked rather than assumed, so the stamp cannot
        # be cleaner than what executed.
        _tree_against_pin(clone, commit, curricular)
        with _importable(clone):
            backbones = importlib.import_module("uniportrait.curricular_face.backbone")
        facerecog = backbones.get_model("IR_101")([112, 112])
        facerecog.load_state_dict(torch.load(str(backbone), map_location="cpu"))
        facerecog = facerecog.to(device, dtype).eval()

        # uniportrait_pipeline.py:55-57, upstream's own processor settings.
        # `use_square_size` is accepted at image_processing_clip.py:36-40 --
        # it rewrites `size` to {"height": 224, "width": 224} -- but is absent
        # from the typed `ImagesKwargs`, so the kwargs are passed as a mapping
        # rather than rewritten into their post-condition here.
        settings: dict[str, Any] = {
            "size": {"shortest_edge": 224},
            "do_center_crop": False,
            "use_square_size": True,
        }
        processor = CLIPImageProcessor(**settings)
        vision = CLIPVisionModelWithProjection.from_pretrained(str(encoder))
        hidden = vision.config.hidden_size
        clip_vision = _placed(vision, device, dtype)

        namespace: dict[str, Any] = {"torch": torch, "nn": torch.nn, "math": math}
        proof = None
        for name in ("FeedForward", "reshape_tensor", "PerceiverAttention", "UniPortraitFaceIDResampler"):
            loaded, proof = pinned_source.load_symbol(clone, commit, "uniportrait/resampler.py", name, dict(namespace))
            namespace[name] = loaded
        resampler = (
            namespace["UniPortraitFaceIDResampler"](
                intrinsic_id_embedding_dim=512,
                structure_embedding_dim=64 + 128 + 256 + hidden,
                num_tokens=16,
                depth=6,
                dim=768,
                dim_head=64,
                heads=12,
                ff_mult=4,
                output_dim=768,
            )
            .to(device, dtype)
            .eval()
        )
        # uniportrait_pipeline.py:150, strict=True as upstream loads it.
        resampler.load_state_dict(torch.load(str(faceid), map_location="cpu")["faceid_proj"], strict=True)

        method, _ = pinned_source.load_symbol(
            clone,
            commit,
            "uniportrait/uniportrait_pipeline.py",
            "UniPortraitPipeline.get_single_faceid_embeds",
            {"torch": torch, "F": torch.nn.functional},
        )

        class Stand:
            """Exactly the attributes the method reads, and no others."""

            def __init__(self) -> None:
                self.clip_image_processor = processor
                self.clip_image_encoder = clip_vision
                self.facerecog_model = facerecog
                self.faceid_proj_model = resampler
                self.device = device
                self.torch_dtype = dtype

        bgr = _decode_rgb(fixture)[:, :, ::-1].copy()
        app = _buffalo(device)
        best = _largest_face(app.get(bgr), "UniPortrait's own example")
        seen_pack = observed_pack(app)
        crop = face_align.norm_crop(bgr, landmark=np.array(best.kps), image_size=224)
        seen_crop = observed_crop_size(crop)
        aligned = Image.fromarray(np.ascontiguousarray(crop[:, :, ::-1]))
        with torch.no_grad():
            cond, uncond = method(Stand(), [aligned], 0.0)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError, KeyError) as problem:
        held.reason = f"{type(problem).__name__}: {problem}"
        held.seconds = time.perf_counter() - began
        return held

    out = np.asarray(cond.detach().float().cpu().numpy())
    null = np.asarray(uncond.detach().float().cpu().numpy())
    held.ran = True
    held.seconds = time.perf_counter() - began
    held.boundary = {
        "device": device,
        "dtype": str(dtype),
        "symbol_sha256": proof.blob_sha256 if proof else "",
        "curricular_face_blobs": sources,
        "selection": "largest_bbox_area",
        "observed_pack": seen_pack,
        "observed_crop_size": seen_crop,
        # 224, where every antelopev2 consumer in this population aligns at
        # 112. A retained 112 crop cannot serve this consumer.
        "align_crop_size": 224,
        "face_structure_scale": 0.0,
        "id_cond": {"shape": list(out.shape), "dtype": str(out.dtype), "sha256": _digest_array(out)},
        "uncond": {"shape": list(null.shape), "sha256": _digest_array(null)},
    }
    return held


def run_photomaker() -> Acceptance:
    """PhotoMaker v2's own detection sweep and ID stack, on its own examples.

    `PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken.forward` (model_v2.py:142)
    takes `prompt_embeds`, so the encoder itself needs SDXL's text tower and
    is UNSUPPORTED here. Everything upstream computes BEFORE it is not:
    `analyze_faces` (photomaker/insightface_package.py:20-29) and the stack at
    inference_pmv2.py:71-81 produce `id_embeds` with no diffusion model in
    sight, and that is the tensor the encoder is handed.

    Two properties of the sweep are why this runs rather than being read.
    `FaceAnalysis2.get` (:14-18) MUTATES `self.det_model.input_size` and never
    restores it, and `analyze_faces` begins its list with `None`, which means
    "whatever size the previous call left behind". So which det_size found a
    face depends on what ran before it, and the size that succeeded is
    recorded nowhere downstream. Both are measured here.

    The retained primitive is `faces[0]['embedding']` -- RAW, not
    `normed_embedding`, which is what IP-Adapter FaceID takes off the same
    pack. One stored vector cannot be both.
    """
    import torch

    from compat.harness import pinned_source

    manifest = provenance.load_manifest()
    row = next(one for one in manifest["consumers"] if one["id"] == "photomaker_v2")
    refs_root = (ROOT.parent / manifest["refs_root"]).resolve()
    clone = provenance.clone_dir(refs_root, row["repo"])
    commit = row["commit"]

    # inference_pmv2.py:59 reads a whole identity folder; newton_man is the
    # four-image set upstream ships and the one this suite already quotes.
    committed = (
        "examples/newton_man/newton_0.jpg",
        "examples/newton_man/newton_1.jpg",
        "examples/newton_man/newton_2.png",
        "examples/newton_man/newton_3.jpg",
    )
    fixtures = [_vendor_blob(clone, commit, one, "photomaker_v2", Path(one).name) for one in committed]
    present = [one for one in fixtures if one is not None]

    held = Acceptance(
        consumer_id="photomaker_v2",
        entrypoint="photomaker/insightface_package.py::analyze_faces",
        repo=row["repo"],
        commit=commit,
        fixture_path=", ".join(str(one) for one in present),
        fixture_sha256=hashlib.sha256("".join(_sha256(one) for one in present).encode("ascii")).hexdigest(),
        ran=False,
        weights=_weight_rows(BUFFALO_ROOT, BUFFALO),
    )
    if len(present) != len(committed):
        held.reason = f"photomaker fixtures absent: {len(committed) - len(present)} of {len(committed)}"
        return held
    if len(held.weights) != len(BUFFALO):
        held.reason = f"UNSUPPORTED on this machine: buffalo_l incomplete under {BUFFALO_ROOT}"
        return held

    began = time.perf_counter()
    try:
        from insightface.app import FaceAnalysis
        from insightface.data import get_image as ins_get_image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        namespace: dict[str, Any] = {"np": np, "FaceAnalysis": FaceAnalysis, "ins_get_image": ins_get_image}
        analysis_cls, proof = pinned_source.load_symbol(
            clone, commit, "photomaker/insightface_package.py", "FaceAnalysis2", dict(namespace)
        )
        analyze, _ = pinned_source.load_symbol(
            clone, commit, "photomaker/insightface_package.py", "analyze_faces", dict(namespace)
        )

        # inference_pmv2.py:13 passes no `name=`, which is insightface's
        # buffalo_l default, and restricts the modules to detection and
        # recognition -- so no landmark_2d_106, no pose, no genderage.
        detector = analysis_cls(
            name="buffalo_l",
            root=str(BUFFALO_ROOT),
            providers=["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        detector.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

        sweep: list[dict[str, Any]] = []
        vectors: list[Any] = []
        # Bound before the loop: it is only assigned when a photograph yields
        # a face, and a run where none does must fail on `_nonempty` below
        # rather than on an unbound name three lines later.
        seen_embedding: dict[str, Any] = {}
        per_reference: list[dict[str, Any]] = []
        for one in present:
            before = tuple(detector.det_model.input_size)
            found = analyze(detector, _decode_rgb(one)[:, :, ::-1].copy())
            sweep.append(
                {
                    "fixture": one.name,
                    "input_size_before": list(before),
                    "input_size_after": list(detector.det_model.input_size),
                    "faces": len(found),
                }
            )
            if found:
                vectors.append(torch.from_numpy(found[0]["embedding"]))
                # Every reference's kind, not the last one's. `id_embeds`
                # stacks all of them, so a single overwritten dict described
                # whichever photograph happened to come last while the tensor
                # beside it described all four -- and `declared_against_observed`
                # read that one dict as the run's embedding kind.
                per_reference.append(observed_embedding_kind(found[0]["embedding"]))
        id_embeds = torch.stack(_nonempty(vectors, "any PhotoMaker example"))
        kinds = {one["kind"] for one in per_reference}
        seen_embedding = (
            {**per_reference[0], "references": len(per_reference)}
            if len(kinds) == 1 and per_reference
            else {"kind": f"MIXED {sorted(kinds)}", "dims": 0, "l2_norm": 0.0, "references": len(per_reference)}
        )
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError, KeyError) as problem:
        held.reason = f"{type(problem).__name__}: {problem}"
        held.seconds = time.perf_counter() - began
        return held

    out = np.asarray(id_embeds.detach().float().cpu().numpy())
    held.ran = True
    held.seconds = time.perf_counter() - began
    held.boundary = {
        "device": device,
        "symbol_sha256": proof.blob_sha256,
        "selection": "faces[0]",
        "observed_pack": observed_pack(detector),
        "observed_embedding": seen_embedding,
        "retained_primitive": "embedding",
        "combiner": "torch.stack",
        "references": len(vectors),
        "detection_sweep": sweep,
        # The sweep mutates the detector and restores nothing, so a later
        # call's leading `None` inherits this. Recorded because nothing in
        # upstream's own path records it.
        "det_size_left_behind": list(sweep[-1]["input_size_after"]),
        "id_embeds": {"shape": list(out.shape), "dtype": str(out.dtype), "sha256": _digest_array(out)},
        "encoder_unsupported": (
            "PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken.forward (model_v2.py:142) "
            "requires SDXL prompt_embeds; no SDXL checkpoint on this machine"
        ),
    }
    return held


def run_instantid() -> Acceptance:
    """InstantID's own ID path, on its own committed image.

    Two artifacts condition InstantID and neither needs SDXL. The first is
    `_encode_prompt_image_emb` (pipeline_stable_diffusion_xl_instantid.py:220-240)
    over the `Resampler` that `set_image_proj_model` (:162-183) builds from
    `ip-adapter.bin`'s `image_proj` group. The second is `draw_kps` (:107-134),
    which RASTERISES the five keypoints into an image for the ControlNet.

    `draw_kps` is why this consumer is not served by a stored embedding alone.
    It takes the keypoints AND the image it draws onto, so its output depends
    on the size `resize_img` (infer.py:12-33) produced -- not on the original
    frame. Keypoints retained against one size do not reconstruct the drawing
    at another, so the size is part of what must be retained.

    `output_dim` is 2048 because upstream passes
    `self.unet.config.cross_attention_dim` and SDXL base is 2048; the
    `strict=True` load of upstream's own checkpoint is what proves it rather
    than the number being asserted here.

    antelopev2, not buffalo_l (infer.py:39) -- a different recognition space
    from the three vendors above it, and largest-by-bbox-area, not first
    (infer.py:67).
    """
    import math

    import cv2
    import PIL.Image
    import torch
    from diffusers.utils import load_image
    from PIL import Image

    from compat.harness import pinned_source

    manifest = provenance.load_manifest()
    row = next(one for one in manifest["consumers"] if one["id"] == "instantid_upstream")
    refs_root = (ROOT.parent / manifest["refs_root"]).resolve()
    clone = provenance.clone_dir(refs_root, row["repo"])
    commit = row["commit"]

    weight = VENDOR / "instantid" / "ip-adapter.bin"
    fixture = _vendor_blob(clone, commit, "examples/yann-lecun_resize.jpg", "instantid", "yann-lecun_resize.jpg")
    held = Acceptance(
        consumer_id="instantid_upstream",
        entrypoint="pipeline_stable_diffusion_xl_instantid.py::_encode_prompt_image_emb + draw_kps",
        repo=row["repo"],
        commit=commit,
        fixture_path=str(fixture) if fixture else "",
        fixture_sha256=_sha256(fixture) if fixture else "",
        ran=False,
        weights=[
            *([{"file": weight.name, "sha256": _sha256(weight)}] if weight.is_file() else []),
            *[
                {"file": one, "sha256": _sha256(CONSISID_MODELS / one)}
                for one in REQUIRED
                if (CONSISID_MODELS / one).is_file()
            ],
        ],
    )
    if fixture is None:
        held.reason = "instantid fixture absent: examples/yann-lecun_resize.jpg"
        return held
    absent = [*missing_weights(), *([] if weight.is_file() else [str(weight)])]
    if absent:
        held.reason = f"UNSUPPORTED on this machine: {', '.join(absent)}"
        return held

    began = time.perf_counter()
    try:
        from insightface.app import FaceAnalysis

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        namespace: dict[str, Any] = {"torch": torch, "nn": torch.nn, "math": math}
        proof = None
        for name in ("FeedForward", "reshape_tensor", "PerceiverAttention", "Resampler"):
            loaded, proof = pinned_source.load_symbol(clone, commit, "ip_adapter/resampler.py", name, dict(namespace))
            namespace[name] = loaded
        model = namespace["Resampler"](
            dim=1280, depth=4, dim_head=64, heads=20, num_queries=16, embedding_dim=512, output_dim=2048, ff_mult=4
        )
        state = torch.load(str(weight), map_location="cpu")
        model.load_state_dict(state.get("image_proj", state))
        model.to(device, dtype).eval()

        resize_img, _ = pinned_source.load_symbol(clone, commit, "infer.py", "resize_img", {"Image": Image, "np": np})
        draw_kps, _ = pinned_source.load_symbol(
            clone,
            commit,
            "pipeline_stable_diffusion_xl_instantid.py",
            "draw_kps",
            {"np": np, "cv2": cv2, "math": math, "PIL": PIL},
        )
        method, _ = pinned_source.load_symbol(
            clone,
            commit,
            "pipeline_stable_diffusion_xl_instantid.py",
            "StableDiffusionXLInstantIDPipeline._encode_prompt_image_emb",
            {"torch": torch},
        )

        app = FaceAnalysis(
            name="antelopev2",
            root=str(CONSISID_MODELS / "face_encoder"),
            providers=["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

        # infer.py:63-69, upstream's own order and upstream's own loader:
        # `diffusers.utils.load_image` (infer.py:6, 63), then resize, then
        # detect on BGR, then the maximum-area face, then draw onto the
        # RESIZED image.
        opened = load_image(str(fixture))
        original = list(opened.size)
        frame = resize_img(opened)
        bgr = np.ascontiguousarray(np.array(frame)[:, :, ::-1])
        best = _largest_face(app.get(bgr), "InstantID's own example")
        seen_pack = observed_pack(app)
        drawn = draw_kps(frame, best["kps"])

        class Stand:
            """Exactly the attributes `_encode_prompt_image_emb` reads."""

            def __init__(self) -> None:
                self.image_proj_model = model
                self.image_proj_model_in_features = 512

        with torch.no_grad():
            # do_classifier_free_guidance=True is what `__call__` passes for
            # any guidance_scale > 1, which is every published example.
            cond = method(Stand(), best["embedding"], device, 1, dtype, True)
    except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError, KeyError) as problem:
        held.reason = f"{type(problem).__name__}: {problem}"
        held.seconds = time.perf_counter() - began
        return held

    out = np.asarray(cond.detach().float().cpu().numpy())
    kps_image = np.asarray(drawn, dtype=np.uint8)
    held.ran = True
    held.seconds = time.perf_counter() - began
    held.boundary = {
        "device": device,
        "dtype": str(dtype),
        "symbol_sha256": proof.blob_sha256 if proof else "",
        "selection": "largest_bbox_area",
        "observed_pack": seen_pack,
        "retained_primitive": "embedding",
        "id_cond": {"shape": list(out.shape), "dtype": str(out.dtype), "sha256": _digest_array(out)},
        # The SECOND artifact. Rasterised from kps, so it is reproducible from
        # retained keypoints only if the size it was drawn at is retained too.
        "face_kps_image": {"shape": list(kps_image.shape), "sha256": _digest_array(kps_image)},
        "resized_to": list(frame.size),
        "original_size": original,
    }
    return held


#: Every vendor whose ID side runs on this box, by the consumer it accepts.
RUNNERS: Final[dict[str, Callable[[], Acceptance]]] = {
    "consisid": run_consisid,
    "instantid_upstream": run_instantid,
    "pulid_upstream": run_pulid,
    "infiniteyou": run_infiniteyou,
    "reactor": run_reactor,
    "ipadapter_upstream": run_ipadapter_faceid,
    "uniportrait": run_uniportrait,
    "photomaker_v2": run_photomaker,
}


def _boundary_digest(boundary: dict[str, Any]) -> str:
    """One vendor's whole boundary as one number.

    The WHOLE boundary, not a chosen tensor: the seven boundaries share no
    shape and no key, so quoting `id_cond` would measure nothing for ReActor,
    which returns a persisted file, or for PhotoMaker, whose detection sweep
    is half of what it produces. Everything recorded is deterministic by
    intent, so anything that moves between runs is the finding.
    """
    return hashlib.sha256(json.dumps(boundary, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def determinism(first: list[Acceptance], times: int = 2) -> dict[str, Any]:
    """Whether each vendor's own boundary repeats on identical input.

    Not a formality. Two ConsisID runs in this suite produced id_cond digests
    de0a78de713b57c6 and 04d3b13c1a89873d from the same fixture, the same
    weights and the same code. onnxruntime reports
    `cudnn_conv_algo_search: EXHAUSTIVE` and the run is fp16, so the
    convolution algorithm -- and with it the reduction order -- is chosen per
    process.

    A boundary that does not repeat cannot serve as a baseline: every
    downstream "our adapter matches" would be comparing against a number that
    moves. Measured for EVERY accepted vendor rather than the one it was
    first seen in -- most of the eight run fp16 on the same EXHAUSTIVE
    onnxruntime, so nothing about the mechanism is specific to ConsisID.

    `first` supplies each vendor's opening run so the survey is not paid for
    twice. Every REPEAT is a fresh interpreter, because the mechanism above is
    per-process: re-calling `runner()` in this one holds the algorithm choice
    fixed and the comparison becomes two reads of the same decision. Measured
    that way it reported 8 vendors, 2 runs, 1 digest each, stable -- and could
    not have reported anything else.

    A vendor that never ran is `not_run`, not `unstable`. The two were one
    field, so an absent base checkpoint read as a moving boundary.
    """
    opened = {one.consumer_id: one for one in first if one.ran}
    vendors: dict[str, Any] = {}
    for name in RUNNERS:
        digests: list[str] = []
        reason = ""
        held = opened.get(name)
        if held is None:
            vendors[name] = {
                "digests": [],
                "stable": None,
                "not_run": True,
                "reason": "did not run here; nothing to repeat",
            }
            continue
        digests.append(_boundary_digest(held.boundary))
        for _ in range(times - 1):
            again = _digest_in_subprocess(name)
            if again is None:
                reason = "the repeat could not be run in a fresh interpreter"
                break
            digests.append(again)
        stable = len(digests) == times and len(set(digests)) == 1
        vendors[name] = {
            "digests": digests,
            "stable": stable,
            "not_run": False,
            "reason": reason or ("" if stable else "identical input produced different bytes"),
        }
    unstable = sorted(name for name, one in vendors.items() if one["stable"] is False)
    not_run = sorted(name for name, one in vendors.items() if one.get("not_run"))
    return {
        "runs": times,
        "vendors": vendors,
        "stable": not unstable,
        "unstable": unstable,
        "not_run": not_run,
    }


def _digest_in_subprocess(name: str) -> str | None:
    """One vendor's boundary digest, from an interpreter of its own.

    The whole point of the repeat. `python -m compat.vendor.acceptance --one
    <name>` runs exactly one runner and prints its boundary digest, so the
    convolution algorithm, the memory arena and every other per-process choice
    are made again rather than inherited.
    """
    _, out, _ = proc.text(
        [sys.executable, "-m", "compat.vendor.acceptance", "--one", name],
        timeout=1800,
        cwd=ROOT.parent,
    )
    for line in reversed(out.splitlines()):
        held = line.strip()
        if len(held) == 64 and all(one in "0123456789abcdef" for one in held):
            return held
    return None


def declared_against_observed(rows: list[Acceptance]) -> list[dict[str, Any]]:
    """The manifest's declared pack against the pack the vendor actually loaded.

    This is the gate the suite was missing, and the reason it was missing is
    worth stating: every other check here is satisfied by a WELL-FORMED claim
    rather than a TRUE one.

      - `citations` proves a path:line resolves and names a real symbol. A
        wrong `pack` cited to a line that constructs a FaceAnalysis passes it.
      - the `stored_glintr100_substituted` ablation derives `expect_breaks`
        FROM `pack`. With the wrong pack it expects no break and observes
        none; with the right pack it expects one and observes one. Green
        either way, so it cannot distinguish them.

    Four `pack` values were wrong in this manifest and both checks passed on
    all four. What separates a claim from a fact here is that
    `observed_pack()` reads the model files off the running analyser, so this
    comparison is manifest-versus-run rather than typing-versus-typing.

    A consumer whose acceptance does not build an analyser is reported as
    UNOBSERVED rather than counted as agreement.
    """
    manifest = provenance.load_manifest()
    declared = {
        one["id"]: (one.get("vendor_setup") or {}).get("pack")
        for one in manifest.get("consumers", [])
        if one.get("vendor_setup")
    }

    out: list[dict[str, Any]] = []
    for row in rows:
        if not row.ran:
            continue
        seen = row.boundary.get("observed_pack")
        want = declared.get(row.consumer_id)
        if seen is None or want is None:
            out.append(
                {
                    "consumer_id": row.consumer_id,
                    "declared": want,
                    "observed": None,
                    "agrees": None,
                    "detail": "the acceptance builds no analyser to observe" if seen is None else "no pack declared",
                }
            )
            continue
        got = seen["pack"]
        out.append(
            {
                "consumer_id": row.consumer_id,
                "declared": want,
                "observed": got,
                "modules": seen["modules"],
                "agrees": got == want,
                "detail": "" if got == want else f"the manifest says {want!r}; the run loaded {got!r}",
            }
        )

    # The same treatment for the two other fields a run can settle. `pack`
    # proved the pattern; these close the rest of the gap between what the
    # manifest declares and what the vendor's code does.
    for row in rows:
        if not row.ran:
            continue
        setup = next(
            ((one.get("vendor_setup") or {}) for one in manifest.get("consumers", []) if one["id"] == row.consumer_id),
            {},
        )
        seen_embedding = row.boundary.get("observed_embedding")
        if seen_embedding and setup.get("embedding") in {"raw", "normed"}:
            out.append(
                {
                    "consumer_id": row.consumer_id,
                    "field": "embedding",
                    "declared": setup["embedding"],
                    "observed": seen_embedding["kind"],
                    "agrees": seen_embedding["kind"] == setup["embedding"],
                    "detail": f"L2 norm {seen_embedding['l2_norm']:.6f} over {seen_embedding['dims']} dims",
                }
            )
        seen_crop = row.boundary.get("observed_crop_size")
        if seen_crop is not None and setup.get("crop_sizes"):
            out.append(
                {
                    "consumer_id": row.consumer_id,
                    "field": "crop_sizes",
                    "declared": setup["crop_sizes"],
                    "observed": seen_crop,
                    "agrees": seen_crop in setup["crop_sizes"],
                    "detail": f"the crop the vendor's code produced is {seen_crop}x{seen_crop}",
                }
            )
    return out


def survey() -> dict[str, Any]:
    # From RUNNERS, so the set that is surveyed and the set whose determinism
    # is measured cannot drift apart.
    rows: list[Acceptance] = [runner() for runner in RUNNERS.values()]
    repeats = determinism(rows)
    against_manifest = declared_against_observed(rows)
    upstream = against_upstream(rows)
    manifest = provenance.load_manifest()
    declared = {one["id"] for one in manifest.get("consumers", [])}
    ran = {one.consumer_id for one in rows if one.ran}
    attempted = {one.consumer_id for one in rows}
    # ACCEPTED is now "reproduced the boundary upstream declares", not "did not
    # raise". A vendor that ran without an upstream statement to check against
    # is neither accepted nor failed: it is VENDOR_BASELINE_UNAVAILABLE, which
    # is a fact about the upstream and the verdict `contracts/case.py` reserves
    # for exactly this.
    agreed = {one["consumer_id"] for one in upstream if one["agrees"] is True}
    disagreed = {one["consumer_id"] for one in upstream if one["agrees"] is False}
    unstated = {one["consumer_id"] for one in upstream if not one["stated"]}
    return {
        "runtime": provenance.runtime_identity(),
        "acceptance": [asdict(one) for one in rows],
        "determinism": repeats,
        "declared_against_observed": against_manifest,
        "against_upstream": upstream,
        "population": {
            "declared": sorted(declared),
            "ran_without_raising": sorted(ran),
            "vendor_accepted": sorted(agreed),
            "reproduced_wrong_boundary": sorted(disagreed),
            "vendor_baseline_unavailable": sorted(unstated),
            "attempted_and_failed": sorted(attempted - ran),
            # Never attempted: their base checkpoints are not on this machine.
            # UNSUPPORTED, a fact about the box, not about the upstream.
            "not_attempted": sorted(declared - attempted),
        },
    }


def against_upstream(rows: list[Acceptance]) -> list[dict[str, Any]]:
    """Each run compared to the shape UPSTREAM declares for its boundary.

    `ran` means the call did not raise. That is a fact about this machine and
    not about the vendor reproducing anything, and it was the whole of LAYER
    ONE's verdict: eight vendors reported accepted because eight calls
    returned. A call that returns the wrong tensor returns just as quietly.

    `manifest.toml` carries `[consumers.acceptance_expected]` for every vendor
    whose own source states its boundary, each with the file and line that
    states it. Two forms, because upstream states it two ways:

        shape   the whole tuple, when upstream writes it out -- ConsisID's
                own comment says `torch.Size([1, 1280])`
        tokens  the ONE axis upstream fixes, when the width comes from
                whichever base model is loaded and is therefore a fact about
                the checkpoint rather than about the vendor

    A vendor with no such statement is recorded `stated: false` and is NOT
    counted as reproducing its own boundary. That is the honest reading of
    "upstream supplies no runnable first-party expectation", and it keeps the
    absence visible instead of letting `ran` stand in for it.
    """
    manifest = provenance.load_manifest()
    wanted = {
        one["id"]: one["acceptance_expected"] for one in manifest.get("consumers", []) if one.get("acceptance_expected")
    }
    out: list[dict[str, Any]] = []
    for row in rows:
        if not row.ran:
            continue
        expected = wanted.get(row.consumer_id)
        if expected is None:
            out.append(
                {
                    "consumer_id": row.consumer_id,
                    "stated": False,
                    "agrees": None,
                    "detail": "upstream states no expected boundary at the pinned commit",
                }
            )
            continue
        # `row.boundary` is the dict the runners in this module write, keyed
        # "id_cond". It is a different namespace from `[[consumers]].boundary`
        # in compat/manifest.toml, which holds upstream's own names.
        held = (row.boundary or {}).get(expected["boundary_key"])
        shape = list(held.get("shape", [])) if isinstance(held, dict) else None
        if not shape:
            out.append(
                {
                    "consumer_id": row.consumer_id,
                    "stated": True,
                    "agrees": False,
                    "detail": f"the run recorded no {expected['boundary_key']!r} shape to compare",
                }
            )
            continue
        if "shape" in expected:
            agrees = shape == list(expected["shape"])
            detail = f"{expected['boundary_key']} {shape} against upstream's {list(expected['shape'])}"
        else:
            # The token axis is the second-to-last: [batch, tokens, width].
            tokens = shape[-2] if len(shape) >= 2 else None
            agrees = tokens == int(expected["tokens"])
            detail = (
                f"{expected['boundary_key']} {shape} carries {tokens} tokens, upstream declares {expected['tokens']}"
            )
        out.append(
            {
                "consumer_id": row.consumer_id,
                "stated": True,
                "agrees": agrees,
                "detail": detail,
                "cited": list(expected.get("cited", [])),
            }
        )
    return sorted(out, key=lambda one: one["consumer_id"])


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if "--one" in args:
        # One vendor, one interpreter, one line of output: the digest.
        # `determinism` shells out to this so its repeat is a genuinely fresh
        # process rather than a second call inside the first.
        name = args[args.index("--one") + 1]
        runner = RUNNERS.get(name)
        if runner is None:
            print(f"no such vendor runner: {name!r}", file=sys.stderr)
            return 2
        held = runner()
        if not held.ran:
            print(f"{name} did not run: {held.reason}", file=sys.stderr)
            return 1
        print(_boundary_digest(held.boundary))
        return 0

    out = survey()
    for row in out["acceptance"]:
        mark = "ok " if row["ran"] else "!! "
        print(f"{mark}{row['consumer_id']:<18} {row['entrypoint']}")
        print(
            f"    fixture {row['fixture_sha256'][:16]} ({row['fixture_origin']})"
            f"  weights {len(row['weights'])}  {row['seconds']:.1f}s"
        )
        if row["ran"]:
            held = row["boundary"]
            # Printed per key present: the boundaries are NOT the same shape
            # across vendors -- ConsisID returns id_cond [1,1280] plus
            # face_kps and five ViT hidden states, PuLID returns
            # cat(uncond, cond) at [2,10,2048] and no keypoints at all,
            # PhotoMaker returns a stack of raw embeddings, and ReActor
            # returns no tensor at all but a persisted file.
            for name in ("id_cond", "id_embeds"):
                if name in held:
                    print(f"    {name} {held[name]['shape']} {held[name]['sha256'][:16]} on {held['device']}")
            if "face_kps" in held:
                print(f"    face_kps {held['face_kps']['shape']}")
            if "id_vit_hidden" in held:
                print(f"    id_vit_hidden {held['id_vit_hidden']}")
            if "persists" in held:
                print(
                    f"    persists {len(held['persists'])} keys, {held['bytes']} B, "
                    f"all survive={held['every_key_survives']}"
                )
                print(f"    NOT persisted: {held['producer_keys_not_persisted']}")
            if "detection_sweep" in held:
                for one in held["detection_sweep"]:
                    print(
                        f"    sweep {one['fixture']:<14} {one['input_size_before']} -> "
                        f"{one['input_size_after']}  faces={one['faces']}"
                    )
        else:
            print(f"    {row['reason'][:200]}")

    repeats = out["determinism"]
    mark = "ok " if repeats["stable"] else "!! "
    print(
        f"\n{mark}determinism: {repeats['runs']} runs each in separate interpreters, "
        f"unstable={repeats['unstable']}, not run here={repeats['not_run']}"
    )
    for name, one in sorted(repeats["vendors"].items()):
        seen = " ".join(digest[:16] for digest in one["digests"])
        flag = "-- " if one["stable"] is None else ("ok " if one["stable"] else "!! ")
        print(f"    {flag}{name:<18} {seen}  {one['reason']}")

    diff = out["declared_against_observed"]
    disagreed = [one for one in diff if one["agrees"] is False]
    unobserved = [one for one in diff if one["agrees"] is None]
    print(f"\n{'!! ' if disagreed else 'ok '}manifest pack vs the pack the run actually loaded:")
    for one in diff:
        mark = "ok " if one["agrees"] else ("-- " if one["agrees"] is None else "!! ")
        print(
            f"    {mark}{one['consumer_id']:<20} declared={one['declared']!s:<12} "
            f"observed={one['observed']!s:<12} {one['detail']}"
        )
    print(f"    {len(disagreed)} disagree, {len(unobserved)} unobserved")

    print("\nAGAINST UPSTREAM'S OWN DECLARED BOUNDARY")
    for one in out["against_upstream"]:
        mark = "ok " if one["agrees"] else ("-- " if one["agrees"] is None else "!! ")
        print(f"    {mark}{one['consumer_id']:<20} {one['detail']}")

    pop = out["population"]
    print(f"\nran without raising: {len(pop['ran_without_raising'])}  {pop['ran_without_raising']}")
    print(f"VENDOR ACCEPTED    : {len(pop['vendor_accepted'])}  {pop['vendor_accepted']}")
    print(f"wrong boundary     : {len(pop['reproduced_wrong_boundary'])}  {pop['reproduced_wrong_boundary']}")
    print(
        f"no upstream expectation: {len(pop['vendor_baseline_unavailable'])}  "
        f"{pop['vendor_baseline_unavailable']} (VENDOR_BASELINE_UNAVAILABLE)"
    )
    print(f"attempted, failed  : {len(pop['attempted_and_failed'])}  {pop['attempted_and_failed']}")
    print(f"not attempted      : {len(pop['not_attempted'])} (base checkpoints absent on this machine)")

    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    target = generated / "vendor_acceptance.json"
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True, default=str))
        handle.write("\n")
    print(f"wrote {target}")

    # Three ways this lane can be wrong, and it gates on all three. A vendor
    # that raised and a boundary that will not repeat are both fatal to
    # "LAYER ONE: the reference itself is known to run" -- reporting them and
    # exiting 0 is the shape of a suite reporting success it did not earn.
    unstable = out["determinism"]["unstable"]
    failed = pop["attempted_and_failed"]
    blocking = {
        "manifest disagrees with the run": [one["consumer_id"] for one in disagreed],
        "boundary did not repeat": unstable,
        "attempted and raised": failed,
        # The fourth, and the one this lane exists for. A run whose boundary
        # is not the shape upstream declares has not reproduced upstream's
        # example, whatever it did without raising.
        "boundary is not upstream's": pop["reproduced_wrong_boundary"],
    }
    bad = {why: names for why, names in blocking.items() if names}
    if bad:
        print("\nacceptance NOT clean:")
        for why, names in bad.items():
            print(f"    {why}: {', '.join(names)}")
        return 1
    print("\nacceptance clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
