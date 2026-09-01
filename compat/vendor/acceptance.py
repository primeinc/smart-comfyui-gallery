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
from compat.harness import failfast, provenance
from compat.producers import insightface_pass as producer

ROOT: Final[Path] = Path(__file__).resolve().parent.parent


CONSISID_MODELS: Final[Path] = ROOT.parent.parent / "sg-vendor-fixtures" / "consisid"


PULID_WEIGHT: Final[Path] = ROOT.parent.parent / "sg-vendor-fixtures" / "pulid" / "pulid_v1.bin"


VENDOR: Final[Path] = ROOT.parent.parent / "sg-vendor-fixtures"


BUFFALO_ROOT: Final[Path] = producer.MODELS_ROOT
BUFFALO: Final[tuple[str, ...]] = (
    "models/buffalo_l/det_10g.onnx",
    "models/buffalo_l/w600k_r50.onnx",
    "models/buffalo_l/1k3d68.onnx",
    "models/buffalo_l/2d106det.onnx",
    "models/buffalo_l/genderage.onnx",
)


REQUIRED: Final[tuple[str, ...]] = (
    "face_encoder/EVA02_CLIP_L_336_psz14_s6B.pt",
    "face_encoder/detection_Resnet50_Final.pth",
    "face_encoder/parsing_bisenet.pth",
    "face_encoder/models/antelopev2/glintr100.onnx",
    "face_encoder/models/antelopev2/scrfd_10g_bnkps.onnx",
)


@dataclass
class Acceptance:
    consumer_id: str
    entrypoint: str
    repo: str
    commit: str
    fixture_path: str
    fixture_sha256: str
    ran: bool

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

        dtype = torch.float32

        helper, ante, clip_vision, app, mean, std = prepare_face_models(str(CONSISID_MODELS), device, dtype)

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

        encoder_cls, _ = pinned_source.load_symbol(
            clone, row["commit"], "pulid/encoders.py", "IDEncoder", {"torch": torch, "nn": torch.nn}
        )
        adapter = encoder_cls().to(device)

        held_state = torch.load(str(PULID_WEIGHT), map_location="cpu")
        prefix = "id_adapter."
        adapter.load_state_dict(
            {k[len(prefix) :]: v for k, v in held_state.items() if k.startswith(prefix)}, strict=True
        )
        adapter.eval()

        class Stand:
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

            to_gray = gray_method

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
    if not found:
        raise ValueError(f"no face detected in {where}")
    return max(found, key=lambda one: (one.bbox[2] - one.bbox[0]) * (one.bbox[3] - one.bbox[1]))


def _first_face(found: list[Any], where: str) -> Any:
    if not found:
        raise ValueError(f"no face detected in {where}")
    return found[0]


def _placed(model: Any, device: str, dtype: Any) -> Any:
    return model.to(device, dtype).eval()


def _nonempty(values: list[Any], where: str) -> list[Any]:
    if not values:
        raise ValueError(f"no face detected in {where}")
    return values


def _written(target: Path) -> Path:
    if not target.is_file():
        raise ValueError(f"the vendor's writer produced no file at {target}")
    return target


def _decode_rgb(where: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imdecode(np.frombuffer(where.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cv2 could not decode {where}")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def _blob_bytes(clone: Path, commit: str, path: str) -> bytes:
    code, out, _ = proc.run(
        ["git", "-C", str(clone), "cat-file", "blob", f"{commit}:{path}"], timeout=proc.LOCAL_SECONDS
    )
    if code != 0:
        raise LookupError(f"{path} is not at {commit[:12]} in {clone.name}")
    return out


def _tree_against_pin(clone: Path, commit: str, paths: tuple[str, ...]) -> dict[str, str]:
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
    where = ROOT.parent.parent / "sg-vendor-fixtures" / into / name
    where.parent.mkdir(parents=True, exist_ok=True)
    code, blob, _ = proc.run(
        ["git", "-C", str(clone), "cat-file", "blob", f"{commit}:{path}"], timeout=proc.LOCAL_SECONDS
    )
    if code != 0:
        return None

    if where.is_file() and where.read_bytes() == blob:
        return where
    where.write_bytes(blob)
    return where


def run_infiniteyou() -> Acceptance:
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
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(flat))
    return {
        "l2_norm": norm,
        "kind": "normed" if abs(norm - 1.0) < 1e-3 else "raw",
        "dims": flat.size,
    }


def observed_crop_size(crop: np.ndarray) -> int:
    return int(np.asarray(crop).shape[0])


def observed_pack(app: Any) -> dict[str, Any]:
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
        "pack": next(iter(packs)) if len(packs) == 1 else f"MIXED:{sorted(packs)}",
        "modules": sorted(files),
        "files": files,
    }


def _buffalo(device: str) -> Any:
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        root=str(BUFFALO_ROOT),
        providers=["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
    return app


def reactor_core(clone: Path, commit: str) -> tuple[Any, dict[str, str]]:
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

        analyzer, core_blobs = reactor_core(clone, commit)
        app = analyzer.ReActorFaceAnalysis(
            name="buffalo_l",
            root=str(BUFFALO_ROOT),
            providers=["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

        bgr = _decode_rgb(Path(sample[0]))[:, :, ::-1].copy()

        face = _first_face(app.get(bgr), sample[0])
        seen_pack = observed_pack(app)
        observed = Observation.of(face)

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
        "producer_keys_not_persisted": sorted(set(observed) - set(persists)),
        "vs_insightface": against_insightface,
    }
    return held


def run_ipadapter_faceid() -> Acceptance:
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

        model = proj_cls(cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4)
        model.load_state_dict(torch.load(str(weight), map_location="cpu")["image_proj"])
        model.to(device, dtype).eval()

        method, _ = pinned_source.load_symbol(
            clone, commit, "ip_adapter/ip_adapter_faceid.py", "IPAdapterFaceID.get_image_embeds", {"torch": torch}
        )

        class Stand:
            def __init__(self) -> None:
                self.device = device
                self.torch_dtype = dtype
                self.image_proj_model = model

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

        _tree_against_pin(clone, commit, curricular)
        with _importable(clone):
            backbones = importlib.import_module("uniportrait.curricular_face.backbone")
        facerecog = backbones.get_model("IR_101")([112, 112])
        facerecog.load_state_dict(torch.load(str(backbone), map_location="cpu"))
        facerecog = facerecog.to(device, dtype).eval()

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

        resampler.load_state_dict(torch.load(str(faceid), map_location="cpu")["faceid_proj"], strict=True)

        method, _ = pinned_source.load_symbol(
            clone,
            commit,
            "uniportrait/uniportrait_pipeline.py",
            "UniPortraitPipeline.get_single_faceid_embeds",
            {"torch": torch, "F": torch.nn.functional},
        )

        class Stand:
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
        "align_crop_size": 224,
        "face_structure_scale": 0.0,
        "id_cond": {"shape": list(out.shape), "dtype": str(out.dtype), "sha256": _digest_array(out)},
        "uncond": {"shape": list(null.shape), "sha256": _digest_array(null)},
    }
    return held


def run_photomaker() -> Acceptance:
    import torch

    from compat.harness import pinned_source

    manifest = provenance.load_manifest()
    row = next(one for one in manifest["consumers"] if one["id"] == "photomaker_v2")
    refs_root = (ROOT.parent / manifest["refs_root"]).resolve()
    clone = provenance.clone_dir(refs_root, row["repo"])
    commit = row["commit"]

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

        detector = analysis_cls(
            name="buffalo_l",
            root=str(BUFFALO_ROOT),
            providers=["CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        detector.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

        sweep: list[dict[str, Any]] = []
        vectors: list[Any] = []

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
        "det_size_left_behind": list(sweep[-1]["input_size_after"]),
        "id_embeds": {"shape": list(out.shape), "dtype": str(out.dtype), "sha256": _digest_array(out)},
        "encoder_unsupported": (
            "PhotoMakerIDEncoder_CLIPInsightfaceExtendtoken.forward (model_v2.py:142) "
            "requires SDXL prompt_embeds; no SDXL checkpoint on this machine"
        ),
    }
    return held


def run_instantid() -> Acceptance:
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

        opened = load_image(str(fixture))
        original = list(opened.size)
        frame = resize_img(opened)
        bgr = np.ascontiguousarray(np.array(frame)[:, :, ::-1])
        best = _largest_face(app.get(bgr), "InstantID's own example")
        seen_pack = observed_pack(app)
        drawn = draw_kps(frame, best["kps"])

        class Stand:
            def __init__(self) -> None:
                self.image_proj_model = model
                self.image_proj_model_in_features = 512

        with torch.no_grad():
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
        "face_kps_image": {"shape": list(kps_image.shape), "sha256": _digest_array(kps_image)},
        "resized_to": list(frame.size),
        "original_size": original,
    }
    return held


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
    return hashlib.sha256(json.dumps(boundary, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def determinism(first: list[Acceptance], times: int = 2) -> dict[str, Any]:
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

    rows: list[Acceptance] = [runner() for runner in RUNNERS.values()]
    repeats = determinism(rows)
    against_manifest = declared_against_observed(rows)
    upstream = against_upstream(rows)
    manifest = provenance.load_manifest()
    declared = {one["id"] for one in manifest.get("consumers", [])}
    ran = {one.consumer_id for one in rows if one.ran}
    attempted = {one.consumer_id for one in rows}

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
            "not_attempted": sorted(declared - attempted),
        },
    }


def against_upstream(rows: list[Acceptance]) -> list[dict[str, Any]]:
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
    failfast.arm()
    args = list(argv if argv is not None else sys.argv[1:])
    if "--one" in args:
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

    unstable = out["determinism"]["unstable"]
    failed = pop["attempted_and_failed"]
    blocking = {
        "manifest disagrees with the run": [one["consumer_id"] for one in disagreed],
        "boundary did not repeat": unstable,
        "attempted and raised": failed,
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
