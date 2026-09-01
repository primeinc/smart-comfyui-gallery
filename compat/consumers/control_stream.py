from __future__ import annotations

import hashlib
import importlib
import io
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from compat.contracts.case import (
    Ablation,
    Artifact,
    Case,
    Fixture,
    Measurement,
    RetainedState,
    Tier,
    UInt8Array,
)
from compat.harness import provenance

CONSUMER_ID: Final[str] = "id_v2v"


SAMPLES: Final[str] = "test_samples"


CHECKPOINT_ROOT: Final[str] = "IDV2V_CHECKPOINTS"


STREAMS: Final[dict[str, tuple[str, str, tuple[str, ...]]]] = {
    "orig_pixel_mp4": ("sam3", "sam3", ("idv2v", "idv2v_with_normal_depth")),
    "david_normal_mp4": ("david", "david/multi-task-model-vitl16_384.onnx", ("idv2v_with_normal_depth",)),
    "depth_mp4": ("depth_anything_v2", "depthv2/depth_anything_v2_vitl.pth", ("idv2v_with_normal_depth",)),
}


CLIPS: Final[int] = 4


SAM_PROMPT: Final[str] = "person"
GRAY_VALUE: Final[int] = 127


@dataclass(frozen=True)
class Clip:
    label: str
    path: Path
    fixture: Fixture


def _digest(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            hasher.update(chunk)
            total += len(chunk)
    return hasher.hexdigest(), total


def repo_root() -> Path:
    manifest = provenance.load_manifest()
    for row in manifest.get("consumers", []):
        if row["id"] == CONSUMER_ID:
            return provenance.clone_dir(
                (Path(__file__).resolve().parent.parent.parent / manifest["refs_root"]).resolve(),
                row["repo"],
            )
    raise KeyError(f"{CONSUMER_ID} is not in the manifest")


def clips(limit: int = CLIPS) -> list[Clip]:
    root = repo_root() / SAMPLES
    if not root.is_dir():
        return []
    found: list[Clip] = []
    for source in sorted(root.rglob("source.mp4")):
        sha, size = _digest(source)

        label = source.parent.relative_to(root).as_posix().replace("/", "_")
        found.append(
            Clip(
                label=label,
                path=source,
                fixture=Fixture(
                    name=f"idv2v_{label}",
                    path=str(source),
                    sha256=sha,
                    kind="vendor_sample_video",
                    note=f"{size:,} B, real footage shipped at the pinned commit, not vendored",
                ),
            )
        )
    return sorted(found, key=lambda one: one.fixture.sha256)[:limit]


def checkpoint_of(stream: str) -> Path | None:
    from compat.harness import provision

    _, relative, _ = STREAMS[stream]
    candidate = provision.root_of(provenance.load_manifest()) / relative
    return candidate if candidate.exists() else None


def decode(blob: bytes) -> UInt8Array:
    import av

    frames: list[UInt8Array] = []
    try:
        with av.open(io.BytesIO(blob), mode="r") as container:
            frames.extend(
                np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8) for frame in container.decode(video=0)
            )
    except av.FFmpegError as problem:
        raise ValueError(f"these bytes are not decodable video: {type(problem).__name__}: {problem}") from problem
    if not frames:
        raise ValueError("no frames decoded from the retained bytes")
    return np.asarray(np.stack(frames), dtype=np.uint8)


STREAM_FPS: Final[int] = 16
STREAM_CRF: Final[str] = "28"


def encode(stack: UInt8Array) -> bytes:
    import av

    height, width = stack.shape[1:3]
    held = io.BytesIO()
    writing = "w"
    container = av.open(held, writing, format="mp4")
    stream = container.add_stream("libx264", rate=STREAM_FPS)
    stream.width, stream.height = int(width), int(height)
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": STREAM_CRF}
    for one in stack:
        container.mux(stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(one), format="rgb24")))
    container.mux(stream.encode())
    container.close()
    return held.getvalue()


def _artifact(name: str, values: np.ndarray) -> Artifact:
    from compat.assertions.arrays import digest

    return Artifact(name=name, dtype=str(values.dtype), shape=values.shape, sha256=digest(values), values=values)


DERIVE_FRAMES: Final[int] = 8


_LOADED: dict[str, Any] = {}


def _david_normals(checkpoint: Path, frames: UInt8Array) -> UInt8Array:

    import torch

    from compat.vendor.acceptance import _importable

    del torch
    clone = repo_root()
    with _importable(clone / "src"):
        if "david" not in _LOADED:
            estimator_module = importlib.import_module("idv2v.preprocess.david_runtime.multi_task_estimator")
            _LOADED["david"] = estimator_module.MultiTaskEstimator(str(checkpoint))
        estimator = _LOADED["david"]
        out = [estimator.estimate_normal(one) for one in frames]

    stacked = np.asarray(out, dtype=np.float32)
    return np.asarray(((stacked + 1.0) * 127.5).clip(0, 255), dtype=np.uint8)


def _depth_v2(checkpoint: Path, frames: UInt8Array) -> UInt8Array:
    import torch

    from compat.vendor.acceptance import _importable

    with _importable(repo_root() / "src"):
        if "depth" not in _LOADED:
            dpt = importlib.import_module("idv2v.preprocess.vace_annotators.annotators.depth_anything_v2.dpt")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            built = dpt.DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]).to(device)
            built.load_state_dict(torch.load(str(checkpoint), map_location=device))
            built.eval()
            _LOADED["depth"] = built
        model = _LOADED["depth"]

        out: list[UInt8Array] = []
        with torch.inference_mode():
            for one in frames:
                depth = model.infer_image(np.asarray(one))
                shifted = depth - float(np.min(depth))
                span = float(np.max(shifted)) or 1.0
                grey = np.asarray((shifted / span * 255.0).clip(0, 255), dtype=np.uint8)
                out.append(np.repeat(grey[..., np.newaxis], 3, axis=2))
    return np.asarray(out, dtype=np.uint8)


def _orig_pixel(checkpoint: Path, frames: UInt8Array) -> UInt8Array:
    import torch
    from PIL import Image

    from compat.vendor.acceptance import _importable

    with _importable(repo_root() / "src"):
        sam3 = importlib.import_module("idv2v.preprocess.sam3")
        if "sam3" not in _LOADED:
            _LOADED["sam3"] = sam3.init_sam3_video(model_path=str(checkpoint), dtype=torch.bfloat16)
        model, processor, device = _LOADED["sam3"]
        images = [Image.fromarray(np.asarray(one, dtype=np.uint8)).convert("RGB") for one in frames]
        per_frame = sam3.pack_per_frame_instances(
            sam3.run_sam3_video(model, processor, device, images, prompt=SAM_PROMPT, dtype=torch.bfloat16)
        )

    height, width = frames.shape[1], frames.shape[2]
    out: list[UInt8Array] = []
    for index, frame in enumerate(frames):
        composited = np.full((height, width, 3), GRAY_VALUE, dtype=np.uint8)
        union = np.zeros((height, width), dtype=bool)
        for instance in per_frame.get(index, []):
            mask = instance["mask"]
            union |= np.asarray(mask if mask.dtype == np.bool_ else mask > 0.5, dtype=bool)
        composited[union] = np.asarray(frame, dtype=np.uint8)[union]
        out.append(composited)
    return np.asarray(out, dtype=np.uint8)


DERIVATIONS: Final[dict[str, Callable[[Path, UInt8Array], UInt8Array]]] = {
    "david_normal_mp4": _david_normals,
    "depth_mp4": _depth_v2,
    "orig_pixel_mp4": _orig_pixel,
}


def _derive(stream: str, clip: Clip) -> UInt8Array:
    producer, relative, _ = STREAMS[stream]
    checkpoint = checkpoint_of(stream)
    if checkpoint is None:
        raise FileNotFoundError(
            f"{stream} for {clip.label}: the {producer} weight is not on this machine "
            f"(${CHECKPOINT_ROOT}/{relative}). Download it and the case runs."
        )

    frames = decode(clip.path.read_bytes())[:DERIVE_FRAMES]
    derive = DERIVATIONS.get(stream)
    if derive is None:
        raise ValueError(
            f"{stream} for {clip.label}: the {producer} weight is at {checkpoint} and no pass reads it. "
            f"DERIVATIONS covers {sorted(DERIVATIONS)}."
        )
    return derive(checkpoint, frames)


class IdV2VControlStreamRunner:
    consumer_id = CONSUMER_ID

    def __init__(self) -> None:
        self._clips = {one.label: one for one in clips()}

    def _parts(self, case: Case) -> tuple[str, Clip]:
        kind, _, label = case.boundary.partition("|")
        return kind, self._clips[label]

    def cases(self) -> tuple[Case, ...]:
        out: list[Case] = []
        for clip in self._clips.values():
            out.append(
                Case(
                    name=f"id_v2v_source_frames_{clip.label}",
                    consumer_id=CONSUMER_ID,
                    tier=Tier.CONSUMER,
                    fixture=clip.fixture,
                    boundary=f"decoded_source_frames|{clip.label}",
                    exact_bytes=True,
                    rtol=0.0,
                    atol=0.0,
                    retained=("source_video_bytes",),
                    ablations=(Ablation(primitive="source_video_bytes", expect_breaks=True),),
                    measurements=("frames_and_bytes",),
                    note=(
                        "the frames every stream producer is handed; proves the media "
                        "round-trips, NOT that a stream is free to re-derive"
                    ),
                )
            )
            for stream, (producer, _, variants) in STREAMS.items():
                out.append(
                    Case(
                        name=f"id_v2v_{stream}_{clip.label}",
                        consumer_id=CONSUMER_ID,
                        tier=Tier.CONSUMER,
                        fixture=clip.fixture,
                        boundary=f"{stream}|{clip.label}",
                        exact_bytes=True,
                        rtol=0.0,
                        atol=0.0,
                        retained=(stream,),
                        ablations=(
                            Ablation(primitive=stream, expect_breaks=True),
                            Ablation(
                                primitive=stream,
                                swap="video_round_trip",
                                expect_breaks=True,
                                kind="substitution",
                            ),
                        ),
                        measurements=("rederivation_cost",),
                        note=f"{producer} writes this stream; consumed by {', '.join(variants)}",
                    )
                )
        return tuple(out)

    def retained_for(self, case: Case) -> RetainedState:
        kind, clip = self._parts(case)
        if kind == "decoded_source_frames":
            return RetainedState(source_video_bytes=np.frombuffer(clip.path.read_bytes(), dtype=np.uint8))
        return RetainedState(**{kind: _derive(kind, clip)})

    def baseline(self, case: Case) -> Artifact:
        kind, clip = self._parts(case)
        if kind == "decoded_source_frames":
            return _artifact(case.boundary, decode(clip.path.read_bytes()))
        return _artifact(case.boundary, _derive(kind, clip))

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        kind, _ = self._parts(case)
        if kind == "decoded_source_frames":
            return _artifact(case.boundary, decode(retained.pixels("source_video_bytes").tobytes()))
        return _artifact(case.boundary, retained.pixels(kind))

    def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState:
        del case
        if ablation.swap == "video_round_trip":
            return retained.replacing(ablation.primitive, decode(encode(retained.pixels(ablation.primitive))))
        return retained.without(ablation.primitive)

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        kind, clip = self._parts(case)
        if name == "frames_and_bytes":
            blob = retained.pixels("source_video_bytes")
            frames = decode(blob.tobytes())
            return Measurement(
                name=name,
                unit="bytes",
                value=float(blob.nbytes),
                basis="the encoded source against the frame stack every stream producer is handed",
                detail=(
                    f"{clip.label}: source {blob.nbytes:,} B decodes to {frames.shape} = "
                    f"{frames.nbytes:,} B ({frames.nbytes / blob.nbytes:.1f}x)"
                ),
            )
        if name != "rederivation_cost":
            raise KeyError(f"{CONSUMER_ID} has no measurement called {name!r}")
        producer, relative, variants = STREAMS[kind]
        checkpoint = checkpoint_of(kind)
        return Measurement(
            name=name,
            unit="frames",
            value=None,
            basis=f"whether {producer} can run here at all",
            detail=(
                f"{kind} for {clip.label}: {producer} weight "
                f"{'at ' + str(checkpoint) if checkpoint else 'ABSENT (' + CHECKPOINT_ROOT + '/' + relative + ')'}; "
                f"consumed by {', '.join(variants)}"
            ),
        )


def all_runners() -> tuple[IdV2VControlStreamRunner, ...]:
    return (IdV2VControlStreamRunner(),)
