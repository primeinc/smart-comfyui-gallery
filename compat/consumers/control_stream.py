"""ID-V2V's control streams, each a durable candidate in its own right.

Two non-interchangeable checkpoints (ID-V2V@33dd047835cf checkpoints/README.md,
"ID-V2V checkpoints" table):

    idv2v.pth                   1 VACE condition   orig_pixel.mp4
    idv2v_with_normal_depth.pth 3 VACE conditions  orig_pixel.mp4,
                                                   david_normal.mp4, depth.mp4

Producers, one per stream:

    orig_pixel.mp4    SAM3 person segmentation, then foreground-on-gray
                      scripts/preprocess.sh:57-67
    david_normal.mp4  DAViD multi-task ONNX, surface normals
                      scripts/idv2v_with_normal_depth/preprocess_with_depth.sh:64-68
    depth.mp4         DepthAnything-V2 ViT-L
                      scripts/idv2v_with_normal_depth/preprocess_with_depth.sh:70-75

WHAT THIS MODULE REPLACED, AND WHY
----------------------------------
The previous runner asserted "the control stream is derived per frame from
this and is not durable state", declared its boundary as the decoded frame
stack, and ran on a synthetic 12-frame clip. Three defects:

  1. It contradicted its own manifest row, which declares
     `retained = ["source_video", "control_stream"]`.
  2. Re-deriving a stream is a SAM3, DAViD or DepthAnything-V2 pass over
     every frame. The question this suite asks is what must be retained so an
     expensive producer is not re-run; a boundary placed before the expensive
     producer cannot answer it.
  3. The fixture was generated. `test_samples/` carries 14 real `source.mp4`
     files of real people at the pinned commit.

Every stream is DERIVED or the lane fails. There is no verdict for a boundary
nobody wrote: `_derive` runs the producer, and an absent weight or an absent
implementation raises out of `run_case` and reds the shard.

Silently falling back to source-video-only is the failure this module exists
to stop.
"""

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

#: Repo-relative sample media, real video of real people, at the pinned commit.
SAMPLES: Final[str] = "test_samples"

#: Where a user's downloaded checkpoints live, per checkpoints/README.md.
#: `checkpoints/` is gitignored in the upstream repo, so a read-only ref mirror
#: never holds them; `IDV2V_CHECKPOINTS` names a real download when there is one.
CHECKPOINT_ROOT: Final[str] = "IDV2V_CHECKPOINTS"

#: Stream -> (producer upstream key, checkpoint path under the checkpoint root,
#: the variants that consume it). Paths are the defaults the vendor's own
#: scripts set.
STREAMS: Final[dict[str, tuple[str, str, tuple[str, ...]]]] = {
    "orig_pixel_mp4": ("sam3", "sam3", ("idv2v", "idv2v_with_normal_depth")),
    "david_normal_mp4": ("david", "david/multi-task-model-vitl16_384.onnx", ("idv2v_with_normal_depth",)),
    "depth_mp4": ("depth_anything_v2", "depthv2/depth_anything_v2_vitl.pth", ("idv2v_with_normal_depth",)),
}

#: How many sample clips to exercise. Every one is real footage; the cap keeps
#: a decode lane from turning into a codec benchmark.
CLIPS: Final[int] = 4

#: The concept SAM3 segments, and the fill for everything outside it. Both are
#: upstream's own defaults: `scripts/preprocess.sh` passes `--sam_prompt` and
#: `orig_pixel.py` fills the background with 127.
SAM_PROMPT: Final[str] = "person"
GRAY_VALUE: Final[int] = 127


@dataclass(frozen=True)
class Clip:
    """One vendor sample, by content."""

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
    """The pinned ID-V2V clone, from the manifest's own refs root."""
    manifest = provenance.load_manifest()
    for row in manifest.get("consumers", []):
        if row["id"] == CONSUMER_ID:
            return provenance.clone_dir(
                (Path(__file__).resolve().parent.parent.parent / manifest["refs_root"]).resolve(),
                row["repo"],
            )
    raise KeyError(f"{CONSUMER_ID} is not in the manifest")


def clips(limit: int = CLIPS) -> list[Clip]:
    """The vendor's own sample clips, chosen by digest.

    Sorted by sha256 rather than by path so the slice does not move when a
    directory listing does.
    """
    root = repo_root() / SAMPLES
    if not root.is_dir():
        return []
    found: list[Clip] = []
    for source in sorted(root.rglob("source.mp4")):
        sha, size = _digest(source)
        # The path from `test_samples` down, which is unique. A parent
        # directory name is not: `cover_eyes` sits under more than one group,
        # and the runner keys its clips by label.
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
    """The weight this stream's producer needs, when it is on this machine.

    Through the manifest's `[provisioned]` root, so an unset environment
    variable is not the same fact as an absent file. It used to return None
    the moment `IDV2V_CHECKPOINTS` was unset, which it always was, and every
    stream read ABSENT whatever was on disk. `just compat weights` downloads
    into that same root.
    """
    from compat.harness import provision

    _, relative, _ = STREAMS[stream]
    candidate = provision.root_of(provenance.load_manifest()) / relative
    return candidate if candidate.exists() else None


def decode(blob: bytes) -> UInt8Array:
    """Every frame the bytes carry, stacked.

    From BYTES, never a path: the claim under test is that the media itself is
    the durable artifact, and a replay reaching for a filename would be leaning
    on the source file still existing.
    """
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


#: The rate upstream muxes its control streams at, and the quality a store
#: choosing mp4 over raw frames would pick. Neither changes a round trip's
#: SHAPE, which is what makes the substitution weigh the consumer's output.
STREAM_FPS: Final[int] = 16
STREAM_CRF: Final[str] = "28"


def encode(stack: UInt8Array) -> bytes:
    """The frame stack as an mp4, the way upstream ships a control stream.

    The point of the substitution this serves: a store that keeps a stream as
    the mp4 upstream ships, rather than as frames, returns the same shape and
    different pixels. A removal cannot establish that -- the replay indexes
    the key it was just denied -- and a comparison that settles on shape or an
    exception never reaches the consumer's output at all.
    """
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


#: Frames a derivation runs over, recorded in the evidence so the bound is
#: stated. A producer pass costs seconds per frame and the first frames carry
#: the claim.
DERIVE_FRAMES: Final[int] = 8


#: One loaded model per stream, per process. `_derive` runs once per case and
#: `id_v2v` declares sixteen, over three models of 3.4, 1.4 and 1.3 GB.
_LOADED: dict[str, Any] = {}


def _david_normals(checkpoint: Path, frames: UInt8Array) -> UInt8Array:
    """DAViD's surface normals, from the vendor's own estimator at its pin.

    Imported out of the pinned ID-V2V clone rather than reimplemented: the
    normals are the boundary, and a second implementation of them would be
    comparing this suite against itself.
    """
    # torch FIRST, before any ORT session exists. onnxruntime-gpu carries no
    # CUDA wheels and finds the runtime in torch's own lib directory, so the
    # other order is a symbol mismatch: WinError 127.
    import torch

    from compat.vendor.acceptance import _importable

    del torch
    clone = repo_root()
    with _importable(clone / "src"):
        # Through `import_module`, as `acceptance.py` does for reactor and
        # uniportrait: the module exists only while the clone is on sys.path,
        # so a static import cannot resolve and this tree bans suppressions.
        if "david" not in _LOADED:
            estimator_module = importlib.import_module("idv2v.preprocess.david_runtime.multi_task_estimator")
            _LOADED["david"] = estimator_module.MultiTaskEstimator(str(checkpoint))
        estimator = _LOADED["david"]
        out = [estimator.estimate_normal(one) for one in frames]
    # The estimator returns float normals in [-1, 1]; the stream is the image
    # ID-V2V writes, so it is encoded the way `david.py` writes it.
    stacked = np.asarray(out, dtype=np.float32)
    return np.asarray(((stacked + 1.0) * 127.5).clip(0, 255), dtype=np.uint8)


def _depth_v2(checkpoint: Path, frames: UInt8Array) -> UInt8Array:
    """DepthAnything-V2 dense depth, through ID-V2V's own vendored annotator.

    `DepthV2Annotator` is called directly rather than through
    `run_vace_preprocess`: that writes an mp4 and reads it back, so the
    boundary would carry an H.264 round trip that has nothing to do with the
    model. The per-frame normalise-and-repeat is upstream's own
    (annotators/depth.py:68-79) and it IS the artifact, so it is used rather
    than reimplemented.
    """
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
    """SAM3 person masks, composited foreground-on-gray.

    Both halves are upstream's, called rather than reimplemented:
    `sam3.init_sam3_video` / `run_sam3_video` produce the per-frame instance
    masks, and `orig_pixel.py` defines the composite -- keep the pixels inside
    the union of the masks, fill the rest with 127, and a frame with no mask
    becomes fully gray.

    `Image.fromarray` creates the image from the array interface
    (python-pillow/Pillow src/PIL/Image.py:3378), which is what the video
    session wants; the decoded frames are already uint8 RGB.
    """
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


#: The pass that produces each stream. A table rather than a chain of `if`s,
#: so a stream declared in STREAMS with no pass behind it is a missing key
#: here rather than a fallthrough that has to name itself unimplemented.
DERIVATIONS: Final[dict[str, Callable[[Path, UInt8Array], UInt8Array]]] = {
    "david_normal_mp4": _david_normals,
    "depth_mp4": _depth_v2,
    "orig_pixel_mp4": _orig_pixel,
}


def _derive(stream: str, clip: Clip) -> UInt8Array:
    """Run this stream's producer over the clip's first frames.

    Raises rather than returning a verdict. A stream nobody derived is a fact
    about this suite, and `run_case` records the raise as DIVERGED -- red, and
    carrying the reason -- so the other consumers in the shard still run.
    """
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
    """The source video, and each control stream derived from it."""

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
                    ablations=(
                        Ablation(primitive="source_video_bytes", expect_breaks=True),
                        Ablation(
                            primitive="source_video_bytes",
                            swap="face_row",
                            expect_breaks=True,
                            kind="substitution",
                        ),
                        # The source kept as a re-encode rather than the bytes
                        # that arrived: it decodes to the SAME shape, so the
                        # comparison reaches the frames themselves.
                        Ablation(
                            primitive="source_video_bytes",
                            swap="transcoded",
                            expect_breaks=True,
                            kind="substitution",
                        ),
                    ),
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
                            # The stream kept as the mp4 upstream ships rather
                            # than as frames: same shape, lossy pixels, so the
                            # comparison weighs what the consumer receives.
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
        if ablation.swap == "face_row":
            # A 512-d embedding plus five keypoints: the whole of what a face
            # row carries, offered where video bytes belong.
            rng = np.random.default_rng(20260828)
            row = rng.standard_normal(512 + 10).astype(np.float32)
            return retained.replacing("source_video_bytes", np.frombuffer(row.tobytes(), dtype=np.uint8))
        if ablation.swap == "transcoded":
            held = encode(decode(retained.pixels("source_video_bytes").tobytes()))
            return retained.replacing("source_video_bytes", np.frombuffer(held, dtype=np.uint8))
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
