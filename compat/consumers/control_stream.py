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

A stream whose weights are absent is UNSUPPORTED and says which weight.
Silently falling back to source-video-only is the failure this module exists
to stop.
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

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
        label = source.parent.name
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
    """The weight this stream's producer needs, when it is on this machine."""
    root = os.environ.get(CHECKPOINT_ROOT)
    if not root:
        return None
    _, relative, _ = STREAMS[stream]
    candidate = Path(root) / relative
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


def _artifact(name: str, values: np.ndarray) -> Artifact:
    from compat.assertions.arrays import digest

    return Artifact(name=name, dtype=str(values.dtype), shape=values.shape, sha256=digest(values), values=values)


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
                        ablations=(Ablation(primitive=stream, expect_breaks=True),),
                        measurements=("rederivation_cost",),
                        note=f"{producer} writes this stream; consumed by {', '.join(variants)}",
                    )
                )
        return tuple(out)

    def _unavailable(self, stream: str, label: str) -> NotImplementedError:
        """Why this stream cannot be compared, with both facts and in order.

        The unimplemented stream is named before the absent weight. Reporting
        the weight alone makes the nine UNSUPPORTED rows in `cases.json` read
        as "install SAM3 / DepthAnything / DAVID and these will run". They
        will not: nothing here derives any stream but `decoded_source_frames`,
        whether
        or not the weight was there.

        So the unimplemented derivation is stated first, because it is the
        binding reason, and the checkpoint's presence is stated after it as
        the fact it is.
        """
        producer, relative, _ = STREAMS[stream]
        checkpoint = checkpoint_of(stream)
        where = (
            f"the {producer} weight IS present at {relative}"
            if checkpoint is not None
            else f"the {producer} weight at ${CHECKPOINT_ROOT}/{relative} is also absent"
        )
        return NotImplementedError(
            f"{stream} has no derivation in this suite: nothing here runs {producer} over {label}, "
            f"so there is no derived stream to retain and none to compare against. Installing a weight "
            f"would not make this case run. ({where}.) Re-deriving this stream is a {producer} pass over "
            f"every frame, which is the cost the retention question is about."
        )

    def retained_for(self, case: Case) -> RetainedState:
        kind, clip = self._parts(case)
        if kind == "decoded_source_frames":
            return RetainedState(source_video_bytes=np.frombuffer(clip.path.read_bytes(), dtype=np.uint8))
        raise self._unavailable(kind, clip.label)

    def baseline(self, case: Case) -> Artifact:
        kind, clip = self._parts(case)
        if kind == "decoded_source_frames":
            return _artifact(case.boundary, decode(clip.path.read_bytes()))
        raise self._unavailable(kind, clip.label)

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
