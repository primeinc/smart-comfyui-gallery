"""The two consumers whose identity is not carried by a photograph.

They are in this population for one reason: to stop the storage design being
written as though every identity fact lives on a face row. One conditions on a
voice, the other on a video, and neither can be served by anything the face
lane produces however complete that lane becomes.

    id_lora   LTXVReferenceAudio.execute, ComfyUI@a9ab2b62dac1
              nodes_lt.py:881-893. Reads `reference_audio["waveform"]` and
              `["sample_rate"]`, resamples to the VAE's rate when they differ,
              and only then encodes. The latents and the `ref_tokens` built
              from them are the CONSUMER's, produced by its own VAE at its own
              rate -- storing them would freeze one model's opinion of a voice
              the way storing an aligned crop would freeze one model's opinion
              of a face. The durable artifact is the waveform.

    id_v2v    scripts/preprocess.sh, ID-V2V@33dd047835cf
              Reads `<SAMPLE_DIR>/source.mp4`, runs SAM3 person segmentation,
              and writes `orig_pixel.mp4` -- foreground on grey. The control
              stream is derived from the source video frame by frame and is
              keyed to the media, not to any face in it.

The boundary in both cases stops before the model, the same rule every image
consumer here follows: the resampled waveform for one, the decoded frame stack
for the other. Running SAM3 or an audio VAE would measure those models rather
than the storage contract.

FIXTURES ARE GENERATED, and that is a stated limit rather than a convenience.
The corpus is photographs; there is no voice and no video in it. A synthetic
waveform proves the resample contract because resampling has no notion of a
voice, and a synthetic clip proves the decode contract for the same reason.
Neither proves anything about SAM3's segmentation or about a real speaker, and
no case here claims to -- which is why both stay CONSUMER tier with their
boundary set short of the model.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from compat.contracts.case import (
    Ablation,
    Artifact,
    Case,
    Fixture,
    Float32Array,
    Measurement,
    RetainedState,
    Tier,
    UInt8Array,
)

#: nodes_lt.py:884. What the node falls back to when the VAE does not say.
VAE_SAMPLE_RATE: Final[int] = 44100

#: nodes_lt.py:867 tooltip: "~5 seconds recommended (training duration)".
CLIP_SECONDS: Final[float] = 5.0

#: A capture rate that is NOT the VAE's, so the resample branch is the one
#: under test. `torchaudio.functional.resample` returns the waveform untouched
#: when the rates match (functional.py:1473-1474), and so does the node, so
#: equal rates would exercise neither.
CAPTURE_SAMPLE_RATE: Final[int] = 16000

SEED: Final[int] = 20260828

#: A short deterministic clip. Frames and size are small on purpose: the claim
#: is about which artifact is durable, not about codec throughput.
VIDEO_FRAMES: Final[int] = 12
VIDEO_WIDTH: Final[int] = 160
VIDEO_HEIGHT: Final[int] = 120
VIDEO_FPS: Final[int] = 12


def waveform() -> Float32Array:
    """A deterministic stereo clip, structured rather than pure noise.

    Two tones per channel plus seeded noise, and the channels differ: a
    resample that collapsed or swapped them would otherwise land on
    statistically identical numbers and pass.
    """
    rng = np.random.default_rng(SEED)
    count = int(CLIP_SECONDS * CAPTURE_SAMPLE_RATE)
    t = np.arange(count, dtype=np.float64) / CAPTURE_SAMPLE_RATE
    left = 0.45 * np.sin(2 * np.pi * 220.0 * t) + 0.20 * np.sin(2 * np.pi * 1310.0 * t)
    right = 0.40 * np.sin(2 * np.pi * 277.2 * t) + 0.15 * np.sin(2 * np.pi * 990.0 * t)
    noise = rng.normal(0.0, 0.01, size=(2, count))
    return np.clip(np.stack([left, right]) + noise, -1.0, 1.0).astype(np.float32)


def resampled(clip: Float32Array, source_rate: int, target_rate: int) -> Float32Array:
    """Upstream's own resample call, nodes_lt.py:886.

    `torchaudio.functional.resample(waveform, orig_freq, new_freq)`, which
    takes `(..., time)` (functional.py:1455) -- so the batch axis ComfyUI
    audio carries passes straight through. Defaults are upstream's:
    lowpass_filter_width 6, rolloff 0.99, sinc_interp_hann. The node resamples
    BEFORE `movedim(1, -1)`, so the batched shape is what is resampled here.
    """
    import torch
    from torchaudio import functional

    if source_rate == target_rate:
        return clip
    out = functional.resample(torch.from_numpy(clip).unsqueeze(0), source_rate, target_rate)
    return np.asarray(out.squeeze(0).numpy(), dtype=np.float32)


def digest(values: npt.NDArray[np.generic]) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(values.dtype).encode("ascii"))
    hasher.update(repr(values.shape).encode("ascii"))
    hasher.update(np.ascontiguousarray(values).tobytes())
    return hasher.hexdigest()


def _artifact(name: str, values: Float32Array | UInt8Array) -> Artifact:
    return Artifact(
        name=name,
        dtype=str(values.dtype),
        shape=tuple(int(one) for one in values.shape),
        sha256=digest(values),
        values=values,
    )


class IdLoraAudioRunner:
    """The waveform is durable; the VAE's latents are not."""

    consumer_id: str = "id_lora"

    def __init__(self) -> None:
        self._clip = waveform()
        self._fixture = Fixture(
            name="deterministic_stereo_clip",
            path="compat/consumers/other_media.py::waveform",
            sha256=digest(self._clip),
            kind="synthetic_waveform",
            note=f"seed {SEED}, {CLIP_SECONDS}s stereo at {CAPTURE_SAMPLE_RATE} Hz; two tones plus seeded noise",
        )

    def cases(self) -> tuple[Case, ...]:
        return (
            Case(
                name="id_lora_reference_waveform",
                consumer_id=self.consumer_id,
                tier=Tier.CONSUMER,
                fixture=self._fixture,
                boundary=f"resampled_waveform@{VAE_SAMPLE_RATE}",
                exact_bytes=True,
                rtol=0.0,
                atol=0.0,
                retained=("audio_waveform", "audio_sample_rate"),
                ablations=(
                    Ablation(primitive="audio_waveform", expect_breaks=True),
                    # The rate is not decoration. Without it the node cannot
                    # know whether to resample, and a clip replayed at the
                    # wrong rate is the same voice at the wrong pitch -- which
                    # nothing downstream would flag.
                    Ablation(primitive="audio_sample_rate", expect_breaks=True),
                ),
                measurements=("waveform_against_latent_cost",),
                note="boundary is the resampled waveform, the last deterministic artifact before the audio VAE",
            ),
        )

    def retained_for(self, case: Case) -> RetainedState:
        return RetainedState(audio_waveform=self._clip.copy(), audio_sample_rate=float(CAPTURE_SAMPLE_RATE))

    def baseline(self, case: Case) -> Artifact:
        return _artifact(case.boundary, resampled(self._clip, CAPTURE_SAMPLE_RATE, VAE_SAMPLE_RATE))

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        clip = retained.points("audio_waveform")
        rate = int(retained.number("audio_sample_rate"))
        return _artifact(case.boundary, resampled(clip, rate, VAE_SAMPLE_RATE))

    def ablate(self, case: Case, retained: RetainedState, primitive: str) -> RetainedState:
        return retained.without(primitive)

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        """What the waveform costs, and why the tokens are not an alternative."""
        if name != "waveform_against_latent_cost":
            raise KeyError(f"{self.consumer_id} has no measurement called {name!r}")
        clip = retained.points("audio_waveform")
        rate = int(retained.number("audio_sample_rate"))
        at_vae = resampled(clip, rate, VAE_SAMPLE_RATE)
        return Measurement(
            name=name,
            unit="bytes",
            value=float(clip.nbytes),
            basis="float32 waveform at capture rate, against the same clip resampled to the VAE's rate",
            detail=(
                f"stored {clip.shape} at {rate} Hz = {clip.nbytes:,} B; resampled to {VAE_SAMPLE_RATE} Hz "
                f"it becomes {at_vae.shape} = {at_vae.nbytes:,} B ({at_vae.nbytes / clip.nbytes:.2f}x). "
                f"Storing the VAE's tokens instead would freeze one model's rate and one model's encoder, "
                f"and could not be resampled back"
            ),
        )


class IdV2VVideoRunner:
    """The source video is durable; the control stream is derived from it."""

    consumer_id: str = "id_v2v"

    def __init__(self) -> None:
        self._scratch = Path(tempfile.mkdtemp(prefix="compat_idv2v_"))
        self._path = self._scratch / "source.mp4"
        self._write_source()
        self._fixture = Fixture(
            name="deterministic_source_mp4",
            path=str(self._path),
            sha256=hashlib.sha256(self._path.read_bytes()).hexdigest(),
            kind="synthetic_video",
            note=f"seed {SEED}, {VIDEO_FRAMES} frames at {VIDEO_WIDTH}x{VIDEO_HEIGHT}, mpeg4 yuv420p",
        )

    def _frames(self) -> list[UInt8Array]:
        """Frames with motion, so a decoder that dropped or reordered shows.

        A static clip would let a decoder returning frame zero twelve times
        reproduce perfectly, which is precisely the failure a video case is
        supposed to be able to see.
        """
        rng = np.random.default_rng(SEED)
        base = rng.integers(0, 200, size=(VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        out: list[UInt8Array] = []
        for index in range(VIDEO_FRAMES):
            frame = base.copy()
            x = (index * 11) % (VIDEO_WIDTH - 20)
            frame[20:60, x : x + 20] = np.array([255, 40, 40], dtype=np.uint8)
            out.append(np.asarray(frame, dtype=np.uint8))
        return out

    def _write_source(self) -> None:
        """Encode the fixture, following PyAV's own numpy example.

        `examples/numpy/generate_video.py`: add_stream, set width/height/
        pix_fmt, mux each packet `stream.encode(frame)` yields, then flush with
        a bare `stream.encode()`. Skipping the flush leaves the tail of the
        clip unwritten, which would silently shorten the fixture.
        """
        import av

        with av.open(str(self._path), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=VIDEO_FPS)
            stream.width = VIDEO_WIDTH
            stream.height = VIDEO_HEIGHT
            stream.pix_fmt = "yuv420p"
            for frame in self._frames():
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    def decode(self, blob: bytes) -> UInt8Array:
        """Every frame the source carries, as one stacked array.

        Decoded from BYTES rather than from a path, because that is the claim
        under test: the durable artifact is the media itself, and a replay that
        reached for a filename would be leaning on the source file still being
        there -- which is the thing the whole suite is trying not to assume.
        """
        import io

        import av

        frames: list[UInt8Array] = []
        try:
            # `mode="r"` explicitly: PyAV's stubs overload on it
            # (av/container/core.pyi:108-119), and without it the return type
            # is the InputContainer|OutputContainer union, which has no
            # `decode`. Stating the mode is also just true.
            with av.open(io.BytesIO(blob), mode="r") as container:
                frames.extend(
                    np.asarray(frame.to_ndarray(format="rgb24"), dtype=np.uint8) for frame in container.decode(video=0)
                )
        except av.FFmpegError as problem:
            # Translated for the MESSAGE, not for the catch: PyAV's
            # `FFmpegError` already subclasses `ValueError`, so the executor
            # would record this as a break either way. What it would record is
            # `InvalidDataError: [Errno 1094995529] Invalid data found`, which
            # says nothing about what was asked. The ablation that lands here
            # is the one offering a face row where video bytes belong, and the
            # evidence should say so.
            raise ValueError(f"these bytes are not decodable video: {type(problem).__name__}: {problem}") from problem
        if not frames:
            raise ValueError("no frames decoded from the retained bytes")
        return np.asarray(np.stack(frames), dtype=np.uint8)

    def cases(self) -> tuple[Case, ...]:
        return (
            Case(
                name="id_v2v_source_frames",
                consumer_id=self.consumer_id,
                tier=Tier.CONSUMER,
                fixture=self._fixture,
                boundary="decoded_source_frames",
                exact_bytes=True,
                rtol=0.0,
                atol=0.0,
                retained=("source_video_bytes",),
                ablations=(
                    Ablation(primitive="source_video_bytes", expect_breaks=True),
                    # A face row is the whole of what the face lane can offer.
                    # It does not decode into frames, and that is the point.
                    Ablation(primitive="face_row_substituted", expect_breaks=True, kind="substitution"),
                ),
                measurements=("frames_and_bytes",),
                note="boundary is the decoded frame stack, before SAM3; the control stream is derived per frame",
            ),
        )

    def retained_for(self, case: Case) -> RetainedState:
        return RetainedState(source_video_bytes=np.frombuffer(self._path.read_bytes(), dtype=np.uint8))

    def baseline(self, case: Case) -> Artifact:
        return _artifact(case.boundary, self.decode(self._path.read_bytes()))

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        return _artifact(case.boundary, self.decode(retained.pixels("source_video_bytes").tobytes()))

    def ablate(self, case: Case, retained: RetainedState, primitive: str) -> RetainedState:
        if primitive == "face_row_substituted":
            # A 512-d embedding plus five keypoints: the whole of what a face
            # row carries, offered where the video bytes were.
            rng = np.random.default_rng(SEED)
            row = rng.standard_normal(512 + 10).astype(np.float32)
            return retained.replacing("source_video_bytes", np.frombuffer(row.tobytes(), dtype=np.uint8))
        return retained.without(primitive)

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        if name != "frames_and_bytes":
            raise KeyError(f"{self.consumer_id} has no measurement called {name!r}")
        blob = retained.pixels("source_video_bytes")
        frames = self.decode(blob.tobytes())
        return Measurement(
            name=name,
            unit="bytes",
            value=float(blob.nbytes),
            basis="the encoded source against the frame stack SAM3 would be handed",
            detail=(
                f"source {blob.nbytes:,} B decodes to {frames.shape} = {frames.nbytes:,} B of frames "
                f"({frames.nbytes / blob.nbytes:.1f}x). The control stream (orig_pixel.mp4) is derived "
                f"per frame from this and is not durable state"
            ),
        )


def all_runners() -> list[Any]:
    return [IdLoraAudioRunner(), IdV2VVideoRunner()]
