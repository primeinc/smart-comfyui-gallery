"""The consumer whose identity is carried by a voice, not a photograph.

It is in this population for one reason: to stop the storage design being
written as though every identity fact lives on a face row. It conditions on a
recording, and nothing the face lane produces can serve it however complete
that lane becomes.

`id_v2v` is not here. Its boundary is the three VACE condition streams rather
than the decoded source, so it belongs to
`compat/consumers/control_stream.py`, and the
dead copy is gone.

    id_lora   LTXVReferenceAudio.execute, ComfyUI@a9ab2b62dac1
              nodes_lt.py:881-893. Reads `reference_audio["waveform"]` and
              `["sample_rate"]`, resamples to the VAE's rate when they differ,
              and only then encodes. The latents and the `ref_tokens` built
              from them are the CONSUMER's, produced by its own VAE at its own
              rate -- storing them would freeze one model's opinion of a voice
              the way storing an aligned crop would freeze one model's opinion
              of a face. The durable artifact is the waveform.

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

from typing import Any, Final

import numpy as np

from compat.assertions.arrays import digest
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
from compat.storage import precision

#: nodes_lt.py:884. What the node falls back to when the VAE does not say.
VAE_SAMPLE_RATE: Final[int] = 44100

#: nodes_lt.py:867 tooltip: "~5 seconds recommended (training duration)".
CLIP_SECONDS: Final[float] = 5.0

#: A capture rate that is NOT the VAE's, so the resample branch is under test:
#: `torchaudio.functional.resample` returns the waveform untouched when the
#: rates match (functional.py:1473-1474), and so does the node.
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
                    # Every codec this application could keep is
                    # integer PCM, so the storable question for a
                    # float waveform is whether 16-bit serves.
                    Ablation(
                        primitive="audio_waveform",
                        swap="pcm_16_bit",
                        expect_breaks=True,
                        kind="substitution",
                    ),
                    # Without the rate the node cannot know whether to
                    # resample, and a clip replayed at the wrong one is the
                    # same voice at the wrong pitch.
                    Ablation(primitive="audio_sample_rate", expect_breaks=True),
                    # The rate the VAE wants, offered as though it were the
                    # capture rate: a store keeping the waveform and the wrong
                    # rate is the same voice at the wrong pitch.
                    Ablation(
                        primitive="audio_sample_rate",
                        swap="vae_rate_assumed",
                        expect_breaks=True,
                        kind="substitution",
                    ),
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

    def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState:
        if ablation.swap == "vae_rate_assumed":
            return retained.replacing("audio_sample_rate", float(VAE_SAMPLE_RATE))
        if ablation.swap == "pcm_16_bit":
            return retained.replacing("audio_waveform", precision.quantised(retained.points("audio_waveform")))
        return retained.without(ablation.primitive)

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


def all_runners() -> list[Any]:
    return [IdLoraAudioRunner()]
