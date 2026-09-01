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

VAE_SAMPLE_RATE: Final[int] = 44100


CLIP_SECONDS: Final[float] = 5.0


CAPTURE_SAMPLE_RATE: Final[int] = 16000

SEED: Final[int] = 20260828


VIDEO_FRAMES: Final[int] = 12
VIDEO_WIDTH: Final[int] = 160
VIDEO_HEIGHT: Final[int] = 120
VIDEO_FPS: Final[int] = 12


def waveform() -> Float32Array:
    rng = np.random.default_rng(SEED)
    count = int(CLIP_SECONDS * CAPTURE_SAMPLE_RATE)
    t = np.arange(count, dtype=np.float64) / CAPTURE_SAMPLE_RATE
    left = 0.45 * np.sin(2 * np.pi * 220.0 * t) + 0.20 * np.sin(2 * np.pi * 1310.0 * t)
    right = 0.40 * np.sin(2 * np.pi * 277.2 * t) + 0.15 * np.sin(2 * np.pi * 990.0 * t)
    noise = rng.normal(0.0, 0.01, size=(2, count))
    return np.clip(np.stack([left, right]) + noise, -1.0, 1.0).astype(np.float32)


def resampled(clip: Float32Array, source_rate: int, target_rate: int) -> Float32Array:
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
                    Ablation(primitive="audio_sample_rate", expect_breaks=True),
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
        return retained.without(ablation.primitive)

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
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
