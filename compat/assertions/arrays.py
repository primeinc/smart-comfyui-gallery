"""Comparing two boundary artifacts, and saying exactly how they differ.

A comparison that reports only pass/fail cannot be reviewed. Every result here
carries the worst observed difference and where it was, so a divergence can be
argued about rather than merely believed -- and so a reviewer can tell a
one-pixel resampling edge apart from a crop taken from the wrong place.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Comparison:
    """What comparing two arrays actually found."""

    equal: bool
    method: str
    detail: str
    max_abs_diff: float | None = None
    differing: int = 0
    total: int = 0

    @property
    def fraction_differing(self) -> float:
        return (self.differing / self.total) if self.total else 0.0


def digest(values: npt.NDArray[np.generic]) -> str:
    """sha256 over dtype, shape and C-ordered bytes.

    Shape and dtype are hashed with the data: two arrays holding identical
    bytes under different shapes are not the same artifact, and a digest that
    cannot tell them apart is not evidence.
    """
    hasher = hashlib.sha256()
    hasher.update(str(values.dtype).encode("ascii"))
    hasher.update(repr(values.shape).encode("ascii"))
    hasher.update(np.ascontiguousarray(values).tobytes())
    return hasher.hexdigest()


def _measurable(values: npt.NDArray[np.generic]) -> bool:
    """Whether a difference can be taken over these values at all."""
    return values.dtype.kind in "biufc"


def _wide(values: npt.NDArray[np.generic]) -> npt.NDArray[np.number[Any]]:
    """The dtype a difference may be taken in without wrapping OR truncating.

    Both halves matter and only one was here. uint8 subtraction wraps, so a
    real difference of 255 reported as 1 reads as a rounding error -- that is
    why integers widen to int64. But float32 cast to int64 TRUNCATES, so two
    embeddings differing by 0.7 reported as 0: every float comparison in this
    suite recorded `worst 0.0` while every element differed, and
    `answer.json` published "512 of 512 elements differ, worst 0.0" as a
    substitution proof.
    """
    if values.dtype.kind in "biu":
        return values.astype(np.int64)
    if values.dtype.kind == "c":
        return values.astype(np.complex128)
    return values.astype(np.float64)


def _worst(baseline: npt.NDArray[np.generic], replay: npt.NDArray[np.generic]) -> float | None:
    """The largest absolute difference, or None when there is no such number."""
    if not (_measurable(baseline) and _measurable(replay)) or baseline.size == 0:
        return None
    return float(np.max(np.abs(_wide(baseline) - _wide(replay))))


def compare(
    baseline: npt.NDArray[np.generic],
    replay: npt.NDArray[np.generic],
    *,
    exact_bytes: bool,
    rtol: float,
    atol: float,
) -> Comparison:
    """Baseline against replay, at the case's own tolerance.

    Shape and dtype are checked before values, and separately: an array that
    differs in shape has not "diverged numerically", it has been reconstructed
    wrong, and folding the two into one number would hide which happened.
    """
    if baseline.shape != replay.shape:
        return Comparison(
            equal=False,
            method="shape",
            detail=f"baseline {baseline.shape} against replay {replay.shape}",
            total=baseline.size,
        )
    if baseline.dtype != replay.dtype:
        # Measured anyway when the two are numerically comparable: a dtype
        # change is a divergence, but returning without a number lets a reader
        # print the absent measurement as "worst 0".
        apart = _worst(baseline, replay) if baseline.shape == replay.shape else None
        return Comparison(
            equal=False,
            method="dtype",
            detail=(
                f"baseline {baseline.dtype} against replay {replay.dtype}"
                + ("" if apart is None else f", values differ by at most {apart}")
            ),
            max_abs_diff=apart,
            total=baseline.size,
        )

    total = baseline.size
    if exact_bytes:
        same = np.array_equal(baseline, replay)
        differing = 0 if same else int(np.count_nonzero(baseline != replay))
        worst = 0.0 if same else (_worst(baseline, replay) or 0.0)
        return Comparison(
            equal=same,
            method="exact_bytes",
            detail="byte-identical" if same else f"{differing} of {total} elements differ, worst {worst}",
            max_abs_diff=worst,
            differing=differing,
            total=total,
        )

    left = baseline.astype(np.float64)
    right = replay.astype(np.float64)
    close = np.isclose(left, right, rtol=rtol, atol=atol)
    differing = int(np.count_nonzero(~close))
    worst = float(np.max(np.abs(left - right))) if total else 0.0
    return Comparison(
        equal=differing == 0,
        method=f"allclose rtol={rtol} atol={atol}",
        detail=("within tolerance" if differing == 0 else f"{differing} of {total} elements outside, worst {worst}"),
        max_abs_diff=worst,
        differing=differing,
        total=total,
    )
