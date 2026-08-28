"""Comparing two boundary artifacts, and saying exactly how they differ.

A comparison that reports only pass/fail cannot be reviewed. Every result here
carries the worst observed difference and where it was, so a divergence can be
argued about rather than merely believed -- and so a reviewer can tell a
one-pixel resampling edge apart from a crop taken from the wrong place.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

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
        return Comparison(
            equal=False,
            method="dtype",
            detail=f"baseline {baseline.dtype} against replay {replay.dtype}",
            total=baseline.size,
        )

    total = baseline.size
    if exact_bytes:
        same = np.array_equal(baseline, replay)
        differing = 0 if same else int(np.count_nonzero(baseline != replay))
        worst = 0.0
        if not same:
            # Cast through int64 first: uint8 subtraction wraps, and a wrapped
            # difference of 1 for a real difference of 255 would report a
            # catastrophic divergence as a rounding error.
            worst = float(np.max(np.abs(baseline.astype(np.int64) - replay.astype(np.int64))))
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
