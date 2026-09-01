from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Comparison:
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
    hasher = hashlib.sha256()
    hasher.update(str(values.dtype).encode("ascii"))
    hasher.update(repr(values.shape).encode("ascii"))
    hasher.update(np.ascontiguousarray(values).tobytes())
    return hasher.hexdigest()


def _measurable(values: npt.NDArray[np.generic]) -> bool:
    return values.dtype.kind in "biufc"


def _wide(values: npt.NDArray[np.generic]) -> npt.NDArray[np.number[Any]]:
    if values.dtype.kind in "biu":
        return values.astype(np.int64)
    if values.dtype.kind == "c":
        return values.astype(np.complex128)
    return values.astype(np.float64)


def _worst(baseline: npt.NDArray[np.generic], replay: npt.NDArray[np.generic]) -> float | None:
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
    if baseline.shape != replay.shape:
        return Comparison(
            equal=False,
            method="shape",
            detail=f"baseline {baseline.shape} against replay {replay.shape}",
            total=baseline.size,
        )
    if baseline.dtype != replay.dtype:
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
