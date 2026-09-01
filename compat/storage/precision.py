from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def half(values: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    if values.dtype.kind != "f":
        return values
    return values.astype(np.float16).astype(values.dtype)


def quantised(values: npt.NDArray[np.generic], bits: int = 16) -> npt.NDArray[np.generic]:
    if values.dtype.kind != "f":
        return values
    full = float(2 ** (bits - 1) - 1)
    scaled: npt.NDArray[np.floating[Any]] = np.clip(values.astype(np.float64), -1.0, 1.0) * full
    return (np.rint(scaled) / full).astype(values.dtype)


def narrows(values: npt.NDArray[np.generic]) -> bool:
    return values.dtype.kind == "f" and not np.array_equal(values, half(values))
