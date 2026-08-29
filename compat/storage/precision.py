"""Cheaper widths a column could actually be, so "minimum" means bytes too.

A removal answers "is this key indexed", which every key answers the same way.
The question this suite exists for is narrower and has a number in it: what is
the MINIMUM canonical evidence. Minimum in keys, and minimum in width.

So each retained numeric value gets one substitution that offers the same
value in a narrower storable form and asks whether the consumer still
reproduces. Both outcomes are informative and neither is guaranteed:

    it serves     the column can be half as wide, and the byte budget in
                  `answer.json` drops by that much
    it fails      the producer's width is load-bearing, which is a fact about
                  the schema and not about `numpy.copy`

`half` round-trips through float16 and comes back in the ORIGINAL dtype. The
dtype is restored on purpose: a column declared REAL and read back as float64
is a different finding (`gallery_v45` records exactly that for `det_score`),
and mixing the two would make every result here a dtype divergence with the
value loss unmeasured behind it.

float16 rather than an arbitrary rounding because it is a storable width --
numpy writes it, SQLite BLOBs hold it, safetensors names it -- and because its
error is a property of IEEE 754 binary16 rather than of a constant chosen
here: 10 explicit mantissa bits, so roughly 3.3 decimal digits, and a smallest
normal of 6.1e-5.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def half(values: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    """The same value through float16, returned in its original dtype.

    Non-floating input is returned unchanged: an integer or a byte string has
    no narrower IEEE form, and silently reinterpreting one would measure a
    cast rather than a precision budget.
    """
    if values.dtype.kind != "f":
        return values
    return values.astype(np.float16).astype(values.dtype)


def quantised(values: npt.NDArray[np.generic], bits: int = 16) -> npt.NDArray[np.generic]:
    """A signed waveform through a `bits`-wide integer and back.

    What a store holds for audio: WAV, FLAC and every codec this application
    could keep are integer PCM, so the question for a float waveform is
    whether the integer form serves. Returned in the original dtype for the
    same reason `half` is.
    """
    if values.dtype.kind != "f":
        return values
    full = float(2 ** (bits - 1) - 1)
    scaled: npt.NDArray[np.floating[Any]] = np.clip(values.astype(np.float64), -1.0, 1.0) * full
    return (np.rint(scaled) / full).astype(values.dtype)


def narrows(values: npt.NDArray[np.generic]) -> bool:
    """Whether `half` actually changes this value.

    The expectation a width substitution is allowed to carry. Asserting that
    narrowing always breaks is the same mistake as asserting that reversing a
    reference set always breaks: `build.KEYPOINTS` are whole numbers under
    2048, every one exactly representable in binary16, so the substitute WAS
    the original and four cases came back CONTRADICTED for a reason that had
    nothing to do with the crop.

    This is not the same proposition as the case's outcome, so it is not
    circular. It says the stored value changed; whether the consumer's output
    changes with it is what the ablation measures, and a value that narrowed
    while the output held is a real finding about the transform.
    """
    return values.dtype.kind == "f" and not np.array_equal(values, half(values))
