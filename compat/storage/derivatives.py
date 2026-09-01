from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from compat.assertions.arrays import digest

UInt8Array = npt.NDArray[np.uint8]


_SCRATCH: Final[Path] = Path(tempfile.mkdtemp(prefix="compat-derivatives-"))
atexit.register(shutil.rmtree, _SCRATCH, True)

_COUNTER: list[int] = [0]


def _scratch_path(suffix: str) -> Path:
    _COUNTER[0] += 1
    return _SCRATCH / f"{_COUNTER[0]:06d}{suffix}"


def _to_pil(bgr: UInt8Array) -> Any:
    from PIL import Image

    if bgr.ndim == 2:
        return Image.fromarray(bgr, mode="L")
    return Image.fromarray(bgr[:, :, ::-1])


def _from_pil(image: Any) -> UInt8Array:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)[:, :, ::-1].copy()


def preview(bgr: UInt8Array) -> tuple[UInt8Array, int]:
    from vision import decode, thumbs

    target = _scratch_path(".webp")
    thumbs.write(target, thumbs.fit(_to_pil(bgr), thumbs.EDGES["preview"]))
    with decode.open_still(target) as opened:
        opened.load()
        return _from_pil(opened), target.stat().st_size


def lossless(bgr: UInt8Array) -> tuple[UInt8Array, int]:
    from vision import decode

    target = _scratch_path(".png")
    _to_pil(bgr).save(target, format="PNG", optimize=False)
    with decode.open_still(target) as opened:
        opened.load()
        return _from_pil(opened), target.stat().st_size


def encoded(bgr: UInt8Array) -> tuple[UInt8Array, int]:
    from vision import decode, thumbs

    target = _scratch_path(".webp")
    thumbs.write(target, _to_pil(bgr))
    with decode.open_still(target) as opened:
        opened.load()
        return _from_pil(opened), target.stat().st_size


_ENCODED: dict[str, int] = {}


def lossless_bytes(frame: npt.NDArray[np.uint8]) -> int:
    key = digest(frame)
    if key not in _ENCODED:
        _ENCODED[key] = lossless(frame)[1]
    return _ENCODED[key]
