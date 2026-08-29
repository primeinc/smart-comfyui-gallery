"""What this application can hand back WITHOUT reopening the source file.

The question this suite answers names the constraint outright: served "without
reopening the original source media". For a face that means the columns of a
row. For a picture it means a derived artifact, and this application already
keeps exactly two of them -- `vision/thumbs.py` EDGES, a `thumb` at 512 and a
`preview` at 1440, both WebP at QUALITY 82 and METHOD 2, both keyed on
`content_sha256` and both explicitly "a cache of something regenerable"
(`db/oriented.for_derivatives`).

So a whole-reference consumer has two candidate answers and this module builds
both by executing the application's own code:

    preview     what the store can return today. Lossy and 1440-capped.
    lossless    what the store would have to ADD to serve these consumers: a
                full-resolution rendering, encoded losslessly.

Neither is declared to work. Both are measured, at the vendor's own boundary,
by the case that uses them -- which is the difference between this and a
`shot.frame.copy()` that could only ever reproduce itself.

PNG rather than a raw array for the lossless candidate because a store holds
bytes, not objects, and the byte count is half the answer: `answer.json` must
say what the minimum state COSTS, and an in-memory array has no size a schema
could budget for.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

UInt8Array = npt.NDArray[np.uint8]

#: One scratch directory for the process, removed when it exits. The encoders
#: this module executes are the application's own and both write to a path;
#: `tempfile.mkdtemp` with no owner is what left a directory behind per run.
_SCRATCH: Final[Path] = Path(tempfile.mkdtemp(prefix="compat-derivatives-"))
atexit.register(shutil.rmtree, _SCRATCH, True)

_COUNTER: list[int] = [0]


def _scratch_path(suffix: str) -> Path:
    _COUNTER[0] += 1
    return _SCRATCH / f"{_COUNTER[0]:06d}{suffix}"


def _to_pil(bgr: UInt8Array) -> Any:
    from PIL import Image

    return Image.fromarray(bgr[:, :, ::-1])


def _from_pil(image: Any) -> UInt8Array:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)[:, :, ::-1].copy()


def preview(bgr: UInt8Array) -> tuple[UInt8Array, int]:
    """The picture as the derived-image cache returns it, and its byte cost.

    `vision.thumbs.fit`, `vision.thumbs.write` and `vision.decode.open_still`
    are called rather than reimplemented: the edge, the quality and the encoder method are facts about
    this application, and a copy of them here would go stale silently the first
    time one moved.
    """
    from vision import decode, thumbs

    target = _scratch_path(".webp")
    thumbs.write(target, thumbs.fit(_to_pil(bgr), thumbs.EDGES["preview"]))
    with decode.open_still(target) as opened:
        opened.load()
        return _from_pil(opened), target.stat().st_size


def lossless(bgr: UInt8Array) -> tuple[UInt8Array, int]:
    """The picture as a full-resolution lossless artifact returns it.

    The candidate a store would have to ADD. PNG because it is lossless for
    8-bit RGB and because the encoded size is the number a schema budgets.
    """
    from vision import decode

    target = _scratch_path(".png")
    _to_pil(bgr).save(target, format="PNG", optimize=False)
    with decode.open_still(target) as opened:
        opened.load()
        return _from_pil(opened), target.stat().st_size


def encoded(bgr: UInt8Array) -> tuple[UInt8Array, int]:
    """The picture through the store's encoder at its NATIVE size.

    `preview` answers "can the 1440-capped derivative serve"; this answers the
    narrower question underneath it -- whether the WebP encode alone costs
    anything, with the resize taken out. `vision/thumbs` writes every raster
    variant this way, including the avatar crop, so a face patch the store
    keeps is a patch that went through exactly this.
    """
    from vision import decode, thumbs

    target = _scratch_path(".webp")
    thumbs.write(target, _to_pil(bgr))
    with decode.open_still(target) as opened:
        opened.load()
        return _from_pil(opened), target.stat().st_size
