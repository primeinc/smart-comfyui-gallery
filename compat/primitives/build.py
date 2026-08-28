"""Deterministic inputs for the PRIMITIVE tier only.

This is not the corpus. A primitive case tests one transform -- `norm_crop` at
a size, `draw_kps` onto a canvas -- and a transform needs pixels with known
geometry, not a photograph. Generated from a fixed seed so every machine
produces byte-identical bytes, which is what lets a baseline hash recorded
here mean anything on another machine.

`compat/corpus/` is the different thing and must not be confused with this
one. A consumer case runs a detector, so it needs real faces spanning real
axes -- frontal/profile, expression, face size, occlusion, accessories,
resolution, one reference against many, same identity against mixed as a
negative control, and stills against video. Those are fetched, hashed the same
way, and never generated: synthetic pixels prove a warp and prove nothing
about a detector.

One axis belongs there that is easy to miss: the same identity through a
DIFFERENT capture and decode path. What the detector sees is `db/oriented.py`
`for_model` -- EXIF-turned and UNCAPPED, because that module states the rule
outright at :137-142: "a detector or an embedder is owed the real pixels,
because changing what a model sees changes what it records". The 1600 cap
belongs to `for_derivatives`, which serves thumbnails. So detections are at
full source resolution, and a corpus built only from our own downscaled
renderings would be measuring a decode path the producer never takes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

HERE: Path = Path(__file__).resolve().parent

#: One frame size for every generated fixture. Deliberately not square and
#: deliberately larger than any crop the consumers ask for (336 is the biggest
#: arcface size, 512 the biggest facexlib one), so a crop never runs off the
#: edge and silently tests the border-fill instead of the warp.
FRAME_WIDTH: int = 900
FRAME_HEIGHT: int = 1200

#: Five points in source pixels, in insightface's order:
#: right eye, left eye, nose, right mouth, left mouth. Placed off-centre and
#: slightly rotated so an alignment that ignores rotation still fails.
KEYPOINTS: tuple[tuple[float, float], ...] = (
    (352.0, 471.0),
    (528.0, 455.0),
    (447.0, 566.0),
    (369.0, 663.0),
    (521.0, 649.0),
)

SEED: int = 20260828


def frame() -> npt.NDArray[np.uint8]:
    """A deterministic BGR frame with structure at every scale.

    Noise alone would let a wrong-but-plausible warp pass by landing on
    statistically identical pixels, so this carries a coarse gradient, a fine
    checker and seeded noise together: a crop taken from the wrong place
    differs in all three.
    """
    rng = np.random.default_rng(SEED)
    ys, xs = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH]

    gradient = ((xs / FRAME_WIDTH) * 160.0 + (ys / FRAME_HEIGHT) * 60.0).astype(np.float64)
    checker = (((xs // 7) + (ys // 7)) % 2) * 35.0
    noise = rng.integers(0, 24, size=(FRAME_HEIGHT, FRAME_WIDTH)).astype(np.float64)

    plane = np.clip(gradient + checker + noise, 0, 255)
    out = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    # Distinct per channel, so a BGR/RGB swap anywhere in a pipeline shows up
    # as a divergence rather than passing silently.
    out[:, :, 0] = plane.astype(np.uint8)
    out[:, :, 1] = np.clip(plane * 0.6 + 40.0, 0, 255).astype(np.uint8)
    out[:, :, 2] = np.clip(255.0 - plane, 0, 255).astype(np.uint8)
    return out


def digest(values: npt.NDArray[np.generic]) -> str:
    """sha256 over the canonical bytes: C-order, plus dtype and shape.

    dtype and shape are hashed with the data because two arrays holding the
    same bytes under different shapes are not the same artifact, and a digest
    that cannot tell them apart is not evidence.
    """
    hasher = hashlib.sha256()
    hasher.update(str(values.dtype).encode("ascii"))
    hasher.update(repr(values.shape).encode("ascii"))
    hasher.update(np.ascontiguousarray(values).tobytes())
    return hasher.hexdigest()


def keypoints() -> npt.NDArray[np.float32]:
    return np.array(KEYPOINTS, dtype=np.float32)


def manifest() -> dict[str, object]:
    """What was generated, by content, so a reviewer can rebuild and compare."""
    pixels = frame()
    return {
        "seed": SEED,
        "frame": {
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "dtype": str(pixels.dtype),
            "sha256": digest(pixels),
        },
        "keypoints": {
            "order": "right_eye,left_eye,nose,right_mouth,left_mouth",
            "space": "source_pixels",
            "values": [list(one) for one in KEYPOINTS],
            "sha256": digest(keypoints()),
        },
    }


def main() -> int:
    out = HERE / "fixtures.json"
    with out.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(manifest(), indent=2, sort_keys=True))
        handle.write("\n")
    print(json.dumps(manifest(), indent=2, sort_keys=True))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
