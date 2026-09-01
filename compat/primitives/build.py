from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from compat.assertions.arrays import digest

HERE: Path = Path(__file__).resolve().parent


FRAME_WIDTH: int = 900
FRAME_HEIGHT: int = 1200


KEYPOINTS: tuple[tuple[float, float], ...] = (
    (352.0, 471.0),
    (528.0, 455.0),
    (447.0, 566.0),
    (369.0, 663.0),
    (521.0, 649.0),
)

SEED: int = 20260828


def frame() -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng(SEED)
    ys, xs = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH]

    gradient = ((xs / FRAME_WIDTH) * 160.0 + (ys / FRAME_HEIGHT) * 60.0).astype(np.float64)
    checker = (((xs // 7) + (ys // 7)) % 2) * 35.0
    noise = rng.integers(0, 24, size=(FRAME_HEIGHT, FRAME_WIDTH)).astype(np.float64)

    plane = np.clip(gradient + checker + noise, 0, 255)
    out = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)

    out[:, :, 0] = plane.astype(np.uint8)
    out[:, :, 1] = np.clip(plane * 0.6 + 40.0, 0, 255).astype(np.uint8)
    out[:, :, 2] = np.clip(255.0 - plane, 0, 255).astype(np.uint8)
    return out


def keypoints() -> npt.NDArray[np.float32]:
    return np.array(KEYPOINTS, dtype=np.float32)


def manifest() -> dict[str, object]:
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
    body = json.dumps(manifest(), indent=2, sort_keys=True) + "\n"

    was = out.read_text(encoding="utf-8") if out.is_file() else ""
    moved = bool(was) and was != body

    with out.open("w", encoding="utf-8", newline="") as handle:
        handle.write(body)
    print(body)
    print(f"wrote {out}")
    if moved:
        print("\n!! the regenerated fixtures do not match the committed ones.")
        print("   This module claims every machine produces byte-identical bytes; here, one did not.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
