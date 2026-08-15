"""Real box/point-prompted segmentation via MobileSAM (Apache-2.0).

MobileSAM is a distilled Segment Anything model (ViT-T image encoder,
~40 MB weights) with the same predictor API as the original SAM; CPU
inference takes roughly a second per image. Weights load ONLY from
`<models_dir>/mobile_sam.pt` — never downloaded at runtime.

This backend feeds `review.generate_finding_mask`, which remains the sole
gate: only localizable findings with real box/point grounding ever reach
`segment()`, so global findings can never be forced into fake masks.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from PIL import Image

from smartgallery_ai.embedders import BackendUnavailable
from smartgallery_ai.review import SegmenterBackend

WEIGHTS_FILENAME = "mobile_sam.pt"


class MobileSamSegmenter(SegmenterBackend):
    """Box/point-prompted MobileSAM predictor behind the
    `SegmenterBackend` contract; construction fails with
    `BackendUnavailable` unless weights and the torch/mobile_sam runtime
    are provisioned."""

    model_id = "ChaoningZhang/MobileSAM"
    model_version = "mobile_sam-vit_t-v1"

    def __init__(self, models_dir: str):
        """Load the ViT-T checkpoint from `models_dir`; raises
        `BackendUnavailable` when weights or the runtime are missing or
        fail to load."""
        # Weights check precedes the runtime import: 'auto' resolution on
        # an unprovisioned system must stay fast and side-effect-free.
        weights_path = os.path.join(models_dir, WEIGHTS_FILENAME)
        if not os.path.isfile(weights_path):
            raise BackendUnavailable(f"mobile_sam weights not found at {weights_path}")

        try:
            import torch
            from mobile_sam import SamPredictor, sam_model_registry
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"mobile_sam unavailable: {exc}") from exc
        try:
            model = sam_model_registry["vit_t"](checkpoint=weights_path)
            model.eval()
            self._predictor = SamPredictor(model)
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"failed to load mobile_sam weights: {exc}") from exc
        self._torch = torch

    def segment(self, img: Image.Image, bbox: Optional[tuple] = None,
                points: Optional[list] = None) -> np.ndarray:
        """Predict one boolean HxW mask from normalized-[0,1] prompts:
        `bbox` as (x, y, w, h), `points` as foreground clicks. At least
        one prompt is required; single-mask mode keeps output
        deterministic for a given prompt."""
        if bbox is None and not points:
            raise ValueError("segment() requires a bbox or points prompt")
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]

        with self._torch.no_grad():
            self._predictor.set_image(rgb)
            box_arr = None
            point_coords = None
            point_labels = None
            if bbox is not None:
                x, y, bw, bh = bbox
                box_arr = np.array(
                    [x * w, y * h, (x + bw) * w, (y + bh) * h], dtype=np.float32)
            if points:
                point_coords = np.array(
                    [[px * w, py * h] for px, py in points], dtype=np.float32)
                point_labels = np.ones(len(points), dtype=np.int32)
            masks, scores, _ = self._predictor.predict(
                point_coords=point_coords, point_labels=point_labels,
                box=box_arr, multimask_output=False)
        return masks[0].astype(bool)
