"""Real box/point-prompted segmentation via MobileSAM (Apache-2.0).

MobileSAM is a distilled Segment Anything model (ViT-T image encoder,
~40 MB weights) with the same predictor API as the original SAM; CPU
inference takes roughly a second per image. Weights load ONLY from
`<models_dir>/mobile_sam.pt`; this module never downloads (weights
arrive via smartgallery_ai.provision).

This backend feeds `review.generate_finding_mask`, which remains the sole
gate: only localizable findings with real box/point grounding ever reach
`segment()`, so global findings can never be forced into fake masks.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import warnings

import numpy as np
from PIL import Image

from smartgallery_ai.embedders import BackendUnavailable
from smartgallery_ai.review import SegmenterBackend

WEIGHTS_FILENAME = "mobile_sam.pt"

_logger = logging.getLogger(__name__)


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

        # mobile_sam's import spews timm deprecation FutureWarnings and
        # model-registry-overwrite UserWarnings, and its checkpoint build
        # prints a tqdm bar to stderr; none of it is actionable by the
        # operator, so keep it off the server console. Errors still raise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", UserWarning)
            try:
                import torch
                from mobile_sam import SamPredictor, sam_model_registry
            except Exception as exc:
                raise BackendUnavailable(f"mobile_sam unavailable: {exc}") from exc
            try:
                from smartgallery_ai.embedders import (
                    pick_torch_device,
                    warn_if_vram_pressure,
                )
                device = pick_torch_device(torch, role="segmenter")
                self._device = device
                _logger.info("[AI] %s on device %s", self.model_id, device)
                warn_if_vram_pressure(torch, device, self.model_id)
                with contextlib.redirect_stderr(io.StringIO()):
                    model = sam_model_registry["vit_t"](checkpoint=weights_path)
                model.eval()
                self._predictor = SamPredictor(model.to(device))
            except Exception as exc:
                raise BackendUnavailable(
                    f"failed to load mobile_sam weights: {exc}") from exc
        self._torch = torch

    def segment(self, img: Image.Image, bbox: tuple | None = None,
                points: list | None = None) -> np.ndarray:
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
            masks, _scores, _ = self._predictor.predict(
                point_coords=point_coords, point_labels=point_labels,
                box=box_arr, multimask_output=False)
        return masks[0].astype(bool)
