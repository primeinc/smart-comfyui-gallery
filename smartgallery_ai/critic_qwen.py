"""Decomposed local VLM critic: Qwen2.5-VL-7B (Apache-2.0) via llama.cpp.

Why decomposed: the measured failure of monolithic small-VLM critics in
this repo was never "the model can't see" — it was free-form structured
output (schema violations, truncation, and worst, schema-valid
fabrication by parroting the prompt's example). This module applies the
same architecture that made OmniQuery work: the model only ever answers
SMALL, GRAMMAR-CONSTRAINED questions, and deterministic code assembles
the typed review.

Protocol per image:
  1. DESCRIBE  — short free-text factual description (nothing to parrot).
  2. GROUND    — deterministic anti-fabrication gate: CLIPScore between
                 the description and the image via the runtime-proven
                 OpenCLIP space. A description that does not match the
                 image aborts the review (CriticGroundingError) instead of
                 storing a plausible lie. No validator can catch a
                 well-formed fabrication; this gate can.
  3. ASSESS    — one JSON-schema-constrained call: quality score + up to
                 3 defects, each typed from the fixed vocabulary with
                 severity/confidence and a coarse region enum.
  4. LOCALIZE  — for each non-whole-image defect, one schema-constrained
                 bbox call (fractional coords). Invalid boxes degrade to
                 the coarse region's box — still model-claimed grounding,
                 never invented by us.
  5. ASSEMBLE  — our code builds the payload; prompt_alignment_score is
                 computed OUTSIDE the VLM as CLIPScore(prompt, image)
                 mapped to 0-10 (min(10, 25*max(cos,0)) — the standard
                 CLIPScore w=2.5 scaling on a 0-10 scale). The payload
                 still goes through validate_review_payload like every
                 other critic.

Weights load ONLY from the models dir (never downloaded at runtime):
  <models_dir>/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
  <models_dir>/mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Optional

import numpy as np
from PIL import Image

from smartgallery_ai.embedders import BackendUnavailable, SemanticEmbedder
from smartgallery_ai.review import FINDING_TYPES, CriticBackend

# Both published quantizations of the official ggml-org conversion are
# supported; the first provisioned file wins (higher fidelity first).
MODEL_FILENAMES = (
    "Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
    "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
)
MMPROJ_FILENAMES = (
    "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf",
    "mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf",
)

# Measured on the calibration set in tests/benchmarks (see AI_MODELS.md):
# grounded descriptions of matching images score well above this; mismatched
# descriptions fall below it.
DEFAULT_GROUNDING_MIN_COS = 0.20

_REGIONS = ("whole-image", "top-left", "top-right", "bottom-left",
            "bottom-right", "center")

_REGION_BOXES = {
    "top-left": (0.0, 0.0, 0.5, 0.5),
    "top-right": (0.5, 0.0, 0.5, 0.5),
    "bottom-left": (0.0, 0.5, 0.5, 0.5),
    "bottom-right": (0.5, 0.5, 0.5, 0.5),
    "center": (0.25, 0.25, 0.5, 0.5),
}

_ASSESS_SCHEMA = {
    "type": "object",
    "properties": {
        "quality_score": {"type": "number"},
        "defects": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"enum": list(FINDING_TYPES)},
                    "severity": {"enum": ["low", "medium", "high"]},
                    "confidence": {"type": "number"},
                    "region": {"enum": list(_REGIONS)},
                    "what": {"type": "string"},
                },
                "required": ["type", "severity", "confidence", "region", "what"],
            },
        },
    },
    "required": ["quality_score", "defects"],
}

_BBOX_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": "number"}, "y": {"type": "number"},
        "w": {"type": "number"}, "h": {"type": "number"},
    },
    "required": ["x", "y", "w", "h"],
}


def _first_existing(models_dir: str, names: tuple) -> Optional[str]:
    for name in names:
        p = os.path.join(models_dir, name)
        if os.path.isfile(p):
            return p
    return None


class CriticGroundingError(RuntimeError):
    """The critic's description failed the CLIP grounding gate: the model
    is not talking about this image. The review is aborted, never stored."""


def _data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))


def clip_score_10(cos: float) -> float:
    """CLIPScore-style mapping (w=2.5) onto the 0-10 UI scale."""
    return min(10.0, 25.0 * max(cos, 0.0))


class QwenVlCritic(CriticBackend):
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_version = "qwen2.5-vl-7b-q4_k_m+decomposed-v1"

    def __init__(self, models_dir: str,
                 semantic_embedder: Optional[SemanticEmbedder] = None,
                 grounding_min_cos: float = DEFAULT_GROUNDING_MIN_COS,
                 n_ctx: int = 8192, n_threads: int = 4):
        # Weights check precedes the runtime import: 'auto' resolution on
        # an unprovisioned system must stay fast and side-effect-free.
        model_path = _first_existing(models_dir, MODEL_FILENAMES)
        mmproj_path = _first_existing(models_dir, MMPROJ_FILENAMES)
        if model_path is None or mmproj_path is None:
            raise BackendUnavailable(
                f"qwen-vl weights not found under {models_dir} "
                f"(model: one of {MODEL_FILENAMES}; mmproj: one of {MMPROJ_FILENAMES})")
        self.model_version = (
            f"qwen2.5-vl-7b-{os.path.basename(model_path).rsplit('-', 1)[-1].removesuffix('.gguf').lower()}"
            f"+decomposed-v1")

        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"qwen-vl critic unavailable: {exc}") from exc
        try:
            handler = Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=False)
            self._llm = Llama(model_path=model_path, chat_handler=handler,
                              n_ctx=n_ctx, n_threads=n_threads, verbose=False)
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"failed to load qwen-vl weights: {exc}") from exc

        self._embedder = semantic_embedder
        self._grounding_min_cos = grounding_min_cos

    # -- constrained single-turn helpers ------------------------------------

    def _chat(self, img_uri: str, text: str, schema: Optional[dict],
              max_tokens: int) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": img_uri}},
                {"type": "text", "text": text},
            ],
        }]
        kwargs = dict(messages=messages, max_tokens=max_tokens, temperature=0.0)
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_object",
                "schema": schema,
            }
        out = self._llm.create_chat_completion(**kwargs)
        return out["choices"][0]["message"]["content"] or ""

    # -- protocol ------------------------------------------------------------

    def review(self, img: Image.Image, prompt_text: Optional[str],
               rubric_version: str) -> dict:
        img = img.convert("RGB")
        # Keep the vision token budget bounded and deterministic.
        if max(img.size) > 768:
            img = img.copy()
            img.thumbnail((768, 768), Image.LANCZOS)
        uri = _data_uri(img)

        # 1. DESCRIBE (free text; nothing schema-shaped to parrot)
        description = self._chat(
            uri, "Describe this image factually in two short sentences.",
            schema=None, max_tokens=120).strip()

        # 2. GROUND — deterministic anti-fabrication gate
        grounding_cos = None
        if self._embedder is not None:
            if not description:
                raise CriticGroundingError("critic produced no description")
            grounding_cos = _cos(self._embedder.embed_text(description),
                                 self._embedder.embed_image(img))
            if grounding_cos < self._grounding_min_cos:
                raise CriticGroundingError(
                    f"description does not match image "
                    f"(CLIP cos {grounding_cos:.3f} < {self._grounding_min_cos}); "
                    f"refusing to store an ungrounded review")

        # 3. ASSESS (grammar-constrained)
        assess_raw = self._chat(
            uri,
            "You are reviewing an AI-generated image for defects. "
            "Report overall technical quality 0-10 and up to 3 concrete "
            "defects you can actually see (empty list if none). For each "
            "defect give its category, severity, your confidence 0-1, the "
            "region where it is, and a few words describing it.",
            schema=_ASSESS_SCHEMA, max_tokens=400)
        assess = json.loads(assess_raw)

        findings = []
        for defect in (assess.get("defects") or [])[:3]:
            region = defect.get("region", "whole-image")
            localizable = region != "whole-image"
            bbox = None
            if localizable:
                # 4. LOCALIZE (grammar-constrained bbox for this defect)
                bbox = self._localize(uri, defect)
                if bbox is None:
                    bbox = _REGION_BOXES[region]
            finding = {
                "type": defect["type"],
                "severity": defect["severity"],
                "confidence": _clamp(defect.get("confidence", 0.5), 0.0, 1.0),
                "localizable": localizable,
                "description": str(defect.get("what", ""))[:300] or defect["type"],
            }
            if localizable:
                finding["bbox"] = list(bbox)
            findings.append(finding)

        # 5. ASSEMBLE (deterministic; alignment computed outside the VLM)
        alignment = None
        if prompt_text and self._embedder is not None:
            alignment = clip_score_10(
                _cos(self._embedder.embed_text(prompt_text),
                     self._embedder.embed_image(img)))

        summary = description[:280] if description else "(no description)"
        if grounding_cos is not None:
            summary += f" [grounding cos {grounding_cos:.2f}]"

        return {
            "quality_score": _clamp(assess.get("quality_score", 5.0), 0.0, 10.0),
            "prompt_alignment_score": alignment,
            "summary": summary,
            "findings": findings,
        }

    def _localize(self, uri: str, defect: dict) -> Optional[tuple]:
        try:
            raw = self._chat(
                uri,
                f"Locate this defect in the image: {defect.get('what', defect['type'])}. "
                "Give its bounding box as fractions of the image size "
                "(x, y = top-left corner; w, h = size; all between 0 and 1).",
                schema=_BBOX_SCHEMA, max_tokens=80)
            b = json.loads(raw)
            x = _clamp(b["x"], 0.0, 1.0)
            y = _clamp(b["y"], 0.0, 1.0)
            w = _clamp(b["w"], 0.0, 1.0 - x)
            h = _clamp(b["h"], 0.0, 1.0 - y)
            if w < 0.01 or h < 0.01:
                return None
            return (x, y, w, h)
        except Exception:  # noqa: BLE001 - degrade to the coarse region box
            return None


def _clamp(v, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return lo
