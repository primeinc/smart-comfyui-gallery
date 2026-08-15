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
  4. LOCALIZE  — for defects whose TYPE is inherently spatial (anatomy,
                 artifact, text_render, detail_loss, other) and whose
                 region is not whole-image: one schema-constrained bbox
                 call (fractional coords). A finding is localizable ONLY
                 if this step yields a geometrically valid model-emitted
                 box; otherwise it becomes a GLOBAL finding (region kept
                 as text). No region-rectangle fallbacks — invented
                 geometry is the "fake mask" the ticket forbids.
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

# Only finding types with an inherent spatial locus may be localizable.
# lighting/style/composition/prompt_mismatch are properties of the whole
# image; letting a region enum turn them into mask-bearing findings would
# be exactly the "forced fake mask" the ticket forbids.
_LOCALIZABLE_TYPES = frozenset(
    {"anatomy", "artifact", "text_render", "detail_loss", "other"})

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


def check_grounding(embedder: SemanticEmbedder, description: str,
                    img: Image.Image,
                    min_cos: float = DEFAULT_GROUNDING_MIN_COS) -> float:
    """The deterministic anti-fabrication gate, exposed as a pure function
    so its negative cases are testable without loading the VLM. Returns the
    CLIP cosine on success; raises CriticGroundingError when the
    description does not match the image (or is empty)."""
    if not description:
        raise CriticGroundingError("critic produced no description")
    cos = _cos(embedder.embed_text(description), embedder.embed_image(img))
    # Fail CLOSED on NaN: `cos < min_cos` would be False for NaN (a
    # degenerate zero-norm embedding), silently passing the gate.
    if not (cos >= min_cos):
        raise CriticGroundingError(
            f"description does not match image (CLIP cos {cos:.3f} < "
            f"{min_cos}); refusing to store an ungrounded review")
    return cos


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
        # FAIL CLOSED on the grounding dependency: the CLIP gate is the
        # anti-fabrication mechanism this critic's measured record (and its
        # 'auto' enablement) relies on. Running the VLM without it would be
        # a strictly weaker, unmeasured configuration, so it is not allowed
        # to exist. It also carries prompt-alignment scoring, so a
        # prompt-bearing review can never silently lose its score.
        if semantic_embedder is None:
            raise BackendUnavailable(
                "qwen-vl critic requires the semantic (OpenCLIP) backend for "
                "its grounding gate and prompt-alignment scoring; provision "
                "the OpenCLIP weights or disable the critic")
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

        # 2. GROUND — deterministic anti-fabrication gate (embedder is a
        # constructor-enforced hard dependency; this can never be skipped)
        grounding_cos = check_grounding(self._embedder, description, img,
                                        self._grounding_min_cos)

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
            # Localizable requires BOTH an inherently spatial finding type
            # AND a successful model-emitted bbox from the LOCALIZE step.
            # No region-rectangle fallbacks: a finding we cannot genuinely
            # ground becomes GLOBAL (the region, if any, stays as text in
            # the description) rather than carrying invented geometry —
            # per the ticket's no-fake-masks stop condition.
            bbox = None
            if defect["type"] in _LOCALIZABLE_TYPES and region != "whole-image":
                # 4. LOCALIZE (grammar-constrained bbox for this defect)
                bbox = self._localize(uri, defect)
            localizable = bbox is not None
            description = str(defect.get("what", ""))[:300] or defect["type"]
            if not localizable and region != "whole-image":
                description = f"{description} (reported region: {region})"
            finding = {
                "type": defect["type"],
                "severity": defect["severity"],
                # Grammar constrains structure, not numeric ranges; an
                # out-of-range confidence is the model failing the protocol
                # and is REJECTED downstream by validate_review_payload —
                # never clamped into plausibility.
                "confidence": defect.get("confidence", 0.5),
                "localizable": localizable,
                "description": description,
            }
            if localizable:
                finding["bbox"] = list(bbox)
            findings.append(finding)

        # 5. ASSEMBLE (deterministic; alignment computed outside the VLM).
        # The embedder is mandatory, so a prompt-bearing review always gets
        # a real alignment score; None means only "no prompt available".
        alignment = None
        if prompt_text:
            alignment = clip_score_10(
                _cos(self._embedder.embed_text(prompt_text),
                     self._embedder.embed_image(img)))

        summary = f"{description[:280]} [grounding cos {grounding_cos:.2f}]"

        return {
            # Not clamped: an out-of-range score is protocol failure and is
            # rejected by validate_review_payload (the review then errors
            # rather than storing a laundered number).
            "quality_score": assess.get("quality_score"),
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
            x, y, w, h = (float(b[k]) for k in ("x", "y", "w", "h"))
            # Strict geometric validity — a box the model cannot state
            # coherently is a failed localization, never repaired for it.
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
                    and 0.01 <= w <= 1.0 - x + 1e-6
                    and 0.01 <= h <= 1.0 - y + 1e-6):
                return None
            return (x, y, min(w, 1.0 - x), min(h, 1.0 - y))
        except Exception:  # noqa: BLE001 - failed localization -> global finding
            return None


