"""Decomposed local VLM critic: Qwen2.5-VL-7B (Apache-2.0) via llama.cpp.

The model only ever answers small, grammar-constrained questions;
deterministic code assembles the typed review. Design rationale and the
measurement record live in docs/AI_MODELS.md.

Protocol per image:
  1. DESCRIBE  — short free-text factual description.
  2. GROUND    — contrastive gate: the description must clear an absolute
                 CLIP-cosine floor AND beat a generic baseline text on the
                 same image by a margin, else CriticGroundingError. This
                 filters the description stage only; per-finding
                 verification happens in step 4/5.
  3. ASSESS    — one JSON-schema-constrained call: quality score + up to
                 3 defects (type from the fixed vocabulary, severity,
                 confidence, coarse region enum).
  4. LOCALIZE  — only for defects whose type is inherently spatial and
                 whose region is not whole-image: one schema-constrained
                 bbox call. Localizable requires a geometrically valid
                 model-emitted box; otherwise the finding stays GLOBAL.
                 There are no region-rectangle fallbacks.
  5. ASSEMBLE  — deterministic payload assembly; prompt_alignment_score is
                 CLIPScore(prompt, image) mapped to 0-10
                 (min(10, 25*max(cos,0))), computed outside the VLM. The
                 payload still goes through validate_review_payload.

Weights load ONLY from the models dir (this module never downloads;
provisioning is smartgallery_ai.provision's job):
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

# Contrastive gate thresholds. Chosen from the sweep in
# benchmarks/results/grounding_calibration.json (written by
# probes/grounding_calibration.py); changing them without re-running that
# probe invalidates the 'auto' enablement check in review.py.
DEFAULT_GROUNDING_MIN_COS = 0.20
DEFAULT_GROUNDING_MIN_MARGIN = 0.09
GROUNDING_BASELINE_TEXT = "an image with some shapes and colors"

# Per-finding topical verification: a localizable finding's description
# must beat the baseline on its own bbox crop by this margin. Verifies the
# named defect is visually present in the claimed region, not subjective
# quality judgments.
DEFAULT_FINDING_MIN_MARGIN = 0.0

# Coarse region vocabulary for the ASSESS step; any value other than
# 'whole-image' invites a LOCALIZE attempt for spatial finding types.
_REGIONS = ("whole-image", "top-left", "top-right", "bottom-left",
            "bottom-right", "center")

# Only finding types with an inherent spatial locus may be localizable;
# lighting/style/composition/prompt_mismatch are whole-image properties
# and must never carry a bbox or mask.
_LOCALIZABLE_TYPES = frozenset(
    {"anatomy", "artifact", "text_render", "detail_loss", "other"})

# JSON schema enforced (via llama.cpp grammar decoding) on the ASSESS reply.
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

# JSON schema enforced on each LOCALIZE reply; geometric validity is
# checked separately in `_localize`.
_BBOX_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": "number"}, "y": {"type": "number"},
        "w": {"type": "number"}, "h": {"type": "number"},
    },
    "required": ["x", "y", "w", "h"],
}


def _first_existing(models_dir: str, names: tuple) -> Optional[str]:
    """First provisioned file among `names` under `models_dir` (tuple order
    encodes preference), or None when none exists."""
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
                    min_cos: float = DEFAULT_GROUNDING_MIN_COS,
                    min_margin: float = DEFAULT_GROUNDING_MIN_MARGIN) -> float:
    """The deterministic contrastive anti-fabrication gate, exposed as a
    pure function so its negative cases are testable without loading the
    VLM. Requires BOTH an absolute cosine floor AND a positive margin
    over the generic-baseline text on the same image. Returns the margin on
    success; raises CriticGroundingError otherwise. All comparisons fail
    CLOSED on NaN (`not (x >= t)` instead of `x < t`)."""
    if not description:
        raise CriticGroundingError("critic produced no description")
    iv = embedder.embed_image(img)
    cos = _cos(embedder.embed_text(description), iv)
    if not (cos >= min_cos):
        raise CriticGroundingError(
            f"description does not match image (CLIP cos {cos:.3f} < "
            f"{min_cos}); refusing to store an ungrounded review")
    margin = cos - _cos(embedder.embed_text(GROUNDING_BASELINE_TEXT), iv)
    if not (margin >= min_margin):
        raise CriticGroundingError(
            f"description is not specific to this image (margin {margin:.3f}"
            f" < {min_margin} over the generic baseline); vacuous or "
            f"copied descriptions are rejected")
    return margin


def verify_finding_region(embedder: SemanticEmbedder, description: str,
                          bbox: tuple, img: Image.Image,
                          min_margin: float = DEFAULT_FINDING_MIN_MARGIN) -> bool:
    """Topical grounding check for one localizable finding: does the named
    defect/content actually appear in the claimed region? Crops the bbox
    (with 10% padding), and requires the finding text to beat the generic
    baseline on that crop. Returns False for findings naming things that
    are not visually there (wrong region, invented object). Cannot judge
    subjective quality claims — that scope is documented."""
    w, h = img.size
    x, y, bw, bh = bbox
    pad_x, pad_y = 0.1 * bw, 0.1 * bh
    box = (max(0, int((x - pad_x) * w)), max(0, int((y - pad_y) * h)),
           min(w, int((x + bw + pad_x) * w)), min(h, int((y + bh + pad_y) * h)))
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        return False
    crop = img.crop(box)
    if max(crop.size) < 224:
        crop = crop.resize((224, 224), Image.LANCZOS)
    cv = embedder.embed_image(crop)
    margin = (_cos(embedder.embed_text(description), cv)
              - _cos(embedder.embed_text(GROUNDING_BASELINE_TEXT), cv))
    return bool(margin >= min_margin)


def _data_uri(img: Image.Image) -> str:
    """Encode the image as a PNG data: URI -- the image form llama.cpp's
    chat handler accepts without touching the filesystem or network."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity; normalizes both vectors, so inputs need not be
    unit length."""
    return float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))


def clip_score_10(cos: float) -> float:
    """CLIPScore-style mapping (w=2.5) onto the 0-10 UI scale."""
    return min(10.0, 25.0 * max(cos, 0.0))


class QwenVlCritic(CriticBackend):
    """The decomposed Qwen2.5-VL critic (protocol in the module docstring).
    Exists only with a semantic embedder attached: the grounding gate and
    prompt-alignment scoring are inseparable from the critic."""

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_version = "qwen2.5-vl-7b-q4_k_m+decomposed-v1"  # refined per provisioned quantization in __init__

    def __init__(self, models_dir: str,
                 semantic_embedder: Optional[SemanticEmbedder] = None,
                 grounding_min_cos: float = DEFAULT_GROUNDING_MIN_COS,
                 n_ctx: int = 8192, n_threads: int = 4):
        """Raises `BackendUnavailable` unless the embedder, the GGUF
        weights (model + mmproj), and the llama.cpp runtime are all
        present and loadable."""
        # The CLIP gate is a hard dependency: this critic must not exist in
        # a gate-less configuration, and the embedder also carries
        # prompt-alignment scoring.
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
            from smartgallery_ai.llama_runtime import prepare_llama_runtime
            prepare_llama_runtime()
            import llama_cpp
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        except Exception as exc:
            raise BackendUnavailable(f"qwen-vl critic unavailable: {exc}") from exc

        # llama.cpp logs its internals -- including FULL PROMPTS and
        # per-image encode timings -- to the console via its native log
        # callback, ignoring verbose=False on newer multimodal builds
        # (observed live). Install a no-op callback, keeping a reference
        # on self so ctypes never garbage-collects it mid-call.
        # AI_DAM_LLAMA_VERBOSE=1 keeps the native logs for debugging.
        if not os.environ.get("AI_DAM_LLAMA_VERBOSE"):
            try:
                self._llama_log_cb = llama_cpp.llama_log_callback(
                    lambda _level, _text, _user_data: None)
                llama_cpp.llama_log_set(self._llama_log_cb, None)
            except Exception:  # silencing is best-effort
                pass
        # Full GPU offload by default when the llama.cpp build has CUDA
        # support; a CPU-only build ignores every GPU knob, so this is safe
        # everywhere. AI_DAM_DEVICE=cpu forces 0 layers; =cuda:N pins the
        # primary card; AI_DAM_GPU_LAYERS tunes partial offload for
        # VRAM-constrained cards; AI_DAM_TENSOR_SPLIT ("0.6,0.4") sets the
        # per-GPU proportions (llama.cpp already layer-splits across all
        # visible GPUs by default when several are present).
        device = os.environ.get("AI_DAM_DEVICE", "").lower()
        gpu_kwargs: dict = {}
        if device == "cpu":
            gpu_kwargs["n_gpu_layers"] = 0
        else:
            gpu_kwargs["n_gpu_layers"] = int(os.environ.get("AI_DAM_GPU_LAYERS", "-1"))
            if device.startswith("cuda:"):
                gpu_kwargs["main_gpu"] = int(device.split(":", 1)[1])
        split = os.environ.get("AI_DAM_TENSOR_SPLIT", "").strip()
        if split and device != "cpu":
            gpu_kwargs["tensor_split"] = [float(p) for p in split.split(",")]
        # Upstream hardcodes the mtmd vision encoder onto GPU
        # (llama_chat_format.py Llava15ChatHandler._init_mtmd_context:
        # `ctx_params.use_gpu = True  # TODO: Make this configurable`).
        # On this CUDA-13 build, GPU image-slice encoding faults with a
        # null read ("access violation reading 0x0" during
        # "encoding image slice..."), killing every review. Subclass to
        # make it configurable; vision preprocessing runs on CPU unless
        # AI_DAM_VISION_GPU=1. Text generation stays fully on GPU either
        # way.
        vision_gpu = os.environ.get("AI_DAM_VISION_GPU", "0") == "1"

        def _init_mtmd_cpu_capable(handler_self, llama_model):
            if handler_self.mtmd_ctx is not None:
                return
            mtmd_cpp = handler_self._mtmd_cpp
            ctx_params = mtmd_cpp.mtmd_context_params_default()
            ctx_params.use_gpu = vision_gpu
            ctx_params.print_timings = handler_self.verbose
            ctx_params.n_threads = llama_model.n_threads
            ctx_params.flash_attn_type = (
                llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED
                if (llama_model.context_params.flash_attn_type
                    == llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED)
                else llama_cpp.LLAMA_FLASH_ATTN_TYPE_DISABLED
            )
            handler_self.mtmd_ctx = mtmd_cpp.mtmd_init_from_file(
                handler_self.clip_model_path.encode(), llama_model.model, ctx_params
            )
            if handler_self.mtmd_ctx is None:
                raise ValueError(
                    f"Failed to load mtmd context from: {handler_self.clip_model_path}"
                )
            if not mtmd_cpp.mtmd_support_vision(handler_self.mtmd_ctx):
                raise ValueError("Vision is not supported by this model")

            def mtmd_free():
                if handler_self.mtmd_ctx is not None:
                    mtmd_cpp.mtmd_free(handler_self.mtmd_ctx)
                    handler_self.mtmd_ctx = None

            handler_self._exit_stack.callback(mtmd_free)

        try:
            handler = Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=False)
            # Bound per-instance so test fakes (MagicMock handlers) are
            # unaffected; only a real handler carries these attributes.
            if hasattr(handler, "_mtmd_cpp"):
                import types as _types

                handler._init_mtmd_context = _types.MethodType(
                    _init_mtmd_cpu_capable, handler
                )
            self._llm = Llama(model_path=model_path, chat_handler=handler,
                              n_ctx=n_ctx, n_threads=n_threads,
                              verbose=False, **gpu_kwargs)
        except Exception as exc:
            raise BackendUnavailable(f"failed to load qwen-vl weights: {exc}") from exc

        self._embedder = semantic_embedder
        self._grounding_min_cos = grounding_min_cos

    # -- constrained single-turn helpers ------------------------------------

    def _chat(self, img_uri: str, text: str, schema: Optional[dict],
              max_tokens: int) -> str:
        """One greedy-decoded single-turn image+text call. A non-None
        `schema` constrains decoding to that JSON shape via llama.cpp's
        grammar support; returns the raw reply text ('' when the model
        emits nothing)."""
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
               _rubric_version: str) -> dict:
        """Run the DESCRIBE/GROUND/ASSESS/LOCALIZE/ASSEMBLE protocol and
        return the RAW payload dict for `validate_review_payload`. Raises
        `CriticGroundingError` when the description fails the gate --
        nothing is stored for that image."""
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
        grounding_margin = check_grounding(self._embedder, description, img,
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
        dropped_unverified = 0
        for defect in (assess.get("defects") or [])[:3]:
            region = defect.get("region", "whole-image")
            # Localizable requires all of: an inherently spatial finding
            # type, a valid model-emitted bbox, and passing crop
            # verification. No fallbacks: without a genuine locus the
            # finding stays GLOBAL; failing verification drops it.
            # Kept separate from `description` — the summary must quote the
            # step-1 description the grounding margin was computed for,
            # never per-defect text.
            finding_description = str(defect.get("what", ""))[:300] or defect["type"]
            bbox = None
            if defect["type"] in _LOCALIZABLE_TYPES and region != "whole-image":
                # 4. LOCALIZE (grammar-constrained bbox for this defect)
                bbox = self._localize(uri, defect)
                if bbox is not None and not verify_finding_region(
                        self._embedder, finding_description, bbox, img):
                    dropped_unverified += 1
                    continue
            localizable = bbox is not None
            if not localizable and region != "whole-image":
                finding_description = f"{finding_description} (reported region: {region})"
            finding = {
                "type": defect["type"],
                "severity": defect["severity"],
                # Not clamped: validate_review_payload rejects out-of-range
                # values.
                "confidence": defect.get("confidence", 0.5),
                "localizable": localizable,
                "description": finding_description,
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

        summary = f"{description[:260]} [grounding margin {grounding_margin:.2f}]"
        if dropped_unverified:
            summary += (f" [{dropped_unverified} finding(s) dropped: region "
                        f"verification failed]")

        return {
            # Not clamped: validate_review_payload rejects out-of-range
            # values.
            "quality_score": assess.get("quality_score"),
            "prompt_alignment_score": alignment,
            "summary": summary,
            "findings": findings,
        }

    def _localize(self, uri: str, defect: dict) -> Optional[tuple]:
        """Ask for one defect's bounding box; returns a normalized
        (x, y, w, h) clipped to the frame, or None when the model cannot
        state a geometrically coherent box (the finding then stays
        global)."""
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
        except Exception:  # failed localization -> global finding
            return None


