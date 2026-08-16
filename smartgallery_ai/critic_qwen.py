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
  3. ASSESS    — one JSON-schema-constrained call: quality score + every
                 defect the model can actually see (type from the fixed
                 vocabulary, severity, confidence, coarse region enum;
                 deliberately no numeric cap in prompt or schema — a cap
                 anchors the model into inventing exactly that many).
  4. LOCALIZE  — only for defects whose type is inherently spatial and
                 whose region is not whole-image: one schema-constrained
                 bbox call. Localizable requires a geometrically valid
                 model-emitted box; otherwise the finding stays GLOBAL.
                 There are no region-rectangle fallbacks.
  4.5 ALIGN    — when a prompt exists: expected elements are extracted
                 DETERMINISTICALLY from the prompt (verbatim slices;
                 negative-prompt terms excluded), and one fixed-length
                 schema call answers present/absent/where per element,
                 with the step-1 caption supplied as context. Located
                 satisfied elements get a bbox (crop-verified, same rule
                 as findings) so the panel can highlight what landed;
                 confidently-absent elements also become prompt_mismatch
                 findings. The VLM never chooses what to expect.
  5. ASSEMBLE  — deterministic payload assembly; prompt_alignment_score is
                 satisfied/total over the ALIGN elements, on 0..1 — the
                 fraction of the user's own prompt the image delivered,
                 explained element by element rather than asserted as a
                 similarity number. The payload still goes through
                 validate_review_payload.

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
import re
from typing import Optional

import numpy as np
from PIL import Image

from smartgallery_ai import _env_num
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

_ALIGN_MAX_ELEMENTS = 12


def extract_prompt_elements(prompt: Optional[str], negative: Optional[str] = None) -> list:
    """Deterministic expected-element extraction for the ALIGN step — the
    expected-text guard: every element is a verbatim-derived slice of the
    actual prompt (weight/lora syntax stripped; comma/newline/BREAK
    segmentation), and any element whose text also appears in the
    negative prompt is excluded (it was requested ABSENT). The VLM never
    chooses what to expect."""
    text = re.sub(r"<[^>]*>", " ", prompt or "")            # lora tags
    text = re.sub(r":\d+(?:\.\d+)?", " ", text)             # :1.2 weights
    text = re.sub(r"[()\[\]{}]", " ", text)                 # weight brackets
    neg = " ".join((negative or "").lower().split())
    seen: set = set()
    elements: list = []
    for seg in re.split(r"[,\n]|\bBREAK\b", text):
        seg = " ".join(seg.split()).strip(" .;:-")
        low = seg.lower()
        if len(low) < 3 or low in seen:
            continue
        if neg and low in neg:
            continue
        seen.add(low)
        elements.append(seg)
        if len(elements) >= _ALIGN_MAX_ELEMENTS:
            break
    return elements


def _align_schema(n: int) -> dict:
    """Grammar schema for the ALIGN reply: one verdict per element, in
    order, fixed length. `where` is the coarse region the element was found
    in -- 'absent' when it is not there, 'whole-image' for properties with
    no locus (style, mood, lighting) -- and drives the per-element
    localization that the panel highlights."""
    return {
        "type": "object",
        "properties": {
            "elements": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": {
                    "type": "object",
                    "properties": {
                        "present": {"type": "boolean"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "where": {"enum": ["absent", *_REGIONS]},
                    },
                    "required": ["present", "confidence", "where"],
                },
            },
        },
        "required": ["elements"],
    }


# JSON schema enforced (via llama.cpp grammar decoding) on the ASSESS reply.
_ASSESS_SCHEMA = {
    "type": "object",
    "properties": {
        "quality_score": {"type": "number"},
        "defects": {
            "type": "array",
            # Unbounded on purpose: a numeric cap in the schema (like a
            # number in the prompt) anchors the model into producing
            # exactly that many. An honest empty list must cost nothing.
            "maxItems": 16,
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


class QwenVlCritic(CriticBackend):
    """The decomposed Qwen2.5-VL critic (protocol in the module docstring).
    Exists only with a semantic embedder attached: the grounding gate and
    the per-region verification behind every bbox it emits are inseparable
    from the critic."""

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    model_version = "qwen2.5-vl-7b-q4_k_m+decomposed-v2"  # refined per provisioned quantization in __init__

    def __init__(self, models_dir: str,
                 semantic_embedder: Optional[SemanticEmbedder] = None,
                 grounding_min_cos: float = DEFAULT_GROUNDING_MIN_COS,
                 n_ctx: int = 8192, n_threads: int = 4):
        """Raises `BackendUnavailable` unless the embedder, the GGUF
        weights (model + mmproj), and the llama.cpp runtime are all
        present and loadable."""
        # The CLIP gate is a hard dependency: this critic must not exist in
        # a gate-less configuration, and the embedder also verifies every
        # region it claims (findings and located prompt elements alike).
        if semantic_embedder is None:
            raise BackendUnavailable(
                "qwen-vl critic requires the semantic (OpenCLIP) backend for "
                "its grounding gate and region verification; provision "
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
            f"+decomposed-v2")

        try:
            from smartgallery_ai.llama_runtime import (
                activate_llama_backends, prepare_llama_runtime)
            prepare_llama_runtime()
            import llama_cpp
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
            activate_llama_backends()
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
            gpu_kwargs["n_gpu_layers"] = _env_num("AI_DAM_GPU_LAYERS", -1)
            if device.startswith("cuda:"):
                gpu_kwargs["main_gpu"] = int(device.split(":", 1)[1])
        split = os.environ.get("AI_DAM_TENSOR_SPLIT", "").strip()
        if split and device != "cpu":
            gpu_kwargs["tensor_split"] = [float(p) for p in split.split(",")]
        # GPU vision encode without flash attention crashes on images
        # larger than the vendored warmup reservation (2116 image tokens):
        # llama.cpp's grow-on-bigger-graph path overflows a compute-buffer
        # chunk (GGML_ASSERT ggml-backend.cpp:2000; under VRAM pressure a
        # failed re-reserve is ignored at ggml-backend.cpp:1531 and the
        # null buffer deref surfaces as "access violation reading 0x0").
        # With flash attention the same grow path succeeds across the full
        # legal 4096-token range, so the vision context runs on GPU with
        # FA enabled. Upstream's handler hardcodes use_gpu=True and only
        # inherits FA from the text model, hence this override.
        # AI_DAM_VISION_GPU=0 / AI_DAM_VISION_FA=0 opt out.
        vision_gpu = os.environ.get("AI_DAM_VISION_GPU", "1") == "1"
        vision_fa = os.environ.get("AI_DAM_VISION_FA", "1") == "1"

        def _init_mtmd_cpu_capable(handler_self, llama_model):
            if handler_self.mtmd_ctx is not None:
                return
            mtmd_cpp = handler_self._mtmd_cpp
            ctx_params = mtmd_cpp.mtmd_context_params_default()
            ctx_params.use_gpu = vision_gpu
            ctx_params.print_timings = handler_self.verbose
            ctx_params.n_threads = llama_model.n_threads
            ctx_params.flash_attn_type = (
                llama_cpp.LLAMA_FLASH_ATTN_TYPE_ENABLED if vision_fa
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
               _rubric_version: str, negative_text: Optional[str] = None) -> dict:
        """Run the DESCRIBE/GROUND/ASSESS/ALIGN/LOCALIZE/ASSEMBLE protocol
        and return the RAW payload dict for `validate_review_payload`.
        Raises `CriticGroundingError` when the description fails the gate
        -- nothing is stored for that image."""
        img = img.convert("RGB")
        # Keep the vision token budget bounded and deterministic.
        if max(img.size) > 768:
            img = img.copy()
            img.thumbnail((768, 768), Image.LANCZOS)
        uri = _data_uri(img)

        # 1. DESCRIBE (free text; nothing schema-shaped to parrot)
        self._emit("describe")
        description = self._chat(
            uri, "Describe this image factually in two short sentences.",
            schema=None, max_tokens=120).strip()

        # 2. GROUND — deterministic anti-fabrication gate (embedder is a
        # constructor-enforced hard dependency; this can never be skipped)
        self._emit("ground", description=description)
        grounding_margin = check_grounding(self._embedder, description, img,
                                        self._grounding_min_cos)

        # 3. ASSESS (grammar-constrained)
        self._emit("assess", grounding_margin=round(grounding_margin, 3))
        assess_raw = self._chat(
            uri,
            "You are reviewing an AI-generated image for defects. "
            "Report overall technical quality 0-10 and list every concrete "
            "defect you can actually see — the list may well be empty. For "
            "each defect give its category, severity, your confidence 0-1, "
            "the region where it is, and a few words describing it.",
            schema=_ASSESS_SCHEMA, max_tokens=900)
        assess = json.loads(assess_raw)

        findings = []
        dropped_unverified = 0
        for defect in (assess.get("defects") or []):
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

        # 4.5 ALIGN — its own pass over the prompt, in its own context.
        # Elements come from the deterministic expected-text guard
        # (verbatim prompt slices, negative-prompt terms excluded), so a
        # verdict can never be about content the prompt did not request.
        # The step-1 caption rides along as context: the model has already
        # committed to what it sees, and judging the prompt against that
        # commitment is harder to talk itself out of than judging the
        # prompt against the pixels a second time.
        alignment_elements: list = []
        adherence_note = ""
        if prompt_text:
            self._emit("align", findings=len(findings))
            expected = extract_prompt_elements(prompt_text, negative_text)
            if expected:
                listing = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(expected))
                try:
                    align_raw = self._chat(
                        uri,
                        f"You already described this image as: {description}\n\n"
                        "It was generated from a prompt requesting the "
                        "following elements:\n" + listing + "\n"
                        "For each element, in order, say whether it is "
                        "visibly present in the image, your confidence 0-1, "
                        "and where it is — a region name, or 'absent' when "
                        "it is not there, or 'whole-image' when it is an "
                        "overall property rather than a thing in one place.",
                        schema=_align_schema(len(expected)),
                        max_tokens=40 * len(expected) + 40)
                    verdicts = json.loads(align_raw).get("elements") or []
                except Exception:
                    verdicts = []  # ALIGN failure never sinks the review
                if len(verdicts) == len(expected):
                    for ordinal, (text, verdict) in enumerate(zip(expected, verdicts)):
                        satisfied = bool(verdict.get("present"))
                        confidence = verdict.get("confidence", 0.5)
                        where = verdict.get("where", "whole-image")
                        # A satisfied element with a real region gets one
                        # localization attempt so the panel can highlight
                        # exactly what landed. No box -> no highlight; a
                        # rectangle invented here would be a confident
                        # claim the model never made.
                        bbox = None
                        if satisfied and where not in ("absent", "whole-image"):
                            bbox = self._localize(
                                uri, {"type": "other", "what": text})
                            if bbox is not None and not verify_finding_region(
                                    self._embedder, text, bbox, img):
                                bbox = None
                        element = {
                            "ordinal": ordinal,
                            "text": text,
                            "satisfied": satisfied,
                            # Not clamped: validate_review_payload rejects
                            # out-of-range values.
                            "confidence": confidence,
                        }
                        if bbox is not None:
                            element["bbox"] = list(bbox)
                        alignment_elements.append(element)
                        # A confidently-absent element is also a defect, so
                        # it stays in the findings list the reviewer reads.
                        if not satisfied and float(confidence or 0) >= 0.5:
                            findings.append({
                                "type": "prompt_mismatch",
                                "severity": "medium",
                                "confidence": confidence,
                                "localizable": False,
                                "description": f'requested "{text[:120]}" is not visible',
                            })
                    satisfied_count = sum(
                        1 for e in alignment_elements if e["satisfied"])
                    adherence_note = (f" [adherence {satisfied_count}"
                                      f"/{len(alignment_elements)} prompt elements]")

        # 5. ASSEMBLE (deterministic). Prompt-following is the fraction of
        # the user's own prompt the image actually delivered — countable,
        # and explained element by element by the rows above. None means
        # only "this file carries no generation prompt to follow".
        alignment_score = None
        if alignment_elements:
            alignment_score = (sum(1 for e in alignment_elements if e["satisfied"])
                               / len(alignment_elements))

        self._emit("assemble", findings=len(findings),
                   alignment=len(alignment_elements),
                   alignment_score=alignment_score)
        summary = f"{description[:260]} [grounding margin {grounding_margin:.2f}]"
        if dropped_unverified:
            summary += (f" [{dropped_unverified} finding(s) dropped: region "
                        f"verification failed]")
        summary += adherence_note

        return {
            # Not clamped: validate_review_payload rejects out-of-range
            # values.
            "quality_score": assess.get("quality_score"),
            "prompt_alignment_score": alignment_score,
            "summary": summary,
            "findings": findings,
            "alignment": alignment_elements,
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


