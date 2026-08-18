"""Decomposed image review over any transformers vision-language model.

The model only ever answers small, narrow questions; deterministic code
assembles the typed review. Design rationale and the measurement record
live in docs/AI_MODELS.md.

Protocol per image, all of it inside ONE `ai_models.Chat` so the image is
encoded once and every later step reuses its keys and values:
  1. DESCRIBE  - a short factual description of what is there.
  2. GROUND    - contrastive gate: the description must clear an absolute
                 CLIP-cosine floor AND beat a generic baseline text on the
                 same image by a margin, else UngroundedReviewError. This
                 filters the description stage only; per-finding
                 verification happens in steps 4 and 4.5.
  3. ASSESS    - one `assess` tool call: quality score plus every defect
                 the model can actually see (type from the fixed
                 vocabulary, severity, confidence, coarse region enum;
                 deliberately no numeric cap in the prompt - a cap anchors
                 the model into inventing exactly that many).
  4. LOCALIZE  - only for defects whose type is inherently spatial and
                 whose region is not whole-image: one `locate` call.
                 Localizable requires a geometrically valid model-emitted
                 box that also passes crop verification. There are no
                 region-rectangle fallbacks.
  4.5 ALIGN    - when a prompt exists: expected elements are extracted
                 DETERMINISTICALLY from the prompt (verbatim slices;
                 negative-prompt terms excluded), and one fixed-length
                 `align` call answers present/absent/where per element,
                 with the step-1 caption supplied as context. Located
                 satisfied elements get a crop-verified bbox so the panel
                 can highlight what landed; confidently-absent elements
                 also become prompt_mismatch findings. The model never
                 chooses what to expect.
  5. ASSEMBLE  - deterministic assembly; prompt_alignment_score is
                 satisfied/total over the ALIGN elements, on 0..1 - the
                 fraction of the user's own prompt the image delivered,
                 explained element by element rather than asserted as a
                 similarity number. The payload still goes through
                 validate_review_payload.

Every structured step goes through the model's OWN tool-calling contract
(see smartgallery_ai.models): the chat template renders the schemas into
<tools></tools> and asks for <tool_call> replies. EVERY tool is declared
up front, because they render into the system block at position zero and
that is what the KV cache is built on -- which is also why DESCRIBE is a
tool rather than free text. A model told it may call functions answers the
first turn with a call; observed live, a plain-sentence request came back
as a <tool_call> for a different tool, and that text then became the
"description" the grounding gate scored.

Nothing here constrains decoding to a schema - transformers has no such
facility. Replies are therefore read DEFENSIVELY: a defect missing a
required key is skipped, not indexed into. `validate_review_payload`
remains the only gate into the database.

There is deliberately no per-model class. Every checkpoint loads through
the same auto-classes, so swapping Qwen3-VL for Phi-3.5-vision, SmolVLM or
Gemma is a configuration string, not a new class, a weights-filename
table, or a chat handler.
"""

from __future__ import annotations

import contextlib
import logging
import re

import numpy as np
from PIL import Image

from smartgallery_ai import models as ai_models
from smartgallery_ai.embedders import BackendUnavailable, SemanticEmbedder
from smartgallery_ai.review import FINDING_TYPES

_logger = logging.getLogger(__name__)

#: Default vision-language model. Any transformers image-text-to-text
#: checkpoint works -- this is a configuration value, not a code
#: dependency. Override with AI_DAM_CRITIC_MODEL.
DEFAULT_REVIEW_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

#: Bumping this invalidates every stored review (see invalidation.py).
PROTOCOL_VERSION = "decomposed-v2"

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
_REGIONS = ("whole-image", "top-left", "top-right", "bottom-left", "bottom-right", "center")

# Only finding types with an inherent spatial locus may be localizable;
# lighting/style/composition/prompt_mismatch are whole-image properties
# and must never carry a bbox or mask.
_LOCALIZABLE_TYPES = frozenset({"anatomy", "artifact", "text_render", "detail_loss", "other"})

# Upper bound on ALIGN elements. Each one costs a schema slot and tokens in
# a single fixed-length reply, so an unbounded prompt cannot be allowed to
# set the reply size. When it bites, `extract_prompt_elements_report` says
# so -- the score is satisfied/total, and a silently shortened total is a
# score computed over a set the user never saw.
_ALIGN_MAX_ELEMENTS = 24

_SEVERITIES = ("low", "medium", "high")

# Segment separators. Split by INDEX so each element can be sliced out of
# the ORIGINAL string rather than out of a rewritten copy.
_SEGMENT_RE = re.compile(r",|\n|\bBREAK\b")

# Comparison-only normalization: lora tags, weights, weight brackets and
# runs of whitespace. Used to dedupe and to test negative-prompt membership.
# NEVER used to produce the text handed to the model or stored in the DB --
# that text is always the user's own.
_NORM_STRIP = (
    re.compile(r"<[^>]*>"),  # <lora:name:0.8>
    re.compile(r":\d+(?:\.\d+)?"),  # :1.2
    re.compile(r"[()\[\]{}]"),  # weight brackets
)


class UngroundedReviewError(RuntimeError):
    """The description failed the CLIP grounding gate: the model is not
    talking about this image. The review is aborted, never stored."""


# --- deterministic prompt handling -------------------------------------------


def _norm_for_match(text: str) -> str:
    """Normalized form of one element, for dedupe and negative-prompt
    matching ONLY. Two spellings of the same ask should not both be
    scored, and `(blurry:1.3)` in the positive should still be recognised
    as the `blurry` the negative excluded."""
    for pattern in _NORM_STRIP:
        text = pattern.sub(" ", text)
    return " ".join(text.lower().split()).strip(" .;:-")


def extract_prompt_elements_report(prompt: str | None, negative: str | None = None) -> tuple:
    """`(elements, truncated)` for the ALIGN step.

    Every element is a TRUE substring of `prompt` -- sliced out by index
    and stripped only of surrounding whitespace. The model is shown, and
    the database stores, exactly what the user wrote: `(masterpiece:1.4)`
    stays `(masterpiece:1.4)`. Rewriting the ask before scoring adherence
    to it means scoring adherence to a prompt nobody issued, and it made
    the panel display text that appeared nowhere in the user's own prompt.

    Dropped only when a segment has no alphanumeric content at all (pure
    punctuation), when its normalized form repeats, or when the negative
    prompt asked for it to be ABSENT. Short asks like `8k` are kept: the
    old three-character floor silently discarded them, which also silently
    shrank the denominator of the adherence score.

    `truncated` is True when `_ALIGN_MAX_ELEMENTS` bit, so the caller can
    say so instead of reporting a fraction of a set it quietly shortened.
    """
    raw = prompt or ""
    neg = _norm_for_match(negative or "")
    seen: set = set()
    elements: list = []
    truncated = False

    start = 0
    bounds = []
    for match in _SEGMENT_RE.finditer(raw):
        bounds.append((start, match.start()))
        start = match.end()
    bounds.append((start, len(raw)))

    for begin, end in bounds:
        segment = raw[begin:end].strip()
        if not segment or not any(ch.isalnum() for ch in segment):
            continue
        key = _norm_for_match(segment)
        if not key or key in seen:
            continue
        if neg and key in neg:
            continue
        seen.add(key)
        if len(elements) >= _ALIGN_MAX_ELEMENTS:
            truncated = True
            break
        elements.append(segment)
    return elements, truncated


def extract_prompt_elements(prompt: str | None, negative: str | None = None) -> list:
    """The elements alone; see `extract_prompt_elements_report`."""
    return extract_prompt_elements_report(prompt, negative)[0]


# --- the grounding gate ------------------------------------------------------


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity; normalizes both vectors, so inputs need not be
    unit length. `np.dot` on two 1-D arrays is the inner product and
    `np.linalg.norm` with no `ord` is the 2-norm."""
    return float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))


def check_grounding(
    embedder: SemanticEmbedder,
    description: str,
    img: Image.Image,
    min_cos: float = DEFAULT_GROUNDING_MIN_COS,
    min_margin: float = DEFAULT_GROUNDING_MIN_MARGIN,
) -> float:
    """The deterministic contrastive anti-fabrication gate, exposed as a
    pure function so its negative cases are testable without loading a
    vision-language model. Requires BOTH an absolute cosine floor AND a
    positive margin over the generic-baseline text on the same image.
    Returns the margin on success; raises UngroundedReviewError otherwise.
    All comparisons fail CLOSED on NaN (`not (x >= t)` instead of
    `x < t`)."""
    if not description:
        raise UngroundedReviewError("the model produced no description")
    iv = embedder.embed_image(img)
    cos = _cos(embedder.embed_text(description), iv)
    if not (cos >= min_cos):
        raise UngroundedReviewError(
            f"description does not match image (CLIP cos {cos:.3f} < {min_cos}); refusing to store an ungrounded review"
        )
    margin = cos - _cos(embedder.embed_text(GROUNDING_BASELINE_TEXT), iv)
    if not (margin >= min_margin):
        raise UngroundedReviewError(
            f"description is not specific to this image (margin {margin:.3f}"
            f" < {min_margin} over the generic baseline); vacuous or "
            f"copied descriptions are rejected"
        )
    return margin


def verify_finding_region(
    embedder: SemanticEmbedder,
    description: str,
    bbox: tuple,
    img: Image.Image,
    min_margin: float = DEFAULT_FINDING_MIN_MARGIN,
) -> bool:
    """Topical grounding check for one localizable finding: does the named
    defect/content actually appear in the claimed region? Crops the bbox
    (with 10% padding), and requires the finding text to beat the generic
    baseline on that crop. Returns False for findings naming things that
    are not visually there (wrong region, invented object). Cannot judge
    subjective quality claims -- that scope is documented."""
    w, h = img.size
    x, y, bw, bh = bbox
    pad_x, pad_y = 0.1 * bw, 0.1 * bh
    box = (
        max(0, int((x - pad_x) * w)),
        max(0, int((y - pad_y) * h)),
        min(w, int((x + bw + pad_x) * w)),
        min(h, int((y + bh + pad_y) * h)),
    )
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        return False
    crop = img.crop(box)
    if max(crop.size) < 224:
        crop = crop.resize((224, 224), Image.Resampling.LANCZOS)
    cv = embedder.embed_image(crop)
    margin = _cos(embedder.embed_text(description), cv) - _cos(embedder.embed_text(GROUNDING_BASELINE_TEXT), cv)
    return bool(margin >= min_margin)


# --- tool contracts ----------------------------------------------------------


def _describe_tool() -> dict:
    """The DESCRIBE contract.

    Describing is a tool call rather than free text because every tool is
    declared up front (they render into the system block the KV cache is
    built on), and a model told it may call functions answers the FIRST
    turn with one. Observed live: asking for a plain sentence returned a
    <tool_call> for `align`, which then became the "description" the
    grounding gate scored. Uniform turns, no special case.
    """
    return ai_models.tool(
        "describe",
        "State plainly what is visible in the image.",
        {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Two short factual sentences about what is in the image.",
                }
            },
            "required": ["description"],
        },
    )


def _assess_tool() -> dict:
    """The ASSESS contract. `defects` is capped only to bound the reply
    length; there is no minimum, because an honest empty list must cost
    nothing and a stated number anchors the model into producing it."""
    return ai_models.tool(
        "assess",
        "Report the technical quality of an AI-generated image and every concrete visible defect.",
        {
            "type": "object",
            "properties": {
                "quality_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 10,
                    "description": "Overall technical quality, 0-10.",
                },
                "defects": {
                    "type": "array",
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"enum": list(FINDING_TYPES)},
                            "severity": {"enum": list(_SEVERITIES)},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "region": {"enum": list(_REGIONS)},
                            "what": {"type": "string", "description": "A few words describing it."},
                        },
                        "required": ["type", "severity", "confidence", "region", "what"],
                    },
                },
            },
            "required": ["quality_score", "defects"],
        },
    )


def _locate_tool() -> dict:
    """The LOCALIZE contract: one box, as fractions of the image size, so
    the answer does not depend on how the processor scaled the image."""
    return ai_models.tool(
        "locate",
        "Give the bounding box of one thing in the image, as fractions of the image size.",
        {
            "type": "object",
            "properties": {
                "x": {"type": "number", "minimum": 0, "maximum": 1, "description": "Left edge, fraction of the width."},
                "y": {"type": "number", "minimum": 0, "maximum": 1, "description": "Top edge, fraction of the height."},
                "w": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Width, fraction of the image width.",
                },
                "h": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Height, fraction of the image height.",
                },
            },
            "required": ["x", "y", "w", "h"],
        },
    )


def _align_tool(n: int) -> dict:
    """The ALIGN contract: one verdict per expected element, in order,
    fixed length. `where` is the coarse region the element was found in --
    'absent' when it is not there, 'whole-image' for properties with no
    locus (style, mood, lighting) -- and drives the per-element
    localization that the panel highlights."""
    return ai_models.tool(
        "align",
        "Say whether each requested prompt element is visible in the image.",
        {
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
        },
    )


# --- defensive reads ---------------------------------------------------------
#
# Decoding is not schema-constrained, so every field the model produces is
# checked before use. The old llama.cpp path could index a reply directly
# because grammar decoding guaranteed its shape; nothing does that here.


def _as_defect(raw) -> dict | None:
    """One well-formed defect, or None. A malformed entry is dropped
    rather than repaired: an invented severity or type would be a claim
    the model never made."""
    if not isinstance(raw, dict):
        return None
    kind, severity = raw.get("type"), raw.get("severity")
    if kind not in FINDING_TYPES or severity not in _SEVERITIES:
        return None
    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        return None
    region = raw.get("region")
    return {
        "type": kind,
        "severity": severity,
        "confidence": confidence,
        "region": region if region in _REGIONS else "whole-image",
        "what": str(raw.get("what") or "")[:300] or str(kind),
    }


def _as_bbox(raw) -> tuple | None:
    """A geometrically coherent normalized (x, y, w, h), or None.

    A box the model cannot state coherently is a failed localization,
    never repaired into one."""
    if not isinstance(raw, dict):
        return None
    try:
        x, y, w, h = (float(raw[key]) for key in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError):
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.01 <= w <= 1.0 - x + 1e-6 and 0.01 <= h <= 1.0 - y + 1e-6):
        return None
    return (x, y, min(w, 1.0 - x), min(h, 1.0 - y))


def _as_verdict(raw) -> dict | None:
    """One well-formed ALIGN verdict, or None."""
    if not isinstance(raw, dict) or "present" not in raw:
        return None
    try:
        confidence = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    where = raw.get("where")
    return {
        "present": bool(raw["present"]),
        "confidence": confidence,
        "where": where if where in ("absent", *_REGIONS) else "whole-image",
    }


class Reviewer:
    """Runs the review protocol over one configured checkpoint.

    Exists only with a semantic embedder attached: the grounding gate and
    the per-region verification behind every bbox it emits are inseparable
    from the review.

    `model_id`/`model_version` are recorded as provenance with every stored
    review. `progress` is an optional live sink, `fn(stage, detail)`: the
    interactive runner installs one so a long review reports each stage as
    it lands, and background indexing leaves it None. Never load-bearing.
    """

    progress = None

    def __init__(
        self,
        models_dir: str,
        semantic_embedder: SemanticEmbedder | None = None,
        model_ref: str = DEFAULT_REVIEW_MODEL,
        grounding_min_cos: float = DEFAULT_GROUNDING_MIN_COS,
        device: str = "",
    ):
        """Raises `BackendUnavailable` unless the embedder, the weights and
        the transformers runtime are all present and loadable."""
        # The CLIP gate is a hard dependency: a review must not exist in a
        # gate-less configuration, and the embedder also verifies every
        # region it claims (findings and located prompt elements alike).
        if semantic_embedder is None:
            raise BackendUnavailable(
                "the reviewer requires the semantic (OpenCLIP) backend for "
                "its grounding gate and region verification; provision the "
                "OpenCLIP weights or disable reviews"
            )
        self.model_id = model_ref
        self.model_version = f"{model_ref}+{PROTOCOL_VERSION}"
        self._models_dir = models_dir
        self._device = device
        self._embedder = semantic_embedder
        self._grounding_min_cos = grounding_min_cos
        # Load now, not on first review: an unavailable checkpoint must be
        # reported when the capability is resolved, not thousands of files
        # into an indexing run.
        try:
            ai_models.load(model_ref, models_dir=models_dir, device=device)
        except ai_models.ModelUnavailable as exc:
            raise BackendUnavailable(str(exc)) from exc

    def _emit(self, stage: str, **detail) -> None:
        """Report one protocol stage to the progress sink, if any.

        Swallows sink failures on purpose: a disconnected SSE client or a
        slow consumer must never abort a review that is otherwise fine.
        Observation must not be able to break the thing observed."""
        sink = self.progress
        if sink is None:
            return
        with contextlib.suppress(Exception):
            sink(stage, detail)

    def review(
        self, img: Image.Image, prompt_text: str | None, rubric_version: str, negative_text: str | None = None
    ) -> dict:
        """Run the protocol and return the RAW payload dict for
        `validate_review_payload`. Raises `UngroundedReviewError` when the
        description fails the gate -- nothing is stored for that image.

        The image is NOT pre-resized: the processor's own vision budget
        bounds the token cost, and the grounding crops then run against the
        full-resolution original rather than a thumbnail of it.
        """
        del rubric_version  # provenance is model_version; the rubric is fixed
        img = img.convert("RGB")

        # Expected elements are known before the conversation starts, which
        # matters: the ALIGN contract is fixed-length, and every tool has to
        # be declared up front because they render into the system block the
        # KV cache is built on.
        expected, truncated = extract_prompt_elements_report(prompt_text, negative_text) if prompt_text else ([], False)
        tools = [_describe_tool(), _assess_tool(), _locate_tool()]
        if expected:
            tools.append(_align_tool(len(expected)))

        chat = ai_models.Chat(
            self.model_id,
            [img],
            models_dir=self._models_dir,
            device=self._device,
            tools=tools,
            system="You are a strict reviewer of AI-generated images. "
            "Answer only by calling the tool you were asked for.",
        )

        # 1. DESCRIBE -- what the model commits to seeing, before it is
        # asked to judge anything. Nothing schema-shaped to parrot: the
        # contract is one free-form string.
        self._emit("describe")
        try:
            described = chat.ask_json("Call describe for this image.", name="describe", max_new_tokens=160)
        except ValueError as exc:
            raise UngroundedReviewError(f"the model produced no description: {exc}") from exc
        description = str((described or {}).get("description", "")).strip()

        # 2. GROUND -- deterministic anti-fabrication gate (the embedder is
        # a constructor-enforced hard dependency; this can never be skipped)
        self._emit("ground", description=description)
        grounding_margin = check_grounding(self._embedder, description, img, self._grounding_min_cos)

        # 3. ASSESS
        self._emit("assess", grounding_margin=round(grounding_margin, 3))
        try:
            assessment = chat.ask_json(
                "Call assess for this image. Report overall technical quality "
                "0-10 and list every concrete defect you can actually see -- "
                "the list may well be empty.",
                name="assess",
                max_new_tokens=900,
            )
        except ValueError as exc:
            raise UngroundedReviewError(f"the model produced no usable assessment: {exc}") from exc
        if not isinstance(assessment, dict):
            raise UngroundedReviewError("the assessment was not an object")

        findings, dropped_unverified = self._findings(chat, assessment, img)

        # 4.5 ALIGN -- its own pass over the prompt. Elements come from the
        # deterministic expected-text guard (verbatim prompt slices,
        # negative-prompt terms excluded), so a verdict can never be about
        # content the prompt did not request. The step-1 caption rides along
        # as context: the model has already committed to what it sees, and
        # judging the prompt against that commitment is harder to talk
        # itself out of than judging the prompt against the pixels again.
        alignment_elements: list = []
        adherence_note = ""
        if expected:
            self._emit("align", findings=len(findings))
            alignment_elements = self._alignment(chat, expected, description, img, findings)
            if alignment_elements:
                satisfied = sum(1 for e in alignment_elements if e["satisfied"])
                adherence_note = f" [adherence {satisfied}/{len(alignment_elements)} prompt elements]"
                if truncated:
                    # Say it. The score is satisfied/total, so a quietly
                    # shortened total reads as a complete verdict on a
                    # prompt that was never fully checked.
                    adherence_note += f" [only the first {_ALIGN_MAX_ELEMENTS} prompt elements were checked]"

        # 5. ASSEMBLE (deterministic). Prompt-following is the fraction of
        # the user's own prompt the image actually delivered -- countable,
        # and explained element by element by the rows above. None means
        # only "this file carries no generation prompt to follow".
        alignment_score = None
        if alignment_elements:
            alignment_score = sum(1 for e in alignment_elements if e["satisfied"]) / len(alignment_elements)

        self._emit(
            "assemble", findings=len(findings), alignment=len(alignment_elements), alignment_score=alignment_score
        )
        summary = f"{description[:260]} [grounding margin {grounding_margin:.2f}]"
        if dropped_unverified:
            summary += f" [{dropped_unverified} finding(s) dropped: region verification failed]"
        summary += adherence_note

        return {
            # Not clamped: validate_review_payload rejects out-of-range
            # values rather than quietly admitting them.
            "quality_score": assessment.get("quality_score"),
            "prompt_alignment_score": alignment_score,
            "summary": summary,
            "findings": findings,
            "alignment": alignment_elements,
        }

    def _findings(self, chat, assessment: dict, img: Image.Image) -> tuple:
        """`(findings, dropped_unverified)` from the ASSESS reply.

        Localizable requires all of: an inherently spatial finding type, a
        valid model-emitted bbox, and passing crop verification. No
        fallbacks -- without a genuine locus the finding stays GLOBAL, and
        failing verification drops it."""
        findings: list = []
        dropped = 0
        raw_defects = assessment.get("defects")
        for raw in raw_defects if isinstance(raw_defects, list) else []:
            defect = _as_defect(raw)
            if defect is None:
                continue
            # Kept separate from the step-1 description: the summary must
            # quote the description the grounding margin was computed for,
            # never per-defect text.
            text = defect["what"]
            bbox = None
            if defect["type"] in _LOCALIZABLE_TYPES and defect["region"] != "whole-image":
                bbox = self._locate(chat, text)
                if bbox is not None and not verify_finding_region(self._embedder, text, bbox, img):
                    dropped += 1
                    continue
            if bbox is None and defect["region"] != "whole-image":
                text = f"{text} (reported region: {defect['region']})"
            finding = {
                "type": defect["type"],
                "severity": defect["severity"],
                "confidence": defect["confidence"],
                "localizable": bbox is not None,
                "description": text,
            }
            if bbox is not None:
                finding["bbox"] = list(bbox)
            findings.append(finding)
        return findings, dropped

    def _alignment(self, chat, expected: list, description: str, img: Image.Image, findings: list) -> list:
        """The ALIGN elements, appending prompt_mismatch findings for
        confidently-absent ones. A failed ALIGN never sinks the review."""
        listing = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(expected))
        try:
            reply = chat.ask_json(
                f"You already described this image as: {description}\n\n"
                "It was generated from a prompt requesting the following "
                "elements:\n" + listing + "\n"
                "Call align. For each element, in order, say whether it is "
                "visibly present in the image, your confidence 0-1, and "
                "where it is -- a region name, or 'absent' when it is not "
                "there, or 'whole-image' when it is an overall property "
                "rather than a thing in one place.",
                name="align",
                max_new_tokens=40 * len(expected) + 80,
            )
            raw = reply.get("elements") if isinstance(reply, dict) else None
        except Exception as exc:
            _logger.debug("[AI] ALIGN failed, the review continues: %s", exc)
            return []
        if not isinstance(raw, list) or len(raw) != len(expected):
            # A short or long list cannot be zipped to the elements without
            # guessing which verdict belongs to which ask.
            return []
        verdicts = [_as_verdict(entry) for entry in raw]
        if any(verdict is None for verdict in verdicts):
            return []

        elements: list = []
        for ordinal, (text, verdict) in enumerate(zip(expected, verdicts, strict=False)):
            if verdict is None:  # unreachable; narrows for the type checker
                continue
            satisfied, confidence = verdict["present"], verdict["confidence"]
            # A satisfied element with a real region gets one localization
            # attempt so the panel can highlight exactly what landed. No box
            # -> no highlight; a rectangle invented here would be a
            # confident claim the model never made.
            bbox = None
            if satisfied and verdict["where"] not in ("absent", "whole-image"):
                bbox = self._locate(chat, text)
                if bbox is not None and not verify_finding_region(self._embedder, text, bbox, img):
                    bbox = None
            element = {"ordinal": ordinal, "text": text, "satisfied": satisfied, "confidence": confidence}
            if bbox is not None:
                element["bbox"] = list(bbox)
            elements.append(element)
            # A confidently-absent element is also a defect, so it stays in
            # the findings list the reviewer reads.
            if not satisfied and confidence >= 0.5:
                findings.append(
                    {
                        "type": "prompt_mismatch",
                        "severity": "medium",
                        "confidence": confidence,
                        "localizable": False,
                        "description": f'requested "{text[:120]}" is not visible',
                    }
                )
        return elements

    def _locate(self, chat, what: str) -> tuple | None:
        """One normalized (x, y, w, h) for `what`, or None when the model
        cannot state a geometrically coherent box (the finding then stays
        global)."""
        try:
            return _as_bbox(
                chat.ask_json(f"Call locate for this: {what}", name="locate", max_new_tokens=96, attempts=1)
            )
        except Exception:  # a failed localization is a global finding
            _logger.debug("handled a failure in _locate", exc_info=True)
            return None
