"""Generation review: a typed, strictly-validated critique of one asset
(quality score, prompt-alignment score, and typed findings), plus optional
per-finding segmentation masks.

`validate_review_payload` is the ONLY door from raw model/backend JSON into
a `ReviewResult`: any dict a `CriticBackend` returns, however it was
produced, must pass through it before touching the database. It is strict
by design -- unknown keys, wrong types, and out-of-range scores are
rejected outright (never silently clamped or coerced) so a malformed or
hallucinated payload fails loudly instead of writing garbage.

`localizable` findings may carry a bounding box and/or points and, later, a
segmentation mask; `localizable=False` ("global") findings may carry none
of those -- enforced both here and by the `ai_review_findings` CHECK
constraint in schema.py, so the invariant holds even for rows written
outside this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from smartgallery_ai import AIConfig
from smartgallery_ai.embedders import BackendUnavailable

__all__ = [
    "FINDING_TYPES",
    "Finding",
    "ReviewResult",
    "ReviewSchemaError",
    "validate_review_payload",
    "CriticBackend",
    "StubCritic",
    "get_critic_backend",
    "store_review",
    "SegmenterBackend",
    "StubSegmenter",
    "MaskNotAllowedError",
    "generate_finding_mask",
]

# Closed vocabulary of finding categories; the DB CHECK constraint and every
# critic prompt/schema reference exactly this set.
FINDING_TYPES = (
    "anatomy",
    "artifact",
    "composition",
    "lighting",
    "text_render",
    "prompt_mismatch",
    "style",
    "detail_loss",
    "other",
)

_SEVERITIES = ("low", "medium", "high")
# Exhaustive key sets: any key outside these fails validation outright.
_TOP_LEVEL_KEYS = {"quality_score", "prompt_alignment_score", "summary", "findings"}
_FINDING_KEYS = {"type", "severity", "confidence", "localizable", "description", "bbox", "points"}


@dataclass
class Finding:
    """One typed defect/observation within a review. Geometry is normalized
    to the unit square and permitted only when `localizable` is True."""

    type: str  # one of FINDING_TYPES
    severity: str  # 'low' | 'medium' | 'high'
    confidence: float  # 0..1
    localizable: bool  # True = tied to a specific image region; False = whole-image
    description: str
    bbox: Optional[tuple] = None  # (x, y, w, h), normalized -- localizable only
    points: Optional[list] = None  # list[(x, y)], normalized -- localizable only


@dataclass
class ReviewResult:
    """Validated critique of one asset -- the only shape `store_review`
    accepts. Construct via `validate_review_payload`, never by hand from
    raw model output."""

    quality_score: float  # 0..10
    prompt_alignment_score: Optional[float]  # 0..10, or None if not applicable
    summary: str
    findings: list  # list[Finding]


class ReviewSchemaError(ValueError):
    """Raised by `validate_review_payload`, naming the first offending field.

    `path` is a JSON-path-ish string (e.g. "findings[2].confidence") for
    the first validation failure encountered, in a fixed, deterministic
    check order.
    """

    def __init__(self, path: str, message: str):
        """`path` locates the offending field; the exception text reads
        "<path>: <message>"."""
        self.path = path
        super().__init__(f"{path}: {message}")


def _is_number(value) -> bool:
    """True for int/float only; bool subclasses int and must never satisfy
    a numeric-field check."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require(condition: bool, path: str, message: str) -> None:
    """Enforce one schema rule: raises `ReviewSchemaError` at `path` when
    `condition` is false."""
    if not condition:
        raise ReviewSchemaError(path, message)


def _validate_point(value, path: str) -> tuple:
    """Validate one [x, y] pair into a float tuple; coordinates are
    fractions of image size and must lie in [0, 1]."""
    _require(
        isinstance(value, (list, tuple)) and len(value) == 2, path, "must be a 2-element [x, y]"
    )
    x, y = value
    _require(_is_number(x) and _is_number(y), path, "coordinates must be numbers")
    _require(
        0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0,
        path,
        "coordinates must be within [0, 1]",
    )
    return (float(x), float(y))


def _validate_bbox(value, path: str) -> tuple:
    """Validate one [x, y, w, h] box into a float tuple: components in
    [0, 1], positive area, and the whole box inside the unit frame."""
    _require(isinstance(value, (list, tuple)) and len(value) == 4, path, "must be [x, y, w, h]")
    _require(all(_is_number(v) for v in value), path, "all components must be numbers")
    x, y, w, h = (float(v) for v in value)
    _require(all(0.0 <= v <= 1.0 for v in (x, y, w, h)), path, "components must be within [0, 1]")
    # Degenerate or out-of-frame boxes are rejected, not repaired: a
    # zero-area box would produce an empty mask the UI still advertises,
    # and x+w/y+h beyond the frame is not a real image region.
    _require(w > 0.0 and h > 0.0, path, "bbox must have positive area")
    _require(x + w <= 1.0 + 1e-6 and y + h <= 1.0 + 1e-6, path,
             "bbox must lie within the image frame")
    return (x, y, w, h)


def _validate_finding(raw, index: int) -> Finding:
    """Validate one raw finding dict into a `Finding`, enforcing the
    geometry rule: localizable findings need bbox or points; global
    findings may carry neither."""
    prefix = f"findings[{index}]"
    _require(isinstance(raw, dict), prefix, "must be an object")
    unknown = set(raw) - _FINDING_KEYS
    _require(not unknown, prefix, f"unknown key(s): {sorted(unknown)}")

    for key in ("type", "severity", "confidence", "localizable", "description"):
        _require(key in raw, f"{prefix}.{key}", "missing required field")

    ftype = raw["type"]
    _require(
        isinstance(ftype, str) and ftype in FINDING_TYPES,
        f"{prefix}.type",
        f"must be one of {FINDING_TYPES}",
    )

    severity = raw["severity"]
    _require(
        isinstance(severity, str) and severity in _SEVERITIES,
        f"{prefix}.severity",
        f"must be one of {_SEVERITIES}",
    )

    confidence = raw["confidence"]
    _require(_is_number(confidence), f"{prefix}.confidence", "must be a number")
    _require(0.0 <= float(confidence) <= 1.0, f"{prefix}.confidence", "must be within [0, 1]")

    localizable = raw["localizable"]
    _require(isinstance(localizable, bool), f"{prefix}.localizable", "must be a boolean")

    description = raw["description"]
    _require(isinstance(description, str), f"{prefix}.description", "must be a string")

    bbox_raw = raw.get("bbox")
    points_raw = raw.get("points")
    if localizable:
        _require(
            bbox_raw is not None or points_raw is not None,
            prefix,
            "localizable findings require bbox or points",
        )
    else:
        _require(
            bbox_raw is None and points_raw is None,
            prefix,
            "non-localizable findings must not carry bbox or points",
        )

    bbox = _validate_bbox(bbox_raw, f"{prefix}.bbox") if bbox_raw is not None else None
    points = None
    if points_raw is not None:
        _require(
            isinstance(points_raw, list) and len(points_raw) > 0,
            f"{prefix}.points",
            "must be a non-empty list of [x, y] pairs",
        )
        points = [_validate_point(p, f"{prefix}.points[{i}]") for i, p in enumerate(points_raw)]

    return Finding(
        type=ftype,
        severity=severity,
        confidence=float(confidence),
        localizable=localizable,
        description=description,
        bbox=bbox,
        points=points,
    )


def validate_review_payload(payload: dict) -> ReviewResult:
    """Strictly validate a raw critic payload into a `ReviewResult`.

    Rejects (rather than coerces or clamps): non-dict input, any key
    outside the known schema, wrong-typed values, out-of-range scores,
    localizable findings missing grounding geometry, and non-localizable
    findings carrying bbox/points. Raises `ReviewSchemaError` naming the
    first offending field.
    """
    _require(isinstance(payload, dict), "$", "payload must be an object")
    unknown = set(payload) - _TOP_LEVEL_KEYS
    _require(not unknown, "$", f"unknown key(s): {sorted(unknown)}")

    for key in ("quality_score", "summary", "findings"):
        _require(key in payload, key, "missing required field")

    quality_score = payload["quality_score"]
    _require(_is_number(quality_score), "quality_score", "must be a number")
    _require(0.0 <= float(quality_score) <= 10.0, "quality_score", "must be within [0, 10]")

    prompt_alignment_score = payload.get("prompt_alignment_score")
    if prompt_alignment_score is not None:
        _require(
            _is_number(prompt_alignment_score),
            "prompt_alignment_score",
            "must be a number or null",
        )
        _require(
            0.0 <= float(prompt_alignment_score) <= 10.0,
            "prompt_alignment_score",
            "must be within [0, 10]",
        )

    summary = payload["summary"]
    _require(isinstance(summary, str), "summary", "must be a string")

    findings_raw = payload["findings"]
    _require(isinstance(findings_raw, list), "findings", "must be a list")
    findings = [_validate_finding(f, i) for i, f in enumerate(findings_raw)]

    return ReviewResult(
        quality_score=float(quality_score),
        prompt_alignment_score=(
            float(prompt_alignment_score) if prompt_alignment_score is not None else None
        ),
        summary=summary,
        findings=findings,
    )


class CriticBackend(ABC):
    """A model that critiques one image and emits a raw payload dict;
    `model_id`/`model_version` are recorded as provenance with every
    stored review."""

    model_id: str  # stable identifier of the underlying model (e.g. HF repo id)
    model_version: str  # provenance tag stored with every review row

    @abstractmethod
    def review(self, img: Image.Image, prompt_text: Optional[str], rubric_version: str) -> dict:
        """Return a RAW payload dict; the caller must validate it via
        `validate_review_payload` before it touches the database."""


class StubCritic(CriticBackend):
    """TEST/DEV STUB -- derives a payload from crude image statistics.

    Not a real generation critic. Two deliberately simple, deterministic
    heuristics exist purely so tests can construct images that trigger a
    known finding:
      - mean brightness below a threshold -> one global ('lighting') finding
      - a solid, roughly-1/16th-image-area red rectangle -> one localizable
        ('artifact') finding whose bbox is the rectangle's bounding box
    """

    model_id = "stub-critic"
    model_version = "stub-v1"

    _DARK_MEAN_THRESHOLD = 40.0  # mean RGB (0-255) below this -> 'lighting' finding
    _RED_MIN_R = 180  # red-channel floor (0-255) for the artifact-rectangle mask
    _RED_MAX_GB = 80  # green/blue ceiling (0-255) for the artifact-rectangle mask

    def review(self, img: Image.Image, prompt_text: Optional[str], rubric_version: str) -> dict:
        """Apply the two stub heuristics and emit a raw payload dict; the
        quality score drops 2 points per finding."""
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        findings = []

        mean_brightness = float(rgb.mean())
        if mean_brightness < self._DARK_MEAN_THRESHOLD:
            findings.append(
                {
                    "type": "lighting",
                    "severity": "medium",
                    "confidence": 0.8,
                    "localizable": False,
                    "description": f"Image is very dark (mean brightness {mean_brightness:.1f}/255).",
                }
            )

        red_bbox = self._find_red_rectangle(rgb)
        if red_bbox is not None:
            findings.append(
                {
                    "type": "artifact",
                    "severity": "high",
                    "confidence": 0.9,
                    "localizable": True,
                    "description": "Solid red rectangular artifact detected.",
                    "bbox": list(red_bbox),
                }
            )

        quality_score = max(0.0, min(10.0, 10.0 - 2.0 * len(findings)))
        return {
            "quality_score": quality_score,
            "prompt_alignment_score": 5.0 if prompt_text else None,
            "summary": f"Stub critic ({rubric_version}) found {len(findings)} finding(s).",
            "findings": findings,
        }

    @classmethod
    def _find_red_rectangle(cls, rgb: np.ndarray) -> Optional[tuple]:
        """Normalized (x, y, w, h) bounding box of all strongly-red pixels,
        or None when the image has none."""
        h, w = rgb.shape[:2]
        r = rgb[..., 0].astype(np.int16)
        g = rgb[..., 1].astype(np.int16)
        b = rgb[..., 2].astype(np.int16)
        mask = (r >= cls._RED_MIN_R) & (g <= cls._RED_MAX_GB) & (b <= cls._RED_MAX_GB)
        if not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        return (x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h)


# Larger checkpoints win when several are provisioned side by side.
_SMOLVLM_DIRNAMES = ("smolvlm2-2.2b", "smolvlm2-500m")
# dirname -> (model_id, model_version) provenance recorded with reviews.
_SMOLVLM_MODEL_IDS = {
    "smolvlm2-2.2b": ("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "smolvlm2-2.2b-instruct-v1"),
    "smolvlm2-500m": ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "smolvlm2-500m-video-instruct-v1"),
}

# Single-turn instruction for SmolVLM: demands one JSON object in the exact
# review schema; `validate_review_payload` rejects any deviation downstream.
_CRITIC_INSTRUCTION = """You are a strict image-generation quality reviewer. Reply with ONLY one JSON object, no other text.
Required keys: quality_score (0-10), prompt_alignment_score (0-10, or null when no prompt given), summary (one sentence), findings (list, may be empty).
EVERY finding must have ALL of these keys: type (one of anatomy, artifact, composition, lighting, text_render, prompt_mismatch, style, detail_loss, other), severity (low, medium or high), confidence (0-1), localizable (true or false), description (short text). Add bbox [x,y,w,h] (fractions 0-1) ONLY when localizable is true.
Example of a complete, valid reply:
{"quality_score": 6.5, "prompt_alignment_score": 7.0, "summary": "Good portrait with one artifact.", "findings": [{"type": "artifact", "severity": "high", "confidence": 0.9, "localizable": true, "description": "red square artifact", "bbox": [0.55, 0.6, 0.2, 0.2]}, {"type": "lighting", "severity": "low", "confidence": 0.6, "localizable": false, "description": "slightly flat lighting"}]}
Keep the reply short: at most 3 findings."""


class SmolVlmCritic(CriticBackend):
    """EXPERIMENTAL local VLM critic via SmolVLM2 (Apache-2.0), loaded ONLY
    from `models_dir/smolvlm2-2.2b` or `models_dir/smolvlm2-500m`
    (local_files_only) -- never downloaded at runtime.

    Explicit 'smolvlm' opt-in only; 'auto' never resolves here (see
    docs/AI_MODELS.md for the record behind that). Emits the RAW payload
    dict; `validate_review_payload` remains the only gate into the
    database. Validation cannot catch a schema-valid fabrication, which is
    the failure mode that keeps this opt-in.
    """

    model_id = "HuggingFaceTB/SmolVLM2"  # refined per provisioned checkpoint
    model_version = "smolvlm2-v1"

    def __init__(self, models_dir: str, max_new_tokens: int = 640):
        """Load processor + weights from the provisioned checkpoint dir;
        raises `BackendUnavailable` when weights are absent or the
        torch/transformers runtime is missing or fails to load them."""
        # Weights check precedes the runtime import: resolution on an
        # unprovisioned system must stay fast and side-effect-free.
        weights_dir = None
        for dirname in _SMOLVLM_DIRNAMES:
            candidate = os.path.join(models_dir, dirname)
            if os.path.isdir(candidate):
                weights_dir = candidate
                self.model_id, self.model_version = _SMOLVLM_MODEL_IDS[dirname]
                break
        if weights_dir is not None:
            try:
                import torch  # noqa: F401
                from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: F401
            except Exception as exc:  # noqa: BLE001
                raise BackendUnavailable(f"smolvlm critic unavailable: {exc}") from exc
        if weights_dir is None:
            raise BackendUnavailable(
                f"smolvlm weights not found under {models_dir} "
                f"(looked for {', '.join(_SMOLVLM_DIRNAMES)})")
        try:
            self._processor = AutoProcessor.from_pretrained(
                weights_dir, local_files_only=True
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                weights_dir, local_files_only=True, torch_dtype=torch.float32
            )
            self._model.eval()
        except Exception as exc:  # noqa: BLE001
            raise BackendUnavailable(f"failed to load smolvlm weights: {exc}") from exc
        self._torch = torch
        self._max_new_tokens = max_new_tokens

    def review(self, img: Image.Image, prompt_text: Optional[str], rubric_version: str) -> dict:
        """Single greedy-decoded image+instruction turn; returns the first
        JSON object in the reply (ValueError when there is none)."""
        instruction = _CRITIC_INSTRUCTION
        if prompt_text:
            instruction += f'\nGeneration prompt: "{prompt_text}"'
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": img.convert("RGB")},
                        {"type": "text", "text": instruction}],
        }]
        inputs = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        with self._torch.no_grad():
            generated = self._model.generate(
                **inputs, do_sample=False, max_new_tokens=self._max_new_tokens
            )
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)
        return _extract_json_object(text)


def _extract_json_object(text: str) -> dict:
    """Pull the first balanced JSON object out of model output. Raises
    ValueError when there is none -- the caller treats that as a failed
    review, never as data.

    Brace counting is string-aware: braces (and escaped quotes) inside
    JSON string values do not affect nesting depth, so a summary
    containing '}' still parses."""
    start = text.find("{")
    if start < 0:
        raise ValueError(f"critic output contains no JSON object: {text[:120]!r}")
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("critic output has an unterminated JSON object")


# 'auto' -> qwen-vl resolution requires the committed grounding-gate
# calibration report to meet these bounds at the shipped margin threshold.
_CALIBRATION_REPORT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "benchmarks", "results", "grounding_calibration.json")
_AUTO_CRITIC_MAX_FAR = 0.05  # false-accept-rate ceiling (ungrounded text passing the gate)
_AUTO_CRITIC_MAX_FRR = 0.30  # false-reject-rate ceiling (grounded text failing the gate)


# Repo-relative calibration input that must appear (hash-verified) in the
# report's input manifest.
_CALIBRATION_PORTRAIT_REL = "probes/data/calibration_portrait.png"


def _sha256_of_file(path: str) -> str:
    """Hex SHA-256 of a file's bytes, streamed so large media never loads
    into memory whole."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _auto_critic_measurement_passed(report_path: Optional[str] = None) -> bool:
    """Whether the calibration report at `report_path` (default: the
    committed benchmarks/results/grounding_calibration.json, written by
    probes/grounding_calibration.py) authorizes 'auto' critic resolution.

    Acceptance is bound to the evidence's IDENTITY, not just its numbers.
    A report qualifies only when ALL hold:
      - its `backend` names the exact embedding backend the shipped gate
        runs on (OpenClipSemanticEmbedder model_id/model_version);
      - its `baseline_text` is the shipped GROUNDING_BASELINE_TEXT;
      - its input manifest includes the committed portrait input, and
        every file-backed manifest entry's SHA-256 matches the file in
        THIS checkout (a report calibrated on a different population
        cannot authorize this one);
      - the sweep row at the shipped margin threshold shows
        FAR <= _AUTO_CRITIC_MAX_FAR and FRR <= _AUTO_CRITIC_MAX_FRR.
    Anything missing, malformed, mismatched, or out of bounds -> False.
    """
    try:
        from smartgallery_ai.critic_qwen import (
            DEFAULT_GROUNDING_MIN_MARGIN, GROUNDING_BASELINE_TEXT)
        from smartgallery_ai.embedders import OpenClipSemanticEmbedder
        with open(report_path or _CALIBRATION_REPORT_PATH, "r", encoding="utf-8") as fh:
            report = json.load(fh)

        backend = report["backend"]
        if (backend["model_id"] != OpenClipSemanticEmbedder.model_id
                or backend["model_version"] != OpenClipSemanticEmbedder.model_version):
            return False
        if report["baseline_text"] != GROUNDING_BASELINE_TEXT:
            return False

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        inputs = report["inputs"]
        file_entries = [e for e in inputs if "file" in e]
        if not inputs or not any(
                e["file"] == _CALIBRATION_PORTRAIT_REL for e in file_entries):
            return False
        for entry in file_entries:
            if _sha256_of_file(os.path.join(repo_root, entry["file"])) != entry["file_sha256"]:
                return False

        row = next(
            s for s in report["sweep"]
            if abs(float(s["margin_threshold"]) - DEFAULT_GROUNDING_MIN_MARGIN) < 1e-9)
        return (float(row["false_accept_rate"]) <= _AUTO_CRITIC_MAX_FAR
                and float(row["false_reject_rate"]) <= _AUTO_CRITIC_MAX_FRR)
    except Exception:
        return False


def get_critic_backend(config: AIConfig) -> Optional[CriticBackend]:
    """Resolve `config.critic_backend`.

    'qwen-vl' loads the decomposed Qwen2.5-VL critic
    (smartgallery_ai.critic_qwen). 'auto' resolves to it only when
    `_auto_critic_measurement_passed()` accepts the committed calibration
    evidence. 'smolvlm' is explicit-opt-in; 'stub' is test-only and never
    reachable implicitly.
    """
    name = config.critic_backend
    if name == "none":
        return None
    if name in ("auto", "qwen-vl"):
        if name == "auto" and not _auto_critic_measurement_passed():
            return None
        try:
            from smartgallery_ai.critic_qwen import QwenVlCritic

            from smartgallery_ai.embedders import get_semantic_backend
            # A missing semantic backend makes the critic unavailable;
            # it must never run without its grounding gate.
            embedder = get_semantic_backend(config)
            if embedder is None:
                raise BackendUnavailable(
                    "qwen-vl critic requires the semantic (OpenCLIP) backend "
                    "for grounding and prompt-alignment; it is unavailable")
            return QwenVlCritic(config.models_dir, semantic_embedder=embedder)
        except BackendUnavailable:
            if name == "qwen-vl":
                raise
            return None
    if name == "smolvlm":
        return SmolVlmCritic(config.models_dir)
    if name == "stub":
        return StubCritic()
    raise ValueError(f"unknown critic_backend: {name!r}")


def get_segmenter_backend(config: AIConfig) -> Optional[SegmenterBackend]:
    """Resolve `config.segmenter_backend`.

    'auto'/'mobilesam' -> MobileSAM (smartgallery_ai.segmenter_mobilesam)
    when weights + runtime are provisioned; 'auto' degrades to None,
    'mobilesam' raises. 'stub' is test-only and explicit.
    """
    name = config.segmenter_backend
    if name == "none":
        return None
    if name in ("auto", "mobilesam"):
        try:
            from smartgallery_ai.segmenter_mobilesam import MobileSamSegmenter

            return MobileSamSegmenter(config.models_dir)
        except BackendUnavailable:
            if name == "mobilesam":
                raise
            return None
    if name == "stub":
        return StubSegmenter()
    raise ValueError(f"unknown segmenter_backend: {name!r}")


def store_review(
    conn: sqlite3.Connection,
    file_id: str,
    result: ReviewResult,
    model_id: str,
    model_version: str,
    rubric_version: str,
    raw_response: Optional[str],
    source_mtime: float,
    now: float,
) -> int:
    """Upsert one review by UNIQUE(file_id, rubric_version, model_id).

    Deletes any prior review (and its findings, via `review_id`) under the
    same key, then inserts fresh rows. Global (non-localizable) findings
    always write NULL bbox/mask columns, honoring the `ai_review_findings`
    CHECK regardless of what a `Finding` happens to carry.
    """
    superseded_masks: list = []
    try:
        old = conn.execute(
            "SELECT review_id FROM ai_reviews WHERE file_id = ? AND rubric_version = ? AND model_id = ?",
            (file_id, rubric_version, model_id),
        ).fetchone()
        if old is not None:
            # Collect superseded mask paths up front; unlink only after the
            # replacement commits — rollback() cannot undo os.unlink.
            superseded_masks = [m for (m,) in conn.execute(
                "SELECT mask_path FROM ai_review_findings "
                "WHERE review_id = ? AND mask_path IS NOT NULL", (old[0],))]
            conn.execute("DELETE FROM ai_review_findings WHERE review_id = ?", (old[0],))
            conn.execute("DELETE FROM ai_reviews WHERE review_id = ?", (old[0],))

        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ai_reviews
                (file_id, rubric_version, model_id, model_version, quality_score,
                 prompt_alignment_score, summary, raw_response, source_mtime, computed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                rubric_version,
                model_id,
                model_version,
                result.quality_score,
                result.prompt_alignment_score,
                result.summary,
                raw_response,
                source_mtime,
                now,
            ),
        )
        review_id = cur.lastrowid

        for finding in result.findings:
            if finding.localizable:
                bbox = finding.bbox or (None, None, None, None)
                bbox_x, bbox_y, bbox_w, bbox_h = bbox
                points_json = (
                    json.dumps([[p[0], p[1]] for p in finding.points])
                    if finding.points
                    else None
                )
            else:
                bbox_x = bbox_y = bbox_w = bbox_h = None
                points_json = None
            cur.execute(
                """
                INSERT INTO ai_review_findings
                    (review_id, file_id, type, severity, confidence, localizable,
                     bbox_x, bbox_y, bbox_w, bbox_h, points, description,
                     mask_path, mask_model_id, mask_model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
                """,
                (
                    review_id,
                    file_id,
                    finding.type,
                    finding.severity,
                    finding.confidence,
                    1 if finding.localizable else 0,
                    bbox_x,
                    bbox_y,
                    bbox_w,
                    bbox_h,
                    points_json,
                    finding.description,
                ),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    for old_mask in superseded_masks:
        try:
            os.unlink(old_mask)
        except OSError:
            pass
    return review_id


class SegmenterBackend(ABC):
    """A promptable segmentation model turning box/point grounding into a
    pixel mask; `model_id`/`model_version` are recorded as provenance on
    each finding's mask columns."""

    model_id: str  # stable identifier of the underlying model
    model_version: str  # provenance tag stored with each generated mask

    @abstractmethod
    def segment(
        self,
        img: Image.Image,
        bbox: Optional[tuple] = None,
        points: Optional[list] = None,
    ) -> np.ndarray:
        """Return a boolean HxW mask (True = part of the finding); `bbox`
        and `points` are normalized to [0, 1]."""


class StubSegmenter(SegmenterBackend):
    """Degenerate TEST/DEV segmenter: rasterizes the bbox rectangle as-is.

    Not a real segmentation model. Ignores `points` unless no `bbox` is
    given, in which case it falls back to the points' bounding box. Exists
    so mask generation can be exercised end-to-end without a real backend.
    """

    model_id = "stub-segmenter"
    model_version = "stub-v1"

    def segment(
        self,
        img: Image.Image,
        bbox: Optional[tuple] = None,
        points: Optional[list] = None,
    ) -> np.ndarray:
        """Rasterize the normalized bbox (or the points' bounding box) onto
        an all-False HxW canvas."""
        w, h = img.size
        mask = np.zeros((h, w), dtype=np.bool_)
        if bbox is None:
            if not points:
                return mask
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        x, y, bw, bh = bbox
        x0 = max(0, min(w, int(round(x * w))))
        y0 = max(0, min(h, int(round(y * h))))
        x1 = max(0, min(w, int(round((x + bw) * w))))
        y1 = max(0, min(h, int(round((y + bh) * h))))
        mask[y0:y1, x0:x1] = True
        return mask


class MaskNotAllowedError(Exception):
    """A mask was requested for a non-localizable or ungrounded finding."""


def _safe_path_component(value: str) -> str:
    """Collapse a caller-supplied id into a single safe path segment so a
    crafted `file_id`/`finding_id` can't be used to escape `cache_dir`."""
    value = value.replace("\\", "_").replace("/", "_").replace("..", "_")
    return value or "_"


def generate_finding_mask(
    conn: sqlite3.Connection,
    cache_dir: str,
    img: Image.Image,
    file_id: str,
    finding_id: int,
    segmenter: SegmenterBackend,
) -> str:
    """Generate and persist a mask PNG for one localizable finding.

    Loads the finding row fresh from the DB (never trusts caller-supplied
    geometry) and raises `MaskNotAllowedError` if it is not localizable or
    has no grounding geometry. Saves an 'L' mode 0/255 PNG under
    `cache_dir/masks/<file_id>/<finding_id>.png`, asserting the resolved
    path stays inside `cache_dir` even for a maliciously crafted id. Never
    opens or writes to the source media path -- `img` is caller-provided
    pixels only.
    """
    row = conn.execute(
        """
        SELECT localizable, bbox_x, bbox_y, bbox_w, bbox_h, points
        FROM ai_review_findings
        WHERE finding_id = ? AND file_id = ?
        """,
        (finding_id, file_id),
    ).fetchone()
    if row is None:
        raise ValueError(f"no finding {finding_id!r} for file {file_id!r}")

    localizable, bbox_x, bbox_y, bbox_w, bbox_h, points_json = row
    if not localizable:
        raise MaskNotAllowedError(f"finding {finding_id} is not localizable")

    bbox = (bbox_x, bbox_y, bbox_w, bbox_h) if bbox_x is not None else None
    points = [tuple(p) for p in json.loads(points_json)] if points_json else None
    if bbox is None and not points:
        raise MaskNotAllowedError(f"finding {finding_id} has no grounding geometry")

    mask = segmenter.segment(img, bbox=bbox, points=points)
    mask_img = Image.fromarray(np.where(mask, np.uint8(255), np.uint8(0)), mode="L")

    masks_root = os.path.realpath(os.path.join(cache_dir, "masks"))
    file_dir = os.path.realpath(os.path.join(masks_root, _safe_path_component(str(file_id))))
    mask_path = os.path.realpath(
        os.path.join(file_dir, f"{_safe_path_component(str(finding_id))}.png")
    )
    if os.path.commonpath([masks_root, mask_path]) != masks_root:
        raise ValueError("resolved mask path escapes cache_dir")

    os.makedirs(file_dir, exist_ok=True)
    mask_img.save(mask_path)

    conn.execute(
        """
        UPDATE ai_review_findings
        SET mask_path = ?, mask_model_id = ?, mask_model_version = ?
        WHERE finding_id = ?
        """,
        (mask_path, segmenter.model_id, segmenter.model_version, finding_id),
    )
    conn.commit()
    return mask_path
