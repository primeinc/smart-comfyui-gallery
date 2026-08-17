"""Typed generation-parameter model: the single coercion point between
tool metadata shapes and first-class storage.

Type authority per tool:
  - SwarmUI (`docs/Image Metadata Format.md`:40, refs/mcmonkeyprojects):
    values "will be presented as strings regardless; consumers that need
    to read these values should use data type forcing" — coercion here is
    the documented contract, not a workaround.
  - A1111 / Forge infotext is stringly by format (modules/infotext_utils.py);
    the numeric semantics of Seed/Steps/CFG scale/Denoising strength/
    Clip skip are the manual typing source.
  - InvokeAI / NovelAI / Fooocus / Draw Things / Easy Diffusion / ComfyUI
    emit native JSON types; coercion is a lossless pass-through there.

Every field carries a real Python type. A value that cannot be coerced is
never guessed at or dropped: it stays verbatim (original key, original
string) in `extra`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .model import ParsedMetadata

_SIZE_RE = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\s*$", re.IGNORECASE)


def to_int(value) -> Optional[int]:
    """Strict int coercion: ints, integral floats, and clean numeric
    strings; anything else is None (caller keeps the original in extra)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    text = str(value).strip()
    try:
        return int(text, 10)
    except ValueError:
        try:
            number = float(text)
        except ValueError:
            return None
        return int(number) if number.is_integer() else None


def to_float(value) -> Optional[float]:
    """Strict float coercion; None for anything non-numeric."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def split_size(value) -> tuple:
    """'832x1216' -> (832, 1216); anything else -> (None, None)."""
    if value is None:
        return None, None
    match = _SIZE_RE.match(str(value))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


@dataclass
class GenerationParams:
    """First-class, typed generation parameters for one file."""

    tool: str
    detection: str                      # marker | heuristic | stealth | graph
    positive_prompt: str = ""
    negative_prompt: str = ""
    model: Optional[str] = None
    model_hash: Optional[str] = None
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    cfg: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    denoise: Optional[float] = None
    clip_skip: Optional[int] = None
    version: Optional[str] = None
    loras: List[dict] = field(default_factory=list)   # [{"name": ..., "weight": ...}]
    extra: Dict[str, object] = field(default_factory=dict)  # unmapped keys, verbatim

    _INT_FIELDS = ("seed", "steps", "clip_skip")
    _FLOAT_FIELDS = ("cfg", "denoise")
    _STR_FIELDS = ("model", "model_hash", "sampler", "scheduler", "version")

    @classmethod
    def from_parsed(cls, parsed: ParsedMetadata) -> "GenerationParams":
        """Type a metaparse result. Canonical slots coerce to their real
        types; a slot that fails coercion moves to extra under its
        canonical key; every adapter-preserved extra key rides along."""
        gp = cls(tool=parsed.tool, detection=parsed.detection,
                 positive_prompt=parsed.positive or "",
                 negative_prompt=parsed.negative or "")
        params = dict(parsed.params)
        for name in cls._STR_FIELDS:
            value = params.pop(name, None)
            if value is not None:
                setattr(gp, name, str(value))
        for name, coerce in ((n, to_int) for n in cls._INT_FIELDS):
            value = params.pop(name, None)
            if value is None:
                continue
            typed = coerce(value)
            if typed is None:
                gp.extra[name] = value
            else:
                setattr(gp, name, typed)
        for name in cls._FLOAT_FIELDS:
            value = params.pop(name, None)
            if value is None:
                continue
            typed = to_float(value)
            if typed is None:
                gp.extra[name] = value
            else:
                setattr(gp, name, typed)
        gp.width, gp.height = split_size(params.pop("size", None))
        if (gp.width, gp.height) == (None, None) and "size" in parsed.params:
            gp.extra["size"] = parsed.params["size"]
        gp.extra.update(params)
        gp.extra.update(parsed.extra)
        return gp

    @classmethod
    def from_comfy(cls, meta: dict) -> "GenerationParams":
        """Type a ComfyUI graph-trace result (the gallery's
        ComfyMetadataParser.parse() dict). Graph values are native JSON
        types; coercion is a guard, not a translation."""
        meta = dict(meta)
        gp = cls(tool="ComfyUI", detection="graph",
                 positive_prompt=str(meta.pop("positive_prompt", "") or ""),
                 negative_prompt=str(meta.pop("negative_prompt", "") or ""))
        gp.model = (str(meta.pop("model")) if meta.get("model") is not None
                    else meta.pop("model", None))
        gp.sampler = (str(meta.pop("sampler")) if meta.get("sampler") is not None
                      else meta.pop("sampler", None))
        gp.scheduler = (str(meta.pop("scheduler")) if meta.get("scheduler") is not None
                        else meta.pop("scheduler", None))
        gp.seed = to_int(meta.pop("seed", None))
        gp.steps = to_int(meta.pop("steps", None))
        gp.cfg = to_float(meta.pop("cfg", None))
        gp.width = to_int(meta.pop("width", None))
        gp.height = to_int(meta.pop("height", None))
        gp.denoise = to_float(meta.pop("denoise", None))
        loras = meta.pop("loras", None)
        if isinstance(loras, list):
            gp.loras = [entry for entry in loras if isinstance(entry, dict)]
        meta.pop("positive_prompt_clean", None)
        for key, value in meta.items():
            if value is None or value == "":
                continue
            gp.extra[key] = value
        return gp

    @property
    def has_content(self) -> bool:
        return bool(self.positive_prompt or self.negative_prompt
                    or self.model or self.seed is not None)

    def to_row(self, file_id: str, parsed_at: float) -> tuple:
        """The generation_params DB row, column order matching schema."""
        return (
            file_id, self.tool, self.detection,
            self.positive_prompt, self.negative_prompt,
            self.model, self.model_hash, self.sampler, self.scheduler,
            self.seed, self.steps, self.cfg,
            self.width, self.height, self.denoise, self.clip_skip,
            self.version,
            json.dumps(self.loras) if self.loras else None,
            json.dumps(self.extra, default=str) if self.extra else None,
            parsed_at,
        )


# Column list matching to_row(), for INSERT statements and readers.
ROW_COLUMNS = (
    "file_id", "tool", "detection", "positive_prompt", "negative_prompt",
    "model", "model_hash", "sampler", "scheduler", "seed", "steps", "cfg",
    "width", "height", "denoise", "clip_skip", "version", "loras", "extra",
    "parsed_at",
)
