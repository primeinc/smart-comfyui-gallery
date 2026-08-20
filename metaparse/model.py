"""Normalized result model for generation-metadata parsing."""

from dataclasses import dataclass, field

# Canonical parameter slots every adapter maps into. Anything a tool emits
# beyond these lands in `extra` with its original key.
CANONICAL_KEYS = (
    "model",
    "model_hash",
    "sampler",
    "scheduler",
    "seed",
    "steps",
    "cfg",
    "size",
    "denoise",
    "clip_skip",
    "version",
)


@dataclass
class ParsedMetadata:
    tool: str  # display name, e.g. "SwarmUI", "A1111 / Forge"
    positive: str = ""
    negative: str = ""
    params: dict[str, str] = field(default_factory=dict)  # canonical keys only
    extra: dict[str, str] = field(default_factory=dict)  # tool-specific leftovers
    raw: str = ""  # the embedded text as found (infotext or JSON)
    detection: str = "marker"  # "marker" | "heuristic" | "stealth"

    @property
    def renderable(self) -> bool:
        """True when there is enough content for a human-readable report."""
        return bool(self.positive or self.negative or self.params)


def set_param(target: "ParsedMetadata", key: str, value) -> None:
    """Assign a canonical param if the value is meaningful."""
    if value is None:
        return
    text = str(value).strip()
    if not text or text.lower() == "none":
        return
    if key in CANONICAL_KEYS:
        target.params[key] = text
    else:
        target.extra[key] = text


def size_string(width, height) -> str | None:
    try:
        w, h = int(width), int(height)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return f"{w}x{h}"
