"""Normalized result model for generation-metadata parsing."""

import re
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


# A1111's extra-network grammar: `<kind:arg1:arg2:...>`, where kind is \w+
# and the args are colon-separated (modules/extra_networks.py:175-191). An
# arg containing '=' is named rather than positional (extra_networks.py:38-42).
_EXTRA_NETWORK_RE = re.compile(r"<(\w+):([^>]+)>")

# `lyco` is registered as an alias of the LoRA network rather than a kind of
# its own (extensions-builtin/Lora/scripts/lora_script.py:24).
_LORA_KINDS = frozenset({"lora", "lyco"})


def extract_networks(*texts) -> list[dict]:
    """LoRAs named inside prompt text, as records rather than substrings.

    This is the only place the tag is read. Four different regexes for it
    had accumulated across the app, each with a different idea of what a
    name may contain and none of them feeding a table -- which is how a
    LoRA ends up being a string you match against instead of a thing you
    join to.

    Weights follow the loader: positional[1] is the text-encoder multiplier
    defaulting to 1.0, positional[2] the unet multiplier defaulting to the
    text-encoder one, and `te=` / `unet=` override either
    (extensions-builtin/Lora/extra_networks_lora.py:30-39).

    Two deliberate departures from upstream, both because this reads
    metadata rather than generates from it: a non-numeric weight falls back
    to the default instead of raising, and the tag is left in the prompt it
    was read from -- A1111 strips it before generating, but here the prompt
    is a stored entity and the text is what the person wrote.
    """
    found: list[dict] = []
    seen: set[str] = set()
    for text in texts:
        for kind, arguments in _EXTRA_NETWORK_RE.findall(text or ""):
            if kind.lower() not in _LORA_KINDS:
                continue
            positional: list[str] = []
            named: dict[str, str] = {}
            for item in arguments.split(":"):
                parts = item.split("=", 2)
                if len(parts) == 2:
                    named[parts[0]] = parts[1]
                else:
                    positional.append(item)
            if not positional or not positional[0].strip():
                continue
            name = positional[0].strip()
            if name.lower() in seen:
                continue
            seen.add(name.lower())
            text_encoder = _weight(named.get("te", positional[1] if len(positional) > 1 else None))
            if text_encoder is None:
                text_encoder = 1.0
            unet = _weight(named.get("unet", positional[2] if len(positional) > 2 else None))
            if unet is None:
                unet = text_encoder
            found.append({"name": name, "weight": unet, "clip_weight": text_encoder})
    return found


def _weight(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def size_string(width, height) -> str | None:
    try:
        w, h = int(width), int(height)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return f"{w}x{h}"
