"""Human-readable report rendering for parsed generation metadata."""

from .model import ParsedMetadata, extract_networks

_EMOJI = {
    "sampling": "\U0001f3af",
    "dimensions": "\U0001f4cf",
    "prompts": "\U0001f4dd",
    "models": "\U0001f9e0",
    "lora": "\U0001f3a8",
    "advanced": "⚙️",
}


def render_report(parsed: ParsedMetadata, include_emojis: bool = True) -> str | None:
    """Format normalized metadata into the panel text shown in file details.

    Returns None when the parse carries nothing worth displaying (e.g. a pure
    ComfyUI graph, which the workflow panel already covers).
    """
    if parsed is None or not parsed.renderable:
        return None
    emoji = _EMOJI if include_emojis else dict.fromkeys(_EMOJI, "")
    params = parsed.params
    out = [f"=== {parsed.tool} Generation Parameters ===\n"]

    out.append(f"{emoji['models']} MODEL: {params.get('model', 'N/A')}\n")
    out.append(f"{emoji['prompts']} PROMPTS:\n")
    out.append(f"  Positive:\n           {parsed.positive or '(empty)'}\n")
    if parsed.negative:
        out.append(f"  Negative:\n           {parsed.negative}")
    out.append("")

    sampling = [
        ("Seed", params.get("seed")),
        ("Steps", params.get("steps")),
        ("CFG Scale", params.get("cfg")),
        ("Sampler", params.get("sampler")),
        ("Scheduler", params.get("scheduler")),
        ("Denoise", params.get("denoise")),
    ]
    if any(v for _, v in sampling):
        out.append(f"{emoji['sampling']} SAMPLING SETTINGS:")
        out.extend(f"  {label}: {value}" for label, value in sampling if value)
        out.append("")

    if params.get("size"):
        out.append(f"{emoji['dimensions']} IMAGE DIMENSIONS:")
        out.append(f"  Resolution: {params['size']}")
        out.append("")

    if params.get("model") or params.get("model_hash"):
        out.append(f"{emoji['models']} MODELS & COMPONENTS:")
        if params.get("model"):
            out.append(f"  Checkpoint: {params['model']}")
        if params.get("model_hash"):
            out.append(f"  Model Hash: {params['model_hash']}")
        out.append("")

    loras = extract_networks(parsed.positive, parsed.raw)
    if loras:
        out.append(f"{emoji['lora']} LORA MODELS:")
        out.extend(f"  {lora['name']} (Strength: {lora['weight']})" for lora in loras)
        out.append("")

    advanced = [
        ("Clip Skip", params.get("clip_skip")),
        ("Tool Version", params.get("version")),
    ]
    if any(v for _, v in advanced):
        out.append(f"{emoji['advanced']} ADVANCED SETTINGS:")
        out.extend(f"  {label}: {value}" for label, value in advanced if value)
        out.append("")

    if parsed.extra:
        out.append(f"{emoji['advanced']} OTHER SETTINGS:")
        out.extend(f"  {key}: {value}" for key, value in parsed.extra.items())
        out.append("")

    return "\n".join(out).rstrip() + "\n"
