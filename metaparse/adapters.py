"""Per-tool metadata adapters with marker-first detection.

Detection contract:
  1. Marker pass — each adapter declares explicit, tool-unique markers
     (named chunks like ``sui_image_params`` / ``fooocus_scheme`` /
     ``invokeai_metadata``); the first marker hit wins.
  2. Heuristic pass — shape-based fallbacks (bare infotext, bare JSON in a
     UserComment) ordered by tool popularity.
  3. Stealth pass (opt-in) — LSB-embedded payload, re-dispatched by content.

Format references (cloned under ../refs):
  AUTOMATIC1111/stable-diffusion-webui  modules/images.py, infotext_utils.py
  Comfy-Org/ComfyUI                     nodes.py, comfy_api/latest/_ui.py
  mcmonkeyprojects/SwarmUI              docs/Image Metadata Format.md
  lllyasviel/Fooocus                    modules/private_logger.py, meta_parser.py
  lllyasviel/stable-diffusion-webui-forge  modules/stealth_infotext.py
  invoke-ai/InvokeAI                    app/services/image_files/image_files_disk.py
  receyuki/stable-diffusion-prompt-reader  sd_prompt_reader/ (MIT; detection
    cascade and NovelAI/EasyDiffusion/DrawThings field maps ported from it)
"""

import json
import re
from typing import Optional
from xml.dom import minidom

from .containers import RawMetadata, load_raw
from .model import ParsedMetadata, set_param, size_string

# ---------------------------------------------------------------------------
# A1111 infotext grammar (ported from modules/infotext_utils.py)
# ---------------------------------------------------------------------------

_RE_PARAM = re.compile(r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)')
_RE_IMAGESIZE = re.compile(r"^(\d+)x(\d+)$")

# infotext key -> canonical param slot
_INFOTEXT_CANONICAL = {
    "Model": "model", "Model hash": "model_hash", "Sampler": "sampler",
    "Schedule type": "scheduler", "Seed": "seed", "Steps": "steps",
    "CFG scale": "cfg", "Size": "size", "Denoising strength": "denoise",
    "Clip skip": "clip_skip", "Version": "version",
}


def _unquote(text: str) -> str:
    if len(text) < 2 or text[0] != '"' or text[-1] != '"':
        return text
    try:
        return json.loads(text)
    except Exception:
        return text


def looks_like_infotext(text: str) -> bool:
    if not text or text.lstrip().startswith(("{", "[")):
        return False
    lastline = text.strip().split("\n")[-1]
    return "Negative prompt:" in text or len(_RE_PARAM.findall(lastline)) >= 3


def parse_infotext(text: str, tool: str, detection: str = "marker") -> ParsedMetadata:
    """Parse an A1111-style infotext block into normalized metadata."""
    result = ParsedMetadata(tool=tool, raw=text, detection=detection)

    *lines, lastline = text.strip().split("\n")
    if len(_RE_PARAM.findall(lastline)) < 3:
        lines.append(lastline)
        lastline = ""

    prompt, negative, in_negative = [], [], False
    for line in lines:
        line = line.strip()
        if line.startswith("Negative prompt:"):
            in_negative = True
            line = line[len("Negative prompt:"):].strip()
        (negative if in_negative else prompt).append(line)
    result.positive = "\n".join(prompt).strip()
    result.negative = "\n".join(negative).strip()

    for key, value in _RE_PARAM.findall(lastline):
        key, value = key.strip(), _unquote(value.strip())
        if not key or value == "":
            continue
        set_param(result, _INFOTEXT_CANONICAL.get(key, key), value)
    return result


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class SwarmUIAdapter:
    """SwarmUI: JSON in the `parameters` chunk / EXIF, keyed sui_image_params."""
    tool = "SwarmUI"

    @staticmethod
    def _payload(raw: RawMetadata) -> Optional[str]:
        for candidate in (raw.text.get("parameters"), raw.user_comment, raw.exif_model):
            if candidate and "sui_image_params" in candidate:
                return candidate
        return None

    @classmethod
    def match(cls, raw: RawMetadata) -> bool:
        return cls._payload(raw) is not None

    @classmethod
    def parse(cls, raw: RawMetadata) -> Optional[ParsedMetadata]:
        return cls.parse_text(cls._payload(raw))

    @classmethod
    def parse_text(cls, payload: str, detection: str = "marker") -> Optional[ParsedMetadata]:
        try:
            obj = json.loads(payload)
            params = dict(obj["sui_image_params"])
        except Exception:
            return None
        result = ParsedMetadata(tool=cls.tool, raw=payload, detection=detection)
        result.positive = str(params.pop("prompt", "") or "").strip()
        result.negative = str(params.pop("negativeprompt", "") or "").strip()
        set_param(result, "model", params.pop("model", None))
        set_param(result, "seed", params.pop("seed", None))
        set_param(result, "steps", params.pop("steps", None))
        set_param(result, "cfg", params.pop("cfgscale", None))
        sampler = params.pop("comfyuisampler", None) or params.pop("autowebuisampler", None)
        set_param(result, "sampler", sampler)
        set_param(result, "scheduler", params.pop("comfyuischeduler", None))
        set_param(result, "version", params.pop("swarm_version", None))
        size = size_string(params.pop("width", None), params.pop("height", None))
        set_param(result, "size", size)
        for key, value in params.items():
            set_param(result, key, value)
        for key, value in (obj.get("sui_extra_data") or {}).items():
            set_param(result, key, value)
        models = obj.get("sui_models") or []
        if models:
            names = ", ".join(str(m.get("name", "?")) for m in models if isinstance(m, dict))
            set_param(result, "used models", names)
        return result


class FooocusAdapter:
    """Fooocus: `parameters` + `fooocus_scheme` chunks (or EXIF MakerNote scheme);
    legacy builds used a bare `Comment`/`comment` JSON."""
    tool = "Fooocus"
    _JSON_KEYS = {"prompt", "negative_prompt"}

    @staticmethod
    def _scheme_and_payload(raw: RawMetadata):
        scheme = raw.text.get("fooocus_scheme")
        if scheme:
            return scheme, raw.text.get("parameters")
        if raw.maker_note in ("fooocus", "a1111") and raw.user_comment:
            return raw.maker_note, raw.user_comment
        for key in ("Comment", "comment"):
            value = raw.text.get(key)
            if value:
                try:
                    obj = json.loads(value)
                except Exception:
                    continue
                if isinstance(obj, dict) and FooocusAdapter._JSON_KEYS <= set(obj):
                    return "fooocus", value
        return None, None

    @classmethod
    def match(cls, raw: RawMetadata) -> bool:
        # NovelAI also writes a Comment chunk; its Software marker is checked
        # earlier in the registry, so no clash here.
        scheme, payload = cls._scheme_and_payload(raw)
        return scheme is not None and bool(payload)

    @classmethod
    def parse(cls, raw: RawMetadata) -> Optional[ParsedMetadata]:
        scheme, payload = cls._scheme_and_payload(raw)
        if not payload:
            return None
        if scheme == "a1111" or looks_like_infotext(payload):
            return parse_infotext(payload, cls.tool)
        try:
            data = json.loads(payload)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        result = ParsedMetadata(tool=cls.tool, raw=payload, detection="marker")
        result.positive = str(data.pop("prompt", "") or "").strip()
        result.negative = str(data.pop("negative_prompt", "") or "").strip()
        set_param(result, "model", data.pop("base_model", None))
        set_param(result, "sampler", data.pop("sampler", None))
        set_param(result, "scheduler", data.pop("scheduler", None))
        set_param(result, "seed", data.pop("seed", None))
        set_param(result, "steps", data.pop("steps", None))
        set_param(result, "cfg", data.pop("guidance_scale", data.pop("cfg", None)))
        size = size_string(data.pop("width", None), data.pop("height", None))
        set_param(result, "size", size or data.pop("resolution", None))
        for key, value in data.items():
            set_param(result, key, value)
        return result


class InvokeAIAdapter:
    """InvokeAI: `invokeai_metadata` (v3+), `sd-metadata` (v2) or `Dream` (v1)."""
    tool = "InvokeAI"

    @classmethod
    def match(cls, raw: RawMetadata) -> bool:
        return any(k in raw.text for k in ("invokeai_metadata", "sd-metadata", "Dream"))

    @classmethod
    def parse(cls, raw: RawMetadata) -> Optional[ParsedMetadata]:
        if "invokeai_metadata" in raw.text:
            return cls._parse_v3(raw.text["invokeai_metadata"])
        if "sd-metadata" in raw.text:
            return cls._parse_v2(raw.text["sd-metadata"])
        return cls._parse_dream(raw.text.get("Dream", ""))

    @classmethod
    def _parse_v3(cls, payload: str) -> Optional[ParsedMetadata]:
        try:
            data = json.loads(payload)
        except Exception:
            return None
        result = ParsedMetadata(tool=cls.tool, raw=payload, detection="marker")
        result.positive = str(data.pop("positive_prompt", "") or "").strip()
        result.negative = str(data.pop("negative_prompt", "") or "").strip()
        model = data.pop("model", None)
        if isinstance(model, dict):
            model = model.get("model_name") or model.get("name")
        set_param(result, "model", model)
        set_param(result, "sampler", data.pop("scheduler", None))
        set_param(result, "seed", data.pop("seed", None))
        set_param(result, "steps", data.pop("steps", None))
        set_param(result, "cfg", data.pop("cfg_scale", None))
        set_param(result, "size", size_string(data.pop("width", None), data.pop("height", None)))
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                continue  # loras/controlnets etc. stay out of the flat report
            set_param(result, key, value)
        return result

    @classmethod
    def _parse_v2(cls, payload: str) -> Optional[ParsedMetadata]:
        try:
            data = json.loads(payload)
            image = data.pop("image")
        except Exception:
            return None
        prompt = image.pop("prompt", "")
        if isinstance(prompt, list) and prompt:
            prompt = prompt[0].get("prompt", "")
        result = ParsedMetadata(tool=cls.tool, raw=payload, detection="marker")
        result.positive, result.negative = _split_bracket_prompt(str(prompt))
        set_param(result, "model", data.pop("model_weights", None))
        set_param(result, "sampler", image.pop("sampler", None))
        set_param(result, "seed", image.pop("seed", None))
        set_param(result, "steps", image.pop("steps", None))
        set_param(result, "cfg", image.pop("cfg_scale", None))
        set_param(result, "size", size_string(image.pop("width", None), image.pop("height", None)))
        for source in (data, image):
            for key, value in source.items():
                if not isinstance(value, (dict, list)):
                    set_param(result, key, value)
        return result

    @classmethod
    def _parse_dream(cls, payload: str) -> Optional[ParsedMetadata]:
        match = re.search(r'"(.*?)"\s*(.*?)$', payload)
        if not match:
            return None
        prompt, flags = match.groups()
        result = ParsedMetadata(tool=cls.tool, raw=payload, detection="marker")
        result.positive, result.negative = _split_bracket_prompt(prompt.strip('" '))
        opts = dict(re.findall(r"-(\w+)\s+([\w.-]+)", flags))
        set_param(result, "sampler", opts.get("A"))
        set_param(result, "seed", opts.get("S"))
        set_param(result, "steps", opts.get("s"))
        set_param(result, "cfg", opts.get("C"))
        set_param(result, "size", size_string(opts.get("W"), opts.get("H")))
        return result


def _split_bracket_prompt(prompt: str):
    """InvokeAI legacy 'positive [negative]' prompt convention."""
    match = re.match(r"^(.*?)\[(.*?)\]$", prompt)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return prompt.strip(), ""


class NovelAIAdapter:
    """NovelAI: tEXt Software=NovelAI, Description + Comment JSON."""
    tool = "NovelAI"

    @classmethod
    def match(cls, raw: RawMetadata) -> bool:
        return raw.text.get("Software") == "NovelAI"

    @classmethod
    def parse(cls, raw: RawMetadata) -> Optional[ParsedMetadata]:
        comment = raw.text_json("Comment") or {}
        result = ParsedMetadata(
            tool=cls.tool, detection="marker",
            raw=raw.text.get("Comment") or raw.text.get("Description", ""),
        )
        result.positive = str(raw.text.get("Description", "") or "").strip()
        cls._apply_comment(result, comment)
        if not result.params.get("size"):
            set_param(result, "size", size_string(raw.width, raw.height))
        return result

    @classmethod
    def parse_stealth_json(cls, data: dict, payload: str) -> ParsedMetadata:
        result = ParsedMetadata(tool=cls.tool, raw=payload, detection="stealth")
        result.positive = str(data.pop("Description", "") or "").strip()
        comment = data.pop("Comment", None)
        if isinstance(comment, str):
            try:
                comment = json.loads(comment)
            except Exception:
                comment = {}
        cls._apply_comment(result, comment or {})
        for key, value in data.items():
            if not isinstance(value, (dict, list)):
                set_param(result, key, value)
        return result

    @staticmethod
    def _apply_comment(result: ParsedMetadata, comment: dict) -> None:
        if not isinstance(comment, dict):
            return
        comment = dict(comment)
        prompt = comment.pop("prompt", None)
        if prompt and not result.positive:
            result.positive = str(prompt).strip()
        result.negative = str(comment.pop("uc", "") or "").strip()
        set_param(result, "sampler", comment.pop("sampler", None))
        set_param(result, "seed", comment.pop("seed", None))
        set_param(result, "steps", comment.pop("steps", None))
        set_param(result, "cfg", comment.pop("scale", None))
        set_param(result, "size", size_string(comment.pop("width", None), comment.pop("height", None)))
        for key, value in comment.items():
            if not isinstance(value, (dict, list)):
                set_param(result, key, value)


class EasyDiffusionAdapter:
    """Easy Diffusion: individual PNG chunks, or one JSON UserComment."""
    tool = "Easy Diffusion"
    # spaced-key variant -> snake variant (older vs newer exports)
    _KEYS = {
        "Prompt": "prompt", "Negative Prompt": "negative_prompt",
        "Seed": "seed", "Stable Diffusion model": "use_stable_diffusion_model",
        "Sampler": "sampler_name", "Width": "width", "Height": "height",
        "Steps": "num_inference_steps", "Guidance Scale": "guidance_scale",
        "Clip Skip": "clip_skip", "VAE model": "use_vae_model",
    }

    @classmethod
    def match(cls, raw: RawMetadata) -> bool:
        return "negative_prompt" in raw.text or "Negative Prompt" in raw.text

    @classmethod
    def match_heuristic(cls, raw: RawMetadata) -> bool:
        data = _json_or_none(raw.user_comment)
        return isinstance(data, dict) and any(
            key in data for key in ("use_stable_diffusion_model", "num_inference_steps", "sampler_name")
        )

    @classmethod
    def parse(cls, raw: RawMetadata) -> Optional[ParsedMetadata]:
        data = _json_or_none(raw.user_comment)
        if isinstance(data, dict):
            return cls._parse_dict(data, raw.user_comment, "heuristic")
        return cls._parse_dict(dict(raw.text), json.dumps(raw.text), "marker")

    @classmethod
    def _parse_dict(cls, data: dict, payload: str, detection: str) -> ParsedMetadata:
        if "prompt" not in data:  # translate spaced keys to snake keys
            data = {cls._KEYS.get(k, k): v for k, v in data.items()}
        result = ParsedMetadata(tool=cls.tool, raw=payload, detection=detection)
        result.positive = str(data.pop("prompt", "") or "").strip()
        result.negative = str(data.pop("negative_prompt", "") or "").strip()
        model = data.pop("use_stable_diffusion_model", None)
        if model:
            model = re.split(r"[\\/]", str(model))[-1]
        set_param(result, "model", model)
        set_param(result, "sampler", data.pop("sampler_name", None))
        set_param(result, "seed", data.pop("seed", None))
        set_param(result, "steps", data.pop("num_inference_steps", None))
        set_param(result, "cfg", data.pop("guidance_scale", None))
        set_param(result, "size", size_string(data.pop("width", None), data.pop("height", None)))
        for key, value in data.items():
            if not isinstance(value, (dict, list)):
                set_param(result, key, value)
        return result


class DrawThingsAdapter:
    """Draw Things: JSON inside XMP exif:UserComment."""
    tool = "Draw Things"

    @classmethod
    def match(cls, raw: RawMetadata) -> bool:
        return bool(raw.xmp) and "exif:UserComment" in raw.xmp

    @classmethod
    def parse(cls, raw: RawMetadata) -> Optional[ParsedMetadata]:
        try:
            dom = minidom.parseString(raw.xmp)
            nodes = dom.getElementsByTagName("exif:UserComment")
            texts = []
            for node in nodes:
                stack = list(node.childNodes)
                while stack:
                    child = stack.pop(0)
                    if child.nodeType == child.TEXT_NODE and child.data.strip():
                        texts.append(child.data.strip())
                    stack.extend(getattr(child, "childNodes", []))
            data = json.loads(texts[0])
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        result = ParsedMetadata(tool=cls.tool, raw=json.dumps(data), detection="marker")
        result.positive = str(data.pop("c", "") or "").strip()
        result.negative = str(data.pop("uc", "") or "").strip()
        set_param(result, "model", data.pop("model", None))
        set_param(result, "sampler", data.pop("sampler", None))
        set_param(result, "seed", data.pop("seed", None))
        set_param(result, "steps", data.pop("steps", None))
        set_param(result, "cfg", data.pop("scale", None))
        set_param(result, "size", data.pop("size", None))
        for key, value in data.items():
            if not isinstance(value, (dict, list)):
                set_param(result, key, value)
        return result


class ComfyUIAdapter:
    """ComfyUI: `prompt` (API graph) / `workflow` (UI graph) JSON chunks,
    or the WebP EXIF Make/Model encoding. Workflow handling itself lives in
    the gallery's existing pipeline; this adapter only identifies the tool
    (and parses the infotext when a node also wrote an A1111-compatible
    `parameters` chunk)."""
    tool = "ComfyUI"

    @classmethod
    def match(cls, raw: RawMetadata) -> bool:
        workflow = raw.text_json("workflow")
        if isinstance(workflow, dict) and "nodes" in workflow:
            return True
        prompt = raw.text_json("prompt")
        if isinstance(prompt, dict) and any(
            isinstance(v, dict) and "class_type" in v for v in prompt.values()
        ):
            return True
        for tag in (raw.exif_make, raw.exif_model):
            if tag and (tag.startswith("workflow:{") or tag.startswith("prompt:{")):
                return True
        return False

    @classmethod
    def parse(cls, raw: RawMetadata) -> Optional[ParsedMetadata]:
        params = raw.text.get("parameters")
        if params and looks_like_infotext(params):
            result = parse_infotext(params, "ComfyUI (A1111-compatible)")
            return result
        payload = raw.text.get("workflow") or raw.text.get("prompt") or ""
        return ParsedMetadata(tool=cls.tool, raw=payload, detection="marker")


class A1111Adapter:
    """A1111 / Forge: infotext in the `parameters` chunk, EXIF UserComment,
    or a GIF comment."""
    tool = "A1111 / Forge"

    @staticmethod
    def _payload(raw: RawMetadata) -> Optional[str]:
        params = raw.text.get("parameters")
        if params and looks_like_infotext(params):
            return params
        if "postprocessing" in raw.text:
            return params or raw.text["postprocessing"]
        return None

    @classmethod
    def match(cls, raw: RawMetadata) -> bool:
        return cls._payload(raw) is not None

    @classmethod
    def match_heuristic(cls, raw: RawMetadata) -> bool:
        for candidate in (raw.user_comment, raw.gif_comment):
            if candidate and looks_like_infotext(candidate):
                return True
        return False

    @classmethod
    def parse(cls, raw: RawMetadata) -> Optional[ParsedMetadata]:
        payload = cls._payload(raw)
        detection = "marker"
        if payload is None:
            detection = "heuristic"
            for candidate in (raw.user_comment, raw.gif_comment):
                if candidate and looks_like_infotext(candidate):
                    payload = candidate
                    break
        if payload is None:
            return None
        result = parse_infotext(payload, cls.tool, detection)
        extra = raw.text.get("postprocessing")
        if extra and extra is not payload:
            set_param(result, "postprocessing", extra)
        return result


def _json_or_none(text: Optional[str]):
    if not text or not text.lstrip().startswith("{"):
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# Marker pass: unique-marker adapters first; ComfyUI before A1111 so that
# images carrying both a graph and an A1111-compatible infotext are labeled
# as ComfyUI output.
MARKER_ADAPTERS = (
    SwarmUIAdapter, NovelAIAdapter, FooocusAdapter, InvokeAIAdapter,
    EasyDiffusionAdapter, DrawThingsAdapter, ComfyUIAdapter, A1111Adapter,
)

# Heuristic pass: fall through by popularity.
HEURISTIC_ADAPTERS = (A1111Adapter, EasyDiffusionAdapter)


def parse_stealth_text(text: str) -> Optional[ParsedMetadata]:
    """Classify a decoded stealth payload by content."""
    if not text:
        return None
    data = _json_or_none(text)
    if data is not None:
        if "sui_image_params" in data:
            result = SwarmUIAdapter.parse_text(text, detection="stealth")
            if result:
                result.tool += " (stealth)"
            return result
        if "Comment" in data or "Description" in data:
            result = NovelAIAdapter.parse_stealth_json(data, text)
            result.tool += " (stealth)"
            return result
        return None
    if looks_like_infotext(text):
        return parse_infotext(text, "A1111 / Forge (stealth)", detection="stealth")
    return None


def parse_raw(raw: RawMetadata) -> Optional[ParsedMetadata]:
    for adapter in MARKER_ADAPTERS:
        try:
            if adapter.match(raw):
                result = adapter.parse(raw)
                if result is not None:
                    return result
        except Exception:
            continue
    for adapter in HEURISTIC_ADAPTERS:
        try:
            if adapter.match_heuristic(raw):
                result = adapter.parse(raw)
                if result is not None:
                    result.detection = "heuristic"
                    return result
        except Exception:
            continue
    stealth_text = raw.stealth()
    if stealth_text:
        return parse_stealth_text(stealth_text)
    return None


def parse_file(filepath: str, allow_stealth: bool = False) -> Optional[ParsedMetadata]:
    """Parse generation metadata from an image file.

    allow_stealth enables the LSB pixel scan (detail views); keep it off in
    bulk indexing paths. The scan is a second pass so images with regular
    metadata never pay for a pixel decode.
    """
    raw = load_raw(filepath, want_stealth=False)
    result = parse_raw(raw) if raw is not None else None
    if result is None and allow_stealth:
        raw = load_raw(filepath, want_stealth=True)
        if raw is not None:
            result = parse_raw(raw)
    return result
