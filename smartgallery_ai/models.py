"""Chat with a local model: optionally some images, a sequence of
questions, text back. Every model this project generates with loads here.

ONE loader for every checkpoint. The auto-classes resolve the architecture
from the model's own config, so Qwen3-VL, Phi-3.5-vision, SmolVLM, Gemma,
InternVL and text-only checkpoints all load through the same calls.
Choosing a model is a configuration string -- never a subclass, a
weights-filename table, or a per-model chat handler.

Vision or text is decided by the checkpoint, not by the caller: a config
registered in `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING` loads through
`AutoModelForImageTextToText`, anything else through `AutoModelForCausalLM`.
That mapping is the same registry `AutoModel` dispatches on
(auto_factory.py:209, `type(config) in cls._model_mapping`), so this
follows the library's own answer instead of guessing from the repo name or
try/except-ing one class against the other.

`model_ref` is either a directory under the models dir (offline, the
provisioned case) or a Hugging Face repo id. Weights are cached per
(model_ref, models_dir, device, attn): loading costs seconds and gigabytes
of VRAM, and the worker reviews thousands of files.

The unit of work is a `Chat` -- one conversation about a fixed set of
images. Images are encoded once, on the first turn, and every later turn
reuses their keys and values from the KV cache. A protocol that asks four
questions about one image therefore pays for one vision encode, not four.
A Chat with no images is a plain text conversation over the same path.

Upstream pattern, followed deliberately (transformers v5.15.0):
  - iterative generation passes the FULL token sequence plus the cache and
    lets `generate` skip the cached prefix (kv_cache.md:258-260). It does
    NOT pass only the new tokens: with a populated cache that slices to
    zero and the model raises "cannot reshape tensor of 0 elements".
    Measured here on Qwen3-VL, transformers 5.15.0.
  - only the text is re-rendered per turn; the image is not re-processed
    (tasks/image_text_to_text.md:335-404). We keep the processor's own
    expanded ids from turn one and append new text ids to them, so the
    image placeholders stay expanded without touching the vision tower.
  - inputs.pop("token_type_ids")   qwen3_vl.md:67; some processors emit a
                                   key generate() will not accept
  - clean_up_tokenization_spaces=False   on decode, per the model card
  - a system turn only counts at messages[0] -- the templates test
    `messages[0].role == 'system'` and silently ignore it anywhere else

`dtype` is not passed: `from_pretrained` already reads it from the
checkpoint's config and falls back to the first floating-point weight
(models.md:233). Passing "auto" restates the default.

ONE deviation from the Qwen3-VL model card: it publishes sampling defaults
(temperature 0.7, top_p 0.8). This decodes greedily instead, because a
review is cached against its (file, model, prompt) key -- the same image
must yield the same answer or staleness detection and re-review churn
forever.

Structured output goes through the model's own tool-calling contract:
nothing in transformers constrains decoding to a JSON schema, but the chat
templates render `tools` into <tools></tools> and ask for
<tool_call>{"name":..., "arguments":...}</tool_call>. Tools are fixed for
the life of a Chat because they render into the system block at position
zero -- changing them mid-conversation rewrites the prefix the cache is
built on. Callers validate the parsed payload regardless, so this only has
to yield well-formed JSON, not correct JSON.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import threading
from collections.abc import Sequence
from typing import Any

_logger = logging.getLogger(__name__)

#: (model_ref, models_dir, device, attn) -> (processor, model)
_cache: dict = {}

#: Guards `_cache` AND every forward through a cached model.
#:
#: One entry is shared by the worker's review stage and runner's interactive
#: run, which are different threads. runner._RUN_LOCK excludes two
#: interactive runs from each other; it does not exclude the worker. So two
#: generate() calls could interleave on one HF model, each with its own
#: DynamicCache, on weights that keep no per-call state of their own but are
#: driven through one set of module buffers. embedders.OpenClipSemanticEmbedder
#: already carries `_infer_lock` for exactly this reason, and
#: backends.serializes_internally refuses to lend out anything without it --
#: the VLM simply had no equivalent.
#:
#: Reentrant because ask_json retries call ask() while already holding it.
_MODEL_LOCK = threading.RLock()

#: Vision tokens per image. The processor downsamples to fit rather than us
#: resizing the source, so the model still sees the whole frame. 576 is a
#: 768x768-equivalent budget; the checkpoint default is often far larger
#: (Qwen3-VL ships 16384, i.e. every image encoded at up to 4096x4096).
DEFAULT_MAX_VISION_TOKENS = 576


class ModelUnavailable(RuntimeError):
    """The transformers runtime or a checkpoint's weights are absent.

    Raised instead of ImportError/OSError so callers can report the
    capability as off rather than crashing the worker.
    """


def resolve_device(explicit: str = "") -> str:
    """The torch device to place a model on: AI_DAM_DEVICE when set
    ('cpu', 'cuda', 'cuda:N'), else the GPU with the most free memory, else
    CPU. One rule for every model in the process."""
    device = (explicit or os.environ.get("AI_DAM_DEVICE", "")).strip().lower()
    if device:
        return device
    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            return "cpu"
        best, best_free = 0, -1
        for index in range(torch.cuda.device_count()):
            free = torch.cuda.mem_get_info(index)[0]
            if free > best_free:
                best, best_free = index, free
    except Exception:
        _logger.debug("handled a failure in resolve_device", exc_info=True)
        return "cpu"
    else:
        return f"cuda:{best}"


def resolve_attn(explicit: str = "") -> str | None:
    """The attention backend, or None to let transformers choose (sdpa).

    AI_DAM_ATTN takes a backend name. 'kernels-community/flash-attn2'
    fetches a prebuilt FlashAttention kernel from the Hub at load time,
    which is the only way to get FlashAttention on a machine with no
    compiler -- the pip package ships no Windows wheels
    (attention_interface.md:55-66).
    """
    return (explicit or os.environ.get("AI_DAM_ATTN", "")).strip() or None


def _weights_location(model_ref: str, models_dir: str) -> tuple:
    """`(location, local_only)`. A directory under `models_dir` is loaded
    offline; anything else is treated as a Hugging Face repo id."""
    if models_dir:
        for candidate in (
            os.path.join(models_dir, *model_ref.split("/")),
            os.path.join(models_dir, model_ref.rsplit("/", maxsplit=1)[-1]),
        ):
            if os.path.isdir(candidate):
                return candidate, True
    return model_ref, False


def is_provisioned(model_ref: str, models_dir: str) -> bool:
    """True when `model_ref` resolves to a directory under `models_dir`.

    Lets a capability answer "am I available?" without loading gigabytes or
    touching the network. A hub id that was never provisioned reads as
    False even though `load` could still fetch it -- availability here
    means "ready now", not "obtainable".
    """
    return _weights_location(model_ref, models_dir)[1]


def load(model_ref: str, models_dir: str = "", device: str = "", attn: str = "") -> tuple:
    """`(processor, model)` for `model_ref`, cached per device and backend."""
    device, attn_impl = resolve_device(device), resolve_attn(attn)
    key = (model_ref, models_dir, device, attn_impl)
    with _MODEL_LOCK:
        if key in _cache:
            return _cache[key]
    try:
        # torchvision BEFORE transformers: transformers freezes its
        # torchvision-availability flag at first import and the image
        # processors hard-require it, so importing it later leaves the
        # vision backends dead until the process restarts.
        importlib.import_module("torch")
        importlib.import_module("torchvision")
        transformers = importlib.import_module("transformers")
    except Exception as exc:
        raise ModelUnavailable(f"transformers runtime unavailable: {exc}") from exc

    location, provisioned = _weights_location(model_ref, models_dir)
    # ALWAYS offline. Loading a backend is not a provisioning action: a
    # deployment configured for no egress must not reach the Hub because a
    # checkpoint happens to be absent, and an unpinned fetch at load time
    # would decide which weights run without a revision or a digest saying
    # which ones those are. `local_files_only=True` still resolves a hub id
    # out of the local huggingface cache when it is already there
    # (transformers/utils/hub.py: "will only try to load ... from local
    # files"), so a previously fetched model keeps working; one that was
    # never fetched raises ModelUnavailable instead of downloading.
    options: dict = {"local_files_only": True}
    if attn_impl:
        options["attn_implementation"] = attn_impl
    try:
        config = transformers.AutoConfig.from_pretrained(location, local_files_only=True)
        vision = type(config) in transformers.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING
        # AutoProcessor wraps a tokenizer plus the image processor; a
        # text-only checkpoint has no processor to wrap, so it loads its
        # tokenizer directly. Chat only ever needs apply_chat_template,
        # batch_decode and a callable -- both satisfy that.
        loader = transformers.AutoProcessor if vision else transformers.AutoTokenizer
        processor = loader.from_pretrained(location, local_files_only=True)
        auto_model = transformers.AutoModelForImageTextToText if vision else transformers.AutoModelForCausalLM
        model = auto_model.from_pretrained(location, **options)
        model.to(device)
        model.eval()
    except Exception as exc:
        raise ModelUnavailable(f"cannot load weights '{model_ref}' from {location}: {exc}") from exc
    _logger.info(
        "[AI] %s model %s on %s (%s, attn=%s)",
        "vision-language" if vision else "text",
        model_ref,
        device,
        "models_dir" if provisioned else "huggingface cache",
        attn_impl or "default",
    )
    # Re-check under the lock: two threads that both missed above would
    # otherwise each load the full checkpoint and the loser's copy would sit
    # in VRAM with nothing referencing it. Loading stays OUTSIDE the lock --
    # it takes minutes and would block every other model's forwards.
    with _MODEL_LOCK:
        return _cache.setdefault(key, (processor, model))


def vision_budget(processor, max_vision_tokens: int) -> dict:
    """Image-processor kwargs that cap one image at `max_vision_tokens`.

    Derived from the processor's OWN declared geometry, not a per-model
    table: a patch of `patch_size` pixels a side, merged `merge_size` to a
    side, is one token. Processors in this family express `size` as pixel
    COUNTS (Qwen3-VL ships shortest_edge=65536, longest_edge=16777216),
    which is why the budget can be written as an area at all.

    Returns `{}` when the processor doesn't declare that geometry -- the
    caller then gets the checkpoint default, which is correct but may be
    expensive. Warns rather than guessing.

    Do NOT reach for `max_pixels` here. It is accepted and then dropped
    unless `min_pixels` is passed with it (qwen2_vl/image_processing_qwen2_vl.py:127),
    so passing it alone silently changes nothing -- measured: 2040 image
    tokens with it, 2040 without, 576 with `size`.
    """
    image_processor = getattr(processor, "image_processor", None)
    patch = getattr(image_processor, "patch_size", None)
    merge = getattr(image_processor, "merge_size", None)
    size = getattr(image_processor, "size", None)
    longest = getattr(size, "longest_edge", None) if size is not None else None
    if not (patch and merge and longest):
        _logger.warning(
            "[AI] %s does not declare patch/merge geometry; vision budget of "
            "%d tokens not applied, using the checkpoint default",
            type(image_processor).__name__,
            max_vision_tokens,
        )
        return {}
    per_token = (patch * merge) ** 2
    ceiling = max_vision_tokens * per_token
    floor = getattr(size, "shortest_edge", None) or per_token
    return {"size": {"shortest_edge": min(floor, ceiling), "longest_edge": ceiling}}


class Chat:
    """A conversation about a fixed set of images.

    The images are encoded on the first `ask`; every later turn reuses
    their keys and values. Build one per subject (per file under review),
    ask it as many questions as the protocol needs, then drop it.

    `tools` is fixed for the life of the conversation on purpose -- see the
    module docstring.
    """

    def __init__(
        self,
        model_ref: str,
        images: Sequence = (),
        *,
        models_dir: str = "",
        device: str = "",
        attn: str = "",
        system: str | None = None,
        tools: list | None = None,
        max_vision_tokens: int = DEFAULT_MAX_VISION_TOKENS,
    ):
        self._processor, self._model = load(model_ref, models_dir, device, attn)
        # AutoProcessor nests its tokenizer; AutoTokenizer IS one.
        self._tokenizer = getattr(self._processor, "tokenizer", self._processor)
        # Annotated loose because tests substitute a stand-in for all three.
        self._torch: Any = importlib.import_module("torch")
        self._images = [image.convert("RGB") for image in images]
        self._tools = list(tools) if tools else None
        self._image_kwargs = vision_budget(self._processor, max_vision_tokens) if self._images else {}
        self._messages: list = []
        if system:
            self._messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
        self._cache = None
        self._ids = None

    def _render(self, messages: list, generation_prompt: bool) -> str:
        return self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation_prompt, tools=self._tools
        )

    def ask(self, prompt: str, max_new_tokens: int = 256) -> str:
        """One greedy turn. Returns the reply text."""
        # The image placeholders belong to the FIRST user turn -- that is
        # where the template expands them and where the vision tower runs.
        first = self._ids is None
        content: list = []
        if first:
            content += [{"type": "image"} for _ in self._images]
        content.append({"type": "text", "text": prompt})
        cached_text = "" if first else self._render(self._messages, False)
        self._messages.append({"role": "user", "content": content})
        full_text = self._render(self._messages, True)

        if first:
            inputs = self._processor(
                text=full_text, images=self._images or None, return_tensors="pt", **self._image_kwargs
            )
            inputs.pop("token_type_ids", None)
            inputs = inputs.to(self._model.device)
            prompt_length = inputs["input_ids"].shape[1]
            cache_module = importlib.import_module("transformers.cache_utils")
            self._cache = cache_module.DynamicCache()
            call = dict(inputs)
        else:
            # Append only the new text to the ids we already hold. Those ids
            # came out of the processor with the image placeholders already
            # expanded, so the sequence stays aligned with the cache without
            # the vision tower running again.
            delta = self._tokenizer(full_text[len(cached_text) :], add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ].to(self._model.device)
            input_ids = self._torch.cat([self._ids, delta], dim=-1)
            prompt_length = input_ids.shape[1]
            call = {"input_ids": input_ids}

        with _MODEL_LOCK, self._torch.no_grad():
            generated = self._model.generate(
                **call, past_key_values=self._cache, do_sample=False, max_new_tokens=max_new_tokens
            )
        self._ids = generated
        reply = self._processor.batch_decode(
            generated[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        self._messages.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
        return reply

    def ask_json(self, prompt: str, name: str = "", max_new_tokens: int = 512, attempts: int = 2) -> Any:
        """Like `ask`, but returns the parsed arguments of the tool call.

        Requires `tools` to have been declared on the Chat. `name` demands
        that specific tool: with several declared the model picks one, and
        a protocol step that silently accepted a different tool's arguments
        would parse a reply to a question it never asked.

        Retries a bounded number of times. Decoding is not schema
        constrained -- nothing in transformers constrains generation to a
        JSON schema -- so a malformed reply is expected occasionally rather
        than exceptional.
        """
        if not self._tools:
            raise ValueError(
                "ask_json needs tools declared on the Chat: they render into "
                "the system block and cannot be introduced mid-conversation"
            )
        if name and not any(t["function"]["name"] == name for t in self._tools):
            raise ValueError(f"tool {name!r} was not declared on this Chat")
        failure: Exception = ValueError("no attempts made")
        for attempt in range(max(1, attempts)):
            reply = self.ask(prompt, max_new_tokens=max_new_tokens)
            try:
                return tool_arguments(reply, expect=name)
            except ValueError as exc:
                failure = exc
                _logger.debug("[AI] unusable reply (attempt %d/%d): %s", attempt + 1, attempts, reply[:200])
                prompt = (
                    f"That was not a call to {name}. Answer again as a single <tool_call> block calling {name}."
                    if name
                    else "That was not a tool call. Answer again as a single <tool_call> block."
                )
        raise failure


def tool(name: str, description: str, schema: dict) -> dict:
    """A tool definition in the JSON-schema shape every chat template
    expects (chat_templating_writing.md:210-234)."""
    return {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}


def extract_json_object(text: str) -> Any:
    """The first balanced JSON object in `text`.

    String-aware: braces inside string literals do not change the depth, so
    a description containing '{' cannot truncate the object. Raises
    ValueError when there is none.
    """
    start = text.find("{")
    while start != -1:
        depth, in_string, escaped = 0, False, False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError(f"no JSON object in model output: {text[:200]!r}")


def tool_arguments(text: str, expect: str = "") -> Any:
    """The `arguments` object from a <tool_call> reply.

    `expect` names the tool the caller asked for; a call to a different
    tool raises ValueError rather than handing back arguments shaped for
    another question.

    Falls back to the first balanced JSON object when the model answers
    with bare JSON instead of a tool call -- both shapes are common, and
    the caller validates the payload either way. A bare object carries no
    name, so `expect` cannot be checked against it.
    """
    call = text
    opening = text.find("<tool_call>")
    if opening != -1:
        closing = text.find("</tool_call>", opening)
        call = text[opening + len("<tool_call>") : closing if closing != -1 else len(text)]
    obj = extract_json_object(call)
    if isinstance(obj, dict) and "arguments" in obj:
        called = obj.get("name")
        if expect and called is not None and called != expect:
            raise ValueError(f"model called {called!r}, not {expect!r}")
        arguments = obj["arguments"]
        if isinstance(arguments, str):  # some models stringify them
            return extract_json_object(arguments)
        return arguments
    return obj


def unload() -> None:
    """Drop every cached model and release its VRAM.

    clear() alone only dropped the cache ENTRY. Live Chat objects hold
    self._model, and torch's caching allocator holds the blocks regardless,
    so "unload" left the weights resident and a reload doubled VRAM -- on a
    box already sharing the card with ComfyUI. Emptying the allocator cache
    is what actually returns it; it is a no-op without CUDA.
    """
    with _MODEL_LOCK:
        _cache.clear()
    try:
        torch = importlib.import_module("torch")
    except Exception:
        _logger.debug("handled a failure in unload", exc_info=True)
        return
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
