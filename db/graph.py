"""Reading a ComfyUI graph into the recipe it describes.

This is a ComfyUI gallery whose recipe axis was empty for ComfyUI. Every
other tool writes its settings as text and `metaparse` reads them; ComfyUI
writes a node graph, and metaparse says so in its own adapter -- "Workflow
handling itself lives in the gallery's existing pipeline; this adapter only
identifies the tool" (metaparse/adapters.py:474-479). The greenfield layer
had no such pipeline, so a ComfyUI picture arrived with `tool='ComfyUI'` and
NULL seed, steps, cfg, sampler, model and prompt -- no checkpoint row, no
LoRA rows, nothing on the model page, nothing for LoRA synergy to join.

The graph is the API `prompt` chunk: `{node_id: {class_type, inputs}}`, where
an input is either a literal or a link `[node_id, output_index]` -- a two-item
list whose first item is a string and second a number
(refs/Comfy-Org/ComfyUI/comfy_execution/graph_utils.py:1-10).

**Walked backwards from the picture, not forwards from the first sampler.**
A workflow routinely holds several samplers -- a base and a refiner, a
generate and an upscale pass -- and the one that made the file is the one
whose latent reaches the node that saved it. Taking the first sampler found
reports the settings of a pass whose output was thrown away, which is worse
than reporting nothing because it looks like an answer.

Node names come from the nodes themselves, not from memory:
CheckpointLoaderSimple takes `ckpt_name` (nodes.py:616-623), UNETLoader takes
`unet_name` (:982-987), LoraLoader takes `lora_name`, `strength_model` and
`strength_clip` (:709-725), KSampler takes seed/steps/cfg/sampler_name/
scheduler/positive/negative/latent_image/denoise (:1596-1611), KSamplerAdvanced
spells its seed `noise_seed` (:1625-1641), EmptyLatentImage takes width and
height (:1245-1254), SaveImage takes `images` (:1659-1672) and VAEDecode takes
`samples` (:316-330).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

#: Nodes that end a workflow by writing or showing the picture.
_OUTPUTS = ("SaveImage", "PreviewImage", "SaveAnimatedWEBP", "SaveAnimatedPNG", "SaveWEBM")

#: How each sampler spells its seed. Everything else it carries is spelled
#: the same across them.
_SEED_INPUT = ("seed", "noise_seed")

#: Where a checkpoint or diffusion model names itself.
_MODEL_INPUT = ("ckpt_name", "unet_name", "model_name", "model_path")

#: Inputs to follow when walking back towards the model through a chain of
#: patchers -- LoRAs, ControlNets, samplers-with-model-inputs.
_UPSTREAM_MODEL = ("model", "unet")


@dataclass
class Recipe:
    """What a graph says was asked for."""

    seed: int | None = None
    steps: int | None = None
    cfg: float | None = None
    denoise: float | None = None
    sampler: str | None = None
    scheduler: str | None = None
    width: int | None = None
    height: int | None = None
    model: str | None = None
    positive: str = ""
    negative: str = ""
    #: (name, model_weight, clip_weight), in the order the chain applies them.
    loras: list[tuple[str, float | None, float | None]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (
            self.model
            or self.positive
            or self.negative
            or self.loras
            or self.seed is not None
            or self.steps is not None
        )


def _linked(value):
    """`(node_id, slot)` a value points at, or None if it is a literal.

    Matches ComfyUI's own `is_link`: a two-item list, string then number.

    The SLOT is kept, not discarded. A node that conditions both prompts --
    ControlNetApplyAdvanced and the rest of `_BOTH_SIDES` -- returns the
    positive chain on slot 0 and the negative chain on slot 1, so the slot
    is the only thing that says which prompt a link carries. Dropping it
    and following the first conditioning input reports a negative prompt as
    the positive one: a wrong answer that looks like an answer.
    """
    if (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], (int, float))
        and not isinstance(value[1], bool)
    ):
        return value[0], int(value[1])
    return None


def _link(value):
    """The node id a value points at, or None if it is a literal."""
    held = _linked(value)
    return None if held is None else held[0]


#: Inputs that carry a conditioning onward, on nodes that take exactly one
#: chain. Following any of them cannot confuse the two prompts.
_ONE_SIDE: tuple[str, ...] = (
    "conditioning",
    "conditioning_1",
    "conditioning_2",
    "conditioning_to",
    "conditioning_from",
    "cond",
)

#: Inputs on a node that conditions BOTH prompts. Which one to follow is
#: decided by the slot the link arrived on, never by order.
_BOTH_SIDES: tuple[str, str] = ("positive", "negative")

#: Nodes that DISCARD the words they are given. Walking through one reports
#: the text it erased, which is worse than reporting nothing.
#:
#: `ConditioningZeroOut` is how a workflow says it has no negative prompt: it
#: takes the POSITIVE conditioning and zeroes it, so the link exists and the
#: text behind it is real, and following it put the positive prompt in the
#: negative field of three ComfyUI_examples workflows -- sd3_anime_example,
#: sd3_controlnet_example, sd3.5_large_canny_controlnet_example.
_ERASERS: frozenset[str] = frozenset({"ConditioningZeroOut"})


class _Graph:
    """One API graph, with the walks the recipe needs."""

    def __init__(self, nodes: dict):
        self.nodes = {
            key: value
            for key, value in nodes.items()
            if isinstance(value, dict) and isinstance(value.get("inputs"), dict)
        }

    def kind(self, node_id) -> str:
        return str((self.nodes.get(node_id) or {}).get("class_type") or "")

    def inputs(self, node_id) -> dict:
        return (self.nodes.get(node_id) or {}).get("inputs") or {}

    def value(self, node_id, *names):
        """The first of `names` this node holds as a literal."""
        held = self.inputs(node_id)
        for name in names:
            if name in held and _link(held[name]) is None:
                return held[name]
        return None

    def follow(self, node_id, name):
        """The node one of this node's inputs comes from, or None."""
        return _link(self.inputs(node_id).get(name))

    def back(self, start, wanted, seen=None):
        """Walk upstream from `start` to the first node `wanted` accepts.

        Every walk here is guarded: a graph is meant to be acyclic and a
        malformed one is still a file somebody has in their library, so a
        cycle must end the walk rather than the scan.
        """
        seen = seen if seen is not None else set()
        if start is None or start in seen:
            return None
        seen.add(start)
        if wanted(self.kind(start)):
            return start
        for value in self.inputs(start).values():
            upstream = _link(value)
            if upstream is None:
                continue
            found = self.back(upstream, wanted, seen)
            if found is not None:
                return found
        return None

    def text_of(self, node_id, seen=None, slot: int = 0) -> str:
        """The prompt a conditioning node encodes.

        Followed one link at a time rather than read directly: a workflow
        that routes its prompt through a primitive, a concat or a wildcard
        node has the words a node or two upstream, and reading only the
        literal reports an empty prompt for the workflows most likely to
        have an interesting one.

        A conditioning PASS-THROUGH is followed too. Reading only the text
        inputs stopped at the first ControlNetApply, unCLIPConditioning,
        ConditioningCombine or FluxGuidance in the way, because none of them
        has an input named `text` -- and reported no positive prompt for a
        workflow that plainly had one. Measured over the 92 prompt-bearing
        files of comfyanonymous/ComfyUI_examples@f9431bb000ce: 22 returned a
        negative prompt and no positive, which no real workflow does.

        `slot` is the output the caller arrived through, and it is what
        keeps the two prompts apart. A node holding both `positive` and
        `negative` emits the positive chain on slot 0 and the negative on
        slot 1, so the slot picks the input to follow. Following the first
        conditioning input instead would report a negative prompt as the
        positive one.
        """
        seen = seen if seen is not None else set()
        if node_id is None or node_id in seen:
            return ""
        seen.add(node_id)
        if self.kind(node_id) in _ERASERS:
            return ""
        held = self.inputs(node_id)
        for name in ("text", "text_g", "string", "value", "prompt", "populated_text"):
            value = held.get(name)
            if value is None:
                continue
            onward = _linked(value)
            if onward is None:
                if isinstance(value, str) and value.strip():
                    return value.strip()
            else:
                found = self.text_of(onward[0], seen, onward[1])
                if found:
                    return found

        # No words here: this is a pass-through. Which way depends on
        # whether the node carries one chain or both.
        if all(name in held for name in _BOTH_SIDES):
            side = _BOTH_SIDES[1] if slot == 1 else _BOTH_SIDES[0]
            onward = _linked(held.get(side))
            if onward is not None:
                return self.text_of(onward[0], seen, onward[1])
            return ""
        for name in _ONE_SIDE:
            onward = _linked(held.get(name))
            if onward is None:
                continue
            found = self.text_of(onward[0], seen, onward[1])
            if found:
                return found
        return ""


def _samplers(graph: _Graph) -> list[str]:
    return [
        node_id
        for node_id in graph.nodes
        if "sampler" in graph.kind(node_id).lower()
        # `guider` is here because a custom sampler has none of the others:
        # SamplerCustomAdvanced takes noise, guider, sampler, sigmas and a
        # latent, so this listed it as not-a-sampler and a graph without a
        # SaveImage node to walk back from lost it entirely.
        and any(name in graph.inputs(node_id) for name in ("positive", "steps", "noise_seed", "seed", "guider"))
    ]


def _final_sampler(graph: _Graph) -> str | None:
    """The sampler whose work reached the saved picture.

    A workflow with a refiner or an upscale pass holds several. Reporting the
    first one found describes a pass whose output was discarded, which is
    worse than reporting nothing because it looks like an answer.
    """
    for node_id in graph.nodes:
        if graph.kind(node_id) in _OUTPUTS:
            found = graph.back(node_id, lambda kind: "sampler" in kind.lower())
            if found is not None:
                return found
    known = _samplers(graph)
    return known[-1] if known else None


def read(payload) -> Recipe | None:
    """The recipe a ComfyUI API graph describes, or None if it is not one."""
    if isinstance(payload, (str, bytes)):
        try:
            payload = json.loads(payload)
        except ValueError:
            return None
    if not isinstance(payload, dict) or not payload:
        return None
    graph = _Graph(payload)
    if not graph.nodes:
        return None
    if not any("class_type" in (node or {}) for node in payload.values() if isinstance(node, dict)):
        return None

    out = Recipe()
    sampler = _final_sampler(graph)
    if sampler is not None:
        seed = graph.value(sampler, *_SEED_INPUT)
        out.seed = int(seed) if isinstance(seed, (int, float)) and not isinstance(seed, bool) else None
        steps = graph.value(sampler, "steps")
        out.steps = int(steps) if isinstance(steps, (int, float)) else None
        cfg = graph.value(sampler, "cfg", "guidance")
        out.cfg = float(cfg) if isinstance(cfg, (int, float)) else None
        denoise = graph.value(sampler, "denoise")
        out.denoise = float(denoise) if isinstance(denoise, (int, float)) else None
        name = graph.value(sampler, "sampler_name")
        out.sampler = name if isinstance(name, str) else None
        scheduler = graph.value(sampler, "scheduler")
        out.scheduler = scheduler if isinstance(scheduler, str) else None

        # The slot each link arrives on rides with it: a node conditioning
        # both prompts is told which one is being asked for.
        held = graph.inputs(sampler)
        for side in _BOTH_SIDES:
            onward = _linked(held.get(side))
            if onward is not None:
                setattr(out, side, graph.text_of(onward[0], None, onward[1]))

        # A custom sampler takes no prompt at all: it takes a GUIDER, and the
        # conditioning hangs off that. Every flux workflow in
        # comfyanonymous/ComfyUI_examples@f9431bb000ce is shaped this way --
        # SamplerCustomAdvanced -> BasicGuider -> FluxGuidance ->
        # CLIPTextEncode -- and reading only `positive` reported no prompt
        # for 11 of them.
        #
        # A BasicGuider holds ONE chain, and it is the positive one. Asking
        # it for a negative would hand back the positive prompt, so the
        # negative is only read from a guider that has one (CFGGuider).
        if not out.positive and not out.negative:
            guider = _linked(held.get("guider"))
            if guider is not None:
                out.positive = graph.text_of(guider[0], None, 0)
                if "negative" in graph.inputs(guider[0]):
                    out.negative = graph.text_of(guider[0], None, 1)

        latent = graph.back(
            graph.follow(sampler, "latent_image"),
            lambda kind: "latent" in kind.lower() or "image" in kind.lower(),
        )
        if latent is not None:
            width, height = graph.value(latent, "width"), graph.value(latent, "height")
            out.width = int(width) if isinstance(width, (int, float)) else None
            out.height = int(height) if isinstance(height, (int, float)) else None

    # The model chain, from the sampler back through every patcher applied to
    # it. Order matters: it is the order the LoRAs were stacked.
    start = graph.follow(sampler, "model") if sampler is not None else None
    if start is None:
        start = next((n for n in graph.nodes if graph.value(n, *_MODEL_INPUT) is not None), None)
    seen: set[str] = set()
    node_id = start
    while node_id is not None and node_id not in seen:
        seen.add(node_id)
        named = graph.value(node_id, *_MODEL_INPUT)
        kind = graph.kind(node_id)
        if "lora" in kind.lower():
            lora = graph.value(node_id, "lora_name")
            if isinstance(lora, str) and lora.strip():
                model_weight = graph.value(node_id, "strength_model", "strength")
                clip_weight = graph.value(node_id, "strength_clip", "strength")
                out.loras.append(
                    (
                        lora.strip(),
                        float(model_weight) if isinstance(model_weight, (int, float)) else None,
                        float(clip_weight) if isinstance(clip_weight, (int, float)) else None,
                    )
                )
        elif isinstance(named, str) and named.strip() and out.model is None:
            out.model = named.strip()
        node_id = next(
            (graph.follow(node_id, name) for name in _UPSTREAM_MODEL if graph.follow(node_id, name) is not None),
            None,
        )

    # Collected walking backwards, so reverse to the order they were applied.
    out.loras.reverse()
    return None if out.is_empty else out
