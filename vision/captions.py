"""Captions: one sentence a vision-language model says about a picture.

The annotate job is the one caller that may provision
(docs/AI_MODELS.md); everything else loads from disk or refuses by
name. Lookup is two-deep, the run's models_dir then the shared HF
cache, and a snapshot counts only when every file loading touches is
present -- weights alone would fail halfway into from_pretrained.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Protocol, cast

from PIL import Image

from vision.weights import Unprovisioned, hub_cached

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from transformers import BatchFeature

#: The default `caption_model` setting: BLIP base, captioning head, BSD-3.
MODEL = "Salesforce/blip-image-captioning-base"
REVISION = "main"
#: Tokens a caption may run to (generate's max_new_tokens); BLIP
#: captions are one sentence.
MOST_TOKENS = 40

#: How many pictures go through the model at once.
#:
#: Read off benchmarks/results/caption_batch.json, 48 real pictures on a
#: 3070 Ti (`just bench captions` regenerates it): 3.62 pictures/sec at
#: one, 12.81 at eight, 15.72 at sixteen, every float32 row reporting
#: same_as_baseline 1.0 -- identical captions to the ones captioned alone.
#: Sixteen is where the curve flattens; past it the batch is mostly a
#: longer wait for whoever asked the job to stop, since cancellation is
#: checked between items and a batch runs inside one.
BATCH = 16
_WEIGHTS = ("model.safetensors", "pytorch_model.bin")
_SNAPSHOT_FILES = ("config.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.txt")


class Captioner(Protocol):
    model_id: str
    model_version: str

    def describe(self, image: Image.Image) -> str: ...

    def describe_many(self, images: Sequence[Image.Image]) -> list[str]:
        """Several pictures in one pass, in order. `describe` is the
        batch of one, so a caller with several never has a reason to
        loop."""
        ...


class CaptionerUnavailable(LookupError):
    """The runtime does not import here. A LookupError, so the job
    records it on the item by name (db/runner.py ITEM_FAILURES)."""


def _cached_snapshot(models_dir: str, model: str, revision: str = REVISION) -> str | None:
    """The local snapshot directory holding this model COMPLETE, or
    None. Disk only."""
    held = None
    for name in _WEIGHTS:
        held = hub_cached(model, name, models_dir, revision=revision)
        if held is not None:
            break
    if held is None:
        return None
    if any(hub_cached(model, name, models_dir, revision=revision) is None for name in _SNAPSHOT_FILES):
        return None
    return str(pathlib.Path(held).parent)


class BlipCaptioner:
    """One loaded BLIP captioning model; a PIL image in, a sentence out."""

    def __init__(self, models_dir: str, model: str = MODEL, *, provision: bool = False):
        found = _cached_snapshot(models_dir, model)
        if found is None and not provision:
            raise Unprovisioned(f"{model} is not under {models_dir} or the shared HF cache; run /jobs/annotate once")
        try:
            import torch
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except (ImportError, OSError) as why:  # OSError: a native dependency failed to load
            raise CaptionerUnavailable(f"the transformers runtime does not import here: {why}") from why
        # A resolved snapshot loads by directory path -- no hub call can
        # hide in a local-path load; only provisioning passes the repo id.
        source = found if found is not None else model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(source, cache_dir=models_dir, revision=REVISION)
        loaded = BlipForConditionalGeneration.from_pretrained(source, cache_dir=models_dir, revision=REVISION)
        loaded.eval()
        # Unbound, and the return value is discarded. Two separate facts:
        #
        # `loaded.to(...)` does not type check, because transformers
        # decorates `Module.to` with a functools wrapper whose `__call__`
        # still wants the model as its first argument -- the descriptor
        # never binds, so the device string lands in the `self` slot.
        #
        # And the return has to go unused, because taking it costs the
        # concrete class: `Module.to` is annotated to return Self, ty
        # resolves that through the unbound call and pyrefly does not, so
        # the assigned name becomes plain `Module` for one of the two and
        # `self.model.generate` stops existing. Both calls mutate the
        # module in place and hand back the same object, so `loaded` is
        # already the moved model and keeps the type it was loaded with.
        torch.nn.Module.to(loaded, self.device)
        # Half precision on CUDA only; on a CPU most of these kernels are
        # emulated and slower than float32. Batched, it raises throughput and
        # halves peak VRAM at same_as_baseline 0.979 (benchmarks/results/caption_batch.json).
        if self.device == "cuda":
            torch.nn.Module.half(loaded)
        self.model = loaded
        self.model_id = model
        # In the hub cache layout the snapshot directory IS the commit:
        # the version recorded on every caption is immutable, never `main`.
        landed = _cached_snapshot(models_dir, model)
        self.model_version = pathlib.Path(landed).name if landed else REVISION

    def describe(self, image: Image.Image) -> str:
        """One picture, one sentence. A batch of exactly one."""
        return self.describe_many([image])[0]

    def describe_many(self, images: Sequence[Image.Image]) -> list[str]:
        """Several pictures in ONE forward pass, in order.

        A caption per `generate()` left the GPU idle between pictures and
        paid a kernel launch and a copy back for each one. Measured over
        48 real pictures on a 3070 Ti (`just bench captions`):

            batch  1   3.62 pictures/sec
            batch  8  12.81
            batch 16  15.72   -- 4.3x, and every caption identical

        Identical because nothing here is shared between members: the
        vision encoder sees each picture on its own, and greedy decoding
        (`num_beams=1, do_sample=False` in the checkpoint's own
        generation config) is deterministic per sequence. The benchmark
        asserts that rather than assuming it, because a batch that
        changed what the model SAID would be a different feature wearing
        a speedup's clothes.
        """
        import torch

        if not images:
            return []
        # the processor's typed kwargs nest `return_tensors` under a modality
        # group the call merges at runtime; the call itself takes it flat
        encode = cast("Callable[..., BatchFeature]", self.processor)
        inputs = encode(images=[one.convert("RGB") for one in images], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MOST_TOKENS)
        return [text.strip() for text in self.processor.batch_decode(out, skip_special_tokens=True)]


def captioner_for(models_dir: str, model: str = MODEL, *, provision: bool = False) -> Captioner:
    """The captioner the `caption_model` setting names: a BLIP
    captioning checkpoint on the Hub, by repository id."""
    if not model or "/" not in model:
        raise ValueError(f"caption_model must be a Hub repository id like {MODEL}, not {model!r}")
    return BlipCaptioner(models_dir, model, provision=provision)
