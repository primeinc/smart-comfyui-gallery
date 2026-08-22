"""Captions: one sentence a vision-language model says about a picture.

The annotate job is the one caller that may provision
(docs/AI_MODELS.md); everything else loads from disk or refuses by
name. Lookup is two-deep, the run's models_dir then the shared HF
cache, and a snapshot counts only when every file loading touches is
present -- weights alone would fail halfway into from_pretrained.
"""

from __future__ import annotations

import pathlib
from typing import Protocol

from PIL import Image

from vision.weights import Unprovisioned, hub_cached

#: The default `caption_model` setting: BLIP base, captioning head, BSD-3.
MODEL = "Salesforce/blip-image-captioning-base"
REVISION = "main"
#: Words a caption may run to; BLIP captions are one sentence.
MOST_TOKENS = 40
_WEIGHTS = ("model.safetensors", "pytorch_model.bin")
_SNAPSHOT_FILES = ("config.json", "preprocessor_config.json", "tokenizer_config.json", "vocab.txt")


class Captioner(Protocol):
    model_id: str
    model_version: str

    def describe(self, image: Image.Image) -> str: ...


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
        self.model = loaded.to(self.device)
        self.model_id = model
        # In the hub cache layout the snapshot directory IS the commit:
        # the version recorded on every caption is immutable, never `main`.
        landed = _cached_snapshot(models_dir, model)
        self.model_version = pathlib.Path(landed).name if landed else REVISION

    def describe(self, image: Image.Image) -> str:
        import torch

        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MOST_TOKENS)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


def captioner_for(models_dir: str, model: str = MODEL, *, provision: bool = False) -> Captioner:
    """The captioner the `caption_model` setting names: a BLIP
    captioning checkpoint on the Hub, by repository id."""
    if not model or "/" not in model:
        raise ValueError(f"caption_model must be a Hub repository id like {MODEL}, not {model!r}")
    return BlipCaptioner(models_dir, model, provision=provision)
