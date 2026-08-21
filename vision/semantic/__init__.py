"""The semantic embedding seam: adapters in, one small contract out.

Every joint image/text model lives behind this seam as an ADAPTER, and
the rest of the application knows only the contract -- never a
tokenizer, a transform, a checkpoint format, a device choice, or an
instruction policy. Deleting one adapter tomorrow leaves the others
working; adding one next month adds a module here and a provider name
in the `semantic_model` setting, nothing else.

The contract, whole:

    adapter.provider     -> str, the provider name ("openclip", ...)
    adapter.model        -> str
    adapter.checkpoint   -> str
    adapter.dimensions   -> int, known once loaded
    adapter.space()      -> SpaceSpec: the immutable identity of the
                            joint space this exact configuration writes
                            into -- producer, checkpoint, preprocess
                            policy, dimensions
    adapter.encode_media(media) -> unit float32[D] for one MediaRef.
                            The reference describes the media instead of
                            pre-decoding it, because adapters disagree
                            about what pixels they want: a CLIP model
                            consumes one representative frame, a
                            video-native model samples the file itself.
    adapter.encode_query(text)  -> unit float32[D] in the SAME space

Each provider module also exposes `space(model, checkpoint, dimensions)`
so registry lookups can name a space without loading any weights.

Vectors from different adapters -- or different checkpoints of one
adapter -- are never comparable: each is its own immutable
similarity_space, and the retrieval layer (db/retrieval.py) merges
RANKS across spaces, never raw scores.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any


@dataclasses.dataclass(frozen=True)
class MediaRef:
    """One piece of media, described rather than pre-decoded.

    `path` and `kind` are the two doors an adapter can take: read the
    file natively (a video model with its own frame sampling), or call
    `frame()` for the repo's canonical representative frame -- decoded,
    EXIF-oriented, a poster frame for video -- computed lazily so an
    adapter that never wants it never pays for it. The repo owns what
    "the representative frame" means; the adapter owns everything after
    the pixels.
    """

    path: str
    kind: str  # 'image' | 'animated_image' | 'video'
    frame: Callable[[], Any]


#: provider name -> module path. A provider not in this table is a
#: refused configuration, loudly, never a silent substitution.
PROVIDERS = {
    "openclip": "vision.semantic.openclip",
}


def provider_module(provider: str):
    import importlib

    where = PROVIDERS.get(provider)
    if where is None:
        raise ValueError(f"unknown embedding provider {provider!r}; known: {', '.join(sorted(PROVIDERS))}")
    return importlib.import_module(where)


def encoder(provider: str, models_dir: str, model: str, checkpoint: str, *, offline: bool = False):
    """A loaded adapter, cached per process by the provider module.

    `offline=True` refuses to download weights -- a search request must
    never silently begin acquiring hundreds of megabytes; provisioning
    belongs to /jobs/embed. An unprovisioned model raises LookupError
    naming the fix.
    """
    return provider_module(provider).encoder(models_dir, model, checkpoint, offline=offline)


def space(provider: str, model: str, checkpoint: str, dimensions: int):
    """The immutable space identity for a configuration, weights unloaded."""
    return provider_module(provider).space(model, checkpoint, dimensions)
