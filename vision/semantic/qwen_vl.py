"""The Qwen3-VL-Embedding adapter -- retrieval-trained, video-native.

Qwen3-VL-Embedding pools the LAST attended position of a Qwen3-VL
decoder into one vector, trained for retrieval over text, images and
video (refs/QwenLM/Qwen3-VL-Embedding README + src/models/
qwen3_vl_embedding.py, ported below under its Apache-2.0 license).
Unlike CLIP there is no separate text tower: every input -- a picture,
a phrase, a video -- is a chat conversation the same decoder embeds,
which is why the preprocessing IS the chat template and why this
adapter takes MediaRef's native door for video: the model samples
frames itself (refs/QwenLM/qwen_vl_utils src/qwen_vl_utils/
vision_process.py fetch_video: fps and max_frames budgets, smart_resize
pixel caps) instead of judging a whole clip by one poster.

Space identity: the checkpoint is a Hugging Face revision of the
Qwen/<model> repository, named-weights-by-convention exactly as an
open_clip pretrained tag is. The preprocess version pins everything
that changes meaning without changing weights: the two instructions,
the pixel and frame budgets below, EOS pooling, L2 normalization.
Changing any of them is a new preprocess version, hence a new space.

The upstream wrapper degrades a failed media decode into embedding the
literal text "NULL" to keep an evaluation batch alive. This adapter
deliberately does not: an item that cannot decode fails its job item
loudly, because a vector of the word NULL sitting in the space is a
picture that answers queries about nothing.
"""

from __future__ import annotations

import threading

# Module scope, not inside _for_embedding: transformers' __init_subclass__
# runs get_type_hints over the class, which evaluates the (lazily stringified)
# ClassVar annotation in THIS module's globals -- a function-local import is
# invisible there and the class definition dies with a NameError.
from typing import ClassVar

PROVIDER = "qwen"

#: The 2B embedding model: 2048 dimensions, MRL-capable, the smallest
#: retrieval-trained Qwen3-VL. The `semantic_model` setting names others
#: as `qwen:<org>/<repo>[@revision]` -- the Hugging Face repo id AS the
#: model reference, because that is what the thing is called upstream;
#: splitting it into a fake "model/checkpoint" pair would misname both
#: halves.
MODEL = "Qwen/Qwen3-VL-Embedding-2B"
CHECKPOINT = "main"


def parse(reference: str) -> tuple[str, str]:
    """One `semantic_model` reference in this provider's own grammar:
    a Hugging Face `<org>/<repo>`, optionally pinned `@<revision>`
    (default `main`)."""
    repo, at, revision = reference.partition("@")
    org, slash, name = repo.partition("/")
    if not slash or not org or not name or "/" in name:
        raise ValueError(f"a qwen entry is '<org>/<repo>[@revision]' (a Hugging Face repo id), not {reference!r}")
    if at and not revision:
        raise ValueError(f"a trailing '@' names no revision: {reference!r}")
    return repo, revision or CHECKPOINT


#: refs/QwenLM/Qwen3-VL-Embedding src/models/qwen3_vl_embedding.py:24-33,
#: verbatim: token budget, patch-derived pixel budgets, frame sampling.
MAX_LENGTH = 8192
IMAGE_FACTOR = 32
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR
FPS = 1
MAX_FRAMES = 64

#: The instructions are part of the space's meaning: the model was
#: trained instruction-aware, so stored vectors and query vectors must
#: each carry the role they were trained for (README usage example).
MEDIA_INSTRUCTION = "Represent the user's input."
QUERY_INSTRUCTION = "Retrieve images or text relevant to the user's query."


def policy_digest(*facts) -> str:
    """A short stable digest of preprocessing facts -- the mechanism the
    space identity uses to make 'same knobs' checkable rather than
    asserted."""
    import hashlib

    return hashlib.sha256(";".join(str(fact) for fact in facts).encode("utf-8")).hexdigest()[:12]


def policy_version() -> str:
    """Everything that changes a vector WITHOUT changing weights, as one
    token: the pixel and frame budgets, the token budget, both
    instructions, and the versions of the two packages whose code is the
    preprocessing (transformers renders the chat template and packs the
    patches; qwen_vl_utils resizes and samples). Editing any of them
    changes this string, which changes the spec hash, which mints a NEW
    space -- 'v1' as a hand-bumped label was a promise, this is a
    property. Deliberately over-invalidating: the query instruction
    shapes no stored vector, but a space that cannot say which query
    policy it answers under is provenance with a hole in it."""
    import importlib.metadata

    digest = policy_digest(MAX_LENGTH, MIN_PIXELS, MAX_PIXELS, FPS, MAX_FRAMES, MEDIA_INSTRUCTION, QUERY_INSTRUCTION)
    told = {}
    for package in ("transformers", "qwen-vl-utils"):
        try:
            told[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            told[package] = "unknown"
    return f"tf{told['transformers']}+qvu{told['qwen-vl-utils']}+{digest}"


def space(model: str, checkpoint: str, dimensions: int):
    """The immutable identity of this configuration's joint space."""
    from vision.faiss_index import SpaceSpec

    return SpaceSpec(
        key=f"semantic.qwen.{model}.{checkpoint}",
        representation="float32",
        dimensions=int(dimensions),
        metric="cosine",
        producer=f"qwen3vl:{model}",
        producer_version=checkpoint,
        preprocess="qwen3vl.chat-template",
        preprocess_version=policy_version(),
    )


#: What a loadable snapshot must hold: the model, its processor, its
#: tokenizer, and the chat template the whole flow starts from. Weight
#: presence alone is not provisioned -- from_pretrained on a snapshot
#: missing its tokenizer fails halfway into serving instead of refusing
#: up front with the fix named.
_SNAPSHOT_FILES = (
    "config.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
)


def _cached_snapshot(models_dir: str, model: str, checkpoint: str) -> str | None:
    """The local snapshot directory holding this revision COMPLETE --
    weights (single-file or sharded) plus every file loading touches --
    or None. Answered from disk alone, the same doctrine as the openclip
    adapter's `_cached_checkpoint`."""
    import pathlib

    from huggingface_hub import try_to_load_from_cache

    held = None
    for name in ("model.safetensors", "model.safetensors.index.json"):
        found = try_to_load_from_cache(model, name, cache_dir=models_dir, revision=checkpoint)
        if isinstance(found, str):
            held = found
            break
    if held is None:
        return None
    for name in _SNAPSHOT_FILES:
        if not isinstance(try_to_load_from_cache(model, name, cache_dir=models_dir, revision=checkpoint), str):
            return None
    return str(pathlib.Path(held).parent)


def _for_embedding():
    """The model class, declared at call time because transformers is a
    heavyweight import this module must not force on registry probes.

    A minimal port of Qwen3VLForEmbedding (refs/QwenLM/
    Qwen3-VL-Embedding src/models/qwen3_vl_embedding.py:42-117,
    Apache-2.0): the checkpoint stores its decoder under `model.*`, so
    the wrapper's one structural job is holding Qwen3VLModel under that
    attribute and returning the last hidden state unpooled.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel, Qwen3VLPreTrainedModel

    class ForEmbedding(Qwen3VLPreTrainedModel):
        _checkpoint_conversion_mapping: ClassVar[dict] = {}
        accepts_loss_kwargs = False

        def __init__(self, config):
            super().__init__(config)
            self.model = Qwen3VLModel(config)
            self.post_init()

        def get_input_embeddings(self):
            return self.model.get_input_embeddings()

        def set_input_embeddings(self, value):
            self.model.set_input_embeddings(value)

        def forward(self, **inputs):
            return self.model(**inputs)

    return ForEmbedding


class QwenBackend:
    """One loaded Qwen3-VL embedding model, every modality through the
    chat template, numpy in and out."""

    provider = PROVIDER

    def __init__(self, models_dir: str, model: str = MODEL, checkpoint: str = CHECKPOINT, *, offline: bool = False):
        import torch
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        self.model_name = model
        self.checkpoint = checkpoint
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        found = _cached_snapshot(models_dir, model, checkpoint)
        if offline and found is None:
            raise LookupError(
                f"{model}/{checkpoint} is not provisioned under {models_dir}; run /jobs/embed once to download it"
            )
        # A resolved snapshot loads by directory path -- no hub call has
        # anywhere to hide in a local-path load; only the provisioning
        # path (the embed job) ever passes the repository id.
        source = found if found is not None else model
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        loaded = _for_embedding().from_pretrained(source, revision=checkpoint, cache_dir=models_dir, dtype=dtype)
        self.processor = Qwen3VLProcessor.from_pretrained(
            source, revision=checkpoint, cache_dir=models_dir, padding_side="right"
        )
        loaded.eval()
        self.model = loaded.to(self.device)
        self.dimensions = int(self.encode_query("probe").shape[0])

    @property
    def model_id(self) -> str:
        return self.model_name

    def space(self):
        return space(self.model_name, self.checkpoint, self.dimensions)

    def encode_media(self, media):
        """One MediaRef to one unit-length vector. Video takes the native
        door -- the model samples its own frames under the fps and frame
        budgets; everything else embeds the canonical frame."""
        if media.kind == "video":
            content = {"type": "video", "video": f"file://{media.path}", "fps": FPS, "max_frames": MAX_FRAMES}
        else:
            frame = media.frame().convert("RGB")
            content = {"type": "image", "image": frame, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
        return self._embed(MEDIA_INSTRUCTION, content)

    def encode_query(self, text: str):
        """One phrase to one unit-length vector, in the same space."""
        return self._embed(QUERY_INSTRUCTION, {"type": "text", "text": text})

    def _embed(self, instruction: str, content: dict):
        """The wrapper's flow (qwen3_vl_embedding.py:329-393), one input
        at a time: chat template -> qwen_vl_utils vision fetch ->
        processor with do_resize=False (the fetch already resized) ->
        last-attended-position pooling -> L2 normalize."""
        import torch
        from qwen_vl_utils.vision_process import process_vision_info
        from torch.nn import functional

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": [content]},
        ]
        text = self.processor.apply_chat_template([conversation], add_generation_prompt=True, tokenize=False)
        images, video_inputs, video_kwargs = process_vision_info(
            [conversation], image_patch_size=16, return_video_metadata=True, return_video_kwargs=True
        )
        if video_inputs is not None:
            pairs = list(video_inputs)
            videos = [clip for clip, _ in pairs]
            video_metadata = [meta for _, meta in pairs]
        else:
            videos, video_metadata = None, None
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            hidden = self.model(**inputs).last_hidden_state
        # EOS pooling: the last position the attention mask admits --
        # flip, argmax finds the first 1 from the end (:365-370).
        mask = inputs["attention_mask"]
        last = mask.shape[1] - mask.flip(dims=[1]).argmax(dim=1) - 1
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        return functional.normalize(pooled, p=2, dim=-1)[0].cpu().float().numpy()


#: One loaded model per (models_dir, model, checkpoint) per process --
#: loading is seconds and gigabytes; encoding is milliseconds.
_LOADED: dict[tuple, QwenBackend] = {}
_LOCK = threading.Lock()


def encoder(models_dir: str, model: str = MODEL, checkpoint: str = CHECKPOINT, *, offline: bool = False) -> QwenBackend:
    key = (str(models_dir), model, checkpoint)
    with _LOCK:
        if key not in _LOADED:
            _LOADED[key] = QwenBackend(str(models_dir), model, checkpoint, offline=offline)
        return _LOADED[key]
