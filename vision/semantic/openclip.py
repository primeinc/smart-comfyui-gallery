"""The OpenCLIP adapter -- one joint model, two links into one space.

OpenCLIP trains an image encoder and a text encoder into the SAME vector
space: embed every picture once, and a typed phrase becomes a query
vector against those stored image vectors. The API here is v3's exactly
(mlfoundations/open_clip@92433b5 src/open_clip/model.py:326-341, and the
README "Usage"): `create_model_and_transforms` returns
the model and the inference transform, `encode_image` / `encode_text`
take `normalize=True` so inner product IS the cosine, and `model.eval()`
is mandatory -- models construct in train mode. `torch.no_grad` wraps
every encode (refs/pytorch/torch torch/autograd/grad_mode.py:22-36).

Provenance is the whole joint model: image vectors from one checkpoint
answered with another checkpoint's text encoder may share dimensions and
still mean nothing to each other. The space's producer is therefore
model+checkpoint, and its preprocess version is the open_clip package
version -- the transforms and tokenizer ship with the package.

Weights land under the run's models_dir (`cache_dir`), the doctrine
every model in this application follows. `offline=True` makes a missing
checkpoint a refusal instead of a download: huggingface_hub honours
HF_HUB_OFFLINE, so an unprovisioned model fails fast and names
/jobs/embed as the fix.
"""

from __future__ import annotations

import threading

PROVIDER = "openclip"

#: The default joint model: small, fast, and good enough to prove the
#: space -- the `semantic_model` setting (db/settings.py) names others.
MODEL = "ViT-B-32"
CHECKPOINT = "laion2b_s34b_b79k"

#: Pictures per encoder pass. Throughput is flat from 64 to 256 while peak
#: VRAM rises from 774.7 MB to 1246.3 (benchmarks/results/openclip_batch.json).
BATCH = 64

#: Threads for the CLIP transform inside one batch. Capped rather than scaled to
#: the machine, the job already keeping about 7 cores busy on its own, and 8 is
#: where preprocess time flattens (benchmarks/results/openclip_batch.json).

#: The transform is bit-identical across widths regardless
#: (`preprocess_equivalence`, max_abs_difference 0.0 at 2/4/8/16), so the thread
#: count cannot change a vector.
BATCH_WORKERS = 8


def parse(reference: str) -> tuple[str, str]:
    """One `semantic_model` reference in this provider's own grammar:
    `<model>/<pretrained-tag>`, both halves from open_clip's registry."""
    model, slash, checkpoint = reference.partition("/")
    if not slash or not model or not checkpoint:
        raise ValueError(f"an openclip entry is '<model>/<pretrained-tag>', not {reference!r}")
    return model, checkpoint


def openclip_version() -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version("open_clip_torch")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def space(model: str, checkpoint: str, dimensions: int):
    """The immutable identity of this configuration's joint space."""
    from vision.faiss_index import SpaceSpec

    return SpaceSpec(
        key=f"semantic.openclip.{model}.{checkpoint}",
        representation="float32",
        dimensions=dimensions,
        metric="cosine",
        producer=f"open_clip:{model}",
        producer_version=checkpoint,
        preprocess="open_clip.transforms",
        preprocess_version=openclip_version(),
    )


def query_policy(model: str, checkpoint: str) -> dict:
    """Everything that turns a TEXT into THIS model's query vector: the
    tokenizer and text tower are the model+checkpoint, the transforms
    ship with the package. Distinct from the stored-media space identity
    in principle, identical in facts for open_clip -- named separately so
    a consumer of query vectors (story planning) hashes the right thing."""
    return {
        "provider": "openclip",
        "model": model,
        "checkpoint": checkpoint,
        "text_tower": "open_clip.encode_text",
        "normalize": True,
        "package": openclip_version(),
    }


def immutable(checkpoint: str) -> bool:
    """An open_clip pretrained tag names fixed weights."""
    return bool(checkpoint)


def _record_path(models_dir: str):
    import pathlib

    return pathlib.Path(models_dir) / "provisioned.json"


def _read_record(models_dir: str) -> dict:
    """The record, or an empty one. Anything unreadable, undecodable or
    not an object reads as "nothing provisioned": the caller is the fast
    offline REFUSAL, and a refusal is the right answer for a record it
    cannot trust. Returning the parsed value unchecked handed
    `provisioned()` a list or a None to call `.get` on, turning that
    refusal into a 500."""
    import json

    try:
        held = json.loads(_record_path(models_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    return held if isinstance(held, dict) else {}


def record_provision(models_dir: str, model: str, checkpoint: str, repo: str, names: tuple[str, ...]) -> None:
    """The (repo, file) coordinates a checkpoint resolved to, written by
    the code that just loaded it -- the one moment those coordinates
    exist without asking open_clip, whose import is torch's. The record
    is what lets the serving guard refuse an unprovisioned model in
    milliseconds instead of paying nine seconds of ML import to say no."""
    import contextlib
    import json
    import os
    import tempfile

    held = _read_record(models_dir)
    held[f"{model}/{checkpoint}"] = {"repo": repo, "names": list(names)}
    path = _record_path(models_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Through a temp file in the same directory: `_LOCK` orders writers inside ONE
    # process, and two processes sharing a models_dir lose each other's entry in
    # the read-modify-write above. `os.replace` also hides a half-written record.
    fd, tmp = tempfile.mkstemp(prefix=".provisioned-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as out:
            out.write(json.dumps(held, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def provisioned(models_dir: str, model: str, checkpoint: str) -> str | None:
    """The recorded weight file, if this pair was ever provisioned here
    and its bytes still sit in a cache -- answered from the record and
    the hub cache alone, no ML import anywhere on the path."""
    from vision.weights import hub_cached

    held = _read_record(models_dir).get(f"{model}/{checkpoint}")
    if held is None:
        return None
    for name in held["names"]:
        found = hub_cached(held["repo"], name, models_dir)
        if found is not None:
            return found
    return None


def _unprovisioned(models_dir: str, model: str, checkpoint: str) -> LookupError:
    return LookupError(
        f"{model}/{checkpoint} is not provisioned under {models_dir}; run /jobs/embed once to download it"
    )


def _tag_text(tag: dict, key: str) -> str | None:
    """One string-valued key of an open_clip pretrained-tag config.

    The config is a heterogeneous dict -- `url`, `hf_hub`,
    `interpolation` and `resize_mode` are strings while `mean` and `std`
    are float triples, and `_pcfg` merges arbitrary `**kwargs` on top
    (refs/mlfoundations/open_clip src/open_clip/pretrained.py:38-49). So
    a value read out of it is that whole union until something says
    which key it came from; this and `_tag_numbers` are that something,
    and a key holding the wrong shape reads as absent rather than
    travelling on into a factory argument.
    """
    value = tag.get(key)
    return value if isinstance(value, str) else None


def _tag_numbers(tag: dict, key: str) -> tuple[float, ...] | None:
    """One float-sequence key of an open_clip pretrained-tag config."""
    value = tag.get(key)
    if isinstance(value, (list, tuple)):
        return tuple(float(one) for one in value)
    return None


def _hub_names(model: str, checkpoint: str) -> tuple[str, tuple[str, ...]] | None:
    """The hub repo and candidate file names open_clip gives this tag,
    or None for a tag that is not hub-hosted. Imports open_clip."""
    import os

    from open_clip.constants import HF_SAFE_WEIGHTS_NAME, HF_WEIGHTS_NAME
    from open_clip.pretrained import get_pretrained_cfg

    hf_hub = _tag_text(get_pretrained_cfg(model, checkpoint) or {}, "hf_hub") or ""
    if not hf_hub:
        return None
    repo, filename = os.path.split(hf_hub)
    if not filename:
        return repo, (HF_SAFE_WEIGHTS_NAME, HF_WEIGHTS_NAME)
    if filename.endswith((".bin", ".pth")):
        return repo, (filename[:-4] + ".safetensors", filename)  # safetensors preferred, as upstream prefers it
    return repo, (filename,)


def _cached_checkpoint(models_dir: str, model: str, checkpoint: str) -> str | None:
    """The exact weight file this checkpoint resolves to in the local
    cache, or None -- answered WITHOUT any network access.

    open_clip names the artifact (refs/mlfoundations/open_clip
    src/open_clip/pretrained.py: get_pretrained_cfg's 'hf_hub' key is
    'org/repo/' or 'org/repo/file'; download_pretrained_from_hf tries
    the safetensors alternative first, then the named file) and
    `try_to_load_from_cache` answers from disk alone -- "This function
    will not raise any exception if the file in not cached"
    (refs/huggingface/huggingface_hub src/huggingface_hub/
    file_download.py:1475). Repo presence is not enough: a cache can
    hold the repo's config without the weight file, and open_clip's tag
    resolver would then reach for the network. Setting HF_HUB_OFFLINE at
    runtime does not help: huggingface_hub reads it at import, so a guard
    that sets it later can still download.
    """
    from vision.weights import hub_cached

    named = _hub_names(model, checkpoint)
    if named is None:
        return None
    repo, names = named
    for name in names:
        found = hub_cached(repo, name, models_dir)  # models_dir, then the machine's shared HF cache
        if found is not None:
            return found
    return None


class ClipBackend:
    """One loaded OpenCLIP model, both encoders, numpy in and out."""

    provider = PROVIDER

    def __init__(self, models_dir: str, model: str = MODEL, checkpoint: str = CHECKPOINT, *, offline: bool = False):
        import open_clip
        import torch
        from torchvision.transforms import Compose

        self.model_name = model
        self.checkpoint = checkpoint
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        found = _cached_checkpoint(models_dir, model, checkpoint)
        if found is None:
            # Downloading belongs to /jobs/embed; the serving path is
            # structurally incapable of it -- a local weight file is the
            # only thing it will hand to open_clip.
            if offline:
                raise _unprovisioned(models_dir, model, checkpoint)
            loaded, _train_tf, preprocess = open_clip.create_model_and_transforms(
                model, pretrained=checkpoint, cache_dir=models_dir
            )
        else:
            # `pretrained=<file>` takes the local-file branch, with no hub call, but
            # that branch skips the tag's preprocess merge, so its four keys ride
            # along (refs/mlfoundations/open_clip src/open_clip/factory.py:412-421 vs :414).
            from open_clip.pretrained import get_pretrained_cfg

            tag = get_pretrained_cfg(model, checkpoint) or {}
            loaded, _train_tf, preprocess = open_clip.create_model_and_transforms(
                model,
                pretrained=found,
                cache_dir=models_dir,
                image_mean=_tag_numbers(tag, "mean"),
                image_std=_tag_numbers(tag, "std"),
                image_interpolation=_tag_text(tag, "interpolation"),
                image_resize_mode=_tag_text(tag, "resize_mode"),
            )
        # the factory returns nn.Module; the two encode_* links live on
        # the contrastive classes, and only those are a space here
        if not isinstance(loaded, (open_clip.CLIP, open_clip.CustomTextCLIP, open_clip.CoCa)):
            raise TypeError(f"{model} built {type(loaded).__name__}, which has no image/text encoders")
        # the inference transform is one Compose; the factory's training
        # branch can hand back timm's (train, eval, eval) triple instead
        if not isinstance(preprocess, Compose):
            raise TypeError(f"{model} built {type(preprocess).__name__} for preprocessing, not one transform")
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(model, cache_dir=models_dir)
        loaded.eval()  # models construct in train mode; see module docstring
        self.model = loaded.to(self.device)
        self.dimensions = int(self.encode_query("probe").shape[0])
        # This load proved the weights exist; write the coordinates down
        # so the next unprovisioned refusal never has to import anything
        # to know them (see `provisioned`).
        named = _hub_names(model, checkpoint)
        if named is not None:
            record_provision(models_dir, model, checkpoint, named[0], named[1])

    @property
    def model_id(self) -> str:
        return self.model_name

    def space(self):
        return space(self.model_name, self.checkpoint, self.dimensions)

    def encode_media(self, media):
        """One MediaRef to one unit-length vector. CLIP consumes exactly
        one frame, so every kind of media enters through the reference's
        canonical representative frame.

        The three stages are named separately because they are three
        different machines and they do not cost alike. Measured over a
        real job, the whole of this was 52% of an item under one label,
        which says nothing about whether the answer is a cheaper raster,
        a bigger batch, or a faster copy. `media.phase` is the runner's
        reporter when there is one and does nothing when there is not.
        """
        import torch

        media.phase("preprocess", model=self.model_name)
        tensor = self.preprocess(media.frame().convert("RGB")).unsqueeze(0)
        media.phase("to-device")
        tensor = tensor.to(self.device)
        media.phase("inference", batch=1)
        with torch.no_grad():
            features = self.model.encode_image(tensor, normalize=True)
        # `from-device` is where this program waits for the GPU: `.cpu()` blocks
        # until the result exists. The boundaries above are host time, because
        # torch.cuda.synchronize would fence the overlap batching creates.
        media.phase("from-device")
        return features[0].cpu().float().numpy()

    def encode_many(self, framers):
        """Many pictures to many unit-length vectors, in one pass.

        `framers` are zero-argument callables that each produce one PIL
        frame -- not the frames themselves. Decoding is what makes a
        batch expensive to hold: sixty-four 22-megapixel frames is four
        gigabytes, while sixty-four preprocessed tensors is thirty-eight
        megabytes. Each worker therefore decodes ITS picture and shrinks
        it immediately, so what accumulates is the small thing and what
        exists at once is one frame per thread.

        Against encoding one picture at a time, per image, read off
        benchmarks/results/openclip_batch.json:

            preprocess    7.28 ms  ->  1.85 ms   batch 64, 1 -> 8 workers
            inference     6.78 ms  ->  1.17 ms   batch 1 -> batch 64
            copy back     one per image -> one per batch

        Threads help because PIL's resize and torch's normalise both drop
        the GIL, and the tensors they produce are bit-identical to the
        serial ones -- `preprocess_equivalence` in that file reports
        max_abs_difference 0.0 at 2, 4, 8 and 16 workers.

        The VECTORS are not bit-identical across batch widths, which is a
        different claim and the file records it separately:
        `vector_equivalence` gives max_abs_difference 2.2e-03 and minimum
        cosine 0.99995 against batch 1. Batching moves the last bits of a
        vector; it does not move the preprocessing.

        Batch 64 is where throughput flattens on this hardware -- 375.9,
        400.5 and 395.2 img/s at 64, 128 and 256, a 6.5% spread for 1.6x
        the VRAM (`just bench clip-batch`).

        Nothing is pinned and no second stream is used. Both are needed
        together to overlap a copy with a kernel, and there is nothing
        here to overlap WITH: one batch goes over, one batch comes back.
        """
        from concurrent.futures import ThreadPoolExecutor

        import torch

        if not framers:
            return []

        def prepared(framer):
            with framer() as frame:
                return self.preprocess(frame.convert("RGB"))

        if len(framers) == 1:
            stacked = torch.stack([prepared(framers[0])])
        else:
            with ThreadPoolExecutor(min(BATCH_WORKERS, len(framers))) as pool:
                stacked = torch.stack(list(pool.map(prepared, framers)))
        with torch.no_grad():
            features = self.model.encode_image(stacked.to(self.device), normalize=True)
        # One copy back for the whole batch rather than one per picture.
        return list(features.cpu().float().numpy())

    def encode_query(self, text: str):
        """One phrase to one unit-length vector, in the same space."""
        import torch

        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens, normalize=True)
        return features[0].cpu().float().numpy()


#: One loaded model per (models_dir, model, checkpoint) per process --
#: loading is seconds and hundreds of megabytes; encoding is milliseconds.
_LOADED: dict[tuple, ClipBackend] = {}
_LOCK = threading.Lock()


def encoder(models_dir: str, model: str = MODEL, checkpoint: str = CHECKPOINT, *, offline: bool = False) -> ClipBackend:
    key = (models_dir, model, checkpoint)
    with _LOCK:
        if key not in _LOADED:
            # The offline refusal must not cost torch's import: without this record
            # check, the 400 for an unprovisioned model paid the whole
            # open_clip+torch import (test_search_never_downloads_a_model).

            # So the record is authoritative here, not the cache: weights another
            # tool dropped into the shared HF cache refuse offline until
            # /jobs/embed writes the record back, and ClipBackend guards the load.
            if offline and provisioned(models_dir, model, checkpoint) is None:
                raise _unprovisioned(models_dir, model, checkpoint)
            _LOADED[key] = ClipBackend(models_dir, model, checkpoint, offline=offline)
        return _LOADED[key]
