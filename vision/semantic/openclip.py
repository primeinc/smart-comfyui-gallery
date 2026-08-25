"""The OpenCLIP adapter -- one joint model, two links into one space.

OpenCLIP trains an image encoder and a text encoder into the SAME vector
space: embed every picture once, and a typed phrase becomes a query
vector against those stored image vectors. The API here is v3's exactly
(refs/mlfoundations/open_clip@92433b5, README "Usage" +
src/open_clip/model.py:326-341): `create_model_and_transforms` returns
the model and the inference transform, `encode_image` / `encode_text`
take `normalize=True` so inner product IS the cosine, and `model.eval()`
is mandatory -- models construct in train mode. `torch.no_grad` wraps
every encode (refs/pytorch/torch torch/autograd/grad_mode.py:22-36).

Provenance is the whole joint model: image vectors from one checkpoint
queried with another checkpoint's text encoder may share dimensions and
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

#: Pictures per encoder pass. Throughput plateaus here: measured over 512
#: generated PNGs, batch 64 through 256 sit within 5% of each other while
#: peak VRAM climbs 775 -> 1246 MB, so past this the memory is spent for
#: nothing (`just bench clip-batch`).
BATCH = 64

#: Threads for the CLIP transform inside one batch. It plateaus at 8 on a
#: 16-core machine -- 8.13 ms per image serially, 1.83 at 8 workers, 1.80
#: at 16 -- and the job already keeps about 7 cores busy on its own, so
#: asking for all of them would be taking them from the decoders. Capped
#: rather than scaled to the machine for that reason.
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
        dimensions=int(dimensions),
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


def _cached_checkpoint(models_dir: str, model: str, checkpoint: str) -> str | None:
    """The exact weight file this checkpoint resolves to in the local
    cache, or None -- determined WITHOUT any network access.

    open_clip names the artifact (refs/mlfoundations/open_clip
    src/open_clip/pretrained.py: get_pretrained_cfg's 'hf_hub' key is
    'org/repo/' or 'org/repo/file'; download_pretrained_from_hf tries
    the safetensors alternative first, then the named file) and
    `try_to_load_from_cache` resolves from disk alone -- "This function
    will not raise any exception if the file in not cached"
    (refs/huggingface/huggingface_hub src/huggingface_hub/
    file_download.py:1475). Repo presence is not enough: a cache can
    hold the repo's config without the weight file, and open_clip's tag
    resolver would then reach for the network. Setting HF_HUB_OFFLINE
    at runtime is theater -- huggingface_hub reads it at import --
    which is how the first version of this guard downloaded 600MB
    while claiming it would not.
    """
    import os

    from open_clip.constants import HF_SAFE_WEIGHTS_NAME, HF_WEIGHTS_NAME
    from open_clip.pretrained import get_pretrained_cfg

    from vision.weights import hub_cached

    hf_hub = (get_pretrained_cfg(model, checkpoint) or {}).get("hf_hub", "")
    if not hf_hub:
        return None
    repo, filename = os.path.split(hf_hub)
    if not filename:
        names = (HF_SAFE_WEIGHTS_NAME, HF_WEIGHTS_NAME)
    elif filename.endswith((".bin", ".pth")):
        names = (filename[:-4] + ".safetensors", filename)  # safetensors preferred, as upstream prefers it
    else:
        names = (filename,)
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
                raise LookupError(
                    f"{model}/{checkpoint} is not provisioned under {models_dir}; run /jobs/embed once to download it"
                )
            loaded, _train_tf, preprocess = open_clip.create_model_and_transforms(
                model, pretrained=checkpoint, cache_dir=models_dir
            )
        else:
            # `pretrained=<file>` takes factory.py's local-file branch --
            # no hub call anywhere in it -- but that branch skips the
            # tag's preprocess merge, so the tag's four preprocess keys
            # ride along explicitly (refs/mlfoundations/open_clip
            # src/open_clip/factory.py:412-421 vs :414; pretrained.py
            # _pcfg: mean, std, interpolation, resize_mode).
            from open_clip.pretrained import get_pretrained_cfg

            tag = get_pretrained_cfg(model, checkpoint) or {}
            loaded, _train_tf, preprocess = open_clip.create_model_and_transforms(
                model,
                pretrained=found,
                cache_dir=models_dir,
                image_mean=tag.get("mean"),
                image_std=tag.get("std"),
                image_interpolation=tag.get("interpolation"),
                image_resize_mode=tag.get("resize_mode"),
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
        # `from-device` is where this program actually waits for the GPU:
        # `.cpu()` blocks until the result exists. The phase boundaries
        # above are host time, which is what a phase boundary can honestly
        # be without changing the program.
        #
        # An earlier version put `torch.cuda.synchronize()` at each
        # boundary so every phase would read as its own GPU cost. It gave
        # a tidier table and a slower encoder: synchronize waits for ALL
        # kernels in ALL streams on the device, so it fences exactly the
        # overlap that batching exists to create. Measuring by preventing
        # the thing being measured. Per-kernel GPU time, if it is ever
        # wanted, is torch.cuda.Event with enable_timing -- it timestamps
        # on the stream and is resolved when something waits anyway.
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

        Measured against calling encode_media in a loop, per image:

            preprocess    8.13 ms  ->  1.83 ms   threads, bit-identical
            inference     7.00 ms  ->  1.17 ms   batch 64
            copy back     one per image -> one per batch

        Threads help because PIL's resize and torch's normalise both drop
        the GIL, and the tensors they produce are identical to the serial
        ones -- checked, not assumed. Batch 64 is where throughput
        plateaus on this hardware; 128 and 256 are within 5% and cost
        VRAM (`just bench clip-batch`).

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
    key = (str(models_dir), model, checkpoint)
    with _LOCK:
        if key not in _LOADED:
            _LOADED[key] = ClipBackend(str(models_dir), model, checkpoint, offline=offline)
        return _LOADED[key]
