"""The OpenCLIP adapter -- one joint model, two doors into one space.

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


def _provisioned(models_dir: str, model: str, checkpoint: str) -> bool:
    """Whether this checkpoint's weights already sit in the local cache,
    answered WITHOUT any network access: open_clip names the Hugging
    Face repo (refs/mlfoundations/open_clip src/open_clip/pretrained.py
    get_pretrained_cfg, the 'hf_hub' key) and `scan_cache_dir`
    enumerates what the cache directory holds (refs/huggingface/
    huggingface_hub src/huggingface_hub/utils/_cache_manager.py:588 --
    HFCacheInfo.repos of CachedRepoInfo(repo_id, revisions)). Setting
    HF_HUB_OFFLINE at runtime is theater -- huggingface_hub reads it at
    import -- which is how the first version of this guard downloaded
    600MB while claiming it would not."""
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound
    from open_clip.pretrained import get_pretrained_cfg

    repo = (get_pretrained_cfg(model, checkpoint) or {}).get("hf_hub", "").rstrip("/")
    if not repo:
        return False
    try:
        held = scan_cache_dir(models_dir)
    except CacheNotFound:
        return False  # no cache directory yet IS unprovisioned
    return any(cached.repo_id == repo and len(cached.revisions) > 0 for cached in held.repos)


class ClipBackend:
    """One loaded OpenCLIP model, both encoders, numpy in and out."""

    provider = PROVIDER

    def __init__(self, models_dir: str, model: str = MODEL, checkpoint: str = CHECKPOINT, *, offline: bool = False):
        import open_clip
        import torch

        self.model_name = model
        self.checkpoint = checkpoint
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if offline and not _provisioned(models_dir, model, checkpoint):
            raise LookupError(
                f"{model}/{checkpoint} is not provisioned under {models_dir}; run /jobs/embed once to download it"
            )
        loaded, _train_tf, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=checkpoint, cache_dir=models_dir
        )
        self.tokenizer = open_clip.get_tokenizer(model, cache_dir=models_dir)
        loaded.eval()  # models construct in train mode; see module docstring
        self.model = loaded.to(self.device)
        self.dimensions = int(self.encode_query("probe").shape[0])

    @property
    def model_id(self) -> str:
        return self.model_name

    def space(self):
        return space(self.model_name, self.checkpoint, self.dimensions)

    def encode_media(self, frame):
        """One decoded, oriented PIL frame to one unit-length vector."""
        import torch

        tensor = self.preprocess(frame.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(tensor, normalize=True)
        return features[0].cpu().float().numpy()

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
