"""The joint image/text encoder -- one model, two doors into one space.

OpenCLIP trains an image encoder and a text encoder into the SAME vector
space, which is the entire feature: embed every picture once, and a
typed phrase becomes a query vector against those stored image vectors.
No tags, no captions, no metadata -- "creepy abandoned building" finds
creepy abandoned buildings because the text encoder and the image
encoder agree about what that looks like.

The API is v3's exactly (refs/mlfoundations/open_clip@92433b5, README
"Usage" + src/open_clip/model.py:326-341): `create_model_and_transforms`
returns the model and the inference transform, `encode_image` /
`encode_text` take `normalize=True` so inner product IS the cosine, and
`model.eval()` is mandatory -- models construct in train mode.
`torch.no_grad` wraps every encode: inference never needs the autograd
graph, and the context is thread-local (refs/pytorch/torch
torch/autograd/grad_mode.py:22-36).

Provenance is the whole joint model: image vectors from one checkpoint
answered with another checkpoint's text encoder may share dimensions and
still mean nothing to each other. The space's producer is therefore
model+checkpoint, and its preprocess version is the open_clip package
version -- the transforms and tokenizer ship with the package, so a
package upgrade is a preprocessing change until proven otherwise.

Weights land under the run's models_dir (`cache_dir`), the same doctrine
every other model in this application follows: a run is one directory,
and deleting it deletes the run.
"""

from __future__ import annotations

import threading

#: The default joint model: small, fast, and good enough to prove the
#: space -- the `semantic_model` setting (db/settings.py) names another
#: as "<model>/<pretrained-checkpoint>" from open_clip.list_pretrained().
MODEL = "ViT-B-32"
CHECKPOINT = "laion2b_s34b_b79k"


def openclip_version() -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version("open_clip_torch")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class ClipBackend:
    """One loaded OpenCLIP model, both encoders, numpy in and out."""

    def __init__(self, models_dir: str, model: str = MODEL, checkpoint: str = CHECKPOINT):
        import open_clip
        import torch

        self.model_name = model
        self.checkpoint = checkpoint
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        loaded, _train_tf, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=checkpoint, cache_dir=models_dir
        )
        loaded.eval()  # models construct in train mode; see module docstring
        self.model = loaded.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model, cache_dir=models_dir)
        self.dimensions = int(self.encode_text("probe").shape[0])

    def encode_image(self, image):
        """One PIL frame to one unit-length float32 vector."""
        import torch

        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(tensor, normalize=True)
        return features[0].cpu().float().numpy()

    def encode_text(self, text: str):
        """One phrase to one unit-length float32 vector, in the same space."""
        import torch

        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens, normalize=True)
        return features[0].cpu().float().numpy()


#: One loaded model per (models_dir, model, checkpoint) per process --
#: loading is seconds and hundreds of megabytes; encoding is milliseconds.
_LOADED: dict[tuple, ClipBackend] = {}
_LOCK = threading.Lock()


def backend(models_dir: str, model: str = MODEL, checkpoint: str = CHECKPOINT) -> ClipBackend:
    key = (str(models_dir), model, checkpoint)
    with _LOCK:
        if key not in _LOADED:
            _LOADED[key] = ClipBackend(str(models_dir), model, checkpoint)
        return _LOADED[key]
