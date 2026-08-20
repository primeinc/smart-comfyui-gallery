"""Embedder backends for the semantic and visual similarity spaces.

`StubSemanticEmbedder` and `StubVisualEmbedder` are TEST/DEV STUBS ONLY.
They are deterministic, dependency-free placeholder algorithms -- a seeded
pseudo-random projection of quantized pixels, a char-trigram hash, and a
color histogram -- and do NOT produce meaningful semantic or visual
embeddings. They exist purely so the rest of the AI DAM (vector store,
worker, API) can be built and tested end-to-end without any model weights.
Do not wire them into a user-facing "find similar" result in production;
use `get_semantic_backend`/`get_visual_backend` with a real backend name
(`open_clip`, `dinov2`) instead.

Real adapters lazy-import their runtime (torch/open_clip/transformers) and
load weights ONLY from `AIConfig.models_dir`; nothing in this module
downloads (weights arrive via smartgallery_ai.provision).
When the runtime or weights are missing they raise `BackendUnavailable`
rather than crashing at import time.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import threading
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from smartgallery_ai import AIConfig

_logger = logging.getLogger(__name__)


class BackendUnavailable(Exception):
    """A real embedder backend's runtime or weights are not present locally."""


class SemanticEmbedder(ABC):
    """Joint image/text embedding space (see `SPACE_SEMANTIC`)."""

    model_id: str  # identifies the producing model; stored with every vector
    model_version: str  # separates incompatible vector generations; versions never mix
    dim: int  # length of every embedding this backend returns
    # True only when any number of threads may call this instance at once --
    # either it keeps no per-call state or it guards its own forwards. The
    # default is the safe answer; `smartgallery_ai.backends` leases anything
    # still False exclusively rather than letting two callers share it.
    thread_safe: bool = False

    @abstractmethod
    def embed_image(self, img: Image.Image) -> np.ndarray:
        """Map an image into this space: float32, length `dim`, unit L2 norm."""
        ...

    def embed_images(self, imgs: list) -> list:
        """Batch form of `embed_image`; backends with real batched inference
        override this. Same output per element as embed_image."""
        return [self.embed_image(img) for img in imgs]

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Map text into the same space as images, so image/text cosine is meaningful."""
        ...


class VisualEmbedder(ABC):
    """Image-only self-supervised embedding space (see `SPACE_VISUAL`)."""

    model_id: str  # identifies the producing model; stored with every vector
    model_version: str  # separates incompatible vector generations; versions never mix
    dim: int  # length of every embedding this backend returns
    thread_safe: bool = False  # see SemanticEmbedder.thread_safe

    @abstractmethod
    def embed_image(self, img: Image.Image) -> np.ndarray:
        """Map an image into this space: float32, length `dim`, unit L2 norm."""
        ...

    def embed_images(self, imgs: list) -> list:
        """Batch form of `embed_image`; backends with real batched inference
        override this. Same output per element as embed_image."""
        return [self.embed_image(img) for img in imgs]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Scale to unit L2 norm as float32; a zero vector passes through unchanged."""
    vec = vec.astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else (vec / norm).astype(np.float32)


class StubSemanticEmbedder(SemanticEmbedder):
    """TEST/DEV STUB -- NOT a semantic model. See module docstring."""

    model_id = "stub-semantic"
    model_version = "stub-v1"
    dim = 64
    thread_safe = True  # both methods are pure functions of their argument

    def embed_image(self, img: Image.Image) -> np.ndarray:
        """Pseudo-embedding seeded from a hash of the 8x8 quantized grayscale
        thumbnail: identical pixels give identical vectors, but visually
        similar images do not land near each other."""
        small = img.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        quantized = (np.asarray(small, dtype=np.uint8) // 32).astype(np.uint8)
        digest = hashlib.sha256(quantized.tobytes()).digest()
        seed = int.from_bytes(digest[:8], "big")
        vec = np.random.default_rng(seed).standard_normal(self.dim)
        return _l2_normalize(vec)

    def embed_text(self, text: str) -> np.ndarray:
        """Hashed character-trigram bag: near-duplicate strings overlap in
        buckets and score similar; there is no semantic understanding."""
        vec = np.zeros(self.dim, dtype=np.float32)
        lowered = text.lower()
        trigrams = [lowered[i : i + 3] for i in range(len(lowered) - 2)] or [lowered]
        for trigram in trigrams:
            digest = hashlib.sha256(trigram.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.dim
            vec[bucket] += 1.0
        return _l2_normalize(vec)


class StubVisualEmbedder(VisualEmbedder):
    """TEST/DEV STUB -- NOT a visual model. See module docstring."""

    model_id = "stub-visual"
    model_version = "stub-v1"
    dim = 64  # 4x4x4 RGB color histogram
    thread_safe = True  # embed_image is a pure function of its argument

    def embed_image(self, img: Image.Image) -> np.ndarray:
        """L2-normalized 4x4x4 RGB histogram: reflects palette overlap only,
        never layout or content."""
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8).reshape(-1, 3)
        bins = rgb.astype(np.int32) // 64  # 4 bins per channel (0..3)
        indices = bins[:, 0] * 16 + bins[:, 1] * 4 + bins[:, 2]
        hist = np.bincount(indices, minlength=self.dim)[: self.dim].astype(np.float32)
        return _l2_normalize(hist)


def pick_torch_device(torch_module, role: str | None = None) -> str:
    """Best available torch device: honors AI_DAM_<ROLE>_DEVICE (e.g.
    AI_DAM_VISUAL_DEVICE=cuda:1 pins one backend to one card, spreading
    VRAM across GPUs), then AI_DAM_DEVICE, otherwise CUDA > MPS > CPU.
    With several CUDA devices the one with the most total VRAM wins —
    bare 'cuda' would silently mean enumeration order (PCI slot), not the
    better card. Defensive against builds lacking a backend attribute."""
    if role:
        per_role = os.environ.get(f"AI_DAM_{role.upper()}_DEVICE", "").lower()
        if per_role:
            return per_role
    forced = os.environ.get("AI_DAM_DEVICE", "").lower()
    if forced:
        return forced
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        try:
            count = cuda.device_count()
            if count > 1:

                def rank(i):
                    props = cuda.get_device_properties(i)
                    # Most VRAM wins; equal VRAM goes to the newer
                    # generation (higher compute capability) -- otherwise
                    # an older card that happens to enumerate first would
                    # win the tie.
                    return (props.total_memory, getattr(props, "major", 0), getattr(props, "minor", 0))

                return f"cuda:{max(range(count), key=rank)}"
        except Exception:  # enumeration is best-effort; cuda:0 still works
            _logger.debug("ignored a failure in pick_torch_device", exc_info=True)
        return "cuda"
    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


# Free-VRAM floor below which loading another model on the card is asking
# for an OOM (another app -- typically ComfyUI mid-generation -- owns it).
_VRAM_PRESSURE_FLOOR_BYTES = 2 << 30


def warn_if_vram_pressure(torch_module, device: str, model_id: str) -> None:
    """Log a warning when the chosen CUDA device is nearly out of free
    VRAM at model-load time: the load may OOM or evict whatever else
    (ComfyUI) is using the card. Detection is best-effort -- non-CUDA
    devices and torch builds without mem_get_info stay silent."""
    if not str(device).startswith("cuda"):
        return
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not hasattr(cuda, "mem_get_info"):
        return
    try:
        if ":" in str(device):
            free, total = cuda.mem_get_info(int(str(device).split(":", 1)[1]))
        else:
            free, total = cuda.mem_get_info()
    except Exception:  # pressure check must never block loading
        _logger.debug("handled a failure in warn_if_vram_pressure", exc_info=True)
        return
    if free < _VRAM_PRESSURE_FLOOR_BYTES:
        _logger.warning(
            "[AI] %s: %s has only %.1f GiB of %.1f GiB VRAM free — another "
            "app (ComfyUI?) is using this card; loading here may OOM. Pin a "
            "different device with AI_DAM_DEVICE=cuda:N (or AI_DAM_DEVICE=cpu).",
            model_id,
            device,
            free / (1 << 30),
            total / (1 << 30),
        )


class OpenClipSemanticEmbedder(SemanticEmbedder):
    """Joint image/text embedding via open_clip ViT-B-32 (laion2b_s34b_b79k).

    Weights are loaded ONLY from a local file under `models_dir` -- never
    downloaded. Requires the `open_clip_torch` + `torch` packages.
    """

    model_id = "open_clip/ViT-B-32/laion2b_s34b_b79k"
    model_version = "open_clip-vit-b-32-laion2b_s34b_b79k-v1"
    dim = 512
    thread_safe = True  # every forward runs under self._infer_lock

    def __init__(self, models_dir: str):
        """Raises `BackendUnavailable` when the weights file or the
        torch/open_clip runtime is missing; never triggers a download."""
        # Check the weights BEFORE importing the heavy runtime: 'auto'
        # resolution on an unprovisioned system must stay fast and
        # side-effect-free (a cold torch import costs ~10s).
        weights_path = os.path.join(models_dir, "open_clip", "ViT-B-32_laion2b_s34b_b79k.bin")
        if not os.path.isfile(weights_path):
            raise BackendUnavailable(f"open_clip weights not found at {weights_path}")

        try:
            import open_clip
            import torch
        except Exception as exc:  # any import failure (incl. a
            # torch/torchvision pairing RuntimeError) means unavailable
            raise BackendUnavailable(f"open_clip backend unavailable: {exc}") from exc

        self._torch = torch
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=weights_path, cache_dir=None
            )
        except Exception as exc:
            raise BackendUnavailable(f"failed to load open_clip weights: {exc}") from exc
        self._device = pick_torch_device(torch, role="semantic")
        _logger.info("[AI] %s on device %s (torch %s)", self.model_id, self._device, getattr(torch, "__version__", "?"))
        warn_if_vram_pressure(torch, self._device, self.model_id)
        model.eval()
        self._model = model.to(self._device)
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
        # Serializes inference so the semantic-search request path can
        # borrow the worker's loaded instance: pure forwards are
        # practically thread-safe, but the lock makes concurrent
        # worker-embed vs. search-encode airtight.
        self._infer_lock = threading.Lock()

    def embed_image(self, img: Image.Image) -> np.ndarray:
        """CLIP image feature, unit-normalized so image/text cosine works."""
        return self.embed_images([img])[0]

    def embed_images(self, imgs: list) -> list:
        """One batched forward for the whole list — the throughput path the
        worker feeds decoded chunks into."""
        batch = self._torch.stack([self._preprocess(img.convert("RGB")) for img in imgs]).to(self._device)
        with self._infer_lock, self._torch.no_grad():
            features = self._model.encode_image(batch)
        return [_l2_normalize(f.cpu().numpy()) for f in features]

    def embed_text(self, text: str) -> np.ndarray:
        """CLIP text feature, unit-normalized so image/text cosine works."""
        tokens = self._tokenizer([text]).to(self._device)
        with self._infer_lock, self._torch.no_grad():
            features = self._model.encode_text(tokens)
        return _l2_normalize(features.squeeze(0).cpu().numpy())


class Dinov2VisualEmbedder(VisualEmbedder):
    """Self-supervised visual embedding via facebook/dinov2-small.

    Weights are loaded ONLY from a local directory under `models_dir`
    (`local_files_only=True`) -- never downloaded. Requires `torch` +
    `transformers`.
    """

    model_id = "facebook/dinov2-small"
    model_version = "dinov2-small-v1"
    dim = 384

    def __init__(self, models_dir: str):
        """Raises `BackendUnavailable` when the weights directory or the
        torch/transformers runtime is missing; never triggers a download."""
        # Weights check precedes the heavy import (see OpenClip above).
        weights_dir = os.path.join(models_dir, "dinov2-small")
        if not os.path.isdir(weights_dir):
            raise BackendUnavailable(f"dinov2 weights not found at {weights_dir}")

        try:
            import torch

            # torchvision MUST come before transformers: transformers
            # freezes its torchvision-availability flag when first imported,
            # and AutoImageProcessor hard-requires torchvision. Importing
            # transformers in a torchvision-less process would keep this
            # backend dead until a full restart even after auto-provisioning
            # installs torchvision; failing here first keeps transformers
            # unimported so the post-provision re-probe activates cleanly.
            importlib.import_module("torchvision")
            from transformers import AutoImageProcessor, AutoModel
        except Exception as exc:  # see open_clip note above
            raise BackendUnavailable(f"dinov2 backend unavailable: {exc}") from exc

        try:
            self._processor = AutoImageProcessor.from_pretrained(weights_dir, local_files_only=True)
            self._model = AutoModel.from_pretrained(weights_dir, local_files_only=True)
        except Exception as exc:
            raise BackendUnavailable(f"failed to load dinov2 weights: {exc}") from exc
        self._model.eval()
        self._device = pick_torch_device(torch, role="visual")
        _logger.info("[AI] %s on device %s (torch %s)", self.model_id, self._device, getattr(torch, "__version__", "?"))
        warn_if_vram_pressure(torch, self._device, self.model_id)
        self._model = self._model.to(self._device)
        self._torch = torch

    def embed_image(self, img: Image.Image) -> np.ndarray:
        """DINOv2 global image descriptor (CLS token), unit-normalized."""
        return self.embed_images([img])[0]

    def embed_images(self, imgs: list) -> list:
        """One batched forward for the whole list (the processor natively
        accepts image lists)."""
        inputs = self._processor(images=[img.convert("RGB") for img in imgs], return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        cls_tokens = outputs.last_hidden_state[:, 0, :]  # position 0 is the CLS summary token
        return [_l2_normalize(t.cpu().numpy()) for t in cls_tokens]


def get_semantic_backend(config: AIConfig) -> SemanticEmbedder | None:
    """Resolve `config.semantic_backend` to an embedder instance, or None.

    'auto' tries the real backend and falls back to None (never to the
    stub) so production code paths can't silently serve fake embeddings.
    """
    name = config.semantic_backend
    if name == "none":
        return None
    if name == "stub":
        return StubSemanticEmbedder()
    if name == "open_clip":
        return OpenClipSemanticEmbedder(config.models_dir)
    if name == "auto":
        try:
            return OpenClipSemanticEmbedder(config.models_dir)
        except BackendUnavailable:
            return None
    raise ValueError(f"unknown semantic_backend: {name!r}")


def get_visual_backend(config: AIConfig) -> VisualEmbedder | None:
    """Resolve `config.visual_backend` to an embedder instance, or None.

    'auto' tries the real backend and falls back to None (never to the
    stub) so production code paths can't silently serve fake embeddings.
    """
    name = config.visual_backend
    if name == "none":
        return None
    if name == "stub":
        return StubVisualEmbedder()
    if name == "dinov2":
        return Dinov2VisualEmbedder(config.models_dir)
    if name == "auto":
        try:
            return Dinov2VisualEmbedder(config.models_dir)
        except BackendUnavailable:
            return None
    raise ValueError(f"unknown visual_backend: {name!r}")
