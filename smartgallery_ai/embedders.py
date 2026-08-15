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
load weights ONLY from `AIConfig.models_dir` (local files, never downloaded).
When the runtime or weights are missing they raise `BackendUnavailable`
rather than crashing at import time.
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from PIL import Image

from smartgallery_ai import AIConfig


class BackendUnavailable(Exception):
    """A real embedder backend's runtime or weights are not present locally."""


class SemanticEmbedder(ABC):
    """Joint image/text embedding space (see `SPACE_SEMANTIC`)."""

    model_id: str
    model_version: str
    dim: int

    @abstractmethod
    def embed_image(self, img: Image.Image) -> np.ndarray: ...

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray: ...


class VisualEmbedder(ABC):
    """Image-only self-supervised embedding space (see `SPACE_VISUAL`)."""

    model_id: str
    model_version: str
    dim: int

    @abstractmethod
    def embed_image(self, img: Image.Image) -> np.ndarray: ...


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else (vec / norm).astype(np.float32)


class StubSemanticEmbedder(SemanticEmbedder):
    """TEST/DEV STUB -- NOT a semantic model. See module docstring."""

    model_id = "stub-semantic"
    model_version = "stub-v1"
    dim = 64

    def embed_image(self, img: Image.Image) -> np.ndarray:
        small = img.convert("L").resize((8, 8), Image.LANCZOS)
        quantized = (np.asarray(small, dtype=np.uint8) // 32).astype(np.uint8)
        digest = hashlib.sha256(quantized.tobytes()).digest()
        seed = int.from_bytes(digest[:8], "big")
        vec = np.random.default_rng(seed).standard_normal(self.dim)
        return _l2_normalize(vec)

    def embed_text(self, text: str) -> np.ndarray:
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

    def embed_image(self, img: Image.Image) -> np.ndarray:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8).reshape(-1, 3)
        bins = (rgb.astype(np.int32) // 64)  # 4 bins per channel (0..3)
        indices = bins[:, 0] * 16 + bins[:, 1] * 4 + bins[:, 2]
        hist = np.bincount(indices, minlength=self.dim)[: self.dim].astype(np.float32)
        return _l2_normalize(hist)


class OpenClipSemanticEmbedder(SemanticEmbedder):
    """Joint image/text embedding via open_clip ViT-B-32 (laion2b_s34b_b79k).

    Weights are loaded ONLY from a local file under `models_dir` -- never
    downloaded. Requires the `open_clip_torch` + `torch` packages.
    """

    model_id = "open_clip/ViT-B-32/laion2b_s34b_b79k"
    model_version = "open_clip-vit-b-32-laion2b_s34b_b79k-v1"
    dim = 512

    def __init__(self, models_dir: str):
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise BackendUnavailable(f"open_clip backend unavailable: {exc}") from exc

        weights_path = os.path.join(
            models_dir, "open_clip", "ViT-B-32_laion2b_s34b_b79k.bin"
        )
        if not os.path.isfile(weights_path):
            raise BackendUnavailable(f"open_clip weights not found at {weights_path}")

        self._torch = torch
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=weights_path, cache_dir=None
            )
        except Exception as exc:
            raise BackendUnavailable(f"failed to load open_clip weights: {exc}") from exc
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def embed_image(self, img: Image.Image) -> np.ndarray:
        tensor = self._preprocess(img.convert("RGB")).unsqueeze(0)
        with self._torch.no_grad():
            features = self._model.encode_image(tensor)
        return _l2_normalize(features.squeeze(0).cpu().numpy())

    def embed_text(self, text: str) -> np.ndarray:
        tokens = self._tokenizer([text])
        with self._torch.no_grad():
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
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise BackendUnavailable(f"dinov2 backend unavailable: {exc}") from exc

        weights_dir = os.path.join(models_dir, "dinov2-small")
        if not os.path.isdir(weights_dir):
            raise BackendUnavailable(f"dinov2 weights not found at {weights_dir}")

        try:
            self._processor = AutoImageProcessor.from_pretrained(
                weights_dir, local_files_only=True
            )
            self._model = AutoModel.from_pretrained(weights_dir, local_files_only=True)
        except Exception as exc:
            raise BackendUnavailable(f"failed to load dinov2 weights: {exc}") from exc
        self._model.eval()
        self._torch = torch

    def embed_image(self, img: Image.Image) -> np.ndarray:
        inputs = self._processor(images=img.convert("RGB"), return_tensors="pt")
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return _l2_normalize(cls_token.squeeze(0).cpu().numpy())


def get_semantic_backend(config: AIConfig) -> Optional[SemanticEmbedder]:
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


def get_visual_backend(config: AIConfig) -> Optional[VisualEmbedder]:
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
