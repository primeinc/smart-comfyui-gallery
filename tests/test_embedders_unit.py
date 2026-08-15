"""Unit tests for smartgallery_ai.embedders: stub embedder vector contracts
(determinism, dim, L2 norm, input sensitivity), the fail-closed constructor
contract of the real backends when weights/runtimes are missing (including
"weights check precedes the heavy runtime import"), and the factory
resolution policy of get_semantic_backend / get_visual_backend.

Model-free: never loads torch/open_clip/transformers for real -- missing
runtimes are simulated via sys.modules entries, and the heavy-import check
runs in a clean subprocess.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import types

import numpy as np
import pytest
from PIL import Image

from smartgallery_ai import AIConfig
from smartgallery_ai.embedders import (
    BackendUnavailable,
    Dinov2VisualEmbedder,
    OpenClipSemanticEmbedder,
    StubSemanticEmbedder,
    StubVisualEmbedder,
    get_semantic_backend,
    get_visual_backend,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(models_dir: str, **overrides) -> AIConfig:
    base = dict(enabled=True, models_dir=models_dir)
    base.update(overrides)
    return AIConfig(**base)


def _fake_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _touch_openclip_weights(models_dir) -> str:
    weights = os.path.join(str(models_dir), "open_clip", "ViT-B-32_laion2b_s34b_b79k.bin")
    os.makedirs(os.path.dirname(weights), exist_ok=True)
    with open(weights, "wb") as fh:
        fh.write(b"not a real checkpoint")
    return weights


def _make_dinov2_weights_dir(models_dir) -> str:
    weights_dir = os.path.join(str(models_dir), "dinov2-small")
    os.makedirs(weights_dir, exist_ok=True)
    return weights_dir


# --- stub embedders: vector contracts ----------------------------------------


def test_stub_semantic_embed_text_deterministic_unit_vector():
    """Same text always maps to the same L2-normalized float32 vector of dim 64."""
    emb = StubSemanticEmbedder()
    v1 = emb.embed_text("a red apple on a table")
    v2 = emb.embed_text("a red apple on a table")
    assert v1.shape == (emb.dim,) == (64,)
    assert v1.dtype == np.float32
    assert np.linalg.norm(v1) == pytest.approx(1.0, abs=1e-5)
    assert np.array_equal(v1, v2)


def test_stub_semantic_embed_text_distinct_texts_distinct_vectors():
    """Different texts produce different vectors (the space is not degenerate)."""
    emb = StubSemanticEmbedder()
    assert not np.array_equal(
        emb.embed_text("a red apple on a table"),
        emb.embed_text("a blue car driving at night"),
    )


def test_stub_semantic_embed_text_is_case_insensitive():
    """Text embedding lowercases input: casing never changes the vector."""
    emb = StubSemanticEmbedder()
    assert np.array_equal(emb.embed_text("Red APPLE"), emb.embed_text("red apple"))


def test_stub_semantic_embed_text_short_text_falls_back_to_whole_string():
    """Texts shorter than one trigram (incl. empty) still embed to a unit vector."""
    emb = StubSemanticEmbedder()
    for text in ("", "a", "hi"):
        vec = emb.embed_text(text)
        assert vec.shape == (64,)
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-5)
        assert np.array_equal(vec, emb.embed_text(text))
    assert not np.array_equal(emb.embed_text("hi"), emb.embed_text("a"))


def test_stub_semantic_embed_image_deterministic_and_input_sensitive():
    """Same image -> identical unit vector; different images -> different vectors."""
    emb = StubSemanticEmbedder()
    rng = np.random.default_rng(42)
    img_a = Image.fromarray((rng.random((32, 32, 3)) * 255).astype("uint8"))
    img_b = Image.new("RGB", (32, 32), (255, 255, 255))
    va1, va2, vb = emb.embed_image(img_a), emb.embed_image(img_a), emb.embed_image(img_b)
    assert va1.shape == (64,) and va1.dtype == np.float32
    assert np.linalg.norm(va1) == pytest.approx(1.0, abs=1e-5)
    assert np.array_equal(va1, va2)
    assert not np.array_equal(va1, vb)


def test_stub_visual_solid_color_is_one_hot_histogram_bucket():
    """A solid-color image L2-normalizes to exactly one histogram bucket == 1.0."""
    emb = StubVisualEmbedder()
    # (255, 20, 20) -> bins (3, 0, 0) -> index 3*16 + 0*4 + 0 = 48
    vec = emb.embed_image(Image.new("RGB", (16, 16), (255, 20, 20)))
    assert vec.shape == (emb.dim,) == (64,)
    assert vec[48] == pytest.approx(1.0)
    assert np.count_nonzero(vec) == 1


def test_stub_visual_deterministic_and_distinct_across_palettes():
    """Same image -> identical vector; different color content -> different vector."""
    emb = StubVisualEmbedder()
    red = Image.new("RGB", (16, 16), (200, 0, 0))
    blue = Image.new("RGB", (16, 16), (0, 0, 200))
    assert np.array_equal(emb.embed_image(red), emb.embed_image(red))
    assert not np.array_equal(emb.embed_image(red), emb.embed_image(blue))


# --- real backend constructors: fail-closed without weights ------------------


def test_openclip_ctor_missing_weights_names_expected_path(tmp_path):
    """Missing weights raise BackendUnavailable naming the exact expected file path."""
    expected = os.path.join(str(tmp_path), "open_clip", "ViT-B-32_laion2b_s34b_b79k.bin")
    with pytest.raises(BackendUnavailable) as exc_info:
        OpenClipSemanticEmbedder(str(tmp_path))
    assert expected in str(exc_info.value)


def test_dinov2_ctor_missing_weights_names_expected_dir(tmp_path):
    """Missing weights dir raises BackendUnavailable naming the exact expected dir."""
    expected = os.path.join(str(tmp_path), "dinov2-small")
    with pytest.raises(BackendUnavailable) as exc_info:
        Dinov2VisualEmbedder(str(tmp_path))
    assert expected in str(exc_info.value)


def test_missing_weights_check_never_imports_heavy_runtimes(tmp_path):
    """With no weights, neither ctor imports torch/open_clip/transformers
    (checked in a clean subprocess so suite ordering can't mask a leak)."""
    script = textwrap.dedent(
        f"""
        import sys
        from smartgallery_ai.embedders import (
            BackendUnavailable, Dinov2VisualEmbedder, OpenClipSemanticEmbedder,
        )
        for cls in (OpenClipSemanticEmbedder, Dinov2VisualEmbedder):
            try:
                cls({str(tmp_path)!r})
            except BackendUnavailable:
                pass
            else:
                sys.exit(cls.__name__ + " did not raise BackendUnavailable")
        leaked = [m for m in ("torch", "open_clip", "transformers")
                  if m in sys.modules]
        sys.exit("heavy runtimes imported: %r" % leaked if leaked else 0)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr


def test_openclip_ctor_missing_runtime_raises_backend_unavailable(tmp_path, monkeypatch):
    """Weights present but open_clip not importable -> BackendUnavailable, not ImportError."""
    _touch_openclip_weights(tmp_path)
    # A None sys.modules entry makes `import open_clip` raise ImportError
    # without touching the real (installed) package.
    monkeypatch.setitem(sys.modules, "open_clip", None)
    with pytest.raises(BackendUnavailable, match="open_clip backend unavailable"):
        OpenClipSemanticEmbedder(str(tmp_path))


def test_dinov2_ctor_missing_runtime_raises_backend_unavailable(tmp_path, monkeypatch):
    """Weights present but torch not importable -> BackendUnavailable, not ImportError."""
    _make_dinov2_weights_dir(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(BackendUnavailable, match="dinov2 backend unavailable"):
        Dinov2VisualEmbedder(str(tmp_path))


def test_openclip_ctor_wraps_weight_load_failure(tmp_path, monkeypatch):
    """A failure while loading present-but-bad weights is wrapped into
    BackendUnavailable with the original error chained as __cause__."""
    _touch_openclip_weights(tmp_path)
    boom = RuntimeError("corrupt checkpoint")

    def _fail(*args, **kwargs):
        raise boom

    monkeypatch.setitem(sys.modules, "torch", _fake_module("torch"))
    monkeypatch.setitem(
        sys.modules, "open_clip",
        _fake_module("open_clip", create_model_and_transforms=_fail),
    )
    with pytest.raises(BackendUnavailable, match="failed to load open_clip weights") as exc_info:
        OpenClipSemanticEmbedder(str(tmp_path))
    assert exc_info.value.__cause__ is boom


def test_dinov2_ctor_wraps_weight_load_failure(tmp_path, monkeypatch):
    """A transformers load failure on a present weights dir becomes BackendUnavailable."""
    _make_dinov2_weights_dir(tmp_path)

    class _FailingAuto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise OSError("corrupt checkpoint dir")

    monkeypatch.setitem(sys.modules, "torch", _fake_module("torch"))
    monkeypatch.setitem(
        sys.modules, "transformers",
        _fake_module("transformers", AutoImageProcessor=_FailingAuto, AutoModel=_FailingAuto),
    )
    with pytest.raises(BackendUnavailable, match="failed to load dinov2 weights"):
        Dinov2VisualEmbedder(str(tmp_path))


# --- factory resolution policy -----------------------------------------------


def test_get_semantic_backend_resolution_policy(tmp_path):
    """'none'->None, 'stub'->stub, 'auto' without weights degrades to None
    (never the stub), explicit 'open_clip' without weights raises, unknown
    name -> ValueError naming it."""
    assert get_semantic_backend(_cfg(str(tmp_path), semantic_backend="none")) is None
    assert isinstance(
        get_semantic_backend(_cfg(str(tmp_path), semantic_backend="stub")),
        StubSemanticEmbedder,
    )
    assert get_semantic_backend(_cfg(str(tmp_path), semantic_backend="auto")) is None
    with pytest.raises(BackendUnavailable):
        get_semantic_backend(_cfg(str(tmp_path), semantic_backend="open_clip"))
    with pytest.raises(ValueError, match="unknown semantic_backend: 'needle2'"):
        get_semantic_backend(_cfg(str(tmp_path), semantic_backend="needle2"))


def test_get_visual_backend_resolution_policy(tmp_path):
    """'none'->None, 'stub'->stub, 'auto' without weights degrades to None
    (never the stub), explicit 'dinov2' without weights raises, unknown
    name -> ValueError naming it."""
    assert get_visual_backend(_cfg(str(tmp_path), visual_backend="none")) is None
    assert isinstance(
        get_visual_backend(_cfg(str(tmp_path), visual_backend="stub")),
        StubVisualEmbedder,
    )
    assert get_visual_backend(_cfg(str(tmp_path), visual_backend="auto")) is None
    with pytest.raises(BackendUnavailable):
        get_visual_backend(_cfg(str(tmp_path), visual_backend="dinov2"))
    with pytest.raises(ValueError, match="unknown visual_backend: 'clip'"):
        get_visual_backend(_cfg(str(tmp_path), visual_backend="clip"))


def test_auto_degrades_to_none_even_when_weight_loading_fails(tmp_path, monkeypatch):
    """'auto' swallows load-time BackendUnavailable too: bad weights on disk
    still resolve to None instead of crashing the caller."""
    _touch_openclip_weights(tmp_path)

    def _fail(*args, **kwargs):
        raise RuntimeError("corrupt checkpoint")

    monkeypatch.setitem(sys.modules, "torch", _fake_module("torch"))
    monkeypatch.setitem(
        sys.modules, "open_clip",
        _fake_module("open_clip", create_model_and_transforms=_fail),
    )
    assert get_semantic_backend(_cfg(str(tmp_path), semantic_backend="auto")) is None


def test_stub_backends_expose_model_identity_and_dim():
    """Stub backends report the documented model_id/model_version/dim used to
    key derived rows in the vector store."""
    sem = get_semantic_backend(_cfg("", semantic_backend="stub"))
    vis = get_visual_backend(_cfg("", visual_backend="stub"))
    assert (sem.model_id, sem.model_version, sem.dim) == ("stub-semantic", "stub-v1", 64)
    assert (vis.model_id, vis.model_version, vis.dim) == ("stub-visual", "stub-v1", 64)
