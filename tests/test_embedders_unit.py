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

import builtins
import logging
import os
import sys
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
    pick_torch_device,
    warn_if_vram_pressure,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(models_dir: str, **overrides) -> AIConfig:
    base = {"enabled": True, "models_dir": models_dir}
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
    assert va1.shape == (64,)
    assert va1.dtype == np.float32
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


_HEAVY = ("torch", "open_clip", "transformers")


@pytest.mark.parametrize("backend",
                         [OpenClipSemanticEmbedder, Dinov2VisualEmbedder])
def test_missing_weights_check_never_imports_heavy_runtimes(tmp_path, backend):
    """With no weights, neither constructor pulls in torch, open_clip or
    transformers -- the gallery has to start on a machine that has none of
    them, and the weights check runs on every start.

    Asked as "did this call import them", not "are they absent from the
    process". The old form needed a clean interpreter precisely because the
    absolute claim is false the moment any other test imports torch, which
    made suite ordering part of the answer. The difference between
    sys.modules before and after is the actual claim, it survives any
    ordering, and it costs nothing.
    """
    before = set(sys.modules)

    with pytest.raises(BackendUnavailable):
        backend(str(tmp_path))

    newly = set(sys.modules) - before
    leaked = sorted(name for name in newly
                    if name in _HEAVY or name.split(".")[0] in _HEAVY)
    assert not leaked, (
        f"{backend.__name__} imported {leaked} just to notice its weights "
        f"were missing")


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

    def _fail(*_args, **_kwargs):
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
        def from_pretrained(*_args, **_kwargs):
            raise OSError("corrupt checkpoint dir")

    monkeypatch.setitem(sys.modules, "torch", _fake_module("torch"))
    monkeypatch.setitem(sys.modules, "torchvision", _fake_module("torchvision"))
    monkeypatch.setitem(
        sys.modules, "transformers",
        _fake_module("transformers", AutoImageProcessor=_FailingAuto, AutoModel=_FailingAuto),
    )
    with pytest.raises(BackendUnavailable, match="failed to load dinov2 weights"):
        Dinov2VisualEmbedder(str(tmp_path))


def test_openclip_ctor_wraps_runtime_error_during_import(tmp_path, monkeypatch):
    """A non-ImportError raised WHILE `import open_clip` executes (e.g. the
    torch/torchvision wheel-index mismatch: 'operator torchvision::nms does
    not exist') is wrapped into BackendUnavailable, and 'auto' resolution
    still degrades to None instead of crashing the caller."""
    _touch_openclip_weights(tmp_path)
    boom = RuntimeError("operator torchvision::nms does not exist")
    real_import = builtins.__import__

    def _exploding_import(name, *args, **kwargs):
        if name == "open_clip":
            raise boom
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "open_clip", raising=False)
    monkeypatch.setattr(builtins, "__import__", _exploding_import)

    with pytest.raises(BackendUnavailable, match="open_clip backend unavailable") as exc_info:
        OpenClipSemanticEmbedder(str(tmp_path))
    assert exc_info.value.__cause__ is boom
    assert get_semantic_backend(_cfg(str(tmp_path), semantic_backend="auto")) is None


def test_dinov2_ctor_wraps_runtime_error_during_import(tmp_path, monkeypatch):
    """Same containment for the visual backend: a RuntimeError during the
    torch/transformers import becomes BackendUnavailable and 'auto' -> None."""
    _make_dinov2_weights_dir(tmp_path)
    boom = RuntimeError("operator torchvision::nms does not exist")
    real_import = builtins.__import__

    def _exploding_import(name, *args, **kwargs):
        if name == "transformers":
            raise boom
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "torch", _fake_module("torch"))
    monkeypatch.setitem(sys.modules, "torchvision", _fake_module("torchvision"))
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    monkeypatch.setattr(builtins, "__import__", _exploding_import)

    with pytest.raises(BackendUnavailable, match="dinov2 backend unavailable") as exc_info:
        Dinov2VisualEmbedder(str(tmp_path))
    assert exc_info.value.__cause__ is boom
    assert get_visual_backend(_cfg(str(tmp_path), visual_backend="auto")) is None


def test_dinov2_ctor_fails_before_transformers_when_torchvision_missing(tmp_path, monkeypatch):
    """When torchvision is missing the ctor must fail WITHOUT importing
    transformers: transformers freezes its torchvision-availability flag at
    import time, so importing it early would keep the visual backend dead
    for the whole process even after auto-provisioning installs torchvision
    (transformers then demands a runtime restart)."""
    _make_dinov2_weights_dir(tmp_path)
    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name == "torchvision":
            raise ImportError("No module named 'torchvision'")
        if name == "transformers":
            raise AssertionError(
                "transformers imported in a torchvision-less process")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "torch", _fake_module("torch"))
    monkeypatch.delitem(sys.modules, "torchvision", raising=False)
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    monkeypatch.setattr(builtins, "__import__", _guarded_import)

    with pytest.raises(BackendUnavailable, match="dinov2 backend unavailable"):
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

    def _fail(*_args, **_kwargs):
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


# --- pick_torch_device: multi-GPU selection and pinning -----------------------


def _fake_torch_with_gpus(cards):
    """torch stand-in with one CUDA device per entry: either a bare
    total_memory int or a (total_memory, cc_major, cc_minor) tuple."""

    def _props(index):
        entry = cards[index]
        if isinstance(entry, tuple):
            mem, major, minor = entry
        else:
            mem, major, minor = entry, 0, 0
        return types.SimpleNamespace(total_memory=mem, major=major, minor=minor)

    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: len(cards),
        get_device_properties=_props,
    )
    return types.SimpleNamespace(cuda=cuda, backends=None)


def test_pick_torch_device_prefers_largest_vram_gpu(monkeypatch):
    """With several CUDA devices the largest-VRAM card is chosen explicitly
    (cuda:<i>), never bare 'cuda' (which means PCI enumeration order)."""
    monkeypatch.delenv("AI_DAM_DEVICE", raising=False)
    torch = _fake_torch_with_gpus([8 << 30, 24 << 30, 12 << 30])
    assert pick_torch_device(torch) == "cuda:1"


def test_pick_torch_device_single_gpu_stays_bare_cuda(monkeypatch):
    """One CUDA device needs no index; a failing enumeration degrades to
    bare 'cuda' instead of crashing device selection."""
    monkeypatch.delenv("AI_DAM_DEVICE", raising=False)
    assert pick_torch_device(_fake_torch_with_gpus([8 << 30])) == "cuda"

    def _boom():
        raise RuntimeError("driver hiccup")
    broken = _fake_torch_with_gpus([1, 2])
    broken.cuda.device_count = _boom
    assert pick_torch_device(broken) == "cuda"


def test_pick_torch_device_role_override_beats_global(monkeypatch):
    """AI_DAM_<ROLE>_DEVICE pins one backend to one card even when the
    global AI_DAM_DEVICE says something else; other roles follow the
    global setting."""
    monkeypatch.setenv("AI_DAM_DEVICE", "cpu")
    monkeypatch.setenv("AI_DAM_VISUAL_DEVICE", "cuda:1")
    torch = _fake_torch_with_gpus([8 << 30])
    assert pick_torch_device(torch, role="visual") == "cuda:1"
    assert pick_torch_device(torch, role="semantic") == "cpu"
    assert pick_torch_device(torch) == "cpu"


def test_pick_torch_device_ties_go_to_the_newer_generation(monkeypatch):
    """Equal VRAM: the higher compute-capability card wins the tie, not
    whichever happens to enumerate first (mixed-generation rigs: an 8GB
    Ampere in slot 0 must not beat an 8GB Blackwell in slot 1)."""
    monkeypatch.delenv("AI_DAM_DEVICE", raising=False)
    torch = _fake_torch_with_gpus([(8 << 30, 8, 6), (8 << 30, 12, 0)])
    assert pick_torch_device(torch) == "cuda:1"


def test_vram_pressure_warns_only_when_the_chosen_card_is_nearly_full(caplog):
    """Loading onto a CUDA card with under ~2 GiB free logs a warning
    naming the device and the escape hatch; ample free VRAM, CPU devices,
    and torch builds without mem_get_info all stay silent."""


    def torch_with_free(free_bytes):
        cuda = types.SimpleNamespace(
            mem_get_info=lambda _index=0: (free_bytes, 16 << 30))
        return types.SimpleNamespace(cuda=cuda)

    with caplog.at_level(logging.WARNING, logger="smartgallery_ai.embedders"):
        warn_if_vram_pressure(torch_with_free(1 << 30), "cuda:1", "model-x")
    assert any("cuda:1" in r.getMessage() and "VRAM free" in r.getMessage()
               for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="smartgallery_ai.embedders"):
        warn_if_vram_pressure(torch_with_free(8 << 30), "cuda:1", "model-x")
        warn_if_vram_pressure(torch_with_free(1 << 30), "cpu", "model-x")
        warn_if_vram_pressure(types.SimpleNamespace(cuda=None), "cuda", "model-x")
    assert not caplog.records
