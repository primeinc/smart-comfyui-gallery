"""Unit tests for smartgallery_ai.segmenter_mobilesam: the fail-closed
constructor contract of MobileSamSegmenter (missing weights raise
BackendUnavailable naming the expected path, BEFORE any torch/mobile_sam
import; missing runtime and bad weights are wrapped, never leaked), the
model identity constants, and segment()'s prompt-required guard.

Model-free: never loads torch/mobile_sam for real -- the runtime is
simulated via sys.modules entries, and the heavy-import check runs in a
clean subprocess. segment()'s actual inference body is covered only by the
opt-in real-backend suite (tests/test_real_backends.py).
"""

from __future__ import annotations

import os
import sys
import types
import warnings

import pytest
from PIL import Image

from smartgallery_ai.embedders import BackendUnavailable
from smartgallery_ai.review import SegmenterBackend
from smartgallery_ai.segmenter_mobilesam import WEIGHTS_FILENAME, MobileSamSegmenter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fake_module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _touch_mobilesam_weights(models_dir) -> str:
    weights = os.path.join(str(models_dir), WEIGHTS_FILENAME)
    with open(weights, "wb") as fh:
        fh.write(b"not a real checkpoint")
    return weights


def _install_fake_runtime(monkeypatch):
    """Simulate a working torch/mobile_sam runtime whose model loads fine
    (including the .to(device) placement every real SAM model supports)."""
    model = types.SimpleNamespace(eval=lambda: None)
    model.to = lambda _device: model

    def _build(checkpoint):
        del checkpoint  # accepted only for the registry's call-signature compatibility (kwarg call)
        return model

    monkeypatch.setitem(sys.modules, "torch", _fake_module("torch"))
    monkeypatch.setitem(
        sys.modules, "mobile_sam",
        _fake_module(
            "mobile_sam",
            SamPredictor=lambda m: types.SimpleNamespace(model=m),
            sam_model_registry={"vit_t": _build},
        ),
    )


# --- constructor: fail-closed contract ---------------------------------------


def test_ctor_missing_weights_names_expected_path(tmp_path):
    """Missing weights raise BackendUnavailable naming the exact mobile_sam.pt path."""
    expected = os.path.join(str(tmp_path), "mobile_sam.pt")
    with pytest.raises(BackendUnavailable) as exc_info:
        MobileSamSegmenter(str(tmp_path))
    assert expected in str(exc_info.value)


_HEAVY = ("torch", "mobile_sam")


def test_missing_weights_check_never_imports_torch_or_mobile_sam(tmp_path):
    """With no weights the ctor raises before importing torch or
    mobile_sam -- the gallery has to start on a machine with neither, and
    the weights check runs on every start.

    Asked as "did this call import them", not "are they absent from the
    process". The old form needed a clean interpreter precisely because the
    absolute claim is false the moment any other test imports torch, which
    made suite ordering part of the answer. The difference between
    sys.modules before and after is the actual claim and survives any
    ordering.
    """
    before = set(sys.modules)

    with pytest.raises(BackendUnavailable):
        MobileSamSegmenter(str(tmp_path))

    newly = set(sys.modules) - before
    leaked = sorted(name for name in newly
                    if name in _HEAVY or name.split(".")[0] in _HEAVY)
    assert not leaked, (
        f"MobileSamSegmenter imported {leaked} just to notice its weights "
        f"were missing")


def test_ctor_missing_runtime_raises_backend_unavailable(tmp_path, monkeypatch):
    """Weights present but runtime not importable -> BackendUnavailable, not ImportError."""
    _touch_mobilesam_weights(tmp_path)
    # A None sys.modules entry makes the import raise ImportError without
    # touching any real (installed) package.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "mobile_sam", None)
    with pytest.raises(BackendUnavailable, match="mobile_sam unavailable"):
        MobileSamSegmenter(str(tmp_path))


def test_ctor_wraps_weight_load_failure(tmp_path, monkeypatch):
    """A failure while loading present-but-bad weights is wrapped into
    BackendUnavailable with the original error chained as __cause__."""
    _touch_mobilesam_weights(tmp_path)
    boom = RuntimeError("corrupt checkpoint")

    def _fail(*_args, **_kwargs):
        raise boom

    monkeypatch.setitem(sys.modules, "torch", _fake_module("torch"))
    monkeypatch.setitem(
        sys.modules, "mobile_sam",
        _fake_module(
            "mobile_sam",
            SamPredictor=lambda _m: None,
            sam_model_registry={"vit_t": _fail},
        ),
    )
    with pytest.raises(BackendUnavailable, match="failed to load mobile_sam weights") as exc_info:
        MobileSamSegmenter(str(tmp_path))
    assert exc_info.value.__cause__ is boom


def test_ctor_contains_third_party_warning_and_stderr_noise(tmp_path, monkeypatch, capsys):
    """Construction keeps the server console clean: timm-style
    FutureWarnings/UserWarnings and tqdm-style stderr output emitted while
    the model builds never reach the caller, and the model still loads."""
    _touch_mobilesam_weights(tmp_path)
    model = types.SimpleNamespace(eval=lambda: None)
    model.to = lambda _device: model

    def _noisy_build(checkpoint):
        del checkpoint  # accepted only for the registry's call-signature compatibility (kwarg call)
        warnings.warn("Importing from timm.models.layers is deprecated", FutureWarning, stacklevel=2)
        warnings.warn("Overwriting tiny_vit_5m_224 in registry", UserWarning, stacklevel=2)
        print("Loading weights: 100%|#| 223/223", file=sys.stderr)
        return model

    monkeypatch.setitem(sys.modules, "torch", _fake_module("torch"))
    monkeypatch.setitem(
        sys.modules, "mobile_sam",
        _fake_module(
            "mobile_sam",
            SamPredictor=lambda m: types.SimpleNamespace(model=m),
            sam_model_registry={"vit_t": _noisy_build},
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        seg = MobileSamSegmenter(str(tmp_path))
    assert seg is not None
    assert caught == []
    assert capsys.readouterr().err == ""


# --- model identity ----------------------------------------------------------


def test_model_identity_constants_readable_without_weights():
    """The class exposes the documented model_id/model_version (used to key
    derived rows) as class attributes, and is a SegmenterBackend."""
    assert MobileSamSegmenter.model_id == "ChaoningZhang/MobileSAM"
    assert MobileSamSegmenter.model_version == "mobile_sam-vit_t-v1"
    assert issubclass(MobileSamSegmenter, SegmenterBackend)
    assert WEIGHTS_FILENAME == "mobile_sam.pt"


# --- segment(): prompt-required guard ----------------------------------------


def test_segment_without_prompt_raises_value_error(tmp_path, monkeypatch):
    """segment() with neither bbox nor points raises ValueError (the
    anti-fabrication guard: no mask without real grounding), and an empty
    points list counts as no prompt."""
    _touch_mobilesam_weights(tmp_path)
    _install_fake_runtime(monkeypatch)
    seg = MobileSamSegmenter(str(tmp_path))
    img = Image.new("RGB", (8, 8), (128, 128, 128))
    with pytest.raises(ValueError, match="requires a bbox or points prompt"):
        seg.segment(img)
    with pytest.raises(ValueError, match="requires a bbox or points prompt"):
        seg.segment(img, bbox=None, points=[])
