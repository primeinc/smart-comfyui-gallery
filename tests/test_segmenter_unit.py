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
import subprocess
import sys
import textwrap
import types

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
    """Simulate a working torch/mobile_sam runtime whose model loads fine."""
    model = types.SimpleNamespace(eval=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", _fake_module("torch"))
    monkeypatch.setitem(
        sys.modules, "mobile_sam",
        _fake_module(
            "mobile_sam",
            SamPredictor=lambda m: types.SimpleNamespace(model=m),
            sam_model_registry={"vit_t": lambda checkpoint: model},
        ),
    )


# --- constructor: fail-closed contract ---------------------------------------


def test_ctor_missing_weights_names_expected_path(tmp_path):
    """Missing weights raise BackendUnavailable naming the exact mobile_sam.pt path."""
    expected = os.path.join(str(tmp_path), "mobile_sam.pt")
    with pytest.raises(BackendUnavailable) as exc_info:
        MobileSamSegmenter(str(tmp_path))
    assert expected in str(exc_info.value)


def test_missing_weights_check_never_imports_torch_or_mobile_sam(tmp_path):
    """With no weights the ctor raises before importing torch/mobile_sam
    (checked in a clean subprocess so suite ordering can't mask a leak)."""
    script = textwrap.dedent(
        f"""
        import sys
        from smartgallery_ai.embedders import BackendUnavailable
        from smartgallery_ai.segmenter_mobilesam import MobileSamSegmenter
        try:
            MobileSamSegmenter({str(tmp_path)!r})
        except BackendUnavailable:
            pass
        else:
            sys.exit("MobileSamSegmenter did not raise BackendUnavailable")
        leaked = [m for m in ("torch", "mobile_sam") if m in sys.modules]
        sys.exit("heavy runtimes imported: %r" % leaked if leaked else 0)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr


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

    def _fail(*args, **kwargs):
        raise boom

    monkeypatch.setitem(sys.modules, "torch", _fake_module("torch"))
    monkeypatch.setitem(
        sys.modules, "mobile_sam",
        _fake_module(
            "mobile_sam",
            SamPredictor=lambda m: None,
            sam_model_registry={"vit_t": _fail},
        ),
    )
    with pytest.raises(BackendUnavailable, match="failed to load mobile_sam weights") as exc_info:
        MobileSamSegmenter(str(tmp_path))
    assert exc_info.value.__cause__ is boom


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
