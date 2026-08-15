"""Opt-in integration tests against the REAL provisioned model backends.

These prove the actual model stack (not stubs) end to end. They are skipped
unless RUN_REAL_BACKEND_TESTS=1 AND the corresponding weights exist under
the models directory, because they load real checkpoints (seconds each) and
require the optional torch/transformers/open_clip runtimes.

Run:  RUN_REAL_BACKEND_TESTS=1 python -m pytest tests/test_real_backends.py -v
"""

import os

import numpy as np
import pytest
from PIL import Image, ImageEnhance

from smartgallery_ai import AIConfig

MODELS_DIR = os.environ.get(
    "AI_DAM_MODELS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 ".AImodels"),
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_BACKEND_TESTS") != "1",
    reason="real-backend tests are opt-in (RUN_REAL_BACKEND_TESTS=1)",
)


def _cfg(**overrides) -> AIConfig:
    base = dict(enabled=True, models_dir=MODELS_DIR)
    base.update(overrides)
    return AIConfig(**base)


def _test_images():
    """A textured image, a mild edit of it, and a structurally different
    smooth gradient. (Two independent noise textures are NOT a good
    dissimilar pair: DINOv2 legitimately embeds them close together.)"""
    rng = np.random.default_rng(20)
    base = Image.fromarray(
        (rng.random((64, 64, 3)) * 255).astype("uint8")
    ).resize((448, 448), Image.LANCZOS)
    edited = ImageEnhance.Contrast(base).enhance(1.3)
    yy, xx = np.mgrid[0:448, 0:448].astype(np.float32) / 448.0
    gradient = Image.fromarray(
        np.stack([xx * 255, yy * 255, (1 - xx) * 255], axis=-1).astype("uint8"))
    return base, edited, gradient


def _cos(a, b):
    return float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))


def test_real_visual_backend_dinov2_separates_similarity():
    if not os.path.isdir(os.path.join(MODELS_DIR, "dinov2-small")):
        pytest.skip("dinov2-small weights not provisioned")
    from smartgallery_ai.embedders import get_visual_backend

    vis = get_visual_backend(_cfg(visual_backend="dinov2"))
    assert vis is not None and vis.dim == 384
    base, edited, gradient = _test_images()
    vb, ve, vg = vis.embed_image(base), vis.embed_image(edited), vis.embed_image(gradient)
    assert _cos(vb, ve) > 0.8
    assert _cos(vb, ve) > _cos(vb, vg) + 0.3


def test_real_semantic_backend_openclip_text_to_image():
    if not os.path.isfile(os.path.join(
            MODELS_DIR, "open_clip", "ViT-B-32_laion2b_s34b_b79k.bin")):
        pytest.skip("open_clip weights not provisioned")
    from smartgallery_ai.embedders import get_semantic_backend

    sem = get_semantic_backend(_cfg(semantic_backend="open_clip"))
    assert sem is not None and sem.dim == 512
    # Text->image: a red image should match "a solid red image" over
    # unrelated text, and over a blue image for the same text.
    red = Image.new("RGB", (224, 224), (220, 20, 20))
    blue = Image.new("RGB", (224, 224), (20, 20, 220))
    ir, ib = sem.embed_image(red), sem.embed_image(blue)
    t_red = sem.embed_text("a solid red image")
    t_dog = sem.embed_text("a photo of a dog in a park")
    assert _cos(t_red, ir) > _cos(t_dog, ir)
    assert _cos(t_red, ir) > _cos(t_red, ib)


def test_real_face_backend_yunet_sface():
    for f in ("face_detection_yunet_2023mar.onnx",
              "face_recognition_sface_2021dec.onnx"):
        if not os.path.isfile(os.path.join(MODELS_DIR, f)):
            pytest.skip(f"{f} not provisioned")
    from smartgallery_ai.faces import OpenCVFaceBackend

    backend = OpenCVFaceBackend(MODELS_DIR)
    assert backend.model_id == "opencv/yunet+sface"
    # No real photo ships in the repo; prove the wiring on a blank image
    # (must return an empty list, not crash) and full geometry/embedding
    # invariants when a face IS present via any provisioned test photo.
    blank = Image.new("RGB", (320, 320), (128, 128, 128))
    assert backend.detect(blank) == []

    probe_photo = os.environ.get("REAL_FACE_TEST_IMAGE")
    if probe_photo and os.path.isfile(probe_photo):
        dets = backend.detect(Image.open(probe_photo).convert("RGB"))
        assert dets, "expected at least one face in REAL_FACE_TEST_IMAGE"
        d = dets[0]
        assert all(0.0 <= v <= 1.0 for v in d.bbox)
        assert d.embedding is not None and d.embedding.shape == (128,)


def test_real_segmenter_mobilesam_box_prompt_iou():
    if not os.path.isfile(os.path.join(MODELS_DIR, "mobile_sam.pt")):
        pytest.skip("mobile_sam.pt not provisioned")
    from smartgallery_ai.segmenter_mobilesam import MobileSamSegmenter

    seg = MobileSamSegmenter(MODELS_DIR)
    # Solid red square on a textured background; a loose box prompt around
    # it must segment the square precisely (high IoU vs ground truth).
    rng = np.random.default_rng(31)
    base = Image.fromarray(
        (rng.random((64, 64, 3)) * 255).astype("uint8")
    ).resize((512, 512), Image.LANCZOS)
    from PIL import ImageDraw
    ImageDraw.Draw(base).rectangle([300, 300, 419, 419], fill=(255, 20, 20))
    gt = np.zeros((512, 512), bool)
    gt[300:420, 300:420] = True

    mask = seg.segment(base, bbox=(0.55, 0.55, 0.28, 0.28))
    iou = (mask & gt).sum() / (mask | gt).sum()
    assert iou > 0.7, f"MobileSAM IoU too low: {iou:.3f}"


def test_segmenter_factory_resolution(tmp_path):
    """Factory policy: 'auto' degrades to None without weights; 'mobilesam'
    raises; 'none'/'stub' behave as documented. Model-free (empty dir)."""
    from smartgallery_ai import AIConfig
    from smartgallery_ai.embedders import BackendUnavailable
    from smartgallery_ai.review import StubSegmenter, get_segmenter_backend

    assert get_segmenter_backend(
        AIConfig(enabled=True, models_dir=str(tmp_path),
                 segmenter_backend="none")) is None
    assert get_segmenter_backend(
        AIConfig(enabled=True, models_dir=str(tmp_path),
                 segmenter_backend="auto")) is None
    assert isinstance(get_segmenter_backend(
        AIConfig(enabled=True, models_dir=str(tmp_path),
                 segmenter_backend="stub")), StubSegmenter)
    with pytest.raises(BackendUnavailable):
        get_segmenter_backend(
            AIConfig(enabled=True, models_dir=str(tmp_path),
                     segmenter_backend="mobilesam"))
