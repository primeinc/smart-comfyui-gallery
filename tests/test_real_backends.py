"""Opt-in integration tests against the REAL provisioned model backends.

These prove the actual model stack (not stubs) end to end. They are skipped
unless RUN_REAL_BACKEND_TESTS=1 AND the corresponding weights exist under
the models directory, because they load real checkpoints (seconds each) and
require the optional torch/transformers/open_clip runtimes.

Run:  RUN_REAL_BACKEND_TESTS=1 python -m pytest tests/test_real_backends.py -v
"""

import os
import sqlite3
import tempfile
import time as _time

import numpy as np
import pytest
from flask import Flask
from PIL import Image, ImageDraw, ImageEnhance

from smartgallery_ai import AIConfig, schema
from smartgallery_ai import models as ai_models
from smartgallery_ai.embedders import BackendUnavailable, get_semantic_backend, get_visual_backend
from smartgallery_ai.faces import OpenCVFaceBackend
from smartgallery_ai.review import StubSegmenter, get_segmenter_backend
from smartgallery_ai.reviewer import DEFAULT_REVIEW_MODEL, UngroundedReviewError, check_grounding
from smartgallery_ai.segmenter_mobilesam import MobileSamSegmenter
from smartgallery_ai.service import create_ai_blueprint
from smartgallery_ai.worker import AIWorker

MODELS_DIR = os.environ.get(
    "AI_DAM_MODELS_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".AImodels"),
)

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_BACKEND_TESTS") != "1",
    reason="real-backend tests are opt-in (RUN_REAL_BACKEND_TESTS=1)",
)


def _cfg(**overrides) -> AIConfig:
    base = {"enabled": True, "models_dir": MODELS_DIR}
    base.update(overrides)
    return AIConfig(**base)


def _test_images():
    """A textured image, a mild edit of it, and a structurally different
    smooth gradient. (Two independent noise textures are NOT a good
    dissimilar pair: DINOv2 legitimately embeds them close together.)"""
    rng = np.random.default_rng(20)
    base = Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8")).resize((448, 448), Image.LANCZOS)
    edited = ImageEnhance.Contrast(base).enhance(1.3)
    yy, xx = np.mgrid[0:448, 0:448].astype(np.float32) / 448.0
    gradient = Image.fromarray(np.stack([xx * 255, yy * 255, (1 - xx) * 255], axis=-1).astype("uint8"))
    return base, edited, gradient


def _cos(a, b):
    return float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)))


def test_real_visual_backend_dinov2_separates_similarity():
    if not os.path.isdir(os.path.join(MODELS_DIR, "dinov2-small")):
        pytest.skip("dinov2-small weights not provisioned")

    vis = get_visual_backend(_cfg(visual_backend="dinov2"))
    assert vis is not None
    assert vis.dim == 384
    base, edited, gradient = _test_images()
    vb, ve, vg = vis.embed_image(base), vis.embed_image(edited), vis.embed_image(gradient)
    assert _cos(vb, ve) > 0.8
    assert _cos(vb, ve) > _cos(vb, vg) + 0.3


def test_real_semantic_backend_openclip_text_to_image():
    if not os.path.isfile(os.path.join(MODELS_DIR, "open_clip", "ViT-B-32_laion2b_s34b_b79k.bin")):
        pytest.skip("open_clip weights not provisioned")

    sem = get_semantic_backend(_cfg(semantic_backend="open_clip"))
    assert sem is not None
    assert sem.dim == 512
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
    for f in ("face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx"):
        if not os.path.isfile(os.path.join(MODELS_DIR, f)):
            pytest.skip(f"{f} not provisioned")

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
        assert d.embedding is not None
        assert d.embedding.shape == (128,)


def test_real_segmenter_mobilesam_box_prompt_iou():
    if not os.path.isfile(os.path.join(MODELS_DIR, "mobile_sam.pt")):
        pytest.skip("mobile_sam.pt not provisioned")

    seg = MobileSamSegmenter(MODELS_DIR)
    # Solid red square on a textured background; a loose box prompt around
    # it must segment the square precisely (high IoU vs ground truth).
    rng = np.random.default_rng(31)
    base = Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8")).resize((512, 512), Image.LANCZOS)
    ImageDraw.Draw(base).rectangle([300, 300, 419, 419], fill=(255, 20, 20))
    gt = np.zeros((512, 512), bool)
    gt[300:420, 300:420] = True

    mask = seg.segment(base, bbox=(0.55, 0.55, 0.28, 0.28))
    iou = (mask & gt).sum() / (mask | gt).sum()
    assert iou > 0.7, f"MobileSAM IoU too low: {iou:.3f}"


def test_segmenter_factory_resolution(tmp_path):
    """Factory policy: 'auto' degrades to None without weights; 'mobilesam'
    raises; 'none'/'stub' behave as documented. Model-free (empty dir)."""

    assert get_segmenter_backend(AIConfig(enabled=True, models_dir=str(tmp_path), segmenter_backend="none")) is None
    assert get_segmenter_backend(AIConfig(enabled=True, models_dir=str(tmp_path), segmenter_backend="auto")) is None
    assert isinstance(
        get_segmenter_backend(AIConfig(enabled=True, models_dir=str(tmp_path), segmenter_backend="stub")), StubSegmenter
    )
    with pytest.raises(BackendUnavailable):
        get_segmenter_backend(AIConfig(enabled=True, models_dir=str(tmp_path), segmenter_backend="mobilesam"))


def test_real_grounding_gate_negative_cases():
    """The reviewer's anti-fabrication gate, proven on the REAL OpenCLIP
    space without loading the VLM: a grounded description passes, the
    previously-measured fabricated description and an unrelated one raise
    UngroundedReviewError."""
    if not os.path.isfile(os.path.join(MODELS_DIR, "open_clip", "ViT-B-32_laion2b_s34b_b79k.bin")):
        pytest.skip("open_clip weights not provisioned")

    sem = get_semantic_backend(_cfg(semantic_backend="open_clip"))
    red = Image.new("RGB", (224, 224), (220, 20, 20))
    margin = check_grounding(sem, "a plain solid red image", red)
    assert margin >= 0.09
    # Unrelated content: rejected.
    with pytest.raises(UngroundedReviewError):
        check_grounding(sem, "a portrait photo of an astronaut in a spacesuit", red)
    # Empty: rejected.
    with pytest.raises(UngroundedReviewError):
        check_grounding(sem, "", red)
    # Adversarial classes from the oracle review — v2's contrastive margin
    # rejects what the v1 absolute-cosine gate accepted:
    # (a) vacuous description == the baseline -> margin ~ 0
    with pytest.raises(UngroundedReviewError):
        check_grounding(sem, "This is an image. It contains some shapes and colors.", red)
    # (b) the parroted schema example on an image it does not describe
    with pytest.raises(UngroundedReviewError):
        check_grounding(
            sem,
            "Good portrait with one artifact. The image shows a "
            "red square artifact in the lower right and slightly "
            "flat lighting.",
            Image.new("RGB", (224, 224), (20, 20, 220)),
        )


def test_real_critic_to_mask_chain():
    """FULL AC6+AC7 chain with zero stubs: the real reviewer reviews a
    flawed image -> validated typed findings -> worker generates real
    MobileSAM masks -> API serves them. Minutes, so it needs
    RUN_REAL_CRITIC_TESTS=1 on top of the suite's own opt-in."""
    if os.environ.get("RUN_REAL_CRITIC_TESTS") != "1":
        pytest.skip("critic chain test is a second-tier opt-in (RUN_REAL_CRITIC_TESTS=1)")

    if not ai_models.is_provisioned(DEFAULT_REVIEW_MODEL, MODELS_DIR):
        pytest.skip(f"{DEFAULT_REVIEW_MODEL} not provisioned")
    for f in ("mobile_sam.pt", os.path.join("open_clip", "ViT-B-32_laion2b_s34b_b79k.bin")):
        if not os.path.isfile(os.path.join(MODELS_DIR, f)):
            pytest.skip(f"{f} not provisioned")

    tmp = tempfile.mkdtemp(prefix="sg_critic_chain_")
    media = os.path.join(tmp, "m")
    os.makedirs(media)
    rng = np.random.default_rng(17)
    img = Image.fromarray((rng.random((64, 64, 3)) * 255).astype("uint8")).resize((512, 512), Image.LANCZOS)
    ImageDraw.Draw(img).rectangle([300, 300, 419, 419], fill=(255, 20, 20))
    path = os.path.join(media, "flawed.png")
    img.save(path)

    db = os.path.join(tmp, "g.sqlite")
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE files (id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE,"
        " mtime REAL NOT NULL, name TEXT, type TEXT, workflow_prompt TEXT)"
    )
    schema.init_schema(conn)
    conn.execute(
        "INSERT INTO files VALUES ('fc1', ?, ?, 'flawed.png', 'image', 'abstract colorful texture')",
        (path, os.path.getmtime(path)),
    )
    conn.commit()
    conn.close()

    cfg = _cfg(
        base_path=tmp,
        db_path=db,
        cache_dir=os.path.join(tmp, "cache"),
        semantic_backend="open_clip",
        visual_backend="none",
        face_backend="none",
        critic_backend="vlm",
        segmenter_backend="auto",
    )
    worker = AIWorker(cfg, db, poll_interval=0.2)
    worker.start()
    try:
        deadline = _time.time() + 900
        stored = False
        while _time.time() < deadline and not stored:
            c = sqlite3.connect(db)
            stored = c.execute("SELECT 1 FROM ai_scan_log WHERE file_id='fc1' AND kind='review'").fetchone() is not None
            c.close()
            if not stored:
                _time.sleep(5)
    finally:
        worker.stop(timeout=30)
    assert stored, f"review never completed; worker stats: {worker.stats}"

    c = sqlite3.connect(db)
    c.row_factory = sqlite3.Row
    review_row = c.execute("SELECT * FROM ai_reviews WHERE file_id='fc1'").fetchone()
    findings = c.execute("SELECT * FROM ai_review_findings WHERE file_id='fc1'").fetchall()
    c.close()
    assert review_row is not None
    assert 0.0 <= review_row["quality_score"] <= 10.0
    # prompt was provided -> alignment must be a real score, never None
    assert review_row["prompt_alignment_score"] is not None
    assert findings, "critic emitted no findings for a defective image"
    localizable = [f for f in findings if f["localizable"]]
    for f in localizable:
        assert f["mask_path"], "localizable finding missing a real mask"

    app = Flask(__name__)
    app.register_blueprint(create_ai_blueprint(cfg), url_prefix="/aidam")
    client = app.test_client()
    body = client.get("/aidam/review/fc1").get_json()
    assert body["review"] is not None
    assert len(body["findings"]) == len(findings)
    served = 0
    for f in body["findings"]:
        if f.get("mask_url"):
            resp = client.get(f["mask_url"].replace("/galleryout/api/aidam", "/aidam"))
            assert resp.status_code == 200
            assert len(resp.data) > 100
            served += 1
    if localizable:
        assert served > 0, "no masks served through the API"


# --- ai_models.Chat against real Qwen3-VL weights ---------------------------------
#
# The model-free counterparts are in tests/test_vlm_unit.py. These prove the
# two claims that only real weights can settle: the vision budget actually
# shrinks the encoded image, and a cached follow-up turn returns the same
# answer as a full re-encode.

CHAT_REF = os.environ.get("AI_DAM_VLM_MODEL", "Qwen/Qwen3-VL-2B-Instruct")


def _chat_model_or_skip():

    try:
        return ai_models, ai_models.load(CHAT_REF, models_dir=MODELS_DIR)
    except ai_models.ModelUnavailable as exc:
        pytest.skip(f"{CHAT_REF} not available: {exc}")


def _shapes_image():
    """A wide image with two unambiguous shapes, so a wrong answer is a
    real regression rather than a judgement call."""

    img = Image.new("RGB", (1920, 1080), (18, 22, 40))
    draw = ImageDraw.Draw(img)
    draw.ellipse([1300, 300, 1700, 700], fill=(250, 210, 60))
    draw.rectangle([200, 600, 700, 950], fill=(200, 60, 60))
    return img


def test_real_vlm_vision_budget_shrinks_the_encoded_image():
    ai_models, (processor, _model) = _chat_model_or_skip()
    img = _shapes_image()

    def image_tokens(**kwargs):
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe."}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out = processor(text=text, images=[img], return_tensors="pt", **kwargs)
        ids = out["input_ids"][0]
        return int((ids == processor.image_token_id).sum())

    default_tokens = image_tokens()
    budgeted_tokens = image_tokens(**ai_models.vision_budget(processor, 576))

    # The budget must bite, and land on the requested count.
    assert budgeted_tokens <= 576
    assert budgeted_tokens < default_tokens

    # `max_pixels` on its own is silently dropped by the image processor --
    # this is the regression that made every review encode a full-size image.
    assert image_tokens(max_pixels=576 * 1024) == default_tokens


def test_real_vlm_cached_turns_answer_the_same_as_a_full_re_encode():
    ai_models, _ = _chat_model_or_skip()
    img = _shapes_image()
    questions = [
        "What colour is the circle? One word.",
        "How many shapes are there? One word.",
        "Is the background dark or light? One word.",
    ]
    system = "Answer in one short sentence."

    # Cached: one Chat, image encoded once, three questions.
    chat = ai_models.Chat(CHAT_REF, [img], models_dir=MODELS_DIR, system=system)
    cached = [chat.ask(question, max_new_tokens=24) for question in questions]

    # Re-encoded: a fresh Chat per question, image encoded three times.
    fresh = [
        ai_models.Chat(CHAT_REF, [img], models_dir=MODELS_DIR, system=system).ask(question, max_new_tokens=24)
        for question in questions
    ]

    assert all(answer.strip() for answer in cached)
    # Greedy decoding plus an intact prefix means these must agree exactly.
    # A mismatch says the cached sequence drifted out of alignment.
    assert cached[0].lower().startswith(fresh[0].lower()[:6])
    assert "yellow" in cached[0].lower()
    assert "two" in cached[1].lower() or "2" in cached[1]
    assert "dark" in cached[2].lower()


def test_real_vlm_returns_a_parsed_tool_call():
    ai_models, _ = _chat_model_or_skip()
    schema = {
        "type": "object",
        "properties": {
            "shapes": {"type": "integer", "description": "How many shapes"},
            "dominant_colour": {"type": "string"},
        },
        "required": ["shapes", "dominant_colour"],
    }
    chat = ai_models.Chat(
        CHAT_REF,
        [_shapes_image()],
        models_dir=MODELS_DIR,
        system="You inspect images precisely.",
        tools=[ai_models.tool("report", "Report the image contents.", schema)],
    )

    payload = chat.ask_json("Call report for this image.")

    assert isinstance(payload, dict)
    assert payload["shapes"] == 2
    assert isinstance(payload["dominant_colour"], str)
