"""Model-free unit tests for smartgallery_ai.critic_qwen beyond the critic
tests in tests/test_review.py: the contrastive grounding gate's fail-closed
NaN handling and inclusive (>=) thresholds, per-finding crop verification
(bounds clamping, degenerate-crop rejection, below-baseline rejection),
clip_score_10 mapping, _first_existing preference order, _data_uri encoding,
the constructor's weights-before-llama-import availability contract,
model_version derivation from the provisioned filename, and the review()
localization policy (whole-image / non-spatial / malformed-bbox demotion)
via a monkeypatched _chat. No VLM or CLIP weights are ever loaded."""

import base64
import io
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest
from PIL import Image

from smartgallery_ai import critic_qwen as CQ
from smartgallery_ai.critic_qwen import (
    GROUNDING_BASELINE_TEXT,
    MMPROJ_FILENAMES,
    MODEL_FILENAMES,
    CriticGroundingError,
    QwenVlCritic,
    _cos,
    _data_uri,
    _first_existing,
    check_grounding,
    clip_score_10,
    verify_finding_region,
)
from smartgallery_ai.embedders import BackendUnavailable, SemanticEmbedder
from smartgallery_ai.review import validate_review_payload

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(CQ.__file__)))


# --- fixtures / helpers -----------------------------------------------------


class VecEmbedder(SemanticEmbedder):
    """Deterministic SemanticEmbedder stand-in: a fixed vector per known
    text, one fixed vector for every image; records the size of each image
    it is asked to embed. Unknown text raises KeyError so tests notice
    unexpected embed_text calls."""

    model_id = "fake/vec"
    model_version = "v0"
    dim = 3

    def __init__(self, text_vecs, image_vec=(1.0, 0.0, 0.0)):
        self._text_vecs = dict(text_vecs)
        self._image_vec = tuple(image_vec)
        self.image_calls = []

    def embed_image(self, img):
        self.image_calls.append(img.size)
        return np.asarray(self._image_vec, dtype=np.float32)

    def embed_text(self, text):
        return np.asarray(self._text_vecs[text], dtype=np.float32)


def solid_color_image(size=(64, 64), color=(30, 30, 30)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def _fail_if_called(*args, **kwargs):
    raise AssertionError("verify_finding_region must not be consulted here")


def _bare_critic(monkeypatch, chats, embedder=None, verify=_fail_if_called):
    """A QwenVlCritic with no loaded weights: _chat serves the scripted
    responses in order (recording each call), check_grounding is stubbed to
    a fixed margin, verify_finding_region to `verify`."""
    critic = object.__new__(QwenVlCritic)
    critic._embedder = embedder if embedder is not None else object()
    critic._grounding_min_cos = CQ.DEFAULT_GROUNDING_MIN_COS
    calls = []
    seq = iter(chats)

    def fake_chat(self, uri, text, schema, max_tokens):
        calls.append({"uri": uri, "text": text, "schema": schema})
        return next(seq)

    monkeypatch.setattr(QwenVlCritic, "_chat", fake_chat)
    monkeypatch.setattr(CQ, "check_grounding", lambda *a, **k: 0.15)
    monkeypatch.setattr(CQ, "verify_finding_region", verify)
    return critic, calls


def _assess(defects, quality=7.0):
    return json.dumps({"quality_score": quality, "defects": defects})


def _defect(type_, region, what="melted hand", severity="medium", confidence=0.8):
    return {"type": type_, "severity": severity, "confidence": confidence,
            "region": region, "what": what}


# --- _first_existing preference order ----------------------------------------


def test_first_existing_prefers_q8_model_when_both_provisioned(tmp_path):
    """With both quantizations present, _first_existing returns the higher-fidelity Q8_0 path."""
    for name in MODEL_FILENAMES:
        (tmp_path / name).write_bytes(b"")
    assert _first_existing(str(tmp_path), MODEL_FILENAMES) == str(
        tmp_path / "Qwen2.5-VL-7B-Instruct-Q8_0.gguf")


def test_first_existing_falls_back_to_q4_when_only_it_exists(tmp_path):
    """With only the Q4_K_M file present, _first_existing returns that path."""
    (tmp_path / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf").write_bytes(b"")
    assert _first_existing(str(tmp_path), MODEL_FILENAMES) == str(
        tmp_path / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")


def test_first_existing_empty_dir_returns_none(tmp_path):
    """An unprovisioned directory resolves to None, never a nonexistent path."""
    assert _first_existing(str(tmp_path), MODEL_FILENAMES) is None


def test_first_existing_mmproj_prefers_f16(tmp_path):
    """With both mmproj files present, the f16 projection is preferred over Q8_0."""
    for name in MMPROJ_FILENAMES:
        (tmp_path / name).write_bytes(b"")
    assert _first_existing(str(tmp_path), MMPROJ_FILENAMES) == str(
        tmp_path / "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf")


# --- _data_uri ----------------------------------------------------------------


def test_data_uri_produces_decodable_png_with_rgb_conversion():
    """_data_uri yields a base64 PNG data URI that decodes back to the same pixels, RGBA converted to RGB."""
    img = Image.new("RGBA", (10, 7), (255, 0, 0, 128))
    uri = _data_uri(img)
    assert uri.startswith("data:image/png;base64,")
    decoded = base64.b64decode(uri.split(",", 1)[1])
    with Image.open(io.BytesIO(decoded)) as reopened:
        assert reopened.format == "PNG"
        assert reopened.size == (10, 7)
        assert reopened.mode == "RGB"
        assert reopened.getpixel((0, 0)) == (255, 0, 0)


# --- clip_score_10 ------------------------------------------------------------


def test_clip_score_10_mapping_zero_negative_cap_and_linear_region():
    """clip_score_10 maps cos 0 -> 0, negative cos -> 0, caps at 10 from cos 0.4, and is 25*cos below the cap."""
    assert clip_score_10(0.0) == 0.0
    assert clip_score_10(-0.5) == 0.0
    assert clip_score_10(0.4) == 10.0
    assert clip_score_10(1.0) == 10.0
    assert clip_score_10(0.2) == pytest.approx(5.0)


# --- check_grounding: fail-closed + inclusive thresholds ----------------------


def test_check_grounding_empty_description_raises():
    """An empty description is rejected before any embedding is computed."""
    emb = VecEmbedder({})
    with pytest.raises(CriticGroundingError, match="no description"):
        check_grounding(emb, "", solid_color_image())
    assert emb.image_calls == []


def test_check_grounding_nan_cosine_fails_closed():
    """A NaN description-image cosine can never pass the absolute floor: the gate raises instead of comparing to True."""
    emb = VecEmbedder({"a red cube": (float("nan"), 0.0, 0.0)})
    with pytest.raises(CriticGroundingError, match="does not match image"):
        check_grounding(emb, "a red cube", solid_color_image())


def test_check_grounding_nan_margin_fails_closed():
    """A NaN baseline margin fails the contrastive check even when the absolute cosine floor passes."""
    emb = VecEmbedder({
        "a red cube": (1.0, 0.0, 0.0),
        GROUNDING_BASELINE_TEXT: (float("nan"), 0.0, 0.0),
    })
    with pytest.raises(CriticGroundingError, match="not specific to this image"):
        check_grounding(emb, "a red cube", solid_color_image())


def test_check_grounding_passes_at_exact_thresholds_inclusive():
    """Both thresholds are inclusive (>=): cos == min_cos and margin == min_margin pass, returning the margin."""
    emb = VecEmbedder({
        "a red cube": (0.8, 0.6, 0.0),
        GROUNDING_BASELINE_TEXT: (0.5, 0.8, 0.33),
    })
    img = solid_color_image()
    iv = emb.embed_image(img)
    cos_desc = _cos(emb.embed_text("a red cube"), iv)
    margin = cos_desc - _cos(emb.embed_text(GROUNDING_BASELINE_TEXT), iv)
    result = check_grounding(emb, "a red cube", img,
                             min_cos=cos_desc, min_margin=margin)
    assert result == margin


def test_check_grounding_rejects_cos_one_ulp_below_floor():
    """A cosine one float ULP below min_cos is rejected with the does-not-match message."""
    emb = VecEmbedder({
        "a red cube": (0.8, 0.6, 0.0),
        GROUNDING_BASELINE_TEXT: (0.5, 0.8, 0.33),
    })
    img = solid_color_image()
    cos_desc = _cos(emb.embed_text("a red cube"), emb.embed_image(img))
    with pytest.raises(CriticGroundingError, match="does not match image"):
        check_grounding(emb, "a red cube", img,
                        min_cos=float(np.nextafter(cos_desc, np.inf)),
                        min_margin=-1.0)


def test_check_grounding_rejects_margin_one_ulp_below_floor():
    """A margin one float ULP below min_margin is rejected with the not-specific message."""
    emb = VecEmbedder({
        "a red cube": (0.8, 0.6, 0.0),
        GROUNDING_BASELINE_TEXT: (0.5, 0.8, 0.33),
    })
    img = solid_color_image()
    iv = emb.embed_image(img)
    margin = (_cos(emb.embed_text("a red cube"), iv)
              - _cos(emb.embed_text(GROUNDING_BASELINE_TEXT), iv))
    with pytest.raises(CriticGroundingError, match="not specific to this image"):
        check_grounding(emb, "a red cube", img, min_cos=0.0,
                        min_margin=float(np.nextafter(margin, np.inf)))


# --- verify_finding_region ----------------------------------------------------


def test_verify_finding_region_full_image_bbox_padding_clamped_to_bounds():
    """A (0,0,1,1) bbox's 10% padding is clamped: the embedded crop is exactly the image, never padded beyond it."""
    emb = VecEmbedder({"melted hand": (1.0, 0.0, 0.0),
                       GROUNDING_BASELINE_TEXT: (0.0, 1.0, 0.0)})
    img = solid_color_image(size=(300, 300))
    ok = verify_finding_region(emb, "melted hand", (0.0, 0.0, 1.0, 1.0), img)
    assert ok is True
    # unclamped padding would have produced a 360x360 crop
    assert emb.image_calls == [(300, 300)]


def test_verify_finding_region_corner_bbox_small_crop_resized_to_224():
    """An edge-corner bbox stays in bounds and its sub-224px crop is upscaled to exactly 224x224 for embedding."""
    emb = VecEmbedder({"melted hand": (1.0, 0.0, 0.0),
                       GROUNDING_BASELINE_TEXT: (0.0, 1.0, 0.0)})
    img = solid_color_image(size=(300, 300))
    ok = verify_finding_region(emb, "melted hand", (0.9, 0.9, 0.1, 0.1), img)
    assert ok is True
    assert emb.image_calls == [(224, 224)]


def test_verify_finding_region_degenerate_crop_returns_false_without_embedding():
    """A bbox whose crop is under 8px in either dimension returns False before any embedding is computed."""
    emb = VecEmbedder({})  # any embed_text call would KeyError
    img = solid_color_image(size=(100, 100))
    ok = verify_finding_region(emb, "melted hand", (0.5, 0.5, 0.001, 0.001), img)
    assert ok is False
    assert emb.image_calls == []


def test_verify_finding_region_below_baseline_returns_false():
    """A finding text scoring below the generic baseline on its own crop is rejected."""
    emb = VecEmbedder({"melted hand": (0.0, 1.0, 0.0),
                       GROUNDING_BASELINE_TEXT: (1.0, 0.0, 0.0)})
    img = solid_color_image(size=(300, 300))
    ok = verify_finding_region(emb, "melted hand", (0.1, 0.1, 0.6, 0.6), img)
    assert ok is False


def test_verify_finding_region_zero_margin_passes_inclusive_default():
    """The default min_margin of 0.0 is inclusive: a finding scoring exactly at the baseline passes."""
    emb = VecEmbedder({"melted hand": (0.6, 0.8, 0.0),
                       GROUNDING_BASELINE_TEXT: (0.6, 0.8, 0.0)})
    img = solid_color_image(size=(300, 300))
    assert verify_finding_region(
        emb, "melted hand", (0.1, 0.1, 0.6, 0.6), img) is True


# --- QwenVlCritic constructor availability contract ---------------------------


def test_qwen_critic_missing_weights_raises_naming_dir_and_candidates(tmp_path):
    """An unprovisioned models_dir raises BackendUnavailable naming the dir and the model filenames."""
    with pytest.raises(BackendUnavailable) as exc:
        QwenVlCritic(str(tmp_path), semantic_embedder=object())
    msg = str(exc.value)
    assert str(tmp_path) in msg
    assert "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" in msg


def test_qwen_critic_model_without_mmproj_is_unavailable(tmp_path):
    """A provisioned model file without its mmproj projection still raises BackendUnavailable."""
    (tmp_path / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf").write_bytes(b"")
    with pytest.raises(BackendUnavailable, match="weights not found"):
        QwenVlCritic(str(tmp_path), semantic_embedder=object())


def test_qwen_critic_weights_check_precedes_llama_import(tmp_path):
    """Resolution on an unprovisioned system is side-effect-free: the failed weights check never imports llama_cpp."""
    script = (
        "import sys\n"
        "from smartgallery_ai.critic_qwen import QwenVlCritic\n"
        "from smartgallery_ai.embedders import BackendUnavailable\n"
        "try:\n"
        "    QwenVlCritic(sys.argv[1], semantic_embedder=object())\n"
        "    print('outcome=CONSTRUCTED')\n"
        "except BackendUnavailable:\n"
        "    print('outcome=UNAVAILABLE')\n"
        "except Exception as exc:\n"
        "    print('outcome=WRONG-' + type(exc).__name__)\n"
        "print('llama_loaded=%s' % ('llama_cpp' in sys.modules))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "outcome=UNAVAILABLE" in proc.stdout
    assert "llama_loaded=False" in proc.stdout


# --- model_version derivation from the provisioned filename -------------------
#
# The constructor derives model_version BEFORE importing llama_cpp; blocking
# that import (sys.modules[...] = None) stops it right after derivation, so
# the version logic is observable on the partially-initialized instance
# without ever touching a real GGUF loader.


def test_qwen_critic_model_version_prefers_q8_filename(tmp_path, monkeypatch):
    """With both quantizations provisioned, model_version derives from the preferred Q8_0 filename."""
    for name in (MODEL_FILENAMES[0], MODEL_FILENAMES[1], MMPROJ_FILENAMES[0]):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    critic = object.__new__(QwenVlCritic)
    with pytest.raises(BackendUnavailable, match="qwen-vl critic unavailable"):
        critic.__init__(str(tmp_path), semantic_embedder=object())
    assert critic.model_version == "qwen2.5-vl-7b-q8_0+decomposed-v1"


def test_qwen_critic_model_version_q4_matches_class_default(tmp_path, monkeypatch):
    """With only Q4_K_M provisioned, the derived version equals the class default, and the class attribute is untouched."""
    for name in (MODEL_FILENAMES[1], MMPROJ_FILENAMES[1]):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    critic = object.__new__(QwenVlCritic)
    with pytest.raises(BackendUnavailable, match="qwen-vl critic unavailable"):
        critic.__init__(str(tmp_path), semantic_embedder=object())
    assert critic.model_version == "qwen2.5-vl-7b-q4_k_m+decomposed-v1"
    assert QwenVlCritic.model_version == "qwen2.5-vl-7b-q4_k_m+decomposed-v1"


# --- review(): localization policy via monkeypatched _chat --------------------


def test_review_whole_image_region_spatial_type_never_localizes(monkeypatch):
    """A spatial-type defect reported as whole-image gets no LOCALIZE call and stays global with no region note."""
    critic, calls = _bare_critic(monkeypatch, [
        "A dog on grass.",
        _assess([_defect("artifact", "whole-image", what="grain everywhere")],
                quality=8.0),
    ])
    payload = critic.review(solid_color_image(), None, "rubric-1")
    assert len(calls) == 2  # DESCRIBE + ASSESS only, no bbox call
    (finding,) = payload["findings"]
    assert finding["localizable"] is False
    assert "bbox" not in finding
    assert finding["description"] == "grain everywhere"
    assert payload["quality_score"] == 8.0
    assert payload["prompt_alignment_score"] is None


def test_review_non_spatial_type_never_localizes_even_with_region(monkeypatch):
    """A lighting defect never gets a bbox call even with a concrete region; the region is demoted to a description note."""
    critic, calls = _bare_critic(monkeypatch, [
        "A dim portrait.",
        _assess([_defect("lighting", "top-left", what="harsh shadow")]),
    ])
    payload = critic.review(solid_color_image(), None, "rubric-1")
    assert len(calls) == 2
    (finding,) = payload["findings"]
    assert finding["localizable"] is False
    assert "bbox" not in finding
    assert finding["description"] == "harsh shadow (reported region: top-left)"


@pytest.mark.parametrize("bbox_json", [
    '{"x": 0.9, "y": 0.1, "w": 0.3, "h": 0.2}',    # x + w > 1
    '{"x": 0.1, "y": 0.9, "w": 0.2, "h": 0.3}',    # y + h > 1
    '{"x": 0.1, "y": 0.1, "w": 0.005, "h": 0.5}',  # w below 0.01 minimum
    '{"x": "left", "y": 0.1, "w": 0.5, "h": 0.5}', # non-numeric coordinate
])
def test_review_malformed_bbox_demotes_finding_to_global(monkeypatch, bbox_json):
    """A geometrically invalid model-emitted bbox is discarded: the finding stays global with a region note, never repaired."""
    critic, calls = _bare_critic(monkeypatch, [
        "A hand holding a cup.",
        _assess([_defect("anatomy", "bottom-right")]),
        bbox_json,
    ])
    payload = critic.review(solid_color_image(), None, "rubric-1")
    assert len(calls) == 3  # the LOCALIZE call happened and was rejected
    (finding,) = payload["findings"]
    assert finding["localizable"] is False
    assert "bbox" not in finding
    assert finding["description"] == "melted hand (reported region: bottom-right)"


def test_review_valid_bbox_verified_yields_localizable_finding(monkeypatch):
    """A valid model bbox that passes crop verification yields a localizable finding carrying that bbox, schema-valid."""
    critic, calls = _bare_critic(monkeypatch, [
        "A hand holding a cup.",
        _assess([_defect("anatomy", "bottom-right")], quality=6.5),
        '{"x": 0.6, "y": 0.55, "w": 0.2, "h": 0.3}',
    ], verify=lambda *a, **k: True)
    payload = critic.review(solid_color_image(), None, "rubric-1")
    assert len(calls) == 3
    (finding,) = payload["findings"]
    assert finding["localizable"] is True
    assert finding["bbox"] == [0.6, 0.55, 0.2, 0.3]
    assert finding["description"] == "melted hand"  # no region note
    # the assembled payload must satisfy the strict review schema
    result = validate_review_payload(payload)
    assert result.findings[0].bbox == (0.6, 0.55, 0.2, 0.3)


def test_review_prompt_alignment_is_clip_score_of_prompt_and_image(monkeypatch):
    """With a prompt, alignment is clip_score_10 of the prompt-image cosine from the embedder, computed outside the VLM."""
    emb = VecEmbedder({"a red cube": (1.0, 0.0, 0.0)}, image_vec=(1.0, 0.0, 0.0))
    critic, calls = _bare_critic(monkeypatch, [
        "A red cube.",
        _assess([], quality=9.0),
    ], embedder=emb)
    payload = critic.review(solid_color_image(), "a red cube", "rubric-1")
    assert payload["prompt_alignment_score"] == 10.0  # cos 1.0 -> capped 10
    assert payload["findings"] == []


def test_review_downscales_oversized_image_before_chat(monkeypatch):
    """An image over 768px is thumbnailed before encoding: the VLM receives at most 768px, aspect preserved."""
    critic, calls = _bare_critic(monkeypatch, [
        "A wide banner.",
        _assess([], quality=9.0),
    ])
    critic.review(solid_color_image(size=(1024, 512)), None, "rubric-1")
    uri = calls[0]["uri"]
    decoded = base64.b64decode(uri.split(",", 1)[1])
    with Image.open(io.BytesIO(decoded)) as sent:
        assert sent.size == (768, 384)


def test_review_null_describe_content_fails_grounding_closed(monkeypatch):
    """A None-content DESCRIBE completion coerces to an empty description, and the grounding gate aborts the review."""

    class _NullContentLlm:
        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": None}}]}

    critic = object.__new__(QwenVlCritic)
    critic._embedder = VecEmbedder({})
    critic._grounding_min_cos = CQ.DEFAULT_GROUNDING_MIN_COS
    critic._llm = _NullContentLlm()
    with pytest.raises(CriticGroundingError, match="no description"):
        critic.review(solid_color_image(), None, "rubric-1")


# --- GPU offload knobs passed to llama.cpp ------------------------------------


def _install_fake_llama(monkeypatch, recorded: dict):
    """Working llama_cpp stand-in whose Llama records its kwargs."""

    class _FakeLlama:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    fake_fmt = types.ModuleType("llama_cpp.llama_chat_format")
    fake_fmt.Qwen25VLChatHandler = lambda clip_model_path, verbose: object()
    fake = types.ModuleType("llama_cpp")
    fake.Llama = _FakeLlama
    fake.llama_chat_format = fake_fmt
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", fake_fmt)


def _touch_qwen_weights(tmp_path):
    (tmp_path / MODEL_FILENAMES[1]).write_bytes(b"")
    (tmp_path / CQ.MMPROJ_FILENAMES[1]).write_bytes(b"")


def test_llama_defaults_to_full_gpu_offload(tmp_path, monkeypatch):
    """Without overrides the critic asks llama.cpp for all layers on GPU
    (a CPU-only build ignores it); no pin, no custom split."""
    _touch_qwen_weights(tmp_path)
    for var in ("AI_DAM_DEVICE", "AI_DAM_GPU_LAYERS", "AI_DAM_TENSOR_SPLIT"):
        monkeypatch.delenv(var, raising=False)
    recorded: dict = {}
    _install_fake_llama(monkeypatch, recorded)

    QwenVlCritic(str(tmp_path), semantic_embedder=object())
    assert recorded["n_gpu_layers"] == -1
    assert "main_gpu" not in recorded
    assert "tensor_split" not in recorded


def test_llama_cpu_optout_forces_zero_layers_and_ignores_split(tmp_path, monkeypatch):
    """AI_DAM_DEVICE=cpu keeps the whole model on CPU even when a tensor
    split is configured."""
    _touch_qwen_weights(tmp_path)
    monkeypatch.setenv("AI_DAM_DEVICE", "cpu")
    monkeypatch.setenv("AI_DAM_TENSOR_SPLIT", "0.5,0.5")
    recorded: dict = {}
    _install_fake_llama(monkeypatch, recorded)

    QwenVlCritic(str(tmp_path), semantic_embedder=object())
    assert recorded["n_gpu_layers"] == 0
    assert "tensor_split" not in recorded


def test_llama_cuda_pin_and_tensor_split_and_partial_layers(tmp_path, monkeypatch):
    """AI_DAM_DEVICE=cuda:1 pins llama's primary GPU, AI_DAM_TENSOR_SPLIT
    becomes the per-GPU proportions, AI_DAM_GPU_LAYERS tunes offload."""
    _touch_qwen_weights(tmp_path)
    monkeypatch.setenv("AI_DAM_DEVICE", "cuda:1")
    monkeypatch.setenv("AI_DAM_TENSOR_SPLIT", "0.6,0.4")
    monkeypatch.setenv("AI_DAM_GPU_LAYERS", "20")
    recorded: dict = {}
    _install_fake_llama(monkeypatch, recorded)

    QwenVlCritic(str(tmp_path), semantic_embedder=object())
    assert recorded["main_gpu"] == 1
    assert recorded["tensor_split"] == [0.6, 0.4]
    assert recorded["n_gpu_layers"] == 20


def test_llama_native_logging_is_silenced_unless_opted_in(tmp_path, monkeypatch):
    """The ctor installs a no-op llama.cpp log callback (native logs leak
    full prompts to the console) and keeps a reference on self;
    AI_DAM_LLAMA_VERBOSE opts back into the native logs."""
    _touch_qwen_weights(tmp_path)
    monkeypatch.delenv("AI_DAM_LLAMA_VERBOSE", raising=False)
    monkeypatch.delenv("AI_DAM_DEVICE", raising=False)
    recorded: dict = {}
    _install_fake_llama(monkeypatch, recorded)

    fake = sys.modules["llama_cpp"]
    log_sets = []
    fake.llama_log_callback = lambda fn: ("cb", fn)
    fake.llama_log_set = lambda cb, user: log_sets.append(cb)

    critic = QwenVlCritic(str(tmp_path), semantic_embedder=object())
    assert len(log_sets) == 1
    assert critic._llama_log_cb is log_sets[0]

    monkeypatch.setenv("AI_DAM_LLAMA_VERBOSE", "1")
    log_sets.clear()
    QwenVlCritic(str(tmp_path), semantic_embedder=object())
    assert log_sets == []
