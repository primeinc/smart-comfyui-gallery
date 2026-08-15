"""Edge-behavior tests for smartgallery_ai.review beyond tests/test_review.py:
SmolVLM critic availability contracts (the weights check precedes any torch
import), _extract_json_object error contracts, points-only findings through
store_review and generate_finding_mask, generate_finding_mask identity and
containment error contracts, segmenter backend resolution, superseded-mask
tolerance, StubSegmenter fallbacks, and the 'auto' critic calibration gate's
fail-closed handling of malformed evidence."""

import copy
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from smartgallery_ai import AIConfig
from smartgallery_ai import review as REV
from smartgallery_ai.embedders import BackendUnavailable
from smartgallery_ai.review import (
    Finding,
    MaskNotAllowedError,
    ReviewResult,
    SmolVlmCritic,
    StubCritic,
    StubSegmenter,
    _extract_json_object,
    generate_finding_mask,
    get_critic_backend,
    get_segmenter_backend,
    store_review,
    validate_review_payload,
)
from smartgallery_ai.schema import init_schema

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(REV.__file__)))


# --- fixtures / helpers (same shape as tests/test_review.py) -----------------


def make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE files (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            mtime REAL NOT NULL,
            name TEXT NOT NULL,
            type TEXT
        )
        """
    )
    init_schema(conn)
    return conn


def add_file(conn, file_id, mtime=1000.0):
    conn.execute(
        "INSERT INTO files (id, path, mtime, name, type) VALUES (?, ?, ?, ?, ?)",
        (file_id, f"/gallery/{file_id}.png", mtime, file_id, "image"),
    )
    conn.commit()


def solid_color_image(size=(64, 64), color=(30, 30, 30)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def _store_one_finding(conn, file_id, localizable, bbox=None, points=None):
    finding = Finding(
        type="artifact",
        severity="high",
        confidence=0.9,
        localizable=localizable,
        description="test finding",
        bbox=bbox,
        points=points,
    )
    result = ReviewResult(
        quality_score=5.0, prompt_alignment_score=None, summary="s", findings=[finding]
    )
    review_id = store_review(
        conn, file_id, result, "critic-x", "v1", "rubric-1", None, 1000.0, 2000.0
    )
    finding_id = conn.execute(
        "SELECT finding_id FROM ai_review_findings WHERE review_id = ?", (review_id,)
    ).fetchone()[0]
    return finding_id


# --- SmolVlmCritic availability contract -------------------------------------


def test_smolvlm_missing_weights_raises_naming_dir_and_candidates(tmp_path):
    """An unprovisioned models_dir raises BackendUnavailable naming the dir and both checkpoint dirnames."""
    with pytest.raises(BackendUnavailable) as exc:
        SmolVlmCritic(str(tmp_path))
    msg = str(exc.value)
    assert str(tmp_path) in msg
    assert "smolvlm2-2.2b" in msg
    assert "smolvlm2-500m" in msg


def test_smolvlm_weights_check_does_not_import_torch(tmp_path):
    """Resolution on an unprovisioned system is side-effect-free: the failed weights check must never pull torch in."""
    script = (
        "import sys\n"
        "from smartgallery_ai.review import SmolVlmCritic\n"
        "from smartgallery_ai.embedders import BackendUnavailable\n"
        "try:\n"
        "    SmolVlmCritic(sys.argv[1])\n"
        "    print('outcome=CONSTRUCTED')\n"
        "except BackendUnavailable:\n"
        "    print('outcome=UNAVAILABLE')\n"
        "except Exception as exc:\n"
        "    print('outcome=WRONG-' + type(exc).__name__)\n"
        "print('torch_loaded=%s' % ('torch' in sys.modules))\n"
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
    assert "torch_loaded=False" in proc.stdout


def test_smolvlm_unloadable_checkpoint_dir_raises_backend_unavailable(tmp_path):
    """A provisioned-looking but empty checkpoint dir yields BackendUnavailable,
    never a raw loader exception. Run in a subprocess: this path imports
    torch/transformers, which must not be pulled into the test process."""
    script = (
        "import sys\n"
        "from smartgallery_ai.review import SmolVlmCritic\n"
        "from smartgallery_ai.embedders import BackendUnavailable\n"
        "try:\n"
        "    SmolVlmCritic(sys.argv[1])\n"
        "    print('outcome=CONSTRUCTED')\n"
        "except BackendUnavailable as exc:\n"
        "    print('outcome=UNAVAILABLE' if 'smolvlm' in str(exc) else 'outcome=WRONG-MESSAGE')\n"
        "except Exception as exc:\n"
        "    print('outcome=RAW-' + type(exc).__name__)\n"
    )
    (tmp_path / "smolvlm2-2.2b").mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "outcome=UNAVAILABLE" in proc.stdout


def test_get_critic_backend_smolvlm_propagates_backend_unavailable(tmp_path):
    """Explicit critic_backend='smolvlm' surfaces BackendUnavailable instead of degrading to None."""
    cfg = AIConfig(critic_backend="smolvlm", models_dir=str(tmp_path))
    with pytest.raises(BackendUnavailable, match="smolvlm weights not found"):
        get_critic_backend(cfg)


# --- _extract_json_object -----------------------------------------------------


def test_extract_json_object_unterminated_raises_value_error():
    """An opened-but-never-closed object raises ValueError naming the unterminated JSON object."""
    with pytest.raises(ValueError, match="unterminated JSON object"):
        _extract_json_object('prefix {"quality_score": 5, "findings": [')


def test_extract_json_object_without_any_object_raises_value_error():
    """Output with no '{' at all raises ValueError saying there is no JSON object."""
    with pytest.raises(ValueError, match="no JSON object"):
        _extract_json_object("I could not produce a review, sorry.")


def test_extract_json_object_tolerates_prose_prefix_and_suffix():
    """The first balanced object is parsed even when wrapped in chatty prose, including nested braces."""
    text = 'Sure! Here it is: {"quality_score": 6, "meta": {"nested": true}} Hope that helps.'
    assert _extract_json_object(text) == {"quality_score": 6, "meta": {"nested": True}}


def test_extract_json_object_braces_inside_string_values_parse():
    """Braces inside JSON string values do not affect nesting: the full
    object parses and the brace-bearing string survives intact."""
    text = 'noise {"summary": "brace } inside { here", "quality_score": 5} tail'
    obj = _extract_json_object(text)
    assert obj == {"summary": "brace } inside { here", "quality_score": 5}


def test_extract_json_object_escaped_quote_inside_string_parses():
    """An escaped quote inside a string value does not terminate the
    string, so a following brace is still treated as content."""
    text = r'{"summary": "she said \"}\" loudly", "quality_score": 4}'
    obj = _extract_json_object(text)
    assert obj == {"summary": 'she said "}" loudly', "quality_score": 4}


# --- StubCritic heuristic branches -------------------------------------------


def test_stub_critic_clean_image_yields_no_findings_and_full_score():
    """A bright, red-free image triggers neither heuristic: zero findings and quality 10.0."""
    critic = StubCritic()
    raw = critic.review(
        solid_color_image(color=(200, 200, 200)), prompt_text=None, rubric_version="r-v1"
    )
    result = validate_review_payload(raw)
    assert result.findings == []
    assert result.quality_score == 10.0
    assert result.prompt_alignment_score is None
    assert "0 finding(s)" in result.summary


def test_stub_critic_dark_image_with_red_square_yields_both_findings():
    """Both heuristics can fire on one image, and each finding costs 2 points: quality 6.0."""
    arr = np.full((64, 64, 3), (5, 5, 5), dtype=np.uint8)
    arr[24:40, 24:40] = (255, 0, 0)  # 16x16 = 1/16 of the image
    raw = StubCritic().review(
        Image.fromarray(arr, mode="RGB"), prompt_text="a dark scene", rubric_version="r-v1"
    )
    result = validate_review_payload(raw)
    assert sorted(f.type for f in result.findings) == ["artifact", "lighting"]
    assert result.quality_score == 6.0
    assert result.prompt_alignment_score == 5.0


# --- store_review: points-only findings --------------------------------------


def test_store_review_points_only_finding_writes_points_json_and_null_bbox():
    """A localizable points-only finding persists its points as JSON with all four bbox columns NULL."""
    conn = make_conn()
    add_file(conn, "f1")
    finding_id = _store_one_finding(
        conn, "f1", localizable=True, points=[(0.2, 0.3), (0.4, 0.5)]
    )
    row = conn.execute(
        "SELECT localizable, bbox_x, bbox_y, bbox_w, bbox_h, points "
        "FROM ai_review_findings WHERE finding_id = ?",
        (finding_id,),
    ).fetchone()
    assert row[:5] == (1, None, None, None, None)
    assert json.loads(row[5]) == [[0.2, 0.3], [0.4, 0.5]]


# --- generate_finding_mask error contracts + points path ---------------------


def test_generate_finding_mask_points_only_masks_points_bounding_box(tmp_path):
    """A points-only finding produces a 0/255 mask covering exactly the points' bounding box."""
    conn = make_conn()
    add_file(conn, "f1")
    finding_id = _store_one_finding(
        conn, "f1", localizable=True, points=[(0.25, 0.25), (0.75, 0.75)]
    )
    img = solid_color_image(size=(40, 40))
    mask_path = generate_finding_mask(
        conn, str(tmp_path / "cache"), img, "f1", finding_id, StubSegmenter()
    )
    with Image.open(mask_path) as mask_img:
        assert mask_img.mode == "L"
        arr = np.asarray(mask_img)
    assert arr[20, 20] == 255  # center of the points' bbox
    assert arr[2, 2] == 0  # outside it
    assert int((arr == 255).sum()) == 20 * 20  # exactly the 0.5x0.5 rectangle


def test_generate_finding_mask_unknown_finding_id_raises_value_error(tmp_path):
    """A finding_id that does not exist raises ValueError naming the id and file."""
    conn = make_conn()
    add_file(conn, "f1")
    with pytest.raises(ValueError, match=r"no finding 999 for file 'f1'"):
        generate_finding_mask(
            conn, str(tmp_path / "cache"), solid_color_image(), "f1", 999, StubSegmenter()
        )


def test_generate_finding_mask_finding_of_other_file_raises_value_error(tmp_path):
    """A real finding requested under a different file_id is treated as nonexistent, and no mask is recorded."""
    conn = make_conn()
    add_file(conn, "f1")
    add_file(conn, "f2")
    finding_id = _store_one_finding(conn, "f1", localizable=True, bbox=(0.1, 0.1, 0.5, 0.5))
    with pytest.raises(ValueError, match="no finding"):
        generate_finding_mask(
            conn, str(tmp_path / "cache"), solid_color_image(), "f2", finding_id, StubSegmenter()
        )
    row = conn.execute(
        "SELECT mask_path FROM ai_review_findings WHERE finding_id = ?", (finding_id,)
    ).fetchone()
    assert row == (None,)


def test_generate_finding_mask_geometryless_localizable_raises_mask_not_allowed(tmp_path):
    """A localizable row with neither bbox nor points (writable via store_review) raises MaskNotAllowedError."""
    conn = make_conn()
    add_file(conn, "f1")
    # store_review does not re-validate; the DB CHECK permits localizable=1
    # with all-NULL geometry, so such a row can legitimately exist.
    finding_id = _store_one_finding(conn, "f1", localizable=True, bbox=None, points=None)
    with pytest.raises(MaskNotAllowedError, match="no grounding geometry"):
        generate_finding_mask(
            conn, str(tmp_path / "cache"), solid_color_image(), "f1", finding_id, StubSegmenter()
        )


def test_generate_finding_mask_symlinked_file_dir_cannot_escape_cache_dir(tmp_path):
    """A pre-planted symlink under masks/ that resolves outside cache_dir is rejected and nothing is written there."""
    conn = make_conn()
    add_file(conn, "f1")
    finding_id = _store_one_finding(conn, "f1", localizable=True, bbox=(0.1, 0.1, 0.5, 0.5))

    cache_dir = tmp_path / "cache"
    masks_dir = cache_dir / "masks"
    masks_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(str(outside), str(masks_dir / "f1"))

    with pytest.raises(ValueError, match="escapes cache_dir"):
        generate_finding_mask(
            conn, str(cache_dir), solid_color_image(), "f1", finding_id, StubSegmenter()
        )
    assert list(outside.iterdir()) == []
    row = conn.execute(
        "SELECT mask_path FROM ai_review_findings WHERE finding_id = ?", (finding_id,)
    ).fetchone()
    assert row == (None,)


# --- store_review: superseded mask already gone ------------------------------


def test_store_review_replacement_tolerates_already_missing_mask_file(tmp_path):
    """Replacing a review whose recorded mask file no longer exists on disk still commits the replacement."""
    conn = make_conn()
    add_file(conn, "f1")
    finding_id = _store_one_finding(conn, "f1", localizable=True, bbox=(0.25, 0.25, 0.5, 0.5))
    ghost_mask = str(tmp_path / "already-deleted.png")  # never created
    conn.execute(
        "UPDATE ai_review_findings SET mask_path = ?, mask_model_id = 'seg', "
        "mask_model_version = 'v1' WHERE finding_id = ?",
        (ghost_mask, finding_id),
    )
    conn.commit()

    replacement = ReviewResult(
        quality_score=6.0, prompt_alignment_score=None, summary="replacement", findings=[]
    )
    new_id = store_review(
        conn, "f1", replacement, "critic-x", "v1", "rubric-1", None, 1000.0, 3000.0
    )
    assert conn.execute("SELECT review_id, summary FROM ai_reviews").fetchall() == [
        (new_id, "replacement")
    ]


# --- StubSegmenter fallbacks --------------------------------------------------


def test_stub_segmenter_without_geometry_returns_all_false_mask():
    """With neither bbox nor points, segment() returns an all-False HxW mask matching the image."""
    img = Image.new("RGB", (20, 10))  # w=20, h=10
    mask = StubSegmenter().segment(img)
    assert mask.shape == (10, 20)
    assert mask.dtype == np.bool_
    assert not mask.any()


def test_stub_segmenter_points_fallback_rasterizes_points_bounding_box():
    """Without a bbox, segment() rasterizes exactly the points' bounding box."""
    img = Image.new("RGB", (20, 10))
    mask = StubSegmenter().segment(img, points=[(0.25, 0.2), (0.75, 0.8)])
    expected = np.zeros((10, 20), dtype=bool)
    expected[2:8, 5:15] = True
    assert np.array_equal(mask, expected)


# --- get_segmenter_backend resolution ----------------------------------------


def test_get_segmenter_backend_mobilesam_raises_and_auto_degrades(tmp_path):
    """Explicit 'mobilesam' without weights raises BackendUnavailable; 'auto' degrades to None."""
    with pytest.raises(BackendUnavailable, match="mobile_sam weights not found"):
        get_segmenter_backend(
            AIConfig(segmenter_backend="mobilesam", models_dir=str(tmp_path))
        )
    assert (
        get_segmenter_backend(AIConfig(segmenter_backend="auto", models_dir=str(tmp_path)))
        is None
    )


def test_get_segmenter_backend_stub_none_and_unknown():
    """'stub' resolves to StubSegmenter, 'none' to None, and an unknown name raises ValueError naming it."""
    assert isinstance(
        get_segmenter_backend(AIConfig(segmenter_backend="stub")), StubSegmenter
    )
    assert get_segmenter_backend(AIConfig(segmenter_backend="none")) is None
    with pytest.raises(ValueError, match="bogus"):
        get_segmenter_backend(AIConfig(segmenter_backend="bogus"))


# --- 'auto' critic calibration gate: malformed evidence fails closed ----------


def test_auto_critic_gate_non_numeric_sweep_rates_fail_closed(tmp_path):
    """A sweep row whose rates are not numbers can never authorize 'auto', even with valid identity."""
    from smartgallery_ai.critic_qwen import DEFAULT_GROUNDING_MIN_MARGIN
    from smartgallery_ai.review import _auto_critic_measurement_passed

    with open(REV._CALIBRATION_REPORT_PATH, "r", encoding="utf-8") as fh:
        real = json.load(fh)
    # Arrange sanity: the committed evidence itself qualifies.
    assert _auto_critic_measurement_passed() is True

    report = copy.deepcopy(real)
    row = next(
        s
        for s in report["sweep"]
        if abs(float(s["margin_threshold"]) - DEFAULT_GROUNDING_MIN_MARGIN) < 1e-9
    )
    row["false_accept_rate"] = "not-a-number"
    path = str(tmp_path / "bad_rates.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh)

    assert _auto_critic_measurement_passed(path) is False


def test_get_critic_backend_auto_gate_is_the_deciding_factor(tmp_path, monkeypatch):
    """The calibration gate causally decides 'auto' resolution: with a
    constructible critic stubbed in, a rejecting gate yields None and an
    accepting gate yields the critic — so None is attributable to the gate,
    not to missing weights."""
    import smartgallery_ai.critic_qwen as CQ
    import smartgallery_ai.embedders as EMB

    sentinel = object()
    monkeypatch.setattr(EMB, "get_semantic_backend", lambda _cfg: object())
    monkeypatch.setattr(CQ, "QwenVlCritic", lambda *_a, **_k: sentinel)
    cfg = AIConfig(critic_backend="auto", models_dir=str(tmp_path))

    monkeypatch.setattr(REV, "_auto_critic_measurement_passed", lambda *_a: False)
    assert get_critic_backend(cfg) is None

    monkeypatch.setattr(REV, "_auto_critic_measurement_passed", lambda *_a: True)
    assert get_critic_backend(cfg) is sentinel
