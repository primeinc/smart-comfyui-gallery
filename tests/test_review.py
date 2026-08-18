"""Tests for smartgallery_ai.review: strict payload validation (acceptance +
rejection cases), store_review upsert + the live ai_review_findings CHECK
constraint, StubReviewer heuristics, and generate_finding_mask (including the
path-traversal guard and source-file untouched guarantee)."""

import copy
import json as _json
import os
import sqlite3

import numpy as np
import pytest
import pytest as _pytest
from PIL import Image

from smartgallery_ai import AIConfig
from smartgallery_ai import review as REV
from smartgallery_ai import reviewer as CQ
from smartgallery_ai.embedders import BackendUnavailable
from smartgallery_ai.review import (
    Finding,
    MaskNotAllowedError,
    ReviewResult,
    ReviewSchemaError,
    StubReviewer,
    StubSegmenter,
    _auto_critic_measurement_passed,
    generate_finding_mask,
    get_reviewer,
    store_review,
    validate_review_payload,
)
from smartgallery_ai.reviewer import DEFAULT_GROUNDING_MIN_MARGIN, Reviewer
from smartgallery_ai.schema import init_schema

# --- fixtures / helpers -----------------------------------------------------


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


def valid_payload(**overrides) -> dict:
    payload = {
        "quality_score": 7.5,
        "prompt_alignment_score": 0.6,
        "summary": "Mostly good, minor artifact.",
        "findings": [
            {
                "type": "lighting",
                "severity": "low",
                "confidence": 0.6,
                "localizable": False,
                "description": "Slightly flat lighting.",
            },
            {
                "type": "artifact",
                "severity": "high",
                "confidence": 0.95,
                "localizable": True,
                "description": "Extra finger.",
                "bbox": [0.1, 0.2, 0.1, 0.1],
            },
        ],
    }
    payload.update(overrides)
    return payload


def solid_color_image(size=(64, 64), color=(30, 30, 30)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def image_with_red_square(size=(64, 64), square=(24, 24, 16, 16), bg=(20, 150, 20)) -> Image.Image:
    arr = np.full((size[1], size[0], 3), bg, dtype=np.uint8)
    x, y, w, h = square
    arr[y : y + h, x : x + w] = (255, 0, 0)
    return Image.fromarray(arr, mode="RGB")


# --- validate_review_payload: acceptance ------------------------------------


def test_validate_review_payload_accepts_well_formed_payload():
    result = validate_review_payload(valid_payload())
    assert isinstance(result, ReviewResult)
    assert result.quality_score == 7.5
    assert result.prompt_alignment_score == 0.6
    assert len(result.findings) == 2
    assert result.findings[0].localizable is False
    assert result.findings[0].bbox is None
    assert result.findings[1].localizable is True
    assert result.findings[1].bbox == (0.1, 0.2, 0.1, 0.1)


def test_validate_review_payload_accepts_null_prompt_alignment_and_no_findings():
    payload = valid_payload(prompt_alignment_score=None, findings=[])
    result = validate_review_payload(payload)
    assert result.prompt_alignment_score is None
    assert result.findings == []


def test_validate_review_payload_accepts_points_only_finding():
    payload = valid_payload(
        findings=[
            {
                "type": "anatomy",
                "severity": "medium",
                "confidence": 0.5,
                "localizable": True,
                "description": "Odd hand shape.",
                "points": [[0.2, 0.3], [0.25, 0.35]],
            }
        ]
    )
    result = validate_review_payload(payload)
    assert result.findings[0].points == [(0.2, 0.3), (0.25, 0.35)]
    assert result.findings[0].bbox is None


def test_validate_review_payload_missing_prompt_alignment_key_defaults_none():
    payload = valid_payload()
    del payload["prompt_alignment_score"]
    result = validate_review_payload(payload)
    assert result.prompt_alignment_score is None


# --- validate_review_payload: rejections ------------------------------------


def test_validate_alignment_elements_and_derived_score():
    """Elements round-trip in prompt order, and the score is accepted when
    it equals satisfied/total."""
    payload = valid_payload(
        prompt_alignment_score=0.5,
        alignment=[
            {"ordinal": 0, "text": "a red cube", "satisfied": True, "confidence": 0.9, "bbox": [0.1, 0.1, 0.2, 0.2]},
            {"ordinal": 1, "text": "a blue sphere", "satisfied": False, "confidence": 0.8},
        ],
    )
    result = validate_review_payload(payload)
    assert [(e.ordinal, e.text, e.satisfied) for e in result.alignment] == [
        (0, "a red cube", True),
        (1, "a blue sphere", False),
    ]
    assert result.alignment[0].bbox == (0.1, 0.1, 0.2, 0.2)
    assert result.alignment[1].bbox is None


def test_reject_alignment_score_disagreeing_with_its_own_elements():
    """A score that contradicts the element list is incoherent -- neither half
    is silently trusted."""
    payload = valid_payload(
        prompt_alignment_score=0.9,
        alignment=[
            {"ordinal": 0, "text": "a", "satisfied": True, "confidence": 0.9},
            {"ordinal": 1, "text": "b", "satisfied": False, "confidence": 0.9},
        ],
    )
    with pytest.raises(ReviewSchemaError) as exc:
        validate_review_payload(payload)
    assert exc.value.path == "prompt_alignment_score"


def test_reject_absent_alignment_element_carrying_a_bbox():
    """An element that is not in the image cannot have been located."""
    payload = valid_payload(
        prompt_alignment_score=0.0,
        alignment=[
            {"ordinal": 0, "text": "a blue sphere", "satisfied": False, "confidence": 0.8, "bbox": [0.1, 0.1, 0.2, 0.2]}
        ],
    )
    with pytest.raises(ReviewSchemaError) as exc:
        validate_review_payload(payload)
    assert exc.value.path == "alignment[0]"


def test_reject_duplicate_alignment_ordinals():
    payload = valid_payload(
        prompt_alignment_score=1.0,
        alignment=[
            {"ordinal": 0, "text": "a", "satisfied": True, "confidence": 0.9},
            {"ordinal": 0, "text": "b", "satisfied": True, "confidence": 0.9},
        ],
    )
    with pytest.raises(ReviewSchemaError) as exc:
        validate_review_payload(payload)
    assert exc.value.path == "alignment"


def test_reject_alignment_score_above_one():
    """The score is a fraction now, not a 0-10 rating."""
    with pytest.raises(ReviewSchemaError) as exc:
        validate_review_payload(valid_payload(prompt_alignment_score=6.0))
    assert exc.value.path == "prompt_alignment_score"


def test_store_review_persists_alignment_elements_and_replaces_them_on_upsert():
    conn = make_conn()
    add_file(conn, "f1")
    result = validate_review_payload(
        valid_payload(
            prompt_alignment_score=0.5,
            alignment=[
                {
                    "ordinal": 0,
                    "text": "a red cube",
                    "satisfied": True,
                    "confidence": 0.9,
                    "bbox": [0.1, 0.1, 0.2, 0.2],
                },
                {"ordinal": 1, "text": "a blue sphere", "satisfied": False, "confidence": 0.8},
            ],
        )
    )
    review_id = store_review(conn, "f1", result, "critic-x", "v1", "review-rubric-v2", None, 1000.0, 2000.0)
    rows = conn.execute(
        "SELECT ordinal, text, satisfied, bbox_x FROM ai_review_alignment WHERE review_id = ? ORDER BY ordinal",
        (review_id,),
    ).fetchall()
    assert [tuple(r) for r in rows] == [
        (0, "a red cube", 1, 0.1),
        (1, "a blue sphere", 0, None),
    ]

    # re-reviewing the same (file, rubric, model) replaces the old elements
    # rather than accumulating a second prompt's worth of rows
    replacement = validate_review_payload(
        valid_payload(
            prompt_alignment_score=1.0,
            alignment=[{"ordinal": 0, "text": "a green field", "satisfied": True, "confidence": 0.7}],
        )
    )
    store_review(conn, "f1", replacement, "critic-x", "v1", "review-rubric-v2", None, 1000.0, 3000.0)
    assert conn.execute("SELECT COUNT(*) FROM ai_review_alignment").fetchone()[0] == 1
    assert conn.execute("SELECT text FROM ai_review_alignment").fetchone()[0] == "a green field"


def test_alignment_check_constraint_is_live_for_direct_sql():
    """The 'absent elements have no locus' rule holds even for rows written
    outside validate_review_payload."""
    conn = make_conn()
    add_file(conn, "f1")
    result = validate_review_payload(valid_payload(prompt_alignment_score=None))
    review_id = store_review(conn, "f1", result, "critic-x", "v1", "review-rubric-v2", None, 1000.0, 2000.0)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ai_review_alignment "
            "(review_id, file_id, ordinal, text, satisfied, confidence, bbox_x) "
            "VALUES (?, 'f1', 0, 'x', 0, 0.5, 0.1)",
            (review_id,),
        )


def test_reject_non_dict_payload():
    with pytest.raises(ReviewSchemaError):
        validate_review_payload(["not", "a", "dict"])


def test_reject_bad_type_quality_score_as_string():
    with pytest.raises(ReviewSchemaError) as exc:
        validate_review_payload(valid_payload(quality_score="7.5"))
    assert exc.value.path == "quality_score"


def test_reject_bad_severity_value():
    payload = valid_payload()
    payload["findings"][0]["severity"] = "critical"
    with pytest.raises(ReviewSchemaError) as exc:
        validate_review_payload(payload)
    assert exc.value.path == "findings[0].severity"


def test_reject_confidence_out_of_range_1_5():
    payload = valid_payload()
    payload["findings"][0]["confidence"] = 1.5
    with pytest.raises(ReviewSchemaError) as exc:
        validate_review_payload(payload)
    assert exc.value.path == "findings[0].confidence"


def test_reject_localizable_without_geometry():
    payload = valid_payload()
    payload["findings"] = [
        {
            "type": "artifact",
            "severity": "high",
            "confidence": 0.9,
            "localizable": True,
            "description": "Missing bbox/points.",
        }
    ]
    with pytest.raises(ReviewSchemaError) as exc:
        validate_review_payload(payload)
    assert exc.value.path == "findings[0]"


def test_reject_global_finding_with_bbox():
    payload = valid_payload()
    payload["findings"] = [
        {
            "type": "lighting",
            "severity": "low",
            "confidence": 0.5,
            "localizable": False,
            "description": "Global finding wrongly grounded.",
            "bbox": [0.0, 0.0, 0.1, 0.1],
        }
    ]
    with pytest.raises(ReviewSchemaError) as exc:
        validate_review_payload(payload)
    assert exc.value.path == "findings[0]"


def test_reject_unknown_top_level_key():
    payload = valid_payload()
    payload["extra_field"] = "surprise"
    with pytest.raises(ReviewSchemaError) as exc:
        validate_review_payload(payload)
    assert exc.value.path == "$"


def test_reject_unknown_finding_key():
    payload = valid_payload()
    payload["findings"][0]["mask"] = "sneaky.png"
    with pytest.raises(ReviewSchemaError) as exc:
        validate_review_payload(payload)
    assert exc.value.path == "findings[0]"


def test_reject_missing_description():
    payload = valid_payload()
    del payload["findings"][0]["description"]
    with pytest.raises(ReviewSchemaError) as exc:
        validate_review_payload(payload)
    assert exc.value.path == "findings[0].description"


def test_reject_quality_score_11():
    with pytest.raises(ReviewSchemaError) as exc:
        validate_review_payload(valid_payload(quality_score=11))
    assert exc.value.path == "quality_score"


def test_reject_localizable_wrong_type():
    payload = valid_payload()
    payload["findings"][0]["localizable"] = "false"
    with pytest.raises(ReviewSchemaError) as exc:
        validate_review_payload(payload)
    assert exc.value.path == "findings[0].localizable"


# --- store_review + live CHECK constraint -----------------------------------


def test_store_review_inserts_review_and_findings():
    conn = make_conn()
    add_file(conn, "f1")
    result = validate_review_payload(valid_payload())
    review_id = store_review(conn, "f1", result, "critic-x", "v1", "review-rubric-v1", '{"raw": true}', 1000.0, 2000.0)
    review_row = conn.execute(
        "SELECT file_id, rubric_version, model_id, quality_score, prompt_alignment_score, summary "
        "FROM ai_reviews WHERE review_id = ?",
        (review_id,),
    ).fetchone()
    assert review_row == ("f1", "review-rubric-v1", "critic-x", 7.5, 0.6, "Mostly good, minor artifact.")

    findings = conn.execute(
        "SELECT type, localizable, bbox_x, bbox_y, bbox_w, bbox_h FROM ai_review_findings "
        "WHERE review_id = ? ORDER BY finding_id",
        (review_id,),
    ).fetchall()
    assert findings[0] == ("lighting", 0, None, None, None, None)
    assert findings[1][:2] == ("artifact", 1)
    assert findings[1][2:] == pytest.approx((0.1, 0.2, 0.1, 0.1))


def test_store_review_upserts_by_file_rubric_model():
    conn = make_conn()
    add_file(conn, "f1")
    result1 = validate_review_payload(valid_payload(summary="first pass"))
    review_id_1 = store_review(conn, "f1", result1, "critic-x", "v1", "rubric-1", None, 1000.0, 2000.0)

    result2 = validate_review_payload(valid_payload(summary="second pass", findings=[]))
    review_id_2 = store_review(conn, "f1", result2, "critic-x", "v1", "rubric-1", None, 1000.0, 2001.0)

    reviews = conn.execute("SELECT review_id, summary FROM ai_reviews WHERE file_id = ?", ("f1",)).fetchall()
    assert reviews == [(review_id_2, "second pass")]
    assert review_id_1 != review_id_2
    # old findings must be gone too (deleted with the old review)
    leftover = conn.execute("SELECT COUNT(*) FROM ai_review_findings WHERE review_id = ?", (review_id_1,)).fetchone()[0]
    assert leftover == 0


def test_review_findings_check_constraint_is_live_for_direct_sql():
    conn = make_conn()
    add_file(conn, "f1")
    result = validate_review_payload(valid_payload(findings=[]))
    review_id = store_review(conn, "f1", result, "critic-x", "v1", "rubric-1", None, 1000.0, 2000.0)

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO ai_review_findings
                (review_id, file_id, type, severity, confidence, localizable,
                 bbox_x, bbox_y, bbox_w, bbox_h, description)
            VALUES (?, 'f1', 'lighting', 'low', 0.5, 0, 0.1, 0.1, 0.1, 0.1, 'bad global finding')
            """,
            (review_id,),
        )


# --- StubReviewer ----------------------------------------------------------


def test_stub_critic_dark_image_yields_validated_global_lighting_finding():
    critic = StubReviewer()
    img = solid_color_image(color=(5, 5, 5))
    raw = critic.review(img, prompt_text=None, rubric_version="review-rubric-v1")
    result = validate_review_payload(raw)
    lighting = [f for f in result.findings if f.type == "lighting"]
    assert len(lighting) == 1
    assert lighting[0].localizable is False
    assert lighting[0].bbox is None
    assert result.prompt_alignment_score is None


def test_stub_critic_red_square_yields_localizable_artifact_overlapping_square():
    critic = StubReviewer()
    size = (64, 64)
    square = (24, 24, 16, 16)  # x, y, w, h in pixels; 16*16=256 = 1/16 of 4096
    img = image_with_red_square(size=size, square=square)
    raw = critic.review(img, prompt_text="a green field", rubric_version="review-rubric-v2")
    result = validate_review_payload(raw)

    artifacts = [f for f in result.findings if f.type == "artifact"]
    assert len(artifacts) == 1
    finding = artifacts[0]
    assert finding.localizable is True
    assert finding.bbox is not None

    bx, by, bw, bh = finding.bbox
    sx, sy, sw, sh = square
    sx_n, sy_n, sw_n, sh_n = sx / size[0], sy / size[1], sw / size[0], sh / size[1]
    # bbox found by the stub must overlap the square we actually drew
    overlap_x = max(0.0, min(bx + bw, sx_n + sw_n) - max(bx, sx_n))
    overlap_y = max(0.0, min(by + bh, sy_n + sh_n) - max(by, sy_n))
    assert overlap_x > 0.0
    assert overlap_y > 0.0
    # one prompt slice, satisfied -> the whole prompt was followed
    assert result.prompt_alignment_score == 1.0


def test_get_reviewer_stub_explicit_only():
    assert get_reviewer(AIConfig(critic_backend="none")) is None
    # 'auto' with a default (empty) models_dir: the qwen-vl critic is
    # unavailable because neither the OpenCLIP grounding dependency nor its
    # own weights resolve — this asserts the POST-flip fail-closed
    # behavior on an unprovisioned system, not the old always-None policy.
    assert get_reviewer(AIConfig(critic_backend="auto")) is None
    assert isinstance(get_reviewer(AIConfig(critic_backend="stub")), StubReviewer)
    with pytest.raises(ValueError):
        get_reviewer(AIConfig(critic_backend="bogus"))


# --- generate_finding_mask ----------------------------------------------------


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
    result = ReviewResult(quality_score=5.0, prompt_alignment_score=None, summary="s", findings=[finding])
    review_id = store_review(conn, file_id, result, "critic-x", "v1", "rubric-1", None, 1000.0, 2000.0)
    return conn.execute("SELECT finding_id FROM ai_review_findings WHERE review_id = ?", (review_id,)).fetchone()[0]


def test_generate_finding_mask_localizable_creates_png_source_untouched(tmp_path):
    conn = make_conn()
    add_file(conn, "f1")
    finding_id = _store_one_finding(conn, "f1", localizable=True, bbox=(0.25, 0.25, 0.5, 0.5))

    source_path = tmp_path / "source.png"
    solid_color_image(size=(40, 40)).save(source_path)
    source_bytes_before = source_path.read_bytes()
    source_mtime_before = os.path.getmtime(source_path)

    cache_dir = tmp_path / "cache"
    with Image.open(source_path) as img:
        mask_path = generate_finding_mask(conn, str(cache_dir), img.copy(), "f1", finding_id, StubSegmenter())

    assert os.path.isfile(mask_path)
    assert mask_path.startswith(os.path.realpath(str(cache_dir)))
    with Image.open(mask_path) as mask_img:
        # RGBA: the mask lives in the alpha channel so the panel can tint it
        assert mask_img.mode == "RGBA"
        mask_arr = np.asarray(mask_img)[..., 3]
        assert set(np.unique(mask_arr).tolist()) <= {0, 255}
        assert mask_arr[20, 20] == 255  # inside the bbox center

    row = conn.execute(
        "SELECT mask_path, mask_model_id, mask_model_version FROM ai_review_findings WHERE finding_id = ?",
        (finding_id,),
    ).fetchone()
    assert row == (mask_path, StubSegmenter.model_id, StubSegmenter.model_version)

    # source file must be byte-for-byte and mtime-for-mtime untouched
    assert source_path.read_bytes() == source_bytes_before
    assert os.path.getmtime(source_path) == source_mtime_before


def test_generate_finding_mask_raises_for_global_finding(tmp_path):
    conn = make_conn()
    add_file(conn, "f1")
    finding_id = _store_one_finding(conn, "f1", localizable=False)
    img = solid_color_image()
    with pytest.raises(MaskNotAllowedError):
        generate_finding_mask(conn, str(tmp_path / "cache"), img, "f1", finding_id, StubSegmenter())


def test_generate_finding_mask_path_traversal_stays_inside_cache_dir(tmp_path):
    conn = make_conn()
    # craft a file_id containing traversal sequences
    evil_file_id = "../../../../etc/evil"
    conn.execute(
        "INSERT INTO files (id, path, mtime, name, type) VALUES (?, ?, ?, ?, ?)",
        (evil_file_id, "/gallery/evil.png", 1000.0, "evil", "image"),
    )
    conn.commit()
    finding_id = _store_one_finding(conn, evil_file_id, localizable=True, bbox=(0.0, 0.0, 1.0, 1.0))

    cache_dir = tmp_path / "cache"
    img = solid_color_image()
    mask_path = generate_finding_mask(conn, str(cache_dir), img, evil_file_id, finding_id, StubSegmenter())

    real_cache = os.path.realpath(str(cache_dir))
    assert os.path.commonpath([real_cache, os.path.realpath(mask_path)]) == real_cache
    assert os.path.isfile(mask_path)
    # nothing was written outside the cache dir
    assert not os.path.exists(os.path.realpath(os.path.join(str(cache_dir), "..", "..", "..", "..", "etc", "evil")))


# --- Fail-closed critic dependency invariant (adversarial-audit fix) ---


def test_qwen_critic_requires_semantic_embedder():
    """The CLIP grounding gate is the anti-fabrication mechanism the
    critic's measured record relies on: constructing the critic without an
    embedder must be impossible, in the class AND through the factory."""

    # Class-level invariant: embedder=None is rejected before anything else
    # (no weights needed for this check to fire).
    with _pytest.raises(BackendUnavailable, match="grounding"):
        Reviewer("/nonexistent", semantic_embedder=None)

    # Factory 'auto': no semantic backend available -> critic unavailable
    # (returns None), never a gate-less critic.
    cfg = AIConfig(enabled=True, models_dir="/nonexistent", semantic_backend="none", critic_backend="auto")
    assert get_reviewer(cfg) is None

    # Factory explicit 'qwen-vl': surfaces the configuration error.
    cfg2 = AIConfig(enabled=True, models_dir="/nonexistent", semantic_backend="none", critic_backend="vlm")
    with _pytest.raises(BackendUnavailable):
        get_reviewer(cfg2)


# --- store_review replacement vs. mask files ---------------------------------


def _plant_mask(conn, finding_id, tmp_path, name="old_mask.png"):
    mask_file = tmp_path / name
    mask_file.write_bytes(b"mask-bytes")
    conn.execute(
        "UPDATE ai_review_findings SET mask_path = ?, mask_model_id = 'seg', "
        "mask_model_version = 'v1' WHERE finding_id = ?",
        (str(mask_file), finding_id),
    )
    conn.commit()
    return mask_file


def test_store_review_failed_replacement_preserves_old_review_and_mask(tmp_path):
    """A replacement that fails mid-transaction must roll back to the old
    review AND leave its mask file on disk — the file may only be unlinked
    once the replacement has committed."""
    conn = make_conn()
    add_file(conn, "f1")
    finding_id = _store_one_finding(conn, "f1", localizable=True, bbox=(0.25, 0.25, 0.5, 0.5))
    mask_file = _plant_mask(conn, finding_id, tmp_path)

    bad = ReviewResult(
        quality_score=5.0,
        prompt_alignment_score=None,
        summary="new",
        findings=[
            Finding(type="artifact", severity="catastrophic", confidence=0.9, localizable=False, description="d")
        ],
    )
    with pytest.raises(sqlite3.IntegrityError):
        store_review(conn, "f1", bad, "critic-x", "v1", "rubric-1", None, 1000.0, 3000.0)

    row = conn.execute(
        "SELECT r.summary, fi.mask_path FROM ai_reviews r JOIN ai_review_findings fi ON fi.review_id = r.review_id"
    ).fetchone()
    assert row == ("s", str(mask_file))
    assert mask_file.exists()


def test_store_review_successful_replacement_unlinks_old_mask(tmp_path):
    conn = make_conn()
    add_file(conn, "f1")
    finding_id = _store_one_finding(conn, "f1", localizable=True, bbox=(0.25, 0.25, 0.5, 0.5))
    mask_file = _plant_mask(conn, finding_id, tmp_path)

    good = ReviewResult(quality_score=6.0, prompt_alignment_score=None, summary="replacement", findings=[])
    store_review(conn, "f1", good, "critic-x", "v1", "rubric-1", None, 1000.0, 3000.0)
    assert conn.execute("SELECT summary FROM ai_reviews").fetchone()[0] == "replacement"
    assert not mask_file.exists()


# --- 'auto' critic enablement is derived from committed evidence --------------


def test_auto_critic_gate_binds_to_evidence_identity(tmp_path):
    """Acceptance requires the committed report's IDENTITY — backend,
    baseline text, and input manifest hashes matching this checkout — not
    merely in-bounds numbers. A synthetic numbers-only report must never
    enable 'auto'."""

    with open(REV._CALIBRATION_REPORT_PATH, encoding="utf-8") as fh:
        real = _json.load(fh)

    # The committed evidence itself passes ('auto' ships on).
    assert _auto_critic_measurement_passed() is True

    counter = iter(range(100))

    def variant(mutate):
        report = copy.deepcopy(real)
        mutate(report)
        path = str(tmp_path / f"r{next(counter)}.json")
        with open(path, "w", encoding="utf-8") as fh:
            _json.dump(report, fh)
        return _auto_critic_measurement_passed(path)

    def shipped_row(report):
        return next(s for s in report["sweep"] if abs(s["margin_threshold"] - DEFAULT_GROUNDING_MIN_MARGIN) < 1e-9)

    # A faithful copy elsewhere passes: acceptance is content-bound.
    assert variant(lambda _r: None) is True

    # Out-of-bounds numbers fail even with correct identity.
    assert variant(lambda r: shipped_row(r).update(false_accept_rate=0.5)) is False
    assert variant(lambda r: shipped_row(r).update(false_reject_rate=0.9)) is False
    assert variant(lambda r: r.update(sweep=[])) is False

    # Identity mismatches fail even with in-bounds numbers.
    assert variant(lambda r: r["backend"].update(model_id="someone/else")) is False
    assert variant(lambda r: r["backend"].update(model_version="v0-other")) is False
    assert variant(lambda r: r.update(baseline_text="different baseline")) is False
    assert variant(lambda r: r.update(inputs=[])) is False
    assert variant(lambda r: r.pop("inputs")) is False

    def tamper_hash(r):
        entry = next(e for e in r["inputs"] if e.get("file") == "probes/data/calibration_portrait.png")
        entry["file_sha256"] = "0" * 64

    assert variant(tamper_hash) is False

    def drop_portrait(r):
        r["inputs"] = [e for e in r["inputs"] if e.get("file") != "probes/data/calibration_portrait.png"]

    assert variant(drop_portrait) is False

    # The failure mode this gate exists for: a bare numbers-only document.
    minimal = str(tmp_path / "minimal.json")
    with open(minimal, "w", encoding="utf-8") as fh:
        _json.dump(
            {
                "sweep": [
                    {
                        "margin_threshold": DEFAULT_GROUNDING_MIN_MARGIN,
                        "false_accept_rate": 0.0,
                        "false_reject_rate": 0.0,
                    }
                ]
            },
            fh,
        )
    assert _auto_critic_measurement_passed(minimal) is False

    assert _auto_critic_measurement_passed(str(tmp_path / "absent.json")) is False


# --- summary quotes the grounded description, never rejected finding text -----


def test_qwen_critic_summary_survives_rejected_last_finding(monkeypatch):
    """When the final defect fails crop verification, the summary must keep
    quoting the step-1 (grounded) description; the rejected defect text must
    not appear anywhere in the payload."""

    reviewer = object.__new__(CQ.Reviewer)
    reviewer._embedder = object()  # patched functions below never touch it
    reviewer._grounding_min_cos = CQ.DEFAULT_GROUNDING_MIN_COS
    reviewer._models_dir, reviewer._device = "", "cpu"
    reviewer.model_id = "fake/model"

    class _ScriptedChat:
        """One conversation with scripted answers, keyed by tool name.

        `ask_json` returns PARSED objects, because the real one already
        parsed the tool call. Every protocol step is a tool call --
        including describe."""

        def ask_json(self, _prompt, name="", max_new_tokens=512, attempts=2):
            del _prompt, max_new_tokens, attempts
            if name == "describe":
                return {"description": "A cat sitting on a red sofa. The room is bright."}
            if name == "assess":
                return {
                    "quality_score": 7.0,
                    "defects": [
                        {
                            "type": "artifact",
                            "severity": "low",
                            "confidence": 0.9,
                            "region": "bottom-right",
                            "what": "fabricated glitch text",
                        }
                    ],
                }
            return {"x": 0.6, "y": 0.6, "w": 0.2, "h": 0.2}  # locate

    monkeypatch.setattr(CQ.ai_models, "Chat", lambda *_a, **_k: _ScriptedChat())
    monkeypatch.setattr(CQ, "check_grounding", lambda *_a, **_k: 0.12)
    monkeypatch.setattr(CQ, "verify_finding_region", lambda *_a, **_k: False)

    payload = reviewer.review(solid_color_image(size=(64, 64)), None, "rubric-1")
    assert payload["findings"] == []
    assert payload["summary"].startswith("A cat sitting on a red sofa.")
    assert "fabricated glitch text" not in payload["summary"]
    assert "1 finding(s) dropped" in payload["summary"]
