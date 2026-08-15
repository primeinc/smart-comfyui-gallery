"""Tests for smartgallery_ai.review: strict payload validation (acceptance +
rejection cases), store_review upsert + the live ai_review_findings CHECK
constraint, StubCritic heuristics, and generate_finding_mask (including the
path-traversal guard and source-file untouched guarantee)."""

import os
import sqlite3

import numpy as np
import pytest
from PIL import Image

from smartgallery_ai.review import (
    Finding,
    MaskNotAllowedError,
    ReviewResult,
    ReviewSchemaError,
    StubCritic,
    StubSegmenter,
    generate_finding_mask,
    get_critic_backend,
    store_review,
    validate_review_payload,
)
from smartgallery_ai.schema import init_schema
from smartgallery_ai import AIConfig


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
        "prompt_alignment_score": 6.0,
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
    assert result.prompt_alignment_score == 6.0
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
    review_id = store_review(
        conn, "f1", result, "critic-x", "v1", "review-rubric-v1", '{"raw": true}', 1000.0, 2000.0
    )
    review_row = conn.execute(
        "SELECT file_id, rubric_version, model_id, quality_score, prompt_alignment_score, summary "
        "FROM ai_reviews WHERE review_id = ?",
        (review_id,),
    ).fetchone()
    assert review_row == ("f1", "review-rubric-v1", "critic-x", 7.5, 6.0, "Mostly good, minor artifact.")

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
    leftover = conn.execute(
        "SELECT COUNT(*) FROM ai_review_findings WHERE review_id = ?", (review_id_1,)
    ).fetchone()[0]
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


# --- StubCritic ----------------------------------------------------------


def test_stub_critic_dark_image_yields_validated_global_lighting_finding():
    critic = StubCritic()
    img = solid_color_image(color=(5, 5, 5))
    raw = critic.review(img, prompt_text=None, rubric_version="review-rubric-v1")
    result = validate_review_payload(raw)
    lighting = [f for f in result.findings if f.type == "lighting"]
    assert len(lighting) == 1
    assert lighting[0].localizable is False
    assert lighting[0].bbox is None
    assert result.prompt_alignment_score is None


def test_stub_critic_red_square_yields_localizable_artifact_overlapping_square():
    critic = StubCritic()
    size = (64, 64)
    square = (24, 24, 16, 16)  # x, y, w, h in pixels; 16*16=256 = 1/16 of 4096
    img = image_with_red_square(size=size, square=square)
    raw = critic.review(img, prompt_text="a green field", rubric_version="review-rubric-v1")
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
    assert overlap_x > 0.0 and overlap_y > 0.0
    assert result.prompt_alignment_score == 5.0


def test_get_critic_backend_stub_explicit_only():
    assert get_critic_backend(AIConfig(critic_backend="none")) is None
    assert get_critic_backend(AIConfig(critic_backend="auto")) is None
    assert isinstance(get_critic_backend(AIConfig(critic_backend="stub")), StubCritic)
    with pytest.raises(ValueError):
        get_critic_backend(AIConfig(critic_backend="bogus"))


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
    finding_id = conn.execute(
        "SELECT finding_id FROM ai_review_findings WHERE review_id = ?", (review_id,)
    ).fetchone()[0]
    return finding_id


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
        assert mask_img.mode == "L"
        mask_arr = np.asarray(mask_img)
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
