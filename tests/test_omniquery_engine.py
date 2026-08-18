"""End-to-end tests for omniquery/engine.py against the deterministic
fixture database (omniquery/benchmark/fixtures.py)."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime

import pytest

from omniquery.ast import parse_query
from omniquery.benchmark.fixtures import (
    ANCHOR_EPOCH,
    FIXTURE_BASE_PATH,
    FIXTURE_EXPECTATIONS,
    FIXTURE_FILES,
    build_fixture_db,
)
from omniquery.engine import OmniQueryEngine
from omniquery.validation import AuthContext

GUEST = AuthContext(role="GUEST", user_id=None, client_uuid=None, ai_enabled=False)
STAFF_AI = AuthContext(role="STAFF", user_id="3", client_uuid="3", ai_enabled=True)
ADMIN_NO_AI = AuthContext(role="ADMIN", user_id="5", client_uuid="5", ai_enabled=False)


@pytest.fixture(scope="module")
def fixture_db_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("omniquery") / "fixture.db")
    build_fixture_db(path, seed=42)
    return path


@pytest.fixture
def engine(fixture_db_path):
    return OmniQueryEngine(
        db_path=fixture_db_path,
        base_path=FIXTURE_BASE_PATH,
        ai_resolvers={
            "similar_to_semantic": lambda _v: ["f001", "f002"],
            "similar_to_visual": lambda _v: [],
            "near_dup_of": lambda _v: ["f003"],
        },
    )


def _ids(files, **conditions):
    out = set()
    for f in files:
        if all(f[k] == v for k, v in conditions.items()):
            out.add(f["id"])
    return out


# ---------------------------------------------------------------------------
# 1. text
# ---------------------------------------------------------------------------


def test_text_contains(engine):
    out = engine.run(
        {"where": {"field": "workflow_prompt", "op": "contains", "value": "cyberpunk"}, "limit": 200},
        GUEST,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    assert set(out.ids) == FIXTURE_EXPECTATIONS["workflow_prompt_contains_cyberpunk"]


# ---------------------------------------------------------------------------
# 2. number
# ---------------------------------------------------------------------------


def test_number_comparison(engine):
    out = engine.run(
        {"where": {"field": "size_bytes", "op": "gt", "value": 20 * 1024 * 1024}, "limit": 200},
        GUEST,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    assert set(out.ids) == FIXTURE_EXPECTATIONS["size_gt_20mb"]


def test_number_between_duration(engine):
    expected = {
        f["id"] for f in FIXTURE_FILES if f["duration_seconds"] is not None and 60 <= f["duration_seconds"] <= 600
    }
    out = engine.run(
        {"where": {"field": "duration_seconds", "op": "between", "value": [60, 600]}, "limit": 200},
        GUEST,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    assert set(out.ids) == expected


# ---------------------------------------------------------------------------
# 3. date
# ---------------------------------------------------------------------------


def test_date_relative_days_ago(engine):
    out = engine.run(
        {"where": {"field": "mtime", "op": "ge", "value": {"days_ago": 60}}, "limit": 200},
        GUEST,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    expected = {f["id"] for f in FIXTURE_FILES if f["mtime"] >= ANCHOR_EPOCH - 60 * 86400.0}
    assert set(out.ids) == expected
    assert len(expected) > 0


def test_date_between_bare_dates_matches_calendar_days(engine):
    lo = time.mktime(datetime(2024, 1, 1).timetuple())
    hi = time.mktime(datetime(2025, 6, 1).timetuple())
    out = engine.run(
        {"where": {"field": "mtime", "op": "between", "value": ["2024-01-01", "2025-05-31"]}, "limit": 200},
        GUEST,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    expected = {f["id"] for f in FIXTURE_FILES if lo <= f["mtime"] < hi}
    assert set(out.ids) == expected


# ---------------------------------------------------------------------------
# 4. bool
# ---------------------------------------------------------------------------


def test_bool_eq(engine):
    out = engine.run(
        {"where": {"field": "is_favorite", "op": "eq", "value": True}, "limit": 200},
        GUEST,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    assert set(out.ids) == FIXTURE_EXPECTATIONS["is_favorite_true"]


# ---------------------------------------------------------------------------
# 5. enum
# ---------------------------------------------------------------------------


def test_enum_eq(engine):
    out = engine.run(
        {"where": {"field": "type", "op": "eq", "value": "image"}, "limit": 200},
        GUEST,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    assert set(out.ids) == FIXTURE_EXPECTATIONS["type_image"]


def test_enum_in(engine):
    out = engine.run(
        {"where": {"field": "type", "op": "in", "value": ["video", "audio"]}, "limit": 200},
        GUEST,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    assert set(out.ids) == FIXTURE_EXPECTATIONS["type_video_or_audio"]


# ---------------------------------------------------------------------------
# 6. boolean logic: NOT + nested OR inside AND
# ---------------------------------------------------------------------------


def test_not_and_nested_or(engine):
    where = {
        "op": "and",
        "children": [
            {"field": "type", "op": "eq", "value": "image"},
            {"op": "not", "child": {"field": "is_favorite", "op": "eq", "value": True}},
            {
                "op": "or",
                "children": [
                    {"field": "has_workflow", "op": "eq", "value": True},
                    {"field": "ai_caption", "op": "not_null"},
                ],
            },
        ],
    }
    out = engine.run({"where": where, "limit": 200}, GUEST, now_epoch=ANCHOR_EPOCH)
    assert out.ok
    expected = {
        f["id"]
        for f in FIXTURE_FILES
        if f["type"] == "image" and f["is_favorite"] != 1 and (f["has_workflow"] == 1 or f["ai_caption"] is not None)
    }
    assert set(out.ids) == expected
    assert len(expected) > 0


# ---------------------------------------------------------------------------
# 7-10. joins: ratings, comments, collections, status flags
# ---------------------------------------------------------------------------


def test_join_rating_avg(engine):
    out = engine.run(
        {"where": {"field": "rating_avg", "op": "ge", "value": 4}, "limit": 200},
        GUEST,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    assert set(out.ids) == FIXTURE_EXPECTATIONS["rating_avg_ge_4"]


def test_join_comment_contains(engine):
    out = engine.run(
        {"where": {"field": "comment_contains", "op": "contains", "value": "amazing"}, "limit": 200},
        GUEST,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    assert set(out.ids) == FIXTURE_EXPECTATIONS["comment_contains_amazing"]


def test_join_collection(engine):
    out = engine.run(
        {"where": {"field": "collection", "op": "eq", "value": "Portfolio"}, "limit": 200},
        GUEST,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    assert set(out.ids) == FIXTURE_EXPECTATIONS["collection_portfolio"]


def test_join_status_flag(engine):
    out = engine.run(
        {"where": {"field": "status_flag", "op": "eq", "value": "Approved"}, "limit": 200},
        GUEST,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    assert set(out.ids) == FIXTURE_EXPECTATIONS["status_flag_approved"]


def test_join_rated_by_user_privileged(engine):
    out = engine.run(
        {"where": {"field": "rated_by_user", "op": "eq", "value": "carol"}, "limit": 200},
        STAFF_AI,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    assert set(out.ids) == FIXTURE_EXPECTATIONS["rated_by_carol"]


def test_join_rated_by_user_denied_for_guest(engine):
    out = engine.run(
        {"where": {"field": "rated_by_user", "op": "eq", "value": "carol"}},
        GUEST,
        now_epoch=ANCHOR_EPOCH,
    )
    assert not out.ok
    assert "privileged" in out.error


# ---------------------------------------------------------------------------
# 11. counts
# ---------------------------------------------------------------------------


def test_count_result(engine):
    out = engine.run(
        {"result": "count", "where": {"field": "is_favorite", "op": "eq", "value": True}},
        GUEST,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    assert out.kind == "count"
    assert out.ids is None
    assert out.count == len(FIXTURE_EXPECTATIONS["is_favorite_true"])


def test_count_matches_ids_length_for_same_query(engine):
    where = {"field": "type", "op": "eq", "value": "image"}
    ids_out = engine.run({"where": where, "limit": 2000}, GUEST, now_epoch=ANCHOR_EPOCH)
    count_out = engine.run({"result": "count", "where": where}, GUEST, now_epoch=ANCHOR_EPOCH)
    assert ids_out.ok
    assert count_out.ok
    assert len(ids_out.ids) == count_out.count


# ---------------------------------------------------------------------------
# 12. AI predicates with stub resolvers + has_faces/face_cluster/review_issue
# ---------------------------------------------------------------------------


def test_ai_predicate_has_faces(engine):
    out = engine.run(
        {"where": {"field": "has_faces", "op": "eq", "value": True}, "limit": 200},
        STAFF_AI,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    assert set(out.ids) == FIXTURE_EXPECTATIONS["has_faces_true"]
    assert len(out.ids) > 0


def test_ai_predicate_similar_to_semantic_uses_stub_resolver(engine):
    out = engine.run(
        {"where": {"field": "similar_to_semantic", "op": "eq", "value": {"file_id": "f010", "k": 5}}},
        STAFF_AI,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    assert set(out.ids) == {"f001", "f002"}


def test_ai_predicate_similar_to_visual_empty_resolution(engine):
    out = engine.run(
        {"where": {"field": "similar_to_visual", "op": "eq", "value": "f010"}},
        STAFF_AI,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    assert out.ids == []


def test_ai_predicate_near_dup_of(engine):
    out = engine.run(
        {"where": {"field": "near_dup_of", "op": "eq", "value": "f010"}},
        STAFF_AI,
        now_epoch=ANCHOR_EPOCH,
    )
    assert out.ok
    assert out.ids == ["f003"]


# ---------------------------------------------------------------------------
# 13. unsupported-AI error paths
# ---------------------------------------------------------------------------


def test_ai_disabled_produces_validation_error_before_touching_resolvers(engine):
    out = engine.run(
        {"where": {"field": "has_faces", "op": "eq", "value": True}},
        GUEST,
        now_epoch=ANCHOR_EPOCH,
    )
    assert not out.ok
    assert "AI layer" in out.error


def test_missing_resolver_produces_ai_unavailable_error(fixture_db_path):
    engine_without_resolvers = OmniQueryEngine(
        db_path=fixture_db_path,
        base_path=FIXTURE_BASE_PATH,
        ai_resolvers=None,
    )
    out = engine_without_resolvers.run(
        {"where": {"field": "similar_to_semantic", "op": "eq", "value": "f001"}},
        STAFF_AI,
        now_epoch=ANCHOR_EPOCH,
    )
    assert not out.ok
    assert "AI feature unavailable" in out.error


def test_resolver_exception_produces_ai_unavailable_error(fixture_db_path):
    def _boom(_value):
        raise RuntimeError("model not loaded")

    engine_broken = OmniQueryEngine(
        db_path=fixture_db_path,
        base_path=FIXTURE_BASE_PATH,
        ai_resolvers={"similar_to_semantic": _boom},
    )
    out = engine_broken.run(
        {"where": {"field": "similar_to_semantic", "op": "eq", "value": "f001"}},
        STAFF_AI,
        now_epoch=ANCHOR_EPOCH,
    )
    assert not out.ok
    assert "AI feature unavailable" in out.error


# ---------------------------------------------------------------------------
# Structural / parse / validation error surfacing through run()
# ---------------------------------------------------------------------------


def test_run_surfaces_ast_errors(engine):
    out = engine.run({"target": "not_files"}, GUEST, now_epoch=ANCHOR_EPOCH)
    assert not out.ok
    assert out.ids is None
    assert out.count is None
    assert "invalid query" in out.error


def test_run_surfaces_validation_errors(engine):
    out = engine.run(
        {"where": {"field": "does_not_exist", "op": "eq", "value": "x"}},
        GUEST,
        now_epoch=ANCHOR_EPOCH,
    )
    assert not out.ok
    assert "unknown field" in out.error


def test_run_accepts_a_pre_parsed_query_object(engine):
    q = parse_query({"where": {"field": "type", "op": "eq", "value": "document"}, "limit": 200})
    out = engine.run(q, GUEST, now_epoch=ANCHOR_EPOCH)
    assert out.ok
    assert set(out.ids) == {f["id"] for f in FIXTURE_FILES if f["type"] == "document"}


# ---------------------------------------------------------------------------
# Read-only guarantee: no write can happen through this engine, even if
# compile() is bypassed entirely and raw SQL is handed to the executor.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_sql",
    [
        "INSERT INTO files (id, path, mtime, name) VALUES ('x', '/tmp/x', 1.0, 'x')",
        "UPDATE files SET is_favorite = 1",
        "DELETE FROM files",
        "DROP TABLE files",
        "ATTACH DATABASE ':memory:' AS evil",
        "PRAGMA journal_mode=WAL",
    ],
)
def test_authorizer_blocks_every_write_attempt(engine, bad_sql):
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        engine._execute(bad_sql, ())


def test_authorizer_allows_plain_select(engine):
    rows = engine._execute("SELECT id FROM files LIMIT 1", ())
    assert len(rows) == 1


def test_engine_run_never_mutates_row_count(engine):
    before = engine._execute("SELECT COUNT(*) FROM files", ())[0][0]
    engine.run({"where": {"field": "type", "op": "eq", "value": "image"}}, GUEST, now_epoch=ANCHOR_EPOCH)
    after = engine._execute("SELECT COUNT(*) FROM files", ())[0][0]
    assert before == after == len(FIXTURE_FILES)
