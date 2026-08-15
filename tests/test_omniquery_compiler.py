"""Tests for omniquery/compiler.py: golden SQL, determinism, injection
safety, and the file_ref resolution hand-off."""

from __future__ import annotations

import sqlite3
import time

import pytest

from omniquery.ast import parse_query
from omniquery.compiler import CompileError, CompileParams, compile as compile_query, resolution_key
from omniquery.validation import AuthContext, validate

STAFF = AuthContext(role="STAFF", user_id="3", client_uuid="client-3", ai_enabled=True)
GUEST = AuthContext(role="GUEST", user_id=None, client_uuid=None, ai_enabled=False)
NOW = 1735689600.0  # 2025-01-01T00:00:00 local


def _compile(obj, ctx=STAFF, ai_resolutions=None, base_path="/gallery"):
    vq = validate(parse_query(obj), ctx)
    params = CompileParams(now_epoch=NOW, base_path=base_path, client_uuid=ctx.client_uuid,
                            ai_resolutions=ai_resolutions or {})
    return compile_query(vq, params)


# ---------------------------------------------------------------------------
# Golden SQL snapshots
# ---------------------------------------------------------------------------

def test_golden_simple_eq():
    cq = _compile({"where": {"field": "name", "op": "eq", "value": "sunset.png"}})
    assert cq.sql == (
        "SELECT DISTINCT f.id FROM files f WHERE f.name LIKE ? ESCAPE '\\' "
        "ORDER BY f.id ASC LIMIT ?"
    )
    assert cq.params == ("sunset.png", 500)


def test_golden_enum_in():
    cq = _compile({"where": {"field": "type", "op": "in", "value": ["image", "video"]}})
    assert cq.sql == (
        "SELECT DISTINCT f.id FROM files f WHERE f.type IN (?,?) ORDER BY f.id ASC LIMIT ?"
    )
    assert cq.params == ("image", "video", 500)


def test_golden_folder_contains():
    cq = _compile({"where": {"field": "folder", "op": "contains", "value": "landscapes"}})
    assert cq.sql == (
        "SELECT DISTINCT f.id FROM files f WHERE (REPLACE(f.path, '\\', '/') LIKE ? ESCAPE '\\' "
        "AND REPLACE(f.path, '\\', '/') LIKE ? ESCAPE '\\') ORDER BY f.id ASC LIMIT ?"
    )
    assert cq.params == ("/gallery/%", "%landscapes%", 500)


def test_golden_folder_eq():
    cq = _compile({"where": {"field": "folder", "op": "eq", "value": "landscapes/2024"}})
    assert cq.sql == (
        "SELECT DISTINCT f.id FROM files f WHERE REPLACE(f.path, '\\', '/') LIKE ? ESCAPE '\\' "
        "ORDER BY f.id ASC LIMIT ?"
    )
    assert cq.params == ("/gallery/landscapes/2024/%", 500)


def test_folder_predicates_match_windows_separators():
    """Folder predicates must match rows regardless of which separator the
    scanning host stored, and accept backslashes in base_path/value."""
    cq = _compile({"where": {"field": "folder", "op": "eq", "value": "landscapes\\2024"}},
                  base_path="C:\\gallery")
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE files (id TEXT PRIMARY KEY, path TEXT)")
    conn.executemany("INSERT INTO files VALUES (?, ?)", [
        ("p1", "C:/gallery/landscapes/2024/b.png"),
        ("w1", "C:\\gallery\\landscapes\\2024\\a.png"),
        ("x1", "C:\\gallery\\other\\c.png"),
    ])
    assert [r[0] for r in conn.execute(cq.sql, cq.params)] == ["p1", "w1"]

    cq2 = _compile({"where": {"field": "folder", "op": "contains", "value": "landscapes"}},
                   base_path="C:\\gallery")
    assert [r[0] for r in conn.execute(cq2.sql, cq2.params)] == ["p1", "w1"]


@pytest.mark.skipif(not hasattr(time, "tzset"),
                    reason="requires time.tzset (POSIX) to control the process timezone")
def test_between_bare_dates_dst_transition_days():
    """A bare-date 'between' upper bound extends to the next local calendar
    midnight, which is 23h away on spring-forward day and 25h on fall-back
    day — never a fixed 86400s."""
    import os as _os
    import time as _time

    old_tz = _os.environ.get("TZ")
    _os.environ["TZ"] = "America/New_York"
    _time.tzset()
    try:
        spring = _compile({"where": {"field": "mtime", "op": "between",
                                     "value": ["2025-03-09", "2025-03-09"]}})
        lo, hi, _ = spring.params
        assert hi - lo == 23 * 3600.0
        fall = _compile({"where": {"field": "mtime", "op": "between",
                                   "value": ["2025-11-02", "2025-11-02"]}})
        lo2, hi2, _ = fall.params
        assert hi2 - lo2 == 25 * 3600.0
    finally:
        if old_tz is None:
            _os.environ.pop("TZ", None)
        else:
            _os.environ["TZ"] = old_tz
        _time.tzset()


def test_duration_seconds_expr_handles_hms_and_ms():
    """duration is stored as text in either MM:SS or H:MM:SS."""
    cq = _compile({"where": {"field": "duration_seconds", "op": "ge", "value": 3600}})
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE files (id TEXT PRIMARY KEY, path TEXT, duration TEXT)")
    conn.executemany("INSERT INTO files VALUES (?, '', ?)", [
        ("long", "1:02:03"),   # 3723s
        ("short", "02:03"),    # 123s
        ("nodur", None),
    ])
    assert [r[0] for r in conn.execute(cq.sql, cq.params)] == ["long"]

    cq2 = _compile({"where": {"field": "duration_seconds", "op": "between",
                              "value": [120, 300]}})
    assert [r[0] for r in conn.execute(cq2.sql, cq2.params)] == ["short"]


def test_golden_mtime_between_bare_dates_covers_whole_days():
    cq = _compile({"where": {"field": "mtime", "op": "between",
                              "value": ["2025-01-01", "2025-01-31"]}})
    assert cq.sql == (
        "SELECT DISTINCT f.id FROM files f WHERE f.mtime >= ? AND f.mtime < ? "
        "ORDER BY f.id ASC LIMIT ?"
    )
    lo, hi, limit = cq.params
    assert hi - lo == 31 * 86400.0  # Jan 1 00:00 .. Feb 1 00:00, half-open
    assert limit == 500


def test_golden_my_rating():
    cq = _compile({"where": {"field": "my_rating", "op": "ge", "value": 4}})
    assert cq.sql == (
        "SELECT DISTINCT f.id FROM files f WHERE "
        "(SELECT r.rating FROM file_ratings r WHERE r.file_id = f.id AND r.client_uuid = ?) >= ? "
        "ORDER BY f.id ASC LIMIT ?"
    )
    assert cq.params == ("client-3", 4, 500)


def test_golden_status_flag_ne_is_not_exists():
    cq = _compile({"where": {"field": "status_flag", "op": "ne", "value": "Approved"}})
    assert cq.sql == (
        "SELECT DISTINCT f.id FROM files f WHERE NOT EXISTS "
        "(SELECT 1 FROM collection_files cf JOIN collections c ON c.id = cf.collection_id "
        "WHERE cf.file_id = f.id AND c.type = ? AND c.name = ?) ORDER BY f.id ASC LIMIT ?"
    )
    assert cq.params == ("system_flag", "Approved", 500)


def test_golden_review_issue_in():
    cq = _compile({"where": {"field": "review_issue", "op": "in",
                              "value": ["anatomy", "artifact"]}})
    assert cq.sql == (
        "SELECT DISTINCT f.id FROM files f WHERE EXISTS "
        "(SELECT 1 FROM ai_review_findings rf WHERE rf.file_id = f.id AND rf.type IN (?,?)) "
        "ORDER BY f.id ASC LIMIT ?"
    )
    assert cq.params == ("anatomy", "artifact", 500)


def test_golden_count_query_shape():
    cq = _compile({"result": "count", "where": {"field": "is_favorite", "op": "eq", "value": True}})
    assert cq.sql == "SELECT COUNT(*) FROM (SELECT DISTINCT f.id FROM files f WHERE f.is_favorite = ?)"
    assert cq.params == (1,)


def test_golden_count_query_no_where():
    cq = _compile({"result": "count"})
    assert cq.sql == "SELECT COUNT(*) FROM (SELECT DISTINCT f.id FROM files f)"
    assert cq.params == ()


def test_golden_no_where_no_order_by_uses_id_tiebreak():
    cq = _compile({})
    assert cq.sql == "SELECT DISTINCT f.id FROM files f ORDER BY f.id ASC LIMIT ?"
    assert cq.params == (500,)


def test_golden_multi_order_by_appends_id_tiebreak():
    cq = _compile({"order_by": [{"field": "name", "dir": "asc"},
                                 {"field": "mtime", "dir": "desc"}], "limit": 10})
    assert cq.sql == (
        "SELECT DISTINCT f.id FROM files f "
        "ORDER BY f.name ASC, f.mtime DESC, f.id ASC LIMIT ?"
    )
    assert cq.params == (10,)


def test_golden_is_null():
    cq = _compile({"where": {"field": "ai_caption", "op": "is_null"}})
    assert cq.sql == "SELECT DISTINCT f.id FROM files f WHERE f.ai_caption IS NULL ORDER BY f.id ASC LIMIT ?"
    assert cq.params == (500,)


def test_golden_and_not_group():
    cq = _compile({"where": {"op": "and", "children": [
        {"field": "type", "op": "eq", "value": "image"},
        {"op": "not", "child": {"field": "is_favorite", "op": "eq", "value": True}},
    ]}})
    assert cq.sql == (
        "SELECT DISTINCT f.id FROM files f WHERE "
        "(f.type = ? AND NOT (f.is_favorite = ?)) ORDER BY f.id ASC LIMIT ?"
    )
    assert cq.params == ("image", 1, 500)


# ---------------------------------------------------------------------------
# Injection safety
# ---------------------------------------------------------------------------

def test_injection_attempt_stays_a_bound_value():
    payload = "x' OR 1=1 --"
    cq = _compile({"where": {"field": "name", "op": "eq", "value": payload}})
    assert payload not in cq.sql  # never interpolated into the SQL text
    assert cq.sql == "SELECT DISTINCT f.id FROM files f WHERE f.name LIKE ? ESCAPE '\\' ORDER BY f.id ASC LIMIT ?"
    assert cq.params == (payload, 500)

    # And it genuinely behaves as an inert literal against a real DB.
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE files (id TEXT, name TEXT)")
    conn.executemany("INSERT INTO files VALUES (?, ?)",
                      [("f1", "real.png"), ("f2", "other.png")])
    rows = conn.execute("SELECT id FROM files WHERE name LIKE ? ESCAPE '\\'", (payload,)).fetchall()
    assert rows == []


def test_like_metacharacters_in_value_are_escaped():
    cq = _compile({"where": {"field": "name", "op": "contains", "value": "100%_done"}})
    pattern = cq.params[0]
    assert pattern == "%100\\%\\_done%"

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE files (id TEXT, name TEXT)")
    conn.executemany("INSERT INTO files VALUES (?, ?)",
                      [("f1", "100%_done_report.png"), ("f2", "100Xdone_report.png")])
    rows = conn.execute("SELECT id FROM files WHERE name LIKE ? ESCAPE '\\'", (pattern,)).fetchall()
    assert [r[0] for r in rows] == ["f1"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_compiling_same_query_twice_is_byte_identical():
    obj = {"where": {"op": "or", "children": [
        {"field": "type", "op": "eq", "value": "image"},
        {"field": "rating_avg", "op": "ge", "value": 4},
    ]}, "order_by": [{"field": "rating_avg", "dir": "desc"}], "limit": 25}
    cq1 = _compile(obj)
    cq2 = _compile(obj)
    assert cq1.sql == cq2.sql
    assert cq1.params == cq2.params


def test_reordered_and_children_compile_to_different_but_stable_sql():
    obj_a = {"where": {"op": "and", "children": [
        {"field": "type", "op": "eq", "value": "image"},
        {"field": "is_favorite", "op": "eq", "value": True},
    ]}}
    obj_b = {"where": {"op": "and", "children": [
        {"field": "is_favorite", "op": "eq", "value": True},
        {"field": "type", "op": "eq", "value": "image"},
    ]}}
    cq_a1 = _compile(obj_a)
    cq_a2 = _compile(obj_a)
    cq_b = _compile(obj_b)
    assert cq_a1.sql == cq_a2.sql and cq_a1.params == cq_a2.params
    assert cq_a1.sql != cq_b.sql  # compiler does not canonicalize child order


# ---------------------------------------------------------------------------
# Date resolution against a fixed now_epoch
# ---------------------------------------------------------------------------

def test_days_ago_resolves_against_injected_now_epoch():
    cq = _compile({"where": {"field": "mtime", "op": "ge", "value": {"days_ago": 7}}})
    assert cq.params[0] == NOW - 7 * 86400.0


def test_hours_ago_resolves_against_injected_now_epoch():
    cq = _compile({"where": {"field": "mtime", "op": "ge", "value": {"hours_ago": 3}}})
    assert cq.params[0] == NOW - 3 * 3600.0


def test_bare_date_resolves_to_local_midnight():
    import time
    from datetime import datetime
    cq = _compile({"where": {"field": "mtime", "op": "ge", "value": "2025-06-15"}})
    expected = time.mktime(datetime(2025, 6, 15).timetuple())
    assert cq.params[0] == expected


def test_full_datetime_between_is_not_day_shifted():
    cq = _compile({"where": {"field": "mtime", "op": "between",
                              "value": ["2025-01-01T08:00:00", "2025-01-01T20:00:00"]}})
    lo, hi, _ = cq.params
    assert hi - lo == 12 * 3600.0  # exact span, no whole-day rounding


# ---------------------------------------------------------------------------
# file_ref resolution hand-off
# ---------------------------------------------------------------------------

def test_near_dup_in_list_expansion():
    key = resolution_key("near_dup_of", "f001")
    cq = _compile({"where": {"field": "near_dup_of", "op": "eq", "value": "f001"}},
                   ai_resolutions={key: ["f002", "f003", "f004"]})
    assert cq.sql == "SELECT DISTINCT f.id FROM files f WHERE f.id IN (?,?,?) ORDER BY f.id ASC LIMIT ?"
    assert cq.params == ("f002", "f003", "f004", 500)


def test_empty_resolution_compiles_to_false_predicate():
    key = resolution_key("near_dup_of", "f001")
    cq = _compile({"where": {"field": "near_dup_of", "op": "eq", "value": "f001"}},
                   ai_resolutions={key: []})
    assert cq.sql == "SELECT DISTINCT f.id FROM files f WHERE 0=1 ORDER BY f.id ASC LIMIT ?"
    assert cq.params == (500,)


def test_missing_resolution_raises_compile_error():
    with pytest.raises(CompileError, match="no AI resolution supplied"):
        _compile({"where": {"field": "near_dup_of", "op": "eq", "value": "f001"}})


def test_similar_to_dict_value_resolution_key_distinguishes_k():
    key5 = resolution_key("similar_to_semantic", {"file_id": "f001", "k": 5})
    key10 = resolution_key("similar_to_semantic", {"file_id": "f001", "k": 10})
    assert key5 != key10
    cq = _compile(
        {"where": {"field": "similar_to_semantic", "op": "eq", "value": {"file_id": "f001", "k": 5}}},
        ai_resolutions={key5: ["f009"]},
    )
    assert cq.params == ("f009", 500)


# ---------------------------------------------------------------------------
# compile() requires a validated query
# ---------------------------------------------------------------------------

def test_compile_rejects_non_validated_query_object():
    with pytest.raises(AssertionError):
        compile_query("not a ValidatedQuery", CompileParams(now_epoch=NOW, base_path="/gallery"))
