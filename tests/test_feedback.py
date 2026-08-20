"""Tests for smartgallery_ai.feedback: record/export round-trip, JSONL
shape, exported_at stamping, and validation against the ai_feedback CHECKs
(target_kind, verdict, rating bounds)."""

import json
import sqlite3

import pytest

from smartgallery_ai.feedback import (
    FeedbackValidationError,
    export_feedback,
    list_feedback,
    record_feedback,
)
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


def add_file(conn, file_id):
    conn.execute(
        "INSERT INTO files (id, path, mtime, name, type) VALUES (?, ?, ?, ?, ?)",
        (file_id, f"/gallery/{file_id}.png", 1000.0, file_id, "image"),
    )
    conn.commit()


# --- record_feedback: acceptance ---------------------------------------------


def test_record_feedback_minimal_insert():
    conn = make_conn()
    add_file(conn, "f1")
    feedback_id = record_feedback(conn, "review", "42", "accept", file_id="f1", now=1000.0)
    row = conn.execute(
        "SELECT target_kind, target_id, file_id, verdict, rating, note, created_by, created_at, exported_at "
        "FROM ai_feedback WHERE feedback_id = ?",
        (feedback_id,),
    ).fetchone()
    assert row == ("review", "42", "f1", "accept", None, None, None, 1000.0, None)


def test_record_feedback_with_rating():
    conn = make_conn()
    feedback_id = record_feedback(
        conn, "face_cluster", "7", "rating", rating=4, note="looks right", created_by="will", now=500.0
    )
    row = conn.execute(
        "SELECT rating, note, created_by FROM ai_feedback WHERE feedback_id = ?", (feedback_id,)
    ).fetchone()
    assert row == (4, "looks right", "will")


def test_record_feedback_defaults_now_to_current_time():
    conn = make_conn()
    before = __import__("time").time()
    feedback_id = record_feedback(conn, "duplicate", "d1", "reject")
    after = __import__("time").time()
    created_at = conn.execute("SELECT created_at FROM ai_feedback WHERE feedback_id = ?", (feedback_id,)).fetchone()[0]
    assert before <= created_at <= after


# --- record_feedback: rejections ---------------------------------------------


def test_record_feedback_rejects_invalid_target_kind():
    conn = make_conn()
    with pytest.raises(FeedbackValidationError):
        record_feedback(conn, "not_a_kind", "1", "accept")
    assert conn.execute("SELECT COUNT(*) FROM ai_feedback").fetchone()[0] == 0


def test_record_feedback_rejects_invalid_verdict():
    conn = make_conn()
    with pytest.raises(FeedbackValidationError):
        record_feedback(conn, "review", "1", "maybe")
    assert conn.execute("SELECT COUNT(*) FROM ai_feedback").fetchone()[0] == 0


@pytest.mark.parametrize("bad_rating", [0, 6, -1, 100])
def test_record_feedback_rejects_out_of_bounds_rating(bad_rating):
    conn = make_conn()
    with pytest.raises(FeedbackValidationError):
        record_feedback(conn, "review", "1", "rating", rating=bad_rating)


def test_record_feedback_rejects_non_integer_rating():
    conn = make_conn()
    with pytest.raises(FeedbackValidationError):
        record_feedback(conn, "review", "1", "rating", rating=3.5)
    with pytest.raises(FeedbackValidationError):
        record_feedback(conn, "review", "1", "rating", rating=True)


@pytest.mark.parametrize("good_rating", [1, 2, 3, 4, 5])
def test_record_feedback_accepts_boundary_ratings(good_rating):
    conn = make_conn()
    feedback_id = record_feedback(conn, "review", "1", "rating", rating=good_rating)
    got = conn.execute("SELECT rating FROM ai_feedback WHERE feedback_id = ?", (feedback_id,)).fetchone()[0]
    assert got == good_rating


def test_record_feedback_rejects_empty_target_id():
    conn = make_conn()
    with pytest.raises(FeedbackValidationError):
        record_feedback(conn, "review", "", "accept")


def test_ai_feedback_check_constraints_are_also_live_at_db_level():
    conn = make_conn()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO ai_feedback (target_kind, target_id, verdict, created_at) "
            "VALUES ('bogus_kind', '1', 'accept', 1000.0)"
        )


# --- list_feedback / export_feedback -----------------------------------------


def test_list_feedback_orders_by_id_and_filters_unexported():
    conn = make_conn()
    id1 = record_feedback(conn, "review", "1", "accept", now=100.0)
    id2 = record_feedback(conn, "review", "2", "reject", now=200.0)
    export_feedback(conn, mark=True)
    id3 = record_feedback(conn, "review", "3", "accept", now=300.0)

    all_rows = list_feedback(conn)
    assert [r["feedback_id"] for r in all_rows] == [id1, id2, id3]

    unexported = list_feedback(conn, unexported_only=True)
    assert [r["feedback_id"] for r in unexported] == [id3]


def test_export_feedback_round_trip_jsonl_and_stamps_exported_at(tmp_path):
    conn = make_conn()
    id1 = record_feedback(conn, "review", "1", "accept", file_id="f1", now=100.0)
    id2 = record_feedback(conn, "finding", "2", "false_positive", rating=None, note="oops", now=200.0)

    out_path = tmp_path / "feedback.jsonl"
    text = export_feedback(conn, out_path=str(out_path), mark=True)

    # written file matches returned string
    assert out_path.read_text(encoding="utf-8") == text

    lines = [line for line in text.splitlines() if line]
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]

    # canonical JSON: sorted keys, one object per line
    for line in lines:
        obj = json.loads(line)
        assert list(json.loads(line).keys()) == sorted(obj.keys())

    by_id = {row["feedback_id"]: row for row in parsed}
    assert by_id[id1]["verdict"] == "accept"
    assert by_id[id1]["file_id"] == "f1"
    assert by_id[id2]["note"] == "oops"

    # all columns present
    expected_columns = {
        "feedback_id",
        "target_kind",
        "target_id",
        "file_id",
        "verdict",
        "rating",
        "note",
        "created_by",
        "created_at",
        "exported_at",
    }
    assert set(by_id[id1].keys()) == expected_columns

    # exported_at was stamped in the DB for both rows
    db_rows = conn.execute("SELECT feedback_id, exported_at FROM ai_feedback").fetchall()
    for feedback_id, exported_at in db_rows:
        assert exported_at is not None
        assert by_id[feedback_id]["exported_at"] == pytest.approx(exported_at)


def test_export_feedback_preserves_original_exported_at_on_second_export():
    conn = make_conn()
    record_feedback(conn, "review", "1", "accept", now=100.0)
    export_feedback(conn, mark=True)
    first_exported_at = conn.execute("SELECT exported_at FROM ai_feedback").fetchone()[0]

    record_feedback(conn, "review", "2", "accept", now=200.0)
    export_feedback(conn, mark=True)
    exported_ats = [r[0] for r in conn.execute("SELECT exported_at FROM ai_feedback ORDER BY feedback_id")]
    assert exported_ats[0] == pytest.approx(first_exported_at)
    assert exported_ats[1] is not None


def test_export_feedback_without_mark_leaves_exported_at_null():
    conn = make_conn()
    record_feedback(conn, "review", "1", "accept", now=100.0)
    text = export_feedback(conn, mark=False)
    obj = json.loads(text.strip())
    assert obj["exported_at"] is None
    db_value = conn.execute("SELECT exported_at FROM ai_feedback").fetchone()[0]
    assert db_value is None


def test_export_feedback_empty_table_returns_empty_string():
    conn = make_conn()
    assert export_feedback(conn) == ""
