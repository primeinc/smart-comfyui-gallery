"""The upgrade path and the derived-state maintenance CLI.

Every existing install reaches a new release by opening its old database
and running `init_schema` at startup, then (optionally) the documented
`python -m smartgallery_ai rebuild` / `status` commands. Neither was
covered: `test_provision.py` exercises only the `provision` subcommand,
and nothing asserted that a database predating a newly added table gains
it without disturbing what is already there.

These tests are table-agnostic on purpose -- they derive their fixtures
from `schema.DERIVED_TABLES`, so a table added tomorrow is covered the
day it lands rather than when someone remembers to extend a list.
"""

from __future__ import annotations

import sqlite3

import pytest

from smartgallery_ai import schema
from smartgallery_ai.__main__ import main


def _base_db(path: str) -> sqlite3.Connection:
    """A gallery DB with one file row and the full AI schema."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE files (id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, "
        "mtime REAL NOT NULL, name TEXT NOT NULL, type TEXT, "
        "workflow_prompt TEXT DEFAULT '')"
    )
    conn.execute("INSERT INTO files VALUES ('f1', '/x/a.png', 1000.0, 'a.png', 'image', 'a red square')")
    schema.init_schema(conn)
    return conn


def _tables(conn) -> set:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _seed_review(conn) -> None:
    conn.execute(
        "INSERT INTO ai_reviews (file_id, rubric_version, model_id, model_version, "
        "quality_score, prompt_alignment_score, summary, raw_response, "
        "source_mtime, computed_at) VALUES ('f1', 'r1', 'm', 'v1', 8.0, 0.5, "
        "'s', '{}', 1000.0, 1.0)"
    )
    conn.commit()


@pytest.mark.parametrize("dropped", schema.DERIVED_TABLES)
def test_startup_restores_any_missing_derived_table(tmp_path, dropped):
    """A database predating any one derived table gains it on the next
    startup, and unrelated rows are untouched. This is the upgrade every
    existing install performs."""
    db = str(tmp_path / "g.sqlite")
    conn = _base_db(db)
    try:
        # ai_reviews is the parent of the rows we seed; seed first, then
        # simulate the older install missing `dropped`.
        if dropped != "ai_reviews":
            _seed_review(conn)
        conn.execute(f"DROP TABLE IF EXISTS {dropped}")
        conn.commit()
        assert dropped not in _tables(conn)

        schema.init_schema(conn)  # what app startup does

        assert dropped in _tables(conn), f"{dropped} not recreated on upgrade"
        if dropped != "ai_reviews":
            assert conn.execute("SELECT COUNT(*) FROM ai_reviews").fetchone()[0] == 1, (
                f"recreating {dropped} disturbed existing rows"
            )
    finally:
        conn.close()


def test_derived_tables_lists_every_ai_table_the_schema_creates(tmp_path):
    """`rebuild` drops exactly DERIVED_TABLES, so a table this package
    creates but forgets to list survives a rebuild carrying stale derived
    rows -- silently, since nothing else looks. Deriving the expectation
    from the live schema means a table added tomorrow is covered without
    anyone remembering to extend a list.

    (Drop ORDER deliberately goes unasserted: every child FK here is ON
    DELETE CASCADE, so both orders succeed -- a test claiming to guard the
    order would pass no matter what it was, which is worse than no test.)
    """
    db = str(tmp_path / "g.sqlite")
    conn = _base_db(db)
    try:
        created = {t for t in _tables(conn) if t.startswith("ai_")}
        listed = set(schema.DERIVED_TABLES)
        assert created - listed == set(), (
            f"tables created but missing from DERIVED_TABLES (rebuild would "
            f"leave them stale): {sorted(created - listed)}"
        )
        assert listed - created == set(), (
            f"DERIVED_TABLES names tables the schema never creates: {sorted(listed - created)}"
        )
    finally:
        conn.close()


def test_cli_rebuild_clears_derived_state_but_keeps_feedback(tmp_path, capsys):
    db = str(tmp_path / "g.sqlite")
    conn = _base_db(db)
    _seed_review(conn)
    conn.execute(
        "INSERT INTO ai_feedback (target_kind, target_id, file_id, "
        "verdict, created_at) VALUES ('finding', 1, 'f1', 'accept', 1.0)"
    )
    conn.commit()
    conn.close()

    assert main(["rebuild", "--db", db]) == 0
    assert "dropped" in capsys.readouterr().out.lower()

    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM ai_reviews").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM ai_feedback").fetchone()[0] == 1, (
            "human feedback is not recomputable and must survive a rebuild"
        )
        # The schema is recreated, not merely emptied.
        assert set(schema.DERIVED_TABLES) <= _tables(conn)
    finally:
        conn.close()


def test_cli_rebuild_drop_feedback_removes_it(tmp_path):
    db = str(tmp_path / "g.sqlite")
    conn = _base_db(db)
    conn.execute(
        "INSERT INTO ai_feedback (target_kind, target_id, file_id, "
        "verdict, created_at) VALUES ('finding', 1, 'f1', 'accept', 1.0)"
    )
    conn.commit()
    conn.close()

    assert main(["rebuild", "--db", db, "--drop-feedback"]) == 0
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM ai_feedback").fetchone()[0] == 0
    finally:
        conn.close()


def test_cli_status_reports_every_derived_table(tmp_path, capsys):
    db = str(tmp_path / "g.sqlite")
    conn = _base_db(db)
    _seed_review(conn)
    conn.close()

    assert main(["status", "--db", db]) == 0
    out = capsys.readouterr().out
    for table in schema.DERIVED_TABLES:
        assert table in out, f"status does not report {table}"
    assert "ai_reviews               1" in " ".join(out.split("\n"))


def test_cli_status_marks_absent_tables_missing(tmp_path, capsys):
    """A table the DB has never had reads as 'missing', not as a crash --
    `status` has to work on a database from any older version."""
    db = str(tmp_path / "g.sqlite")
    conn = _base_db(db)
    conn.execute("DROP TABLE ai_review_findings")
    conn.commit()
    conn.close()

    assert main(["status", "--db", db]) == 0
    assert "missing" in capsys.readouterr().out
