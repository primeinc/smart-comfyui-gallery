"""Tests for the nl2sql SQL search path: the sandboxed executor
(omniquery.sqlexec) and SqlSearch's agentic generate/execute/read-results
loop. Model-free: generations are scripted by monkeypatching
SqlSearch._chat; every execution runs through the REAL sandbox
against a real temp database."""

from __future__ import annotations

import sqlite3

import pytest

from omniquery.parsers import nl2sql
from omniquery.parsers.nl2sql import SqlSearch, _extract_sql, schema_block
from omniquery.sqlexec import run_readonly_select


@pytest.fixture
def db(tmp_path):
    path = str(tmp_path / "g.sqlite")
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE files (id TEXT PRIMARY KEY, name TEXT, type TEXT, workflow_prompt TEXT, is_favorite INTEGER)"
    )
    conn.executemany(
        "INSERT INTO files VALUES (?,?,?,?,?)",
        [
            ("f1", "a.png", "image", "a girlnextdoor portrait", 1),
            ("f2", "b.mp4", "video", "", 0),
            ("f3", "c.png", "image", "mountain landscape", 0),
        ],
    )
    conn.execute("CREATE TABLE generation_params (file_id TEXT PRIMARY KEY, loras TEXT, positive_prompt TEXT)")
    conn.execute("CREATE TABLE collections (id INTEGER PRIMARY KEY, name TEXT, type TEXT)")
    conn.execute("INSERT INTO collections VALUES (1, 'Approved', 'system_flag')")
    conn.commit()
    conn.close()
    nl2sql._SCHEMA_CACHE.clear()
    return path


# ---------------------------------------------------------------------------
# sqlexec: the one sandboxed gate
# ---------------------------------------------------------------------------


def test_sqlexec_select_returns_first_column_deduped(db):
    r = run_readonly_select(db, "SELECT id FROM files UNION ALL SELECT id FROM files")
    assert r.ok
    assert sorted(r.ids) == ["f1", "f2", "f3"]


def test_sqlexec_rejects_non_select(db):
    for sql in (
        "DELETE FROM files",
        "UPDATE files SET is_favorite=1",
        "PRAGMA user_version",
        "ATTACH ':memory:' AS x",
        "/* sneaky */ DROP TABLE files",
        "",
    ):
        r = run_readonly_select(db, sql)
        assert not r.ok, sql


def test_sqlexec_blocks_writes_at_engine_level(db):
    # A statement that passes the prefix check but tries to smuggle a
    # write dies inside the engine (authorizer / read-only VFS), and the
    # data is untouched.
    r = run_readonly_select(db, "SELECT id FROM files WHERE id IN (SELECT id FROM files); DELETE FROM files")
    assert not r.ok
    check = sqlite3.connect(db).execute("SELECT COUNT(*) FROM files").fetchone()[0]
    assert check == 3


def test_sqlexec_engine_error_is_reported_not_raised(db):
    r = run_readonly_select(db, "SELECT nope FROM files")
    assert not r.ok
    assert "no such column" in r.error


# ---------------------------------------------------------------------------
# schema_block: the live schema the model reads
# ---------------------------------------------------------------------------


def test_schema_block_is_live_ddl_with_value_hints(db):
    block = schema_block(db)
    assert "CREATE TABLE files" in block
    assert "generation_params" in block
    assert "files.type values: image, video" in block
    assert "Approved" in block


def test_schema_block_never_includes_bookkeeping_tables(db):
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE omniquery_sessions (session_id TEXT)")
    conn.commit()
    conn.close()
    nl2sql._SCHEMA_CACHE.clear()
    assert "omniquery_sessions" not in schema_block(db)


# ---------------------------------------------------------------------------
# _extract_sql: the model's raw text -> one statement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("SELECT id FROM files", "SELECT id FROM files"),
        ("```sql\nSELECT id FROM files\n```", "SELECT id FROM files"),
        # Observed live: free-running past the answer.
        ("SELECT id FROM files; [INST] SELECT 1", "SELECT id FROM files"),
        ("  SELECT id FROM files;  ", "SELECT id FROM files"),
        # Observed live: chat-template tokens as literal trailing text.
        ("SELECT id FROM files WHERE x < 5 <tool_call><s>", "SELECT id FROM files WHERE x < 5"),
        ("SELECT id FROM files <|im_end|>", "SELECT id FROM files"),
        # Legal comparisons survive: '<' before space/digit/'='/'>' is SQL.
        (
            "SELECT id FROM files WHERE a < 5 AND b <= 2 AND c <> 'x'",
            "SELECT id FROM files WHERE a < 5 AND b <= 2 AND c <> 'x'",
        ),
    ],
)
def test_extract_sql(content, expected):
    assert _extract_sql(content) == expected


# ---------------------------------------------------------------------------
# SqlSearch.search: the agentic loop reads execution results
# ---------------------------------------------------------------------------


class _ScriptedChat:
    """A models.Chat stand-in: replies in order, recording each prompt.

    The real Chat keeps the conversation itself, so a round only ever
    receives the NEW instruction -- which is exactly what `seen` records.
    """

    def __init__(self, generations, seen):
        self._generations = generations
        self._seen = seen

    def ask(self, prompt, max_new_tokens=256):
        del max_new_tokens
        self._seen.append(prompt)
        return self._generations[len(self._seen) - 1]


def _scripted(monkeypatch, db, generations):
    """A SqlSearch whose model 'generations' are scripted in order; the
    prompt each round received is recorded for assertions."""
    s = SqlSearch(db_path=db)
    seen = []
    monkeypatch.setattr(SqlSearch, "_chat", lambda self: _ScriptedChat(generations, seen))
    return s, seen


def test_first_try_with_rows_is_accepted(monkeypatch, db):
    s, seen = _scripted(monkeypatch, db, ["SELECT id FROM files WHERE is_favorite = 1"])
    ids, _sql, err = s.search("favorites")
    assert (ids, err) == (["f1"], None)
    assert len(seen) == 1


def test_execution_error_goes_back_for_repair(monkeypatch, db):
    s, seen = _scripted(
        monkeypatch,
        db,
        [
            "SELECT id FROM files JOIN nope ON 1=1",
            "SELECT id FROM files WHERE type = 'video'",
        ],
    )
    ids, _sql, err = s.search("videos")
    assert ids == ["f2"]
    assert err is None
    # Round 2's prompt carried the engine error back to the model.
    assert "failed with" in seen[1]
    assert "no such table" in seen[1]


def test_zero_rows_offers_broadening_and_broadened_query_wins(monkeypatch, db):
    s, seen = _scripted(
        monkeypatch,
        db,
        [
            "SELECT id FROM files WHERE name LIKE '%girlnextdoor%'",
            ("SELECT id FROM files WHERE name LIKE '%girlnextdoor%' OR workflow_prompt LIKE '%girlnextdoor%'"),
        ],
    )
    ids, _sql, err = s.search("girlnextdoor")
    assert ids == ["f1"]
    assert err is None
    assert "0 rows" in seen[1]


def test_repeating_the_same_query_asserts_empty_is_the_answer(monkeypatch, db):
    same = "SELECT id FROM files WHERE name LIKE '%zebra%'"
    s, seen = _scripted(monkeypatch, db, [same, same])
    ids, _sql, err = s.search("zebra")
    assert ids == []
    assert err is None
    assert len(seen) == 2


def test_rounds_exhausted_while_broadening_returns_empty(monkeypatch, db):
    s, _seen = _scripted(
        monkeypatch,
        db,
        [
            "SELECT id FROM files WHERE name LIKE '%zebra%'",
            "SELECT id FROM files WHERE name LIKE '%zebra%' OR type = 'zebra'",
            "SELECT id FROM files WHERE workflow_prompt LIKE '%zebra%'",
        ],
    )
    ids, _sql, err = s.search("zebra")
    assert ids == []
    assert err is None


def test_persistent_errors_fail_closed(monkeypatch, db):
    s, _seen = _scripted(
        monkeypatch,
        db,
        [
            "SELECT nope FROM files",
            "SELECT nope2 FROM files",
            "SELECT nope3 FROM files",
        ],
    )
    ids, _sql, err = s.search("anything")
    assert ids is None
    assert err is not None
    assert "no such column" in err


def test_generation_exception_never_raises(monkeypatch, db):
    s = SqlSearch(db_path=db)

    class _Boom:
        def ask(self, prompt, max_new_tokens=256):
            raise RuntimeError("engine died")

    monkeypatch.setattr(SqlSearch, "_chat", lambda self: _Boom())
    ids, _sql, err = s.search("anything")
    assert ids is None
    assert "generation error" in err
