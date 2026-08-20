"""`ENABLE_AI_SEARCH` queues work that nothing processes.

The flag adds the old AI Search box and AI Manager panel. With it on, the
gallery writes to two queues -- `ai_search_queue` for searches and
`ai_indexing_queue` for files -- and no code in this repository ever
claims a row from either: nothing sets a row to 'processing', nothing
marks one 'completed', and no companion process ships alongside. The AI
layer that replaced it indexes straight from the library and uses neither
queue.

That is recorded in docs/CONFIGURATION.md, and these tests keep the record
true. The grep test is deliberate: if someone gives the queue a consumer,
the documentation becomes wrong, and this is what says so.

The queue also had no bound. A row per file was written on every scan and
only 'completed' rows were ever swept -- a status nothing sets -- so the
table grew with the library and stayed. Stale 'pending' rows are now swept
on the same three-day rule, which is safe because the scan re-queues on
conflict: anything still wanted returns on the next pass.
"""

from __future__ import annotations

import pathlib
import time

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CLAIMS = ("processing", "completed")


def _source_files():
    yield from (
        sorted(_ROOT.glob("*.py"))
        + sorted(_ROOT.glob("smartgallery_ai/*.py"))
        + sorted(_ROOT.glob("omniquery/*.py"))
        + sorted(_ROOT.glob("metaparse/*.py"))
    )


def test_nothing_claims_a_row_from_the_indexing_queue():
    """If this fails, someone implemented the consumer -- which is good
    news, and means docs/CONFIGURATION.md must stop saying the flag is
    inert."""
    setters = []
    for path in _source_files():
        text = pathlib.Path(path).read_text(encoding="utf-8")
        if "ai_indexing_queue" not in text:
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            lowered = line.lower()
            if "ai_indexing_queue" not in lowered:
                continue
            if "update" in lowered and any(f"'{c}'" in lowered for c in _CLAIMS):
                setters.append(f"{path.name}:{line_no}")

    assert setters == [], (
        "something now claims rows from ai_indexing_queue: "
        f"{setters}. Update the 'ENABLE_AI_SEARCH is inert' section of "
        "docs/CONFIGURATION.md, which tells people the flag does nothing."
    )


@pytest.fixture
def legacy_enabled(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "ENABLE_AI_SEARCH", True)
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM ai_indexing_queue")
        conn.commit()
    finally:
        conn.close()
    yield smartgallery_app
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM ai_indexing_queue")
        conn.commit()
    finally:
        conn.close()


def _queue_size(smartgallery_app):
    conn = smartgallery_app.get_db_connection()
    try:
        return conn.execute("SELECT COUNT(*) FROM ai_indexing_queue").fetchone()[0]
    finally:
        conn.close()


def test_stale_queue_rows_are_swept(legacy_enabled):
    """The bound: a pending row older than three days goes, so the table
    does not grow with the library and stay there for ever."""

    conn = legacy_enabled.get_db_connection()
    try:
        old = time.time() - (4 * 86400)
        conn.execute(
            "INSERT INTO ai_indexing_queue (file_path, file_id, status, "
            "created_at, force_index, params) "
            "VALUES ('/x/stale.png', 'stale1', 'pending', ?, 0, '{}')",
            (old,),
        )
        conn.execute(
            "INSERT INTO ai_indexing_queue (file_path, file_id, status, "
            "created_at, force_index, params) "
            "VALUES ('/x/fresh.png', 'fresh1', 'pending', ?, 0, '{}')",
            (time.time(),),
        )
        conn.commit()
    finally:
        conn.close()
    assert _queue_size(legacy_enabled) == 2

    conn = legacy_enabled.get_db_connection()
    try:
        removed = legacy_enabled.sweep_stale_index_queue(conn)
        remaining = [r[0] for r in conn.execute("SELECT file_id FROM ai_indexing_queue").fetchall()]
    finally:
        conn.close()

    assert removed == 1, f"the sweep removed {removed} rows"
    assert remaining == ["fresh1"], f"the stale row survived the three-day sweep: {remaining}"


def test_the_sweep_leaves_a_row_that_is_being_worked_on(legacy_enabled):
    """If a consumer ever exists, its in-flight row must not be swept out
    from under it -- only the two resting states are cleared."""

    conn = legacy_enabled.get_db_connection()
    try:
        old = time.time() - (4 * 86400)
        conn.execute(
            "INSERT INTO ai_indexing_queue (file_path, file_id, status, "
            "created_at, force_index, params) "
            "VALUES ('/x/busy.png', 'busy1', 'processing', ?, 0, '{}')",
            (old,),
        )
        conn.commit()
        legacy_enabled.sweep_stale_index_queue(conn)
        remaining = [r[0] for r in conn.execute("SELECT file_id FROM ai_indexing_queue").fetchall()]
    finally:
        conn.close()

    assert remaining == ["busy1"], "the sweep took a row that was being processed"
