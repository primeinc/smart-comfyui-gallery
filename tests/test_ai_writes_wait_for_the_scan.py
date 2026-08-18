"""The AI worker and the gallery write to one database and must agree on
how long to wait for each other.

Everything derived by the AI layer lives in the gallery's own SQLite file,
so foreign keys to files(id) cascade. In WAL a reader never waits, but a
writer waits for a writer, and how long it waits is the connection's
timeout. Python's default is five seconds; a scan's bulk insert passes
that without trying.

The connections did not agree. runner.py asked for 30 seconds -- somebody
met this once and fixed it there -- while worker.py and service.py, which
are the ones writing continuously while a scan runs, took the default.
Measured against a write held for eight seconds:

    service.py / worker.py   sqlite3.connect(path)  after 5.6s  FAILED: database is locked
    runner.py                timeout=30             after 2.4s  wrote it

What that costs is not an error anyone sees. The worker's stages sit
inside broad `except Exception` handlers, so a lock is indistinguishable
from a file that could not be processed: indexing quietly skips it. And it
only happens once a library is big enough for the scan to hold the lock
past five seconds, which is to say on the libraries that need indexing
most.

Every connection to the gallery database now comes from schema.connect,
at the gallery's own sixty seconds.
"""

from __future__ import annotations

import ast
import pathlib
import sqlite3
import threading
import time

import pytest

from smartgallery_ai import schema, service

_DEFAULT_MS = 5000  # sqlite3.connect(timeout=5.0), per the Python docs


@pytest.fixture
def gallery_db(tmp_path):
    path = str(tmp_path / "gallery.db")
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("CREATE TABLE files (id TEXT PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()
    return path


def test_the_shared_connection_waits_as_long_as_the_gallery(gallery_db):
    conn = schema.connect(gallery_db)
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == schema.DB_TIMEOUT_SECONDS * 1000
    finally:
        conn.close()


def test_the_default_really_is_five_seconds(gallery_db):
    """Control. The checks here are all 'not the default', which means
    nothing unless the default is what the docs say it is."""
    conn = sqlite3.connect(gallery_db)
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == _DEFAULT_MS
    finally:
        conn.close()


def test_the_service_opens_it_the_same_way(gallery_db):
    """service.py answers the panel while the worker writes; it had the
    default too."""

    class _Config:
        db_path = gallery_db

    conn = service._connect(_Config())
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == schema.DB_TIMEOUT_SECONDS * 1000
        assert conn.row_factory is sqlite3.Row, "rows stopped being name-addressable; every caller reads by column name"
    finally:
        conn.close()


def test_a_write_survives_a_scan_holding_the_lock(gallery_db):
    """The symptom itself, against a real held lock.

    The control runs first and inside the same window: a connection that
    will not wait must fail, or the lock is not held and this is measuring
    nothing.

    The lock is held and released on an Event rather than for a duration.
    Nothing here sleeps: every wait blocks until another thread says so,
    and the assertion that the gallery's timeout clears sqlite3's own 5s
    default is a property, not a race to be sat through."""
    assert schema.DB_TIMEOUT_SECONDS > 5.0, (
        "sqlite3.connect defaults to a 5s busy timeout; the gallery's own "
        f"connect must wait longer, not {schema.DB_TIMEOUT_SECONDS}s"
    )

    holding = threading.Event()
    release = threading.Event()
    done = threading.Event()

    def bulk_write():
        conn = sqlite3.connect(gallery_db, timeout=60)
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO files VALUES ('scanned', 'x')")
        holding.set()
        release.wait(30)
        conn.commit()
        conn.close()
        done.set()

    scan = threading.Thread(target=bulk_write, daemon=True)
    scan.start()
    assert holding.wait(10), "the scan never took the lock"

    # Control, in the same window: a connection that refuses to wait fails.
    impatient = sqlite3.connect(gallery_db, timeout=0)
    with pytest.raises(sqlite3.OperationalError) as refused:
        impatient.execute("INSERT INTO files VALUES ('by_default', 'x')")
        impatient.commit()
    impatient.close()
    assert "locked" in str(refused.value), refused.value

    # The fix, against the same lock, still held. It has to block rather
    # than fail, so it runs on its own thread and must not finish until the
    # lock is released -- which is the whole claim.
    wrote = threading.Event()

    def worker_write():
        conn = schema.connect(gallery_db)
        try:
            conn.execute("INSERT INTO files VALUES ('by_worker', 'x')")
            conn.commit()
            wrote.set()
        finally:
            conn.close()

    writer = threading.Thread(target=worker_write, daemon=True)
    writer.start()
    assert not wrote.wait(0.05), (
        "the worker's write completed while the scan still held the lock; it cannot have waited for anything"
    )

    release.set()
    assert wrote.wait(30), "the worker's write never completed"
    assert done.wait(30), "the scan never released the lock"

    check = sqlite3.connect(gallery_db)
    try:
        rows = {row[0] for row in check.execute("SELECT id FROM files")}
    finally:
        check.close()
    assert rows == {"scanned", "by_worker"}, f"the worker's write did not land: {sorted(rows)}"


def test_an_uncontended_write_is_not_slowed(gallery_db):
    """Over-reach guard: waiting longer must only apply to waiting. With
    nothing holding the lock a write returns at once."""
    conn = schema.connect(gallery_db)
    try:
        started = time.monotonic()
        conn.execute("INSERT INTO files VALUES ('alone', 'x')")
        conn.commit()
        elapsed = time.monotonic() - started
    finally:
        conn.close()

    assert elapsed < 5, f"an uncontended write took {elapsed:.1f}s"


def test_a_caller_that_wants_plain_rows_still_gets_them(gallery_db):
    """Over-reach guard. The vector store built its connection without a
    row factory; routing it through the shared opener must not change
    what its rows are."""
    conn = schema.connect(gallery_db, row_factory=None)
    try:
        assert conn.row_factory is None
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == schema.DB_TIMEOUT_SECONDS * 1000
    finally:
        conn.close()


def test_nothing_in_the_ai_layer_opens_the_database_on_its_own():
    """The sweep that found it. A new connection made the direct way is a
    new place that gives up after five seconds, and it would look exactly
    like the two that did."""
    package = pathlib.Path(schema.__file__).parent

    offenders = []
    for module in sorted(package.rglob("*.py")):
        tree = ast.parse(pathlib.Path(module).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "connect"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "sqlite3"
            ):
                continue
            # schema.connect is the one that is allowed to.
            if module.name == "schema.py":
                continue
            if "timeout" in {kw.arg for kw in node.keywords if kw.arg}:
                continue  # said so explicitly; not silently the default
            offenders.append(f"{module.name}:{node.lineno}")

    assert offenders == [], (
        f"sqlite3.connect with no timeout at {offenders}. Use "
        f"schema.connect, or name a timeout -- the default is five seconds "
        f"and the gallery holds the write lock for longer than that during "
        f"a scan."
    )
