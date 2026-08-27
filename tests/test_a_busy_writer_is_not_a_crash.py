"""One writer means waiting, and waiting is not a defect.

SQLite has a single write lane per database file. A scan used to hold it
for its whole duration -- `sha256_of` reads every changed file off the
disk -- so everything else that wanted to write was stuck behind it. That
half is fixed in db/scan.py, where the walk now writes nothing.

The lane can still be held longer than `busy_timeout` on a big enough
library, so what happens WHEN it is remains a contract. It used to be
this, every few seconds, for the length of a scan:

    ERROR - a worker turn died; the job's lease will be reclaimed
    Traceback ...
      db/runner.py in run_next
        claimed = jobs.claim(conn, owner, now, kinds=kinds, gate=gate)
      sqlite3.OperationalError: database is locked

Both halves of that sentence are false. The CLAIM is what failed, so no
job was claimed and no lease exists to reclaim. A log that reports
healthy backpressure as a crash is one people learn to scroll past.

The contract asserted here: a busy database is `None` -- the same answer
`run_next` already gives when there is no job -- and the worker's own
idle wait handles it. And only BUSY: a real defect stays loud.
"""

from __future__ import annotations

import logging
import pathlib
import sqlite3
import threading

import pytest

from db import connect, jobs, runner
from tests.staging import NOW

SCHEMA = pathlib.Path(__file__).resolve().parents[1] / "db" / "schema.sql"

#: A kind the schema admits. `job.kind` is a closed vocabulary, so an
#: invented one is refused by a CHECK before any of this can be measured.
KIND = "hash"


@pytest.fixture
def library(tmp_path):
    """A real file on disk: two connections cannot contend over
    `:memory:`, where each connection IS its own database."""
    path = tmp_path / "gallery.db"
    conn = connect.connect(str(path))
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.commit()
    yield path, conn
    connect.close(conn)


class _Holding:
    """Another writer with the lane, the way a long scan has it."""

    def __init__(self, path: pathlib.Path):
        self._path = path
        self._held = threading.Event()
        self._release = threading.Event()
        self._failed: Exception | None = None
        self._thread = threading.Thread(target=self._hold, daemon=True)

    def _hold(self) -> None:
        conn = connect.connect(str(self._path))
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO job(kind, state, created_at) VALUES(?, 'queued', 0)", (KIND,))
            self._held.set()
            self._release.wait(30)
            conn.rollback()
        except (sqlite3.Error, OSError) as why:  # surfaced by __enter__, never swallowed
            self._failed = why
            self._held.set()
        finally:
            connect.close(conn)

    def __enter__(self):
        self._thread.start()
        assert self._held.wait(10), "the other writer never started"
        if self._failed is not None:
            raise AssertionError(f"the other writer could not take the lane: {self._failed!r}")
        return self

    def __exit__(self, *_):
        self._release.set()
        self._thread.join(10)


def _blocked(conn) -> sqlite3.OperationalError | None:
    """Try to write; hand back the refusal if there is one."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO job(kind, state, created_at) VALUES(?, 'queued', 0)", (KIND,))
    except sqlite3.OperationalError as why:
        return why
    return None


def test_the_lane_really_is_held(library):
    """The control. Everything below is about behaviour under SQLITE_BUSY,
    so a run where nothing was busy would prove nothing at all."""
    path, conn = library
    conn.execute("PRAGMA busy_timeout=200")
    with _Holding(path):
        refused = _blocked(conn)
    assert refused is not None, "the write went through, so the lane was never held"
    assert refused.sqlite_errorname == "SQLITE_BUSY", refused.sqlite_errorname
    conn.rollback()


def test_a_busy_database_is_no_turn_rather_than_an_exception(library):
    """The defect, stated: this used to raise out of `run_next`."""
    path, conn = library
    conn.execute("PRAGMA busy_timeout=200")
    with _Holding(path):
        assert runner.run_next(conn, owner="worker-test", now=NOW) is None
    conn.rollback()


def test_it_says_so_at_info_rather_than_as_a_traceback(library, caplog):
    """Quiet, but not silent: somebody watching a long scan should be
    able to find out why the worker is idle."""
    path, conn = library
    conn.execute("PRAGMA busy_timeout=200")
    with caplog.at_level(logging.INFO, logger="db.runner"), _Holding(path):
        assert runner.run_next(conn, owner="worker-test", now=NOW) is None
    conn.rollback()
    said = [one for one in caplog.records if one.name == "db.runner"]
    assert said, "the worker went idle and said nothing about why"
    assert all(one.levelno < logging.WARNING for one in said), [(o.levelname, o.getMessage()) for o in said]
    assert any("busy" in one.getMessage() for one in said), [o.getMessage() for o in said]


def test_a_console_click_during_a_long_write_is_a_503_not_a_500(tmp_path):
    """The HTTP half of this module's claim. The worker answers a held
    lane with "no turn this pass"; a console form gets the same honesty
    as a 503 with a retry message -- reported from a real run, where
    POST /operations/jobs/events answered a 500 traceback instead."""
    from litestar.testing import TestClient

    from sg_web import home
    from sg_web.app import build_app

    burrow = tmp_path / "home"
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        db_path = home.db_path(home.home(str(burrow)))
        with _Holding(db_path):
            answer = client.post("/operations/jobs/events")
        assert answer.status_code == 503
        assert "busy" in answer.text
        assert "Traceback" not in answer.text
        released = client.post("/operations/jobs/events")
        assert released.status_code == 200, "the lane freed; the same click works"


def test_a_free_database_still_claims_the_job(library):
    """The other half of the contract. A change that answered None
    whenever anything went wrong would pass every test above and stop
    the worker for ever."""
    _path, conn = library
    conn.execute("INSERT INTO job(kind, state, created_at) VALUES(?, 'queued', 0)", (KIND,))
    conn.commit()
    assert jobs.claim(conn, "worker-test", NOW) is not None, "nothing was claimable, so this says nothing"
    conn.rollback()


def test_a_real_failure_is_still_loud(library, monkeypatch):
    """Only BUSY and LOCKED are backpressure. An OperationalError meaning
    something else -- a missing table, a broken statement -- must still
    reach the worker, or the fix has quietly muted real defects."""
    _path, conn = library

    def broken(*_args, **_kwargs):
        raise sqlite3.OperationalError("no such table: job")

    monkeypatch.setattr(jobs, "claim", broken)
    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        runner.run_next(conn, owner="worker-test", now=NOW)


def test_the_busy_names_are_the_ones_sqlite_uses():
    """`BUSY` matches by result-code NAME, not by message, because the
    messages ("database is locked", "database table is locked" --
    sqlite/sqlite src/main.c:1667-1668) are prose a release may reword.
    This pins that the names are real, so a typo cannot make the guard
    silently never fire."""
    assert sorted(runner.BUSY) == ["SQLITE_BUSY", "SQLITE_LOCKED"]
    for name in runner.BUSY:
        assert hasattr(sqlite3, name), f"{name} is not a result code sqlite3 knows"
