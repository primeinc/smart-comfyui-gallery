"""One writer means waiting, and waiting is not a defect.

SQLite has a single write lane per database file, and the walk in db/scan.py
writes nothing so it does not hold that lane for the length of a scan.

The lane can still be held longer than `busy_timeout` on a big enough
library, so what happens WHEN it is remains a contract. Reporting that as
`a worker turn died; the job's lease will be reclaimed` says two false
things: the CLAIM is what failed, so no job was claimed and no lease exists
to reclaim.

The contract asserted here: a busy database is `None` -- the same answer
`run_next` already gives when there is no job -- and the worker's own
idle wait handles it. And only BUSY: a real defect stays loud.
"""

from __future__ import annotations

import logging
import pathlib
import sqlite3

import pytest

from db import connect, jobs, runner
from tests.staging import NOW
from tests.staging import Holding as _Holding

SCHEMA = pathlib.Path(__file__).resolve().parents[1] / "db" / "schema.sql"

#: A kind the schema admits. `job.kind` is a closed vocabulary, so an
#: invented one is refused by a CHECK before any of this can be measured.
KIND = "hash"


@pytest.fixture(scope="module")
def _master(tmp_path_factory):
    """The schema written to disk once: executescript onto a file costs
    ~55ms per run, and every test here starts from exactly this."""
    path = tmp_path_factory.mktemp("busy-master") / "master.db"
    conn = connect.connect(str(path))
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.commit()
    connect.close(conn)
    return path


@pytest.fixture
def library(tmp_path, _master):
    """A real file on disk: two connections cannot contend over
    `:memory:`, where each connection IS its own database. A closed
    database file is a file; the copy is the whole setup."""
    import shutil

    path = tmp_path / "gallery.db"
    shutil.copy(_master, path)
    conn = connect.connect(str(path))
    yield path, conn
    connect.close(conn)


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
    conn.execute("PRAGMA busy_timeout=5")
    with _Holding(path):
        refused = _blocked(conn)
    assert refused is not None, "the write went through, so the lane was never held"
    assert refused.sqlite_errorname == "SQLITE_BUSY", refused.sqlite_errorname
    conn.rollback()


def test_a_busy_database_is_no_turn_rather_than_an_exception(library):
    """The contract, stated: this does not raise out of `run_next`."""
    path, conn = library
    conn.execute("PRAGMA busy_timeout=5")
    with _Holding(path):
        assert runner.run_next(conn, owner="worker-test", now=NOW) is None
    conn.rollback()


def test_it_says_so_at_info_rather_than_as_a_traceback(library, caplog):
    """Quiet, but not silent: somebody watching a long scan should be
    able to find out why the worker is idle."""
    path, conn = library
    conn.execute("PRAGMA busy_timeout=5")
    with caplog.at_level(logging.INFO, logger="db.runner"), _Holding(path):
        assert runner.run_next(conn, owner="worker-test", now=NOW) is None
    conn.rollback()
    said = [one for one in caplog.records if one.name == "db.runner"]
    assert said, "the worker went idle and said nothing about why"
    assert all(one.levelno < logging.WARNING for one in said), [(o.levelname, o.getMessage()) for o in said]
    assert any("busy" in one.getMessage() for one in said), [o.getMessage() for o in said]


def test_the_console_seam_words_busy_as_503_and_nothing_else():
    """The HTTP half of this module's claim, at its seam: the handler
    that turns a busy lane into a 503 with a retry message -- reported
    from a real run, where POST /operations/jobs/events answered a 500
    traceback instead. The handler is pure logic; a whole application
    boot bought this proof nothing but 11.3 seconds. The wired,
    lane-actually-held pass lives where an app already runs
    (test_the_shell_mounts_every_surface)."""
    from litestar.testing import RequestFactory

    from sg_web import operations

    # Litestar's own factory, which is exactly this case -- "logic that
    # expects to receive a request object" (docs/usage/testing.rst:343).
    # Not a stand-in and not a hand-built scope: `HTTPScope` is a dozen
    # keys the handler never reads, and filling them by hand is a fake
    # wearing the real type. No application, no transport either way.
    request = RequestFactory().post(path="/operations/jobs/events")
    answer = operations.busy(request, _real_busy_error())
    assert answer.status_code == 503
    assert "busy" in answer.context["error"]
    # Built by hand, so it carries no sqlite_errorname -- exactly the
    # shape of an OperationalError that is not backpressure.
    defect = operations.busy(request, sqlite3.OperationalError("no such table: job"))
    assert defect.status_code == 500, "an OperationalError that is not backpressure must stay a defect"


def _real_busy_error(tmp=None) -> sqlite3.OperationalError:
    """An OperationalError the C layer stamped SQLITE_BUSY: only SQLite
    itself sets `sqlite_errorname`, so the seam test cannot fake one.
    Two plain connections on one file, no threads, milliseconds."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(pathlib.Path(tmpdir) / "lane.db")
        holder = connect.connect(path)
        prober = connect.connect(path)
        prober.execute("PRAGMA busy_timeout=1")
        try:
            holder.execute("CREATE TABLE t(x)")
            holder.commit()
            holder.execute("BEGIN IMMEDIATE")
            holder.execute("INSERT INTO t VALUES(1)")
            with pytest.raises(sqlite3.OperationalError) as caught:
                prober.execute("BEGIN IMMEDIATE")
            why = caught.value
            assert why.sqlite_errorname == "SQLITE_BUSY", why.sqlite_errorname
            return why
        finally:
            holder.rollback()
            connect.close(prober)
            connect.close(holder)


def test_the_console_router_registers_the_busy_handler():
    """The wiring, pinned structurally: the operations Router hands
    sqlite3.OperationalError to `busy`. Without this, the seam test
    above could pass against a handler nothing ever calls."""
    from sg_web import operations

    assert operations.router.exception_handlers[sqlite3.OperationalError] is operations.busy


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
