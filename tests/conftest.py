"""Shared pytest behaviour for the greenfield suite.

No hooks; one autouse fixture that closes every in-memory database a
test opened. Tests own their databases (in-memory or under tmp_path), the
application under test is `db` + `sg_web` + `vision` + `metaparse`, and
nothing here points environment variables at anything -- a suite that
needs its environment arranged before import is a suite whose subject
reads configuration at import time, and that defect died with the
application that had it. No test starts a program (sglint SG006): the
checks that need one (git, a checkout) are `python -m sglint --repo`.
"""

import sqlite3

import pytest

from tests import staging


@pytest.fixture(autouse=True)
def _close_memory_databases(monkeypatch):
    """Every `:memory:` connection opened during a test is closed when the
    test ends, whether or not the test closed it (close is idempotent):
    an open connection reaching the garbage collector is a
    ResourceWarning the lane refuses. The per-process schema masters
    (`staging._MASTERS`) and what `staging.keep` marked outlive tests on
    purpose and are left alone."""
    opened: list[sqlite3.Connection] = []
    real = sqlite3.connect

    def recording(database, *args, **kwargs):
        conn = real(database, *args, **kwargs)
        if database == ":memory:":
            opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", recording)
    yield
    kept = {id(conn) for conn in staging._MASTERS.values()} | staging.LONG_LIVED
    for conn in opened:
        if id(conn) not in kept:
            conn.close()
