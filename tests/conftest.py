"""Shared pytest behaviour for the greenfield suite.

No hooks; one autouse fixture that closes what `staging.fresh_schema`
handed out. Tests own their databases (in-memory or under tmp_path), the
application under test is `db` + `sg_web` + `vision` + `metaparse`, and
nothing here points environment variables at anything -- a suite that
needs its environment arranged before import is a suite whose subject
reads configuration at import time, and that defect died with the
application that had it. No test starts a program (sglint SG006): the
checks that need one (git, a checkout) are `python -m sglint --repo`.
"""

import pytest

from tests import staging


@pytest.fixture(autouse=True)
def _close_fresh_databases():
    """A test's in-memory database is closed when the test ends, whether
    or not the test closed it (close is idempotent): an open connection
    reaching the garbage collector is a ResourceWarning the lane refuses."""
    yield
    while staging.OPENED:
        staging.OPENED.pop().close()
