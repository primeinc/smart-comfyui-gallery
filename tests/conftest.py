"""Shared pytest behaviour for the greenfield suite.

One hook. Tests own their databases (in-memory or under tmp_path), the
application under test is `db` + `sg_web` + `vision` + `metaparse`, and
nothing here points environment variables at anything -- a suite that
needs its environment arranged before import is a suite whose subject
reads configuration at import time, and that defect died with the
application that had it.
"""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    """Hold back the tests that start a child process.

    A handful of checks can only be answered by another program -- is this
    committed file CRLF (git), does the repository's policy hold on every
    checkout. They are real checks and they stay in the tree, but a process
    start costs more than everything they assert, and a suite nobody runs
    catches nothing.

    So a plain `pytest` starts no child at all, and `RUN_TOOL_TESTS=1`
    runs them. Marked rather than deleted, so `-m spawns` still lists
    exactly which checks are being traded away.
    """
    if os.environ.get("RUN_TOOL_TESTS") == "1":
        return

    held_back = pytest.mark.skip(reason="starts a child process; run with RUN_TOOL_TESTS=1")
    for item in items:
        if "spawns" in item.keywords:
            item.add_marker(held_back)
