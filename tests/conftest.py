"""Shared pytest behaviour for the greenfield suite.

Two fixtures and no hooks. One closes every in-memory database a test
opened. The other, `live`, is the browser tests' server: the application
in a subprocess through Litestar's own runner, over a library the test
module writes, with pytest-playwright's `page` pointed at it through
`base_url`. Tests own their databases (in-memory or under tmp_path), the
application under test is `db` + `sg_web` + `vision` + `metaparse`, and
the one environment variable here (`SG_TEST_HOME`) is the harness
telling its own subprocess where the run lives -- the product reads no
environment. No test starts a program by hand (sglint SG006): the server
is Litestar's `run_app`, the checks that need git are `sglint --repo`.
"""

from __future__ import annotations

import os
import pathlib
import sqlite3
from dataclasses import dataclass

import httpx
import pytest

from db import connect
from tests import staging

REPO = pathlib.Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _close_memory_databases(monkeypatch):
    """Every `:memory:` connection opened during a test is closed when the
    test ends, whether or not the test closed it (close is idempotent):
    an open connection reaching the garbage collector is a
    ResourceWarning the lane refuses. The per-process schema masters
    (`staging._MASTERS`) and what `staging.keep` marked outlive tests on
    purpose and are left alone."""
    opened: list[sqlite3.Connection] = []
    real = connect.memory

    def recording():
        conn = real()
        opened.append(conn)
        return conn

    monkeypatch.setattr(connect, "memory", recording)
    yield
    kept = {id(conn) for conn in staging._MASTERS.values()} | staging.LONG_LIVED
    for conn in opened:
        if id(conn) not in kept:
            conn.close()


@dataclass(frozen=True)
class Live:
    """A served run: its address, an httpx client on it, the library on
    disk, and whatever the module's `prepare` returned."""

    url: str
    api: httpx.Client
    root: pathlib.Path
    prepared: object


@pytest.fixture(scope="module")
def live(request, tmp_path_factory):
    """The application served in a subprocess (Litestar's `run_app`) over
    a library this module wrote: the module defines `write_library(root)`
    and may define `prepare(api, root)` for its once-only setup through
    the routes. The real worker runs; nothing is stepped by hand."""
    from litestar.testing.client.subprocess_client import run_app

    tmp = tmp_path_factory.mktemp(request.module.__name__.rsplit(".", 1)[-1])
    root = tmp / "lib"
    root.mkdir()
    request.module.write_library(root)
    before = {key: os.environ.get(key) for key in ("SG_TEST_HOME", "PATH")}
    os.environ["SG_TEST_HOME"] = str(tmp / "run")
    # the `litestar` console script lives beside the interpreter; the lane
    # runs that interpreter by path, so its Scripts dir is not on PATH
    os.environ["PATH"] = str(REPO / ".venv" / "Scripts") + os.pathsep + (before["PATH"] or "")
    try:
        with (
            run_app(workdir=REPO, app="tests.live_app:create_app", capture_output=False) as url,
            httpx.Client(base_url=url, timeout=10) as api,
        ):
            prepare = getattr(request.module, "prepare", None)
            yield Live(url, api, root, prepare(api, root) if prepare else None)
    finally:
        for key, value in before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(scope="module")
def browser_context_args(browser_context_args: dict, live: Live) -> dict:
    """pytest-playwright's `page.goto("/path")` resolves against the served
    run: its `base_url` is the context's (the plugin's own seam for it;
    pytest-base-url's session fixture cannot know a per-module server)."""
    return {**browser_context_args, "base_url": live.url}
