"""Shared pytest behaviour for the greenfield suite.

One hook and two fixtures. The hook marks EVERY test slow: `just test`
and `just check` are each held to ten seconds, and there is no test here
that fits -- the cheapest one opens a database, and the rest serve the
application or drive a browser. A fast lane holding a handful of them
would be a lane whose green says nothing about the suite, so the fast
lane holds nothing and `just test-slow` holds the suite.

Of the fixtures, one closes every in-memory database a test
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


def pytest_collection_modifyitems(items):
    """Every collected test is slow, whatever its module says.

    Written here rather than as `pytestmark` in sixty files: "how long
    may a lane take" is one repository-wide policy, and sixty copies of
    it are sixty chances for a new module to be born into the wrong
    lane by saying nothing.
    """
    for item in items:
        item.add_marker(pytest.mark.slow)


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
    for conn in opened:
        # by identity: an id() is unique only among live objects, and a
        # kept connection its owner has already closed leaves its address
        # free for the next one (tests/staging.py keep)
        if not staging.is_kept(conn):
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
            # 50ms poll, not litestar's 1s default: the readiness loop
            # sleeps retry_timeout between probes (litestar
            # subprocess_client.py:54-60), so a ~1.5s boot was rounded
            # up to whole seconds in EVERY browser module's setup.
            run_app(
                workdir=REPO, app="tests.live_app:create_app", capture_output=False, retry_count=600, retry_timeout=0.05
            ) as url,
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


@pytest.fixture(autouse=True)
def _unbroken_unless_excused(request):
    """`unbroken` for every browser test, without each one asking.

    It used to be opt-in, and nine of a hundred browser tests had simply
    not opted -- including four over the timeline, which is the surface
    the backlog says still points at a raster route that 404s. A check
    an author has to remember is a check that measures who remembered.

    Only for tests that drive a browser (`page` among their fixtures);
    everything else must not pay for a served application. A test that
    means to provoke a refusal says so with
    `@pytest.mark.expects_broken`, which is a sentence in the test
    rather than an omission nobody can see.
    """
    if "page" not in request.fixturenames or request.node.get_closest_marker("expects_broken"):
        return
    request.getfixturevalue("unbroken")


@pytest.fixture
def unbroken(page, live: Live):
    """Nothing the browser itself calls broken, for the duration of a test.

    A first-party 404 is not a warning: `media.js` returning one rendered
    a photograph with no viewer behind it -- no zoom, no keys, no walk --
    and every browser test passed, because they watched for 500s and
    uncaught exceptions and a missing script is neither.

    Still requestable by name, because a test that wants to READ what
    the browser reported -- rather than only fail on it -- takes the
    list this yields.
    """
    found: list[str] = []

    def answered(response) -> None:
        if response.url.startswith(live.url) and response.status >= 400:
            found.append(f"{response.status} {response.url}")

    def logged(message) -> None:
        if message.type == "error":
            found.append(f"console.error {message.text}")

    page.on("response", answered)
    page.on("pageerror", lambda error: found.append(f"pageerror {error}"))
    page.on("console", logged)
    yield found
    assert not found, "the browser reported these while the test ran:\n  " + "\n  ".join(found)


@pytest.fixture(scope="module")
def browser_context_args(browser_context_args: dict, live: Live) -> dict:
    """pytest-playwright's `page.goto("/path")` resolves against the served
    run: its `base_url` is the context's (the plugin's own seam for it;
    pytest-base-url's session fixture cannot know a per-module server)."""
    return {**browser_context_args, "base_url": live.url}
