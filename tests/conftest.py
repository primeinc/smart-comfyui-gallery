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

import contextlib
import inspect
import logging
import os
import pathlib
import sqlite3
import threading
import time
import typing
from dataclasses import dataclass

import pytest

from db import connect
from tests import staging
from tests.staging import POLL

# Imported by `live`, the one fixture that opens a client, and not here:
# this file is imported once per module in the suite and httpx is 0.19s
# of that. The only other mention is the annotation on `Live.api`, which
# `from __future__ import annotations` leaves as a string.
if typing.TYPE_CHECKING:
    import httpx


def pytest_collection_modifyitems(session, items):
    """Every collected test is slow, whatever its module says.

    Written here rather than as `pytestmark` in sixty files: "how long
    may a lane take" is one repository-wide policy, and sixty copies of
    it are sixty chances for a new module to be born into the wrong
    lane by saying nothing.

    Collection also answers whether this run will drive a browser at
    all, which is what lets the server pool start booting before
    anything asks for it -- see `_servers`. Starting it HERE instead,
    the earliest the answer exists, was measured and is flat: nothing
    pytest does between collection and the first fixture is long enough
    to hide any of a 1.45s boot behind
    (test_an_answer_can_be_described 5.41s -> 5.31/5.31/5.39s).
    """
    for item in items:
        item.add_marker(pytest.mark.slow)
    # How many modules will ask the pool for a server: one per module,
    # because `live` is module-scoped. Both the first boot and whether a
    # spare is worth starting behind it follow from this.
    session.config.sg_browser_modules = len(
        {item.module for item in items if "page" in item.fixturenames or "live" in item.fixturenames}
    )


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
    home: pathlib.Path
    prepared: object


@dataclass
class _Served:
    """One booted application: the entered `run_app`, its address, its home."""

    held: contextlib.AbstractContextManager
    url: str
    home: pathlib.Path

    def stop(self) -> None:
        self.held.__exit__(None, None, None)


class _Servers:
    """A booted application server, ready BEFORE the module that wants it.

    A boot costs ~1.9s -- an interpreter, uvicorn, litestar and this
    application's forty-five modules -- and the suite boots one per
    browser module. Measured over the whole suite: twenty-three boots,
    ~44s, a fifth of a 224s run spent waiting for a process to become
    able to answer. With a spare: 224.01s -> 206.83s and 200.24s.

    None of that boot belongs to the module that asked for it. The
    application starts on an EMPTY home, and the module's library is
    registered afterwards through `/roots` by its own `prepare` -- so a
    server booted for nobody in particular is the server any module
    wanted, and it can be booted while the PREVIOUS module is still
    running its tests, where the wait costs nothing.

    One spare, never more, so at most two servers are alive at a time and
    the isolation is exactly what it was: one fresh subprocess, one fresh
    home, per module.

    The spare's boot does compete for CPU, and that was briefly blamed
    for the suite's `Execution context was destroyed` failures. It is
    not: the same failure appears without a spare (one run in two),
    because an authored surface reloads itself under a one-shot
    `page.evaluate` (frontend/src/authored.ts:113-121). The reads that
    met it are fixed where they are; this stays.

    `run_app` is driven by hand rather than with `with` because a spare
    outlives the call that booted it. It is still litestar's own runner
    (subprocess_client.py:27) -- no test spawns anything (sglint SG006).
    """

    def __init__(self, repo: pathlib.Path, homes, wanted: int = 0) -> None:
        self._repo = repo
        self._homes = homes
        #: How many modules are still going to ask. A spare is only worth
        #: booting while one of them is coming: a run naming ONE browser
        #: module used to boot a second server the moment it handed the
        #: first over, and that boot -- an interpreter and forty-five
        #: modules -- then competed for CPU with the very tests it would
        #: never serve. Counted at collection (`sg_browser_modules`).
        self._wanted = wanted
        # `run_app` takes no `env=` (subprocess_client.py:45-50), so the
        # child reads SG_TEST_HOME from the parent's environment at spawn.
        # That is process-global, so only one boot may be inside it.
        self._gate = threading.Lock()
        self._spare: _Served | None = None
        self._booting: threading.Thread | None = None
        self._broke: BaseException | None = None

    #: ONE server for the whole session was tried, and it is slower.
    #: A module's library is registered at runtime, so the application
    #: need not belong to the module that asked for it -- but then every
    #: module has to hand its roots back (`DELETE /roots/{id}`), and that
    #: cascade across files, folders and derived rows, plus every later
    #: module querying a database six modules have already churned, costs
    #: more than the boots it saves. Measured over the suite: 62.2s a
    #: server per module, 80.1s one server per worker. Both green.

    def take(self) -> _Served:
        """The warm server, and a fresh one started for whoever is next.

        Only for whoever is NEXT: when this take was the last one the
        collection accounted for, nothing is started behind it.
        """
        self._settle()
        served, self._spare = self._spare, None
        if served is None:
            served = self._boot(self._homes())
        self._wanted -= 1
        if self._wanted > 0:
            self._begin()
        return served

    def _settle(self) -> None:
        """Wait for any boot in flight, and re-raise what it hit.

        Raised rather than retried: a server that would not start is a
        broken harness, and swallowing it here would turn every module
        that follows into a mystery.
        """
        if self._booting is not None:
            self._booting.join()
            self._booting = None
        if self._broke is not None:
            broke, self._broke = self._broke, None
            raise broke

    def _begin(self) -> None:
        # The home is minted HERE, on the calling thread, and handed to
        # the boot. `self._homes` is `tmp_path_factory.mktemp`, which
        # pytest does not promise is thread-safe -- and since this boot
        # now starts at session start it runs alongside the module's own
        # fixtures, which mint their directories from the same factory.
        # Minting inside the thread raced them, and the module errored
        # in `live` about one run in two.
        home = self._homes()

        def boot() -> None:
            try:
                self._spare = self._boot(home)
            except Exception as error:
                # Kept for the next `_settle`, which raises it on the
                # thread that asked for a server -- and said out loud
                # here as well, because a boot that fails in the
                # background is otherwise silent until something happens
                # to ask, and may never be asked at all.
                self._broke = error
                logging.getLogger(__name__).exception("a spare server did not boot")

        self._booting = threading.Thread(target=boot, daemon=True)
        self._booting.start()

    def _boot(self, home: pathlib.Path) -> _Served:
        from litestar.testing.client.subprocess_client import run_app

        with self._gate:
            before = {key: os.environ.get(key) for key in ("SG_TEST_HOME", "PATH")}
            os.environ["SG_TEST_HOME"] = str(home)
            # the `litestar` console script lives beside the interpreter;
            # the lane runs that interpreter by path, so its Scripts dir
            # is not on PATH
            os.environ["PATH"] = str(self._repo / ".venv" / "Scripts") + os.pathsep + (before["PATH"] or "")
            try:
                # 50ms poll, not litestar's 1s default: the readiness loop
                # sleeps retry_timeout between probes (litestar
                # subprocess_client.py:54-60), so a ~1.5s boot was rounded
                # up to whole seconds.
                held = run_app(
                    workdir=self._repo,
                    app="tests.live_app:create_app",
                    capture_output=False,
                    retry_count=600,
                    # `int` upstream, but it is handed straight to
                    # `time.sleep` (subprocess_client.py:60), which takes a
                    # float. The annotation is narrower than the contract.
                    retry_timeout=typing.cast("int", 0.05),
                )
                url = held.__enter__()
            finally:
                for key, value in before.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        return _Served(held, url, home)

    def close(self) -> None:
        self._settle()
        if self._spare is not None:
            self._spare.stop()
            self._spare = None


@pytest.fixture(scope="session", autouse=True)
def _servers(request, tmp_path_factory):
    """The pool, and the spare it leaves warm, for the whole session.

    Autouse so it is built at the START of the session rather than when
    `live` first asks. That is what puts the first boot in front of
    pytest-playwright's `browser`: the two used to run one after the
    other -- launch chromium, THEN wait ~1.45s for a server -- because
    nothing reached this fixture until `browser_context_args` did, by
    which time chromium was already up. Booting here overlaps them.

    Only when the run drives a browser. Ninety of these modules never
    serve anything, and a spare booted for them is a subprocess and
    1.45s of CPU spent on nobody.

    Here, and not earlier. `pytest_configure` runs before collection, so
    starting the boot there would have ~0.6s of conftest and module
    importing to hide behind rather than only chromium's 0.37s. Probed
    by booting unconditionally at configure: the first setup went 2.27s
    -> 2.08/2.09/2.17s and the wall barely moved, because the boot is a
    subprocess compiling and importing the same application while
    collection imports playwright, PIL and numpy -- both want the CPU,
    so overlapping them buys a fraction of the window. Not worth an
    unconditional spare, nor the predicate that would be needed to guess
    at configure time whether this run drives a browser at all.
    """
    repo = pathlib.Path(__file__).resolve().parent.parent
    # THIS WORKER's share, not the run's. `sg_browser_modules` is counted
    # from the full collection, and every xdist worker runs its own
    # session -- so each of them read the whole suite's browser modules as
    # its own backlog and kept a spare booting for modules the scheduler
    # had already given to somebody else. The pool's own docstring says
    # "at most two servers are alive at a time", which is true per worker
    # and false per run: at -n 4 that is eight application subprocesses,
    # each importing forty-five modules, behind four chromiums.
    #
    # `--dist loadscope` hands out whole modules, so the share is the
    # count over the worker count -- rounded up, because the remainder
    # lands on somebody.
    wanted = getattr(request.config, "sg_browser_modules", 0)
    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1") or 1)
    wanted = -(-wanted // workers)
    pool = _Servers(repo, lambda: tmp_path_factory.mktemp("served") / "run", wanted)
    if wanted:
        pool._begin()
    yield pool
    pool.close()


def _prepared(module, api, root: pathlib.Path, home: pathlib.Path):
    """The module's `prepare`, given the home only if it asks for one.

    Nearly every module sets itself up through the routes and wants
    `(api, root)`. The one that reaches past them -- writing faces a real
    detector would have found -- needs the database the run is serving,
    and used to read it from `SG_TEST_HOME`. That worked only because the
    old fixture left the variable set for the whole module; a warm spare
    boots DURING the tests and re-points it while it does, so a process
    global is no longer a way to ask where the run lives.
    """
    prepare = getattr(module, "prepare", None)
    if prepare is None:
        return None
    wants = inspect.signature(prepare).parameters
    return prepare(api, root, home) if len(wants) > 2 else prepare(api, root)


@pytest.fixture(scope="module")
def live(request, tmp_path_factory, _servers):
    """The application served in a subprocess (Litestar's `run_app`) over
    a library this module wrote: the module defines `write_library(root)`
    and may define `prepare(api, root)` for its once-only setup through
    the routes. The real worker runs; nothing is stepped by hand.

    A subprocess ON PURPOSE: served in a thread (uvicorn.Server per its
    own tests), the Windows Proactor transports reach the collector
    between requests and `filterwarnings = error` rightly fails the test
    that happens to be running -- measured here before this comment was
    written, along with a 10s hang in the delayed-arrival test. The
    isolation is what the boot buys, and `_Servers` is how it stops being
    paid for in this fixture's setup.
    """
    import httpx

    tmp = tmp_path_factory.mktemp(request.module.__name__.rsplit(".", 1)[-1])
    root = tmp / "lib"
    root.mkdir()
    request.module.write_library(root)
    served = _servers.take()
    try:
        with httpx.Client(base_url=served.url, timeout=10) as api:
            prepared = _prepared(request.module, api, root, served.home)
            _settled(api)
            yield Live(served.url, api, root, served.home, prepared)
    finally:
        served.stop()


def _settled(api, timeout: float = 120.0) -> None:
    """Nothing still queued or running before the module's first test.

    An authored surface settles by asking `/g/locate/{slug}`, and reloads
    itself when the answer has moved since the server drew the page
    (frontend/src/authored.ts:124-127). Background derivation moves it --
    so a module whose setup returns while its own work is still draining
    hands its first test a page that is about to be replaced, and any
    one-shot read of that page (`page.evaluate`, `inner_text`,
    `get_attribute`) dies with "Execution context was destroyed".

    Three modules hit that on three separate runs -- browsing, the filter
    surface, the answer description, the filmstrip -- and each looked
    like its own flake. They are one cause, so this is one wait, here,
    rather than a `_drained` helper each module has to remember to call:
    a check an author has to remember is a check that measures who
    remembered.

    Costs nothing on a module with no jobs, and on the others it waits
    for work that would otherwise have run DURING the tests, competing
    with them.
    """
    deadline = time.monotonic() + timeout
    while True:
        running = [one["id"] for one in api.get("/jobs").json() if one["state"] in ("queued", "running")]
        if not running:
            return
        assert time.monotonic() < deadline, f"jobs still running: {running}"
        time.sleep(POLL)


#: Where a failure's own words are kept, so they survive the terminal.
#:
#: `--full-trace` prints twenty to thirty kilobytes of pytest's internal
#: frames for a single failure, and a console that truncates keeps the
#: ends and drops the middle -- which is exactly where the assertion and
#: its message are. The run says FAILED and refuses to say why.
#:
#: The report object already holds the text. Writing it to a file costs
#: nothing on a green run, changes no flag, and means the reason for a
#: failure is readable afterwards instead of being a thing you had to be
#: watching to catch.
FAILURES = pathlib.Path(__file__).resolve().parent.parent / ".pytest_cache" / "failures.txt"


def pytest_sessionstart(session) -> None:
    """Start each run with an empty ledger, so what is in it is THIS run."""
    with contextlib.suppress(OSError):
        FAILURES.parent.mkdir(parents=True, exist_ok=True)
        FAILURES.unlink(missing_ok=True)


def pytest_runtest_logreport(report) -> None:
    if not report.failed:
        return
    with contextlib.suppress(OSError):
        FAILURES.parent.mkdir(parents=True, exist_ok=True)
        # newline="" so the file keeps the LF this repository stores
        # (.gitattributes eol=lf) rather than being rewritten to CRLF.
        with FAILURES.open("a", encoding="utf-8", newline="") as ledger:
            ledger.write(f"\n=== {report.nodeid} ({report.when}) ===\n{report.longreprtext}\n")


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

    def crashed(error) -> None:
        found.append(f"pageerror {error}")

    page.on("response", answered)
    page.on("pageerror", crashed)
    page.on("console", logged)
    try:
        yield found
    finally:
        # Taken off again because the page OUTLIVES the test (`page`
        # below). Left attached, every test's handlers would still be
        # listening during the next one -- the lists grow without bound
        # and each test pays a callback per handler for every response
        # the whole module makes.
        for event, handler in (("response", answered), ("pageerror", crashed), ("console", logged)):
            with contextlib.suppress(Exception):
                page.remove_listener(event, handler)
    assert not found, "the browser reported these while the test ran:\n  " + "\n  ".join(found)


@pytest.fixture(scope="module")
def browser_context_args(browser_context_args: dict, live: Live) -> dict:
    """pytest-playwright's `page.goto("/path")` resolves against the served
    run: its `base_url` is the context's (the plugin's own seam for it;
    pytest-base-url's session fixture cannot know a per-module server)."""
    return {**browser_context_args, "base_url": live.url}


@pytest.fixture(scope="module")
def _shared_context(browser, browser_context_args: dict):
    """ONE browser context per module, so the cache is warm.

    pytest-playwright builds a context per TEST
    (pytest_playwright/pytest_playwright.py:414), and a context carries
    its own HTTP cache. Every browser test therefore fetched and parsed
    the same bundle, stylesheet and pictures from cold -- sixty times
    over in the viewer's module alone.

    What a test needs from the one before it is a fresh DOM and fresh JS
    globals, and that is a fresh PAGE. It was being bought with a fresh
    browser profile, which is the cache as well.
    """
    context = browser.new_context(**browser_context_args)
    yield context
    context.close()


@pytest.fixture(scope="module")
def _module_page(_shared_context):
    """ONE page per module, for the same reason there is one context.

    A page is not free: opening and closing one is ~40 ms of the setup
    every browser test pays, in front of assertions that answer in a
    tenth. Measured across the suite's browser modules, that is seconds
    spent building a tab to look at the same server through.

    What a test actually needs from the one before it is a fresh DOM and
    fresh JS globals -- and a `goto` gives both, because a navigation
    replaces the document and its realm. The page is the tab; the
    document is what the test is about.
    """
    page = _shared_context.new_page()
    yield page
    with contextlib.suppress(Exception):
        page.evaluate("() => { localStorage.clear(); sessionStorage.clear(); }")
    page.close()


@pytest.fixture
def page(_module_page):
    """The module's page, carrying nothing over.

    Storage belongs to the CONTEXT, so it survives a navigation -- and it
    really does leak: `showInspector` calls `remember()` on every press
    of `i` (frontend/src/viewer.ts), so a test that opens the inspector
    would leave it open for whatever runs next.

    Cleared on the way IN, which a shared page can do and a fresh one
    could not: the page is already sitting on the origin whose storage
    needs emptying. Suppressed for the module's first test, where there
    is no origin yet and so nothing to clear.

    Routes come off for the same reason. A `page.route` outlives the
    test that installed it, so the module that fulfils every thumbnail
    with a 404 to prove a broken picture says what it is went on
    answering 404 for the test after it -- which is about a picture that
    LOADS, and which then failed on the very 404s the test before it
    asked for.
    """
    with contextlib.suppress(Exception):
        _module_page.evaluate("() => { localStorage.clear(); sessionStorage.clear(); }")
    with contextlib.suppress(Exception):
        _module_page.unroute_all()
    # Dialog handlers are NOT cleared here, though they outlive their
    # test the way a route does. Removing them was tried and did not fix
    # the module it was written for
    # (`test_a_keyword_vocabulary_can_be_kept_honest`, which keeps a page
    # of its own and says why), so it is not kept: a fix that does not
    # fix anything is a line the next person has to disprove again.
    # A blank document LAST, which is what a fresh page was really being
    # bought for: a navigation replaces the document and its JS realm, so
    # the next test starts with no DOM, no listeners and no timers of the
    # last one's -- an open confirmation, a mounted widget, a pending
    # poll. Storage is cleared before it, while there is still an origin
    # to clear it on.
    with contextlib.suppress(Exception):
        _module_page.goto("about:blank")
    return _module_page
