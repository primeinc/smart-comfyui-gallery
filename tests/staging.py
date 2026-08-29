"""One prepared world per module, a millisecond restore per test.

A library on disk, the application over it, and the database after the
module's setup are built ONCE (`staged`, module scope) and snapshotted.
Each test gets the snapshot back -- a file copy, about a millisecond --
instead of a rebuilt application, about 170 ms (measured: build_app 66 ms
on an existing database, app+client+one GET 172 ms, Connection.backup
8 ms, shutil.copy 1.5 ms). The application holds no connection between
requests (sg_web/app.py state: home, db_path, actor_id), so swapping the
file under it is sound; the snapshot is taken through Connection.backup,
which copies the committed state WAL included (cpython
Doc/library/sqlite3.rst:1187-1193).

The library directory is part of the world. The scanner's identity for a
file is (size, mtime, fs_id) (db/scan.py observe_tree), and a filesystem id
cannot be put back by copying -- so a test that deleted or rewrote media
leaves a world that cannot be restored, and the next test gets a rebuilt
one. A test that only read, or only wrote the database, costs nothing.

    @pytest.fixture(scope="module")
    def _world(tmp_path_factory):
        with staged(tmp_path_factory, "name", write_library, setup) as stage:
            yield stage

    @pytest.fixture
    def world(_world):
        _world.restore()
        return _world.client
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import os
import pathlib
import shutil
import sqlite3
import tempfile
import typing
from collections.abc import Callable, Generator
from dataclasses import dataclass, field

from db import connect, when

#: The application and the client that drives it are imported by the two
#: functions that build a world, not here.
#:
#: `tests/conftest.py` imports this module, so every module in the suite
#: pays whatever this line does -- and importing `sg_web.app` and
#: `litestar.testing` is 0.6s of the 0.95s that conftest import costs.
#: Ninety of these modules never stage anything and were paying it to
#: reach a function they do not call. The world builders pay the same
#: total; it simply lands inside the fixture that wanted an application.
#: Every other mention here is an annotation, and
#: `from __future__ import annotations` leaves those as strings.
#:
#: Measured on the wall clock, which is where this lives -- pytest's own
#: "passed in Xs" does not count a conftest import at all, so it showed
#: none of it: test_a_table_column_orders_the_answer 2366ms -> 1537ms,
#: test_metaparse 2428ms -> 1744ms, test_schema_contract 3378ms ->
#: 2573ms. Modules that DO stage are unchanged (3004 -> 3036, 6631 ->
#: 6720); they pay the same import, only inside the fixture that wanted
#: it, where it is at least visible.
if typing.TYPE_CHECKING:
    from litestar.testing import TestClient

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"

#: How often a wait-for-work loop asks again, in seconds.
#:
#: GRANULARITY, not a margin. Every loop that uses it ends the moment the
#: row it is watching is terminal, so the interval is only ever overshoot
#: past work that has ALREADY finished -- and a test that drains several
#: jobs pays it once per job. It was 0.05 in fourteen hand-written copies
#: of the same loop and 0.02 here; at 0.01, measured twice each,
#: `test_a_per_item_failure_shows_its_exact_recorded_error` went 1.15s ->
#: 0.67s/0.72s and `test_the_end_of_the_answer_is_the_end` 1.15s ->
#: 0.77s/0.77s.
#:
#: Not smaller: each turn is a real request against the run being waited
#: on, so the poll competes with the server it is watching. A hundredth
#: is a few percent of one thread; a thousandth would be spinning on it.
POLL = 0.01

#: The suite's fixed clock. It was declared, identically, in 38 test
#: modules; a fixture clock is one fact about the whole suite.
NOW = 1_700_000_000.0
#: The fixture anchor instant, 2023-06-10 00:00 UTC. Six modules spelled
#: it four ways, one of them under the name DAY.
JUNE_10 = 1_686_355_200.0
#: db/when.py's own units, re-exported so a fixture and the code under
#: test cannot mean different lengths of an hour -- eleven modules typed
#: HOUR and five DAY, and three different years coexisted here once.
HOUR = when.HOUR
DAY = when.DAY
_MASTERS: dict[str, sqlite3.Connection] = {}
#: Connections that outlive the test that opened them on purpose -- a
#: module's master built inside a function-scoped fixture. conftest closes
#: every other in-memory connection when its test ends.
#:
#: The CONNECTIONS, not their id()s. See `keep`.
LONG_LIVED: list[sqlite3.Connection] = []


def keep(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Mark a connection as outliving its test; the owner closes it.

    The connection itself is held. This was a set of `id(conn)`, and an
    id is unique only among LIVE objects: a kept master is closed and
    dropped by its owner in the ordinary course of things -- the pages
    fixture rebuilds its master inside a test, and closes it again at
    module teardown -- and the moment that object dies its address is
    free. CPython handed the very next `connect.memory()` the same
    address (measured: reused on the first allocation), conftest read
    that live, unrelated connection as long-lived, and never closed it.
    It reached the collector still open as `ResourceWarning: unclosed
    database`, blamed on whichever test was running by then.

    Holding the object is what makes the mark mean one connection: while
    it stands the object cannot die, so its address cannot be handed to
    anything else.
    """
    LONG_LIVED.append(conn)
    return conn


def is_kept(conn: sqlite3.Connection) -> bool:
    """Whether this exact connection outlives its test -- a schema master
    or something `keep` was given. Compared by identity, never by id()."""
    return any(conn is one for one in (*_MASTERS.values(), *LONG_LIVED))


def fresh_schema(ddl: str | None = None) -> sqlite3.Connection:
    """An in-memory database holding the schema, foreign keys on.

    executescript of the DDL is 11 ms; a backup from a master built once
    per process is 0.5 ms (measured), and ~250 tests start from exactly
    this. A custom `ddl` (the contract's mutation sweep) gets its own
    master, keyed on the text."""
    text = SCHEMA.read_text(encoding="utf-8") if ddl is None else ddl
    master = _MASTERS.get(text)
    if master is None:
        master = connect.memory()
        master.executescript(text)
        _MASTERS[text] = master
    conn = connect.memory()
    master.backup(conn)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@functools.cache
def _built_master() -> pathlib.Path:
    """One built database, on disk, closed -- kept ACROSS processes.

    `build.build` writes the schema, stamps the version and runs
    whatever else a fresh database needs; doing that is 67.7 ms
    (measured) and copying the closed file it produced is 1.6 ms. It was
    a per-process temporary, so every module that stages a world paid
    the 67.7 ms again -- and the suite is read one module at a time.

    Named for what determines it, and nothing else. `build.build` is
    `connect(path)`, `executescript(schema_sql())`, and a check that the
    DDL's own PRAGMAs match this build (db/build.py:130-166), so the
    artifact is a function of `db/schema.sql` and `db/connect.py` --
    the DDL, the pragmas that connection applies, and the USER_VERSION
    and APPLICATION_ID it is checked against. Either file changing is a
    different digest and a fresh build, which is also what re-runs that
    check.

    Under `.pytest_cache` beside the other cross-run masters, so
    `--cache-clear` reaches it.
    """
    from db import build

    stamp = hashlib.sha256()
    here = pathlib.Path(__file__).resolve().parent.parent
    for one in ("db/schema.sql", "db/connect.py"):
        stamp.update((here / one).read_bytes())
    where = here / ".pytest_cache" / "built-master"
    where.mkdir(parents=True, exist_ok=True)
    master = where / f"master-{stamp.hexdigest()[:16]}.db"
    if not master.exists():
        # Built under a name of its own and moved on: `build.build`
        # refuses to overwrite a database holding rows, and a run that
        # dies mid-executescript must not leave a half-written file for
        # the next one to copy as a database.
        building = where / f"{master.stem}.{os.getpid()}.building"
        build.build(building)
        try:
            os.replace(building, master)
        except PermissionError:
            # Four xdist workers are four processes, and this cache is
            # per-process, so all four find no master and all four build
            # one. Windows refuses a rename onto a file another process
            # holds open, which is what the winner's readers are doing;
            # POSIX allows it, so this only ever fails here. The name is
            # a digest of the DDL and the connection, so whoever won
            # built THIS artifact -- take theirs and drop ours. A
            # missing master means the refusal was about something else.
            building.unlink(missing_ok=True)
            if not master.exists():
                raise
    return master


def seeded(home: pathlib.Path) -> pathlib.Path:
    """The home's database put there by COPY, before `build_app` looks.

    `build_app` creates a database it does not find with `connect.create`
    -- `connect(path)` and then executescript of the whole DDL, ~60 ms.
    `db.build.build`, which `_built_master` runs, IS that executescript
    plus a check that the DDL's own stamps match this build: same
    objects, same pragmas, no extra rows, which is why `fresh_db`
    already hands these copies out as real databases. Copying is ~2 ms.

    Worth it from the SECOND world in a process: the master costs one
    `build` to make, so a module that stages one world pays exactly what
    it saves (measured: 0.41s -> 0.42s on a single-world module) and one
    that stages three stops paying twice.

    `build_app` then takes its other branch and calls `migrate.migrate`,
    which reads a current `user_version` and has nothing to do; the
    create branch keeps its coverage from every test that boots an
    application over a genuinely empty home.
    """
    home.mkdir(parents=True, exist_ok=True)
    where = home / "gallery.db"
    shutil.copy(_built_master(), where)
    return where


def fresh_db(path: pathlib.Path) -> sqlite3.Connection:
    """A REAL FILE holding the current schema, foreign keys on.

    For the claims :memory: cannot carry -- anything reading
    `PRAGMA data_version`, taking a second connection, or proving what a
    restore does -- where `fresh_schema` would be the wrong shape rather
    than merely a faster one."""
    shutil.copy(_built_master(), path)
    return connect.connect(str(path))


def settled(client: TestClient, job_id: int, timeout: float = 120.0) -> str:
    """ONE job's terminal state, read off its row. The feed carries every
    job's deltas, so a reader that takes the first terminal state it
    sees may be reading somebody else's; the row cannot be mistaken.

    The deadline is for HANGS, not pacing -- polling returns the moment
    the row settles. 120 because a paused-and-resumed job on a saturated
    runner legitimately outlived 30, three pushes in a row -- and the
    runner is four workers wide, and eight was tried and put this very
    kind of clock over its head."""
    import time

    deadline = time.monotonic() + timeout
    while True:
        state = client.get(f"/jobs/{job_id}").json()["state"]
        if state in ("done", "failed", "cancelled"):
            return state
        assert time.monotonic() < deadline, f"job {job_id} still {state} after {timeout}s"
        time.sleep(POLL)


def _snapshot_dir() -> pathlib.Path:
    repo = pathlib.Path(__file__).resolve().parent.parent
    cache_dir = repo / ".pytest_cache" / "corpus-snapshots"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _scanned(root: pathlib.Path) -> list[tuple[pathlib.Path, tuple[int, int]]]:
    """(path, (size, mtime_ns)) for every regular file under `root`.

    Through `os.scandir`, which on Windows carries the size and the times
    out of the directory read itself -- `rglob` then `stat()` throws that
    away and pays a syscall per file, which over a corpus of thousands is
    most of the cache probe it exists to make cheap."""
    found: list[tuple[pathlib.Path, tuple[int, int]]] = []
    stack = [root]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(pathlib.Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        held = entry.stat()
                        found.append((pathlib.Path(entry.path), (held.st_size, held.st_mtime_ns)))
        except OSError:  # vanished, or never readable
            continue
    return found


@functools.cache
def _corpus_key(corpus: pathlib.Path) -> str:
    """The honest cache key: the corpus listing (name, size, mtime) AND
    the exact bytes of every module of db/, metaparse/ and vision/,
    enumerated through the import system (pkgutil + find_spec, nothing
    executed) -- which is what those packages ARE to any derivation over
    the corpus. Any edit rekeys; a stale constant can never vouch for
    new code.

    Cached per process for the same reason `_code_key` is: walking the
    sourced corpus is ~1 s, no test writes into it, and probing three
    cached constants was costing three walks to save none."""
    import hashlib

    from db import library

    key = hashlib.sha256()
    if corpus.is_file():
        # A single frozen file is a corpus of one.
        held = corpus.stat()
        key.update(f"{corpus.name}|{held.st_size}|{held.st_mtime_ns}\n".encode())
    else:
        for path, held in sorted(_scanned(corpus)):
            # The root marker is the APPLICATION's writing -- registering
            # the corpus as a root re-stamps it on every build, and keying
            # on it made every run see a "changed corpus" and rebuild the
            # 11-second constant forever.
            if path.name == library.MARKER:
                continue
            key.update(f"{path.relative_to(corpus)}|{held[0]}|{held[1]}\n".encode())
    key.update(_code_key().encode())
    return key.hexdigest()[:16]


@functools.cache
def _code_key() -> str:
    """The exact bytes of every module of db/, metaparse/ and vision/,
    hashed -- the code half of every derivation key. Cached per process:
    the code cannot change mid-run, and reading ~85 files per cache
    probe made every probe cost what it was built to save."""
    import hashlib
    import importlib.util
    import pkgutil

    import db as db_package
    import metaparse as metaparse_package
    import vision as vision_package

    key = hashlib.sha256()
    for package in (db_package, metaparse_package, vision_package):
        names = [package.__name__]
        names += [found.name for found in pkgutil.walk_packages(package.__path__, prefix=f"{package.__name__}.")]
        for name in sorted(names):
            spec = importlib.util.find_spec(name)
            if spec is not None and spec.origin is not None and spec.origin != "namespace":
                key.update(name.encode())
                key.update(pathlib.Path(spec.origin).read_bytes())
    return key.hexdigest()


def corpus_measurement(corpus: pathlib.Path, name: str, measure: Callable[[], str]) -> str:
    """A frozen corpus's measured constant, cached across runs.

    The same contract as `corpus_snapshot`, for measurements that are
    text rather than a database: re-measuring an unchanged corpus with
    unchanged readers recomputes a constant, so the JSON `measure()`
    returns is stored under the corpus+code key and handed back until
    either moves."""
    target = _snapshot_dir() / f"{name}-{_corpus_key(corpus)}.json"
    if target.exists():
        print(f"corpus measurement HIT {target.name}")
        return target.read_text(encoding="utf-8")
    strayed = sorted(stale.name for stale in target.parent.glob(f"{name}-*.json"))
    print(f"corpus measurement BUILD {target.name}; replacing {strayed or 'nothing'}")
    for stale in target.parent.glob(f"{name}-*.json"):
        stale.unlink()
    text = measure()
    target.write_text(text, encoding="utf-8")
    return text


def cached_capture(path: pathlib.Path, *, video: bool = False):
    """`capture.read(path)` (or `read_video`), cached across runs keyed
    on the file's identity and the reader packages' exact bytes -- one
    real maker-note parse per (bytes, code), the same contract the reach
    trace and the decode sweep already carry. Binaries ride as hex;
    params come back as the tuples the dataclass holds."""
    import dataclasses
    import json

    from db import capture

    def measure() -> str:
        read = capture.read_video if video else capture.read
        held = dataclasses.asdict(read(path))
        held["binaries"] = [[slot, blob.hex()] for slot, blob in held["binaries"]]
        return json.dumps(held)

    told = json.loads(corpus_measurement(path, f"capture-{'clip-' if video else ''}{path.stem}", measure))
    told["binaries"] = [(slot, bytes.fromhex(blob)) for slot, blob in told["binaries"]]
    told["params"] = [tuple(one) for one in told["params"]]
    return capture.Capture(**told)


def migrated(path: pathlib.Path, target: int | None = None, *, name: str = "step") -> list[int]:
    """`migrate.migrate(path)`, cached across runs.

    Replaying thirty shipped steps over an unchanged fixture with
    unchanged code recomputes a constant, exactly like re-deriving a
    frozen corpus. The key is the database's LOGICAL content -- iterdump,
    the schema and rows as SQL -- plus the exact code of db/, so any
    edit to a step or to what the test staged replays for real. Not the
    file's bytes: sqlite lays the same logic out differently from one
    process to the next, and a key that drifts rebuilds the constant
    forever while reporting nothing. Only a replay that SUCCEEDED is
    cached; a refusal is a behaviour a test asserts live.
    """
    import hashlib
    import json
    import shutil

    from db import migrate

    key = hashlib.sha256()
    src = connect.connect(path, read_only=True)
    try:
        for line in src.iterdump():
            # sqlite_stat* is query history, not the fixture: `PRAGMA
            # optimize` on close writes rows that vary with whatever ran
            # on the connection, and keying on them drifted a fresh key
            # every run -- an unbounded cache with a 0% hit rate.
            if "sqlite_stat" in line:
                continue
            key.update(line.encode())
    finally:
        src.close()
    key.update(str(target).encode())
    key.update(_code_key().encode())
    stem = f"migrated-{name}-{key.hexdigest()[:16]}"
    held_db = _snapshot_dir() / f"{stem}.db"
    held_steps = _snapshot_dir() / f"{stem}.json"
    if held_db.exists() and held_steps.exists():
        print(f"migration snapshot HIT {held_db.name}")
        shutil.copyfile(held_db, path)
        return json.loads(held_steps.read_text(encoding="utf-8"))
    # One snapshot per site: a drifting key would otherwise grow orphans
    # forever, and the count of evictions is the drift made visible.
    for stale in _snapshot_dir().glob(f"migrated-{name}-*"):
        stale.unlink()
    print(f"migration snapshot BUILD {held_db.name}")
    applied = migrate.migrate(path) if target is None else migrate.migrate(path, target=target)
    shutil.copyfile(path, held_db)
    held_steps.write_text(json.dumps(applied), encoding="utf-8")
    return applied


def corpus_snapshot(corpus: pathlib.Path, build: Callable[[pathlib.Path], pathlib.Path]) -> pathlib.Path:
    """A frozen corpus's derived database, built once and reused across
    runs. Ingesting a real corpus is disk-bound -- hashing 9 GB of CR2
    bytes takes seconds however the code is arranged -- but the corpus
    never changes and the derivation is deterministic, so rebuilding it
    every run recomputes a constant.

    The key is honest or the cache is a lie: the corpus listing
    (name, size, mtime) AND the exact bytes of the code that interprets
    them -- every module of db/, metaparse/ and vision/, enumerated
    through the import system (pkgutil + find_spec, nothing executed),
    which is what those packages ARE to the derivation. Any edit
    rebuilds; a stale snapshot can never vouch for new code. Not a
    filesystem sweep and not a git call, because a test that globs
    source or starts a program is SG007/SG006's whole subject; this is
    a cache fingerprint, not an assertion about source. Assertions
    still run every time -- only the constant is cached, under
    .pytest_cache beside the suite's other cross-run state.

    `build(home)` performs the full derivation into `home` and returns
    the database path; the snapshot is a copy of that file.
    """
    import shutil

    cache_dir = _snapshot_dir()
    target = cache_dir / f"{corpus.name}-{_corpus_key(corpus)}.db"
    # Said out loud on every decision: a key that silently drifts makes
    # every run rebuild an 11-second constant while reporting nothing.
    if target.exists():
        print(f"corpus snapshot HIT {target.name}")
        return target
    strayed = sorted(stale.name for stale in cache_dir.glob(f"{corpus.name}-*.db"))
    print(f"corpus snapshot BUILD {target.name}; replacing {strayed or 'nothing'}")
    for stale in cache_dir.glob(f"{corpus.name}-*.db"):
        stale.unlink()
    with tempfile.TemporaryDirectory(prefix="sg-corpus-") as tmp:
        built = build(pathlib.Path(tmp) / "run")
        shutil.copyfile(built, target)
    return target


class Holding:
    """Another writer with SQLite's one write lane, the way a long scan
    has it: a thread takes BEGIN IMMEDIATE and keeps it until released.
    Shared here because two modules prove busy-lane behaviour -- the
    worker's (test_a_busy_writer...) and the console's HTTP seam."""

    def __init__(self, path: pathlib.Path):
        import threading

        self._path = path
        self._held = threading.Event()
        self._release = threading.Event()
        self._failed: Exception | None = None
        self._thread = threading.Thread(target=self._hold, daemon=True)

    def _hold(self) -> None:
        conn = connect.connect(str(self._path))
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO job(kind, state, created_at) VALUES('hash', 'queued', 0)")
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


def _listing(root: pathlib.Path) -> dict[pathlib.Path, tuple[int, int]]:
    """The library as it stands, by size and mtime.

    Through `_scanned`, which reads sizes and times out of the directory
    enumeration rather than paying a `stat()` per file -- the same
    saving, for the same reason, as the corpus probe above. This one is
    paid harder: EVERY test in a staged module walks the library here,
    so a hundred-test module walked it a hundred times.

    Unsorted, because the answer is a dict and nothing reads it in order;
    `sorted()` over the paths was ordering a thing that is then thrown
    away.
    """
    return {path.relative_to(root): held for path, held in _scanned(root)}


@dataclass
class Stage:
    """One module's prepared world: the client over `home`, the media
    under `root`, the database snapshot, and whatever the setup held."""

    client: TestClient
    home: pathlib.Path
    root: pathlib.Path
    db: pathlib.Path
    template: pathlib.Path
    library: dict[pathlib.Path, tuple[int, int]]
    held: dict = field(default_factory=dict)
    #: Whether `restore` empties the thumbnail cache. A module with a
    #: test about first renders needs it gone; one whose tests only read
    #: the steady state pays a re-render per test for nothing.
    wipes_thumbs: bool = True
    rebuilds: int = 0
    _rebuild: Callable[[], Stage] | None = None
    #: An idle reader whose only job is `PRAGMA data_version`: the value
    #: moves when any OTHER connection commits (sqlite/sqlite@HEAD
    #: src/sqlite.h.in:1181-1196), and every write a test makes goes
    #: through the app's connections, never this one. It must never
    #: write -- its own changes are the one thing the pragma omits.
    _monitor: sqlite3.Connection | None = None
    #: The snapshot, held open READ-ONLY as the backup's source. It is
    #: the same bytes for every restore in the module, and opening it
    #: again per test is a second connection's worth of PRAGMAs on the
    #: hot path -- a hundred-test module pays that a hundred times.
    #: Re-opened after a rebuild, which writes a new template.
    _frozen: sqlite3.Connection | None = None
    #: The backup's DESTINATION, held open for the same reason. It is
    #: not in a transaction between restores, so it holds no lock and
    #: the application's own connections write as they always did.
    _into: sqlite3.Connection | None = None
    _seen_version: int | None = None

    def _data_version(self) -> int:
        if self._monitor is None:
            self._monitor = connect.connect(self.db, read_only=True)
        return self._monitor.execute("PRAGMA data_version").fetchone()[0]

    def _from_template(self) -> sqlite3.Connection:
        if self._frozen is None:
            self._frozen = connect.connect(self.template, read_only=True)
        return self._frozen

    def _into_db(self) -> sqlite3.Connection:
        if self._into is None:
            self._into = connect.connect(self.db)
        return self._into

    def close_held(self) -> None:
        """End every connection the Stage keeps across tests.

        Three of them, all for the same reason: opening one costs a
        round of PRAGMAs, and a hundred-test module would pay that a
        hundred times per connection. They belong to the module, so
        this is the module's teardown.
        """
        for held in (self._monitor, self._frozen, self._into):
            if held is not None:
                held.close()
        self._monitor = self._frozen = self._into = None

    def snapshot(self) -> None:
        """Freeze the database and the library's identity as they are."""
        # The held reader is on the template about to be replaced.
        if self._frozen is not None:
            self._frozen.close()
            self._frozen = None
        src = connect.connect(self.db)
        try:
            with contextlib.suppress(FileNotFoundError):
                self.template.unlink()
            dst = connect.connect(self.template, autocommit=True)
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            connect.close(src)
        self.library = _listing(self.root)
        self._seen_version = self._data_version()

    def restore(self) -> None:
        """The snapshot back under the running application. No request is
        in flight between tests, so no connection is open on the file.
        The thumbnail cache is content-keyed and safe to delete whole
        (sg_web/home.py), and a test about first renders needs it gone."""
        thumbs = self.home / "thumbs"
        if self.wipes_thumbs:
            # Through `_scanned`, which already knows which entries are
            # files from the directory read. `rglob` then `is_file()` asks
            # the filesystem again per entry, and this runs before every
            # test in the module -- over a cache that grows as the module
            # renders more of them. It also needs no `exists()` first:
            # scandir on a directory that is not there is the same
            # OSError it already steps over.
            for cached, _held in _scanned(thumbs):
                cached.unlink(missing_ok=True)
        if _listing(self.root) != self.library:
            assert self._rebuild is not None
            self.close_held()
            fresh = self._rebuild()
            self.__dict__.update(fresh.__dict__)
            self.rebuilds += 1
            return
        # A test that committed nothing left the database as the snapshot
        # made it, and `data_version` is the proof: the monitor never
        # writes, so an unchanged value means no other connection did.
        if self._seen_version is not None and self._data_version() == self._seen_version:
            return
        # Through the backup API, not a file copy: a connection a test
        # leaked still holds the -wal open on Windows, and the backup
        # writes the destination through its own pager with such
        # connections present (sqlite3.rst: backup into a live database).
        self._from_template().backup(self._into_db())
        # The backup itself is another connection's write; rebase on it.
        self._seen_version = self._data_version()

    def conn(self) -> sqlite3.Connection:
        return connect.connect(self.db)


def _rebuilt_none(name: str, stage: Stage, allowed: int) -> None:
    """A staged module that rebuilt its world paid for it, silently.

    `restore` compares the library by (size, mtime) and, when it differs,
    throws the whole world away: a fresh application, library, scan and
    setup. That is the right answer -- a file that is gone cannot be put
    back as the same file -- but the 0.3s lands on the SETUP of whichever
    test happens to run next, not on the test that moved the file. It
    reads as an innocent test being slow, which is why three modules
    carried one unnoticed.

    So the count is asserted rather than merely kept. A test that changes
    the library on disk has three honest endings, and the message names
    them: put the file back (`os.utime` restores a stamp and `os.replace`
    a name, neither rewriting bytes, so identity survives), remove what
    it added, or run last so nothing follows it to pay.

    `rebuilds` on `staged`/`hosting` is the fourth ending, for the case
    where none of those is available: a test that REMOVES a file cannot
    put it back -- the identity a rescan reads is not in the bytes -- and
    moving it last is only free when its position is not part of what it
    proves. `test_events_are_grouping_hypotheses` is both: the departure
    test deletes, and it fails when moved, because the tests around it
    are a sequence about one library shrinking. So that module declares
    its one rebuild rather than hiding it.

    A BUDGET, not a count. The declared number is what the module may
    spend, and spending less is not a defect: the rebuild is caused by
    one test in the module, so any run that selects a subset without it
    rebuilds fewer times than the module declares. `prove-push` selects
    exactly that way -- pytest-testmon by measured coverage -- and an
    equality check turned a passing test into a failed push, in
    teardown, naming a library nobody had touched. `-k` and `--lf` reach
    it the same way. What this exists to catch is a rebuild nobody
    declared, and `>` still catches every one of them.
    """
    if stage.rebuilds > allowed:
        raise AssertionError(
            f"the staged world for {name!r} was rebuilt {stage.rebuilds} time(s), over the {allowed} it declares:"
            " a test changed the library on disk and left it changed, so the NEXT test paid a whole fresh"
            " application, library and scan in its setup. Put the library back at the end of that test, move"
            " the test last, or -- if it deletes and its position is load-bearing -- pass `rebuilds=` and say why."
        )


@contextlib.contextmanager
def hosting(tmp_path_factory, name: str, *, worker: bool = False, rebuilds: int = 0) -> Generator[Stage]:
    """A module-scoped application over an EMPTY home: no root, no
    library. For modules whose tests each register their own root over
    their own tmp files -- one boot per module instead of one per test,
    a restore between tests, and `/roots` numbering starts at 1 for
    every test because the snapshot holds none."""
    from litestar.testing import TestClient

    from sg_web.app import build_app

    opened: list[TestClient] = []
    built: list[Stage] = []

    def build() -> Stage:
        base = tmp_path_factory.mktemp(name)
        home = base / "run"
        seeded(home)
        client = TestClient(app=build_app(str(home), worker=worker))
        client.__enter__()
        opened.append(client)
        stage = Stage(
            client=client,
            home=home,
            root=base / "lib",
            db=pathlib.Path(client.app.state.db_path),
            template=base / "template.db",
            library={},
        )
        stage._rebuild = build
        stage.root.mkdir(exist_ok=True)
        stage.snapshot()
        built.append(stage)
        return stage

    first = build()
    try:
        yield first
    finally:
        for stage in built:
            stage.close_held()
        for client in reversed(opened):
            client.__exit__(None, None, None)
        _rebuilt_none(name, first, rebuilds)


@contextlib.contextmanager
def staged(
    tmp_path_factory,
    name: str,
    write_library: Callable[[pathlib.Path], None],
    setup: Callable[[Stage], None] | None = None,
    *,
    worker: bool = False,
    keep_thumbs: bool = False,
    rebuilds: int = 0,
) -> Generator[Stage]:
    """A module-scoped world. `write_library(root)` puts the media on
    disk; `setup(stage)` does the module's once-only preparation through
    the client or a connection; the result is snapshotted and yielded."""
    from litestar.testing import TestClient

    from sg_web.app import build_app

    opened: list[TestClient] = []
    built: list[Stage] = []

    def build() -> Stage:
        base = tmp_path_factory.mktemp(name)
        root = base / "lib"
        root.mkdir()
        write_library(root)
        home = base / "run"
        seeded(home)
        client = TestClient(app=build_app(str(home), worker=worker))
        client.__enter__()
        opened.append(client)
        stage = Stage(
            client=client,
            home=home,
            root=root,
            db=pathlib.Path(client.app.state.db_path),
            template=base / "template.db",
            library={},
            wipes_thumbs=not keep_thumbs,
        )
        stage._rebuild = build
        client.post("/roots", json={"path": str(root)})
        precache = client.post("/roots/1/scan").json()["precache"]
        if worker and precache is not None:
            # The scan queued the thumbnail job; a world snapshotted while
            # it runs is one whose cache files are being written under
            # the next restore's unlink, and whose feed already carries a
            # running job nobody in the module asked for.
            settled(client, precache)
        if setup is not None:
            setup(stage)
        stage.snapshot()
        built.append(stage)
        return stage

    first = build()
    try:
        yield first
    finally:
        for stage in built:
            stage.close_held()
        for client in reversed(opened):
            client.__exit__(None, None, None)
        _rebuilt_none(name, first, rebuilds)
