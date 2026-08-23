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
file is (size, mtime, inode) (db/scan.py observe_tree), and an inode
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
import pathlib
import sqlite3
from collections.abc import Callable, Generator
from dataclasses import dataclass, field

from litestar.testing import TestClient

from db import connect
from sg_web.app import build_app

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"
_MASTERS: dict[str, sqlite3.Connection] = {}
#: Connections that outlive the test that opened them on purpose -- a
#: module's master built inside a function-scoped fixture. conftest closes
#: every other in-memory connection when its test ends.
LONG_LIVED: set[int] = set()


def keep(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Mark a connection as outliving its test; the owner closes it."""
    LONG_LIVED.add(id(conn))
    return conn


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


def settled(client: TestClient, job_id: int, timeout: float = 30.0) -> str:
    """ONE job's terminal state, read off its row. The feed carries every
    job's deltas, so a reader that takes the first terminal state it
    sees may be reading somebody else's; the row cannot be mistaken."""
    import time

    deadline = time.monotonic() + timeout
    while True:
        state = client.get(f"/jobs/{job_id}").json()["state"]
        if state in ("done", "failed", "cancelled"):
            return state
        assert time.monotonic() < deadline, f"job {job_id} still {state} after {timeout}s"
        time.sleep(0.02)


def _listing(root: pathlib.Path) -> dict[pathlib.Path, tuple[int, int]]:
    return {
        p.relative_to(root): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.rglob("*")) if p.is_file()
    }


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
    rebuilds: int = 0
    _rebuild: Callable[[], Stage] | None = None

    def snapshot(self) -> None:
        """Freeze the database and the library's identity as they are."""
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

    def restore(self) -> None:
        """The snapshot back under the running application. No request is
        in flight between tests, so no connection is open on the file.
        The thumbnail cache is content-keyed and safe to delete whole
        (sg_web/home.py), and a test about first renders needs it gone."""
        thumbs = self.home / "thumbs"
        if thumbs.exists():
            for cached in thumbs.rglob("*"):
                if cached.is_file():
                    cached.unlink()
        if _listing(self.root) != self.library:
            assert self._rebuild is not None
            fresh = self._rebuild()
            self.__dict__.update(fresh.__dict__)
            self.rebuilds += 1
            return
        # Through the backup API, not a file copy: a connection a test
        # leaked still holds the -wal open on Windows, and the backup
        # writes the destination through its own pager with such
        # connections present (sqlite3.rst: backup into a live database).
        src = connect.connect(self.template, read_only=True)
        try:
            dst = connect.connect(self.db)
            try:
                src.backup(dst)
            finally:
                connect.close(dst)
        finally:
            src.close()

    def conn(self) -> sqlite3.Connection:
        return connect.connect(self.db)


@contextlib.contextmanager
def staged(
    tmp_path_factory,
    name: str,
    write_library: Callable[[pathlib.Path], None],
    setup: Callable[[Stage], None] | None = None,
    *,
    worker: bool = False,
) -> Generator[Stage]:
    """A module-scoped world. `write_library(root)` puts the media on
    disk; `setup(stage)` does the module's once-only preparation through
    the client or a connection; the result is snapshotted and yielded."""
    opened: list[TestClient] = []

    def build() -> Stage:
        base = tmp_path_factory.mktemp(name)
        root = base / "lib"
        root.mkdir()
        write_library(root)
        home = base / "run"
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
        return stage

    try:
        yield build()
    finally:
        for client in reversed(opened):
            client.__exit__(None, None, None)
