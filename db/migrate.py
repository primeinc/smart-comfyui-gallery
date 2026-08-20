"""Moving a database forward without destroying what a person put in it.

`check_version` refuses a database this build does not recognise, and until
now the only way past that refusal was `db.build --force`, which deletes every
rating, comment, album and name in the file. That is a data-loss upgrade path
wearing a version check.

Forward only. Rolling a schema change back is the case the Well-Architected
guidance singles out as hardest -- "Rolling back changes, especially database,
schema, or other stateful component changes, can be complex. Your SDP
guidelines should provide clear instructions on how to deal with data changes"
(refs/MicrosoftDocs/well-architected/well-architected/operational-excellence/
safe-deployments.md:67). This is that instruction: there is no down step, a
downgrade is refused by name, and the way back is the backup taken on the way
out (:108, "Preserve state before removal. Take a snapshot, export, or backup")
which is named for the version it holds (:71, versioning across artifacts).

Three things about the driver decide the shape of the runner, all of them
from Doc/library/sqlite3.rst in the CPython tree:

`isolation_level=None` is the only mode where nothing is implicitly opened:
"If isolation_level is set to None, no transactions are implicitly opened at
all ... but also allows the user to perform their own transaction handling
using explicit SQL statements" (:2722-2728). The runner needs that, because a
pragma it depends on does not apply inside a transaction.

`PRAGMA foreign_keys` inside a transaction is a silent no-op -- no error, it
simply does not take. Verified on SQLite 3.47.1 rather than assumed. So the
pragma is set before BEGIN, never within, and the default connection would
have opened one already: under the legacy default "a transaction is implicitly
opened before executing sql" for any INSERT/UPDATE/DELETE/REPLACE (:1504-1509).

`executescript` must never appear in a migration step. It "implicitly commits
any pending transaction before execution of the given SQL script, regardless
of the value of isolation_level" (:2730-2732) -- so one call inside a step
would commit the half-finished migration and take atomicity with it.
"""

from __future__ import annotations

import pathlib
import shutil
import sqlite3
from collections.abc import Callable

from .connect import APPLICATION_ID, USER_VERSION

#: from_version -> the step that takes a database to from_version + 1.
STEPS: dict[int, Callable[[sqlite3.Connection], None]] = {}


class Downgrade(Exception):
    """The file is newer than this build. There is no way back."""


class StepMissing(Exception):
    """No migration exists for a version this build is supposed to reach."""


class NotOurDatabase(Exception):
    """The file is not a gallery database."""


def step(from_version: int):
    """Register the step that moves a database off `from_version`.

    Steps are numbered by where they start, not where they end, so a gap is
    obvious: the runner asks for `STEPS[3]` and either has it or stops.
    """

    def register(fn):
        if from_version in STEPS:
            raise ValueError(f"two migrations claim v{from_version}")
        STEPS[from_version] = fn
        return fn

    return register


def version_of(conn: sqlite3.Connection) -> int:
    app = conn.execute("PRAGMA application_id").fetchone()[0]
    if app != APPLICATION_ID:
        raise NotOurDatabase(f"application_id={app:#x}, not a gallery database")
    return conn.execute("PRAGMA user_version").fetchone()[0]


def pending(path) -> list[int]:
    """The versions a migration would pass through, without touching anything."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        current = version_of(conn)
    finally:
        conn.close()
    if current > USER_VERSION:
        raise Downgrade(f"database is v{current}, this build is v{USER_VERSION}")
    return list(range(current + 1, USER_VERSION + 1))


def snapshot(path, version: int) -> pathlib.Path:
    """Copy the database aside before changing it.

    `Connection.backup` rather than a file copy: it "Works even if the
    database is being accessed by other clients or concurrently by the same
    connection" (sqlite3.rst:1130-1135), where copying the file while a WAL
    is live captures a torn database that looks fine until it is opened.

    Named for the version it holds, so the file itself says what it can be
    restored into.
    """
    path = pathlib.Path(path)
    target = path.with_suffix(f".v{version}.backup")
    if target.exists():
        target.unlink()
    source = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    destination = sqlite3.connect(str(target))
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    return target


def migrate(path, *, target: int = USER_VERSION, take_snapshot: bool = True) -> list[int]:
    """Bring a database up to `target`, one version at a time.

    Returns the versions applied. Each step is its own transaction, so a
    failure leaves the file at the last version that completed rather than
    somewhere between two.
    """
    path = pathlib.Path(path)
    # isolation_level=None: explicit transaction control, and pragmas that
    # actually apply. See the module docstring.
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        current = version_of(conn)
        if current > target:
            raise Downgrade(
                f"database is v{current}, this build is v{target}. Restore the "
                f"backup taken when it was upgraded; there is no down step."
            )
        if current == target:
            return []

        missing = [v for v in range(current, target) if v not in STEPS]
        if missing:
            raise StepMissing(
                f"no migration from v{missing[0]}: this build cannot open a "
                f"v{current} database without one"
            )

        if take_snapshot:
            snapshot(path, current)

        conn.execute("PRAGMA busy_timeout=5000")
        # Before BEGIN, never inside: the pragma is silently ignored within a
        # transaction. A step that rebuilds a table needs the keys off, and
        # `foreign_key_check` below is what proves nothing was left dangling.
        conn.execute("PRAGMA foreign_keys=OFF")

        applied: list[int] = []
        try:
            for version in range(current, target):
                conn.execute("BEGIN IMMEDIATE")
                try:
                    STEPS[version](conn)
                    conn.execute(f"PRAGMA user_version = {version + 1}")
                    broken = conn.execute("PRAGMA foreign_key_check").fetchall()
                    if broken:
                        raise sqlite3.IntegrityError(
                            f"v{version} -> v{version + 1} left "
                            f"{len(broken)} dangling references: {broken[:5]}"
                        )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
                applied.append(version + 1)
        finally:
            conn.execute("PRAGMA foreign_keys=ON")
        return applied
    finally:
        conn.close()


def restore(backup, path) -> None:
    """Put a snapshot back. The only way out of a failed upgrade."""
    path = pathlib.Path(path)
    for suffix in ("-wal", "-shm"):
        pathlib.Path(str(path) + suffix).unlink(missing_ok=True)
    shutil.copyfile(backup, path)


def optimize(conn: sqlite3.Connection) -> None:
    """Let SQLite refresh the statistics the planner runs on.

    A table is analyzed only when an index has no `sqlite_stat1` entry, when
    the planner used those statistics on this connection, or when the table
    grew or shrank tenfold
    (refs/sqlite/sqlite/src/pragma.c:2485-2510). Cheap in the normal case --
    the usual outcome is that no ANALYZE runs at all.

    Worth calling when a connection closes and after a scan, because a
    library that has just gone from empty to 100k files has statistics that
    describe a database that no longer exists.
    """
    conn.execute("PRAGMA optimize")


def analyze(conn: sqlite3.Connection) -> None:
    """Force the full pass over every table, not only the queried ones.

    `PRAGMA optimize` alone will not look at a table the connection never
    queried; the 0x10000 bit is what makes it consider every table
    ("Look at tables to see if they need to be reanalyzed due to growth or
    shrinkage even if they have not been queried during the current
    connection. Off by default." -- pragma.c:2475-2478).

    Worth running after a first scan, when a library has just gone from empty
    to a hundred thousand files and the planner's statistics describe a
    database that no longer exists.

    Measured on 100k files, the people page: 17.7 ms with no statistics,
    5.4 ms with them. The plan changes from scanning `file` to driving from
    300 people through an index -- which is right, and which needs a page
    cache big enough to hold the random lookups it implies. At SQLite's 2 MiB
    default the same analyzed plan measures 60 ms, and the first reading of
    that number here was that ANALYZE had made things worse. It had not; see
    `connect.CACHE_KIB`. A plan is only as good as the cache under it, and a
    benchmark that leaves the cache at its default is measuring the default.
    """
    conn.execute("PRAGMA optimize=0x10012")
