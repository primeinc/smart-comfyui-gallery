"""Opening the gallery database correctly, in one place.

`PRAGMA foreign_keys` is per-connection and OFF by default, so the line at the
top of schema.sql governs nothing at runtime -- it applies only to the
connection that runs the script. The application this one replaces learned
that in production and left the lesson in its schema module (git history).

Every consumer goes through `connect()`, so a forgotten pragma cannot make all
sixty-one foreign keys inert while the test suite stays green.
"""

from __future__ import annotations

import contextlib
import pathlib
import sqlite3
import time

SCHEMA = pathlib.Path(__file__).resolve().parent / "schema.sql"

#: Must equal the `PRAGMA user_version` stamped at the end of schema.sql --
#: pinned by test_the_database_states_its_version, because build.py used to
#: re-stamp the file after running the DDL and that hid a two-version gap.
#:
#: Bumped whenever schema.sql changes in a way a built database must match.
#: A bump is not enough on its own: db/migrate.py needs a step registered for
#: the version being left behind, or an existing database cannot be opened.
#: test_every_version_left_behind_has_a_step_off_it enforces that.
USER_VERSION = 21
#: "SGLY" -- distinguishes our file from any other SQLite database.
APPLICATION_ID = 0x53474C59
#: Page cache per connection, in KiB. See `connect` for what it is worth.
CACHE_KIB = 65_536


class WrongVersion(RuntimeError):
    """The file's schema is not the one this build expects.

    Carries both versions so a caller can decide between migrating forward
    and refusing, rather than only being told no.
    """

    def __init__(self, found: int, expected: int):
        self.found = found
        self.expected = expected
        direction = "older" if found < expected else "newer"
        super().__init__(
            f"database is schema v{found}, this build expects v{expected} "
            f"(the file is {direction}). Run db.migrate to bring it forward."
        )


class NotOurDatabase(RuntimeError):
    """The file is a database, but not this application's."""


def _mode(conn: sqlite3.Connection) -> str:
    return (conn.execute("PRAGMA journal_mode").fetchone()[0] or "").lower()


def _ensure_wal(conn: sqlite3.Connection, *, seconds: float = 5.0) -> None:
    """Put the file in WAL, tolerating everyone else doing the same.

    Converting takes an exclusive lock, and `busy_timeout` does not cover it,
    so several processes opening one library at once all raced and all but one
    died inside `connect` with "database is locked" -- before touching
    anything, on a database that was about to be perfectly fine. Its own wait,
    because SQLite will not do this one for us.

    Asking at all is skipped when the file is already in WAL: setting a mode
    the file already has never reaches the locking path
    (sqlite/sqlite@b09c88c14 src/vdbe.c:8096 guards it on `eNew != eOld`), which
    makes the normal case free as well as safe.
    """
    deadline = time.monotonic() + seconds
    while True:
        if _mode(conn) == "wal":
            return
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
            continue
        if _mode(conn) == "wal":
            return
        if time.monotonic() >= deadline:
            raise sqlite3.OperationalError("could not put the database into WAL mode; something else holds it")
        time.sleep(0.05)


def connect(path, *, read_only: bool = False, cross_thread: bool = False) -> sqlite3.Connection:
    """Open the database with the settings the schema assumes.

    `cross_thread=True` lifts sqlite3's same-thread check for a connection
    that outlives any one request thread -- the caller owns serialising
    access with its own lock, because the check it turned off was the only
    other guard (python/cpython Doc/library/sqlite3.rst check_same_thread:
    "Writing operations may need to be serialized by the user").
    """
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=not cross_thread)
    else:
        if cross_thread:
            # A shared writer would need every consumer to serialise every
            # statement; nothing in this application wants that, and a
            # silently ignored flag would look honoured.
            raise ValueError("cross_thread connections are read-only")
        # IMMEDIATE, not the default DEFERRED. Under legacy transaction
        # control sqlite3 opens a transaction before every INSERT, UPDATE,
        # DELETE and REPLACE, and `isolation_level` chooses which BEGIN it
        # issues (python/cpython@a646c99e Doc/library/sqlite3.rst:2709-2720).
        #
        # DEFERRED takes no lock until the first write inside the
        # transaction, so two writers can both start, both read, and one
        # then fails when it tries to upgrade -- halfway through, holding a
        # library whose files are parked under placeholder names. IMMEDIATE
        # takes the write lock at the start, so the second writer waits out
        # its busy_timeout at the door instead of failing in the middle.
        conn = sqlite3.connect(str(path), isolation_level="IMMEDIATE")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    # Negative N means approximately abs(N*1024) BYTES rather than a page
    # count (sqlite/sqlite@b09c88c14 src/pcache.c:284-288), so this is 64 MiB.
    #
    # The default is 2 MiB, which is not a tuning detail on a library-sized
    # database -- it decides which query plans are viable. Measured on 100k
    # files (89 MB), the people page: 60.5 ms at the 2 MiB default, 33.0 ms
    # at 8 MiB, 5.4 ms at 32 MiB and above. The plan was identical every
    # time; only the cache changed.
    #
    # That mattered more than it looks. With the default cache the analyzed
    # plan measured three times slower than the unanalyzed one, and the
    # conclusion drawn from it -- that ANALYZE hurt this schema -- was wrong.
    # The plan drives from 300 people into 14k random row lookups, which is
    # correct and which a 2 MiB cache cannot hold. Given room, statistics
    # make that page three times FASTER than no statistics.
    conn.execute(f"PRAGMA cache_size=-{CACHE_KIB}")
    if not read_only:
        # journal_mode is a write: setting it on a read-only connection raises,
        # and the mode is a property of the file anyway, not of the connection.
        #
        # Asked only when it would change something. The conversion takes an
        # exclusive lock and `busy_timeout` does not cover it, so several
        # processes opening one library at once raced and all but one died in
        # `connect` itself with "database is locked" -- before doing any work,
        # on a database that was about to be fine. Setting a mode the file is
        # already in is a no-op that never reaches the locking path
        # (sqlite/sqlite@b09c88c14 src/vdbe.c:8096, which guards it on eNew!=eOld).
        _ensure_wal(conn)
        # Set explicitly because the default is a COMPILE-TIME choice, not
        # SQLite's: this Python ships DEFAULT_WAL_SYNCHRONOUS=2, so every
        # commit fsyncs, and a different interpreter would behave differently
        # for reasons nothing in this repo controls.
        #
        # NORMAL is safe here, not merely faster. Under WAL the fsyncs move to
        # checkpoint rather than disappearing: the WAL is synced before its
        # content is written into the database, and the database is synced
        # before the WAL is deleted (sqlite/sqlite@b09c88c14 src/wal.c:2175-2188).
        # So a crashed process cannot corrupt the file; a power loss can cost
        # the last few transactions. For a library whose durable facts are
        # re-derivable from disk, and whose authored rows are written one at a
        # time by a person, that is the right side of the trade -- and it is a
        # trade, which is why it is stated rather than assumed.
        conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def schema_sql() -> str:
    return SCHEMA.read_text(encoding="utf-8")


def check_version(conn: sqlite3.Connection) -> None:
    """Refuse a database this build does not recognise.

    A stale build is indistinguishable from a current one without this, which
    is exactly how one went unnoticed: the file said `content_hash` long after
    the DDL had split it into `content_sha256` and `quoted_hash`.
    """
    app = conn.execute("PRAGMA application_id").fetchone()[0]
    ver = conn.execute("PRAGMA user_version").fetchone()[0]
    if app != APPLICATION_ID:
        raise NotOurDatabase(f"not a gallery database (application_id={app:#x})")
    if ver != USER_VERSION:
        raise WrongVersion(ver, USER_VERSION)


def close(conn: sqlite3.Connection) -> None:
    """Close a connection, letting SQLite refresh what it learned.

    `PRAGMA optimize` on close is the documented shape: in the usual case no
    ANALYZE runs at all, and when one does it is bounded
    (sqlite/sqlite@b09c88c14 src/pragma.c:2465-2473). Without it the planner keeps
    running on whatever statistics existed when the library was smaller.
    """
    # A read-only or already-failing connection must still close. Losing a
    # statistics refresh is not worth raising over during shutdown.
    with contextlib.suppress(sqlite3.Error):
        conn.execute("PRAGMA optimize")
    conn.close()
