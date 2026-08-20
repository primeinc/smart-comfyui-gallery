"""Opening the gallery database correctly, in one place.

`PRAGMA foreign_keys` is per-connection and OFF by default, so the line at the
top of schema.sql governs nothing at runtime -- it applies only to the
connection that runs the script. This repo already learned that once:
smartgallery_ai/schema.py:43 says so in a comment and sets it at :57.

Every consumer goes through `connect()`, so a forgotten pragma cannot make all
sixty-one foreign keys inert while the test suite stays green.
"""

from __future__ import annotations

import pathlib
import sqlite3

SCHEMA = pathlib.Path(__file__).resolve().parent / "schema.sql"

#: Bumped whenever schema.sql changes in a way a built database must match.
USER_VERSION = 1
#: "SGLY" -- distinguishes our file from any other SQLite database.
APPLICATION_ID = 0x53474C59


def connect(path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open the database with the settings the schema assumes."""
    if read_only:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    if not read_only:
        # journal_mode is a write: setting it on a read-only connection raises,
        # and the mode is a property of the file anyway, not of the connection.
        conn.execute("PRAGMA journal_mode=WAL")
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
        raise RuntimeError(f"not a gallery database (application_id={app:#x})")
    if ver != USER_VERSION:
        raise RuntimeError(f"database is schema v{ver}, this build expects v{USER_VERSION}")
