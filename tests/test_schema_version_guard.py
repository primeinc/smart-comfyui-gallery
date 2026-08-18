"""The schema version marker, in both directions.

The startup check compared the database's version with the build's using
`!=`, which treats "older" and "newer" as the same thing. An older build
opening a newer database therefore stamped the version DOWN -- erasing the
only record that the newer migrations had already run, so the newer build
would run them again over its own work.

Two installs sharing one gallery folder is how that happens: a container
and a local copy at different versions, or a rollback after an upgrade.
Nothing warns you that the folder is shared, and the message it printed --
"Updating Database Schema Version: 30 -> 27" -- reads like ordinary
progress.

The same function also announced every migration step on a database it had
just created, because they all run with IF NOT EXISTS: a new user's first
start reported six schema updates and a version upgrade, which reads as
"already out of date" rather than "created".

Each case used to start a fresh interpreter and load the whole gallery to
call one function against one database. init_db already takes the
connection to work on, so the cases open their own throwaway file and pass
it in; capsys reads back what the subprocess's stdout used to
(pytest doc/en/how-to/capture-stdout-stderr.rst:112-142).
"""

from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture
def fresh_database(smartgallery_app, tmp_path):
    """A throwaway gallery database, opened the way the gallery opens one."""
    path = tmp_path / "gallery_cache.sqlite"
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


def _version(conn):
    return conn.execute("PRAGMA user_version").fetchone()[0]


def test_a_first_run_does_not_announce_migrations(smartgallery_app, fresh_database, capsys):
    """The regression: six 'Updating Database Schema' lines on a database
    created a moment earlier."""
    smartgallery_app.init_db(fresh_database)

    printed = capsys.readouterr().out
    assert "Updating Database Schema" not in printed, f"a brand new database reported migrations:\n{printed}"


def test_a_first_run_still_builds_the_schema(smartgallery_app, fresh_database):
    """Silence must not mean it did nothing -- that would satisfy the test
    above for ever."""
    smartgallery_app.init_db(fresh_database)

    tables = {row[0] for row in fresh_database.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    assert "files" in tables, f"init_db created no files table: {sorted(tables)}"
    assert _version(fresh_database) == smartgallery_app.DB_SCHEMA_VERSION


def test_a_real_upgrade_still_announces_itself(smartgallery_app, fresh_database, capsys):
    """The counterpart: silence on a fresh database must not mean silence
    on an actual migration, which is the one someone needs to see."""
    smartgallery_app.init_db(fresh_database)
    fresh_database.execute("PRAGMA user_version = 3")
    fresh_database.commit()
    capsys.readouterr()

    smartgallery_app.init_db(fresh_database)

    printed = capsys.readouterr().out
    assert "Updating Database Schema Version: 3 ->" in printed, f"an upgrade from version 3 said nothing:\n{printed}"


def test_a_newer_database_is_not_stamped_backwards(smartgallery_app, fresh_database):
    """The regression that matters: the marker recording that newer
    migrations ran must survive an older build opening the file."""
    smartgallery_app.init_db(fresh_database)
    fresh_database.execute("PRAGMA user_version = 999")
    fresh_database.commit()

    smartgallery_app.init_db(fresh_database)

    assert _version(fresh_database) == 999, "the version marker was rewritten downwards"


def test_a_newer_database_says_so_loudly(smartgallery_app, fresh_database, capsys):
    """A silent refusal to downgrade would leave someone wondering why
    their newer data is missing."""
    smartgallery_app.init_db(fresh_database)
    fresh_database.execute("PRAGMA user_version = 999")
    fresh_database.commit()
    capsys.readouterr()

    smartgallery_app.init_db(fresh_database)

    printed = capsys.readouterr().out
    assert "NEWER SmartGallery" in printed, printed
    assert "999" in printed, printed
    assert str(smartgallery_app.DB_SCHEMA_VERSION) in printed, printed
