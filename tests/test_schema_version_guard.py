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
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(env_extra, body, tmp_path):
    gallery = tmp_path / "gallery"
    output = tmp_path / "output"
    gallery.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    env = dict(os.environ, ENABLE_AI_DAM="false", AI_DAM_AUTO_PROVISION="false",
               BASE_OUTPUT_PATH=str(output), BASE_SMARTGALLERY_PATH=str(gallery),
               **env_extra)
    script = ("import sys, os\n"
              "sys.argv = ['smartgallery.py']\n"
              "import smartgallery as sg\n"
              "os.makedirs(sg.SQLITE_CACHE_DIR, exist_ok=True)\n" + body)
    return subprocess.run([sys.executable, "-c", script], cwd=_ROOT, env=env,
                          capture_output=True, text=True, timeout=300)


def test_a_first_run_does_not_announce_migrations(tmp_path):
    """The regression: six 'Updating Database Schema' lines on a database
    created a moment earlier."""
    proc = _run({}, "sg.init_db()\nprint('DONE')\n", tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "DONE" in proc.stdout
    assert "Updating Database Schema" not in proc.stdout, (
        f"a brand new database reported migrations:\n{proc.stdout}")


def test_a_real_upgrade_still_announces_itself(tmp_path):
    """The counterpart: silence on a fresh database must not mean silence
    on an actual migration, which is the one someone needs to see."""
    body = ("sg.init_db()\n"
            "conn = sg.get_db_connection()\n"
            "conn.execute('PRAGMA user_version = 3')\n"
            "conn.commit(); conn.close()\n"
            "sg.init_db()\n"
            "print('DONE')\n")
    proc = _run({}, body, tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "Updating Database Schema Version: 3 ->" in proc.stdout, (
        f"an upgrade from version 3 said nothing:\n{proc.stdout}")


def test_a_newer_database_is_not_stamped_backwards(tmp_path):
    """The regression that matters: the marker recording that newer
    migrations ran must survive an older build opening the file."""
    body = ("sg.init_db()\n"
            "conn = sg.get_db_connection()\n"
            "conn.execute('PRAGMA user_version = 999')\n"
            "conn.commit(); conn.close()\n"
            "sg.init_db()\n"
            "conn = sg.get_db_connection()\n"
            "print('VERSION=', conn.execute('PRAGMA user_version').fetchone()[0])\n")
    proc = _run({}, body, tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "VERSION= 999" in proc.stdout, (
        f"the version marker was rewritten downwards:\n{proc.stdout}")


def test_a_newer_database_says_so_loudly(tmp_path):
    """A silent refusal to downgrade would leave someone wondering why
    their newer data is missing."""
    body = ("sg.init_db()\n"
            "conn = sg.get_db_connection()\n"
            "conn.execute('PRAGMA user_version = 999')\n"
            "conn.commit(); conn.close()\n"
            "sg.init_db()\n"
            "print('DONE')\n")
    proc = _run({}, body, tmp_path)

    out = proc.stdout
    assert "NEWER SmartGallery" in out, out
    assert "999" in out and str(_current_build_version()) in out, out


def _current_build_version():
    src = open(os.path.join(_ROOT, "smartgallery.py"), encoding="utf-8").read()
    for line in src.splitlines():
        if line.startswith("DB_SCHEMA_VERSION"):
            return int(line.split("=")[1].strip())
    raise AssertionError("DB_SCHEMA_VERSION not found")
