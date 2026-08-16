"""A library that has gone must be announced, not quietly replaced.

sqlite3.connect CREATES the file when it is missing. So losing
gallery_cache.sqlite -- an antivirus quarantine, a sync conflict, someone
tidying a folder with "cache" in its name -- raises nothing. The gallery
builds a fresh schema, rescans, and comes up looking like a new install.

Measured against the shipped code: marker row gone, files 0, and not one
line of output about it. The pictures are all still there, which is what
makes it convincing. Ratings, comments, albums, collections and tags are
not, and the first anyone hears of it is when they go looking for them.

The two cases have to be told apart, because a first run must stay quiet:

  no marker + empty database   a new gallery folder      say nothing
  marker    + empty database   a library that has gone   say so, loudly

The marker sits beside the cache directory rather than inside it, so
whatever removed the database does not remove the evidence that there was
one.

Removing the whole .sqlite_cache directory is the same event with a louder
symptom -- every request answering "unable to open database file" until a
restart -- so it is put back, on the same condition as the thumbnail cache
and the trash: only inside a gallery root that still exists.
"""

from __future__ import annotations

import glob
import os
import shutil

import pytest


@pytest.fixture()
def gallery(smartgallery_app, tmp_path, monkeypatch):
    """A gallery folder of this test's own, so the suite's real one is
    never emptied."""
    root = tmp_path / "gallery"
    cache = root / ".sqlite_cache"
    cache.mkdir(parents=True)
    monkeypatch.setattr(smartgallery_app, "BASE_SMARTGALLERY_PATH", str(root))
    monkeypatch.setattr(smartgallery_app, "SQLITE_CACHE_DIR", str(cache))
    monkeypatch.setattr(smartgallery_app, "DATABASE_FILE",
                        str(cache / "gallery_cache.sqlite"))
    # raising=False so the fixture still builds against a build that has no
    # marker at all. Otherwise the tests below would ERROR here rather than
    # fail on the thing they exist to catch, and would never be shown to
    # detect it.
    monkeypatch.setattr(smartgallery_app, "LIBRARY_MARKER_FILE",
                        str(root / ".smartgallery_library"), raising=False)
    return root, cache


def test_a_first_run_is_not_warned_at_and_leaves_a_marker(smartgallery_app, gallery):
    """Control, and the one that stops this being a nuisance: a new
    gallery folder has an empty database for the ordinary reason."""
    root, _cache = gallery

    warned = smartgallery_app.check_library_continuity(is_new_database=True)

    assert warned is False
    assert os.path.exists(smartgallery_app.LIBRARY_MARKER_FILE), (
        "nothing recorded that a library exists here, so the loss of one "
        "could never be noticed")


def test_an_ordinary_run_says_nothing(smartgallery_app, gallery):
    """A database that is already there is not news either."""
    smartgallery_app.check_library_continuity(is_new_database=True)   # first run

    warned = smartgallery_app.check_library_continuity(is_new_database=False)

    assert warned is False


def test_a_lost_database_is_announced(smartgallery_app, gallery, capsys):
    """The bug: this was the silent case."""
    smartgallery_app.check_library_continuity(is_new_database=True)   # first run
    capsys.readouterr()

    warned = smartgallery_app.check_library_continuity(is_new_database=True)

    assert warned is True
    said = capsys.readouterr().out
    assert "held a library before" in said, said
    assert "Ratings, comments, albums" in said, said
    assert "backup" in said, said


def test_it_survives_a_gallery_folder_it_cannot_write_to(smartgallery_app,
                                                         tmp_path, monkeypatch):
    """It runs inside init_db, so it must never be the reason the gallery
    fails to start."""
    monkeypatch.setattr(smartgallery_app, "LIBRARY_MARKER_FILE",
                        str(tmp_path / "no" / "such" / "place" / ".marker"))

    assert smartgallery_app.check_library_continuity(is_new_database=True) is False


def test_the_database_folder_is_put_back(smartgallery_app, gallery):
    """Removing .sqlite_cache answered every request with "unable to open
    database file" until a restart."""
    _root, cache = gallery
    shutil.rmtree(cache)

    conn = smartgallery_app.get_db_connection()
    try:
        assert os.path.isdir(cache)
    finally:
        conn.close()


def test_no_database_is_started_outside_an_existing_gallery(smartgallery_app,
                                                            tmp_path, monkeypatch):
    """The condition that keeps recovery from being reckless. An unplugged
    drive leaves a writable empty mount point; a library started there is
    invisible the moment the real one comes back."""
    absent = tmp_path / "unplugged"
    monkeypatch.setattr(smartgallery_app, "BASE_SMARTGALLERY_PATH", str(absent))
    monkeypatch.setattr(smartgallery_app, "SQLITE_CACHE_DIR",
                        str(absent / ".sqlite_cache"))
    monkeypatch.setattr(smartgallery_app, "DATABASE_FILE",
                        str(absent / ".sqlite_cache" / "gallery_cache.sqlite"))

    with pytest.raises(Exception):
        smartgallery_app.get_db_connection()

    assert not absent.exists(), "a second, invisible library was started"


def test_the_whole_sequence_end_to_end(smartgallery_app, gallery, capsys):
    """First run, ordinary run, then the database is deleted underneath a
    live gallery -- which is exactly how this happens."""
    _root, cache = gallery

    conn = smartgallery_app.get_db_connection()
    smartgallery_app.init_db(conn)
    conn.close()
    assert "held a library before" not in capsys.readouterr().out

    conn = smartgallery_app.get_db_connection()
    smartgallery_app.init_db(conn)
    conn.close()
    assert "held a library before" not in capsys.readouterr().out

    for path in glob.glob(str(cache / "gallery_cache.sqlite*")):
        os.remove(path)

    conn = smartgallery_app.get_db_connection()
    smartgallery_app.init_db(conn)
    conn.close()

    assert "held a library before" in capsys.readouterr().out
