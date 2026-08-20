"""The thumbnail cache has to give space back.

A thumbnail is named for md5(path + str(mtime)), so it belongs to one
version of one file, and nothing ever removed one. Deleting a picture left
its thumbnail behind. So did anything that changed a file's mtime -- an
edit, a sync tool, a metadata strip -- because that is a different name
and the old one is simply never asked for again.

Measured on six pictures, browsed so every thumbnail existed, then three
deleted through the route and one touched and rescanned:

    after browsing 6 pics: 6 cache entries
    after deleting 3     : 6 cache entries   files in library: 3
    after touching 1     : 7 cache entries   files in library: 3

Three files, seven thumbnails, and no way for the count to come down. It
sits in the gallery folder, which is frequently the library's own disk or
a synced one.

The sweep runs after the startup scan, which is the moment the library
rows describe what is actually on disk, and takes the cost of guessing
wrong seriously: an empty result set means nothing is removed, because a
database with no rows reads exactly like one that has not been read yet.
"""

from __future__ import annotations

import ast
import os

import pytest

import smartgallery


def _key(path, mtime):
    return smartgallery.content_digest(path + str(mtime))


@pytest.fixture
def cache(smartgallery_app, tmp_path, monkeypatch):
    directory = tmp_path / "thumbs"
    directory.mkdir()
    monkeypatch.setattr(smartgallery_app, "THUMBNAIL_CACHE_DIR", str(directory))
    return directory


@pytest.fixture
def library(smartgallery_app):
    """Three files the library knows about, cleaned up afterwards."""
    rows = [(f"prune{i:027d}", f"/lib/pic_{i}.png", 1700000000.0 + i) for i in range(3)]
    conn = smartgallery_app.get_db_connection()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO files (id, path, mtime, name, type) VALUES (?,?,?,?,?)",
            [(i, p, m, os.path.basename(p), "image") for i, p, m in rows],
        )
        conn.commit()
    finally:
        conn.close()

    yield rows

    conn = smartgallery_app.get_db_connection()
    try:
        conn.executemany("DELETE FROM files WHERE id = ?", [(r[0],) for r in rows])
        conn.commit()
    finally:
        conn.close()


def _prune(smartgallery_app):
    conn = smartgallery_app.get_db_connection()
    try:
        return smartgallery_app.prune_thumbnail_cache(conn)
    finally:
        conn.close()


def test_touching_a_file_really_does_orphan_its_thumbnail():
    """Control, and the premise the rest of the file rests on.

    Everything below is about removing entries nothing refers to. That is
    only worth doing if such entries appear in the first place, so this
    checks the naming directly: the same file at two mtimes is two names,
    and the gallery only ever asks for the current one.

    Deliberately uses nothing but the naming rule, so it holds against the
    build that had no sweep -- the behavioural checks below cannot, since
    the thing they call did not exist there."""
    path = "/lib/pic.png"

    before = _key(path, 1700000000.0)
    after = _key(path, 1700000600.0)

    assert before != after, (
        "a file's thumbnail name no longer changes with its mtime, so "
        "editing one does not leave the old one behind and there is "
        "nothing here to collect"
    )
    assert len(before) == 32
    assert before.isalnum()


def test_a_thumbnail_whose_file_is_gone_is_removed(smartgallery_app, cache, library):
    """The bug: deleting a picture left its thumbnail for good."""
    live = cache / f"{_key(library[0][1], library[0][2])}.jpeg"
    live.write_bytes(b"live")
    orphan = cache / f"{_key('/lib/deleted.png', 1700000009.0)}.jpeg"
    orphan.write_bytes(b"orphan")

    removed_files, _dirs = _prune(smartgallery_app)

    assert removed_files == 1
    assert not orphan.exists()
    assert live.exists(), "a thumbnail still in use was removed"


def test_the_old_thumbnail_of_a_touched_file_is_removed(smartgallery_app, cache, library):
    """The one that adds up. Same file, same path, different mtime: the
    name changes and the old one is never asked for again."""
    path, mtime = library[0][1], library[0][2]
    current = cache / f"{_key(path, mtime)}.jpeg"
    current.write_bytes(b"current")
    before_the_edit = cache / f"{_key(path, mtime - 500)}.jpeg"
    before_the_edit.write_bytes(b"stale")

    _prune(smartgallery_app)

    assert not before_the_edit.exists()
    assert current.exists()


def test_every_kind_of_cache_entry_is_covered(smartgallery_app, cache, library):
    """Waveforms and storyboard frames are named from the same key and
    were left behind the same way."""
    dead = _key("/lib/deleted.png", 1700000009.0)
    plain = cache / f"{dead}.jpeg"
    plain.write_bytes(b"x")
    wave = cache / f"{dead}_wave.png"
    wave.write_bytes(b"x")
    louder = cache / f"{dead}_wave_1.5.png"
    louder.write_bytes(b"x")
    frames = cache / dead
    frames.mkdir()
    (frames / "frame_001.jpg").write_bytes(b"x")

    removed_files, removed_dirs = _prune(smartgallery_app)

    assert removed_files == 3, removed_files
    assert removed_dirs == 1, removed_dirs
    assert not plain.exists()
    assert not wave.exists()
    assert not louder.exists()
    assert not frames.exists()


def test_a_storyboard_folder_still_in_use_is_kept(smartgallery_app, cache, library):
    """Over-reach guard for the directory branch, which is the one that
    removes more than a single file."""
    keep = cache / _key(library[1][1], library[1][2])
    keep.mkdir()
    (keep / "frame_001.jpg").write_bytes(b"x")

    _prune(smartgallery_app)

    assert (keep / "frame_001.jpg").exists()


def test_a_render_in_flight_is_left_alone(smartgallery_app, cache, library):
    """Over-reach guard: tmp_ files are a thumbnail being written right
    now, by a request that has not finished. They have their own sweep at
    startup, and taking one here would corrupt a live render."""
    dead = _key("/lib/deleted.png", 1700000009.0)
    in_flight = cache / f"tmp_{dead}.jpeg"
    in_flight.write_bytes(b"half")

    _prune(smartgallery_app)

    assert in_flight.exists()


def test_anything_not_named_like_a_thumbnail_is_left_alone(smartgallery_app, cache, library):
    """Over-reach guard. This runs inside a folder in somebody's gallery;
    it may only remove what it can positively identify."""
    stranger = cache / "notes.txt"
    stranger.write_bytes(b"someone's file")
    readme = cache / "README"
    readme.write_bytes(b"x")

    _prune(smartgallery_app)

    assert stranger.exists()
    assert readme.exists()


def test_an_empty_library_removes_nothing(smartgallery_app, cache):
    """The guard that matters most. A database with no rows reads exactly
    like one that has not been read yet, and the cost of getting it wrong
    is every thumbnail in the gallery."""
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("CREATE TEMP VIEW IF NOT EXISTS _unused AS SELECT 1")
    finally:
        conn.close()

    entries = [cache / f"{_key('/lib/a.png', 1.0)}.jpeg", cache / f"{_key('/lib/b.png', 2.0)}.jpeg"]
    for entry in entries:
        entry.write_bytes(b"x")

    class _Empty:
        def execute(self, *_args):
            return self

        def fetchall(self):
            return []

    removed_files, removed_dirs = smartgallery.prune_thumbnail_cache(_Empty())

    assert (removed_files, removed_dirs) == (0, 0)
    assert all(entry.exists() for entry in entries), "an unreadable or unscanned library emptied the whole cache"


def test_a_missing_cache_directory_is_not_an_error(smartgallery_app, tmp_path, monkeypatch, library):
    """The cache folder gets removed by disk cleaners -- there is already
    a fix for that elsewhere. This must not be what fails first."""
    monkeypatch.setattr(smartgallery_app, "THUMBNAIL_CACHE_DIR", str(tmp_path / "not_there"))

    assert _prune(smartgallery_app) == (0, 0)


def test_startup_runs_the_sweep(gallery_tree):
    """It has to run somewhere, and after the scan is the only point where
    the rows describe what is on disk."""

    tree = gallery_tree

    start = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "initialize_gallery"),
        None,
    )
    assert start is not None, "initialize_gallery is gone"

    calls = [node for node in ast.walk(start) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    names = [node.func.id for node in calls]

    assert "prune_thumbnail_cache" in names, (
        "startup never clears thumbnails for files the library no longer has, so the cache only grows"
    )
    assert names.index("full_sync_database") < names.index("prune_thumbnail_cache"), (
        "the sweep runs before the scan, so it would judge the cache "
        "against rows that do not yet describe what is on disk"
    )
