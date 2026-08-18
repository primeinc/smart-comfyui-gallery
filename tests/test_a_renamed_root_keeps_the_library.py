"""Reaching the same pictures by another name must not empty the library.

A file is identified by its path, and the scan compares paths as strings.
Point the gallery at the same folder spelled differently and every row is
missing while every picture is new -- so the scan deletes all the rows,
which cascades the ratings, comments, album membership and tags away, and
indexes the same pictures again as if they had just arrived.

Measured on three rated pictures, restarting with only the case of
BASE_OUTPUT_PATH changed, and then changed back:

    forward  seed   3 files, 3 favourites, 3 ratings
    forward  check  3 files, 3 favourites, 3 ratings
    upper    check  3 files, 0 favourites, 0 ratings
    forward  check  3 files, 0 favourites, 0 ratings

One start under the other spelling, and putting the setting back brings
none of it back. Nothing looks wrong afterwards -- the pictures are all
there, which is what makes it convincing.

The guard that was already here covers a root that is not THERE
(test_offline_drive_guard). This one is reachable and full; only the name
it is reached by changed -- a Docker mount moved, a library copied to
another drive, the same path typed in another case on Windows, where both
spellings open the same folder.

What it costs when it fires: nothing is deleted, so until the address
matches again the gallery lists both sets and every picture appears
twice. That is said in the message, and it is recoverable, which silent
deletion was not.
"""

from __future__ import annotations

import ast
import os

import pytest

import smartgallery

_OLD = "C:/ComfyUI/output"
_NEW = "C:/COMFYUI/OUTPUT"


def _paths(root, names):
    return {f"{root}/{name}" for name in names}


@pytest.fixture
def a_library_of_its_own(smartgallery_app, tmp_path, monkeypatch):
    """A gallery root nothing else has touched.

    Both end-to-end checks below turn on exactly which files are on disk
    and which rows exist, and the shared test gallery carries whatever the
    tests before them left there. Measured: this file passed under the
    full suite and failed under `just audit`, because an upload test had
    left a small.png in the root -- so the names on disk did not match the
    names in the rows, and the rule correctly declined to fire. The rule
    was right; the test was reading a different situation than it meant
    to.
    """
    root = tmp_path / "own_output"
    root.mkdir()
    monkeypatch.setattr(smartgallery_app, "BASE_OUTPUT_PATH", str(root))
    smartgallery_app.get_dynamic_folder_config(force_refresh=True)

    conn = smartgallery_app.get_db_connection()
    try:
        kept = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        conn.execute("DELETE FROM files")
        conn.commit()
    finally:
        conn.close()

    yield str(root), kept

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files")
        conn.commit()
    finally:
        conn.close()
    smartgallery_app.get_dynamic_folder_config(force_refresh=True)


def test_the_same_names_at_a_new_address_is_a_rename():
    """The bug: every row would go and the same pictures just arrived."""
    names = ["ComfyUI_00001_.png", "ComfyUI_00002_.png", "a/nested.png"]
    db = _paths(_OLD, names)
    disk = _paths(_NEW, names)

    assert smartgallery.looks_like_a_renamed_root(db, db - disk, disk - db)


def test_a_library_that_really_was_emptied_is_still_emptied():
    """Over-reach guard, and the one that decides whether this is safe.
    Nothing arrived, so the files really are gone and the rows have to
    follow -- otherwise the gallery fills with entries for pictures that
    do not exist."""
    db = _paths(_OLD, ["one.png", "two.png"])

    assert not smartgallery.looks_like_a_renamed_root(db, db, set())


def test_one_file_in_common_is_not_a_rename():
    """Over-reach guard: if anything at all still lines up, the address on
    record is the right one and the missing files are missing."""
    db = _paths(_OLD, ["one.png", "two.png", "three.png"])
    disk = _paths(_OLD, ["one.png"])

    assert not smartgallery.looks_like_a_renamed_root(db, db - disk, disk - db)


def test_the_same_number_of_different_files_is_not_a_rename():
    """Over-reach guard, and why this matches names rather than counting.
    Two hundred pictures deleted and two hundred others added is a busy
    afternoon, not a moved folder."""
    db = _paths(_OLD, ["old_%d.png" % i for i in range(200)])
    disk = _paths(_OLD, ["new_%d.png" % i for i in range(200)])

    assert not smartgallery.looks_like_a_renamed_root(db, db - disk, disk - db)


def test_an_empty_library_is_not_a_rename():
    """Over-reach guard: a first run has nothing to protect and must not
    be told anything."""
    disk = _paths(_NEW, ["one.png"])

    assert not smartgallery.looks_like_a_renamed_root(set(), set(), disk)


def test_a_rewrite_in_place_keeps_its_ratings():
    """The one case the name rule treats as a rename without it being one:
    every file replaced by a file of the same name, at the same address.
    Keeping the ratings there is what somebody would want -- these are
    the same pictures re-saved -- and it is the deliberate cost of a rule
    that cannot be fooled by a busy afternoon."""
    names = ["ComfyUI_00001_.png", "ComfyUI_00002_.png"]
    db = _paths(_OLD, names)
    disk = _paths(_OLD, names)

    # Nothing differs at all, so nothing is deleted either way.
    assert db - disk == set()
    assert not smartgallery.looks_like_a_renamed_root(db, db - disk, disk - db)


def test_a_file_that_really_went_is_still_removed(smartgallery_app, a_library_of_its_own):
    """The control this file most needs, and the only one that can run
    against the build before the change: the guard must not stop ordinary
    cleanup. A scan that keeps rows for pictures that are gone fills the
    gallery with entries that open nothing.

    Through the real sync rather than the decision function, so it holds
    either way -- which is what makes it a control rather than another
    check of the new code."""
    root, _kept = a_library_of_its_own
    folder = os.path.join(root, "gone_probe")
    os.makedirs(folder, exist_ok=True)
    present = os.path.join(folder, "still_here.png")
    with open(present, "wb") as fh:
        fh.write(b"x")

    vanished = os.path.join(folder, "deleted.png").replace(os.sep, "/")
    ghost_id = smartgallery.content_digest(vanished)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO files (id, path, mtime, name, type) VALUES (?,?,?,?,?)",
            (ghost_id, vanished, 1700000000.0, "deleted.png", "image"),
        )
        conn.commit()
    finally:
        conn.close()

    try:
        smartgallery_app.get_dynamic_folder_config(force_refresh=True)
        conn = smartgallery_app.get_db_connection()
        try:
            smartgallery_app.full_sync_database(conn)
            left = conn.execute("SELECT COUNT(*) FROM files WHERE id = ?", (ghost_id,)).fetchone()[0]
        finally:
            conn.close()

        assert left == 0, (
            "a row for a picture that is genuinely gone survived the scan; "
            "the gallery will list something that opens nothing"
        )
    finally:
        conn = smartgallery_app.get_db_connection()
        try:
            conn.execute("DELETE FROM files WHERE path LIKE ?", ("%gone_probe%",))
            conn.commit()
        finally:
            conn.close()
        __import__("shutil").rmtree(folder, ignore_errors=True)
        smartgallery_app.get_dynamic_folder_config(force_refresh=True)


def test_the_scan_asks_before_it_deletes(gallery_tree):
    """Placement: the decision has to be made where to_delete is worked
    out, not after the rows have gone."""

    tree = gallery_tree

    fn = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "full_sync_database"),
        None,
    )
    assert fn is not None, "full_sync_database is gone"

    called = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "looks_like_a_renamed_root"
    ]
    assert called, "the scan works out what to delete without asking whether the library simply moved"


def test_the_library_survives_a_scan_at_the_new_address(smartgallery_app, a_library_of_its_own):
    """End to end through the real sync: rate a picture, reach the folder
    by another spelling, scan, and the rating must still be there."""
    root, _kept = a_library_of_its_own
    folder = os.path.join(root, "rootmove_probe")
    os.makedirs(folder, exist_ok=True)
    names = ["ComfyUI_90001_.png", "ComfyUI_90002_.png"]
    ids = []
    conn = smartgallery_app.get_db_connection()
    try:
        for _index, name in enumerate(names):
            target = os.path.join(folder, name)
            with open(target, "wb") as fh:
                fh.write(b"x")
            # recorded under a spelling the scan will not produce
            recorded = target.replace(os.sep, "/").replace("rootmove_probe", "ROOTMOVE_PROBE")
            file_id = smartgallery.content_digest(recorded)
            ids.append(file_id)
            conn.execute(
                "INSERT OR REPLACE INTO files (id, path, mtime, name, type) VALUES (?,?,?,?,?)",
                (file_id, recorded, os.path.getmtime(target), name, "image"),
            )
            conn.execute(
                "INSERT OR REPLACE INTO file_ratings (file_id, client_uuid, rating) VALUES (?,?,?)",
                (file_id, "someone", 5),
            )
        conn.commit()
    finally:
        conn.close()

    try:
        conn = smartgallery_app.get_db_connection()
        try:
            before = conn.execute("SELECT COUNT(*) FROM file_ratings WHERE file_id IN (?,?)", ids).fetchone()[0]
            assert before == 2, "the fixture did not record the ratings"

            smartgallery_app.get_dynamic_folder_config(force_refresh=True)
            smartgallery_app.full_sync_database(conn)

            after = conn.execute("SELECT COUNT(*) FROM file_ratings WHERE file_id IN (?,?)", ids).fetchone()[0]
        finally:
            conn.close()

        assert after == 2, (
            "a scan that found the same pictures under a different spelling "
            "of the folder deleted the rows, and the ratings went with them"
        )
    finally:
        conn = smartgallery_app.get_db_connection()
        try:
            conn.execute("DELETE FROM files WHERE path LIKE ?", ("%OOTMOVE_PROBE%",))
            conn.execute("DELETE FROM files WHERE path LIKE ?", ("%ootmove_probe%",))
            conn.commit()
        finally:
            conn.close()
        __import__("shutil").rmtree(folder, ignore_errors=True)
        smartgallery_app.get_dynamic_folder_config(force_refresh=True)
