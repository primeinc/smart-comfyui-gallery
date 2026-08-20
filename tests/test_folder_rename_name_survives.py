"""A renamed folder must be stored under the name it actually got.

Renaming a folder cleaned the requested name with a rule of its own --
re.sub(r'[\\\\/:*?"<>|]', '', name) -- which removes the characters Windows
FORBIDS and keeps the ones Windows silently CHANGES.

The route .strip()s first, which happens to catch a plain trailing space.
A trailing dot is not caught, and neither is anything ending in one:

    requested      after strip    after regex    Windows keeps
    'holiday '     'holiday'      'holiday'      'holiday'
    'holiday.'     'holiday.'     'holiday.'     'holiday'   <-- differs
    'holiday. '    'holiday.'     'holiday.'     'holiday'   <-- differs
    'holiday..'    'holiday..'    'holiday..'    'holiday'   <-- differs
    'holiday  .'   'holiday  .'   'holiday  .'   'holiday'   <-- differs

Those do not fail. The folder is created without the dot and the rows are
written with it:

    database says : holiday./ComfyUI_00001_.png
    disk actually : holiday/ComfyUI_00001_.png

A file's id is its path, so those are different files. The next scan walks
the disk, finds a picture with no row and a row with no picture, and
deletes the row -- ratings, comments and album membership with it. The
rename reported success.

os.path.exists is no help: Windows resolves "holiday." to "holiday" when
asked whether it exists, so a lookup agrees with the database and only a
directory listing disagrees. A scan walks, so the listing is what counts,
and that is what the check below compares against.

safe_media_filename already knew this, because uploads and moves hit it
first: "the trailing dots and spaces Windows silently strips". That rule
is now its own function, strip_what_windows_drops, and both callers use
it.

Not the whole of safe_media_filename, though, and the first attempt at
this got it wrong: that function takes a basename, which is right for an
upload arriving as a path and wrong for someone typing a folder name --
it turns "a/b" into "b" and drops half of what they typed. The existing
test above this route pins separator handling for containment, and it
caught exactly that. The forbidden characters keep their own removal
here; only the silent ones are shared.
"""

from __future__ import annotations

import ast
import inspect
import os
import re

import pytest

import smartgallery

# The rule that was there before, kept so the tests can show what it let
# through rather than describing it.
_OLD_RULE = re.compile(r'[\\/:*?"<>|]')


def _survives_a_round_trip(tmp_path, name):
    """Make a directory called `name` and report what it is really called."""
    target = tmp_path / name
    try:
        os.mkdir(str(target))
    except OSError:
        return None
    return os.listdir(str(tmp_path))[0]


@pytest.mark.parametrize("requested", ["holiday ", "holiday.", "holiday. ", "holiday.."])
def test_a_name_the_filesystem_would_change_is_changed_first(requested):
    """The bug. What is stored has to be what the disk will hold."""
    cleaned = smartgallery.safe_media_filename(requested, fallback="")

    assert cleaned == cleaned.rstrip(". "), (
        f"{requested!r} was cleaned to {cleaned!r}, which still ends in a "
        f"dot or space -- Windows drops those when it creates the folder, "
        f"and the rows keep them"
    )


@pytest.mark.parametrize("requested", ["holiday.", "holiday. ", "holiday..", "holiday  ."])
def test_the_rule_that_was_there_let_it_through(requested):
    """Control, modelling the route exactly -- .strip() and then the old
    regex. Without it the checks above prove nothing was ever wrong.

    A plain trailing space is deliberately not in this list: .strip()
    already caught that one, and claiming otherwise would overstate what
    was broken."""
    old = _OLD_RULE.sub("", requested.strip())

    assert old != old.rstrip(". "), f"the previous rule already handled {requested!r}, so there was nothing to fix"


def test_a_plain_trailing_space_was_already_handled():
    """The other half of being accurate about it: .strip() ran first, so
    this case never reached the disk wrong."""
    assert _OLD_RULE.sub("", "holiday ".strip()) == "holiday"


def test_windows_really_does_drop_them(tmp_path):
    """Control for the consequence, on whichever system is running this.

    Skips where the filesystem keeps such names, because there the bug
    does not exist and the checks above are simply harmless."""
    kept = _survives_a_round_trip(tmp_path, "holiday ")

    if kept == "holiday ":
        pytest.skip("this filesystem keeps trailing spaces")
    assert kept == "holiday", kept


@pytest.fixture
def a_folder_with_a_picture(smartgallery_app, monkeypatch):
    """A real folder under the gallery root, known to the database."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    root = smartgallery_app.BASE_OUTPUT_PATH
    folder = os.path.join(root, "rename_probe")
    os.makedirs(folder, exist_ok=True)
    picture = os.path.join(folder, "ComfyUI_00001_.png")
    with open(picture, "wb") as fh:
        fh.write(b"not really a png")

    stored = picture.replace(os.sep, "/")
    file_id = smartgallery.content_digest(stored)
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO files (id, path, mtime, name, type) VALUES (?,?,?,?,?)",
            (file_id, stored, os.path.getmtime(picture), "ComfyUI_00001_.png", "image"),
        )
        conn.commit()
    finally:
        conn.close()

    smartgallery_app.STATE.folder_config = None
    folders = smartgallery_app.get_dynamic_folder_config()
    key = next(
        (
            k
            for k, v in folders.items()
            if os.path.normcase(os.path.normpath(v["path"])) == os.path.normcase(os.path.normpath(folder))
        ),
        None,
    )

    yield key, file_id, folder

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE path LIKE ?", (root.replace(os.sep, "/") + "/rename_probe%",))
        conn.execute("DELETE FROM files WHERE path LIKE ?", (root.replace(os.sep, "/") + "/holiday%",))
        conn.commit()
    finally:
        conn.close()
    for leftover in os.listdir(root):
        full = os.path.join(root, leftover)
        if os.path.isdir(full) and (leftover.startswith(("rename_probe", "holiday"))):
            __import__("shutil").rmtree(full, ignore_errors=True)
    smartgallery_app.STATE.folder_config = None


@pytest.mark.parametrize("requested", ["holiday ", "holiday."])
def test_after_renaming_every_stored_path_is_really_there(smartgallery_app, a_folder_with_a_picture, requested):
    """The symptom, through the route: rename a folder to a name the
    filesystem will not keep, and the database must still describe where
    the files actually are."""
    key, _file_id, _folder = a_folder_with_a_picture
    if key is None:
        pytest.skip("the probe folder is not in the folder config")

    client = smartgallery_app.app.test_client()
    response = client.post(f"/galleryout/rename_folder/{key}", json={"new_name": requested})
    assert response.status_code == 200, response.get_json()

    conn = smartgallery_app.get_db_connection()
    try:
        rows = conn.execute(
            "SELECT path FROM files WHERE path LIKE ?",
            (smartgallery_app.BASE_OUTPUT_PATH.replace(os.sep, "/") + "/holiday%",),
        ).fetchall()
    finally:
        conn.close()

    assert rows, "the rename left no rows under the new name at all"

    # What a scan sees is a directory listing, not a path lookup. Windows
    # resolves "holiday " to "holiday" when asked whether it exists, so
    # os.path.exists agrees with the database and the scan does not --
    # which is exactly how this stayed invisible.
    listing = os.listdir(smartgallery_app.BASE_OUTPUT_PATH)
    for row in rows:
        stored = row["path"].replace("\\", "/")
        folder_name = stored.rsplit("/", 2)[-2]
        assert folder_name in listing, (
            f"the database says a picture is in a folder called "
            f"{folder_name!r}, and the folder on disk is one of {listing}. "
            f"A scan walks the disk, so it finds a picture with no row and "
            f"a row with no picture -- and deletes the row, with its "
            f"ratings, comments and album membership."
        )


def test_an_ordinary_name_is_untouched():
    """Over-reach guard, and every rename anybody actually does."""
    for name in ["holiday", "Renders 2026", "第一章", "Ordner-Größe", "my.folder.v2"]:
        assert smartgallery.safe_media_filename(name, fallback="") == name, name


def test_a_name_that_cleans_away_to_nothing_is_refused(smartgallery_app, monkeypatch):
    """The old rule rejected '.' and '..' by name. The new one has to
    refuse them too, rather than renaming the folder to a fallback nobody
    asked for."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    client = smartgallery_app.app.test_client()

    for name in [".", "..", "   ", "..."]:
        response = client.post("/galleryout/rename_folder/bm90X3JlYWw=", json={"new_name": name})
        body = response.get_json()
        assert response.status_code == 400, (name, response.status_code, body)
        assert body["message"] == "Invalid name.", (name, body)


def test_both_names_go_through_one_trailing_rule(gallery_tree):
    """Two rules for one idea is how they drift apart. This one only came
    back because the folder path had its own."""

    tree = gallery_tree

    fn = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "rename_folder"), None
    )
    assert fn is not None, "rename_folder is gone"

    called = {node.func.id for node in ast.walk(fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "strip_what_windows_drops" in called, (
        "rename_folder does not remove what Windows removes, so the name it records is not the name the folder gets"
    )

    body = inspect.getsource(smartgallery.safe_media_filename)
    assert "strip_what_windows_drops" in body, (
        "safe_media_filename has its own copy of the trailing rule again; "
        "one idea, two spellings, is how they came apart in the first place"
    )
