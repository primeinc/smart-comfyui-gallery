"""The folder lifecycle: create, rename, delete.

Folder deletion is the most destructive operation the app offers -- it
takes everything inside with it -- and none of these routes had a test.
Protected keys must stay protected, a rename must carry the files' rows
with it (a row still pointing at the old path is a thumbnail that 404s),
and a delete must clear the rows it destroyed.
"""

from __future__ import annotations

import os
import shutil

import pytest
from inline_executor import InlineExecutor
from PIL import Image

_PREFIX = "fldroute_"


@pytest.fixture
def client(smartgallery_app):
    return smartgallery_app.app.test_client()


def _key_for(smartgallery_app, path):
    folders = smartgallery_app.get_dynamic_folder_config(force_refresh=True)
    return next((k for k, v in folders.items() if os.path.normpath(v["path"]) == os.path.normpath(path)), None)


def _make_folder(smartgallery_app, name, with_file=True, monkeypatch=None):
    """A real folder in the gallery, indexed by the REAL scanner.

    Hand-written rows are not good enough here: the scanner stores
    `<config path>` joined with the filename, which on Windows mixes the
    config's forward slashes with a native separator
    ("C:/gallery/sub\\a.png"). Both rename_folder and delete_folder match
    rows by that prefix, so a row seeded with all-backslash paths would
    fail to match and make correct routes look broken.
    """
    path = os.path.join(smartgallery_app.BASE_OUTPUT_PATH, name)
    os.makedirs(path, exist_ok=True)
    if with_file:
        Image.new("RGB", (16, 16), (200, 120, 40)).save(os.path.join(path, f"{_PREFIX}inside.png"))
        os.makedirs(smartgallery_app.THUMBNAIL_CACHE_DIR, exist_ok=True)
        if monkeypatch is not None:
            monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
        conn = smartgallery_app.get_db_connection()
        try:
            smartgallery_app.full_sync_database(conn)
        finally:
            conn.close()
    return path


@pytest.fixture(autouse=True)
def _cleanup(smartgallery_app):
    yield
    root = smartgallery_app.BASE_OUTPUT_PATH
    for entry in os.listdir(root):
        if entry.startswith(_PREFIX):
            shutil.rmtree(os.path.join(root, entry), ignore_errors=True)
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE id LIKE '{_PREFIX}%'")
        conn.commit()
    finally:
        conn.close()
    smartgallery_app.get_dynamic_folder_config(force_refresh=True)


def _rows_under(smartgallery_app, path):
    """Rows whose file lives under `path`.

    Compared as normalised paths rather than a SQL LIKE: the scanner
    stores the folder-config prefix verbatim (forward slashes) joined to
    the filename with the native separator, so a LIKE built from
    os.path.join would miss every row on Windows.
    """
    wanted = os.path.normcase(os.path.normpath(path)) + os.sep
    conn = smartgallery_app.get_db_connection()
    try:
        rows = [r[0] for r in conn.execute("SELECT path FROM files").fetchall()]
    finally:
        conn.close()
    return sum(1 for p in rows if os.path.normcase(os.path.normpath(p)).startswith(wanted))


def test_create_folder_makes_it_on_disk(smartgallery_app, client):
    name = f"{_PREFIX}created"
    resp = client.post("/galleryout/create_folder", json={"folder_name": name, "parent_key": "_root_"})
    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert os.path.isdir(os.path.join(smartgallery_app.BASE_OUTPUT_PATH, name))


def test_delete_folder_removes_it_and_clears_its_rows(smartgallery_app, client, monkeypatch):
    path = _make_folder(smartgallery_app, f"{_PREFIX}doomed", monkeypatch=monkeypatch)
    key = _key_for(smartgallery_app, path)
    if key is None:
        pytest.skip("new folder is not exposed by the folder config")
    assert _rows_under(smartgallery_app, path) == 1

    resp = client.post(f"/galleryout/delete_folder/{key}")

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert not os.path.exists(path), "the folder is still on disk"
    assert _rows_under(smartgallery_app, path) == 0, "rows still point inside a folder that no longer exists"


def test_delete_folder_refuses_a_protected_key(smartgallery_app, client):
    for key in smartgallery_app.PROTECTED_FOLDER_KEYS:
        resp = client.post(f"/galleryout/delete_folder/{key}")
        assert resp.status_code == 403, f"{key} was not protected from deletion"
    assert os.path.isdir(smartgallery_app.BASE_OUTPUT_PATH)


def test_delete_folder_rejects_an_unknown_key(client):
    assert client.post("/galleryout/delete_folder/no_such_key").status_code == 404


def test_rename_folder_moves_it_and_carries_its_rows(smartgallery_app, client, monkeypatch):
    """A row left pointing at the old path renders as a thumbnail that
    404s and cannot be cleaned up through the UI."""
    path = _make_folder(smartgallery_app, f"{_PREFIX}before", monkeypatch=monkeypatch)
    key = _key_for(smartgallery_app, path)
    if key is None:
        pytest.skip("new folder is not exposed by the folder config")

    resp = client.post(f"/galleryout/rename_folder/{key}", json={"new_name": f"{_PREFIX}after"})

    assert resp.status_code == 200, resp.get_data(as_text=True)
    new_path = os.path.join(smartgallery_app.BASE_OUTPUT_PATH, f"{_PREFIX}after")
    assert os.path.isdir(new_path), "the folder was not renamed on disk"
    assert not os.path.exists(path)
    assert _rows_under(smartgallery_app, new_path) == 1, "the file's row did not follow the folder rename"
    assert _rows_under(smartgallery_app, path) == 0, "a row still points at the old path"


def test_rename_folder_refuses_a_protected_key(smartgallery_app, client):
    for key in smartgallery_app.PROTECTED_FOLDER_KEYS:
        resp = client.post(f"/galleryout/rename_folder/{key}", json={"new_name": f"{_PREFIX}nope"})
        assert resp.status_code == 403, f"{key} was not protected from renaming"


@pytest.mark.parametrize("bad", ["", "   ", ".", ".."])
def test_rename_folder_rejects_empty_and_dot_names(smartgallery_app, client, bad):
    path = _make_folder(smartgallery_app, f"{_PREFIX}guard", with_file=False)
    key = _key_for(smartgallery_app, path)
    if key is None:
        pytest.skip("new folder is not exposed by the folder config")

    resp = client.post(f"/galleryout/rename_folder/{key}", json={"new_name": bad})

    assert resp.status_code >= 400, f"{bad!r} was accepted as a folder name"
    assert os.path.isdir(path), "the folder changed despite an invalid name"


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [
        ("a/b", "ab"),
        ("a" + chr(92) + "b", "ab"),
        ("up:down", "updown"),
    ],
)
def test_rename_folder_strips_separators_instead_of_escaping(smartgallery_app, client, attempt, expected):
    """Path characters are stripped rather than refused, which is fine --
    what matters is that the folder cannot land outside the gallery."""
    path = _make_folder(smartgallery_app, f"{_PREFIX}strip", with_file=False)
    key = _key_for(smartgallery_app, path)
    if key is None:
        pytest.skip("new folder is not exposed by the folder config")

    resp = client.post(f"/galleryout/rename_folder/{key}", json={"new_name": attempt})

    assert resp.status_code == 200, resp.get_data(as_text=True)
    landed = os.path.join(smartgallery_app.BASE_OUTPUT_PATH, expected)
    assert os.path.isdir(landed), f"expected the folder at {landed}"
    parent = os.path.abspath(os.path.join(smartgallery_app.BASE_OUTPUT_PATH, os.pardir))
    assert not os.path.exists(os.path.join(parent, expected)), "the rename escaped the gallery root"
    shutil.rmtree(landed, ignore_errors=True)
