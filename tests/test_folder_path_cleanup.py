"""Folder operations have to reach the files inside subfolders.

Stored paths mix separators. The scanner joins a folder-config path, which
uses forward slashes, with `os.sep`, so on Windows a file sitting directly
in a folder is stored as `C:/gallery/box\\a.png` while one a level deeper
is `C:/gallery/box/sub\\b.png`.

Every folder operation identifies "the files under this folder" by a
string prefix, and a prefix of `folder + os.sep` only ever matches the
first shape. On Linux `os.sep` is '/' and the same prefix matches at every
depth, so these operations behave differently on the two platforms -- and
ComfyUI writes into date-stamped subfolders by default, which is exactly
the shape that gets missed.

The consequences are invisible until someone looks: deleting a folder
leaves rows pointing at files that no longer exist, and renaming one
rewrites nested files to a path they were never at.
"""

from __future__ import annotations

import contextlib
import os

import pytest
from inline_executor import InlineExecutor
from PIL import Image

_PREFIX = "fpc_"
_BOX = f"{_PREFIX}box"
_SIBLING = f"{_PREFIX}box_archive"  # shares a string prefix with _BOX


def _rows(smartgallery_app):
    conn = smartgallery_app.get_db_connection()
    try:
        return [r[0] for r in conn.execute("SELECT path FROM files WHERE name LIKE ?", (f"{_PREFIX}%",)).fetchall()]
    finally:
        conn.close()


def _key_for(smartgallery_app, suffix):
    folders = smartgallery_app.get_dynamic_folder_config(force_refresh=True)
    for key, info in folders.items():
        if str(info["path"]).replace("\\", "/").endswith(suffix):
            return key
    pytest.skip(f"folder {suffix} is not exposed by the folder config")


@pytest.fixture
def library(smartgallery_app, monkeypatch):
    """A folder with a file in it and a file one level deeper, plus a
    sibling folder whose name starts with the same string."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    base = smartgallery_app.BASE_OUTPUT_PATH
    deep_dir = os.path.join(base, _BOX, "2026-08-15")
    os.makedirs(deep_dir, exist_ok=True)
    os.makedirs(os.path.join(base, _SIBLING), exist_ok=True)

    made = [
        os.path.join(base, _BOX, f"{_PREFIX}flat.png"),
        os.path.join(deep_dir, f"{_PREFIX}deep.png"),
        os.path.join(base, _SIBLING, f"{_PREFIX}sibling.png"),
    ]
    for path in made:
        Image.new("RGB", (16, 16), (120, 120, 120)).save(path)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
        smartgallery_app.full_sync_database(conn)
    finally:
        conn.close()

    yield

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
    finally:
        conn.close()
    for dirpath, _dirs, names in os.walk(base, topdown=False):
        for name in names:
            if name.startswith(_PREFIX):
                with contextlib.suppress(OSError):
                    os.remove(os.path.join(dirpath, name))
        if os.path.basename(dirpath).startswith(_PREFIX) or dirpath.endswith("2026-08-15"):
            with contextlib.suppress(OSError):
                os.rmdir(dirpath)


@pytest.fixture
def client(smartgallery_app):
    return smartgallery_app.app.test_client()


def test_the_fixture_indexes_both_depths(smartgallery_app, library):
    """Control: without this the later assertions could pass on an empty
    database and prove nothing."""
    names = sorted(os.path.basename(p.replace("\\", "/")) for p in _rows(smartgallery_app))
    assert names == [f"{_PREFIX}deep.png", f"{_PREFIX}flat.png", f"{_PREFIX}sibling.png"], names


def test_deleting_a_folder_removes_rows_for_files_in_its_subfolders(smartgallery_app, client, library):
    """The regression: the folder and its subfolders are gone from disk, so
    a surviving row is a gallery entry whose image 404s forever."""
    key = _key_for(smartgallery_app, _BOX)

    resp = client.post(f"/galleryout/delete_folder/{key}")
    assert resp.status_code == 200, resp.get_data(as_text=True)

    left = [p for p in _rows(smartgallery_app) if f"/{_BOX}/" in p.replace("\\", "/")]
    assert left == [], f"rows survived a deleted folder: {left}"

    # The sibling shares a name prefix and must be untouched -- a delete
    # that took it too would also make the assertion above pass.
    survivors = [p.replace("\\", "/") for p in _rows(smartgallery_app)]
    assert any(f"/{_SIBLING}/" in p for p in survivors), f"deleting {_BOX} also removed {_SIBLING}: {survivors}"


def test_renaming_a_folder_keeps_the_files_in_their_subfolders(smartgallery_app, client, library):
    """A nested file must end up under the renamed folder AND its subfolder.
    Rebuilding the path from the basename alone flattens it to the top
    level, where no such file exists."""
    key = _key_for(smartgallery_app, _BOX)
    new_name = f"{_PREFIX}renamed"

    resp = client.post(f"/galleryout/rename_folder/{key}", json={"new_name": new_name})
    assert resp.status_code == 200, resp.get_data(as_text=True)

    paths = [p.replace("\\", "/") for p in _rows(smartgallery_app)]
    deep = [p for p in paths if p.endswith(f"{_PREFIX}deep.png")]
    assert deep, f"the nested file lost its row entirely: {paths}"
    assert f"/{new_name}/2026-08-15/" in deep[0], f"the nested file was flattened out of its subfolder: {deep[0]}"
    assert os.path.exists(deep[0].replace("/", os.sep)), f"the row points at a file that does not exist: {deep[0]}"


def test_unmounting_forgets_files_in_subfolders_of_the_mount(smartgallery_app, client, tmp_path, monkeypatch):
    """Mounting is how an external drive joins the gallery, and unmounting
    has to leave no trace of it in the database -- while never touching the
    files themselves, which live on the other drive.

    This drives the real routes, so it needs a real link; on Windows that
    is a junction, which normally needs no privileges."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    external = tmp_path / "external"
    (external / "dated").mkdir(parents=True)
    flat = external / f"{_PREFIX}mflat.png"
    deep = external / "dated" / f"{_PREFIX}mdeep.png"
    for path in (flat, deep):
        Image.new("RGB", (16, 16), (10, 90, 200)).save(path)

    link_name = f"{_PREFIX}mount"
    resp = client.post("/galleryout/mount_folder", json={"link_name": link_name, "target_path": str(external)})
    if resp.status_code != 200:
        pytest.skip(f"cannot create a link in this environment: {resp.get_data(as_text=True)}")

    try:
        conn = smartgallery_app.get_db_connection()
        try:
            smartgallery_app.full_sync_database(conn)
        finally:
            conn.close()
        assert len(_rows(smartgallery_app)) == 2, f"the mount was not indexed: {_rows(smartgallery_app)}"

        key = _key_for(smartgallery_app, link_name)
        resp = client.post("/galleryout/unmount_folder", json={"folder_key": key})
        assert resp.status_code == 200, resp.get_data(as_text=True)

        assert _rows(smartgallery_app) == [], f"rows survived the unmount: {_rows(smartgallery_app)}"
        assert flat.exists(), "unmounting deleted the files on the other drive"
        assert deep.exists(), "unmounting deleted the files on the other drive"
    finally:
        link_path = os.path.join(smartgallery_app.BASE_OUTPUT_PATH, link_name)
        if os.path.isdir(link_path):
            with contextlib.suppress(OSError):
                os.rmdir(link_path)
        conn = smartgallery_app.get_db_connection()
        try:
            conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
            conn.execute("DELETE FROM mounted_folders")
            conn.commit()
        finally:
            conn.close()


def test_renaming_a_folder_leaves_a_similarly_named_sibling_alone(smartgallery_app, client, library):
    """`fpc_box_archive` starts with `fpc_box`. Matching the prefix without
    requiring a separator after it drags the sibling's files along."""
    key = _key_for(smartgallery_app, _BOX)

    resp = client.post(f"/galleryout/rename_folder/{key}", json={"new_name": f"{_PREFIX}renamed"})
    assert resp.status_code == 200, resp.get_data(as_text=True)

    sibling = [p.replace("\\", "/") for p in _rows(smartgallery_app) if p.endswith(f"{_PREFIX}sibling.png")]
    assert sibling, "the sibling's row disappeared"
    assert f"/{_SIBLING}/" in sibling[0], f"a folder with a shared name prefix was rewritten too: {sibling[0]}"
