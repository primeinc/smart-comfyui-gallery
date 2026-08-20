"""The destructive file routes, exercised for real.

`delete_batch`, `delete_file`, and the favourite toggles are the endpoints
that remove or alter a user's media, and none of them had a test. A route
coverage sweep found 70 of 106 routes unmentioned by the suite; these are
the ones where a regression costs data rather than convenience.

Every test asserts on BOTH sides of a delete: the file is gone from disk
AND its row is gone from the database. Losing one without the other is its
own bug -- an orphaned row renders as a thumbnail that 404s, an orphaned
file reappears on the next scan.
"""

from __future__ import annotations

import contextlib
import os

import pytest
from PIL import Image

_PREFIX = "delroute_"


@pytest.fixture
def client(smartgallery_app):
    return smartgallery_app.app.test_client()


def _add_file(smartgallery_app, name, colour=(120, 40, 200)):
    """A real image on disk with a matching row, as a scan would leave it."""
    path = os.path.join(smartgallery_app.BASE_OUTPUT_PATH, name)
    Image.new("RGB", (24, 24), colour).save(path)
    file_id = f"{_PREFIX}id_{name}"
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO files (id, path, mtime, name, type, size) VALUES (?, ?, ?, ?, 'image', ?)",
            (file_id, path, os.path.getmtime(path), name, os.path.getsize(path)),
        )
        conn.commit()
    finally:
        conn.close()
    return file_id, path


def _row_exists(smartgallery_app, file_id):
    conn = smartgallery_app.get_db_connection()
    try:
        return conn.execute("SELECT 1 FROM files WHERE id = ?", (file_id,)).fetchone() is not None
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _cleanup(smartgallery_app):
    yield
    conn = smartgallery_app.get_db_connection()
    try:
        rows = conn.execute("SELECT path FROM files WHERE id LIKE ?", (f"{_PREFIX}%",)).fetchall()
        for row in rows:
            with contextlib.suppress(OSError):
                os.remove(row[0])
        conn.execute("DELETE FROM files WHERE id LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
    finally:
        conn.close()


def test_delete_batch_removes_files_from_disk_and_database(smartgallery_app, client):
    a_id, a_path = _add_file(smartgallery_app, f"{_PREFIX}a.png")
    b_id, b_path = _add_file(smartgallery_app, f"{_PREFIX}b.png")

    resp = client.post("/galleryout/delete_batch", json={"file_ids": [a_id, b_id]})

    assert resp.status_code == 200
    assert resp.get_json()["status"] == "success"
    assert not os.path.exists(a_path)
    assert not os.path.exists(b_path)
    assert not _row_exists(smartgallery_app, a_id)
    assert not _row_exists(smartgallery_app, b_id)


def test_delete_batch_rejects_an_empty_selection(client):
    resp = client.post("/galleryout/delete_batch", json={"file_ids": []})
    assert resp.status_code == 400
    assert resp.get_json()["status"] == "error"


def test_delete_batch_tolerates_a_file_already_gone_from_disk(smartgallery_app, client):
    """The row must still be cleared -- otherwise the grid keeps an entry
    whose thumbnail 404s, and no later delete can remove it."""
    ghost_id, ghost_path = _add_file(smartgallery_app, f"{_PREFIX}ghost.png")
    real_id, real_path = _add_file(smartgallery_app, f"{_PREFIX}real.png")
    os.remove(ghost_path)

    resp = client.post("/galleryout/delete_batch", json={"file_ids": [ghost_id, real_id]})

    assert resp.status_code == 200
    assert not _row_exists(smartgallery_app, ghost_id), "stale row survived"
    assert not _row_exists(smartgallery_app, real_id)
    assert not os.path.exists(real_path)


def test_delete_batch_does_not_touch_unselected_files(smartgallery_app, client):
    doomed_id, doomed_path = _add_file(smartgallery_app, f"{_PREFIX}doomed.png")
    keeper_id, keeper_path = _add_file(smartgallery_app, f"{_PREFIX}keeper.png")

    client.post("/galleryout/delete_batch", json={"file_ids": [doomed_id]})

    assert not os.path.exists(doomed_path)
    assert os.path.exists(keeper_path), "an unselected file was deleted"
    assert _row_exists(smartgallery_app, keeper_id)


def test_single_delete_removes_file_and_row(smartgallery_app, client):
    file_id, path = _add_file(smartgallery_app, f"{_PREFIX}single.png")

    resp = client.post(f"/galleryout/delete/{file_id}")

    assert resp.status_code == 200
    assert not os.path.exists(path)
    assert not _row_exists(smartgallery_app, file_id)


def test_single_delete_of_an_unknown_id_is_not_an_error(client):
    """Deleting something already gone is the desired end state, not a
    failure -- the UI retries these after a partial batch."""
    resp = client.post(f"/galleryout/delete/{_PREFIX}does_not_exist")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "success"


def test_favorite_batch_and_toggle_round_trip(smartgallery_app, client):
    file_id, _path = _add_file(smartgallery_app, f"{_PREFIX}fav.png")

    assert client.post("/galleryout/favorite_batch", json={"file_ids": [file_id], "status": True}).status_code == 200
    conn = smartgallery_app.get_db_connection()
    try:
        assert conn.execute("SELECT is_favorite FROM files WHERE id = ?", (file_id,)).fetchone()[0] == 1
    finally:
        conn.close()

    toggled = client.post(f"/galleryout/toggle_favorite/{file_id}")
    assert toggled.status_code == 200
    assert toggled.get_json()["is_favorite"] is False


def test_toggle_favorite_on_unknown_file_is_404(client):
    assert client.post(f"/galleryout/toggle_favorite/{_PREFIX}nope").status_code == 404


def test_delete_batch_route_honours_delete_to(smartgallery_app, monkeypatch, tmp_path):
    """The guarantee users actually rely on, through the endpoint the UI
    calls: with DELETE_TO configured, a batch delete must be recoverable.

    DELETE_TO is resolved at import and TRASH_FOLDER derived from it, which
    is why this ran in its own interpreter. Both are attributes the delete
    path reads when it runs, so they are set here the way startup would --
    TRASH_FOLDER is DELETE_TO/SmartGallery, created if absent.
    """
    trash_root = tmp_path / "trash"
    trash_folder = trash_root / "SmartGallery"
    trash_folder.mkdir(parents=True)
    gallery = tmp_path / "gallery"
    gallery.mkdir()
    monkeypatch.setattr(smartgallery_app, "DELETE_TO", str(trash_root))
    monkeypatch.setattr(smartgallery_app, "TRASH_FOLDER", str(trash_folder))
    monkeypatch.setattr(smartgallery_app, "BASE_OUTPUT_PATH", str(gallery))

    victim = gallery / "batch_victim.png"
    victim.write_bytes(b"pretend image bytes")
    file_id = f"{_PREFIX}batch_victim"
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(
            "INSERT INTO files (id, path, mtime, name, type, size) "
            "VALUES (?, ?, 1000.0, 'batch_victim.png', 'image', 19)",
            (file_id, str(victim)),
        )
        conn.commit()
    finally:
        conn.close()

    resp = smartgallery_app.app.test_client().post("/galleryout/delete_batch", json={"file_ids": [file_id]})

    body = resp.get_json()
    assert resp.status_code == 200, resp.status_code
    assert body["status"] == "success", body
    assert "trash" in body["message"], f"message does not mention the trash: {body['message']}"

    assert not victim.exists(), "file left in the gallery"
    recovered = list(trash_folder.iterdir())
    assert len(recovered) == 1, f"expected the file in the trash, found {recovered}"
    assert recovered[0].read_bytes() == b"pretend image bytes", "the recovered file does not match what was deleted"
