"""When metadata cannot be stripped, the file must not be served anyway.

`--exhibition` and `--force-login` exist so that people who are not staff
can look at the pictures without reading how they were made: ComfyUI
writes the full prompt and workflow into the file itself, and
`should_strip_metadata()` turns on a cleaning pass for every guest
request.

The cleaning pass can fail, and the common way is not exotic. Video and
audio are stripped by shelling out to ffmpeg, and that branch is only
entered when an ffmpeg was located at all -- so on any install without
one, stripping a video returns False. It also returns False when ffmpeg
cannot stream-copy the codec into that container, and when Pillow raises
on an image.

Whatever the cause, the caller served the original file instead and wrote
a warning to a console the operator is not reading. The mode kept working,
looked correct, and quietly handed out the prompts.

A privacy control that cannot do its job has to refuse, not degrade.
"""

from __future__ import annotations

import concurrent.futures
import os

import pytest
from PIL import Image, PngImagePlugin

_PREFIX = "strip_"
_SECRET = "a very secret prompt: nobody should read this"


class _InlineExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


@pytest.fixture()
def guarded(smartgallery_app, monkeypatch):
    """A file in the library, and a server that owes guests a cleaned copy."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures,
                        "ProcessPoolExecutor", _InlineExecutor)
    # Exhibition mode is where this matters: it is the mode that shows the
    # library to visitors, and the only one where a non-staff caller may
    # fetch a file at all (is_file_accessible refuses them outright under
    # --force-login, so the stripping path is never reached there).
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)

    base = smartgallery_app.BASE_OUTPUT_PATH
    path = os.path.join(base, f"{_PREFIX}pic.png")
    image = Image.new("RGB", (32, 32), (140, 20, 60))
    info = PngImagePlugin.PngInfo()
    info.add_text("prompt", _SECRET)  # exactly how ComfyUI records it
    image.save(path, pnginfo=info)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        file_id = conn.execute("SELECT id FROM files WHERE name = ?",
                               (f"{_PREFIX}pic.png",)).fetchone()[0]
        # A visitor may only see files in a public album, so the file has to
        # be in one for the request to reach the stripping step.
        conn.execute("INSERT INTO collections (name, type, is_public, created_at) "
                     "VALUES (?, 'user_album', 1, 1.0)", (f"{_PREFIX}album",))
        coll_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("INSERT INTO collection_files (collection_id, file_id) VALUES (?, ?)",
                     (coll_id, file_id))
        conn.commit()
    finally:
        conn.close()

    yield file_id

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.execute(f"DELETE FROM collections WHERE name LIKE '{_PREFIX}%'")
        conn.commit()
    finally:
        conn.close()
    try:
        os.remove(path)
    except OSError:
        pass


def _as(smartgallery_app, role):
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 9
        session["role"] = role
    return client


def test_the_secret_really_is_in_the_file(smartgallery_app, guarded):
    """Control: if the fixture cannot embed the prompt then a later test
    finding no prompt in the response proves nothing at all."""
    path = os.path.join(smartgallery_app.BASE_OUTPUT_PATH, f"{_PREFIX}pic.png")
    with open(path, "rb") as handle:
        assert _SECRET.encode() in handle.read(), "the fixture embedded no prompt"


def test_a_guest_is_refused_when_stripping_fails(smartgallery_app, guarded, monkeypatch):
    """The regression: no ffmpeg, so the strip fails and the original used
    to go out with the prompt still inside it."""
    monkeypatch.setattr(smartgallery_app, "strip_media_metadata",
                        lambda *a, **k: False)
    client = _as(smartgallery_app, "CUSTOMER")

    resp = client.get(f"/galleryout/file/{guarded}")

    assert _SECRET.encode() not in resp.get_data(), (
        "the prompt was served to a guest because stripping failed")
    assert resp.status_code != 200, (
        f"a guest got a 200 for a file that could not be cleaned ({resp.status_code})")


def test_staff_still_get_the_original(smartgallery_app, guarded, monkeypatch):
    """The counterpart: stripping is for guests. Staff must keep getting
    the real file, or the fix has broken the gallery for its owner."""
    monkeypatch.setattr(smartgallery_app, "strip_media_metadata",
                        lambda *a, **k: False)
    client = _as(smartgallery_app, "ADMIN")

    resp = client.get(f"/galleryout/file/{guarded}")

    assert resp.status_code == 200, resp.status_code
    assert _SECRET.encode() in resp.get_data(), "staff were denied their own metadata"


def test_a_guest_cannot_download_what_could_not_be_cleaned(
        smartgallery_app, guarded, monkeypatch):
    """The download route had the same fallback, and it matters more: the
    visitor keeps the file and can read it at leisure."""
    monkeypatch.setattr(smartgallery_app, "strip_media_metadata",
                        lambda *a, **k: False)
    client = _as(smartgallery_app, "CUSTOMER")

    resp = client.get(f"/galleryout/download/{guarded}")

    assert _SECRET.encode() not in resp.get_data(), (
        "the original file was handed to a guest because stripping failed")
    assert resp.status_code != 200, resp.status_code


def test_the_workflow_itself_stays_behind_its_own_gate(smartgallery_app, guarded):
    """Not a regression -- a check that the obvious way round the whole
    thing is already shut, since stripping the file would be pointless if
    the workflow could simply be asked for."""
    client = _as(smartgallery_app, "CUSTOMER")

    assert client.get(f"/galleryout/workflow/{guarded}").status_code == 403
    assert client.get(f"/galleryout/node_summary/{guarded}").status_code == 403


def test_a_guest_gets_the_picture_when_stripping_works(smartgallery_app, guarded):
    """And the whole point: a guest still sees the image, minus the prompt.
    Without this the refusal above could be satisfied by never serving
    anything."""
    client = _as(smartgallery_app, "CUSTOMER")

    resp = client.get(f"/galleryout/file/{guarded}")

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
    body = resp.get_data()
    assert body[:8] == b"\x89PNG\r\n\x1a\n", "a guest did not receive a PNG"
    assert _SECRET.encode() not in body, "the cleaned copy still carries the prompt"
