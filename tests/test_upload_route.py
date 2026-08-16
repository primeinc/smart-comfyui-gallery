"""The upload endpoint.

Uploading is one of two ways a file arrives in a folder (moving is the
other), and the two disagreed: move renamed around a name collision while
upload silently overwrote whatever was already there and still reported
success. These pin the agreement, plus the extension policy and filename
sanitisation that keep the endpoint from writing anything it likes.
"""

from __future__ import annotations

import io
import os

import pytest


@pytest.fixture()
def upload_target(smartgallery_app):
    """(client, folder_key, folder_path) for the gallery root."""
    folders = smartgallery_app.get_dynamic_folder_config(force_refresh=True)
    root = smartgallery_app.BASE_OUTPUT_PATH
    key = next((k for k, v in folders.items()
                if os.path.normpath(v["path"]) == os.path.normpath(root)), None)
    if key is None:
        pytest.skip("gallery root is not exposed as a folder key")
    return smartgallery_app.app.test_client(), key, root


def _upload(client, key, name, body=b"uploaded bytes"):
    return client.post("/galleryout/upload", data={
        "folder_key": key,
        "files": (io.BytesIO(body), name),
    }, content_type="multipart/form-data")


def _cleanup(folder, *names):
    for name in names:
        try:
            os.remove(os.path.join(folder, name))
        except OSError:
            pass


def test_upload_writes_the_file_into_the_chosen_folder(upload_target):
    client, key, folder = upload_target
    try:
        resp = _upload(client, key, "upl_plain.png", b"first upload")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "success"
        written = os.path.join(folder, "upl_plain.png")
        assert os.path.isfile(written)
        with open(written, "rb") as fh:
            assert fh.read() == b"first upload"
    finally:
        _cleanup(folder, "upl_plain.png")


def test_upload_never_overwrites_an_existing_file(upload_target):
    """The regression: this used to replace the existing file's bytes and
    report success, destroying media the user never chose to delete."""
    client, key, folder = upload_target
    existing = os.path.join(folder, "upl_clash.png")
    with open(existing, "wb") as fh:
        fh.write(b"ORIGINAL CONTENT")
    try:
        resp = _upload(client, key, "upl_clash.png", b"INCOMING CONTENT")
        assert resp.status_code == 200

        with open(existing, "rb") as fh:
            assert fh.read() == b"ORIGINAL CONTENT", (
                "the upload overwrote a file already in the folder")
        siblings = [f for f in os.listdir(folder)
                    if f.startswith("upl_clash(") and f.endswith(".png")]
        assert siblings, f"the uploaded file was dropped entirely: {os.listdir(folder)}"
        with open(os.path.join(folder, siblings[0]), "rb") as fh:
            assert fh.read() == b"INCOMING CONTENT"
    finally:
        _cleanup(folder, "upl_clash.png",
                 *[f for f in os.listdir(folder) if f.startswith("upl_clash(")])


def test_upload_refuses_a_disallowed_extension(upload_target):
    client, key, folder = upload_target
    resp = _upload(client, key, "upl_evil.exe", b"MZ...")
    assert resp.status_code == 207, "a rejected upload should report partial success"
    assert not os.path.exists(os.path.join(folder, "upl_evil.exe"))


def test_upload_sanitises_a_traversing_filename(upload_target):
    """A name carrying a path must land in the destination folder, never
    above it."""
    client, key, folder = upload_target
    parent = os.path.abspath(os.path.join(folder, os.pardir))
    escaped = os.path.join(parent, "upl_escape.png")
    try:
        _upload(client, key, "../upl_escape.png", b"payload")
        assert not os.path.exists(escaped), "upload wrote outside the destination"
        assert os.path.isfile(os.path.join(folder, "upl_escape.png"))
    finally:
        _cleanup(folder, "upl_escape.png")
        try:
            os.remove(escaped)
        except OSError:
            pass


def test_upload_rejects_an_unknown_destination(smartgallery_app):
    client = smartgallery_app.app.test_client()
    resp = client.post("/galleryout/upload", data={
        "folder_key": "no_such_folder",
        "files": (io.BytesIO(b"x"), "upl_nowhere.png"),
    }, content_type="multipart/form-data")
    assert resp.status_code == 404


def test_upload_without_a_folder_key_is_rejected(smartgallery_app):
    client = smartgallery_app.app.test_client()
    resp = client.post("/galleryout/upload", data={
        "files": (io.BytesIO(b"x"), "upl_nokey.png"),
    }, content_type="multipart/form-data")
    assert resp.status_code == 400
