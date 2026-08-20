"""The storyboard frame route must not become a way to read the gallery.

Frames extracted from a video are cached in a per-file subfolder of the
thumbnail cache, and served by a route that takes the folder name straight
from the URL:

    directory = os.path.join(THUMBNAIL_CACHE_DIR, file_hash)
    return send_from_directory(directory, secure_filename(filename))

`send_from_directory` is careful about the filename but takes the
directory on trust, and Flask's default converter matches any segment
without a slash -- `..` included. So `/galleryout/storyboard_frame/../x.png`
resolved to the gallery root, which is where the pictures are: with
BASE_SMARTGALLERY_PATH left at its default it IS the output folder.

The route had no access check of any kind, so this answered for a caller
with no session at all, on a server started with --force-login. It went
around the login, around exhibition mode, and around the metadata
stripping in one request.
"""

from __future__ import annotations

import contextlib
import os

import pytest

_PAYLOAD = "STORYBOARD_TRAVERSAL_PAYLOAD"


@pytest.fixture
def victim_file(smartgallery_app):
    """A file sitting in the gallery root, one level above the cache."""
    gallery = smartgallery_app.BASE_SMARTGALLERY_PATH
    os.makedirs(gallery, exist_ok=True)
    path = os.path.join(gallery, "sbframe_secret.txt")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(_PAYLOAD)

    yield "sbframe_secret.txt"

    with contextlib.suppress(OSError):
        os.remove(path)


@pytest.fixture
def real_frame(smartgallery_app):
    """A genuine cached frame, so the route is known to work at all."""
    cache = smartgallery_app.THUMBNAIL_CACHE_DIR
    digest = "0123456789abcdef0123456789abcdef"
    folder = os.path.join(cache, digest)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "frame_0.jpg")
    with open(path, "wb") as handle:
        handle.write(b"\xff\xd8\xff\xe0 not really a jpeg")

    yield digest, "frame_0.jpg"

    try:
        os.remove(path)
        os.rmdir(folder)
    except OSError:
        pass


@pytest.mark.parametrize("segment", ["..", "%2e%2e", ".%2e", "%2E%2E"])
def test_the_cache_folder_cannot_be_escaped(smartgallery_app, victim_file, monkeypatch, segment):
    """No session, logins required: this returned the file's contents."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    client = smartgallery_app.app.test_client()

    resp = client.get(f"/galleryout/storyboard_frame/{segment}/{victim_file}")

    assert _PAYLOAD not in resp.get_data(as_text=True), f"{segment!r} escaped the frame cache and read the gallery root"
    assert resp.status_code != 200, resp.status_code


def test_an_anonymous_caller_is_refused_when_logins_are_required(smartgallery_app, real_frame, monkeypatch):
    """Even a well-formed request: nothing is served to someone who has not
    logged in on a server that demands it."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    digest, filename = real_frame

    resp = smartgallery_app.app.test_client().get(f"/galleryout/storyboard_frame/{digest}/{filename}")

    assert resp.status_code == 403, resp.status_code


def test_a_logged_in_caller_still_gets_their_frames(smartgallery_app, real_frame, monkeypatch):
    """The counterpart -- without this, refusing everything would pass the
    tests above while breaking the storyboard for everyone."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    digest, filename = real_frame

    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 9
        session["role"] = "CUSTOMER"

    resp = client.get(f"/galleryout/storyboard_frame/{digest}/{filename}")

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
    assert resp.get_data().startswith(b"\xff\xd8"), "the frame did not come back"


def test_the_default_local_install_still_serves_frames(smartgallery_app, real_frame, monkeypatch):
    """No login configured: one person, and the frames are theirs."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    digest, filename = real_frame

    resp = smartgallery_app.app.test_client().get(f"/galleryout/storyboard_frame/{digest}/{filename}")

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
