"""Input images must be servable under the name they actually have.

/galleryout/input_file/ ran secure_filename() on the name before looking
for it. That function exists to make a safe name for a file you are about
to WRITE; using it to find one you are about to READ throws away most of
the world's filenames:

    测试.png           -> 'png'
    рисунок.png        -> 'png'
    イラスト.png         -> 'png'
    Ordner-Größe.png   -> 'Ordner-Groe.png'

Each of those 404'd. It is the same fault that was fixed for uploads
earlier in this branch, in a route that was missed.

It also flattens separators, so clipspace/pasted.png became
clipspace_pasted.png and 404'd -- and clipspace is where ComfyUI puts
every pasted and masked image, so that half was broken for everyone,
whatever language they use.

Containment was never what secure_filename was providing here. It is
checked on the resolved path, and again by send_from_directory's
safe_join. These tests hold both ends: the names come back, and nothing
outside the input folder does.
"""

from __future__ import annotations

import os
import urllib.parse

import pytest
from PIL import Image

_NAMES = ["plain.png", "测试.png", "Ordner-Größe.png", "рисунок.png", "イラスト.png", "ЖУРНАЛ.png"]


@pytest.fixture
def input_folder(smartgallery_app, monkeypatch):
    """Pictures in the ComfyUI input folder, plus something outside it that
    must stay unreachable."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    root = smartgallery_app.BASE_INPUT_PATH
    os.makedirs(os.path.join(root, "clipspace"), exist_ok=True)
    for name in _NAMES:
        Image.new("RGB", (8, 8), (5, 5, 5)).save(os.path.join(root, name))
    Image.new("RGB", (8, 8), (6, 6, 6)).save(os.path.join(root, "clipspace", "pasted.png"))

    outside = os.path.join(os.path.dirname(os.path.abspath(root)), "outside.txt")
    with open(outside, "w", encoding="utf-8") as handle:
        handle.write("must never be served")

    return smartgallery_app.app.test_client(), outside


def _get(client, name):
    return client.get("/galleryout/input_file/" + urllib.parse.quote(name))


@pytest.mark.parametrize("name", _NAMES)
def test_a_name_in_any_language_is_served(input_folder, name):
    """The bug: everything but plain ASCII came back 404."""
    client, _outside = input_folder

    response = _get(client, name)

    assert response.status_code == 200, name
    assert len(response.get_data()) > 0


def test_the_clipspace_subfolder_is_served(input_folder):
    """ComfyUI writes every pasted and masked image there, so this half was
    broken regardless of language."""
    client, _outside = input_folder

    response = _get(client, "clipspace/pasted.png")

    assert response.status_code == 200
    assert len(response.get_data()) > 0


def test_a_file_that_is_not_there_is_still_missing(input_folder):
    """Control. Serving everything with a 200 would satisfy the tests above
    and would be a far worse bug than the one being fixed."""
    client, _outside = input_folder

    assert _get(client, "no_such_image.png").status_code == 404


@pytest.mark.parametrize(
    "attempt",
    [
        "../outside.txt",
        "../../outside.txt",
        "clipspace/../../outside.txt",
        "..%2Foutside.txt",
        "C:/Windows/win.ini",
        "/etc/passwd",
    ],
)
def test_nothing_outside_the_input_folder_is_served(input_folder, attempt):
    """secure_filename was doing this by accident. Removing it must not
    remove the protection, so every shape is held here."""
    client, outside = input_folder
    assert os.path.exists(outside), "the fixture's decoy is missing"

    # Followed, because a leading-slash attempt makes a double-slash URL
    # that Werkzeug answers with a 308 to the normalised path. The hop is
    # not the answer; where it lands is.
    response = client.get("/galleryout/input_file/" + attempt, follow_redirects=True)

    assert response.status_code != 200, f"{attempt} was served: {response.status_code}"
    assert b"must never be served" not in response.get_data()


def test_a_guest_is_still_refused_when_a_login_is_required(smartgallery_app, input_folder, monkeypatch):
    """The route's other half. Loosening the name handling must not loosen
    who may ask."""
    client, _outside = input_folder
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)

    with client.session_transaction() as session:
        session["role"] = "GUEST"

    assert _get(client, "plain.png").status_code == 403


def test_a_signed_in_manager_is_allowed(smartgallery_app, input_folder, monkeypatch):
    """Control for the test above: it must be refusing the role, not
    refusing everyone."""
    client, _outside = input_folder
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)

    with client.session_transaction() as session:
        session["role"] = "MANAGER"
        session["user_id"] = 2

    assert _get(client, "plain.png").status_code == 200
