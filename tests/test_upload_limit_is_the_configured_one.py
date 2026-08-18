"""COMFYUI_MAX_UPLOAD_MB has to be the limit, not a suggestion.

The gallery sets Flask's ceiling from that setting, and then hands the web
server underneath it a ceiling written in as a fixed 2 GiB. The server
refuses a request before Flask ever sees it, so anything above 2048 MB was
agreed to by the app and refused by the transport. Measured with the
setting at 4000:

    COMFYUI_MAX_UPLOAD_MB      = 4000
    Flask MAX_CONTENT_LENGTH   = 4194304000 bytes (4000 MB)
    waitress max_request_body  = 2147483648 bytes (2048 MB)  <- literal

waitress/src/waitress/parser.py refuses at `content_length >= max_body`,
raising RequestEntityTooLarge from the transport, which does not reach any
Flask handler. So a 3 GB video -- ordinary for the professional formats
the stream route exists to play -- failed on a gallery configured to take
4 GB, and the number in the message corresponded to no setting.

Two things follow. The transport ceiling is derived from the same setting,
with headroom, so the app is always the one that decides; and the app now
says which limit was hit and what it is, because without a handler a 413
is an HTML page and the upload screen had only the number 413 to show.
"""

from __future__ import annotations

import ast
import io

import pytest
from flask import Flask, request

import smartgallery

_MIB = 1024 * 1024
_OLD_LITERAL = 2147483648  # the fixed 2 GiB that used to be passed


def test_the_transport_ceiling_is_not_written_in(gallery_tree):
    """The regression as it was: a literal in the serve() call, which no
    setting can move."""
    tree = gallery_tree

    serves = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "serve"
    ]
    assert serves, "the serve() call is gone; this check is stale"

    for call in serves:
        for keyword in call.keywords:
            if keyword.arg != "max_request_body_size":
                continue
            assert not isinstance(keyword.value, ast.Constant), (
                f"serve() at line {call.lineno} passes a fixed "
                f"max_request_body_size of {getattr(keyword.value, 'value', '?')}; "
                f"COMFYUI_MAX_UPLOAD_MB cannot raise it"
            )


def test_the_transport_ceiling_clears_the_app_ceiling():
    """An upload is a multipart body, so it is larger than the file inside
    it; and waitress refuses at >= its ceiling while Flask refuses above
    its own. Level ceilings would let the transport answer first, which is
    the answer that cannot name the setting."""
    app_limit = smartgallery.app.config["MAX_CONTENT_LENGTH"]

    assert app_limit < smartgallery.MAX_REQUEST_BODY_BYTES, (
        f"transport ceiling {smartgallery.MAX_REQUEST_BODY_BYTES} does not clear the app ceiling {app_limit}"
    )


@pytest.mark.parametrize("configured", [4000, 8000, 2049])
def test_a_setting_above_two_gigabytes_is_not_quietly_capped(configured):
    """The bug itself. It only shows up above 2048, which is why the
    default hid it: 2000 fits under the old literal.

    Asked of derive_upload_ceilings directly. It needed its own interpreter
    while the setting was read and turned into both limits at module scope;
    the arithmetic is a function now, so any value can be put to it without
    starting a gallery."""
    app_limit, transport_limit = smartgallery.derive_upload_ceilings(configured)

    assert app_limit == configured * _MIB
    assert transport_limit > app_limit, (
        f"COMFYUI_MAX_UPLOAD_MB={configured} gives an app ceiling of "
        f"{app_limit // _MIB} MB and a transport ceiling of "
        f"{transport_limit // _MIB} MB, so the real limit is the lower one"
    )
    assert transport_limit > _OLD_LITERAL, (
        f"the transport ceiling is still {transport_limit // _MIB} MB with "
        f"{configured} MB configured; uploads stop at the old fixed 2048 MB"
    )


def test_the_shipped_ceilings_came_from_that_same_arithmetic():
    """The function is only worth anything if startup still uses it -- the
    two limits were derived inline once and could be again."""
    expected = smartgallery.derive_upload_ceilings(smartgallery.MAX_UPLOAD_MB)

    assert (smartgallery.app.config["MAX_CONTENT_LENGTH"], smartgallery.MAX_REQUEST_BODY_BYTES) == expected, (
        "the running ceilings do not match derive_upload_ceilings, so the "
        "setting and the limits have parted company again"
    )


@pytest.fixture
def uploader(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "ADMIN"
    return client


def test_a_refused_upload_says_what_the_limit_is(smartgallery_app, uploader, monkeypatch):
    """Nothing handled 413, so the reply was an HTML error page and the
    upload screen fell back to showing the number 413."""
    monkeypatch.setitem(smartgallery_app.app.config, "MAX_CONTENT_LENGTH", 1024)

    response = uploader.post(
        "/galleryout/upload",
        data={
            "folder_key": "_root_",
            "files": (io.BytesIO(b"x" * 40000), "big.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 413, response.status_code
    body = response.get_json()
    assert body is not None, (
        f"a refused upload answered with something that is not JSON, so the "
        f"screen has only the number 413 to show: "
        f"{response.get_data(as_text=True)[:200]}"
    )
    assert "MB" in body["message"], body["message"]
    assert "COMFYUI_MAX_UPLOAD_MB" in body["message"], body["message"]


def test_the_limit_named_is_the_limit_in_force(smartgallery_app, uploader, monkeypatch):
    """A message naming a number that is not the setting would be worse
    than none -- that is the fault being fixed, one layer up."""
    monkeypatch.setitem(smartgallery_app.app.config, "MAX_CONTENT_LENGTH", 1024)
    monkeypatch.setattr(smartgallery_app, "MAX_UPLOAD_MB", 7, raising=False)

    response = uploader.post(
        "/galleryout/upload",
        data={
            "folder_key": "_root_",
            "files": (io.BytesIO(b"x" * 40000), "big.png"),
        },
        content_type="multipart/form-data",
    )

    assert "7 MB" in response.get_json()["message"], response.get_json()["message"]


def test_an_upload_within_the_limit_is_not_refused(smartgallery_app, uploader, monkeypatch):
    """Over-reach guard. A ceiling that refuses everything would satisfy
    every check above."""
    monkeypatch.setitem(smartgallery_app.app.config, "MAX_CONTENT_LENGTH", 50 * _MIB)

    response = uploader.post(
        "/galleryout/upload",
        data={
            "folder_key": "_root_",
            "files": (io.BytesIO(b"x" * 4096), "small.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code != 413, "an upload well under the limit was refused as too large"
    assert response.get_json() is not None


def test_an_unhandled_413_would_not_be_json():
    """Control for the check above: Flask's own answer to a body over the
    ceiling is an HTML page, so finding JSON there means a handler ran."""

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 1024

    @app.route("/take", methods=["POST"])
    def _take():
        return str(len(request.form))

    response = app.test_client().post(
        "/take",
        data={
            "files": (io.BytesIO(b"x" * 40000), "big.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 413, response.status_code
    assert response.get_json() is None, (
        "an unhandled 413 produced JSON, so the check above would pass without any handler at all"
    )
