"""Uploading a file whose name is not written in Latin script.

Uploads were sanitised with werkzeug's `secure_filename`, which
transliterates to ASCII and drops whatever will not convert. For a token
that is right; for someone's media it is not. It reduces "测试.png" to
"png" -- no extension left, so the upload is then refused for having an
unsupported type -- and quietly shortens "한글_0001_.png" to "0001_.png".

Chinese, Korean, Japanese, Cyrillic, Greek, Hebrew and Arabic filenames
all lost part or all of themselves, and in the worst case the upload
failed outright with a message naming a file the user had never heard of.

What is genuinely dangerous is still removed: directory parts, the
characters Windows forbids, control characters, trailing dots and spaces,
and the reserved device names.
"""

from __future__ import annotations

import contextlib
import io
import os

import pytest
from PIL import Image

_PREFIX = "uni_"


@pytest.fixture
def root_key(smartgallery_app):
    base = smartgallery_app.BASE_OUTPUT_PATH
    folders = smartgallery_app.get_dynamic_folder_config(force_refresh=True)
    for key, info in folders.items():
        if os.path.normpath(str(info["path"])) == os.path.normpath(base):
            yield key
            break
    else:
        pytest.skip("gallery root is not exposed as a folder key")

    for name in os.listdir(base):
        if name.startswith(_PREFIX) or name in ("png", "upload.png"):
            with contextlib.suppress(OSError):
                os.remove(os.path.join(base, name))
    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
    finally:
        conn.close()


def _upload(smartgallery_app, key, filename):
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (11, 22, 33)).save(buf, format="PNG")
    client = smartgallery_app.app.test_client()
    return client.post(
        "/galleryout/upload",
        data={"folder_key": key, "files": (io.BytesIO(buf.getvalue()), filename)},
        content_type="multipart/form-data",
    )


@pytest.mark.parametrize(
    "filename",
    [
        f"{_PREFIX}测试.png",  # Chinese
        f"{_PREFIX}한글.png",  # Korean
        f"{_PREFIX}рисунок.png",  # Cyrillic
        f"{_PREFIX}テスト_0001_.png",  # Japanese
        f"{_PREFIX}café.png",  # Latin with an accent
        f"{_PREFIX}Ελληνικά.png",  # Greek
    ],
)
def test_a_non_latin_name_survives_the_upload(smartgallery_app, root_key, filename):
    """The regression: these were renamed, and some were refused outright."""
    resp = _upload(smartgallery_app, root_key, filename)

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert os.path.exists(os.path.join(smartgallery_app.BASE_OUTPUT_PATH, filename)), (
        f"{filename} is not on disk under its own name; "
        f"the folder holds {sorted(os.listdir(smartgallery_app.BASE_OUTPUT_PATH))}"
    )


def test_the_upload_still_refuses_a_disallowed_type(smartgallery_app, root_key):
    """Keeping the name must not mean keeping anything: the extension
    whitelist still decides what may be written."""
    resp = _upload(smartgallery_app, root_key, f"{_PREFIX}payload.exe")

    assert not os.path.exists(os.path.join(smartgallery_app.BASE_OUTPUT_PATH, f"{_PREFIX}payload.exe")), (
        "a disallowed type was written to the gallery"
    )
    body = resp.get_data(as_text=True)
    assert "Successfully uploaded 0 files" in body, body


@pytest.mark.parametrize(
    ("hostile", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\win.ini", "win.ini"),
        ("..", "upload"),
        ("", "upload"),
        ("   ", "upload"),
        ("con.png", "_con.png"),
        ("COM1.png", "_COM1.png"),
        ("trailing.   ", "trailing"),
        ("with\x00null.png", "with_null.png"),
        ("a<b>c:d|e?f*g.png", "a_b_c_d_e_f_g.png"),
        ("测试.png", "测试.png"),
        # A leading dot is how a hidden file is spelled, so it is kept as-is.
        # The upload's extension whitelist then refuses it, which is correct:
        # splitext(".png") has no extension.
        (".png", ".png"),
    ],
)
def test_the_sanitiser_removes_what_is_actually_dangerous(smartgallery_app, hostile, expected):
    assert smartgallery_app.safe_media_filename(hostile) == expected


def test_a_hostile_name_cannot_escape_the_folder(smartgallery_app, root_key):
    """The property behind the table: whatever is written lands in the
    destination folder and nowhere else."""
    base = smartgallery_app.BASE_OUTPUT_PATH
    resp = _upload(smartgallery_app, root_key, f"../../{_PREFIX}escaped.png")

    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert os.path.exists(os.path.join(base, f"{_PREFIX}escaped.png"))
    outside = os.path.join(os.path.dirname(os.path.dirname(base)), f"{_PREFIX}escaped.png")
    assert not os.path.exists(outside), f"the upload was written outside the gallery: {outside}"
