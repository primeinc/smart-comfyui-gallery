"""The default install: one person, no login, everything theirs.

This is what almost everyone runs, and it is the configuration where every
guard added in this branch behaves differently: management_api_only lets an
unauthenticated caller through, should_strip_metadata() is False, and
is_file_accessible answers True for everything. The other two journey tests
cover exhibition mode and a login-protected gallery -- neither exercises
this one, and a mistake here reaches the most people.

The direction of the risk is the opposite of the other two. There, the
danger is that something is not withheld; here, it is that something is
withheld from the person who owns it. A gallery that hides its owner's own
prompts, or refuses them their own tools because a guard was written for a
different audience, is broken in a way no security test would notice.
"""

from __future__ import annotations

import contextlib
import os

import pytest
from inline_executor import InlineExecutor
from PIL import Image, PngImagePlugin

_PREFIX = "local_"
_PROMPT = "LOCALPROMPT a red barn at sunrise"
_MODEL = "LOCALMODEL_v2.safetensors"


@pytest.fixture
def local_gallery(smartgallery_app, monkeypatch):
    """No login configured, one picture carrying a prompt."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    base = smartgallery_app.BASE_OUTPUT_PATH
    path = os.path.join(base, f"{_PREFIX}pic.png")
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", _PROMPT)
    Image.new("RGB", (40, 40), (200, 80, 60)).save(path, pnginfo=info)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name LIKE ?", (f"{_PREFIX}%",))
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        file_id = conn.execute("SELECT id FROM files WHERE name = ?", (f"{_PREFIX}pic.png",)).fetchone()[0]
        conn.execute(
            "UPDATE files SET workflow_prompt = ?, workflow_files = ? WHERE id = ?", (_PROMPT, _MODEL, file_id)
        )
        conn.commit()
    finally:
        conn.close()

    yield file_id

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
        if os.path.basename(dirpath).startswith(_PREFIX):
            with contextlib.suppress(OSError):
                os.rmdir(dirpath)


def test_no_login_is_asked_for(smartgallery_app, local_gallery):
    """The first thing that would break if a guard were written for the
    wrong audience: the gallery opening at all."""
    page = smartgallery_app.app.test_client().get("/galleryout/view/_root_", follow_redirects=True)

    assert page.status_code == 200, page.status_code
    body = page.get_data(as_text=True)
    assert local_gallery in body, "the library did not render"
    # The gallery interface, not the login form. Looking for the word
    # "password" does not distinguish them: index.html carries the user
    # manager, which has password fields of its own. These are the markers
    # test_force_login_mode uses to detect the interface leaking, read the
    # other way round.
    assert "lightbox-toolbar" in body or "gallery-item" in body, (
        "the gallery interface did not render on a local install"
    )


def test_the_owner_sees_their_own_prompts(smartgallery_app, local_gallery):
    """The redactions added for visitors must not reach the person whose
    prompts they are -- which is the whole point of the AI panel."""
    client = smartgallery_app.app.test_client()

    details = client.get(f"/galleryout/api/file_full_details/{local_gallery}")
    assert details.status_code == 200, details.get_data(as_text=True)[:200]
    body = details.get_data(as_text=True)
    assert _PROMPT in body, "the owner's own prompt was withheld from them"
    assert _MODEL in body, "the owner's own model names were withheld from them"

    picture = client.get(f"/galleryout/file/{local_gallery}")
    assert picture.status_code == 200
    assert _PROMPT.encode() in picture.get_data(), "the owner's own file was stripped on the way to them"


def test_the_owner_can_use_the_management_tools(smartgallery_app, local_gallery):
    """Everything that gained management_api_only in this branch has to
    stay open here, where there is nobody to be management OVER."""
    client = smartgallery_app.app.test_client()

    for label, url in [
        ("collections", "/galleryout/api/collections"),
        ("sidebar", "/galleryout/api/sidebar_state"),
        ("ai status", "/galleryout/api/aidam/status"),
        ("indexing status", "/galleryout/ai_indexing/status"),
        ("site settings", "/galleryout/api/site_settings"),
        ("search options", "/galleryout/api/search_options"),
    ]:
        resp = client.get(url)
        assert resp.status_code == 200, f"{label} answered {resp.status_code}"


def test_the_owner_can_work_on_their_library(smartgallery_app, local_gallery):
    """Rate, comment, rename -- with the data still attached afterwards."""
    client = smartgallery_app.app.test_client()

    assert (
        client.post(
            "/galleryout/api/exhibition/rate", json={"file_id": local_gallery, "rating": 5, "client_uuid": "admin"}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/galleryout/api/exhibition/post_comment",
            json={"file_id": local_gallery, "text": "keep this one", "client_uuid": "admin"},
        ).status_code
        == 200
    )

    renamed = client.post(f"/galleryout/rename_file/{local_gallery}", json={"new_name": f"{_PREFIX}renamed.png"})
    assert renamed.status_code == 200, renamed.get_data(as_text=True)

    conn = smartgallery_app.get_db_connection()
    try:
        row = conn.execute("SELECT id FROM files WHERE name = ?", (f"{_PREFIX}renamed.png",)).fetchone()
        assert row, "the renamed file is not in the library"
        rating = conn.execute("SELECT rating FROM file_ratings WHERE file_id = ?", (row[0],)).fetchone()
        comments = conn.execute("SELECT COUNT(*) FROM file_comments WHERE file_id = ?", (row[0],)).fetchone()[0]
    finally:
        conn.close()

    assert rating, "the rating did not survive the rename"
    assert rating[0] == 5, "the rating did not survive the rename"
    assert comments == 1, "the comment did not survive the rename"


def test_my_ratings_only_finds_what_the_owner_rated(smartgallery_app, local_gallery):
    """The identity a local install rates under is 'admin', decided by the
    page. The gallery has to look it up under the same name -- this is the
    first bug fixed in this branch, kept in the journey because it is
    invisible until someone turns the filter on."""
    client = smartgallery_app.app.test_client()
    client.post("/galleryout/api/exhibition/rate", json={"file_id": local_gallery, "rating": 4, "client_uuid": "admin"})

    with client.session_transaction() as session:
        session["my_ratings_only"] = True

    page = client.get("/galleryout/view/_root_?sort_by=rating")

    assert page.status_code == 200
    assert local_gallery in page.get_data(as_text=True), "a file the owner rated is missing from their own ratings view"
