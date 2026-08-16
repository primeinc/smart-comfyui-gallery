"""The per-file details endpoint must not hand a visitor what the mode hides.

Exhibition mode strips prompts out of the pictures, keeps them out of album
listings, and refuses the cluster and workflow endpoints outright. All of
that is undone by one call: `/api/file_full_details/<id>` returned the
whole file row -- prompt and model names included -- plus the workflow's
node pipeline and the file's path on the server, to any visitor who could
see the picture.

It is the natural endpoint for an interface to call when someone opens an
image, which is exactly why it needs the same rule as everything else. The
ratings, comments and collections it also carries are not secret and still
arrive.

`check_metadata` had a narrower version of the same thing: it reports a
file's real path when the entry is a link, which told a visitor where the
library lives.
"""

from __future__ import annotations

import concurrent.futures
import os

import pytest
from PIL import Image

_PREFIX = "fdr_"
_MODEL = "SECRETMODEL_v9.safetensors"
_PROMPT = "a prompt nobody outside should read"


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
def shown_file(smartgallery_app, monkeypatch):
    """A file in a public album, carrying a prompt and a model name."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures,
                        "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)

    base = smartgallery_app.BASE_OUTPUT_PATH
    path = os.path.join(base, f"{_PREFIX}shown.png")
    Image.new("RGB", (16, 16), (4, 4, 4)).save(path)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        file_id = conn.execute("SELECT id FROM files WHERE name = ?",
                               (f"{_PREFIX}shown.png",)).fetchone()[0]
        conn.execute("UPDATE files SET workflow_files = ?, workflow_prompt = ? "
                     "WHERE id = ?", (f"{_MODEL} ||| lora_secret.safetensors",
                                      _PROMPT, file_id))
        conn.execute("INSERT INTO collections (name, type, is_public, created_at) "
                     "VALUES (?, 'user_album', 1, 1.0)", (f"{_PREFIX}album",))
        coll_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("INSERT INTO collection_files (collection_id, file_id) "
                     "VALUES (?, ?)", (coll_id, file_id))
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


def test_a_visitor_gets_the_details_at_all(smartgallery_app, shown_file):
    """Control: the endpoint answers a visitor, so an absent prompt below
    means it was removed and not that the request was refused."""
    resp = _as(smartgallery_app, "CUSTOMER").get(
        f"/galleryout/api/file_full_details/{shown_file}")

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
    body = resp.get_json()
    assert body["status"] == "success"
    assert body["file"]["name"] == f"{_PREFIX}shown.png"
    assert body["collections"], "the parts a visitor may see went missing too"


def test_the_prompt_and_models_do_not_reach_a_visitor(smartgallery_app, shown_file):
    """The regression: all three arrived in one response."""
    body = _as(smartgallery_app, "CUSTOMER").get(
        f"/galleryout/api/file_full_details/{shown_file}").get_data(as_text=True)

    assert _PROMPT not in body, "the prompt was handed to a visitor"
    assert _MODEL not in body, "the model names were handed to a visitor"
    flat = body.replace("\\\\", "/").replace("\\", "/")
    assert smartgallery_app.BASE_OUTPUT_PATH.replace("\\", "/") not in flat, (
        "the server path was handed to a visitor")


def test_staff_still_get_everything(smartgallery_app, shown_file):
    """The counterpart: this endpoint is what fills the Asset Info panel."""
    body = _as(smartgallery_app, "ADMIN").get(
        f"/galleryout/api/file_full_details/{shown_file}").get_data(as_text=True)

    assert _PROMPT in body, "staff lost the prompt"
    assert _MODEL in body, "staff lost the model names"


def test_check_metadata_keeps_the_real_path_from_a_visitor(smartgallery_app, shown_file):
    resp = _as(smartgallery_app, "CUSTOMER").get(
        f"/galleryout/check_metadata/{shown_file}")

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
    assert resp.get_json().get("real_path") is None, (
        "a visitor was told where the file lives on the server")


def test_the_default_local_install_is_unaffected(smartgallery_app, shown_file,
                                                  monkeypatch):
    """With no login there is one person and it is all theirs."""
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)

    body = smartgallery_app.app.test_client().get(
        f"/galleryout/api/file_full_details/{shown_file}").get_data(as_text=True)

    assert _PROMPT in body and _MODEL in body
