"""Comments belong to a picture, and follow who may see it.

Reading and posting comments asked only whether somebody was signed in,
never whether this caller may see the picture the comments are about. So a
visitor given one album could read the owner's notes on every other
picture in the library -- including ones the gallery refuses to send them
-- and could attach comments of their own to any of them.

Measured before the fix, as a guest with a picture deliberately left out
of the public album:

    visitor may fetch private : 403
    comments read (private)   : 200  LEAKS COMMENT
    comment post  (private)   : 200
    comments the visitor wrote on the private file: 1

The read is the worse half. Comments on a client's pictures are where an
owner writes about the client.

This is the same fault as the ratings one before it, found by sweeping
every route that takes a file id for whether it ever asks about
visibility. The last test here is that sweep, so the next such route has
to answer the question too.
"""

from __future__ import annotations

import ast
import concurrent.futures
import contextlib
import os

import pytest
from PIL import Image

from inline_executor import InlineExecutor

_PREFIX = "cmtvis_"
_MISSING_ID = "e" * 32
_OWNER_NOTE = "PRIVATE-NOTE about the client"


@pytest.fixture
def library(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", InlineExecutor)
    base = smartgallery_app.BASE_OUTPUT_PATH
    names = [f"{_PREFIX}shared.png", f"{_PREFIX}private.png"]
    for name in names:
        Image.new("RGB", (8, 8), (6, 6, 6)).save(os.path.join(base, name))

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.execute("DELETE FROM collections WHERE name = 'Comment Album'")
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        ids = {
            r["name"]: r["id"]
            for r in conn.execute(f"SELECT name, id FROM files WHERE name LIKE '{_PREFIX}%'").fetchall()
        }
        conn.execute(
            "INSERT INTO collections (name, type, is_public) VALUES (?, ?, 1)", ("Comment Album", "user_album")
        )
        coll_id = conn.execute("SELECT id FROM collections WHERE name = ?", ("Comment Album",)).fetchone()[0]
        conn.execute("INSERT INTO collection_files (collection_id, file_id) VALUES (?, ?)", (coll_id, ids[names[0]]))
        for name in names:
            conn.execute(
                "INSERT INTO file_comments (file_id, client_uuid, author_name, "
                "comment_text, target_audience) VALUES (?, 'admin', 'Owner', ?, "
                "'public')",
                (ids[name], _OWNER_NOTE),
            )
        conn.commit()
    finally:
        conn.close()

    yield ids

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.execute("DELETE FROM collections WHERE name = 'Comment Album'")
        conn.commit()
    finally:
        conn.close()
    for name in names:
        with contextlib.suppress(OSError):
            os.remove(os.path.join(base, name))


def _visitor(smartgallery_app, monkeypatch, role="GUEST"):
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 42
        session["role"] = role
        session["full_name"] = "A Visitor"
    return client


def _written(smartgallery_app, file_id):
    conn = smartgallery_app.get_db_connection()
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM file_comments WHERE file_id = ? AND client_uuid = '42'", (file_id,)
        ).fetchone()[0]
    finally:
        conn.close()


def test_a_visitor_reads_and_writes_on_a_shared_picture(smartgallery_app, library, monkeypatch):
    """Control. Every refusal below is only worth something while the
    feature still works where it should."""
    client = _visitor(smartgallery_app, monkeypatch)
    shared = library[f"{_PREFIX}shared.png"]

    read = client.get(f"/galleryout/api/exhibition/comments?file_id={shared}")
    assert read.status_code == 200, read.get_json()
    assert _OWNER_NOTE in read.get_data(as_text=True)

    posted = client.post("/galleryout/api/exhibition/post_comment", json={"file_id": shared, "text": "lovely"})
    assert posted.status_code == 200, posted.get_json()
    assert _written(smartgallery_app, shared) == 1


def test_a_visitor_cannot_read_comments_on_a_private_picture(smartgallery_app, library, monkeypatch):
    """The leak: an owner's notes about a picture nobody shared."""
    client = _visitor(smartgallery_app, monkeypatch)
    private = library[f"{_PREFIX}private.png"]
    assert client.get(f"/galleryout/file/{private}").status_code == 403

    response = client.get(f"/galleryout/api/exhibition/comments?file_id={private}")

    assert response.status_code == 404, response.get_json()
    assert _OWNER_NOTE not in response.get_data(as_text=True)


def test_a_visitor_cannot_comment_on_a_private_picture(smartgallery_app, library, monkeypatch):
    client = _visitor(smartgallery_app, monkeypatch)
    private = library[f"{_PREFIX}private.png"]

    response = client.post(
        "/galleryout/api/exhibition/post_comment", json={"file_id": private, "text": "I can write here"}
    )

    assert response.status_code == 404, response.get_json()
    assert _written(smartgallery_app, private) == 0


def test_hidden_and_missing_look_the_same(smartgallery_app, library, monkeypatch):
    """Otherwise the route says which ids a library holds."""
    client = _visitor(smartgallery_app, monkeypatch)
    private = library[f"{_PREFIX}private.png"]

    def ask(file_id):
        answer = client.get(f"/galleryout/api/exhibition/comments?file_id={file_id}")
        return answer.status_code, answer.get_json()

    assert ask(private) == ask(_MISSING_ID)


def test_a_manager_still_sees_everything(smartgallery_app, library, monkeypatch):
    """Over-reach guard: moderation needs every comment on every picture."""
    client = _visitor(smartgallery_app, monkeypatch, role="MANAGER")
    private = library[f"{_PREFIX}private.png"]

    response = client.get(f"/galleryout/api/exhibition/comments?file_id={private}")

    assert response.status_code == 200, response.get_json()
    assert _OWNER_NOTE in response.get_data(as_text=True)


def test_a_local_gallery_is_untouched(smartgallery_app, library, monkeypatch):
    """The commonest install: no login, every picture the owner's own."""
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    client = smartgallery_app.app.test_client()
    private = library[f"{_PREFIX}private.png"]

    response = client.get(f"/galleryout/api/exhibition/comments?file_id={private}")

    assert response.status_code == 200, response.get_json()


def test_every_route_taking_a_file_id_decides_who_may_use_it(gallery_tree):
    """The sweep that found this, and the ratings fault before it.

    A route that takes a file id and checks only that somebody is signed
    in passes the route audit -- it IS gated -- while never asking WHICH
    file. That is the shape both bugs had. Three answers are accepted:

      management_api_only    privileged callers only
      is_file_accessible     asks about this particular file
      should_strip_metadata  refuses non-privileged callers outright

    Anything else has to justify itself here."""
    tree = gallery_tree
    keys = {"file_id", "file_ids", "fileId"}

    def calls(node):
        found = set()
        for call in ast.walk(node):
            if isinstance(call, ast.Call):
                if isinstance(call.func, ast.Name):
                    found.add(call.func.id)
                elif isinstance(call.func, ast.Attribute):
                    found.add(call.func.attr)
        return found

    unguarded, examined = [], 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        routes = [
            d
            for d in node.decorator_list
            if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "route"
        ]
        if not routes:
            continue

        takes_a_file = any(a.arg in keys for a in node.args.args) or any(
            isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and c.func.attr in ("get", "getlist")
            and c.args
            and isinstance(c.args[0], ast.Constant)
            and str(c.args[0].value) in keys
            for c in ast.walk(node)
        )
        if not takes_a_file:
            continue

        examined += 1
        named = calls(node)
        decorated = {d.id for d in node.decorator_list if isinstance(d, ast.Name)}
        if decorated & {"management_api_only"}:
            continue
        if named & {"is_file_accessible", "_check_file_access", "should_strip_metadata"}:
            continue
        unguarded.append(f"{node.name} (line {node.lineno})")

    assert examined > 20, f"only {examined} routes take a file id; the sweep is not working"
    assert not unguarded, f"{len(unguarded)} route(s) take a file id and never decide who may use it: {unguarded}"
