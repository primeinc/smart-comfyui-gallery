"""Exhibition mode shows a curated collection and nothing behind it.

The posture is the whole point: a visitor is sent to what was shared, the
raw folder tree is refused, and every route that changes or destroys media
answers 403 no matter what the session claims. An exhibition that leaked
the library, or let a forged ADMIN session delete from it, would be worse
than not having the mode.

The mode also refuses to start against a gallery that was never run
normally. Creating an empty database beside the real one is the failure
that made the pre-flight necessary: the exhibition comes up blank, the
owner sees nothing, and the real library is untouched but invisible.

Each case used to start a fresh interpreter with `--exhibition` on argv --
and the fixture ran one more just to prepare the database. IS_EXHIBITION_MODE
is read at request time, and the pre-flight is already a function reading
IS_EXHIBITION_MODE and DATABASE_FILE, so both are set on the loaded gallery
instead (pytest doc/en/how-to/monkeypatch.rst:243-247), and SystemExit plus
capsys carry what the exit code and stdout used to
(doc/en/how-to/capture-stdout-stderr.rst:112-142).
"""

from __future__ import annotations

import os
import sqlite3

import pytest

_DESTRUCTIVE = [
    ('/galleryout/delete_batch', {'file_ids': ['x']}),
    ('/galleryout/move_batch', {'file_ids': ['x'], 'destination_folder': '_root_'}),
    ('/galleryout/copy_batch', {'file_ids': ['x'], 'destination_folder': '_root_'}),
    ('/galleryout/delete_folder/_root_', {}),
    ('/galleryout/rename_file/x', {'new_name': 'y.png'}),
    ('/galleryout/prepare_batch_zip', {'file_ids': ['x']}),
    ('/galleryout/create_folder', {'folder_name': 'x', 'parent_key': '_root_'}),
]


@pytest.fixture
def exhibiting(smartgallery_app, monkeypatch):
    """The gallery as `--exhibition` with no admin password leaves it, and
    with something to exhibit.

    The mode flags are taken from derive_login_policy rather than invented,
    because they are not independent: --exhibition without --admin-pass
    also sets ADMIN_CONFIG_MISSING, and that is what makes gallery_view
    answer 403. Setting only IS_EXHIBITION_MODE left folder browsing
    answering 200 -- a state the launcher cannot actually produce.
    """
    force_login, missing, _short = smartgallery_app.derive_login_policy(
        None, exhibition=True, force_login=False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", force_login)
    monkeypatch.setattr(smartgallery_app, "ADMIN_CONFIG_MISSING", missing)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(
            "INSERT INTO collections (name, type, color, is_public, created_at) "
            "VALUES ('Public Show', 'user_album', '#fff', 1, 1000.0)")
        conn.commit()
    finally:
        conn.close()

    yield smartgallery_app

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM collections WHERE name = 'Public Show'")
        conn.commit()
    finally:
        conn.close()


def test_the_flag_still_reaches_the_setting(smartgallery_app):
    """Control: if --exhibition stopped setting IS_EXHIBITION_MODE, every
    test below would still pass while the mode did nothing."""
    parsed, _unknown = smartgallery_app._parser.parse_known_args(["--exhibition"])

    assert parsed.exhibition is True, "--exhibition no longer parses to anything"


def test_exhibition_hides_the_underlying_folders(exhibiting):
    """A visitor is shown a curated collection, never the library behind
    it."""
    client = exhibiting.app.test_client()

    entrance = client.get('/galleryout/')
    browse = client.get('/galleryout/view/_root_')

    assert entrance.status_code in (301, 302), entrance.status_code
    assert browse.status_code == 403, (
        f'raw folder browsing answered {browse.status_code} in exhibition mode')


@pytest.mark.parametrize(("path", "payload"), _DESTRUCTIVE)
def test_exhibition_refuses_the_destructive_apis(exhibiting, path, payload):
    """Every management route must answer 403 in this mode, whatever the
    session claims -- these are the routes that delete and move media."""
    client = exhibiting.app.test_client()
    with client.session_transaction() as session:
        session['role'] = 'ADMIN'          # even claiming admin must not help
        session['user_id'] = 1

    resp = client.post(path, json=payload)

    assert resp.status_code == 403, f'not locked down: {path} -> {resp.status_code}'


def test_exhibition_exits_rather_than_create_a_ghost_database(
        smartgallery_app, monkeypatch, capsys, tmp_path):
    """Pointed at a gallery that was never run normally, the mode must stop
    with an explanation instead of quietly making an empty database beside
    the real one."""
    never_used = tmp_path / "never_used"
    never_used.mkdir()
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)
    monkeypatch.setattr(smartgallery_app, "DATABASE_FILE",
                        str(never_used / "gallery_cache.sqlite"))

    with pytest.raises(SystemExit) as refused:
        smartgallery_app.check_exhibition_requirements()

    assert refused.value.code == 1, "a misconfigured exhibition launch was allowed"
    assert "Database Not Found" in capsys.readouterr().out, (
        "the refusal did not explain itself")
    leftovers = list(never_used.rglob("*.sqlite"))
    assert not leftovers, f"a ghost database was created anyway: {leftovers}"


def test_exhibition_exits_when_there_is_nothing_to_exhibit(
        smartgallery_app, monkeypatch, capsys, tmp_path):
    """A database with no public or shared collection would come up empty,
    which looks identical to a broken install from the visitor's side."""
    database = tmp_path / "empty_show.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE collections (name TEXT, type TEXT, "
                     "is_public INTEGER, shared_users TEXT)")
        conn.commit()
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)
    monkeypatch.setattr(smartgallery_app, "DATABASE_FILE", str(database))

    with pytest.raises(SystemExit) as refused:
        smartgallery_app.check_exhibition_requirements()

    assert refused.value.code == 1
    assert "No Exhibition Collections Found" in capsys.readouterr().out


def test_the_preflight_says_nothing_when_the_mode_is_off(
        smartgallery_app, monkeypatch, capsys):
    """The counterpart: a normal launch must not be pre-flighted at all, or
    every ordinary start would refuse on a fresh install."""
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    monkeypatch.setattr(smartgallery_app, "DATABASE_FILE",
                        os.path.join("definitely", "not", "here.sqlite"))

    smartgallery_app.check_exhibition_requirements()

    assert capsys.readouterr().out == "", "a normal launch ran the exhibition checks"


def test_startup_still_runs_the_preflight(gallery_tree):
    """The check is only worth anything if initialize_gallery still calls
    it before anything can create a database."""
    import ast

    gallery_init = next(
        (node for node in ast.walk(gallery_tree)
         if isinstance(node, ast.FunctionDef) and node.name == "initialize_gallery"),
        None)
    assert gallery_init is not None, "initialize_gallery is gone; this check is stale"

    called = {node.func.id for node in ast.walk(gallery_init)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}

    assert "check_exhibition_requirements" in called, (
        "initialize_gallery no longer pre-flights, so a misconfigured "
        "exhibition can create a ghost database again")
