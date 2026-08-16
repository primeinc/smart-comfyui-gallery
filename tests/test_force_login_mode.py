"""`--force-login`: the flag people rely on when the gallery is reachable.

This is the switch someone flips before putting the gallery on a LAN or
behind a tunnel. Everything about that decision rests on two properties:
an anonymous visitor gets the login page instead of the library, and the
destructive APIs refuse an unauthenticated caller instead of running.

Both fail silently if broken -- the gallery still works perfectly for the
owner, who is logged in and would never notice. That is exactly the shape
of bug worth a test.

Each test runs a fresh interpreter, because the mode is decided from argv
at import time.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Routes that change or destroy media. An unauthenticated caller must not
# reach any of them while the flag is on.
_DESTRUCTIVE = """[
    ('/galleryout/delete_batch', {'file_ids': ['x']}),
    ('/galleryout/move_batch', {'file_ids': ['x'], 'destination_folder': '_root_'}),
    ('/galleryout/copy_batch', {'file_ids': ['x'], 'destination_folder': '_root_'}),
    ('/galleryout/delete_folder/_root_', {}),
    ('/galleryout/rename_file/x', {'new_name': 'y.png'}),
    ('/galleryout/create_folder', {'folder_name': 'x', 'parent_key': '_root_'}),
    ('/galleryout/prepare_batch_zip', {'file_ids': ['x']}),
    ('/galleryout/favorite_batch', {'file_ids': ['x'], 'status': True}),
]"""


def _run(script, env_extra, timeout=300):
    env = dict(os.environ, ENABLE_AI_DAM="false", AI_DAM_AUTO_PROVISION="false",
               **env_extra)
    return subprocess.run([sys.executable, "-c", script], cwd=_ROOT, env=env,
                          capture_output=True, text=True, timeout=timeout)


@pytest.fixture()
def gallery_env(tmp_path):
    gallery = tmp_path / "gallery"
    output = tmp_path / "output"
    gallery.mkdir()
    output.mkdir()
    return {"BASE_OUTPUT_PATH": str(output), "BASE_SMARTGALLERY_PATH": str(gallery)}


def test_anonymous_visitor_gets_the_login_page_not_the_library(gallery_env):
    script = """
import os, sys
sys.argv = ['smartgallery.py', '--force-login']
import smartgallery as sg
os.makedirs(sg.SQLITE_CACHE_DIR, exist_ok=True)
sg.init_db()
assert sg.FORCE_LOGIN is True, 'the flag did not take effect'

client = sg.app.test_client()
for path in ('/galleryout/', '/galleryout/view/_root_'):
    resp = client.get(path, follow_redirects=True)
    body = resp.get_data(as_text=True)
    assert resp.status_code == 403, f'{path} answered {resp.status_code}'
    # The denial has to be legible -- an owner who locked themselves out
    # needs to see that a password is wanted, not a blank page.
    assert 'password' in body.lower() or 'login' in body.lower(), (
        f'{path} denied without saying why')
    # And it must not carry the gallery itself.
    assert 'lightbox-toolbar' not in body, f'{path} leaked the gallery interface'
    assert 'gallery-item' not in body, f'{path} leaked file listings'
print('GATED')
"""
    proc = _run(script, gallery_env)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "GATED" in proc.stdout


def test_destructive_apis_refuse_an_anonymous_caller(gallery_env):
    script = """
import os, sys
sys.argv = ['smartgallery.py', '--force-login']
import smartgallery as sg
os.makedirs(sg.SQLITE_CACHE_DIR, exist_ok=True)
sg.init_db()
client = sg.app.test_client()

bad = []
for path, payload in %s:
    r = client.post(path, json=payload)
    if r.status_code not in (401, 403):
        bad.append((path, r.status_code))
assert not bad, f'reachable without logging in: {bad}'
print('REFUSED')
""" % _DESTRUCTIVE
    proc = _run(script, gallery_env)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "REFUSED" in proc.stdout


def test_a_logged_in_customer_still_cannot_use_the_management_apis(gallery_env):
    """Authentication is not authorisation: a CUSTOMER account exists to
    view and rate, and must not be able to delete the library."""
    script = """
import os, sys
sys.argv = ['smartgallery.py', '--force-login']
import smartgallery as sg
os.makedirs(sg.SQLITE_CACHE_DIR, exist_ok=True)
sg.init_db()
client = sg.app.test_client()
with client.session_transaction() as s:
    s['user_id'] = 5
    s['role'] = 'CUSTOMER'

bad = []
for path, payload in %s:
    r = client.post(path, json=payload)
    if r.status_code != 403:
        bad.append((path, r.status_code))
assert not bad, f'a CUSTOMER reached management routes: {bad}'
print('FORBIDDEN')
""" % _DESTRUCTIVE
    proc = _run(script, gallery_env)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "FORBIDDEN" in proc.stdout


def test_without_the_flag_a_local_install_stays_usable(gallery_env):
    """The counterpart: the default single-user install must NOT demand a
    login, or the flag would be meaningless and every owner locked out."""
    script = """
import os, sys
sys.argv = ['smartgallery.py']
import smartgallery as sg
os.makedirs(sg.SQLITE_CACHE_DIR, exist_ok=True)
sg.init_db()
assert sg.FORCE_LOGIN is False
r = sg.app.test_client().post('/galleryout/favorite_batch',
                              json={'file_ids': ['nope'], 'status': True})
assert r.status_code == 200, r.status_code
print('OPEN')
"""
    proc = _run(script, gallery_env)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "OPEN" in proc.stdout
