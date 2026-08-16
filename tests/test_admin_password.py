"""`--admin-pass` / `ADMIN_PASSWORD`: the credential everything else rests on.

Setting an admin password is what turns a single-user gallery into one
that can face a network -- it even switches `--force-login` on by itself.
The login path had no tests, and its failure modes are the quiet kind: a
password that is stored readable, a comparison that leaks whether a
username exists, or a crafted request that crashes the one route standing
between a stranger and the library.

Each test runs a fresh interpreter, because the password is read from argv
at import time.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PASSWORD = "correct-horse-battery"


def _run(script, env_extra, argv_extra="", timeout=300):
    env = dict(os.environ, ENABLE_AI_DAM="false", AI_DAM_AUTO_PROVISION="false",
               **env_extra)
    full = f"import sys\nsys.argv = ['smartgallery.py'{argv_extra}]\n" + script
    return subprocess.run([sys.executable, "-c", full], cwd=_ROOT, env=env,
                          capture_output=True, text=True, timeout=timeout)


@pytest.fixture()
def gallery_env(tmp_path):
    gallery = tmp_path / "gallery"
    output = tmp_path / "output"
    gallery.mkdir()
    output.mkdir()
    return {"BASE_OUTPUT_PATH": str(output), "BASE_SMARTGALLERY_PATH": str(gallery)}


_BOOT = """
import os
import smartgallery as sg
os.makedirs(sg.SQLITE_CACHE_DIR, exist_ok=True)
sg.initialize_gallery()
client = sg.app.test_client()
"""


def test_setting_a_password_switches_login_on_by_itself(gallery_env):
    """Nobody should have to remember to add --force-login as well."""
    script = _BOOT + """
assert sg.FORCE_LOGIN is True, 'an admin password did not enforce login'
assert sg.ADMIN_CONFIG_MISSING is False, 'a configured password still read as missing'

# Properly configured, the gallery answers with its login page rather than
# the lockdown notice -- but still without any of the library.
r = client.get('/galleryout/view/_root_')
body = r.get_data(as_text=True)
assert r.status_code == 200, r.status_code
assert 'password' in body.lower(), 'no password field on the login page'
assert 'lightbox-toolbar' not in body, 'the gallery interface leaked before login'
assert 'gallery-item' not in body, 'file listings leaked before login'
print('ENFORCED')
"""
    proc = _run(script, gallery_env, f", '--admin-pass', '{_PASSWORD}'")
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "ENFORCED" in proc.stdout


def test_the_password_is_never_stored_readable(gallery_env):
    """The row must hold an Argon2id hash, not the password."""
    script = _BOOT + """
conn = sg.get_db_connection()
row = conn.execute("SELECT password FROM users WHERE username = 'admin'").fetchone()
conn.close()
assert row is not None, 'no admin user was created'
stored = row[0]
assert %r not in str(stored), 'the admin password is stored in readable form'
assert str(stored).startswith('$argon2'), f'not an argon2 hash: {str(stored)[:24]}'
print('HASHED')
""" % _PASSWORD
    proc = _run(script, gallery_env, f", '--admin-pass', '{_PASSWORD}'")
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "HASHED" in proc.stdout


def test_the_right_password_logs_in_and_a_wrong_one_does_not(gallery_env):
    script = _BOOT + """
ok = client.post('/galleryout/login',
                 json={'username': 'admin', 'password': %r})
assert ok.status_code == 200, ok.status_code
assert ok.get_json().get('status') == 'success', ok.get_json()

fresh = sg.app.test_client()
bad = fresh.post('/galleryout/login',
                 json={'username': 'admin', 'password': 'not-the-password'})
assert bad.get_json().get('status') != 'success', 'a wrong password was accepted'
with fresh.session_transaction() as s:
    assert not s.get('user_id'), 'a failed login still created a session'
print('AUTHED')
""" % _PASSWORD
    proc = _run(script, gallery_env, f", '--admin-pass', '{_PASSWORD}'")
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "AUTHED" in proc.stdout


def test_a_crafted_password_cannot_crash_the_login_route(gallery_env):
    """The route is reachable by anyone who can reach the server, so a JSON
    int, list or non-ASCII string must be refused, never a 500."""
    script = _BOOT + """
payloads = [123, ['a', 'b'], {'x': 1}, None, 'pässwörd-ünïcode', 'x' * 5000]
bad = []
for value in payloads:
    r = client.post('/galleryout/login',
                    json={'username': 'admin', 'password': value})
    if r.status_code >= 500:
        bad.append((repr(value)[:30], r.status_code))
assert not bad, f'login crashed on: {bad}'
print('ROBUST')
"""
    proc = _run(script, gallery_env, f", '--admin-pass', '{_PASSWORD}'")
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "ROBUST" in proc.stdout


def test_an_unknown_username_is_refused_without_revealing_itself(gallery_env):
    """A missing user still runs one verification against a decoy, so the
    response cannot be told apart from a wrong password."""
    script = _BOOT + """
missing = client.post('/galleryout/login',
                      json={'username': 'nobody-here', 'password': 'whatever'})
wrong = client.post('/galleryout/login',
                    json={'username': 'admin', 'password': 'whatever'})
assert missing.status_code == wrong.status_code, (
    f'status differs: {missing.status_code} vs {wrong.status_code}')
assert missing.get_json().get('status') == wrong.get_json().get('status')
assert missing.get_json().get('message') == wrong.get_json().get('message'), (
    'the message reveals whether the account exists')
print('OPAQUE')
"""
    proc = _run(script, gallery_env, f", '--admin-pass', '{_PASSWORD}'")
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "OPAQUE" in proc.stdout


def test_a_short_password_is_flagged(gallery_env):
    script = _BOOT + """
assert sg.ADMIN_PASS_TOO_SHORT, 'a 4-character admin password was not flagged'
print('FLAGGED')
"""
    proc = _run(script, gallery_env, ", '--admin-pass', 'shrt'")
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "FLAGGED" in proc.stdout


def test_the_environment_variable_works_like_the_flag(gallery_env):
    """CONFIGURATION.md documents ADMIN_PASSWORD as equivalent."""
    script = _BOOT + """
assert sg.FORCE_LOGIN is True, 'ADMIN_PASSWORD did not enforce login'
r = client.post('/galleryout/login', json={'username': 'admin', 'password': %r})
assert r.get_json().get('status') == 'success', r.get_json()
print('ENVOK')
""" % _PASSWORD
    proc = _run(script, dict(gallery_env, ADMIN_PASSWORD=_PASSWORD))
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "ENVOK" in proc.stdout
