"""`--enable-guest-login`: passwordless access that must stay anonymous.

Guest login exists so a visitor can rate and comment without an account,
and a returning guest may present their previous id so their own ratings
stay theirs. That id arrives in the request body, which is the whole
danger: ownership of comments and ratings is decided by comparing the
session id against a row's `client_uuid`.

Before the fix a guest could log in -- no password -- claiming any
identity at all, including a real account's user_id, and inherit that
account's comments and ratings. Verified by deleting another user's
comment through the API.

The line drawn is GUESSABLE versus not. Account ids are small integers and
the admin comments as the literal 'admin'; those may never be claimed. The
two accepted shapes each carry 64+ bits of entropy, so presenting one is
itself evidence of having been issued it: `guest_<hex>` as minted by this
server, and the RFC-4122 UUID a browser generates for itself with
crypto.randomUUID (both templates do this and store it under the same key
the login sends, so rejecting that shape would silently orphan every
existing visitor's ratings). Anything else is discarded and a fresh id
issued.

Each test runs a fresh interpreter, because the flags come from argv at
import time.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ARGV = ("'--enable-guest-login', '--force-login', "
         "'--admin-pass', 'correct-horse-battery'")


def _run(script, env_extra, timeout=300):
    env = dict(os.environ, ENABLE_AI_DAM="false", AI_DAM_AUTO_PROVISION="false",
               **env_extra)
    full = f"import sys\nsys.argv = ['smartgallery.py', {_ARGV}]\n" + script
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


def test_a_guest_cannot_claim_a_real_accounts_identity(gallery_env):
    """The regression: claiming user_id 41 used to hand a stranger that
    account's comments."""
    script = _BOOT + """
conn = sg.get_db_connection()
conn.execute("INSERT OR REPLACE INTO files (id, path, mtime, name, type, size) "
             "VALUES ('f1', '/x/a.png', 1.0, 'a.png', 'image', 1)")
conn.execute("INSERT INTO file_comments (file_id, client_uuid, author_name, "
             "comment_text, target_audience, created_at) "
             "VALUES ('f1', '41', 'RealUser', 'private words', 'public', 1.0)")
comment_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
conn.commit(); conn.close()

issued = client.post('/galleryout/login',
                     json={'username': 'guest', 'provided_uuid': '41'})
identity = issued.get_json()['client_uuid']
assert identity != '41', f'the server handed out the claimed identity: {identity}'
assert identity.startswith('guest_'), identity

resp = client.post('/galleryout/api/exhibition/delete_comment',
                   json={'comment_id': comment_id})
assert resp.status_code == 403, f'a guest deleted another account comment ({resp.status_code})'

conn = sg.get_db_connection()
alive = conn.execute('SELECT 1 FROM file_comments WHERE id = ?',
                     (comment_id,)).fetchone() is not None
conn.close()
assert alive, "the victim's comment was destroyed"
print('BLOCKED')
"""
    proc = _run(script, gallery_env)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "BLOCKED" in proc.stdout


def test_a_guest_cannot_overwrite_an_accounts_rating(gallery_env):
    """The same identity check protects ratings, which are aggregated: a
    guest claiming account 41 would have replaced that account's vote and
    moved the average.

    Note what is NOT claimed here. A guest id is a bearer token -- 64 bits
    from secrets.token_hex(8) -- so presenting a well-formed one legitimately
    grants that guest identity; that IS the continuity feature. The line
    that matters is between guessable account ids and unguessable guest
    ones.
    """
    script = _BOOT + """
conn = sg.get_db_connection()
conn.execute("INSERT OR REPLACE INTO files (id, path, mtime, name, type, size) "
             "VALUES ('f2', '/x/b.png', 1.0, 'b.png', 'image', 1)")
conn.execute("INSERT INTO file_ratings (file_id, client_uuid, rating, created_at) "
             "VALUES ('f2', '41', 5, 1.0)")
conn.commit(); conn.close()

client.post('/galleryout/login', json={'username': 'guest', 'provided_uuid': '41'})
client.post('/galleryout/api/exhibition/rate', json={'file_id': 'f2', 'rating': 1})

conn = sg.get_db_connection()
rows = [(r[0], r[1]) for r in conn.execute(
    "SELECT client_uuid, rating FROM file_ratings WHERE file_id = 'f2'").fetchall()]
conn.close()
victim = [r for r in rows if r[0] == '41']
assert victim and victim[0][1] == 5, f"the account's rating was overwritten: {rows}"
print('SAFE')
"""
    proc = _run(script, gallery_env)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "SAFE" in proc.stdout


@pytest.mark.parametrize("claimed", [
    "41", "admin", "1", "guest", "GUEST_ABC", "guest_", "guest_zzzz",
    "guest_dead beef", "../guest_deadbeef", "guest_deadbeef' OR '1'='1",
    "3f2b1c4d-aaaa-bbbb", "not-a-uuid-at-all",
])
def test_guessable_identities_are_never_accepted(gallery_env, claimed):
    script = _BOOT + """
issued = client.post('/galleryout/login',
                     json={'username': 'guest', 'provided_uuid': %r})
identity = issued.get_json()['client_uuid']
assert identity != %r, 'a malformed identity was accepted verbatim'
assert identity.startswith('guest_'), identity
print('MINTED')
""" % (claimed, claimed)
    proc = _run(script, gallery_env)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "MINTED" in proc.stdout


def test_a_browser_generated_uuid_is_honoured(gallery_env):
    """The exhibition and main templates mint their own identity with
    crypto.randomUUID when they have never been issued one, and store it
    under the same key the guest login sends. Rejecting that shape would
    silently orphan every existing visitor's ratings, so a well-formed
    UUID -- unguessable, and never colliding with an integer account id --
    is accepted as-is."""
    script = _BOOT + """
existing = '3f2b1c4d-9e7a-4b21-8f6c-1a2b3c4d5e6f'
issued = client.post('/galleryout/login',
                     json={'username': 'guest', 'provided_uuid': existing})
identity = issued.get_json()['client_uuid']
assert identity == existing, (
    f'a browser-generated identity was discarded: {identity}')
print('HONOURED')
"""
    proc = _run(script, gallery_env)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "HONOURED" in proc.stdout


def test_a_returning_guest_keeps_their_own_id(gallery_env):
    """The feature still has to work: a guest who comes back with the id
    this server minted keeps it, so their ratings remain theirs."""
    script = _BOOT + """
first = client.post('/galleryout/login',
                    json={'username': 'guest'}).get_json()['client_uuid']
assert first.startswith('guest_'), first

again = client.post('/galleryout/login',
                    json={'username': 'guest', 'provided_uuid': first}).get_json()['client_uuid']
assert again == first, f'a returning guest lost their identity: {first} -> {again}'
print('KEPT')
"""
    proc = _run(script, gallery_env)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "KEPT" in proc.stdout


def test_a_guest_is_still_refused_the_management_apis(gallery_env):
    script = _BOOT + """
client.post('/galleryout/login', json={'username': 'guest'})
bad = []
for path, payload in [
    ('/galleryout/delete_batch', {'file_ids': ['x']}),
    ('/galleryout/delete_folder/_root_', {}),
    ('/galleryout/rename_file/x', {'new_name': 'y.png'}),
]:
    r = client.post(path, json=payload)
    if r.status_code != 403:
        bad.append((path, r.status_code))
assert not bad, f'a guest reached management routes: {bad}'
print('FORBIDDEN')
"""
    proc = _run(script, gallery_env)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "FORBIDDEN" in proc.stdout


def test_guest_login_is_refused_when_the_flag_is_off(gallery_env):
    """Without --enable-guest-login the passwordless path must not exist."""
    env = dict(os.environ, ENABLE_AI_DAM="false", AI_DAM_AUTO_PROVISION="false",
               **gallery_env)
    script = ("import sys\n"
              "sys.argv = ['smartgallery.py', '--force-login', "
              "'--admin-pass', 'correct-horse-battery']\n" + _BOOT + """
resp = client.post('/galleryout/login', json={'username': 'guest'})
assert resp.get_json().get('status') != 'success', (
    'guest login worked without the flag')
with client.session_transaction() as s:
    assert not s.get('user_id'), 'a session was created anyway'
print('OFF')
""")
    proc = subprocess.run([sys.executable, "-c", script], cwd=_ROOT, env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "OFF" in proc.stdout
