"""Who can read which comments.

A comment carries a `target_audience`: `public`, `internal` (staff notes),
or `user:<id>` (a private message to one person). The read endpoint takes
a `client_uuid` query parameter, which is the shape of bug worth checking
after finding that the login route trusted a caller-supplied identity --
if this one did the same, any visitor could read staff notes and other
people's private messages by asking for them.

It does not: the parameter is only consulted when there is no session, and
in every mode where sessions are required the endpoint refuses anonymous
callers outright. The mode that does allow them treats the caller as the
local admin, who may see everything anyway. These tests hold that shut.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Exhibition rather than --force-login: the callers below are a CUSTOMER
# and a GUEST, and --force-login admits only ADMIN, MANAGER and STAFF to
# the interface, so neither could reach a picture there to read comments
# on. Exhibition is where those roles exist. Both modes require a session,
# so the anonymous case is unchanged.
_ARGV = "'--exhibition', '--admin-pass', 'correct-horse-battery'"


def _spawn(argv, script, env_extra, timeout=300):
    env = dict(os.environ, ENABLE_AI_DAM="false", AI_DAM_AUTO_PROVISION="false",
               **env_extra)
    full = f"import sys\nsys.argv = ['smartgallery.py'{argv}]\n" + script
    return subprocess.run([sys.executable, "-c", full], cwd=_ROOT, env=env,
                          capture_output=True, text=True, timeout=timeout)


def _run(script, env_extra, timeout=300):
    """Seed with no flags, then assert under --exhibition.

    Exhibition refuses to start without a database, and says so itself:
    "You must run the standard gallery AT LEAST ONCE before using
    Exhibition Mode." That is the documented order, so the seeding gets its
    own run exactly as a real install would do it, and only the assertions
    see the exhibition process.
    """
    seeded = _spawn("", _SEED, env_extra, timeout)
    if seeded.returncode != 0 or "SEEDED" not in seeded.stdout:
        return seeded
    return _spawn(", " + _ARGV, _CLIENT + script, env_extra, timeout)


@pytest.fixture()
def gallery_env(tmp_path):
    gallery = tmp_path / "gallery"
    output = tmp_path / "output"
    gallery.mkdir()
    output.mkdir()
    return {"BASE_OUTPUT_PATH": str(output), "BASE_SMARTGALLERY_PATH": str(gallery)}


_SEED = """
import os
import smartgallery as sg
os.makedirs(sg.SQLITE_CACHE_DIR, exist_ok=True)
sg.initialize_gallery()

conn = sg.get_db_connection()
conn.execute("INSERT OR REPLACE INTO files (id, path, mtime, name, type, size) "
             "VALUES ('f1', '/x/a.png', 1.0, 'a.png', 'image', 1)")
rows = [
    ('41', 'Alice', 'a public remark', 'public'),
    ('admin', 'Staff', 'internal staff note', 'internal'),
    ('admin', 'Staff', 'private note for user 41', 'user:41'),
    ('admin', 'Staff', 'private note for user 77', 'user:77'),
    ('77', 'Bob', 'bob wrote this', 'public'),
]
# In a public album, so a visitor may see the picture at all: reading
# comments now refuses a file the caller has no access to, and these
# tests are about which comments they then get, not about that.
conn.execute("INSERT INTO collections (name, type, is_public) "
             "VALUES ('Shown', 'user_album', 1)")
_album = conn.execute("SELECT id FROM collections WHERE name='Shown'").fetchone()[0]
conn.execute("INSERT INTO collection_files (collection_id, file_id) "
             "VALUES (?, 'f1')", (_album,))
for uuid_, author, text, audience in rows:
    conn.execute("INSERT INTO file_comments (file_id, client_uuid, author_name, "
                 "comment_text, target_audience, created_at) "
                 "VALUES ('f1', ?, ?, ?, ?, 1.0)", (uuid_, author, text, audience))
conn.commit(); conn.close()
print('SEEDED')
"""

# Runs in the second process, which imports the module again under
# --exhibition and so needs its own client.
_CLIENT = """
import smartgallery as sg
client = sg.app.test_client()

def texts(resp):
    body = resp.get_json()
    return sorted(c['comment_text'] for c in (body.get('comments') or []))
"""


def test_a_user_cannot_read_notes_addressed_to_someone_else(gallery_env):
    """The parameter is ignored in favour of the session, so asking for
    another user's id does not hand over their messages."""
    script = """
with client.session_transaction() as s:
    s['user_id'] = 41
    s['role'] = 'CUSTOMER'

seen = texts(client.get('/galleryout/api/exhibition/comments'
                        '?file_id=f1&client_uuid=77'))
assert 'private note for user 77' not in seen, (
    f"a user read another person's private note: {seen}")
assert 'internal staff note' not in seen, f'staff notes leaked: {seen}'
assert 'private note for user 41' in seen, f'own message missing: {seen}'
assert 'a public remark' in seen, seen
print('SCOPED')
"""
    proc = _run(script, gallery_env)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "SCOPED" in proc.stdout


def test_staff_see_everything(gallery_env):
    script = """
with client.session_transaction() as s:
    s['user_id'] = 1
    s['role'] = 'ADMIN'

seen = texts(client.get('/galleryout/api/exhibition/comments?file_id=f1'))
for expected in ('internal staff note', 'private note for user 41',
                 'private note for user 77', 'a public remark'):
    assert expected in seen, f'{expected!r} hidden from staff: {seen}'
print('FULL')
"""
    proc = _run(script, gallery_env)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "FULL" in proc.stdout


def test_an_anonymous_caller_is_refused_entirely(gallery_env):
    """With logins in play the query parameter is never even reached."""
    script = """
resp = client.get('/galleryout/api/exhibition/comments?file_id=f1&client_uuid=41')
assert resp.status_code == 401, resp.status_code
body = resp.get_data(as_text=True)
assert 'private note' not in body and 'internal staff note' not in body
print('REFUSED')
"""
    proc = _run(script, gallery_env)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "REFUSED" in proc.stdout


def test_a_guest_sees_only_public_and_their_own(gallery_env):
    script = """
with client.session_transaction() as s:
    s['user_id'] = 'guest_deadbeefdeadbeef'
    s['role'] = 'GUEST'

seen = texts(client.get('/galleryout/api/exhibition/comments'
                        '?file_id=f1&client_uuid=admin'))
assert seen == ['a public remark', 'bob wrote this'], (
    f'a guest saw more than the public comments: {seen}')
print('PUBLIC_ONLY')
"""
    proc = _run(script, gallery_env)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "PUBLIC_ONLY" in proc.stdout
