"""Exhibition mode boots, locks down, and shows only what was shared.

`--exhibition` is a documented launch mode with its own templates, its own
security posture, and a pre-flight check that exits rather than create a
"ghost" database. Nothing had ever started the app in it: the suite runs
the default mode, so every exhibition-only branch was unexercised.

The mode exists to put a gallery in front of people who are not the owner,
so the properties worth holding are the ones that fail quietly: the
destructive management APIs must be refused outright, and the pre-flight
check must stop a misconfigured launch instead of silently creating an
empty database beside the real one.

Each test runs a fresh interpreter, because the mode is decided from argv
at import time.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(script, env_extra, timeout=300):
    env = dict(os.environ, ENABLE_AI_DAM="false", AI_DAM_AUTO_PROVISION="false",
               **env_extra)
    return subprocess.run([sys.executable, "-c", script], cwd=_ROOT, env=env,
                          capture_output=True, text=True, timeout=timeout)


@pytest.fixture()
def prepared_gallery(tmp_path):
    """A gallery that has been run once normally, as the mode requires."""
    gallery = tmp_path / "gallery"
    output = tmp_path / "output"
    gallery.mkdir()
    output.mkdir()
    setup = """
import os, sys
sys.argv = ['smartgallery.py']
import smartgallery as sg
os.makedirs(sg.SQLITE_CACHE_DIR, exist_ok=True)
sg.init_db()
conn = sg.get_db_connection()
conn.execute("INSERT INTO collections (name, type, color, is_public, created_at) "
             "VALUES ('Public Show', 'user', '#fff', 1, 1000.0)")
conn.commit(); conn.close()
print('PREPARED')
"""
    env = {"BASE_OUTPUT_PATH": str(output), "BASE_SMARTGALLERY_PATH": str(gallery)}
    proc = _run(setup, env)
    assert "PREPARED" in proc.stdout, f"{proc.stdout}\n{proc.stderr}"
    return env


def test_exhibition_boots_and_hides_the_underlying_folders(prepared_gallery):
    """The posture of the mode: the entrance redirects to what was shared,
    and the raw folder tree is refused. A visitor is shown a curated
    collection, never the library behind it."""
    script = """
import sys
sys.argv = ['smartgallery.py', '--exhibition']
import smartgallery as sg
assert sg.IS_EXHIBITION_MODE is True, 'the flag did not take effect'
client = sg.app.test_client()

entrance = client.get('/galleryout/')
assert entrance.status_code in (301, 302), entrance.status_code

browse = client.get('/galleryout/view/_root_')
assert browse.status_code == 403, (
    f'raw folder browsing answered {browse.status_code} in exhibition mode')
print('BOOTED')
"""
    proc = _run(script, prepared_gallery)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "BOOTED" in proc.stdout


def test_exhibition_refuses_the_destructive_apis(prepared_gallery):
    """Every management route must answer 403 in this mode, whatever the
    session claims -- these are the routes that delete and move media."""
    script = """
import sys
sys.argv = ['smartgallery.py', '--exhibition']
import smartgallery as sg
client = sg.app.test_client()
with client.session_transaction() as s:
    s['role'] = 'ADMIN'          # even claiming admin must not help
    s['user_id'] = 1

checks = [
    ('/galleryout/delete_batch', {'file_ids': ['x']}),
    ('/galleryout/move_batch', {'file_ids': ['x'], 'destination_folder': '_root_'}),
    ('/galleryout/copy_batch', {'file_ids': ['x'], 'destination_folder': '_root_'}),
    ('/galleryout/delete_folder/_root_', {}),
    ('/galleryout/rename_file/x', {'new_name': 'y.png'}),
    ('/galleryout/prepare_batch_zip', {'file_ids': ['x']}),
    ('/galleryout/create_folder', {'folder_name': 'x', 'parent_key': '_root_'}),
]
bad = []
for path, payload in checks:
    r = client.post(path, json=payload)
    if r.status_code != 403:
        bad.append((path, r.status_code))
assert not bad, f'not locked down: {bad}'
print('LOCKED')
"""
    proc = _run(script, prepared_gallery)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "LOCKED" in proc.stdout


def test_exhibition_exits_rather_than_create_a_ghost_database(tmp_path):
    """Pointed at a gallery that was never run normally, the mode must
    stop with an explanation instead of quietly making an empty database
    beside the real one."""
    empty = tmp_path / "never_used"
    output = tmp_path / "out"
    empty.mkdir()
    output.mkdir()
    script = """
import sys
sys.argv = ['smartgallery.py', '--exhibition']
import smartgallery as sg
# The pre-flight runs in initialize_gallery(), not at import.
sg.initialize_gallery()
print('SHOULD NOT REACH HERE')
"""
    proc = _run(script, {"BASE_OUTPUT_PATH": str(output),
                         "BASE_SMARTGALLERY_PATH": str(empty)})

    assert proc.returncode != 0, "a misconfigured exhibition launch was allowed"
    assert "SHOULD NOT REACH HERE" not in proc.stdout
    assert "Database Not Found" in proc.stdout, (
        f"the refusal did not explain itself:\n{proc.stdout}")
    leftovers = [p for p in empty.rglob("*.sqlite")]
    assert not leftovers, f"a ghost database was created anyway: {leftovers}"
