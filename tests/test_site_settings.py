"""Site setting: thumbnail generation toggle.

The requirement it binds: turning generation off must save the compute
without harming the cache — existing thumbnails keep serving, nothing is
deleted, and re-enabling only generates what is actually missing (cache
keys are md5(path+mtime), so only changed files ever regenerate).
"""

import hashlib
import os

import pytest
from PIL import Image


@pytest.fixture
def sg(smartgallery_app):
    yield smartgallery_app
    with smartgallery_app.get_db_connection() as conn:
        conn.execute("DELETE FROM ai_metadata WHERE key = 'thumbnail_generation'")
        conn.execute("DELETE FROM files WHERE id LIKE 'tset:%'")
        conn.commit()
    smartgallery_app._THUMBNAIL_SETTING_CACHE['value'] = None


def _set(sg, enabled):
    resp = sg.app.test_client().post(
        '/galleryout/api/site_settings', json={'thumbnail_generation': enabled}
    )
    assert resp.get_json()['status'] == 'success'


def _make_png(tmp_path, name="t.png"):
    path = tmp_path / name
    Image.new("RGB", (32, 32), (10, 120, 200)).save(path)
    return str(path).replace('\\', '/')


def test_default_is_enabled(sg):
    sg._THUMBNAIL_SETTING_CACHE['value'] = None
    assert sg.thumbnail_generation_enabled() is True


def test_toggle_roundtrip_via_api(sg):
    client = sg.app.test_client()

    _set(sg, False)
    assert sg.thumbnail_generation_enabled() is False
    assert client.get('/galleryout/api/site_settings').get_json()['thumbnail_generation'] is False

    _set(sg, True)
    assert sg.thumbnail_generation_enabled() is True


def test_scan_skips_thumbnail_when_disabled(sg, tmp_path, monkeypatch):
    path = _make_png(tmp_path)
    calls = []
    monkeypatch.setattr(sg, 'create_thumbnail', lambda *a, **_k: calls.append(a))

    _set(sg, False)
    assert sg.process_single_file(path) is not None
    assert calls == []

    _set(sg, True)
    assert sg.process_single_file(path) is not None
    assert len(calls) == 1


def test_serve_thumbnail_falls_back_to_original_when_disabled(sg, tmp_path):
    # Arrange: an indexed image with no cached thumbnail.
    path = _make_png(tmp_path, "orig.png")
    mtime = os.path.getmtime(path)
    file_id = hashlib.md5(path.encode()).hexdigest()
    with sg.get_db_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO files (id, path, mtime, name, type) VALUES (?, ?, ?, 'orig.png', 'image')",
            (file_id, path, mtime),
        )
        conn.commit()

    _set(sg, False)
    try:
        resp = sg.app.test_client().get(f'/galleryout/thumbnail/{file_id}')

        # Assert: the ORIGINAL bytes come back and no thumbnail was created.
        assert resp.status_code == 200
        assert resp.data == open(path, 'rb').read()
        import glob as _glob
        file_hash = hashlib.md5((path + str(mtime)).encode()).hexdigest()
        assert _glob.glob(os.path.join(sg.THUMBNAIL_CACHE_DIR, f"{file_hash}.*")) == []
    finally:
        with sg.get_db_connection() as conn:
            conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
            conn.commit()


def test_toggle_never_touches_cached_thumbnails(sg):
    # Arrange: a pre-existing cached thumbnail.
    os.makedirs(sg.THUMBNAIL_CACHE_DIR, exist_ok=True)
    cached = os.path.join(sg.THUMBNAIL_CACHE_DIR, "deadbeef00.jpeg")
    with open(cached, 'wb') as f:
        f.write(b"cached-thumb")

    try:
        # Act: toggle off and back on.
        _set(sg, False)
        _set(sg, True)

        # Assert: the cache file is untouched.
        assert open(cached, 'rb').read() == b"cached-thumb"
    finally:
        os.remove(cached)
