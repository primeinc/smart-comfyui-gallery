"""Losing the thumbnail cache folder must not cost the rest of the session.

`.thumbnails_cache` is created once, at startup, and every writer then
assumes it is still there. It carries the word "cache" in its name, and it
lives wherever the person pointed BASE_SMARTGALLERY_PATH -- often a synced
folder or a second drive. Disk cleaners look for exactly that.

Once it was gone, every thumbnail failed for the remaining life of the
process. The only sign was one console line per file:

    ERROR (Pillow): Thumbnail failed for pic.png: [Errno 2] No such file
    or directory: '...\\.thumbnails_cache\\tmp_13940c72....jpeg'

which names a temporary file nobody has heard of and never mentions the
folder that went missing. Restarting fixed it, and there was nothing to
suggest that.

It is put back now, on the one condition that matters: only inside a
gallery root that still exists. Recreating it blindly is worse than the
bug -- when the root is an unplugged drive or a share that stopped
answering, the mount point is an empty directory that is very much
writable, and thumbnails would accumulate on the wrong filesystem where
nothing will ever look for them again.
"""

from __future__ import annotations

import os
import shutil

import pytest
from PIL import Image


@pytest.fixture
def picture(smartgallery_app, tmp_path):
    path = tmp_path / "recover.png"
    Image.new("RGB", (64, 64), (30, 90, 140)).save(path)
    return str(path)


def _cache_dir(smartgallery_app):
    return smartgallery_app.THUMBNAIL_CACHE_DIR


def test_a_thumbnail_is_written_when_the_cache_is_there(smartgallery_app, picture):
    """Control. Everything below is about the folder being absent, so this
    pins that the ordinary path works and that the assertions are looking
    in the right place."""
    os.makedirs(_cache_dir(smartgallery_app), exist_ok=True)

    made = smartgallery_app.create_thumbnail(picture, "ctrl_present", "image")

    assert made and os.path.isfile(made), made


def test_the_folder_is_put_back_when_it_has_gone(smartgallery_app, picture):
    """The bug: from here on, every thumbnail failed until a restart."""
    cache = _cache_dir(smartgallery_app)
    shutil.rmtree(cache, ignore_errors=True)
    assert not os.path.isdir(cache), "the folder should be gone for this test"

    made = smartgallery_app.create_thumbnail(picture, "recovered", "image")

    assert os.path.isdir(cache), "the cache folder was not put back"
    assert made and os.path.isfile(made), made


def test_it_is_not_recreated_outside_an_existing_gallery(smartgallery_app, picture, tmp_path, monkeypatch):
    """The condition that makes putting it back safe rather than reckless.

    An unplugged drive leaves a mount point that is an ordinary, writable,
    empty directory. Building the cache there would silently move
    everyone's thumbnails onto the wrong filesystem, and the gallery would
    never look at them again."""
    absent_root = tmp_path / "unplugged"
    monkeypatch.setattr(smartgallery_app, "BASE_SMARTGALLERY_PATH", str(absent_root))
    monkeypatch.setattr(smartgallery_app, "THUMBNAIL_CACHE_DIR", str(absent_root / ".thumbnails_cache"))

    made = smartgallery_app.create_thumbnail(picture, "unplugged", "image")

    assert made is None, made
    assert not absent_root.exists(), "a gallery tree was created on top of a missing root"


def test_the_helper_reports_rather_than_raising(smartgallery_app, tmp_path, monkeypatch):
    """It is called from the scan's worker path, so it has to answer with a
    boolean whatever the filesystem says -- an exception there costs the
    file its thumbnail at best and the scan at worst."""
    assert smartgallery_app.ensure_thumbnail_cache_dir() is True

    monkeypatch.setattr(smartgallery_app, "BASE_SMARTGALLERY_PATH", str(tmp_path / "nope"))
    monkeypatch.setattr(smartgallery_app, "THUMBNAIL_CACHE_DIR", str(tmp_path / "nope" / ".thumbnails_cache"))

    assert smartgallery_app.ensure_thumbnail_cache_dir() is False


def test_waveforms_stop_instead_of_failing_per_file(smartgallery_app, tmp_path, monkeypatch):
    """The other writer into the same folder. It returns None for a missing
    cache rather than letting ffmpeg write to a path that is not there."""
    monkeypatch.setattr(smartgallery_app, "GENERATE_WAVEFORMS", True)
    monkeypatch.setattr(smartgallery_app, "FFPROBE_EXECUTABLE_PATH", "ffprobe")
    monkeypatch.setattr(smartgallery_app, "BASE_SMARTGALLERY_PATH", str(tmp_path / "gone"))
    monkeypatch.setattr(smartgallery_app, "THUMBNAIL_CACHE_DIR", str(tmp_path / "gone" / ".thumbnails_cache"))

    assert smartgallery_app.create_waveform("whatever.mp3", "h", "audio") is None
    assert not (tmp_path / "gone").exists()
