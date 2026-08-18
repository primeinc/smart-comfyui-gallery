"""Thumbnail and waveform generation, executed for real.

These encoders had no functional coverage: the one test that mentioned
`create_thumbnail` monkeypatched it away. They write into a cache the rest
of the app trusts by existence alone -- `glob(f"{file_hash}.*")` means "this
file is done" -- so a half-written file at the final name is served as a
broken thumbnail forever. That is why every encoder writes to a `tmp_` path
and promotes it with os.replace, and why these tests check what is left
behind on failure as carefully as what is produced on success.
"""

from __future__ import annotations

import glob
import os

import pytest
from PIL import Image


@pytest.fixture
def cache_dir(smartgallery_app):
    os.makedirs(smartgallery_app.THUMBNAIL_CACHE_DIR, exist_ok=True)
    return smartgallery_app.THUMBNAIL_CACHE_DIR


def _tmp_leftovers(cache_dir):
    return glob.glob(os.path.join(cache_dir, "tmp_*"))


def test_static_image_thumbnail_is_a_real_downscaled_jpeg(smartgallery_app, cache_dir, tmp_path):
    src = str(tmp_path / "big.png")
    Image.new("RGB", (1200, 900), (10, 120, 200)).save(src)

    out = smartgallery_app.create_thumbnail(src, "t_static", "image")

    assert out
    assert os.path.isfile(out)
    with Image.open(out) as im:
        assert im.format == "JPEG"
        # Bounded by THUMBNAIL_WIDTH, aspect ratio preserved.
        assert im.size[0] <= smartgallery_app.THUMBNAIL_WIDTH
        assert im.size[0] / im.size[1] == pytest.approx(1200 / 900, rel=0.02)
    assert not _tmp_leftovers(cache_dir)


def test_animated_gif_keeps_its_frames(smartgallery_app, cache_dir, tmp_path):
    src = str(tmp_path / "anim.gif")
    frames = [Image.new("RGB", (400, 300), c)
              for c in ((255, 0, 0), (0, 255, 0), (0, 0, 255))]
    frames[0].save(src, save_all=True, append_images=frames[1:], duration=100, loop=0)

    out = smartgallery_app.create_thumbnail(src, "t_anim", "animated_image")

    assert out
    assert os.path.isfile(out)
    with Image.open(out) as im:
        assert im.format == "GIF"
        assert getattr(im, "is_animated", False)
        assert im.n_frames == 3
    assert not _tmp_leftovers(cache_dir)


def test_unreadable_source_returns_none_and_strands_nothing(smartgallery_app, cache_dir, tmp_path):
    src = str(tmp_path / "corrupt.png")
    with open(src, "wb") as fh:
        fh.write(b"not a png at all")

    assert smartgallery_app.create_thumbnail(src, "t_bad", "image") is None
    assert not _tmp_leftovers(cache_dir)
    assert not glob.glob(os.path.join(cache_dir, "t_bad.*")), (
        "a failed encode left a file at the final cache name, which the "
        "app's existence check would treat as a finished thumbnail")


def test_encoder_dying_mid_write_leaves_no_partial_thumbnail(
        smartgallery_app, cache_dir, tmp_path, monkeypatch):
    """The disk-full case this design exists for: a save that writes some
    bytes and then fails must not leave those bytes at the final name."""
    src = str(tmp_path / "ok.png")
    Image.new("RGB", (600, 400), (200, 50, 50)).save(src)

    real_save = Image.Image.save

    def dying_save(self, fp, *args, **kwargs):
        if isinstance(fp, str) and os.path.basename(fp).startswith("tmp_"):
            with open(fp, "wb") as fh:      # partial output, as a full disk gives
                fh.write(b"\xff\xd8\xff\xe0 truncated")
            raise OSError(28, "No space left on device")
        return real_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", dying_save)

    assert smartgallery_app.create_thumbnail(src, "t_full", "image") is None
    assert not glob.glob(os.path.join(cache_dir, "t_full.*")), (
        "partial bytes landed at the final cache name")
    assert not _tmp_leftovers(cache_dir), "the failed temp file was not cleaned up"


def test_retry_after_a_failed_encode_succeeds(smartgallery_app, cache_dir, tmp_path, monkeypatch):
    """Because nothing was left behind, the next attempt is a clean one --
    the whole point of not poisoning the cache."""
    src = str(tmp_path / "retry.png")
    Image.new("RGB", (600, 400), (30, 200, 90)).save(src)

    real_save = Image.Image.save
    monkeypatch.setattr(Image.Image, "save", lambda self, fp, *a, **k: (_ for _ in ()).throw(
        OSError(28, "No space left on device")))
    assert smartgallery_app.create_thumbnail(src, "t_retry", "image") is None

    monkeypatch.setattr(Image.Image, "save", real_save)
    out = smartgallery_app.create_thumbnail(src, "t_retry", "image")
    assert out
    assert os.path.isfile(out)
    with Image.open(out) as im:
        assert im.format == "JPEG"


def test_waveform_promotes_only_a_complete_render(smartgallery_app, cache_dir, tmp_path):
    """ffmpeg is killed by the 20s timeout on pathological input; a partial
    PNG must never be promoted, because create_waveform returns any existing
    file at the final name without re-checking it."""
    if not smartgallery_app.FFPROBE_EXECUTABLE_PATH:
        pytest.skip("ffprobe not available")
    if not smartgallery_app.GENERATE_WAVEFORMS:
        pytest.skip("waveform generation disabled by configuration")
    src = str(tmp_path / "not-audio.bin")
    with open(src, "wb") as fh:
        fh.write(b"\x00" * 2048)

    out = smartgallery_app.create_waveform(src, "t_wave", "audio")

    assert out is None, "a non-audio file produced a waveform"
    assert not _tmp_leftovers(cache_dir)
    assert not glob.glob(os.path.join(cache_dir, "t_wave_wave*"))
