"""One image with an absurd header must not take the process down.

Pillow refuses to decode images beyond a pixel ceiling, because the cost of
decoding is set by the header rather than by the file: a few kilobytes can
declare 65535x65535 and ask for about 13 GB.

That ceiling defaults to 89 megapixels, which a legitimate 16384x16384
upscale (268 Mpx) exceeds -- so `create_thumbnail` cleared it outright with
`Image.MAX_IMAGE_PIXELS = None`. The need was real and the fix was too
broad in two ways: it removed the ceiling rather than raising it, and the
setting is global, so the first thumbnail anyone made switched the guard
off for every later image in that process, including the ones served to
visitors.

There is now a bounded ceiling instead, set once at import: big upscales
decode, absurd headers are refused, and a refusal costs that file its
thumbnail rather than costing the gallery.
"""

from __future__ import annotations

import os

from PIL import Image

import smartgallery


def test_the_guard_is_on():
    """The regression: this was None, meaning no ceiling at all."""
    assert Image.MAX_IMAGE_PIXELS is not None, "Pillow's decompression guard is disabled process-wide"
    assert Image.MAX_IMAGE_PIXELS == smartgallery.MAX_DECODED_PIXELS


def test_a_large_upscale_is_still_allowed():
    """The reason the guard was turned off in the first place. A 16k x 16k
    upscale is 268 Mpx; refusing those would be a worse bug than the one
    being fixed, so the ceiling is pinned above it."""
    upscale_16k = 16384 * 16384

    assert upscale_16k < smartgallery.MAX_DECODED_PIXELS, (
        f"the ceiling ({smartgallery.MAX_DECODED_PIXELS}) would refuse a 16384x16384 upscale ({upscale_16k})"
    )


def test_an_absurd_header_is_refused():
    """Pillow raises above twice the ceiling. A header claiming 65535 square
    asks for ~13 GB and must not be attempted."""
    bogus = 65535 * 65535

    assert bogus > 2 * smartgallery.MAX_DECODED_PIXELS, "a 65535x65535 header would still be decoded"


def test_making_a_thumbnail_does_not_switch_the_guard_off(tmp_path):
    """The part that made it process-wide: the ceiling was cleared inside
    create_thumbnail, so one ordinary thumbnail disarmed it for everything
    that came after."""
    source = tmp_path / "ordinary.png"
    Image.new("RGB", (64, 48), (10, 20, 30)).save(source)
    os.makedirs(smartgallery.THUMBNAIL_CACHE_DIR, exist_ok=True)

    before = Image.MAX_IMAGE_PIXELS
    result = smartgallery.create_thumbnail(str(source), "ceilingprobe", "image")

    assert before == Image.MAX_IMAGE_PIXELS, "making a thumbnail changed the process-wide pixel ceiling"
    assert result and os.path.exists(result), "the ordinary thumbnail stopped working"
    os.remove(result)


def test_an_oversized_declaration_fails_that_file_only(tmp_path, monkeypatch):
    """A refused image costs its own thumbnail and nothing else: the caller
    already treats a raised decode as "no thumbnail"."""
    # A real image, with the ceiling lowered under it rather than a 13 GB
    # file on disk.
    source = tmp_path / "big.png"
    Image.new("RGB", (400, 400), (1, 2, 3)).save(source)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)  # 400x400 is 160000
    os.makedirs(smartgallery.THUMBNAIL_CACHE_DIR, exist_ok=True)

    result = smartgallery.create_thumbnail(str(source), "ceilingprobe2", "image")

    assert result is None, "an image over the ceiling produced a thumbnail anyway"

    # And the next, ordinary file still works once the ceiling is normal.
    monkeypatch.undo()
    ok = tmp_path / "small.png"
    Image.new("RGB", (32, 32), (4, 5, 6)).save(ok)
    made = smartgallery.create_thumbnail(str(ok), "ceilingprobe3", "image")
    assert made
    assert os.path.exists(made)
    os.remove(made)
