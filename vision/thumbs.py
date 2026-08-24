"""The derived-image cache: what layouts need, rendered once per content.

Three variants, because gallery layouts need exactly three things:

- ``thumb`` -- longest side 512, aspect kept. The grid and masonry cell;
  a square cell is the browser's ``object-fit`` crop of this, not a
  separate file.
- ``preview`` -- longest side 1440, aspect kept. The lightbox image, so
  opening a picture does not decode a 50-megapixel original.
- an avatar -- a square crop around one detected face, for the people
  index. Keyed by face instance, not by file: one picture holds many
  faces.

Thumb and preview are keyed on `content_sha256`, so the same bytes never
render twice, a moved or renamed file keeps its cache, and the whole
directory is safe to delete -- it holds nothing that cannot be recomputed.

This module is the Pillow half, and takes callers who ALREADY HOLD
decoded pixels: face detection decodes every picture and sampled video
frame, so it caches on the way past rather than throwing them away for
something else to decode again. Rendering from a path is vision/derive.py,
which reaches libvips first and falls back here.

Writes are atomic -- a temp file in the same directory, then `os.replace`
-- so a killed process leaves no half-written image a browser would
receive as a broken picture forever.

`fit` is the resize and never enlarges; what that cost is in its own
docstring. `ImageOps.fit` squares the avatar (python-pillow/Pillow@bb1d8e8
src/PIL/ImageOps.py:518-563, crop to aspect then resize). The WebP writer
converts whatever mode arrives (src/PIL/WebPImagePlugin.py:152-155, 297),
so palette, CMYK and 16-bit frames need no handling here.
"""

from __future__ import annotations

import os
import pathlib

from PIL import Image, ImageOps

#: Longest side per raster variant, in pixels. Thumb serves grid cells
#: well under half its size on high-density screens; preview covers a
#: lightbox on ordinary displays without shipping originals.
EDGES = {"thumb": 512, "preview": 1440}

#: The cache's directory under a run's home (sg_web/home.py thumbs_dir);
#: named here so a migration that must reach the cache finds it by the
#: same word.
DIRNAME = "thumbs"

#: Avatar square side. A face crop this size stays sharp in a 96px circle.
AVATAR = 256

#: How much of the surrounding picture an avatar keeps around the face
#: box: the crop's side is the face's longer side times this. Two gives
#: hair, chin and some shoulder -- a person, not a passport stamp.
AVATAR_CONTEXT = 2.0

#: WebP quality. At these sizes the difference from higher settings is
#: bytes, not appearance.
QUALITY = 82

#: libwebp's speed/size dial, 0 fastest to 6 smallest. Pillow's default
#: for stills is 4 (python-pillow/Pillow src/PIL/WebPImagePlugin.py:294),
#: which on this workload is the single most expensive phase there is --
#: encoding cost the same 200-280 ms whether the source was 22 megapixels
#: or 0.03, because it depends on the encoder's effort and not on the
#: picture. Measured over the real library, `just bench thumbs-phases`:
#:
#:     method 0   171 ms/file   5.83 files/sec   132.5 KiB cached
#:     method 2   215 ms/file   4.65 files/sec   102.8 KiB
#:     method 4   368 ms/file   2.72 files/sec    99.4 KiB
#:
#: 2 buys 1.7x the throughput for 3.4% more disk. 0 buys a further 1.25x
#: for 29% more, which is the wrong trade for a cache that is written
#: once and read forever.
METHOD = 2


def path_for(cache_dir: pathlib.Path, sha: str, kind: str = "thumb") -> pathlib.Path:
    """Where this content's `kind` variant lives, existing or not. Fanned
    out by the first byte of the hash so no directory grows unbounded."""
    if kind not in EDGES:
        raise ValueError(f"{kind!r} is not a variant; EDGES in vision/thumbs.py is the vocabulary")
    suffix = "" if kind == "thumb" else f".{kind}"
    return cache_dir / sha[:2] / f"{sha}{suffix}.webp"


def avatar_path(cache_dir: pathlib.Path, face_id: int) -> pathlib.Path:
    return cache_dir / "avatar" / f"{face_id}.webp"


def fit(image: Image.Image, edge: int) -> Image.Image:
    """The image no larger than `edge` on its longest side.

    It never enlarges. `ImageOps.contain` does, and that cost more than
    anything else in the pipeline for small sources: a 200x150 animated
    WebP was blown up to 1440x1080 and encoded at 1.5 megapixels, 173 ms
    to invent pixels that were never in the file. The grid does not need
    them -- every cell is `object-fit: cover` (gallery.css:97-100), so the
    browser scales a small picture to fill its cell for free, and the
    lightbox is `object-fit: contain`.

    Returns the image itself when it already fits, so a caller can see
    that nothing was allocated. `reducing_gap=3.0` is Pillow's two-step
    resize -- an integer `reduce` then a resample -- documented as
    indistinguishable from fair resampling in most cases (Image.py
    :2352-2363).
    """
    if max(image.size) <= edge:
        return image
    scale = edge / max(image.size)
    size = (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale)))
    return image.resize(size, Image.Resampling.LANCZOS, reducing_gap=3.0)


def put(cache_dir: pathlib.Path, sha: str, image: Image.Image, kind: str = "thumb") -> pathlib.Path:
    """Cache one variant from already-decoded pixels; a hit costs a stat.

    The caller's image is not touched: `fit` either returns a new image
    or the same one, and this only reads it.
    """
    target = path_for(cache_dir, sha, kind)
    if target.exists():
        return target
    return write(target, fit(image, EDGES[kind]))


def put_all(cache_dir: pathlib.Path, sha: str, image: Image.Image) -> None:
    """Every content-keyed variant at once -- the byproduct call, for a
    producer holding pixels it would otherwise discard.

    The thumb comes off the PREVIEW, not off the caller's image a second
    time. Resizing 22 megapixels down to 512 costs 111 ms; resizing the
    1440 preview down to 512 costs 14 ms, and at 512 the difference
    between the two results is not visible. Reading EDGES largest-first
    is what makes that ordering a fact rather than a coincidence.
    """
    frame = image
    for kind in sorted(EDGES, key=lambda name: -EDGES[name]):
        frame = fit(frame, EDGES[kind])
        target = path_for(cache_dir, sha, kind)
        if not target.exists():
            write(target, frame)


def put_avatar(
    cache_dir: pathlib.Path, face_id: int, image: Image.Image, bbox: tuple[float, float, float, float]
) -> pathlib.Path:
    """Cache the square face crop for one detected face.

    `bbox` is the detection's normalized (x, y, w, h) in 0..1, the shape
    every backend reports (vision/faces.py) and `region` stores. The crop
    is a square of AVATAR_CONTEXT times the face around its centre,
    clamped to the frame; `fit` re-centres whatever clamping cut away.
    """
    target = avatar_path(cache_dir, face_id)
    if target.exists():
        return target
    x, y, w, h = bbox
    width, height = image.size
    side = max(w * width, h * height) * AVATAR_CONTEXT
    centre_x, centre_y = (x + w / 2) * width, (y + h / 2) * height
    left = max(0, round(centre_x - side / 2))
    top = max(0, round(centre_y - side / 2))
    right = min(width, round(centre_x + side / 2))
    bottom = min(height, round(centre_y + side / 2))
    face = image.crop((left, top, right, bottom))
    return write(target, ImageOps.fit(face, (AVATAR, AVATAR)))


def write(target: pathlib.Path, small: Image.Image) -> pathlib.Path:
    """One already-sized Pillow image, encoded and published atomically."""
    return _publish(target, lambda staging: small.save(staging, format="WEBP", quality=QUALITY, method=METHOD))


def write_bytes(target: pathlib.Path, blob: bytes) -> pathlib.Path:
    """The same, for an encoder that hands back bytes rather than writing
    them -- libvips does (vision/derive.py)."""
    return _publish(target, lambda staging: staging.write_bytes(blob))


def _publish(target: pathlib.Path, put) -> pathlib.Path:
    """Atomic, and safe under concurrent writers: the staging name is
    per-thread, so two requests racing to fill the same miss each write
    their own bytes and the second `os.replace` is a no-op in effect."""
    import threading

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f"{target.name}.{os.getpid()}-{threading.get_ident()}.tmp"
    put(staging)
    os.replace(staging, target)
    return target
