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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

#: Longest side per raster variant, in pixels. Thumb serves grid cells
#: well under half its size on high-density screens; preview covers a
#: lightbox on ordinary displays without shipping originals.
EDGES = {"thumb": 512, "preview": 1440}

#: The file kinds a raster variant can be MADE from. Audio and documents
#: have no picture to take, and the routes that serve variants refuse
#: them outright -- so a surface that asks for one anyway gets a 404 and
#: draws a broken image. Named here, beside what a derivative IS, so a
#: surface can ask before pointing at one instead of finding out from
#: the network.
PICTURED = ("image", "animated_image", "video")

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
    from PIL import Image

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
    from PIL import ImageOps

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


def asset_url(sha: str | None, slug: str, variant: str = "thumb", medium: str | None = None) -> str | None:
    """Where a surface should point a picture's `src`, or None.

    The content-addressed asset when the bytes have been hashed, and the
    slug route when they have not.

    ONE function, because the choice is a fact about the data and not a
    preference of whichever template is rendering: a grid cell, a
    filmstrip frame, a rail preview and a compare tray thumbnail must all
    reach the same conclusion, or some of them go on paying for a
    database round trip nobody can see.

    `/thumbs/<shard>/<name>` mirrors the on-disk layout exactly, so the
    route that serves it is a path join with no query, no slug and no
    connection. The address names the BYTES, so it can be cached for a
    year -- the same bargain PhotoPrism and Immich make.

    **None when the file's medium has no picture to take.** `medium` is
    the file's kind -- image, video, audio, document -- and the two
    unpictured ones are why this returns an option rather than a string.
    Hashing is what mints an asset address, and audio and documents get
    hashed like everything else, so an address existed for them and the
    routes behind it refused: measured over a mixed eight-file library,
    three of eight grid cells answered 404 -- as UNCAUGHT exceptions with
    tracebacks, because rendering is where the kind was finally
    consulted. A surface must be able to ask before it points, and this
    is where the question is answered for all of them at once.

    `medium` is optional only so a caller that already knows it holds a
    picture need not say so. Passing it is the safe default.
    """
    if variant not in EDGES:
        raise ValueError(f"{variant!r} is not a variant; EDGES in vision/thumbs.py is the vocabulary")
    if medium is not None and medium not in PICTURED:
        return None
    if not sha:
        # Not yet hashed -- ingest has not reached it. The slug route can
        # still answer, at the cost this exists to avoid, which is the
        # right trade for a file nobody has finished reading.
        return f"/{variant}/{slug}"
    suffix = "" if variant == "thumb" else f".{variant}"
    return f"/thumbs/{sha[:2]}/{sha}{suffix}.webp"


def address(rows: list[dict]) -> list[dict]:
    """Point every row of an answer at its thumbnail, in place.

    The one step between "the ResultSet returned these" and "a template
    renders them", and it exists because four surfaces had each written
    `/thumb/<slug>` into their own markup. That address is a route with
    a slug lookup behind it -- a database connection per picture -- and
    the grid stopped paying it while the person, folder, album and
    artifact pages went on doing so, because nothing named the step they
    were all skipping.

    In place and returning the same list, so a caller can wrap the
    expression it already had. `sha` and `kind` come out of the row the
    ResultSet already read, so this is arithmetic and not a query.
    """
    for row in rows:
        row["thumb"] = asset_url(row.get("sha"), row["slug"], medium=row.get("kind"))
    return rows
