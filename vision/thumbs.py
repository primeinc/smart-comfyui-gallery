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

Two producers write into it, because decoding is the expensive part and
throwing decoded pixels away to re-decode them later makes no sense:

- Jobs that already hold a decoded frame (face detection decodes every
  picture and sampled video frame) cache on the way past, gated by the
  `thumbnail_precache` setting.
- The serving layer caches a frame it decoded on first request, for
  files no job has touched yet.

Writes are atomic -- a temp file in the same directory, then `os.replace`
-- so a killed process leaves no half-written image a browser would
receive as a broken picture forever.

`ImageOps.contain` is the resize: aspect-preserving to the box's longest
side in both directions (python-pillow/Pillow@bb1d8e8 src/PIL/ImageOps.py:
272-300), so a tiny source is enlarged to grid size rather than rendered
as a speck. `ImageOps.fit` squares the avatar (:518-563, crop to aspect
then resize). The WebP writer converts whatever mode arrives
(src/PIL/WebPImagePlugin.py:152-155, 297), so palette, CMYK and 16-bit
frames need no handling here.
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


def path_for(cache_dir: pathlib.Path, sha: str, kind: str = "thumb") -> pathlib.Path:
    """Where this content's `kind` variant lives, existing or not. Fanned
    out by the first byte of the hash so no directory grows unbounded."""
    if kind not in EDGES:
        raise ValueError(f"{kind!r} is not a variant; EDGES in vision/thumbs.py is the vocabulary")
    suffix = "" if kind == "thumb" else f".{kind}"
    return cache_dir / sha[:2] / f"{sha}{suffix}.webp"


def avatar_path(cache_dir: pathlib.Path, face_id: int) -> pathlib.Path:
    return cache_dir / "avatar" / f"{face_id}.webp"


def put(cache_dir: pathlib.Path, sha: str, image: Image.Image, kind: str = "thumb") -> pathlib.Path:
    """Cache one variant from already-decoded pixels; a hit costs a stat.

    The caller's image is not touched -- `contain` returns a new image.
    """
    target = path_for(cache_dir, sha, kind)
    if target.exists():
        return target
    return _write(target, ImageOps.contain(image, (EDGES[kind], EDGES[kind])))


def put_all(cache_dir: pathlib.Path, sha: str, image: Image.Image) -> None:
    """Every content-keyed variant at once -- the byproduct call, for a
    producer holding pixels it would otherwise discard."""
    for kind in EDGES:
        put(cache_dir, sha, image, kind)


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
    return _write(target, ImageOps.fit(face, (AVATAR, AVATAR)))


def _write(target: pathlib.Path, small: Image.Image) -> pathlib.Path:
    """Atomic, and safe under concurrent writers: the staging name is
    per-thread, so two requests racing to fill the same miss each write
    their own bytes and the second `os.replace` is a no-op in effect."""
    import threading

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f"{target.name}.{os.getpid()}-{threading.get_ident()}.tmp"
    small.save(staging, format="WEBP", quality=QUALITY)
    os.replace(staging, target)
    return target
