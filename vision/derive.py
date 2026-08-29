"""A file's derived images, by the cheapest correct route.

One question -- "cache the derivatives of this file" -- with two answers
behind it, because no single library reads everything this gallery
accepts.

libvips is the fast path and takes whatever it can read. It decodes and
resizes as one streaming operation using each format's own shrink-on-load,
so nothing materialises at full size. That matters most for PNG, which is
what an image generator writes and which Pillow cannot shrink during
decode at all.

`effort` is the same libwebp dial Pillow calls `method`, so thumbs.METHOD
governs both and 2 is the same choice for the same reason: it matches the
default's output size at a fraction of its cost. The gain here is the
decoder rather than stolen cores, so raising concurrency above this will
not fight it.

Pillow answers for the rest, and the rest is not small. This build of
libvips refuses Canon raws outright ("Old-style JPEG compression support
is not configured") and has no video loader, so raws keep going through
rawpy's embedded preview and video through PyAV. Callers already holding
decoded pixels -- face detection caches its frame on the way past -- have
nothing to gain from either and keep calling vision/thumbs directly.

Which route is taken is decided by trying, not by a table of suffixes. A
missing loader, a truncated file and a format that parses but will not
read all fail in the same place and all mean the same thing here.

Orientation is applied once, from the tag ingest recorded: `no_rotate=True`
tells libvips to leave the frame as stored, the way `user_flip=0` tells
LibRaw. Two libraries each helpfully turning a picture is how a portrait
ends up upside down.
"""

from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import threading
import typing

import pyvips

from vision import thumbs

_logger = logging.getLogger(__name__)


#: Priority between a browser waiting on a cell and the precache queue: the
#: queue stops starting pictures while a request is outstanding. An Event, not
#: a lock, so the request never waits and a running render is not interrupted.
class _Waiting:
    """How many people are blocked on a picture, and a gate that is open
    exactly when nobody is."""

    __slots__ = ("count", "lock", "nobody")

    def __init__(self) -> None:
        self.count = 0
        self.lock = threading.Lock()
        self.nobody = threading.Event()
        self.nobody.set()


_WAITING = _Waiting()

#: The longest the queue will stand aside for one picture. A marker that
#: leaks -- a request killed between the halves of `waited_on` -- must
#: cost the precache a pause, never the whole queue.
PATIENCE = 5.0


@contextlib.contextmanager
def waited_on():
    """Somebody is blocked on this render; the queue should stand aside.

    Counted rather than flagged: a grid is thirty of these at once, and
    the last one to finish is the one that decides when the queue may
    carry on.
    """
    with _WAITING.lock:
        _WAITING.count += 1
        _WAITING.nobody.clear()
    try:
        yield
    finally:
        with _WAITING.lock:
            _WAITING.count = max(0, _WAITING.count - 1)
            if not _WAITING.count:
                _WAITING.nobody.set()


def stand_aside(patience: float = PATIENCE) -> None:
    """Hold a speculative render back while a person is waiting.

    Bounded, so this can only ever slow the queue down and never stop
    it: past `patience` the picture is rendered anyway, which is exactly
    the behaviour there was before any of this.
    """
    _WAITING.nobody.wait(patience)


# pyvips logs every operation at INFO -- "reducev: 15 point mask" -- which is a
# dozen lines per thumbnail. That is a library's debugging channel rather than
# this program's output, so it is raised to WARNING and nothing else changes.
logging.getLogger("pyvips").setLevel(logging.WARNING)

#: pyvips builds every operation at runtime from libvips' own introspection and
#: ships neither stubs nor py.typed, so `rot`, `thumbnail_image` and
#: `webpsave_buffer` do not exist as far as a type checker is concerned.

#: A libvips image, deliberately not `pyvips.Image`: a named alias says
#: dynamic-on-purpose where a bare Any looks careless. `pyvips.Image.thumbnail`
#: and `pyvips.Error` below are real attributes and are used as themselves.
type Raster = typing.Any

#: EXIF orientation -> the turn libvips must make. db/oriented.TURNS is the same
#: mapping for Pillow and the two are NOT interchangeable: `rot` angles are
#: clockwise where Pillow's ROTATE_* are counter-clockwise, so 6 and 8 swap.
TURNS = {
    2: ("flip", "horizontal"),
    3: ("rot", "d180"),
    4: ("flip", "vertical"),
    5: ("transpose", ""),
    6: ("rot", "d90"),
    7: ("transverse", ""),
    8: ("rot", "d270"),
}


def upright(image: Raster, orientation: int | None) -> Raster:
    """A libvips image turned the way its tag asks.

    The counterpart of db/oriented.upright, which does this for Pillow.
    Orientation 1 returns the image untouched, as there.
    """
    what = TURNS.get(orientation or 1)
    if what is None:
        return image
    how, argument = what
    if how == "rot":
        return image.rot(argument)
    if how == "flip":
        return image.flip(argument)
    # The two diagonal mirrors libvips has no single operation for.
    if how == "transpose":
        return image.rot("d90").flip("horizontal")
    return image.rot("d270").flip("horizontal")


def fit(image: Raster, edge: int) -> Raster:
    """A libvips image no larger than `edge` on its longest side.

    `size='down'` is libvips' own spelling of never-enlarge, which is what
    vision/thumbs.fit does for Pillow. Returns the image itself when it
    already fits.
    """
    if max(image.width, image.height) <= edge:
        return image
    return image.thumbnail_image(edge, size="down")


def opened(path: pathlib.Path, want: int, orientation: int | None) -> Raster | None:
    """The file through libvips at no more than `want`, or None.

    None means libvips will not read it and the caller must take the
    Pillow route. Not an error: raws and video reach it every time.
    """
    try:
        loaded = pyvips.Image.thumbnail(os.fspath(path), want, size="down", no_rotate=True)
        # Materialised because a `thumbnail` pipeline reads sequentially and may
        # be consumed once: encoding it and then deriving the smaller variant
        # from the same object fails with "out of order read".

        # It beats reading the file twice, PNG having no shrink-on-load to make
        # a second read cheap, and at the preview edge this is the largest thing
        # held either way.
        return upright(loaded, orientation).copy_memory()
    except pyvips.Error as why:
        _logger.debug("%s: libvips will not read this, using Pillow: %s", path, why)
        return None


def put_all(cache_dir: pathlib.Path, sha: str, path: pathlib.Path, kind: str, orientation: int | None) -> None:
    """Cache every content-keyed variant of this file.

    Largest first, each smaller one taken off the one above it, which is
    the rule vision/thumbs.put_all follows for the same reason: resizing
    the preview costs a fraction of resizing the source again.
    """
    _put(cache_dir, sha, path, kind, orientation, list(thumbs.EDGES))


def put_one(cache_dir: pathlib.Path, sha: str, path: pathlib.Path, kind: str, orientation: int | None, variant: str):
    """Cache one variant -- what a browser asking for a single miss needs.

    It still decodes at the variant's own edge rather than the preview's,
    so a thumb costs a thumb.
    """
    _put(cache_dir, sha, path, kind, orientation, [variant])


def _put(
    cache_dir: pathlib.Path,
    sha: str,
    path: pathlib.Path,
    kind: str,
    orientation: int | None,
    variants: list[str],
) -> None:
    from db import oriented
    from vision import decode

    wanted = sorted(variants, key=lambda name: -thumbs.EDGES[name])
    if all(thumbs.path_for(cache_dir, sha, name).exists() for name in wanted):
        return
    want = thumbs.EDGES[wanted[0]]

    if kind != "video":
        frame = opened(path, want, orientation)
        if frame is not None:
            for name in wanted:
                frame = fit(frame, thumbs.EDGES[name])
                target = thumbs.path_for(cache_dir, sha, name)
                if not target.exists():
                    thumbs.write_bytes(target, frame.webpsave_buffer(Q=thumbs.QUALITY, effort=thumbs.METHOD))
            return

    picture = decode.poster(path) if kind == "video" else oriented.for_derivatives(path, want, orientation)
    if picture is None:
        raise ValueError(f"{path} has no decodable frame to render")
    for name in wanted:
        picture = thumbs.fit(picture, thumbs.EDGES[name])
        target = thumbs.path_for(cache_dir, sha, name)
        if not target.exists():
            thumbs.write(target, picture)
