"""Opening a picture the way it is meant to be seen.

A phone stores the sensor's frame and writes a tag saying which way up it
goes. Nothing that looks at pixels may skip that step. Measured on a 105-photo
face dataset, 14 were stored sideways -- and a face detector shown a sideways
face finds a wall instead: the false detections in that run were a patch of
wallpaper and a patch of hair, which then became their own "people" on the
People page.

So this is the only way a model-facing reader opens a file. It is not a
convenience; an unrotated frame is a different picture, and every embedding
taken from one is a measurement of something nobody has ever seen.

**Cheap by construction, in three ways.**

The tag is read from the database when the caller has it. `capture.orientation`
was recorded at ingest, so the common path never re-parses EXIF at all.

Orientation 1 -- the overwhelming majority -- returns the image untouched. No
copy, no work.

The rest is `Image.transpose`, which for the four right-angle cases is a
memory reorder rather than a resample: no decode of anything new, no
interpolation, no quality lost. `ImageOps.exif_transpose` does the same thing
but re-reads EXIF, rewrites the tag and copies even when there is nothing to
do, which on a scan is once per file for no reason.

The mapping is upstream's, not invented here
(python-pillow/Pillow@bb1d8e8 src/PIL/ImageOps.py:705-713).
"""

from __future__ import annotations

from PIL import Image, ImageOps

#: EXIF orientation -> what to do about it. Straight from Pillow's own
#: `exif_transpose`; 1 is absent because 1 means "already upright".
TURNS = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def upright(image: Image.Image, orientation: int | None = None) -> Image.Image:
    """The image as it is meant to be seen.

    `orientation` is the tag if the caller already knows it -- from
    `capture.orientation`, which ingest recorded. Passing it skips the EXIF
    parse entirely, which is the whole cost on an already-upright file.
    """
    if orientation is None:
        orientation = image.getexif().get(274, 1)
    turn = TURNS.get(int(orientation or 1))
    return image if turn is None else image.transpose(turn)


def open_upright(path, orientation: int | None = None) -> Image.Image:
    """Open a file and turn it the right way up.

    Through the decoder door, so HEIC, JPEG XL and the RAW family arrive
    here exactly as a JPEG does -- the suffix decides the decoder, never
    the caller."""
    from vision import decode

    return upright(decode.open_still(path), orientation)


def orientation_of(conn, file_id: int) -> int | None:
    """What ingest recorded for this file, or None when nothing has looked.

    So a job over ten thousand pictures asks the database rather than
    re-opening and re-parsing every file's EXIF to learn something already
    stored in a column.

    A capture row with no tag means ingest looked and the file carries
    none: 1, no parse. NO row means ingest has not run, and unknown is not
    upright -- None makes `upright` read the file's own EXIF, so a freshly
    scanned phone photo is served the right way up before its first
    ingest. Collapsing both to 1 served sideways thumbnails for exactly
    the window between scan and ingest, found by a real browser.
    """
    row = conn.execute("SELECT orientation FROM capture WHERE file_id = ?", (file_id,)).fetchone()
    if row is None:
        return None
    return int(row[0]) if row[0] else 1


def for_model(conn, file_id: int, path) -> Image.Image:
    """The picture as a model should see it: upright, using the stored tag.

    The one call a detector, an embedder or a thumbnailer should make. Taking
    the tag from the row rather than the file is what keeps it cheap; taking
    it at all is what keeps it correct.
    """
    return open_upright(path, orientation_of(conn, file_id))


def is_turned(orientation: int | None) -> bool:
    """Whether this orientation changes which way up the picture is."""
    return int(orientation or 1) in TURNS


__all__ = ["TURNS", "for_model", "is_turned", "open_upright", "orientation_of", "upright"]
# ImageOps is imported for the reference in the module docstring's comparison
# and to keep the dependency explicit for anyone reaching for it instead.
_ = ImageOps
