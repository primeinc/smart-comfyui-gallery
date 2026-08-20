"""Camera EXIF, read into rows rather than left in the file.

`metaparse` reads the five tags generation tools abuse as carriers -- Make,
Model, Software, UserComment, MakerNote. Nothing has ever read the tags a
camera writes about the photograph: when the shutter opened, at what
aperture, through which lens, standing where. So `capture` had a schema and
no producer, and every claim about a places page was a claim about data that
did not exist.

Three conversions here are the whole reason this cannot be a dict copy.

**Time.** DateTimeOriginal is a wall clock with no zone. OffsetTimeOriginal
carries the zone when the camera bothered to write one, and most do not. An
absent offset is stored as NULL and the wall clock is read as UTC -- not
because it is UTC, but because that is the only reading that gives the same
number on every machine. NULL therefore means "this is a wall clock, not an
instant", and anything that renders it must say so rather than convert it.

**Position.** GPS is three rationals and a letter: degrees, minutes,
seconds, and N/S/E/W. The letter carries the sign, and altitude has a
reference byte of its own where 1 means below sea level. A reader that takes
the degrees and ignores the reference puts half the planet in the wrong
hemisphere.

**Numbers.** EXIF stores f/1.4 and 1/250s as pairs of integers. They reach
the database as REAL, once, here -- rather than as strings that every caller
re-parses slightly differently.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
from dataclasses import dataclass, field
from typing import NamedTuple

from PIL import ExifTags, Image

from metaparse.containers import decode_user_comment

from .exif_labels import label_for

#: Tags that become columns on `capture`, so they are not also long tail.
_CLAIMED = {
    ExifTags.Base.DateTimeOriginal,
    ExifTags.Base.OffsetTimeOriginal,
    ExifTags.Base.ISOSpeedRatings,
    ExifTags.Base.RecommendedExposureIndex,
    ExifTags.Base.FNumber,
    ExifTags.Base.ExposureTime,
    ExifTags.Base.FocalLength,
    ExifTags.Base.FocalLengthIn35mmFilm,
    ExifTags.Base.Orientation,
    ExifTags.Base.Make,
    ExifTags.Base.Model,
    ExifTags.Base.LensMake,
    ExifTags.Base.LensModel,
}

_GPS_CLAIMED = {
    ExifTags.GPS.GPSLatitude,
    ExifTags.GPS.GPSLatitudeRef,
    ExifTags.GPS.GPSLongitude,
    ExifTags.GPS.GPSLongitudeRef,
    ExifTags.GPS.GPSAltitude,
    ExifTags.GPS.GPSAltitudeRef,
    # Byte offsets to the sub-IFDs. They describe the file's layout, not the
    # photograph, and stored as parameters they read as searchable facts
    # ("GPSInfo = 256") that mean nothing outside this one file's bytes.
    ExifTags.IFD.Exif,
    ExifTags.IFD.GPSInfo,
    ExifTags.IFD.Interop,
}
_CLAIMED |= _GPS_CLAIMED

#: Tags whose value is a payload behind an 8-byte character-code prefix,
#: not a plain string (EXIF 2.32 §4.6.5 gives UserComment, GPSProcessingMethod
#: and GPSAreaInformation the same encoding).
_PREFIXED_TEXT = {
    ExifTags.Base.UserComment,
    ExifTags.GPS.GPSProcessingMethod,
    ExifTags.GPS.GPSAreaInformation,
}


@dataclass
class Capture:
    """What one photograph says about how it was taken."""

    captured_at: float | None = None
    tz_offset_min: int | None = None
    iso: int | None = None
    f_number: float | None = None
    exposure_time: float | None = None
    focal_length: float | None = None
    focal_35mm: float | None = None
    orientation: int | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    gps_alt: float | None = None
    camera: str | None = None
    lens: str | None = None
    #: (key, value_text, value_num) for every tag that is not a column.
    params: list[tuple[str, str, float | None]] = field(default_factory=list)
    #: (slot, payload) for tags that are binary and stay binary.
    binaries: list[tuple[str, bytes]] = field(default_factory=list)
    #: Tags the camera wrote with no value in them: a rational over zero, or
    #: a string that is only spaces and NULs. Absent from the database on
    #: purpose, and kept apart from `homeless` because "there is no value"
    #: and "we cannot store this value" are different problems and only the
    #: second is a defect. Measured on a real library, folding them together
    #: put 28 tag names in `homeless` -- all of them blank padding -- which
    #: is enough noise to hide the one entry that would have meant something.
    unrecorded: list[str] = field(default_factory=list)
    #: Tags with no home at all. Empty is the contract.
    homeless: list[tuple[str, str]] = field(default_factory=list)
    #: Why the file could not be opened at all, when it could not. Distinct
    #: from every field above being empty, which means the file was read and
    #: had nothing to say.
    unreadable: str | None = None

    @property
    def is_empty(self) -> bool:
        return not (self.params or self.binaries or self.homeless or self.captured_at)


def _number(value) -> float | None:
    """EXIF rationals, ints and numeric strings as one float.

    A rational with a zero denominator is how a body writes "not recorded",
    and it does not raise -- Pillow evaluates 0/0 to NaN. Storing that as a
    measurement is worse than storing nothing: NaN compares false against
    everything including itself, so a range facet silently drops the row and
    an equality test can never match it.
    """
    if value is None or isinstance(value, (bytes, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return number if math.isfinite(number) else None


def _text(value) -> str | None:
    """A tag as text, or None when it is not text.

    Bytes are decoded only when they really are a string: strictly valid
    UTF-8 with no control characters once the trailing NUL padding EXIF adds
    is removed. Forcing the rest through `errors="ignore"` turned a maker
    note into mojibake and truncated `GPSVersionID` from four bytes to two,
    both of which read afterwards as data rather than as damage.
    """
    if isinstance(value, str):
        return value.strip("\x00 ").strip() or None
    if isinstance(value, bytes):
        try:
            decoded = value.decode("utf-8")
        except UnicodeDecodeError:
            return None
        decoded = decoded.rstrip("\x00").strip()
        if not decoded or any(ch < " " and ch not in "\t\n\r" for ch in decoded):
            return None
        return decoded
    return None


def _offset_minutes(value) -> int | None:
    """`+02:00` / `-05:30` to minutes east of UTC."""
    text = _text(value)
    if not text or text[0] not in "+-" or ":" not in text:
        return None
    sign = -1 if text[0] == "-" else 1
    hours, _, minutes = text[1:].partition(":")
    try:
        return sign * (int(hours) * 60 + int(minutes))
    except ValueError:
        return None


def _timestamp(value, offset_min) -> float | None:
    """`YYYY:MM:DD HH:MM:SS` to epoch seconds.

    Read as UTC when the file carries no offset. See the module docstring:
    the alternative reads a different instant on every machine.
    """
    text = _text(value)
    if not text:
        return None
    try:
        naive = dt.datetime.strptime(text[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    zone = dt.timezone(dt.timedelta(minutes=offset_min)) if offset_min is not None else dt.timezone.utc
    return naive.replace(tzinfo=zone).timestamp()


def _degrees(value, ref) -> float | None:
    """Degrees/minutes/seconds plus a hemisphere letter to signed decimal.

    The tag is `Count => 3` (refs/exiftool/exiftool/lib/Image/ExifTool/GPS.pm:79)
    and bodies write fewer. Indexing all three unconditionally raised
    IndexError out of `read` and took the whole ingest pass with it, on one
    photograph. Upstream reads "1-3 decimal numbers" and treats the absent
    places as zero -- `$d + (($m || 0) + ($s || 0)/60) / 60` (GPS.pm:594) --
    so degrees alone is a coordinate, not damage.
    """
    if not isinstance(value, (tuple, list)) or not value:
        return None
    parts = [_number(part) for part in value[:3]] + [None, None]
    if parts[0] is None:
        return None
    degrees = parts[0]
    degrees += (parts[1] or 0.0) / 60.0
    degrees += (parts[2] or 0.0) / 3600.0
    if (_text(ref) or "").upper() in ("S", "W"):
        degrees = -degrees
    return degrees


def _camera_name(make, model) -> str | None:
    """`Make Model`, without repeating a make the model already carries.

    A camera body has no bytes to hash, so its name is its whole identity,
    and two manufacturers using the same model name would otherwise become
    one row. That is why the make is joined on at all.

    The test is whether the model's first word is one of the make's words,
    not whether the model starts with the whole make string. Bodies write
    the legal entity in Make and the brand in Model, so a prefix test leaves
    `NIKON CORPORATION NIKON D2X` and `EASTMAN KODAK COMPANY KODAK C310
    DIGITAL CAMERA` -- both observed in a real library, along with Pentax
    doing the same. `Xiaomi` + `Mi 9 Lite` correctly stays joined, because
    `Mi` is not a word of `Xiaomi`.
    """
    make, model = _text(make), _text(model)
    if not model:
        return make
    if not make:
        return model
    first = model.split()[0].lower()
    if first in {word.lower() for word in make.split()}:
        return model
    return f"{make} {model}"


def _scalar(value):
    """A tag value as (text, number), or None when it has no scalar form."""
    number = _number(value)
    if number is not None and not isinstance(value, (tuple, list)):
        text = _text(value) or str(value)
        return text, number
    text = _text(value)
    if text is not None:
        return text, None
    if isinstance(value, (tuple, list)):
        scalars = [_scalar(part) for part in value]
        if scalars and all(part is not None for part in scalars):
            return ", ".join(part[0] for part in scalars), None
    return None


def _is_blank(value) -> bool:
    """The tag is present but holds nothing.

    Cameras pad fixed-width fields rather than omitting them, so a library
    is full of `ImageDescription` that is thirty spaces and `UserComment`
    that is a run of NULs. Measured on a real library: 52 blank descriptions,
    69 blank comments, 11 blank copyrights.
    """
    if isinstance(value, str):
        return not value.strip("\x00 \t\r\n")
    if isinstance(value, bytes):
        return not value.strip(b"\x00 \t\r\n")
    return False


class Held(NamedTuple):
    """Where one tag's value belongs."""

    kind: str  # text | binary | unrecorded
    text: str | None = None
    number: float | None = None
    payload: bytes | None = None


def _tag_value(tag, value) -> Held | None:
    """One tag, routed to the storage that can actually hold it.

    Binary stays binary. `file_param` holds text and numbers; a maker note is
    a proprietary blob that no amount of decoding turns into either, and the
    schema already has somewhere for exactly this -- `blob.payload_bin`, with
    `file_blob` recording which slot it came from and whether anything has
    understood it yet.
    """
    if tag in _PREFIXED_TEXT:
        decoded = decode_user_comment(value)
        if decoded:
            return Held("text", text=decoded)
        return Held("unrecorded")
    if _is_blank(value):
        # Checked before the binary fallback, so a field of NULs is recorded
        # as an empty tag rather than filed in the blob store as a payload
        # nobody has decoded yet.
        return Held("unrecorded")
    label = label_for(tag, value)
    if label is not None:
        # Both columns, because they answer different questions: the phrase is
        # what a person searches for and reads, the code is what stays
        # comparable. Storing only the number makes "Flash = 89" a fact nobody
        # can use; storing only the phrase throws away the ordering.
        return Held("text", text=label, number=float(value))
    scalar = _scalar(value)
    if scalar is not None:
        return Held("text", text=scalar[0], number=scalar[1])
    if isinstance(value, bytes) and value:
        return Held("binary", payload=value)
    if not isinstance(value, (str, tuple, list)) and _number(value) is None:
        # A number-shaped tag that yielded no finite number: the body wrote a
        # rational over zero, which is EXIF for "not recorded".
        return Held("unrecorded")
    return None


def read(path) -> Capture:
    """Every camera tag in one file, as columns plus a long tail.

    A file that cannot be opened is reported, not raised. One truncated JPEG
    in a library used to end the whole ingest pass: `Image.open` raises
    `UnidentifiedImageError(OSError)` (refs/python-pillow/Pillow/src/PIL/
    __init__.py:78), `ValueError`, or `DecompressionBombError`, which is a
    bare Exception (Image.py:79) and so escapes an `except OSError`. Nothing
    between here and the caller caught any of them.
    """
    out = Capture()
    try:
        return _read(path, out)
    except (OSError, ValueError, Image.DecompressionBombError) as problem:
        out.unreadable = f"{type(problem).__name__}: {problem}"
        return out


def _read(path, out: Capture) -> Capture:
    with Image.open(path) as image:
        exif = image.getexif()
        if not exif:
            return out
        photo = exif.get_ifd(ExifTags.IFD.Exif)
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)

        merged = dict(exif)
        merged.update(photo)

        out.tz_offset_min = _offset_minutes(merged.get(ExifTags.Base.OffsetTimeOriginal))
        out.captured_at = _timestamp(
            merged.get(ExifTags.Base.DateTimeOriginal), out.tz_offset_min
        )
        # 0x8827 is int16u, so it cannot express an ISO above 65535 and a body
        # shooting higher writes 65535 there and the real figure in
        # RecommendedExposureIndex (int32u)
        # (refs/exiftool/exiftool/lib/Image/ExifTool/Exif.pm:2145-2154).
        # Preferring 0x8827 unconditionally therefore files every high-ISO
        # frame in the library at exactly 65535.
        iso = merged.get(ExifTags.Base.ISOSpeedRatings)
        # Count is -1: a body may write several values. The first is the one
        # the picture was taken at.
        if isinstance(iso, (tuple, list)):
            iso = iso[0] if iso else None
        if iso is None or iso == 0xFFFF:
            iso = merged.get(ExifTags.Base.RecommendedExposureIndex) or iso
        out.iso = int(iso) if isinstance(iso, (int, float)) else None
        out.f_number = _number(merged.get(ExifTags.Base.FNumber))
        out.exposure_time = _number(merged.get(ExifTags.Base.ExposureTime))
        out.focal_length = _number(merged.get(ExifTags.Base.FocalLength))
        out.focal_35mm = _number(merged.get(ExifTags.Base.FocalLengthIn35mmFilm))
        orientation = merged.get(ExifTags.Base.Orientation)
        out.orientation = int(orientation) if isinstance(orientation, int) else None

        out.camera = _camera_name(
            merged.get(ExifTags.Base.Make), merged.get(ExifTags.Base.Model)
        )
        out.lens = _camera_name(
            merged.get(ExifTags.Base.LensMake), merged.get(ExifTags.Base.LensModel)
        )

        if gps:
            out.gps_lat = _degrees(
                gps.get(ExifTags.GPS.GPSLatitude), gps.get(ExifTags.GPS.GPSLatitudeRef)
            )
            out.gps_lon = _degrees(
                gps.get(ExifTags.GPS.GPSLongitude), gps.get(ExifTags.GPS.GPSLongitudeRef)
            )
            altitude = _number(gps.get(ExifTags.GPS.GPSAltitude))
            if altitude is not None:
                # GPSAltitudeRef is a byte, not a letter, and Exif 3.0 gives it
                # four values, not two: 0/1 are above/below the ellipsoid and
                # 2/3 above/below the sea-level reference
                # (refs/exiftool/exiftool/lib/Image/ExifTool/GPS.pm:113-116).
                # Testing only for 1 stores a below-sea-level frame as above it.
                #
                # GPSAltitude is rational64u -- unsigned -- so the sign lives
                # entirely in the reference, and abs() keeps a file that wrote
                # one anyway from cancelling it out.
                reference = gps.get(ExifTags.GPS.GPSAltitudeRef)
                if isinstance(reference, bytes):
                    reference = reference[0] if reference else 0
                out.gps_alt = -abs(altitude) if reference in (1, 3) else abs(altitude)

        for source, claimed, names in (
            (merged, _CLAIMED, ExifTags.TAGS),
            (gps or {}, _GPS_CLAIMED, ExifTags.GPSTAGS),
        ):
            prefix = "GPS" if names is ExifTags.GPSTAGS else ""
            for tag, value in source.items():
                if tag in claimed:
                    continue
                name = names.get(tag) or f"{prefix}Tag{tag:#06x}"
                held = _tag_value(tag, value)
                if held is None:
                    out.homeless.append((name, type(value).__name__))
                elif held.kind == "unrecorded":
                    out.unrecorded.append(name)
                elif held.kind == "binary" and held.payload is not None:
                    out.binaries.append((name, held.payload))
                elif held.text is not None:
                    out.params.append((name, held.text, held.number))
    return out


def store(conn, file_id: int, found: Capture, now: float, mint) -> None:
    """Write one file's camera metadata.

    `mint` is the entity minter, passed in rather than imported, so this
    module never has to know how addresses are allocated.
    """
    if found.is_empty:
        return
    conn.execute(
        "INSERT OR REPLACE INTO capture(file_id, captured_at, tz_offset_min, iso,"
        " f_number, exposure_time, focal_length, focal_35mm, orientation,"
        " gps_lat, gps_lon, gps_alt, parsed_at)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            file_id, found.captured_at, found.tz_offset_min, found.iso,
            found.f_number, found.exposure_time, found.focal_length,
            found.focal_35mm, found.orientation, found.gps_lat, found.gps_lon,
            found.gps_alt, now,
        ),
    )
    for kind, name, role in (
        ("camera", found.camera, "captured_with"),
        ("lens", found.lens, "mounted_lens"),
    ):
        if not name:
            continue
        key = "".join(c for c in name.lower() if c.isalnum())
        row = conn.execute(
            "SELECT id FROM artifact WHERE kind = ? AND name_key = ?", (kind, key)
        ).fetchone()
        if row:
            artifact_id = row[0]
        else:
            artifact_id = mint(conn, "artifact", f"{kind}-{name}")
            conn.execute(
                "INSERT INTO artifact(id, kind, name, name_key, first_seen_at)"
                " VALUES(?, ?, ?, ?, ?)",
                (artifact_id, kind, name, key, now),
            )
        conn.execute(
            "INSERT OR REPLACE INTO file_artifact(file_id, ordinal, artifact_id, role)"
            " VALUES(?, 0, ?, ?)",
            (file_id, artifact_id, role),
        )
    for name, text, number in found.params:
        # Never INSERT OR REPLACE; see db/ingest.py and the schema trigger
        # that refuses it.
        conn.execute(
            "INSERT INTO file_param(file_id, source, key, value_text, value_num)"
            " VALUES(?, 'exif', ?, ?, ?)"
            " ON CONFLICT(file_id, source, key) DO UPDATE SET"
            " value_text = excluded.value_text, value_num = excluded.value_num",
            (file_id, name, text, number),
        )
    for slot, payload in found.binaries:
        digest = hashlib.sha256(payload).hexdigest()
        conn.execute(
            "INSERT OR IGNORE INTO blob(hash, payload_bin, byte_len) VALUES(?, ?, ?)",
            (digest, payload, len(payload)),
        )
        # parsed_by stays NULL: the bytes are kept, and nothing here claims to
        # have understood them. That is what makes unparsed metadata a
        # queryable backlog rather than a silent loss.
        # Not INSERT OR REPLACE: REPLACE fires no DELETE trigger, so the
        # payload the row used to point at is never reclaimed.
        conn.execute(
            "INSERT INTO file_blob(file_id, carrier, slot, blob_hash, seen_at)"
            " VALUES(?, 'exif', ?, ?, ?)"
            " ON CONFLICT(file_id, carrier, slot) DO UPDATE SET"
            " blob_hash = excluded.blob_hash, seen_at = excluded.seen_at",
            (file_id, slot, digest, now),
        )
