"""What a camera writes, and whether the schema can hold it.

`capture` had columns and no producer: nothing in the app had ever read the
tags a camera writes about the photograph, only the five that generation
tools abuse as carriers. So every question this table exists to answer --
what lens, what aperture, taken where -- was answerable only in principle.

Each test writes real EXIF into a real file with Pillow, reads it back
through the real reader, and stores it. The conversions are the point:
EXIF keeps f/1.4 as two integers, position as three rationals and a letter,
and time as a wall clock that may carry no zone at all.
"""

import pathlib
import sqlite3

import pytest
from PIL import ExifTags, Image
from PIL.TiffImagePlugin import IFDRational

from db import capture, scan

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"
NOW = 1_700_000_000.0


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'C:/lib','library',0)")
    conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(1,X'00000000000000000000000000000001','folder','lib')")
    conn.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(1,1,NULL,'lib',0)")
    yield conn
    conn.close()


@pytest.fixture
def a_file(db):
    """A stored file to hang camera metadata on."""
    file_id = scan.mint(db, "file", "dsc-0001")
    db.execute(
        "INSERT INTO file(id,folder_id,name,kind,size,mtime,first_seen_at,last_seen_at)"
        " VALUES(?,1,'DSC_0001.jpg','image',1,0,0,0)",
        (file_id,),
    )
    return file_id


def photograph(path, base=None, photo=None, gps=None):
    """A JPEG carrying the EXIF a camera would have written."""
    exif = Image.Exif()
    for tag, value in (base or {}).items():
        exif[tag] = value
    if photo:
        exif.get_ifd(ExifTags.IFD.Exif).update(photo)
    if gps:
        exif.get_ifd(ExifTags.IFD.GPSInfo).update(gps)
    Image.new("RGB", (16, 16), (90, 110, 130)).save(path, exif=exif)
    return str(path)


# A Fujifilm X-T5 frame, as the body writes it.
FUJI_BASE = {
    ExifTags.Base.Make: "FUJIFILM",
    ExifTags.Base.Model: "X-T5",
    ExifTags.Base.Orientation: 1,
    ExifTags.Base.Software: "Digital Camera X-T5 Ver1.10",
}
FUJI_PHOTO = {
    ExifTags.Base.DateTimeOriginal: "2026:08:19 14:23:01",
    ExifTags.Base.OffsetTimeOriginal: "+02:00",
    ExifTags.Base.ExposureTime: IFDRational(1, 250),
    ExifTags.Base.FNumber: IFDRational(14, 10),
    ExifTags.Base.ISOSpeedRatings: 400,
    ExifTags.Base.FocalLength: IFDRational(35, 1),
    ExifTags.Base.FocalLengthIn35mmFilm: 53,
    ExifTags.Base.LensMake: "FUJIFILM",
    ExifTags.Base.LensModel: "XF35mmF1.4 R",
    ExifTags.Base.ExposureBiasValue: IFDRational(-1, 3),
}
# Greenwich Observatory, south of the equator's sign convention on purpose.
GPS_LONDON = {
    ExifTags.GPS.GPSLatitudeRef: "N",
    ExifTags.GPS.GPSLatitude: (IFDRational(51, 1), IFDRational(28, 1), IFDRational(4008, 100)),
    ExifTags.GPS.GPSLongitudeRef: "W",
    ExifTags.GPS.GPSLongitude: (IFDRational(0, 1), IFDRational(0, 1), IFDRational(2316, 100)),
    ExifTags.GPS.GPSAltitudeRef: 0,
    ExifTags.GPS.GPSAltitude: IFDRational(4520, 100),
}


def stored(db, file_id):
    return db.execute(
        "SELECT captured_at, tz_offset_min, iso, f_number, exposure_time,"
        " focal_length, focal_35mm, orientation, gps_lat, gps_lon, gps_alt"
        " FROM capture WHERE file_id = ?",
        (file_id,),
    ).fetchone()


# --- the conversions -------------------------------------------------------


def test_a_frame_becomes_a_capture_row(db, a_file, tmp_path):
    path = photograph(tmp_path / "a.jpg", FUJI_BASE, FUJI_PHOTO, GPS_LONDON)
    found = capture.read(path)
    capture.store(db, a_file, found, NOW, scan.mint)

    row = stored(db, a_file)
    assert row is not None, "a camera frame stored no capture row"
    tz, iso, f_number, exposure, focal, focal35, orientation = row[1:8]
    assert (iso, orientation, focal35) == (400, 1, 53)
    assert f_number == pytest.approx(1.4), "f/1.4 is stored as 14/10 and must arrive as a number"
    assert exposure == pytest.approx(1 / 250)
    assert focal == pytest.approx(35.0)
    assert tz == 120, "+02:00 is 120 minutes east"


def test_the_shutter_time_is_an_instant_when_the_zone_is_known(db, a_file, tmp_path):
    """14:23 at +02:00 is 12:23 UTC, and must not depend on the reader's clock."""
    path = photograph(tmp_path / "a.jpg", FUJI_BASE, FUJI_PHOTO)
    capture.store(db, a_file, capture.read(path), NOW, scan.mint)
    import datetime as dt

    captured_at = stored(db, a_file)[0]
    assert dt.datetime.fromtimestamp(captured_at, dt.timezone.utc).isoformat() == ("2026-08-19T12:23:01+00:00")


def test_a_frame_with_no_zone_says_so_instead_of_guessing(db, a_file, tmp_path):
    """Most cameras write no offset. The wall clock is read as UTC so the
    number is the same everywhere, and the NULL is what says it is a wall
    clock rather than an instant."""
    photo = dict(FUJI_PHOTO)
    del photo[ExifTags.Base.OffsetTimeOriginal]
    path = photograph(tmp_path / "a.jpg", FUJI_BASE, photo)
    capture.store(db, a_file, capture.read(path), NOW, scan.mint)
    import datetime as dt

    captured_at, tz = stored(db, a_file)[:2]
    assert tz is None, "an absent offset must not be invented"
    assert dt.datetime.fromtimestamp(captured_at, dt.timezone.utc).isoformat() == ("2026-08-19T14:23:01+00:00")


def test_a_high_iso_is_not_filed_at_the_ceiling(db, a_file, tmp_path):
    """ISOSpeedRatings is int16u and cannot hold a value above 65535, so a
    body shooting higher writes the ceiling there and the real figure in
    RecommendedExposureIndex
    (refs/exiftool/exiftool/lib/Image/ExifTool/Exif.pm:2145-2154). Reading
    the first tag unconditionally files every high-ISO frame at 65535.
    """
    photo = dict(FUJI_PHOTO)
    photo[ExifTags.Base.ISOSpeedRatings] = 65535
    photo[ExifTags.Base.RecommendedExposureIndex] = 102400
    path = photograph(tmp_path / "a.jpg", FUJI_BASE, photo)
    capture.store(db, a_file, capture.read(path), NOW, scan.mint)
    assert stored(db, a_file)[2] == 102400


def test_an_ordinary_iso_is_taken_from_the_ordinary_tag(db, a_file, tmp_path):
    """The control for the test above: the fallback must not fire when the
    normal tag holds a usable value."""
    photo = dict(FUJI_PHOTO)
    photo[ExifTags.Base.RecommendedExposureIndex] = 102400
    path = photograph(tmp_path / "a.jpg", FUJI_BASE, photo)
    capture.store(db, a_file, capture.read(path), NOW, scan.mint)
    assert stored(db, a_file)[2] == 400


def test_position_carries_its_hemisphere(db, a_file, tmp_path):
    """The degrees are unsigned; the letter is the sign. Ignoring it puts
    half the planet in the wrong hemisphere."""
    path = photograph(tmp_path / "a.jpg", FUJI_BASE, FUJI_PHOTO, GPS_LONDON)
    capture.store(db, a_file, capture.read(path), NOW, scan.mint)
    lat, lon, alt = stored(db, a_file)[8:]
    assert lat == pytest.approx(51.4778, abs=1e-4)
    assert lon == pytest.approx(-0.0064, abs=1e-4), "W must be negative"
    assert alt == pytest.approx(45.2, abs=1e-2)


@pytest.mark.parametrize(
    ("reference", "sign"),
    [
        (0, 1),  # above the ellipsoid
        (1, -1),  # below the ellipsoid
        (2, 1),  # above the sea-level reference   (Exif 3.0)
        (3, -1),  # below the sea-level reference   (Exif 3.0)
    ],
)
def test_altitude_takes_its_sign_from_the_reference(db, a_file, tmp_path, reference, sign):
    """GPSAltitudeRef is a byte with four meanings, not a flag with two.

    Exif 3.0 added 2 and 3 for the sea-level reference
    (refs/exiftool/exiftool/lib/Image/ExifTool/GPS.pm:113-116), so a reader
    that only tests for 1 files a below-sea-level frame above it. The value
    itself is rational64u -- unsigned -- so the reference is the only place
    the sign can come from.
    """
    gps = dict(GPS_LONDON)
    gps[ExifTags.GPS.GPSAltitudeRef] = reference
    path = photograph(tmp_path / "a.jpg", FUJI_BASE, FUJI_PHOTO, gps)
    capture.store(db, a_file, capture.read(path), NOW, scan.mint)
    assert stored(db, a_file)[10] == pytest.approx(sign * 45.2, abs=1e-2)


def test_a_frame_with_no_position_stores_no_position(db, a_file, tmp_path):
    """A places page must be able to tell "not there" from "at zero"."""
    path = photograph(tmp_path / "a.jpg", FUJI_BASE, FUJI_PHOTO)
    capture.store(db, a_file, capture.read(path), NOW, scan.mint)
    assert stored(db, a_file)[8:] == (None, None, None)


# --- the equipment becomes an entity ---------------------------------------


def test_the_body_and_the_lens_become_artifacts(db, a_file, tmp_path):
    path = photograph(tmp_path / "a.jpg", FUJI_BASE, FUJI_PHOTO)
    capture.store(db, a_file, capture.read(path), NOW, scan.mint)
    rows = db.execute(
        "SELECT a.kind, a.name, fa.role FROM file_artifact fa"
        " JOIN artifact a ON a.id = fa.artifact_id WHERE fa.file_id = ?"
        " ORDER BY a.kind",
        (a_file,),
    ).fetchall()
    assert rows == [
        ("camera", "FUJIFILM X-T5", "captured_with"),
        ("lens", "FUJIFILM XF35mmF1.4 R", "mounted_lens"),
    ], rows


@pytest.mark.parametrize(
    ("make", "model", "expected"),
    [
        # the make is a legal entity and the model carries the brand
        ("NIKON CORPORATION", "NIKON D2X", "NIKON D2X"),
        ("EASTMAN KODAK COMPANY", "KODAK C310 DIGITAL CAMERA", "KODAK C310 DIGITAL CAMERA"),
        ("PENTAX Corporation", "PENTAX *ist DL", "PENTAX *ist DL"),
        ("Canon", "Canon EOS R5", "Canon EOS R5"),
        # the model does not name the brand, so the make is needed to tell
        # two manufacturers' identically named bodies apart
        ("FUJIFILM", "X-T5", "FUJIFILM X-T5"),
        ("Xiaomi", "Mi 9 Lite", "Xiaomi Mi 9 Lite"),
        ("SONY", "CYBERSHOT", "SONY CYBERSHOT"),
        # only one of the two present
        ("Apple", None, "Apple"),
        (None, "DMC-FZ50", "DMC-FZ50"),
    ],
)
def test_a_body_is_named_once(tmp_path, make, model, expected):
    """All nine cases are shapes observed in a real library.

    A prefix test is not enough: bodies put the legal entity in Make and the
    brand in Model, which leaves 'NIKON CORPORATION NIKON D2X'. Testing the
    model's first word against the make's words handles that without
    collapsing 'Xiaomi' + 'Mi 9 Lite', where 'Mi' is not a word of 'Xiaomi'.
    """
    base = {}
    if make:
        base[ExifTags.Base.Make] = make
    if model:
        base[ExifTags.Base.Model] = model
    path = photograph(tmp_path / "a.jpg", base, FUJI_PHOTO)
    assert capture.read(path).camera == expected


def test_an_enumerated_value_is_stored_as_the_phrase_it_stands_for(db, a_file, tmp_path):
    """`Flash = 89` is unsearchable and unreadable: nobody types 89 and
    nobody recognises it. The phrase goes in value_text so a person can find
    it, and the code stays in value_num so it is still comparable
    (refs/exiftool/exiftool/lib/Image/ExifTool/Exif.pm:175-209)."""
    photo = dict(FUJI_PHOTO)
    photo[ExifTags.Base.Flash] = 0x59
    photo[ExifTags.Base.MeteringMode] = 5
    photo[ExifTags.Base.ExposureProgram] = 3
    path = photograph(tmp_path / "a.jpg", FUJI_BASE, photo)
    capture.store(db, a_file, capture.read(path), NOW, scan.mint)

    stored_params = dict(
        db.execute(
            "SELECT key, value_text FROM file_param WHERE file_id = ? AND source='exif'",
            (a_file,),
        )
    )
    assert stored_params["Flash"] == "Auto, Fired, Red-eye reduction"
    assert stored_params["MeteringMode"] == "Multi-segment"
    assert stored_params["ExposureProgram"] == "Aperture-priority AE"
    code = db.execute("SELECT value_num FROM file_param WHERE file_id = ? AND key='Flash'", (a_file,)).fetchone()[0]
    assert code == 0x59, "the code must survive alongside the phrase"


def test_an_unlisted_code_is_not_given_an_invented_label(db, a_file, tmp_path):
    """A value missing from the tables means the table needs extending.
    Labelling it anyway would hide that, so it stays the raw code."""
    photo = dict(FUJI_PHOTO)
    photo[ExifTags.Base.MeteringMode] = 200  # not a defined MeteringMode
    path = photograph(tmp_path / "a.jpg", FUJI_BASE, photo)
    capture.store(db, a_file, capture.read(path), NOW, scan.mint)
    row = db.execute(
        "SELECT value_text, value_num FROM file_param WHERE file_id = ? AND key='MeteringMode'",
        (a_file,),
    ).fetchone()
    assert row == ("200", 200.0), row


def test_two_frames_from_one_body_share_its_row(db, tmp_path):
    """Otherwise a camera page counts bodies instead of photographs."""
    for i in range(2):
        file_id = scan.mint(db, "file", f"frame-{i}")
        db.execute(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,first_seen_at,last_seen_at)"
            " VALUES(?,1,?,'image',1,0,0,0)",
            (file_id, f"DSC_{i}.jpg"),
        )
        path = photograph(tmp_path / f"{i}.jpg", FUJI_BASE, FUJI_PHOTO)
        capture.store(db, file_id, capture.read(path), NOW, scan.mint)

    bodies = db.execute("SELECT name, count(*) FROM artifact WHERE kind='camera'").fetchone()
    assert bodies == ("FUJIFILM X-T5", 1)
    used = db.execute("SELECT count(*) FROM file_artifact WHERE role='captured_with'").fetchone()[0]
    assert used == 2


def test_a_camera_may_not_claim_a_content_hash(db, a_file, tmp_path):
    """A body has no bytes. NULL there means 'not applicable', and it must not
    be confused with a checkpoint whose file is merely not on this machine."""
    path = photograph(tmp_path / "a.jpg", FUJI_BASE, FUJI_PHOTO)
    capture.store(db, a_file, capture.read(path), NOW, scan.mint)
    camera_id = db.execute("SELECT id FROM artifact WHERE kind='camera'").fetchone()[0]
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE artifact SET content_sha256 = 'x' WHERE id = ?", (camera_id,))


# --- the long tail ---------------------------------------------------------


def test_every_remaining_tag_is_queryable(db, a_file, tmp_path):
    """A tag with no scalar form is a gap, not an acceptable loss: the whole
    reason `generation` has no `extra` JSON column is that nothing could
    search it."""
    path = photograph(tmp_path / "a.jpg", FUJI_BASE, FUJI_PHOTO, GPS_LONDON)
    found = capture.read(path)
    assert found.homeless == [], f"no scalar home for {found.homeless}"

    capture.store(db, a_file, found, NOW, scan.mint)
    keys = dict(
        db.execute(
            "SELECT key, value_text FROM file_param WHERE file_id = ? AND source='exif'",
            (a_file,),
        )
    )
    assert "Software" in keys, keys
    assert keys["Software"] == "Digital Camera X-T5 Ver1.10"


def test_a_tag_that_is_a_number_is_stored_as_one(db, a_file, tmp_path):
    """Otherwise "exposure compensation below zero" is a string comparison."""
    path = photograph(tmp_path / "a.jpg", FUJI_BASE, FUJI_PHOTO)
    capture.store(db, a_file, capture.read(path), NOW, scan.mint)
    row = db.execute(
        "SELECT value_num FROM file_param WHERE file_id = ? AND key = 'ExposureBiasValue'",
        (a_file,),
    ).fetchone()
    assert row is not None, "the tag was not stored at all"
    assert row[0] == pytest.approx(-1 / 3)


def test_the_registry_learns_the_camera_vocabulary(db, a_file, tmp_path):
    """param_key is how the app can offer a facet for a tag nobody predicted."""
    path = photograph(tmp_path / "a.jpg", FUJI_BASE, FUJI_PHOTO, GPS_LONDON)
    capture.store(db, a_file, capture.read(path), NOW, scan.mint)
    learned = dict(db.execute("SELECT key, occurrences FROM param_key WHERE source='exif'"))
    counted = dict(db.execute("SELECT key, count(*) FROM file_param WHERE source='exif' GROUP BY key"))
    assert learned == counted, "the registry disagrees with the rows it counts"
    assert learned, "a frame full of EXIF taught the registry nothing"


def test_a_column_tag_is_not_also_long_tail(db, a_file, tmp_path):
    """A value stored twice drifts. FNumber is a column; it must not also be
    a row nothing keeps in step with it."""
    path = photograph(tmp_path / "a.jpg", FUJI_BASE, FUJI_PHOTO, GPS_LONDON)
    capture.store(db, a_file, capture.read(path), NOW, scan.mint)
    duplicated = db.execute(
        "SELECT key FROM file_param WHERE file_id = ? AND source='exif'"
        " AND key IN ('FNumber','ISOSpeedRatings','DateTimeOriginal','Make','Model',"
        "'LensModel','Orientation','GPSLatitude','FocalLength')",
        (a_file,),
    ).fetchall()
    assert duplicated == [], duplicated


# The awkward half of real EXIF. Everything above was written by Pillow and
# read back by Pillow, which agrees with itself about types; a camera does
# not. These are the shapes that actually arrive: binary blobs, a version as
# four bytes, an unknown-rational meaning "not recorded", a tag no table
# names, and text that is not ASCII.
AWKWARD_PHOTO = {
    ExifTags.Base.ExifVersion: b"0232",
    ExifTags.Base.ComponentsConfiguration: b"\x01\x02\x03\x00",
    ExifTags.Base.MakerNote: b"FUJIFILM\x0c\x00\x00\x00\x1a\xff\x00\x91",
    ExifTags.Base.UserComment: b"ASCII\x00\x00\x00handheld, wide open",
    ExifTags.Base.SubjectLocation: (1536, 1024, 320, 320),
    # denominator 0 is how a body writes "not recorded"
    ExifTags.Base.MaxApertureValue: IFDRational(0, 0),
    0xFDE8: "a tag no table names",
}
AWKWARD_BASE = {
    ExifTags.Base.Make: "FUJIFILM",
    ExifTags.Base.Model: "X-T5",
    ExifTags.Base.Artist: "Björn Åberg",
    ExifTags.Base.Copyright: "© 2026",
}
AWKWARD_GPS = {
    ExifTags.GPS.GPSVersionID: b"\x02\x03\x00\x00",
    ExifTags.GPS.GPSLatitudeRef: "S",
    ExifTags.GPS.GPSLatitude: (IFDRational(33, 1), IFDRational(51, 1), IFDRational(3540, 100)),
    ExifTags.GPS.GPSLongitudeRef: "E",
    ExifTags.GPS.GPSLongitude: (IFDRational(151, 1), IFDRational(12, 1), IFDRational(4000, 100)),
    ExifTags.GPS.GPSProcessingMethod: b"ASCII\x00\x00\x00GPS",
}


@pytest.fixture
def awkward(db, a_file, tmp_path):
    """One frame carrying the shapes a real body writes, stored."""
    path = photograph(tmp_path / "a.jpg", AWKWARD_BASE, AWKWARD_PHOTO, AWKWARD_GPS)
    found = capture.read(path)
    capture.store(db, a_file, found, NOW, scan.mint)
    return found


def params(db, file_id):
    return dict(
        db.execute(
            "SELECT key, value_text FROM file_param WHERE file_id = ? AND source='exif'",
            (file_id,),
        )
    )


def test_the_shapes_a_camera_actually_writes(db, a_file, awkward):
    """Nothing raises, and nothing is left with no home at all.

    The control matters as much as the assertion: this only means anything
    if the awkward tags actually survived being written, so the tags are
    named rather than counted.
    """
    assert awkward.homeless == [], f"no home at all for {awkward.homeless}"
    stored_keys = set(params(db, a_file)) | {
        slot for (slot,) in db.execute("SELECT slot FROM file_blob WHERE file_id = ? AND carrier='exif'", (a_file,))
    }
    assert {"ExifVersion", "MakerNote", "UserComment", "SubjectLocation"} <= stored_keys, (
        f"the awkward tags never reached the reader: {sorted(stored_keys)}"
    )


def test_binary_stays_binary_instead_of_becoming_mojibake(db, a_file, awkward):
    """A maker note is a proprietary blob. Forced through a lossy decode it
    became text that looks like data, so nothing downstream could tell it had
    been damaged. It belongs in the blob store, unparsed and honest about it."""
    row = db.execute(
        "SELECT b.payload_bin, b.payload, fb.parsed_by FROM file_blob fb"
        " JOIN blob b ON b.hash = fb.blob_hash"
        " WHERE fb.file_id = ? AND fb.slot = 'MakerNote'",
        (a_file,),
    ).fetchone()
    assert row is not None, "the maker note was dropped"
    assert row[0] == AWKWARD_PHOTO[ExifTags.Base.MakerNote], "the bytes were altered"
    assert row[1] is None, "binary must not be stored as text"
    assert row[2] is None, "nothing understood it, and the row must say so"
    assert "MakerNote" not in params(db, a_file)


def test_a_four_byte_version_is_not_truncated_to_two(db, a_file, awkward):
    """GPSVersionID is b'\\x02\\x03\\x00\\x00'. Stripping NULs as if they were
    padding threw away half of it and stored the rest as text."""
    assert "GPSVersionID" not in params(db, a_file)
    payload = db.execute(
        "SELECT b.payload_bin FROM file_blob fb JOIN blob b ON b.hash = fb.blob_hash"
        " WHERE fb.file_id = ? AND fb.slot = 'GPSVersionID'",
        (a_file,),
    ).fetchone()
    assert payload is not None
    assert payload[0] == b"\x02\x03\x00\x00"


def test_a_charset_prefixed_comment_is_decoded(db, a_file, awkward):
    """UserComment is an 8-byte character-code prefix and then the text. The
    repo already had a decoder for this; storing the prefix as part of the
    value made every comment start with the word ASCII."""
    assert params(db, a_file).get("UserComment") == "handheld, wide open"


def test_a_pointer_to_a_sub_ifd_is_not_a_searchable_fact(db, a_file, awkward):
    """ExifOffset and GPSInfo are byte offsets into this one file. Stored as
    parameters they become facets that mean nothing and match nothing."""
    stored_keys = params(db, a_file)
    assert "ExifOffset" not in stored_keys
    assert "GPSInfo" not in stored_keys


def test_no_number_stored_is_nan(db, a_file, awkward):
    """A body writes 0/0 for a value it did not record, and Pillow evaluates
    that to NaN rather than raising. NaN compares false against everything
    including itself, so one stored NaN makes a range facet silently drop the
    row and an equality test never match it."""
    bad = db.execute(
        "SELECT key, value_num FROM file_param WHERE file_id = ? AND value_num IS NOT NULL AND value_num <> value_num",
        (a_file,),
    ).fetchall()
    assert bad == [], f"non-finite numbers stored: {bad}"
    assert params(db, a_file).get("MaxApertureValue") != "nan"


def test_a_southern_hemisphere_frame_is_south(db, a_file, tmp_path):
    """Sydney is negative latitude, positive longitude. A reader that drops
    the reference letter puts it off the coast of Morocco."""
    path = photograph(tmp_path / "a.jpg", AWKWARD_BASE, AWKWARD_PHOTO, AWKWARD_GPS)
    capture.store(db, a_file, capture.read(path), NOW, scan.mint)
    lat, lon = stored(db, a_file)[8:10]
    assert lat == pytest.approx(-33.8598, abs=1e-4)
    assert lon == pytest.approx(151.2111, abs=1e-4)


def test_an_unrecorded_rational_is_absent_not_zero(db, a_file, awkward):
    """A camera writes 0/0 for a value it did not record. Storing 0 would
    claim a measurement that was never taken."""
    row = db.execute(
        "SELECT value_num FROM file_param WHERE file_id = ? AND key = 'MaxApertureValue'",
        (a_file,),
    ).fetchone()
    assert row is None or row[0] is None, f"0/0 became {row}"


def test_a_file_with_no_exif_writes_nothing(db, a_file, tmp_path):
    """A generated PNG is not a photograph, and must not get an empty
    capture row that a places page would then have to filter out."""
    Image.new("RGB", (16, 16)).save(tmp_path / "plain.png")
    found = capture.read(tmp_path / "plain.png")
    capture.store(db, a_file, found, NOW, scan.mint)
    assert stored(db, a_file) is None
    assert db.execute("SELECT count(*) FROM artifact").fetchone()[0] == 0


# --- facts that live in columns rather than in the long tail ----------------


def _bare(tmp_path, name, tags, size=(320, 180)):
    """A picture carrying exactly the tags named and nothing else."""
    image = Image.new("RGB", size, (30, 60, 90))
    exif = image.getexif()
    for tag, value in tags.items():
        exif[tag] = value
    path = tmp_path / name
    image.save(path, exif=exif)
    return path


@pytest.mark.parametrize(
    ("label", "tags", "reads"),
    [
        ("a body with no clock", {271: "NIKON CORPORATION", 272: "NIKON D2X"}, lambda c: c.camera == "NIKON D2X"),
        ("an ISO and nothing else", {34855: 400}, lambda c: c.iso == 400),
        ("an orientation alone", {274: 6}, lambda c: c.orientation == 6),
        ("a lens alone", {42035: "Nikon", 42036: "AF-S 50mm"}, lambda c: c.lens == "Nikon AF-S 50mm"),
    ],
)
def test_a_reading_that_lands_in_a_column_is_still_a_reading(tmp_path, label, tags, reads):
    """`is_empty` asked only about the long tail and the timestamp.

    So a photograph whose camera facts all landed in columns was discarded
    whole: Make and Model with no date recorded no camera at all, an ISO
    recorded no ISO, an orientation recorded nothing. It read as correct
    because most photographs carry DateTimeOriginal and everything else rode
    in on that -- a scan, a screenshot or an exported crop did not, and their
    camera never reached the /cameras page.
    """
    found = capture.read(_bare(tmp_path, f"{label.replace(' ', '_')}.jpg", tags))
    assert not found.is_empty, f"{label}: every one of these facts was thrown away"
    assert reads(found), f"{label}: read the wrong value"


def test_a_picture_with_nothing_in_it_is_still_empty(tmp_path):
    """The control. Without it the test above passes on an `is_empty` that
    always says False, and every file grows an empty capture row."""
    image = Image.new("RGB", (8, 8), (1, 2, 3))
    path = tmp_path / "nothing.png"
    image.save(path)
    assert capture.read(path).is_empty


@pytest.mark.parametrize(
    ("orientation", "expected"),
    [
        (1, (320, 180)),
        (2, (320, 180)),
        (3, (320, 180)),
        (4, (320, 180)),
        (5, (180, 320)),
        (6, (180, 320)),
        (7, (180, 320)),
        (8, (180, 320)),
    ],
)
def test_a_turned_photograph_reports_the_size_it_is_seen_at(db, tmp_path, orientation, expected):
    """The decode reports the stored frame and the tag says to turn it.

    Storing the stored size files every portrait photograph in the library as
    landscape, so a layout draws a landscape box around an upright picture.
    5 through 8 all turn it a quarter
    (refs/exiftool/exiftool/lib/Image/ExifTool/Exif.pm:291-300); 2 and 4
    mirror without turning and 3 is a half turn, so those keep their shape.
    """
    from db import ingest, library, scan

    root = tmp_path / "lib"
    root.mkdir()
    _bare(root, "phone.jpg", {274: orientation})
    root_id = library.add_root(db, root, "library", 0.0)
    scan.scan(db, root_id, root, 0.0)
    file_id = db.execute("SELECT id FROM file").fetchone()[0]
    ingest.one(db, file_id, root / "phone.jpg", 0.0)

    assert db.execute("SELECT width, height FROM file WHERE id = ?", (file_id,)).fetchone() == expected
