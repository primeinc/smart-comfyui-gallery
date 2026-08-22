"""Every suffix the scanner claims is proven, end to end, on real bytes.

The rule vision/decode.py states -- a claimed suffix ships its decoder --
is only a rule if something fails when it stops being true. This module is
that something: for every suffix in `scan.KIND_BY_SUFFIX` a real file is
written by real libraries (Pillow and its shipped plugins, psd-tools,
tifffile's DNG writer, PyAV's encoders, pypdf), scanned, ingested, and
then asked the question its kind exists to answer -- pixels for stills, a
poster frame and a duration for video, a duration for audio, a page count
for documents.

The RAW family shares one decoder (LibRaw) and one synthesizable member:
DNG. The fixture is a genuine RGGB Bayer mosaic with the DNG tags, the
shape tifffile's own example writes and Adobe's validator accepts
(cgohlke/tifffile@ce19e3e examples/write_dng.py), which LibRaw then
demosaics exactly as it would a camera's file. The other RAW suffixes
cannot be synthesized -- they are camera-vendor sensor dumps -- so they
are held to routing through the same door the DNG proves; the door
itself is LibRaw's, whose camera coverage is its own tested claim
(letmaik/rawpy@326494b README.md).
"""

from __future__ import annotations

import pytest

from db import ingest, library, oriented, scan
from db import sample as sample_module
from tests.staging import fresh_schema
from vision import decode

SIZE = (64, 48)


def _still(fmt, **save):
    """fmt=None lets Pillow pick the writer from the suffix -- how the
    JPEG2000 family distinguishes a raw codestream (.j2k) from a boxed
    file (.jp2) at save time."""

    def write(path):
        from PIL import Image

        decode.ensure_decoders()
        frame = Image.new("RGB", SIZE, (200, 40, 90))
        frame.save(path, format=fmt, **save)

    return write


def _burst(fmt, **save):
    def write(path):
        from PIL import Image

        decode.ensure_decoders()
        frames = [Image.new("RGB", SIZE, (n * 60, 20, 20)) for n in range(3)]
        frames[0].save(path, format=fmt, save_all=True, append_images=frames[1:], **save)

    return write


def _psd(path):
    from PIL import Image
    from psd_tools import PSDImage

    PSDImage.frompil(Image.new("RGB", SIZE, (10, 200, 60))).save(str(path))


def _dng(path):
    """An RGGB Bayer DNG, the shape tifffile's own example validates
    (cgohlke/tifffile@ce19e3e examples/write_dng.py:141-168)."""
    import numpy as np
    from tifffile import TiffWriter, rational

    width, height = SIZE
    rng = np.random.default_rng(42)
    mosaic = ((np.arange(height * width, dtype=np.uint16).reshape(height, width) * 37) % 3000) + rng.integers(
        0, 16, (height, width), dtype=np.uint16
    )
    color_matrix = (0.4361, 0.3851, 0.1431, 0.2225, 0.7169, 0.0606, 0.0139, 0.0971, 0.7141)
    tags = [
        ("DNGVersion", 1, 4, (1, 4, 0, 0), True),
        ("UniqueCameraModel", 2, 0, "SmartGallery Synthetic", True),
        ("Orientation", 3, 1, 1, True),
        ("CalibrationIlluminant1", 3, 1, 21, True),
        ("ColorMatrix1", 10, 9, [rational(v) for v in color_matrix], True),
        ("AsShotNeutral", 5, 3, [rational(1.0)] * 3, True),
        ("CFARepeatPatternDim", 3, 2, (2, 2), True),
        ("CFAPattern", 1, 4, (0, 1, 1, 2), True),  # RGGB
        ("CFAPlaneColor", 1, 3, (0, 1, 2), True),
        ("CFALayout", 3, 1, 1, True),
        ("BlackLevel", 5, 1, rational(0), True),
        ("WhiteLevel", 4, 1, 4095, True),
    ]
    with TiffWriter(str(path), byteorder="<", kind="generic") as tif:
        tif.write(mosaic, photometric="cfa", subfiletype=0, extratags=tags)


def _video(fmt, codec, *, size=SIZE, rate=10, frames=8, pix="yuv420p"):
    def write(path):
        import av
        import numpy as np

        width, height = size
        with av.open(str(path), "w", format=fmt) as container:
            stream = container.add_stream(codec, rate=rate)
            stream.width, stream.height = width, height
            stream.pix_fmt = pix
            for n in range(frames):
                frame = av.VideoFrame.from_ndarray(
                    np.full((height, width, 3), (n * 30) % 256, dtype=np.uint8), format="rgb24"
                )
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    return write


def _audio(fmt, codec, *, rate=44100, bit_rate=None, options=None):
    def write(path):
        import av
        import numpy as np

        with av.open(str(path), "w", format=fmt) as container:
            stream = container.add_stream(codec, rate=rate, options=options or {})
            if bit_rate:
                stream.codec_context.bit_rate = bit_rate
            samples = (np.sin(np.arange(rate) * 0.05) * 20000).astype(np.int16).reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
            frame.sample_rate = rate
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    return write


def _pdf(path):
    import pypdf

    writer = pypdf.PdfWriter()
    writer.add_blank_page(612, 792)
    writer.add_blank_page(612, 792)
    with open(path, "wb") as handle:
        writer.write(handle)


#: suffix -> writer. h263 needs its legal frame size; 3gp carries it. The
#: FLV is period-correct Sorenson Spark and the RealMedia files are RV10 --
#: the codecs those containers actually held.
WRITERS = {
    ".png": _still("PNG"),
    ".jpg": _still("JPEG"),
    ".jpeg": _still("JPEG"),
    ".jpe": _still("JPEG"),
    ".jfif": _still("JPEG"),
    ".jif": _still("JPEG"),
    ".webp": _still("WEBP"),
    ".bmp": _still("BMP"),
    ".dib": _still("BMP"),
    ".tif": _still("TIFF"),
    ".tiff": _still("TIFF"),
    ".avif": _still("AVIF"),
    ".jxl": _still("JXL"),
    ".heic": _still("HEIF"),
    ".heif": _still("HEIF"),
    ".hif": _still("HEIF"),
    ".heics": _burst("HEIF"),
    ".heifs": _burst("HEIF"),
    ".jp2": _still("JPEG2000"),
    ".j2k": _still(None),
    ".jpf": _still(None),
    ".jpx": _still(None),
    ".mpo": _burst("MPO"),
    ".psd": _psd,
    ".gif": _burst("GIF"),
    ".apng": _burst("PNG"),
    ".dng": _dng,
    ".mp4": _video("mp4", "h264"),
    ".m4v": _video("mp4", "h264"),
    ".mov": _video("mov", "h264"),
    ".qt": _video("mov", "h264"),
    ".mkv": _video("matroska", "h264"),
    ".webm": _video("webm", "libvpx-vp9"),
    ".avi": _video("avi", "mpeg4"),
    ".divx": _video("avi", "mpeg4"),
    # mpeg1video accepts only the broadcast frame rates; 25 is one of them
    ".mpg": _video("mpeg", "mpeg1video", rate=25),
    ".mpeg": _video("mpeg", "mpeg1video", rate=25),
    ".mpe": _video("mpeg", "mpeg1video", rate=25),
    ".m2v": _video("mpeg2video", "mpeg2video", rate=25),
    # MJPEG is full-range JPEG frames; the encoder wants yuvj420p
    ".mjpeg": _video("mjpeg", "mjpeg", pix="yuvj420p"),
    ".mjpg": _video("mjpeg", "mjpeg", pix="yuvj420p"),
    ".ogv": _video("ogg", "libvpx"),
    ".ogm": _video("ogg", "libvpx"),
    ".vob": _video("vob", "mpeg2video"),
    ".ts": _video("mpegts", "h264"),
    ".mts": _video("mpegts", "h264"),
    ".m2ts": _video("mpegts", "h264"),
    ".m2t": _video("mpegts", "h264"),
    ".3gp": _video("3gp", "h263", size=(176, 144)),
    ".3gpp": _video("3gp", "h263", size=(176, 144)),
    ".wmv": _video("asf", "wmv2"),
    ".asf": _video("asf", "wmv2"),
    ".flv": _video("flv", "flv"),
    ".f4v": _video("mp4", "h264"),
    ".mxf": _video("mxf", "mpeg2video", rate=25),
    ".rm": _video("rm", "rv10"),
    ".rmvb": _video("rm", "rv10"),
    ".wav": _audio("wav", "pcm_s16le"),
    ".mp3": _audio("mp3", "libmp3lame"),
    ".mp2": _audio("mp2", "mp2"),
    ".flac": _audio("flac", "flac"),
    ".m4a": _audio("ipod", "aac"),
    # FFmpeg gates its native vorbis encoder behind strict=experimental
    ".ogg": _audio("ogg", "vorbis", options={"strict": "experimental"}),
    ".opus": _audio("ogg", "libopus", rate=48000),
    ".oga": _audio("ogg", "vorbis", options={"strict": "experimental"}),
    ".mka": _audio("matroska", "aac"),
    ".weba": _audio("webm", "libopus", rate=48000),
    ".caf": _audio("caf", "pcm_s16le"),
    ".au": _audio("au", "pcm_s16be"),
    ".aac": _audio("adts", "aac"),
    ".wma": _audio("asf", "wmav2", bit_rate=128000),
    ".aiff": _audio("aiff", "pcm_s16be"),
    ".aif": _audio("aiff", "pcm_s16be"),
    ".pdf": _pdf,
}

#: One suffix per writer configuration -- the decoder families. These run
#: in the fast lane; the aliases that share a writer (.jpeg/.jpe/.jfif
#: for JPEG, .m2ts/.mts/.m2t for MPEG-TS, ...) are the same bytes through
#: the same door and run in the slow lane, where the whole claim is proven.
FAMILIES = {
    ".png", ".jpg", ".webp", ".bmp", ".tif", ".avif", ".jxl", ".heic", ".heics", ".jp2", ".j2k",
    ".mpo", ".psd", ".gif", ".apng", ".dng",
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".mpg", ".m2v", ".mjpeg", ".ogv", ".vob", ".ts",
    ".3gp", ".wmv", ".flv", ".mxf", ".rm",
    ".wav", ".mp3", ".mp2", ".flac", ".m4a", ".ogg", ".opus", ".mka", ".weba", ".caf", ".au",
    ".aac", ".wma", ".aiff",
    ".pdf",
}  # fmt: skip

#: Elementary streams: raw codec data with no container clock around it.
#: They decode and poster like any video; a duration is not theirs to state.
ELEMENTARY = {".m2v", ".mjpeg", ".mjpg"}

#: RAW suffixes that cannot be synthesized: vendor sensor dumps. They route
#: through the LibRaw door the .dng writer proves.
ROUTING_ONLY = decode.RAW_SUFFIXES - {".dng"}


def test_the_registry_and_the_writers_agree_exactly():
    """Every claimed suffix is either written here or routing-only RAW, and
    nothing is written that the registry does not claim -- the two lists
    cannot drift apart silently in either direction."""
    claimed = set(scan.KIND_BY_SUFFIX)
    proven = set(WRITERS) | ROUTING_ONLY
    assert claimed == proven, {
        "claimed but never proven": sorted(claimed - proven),
        "proven but never claimed": sorted(proven - claimed),
    }


def test_every_raw_suffix_routes_through_libraw():
    for suffix in sorted(ROUTING_ONLY):
        assert scan.KIND_BY_SUFFIX[suffix] == "image", suffix
        assert suffix in decode.RAW_SUFFIXES, suffix


def test_every_family_is_one_of_the_writers():
    assert set(WRITERS) >= FAMILIES, sorted(FAMILIES - set(WRITERS))


@pytest.mark.parametrize(
    "suffix",
    [suffix if suffix in FAMILIES else pytest.param(suffix, marks=pytest.mark.slow) for suffix in sorted(WRITERS)],
)
def test_the_whole_pipeline_answers_for(suffix, tmp_path):
    kind = scan.KIND_BY_SUFFIX[suffix]
    root = tmp_path / "lib"
    root.mkdir()
    path = root / f"fixture{suffix}"
    WRITERS[suffix](path)
    assert path.stat().st_size > 0, f"the {suffix} writer produced nothing"

    conn = fresh_schema()
    root_id = library.add_root(conn, str(root), "library", 0.0)
    scan.scan(conn, root_id, str(root), 0.0)

    row = conn.execute("SELECT id, kind FROM file").fetchone()
    assert row is not None, f"the scanner does not recognise {suffix}"
    file_id, scanned_kind = row
    assert scanned_kind == kind, f"{suffix} scanned as {scanned_kind}, registry says {kind}"

    result = ingest.one(conn, file_id, path, 0.0)

    stored_kind, width, height, duration = conn.execute(
        "SELECT kind, width, height, duration FROM file WHERE id = ?", (file_id,)
    ).fetchone()

    if kind in ("image", "animated_image"):
        picture = oriented.for_model(conn, file_id, path)
        assert picture.size[0] > 0, f"{suffix} did not decode"
        # Animation is a decoded fact -- but only for suffixes that CAN
        # animate. An MPO reports extra frames too, and those are stereo
        # viewpoints, not motion, which is why ingest consults the decoder
        # only inside the possibly-animated set.
        if suffix in ingest._POSSIBLY_ANIMATED:
            moving = decode.is_animated(decode.open_still(path))
            assert stored_kind == ("animated_image" if moving else "image"), (
                f"{suffix}: decoded is_animated={moving} but the row says {stored_kind}"
            )
        else:
            assert stored_kind == "image", f"{suffix} row says {stored_kind}"
    elif kind == "video":
        assert result.probed, f"{suffix}: {result.unreadable}"
        if suffix not in ELEMENTARY:
            assert duration is not None, f"{suffix} cannot state its length"
            assert duration > 0, f"{suffix} cannot state its length"
        expected = (176, 144) if suffix in (".3gp", ".3gpp") else SIZE
        assert (width, height) == expected, f"{suffix} stored {width}x{height}"
        poster = decode.poster(path)
        assert poster is not None, f"{suffix} yields no poster frame"
        assert poster.size[0] > 0, f"{suffix} yields no poster frame"
        # sampling is a job's deliberate act, never ingest's side effect
        assert sample_module.taken(conn, file_id) == []
    elif kind == "audio":
        assert result.probed, f"{suffix}: {result.unreadable}"
        assert duration is not None, f"{suffix} cannot state its length"
        assert duration > 0, f"{suffix} cannot state its length"
    elif kind == "document":
        pages = conn.execute(
            "SELECT value_num FROM file_param WHERE file_id = ? AND key = 'Pages'", (file_id,)
        ).fetchone()
        assert pages is not None, f"{suffix} cannot state its page count"
        assert pages[0] == 2, f"{suffix} cannot state its page count"
        samples = conn.execute(
            "SELECT count(*) FROM derived_media_sample WHERE file_id = ? AND kind = 'page'",
            (file_id,),
        ).fetchone()[0]
        assert samples == 2, "a document's pages are moments, and they were not written"
    conn.close()


def test_a_dng_develops_through_the_libraw_door(tmp_path):
    """The RAW door itself: a Bayer mosaic written with the DNG tags,
    routed by suffix, demosaicked by LibRaw into a color picture."""
    path = tmp_path / "shot.dng"
    _dng(path)
    picture = decode.open_still(path)
    assert picture.size == SIZE
    assert picture.mode == "RGB", "LibRaw demosaics a CFA into color"


def _library_of(root):
    conn = fresh_schema()
    root_id = library.add_root(conn, str(root), "library", 0.0)
    scan.scan(conn, root_id, str(root), 0.0)
    return conn


def test_the_bytes_decide_what_a_file_is_whatever_its_suffix_says(tmp_path):
    """Three liars in one library, one scan. The suffix proposes a kind;
    the sniff re-identifies an MP4 wearing .jpg and a PNG wearing .mp4 by
    their bytes (routed by suffix the first would hit Pillow, fail, and be
    a broken image forever); an executable wearing .png matches no
    signature, the decoder refuses, and the row records the refusal
    instead of pretending."""
    root = tmp_path / "lib"
    root.mkdir()
    WRITERS[".mp4"](root / "holiday.jpg")
    WRITERS[".png"](root / "clip.mp4")
    (root / "totally-a-picture.png").write_bytes(b"MZ" + bytes(range(256)) * 4)

    conn = _library_of(root)
    rows = {name: (file_id, kind) for file_id, name, kind in conn.execute("SELECT id, name, kind FROM file")}
    assert {name: kind for name, (_, kind) in rows.items()} == {
        "holiday.jpg": "image",
        "clip.mp4": "video",
        "totally-a-picture.png": "image",
    }, "the suffix proposes"

    movie, _ = rows["holiday.jpg"]
    ingest.one(conn, movie, root / "holiday.jpg", 0.0)
    kind, duration = conn.execute("SELECT kind, duration FROM file WHERE id = ?", (movie,)).fetchone()
    assert kind == "video", "the bytes decide"
    assert duration is not None
    assert duration > 0
    fields = dict(
        conn.execute(
            "SELECT key, value_text FROM file_param WHERE file_id = ? AND source = 'container'",
            (movie,),
        )
    )
    assert fields["SniffedFormat"] == "mp4"
    assert fields["SuffixClaimed"] == "image"

    still, _ = rows["clip.mp4"]
    ingest.one(conn, still, root / "clip.mp4", 0.0)
    assert conn.execute("SELECT kind FROM file WHERE id = ?", (still,)).fetchone()[0] == "image"
    picture = oriented.for_model(conn, still, root / "clip.mp4")
    assert picture.size == SIZE

    bald, _ = rows["totally-a-picture.png"]
    result = ingest.one(conn, bald, root / "totally-a-picture.png", 0.0)
    assert result.unreadable, "a lie this bald must be recorded, not absorbed"
    sniffed = conn.execute(
        "SELECT count(*) FROM file_param WHERE file_id = ? AND key = 'SniffedFormat'",
        (bald,),
    ).fetchone()[0]
    assert sniffed == 0, "no signature matched, so no format may be claimed"
    conn.close()


def test_dimensions_answers_from_headers_for_every_kind(tmp_path):
    """The decoder door's geometry probe: stills through the registered
    openers, video through the container's stream entry -- never a frame
    decoded, never bare Image.open, so the answer does not depend on
    which code ran first in the process. Unreadable bytes answer None,
    ranking last wherever geometry decides."""
    import av
    import numpy as np
    from PIL import Image

    from vision import decode

    still = tmp_path / "still.png"
    Image.new("RGB", (96, 64), (10, 120, 30)).save(still)
    assert decode.dimensions(still, "image") == (96, 64)

    webp = tmp_path / "still.webp"
    Image.new("RGB", (48, 32), (10, 120, 30)).save(webp, quality=75)
    assert decode.dimensions(webp, "image") == (48, 32)

    clip = tmp_path / "clip.mp4"
    with av.open(str(clip), "w") as container:
        stream = container.add_stream("h264", rate=5)
        stream.width, stream.height = 320, 180
        stream.pix_fmt = "yuv420p"
        for _ in range(4):
            frame = av.VideoFrame.from_ndarray(np.zeros((180, 320, 3), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    assert decode.dimensions(clip, "video") == (320, 180)

    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a picture at all")
    assert decode.dimensions(broken, "image") is None
    assert decode.dimensions(broken, "video") is None
    assert decode.dimensions(tmp_path / "absent.png", "image") is None
