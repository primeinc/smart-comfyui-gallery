"""Every decoder this application ships, behind one entry point.

The rule this module enforces: a suffix the scanner claims is a suffix the
install decodes -- "not installed" is a dependency-declaration failure, not
a support policy. Stills go through Pillow with the HEIF and JPEG XL
plugins registered here and nowhere else (register_heif_opener per
bigcat88/pillow_heif@657f27d README.md; `import pillow_jxl` registers itself
per Isotr0py/pillow-jpegxl-plugin@9d817d1 README.md). The RAW family goes
through LibRaw -- rawpy's imread/postprocess, letmaik/rawpy@326494b README.md
-- and moving pictures go through PyAV, which carries FFmpeg's libraries
inside its wheel, so probing, posters and frame sampling need no system
binary.

All of it was proven on real bytes before being claimed: HEIC/JXL/AVIF
round-trip; 29 of 29 claimed video and audio containers mux and decode;
the codecs a real 2000s library contains -- Sorenson Spark, VP6,
Nellymoser, MS-MPEG4, WMV7/9, VC-1, SVQ, Cinepak, Indeo, RealVideo 1-4,
RealAudio/Cook, ATRAC3 -- all answer to the decoder, and a period-correct
Spark+MP3 FLV and an RV10 .rm both round-tripped.

Video frames come back upright. A phone stores landscape and writes a
display matrix; PyAV surfaces it as `VideoFrame.rotation`
(PyAV-Org/PyAV@040da79 av/video/frame.py:677-684, counter-clockwise degrees),
and ignoring it is the EXIF-orientation defect one medium over.
"""

from __future__ import annotations

import functools
import importlib
import logging
import os
import pathlib
import typing

if typing.TYPE_CHECKING:
    from PIL import Image

_logger = logging.getLogger(__name__)

#: Seconds before an unreadable stream is a fact rather than a hang: a
#: truncated file or network storage that stops answering costs the one
#: file, never the scan around it.
TIMEOUT = 30.0

#: The RAW family, decoded through LibRaw.
#:
#: Started as immich's list (immich-app/immich@f88fb62
#: server/src/utils/mime-types.ts:4-35) and was described here as "LibRaw's
#: coverage spelled as suffixes". It is not that. immich's table maps a
#: suffix to the MIME TYPES IT SERVES; whether LibRaw decodes the bytes is
#: a different question, and the two disagree.
#:
#: `.cin` and `.ari` were the disagreements, and both are gone.
#:
#: immich spells them `image/x-phantom-cin` and `image/x-arriflex-ari` --
#: a high-speed camera's VIDEO format and a cinema camera's -- and LibRaw
#: decodes neither. Searched LibRaw@HEAD src/, internal/ and libraw/:
#: `cineon` 0 hits; `phantom` 1, which is "DJI Phantom4 Pro/Pro+" at
#: cameralist.cpp:310; `\bARRI\b|ARRIFLEX|\bAlexa\b` 0. A bare `arri`
#: search returns 3 and every one is the word `barrier`.
#:
#: Controls in the same trees with the same flags, because 0 hits alone
#: is not absence: a format LibRaw really does read leaves a trail --
#: Sigma 102 hits across 7 files, Hasselblad 16 files including its own
#: decoder and model module, Phase One 15.
#:
#: The corpus HAD a real Arri Alexa Mini frame, CC0 from raw.pixls.us,
#: and LibRaw answered `Unsupported file format or not RAW file`. That
#: read as a corpus problem for as long as the claim stood. It was the
#: claim.
#:
#: `.cap` (Phase One) and `.k25` (Kodak DC25 -- cameralist.cpp:513,
#: identify.cpp:3256) stay: LibRaw reads both. Neither has a sample in
#: the corpus, and that is a corpus gap the ledger reports, not a claim
#: to edit.
RAW_SUFFIXES = frozenset(
    {
        ".3fr",
        ".arw",
        ".cap",
        ".cr2",
        ".cr3",
        ".crw",
        ".dcr",
        ".dng",
        ".erf",
        ".fff",
        ".iiq",
        ".k25",
        ".kdc",
        ".mrw",
        ".nef",
        ".nrw",
        ".orf",
        ".ori",
        ".pef",
        ".raf",
        ".rw2",
        ".rwl",
        ".sr2",
        ".srf",
        ".srw",
        ".x3f",
    }
)


@functools.cache
def _turns() -> dict:
    """Display-matrix rotation -> the lossless transpose that undoes it.
    `VideoFrame.rotation` is counter-clockwise; Pillow's ROTATE_* are
    counter-clockwise too, so the mapping is direct.

    A function so PIL.Image is imported when a frame needs turning, the
    way `av` is: 23ms of import (-X importtime) every app boot was paying
    for a table only a decode ever reads.
    """
    from PIL import Image

    return {
        90: Image.Transpose.ROTATE_90,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_270,
    }


@functools.cache
def ensure_decoders() -> None:
    """Register every Pillow plugin this application ships, once.

    Cached rather than flag-guarded: the registration is idempotent and the
    cache is the "once". pillow_jxl registers itself the moment it is
    imported -- the import IS the registration, which is what the explicit
    import_module states -- while pillow_heif asks to be told.
    """
    import pillow_heif

    importlib.import_module("pillow_jxl")
    pillow_heif.register_heif_opener()


def open_still(path: str | os.PathLike[str]) -> Image.Image:
    """Open any still the registry claims, whatever decodes it.

    RAW goes through LibRaw and comes back as a rendered image -- a RAW
    file is sensor data, and "the picture" is a development of it, which
    is what postprocess() performs with its neutral defaults -- except
    the flip. Every still leaves this function AS STORED, its orientation
    tag applied once, by db/oriented.py, from the tag ingest recorded.
    LibRaw's default (`user_flip=-1`, letmaik/rawpy rawpy/_rawpy.pyx
    :1282-1283, :1329) would apply the camera's flip here, and the tag
    would turn the frame a second time: every portrait CR2 sideways.
    """
    ensure_decoders()
    from PIL import Image

    if pathlib.Path(path).suffix.lower() in RAW_SUFFIXES:
        import numpy as np
        import rawpy
        from rawpy._rawpy import LibRawError

        # LibRawError descends from Exception and nothing else
        # (letmaik/rawpy@326494be83cb rawpy/_rawpy.pyx:346), so it falls
        # outside ITEM_FAILURES (7016dab db/runner.py:36) and a file
        # LibRaw cannot read ended the whole job instead of failing as one
        # item -- against that module's stated contract (7016dab
        # db/runner.py:9-15).
        #
        # Measured over ../sg-corpus at 7cf254e: 9 of 811 image-kind files
        # raise here. CanonRaw.cr2/.cr3/.crw, FujiFilm.raf, Minolta.mrw,
        # PhaseOne.iiq, Sigma.x3f, SigmaDP2.x3f, Nikon.nef -- ExifTool
        # specimens (exiftool/exiftool@2200871d9cef t/images) truncated to
        # their metadata. Each one alone stopped a full-library scan.
        #
        # Translated, not swallowed: the file still fails. `_raw_preview`
        # below already treats LibRawError as expected.
        #
        # ValueError, not OSError, and the difference is load-bearing: both
        # are in ITEM_FAILURES so the job survives either way, but the
        # thumbnail route turns ValueError into a 404 and anything else
        # into a 500 (sg_web/app.py:1216-1227). derive.py:257 raises
        # ValueError for the same situation. A file with no picture in it
        # is a 404, not a defect.
        try:
            with rawpy.imread(os.fspath(path)) as raw:
                rendered = raw.postprocess(user_flip=0)
        except LibRawError as why:
            raise ValueError(f"{path}: LibRaw cannot read this file: {why}") from why
        array = np.asarray(rendered)
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        return Image.fromarray(array)
    return Image.open(path)


def open_bounded(path: str | os.PathLike[str], want: int, *, edge: str = "longest") -> Image.Image:
    """A still bounded to `want` on one edge, decoded cheaply.

    The same picture `open_still` returns, except that nothing here ever
    owns more pixels than the caller said it needs. That is a different
    contract from `oriented.for_model`, which owes a model the real
    pixels, and the two must not be collapsed: a thumbnailer discarding
    97% of a decode and an embedder measuring it are not the same job,
    and changing what a model sees changes what it records.

    Two shortcuts, both the decoder's own:

    JPEG carries its image at several scales, so asking before `load()`
    lets libjpeg return a smaller one -- `draft` configures the reader,
    and it is a no-op once the reader is configured or the format has no
    such trick (python-pillow/Pillow src/PIL/Image.py:2899-2904, which is
    how `thumbnail()` uses it). Measured on 22-megapixel JPEGs: 107 ms of
    decode became 74, and every later phase then works on 5.5 megapixels
    instead of 22.

    RAW files carry a full JPEG preview so the camera's own screen has
    something to show. `postprocess()` develops the sensor instead, which
    is right for a model and absurd for a thumbnail: 1398 ms against 47.
    The preview is used only when it is at least as large as the biggest
    derivative asked for, so a shortcut never costs a worse picture, and
    development remains the fallback.

    WHICH edge is bounded is the caller's, and the two answers are not
    interchangeable. A derivative wants its LONGEST side capped: a 1440
    preview is 1440 at its widest whatever its shape. A model wants its
    SHORTEST side floored, because a transform that resizes the short
    edge to 224 and centre-crops (open_clip's, and most others') would
    UPSCALE from anything smaller -- inventing detail the original had,
    which is worse than the decode it saved.

    Orientation is NOT applied here. Every still leaves this module as
    stored, turned once by db/oriented.py from the tag ingest recorded --
    the same rule `open_still` states, for the same reason.
    """
    ensure_decoders()
    if pathlib.Path(path).suffix.lower() in RAW_SUFFIXES:
        # The embedded preview is accepted on the same test either way:
        # it must not be smaller than what was asked for.
        preview = _raw_preview(path, want, edge=edge)
        if preview is not None:
            return preview
        return open_still(path)
    from PIL import Image

    opened = Image.open(path)
    _draft_to_edge(opened, want, edge)
    return opened


def _draft_to_edge(opened: Image.Image, want: int, edge: str = "longest") -> None:
    """Ask the reader for the smallest scale that still covers `want`.

    The box must keep the picture's own aspect. `draft` chooses a scale
    that covers the request in BOTH directions, so asking for a square
    `(want, want)` overshoots by the aspect ratio: a 5760x3840 JPEG asked
    for (1440, 1440) comes back 2880x1920, because 1440x960 is under 1440
    tall. Four times the pixels, for a derivative whose longest side is
    1440 either way. Asked as (1440, 960) it comes back 1440x960.

    `edge` says which side the bound is on: "longest" caps it, which is
    what a derivative wants, and "shortest" floors it, which is what a
    model's own transform needs.

    A no-op for every reader without scales to choose from, and for a
    picture already smaller than `want`.
    """
    width, height = opened.size
    measured = max(width, height) if edge == "longest" else min(width, height)
    if measured <= want:
        return
    scale = want / measured
    opened.draft(None, (max(1, round(width * scale)), max(1, round(height * scale))))


def _raw_preview(path: str | os.PathLike[str], want: int, *, edge: str = "longest") -> Image.Image | None:
    """A RAW file's embedded JPEG preview, or None to develop it instead.

    None covers every case where the shortcut is not honestly available:
    no preview, a preview LibRaw cannot hand over, a bitmap rather than
    JPEG, or one too small to serve the derivative being asked for.
    """
    import rawpy

    # rawpy re-exports these unmarked; the stub defines them here, which
    # is why `dimensions` imports LibRawError from the same place.
    from rawpy._rawpy import (
        LibRawError,
        LibRawNoThumbnailError,
        LibRawUnsupportedThumbnailError,
        ThumbFormat,
    )

    try:
        with rawpy.imread(os.fspath(path)) as raw:
            found = raw.extract_thumb()
    except (LibRawNoThumbnailError, LibRawUnsupportedThumbnailError, LibRawError, OSError, ValueError) as why:
        _logger.debug("%s: no embedded preview, developing instead: %s", path, why)
        return None
    if found.format != ThumbFormat.JPEG or not isinstance(found.data, bytes):
        return None
    preview = open_bytes(found.data)
    _draft_to_edge(preview, want, edge)
    preview.load()
    if (max(preview.size) if edge == "longest" else min(preview.size)) < want:
        # Smaller than what was asked for. Enlarging a preview is not a
        # shortcut, it is a worse picture, so pay for the development.
        return None
    return preview


def open_header(path: str | os.PathLike[str]) -> Image.Image:
    """The file as its container presents it -- format, EXIF, size --
    with no RAW development: a CR2 opens as the TIFF it is, so its
    orientation tag is readable; the pixels of a RAW are open_still's."""
    ensure_decoders()
    from PIL import Image

    return Image.open(path)


def open_bytes(data: bytes) -> Image.Image:
    """A still from bytes already in memory -- served thumbnails, a
    buffer a test wrote -- through the same registered decoders."""
    import io

    ensure_decoders()
    from PIL import Image

    return Image.open(io.BytesIO(data))


def dimensions(path: str | os.PathLike[str], kind: str) -> tuple[int, int] | None:
    """The media's geometry from its headers alone -- no frame decoded,
    no RAW developed. Video answers from the stream's declared size, RAW
    from LibRaw's size block, and every still from the registered Pillow
    opener's header -- the same readers open_still and poster use, so a
    HEIC or JXL answers on a cold process instead of only after some
    unrelated decode happened to register the plugins. None when the
    file cannot say, so an unreadable member ranks last wherever
    geometry decides."""
    if kind == "video":
        import av
        from av.error import FFmpegError

        try:
            with av.open(os.fspath(path), "r", timeout=TIMEOUT) as container:
                if not container.streams.video:
                    return None
                stream = container.streams.video[0]
                if not stream.width or not stream.height:
                    return None
                return int(stream.width), int(stream.height)
        except (FFmpegError, OSError, ValueError) as why:
            _logger.warning("%s: no dimensions: %s: %s", path, type(why).__name__, why)
            return None
    if pathlib.Path(path).suffix.lower() in RAW_SUFFIXES:
        import rawpy
        from rawpy._rawpy import LibRawError  # rawpy re-exports it unmarked; the stub defines it here

        try:
            with rawpy.imread(os.fspath(path)) as raw:
                held = raw.sizes
                return int(held.width), int(held.height)
        except (LibRawError, OSError, ValueError) as why:  # the documented name (docs/api/exceptions.rst)
            _logger.warning("%s: no dimensions: %s: %s", path, type(why).__name__, why)
            return None
    ensure_decoders()
    from PIL import Image

    try:
        with Image.open(path) as image:
            return int(image.size[0]), int(image.size[1])
    except (OSError, ValueError, Image.DecompressionBombError) as why:
        _logger.warning("%s: no dimensions: %s: %s", path, type(why).__name__, why)
        return None


def is_animated(image: Image.Image) -> bool:
    """Whether this picture moves -- a per-file fact, never a suffix fact.

    An animated WebP, AVIF, APNG or HEIF wears the same suffix as its
    still sibling; only the decoded file can say. Suffix-based kinds
    called every .webp a still, which is how animated ones would have
    been filed wrong.
    """
    return bool(getattr(image, "is_animated", False))


def _upright(frame) -> Image.Image:
    """A decoded video frame, turned the way the display matrix asks."""
    image = frame.to_image()
    turn = _turns().get(int(frame.rotation) % 360)
    return image if turn is None else image.transpose(turn)


def frames_at(path: str | os.PathLike[str], offsets_ms):
    """Yield `(offset_ms, upright PIL image)` for each requested moment.

    One container open for the whole pass. Each seek lands on the keyframe
    at or before the target -- av's default backward seek -- and decoding
    then runs FORWARD to the frame whose presentation time reaches the
    target (`frame.time`, PyAV-Org/PyAV@040da79 av/frame.py:127-137).
    Taking the first frame after the seek instead returned the keyframe
    itself: with x264's default 250-frame GOP that was pixels from up to
    ten seconds before the moment the sample row claims, and every sample
    of a one-keyframe clip was silently frame zero. A moment past the last
    frame yields the last frame -- the probed duration bounds the ask, and
    the end of a stream is its own closest moment.
    """
    import av
    from av.error import FFmpegError

    # FFmpegError descends from Exception; its subclasses descend from
    # assorted builtins (PyAV-Org/PyAV@040da79 av/error.pyi:9,27,59), some
    # covered by ITEM_FAILURES and some not. Measured over ../sg-corpus at
    # 7cf254e: 14 video-kind files, 8 decode, 6 fail. ASF.wmv raised
    # InvalidDataError (a ValueError) and failed as one item; Matroska.mkv
    # raised EOFError and ended the whole job. Which container a file was
    # decided which -- for the same fact about the file.
    #
    # EVERY FFmpegError is translated, not only the uncovered ones: one
    # rule cannot drift out of step with a taxonomy in another package.
    # Translated here, like the LibRawError above, so format knowledge
    # stays in one layer and db/runner.py need not import av.
    #
    # To ValueError for the reason `open_still` states: the thumbnail route
    # answers 404 for a ValueError and 500 for anything else. Translating
    # these to OSError instead turned a truncated mp4 into a 500 with a
    # traceback, which is the exact regression
    # test_a_file_with_no_decodable_frame_is_a_404_not_a_500 exists to catch.
    try:
        with av.open(os.fspath(path), "r", timeout=TIMEOUT) as container:
            if not container.streams.video:
                return
            stream = container.streams.video[0]
            for offset_ms in offsets_ms:
                container.seek(int(offset_ms) * 1000)
                target = int(offset_ms) / 1000.0
                found = None
                for frame in container.decode(stream):
                    found = frame
                    if frame.time is None or frame.time >= target:
                        break
                if found is not None:
                    yield int(offset_ms), _upright(found)
    except FFmpegError as why:
        raise ValueError(f"{path}: cannot be read as video: {why}") from why


def poster(path: str | os.PathLike[str], offset_ms: int = 0) -> Image.Image | None:
    """The one frame that stands for a video, upright, or None."""
    for _, image in frames_at(path, [offset_ms]):
        return image
    return None
