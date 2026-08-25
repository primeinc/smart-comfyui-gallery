"""What a container says about itself, for the media Pillow cannot open.

An image tells you its size by being decoded. A video does not, and neither
does audio, so `file.duration` had no producer and `file.width`/`file.height`
were NULL for every video in the library -- while the DDL sold both as facts
about the pixels on disk. A gallery whose plan says image and video are equal
citizens cannot have one of them unable to state its own length.

PyAV is the reader: FFmpeg's own libavformat, shipped inside the `av` wheel,
so probing needs no external binary and "ffprobe is not installed" stopped
being a state this application can be in. Opening carries a timeout for the
same reason the old subprocess did -- a truncated file or network storage
that stops answering costs the one file, never the scan around it
(`av.open(timeout=...)`, PyAV-Org/PyAV@040da79, av.open docstring).

**A file that will not open is a fact, not an error.** `av.FFmpegError` --
InvalidDataError for bytes that are not media -- is recorded as
`unreadable` and the scan continues; one bad video costs that video.

**Stored dimensions are not displayed dimensions.** A phone records
landscape and writes a display matrix saying to turn it. PyAV surfaces the
matrix on the decoded frame as `VideoFrame.rotation`
(PyAV-Org/PyAV@040da79 av/video/frame.py:677-684), so the first frame is
decoded here and the stored width and height are swapped when it asks for a
quarter turn -- the same defect as ignoring EXIF orientation, one medium
over. The decode also proves the stream's codec actually answers, which a
header parse never could.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from fractions import Fraction

import pypdf
import pypdf.errors

_logger = logging.getLogger(__name__)

#: Seconds to wait for bytes before an unreadable file is a finding.
TIMEOUT = 30.0

#: Microseconds per second: `InputContainer.duration` is in AV_TIME_BASE
#: units (PyAV-Org/PyAV@040da79 av/container/input.py:123-127).
_AV_TIME_BASE = 1_000_000


@dataclass
class Probed:
    """What one container says about itself."""

    duration: float | None = None
    width: int | None = None
    height: int | None = None
    #: (key, value_text, value_num) for the long tail, as `file_param` wants.
    params: list[tuple[str, str, float | None]] = field(default_factory=list)
    #: Why nothing could be read, when nothing could. Distinct from every
    #: field being empty, which means the file was read and had little to say.
    unreadable: str | None = None

    @property
    def is_empty(self) -> bool:
        return not (self.params or self.duration or self.width)


def _number(value) -> float | None:
    """A finite number or nothing. NaN and infinities are refused for the
    reason `capture._number` gives: a range facet silently drops a row it
    cannot compare, so storing one is worse than storing nothing."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_frame(container, stream):
    """The first decodable frame, or None -- never an exception.

    Damaged streams are what a real library contains; a stream whose
    header parses but whose frames will not decode still probes, it just
    cannot say which way up it is.
    """
    from av.error import FFmpegError

    try:
        for frame in container.decode(stream):
            return frame
    except (FFmpegError, OSError, ValueError) as why:
        _logger.warning("%s: no decodable frame: %s: %s", container.name, type(why).__name__, why)
        return None
    return None


def read(path: str | os.PathLike[str]) -> Probed:
    """Everything one container says, as columns plus a long tail."""
    import av
    from av.error import FFmpegError

    out = Probed()
    try:
        with av.open(os.fspath(path), "r", timeout=TIMEOUT, metadata_errors="replace") as container:
            video = container.streams.video[0] if container.streams.video else None
            audio = container.streams.audio[0] if container.streams.audio else None
            # A stream can exist and carry NO codec context. PyAV opens
            # the container, finds a video stream, and hands back a
            # stream whose `codec_context` is None -- which every read
            # below then dereferenced. Two real files do it, both on
            # suffixes this application claims: a Canon CR3 (ISOBMFF
            # RAW, opened as a container and described by nothing) and a
            # JPEG XL. An AttributeError out of a reader is not a
            # refusal any caller can handle.
            vcodec = video.codec_context if video is not None else None
            acodec = audio.codec_context if audio is not None else None

            # The container's duration, not the stream's: a stream
            # frequently omits it -- matroska routinely does -- while the
            # container almost always carries it.
            if container.duration is not None:
                out.duration = _number(container.duration / _AV_TIME_BASE)
            elif video is not None and video.duration is not None and video.time_base:
                out.duration = _number(video.duration * video.time_base)

            turn = 0
            if video is not None:
                frame = _first_frame(container, video)
                if frame is not None:
                    turn = int(frame.rotation) % 360
                width = int(getattr(vcodec, "width", 0) or 0) or None
                height = int(getattr(vcodec, "height", 0) or 0) or None
                if turn in (90, 270):
                    width, height = height, width
                out.width, out.height = width, height
                if turn:
                    out.params.append(("Rotation", str(turn), float(turn)))

            for key, value in (
                ("Format", container.format.name if container.format else None),
                ("BitRate", container.bit_rate or None),
                ("VideoCodec", vcodec.name if vcodec is not None else None),
                ("PixelFormat", vcodec.pix_fmt if vcodec is not None else None),
                ("FrameCount", (video.frames or None) if video is not None else None),
                ("AudioCodec", acodec.name if acodec is not None else None),
                ("SampleRate", acodec.sample_rate if acodec is not None else None),
                ("Channels", acodec.channels if acodec is not None else None),
            ):
                if value is None or str(value).strip() == "":
                    continue
                out.params.append((key, str(value).strip(), _number(value)))

            if video is not None:
                rate = video.average_rate
                if isinstance(rate, Fraction) and rate:
                    out.params.append(("FrameRate", f"{float(rate):.6g}", float(rate)))
    except (FFmpegError, OSError, ValueError) as problem:
        out.unreadable = f"{type(problem).__name__}: {problem}"
    return out


def document(path) -> Probed:
    """How many pages a PDF has, and what it says about itself.

    A document was the last second-class citizen: `.pdf` is a kind the
    scanner recognises, and a document that cannot say how long it is has
    exactly the hole a video had before the container reader was wired in
    -- `page` was a value in the sample table's CHECK that nothing could
    ever write.

    `strict=False` is pypdf's default and the right one here: "a lot of PDF
    files are not strictly following the specification", and the forgiving
    reader "will try to be forgiving and do something reasonable"
    (py-pdf/pypdf@0feaf26 docs/user/robustness.md:36-50). A library is full of
    files nobody validated, and refusing to count the pages of a slightly
    malformed one helps no one.

    An encrypted PDF raises `FileNotDecryptedError`, a subclass of
    `PdfReadError` and so of `PyPdfError` (pypdf/errors.py:19-49). That is a
    fact about the file, reported, not an error to end a scan with.
    """
    out = Probed()
    try:
        reader = pypdf.PdfReader(os.fspath(path))
        count = len(reader.pages)
    except (pypdf.errors.PyPdfError, OSError, ValueError, RecursionError) as problem:
        out.unreadable = f"{type(problem).__name__}: {problem}"
        return out

    out.params.append(("Pages", str(count), float(count)))
    # The first page's size, in PDF points, because "how big is this
    # document" is the same question a picture answers with its pixels.
    if count:
        try:
            box = reader.pages[0].mediabox
            out.width, out.height = int(float(box.width)), int(float(box.height))
        except (pypdf.errors.PyPdfError, AttributeError, TypeError, ValueError) as why:
            _logger.warning("%s: first page has no size: %s: %s", path, type(why).__name__, why)
    try:
        meta = reader.metadata
    except (pypdf.errors.PyPdfError, ValueError) as why:
        _logger.warning("%s: metadata unreadable: %s: %s", path, type(why).__name__, why)
        meta = None
    for key, value in (
        ("Title", getattr(meta, "title", None)),
        ("Author", getattr(meta, "author", None)),
        ("Producer", getattr(meta, "producer", None)),
        ("Creator", getattr(meta, "creator", None)),
    ):
        text = str(value).strip() if value is not None else ""
        if text:
            out.params.append((key, text, None))
    return out


def pages_of(conn, file_id: int, found: Probed):
    """Record one page sample per page, so a document has moments the way a
    video has moments -- somewhere for a caption or a piece of OCR to point.

    Takes the reading rather than the path: opening a PDF twice to answer one
    question is the sort of thing that is free on a fixture and costs a
    second per document on a real library.

    Returns the sample ids in page order. Idempotent, as frame sampling is:
    an interrupted job resumes rather than raising on page one.
    """
    from . import derived

    pages = next((int(n) for key, _, n in found.params if key == "Pages" and n), 0)
    return [derived.add_sample(conn, file_id, "page", "every-page", page_index=index) for index in range(pages)]


def store(conn, file_id: int, found: Probed, now: float) -> None:
    """Write one container's facts.

    The dimensions and the length go on `file` because they are what the row
    is; everything else is a searchable field like any other.
    """
    if found.unreadable is not None and found.is_empty:
        return
    conn.execute(
        "UPDATE file SET duration = COALESCE(?, duration),"
        " width = COALESCE(?, width), height = COALESCE(?, height) WHERE id = ?",
        (found.duration, found.width, found.height, file_id),
    )
    for key, text, number in found.params:
        conn.execute(
            "INSERT INTO file_param(file_id, source, key, value_text, value_num)"
            " VALUES(?, 'container', ?, ?, ?)"
            " ON CONFLICT(file_id, source, key) DO UPDATE SET"
            " value_text = excluded.value_text, value_num = excluded.value_num",
            (file_id, key, text, number),
        )
