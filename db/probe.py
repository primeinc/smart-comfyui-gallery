"""What a container says about itself, for the media Pillow cannot open.

An image tells you its size by being decoded. A video does not, and neither
does audio, so `file.duration` had no producer and `file.width`/`file.height`
were NULL for every video in the library -- while the DDL sold both as facts
about the pixels on disk. A gallery whose plan says image and video are equal
citizens cannot have one of them unable to state its own length.

ffprobe is the reader, run once per file at ingest. Three things about it
decide the shape here, all from its own documentation.

**JSON, and fields that may simply be absent.** `-output_format json`
(refs/FFmpeg/FFmpeg/doc/ffprobe.texi:86-95) and `-show_optional_fields`
defaults to `auto`, under which "JSON and XML omit the printing of fields with
invalid or non-applicable values" (:347-351). So every field is read with a
default; none may be indexed.

**A positive exit code means "not media".** "If the url cannot be opened or
recognized as a multimedia file, a positive exit code is returned" (:27-29).
That is a fact about the file, not an error to raise: one unreadable video
must cost that video, not the scan around it.

**Stored dimensions are not displayed dimensions.** A phone records landscape
and writes a display matrix saying to turn it. ffprobe reports the stored
size and the rotation separately -- verified against a real file, a 320x180
stream carrying `side_data_list: [{"side_data_type": "Display Matrix",
"rotation": 90}]` -- so a reader that takes width and height at face value
files every portrait video in the library as landscape. This is the same
defect as ignoring EXIF orientation, one medium over.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

import pypdf
import pypdf.errors

#: Both tools hang readily on a truncated file, on a path that has gone away
#: mid-read, and on network storage that stops answering. A timeout costs the
#: one file; no timeout costs the scan.
TIMEOUT = 30

#: Set to point at a particular ffprobe. Without it the one on PATH is used.
ENV_VAR = "FFPROBE_PATH"


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


def prober() -> str | None:
    """Where ffprobe is, or None. Looked up per call rather than cached, so
    installing it does not require restarting the application."""
    named = os.environ.get(ENV_VAR)
    if named and os.path.isfile(named):
        return named
    return shutil.which("ffprobe")


def _number(value) -> float | None:
    """ffprobe writes numbers as JSON strings. NaN and infinities are refused
    for the reason `capture._number` gives: a range facet silently drops a row
    it cannot compare, so storing one is worse than storing nothing."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rate(value) -> float | None:
    """`r_frame_rate` is a rational string: "30000/1001", and "0/0" when the
    container does not know. Stored as a number because "is this 60fps" is a
    question, and "30000/1001" is not an answer anything can filter on."""
    if not isinstance(value, str) or "/" not in value:
        return _number(value)
    top, _, bottom = value.partition("/")
    top, bottom = _number(top), _number(bottom)
    if not top or not bottom:
        return None
    return top / bottom


def _rotation(stream) -> int:
    """Degrees the display matrix asks for, normalised to 0/90/180/270."""
    for entry in stream.get("side_data_list") or ():
        if not isinstance(entry, dict):
            continue
        turn = _number(entry.get("rotation"))
        if turn is not None:
            return round(turn) % 360
    # Containers written before the display matrix put it in a tag instead.
    turn = _number((stream.get("tags") or {}).get("rotate"))
    return round(turn) % 360 if turn is not None else 0


def read(path) -> Probed:
    """Everything one container says, as columns plus a long tail."""
    out = Probed()
    tool = prober()
    if tool is None:
        out.unreadable = (
            f"no ffprobe on PATH and {ENV_VAR} is not set, so this file cannot state its length or its size"
        )
        return out

    try:
        finished = subprocess.run(
            [
                tool,
                "-v",
                "error",
                "-output_format",
                "json",
                "-show_format",
                "-show_streams",
                os.fspath(path),
            ],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            # Without it every file in a scan flashes a console window over
            # whatever the person is looking at.
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as problem:
        out.unreadable = f"{type(problem).__name__}: {problem}"
        return out

    if finished.returncode != 0:
        out.unreadable = (finished.stderr or "ffprobe declined the file").strip()[:400]
        return out
    try:
        document = json.loads(finished.stdout or "{}")
    except ValueError as problem:
        out.unreadable = f"ffprobe wrote something that is not JSON: {problem}"
        return out

    container = document.get("format") or {}
    streams = [s for s in document.get("streams") or () if isinstance(s, dict)]
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    # The container's duration, not the stream's. A stream frequently omits it
    # -- matroska routinely does -- while the container almost always carries
    # it, so preferring the stream leaves the common case with no length.
    out.duration = _number(container.get("duration"))
    if out.duration is None and video is not None:
        out.duration = _number(video.get("duration"))

    if video is not None:
        width, height = video.get("width"), video.get("height")
        turn = _rotation(video)
        if turn in (90, 270):
            width, height = height, width
        out.width = int(width) if isinstance(width, int) else None
        out.height = int(height) if isinstance(height, int) else None
        if turn:
            out.params.append(("Rotation", str(turn), float(turn)))

    for key, value in (
        ("Format", container.get("format_name")),
        ("BitRate", container.get("bit_rate")),
        ("VideoCodec", (video or {}).get("codec_name")),
        ("PixelFormat", (video or {}).get("pix_fmt")),
        ("FrameCount", (video or {}).get("nb_frames")),
        ("AudioCodec", (audio or {}).get("codec_name")),
        ("SampleRate", (audio or {}).get("sample_rate")),
        ("Channels", (audio or {}).get("channels")),
    ):
        if value is None or str(value).strip() == "":
            continue
        out.params.append((key, str(value).strip(), _number(value)))

    if video is not None:
        rate = _rate(video.get("r_frame_rate"))
        if rate:
            out.params.append(("FrameRate", f"{rate:.6g}", rate))

    return out


def document(path) -> Probed:
    """How many pages a PDF has, and what it says about itself.

    A document was the last second-class citizen: `.pdf` is a kind the
    scanner recognises, and a document that cannot say how long it is has
    exactly the hole a video had before ffprobe was wired in -- `page` was a
    value in the sample table's CHECK that nothing could ever write.

    `strict=False` is pypdf's default and the right one here: "a lot of PDF
    files are not strictly following the specification", and the forgiving
    reader "will try to be forgiving and do something reasonable"
    (refs/py-pdf/pypdf/docs/user/robustness.md:36-50). A library is full of
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
        except (pypdf.errors.PyPdfError, AttributeError, TypeError, ValueError):
            pass
    try:
        meta = reader.metadata
    except (pypdf.errors.PyPdfError, ValueError):
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
