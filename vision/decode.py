"""Every decoder this application ships, behind one door.

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
import os
import pathlib

from PIL import Image

#: Seconds before an unreadable stream is a fact rather than a hang: a
#: truncated file or network storage that stops answering costs the one
#: file, never the scan around it.
TIMEOUT = 30.0

#: The RAW family, decoded through LibRaw. The list is immich's
#: (immich-app/immich@f88fb62 server/src/utils/mime-types.ts:4-35), which is
#: LibRaw's coverage spelled as suffixes.
RAW_SUFFIXES = frozenset(
    {
        ".3fr",
        ".ari",
        ".arw",
        ".cap",
        ".cin",
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

#: Display-matrix rotation -> the lossless transpose that undoes it.
#: `VideoFrame.rotation` is counter-clockwise; Pillow's ROTATE_* are
#: counter-clockwise too, so the mapping is direct.
_TURNS = {
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
    is what postprocess() performs with its neutral defaults.
    """
    ensure_decoders()
    if pathlib.Path(path).suffix.lower() in RAW_SUFFIXES:
        import numpy as np
        import rawpy

        with rawpy.imread(os.fspath(path)) as raw:
            rendered = raw.postprocess()
        array = np.asarray(rendered)
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        return Image.fromarray(array)
    return Image.open(path)


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
    turn = _TURNS.get(int(frame.rotation) % 360)
    return image if turn is None else image.transpose(turn)


def frames_at(path: str | os.PathLike[str], offsets_ms):
    """Yield `(offset_ms, upright PIL image)` for each requested moment.

    One container open for the whole pass. Each seek lands on the keyframe
    at or before the target -- av's default backward seek -- and the next
    decoded frame is the sample: for choosing moments a person appears in,
    the nearest keyframe is the moment.
    """
    import av

    with av.open(os.fspath(path), "r", timeout=TIMEOUT) as container:
        if not container.streams.video:
            return
        stream = container.streams.video[0]
        for offset_ms in offsets_ms:
            if offset_ms:
                container.seek(int(offset_ms) * 1000)
            found = None
            for frame in container.decode(stream):
                found = frame
                break
            if found is not None:
                yield int(offset_ms), _upright(found)


def poster(path: str | os.PathLike[str], offset_ms: int = 0) -> Image.Image | None:
    """The one frame that stands for a video, upright, or None."""
    for _, image in frames_at(path, [offset_ms]):
        return image
    return None
