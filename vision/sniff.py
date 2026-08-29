"""What the bytes say the file is, before any suffix is believed.

The suffix proposes; the content decides. A library accumulates liars --
an MP4 exported as .jpg, a HEIC a phone renamed on share, an executable
wearing .png -- and a pipeline that routes by suffix alone feeds the wrong
decoder, or worse, serves the wrong Content-Type. Sniffing is the cheap
first check (a few hundred bytes, no decode); the decoders are the proof.
Both run: sniff routes, decode confirms, and a failure of either is a
recorded fact about the file.

The core patterns are WHATWG mimesniff's -- the image table, the
audio/video table, and the MP4/WebM/MP3 signature algorithms
(whatwg/mimesniff@39aa535 mimesniff.bs:860-1360) -- because that is also the
standard a browser applies to whatever we later serve. On top of the
browser's set sit the formats a media library holds and a browser does
not sniff: the TIFF family (which is also every TIFF-based camera RAW),
HEIF brands inside ftyp, JPEG XL, Photoshop, FLV, RealMedia, MPEG
program and transport streams, MXF, FLAC, ASF and PDF.
"""

from __future__ import annotations

import logging
import os

_logger = logging.getLogger(__name__)

#: Bytes read for a sniff. The WebM scan looks at most 38 bytes in, the
#: MP4 brand walk stays inside the first box, and an MPEG transport stream
#: is recognised by sync bytes one packet (188 bytes) apart.
HEAD = 512

#: ftyp brands that are still pictures in a movie's container. HEIF and
#: AVIF live inside ISO-BMFF exactly as MP4 does; Canon's CR3 too.
_STILL_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heif", b"mif1", b"msf1", b"avif", b"avis", b"crx "}

#: ftyp brands that are sound in a movie's container.
#:
#: mimesniff does not make this distinction -- its MP4 walk returns
#: "video/mp4" for every ISO-BMFF file it recognises (mimesniff.bs:1146)
#: -- and for a browser choosing a decoder that is fine, because an
#: <audio> and a <video> element take the same container. It is not fine
#: here: `kind` decides whether a file has a picture, and calling an
#: album track a video minted it a thumbnail address, sent the renderer
#: looking for a frame that does not exist, and answered a grid of
#: 500s. This is the same extension `_STILL_BRANDS` already makes to the
#: same walk, for the same reason and one family over.
#:
#: Apple's audio brands only. `mp42` and `isom` are deliberately absent:
#: they are used by both, so the brand does not say, and the container
#: reader is what settles it.
_SOUND_BRANDS = {b"M4A ", b"M4B ", b"M4P ", b"F4A ", b"F4B "}


def _mp4_family(head: bytes) -> tuple[str, str] | None:
    """The ISO-BMFF split: mimesniff's MP4 signature walk
    (mimesniff.bs:1160-1215), extended to name the still-image brands."""
    if len(head) < 12 or head[4:8] != b"ftyp":
        return None
    brand = head[8:12]
    if brand in _STILL_BRANDS:
        return ("image", "avif" if brand in (b"avif", b"avis") else "heif")
    if brand in _SOUND_BRANDS:
        return ("audio", "m4a")
    if brand == b"crx ":
        return ("image", "cr3")
    if brand[:2] == b"qt":
        return ("video", "mov")
    if brand[:3] == b"3gp":
        return ("video", "3gp")
    return ("video", "mp4")


def _ebml(head: bytes) -> tuple[str, str] | None:
    """Matroska/WebM: the EBML magic, per mimesniff's WebM steps
    (mimesniff.bs:1216-1258). The DocType distinction does not change the
    reader here, so both come back as the matroska family."""
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return ("video", "matroska")
    return None


def _riff(head: bytes) -> tuple[str, str] | None:
    if head[:4] != b"RIFF" or len(head) < 14:
        return None
    four = head[8:12]
    if head[8:14].startswith(b"WEBPVP") or four == b"WEBP":
        return ("image", "webp")
    if four == b"AVI ":
        return ("video", "avi")
    if four == b"WAVE":
        return ("audio", "wav")
    return None


def _mpeg_ts(head: bytes) -> bool:
    """Transport stream: 0x47 sync bytes one 188-byte packet apart."""
    return len(head) > 188 and head[0] == 0x47 and head[188] == 0x47


def _mp3_sync(head: bytes) -> bool:
    """A bare MPEG audio frame header, mimesniff's MP3-without-ID3 shape
    (mimesniff.bs:1326-1360): sync bits plus sane version/layer fields."""
    if len(head) < 4 or head[0] != 0xFF or (head[1] & 0xE0) != 0xE0:
        return False
    layer = (head[1] >> 1) & 0x03
    bitrate = (head[2] >> 4) & 0x0F
    return layer != 0 and bitrate not in (0, 15)


def sniff(head: bytes) -> tuple[str, str] | None:
    """`(kind, format token)` for a byte head, or None for no opinion.

    Kind is the pipeline's word -- image, video, audio, document -- and
    the token names the sniffed container so a mismatch can say what the
    file really was.
    """
    # --- stills (mimesniff image table, mimesniff.bs:860-985, extended) ---
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return ("image", "png")
    if head[:3] == b"\xff\xd8\xff":
        return ("image", "jpeg")
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return ("image", "gif")
    if head[:2] == b"BM":
        return ("image", "bmp")
    riff = _riff(head)
    if riff:
        return riff
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        # TIFF magic is also every TIFF-based camera RAW (DNG, NEF, ARW,
        # CR2...); which one is a matter of tags deeper than a sniff. The
        # family is stills either way.
        return ("image", "tiff")
    if head[:2] == b"\xff\x0a" or head[:12] == b"\x00\x00\x00\x0cJXL \r\n\x87\n":
        return ("image", "jxl")
    if head[:4] == b"8BPS":
        return ("image", "psd")
    if head[:4] == b"\x00\x00\x00\x0c" and head[4:8] == b"jP  ":
        return ("image", "jp2")
    if head[:4] == b"\xff\x4f\xff\x51":
        return ("image", "jp2")

    # --- moving pictures and sound ---------------------------------------
    iso = _mp4_family(head)
    if iso:
        return iso
    ebml = _ebml(head)
    if ebml:
        return ebml
    if head[:3] == b"FLV":
        return ("video", "flv")
    if head[:4] == b".RMF":
        return ("video", "rm")
    if head[:4] == b"\x00\x00\x01\xba":
        return ("video", "mpeg-ps")
    if _mpeg_ts(head):
        return ("video", "mpeg-ts")
    if head[:14] == b"\x06\x0e\x2b\x34\x02\x05\x01\x01\x0d\x01\x02\x01\x01\x02":
        return ("video", "mxf")
    if head[:16] == b"\x30\x26\xb2\x75\x8e\x66\xcf\x11\xa6\xd9\x00\xaa\x00\x62\xce\x6c":
        # ASF holds both WMV and WMA; the stream types are deeper than a
        # sniff, and PyAV reads either, so the family call is enough.
        return ("video", "asf")
    if head[:4] == b"OggS":
        return ("audio", "ogg")
    if head[:4] == b"fLaC":
        return ("audio", "flac")
    if head[:3] == b"ID3" or _mp3_sync(head):
        return ("audio", "mp3")
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return ("audio", "aiff")
    if head[:2] == b"\xff\xf1" or head[:2] == b"\xff\xf9":
        return ("audio", "aac-adts")

    # --- documents --------------------------------------------------------
    if head[:5] == b"%PDF-":
        return ("document", "pdf")
    return None


def sniff_path(path: str | os.PathLike[str]) -> tuple[str, str] | None:
    """Sniff a file on disk; an unreadable file is no opinion, logged with why."""
    try:
        with open(path, "rb") as handle:
            return sniff(handle.read(HEAD))
    except OSError as why:
        _logger.warning("%s: unreadable: %s: %s", path, type(why).__name__, why)
        return None


#: Content-Type per sniff token, for serving the bytes the sniff looked at.
#: Types are the essence strings mimesniff itself matches on
#: (whatwg/mimesniff@39aa535 mimesniff.bs:860-1360); the formats beyond the
#: browser's set carry their registered types as immich serves them
#: (immich-app/immich@f88fb62 server/src/utils/mime-types.ts).
MIME = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "webp": "image/webp",
    "tiff": "image/tiff",
    "jxl": "image/jxl",
    "psd": "image/vnd.adobe.photoshop",
    "jp2": "image/jp2",
    "avif": "image/avif",
    "heif": "image/heif",
    "cr3": "image/x-canon-cr3",
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "3gp": "video/3gpp",
    "matroska": "video/x-matroska",
    "flv": "video/x-flv",
    "rm": "application/vnd.rn-realmedia",
    "mpeg-ps": "video/mpeg",
    "mpeg-ts": "video/mp2t",
    "mxf": "application/mxf",
    "asf": "video/x-ms-asf",
    "avi": "video/x-msvideo",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "aiff": "audio/aiff",
    "aac-adts": "audio/aac",
    "pdf": "application/pdf",
}


def content_type(sniffed: tuple[str, str] | None) -> str:
    """The Content-Type to serve for a sniff result. Bytes no signature
    matched are exactly what application/octet-stream is for."""
    if sniffed is None:
        return "application/octet-stream"
    return MIME.get(sniffed[1], "application/octet-stream")
