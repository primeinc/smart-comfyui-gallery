"""Raw metadata container extraction (Pillow-only, no piexif dependency).

One Image.open per file; every adapter then works off the same RawMetadata
snapshot, so detection never re-reads the file.
"""

import contextlib
import json
import logging
from dataclasses import dataclass, field

from vision import decode

from .stealth import read_stealth_metadata

_logger = logging.getLogger(__name__)

# EXIF tags used by the tools we support.
_TAG_MAKE = 0x010F  # ComfyUI WebP: "workflow:{...}"
_TAG_MODEL = 0x0110  # ComfyUI WebP: "prompt:{...}"; legacy SwarmUI: sui JSON
_TAG_SOFTWARE = 0x0131
_TAG_USER_COMMENT = 0x9286
_TAG_MAKER_NOTE = 0x927C  # Fooocus: metadata scheme name
_IFD_EXIF = 0x8769


def decode_user_comment(value) -> str | None:
    """Decode an EXIF UserComment payload (8-byte charset prefix + data)."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip("\x00 ") or None
    if not isinstance(value, bytes):
        return None
    prefix, data = value[:8], value[8:]
    if prefix.startswith(b"UNICODE"):
        # The spec leaves byte order to the writer: piexif emits UTF-16BE,
        # some tools emit UTF-16LE. Both often decode "successfully" (ASCII
        # read with the wrong order becomes CJK codepoints), so keep the
        # decode with the highest share of printable-ASCII characters.
        best = None
        for enc in ("utf-16-be", "utf-16-le"):
            try:
                text = data.decode(enc).strip("\x00 ")
            except UnicodeDecodeError:
                continue
            if not text:
                continue
            score = sum(1 for ch in text if 0x20 <= ord(ch) < 0x7F or ch in "\n\r\t") / len(text)
            if best is None or score > best[0]:
                best = (score, text)
        if best is not None:
            return best[1]
    elif prefix.startswith(b"ASCII"):
        return data.decode("ascii", errors="ignore").strip("\x00 ") or None
    else:
        data = value  # no recognized prefix: treat the whole payload as text
    return data.decode("utf-8", errors="ignore").strip("\x00 ") or None


@dataclass
class RawMetadata:
    path: str = ""
    format: str = ""  # PIL format name: PNG / JPEG / WEBP / GIF ...
    width: int = 0
    height: int = 0
    mode: str = ""
    text: dict[str, str] = field(default_factory=dict)  # PNG tEXt/iTXt + img.info strings
    user_comment: str | None = None
    exif_make: str | None = None
    exif_model: str | None = None
    exif_software: str | None = None
    maker_note: str | None = None
    xmp: str | None = None
    gif_comment: str | None = None
    #: What happened when EXIF was looked for: "present", "absent", or
    #: "failed". Three states and not a bool, because only ONE of them
    #: lets a caller skip re-opening the file, and a bool would have to
    #: fold "there is none" together with "the read threw" -- which reads
    #: a damaged file as a clean one carrying nothing.
    #:
    #: "failed" is the default so a RawMetadata built anywhere else never
    #: licenses a skip it has not earned.
    exif_state: str = "failed"
    _stealth_text: str | None = None
    _stealth_checked: bool = False
    _img: object = None  # open PIL image, only while inside load_raw()

    def stealth(self) -> str | None:
        """Lazily decode stealth-pnginfo. Only valid inside load_raw()'s scope."""
        if not self._stealth_checked:
            self._stealth_checked = True
            if self._img is not None:
                try:
                    self._stealth_text = read_stealth_metadata(self._img)
                except Exception:
                    _logger.debug("handled a failure in stealth", exc_info=True)
                    self._stealth_text = None
        return self._stealth_text

    def text_json(self, key: str):
        """Parse a text chunk as JSON, or None."""
        value = self.text.get(key)
        if not value or not isinstance(value, str):
            return None
        try:
            parsed = json.loads(value)
        except Exception:
            _logger.debug("handled a failure in text_json", exc_info=True)
            return None
        return parsed


def _as_text(value) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return None


def load_raw(filepath: str, want_stealth: bool = False, image=None) -> RawMetadata | None:
    """Open an image and snapshot every metadata container we know about.

    When want_stealth is False the pixel data is never touched, so this stays
    cheap enough for bulk indexing.

    `image` is an already-open header, for a caller that needs to read the
    same file twice. This module's whole premise is one `Image.open` per
    file, and the cost is not theoretical: a generated PNG carries its
    workflow graph in its text chunks and Pillow parses them during
    `open`, so each one costs about 23 ms. A caller holding the image
    passes it rather than paying that again. The handle stays the
    caller's -- it is not closed here.
    """
    try:
        with contextlib.nullcontext(image) if image is not None else decode.open_header(filepath) as img:
            raw = RawMetadata(
                path=filepath,
                format=img.format or "",
                width=img.width,
                height=img.height,
                mode=img.mode,
            )
            for key, value in (img.info or {}).items():
                if key == "exif" or not isinstance(key, str):
                    continue
                text = _as_text(value)
                if text is not None:
                    if key == "comment" and img.format == "GIF":
                        raw.gif_comment = text
                    else:
                        raw.text[key] = text
            raw.xmp = raw.text.pop("XML:com.adobe.xmp", None) or raw.text.pop("xmp", None)

            try:
                exif = img.getexif()
            except Exception:
                _logger.debug("handled a failure in load_raw", exc_info=True)
                exif = None
                raw.exif_state = "failed"
            else:
                raw.exif_state = "present" if exif else "absent"
            if exif:
                raw.exif_make = _as_text(exif.get(_TAG_MAKE))
                raw.exif_model = _as_text(exif.get(_TAG_MODEL))
                raw.exif_software = _as_text(exif.get(_TAG_SOFTWARE))
                try:
                    exif_ifd = exif.get_ifd(_IFD_EXIF)
                except Exception:
                    _logger.debug("handled a failure in load_raw", exc_info=True)
                    exif_ifd = {}
                raw.user_comment = decode_user_comment(exif_ifd.get(_TAG_USER_COMMENT))
                maker = exif_ifd.get(_TAG_MAKER_NOTE)
                raw.maker_note = decode_user_comment(maker) if maker is not None else None

            # Fallback for UserComment buried in raw EXIF bytes Pillow didn't
            # surface (mirrors the tolerant scan the gallery always used).
            if raw.user_comment is None:
                exif_bytes = (img.info or {}).get("exif")
                if isinstance(exif_bytes, bytes) and b"UNICODE" in exif_bytes:
                    idx = exif_bytes.find(b"UNICODE")
                    raw.user_comment = decode_user_comment(exif_bytes[idx:])

            if want_stealth:
                raw._img = img
                raw.stealth()  # decode while the file is still open
                raw._img = None
            return raw
    except Exception:
        _logger.debug("handled a failure in load_raw", exc_info=True)
        return None
