"""Raw metadata container extraction (Pillow-only, no piexif dependency).

One Image.open per file; every adapter then works off the same RawMetadata
snapshot, so detection never re-reads the file.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Optional

from PIL import Image

from .stealth import read_stealth_metadata

# EXIF tags used by the tools we support.
_TAG_MAKE = 0x010F        # ComfyUI WebP: "workflow:{...}"
_TAG_MODEL = 0x0110       # ComfyUI WebP: "prompt:{...}"; legacy SwarmUI: sui JSON
_TAG_SOFTWARE = 0x0131
_TAG_USER_COMMENT = 0x9286
_TAG_MAKER_NOTE = 0x927C  # Fooocus: metadata scheme name
_IFD_EXIF = 0x8769


def decode_user_comment(value) -> Optional[str]:
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
    format: str = ""          # PIL format name: PNG / JPEG / WEBP / GIF ...
    width: int = 0
    height: int = 0
    mode: str = ""
    text: Dict[str, str] = field(default_factory=dict)  # PNG tEXt/iTXt + img.info strings
    user_comment: Optional[str] = None
    exif_make: Optional[str] = None
    exif_model: Optional[str] = None
    exif_software: Optional[str] = None
    maker_note: Optional[str] = None
    xmp: Optional[str] = None
    gif_comment: Optional[str] = None
    _stealth_text: Optional[str] = None
    _stealth_checked: bool = False
    _img: object = None       # open PIL image, only while inside load_raw()

    def stealth(self) -> Optional[str]:
        """Lazily decode stealth-pnginfo. Only valid inside load_raw()'s scope."""
        if not self._stealth_checked:
            self._stealth_checked = True
            if self._img is not None:
                try:
                    self._stealth_text = read_stealth_metadata(self._img)
                except Exception:
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
            return None
        return parsed


def _as_text(value) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return None


def load_raw(filepath: str, want_stealth: bool = False) -> Optional[RawMetadata]:
    """Open an image and snapshot every metadata container we know about.

    When want_stealth is False the pixel data is never touched, so this stays
    cheap enough for bulk indexing.
    """
    try:
        with Image.open(filepath) as img:
            raw = RawMetadata(
                path=filepath, format=img.format or "",
                width=img.width, height=img.height, mode=img.mode,
            )
            for key, value in (img.info or {}).items():
                if key == "exif":
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
                exif = None
            if exif:
                raw.exif_make = _as_text(exif.get(_TAG_MAKE))
                raw.exif_model = _as_text(exif.get(_TAG_MODEL))
                raw.exif_software = _as_text(exif.get(_TAG_SOFTWARE))
                try:
                    exif_ifd = exif.get_ifd(_IFD_EXIF)
                except Exception:
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
        return None
