"""Stealth-pnginfo reader (LSB steganography).

Decodes the scheme shared by NovelAI, Forge (modules/stealth_infotext.py) and
SwarmUI's StealthMetadata option: pixels are walked column-major (x outer,
y inner); the payload is  signature + 32-bit bit-length + data,  carried in
either the alpha-channel LSB ("alpha" mode) or the R,G,B LSBs ("rgb" mode).

Signatures:
    stealth_pnginfo  alpha, plain utf-8      stealth_pngcomp  alpha, gzip
    stealth_rgbinfo  rgb,   plain utf-8      stealth_rgbcomp  rgb,   gzip
"""

import gzip
import logging

import numpy as np

_logger = logging.getLogger(__name__)

_SIG_LEN_BITS = 15 * 8  # every signature is 15 ascii bytes
_ALPHA_SIGS = {"stealth_pnginfo": False, "stealth_pngcomp": True}
_RGB_SIGS = {"stealth_rgbinfo": False, "stealth_rgbcomp": True}
_MAX_PAYLOAD_BITS = 64 * 1024 * 1024 * 8  # sanity cap: 64 MB


def _bits_to_text(bits: np.ndarray, compressed: bool):
    data = np.packbits(bits).tobytes()
    try:
        if compressed:
            data = gzip.decompress(data)
        return data.decode("utf-8", errors="ignore")
    except Exception:
        _logger.debug("handled a failure in _bits_to_text", exc_info=True)
        return None


def _decode_channel(bits: np.ndarray, signatures: dict):
    if bits.size < _SIG_LEN_BITS + 32:
        return None
    sig = np.packbits(bits[:_SIG_LEN_BITS]).tobytes().decode("utf-8", errors="ignore")
    if sig not in signatures:
        return None
    length = int.from_bytes(np.packbits(bits[_SIG_LEN_BITS : _SIG_LEN_BITS + 32]).tobytes(), "big")
    start = _SIG_LEN_BITS + 32
    if length <= 0 or length > _MAX_PAYLOAD_BITS or start + length > bits.size:
        return None
    return _bits_to_text(bits[start : start + length], signatures[sig])


def read_stealth_metadata(img):
    """Return the embedded text from a stealth-pnginfo image, or None.

    `img` is an open PIL image. Cheap on non-stealth images: the signature
    check fails after inspecting the first bytes of the bit stream.
    """
    if img.mode not in ("RGBA", "RGB"):
        return None
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3:
        return None

    # Column-major pixel walk: transpose (H, W, C) -> (W, H, C).
    arr = arr.transpose(1, 0, 2)

    if img.mode == "RGBA":
        alpha_bits = arr[..., 3].reshape(-1) & 1
        text = _decode_channel(alpha_bits, _ALPHA_SIGS)
        if text is not None:
            return text
    rgb_bits = arr[..., :3].reshape(-1) & 1
    return _decode_channel(rgb_bits, _RGB_SIGS)
