"""Perceptual identity: the same picture, whatever became of its bytes.

imagehash computes the hashes (JohannesBuchner/imagehash@4.3.2
imagehash/__init__.py): `phash` median-thresholds the low-frequency 8x8
corner of a 32x32 luma DCT, so it survives re-encoding, resizing and
generation-loss copies; `dhash` keeps horizontal gradient signs and
assists where tonal shifts move the DCT. `hash_size=8` on both is what
makes them the 64-bit values `derived_file_hash.phash64/dhash64` store.

SQLite INTEGER is SIGNED 64-bit -- the schema says so on the column --
so the unsigned hash is folded into the signed range on the way in and
unfolded for comparison. Hamming distance is the only comparison that
means anything; ordering these numerically is meaningless.
"""

from __future__ import annotations

import imagehash


def _signed64(value: int) -> int:
    return value - (1 << 64) if value >= (1 << 63) else value


def _unsigned64(value: int) -> int:
    return value + (1 << 64) if value < 0 else value


def perceptual(image) -> tuple[int, int]:
    """`(phash64, dhash64)` for one upright frame, as the signed ints the
    schema stores. The frame must already be oriented -- a hash of a
    sideways render is a hash of a different-looking picture."""
    return (
        _signed64(int(str(imagehash.phash(image)), 16)),
        _signed64(int(str(imagehash.dhash(image)), 16)),
    )


def hamming(ours: int, theirs: int) -> int:
    """Bits of disagreement between two stored hashes."""
    return (_unsigned64(ours) ^ _unsigned64(theirs)).bit_count()
