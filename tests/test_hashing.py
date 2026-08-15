"""Tests for smartgallery_ai.hashing: sha256, phash64/dhash64, upsert, and
exact/near-duplicate queries over a synthetic files+ai_file_hashes DB."""

import os
import sqlite3
import time

import numpy as np
import pytest
from PIL import Image, ImageEnhance

from smartgallery_ai.hashing import (
    HashResult,
    compute_hashes_for_file,
    dhash64,
    find_exact_duplicates,
    find_near_duplicates,
    hamming64,
    near_duplicate_pairs,
    phash64,
    sha256_file,
    to_signed64,
    to_unsigned64,
    upsert_hashes,
)
from smartgallery_ai.schema import init_schema


# --- fixtures / helpers -----------------------------------------------------


def make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE files (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            mtime REAL NOT NULL,
            name TEXT NOT NULL,
            type TEXT
        )
        """
    )
    init_schema(conn)
    return conn


def add_file(conn, file_id, mtime=1000.0, file_type="image"):
    conn.execute(
        "INSERT INTO files (id, path, mtime, name, type) VALUES (?, ?, ?, ?, ?)",
        (file_id, f"/gallery/{file_id}.png", mtime, file_id, file_type),
    )
    conn.commit()


def smooth_photo(seed: int, size: int = 256, low: int = 16) -> Image.Image:
    """A believable low-frequency 'photo': upsampled random blobs, not noise."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(low, low, 3), dtype=np.uint8)
    return Image.fromarray(small, mode="RGB").resize((size, size), Image.BICUBIC)


def checkerboard(size: int = 256, cell: int = 32) -> Image.Image:
    arr = np.zeros((size, size), dtype=np.uint8)
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                arr[y : y + cell, x : x + cell] = 255
    return Image.fromarray(arr, mode="L").convert("RGB")


def radial_gradient(size: int = 256) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size]
    cx = cy = size / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    normed = (dist / dist.max() * 255).astype(np.uint8)
    return Image.fromarray(normed, mode="L").convert("RGB")


# --- phash64 -----------------------------------------------------------------


def test_phash_identical_image_distance_zero():
    img = smooth_photo(seed=1)
    assert hamming64(phash64(img), phash64(img.copy())) == 0


def test_phash_brightness_contrast_change_small_distance():
    base = smooth_photo(seed=1)
    adjusted = ImageEnhance.Contrast(ImageEnhance.Brightness(base).enhance(1.15)).enhance(1.15)
    assert hamming64(phash64(base), phash64(adjusted)) <= 10


def test_phash_gaussian_noise_variant_bounded_distance():
    base = smooth_photo(seed=1)
    arr = np.asarray(base, dtype=np.float32)
    rng = np.random.default_rng(101)
    noisy_arr = np.clip(arr + rng.normal(0, 25, arr.shape), 0, 255).astype(np.uint8)
    noisy = Image.fromarray(noisy_arr, mode="RGB")
    assert hamming64(phash64(base), phash64(noisy)) <= 16


def test_phash_structurally_different_images_large_distance():
    dist = hamming64(phash64(checkerboard()), phash64(radial_gradient()))
    assert dist >= 20


def test_phash_returns_python_int_in_signed_range():
    value = phash64(smooth_photo(seed=2))
    assert isinstance(value, int)
    assert -(1 << 63) <= value < (1 << 63)


# --- dhash64 -------------------------------------------------------------


def test_dhash_identical_image_distance_zero():
    img = smooth_photo(seed=3)
    assert hamming64(dhash64(img), dhash64(img.copy())) == 0


def test_dhash_distinguishes_structurally_different_images():
    dist = hamming64(dhash64(checkerboard()), dhash64(radial_gradient()))
    assert dist > 0


def test_dhash_small_change_stays_close():
    base = smooth_photo(seed=3)
    adjusted = ImageEnhance.Brightness(base).enhance(1.1)
    assert hamming64(dhash64(base), dhash64(adjusted)) <= 10


# --- hamming64 / signed round-trip -----------------------------------------


def test_hamming64_self_distance_zero_for_extreme_values():
    for value in (0, -1, 1 << 62, -(1 << 62), to_signed64(0xFFFFFFFFFFFFFFFF)):
        assert hamming64(value, value) == 0


def test_hamming64_all_bits_differ():
    assert hamming64(to_signed64(0), to_signed64(0xFFFFFFFFFFFFFFFF)) == 64


def test_to_signed_unsigned_round_trip():
    for unsigned in (0, 1, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF):
        signed = to_signed64(unsigned)
        assert -(1 << 63) <= signed < (1 << 63)
        assert to_unsigned64(signed) == unsigned


def test_signed_hash_round_trips_through_sqlite_integer():
    conn = make_conn()
    add_file(conn, "f1")
    # Values whose unsigned bit pattern has the top bit set become negative
    # once stored as a signed 64-bit two's complement SQLite INTEGER.
    tricky_unsigned = 0xFFFFFFFF00000001
    result = HashResult(sha256="deadbeef", phash64=to_signed64(tricky_unsigned), dhash64=to_signed64(0))
    upsert_hashes(conn, "f1", result, source_mtime=1000.0, algo_version="v1", now=time.time())
    row = conn.execute("SELECT phash64, dhash64 FROM ai_file_hashes WHERE file_id = ?", ("f1",)).fetchone()
    assert to_unsigned64(row[0]) == tricky_unsigned
    assert row[1] == 0


# --- sha256_file / compute_hashes_for_file --------------------------------


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(os.urandom(1 << 16) + b"\x00" * 100)
    import hashlib

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_file(str(path), chunk_size=4096) == expected


def test_compute_hashes_for_file_image(tmp_path):
    path = tmp_path / "a.png"
    smooth_photo(seed=5).save(path)
    result = compute_hashes_for_file(str(path), "image")
    assert len(result.sha256) == 64
    assert result.phash64 is not None
    assert result.dhash64 is not None


def test_compute_hashes_for_file_animated_gif_uses_first_frame(tmp_path):
    path = tmp_path / "a.gif"
    frames = [smooth_photo(seed=6), smooth_photo(seed=7)]
    frames[0].save(path, save_all=True, append_images=frames[1:])
    result = compute_hashes_for_file(str(path), "animated_image")
    with Image.open(path) as img:
        expected_phash = phash64(img.copy())
    assert result.phash64 == expected_phash


def test_compute_hashes_for_file_document_has_no_perceptual_hash(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello world")
    result = compute_hashes_for_file(str(path), "document")
    assert result.phash64 is None
    assert result.dhash64 is None
    assert len(result.sha256) == 64


def test_compute_hashes_for_file_audio_has_no_perceptual_hash(tmp_path):
    path = tmp_path / "sound.wav"
    path.write_bytes(b"RIFF....WAVEfmt ")
    result = compute_hashes_for_file(str(path), "audio")
    assert result.phash64 is None
    assert result.dhash64 is None


def test_compute_hashes_for_file_video_first_frame(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (64, 64))
    if not writer.isOpened():
        pytest.skip("no working video encoder available in this environment")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :, 0] = 200
    for _ in range(5):
        writer.write(frame)
    writer.release()

    result = compute_hashes_for_file(str(path), "video")
    assert len(result.sha256) == 64
    assert result.phash64 is not None


# --- upsert_hashes -----------------------------------------------------------


def test_upsert_hashes_then_update():
    conn = make_conn()
    add_file(conn, "f1")
    r1 = HashResult(sha256="a" * 64, phash64=1, dhash64=2)
    upsert_hashes(conn, "f1", r1, source_mtime=1000.0, algo_version="v1", now=1.0)
    r2 = HashResult(sha256="b" * 64, phash64=3, dhash64=4)
    upsert_hashes(conn, "f1", r2, source_mtime=1001.0, algo_version="v2", now=2.0)

    rows = conn.execute("SELECT sha256, phash64, dhash64, algo_version, source_mtime FROM ai_file_hashes").fetchall()
    assert rows == [("b" * 64, 3, 4, "v2", 1001.0)]


# --- find_exact_duplicates ---------------------------------------------------


def test_find_exact_duplicates_groups_by_sha256():
    conn = make_conn()
    for fid in ("f1", "f2", "f3", "f4"):
        add_file(conn, fid)
    upsert_hashes(conn, "f1", HashResult("dup", None, None), 1.0, "v1", 1.0)
    upsert_hashes(conn, "f2", HashResult("dup", None, None), 1.0, "v1", 1.0)
    upsert_hashes(conn, "f3", HashResult("unique", None, None), 1.0, "v1", 1.0)
    upsert_hashes(conn, "f4", HashResult("dup", None, None), 1.0, "v1", 1.0)

    groups = find_exact_duplicates(conn)
    assert groups == [["f1", "f2", "f4"]]


def test_find_exact_duplicates_empty_when_all_unique():
    conn = make_conn()
    add_file(conn, "f1")
    add_file(conn, "f2")
    upsert_hashes(conn, "f1", HashResult("a", None, None), 1.0, "v1", 1.0)
    upsert_hashes(conn, "f2", HashResult("b", None, None), 1.0, "v1", 1.0)
    assert find_exact_duplicates(conn) == []


# --- find_near_duplicates / near_duplicate_pairs ---------------------------


def _phash_conn_with(entries):
    """entries: dict[file_id -> unsigned 64-bit phash pattern]."""
    conn = make_conn()
    for fid, unsigned_hash in entries.items():
        add_file(conn, fid)
        result = HashResult(sha256=fid, phash64=to_signed64(unsigned_hash), dhash64=None)
        upsert_hashes(conn, fid, result, source_mtime=1.0, algo_version="v1", now=1.0)
    return conn


def test_find_near_duplicates_excludes_self_and_sorts_by_distance():
    base = 0b1010101010101010101010101010101010101010101010101010101010101010 & 0xFFFFFFFFFFFFFFFF
    entries = {
        "target": base,
        "dist1": base ^ 0b1,          # hamming distance 1
        "dist2": base ^ 0b11,         # hamming distance 2
        "dist5": base ^ 0b11111,      # hamming distance 5
        "far": base ^ 0xFFFFFFFF,     # hamming distance 32, excluded by max_distance
    }
    conn = _phash_conn_with(entries)

    results = find_near_duplicates(conn, "target", max_distance=8)
    assert [r[0] for r in results] == ["dist1", "dist2", "dist5"]
    assert [r[1] for r in results] == [1, 2, 5]
    assert all(fid != "target" for fid, _ in results)


def test_find_near_duplicates_unknown_file_returns_empty():
    conn = make_conn()
    assert find_near_duplicates(conn, "nope", max_distance=8) == []


def test_find_near_duplicates_deterministic_tie_order():
    base = 0
    conn = _phash_conn_with({"target": base, "b": base ^ 0b1, "a": base ^ 0b1, "c": base ^ 0b1})
    results = find_near_duplicates(conn, "target", max_distance=8)
    # all three tie at distance 1: must break ties by ascending file_id
    assert [r[0] for r in results] == ["a", "b", "c"]


def test_near_duplicate_pairs_full_sweep_matches_pairwise_hamming():
    entries = {
        "f1": 0,
        "f2": 0b1,
        "f3": 0b111,
        "f4": 0xFFFFFFFFFFFFFFFF,
    }
    conn = _phash_conn_with(entries)
    pairs = near_duplicate_pairs(conn, max_distance=3)

    # brute-force reference using hamming64 directly
    ids = list(entries)
    expected = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = bin(entries[ids[i]] ^ entries[ids[j]]).count("1")
            if d <= 3:
                expected.append(tuple(sorted((ids[i], ids[j]))) + (d,))
    expected.sort(key=lambda t: (t[2], t[0], t[1]))

    normalized = sorted((tuple(sorted((a, b))) + (d,) for a, b, d in pairs), key=lambda t: (t[2], t[0], t[1]))
    assert normalized == expected


def test_near_duplicate_pairs_empty_with_fewer_than_two_rows():
    conn = make_conn()
    add_file(conn, "f1")
    upsert_hashes(conn, "f1", HashResult("a", to_signed64(0), None), 1.0, "v1", 1.0)
    assert near_duplicate_pairs(conn, max_distance=8) == []


# --- per-bit sensitivity: every bit in the 64-bit masks makes a difference ---


def test_hamming64_every_single_bit_flip_is_distance_one():
    """Flipping any one of the 64 bit positions changes the Hamming
    distance by exactly 1 — including bit 63, whose flip crosses the
    signed/unsigned two's-complement boundary SQLite stores."""
    from smartgallery_ai.hashing import hamming64, phash64, to_signed64

    # Arrange: one irregular synthetic pattern and one real image hash.
    bases = [0x5DEECE66DAB0FF1E, to_signed64(phash64(checkerboard()))]
    for base in bases:
        for bit in range(64):
            # Act: flip exactly one bit (in signed storage representation).
            flipped = to_signed64(base ^ (1 << bit))
            # Assert: distance is exactly 1, symmetrically.
            assert hamming64(base, flipped) == 1, f"bit {bit} of {base:#x}"
            assert hamming64(flipped, base) == 1, f"bit {bit} of {base:#x}"


def test_near_duplicate_query_is_per_bit_exact_at_the_threshold():
    """The near-duplicate predicate distinguishes every individual bit:
    all 64 one-bit neighbors are inside max_distance=1, none are inside
    max_distance=0, and a two-bit neighbor is excluded at 1."""
    from smartgallery_ai.hashing import HashResult, find_near_duplicates, to_signed64, upsert_hashes

    conn = make_conn()
    target_phash = 0x0123456789ABCDEF

    # Arrange: the target plus one neighbor per flipped bit, plus one
    # two-bit neighbor.
    add_file(conn, "target")
    upsert_hashes(conn, "target",
                  HashResult(sha256="t" * 64, phash64=to_signed64(target_phash), dhash64=0),
                  1000.0, "algo-v1", 2000.0)
    for bit in range(64):
        fid = f"flip{bit:02d}"
        add_file(conn, fid)
        upsert_hashes(conn, fid,
                      HashResult(sha256=f"{bit:064d}"[:64],
                                 phash64=to_signed64(target_phash ^ (1 << bit)), dhash64=0),
                      1000.0, "algo-v1", 2000.0)
    add_file(conn, "twobits")
    upsert_hashes(conn, "twobits",
                  HashResult(sha256="u" * 64,
                             phash64=to_signed64(target_phash ^ 0b11), dhash64=0),
                  1000.0, "algo-v1", 2000.0)

    # Act
    at_one = find_near_duplicates(conn, "target", max_distance=1)
    at_zero = find_near_duplicates(conn, "target", max_distance=0)

    # Assert: exactly the 64 single-bit neighbors at distance 1; nothing
    # at distance 0; the two-bit neighbor excluded.
    assert [fid for fid, _ in at_one] == sorted(f"flip{b:02d}" for b in range(64))
    assert all(dist == 1 for _, dist in at_one)
    assert at_zero == []
    assert "twobits" not in {fid for fid, _ in at_one}


def test_near_duplicate_pairs_faiss_and_numpy_paths_agree(monkeypatch):
    """The IndexBinaryFlat sweep and the numpy XOR+popcount sweep must return
    the identical pair list, including the boundary distance (radius is
    exclusive in faiss range_search, inclusive in our contract)."""
    import sys

    pytest.importorskip("faiss")
    entries = {
        "a": 0,
        "b": 0b1,
        "c": 0b1111,  # exactly max_distance from b at 3
        "d": 0xFFFFFFFFFFFFFFFF,
        "e": 0b111,  # exactly max_distance from a at 3
    }
    conn = _phash_conn_with(entries)
    via_faiss = near_duplicate_pairs(conn, max_distance=3)
    monkeypatch.setitem(sys.modules, "faiss", None)
    via_numpy = near_duplicate_pairs(conn, max_distance=3)
    assert via_faiss == via_numpy
    assert any(d == 3 for _, _, d in via_faiss)  # boundary distance included
