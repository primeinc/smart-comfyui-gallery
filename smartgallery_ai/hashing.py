"""Content hashing: exact duplicates (SHA-256) and near-duplicates
(perceptual hashes) -- no GPU, no model weights required.

Perceptual hashes are stored as signed 64-bit integers (SQLite INTEGER is a
signed 64-bit two's complement type) but are manipulated as unsigned 64-bit
bit patterns internally; see `to_signed64`/`to_unsigned64`.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from smartgallery_ai.faiss_runtime import import_faiss

_logger = logging.getLogger(__name__)

_MASK64 = (1 << 64) - 1  # keeps the low 64 bits of an arbitrary Python int
_SIGN_BIT64 = 1 << 63  # sign bit of a 64-bit two's complement integer

# Host-app file_type values whose frames decode via PIL / cv2 respectively.
IMAGE_FILE_TYPES = frozenset({"image", "animated_image"})
VIDEO_FILE_TYPES = frozenset({"video"})

# Bounded so a corrupt/undecodable video can't spin forever probing frames.
_MAX_VIDEO_FRAME_ATTEMPTS = 60


def to_unsigned64(value: int) -> int:
    """Reinterpret `value` as an unsigned 64-bit bit pattern."""
    return value & _MASK64


def to_signed64(value: int) -> int:
    """Reinterpret `value` as a signed 64-bit two's complement integer.

    This is the representation SQLite's INTEGER column round-trips exactly.
    """
    value &= _MASK64
    if value & _SIGN_BIT64:
        value -= 1 << 64
    return value


def sha256_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Chunked SHA-256 over file bytes (constant memory for large files)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dct_basis(n: int) -> np.ndarray:
    """Orthonormal n x n DCT-II basis matrix (built from scratch, no scipy)."""
    k = np.arange(n).reshape(-1, 1)
    x = np.arange(n).reshape(1, -1)
    basis = np.cos(np.pi / n * (x + 0.5) * k) * np.sqrt(2.0 / n)
    basis[0, :] = np.sqrt(1.0 / n)
    return basis


_DCT32 = _dct_basis(32)  # cached 32x32 basis shared by every phash64 call


def _bits_to_int(bits: np.ndarray) -> int:
    """Pack a flat 0/1 array into a 64-bit int, first element as MSB."""
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def phash64(img: Image.Image) -> int:
    """64-bit perceptual hash via 2D DCT-II (top-left 8x8 low-frequency block).

    Grayscale -> resize 32x32 (LANCZOS) -> 2D DCT -> top-left 8x8 block ->
    median of the 63 AC coefficients (DC coefficient excluded from the
    median, per the classic pHash refinement) -> bit = 1 iff coefficient
    >= median, including the DC coefficient itself in the output bits.
    """
    gray = img.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float64)
    dct = _DCT32 @ pixels @ _DCT32.T
    block = dct[:8, :8].flatten()
    ac_only = block[1:]  # drop DC (index 0) from the median computation
    median = np.median(ac_only)
    bits = (block >= median).astype(np.uint8)
    return to_signed64(_bits_to_int(bits))


def dhash64(img: Image.Image) -> int:
    """64-bit difference hash: 9x8 horizontal gradient, LEFT > RIGHT -> bit."""
    gray = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)  # shape (8, 9)
    diff = (pixels[:, :-1] > pixels[:, 1:]).astype(np.uint8).flatten()  # (8,8)
    return to_signed64(_bits_to_int(diff))


def hamming64(a: int, b: int) -> int:
    """Hamming distance between two 64-bit hashes (either signed or unsigned)."""
    return (to_unsigned64(a) ^ to_unsigned64(b)).bit_count()


@dataclass
class HashResult:
    """Hashes for one media file; perceptual fields are None when no frame
    could be decoded (non-visual or corrupt content)."""

    sha256: str  # hex digest of the raw file bytes
    phash64: int | None  # signed 64-bit form (SQLite representation)
    dhash64: int | None  # signed 64-bit form (SQLite representation)


def _first_video_frame(path: str) -> Image.Image | None:
    """First decodable frame as an RGB PIL image, or None when the container
    cannot be opened or yields nothing within the attempt budget."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        for _ in range(_MAX_VIDEO_FRAME_ATTEMPTS):
            ok, frame = cap.read()
            if not ok:
                return None
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb)
        return None
    finally:
        cap.release()


def compute_hashes_for_file(path: str, file_type: str) -> HashResult:
    """Compute SHA-256 always; pHash/dHash only when a frame can be read.

    Images (incl. animated) use PIL's first frame; video uses the first
    decodable frame via cv2.VideoCapture; audio/document files get sha256
    only (perceptual hashing is meaningless for non-visual content).
    """
    sha = sha256_file(path)
    frame: Image.Image | None = None
    if file_type in IMAGE_FILE_TYPES:
        try:
            with Image.open(path) as img:
                frame = img.copy()
        except Exception:
            _logger.debug("handled a failure in compute_hashes_for_file", exc_info=True)
            frame = None
    elif file_type in VIDEO_FILE_TYPES:
        frame = _first_video_frame(path)

    if frame is None:
        return HashResult(sha256=sha, phash64=None, dhash64=None)
    return HashResult(sha256=sha, phash64=phash64(frame), dhash64=dhash64(frame))


def upsert_hashes(
    conn,
    file_id: str,
    result: HashResult,
    source_mtime: float,
    algo_version: str,
    now: float,
) -> None:
    """Insert or fully replace the `ai_file_hashes` row for `file_id`; commits."""
    conn.execute(
        """
        INSERT INTO ai_file_hashes
            (file_id, sha256, phash64, dhash64, algo_version, source_mtime, computed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_id) DO UPDATE SET
            sha256=excluded.sha256,
            phash64=excluded.phash64,
            dhash64=excluded.dhash64,
            algo_version=excluded.algo_version,
            source_mtime=excluded.source_mtime,
            computed_at=excluded.computed_at
        """,
        (
            file_id,
            result.sha256,
            result.phash64,
            result.dhash64,
            algo_version,
            source_mtime,
            now,
        ),
    )
    conn.commit()


def find_exact_duplicates(conn) -> list[list[str]]:
    """Groups (>=2 members) of file_ids sharing an identical sha256."""
    rows = conn.execute("SELECT sha256, file_id FROM ai_file_hashes ORDER BY sha256, file_id").fetchall()
    groups: dict[str, list[str]] = {}
    for sha256, file_id in rows:
        groups.setdefault(sha256, []).append(file_id)
    return [ids for ids in groups.values() if len(ids) >= 2]


def find_near_duplicates(conn, file_id: str, max_distance: int) -> list[tuple[str, int]]:
    """Brute-force nearest phash64 neighbors of `file_id`, self excluded."""
    row = conn.execute("SELECT phash64 FROM ai_file_hashes WHERE file_id = ?", (file_id,)).fetchone()
    if row is None or row[0] is None:
        return []
    target = row[0]
    others = conn.execute(
        "SELECT file_id, phash64 FROM ai_file_hashes WHERE file_id != ? AND phash64 IS NOT NULL",
        (file_id,),
    ).fetchall()
    results = [(other_id, hamming64(target, other_phash)) for other_id, other_phash in others]
    results = [pair for pair in results if pair[1] <= max_distance]
    results.sort(key=lambda pair: (pair[1], pair[0]))
    return results


_POPCOUNT_TABLE = np.array([(i).bit_count() for i in range(256)], dtype=np.uint8)  # set-bit count per byte value 0..255


def _popcount_u64(values: np.ndarray) -> np.ndarray:
    """Vectorized popcount over an array of uint64 via a byte lookup table.

    Byte order is irrelevant here: popcount sums bits regardless of how the
    8 bytes of each uint64 are arranged.
    """
    byte_view = values.astype(np.uint64, copy=False).view(np.uint8).reshape(-1, 8)
    return _POPCOUNT_TABLE[byte_view].sum(axis=1, dtype=np.int64)


def near_duplicate_pairs(conn, max_distance: int) -> list[tuple[str, str, int]]:
    """Full near-duplicate sweep over phash64.

    Uses FAISS `IndexBinaryFlat` when installed — Hamming distance with
    popcount instructions in C++ (faiss wiki Binary indexes); its
    `range_search` returns distances strictly BELOW the radius
    (faiss tests/test_index_binary.py), hence `max_distance + 1`. Falls
    back to the chunked numpy XOR + popcount sweep. Both paths return the
    identical pair set, sorted by (distance, file_id, file_id).
    """
    rows = conn.execute(
        "SELECT file_id, phash64 FROM ai_file_hashes WHERE phash64 IS NOT NULL ORDER BY file_id"
    ).fetchall()
    if len(rows) < 2:
        return []
    ids = [r[0] for r in rows]
    values = np.array([to_unsigned64(r[1]) for r in rows], dtype=np.uint64)
    n = len(ids)
    pairs: list[tuple[str, str, int]] = []

    try:
        faiss = import_faiss()
    except ImportError:
        faiss = None
    if faiss is not None:
        packed = values.view(np.uint8).reshape(n, 8)  # 64 bits -> 8 bytes;
        # byte order is irrelevant: both sides of every XOR share the layout.
        index = faiss.IndexBinaryFlat(64)
        index.add(packed)
        lims, dists, neigh = index.range_search(packed, int(max_distance) + 1)
        for i in range(n):
            for pos in range(int(lims[i]), int(lims[i + 1])):
                j = int(neigh[pos])
                if j > i:
                    pairs.append((ids[i], ids[j], int(dists[pos])))
        pairs.sort(key=lambda t: (t[2], t[0], t[1]))
        return pairs

    for i in range(n - 1):
        xor = values[i] ^ values[i + 1 :]
        dists = _popcount_u64(xor)
        close = np.nonzero(dists <= max_distance)[0]
        for offset in close:
            j = i + 1 + int(offset)
            pairs.append((ids[i], ids[j], int(dists[offset])))
    pairs.sort(key=lambda t: (t[2], t[0], t[1]))
    return pairs
