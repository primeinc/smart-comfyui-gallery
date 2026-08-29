"""Real files, pinned; and real files broken on purpose.

The synthetic corpus reaches 27% of the readers. It cannot reach more,
and not because it is small: it is written by ONE writer, so it can only
produce what that writer produces. A Canon body, a 2001 Nikon, a Pixel 8
and a truncated CR2 are four writers, and each one reaches lines the
others never touch. Measured: adding these real files takes the readers
from 27.0% to 36.3%.

THREE THINGS THIS IS NOT.

Not vendored. ExifTool is GPL-3 and its test images are collected from
mixed sources, so the bytes stay in the pinned mirror and this module
references them by commit and checksum. Nothing enters our tree, which
also settles the other problem: those files carry REAL coordinates
(Apple.jpg 53.38N, Google.jpg 40.40N), and a file that never enters the
repository never leaks from it.

Not decodable. 134 of the 194 do not open as images, and of the 60 that
do the median is 8x8 -- Phil Harvey stripped the pixels on purpose,
which is exactly why the whole corpus is smaller than one photograph.
They are here to be READ, not shown. `INTENT` records that per file so
no test ever asserts a thumbnail on a file that has no pixels.

Not a conformance suite. Nothing here states what a field should hold.
The lockfile carries a checksum and one sentence about why the file is
present; every expectation belongs to a test, computed at read time.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import os
import pathlib
import tarfile
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent

#: Where the real media lives, and how it gets there.
#:
#: NOT `../refs`: a sibling of the repository is a directory on one
#: machine. Pointing tests at it meant seven of eight skipped everywhere
#: else and the suite went green having proved nothing -- which is worse
#: than no tests, because it reads as coverage.
#:
#: Fetched from GitHub at a pinned tag instead, into a cache outside the
#: repo. ExifTool is GPL-3 and its images are collected from mixed
#: sources with real coordinates in them, so they are DOWNLOADED and
#: never committed.
CORPUS = pathlib.Path(os.environ.get("SG_CORPUS", REPO.parent / "sg-corpus"))
IMAGES = CORPUS / "exiftool"
TAG = "13.59"
TARBALL = f"https://codeload.github.com/exiftool/exiftool/tar.gz/refs/tags/{TAG}"
LOCKFILE = REPO / "tests" / "sourced.lock.json"

#: How many files the tarball's `t/images` holds at that tag. A fetch
#: that lands fewer is a fetch that went wrong, and silence about it
#: would put a half-corpus behind every measurement.
EXPECTED = 194


def fetch(force: bool = False) -> pathlib.Path:
    """Get the real media, once. Returns where it landed.

    Idempotent: present and complete means no network. This is what lets
    a test REQUIRE the corpus instead of skipping without it.
    """
    if not force and IMAGES.is_dir() and len(list(IMAGES.iterdir())) >= EXPECTED:
        return IMAGES
    IMAGES.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(TARBALL, timeout=300) as answer:
        raw = answer.read()
    got = 0
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for member in tar.getmembers():
            if "/t/images/" not in member.name or not member.isfile():
                continue
            name = member.name.split("/t/images/", 1)[1]
            if "/" in name:
                continue
            held = tar.extractfile(member)
            if held is not None:
                (IMAGES / name).write_bytes(held.read())
                got += 1
    if got < EXPECTED:
        raise RuntimeError(f"{TARBALL} yielded {got} files, expected at least {EXPECTED}")
    return IMAGES


@dataclasses.dataclass(frozen=True)
class Specimen:
    """One real file, and the single reason it is here.

    `why` is the intent label and the ONLY claim this module makes about
    a file. It is a statement about why the file was chosen, never about
    what its metadata holds -- so it cannot be falsified by learning more
    about media formats, which is what lets it be frozen.
    """

    name: str
    why: str


#: Why each file is here. Chosen from the frozen target set rather than
#: from an instinct about coverage: each names a reader branch the
#: synthetic corpus cannot reach, because only a real writer emits it.
INTENT: tuple[Specimen, ...] = (
    Specimen("Google.jpg", "OffsetTimeOriginal (+02:00) and 3-digit SubSecTimeOriginal"),
    Specimen("Apple.jpg", "an iPhone with GPS and no zone -- the NULL-offset arm of judge_capture"),
    Specimen("Pentax.jpg", "Orientation=8: a sideways photograph, which nothing synthetic writes"),
    Specimen("MWG.jpg", "Orientation=6, plus MWG-scheme metadata"),
    Specimen("GoPro.jpg", "a real BodySerialNumber, and an action camera's tag set"),
    Specimen("Canon.jpg", "a MakerNote with offset-relative internal pointers"),
    Specimen("Nikon.jpg", "a 2001 body: no subsecond, no zone, no serial -- the absent-field arms"),
    Specimen("Sony.jpg", "another maker's MakerNote dialect"),
    Specimen("IPTC.jpg", "APP13 Photoshop IRB / IPTC, a carrier metaparse must skip cleanly"),
    Specimen("XMP.jpg", "XMP in APP1, the other metadata carrier on a JPEG"),
    Specimen("ExtendedXMP.jpg", "XMP split across several APP1 segments"),
    Specimen("CanonRaw.cr2", "RAW that is TIFF-structured"),
    Specimen("CanonRaw.cr3", "RAW that is ISOBMFF, not TIFF at all -- a different reader path"),
    Specimen("Nikon.nef", "another RAW container"),
    Specimen("FujiFilm.raf", "a proprietary header wrapping an embedded JPEG"),
    Specimen("Panasonic.rw2", "RAW with nonstandard tags"),
    Specimen("Sigma.x3f", "a fully proprietary RAW container"),
    Specimen("QuickTime.heic", 'ftyp brand "mif1", not "heic" -- the brand people hardcode'),
    Specimen("QuickTime.mov", "a real QuickTime atom tree"),
    Specimen("Matroska.mkv", "a Matroska container"),
    Specimen("M2TS.mts", "an AVCHD transport stream, which no library writes by accident"),
    Specimen("RIFF.avi", "a RIFF video container"),
    Specimen("MP3.mp3", "ID3 tags on audio -- a kind the synthetic corpus never wrote at all"),
    Specimen("FLAC.flac", "Vorbis comments"),
    Specimen("RIFF.wav", "a RIFF audio container"),
    Specimen("QuickTime.m4a", "iTunes-style atoms on audio"),
    Specimen("JXL.jxl", "a naked JXL codestream -- and it CRASHES capture.read (RuntimeError)"),
    Specimen("JXL2.jxl", "ISOBMFF-boxed JXL: the same format, a different container"),
    Specimen("Jpeg2000.jp2", "JP2 boxed"),
    Specimen("Jpeg2000.j2c", "the same codec as a raw codestream"),
    Specimen("RIFF.webp", "VP8X extended WebP rather than plain VP8"),
    Specimen("Photoshop.psd", "a PSD, which db/scan.py claims and nothing exercised"),
    Specimen("BigTIFF.btf", "BigTIFF: 64-bit offsets, a different TIFF entirely"),
    Specimen("GeoTiff.tif", "TIFF carrying GeoTIFF tags"),
    Specimen("PDF.pdf", "a real PDF, versus the one Pillow writes"),
)


def available() -> bool:
    """Whether the real media is on disk. Not a reason to skip -- a
    reason to call `fetch()`."""
    return IMAGES.is_dir() and len(list(IMAGES.iterdir())) >= EXPECTED


def decodes(path: pathlib.Path) -> bool:
    """Whether the file opens as an image at all.

    MEASURED, never declared. 134 of the mirror's 194 files are truncated
    to their metadata on purpose, and when this was a field somebody
    typed it was wrong for ten of thirty-five -- which is the same defect
    as inventing EXIF, one layer up. The lockfile records what was
    measured; the gate checks the measurement still holds.
    """
    import warnings

    from vision import decode

    warnings.filterwarnings("ignore")
    with contextlib.suppress(Exception):
        held = decode.open_still(path)
        held.close()
        return True
    return False


def digest(path: pathlib.Path) -> str:
    held = hashlib.sha256()
    held.update(path.read_bytes())
    return held.hexdigest()


def specimens() -> list[tuple[Specimen, pathlib.Path]]:
    """The chosen files that are actually on disk, with their intent."""
    return [(one, IMAGES / one.name) for one in INTENT if (IMAGES / one.name).is_file()]


def lock() -> dict:
    """The lockfile's content: what was pinned, and one line per file."""
    return {
        "what": "Real media referenced by the corpus. NOT vendored -- ExifTool is GPL-3 and "
        "these images carry real GPS. The bytes stay in the pinned mirror.",
        "source": {"repo": "exiftool/exiftool", "path": "t/images", "tag": TAG, "url": TARBALL},
        "files": [
            {"name": one.name, "sha256": digest(path), "renders": decodes(path), "why": one.why}
            for one, path in specimens()
        ],
    }


def write_lock() -> pathlib.Path:
    LOCKFILE.write_text(json.dumps(lock(), indent=2) + "\n", encoding="utf-8")
    return LOCKFILE


# --- broken on purpose -------------------------------------------------------


#: How a real file gets damaged. Every reader here is full of arms for
#: files that lie about themselves, and a corpus of VALID files reaches
#: none of them -- which is most of what the frozen target set is.
BREAKAGES: tuple[str, ...] = ("truncated", "header_only", "zeroed_tail", "bitflip", "empty")


def broken(source: pathlib.Path, how: str) -> bytes:
    """`source`'s bytes, damaged the named way.

    Damage is DETERMINISTIC and derived from the file's own length, so
    the same source and the same breakage give the same bytes on every
    machine -- a corpus whose damage moved would be a corpus whose
    coverage moved.
    """
    raw = source.read_bytes()
    if how == "empty":
        return b""
    if how == "header_only":
        return raw[:64]
    if how == "truncated":
        return raw[: len(raw) // 2]
    if how == "zeroed_tail":
        keep = len(raw) // 2
        return raw[:keep] + bytes(len(raw) - keep)
    if how == "bitflip":
        if len(raw) < 40:
            return raw
        at = len(raw) // 3
        held = bytearray(raw)
        held[at] ^= 0xFF
        return bytes(held)
    raise ValueError(f"no breakage named {how!r}; one of {', '.join(BREAKAGES)}")


def wreck(into: pathlib.Path, most: int = 8) -> list[pathlib.Path]:
    """Write a damaged copy of the first `most` specimens, every way.

    Into a directory the caller owns -- never beside the mirror, which is
    read-only by policy and is somebody else's repository.
    """
    into.mkdir(parents=True, exist_ok=True)
    made: list[pathlib.Path] = []
    for _one, path in specimens()[:most]:
        for how in BREAKAGES:
            stem, suffix = path.stem, path.suffix
            target = into / f"{stem}__{how}{suffix}"
            target.write_bytes(broken(path, how))
            made.append(target)
    return made
