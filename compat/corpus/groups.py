"""Real photographs holding more than one real person, by first-party count.

Source: `people_detection`, 200 photographs from Shutterstock's People
category, with `metadata/model_metadata.csv` giving one row per
(ASSET_ID, MODEL_RELEASE_ID). The number of distinct MODEL_RELEASE_ID values
for an asset is that photograph's released-person count, stated by the
dataset rather than inferred from a detector.

Licence: `LICENSE` in the dataset root. Shutterstock Evaluation Content,
60-day evaluation term from download (LICENSE:6), and LICENSE:8 prohibits
public display or transfer to any third party. So nothing here copies,
resizes, rewrites or emits a file: the index holds absolute paths and sha256
digests, the same rule `compat/corpus/index.py` applies to the KYC set.

Why this set and not a montage: selection semantics differ between consumers
only when one photograph holds several faces at different sizes
(`first` against `largest_bbox_area`). A composite pasted together from
single-subject frames proves the paste, not the detector -- the faces would
carry no shared optics, no shared lighting and no shared depth of field.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from compat.corpus.index import DATASETS, digest_file

#: Dataset root. Named here rather than assembled by a caller so the licence
#: file and the metadata cannot be read from two different copies.
PEOPLE_DETECTION: Final[Path] = DATASETS / "people_detection"

IMAGES: Final[Path] = PEOPLE_DETECTION / "files" / "medium"
MODEL_METADATA: Final[Path] = PEOPLE_DETECTION / "metadata" / "model_metadata.csv"
ASSET_METADATA: Final[Path] = PEOPLE_DETECTION / "metadata" / "asset_metadata.csv"

#: From the dataset's own LICENSE, recorded in the index so a reviewer sees
#: the terms without opening the corpus.
LICENCE: Final[str] = "shutterstock-evaluation-60d (LICENSE:6); no third-party disclosure (LICENSE:8)"


@dataclass(frozen=True)
class Group:
    """One photograph and the number of released people the dataset counts."""

    asset_id: str
    path: str
    sha256: str
    bytes: int
    released_people: int
    age_ranges: tuple[str, ...]
    description: str = ""


def released_counts(path: Path = MODEL_METADATA) -> dict[str, list[str]]:
    """ASSET_ID -> its distinct MODEL_RELEASE_ID values, in file order.

    `csv.DictReader`, not a split: DESCRIPTION-adjacent free-text fields in
    this dataset are quoted and contain commas.
    """
    out: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            asset = (row.get("ASSET_ID") or "").strip()
            release = (row.get("MODEL_RELEASE_ID") or "").strip()
            if asset and release and release not in out[asset]:
                out[asset].append(release)
    return dict(out)


def age_ranges(path: Path = MODEL_METADATA) -> dict[str, list[str]]:
    """ASSET_ID -> the AGE_RANGE of each released person."""
    out: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            asset = (row.get("ASSET_ID") or "").strip()
            if asset:
                out[asset].append((row.get("AGE_RANGE") or "").strip())
    return dict(out)


def descriptions(path: Path = ASSET_METADATA) -> dict[str, str]:
    """ASSET_ID -> the dataset's own description, for the evidence row."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            asset = (row.get("ASSET_ID") or row.get("id") or "").strip()
            if asset:
                out[asset] = (row.get("DESCRIPTION") or row.get("description") or "").strip()
    return out


def scan(least: int = 2) -> list[Group]:
    """Every photograph the dataset releases `least` or more people in.

    Sorted by sha256 so the selection does not move when a directory listing
    does, and so the same slice comes back on any machine holding the set.
    """
    if not IMAGES.is_dir() or not MODEL_METADATA.is_file():
        return []
    counts = released_counts()
    ages = age_ranges()
    described = descriptions()

    out: list[Group] = []
    for image in sorted(IMAGES.iterdir(), key=lambda one: one.name):
        if not image.is_file():
            continue
        asset = image.stem
        people = len(counts.get(asset, ()))
        if people < least:
            continue
        sha, size = digest_file(image)
        out.append(
            Group(
                asset_id=asset,
                path=str(image),
                sha256=sha,
                bytes=size,
                released_people=people,
                age_ranges=tuple(ages.get(asset, ())),
                description=described.get(asset, ""),
            )
        )
    return sorted(out, key=lambda one: one.sha256)


def summarise(groups: list[Group]) -> dict[str, object]:
    by_count: dict[int, int] = defaultdict(int)
    for one in groups:
        by_count[one.released_people] += 1
    return {
        "photographs": len(groups),
        "by_released_people": dict(sorted(by_count.items())),
        "distinct_content": len({one.sha256 for one in groups}),
    }
