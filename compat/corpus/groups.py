from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from compat.corpus.index import DATASETS, digest_file

PEOPLE_DETECTION: Final[Path] = DATASETS / "people_detection"

IMAGES: Final[Path] = PEOPLE_DETECTION / "files" / "medium"
MODEL_METADATA: Final[Path] = PEOPLE_DETECTION / "metadata" / "model_metadata.csv"
ASSET_METADATA: Final[Path] = PEOPLE_DETECTION / "metadata" / "asset_metadata.csv"


LICENCE: Final[str] = "shutterstock-evaluation-60d (LICENSE:6); no third-party disclosure (LICENSE:8)"


@dataclass(frozen=True)
class Group:
    asset_id: str
    path: str
    sha256: str
    bytes: int
    released_people: int
    age_ranges: tuple[str, ...]
    description: str = ""


def released_counts(path: Path = MODEL_METADATA) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            asset = (row.get("ASSET_ID") or "").strip()
            release = (row.get("MODEL_RELEASE_ID") or "").strip()
            if asset and release and release not in out[asset]:
                out[asset].append(release)
    return dict(out)


def age_ranges(path: Path = MODEL_METADATA) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            asset = (row.get("ASSET_ID") or "").strip()
            if asset:
                out[asset].append((row.get("AGE_RANGE") or "").strip())
    return dict(out)


def descriptions(path: Path = ASSET_METADATA) -> dict[str, str]:
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
