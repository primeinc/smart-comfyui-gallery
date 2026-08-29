"""The corpus, indexed by content and never copied.

The sample datasets already on this machine are what the production face
benchmarks run against (`benchmarks/face_pipeline_validation.py:38`), and the
KYC set is the one that matters here: 7 identities, 15 images each -- 2 ID
photos and 13 selfies -- with a CSV giving id, age, gender and country.

That single set supplies four axes a synthetic fixture cannot:

    multi-reference       15 images of one person, so `A,B` against `B,A` and
                          `A,A` can separate a reference SET from an ordered
                          sequence from a centroid
    negative control      7 identities to cross, so a replay that "works" for
                          the wrong person is caught
    capture path          an ID photo and a selfie of one person came through
                          different optics and different processing. What this
                          application stores is already a derived rendering
                          (`vision/oriented.for_model`, EXIF-turned, capped at
                          1600), so a corpus of one capture path proves replay
                          only for our own renderings
    demography            age, gender and country per identity, recorded rather
                          than assumed

NOT VENDORED. The licence is cc-by-nc-nd-4.0 -- non-commercial, no derivatives
-- so nothing here copies, resizes or rewrites a single file. The index holds
absolute paths and sha256 digests, which is the same rule `docs/BACKLOG.md`
states for the ExifTool corpus: "referenced by checksum and never vendored".

A digest rather than a path is what makes the evidence honest: a file edited
in place under the same name would otherwise invalidate every baseline
recorded against it without changing one line of the index.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

#: Where the sample datasets live on this machine, the same root the
#: production face benchmark names. Overridable because a hardcoded drive
#: letter makes the corpus lanes unrunnable elsewhere.
DATASETS: Path = Path(os.environ.get("COMPAT_DATASETS", "C:/ComfyUI/output/sample-datasets"))

#: The labelled identity set. Seven folders, one per person.
KYC: Path = DATASETS / "caucasian-people-kyc-photo-dataset"

#: Read from the dataset's own README rather than assumed. Recorded in the
#: index so a reviewer sees the terms without opening the corpus.
LICENCE: str = "cc-by-nc-nd-4.0"

CHUNK: int = 1 << 20


@dataclass(frozen=True)
class Sample:
    """One corpus image, by content and by what is known about it."""

    identity: str
    role: str
    path: str
    sha256: str
    bytes: int
    age: int | None = None
    gender: str = ""
    country: str = ""


def digest_file(path: Path) -> tuple[str, int]:
    """sha256 and size, streamed. Never loads a whole image to hash it."""
    hasher = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            hasher.update(chunk)
            total += len(chunk)
    return hasher.hexdigest(), total


def read_labels(csv_path: Path) -> dict[str, dict[str, str]]:
    """The dataset's own CSV, parsed with the csv module.

    A real parser, not a split on commas: a country or gender field is free
    text as far as this is concerned, and a quoted comma would silently shift
    every column after it.
    """
    out: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            identity = (row.get("id") or "").strip()
            if identity:
                out[identity] = row
    return out


def role_of(name: str) -> str:
    """ID photo or selfie, from the dataset's own filename convention.

    The distinction is the capture-path axis, so it is recorded per file
    rather than inferred later from a folder.
    """
    lowered = name.lower()
    if lowered.startswith("id_"):
        return "id_document"
    if lowered.startswith("selfie_"):
        return "selfie"
    return "unknown"


def scan_kyc(root: Path = KYC) -> list[Sample]:
    """Every KYC image, hashed, with its identity and demography attached."""
    labels = read_labels(root / "caucasian_kyc_dataset.csv")
    out: list[Sample] = []
    files = root / "files"
    for folder in sorted(files.iterdir(), key=lambda one: one.name):
        if not folder.is_dir():
            continue
        row = labels.get(folder.name, {})
        raw_age = (row.get("age") or "").strip()
        for image in sorted(folder.iterdir(), key=lambda one: one.name):
            if not image.is_file():
                continue
            sha, size = digest_file(image)
            out.append(
                Sample(
                    identity=folder.name,
                    role=role_of(image.name),
                    path=str(image),
                    sha256=sha,
                    bytes=size,
                    age=int(raw_age) if raw_age.isdigit() else None,
                    gender=(row.get("gender") or "").strip(),
                    country=(row.get("country") or "").strip(),
                )
            )
    return out


def summarise(samples: list[Sample]) -> dict[str, object]:
    identities = sorted({one.identity for one in samples})
    roles: dict[str, int] = {}
    per_identity: dict[str, int] = {}
    for one in samples:
        roles[one.role] = roles.get(one.role, 0) + 1
        per_identity[one.identity] = per_identity.get(one.identity, 0) + 1
    return {
        "identities": len(identities),
        "images": len(samples),
        "by_role": roles,
        "per_identity": per_identity,
        "distinct_content": len({one.sha256 for one in samples}),
    }


def build() -> dict[str, object]:
    samples = scan_kyc()
    return {
        "source": {
            "root": str(KYC),
            "licence": LICENCE,
            "vendored": False,
            "note": "referenced by absolute path and sha256; no file is copied, resized or rewritten",
        },
        "summary": summarise(samples),
        "samples": [asdict(one) for one in samples],
    }


def main() -> int:
    if not KYC.is_dir():
        print(f"corpus absent at {KYC}")
        print("UNSUPPORTED: consumer-tier cases cannot run without it")
        return 1

    index = build()
    out = Path(__file__).resolve().parent / "kyc.json"
    body = json.dumps(index, indent=2, sort_keys=True) + "\n"

    # Regenerated and DIFFED against the committed copy: no module reads this
    # file, so as a drift check it says whether the committed index and the
    # corpus on this machine still agree.
    was = out.read_text(encoding="utf-8") if out.is_file() else ""
    moved = bool(was) and was != body

    with out.open("w", encoding="utf-8", newline="") as handle:
        handle.write(body)

    print(json.dumps(index["summary"], indent=2, sort_keys=True))
    print(f"\nlicence: {LICENCE} (not vendored)")
    print(f"wrote {out}")
    if moved:
        print("\n!! the corpus on this machine does not match the committed index.")
        print("   Evidence recorded against the old one describes different photographs.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
