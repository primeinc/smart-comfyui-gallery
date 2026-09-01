from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

DATASETS: Path = Path(os.environ.get("COMPAT_DATASETS", "C:/ComfyUI/output/sample-datasets"))


KYC: Path = DATASETS / "caucasian-people-kyc-photo-dataset"


LICENCE: str = "cc-by-nc-nd-4.0"

CHUNK: int = 1 << 20


@dataclass(frozen=True)
class Sample:
    identity: str
    role: str
    path: str
    sha256: str
    bytes: int
    age: int | None = None
    gender: str = ""
    country: str = ""


def digest_file(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            hasher.update(chunk)
            total += len(chunk)
    return hasher.hexdigest(), total


def read_labels(csv_path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            identity = (row.get("id") or "").strip()
            if identity:
                out[identity] = row
    return out


def role_of(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("id_"):
        return "id_document"
    if lowered.startswith("selfie_"):
        return "selfie"
    return "unknown"


def scan_kyc(root: Path = KYC) -> list[Sample]:
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
