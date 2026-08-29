"""Real RAW files from raw.pixls.us, one per suffix the readers still miss.

`tests/needs.py` reads the declared suffixes out of `db/scan.py KIND_BY_SUFFIX` and
finds most of the RAW ones UNSATISFIED: the corpus had Canon, Nikon, Fuji,
Panasonic, Minolta, Phase One and Sigma from the ExifTool specimens, and
those are truncated to their metadata, so LibRaw refuses every one of them.
Nothing in the corpus was a RAW file a camera wrote and a decoder can open.

raw.pixls.us is the archive the darktable and RawTherapee projects keep for
exactly this: 2016 files, one per camera and per mode, each with a published
SHA-256. Only the CC0 half is taken -- 1870 of the 2016 -- because the rest
is CC BY-NC-SA and a corpus that might be redistributed cannot carry a
non-commercial clause it did not notice.

SMALLEST PER SUFFIX, not first. A suffix is a reader path, and the cheapest
file that exercises it exercises it exactly as well as a 60 MB one; the
budget is better spent on breadth. Sizes come from the index, so the choice
is made before anything is fetched.

The published checksum is verified. A file whose bytes do not match what the
index describes is not the file the provenance row names, so it is refused
rather than kept.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import time

import httpx

REPO = pathlib.Path(__file__).resolve().parent.parent

CORPUS = pathlib.Path(os.environ.get("SG_CORPUS", REPO.parent / "sg-corpus"))
IMAGES = CORPUS / "raw-cameras"
LOCKFILE = REPO / "tests" / "rawsamples.lock.json"

INDEX = "https://raw.pixls.us/json/getrepository.php?format=json"

#: The one licence taken. `co` is the index's mark for CC0; `cbna` is
#: CC BY-NC-SA and is left where it is.
CC0 = "co"

ALLOWED = ("https://raw.pixls.us/",)

#: The `+url` form is the contact, per the same convention Wikimedia's
#: UA policy accepts -- a reachable page in place of an address.
AGENT = "smart-comfyui-gallery-corpus/1.0 (+https://github.com/primeinc/smart-comfyui-gallery)"

#: raw.pixls.us is one volunteer's server, not a CDN.
PAUSE = 0.5

TROUBLE = (OSError, RuntimeError, ValueError, KeyError)


def _fetched(url: str, timeout: int) -> httpx.Response:
    if not url.startswith(ALLOWED):
        raise ValueError(f"refusing to open {url!r}: not a raw.pixls.us address")
    answer = httpx.get(url, headers={"User-Agent": AGENT}, timeout=timeout, follow_redirects=True)
    answer.raise_for_status()
    time.sleep(PAUSE)
    return answer


def filename(name: str) -> str:
    """A name this filesystem will accept, with the camera still readable.

    The archive names a frame after the body and its mode, and those names
    legally contain characters Windows refuses: `Pentax - *ist DL` and an
    Olympus row carrying quotes both failed with `[Errno 22]` and left two
    suffixes uncovered while the ledger said they had been fetched. The
    same defect was fixed in `tests/commons.py` first and not carried here,
    which is why it happened twice.

    A colon is the dangerous one and it does not raise: Windows reads
    `A (3:2).CRW` as the file `A (3` plus an NTFS alternate data stream, so
    seventeen frames landed with their extension GONE -- named
    `Canon - EOS D30 - RAW (3` -- and a suffix ledger counted them as nothing
    while the download reported success. The suffix is checked rather than
    trusted.
    """
    held = name
    for bad in '<>:"/\\|?*':
        held = held.replace(bad, "_")
    held = held.strip(". ")[:180]
    # A name that lost its suffix lost the only thing this file was fetched
    # for. The docstring says how a colon does that without raising.
    if pathlib.Path(name).suffix and not pathlib.Path(held).suffix:
        raise ValueError(f"sanitising {name!r} dropped its suffix and produced {held!r}")
    return held


def index() -> list[dict]:
    """Every CC0 frame the archive lists, as plain rows.

    The index is positional -- a list per frame, not a mapping -- so the
    columns are named here once rather than indexed by number at each use.
    """
    rows = json.loads(_fetched(INDEX, 120).text)["data"]
    out = []
    for row in rows:
        licence = re.sub("<[^>]+>", "", row[5] or "").strip()
        if licence != CC0:
            continue
        found = re.search(r"getfile\.php/\d+/nice/([^']+)", row[7] or "")
        checksum = re.search(r"sha256 Checksum'>([0-9a-f]{64})", row[7] or "")
        if not found or not checksum:
            continue
        name = found.group(1)
        out.append(
            {
                "maker": row[0],
                "model": row[1],
                "variant": row[2],
                "megabytes": row[3],
                "name": name,
                "suffix": "." + name.rsplit(".", 1)[-1].lower(),
                "url": "https://raw.pixls.us/" + found.group(0),
                "sha256": checksum.group(1),
            }
        )
    return out


def pick(rows: list[dict], suffixes: set[str]) -> list[dict]:
    """The smallest CC0 file for each wanted suffix."""

    def size(one: dict) -> float:
        # A row can carry no size at all. Sorting None as though it were zero
        # makes an unmeasured file win every comparison, so an unknown size
        # loses instead and is picked only when nothing else offers that suffix.
        held = one.get("megabytes")
        return float(held) if isinstance(held, (int, float)) else float("inf")

    best: dict[str, dict] = {}
    for one in rows:
        if one["suffix"] not in suffixes:
            continue
        held = best.get(one["suffix"])
        if held is None or size(one) < size(held):
            best[one["suffix"]] = one
    return [best[key] for key in sorted(best)]


def wanted() -> set[str]:
    """Suffixes `tests/needs.py` cannot satisfy from the corpus as it is."""
    from tests import needs

    ledger = needs.LEDGER
    if not ledger.is_file():
        raise RuntimeError(f"run `just corpus needs` first: {ledger} does not exist")
    held = json.loads(ledger.read_text(encoding="utf-8"))
    # BLOCKED_EXTERNALLY is retried on purpose: the block register in
    # `tests/needs.py` survives only while the world still offers no specimen,
    # and this fetch is what challenges that on every run.
    return {
        one["need"].removeprefix("suffix:")
        for one in held["needs"]
        if one["need"].startswith("suffix:") and one["state"] in ("UNSATISFIED", "PARTIAL", "BLOCKED_EXTERNALLY")
    }


def fetch(only: set[str] | None = None) -> dict:
    rows = index()
    chosen = pick(rows, only if only is not None else wanted())
    IMAGES.mkdir(parents=True, exist_ok=True)
    kept, trouble = [], []
    for one in chosen:
        target = IMAGES / filename(one["name"])
        try:
            if not target.is_file():
                raw = _fetched(one["url"], 600).content
                got = hashlib.sha256(raw).hexdigest()
                if got != one["sha256"]:
                    raise RuntimeError(f"sha256 {got} but the index said {one['sha256']}")
                target.write_bytes(raw)
            kept.append({**one, "path": f"{IMAGES.name}/{target.name}", "bytes": target.stat().st_size})
        except TROUBLE as why:
            trouble.append({"name": one["name"], "why": str(why)[:160]})
    held = {
        "what": "Real RAW files, one per suffix, from the darktable/RawTherapee sample archive.",
        "source": {"index": INDEX, "licence": "CC0 only (the index's `co` mark)"},
        "files": kept,
        "trouble": trouble,
    }
    LOCKFILE.write_text(json.dumps(held, indent=2) + "\n", encoding="utf-8")
    return held


if __name__ == "__main__":
    got = fetch()
    for row in got["files"]:
        print(f"  {row['suffix']:7s} {row['bytes'] / 1e6:7.1f} MB  {row['maker']} {row['model']}")
    print(f"  {len(got['files'])} files, {sum(r['bytes'] for r in got['files']) / 1e9:.2f} GB into {IMAGES}")
    if got["trouble"]:
        print(f"  trouble: {len(got['trouble'])}, see {LOCKFILE}")
