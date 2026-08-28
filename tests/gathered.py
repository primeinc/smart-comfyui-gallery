"""Media the corpus takes from this machine and from the reference mirrors.

Three things the network could not supply:

- ComfyUI's own dialect. SwarmUI writes a `parameters` chunk; ComfyUI writes
  `prompt` and `workflow`. Only the second proves the reader that this
  application is named after. `comfyanonymous/ComfyUI_examples` is the author's
  own set and its LICENSE grants use, copy, modify and distribute.
- Generated media with no provenance at all. An image with a stripped chunk is
  a case the gallery meets constantly and no generator writes on purpose.
- Real RAW with a matching JPEG. A CR2 and its out-of-camera JPEG share a
  capture instant and differ in every byte, which is the duplicate case a
  checksum cannot see and a corpus of unrelated files never produces.

SOURCES IS AN ALLOWLIST, DELIBERATELY. The same tree holds face-recognition and
KYC photograph sets of real people. Those are not licensed for redistribution
and must never reach a published corpus, so this module names what it takes
rather than naming what it skips: a set added to that tree later is excluded by
default instead of being swept in.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parent.parent

CORPUS = pathlib.Path(os.environ.get("SG_CORPUS", REPO.parent / "sg-corpus"))
LOCKFILE = REPO / "tests" / "gathered.lock.json"

#: Where the reference mirrors are. Read-only; files are copied out of them.
REFS = pathlib.Path(os.environ.get("SG_REFS", REPO.parent / "refs"))

#: Where this machine keeps generated output.
LOCAL = pathlib.Path(os.environ.get("SG_LOCAL_SAMPLES", "C:/ComfyUI/output/sample-datasets"))

#: What copying one file is allowed to fail with -- a permission, a name the
#: filesystem refuses, a source that moved mid-run. Anything else is a defect
#: in this module rather than a fact about the file, and propagates.
TROUBLE = (OSError, ValueError)


@dataclasses.dataclass(frozen=True)
class Source:
    """One place files come from, and the single reason to take them."""

    into: str
    root: pathlib.Path
    patterns: tuple[str, ...]
    most: int
    origin: str
    license: str
    why: str


SOURCES: tuple[Source, ...] = (
    Source(
        into="comfyui",
        root=REFS / "comfyanonymous" / "ComfyUI_examples",
        patterns=("**/*.png",),
        most=200,
        origin="github.com/comfyanonymous/ComfyUI_examples",
        license="permissive (use/copy/modify/distribute granted; see repo LICENSE)",
        why="ComfyUI's native prompt+workflow chunks, written by ComfyUI itself",
    ),
    Source(
        into="swarm",
        root=LOCAL / "swarm-mixed",
        patterns=("**/*.png", "**/*.webp", "**/*.jpg", "**/*.mp4"),
        most=200,
        origin="local SwarmUI output",
        license="author's own output",
        why="SwarmUI's parameters chunk, and generated video beside generated stills",
    ),
    Source(
        into="swarm-i2i",
        root=LOCAL / "i2i-test-output",
        patterns=("**/*.png",),
        most=60,
        origin="local SwarmUI output",
        license="author's own output",
        why="image-to-image runs: many outputs sharing one source image",
    ),
    Source(
        into="generated-bare",
        root=LOCAL / "chatgpt-bananas",
        patterns=("**/*.png",),
        most=20,
        origin="local ChatGPT image output",
        license="author's own output",
        why="generated media carrying NO chunk and NO EXIF -- measured, not assumed",
    ),
    Source(
        into="raw-canon",
        root=LOCAL / "RAW",
        patterns=("**/*.JPG",),
        most=226,
        origin="local Canon EOS 5D Mark III shoot",
        license="author's own photographs",
        why="a 2013 body's out-of-camera JPEGs: real EXIF, one shoot, real burst timing",
    ),
    Source(
        into="raw-canon",
        root=LOCAL / "RAW",
        patterns=("**/*.CR2",),
        most=100,
        origin="local Canon EOS 5D Mark III shoot",
        license="author's own photographs",
        why="the CR2 beside its JPEG: same capture, different bytes, a real duplicate pair",
    ),
)


def digest(path: pathlib.Path) -> str:
    held = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            held.update(block)
    return held.hexdigest()


def take(source: Source, into: pathlib.Path) -> tuple[list[dict], list[dict]]:
    """Copy a source's files, and say what could not be taken.

    A missing root is reported, never silently skipped: a source that vanished
    would otherwise leave the corpus smaller with nothing to show why.
    """
    if not source.root.is_dir():
        return [], [{"source": source.into, "root": str(source.root), "why": "root is not a directory"}]

    found: list[pathlib.Path] = []
    for pattern in source.patterns:
        found.extend(p for p in source.root.glob(pattern) if p.is_file())
    found = sorted(set(found))[: source.most]

    where = into / source.into
    where.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    trouble: list[dict] = []
    for one in found:
        target = where / one.name
        try:
            if not (target.is_file() and target.stat().st_size == one.stat().st_size):
                shutil.copy2(one, target)
            rows.append(
                {
                    "path": f"{source.into}/{target.name}",
                    "sha256": digest(target),
                    "bytes": target.stat().st_size,
                    "origin": source.origin,
                    "original": str(one.relative_to(source.root)).replace("\\", "/"),
                    "license": source.license,
                    "why": source.why,
                    "kind": "generated" if source.into.startswith(("comfyui", "swarm", "generated")) else "real",
                }
            )
        except TROUBLE as why:
            trouble.append({"file": str(one), "why": str(why)})
    return rows, trouble


def gather(into: pathlib.Path | None = None) -> dict:
    into = into or CORPUS
    rows: list[dict] = []
    trouble: list[dict] = []
    for source in SOURCES:
        got, bad = take(source, into)
        rows.extend(got)
        trouble.extend(bad)
    held = {
        "what": "Media taken from this machine and from the reference mirrors.",
        "excluded": "face-recognition and KYC photograph sets in the same tree; not licensed for redistribution",
        "files": rows,
        "trouble": trouble,
    }
    LOCKFILE.write_text(json.dumps(held, indent=2) + "\n", encoding="utf-8")
    return held


if __name__ == "__main__":
    got = gather()
    by = {}
    for row in got["files"]:
        part = row["path"].split("/")[0]
        held = by.setdefault(part, [0, 0])
        held[0] += 1
        held[1] += row["bytes"]
    for part, (count, size) in sorted(by.items()):
        print(f"  {part:16s} {count:5d} files {size / 1e9:7.3f} GB")
    print(f"  {'TOTAL':16s} {len(got['files']):5d} files {sum(r['bytes'] for r in got['files']) / 1e9:7.3f} GB")
    if got["trouble"]:
        print(f"  trouble: {len(got['trouble'])}, see {LOCKFILE}")
