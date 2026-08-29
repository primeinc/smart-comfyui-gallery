"""Photographs whose only date is the folder they sit in.

The corpus was all camera output and generator output, and both carry their
own timestamps. A real library does not: a box of scanned prints becomes
`1998/`, an export becomes `2003-07/`, and a shoebox becomes `1970s/`. Those
files have no EXIF, no stamped name, and nothing but a directory to say when
they happened.

That shape is why `db/when.py folder_when` and the coarse half of
`time_precision` exist. It is also the shape that proved they were missing:
before this, every file below fell through to `mtime` and was dated by when
it was last copied.

The pictures are painted, and say so. What is under test is the DATING, not
the pixels -- a real photograph would prove nothing more here, and the six
pre-1990 Commons photographs already carry the real-EXIF half of that case.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

CORPUS = pathlib.Path(os.environ.get("SG_CORPUS", REPO.parent / "sg-corpus"))
IMAGES = CORPUS / "scanned"
LOCKFILE = REPO / "tests" / "scanned.lock.json"

#: `(folder chain, file name, the rung it should land on)`. The expectation is
#: stated so the lockfile records what each file was written to prove, and a
#: disagreement with what `tests/needs.py` measured is the finding.
SHAPES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("1964",), "seaside with nan.jpg", "year"),
    (("1978",), "the old ford.jpg", "year"),
    (("1998",), "scan0042.jpg", "year"),
    (("2003-07",), "holiday 12.jpg", "month"),
    (("2011_03",), "kitchen before.jpg", "month"),
    (("1970s",), "grandad in the garden.jpg", "decade"),
    (("1980's",), "school photo.jpg", "decade"),
    (("2013", "02"), "chain month.jpg", "month"),
    (("2013", "02", "10"), "chain day.jpg", "day"),
    (("2016-08-21",), "dated folder.jpg", "day"),
)


#: A photograph that was ALSO generated from. `db/schema.sql` calls
#: `has_generation = 1 AND has_capture = 1` `mixed`; camera output beside
#: generator output is never both, and no file had ever landed there.
MIXED_NAME = "img2img over a photograph.png"
MIXED_CAMERA = ("Canon", "Canon EOS 5D Mark III")
MIXED_TAKEN = "2013:02:10 14:23:01"
MIXED_ISO = 400
MIXED_PARAMETERS = (
    "a castle at dawn, oil painting\n"
    "Negative prompt: blurry, watermark\n"
    "Steps: 24, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345, "
    "Size: 512x512, Model: dreamshaper_v8, Denoising strength: 0.45"
)


def _mixed(path: pathlib.Path) -> None:
    """One file carrying a camera's EXIF and a generator's recipe."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    exif = Image.Exif()
    exif[271], exif[272] = MIXED_CAMERA
    exif[36867] = MIXED_TAKEN
    exif[34855] = MIXED_ISO
    chunks = PngInfo()
    chunks.add_text("parameters", MIXED_PARAMETERS)
    Image.new("RGB", (80, 60), (120, 80, 40)).save(path, pnginfo=chunks, exif=exif.tobytes())


def write(into: pathlib.Path | None = None) -> dict:
    from PIL import Image

    where = into or IMAGES
    rows = []
    for index, (folders, name, expect) in enumerate(SHAPES):
        target = where.joinpath(*folders)
        target.mkdir(parents=True, exist_ok=True)
        path = target / name
        # A different shade per file, so no two are duplicates of each other
        # and the duplicate surfaces are not drowned by this set.
        picture = Image.new("RGB", (80, 60), (40 + index * 20, 90, 200 - index * 15))
        picture.save(path, format="JPEG")
        raw = path.read_bytes()
        rows.append(
            {
                "path": "/".join((where.name, *folders, name)),
                "folders": list(folders),
                "expects": expect,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "why": f"no EXIF, no stamped name: the folder is the only claim, at {expect} precision",
            }
        )
    where.mkdir(parents=True, exist_ok=True)
    mixed = where / MIXED_NAME
    _mixed(mixed)
    raw = mixed.read_bytes()
    rows.append(
        {
            "path": f"{where.name}/{MIXED_NAME}",
            "folders": [],
            "expects": "origin=mixed",
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "why": "camera EXIF AND a generator recipe: the only shape the schema calls `mixed`",
        }
    )

    held = {
        "what": "Undated photographs in dated folders: the coarse rungs of db/when.py.",
        "not_evidence_of": "anything about pixels; these are painted",
        "files": rows,
    }
    LOCKFILE.write_text(json.dumps(held, indent=2) + "\n", encoding="utf-8")
    return held


def judged() -> list[tuple[str, str, str]]:
    """`(path, expected rung, what db/when.py actually says)`.

    Read straight from the judge rather than from a scan, so this answers
    even before the library has been re-ingested.
    """
    from db import when

    out = []
    for folders, name, expect in SHAPES:
        verdict = when.judge_file(name=name, folders=[IMAGES.name, *folders], mtime=1_700_000_000.0, btime=None)
        got = verdict.precision if verdict is not None else "none"
        out.append(("/".join((*folders, name)), expect, got))
    return out


if __name__ == "__main__":
    got = write()
    print(f"  {len(got['files'])} files into {IMAGES}")
    wrong = 0
    for path, expect, actual in judged():
        mark = "ok " if expect == actual else "BAD"
        wrong += expect != actual
        print(f"  {mark} {path:36s} expected {expect:8s} judged {actual}")
    print(f"  {wrong} disagreements")
