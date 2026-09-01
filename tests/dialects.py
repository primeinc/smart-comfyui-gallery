"""One file per generator the application claims to read.

Eight adapters are registered in `metaparse/adapters.py`. The corpus reached
two of them from real files -- ComfyUI, from its author's own example
workflows, and SwarmUI, from this machine's output. The other six have no
obtainable specimen: 925 candidate images across four upstream mirrors
(`receyuki/stable-diffusion-prompt-reader`, `invoke-ai/InvokeAI`,
`lllyasviel/Fooocus`, `easydiffusion/easydiffusion`) carry ZERO generator
metadata, being screenshots and documentation art, and 100 Fooocus style
samples from `crystantine/Fooocus-2.3.1` carry none either. Hugging Face has
no dataset shipping loose images with the chunk intact; the image datasets
there are parquet, which strips the container the dialect lives in.

So these are WRITTEN, and labelled written. They are ports of the real
writers, not payloads recalled from memory: `tests/test_metadata_lands_in_
the_schema.py WRITERS` is the same table the writer tests are held to, and
its stealth encoder cites Forge's `modules/stealth_infotext.py` line for
line. Reusing that table rather than inventing a second one is the point --
a corpus file and the test that proves the reader agree by construction.

What they are NOT: evidence about how a real installation of any of these
tools writes today. A spec-derived file proves the reader parses the format
as this repository understands it. It cannot falsify that understanding,
which is exactly what a real file from a real installation would do. When one
becomes obtainable it replaces the written one; `docs/CORPUS_SOURCES.md`
records the gap rather than closing it.
"""

from __future__ import annotations

import json
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

CORPUS = pathlib.Path(os.environ.get("SG_CORPUS", REPO.parent / "sg-corpus"))
IMAGES = CORPUS / "dialects"
LOCKFILE = REPO / "tests" / "dialects.lock.json"

#: Which of these a REAL file already covers, and where that file came from.
#: Written here anyway, because the written one is the deterministic case and
#: the real one is the messy case, and a reader should meet both.
REAL_ALREADY = {
    "comfyui": "comfyui/ -- comfyanonymous/ComfyUI_examples@f9431bb000ce",
    "swarmui": "swarm/, swarm-i2i/ -- this machine's SwarmUI output",
}


def write(into: pathlib.Path | None = None) -> dict:
    """Write one file per registered writer. Returns what was written."""
    import hashlib

    from tests.test_metadata_lands_in_the_schema import WRITERS
    from tests.test_metaparse import DRAWTHINGS_XMP, make_png

    # Draw Things is registered in `metaparse/adapters.py` and absent from
    # WRITERS, so `tests/needs.py` reported one dialect with no file anywhere.
    # Its fixture does exist, one module over, as an XMP payload in an iTXt chunk.
    writers = dict(WRITERS)
    writers["drawthings"] = (
        lambda d: make_png(d / "drawthings.png", {"XML:com.adobe.xmp": DRAWTHINGS_XMP}, itxt=("XML:com.adobe.xmp",)),
        {},
    )

    where = into or IMAGES
    where.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in sorted(writers):
        build, _options = writers[name]
        made = pathlib.Path(build(where))
        raw = made.read_bytes()
        rows.append(
            {
                "path": f"{where.name}/{made.name}",
                "writer": name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "kind": "generated",
                "origin": "written from tests/test_metadata_lands_in_the_schema.py WRITERS",
                "derived_from": "the repository's port of that writer's own format",
                "real_file_also_covers": REAL_ALREADY.get(name, ""),
            }
        )
    held = {
        "what": "One file per generator adapter, written from the repository's writer ports.",
        "why": "Six of eight dialects have no obtainable real specimen; see the module docstring.",
        "not_evidence_of": "how a real installation of these tools writes today",
        "files": rows,
    }
    # newline="" or Windows writes CRLF into a file the repo stores as LF,
    # dirtying a tracked lockfile with zero content delta and reddening the
    # commit gate for whoever is holding a candidate.
    with LOCKFILE.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(held, indent=2) + "\n")
    return held


if __name__ == "__main__":
    got = write()
    for row in got["files"]:
        also = f"  (real: {row['real_file_also_covers']})" if row["real_file_also_covers"] else ""
        print(f"  {row['writer']:16s} {row['bytes']:7d}B  {row['path']}{also}")
    print(f"  {len(got['files'])} written into {IMAGES}")
