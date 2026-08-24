"""Where thumbnail time actually goes, phase by phase, per media kind.

Run this before changing the pipeline and again after. It measures the
production path -- `vision/decode.open_still`, `db/oriented.upright`,
`vision/thumbs` -- split into the steps that can be optimised separately,
so a change can be judged by which phase it moved.

The phases are what the current code does, in order:

    decode          open the file and get real pixels
    orient          apply the EXIF tag, plus the full-size copy the
                    current path makes when the tag is 1
    resize_preview  original -> 1440
    encode_preview  that preview -> WebP bytes
    resize_thumb    original -> 512, again from the original today
    encode_thumb    that thumb -> WebP bytes
    write           both files to disk

Reported per (kind, extension) as p50 and p95 milliseconds. One average
over a library holding 0.3 MP JPEGs and 22 MP raws describes nothing in
it.

Sources come from a run's own database, so the corpus is real files at
real sizes rather than fixtures chosen for convenience.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import pathlib
import statistics
import sys
import tempfile
import time

# The repo root on sys.path, so the script runs from any cwd without
# installation -- the same shape face_pipeline_validation.py uses (:31-34).
REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PHASES = ("decode", "orient", "resize_preview", "encode_preview", "resize_thumb", "encode_thumb", "write")


class Clock:
    """Elapsed milliseconds per named phase, for one file."""

    def __init__(self) -> None:
        self.spans: dict[str, float] = {}

    def time(self, name: str) -> _Span:
        return _Span(self, name)


class _Span:
    def __init__(self, clock: Clock, name: str) -> None:
        self.clock, self.name = clock, name
        self.started = 0.0

    def __enter__(self) -> _Span:
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        # None, not False: a bool return type tells a type checker this
        # might swallow the exception, which makes everything after a
        # `with` block "possibly unbound".
        self.clock.spans[self.name] = (time.perf_counter() - self.started) * 1000.0


def corpus(db: pathlib.Path, per_group: int) -> list[tuple[int, str, pathlib.Path]]:
    """Up to `per_group` present files of each (kind, extension).

    Largest first within a group: the expensive members are the ones a
    change has to move, and a random sample is mostly whatever the
    library holds most of.
    """
    from db import connect, detect

    grouped: dict[tuple[str, str], list[tuple[int, str, pathlib.Path]]] = collections.defaultdict(list)
    with connect.connect(db, read_only=True) as conn:
        rows = conn.execute("SELECT id, kind, name FROM file WHERE missing_since IS NULL ORDER BY size DESC").fetchall()
        for file_id, kind, name in rows:
            suffix = pathlib.Path(name).suffix.lower().lstrip(".") or "none"
            group = (kind, suffix)
            if len(grouped[group]) >= per_group:
                continue
            # The runner's own resolver: folders are a parent_id tree, and a
            # second way of walking it here would be a second thing to be wrong.
            path = pathlib.Path(detect.path_of(conn, file_id))
            if path.is_file():
                grouped[group].append((file_id, kind, path))
    return [item for members in grouped.values() for item in members]


def measure(path: pathlib.Path, kind: str, orientation: int | None, staging: pathlib.Path) -> dict:
    """One file through the current pipeline, phase by phase.

    Deliberately not calling `oriented.for_model`: that needs a database
    connection per file and hides the copy inside it. These are the same
    steps, spelled out so each one can be timed.
    """
    from PIL import ImageOps

    from db import oriented
    from vision import decode, thumbs

    preview_edge, thumb_edge = thumbs.EDGES["preview"], thumbs.EDGES["thumb"]
    clock = Clock()
    try:
        with clock.time("decode"):
            if kind == "video":
                poster = decode.poster(path)
                if poster is None:
                    return {"path": str(path), "error": "no poster frame"}
                opened = poster
            else:
                opened = decode.open_still(path)
                opened.load()
            source = opened.size

        with clock.time("orient"):
            # Exactly what open_upright does, including the copy it makes
            # when the tag is 1 and nothing needed doing.
            turned = oriented.upright(opened, orientation)
            frame = turned if turned is not opened else opened.copy()

        with clock.time("resize_preview"):
            preview = ImageOps.contain(frame, (preview_edge, preview_edge))
        with clock.time("encode_preview"):
            preview_bytes = io.BytesIO()
            preview.save(preview_bytes, format="WEBP", quality=thumbs.QUALITY)

        with clock.time("resize_thumb"):
            # From the original, which is what put_all does today.
            thumb = ImageOps.contain(frame, (thumb_edge, thumb_edge))
        with clock.time("encode_thumb"):
            thumb_bytes = io.BytesIO()
            thumb.save(thumb_bytes, format="WEBP", quality=thumbs.QUALITY)

        with clock.time("write"):
            for name, buffer in (("p.webp", preview_bytes), ("t.webp", thumb_bytes)):
                staging_file = staging / f"{name}.tmp"
                staging_file.write_bytes(buffer.getvalue())
                os.replace(staging_file, staging / name)
    except (OSError, ValueError, MemoryError, RuntimeError) as why:
        # A real corpus has unreadable and truncated members. One bad file
        # is a row in the report, not the end of the run.
        return {"path": str(path), "error": f"{type(why).__name__}: {why}"}

    return {
        "path": str(path),
        "kind": kind,
        "suffix": path.suffix.lower().lstrip("."),
        "megapixels": round(source[0] * source[1] / 1e6, 2),
        "source_bytes": path.stat().st_size,
        "preview_bytes": len(preview_bytes.getvalue()),
        "thumb_bytes": len(thumb_bytes.getvalue()),
        "total": round(sum(clock.spans.values()), 2),
        **{phase: round(clock.spans.get(phase, 0.0), 2) for phase in PHASES},
    }


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) < 2:
        return ordered[0]
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


def report(measured: list[dict]) -> dict:
    good = [row for row in measured if "error" not in row]
    bad = [row for row in measured if "error" in row]
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for row in good:
        groups[f"{row['kind']}/{row['suffix']}"].append(row)

    print(f"\n\n{len(good)} files measured, {len(bad)} failed\n")
    head = (
        f"{'group':17} {'n':>3} {'MP':>6} {'decode':>8} {'orient':>7} {'rszP':>6} "
        f"{'encP':>6} {'rszT':>6} {'encT':>6} {'write':>6} {'p50':>8} {'p95':>8} {'/sec':>6}"
    )
    print(head)
    print("-" * len(head))

    summary = {}
    ordered = sorted(groups.items(), key=lambda kv: -statistics.median([r["total"] for r in kv[1]]))
    for name, rows in ordered:
        totals = [r["total"] for r in rows]
        med = {phase: statistics.median([r[phase] for r in rows]) for phase in PHASES}
        mp = statistics.median([r["megapixels"] for r in rows])
        middle = statistics.median(totals)
        print(
            f"{name:17} {len(rows):3} {mp:6.1f} {med['decode']:8.1f} {med['orient']:7.1f} "
            f"{med['resize_preview']:6.1f} {med['encode_preview']:6.1f} {med['resize_thumb']:6.1f} "
            f"{med['encode_thumb']:6.1f} {med['write']:6.1f} {middle:8.1f} {p95(totals):8.1f} "
            f"{1000.0 / middle if middle else 0:6.1f}"
        )
        summary[name] = {
            "n": len(rows),
            "median_megapixels": round(mp, 2),
            "phases_p50_ms": {phase: round(med[phase], 2) for phase in PHASES},
            "total_p50_ms": round(middle, 2),
            "total_p95_ms": round(p95(totals), 2),
            "files_per_second_p50": round(1000.0 / middle, 2) if middle else None,
        }

    if good:
        whole = [r["total"] for r in good]
        middle = statistics.median(whole)
        print(f"\nwhole corpus: p50 {middle:.1f} ms   p95 {p95(whole):.1f} ms")
        print(f"serial throughput at p50: {1000.0 / middle:.2f} files/sec")
    for row in bad:
        print(f"  FAILED {row['path']}: {row['error']}")
    return {"groups": summary, "failures": bad}


def main() -> None:
    parser = argparse.ArgumentParser(description="phase timings for the thumbnail pipeline")
    parser.add_argument("--db", default=str(pathlib.Path.home() / ".smartgallery" / "gallery.db"))
    parser.add_argument("--per-group", type=int, default=25, help="files per (kind, extension)")
    parser.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "thumbnail_phases.json"))
    asked = parser.parse_args()

    db = pathlib.Path(asked.db)
    if not db.is_file():
        raise SystemExit(f"no database at {db}")

    from db import connect, oriented

    files = corpus(db, asked.per_group)
    print(f"corpus: {len(files)} files from {db}")

    measured = []
    with tempfile.TemporaryDirectory(prefix="thumbnail_phases-") as scratch:
        staging = pathlib.Path(scratch)
        with connect.connect(db, read_only=True) as conn:
            for index, (file_id, kind, path) in enumerate(files, 1):
                measured.append(measure(path, kind, oriented.orientation_of(conn, file_id), staging))
                print(f"\r  {index}/{len(files)}  {path.name[:50]:50}", end="", flush=True)

    written = report(measured)
    out = pathlib.Path(asked.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"per_file": measured, **written}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
