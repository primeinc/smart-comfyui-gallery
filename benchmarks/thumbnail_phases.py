"""Where thumbnail time actually goes, phase by phase, per media kind.

Run this before changing the pipeline and again after. It measures the
production path -- `vision/decode.open_still`, `db/oriented.upright`,
`vision/thumbs` -- split into the steps that can be optimised separately,
so a change can be judged by which phase it moved.

Two lanes over the same files, so a change is judged against the shape
it replaced rather than against a memory:

    before    the pipeline as it stood: full decode, a full-resolution
              copy, and both derivatives resized from the original
    shipped   the production functions themselves -- decode.open_bounded,
              oriented.upright, thumbs.fit -- never a copy of them, so
              this cannot report an improvement the application lacks

Seven phases each: decode, orient, resize_preview, encode_preview,
resize_thumb, encode_thumb, write.

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


def measure_shipped(path: pathlib.Path, kind: str, orientation: int | None, staging: pathlib.Path, method: int) -> dict:
    """The same seven phases through the code that actually ships.

    It makes the SAME routing decision production makes -- libvips first
    through `derive.opened`, Pillow when that returns None -- because a
    lane that always takes one branch reports a pipeline nobody runs. The
    `decoder` field says which one responded, so a row can be read without
    guessing.

    Everything here is a production function. A benchmark that
    reimplements what it measures can report an improvement the
    application does not have; this one measured 3.3x that way once, and
    2.8x once it called the real thing.

    `method` overrides thumbs.METHOD so the encoder dial can be swept
    without editing production; the default is what runs.
    """
    from db import oriented
    from vision import decode, derive, thumbs

    preview_edge, thumb_edge = thumbs.EDGES["preview"], thumbs.EDGES["thumb"]
    clock = Clock()
    decoder = "libvips"
    try:
        with clock.time("decode"):
            frame = None if kind == "video" else derive.opened(path, preview_edge, orientation)
            if frame is None:
                decoder = "pyav" if kind == "video" else "pillow"
                picture = (
                    decode.poster(path)
                    if kind == "video"
                    else oriented.for_derivatives(path, preview_edge, orientation)
                )
                if picture is None:
                    return {"path": str(path), "error": "no decodable frame"}
                frame = picture
            source = frame.size if decoder != "libvips" else (frame.width, frame.height)

        with clock.time("orient"):
            pass  # applied inside the decode step by both routes

        if decoder == "libvips":
            with clock.time("resize_preview"):
                preview = derive.fit(frame, preview_edge)
            with clock.time("encode_preview"):
                preview_bytes = preview.webpsave_buffer(Q=thumbs.QUALITY, effort=method)
            with clock.time("resize_thumb"):
                thumb = derive.fit(preview, thumb_edge)
            with clock.time("encode_thumb"):
                thumb_bytes = thumb.webpsave_buffer(Q=thumbs.QUALITY, effort=method)
            preview_size, thumb_size = [preview.width, preview.height], [thumb.width, thumb.height]
        else:
            with clock.time("resize_preview"):
                preview = thumbs.fit(frame, preview_edge)
            with clock.time("encode_preview"):
                held = io.BytesIO()
                preview.save(held, format="WEBP", quality=thumbs.QUALITY, method=method)
                preview_bytes = held.getvalue()
            with clock.time("resize_thumb"):
                thumb = thumbs.fit(preview, thumb_edge)
            with clock.time("encode_thumb"):
                held = io.BytesIO()
                thumb.save(held, format="WEBP", quality=thumbs.QUALITY, method=method)
                thumb_bytes = held.getvalue()
            preview_size, thumb_size = list(preview.size), list(thumb.size)

        with clock.time("write"):
            for name, blob in (("p.webp", preview_bytes), ("t.webp", thumb_bytes)):
                staging_file = staging / f"{name}.tmp"
                staging_file.write_bytes(blob)
                os.replace(staging_file, staging / name)
    except (OSError, ValueError, MemoryError, RuntimeError) as why:
        return {"path": str(path), "error": f"{type(why).__name__}: {why}"}

    return {
        "path": str(path),
        "kind": kind,
        "suffix": path.suffix.lower().lstrip("."),
        "decoder": decoder,
        "megapixels": round(source[0] * source[1] / 1e6, 2),
        "source_bytes": path.stat().st_size,
        "preview_bytes": len(preview_bytes),
        "thumb_bytes": len(thumb_bytes),
        "preview_size": preview_size,
        "thumb_size": thumb_size,
        "total": round(sum(clock.spans.values()), 2),
        **{phase: round(clock.spans.get(phase, 0.0), 2) for phase in PHASES},
    }


def measure(path: pathlib.Path, kind: str, orientation: int | None, staging: pathlib.Path) -> dict:
    """One file through the pipeline AS IT WAS, phase by phase.

    Kept so the shipped lane has something to be measured against. This
    is deliberately a frozen copy of the old shape rather than a call
    into production -- production no longer does any of it, and a
    baseline that moves is not a baseline.
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
        f"{'encP':>6} {'rszT':>6} {'encT':>6} {'write':>6} {'p50':>8} {'p95':>8} {'/sec':>6} {'cacheKB':>8}"
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
        # What the two derivatives cost on disk. Encoder speed is bought
        # with bytes, so the dial cannot be judged on time alone.
        cache = statistics.median([r["preview_bytes"] + r["thumb_bytes"] for r in rows]) / 1024.0
        print(
            f"{name:17} {len(rows):3} {mp:6.1f} {med['decode']:8.1f} {med['orient']:7.1f} "
            f"{med['resize_preview']:6.1f} {med['encode_preview']:6.1f} {med['resize_thumb']:6.1f} "
            f"{med['encode_thumb']:6.1f} {med['write']:6.1f} {middle:8.1f} {p95(totals):8.1f} "
            f"{1000.0 / middle if middle else 0:6.1f} {cache:8.1f}"
        )
        summary[name] = {
            "n": len(rows),
            "median_megapixels": round(mp, 2),
            "phases_p50_ms": {phase: round(med[phase], 2) for phase in PHASES},
            "total_p50_ms": round(middle, 2),
            "total_p95_ms": round(p95(totals), 2),
            "files_per_second_p50": round(1000.0 / middle, 2) if middle else None,
            "cache_kib_p50": round(cache, 1),
        }

    if good:
        whole = [r["total"] for r in good]
        middle = statistics.median(whole)
        bytes_each = statistics.median([r["preview_bytes"] + r["thumb_bytes"] for r in good]) / 1024.0
        print(f"\nwhole corpus: p50 {middle:.1f} ms   p95 {p95(whole):.1f} ms   cache {bytes_each:.1f} KiB/file")
        print(f"serial throughput at p50: {1000.0 / middle:.2f} files/sec")
    for row in bad:
        print(f"  FAILED {row['path']}: {row['error']}")
    return {"groups": summary, "failures": bad}


def main() -> None:
    parser = argparse.ArgumentParser(description="phase timings for the thumbnail pipeline")
    parser.add_argument("--db", default=str(pathlib.Path.home() / ".smartgallery" / "gallery.db"))
    parser.add_argument("--per-group", type=int, default=25, help="files per (kind, extension)")
    parser.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "thumbnail_phases.json"))
    parser.add_argument(
        "--webp-method",
        type=int,
        default=None,
        help="override vision.thumbs.METHOD, libwebp's 0-fastest to 6-smallest dial",
    )
    asked = parser.parse_args()

    db = pathlib.Path(asked.db)
    if not db.is_file():
        raise SystemExit(f"no database at {db}")

    from db import connect, oriented

    files = corpus(db, asked.per_group)
    print(f"corpus: {len(files)} files from {db}")

    from vision import thumbs

    method = thumbs.METHOD if asked.webp_method is None else asked.webp_method
    print(f"webp method: {method}" + ("" if asked.webp_method is None else f" (thumbs.METHOD is {thumbs.METHOD})"))

    runs: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory(prefix="thumbnail_phases-") as scratch:
        staging = pathlib.Path(scratch)
        with connect.connect(db, read_only=True) as conn:
            tags = [(file_id, kind, path, oriented.orientation_of(conn, file_id)) for file_id, kind, path in files]
        for name, run in (
            ("before", lambda path, kind, tag: measure(path, kind, tag, staging)),
            ("shipped", lambda path, kind, tag: measure_shipped(path, kind, tag, staging, method)),
        ):
            print(f"\n{name}:")
            measured = []
            for index, (_file_id, kind, path, tag) in enumerate(tags, 1):
                measured.append(run(path, kind, tag))
                print(f"\r  {index}/{len(tags)}  {path.name[:50]:50}", end="", flush=True)
            runs[name] = measured

    written = {}
    for name, measured in runs.items():
        print(f"\n\n===== {name} =====")
        written[name] = report(measured)

    before = [r["total"] for r in runs["before"] if "error" not in r]
    after = [r["total"] for r in runs["shipped"] if "error" not in r]
    if before and after:
        was, now = statistics.median(before), statistics.median(after)
        print(f"\np50 {was:.1f} ms -> {now:.1f} ms   ({was / now:.1f}x)")
        print(f"serial throughput {1000.0 / was:.2f} -> {1000.0 / now:.2f} files/sec")

    out = pathlib.Path(asked.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"webp_method": method, "per_file": runs, "summary": written}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
