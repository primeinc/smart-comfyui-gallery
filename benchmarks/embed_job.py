"""What the embed job actually achieves, end to end.

`openclip_batch.py` measures the encoder with the pictures already
decoded and no database in sight. That number is a ceiling and was never
job throughput; this is the job.

Everything the runner really does is in here: claiming the job, resolving
each file's path, decoding and orienting it, preprocessing, the transfer,
inference, writing the vector, updating similarity, and committing each
item's completion to the ledger. The gap between this and the encoder
ceiling is what the application costs around the model, and it decides
whether faster encoding is worth anything.

Per-item timings come from the runner's own event stream rather than from
instrumentation added here: `item.started` and `item.finished` are
already committed with timestamps, so subtracting them measures the item
as the console sees it.

The database is a temporary one. The library's roots are scanned
read-only into it, so a benchmark can never write to a real run's
gallery.db, and every job starts from an empty cache instead of skipping
work already done.
"""

from __future__ import annotations

import argparse
import json
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


def staged(home: pathlib.Path, roots: list[pathlib.Path], limit: int):
    """A fresh database with those roots scanned into it.

    Returns the connection and the files it found, newest last. Scanning
    is the production walk, so what lands here is what a real library
    would hold.
    """
    from db import connect, library, scan

    conn = connect.connect(home / "gallery.db")
    conn.executescript(connect.schema_sql())
    conn.execute("PRAGMA foreign_keys=ON")
    for root in roots:
        root_id = library.add_root(conn, str(root), "library", 0.0)
        scan.scan(conn, root_id, str(root), 0.0)
    conn.commit()

    # Every kind submit_embed will queue, not just stills: it takes
    # image, animated_image AND video (db/runner.py submit_embed), so
    # counting only stills here reported 100 staged against a job of 105
    # and made the two numbers look unrelated.
    kept = [
        row[0]
        for row in conn.execute(
            "SELECT id FROM file WHERE missing_since IS NULL "
            "AND kind IN ('image', 'animated_image', 'video') ORDER BY id"
        )
    ]
    if limit and len(kept) > limit:
        # Trimmed by deleting the excess rather than by handing a short
        # list to submit_embed, which takes no list: the job must decide
        # its own items the way the application's would.
        conn.executemany("DELETE FROM file WHERE id = ?", [(file_id,) for file_id in kept[limit:]])
        conn.commit()
        kept = kept[:limit]
    return conn, kept


def run(conn, models_dir: str, owner: str) -> dict:
    """One embed job, timed, with the runner's own per-item stamps."""
    from db import runner

    made = runner.submit_embed(conn, 0.0, models_dir=models_dir)
    if not made:
        raise SystemExit("nothing to embed; the cache already covers this corpus")
    started: dict[int, float] = {}
    latencies: list[float] = []
    phases: list[tuple[float, str]] = []

    def heard(event) -> None:
        # The runner's own vocabulary: an item settles as `item.done` or
        # `item.failed`, and its phases arrive as `item.observed`
        # (db/runner.py:1186, :1206, :1242). Guessing at "item.finished"
        # collected nothing at all and reported no latencies rather than
        # an error, which is the quiet kind of wrong.
        kind = event.get("type")
        item = event.get("item_id")
        when = time.perf_counter()
        if kind == "item.started" and item is not None:
            started[item] = when
        elif kind in ("item.done", "item.failed") and item is not None and item in started:
            latencies.append((when - started.pop(item)) * 1000)
        elif kind == "item.observed":
            phases.append((when, str(event.get("phase"))))

    opened = time.perf_counter()
    summary = runner.run_next(conn, owner, 1.0, clock=time.time, on_event=heard)
    whole = time.perf_counter() - opened
    return {
        "job": summary,
        "wall_ms": whole * 1000,
        "latencies_ms": latencies,
        "phases": phases,
    }


def ceiling(conn, files: list[int], models_dir: str, batch: int) -> dict:
    """The same pictures through decode and the encoder alone.

    Not the job: no database writes, no ledger, no per-item commit. The
    difference between this and the job is what the application costs
    around the model, which is the whole reason to measure both.
    """
    import torch

    from db import detect, oriented
    from vision import decode
    from vision.semantic import openclip

    backend = openclip.encoder(models_dir)
    opened = time.perf_counter()
    pictures = []
    for file_id in files:
        # The same representative frame `_embed_item` takes, video
        # included: a ceiling computed over stills alone would be
        # compared against a job that also seeked through clips.
        kind = conn.execute("SELECT kind FROM file WHERE id = ?", (file_id,)).fetchone()[0]
        path = detect.path_of(conn, file_id)
        frame = decode.poster(path) if kind == "video" else oriented.for_model(conn, file_id, path)
        if frame is not None:
            pictures.append(frame.convert("RGB"))
    decode_ms = (time.perf_counter() - opened) * 1000

    opened = time.perf_counter()
    for at in range(0, len(pictures), batch):
        tensor = torch.stack([backend.preprocess(picture) for picture in pictures[at : at + batch]])
        with torch.no_grad():
            backend.model.encode_image(tensor.to(backend.device), normalize=True)
    if backend.device == "cuda":
        torch.cuda.synchronize()
    encode_ms = (time.perf_counter() - opened) * 1000
    return {"decode_ms": decode_ms, "encode_ms": encode_ms, "pictures": len(pictures)}


def main() -> None:
    parser = argparse.ArgumentParser(description="end-to-end throughput of the embed job")
    parser.add_argument("--db", default=str(pathlib.Path.home() / ".smartgallery" / "gallery.db"))
    parser.add_argument("--models-dir", default=str(pathlib.Path.home() / ".smartgallery" / "models"))
    parser.add_argument("--under", default=None, help="only roots whose path contains this")
    parser.add_argument("--count", type=int, default=200, help="pictures in the job")
    parser.add_argument("--batch", type=int, default=64, help="batch for the encoder-ceiling comparison")
    parser.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "embed_job.json"))
    asked = parser.parse_args()

    from db import connect

    source = pathlib.Path(asked.db)
    if not source.is_file():
        raise SystemExit(f"no database at {source}")
    with connect.connect(source, read_only=True) as reading:
        roots = [pathlib.Path(row[0]) for row in reading.execute("SELECT path FROM root ORDER BY id")]
    if asked.under:
        roots = [root for root in roots if asked.under in str(root)]
    roots = [root for root in roots if root.is_dir()]
    if not roots:
        raise SystemExit("no readable roots matched")
    print(f"roots: {', '.join(root.name for root in roots)}")

    with tempfile.TemporaryDirectory(prefix="embed_job-") as scratch:
        home = pathlib.Path(scratch)
        conn, files = staged(home, roots, asked.count)
        print(f"staged {len(files)} pictures into a temporary database")
        if not files:
            raise SystemExit("nothing scanned")

        limit = ceiling(conn, files, asked.models_dir, asked.batch)
        print(
            f"\nencoder ceiling over the same files:"
            f"\n  decode {limit['decode_ms']:.0f} ms   encode at batch {asked.batch} {limit['encode_ms']:.0f} ms"
            f"\n  {len(files) / ((limit['decode_ms'] + limit['encode_ms']) / 1000):.1f} pictures/sec"
        )

        print("\nrunning the job:")
        found = run(conn, asked.models_dir, "bench")
        conn.close()

    summary = found["job"]
    did = summary.get("did", 0) if summary else 0
    per_second = did / (found["wall_ms"] / 1000) if found["wall_ms"] else 0
    print(f"  {summary}")
    print(f"  {found['wall_ms']:.0f} ms for {did} items -> {per_second:.1f} items/sec")
    latencies = sorted(found["latencies_ms"])
    if latencies:
        print(
            f"  per item: p50 {statistics.median(latencies):.1f} ms   "
            f"p95 {latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))]:.1f} ms   "
            f"max {latencies[-1]:.1f} ms"
        )
    inside = limit["decode_ms"] + limit["encode_ms"]
    print(
        f"\n  the job spends {found['wall_ms']:.0f} ms where decode and encode alone are {inside:.0f} ms:"
        f" {found['wall_ms'] - inside:.0f} ms is the application around the model"
        f" ({(found['wall_ms'] - inside) / found['wall_ms'] * 100:.0f}%)"
    )

    out = pathlib.Path(asked.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "pictures": len(files),
                "batch_for_ceiling": asked.batch,
                "job": summary,
                "wall_ms": round(found["wall_ms"], 1),
                "items_per_second": round(per_second, 2),
                "latency_ms": {
                    "p50": round(statistics.median(latencies), 1) if latencies else None,
                    "p95": round(latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))], 1)
                    if latencies
                    else None,
                    "max": round(latencies[-1], 1) if latencies else None,
                },
                "ceiling": {key: round(value, 1) for key, value in limit.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
