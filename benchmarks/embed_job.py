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
instrumentation added here: `item.started` and `item.done` are already
committed with timestamps, so subtracting them measures the item as the
console sees it.

Resources are sampled while the job runs. `time.process_time()` over wall
time gives the mean cores this process kept busy, which is the contention
question and costs nothing; GPU utilisation comes from nvidia-smi,
because pynvml is not installed and is not worth a dependency here.

The database is a temporary one. The library's roots are scanned
read-only into it, so a benchmark can never write to a real run's
gallery.db, and every job starts from an empty cache instead of skipping
work already done.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile
import threading
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


class Watching:
    """GPU utilisation and memory, sampled while something else runs.

    Through `nvidia-smi` rather than a library: pynvml and psutil are not
    installed and neither is worth a dependency for a benchmark. CPU is
    not sampled at all -- `time.process_time()` over wall time already
    says how many cores this process kept busy, which is the contention
    question, and it needs nothing.
    """

    def __init__(self, every: float = 0.25) -> None:
        self.every = every
        self.utilisation: list[int] = []
        self.memory_mb: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Watching:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _sample(self) -> None:
        query = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
        while not self._stop.is_set():
            try:
                found = subprocess.run(query, capture_output=True, text=True, timeout=5, check=False)
            except (OSError, subprocess.SubprocessError):
                return
            line = found.stdout.strip().splitlines()
            if line:
                parts = [part.strip() for part in line[0].split(",")]
                if len(parts) == 2 and all(part.isdigit() for part in parts):
                    self.utilisation.append(int(parts[0]))
                    self.memory_mb.append(int(parts[1]))
            self._stop.wait(self.every)

    def summary(self) -> dict:
        if not self.utilisation:
            return {"samples": 0}
        return {
            "samples": len(self.utilisation),
            "gpu_percent_mean": round(statistics.mean(self.utilisation), 1),
            "gpu_percent_max": max(self.utilisation),
            "gpu_memory_mb_max": max(self.memory_mb),
        }


def run(conn, models_dir: str, owner: str) -> dict:
    """One embed job, timed, with the runner's own per-item stamps."""
    from db import runner

    made = runner.submit_embed(conn, 0.0, models_dir=models_dir)
    if not made:
        raise SystemExit("nothing to embed; the cache already covers this corpus")
    started: dict[int, float] = {}
    latencies: list[float] = []
    settled: list[tuple[int, float]] = []
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
            took = (when - started.pop(item)) * 1000
            latencies.append(took)
            settled.append((item, took))
        elif kind == "item.observed":
            phases.append((when, str(event.get("phase"))))

    kinds = dict(conn.execute("SELECT id, kind FROM file").fetchall())
    per_kind: dict[str, list[float]] = {}

    opened = time.perf_counter()
    cpu_opened = time.process_time()
    with Watching() as watching:
        summary = runner.run_next(conn, owner, 1.0, clock=time.time, on_event=heard)
        whole = time.perf_counter() - opened
        cpu = time.process_time() - cpu_opened
    for item, took in settled:
        per_kind.setdefault(kinds.get(item, "?"), []).append(took)
    return {
        "job": summary,
        "wall_ms": whole * 1000,
        "cpu_seconds": cpu,
        "cores_busy": cpu / whole if whole else 0.0,
        "latencies_ms": latencies,
        "by_kind_ms": per_kind,
        "gpu": watching.summary(),
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
    by_kind = found["by_kind_ms"]
    if len(by_kind) > 1:
        print("  by media kind:")
        for kind, took in sorted(by_kind.items(), key=lambda pair: -statistics.median(pair[1])):
            print(f"    {kind:16} n={len(took):4}  p50 {statistics.median(took):7.1f} ms  max {max(took):7.1f} ms")
    print(
        f"  cores busy: {found['cores_busy']:.2f} of {os.cpu_count()}"
        f"  (CPU {found['cpu_seconds']:.1f} s over {found['wall_ms'] / 1000:.1f} s wall)"
    )
    gpu = found["gpu"]
    if gpu.get("samples"):
        print(
            f"  gpu: {gpu['gpu_percent_mean']:.0f}% mean, {gpu['gpu_percent_max']}% peak, "
            f"{gpu['gpu_memory_mb_max']} MB peak ({gpu['samples']} samples)"
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
                "cores_busy": round(found["cores_busy"], 2),
                "cpu_count": os.cpu_count(),
                "gpu": found["gpu"],
                "by_kind_p50_ms": {
                    kind: round(statistics.median(took), 1) for kind, took in found["by_kind_ms"].items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
