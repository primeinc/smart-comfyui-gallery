"""What batch size costs and buys for the OpenCLIP image encoder.

The runner works one job item at a time, and `ClipBackend.encode_media`
turns each one into a batch of one (`unsqueeze(0)`). A ViT is built to
process batches: the weights stay resident either way, so a batch of one
pays a kernel launch and a synchronisation per picture and leaves the
device mostly idle between them.

This measures the encoder ALONE -- no database, no job rows, decoding
done up front -- so the number is the ceiling any batching in the runner
could reach. Per batch size:

    preprocess   CPU: the CLIP transform and the host-to-device copy
    inference    GPU: the encode_image call
    images/sec   end to end, preprocess included

Every timed region is followed by `torch.cuda.synchronize()`, because
CUDA launches are asynchronous and timing them without it measures how
fast Python can queue work (torch/cuda/__init__.py:1209-1219). Peak VRAM
is read against a reset (torch/cuda/memory.py:380-396, :550-568).

And, because a faster wrong answer is not an answer, every batched vector
is checked against the same picture encoded alone. Batching must not move
a vector: the stored space is keyed by model and checkpoint, and the
vectors already in the index came from batch-1 runs, so any difference
beyond floating-point reassociation would silently mix two
representations in one index.

Sources come from a run's own database, decoded through the same
`oriented.for_model` the embed job uses, so the pixels are the pixels.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# The repo root on sys.path, so the script runs from any cwd without
# installation -- the same shape face_pipeline_validation.py uses (:31-34).
REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
WORKERS = (1, 2, 4, 8, 16)
#: Overlap is swept over a fixed set. Above about 32 the sequential
#: numbers are within noise of each other, so a "best" batch chosen from
#: them is a coin toss dressed as a measurement.
OVERLAP_BATCHES = (32, 64, 128)


def corpus(db: pathlib.Path, wanted: int, under: str | None) -> list[tuple[int, pathlib.Path]]:
    """`wanted` present pictures, optionally only from roots matching
    `under`.

    The filter is not a convenience. The CLIP transform's first step is a
    resize to 224, and its cost is linear in the SOURCE megapixels: 105 ms
    from 22 MP, 6.3 ms from 1 MP, 0.3 ms from 224x224. A corpus of camera
    JPEGs and a corpus of generated PNGs therefore answer two different
    questions about the same encoder, and a sweep that silently picks one
    reports the other's conclusion.

    Video is excluded: its representative frame costs a seek and a decode
    that would dominate a measurement about the encoder.
    """
    from db import connect, detect

    found: list[tuple[int, pathlib.Path]] = []
    with connect.connect(db, read_only=True) as conn:
        sql = (
            "SELECT f.id FROM file f "
            "JOIN folder fo ON fo.id = f.folder_id JOIN root r ON r.id = fo.root_id "
            "WHERE f.missing_since IS NULL AND f.kind = 'image'"
        )
        rows = (
            conn.execute(sql + " AND r.path LIKE ? ORDER BY f.id DESC", (f"%{under}%",)).fetchall()
            if under
            else conn.execute(sql + " ORDER BY f.id DESC").fetchall()
        )
        for (file_id,) in rows:
            if len(found) >= wanted:
                break
            path = pathlib.Path(detect.path_of(conn, file_id))
            if path.is_file():
                found.append((file_id, path))
    return found


def frames(db: pathlib.Path, files: list[tuple[int, pathlib.Path]]) -> list:
    """Every source decoded once, upright, held in memory.

    Decoding sits outside every timed region: this is a question about
    the encoder, and leaving decode in would measure what the thumbnail
    work already answered.
    """
    from db import connect, oriented

    held = []
    with connect.connect(db, read_only=True) as conn:
        for index, (file_id, path) in enumerate(files, 1):
            held.append(oriented.for_model(conn, file_id, path).convert("RGB"))
            print(f"\r  decoding {index}/{len(files)}", end="", flush=True)
    print()
    return held


def prepared(backend, chunk: list, pool: ThreadPoolExecutor | None):
    """One chunk through the EXACT preprocess, serially or across threads.

    `backend.preprocess` is the transform open_clip built, unchanged and
    unwrapped. Threads help because its expensive half is PIL's resize and
    torch's normalise, both of which drop the GIL; nothing about the
    pixels it produces depends on which thread produced them.
    """
    import torch

    if pool is None:
        made = [backend.preprocess(picture) for picture in chunk]
    else:
        made = list(pool.map(backend.preprocess, chunk))
    return torch.stack(made)


def _produce(backend, pictures: list, size: int, pool: ThreadPoolExecutor | None, waiting: queue.Queue) -> None:
    """Prepare every batch and hand each to the consumer, then the sentinel.

    A module-level function taking what it needs, not a closure over the
    measurement loop: it runs on another thread while that loop is free to
    move on, and a closure would read whatever the variables became.

    `non_blocking=True` lets the host-to-device copy overlap the GPU work
    already queued, which is the point of running this on its own thread.
    """
    for at in range(0, len(pictures), size):
        tensor = prepared(backend, pictures[at : at + size], pool)
        waiting.put(tensor.to(backend.device, non_blocking=True))
    waiting.put(None)


def measure(backend, pictures: list, size: int, workers: int, repeats: int, *, overlap: bool) -> dict:
    """One (batch, workers) pair, `repeats` passes over the same pictures.

    The fastest pass is reported: a median folds in whatever else the
    machine was doing, and the floor is the property of the code.

    `overlap` runs preprocessing and inference concurrently, which is what
    production should do -- they use different hardware, so the ceiling is
    `max(cpu, gpu)` rather than their sum. The phase columns are then
    wall-clock inside each stage and no longer add up to the total; that
    is the point of the mode, not a defect in it.
    """
    import torch

    on_cuda = backend.device == "cuda"
    passes = []
    for _ in range(repeats):
        preprocessed = inferred = pipelined = 0.0
        started = time.perf_counter()
        pool = ThreadPoolExecutor(workers) if workers > 1 else None
        try:
            if overlap:
                waiting: queue.Queue = queue.Queue(maxsize=2)
                mark = time.perf_counter()
                producer = threading.Thread(target=_produce, args=(backend, pictures, size, pool, waiting), daemon=True)
                producer.start()
                while True:
                    tensor = waiting.get()
                    if tensor is None:
                        break
                    with torch.no_grad():
                        backend.model.encode_image(tensor, normalize=True)
                if on_cuda:
                    torch.cuda.synchronize()
                producer.join()
                pipelined = time.perf_counter() - mark
            else:
                for at in range(0, len(pictures), size):
                    mark = time.perf_counter()
                    tensor = prepared(backend, pictures[at : at + size], pool).to(backend.device)
                    if on_cuda:
                        torch.cuda.synchronize()
                    preprocessed += time.perf_counter() - mark

                    mark = time.perf_counter()
                    with torch.no_grad():
                        backend.model.encode_image(tensor, normalize=True)
                    if on_cuda:
                        torch.cuda.synchronize()
                    inferred += time.perf_counter() - mark
        finally:
            if pool is not None:
                pool.shutdown()
        passes.append((time.perf_counter() - started, preprocessed, inferred, pipelined))

    whole, preprocess, inference, pipeline = min(passes, key=lambda run: run[0])
    row = {
        "batch": size,
        "workers": workers,
        "overlap": overlap,
        "images_per_second": round(len(pictures) / whole, 1),
        "total_ms": round(whole * 1000, 1),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if on_cuda else None,
    }
    if overlap:
        # NOT reported as preprocess and inference. In this mode the two
        # run at once, so there is one composed region and no honest way
        # to split it from the outside; reporting `preprocess 0.0` beside
        # an `inference` that is really the whole pipeline reads as an
        # encoder that got slower.
        row["pipeline_ms"] = round(pipeline * 1000, 1)
    else:
        row["preprocess_ms"] = round(preprocess * 1000, 1)
        row["inference_ms"] = round(inference * 1000, 1)
        row["inference_ms_per_image"] = round(inference * 1000 / len(pictures), 2)
    return row


def vectors(backend, pictures: list, size: int):
    """Every picture's vector, encoded `size` at a time."""
    import numpy as np
    import torch

    out = []
    for at in range(0, len(pictures), size):
        chunk = pictures[at : at + size]
        tensor = torch.stack([backend.preprocess(picture) for picture in chunk]).to(backend.device)
        with torch.no_grad():
            features = backend.model.encode_image(tensor, normalize=True)
        out.extend(features.cpu().float().numpy())
    return np.stack(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCLIP throughput against batch size")
    parser.add_argument("--db", default=str(pathlib.Path.home() / ".smartgallery" / "gallery.db"))
    parser.add_argument("--models-dir", default=str(pathlib.Path.home() / ".smartgallery" / "models"))
    parser.add_argument("--count", type=int, default=128, help="pictures in the corpus")
    parser.add_argument("--under", default=None, help="only roots whose path contains this, e.g. swarm-mixed")
    parser.add_argument("--repeats", type=int, default=3, help="passes per batch size; the fastest is reported")
    parser.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "openclip_batch.json"))
    asked = parser.parse_args()

    import torch

    from vision.semantic import openclip

    db = pathlib.Path(asked.db)
    if not db.is_file():
        raise SystemExit(f"no database at {db}")

    files = corpus(db, asked.count, asked.under)
    if not files:
        raise SystemExit("no pictures in that library")
    print(f"corpus: {len(files)} pictures from {db}" + (f" under {asked.under!r}" if asked.under else ""))
    pictures = frames(db, files)
    megapixels = sorted(picture.size[0] * picture.size[1] / 1e6 for picture in pictures)
    print(f"source megapixels: median {megapixels[len(megapixels) // 2]:.2f}, largest {megapixels[-1]:.2f}")
    if len(pictures) < asked.count:
        # Cycled to reach the target. Honest for encoder throughput --
        # the pictures are already decoded and resident, and neither the
        # transform nor the GPU cares that it has seen one before -- and
        # it is the only way to sweep a batch larger than the population.
        whole = list(pictures)
        pictures = list(itertools.islice(itertools.cycle(whole), asked.count))
        print(f"cycled {len(whole)} distinct pictures up to {len(pictures)} to sweep larger batches")

    backend = openclip.encoder(asked.models_dir)
    on_cuda = backend.device == "cuda"
    print(f"model: {backend.model_name}/{backend.checkpoint} on {backend.device}, {backend.dimensions} dimensions")
    if on_cuda:
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    # A cold first call builds kernels; timing it would slander batch 1.
    measure(backend, pictures[: min(8, len(pictures))], 4, 1, 1, overlap=False)

    rows = []

    def show(row: dict) -> None:
        rows.append(row)
        if row["overlap"]:
            stages = f"{'-':>11} {'-':>10} {row['pipeline_ms']:10.1f}"
        else:
            stages = f"{row['preprocess_ms']:11.1f} {row['inference_ms']:10.1f} {'-':>10}"
        print(
            f"{row['batch']:6} {row['workers']:8} {'yes' if row['overlap'] else 'no':>8} "
            f"{row['images_per_second']:9.1f} {row['total_ms']:9.1f} {stages} "
            f"{row['peak_vram_mb'] or 0:9.1f}"
        )

    head = (
        f"{'batch':>6} {'workers':>8} {'overlap':>8} {'img/sec':>9} {'total ms':>9} "
        f"{'preprocess':>11} {'inference':>10} {'pipeline':>10} {'VRAM MB':>9}"
    )
    print()
    print(head)
    print("-" * len(head))

    # 0. what production does today: one picture, one thread. Every
    #    later number is a multiple of this one, so it is measured, not
    #    borrowed from whichever row happened to be printed first.
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    show(measure(backend, pictures, 1, 1, asked.repeats, overlap=False))
    baseline = rows[0]
    print(f"  -> what the runner does now: {baseline['images_per_second']:.1f} img/sec")

    # 1. how far the EXACT preprocess scales across threads
    print()
    for count in WORKERS:
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        show(measure(backend, pictures, 64, count, asked.repeats, overlap=False))
    best_workers = max(rows[1:], key=lambda row: row["images_per_second"])["workers"]
    print(f"  -> preprocessing plateaus at {best_workers} workers")

    # 2. where batch throughput actually plateaus, at that worker count
    print()
    for size in BATCHES:
        if size > len(pictures):
            break
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        show(measure(backend, pictures, size, best_workers, asked.repeats, overlap=False))
    plateau = [row for row in rows if not row["overlap"] and row["workers"] == best_workers and row["batch"] >= 32]
    if plateau:
        best_of = max(plateau, key=lambda row: row["images_per_second"])["images_per_second"]
        within = [row["batch"] for row in plateau if row["images_per_second"] >= best_of * 0.95]
        print(f"  -> sequential plateau from batch {min(within)}: {min(within)}-{max(within)} are within 5%")

    # 3. the same, with preprocessing and inference overlapped. Fixed
    #    sizes, not ones derived from the sequential winner: run to run
    #    the sequential plateau moves around inside 400-460 img/sec, so
    #    picking from it printed whichever batch won the noise, twice.
    print()
    for size in OVERLAP_BATCHES:
        if size > len(pictures):
            continue
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        show(measure(backend, pictures, size, best_workers, asked.repeats, overlap=True))

    # A2: the threaded preprocess must produce the SAME TENSORS, not
    # merely call the same function. Asserting equivalence from a shared
    # function name is how a "semantics-free" optimisation stops being
    # one, so this compares bytes.
    print()
    print("preprocessed tensors, serial against threaded:")
    serial = [backend.preprocess(picture) for picture in pictures[:64]]
    identical = {}
    for count in (2, 4, 8, 16):
        with ThreadPoolExecutor(count) as pool:
            threaded = list(pool.map(backend.preprocess, pictures[:64]))
        same = all(torch.equal(a, b) for a, b in zip(serial, threaded, strict=True))
        worst = max(float((a - b).abs().max()) for a, b in zip(serial, threaded, strict=True))
        identical[str(count)] = {"bit_identical": same, "max_abs_difference": worst}
        print(f"  {count:2} workers: bit-identical {same}, largest element differs by {worst:.1e}")

    # B2: every batch size that could be chosen, not a sample of two. The
    # winner was batch 64 while only 8 and 32 had ever been compared.
    alone = vectors(backend, pictures, 1)
    drift = {}
    print()
    print("vectors, each batch size against batch 1:")
    for size in (2, 8, 32, 64, 128):
        if size > len(pictures):
            continue
        together = vectors(backend, pictures, size)
        worst = float(abs(together - alone).max())
        cosine = float((together * alone).sum(axis=1).min())
        drift[str(size)] = {
            "max_abs_difference": worst,
            "min_cosine_against_batch_1": cosine,
            "bit_identical": bool((together == alone).all()),
        }
        print(f"  batch {size:4}: largest element differs by {worst:.3e}, worst cosine {cosine:.9f}")
    print("  Drift reaches a similar magnitude from batch 8 through 128 rather")
    print("  than growing with batch size. WHY is not established here -- that")
    print("  would need kernel selection inspected, which this does not do.")
    print("  Nor does a similar magnitude make these one equivalence class:")
    print("  retrieval has to be tested against the batch actually proposed.")

    best = max(rows, key=lambda row: row["images_per_second"])
    first = rows[0]
    print(
        f"\nbatch 1 {first['images_per_second']:.1f} img/sec -> "
        f"batch {best['batch']} {best['images_per_second']:.1f} img/sec "
        f"({best['images_per_second'] / first['images_per_second']:.1f}x)"
    )

    out = pathlib.Path(asked.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "device": backend.device,
                "gpu": torch.cuda.get_device_name(0) if on_cuda else None,
                "model": f"{backend.model_name}/{backend.checkpoint}",
                "pictures": len(pictures),
                "runs": rows,
                "preprocess_equivalence": identical,
                "vector_equivalence": drift,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
