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
import json
import pathlib
import sys
import time

# The repo root on sys.path, so the script runs from any cwd without
# installation -- the same shape face_pipeline_validation.py uses (:31-34).
REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BATCHES = (1, 2, 4, 8, 16, 32, 64)


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


def measure(backend, pictures: list, size: int, repeats: int) -> dict:
    """One batch size, `repeats` passes over the same pictures.

    The fastest pass is reported. A median would fold in whatever else
    the machine was doing; the floor is the property of the code.
    """
    import torch

    on_cuda = backend.device == "cuda"
    passes = []
    for _ in range(repeats):
        preprocessed = inferred = 0.0
        started = time.perf_counter()
        for at in range(0, len(pictures), size):
            chunk = pictures[at : at + size]
            mark = time.perf_counter()
            tensor = torch.stack([backend.preprocess(picture) for picture in chunk]).to(backend.device)
            if on_cuda:
                torch.cuda.synchronize()
            preprocessed += time.perf_counter() - mark

            mark = time.perf_counter()
            with torch.no_grad():
                backend.model.encode_image(tensor, normalize=True)
            if on_cuda:
                torch.cuda.synchronize()
            inferred += time.perf_counter() - mark
        passes.append((time.perf_counter() - started, preprocessed, inferred))

    whole, preprocess, inference = min(passes, key=lambda run: run[0])
    return {
        "batch": size,
        "images_per_second": round(len(pictures) / whole, 1),
        "total_ms": round(whole * 1000, 1),
        "preprocess_ms": round(preprocess * 1000, 1),
        "inference_ms": round(inference * 1000, 1),
        "inference_ms_per_image": round(inference * 1000 / len(pictures), 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if on_cuda else None,
    }


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
    print(f"source megapixels: median {megapixels[len(megapixels) // 2]:.1f}, largest {megapixels[-1]:.1f}")

    backend = openclip.encoder(asked.models_dir)
    on_cuda = backend.device == "cuda"
    print(f"model: {backend.model_name}/{backend.checkpoint} on {backend.device}, {backend.dimensions} dimensions")
    if on_cuda:
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    # A cold first call builds kernels; timing it would slander batch 1.
    measure(backend, pictures[: min(8, len(pictures))], 4, 1)

    head = (
        f"\n{'batch':>6} {'img/sec':>9} {'total ms':>9} {'preprocess':>11} "
        f"{'inference':>10} {'infer/img':>10} {'VRAM MB':>9}"
    )
    print(head)
    print("-" * (len(head) - 1))
    rows = []
    for size in BATCHES:
        if size > len(pictures):
            break
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        row = measure(backend, pictures, size, asked.repeats)
        rows.append(row)
        print(
            f"{row['batch']:6} {row['images_per_second']:9.1f} {row['total_ms']:9.1f} "
            f"{row['preprocess_ms']:11.1f} {row['inference_ms']:10.1f} "
            f"{row['inference_ms_per_image']:10.2f} {row['peak_vram_mb'] or 0:9.1f}"
        )

    alone = vectors(backend, pictures, 1)
    drift = {}
    for size in (8, 32):
        if size > len(pictures):
            continue
        together = vectors(backend, pictures, size)
        worst = float(abs(together - alone).max())
        cosine = float((together * alone).sum(axis=1).min())
        drift[str(size)] = {"max_abs_difference": worst, "min_cosine_against_batch_1": cosine}
        print(f"\nbatch {size} against batch 1: largest element differs by {worst:.2e}, worst cosine {cosine:.8f}")

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
                "by_batch": rows,
                "equivalence": drift,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
