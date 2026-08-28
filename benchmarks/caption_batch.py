"""Captioning throughput, and what it costs to buy it.

The caption job captions ONE picture per `generate()` call, in float32,
from a full-size decode. Three things are suspect there and this measures
all three against the same corpus, because two of them change what the
model SEES and therefore what it SAYS -- and a caption that got faster by
becoming a different caption is not the same feature.

    batch        one `generate()` per picture leaves the device idle
                 between them. The embed job already batches; this is the
                 same shape (db/runner.py `_Ahead`).
    precision    BLIP-base loads float32. On a GPU that is roughly twice
                 the work for a model whose output is a short sentence.
    decode       the processor resizes every picture to 384x384
                 (`image_processor.size`), so a 24-megapixel photograph is
                 decoded at eighty times the pixels the model looks at.

EQUIVALENCE IS THE POINT. Each variant's captions are compared with the
float32, batch-1, full-decode captions word for word. A speedup with a
low match rate is not a speedup, it is a different product, and the
number to look at is the match rate beside the images per second.

Run through `just bench captions`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Batch sizes worth sweeping. Past sixteen a 384x384 model is bound by
#: something other than occupancy, and the batch is a longer wait for
#: whoever asked the job to stop.
BATCHES = (1, 2, 4, 8, 16)


def corpus(db: pathlib.Path, count: int, under: str | None) -> list[tuple[int, str]]:
    """Real pictures from a real library, newest first."""
    from db import connect

    conn = connect.connect(str(db))
    try:
        sql = (
            "SELECT f.id, f.name FROM file f JOIN folder fo ON fo.id = f.folder_id"
            " JOIN root r ON r.id = fo.root_id"
            " WHERE f.missing_since IS NULL AND f.kind = 'image'"
        )
        args: list = []
        if under:
            sql += " AND r.path LIKE ?"
            args.append(f"%{under}%")
        sql += " ORDER BY f.mtime DESC LIMIT ?"
        args.append(count)
        return [(row[0], row[1]) for row in conn.execute(sql, args)]
    finally:
        connect.close(conn)


def paths_of(db: pathlib.Path, files: list[tuple[int, str]]) -> list[str]:
    from db import connect, detect

    conn = connect.connect(str(db))
    try:
        return [detect.path_of(conn, file_id) for file_id, _ in files]
    finally:
        connect.close(conn)


def _decoded(paths: list[str], bound: int | None):
    """The pictures, decoded the way the variant under test decodes them.

    `bound` None is what the job does today: the whole picture, upright,
    as `oriented.for_model` hands it over. A number is the bounded decode
    -- never larger than that on its longest side -- which is the cheaper
    route the processor's own 384 box makes tempting.
    """
    from vision import decode

    made = []
    for path in paths:
        # Through vision/decode either way: `open_still` is what the job
        # reaches (via `oriented.for_model`), and a bare `Image.open`
        # would skip the plugin registration and the RAW reader, so the
        # measurement would not be of this application's decode.
        held = decode.open_still(path) if bound is None else decode.open_bounded(path, bound)
        made.append(held.convert("RGB"))
    return made


def _captions(captioner, pictures, batch: int) -> tuple[list[str], float, float]:
    """Caption every picture at this batch size.

    Returns the captions, the preprocess milliseconds and the generate
    milliseconds -- split, because which one a change moves is the whole
    question.
    """
    import torch

    from vision.captions import MOST_TOKENS

    said: list[str] = []
    preprocess = 0.0
    generate = 0.0
    for at in range(0, len(pictures), batch):
        held = pictures[at : at + batch]
        started = time.perf_counter()
        inputs = captioner.processor(images=held, return_tensors="pt").to(captioner.device)
        if captioner.device == "cuda":
            torch.cuda.synchronize()
        middle = time.perf_counter()
        with torch.inference_mode():
            out = captioner.model.generate(**inputs, max_new_tokens=MOST_TOKENS)
        if captioner.device == "cuda":
            torch.cuda.synchronize()
        ended = time.perf_counter()
        preprocess += middle - started
        generate += ended - middle
        said.extend(text.strip() for text in captioner.processor.batch_decode(out, skip_special_tokens=True))
    return said, preprocess * 1000, generate * 1000


def _same(a: list[str], b: list[str]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    return sum(1 for one, two in zip(a, b, strict=True) if one == two) / len(a)


def main() -> None:
    parser = argparse.ArgumentParser(description="Captioning throughput, and what it costs to buy it")
    parser.add_argument("--db", default=str(pathlib.Path.home() / ".smartgallery" / "gallery.db"))
    parser.add_argument("--models-dir", default="C:/ComfyUI/output/.AImodels")
    parser.add_argument("--count", type=int, default=48, help="pictures in the corpus")
    parser.add_argument("--under", default=None, help="only roots whose path contains this")
    parser.add_argument("--bound", type=int, default=768, help="longest side for the bounded-decode variant")
    parser.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "caption_batch.json"))
    asked = parser.parse_args()

    import torch

    from vision import captions as captions_module

    db = pathlib.Path(asked.db)
    if not db.is_file():
        raise SystemExit(f"no database at {db}")
    files = corpus(db, asked.count, asked.under)
    if not files:
        raise SystemExit("no pictures in that library")
    paths = paths_of(db, files)
    paths = [one for one in paths if pathlib.Path(one).is_file()]
    if not paths:
        raise SystemExit("none of those pictures are on disk")
    if len(paths) < asked.count:
        whole = list(paths)
        paths = list(itertools.islice(itertools.cycle(whole), asked.count))
        print(f"cycled {len(whole)} distinct pictures up to {len(paths)} to sweep larger batches")
    print(f"corpus: {len(paths)} pictures from {db}" + (f" under {asked.under!r}" if asked.under else ""))

    # The concrete class, not the Protocol: this reaches past `describe`
    # into the processor and the model, which is the whole point.
    captioner = captions_module.BlipCaptioner(asked.models_dir, provision=True)
    print(f"model: {captioner.model_id} on {captioner.device}", flush=True)
    if captioner.device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    whole = _decoded(paths, None)
    megapixels = sorted(one.size[0] * one.size[1] / 1e6 for one in whole)
    print(f"source megapixels: median {megapixels[len(megapixels) // 2]:.2f}, largest {megapixels[-1]:.2f}")
    bounded = _decoded(paths, asked.bound)

    # A cold call builds kernels; timing it would slander whatever runs
    # first, which is always batch 1.
    _captions(captioner, whole[: min(4, len(whole))], 1)

    rows: list[dict] = []
    baseline: list[str] = []

    def run(label: str, pictures, batch: int, half: bool) -> None:
        nonlocal baseline
        want = torch.float16 if half else torch.float32
        if next(captioner.model.parameters()).dtype != want:
            # Unbound and unassigned, exactly as vision/captions.py does
            # and for the same two reasons: transformers' functools wrapper
            # around `Module.to` never binds, and the call mutates in place,
            # so taking its return would only cost the concrete class.
            torch.nn.Module.to(captioner.model, want)
        if captioner.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        said, preprocess, generate = _captions(captioner, pictures, batch)
        spent = (time.perf_counter() - started) * 1000
        if not baseline:
            baseline = said
        row = {
            "label": label,
            "batch": batch,
            "precision": "fp16" if half else "fp32",
            "decode": "bounded" if pictures is bounded else "whole",
            "images_per_second": round(len(pictures) / (spent / 1000), 2),
            "total_ms": round(spent, 1),
            "preprocess_ms": round(preprocess, 1),
            "generate_ms": round(generate, 1),
            "same_as_baseline": round(_same(baseline, said), 3),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if captioner.device == "cuda" else None,
        }
        rows.append(row)
        print(
            f"{row['label']:<28} {row['batch']:>5} {row['precision']:>6} {row['decode']:>8} "
            f"{row['images_per_second']:>9.2f} {row['total_ms']:>9.1f} {row['preprocess_ms']:>11.1f} "
            f"{row['generate_ms']:>10.1f} {row['same_as_baseline']:>7.1%} {row['peak_vram_mb'] or 0:>8.0f}",
            flush=True,
        )

    print(
        f"\n{'variant':<28} {'batch':>5} {'prec':>6} {'decode':>8} {'img/sec':>9} {'total ms':>9} "
        f"{'preprocess':>11} {'generate':>10} {'same':>7} {'VRAM':>8}"
    )
    # The baseline FIRST, because every match rate is measured against it.
    run("today", whole, 1, half=False)
    for batch in BATCHES[1:]:
        run(f"batched x{batch}", whole, batch, half=False)
    run("fp16", whole, 1, half=True)
    for batch in BATCHES[1:]:
        run(f"fp16 batched x{batch}", whole, batch, half=True)
    run(f"fp16 batched x8 + {asked.bound}px decode", bounded, 8, half=True)
    run(f"{asked.bound}px decode alone", bounded, 1, half=False)

    out = pathlib.Path(asked.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as sheet:
        json.dump({"corpus": len(paths), "device": captioner.device, "rows": rows}, sheet, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
