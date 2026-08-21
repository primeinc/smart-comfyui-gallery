"""Calibrate the fingerprint thresholds against real pictures.

`dupe_threshold` (pHash radius) and `dupe_dhash_verify` (the dHash
second opinion) shipped as policy guesses. This measures them: real
images from the sample datasets, a ladder of labeled transformations,
and precision/recall over the actual two-stage decision the dupes job
makes -- pHash proposes, dHash verifies.

Usage:
    python benchmarks/fingerprint_calibration.py [--datasets DIR] [--sample N]

Classes, by product intent:

    duplicate   the same picture as files actually rot: re-encodes,
                resizes, double-JPEG. MUST group.
    variant     recognisably the same shot, touched: crops, brightness,
                a watermark. The product's "likely variant" tier --
                reported, not scored as either positive or negative.
    edited      meaningfully changed content: a quarter of the frame
                replaced, half composited from another picture. Must NOT
                group as a duplicate.
    unrelated   different pictures -- including WITHIN-dataset pairs,
                which are the hard negatives (two KYC portraits share
                composition the way false positives actually happen).

Writes benchmarks/results/fingerprint_calibration.json and prints the
operating-point table. Deterministic: seeded sampling, fixed ladder.
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from vision import dupes

DATASETS = "C:/ComfyUI/output/sample-datasets"
RESULTS = pathlib.Path(__file__).resolve().parent / "results" / "fingerprint_calibration.json"

P_GRID = (2, 4, 6, 8, 10, 12, 16)
D_GRID = (4, 8, 12, 16, 20, 24, 32, None)  # None = verification off


def originals(root: pathlib.Path, sample: int, rng) -> list[pathlib.Path]:
    """A stratified sample across every dataset directory, so portraits,
    renders and interiors all weigh in."""
    per_set: dict[str, list[pathlib.Path]] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and "copies-of-copies" not in path.parts:
            per_set.setdefault(path.relative_to(root).parts[0], []).append(path)
    told: list[pathlib.Path] = []
    names = sorted(per_set)
    share = max(sample // max(len(names), 1), 1)
    for name in names:
        held = sorted(per_set[name])
        picks = rng.choice(len(held), size=min(share, len(held)), replace=False)
        told.extend(held[i] for i in sorted(picks))
    return told[:sample]


def _reencoded(image: Image.Image, encoding: str, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format=encoding, quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def variants(image: Image.Image, other: Image.Image, rng) -> list[tuple[str, str, Image.Image]]:
    """(class, transform, image) -- the ladder, labelled."""
    w, h = image.size
    told: list[tuple[str, str, Image.Image]] = [
        ("duplicate", "jpeg_q80", _reencoded(image, "JPEG", 80)),
        ("duplicate", "jpeg_q50", _reencoded(image, "JPEG", 50)),
        ("duplicate", "resize_half", image.resize((max(8, w // 2), max(8, h // 2)))),
        ("duplicate", "resize_quarter_q80", _reencoded(image.resize((max(8, w // 4), max(8, h // 4))), "JPEG", 80)),
        ("duplicate", "double_jpeg", _reencoded(_reencoded(image, "JPEG", 90), "JPEG", 60)),
        ("variant", "crop_12th", image.crop((w // 12, h // 12, w - w // 12, h - h // 12))),
        ("variant", "crop_6th", image.crop((w // 6, h // 6, w - w // 6, h - h // 6))),
        ("variant", "brightness_+15", Image.eval(image, lambda px: min(int(px * 1.15), 255))),
        ("variant", "watermark", _watermarked(image)),
    ]
    # edited: a quarter of the frame replaced by another picture's block
    patched = image.copy()
    block = other.resize((w // 2, h // 2))
    patched.paste(block, (int(rng.integers(0, max(w // 2, 1))), int(rng.integers(0, max(h // 2, 1)))))
    told.append(("edited", "quarter_replaced", patched))
    # edited: half the frame composited from another picture
    halved = image.copy()
    halved.paste(other.resize((w, h // 2)), (0, h // 2))
    told.append(("edited", "half_composited", halved))
    return told


def _watermarked(image: Image.Image) -> Image.Image:
    """A light banner strip -- the shape site watermarks take."""
    marked = image.convert("RGB").copy()
    w, h = marked.size
    strip = Image.new("RGB", (w, max(h // 12, 1)), (255, 255, 255))
    marked.paste(Image.blend(marked.crop((0, h - strip.size[1], w, h)), strip, 0.6), (0, h - strip.size[1]))
    return marked


def measure(root: pathlib.Path, sample: int) -> dict:
    rng = np.random.default_rng(7)
    sources = originals(root, sample, rng)
    if len(sources) < 20:
        raise SystemExit(f"only {len(sources)} usable images under {root}; calibration needs a real corpus")
    print(f"corpus: {len(sources)} originals across {root}", flush=True)

    pairs: list[dict] = []
    hashes: list[tuple[str, int, int]] = []  # (dataset, phash, dhash) of each original
    t0 = time.perf_counter()

    def hash_one(at, path):
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                other_path = sources[(at + len(sources) // 2) % len(sources)]
                with Image.open(other_path) as second:
                    other = second.convert("RGB")
                base_p, base_d = dupes.perceptual(image)
                hashes.append((path.relative_to(root).parts[0], base_p, base_d))
                for label, transform, body in variants(image, other, rng):
                    body_p, body_d = dupes.perceptual(body)
                    pairs.append(
                        {
                            "class": label,
                            "transform": transform,
                            "p": dupes.hamming(base_p, body_p),
                            "d": dupes.hamming(base_d, body_d),
                        }
                    )
        except OSError as why:
            print(f"  skipped {path.name}: {why}", flush=True)

    for at, path in enumerate(sources):
        hash_one(at, path)
    print(f"transforms hashed in {time.perf_counter() - t0:.0f}s; {len(pairs)} labelled pairs", flush=True)

    # unrelated: every consecutive within-dataset pair (hard) and a
    # cross-dataset sample (easy)
    for (set_a, pa, da), (set_b, pb, db) in itertools.pairwise(hashes):
        kind = "unrelated_within" if set_a == set_b else "unrelated_cross"
        pairs.append({"class": "unrelated", "transform": kind, "p": dupes.hamming(pa, pb), "d": dupes.hamming(da, db)})
    rng2 = np.random.default_rng(11)
    for _ in range(len(hashes)):
        i, j = rng2.choice(len(hashes), size=2, replace=False)
        pairs.append(
            {
                "class": "unrelated",
                "transform": "unrelated_random",
                "p": dupes.hamming(hashes[i][1], hashes[j][1]),
                "d": dupes.hamming(hashes[i][2], hashes[j][2]),
            }
        )
    return {"originals": len(sources), "pairs": pairs}


def analyse(measured: dict) -> dict:
    pairs = measured["pairs"]

    def stats(rows, key):
        held = sorted(row[key] for row in rows)
        if not held:
            return {}
        return {
            "n": len(held),
            "p50": held[len(held) // 2],
            "p90": held[int(len(held) * 0.9)],
            "max": held[-1],
        }

    by_class: dict = {}
    for label in ("duplicate", "variant", "edited", "unrelated"):
        rows = [row for row in pairs if row["class"] == label]
        transforms = sorted({row["transform"] for row in rows})
        by_class[label] = {
            "phash": stats(rows, "p"),
            "dhash": stats(rows, "d"),
            "transforms": {
                t: {
                    "phash": stats([r for r in rows if r["transform"] == t], "p"),
                    "dhash": stats([r for r in rows if r["transform"] == t], "d"),
                }
                for t in transforms
            },
        }

    positives = [row for row in pairs if row["class"] == "duplicate"]
    negatives = [row for row in pairs if row["class"] in ("edited", "unrelated")]
    grid = []
    for p_cut in P_GRID:
        for d_cut in D_GRID:

            def admitted(row, p_cut=p_cut, d_cut=d_cut):
                return row["p"] <= p_cut and (d_cut is None or row["d"] <= d_cut)

            tp = sum(1 for row in positives if admitted(row))
            fp = sum(1 for row in negatives if admitted(row))
            vetoed_fp = sum(1 for row in negatives if row["p"] <= p_cut and d_cut is not None and row["d"] > d_cut)
            vetoed_tp = sum(1 for row in positives if row["p"] <= p_cut and d_cut is not None and row["d"] > d_cut)
            grid.append(
                {
                    "phash": p_cut,
                    "dhash": d_cut,
                    "recall": round(tp / len(positives), 4) if positives else None,
                    "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                    "false_positives": fp,
                    "verifier_vetoed_negatives": vetoed_fp,
                    "verifier_vetoed_positives": vetoed_tp,
                }
            )
    return {"by_class": by_class, "grid": grid, "positives": len(positives), "negatives": len(negatives)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=DATASETS)
    parser.add_argument("--sample", type=int, default=240)
    args = parser.parse_args()

    measured = measure(pathlib.Path(args.datasets), args.sample)
    told = analyse(measured)
    told["datasets"] = args.datasets
    told["sample"] = args.sample
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(told, indent=1), encoding="utf-8")

    print(f"\npositives={told['positives']} negatives={told['negatives']}")
    print(f"{'P':>3} {'D':>4} {'recall':>7} {'precision':>9} {'FP':>4} {'D vetoed FP':>11} {'D vetoed TP':>11}")
    for row in told["grid"]:
        d_label = "off" if row["dhash"] is None else row["dhash"]
        print(
            f"{row['phash']:>3} {d_label:>4} {row['recall']:>7} {row['precision']:>9}"
            f" {row['false_positives']:>4} {row['verifier_vetoed_negatives']:>11}"
            f" {row['verifier_vetoed_positives']:>11}"
        )
    print(f"\nwritten: {RESULTS}")


if __name__ == "__main__":
    main()
