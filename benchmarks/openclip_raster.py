"""How small a picture OpenCLIP can be given before its answers move.

The embed job spends 39% of an item decoding, and it decodes at full
resolution so that a transform can immediately reduce the shortest side
to 224 and crop. Handing the decoder a bound instead is the largest
remaining saving -- and unlike batching it changes WHAT THE MODEL SEES,
so it cannot be adopted on a throughput number alone.

This measures both halves of that trade at once. For each candidate
bound the same distinct pictures are decoded with their SHORTEST side
floored at it, encoded, and compared against the same pictures decoded
whole:

    decode ms        what the bound saves
    max |delta|      how far the vectors moved
    top-1 / top-20   whether the ANSWERS moved, text and image queries

The shortest side is what matters, not the longest. open_clip resizes the
short edge to 224 and centre-crops (`Resize(224, shortest)` then
`CenterCrop(224)`), so a picture whose short edge is already under 224
gets UPSCALED -- inventing detail the original had and paying a decode to
lose information. A bound below 224 is therefore not a cheaper answer, it
is a worse one, and the sweep says so rather than assuming it.

`None` is the baseline: the full decode, which is what production does
today and what every vector in the index came from.

WHAT IT FOUND, run separately over 60 files of each population:

    generated PNG   2687 ms -> 2336 ms   13% saved   drift 0.0
    camera raw     47074 ms -> 2290 ms   95% saved   drift 1.15e-01

The two do not average into anything. On generated PNGs -- what this
gallery is for -- a bound does NOTHING: PNG has no shrink-on-load, so the
decode is irreducible, the shortest side comes back 1328 either way, and
the vectors are bit-identical. There is no trade here to make.

Every byte of the saving and every bit of the drift is RAW, and it is not
resolution. `open_bounded` takes the camera's embedded JPEG preview
rather than developing the sensor, and a camera's rendering is not
LibRaw's -- about 35/255 in tone and white balance, which lands as a
worst cosine of 0.92 against the whole-decode vector. That is fifty times
the movement batching caused, and it is a decision about which rendering
of a raw file the model should measure, not a decode optimisation.

So this is a benchmark that argued against its own premise, which is the
point of running it before writing the code.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

# The repo root on sys.path, so the script runs from any cwd without
# installation -- the same shape face_pipeline_validation.py uses (:31-34).
REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Multiples of the model's own 224 input, plus one below it to show what
#: happens when the bound starves the transform.
BOUNDS = (168, 224, 336, 448, 672, None)


def corpus(db: pathlib.Path, wanted: int, under: str | None = None):
    """Distinct present pictures, round-robin across the library's roots.

    `--under` is load-bearing here, not a convenience. A bound does
    entirely different things to different source formats -- everything
    this measures collapses to noise when the two are averaged -- so the
    populations are meant to be run separately.
    """
    import collections

    from db import connect, detect, oriented

    by_root: dict[str, list[int]] = collections.defaultdict(list)
    found = []
    with connect.connect(db, read_only=True) as conn:
        for root, file_id in conn.execute(
            "SELECT r.path, f.id FROM file f "
            "JOIN folder fo ON fo.id = f.folder_id JOIN root r ON r.id = fo.root_id "
            "WHERE f.missing_since IS NULL AND f.kind = 'image' ORDER BY f.id"
        ):
            if under and under not in root:
                continue
            by_root[root].append(file_id)
        turns = [iter(ids) for ids in by_root.values()]
        while turns and len(found) < wanted:
            for stream in list(turns):
                if len(found) >= wanted:
                    break
                try:
                    file_id = next(stream)
                except StopIteration:
                    turns.remove(stream)
                    continue
                path = pathlib.Path(detect.path_of(conn, file_id))
                if path.is_file():
                    found.append((path, oriented.orientation_of(conn, file_id)))
    return found


def encoded(backend, files, bound: int | None):
    """Every picture decoded at `bound` and encoded, with the decode timed.

    Production functions throughout: `decode.open_bounded` with the
    shortest-side floor, `oriented.upright` with the stored tag. A
    benchmark that reimplements the decode it is judging would be
    measuring itself.
    """
    import numpy as np
    import torch

    from db import oriented
    from vision import decode

    frames = []
    started = time.perf_counter()
    for path, tag in files:
        opened = decode.open_still(path) if bound is None else decode.open_bounded(path, bound, edge="shortest")
        with opened:
            opened.load()
            frames.append(oriented.upright(opened, tag).convert("RGB"))
    decode_ms = (time.perf_counter() - started) * 1000

    sizes = [min(frame.size) for frame in frames]
    vectors = []
    for at in range(0, len(frames), 64):
        chunk = frames[at : at + 64]
        stacked = torch.stack([backend.preprocess(frame) for frame in chunk]).to(backend.device)
        with torch.no_grad():
            vectors.extend(backend.model.encode_image(stacked, normalize=True).cpu().float().numpy())
    return np.stack(vectors), decode_ms, sizes


def agreement(base, other, queries, depths):
    """Top-k agreement between two indexes over the same queries."""
    import numpy as np

    base_order = np.argsort(-(queries @ base.T), axis=1)
    other_order = np.argsort(-(queries @ other.T), axis=1)
    out = {}
    for depth in depths:
        shared = [
            len(set(base_order[q, :depth]) & set(other_order[q, :depth])) / depth for q in range(base_order.shape[0])
        ]
        out[str(depth)] = round(float(np.mean(shared)), 6)
    return out


def neighbours_agree(whole, bounded, depths):
    """Each index asked with ITS OWN vectors: do the neighbours match?

    Leave-one-out on both sides, because a picture is always its own best
    match and counting that would report an agreement neither index has.

    This is the migrated condition. A bounded index queried by bounded
    vectors is internally consistent; the question is whether it groups
    the same pictures together as the whole-decode one did, not whether
    its coordinates moved -- coordinates are allowed to move.
    """
    import numpy as np

    out = {}
    orders = []
    for index in (whole, bounded):
        scores = index @ index.T
        np.fill_diagonal(scores, -2.0)
        orders.append(np.argsort(-scores, axis=1))
    for depth in depths:
        shared = [len(set(orders[0][q, :depth]) & set(orders[1][q, :depth])) / depth for q in range(orders[0].shape[0])]
        out[str(depth)] = round(float(np.mean(shared)), 6)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="how far OpenCLIP's answers move when its input is bounded")
    parser.add_argument("--db", default=str(pathlib.Path.home() / ".smartgallery" / "gallery.db"))
    parser.add_argument("--models-dir", default=str(pathlib.Path.home() / ".smartgallery" / "models"))
    parser.add_argument("--count", type=int, default=400, help="DISTINCT pictures")
    parser.add_argument("--under", default=None, help="only roots whose path contains this; run populations apart")
    parser.add_argument("--out", default=None, help="default: benchmarks/results/openclip_raster[_<under>].json")
    asked = parser.parse_args()

    import numpy as np

    from benchmarks.openclip_retrieval import QUERIES
    from vision.semantic import openclip

    db = pathlib.Path(asked.db)
    if not db.is_file():
        raise SystemExit(f"no database at {db}")
    files = corpus(db, asked.count, asked.under)
    print(f"corpus: {len(files)} DISTINCT pictures from {db}")

    backend = openclip.encoder(asked.models_dir)
    print(f"model: {backend.model_name}/{backend.checkpoint} on {backend.device}")
    text = np.stack([backend.encode_query(phrase) for phrase in QUERIES])

    whole, whole_ms, whole_sizes = encoded(backend, files, None)
    print(f"\nfull decode: {whole_ms:.0f} ms, median shortest side {statistics.median(whole_sizes):.0f} px")

    head = (
        f"{'bound':>6} {'decode ms':>10} {'saved':>7} {'shortest':>9} {'max|delta|':>11} "
        f"{'text top1':>10} {'mixed t1':>11} {'mixed t20':>12} {'migr t1':>11} {'migr t20':>12}"
    )
    print()
    print(head)
    print("-" * len(head))
    rows = {}
    for bound in BOUNDS:
        if bound is None:
            continue
        vectors, ms, sizes = encoded(backend, files, bound)
        drift = float(np.abs(vectors - whole).max())
        by_text = agreement(whole, vectors, text, (1, 20))
        # Two conditions, and they answer different questions.
        #
        # MIXED: the full-decode vectors already in the index, asked
        # against a bounded one. That is what happens if a bound ships
        # and nothing is re-embedded.
        #
        # MIGRATED: every vector bounded, queries included, compared with
        # every vector whole. That is what happens if a bound ships AND
        # the library is re-embedded, and it is the fairer test of
        # whether the smaller raster is a worse picture or merely a
        # different one.
        mixed = agreement(whole, vectors, whole, (1, 20))
        migrated = neighbours_agree(whole, vectors, (1, 20))
        rows[str(bound)] = {
            "decode_ms": round(ms, 1),
            "saved_percent": round((whole_ms - ms) / whole_ms * 100, 1),
            "median_shortest_px": statistics.median(sizes),
            "max_abs_difference": drift,
            "text": by_text,
            "image_mixed": mixed,
            "image_migrated": migrated,
        }
        print(
            f"{bound:6} {ms:10.0f} {(whole_ms - ms) / whole_ms * 100:6.0f}% {statistics.median(sizes):9.0f} "
            f"{drift:11.3e} {by_text['1']:10.4f} {mixed['1']:11.4f} {mixed['20']:12.4f} "
            f"{migrated['1']:11.4f} {migrated['20']:12.4f}"
        )

    print(
        "\n  image queries here are the FULL-decode vectors asked against each"
        "\n  bounded index, which is the migration condition: a library embedded"
        "\n  before a bound change, queried after it."
    )

    named = f"openclip_raster{'_' + asked.under if asked.under else ''}.json"
    out = pathlib.Path(asked.out or REPO / "benchmarks" / "results" / named)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "pictures": len(files),
                "model": f"{backend.model_name}/{backend.checkpoint}",
                "full_decode_ms": round(whole_ms, 1),
                "model_input_px": 224,
                "by_bound": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
