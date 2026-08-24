"""Does batching change what a search returns?

The throughput work established that batched image vectors differ from
batch-1 vectors by about 2.2e-03 per element, worst cosine 0.99994. That
bounds the VECTORS. Retrieval is a different question: a ranking is
decided by the gaps BETWEEN neighbours, and two candidates separated by
less than the drift can swap places however small the drift is.

So this asks directly. The same distinct pictures are encoded twice --
once one at a time, once at the batch size actually proposed -- and both
sets of vectors are searched with the same queries.

    top-1 agreement      did the best answer change
    top-k overlap        did the SET change, ignoring order
    rank displacement    how far did the top-k items move
    score delta          how far did the similarities move
    near-tie swaps       of the adjacent pairs closer together than the
                         drift, how many actually crossed

The last one is the point. It counts the pairs that COULD reorder under a
perturbation this size and then how many did, which is the difference
between "the vectors barely moved" and "the answers barely moved".

Two query kinds, because they stress different parts of the space. Text
goes through the same text tower a person's search uses. Image queries
are leave-one-out -- every picture asked for its neighbours, which is
what the duplicate and similarity surfaces do, and which produces
hundreds of queries rather than a phrase list's handful.

A third index takes alternating vectors from each run. That is the
MIGRATION condition: a library embedded before this change and topped up
after it. It is what would ship first, so it is the one worth knowing.

The corpus is DISTINCT and spread across roots. openclip_batch.py cycles
a small population to reach a target so it can sweep large batches, which
is right for measuring an encoder and useless here -- a ranking test over
repeated pictures compares a picture with copies of itself.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

# The repo root on sys.path, so the script runs from any cwd without
# installation -- the same shape face_pipeline_validation.py uses (:31-34).
REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Phrases a person might type at a gallery holding both generated images
#: and camera photographs. A mix of subject, style and setting, so the
#: queries do not all land in one corner of the space.
QUERIES = (
    "a portrait of a woman",
    "a man smiling at the camera",
    "a group of people outdoors",
    "a landscape at sunset",
    "a city street at night",
    "an animal close up",
    "a plate of food",
    "a black and white photograph",
    "a colourful abstract pattern",
    "a screenshot of a user interface",
    "a drawing in pencil",
    "a photograph taken indoors",
    "text written on a sign",
    "a car on a road",
    "a building seen from below",
    "something red",
)


def corpus(db: pathlib.Path, wanted: int) -> list[tuple[int, pathlib.Path]]:
    """Distinct present pictures, round-robin across the library's roots.

    Not `ORDER BY id`, which takes them all from whichever root holds the
    newest files -- the mistake that had an earlier measurement report
    the camera JPEGs' answer as the whole library's.
    """
    from db import connect, detect

    by_root: dict[str, list[int]] = collections.defaultdict(list)
    found: list[tuple[int, pathlib.Path]] = []
    with connect.connect(db, read_only=True) as conn:
        rows = conn.execute(
            "SELECT r.path, f.id FROM file f "
            "JOIN folder fo ON fo.id = f.folder_id JOIN root r ON r.id = fo.root_id "
            "WHERE f.missing_since IS NULL AND f.kind = 'image' ORDER BY f.id"
        ).fetchall()
        for root, file_id in rows:
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
                    found.append((file_id, path))
    return found


def decoded(db: pathlib.Path, files: list[tuple[int, pathlib.Path]]) -> list:
    """Every picture, decoded once and held.

    Once, not once per run: the two encodings must see IDENTICAL pixels,
    or the comparison has two variables in it and answers neither.
    """
    from db import connect, oriented

    held = []
    with connect.connect(db, read_only=True) as conn:
        for index, (file_id, path) in enumerate(files, 1):
            held.append(oriented.for_model(conn, file_id, path).convert("RGB"))
            print(f"\r  decoding {index}/{len(files)}", end="", flush=True)
    print()
    return held


def encoded(backend, pictures: list, size: int):
    """Every picture's vector, encoded `size` at a time."""
    import numpy as np
    import torch

    out = []
    for at in range(0, len(pictures), size):
        chunk = pictures[at : at + size]
        tensor = torch.stack([backend.preprocess(picture) for picture in chunk]).to(backend.device)
        with torch.no_grad():
            out.extend(backend.model.encode_image(tensor, normalize=True).cpu().float().numpy())
        print(f"\r  encoding at batch {size}: {min(at + size, len(pictures))}/{len(pictures)}", end="", flush=True)
    print()
    return np.stack(out)


def searched(index, queries, *, leave_one_out: bool):
    """Similarity of every query against every vector, and the ordering.

    Both sides are unit length (open_clip normalises), so the inner
    product IS the cosine and a descending sort is the ranking.

    `leave_one_out` removes each picture from its own results. A picture
    is always its own best match, and counting that would report an
    agreement the search never actually has.
    """
    import numpy as np

    scores = queries @ index.T
    if leave_one_out:
        np.fill_diagonal(scores, -2.0)
    return scores, np.argsort(-scores, axis=1)


def compare(name: str, base_scores, base_order, other_scores, other_order, depths, drift: float) -> dict:
    """One index against the batch-1 index, over every query."""
    import numpy as np

    queries = base_order.shape[0]
    deepest = max(depths)
    top1 = float(np.mean(base_order[:, 0] == other_order[:, 0]))

    overlaps = {}
    for depth in depths:
        shared = [len(set(base_order[q, :depth]) & set(other_order[q, :depth])) / depth for q in range(queries)]
        overlaps[str(depth)] = round(float(np.mean(shared)), 6)

    # Where each of the base top-k landed in the other ordering. Reported
    # instead of a rank correlation because "the third result moved to
    # fourth" is the thing a person would notice, and a correlation
    # coefficient hides which end of the list moved.
    displacements = []
    for q in range(queries):
        landed = np.empty(other_order.shape[1], dtype=np.int64)
        landed[other_order[q]] = np.arange(other_order.shape[1])
        displacements.append(np.abs(landed[base_order[q, :deepest]] - np.arange(deepest)))
    displacement = np.concatenate(displacements)

    moved = np.abs(base_scores - other_scores)

    # Adjacent pairs closer together than the drift are the ones that
    # COULD swap, counted twice: over the whole ranking, and over the
    # top-k alone.
    #
    # The whole-ranking number is nearly useless on its own and is
    # reported to say so. Most items are far from any given query, so
    # their scores bunch up in the tail and around 90% of all adjacent
    # pairs sit inside the drift -- a swap between the 600th and 601st
    # result is not a changed answer. The top-k count is the one that
    # corresponds to something a person sees.
    close = swapped = 0
    close_top = swapped_top = 0
    for q in range(queries):
        order = base_order[q]
        gaps = base_scores[q, order[:-1]] - base_scores[q, order[1:]]
        landed = np.empty(other_order.shape[1], dtype=np.int64)
        landed[other_order[q]] = np.arange(other_order.shape[1])

        at_risk = np.flatnonzero(gaps < drift)
        close += int(at_risk.size)
        swapped += int(np.count_nonzero(landed[order[at_risk]] > landed[order[at_risk + 1]]))

        near_top = at_risk[at_risk < deepest]
        close_top += int(near_top.size)
        swapped_top += int(np.count_nonzero(landed[order[near_top]] > landed[order[near_top + 1]]))

    return {
        "index": name,
        "top1_agreement": round(top1, 6),
        "top_k_overlap": overlaps,
        "rank_displacement_mean": round(float(displacement.mean()), 6),
        "rank_displacement_max": int(displacement.max()),
        "rank_displacement_unmoved": round(float(np.mean(displacement == 0)), 6),
        "score_delta_max": round(float(moved.max()), 8),
        "score_delta_mean": round(float(moved.mean()), 8),
        "adjacent_pairs_closer_than_drift": close,
        "of_those_swapped": swapped,
        "top_k_adjacent_pairs_closer_than_drift": close_top,
        "of_those_swapped_in_top_k": swapped_top,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="does batching change what a search returns")
    parser.add_argument("--db", default=str(pathlib.Path.home() / ".smartgallery" / "gallery.db"))
    parser.add_argument("--models-dir", default=str(pathlib.Path.home() / ".smartgallery" / "models"))
    parser.add_argument("--count", type=int, default=800, help="DISTINCT pictures; never cycled")
    parser.add_argument("--batch", type=int, default=64, help="the batch size actually proposed for production")
    parser.add_argument("--depths", default="1,5,20", help="top-k depths to compare")
    parser.add_argument("--out", default=str(REPO / "benchmarks" / "results" / "openclip_retrieval.json"))
    asked = parser.parse_args()

    import numpy as np

    from vision.semantic import openclip

    db = pathlib.Path(asked.db)
    if not db.is_file():
        raise SystemExit(f"no database at {db}")
    depths = [int(part) for part in asked.depths.split(",")]

    files = corpus(db, asked.count)
    if len(files) < max(depths) * 2:
        raise SystemExit(f"only {len(files)} pictures; too few to rank")
    print(f"corpus: {len(files)} DISTINCT pictures from {db}")

    backend = openclip.encoder(asked.models_dir)
    print(f"model: {backend.model_name}/{backend.checkpoint} on {backend.device}")

    pictures = decoded(db, files)
    print("batch 1:")
    one = encoded(backend, pictures, 1)
    print(f"batch {asked.batch}:")
    many = encoded(backend, pictures, asked.batch)

    drift = float(np.abs(many - one).max())
    print(f"\nvector drift: largest element {drift:.3e}, worst cosine {float((many * one).sum(axis=1).min()):.9f}")

    # The migration condition. Alternating rather than a split half, so
    # both generations are spread through the corpus instead of
    # correlating with insertion order.
    mixed = one.copy()
    mixed[1::2] = many[1::2]

    text = np.stack([backend.encode_query(phrase) for phrase in QUERIES])
    results = {}
    for label, queries, leave_out in (("text", text, False), ("image", one, True)):
        base_scores, base_order = searched(one, queries, leave_one_out=leave_out)
        print(f"\n{label} queries ({len(queries)}):")
        for name, index in (("batched", many), ("mixed", mixed)):
            other_scores, other_order = searched(index, queries, leave_one_out=leave_out)
            found = compare(name, base_scores, base_order, other_scores, other_order, depths, drift)
            results[f"{label}:{name}"] = found
            overlap = "  ".join(f"top{d} {found['top_k_overlap'][str(d)]:.4f}" for d in depths)
            print(f"  {name:8} top1 {found['top1_agreement']:.4f}   {overlap}")
            print(
                f"           rank displacement mean {found['rank_displacement_mean']:.3f}, "
                f"max {found['rank_displacement_max']}, "
                f"{found['rank_displacement_unmoved'] * 100:.1f}% unmoved"
            )
            print(
                f"           score delta max {found['score_delta_max']:.2e}   "
                f"near-ties swapped: {found['of_those_swapped_in_top_k']}/"
                f"{found['top_k_adjacent_pairs_closer_than_drift']} in top{max(depths)}, "
                f"{found['of_those_swapped']}/{found['adjacent_pairs_closer_than_drift']} overall"
            )

    out = pathlib.Path(asked.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "pictures": len(files),
                "batch": asked.batch,
                "model": f"{backend.model_name}/{backend.checkpoint}",
                "vector_drift_max": drift,
                "text_queries": list(QUERIES),
                "comparisons": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
