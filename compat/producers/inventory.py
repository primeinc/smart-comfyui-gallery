"""Run the real pass over real photographs and write down what came out.

This is the observation half of the storage question. The consumer lane asks
what downstream code needs; this asks what the producer actually hands over,
which is the set anything downstream could possibly be served from.

The corpus is sampled rather than exhausted -- every identity, both capture
paths, on CPU -- because the inventory is about the SHAPE of what a pass
emits, and 105 images of the same shapes cost twenty minutes to say the same
thing. The sample is deterministic and the images are named by digest, so the
evidence points at exactly which files produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from compat.corpus import index as corpus
from compat.producers import insightface_pass as producer

HERE: Path = Path(__file__).resolve().parent
GENERATED: Path = HERE.parent / "generated"

#: Images per identity, per capture role. A WIDER SLICE than
#: `corpus.loaded.CORPUS_IMAGES` on purpose, because this lane answers which
#: keys the producer ever emits; the slice is recorded in the artifact.
PER_ROLE: int = 2


def sample(samples: list[corpus.Sample], per_role: int = PER_ROLE) -> list[corpus.Sample]:
    """A deterministic slice: `per_role` of each role, for every identity.

    Sorted by digest rather than by filename so the choice does not move when
    a directory listing does, and so the same slice comes back on any machine
    holding the same corpus.
    """
    buckets: dict[tuple[str, str], list[corpus.Sample]] = {}
    for one in samples:
        buckets.setdefault((one.identity, one.role), []).append(one)

    out: list[corpus.Sample] = []
    for key in sorted(buckets):
        chosen = sorted(buckets[key], key=lambda one: one.sha256)[:per_role]
        out.extend(chosen)
    return out


def run(per_role: int = PER_ROLE) -> dict[str, Any]:
    if not corpus.KYC.is_dir():
        raise FileNotFoundError(f"corpus absent at {corpus.KYC}")

    chosen = sample(corpus.scan_kyc(), per_role)
    app = producer.analysis()
    heads = producer.loaded_models(app)

    reports: list[producer.ImageReport] = [producer.run_image(app, Path(one.path)) for one in chosen]

    faces = [face for image in reports for face in image.faces]
    derivable = [face.normed_is_derivable for face in faces]
    worst_normed = max((face.normed_max_abs_diff or 0.0) for face in faces) if faces else 0.0

    return {
        # The photographs this inventory saw, by content: three lanes select a
        # corpus slice and this one does not share the others' selector.
        "corpus_slice": {
            "per_role": per_role,
            "images": len(chosen),
            "identities": sorted({one.identity for one in chosen}),
            "sha256": sorted(one.sha256 for one in chosen),
            "selector": "compat.producers.inventory.sample: per_role per (identity, role), ordered by sha256",
        },
        "producer": {
            "pack": producer.PACK,
            "root": str(producer.MODELS_ROOT),
            "providers": list(producer.PROVIDERS),
            "det_size": list(producer.DET_SIZE),
            "heads": heads,
        },
        "runtime": {"images": len(reports), "faces": len(faces)},
        "corpus": {
            "root": str(corpus.KYC),
            "licence": corpus.LICENCE,
            "vendored": False,
            "sampled": [{"identity": one.identity, "role": one.role, "sha256": one.sha256} for one in chosen],
        },
        "fields": producer.field_costs(reports),
        "derivability": {
            "normed_embedding_from_embedding": {
                "faces": len(faces),
                "exact": sum(1 for one in derivable if one),
                "inexact": sum(1 for one in derivable if one is False),
                "worst_max_abs_diff": worst_normed,
                "rule": "raw / linalg.norm(raw), float32",
            }
        },
        # `seconds` is dropped for the same reason the case evidence drops it:
        # a fact about this machine, not the observation, and the only thing
        # that changes between consecutive runs.
        "images": [{key: value for key, value in asdict(one).items() if key != "seconds"} for one in reports],
    }


def report(out: dict[str, Any]) -> None:
    print(f"pack {out['producer']['pack']}  providers {out['producer']['providers']}")
    print("heads:")
    for name, kind in sorted(out["producer"]["heads"].items()):
        print(f"    {name:<14} {kind}")

    print(f"\nimages {out['runtime']['images']}   faces {out['runtime']['faces']}")

    print(f"\n{'field':<20} {'kind':<9} {'dtype':<10} {'shape':<14} {'bytes':>8} {'at 22k':>12} {'at 1M':>14}")
    rows = sorted(out["fields"].items(), key=lambda one: -one[1]["bytes_per_face"])
    for key, row in rows:
        shapes = ",".join(row["shapes"])
        print(
            f"{key:<20} {row['kind']:<9} {row['dtype']:<10} {shapes:<14} "
            f"{row['bytes_per_face']:>8} {row['at_22k']:>12,} {row['at_1m']:>14,}"
        )
    total = sum(row["bytes_per_face"] for row in out["fields"].values())
    print(f"{'TOTAL':<20} {'':<9} {'':<10} {'':<14} {total:>8} {total * 22_000:>12,} {total * 1_000_000:>14,}")

    derived = out["derivability"]["normed_embedding_from_embedding"]
    print(
        f"\nnormed_embedding derivable from embedding: {derived['exact']}/{derived['faces']} exact"
        f"   worst |diff| {derived['worst_max_abs_diff']}"
    )


def main() -> int:
    out = run()
    report(out)

    GENERATED.mkdir(parents=True, exist_ok=True)
    where = GENERATED / "producer_inventory.json"
    with where.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True, default=str))
        handle.write("\n")
    print(f"\nwrote {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
