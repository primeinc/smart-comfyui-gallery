from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

from compat.corpus.loaded import CORPUS_IMAGES, shots
from compat.harness import failfast
from compat.producers.registry import Availability, Emission, every_producer

ROOT: Final[Path] = Path(__file__).resolve().parent.parent


PHOTOGRAPHS: Final[int] = CORPUS_IMAGES


def _describe(values: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "dtype": str(one.dtype),
            "shape": [int(n) for n in one.shape],
            "bytes": int(one.nbytes),
        }
        for key, one in values.items()
    }


def survey() -> dict[str, Any]:
    found = shots()[:PHOTOGRAPHS]
    producers: list[dict[str, Any]] = []
    union: dict[str, dict[str, Any]] = {}
    emitters: dict[str, list[str]] = defaultdict(list)

    for producer in every_producer():
        ready: Availability = producer.available()
        row: dict[str, Any] = {
            "producer": producer.name,
            "ready": ready.ready,
            "reason": ready.reason,
            "identity": ready.identity,
            "faces": 0,
            "keys": {},
        }
        if ready.ready and found:
            try:
                seen: list[Emission] = []
                for shot in found:
                    seen.extend(producer.observe(shot.frame))
                row["faces"] = len(seen)
                for one in seen:
                    row["keys"].update(_describe(one.values))
                for key, described in row["keys"].items():
                    union.setdefault(key, described)
                    if producer.name not in emitters[key]:
                        emitters[key].append(producer.name)
            except (ImportError, ValueError, TypeError, RuntimeError, OSError) as problem:
                row["ready"] = False
                row["reason"] = f"{type(problem).__name__}: {problem}"

                for key in row["keys"]:
                    if producer.name in emitters.get(key, []):
                        emitters[key].remove(producer.name)
                    if not emitters.get(key):
                        union.pop(key, None)
                        emitters.pop(key, None)
                row["keys"] = {}
                row["faces"] = 0
        producers.append(row)

    unavailable = [one["producer"] for one in producers if not one["ready"]]
    return {
        "photographs": [one.fixture.sha256 for one in found],
        "producers": producers,
        "union": {key: {**described, "emitted_by": sorted(emitters[key])} for key, described in sorted(union.items())},
        "population": {
            "producers": len(producers),
            "ran": len(producers) - len(unavailable),
            "unavailable": unavailable,
            "distinct_keys": len(union),
            "bytes_per_face_union": sum(one["bytes"] for one in union.values()),
        },
    }


def main() -> int:
    failfast.arm()
    out = survey()
    for row in out["producers"]:
        mark = "ok " if row["ready"] else "!! "
        print(f"{mark}{row['producer']:<32} faces {row['faces']:>3}  keys {len(row['keys']):>3}  {row['reason'][:70]}")

    print("\nUNION -- every key any available producer emitted:")
    for key, described in out["union"].items():
        shape = "x".join(str(n) for n in described["shape"]) or "scalar"
        print(f"  {key:<26} {described['dtype']:<8} {shape:<12} {described['bytes']:>8,} B  {described['emitted_by']}")

    pop = out["population"]
    print(f"\nproducers            : {pop['ran']} of {pop['producers']} ran")
    print(f"unavailable          : {pop['unavailable']}")
    print(f"distinct keys        : {pop['distinct_keys']}")
    print(f"bytes per face, union: {pop['bytes_per_face_union']:,}")

    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    target = generated / "producer_union.json"
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True, default=str))
        handle.write("\n")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
