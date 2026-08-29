"""The answer, derived from evidence rather than argued.

    What is the minimum canonical face/media evidence this application must
    durably retain after an expensive observation pass so that every supported
    downstream consumer can later be served without reopening the original
    source media or repeating that expensive producer computation?

Every other module produces evidence. This one reads it and states the answer,
with the case name that proves each line. Nothing here decides anything: a
primitive is in the minimum set because an ablation removed it and a replay
broke, and it is out because one did not.

DERIVED FROM, and nothing else:

    cases.json              which primitives were ablated, and what broke
    producer_union.json     every key any producer emits, and its byte cost
    vendor_acceptance.json  which vendors reproduced their own example

THE THREE RULES
---------------
NECESSARY    an ablation removed it and the replay broke. It must be stored,
             or the consumer that needed it cannot be served.
DERIVABLE    an ablation removed it and the replay still reproduced. Storing
             it is storing the same fact twice -- CONTRADICTED, and the
             suite treats a passing necessity claim as a failure.
SUBSTITUTED  a cheaper stored value was offered in its place and the replay
             broke. That is not a claim about the primitive; it is a claim
             that the cheap thing does not serve, and it is what rules a
             storage strategy OUT rather than in.

UNRETAINABLE is the fourth state and the one that matters most for schema
work: a key some producer emits that the store demonstrably cannot give back.
It is not a loss to fix by widening a column; there is no column.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
GENERATED: Final[Path] = ROOT / "generated"


@dataclass
class Primitive:
    """One retained value, and what the evidence says about it."""

    name: str
    verdict: str
    consumers: list[str] = field(default_factory=list)
    proof: list[str] = field(default_factory=list)
    bytes_per_face: int = 0
    note: str = ""


def _read(name: str) -> dict[str, Any]:
    where = GENERATED / name
    if not where.is_file():
        raise FileNotFoundError(
            f"{where} is absent. The answer is a VIEW over evidence and cannot be produced without it: "
            f"run `just compat run` first."
        )
    return json.loads(where.read_text(encoding="utf-8"))


def _optional(name: str) -> dict[str, Any]:
    where = GENERATED / name
    return json.loads(where.read_text(encoding="utf-8")) if where.is_file() else {}


def necessity() -> dict[str, Primitive]:
    """Every ablated primitive, classified by what actually happened.

    `expect_breaks` is the CLAIM; `observed_break` is the fact. They are read
    separately on purpose: a primitive whose removal was expected to break and
    did not is the finding, not a mismatch to smooth over.

    AGGREGATION IS BY WORST CASE, NOT BY LAST CASE. The first version let the
    last row seen decide, which collapsed distinct cases into one verdict:
    `order_reversed` reported SUBSTITUTE_SERVES because `refset_stack_single_A`
    reverses a ONE-element set, where reversal is a no-op -- while the same
    primitive breaks a three-element stack. A primitive that is necessary
    ANYWHERE is necessary; a substitute that fails ANYWHERE does not serve.
    """
    cases = _read("cases.json")
    out: dict[str, Primitive] = {}
    broke_somewhere: dict[str, bool] = {}
    survived_somewhere: dict[str, bool] = {}
    kinds: dict[str, str] = {}

    for row in cases["results"]:
        for one in row.get("ablations", []):
            name = one["primitive"]
            held = out.setdefault(name, Primitive(name=name, verdict="UNTESTED"))
            if row["consumer_id"] not in held.consumers:
                held.consumers.append(row["consumer_id"])
            broke = bool(one["observed_break"])
            kinds[name] = one.get("kind", "removal")
            broke_somewhere[name] = broke_somewhere.get(name, False) or broke
            survived_somewhere[name] = survived_somewhere.get(name, False) or not broke
            # Only a BREAK is quoted. Both verdicts that matter -- NECESSARY
            # and SUBSTITUTE_FAILS -- rest on something breaking, so a quoted
            # survival would read as evidence against the verdict beside it.
            # DERIVABLE has no break to quote and is labelled as such below.
            if broke and len(held.proof) < 3:
                held.proof.append(f"{row['case']}: {one['detail'][:60]}")

    for name, held in out.items():
        if kinds[name] == "substitution":
            held.verdict = "SUBSTITUTE_FAILS" if broke_somewhere[name] else "SUBSTITUTE_SERVES"
            if broke_somewhere[name] and survived_somewhere[name]:
                held.note = "fails on some arrangements and serves on others; see per-case evidence"
        elif broke_somewhere[name]:
            held.verdict = "NECESSARY"
            if survived_somewhere[name]:
                held.note = "removal broke some cases and not others; necessary where it broke"
        else:
            held.verdict = "DERIVABLE"
        if not held.proof:
            # DERIVABLE has no break to quote -- the whole finding is that
            # nothing broke. Say that, rather than reporting an absent record.
            held.proof.append("every removal reproduced the baseline: derivable from what remained")
    return out


def store_returns() -> set[str]:
    """Keys `gallery_storage` actually gave back, from its own case rows.

    The authority on what the store can return is what it returned. Read from
    the boundary name -- "<candidate>|<key>|<shot>" -- rather than declared.
    """
    cases = _read("cases.json")
    out: set[str] = set()
    for row in cases["results"]:
        if row["consumer_id"] != "gallery_storage":
            continue
        parts = str(row.get("baseline", {}).get("name") or "").split("|")
        if len(parts) >= 2:
            out.add(parts[1])
    return out


def unretainable() -> list[Primitive]:
    """Keys a producer emits that the store has no column for.

    Derived by SUBTRACTION -- the producer union minus what gallery_storage
    returned -- rather than by a lane of its own.

    There was such a lane. It compared `ones(1)` against
    `ones(1) if key in a set else empty`, which executes nothing: it was a
    bookkeeping assertion wearing a case's clothes, and it leaked its own
    scaffolding variable `storable` into the MUST RETAIN list as though the
    store needed to keep it. Subtraction over evidence two real lanes already
    produced says the same thing and claims nothing extra.
    """
    union = _optional("producer_union.json").get("union", {})
    returned = store_returns()
    out = [
        Primitive(
            name=key,
            verdict="UNRETAINABLE",
            bytes_per_face=int(described.get("bytes", 0)),
            consumers=list(described.get("emitted_by", [])),
            proof=[f"emitted by {', '.join(described.get('emitted_by', []))}; gallery_storage returned no such key"],
            note="no column",
        )
        for key, described in union.items()
        if key not in returned
    ]
    return sorted(out, key=lambda one: -one.bytes_per_face)


def lossy() -> list[Primitive]:
    """Keys the store keeps but does not return unchanged."""
    cases = _read("cases.json")
    union = _optional("producer_union.json").get("union", {})
    worst: dict[str, float] = {}
    method: dict[str, str] = {}
    for row in cases["results"]:
        if row["consumer_id"] != "gallery_storage" or row["verdict"] != "FAIL":
            continue
        key = str(row.get("baseline", {}).get("name") or "").split("|")
        if len(key) < 2:
            continue
        name = key[1]
        if row.get("max_abs_diff") is not None:
            worst[name] = max(worst.get(name, 0.0), float(row["max_abs_diff"]))
        method[name] = row["comparison"].split(":")[0]
    return [
        Primitive(
            name=name,
            verdict="LOSSY",
            bytes_per_face=int(union.get(name, {}).get("bytes", 0)),
            consumers=list(union.get(name, {}).get("emitted_by", [])),
            proof=[f"worst {worst.get(name, 0.0):g} ({method.get(name, '?')})"],
            note="stored, but not returned unchanged",
        )
        for name in sorted(method)
    ]


def build() -> dict[str, Any]:
    ablations = necessity()
    union = _optional("producer_union.json")
    accepted = _optional("vendor_acceptance.json").get("population", {}).get("vendor_accepted", [])

    by_verdict: dict[str, list[Primitive]] = defaultdict(list)
    for one in ablations.values():
        by_verdict[one.verdict].append(one)

    # Names belonging to a lane's own scaffolding rather than to anything a
    # store would keep. `derive_256_from_336` is aligned_crop asking whether
    # one crop size can be resampled from another -- a question about a
    # transform, not a retained value. Excluded from the durable set and
    # reported separately, so the exclusion is visible rather than silent.
    scaffolding = {"derive_256_from_336"}
    keep = sorted((one for one in by_verdict["NECESSARY"] if one.name not in scaffolding), key=lambda one: one.name)
    internal = sorted((one for one in by_verdict["NECESSARY"] if one.name in scaffolding), key=lambda one: one.name)
    drop = sorted(by_verdict["DERIVABLE"], key=lambda one: one.name)
    ruled_out = sorted(by_verdict["SUBSTITUTE_FAILS"], key=lambda one: one.name)
    cheaper_serves = sorted(by_verdict["SUBSTITUTE_SERVES"], key=lambda one: one.name)

    return {
        "question": (
            "the minimum canonical evidence this application must durably retain after one "
            "expensive observation pass, so every supported consumer is served without reopening "
            "the source media or re-running the producer"
        ),
        "derived_from": {
            "cases": _read("cases.json")["cases"],
            "producers_ran": union.get("population", {}).get("ran", 0),
            "union_keys": union.get("population", {}).get("distinct_keys", 0),
            "union_bytes_per_face": union.get("population", {}).get("bytes_per_face_union", 0),
            "vendor_accepted": accepted,
        },
        "must_retain": [asdict(one) for one in keep],
        "lane_internal": [asdict(one) for one in internal],
        "derivable": [asdict(one) for one in drop],
        "substitutes_that_fail": [asdict(one) for one in ruled_out],
        "substitutes_that_serve": [asdict(one) for one in cheaper_serves],
        "unretainable_today": [asdict(one) for one in unretainable()],
        "lossy_today": [asdict(one) for one in lossy()],
    }


def main() -> int:
    out = build()
    held = out["derived_from"]
    print("QUESTION")
    print(f"  {out['question']}")
    print(f"\nDERIVED FROM {held['cases']} executed cases, {held['producers_ran']} producers, ")
    print(f"  union of {held['union_keys']} keys = {held['union_bytes_per_face']:,} B per face")
    print(f"  vendor-accepted consumers: {held['vendor_accepted'] or 'none'}")

    print(f"\nMUST RETAIN -- removal broke a replay ({len(out['must_retain'])})")
    for one in out["must_retain"]:
        print(f"  {one['name']:<28} {len(one['consumers']):>2} consumers   {one['proof'][0][:78]}")

    if out["lane_internal"]:
        print(f"\nLANE-INTERNAL -- necessary to a case, not a stored value ({len(out['lane_internal'])})")
        for one in out["lane_internal"]:
            print(f"  {one['name']:<28} {one['proof'][0][:78]}")

    print(f"\nDERIVABLE -- removal broke nothing ({len(out['derivable'])})")
    for one in out["derivable"]:
        print(f"  {one['name']:<28} {one['proof'][0][:78]}")
    if not out["derivable"]:
        print("  (none: every retained primitive was needed by something)")

    print(f"\nSUBSTITUTES THAT FAIL -- rules a cheaper strategy OUT ({len(out['substitutes_that_fail'])})")
    for one in out["substitutes_that_fail"]:
        print(f"  {one['name']:<28} {one['proof'][0][:78]}")

    print(f"\nSUBSTITUTES THAT SERVE -- a cheaper strategy WORKS ({len(out['substitutes_that_serve'])})")
    for one in out["substitutes_that_serve"]:
        print(f"  {one['name']:<28} {one['proof'][0][:78]}")
    if not out["substitutes_that_serve"]:
        print("  (none)")

    absent = out["unretainable_today"]
    print(f"\nUNRETAINABLE TODAY -- emitted, and the store has no column ({len(absent)})")
    for one in absent:
        print(f"  {one['name']:<28} {one['bytes_per_face']:>7,} B  {', '.join(one['consumers'])}")
    print(f"  total {sum(one['bytes_per_face'] for one in absent):,} B per face with nowhere to go")

    print(f"\nLOSSY TODAY -- stored, not returned unchanged ({len(out['lossy_today'])})")
    for one in out["lossy_today"]:
        print(f"  {one['name']:<28} {one['proof'][0]}")

    GENERATED.mkdir(parents=True, exist_ok=True)
    target = GENERATED / "answer.json"
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True, default=str))
        handle.write("\n")
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
