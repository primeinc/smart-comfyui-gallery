from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from compat.contracts.case import settled_by_measurement
from compat.harness import provenance

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
GENERATED: Final[Path] = ROOT / "generated"


@dataclass
class Primitive:
    name: str
    verdict: str
    consumers: list[str] = field(default_factory=list)
    proof: list[str] = field(default_factory=list)
    bytes_per_face: int = 0
    note: str = ""

    measured: bool = False


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


def swap_of(ablation: dict[str, Any]) -> str:
    held = ablation.get("swap")
    if not held:
        raise KeyError(
            f"the substitution of {ablation.get('primitive')!r} carries no `swap`. "
            f"This evidence predates `Ablation.swap`; re-run `just compat cases`."
        )
    return str(held)


def retained_cost(cases: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in cases["results"]:
        for name, size in (row.get("retained_bytes") or {}).items():
            out[name] = max(out.get(name, 0), int(size))
    return out


def _quote(row: dict[str, Any], one: dict[str, Any]) -> str:
    return f"{row['case']}: {one['detail'][:60]}"


def _shape_only(proof: list[str]) -> bool:
    return bool(proof) and all(one.split(": ", 1)[-1].startswith("shape: ") for one in proof)


def necessity() -> dict[str, Primitive]:
    cases = _read("cases.json")
    union = _optional("producer_union.json").get("union", {})
    measured = retained_cost(cases)
    out: dict[str, Primitive] = {}
    broke_somewhere: dict[str, bool] = {}
    survived_somewhere: dict[str, bool] = {}
    untested_somewhere: dict[str, bool] = {}

    for row in cases["results"]:
        for one in row.get("ablations", []):
            if one.get("kind", "removal") != "removal":
                continue
            name = one["primitive"]
            held = out.setdefault(name, Primitive(name=name, verdict="UNTESTED"))
            if row["consumer_id"] not in held.consumers:
                held.consumers.append(row["consumer_id"])

            broke = one["observed_break"]
            if broke is None:
                untested_somewhere[name] = True
            else:
                broke_somewhere[name] = broke_somewhere.get(name, False) or broke
                survived_somewhere[name] = survived_somewhere.get(name, False) or not broke

            if broke and len(held.proof) < 3:
                held.proof.append(_quote(row, one))

    for name, held in out.items():
        if _shape_only(held.proof):
            held.note = (held.note + "; " if held.note else "") + (
                "every break here is a SHAPE divergence: the replay is a different size, "
                "which says nothing about whether the value carries the information"
            )

        held.bytes_per_face = int(union.get(name, {}).get("bytes", 0)) or measured.get(name, 0)

        broke = broke_somewhere.get(name, False)
        survived = survived_somewhere.get(name, False)
        untested = untested_somewhere.get(name, False)

        if broke:
            held.verdict = "NECESSARY"
            if survived:
                held.note = "removal broke some cases and not others; necessary where it broke"
        elif survived:
            held.verdict = "DERIVABLE"
        else:
            held.verdict = "UNPROVEN"
            held.note = (
                "every removal ended in MissingPrimitive: the replay indexes "
                "this key, it was not shown to need the value"
            )
        if untested and held.verdict != "UNPROVEN":
            held.note = (held.note + "; " if held.note else "") + "some ablations were INCONCLUSIVE"
        if not held.proof:
            held.proof.append(
                "every removal reproduced the baseline: derivable from what remained"
                if held.verdict == "DERIVABLE"
                else "no removal of this primitive produced a measurable difference"
            )
    return out


def substitutions() -> dict[tuple[str, str], Primitive]:
    cases = _read("cases.json")
    union = _optional("producer_union.json").get("union", {})
    measured = retained_cost(cases)
    out: dict[tuple[str, str], Primitive] = {}
    broke_somewhere: dict[tuple[str, str], bool] = {}
    survived_somewhere: dict[tuple[str, str], bool] = {}
    methods: dict[tuple[str, str], set[str]] = {}

    for row in cases["results"]:
        for one in row.get("ablations", []):
            if one.get("kind", "removal") != "substitution":
                continue
            key = (one["primitive"], swap_of(one))
            held = out.setdefault(key, Primitive(name=f"{key[0]} <- {key[1]}", verdict="UNPROVEN"))
            if row["consumer_id"] not in held.consumers:
                held.consumers.append(row["consumer_id"])
            broke = one["observed_break"]
            if broke is not None:
                broke_somewhere[key] = broke_somewhere.get(key, False) or broke
                survived_somewhere[key] = survived_somewhere.get(key, False) or not broke
            if broke:
                method = str(one.get("compare_method", ""))
                methods.setdefault(key, set()).add(method)
                held.measured = held.measured or settled_by_measurement(method)
            if broke and len(held.proof) < 3:
                held.proof.append(_quote(row, one))

    for key, held in out.items():
        if methods.get(key) and methods[key] <= {"shape"}:
            held.note = (held.note + "; " if held.note else "") + (
                "every break here is a SHAPE divergence: the substitute is a different size, "
                "which says nothing about whether it carries the information"
            )
        held.bytes_per_face = int(union.get(key[0], {}).get("bytes", 0)) or measured.get(key[0], 0)
        broke = broke_somewhere.get(key, False)
        survived = survived_somewhere.get(key, False)
        if broke:
            held.verdict = "SUBSTITUTE_FAILS"
            if survived:
                held.note = (held.note + "; " if held.note else "") + (
                    "fails on some arrangements and serves on others; see per-case evidence"
                )
        elif survived:
            held.verdict = "SUBSTITUTE_SERVES"
        else:
            held.note = (held.note + "; " if held.note else "") + (
                "no case answered: every substitution ended INCONCLUSIVE"
            )
        if not held.proof:
            held.proof.append("the substitute reproduced the baseline in every case that ran")
    return out


def store_returns() -> set[str]:
    cases = _read("cases.json")
    out: set[str] = set()
    for row in cases["results"]:
        if row["consumer_id"] != "gallery_storage":
            continue
        parts = str(row.get("baseline", {}).get("name") or "").split("|")
        if len(parts) < 2:
            continue

        size = 1
        for one in tuple(row.get("replay", {}).get("shape") or ()):
            size *= int(one)
        if size:
            out.add(parts[1])
    return out


def unretainable() -> list[Primitive]:
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
    cases = _read("cases.json")
    union = _optional("producer_union.json").get("union", {})
    worst: dict[str, float | None] = {}
    method: dict[str, str] = {}
    for row in cases["results"]:
        if row["consumer_id"] != "gallery_storage" or row["verdict"] != "FAIL":
            continue
        key = str(row.get("baseline", {}).get("name") or "").split("|")
        if len(key) < 2:
            continue
        name = key[1]
        apart = row.get("max_abs_diff")
        if apart is not None:
            worst[name] = max(worst.get(name) or 0.0, float(apart))
        elif name not in worst:
            worst[name] = None
        method[name] = row["comparison"].split(":")[0]

    out: list[Primitive] = []
    for name in sorted(method):
        apart = worst.get(name)
        how = method.get(name, "?")
        if apart is None:
            verdict, proof, note = (
                "UNMEASURED",
                f"{how}: the comparison returned before measuring a difference",
                "the divergence was not quantified; a verdict cannot rest on it",
            )
        elif how == "dtype" and apart == 0.0:
            verdict, proof, note = (
                "WIDENED",
                f"{how}: every value survived exactly; only the dtype changed",
                "returned in a wider dtype, value preserved: not a loss",
            )
        else:
            verdict, proof, note = (
                "LOSSY",
                f"{how}: worst {apart:g}",
                "stored, but not returned unchanged",
            )
        out.append(
            Primitive(
                name=name,
                verdict=verdict,
                bytes_per_face=int(union.get(name, {}).get("bytes", 0)),
                consumers=list(union.get(name, {}).get("emitted_by", [])),
                proof=[proof],
                note=note,
            )
        )
    return out


def corroboration() -> list[dict[str, Any]]:
    same_vector = 0.999
    cases = _read("cases.json")
    out: list[dict[str, Any]] = []
    for row in cases["results"]:
        held = {one["name"]: one.get("value") for one in row.get("measurements", []) if one.get("value") is not None}
        for one in row.get("ablations", []):
            if one["observed_break"] is None:
                continue
            broke = bool(one["observed_break"])

            if one.get("swap") == "stored_glintr100" and "stored_vector_agreement" in held:
                cosine = float(held["stored_vector_agreement"])
                identical = cosine >= same_vector
                out.append(
                    {
                        "case": row["case"],
                        "consumer_id": row["consumer_id"],
                        "primitive": one["primitive"],
                        "swap": swap_of(one),
                        "measurement": "stored_vector_agreement",
                        "measured": round(cosine, 6),
                        "expected_break": not identical,
                        "substitution_broke": broke,
                        "corroborated": broke is not identical,
                    }
                )

            if one.get("swap") == "order_reversed" and "reversal_observed" in held:
                observed = bool(held["reversal_observed"])
                out.append(
                    {
                        "case": row["case"],
                        "consumer_id": row["consumer_id"],
                        "primitive": one["primitive"],
                        "swap": swap_of(one),
                        "measurement": "reversal_observed",
                        "measured": held["reversal_observed"],
                        "expected_break": observed,
                        "substitution_broke": broke,
                        "corroborated": broke is observed,
                    }
                )
    return out


def declared_against_derived() -> dict[str, Any]:
    manifest = provenance.load_manifest()
    declared: dict[str, list[str]] = {
        one["id"]: sorted(set(one["retained"])) for one in manifest.get("consumers", []) if one.get("retained")
    }

    cases = _read("cases.json")
    exercised: dict[str, set[str]] = defaultdict(set)
    for row in cases["results"]:
        for one in row.get("ablations", []):
            exercised[row["consumer_id"]].add(one["primitive"])

    undeclared: dict[str, list[str]] = {}
    unexercised: dict[str, list[str]] = {}
    for consumer, names in declared.items():
        ran = exercised.get(consumer, set())

        extra = sorted(ran - set(names) - {"reference_pixels"})
        if extra:
            undeclared[consumer] = extra
        missing = sorted(set(names) - ran)
        if missing:
            unexercised[consumer] = missing

    return {
        "declared_by_consumer": declared,
        "exercised_by_consumer": {name: sorted(values) for name, values in sorted(exercised.items())},
        "exercised_but_not_declared": undeclared,
        "declared_but_not_exercised": unexercised,
        "agrees": not undeclared,
    }


def build() -> dict[str, Any]:
    ablations = necessity()
    union = _optional("producer_union.json")
    accepted = _optional("vendor_acceptance.json").get("population", {}).get("vendor_accepted", [])

    swapped = substitutions()

    by_verdict: dict[str, list[Primitive]] = defaultdict(list)
    for one in ablations.values():
        by_verdict[one.verdict].append(one)
    for one in swapped.values():
        by_verdict[one.verdict].append(one)

    fails: dict[str, list[Primitive]] = defaultdict(list)
    serves: dict[str, set[str]] = defaultdict(set)
    for (primitive, swap), one in swapped.items():
        if one.verdict == "SUBSTITUTE_FAILS":
            if one.measured:
                fails[primitive].append(one)
        elif one.verdict == "SUBSTITUTE_SERVES":
            serves[primitive].add(swap)

    by_removal = {one.name for one in by_verdict["NECESSARY"]}
    by_width = [
        Primitive(
            name=primitive,
            verdict="NECESSARY AT THIS WIDTH",
            consumers=sorted({who for row in rows for who in row.consumers}),
            proof=[line for row in rows for line in row.proof][:3],
            bytes_per_face=max(row.bytes_per_face for row in rows),
            note=(
                f"no cheaper form served: {', '.join(sorted(row.name.split(' <- ')[1] for row in rows))}"
                + (f"; but {', '.join(sorted(serves[primitive]))} did" if serves.get(primitive) else "")
            ),
        )
        for primitive, rows in sorted(fails.items())
        if primitive not in by_removal and not serves.get(primitive)
    ]

    keep = sorted(by_verdict["NECESSARY"] + by_width, key=lambda one: one.name)
    drop = sorted(by_verdict["DERIVABLE"], key=lambda one: one.name)

    settled = {one.name for one in keep}
    open_rows = sorted((one for one in by_verdict["UNPROVEN"] if one.name not in settled), key=lambda one: one.name)

    unproven = [one for one in open_rows if " <- " not in one.name]
    unanswered_swaps = [one for one in open_rows if " <- " in one.name]
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
        "declared_against_derived": declared_against_derived(),
        "unproven": [asdict(one) for one in unproven],
        "substitutions_that_could_not_answer": [asdict(one) for one in unanswered_swaps],
        "derivable": [asdict(one) for one in drop],
        "substitutes_that_fail": [asdict(one) for one in ruled_out],
        "substitutes_that_serve": [asdict(one) for one in cheaper_serves],
        "substitution_corroboration": corroboration(),
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

    print(f"\nMUST RETAIN -- no cheaper form of it served ({len(out['must_retain'])})")
    for one in out["must_retain"]:
        print(f"  {one['name']:<28} {len(one['consumers']):>2} consumers   {one['proof'][0][:78]}")

    print(f"\nUNPROVEN -- the ablation could not answer ({len(out['unproven'])})")
    for one in out["unproven"]:
        print(f"  {one['name']:<28} {one['note'][:78]}")
    if not out["unproven"]:
        print("  (none: every ablation asked a question it could answer)")

    swaps = out["substitutions_that_could_not_answer"]
    print(f"\nSUBSTITUTIONS THAT COULD NOT ANSWER ({len(swaps)})")
    for one in swaps:
        print(f"  {one['name']:<28} {one['note'][:78]}")
    if not swaps:
        print("  (none)")

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

    against = out["declared_against_derived"]
    print(f"\nMANIFEST `retained` AGAINST WHAT EACH LANE RAN -- agrees={against['agrees']}")
    if against["exercised_but_not_declared"]:
        print("  RED -- a lane retains state its manifest row does not name:")
        for consumer, names in sorted(against["exercised_but_not_declared"].items()):
            print(f"    {consumer:<24} {', '.join(names)}")
    if against["declared_but_not_exercised"]:
        print("  coverage gap -- declared and not yet exercised by any case:")
        for consumer, names in sorted(against["declared_but_not_exercised"].items()):
            print(f"    {consumer:<24} {', '.join(names)}")
    if against["agrees"] and not against["declared_but_not_exercised"]:
        print("  every declared name is exercised and every exercised name is declared")

    checked = out["substitution_corroboration"]
    disagreed = [one for one in checked if not one["corroborated"]]
    print(f"\nSUBSTITUTION CORROBORATION -- verdict against measurement ({len(checked)} checked)")
    if not checked:
        print("  (none measured)")
    for row in disagreed:
        print(
            f"  UNCORROBORATED {row['case']:<44} {row['measurement']}={row['measured']} "
            f"predicts break={row['expected_break']} but observed {row['substitution_broke']}"
        )
    if checked and not disagreed:
        print(f"  all {len(checked)} agree with the measurement beside them")

    absent = out["unretainable_today"]
    print(f"\nUNRETAINABLE TODAY -- emitted, and the store has no column ({len(absent)})")
    for one in absent:
        print(f"  {one['name']:<28} {one['bytes_per_face']:>7,} B  {', '.join(one['consumers'])}")
    print(f"  total {sum(one['bytes_per_face'] for one in absent):,} B per face with nowhere to go")

    print(f"\nROUND TRIP -- what the store gave back ({len(out['lossy_today'])})")
    for one in out["lossy_today"]:
        print(f"  {one['name']:<28} {one['verdict']:<12} {one['proof'][0]}")

    GENERATED.mkdir(parents=True, exist_ok=True)
    target = GENERATED / "answer.json"
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True, default=str))
        handle.write("\n")
    print(f"\nwrote {target}")

    uncorroborated = [one for one in out["substitution_corroboration"] if not one["corroborated"]]
    bad: list[str] = []
    if uncorroborated:
        bad.append(f"{len(uncorroborated)} substitution verdict(s) uncorroborated by measurement")

    if not out["must_retain"]:
        bad.append(
            f"MUST RETAIN is empty over {len(out['unproven'])} unproven primitive(s) and "
            f"{len(out['substitutes_that_fail'])} failed substitution(s): this lane produced no answer"
        )
    if bad:
        print("\nanswer NOT clean:")
        for one in bad:
            print(f"    {one}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
