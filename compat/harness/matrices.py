from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compat.contracts.case import settled_by_measurement

ROOT: Path = Path(__file__).resolve().parent.parent
GENERATED: Path = ROOT / "generated"


SCALES: tuple[int, ...] = (1_000, 22_000, 1_000_000)


@dataclass(frozen=True)
class Evidence:
    cases: dict[str, Any]
    provenance: dict[str, Any]
    producer: dict[str, Any]
    union: dict[str, Any]

    @classmethod
    def load(cls, where: Path = GENERATED) -> Evidence:
        def read(name: str) -> dict[str, Any]:
            path = where / name
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} is absent. The matrices are a VIEW over evidence and cannot be produced without it: "
                    f"run `just compat run` first."
                )
            return json.loads(path.read_text(encoding="utf-8"))

        def read_optional(name: str) -> dict[str, Any]:
            path = where / name
            return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

        return cls(
            read("cases.json"),
            read("provenance.json"),
            read("producer_inventory.json"),
            read_optional("producer_union.json"),
        )


def swap_of(ablation: dict[str, Any]) -> str:
    held = ablation.get("swap")
    if not held:
        raise KeyError(
            f"the substitution of {ablation.get('primitive')!r} carries no `swap`. "
            f"This evidence predates `Ablation.swap`; re-run `just compat cases`."
        )
    return str(held)


def consumer_rows(evidence: Evidence) -> list[dict[str, Any]]:
    declared: list[str] = evidence.cases["population"]["declared"]
    by_consumer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in evidence.cases["results"]:
        by_consumer[result["consumer_id"]].append(result)

    pins = {
        row["key"].split(":", 1)[1].split("@", 1)[0]: row
        for row in evidence.provenance["repos"]
        if row["key"].startswith("consumer:")
    }

    rows: list[dict[str, Any]] = []
    for consumer in sorted(declared):
        results = [one for one in by_consumer.get(consumer, []) if one["tier"] == "consumer"]
        primitives = [one for one in by_consumer.get(consumer, []) if one["tier"] == "primitive"]
        verdicts = Counter(one["verdict"] for one in results)
        pin = pins.get(consumer, {})

        necessary: list[str] = []
        derivable: list[str] = []
        untested: list[str] = []
        substitutions: list[dict[str, Any]] = []
        for result in results:
            for ablation in result["ablations"]:
                if ablation.get("kind") == "substitution":
                    key = (ablation["primitive"], swap_of(ablation))
                    if not any((one["primitive"], one["swap"]) == key for one in substitutions):
                        substitutions.append(
                            {
                                "primitive": ablation["primitive"],
                                "swap": swap_of(ablation),
                                "serves": ablation["observed_break"] is False,
                            }
                        )
                    continue

                if ablation["observed_break"] is None:
                    if ablation["primitive"] not in untested:
                        untested.append(ablation["primitive"])
                    continue
                target = necessary if ablation["observed_break"] else derivable
                if ablation["primitive"] not in target:
                    target.append(ablation["primitive"])

        rows.append(
            {
                "consumer": consumer,
                "repo": pin.get("repo", ""),
                "commit": pin.get("pinned_commit", ""),
                "at_pin": pin.get("at_pin", False),
                "consumer_cases": len(results),
                "primitive_cases": len(primitives),
                "verdicts": dict(verdicts),
                "status": _status(results, verdicts),
                "necessary": sorted(necessary),
                "derivable_or_unused": sorted(one for one in derivable if one not in necessary),
                "untested": sorted(one for one in untested if one not in necessary and one not in derivable),
                "substitutions": sorted(substitutions, key=lambda one: (one["primitive"], one["swap"])),
            }
        )
    return rows


def _status(results: list[dict[str, Any]], verdicts: Counter[str]) -> str:
    if not results:
        return "NOT EXERCISED"
    if verdicts.get("FAIL"):
        return "DIVERGED"
    if verdicts.get("CONTRADICTED"):
        return "CONTRADICTED"
    return "REPRODUCED"


def unrepresented(evidence: Evidence) -> list[dict[str, Any]]:
    from compat.harness.run import blocking_failures

    unaccounted = {
        one.split(":", 1)[0] for names in blocking_failures(evidence.cases["results"]).values() for one in names
    }
    declared = set(evidence.cases["population"]["declared"])
    held: dict[str, Counter[str]] = defaultdict(Counter)
    for result in evidence.cases["results"]:
        if result["consumer_id"] in declared or result["verdict"] not in {"FAIL", "CONTRADICTED"}:
            continue

        if result["verdict"] == "CONTRADICTED" or result["case"] in unaccounted:
            held[result["consumer_id"]][result["verdict"]] += 1
    return [
        {"lane": lane, "verdicts": dict(counts), "cases": sum(counts.values())} for lane, counts in sorted(held.items())
    ]


def primitive_rows(evidence: Evidence) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for result in evidence.cases["results"]:
        for ablation in result["ablations"]:
            row = seen.setdefault(
                ablation["primitive"],
                {
                    "primitive": ablation["primitive"],
                    "breaks": 0,
                    "survives": 0,
                    "inconclusive": 0,
                    "substitute_fails": 0,
                    "substitute_serves": 0,
                    "substitute_fails_measured": 0,
                    "consumers": set(),
                    "cases": 0,
                },
            )
            row["cases"] += 1
            row["consumers"].add(result["consumer_id"])
            broke = ablation["observed_break"]
            if ablation.get("kind") == "substitution":
                if broke is not None:
                    row["substitute_fails" if broke else "substitute_serves"] += 1
                    if broke and settled_by_measurement(str(ablation.get("compare_method", ""))):
                        row["substitute_fails_measured"] += 1
                continue

            row["inconclusive" if broke is None else ("breaks" if broke else "survives")] += 1

    out: list[dict[str, Any]] = []
    for row in seen.values():
        consumers = sorted(row.pop("consumers"))
        out.append(
            {
                **row,
                "consumers": consumers,
                "verdict": _primitive_verdict(row),
            }
        )
    return sorted(out, key=lambda one: (-one["breaks"], one["primitive"]))


def _primitive_verdict(row: dict[str, Any]) -> str:
    if row["breaks"] and not row["survives"]:
        return "NECESSARY" if not row["inconclusive"] else "NECESSARY WHERE TESTED"
    if row["breaks"]:
        return "NECESSARY FOR SOME"
    if row["survives"]:
        return "NOT NECESSARY"

    if row.get("substitute_fails") and not row.get("substitute_serves"):
        if not row.get("substitute_fails_measured"):
            return "UNPROVEN"
        return "NECESSARY AT THIS WIDTH"
    if row.get("substitute_fails"):
        return "CHEAPER VALUE SERVES SOMETIMES"
    if row.get("substitute_serves"):
        return "CHEAPER VALUE SERVES"
    return "UNPROVEN"


def unproven_primitives(evidence: Evidence) -> list[str]:
    rows = primitive_rows(evidence)
    return sorted({one["primitive"] for one in rows if one["verdict"] == "UNPROVEN"})


def storage_rows(evidence: Evidence) -> list[dict[str, Any]]:
    fields: dict[str, Any] = evidence.producer["fields"]
    out: list[dict[str, Any]] = []
    for name, row in sorted(fields.items(), key=lambda one: -one[1]["bytes_per_face"]):
        out.append(
            {
                "field": name,
                "dtype": row["dtype"],
                "shapes": row["shapes"],
                "bytes_per_face": row["bytes_per_face"],
                **{f"at_{scale}": row["bytes_per_face"] * scale for scale in SCALES},
            }
        )
    return out


def build(evidence: Evidence) -> dict[str, Any]:
    consumers = consumer_rows(evidence)
    return {
        "generated_from": {
            "cases": evidence.cases["cases"],
            "runtime": evidence.cases["runtime"],
            "provenance_ok": evidence.provenance["provenance_ok"],
            "producer_heads": evidence.producer["producer"]["heads"],
            "weights": [
                {"pack": one["pack"], "file": one["file"], "sha256": one["sha256"]}
                for one in evidence.provenance.get("weights", [])
            ],
        },
        "consumers": consumers,
        "primitives": primitive_rows(evidence),
        "storage": storage_rows(evidence),
        "unrepresented": unrepresented(evidence),
        "skipped": list(evidence.cases.get("skipped", [])),
        "totals": {
            "declared": len(consumers),
            "reproduced": sum(1 for one in consumers if one["status"] == "REPRODUCED"),
            "not_exercised": sum(1 for one in consumers if one["status"] == "NOT EXERCISED"),
            "diverged": sum(1 for one in consumers if one["status"] == "DIVERGED"),
            "contradicted": sum(1 for one in consumers if one["status"] == "CONTRADICTED"),
            "unrepresented_failures": sum(one["cases"] for one in unrepresented(evidence)),
            "primitives_unproven": len(unproven_primitives(evidence)),
            "primitives_unproven_names": unproven_primitives(evidence),
            "skipped_inputs": len(evidence.cases.get("skipped", [])),
            "bytes_per_observation": sum(one["bytes_per_face"] for one in storage_rows(evidence)),
        },
    }


def as_markdown(out: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Consumer compatibility matrix",
        "",
        "GENERATED. Do not edit: rebuilt from `compat/generated/*.json` by",
        "`compat/harness/matrices.py`. Every cell traces to a case that ran.",
        "",
        f"- cases executed: **{out['generated_from']['cases']}**",
        f"- provenance: **{'PASS' if out['generated_from']['provenance_ok'] else 'FAIL'}**",
        f"- consumers reproduced: **{out['totals']['reproduced']} of {out['totals']['declared']}**",
        (f"- diverged: **{out['totals']['diverged']}** / not exercised: **{out['totals']['not_exercised']}**"),
        f"- failing cases with no row below: **{out['totals']['unrepresented_failures']}**",
        f"- inputs skipped before a case was built: **{out['totals']['skipped_inputs']}**",
        "",
        "## Consumers",
        "",
        "| consumer | commit | status | cases | necessary primitives |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in out["consumers"]:
        necessary = ", ".join(row["necessary"]) or "--"
        lines.append(
            f"| `{row['consumer']}` | `{row['commit'][:12] or '--'}` | {row['status']} "
            f"| {row['consumer_cases']} | {necessary} |"
        )

    lines += [
        "",
        "## Primitives",
        "",
        "| primitive | verdict | breaks | survives | untested | consumers |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in out["primitives"]:
        lines.append(
            f"| `{row['primitive']}` | {row['verdict']} | {row['breaks']} | {row['survives']} "
            f"| {row['inconclusive']} | {len(row['consumers'])} |"
        )

    lines += [
        "",
        "## Substitutions",
        "",
        "Not necessity claims. Each asks whether a value the store ALREADY holds",
        "can stand in for the one a consumer actually wants.",
        "",
        "| consumer | primitive | replaced by | does it serve? |",
        "| --- | --- | --- | --- |",
    ]
    for row in out["consumers"]:
        for swap in row["substitutions"]:
            lines.append(
                f"| `{row['consumer']}` | `{swap['primitive']}` | `{swap['swap']}` | "
                f"{'yes' if swap['serves'] else '**no**'} |"
            )

    if out["unrepresented"]:
        lines += [
            "",
            "## Failures with no consumer row",
            "",
            "Lanes that are not declared consumers in the manifest, so the table",
            "above has no row for them. Counted in the totals and gated in `main`.",
            "",
            "| lane | cases | verdicts |",
            "| --- | --- | --- |",
        ]
        for row in out["unrepresented"]:
            spread = ", ".join(f"{name} {count}" for name, count in sorted(row["verdicts"].items()))
            lines.append(f"| `{row['lane']}` | {row['cases']} | {spread} |")

    if out["skipped"]:
        lines += [
            "",
            "## Skipped inputs",
            "",
            "Inputs a lane declined to build a case from. Recorded so the population",
            "cannot shrink without saying so.",
            "",
            "| lane | input | reason |",
            "| --- | --- | --- |",
        ]
        for row in out["skipped"]:
            lines.append(f"| `{row['consumer_id']}` | `{row['what']}` | {row['why']} |")

    lines += [
        "",
        "## Storage per observation",
        "",
        "| field | dtype | shape | bytes | at 22k | at 1M |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in out["storage"]:
        lines.append(
            f"| `{row['field']}` | {row['dtype']} | {','.join(row['shapes'])} | {row['bytes_per_face']:,} "
            f"| {row['at_22000']:,} | {row['at_1000000']:,} |"
        )
    total = out["totals"]["bytes_per_observation"]
    lines.append(f"| **total** | | | **{total:,}** | **{total * 22_000:,}** | **{total * 1_000_000:,}** |")
    return "\n".join(lines) + "\n"


def report(out: dict[str, Any]) -> None:
    print(f"{'consumer':<24} {'status':<15} {'cases':>6}  necessary primitives")
    for row in out["consumers"]:
        print(f"{row['consumer']:<24} {row['status']:<15} {row['consumer_cases']:>6}  {', '.join(row['necessary'])}")
    print(f"\n{'primitive -> swap':<48} {'consumer':<24} serves?")
    for row in out["consumers"]:
        for swap in row["substitutions"]:
            pair = f"{swap['primitive']} -> {swap['swap']}"
            print(f"{pair:<48} {row['consumer']:<24} {'yes' if swap['serves'] else 'NO'}")
    print(f"\n{'primitive (removals only)':<32} {'verdict':<24} {'breaks':>7} {'survives':>9} {'untested':>9}")
    for row in out["primitives"]:
        print(
            f"{row['primitive']:<32} {row['verdict']:<24} "
            f"{row['breaks']:>7} {row['survives']:>9} {row['inconclusive']:>9}"
        )
    totals = out["totals"]
    print(
        f"\nreproduced {totals['reproduced']}/{totals['declared']}   "
        f"not exercised {totals['not_exercised']}   "
        f"{totals['bytes_per_observation']:,} B per observation"
    )


BLOCKING: tuple[str, ...] = (
    "not_exercised",
    "diverged",
    "contradicted",
    "unrepresented_failures",
    "primitives_unproven",
    "skipped_inputs",
)


def blocking(out: dict[str, Any]) -> dict[str, int]:
    totals = out["totals"]
    return {name: totals[name] for name in BLOCKING if totals.get(name)}


def main() -> int:
    evidence = Evidence.load()
    out = build(evidence)
    report(out)

    GENERATED.mkdir(parents=True, exist_ok=True)
    for name, body in (
        ("compatibility-matrix.json", json.dumps(out, indent=2, sort_keys=True, default=str) + "\n"),
        ("compatibility-matrix.md", as_markdown(out)),
    ):
        with (GENERATED / name).open("w", encoding="utf-8", newline="") as handle:
            handle.write(body)
        print(f"wrote {GENERATED / name}")

    if out["skipped"]:
        print(f"\nskipped inputs ({len(out['skipped'])}):")
        for row in out["skipped"]:
            print(f"    {row['consumer_id']:<24} {row['what'][:60]:<60} {row['why'][:60]}")

    bad = blocking(out)
    if bad:
        print("\nmatrix NOT clean: " + ", ".join(f"{name}={count}" for name, count in bad.items()))
        return 1
    print("\nmatrix clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
