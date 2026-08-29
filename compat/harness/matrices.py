"""Generated views over the raw case evidence. Never hand-maintained.

Every number here is read back out of `generated/cases.json`,
`generated/provenance.json` and `generated/producer_inventory.json`. Nothing
in this module knows a fact about a consumer; if a row says a primitive is
necessary, that is because an ablation removed it and the replay broke, and
the row can be traced to the case that observed it.

That constraint is the point. A compatibility table somebody types is a
summary of what they believed when they typed it, and it keeps reading as
true after the thing it describes has changed. A generated one goes stale
loudly: it is rebuilt from the artifacts, so a case that stopped running
disappears from it and a consumer that was never exercised shows as
NOT EXERCISED rather than as a blank cell nobody notices.

Written as Markdown and JSON side by side: the JSON is what another tool
reads, the Markdown is what a person reads, and both come off the same pass
so they cannot disagree.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT: Path = Path(__file__).resolve().parent.parent
GENERATED: Path = ROOT / "generated"

#: Scales the storage view reports at. 22k is the working library this was
#: measured against; the others bracket it.
SCALES: tuple[int, ...] = (1_000, 22_000, 1_000_000)


@dataclass(frozen=True)
class Evidence:
    """Everything the generated views are allowed to read."""

    cases: dict[str, Any]
    provenance: dict[str, Any]
    producer: dict[str, Any]
    union: dict[str, Any]
    """The producer UNION. Optional: the retention section is empty without
    it rather than absent, because a matrix that silently dropped a section
    would read as "nothing to report" instead of "not measured"."""

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


def consumer_rows(evidence: Evidence) -> list[dict[str, Any]]:
    """One row per DECLARED consumer, exercised or not.

    Built from the declared population rather than from the results, so a
    consumer nothing ran cannot vanish. That is the single most important
    property of this table: a suite that lists only what it managed to run
    reports a pass rate over a population it chose after the fact.
    """
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

        # Removals answer necessity. Substitutions answer "does the cheaper
        # thing serve", which is a different question about a different value
        # -- counting them together lists `face_patch_substituted` as though
        # it were a column the database keeps.
        necessary: list[str] = []
        derivable: list[str] = []
        substitutions: list[dict[str, Any]] = []
        for result in results:
            for ablation in result["ablations"]:
                if ablation.get("kind") == "substitution":
                    if not any(one["swap"] == ablation["primitive"] for one in substitutions):
                        substitutions.append({"swap": ablation["primitive"], "serves": not ablation["observed_break"]})
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
                "substitutions": sorted(substitutions, key=lambda one: one["swap"]),
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
    if verdicts.get("UNSUPPORTED") and not verdicts.get("PASS"):
        return "UNSUPPORTED"
    return "REPRODUCED"


def primitive_rows(evidence: Evidence) -> list[dict[str, Any]]:
    """One row per retained primitive, with the cases that tested it.

    `breaks` counts ablations where removing it stopped the replay;
    `survives` counts where it did not. A primitive with zero breaks is not
    durable truth -- whatever it holds was reproducible without it.
    """
    seen: dict[str, dict[str, Any]] = {}
    for result in evidence.cases["results"]:
        for ablation in result["ablations"]:
            if ablation.get("kind") == "substitution":
                continue
            row = seen.setdefault(
                ablation["primitive"],
                {"primitive": ablation["primitive"], "breaks": 0, "survives": 0, "consumers": set(), "cases": 0},
            )
            row["cases"] += 1
            row["consumers"].add(result["consumer_id"])
            row["breaks" if ablation["observed_break"] else "survives"] += 1

    out: list[dict[str, Any]] = []
    for row in seen.values():
        consumers = sorted(row.pop("consumers"))
        out.append(
            {
                **row,
                "consumers": consumers,
                "verdict": "NECESSARY" if row["breaks"] and not row["survives"] else _mixed(row),
            }
        )
    return sorted(out, key=lambda one: (-one["breaks"], one["primitive"]))


def _mixed(row: dict[str, Any]) -> str:
    if not row["breaks"]:
        return "NOT NECESSARY"
    return "NECESSARY FOR SOME"


def storage_rows(evidence: Evidence) -> list[dict[str, Any]]:
    """Byte cost per observation, from the producer inventory."""
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
        "totals": {
            "declared": len(consumers),
            "reproduced": sum(1 for one in consumers if one["status"] == "REPRODUCED"),
            "not_exercised": sum(1 for one in consumers if one["status"] == "NOT EXERCISED"),
            "bytes_per_observation": sum(one["bytes_per_face"] for one in storage_rows(evidence)),
        },
    }


def as_markdown(out: dict[str, Any]) -> str:
    """The same content a person can read, off the same pass as the JSON."""
    lines: list[str] = [
        "# Consumer compatibility matrix",
        "",
        "GENERATED. Do not edit: rebuilt from `compat/generated/*.json` by",
        "`compat/harness/matrices.py`. Every cell traces to a case that ran.",
        "",
        f"- cases executed: **{out['generated_from']['cases']}**",
        f"- provenance: **{'PASS' if out['generated_from']['provenance_ok'] else 'FAIL'}**",
        f"- consumers reproduced: **{out['totals']['reproduced']} of {out['totals']['declared']}**",
        f"- not exercised: **{out['totals']['not_exercised']}**",
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
        "| primitive | verdict | breaks | survives | consumers |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in out["primitives"]:
        lines.append(
            f"| `{row['primitive']}` | {row['verdict']} | {row['breaks']} | {row['survives']} "
            f"| {len(row['consumers'])} |"
        )

    lines += [
        "",
        "## Substitutions",
        "",
        "Not necessity claims. Each asks whether a value the store ALREADY holds",
        "can stand in for the one a consumer actually wants.",
        "",
        "| consumer | swap | does it serve? |",
        "| --- | --- | --- |",
    ]
    for row in out["consumers"]:
        for swap in row["substitutions"]:
            lines.append(f"| `{row['consumer']}` | `{swap['swap']}` | {'yes' if swap['serves'] else '**no**'} |")

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
    print(f"\n{'substitution':<32} {'consumer':<24} serves?")
    for row in out["consumers"]:
        for swap in row["substitutions"]:
            print(f"{swap['swap']:<32} {row['consumer']:<24} {'yes' if swap['serves'] else 'NO'}")
    print(f"\n{'primitive (removals only)':<32} {'verdict':<20} {'breaks':>7} {'survives':>9}")
    for row in out["primitives"]:
        print(f"{row['primitive']:<32} {row['verdict']:<20} {row['breaks']:>7} {row['survives']:>9}")
    totals = out["totals"]
    print(
        f"\nreproduced {totals['reproduced']}/{totals['declared']}   "
        f"not exercised {totals['not_exercised']}   "
        f"{totals['bytes_per_observation']:,} B per observation"
    )


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

    # Red while the population is incomplete, exactly like the case runner.
    # A matrix that reports green over 19 of 22 is the failure this file
    # exists to prevent.
    return 0 if out["totals"]["not_exercised"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
