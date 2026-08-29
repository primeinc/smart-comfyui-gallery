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

from compat.contracts.case import settled_by_measurement

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


def swap_of(ablation: dict[str, Any]) -> str:
    """The swap a substitution names, or a readable refusal.

    NOT `.get("swap", "")`. An empty default would silently merge every
    substitution in a consumer into one row keyed on the empty string, which
    is the aggregation bug the field was added to remove. Evidence written
    before the field exists is evidence under a different contract and says so
    instead of being read as though it were current.
    """
    held = ablation.get("swap")
    if not held:
        raise KeyError(
            f"the substitution of {ablation.get('primitive')!r} carries no `swap`. "
            f"This evidence predates `Ablation.swap`; re-run `just compat cases`."
        )
    return str(held)


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

        # Removals answer necessity; substitutions answer whether the cheaper
        # thing serves. An ablation names the primitive it touches and, for a
        # substitution, the `swap` that replaced it.
        necessary: list[str] = []
        derivable: list[str] = []
        untested: list[str] = []
        substitutions: list[dict[str, Any]] = []
        for result in results:
            for ablation in result["ablations"]:
                if ablation.get("kind") == "substitution":
                    # Keyed on the PAIR: deduping on the swap alone lets case
                    # order decide `serves`, and collapses two primitives
                    # replaced by the same value into one row.
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
                # None is INCONCLUSIVE and belongs in neither column: the
                # ablation showed the runner indexes the key, not that the
                # consumer needs the value.
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
    """One consumer's standing, and never better than its worst case.

    PARTIAL exists because `id_v2v` ran 3 of its 12 cases and published as
    REPRODUCED: the old rule only said UNSUPPORTED when there was no PASS at
    all, so a single passing case promoted a consumer over nine that could not
    run. "Reproduced" has to mean every case reproduced, or the word is doing
    no work in the one table a person reads.
    """
    if not results:
        return "NOT EXERCISED"
    if verdicts.get("FAIL"):
        return "DIVERGED"
    if verdicts.get("CONTRADICTED"):
        return "CONTRADICTED"
    if verdicts.get("UNSUPPORTED"):
        return "UNSUPPORTED" if not verdicts.get("PASS") else "PARTIAL"
    return "REPRODUCED"


def unrepresented(evidence: Evidence) -> list[dict[str, Any]]:
    """Failing cases belonging to no DECLARED consumer, so no row shows them.

    `consumer_rows` iterates the manifest population, which is right -- a
    consumer nothing ran must not vanish. The cost is that a lane which is not
    a manifest consumer has no row at all, and `gallery_storage` is exactly
    that: a primitive-tier lane holding all 19 of this suite's failures. The
    matrix published "22 of 22 reproduced" beside them.

    Counted here and gated in `main`, so the table cannot read complete while
    evidence it does not display is red.

    `run.blocking_failures` is the single definition of which cases are a
    failure, and this reads it rather than keeping a second opinion.
    """
    from compat.harness.run import blocking_failures

    unaccounted = {
        one.split(":", 1)[0] for names in blocking_failures(evidence.cases["results"]).values() for one in names
    }
    declared = set(evidence.cases["population"]["declared"])
    held: dict[str, Counter[str]] = defaultdict(Counter)
    for result in evidence.cases["results"]:
        if result["consumer_id"] in declared or result["verdict"] not in {"FAIL", "CONTRADICTED"}:
            continue
        # CONTRADICTED is always unaccounted for: nothing may declare that a
        # necessity claim is allowed to be wrong.
        if result["verdict"] == "CONTRADICTED" or result["case"] in unaccounted:
            held[result["consumer_id"]][result["verdict"]] += 1
    return [
        {"lane": lane, "verdicts": dict(counts), "cases": sum(counts.values())} for lane, counts in sorted(held.items())
    ]


def primitive_rows(evidence: Evidence) -> list[dict[str, Any]]:
    """One row per retained primitive, with the cases that tested it.

    `breaks` counts ablations where removing it stopped the replay;
    `survives` counts where it did not. A primitive with zero breaks is not
    durable truth -- whatever it holds was reproducible without it.
    """
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
                # `primitive` holds the degraded value's name, not the swap's,
                # so this is evidence about THIS primitive and is the only
                # evidence most of them have.
                if broke is not None:
                    row["substitute_fails" if broke else "substitute_serves"] += 1
                    if broke and settled_by_measurement(str(ablation.get("compare_method", ""))):
                        row["substitute_fails_measured"] += 1
                continue
            # `broke is None` is INCONCLUSIVE and had to be split out: the old
            # expression put it in `survives`, because None is falsy, which
            # would have read as positive evidence the primitive is derivable.
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
    """What the ablations actually established about one primitive.

    UNPROVEN is the verdict that was missing. Every removal in this suite
    ended in `MissingPrimitive` -- the replay indexing the key it had just
    been denied -- and the old rule read that as `breaks`, so all 22 exercised
    primitives reported NECESSARY over `survives: 0`. A claim that cannot come
    out the other way is not a finding.

    Removals are read first because they answer the sharper question. When
    they answer nothing, a SUBSTITUTION still can, and for most primitives
    here it is the only evidence there is: removing the one key a replay
    indexes proves the replay indexes it, while offering the same value in the
    narrowest storable float and watching the consumer diverge proves the
    store must keep it AT THAT WIDTH. That is a stronger claim than the
    removal was ever able to make, and reading it as UNPROVEN threw it away.
    """
    if row["breaks"] and not row["survives"]:
        return "NECESSARY" if not row["inconclusive"] else "NECESSARY WHERE TESTED"
    if row["breaks"]:
        return "NECESSARY FOR SOME"
    if row["survives"]:
        return "NOT NECESSARY"
    # No removal answered. Fall through to what the substitutions established.
    if row.get("substitute_fails") and not row.get("substitute_serves"):
        # Only a substitution whose comparison weighed the consumer's output
        # establishes this; `shape`, `dtype` and an exception settle before the
        # consumer runs.
        if not row.get("substitute_fails_measured"):
            return "UNPROVEN"
        return "NECESSARY AT THIS WIDTH"
    if row.get("substitute_fails"):
        return "CHEAPER VALUE SERVES SOMETIMES"
    if row.get("substitute_serves"):
        return "CHEAPER VALUE SERVES"
    return "UNPROVEN"


def unproven_primitives(evidence: Evidence) -> list[str]:
    """Primitives no ablation established anything about.

    Every one of them reds this lane. A primitive nothing proved is an
    experiment that has not been built, and there is no declaration that
    excuses one.
    """
    rows = primitive_rows(evidence)
    return sorted({one["primitive"] for one in rows if one["verdict"] == "UNPROVEN"})


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
        # Failures with no row above. Displayed and gated, because a total
        # computed only over the rows a table happens to hold is a pass rate
        # over a population that table chose.
        "unrepresented": unrepresented(evidence),
        # Inputs no case was built from, displayed because an unreported
        # `continue` computes a pass rate over whatever survived. Not blocking:
        # the reason is recorded beside each one.
        "skipped": list(evidence.cases.get("skipped", [])),
        "totals": {
            "declared": len(consumers),
            "reproduced": sum(1 for one in consumers if one["status"] == "REPRODUCED"),
            "not_exercised": sum(1 for one in consumers if one["status"] == "NOT EXERCISED"),
            "diverged": sum(1 for one in consumers if one["status"] == "DIVERGED"),
            "contradicted": sum(1 for one in consumers if one["status"] == "CONTRADICTED"),
            "partial": sum(1 for one in consumers if one["status"] == "PARTIAL"),
            "unsupported": sum(1 for one in consumers if one["status"] == "UNSUPPORTED"),
            "unrepresented_failures": sum(one["cases"] for one in unrepresented(evidence)),
            "primitives_unproven": len(unproven_primitives(evidence)),
            "primitives_unproven_names": unproven_primitives(evidence),
            "skipped_inputs": len(evidence.cases.get("skipped", [])),
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
        (
            f"- diverged: **{out['totals']['diverged']}**"
            f" / partial: **{out['totals']['partial']}**"
            f" / unsupported: **{out['totals']['unsupported']}**"
            f" / not exercised: **{out['totals']['not_exercised']}**"
        ),
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


#: Totals that must be zero for the matrix to be clean, named once so the gate
#: and `attack.attack_positive_control` cannot disagree about what clean
#: means.
BLOCKING: tuple[str, ...] = (
    "not_exercised",
    "diverged",
    "contradicted",
    "unrepresented_failures",
    # A primitive whose every ablation was INCONCLUSIVE is published in
    # `answer.json` as durable state on no evidence at all.
    "primitives_unproven",
)


def blocking(out: dict[str, Any]) -> dict[str, int]:
    """Which blocking totals are non-zero, and by how much."""
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

    # Red on the same conditions the case runner reds on, and on failures this
    # table has no row for: a failure in a lane with no row moves no total, so
    # gating on `not_exercised` alone exits 0 over it.
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
