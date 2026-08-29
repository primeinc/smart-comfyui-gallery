"""Run the population in separate processes, then merge one evidence file.

Sixteen lanes now hold model packs at once -- antelopev2 and buffalo_l
(glintr100 260 MB, w600k_r50 174 MB, 1k3d68 143 MB each), facexlib's
retinaface, arcface and bisenet, torch, mediapipe -- plus every decoded
corpus frame, memoised, at up to 96 MB apiece. A single process holding all
of it died during model loading with no traceback and exit 1.

Shrinking the memo would undo the caching that took this suite from
recomputing everything to computing each answer once. Dropping lanes would
shrink the population, which is the one thing the suite must never do. So the
lanes are split across processes instead: each shard loads only what its own
runners need, writes a partial, and this merges them.

WHAT MERGING MAY AND MAY NOT DO
-------------------------------
It concatenates case results and recomputes the population from the FROZEN
manifest, never from what happened to run. A shard that crashes therefore
shows up as consumers missing from `consumer_tier_covered` and lands in
`unexercised` -- red -- rather than as a smaller green population. A merge
that could hide a dead shard would be worse than the memory problem it
solves.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

from compat.contracts.case import Tier, Verdict
from compat.harness import identity as evidence_identity
from compat.harness import provenance
from compat.harness import run as case_runner
from compat.harness.run import blocking_failures

ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: One shard per group of lanes that share the model packs they load. Grouped
#: by what they LOAD, not by what they mean: `embedding_spaces` holds three
#: recognition models and `union_storage` holds none, so pairing them costs
#: nothing while pairing two model-heavy lanes is what broke.
SHARDS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (
        "face_family",
        (
            "instantid",
            "ipadapter_faceid",
            "ipadapter_faceid_plus",
            "pulid_comfyui",
            "reactor",
            "infiniteyou",
            "uniportrait",
            "photomaker_v2",
            "instantid_upstream",
            "ipadapter_upstream",
            "pulid_upstream",
        ),
    ),
    ("whole_reference", ("uso", "umo", "uno", "omnigen2", "instantcharacter", "qwen_image_edit_2509", "xverse")),
    ("masked_and_media", ("anystory", "id_lora", "id_v2v")),
    ("consisid", ("consisid",)),
    ("primitives", ("aligned_crop", "insightface_producer", "face_selection", "reference_sets")),
    ("storage", ("gallery_storage", "embedding_spaces")),
)


def shard_names() -> tuple[str, ...]:
    return tuple(name for name, _ in SHARDS)


def run_shard(name: str, lanes: tuple[str, ...], where: Path) -> dict[str, Any]:
    """One shard, in its own interpreter, returning its partial evidence."""
    argv = [sys.executable, "-m", "compat.harness.run", "--json", str(where), ",".join(lanes)]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, check=False, cwd=str(ROOT.parent), timeout=1800)
    except subprocess.TimeoutExpired:
        # A red row, not a traceback out of the lane. The timeout existed and
        # nothing caught it, so one hung shard aborted the merge and the other
        # five shards' work was discarded with it.
        return {"shard": name, "results": [], "failed": f"shard {name} timed out after 1800s"}
    if not where.is_file():
        return {
            "shard": name,
            "results": [],
            "failed": f"shard {name} wrote no partial, exit {done.returncode}: "
            f"{(done.stderr or done.stdout).strip()[-400:]}",
        }
    held: dict[str, Any] = json.loads(where.read_text(encoding="utf-8"))
    held["exit"] = done.returncode
    held["shard"] = name
    return held


def merge(partials: list[dict[str, Any]]) -> dict[str, Any]:
    """Every shard's rows, with the population recomputed from the manifest."""
    results: list[dict[str, Any]] = []
    broken: list[str] = []
    exited: list[str] = []
    skipped: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    duplicated: list[str] = []
    for one in partials:
        for row in one.get("results", []):
            # `Registry` enforces "one case name, one case" INSIDE a shard
            # (contracts/case.py:370-381) and cannot see across them. Six
            # processes each keeping their own register is six registers, so
            # two shards emitting the same case name went unnoticed and the
            # later row silently decided the verdict.
            name = row["case"]
            if name in seen:
                duplicated.append(name)
                continue
            seen[name] = row["consumer_id"]
            results.append(row)
        # Each shard skips in its own process, so its ledger comes back in its
        # partial. Dropping them here would put the record back where it was.
        skipped.extend(one.get("skipped", []))
        if one.get("failed"):
            broken.append(one["failed"])
        elif one.get("exit"):
            # A shard that WROTE its partial and then exited nonzero was
            # accepted in silence: `run_shard` recorded the code and nothing
            # read it. Its rows are real, but whatever made it exit is not a
            # detail the merge may drop.
            #
            # An exit the shard's OWN rows already account for is not repeated:
            # reporting it said "1 shard failed" with no name attached, beside
            # a verdict table that had already said the same thing.
            explains = blocking_failures(one.get("results", []))
            if explains or any(row["verdict"] == Verdict.CONTRADICTED.value for row in one.get("results", [])):
                broken.append(f"shard {one.get('shard', '?')} exited {one['exit']}: {sorted(explains)}")
            else:
                exited.append(f"shard {one.get('shard', '?')} exited {one['exit']} over declared findings")

    manifest = provenance.load_manifest()
    declared = {one["id"] for one in manifest.get("consumers", [])}
    at_tier = {
        tier: {one["consumer_id"] for one in results if one["tier"] == tier}
        for tier in (Tier.PRIMITIVE.value, Tier.CONSUMER.value)
    }
    covered = at_tier[Tier.CONSUMER.value]

    out: dict[str, Any] = {
        "runtime": provenance.runtime_identity(),
        "identity": evidence_identity.identity(),
        "cases": len(results),
        # Sorted through the SAME helper the single-process executor uses, so
        # the evidence bytes are a property of the results rather than of
        # which entrypoint produced them. Without this the sharded file and an
        # in-process re-run differ by ordering alone, and
        # `attack.evidence_not_reproducible` reads that as the pipeline being
        # non-deterministic.
        "results": case_runner.canonical(results),
        # DEDUPLICATED across shards, exactly as `contracts.case.skipped`
        # deduplicates within one. `runners(only)` builds every runner before
        # filtering, so all six shards construct `FaceSelectionRunner` and all
        # six report the same undetectable photograph. Six copies of one fact
        # is not six facts, and concatenating them made this file disagree
        # with a single-process rebuild that holds one.
        "skipped": [
            dict(one)
            for one in sorted(
                {tuple(sorted(row.items())) for row in skipped},
                key=lambda row: tuple(str(value) for _, value in row),
            )
        ],
        "duplicated_cases": sorted(set(duplicated)),
        "shards_failed": broken,
        # Recorded, never blocking: a shard that exited over the very
        # findings this evidence publishes is not a second failure.
        "shards_exited_over_findings": exited,
        "population": {
            "declared": sorted(declared),
            "consumer_tier_covered": sorted(covered & declared),
            "primitive_tier_only": sorted(at_tier[Tier.PRIMITIVE.value] - covered),
            "unexercised": sorted(declared - covered),
        },
        "verdicts": {one.value: sum(1 for row in results if row["verdict"] == one.value) for one in Verdict},
    }
    return case_runner.evidence_shape(out, "sharded.merge")


def main() -> int:
    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    shards = generated / "shards"
    shards.mkdir(exist_ok=True)

    partials: list[dict[str, Any]] = []
    seconds: dict[str, float] = {}
    for name, lanes in SHARDS:
        target = shards / f"{name}.json"
        target.unlink(missing_ok=True)
        held = run_shard(name, lanes, target)
        mark = "!! " if held.get("failed") else "ok "
        print(f"{mark}{name:<18} {len(held.get('results', [])):>4} cases  {held.get('failed', '')[:90]}")
        # Beside the partial, not inside it: the partials are evidence and must
        # serialise identically across two runs, the timings are a fact about
        # the machine. `run.main` writes `<partial>.timings.json`.
        beside = target.with_suffix(".timings.json")
        if beside.is_file():
            seconds.update(json.loads(beside.read_text(encoding="utf-8")))
        partials.append(held)

    out = merge(partials)
    target = generated / "cases.json"
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True, default=str))
        handle.write("\n")

    print(f"\nverdicts: {out['verdicts']}")
    pop = out["population"]
    print(f"covered at CONSUMER tier: {len(pop['consumer_tier_covered'])} of {len(pop['declared'])}")
    print(f"NOT exercised           : {len(pop['unexercised'])}  {pop['unexercised']}")
    print(f"shards failed           : {len(out['shards_failed'])}  {out['shards_failed']}")
    print(f"shards exited on findings: {len(out['shards_exited_over_findings'])}")
    print(f"duplicated case names   : {len(out['duplicated_cases'])}  {out['duplicated_cases']}")
    print(f"wrote {target}")

    # Beside the evidence, never inside it. `.gitignore:156` says excluding
    # timings is what lets `just compat attack` assert two runs produce
    # byte-identical evidence -- but this lane, the one that actually runs the
    # cases, never wrote the file at all, so the wall clock it strips was
    # simply lost.
    timings = generated / "timings.json"
    with timings.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps({"runtime": out["runtime"], "seconds_by_case": seconds}, indent=2, sort_keys=True))
        handle.write("\n")
    print(f"wrote {timings}  ({len(seconds)} cases timed)")

    blocking = blocking_failures(out["results"])
    if blocking:
        print("\nDIVERGED -- the store did not give back what the producer emitted:")
        for why, names in blocking.items():
            print(f"    {why}:")
            for one in names:
                print(f"        {one}")

    # DIVERGED is the test again. `blocking_failures` lists every one of
    # them and there is no declaration that excuses any.
    clean = not blocking and out["verdicts"][Verdict.CONTRADICTED.value] == 0
    complete = not pop["unexercised"] and not out["shards_failed"] and not out["duplicated_cases"]
    print(f"\ncases: {'clean' if clean else 'NOT clean'}   population: {'complete' if complete else 'INCOMPLETE'}")
    return 0 if (clean and complete) else 1


if __name__ == "__main__":
    raise SystemExit(main())
