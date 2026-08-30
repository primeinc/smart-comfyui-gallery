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

import contextlib
import json
import os
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any, Final

import proc
from compat.contracts.case import Tier, Verdict
from compat.harness import identity as evidence_identity
from compat.harness import provenance
from compat.harness import run as case_runner
from compat.harness.run import blocking_failures

ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: How long one shard may take. Real model work over the whole corpus, so it is
#: measured in half-hours rather than in the seconds a local command takes.
SHARD_SECONDS: Final[float] = 1800.0

#: One shard per group of lanes that share the model packs they load, grouped
#: by what they LOAD rather than what they mean: pairing `embedding_spaces`
#: with a lane holding no models costs nothing.
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
    code, out, err = proc.text(argv, timeout=SHARD_SECONDS, cwd=ROOT.parent)
    if code == proc.TIMED_OUT:
        # A red row, not a traceback out of the lane. The timeout existed and
        # nothing caught it, so one hung shard aborted the merge and the other
        # five shards' work was discarded with it.
        return {"shard": name, "results": [], "failed": f"shard {name} timed out after {SHARD_SECONDS}s"}
    if not where.is_file():
        return {
            "shard": name,
            "results": [],
            "failed": f"shard {name} wrote no partial, exit {code}: {(err or out).strip()[-400:]}",
        }
    held: dict[str, Any] = json.loads(where.read_text(encoding="utf-8"))
    held["exit"] = code
    held["shard"] = name
    return held


def merge(partials: list[dict[str, Any]]) -> dict[str, Any]:
    """Every shard's rows, with the population recomputed from the manifest."""
    results: list[dict[str, Any]] = []
    broken: list[str] = []
    exited: list[str] = []
    skipped: list[dict[str, Any]] = []
    considered: list[dict[str, Any]] = []
    observed: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicated: list[str] = []
    for one in partials:
        for row in one.get("results", []):
            # `Registry` enforces "one case name, one case" INSIDE a shard
            # (contracts/case.py:370-381) and cannot see across them, so two
            # shards emitting one name would let the later row decide.
            name = row["case"]
            if name in seen:
                duplicated.append(name)
                continue
            seen.add(name)
            results.append(row)
        # Each shard skips in its own process, so its ledger comes back in its
        # partial. Dropping them here would put the record back where it was.
        skipped.extend(one.get("skipped", []))
        considered.extend(one.get("considered", []))
        # Each shard resolves its own artifacts in its own process, so what it
        # opened comes back in its partial or not at all.
        observed.extend(one.get("observed", []))
        if one.get("failed"):
            broken.append(one["failed"])
        elif one.get("exit"):
            # A shard that wrote its partial and exited nonzero has real rows,
            # but whatever made it exit is not a detail the merge may drop. An
            # exit its own rows already account for is not repeated.
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
        # the evidence bytes belong to the results rather than to the
        # entrypoint that produced them.
        "results": case_runner.canonical(results),
        # DEDUPLICATED across shards, as `contracts.case.skipped` deduplicates
        # within one: `runners(only)` builds every runner before filtering, so
        # all six shards report the same undetectable photograph.
        "skipped": [
            dict(one)
            for one in sorted(
                {tuple(sorted(row.items())) for row in skipped},
                key=lambda row: tuple(str(value) for _, value in row),
            )
        ],
        "considered": [
            dict(one)
            for one in sorted(
                {tuple(sorted(row.items())) for row in considered},
                key=lambda row: tuple(str(value) for _, value in row),
            )
        ],
        # UNIONED by (loader, identity): six shards each load antelopev2, and
        # counting that six times would make a number about sharding.
        "observed": [
            dict(one)
            for one in sorted(
                {tuple(sorted(row.items())) for row in observed},
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


def _alive(pid: int) -> bool:
    """Whether that process still exists.

    A pid can be reused, so this can say "alive" of a stranger. It cannot say
    "alive" of nothing, which is the direction that matters here: the lock is
    only ever reclaimed on a definite no.
    """
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    import ctypes

    query_limited_information, still_active = 0x1000, 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(query_limited_information, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _owner(lock: Path) -> int:
    """The pid the lock names, or 0 when it names none."""
    held = ""
    with contextlib.suppress(OSError):
        held = lock.read_text(encoding="utf-8").strip()
    _, _, digits = held.partition("pid ")
    return int(digits) if digits.isdigit() else 0


@contextlib.contextmanager
def only_one_run(generated: Path) -> Generator[None]:
    """Refuse to start while another LIVE run owns the shard directory.

    Every shard writes to `generated/shards/<name>.json` under a fixed name,
    and `run_shard` unlinks that name before it starts. Two runs therefore
    interleave into one set of files and the merge reads a mixture of
    generations -- observed, not hypothesised: a second run deleted a partial
    the first had already written, and the first merged 22 consumers while the
    directory on disk held shards from both.

    `O_EXCL` is the whole mechanism: the create either wins or raises, with no
    window between the check and the claim.

    The owner's liveness is CHECKED rather than assumed. `finally` does not run
    in a killed process, so an interrupted run left a lock nothing could clear,
    and the next run skipped `cases` entirely and let every later lane read the
    previous run's evidence -- a stale-lock failure that reported itself as 154
    blocked ledger cells and a tree mismatch.
    """
    lock = generated / "shards.lock"
    while True:
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            owner = _owner(lock)
            if owner and _alive(owner):
                raise SystemExit(f"another case run holds {lock} (pid {owner}); wait for it to finish.") from None
            print(f"reclaiming {lock}: pid {owner or 'unrecorded'} is not running")
            # A losing racer finds the lock already gone or already retaken,
            # and comes back through `O_EXCL` either way.
            lock.unlink(missing_ok=True)
    try:
        os.write(handle, f"pid {os.getpid()}".encode())
        os.close(handle)
        yield
    finally:
        lock.unlink(missing_ok=True)


def main() -> int:
    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    with only_one_run(generated):
        return _run_every_shard(generated)


def _run_every_shard(generated: Path) -> int:
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

    # Beside the evidence, never inside it: `.gitignore:156` records that
    # excluding timings is what lets `just compat attack` compare two runs byte
    # for byte.
    timings = generated / "timings.json"
    with timings.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps({"runtime": out["runtime"], "seconds_by_case": seconds}, indent=2, sort_keys=True))
        handle.write("\n")
    print(f"wrote {timings}  ({len(seconds)} cases timed)")

    blocking = blocking_failures(out["results"])
    if blocking:
        print("\nBLOCKING:")
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
