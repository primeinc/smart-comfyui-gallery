from __future__ import annotations

import copy
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from compat.harness import closure, ledger, ledger_attack

#: This suite's OWN bound on the ratchet, deliberately not read from closure.
#: Building ALLOWANCE+1 rows from the constant made the mutation pass at every
#: value of it -- including 1037, where the condition held over 1037 unconcluded.
PINNED_ALLOWANCE: Final[int] = 497


def allowance_within_the_pin() -> tuple[bool, str]:
    held = closure.ABLATION_INCONCLUSIVE_ALLOWANCE
    return held <= PINNED_ALLOWANCE, f"allowance {held} against this suite's pin of {PINNED_ALLOWANCE}"


def _captured() -> str:
    # The single read of the tree identity for one threaded run. Every fixture
    # stamp and every ledger.build below is handed THIS value, so no two of them
    # can read identity() at different moments and disagree.
    from compat.harness import identity as evidence_identity

    return str(evidence_identity.identity()["digest"])


def totals_are_read(digest: str) -> tuple[bool, str]:
    """The ledger's four totals are read, not merely written.

    build() derives them, so no input mutation can falsify one: the only way to
    prove closure reads them is to corrupt the derived ledger on disk and require
    the verdict to notice. Three of the four had no reader at all.
    """
    with tempfile.TemporaryDirectory(prefix="closure_totals_") as raw:
        where = Path(raw)
        _write(where, green_fixture(digest=digest), digest=digest)
        before, why = _closed(where)
        if not before:
            return False, f"the control did not close before the corruption: {why}"

        path = where / "ledger.json"
        held = json.loads(path.read_text(encoding="utf-8"))
        held["totals"]["green"] = int(held["totals"]["green"]) + 1
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(held, indent=2, sort_keys=True, default=str))
            handle.write("\n")

        after = {one.name: one for one in closure.conditions(where)}
        caught = after.get("the ledger's totals agree with its rows")
        if caught is None or caught.green:
            return False, "a falsified `green` total was not noticed"
        return True, f"a falsified `green` total is caught: {caught.detail}"


def states_must_be_distinct(digest: str) -> tuple[bool, str]:
    """The allowlist is only an allowlist while VERIFIED spells its own string.

    closure imports the ledger's VERIFIED so the two cannot drift apart, which
    also means they move TOGETHER: alias it onto BLOCKED and every blocked cell
    grades green off a control that agrees with the defect.
    """
    from unittest import mock

    with tempfile.TemporaryDirectory(prefix="closure_alias_") as raw:
        where = Path(raw)
        base = green_fixture(digest=digest)
        _write(where, base, digest=digest)
        before, why = _closed(where)
        if not before:
            return False, f"the control did not close before the alias: {why}"

        # Derived UNDER the alias, so the cells on disk carry the aliased spelling
        # and are genuinely indistinguishable from blocked ones -- the defect as it
        # would really arrive, not a spelling mismatch that reddens for free.
        with mock.patch.object(ledger, "VERIFIED", ledger.BLOCKED):
            _write(where, base, digest=digest)
            under = {one.name: one for one in closure.conditions(where)}

        cells = under.get("every ledger cell VERIFIED")
        totals = under.get("the ledger's totals agree with its rows")
        if cells is None or cells.green:
            return False, "VERIFIED aliased onto BLOCKED and every cell still graded green"
        if "both spell" not in cells.detail:
            return False, f"the cell condition went red for the wrong reason: {cells.detail}"
        if totals is None or totals.green:
            return False, "green and with_blocked counted the same rows and the totals still agreed"
        # Both counts are recomputed with the aliased constant, so they still match
        # what was recorded: the disjointness is the only thing that can object, and
        # naming it here keeps the control from passing on some other objection.
        if "exceeds" not in totals.detail:
            return False, f"the totals objected for another reason: {totals.detail}"

        # And with NOTHING to check. A collision is a fact about the code, so it
        # must not depend on the cell population being non-empty -- that is how a
        # global invariant riding on a per-datum population goes quiet.
        bare = copy.deepcopy(base)
        bare["cases.json"]["results"] = []
        with mock.patch.object(ledger, "VERIFIED", ledger.BLOCKED):
            _write(where, bare, digest=digest)
            empty = {one.name: one for one in closure.conditions(where)}
        starved = empty.get("every ledger cell VERIFIED")
        if starved is None or starved.green or "both spell" not in starved.detail:
            return False, f"with no cells to check the collision went unreported: {starved and starved.detail}"

        return True, f"aliasing VERIFIED onto BLOCKED is caught: {cells.detail.split(':', 1)[-1].strip()}"


def hermetic(threaded: str) -> tuple[bool, str]:
    """The observed-form tripwire ON TOP of the threading, not the guarantee.

    Threading is what makes one identity true: every fixture stamp and every
    build in this run is handed `threaded`. This asserts the weaker, still useful
    thing -- that the snapshot the run threads has not moved underneath it, which
    is what breaks if the memo is dropped or forget() is called mid-lane.
    """
    from compat.harness import identity as evidence_identity

    before = str(evidence_identity.identity()["digest"])
    probe = Path(__file__).resolve().parent / "_hermetic_probe.py"
    try:
        probe.write_bytes(b"# transient hermeticity control\n")
        during = str(evidence_identity.identity()["digest"])
    finally:
        probe.unlink(missing_ok=True)
    after = str(evidence_identity.identity()["digest"])

    steady = before == during == after == threaded
    # Deliberately NOT "one identity across a write". This shows the snapshot
    # survived; it does not show the tree held still, and the memo is why.
    told = (
        f"the threaded snapshot {threaded[:12]} survived a mid-run write to a digested file"
        if steady
        else f"IDENTITY MOVED MID-RUN: threaded {threaded[:12]}, before {before[:12]}, during {during[:12]}"
    )
    return steady, told


def green_fixture(*, digest: str) -> dict[str, Any]:
    # INPUTS ONLY. The ledger is derived from these by ledger.build(), never written
    # by hand: hand-written rows meant this suite tested closure's reaction to a
    # FAILED cell and never tested whether the ledger PRODUCES one.
    return ledger_attack.green_fixture(digest=digest)


def _source_proof_deleted(held: dict[str, Any]) -> None:
    del held["provenance.json"]["repos"][0]


def _first(held: dict[str, Any]) -> str:
    return str(held["cases.json"]["results"][0]["consumer_id"])


def _producer_prevented(held: dict[str, Any]) -> None:
    who = _first(held)
    for row in held["cases.json"]["results"]:
        if row["consumer_id"] == who:
            row["baseline"] = None


def _stored_field_dropped(held: dict[str, Any]) -> None:
    who = _first(held)
    for row in held["cases.json"]["results"]:
        if row["consumer_id"] == who:
            # Dropped means dropped. Nulling it stopped being a suppression when
            # the ledger moved from truthiness to presence for the stage fields.
            row["durable"].pop("read_back_sha256", None)


def _attestation_removed(held: dict[str, Any]) -> None:
    weight = held["provenance.json"]["weights"][0]
    weight["attestations"] = []
    weight["state"] = "UNATTESTED"


def _attested_digest_altered(held: dict[str, Any]) -> None:
    held["provenance.json"]["weights"][0]["state"] = "MISMATCH"


def _weights_empty(held: dict[str, Any]) -> None:
    held["provenance.json"]["weights"] = []


def _weight_state_unknown(held: dict[str, Any]) -> None:
    held["provenance.json"]["weights"][0]["state"] = "SKIPPED"


def _input_skipped(held: dict[str, Any]) -> None:
    held["cases.json"]["skipped"] = [
        {"consumer_id": _first(held), "what": "one photograph", "why": "the detector found no face"}
    ]


def _consumer_ran_undeclared(held: dict[str, Any]) -> None:
    # A consumer that produced evidence while no declaration knew about it. Six
    # real ones sit in this state today, silent in every ledger cell because the
    # rows are built from the declarations rather than checked against the runner.
    held["cases.json"]["results"] = [
        *held["cases.json"]["results"],
        {"consumer_id": "a_consumer_nobody_declared", "case": "it ran anyway", "ablations": []},
    ]


def _case_names_no_consumer(held: dict[str, Any]) -> None:
    # The unattributable case. Filtering these out of the set to be checked is how
    # the row-coverage condition would have admitted them, so the shape gets its
    # own mutation rather than resting on the author's care.
    held["cases.json"]["results"] = [
        *held["cases.json"]["results"],
        {"case": "it ran, attributed to nobody", "ablations": []},
    ]


def _shard_killed(held: dict[str, Any]) -> None:
    held["cases.json"]["shards_failed"] = ["shard primitives wrote no partial, exit 1"]


def _case_diverged(held: dict[str, Any]) -> None:
    held["cases.json"]["results"][0]["verdict"] = "FAIL"


def _ablation_contradicted(held: dict[str, Any]) -> None:
    # An ablation that behaved opposite to its declaration. 497 of 1037 shipped
    # green because nothing aggregated these at all.
    held["cases.json"]["results"][0]["ablations"][0]["verdict"] = "CONTRADICTED"


def _ablations_over_allowance(held: dict[str, Any]) -> None:
    row = held["cases.json"]["results"][0]
    one = dict(row["ablations"][0], verdict="INCONCLUSIVE")
    # A FIXED count above this suite's own pin, never ALLOWANCE+1: derived from the
    # constant, this mutation went red at every value the constant could take.
    row["ablations"] = [one] * (PINNED_ALLOWANCE + 1)


def _lane_failed(held: dict[str, Any]) -> None:
    held["lanes.json"]["lanes"]["attack"] = 1


def _lane_declared_but_missing(held: dict[str, Any]) -> None:
    # A lane deleted from compat.just's run recipe. Judged against the record's own
    # keys this was invisible; judged against the declaration it is a missing lane.
    held["lanes.json"]["lanes"].pop("attack", None)


def _lane_record_empty(held: dict[str, Any]) -> None:
    held["lanes.json"]["lanes"] = {}


def _evidence_from_another_tree(held: dict[str, Any]) -> None:
    held["cases.json"]["identity"] = {"digest": "e" * 64}


MUTATIONS: Final[tuple[tuple[str, Callable[[dict[str, Any]], None]], ...]] = (
    ("attestation_removed", _attestation_removed),
    ("attested_digest_altered", _attested_digest_altered),
    ("weights_empty", _weights_empty),
    ("weight_state_unknown", _weight_state_unknown),
    ("source_proof_deleted", _source_proof_deleted),
    ("producer_prevented", _producer_prevented),
    ("stored_field_dropped", _stored_field_dropped),
    ("case_diverged", _case_diverged),
    ("ablation_contradicted", _ablation_contradicted),
    ("ablations_over_allowance", _ablations_over_allowance),
    ("input_skipped", _input_skipped),
    ("consumer_ran_undeclared", _consumer_ran_undeclared),
    ("case_names_no_consumer", _case_names_no_consumer),
    ("shard_killed", _shard_killed),
    ("lane_failed", _lane_failed),
    ("lane_declared_but_missing", _lane_declared_but_missing),
    ("lane_record_empty", _lane_record_empty),
    ("evidence_from_another_tree", _evidence_from_another_tree),
)


@dataclass
class Result:
    name: str
    red_under_mutation: bool
    green_after_revert: bool
    detail: str

    @property
    def ok(self) -> bool:
        return self.red_under_mutation and self.green_after_revert


def _write(where: Path, held: dict[str, Any], *, digest: str) -> None:
    for name, body in held.items():
        with (where / name).open("w", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True, default=str))
            handle.write("\n")

    # The ledger is DERIVED here. Every mutation above perturbs an input and must
    # travel through ledger.build() to reach closure, which is what G5 asked for.
    built = ledger.build(where, digest=digest)
    with (where / "ledger.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(built, indent=2, sort_keys=True, default=str))
        handle.write("\n")


def _closed(where: Path) -> tuple[bool, str]:
    held = closure.conditions(where)
    broken = [f"{one.name} [{one.state}]" for one in held if not one.green]
    return not broken, ", ".join(broken[:3])


def run_all(digest: str) -> tuple[list[Result], bool]:
    out: list[Result] = []
    with tempfile.TemporaryDirectory(prefix="closure_attack_") as raw:
        where = Path(raw)
        base = green_fixture(digest=digest)
        _write(where, base, digest=digest)
        control, why = _closed(where)
        if not control:
            print(f"the green control did not close: {why}")
            return [], False

        for name, mutate in MUTATIONS:
            held = copy.deepcopy(base)
            mutate(held)
            _write(where, held, digest=digest)
            closed_under, broke = _closed(where)

            _write(where, base, digest=digest)
            closed_after, _ = _closed(where)

            out.append(
                Result(
                    name=name,
                    red_under_mutation=not closed_under,
                    green_after_revert=closed_after,
                    detail=f"red on: {broke}" if not closed_under else "THE GATE STAYED GREEN",
                )
            )
    return out, True


def main() -> int:
    # ONE capture, threaded into every fixture, build and control below.
    digest = _captured()
    results, control = run_all(digest)
    if not control:
        print("every mutation below would pass for the wrong reason")
        return 1

    steady, told = hermetic(digest)
    if not steady:
        print(f"NOT HERMETIC: {told}")
        print("every result below would be order-dependent on concurrent commits")
        return 1

    read, totals_told = totals_are_read(digest)
    if not read:
        print(f"TOTALS UNREAD: {totals_told}")
        return 1

    distinct, alias_told = states_must_be_distinct(digest)
    if not distinct:
        print(f"STATES NOT DISTINCT: {alias_told}")
        return 1

    pinned, why = allowance_within_the_pin()
    if not pinned:
        print(f"RATCHET BROKEN: {why}")
        return 1
    print(f"closure control: GREEN (ledger derived from synthetic inputs); {why}\n")
    print(f"hermetic: {told}")
    print(f"totals: {totals_told}")
    print(f"states: {alias_told}")
    print()
    print(f"{'mutation':<28} {'red under it':<14} {'green after revert':<20} observed")
    for one in results:
        print(f"{one.name:<28} {one.red_under_mutation!s:<14} {one.green_after_revert!s:<20} {one.detail}")

    missed = [one.name for one in results if not one.ok]
    print(f"\n{len(results)} mutations, {len(missed)} the gate did not catch: {missed or 'none'}")
    return 0 if not missed else 1


if __name__ == "__main__":
    raise SystemExit(main())
