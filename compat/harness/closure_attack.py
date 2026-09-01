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


def green_fixture() -> dict[str, Any]:
    # INPUTS ONLY. The ledger is derived from these by ledger.build(), never written
    # by hand: hand-written rows meant this suite tested closure's reaction to a
    # FAILED cell and never tested whether the ledger PRODUCES one.
    return ledger_attack.green_fixture()


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
            row["durable"]["read_back_sha256"] = None


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


def _write(where: Path, held: dict[str, Any]) -> None:
    for name, body in held.items():
        with (where / name).open("w", encoding="utf-8", newline="") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True, default=str))
            handle.write("\n")

    # The ledger is DERIVED here. Every mutation above perturbs an input and must
    # travel through ledger.build() to reach closure, which is what G5 asked for.
    built = ledger.build(where)
    with (where / "ledger.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(built, indent=2, sort_keys=True, default=str))
        handle.write("\n")


def _closed(where: Path) -> tuple[bool, str]:
    held = closure.conditions(where)
    broken = [f"{one.name} [{one.state}]" for one in held if not one.green]
    return not broken, ", ".join(broken[:3])


def run_all() -> tuple[list[Result], bool]:
    out: list[Result] = []
    with tempfile.TemporaryDirectory(prefix="closure_attack_") as raw:
        where = Path(raw)
        base = green_fixture()
        _write(where, base)
        control, why = _closed(where)
        if not control:
            print(f"the green control did not close: {why}")
            return [], False

        for name, mutate in MUTATIONS:
            held = copy.deepcopy(base)
            mutate(held)
            _write(where, held)
            closed_under, broke = _closed(where)

            _write(where, base)
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
    results, control = run_all()
    if not control:
        print("every mutation below would pass for the wrong reason")
        return 1

    pinned, why = allowance_within_the_pin()
    if not pinned:
        print(f"RATCHET BROKEN: {why}")
        return 1
    print(f"closure control: GREEN (ledger derived from synthetic inputs); {why}\n")
    print(f"{'mutation':<28} {'red under it':<14} {'green after revert':<20} observed")
    for one in results:
        print(f"{one.name:<28} {one.red_under_mutation!s:<14} {one.green_after_revert!s:<20} {one.detail}")

    missed = [one.name for one in results if not one.ok]
    print(f"\n{len(results)} mutations, {len(missed)} the gate did not catch: {missed or 'none'}")
    return 0 if not missed else 1


if __name__ == "__main__":
    raise SystemExit(main())
