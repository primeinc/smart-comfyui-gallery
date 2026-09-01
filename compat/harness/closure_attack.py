from __future__ import annotations

import copy
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from compat.harness import closure
from compat.harness.ledger import STAGES

CONSUMERS: Final[tuple[str, ...]] = ("alpha", "beta")


def green_fixture() -> dict[str, Any]:
    return {
        "provenance.json": {
            "provenance_ok": True,
            "weights": [
                {
                    "pack": "antelopev2",
                    "file": "glintr100.onnx",
                    "state": "VERIFIED",
                    "sha256": "a" * 64,
                    "attestations": [
                        {
                            "source_class": "huggingface_snapshot",
                            "repo_id": "org/model",
                            "revision": "b" * 40,
                            "path": "weights/glintr100.onnx",
                            "resolved_sha256": "a" * 64,
                            "evidence": "PROVEN",
                        }
                    ],
                }
            ],
        },
        "cases.json": {
            "skipped": [],
            "shards_failed": [],
            "results": [
                {"consumer_id": who, "case": f"{who}_boundary", "verdict": "PASS", "comparison": "byte-identical"}
                for who in CONSUMERS
            ],
        },
        "lanes.json": {
            "identity": "c" * 64,
            "lanes": {"check": 0, "pins": 0, "cases": 0, "attack": 0, "selftest": 0},
        },
        "ledger.json": {
            "identity": "c" * 64,
            "stages": list(STAGES),
            "rows": [
                {"consumer": who, "cells": {stage: {"state": "VERIFIED", "reason": "ok"} for stage in STAGES}}
                for who in CONSUMERS
            ],
            "totals": {"declared": len(CONSUMERS), "green": len(CONSUMERS), "with_failed": 0, "with_blocked": 0},
        },
    }


def _attestation_removed(held: dict[str, Any]) -> None:
    weight = held["provenance.json"]["weights"][0]
    weight["attestations"] = []
    weight["state"] = "UNATTESTED"


def _attested_digest_altered(held: dict[str, Any]) -> None:
    weight = held["provenance.json"]["weights"][0]
    weight["attestations"][0]["resolved_sha256"] = "d" * 64
    weight["state"] = "MISMATCH"


def _source_proof_deleted(held: dict[str, Any]) -> None:
    held["ledger.json"]["rows"][0]["cells"]["source_provenance"] = {
        "state": "FAILED",
        "reason": "no repo proof was recorded for it",
    }


def _producer_prevented(held: dict[str, Any]) -> None:
    held["ledger.json"]["rows"][0]["cells"]["producer_execution"] = {
        "state": "BLOCKED",
        "reason": "the producer did not run (lane `cases`)",
    }


def _stored_field_dropped(held: dict[str, Any]) -> None:
    held["ledger.json"]["rows"][0]["cells"]["durable_read_back"] = {
        "state": "FAILED",
        "reason": "a stored field did not come back",
    }


def _input_skipped(held: dict[str, Any]) -> None:
    held["cases.json"]["skipped"] = [
        {"consumer_id": "alpha", "what": "one photograph", "why": "the detector found no face"}
    ]


def _shard_killed(held: dict[str, Any]) -> None:
    held["cases.json"]["shards_failed"] = ["shard primitives wrote no partial, exit 1"]


def _lane_failed(held: dict[str, Any]) -> None:
    # The B8 demonstration in miniature: the attack lane exits 1 and every other
    # artifact still says what it said. Before G1 this closed GREEN.
    held["lanes.json"]["lanes"]["attack"] = 1


def _lane_record_empty(held: dict[str, Any]) -> None:
    # A recorded-nothing run must not read as a passed-everything run.
    held["lanes.json"]["lanes"] = {}


MUTATIONS: Final[tuple[tuple[str, Callable[[dict[str, Any]], None]], ...]] = (
    ("attestation_removed", _attestation_removed),
    ("attested_digest_altered", _attested_digest_altered),
    ("source_proof_deleted", _source_proof_deleted),
    ("producer_prevented", _producer_prevented),
    ("stored_field_dropped", _stored_field_dropped),
    ("input_skipped", _input_skipped),
    ("shard_killed", _shard_killed),
    ("lane_failed", _lane_failed),
    ("lane_record_empty", _lane_record_empty),
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


def _closed(where: Path) -> tuple[bool, str]:
    held = closure.conditions(where)
    broken = [one.name for one in held if not one.held]
    return not broken, ", ".join(broken[:3])


def run_all() -> tuple[list[Result], bool]:
    out: list[Result] = []
    with tempfile.TemporaryDirectory(prefix="closure_attack_") as raw:
        where = Path(raw)
        base = green_fixture()
        _write(where, base)
        control, why = _closed(where)
        if not control:
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
        del why
    return out, True


def main() -> int:
    results, control = run_all()
    if not control:
        print("the green control did not close: every mutation below would pass for the wrong reason")
        return 1

    print("closure control: GREEN\n")
    print(f"{'mutation':<26} {'red under it':<14} {'green after revert':<20} observed")
    for one in results:
        print(f"{one.name:<26} {one.red_under_mutation!s:<14} {one.green_after_revert!s:<20} {one.detail}")

    missed = [one.name for one in results if not one.ok]
    print(f"\n{len(results)} mutations, {len(missed)} the gate did not catch: {missed or 'none'}")
    return 0 if not missed else 1


if __name__ == "__main__":
    raise SystemExit(main())
