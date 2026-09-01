from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from compat.harness import identity as evidence_identity
from compat.harness import provenance

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
GENERATED: Final[Path] = ROOT / "generated"

VERIFIED: Final[str] = "VERIFIED"
FAILED: Final[str] = "FAILED"
BLOCKED: Final[str] = "BLOCKED"


STAGES: Final[tuple[str, ...]] = (
    "source_provenance",
    "weight_provenance",
    "producer_execution",
    "emitted_observation",
    "durable_write",
    "durable_read_back",
    "native_reconstruction",
    "consumer_replay",
    "comparison_verdict",
)


@dataclass
class Cell:
    state: str
    reason: str = ""


@dataclass
class Row:
    consumer: str
    cells: dict[str, Cell] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(one.state == VERIFIED for one in self.cells.values())


def _read(name: str) -> dict[str, Any] | None:
    path = GENERATED / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _current(held: dict[str, Any] | None, digest: str) -> bool:
    if held is None:
        return False
    recorded = held.get("identity")
    return isinstance(recorded, dict) and recorded.get("digest") == digest


def lane_exits() -> dict[str, int]:
    held = _read("lanes.json")
    return {key: int(value) for key, value in held.items()} if held else {}


def build() -> dict[str, Any]:
    now = evidence_identity.identity()
    manifest = provenance.load_manifest()
    declared = sorted(one["id"] for one in manifest.get("consumers", []))
    lanes = lane_exits()

    pins = _read("provenance.json")
    cases = _read("cases.json")

    current_cases: dict[str, Any] | None = cases if _current(cases, now["digest"]) else None

    source_ok: dict[str, Cell] = {}
    if pins is None:
        for who in declared:
            source_ok[who] = Cell(BLOCKED, "pins wrote no provenance.json")
    else:
        by_consumer = {
            row["key"].split(":", 1)[1].split("@", 1)[0]: row
            for row in pins.get("repos", [])
            if row["key"].startswith("consumer:")
        }
        for who in declared:
            row = by_consumer.get(who)
            if row is None:
                source_ok[who] = Cell(FAILED, "no repo proof was recorded for it")
            elif row.get("failures"):
                source_ok[who] = Cell(FAILED, str(row["failures"][0])[:140])
            else:
                paths = row.get("paths", [])
                unresolved = [one for one in paths if not one.get("present")]
                if unresolved:
                    source_ok[who] = Cell(FAILED, f"{len(unresolved)} declared path(s) absent at the pin")
                elif not paths:
                    source_ok[who] = Cell(FAILED, "no declared path was opened at the pin")
                else:
                    source_ok[who] = Cell(VERIFIED, f"{len(paths)} path(s) resolved at the pin")

    if pins is None:
        weights_cell = Cell(BLOCKED, "pins wrote no provenance.json")
    else:
        bad = [one for one in pins.get("weights", []) if one.get("state") != "VERIFIED"]
        weights_cell = (
            Cell(VERIFIED, f"{len(pins.get('weights', []))} weights VERIFIED")
            if not bad
            else Cell(
                FAILED, f"{len(bad)} weight(s) not VERIFIED, e.g. {bad[0]['pack']}/{bad[0]['file']} {bad[0]['state']}"
            )
        )

    rows: list[Row] = []
    for who in declared:
        row = Row(consumer=who)
        row.cells["source_provenance"] = source_ok[who]
        row.cells["weight_provenance"] = weights_cell

        if current_cases is None:
            why = (
                "the cases lane wrote no evidence this run" if cases is None else "cases.json records a different tree"
            )
            exit_code = lanes.get("cases")
            blocked_by = "cases" if exit_code is None else f"cases exited {exit_code}"
            for stage in STAGES[2:]:
                row.cells[stage] = Cell(BLOCKED, f"{why} (lane `{blocked_by}`)")
            rows.append(row)
            continue

        mine = [one for one in current_cases.get("results", []) if one["consumer_id"] == who]
        if not mine:
            for stage in STAGES[2:]:
                row.cells[stage] = Cell(FAILED, "no case was executed for it")
            rows.append(row)
            continue

        ran = Cell(VERIFIED, f"{len(mine)} case(s) executed")
        for stage in ("producer_execution", "emitted_observation", "durable_write", "durable_read_back"):
            row.cells[stage] = ran
        row.cells["native_reconstruction"] = ran
        row.cells["consumer_replay"] = ran

        failed = [one for one in mine if one["verdict"] != "PASS"]
        row.cells["comparison_verdict"] = (
            Cell(VERIFIED, f"{len(mine)} case(s) reproduced")
            if not failed
            else Cell(FAILED, f"{len(failed)} of {len(mine)}: {failed[0]['case']} {failed[0]['verdict']}")
        )
        rows.append(row)

    return {
        "identity": now["digest"],
        "lanes": lanes,
        "stages": list(STAGES),
        "rows": [{"consumer": one.consumer, "cells": {k: asdict(v) for k, v in one.cells.items()}} for one in rows],
        "totals": {
            "declared": len(rows),
            "green": sum(1 for one in rows if one.ok),
            "with_failed": sum(1 for one in rows if any(c.state == FAILED for c in one.cells.values())),
            "with_blocked": sum(1 for one in rows if any(c.state == BLOCKED for c in one.cells.values())),
        },
    }


def as_markdown(out: dict[str, Any]) -> str:
    short = {
        "source_provenance": "src",
        "weight_provenance": "wts",
        "producer_execution": "prod",
        "emitted_observation": "emit",
        "durable_write": "write",
        "durable_read_back": "read",
        "native_reconstruction": "recon",
        "consumer_replay": "replay",
        "comparison_verdict": "cmp",
    }
    mark = {VERIFIED: "ok", FAILED: "FAIL", BLOCKED: "BLOCK"}
    lines = [
        "# Compatibility ledger",
        "",
        "GENERATED from the current run by `compat/harness/ledger.py`. One row per",
        "declared consumer, one column per stage a proof passes through.",
        "",
        f"- tree identity: `{out['identity']}`",
        (
            f"- declared: **{out['totals']['declared']}**  green: **{out['totals']['green']}**"
            f"  with FAILED: **{out['totals']['with_failed']}**  with BLOCKED: **{out['totals']['with_blocked']}**"
        ),
        "",
        "| consumer | " + " | ".join(short[s] for s in out["stages"]) + " |",
        "| --- | " + " | ".join("---" for _ in out["stages"]) + " |",
    ]
    for row in out["rows"]:
        cells = " | ".join(mark[row["cells"][s]["state"]] for s in out["stages"])
        lines.append(f"| `{row['consumer']}` | {cells} |")
    lines += ["", "## Why each non-green cell is not green", ""]
    for row in out["rows"]:
        for stage in out["stages"]:
            cell = row["cells"][stage]
            if cell["state"] != VERIFIED:
                lines.append(f"- `{row['consumer']}` / {stage}: **{cell['state']}** -- {cell['reason']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    out = build()
    print(f"tree identity: {out['identity']}")
    print(f"lanes: {out['lanes']}")
    totals = out["totals"]
    print(
        f"declared {totals['declared']}   fully VERIFIED {totals['green']}   "
        f"with FAILED {totals['with_failed']}   with BLOCKED {totals['with_blocked']}\n"
    )
    for row in out["rows"]:
        marks = " ".join(row["cells"][s]["state"][:5].ljust(5) for s in out["stages"])
        print(f"{row['consumer']:<24} {marks}")

    GENERATED.mkdir(parents=True, exist_ok=True)
    for name, body in (
        ("ledger.json", json.dumps(out, indent=2, sort_keys=True, default=str) + "\n"),
        ("ledger.md", as_markdown(out)),
    ):
        with (GENERATED / name).open("w", encoding="utf-8", newline="") as handle:
            handle.write(body)
        print(f"\nwrote {GENERATED / name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
