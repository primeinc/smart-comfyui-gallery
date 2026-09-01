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


#: Stages whose cell comes from provenance or verdicts rather than a case's own
#: record. Every other stage must name the case-result field it reads, below.
DERIVED_ELSEWHERE: Final[frozenset[str]] = frozenset({"source_provenance", "weight_provenance", "comparison_verdict"})


@dataclass(frozen=True)
class Evidence:
    #: The case-result field this stage's cell is derived from -- distinct per stage,
    #: because one Cell aliased into six was the defect and six equal-but-distinct
    #: cells built from one boolean would defeat an object-identity check alone.
    field: str
    absent: str


STAGE_EVIDENCE: Final[dict[str, Evidence]] = {
    "producer_execution": Evidence("baseline.sha256", "no baseline artifact, so nothing records a producer running"),
    "emitted_observation": Evidence("retained_bytes", "no retained field sizes, so nothing records what was captured"),
    "durable_write": Evidence("durable.written_bytes", "no durable write recorded; the runner replays from memory"),
    "durable_read_back": Evidence("durable.read_back_sha256", "no read-back recorded; nothing was re-opened"),
    "native_reconstruction": Evidence("durable.thawed_keys", "no thaw recorded; no native record was rebuilt"),
    "consumer_replay": Evidence("replay.sha256", "no replay artifact, so the stored branch produced nothing"),
}


def stages_are_covered() -> None:
    homeless = sorted(set(STAGES) - set(STAGE_EVIDENCE) - DERIVED_ELSEWHERE)
    if homeless:
        raise KeyError(
            f"{homeless} appear in STAGES with no evidence source. A stage added here must name "
            f"the case-result field its cell derives from, or its column cannot be proven."
        )


def _at(row: dict[str, Any], dotted: str) -> Any:
    held: Any = row
    for part in dotted.split("."):
        if not isinstance(held, dict):
            return None
        held = held.get(part)
    return held


def emits(row: dict[str, Any], dotted: str) -> bool:
    # PRESENCE, not truth. A real durable write of zero bytes and a real thaw of
    # no keys are arrivals, and `if _at(...)` read both as continued absence --
    # the exact event the stage split exists to make visible.
    parts = dotted.split(".")
    held: Any = row
    for part in parts[:-1]:
        if not isinstance(held, dict):
            return False
        held = held.get(part)
    if not isinstance(held, dict) or parts[-1] not in held:
        return False
    # Present AND not null. 0 bytes written and [] keys thawed are real arrivals;
    # an explicit null is how a writer says it recorded nothing, so it is not one.
    return held[parts[-1]] is not None


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


def _read(name: str, where: Path, consumed: dict[str, str]) -> dict[str, Any] | None:
    # `consumed` is REQUIRED. Optional, an omitted argument opened a file and
    # recorded nothing, so a fourth read left the graph at three and the control
    # still passed. Omission is a TypeError now, not merely unlikely.
    path = where / name
    if not path.is_file():
        # The ATTEMPT is the fact, not the success. Recording nothing here shrank
        # the graph silently: three reads, two entries, and the condition over
        # `consumed` held green over a population that had quietly lost a member.
        consumed[name] = ""
        return None
    raw = path.read_bytes()
    consumed[name] = evidence_identity.sha256_of(raw)
    held: dict[str, Any] = json.loads(raw.decode("utf-8"))
    return held


def _current(held: dict[str, Any] | None, digest: str) -> bool:
    if held is None:
        return False
    recorded = held.get("identity")
    return isinstance(recorded, dict) and recorded.get("digest") == digest


def exercised_against_real_evidence(where: Path = GENERATED) -> dict[str, dict[str, int]]:
    """Which declared stage fields any REAL writer actually emits.

    Three of the six stages read under `durable`, and the control fixture
    invents all three: no CaseResult carries that field, so those stages
    are proven only for a shape existing nowhere outside the fixture --
    and they are precisely the three that would demonstrate a store round
    trip. Recording the split turns an artifact that overclaims into one
    naming its own limit, and makes a real durable write a visible event
    rather than a silent BLOCKED-to-VERIFIED flip.

    COUNTS, through the ledger's own `emits`. A boolean made "one case of
    302" and "all 302" the same claim, and having just replaced an
    overclaiming 6/6, a threshold of one would be a softer version of it.
    """
    shipped = where / "cases.json"
    if not shipped.is_file():
        return {stage: {"carrying": 0, "of": 0} for stage in STAGE_EVIDENCE}
    held = json.loads(shipped.read_text(encoding="utf-8"))
    rows = held.get("results") or []
    return {
        stage: {"carrying": sum(1 for row in rows if emits(row, one.field)), "of": len(rows)}
        for stage, one in STAGE_EVIDENCE.items()
    }


def declared_consumers(manifest: dict[str, Any]) -> tuple[list[str], frozenset[str]]:
    """Who gets a row, and which of those are declared first-party.

    ONE list, read by the build and by its controls. Two spellings of
    this set is how the fixture came to declare 22 while the build made
    28 rows, and the control went red on rows it had never been fed.

    The union is DECLARED. Deriving it from the evidence instead would
    make closure's row-coverage condition compare the cases against
    themselves, which is the whole defect that condition exists to catch.
    """
    first_party = frozenset(one["id"] for one in manifest.get("first_party_consumers", []))
    vendor = frozenset(one["id"] for one in manifest.get("consumers", []))
    both = sorted(vendor & first_party)
    if both:
        # The contradictory grading catches this only through a PRESENT pin row,
        # and nothing checks provenance.json's freshness, so on stale pins a
        # double-declared id grades VERIFIED. Refusing here is unconditional.
        raise KeyError(
            f"{both} are declared BOTH vendored and first-party. One says its source is fetched at a "
            f"pin, the other says this tree carries it; a consumer cannot be both."
        )
    return sorted(vendor | first_party), first_party


def source_cell(is_first_party: bool, row: dict[str, Any] | None) -> Cell:
    """G10's four states over the pair (declared first-party, pinned).

    The two halves come from different files -- the classification from
    the manifest, the pin from provenance.json -- so a consumer cannot
    satisfy this by agreeing with itself. A DECLARATION THAT CAN BE WRONG
    AND IS CHECKED, never a category granting exemption: the rejected
    alternative graded first-party consumers by tree identity, which is
    the staleness fact the BLOCKED cells already carry (two cells from
    one object, the G2 defect) and an auto-pass door for reclassifying a
    third-party consumer to dodge its pin.
    """
    if is_first_party and row is not None:
        return Cell(FAILED, "declared first-party AND pinned: one of the two records is wrong about its source")
    if is_first_party:
        return Cell(VERIFIED, "declared first-party: carried by this tree's evidence identity, no upstream to pin")
    if row is None:
        return Cell(FAILED, "neither declared first-party nor pinned: nothing records where its source comes from")
    if row.get("failures"):
        return Cell(FAILED, str(row["failures"][0])[:140])
    paths = row.get("paths", [])
    unresolved = [one for one in paths if not one.get("present")]
    if unresolved:
        return Cell(FAILED, f"{len(unresolved)} declared path(s) absent at the pin")
    if not paths:
        return Cell(FAILED, "no declared path was opened at the pin")
    return Cell(VERIFIED, f"{len(paths)} path(s) resolved at the pin")


def lane_exits(where: Path = GENERATED) -> dict[str, int]:
    from compat.harness import lanes

    return lanes.exits(where)


def build(where: Path = GENERATED, *, digest: str) -> dict[str, Any]:
    # THREADED. The caller captures ONE tree identity and hands it here, so a
    # fixture's stamps and this build cannot read two different values. Required
    # and keyword-only: omitting it is a TypeError, never a quiet recompute.
    stages_are_covered()
    manifest = provenance.load_manifest()
    # Rows built from the vendored list alone left six consumers producing
    # evidence that no row covered -- gallery_storage, the only runner doing a
    # real application-store round trip, among them.
    declared, first_party = declared_consumers(manifest)
    lanes = lane_exits(where)

    consumed: dict[str, str] = {}
    pins = _read("provenance.json", where, consumed)
    cases = _read("cases.json", where, consumed)
    _read("lanes.json", where, consumed)

    current_cases: dict[str, Any] | None = cases if _current(cases, digest) else None

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
            source_ok[who] = source_cell(who in first_party, by_consumer.get(who))

    if pins is None:
        weights_cell = Cell(BLOCKED, "pins wrote no provenance.json")
    else:
        held = pins.get("weights") or []
        bad = [one for one in held if not provenance.weight_is_verified(one)]
        if not held:
            # "0 weights VERIFIED" was a VERIFIED cell whose own reason said it
            # verified nothing. An empty population is no evidence, so it BLOCKS.
            weights_cell = Cell(BLOCKED, "provenance.json carries no weight: nothing was verified")
        elif bad:
            weights_cell = Cell(
                FAILED, f"{len(bad)} weight(s) not VERIFIED, e.g. {bad[0]['pack']}/{bad[0]['file']} {bad[0]['state']}"
            )
        else:
            weights_cell = Cell(VERIFIED, f"{len(held)} weights VERIFIED")

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

        # `.get`, not `[...]`: a case naming no consumer belongs to no row, and a
        # hard subscript made the RECORDER decide that by dying. Closure's
        # row-coverage condition is what reports it, once the ledger survives it.
        mine = [one for one in current_cases.get("results", []) if one.get("consumer_id") == who]
        if not mine:
            for stage in STAGES[2:]:
                row.cells[stage] = Cell(FAILED, "no case was executed for it")
            rows.append(row)
            continue

        for stage, evidence in STAGE_EVIDENCE.items():
            carrying = [one for one in mine if emits(one, evidence.field)]
            if not carrying:
                row.cells[stage] = Cell(BLOCKED, f"{evidence.absent} ({evidence.field} absent from all {len(mine)})")
            elif len(carrying) < len(mine):
                row.cells[stage] = Cell(
                    FAILED, f"{len(mine) - len(carrying)} of {len(mine)} case(s) record no {evidence.field}"
                )
            else:
                row.cells[stage] = Cell(VERIFIED, f"{len(carrying)} case(s) record {evidence.field}")

        failed = [one for one in mine if one["verdict"] != "PASS"]
        row.cells["comparison_verdict"] = (
            Cell(VERIFIED, f"{len(mine)} case(s) reproduced")
            if not failed
            else Cell(FAILED, f"{len(failed)} of {len(mine)}: {failed[0]['case']} {failed[0]['verdict']}")
        )
        rows.append(row)

    return {
        "identity": digest,
        "consumed": consumed,
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
    # The one capture for a real run. Everything downstream is handed this value
    # rather than reading identity() again.
    out = build(digest=str(evidence_identity.identity()["digest"]))
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
