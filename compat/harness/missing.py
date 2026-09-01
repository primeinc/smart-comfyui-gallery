from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from compat.harness import identity as evidence_identity
from compat.harness import provenance

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
GENERATED: Final[Path] = ROOT / "generated"


LANES: Final[tuple[str, ...]] = (
    "check",
    "pins",
    "citations",
    "vendor",
    "acceptance",
    "producer",
    "union",
    "cases",
    "matrices",
    "attack",
    "selftest",
    "staleness",
    "answer",
)


@dataclass
class Unresolved:
    subject: str

    kind: str

    stage: str

    reason: str


@dataclass
class Report:
    identity: str
    declared: list[str] = field(default_factory=list)
    unresolved: list[Unresolved] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unresolved


def _read(name: str) -> dict[str, Any] | None:
    path = GENERATED / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _current(held: dict[str, Any] | None, now: str) -> bool:
    if held is None:
        return False
    recorded = held.get("identity")
    return isinstance(recorded, dict) and recorded.get("digest") == now


def build() -> Report:
    now = evidence_identity.identity()
    manifest = provenance.load_manifest()
    declared = sorted(one["id"] for one in manifest.get("consumers", []))
    out = Report(identity=now["digest"], declared=declared)

    pins = _read("provenance.json")
    if pins is None:
        out.unresolved.append(Unresolved("provenance.json", "artifact", "pins", "the lane wrote no artifact"))
    elif not pins.get("provenance_ok"):
        for row in pins.get("weights", []):
            if not row.get("present"):
                out.unresolved.append(
                    Unresolved(
                        f"{row.get('pack')}/{row.get('file')}",
                        "weight",
                        "pins",
                        "the file is not on this machine",
                    )
                )
            elif not row.get("matches_published"):
                out.unresolved.append(
                    Unresolved(
                        f"{row.get('pack')}/{row.get('file')}",
                        "weight",
                        "pins",
                        "no vendor-published digest attests these bytes",
                    )
                )

    cases = _read("cases.json")
    if cases is None or not _current(cases, now["digest"]):
        why = (
            "no cases.json: the case lane never ran against this tree"
            if cases is None
            else "cases.json records a different tree; it is not evidence for this one"
        )
        for who in declared:
            out.unresolved.append(Unresolved(who, "consumer", "cases", why))
        return out

    reproduced: dict[str, bool] = {}
    for row in cases.get("results", []):
        who = row["consumer_id"]
        reproduced[who] = reproduced.get(who, True) and row["verdict"] == "PASS"
        if row["verdict"] != "PASS":
            out.unresolved.append(
                Unresolved(row["case"], "case", "cases", f"{row['verdict']}: {row.get('comparison', '')[:120]}")
            )
    for who in declared:
        if who not in reproduced:
            out.unresolved.append(Unresolved(who, "consumer", "cases", "no case was executed for it"))

    for row in cases.get("skipped", []):
        out.unresolved.append(
            Unresolved(str(row.get("what", ""))[:80], "input", "cases", str(row.get("why", ""))[:160])
        )
    for one in cases.get("shards_failed", []):
        out.unresolved.append(Unresolved(str(one)[:80], "lane", "cases", "the shard did not complete"))
    for one in cases.get("duplicated_cases", []):
        out.unresolved.append(Unresolved(str(one), "case", "cases", "two shards emitted this case name"))

    matrix = _read("compatibility-matrix.json")
    if matrix is None:
        out.unresolved.append(Unresolved("compatibility-matrix.json", "artifact", "matrices", "not generated"))
    else:
        for name in matrix.get("totals", {}).get("primitives_unproven_names", []):
            out.unresolved.append(
                Unresolved(name, "primitive", "matrices", "no ablation weighed the consumer's output")
            )

    for name, lane in (("attack.json", "attack"), ("selftest.json", "selftest")):
        held = _read(name)
        if held is None:
            out.unresolved.append(Unresolved(name, "artifact", lane, "the lane wrote no artifact"))
            continue
        for one in held.get("attacks", []):
            if not one.get("detected"):
                out.unresolved.append(
                    Unresolved(
                        str(one.get("name")), "case", lane, f"the gate did not see it: {one.get('observed', '')[:110]}"
                    )
                )
    return out


def as_markdown(report: Report) -> str:
    lines = [
        "# Missing work",
        "",
        "GENERATED from the current execution against the current tree by",
        "`compat/harness/missing.py`. Every row is a declared thing with no",
        'executed proof. There is no status here meaning "did not run but does',
        'not count".',
        "",
        f"- tree identity: `{report.identity}`",
        f"- declared consumers: **{len(report.declared)}**",
        f"- unresolved: **{len(report.unresolved)}**",
        "",
    ]
    if report.ok:
        lines += ["Nothing is unresolved.", ""]
        return "\n".join(lines)
    lines += ["| stage | kind | subject | observed |", "| --- | --- | --- | --- |"]
    for one in sorted(
        report.unresolved, key=lambda r: (LANES.index(r.stage) if r.stage in LANES else 99, r.kind, r.subject)
    ):
        subject = one.subject.replace("|", "\\|")
        reason = one.reason.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{one.stage}` | {one.kind} | `{subject}` | {reason} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    report = build()
    print(f"tree identity: {report.identity}")
    print(f"declared consumers: {len(report.declared)}")
    print(f"unresolved: {len(report.unresolved)}\n")
    for one in report.unresolved:
        print(f"{one.stage:<12} {one.kind:<10} {one.subject[:44]:<44} {one.reason[:80]}")

    GENERATED.mkdir(parents=True, exist_ok=True)
    for name, body in (
        (
            "missing.json",
            json.dumps(
                {
                    "identity": report.identity,
                    "declared": report.declared,
                    "unresolved": [asdict(o) for o in report.unresolved],
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
        ),
        ("missing.md", as_markdown(report)),
    ):
        with (GENERATED / name).open("w", encoding="utf-8", newline="") as handle:
            handle.write(body)
        print(f"\nwrote {GENERATED / name}")

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
