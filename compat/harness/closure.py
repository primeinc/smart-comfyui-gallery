from __future__ import annotations

import json
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from compat.harness import identity as evidence_identity
from compat.harness import provenance

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
GENERATED: Final[Path] = ROOT / "generated"

HELD: Final[str] = "held"
FAILED: Final[str] = "failed"
NOT_APPLICABLE: Final[str] = "not-applicable"


#: Ablations the runner could not drive to a conclusion. A RATCHET, not a budget:
#: it may only ever be lowered. 497 of 1037 shipped green while ablation verdicts
#: reached no gate at all -- this number exists so that stops being invisible.
ABLATION_INCONCLUSIVE_ALLOWANCE: Final[int] = 497

ABLATION_INCONCLUSIVE: Final[str] = "INCONCLUSIVE"

#: The causes an INCONCLUSIVE ablation may have, each under its OWN ceiling.
#: The total is content-blind, so 497 tolerated inconclusives could become
#: 497 of something else and nothing would say so. Ratchets, lower-only.
ABLATION_INCONCLUSIVE_BY_CAUSE: Final[dict[str, int]] = {
    "retained_state_lacks_primitive": 448,
    "substitute_identical_to_retained": 45,
    "ablated_state_could_not_be_built": 4,
}

#: Known-good ablation verdicts. Anything else -- including null, absent and a
#: value nobody anticipated -- is unconcluded, because a denylist admits the rest.
ABLATION_CONCLUDED: Final[frozenset[str]] = frozenset({"PASS", "INCONCLUSIVE"})


@dataclass
class Condition:
    name: str
    state: str
    detail: str

    #: `<artifact>:<field>` that must be non-empty for this condition to mean
    #: anything. condition_audit empties exactly this and asserts the condition
    #: stops holding; a population it cannot empty fails that audit.
    population: str = ""

    @property
    def green(self) -> bool:
        return self.state == HELD

    @property
    def mark(self) -> str:
        return {HELD: "ok ", FAILED: "RED", NOT_APPLICABLE: "N/A"}[self.state]


def _read(name: str, where: Path = GENERATED) -> dict[str, Any] | None:
    path = where / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _stamp(held: dict[str, Any]) -> str:
    found = held.get("identity")
    if isinstance(found, dict):
        return str(found.get("digest", ""))
    return str(found or "")


def _over(name: str, population: str, over: Sized, offending: list[str], clean: str, *, unit: str = "") -> Condition:
    # `over` is the COLLECTION, not a count: with size passed separately the
    # declared population could name one collection while the verdict came from
    # another, and the audit then proved a set the condition never reads.
    size = len(over)
    # The empty-population rule. A condition that examined nothing has not held:
    # green must mean "checked, and clean", never "found no reason to complain".
    if size == 0:
        return Condition(name, NOT_APPLICABLE, f"{population} is empty; nothing was checked", population)
    if offending:
        # `unit` exists because one condition counts offences in a different unit
        # from its population, and the clean message said so while the failing one
        # did not -- the reader lost the disambiguation exactly when it mattered.
        scale = unit or f"of {size}"
        return Condition(name, FAILED, f"{len(offending)} {scale}: {', '.join(offending[:4])}", population)
    return Condition(name, HELD, clean, population)


def _one_tree(stamped: dict[str, str]) -> Condition:
    now = str(evidence_identity.identity()["digest"])
    wrong = [f"{name} built under {held[:12] or 'no digest'}" for name, held in sorted(stamped.items()) if held != now]
    return _over(
        "evidence from one tree",
        "generated:identity stamps",
        stamped,
        wrong,
        f"{len(stamped)} artifact(s) stamped {now[:12]}",
    )


def _every_lane_green(lanes: dict[str, Any] | None) -> Condition:
    # A lane's exit code was recorded and read by nothing: `attack` and `selftest`
    # could both exit 1 while this printed GREEN. The lane record is now evidence,
    # and its population is compat.just's declared list rather than its own keys.
    from compat.harness import lanes as lane_record

    found = (lanes or {}).get("lanes")
    held = found if isinstance(found, dict) else {}
    want = lane_record.declared()
    if not want:
        # `want or held` would fall back to judging the record by its own keys --
        # the exact defect this condition was repaired to remove. A declaration
        # that cannot be read is a failure, never a quiet reversion.
        return Condition(
            "every lane exited 0",
            FAILED,
            "compat.just's lane loop could not be parsed, so there is no declared set to judge against",
            "lanes.json:lanes",
        )

    # CLASSIFY, do not coerce. `int(code)` raised on a code nothing recognises, so
    # a hand-edited artifact got a traceback instead of a verdict and the audit
    # could not show this condition its second degenerate shape at all.
    offending: list[str] = []
    for name, code in sorted(held.items()):
        if isinstance(code, bool) or not isinstance(code, int):
            offending.append(f"{name} recorded {code!r}, which is not an exit code")
        elif code != 0:
            offending.append(f"{name} exited {code}")
    offending += [f"{name} declared but never recorded" for name in want if name not in held]
    return _over(
        "every lane exited 0",
        "lanes.json:lanes",
        want or held,
        offending,
        f"{len(held)} lane(s) recorded for {len(want)} declared, all 0",
    )


def _ablations_concluded(cases: dict[str, Any] | None) -> Condition:
    # The only mechanism proving a retained field is load-bearing, and its verdicts
    # were aggregated nowhere. CONTRADICTED gets no allowance: it means an ablation
    # behaved opposite to its declaration, which is a finding, not a shortfall.
    held = [one for row in ((cases or {}).get("results") or []) for one in (row.get("ablations") or [])]
    inconclusive = [one for one in held if one.get("verdict") == ABLATION_INCONCLUSIVE]
    # Allowlist, matching provenance.weight_is_verified. An enumerated-bad-values
    # filter admits every shape it did not think of: null, absent, "", "WOBBLE".
    unconcluded = [one for one in held if one.get("verdict") not in ABLATION_CONCLUDED]

    offending = [f"{one.get('primitive', '?')} {one.get('verdict') or 'no verdict'}" for one in unconcluded[:4]]
    if len(inconclusive) > ABLATION_INCONCLUSIVE_ALLOWANCE:
        offending.append(f"{len(inconclusive)} INCONCLUSIVE over an allowance of {ABLATION_INCONCLUSIVE_ALLOWANCE}")
    return _over(
        "every ablation concluded",
        "cases.json:ablations",
        held,
        offending,
        f"{len(held)} ablation(s), all concluded, {len(inconclusive)} inconclusive "
        f"within the allowance of {ABLATION_INCONCLUSIVE_ALLOWANCE}",
    )


def _inconclusive_causes_are_bounded(cases: dict[str, Any] | None) -> Condition:
    """WHAT the tolerated inconclusives are, not just how many.

    The count ratchet is content-blind. Reproduced by the adversary:
    swapping every `detail` to "model weights failed to load" left `every
    ablation concluded` HELD byte-identical, because a number cannot see
    what it is counting -- a mass foreign-cause event fits inside 497 as
    comfortably as the causes actually observed.

    An ALLOWLIST over a code the WRITER chose, never a phrase parsed back
    out of the prose: recovering the cause from `detail` would guard a
    spelling, which is the same defect one layer down. A cause not named
    here is an offence whatever its count, and each named one carries its
    own pin, so the composition cannot drift under a total that holds.
    """
    held = [one for row in ((cases or {}).get("results") or []) for one in (row.get("ablations") or [])]
    inconclusive = [one for one in held if one.get("verdict") == ABLATION_INCONCLUSIVE]

    seen: dict[str, int] = {}
    for one in inconclusive:
        cause = str(one.get("cause") or "")
        seen[cause] = seen.get(cause, 0) + 1

    offending = [
        f"{count} inconclusive with cause {cause or 'UNRECORDED'}, which is not an allowed cause"
        for cause, count in sorted(seen.items())
        if cause not in ABLATION_INCONCLUSIVE_BY_CAUSE
    ]
    offending += [
        f"{cause}: {count} over its pin of {ABLATION_INCONCLUSIVE_BY_CAUSE[cause]}"
        for cause, count in sorted(seen.items())
        if cause in ABLATION_INCONCLUSIVE_BY_CAUSE and count > ABLATION_INCONCLUSIVE_BY_CAUSE[cause]
    ]
    # The two ratchets must not be able to disagree: per-cause pins summing
    # above the total would tolerate, cause by cause, more than the total
    # says. A fact about the CODE, so it is checked whatever the population.
    pinned = sum(ABLATION_INCONCLUSIVE_BY_CAUSE.values())
    if pinned > ABLATION_INCONCLUSIVE_ALLOWANCE:
        said = f"the per-cause pins sum to {pinned}, over the total allowance of {ABLATION_INCONCLUSIVE_ALLOWANCE}"
        offending.append(said)

    spelled = ", ".join(f"{cause} {count}" for cause, count in sorted(seen.items()))
    return _over(
        "every inconclusive has an allowed cause",
        "cases.json:ablations",
        inconclusive,
        offending,
        f"{len(inconclusive)} inconclusive, each an allowed cause within its pin: {spelled}",
        unit=f"offence(s) over {len(seen)} cause(s) in {len(inconclusive)} inconclusive ablation(s)",
    )


def _consumption_agrees(ledger: dict[str, Any], where: Path) -> Condition:
    # One-tree compares each artifact to the TREE, never to each other, so two
    # artifacts individually current for the tree but not built from one another
    # both passed. `just compat pins` alone after a full run does exactly that.
    consumed: dict[str, str] = ledger.get("consumed") or {}
    drifted: list[str] = []
    for name, digest in sorted(consumed.items()):
        path = where / name
        now = evidence_identity.sha256_of(path.read_bytes()) if path.is_file() else ""
        if not digest:
            # An attempted read that found no file. The ledger records the attempt
            # so the miss is an offending MEMBER here rather than an absence from
            # the population, which is how the graph used to shrink unremarked.
            drifted.append(f"{name} was not there to read when the ledger was built")
        elif now != digest:
            drifted.append(f"{name} was {digest[:8]}, is {now[:8] or 'absent'}")
    return _over(
        "every consumed artifact unchanged",
        "ledger.json:consumed",
        consumed,
        drifted,
        f"{len(consumed)} artifact(s) still the bytes the ledger read",
    )


def _controls_describe_this_evidence(where: Path) -> Condition:
    """The ledger controls' recorded exercised-split, re-derived now.

    ledger_controls.json records which of the six stages any REAL writer
    emits -- the artifact that stopped the G2 control claiming six when
    three of them were only ever exercised against a shape its own
    fixture invents. NOTHING READ IT, so a run that regenerated cases.json
    without re-running the control shipped a split describing evidence
    that no longer exists, and the honest limit it was written to state
    would have quietly stopped being true.

    The reverse arrow of `every consumed artifact unchanged`: that asks
    whether the bytes moved under a reader, this asks whether a claim
    still describes the bytes. Re-deriving with the writer's own function
    is the point -- the inputs differ (numbers frozen at lane time
    against cases.json as it stands), which is exactly the drift.
    """
    from compat.harness.ledger import exercised_against_real_evidence

    population = "ledger_controls.json:exercised_against_real_evidence"
    name = "the ledger controls describe this evidence"
    held = _read("ledger_controls.json", where)
    if held is None:
        why = "no ledger_controls.json: the ledger-attack lane wrote nothing to check"
        return Condition(name, FAILED, why, population)

    recorded: dict[str, Any] = held.get("exercised_against_real_evidence") or {}
    now = exercised_against_real_evidence(where)
    # BOTH directions: walking only the re-derived stages would never look at a
    # key the artifact carries and the derivation does not, so a stage nothing
    # recognises rides along unexamined -- present-but-unclassifiable, here.
    offending = [
        f"{stage}: recorded {_split(recorded.get(stage))}, the evidence now shows {_split(now.get(stage))}"
        for stage in sorted(set(now) | set(recorded))
        if recorded.get(stage) != now.get(stage)
    ]
    return _over(
        name,
        population,
        recorded,
        offending,
        f"{len(recorded)} stage(s): the recorded split still matches cases.json",
    )


def _split(held: Any) -> str:
    if not isinstance(held, dict):
        return "nothing"
    return f"{held.get('carrying', '?')}/{held.get('of', '?')}"


def _evidence_has_a_row(ledger: dict[str, Any], cases: dict[str, Any] | None) -> Condition:
    # G10's general rule: a recorded set is checked against a DECLARED one. Every
    # other coverage check measures the population it was handed; this one asks
    # whether the right members are in it.

    # The runner names a consumer on every case; the ledger builds rows from the
    # declarations. Two producers, so unlike a set checked against itself they
    # cannot be wrong in the same direction.
    results: list[dict[str, Any]] = (cases or {}).get("results") or []
    rows = {str(row.get("consumer")) for row in ledger.get("rows") or []}
    seen = sorted({str(one.get("consumer_id")) for one in results if one.get("consumer_id")})
    offending = [f"{who} produced evidence with no ledger row" for who in seen if who not in rows]

    # A case naming no consumer cannot be covered by any row, and filtering it out
    # of the set to be checked would be this file's own denylist mistake: the
    # population is the cases, so an unattributable one is an offence, not a skip.
    nameless = [one for one in results if not one.get("consumer_id")]
    if nameless:
        offending.append(f"{len(nameless)} case(s) name no consumer, so no row can cover them")

    return _over(
        "every consumer in evidence has a ledger row",
        "cases.json:results",
        results,
        offending,
        f"{len(results)} case(s) across {len(seen)} consumer(s), each with a ledger row",
        unit=f"offence(s) over {len(seen)} consumer(s) in {len(results)} case(s)",
    )


def _totals_agree(ledger: dict[str, Any], rows: list[dict[str, Any]]) -> Condition:
    # The ledger writes four totals and closure read one of them. Recomputing the
    # other three from the rows on disk reads a record that was produced and never
    # read -- and green disjoint from the bad counts survives a state alias.
    from compat.harness.ledger import BLOCKED as CELL_BLOCKED
    from compat.harness.ledger import FAILED as CELL_FAILED
    from compat.harness.ledger import VERIFIED as CELL_VERIFIED

    recorded = ledger.get("totals") or {}
    # Row.ok is `all(... for cells.values())`, so the recomputation walks each
    # row's own cells rather than the stage list; a differing basis would report
    # a mismatch the ledger never made.
    states = [[(one or {}).get("state") for one in (row.get("cells") or {}).values()] for row in rows]
    again = {
        "declared": len(rows),
        "green": sum(1 for one in states if all(state == CELL_VERIFIED for state in one)),
        "with_failed": sum(1 for one in states if any(state == CELL_FAILED for state in one)),
        "with_blocked": sum(1 for one in states if any(state == CELL_BLOCKED for state in one)),
    }
    offending = [
        f"{name}: recorded {recorded.get(name)}, the rows say {value}"
        for name, value in again.items()
        if recorded.get(name) != value
    ]
    # A row counted fully green cannot also carry a failed or blocked cell. This
    # holds whatever the states spell, so aliasing VERIFIED onto a bad state
    # cannot buy the sums back.
    offending += [
        f"green {again['green']} + {name} {again[name]} exceeds the {again['declared']} declared: a row cannot be both"
        for name in ("with_failed", "with_blocked")
        if again["green"] + again[name] > again["declared"]
    ]
    return _over(
        "the ledger's totals agree with its rows",
        "ledger.json:rows",
        rows,
        offending,
        f"{len(rows)} row(s): all four totals recomputed, green disjoint from failed and blocked",
    )


def conditions(where: Path = GENERATED) -> list[Condition]:
    out: list[Condition] = []
    ledger = _read("ledger.json", where)
    pins = _read("provenance.json", where)
    cases = _read("cases.json", where)
    lanes = _read("lanes.json", where)

    if ledger is None:
        return [Condition("ledger present", FAILED, "no ledger.json: the ledger lane did not run", "generated:ledger")]

    stamped: dict[str, str] = {"ledger.json": _stamp(ledger)}
    for name, held in (("cases.json", cases), ("provenance.json", pins), ("lanes.json", lanes)):
        if held is not None:
            stamped[name] = _stamp(held)
    out.append(_one_tree(stamped))
    out.append(_every_lane_green(lanes))
    out.append(_consumption_agrees(ledger, where))
    out.append(_controls_describe_this_evidence(where))

    rows: list[dict[str, Any]] = ledger.get("rows") or []
    declared = int(ledger.get("totals", {}).get("declared", 0))
    out.append(
        _over(
            "every declared member accounted for",
            "ledger.json:rows",
            rows,
            [] if len(rows) == declared else [f"{len(rows)} row(s) for {declared} declared"],
            f"{len(rows)} row(s) for {declared} declared",
        )
    )

    weights: list[dict[str, Any]] = (pins or {}).get("weights") or []
    unverified = [
        f"{one.get('pack', '?')}/{one.get('file', '?')} {one.get('state') or 'no state'}"
        for one in weights
        if not provenance.weight_is_verified(one)
    ]
    out.append(
        _over(
            "every weight VERIFIED",
            "provenance.json:weights",
            weights,
            unverified,
            f"{len(weights)} weight(s) VERIFIED",
        )
    )

    # The ledger's own VERIFIED, imported rather than re-spelled: two definitions
    # of the good state is how the closure/ledger predicates diverged in E10.
    from compat.harness.ledger import BLOCKED as CELL_BLOCKED
    from compat.harness.ledger import FAILED as CELL_FAILED
    from compat.harness.ledger import VERIFIED as CELL_VERIFIED

    # Importing the good state closes one divergence and opens another: were
    # VERIFIED ever to spell a bad state's string, ledger and closure would move
    # together and blocked cells would grade green. An allowlist needs distinctness.
    spelling = {"VERIFIED": CELL_VERIFIED, "FAILED": CELL_FAILED, "BLOCKED": CELL_BLOCKED}
    collided = [
        f"{a} and {b} both spell {spelling[a]!r}"
        for a, b in (("VERIFIED", "FAILED"), ("VERIFIED", "BLOCKED"), ("FAILED", "BLOCKED"))
        if spelling[a] == spelling[b]
    ]

    stages: list[str] = ledger.get("stages") or []
    # ONE allowlist, not two denylists. `== "BLOCKED"` and `== "FAILED"` both
    # admitted a cell carrying a state nothing recognises; the same polarity fix
    # weights took in G4 and ablations took in G6r2, applied to the third site.
    cells = [(row["consumer"], stage, row.get("cells", {}).get(stage, {})) for row in rows for stage in stages]
    unverified = [
        f"{who}/{stage} {(cell.get('state') or 'no state')}"
        for who, stage, cell in cells
        if cell.get("state") != CELL_VERIFIED
    ]
    out.append(
        # A collision is a fact about the CODE, so it cannot be left to ride on a
        # per-datum population: with no cells to check, `_over` returns
        # not-applicable and two constants spelling one string go unreported.
        Condition(
            "every ledger cell VERIFIED",
            FAILED,
            f"the state vocabulary is not a partition: {', '.join(collided)}",
            "ledger.json:rows",
        )
        if collided
        else _over(
            "every ledger cell VERIFIED",
            "ledger.json:rows",
            cells,
            unverified,
            f"{len(cells)} cell(s), all VERIFIED",
        )
    )
    out.append(_totals_agree(ledger, rows))
    out.append(_evidence_has_a_row(ledger, cases))

    results: list[dict[str, Any]] = (cases or {}).get("results") or []
    skipped = [str(one) for one in ((cases or {}).get("skipped") or [])]
    # EVERYTHING CONSIDERED, not only what succeeded: declared over `results`
    # these two inverted their severity -- 22 results beside 10 skipped went
    # RED, 0 results beside the same 10 went not-applicable.
    out.append(
        _over(
            "no skipped input",
            "cases.json:results+skipped",
            [*results, *skipped],
            skipped,
            f"{len(results) + len(skipped)} input(s) considered, none skipped",
        )
    )

    out.append(_ablations_concluded(cases))
    out.append(_inconclusive_causes_are_bounded(cases))

    shards = [str(one) for one in ((cases or {}).get("shards_failed") or [])]
    out.append(
        _over(
            "no shard failed",
            "cases.json:results+shards_failed",
            [*results, *shards],
            shards,
            f"{len(results) + len(shards)} shard outcome(s), none failed",
        )
    )
    return out


def main() -> int:
    held = conditions()
    print("closure conditions\n")
    for one in held:
        print(f"{one.mark} {one.name:<38} {one.detail}")

    closed = all(one.green for one in held)
    print(f"\nCLOSURE: {'GREEN' if closed else 'RED'}")
    if not closed:
        print("the missing-work ledger is compat/generated/ledger.md")

    # STAMPED, like every artifact it judges. Carrying no identity, a
    # closure.json left by an earlier run read as this run's answer; `run` now
    # removes it before the lanes, so a failed run leaves none instead.
    stamp = str(evidence_identity.identity()["digest"])
    with (GENERATED / "closure.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            json.dumps(
                {
                    "identity": stamp,
                    "closed": closed,
                    "conditions": [
                        {
                            "name": o.name,
                            "state": o.state,
                            "green": o.green,
                            "detail": o.detail,
                            "population": o.population,
                        }
                        for o in held
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        handle.write("\n")
    return 0 if closed else 1


if __name__ == "__main__":
    raise SystemExit(main())
