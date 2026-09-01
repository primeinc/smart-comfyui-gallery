from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from compat.harness import observe, reconcile

ROOT: Final[Path] = Path(__file__).resolve().parent.parent

VERDICTS: Final[tuple[str, ...]] = (reconcile.AGREED, reconcile.UNEXERCISED, reconcile.POPULATION_DEFECT)


@dataclass
class Control:
    name: str
    held: bool
    detail: str

    @property
    def mark(self) -> str:
        return "ok " if self.held else "RED"


def _population(*identities: str) -> dict[str, Any]:
    return {
        "identity": "tree",
        "edges": [
            {"consumer_id": "control", "artifact_logical_identity": one, "discovery_status": "REQUIRED"}
            for one in identities
        ],
    }


def _observed(*rows: tuple[str, str]) -> list[dict[str, Any]]:
    return [{"loader": loader, "identity": identity} for loader, identity in rows]


def _verdicts(rows: list[reconcile.Reconciled]) -> dict[str, str]:
    return {one.identity: one.verdict for one in rows}


def control_observed_but_not_static() -> Control:
    got = _verdicts(reconcile.reconcile(_population("org/known"), _observed(("open", "surprise.onnx"))))
    held = got.get("surprise.onnx") == reconcile.POPULATION_DEFECT
    return Control("A observed_but_not_static", held, f"{got}")


def control_static_but_not_observed() -> Control:
    got = _verdicts(reconcile.reconcile(_population("org/known"), []))
    return Control("B static_but_not_observed", got.get("org/known") == reconcile.UNEXERCISED, f"{got}")


def control_agreement_is_reachable() -> Control:
    got = _verdicts(reconcile.reconcile(_population("org/known"), _observed(("open", "org/known"))))
    return Control("C agreement_is_reachable", got.get("org/known") == reconcile.AGREED, f"{got}")


def control_ambiguous_stem() -> Control:
    population = _population(
        "ipadapter/models/image_encoder/model.safetensors",
        "uniportrait/models/image_encoder/model.safetensors",
    )
    rows = reconcile.reconcile(population, _observed(("open", "model.safetensors")))
    defects = [one for one in rows if one.verdict == reconcile.POPULATION_DEFECT]
    agreed = [one for one in rows if one.verdict == reconcile.AGREED]
    held = len(defects) == 1 and "more than once" in defects[0].detail and not agreed
    return Control("D ambiguous_stem", held, f"{len(defects)} ambiguity, {len(agreed)} credited by guess")


def control_path_spelling_agrees() -> Control:
    got = _verdicts(reconcile.reconcile(_population("weights/det_10g.onnx"), _observed(("onnx", "det_10g.onnx"))))
    return Control("E path_spelling_agrees", got.get("weights/det_10g.onnx") == reconcile.AGREED, f"{got}")


def control_windows_spelling_agrees() -> Control:
    observed = _observed(("open", "C:\\models\\antelopev2\\det_10g.onnx"))
    got = _verdicts(reconcile.reconcile(_population("antelopev2/det_10g.onnx"), observed))
    return Control("F windows_spelling_agrees", got.get("antelopev2/det_10g.onnx") == reconcile.AGREED, f"{got}")


def control_extension_stripped_agrees() -> Control:
    got = _verdicts(reconcile.reconcile(_population("antelopev2"), _observed(("zip", "antelopev2.zip"))))
    return Control("G extension_stripped_agrees", got.get("antelopev2") == reconcile.AGREED, f"{got}")


def control_unresolved_artifact_rows() -> Control:
    marker = "UNRESOLVED_ARTIFACT:insightface.app.FaceAnalysis"
    rows = reconcile.reconcile(_population(marker, "org/known"), _observed(("open", marker)))
    got = _verdicts(rows)
    held = got.get(marker) == reconcile.POPULATION_DEFECT and len(rows) == 2
    return Control("H unresolved_artifact_rows", held, f"{got}")


def control_unresolved_call_rows() -> Control:
    rows = reconcile.reconcile(_population("UNRESOLVED_CALL:torch.load", "org/known"), [])
    return Control("I unresolved_call_rows", [one.identity for one in rows] == ["org/known"], f"{_verdicts(rows)}")


def control_empty_identity_dropped() -> Control:
    population = {
        "identity": "tree",
        "edges": [
            {"consumer_id": "control", "artifact_logical_identity": "", "discovery_status": "REQUIRED"},
            {"consumer_id": "control", "artifact_logical_identity": "org/known", "discovery_status": "REQUIRED"},
        ],
    }
    got = _verdicts(reconcile.reconcile(population, _observed(("open", "org/known"))))
    return Control("J empty_identity_dropped", got == {"org/known": reconcile.AGREED}, f"{got}")


def control_native_marker_skipped() -> Control:
    rows = reconcile.reconcile(_population("org/known"), _observed((observe.NATIVE_UNSEEN, "_wfopen")))
    got = _verdicts(rows)
    return Control("K native_marker_skipped", "_wfopen" not in got, f"{got}")


def control_every_artifact_once() -> Control:
    rows = reconcile.reconcile(
        _population("a/one", "b/two", "c/three"),
        _observed(("open", "a/one"), ("open", "a/one"), ("open", "unknown.bin")),
    )
    identities = [one.identity for one in rows]
    totals = {one: sum(1 for row in rows if row.verdict == one) for one in VERDICTS}
    held = (
        len(identities) == len(set(identities)) and sum(totals.values()) == len(rows) and totals[reconcile.AGREED] == 1
    )
    return Control("L every_artifact_once", held, f"{len(rows)} rows, {len(set(identities))} distinct, {totals}")


def control_second_loader_is_kept() -> Control:
    rows = reconcile.reconcile(_population("org/known"), _observed(("open", "org/known"), ("onnx", "org/known")))
    loaders = next((one.loaders for one in rows if one.identity == "org/known"), [])
    return Control("M second_loader_is_kept", sorted(loaders) == ["onnx", "open"], f"loaders={loaders}")


def _tree() -> str:
    from compat.harness import identity as evidence_identity

    return str(evidence_identity.identity()["digest"])


def control_stale_population() -> Control:
    got = reconcile._one_tree({"identity": "an older tree"}, {"identity": {"digest": _tree()}})
    held = not got["agree"] and "artifact_population.json" in got["detail"]
    return Control("N stale_population", held, got["detail"][:110])


def control_fresh_population() -> Control:
    now = _tree()
    got = reconcile._one_tree({"identity": now}, {"identity": {"digest": now}})
    return Control("O fresh_population", bool(got["agree"]), got["detail"][:110])


def control_dead_run_is_not_absence(where: Path) -> Control:
    got = reconcile._observer_complete(where, {"cases": 0, "shards_failed": ["shard x died"]})
    return Control("P dead_run_is_not_absence", got.get("cases_ran") is False, f"{got.get('detail', '')[:100]}")


def control_live_run_is_absence(where: Path) -> Control:
    got = reconcile._observer_complete(where, {"cases": 302, "shards_failed": []})
    return Control("Q live_run_is_absence", got.get("cases_ran") is True, f"{got.get('detail', '')[:100]}")


def control_unmeasured_observer(where: Path) -> Control:
    got = reconcile._observer_complete(where, {"cases": 302, "shards_failed": []})
    held = got.get("known") is False and not got.get("complete")
    return Control("R unmeasured_observer", held, f"{got.get('detail', '')[:100]}")


def run_all() -> list[Control]:
    measured = ROOT / "generated"
    with tempfile.TemporaryDirectory(prefix="reconcile_attack_") as raw:
        unmeasured = Path(raw)
        return [
            control_observed_but_not_static(),
            control_static_but_not_observed(),
            control_agreement_is_reachable(),
            control_ambiguous_stem(),
            control_path_spelling_agrees(),
            control_windows_spelling_agrees(),
            control_extension_stripped_agrees(),
            control_unresolved_artifact_rows(),
            control_unresolved_call_rows(),
            control_empty_identity_dropped(),
            control_native_marker_skipped(),
            control_every_artifact_once(),
            control_second_loader_is_kept(),
            control_stale_population(),
            control_fresh_population(),
            control_dead_run_is_not_absence(measured),
            control_live_run_is_absence(measured),
            control_unmeasured_observer(unmeasured),
        ]


def main() -> int:
    controls = run_all()
    print("reconciler controls\n")
    for one in controls:
        print(f"{one.mark} {one.name}")
        print(f"       {one.detail[:150]}")

    failing = [one.name for one in controls if not one.held]
    print(f"\n{len(controls)} controls, {len(failing)} failing: {failing or 'none'}")

    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    target = generated / "reconcile_controls.json"
    payload = {"controls": [vars(one) for one in controls], "failing": failing}
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True))
        handle.write("\n")
    print(f"wrote {target}")
    return 0 if not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
