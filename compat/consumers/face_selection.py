from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import numpy as np

from compat.assertions.arrays import digest
from compat.consumers.face_family import vendor_setups
from compat.contracts.case import (
    Ablation,
    Artifact,
    Case,
    Fixture,
    Measurement,
    RetainedState,
    Tier,
    note_considered,
)
from compat.corpus import groups
from compat.producers import insightface_pass as producer

CONSUMER_ID: Final[str] = "face_selection"


PHOTOGRAPHS: Final[int] = 3


SCAN_CEILING: Final[int] = 24


def area(face: Any) -> float:
    box = np.asarray(face.bbox, dtype=np.float64)
    return float((box[2] - box[0]) * (box[3] - box[1]))


def select_index(boxes: np.ndarray, rule: str) -> int:
    if rule == "first":
        return 0
    if rule == "largest_bbox_area":
        sides = np.asarray(boxes, dtype=np.float64)
        return int(np.argmax((sides[:, 2] - sides[:, 0]) * (sides[:, 3] - sides[:, 1])))
    raise KeyError(f"no selection rule called {rule!r}")


def select_rows(boxes: np.ndarray, rule: str) -> np.ndarray:
    return np.asarray(boxes, dtype=np.float32)[select_index(boxes, rule)]


def select(found: list[Any], rule: str) -> Any:
    if not found:
        raise ValueError("no faces to select from")
    boxes = np.asarray([one.bbox for one in found], dtype=np.float32)
    return found[select_index(boxes, rule)]


OTHER: Final[dict[str, str]] = {"first": "largest_bbox_area", "largest_bbox_area": "first"}


class FaceSelectionRunner:
    consumer_id = CONSUMER_ID

    def __init__(self) -> None:

        self._detected: dict[str, list[Any]] = {}
        self._setups = vendor_setups()
        self._groups = {one.asset_id: one for one in self._discriminating()}

    def _discriminating(self) -> list[groups.Group]:
        found: list[groups.Group] = []
        self.scanned = 0
        self.hit_ceiling = False
        for candidate in groups.scan(least=2):
            if len(found) >= PHOTOGRAPHS:
                break
            if self.scanned >= SCAN_CEILING:
                self.hit_ceiling = True
                break
            self.scanned += 1
            try:
                faces = self.detections(candidate)
            except (ValueError, OSError) as problem:
                note_considered(CONSUMER_ID, str(candidate), f"not usable: {problem}")
                continue
            if not np.array_equal(select(faces, "first").bbox, select(faces, "largest_bbox_area").bbox):
                found.append(candidate)
        return found

    def _rules_differ(self, group: groups.Group) -> bool:
        found = self.detections(group)
        return not np.array_equal(select(found, "first").bbox, select(found, "largest_bbox_area").bbox)

    def _parts(self, case: Case) -> tuple[str, str, groups.Group]:
        rule, consumer, asset = case.boundary.split("|", 2)
        return rule, consumer, self._groups[asset]

    def detections(self, group: groups.Group) -> list[Any]:
        if group.asset_id not in self._detected:
            frame, _ = producer.decode(Path(group.path))
            found = producer.analysis().get(frame)
            if len(found) < 2:
                raise ValueError(
                    f"{group.asset_id}: the detector found {len(found)} face(s); "
                    f"the dataset releases {group.released_people} people. "
                    f"A photograph the detector does not see two faces in cannot "
                    f"separate `first` from `largest_bbox_area`"
                )
            self._detected[group.asset_id] = found
        return self._detected[group.asset_id]

    def _fixture(self, group: groups.Group) -> Fixture:
        return Fixture(
            name=f"group_{group.asset_id}",
            path=group.path,
            sha256=group.sha256,
            kind="group_photograph",
            note=f"{group.released_people} released people; {groups.LICENCE}; not vendored",
        )

    def cases(self) -> tuple[Case, ...]:

        by_rule: dict[str, list[str]] = {}
        for setup in self._setups.values():
            by_rule.setdefault(setup.select, []).append(setup.consumer_id)

        out: list[Case] = []
        for group in self._groups.values():
            for rule, consumers in sorted(by_rule.items()):
                out.extend(
                    Case(
                        name=f"select_{rule}_{consumer}_{group.asset_id}",
                        consumer_id=CONSUMER_ID,
                        tier=Tier.PRIMITIVE,
                        fixture=self._fixture(group),
                        boundary=f"{rule}|{consumer}|{group.asset_id}",
                        exact_bytes=True,
                        rtol=0.0,
                        atol=0.0,
                        retained=("face_rows", "selection_rule"),
                        ablations=(
                            Ablation(primitive="face_rows", expect_breaks=True),
                            Ablation(
                                primitive="face_rows",
                                swap="integer_pixels",
                                expect_breaks=True,
                                kind="substitution",
                            ),
                            Ablation(primitive="selection_rule", expect_breaks=True),
                            Ablation(
                                primitive="selection_rule",
                                swap="other_selection_rule",
                                expect_breaks=self._rules_differ(group),
                                kind="substitution",
                            ),
                        ),
                        measurements=("rules_disagree",),
                        note=f"{consumer} declares select = {rule}",
                    )
                    for consumer in sorted(consumers)
                )
        return tuple(out)

    def _chosen(self, case: Case, rule: str) -> np.ndarray:
        _, _, group = self._parts(case)
        return np.asarray(select(self.detections(group), rule).bbox, dtype=np.float32)

    def _rows(self, case: Case) -> np.ndarray:
        _, _, group = self._parts(case)
        return np.asarray([one.bbox for one in self.detections(group)], dtype=np.float32)

    def retained_for(self, case: Case) -> RetainedState:
        return RetainedState(face_rows=self._rows(case), selection_rule=self._parts(case)[0])

    def _artifact(self, name: str, values: np.ndarray) -> Artifact:
        return Artifact(name=name, dtype=str(values.dtype), shape=values.shape, sha256=digest(values), values=values)

    def baseline(self, case: Case) -> Artifact:
        rule, _, _ = self._parts(case)
        return self._artifact(case.boundary, self._chosen(case, rule))

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        rows = np.asarray(retained.array("face_rows"), dtype=np.float32)
        rule = retained.text("selection_rule")
        return self._artifact(case.boundary, select_rows(rows, rule))

    def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState:
        del case
        if ablation.swap == "integer_pixels":
            held = np.asarray(retained.array("face_rows"), dtype=np.float32)
            return retained.replacing("face_rows", np.rint(held).astype(np.float32))
        if ablation.swap == "other_selection_rule":
            return retained.replacing("selection_rule", OTHER[retained.text("selection_rule")])
        return retained.without(ablation.primitive)

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        if name != "rules_disagree":
            raise KeyError(f"{CONSUMER_ID} has no measurement called {name!r}")
        rule, consumer, group = self._parts(case)
        found = self.detections(group)
        mine, theirs = select(found, rule), select(found, OTHER[rule])
        return Measurement(
            name=name,
            unit="px^2",
            value=abs(area(mine) - area(theirs)),
            basis="area of the face this rule picks against the area the other rule picks",
            detail=(
                f"{group.asset_id}: {len(found)} faces detected, {group.released_people} released. "
                f"{consumer} takes {rule} -> area {area(mine):,.0f} px^2; "
                f"{OTHER[rule]} -> area {area(theirs):,.0f} px^2"
            ),
        )


def all_runners() -> tuple[FaceSelectionRunner, ...]:
    return (FaceSelectionRunner(),)
