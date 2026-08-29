"""Selection semantics, on photographs holding more than one real person.

`first` and `largest_bbox_area` agree on every single-subject photograph, so a
corpus of one face per frame cannot tell them apart and every selection claim
in the manifest was untested. These cases run both rules over real group
photographs and assert they DISAGREE.

The disagreement is the test. A case whose two rules pick the same face proves
nothing about either, so it is recorded UNSUPPORTED rather than PASS -- the
photograph was not discriminating, which is a fact about the fixture, not
about the consumer.

MUTATION
--------
`selection_rule_mutated` swaps the case's declared rule for the other one. It
MUST break. A consumer whose case survives the swap is not actually selecting
the way its manifest row says, and every downstream claim quoted against that
row is unearned.
"""

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
)
from compat.corpus import groups
from compat.producers import insightface_pass as producer

CONSUMER_ID: Final[str] = "face_selection"

#: How many discriminating group photographs to exercise.
PHOTOGRAPHS: Final[int] = 3

#: How many candidates to look at before giving up. Detection on a
#: multi-megapixel photograph is the expensive step, so the search states
#: its own ceiling instead of scanning the whole dataset. Hitting the cap
#: is recorded, never silently treated as 'none exist'.
SCAN_CEILING: Final[int] = 24


def area(face: Any) -> float:
    box = np.asarray(face.bbox, dtype=np.float64)
    return float((box[2] - box[0]) * (box[3] - box[1]))


def select(found: list[Any], rule: str) -> Any:
    """The two rules the manifest declares, applied to a detection list.

    `first` is detector order. `largest_bbox_area` is max by area. Nothing
    else is offered: a rule this does not know must fail loudly rather than
    fall through to a default that would silently make every case agree.
    """
    if rule == "first":
        return found[0]
    if rule == "largest_bbox_area":
        return max(found, key=area)
    raise KeyError(f"no selection rule called {rule!r}")


OTHER: Final[dict[str, str]] = {"first": "largest_bbox_area", "largest_bbox_area": "first"}


class FaceSelectionRunner:
    """Both selection rules, over real photographs of several people."""

    consumer_id = CONSUMER_ID

    def __init__(self) -> None:
        # Order matters: `_discriminating` detects, so its memo must exist
        # first.
        self._detected: dict[str, list[Any]] = {}
        self._setups = vendor_setups()
        self._groups = {one.asset_id: one for one in self._discriminating()}

    def _discriminating(self) -> list[groups.Group]:
        """Photographs where the two rules actually pick DIFFERENT faces.

        Taking the first N group photographs was not enough: on all three the
        detector's first face was also its largest, so every case came back
        UNSUPPORTED and the lane proved nothing. `released_people >= 2` is the
        dataset's claim about the picture; it is not a claim about detector
        ORDER, and order is what separates `first` from `largest_bbox_area`.

        So the fixture is SEARCHED rather than assumed: scan until enough
        photographs are found where the rules disagree, and record how many
        had to be looked at to find them.
        """
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
            except (ValueError, OSError):
                continue
            if not np.array_equal(select(faces, "first").bbox, select(faces, "largest_bbox_area").bbox):
                found.append(candidate)
        return found

    def _parts(self, case: Case) -> tuple[str, str, groups.Group]:
        rule, consumer, asset = case.boundary.split("|", 2)
        return rule, consumer, self._groups[asset]

    def detections(self, group: groups.Group) -> list[Any]:
        """Our producer's faces for one photograph, detector order preserved."""
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
        # One case per (declared rule, consumer that declares it, photograph).
        # Consumers are grouped by rule so the population shows which rule each
        # one is actually being held to.
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
                        retained=("selected_bbox",),
                        ablations=(
                            Ablation(primitive="selection_rule_mutated", expect_breaks=True, kind="substitution"),
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

    def retained_for(self, case: Case) -> RetainedState:
        rule, _, _ = self._parts(case)
        return RetainedState(selected_bbox=self._chosen(case, rule))

    def _artifact(self, name: str, values: np.ndarray) -> Artifact:
        return Artifact(name=name, dtype=str(values.dtype), shape=values.shape, sha256=digest(values), values=values)

    def baseline(self, case: Case) -> Artifact:
        """The face this consumer's own declared rule picks.

        UNSUPPORTED when the two rules agree on this photograph: the fixture
        could not discriminate, so a match would be evidence of nothing.
        """
        rule, _, group = self._parts(case)
        found = self.detections(group)
        if np.array_equal(select(found, rule).bbox, select(found, OTHER[rule]).bbox):
            raise ValueError(
                f"{group.asset_id}: `first` and `largest_bbox_area` pick the SAME face, so this "
                f"photograph cannot separate them. The detector's first face is also its largest"
            )
        return self._artifact(case.boundary, self._chosen(case, rule))

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        return self._artifact(case.boundary, np.asarray(retained.array("selected_bbox"), dtype=np.float32))

    def ablate(self, case: Case, retained: RetainedState, primitive: str) -> RetainedState:
        if primitive == "selection_rule_mutated":
            rule, _, _ = self._parts(case)
            return retained.replacing("selected_bbox", self._chosen(case, OTHER[rule]))
        return retained.without(primitive)

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
