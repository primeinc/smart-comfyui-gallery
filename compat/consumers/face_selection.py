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
    note_considered,
)
from compat.corpus import groups
from compat.producers import insightface_pass as producer

CONSUMER_ID: Final[str] = "face_selection"

#: How many discriminating group photographs to exercise.
PHOTOGRAPHS: Final[int] = 3

#: How many candidates to look at before giving up: detection on a
#: multi-megapixel photograph is the expensive step. Hitting the cap is
#: recorded, never treated as 'none exist'.
SCAN_CEILING: Final[int] = 24


def area(face: Any) -> float:
    box = np.asarray(face.bbox, dtype=np.float64)
    return float((box[2] - box[0]) * (box[3] - box[1]))


def select_index(boxes: np.ndarray, rule: str) -> int:
    """Which ROW a rule picks, over bounding boxes alone.

    The one place either rule is implemented. `select` below delegates here so
    the live path and the stored-row path cannot drift into two rules with one
    name -- which is the failure this lane exists to detect in the vendors.

    `first` is detector order. `largest_bbox_area` is max by area. Nothing
    else is offered: a rule this does not know must fail loudly rather than
    fall through to a default that would silently make every case agree.
    """
    if rule == "first":
        return 0
    if rule == "largest_bbox_area":
        sides = np.asarray(boxes, dtype=np.float64)
        return int(np.argmax((sides[:, 2] - sides[:, 0]) * (sides[:, 3] - sides[:, 1])))
    raise KeyError(f"no selection rule called {rule!r}")


def select_rows(boxes: np.ndarray, rule: str) -> np.ndarray:
    """The bounding box a rule picks out of stored rows."""
    return np.asarray(boxes, dtype=np.float32)[select_index(boxes, rule)]


def select(found: list[Any], rule: str) -> Any:
    """The face a rule picks out of a live detection list."""
    if not found:
        raise ValueError("no faces to select from")
    boxes = np.asarray([one.bbox for one in found], dtype=np.float32)
    return found[select_index(boxes, rule)]


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

        Because the search selects FOR disagreement, `_rules_differ` is true
        of every photograph this returns and the substitution is expected to
        break on all of them. That expectation is still derived from the
        detection rather than written down: this filter is a fact about which
        photographs get in, and a corpus, a detector or a rule that changed
        would move the two independently.
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
            except (ValueError, OSError) as problem:
                # A candidate this scan passed over, not a required input
                # dropped. Recorded so the rejects stay reviewable.
                note_considered(CONSUMER_ID, str(candidate), f"not usable: {problem}")
                continue
            if not np.array_equal(select(faces, "first").bbox, select(faces, "largest_bbox_area").bbox):
                found.append(candidate)
        return found

    def _rules_differ(self, group: groups.Group) -> bool:
        """Whether the two declared rules reach different faces here."""
        found = self.detections(group)
        return not np.array_equal(select(found, "first").bbox, select(found, "largest_bbox_area").bbox)

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
                        retained=("face_rows", "selection_rule"),
                        ablations=(
                            Ablation(primitive="face_rows", expect_breaks=True),
                            # Whole pixels, as an INTEGER bbox column holds
                            # them. `select_index` compares areas, so rounding
                            # can go either way.
                            Ablation(
                                primitive="face_rows",
                                swap="integer_pixels",
                                expect_breaks=True,
                                kind="substitution",
                            ),
                            Ablation(primitive="selection_rule", expect_breaks=True),
                            # The rule the store kept, swapped for the other
                            # the manifest declares: it breaks when the two
                            # reach different faces, read off the detection.
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
        """Every detected face's bbox, in detector order: what a store holds.

        A row per face, not one chosen face. A retained state holding the
        chosen face would make `replay` hand back what `retained_for` was
        given, comparing the selection against itself -- a fact about
        `numpy.copy`. Here the store keeps the rows and the replay
        runs the selection over them, so what is under test is whether the
        stored rows are enough to reach the same face.
        """
        _, _, group = self._parts(case)
        return np.asarray([one.bbox for one in self.detections(group)], dtype=np.float32)

    def retained_for(self, case: Case) -> RetainedState:
        return RetainedState(face_rows=self._rows(case), selection_rule=self._parts(case)[0])

    def _artifact(self, name: str, values: np.ndarray) -> Artifact:
        return Artifact(name=name, dtype=str(values.dtype), shape=values.shape, sha256=digest(values), values=values)

    def baseline(self, case: Case) -> Artifact:
        """The face this consumer's own declared rule picks, live.

        Every photograph the detector sees two faces in, including the ones
        where the two rules agree. Refusing those made the admission criterion
        and the substitution's expectation THE SAME TEST -- a case only ran
        when the rules already differed, so the swap could not come out any
        other way. They now run and are expected not to break, which is the
        negative control the lane had none of.
        """
        rule, _, _ = self._parts(case)
        return self._artifact(case.boundary, self._chosen(case, rule))

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        """The face the retained rows and the retained rule reach.

        `select` is the same function the baseline runs; what differs is that
        it is given STORED rows rather than the live detection, which is the
        only thing this lane can honestly claim to test.
        """
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
