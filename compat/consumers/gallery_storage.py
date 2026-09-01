from __future__ import annotations

from typing import Final

import numpy as np

from compat.assertions.arrays import digest
from compat.contracts.case import (
    Artifact,
    Case,
    Measurement,
    RetainedState,
    Tier,
)
from compat.corpus.loaded import Shot, our_face, shots
from compat.storage import gallery
from compat.storage.contract import Observation, StorageContract, emitted_keys

CONSUMER_ID: Final[str] = "gallery_storage"


def candidates() -> tuple[StorageContract, ...]:
    return gallery.candidates()


class GalleryStorageRunner:
    consumer_id = CONSUMER_ID

    def __init__(self) -> None:
        self._shots: dict[str, Shot] = {one.label: one for one in shots()}
        self._candidates: dict[str, StorageContract] = {one.name: one for one in candidates()}
        self._stored: dict[tuple[str, str], Observation] = {}
        self._emitted: dict[str, Observation] = {}

    def _parts(self, case: Case) -> tuple[str, str, Shot]:
        candidate, key, label = case.boundary.split("|", 2)
        return candidate, key, self._shots[label]

    def emitted(self, shot: Shot) -> Observation:
        if shot.label not in self._emitted:
            self._emitted[shot.label] = Observation.of(our_face(shot))
        return self._emitted[shot.label]

    def stored(self, candidate: str, shot: Shot) -> Observation:
        memo = (candidate, shot.label)
        if memo not in self._stored:
            self._stored[memo] = self._candidates[candidate].round_trip(our_face(shot), shot.frame, shot.fixture.sha256)
        return self._stored[memo]

    def cases(self) -> tuple[Case, ...]:
        out: list[Case] = []
        for shot in self._shots.values():
            emitted = self.emitted(shot)
            for candidate in self._candidates.values():
                out.extend(
                    Case(
                        name=f"store_{candidate.name}_{key}_{shot.label}",
                        consumer_id=CONSUMER_ID,
                        tier=Tier.PRIMITIVE,
                        fixture=shot.fixture,
                        boundary=f"{candidate.name}|{key}|{shot.label}",
                        rtol=0.0,
                        atol=0.0,
                        exact_bytes=False,
                        retained=(key,),
                        measurements=("stored_form",),
                        note=candidate.described,
                    )
                    for key in emitted
                )
        return tuple(out)

    def retained_for(self, case: Case) -> RetainedState:
        candidate, key, shot = self._parts(case)
        stored = self.stored(candidate, shot)
        held = stored.get(key)
        if held is None:
            held = np.zeros(0, dtype=self.emitted(shot)[key].dtype)
        return RetainedState(**{key: held})

    def _artifact(self, name: str, values: np.ndarray) -> Artifact:
        return Artifact(
            name=name,
            dtype=str(values.dtype),
            shape=values.shape,
            sha256=digest(values),
            values=values,
        )

    def baseline(self, case: Case) -> Artifact:
        _, key, shot = self._parts(case)
        return self._artifact(case.boundary, self.emitted(shot)[key])

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        _, key, _ = self._parts(case)
        return self._artifact(case.boundary, np.asarray(retained.array(key)))

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        if name != "stored_form":
            raise KeyError(f"{CONSUMER_ID} does not measure {name!r}")
        _, key, shot = self._parts(case)
        held = np.asarray(retained.array(key))
        want = self.emitted(shot)[key]
        return Measurement(
            name=name,
            unit="dtype/shape",
            value=float(held.size),
            basis="dtype and shape returned by the candidate's read path",
            detail=f"{key}: producer {want.dtype}{want.shape} -> stored {held.dtype}{held.shape}",
        )


def unlisted_keys() -> frozenset[str]:
    recorded = emitted_keys()
    if not recorded:
        return frozenset()
    live: set[str] = set()
    for shot in shots():
        live |= set(Observation.of(our_face(shot)).keys())
    return frozenset(live - recorded)


def all_runners() -> tuple[GalleryStorageRunner, ...]:
    return (GalleryStorageRunner(),)
