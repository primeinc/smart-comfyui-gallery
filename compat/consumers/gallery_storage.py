"""Storage candidates against the producer's own record, key by key.

One case per (candidate, key, photograph). Keys come from
`Observation.of(face)`, which iterates the producer's record; no field is
named here.

    BASELINE   the producer's value, its own dtype
    REPLAY     the same key after the candidate's round trip

`rtol` and `atol` are 0 and `exact_bytes` is False. Zero because the claim is
conservation. Not `exact_bytes` because that path casts through int64
(compat/assertions/arrays.py:78-81) and reports sub-unit float error as zero.

A key the candidate does not return is retained as an empty array in the
producer's dtype, so it compares as a shape divergence. An omitted key would
raise inside `replay`, and `run_case` classifies a raise as UNSUPPORTED
(compat/harness/run.py:130-140), which means absent runtime, not lost data.

Tier is PRIMITIVE. This is not consumer-tier coverage: a key surviving storage
says nothing about whether any consumer reproduces from it.
"""

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
from compat.storage import gallery_v45
from compat.storage.contract import Observation, StorageContract, emitted_keys

CONSUMER_ID: Final[str] = "gallery_storage"


def candidates() -> tuple[StorageContract, ...]:
    return gallery_v45.candidates()


class GalleryStorageRunner:
    """Every storage candidate, over every key the producer emitted."""

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
        """The producer's record for one photograph."""
        if shot.label not in self._emitted:
            self._emitted[shot.label] = Observation.of(our_face(shot))
        return self._emitted[shot.label]

    def stored(self, candidate: str, shot: Shot) -> Observation:
        """One round trip per (candidate, photograph), shared by its keys.

        Deterministic: fixed clock, fixed detection, fixed model ids.
        """
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
                        # No ablation: with one key retained, removing and
                        # reading it back can only come out "broke". This lane
                        # measures conservation, which the verdict carries.
                        measurements=("stored_form",),
                        note=candidate.described,
                    )
                    for key in emitted
                )
        return tuple(out)

    def retained_for(self, case: Case) -> RetainedState:
        """What the candidate gave back."""
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
        """The producer's value, before any candidate saw it."""
        _, key, shot = self._parts(case)
        return self._artifact(case.boundary, self.emitted(shot)[key])

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        """The stored value. Nothing here opens the photograph."""
        _, key, _ = self._parts(case)
        return self._artifact(case.boundary, np.asarray(retained.array(key)))

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        """dtype and shape as the candidate returned them.

        A value that survives numerically in another width has still changed:
        ReActor builds `torch.tensor(face["age"])`
        (Gourieff/comfyui-reactor-node reactor_utils.py::save_face_model), so
        the dtype is part of the file it writes.
        """
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
    """Producer keys the generated inventory does not record.

    Non-empty means `compat/generated/producer_inventory.json` is stale
    against the live pass.
    """
    recorded = emitted_keys()
    if not recorded:
        return frozenset()
    live: set[str] = set()
    for shot in shots():
        live |= set(Observation.of(our_face(shot)).keys())
    return frozenset(live - recorded)


def all_runners() -> tuple[GalleryStorageRunner, ...]:
    return (GalleryStorageRunner(),)
