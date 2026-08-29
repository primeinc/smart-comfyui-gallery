"""ReActor's face model, written and read back by ReActor's own code.

Every other consumer in the population is checked by comparing our replay
against our baseline. This one is different and it is the reason it goes
first: ReActor ships a LOADER. `load_face_model` is upstream code that takes
a file and returns a `Face`, so the assertion here is not "our two functions
agree" but "upstream accepted our bytes and got its object back".

`save_face_model` (reactor_utils.py, ComfyUI-ReActor@6ad6b35a4df2) names nine
keys and reads each with a bare subscript:

    bbox  kps  det_score  landmark_3d_68  pose  landmark_2d_106
    embedding  gender  age

A missing one raises KeyError -- into a bare `except Exception` that PRINTS
and returns, leaving no file behind. So the ablation evidence is the absence
of the file, and this checks for that rather than for a raised exception: a
runner that only watched for an exception would record every ablation as
surviving, and report that nothing ReActor asks for is necessary.

The retained set is EIGHT of those nine. `pose` is left out and derived,
because `compat/consumers/producer_derivations.py` establishes by execution
that upstream computes it from `landmark_3d_68` with no pixels and no second
inference. If that claim were wrong this case would diverge, which is the
point of running them in the same suite.

The boundary is the safetensors file's bytes -- the actual artifact a ReActor
user loads -- not a tensor-by-tensor comparison that could agree while the
container differed.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from compat.consumers.producer_derivations import Observation, observations, pose_from_landmarks
from compat.contracts.case import Ablation, Artifact, Case, Measurement, RetainedState, Tier
from compat.corpus.loaded import best_face
from compat.producers import insightface_pass as producer
from compat.storage import precision

CONSUMER_ID: Final[str] = "reactor"

#: `pose` is the one key upstream names that this does NOT retain, because
#: `producer_derivations` establishes by execution that upstream computes it
#: from `landmark_3d_68` with no pixels and no second inference.
DERIVED: Final[frozenset[str]] = frozenset({"pose"})


def reactor_keys() -> tuple[str, ...]:
    """The keys `save_face_model` requires, read out of its pinned AST.

    NOT a typed-out list. Upstream states this contract by subscripting
    `face["bbox"]`, `face["kps"]` and so on; retyping those into a constant
    would make the contract something remembered rather than something
    upstream says, and a tenth key added at a later commit would leave the
    copy stale while every case kept passing.

    Source order is preserved because it is also the order the vendor writes
    into the safetensors container, and that file's header records key order.
    """
    from compat.harness import pinned_source

    repo, commit = pinned_clone(CONSUMER_ID)
    return pinned_source.subscript_keys(repo, commit, "reactor_utils.py", "save_face_model", "face")


def retained_keys() -> tuple[str, ...]:
    """What must be durably stored: everything upstream names, minus derived.

    Necessity is not asserted here -- every one of these gets an ablation, and
    a key that turns out not to be needed comes back CONTRADICTED.
    """
    return tuple(one for one in reactor_keys() if one not in DERIVED)


@dataclass(frozen=True)
class FullObservation:
    """One face with every key ReActor names, not just the derivable ones."""

    base: Observation
    bbox: npt.NDArray[np.float32]
    kps: npt.NDArray[np.float32]
    det_score: npt.NDArray[np.float32]
    landmark_2d_106: npt.NDArray[np.float32]
    gender: npt.NDArray[np.int64]
    age: npt.NDArray[np.int64]


def pinned_clone(consumer_id: str) -> tuple[Path, str]:
    """This consumer's clone directory and pinned commit, from the manifest.

    Resolved through `provenance` rather than spelled out here. `refs/` is a
    SIBLING of the repository, not a child, and a hand-built path got that
    wrong once already -- pointing inside the working tree, where the clone
    does not exist. One resolver means one place to be wrong.
    """
    from compat.harness import provenance

    manifest = provenance.load_manifest()
    refs_root = (Path(__file__).resolve().parents[2] / manifest["refs_root"]).resolve()
    for row in manifest["consumers"]:
        if row["id"] == consumer_id:
            return provenance.clone_dir(refs_root, row["repo"]), row["commit"]
    raise LookupError(f"{consumer_id!r} is not in the manifest")


def upstream_io() -> tuple[Any, Any, Any]:
    """`save_face_model`, `load_face_model` and `Face`, as committed.

    Loaded through `pinned_source` rather than imported, because
    `reactor_utils` imports ComfyUI's `folder_paths` and `comfy.utils` at
    module level. Standing ComfyUI up to answer a storage question would make
    the suite depend on an application none of this is about; extracting the
    symbols runs upstream's exact committed bytes instead and records which
    names were supplied from outside.

    `Face` is ReActor's OWN, from `reactor_core/face_objects.py` -- not
    insightface's. Worth stating because getting it wrong is invisible: both
    are dict subclasses, so the wrong one would work and would be testing a
    container upstream never uses.
    """
    import torch
    from safetensors.torch import safe_open, save_file

    from compat.harness import pinned_source

    repo, commit = pinned_clone(CONSUMER_ID)
    face, _ = pinned_source.load_symbol(repo, commit, "reactor_core/face_objects.py", "Face", {"np": np})
    supplied: dict[str, Any] = {"save_file": save_file, "safe_open": safe_open, "torch": torch, "Face": face}
    save, _ = pinned_source.load_symbol(repo, commit, "reactor_utils.py", "save_face_model", dict(supplied))
    load, _ = pinned_source.load_symbol(repo, commit, "reactor_utils.py", "load_face_model", dict(supplied))
    return save, load, face


def save_through_upstream(values: dict[str, Any], where: Path) -> bytes:
    """Upstream's `save_face_model`, then the bytes it left behind.

    The file is removed first and its existence checked after, because
    upstream swallows every exception into a print. Without that, a save that
    never happened is indistinguishable from one that did, and an ablation
    would be recorded as surviving when it had actually failed outright --
    which would report that nothing ReActor asks for is necessary.
    """
    save, _load, face = upstream_io()
    keys = reactor_keys()

    if where.exists():
        where.unlink()
    save(face(values), str(where))
    if not where.is_file():
        raise ValueError(
            f"upstream save_face_model wrote no file: it subscripts all of {keys} and swallows the "
            f"KeyError, so a missing key leaves nothing behind. Present: {sorted(values)}"
        )
    return where.read_bytes()


def load_through_upstream(where: Path) -> dict[str, npt.NDArray[np.generic]]:
    """Upstream's own `load_face_model`, so the reader is theirs too."""
    _save, load, _face = upstream_io()
    return dict(load(str(where)))


def full_observations(limit: int = 4) -> list[FullObservation]:
    """Real faces carrying all nine keys, or an empty list."""
    from compat.corpus import index as corpus

    if not corpus.KYC.is_dir():
        return []

    app = producer.analysis()
    out: list[FullObservation] = []
    for base in observations(limit):
        frame, _ = producer.decode(Path(base.fixture.path))
        faces = app.get(frame)
        if not faces:
            continue
        # `loaded.best_face`, not highest det_score. Two rules for "our
        # producer's face" named different people on any multi-face
        # photograph, and this lane's must-retain evidence sat beside the
        # storage lane's as though they described one subject.
        best = best_face(faces)
        lmk106 = best.get("landmark_2d_106")
        if lmk106 is None or best.gender is None or best.age is None:
            continue
        out.append(
            FullObservation(
                base=base,
                bbox=np.asarray(best.bbox, dtype=np.float32),
                kps=np.asarray(best.kps, dtype=np.float32),
                det_score=np.asarray(best.det_score, dtype=np.float32),
                landmark_2d_106=np.asarray(lmk106, dtype=np.float32),
                gender=np.asarray(best.gender, dtype=np.int64),
                age=np.asarray(best.age, dtype=np.int64),
            )
        )
    return out


class ReactorFaceModelRunner:
    """Nine tensors out of upstream's writer, through upstream's reader."""

    consumer_id: str = CONSUMER_ID

    def __init__(self, found: list[FullObservation] | None = None) -> None:
        self._by_label: dict[str, FullObservation] = {
            one.base.label: one for one in (found if found is not None else full_observations())
        }
        self._scratch = Path(tempfile.mkdtemp(prefix="compat_reactor_"))
        # Removed when the interpreter exits. `tempfile.mkdtemp` has no
        # owner, and one runner per consumer per run left a directory
        # behind every time the suite ran.
        atexit.register(shutil.rmtree, self._scratch, True)

    def cases(self) -> tuple[Case, ...]:
        return tuple(
            Case(
                name=f"reactor_face_model_{label}",
                consumer_id=self.consumer_id,
                tier=Tier.CONSUMER,
                fixture=self._by_label[label].base.fixture,
                boundary=f"face_model_safetensors|{label}",
                exact_bytes=True,
                rtol=0.0,
                atol=0.0,
                retained=retained_keys(),
                # Every retained key gets its own ablation. `pose` gets none:
                # it is NOT retained, and the case reproducing at all is the
                # evidence that deriving it was sufficient.
                ablations=self._ablations(label),
                measurements=("keys_upstream_returns",),
                note="upstream writes and upstream reads; the comparison is the file it produced",
            )
            for label in self._by_label
        )

    def _ablations(self, label: str) -> tuple[Ablation, ...]:
        """One presence question and one WIDTH question per key.

        A removal shows only that the replay indexes the key --
        `save_face_model` subscripts all of them, so every removal
        ends the same way and separates nothing. The width question
        does separate, where it separates at all: `expect_breaks` is
        read off whether narrowing actually MOVES this value, not off a
        typed-out list and not off its dtype alone -- a float whose
        every element is exact in binary16 is unchanged by the swap and
        `run_ablation` records that as INCONCLUSIVE.
        """
        held = self._state(self._by_label[label])
        return (
            *(Ablation(primitive=one, expect_breaks=True) for one in retained_keys()),
            *(
                Ablation(
                    primitive=one,
                    swap="half_precision",
                    expect_breaks=precision.narrows(held.array(one)),
                    kind="substitution",
                )
                for one in retained_keys()
            ),
            # `precision.half` cannot degrade an integer, so `age` and
            # `gender` had a removal that proved nothing and a width
            # question that was a no-op. A decade bucket IS a storable
            # form of an age -- coarser and cheaper, and the one a
            # privacy-minded schema would reach for first.
            Ablation(
                primitive="age",
                swap="decade_bucket",
                expect_breaks=True,
                kind="substitution",
            ),
        )

    def _found(self, case: Case) -> FullObservation:
        return self._by_label[case.boundary.partition("|")[2]]

    def retained_for(self, case: Case) -> RetainedState:
        return self._state(self._found(case))

    def _state(self, found: FullObservation) -> RetainedState:
        return RetainedState(
            bbox=found.bbox.copy(),
            kps=found.kps.copy(),
            det_score=found.det_score.copy(),
            landmark_3d_68=found.base.landmark_3d_68.copy(),
            landmark_2d_106=found.landmark_2d_106.copy(),
            embedding=found.base.embedding.copy(),
            gender=found.gender.copy(),
            age=found.age.copy(),
        )

    def _artifact(self, name: str, blob: bytes) -> Artifact:
        values = np.frombuffer(blob, dtype=np.uint8)
        return Artifact(
            name=name,
            dtype="uint8",
            shape=(values.size,),
            sha256=producer.digest_array(values),
            values=values,
        )

    def baseline(self, case: Case) -> Artifact:
        """The producer's own nine values, through upstream's writer."""
        found = self._found(case)
        values = {
            "bbox": found.bbox,
            "kps": found.kps,
            "det_score": found.det_score,
            "landmark_3d_68": found.base.landmark_3d_68,
            "pose": found.base.pose,
            "landmark_2d_106": found.landmark_2d_106,
            "embedding": found.base.embedding,
            "gender": found.gender,
            "age": found.age,
        }
        blob = save_through_upstream(values, self._scratch / f"{case.name}_baseline.safetensors")
        return self._artifact(case.boundary, blob)

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        """Eight retained values, pose derived, through the same writer.

        Nothing here reads the fixture. A key the retained state does not
        carry raises by name out of `RetainedState`, which is what makes an
        ablation legible rather than a mysterious upstream print.
        """
        values: dict[str, Any] = {}
        for key in retained_keys():
            values[key] = retained.points(key) if key not in {"gender", "age"} else retained.integers(key)

        # Derived, not retained. Proven byte-identical to the producer's own
        # pose in `producer_derivations`; if that stops holding, this case
        # diverges rather than quietly storing a ninth column forever.
        values["pose"] = pose_from_landmarks(values["landmark_3d_68"])

        ordered = {key: values[key] for key in reactor_keys()}
        blob = save_through_upstream(ordered, self._scratch / f"{case.name}_replay.safetensors")
        return self._artifact(case.boundary, blob)

    def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState:
        if ablation.swap == "decade_bucket":
            held = retained.integers("age")
            return retained.replacing("age", (np.rint(held / 10.0) * 10).astype(held.dtype))
        if ablation.swap == "half_precision":
            return retained.replacing(ablation.primitive, precision.half(retained.array(ablation.primitive)))
        return retained.without(ablation.primitive)

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        """What upstream's reader actually hands a ReActor node back."""
        if name != "keys_upstream_returns":
            raise KeyError(f"{self.consumer_id} has no measurement called {name!r}")

        self.replay(case, retained)
        loaded = load_through_upstream(self._scratch / f"{case.name}_replay.safetensors")
        shapes = ", ".join(f"{key}{tuple(loaded[key].shape)}" for key in sorted(loaded))
        missing = sorted(set(reactor_keys()) - set(loaded))
        return Measurement(
            name=name,
            unit="keys",
            value=float(len(loaded)),
            basis="reactor_utils.load_face_model at the pinned ComfyUI-ReActor commit, over the replayed file",
            detail=(
                f"{len(loaded)} of {len(reactor_keys())} keys returned"
                + (f"; MISSING {missing}" if missing else "; none missing")
                + f"; {shapes}"
            ),
        )
