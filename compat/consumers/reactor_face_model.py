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

CONSUMER_ID: Final[str] = "reactor"


DERIVED: Final[frozenset[str]] = frozenset({"pose"})


def reactor_keys() -> tuple[str, ...]:
    from compat.harness import pinned_source

    repo, commit = pinned_clone(CONSUMER_ID)
    return pinned_source.subscript_keys(repo, commit, "reactor_utils.py", "save_face_model", "face")


def retained_keys() -> tuple[str, ...]:
    return tuple(one for one in reactor_keys() if one not in DERIVED)


@dataclass(frozen=True)
class FullObservation:
    base: Observation
    bbox: npt.NDArray[np.float32]
    kps: npt.NDArray[np.float32]
    det_score: npt.NDArray[np.float32]
    landmark_2d_106: npt.NDArray[np.float32]
    gender: npt.NDArray[np.int64]
    age: npt.NDArray[np.int64]
    face: Any


def pinned_clone(consumer_id: str) -> tuple[Path, str]:
    from compat.harness import provenance

    manifest = provenance.load_manifest()
    refs_root = (Path(__file__).resolve().parents[2] / manifest["refs_root"]).resolve()
    for row in manifest["consumers"]:
        if row["id"] == consumer_id:
            return provenance.clone_dir(refs_root, row["repo"]), row["commit"]
    raise LookupError(f"{consumer_id!r} is not in the manifest")


def upstream_io() -> tuple[Any, Any, Any]:
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
    _save, load, _face = upstream_io()
    return dict(load(str(where)))


def full_observations(limit: int = 4) -> list[FullObservation]:
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
                face=best,
            )
        )
    return out


class ReactorFaceModelRunner:
    consumer_id: str = CONSUMER_ID

    def __init__(self, found: list[FullObservation] | None = None) -> None:
        self._by_label: dict[str, FullObservation] = {
            one.base.label: one for one in (found if found is not None else full_observations())
        }
        self._scratch = Path(tempfile.mkdtemp(prefix="compat_reactor_"))

        atexit.register(shutil.rmtree, self._scratch, True)

    def cases(self) -> tuple[Case, ...]:
        replay_cases = tuple(
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
                ablations=self._ablations(label),
                measurements=("keys_upstream_returns",),
                note="upstream writes and upstream reads; the comparison is the file it produced",
            )
            for label in self._by_label
        )

        export_cases = tuple(
            Case(
                name=f"reactor_face_model_app_export_{label}",
                consumer_id=self.consumer_id,
                tier=Tier.CONSUMER,
                fixture=self._by_label[label].base.fixture,
                boundary=f"face_model_safetensors_app|{label}",
                exact_bytes=True,
                rtol=0.0,
                atol=0.0,
                retained=(),
                ablations=(),
                measurements=(),
                note="the application's export from its stored replay, against upstream's own writer",
            )
            for label in self._by_label
        )
        return (*replay_cases, *export_cases)

    def _ablations(self, label: str) -> tuple[Ablation, ...]:
        del label
        return (
            *(Ablation(primitive=one, expect_breaks=True) for one in retained_keys()),
            Ablation(
                primitive="gender",
                swap="opposite_label",
                expect_breaks=True,
                kind="substitution",
            ),
        )

    def _found(self, case: Case) -> FullObservation:
        return self._by_label[case.boundary.partition("|")[2]]

    def _is_app_export(self, case: Case) -> bool:
        return case.boundary.partition("|")[0] == "face_model_safetensors_app"

    def retained_for(self, case: Case) -> RetainedState:
        if self._is_app_export(case):
            return RetainedState()
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
        if self._is_app_export(case):
            return self._artifact(case.boundary, self._app_export(case))
        values: dict[str, Any] = {}
        for key in retained_keys():
            values[key] = retained.points(key) if key not in {"gender", "age"} else retained.integers(key)

        values["pose"] = pose_from_landmarks(values["landmark_3d_68"])

        ordered = {key: values[key] for key in reactor_keys()}
        blob = save_through_upstream(ordered, self._scratch / f"{case.name}_replay.safetensors")
        return self._artifact(case.boundary, blob)

    def _app_export(self, case: Case) -> bytes:
        from compat.storage import gallery
        from db import faces_native

        found = self._found(case)
        frame, sha = producer.decode(Path(found.base.fixture.path))
        native = gallery.native_round_trip(found.face, frame, sha)
        return faces_native.reactor_face_model_bytes(native)

    def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState:
        if ablation.swap == "opposite_label":
            held = retained.integers("gender")

            return retained.replacing("gender", np.asarray(1 - held, dtype=held.dtype))
        return retained.without(ablation.primitive)

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
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
