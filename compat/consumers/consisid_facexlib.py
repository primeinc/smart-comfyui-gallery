from __future__ import annotations

from typing import Any, Final

import numpy as np

from compat.assertions.minimize import Rect
from compat.consumers.aligned_crop import analytic_footprint
from compat.contracts.case import (
    Ablation,
    Artifact,
    Case,
    Measurement,
    RetainedState,
    Tier,
    UInt8Array,
)
from compat.corpus.loaded import Shot, our_face, shots
from compat.producers import insightface_pass as producer
from compat.storage import derivatives

CONSUMER_ID: Final[str] = "consisid"


RESIZE_LONG_EDGE: Final[int] = 1024


FACE_SIZE: Final[int] = 512


START_MARGIN: Final[float] = 2.0


def resize_numpy_image_long(image: UInt8Array, long_edge: int = RESIZE_LONG_EDGE) -> UInt8Array:
    import cv2

    height, width = image.shape[:2]
    if max(height, width) <= long_edge:
        return np.asarray(image, dtype=np.uint8)
    scale = long_edge / max(height, width)
    out = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LANCZOS4)
    return np.asarray(out, dtype=np.uint8)


def scaled(patch: UInt8Array, factor: float) -> UInt8Array:
    import cv2

    if factor == 1.0:
        return np.asarray(patch, dtype=np.uint8)
    height, width = patch.shape[:2]
    out = cv2.resize(patch, (int(width * factor), int(height * factor)), interpolation=cv2.INTER_LANCZOS4)
    return np.asarray(out, dtype=np.uint8)


def face_helper() -> Any:
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper

    if "helper" not in _cached:
        _cached["helper"] = FaceRestoreHelper(
            upscale_factor=1, face_size=FACE_SIZE, crop_ratio=(1, 1), det_model="retinaface_resnet50", device="cpu"
        )
    return _cached["helper"]


_cached: dict[str, Any] = {}


def align_through_facexlib(image_bgr: UInt8Array) -> UInt8Array:
    helper = face_helper()
    helper.clean_all()
    helper.read_image(image_bgr)
    helper.get_face_landmarks_5(only_center_face=True)
    helper.align_warp_face()
    if len(helper.cropped_faces) == 0:
        raise ValueError("facexlib found no face to align in these pixels")
    return np.asarray(helper.cropped_faces[0], dtype=np.uint8)


def _artifact(name: str, values: UInt8Array) -> Artifact:
    return Artifact(
        name=name,
        dtype=str(values.dtype),
        shape=tuple(int(one) for one in values.shape),
        sha256=producer.digest_array(values),
        values=values,
    )


def grown(box: Rect, factor: float, frame_wh: tuple[int, int]) -> Rect:
    width, height = frame_wh
    half_w = (box.width * factor) / 2.0
    half_h = (box.height * factor) / 2.0
    cx = (box.x0 + box.x1) / 2.0
    cy = (box.y0 + box.y1) / 2.0
    return Rect(
        max(int(cx - half_w), 0),
        max(int(cy - half_h), 0),
        min(int(cx + half_w), width),
        min(int(cy + half_h), height),
    )


class ConsisIDRunner:
    consumer_id: str = CONSUMER_ID

    def __init__(self, found: list[Shot] | None = None) -> None:
        self._shots = {one.label: one for one in (found if found is not None else shots())}
        self._baselines: dict[str, Artifact] = {}

    def cases(self) -> tuple[Case, ...]:
        return tuple(
            Case(
                name=f"consisid_align_face_{label}",
                consumer_id=self.consumer_id,
                tier=Tier.CONSUMER,
                fixture=self._shots[label].fixture,
                boundary=f"align_face|{label}",
                exact_bytes=True,
                rtol=0.0,
                atol=0.0,
                retained=("whole_reference_image",),
                ablations=(Ablation(primitive="whole_reference_image", expect_breaks=True),),
                measurements=("patch_divergence",),
                note="facexlib retinaface_resnet50 re-detects; whether a patch suffices is measured, not derived",
            )
            for label in self._shots
        )

    def _shot(self, case: Case) -> Shot:
        return self._shots[case.boundary.partition("|")[2]]

    def _scale(self, shot: Shot) -> float:
        width, height = shot.frame_wh
        longest = max(width, height)
        return RESIZE_LONG_EDGE / longest if longest > RESIZE_LONG_EDGE else 1.0

    def _footprint(self, shot: Shot, factor: float) -> Rect:
        kps = np.asarray(our_face(shot).kps, dtype=np.float32)
        return grown(analytic_footprint(kps, 336, shot.frame_wh), factor, shot.frame_wh)

    def retained_for(self, case: Case) -> RetainedState:
        frame = self._shot(case).frame.copy()
        return RetainedState(whole_reference_image=frame).priced(
            {"whole_reference_image": derivatives.lossless_bytes(frame)}
        )

    def baseline(self, case: Case) -> Artifact:
        if case.name not in self._baselines:
            self._baselines[case.name] = self._compute_baseline(case)
        return self._baselines[case.name]

    def _compute_baseline(self, case: Case) -> Artifact:
        shot = self._shot(case)
        return _artifact(case.boundary, align_through_facexlib(resize_numpy_image_long(shot.frame)))

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        pixels = retained.pixels("whole_reference_image")
        return _artifact(case.boundary, align_through_facexlib(resize_numpy_image_long(pixels)))

    def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState:
        return retained.without(ablation.primitive)

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        if name != "patch_divergence":
            raise KeyError(f"{self.consumer_id} has no measurement called {name!r}")

        shot = self._shot(case)
        against = self.baseline(case)
        if against.values is None:
            raise ValueError("baseline produced no values")
        wanted = np.asarray(against.values, dtype=np.int64)
        factor = self._scale(shot)

        reports: list[str] = []
        worst_fraction = 0.0
        for label, margin in (("arcface@336", 1.0), (f"{START_MARGIN:g}x arcface", START_MARGIN)):
            box = self._footprint(shot, margin)
            for how, pixels in (
                ("crop-then-resize", scaled(shot.frame[box.y0 : box.y1, box.x0 : box.x1], factor)),
                ("resize-then-crop", self._from_resized(shot, box, factor)),
            ):
                try:
                    produced = align_through_facexlib(pixels).astype(np.int64)
                except (ValueError, TypeError, IndexError) as problem:
                    reports.append(f"{label}/{how}: NOT MEASURED -- {type(problem).__name__}: {problem}")
                    continue
                differing = int(np.count_nonzero(wanted != produced))
                fraction = differing / wanted.size
                worst_fraction = max(worst_fraction, fraction)
                reports.append(
                    f"{label}/{how}: {differing:,} of {wanted.size:,} px ({fraction:.1%}), "
                    f"worst {int(np.max(np.abs(wanted - produced)))}"
                )

        return Measurement(
            name=name,
            unit="fraction_of_pixels_differing",
            value=worst_fraction,
            basis=(
                "facexlib retinaface_resnet50 + FFHQ-512 warp over each candidate patch, against the same "
                "path over the whole picture; both resample orderings tried"
            ),
            detail="; ".join(reports),
        )

    def _from_resized(self, shot: Shot, box: Rect, factor: float) -> UInt8Array:
        resized = resize_numpy_image_long(shot.frame)
        edges = [round(one * factor) for one in box.as_tuple()]
        return np.asarray(resized[edges[1] : edges[3], edges[0] : edges[2]], dtype=np.uint8)
