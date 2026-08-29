"""ConsisID: a second detector runs over the pixels, so it must be measured.

Every other face consumer in this population reuses the keypoints insightface
produced. ConsisID does not. `process_face_embeddings`
(diffusers@c1bf18c92c62, pipelines/consisid/consisid_utils.py:148-170) takes
the image twice:

    app.get(image_bgr)                       insightface -> embedding + kps
    face_helper_1.read_image(image_bgr)      facexlib, its OWN retinaface
    face_helper_1.get_face_landmarks_5(only_center_face=True)
    face_helper_1.align_warp_face()          -> align_face, 512x512 RGB

The second path is why this case exists. facexlib re-detects with
`retinaface_resnet50` and aligns to the FFHQ 512 template, so whether a
retained patch of source pixels serves it is not a geometric question with an
analytic answer -- a detector either finds the face in those pixels or it does
not, and the only way to know is to hand it the patch and look.

Two further details from the same file, both load-bearing:

    resize_numpy_image_long(image, 1024)   consisid_utils.py:265
        the whole picture is scaled to a long edge of 1024 BEFORE either
        detector runs, so a patch replayed at original scale presents a face
        of a different pixel size and is not the same input.

    face_kps falls back to facexlib        consisid_utils.py:165-166
        when insightface finds nothing, the keypoints come from the OTHER
        detector -- so the two are not interchangeable and that fallback is a
        real branch rather than a safety net.

The boundary stops at `align_face`. Everything past it is EVA-CLIP, whose
input is exactly that array: if the 512 crop reproduces byte-for-byte then so
does what the vision tower makes of it, and downloading a multi-gigabyte
encoder to re-derive that would measure the encoder rather than the storage
contract.

MEASURED RESULT, and it is not the flat "a patch never works" this file first
claimed from a single image. `patch_divergence` tries four strategies on every
corpus photograph, and the best of them -- a patch twice the arcface footprint,
taken AFTER the vendor's resize so the resample phase matches -- splits by
CAPTURE PATH:

    1_id_document   88.2% of the crop differs
    2_id_document   79.0%
    1_selfie         0.0%   (207 px of 786,432, worst 1)
    2_selfie         0.1%   (448 px, worst 3)

The tighter arcface@336 patch fails everywhere, 88-96%. So a face-shaped
region CAN carry this consumer when the face is large in frame, and cannot
when it is small -- an ID document puts the face in a corner of a document,
and cutting to the face removes context retinaface uses. That is the axis the
corpus's id/selfie split exists to expose, and one photograph would have
reported either half of it as the whole answer.

The durable contract therefore stays "retain the picture": a rule that holds
for half the capture paths is not a rule. The case below is shaped to say
that, and the substitutions record what the alternatives actually cost.
"""

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

#: consisid_utils.py:265. The long edge the whole picture is scaled to before
#: anything detects. Not a detail: it fixes the pixel size of the face both
#: detectors are shown.
RESIZE_LONG_EDGE: Final[int] = 1024

#: FaceRestoreHelper's own defaults, as ConsisID constructs it.
FACE_SIZE: Final[int] = 512

#: The generous face-shaped patch the substitution ablation offers: twice the
#: arcface footprint. Not a claim that this is enough -- the measurement shows
#: it is not, which is the whole point of offering the larger one.
START_MARGIN: Final[float] = 2.0


def resize_numpy_image_long(image: UInt8Array, long_edge: int = RESIZE_LONG_EDGE) -> UInt8Array:
    """ConsisID's own pre-resize, copied at consisid_utils.py:42-62.

    INTER_LANCZOS4, and only ever downward -- an image already inside the
    bound is returned untouched, which is what makes the transform idempotent
    and therefore safe to apply to a replayed patch.
    """
    import cv2

    height, width = image.shape[:2]
    if max(height, width) <= long_edge:
        return np.asarray(image, dtype=np.uint8)
    scale = long_edge / max(height, width)
    out = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LANCZOS4)
    return np.asarray(out, dtype=np.uint8)


def scaled(patch: UInt8Array, factor: float) -> UInt8Array:
    """A patch reduced by the same factor the whole picture would have been.

    LANCZOS4 to match `resize_numpy_image_long`, and a factor of exactly 1
    returns the array untouched: resizing by 1.0 still resamples, and a
    needless resample would change the very pixels the detector is shown.
    """
    import cv2

    if factor == 1.0:
        return np.asarray(patch, dtype=np.uint8)
    height, width = patch.shape[:2]
    out = cv2.resize(patch, (int(width * factor), int(height * factor)), interpolation=cv2.INTER_LANCZOS4)
    return np.asarray(out, dtype=np.uint8)


def face_helper() -> Any:
    """facexlib's `FaceRestoreHelper`, as ConsisID builds it.

    CPU deliberately: this is a torch model, and the CUDA path would put the
    evidence at the mercy of kernel selection for the same reason the ONNX
    side is pinned to the CPU provider.
    """
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper

    if "helper" not in _cached:
        _cached["helper"] = FaceRestoreHelper(
            upscale_factor=1, face_size=FACE_SIZE, crop_ratio=(1, 1), det_model="retinaface_resnet50", device="cpu"
        )
    return _cached["helper"]


_cached: dict[str, Any] = {}


def align_through_facexlib(image_bgr: UInt8Array) -> UInt8Array:
    """`align_face`, through facexlib's own detector and warp.

    The helper is stateful and upstream calls `clean_all()` first; skipping
    that would let a previous image's landmarks decide this one's crop, which
    is exactly the silent carry-over that makes a suite agree with itself and
    with nothing else.
    """
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
    """A footprint grown about its centre and clipped to the frame."""
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
    """facexlib's align_face, from a retained patch instead of the frame."""

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
                ablations=(
                    Ablation(primitive="whole_reference_image", expect_breaks=True),
                    # The arcface footprint serves the rest of the population
                    # and a patch twice its size is the most generous
                    # face-shaped state a store could offer.
                    Ablation(
                        primitive="whole_reference_image",
                        swap="arcface_footprint_only",
                        expect_breaks=True,
                        kind="substitution",
                    ),
                    Ablation(
                        primitive="whole_reference_image",
                        swap="generous_patch",
                        expect_breaks=True,
                        kind="substitution",
                    ),
                ),
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
        """The whole picture, because a patch was measured not to serve.

        The vendor's own pre-resize is NOT applied here. It is part of the
        consumer's path, not of the stored state: storing the reduced picture
        would bake in a long edge of 1024 that only this consumer wants, and
        every other whole-reference consumer asks for a different one.
        """
        frame = self._shot(case).frame.copy()
        return RetainedState(whole_reference_image=frame).priced(
            {"whole_reference_image": derivatives.lossless_bytes(frame)}
        )

    def baseline(self, case: Case) -> Artifact:
        """The vendor's own path, memoised per case.

        Invariant by definition -- it is what every ablation and every
        measurement is compared against, so two calls returning different
        artifacts would already be a defect. Memoising is therefore free
        correctness-wise and not free otherwise: `draw_kps` allocates several
        copies of a 4896x6528x3 canvas per call, and the measurement would
        otherwise trigger a second full render of it.
        """
        if case.name not in self._baselines:
            self._baselines[case.name] = self._compute_baseline(case)
        return self._baselines[case.name]

    def _compute_baseline(self, case: Case) -> Artifact:
        """The vendor's own path over the whole photograph."""
        shot = self._shot(case)
        return _artifact(case.boundary, align_through_facexlib(resize_numpy_image_long(shot.frame)))

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        """The vendor's own path over the retained pixels, never the source."""
        pixels = retained.pixels("whole_reference_image")
        return _artifact(case.boundary, align_through_facexlib(resize_numpy_image_long(pixels)))

    def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState:
        margins = {"arcface_footprint_only": 1.0, "generous_patch": START_MARGIN}
        if ablation.swap in margins:
            shot = self._shot(case)
            box = self._footprint(shot, margins[ablation.swap])
            return retained.replacing("whole_reference_image", shot.frame[box.y0 : box.y1, box.x0 : box.x1].copy())
        return retained.without(ablation.primitive)

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        """How far each face-shaped retention strategy lands from the truth.

        Recorded as a number rather than argued, because "a patch does not
        work" is the kind of claim that quietly becomes folklore. Two
        strategies are tried and both are reported, including the one that
        matches the vendor's resample phase -- so the result cannot be waved
        away as an artefact of resizing the crop separately.
        """
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
                    # The exception is reported, not interpreted: all three
                    # arrive when facexlib finds no face AND when this probe
                    # slices wrongly.
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
        """The same region taken AFTER the vendor's resize.

        This is the charitable variant: it shares the vendor's resample phase
        exactly, so any divergence it still shows is the detector's doing and
        not the interpolation's.
        """
        resized = resize_numpy_image_long(shot.frame)
        edges = [round(one * factor) for one in box.as_tuple()]
        return np.asarray(resized[edges[1] : edges[3], edges[0] : edges[2]], dtype=np.uint8)
