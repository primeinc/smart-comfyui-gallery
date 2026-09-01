from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt

from compat.assertions.arrays import compare
from compat.assertions.minimize import Rect, minimum_extent
from compat.contracts.case import Ablation, Artifact, Case, Fixture, Measurement, RetainedState, Tier, note_skip
from compat.corpus import index as corpus
from compat.primitives import build

SIZES: Final[tuple[int, ...]] = (112, 224, 256, 336)


CONSUMER_ID: Final[str] = "aligned_crop"


CORPUS_IMAGES: Final[int] = 4


def estimate_norm(kps: npt.NDArray[np.float32], size: int) -> npt.NDArray[np.float64]:
    from insightface.utils import face_align

    return np.asarray(face_align.estimate_norm(kps.copy(), size), dtype=np.float64)


def norm_crop(image: npt.NDArray[np.uint8], kps: npt.NDArray[np.float32], size: int) -> npt.NDArray[np.uint8]:
    from insightface.utils import face_align

    out = face_align.norm_crop(image, landmark=kps.copy(), image_size=size)
    return np.asarray(out, dtype=np.uint8)


def shifted_to(matrix: npt.NDArray[np.float64], origin: tuple[int, int]) -> npt.NDArray[np.float64]:
    out = np.asarray(matrix, dtype=np.float64).copy()
    out[:, 2] = out[:, 2] + out[:, :2] @ np.array(origin, dtype=np.float64)
    return out


def warp(image: npt.NDArray[np.uint8], matrix: npt.NDArray[np.float64], size: int) -> npt.NDArray[np.uint8]:
    out = cv2.warpAffine(image, matrix, (size, size), borderValue=0.0)
    return np.asarray(out, dtype=np.uint8)


def analytic_footprint(kps: npt.NDArray[np.float32], size: int, frame_wh: tuple[int, int], *, margin: int = 0) -> Rect:
    inverse = cv2.invertAffineTransform(estimate_norm(kps, size))
    corners = np.array([[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]], dtype=np.float64)
    mapped = np.hstack([corners, np.ones((4, 1), dtype=np.float64)]) @ inverse.T

    width, height = frame_wh
    x0 = int(np.floor(mapped[:, 0].min())) - margin
    y0 = int(np.floor(mapped[:, 1].min())) - margin
    x1 = int(np.ceil(mapped[:, 0].max())) + margin
    y1 = int(np.ceil(mapped[:, 1].max())) + margin
    return Rect(max(x0, 0), max(y0, 0), min(x1, width), min(y1, height))


def _artifact(name: str, crop: npt.NDArray[np.uint8]) -> Artifact:
    return Artifact(
        name=name,
        dtype=str(crop.dtype),
        shape=tuple(int(one) for one in crop.shape),
        sha256=build.digest(crop),
        values=crop,
    )


@dataclass(frozen=True)
class Geometry:
    label: str
    frame: npt.NDArray[np.uint8]
    kps: npt.NDArray[np.float32]
    fixture: Fixture

    @property
    def frame_wh(self) -> tuple[int, int]:
        height, width = self.frame.shape[:2]
        return int(width), int(height)


def synthetic_geometry() -> Geometry:
    frame = build.frame()
    return Geometry(
        label="synthetic",
        frame=frame,
        kps=build.keypoints(),
        fixture=Fixture(
            name="primitive_frame",
            path="compat/primitives/build.py::frame",
            sha256=build.digest(frame),
            kind="synthetic_bgr_frame",
            note="generated, seed 20260828; geometry known, no detector involved",
        ),
    )


def corpus_geometry(limit: int = CORPUS_IMAGES) -> list[Geometry]:
    from compat.producers import insightface_pass as producer

    if not corpus.KYC.is_dir():
        return []

    samples = corpus.scan_kyc()

    buckets: dict[tuple[str, str], list[corpus.Sample]] = {}
    for one in samples:
        buckets.setdefault((one.identity, one.role), []).append(one)
    chosen = [min(buckets[key], key=lambda one: one.sha256) for key in sorted(buckets)][:limit]

    app = producer.analysis()
    out: list[Geometry] = []
    for one in chosen:
        frame, sha = producer.decode(Path(one.path))
        faces = app.get(frame)
        if not faces:
            note_skip(CONSUMER_ID, one.path, "our producer found no face in this photograph")
            continue
        best = max(faces, key=lambda face: float(face.det_score))
        if best.kps is None:
            note_skip(CONSUMER_ID, one.path, "the detected face carries no keypoints")
            continue
        out.append(
            Geometry(
                label=f"{one.identity}_{one.role}",
                frame=frame,
                kps=np.asarray(best.kps, dtype=np.float32),
                fixture=Fixture(
                    name=f"corpus_{one.identity}_{one.role}",
                    path=one.path,
                    sha256=sha,
                    kind="corpus_photograph",
                    note=f"{corpus.LICENCE}, not vendored; kps from antelopev2 SCRFD at det_size 640",
                ),
            )
        )
    return out


class AlignedCropRunner:
    consumer_id: str = CONSUMER_ID

    def __init__(self, geometries: list[Geometry] | None = None) -> None:
        self._geometries: dict[str, Geometry] = {}
        for one in geometries if geometries is not None else [synthetic_geometry(), *corpus_geometry()]:
            self._geometries[one.label] = one

    def cases(self) -> tuple[Case, ...]:
        out: list[Case] = []
        for label, geometry in self._geometries.items():
            for size in SIZES:
                ablations = [
                    Ablation(primitive="source_region_pixels", expect_breaks=True),
                    Ablation(primitive="kps_source_px", expect_breaks=True),
                    Ablation(primitive="patch_origin", expect_breaks=True),
                ]
                out.append(
                    Case(
                        name=f"aligned_crop_{size}_{label}",
                        consumer_id=self.consumer_id,
                        tier=Tier.PRIMITIVE,
                        fixture=geometry.fixture,
                        boundary=f"norm_crop@{size}|{label}",
                        exact_bytes=True,
                        rtol=0.0,
                        atol=0.0,
                        retained=("source_region_pixels", "patch_origin", "kps_source_px"),
                        ablations=tuple(ablations),
                        measurements=("minimum_patch_extent", "local_reestimation_divergence"),
                        note="the warp is deterministic, so anything short of byte equality is a real difference",
                    )
                )
        return tuple(out)

    def _parts(self, case: Case) -> tuple[int, Geometry]:
        head, _, label = case.boundary.partition("|")
        return int(head.rsplit("@", 1)[1]), self._geometries[label]

    def retained_for(self, case: Case) -> RetainedState:
        size, geometry = self._parts(case)
        box = analytic_footprint(geometry.kps, size, geometry.frame_wh)
        return RetainedState(
            source_region_pixels=geometry.frame[box.y0 : box.y1, box.x0 : box.x1].copy(),
            patch_origin=(box.x0, box.y0),
            kps_source_px=geometry.kps.copy(),
        )

    def baseline(self, case: Case) -> Artifact:
        size, geometry = self._parts(case)
        return _artifact(case.boundary, norm_crop(geometry.frame, geometry.kps, size))

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        size, _ = self._parts(case)

        if retained.flag("derive_256_from_336"):
            bigger = self.replay(
                Case(
                    name=case.name,
                    consumer_id=case.consumer_id,
                    tier=case.tier,
                    fixture=case.fixture,
                    boundary=case.boundary.replace(f"@{size}|", "@336|"),
                ),
                retained.without("derive_256_from_336"),
            )
            if bigger.values is None:
                raise ValueError("the 336 replay produced no values to downscale")
            shrunk = cv2.resize(np.asarray(bigger.values, dtype=np.uint8), (size, size), interpolation=cv2.INTER_AREA)
            return _artifact(case.boundary, np.asarray(shrunk, dtype=np.uint8))

        patch = retained.pixels("source_region_pixels")
        origin = retained.pair("patch_origin")
        kps = retained.points("kps_source_px")

        if retained.flag("reestimate_from_local_kps"):
            local = kps - np.array(origin, dtype=np.float32)
            return _artifact(case.boundary, norm_crop(patch, local, size))

        return _artifact(case.boundary, warp(patch, shifted_to(estimate_norm(kps, size), origin), size))

    def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState:
        return retained.without(ablation.primitive)

    def _reestimation_divergence(
        self, case: Case, retained: RetainedState, size: int, geometry: Geometry
    ) -> Measurement:
        against = self.baseline(case)
        shortcut = self.replay(case, retained.replacing("reestimate_from_local_kps", True))
        if against.values is None or shortcut.values is None:
            raise ValueError("baseline or shortcut produced no values")

        left = np.asarray(against.values, dtype=np.int64)
        right = np.asarray(shortcut.values, dtype=np.int64)
        differing = int(np.count_nonzero(left != right))
        worst = int(np.max(np.abs(left - right))) if differing else 0

        origin = retained.pair("patch_origin")
        kps = retained.points("kps_source_px")
        refitted = estimate_norm(kps - np.array(origin, dtype=np.float32), size)
        matrix_gap = float(np.max(np.abs(refitted - shifted_to(estimate_norm(kps, size), origin))))

        return Measurement(
            name="local_reestimation_divergence",
            unit="pixels_differing",
            value=float(differing),
            basis=(
                "Umeyama refitted on patch-local keypoints against the source-space fit translated to the "
                "patch origin; both warped through cv2.warpAffine at borderValue=0.0"
            ),
            detail=(
                f"refitting diverges in {differing} of {left.size} pixels (worst {worst} level"
                f"{'s' if worst != 1 else ''}), max|dM| {matrix_gap:.3e}; "
                f"frame {geometry.frame_wh[0]}x{geometry.frame_wh[1]}, origin {origin}"
            ),
        )

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        size, geometry = self._parts(case)

        if name == "local_reestimation_divergence":
            return self._reestimation_divergence(case, retained, size, geometry)
        if name != "minimum_patch_extent":
            raise KeyError(f"{self.consumer_id} has no measurement called {name!r}")

        against = self.baseline(case)
        if against.values is None:
            raise ValueError("baseline produced no values")
        wanted = np.asarray(against.values, dtype=np.uint8)
        start = analytic_footprint(geometry.kps, size, geometry.frame_wh)

        def reproduces(box: Rect) -> bool:
            trial = RetainedState(
                source_region_pixels=geometry.frame[box.y0 : box.y1, box.x0 : box.x1].copy(),
                patch_origin=(box.x0, box.y0),
                kps_source_px=geometry.kps.copy(),
            )
            try:
                produced = self.replay(case, trial)
            except cv2.error:
                return False
            if produced.values is None:
                return False
            return compare(
                wanted, np.asarray(produced.values, dtype=np.uint8), exact_bytes=True, rtol=0.0, atol=0.0
            ).equal

        found = minimum_extent(reproduces, start)
        margins = ", ".join(f"{one.side}+{one.max_inset}" for one in found.per_side)
        return Measurement(
            name=name,
            unit="pixels_saveable_per_side",
            value=found.saved_fraction,
            basis=(
                f"four-sided binary search inward from the analytic bound {start.as_tuple()}, "
                f"{found.probes} replay probes, byte-exact comparison"
            ),
            detail=(
                f"analytic {start.width}x{start.height}; per-side slack {margins}; "
                f"combined {found.combined.width}x{found.combined.height} "
                f"{'holds' if found.combined_holds else 'DOES NOT HOLD'}; "
                f"saved {found.saved_pixels} of {start.area} px ({found.saved_fraction:.1%})"
                + (f"; walked back {list(found.walked_back)}" if found.walked_back else "")
            ),
        )
