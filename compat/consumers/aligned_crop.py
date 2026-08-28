"""Can an aligned crop be rebuilt from a source patch instead of the frame?

This is the primitive the whole arcface family sits on. IPAdapter FaceID Plus
calls `face_align.norm_crop(img, kps, 336|256|224)`, InfiniteYou calls it at
112, and every recognition model behind them is fed the result. If the crop
can be rebuilt from a bounded patch of source pixels plus the keypoints, then
the patch and the keypoints are the durable primitives and the crop is derived
state. If it cannot, the crop has to be stored per size, per consumer, forever.

The patch extent is derived, not guessed. `estimate_norm` returns the affine M
mapping source to crop; inverting it and mapping the crop's four corners back
gives the source quad the warp reads from, and the integer box containing that
quad is the analytic bound.

How much MORE than the analytic bound is needed is not asserted here, because
a constant asserted is folklore with a threshold attached: it passes while it
is too generous and never locates the edge. `minimum_patch_extent` measures it
instead -- four independent binary searches inward, then the combination --
and the number goes in the evidence.

Geometry comes from two places and both are load-bearing. The synthetic frame
has known structure at every scale, so a wrong-but-plausible warp cannot land
on statistically identical pixels. The corpus frames carry keypoints a real
detector produced on a real photograph, which is the only way the measured
minimum describes faces rather than a fixture.

Ablations, and what each is asking:

    source_region_pixels   remove the patch        -> must break
    kps_source_px          remove the keypoints    -> must break
    patch_origin           remove the origin       -> must break
    derive_256_from_336    downscale the 336 crop  -> MUST break

The last is the real claim. Measured: 224 and 256 share a scale of 1.16529 and
differ only in x-translation, by exactly 16.0, because `estimate_norm` sends
any size divisible by 128 down a branch adding `diff_x = 8.0 * ratio`. So 256
is not a resized 224 nor a downscaled 336 -- same face size, wider frame,
shifted. If that ablation ever passes, the three-family reading of the
template is wrong and the storage design resting on it has to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
import numpy.typing as npt

from compat.assertions.arrays import compare
from compat.assertions.minimize import Rect, minimum_extent
from compat.contracts.case import Ablation, Artifact, Case, Fixture, Measurement, RetainedState, Tier
from compat.corpus import index as corpus
from compat.primitives import build

#: Sizes the pinned consumers ask `norm_crop` for: 112 InfiniteYou,
#: 224/256/336 IPAdapter FaceID Plus by variant.
SIZES: Final[tuple[int, ...]] = (112, 224, 256, 336)

CONSUMER_ID: Final[str] = "ipadapter_faceid_plus"

#: Corpus images used for real geometry. Two identities and both capture
#: paths: an ID document and a selfie went through different optics, and a
#: minimum measured on one framing is a claim about that framing.
CORPUS_IMAGES: Final[int] = 4


def estimate_norm(kps: npt.NDArray[np.float32], size: int) -> npt.NDArray[np.float64]:
    """Upstream's own alignment matrix. Imported inside: insightface is banned
    at module level in this tree, and rightly -- it pulls onnxruntime, which
    must never be reachable from the application's import graph."""
    from insightface.utils import face_align

    return np.asarray(face_align.estimate_norm(kps.copy(), size), dtype=np.float64)


def norm_crop(image: npt.NDArray[np.uint8], kps: npt.NDArray[np.float32], size: int) -> npt.NDArray[np.uint8]:
    """Upstream's own crop, on whatever pixels it is handed."""
    from insightface.utils import face_align

    out = face_align.norm_crop(image, landmark=kps.copy(), image_size=size)
    return np.asarray(out, dtype=np.uint8)


def shifted_to(matrix: npt.NDArray[np.float64], origin: tuple[int, int]) -> npt.NDArray[np.float64]:
    """The same alignment, expressed against a source whose origin moved.

    A source point p becomes q = p - origin in the patch, so M[A|t] applied to
    p equals [A | t + A@origin] applied to q. Exact in floating point: the
    linear part is untouched and only the translation column is rebuilt.

    This exists because the obvious alternative is NOT equivalent. Estimating
    the alignment afresh from patch-local keypoints runs Umeyama's SVD over
    coordinates of a different magnitude, and the accumulated rounding moves
    the matrix by around 1e-5 -- enough to push samples across `warpAffine`'s
    fixed-point interpolation boundary. Measured on a 4032x3024 corpus
    photograph: re-estimating differs from the baseline in 2 of 37632 pixels
    at 112 and 131 of 338688 at 336, each by one level, while translating
    differs in none. `local_reestimation_divergence` records that per case.
    """
    out = np.asarray(matrix, dtype=np.float64).copy()
    out[:, 2] = out[:, 2] + out[:, :2] @ np.array(origin, dtype=np.float64)
    return out


def warp(image: npt.NDArray[np.uint8], matrix: npt.NDArray[np.float64], size: int) -> npt.NDArray[np.uint8]:
    """`norm_crop`'s second line, with the matrix supplied rather than fitted.

    Same call as upstream -- `cv2.warpAffine(img, M, (size, size),
    borderValue=0.0)`, face_align.py at insightface@7fadd420c235 -- so a crop
    produced here and one produced by `norm_crop` differ only by the matrix.
    """
    out = cv2.warpAffine(image, matrix, (size, size), borderValue=0.0)
    return np.asarray(out, dtype=np.uint8)


def analytic_footprint(kps: npt.NDArray[np.float32], size: int, frame_wh: tuple[int, int], *, margin: int = 0) -> Rect:
    """The source box a `norm_crop` at `size` can read from.

    M maps source into crop space, so its inverse maps the crop's corners back
    out. The integer box containing those four points is the analytic bound;
    `margin` grows it, and exists so the measurement has somewhere to search
    from rather than so anybody has to believe a number.

    The box is clipped to the frame, which is not a detail: a face near an
    edge produces a quad that leaves the image, and upstream fills those
    samples from the border rather than from pixels that could be stored.
    """
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
    """One frame and one set of keypoints, with what produced them."""

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
    """Real frames with real detected keypoints, or an empty list.

    Empty rather than raising when the corpus or the pack is absent: a machine
    without them should report the corpus cases as UNSUPPORTED, which the case
    layer does, and not lose the synthetic ones as collateral.
    """
    from compat.producers import insightface_pass as producer

    if not corpus.KYC.is_dir():
        return []

    samples = corpus.scan_kyc()
    # One of each role for the first identities, by digest so the choice is
    # stable across machines and across directory orderings.
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
            continue
        best = max(faces, key=lambda face: float(face.det_score))
        if best.kps is None:
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
    """`norm_crop` from a bounded patch, against `norm_crop` from the frame."""

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
                if size == 256:
                    ablations.append(Ablation(primitive="derive_256_from_336", expect_breaks=True))
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
        """The durable state this case claims is sufficient.

        The patch carries its origin because keypoints recorded in source
        pixels mean nothing against a cropped array -- the origin is what makes
        the geometry portable, and dropping it is its own ablation.
        """
        size, geometry = self._parts(case)
        box = analytic_footprint(geometry.kps, size, geometry.frame_wh)
        return RetainedState(
            source_region_pixels=geometry.frame[box.y0 : box.y1, box.x0 : box.x1].copy(),
            patch_origin=(box.x0, box.y0),
            kps_source_px=geometry.kps.copy(),
        )

    def baseline(self, case: Case) -> Artifact:
        """The pinned upstream path, over the original frame."""
        size, geometry = self._parts(case)
        return _artifact(case.boundary, norm_crop(geometry.frame, geometry.kps, size))

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        """The same boundary from retained state, never touching the frame.

        Missing primitives raise through `RetainedState`, by name. A runner
        that quietly fell back to the original frame would report that every
        primitive is unnecessary, which is the one failure this suite cannot
        afford.
        """
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
            # The shortcut: translate the keypoints and fit again. Kept as a
            # runnable path so the divergence can be measured rather than
            # asserted -- see `shifted_to`.
            local = kps - np.array(origin, dtype=np.float32)
            return _artifact(case.boundary, norm_crop(patch, local, size))

        # The contract: fit in SOURCE space, then move the origin. The
        # keypoints are retained in source pixels precisely so this is
        # possible; patch-local keypoints cannot recover it.
        return _artifact(case.boundary, warp(patch, shifted_to(estimate_norm(kps, size), origin), size))

    def ablate(self, case: Case, retained: RetainedState, primitive: str) -> RetainedState:
        """Retained state with one primitive removed or degraded."""
        if primitive == "derive_256_from_336":
            return retained.replacing("derive_256_from_336", True)
        return retained.without(primitive)

    def _reestimation_divergence(
        self, case: Case, retained: RetainedState, size: int, geometry: Geometry
    ) -> Measurement:
        """How far the obvious-looking replay lands from the baseline.

        Retaining the keypoints is not by itself the contract: WHERE they are
        expressed decides whether the crop comes back. This runs the shortcut
        -- translate the keypoints into the patch, fit the alignment there --
        and reports how many pixels it costs, so the requirement to fit in
        source space is a measured quantity rather than a warning in a
        docstring.

        A result of zero is not a licence to take the shortcut. It means this
        geometry could not tell the two apart, which is a fact about the
        fixture; the corpus rows are the ones with the coordinate magnitudes
        that discriminate.
        """
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
            """One probe: crop to `box`, replay, compare bytes.

            Built from the geometry rather than from `retained` because the
            search is over EXTENTS, and re-slicing the already-sliced patch
            would measure the analytic bound's own margin instead of the
            warp's requirement.
            """
            trial = RetainedState(
                source_region_pixels=geometry.frame[box.y0 : box.y1, box.x0 : box.x1].copy(),
                patch_origin=(box.x0, box.y0),
                kps_source_px=geometry.kps.copy(),
            )
            try:
                produced = self.replay(case, trial)
            except (ValueError, TypeError, KeyError, cv2.error):
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
