from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from compat.consumers.aligned_crop import analytic_footprint, estimate_norm, shifted_to, warp
from compat.contracts.case import (
    Ablation,
    Artifact,
    Case,
    Float32Array,
    Measurement,
    RetainedState,
    Tier,
    UInt8Array,
)
from compat.corpus.loaded import Shot, our_face, our_recovery_face, shots, vendor_face
from compat.harness import provenance
from compat.producers import insightface_pass as producer
from compat.storage import derivatives, precision

PACK_ROOT: Final[Path] = producer.MODELS_ROOT


@dataclass(frozen=True)
class VendorSetup:
    consumer_id: str
    commit: str
    cited: tuple[str, ...]
    pack: str
    prepare_det_size: tuple[int, int]
    sweep: tuple[int, ...]
    det_size_ladder: tuple[int, ...]
    select: str
    embedding: str
    embedding_model: str
    crop_sizes: tuple[int, ...]
    kps_render: str
    allowed_modules: tuple[str, ...]
    retry_pad_scale: float

    @property
    def sweep_sizes(self) -> list[int]:
        if self.det_size_ladder:
            return list(self.det_size_ladder)
        if self.sweep:
            start, stop, step = self.sweep
            return list(range(start, stop, step))
        return [self.prepare_det_size[0]]


def vendor_setups() -> dict[str, VendorSetup]:
    manifest = provenance.load_manifest()
    out: dict[str, VendorSetup] = {}
    for row in manifest.get("consumers", []):
        setup = row.get("vendor_setup")

        if not setup or setup.get("kind") in {"whole_reference", "whole_reference_masked"}:
            continue
        out[row["id"]] = VendorSetup(
            consumer_id=row["id"],
            commit=row["commit"],
            cited=tuple(setup.get("cited", [])),
            pack=setup.get("pack", "antelopev2"),
            prepare_det_size=tuple(setup.get("prepare_det_size", [640, 640])),
            sweep=tuple(setup.get("sweep", [])),
            det_size_ladder=tuple(setup.get("det_size_ladder", [])),
            select=setup.get("select", "first"),
            embedding=setup.get("embedding", "raw"),
            embedding_model=setup.get("embedding_model", "glintr100"),
            crop_sizes=tuple(setup.get("crop_sizes", [])),
            kps_render=setup.get("kps_render", "none"),
            allowed_modules=tuple(setup.get("allowed_modules", [])),
            retry_pad_scale=float(setup.get("retry_pad_scale", 0.0)),
        )
    return out


def analysis_for(setup: VendorSetup) -> Any:
    from insightface.app import FaceAnalysis

    key = (setup.pack, setup.allowed_modules)
    if key not in _apps:
        app = FaceAnalysis(
            name=setup.pack,
            root=str(PACK_ROOT),
            providers=["CPUExecutionProvider"],
            allowed_modules=list(setup.allowed_modules) if setup.allowed_modules else None,
        )
        app.prepare(ctx_id=-1, det_size=tuple(setup.prepare_det_size))
        _apps[key] = app
    return _apps[key]


_apps: dict[tuple[str, tuple[str, ...]], Any] = {}


RECOGNITION: Final[dict[str, str]] = {
    "glintr100": "antelopev2/glintr100.onnx",
    "w600k_r50": "buffalo_l/w600k_r50.onnx",
}


def recognition_model(name: str) -> Any:
    from insightface.model_zoo import model_zoo

    if name not in _recognisers:
        where = PACK_ROOT / "models" / RECOGNITION[name]
        model = model_zoo.get_model(str(where), providers=["CPUExecutionProvider"])
        if model is None:
            raise FileNotFoundError(f"model_zoo could not load {where}")
        model.prepare(ctx_id=-1)
        _recognisers[name] = model
    return _recognisers[name]


_recognisers: dict[str, Any] = {}


def facexlib_arcface() -> Any:
    from facexlib.recognition import init_recognition_model

    if "facexlib_arcface" not in _recognisers:
        _recognisers["facexlib_arcface"] = init_recognition_model("arcface", device="cpu").eval()
    return _recognisers["facexlib_arcface"]


def embed_with(model_name: str, crop112: UInt8Array) -> Float32Array:
    if model_name != "facexlib_arcface":
        return np.asarray(recognition_model(model_name).get_feat(crop112), dtype=np.float32).reshape(-1)

    import torch

    tensor = torch.from_numpy(crop112).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
    tensor = 2 * tensor - 1
    with torch.no_grad():
        out = facexlib_arcface()(tensor.contiguous())[0]
    return np.asarray(out.numpy(), dtype=np.float32).reshape(-1)


def norm_crop112(frame: UInt8Array, kps: Float32Array) -> UInt8Array:
    from insightface.utils import face_align

    return np.asarray(face_align.norm_crop(frame, landmark=kps.copy(), image_size=112), dtype=np.uint8)


def pad_bgr(image: npt.NDArray[np.uint8], scale: float) -> tuple[npt.NDArray[np.uint8], tuple[int, int]]:
    import cv2

    pad = scale - 1.0
    height, width = image.shape[:2]
    top = bottom = int(height * pad)
    left = right = int(width * pad)
    out = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return np.asarray(out, dtype=np.uint8), (left, top)


def detect_for(setup: VendorSetup, shot: Shot) -> Any:
    return vendor_face(
        shot,
        pack=setup.pack,
        allowed_modules=setup.allowed_modules,
        sizes=tuple(setup.sweep_sizes),
        select=setup.select,
        rebuild=lambda frame: vendor_detect(setup, frame),
    )


def vendor_detect(setup: VendorSetup, image: npt.NDArray[np.uint8]) -> Any:
    app = analysis_for(setup)
    found: list[Any] = []

    for size in setup.sweep_sizes:
        if setup.det_size_ladder:
            app.prepare(ctx_id=-1, det_size=(size, size))
        else:
            app.det_model.prepare(-1, input_size=(size, size))
            if tuple(app.det_model.input_sizes) != ((size, size),):
                raise ValueError(
                    f"{setup.consumer_id}: asked the detector for {size} and it holds "
                    f"{app.det_model.input_sizes}; the sweep cannot be performed"
                )
        found = app.get(image)
        if found:
            break

    if not found and setup.retry_pad_scale:
        padded, (left, top) = pad_bgr(image, setup.retry_pad_scale)
        found = app.get(padded)
        offset = np.array([left, top], dtype=np.float32)
        for face in found:
            face.kps = face.kps - offset
            face.bbox = face.bbox - np.array([left, top, left, top], dtype=np.float32)

    if not found:
        raise ValueError(f"{setup.consumer_id}: no face at any of {setup.sweep_sizes}")

    if setup.select == "largest_bbox_area":
        return max(found, key=lambda one: (one.bbox[2] - one.bbox[0]) * (one.bbox[3] - one.bbox[1]))
    return found[0]


def upstream_draw_kps(consumer_id: str) -> Any:
    import math

    import cv2
    import PIL.Image

    from compat.harness import pinned_source

    if consumer_id not in _renderers:
        repo, commit, path = _kps_source(consumer_id)

        fn, proof = pinned_source.load_symbol(
            repo,
            commit,
            path,
            "draw_kps",
            {"np": np, "cv2": cv2, "math": math, "PIL": PIL, "Image": PIL.Image},
        )
        _renderers[consumer_id] = (fn, proof)
    return _renderers[consumer_id][0]


_renderers: dict[str, tuple[Any, Any]] = {}


def _kps_source(consumer_id: str) -> tuple[Path, str, str]:
    manifest = provenance.load_manifest()
    refs_root = (Path(__file__).resolve().parents[2] / manifest["refs_root"]).resolve()
    paths = {
        "instantid": "InstantID.py",
        "instantid_upstream": "pipeline_stable_diffusion_xl_instantid.py",
        "infiniteyou": "utils.py",
    }
    for row in manifest["consumers"]:
        if row["id"] == consumer_id:
            return provenance.clone_dir(refs_root, row["repo"]), row["commit"], paths[consumer_id]
    raise LookupError(f"{consumer_id!r} is not in the manifest")


def render_kps(
    consumer_id: str, style: str, height: int, width: int, kps: npt.NDArray[np.float32]
) -> npt.NDArray[np.uint8]:
    import PIL.Image

    fn = upstream_draw_kps(consumer_id)
    blank = np.zeros([height, width, 3], dtype=np.uint8)
    if style == "draw_kps_pil":
        return np.asarray(fn(PIL.Image.fromarray(blank), np.asarray(kps)), dtype=np.uint8)
    return np.asarray(fn(blank, np.asarray(kps)), dtype=np.uint8)


def _artifact(name: str, values: Float32Array | UInt8Array) -> Artifact:
    return Artifact(
        name=name,
        dtype=str(values.dtype),
        shape=tuple(int(one) for one in values.shape),
        sha256=producer.digest_array(values),
        values=values,
    )


def embedding_ablations(setup: VendorSetup) -> tuple[Ablation, ...]:
    return (
        Ablation(primitive="embedding_raw", expect_breaks=True),
        Ablation(
            primitive="embedding_raw",
            swap="stored_glintr100",
            expect_breaks=setup.embedding_model != "glintr100",
            kind="substitution",
        ),
        Ablation(
            primitive="embedding_raw",
            swap="half_precision",
            expect_breaks=True,
            kind="substitution",
        ),
    )


def crop_ablations() -> tuple[Ablation, ...]:
    return (
        Ablation(primitive="source_region_pixels", expect_breaks=True),
        Ablation(primitive="kps_source_px", expect_breaks=True),
        Ablation(primitive="patch_origin", expect_breaks=True),
        Ablation(
            primitive="patch_origin",
            swap="origin_at_zero",
            expect_breaks=True,
            kind="substitution",
        ),
        Ablation(
            primitive="source_region_pixels",
            swap="webp_encoded",
            expect_breaks=True,
            kind="substitution",
        ),
        Ablation(
            primitive="kps_source_px",
            swap="half_precision",
            expect_breaks=True,
            kind="substitution",
        ),
    )


def kps_ablations() -> tuple[Ablation, ...]:
    return (
        Ablation(primitive="kps_source_px", expect_breaks=True),
        Ablation(primitive="frame_dimensions", expect_breaks=True),
        Ablation(
            primitive="frame_dimensions",
            swap="preview_dimensions",
            expect_breaks=True,
            kind="substitution",
        ),
        Ablation(
            primitive="kps_source_px",
            swap="half_precision",
            expect_breaks=True,
            kind="substitution",
        ),
        Ablation(primitive="reference_pixels", expect_breaks=False),
    )


class FaceFamilyRunner:
    def __init__(self, setup: VendorSetup, found: list[Shot] | None = None) -> None:
        self.setup = setup
        self.consumer_id = setup.consumer_id
        self._shots: dict[str, Shot] = {one.label: one for one in (found if found is not None else shots())}
        self._baselines: dict[str, Artifact] = {}

    def cases(self) -> tuple[Case, ...]:
        out: list[Case] = []
        for label in self._shots:
            if self.setup.embedding in {"raw", "normed"}:
                out.append(
                    self._case(
                        label,
                        "embedding",
                        ("embedding_raw",),
                        embedding_ablations(self.setup),
                        measurements=("stored_vector_agreement",),
                    )
                )
            out.extend(
                self._case(
                    label,
                    f"crop@{size}",
                    ("source_region_pixels", "patch_origin", "kps_source_px"),
                    crop_ablations(),
                    measurements=("detection_equivalent_patch",),
                )
                for size in self.setup.crop_sizes
            )
            if self.setup.kps_render != "none":
                out.append(
                    self._case(
                        label,
                        "kps_render",
                        ("kps_source_px", "frame_dimensions"),
                        kps_ablations(),
                        measurements=("reference_pixels_unused",),
                    )
                )
        return tuple(out)

    def _case(
        self,
        label: str,
        kind: str,
        retained: tuple[str, ...],
        ablations: tuple[Ablation, ...],
        measurements: tuple[str, ...] = (),
    ) -> Case:
        return Case(
            name=f"{self.consumer_id}_{kind.replace('@', '')}_{label}",
            consumer_id=self.consumer_id,
            tier=Tier.CONSUMER,
            fixture=self._shots[label].fixture,
            boundary=f"{kind}|{label}",
            exact_bytes=True,
            rtol=0.0,
            atol=0.0,
            retained=retained,
            ablations=ablations,
            measurements=measurements,
            note=f"vendor setup at {self.setup.commit[:12]}; cited {'; '.join(self.setup.cited)}",
        )

    def _parts(self, case: Case) -> tuple[str, Shot]:
        kind, _, label = case.boundary.partition("|")
        return kind, self._shots[label]

    def retained_for(self, case: Case) -> RetainedState:
        kind, shot = self._parts(case)
        best = our_face(shot)

        if self.setup.retry_pad_scale:
            recovered = our_recovery_face(shot)
            if recovered is not None:
                best = recovered
        kps = np.asarray(best.kps, dtype=np.float32)

        if kind == "embedding":
            crop = norm_crop112(shot.frame, kps)
            return RetainedState(
                embedding_raw=embed_with(self.setup.embedding_model, crop),
                stored_glintr100=np.asarray(best.embedding, dtype=np.float32).reshape(-1),
            )
        if kind == "kps_render":
            return RetainedState(kps_source_px=kps.copy(), frame_dimensions=shot.frame_wh)

        size = int(kind.rsplit("@", 1)[1])
        box = analytic_footprint(kps, size, shot.frame_wh)
        return RetainedState(
            source_region_pixels=shot.frame[box.y0 : box.y1, box.x0 : box.x1].copy(),
            patch_origin=(box.x0, box.y0),
            kps_source_px=kps.copy(),
        )

    def baseline(self, case: Case) -> Artifact:
        if case.name not in self._baselines:
            self._baselines[case.name] = self._compute_baseline(case)
        return self._baselines[case.name]

    def _compute_baseline(self, case: Case) -> Artifact:
        from insightface.utils import face_align

        kind, shot = self._parts(case)
        face = detect_for(self.setup, shot)

        if kind == "embedding":
            if self.setup.embedding_model == "facexlib_arcface":
                raw = embed_with("facexlib_arcface", norm_crop112(shot.frame, np.asarray(face.kps, dtype=np.float32)))
            else:
                raw = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
            values = raw if self.setup.embedding == "raw" else raw / np.linalg.norm(raw)
            return _artifact(case.boundary, values.astype(np.float32))

        if kind == "kps_render":
            import PIL.Image

            fn = upstream_draw_kps(self.consumer_id)
            kps = np.asarray(face.kps, dtype=np.float32)

            if self.setup.kps_render == "draw_kps_pil":
                drawn = fn(PIL.Image.fromarray(shot.frame[:, :, ::-1]), kps)
            else:
                drawn = fn(shot.frame, kps)
            return _artifact(case.boundary, np.asarray(drawn, dtype=np.uint8))

        size = int(kind.rsplit("@", 1)[1])
        crop = face_align.norm_crop(shot.frame, landmark=np.asarray(face.kps, dtype=np.float32), image_size=size)
        return _artifact(case.boundary, np.asarray(crop, dtype=np.uint8))

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        kind, _shot = self._parts(case)

        if kind == "embedding":
            raw = retained.points("embedding_raw")
            values = raw if self.setup.embedding == "raw" else raw / np.linalg.norm(raw)
            return _artifact(case.boundary, values.astype(np.float32))

        if kind == "kps_render":
            kps = retained.points("kps_source_px")
            width, height = retained.pair("frame_dimensions")
            return _artifact(case.boundary, render_kps(self.consumer_id, self.setup.kps_render, height, width, kps))

        size = int(kind.rsplit("@", 1)[1])
        patch = retained.pixels("source_region_pixels")
        origin = retained.pair("patch_origin")

        kps = retained.points("kps_source_px")
        return _artifact(case.boundary, warp(patch, shifted_to(estimate_norm(kps, size), origin), size))

    def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState:
        if ablation.primitive == "reference_pixels" and not ablation.swap:
            return retained
        if ablation.swap == "stored_glintr100":
            return retained.replacing("embedding_raw", retained.points("stored_glintr100"))
        if ablation.swap == "origin_at_zero":
            return retained.replacing("patch_origin", (0, 0))
        if ablation.swap == "webp_encoded":
            return retained.replacing(
                "source_region_pixels", derivatives.encoded(retained.pixels("source_region_pixels"))[0]
            )
        if ablation.swap == "preview_dimensions":
            from vision import thumbs

            width, height = retained.pair("frame_dimensions")
            edge = thumbs.EDGES["preview"]
            scale = min(1.0, edge / max(width, height))
            return retained.replacing("frame_dimensions", (max(1, round(width * scale)), max(1, round(height * scale))))
        if ablation.swap == "half_precision":
            return retained.replacing(ablation.primitive, precision.half(retained.array(ablation.primitive)))
        return retained.without(ablation.primitive)

    def _stored_vector_agreement(self, case: Case, retained: RetainedState) -> Measurement:
        _kind, shot = self._parts(case)
        mine = retained.points("embedding_raw").reshape(-1)
        stored = retained.points("stored_glintr100").reshape(-1)
        left = float(np.linalg.norm(mine))
        right = float(np.linalg.norm(stored))
        agreement = float(np.dot(mine, stored) / (left * right)) if left and right else 0.0
        return Measurement(
            name="stored_vector_agreement",
            unit="cosine",
            value=agreement,
            basis=f"{self.setup.embedding_model} against the stored glintr100 vector, same norm_crop@112",
            detail=(
                f"{shot.label}: declared {self.setup.embedding_model}, cosine {agreement:+.4f} "
                f"(|mine|={left:.3f}, |stored|={right:.3f})"
            ),
        )

    def _detection_equivalent_patch(self, case: Case) -> Measurement:
        _kind, shot = self._parts(case)
        whole = detect_for(self.setup, shot)
        want = np.asarray(whole.kps, dtype=np.float32)
        width, height = shot.frame_wh
        cx, cy = float(want[:, 0].mean()), float(want[:, 1].mean())

        for fraction in (0.25, 0.4, 0.55, 0.7, 0.85, 1.0):
            if fraction >= 1.0:
                x0, y0, x1, y1 = 0, 0, width, height
            else:
                half_w, half_h = width * fraction / 2, height * fraction / 2
                x0, y0 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
                x1, y1 = min(width, int(cx + half_w)), min(height, int(cy + half_h))
            patch = shot.frame[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            try:
                found = vendor_detect(self.setup, np.ascontiguousarray(patch))
            except ValueError:
                continue
            shifted = np.asarray(found.kps, dtype=np.float32) + np.asarray([x0, y0], dtype=np.float32)
            if np.array_equal(shifted, want):
                return Measurement(
                    name="detection_equivalent_patch",
                    unit="fraction_of_frame",
                    value=float(patch.size) / float(shot.frame.size),
                    basis="smallest searched patch whose re-detection equals detection on the whole frame",
                    detail=(
                        f"{patch.shape[1]}x{patch.shape[0]} of {width}x{height} reproduces the vendor's "
                        f"keypoints exactly"
                    ),
                )
        return Measurement(
            name="detection_equivalent_patch",
            unit="fraction_of_frame",
            value=None,
            basis="smallest searched patch whose re-detection equals detection on the whole frame",
            detail="no searched patch, up to the whole frame, reproduced the vendor's keypoints",
        )

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        if name == "stored_vector_agreement":
            return self._stored_vector_agreement(case, retained)
        if name == "detection_equivalent_patch":
            return self._detection_equivalent_patch(case)
        if name != "reference_pixels_unused":
            raise KeyError(f"{self.consumer_id} has no measurement called {name!r}")

        _kind, shot = self._parts(case)
        kps = retained.points("kps_source_px")
        width, height = retained.pair("frame_dimensions")

        vendor = self.baseline(case)
        if vendor.values is None:
            raise ValueError("vendor baseline produced no values")
        theirs = np.asarray(vendor.values, dtype=np.uint8)
        ours = render_kps(self.consumer_id, self.setup.kps_render, height, width, kps)
        differing = int(np.count_nonzero(theirs != ours))
        return Measurement(
            name=name,
            unit="pixels_differing",
            value=float(differing),
            basis="vendor draw_kps over the photograph against draw_kps over zeros at the same shape",
            detail=(
                f"{differing} of {theirs.size} pixels differ; source frame "
                f"{shot.frame_wh[0]}x{shot.frame_wh[1]} contributes only its shape"
            ),
        )


def all_runners(found: list[Shot] | None = None) -> list[FaceFamilyRunner]:
    ready = found if found is not None else shots()
    return [FaceFamilyRunner(setup, ready) for setup in vendor_setups().values()]
