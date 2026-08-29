"""Every consumer that drives insightface's FaceAnalysis, run its own way.

Nine of the frozen population load the same library and then differ in ways
that decide the storage contract. Those differences are not paraphrased here:
each is copied from the vendor's own loader and extractor at the commit
`manifest.toml` pins, recorded in that row's `[consumers.vendor_setup]` with a
`cited` path and line range, and executed from there.

What actually differs between them, measured rather than assumed:

    sweep        InstantID retries detection at range(640, 128, -64);
                 IPAdapter and PuLID at range(640, 256, -64); InfiniteYou
                 builds a SEPARATE FaceAnalysis per size in [640, 320, 160];
                 UniPortrait does not sweep at all and instead pads by 1.25
                 with grey and retries once.
    select       InstantID, PuLID and InfiniteYou take the LARGEST face by
                 bbox area. IPAdapter takes face[0] -- detector order, not
                 size. On a single-face photograph these agree; on a group
                 they do not, and the stored row has to be able to serve both.
    embedding    InstantID and PuLID take `face.embedding` (RAW). IPAdapter
                 takes `normed_embedding` unless the model is portrait-unnorm.
                 UniPortrait takes NO embedding -- `allowed_modules` is
                 detection only. InfiniteYou takes facexlib's arcface, a
                 different model in a different space.
    pack         Everything names antelopev2 except UniPortrait, which passes
                 no `name=` and so gets insightface's DEFAULT_MP_NAME,
                 buffalo_l. Those two packs ship the SAME detector file --
                 `scrfd_10g_bnkps.onnx` and `det_10g.onnx` are byte-identical,
                 sha256 5838f7fe0536 -- so the keypoints agree and only the
                 recognition model differs, which UniPortrait never loads.

The baseline runs that vendor path over the original photograph. The replay
runs it over the retained state and nothing else. Where the vendor re-detects,
the replay re-detects too: a consumer that runs a detector over pixels cannot
be served by a stored keypoint unless the detector agrees, and whether it
agrees is measured, not argued.
"""

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
from compat.corpus.loaded import Shot, our_face, shots, vendor_face
from compat.harness import provenance
from compat.producers import insightface_pass as producer
from compat.storage import derivatives, precision

#: Where the packs live: the root `vision/weights.py` resolves for the
#: application and `producers/insightface_pass.py` resolves here, so the suite
#: and the product read the same files.
PACK_ROOT: Final[Path] = producer.MODELS_ROOT


@dataclass(frozen=True)
class VendorSetup:
    """One vendor's recommended configuration, as committed at its pin."""

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
        """The det_size values this vendor tries, in its own order.

        `sweep` is a three-element `range` spelling copied from the vendor, so
        the sizes are generated the way upstream generates them rather than
        listed out -- a list would silently stop matching if the range moved.
        """
        if self.det_size_ladder:
            return list(self.det_size_ladder)
        if self.sweep:
            start, stop, step = self.sweep
            return list(range(start, stop, step))
        return [self.prepare_det_size[0]]


def vendor_setups() -> dict[str, VendorSetup]:
    """Every `[consumers.vendor_setup]` in the manifest, typed."""
    manifest = provenance.load_manifest()
    out: dict[str, VendorSetup] = {}
    for row in manifest.get("consumers", []):
        setup = row.get("vendor_setup")
        # A whole-reference row carries a `kind` and belongs to
        # `whole_reference.py`; without this filter each would inherit the
        # `embedding = "raw"` default and generate a meaningless case.
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
    """FaceAnalysis built the way this vendor builds it.

    `allowed_modules` is passed through because UniPortrait restricts to
    detection, which changes which heads load and therefore what a Face
    carries. Providers are forced to CPU here, unlike the vendors' `ctx_id=0`:
    the CUDA provider defaults `cudnn_conv_algo_search` to Exhaustive
    (onnxruntime cuda_provider_options.h:24), which benchmarks kernels at
    runtime and can pick differently between runs. Evidence that moves with
    kernel selection is not evidence.
    """
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


#: Recognition models consumers ask for, beyond the one this application
#: stores. Each is a DIFFERENT space, measured rather than assumed from the
#: filename -- see the photomaker_v2 note in the manifest.
RECOGNITION: Final[dict[str, str]] = {
    "glintr100": "antelopev2/glintr100.onnx",
    "w600k_r50": "buffalo_l/w600k_r50.onnx",
}


def recognition_model(name: str) -> Any:
    """One ONNX recognition model, prepared on CPU."""
    from insightface.model_zoo import model_zoo

    if name not in _recognisers:
        where = PACK_ROOT / "models" / RECOGNITION[name]
        model = model_zoo.get_model(str(where), providers=["CPUExecutionProvider"])
        if model is None:
            # `get_model` returns None for a file it does not recognise rather
            # than raising, so an absent or wrong-format weight would otherwise
            # surface as an AttributeError three frames on.
            raise FileNotFoundError(f"model_zoo could not load {where}")
        model.prepare(ctx_id=-1)
        _recognisers[name] = model
    return _recognisers[name]


_recognisers: dict[str, Any] = {}


def facexlib_arcface() -> Any:
    """InfiniteYou's embedding model: facexlib's arcface, not insightface's.

    `init_recognition_model('arcface')` -- nodes.py:113. A torch model over a
    norm_crop@112 scaled to [-1, 1], which is a third space alongside
    glintr100 and w600k_r50.
    """
    from facexlib.recognition import init_recognition_model

    if "facexlib_arcface" not in _recognisers:
        _recognisers["facexlib_arcface"] = init_recognition_model("arcface", device="cpu").eval()
    return _recognisers["facexlib_arcface"]


def embed_with(model_name: str, crop112: UInt8Array) -> Float32Array:
    """The 512-d vector this consumer's own model produces from a 112 crop."""
    if model_name != "facexlib_arcface":
        return np.asarray(recognition_model(model_name).get_feat(crop112), dtype=np.float32).reshape(-1)

    import torch

    # utils.py:24-28, copied as arithmetic rather than as a claim: the crop
    # goes to [0,1], then to [-1,1], NCHW, and the model returns [512].
    tensor = torch.from_numpy(crop112).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
    tensor = 2 * tensor - 1
    with torch.no_grad():
        out = facexlib_arcface()(tensor.contiguous())[0]
    return np.asarray(out.numpy(), dtype=np.float32).reshape(-1)


def norm_crop112(frame: UInt8Array, kps: Float32Array) -> UInt8Array:
    """The 112 aligned crop every recognition model in this family consumes."""
    from insightface.utils import face_align

    return np.asarray(face_align.norm_crop(frame, landmark=kps.copy(), image_size=112), dtype=np.uint8)


def pad_bgr(image: npt.NDArray[np.uint8], scale: float) -> tuple[npt.NDArray[np.uint8], tuple[int, int]]:
    """UniPortrait's `pad_np_bgr_image`, copied at gradio_app.py:69-76.

    Grey (128,128,128) BORDER_CONSTANT, and the offset comes back so bbox and
    kps can be moved into original coordinates the way upstream moves them.
    """
    import cv2

    pad = scale - 1.0
    height, width = image.shape[:2]
    top = bottom = int(height * pad)
    left = right = int(width * pad)
    out = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return np.asarray(out, dtype=np.uint8), (left, top)


def detect_for(setup: VendorSetup, shot: Shot) -> Any:
    """This vendor's detection on this photograph, computed once.

    The memo key carries the pack, the allowed modules, the sweep and the
    selection rule, because all four change the answer -- IPAdapter's
    `face[0]` and InstantID's largest-by-area disagree the moment a
    photograph has two people in it.
    """
    return vendor_face(
        shot,
        pack=setup.pack,
        allowed_modules=setup.allowed_modules,
        sizes=tuple(setup.sweep_sizes),
        select=setup.select,
        rebuild=lambda frame: vendor_detect(setup, frame),
    )


def vendor_detect(setup: VendorSetup, image: npt.NDArray[np.uint8]) -> Any:
    """Detect exactly the way this vendor detects, and select its face.

    This is the part that cannot be shortcut. Every one of these consumers
    re-runs detection over whatever pixels it is handed, so the question the
    storage contract needs answered is whether the SAME detection comes back
    from a retained patch -- and that can only be found out by running it.
    """
    app = analysis_for(setup)
    found: list[Any] = []

    for size in setup.sweep_sizes:
        if setup.det_size_ladder:
            # InfiniteYou builds a separate app per size rather than mutating
            # one, so `prepare` is re-run to match its FaceDetector.
            app.prepare(ctx_id=-1, det_size=(size, size))
        else:
            app.det_model.input_size = (size, size)
        found = app.get(image)
        if found:
            break

    if not found and setup.retry_pad_scale:
        # UniPortrait's single retry: pad, detect again, then subtract the
        # offset so the geometry is back in original coordinates.
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
    """That vendor's OWN `draw_kps`, executed from its pinned blob.

    Not reimplemented here. A hand-copied renderer -- `limbSeq`, `stickwidth`,
    the colour table, the 0.6 dim, the circle radius -- tests the copy, not
    the contract, and would keep passing after upstream changed any of them.

    The two spellings really do differ and the difference is the reason this
    is parameterised at all:

        draw_kps_array   cubiq, ComfyUI_InstantID InstantID.py -- takes an
                         ndarray and reads `h, w, _ = image_pil.shape`
        draw_kps_pil     instantX-research/InstantID and bytedance's
                         InfiniteYou -- take a PIL image and read
                         `w, h = image_pil.size`, returning a PIL image

    Either way the reference contributes only its shape, which is what the
    `reference_pixels` ablation is there to establish rather than assert.
    """
    import math

    import cv2
    import PIL.Image

    from compat.harness import pinned_source

    if consumer_id not in _renderers:
        repo, commit, path = _kps_source(consumer_id)
        # `PIL` is bound to the PACKAGE, not to PIL.Image: upstream writes
        # `PIL.Image.fromarray(...)`, which binding the submodule would resolve
        # to `Image.Image.fromarray`.
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
    """Where this consumer's `draw_kps` is committed."""
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
    """Run that vendor's renderer over a blank of the given shape.

    A blank is what the claim is about: if the reference photograph mattered,
    handing the renderer zeros instead would change the output. It does not,
    and the case proves that by comparing this against the vendor's own call
    on the real photograph.
    """
    import PIL.Image

    fn = upstream_draw_kps(consumer_id)
    blank = np.zeros([height, width, 3], dtype=np.uint8)
    if style == "draw_kps_pil":
        return np.asarray(fn(PIL.Image.fromarray(blank), np.asarray(kps)), dtype=np.uint8)
    return np.asarray(fn(blank, np.asarray(kps)), dtype=np.uint8)


def _artifact(name: str, values: Float32Array | UInt8Array) -> Artifact:
    """One boundary artifact, at one of the two dtypes this suite compares.

    Narrowed rather than widened to `np.generic`: a crop is uint8 and an
    embedding is float32, and an artifact that arrived as float64 or int32
    would compare against a baseline of a different dtype and be reported as
    a divergence rather than as the type error it actually is.
    """
    return Artifact(
        name=name,
        dtype=str(values.dtype),
        shape=tuple(int(one) for one in values.shape),
        sha256=producer.digest_array(values),
        values=values,
    )


def embedding_ablations(setup: VendorSetup) -> tuple[Ablation, ...]:
    """Removing the vector must break; substituting OURS is the real question.

    `stored_glintr100_substituted` asks the thing the storage contract needs
    answered: does the vector this application already keeps serve this
    consumer? For a consumer that also uses glintr100 the answer must be yes,
    so the ablation is declared NOT to break. For PhotoMaker (w600k_r50) and
    InfiniteYou (facexlib arcface) it must break -- and if it ever does not,
    the two-spaces finding is wrong and one vector would suffice after all.

    The expectation is DERIVED FROM `embedding_model`, which is also what the
    experiment runs, so it cannot detect a wrong value: declare glintr100 for
    a consumer that really uses w600k_r50 and both sides of the comparison run
    glintr100, nothing breaks, and the expectation agrees. That is why every
    embedding case now carries `stored_vector_agreement` -- the cosine between
    the two vectors actually in the retained state. A break must be
    accompanied by vectors that genuinely differ, and `answer.py` reports the
    pair as UNCORROBORATED when it is not.
    """
    return (
        Ablation(primitive="embedding_raw", expect_breaks=True),
        Ablation(
            primitive="embedding_raw",
            swap="stored_glintr100",
            expect_breaks=setup.embedding_model != "glintr100",
            kind="substitution",
        ),
        # Width beside identity: a store that keeps the right
        # model's vector at half the width is a different, cheaper
        # answer, and nothing here had measured it.
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
        # The store kept the patch and forgot where it came from. Zero is
        # what a schema with no origin column can offer, and the crop is
        # warped from keypoints expressed relative to it.
        Ablation(
            primitive="patch_origin",
            swap="origin_at_zero",
            expect_breaks=True,
            kind="substitution",
        ),
        # The region as this application's own encoder keeps it.
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
        # The size of the picture the STORE keeps rather than the one the
        # detector saw: `vision/thumbs` caps its largest raster variant at 1440
        # and `draw_kps` renders onto a canvas of these dimensions.
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
        # Inverted on purpose: `draw_kps` reads the reference image for its
        # shape only, so substituting the pixels must leave the output
        # identical or the "no source pixels required" claim is false.
        Ablation(primitive="reference_pixels", expect_breaks=False),
    )


class FaceFamilyRunner:
    """One consumer's vendor path, baseline against replay.

    Constructed per consumer rather than looped inside one runner: the
    population is counted by `consumer_id`, and a single runner covering nine
    consumers would report one.
    """

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
        """The durable state OUR pipeline would have written.

        This is the whole exercise: the row this application keeps after one
        expensive pass, offered to a consumer that never sees the original
        file. It is built from our own producer, not from the vendor's
        detection, because that is what would actually be in the database.
        """
        kind, shot = self._parts(case)
        best = our_face(shot)
        kps = np.asarray(best.kps, dtype=np.float32)

        if kind == "embedding":
            # The vector that would have to be stored for THIS consumer: for
            # most, the glintr100 one already in the database; for PhotoMaker
            # and InfiniteYou, a second vector in a second space.
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
        """The vendor's own path, over the original photograph."""
        from insightface.utils import face_align

        kind, shot = self._parts(case)
        face = detect_for(self.setup, shot)

        if kind == "embedding":
            if self.setup.embedding_model == "facexlib_arcface":
                # InfiniteYou does not read face.embedding at all: it crops to
                # 112 and runs facexlib's arcface (utils.py:22-29).
                raw = embed_with("facexlib_arcface", norm_crop112(shot.frame, np.asarray(face.kps, dtype=np.float32)))
            else:
                raw = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
            values = raw if self.setup.embedding == "raw" else raw / np.linalg.norm(raw)
            return _artifact(case.boundary, values.astype(np.float32))

        if kind == "kps_render":
            import PIL.Image

            fn = upstream_draw_kps(self.consumer_id)
            kps = np.asarray(face.kps, dtype=np.float32)
            # The vendor is handed the REAL photograph here, exactly as it
            # would be in production. The replay is handed zeros. Equality
            # between them is the evidence that the pixels are unused.
            if self.setup.kps_render == "draw_kps_pil":
                drawn = fn(PIL.Image.fromarray(shot.frame[:, :, ::-1]), kps)
            else:
                drawn = fn(shot.frame, kps)
            return _artifact(case.boundary, np.asarray(drawn, dtype=np.uint8))

        size = int(kind.rsplit("@", 1)[1])
        crop = face_align.norm_crop(shot.frame, landmark=np.asarray(face.kps, dtype=np.float32), image_size=size)
        return _artifact(case.boundary, np.asarray(crop, dtype=np.uint8))

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        """The same boundary from retained state, never opening the source."""
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
        """One primitive removed, or -- for `reference_pixels` -- replaced.

        `reference_pixels` is not in the retained state at all, which is the
        claim: `draw_kps` never receives an image. Removing something absent
        cannot break anything, so this ablation returns the state unchanged
        and is declared `expect_breaks=False`. It passes only because the
        replay genuinely does not consult pixels.
        """
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
        """Cosine between the vector this consumer wants and the one we store.

        The number the substitution verdict should rest on. Both vectors are
        already in the retained state -- `embedding_raw` is what this
        consumer's own model produced, `stored_glintr100` is the gallery's
        row -- so measuring them costs one dot product and turns a claim
        derived from a manifest field into one derived from two arrays.

        Recorded whether or not the models are declared to differ: a consumer
        declared to share glintr100 whose vectors nonetheless disagree is the
        finding that the declaration is wrong, and it cannot be seen without
        the number.
        """
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

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        if name == "stored_vector_agreement":
            return self._stored_vector_agreement(case, retained)
        if name != "reference_pixels_unused":
            raise KeyError(f"{self.consumer_id} has no measurement called {name!r}")

        _kind, shot = self._parts(case)
        kps = retained.points("kps_source_px")
        width, height = retained.pair("frame_dimensions")

        # The vendor's own render, against a render that was handed nothing
        # but shape and keypoints. Identical output is the evidence that the
        # reference photograph contributes only its dimensions.
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
    """One runner per consumer carrying a vendor setup."""
    ready = found if found is not None else shots()
    return [FaceFamilyRunner(setup, ready) for setup in vendor_setups().values()]
