"""Consumers that take the whole framed picture, not a face.

Seven of the population never detect a face at all. They accept a reference
image and condition on it, so the storage question flips: instead of asking
which measurements survive, it asks whether the picture itself has to.

The answer is the point of these cases, and it is a negative one. Every case
here carries an ablation that substitutes the FACE PATCH -- the bounded region
the arcface family is proven to reproduce from -- for the whole image. That
ablation is expected to break, and when it does it establishes something no
amount of face evidence could: a durable store holding only face regions
cannot serve this half of the population at all.

Vendor preprocessing is theirs, not ours:

    uno, umo    `preprocess_ref` from uno/flux/pipeline.py, loaded out of the
                pinned blob -- LANCZOS to the long edge, then a //16*16 centre
                crop. 512 for a single reference, 320 for several. UMO adds
                `ImageOps.exif_transpose` first; UNO's own inference.py does
                not, which is a real difference between two consumers sharing
                one function.
    uso         its OWN `preprocess_ref`, uso/flux/pipeline.py:72-87 --
                LANCZOS to a long edge, called from inference.py:147. This was
                recorded as `rgb_only` until the pinned source was read, which
                made every USO case compare a bare array against a resample
                the consumer really performs.
    instantcharacter
                pipeline.py:379 squares the reference to its longer edge, then
                :64-65 resamples to 384 and 768 for two encoder paths. Also
                recorded as `rgb_only` until the source was read.
    omnigen2    max_pixels 1024*1024 and max_input_image_side_length 1024,
                applied per reference at pipeline_omnigen2.py:265 and computed
                at image_processor.py:121-133, then floored to a multiple of
                16. Recorded as `rgb_only` until the source was read.
    qwen, anystory
                `Image.open(...).convert("RGB")` and nothing else. Qwen is
                worth naming: the file this manifest originally cited is prose
                with no code, and the runnable entrypoint hands a PIL image
                straight to the pipeline, so the vendor performs no reference
                preprocessing whatsoever.

Comparison is byte-exact on the RGB array. These transforms are deterministic
resamples with no model in them, so anything short of equality is a real
difference rather than numerical drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

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
from compat.harness import provenance
from compat.producers import insightface_pass as producer

#: The face size used when substituting a patch for the whole picture. 336 is
#: the largest crop any face-native consumer asks for, so the substitution is
#: the most generous face-only state the store could offer.
SUBSTITUTE_CROP: Final[int] = 336


@dataclass(frozen=True)
class WholeSetup:
    """One vendor's reference preprocessing, as committed at its pin."""

    consumer_id: str
    commit: str
    repo: str
    cited: tuple[str, ...]
    preprocess: str
    preprocess_from: str
    preprocess_path: str
    max_pixels: int
    max_side_length: int
    vae_scale_factor: int
    long_size_single: int
    exif_transpose: bool
    mask_mode: str


def whole_setups() -> dict[str, WholeSetup]:
    manifest = provenance.load_manifest()
    out: dict[str, WholeSetup] = {}
    for row in manifest.get("consumers", []):
        setup = row.get("vendor_setup") or {}
        if setup.get("kind") not in {"whole_reference", "whole_reference_masked"}:
            continue
        out[row["id"]] = WholeSetup(
            consumer_id=row["id"],
            commit=row["commit"],
            repo=row["repo"],
            cited=tuple(setup.get("cited", [])),
            preprocess=setup.get("preprocess", "rgb_only"),
            preprocess_path=str(setup.get("preprocess_path") or "uno/flux/pipeline.py"),
            max_pixels=int(setup.get("max_pixels") or 1024 * 1024),
            max_side_length=int(setup.get("max_side_length") or 1024),
            vae_scale_factor=int(setup.get("vae_scale_factor") or 16),
            preprocess_from=str(setup.get("preprocess_from") or row["id"]),
            long_size_single=int(setup.get("long_size_single", 0)),
            exif_transpose=bool(setup.get("exif_transpose", False)),
            mask_mode=setup.get("mask_mode", ""),
        )
    return out


def loaded_preprocess_ref(setup: WholeSetup) -> Any:
    """That vendor's own `preprocess_ref`, executed from the pinned blob.

    UMO imports UNO's rather than reimplementing it, so both resolve to one
    source; USO ships its OWN copy at `uso/flux/pipeline.py`. `preprocess_from`
    names the clone the bytes come out of and `preprocess_path` the file inside
    it, so the evidence records the commit that actually supplied the code.
    """
    from PIL import Image

    from compat.harness import pinned_source

    manifest = provenance.load_manifest()
    refs_root = (Path(__file__).resolve().parents[2] / manifest["refs_root"]).resolve()
    owner = next(row for row in manifest["consumers"] if row["id"] == setup.preprocess_from)
    repo = provenance.clone_dir(refs_root, owner["repo"])

    key = (setup.preprocess_from, owner["commit"], setup.preprocess_path)
    if key not in _preprocessors:
        fn, _proof = pinned_source.load_symbol(
            repo, owner["commit"], setup.preprocess_path, "preprocess_ref", {"Image": Image}
        )
        _preprocessors[key] = fn
    return _preprocessors[key]


_preprocessors: dict[tuple[str, str, str], Any] = {}


def to_pil(bgr: UInt8Array) -> Any:
    """A BGR frame as the RGB PIL image every one of these vendors opens."""
    from PIL import Image

    return Image.fromarray(bgr[:, :, ::-1])


def vendor_preprocess(setup: WholeSetup, bgr: UInt8Array) -> UInt8Array:
    """This vendor's reference preprocessing, over the pixels it is handed."""
    from PIL import ImageOps

    image = to_pil(bgr)
    if setup.exif_transpose:
        # UMO does this and UNO does not, on the same preprocess_ref. Applied
        # here for completeness: the arrays reaching this suite are already
        # upright, so it is a no-op and recorded as one rather than skipped.
        image = ImageOps.exif_transpose(image)

    if setup.preprocess == "preprocess_ref":
        image = loaded_preprocess_ref(setup)(image, setup.long_size_single)
    elif setup.preprocess == "omnigen2_max_pixels":
        # OmniGen2@18e6f9d5271b pipeline_omnigen2.py:265 preprocesses EVERY
        # reference; defaults max_pixels = 1024*1024 and
        # max_input_image_side_length = 1024 (:481-482). image_processor.py
        # :121-133 takes ratio = min(max_pixels_ratio, max_side_length_ratio,
        # 1.0) -- never upscales -- then floors each side to a multiple of
        # vae_scale_factor, which the pipeline sets to vae_scale_factor * 2
        # (:176) and the processor defaults to 16 (:52).
        image = image.convert("RGB")
        width, height = image.size
        by_side = setup.max_side_length / max(width, height)
        by_pixels = (setup.max_pixels / (width * height)) ** 0.5
        ratio = min(by_pixels, by_side, 1.0)
        step = setup.vae_scale_factor
        new_w = int(width * ratio) // step * step
        new_h = int(height * ratio) // step * step
        image = image.resize((new_w, new_h))
    elif setup.preprocess == "square_then_dual_resize":
        # InstantCharacter@5f5c49a98ba1 pipeline.py:379 squares the reference
        # to its LONGER edge, then :64-65 resamples it to 384 and to 768 for
        # two encoder paths. The boundary is the squared image: it is the last
        # artifact before the two branches, and both are derived from it.
        image = image.convert("RGB")
        longest = max(image.size)
        image = image.resize((longest, longest))
    else:
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def _artifact(name: str, values: UInt8Array) -> Artifact:
    return Artifact(
        name=name,
        dtype=str(values.dtype),
        shape=tuple(int(one) for one in values.shape),
        sha256=producer.digest_array(values),
        values=values,
    )


class WholeReferenceRunner:
    """One whole-reference consumer, whole picture against face patch."""

    def __init__(self, setup: WholeSetup, found: list[Shot] | None = None) -> None:
        self.setup = setup
        self.consumer_id = setup.consumer_id
        self._shots = {one.label: one for one in (found if found is not None else shots())}

    def cases(self) -> tuple[Case, ...]:
        return tuple(
            Case(
                name=f"{self.consumer_id}_reference_{label}",
                consumer_id=self.consumer_id,
                tier=Tier.CONSUMER,
                fixture=self._shots[label].fixture,
                boundary=f"reference_image|{label}",
                exact_bytes=True,
                rtol=0.0,
                atol=0.0,
                retained=("whole_reference_image",),
                ablations=(
                    Ablation(primitive="whole_reference_image", expect_breaks=True),
                    # The finding. A face patch is the most a face-only store
                    # could ever offer, and it must fail here.
                    Ablation(primitive="face_patch_substituted", expect_breaks=True, kind="substitution"),
                ),
                measurements=("bytes_whole_against_face_patch",),
                note=f"vendor setup at {self.setup.commit[:12]}; cited {'; '.join(self.setup.cited)}",
            )
            for label in self._shots
        )

    def _shot(self, case: Case) -> Shot:
        return self._shots[case.boundary.partition("|")[2]]

    def _face_patch(self, shot: Shot) -> UInt8Array:
        """The bounded region the face-native consumers are served from."""
        kps = np.asarray(our_face(shot).kps, dtype=np.float32)
        box = analytic_footprint(kps, SUBSTITUTE_CROP, shot.frame_wh)
        return shot.frame[box.y0 : box.y1, box.x0 : box.x1].copy()

    def retained_for(self, case: Case) -> RetainedState:
        return RetainedState(whole_reference_image=self._shot(case).frame.copy())

    def baseline(self, case: Case) -> Artifact:
        shot = self._shot(case)
        return _artifact(case.boundary, vendor_preprocess(self.setup, shot.frame))

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        pixels = retained.pixels("whole_reference_image")
        return _artifact(case.boundary, vendor_preprocess(self.setup, pixels))

    def ablate(self, case: Case, retained: RetainedState, primitive: str) -> RetainedState:
        if primitive == "face_patch_substituted":
            return retained.replacing("whole_reference_image", self._face_patch(self._shot(case)))
        return retained.without(primitive)

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        if name != "bytes_whole_against_face_patch":
            raise KeyError(f"{self.consumer_id} has no measurement called {name!r}")
        shot = self._shot(case)
        whole = retained.pixels("whole_reference_image")
        patch = self._face_patch(shot)
        ratio = whole.nbytes / patch.nbytes if patch.nbytes else 0.0
        return Measurement(
            name=name,
            unit="bytes_ratio",
            value=ratio,
            basis=f"raw uint8 extent of the whole picture against the norm_crop@{SUBSTITUTE_CROP} footprint",
            detail=(
                f"whole {whole.shape} {whole.nbytes:,} B against face patch {patch.shape} "
                f"{patch.nbytes:,} B -- {ratio:.1f}x more to retain the picture"
            ),
        )


def all_runners(found: list[Shot] | None = None) -> list[WholeReferenceRunner]:
    ready = found if found is not None else shots()
    return [WholeReferenceRunner(setup, ready) for setup in whole_setups().values()]
