"""Consumers that take the whole framed picture, not a face.

Seven of the population never detect a face at all. They accept a reference
image and condition on it, so the storage question flips: instead of asking
which measurements survive, it asks whether the picture itself has to.

WHAT THE REPLAY IS ALLOWED TO SEE
--------------------------------
The retained picture is not the decoded frame. It is that frame after a
lossless full-resolution encode and a decode back through this application's
own reader (`compat/storage/derivatives.lossless`), because a store holds
bytes and not objects -- and because a replay handed the same array the
baseline was computed from can only ever reproduce itself, which is a fact
about `numpy.copy` and not about storage. The encoded size is recorded, so the
answer can state what the minimum state COSTS.

Two substitutions carry the finding, and both are things this application can
already produce:

    face_patch_substituted
                the bounded region the arcface family is proven to reproduce
                from. The most a face-only store could ever offer.
    preview_derivative_substituted
                what the derived-image cache returns TODAY -- `vision/thumbs`
                `preview`, longest side 1440, WebP at quality 82. The most the
                store can serve without reopening the source file.

Both are expected to break, and when they do they establish something no
amount of face evidence could: neither a face-only store nor this
application's existing derivatives can serve this half of the population.

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
                worth naming: the file this manifest cites for it is prose
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
from compat.storage import derivatives

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


def _required(setup: dict[str, Any], consumer_id: str, field: str) -> str:
    """A vendor_setup field that changes what the baseline computes.

    Absent is an error, not a default. A default here is indistinguishable
    from a declaration, so a row nobody has finished reading compares a bare
    array against a resample the consumer really performs -- and passes.
    """
    held = setup.get(field)
    if held is None:
        raise KeyError(f"{consumer_id}: vendor_setup declares no {field!r}, and it decides what the baseline computes")
    return str(held)


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
            # `preprocess` decides what the baseline computes, so it is
            # REQUIRED: a default of `rgb_only` makes an omitted field and a
            # declared one the same input.
            preprocess=_required(setup, row["id"], "preprocess"),
            preprocess_path=str(setup.get("preprocess_path") or "uno/flux/pipeline.py"),
            # `or` would turn a DECLARED 0 into 1 MP silently. `.get(k, d)`
            # only defaults when the key is absent, which is the difference
            # between "upstream imposes no cap" and "nobody looked yet".
            max_pixels=int(setup.get("max_pixels", 1024 * 1024)),
            max_side_length=int(setup.get("max_side_length", 1024)),
            vae_scale_factor=int(setup.get("vae_scale_factor", 16)),
            preprocess_from=str(setup.get("preprocess_from") or row["id"]),
            long_size_single=int(setup.get("long_size_single", 0)),
            exif_transpose=bool(setup.get("exif_transpose", False)),
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
        # reference (max_pixels 1024*1024, side length 1024, :481-482);
        # image_processor.py:121-133 never upscales.
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
        # to its LONGER edge, then :64-65 resamples it to 384 and 768. The
        # boundary is the squared image, the last artifact before both.
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
        self._durables: dict[str, tuple[UInt8Array, int]] = {}
        self._previews: dict[str, tuple[UInt8Array, int]] = {}

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
                    Ablation(
                        primitive="whole_reference_image", swap="face_patch", expect_breaks=True, kind="substitution"
                    ),
                    # And the store's EXISTING durable artifact, which is what
                    # "without reopening the source" means for a picture.
                    Ablation(
                        primitive="whole_reference_image",
                        swap="preview_derivative",
                        expect_breaks=True,
                        kind="substitution",
                    ),
                ),
                measurements=("bytes_whole_against_face_patch", "bytes_to_retain_the_picture"),
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

    def _durable(self, shot: Shot) -> tuple[UInt8Array, int]:
        """The picture off a lossless artifact, memoised per shot.

        One encode per picture rather than one per case: seven consumers ask
        the same question of the same frame, and a 36 MB frame encodes in
        about a second.
        """
        if shot.label not in self._durables:
            self._durables[shot.label] = derivatives.lossless(shot.frame)
        return self._durables[shot.label]

    def _preview(self, shot: Shot) -> tuple[UInt8Array, int]:
        if shot.label not in self._previews:
            self._previews[shot.label] = derivatives.preview(shot.frame)
        return self._previews[shot.label]

    def retained_for(self, case: Case) -> RetainedState:
        pixels, encoded = self._durable(self._shot(case))
        # The lossless artifact's own byte count, not the decoded array's.
        # This is what a store would hold to return this picture.
        return RetainedState(whole_reference_image=pixels).priced({"whole_reference_image": encoded})

    def baseline(self, case: Case) -> Artifact:
        shot = self._shot(case)
        return _artifact(case.boundary, vendor_preprocess(self.setup, shot.frame))

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        pixels = retained.pixels("whole_reference_image")
        return _artifact(case.boundary, vendor_preprocess(self.setup, pixels))

    def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState:
        shot = self._shot(case)
        if ablation.swap == "face_patch":
            return retained.replacing("whole_reference_image", self._face_patch(shot))
        if ablation.swap == "preview_derivative":
            return retained.replacing("whole_reference_image", self._preview(shot)[0])
        return retained.without(ablation.primitive)

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        shot = self._shot(case)
        if name == "bytes_to_retain_the_picture":
            _pixels, lossless_bytes = self._durable(shot)
            _preview_pixels, preview_bytes = self._preview(shot)
            return Measurement(
                name=name,
                unit="bytes",
                value=float(lossless_bytes),
                basis="PNG at full resolution against the vision/thumbs preview this application already keeps",
                detail=(
                    f"{lossless_bytes:,} B lossless at {shot.frame.shape[1]}x{shot.frame.shape[0]} "
                    f"against {preview_bytes:,} B for the existing preview -- "
                    f"{lossless_bytes / preview_bytes:.0f}x, and the preview does not reproduce"
                ),
            )
        if name != "bytes_whole_against_face_patch":
            raise KeyError(f"{self.consumer_id} has no measurement called {name!r}")
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
