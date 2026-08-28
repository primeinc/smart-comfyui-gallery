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
    uso, instantcharacter, qwen, anystory, omnigen2
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
    Fixture,
    Measurement,
    RetainedState,
    Tier,
    UInt8Array,
)
from compat.corpus import index as corpus
from compat.harness import provenance
from compat.producers import insightface_pass as producer

#: Corpus photographs per consumer, both capture paths.
CORPUS_IMAGES: Final[int] = 4

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
            preprocess_from=str(setup.get("preprocess_from") or row["id"]),
            long_size_single=int(setup.get("long_size_single", 0)),
            exif_transpose=bool(setup.get("exif_transpose", False)),
            mask_mode=setup.get("mask_mode", ""),
        )
    return out


def uno_preprocess_ref(setup: WholeSetup) -> Any:
    """UNO's own `preprocess_ref`, executed from the pinned blob.

    UMO imports this same function rather than reimplementing it, so both
    consumers resolve to one source -- `preprocess_from` names which clone the
    bytes come out of, and the evidence records that commit rather than the
    importing repository's.
    """
    from PIL import Image

    from compat.harness import pinned_source

    manifest = provenance.load_manifest()
    refs_root = (Path(__file__).resolve().parents[2] / manifest["refs_root"]).resolve()
    owner = next(row for row in manifest["consumers"] if row["id"] == setup.preprocess_from)
    repo = provenance.clone_dir(refs_root, owner["repo"])

    key = (setup.preprocess_from, owner["commit"])
    if key not in _preprocessors:
        fn, _proof = pinned_source.load_symbol(
            repo, owner["commit"], "uno/flux/pipeline.py", "preprocess_ref", {"Image": Image}
        )
        _preprocessors[key] = fn
    return _preprocessors[key]


_preprocessors: dict[tuple[str, str], Any] = {}


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

    if setup.preprocess == "uno_preprocess_ref":
        image = uno_preprocess_ref(setup)(image, setup.long_size_single)
    else:
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.uint8)


@dataclass(frozen=True)
class Shot:
    label: str
    fixture: Fixture
    frame: UInt8Array

    @property
    def frame_wh(self) -> tuple[int, int]:
        height, width = self.frame.shape[:2]
        return int(width), int(height)


def shots(limit: int = CORPUS_IMAGES) -> list[Shot]:
    if not corpus.KYC.is_dir():
        return []
    buckets: dict[tuple[str, str], list[corpus.Sample]] = {}
    for one in corpus.scan_kyc():
        buckets.setdefault((one.identity, one.role), []).append(one)
    chosen = [min(buckets[key], key=lambda one: one.sha256) for key in sorted(buckets)][:limit]

    out: list[Shot] = []
    for one in chosen:
        frame, sha = producer.decode(Path(one.path))
        out.append(
            Shot(
                label=f"{one.identity}_{one.role}",
                fixture=Fixture(
                    name=f"corpus_{one.identity}_{one.role}",
                    path=one.path,
                    sha256=sha,
                    kind="corpus_photograph",
                    note=f"{corpus.LICENCE}, not vendored",
                ),
                frame=frame,
            )
        )
    return out


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
        app = producer.analysis()
        faces = app.get(shot.frame)
        if not faces:
            raise ValueError(f"no face in {shot.label}")
        best = max(faces, key=lambda one: (one.bbox[2] - one.bbox[0]) * (one.bbox[3] - one.bbox[1]))
        kps = np.asarray(best.kps, dtype=np.float32)
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
