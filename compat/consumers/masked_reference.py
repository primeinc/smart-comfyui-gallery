"""AnyStory's mask, which the suite declared and never exercised.

`manifest.toml` gives anystory `boundary = ["subject_images", "subject_masks"]`
and `retained = ["whole_reference_image", "subject_mask"]`.
The mask half of that boundary was untested: `mask_mode` was parsed off the
setup and no case read it. `mask_mode()` below decides the decode here.

Upstream, junjiehe96/AnyStory@c38fef83a35512b2a00c072a95bc0ff56b003f93
inference.py:19-20, 23:

    subject_image = Image.open("assets/examples/1.webp").convert("RGB")
    subject_mask  = Image.open("assets/examples/1_mask.webp").convert("L")
    ... story_pipe.generate(..., images=[subject_image], masks=[subject_mask])

So the boundary is a PAIR, per subject: three-channel RGB and a single-channel
L mask, supplied as parallel lists. The multi-subject example at
inference.py:29-32 supplies two of each.

WHY THE MASK IS ITS OWN PRIMITIVE
---------------------------------
`mask_from_face_bbox` substitutes a rectangle covering the detected face for
the vendor's mask. It MUST break. A rectangle is what a face row can produce;
the vendor's mask is a subject segmentation covering hair, body and clothing,
and a face observation cannot reconstruct it at any precision. That ablation
is the difference between "we store faces" and "we store subjects".

Fixtures are the vendor's own committed webp pairs, extracted from the pinned
commit and never vendored into this repository.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

import proc
from compat.assertions.arrays import digest
from compat.contracts.case import (
    Ablation,
    Artifact,
    Case,
    Fixture,
    Measurement,
    RetainedState,
    Tier,
    UInt8Array,
    note_skip,
)
from compat.harness import provenance
from compat.storage import derivatives

CONSUMER_ID: Final[str] = "anystory"

#: (image, mask) pairs exactly as inference.py opens them.
PAIRS: Final[tuple[tuple[str, str, str], ...]] = (
    ("single", "assets/examples/1.webp", "assets/examples/1_mask.webp"),
    ("multi_1", "assets/examples/6_1.webp", "assets/examples/6_1_mask.webp"),
    ("multi_2", "assets/examples/6_2.webp", "assets/examples/6_2_mask.webp"),
)


@dataclass(frozen=True)
class Pair:
    """One vendor subject: its RGB image and its L mask."""

    label: str
    image: UInt8Array
    mask: UInt8Array
    image_sha256: str
    mask_sha256: str


def _blob(repo: Path, commit: str, path: str) -> bytes:
    code, out, _ = proc.run(
        ["git", "-C", str(repo), "cat-file", "blob", f"{commit}:{path}"], timeout=proc.LOCAL_SECONDS
    )
    if code != 0:
        raise ValueError(f"{path} is not in {repo.name} at {commit[:12]}")
    return out


def _repo() -> tuple[Path, str]:
    manifest = provenance.load_manifest()
    refs_root = (Path(__file__).resolve().parent.parent.parent / manifest["refs_root"]).resolve()
    for row in manifest.get("consumers", []):
        if row["id"] == CONSUMER_ID:
            return provenance.clone_dir(refs_root, row["repo"]), row["commit"]
    raise KeyError(f"{CONSUMER_ID} is not in the manifest")


#: The PIL modes this lane can decode a mask into. `mask_mode` is the
#: manifest's word for the channel count upstream's `.convert(...)` asks for,
#: and `_mask_flag` below is the only place it decides anything.
MASK_MODES: Final[tuple[str, ...]] = ("L", "RGB")


def _mask_flag(mode: str) -> int:
    """The cv2 decode flag for a PIL mode name."""
    import cv2

    return {"L": cv2.IMREAD_GRAYSCALE, "RGB": cv2.IMREAD_COLOR}[mode]


def mask_mode() -> str:
    """The declared mask mode, read rather than assumed.

    `whole_reference.WholeSetup` parsed this field and no case used it, so a
    wrong value was indistinguishable from a right one. Here it selects the
    decode, and an undeclared or unknown value raises instead of falling back
    to grayscale.
    """
    manifest = provenance.load_manifest()
    for row in manifest.get("consumers", []):
        if row["id"] == CONSUMER_ID:
            declared = str((row.get("vendor_setup") or {}).get("mask_mode") or "")
            if declared not in MASK_MODES:
                raise KeyError(
                    f"{CONSUMER_ID}: vendor_setup declares mask_mode={declared!r}; this lane decodes {list(MASK_MODES)}"
                )
            return declared
    raise KeyError(f"{CONSUMER_ID} is not in the manifest")


def pairs() -> list[Pair]:
    """The vendor's own subject/mask pairs, decoded upstream's way.

    `.convert("RGB")` and `.convert("L")` are upstream's calls, not ours: the
    mask's single channel is part of the contract, and decoding it as RGB
    would silently make a three-channel artifact upstream never builds.
    """
    import cv2

    repo, commit = _repo()
    declared = mask_mode()
    out: list[Pair] = []
    for label, image_path, mask_path in PAIRS:
        try:
            image_bytes = _blob(repo, commit, image_path)
            mask_bytes = _blob(repo, commit, mask_path)
        except ValueError as problem:
            # An unclonened ref, a bad commit and a moved asset all arrive
            # here as one ValueError. Which one it was is now recorded.
            note_skip(CONSUMER_ID, f"{image_path} + {mask_path}", f"not at the pinned commit: {problem}")
            continue
        colour = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        grey = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), _mask_flag(declared))
        if colour is None or grey is None:
            note_skip(CONSUMER_ID, f"{image_path} + {mask_path}", "cv2 could not decode the image or the mask")
            continue
        image = np.asarray(colour[:, :, ::-1], dtype=np.uint8)
        mask = np.asarray(grey, dtype=np.uint8)
        out.append(
            Pair(
                label=label,
                image=image,
                mask=mask,
                image_sha256=hashlib.sha256(image_bytes).hexdigest(),
                mask_sha256=hashlib.sha256(mask_bytes).hexdigest(),
            )
        )
    return out


def face_box_mask(pair: Pair) -> tuple[UInt8Array, bool]:
    """A rectangle over the detected face, as a face row could produce.

    The best a stored face observation can do. Full-frame when no face is
    found, which is the most generous possible substitute -- and it still has
    to break, or the vendor's mask was not carrying subject information.
    """
    from compat.producers import insightface_pass as producer

    height, width = pair.mask.shape[:2]
    out = np.zeros((height, width), dtype=np.uint8)
    bgr = pair.image[:, :, ::-1].copy()
    found = producer.analysis().get(bgr)
    if not found:
        # Full frame, and the caller is TOLD it is a fallback: AnyStory's
        # example 1 is a cartoon sheep with no face to box. Reporting it as "a
        # face rectangle" would claim a detection that did not happen.
        out[:, :] = 255
        return out, False
    best = max(found, key=lambda one: (one.bbox[2] - one.bbox[0]) * (one.bbox[3] - one.bbox[1]))
    x1, y1, x2, y2 = (round(float(one)) for one in best.bbox)
    out[max(y1, 0) : min(y2, height), max(x1, 0) : min(x2, width)] = 255
    return out, True


class MaskedReferenceRunner:
    """The (RGB image, L mask) pair AnyStory actually consumes."""

    consumer_id = CONSUMER_ID

    def __init__(self) -> None:
        self._pairs = {one.label: one for one in pairs()}

    def _pair(self, case: Case) -> Pair:
        return self._pairs[case.boundary.partition("|")[2]]

    def cases(self) -> tuple[Case, ...]:
        return tuple(
            Case(
                name=f"anystory_masked_subject_{pair.label}",
                consumer_id=CONSUMER_ID,
                tier=Tier.CONSUMER,
                fixture=Fixture(
                    name=f"anystory_{pair.label}",
                    path=f"{ {a: b for a, b, _ in PAIRS}[pair.label] } + mask",
                    # The PAIR is the fixture: an image without its mask is not
                    # this consumer's input, so the digest covers both.
                    sha256=hashlib.sha256((pair.image_sha256 + pair.mask_sha256).encode("ascii")).hexdigest(),
                    kind="vendor_masked_reference",
                    note=f"image {pair.image_sha256[:12]} + mask {pair.mask_sha256[:12]}, not vendored",
                ),
                boundary=f"masked_subject|{pair.label}",
                exact_bytes=True,
                rtol=0.0,
                atol=0.0,
                retained=("whole_reference_image", "subject_mask"),
                ablations=(
                    Ablation(primitive="whole_reference_image", expect_breaks=True),
                    Ablation(primitive="subject_mask", expect_breaks=True),
                    Ablation(
                        primitive="subject_mask",
                        swap="face_bbox_rectangle",
                        expect_breaks=True,
                        kind="substitution",
                    ),
                ),
                measurements=("mask_coverage",),
                note="inference.py:19-20 RGB image + L mask; :23 images=/masks= parallel lists",
            )
            for pair in self._pairs.values()
        )

    def retained_for(self, case: Case) -> RetainedState:
        pair = self._pair(case)
        image, mask = pair.image.copy(), pair.mask.copy()
        return RetainedState(whole_reference_image=image, subject_mask=mask).priced(
            {
                "whole_reference_image": derivatives.lossless_bytes(image),
                "subject_mask": derivatives.lossless_bytes(mask),
            }
        )

    def _artifact(self, name: str, image: UInt8Array, mask: UInt8Array) -> Artifact:
        # The boundary artifact is the pair, flattened together: a case that
        # compared only the image would pass with any mask at all, which is
        # exactly the hole this module exists to close.
        joined = np.concatenate([image.reshape(-1), mask.reshape(-1)])
        return Artifact(
            name=name,
            dtype=str(joined.dtype),
            # The shape of `joined`, which is 1-D, not the two contributions:
            # `contracts/case.py:117` keeps shape separately so a shape failure
            # is distinguishable from a value failure.
            shape=joined.shape,
            sha256=digest(joined),
            values=joined,
        )

    def baseline(self, case: Case) -> Artifact:
        pair = self._pair(case)
        return self._artifact(case.boundary, pair.image, pair.mask)

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        return self._artifact(
            case.boundary,
            retained.pixels("whole_reference_image"),
            retained.pixels("subject_mask"),
        )

    def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState:
        if ablation.swap == "face_bbox_rectangle":
            substitute, _ = face_box_mask(self._pair(case))
            return retained.replacing("subject_mask", substitute)
        return retained.without(ablation.primitive)

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        """How much of the vendor's mask a face rectangle can account for."""
        if name != "mask_coverage":
            raise KeyError(f"{CONSUMER_ID} has no measurement called {name!r}")
        pair = self._pair(case)
        theirs = pair.mask > 127
        substitute, from_face = face_box_mask(pair)
        rectangle = substitute > 127
        subject = int(theirs.sum())
        overlap = int((theirs & rectangle).sum())
        return Measurement(
            name=name,
            unit="fraction",
            value=(overlap / subject) if subject else 0.0,
            basis="the vendor's subject mask against the best a face row could offer",
            detail=(
                f"{pair.label}: subject mask covers {subject:,} px of {theirs.size:,}; "
                f"{'a face rectangle' if from_face else 'NO FACE DETECTED, so the full frame'} "
                f"covers {int(rectangle.sum()):,} and accounts for "
                f"{(overlap / subject * 100) if subject else 0:.1f}% of it"
            ),
        )


def all_runners() -> tuple[MaskedReferenceRunner, ...]:
    return (MaskedReferenceRunner(),)
