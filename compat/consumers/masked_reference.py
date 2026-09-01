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


PAIRS: Final[tuple[tuple[str, str, str], ...]] = (
    ("single", "assets/examples/1.webp", "assets/examples/1_mask.webp"),
    ("multi_1", "assets/examples/6_1.webp", "assets/examples/6_1_mask.webp"),
    ("multi_2", "assets/examples/6_2.webp", "assets/examples/6_2_mask.webp"),
)


@dataclass(frozen=True)
class Pair:
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


MASK_MODES: Final[tuple[str, ...]] = ("L", "RGB")


def _mask_flag(mode: str) -> int:
    import cv2

    return {"L": cv2.IMREAD_GRAYSCALE, "RGB": cv2.IMREAD_COLOR}[mode]


def mask_mode() -> str:
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
    import cv2

    repo, commit = _repo()
    declared = mask_mode()
    out: list[Pair] = []
    for label, image_path, mask_path in PAIRS:
        try:
            image_bytes = _blob(repo, commit, image_path)
            mask_bytes = _blob(repo, commit, mask_path)
        except ValueError as problem:
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
    from compat.producers import insightface_pass as producer

    height, width = pair.mask.shape[:2]
    out = np.zeros((height, width), dtype=np.uint8)
    bgr = pair.image[:, :, ::-1].copy()
    found = producer.analysis().get(bgr)
    if not found:
        out[:, :] = 255
        return out, False
    best = max(found, key=lambda one: (one.bbox[2] - one.bbox[0]) * (one.bbox[3] - one.bbox[1]))
    x1, y1, x2, y2 = (round(float(one)) for one in best.bbox)
    out[max(y1, 0) : min(y2, height), max(x1, 0) : min(x2, width)] = 255
    return out, True


class MaskedReferenceRunner:
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

        joined = np.concatenate([image.reshape(-1), mask.reshape(-1)])
        return Artifact(
            name=name,
            dtype=str(joined.dtype),
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
