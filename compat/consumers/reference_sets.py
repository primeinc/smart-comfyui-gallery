"""Reference-set semantics, executed rather than declared.

The manifest declares a `combine` per consumer -- stack, mean, concat, none --
and nothing ran any of them. These cases run the declared combiners over the
vendors' OWN reference sets and separate the four possibilities that a
single-reference case cannot:

    order       A,B against B,A          -- ordered sequence or a set?
    arity       A against A,B against A,A -- variable N or fixed?
    duplication A,A against A            -- is a set deduplicated?
    identity    A,B mixed-identity       -- negative control

FIXTURES ARE THE VENDORS' OWN, per-identity folders at the pinned commits:

    TencentARC/PhotoMaker@060b4fcb10b7
        examples/newton_man/            4 images, one man
        examples/scarletthead_woman/    4 images, one woman

Two identities is what makes the negative control possible; four images per
identity is what makes A,A distinguishable from A.

WHAT COMBINE MEANS, from first-party sources
--------------------------------------------
    stack   TencentARC/PhotoMaker@060b4fcb10b7 -- `input_id_images` is a LIST
            and the ID embedding is stacked per image, so N and order are
            both carried.
    mean    Gourieff/Assets@606558ed08f16b99a29ef30b0df0b4622164c524
            comfyui-reactor-node/workflows/ReActor--Build-Blended-Face-Model--v2.json
            -- ReActorBuildFaceModel widgets [True, True, 'test', 'Mean'];
            its Note node lists Mean / Median / Mode. An average discards
            order and cannot distinguish A,A from A.

The boundary here is the COMBINED VECTOR, before any generator: that is the
last deterministic artifact, and it is where order and arity either survive or
do not.
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
    Float32Array,
    Measurement,
    RetainedState,
    Tier,
    note_skip,
)
from compat.harness import provenance
from compat.producers import insightface_pass as producer
from compat.storage import precision

CONSUMER_ID: Final[str] = "reference_sets"

#: Vendor identity folders, at the pinned commit of the repo that ships them.
IDENTITIES: Final[dict[str, tuple[str, ...]]] = {
    "newton_man": (
        "examples/newton_man/newton_0.jpg",
        "examples/newton_man/newton_1.jpg",
        "examples/newton_man/newton_2.png",
        "examples/newton_man/newton_3.jpg",
    ),
    "scarletthead_woman": (
        "examples/scarletthead_woman/scarlett_0.jpg",
        "examples/scarletthead_woman/scarlett_1.jpg",
        "examples/scarletthead_woman/scarlett_2.jpg",
        "examples/scarletthead_woman/scarlett_3.jpg",
    ),
}

#: The set shapes under test. `A` and `B` index into one identity's images;
#: `X` is the OTHER identity, the negative control.
ARRANGEMENTS: Final[dict[str, tuple[str, ...]]] = {
    "single_A": ("A",),
    "pair_AB": ("A", "B"),
    "pair_BA": ("B", "A"),
    "duplicate_AA": ("A", "A"),
    "triple_ABC": ("A", "B", "C"),
    "mixed_AX": ("A", "X"),
}

#: Declared combiners this module knows how to execute.
COMBINERS: Final[tuple[str, ...]] = ("stack", "mean")


@dataclass(frozen=True)
class Reference:
    """One vendor reference image, by content."""

    identity: str
    path: str
    sha256: str


def _repo_root(consumer_id: str) -> tuple[Path, str]:
    manifest = provenance.load_manifest()
    refs_root = (Path(__file__).resolve().parent.parent.parent / manifest["refs_root"]).resolve()
    for row in manifest.get("consumers", []):
        if row["id"] == consumer_id:
            return provenance.clone_dir(refs_root, row["repo"]), row["commit"]
    raise KeyError(f"{consumer_id} is not in the manifest")


def references() -> dict[str, list[Reference]]:
    """The vendor's own images, extracted from the pinned commit to a cache.

    Written outside the repository and never committed: these are third-party
    media under the vendor's own terms.
    """

    root, commit = _repo_root("photomaker_v2")
    cache = Path(__file__).resolve().parent.parent.parent.parent / "sg-vendor-fixtures" / "photomaker_v2"
    out: dict[str, list[Reference]] = {}
    for identity, paths in IDENTITIES.items():
        held: list[Reference] = []
        for path in paths:
            code, blob, why = proc.run(
                ["git", "-C", str(root), "cat-file", "blob", f"{commit}:{path}"], timeout=proc.LOCAL_SECONDS
            )
            if code != 0:
                # A missing blob silently shrank an identity, and `_slots`
                # then raised IndexError, which `run_case` recorded as
                # UNSUPPORTED -- an absent-runtime verdict for an absent file.
                note_skip(CONSUMER_ID, path, f"not at {commit[:12]}: {why.decode('utf-8', 'replace')[:120]}")
                continue
            target = cache / Path(path).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            held.append(Reference(identity=identity, path=str(target), sha256=hashlib.sha256(blob).hexdigest()))
        if held:
            out[identity] = held
    return out


def embed(path: Path) -> Float32Array:
    """One reference image to one raw 512-d vector, our producer's own.

    The per-image vector is the unit a combiner combines. Which recognition
    space it comes from is a separate question from how N of them are folded
    together, and this module answers only the second.
    """
    frame, _ = producer.decode(path)
    found = producer.analysis().get(frame)
    if not found:
        raise ValueError(f"no face detected in the vendor reference {path.name}")
    best = max(found, key=lambda one: (one.bbox[2] - one.bbox[0]) * (one.bbox[3] - one.bbox[1]))
    return np.asarray(best.embedding, dtype=np.float32).reshape(-1)


def combine(vectors: list[Float32Array], how: str) -> Float32Array:
    """The declared combiners, executed.

    `stack` keeps every vector and their order. `mean` averages them, which
    is where order and duplication stop being recoverable -- and that is the
    property these cases exist to demonstrate rather than assert.
    """
    if how == "stack":
        return np.asarray(np.stack(vectors), dtype=np.float32)
    if how == "mean":
        return np.asarray(np.mean(np.stack(vectors), axis=0), dtype=np.float32)
    raise KeyError(f"no combiner called {how!r}")


def _reversal_observable(how: str, arrangement: str) -> bool:
    """Whether reversing THIS set is CLAIMED to change THIS combiner's output.

    A stated rule, not a measurement. It was documented as "Measured, not
    assumed" while taking two strings, touching no vector and never calling
    `combine` -- and `measure` computed the real answer two screens down and
    put it in a detail string.

    It stays a rule on purpose. Deriving the expectation from the same vectors
    the ablation then runs on would make the case unfalsifiable, which is the
    defect `face_family.embedding_ablations` has. So the rule predicts, the
    measurement observes, and `answer.corroboration` reds when they disagree.

    Three cases where the naive "a stack is ordered, a mean is not" gives the
    wrong prediction:

    N == 1          reversal is the identity, so a stack cannot notice it.
    all-equal set   `duplicate_AA` reverses to itself elementwise, so no
                    combiner can notice. A stack of duplicates carries no
                    order information at all.
    mean, N >= 3    float32 addition is commutative but NOT associative:
                    (a+b)+c and (c+b)+a differ in the last bits, which is what
                    an exact_bytes case compares. So a mean IS order-sensitive
                    from three references up, and storing "the centroid" is
                    not well defined unless the summation order is fixed too.

                    A worst absolute difference of 0.0 over differing
                    elements is not sub-precision: widening float32 through
                    int64 truncates every difference under 1 to zero
                    (`compat/assertions/arrays.py`). The claim
                    holds; the number quoted for it was destroyed, and is
                    re-derivable now that the comparator widens by kind.
    """
    slots = ARRANGEMENTS[arrangement]
    if len(slots) < 2 or len(set(slots)) == 1:
        return False
    if how == "stack":
        return True
    return len(slots) >= 3


class ReferenceSetRunner:
    """Every arrangement, through every declared combiner."""

    consumer_id = CONSUMER_ID

    def __init__(self) -> None:
        self._references = references()
        self._vectors: dict[str, Float32Array] = {}

    def _identity_pair(self) -> tuple[str, str]:
        names = sorted(self._references)
        if len(names) < 2:
            raise ValueError(f"a negative control needs two identities; found {names}")
        return names[0], names[1]

    def vector(self, reference: Reference) -> Float32Array:
        if reference.path not in self._vectors:
            self._vectors[reference.path] = embed(Path(reference.path))
        return self._vectors[reference.path]

    def _slots(self, arrangement: str) -> list[Reference]:
        """Resolve A/B/C/X to actual vendor images."""
        primary, other = self._identity_pair()
        mine = self._references[primary]
        theirs = self._references[other]
        picked: list[Reference] = []
        for slot in ARRANGEMENTS[arrangement]:
            if slot == "X":
                picked.append(theirs[0])
            else:
                picked.append(mine["ABC".index(slot)])
        return picked

    def _fixture(self, arrangement: str) -> Fixture:
        chosen = self._slots(arrangement)
        joined = "".join(one.sha256 for one in chosen).encode("ascii")
        return Fixture(
            name=f"refset_{arrangement}",
            path=";".join(one.path for one in chosen),
            # The SET is the fixture, so its identity is the ordered digest of
            # its members: A,B and B,A are different fixtures and must hash
            # differently, or the order case could not exist.
            sha256=hashlib.sha256(joined).hexdigest(),
            kind="vendor_reference_set",
            note=f"{len(chosen)} vendor images: {', '.join(one.sha256[:8] for one in chosen)}",
        )

    def _parts(self, case: Case) -> tuple[str, str]:
        how, _, arrangement = case.boundary.partition("|")
        return how, arrangement

    def cases(self) -> tuple[Case, ...]:
        if len(self._references) < 2:
            return ()
        out: list[Case] = []
        for how in COMBINERS:
            out.extend(
                Case(
                    name=f"refset_{how}_{arrangement}",
                    consumer_id=CONSUMER_ID,
                    tier=Tier.PRIMITIVE,
                    fixture=self._fixture(arrangement),
                    boundary=f"{how}|{arrangement}",
                    exact_bytes=True,
                    rtol=0.0,
                    atol=0.0,
                    retained=("reference_vectors",),
                    ablations=(
                        Ablation(primitive="reference_vectors", expect_breaks=True),
                        # Width, not presence: halving the float asks
                        # how wide the column has to be.
                        Ablation(
                            primitive="reference_vectors",
                            swap="half_precision",
                            expect_breaks=True,
                            kind="substitution",
                        ),
                        # Whether reversal is observable at all depends on the
                        # combiner AND the set: see `_reversal_observable`.
                        Ablation(
                            primitive="reference_vectors",
                            swap="order_reversed",
                            expect_breaks=_reversal_observable(how, arrangement),
                            kind="substitution",
                        ),
                    ),
                    measurements=("set_semantics", "reversal_observed"),
                    note=f"{how} over {len(ARRANGEMENTS[arrangement])} vendor references",
                )
                for arrangement in ARRANGEMENTS
            )
        return tuple(out)

    def retained_for(self, case: Case) -> RetainedState:
        _, arrangement = self._parts(case)
        chosen = [self.vector(one) for one in self._slots(arrangement)]
        return RetainedState(reference_vectors=np.asarray(np.stack(chosen), dtype=np.float32))

    def _artifact(self, name: str, values: np.ndarray) -> Artifact:
        return Artifact(name=name, dtype=str(values.dtype), shape=values.shape, sha256=digest(values), values=values)

    def baseline(self, case: Case) -> Artifact:
        how, arrangement = self._parts(case)
        return self._artifact(case.boundary, combine([self.vector(one) for one in self._slots(arrangement)], how))

    def replay(self, case: Case, retained: RetainedState) -> Artifact:
        how, _ = self._parts(case)
        held = np.asarray(retained.array("reference_vectors"), dtype=np.float32)
        return self._artifact(case.boundary, combine(list(held), how))

    def ablate(self, case: Case, retained: RetainedState, ablation: Ablation) -> RetainedState:
        if ablation.swap == "half_precision":
            return retained.replacing("reference_vectors", precision.half(retained.array("reference_vectors")))
        if ablation.swap == "order_reversed":
            held = np.asarray(retained.array("reference_vectors"), dtype=np.float32)
            return retained.replacing("reference_vectors", held[::-1].copy())
        return retained.without(ablation.primitive)

    def _reversal_observed(self, case: Case, retained: RetainedState) -> Measurement:
        """Whether reversing this set ACTUALLY changed this combiner's output.

        The fact `_reversal_observable` predicts. Recorded as a number so a
        gate can read it: 1.0 observable, 0.0 not. It was already being
        computed inside `set_semantics` and spent on a detail string, so the
        suite held the measurement that would have falsified its own truth
        table and never compared the two.
        """
        how, arrangement = self._parts(case)
        held = np.asarray(retained.array("reference_vectors"), dtype=np.float32)
        folded = combine(list(held), how)
        reversed_fold = combine(list(held[::-1]), how)
        ordered = not np.array_equal(folded, reversed_fold)
        return Measurement(
            name="reversal_observed",
            unit="bool",
            value=1.0 if ordered else 0.0,
            basis=f"{how} over {arrangement} against the same set reversed, compared bytewise",
            detail=(
                f"{arrangement} through {how}: reversing the set "
                f"{'changes' if ordered else 'does not change'} the combined artifact "
                f"(rule predicted {_reversal_observable(how, arrangement)})"
            ),
        )

    def measure(self, case: Case, retained: RetainedState, name: str) -> Measurement:
        """What this arrangement costs, and whether the combiner kept its shape."""
        if name == "reversal_observed":
            return self._reversal_observed(case, retained)
        if name != "set_semantics":
            raise KeyError(f"{CONSUMER_ID} has no measurement called {name!r}")
        how, arrangement = self._parts(case)
        held = np.asarray(retained.array("reference_vectors"), dtype=np.float32)
        folded = combine(list(held), how)
        reversed_fold = combine(list(held[::-1]), how)
        ordered = not np.array_equal(folded, reversed_fold)
        return Measurement(
            name=name,
            unit="bytes",
            value=float(folded.nbytes),
            basis="the combined artifact, and whether reversing the set changes it",
            detail=(
                f"{arrangement} through {how}: {held.shape} -> {folded.shape} = {folded.nbytes:,} B; "
                f"order-sensitive: {ordered}"
            ),
        )


def all_runners() -> tuple[ReferenceSetRunner, ...]:
    return (ReferenceSetRunner(),)
