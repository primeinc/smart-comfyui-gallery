"""What a compatibility case is, and what a run of one records.

The shape follows ONNX's backend test suite -- read at
onnx/onnx@f9b25fb1a5302ef0c833bef8c917cc7f031feb6b onnx/backend/test/case/test_case.py
and onnx/onnx@f9b25fb1a5302ef0c833bef8c917cc7f031feb6b docs/OnnxBackendTest.md -- which is
the same problem: a suite each external implementation runs to prove it still
satisfies a contract, where the cases ARE the specification rather than a
document about it.

Taken from there:

  * tolerance is per case, not global. The stage at which exactness is lost
    differs by consumer, so one project-wide epsilon either hides a real
    divergence or fails a legitimate one.
  * two tiers. ONNX has Node (one operator) and Model (whole graph); here it is
    PRIMITIVE (one transform, e.g. norm_crop at 224) and CONSUMER (the whole
    boundary). Ablation lives in the primitive tier, because that is where a
    single retained value can be removed and the effect seen.
  * large artifacts are fetched, never vendored. Their model protobufs go to
    the cloud and come down on demand; our weights are hundreds of megabytes
    and follow the same rule.
  * `__test__ = False`, or pytest collects a dataclass called TestCase.
  * a repeated case name is an error, not an overwrite.

Not taken: their registry is module-global mutable state carrying a linter
suppression. A registry object is explicit, testable, and this tree bans
suppressions anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

#: The only array type that crosses this boundary. Upstream hands back whatever
#: it likes; an adapter converts once, here, and nothing past that point sees
#: an unannotated value.
Float32Array = npt.NDArray[np.float32]
UInt8Array = npt.NDArray[np.uint8]
Int64Array = npt.NDArray[np.int64]


class Tier(StrEnum):
    """ONNX's Node/Model split, named for what it means here."""

    PRIMITIVE = "primitive"
    """One transform over retained state. Where ablation is meaningful."""

    CONSUMER = "consumer"
    """A whole consumer boundary, end to end."""


class MissingPrimitive(KeyError):
    """A replay asked the retained state for a key that is not there.

    Its own type, because the executor has to tell two situations apart that a
    bare KeyError cannot. Removing a primitive and watching the consumer
    produce a different ANSWER is a necessity result. Removing it and watching
    the runner fail to subscript a dict is a fact about the runner: it holds
    for every key, including keys nothing needs, so it can never come out the
    other way and it measures nothing.
    """


class Verdict(StrEnum):
    """What actually happened, named for the happening.

    `REPRODUCED`/`DIVERGED` rather than `PASS`/`FAIL` because a member called
    PASS reads to a linter as a credential -- and because the verb is the more
    accurate word for what a replay did.
    """

    REPRODUCED = "PASS"
    """Replay reproduced the baseline within the case's own tolerance."""

    DIVERGED = "FAIL"
    """Replay ran and did not reproduce the baseline. The claim is wrong."""

    INCONCLUSIVE = "INCONCLUSIVE"
    """An ablation that could not answer the question it was asked.

    Reserved for the case where a removal stopped the replay because the
    runner dereferenced the absent key, rather than because the consumer
    needed the value. `RetainedState._require` raises `MissingPrimitive` for
    exactly that, and the outcome holds for EVERY key -- so recording it as a
    break made 22 of 23 primitives report `survives: 0` and made every
    necessity claim in `answer.json` unfalsifiable.

    It is not a claim that the primitive is derivable. It is the absence of a
    claim, said out loud instead of counted as evidence."""

    CONTRADICTED = "CONTRADICTED"
    """An ablation that was expected to break the replay did not, so the
    primitive it removed is NOT necessary and must not be stored as durable
    truth. A passing ablation is a failing necessity claim."""

    VENDOR_BASELINE_UNAVAILABLE = "VENDOR_BASELINE_UNAVAILABLE"
    """Upstream supplies no runnable first-party example, or no first-party
    sample input for one, at the pinned commit.

    There is no verdict for "could not run here". An absent weight, an absent
    runtime, an unimplemented derivation and a detector that found no face are
    all a proof that did not happen, and `run_case` lets every one of them
    escape. The verdict that used to absorb them held 16 results, nine of which
    were a boundary nobody had written; naming them made the lane exit 0 over
    work that does not exist. This is a fact about the upstream: there is
    nothing to calibrate against, so no amount of local execution can
    establish that our adapter reproduces the
    vendor's path. The consumer stays visibly unresolved, and what upstream
    does and does not provide is recorded rather than worked around."""


@dataclass(frozen=True)
class Skipped:
    """One input a lane declined to build a case from, and the reason."""

    consumer_id: str
    what: str
    why: str


_skipped: list[Skipped] = []
_considered: list[Skipped] = []


def note_skip(consumer_id: str, what: str, why: str) -> None:
    """Record a REQUIRED input that produced no case.

    A bare `continue` is how a population shrinks without anyone deciding to
    shrink it. The evidence then reports a pass rate over whatever survived,
    which is the one property this suite says it must never have -- and it had
    it at six sites, silently, each for a different reason.

    NOT for a candidate a search looked at and passed over: see
    `note_considered`. Recording both here made `face_selection` report a
    skipped input while its population was complete -- 33 cases, all
    reproduced, over the three discriminating photographs it set out to find.
    """
    _skipped.append(Skipped(consumer_id=consumer_id, what=what, why=why))


def note_considered(consumer_id: str, what: str, why: str) -> None:
    """Record a candidate a search evaluated and did not use.

    Audit evidence, not a population hole. `face_selection` scans group
    photographs for ones where `first` and `largest_bbox_area` reach different
    faces; a photograph the detector sees one face in cannot separate the two
    rules, so it is not a member of the population that lane is proving.

    Recorded rather than dropped, because a search that silently discards its
    rejects cannot be reviewed for having rejected the wrong ones.
    """
    _considered.append(Skipped(consumer_id=consumer_id, what=what, why=why))


def considered() -> tuple[Skipped, ...]:
    """Every DISTINCT candidate a search evaluated and passed over."""
    return tuple(sorted(set(_considered), key=lambda one: (one.consumer_id, one.what, one.why)))


def skipped() -> tuple[Skipped, ...]:
    """Every DISTINCT input a lane declined to build a case from.

    Deduplicated, and sorted by content rather than by arrival. A skip is a
    FACT about an input -- this photograph, this reason -- and the ledger was
    reporting it once per occurrence, which made the evidence a property of
    how many times a process happened to construct a runner rather than of
    what was skipped.

    That is not cosmetic. `runners(only)` builds every runner and then filters,
    so all six shards construct `FaceSelectionRunner`, all six record the same
    skip, and `sharded.merge` concatenated six copies of one photograph. The
    single-process rebuild records it once, so
    `attack.evidence_not_reproducible` compared six against one and reported
    the pipeline as non-deterministic -- correctly, for a ledger that counted
    processes.
    """
    return tuple(sorted(set(_skipped), key=lambda one: (one.consumer_id, one.what, one.why)))


@dataclass(frozen=True)
class Fixture:
    """One input, by content.

    Hashed rather than named because a fixture edited in place under the same
    filename would otherwise invalidate every baseline recorded against it
    without changing a single line of evidence.
    """

    name: str
    path: str
    sha256: str
    kind: str
    note: str = ""


@dataclass(frozen=True)
class Artifact:
    """One boundary output, comparable and hashable.

    `values` is what the comparison reads; `sha256` is over the canonical bytes
    so a reviewer can check the evidence without re-running the model. dtype
    and shape are recorded separately because two arrays that hash differently
    for a dtype reason are a different failure from two that differ in content.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str
    values: Float32Array | UInt8Array | None = None


class RetainedState:
    """The durable state a replay is allowed to see, and nothing else.

    A plain mapping would force every runner to cast at each use, and a cast is
    where a wrong assumption stops being visible. This narrows once, here, with
    accessors that raise by name -- so an ablation that removes a primitive
    produces a readable failure naming it, which is exactly the signal the
    ablation is trying to observe.

    Absence is the mechanism, not an error to smooth over: `without()` returns
    a copy missing one key, and the replay is expected to fail on it.
    """

    def __init__(self, **values: object) -> None:
        self._values: dict[str, object] = {key: one for key, one in values.items() if one is not None}
        #: Encoded bytes for keys whose runner knows what a store would hold.
        #: Set through `priced()`, not the constructor: runners build this over
        #: a dynamic key, which any named parameter here could collide with.
        self._durable: dict[str, int] = {}

    def priced(self, durable: dict[str, int]) -> RetainedState:
        """This state, with the encoded size of each named array.

        Each size must equal `compat.storage.derivatives.lossless_bytes` of
        the value it names; a size that does not is rejected. `sizes()`
        prefers these over `ndarray.nbytes`, so an unchecked number would be a
        storage cost the runner asserts rather than one derived from the
        array.

        Raises KeyError for a name this state does not hold, TypeError for a
        value that is not an array, and ValueError for a size that is not the
        array's encoded length.
        """
        from compat.storage import derivatives

        for name, claimed in durable.items():
            if name not in self._values:
                raise KeyError(f"priced {name!r}, which this state does not hold: {sorted(self._values)}")
            value = self._values[name]
            if not isinstance(value, np.ndarray):
                raise TypeError(f"priced {name!r}, which is {type(value).__name__} rather than an array")
            actual = derivatives.lossless_bytes(value)
            if claimed != actual:
                raise ValueError(
                    f"priced {name!r} at {claimed:,} B; the array encodes to {actual:,} B. "
                    f"A durable size is the artifact's length, not a number the runner chooses"
                )
        held = RetainedState(**self._values)
        held._durable = {**self._durable, **durable}
        return held

    def has(self, key: str) -> bool:
        return key in self._values

    def keys(self) -> tuple[str, ...]:
        return tuple(self._values)

    def without(self, key: str) -> RetainedState:
        """This state minus one primitive. The ablation operator."""
        held = {name: one for name, one in self._values.items() if name != key}
        return RetainedState(**held).priced({k: v for k, v in self._durable.items() if k != key})

    def replacing(self, key: str, value: object) -> RetainedState:
        """This state with one primitive degraded rather than removed.

        The replaced key loses its durable price: the substitute is a
        different artifact and the encoded size of the original is not its
        size.
        """
        return RetainedState(**{**self._values, key: value}).priced(
            {k: v for k, v in self._durable.items() if k != key}
        )

    def sizes(self) -> dict[str, int]:
        """Bytes each retained value occupies, by key.

        The answer this suite produces is a MINIMUM, and a minimum with no
        size attached is half an answer. `producer_union.json` prices the keys
        a producer emits and nothing else, so sixteen of the twenty-three
        names in `must_retain` reported 0 bytes -- including the picture,
        which is the largest thing in the set by three orders of magnitude.

        Measured off the value itself, per case, so a lane-local name is
        priced by what it actually holds rather than by whether some producer
        happens to emit a key of the same name. Scalars and tuples are sized
        by their Python object where no buffer exists, which is honest for a
        two-integer origin and irrelevant beside a 36 MB frame.

        ENCODED WHERE THE RUNNER KNOWS IT, decoded otherwise. `ndarray.nbytes`
        is the in-memory footprint and a store holds encoded bytes: measured
        2026-08-29 on one corpus frame, nbytes 95,883,264 against 21,466,629
        for the lossless PNG `compat/storage/derivatives.lossless` produces
        from it, a 4.5x overstatement. A runner that has encoded the artifact
        passes the real figure as `_durable` and it is preferred here.

        A key with no `_durable` entry is still priced by `nbytes`, which is
        an upper bound rather than a storage cost. `producer_union.json`
        prices only the keys a producer emits, which is why sixteen of the
        twenty-three names in `must_retain` reported 0 bytes before this
        existed -- including the picture.
        """
        out: dict[str, int] = {}
        for key, value in self._values.items():
            if key in self._durable:
                out[key] = self._durable[key]
            elif isinstance(value, np.ndarray):
                out[key] = value.nbytes
            elif isinstance(value, (bytes, bytearray, memoryview)):
                out[key] = len(bytes(value))
            elif isinstance(value, str):
                out[key] = len(value.encode("utf-8"))
            elif isinstance(value, tuple):
                out[key] = 8 * len(value)
            elif isinstance(value, bool):
                out[key] = 1
            elif isinstance(value, (int, float)):
                out[key] = 8
        return out

    def same_as(self, other: RetainedState) -> bool:
        """Whether two states carry the same keys and the same values.

        A substitution that leaves the state untouched separates nothing, for
        exactly the reason a removal the replay indexes separates nothing. It
        happens for real reasons -- `build.KEYPOINTS` are whole numbers under
        2048 and every one is exact in binary16, so narrowing them returns the
        original -- and the ablation must report that it could not answer
        rather than a break it never had a chance to observe.
        """
        if self.keys() != other.keys():
            return False
        for key in self.keys():
            mine, theirs = self._values[key], other._values[key]
            if isinstance(mine, np.ndarray) or isinstance(theirs, np.ndarray):
                if not (isinstance(mine, np.ndarray) and isinstance(theirs, np.ndarray)):
                    return False
                if mine.dtype != theirs.dtype or not np.array_equal(mine, theirs):
                    return False
            elif mine != theirs:
                return False
        return True

    def _require(self, key: str) -> object:
        if key not in self._values:
            raise MissingPrimitive(f"replay needs {key!r}, which the retained state does not carry")
        return self._values[key]

    def _array(self, key: str) -> npt.NDArray[np.generic]:
        value = self._require(key)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{key!r} is {type(value).__name__}, not an array")
        return value

    def array(self, key: str) -> npt.NDArray[np.generic]:
        """The value with its dtype unchanged.

        The accessors below narrow to a dtype the caller names. A case whose
        question IS the dtype cannot use them: `det_score` written to a
        SQLite REAL returns float64, and `points()` would cast it back to the
        producer's float32 and report a match.
        """
        return self._array(key)

    def pixels(self, key: str) -> UInt8Array:
        return self._array(key).astype(np.uint8, copy=False)

    def points(self, key: str) -> Float32Array:
        return self._array(key).astype(np.float32, copy=False)

    def integers(self, key: str) -> Int64Array:
        """A whole-number value, kept whole.

        Separate from `points` because casting an integer through float32
        loses exactness above 2**24 and, worse, changes the dtype a consumer
        then writes into its own container: ReActor's `save_face_model` calls
        `torch.tensor(face["age"])`, so a float here produces a float tensor
        where upstream produces an integer one, and the file diverges for a
        reason that has nothing to do with what was stored.
        """
        return self._array(key).astype(np.int64, copy=False)

    def pair(self, key: str) -> tuple[int, int]:
        """Two integers, narrowed element by element.

        The elements are checked individually rather than asserted: a tuple is
        not enough to know what is inside it, and a coordinate that arrives as
        a string should say so here rather than three frames further on.
        """
        value = self._require(key)
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(f"{key!r} is {value!r}, not a two-element tuple")
        first, second = value
        if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
            raise TypeError(f"{key!r} holds {type(first).__name__} and {type(second).__name__}, not numbers")
        return int(first), int(second)

    def number(self, key: str) -> float:
        """One scalar, narrowed and required.

        Distinct from `flag`, which answers a present/absent question and
        defaults to False. A scale factor read through `flag` would come back
        as True and multiply nothing, so absence has to raise here rather than
        become a plausible-looking 1.
        """
        value = self._require(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key!r} is {value!r}, not a number")
        return float(value)

    def flag(self, key: str) -> bool:
        return bool(self._values.get(key, False))

    def text(self, key: str) -> str:
        """One string the store holds, required and narrowed.

        A retained value is not always a number: `face_selection` keeps the
        selection RULE beside the face rows, because a store that keeps rows
        and forgets which rule reached the face has not retained the answer.
        """
        value = self._require(key)
        if not isinstance(value, str):
            raise TypeError(f"{key!r} is {type(value).__name__}, not a string")
        return value


def settled_by_measurement(method: str) -> bool:
    """Whether a comparison weighed the consumer's OUTPUT.

    `exact_bytes` and `allclose` weigh values, and either can come out the
    other way. `shape` weighs the output too: the replay ran and produced an
    artifact, and an artifact of a different shape is a different output. Some
    primitives DETERMINE that shape -- `frame_dimensions` is the canvas
    `draw_kps` renders onto, `audio_sample_rate` decides how many samples the
    resampler emits -- so for those no wrong value can leave the shape intact,
    and refusing shape made them unprovable by construction.

    What still does not count: `dtype`, a storage-type mismatch rather than a
    different answer, and "", which an ablation records when the replay RAISED
    and there was no output to weigh at all.
    """
    return method in {"exact_bytes", "shape"} or method.startswith("allclose")


@dataclass(frozen=True)
class Ablation:
    """One retained primitive removed, and what that did.

    `expect_breaks` is the claim under test. A primitive is durable truth only
    when removing it actually breaks the replay; if the replay still passes,
    the primitive is derivable from what remained and the verdict is
    CONTRADICTED -- which is the whole reason this dataclass exists.

    `observed_break` is None when the ablation could not answer: the replay
    raised `MissingPrimitive`, which happens for any absent key and therefore
    distinguishes nothing. Such an ablation is INCONCLUSIVE and no generated
    view may count it toward necessity.
    """

    primitive: str
    expect_breaks: bool
    kind: str = "removal"
    #: `assertions.arrays.Comparison.method`, or "" when the ablation raised.
    #: `shape`, `dtype` and an exception settle before the consumer compares
    #: values; `exact_bytes` and `allclose` are measurements of its output.
    compare_method: str = ""
    """`removal` takes the primitive away and asks whether the replay still
    works -- that is the necessity test, and a primitive nothing misses is
    derivable rather than durable.

    `substitution` swaps one retained value for another the store already
    holds, and asks whether the cheaper one serves. It is NOT a necessity
    claim about a primitive, and a generated view that counts the two
    together reports `face_patch_substituted` as though it were something the
    database keeps."""

    swap: str = ""
    """What a substitution put in the primitive's place.

    A substitution names the primitive it DEGRADES in `primitive` and its
    replacement here. Both halves were previously folded into `primitive`, so
    `face_patch_substituted` -- a thing no store holds -- appeared in the
    primitives table beside `kps` and `pose`, and `answer.py` carried a
    hardcoded set to undo it. Empty for a removal, required for a
    substitution.
    """

    observed_break: bool | None = None
    verdict: Verdict | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.kind == "substitution" and not self.swap:
            raise ValueError(f"the substitution for {self.primitive!r} names no swap")
        if self.kind == "removal" and self.swap:
            raise ValueError(f"the removal of {self.primitive!r} carries swap={self.swap!r}")
        if self.kind not in ("removal", "substitution"):
            raise ValueError(f"{self.kind!r} is not an ablation kind; removal and substitution are")

    @property
    def selector(self) -> str:
        """The name a runner's `ablate` dispatches on.

        One string, because a runner matches a name and does not need to know
        which half of the pair it came from: the swap for a substitution, the
        primitive for a removal.
        """
        return self.swap or self.primitive


@dataclass(frozen=True)
class Measurement:
    """A quantity the suite determined by search rather than asserted.

    An ablation answers yes or no, which is the right shape for "is this
    primitive necessary" and the wrong shape for "how much of it". A margin, a
    resolution floor, a minimum patch extent -- those have a VALUE, and a
    binary claim about one is folklore with a threshold attached: it passes
    while the constant is too generous and says nothing about where the edge
    actually is.

    So the value is recorded, not the belief. `value` is what the search
    found; `basis` names what was searched and how, so the number can be
    re-derived instead of trusted.
    """

    name: str
    unit: str
    value: float | None = None
    basis: str = ""
    detail: str = ""


@dataclass
class Case:
    """One compatibility case: a consumer, a fixture, a boundary, a tolerance.

    Tolerances default to ONNX's own (rtol 1e-3, atol 1e-7) so a case that has
    not thought about its numerics inherits a stated precedent rather than a
    number somebody typed. A case comparing bytes sets both to 0.
    """

    name: str
    consumer_id: str
    tier: Tier
    fixture: Fixture
    boundary: str
    rtol: float = 1e-3
    atol: float = 1e-7
    exact_bytes: bool = False
    retained: tuple[str, ...] = ()
    ablations: tuple[Ablation, ...] = ()
    measurements: tuple[str, ...] = ()
    url: str | None = None
    model_dir: str | None = None
    note: str = ""

    # pytest collects anything named like a test; this is a record, not one.
    __test__: bool = False


@dataclass
class CaseResult:
    """What one executed case recorded. This is the evidence row."""

    case: str
    consumer_id: str
    tier: Tier
    verdict: Verdict
    fixture_sha256: str
    baseline: Artifact | None = None
    replay: Artifact | None = None
    comparison: str = ""
    max_abs_diff: float | None = None
    ablations: tuple[Ablation, ...] = ()
    measurements: tuple[Measurement, ...] = ()
    retained_bytes: dict[str, int] = field(default_factory=dict)
    """What each retained value COST, by key. Recorded per case so
    `answer.json` can price a name no producer emits -- the picture, the
    aligned crop, the patch origin -- which `producer_union.json` cannot."""

    seconds: float = 0.0

    __test__: bool = False


@runtime_checkable
class ConsumerRunner(Protocol):
    """What every consumer file must provide.

    Two paths and nothing else. `baseline` starts from the original media and
    runs the pinned upstream preprocessing; `replay` starts from the retained
    state alone and must never open the source. A runner that reaches for the
    fixture inside `replay` defeats the entire suite, so the retained state is
    the only argument it is given.
    """

    consumer_id: str

    def cases(self) -> tuple[Case, ...]:
        """Every case this consumer contributes, with its own tolerances."""
        ...

    def baseline(self, case: Case) -> Artifact:
        """Original media -> pinned upstream path -> boundary artifact."""
        ...

    def replay(self, case: Case, retained: dict[str, object]) -> Artifact:
        """Retained state -> reconstructed path -> the same boundary artifact.

        `retained` carries only what `case.retained` names. An ablation runs
        this again with one key removed, which is why the signature takes a
        mapping rather than the fixture.
        """
        ...


class Registry:
    """Case collection, without module-global state.

    ONNX appends to a module-level list and flips a module-level filter through
    `global`. This holds the same information on an object, so two collections
    cannot interfere and a test can build one without touching import order.
    """

    def __init__(self) -> None:
        self._cases: dict[str, Case] = {}

    def add(self, case: Case) -> None:
        """Register one case. A repeated name is an error, never an overwrite.

        Same rule as upstream's `_existing_names`: silently replacing a case
        means the suite reports on fewer cases than it lists, and the count is
        the first thing anybody trusts.
        """
        if case.name in self._cases:
            raise ValueError(
                f"case {case.name!r} is already registered by consumer {self._cases[case.name].consumer_id!r}"
            )
        self._cases[case.name] = case

    def extend(self, cases: tuple[Case, ...]) -> None:
        for one in cases:
            self.add(one)

    def all(self) -> tuple[Case, ...]:
        return tuple(self._cases.values())

    def __len__(self) -> int:
        return len(self._cases)
