"""What a compatibility case is, and what a run of one records.

The shape follows ONNX's backend test suite -- read at
onnx/onnx@f9b25fb1a5302ef0c833bef8c917cc7f031feb6b,
`onnx/backend/test/case/test_case.py` and `docs/OnnxBackendTest.md` -- which is
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

from dataclasses import dataclass
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


class Verdict(StrEnum):
    """The four honest outcomes. There is no fifth.

    Members are named for what happened and carry the wire value the evidence
    reports. `REPRODUCED`/`DIVERGED` rather than `PASS`/`FAIL` because a member
    called PASS reads to a linter as a credential -- and because the verb is
    the more accurate word for what a replay did.
    """

    REPRODUCED = "PASS"
    """Replay reproduced the baseline within the case's own tolerance."""

    DIVERGED = "FAIL"
    """Replay ran and did not reproduce the baseline. The claim is wrong."""

    CONTRADICTED = "CONTRADICTED"
    """An ablation that was expected to break the replay did not, so the
    primitive it removed is NOT necessary and must not be stored as durable
    truth. A passing ablation is a failing necessity claim."""

    UNSUPPORTED = "UNSUPPORTED"
    """The case could not run here -- absent weights, absent runtime, absent
    device. Never silently folded into PASS, and never dropped from the
    population: a consumer that disappears when it cannot run is how a suite
    reports success it did not earn."""


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

    def has(self, key: str) -> bool:
        return key in self._values

    def keys(self) -> tuple[str, ...]:
        return tuple(self._values)

    def without(self, key: str) -> RetainedState:
        """This state minus one primitive. The ablation operator."""
        return RetainedState(**{name: one for name, one in self._values.items() if name != key})

    def replacing(self, key: str, value: object) -> RetainedState:
        """This state with one primitive degraded rather than removed."""
        return RetainedState(**{**self._values, key: value})

    def _require(self, key: str) -> object:
        if key not in self._values:
            raise KeyError(f"replay needs {key!r}, which the retained state does not carry")
        return self._values[key]

    def _array(self, key: str) -> npt.NDArray[np.generic]:
        value = self._require(key)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{key!r} is {type(value).__name__}, not an array")
        return value

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


@dataclass(frozen=True)
class Ablation:
    """One retained primitive removed, and what that did.

    `expect_breaks` is the claim under test. A primitive is durable truth only
    when removing it actually breaks the replay; if the replay still passes,
    the primitive is derivable from what remained and the verdict is
    CONTRADICTED -- which is the whole reason this dataclass exists.
    """

    primitive: str
    expect_breaks: bool
    kind: str = "removal"
    """`removal` takes the primitive away and asks whether the replay still
    works -- that is the necessity test, and a primitive nothing misses is
    derivable rather than durable.

    `substitution` swaps one retained value for another the store already
    holds, and asks whether the cheaper one serves. It is NOT a necessity
    claim about a primitive, and a generated view that counts the two
    together reports `face_patch_substituted` as though it were something the
    database keeps."""

    observed_break: bool | None = None
    verdict: Verdict | None = None
    detail: str = ""


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
    unsupported_reason: str = ""
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

    def for_consumer(self, consumer_id: str) -> tuple[Case, ...]:
        return tuple(one for one in self._cases.values() if one.consumer_id == consumer_id)

    def __len__(self) -> int:
        return len(self._cases)
