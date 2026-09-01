from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

Float32Array = npt.NDArray[np.float32]
UInt8Array = npt.NDArray[np.uint8]
Int64Array = npt.NDArray[np.int64]


class Tier(StrEnum):
    PRIMITIVE = "primitive"

    CONSUMER = "consumer"


class MissingPrimitive(KeyError):
    pass


#: The reasons an ablation may conclude nothing, as codes rather than prose.
#: Closure holds the INCONCLUSIVE population to this ALLOWLIST and to a pin
#: per cause. Adding one is a decision, and its pin comes with it.
ABLATION_RETAINED_STATE_LACKS_PRIMITIVE: Final[str] = "retained_state_lacks_primitive"
ABLATION_STATE_COULD_NOT_BE_BUILT: Final[str] = "ablated_state_could_not_be_built"
ABLATION_SUBSTITUTE_WAS_IDENTICAL: Final[str] = "substitute_identical_to_retained"


class Verdict(StrEnum):
    REPRODUCED = "PASS"

    DIVERGED = "FAIL"

    INCONCLUSIVE = "INCONCLUSIVE"

    CONTRADICTED = "CONTRADICTED"

    VENDOR_BASELINE_UNAVAILABLE = "VENDOR_BASELINE_UNAVAILABLE"


class CaseVerdict(StrEnum):
    # run_case can only ever reach these two, so the other three were unreachable
    # at this level -- and a gate reading `verdicts["CONTRADICTED"] == 0` over case
    # results was therefore constant-true. They are now unrepresentable here.
    REPRODUCED = "PASS"

    DIVERGED = "FAIL"


@dataclass(frozen=True)
class Skipped:
    consumer_id: str
    what: str
    why: str


_skipped: list[Skipped] = []
_considered: list[Skipped] = []


def note_skip(consumer_id: str, what: str, why: str) -> None:
    _skipped.append(Skipped(consumer_id=consumer_id, what=what, why=why))


def note_considered(consumer_id: str, what: str, why: str) -> None:
    _considered.append(Skipped(consumer_id=consumer_id, what=what, why=why))


def considered() -> tuple[Skipped, ...]:
    return tuple(sorted(set(_considered), key=lambda one: (one.consumer_id, one.what, one.why)))


def skipped() -> tuple[Skipped, ...]:
    return tuple(sorted(set(_skipped), key=lambda one: (one.consumer_id, one.what, one.why)))


@dataclass(frozen=True)
class Fixture:
    name: str
    path: str
    sha256: str
    kind: str
    note: str = ""


@dataclass(frozen=True)
class Artifact:
    name: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str
    values: Float32Array | UInt8Array | None = None


class RetainedState:
    def __init__(self, **values: object) -> None:
        self._values: dict[str, object] = {key: one for key, one in values.items() if one is not None}

        self._durable: dict[str, int] = {}

    def priced(self, durable: dict[str, int]) -> RetainedState:
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
        held = {name: one for name, one in self._values.items() if name != key}
        return RetainedState(**held).priced({k: v for k, v in self._durable.items() if k != key})

    def replacing(self, key: str, value: object) -> RetainedState:
        return RetainedState(**{**self._values, key: value}).priced(
            {k: v for k, v in self._durable.items() if k != key}
        )

    def sizes(self) -> dict[str, int]:
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
        return self._array(key)

    def pixels(self, key: str) -> UInt8Array:
        return self._array(key).astype(np.uint8, copy=False)

    def points(self, key: str) -> Float32Array:
        return self._array(key).astype(np.float32, copy=False)

    def integers(self, key: str) -> Int64Array:
        return self._array(key).astype(np.int64, copy=False)

    def pair(self, key: str) -> tuple[int, int]:
        value = self._require(key)
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(f"{key!r} is {value!r}, not a two-element tuple")
        first, second = value
        if not isinstance(first, (int, float)) or not isinstance(second, (int, float)):
            raise TypeError(f"{key!r} holds {type(first).__name__} and {type(second).__name__}, not numbers")
        return int(first), int(second)

    def number(self, key: str) -> float:
        value = self._require(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{key!r} is {value!r}, not a number")
        return float(value)

    def flag(self, key: str) -> bool:
        return bool(self._values.get(key, False))

    def text(self, key: str) -> str:
        value = self._require(key)
        if not isinstance(value, str):
            raise TypeError(f"{key!r} is {type(value).__name__}, not a string")
        return value


def settled_by_measurement(method: str) -> bool:
    return method in {"exact_bytes", "shape"} or method.startswith("allclose")


@dataclass(frozen=True)
class Ablation:
    primitive: str
    expect_breaks: bool
    kind: str = "removal"

    compare_method: str = ""

    swap: str = ""

    observed_break: bool | None = None
    verdict: Verdict | None = None
    detail: str = ""

    #: WHY an INCONCLUSIVE ablation concluded nothing, as a code the writer
    #: chooses -- never parsed back out of `detail`, which would guard a
    #: spelling. A count cannot see what it is counting.
    cause: str = ""

    def __post_init__(self) -> None:
        if self.kind == "substitution" and not self.swap:
            raise ValueError(f"the substitution for {self.primitive!r} names no swap")
        if self.kind == "removal" and self.swap:
            raise ValueError(f"the removal of {self.primitive!r} carries swap={self.swap!r}")
        if self.kind not in ("removal", "substitution"):
            raise ValueError(f"{self.kind!r} is not an ablation kind; removal and substitution are")

    @property
    def selector(self) -> str:
        return self.swap or self.primitive


@dataclass(frozen=True)
class Measurement:
    name: str
    unit: str
    value: float | None = None
    basis: str = ""
    detail: str = ""


@dataclass
class Case:
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

    __test__: bool = False


@dataclass
class CaseResult:
    case: str
    consumer_id: str
    tier: Tier
    verdict: CaseVerdict
    fixture_sha256: str
    baseline: Artifact | None = None
    replay: Artifact | None = None
    comparison: str = ""
    max_abs_diff: float | None = None
    ablations: tuple[Ablation, ...] = ()
    measurements: tuple[Measurement, ...] = ()
    retained_bytes: dict[str, int] = field(default_factory=dict)

    seconds: float = 0.0

    __test__: bool = False


@runtime_checkable
class ConsumerRunner(Protocol):
    consumer_id: str

    def cases(self) -> tuple[Case, ...]: ...

    def baseline(self, case: Case) -> Artifact: ...

    def replay(self, case: Case, retained: dict[str, object]) -> Artifact: ...


class Registry:
    def __init__(self) -> None:
        self._cases: dict[str, Case] = {}

    def add(self, case: Case) -> None:
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
