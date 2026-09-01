from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt

ROOT: Final[Path] = Path(__file__).resolve().parent.parent


INVENTORY: Final[Path] = ROOT / "generated" / "producer_inventory.json"


def emitted_keys() -> frozenset[str]:
    if not INVENTORY.is_file():
        return frozenset()
    held: dict[str, Any] = json.loads(INVENTORY.read_text(encoding="utf-8"))
    return frozenset(held.get("fields", {}))


class Observation:
    def __init__(self, values: dict[str, npt.NDArray[np.generic]]) -> None:
        self._values = dict(values)

    @classmethod
    def of(cls, face: Any) -> Observation:
        return cls({str(key): np.asarray(face[key]) for key in face if face[key] is not None})

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._values))

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __getitem__(self, key: str) -> npt.NDArray[np.generic]:
        if key not in self._values:
            raise KeyError(f"the observation carries no {key!r}: it holds {self.keys()}")
        return self._values[key]

    def get(self, key: str) -> npt.NDArray[np.generic] | None:
        return self._values.get(key)


class StorageContract(Protocol):
    name: str

    described: str

    def round_trip(self, face: Any, frame: npt.NDArray[np.uint8], sha: str) -> Observation: ...
