"""The storage contract a candidate must satisfy.

The vocabulary is the producer's. `Observation.of` iterates `face.keys()` and
names no field. A schema is a candidate implementation of this contract, never
its definition: a field list taken from a table can only measure whether that
table preserves what it already has room for.

A candidate returns what survived its round trip. It does not declare what it
holds. An absent key compares as a shape divergence.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt

ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Written by `compat.producers.inventory`.
INVENTORY: Final[Path] = ROOT / "generated" / "producer_inventory.json"


def emitted_keys() -> frozenset[str]:
    """Producer keys recorded in generated evidence. Checks the live record;
    never selects from it."""
    if not INVENTORY.is_file():
        return frozenset()
    held: dict[str, Any] = json.loads(INVENTORY.read_text(encoding="utf-8"))
    return frozenset(held.get("fields", {}))


class Observation:
    """One face's emitted record, by key, in the producer's own dtypes.

    dtypes are not coerced. A candidate that changes the width of a value has
    changed it.
    """

    def __init__(self, values: dict[str, npt.NDArray[np.generic]]) -> None:
        self._values = dict(values)

    @classmethod
    def of(cls, face: Any) -> Observation:
        """The producer's record by iteration. `Face` subclasses dict
        (deepinsight/insightface@7fadd420c2351d0ffa8cac403421c1a3ed733365
        python-package/insightface/app/common.py:6), so `keys()` is the whole
        record including keys this suite does not name."""
        return cls({str(key): np.asarray(face[key]) for key in face if face[key] is not None})

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._values))

    def __iter__(self) -> Iterator[str]:
        """Sorted keys, so a caller iterates the record directly.

        Present so `for key in observation` reads as iteration over the
        record rather than over a `.keys()` view of something that is not
        a mapping.
        """
        return iter(self.keys())

    def __getitem__(self, key: str) -> npt.NDArray[np.generic]:
        if key not in self._values:
            raise KeyError(f"the observation carries no {key!r}: it holds {self.keys()}")
        return self._values[key]

    def get(self, key: str) -> npt.NDArray[np.generic] | None:
        return self._values.get(key)


class StorageContract(Protocol):
    """One candidate answer to what this application durably keeps.

    Two methods, no capability declaration: a candidate is graded on what
    comes back, not on what it claims.
    """

    name: str
    """Names this candidate in every case name."""

    described: str
    """What implements it, for the generated matrix."""

    def round_trip(self, face: Any, frame: npt.NDArray[np.uint8], sha: str) -> Observation:
        """Store, then read back, through real code.

        `frame` and `sha` are the only facts about the source a candidate
        gets. Nothing here re-opens the original.
        """
        ...
