from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from compat.harness import provenance

ROOT: Final[Path] = Path(__file__).resolve().parent.parent


CACHE_ROOT: Final[Path] = ROOT / ".cache"


_held: dict[str, str] = {}


#: The codec every entry is written and read through. It MUST contribute to each
#: kind's namespace: without it a codec edit left the hash identical and entries
#: written under one codec were silently reinterpreted under the next.
CODEC: Final[tuple[str, ...]] = ("compat/corpus/cache.py", "vision/facestore.py")


def enabled() -> bool:
    return os.environ.get("COMPAT_CACHE", "1") != "0"


CONTRIBUTORS: Final[dict[str, tuple[str, ...]]] = {
    "frame": ("compat/corpus/loaded.py", *CODEC),
    "ours": (
        "compat/corpus/loaded.py",
        "compat/producers/insightface_pass.py",
        "vision/faces.py",
        *CODEC,
    ),
}


def codec_is_declared() -> None:
    # Enumerated from the registry itself, so a kind added later cannot quietly
    # omit the codec that serves it.
    undeclared = sorted(kind for kind, held in CONTRIBUTORS.items() if not set(CODEC) <= set(held))
    if undeclared:
        raise KeyError(
            f"{undeclared} do not name the codec in CONTRIBUTORS. A codec edit would leave their "
            f"namespace hash identical and reinterpret entries written under the previous one."
        )


WEIGHTED: Final[frozenset[str]] = frozenset({"ours"})


def namespace(kind: str) -> str:
    codec_is_declared()
    if kind not in _held:
        from compat.harness import identity as evidence_identity

        parts: dict[str, Any] = {"runtime": provenance.runtime_identity(), "code": {}}
        for relative in CONTRIBUTORS[kind]:
            where = ROOT.parent / relative
            if not where.is_file():
                raise FileNotFoundError(f"{relative} determines the {kind!r} cache and is not on disk")
            parts["code"][relative] = evidence_identity.sha256_of(where.read_bytes())
        if kind in WEIGHTED:
            parts["weights"] = evidence_identity.weight_digests()
        canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        _held[kind] = evidence_identity.sha256_of(canonical)
    return _held[kind]


def slot(kind: str, name: str) -> Path:
    return CACHE_ROOT / kind / namespace(kind) / name


def _temp(final: Path, suffix: str) -> Path:
    return final.with_name(f"{final.name}.{os.getpid()}{suffix}")


def frame_get(sha: str) -> npt.NDArray[np.uint8] | None:
    if not enabled():
        return None
    where = slot("frame", f"{sha}.npy")
    if not where.is_file():
        return None
    try:
        with where.open("rb") as handle:
            held = np.load(handle, allow_pickle=False)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        missed("frame", "unreadable")
        return None
    if held.dtype != np.uint8:
        missed("frame", "wrong_dtype")
        return None
    return np.ascontiguousarray(held)


def frame_put(sha: str, frame: npt.NDArray[np.uint8]) -> None:
    if not enabled():
        return
    where = slot("frame", f"{sha}.npy")
    where.parent.mkdir(parents=True, exist_ok=True)
    beside = _temp(where, ".raw")
    try:
        with beside.open("wb") as handle:
            np.save(handle, frame, allow_pickle=False)
        back = np.load(beside, allow_pickle=False)
        if back.dtype != frame.dtype or back.shape != frame.shape or not np.array_equal(back, frame):
            missed("frame", "round_trip_disagreed")
            beside.unlink(missing_ok=True)
            return
        os.replace(beside, where)
    except (OSError, ValueError):
        missed("frame", "unwritable")
        beside.unlink(missing_ok=True)


def _same(one: Any, two: Any) -> bool:
    if set(one.keys()) != set(two.keys()):
        return False
    for key in one:
        first, second = one[key], two[key]
        if type(first) is not type(second):
            return False
        if isinstance(first, np.ndarray):
            if first.dtype != second.dtype or first.shape != second.shape or not np.array_equal(first, second):
                return False
        elif first != second:
            return False
    return True


class _NeverRaised(Exception):
    """Stands in for a refusal this build's facestore does not define.

    envelope's capture-level `UnregisteredContainer` lands with the sgface3
    envelope; until that merges there is nothing here to catch, and a
    fallback that nothing raises keeps the branch INERT rather than
    matching an exception it was not written for.
    """


def _unregistered(facestore: Any) -> type[BaseException]:
    return getattr(facestore, "UnregisteredContainer", _NeverRaised)


def _read(where: Path) -> Any | None:
    """One stored entry, or a MISS THAT WAS COUNTED -- never a silent None.

    Every way this can fail is tolerated and none is hidden, which are
    independent choices. `refused` is the coarse one and says so: thaw
    raises ValueError for three different facts -- a foreign or superseded
    magic, a failed digest, and a node type this build does not know. The
    last is the READ side of the capture boundary, and how a record naming
    an adapter this build lacks will arrive. They cannot be told apart
    except by matching the exception's message, which is the string-guard
    defect; separating them needs facestore to raise distinct TYPES, which
    is its owner's change, not this one. Until then the COUNT is what
    makes a cache that stopped reading its own entries visible at all.
    """
    from insightface.app.common import Face

    from vision import facestore

    try:
        raw = where.read_bytes()
    except OSError:
        missed("ours", "unreadable")
        return None
    try:
        # thaw verifies the trailing digest, so corrupt bytes RAISE rather than
        # decode into something plausible. The old read path verified nothing.
        held = facestore.thaw(raw)
    except facestore.Unpreservable:
        missed("ours", "unpreservable")
        return None
    except ValueError:
        missed("ours", "refused")
        return None
    try:
        record = held.record
    except TypeError:
        # A non-mapping root refuses with a PLAIN TypeError, which the clause
        # above cannot catch -- Unpreservable is a TypeError subclass, its
        # parent is not. Uncaught it crashed the load: G8b's shape, on read.
        missed("ours", "not_a_mapping")
        return None
    return Face(**record)


def face_get(sha: str) -> Any | None:
    if not enabled():
        return None
    where = slot("ours", f"{sha}.sgface")
    if not where.is_file():
        return None
    return _read(where)


def face_put(sha: str, face: Any) -> None:
    if not enabled():
        return
    from vision import facestore

    where = slot("ours", f"{sha}.sgface")
    where.parent.mkdir(parents=True, exist_ok=True)
    beside = _temp(where, ".raw")
    try:
        # Inside the try, unlike the encode it replaces: that one sat above the
        # `try` whose `except` named the very exception it raises, so a value the
        # codec could not carry crashed the load instead of missing the cache.
        blob = facestore.freeze(
            {str(key): face[key] for key in face},
            producer="compat.corpus.cache",
            producer_version=namespace("ours"),
            container=f"{type(face).__module__}.{type(face).__qualname__}",
        )
        with beside.open("wb") as handle:
            handle.write(blob)
        back = _read(beside)
        if back is None or not _same(face, back):
            # The write's own round trip disagreed. It was a bare `return`:
            # the one check proving the entry is faithful could fail on every
            # entry and the run would look like a cache that simply never hit.
            missed("ours", "round_trip_disagreed")
            return
        os.replace(beside, where)
    except _unregistered(facestore):
        # LOUD: a container this build cannot rebuild is a CAPTURE-level
        # refusal, not a value this codec cannot carry. Eaten, one machine
        # stores what another refuses and both report success.
        raise
    except (OSError, ValueError, facestore.Unpreservable):
        # Value-level, and a quiet miss is the ruled behaviour (G8b) -- quiet
        # to the caller, never to the run.
        missed("ours", "unwritable")
        return
    finally:
        beside.unlink(missing_ok=True)


_counts: dict[str, int] = {}


def note(kind: str, hit: bool) -> None:
    key = f"{kind}_{'hit' if hit else 'miss'}"
    _counts[key] = _counts.get(key, 0) + 1


def missed(kind: str, why: str) -> None:
    """A tolerated failure, COUNTED, and surfaced by `statistics()`.

    Tolerating an error and hiding it are independent choices and only the
    first is ever permitted. A cache may miss on an entry it cannot read;
    a cache that has silently STOPPED CACHING must be readable off the run
    rather than inferred from it being slow.
    """
    key = f"{kind}_miss_{why}"
    _counts[key] = _counts.get(key, 0) + 1


def statistics() -> dict[str, Any]:
    namespaces = {kind: namespace(kind)[:16] for kind in CONTRIBUTORS} if enabled() else {}
    return {"enabled": enabled(), "namespaces": namespaces, **_counts}
