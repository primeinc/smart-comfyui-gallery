"""The memo, persisted across the processes one run is split into.

`compat/corpus/loaded.py` stopped this suite recomputing work inside a
process (ab2bdef). Then `compat/harness/sharded.py` split the
population across six interpreters, because sixteen lanes holding every model
pack at once exhausted memory -- and a memo that lives in a dict dies with its
process. Six shards decode the same four corpus photographs, one of them
4896x6528, and re-run our own producer over each.

This gives that memo a floor it survives on. It is not a general cache, and it
is built so that serving a wrong answer is not a thing it can do:

    namespace  PER KIND, over the files that actually compute that kind of
               value plus the runtime, and the weight digests for a kind that
               comes out of a model. Anything that can change an answer
               changes the directory the answer is looked up in, so a stale
               entry is never found rather than found and trusted.
    key        the sha256 of the input bytes, inside that namespace.
    guard      every entry is read back and compared to the value in hand
               before it is kept. An entry that does not reconstruct exactly
               is discarded and the caller computes.

The guard is the part that earns the cache. A Face holds numpy arrays beside
numpy scalars, and `producers/insightface_pass._describe` branches on
`isinstance(value, np.ndarray)` FIRST. `det_score` is an np.float32: held in an
npz it comes back as a 0-d ARRAY, which that branch would inventory as
`kind="ndarray"`, `dtype="float32"`, `bytes_raw=4` where the producer reported
`kind="float"`, `dtype="float64"`, `bytes_raw=8`. The storage evidence is a
byte-cost table over exactly those fields, so the cache would have changed the
answer. Recording each value's type and asserting the round trip is what makes
that impossible rather than unlikely.

Upstream read for this file:
  refs/numpy/numpy/numpy/lib/_npyio_impl.py:505 `save` and :756 `_savez` append
  `.npy`/`.npz` to a PATH and leave a file OBJECT alone, so writes go through
  an open handle and the scratch name survives the rename.
  refs/numpy/numpy/numpy/lib/_npyio_impl.py:312 `load` returns an `NpzFile`
  that owns the descriptor only when it opened it, so reads take a path.
  refs/numpy/numpy/numpy/lib/_format_impl.py:157 -- an ndarray SUBCLASS is
  written as its data and read back as a plain ndarray, and :164 a structured
  dtype with empty field names comes back renamed f0, f1. Both would be a
  silently different value; both are refused by `_same`, which compares the
  exact type and dtype rather than the contents alone.
  refs/deepinsight/insightface/python-package/insightface/app/common.py:5
  `Face(dict)` -- `__init__(d=None, **kwargs)` setattrs every key, and
  `__setattr__` coerces list/tuple/dict. `normed_embedding`, `embedding_norm`
  and `sex` are properties over `embedding`/`gender`, so they are derived on
  access and never stored.

WHY THE NAMESPACE IS NOT THE TREE IDENTITY
------------------------------------------
It was, and that made the cache almost unable to hit. `identity()` covers the
manifest, every pinned commit, every weight, the runtime and the sha256 of
every compat source file, so editing a linter rule threw away four decoded
photographs -- one of them 4896x6528 -- and every detection over them. A cache
whose namespace changes on any edit anywhere is a directory of write-only
files.

`CONTRIBUTORS` names, per kind, the chain that computes that kind. A decoded
frame turns on `frame_of` and the runtime; nothing about a weight or a
manifest row can move one. A missing contributor raises rather than dropping
out of the digest, so a renamed file cannot take an entry's staleness with it.

Set COMPAT_CACHE=0 to disable.
"""

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

#: Gitignored. Every entry is derived, machine-local and reproducible by
#: running the suite; nothing here is evidence.
CACHE_ROOT: Final[Path] = ROOT / ".cache"

#: Values that must come back as Python builtins. Anything else recorded as a
#: non-array is a numpy scalar and is rebuilt as one, because the inventory
#: distinguishes them.
_BUILTIN: Final[dict[str, Any]] = {"int": int, "float": float, "bool": bool, "str": str}

#: Prefix marking a non-array value inside the npz. A zip entry name, so it
#: avoids characters that are illegal in a filename on either platform.
_SCALAR: Final[str] = "s__"

#: One namespace digest per kind, memoised per process: the files it hashes
#: are ones this suite never writes, so no answer can change while a process
#: runs.
_held: dict[str, str] = {}


def enabled() -> bool:
    """Off when COMPAT_CACHE is 0, on otherwise."""
    return os.environ.get("COMPAT_CACHE", "1") != "0"


#: What determines each kind of entry, by repo-relative path: every file in
#: the chain that computes the value, and nothing else. A `frame` is bytes
#: through `frame_of`; an `ours` face is that frame through the producer.
CONTRIBUTORS: Final[dict[str, tuple[str, ...]]] = {
    "frame": ("compat/corpus/loaded.py",),
    "ours": (
        "compat/corpus/loaded.py",
        "compat/producers/insightface_pass.py",
        "vision/faces.py",
    ),
}

#: Kinds whose value comes out of a model, and so turns on the weight digests
#: as well as on the code.
WEIGHTED: Final[frozenset[str]] = frozenset({"ours"})


def namespace(kind: str) -> str:
    """The digest THIS KIND of entry may be read and written under.

    Not the tree identity. That covered every compat source, every pinned
    commit and every weight, so editing a linter rule discarded every decoded
    frame and every detection -- a cache that cannot serve a stale value
    because it can hardly serve one at all.

    A missing contributor is an error rather than an omission: a renamed file
    would otherwise drop out of the digest silently and take its entries'
    staleness with it.
    """
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
    """Where one entry lives: <cache>/<kind>/<that kind's digest>/<name>."""
    return CACHE_ROOT / kind / namespace(kind) / name


def _temp(final: Path, suffix: str) -> Path:
    """A sibling name this process owns, so six of them cannot collide."""
    return final.with_name(f"{final.name}.{os.getpid()}{suffix}")


# --- decoded frames ----------------------------------------------------------


def frame_get(sha: str) -> npt.NDArray[np.uint8] | None:
    """The decoded frame held for these file bytes, or None."""
    if not enabled():
        return None
    where = slot("frame", f"{sha}.npy")
    if not where.is_file():
        return None
    try:
        # The handle is opened here rather than by `np.load`:
        # `_npyio_impl.load:471` calls `stack.pop_all()` before constructing
        # `NpzFile`, so a raising constructor leaks the descriptor.
        with where.open("rb") as handle:
            held = np.load(handle, allow_pickle=False)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        # An unreadable entry is a miss, never an error. The caller can always
        # recompute, and a cache able to fail a run is worse than no cache.
        return None
    if held.dtype != np.uint8:
        return None
    return np.ascontiguousarray(held)


def frame_put(sha: str, frame: npt.NDArray[np.uint8]) -> None:
    """Hold a decoded frame, only if it reads back identical."""
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
            beside.unlink(missing_ok=True)
            return
        os.replace(beside, where)
    except (OSError, ValueError):
        # ValueError as well as OSError: _format_impl.write_array:860 raises
        # it for an object array under allow_pickle=False and _read_bytes:1128
        # on a short read.
        beside.unlink(missing_ok=True)


# --- our own producer's face -------------------------------------------------
# `loaded.our_face` is the observation the application would store, and five
# consumer modules build their retained state from it in five processes.


def _parts(face: Any) -> tuple[dict[str, Any], dict[str, str]]:
    """One Face as an npz payload, plus the type each value came in as."""
    payload: dict[str, Any] = {}
    kinds: dict[str, str] = {}
    for key in face:
        name = str(key)
        value = face[key]
        kinds[name] = type(value).__name__
        if isinstance(value, np.ndarray):
            payload[name] = value
        else:
            payload[f"{_SCALAR}{name}"] = np.asarray(value)
    return payload, kinds


def _restore(payload: dict[str, npt.NDArray[Any]], kinds: dict[str, str]) -> Any:
    """A Face again, with every value the type it was recorded as.

    A builtin is rebuilt through its constructor; anything else was a numpy
    scalar and `array[()]` gives that scalar back at its own dtype. The two
    are not interchangeable to the inventory, which is the whole reason the
    types are recorded rather than inferred.
    """
    from insightface.app.common import Face

    held: dict[str, Any] = {}
    for stored, array in payload.items():
        if not stored.startswith(_SCALAR):
            held[stored] = array
            continue
        name = stored[len(_SCALAR) :]
        want = _BUILTIN.get(kinds.get(name, ""))
        held[name] = want(array.item()) if want is not None else array[()]
    return Face(**held)


def _same(one: Any, two: Any) -> bool:
    """Two Faces agreeing on every key, every type and every byte."""
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


def _read(body: Path, kinds_at: Path) -> Any | None:
    """One held Face, or None if anything about the entry does not answer."""
    try:
        kinds: dict[str, str] = json.loads(kinds_at.read_text(encoding="utf-8"))
        # `zipfile.BadZipFile` descends from Exception, NOT from ValueError, so
        # a truncated npz escaped an earlier version of this handler and raised
        # into the caller. Every entry read is arranged to be a miss instead.
        with body.open("rb") as handle, np.load(handle, allow_pickle=False) as held:
            payload = {name: np.asarray(held[name]) for name in held.files}
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
        return None
    return _restore(payload, kinds)


def face_get(sha: str) -> Any | None:
    """The producer's face for these file bytes, or None."""
    if not enabled():
        return None
    body, kinds_at = slot("ours", f"{sha}.npz"), slot("ours", f"{sha}.json")
    if not body.is_file() or not kinds_at.is_file():
        return None
    return _read(body, kinds_at)


def face_put(sha: str, face: Any) -> None:
    """Hold the producer's face, only if it reconstructs exactly.

    The comparison is against the object in hand, not against a schema. When
    this returns without writing, the only cost is that the next process
    computes what it would have computed anyway.
    """
    if not enabled():
        return
    payload, kinds = _parts(face)
    body, kinds_at = slot("ours", f"{sha}.npz"), slot("ours", f"{sha}.json")
    body.parent.mkdir(parents=True, exist_ok=True)
    beside, kinds_beside = _temp(body, ".raw"), _temp(kinds_at, ".raw")
    try:
        with beside.open("wb") as handle:
            np.savez(handle, **payload)
        kinds_beside.write_text(json.dumps(kinds, sort_keys=True), encoding="utf-8")
        back = _read(beside, kinds_beside)
        if back is None or not _same(face, back):
            return
        # The kinds land FIRST: `face_get` requires both, so a reader can find
        # a sidecar without a body (a miss) but never a body without the types
        # needed to read it correctly.
        os.replace(kinds_beside, kinds_at)
        os.replace(beside, body)
    except (OSError, ValueError, zipfile.BadZipFile):
        return
    finally:
        beside.unlink(missing_ok=True)
        kinds_beside.unlink(missing_ok=True)


# --- what it saved -----------------------------------------------------------

_counts: dict[str, int] = {}


def note(kind: str, hit: bool) -> None:
    """Record one lookup, for the run report."""
    key = f"{kind}_{'hit' if hit else 'miss'}"
    _counts[key] = _counts.get(key, 0) + 1


def statistics() -> dict[str, Any]:
    """Hits and misses per kind, and which namespace each was served from.

    One namespace per kind, so a report naming a single digest could not say
    which of them a `frame` hit and an `ours` miss were looked up under.
    """
    namespaces = {kind: namespace(kind)[:16] for kind in CONTRIBUTORS} if enabled() else {}
    return {"enabled": enabled(), "namespaces": namespaces, **_counts}
