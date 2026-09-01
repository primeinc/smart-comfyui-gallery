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


_BUILTIN: Final[dict[str, Any]] = {"int": int, "float": float, "bool": bool, "str": str}


_SCALAR: Final[str] = "s__"


_held: dict[str, str] = {}


def enabled() -> bool:
    return os.environ.get("COMPAT_CACHE", "1") != "0"


CONTRIBUTORS: Final[dict[str, tuple[str, ...]]] = {
    "frame": ("compat/corpus/loaded.py",),
    "ours": (
        "compat/corpus/loaded.py",
        "compat/producers/insightface_pass.py",
        "vision/faces.py",
    ),
}


WEIGHTED: Final[frozenset[str]] = frozenset({"ours"})


def namespace(kind: str) -> str:
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
        return None
    if held.dtype != np.uint8:
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
            beside.unlink(missing_ok=True)
            return
        os.replace(beside, where)
    except (OSError, ValueError):
        beside.unlink(missing_ok=True)


def _parts(face: Any) -> tuple[dict[str, Any], dict[str, str]]:
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
    try:
        kinds: dict[str, str] = json.loads(kinds_at.read_text(encoding="utf-8"))

        with body.open("rb") as handle, np.load(handle, allow_pickle=False) as held:
            payload = {name: np.asarray(held[name]) for name in held.files}
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
        return None
    return _restore(payload, kinds)


def face_get(sha: str) -> Any | None:
    if not enabled():
        return None
    body, kinds_at = slot("ours", f"{sha}.npz"), slot("ours", f"{sha}.json")
    if not body.is_file() or not kinds_at.is_file():
        return None
    return _read(body, kinds_at)


def face_put(sha: str, face: Any) -> None:
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

        os.replace(kinds_beside, kinds_at)
        os.replace(beside, body)
    except (OSError, ValueError, zipfile.BadZipFile):
        return
    finally:
        beside.unlink(missing_ok=True)
        kinds_beside.unlink(missing_ok=True)


_counts: dict[str, int] = {}


def note(kind: str, hit: bool) -> None:
    key = f"{kind}_{'hit' if hit else 'miss'}"
    _counts[key] = _counts.get(key, 0) + 1


def statistics() -> dict[str, Any]:
    namespaces = {kind: namespace(kind)[:16] for kind in CONTRIBUTORS} if enabled() else {}
    return {"enabled": enabled(), "namespaces": namespaces, **_counts}
