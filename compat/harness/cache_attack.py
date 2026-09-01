from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from compat.corpus import cache
from compat.harness import identity as evidence_identity

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
GENERATED: Final[Path] = ROOT / "generated"
REPO: Final[Path] = ROOT.parent

#: What escaping face_put would mean a crash instead of a cache miss. Unpreservable
#: is a TypeError, which is why the encode sitting outside the try mattered.
ESCAPES: Final[tuple[type[BaseException], ...]] = (TypeError, ValueError, OSError, KeyError, AttributeError)


@dataclass
class Probe:
    name: str
    held: bool
    detail: str

    @property
    def mark(self) -> str:
        return "ok " if self.held else "RED"


def _namespaces() -> dict[str, str]:
    cache._held.clear()
    return {kind: cache.namespace(kind) for kind in cache.CONTRIBUTORS}


def _codec_declared_for_every_kind() -> Probe:
    # `set(CODEC) <= set(held)` is vacuously true when CODEC is empty, so the
    # subset test alone passes at the one value that disables the whole guard.
    # CODEC must name real files, and the files must be the ones doing the coding.
    missing = sorted(one for one in cache.CODEC if not (REPO / one).is_file())
    undeclared = sorted(kind for kind, held in cache.CONTRIBUTORS.items() if not set(cache.CODEC) <= set(held))
    detail = f"{len(cache.CONTRIBUTORS)} kind(s), all naming {len(cache.CODEC)} codec file(s) that exist"
    if not cache.CODEC:
        detail = "CODEC IS EMPTY: the subset test would pass for every kind"
    elif missing:
        detail = f"CODEC NAMES FILES NOT ON DISK: {missing}"
    elif undeclared:
        detail = f"NOT DECLARED for {undeclared}"
    return Probe("the codec contributes to every kind", bool(cache.CODEC) and not missing and not undeclared, detail)


def _guard_refuses_an_undeclared_kind() -> Probe:
    # The guard enumerates from CONTRIBUTORS, so a kind added without the codec must
    # fail. Proven by adding one rather than by reading the code.
    held = dict(cache.CONTRIBUTORS)
    try:
        cache.CONTRIBUTORS["a_kind_someone_added"] = ("compat/corpus/loaded.py",)
        cache.codec_is_declared()
    except KeyError as why:
        return Probe("a kind without the codec is refused", True, str(why)[:88])
    finally:
        cache.CONTRIBUTORS.clear()
        cache.CONTRIBUTORS.update(held)
    return Probe("a kind without the codec is refused", False, "THE GUARD ACCEPTED A KIND THAT OMITS THE CODEC")


def _codec_edit_moves_every_namespace() -> Probe:
    # The hole itself: with the codec outside CONTRIBUTORS the hash was identical
    # with a byte appended, so old entries were re-read under the new codec.
    before = _namespaces()
    probe = REPO / cache.CODEC[0]
    original = probe.read_bytes()
    try:
        with probe.open("ab") as handle:
            handle.write(b"\n# transient codec-edit control\n")
        during = _namespaces()
    finally:
        with probe.open("wb") as handle:
            handle.write(original)
    after = _namespaces()

    unmoved = sorted(kind for kind in before if before[kind] == during[kind])
    restored = after == before
    detail = f"all {len(before)} kind(s) re-namespaced, and restored"
    if unmoved:
        detail = f"UNMOVED {unmoved}"
    elif not restored:
        detail = "THE PROBE DID NOT RESTORE cache.py"
    return Probe("a codec edit moves every namespace", not unmoved and restored, detail)


def _corrupt_bytes_are_refused() -> Probe:
    # The read path verified nothing, so an entry written under one codec decoded
    # into something plausible under the next. thaw checks its trailing digest.
    from vision import facestore

    blob = facestore.freeze(
        {"embedding": np.arange(4, dtype=np.float32)},
        producer="compat.corpus.cache",
        producer_version="probe",
        container="dict",
    )
    broken = bytearray(blob)
    broken[len(broken) // 2] ^= 0xFF
    try:
        facestore.thaw(bytes(broken))
    except (ValueError, facestore.Unpreservable) as why:
        return Probe("a corrupted entry is refused", True, str(why)[:88])
    return Probe("a corrupted entry is refused", False, "A CORRUPTED ENVELOPE DECODED INTO A VALUE")


def _unpreservable_is_a_miss_not_a_crash() -> Probe:
    # G8b: the encode sat ABOVE the try whose except named the very exception it
    # raises, so one nested structured field crashed the load instead of missing.
    held = {"embedding": np.arange(2, dtype=np.float32), "nested": {"a": object()}}
    try:
        cache.face_put("cache_attack_probe", held)
    except ESCAPES as why:
        return Probe("an uncarryable value misses, never crashes", False, f"{type(why).__name__}: {why}"[:88])
    return Probe("an uncarryable value misses, never crashes", True, "face_put returned quietly")


def _counted(why: str, act: Any) -> tuple[int, Any]:
    """How much `act` moved one miss counter, and what it returned."""
    key = f"ours_miss_{why}"
    before = cache._counts.get(key, 0)
    out = act()
    return cache._counts.get(key, 0) - before, out


def _an_unregistered_container_is_loud(tmp: Path) -> Probe:
    """The WRITE side of the boundary, and the one refusal here that must
    not be a miss.

    A container this build cannot rebuild is a capture-level refusal: eaten,
    identical producer output is stored by a process whose import graph
    registered the adapter and refused by one that did not, both reporting
    success -- capture behaviour decided by the import graph.

    envelope's `UnregisteredContainer` lands with sgface3, which has not
    merged here, so the type is INJECTED. That is the point: the branch is
    inert in this tree and would otherwise be a check that cannot fail, so
    the control supplies the exception rather than waiting for the merge.

    IT IS INJECTED AS AN `Unpreservable` SUBCLASS, and that is what makes
    this probe discriminate. Injected as a bare Exception it propagated
    whether the loud branch existed or not -- nothing else catches a bare
    Exception here -- so the probe passed with the branch deleted, which is
    the shape it was written to catch. As a subclass, the swallowing clause
    below WOULD take it, and only the loud branch above lets it out.
    """
    from vision import facestore

    name, held = "an unregistered container propagates", getattr(facestore, "UnregisteredContainer", None)
    encode = facestore.freeze

    class UnregisteredContainer(facestore.Unpreservable):
        pass

    def refuses(*_a: Any, **_k: Any) -> bytes:
        raise UnregisteredContainer("no adapter registered for vision.facestore.Face")

    try:
        facestore.UnregisteredContainer = UnregisteredContainer
        facestore.freeze = refuses
        moved, _ = _counted("unwritable", lambda: cache.face_put("cache_attack_unregistered", {"a": 1}))
    except UnregisteredContainer as why:
        return Probe(name, True, f"propagated: {why}"[:88])
    except ESCAPES as why:
        return Probe(name, False, f"WRONG EXCEPTION {type(why).__name__}: {why}"[:88])
    finally:
        facestore.freeze = encode
        if held is None:
            del facestore.UnregisteredContainer
        else:
            facestore.UnregisteredContainer = held
    return Probe(name, False, f"SWALLOWED as a miss (unwritable moved {moved}); the import graph decides capture")


def _a_tolerated_miss_is_counted(tmp: Path) -> Probe:
    """The other half of the ruling: quiet to the CALLER, never to the run.

    A cache may miss on an entry it cannot carry. What it may not do is
    leave the run unable to tell that from a cache that simply never hit.
    """
    name = "a tolerated miss moves a counter"
    held = {"embedding": np.arange(2, dtype=np.float32), "nested": {"a": object()}}
    moved, _ = _counted("unwritable", lambda: cache.face_put("cache_attack_counted", held))
    if moved != 1:
        return Probe(name, False, f"UNCOUNTED: ours_miss_unwritable moved {moved}, not 1")
    return Probe(name, True, "an uncarryable value misses AND says so")


def _a_non_mapping_root_misses(tmp: Path) -> Probe:
    """`Native.record` refuses a non-mapping root with a plain TypeError.

    Unpreservable is a TypeError subclass; its parent is not, so the read
    path's clause could not catch it and the load crashed instead of
    missing -- G8b's shape on the read side, found by walking this one to
    its end rather than by an entry ever arriving that way.
    """
    from vision import facestore

    name = "a non-mapping root misses, never crashes"
    where = tmp / "root.sgface"
    blob = facestore.freeze(
        np.arange(3, dtype=np.float32),
        producer="compat.corpus.cache",
        producer_version="probe",
        container="numpy.ndarray",
    )
    with where.open("wb") as handle:
        handle.write(blob)
    try:
        moved, back = _counted("not_a_mapping", lambda: cache._read(where))
    except ESCAPES as why:
        return Probe(name, False, f"CRASHED {type(why).__name__}: {why}"[:88])
    if back is not None:
        return Probe(name, False, "a non-mapping root read back as a record")
    return Probe(name, moved == 1, f"missed, and ours_miss_not_a_mapping moved {moved}")


def run_all() -> list[Probe]:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="cache_attack_") as raw:
        tmp = Path(raw)
        return [
            _codec_declared_for_every_kind(),
            _guard_refuses_an_undeclared_kind(),
            _codec_edit_moves_every_namespace(),
            _corrupt_bytes_are_refused(),
            _unpreservable_is_a_miss_not_a_crash(),
            _an_unregistered_container_is_loud(tmp),
            _a_tolerated_miss_is_counted(tmp),
            _a_non_mapping_root_misses(tmp),
        ]


def main() -> int:
    held = run_all()
    print("corpus-cache codec controls\n")
    for one in held:
        print(f"{one.mark} {one.name:<44} {one.detail}")

    failing = [one.name for one in held if not one.held]
    print(f"\n{len(held)} probe(s), {len(failing)} failing: {failing or 'none'}")

    GENERATED.mkdir(parents=True, exist_ok=True)
    body: dict[str, Any] = {
        "identity": str(evidence_identity.identity()["digest"]),
        "codec": list(cache.CODEC),
        "contributors": {kind: list(held) for kind, held in sorted(cache.CONTRIBUTORS.items())},
        "probes": [asdict(one) for one in held],
        "failing": failing,
    }
    with (GENERATED / "cache_controls.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(body, indent=2, sort_keys=True))
        handle.write("\n")
    print(f"wrote {GENERATED / 'cache_controls.json'}")
    return 0 if not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
