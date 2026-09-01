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


def run_all() -> list[Probe]:
    return [
        _codec_declared_for_every_kind(),
        _guard_refuses_an_undeclared_kind(),
        _codec_edit_moves_every_namespace(),
        _corrupt_bytes_are_refused(),
        _unpreservable_is_a_miss_not_a_crash(),
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
