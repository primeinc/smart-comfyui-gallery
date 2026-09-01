from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from compat.harness import identity as evidence_identity
from compat.harness.identity import PARTS, digest_of, forget, identity

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
GENERATED: Final[Path] = ROOT / "generated"
REPO: Final[Path] = ROOT.parent


#: Paths that MUST be digested, by the part that has to carry them. Keys are as the
#: part records them: `sources` is relative to compat/, the rest to the repo root.
COVERED: Final[dict[str, tuple[str, ...]]] = {
    "sources": ("__init__.py", "ty.toml", "pyrefly.toml"),
    "application": ("db/schema.sql",),
    # Named from the FINDING, not from the implementation: lefthook.yml decides
    # whether the gates run at all, and it plus the next three were missed by the
    # include list this replaced. The implementation now derives from git.
    "gates": (
        "compat.just",
        "justfile",
        "pyproject.toml",
        "uv.lock",
        "conftest.py",
        "lefthook.yml",
        "pytest.ini",
        ".vale.ini",
        "biome.json",
    ),
}

#: A prefix that must appear at least once, so a whole tree cannot silently drop out.
POPULATED: Final[dict[str, tuple[str, ...]]] = {"gates": ("tests/", "metaparse/")}


@dataclass
class Control:
    name: str
    held: bool
    detail: str

    @property
    def mark(self) -> str:
        return "ok " if self.held else "RED"


def _coverage(now: dict[str, Any]) -> list[Control]:
    out: list[Control] = []
    for part, wanted in COVERED.items():
        held: dict[str, str] = now[part]
        missing = [one for one in wanted if one not in held]
        out.append(
            Control(
                f"{part} digests its declared files",
                not missing,
                f"{len(held)} file(s); all {len(wanted)} present" if not missing else f"MISSING {missing}",
            )
        )
    for part, prefixes in POPULATED.items():
        held = now[part]
        empty = [one for one in prefixes if not any(key.startswith(one) for key in held)]
        out.append(
            Control(
                f"{part} reaches every declared tree",
                not empty,
                "every tree contributes at least one file" if not empty else f"NOTHING under {empty}",
            )
        )
    return out


def _every_tracked_root_file_is_digested(now: dict[str, Any]) -> Control:
    # The exclusion's other half: what git tracks at the root, minus the declared
    # ignore list, must all be present. An include list could silently shrink;
    # this fails the moment a tracked root file is neither digested nor declared.
    from compat.harness.identity import GATE_IGNORED, tracked_root_files

    tracked = [one for one in tracked_root_files(REPO) if one not in GATE_IGNORED]
    held: dict[str, str] = now["gates"]
    missing = [one for one in tracked if one not in held]
    return Control(
        "every tracked root file is digested or declared",
        bool(tracked) and not missing,
        f"{len(tracked)} tracked, {len(GATE_IGNORED)} declared-ignored, none undigested"
        if not missing
        else f"TRACKED BUT NEITHER DIGESTED NOR IGNORED: {missing}",
    )


def _every_just_module_on_disk_is_digested(now: dict[str, Any]) -> Control:
    # COVERED names "compat.just" and GATE_GLOBS globs "*.just", so a glob narrowed
    # back to one file passed both. The tree is the independent source: narrow the
    # glob and the other modules stay on disk and fall out of the digest.
    on_disk = sorted(one.name for one in REPO.glob("*.just"))
    held: dict[str, str] = now["gates"]
    missing = [one for one in on_disk if one not in held]
    return Control(
        "every .just module on disk is digested",
        bool(on_disk) and not missing,
        f"{len(on_disk)} module(s) on disk, all digested" if not missing else f"ON DISK BUT NOT DIGESTED: {missing}",
    )


def _every_computed_part_is_digested(now: dict[str, Any]) -> Control:
    # _sensitivity iterates PARTS, and digest_of uses PARTS too, so a part dropped
    # from PARTS is computed, printed, and tested by nobody. identity()'s own keys
    # are the independent source: what it computes must be what the digest covers.
    computed = set(now) - {"digest"}
    undigested = sorted(computed - set(PARTS))
    return Control(
        "every computed part reaches the digest",
        not undigested,
        f"{len(computed)} part(s), all in PARTS" if not undigested else f"COMPUTED BUT NOT DIGESTED: {undigested}",
    )


def _sensitivity(now: dict[str, Any]) -> list[Control]:
    # Coverage says the bytes are hashed. This says the hash reaches the digest:
    # a part could be computed, printed, and left out of digest_of entirely.
    parts = {key: now[key] for key in PARTS}
    base = digest_of(parts)
    out: list[Control] = []
    for part in PARTS:
        held = parts[part]
        moved = {**held, "__probe__": "x"} if isinstance(held, dict) else f"{held}x"
        out.append(
            Control(
                f"{part} moves the digest",
                digest_of({**parts, part: moved}) != base,
                "a change to it mints a new identity" if digest_of({**parts, part: moved}) != base else "IGNORED",
            )
        )
    return out


def _live_probe() -> Control:
    # End to end against the filesystem, not the dict: a new .just module at the
    # repo root must move the real digest. Additive, so nothing existing is risked.
    probe = REPO / "_identity_probe.just"
    forget()
    before = str(identity()["digest"])
    try:
        with probe.open("w", encoding="utf-8", newline="") as handle:
            handle.write("# transient identity control\n")
        forget()
        during = str(identity()["digest"])
    finally:
        probe.unlink(missing_ok=True)
    forget()
    after = str(identity()["digest"])
    return Control(
        "a new .just module moves the live digest",
        during != before and after == before,
        f"{before[:12]} -> {during[:12]} -> {after[:12]}"
        + ("" if during != before else "  THE NEW MODULE WAS INVISIBLE")
        + ("" if after == before else "  THE PROBE DID NOT CLEAN UP"),
    )


def _uncovered_probe() -> Control:
    # The negative half. generated/ is output, not input: if writing there moved the
    # identity, every lane would invalidate the evidence it just produced.
    probe = GENERATED / "_identity_probe.json"
    forget()
    before = str(identity()["digest"])
    GENERATED.mkdir(parents=True, exist_ok=True)
    try:
        with probe.open("w", encoding="utf-8", newline="") as handle:
            handle.write("{}\n")
        during = str(identity()["digest"])
    finally:
        probe.unlink(missing_ok=True)
    return Control(
        "generated output does NOT move the digest",
        during == before,
        "writing evidence does not invalidate evidence" if during == before else f"{before[:12]} -> {during[:12]}",
    )


def run_all() -> list[Control]:
    now = identity()
    return [
        *_coverage(now),
        _every_just_module_on_disk_is_digested(now),
        _every_tracked_root_file_is_digested(now),
        _every_computed_part_is_digested(now),
        *_sensitivity(now),
        _live_probe(),
        _uncovered_probe(),
    ]


def main() -> int:
    held = run_all()
    print("evidence-identity controls\n")
    for one in held:
        print(f"{one.mark} {one.name:<44} {one.detail}")

    failing = [one.name for one in held if not one.held]
    print(f"\n{len(held)} control(s), {len(failing)} failing: {failing or 'none'}")

    GENERATED.mkdir(parents=True, exist_ok=True)
    body = {
        "identity": str(evidence_identity.identity()["digest"]),
        "controls": [asdict(one) for one in held],
        "failing": failing,
    }
    with (GENERATED / "identity_controls.json").open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(body, indent=2, sort_keys=True))
        handle.write("\n")
    print(f"wrote {GENERATED / 'identity_controls.json'}")
    return 0 if not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
