from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from compat.harness import provision

ROOT: Final[Path] = Path(__file__).resolve().parent.parent


@dataclass
class Control:
    name: str
    held: bool
    detail: str

    @property
    def mark(self) -> str:
        return "ok " if self.held else "RED"


def control_wrong_digest_is_removed(where: Path) -> Control:
    target = where / "wrong.bin"
    target.write_bytes(b"not the weight")
    wrong = provision._verified(target, "0" * 64)
    return Control(
        "A wrong_digest_is_removed",
        bool(wrong) and not target.is_file(),
        f"reported {wrong[:80]!r}; file left on disk={target.is_file()}",
    )


def control_matching_digest_is_kept(where: Path) -> Control:
    target = where / "right.bin"
    body = b"the weight"
    target.write_bytes(body)
    wrong = provision._verified(target, hashlib.sha256(body).hexdigest())
    return Control(
        "B matching_digest_is_kept",
        not wrong and target.is_file(),
        f"reported {wrong!r}; file kept={target.is_file()}",
    )


def control_url_must_sit_under_source(where: Path) -> Control:
    held = provision.fetch(
        {
            "consumer": "control",
            "kind": "url",
            "source": "storage.googleapis.com/mediapipe-models",
            "url": "https://evil.example.com/face_landmarker.task",
            "path": "elsewhere.task",
        },
        where,
    )
    return Control(
        "C url_must_sit_under_source",
        held.verdict == "UNDECLARED" and not (where / "elsewhere.task").exists(),
        f"{held.verdict}: {held.detail[:90]}",
    )


def control_unknown_kind_is_undeclared(where: Path) -> Control:
    held = provision.fetch(
        {"consumer": "control", "kind": "carrier_pigeon", "source": "somewhere", "path": "bird.bin"}, where
    )
    return Control(
        "D unknown_kind_is_undeclared",
        held.verdict == "UNDECLARED" and "carrier_pigeon" in held.detail,
        f"{held.verdict}: {held.detail[:90]}",
    )


def control_empty_download_fails(where: Path) -> Control:
    empty = where / "empty.bin"
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_bytes(b"")
    held = provision.fetch({"consumer": "control", "kind": "url", "source": "x", "path": "empty.bin"}, where)

    return Control(
        "E empty_download_fails",
        held.verdict != "PRESENT",
        f"a zero-byte file reported {held.verdict}",
    )


def run_all() -> list[Control]:
    with tempfile.TemporaryDirectory(prefix="provision_attack_") as raw:
        where = Path(raw)
        return [
            control_wrong_digest_is_removed(where),
            control_matching_digest_is_kept(where),
            control_url_must_sit_under_source(where),
            control_unknown_kind_is_undeclared(where),
            control_empty_download_fails(where),
        ]


def main() -> int:
    controls = run_all()
    print("provisioner controls\n")
    for one in controls:
        print(f"{one.mark} {one.name}")
        print(f"       {one.detail[:150]}")

    failing = [one.name for one in controls if not one.held]
    print(f"\n{len(controls)} controls, {len(failing)} failing: {failing or 'none'}")

    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    target = generated / "provision_controls.json"
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            json.dumps({"controls": [vars(one) for one in controls], "failing": failing}, indent=2, sort_keys=True)
        )
        handle.write("\n")
    print(f"wrote {target}")
    return 0 if not failing else 1


if __name__ == "__main__":
    raise SystemExit(main())
