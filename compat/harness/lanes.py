from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final

from compat.harness import identity as evidence_identity

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
GENERATED: Final[Path] = ROOT / "generated"

RAW: Final[str] = "lanes.raw"
STAMPED: Final[str] = "lanes.json"


def declared(repo: Path = ROOT.parent) -> tuple[str, ...]:
    # Parsed from compat.just's run recipe, never duplicated here. Judged against
    # its own keys, a lane DELETED from the recipe simply never appears, and the
    # gate passes over what remains -- the record cannot be its own population.
    found = re.search(r"for lane in ([^;]+); do", (repo / "compat.just").read_text(encoding="utf-8"))
    return tuple(found.group(1).split()) if found else ()


def read(where: Path = GENERATED) -> dict[str, Any] | None:
    path = where / STAMPED
    if not path.is_file():
        return None
    held: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return held


def exits(where: Path = GENERATED) -> dict[str, int]:
    found = (read(where) or {}).get("lanes")
    if not isinstance(found, dict):
        return {}
    return {str(name): int(code) for name, code in found.items()}


def _recorded(raw: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in raw.read_text(encoding="utf-8").splitlines():
        name, _, code = line.strip().partition(" ")
        if name:
            out[name] = int(code or 0)
    return out


def main() -> int:
    # This module RECORDS; compat.harness.closure JUDGES. An unrecordable run must
    # still produce a ledger, so a missing or empty lane set is a finding for the
    # gate rather than a reason to abandon the run.
    raw = GENERATED / RAW
    lanes = _recorded(raw) if raw.is_file() else {}

    GENERATED.mkdir(parents=True, exist_ok=True)
    body = {"identity": str(evidence_identity.identity()["digest"]), "lanes": lanes}
    target = GENERATED / STAMPED
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(body, indent=2, sort_keys=True))
        handle.write("\n")

    # A raw file left behind would be read by the NEXT run, which is how a lane
    # record outlives the run it describes.
    raw.unlink(missing_ok=True)

    red = sorted(name for name, code in lanes.items() if code != 0)
    print(f"wrote {target}")
    print(f"lanes recorded: {len(lanes)}   red: {len(red)}{'  ' + ', '.join(red) if red else ''}")
    if not lanes:
        print(f"NO LANE WAS RECORDED: {raw} was absent or empty; closure fails on the empty set")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
