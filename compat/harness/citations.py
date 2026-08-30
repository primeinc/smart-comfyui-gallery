"""Every `path:line` in the manifest, resolved against the pinned commit.

Eleven wrong values were found in this manifest by hand in one session --
four recognition packs, two combine rules, two det-size ladders, a max side
length, a preprocessing note and a citation off by one. Every one of them was
written from a careful read of the source that later contradicted it, which
is the argument for checking citations by machine rather than by reading them
again more carefully.

WHAT THIS CAN AND CANNOT PROVE
------------------------------
It resolves the FILE at the pinned commit and bounds the LINE, and it looks
for the identifier the citation names inside the lines it points at. So it
catches a dead path, a line past the end of the file, and a citation that
drifted off its symbol.

It cannot prove a citation SUPPORTS the claim beside it. `pack = "antelopev2"`
cited to a line that really does construct a FaceAnalysis is well-formed and
was still wrong. That remains a reading problem; this only removes the
mechanical half of it.

A citation with no identifier to anchor on is reported as UNANCHORED rather
than passed: it is unknown, not fine.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import proc
from compat.harness import provenance

ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: `path/to/file.py:12` or `:12-34`, at the head of a citation string.
WHERE: Final[re.Pattern[str]] = re.compile(r"^([\w./-]+\.(?:py|ipynb|md|toml|json|txt)):(\d+)(?:-(\d+))?\s*(.*)$")

#: Tokens worth anchoring on: dotted paths, snake_case, UpperCamelCase and
#: lowerCamelCase. Bare English words are not identifiers and would match
#: anything, so they are deliberately not accepted.
ANCHOR: Final[re.Pattern[str]] = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+"
    r"|[a-z]+_[a-z0-9_]+"
    r"|[A-Z][a-z]+[A-Z][A-Za-z0-9]*"
    r"|[a-z]+[A-Z][A-Za-z0-9]*"
    r"|[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+"
)

#: An explicit escape hatch for prose. A README paragraph has no identifier
#: to anchor on, so a citation may quote the text it means and the quoted
#: string is then required to appear in the cited lines verbatim.
QUOTED: Final[re.Pattern[str]] = re.compile(r'"([^"]{4,})"')

#: Words that look like identifiers but carry no locating power.
NOISE: Final[frozenset[str]] = frozenset(
    {"per_reference", "not_exercised", "line", "lines", "and", "the", "then", "value", "no_", "int"}
)


@dataclass
class Citation:
    """One `path:line` and what resolving it against the pin showed."""

    consumer_id: str
    field: str
    text: str
    path: str = ""
    first: int = 0
    last: int = 0
    verdict: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict == "OK"


def _blob(clone: Path, commit: str, path: str) -> bytes | None:
    code, out, _ = proc.run(
        ["git", "-C", str(clone), "cat-file", "blob", f"{commit}:{path}"], timeout=proc.LOCAL_SECONDS
    )
    return out if code == 0 else None


def anchors(description: str) -> list[str]:
    """The identifiers a citation names, if any.

    A dotted form is split as well as kept: `IPAdapterFaceID.get_image_embeds`
    never appears in the file that DEFINES it, because the definition reads
    `def get_image_embeds`. Both halves are candidates.
    """
    found = [one for one in ANCHOR.findall(description) if one not in NOISE]
    out: set[str] = set()
    for one in found:
        out.add(one.rsplit(".", 1)[-1] if one.endswith(".py") else one)
        if "." in one and not one.endswith(".py"):
            out.update(part for part in one.split(".") if part)
    return sorted(out - NOISE)


def encloses(lines: list[str], name: str, first: int, last: int) -> int:
    """The line defining `name`, when the cited range sits inside its body.

    Citing a range INSIDE a function and naming that function is legitimate
    and common -- `IPAdapterPlus.py:350-367 ipadapter_execute face branch`
    points at the face branch and names the function containing it. Without
    this the checker would report every such citation as a mismatch and the
    real off-by-ones would be lost in the noise.

    Returns the definition's line number, or 0.
    """
    opener = re.compile(rf"^(\s*)(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b")
    for index, line in enumerate(lines):
        matched = opener.match(line)
        if matched is None:
            continue
        start = index + 1
        indent = len(matched.group(1))
        end = len(lines)
        for after in range(index + 1, len(lines)):
            body = lines[after]
            if not body.strip():
                continue
            if len(body) - len(body.lstrip()) <= indent:
                end = after
                break
        if start <= first and last <= end:
            return start
    return 0


def check(consumer_id: str, field: str, text: str, clone: Path, commit: str) -> Citation:
    held = Citation(consumer_id=consumer_id, field=field, text=text)
    matched = WHERE.match(text.strip())
    if matched is None:
        held.verdict = "UNPARSED"
        held.detail = "no path:line at the head of the citation"
        return held

    held.path = matched.group(1)
    held.first = int(matched.group(2))
    held.last = int(matched.group(3) or matched.group(2))
    description = matched.group(4)

    blob = _blob(clone, commit, held.path)
    if blob is None:
        held.verdict = "FILE_MISSING"
        held.detail = f"{held.path} is not at {commit[:12]}"
        return held

    lines = blob.decode("utf-8", errors="surrogateescape").splitlines()
    if held.last > len(lines):
        held.verdict = "RANGE_OUT"
        held.detail = f"cites line {held.last}; the file has {len(lines)}"
        return held

    window = "\n".join(lines[held.first - 1 : held.last])

    # A quoted string is an explicit claim about the cited text and outranks
    # any identifier guessed out of the prose around it.
    quoted = QUOTED.findall(description)
    if quoted:
        missing = [one for one in quoted if one not in window]
        held.verdict = "OK" if not missing else "MISMATCH"
        held.detail = f"quoted {quoted[0]!r}" if not missing else f"quoted text absent: {missing}"
        return held

    wanted = anchors(description)
    if not wanted:
        held.verdict = "UNANCHORED"
        held.detail = f"nothing identifier-shaped in {description!r}; quote the text to anchor on it"
        return held
    hit = [one for one in wanted if one in window]
    if hit:
        held.verdict = "OK"
        held.detail = f"anchored on {hit[0]}"
        return held

    for one in wanted:
        at = encloses(lines, one, held.first, held.last)
        if at:
            held.verdict = "OK"
            held.detail = f"inside {one}, defined at line {at}"
            return held

    # Where does it actually live? A near miss is the common case, and the
    # distance separates an off-by-one from a guess.
    near = ""
    for one in wanted:
        at = [i + 1 for i, line in enumerate(lines) if one in line]
        if at:
            closest = min(at, key=lambda n: min(abs(n - held.first), abs(n - held.last)))
            near = f"; {one} is at line {closest}"
            break
    held.verdict = "MISMATCH"
    held.detail = f"none of {wanted} in lines {held.first}-{held.last}{near}"
    return held


def check_entrypoint(consumer_id: str, text: str, clone: Path, commit: str) -> Citation:
    """`path::symbol` -- the file at the pin, and the symbol defined in it.

    An entrypoint is the strongest claim a row makes: it names the code the
    whole row describes. It was the one field nothing checked.
    """
    held = Citation(consumer_id=consumer_id, field="entrypoint", text=text)
    path, _, symbol = text.partition("::")
    held.path = path
    if not symbol:
        held.verdict = "UNPARSED"
        held.detail = "no `::symbol` suffix"
        return held

    blob = _blob(clone, commit, path)
    if blob is None:
        held.verdict = "FILE_MISSING"
        held.detail = f"{path} is not at {commit[:12]}"
        return held

    # A dunder entry is the module's own top level, not a definition.
    if symbol.startswith("__"):
        held.verdict = "OK"
        held.detail = f"{path} present; module-level entry"
        return held

    text_of = blob.decode("utf-8", errors="surrogateescape")
    lines = text_of.splitlines()
    leaf = symbol.rsplit(".", 1)[-1]

    # A shell script has no `def`, so the name is required as literal text.
    # NOT fuzzy: a case-folded match would let `quickstart` pass against any
    # "Quick Start" heading.
    if not path.endswith(".py"):
        if leaf in text_of:
            held.verdict = "OK"
            held.detail = f"{leaf} present as literal text in {path}"
            return held
        held.verdict = "MISMATCH"
        held.detail = f"{leaf} does not appear literally in {path} at {commit[:12]}"
        return held

    opener = re.compile(rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(leaf)}\b")
    at = next((i + 1 for i, line in enumerate(lines) if opener.match(line)), 0)
    if at:
        held.first = held.last = at
        held.verdict = "OK"
        held.detail = f"{leaf} defined at line {at}"
        return held
    held.verdict = "MISMATCH"
    held.detail = f"{leaf} is not defined in {path} at {commit[:12]}"
    return held


def survey() -> dict[str, Any]:
    manifest = provenance.load_manifest()
    refs_root = (ROOT.parent / manifest["refs_root"]).resolve()
    rows: list[Citation] = []

    upstreams = manifest.get("upstreams", {})
    for consumer in manifest.get("consumers", []):
        # A consumer whose entrypoint lives in another upstream declares
        # `entrypoint_in`, and its paths resolve against THAT repository at
        # THAT pin rather than against its own clone.
        host = consumer.get("entrypoint_in")
        source = upstreams[host] if host else consumer
        clone = provenance.clone_dir(refs_root, source["repo"])
        commit = source["commit"]
        setup = consumer.get("vendor_setup") or {}
        entry = consumer.get("entrypoint")
        if entry:
            rows.append(check_entrypoint(consumer["id"], entry, clone, commit))
        # `acceptance_expected.cited` is what the vendor acceptance verdict
        # is measured against, so it resolves with the other two.
        expected = consumer.get("acceptance_expected") or {}
        for field, cited in (
            ("consumer", consumer.get("cited", [])),
            ("vendor_setup", setup.get("cited", [])),
            ("acceptance_expected", expected.get("cited", [])),
        ):
            rows.extend(check(consumer["id"], field, one, clone, commit) for one in cited)

    counts: dict[str, int] = {}
    for one in rows:
        counts[one.verdict] = counts.get(one.verdict, 0) + 1
    return {
        "citations": [asdict(one) for one in rows],
        "counts": counts,
        "clean": all(one.ok for one in rows),
    }


def main() -> int:
    out = survey()
    for row in out["citations"]:
        if row["verdict"] == "OK":
            continue
        print(f"!! {row['consumer_id']:<22} {row['verdict']:<13} {row['text'][:70]}")
        print(f"   {row['detail']}")

    print(f"\ncitations: {out['counts']}")

    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    target = generated / "citations.json"
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True, default=str))
        handle.write("\n")
    print(f"wrote {target}")
    return 0 if out["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
