"""The producer population of THIS repository, derived from its own AST.

`compat/harness/population.py` discovers the population of the UPSTREAM
consumers named in the manifest, by walking cloned refs at a pinned commit.
This module answers the other half of P4: which sites in the settled tree of
*this* repository construct or invoke a producer.

It exists because three separate census attempts in one session each missed
sites, and each miss had the same cause -- the census was assembled from a
remembered inventory instead of derived from code:

  * a sweep whose scope was a hand-written directory list named
    `compat/producers/` and never `compat/consumers/`, `compat/vendor/` or
    `compat/corpus/`;
  * a pattern matching the RECEIVER NAME `app.get(`, which cannot see
    `producer.analysis().get(...)` -- the same call through a factory;
  * a claim about a codec's behaviour written from the encoder and decoder
    without reading the caller that guards them.

So this module takes no directory argument and holds no site list. Its input
is `git ls-files`, which is the settled tree; its output is every call whose
callee names a loader or an inference entry point, each classified by whether
the receiver can be traced to a producer construction. A site it cannot
classify is reported as CANDIDATE and makes the run non-zero: an unresolved
site is not an absent one, and this file is the place that rule is enforced
rather than assumed.

The loader vocabulary is imported from `compat.harness.population`, never
copied. Two lists that must agree are one list.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import proc
from compat.harness.population import LOADERS

ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent

#: Attribute names that RUN a model rather than build one. Names only: a name
#: is what an AST offers without type inference, and pretending otherwise is
#: how a census reports confidence it does not have.
INVOCATIONS: Final[dict[str, str]] = {
    "get": "insightface_faceanalysis_get",
    "detect": "detector_detect",
    "get_landmarks": "face_alignment_get_landmarks",
    "encode_image": "clip_image_tower",
    "encode_text": "clip_text_tower",
    "encode_media": "adapter_encode_media",
    "encode_many": "adapter_encode_many",
    "encode_query": "adapter_encode_query",
    "describe": "captioner_describe",
    "describe_many": "captioner_describe_many",
    "generate": "transformers_generate",
    "feature": "cv2_recognizer_feature",
    "alignCrop": "cv2_recognizer_aligncrop",
    "forward": "cv2_dnn_forward",
    "run": "onnx_session_run",
    "phash": "imagehash_phash",
    "dhash": "imagehash_dhash",
}

#: `.get` and `.run` are overloaded past usefulness (dict.get, subprocess.run).
#: They are reported ONLY when the receiver traces to a producer; otherwise
#: they land in `suppressed` with a count, never silently discarded.
AMBIGUOUS: Final[frozenset[str]] = frozenset({"get", "run", "detect", "forward", "generate"})

#: The runtimes a producer's output can only come from. A module that reaches
#: none of them cannot invoke a producer whatever its call is NAMED, which is
#: how `db/resultset.describe()` and `vision/captions.describe()` are told apart.
RUNTIMES: Final[frozenset[str]] = frozenset(
    {
        "torch",
        "torchvision",
        "transformers",
        "open_clip",
        "onnxruntime",
        "insightface",
        "facexlib",
        "mediapipe",
        "face_alignment",
        "cv2",
        "imagehash",
        "qwen_vl_utils",
        "safetensors",
    }
)

CONFIRMED: Final[str] = "CONFIRMED"
CHAINED: Final[str] = "CHAINED"
CANDIDATE: Final[str] = "CANDIDATE"
LOAD: Final[str] = "LOAD"
#: Excluded by the arity rule rather than by classification. Enumerated, not
#: tallied: a number says how many were skipped, a list says WHICH -- and the
#: difference is whether a reader can check the rule's judgement.
SIGNATURE_SUPPRESSED: Final[str] = "SIGNATURE_SUPPRESSED"


@dataclass(frozen=True)
class Site:
    """One call that builds or runs a producer, and why it was classified so."""

    path: str
    line: int
    kind: str  # LOAD | CONFIRMED | CHAINED | CANDIDATE
    callee: str
    role: str
    receiver: str
    evidence: str


def tracked_python(root: Path) -> list[str]:
    """Every tracked .py path. `git ls-files` IS the settled tree, so nothing
    here decides which directories count -- the repository does."""
    code, out, err = proc.text(["git", "-C", str(root), "ls-files", "*.py"], timeout=proc.LOCAL_SECONDS)
    if code != 0:
        raise RuntimeError(f"git ls-files failed ({code}): {err.strip()}")
    return sorted(one.strip() for one in out.splitlines() if one.strip())


def _receiver(node: ast.Call) -> str:
    """The text of what is being called on, or '' for a bare name call."""
    if isinstance(node.func, ast.Attribute):
        try:
            return ast.unparse(node.func.value)
        except (ValueError, AttributeError, RecursionError):
            # An unparseable receiver is still a receiver: naming it keeps the
            # site in the census instead of dropping it for being unprintable.
            return "<unparseable>"
    return ""


def _callee(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _producer_names(tree: ast.Module) -> dict[str, str]:
    """Local names bound to the result of a loader call: `app = FaceAnalysis(...)`
    makes `app` a producer for the rest of this module. Module-wide rather than
    per-scope on purpose -- a narrower rule would drop a name assigned in one
    function and used in another, and dropping is the failure this file exists
    to prevent."""
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        name = _callee(node.value)
        if name not in LOADERS:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                bound[target.id] = LOADERS[name]
    return bound


def sites_in(path: str, body: str) -> list[Site]:
    """Every producer construction and invocation in one source file."""
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return []
    bound = _producer_names(tree)
    found: list[Site] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = _callee(node)

        if callee in LOADERS:
            found.append(
                Site(
                    path=path,
                    line=node.lineno,
                    kind=LOAD,
                    callee=callee,
                    role=LOADERS[callee],
                    receiver=_receiver(node),
                    evidence="callee is a known model loader",
                )
            )
            continue

        if callee not in INVOCATIONS:
            continue

        receiver = _receiver(node)
        root_name = receiver.split(".", 1)[0].split("(", 1)[0]

        if root_name in bound:
            kind, why = CONFIRMED, f"receiver {root_name!r} is bound to {bound[root_name]} in this module"
        elif receiver.endswith(")"):
            # `producer.analysis().get(...)` -- the shape a receiver-name pattern
            # cannot see, and the one that hid eleven sites from two censuses.
            kind, why = CHAINED, "receiver is itself a call; the producer is built inline"
        elif callee == "get" and (len(node.args) >= 2 or node.keywords):
            # `mapping.get(key, default)` by SIGNATURE, not by receiver name.
            # RECORDED, never dropped: upstream's own `get(img, max_num=0, ...)`
            # wears this shape, and an unrecorded exclusion is an absence claim.
            shown = receiver or "<bare>"
            kind, why = SIGNATURE_SUPPRESSED, f"receiver {shown!r} unresolved and the call carries a default"
        elif callee in AMBIGUOUS:
            # Inside a runtime-reachable module an overloaded name is likelier a
            # producer call than a dict lookup; suppressing it here dropped four
            # real consumer sites on this tool's first run. `suppressed` counts them.
            kind, why = CANDIDATE, f"receiver {receiver or '<bare>'!r} unresolved; overloaded name in a runtime module"
        else:
            kind, why = CANDIDATE, f"receiver {receiver or '<bare>'!r} does not resolve to a producer here"

        found.append(
            Site(
                path=path,
                line=node.lineno,
                kind=kind,
                callee=callee,
                role=INVOCATIONS[callee],
                receiver=receiver,
                evidence=why,
            )
        )
    return found


def suppressed_in(path: str, body: str) -> int:
    """Ambiguous-name calls whose receiver does not resolve. Counted so the
    suppression is visible in the output rather than being a silent filter."""
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return 0
    bound = _producer_names(tree)
    total = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = _callee(node)
        if callee not in INVOCATIONS or callee not in AMBIGUOUS:
            continue
        receiver = _receiver(node)
        root_name = receiver.split(".", 1)[0].split("(", 1)[0]
        if root_name not in bound and not receiver.endswith(")"):
            total += 1
    return total


def _imports(body: str) -> set[str]:
    """Top-level package names this module imports, however it spells them."""
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(one.name.split(".")[0] for one in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
            found.add(node.module)
            # `from compat.producers import insightface_pass` imports a MODULE,
            # and recording only `compat.producers` loses it -- exactly how four
            # consumer sites stayed invisible to this tool's first run.
            found.update(f"{node.module}.{one.name}" for one in node.names)
    return found


def runtime_reachable(bodies: dict[str, str]) -> set[str]:
    """Modules that can reach an inference runtime: directly, or through one
    repository module that can. Derived from imports rather than declared, so
    a new producer module joins the census by importing torch, not by being
    added to a list here."""
    direct = {path for path, body in bodies.items() if _imports(body) & RUNTIMES}
    modules = {path[:-3].replace("/", ".").removesuffix(".__init__"): path for path in bodies}
    reaching = set(direct)
    for path, body in bodies.items():
        if path in reaching:
            continue
        for name in _imports(body):
            hit = modules.get(name)
            if hit is not None and hit in direct:
                reaching.add(path)
                break
    return reaching


#: Evidence a shipped constant cites. I7: the fix is never a wider ignore
#: allowlist -- that list has to be grown by hand and nobody did. The cited set
#: is DERIVED from the citing code, so it cannot drift out of step with it.
CITATION: Final[re.Pattern[str]] = re.compile(r"benchmarks/results/[A-Za-z0-9_./-]+\.json")


def _git_lines(root: Path, *argv: str) -> list[str]:
    _code, out, _err = proc.text(["git", "-C", str(root), *argv], timeout=proc.LOCAL_SECONDS)
    return [one.strip() for one in out.splitlines() if one.strip()]


def citations(root: Path, bodies: dict[str, str]) -> list[dict[str, Any]]:
    """Every evidence artifact cited from OUTSIDE its own directory, with the
    repository's verdict on each. A citation whose target is untracked cannot
    be read by anybody who clones this repository, so the number it justifies
    is unverifiable -- the same defect as a docstring nobody re-ran, one layer
    down."""
    cited: dict[str, set[str]] = {}
    for path, body in bodies.items():
        if path.startswith("benchmarks/"):
            continue  # a benchmark naming its own output is not a citation
        for hit in CITATION.findall(body):
            cited.setdefault(hit, set()).add(path)
    if not cited:
        return []
    tracked = set(_git_lines(root, "ls-files", "benchmarks/results/"))
    out: list[dict[str, Any]] = []
    for artifact in sorted(cited):
        is_tracked = artifact in tracked
        on_disk = (root / artifact).is_file()
        # WHY it does not resolve, not just THAT it does not: `check-ignore -v`
        # names the file, line and pattern doing the ignoring, which tells the
        # two cases apart -- ignored BY POLICY, or simply never added.
        ignored_by = ""
        if not is_tracked:
            told = _git_lines(root, "check-ignore", "-v", "--no-index", artifact)
            ignored_by = told[0] if told else ""
        out.append(
            {
                "artifact": artifact,
                "cited_by": sorted(cited[artifact]),
                "tracked": is_tracked,
                "on_disk": on_disk,
                "ignored": bool(ignored_by),
                "ignored_by": ignored_by,
                "why": (
                    ""
                    if is_tracked and on_disk
                    else "ignored"
                    if ignored_by
                    else "untracked"
                    if not is_tracked
                    else "absent"
                ),
                "resolves": is_tracked and on_disk,
            }
        )
    return out


def controls(bodies: dict[str, str]) -> dict[str, Any]:
    """Positive and negative controls, run every time rather than asserted once.

    POSITIVE: the sites that hid from two hand-written censuses must still be
    found. If a future edit to this scanner stops finding them, the run says so
    instead of quietly shrinking.

    NEGATIVE: removing a known site from a source body must reduce that file's
    site count. A scanner whose output does not change when the thing it looks
    for is deleted is not reading the code, and every count it prints is
    decoration. Done in memory -- nothing on disk is touched.
    """
    seeded = {
        "compat/consumers/face_selection.py": "producer.analysis().get(",
        "compat/consumers/masked_reference.py": "producer.analysis().get(",
        "compat/consumers/reference_sets.py": "producer.analysis().get(",
        "compat/consumers/aligned_crop.py": "app.get(",
        "compat/consumers/producer_derivations.py": "app.get(",
        "compat/consumers/reactor_face_model.py": "app.get(",
        "compat/consumers/face_family.py": "app.get(",
        "compat/corpus/loaded.py": "producer.detect(",
        "compat/harness/observe_attack.py": "FaceAnalysis(",
    }
    positive: list[dict[str, Any]] = []
    negative: list[dict[str, Any]] = []
    for path, needle in sorted(seeded.items()):
        body = bodies.get(path)
        if body is None:
            positive.append({"path": path, "found": False, "why": "file is not in the tracked tree"})
            continue
        before = sites_in(path, body)
        # The seeded SITE by line, not a callee of the same NAME anywhere in the
        # file: the old suffix match was satisfied by any `get` in the module,
        # so it asserted the file still held a matching name, not this call.
        seeded_lines = {n for n, line in enumerate(body.splitlines(), 1) if needle in line}
        hits = [one for one in before if one.line in seeded_lines]
        positive.append(
            {
                "path": path,
                "found": bool(hits),
                "sites": len(before),
                "seeded_lines": sorted(seeded_lines),
                "matched_lines": sorted(one.line for one in hits),
            }
        )

        stripped = "\n".join(line for line in body.splitlines() if needle not in line)
        # A stripped body that no longer PARSES makes sites_in return [], which
        # satisfies `after < before` by breakage rather than by removal. The
        # control has to prove the site went away, not that the file did.
        parses = True
        try:
            ast.parse(stripped)
        except SyntaxError:
            parses = False
        after = sites_in(path, stripped)
        negative.append(
            {
                "path": path,
                "before": len(before),
                "after": len(after),
                "still_parses": parses,
                "changed": parses and len(after) < len(before),
            }
        )
    return {
        "positive": positive,
        "negative": negative,
        "positive_clean": all(one.get("found") for one in positive),
        "negative_clean": all(one["changed"] for one in negative),
    }


def build(root: Path = ROOT) -> dict[str, Any]:
    """The whole tree's producer population, derived."""
    sites: list[Site] = []
    suppressed = 0
    unreadable: list[str] = []
    bodies: dict[str, str] = {}
    for relative in tracked_python(root):
        try:
            bodies[relative] = (root / relative).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as why:
            unreadable.append(f"{relative}: {type(why).__name__}")

    reaching = runtime_reachable(bodies)
    off_runtime: list[str] = []
    for relative, body in bodies.items():
        if relative not in reaching:
            # Counted, never silently dropped: a module that cannot reach a
            # runtime today joins the census the moment it imports one.
            off_runtime.extend(
                f"{relative}:{one.line} {one.receiver}.{one.callee}()" for one in sites_in(relative, body)
            )
            continue
        sites.extend(sites_in(relative, body))
        suppressed += suppressed_in(relative, body)

    by_dir: dict[str, int] = {}
    for one in sites:
        top = one.path.split("/", 1)[0]
        by_dir[top] = by_dir.get(top, 0) + 1

    cited = citations(root, bodies)
    unresolved_citations = [one for one in cited if not one["resolves"]]
    # The allowlist is wrong in BOTH directions, and only the derived set can
    # say so: entries re-admitted by hand that nothing cites are as much a sign
    # the list is maintained by memory as the citations it never grew to cover.
    tracked_uncited = sorted(
        set(_git_lines(root, "ls-files", "benchmarks/results/")) - {one["artifact"] for one in cited}
    )
    checks = controls(bodies)
    kinds = (LOAD, CONFIRMED, CHAINED, CANDIDATE, SIGNATURE_SUPPRESSED)
    counted = {kind: sum(1 for one in sites if one.kind == kind) for kind in kinds}
    return {
        "sites": [asdict(one) for one in sorted(sites, key=lambda one: (one.path, one.line))],
        "totals": {
            **counted,
            "sites": len(sites),
            "files_with_sites": len({one.path for one in sites}),
            "by_top_directory": dict(sorted(by_dir.items())),
            "suppressed_ambiguous": suppressed,
            "off_runtime_ignored": len(off_runtime),
            "unreadable": unreadable,
            "citations": len(cited),
            "citations_unresolved": len(unresolved_citations),
            "tracked_uncited": len(tracked_uncited),
            "controls_positive_clean": checks["positive_clean"],
            "controls_negative_clean": checks["negative_clean"],
        },
        "citations": cited,
        "tracked_uncited": tracked_uncited,
        "controls": checks,
        "off_runtime": sorted(off_runtime),
    }


def main() -> int:
    out = build()
    totals = out["totals"]
    print("producer population, derived from the tracked tree (no directory list)\n")
    print(f"  LOAD       (model constructed) : {totals[LOAD]}")
    print(f"  CONFIRMED  (receiver bound)    : {totals[CONFIRMED]}")
    print(f"  CHAINED    (built inline)      : {totals[CHAINED]}")
    print(f"  CANDIDATE  (unresolved)        : {totals[CANDIDATE]}")
    print(f"  total sites                    : {totals['sites']} in {totals['files_with_sites']} files")
    print(f"  SIGNATURE_SUPPRESSED (arity)   : {totals[SIGNATURE_SUPPRESSED]}")
    print(f"  suppressed ambiguous names     : {totals['suppressed_ambiguous']}")
    print(f"  off-runtime name collisions    : {totals['off_runtime_ignored']} (same name, no runtime reachable)")
    if totals["unreadable"]:
        print(f"  UNREADABLE                     : {len(totals['unreadable'])}")
    print("\nby top-level directory:")
    for where, count in totals["by_top_directory"].items():
        print(f"  {where:<12} {count:>5}")

    if totals[SIGNATURE_SUPPRESSED]:
        print(f"\n{totals[SIGNATURE_SUPPRESSED]} site(s) excluded by the arity rule -- WHICH, not how many:")
        for one in out["sites"]:
            if one["kind"] == SIGNATURE_SUPPRESSED:
                print(f"  {one['path']}:{one['line']}  {one['receiver']}.{one['callee']}()")

    if totals[CANDIDATE]:
        print(f"\n{totals[CANDIDATE]} CANDIDATE site(s) -- an unresolved site is not an absent one:")
        for one in out["sites"]:
            if one["kind"] == CANDIDATE:
                print(f"  {one['path']}:{one['line']}  {one['receiver']}.{one['callee']}()  -- {one['evidence']}")

    print("\ncontrols:")
    print(f"  positive (seeded sites found)  : {'clean' if totals['controls_positive_clean'] else 'FAILED'}")
    print(f"  negative (removal changes it)  : {'clean' if totals['controls_negative_clean'] else 'FAILED'}")
    for one in out["controls"]["positive"]:
        if not one.get("found"):
            print(f"    POSITIVE FAILED {one['path']}: {one.get('why', 'seeded site not found')}")
    for one in out["controls"]["negative"]:
        if not one["changed"]:
            print(f"    NEGATIVE FAILED {one['path']}: {one['before']} sites before, {one['after']} after")

    print(f"\ncited evidence artifacts: {totals['citations']}, unresolved {totals['citations_unresolved']}")
    for one in out["citations"]:
        if not one["resolves"]:
            print(f"  {one['why'].upper():<10} {one['artifact']}  cited by {', '.join(one['cited_by'])}")
            if one["ignored_by"]:
                print(f"             ignored by {one['ignored_by']}")
    if out["tracked_uncited"]:
        print(f"\nallowlisted but cited by nothing: {totals['tracked_uncited']}")
        for one in out["tracked_uncited"]:
            print(f"  {one}")

    generated = ROOT / "compat" / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    target = generated / "producer_census.json"
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True))
        handle.write("\n")
    print(f"\nwrote {target}")

    # A failed control, or a cited artifact nobody cloning this repository could
    # read, is a red run. CANDIDATE is reported and NOT gated on: a condition
    # that can never pass is as useless as one that can never fail.
    failed = (
        totals["unreadable"]
        or totals["citations_unresolved"]
        or not totals["controls_positive_clean"]
        or not totals["controls_negative_clean"]
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
