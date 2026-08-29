"""Check the pins before anything claims to have proved a replay.

A commit hash in a manifest is an assertion until something executes against
it. This resolves every consumer and upstream in `manifest.toml` against the
clone on disk and answers four questions per row:

    is the clone at the commit the manifest names?
    is its working tree clean, so that commit describes what is actually there?
    does every declared path exist AT THAT COMMIT -- `git show <sha>:<path>`,
        not at HEAD, not on disk?
    what are the bytes of each declared path at that commit?

The last one is load-bearing. A path that exists at HEAD says nothing about
the commit the evidence claims, and a reviewer recomputing
`git show <sha>:<path>` has to land on the same hash recorded here or the
evidence is wrong.

Refs are read-only mirrors: nothing here fetches, checks out, or writes to
them. A harness that repairs its own inputs cannot report on them.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

HERE: Path = Path(__file__).resolve().parent
ROOT: Path = HERE.parent
MANIFEST: Path = ROOT / "manifest.toml"

#: Seconds any single git call may take. Every subprocess in this tree carries
#: one: a call that can hang forever turns a red gate into a stalled run, and
#: an operator waiting on nothing cannot tell those apart.
GIT_SECONDS: float = 60.0


@dataclass
class PathProof:
    """One declared path, as it exists at the pinned commit.

    `symbol_present` is the part that makes this a contract rather than a
    locator. A path that exists proves only that somebody typed a filename;
    the claim being recorded is that a NAMED symbol is defined in those bytes,
    so the symbol is resolved out of the blob's AST.

    `locator_only` marks a row that cannot be checked that way -- a markdown
    quickstart, or a script whose contract is its `__main__` body. Those are
    honest locators and are reported as such; they are never counted as a
    verified symbol, because a harness that grades its own weakest rows as
    passes is the failure it exists to prevent.
    """

    path: str
    symbol: str
    present: bool
    blob_sha256: str | None = None
    symbol_present: bool | None = None
    locator_only: bool = False
    note: str | None = None


@dataclass
class RepoProof:
    """One upstream repository, checked against its pin."""

    key: str
    repo: str
    pinned_commit: str
    clone: str
    exists: bool
    head_commit: str | None = None
    at_pin: bool = False
    clean: bool | None = None
    dirty_paths: int = 0
    paths: list[PathProof] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def _git(where: Path | None, *args: str) -> tuple[int, str]:
    """One git call, with stderr kept rather than dropped.

    stdout and stderr are joined because a probe whose failure reports nothing
    reads as an empty result, and that is how a broken check becomes a passing
    one. `check=False` because a non-zero exit is data here, not an exception.
    """
    argv: list[str] = ["git"]
    if where is not None:
        argv += ["-C", str(where)]
    argv += list(args)
    try:
        done = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=GIT_SECONDS)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {GIT_SECONDS}s: {' '.join(argv)}"
    return done.returncode, (done.stdout + done.stderr).strip()


def _git_bytes(where: Path, *args: str) -> tuple[int, bytes]:
    """One git call whose output is bytes, untouched.

    Separate from `_git` on purpose. That one decodes with universal newlines
    and STRIPS -- fine for a status line, wrong for a blob: stripping removes
    the trailing newline that every real file ends with, so a byte comparison
    against an installed file would fail on every row and the check would read
    as "nothing matches its pin" rather than as its own bug.
    """
    argv: list[str] = ["git", "-C", str(where), *args]
    try:
        done = subprocess.run(argv, capture_output=True, check=False, timeout=GIT_SECONDS)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {GIT_SECONDS}s: {' '.join(argv)}".encode()
    return done.returncode, (done.stdout if done.returncode == 0 else done.stderr)


def defines_symbol(source: str, symbol: str) -> bool:
    """Is `symbol` defined in this Python source, at any nesting depth?

    Parsed with `ast`, never matched textually. A substring search finds the
    name in a comment, a docstring, an import, or a call site, and would
    happily confirm a symbol the file does not define -- which is the exact
    class of false pass this whole harness exists to refuse.

    `A.b` walks: the class must be defined, and the method inside it.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False

    def defined_in(body: list[ast.stmt], names: list[str]) -> bool:
        head, rest = names[0], names[1:]
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == head:
                return not rest or defined_in(node.body, rest)
            # Module-level constants and assignments are contracts too: a
            # template array is as load-bearing as the function using it.
            if not rest and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(one, ast.Name) and one.id == head for one in targets):
                    return True
        return False

    return defined_in(tree.body, symbol.split("."))


def clone_dir(refs_root: Path, repo_url: str) -> Path:
    """`https://github.com/<org>/<name>.git` -> `<refs>/<org>/<name>`.

    The mirror is org-scoped, so a flat `<refs>/<name>` lookup finds nothing
    and would be reported as an absent repository rather than a bad guess.
    """
    tail: str = repo_url.removeprefix("https://github.com/").removesuffix(".git")
    org, _, name = tail.partition("/")
    return refs_root / org / name


def declared_paths(entry: dict[str, Any]) -> list[tuple[str, str]]:
    """Every `path::symbol` a row declares, entrypoint first.

    `read` and `also_read` carry the same spelling, so a reviewer sampling any
    claim in the manifest lands on a row this checked.
    """
    out: list[tuple[str, str]] = []
    spellings: list[Any] = [entry.get("entrypoint"), *entry.get("also_read", []), *entry.get("read", [])]
    for spelling in spellings:
        if not spelling:
            continue
        path, _, symbol = str(spelling).partition("::")
        if path:
            out.append((path, symbol))
    return out


def verify_repo(
    key: str,
    entry: dict[str, Any],
    refs_root: Path,
    *,
    paths_from: dict[str, Any] | None = None,
) -> RepoProof:
    """One repository against its pin. Never fetches; never checks out."""
    repo_url: str = entry["repo"]
    pinned: str = entry["commit"]
    where: Path = clone_dir(refs_root, repo_url)
    proof = RepoProof(key=key, repo=repo_url, pinned_commit=pinned, clone=str(where), exists=where.is_dir())

    if not proof.exists:
        proof.failures.append(f"clone absent at {where}")
        return proof

    code, head = _git(where, "rev-parse", "HEAD")
    if code != 0:
        proof.failures.append(f"rev-parse failed: {head}")
        return proof
    proof.head_commit = head
    proof.at_pin = head == pinned
    if not proof.at_pin:
        proof.failures.append(f"HEAD {head} is not the pinned {pinned}")

    code, dirty = _git(where, "status", "--porcelain")
    if code != 0:
        proof.failures.append(f"status failed: {dirty}")
    else:
        lines: list[str] = [one for one in dirty.splitlines() if one.strip()]
        proof.dirty_paths = len(lines)
        proof.clean = not lines
        if lines:
            proof.failures.append(f"{len(lines)} uncommitted path(s): the commit does not describe the tree")

    # Paths are read AT THE PINNED COMMIT, not from the working tree: a file
    # deleted upstream but still sitting on disk would otherwise verify.
    source: dict[str, Any] = paths_from if paths_from is not None else entry
    for path, symbol in declared_paths(source):
        code, blob = _git(where, "show", f"{pinned}:{path}")
        if code != 0:
            proof.paths.append(PathProof(path=path, symbol=symbol, present=False, note=blob[:200]))
            proof.failures.append(f"{path} absent at {pinned[:12]}")
            continue
        digest: str = hashlib.sha256(blob.encode("utf-8", errors="surrogateescape")).hexdigest()

        # Only Python carries a checkable symbol. A markdown quickstart or a
        # `__main__` body is a locator: recorded, hashed, and NEVER counted as
        # a verified contract.
        locator = not path.endswith(".py") or symbol in {"", "__main__", "quickstart", "preprocess"}
        found: bool | None = None
        if not locator:
            found = defines_symbol(blob, symbol)
            if not found:
                proof.failures.append(f"{path} exists at {pinned[:12]} but does not define {symbol}")

        proof.paths.append(
            PathProof(
                path=path,
                symbol=symbol,
                present=True,
                blob_sha256=digest,
                symbol_present=found,
                locator_only=locator,
            )
        )

    return proof


#: Packages whose version changes results and whose source CAN be hashed the
#: ordinary way, so they are recorded by version here and pinned by blob in
#: `[[runtimes]]` where a consumer actually executes them.
LIBRARIES: tuple[str, ...] = (
    "onnx",
    "numpy",
    "opencv-python",
    "opencv-contrib-python",
    "scikit-image",
    "torch",
    "insightface",
    "mediapipe",
    "face-alignment",
)

#: The modules whose arithmetic decides pixel values, by the version the
#: IMPORT resolves to rather than by a distribution name: two distributions can
#: supply one module, so metadata describes what was installed.
IMPORTED: tuple[str, ...] = ("cv2", "numpy", "skimage", "torch", "PIL")


def backend_identity() -> dict[str, Any]:
    """What computed the numbers, as opposed to what code was cited.

    onnxruntime is a BACKEND, not a library: the arithmetic lives in compiled
    kernels, so hashing a `.py` file certifies nothing about what produced an
    embedding. It reports its own identity instead, and that is the only claim
    that can honestly be made about it --

        get_version_string()      the binary's version
        get_build_info()          its git-commit-id and build type
        get_available_providers() which engines this machine offers
        get_device()              what the build was compiled to target

    documented at onnxruntime `docs/python/api_summary.rst`, "Providers" and
    "Build, Version". `get_build_info` is preferred over the wheel's metadata
    version deliberately: metadata describes the package that was installed,
    build info describes the binary that will run, and those can disagree.

    The distinction matters because a reviewer comparing a float recorded here
    against their own needs to know the engine, not the citation.
    """
    out: dict[str, Any] = {}
    for name in LIBRARIES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None

    # What the IMPORT resolves to, which is what actually runs.
    imported: dict[str, Any] = {}
    for name in IMPORTED:
        try:
            module = importlib.import_module(name)
        except ImportError as why:
            imported[name] = f"ABSENT: {why}"
            continue
        imported[name] = {
            "version": str(getattr(module, "__version__", "present")),
            # Where it loaded FROM: two distributions can supply one module
            # name, and the path is what says which won.
            "file": str(getattr(module, "__file__", "?")),
        }
    out["imported"] = imported

    engine: dict[str, Any] = {}
    try:
        import onnxruntime

        engine["version"] = onnxruntime.get_version_string()
        engine["build_info"] = onnxruntime.get_build_info()
        engine["available_providers"] = list(onnxruntime.get_available_providers())
        engine["device"] = onnxruntime.get_device()
    except (ImportError, OSError, AttributeError) as why:
        # A DLL that will not load must not read as "no providers": one is a
        # broken environment, the other a CPU box, and the difference is the
        # reason every downstream UNSUPPORTED case reports.
        engine["error"] = f"{type(why).__name__}: {why}"
    out["onnxruntime"] = engine
    return out


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def _weight_root(row: dict[str, Any]) -> Path:
    """Where a weight actually lives, resolved the way the run resolves it.

    A row may name `root` outright, or name `root_package` -- the importable
    package that owns the weights -- plus `root_subdir`.

    The second spelling exists because the first drifted. `facexlib`'s weights
    were declared under `.venv/...` while the suite runs from `.venv-compat`,
    so the manifest named a directory no interpreter in this project uses and
    the rows read ABSENT. A path typed into a manifest is a claim about an
    environment; asking the interpreter removes the gap between the two.
    """
    package = row.get("root_package")
    if package:
        found = importlib.util.find_spec(package)
        if found is None or not found.origin:
            return Path(row.get("root", f"<{package} not importable>"))
        return Path(found.origin).parent / row.get("root_subdir", "")
    return Path(row["root"])


#: Where the vendors' own checkpoints live. The same root
#: `compat/vendor/acceptance.py` loads them from, named here so provenance can
#: check the files that lane executes against.
VENDOR_ROOT: Final[Path] = ROOT.parent.parent / "sg-vendor-fixtures"


def vendor_weight_identity(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """The vendors' own checkpoints, by content, against what was recorded.

    30 rows carried a sha256 and a byte count and no code read any of them.
    The acceptance lane loads exactly these files and reports `ran = True`
    when the call did not raise, so a checkpoint swapped for another of the
    same shape would have produced a different boundary under an unchanged
    stamp.

    ABSENT is reported and is not a failure -- most of these are the reason a
    consumer is `not_attempted` on this machine. PRESENT with the wrong digest
    is a failure.
    """
    out: list[dict[str, Any]] = []
    for row in manifest.get("vendor_weights", []):
        where = VENDOR_ROOT / row["file"]
        found = where.is_file()
        measured = digest_file(where) if found else None
        size = where.stat().st_size if found else None
        recorded = str(row.get("sha256") or "")
        out.append(
            {
                "consumer": row.get("consumer", "?"),
                "file": row["file"],
                "source": row.get("source", ""),
                "present": found,
                "sha256": measured,
                "recorded_sha256": recorded,
                "bytes": size,
                "recorded_bytes": row.get("bytes"),
                # None when there is nothing to compare: the file is not here.
                "matches": None if not found else (measured == recorded and size == row.get("bytes")),
            }
        )
    return out


def weight_identity(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """The model files themselves, by content.

    The largest single determinant of every number in this evidence is not a
    line of Python: it is `glintr100.onnx`. A suite that pins twenty
    repositories to the commit and leaves the weights unnamed can reproduce
    its own code exactly, produce different embeddings on the next machine,
    and have nothing in the record able to say why.

    Absent files are reported as absent, never skipped: a pack that is not
    there is the reason a case is UNSUPPORTED, and that has to stay visible.
    """
    out: list[dict[str, Any]] = []
    for row in manifest.get("weights", []):
        where = _weight_root(row) / row["file"]
        found = where.is_file()
        measured = digest_file(where) if found else None
        # A hash WE computed proves our copy has not changed since we hashed
        # it, not that it is the one the vendor shipped. A vendor-published
        # digest is recorded and checked separately.
        published = row.get("published_sha256")
        out.append(
            {
                "pack": row["pack"],
                "file": row["file"],
                "path": str(where),
                "role": row.get("role", ""),
                "present": found,
                "bytes": where.stat().st_size if found else 0,
                "sha256": measured,
                "published_sha256": published,
                "published_by": row.get("published_by", ""),
                "matches_published": None if not (published and measured) else measured == published,
            }
        )
    return out


def runtime_identity() -> dict[str, Any]:
    """What ran this. Evidence from another machine is a different claim.

    `platform.platform()` carries the Windows BUILD number, so an OS update
    moves this string and every recorded run reads as taken elsewhere. That is
    deliberate and it is the coarse answer, not an oversight: the evidence
    here is byte-exactness of ONNX Runtime output, and which kernel that
    picks is a function of the driver and the OS underneath it. A digest that
    survived an OS update would be claiming those bytes are portable across a
    boundary nothing here has tested them across.
    """
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "backends": backend_identity(),
    }


def load_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


@dataclass
class RuntimeProof:
    """An INSTALLED package's bytes against the commit the manifest pins.

    Some consumers run from a wheel rather than from the clone -- facexlib's
    detector, ConsisID's entrypoint inside diffusers -- and a wheel is not
    automatically the commit it claims. Without this, the suite could cite a
    commit and execute something else, which is the most expensive kind of
    wrong evidence: it looks fully sourced.

    Line endings are normalised before hashing. A wheel built on one platform
    and a blob read through git on another can differ by nothing but CRLF,
    and reporting that as a divergence would train everyone to ignore the
    check.
    """

    package: str
    path: str
    pinned_commit: str
    installed_file: str | None = None
    installed_sha256: str | None = None
    pinned_sha256: str | None = None
    matches: bool = False
    note: str | None = None


def installed_module_root(package: str) -> Path | None:
    """Where an installed package's own directory lives, or None."""
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def verify_runtime(package: str, repo: Path, commit: str, path: str, installed_path: str) -> RuntimeProof:
    """One installed file against `git show <commit>:<path>`.

    `installed_path` is declared rather than derived. Repositories lay their
    package out under `src/`, under `python-package/`, or at the root, and a
    rule that guesses would silently look in the wrong place and report a
    missing file as a failed pin -- an absence dressed as a divergence.
    """
    proof = RuntimeProof(package=package, path=path, pinned_commit=commit)

    root = installed_module_root(package)
    if root is None:
        proof.note = f"{package} is not importable in this environment"
        return proof

    where = root.parent / installed_path
    proof.installed_file = str(where)
    if not where.is_file():
        proof.note = f"pinned path is not present in the installed package at {where}"
        return proof

    code, blob = _git_bytes(repo, "show", f"{commit}:{path}")
    if code != 0:
        proof.note = f"git show failed: {blob.decode('utf-8', 'replace')[:200]}"
        return proof

    # Line endings normalised on both sides. A wheel built on one platform and
    # a blob read on another can differ by nothing but CRLF, and reporting
    # that as a divergence would train everyone to ignore the check.
    proof.pinned_sha256 = hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()
    proof.installed_sha256 = hashlib.sha256(where.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    proof.matches = proof.installed_sha256 == proof.pinned_sha256
    if not proof.matches:
        proof.note = "the installed package is NOT the pinned commit; evidence from it would cite the wrong source"
    return proof


def verify_runtimes(manifest: dict[str, Any], refs_root: Path) -> list[RuntimeProof]:
    """Every `[[runtimes]]` row: installed bytes against the pinned blob."""
    out: list[RuntimeProof] = []
    for row in manifest.get("runtimes", []):
        upstream = manifest["upstreams"][row["upstream"]]
        out.append(
            verify_runtime(
                package=row["package"],
                repo=clone_dir(refs_root, upstream["repo"]),
                commit=upstream["commit"],
                path=row["path"],
                installed_path=row["installed_path"],
            )
        )
    return out


def verify_all(manifest: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Every upstream and every consumer, against its pin."""
    refs_root: Path = (repo_root / manifest["refs_root"]).resolve()
    proofs: list[RepoProof] = []

    upstreams: dict[str, Any] = manifest.get("upstreams", {})
    for key, entry in upstreams.items():
        proofs.append(verify_repo(f"upstream:{key}", entry, refs_root))

    consumers: list[dict[str, Any]] = manifest.get("consumers", [])
    for consumer in consumers:
        # A consumer whose entrypoint lives in another upstream -- ConsisID in
        # diffusers, ID-LoRA in ComfyUI -- has its paths checked against THAT
        # repository, because that is where the contract is committed.
        host: str | None = consumer.get("entrypoint_in")
        if host:
            entry = dict(upstreams[host])
            proofs.append(verify_repo(f"consumer:{consumer['id']}@{host}", entry, refs_root, paths_from=consumer))
        else:
            proofs.append(verify_repo(f"consumer:{consumer['id']}", consumer, refs_root))

    runtimes: list[RuntimeProof] = verify_runtimes(manifest, refs_root)
    weights: list[dict[str, Any]] = weight_identity(manifest)
    vendor_weights: list[dict[str, Any]] = vendor_weight_identity(manifest)

    unread: list[str] = [one["id"] for one in consumers if one.get("source_unread")]

    return {
        "manifest_version": manifest["version"],
        "recorded_at": manifest["recorded_at"],
        "runtime": runtime_identity(),
        "refs_root": str(refs_root),
        "repos": [asdict(one) for one in proofs],
        "runtimes": [asdict(one) for one in runtimes],
        "weights": weights,
        "population": {
            "total": len(consumers),
            "source_unread": unread,
        },
        # Four checks, none excusing the others: a clone at the right commit,
        # what the interpreter imports, which of OUR weights computed the
        # numbers, and the VENDOR's own checkpoints.
        "vendor_weights": vendor_weights,
        "provenance_ok": (
            all(one.ok for one in proofs)
            and all(one.matches for one in runtimes)
            and all(one["present"] for one in weights)
            # A vendor checkpoint that IS here and does not match what was
            # recorded for it. Absent ones are not a failure: they are why
            # most of these consumers are not attempted on this machine.
            and all(one["matches"] is not False for one in vendor_weights)
            # A weight whose measured digest contradicts the digest its vendor
            # PUBLISHES is not our copy of their file. That is a harder
            # failure than absence: the run would proceed and be wrong.
            and all(one["matches_published"] is not False for one in weights)
        ),
    }


def report(out: dict[str, Any]) -> None:
    """Say what was checked, per row, so a reader can re-run any line of it."""
    repos: list[dict[str, Any]] = out["repos"]
    for repo in repos:
        mark: str = "ok  " if not repo["failures"] else "FAIL"
        print(f"{mark} {repo['key']:<46} {repo['pinned_commit'][:12]}  paths={len(repo['paths'])}")
        for failure in repo["failures"]:
            print(f"       ! {failure}")
        for one in repo["paths"]:
            if not one["present"]:
                print(f"       - {one['path']}::{one['symbol']}  ABSENT AT PIN")
                continue
            if one["locator_only"]:
                grade = "locator"
            elif one["symbol_present"]:
                grade = "symbol "
            else:
                grade = "NO SYM "
            print(f"       {grade} {one['path']}::{one['symbol']}  blob {one['blob_sha256'][:12]}")

    for row in out["runtimes"]:
        mark = "ok  " if row["matches"] else "FAIL"
        state = "matches pin" if row["matches"] else (row["note"] or "does not match pin")
        print(f"{mark} runtime:{row['package']:<38} {row['pinned_commit'][:12]}  {state}")

    for row in out["weights"]:
        mark = "ok  " if row["present"] and row["matches_published"] is not False else "FAIL"
        digest = row["sha256"][:12] if row["sha256"] else "ABSENT"
        against = ""
        if row["matches_published"] is True:
            against = f"  == published by {row['published_by']}"
        elif row["matches_published"] is False:
            against = f"  != PUBLISHED {row['published_sha256'][:12]} ({row['published_by']})"
        print(f"{mark} weight:{row['pack']}/{row['file']:<32} {digest}  {row['bytes']:>12,} bytes{against}")

    verified = [one for one in out["weights"] if one["matches_published"] is True]
    unverified = [one for one in out["weights"] if one["matches_published"] is None]
    print(
        f"\nweights vs vendor-published digests: {len(verified)} verified, {len(unverified)} with no published digest"
    )

    pop: dict[str, Any] = out["population"]
    print(f"\npopulation: {pop['total']} consumers")
    print(f"  source unread: {len(pop['source_unread'])}  {pop['source_unread']}")
    print(f"\nprovenance: {'PASS' if out['provenance_ok'] else 'FAIL'}")


def main() -> int:
    repo_root: Path = ROOT.parent
    manifest: dict[str, Any] = load_manifest()
    out: dict[str, Any] = verify_all(manifest, repo_root)
    report(out)

    where: Path = ROOT / "generated" / "provenance.json"
    where.parent.mkdir(parents=True, exist_ok=True)
    # newline="" so this file is byte-identical on every platform: Windows
    # would translate the line endings and a reviewer hashing the evidence
    # elsewhere would get a different digest for the same run.
    with where.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True))
        handle.write("\n")
    print(f"\nwrote {where}")

    # Provenance passing is not the harness passing. The population gate is
    # separate and stays red until every member is classified by execution.
    return 0 if out["provenance_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
