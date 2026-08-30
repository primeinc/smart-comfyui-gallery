"""Derive the proof population from the pinned consumer source. Never declare it.

The population is `every reachable face-model variant, and every artifact each
variant loads`. It is DISCOVERED by reading the pinned upstream, because every
declaration available to read instead is smaller than the truth:

    every face model  ->  22 consumers      (roots, not the population)
                      ->  13 [[weights]]    (what we happen to keep)
                      ->  30 vendor_weights (what somebody wrote down)

None of those may choose what must be proved, which is why nothing here reads
them.

WHAT THIS REPLACED, AND WHY
---------------------------
The previous version walked the IMPORT graph to a fixed depth and attached
every artifact literal in a FILE to every loader call in it. It reported 102
variants over 947 edges, and 678 of those edges were manufactured: one call
site in `eva_clip/pretrained.py` was credited with fourteen artifacts that
merely appeared in the same file. Its own controls convicted it -- a loader
four calls down was not found AT ALL, `images/output` became a model, and an
unresolved `config.model_name` collapsed into a fake `DEFAULT` variant.

A larger population is not a better proof. This walks CALLS from the declared
entrypoint to a fixpoint, and binds each artifact to the expression that
actually flows into that call.

WHAT A VARIANT IS
-----------------
Any independently selectable or independently loaded model, backend, checkpoint
or pack whose removal or substitution can change or prevent the deterministic
boundary. `FaceAnalysis(name="antelopev2")` and `FaceAnalysis()` are two
variants of one role: the default pack is buffalo_l, so the second loads
different bytes and yields a different embedding space.

Continuous parameters are not variants. A det_size ladder selects no different
model bytes; a `name=` does.

WHAT IS AND IS NOT KNOWN
------------------------
Every edge carries a status, and only two of them are population members:

    REQUIRED         on an unconditional path from the entrypoint
    CONDITIONAL      reachable, behind a branch this cannot evaluate
    UNRESOLVED       the callee or the artifact could not be resolved -- RED
    NOT_ON_BOUNDARY  retained as audit evidence for why it was excluded

An UNRESOLVED row is never silently approximated as complete. It is the honest
output of static reading and it is what dynamic observation exists to close.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import proc
from compat.harness import provenance

ROOT: Final[Path] = Path(__file__).resolve().parent.parent

#: Package roots a module path may sit under. ID-V2V uses a `src/` layout, so
#: `python -m idv2v.preprocess.sam3` is `src/idv2v/...`.
PACKAGE_ROOTS: Final[tuple[str, ...]] = ("", "src/", "python-package/", "lib/")

#: Extensions that are model bytes rather than code.
ARTIFACT_SUFFIX: Final[tuple[str, ...]] = (
    ".onnx",
    ".pth",
    ".pt",
    ".bin",
    ".safetensors",
    ".task",
    ".ckpt",
    ".npy",
    ".zip",
)

#: Loader call names, and the role each plays.
LOADERS: Final[dict[str, str]] = {
    "FaceAnalysis": "face_analysis_pack",
    "InferenceSession": "onnx_session",
    "from_pretrained": "pretrained_model",
    "load_file": "safetensors_load",
    "load": "torch_load",
    "init_detection_model": "facexlib_detection",
    "init_parsing_model": "facexlib_parsing",
    "init_recognition_model": "facexlib_recognition",
    "init_alignment_model": "facexlib_alignment",
    "create_model_and_transforms": "open_clip",
    "get_model": "insightface_model_zoo",
    "load_model": "generic_model_load",
    "hf_hub_download": "hub_download",
    "snapshot_download": "hub_snapshot",
    "load_ckpt": "checkpoint_load",
    "load_state_dict": "state_dict_load",
    "load_flow_model": "flux_flow_model",
    "load_ae": "flux_autoencoder",
    "load_clip": "clip_encoder",
    "load_t5": "t5_encoder",
}

#: The ONLY loaders whose string argument may be read as a hub repository id.
#: `org/name` also matches an ordinary two-component path, so a hub id is
#: accepted when it is bound to an API that expects one, never by its shape.
HUB_LOADERS: Final[frozenset[str]] = frozenset(
    {"from_pretrained", "hf_hub_download", "snapshot_download", "create_model_and_transforms"}
)

#: `python -m package.module` inside a shell entrypoint.
SHELL_MODULE: Final[re.Pattern[str]] = re.compile(r"python[0-9.]*\s+-m\s+([A-Za-z_][A-Za-z0-9_.]*)")

#: Expansions the traversal may make. `None` is the contract: it ends when no
#: new resolvable callable remains, never because an integer said so. A NAME,
#: so `population_attack` can reintroduce a cutoff and require control A red.
MAX_EXPANSIONS: int | None = None

REQUIRED: Final[str] = "REQUIRED"
CONDITIONAL: Final[str] = "CONDITIONAL"
UNRESOLVED: Final[str] = "UNRESOLVED"
NOT_ON_BOUNDARY: Final[str] = "NOT_ON_BOUNDARY"


@dataclass
class Edge:
    """One consumer x variant x loader branch x artifact edge."""

    consumer_id: str
    family: str
    consumer_repo: str
    consumer_revision: str
    entrypoint: str
    boundary_id: str
    boundary_source_locator: str
    model_variant_id: str
    model_variant_role: str
    configuration_id: str
    loader_branch: str
    activation_condition: str
    loader_source_locator: str
    artifact_role: str
    artifact_logical_identity: str
    artifact_source: str = ""
    artifact_revision: str = ""
    artifact_path: str = ""
    local_resolved_path: str = ""
    static_discovered: bool = True
    dynamic_observed: bool = False
    required_for_boundary: bool = False
    variant_static_discovered: bool = True
    variant_dynamic_exercised: bool = False
    shared_artifact_identity: str = ""
    discovery_evidence: str = ""
    discovery_status: str = UNRESOLVED


#: One tree listing per (clone, commit); one blob read per real path.
_TREES: dict[tuple[str, str], frozenset[str]] = {}
_BLOBS: dict[tuple[str, str, str], str | None] = {}


def tree_of(clone: Path, commit: str) -> frozenset[str]:
    key = (str(clone), commit)
    if key not in _TREES:
        argv: list[str] = ["git", "-C", str(clone), "ls-tree", "-r", "--name-only", commit]
        code, out, _ = proc.run(argv, timeout=proc.LOCAL_SECONDS)
        if code != 0:
            _TREES[key] = frozenset()
            return _TREES[key]
        held = out.decode("utf-8", errors="surrogateescape").splitlines()
        _TREES[key] = frozenset(one.strip() for one in held if one.strip())
    return _TREES[key]


def _blob(clone: Path, commit: str, path: str) -> str | None:
    key = (str(clone), commit, path)
    if key in _BLOBS:
        return _BLOBS[key]
    if path not in tree_of(clone, commit):
        _BLOBS[key] = None
        return None
    argv: list[str] = ["git", "-C", str(clone), "cat-file", "blob", f"{commit}:{path}"]
    code, out, _ = proc.run(argv, timeout=proc.LOCAL_SECONDS)
    if code != 0:
        _BLOBS[key] = None
        return None
    _BLOBS[key] = out.decode("utf-8", errors="surrogateescape")
    return _BLOBS[key]


def submodules(clone: Path, commit: str) -> dict[str, str]:
    """Submodule path -> the commit this revision pins it at.

    A submodule is a gitlink, so its contents are absent from this tree at
    every path. UMO reaches its whole pipeline through `projects/UNO`.
    """
    argv: list[str] = ["git", "-C", str(clone), "ls-tree", "-r", commit]
    code, listing, _ = proc.run(argv, timeout=proc.LOCAL_SECONDS)
    if code != 0:
        return {}
    out: dict[str, str] = {}
    for raw in listing.decode("utf-8", errors="surrogateescape").splitlines():
        meta, _, path = raw.partition("\t")
        parts = meta.split()
        if len(parts) >= 3 and parts[0] == "160000":
            out[path.strip()] = parts[2]
    return out


def submodule_clone(clone: Path, commit: str, path: str, refs_root: Path) -> Path | None:
    body = _blob(clone, commit, ".gitmodules")
    if body is None:
        return None
    url = ""
    for block in body.split("[submodule"):
        if f"path = {path}" in block:
            for line in block.splitlines():
                if line.strip().startswith("url = "):
                    url = line.split("=", 1)[1].strip()
    if not url:
        return None
    return refs_root / url.removeprefix("https://github.com/").removesuffix(".git")


@dataclass
class Source:
    """One parsed module, and what a caller can reach through it."""

    path: str
    tree: ast.Module
    #: name -> the dotted module it was imported from, for `from x import y`
    imported: dict[str, str] = field(default_factory=dict)
    #: name -> the function/class defined here
    defined: dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef] = field(default_factory=dict)
    #: module-level `NAME = "literal"`
    constants: dict[str, str] = field(default_factory=dict)


def parse(path: str, body: str) -> Source | None:
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return None
    held = Source(path=path, tree=tree)
    package = path.rsplit("/", 1)[0] if "/" in path else ""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            held.defined[node.name] = node
        elif isinstance(node, ast.Assign):
            text = _literal(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name) and text is not None:
                    held.constants[target.id] = text
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for one in node.names:
                held.imported[one.asname or one.name.split(".")[0]] = one.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split("/") if package else []
                base = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                stem = "/".join([*base, *(node.module.split(".") if node.module else [])])
                module = stem.replace("/", ".")
            else:
                module = node.module or ""
            for one in node.names:
                held.imported[one.asname or one.name] = f"{module}.{one.name}" if module else one.name
    return held


def candidate_paths(module: str) -> list[str]:
    stem = module.replace(".", "/")
    return [f"{root}{stem}{tail}" for root in PACKAGE_ROOTS for tail in (".py", "/__init__.py")]


class Graph:
    """Modules of one repository, loaded on demand and cached."""

    def __init__(self, clone: Path, commit: str, refs_root: Path) -> None:
        self.clone = clone
        self.commit = commit
        self.refs_root = refs_root
        self.subs = submodules(clone, commit)
        self.sources: dict[str, Source | None] = {}

    def read(self, path: str) -> str | None:
        crossed = next((one for one in self.subs if path.startswith(f"{one}/")), "")
        if not crossed:
            return _blob(self.clone, self.commit, path)
        where = submodule_clone(self.clone, self.commit, crossed, self.refs_root)
        if where is None or not (where / ".git").exists():
            return None
        return _blob(where, self.subs[crossed], path[len(crossed) + 1 :])

    def source(self, path: str) -> Source | None:
        if path not in self.sources:
            body = self.read(path)
            self.sources[path] = parse(path, body) if body is not None else None
        return self.sources[path]

    def resolve_module(self, module: str) -> Source | None:
        """The module, tried under every package root and every submodule."""
        for candidate in candidate_paths(module):
            held = self.source(candidate)
            if held is not None:
                return held
        for prefix in self.subs:
            for candidate in candidate_paths(module):
                held = self.source(f"{prefix}/{candidate}")
                if held is not None:
                    return held
        return None


def _literal(node: ast.expr) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _assignments(scope: ast.AST) -> dict[str, str]:
    """`NAME = "literal"` bindings inside one function body."""
    out: dict[str, str] = {}
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    text = _literal(node.value)
                    if text is not None:
                        out[target.id] = text
    return out


def _defaults(scope: ast.AST) -> dict[str, str]:
    """Parameter defaults of the enclosing function.

    `def build(model="facebook/sam3")` flows that default into every call in
    the body that passes `model`, which is how ID-V2V names SAM3.
    """
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {}
    out: dict[str, str] = {}
    args = scope.args
    positional = [*args.posonlyargs, *args.args]
    for name, default in zip(positional[len(positional) - len(args.defaults) :], args.defaults, strict=False):
        text = _literal(default)
        if text is not None:
            out[name.arg] = text
    for name, default in zip(args.kwonlyargs, args.kw_defaults, strict=False):
        text = _literal(default) if default is not None else None
        if text is not None:
            out[name.arg] = text
    return out


def bind_artifact(node: ast.expr, scope: ast.AST, source: Source) -> tuple[str, str]:
    """What flows into ONE argument, and how it was established.

    Only data that can actually reach this expression is considered: a literal,
    a binding in the enclosing function, a parameter default, or a module
    constant. Anything else is unresolved and says so, rather than borrowing a
    string from elsewhere in the file.
    """
    text = _literal(node)
    if text is not None:
        return text, "literal"
    if isinstance(node, ast.Name):
        local = _assignments(scope).get(node.id)
        if local is not None:
            return local, "local assignment"
        default = _defaults(scope).get(node.id)
        if default is not None:
            return default, "parameter default"
        constant = source.constants.get(node.id)
        if constant is not None:
            return constant, "module constant"
        return "", f"name {node.id!r} has no statically resolvable value"
    if isinstance(node, ast.Attribute):
        return "", f"attribute {ast.unparse(node)} is resolved at runtime"
    return "", f"{type(node).__name__} is not a static value"


def _bound_artifact(text: str, loader: str) -> bool:
    """Whether a resolved string names model bytes for THIS loader.

    A filename is an artifact anywhere. A bare `org/name` is a hub id only for
    an API that takes one -- shape alone made `images/output` a model.
    """
    if text.lower().endswith(ARTIFACT_SUFFIX):
        return True
    return loader in HUB_LOADERS and text.count("/") == 1 and "." not in text.split("/", 1)[0]


is_artifact: Callable[[str, str], bool] = _bound_artifact


def _own_arguments(node: ast.Call) -> list[ast.expr]:
    """The expressions that can supply THIS call's artifact.

    Its own arguments and nothing else.
    """
    return [*node.args, *(one.value for one in node.keywords if one.arg)]


#: Three SEAMS, held as callables rather than defs so `population_attack` can
#: reintroduce the defect each removed and require its control red. A `def`
#: cannot be reassigned without a type error, and this tree bans suppressions.
call_arguments: Callable[[ast.Call], list[ast.expr]] = _own_arguments


def _call_site_variant(where: str) -> str:
    """The identity of a selection static reading could not resolve.

    Keyed on the CALL SITE so two unresolved loaders stay two variants rather
    than collapsing into one invented default.
    """
    return f"UNRESOLVED_VARIANT:{where}"


unresolved_variant: Callable[[str], str] = _call_site_variant


def call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _conditions(scope: ast.AST, target: ast.Call) -> str:
    """The branch conditions enclosing one call, outermost first."""
    found: list[str] = []

    def walk(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                walk(child, [*stack, ast.unparse(child.test)[:80]])
                for other in child.orelse:
                    walk(other, [*stack, f"not ({ast.unparse(child.test)[:70]})"])
            else:
                if child is target and not found:
                    found.extend(stack)
                walk(child, stack)

    walk(scope, [])
    return " and ".join(found)


@dataclass
class Reach:
    """One reachable callable, and the branch conditions guarding it."""

    source: Source
    scope: ast.AST
    condition: str


def entry_scopes(graph: Graph, entrypoint: str) -> tuple[list[Reach], list[str]]:
    """The declared entry, resolved to callables. Never the whole file.

    `path.py::symbol` names ONE symbol. Treating the file as executable is how
    an unreachable helper enters the population, which control B exists to
    catch. A shell entrypoint has no symbol: its `python -m` targets are the
    real roots, and each module's top level is the entry.
    """
    path, _, symbol = entrypoint.partition("::")
    unresolved: list[str] = []
    if not path:
        return [], ["the consumer declares no entrypoint"]

    if not path.endswith(".py"):
        body = graph.read(path)
        if body is None:
            return [], [f"{path} is not at the pin"]
        out: list[Reach] = []
        for module in SHELL_MODULE.findall(body):
            held = graph.resolve_module(module)
            if held is None:
                unresolved.append(f"{path}: python -m {module} resolves to no module in this repository")
                continue
            out.append(Reach(held, held.tree, ""))
        if not out:
            unresolved.append(f"{path}: no runnable module was resolved from the shell entrypoint")
        return out, unresolved

    source = graph.source(path)
    if source is None:
        return [], [f"{path} is not at the pin"]
    if not symbol or symbol.startswith("__"):
        return [Reach(source, source.tree, "")], unresolved
    leaf = symbol.rsplit(".", 1)[-1]
    found = source.defined.get(leaf)
    if found is None:
        return [], [f"{path} does not define {symbol}"]
    return [Reach(source, found, "")], unresolved


def reachable_calls(graph: Graph, roots: list[Reach]) -> tuple[list[tuple[Reach, ast.Call]], list[str]]:
    """Every call reachable from the entry, to a fixpoint.

    No hop budget. Traversal ends when no new resolvable callable remains, and
    a call whose target cannot be resolved is recorded rather than silently
    ending that branch.
    """
    seen: set[tuple[str, int]] = set()
    calls: list[tuple[Reach, ast.Call]] = []
    unresolved: list[str] = []
    frontier = list(roots)
    expansions = 0

    while frontier:
        if MAX_EXPANSIONS is not None and expansions >= MAX_EXPANSIONS:
            unresolved.append(f"traversal stopped after {MAX_EXPANSIONS} expansions")
            break
        expansions += 1
        here = frontier.pop()
        key = (here.source.path, id(here.scope))
        if key in seen:
            continue
        seen.add(key)

        for node in ast.walk(here.scope):
            if not isinstance(node, ast.Call):
                continue
            name = call_name(node)
            condition = _conditions(here.scope, node) or here.condition
            if name in LOADERS:
                calls.append((Reach(here.source, here.scope, condition), node))
                continue
            target = here.source.defined.get(name)
            if target is not None:
                frontier.append(Reach(here.source, target, condition))
                continue
            module = here.source.imported.get(name)
            if module is None:
                continue
            held = graph.resolve_module(module)
            if held is None:
                held = graph.resolve_module(module.rsplit(".", 1)[0])
            if held is None:
                unresolved.append(f"{here.source.path}: {name} resolves to no module ({module})")
                continue
            inner = held.defined.get(module.rsplit(".", 1)[-1]) or held.defined.get(name)
            if inner is None:
                unresolved.append(f"{here.source.path}: {name} is not defined in {held.path}")
                continue
            frontier.append(Reach(held, inner, condition))
    return calls, unresolved


def _edge(consumer: dict[str, Any], repo: str, commit: str, entry: str, boundary: str, **rest: Any) -> Edge:
    return Edge(
        consumer_id=consumer["id"],
        family=str(consumer.get("family", "")),
        consumer_repo=repo,
        consumer_revision=commit,
        entrypoint=entry,
        boundary_id=boundary,
        boundary_source_locator=f"{repo}@{commit}:{entry}",
        **rest,
    )


def discover(consumer: dict[str, Any], upstreams: dict[str, Any], refs_root: Path) -> list[Edge]:
    """Every loader call reachable from one consumer's declared entrypoint."""
    host = consumer.get("entrypoint_in")
    source_row = upstreams[host] if host else consumer
    repo = str(source_row["repo"]).removeprefix("https://github.com/").removesuffix(".git")
    commit = str(source_row["commit"])
    clone = provenance.clone_dir(refs_root, source_row["repo"])
    entry = str(consumer.get("entrypoint", ""))
    boundary = ",".join(consumer.get("boundary", []))

    graph = Graph(clone, commit, refs_root)
    roots, why = entry_scopes(graph, entry)
    calls, more = reachable_calls(graph, roots)

    out: list[Edge] = []
    for reach, node in calls:
        name = call_name(node)
        role = LOADERS[name]
        where = f"{repo}@{commit}:{reach.source.path}:{node.lineno}"

        # ONLY the arguments of THIS call. The previous version took every
        # artifact literal in the file, which credited one call site with
        # fourteen artifacts.
        arguments: list[ast.expr] = call_arguments(node)
        bound: list[tuple[str, str]] = []
        reasons: list[str] = []
        for argument in arguments:
            text, how = bind_artifact(argument, reach.scope, reach.source)
            if text and is_artifact(text, name):
                bound.append((text, how))
            elif not text:
                reasons.append(how)

        for artifact, how in bound:
            out.append(
                _edge(
                    consumer,
                    repo,
                    commit,
                    entry,
                    boundary,
                    model_variant_id=f"{name}:{artifact}",
                    model_variant_role=role,
                    configuration_id=how,
                    loader_branch=f"{reach.source.path}:{node.lineno}",
                    activation_condition=reach.condition,
                    loader_source_locator=where,
                    artifact_role=role,
                    artifact_logical_identity=artifact,
                    artifact_path=artifact if artifact.lower().endswith(ARTIFACT_SUFFIX) else "",
                    artifact_source=artifact if name in HUB_LOADERS and "/" in artifact else "",
                    shared_artifact_identity=Path(artifact).name,
                    discovery_evidence=f"{how} at {where}",
                    required_for_boundary=not reach.condition,
                    discovery_status=CONDITIONAL if reach.condition else REQUIRED,
                )
            )
        if bound:
            continue

        # A loader whose selection did not resolve. Its identity is the CALL
        # SITE, so two unresolved sites stay two variants rather than
        # collapsing into one invented default.
        out.append(
            _edge(
                consumer,
                repo,
                commit,
                entry,
                boundary,
                model_variant_id=unresolved_variant(where),
                model_variant_role=role,
                configuration_id="; ".join(reasons)[:200],
                loader_branch=f"{reach.source.path}:{node.lineno}",
                activation_condition=reach.condition,
                loader_source_locator=where,
                artifact_role=role,
                artifact_logical_identity=f"UNRESOLVED_ARTIFACT:{where}",
                discovery_evidence="; ".join(reasons)[:200] or "the loader takes no static argument",
                required_for_boundary=False,
                discovery_status=UNRESOLVED,
            )
        )

    # A root that discovered NO loader is an unresolved surface, never
    # silence: `instantid` receives its FaceAnalysis as a parameter, so the
    # model was selected by a caller this root cannot see.
    if not calls:
        more = [
            *more,
            (
                f"{entry} reaches no loader: the model is selected upstream of this entrypoint "
                f"or arrives as a parameter, so this root cannot see which bytes are loaded"
            ),
        ]

    out.extend(
        _edge(
            consumer,
            repo,
            commit,
            entry,
            boundary,
            model_variant_id=f"UNRESOLVED_CALL:{one[:80]}",
            model_variant_role="unresolved_call",
            configuration_id="",
            loader_branch="",
            activation_condition="",
            loader_source_locator=f"{repo}@{commit}",
            artifact_role="unresolved_call",
            artifact_logical_identity=f"UNRESOLVED_CALL:{one[:80]}",
            discovery_evidence=one,
            required_for_boundary=False,
            discovery_status=UNRESOLVED,
        )
        for one in [*why, *more]
    )
    return out


def build() -> dict[str, Any]:
    manifest = provenance.load_manifest()
    refs_root = (ROOT.parent / manifest["refs_root"]).resolve()
    upstreams = manifest.get("upstreams", {})
    consumers = manifest.get("consumers", [])

    edges: list[Edge] = []
    for consumer in sorted(consumers, key=lambda one: one["id"]):
        edges.extend(discover(consumer, upstreams, refs_root))

    def variants(status: str) -> set[tuple[str, str]]:
        return {(one.consumer_id, one.model_variant_id) for one in edges if one.discovery_status == status}

    per_consumer: dict[str, int] = {}
    for one in edges:
        per_consumer[one.consumer_id] = per_consumer.get(one.consumer_id, 0) + 1
    from compat.harness import identity as evidence_identity

    return {
        # Stamped, so `reconcile` can refuse to compare a population against
        # observations from a different tree.
        "identity": evidence_identity.identity()["digest"],
        "roots": sorted(one["id"] for one in consumers),
        "edges": [asdict(one) for one in edges],
        "totals": {
            "roots": len(consumers),
            "required_variants": len(variants(REQUIRED)),
            "conditional_variants": len(variants(CONDITIONAL)),
            "unresolved_variants": len(variants(UNRESOLVED)),
            "not_on_boundary": len(variants(NOT_ON_BOUNDARY)),
            "edges": len(edges),
            "semantic_artifacts": len({one.artifact_logical_identity for one in edges}),
            "physical_artifacts": len({one.shared_artifact_identity for one in edges if one.shared_artifact_identity}),
            "unresolved_calls": sum(1 for one in edges if one.model_variant_role == "unresolved_call"),
            "edges_per_consumer": per_consumer,
        },
    }


def main() -> int:
    out = build()
    totals = out["totals"]
    print(f"discovery roots                 : {totals['roots']}")
    print(f"REQUIRED variants               : {totals['required_variants']}")
    print(f"CONDITIONAL variants            : {totals['conditional_variants']}")
    print(f"UNRESOLVED variants             : {totals['unresolved_variants']}")
    print(f"NOT_ON_BOUNDARY candidates      : {totals['not_on_boundary']}")
    print(f"consumer x variant x edge rows  : {totals['edges']}")
    print(f"unique semantic artifacts       : {totals['semantic_artifacts']}")
    print(f"unique physical artifacts       : {totals['physical_artifacts']}")
    print(f"unresolved calls / selections   : {totals['unresolved_calls']}\n")

    print(f"{'consumer':<24} {'edges':>6} {'REQ':>5} {'COND':>5} {'UNRES':>6}")
    for who in out["roots"]:
        mine = [one for one in out["edges"] if one["consumer_id"] == who]
        req = sum(1 for one in mine if one["discovery_status"] == REQUIRED)
        cond = sum(1 for one in mine if one["discovery_status"] == CONDITIONAL)
        unres = sum(1 for one in mine if one["discovery_status"] == UNRESOLVED)
        print(f"{who:<24} {len(mine):>6} {req:>5} {cond:>5} {unres:>6}")

    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    target = generated / "artifact_population.json"
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(out, indent=2, sort_keys=True, default=str))
        handle.write("\n")
    print(f"\nwrote {target}")
    # COMPLETE only with no unresolved surface. The evidence stays usable
    # either way; the red says it is not yet a population.
    return 0 if not totals["unresolved_variants"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
