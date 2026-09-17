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


PACKAGE_ROOTS: Final[tuple[str, ...]] = ("", "src/", "python-package/", "lib/")


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


HUB_LOADERS: Final[frozenset[str]] = frozenset(
    {"from_pretrained", "hf_hub_download", "snapshot_download", "create_model_and_transforms"}
)

#: Loader names that are ordinary vocabulary elsewhere: `json.load` and
#: `yaml.load` read data, not model bytes. A name listed here is a loader
#: only when its receiver is one of the named modules.
RECEIVER_REQUIRED: Final[dict[str, frozenset[str]]] = {"load": frozenset({"torch"})}

#: Loaders whose artifact is a bare PACK NAME rather than a path or hub id:
#: insightface's FaceAnalysis("antelopev2") and facexlib's init_* take the
#: model's registry name, and that name IS the variant identity.
PACK_NAME_LOADERS: Final[frozenset[str]] = frozenset(
    {
        "FaceAnalysis",
        "get_model",
        "init_detection_model",
        "init_parsing_model",
        "init_recognition_model",
        "init_alignment_model",
    }
)


SHELL_MODULE: Final[re.Pattern[str]] = re.compile(r"python[0-9.]*\s+-m\s+([A-Za-z_][A-Za-z0-9_.]*)")


#: Declared verdicts for edges the static walk PROVES it cannot settle: a
#: model that arrives from the host process, or a path only an operator can
#: supply.
#:
#: Keyed by consumer and the site (`path:line`) or, for traversal findings,
#: the finding's leading phrase.
#:
#: Each entry is a decision somebody wrote down, with its reason. A ruling
#: that matches nothing in a run is stale, and build() turns it into an
#: UNRESOLVED edge so the lane goes red instead of quietly narrowing.
RULINGS: Final[dict[tuple[str, str], str]] = {
    ("consisid", "src/diffusers/pipelines/consisid/consisid_utils.py::process_face_embeddings reaches no loader"): (
        "the entry receives its face models as parameters; the walk proves no loader is reachable from it, "
        "and the bytes it consumes are pinned by the weights part and exercised by the observation lanes"
    ),
    ("id_lora", "comfy_extras/nodes_lt.py::LTXVReferenceAudio.execute reaches no loader"): (
        "a ComfyUI node: models arrive from the graph's loader nodes, outside this entry's closure"
    ),
    ("infiniteyou", "nodes.py::ExtractIDEmbedding.extract_id_embedding reaches no loader"): (
        "a ComfyUI node: models arrive from the graph's loader nodes, outside this entry's closure"
    ),
    ("instantcharacter", "pipeline.py::InstantCharacterFluxPipeline.__call__ reaches no loader"): (
        "__call__ runs a pipeline whose models were constructed before the declared boundary; "
        "no loader is reachable from the call itself"
    ),
    ("instantid", "InstantID.py::extractFeatures reaches no loader"): (
        "a ComfyUI helper: the insightface handle it uses is built by the host graph"
    ),
    ("ipadapter_upstream", "ip_adapter/ip_adapter_faceid.py::IPAdapterFaceID.get_image_embeds reaches no loader"): (
        "the declared boundary is the embedding call; the adapter's checkpoints are loaded at construction, outside it"
    ),
    ("pulid_upstream", "pulid/pipeline.py::PuLIDPipeline.get_id_embedding reaches no loader"): (
        "the declared boundary is the embedding call; the pipeline's models are loaded at construction, outside it"
    ),
    ("qwen_image_edit_2509", "src/examples/edit_demo.py::infer reaches no loader"): (
        "the entry receives an already-built pipeline; no loader is reachable from it"
    ),
    ("reactor", "reactor_utils.py::save_face_model reaches no loader"): (
        "the entry persists an already-built face model; no loader is reachable from it"
    ),
    ("uniportrait", "gradio_app.py::process_faceid_image reaches no loader"): (
        "a gradio handler over models the app builds at startup, outside this entry's closure"
    ),
    ("ipadapter_faceid", "IPAdapterPlus.py:72"): (
        "IPAdapter.__init__ receives `ipadapter_model`, a state dict its host loaded (IPAdapterPlus.py:50); "
        "the subscripted slices carry those host-loaded bytes"
    ),
    ("ipadapter_faceid_plus", "IPAdapterPlus.py:72"): (
        "IPAdapter.__init__ receives `ipadapter_model`, a state dict its host loaded (IPAdapterPlus.py:50); "
        "the subscripted slices carry those host-loaded bytes"
    ),
    ("omnigen2", "inference.py:175"): (
        "--model_path is required with NO shipped default (inference.py parse_args); the tree names no bytes, "
        "the operator does"
    ),
    ("omnigen2", "inference.py:183"): (
        "--transformer_path defaults to None; this branch activates only under an operator-supplied override"
    ),
    ("omnigen2", "inference.py:188"): (
        "--model_path is required with NO shipped default (inference.py parse_args); the tree names no bytes, "
        "the operator does"
    ),
    ("umo", "eval/UNO/inference_omnicontext.py:83"): (
        "InferenceArgs.lora_path defaults to None; this branch activates only under an operator-supplied --lora_path"
    ),
    ("xverse", "src/flux/pipeline_tools.py:630"): (
        "lora_file joins an operator-supplied ckpt_dir with a fixed filename; the shipped tree names no "
        "directory for it"
    ),
}


#: Loads the harness itself performs: the producers and case fixtures of
#: this repository open these models, so the population claims them the way
#: it claims a consumer repo's loads -- declared, with the opening line.
FIRST_PARTY_EDGES: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("insightface_producer", "antelopev2", "face_analysis_pack", "compat/producers/registry.py:212"),
    ("insightface_producer", "buffalo_l", "face_analysis_pack", "compat/producers/registry.py:213"),
    ("embedding_spaces", "arcface", "facexlib_recognition", "compat/consumers/face_family.py:127"),
    ("consisid", "retinaface_resnet50", "facexlib_detection", "compat/consumers/consisid_facexlib.py:59"),
    ("consisid", "parsenet", "facexlib_parsing", "compat/consumers/consisid_facexlib.py:59"),
    ("id_v2v", "multi-task-model-vitl16_384.onnx", "david_normal_estimator", "compat/consumers/control_stream.py:170"),
)


def first_party_edges() -> list[Edge]:
    out: list[Edge] = []
    for consumer_id, artifact, role, locator in FIRST_PARTY_EDGES:
        out.append(
            Edge(
                consumer_id=consumer_id,
                family="first_party",
                consumer_repo="",
                consumer_revision="",
                entrypoint=locator,
                boundary_id="",
                boundary_source_locator=locator,
                model_variant_id=f"first_party:{artifact}",
                model_variant_role=role,
                configuration_id="declared first-party load",
                loader_branch=locator,
                activation_condition="",
                loader_source_locator=locator,
                artifact_role=role,
                artifact_logical_identity=artifact,
                shared_artifact_identity=artifact,
                discovery_evidence=f"declared: the harness opens it at {locator}",
                required_for_boundary=True,
                static_discovered=False,
                discovery_status=REQUIRED,
            )
        )
    return out


def ruled(consumer_id: str, edge: Edge) -> str:
    direct = RULINGS.get((consumer_id, edge.loader_branch))
    if direct is not None:
        return direct
    for (who, marker), reason in RULINGS.items():
        if who == consumer_id and marker.endswith("reaches no loader") and edge.discovery_evidence.startswith(marker):
            return reason
    return ""


MAX_EXPANSIONS: int | None = None

REQUIRED: Final[str] = "REQUIRED"
CONDITIONAL: Final[str] = "CONDITIONAL"
UNRESOLVED: Final[str] = "UNRESOLVED"
NOT_ON_BOUNDARY: Final[str] = "NOT_ON_BOUNDARY"


@dataclass
class Edge:
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
    path: str
    tree: ast.Module

    imported: dict[str, str] = field(default_factory=dict)

    defined: dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef] = field(default_factory=dict)

    constants: dict[str, str] = field(default_factory=dict)

    #: Module-level dict literals of str keys whose values are constructor
    #: calls with literal keyword arguments: the flux `configs` table shape.
    #: tables[name][key][kwarg] is a literal the evaluator can hand back.
    tables: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)

    #: Class name -> field defaults, from annotated assignments with literal
    #: defaults: the HfArgumentParser dataclass shape. `args.model_type` on a
    #: parameter annotated with the class resolves to the shipped default.
    records: dict[str, dict[str, str]] = field(default_factory=dict)

    #: Module-level dict literals of str -> str.
    mappings: dict[str, dict[str, str]] = field(default_factory=dict)

    #: argparse defaults: `add_argument("--model-path", default="...")`
    #: anywhere in the module, keyed the way the parsed namespace spells it.
    argdefaults: dict[str, str] = field(default_factory=dict)


def _table_of(node: ast.expr) -> dict[str, dict[str, str]] | None:
    if not isinstance(node, ast.Dict):
        return None
    out: dict[str, dict[str, str]] = {}
    for key, value in zip(node.keys, node.values, strict=False):
        name = _literal(key) if key is not None else None
        if name is None or not isinstance(value, ast.Call):
            return None
        row = {one.arg: text for one in value.keywords if one.arg and (text := _literal(one.value)) is not None}
        out[name] = row
    return out or None


def _mapping_of(node: ast.expr) -> dict[str, str] | None:
    if not isinstance(node, ast.Dict):
        return None
    out: dict[str, str] = {}
    for key, value in zip(node.keys, node.values, strict=False):
        name = _literal(key) if key is not None else None
        text = _literal(value)
        if name is None or text is None:
            return None
        out[name] = text
    return out or None


def parse(path: str, body: str) -> Source | None:
    try:
        tree = ast.parse(body)
    except SyntaxError:
        return None
    held = Source(path=path, tree=tree)
    package = path.rsplit("/", 1)[0] if "/" in path else ""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            held.defined[node.name] = node
        elif isinstance(node, ast.ClassDef):
            held.defined[node.name] = node
            fields: dict[str, str] = {}
            for inner in node.body:
                if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    held.defined[f"{node.name}.{inner.name}"] = inner
                elif isinstance(inner, ast.AnnAssign) and isinstance(inner.target, ast.Name):
                    text = _literal(inner.value) if inner.value is not None else None
                    if text is not None:
                        fields[inner.target.id] = text
            if fields:
                held.records[node.name] = fields
        elif isinstance(node, ast.Assign):
            text = _literal(node.value)
            table = _table_of(node.value)
            mapping = _mapping_of(node.value)
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                if text is not None:
                    held.constants[target.id] = text
                if table is not None:
                    held.tables[target.id] = table
                if mapping is not None:
                    held.mappings[target.id] = mapping
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            flag = _literal(node.args[0]) if node.args else None
            fallback = next((_literal(one.value) for one in node.keywords if one.arg == "default"), None)
            if flag and flag.startswith("--") and fallback is not None:
                held.argdefaults[flag.lstrip("-").replace("-", "_")] = fallback
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
        for candidate in candidate_paths(module):
            held = self.source(candidate)
            if held is not None:
                return held
        for prefix in self.subs:
            for candidate in candidate_paths(module):
                held = self.source(f"{prefix}/{candidate}")
                if held is not None:
                    return held
        # The sys.path.append("projects/X") spelling: the module name
        # carries the tree prefix, and `source` crosses any submodule
        # boundary the prefix lands on.
        dotted = module.replace(".", "/")
        if "/" in dotted:
            for tail in (".py", "/__init__.py"):
                held = self.source(f"{dotted}{tail}")
                if held is not None:
                    return held
        return None

    def holds_package(self, module: str) -> bool:
        top = module.split(".", 1)[0]
        tree = tree_of(self.clone, self.commit)
        prefixes = ["", *(f"{one}/" for one in self.subs)]
        stems = [f"{root}{top}" for root in PACKAGE_ROOTS]
        for prefix in prefixes:
            for stem in stems:
                if any(path == f"{prefix}{stem}.py" or path.startswith(f"{prefix}{stem}/") for path in tree):
                    return True
        return False


def _literal(node: ast.expr) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


@dataclass
class Env:
    #: Values carried into a scope from its call site, plus names bound by
    #: loader-call dataflow while the caller's scope was walked.
    names: dict[str, str] = field(default_factory=dict)
    owner: ast.ClassDef | None = None


def _loader_role(node: ast.Call, source: Source) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        role = LOADERS.get(func.attr)
        if role is None:
            return ""
        needed = RECEIVER_REQUIRED.get(func.attr)
        if needed is not None:
            receiver = func.value.id if isinstance(func.value, ast.Name) else ""
            if receiver not in needed:
                return ""
        return role
    if isinstance(func, ast.Name):
        target = source.imported.get(func.id, "")
        leaf = target.rsplit(".", 1)[-1] if target else func.id
        role = LOADERS.get(leaf)
        if role is None:
            return ""
        needed = RECEIVER_REQUIRED.get(leaf)
        if needed is not None:
            head = target.split(".", 1)[0] if target else ""
            if head not in needed:
                return ""
        return role
    return ""


def _defaults(scope: ast.AST) -> dict[str, str]:
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


def _parses_args(name: str, scope: ast.AST) -> bool:
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        parses = isinstance(func, ast.Attribute) and func.attr.startswith("parse_args")
        if parses and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return True
    return False


def _annotation_of(name: str, scope: ast.AST) -> str:
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    for arg in [*scope.args.posonlyargs, *scope.args.args, *scope.args.kwonlyargs]:
        if arg.arg == name and isinstance(arg.annotation, ast.Name):
            return arg.annotation.id
    return ""


def _fold_call(node: ast.Call, scope: ast.AST, source: Source, env: Env, depth: int) -> str:
    # The one dynamic call whose shipped value IS static: an env lookup with
    # a literal fallback names the artifact the tree ships.
    func = node.func
    if not isinstance(func, ast.Attribute):
        return ""
    receiver = ast.unparse(func.value)
    if (receiver, func.attr) in (("os.environ", "get"), ("os", "getenv")) and len(node.args) == 2:
        return _literal(node.args[1]) or ""
    if (receiver, func.attr) == ("os.path", "join") and node.args:
        parts = [_value(one, scope, source, env, depth)[0] for one in node.args]
        if all(parts):
            return "/".join(parts)
    if func.attr == "replace" and len(node.args) == 2:
        base, _ = _value(func.value, scope, source, env, depth)
        first, second = _literal(node.args[0]), _literal(node.args[1])
        if base and first is not None and second is not None:
            return base.replace(first, second)
    return ""


def _loop_value(name: str, scope: ast.AST, source: Source, env: Env, depth: int) -> str:
    # A loop over the items of loaded bytes binds its targets to the
    # identity of what was loaded.
    for node in ast.walk(scope):
        if not isinstance(node, ast.For):
            continue
        targets = node.target.elts if isinstance(node.target, ast.Tuple) else [node.target]
        if not any(isinstance(one, ast.Name) and one.id == name for one in targets):
            continue
        iterated = node.iter
        if isinstance(iterated, ast.Call) and isinstance(iterated.func, ast.Attribute):
            iterated = iterated.func.value
        base, _ = _value(iterated, scope, source, env, depth)
        if base and ("/" in base or base.lower().endswith(ARTIFACT_SUFFIX)):
            return base
    return ""


def _assign_value(name: str, scope: ast.AST, source: Source, env: Env, depth: int) -> str:
    found = ""
    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign):
            continue
        stored = any(
            isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == name
            for target in node.targets
        )
        if stored and not found:
            text, _ = _value(node.value, scope, source, env, depth)
            if not text and isinstance(node.value, ast.Name):
                text = _loop_value(node.value.id, scope, source, env, depth)
            if text:
                found = text
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        text, _ = _value(node.value, scope, source, env, depth)
        if text:
            found = text
            continue
        # Loader-call dataflow: `ckpt_path = hf_hub_download(...)` binds the
        # name to the bytes that call identifies. A later non-static rewrite
        # does NOT clear it -- the bytes still originate at that loader.
        if isinstance(node.value, ast.Call) and _loader_role(node.value, source):
            artifact, _ = _site_artifact(node.value, scope, source, env, depth)
            if artifact:
                found = artifact
    return found


def _value(node: ast.expr, scope: ast.AST, source: Source, env: Env, depth: int = 0) -> tuple[str, str]:
    if depth > 8:
        return "", "resolution depth exceeded"
    text = _literal(node)
    if text is not None:
        return text, "literal"
    if isinstance(node, ast.Name):
        if node.id in env.names:
            return env.names[node.id], "call-site argument"
        local = _assign_value(node.id, scope, source, env, depth + 1)
        if local:
            return local, "local assignment"
        default = _defaults(scope).get(node.id)
        if default is not None:
            return default, "parameter default"
        constant = source.constants.get(node.id)
        if constant is not None:
            return constant, "module constant"
        looped = _loop_value(node.id, scope, source, env, depth + 1)
        if looped:
            return looped, "iterates over loaded bytes"
        return "", f"name {node.id!r} has no statically resolvable value"
    if isinstance(node, ast.Attribute):
        base = node.value
        if isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name):
            table = source.tables.get(base.value.id)
            key, _ = _value(base.slice, scope, source, env, depth + 1)
            if table is not None and key and key in table:
                held = table[key].get(node.attr)
                if held is not None:
                    return held, f"{base.value.id}[{key!r}].{node.attr}"
        if isinstance(base, ast.Name):
            record = source.records.get(_annotation_of(base.id, scope))
            if record is not None and node.attr in record:
                return record[node.attr], f"{base.id}.{node.attr} dataclass default"
            if node.attr in source.argdefaults and (base.id == "args" or _parses_args(base.id, scope)):
                return source.argdefaults[node.attr], f"argparse default for --{node.attr}"
        return "", f"attribute {ast.unparse(node)} is resolved at runtime"
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        mapping = source.mappings.get(node.value.id)
        key, _ = _value(node.slice, scope, source, env, depth + 1)
        if mapping is not None and key and key in mapping:
            return mapping[key], f"{node.value.id}[{key!r}]"
        base, origin = _value(node.value, scope, source, env, depth + 1)
        if base and ("/" in base or base.lower().endswith(ARTIFACT_SUFFIX)):
            # A slice of loaded bytes carries the loaded thing's identity.
            return base, f"slice of {origin}"
        return "", "Subscript is not a static value"
    if isinstance(node, ast.Call):
        folded = _fold_call(node, scope, source, env, depth + 1)
        if folded:
            return folded, "shipped default"
        return "", "Call is not a static value"
    if isinstance(node, ast.IfExp):
        this, _ = _value(node.body, scope, source, env, depth + 1)
        that, _ = _value(node.orelse, scope, source, env, depth + 1)
        if this and this == that:
            return this, "both branches agree"
        return "", "IfExp is not a static value"
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for piece in node.values:
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                parts.append(piece.value)
                continue
            if isinstance(piece, ast.FormattedValue):
                inner, _ = _value(piece.value, scope, source, env, depth + 1)
                if inner:
                    parts.append(inner)
                    continue
            return "", "JoinedStr is not a static value"
        return "".join(parts), "static f-string"
    return "", f"{type(node).__name__} is not a static value"


def _site_artifact(node: ast.Call, scope: ast.AST, source: Source, env: Env, depth: int = 0) -> tuple[str, str]:
    # One artifact per site. A hub loader identified by (repo, filename)
    # yields the pair joined as one identity, never two edges at one line.
    role = _loader_role(node, source)
    loader = call_name(node)
    bound: list[tuple[str, str]] = []
    for argument in call_arguments(node):
        text, how = _value(argument, scope, source, env, depth + 1)
        if text and is_artifact(text, loader):
            bound.append((text, how))
    if not bound and loader in PACK_NAME_LOADERS:
        # The pack name, from the `name=` keyword or the first positional:
        # a registry identifier, not a path, so the shape gates above never
        # accept it. Only that one argument may carry the identity.
        chosen = next((one.value for one in node.keywords if one.arg in {"name", "model_name"}), None)
        if chosen is None and node.args:
            chosen = node.args[0]
        if chosen is not None:
            text, how = _value(chosen, scope, source, env, depth + 1)
            if text and "/" not in text and "\\" not in text and not text.startswith("."):
                bound.append((text, f"pack name; {how}"))
    if not bound and loader in HUB_LOADERS and node.args:
        # A hub-family loader's first positional IS the model -- a local
        # checkout directory names the same bytes a repo id would.
        text, how = _value(node.args[0], scope, source, env, depth + 1)
        if text:
            bound.append((text, f"local model path; {how}"))
    if not bound:
        return "", ""
    if role in {"hub_download", "hub_snapshot"} or loader in HUB_LOADERS:
        repos = [one for one in bound if "/" in one[0] and not one[0].lower().endswith(ARTIFACT_SUFFIX)]
        names = [one for one in bound if one[0].lower().endswith(ARTIFACT_SUFFIX)]
        if repos and names:
            return f"{repos[0][0]}/{names[0][0]}", f"{repos[0][1]} + {names[0][1]}"
    return bound[0]


def bind_artifact(node: ast.expr, scope: ast.AST, source: Source, env: Env | None = None) -> tuple[str, str]:
    return _value(node, scope, source, env or Env())


def _bound_artifact(text: str, loader: str) -> bool:
    if text.lower().endswith(ARTIFACT_SUFFIX):
        return True
    return loader in HUB_LOADERS and text.count("/") == 1 and "." not in text.split("/", 1)[0]


is_artifact: Callable[[str, str], bool] = _bound_artifact


def _own_arguments(node: ast.Call) -> list[ast.expr]:
    return [*node.args, *(one.value for one in node.keywords if one.arg)]


call_arguments: Callable[[ast.Call], list[ast.expr]] = _own_arguments


def _call_site_variant(where: str) -> str:
    return f"UNRESOLVED_VARIANT:{where}"


unresolved_variant: Callable[[str], str] = _call_site_variant


def call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _conditions(scope: ast.AST, target: ast.Call) -> str:
    found: list[str] = []

    def walk(node: ast.AST, stack: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
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
    source: Source
    scope: ast.AST
    condition: str
    env: Env = field(default_factory=Env)


def _bound_call(
    node: ast.Call,
    target: ast.FunctionDef | ast.AsyncFunctionDef,
    scope: ast.AST,
    source: Source,
    env: Env,
    skip_self: bool,
) -> dict[str, str]:
    params = [one.arg for one in [*target.args.posonlyargs, *target.args.args]]
    if skip_self and params and params[0] in {"self", "cls"}:
        params = params[1:]
    out: dict[str, str] = {}
    for value, param in zip(node.args, params, strict=False):
        text, _ = _value(value, scope, source, env, 1)
        if text:
            out[param] = text
    names = {*params, *(one.arg for one in target.args.kwonlyargs)}
    for keyword in node.keywords:
        if keyword.arg and keyword.arg in names:
            text, _ = _value(keyword.value, scope, source, env, 1)
            if text:
                out[keyword.arg] = text
    return out


def entry_scopes(graph: Graph, entrypoint: str) -> tuple[list[Reach], list[str]]:
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
    found = source.defined.get(symbol) or source.defined.get(symbol.rsplit(".", 1)[-1])
    if found is None:
        return [], [f"{path} does not define {symbol}"]
    owner = source.defined.get(symbol.split(".", 1)[0]) if "." in symbol else None
    env = Env(owner=owner if isinstance(owner, ast.ClassDef) else None)
    return [Reach(source, found, "", env)], unresolved


def _local_wrapper(
    graph: Graph, here: Reach, node: ast.Call
) -> tuple[Source, ast.FunctionDef | ast.AsyncFunctionDef, bool] | None:
    # A loader NAME the repository itself defines -- in this file or behind
    # an import -- is a wrapper: the true loads live in its body, so the
    # walk descends instead of stopping at the call.
    func = node.func
    if isinstance(func, ast.Name):
        held = here.source.defined.get(func.id)
        if isinstance(held, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return here.source, held, False
        module = here.source.imported.get(func.id)
        if module is not None:
            landed = graph.resolve_module(module) or graph.resolve_module(module.rsplit(".", 1)[0])
            if landed is not None:
                inner = landed.defined.get(module.rsplit(".", 1)[-1]) or landed.defined.get(func.id)
                if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return landed, inner, False
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "self"
        and here.env.owner is not None
    ):
        held = here.source.defined.get(f"{here.env.owner.name}.{func.attr}")
        if isinstance(held, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return here.source, held, True
    return None


def _own_calls(scope: ast.AST) -> list[ast.Call]:
    # The scope's OWN calls: nested function and class bodies are reached
    # through calls with bound arguments, never by falling through -- a
    # fall-through scan re-records every inner site with an empty env.
    out: list[ast.Call] = []

    def inner(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(child, ast.Call):
                out.append(child)
            inner(child)

    inner(scope)
    return out


def reachable_calls(graph: Graph, roots: list[Reach]) -> tuple[list[tuple[Reach, ast.Call, str]], list[str]]:
    seen: set[tuple[str, int, tuple[tuple[str, str], ...]]] = set()
    calls: list[tuple[Reach, ast.Call, str]] = []
    unresolved: list[str] = []
    frontier = list(roots)
    expansions = 0

    def push(source: Source, scope: ast.AST, condition: str, env: Env) -> None:
        frontier.append(Reach(source, scope, condition, env))

    while frontier:
        if MAX_EXPANSIONS is not None and expansions >= MAX_EXPANSIONS:
            unresolved.append(f"traversal stopped after {MAX_EXPANSIONS} expansions")
            break
        expansions += 1
        here = frontier.pop()
        key = (here.source.path, id(here.scope), tuple(sorted(here.env.names.items())))
        if key in seen:
            continue
        seen.add(key)

        for node in _own_calls(here.scope):
            name = call_name(node)
            condition = _conditions(here.scope, node) or here.condition
            role = _loader_role(node, here.source)
            if role:
                landed = _local_wrapper(graph, here, node)
                if landed is not None:
                    where, inner_def, keep_owner = landed
                    bound = _bound_call(node, inner_def, here.scope, here.source, here.env, keep_owner)
                    owner = here.env.owner if keep_owner else None
                    push(where, inner_def, condition, Env(names=bound, owner=owner))
                    continue
                calls.append((Reach(here.source, here.scope, condition, here.env), node, role))
                continue

            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "self" and here.env.owner is not None:
                    held = here.source.defined.get(f"{here.env.owner.name}.{name}")
                    if isinstance(held, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        bound = _bound_call(node, held, here.scope, here.source, here.env, True)
                        push(here.source, held, condition, Env(names=bound, owner=here.env.owner))
                continue

            target = here.source.defined.get(name)
            if isinstance(target, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bound = _bound_call(node, target, here.scope, here.source, here.env, False)
                push(here.source, target, condition, Env(names=bound))
                continue
            if isinstance(target, ast.ClassDef):
                init = here.source.defined.get(f"{name}.__init__")
                if isinstance(init, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    bound = _bound_call(node, init, here.scope, here.source, here.env, True)
                    push(here.source, init, condition, Env(names=bound, owner=target))
                continue
            module = here.source.imported.get(name)
            if module is None:
                continue
            held = graph.resolve_module(module)
            if held is None:
                held = graph.resolve_module(module.rsplit(".", 1)[0])
            if held is None:
                if not graph.holds_package(module):
                    continue
                unresolved.append(f"{here.source.path}: {name} resolves to no module ({module})")
                continue
            leaf = module.rsplit(".", 1)[-1]
            inner = held.defined.get(leaf) or held.defined.get(name)
            hops = 0
            while inner is None and hops < 3:
                # An __init__ that re-exports: follow where IT imports the
                # name from, at the pin, before calling the symbol missing.
                forwarded = held.imported.get(leaf) or held.imported.get(name)
                if not forwarded:
                    break
                bounce = graph.resolve_module(forwarded) or graph.resolve_module(forwarded.rsplit(".", 1)[0])
                if bounce is None:
                    break
                held = bounce
                leaf = forwarded.rsplit(".", 1)[-1]
                inner = held.defined.get(leaf) or held.defined.get(name)
                hops += 1
            if inner is None:
                unresolved.append(f"{here.source.path}: {name} is not defined in {held.path}")
                continue
            if isinstance(inner, ast.ClassDef):
                init = held.defined.get(f"{inner.name}.__init__")
                if isinstance(init, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    bound = _bound_call(node, init, here.scope, here.source, here.env, True)
                    push(held, init, condition, Env(names=bound, owner=inner))
                continue
            bound = _bound_call(node, inner, here.scope, here.source, here.env, False)
            push(held, inner, condition, Env(names=bound))
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
    for reach, node, role in calls:
        name = call_name(node)
        where = f"{repo}@{commit}:{reach.source.path}:{node.lineno}"

        artifact, how = _site_artifact(node, reach.scope, reach.source, reach.env)
        if artifact:
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
            continue

        reasons: list[str] = []
        for argument in call_arguments(node):
            text, told = _value(argument, reach.scope, reach.source, reach.env)
            if not text:
                reasons.append(told)
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
    for edge in out:
        if edge.discovery_status == UNRESOLVED:
            reason = ruled(str(consumer["id"]), edge)
            if reason:
                edge.discovery_status = NOT_ON_BOUNDARY
                edge.discovery_evidence = f"{edge.discovery_evidence} | ruled: {reason}"[:400]
    return out


def build() -> dict[str, Any]:
    manifest = provenance.load_manifest()
    refs_root = (ROOT.parent / manifest["refs_root"]).resolve()
    upstreams = manifest.get("upstreams", {})
    consumers = manifest.get("consumers", [])

    edges: list[Edge] = []
    for consumer in sorted(consumers, key=lambda one: one["id"]):
        edges.extend(discover(consumer, upstreams, refs_root))
    edges.extend(first_party_edges())
    for key, reason in RULINGS.items():
        who, marker = key
        matched = any(
            one.consumer_id == who and (one.loader_branch == marker or one.discovery_evidence.startswith(marker))
            for one in edges
        )
        if matched:
            continue
        # A stale ruling is a decision about something that no longer
        # exists, which is worse than no ruling: it must go red loudly.
        edges.append(
            Edge(
                consumer_id=key[0],
                family="",
                consumer_repo="",
                consumer_revision="",
                entrypoint="",
                boundary_id="",
                boundary_source_locator="",
                model_variant_id=f"STALE_RULING:{key[1]}",
                model_variant_role="stale_ruling",
                configuration_id=reason[:200],
                loader_branch=key[1],
                activation_condition="",
                loader_source_locator=key[1],
                artifact_role="stale_ruling",
                artifact_logical_identity=f"STALE_RULING:{key[1]}",
                discovery_evidence=f"this ruling matched no finding this run: {reason}"[:300],
                discovery_status=UNRESOLVED,
            )
        )

    def variants(status: str) -> set[tuple[str, str]]:
        return {(one.consumer_id, one.model_variant_id) for one in edges if one.discovery_status == status}

    per_consumer: dict[str, int] = {}
    for one in edges:
        per_consumer[one.consumer_id] = per_consumer.get(one.consumer_id, 0) + 1
    from compat.harness import identity as evidence_identity

    return {
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

    return 0 if not totals["unresolved_variants"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
