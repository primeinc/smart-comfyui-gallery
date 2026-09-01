from __future__ import annotations

import ast
import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import proc
from compat.harness import population

ROOT: Final[Path] = Path(__file__).resolve().parent.parent


CONTROL_FAILURES: Final[tuple[type[BaseException], ...]] = (
    OSError,
    SyntaxError,
    KeyError,
    TypeError,
    ValueError,
    AttributeError,
    IndexError,
)


@dataclass
class Control:
    name: str
    held: bool
    detail: str

    @property
    def mark(self) -> str:
        return "ok " if self.held else "RED"


def _git(where: Path, *args: str) -> str:
    argv: list[str] = ["git", "-C", str(where), *args]
    _, out, _ = proc.text(argv, timeout=proc.LOCAL_SECONDS)
    return out.strip()


def fixture(refs_root: Path, files: dict[str, str]) -> str:
    where = refs_root / "control" / "repo"
    where.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        target = where / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            handle.write(body)
    _git(where, "init", "-q")
    _git(where, "config", "user.email", "control@example.invalid")
    _git(where, "config", "user.name", "control")
    _git(where, "add", "-A")
    _git(where, "commit", "-q", "-m", "control fixture")
    return _git(where, "rev-parse", "HEAD")


def discovered(root: Path, files: dict[str, str], entrypoint: str) -> list[dict[str, Any]]:
    refs_root = root / "refs"
    commit = fixture(refs_root, files)
    consumer = {
        "id": "control",
        "family": "control",
        "repo": "https://github.com/control/repo.git",
        "commit": commit,
        "entrypoint": entrypoint,
        "boundary": ["control_boundary"],
    }

    population._TREES.clear()
    population._BLOBS.clear()
    return [vars(one) for one in population.discover(consumer, {}, refs_root)]


def _artifacts(edges: list[dict[str, Any]]) -> set[str]:
    return {str(one["artifact_logical_identity"]) for one in edges}


def _by_site(edges: list[dict[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for one in edges:
        out.setdefault(str(one["loader_branch"]), set()).add(str(one["artifact_logical_identity"]))
    return out


def control_deep_call_chain(root: Path) -> Control:
    edges = discovered(
        root,
        {
            "entry.py": "from a import one\n\ndef main():\n    one()\n",
            "a.py": "from b import two\n\ndef one():\n    two()\n",
            "b.py": "from c import three\n\ndef two():\n    three()\n",
            "c.py": "from d import four\n\ndef three():\n    four()\n",
            "d.py": 'from transformers import AutoModel\n\ndef four():\n    AutoModel.from_pretrained("deep/model")\n',
        },
        "entry.py::main",
    )
    found = "deep/model" in _artifacts(edges)
    return Control("A deep_call_chain", found, f"{len(edges)} edge(s); deep/model found={found}")


def control_unreachable_loader(root: Path) -> Control:
    edges = discovered(
        root,
        {
            "entry.py": "from helpers import used\n\ndef main():\n    used()\n",
            "helpers.py": (
                "from transformers import AutoModel\n\n"
                'def used():\n    AutoModel.from_pretrained("org/reached")\n\n'
                'def never_called():\n    AutoModel.from_pretrained("fake/unreachable")\n'
            ),
        },
        "entry.py::main",
    )
    required = {
        str(one["artifact_logical_identity"]) for one in edges if str(one.get("discovery_status")) == "REQUIRED"
    }

    reached = "org/reached" in required
    excluded = "fake/unreachable" not in required
    return Control(
        "B unreachable_loader",
        reached and excluded,
        f"required={sorted(required)}; reachable present={reached}; unreachable excluded={excluded}",
    )


def control_two_loaders_one_file(root: Path) -> Control:
    edges = discovered(
        root,
        {
            "entry.py": (
                "from transformers import AutoModel\n"
                "import torch\n\n"
                "def main():\n"
                '    AutoModel.from_pretrained("org/alpha")\n'
                '    torch.load("beta.pth")\n'
            )
        },
        "entry.py::main",
    )
    fanned = {site: sorted(names) for site, names in _by_site(edges).items() if len(names) > 1}
    return Control(
        "C two_loaders_one_file",
        not fanned,
        "no site carries two artifacts" if not fanned else f"FAN-OUT {fanned}",
    )


def control_parameter_default(root: Path) -> Control:
    edges = discovered(
        root,
        {
            "entry.py": (
                "from transformers import AutoModel\n\n"
                'def main(model="facebook/sam3"):\n'
                "    AutoModel.from_pretrained(model)\n"
            )
        },
        "entry.py::main",
    )
    found = "facebook/sam3" in _artifacts(edges)

    moved = discovered(
        root,
        {
            "entry.py": (
                "from transformers import AutoModel\n\n"
                'def main(model="other/model"):\n'
                "    AutoModel.from_pretrained(model)\n"
            )
        },
        "entry.py::main",
    )
    followed = "other/model" in _artifacts(moved) and "facebook/sam3" not in _artifacts(moved)
    return Control(
        "D parameter_default",
        found and followed,
        f"bound={sorted(_artifacts(edges))}; after changing the default={sorted(_artifacts(moved))}",
    )


def control_unrelated_slash_string(root: Path) -> Control:
    edges = discovered(
        root,
        {
            "entry.py": (
                "from transformers import AutoModel\n"
                "import torch\n\n"
                "def main():\n"
                '    output_dir = "images/output"\n'
                "    torch.load(output_dir)\n"
                '    AutoModel.from_pretrained("org/real")\n'
            )
        },
        "entry.py::main",
    )
    names = _artifacts(edges)
    return Control("E unrelated_slash_string", "images/output" not in names, f"artifacts={sorted(names)}")


def control_unresolved_dispatch(root: Path) -> Control:
    edges = discovered(
        root,
        {
            "entry.py": (
                "from transformers import AutoModel\n"
                "import config\n\n"
                "def main():\n"
                "    AutoModel.from_pretrained(config.model_name)\n"
            ),
            "config.py": "model_name = None\n",
        },
        "entry.py::main",
    )
    variants = {str(one["model_variant_id"]) for one in edges}
    fake_default = any(one.endswith(":DEFAULT") for one in variants)
    marked = any("UNRESOLVED" in one for one in variants)
    return Control("F unresolved_dispatch", marked and not fake_default, f"variants={sorted(variants)}")


def control_relative_import(root: Path) -> Control:
    edges = discovered(
        root,
        {
            "pkg/__init__.py": "",
            "pkg/entry.py": "from .helpers import used\n\ndef main():\n    used()\n",
            "pkg/helpers.py": (
                'from transformers import AutoModel\n\ndef used():\n    AutoModel.from_pretrained("org/relative")\n'
            ),
        },
        "pkg/entry.py::main",
    )
    found = "org/relative" in _artifacts(edges)
    return Control("G relative_import", found, f"artifacts={sorted(_artifacts(edges))}")


def control_src_layout(root: Path) -> Control:
    edges = discovered(
        root,
        {
            "entry.py": "from pkg.inner import used\n\ndef main():\n    used()\n",
            "src/pkg/__init__.py": "",
            "src/pkg/inner.py": (
                'from transformers import AutoModel\n\ndef used():\n    AutoModel.from_pretrained("org/srclayout")\n'
            ),
        },
        "entry.py::main",
    )
    found = "org/srclayout" in _artifacts(edges)
    return Control("H src_layout", found, f"artifacts={sorted(_artifacts(edges))}")


def control_shell_python_module(root: Path) -> Control:
    edges = discovered(
        root,
        {
            "scripts/run.sh": "#!/usr/bin/env bash\npython -m pkg.step --flag 1\n",
            "pkg/__init__.py": "",
            "pkg/step.py": (
                "from transformers import AutoModel\n\n"
                'def build():\n    AutoModel.from_pretrained("org/fromshell")\n\n'
                "build()\n"
            ),
        },
        "scripts/run.sh::run",
    )
    found = "org/fromshell" in _artifacts(edges)
    return Control("I shell_python_module", found, f"artifacts={sorted(_artifacts(edges))}")


def control_imported_alias(root: Path) -> Control:
    edges = discovered(
        root,
        {
            "entry.py": "from helpers import used as renamed\n\ndef main():\n    renamed()\n",
            "helpers.py": (
                'from transformers import AutoModel\n\ndef used():\n    AutoModel.from_pretrained("org/aliased")\n'
            ),
        },
        "entry.py::main",
    )
    found = "org/aliased" in _artifacts(edges)
    return Control("K imported_alias", found, f"artifacts={sorted(_artifacts(edges))}")


def control_branch_specific_selection(root: Path) -> Control:
    edges = discovered(
        root,
        {
            "entry.py": (
                "from transformers import AutoModel\n\n"
                "def main(use_big):\n"
                "    if use_big:\n"
                '        AutoModel.from_pretrained("org/big")\n'
                "    else:\n"
                '        AutoModel.from_pretrained("org/small")\n'
            )
        },
        "entry.py::main",
    )
    names = _artifacts(edges)
    both = {"org/big", "org/small"} <= names
    conditional = {
        str(one["artifact_logical_identity"]) for one in edges if str(one.get("discovery_status")) == "CONDITIONAL"
    }
    guarded = {"org/big", "org/small"} <= conditional
    return Control(
        "L branch_specific_selection",
        both and guarded,
        f"both found={both}; both CONDITIONAL={guarded}; conditional={sorted(conditional)}",
    )


def control_pinned_git_submodule(root: Path) -> Control:
    refs_root = root / "refs"

    inner = refs_root / "control" / "inner"
    inner.mkdir(parents=True, exist_ok=True)
    (inner / "deep.py").write_text(
        'from transformers import AutoModel\n\ndef used():\n    AutoModel.from_pretrained("org/insubmodule")\n',
        encoding="utf-8",
        newline="",
    )
    _git(inner, "init", "-q")
    _git(inner, "config", "user.email", "control@example.invalid")
    _git(inner, "config", "user.name", "control")
    _git(inner, "add", "-A")
    _git(inner, "commit", "-q", "-m", "pinned state")
    pinned = _git(inner, "rev-parse", "HEAD")

    (inner / "deep.py").write_text(
        'from transformers import AutoModel\n\ndef used():\n    AutoModel.from_pretrained("org/moved")\n',
        encoding="utf-8",
        newline="",
    )
    _git(inner, "add", "-A")
    _git(inner, "commit", "-q", "-m", "moved on")

    where = refs_root / "control" / "repo"
    where.mkdir(parents=True, exist_ok=True)
    (where / "entry.py").write_text(
        "from vendored.deep import used\n\ndef main():\n    used()\n",
        encoding="utf-8",
        newline="",
    )
    _git(where, "init", "-q")
    _git(where, "config", "user.email", "control@example.invalid")
    _git(where, "config", "user.name", "control")
    _git(where, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(inner), "vendored")
    _git(where, "-C", "vendored", "checkout", "-q", pinned)
    _git(where, "add", "-A")
    _git(where, "commit", "-q", "-m", "parent pinning the submodule")
    commit = _git(where, "rev-parse", "HEAD")

    consumer = {
        "id": "control",
        "family": "control",
        "repo": "https://github.com/control/repo.git",
        "commit": commit,
        "entrypoint": "entry.py::main",
        "boundary": ["control_boundary"],
    }
    population._TREES.clear()
    population._BLOBS.clear()
    edges = [vars(one) for one in population.discover(consumer, {}, refs_root)]

    names = _artifacts(edges)
    crossed = "org/insubmodule" in names
    at_pin = "org/moved" not in names
    return Control(
        "J pinned_git_submodule",
        crossed and at_pin,
        f"crossed the gitlink={crossed}; read the PIN not HEAD={at_pin}; artifacts={sorted(names)}",
    )


CONTROLS: Final[tuple[tuple[str, Callable[[Path], Control]], ...]] = (
    ("A deep_call_chain", control_deep_call_chain),
    ("B unreachable_loader", control_unreachable_loader),
    ("C two_loaders_one_file", control_two_loaders_one_file),
    ("D parameter_default", control_parameter_default),
    ("E unrelated_slash_string", control_unrelated_slash_string),
    ("F unresolved_dispatch", control_unresolved_dispatch),
    ("G relative_import", control_relative_import),
    ("H src_layout", control_src_layout),
    ("I shell_python_module", control_shell_python_module),
    ("J pinned_git_submodule", control_pinned_git_submodule),
    ("K imported_alias", control_imported_alias),
    ("L branch_specific_selection", control_branch_specific_selection),
)


def run_all() -> list[Control]:
    out: list[Control] = []
    for label, build in CONTROLS:
        with tempfile.TemporaryDirectory(prefix="population_control_") as raw:
            try:
                out.append(build(Path(raw)))
            except CONTROL_FAILURES as problem:
                out.append(Control(label, False, f"{type(problem).__name__}: {problem}"))
    return out


@dataclass
class Mutation:
    name: str
    control: str
    red_under_mutation: bool
    green_after_restore: bool
    detail: str

    @property
    def ok(self) -> bool:
        return self.red_under_mutation and self.green_after_restore

    @property
    def mark(self) -> str:
        return "ok " if self.ok else "RED"


def _run(control: Callable[[Path], Control]) -> Control:
    with tempfile.TemporaryDirectory(prefix="population_mutation_") as raw:
        try:
            return control(Path(raw))
        except CONTROL_FAILURES as problem:
            return Control("mutated", False, f"{type(problem).__name__}: {problem}")


def _file_wide_arguments(node: ast.Call) -> list[ast.expr]:
    del node
    return list(_MODULE_STRINGS)


def _slash_is_a_model(text: str, loader: str) -> bool:
    del loader
    return text.lower().endswith(population.ARTIFACT_SUFFIX) or text.count("/") == 1


def _collapse_to_default(where: str) -> str:
    del where
    return "from_pretrained:DEFAULT"


_MODULE_STRINGS: list[ast.expr] = []


def mutation_depth_cutoff() -> Mutation:
    population.MAX_EXPANSIONS = 2
    try:
        under = _run(control_deep_call_chain)
    finally:
        population.MAX_EXPANSIONS = None
    after = _run(control_deep_call_chain)
    return Mutation(
        "depth_cutoff",
        "A deep_call_chain",
        not under.held,
        after.held,
        f"under mutation: {under.detail[:70]}",
    )


def mutation_file_wide_literals() -> Mutation:
    original = population.call_arguments
    _MODULE_STRINGS.clear()
    _MODULE_STRINGS.extend([ast.Constant(value="org/alpha"), ast.Constant(value="beta.pth")])
    population.call_arguments = _file_wide_arguments
    try:
        under = _run(control_two_loaders_one_file)
    finally:
        population.call_arguments = original
    after = _run(control_two_loaders_one_file)
    return Mutation(
        "file_wide_literals",
        "C two_loaders_one_file",
        not under.held,
        after.held,
        f"under mutation: {under.detail[:70]}",
    )


def mutation_slash_shape_hub() -> Mutation:
    original = population.is_artifact
    population.is_artifact = _slash_is_a_model
    try:
        under = _run(control_unrelated_slash_string)
    finally:
        population.is_artifact = original
    after = _run(control_unrelated_slash_string)
    return Mutation(
        "slash_shape_hub",
        "E unrelated_slash_string",
        not under.held,
        after.held,
        f"under mutation: {under.detail[:70]}",
    )


def mutation_unresolved_becomes_default() -> Mutation:
    original = population.unresolved_variant
    population.unresolved_variant = _collapse_to_default
    try:
        under = _run(control_unresolved_dispatch)
    finally:
        population.unresolved_variant = original
    after = _run(control_unresolved_dispatch)
    return Mutation(
        "unresolved_becomes_default",
        "F unresolved_dispatch",
        not under.held,
        after.held,
        f"under mutation: {under.detail[:70]}",
    )


MUTATIONS: Final[tuple[Callable[[], Mutation], ...]] = (
    mutation_depth_cutoff,
    mutation_file_wide_literals,
    mutation_slash_shape_hub,
    mutation_unresolved_becomes_default,
)


def run_mutations() -> list[Mutation]:
    return [build() for build in MUTATIONS]


def main() -> int:
    held = run_all()
    print("static discovery controls\n")
    for one in held:
        print(f"{one.mark} {one.name:<26} {one.detail[:110]}")

    failed = [one.name for one in held if not one.held]
    print(f"\n{len(held)} controls, {len(failed)} failing: {failed or 'none'}")

    print("\nnegative controls over the discovery mechanism\n")
    mutated = run_mutations()
    for one in mutated:
        print(f"{one.mark} {one.name:<28} {one.control:<24} red={one.red_under_mutation}")
        print(f"       restored green={one.green_after_restore}; {one.detail[:90]}")
    missed = [one.name for one in mutated if not one.ok]
    print(f"\n{len(mutated)} mutations, {len(missed)} the controls did not catch: {missed or 'none'}")

    broken = [*failed, *missed]
    if broken:
        print("static discovery is UNTRUSTED; its counts are candidates, not a population")

    generated = ROOT / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    target = generated / "population_controls.json"
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(
            json.dumps(
                {
                    "controls": [vars(one) for one in held],
                    "mutations": [vars(one) for one in mutated],
                    "failing": broken,
                },
                indent=2,
                sort_keys=True,
            )
        )
        handle.write("\n")
    print(f"wrote {target}")
    return 0 if not broken else 1


if __name__ == "__main__":
    raise SystemExit(main())
