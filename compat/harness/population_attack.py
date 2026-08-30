"""Attack static discovery itself, before any count it produces is believed.

`population.py` reported 102 variants over 947 edges, and 678 of those edges
were manufactured: it attached every artifact literal found anywhere in a FILE
to every loader call in that file, so one call site in `eva_clip/pretrained.py`
was credited with fourteen different artifacts. A larger population is not a
better proof, and an oracle that can invent an edge may not decide what must be
proved.

Six controls, each a real git repository built and committed here, discovered
through the same entrypoint-rooted path production uses:

    A deep_call_chain         a loader four calls down MUST be found
    B unreachable_loader      an imported but never-called loader must NOT be required
    C two_loaders_one_file    each loader binds ONLY its own artifact
    D parameter_default       a default flowing into the call binds that artifact
    E unrelated_slash_string  "images/output" is NOT a hub repository
    F unresolved_dispatch     `config.model_name` is UNRESOLVED, never DEFAULT

Every control states what the population MUST contain AND what it must not. A
control asserting only presence would pass an algorithm that returns
everything, which is the defect under attack.
"""

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

#: What a control may raise without the harness itself being broken. Bare
#: `Exception` would need a linter suppression, and this tree bans them; naming
#: the failures keeps a defect in the control from reading as a finding.
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
    """One git call in the fixture repository, argv bound before the call.

    The list is a name rather than a literal at the call site, which is the
    shape `provenance.py` and `citations.py` already use.
    """
    argv: list[str] = ["git", "-C", str(where), *args]
    _, out, _ = proc.text(argv, timeout=proc.LOCAL_SECONDS)
    return out.strip()


def fixture(refs_root: Path, files: dict[str, str]) -> str:
    """A real committed repository at `refs_root/control/repo`, and its sha.

    Real, not simulated: discovery reads `git cat-file blob <sha>:<path>`, so a
    fixture that was only a directory would exercise a path production never
    takes. Built where `provenance.clone_dir` will look for
    `https://github.com/control/repo.git`.
    """
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
    """Run production discovery over one fixture and return its edges."""
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
    # The caches are keyed on (clone, commit) and every fixture reuses one
    # path, so a stale entry would answer for the previous control.
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
    """A loader four calls down must be found. This kills the DEPTH=2 bug."""
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
    """An imported but never-called loader must NOT be required."""
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
    # NON-VACUOUS: `required=[]` satisfies "the unreachable one is absent"
    # against an algorithm that finds nothing, so the reachable loader must be
    # REQUIRED here for the absence to mean anything.
    reached = "org/reached" in required
    excluded = "fake/unreachable" not in required
    return Control(
        "B unreachable_loader",
        reached and excluded,
        f"required={sorted(required)}; reachable present={reached}; unreachable excluded={excluded}",
    )


def control_two_loaders_one_file(root: Path) -> Control:
    """Each loader binds only its own artifact. No Cartesian product."""
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
    """A parameter default flowing into the call binds that artifact."""
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
    # PROVE THE MECHANISM, not the presence. The same fixture with a different
    # default must produce a different edge; if the artifact does not follow
    # the default, it was not bound by dataflow.
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
    """`images/output` is a path, not a hub repository."""
    edges = discovered(
        root,
        {
            # The slash string is PASSED TO A LOADER, so the hub rule decides
            # it: `torch.load` is not a hub API, which makes `images/output` a
            # path under every correct rule and a model only under guessing.
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
    """An unresolved selection is UNRESOLVED, never DEFAULT."""
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
    """`from .helpers import used` must resolve against the importing package.

    Read as a top-level `helpers` it resolves nowhere, and ID-V2V's whole ONNX
    estimator stack sat behind four files that appeared to import nothing.
    """
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
    """A module under `src/` must resolve. ID-V2V ships one."""
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
    """A shell entrypoint's `python -m` targets are the real entry roots."""
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
    """`from x import y as z` then `z()` must follow to y."""
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
    """Two mutually exclusive model loaders stay two visible variants.

    Collapsing them to one default is how a whole embedding space disappears.
    Both must be discovered, and both must read CONDITIONAL rather than
    REQUIRED, because neither runs unconditionally.
    """
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
    """A loader inside a gitlink, at the commit the parent pins.

    A submodule's contents are NOT in the parent tree: every path under it
    reads as absent, so a scanner that stops at the boundary reports the
    consumer as loading nothing. UMO reaches its entire pipeline through
    `projects/UNO` this way and discovered zero loaders until the crossing
    existed.

    Two-sided. The parent pins an OLD commit of the inner repository, and the
    inner repository then moves on to a different model. Discovery must read
    the pinned commit -- finding `org/moved` would mean it read HEAD, which is
    the silent-drift failure the pins exist to prevent.
    """
    refs_root = root / "refs"

    # The inner repository, committed twice: the pinned state, then a later
    # state naming a different model.
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

    # The parent, with the inner repository as a real gitlink at `pinned`.
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


#: Paired with a label: a `Callable` carries no `__name__`, so a control that
#: raises could not name itself.
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


# --- negative controls over the DISCOVERY MECHANISM -------------------------

# A control that has never failed is not known to discriminate. Each mutation
# below reintroduces one defect through the seam that encodes that decision,
# and its control must go red and then green again on restore.


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
    """The defect: every string constant in the enclosing MODULE.

    This is what credited one call site in `eva_clip/pretrained.py` with
    fourteen artifacts, and what made 678 of 947 edges fan-out.
    """
    del node
    return list(_MODULE_STRINGS)


def _slash_is_a_model(text: str, loader: str) -> bool:
    """The defect: `org/name` shape alone means a hub repository."""
    del loader
    return text.lower().endswith(population.ARTIFACT_SUFFIX) or text.count("/") == 1


def _collapse_to_default(where: str) -> str:
    """The defect: every unresolved selection becomes one invented default."""
    del where
    return "from_pretrained:DEFAULT"


#: Held module-level so the file-wide mutation has something to hand back
#: without re-parsing; populated by the mutation that installs it.
_MODULE_STRINGS: list[ast.expr] = []


def mutation_depth_cutoff() -> Mutation:
    """Reintroduce `DEPTH = 2`. Control A must stop finding the deep loader."""
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
    """Bind every module string to every loader. Control C must see fan-out."""
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
    """Classify by slash shape. Control E must accept `images/output`."""
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
    """Collapse unresolved to DEFAULT. Control F must accept the fake."""
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
