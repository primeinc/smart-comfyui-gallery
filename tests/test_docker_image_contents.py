"""The Docker image has to contain what the app imports.

The Dockerfile copied `requirements.txt`, `smartgallery.py` and
`templates/` and nothing else. smartgallery.py imports sg_auth,
smartgallery_ai and metaparse at module scope, so the container died on
line 47 with `ModuleNotFoundError: No module named 'sg_auth'` before
printing a single line of its own. Docker is one of the two installation
methods in the README.

omniquery is a second, quieter case: it is imported inside the OmniQuery
request handlers rather than at the top of the file, so an image without it
starts perfectly and then returns 500 from Search in Plain English.

Both shapes are covered by one check: assemble the file set the Dockerfile
copies, then read every import in every file of it. Module-scope and
request-time imports look the same to a reader, which is the point --
booting the assembly would only have proved the first kind.

Nothing here needs docker installed, and nothing starts a process.
requirements.txt alone is enough to import the app -- verified against a
venv built from it -- so the AI layer being absent from the image is not
the problem here.
"""

from __future__ import annotations

import ast
import pathlib
import re
import shutil

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_DOCKERFILE = _REPO_ROOT / "Dockerfile"

# What the Makefile passes for the build args. The experiments variant
# swaps these two for its own copies; everything else comes from the root.
_BUILD_ARGS = {
    "CHOOSEN_SMARTGALLERY_FILE": "smartgallery.py",
    "CHOOSEN_TEMPLATE_DIR": "templates",
}

_COPY = re.compile(r"^COPY\s+(?:--\S+\s+)*(\S+)\s+(\S+)\s*$", re.MULTILINE)

# Directories that are not part of the running application.
_NOT_SHIPPED = {"tests", "probes", "benchmarks", "experiments", "docs",
                "vendor", "assets", "templates"}


def _expand(value):
    for name, replacement in _BUILD_ARGS.items():
        value = value.replace(f"${{{name}}}", replacement).replace(f"${name}", replacement)
    return value


def _copies():
    """(source, destination) pairs the Dockerfile places under /app."""
    pairs = []
    for source, dest in _COPY.findall(_DOCKERFILE.read_text(encoding="utf-8")):
        source, dest = _expand(source), _expand(dest)
        if dest.startswith("/app/"):
            pairs.append((source.rstrip("/"), dest))
    return pairs


def _dockerignore_patterns():
    path = _REPO_ROOT / ".dockerignore"
    assert path.exists(), (
        "no .dockerignore: the whole working tree, weights and virtualenv "
        "included, is sent to the daemon on every build")
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")}


def _first_party_names():
    """Top-level modules and packages that belong to this repo."""
    names = set()
    for entry in _REPO_ROOT.iterdir():
        if entry.name.startswith((".", "_")) or entry.name in _NOT_SHIPPED:
            continue
        if entry.is_file() and entry.suffix == ".py":
            names.add(entry.stem)
        elif entry.is_dir() and (entry / "__init__.py").exists():
            names.add(entry.name)
    return names


@pytest.fixture(scope="module")
def assembled_image(tmp_path_factory):
    """A directory holding exactly what the Dockerfile puts in /app."""
    app = tmp_path_factory.mktemp("image") / "app"
    app.mkdir()
    for source, dest in _copies():
        origin = _REPO_ROOT / source
        assert origin.exists(), f"Dockerfile copies {source}, which is not in the repo"
        target = app / dest[len("/app/"):].rstrip("/")
        if origin.is_dir():
            shutil.copytree(origin, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
    return app


def test_the_dockerfile_copies_something_recognisable():
    """Control. A regex that matched nothing would make every check below
    pass against an empty file list."""
    sources = {source for source, _dest in _copies()}

    assert "smartgallery.py" in sources, sources
    assert "templates" in sources, sources
    assert len(sources) >= 3, sources


def test_first_party_discovery_finds_the_real_modules():
    """Control for the scan below: if this returned nothing, every import
    would look like a third-party one and the check would be vacuous."""
    names = _first_party_names()

    assert {"sg_auth", "smartgallery_ai", "metaparse", "omniquery"} <= names, names
    assert "tests" not in names and "pytest" not in names, names


def _first_party_imports_under(root):
    """Every first-party module name imported anywhere beneath `root`.

    Returns {name: [file, ...]} so a failure can say which file wanted it.
    """
    first_party = _first_party_names()
    wanted = {}
    for source in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"))
        except SyntaxError:  # not ours to police here
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module or ""]
            else:
                continue
            for dotted in names:
                head = dotted.split(".")[0]
                if head in first_party:
                    wanted.setdefault(head, []).append(
                        str(source.relative_to(root)))
    return wanted


def _modules_missing_from(root):
    """First-party modules the image imports but does not contain.

    {name: [file that imports it, ...]}; empty means the image is complete.
    """
    missing = {}
    for name, wanted_by in _first_party_imports_under(root).items():
        if (root / f"{name}.py").exists() or (root / name / "__init__.py").exists():
            continue
        missing[name] = sorted(set(wanted_by))
    return missing


def test_the_image_carries_every_module_it_imports(assembled_image):
    """The failure as it happened: the container exited at `import sg_auth`.

    Read off the assembled directory rather than by booting it. Booting
    needed a child interpreter with an emptied PATH, and only ever proved
    the module-scope imports -- the ones that run before the first request.
    Reading every import in every file it copies covers those AND the lazy
    ones, which is the case the file's own docstring says booting cannot
    reach.
    """
    assert _first_party_imports_under(assembled_image), (
        "no first-party imports found in the assembled image at all")

    missing = _modules_missing_from(assembled_image)

    assert not missing, (
        f"the image imports {sorted(missing)} but the Dockerfile never "
        f"copies them: {missing}. The container fails -- at startup for a "
        f"module-scope import, or on the request that reaches a lazy one.")


def test_a_missing_module_would_actually_be_noticed(assembled_image):
    """Control. Without it the check above could be passing because it
    finds nothing to look for, and would keep passing with the Dockerfile
    broken."""
    present = assembled_image / "sg_auth.py"
    assert present.exists(), "the assembly no longer includes sg_auth"

    hidden = assembled_image / "sg_auth.py.hidden"
    present.rename(hidden)
    try:
        missing = _modules_missing_from(assembled_image)
    finally:
        hidden.rename(present)

    assert "sg_auth" in missing, (
        "sg_auth was taken out of the image and the check still reported it "
        f"complete (it found {sorted(missing)}), so a Dockerfile that stopped "
        f"copying a module would pass")
    assert missing["sg_auth"], "the report does not say which file wanted it"


def test_every_first_party_import_is_in_the_image():
    """omniquery's case: imported inside a request handler, so the image
    boots without it and only fails when someone searches. Booting proves
    nothing about these, which is why they are checked by name."""
    text = (_REPO_ROOT / "smartgallery.py").read_text(encoding="utf-8")
    copied = {source for source, _dest in _copies()}

    needed = {name for name in _first_party_names()
              if re.search(rf"^\s*(?:import|from)\s+{re.escape(name)}\b", text,
                           re.MULTILINE)}
    assert needed, "no first-party imports found in smartgallery.py at all"

    missing = {name for name in needed
               if name not in copied and f"{name}.py" not in copied}
    assert not missing, (
        f"smartgallery.py imports {sorted(missing)}, which the Dockerfile "
        f"never copies. It copies {sorted(copied)}. The container will fail "
        f"-- at startup for a module-scope import, or on the request that "
        f"reaches a lazy one.")


def test_the_build_context_excludes_the_heavy_directories():
    """There was no .dockerignore, so `docker build .` shipped the whole
    working tree to the daemon -- about 4.5GB once the AI weights are
    provisioned -- before building an image that copies a few megabytes."""
    ignored = _dockerignore_patterns()

    assert {".git", ".venv", ".AImodels", "vendor"} <= ignored, sorted(ignored)


def test_nothing_the_dockerfile_copies_is_excluded():
    """The risk the file introduces: excluding something a COPY needs turns
    a slow build into a failed one."""
    ignored = _dockerignore_patterns()
    assert not any(p.startswith("!") for p in ignored), (
        "a re-include appeared; this check only understands plain excludes")

    for source, _dest in _copies():
        head = source.split("/")[0]
        assert head not in ignored, (
            f"the Dockerfile copies {source}, but .dockerignore excludes "
            f"{head}, so the build cannot see it")

    assert "experiments" not in ignored, (
        "make build_exp copies experiments/, so it has to stay in the context")
