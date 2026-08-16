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

Both shapes are covered here, because they need different tests:

  * assembling the image's file set and importing it catches anything that
    stops the container booting, including things no import scan would
    predict;
  * scanning for first-party imports catches the ones that only fire on a
    request, which a successful boot says nothing about.

Neither needs docker installed. requirements.txt alone is enough to import
the app -- verified against a venv built from it -- so the AI layer being
absent from the image is not the problem here.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys

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


def test_the_image_can_import_the_app(assembled_image, tmp_path):
    """The failure as it happened: the container exited at `import sg_auth`.

    Run out of the assembled directory, so only what the Dockerfile copies
    is importable -- the repo itself is never on the path."""
    gallery = tmp_path / "gallery"
    for sub in (".sqlite_cache", ".thumbnails_cache", ".zip_downloads"):
        (gallery / sub).mkdir(parents=True)
    for name in ("output", "input"):
        (tmp_path / name).mkdir()

    done = subprocess.run(
        [sys.executable, "-c", "import smartgallery; print(smartgallery.APP_VERSION)"],
        cwd=str(assembled_image), capture_output=True, text=True, timeout=600,
        env={"PATH": "", "SYSTEMROOT": "", "BASE_OUTPUT_PATH": str(tmp_path / "output"),
             "BASE_INPUT_PATH": str(tmp_path / "input"),
             "BASE_SMARTGALLERY_PATH": str(gallery),
             "ENABLE_AI_DAM": "false", "AI_DAM_AUTO_PROVISION": "false"})

    assert "ModuleNotFoundError" not in done.stderr, done.stderr
    assert done.returncode == 0, done.stderr


def test_a_missing_module_is_actually_noticed(assembled_image, tmp_path):
    """Control for the test above. Without it that one could be passing
    because the subprocess found the repo on its path rather than the
    assembled directory, and would keep passing with the Dockerfile broken."""
    hidden = assembled_image / "sg_auth.py"
    assert hidden.exists(), "the assembly no longer includes sg_auth"
    hidden.rename(assembled_image / "sg_auth.py.hidden")
    try:
        done = subprocess.run(
            [sys.executable, "-c", "import smartgallery"],
            cwd=str(assembled_image), capture_output=True, text=True, timeout=600,
            env={"PATH": "", "SYSTEMROOT": "",
                 "BASE_OUTPUT_PATH": str(tmp_path), "BASE_INPUT_PATH": str(tmp_path),
                 "BASE_SMARTGALLERY_PATH": str(tmp_path),
                 "ENABLE_AI_DAM": "false", "AI_DAM_AUTO_PROVISION": "false"})
    finally:
        (assembled_image / "sg_auth.py.hidden").rename(hidden)

    assert done.returncode != 0, done.stdout
    assert "No module named 'sg_auth'" in done.stderr, done.stderr


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
