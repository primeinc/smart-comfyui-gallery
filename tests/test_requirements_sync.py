"""The three install surfaces (requirements.txt, requirements-ai.txt,
pyproject.toml) all claim to be kept in sync; this binds the claim.

Contract per surface:
  - every [project.dependencies] entry appears UNCOMMENTED in requirements.txt
  - every dependency-group 'ai' entry appears UNCOMMENTED in requirements-ai.txt
  - every dependency-group 'ai-models' entry appears in requirements-ai.txt at
    least as a commented opt-in line (pip users uncomment; uv installs by
    default)
"""

from __future__ import annotations

import os
import re

import pytest

tomllib = pytest.importorskip("tomllib")  # stdlib on 3.11+; the floor is 3.10

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _norm(name: str) -> str:
    """Canonical package name: the part before any version/extras/URL
    markers, lowercased, - and _ folded."""
    base = re.split(r"[<>=!\[@; ]", name.strip(), maxsplit=1)[0]
    return base.lower().replace("_", "-")


def _pyproject():
    with open(os.path.join(_ROOT, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)


def _requirement_lines(filename: str):
    """(uncommented, commented) normalized package names in a requirements
    file; continuation comment lines (leading whitespace + #) are prose,
    not packages, and are ignored."""
    uncommented, commented = set(), set()
    with open(os.path.join(_ROOT, filename), encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                if line[0] in " \t":
                    continue
                candidate = stripped.lstrip("#").strip()
                if candidate and re.match(r"^[A-Za-z0-9_.-]+", candidate):
                    commented.add(_norm(candidate))
            else:
                uncommented.add(_norm(stripped))
    return uncommented, commented


def test_core_requirements_match_project_dependencies():
    declared = {_norm(d) for d in _pyproject()["project"]["dependencies"]}
    uncommented, _ = _requirement_lines("requirements.txt")
    missing = declared - uncommented
    assert not missing, f"requirements.txt lacks pyproject core deps: {sorted(missing)}"


def test_ai_requirements_match_ai_group():
    groups = _pyproject()["dependency-groups"]
    declared = {_norm(d) for d in groups["ai"] if isinstance(d, str)}
    uncommented, _ = _requirement_lines("requirements-ai.txt")
    missing = declared - uncommented
    assert not missing, f"requirements-ai.txt lacks 'ai' group deps: {sorted(missing)}"


def test_ai_models_group_documented_as_opt_in():
    groups = _pyproject()["dependency-groups"]
    declared = {_norm(d) for d in groups["ai-models"] if isinstance(d, str)}
    uncommented, commented = _requirement_lines("requirements-ai.txt")
    missing = declared - uncommented - commented
    assert not missing, (
        f"requirements-ai.txt does not mention 'ai-models' deps even as commented opt-ins: {sorted(missing)}"
    )
