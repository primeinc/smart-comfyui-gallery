"""The two install surfaces (requirements.txt, pyproject.toml) claim to
be kept in sync; this binds the claim. The AI layer is core -- there is no
optional dependency group and no second requirements file, and the tests
below keep that fiction from creeping back.
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
    """Uncommented normalized package names in a requirements file."""
    uncommented = set()
    with open(os.path.join(_ROOT, filename), encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                uncommented.add(_norm(stripped))
    return uncommented


def test_requirements_match_project_dependencies():
    declared = {_norm(d) for d in _pyproject()["project"]["dependencies"]}
    uncommented = _requirement_lines("requirements.txt")
    missing = declared - uncommented
    assert not missing, f"requirements.txt lacks pyproject deps: {sorted(missing)}"


def test_the_ai_layer_is_core_not_a_group():
    """AI is the product. A dependency group or a second requirements file
    would make it look optional again."""
    groups = _pyproject().get("dependency-groups", {})
    assert set(groups) <= {"dev"}, f"unexpected dependency groups: {sorted(set(groups) - {'dev'})}"
    assert not os.path.exists(os.path.join(_ROOT, "requirements-ai.txt")), (
        "requirements-ai.txt is back; the AI layer is core and lives in requirements.txt"
    )
