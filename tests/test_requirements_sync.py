"""The two install surfaces (requirements.txt, pyproject.toml) claim to
be kept in sync; this binds the claim down to the version specifiers and
environment markers, not names alone -- a bare `transformers` line in
requirements.txt satisfied a name-only check while legally installing a
release without the qwen3_vl family the adapter subclasses. The AI layer
is core: there is no optional dependency group and no second
requirements file, and the tests below keep that fiction from creeping
back.
"""

from __future__ import annotations

import os
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _shape(requirement: Requirement) -> tuple[str, str]:
    """What must agree between the two surfaces: the version constraint
    and the environment marker, canonically spelled."""
    return str(requirement.specifier), str(requirement.marker) if requirement.marker else ""


def _pyproject():
    with open(os.path.join(_ROOT, "pyproject.toml"), "rb") as fh:
        return tomllib.load(fh)


def _requirement_lines(filename: str) -> dict[str, tuple[str, str]]:
    """Uncommented requirements in a requirements file, name -> shape.
    Inline comments are prose; the requirement is what pip would see."""
    held: dict[str, tuple[str, str]] = {}
    with open(os.path.join(_ROOT, filename), encoding="utf-8") as fh:
        for line in fh:
            bare = line.split("#", 1)[0].strip()
            if bare:
                requirement = Requirement(bare)
                held[canonicalize_name(requirement.name)] = _shape(requirement)
    return held


def test_requirements_match_project_dependencies_exactly():
    declared = {}
    for entry in _pyproject()["project"]["dependencies"]:
        requirement = Requirement(entry)
        declared[canonicalize_name(requirement.name)] = _shape(requirement)
    listed = _requirement_lines("requirements.txt")

    missing = set(declared) - set(listed)
    assert not missing, f"requirements.txt lacks pyproject deps: {sorted(missing)}"
    for name, shape in declared.items():
        assert listed[name] == shape, (
            f"{name}: pyproject declares specifier/marker {shape}, requirements.txt says {listed[name]}"
        )


def test_the_ai_layer_is_core_not_a_group():
    """AI is the product. A dependency group or a second requirements file
    would make it look optional again."""
    groups = _pyproject().get("dependency-groups", {})
    assert set(groups) <= {"dev"}, f"unexpected dependency groups: {sorted(set(groups) - {'dev'})}"
    assert not os.path.exists(os.path.join(_ROOT, "requirements-ai.txt")), (
        "requirements-ai.txt is back; the AI layer is core and lives in requirements.txt"
    )
