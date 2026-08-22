"""The suite has to run the way it is documented to run.

This repo is an application, not an installed package -- pyproject sets
`package = false`, so nothing ever puts the repo root on sys.path. The
suite's conftest imports the monolith by name to build its fixture, which
means a test only runs if something else supplied that path.

`python -m pytest` supplies it by accident: the `-m` form prepends the
working directory. Nothing else does. So `uv run pytest tests/` -- the
command pyproject.toml's own header gives as the way to run the suite --
and a bare `pytest` in an activated venv both failed on every single test
with `ModuleNotFoundError`. Someone
following the documentation got a wall of errors and no way to tell
whether they had broken something or merely typed the documented command.

pytest.ini now sets `pythonpath = .`, which pytest resolves against the
rootdir and prepends to sys.path for the session (docs/en/reference/
reference.rst, confval pythonpath).

This used to prove that by launching the console script three times --
three pytest processes, each loading the whole suite -- because the broken
form could not be reproduced from inside a pytest run. What it was really
protecting is one ini setting and where the rootdir lands, and both are
readable from the running session through `pytestconfig`. The end-to-end
launch is gone; the setting it depended on is now asserted directly, along
with the two things that made it load-bearing: that the root is genuinely
on sys.path, and that the monolith is genuinely importable from there.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

#: Repository hygiene, not gallery behaviour: `just repo-check` runs these.
pytestmark = pytest.mark.repo

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_the_setting_that_makes_the_documented_command_work(pytestconfig):
    """The one line in pytest.ini that every invocation depends on."""
    configured = pytestconfig.getini("pythonpath")

    assert configured, (
        "pytest.ini no longer sets `pythonpath`. Without it only "
        "`python -m pytest` works, because that form prepends the working "
        "directory by accident; `uv run pytest` and a bare `pytest` fail on "
        "the first `import db`."
    )
    # pytest resolves the setting against the rootdir, so what comes back is
    # already absolute -- which is the thing worth asserting: not that the
    # file says ".", but that "." lands on this repo.
    resolved = {pathlib.Path(entry).resolve() for entry in configured}
    assert _REPO_ROOT in resolved, (
        f"pythonpath resolves to {sorted(map(str, resolved))} rather than the repo root {_REPO_ROOT}"
    )


def test_the_rootdir_is_the_repo_not_wherever_you_were_standing(pytestconfig):
    """pythonpath is resolved against the rootdir, and the rootdir is found
    by walking up to pytest.ini -- so the directory that gets added is the
    repo root whatever directory the command was typed in."""
    assert pathlib.Path(pytestconfig.rootpath).resolve() == _REPO_ROOT, (
        f"rootdir resolved to {pytestconfig.rootpath}, so `pythonpath = .` would add the wrong directory"
    )


def test_the_repo_root_really_is_on_the_path():
    """The setting is only worth anything if it took effect. This asserts
    the consequence rather than the configuration."""
    on_path = {pathlib.Path(entry).resolve() for entry in sys.path if entry}

    assert _REPO_ROOT in on_path, (
        f"{_REPO_ROOT} is not on sys.path, so `import db` works here only by whatever accident supplied it"
    )


def test_the_application_imports_from_there():
    """The setting exists so the application's packages import from this
    repo. Asserting the consequence, on the package every test needs."""
    import db

    assert pathlib.Path(db.__file__).resolve().parent.parent == _REPO_ROOT, (
        "the db package was imported from somewhere other than this repo"
    )
