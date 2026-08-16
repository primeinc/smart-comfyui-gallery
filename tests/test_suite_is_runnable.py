"""The suite has to run the way it is documented to run.

This repo is an application, not an installed package -- pyproject sets
`package = false`, so nothing ever puts the repo root on sys.path. The
suite's conftest imports the monolith by name to build its fixture, which
means a test only runs if something else supplied that path.

`python -m pytest` supplies it by accident: the `-m` form prepends the
working directory. Nothing else does. So `uv run pytest tests/` -- the
command pyproject.toml's own header gives as the way to run the suite --
and a bare `pytest` in an activated venv both failed on every single test
with `ModuleNotFoundError: No module named 'smartgallery'`. Someone
following the documentation got a wall of errors and no way to tell
whether they had broken something or merely typed the documented command.

pytest.ini now sets `pythonpath = .`, which pytest resolves against the
rootdir and prepends to sys.path for the session (docs/en/reference/
reference.rst, confval pythonpath). These tests spawn the console script,
because that is the form that failed and it cannot be exercised from
inside a pytest process that is already running.

The subprocess runs the probe at the bottom of this file, selected by node
id so it cannot re-enter the tests that spawn it. Collection alone proves
nothing: conftest imports the monolith inside a fixture, so a
`--collect-only` run passes whether the path is right or not.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_THIS_FILE = pathlib.Path(__file__).resolve()
_PROBE = f"{_THIS_FILE}::test_the_monolith_imports_in_the_subprocess"
_IN_SUBPROCESS = "SMARTGALLERY_RUNNABILITY_PROBE"


def _console_script():
    """The `pytest` entry point, installed beside the interpreter."""
    bin_dir = pathlib.Path(sys.executable).parent
    for name in ("pytest.exe", "pytest"):
        candidate = bin_dir / name
        if candidate.exists():
            return candidate
    return None


def _run(extra_args, cwd):
    script = _console_script()
    if script is None:
        pytest.skip("no pytest console script beside "
                    f"{sys.executable}; the failing form does not exist here")

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)  # never let an inherited path do the work
    env[_IN_SUBPROCESS] = "1"
    return subprocess.run([str(script), _PROBE, *extra_args, "-q"],
                          cwd=str(cwd), env=env, capture_output=True,
                          text=True, timeout=600)


def test_the_documented_command_works():
    """`uv run pytest tests/` reduces to this: the console script, from the
    repo root. It errored on every test in the suite before pytest.ini
    carried the setting."""
    done = _run([], cwd=_REPO_ROOT)
    output = done.stdout + done.stderr

    assert "ModuleNotFoundError" not in output, output
    assert done.returncode == 0, output


def test_it_works_from_a_subdirectory_too():
    """rootdir is found by walking up to pytest.ini, so the directory that
    gets added is the repo root rather than wherever the person was
    standing when they ran it."""
    done = _run([], cwd=_REPO_ROOT / "tests")
    output = done.stdout + done.stderr

    assert "ModuleNotFoundError" not in output, output
    assert done.returncode == 0, output


def test_the_setting_is_what_makes_it_work():
    """Control. Without this, the two tests above could be passing because
    the subprocess inherited a path from somewhere, and this guard would go
    on passing after someone deleted the setting.

    Emptying that one ini option -- and changing nothing else -- has to
    reproduce the original failure exactly."""
    done = _run(["-o", "pythonpath="], cwd=_REPO_ROOT)
    output = done.stdout + done.stderr

    assert done.returncode != 0, f"expected the failure, got a clean run:\n{output}"
    assert "No module named 'smartgallery'" in output, output


@pytest.mark.skipif(os.environ.get(_IN_SUBPROCESS) != "1",
                    reason="the probe the three tests above spawn; "
                           "meaningless on its own")
def test_the_monolith_imports_in_the_subprocess(smartgallery_app):
    """Requesting the fixture is the whole point: that is where conftest
    imports the monolith, and that is the line every test in the suite
    used to die on."""
    assert smartgallery_app.__name__ == "smartgallery"
