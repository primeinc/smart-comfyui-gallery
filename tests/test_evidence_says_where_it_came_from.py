"""The browser report's provenance stamp is honest about the tree.

The shipped manifest once said `commit: fc56903` for a run produced by a
dirty tree whose driver did not exist at fc56903 -- evidence attributed
to code that cannot produce it. The stamp now carries `-dirty` whenever
the tree differs from HEAD, and these tests hold it to that against a
real scratch repository.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import subprocess
import sys

import pytest


def _driver():
    where = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "browser_report.py"
    spec = importlib.util.spec_from_file_location("browser_report_under_test", where)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(where, *argv) -> None:
    git = shutil.which("git")
    assert git is not None
    subprocess.run(
        [git, "-c", "user.email=stamp@test", "-c", "user.name=stamp", *argv],
        cwd=where,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )


@pytest.fixture
def scratch_repo(tmp_path):
    _git(tmp_path, "init", "--initial-branch=main")
    (tmp_path / "witness.txt").write_text("as committed\n", encoding="utf-8")
    _git(tmp_path, "add", "witness.txt")
    _git(tmp_path, "commit", "-m", "the state the stamp should name")
    return tmp_path


def test_a_clean_tree_is_stamped_with_its_commit_alone(scratch_repo):
    stamp = _driver()._commit_stamp(scratch_repo)
    assert stamp not in ("", "unknown")
    assert not stamp.endswith("-dirty")
    assert all(ch in "0123456789abcdef" for ch in stamp), stamp


def test_a_dirty_tree_says_so(scratch_repo):
    (scratch_repo / "witness.txt").write_text("changed since\n", encoding="utf-8")
    assert _driver()._commit_stamp(scratch_repo).endswith("-dirty")


def test_an_untracked_file_also_dirties_the_stamp(scratch_repo):
    """The incident's exact shape: the driver itself was a new, untracked
    file, and the stamp named the parent commit as if it built the run."""
    (scratch_repo / "brand_new.py").write_text("print()\n", encoding="utf-8")
    assert _driver()._commit_stamp(scratch_repo).endswith("-dirty")


def test_a_directory_that_is_no_repository_is_not_guessed_at(tmp_path):
    assert _driver()._commit_stamp(tmp_path) == "unknown"


@pytest.mark.skipif(sys.platform != "win32", reason="the stub git is a .bat; this repo's tooling is Windows")
def test_an_unanswerable_cleanliness_check_is_labelled_not_assumed(tmp_path, monkeypatch):
    """rev-parse succeeds, status fails: the stamp must say -unverified
    rather than silently presenting the commit as a clean build. Simulated
    with a stub git, because a real repo cannot half-fail on demand."""
    stub = tmp_path / "half-git.bat"
    stub.write_text(
        '@echo off\r\nif "%1"=="rev-parse" (echo abc1234) else (exit /b 9)\r\n',
        encoding="ascii",
    )
    monkeypatch.setattr(shutil, "which", lambda _tool: str(stub))
    assert _driver()._commit_stamp(tmp_path) == "abc1234-unverified"
