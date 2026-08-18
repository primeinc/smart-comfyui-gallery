"""The repository must not ship anyone's personal launch script.

`.gitignore` has listed `run_smartgallery.bat` since the day it was
committed -- and the file was committed in that same commit. .gitignore has
no effect on a path that is already tracked, so the rule sat there doing
nothing while the file shipped to everyone.

That matters because of what the file is. The README has every user make
`run_smartgallery.bat` themselves by renaming `sample_run_smartgallery.bat`
and filling in their own paths, and then says, for upgrades: "extract the
new version, overwrite your existing files. Remember to keep your existing
launch scripts to preserve your custom paths and settings." A
`run_smartgallery.bat` inside the release is the one file that instruction
cannot survive -- it overwrites the reader's own copy.

The one that shipped pointed BASE_OUTPUT_PATH at nothing, turned
GENERATE_THUMBNAILS off, ran on 8190 rather than the documented 8189, and
named an ffprobe at C:/ffmpeg/bin on somebody else's machine.

The general rule is checked rather than the one filename: anything
.gitignore matches must not be tracked, whichever of the two was added
first.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

pytestmark = pytest.mark.spawns  # every check here runs another program

_REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent

# What the README tells the reader to create for themselves.
_PERSONAL = ("run_smartgallery.bat", "run_exhibition.bat", "run_smartgallery.sh", "run_exhibition.sh")
# What ships as the template for those.
_TEMPLATES = ("sample_run_smartgallery.bat", "sample_run_exhibition.bat")


def _git(*args):
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH; tracking cannot be inspected here")
    return subprocess.run(("git", *args), cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=120)


def _lines(done):
    return [line for line in done.stdout.splitlines() if line.strip()]


def test_git_can_see_this_repository():
    """Control for everything below. Outside a checkout every list would
    come back empty and every assertion would pass for nothing."""
    tracked = _lines(_git("ls-files"))
    if not tracked:
        pytest.skip("not a git checkout")

    assert len(tracked) > 100, len(tracked)
    assert "smartgallery.py" in tracked


def test_nothing_gitignore_matches_is_tracked():
    """The bug, stated as the rule it broke. `git ls-files -i -c` is the
    list of tracked paths that .gitignore matches -- which should always be
    empty, because a rule that names a tracked file does nothing at all."""
    control = _lines(_git("ls-files", "-i", "-o", "--exclude-standard"))
    if not control:
        pytest.skip("no ignored paths on disk; the check cannot be validated")

    offenders = _lines(_git("ls-files", "-i", "-c", "--exclude-standard"))

    assert not offenders, (
        f"tracked although .gitignore matches them: {offenders}. Adding the "
        f"rule did nothing; the file keeps shipping. Untrack it with "
        f"`git rm --cached <path>`, which leaves your own copy on disk."
    )


@pytest.mark.parametrize("name", _PERSONAL)
def test_personal_launchers_are_not_tracked(name):
    """Named explicitly as well as by the rule above, because this is the
    one that overwrites a reader's own settings on upgrade."""
    tracked = set(_lines(_git("ls-files")))
    if not tracked:
        pytest.skip("not a git checkout")

    assert name not in tracked, (
        f"{name} is in the repository. The README has each person create "
        f"that file and carry it across upgrades; shipping one overwrites "
        f"theirs."
    )


@pytest.mark.parametrize("name", _PERSONAL)
def test_personal_launchers_are_ignored(name):
    """So the next one cannot arrive as untracked noise and get swept into
    a commit by `git add -A`."""
    done = _git("check-ignore", "--no-index", "-q", name)

    assert done.returncode == 0, (
        f".gitignore does not match {name}; someone's paths and password can be committed by accident."
    )


@pytest.mark.parametrize("name", _TEMPLATES)
def test_the_templates_are_still_shipped(name):
    """Control for the two tests above: ignoring or deleting the samples
    would satisfy them and leave nobody with a launcher to start from."""
    tracked = set(_lines(_git("ls-files")))
    if not tracked:
        pytest.skip("not a git checkout")

    assert name in tracked, f"{name} is what people copy; it has to ship"
    assert _git("check-ignore", "--no-index", "-q", name).returncode != 0, (
        f"{name} is ignored, so it will not reach anyone"
    )
