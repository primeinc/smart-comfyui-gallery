"""Nothing shipped may carry a real password.

`sample_run_exhibition.bat` -- the launcher whose own instructions tell you
to rename it and run it -- ended its start line with
`--exhibition --admin-pass maffettone`. Exhibition is the mode built to be
shared with family, friends and clients, and its admin password was a word
anyone could read in a public repository. Every other one of the twenty-odd
mentions across the README and the sibling launcher used the placeholder
`yourpassword`, so this was a leftover rather than a decision.

Substituting `yourpassword` would not have fixed it: an unedited launcher
would then hand everybody who downloaded it the same known password. The
sample now sets `ADMIN_PASSWORD` empty instead, and a blank environment
variable reads as unset (`env_or`), so exhibition refuses to start and
prints how to fix it rather than running open.

These tests read the shipped files, so they fail if a real credential is
ever committed again -- in any launcher, compose file, or document.
"""

from __future__ import annotations

import fnmatch
import functools
import os
import pathlib
import re

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# What a person downloads, copies, or pastes from. Not tests/ -- fixtures
# there use obvious throwaways like 'correct-horse-battery' on purpose.
_SHIPPED_GLOBS = ("*.bat", "*.bash", "*.sh", "*.yaml", "*.yml", "Dockerfile", "*.md", "justfile", "*.just")
_SKIP_DIRS = {
    ".git",
    ".venv",
    "tests",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".AImodels",
    "vendor",
    "experiments",
}

# `--admin-pass VALUE`, `--admin-pass=VALUE`, `ADMIN_PASSWORD=VALUE`.
# Backticks are excluded because most mentions live inside markdown code
# spans, where a trailing ` would otherwise be read as part of the password.
_SECRET = re.compile(r"""(?:--admin-pass[=\s]+|ADMIN_PASSWORD=)(["']?)([^\s"'`#]*)""")
_TRAILING = ",.);|"


def _is_placeholder(value: str) -> bool:
    """True when the value could not possibly be somebody's real password."""
    if not value:
        return True  # deliberately unset
    if any(ch in value for ch in "%$<>{}"):  # %VAR%, ${VAR}, <pwd>
        return True
    if re.fullmatch(r"[A-Z][A-Z_]*", value):  # PASSWORD, YOUR_PASSWORD
        return True
    return value.lower() in {"yourpassword", "your_password", "your-password"}


@functools.cache
def _shipped_files():
    """Every file a person downloads, copies, or pastes from.

    One walk that prunes as it goes. This used to rglob the repository
    once per pattern and apply _SKIP_DIRS to the results, so the whole of
    .venv, .git and .AImodels was walked nine times over to arrive at a
    few dozen files -- eleven seconds, and paid twice, because the
    parametrize below calls this at collection as well. Pruning is what
    makes it cheap; the file set is unchanged.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(_REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if name in _SKIP_DIRS:
                continue
            if any(fnmatch.fnmatch(name, pattern) for pattern in _SHIPPED_GLOBS):
                found.append(pathlib.Path(dirpath) / name)
    return sorted(set(found))


def _findings(text: str):
    return [
        value
        for _quote, raw in _SECRET.findall(text)
        for value in [raw.rstrip(_TRAILING)]
        if not _is_placeholder(value)
    ]


def test_the_scan_actually_reaches_the_shipped_files():
    """Control for the sweep below. A broken glob or an over-eager skip list
    would make it pass by reading nothing at all."""
    files = _shipped_files()
    names = {p.name for p in files}

    assert len(files) > 10, f"only found {len(files)} shipped files"
    assert {"sample_run_exhibition.bat", "sample_run_smartgallery.bat", "README.md"} <= names, sorted(names)

    mentions = [
        p
        for p in files
        if "--admin-pass" in p.read_text(encoding="utf-8", errors="replace")
        or "ADMIN_PASSWORD=" in p.read_text(encoding="utf-8", errors="replace")
    ]
    assert mentions, "no shipped file mentions the password at all -- the "
    "sweep below would pass for the wrong reason"


def test_the_detector_catches_the_line_that_shipped():
    """Control for the matcher. Without this the sweep could be passing
    because the regex matches nothing, and a real credential would sail
    straight through."""
    was_shipped = (
        r"..\python\python.exe smartgallery.py --port %SERVER_PORT% "
        r"--exhibition --admin-pass maffettone"
    )

    assert _findings(was_shipped) == ["maffettone"]
    assert _findings('set "ADMIN_PASSWORD=hunter2222"') == ["hunter2222"]
    # and the forms that must NOT be flagged
    assert _findings("--admin-pass yourpassword") == []
    assert _findings('set "ADMIN_PASSWORD="') == []
    assert _findings("--admin-pass ${EXHIBITION_PASS}") == []
    assert _findings("`--admin-pass <pwd>`") == []
    assert _findings("--admin-pass YOUR_STRONG_PASSWORD") == []
    # markdown code spans and table cells: the trailing punctuation is not
    # part of the password, and reading it as one made this fail on the
    # README's own placeholders
    assert _findings("`--admin-pass yourpassword`") == []
    assert _findings("| `--admin-pass PASSWORD` | Set or reset it. |") == []
    assert _findings("run `--admin-pass hunter2222`, then stop.") == ["hunter2222"]


@pytest.mark.parametrize("path", _shipped_files(), ids=lambda p: str(p.relative_to(_REPO_ROOT)))
def test_no_shipped_file_carries_a_real_password(path):
    found = _findings(path.read_text(encoding="utf-8", errors="replace"))

    assert not found, (
        f"{path.relative_to(_REPO_ROOT)} ships a concrete admin password "
        f"{found!r}. Anyone who downloads this gets that password. Use an "
        f"empty ADMIN_PASSWORD, or a placeholder, and let the gallery refuse "
        f"to start until the person sets their own.\n"
        f"If that is prose rather than a command, the next word after the "
        f"flag reads as its value -- reword so the flag is not followed by "
        f"one. Staying strict here is deliberate: a secret scanner that "
        f"guesses at English is one that misses secrets."
    )


def _launch_lines(text):
    """The lines that actually start the app.

    Identified by invoking the interpreter, not by being the first line
    that mentions smartgallery.py -- the launchers now check where the
    app is before running it, and `if exist "smartgallery.py"` is not a
    launch line. Picking by position quietly read that one instead.
    """
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("::", "#")):
            continue
        if "smartgallery.py" not in stripped:
            continue
        if "%PYTHON%" in stripped or "$PYTHON" in stripped:
            lines.append(stripped)
    return lines


@pytest.mark.parametrize("name", ["sample_run_exhibition.bat", "sample_run_exhibition.sh"])
def test_the_exhibition_sample_asks_for_a_password_without_supplying_one(name):
    """The specific shape of the fix: exhibition needs an admin account, so
    the sample has to prompt for one -- and must not put it on the command
    line, where other programs on the machine can read it."""
    path = _REPO_ROOT / name
    assert path.exists(), f"{name} is not shipped"
    text = path.read_text(encoding="utf-8")

    launch = _launch_lines(text)
    assert launch, f"no launch line found in {name}"

    assert "ADMIN_PASSWORD=" in text, text
    for line in launch:
        assert "--admin-pass" not in line, line
    assert any("--exhibition" in line for line in launch), launch


@pytest.mark.parametrize(
    "name",
    [
        "sample_run_smartgallery.bat",
        "sample_run_smartgallery.sh",
        "sample_run_exhibition.bat",
        "sample_run_exhibition.sh",
    ],
)
def test_every_sample_launcher_looks_for_an_environment_first(name):
    """A launcher that runs the wrong interpreter fails in a way nobody can
    read. Each one has to look for .venv and venv, and say what to do when
    there is neither -- both ways of making one, since the project ships a
    uv.lock as well as a requirements.txt."""
    path = _REPO_ROOT / name
    assert path.exists(), f"{name} is not shipped"
    text = path.read_text(encoding="utf-8")

    assert ".venv" in text, f"{name} never looks for .venv"
    assert "venv" in text, f"{name} never looks for venv"
    assert "uv sync" in text, f"{name} does not mention the uv route"
    assert "requirements.txt" in text, f"{name} does not mention the pip route"
    assert "astral.sh/uv/install" in text, f"{name} tells people to run uv without saying how to get it"
