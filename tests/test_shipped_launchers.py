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

import pathlib
import re

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# What a person downloads, copies, or pastes from. Not tests/ -- fixtures
# there use obvious throwaways like 'correct-horse-battery' on purpose.
_SHIPPED_GLOBS = ("*.bat", "*.bash", "*.sh", "*.yaml", "*.yml", "Dockerfile",
                  "*.md", "justfile", "*.just")
_SKIP_DIRS = {".git", ".venv", "tests", "node_modules", "__pycache__",
              ".pytest_cache", ".AImodels", "vendor", "experiments"}

# `--admin-pass VALUE`, `--admin-pass=VALUE`, `ADMIN_PASSWORD=VALUE`.
# Backticks are excluded because most mentions live inside markdown code
# spans, where a trailing ` would otherwise be read as part of the password.
_SECRET = re.compile(r"""(?:--admin-pass[=\s]+|ADMIN_PASSWORD=)(["']?)([^\s"'`#]*)""")
_TRAILING = ",.);|"


def _is_placeholder(value: str) -> bool:
    """True when the value could not possibly be somebody's real password."""
    if not value:
        return True                                   # deliberately unset
    if any(ch in value for ch in "%$<>{}"):           # %VAR%, ${VAR}, <pwd>
        return True
    if re.fullmatch(r"[A-Z][A-Z_]*", value):          # PASSWORD, YOUR_PASSWORD
        return True
    return value.lower() in {"yourpassword", "your_password", "your-password"}


def _shipped_files():
    seen = []
    for pattern in _SHIPPED_GLOBS:
        for path in _REPO_ROOT.rglob(pattern):
            if any(part in _SKIP_DIRS for part in path.relative_to(_REPO_ROOT).parts):
                continue
            if path.is_file():
                seen.append(path)
    return sorted(set(seen))


def _findings(text: str):
    return [value for _quote, raw in _SECRET.findall(text)
            for value in [raw.rstrip(_TRAILING)]
            if not _is_placeholder(value)]


def test_the_scan_actually_reaches_the_shipped_files():
    """Control for the sweep below. A broken glob or an over-eager skip list
    would make it pass by reading nothing at all."""
    files = _shipped_files()
    names = {p.name for p in files}

    assert len(files) > 10, f"only found {len(files)} shipped files"
    assert {"sample_run_exhibition.bat", "sample_run_smartgallery.bat",
            "README.md"} <= names, sorted(names)

    mentions = [p for p in files
                if "--admin-pass" in p.read_text(encoding="utf-8", errors="replace")
                or "ADMIN_PASSWORD=" in p.read_text(encoding="utf-8", errors="replace")]
    assert mentions, "no shipped file mentions the password at all -- the "
    "sweep below would pass for the wrong reason"


def test_the_detector_catches_the_line_that_shipped():
    """Control for the matcher. Without this the sweep could be passing
    because the regex matches nothing, and a real credential would sail
    straight through."""
    was_shipped = (r"..\python\python.exe smartgallery.py --port %SERVER_PORT% "
                   r"--exhibition --admin-pass maffettone")

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


@pytest.mark.parametrize("path", _shipped_files(),
                         ids=lambda p: str(p.relative_to(_REPO_ROOT)))
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
        f"guesses at English is one that misses secrets.")


def test_the_exhibition_sample_asks_for_a_password_without_supplying_one():
    """The specific shape of the fix: exhibition needs an admin account, so
    the sample has to prompt for one -- and must not put it on the command
    line, where other programs on the machine can read it."""
    text = (_REPO_ROOT / "sample_run_exhibition.bat").read_text(encoding="utf-8")
    start_line = next(line for line in text.splitlines()
                      if "smartgallery.py" in line and not line.strip().startswith("::"))

    assert 'set "ADMIN_PASSWORD="' in text, text
    assert "--admin-pass" not in start_line, start_line
    assert "--exhibition" in start_line, start_line
