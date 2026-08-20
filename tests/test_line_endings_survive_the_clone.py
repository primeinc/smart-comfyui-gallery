"""What a clone hands out must be what the repository decided.

Git for Windows installs with core.autocrlf=true, so a clone made there --
which is most of them, this being a ComfyUI companion -- rewrites text
files to CRLF on checkout, and a clone made anywhere else converts
nothing. Neither machine's setting is this repository's decision, so the
decision travels in .gitattributes, and these tests hold the two halves
that survive any set of shipped files: nothing committed may hold a CRLF
(those bytes are handed to everyone whatever their Git is set to), and the
policy file itself must be present so a checkout has a rule to follow.

The controls run one checkout per autocrlf mode and prove the mode is
really in force, because every check here is an absence and an absence is
also what a checkout that did nothing at all would produce.
"""

from __future__ import annotations

import functools
import pathlib
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.spawns  # every check here runs another program

_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Follows whatever the checkout does -- `* text=auto` with no eol rule of
# its own -- so it says which mode is in force.
_BELLWETHER = "pyproject.toml"


def _git(*args, cwd=None):
    git = shutil.which("git") or "git"
    return subprocess.run([git, *args], cwd=str(cwd or _ROOT), capture_output=True, timeout=900, check=False)


def _endings(path):
    data = pathlib.Path(path).read_bytes()
    crlf = data.count(b"\r\n")
    bare = data.count(b"\n") - crlf
    return crlf, bare


@pytest.fixture(scope="module")
def checkouts(tmp_path_factory):
    """The repository as each kind of machine lays it on disk.

    checkout-index runs the same conversion a checkout does, without the
    cost of cloning.
    """
    if shutil.which("git") is None:
        pytest.skip("git is not on PATH")

    made = {}
    for autocrlf in ("true", "false"):
        target = tmp_path_factory.mktemp(f"co_{autocrlf}")
        result = _git("-c", f"core.autocrlf={autocrlf}", "checkout-index", "-a", "-f", f"--prefix={target}/")
        if result.returncode != 0:
            pytest.skip(result.stderr.decode(errors="replace")[:200])
        made[autocrlf] = target
    return made


@pytest.mark.parametrize(("autocrlf", "expect_crlf"), [("true", True), ("false", False)])
def test_the_checkout_is_really_in_that_mode(checkouts, autocrlf, expect_crlf):
    """Control, one per mode, and the thing the rest of the file rests on:
    a file with no rule of its own has to follow the setting."""
    crlf, bare = _endings(checkouts[autocrlf] / _BELLWETHER)

    if expect_crlf:
        not_converting = (
            f"with core.autocrlf=true a file with no rule came out with {crlf} CRLF and {bare} bare "
            f"newlines; this checkout is not converting, so the checks here would pass against no "
            f"policy at all"
        )
        assert crlf > 0, not_converting
        assert bare == 0, not_converting
    else:
        converting = (
            f"with core.autocrlf=false a file with no rule came out with {crlf} CRLF and {bare} bare "
            f"newlines; this checkout is converting on its own, so the checks below would pass "
            f"without any policy"
        )
        assert bare > 0, converting
        assert crlf == 0, converting


@functools.cache
def _committed_line_endings():
    """{path: eolinfo} for the bytes in the index, straight from Git.

    `git ls-files --eol` reports i/<eolinfo> -- the content identification
    of what is stored, one of "-text", "none", "lf", "crlf", "mixed" or ""
    (Documentation/git-ls-files.adoc:198-213). One process for the whole
    repository.
    """
    listed = _git("ls-files", "--eol")
    assert listed.returncode == 0, listed.stderr.decode(errors="replace")

    endings = {}
    for line in listed.stdout.decode("utf-8", "replace").splitlines():
        fields, _tab, path = line.partition("\t")
        if not path:
            continue
        for field in fields.split():
            if field.startswith("i/"):
                endings[path] = field[2:]
    return endings


def test_the_index_reading_finds_the_text_files():
    """Control. The check below is an absence, and an absence is also what
    a parse that understood nothing would produce."""
    endings = _committed_line_endings()

    assert len(endings) > 100, f"only read {len(endings)} tracked files"
    assert sum(1 for kind in endings.values() if kind == "lf") > 50, (
        f"no committed file reads as lf: {sorted(set(endings.values()))}. "
        f"The parse is not reaching the i/<eolinfo> field."
    )


def test_nothing_committed_holds_a_carriage_return():
    """The repository's own side of it. A file committed with CRLF is
    handed out that way to everyone, whatever their checkout does."""
    offenders = sorted(path for path, kind in _committed_line_endings().items() if kind in {"crlf", "mixed"})

    assert offenders == [], (
        f"committed with CRLF: {offenders}. Everyone gets those bytes whatever their own Git is set to."
    )


def test_the_policy_travels_with_the_repository():
    """core.autocrlf is a setting on somebody else's machine, and the
    default there is not the one this repository needs."""
    assert (_ROOT / ".gitattributes").exists(), (
        "no .gitattributes, so what a checkout does to the tree is "
        "whatever the person cloning happens to have configured"
    )
