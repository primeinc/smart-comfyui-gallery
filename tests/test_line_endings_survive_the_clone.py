"""The shipped scripts must still run after the checkout that delivered them.

docker_init.bash is COPYed into the image and is the container's command.
Git for Windows installs with core.autocrlf=true, so a clone made there --
which is most of them, this being a ComfyUI companion -- rewrites text
files to CRLF. There was no .gitattributes, so nothing said otherwise.
Measured, cloning this repository the way Git is configured by default on
Windows:

    docker_init.bash    CRLF=225  bareLF=0   first line: b'#!/bin/bash\\r'

Linux ends the interpreter name at the newline and trims only spaces and
tabs -- fs/binfmt_script.c takes strnchr(buf, size, '\\n') and then
`while (spacetab(i_end[-1])) i_end--` -- so the carriage return stays part
of the path. /bin/bash\\r does not exist, and an image built from that
clone has a container that cannot start.

The .bat files have the opposite requirement and had the same absence of
one: checked out anywhere but Windows they arrive with bare LF.

So the requirement is not "LF" or "CRLF", it is that each script gets what
it needs whoever checks it out. These check both worlds -- the Windows
default that converts everything, and the default everywhere else that
converts nothing -- and each has a control proving that mode is really in
force, because otherwise a checkout that did nothing at all would satisfy
half the file.

Nothing committed here held a CRLF, so declaring the policy changed no
content. What it changes is what a checkout does with it.
"""

from __future__ import annotations

import functools
import pathlib
import shutil
import subprocess

import pytest

import smartgallery

pytestmark = pytest.mark.spawns  # every check here runs another program

_ROOT = pathlib.Path(smartgallery.__file__).resolve().parent

_MUST_BE_LF = ["docker_init.bash"]
_MUST_BE_CRLF = ["sample_run_smartgallery.bat", "sample_run_exhibition.bat"]

# Follows whatever the checkout does, so it says which mode is in force.
_BELLWETHER = "smartgallery.py"


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
    """Control, one per mode, and the thing the rest of the file rests on.

    Every check below says a file was NOT converted, or WAS. Either is
    free if the checkout is not converting anything, so a file with no
    rule of its own has to follow the setting."""
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
            f"newlines; this checkout is converting on its own, so the CRLF checks below would pass "
            f"without any policy"
        )
        assert bare > 0, converting
        assert crlf == 0, converting


@pytest.mark.parametrize("autocrlf", ["true", "false"])
@pytest.mark.parametrize("name", _MUST_BE_LF)
def test_a_script_that_runs_on_linux_keeps_bare_newlines(checkouts, autocrlf, name):
    """The bug: checked out on Windows, the container's own command
    arrived with a carriage return in its shebang."""
    data = (checkouts[autocrlf] / name).read_bytes()

    # Sliced outside the f-string on purpose: a backslash inside an
    # f-string expression is PEP 701, i.e. Python 3.12+, and this project
    # declares requires-python >= 3.10. On 3.10/3.11 the module would not
    # parse at all and every test in it would vanish.
    first_line = data.split(b"\n")[0]
    assert b"\r" not in data, (
        f"with core.autocrlf={autocrlf}, {name} has carriage returns; its "
        f"first line is {first_line!r} and Linux will look for an "
        f"interpreter with that in the name"
    )


@pytest.mark.parametrize("autocrlf", ["true", "false"])
@pytest.mark.parametrize("name", _MUST_BE_CRLF)
def test_a_launcher_that_runs_on_windows_gets_windows_newlines(checkouts, autocrlf, name):
    """The other half, and the one a Windows machine cannot show on its
    own: checked out anywhere else these came with bare LF."""
    crlf, bare = _endings(checkouts[autocrlf] / name)

    assert crlf > 0, f"with core.autocrlf={autocrlf}, {name} has no CRLF"
    assert bare == 0, f"with core.autocrlf={autocrlf}, {name} has {bare} bare newlines mixed in with its CRLF"


@pytest.mark.parametrize("autocrlf", ["true", "false"])
def test_the_shebang_is_exactly_what_it_should_be(checkouts, autocrlf):
    """Named rather than inferred, because this one line is the whole
    difference between a container that starts and one that does not."""
    first = (checkouts[autocrlf] / "docker_init.bash").read_bytes().split(b"\n")[0]

    assert first == b"#!/bin/bash", first


@functools.cache
def _committed_line_endings():
    """{path: eolinfo} for the bytes in the index, straight from Git.

    `git ls-files --eol` reports i/<eolinfo> -- the content identification
    of what is stored, one of "-text", "none", "lf", "crlf", "mixed" or ""
    (Documentation/git-ls-files.adoc:198-213). One process for the whole
    repository; this used to run `git show HEAD:<path>` once per tracked
    file and scan the bytes itself, which is several hundred processes to
    ask Git something it already knows.
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
    default there is not the one this needs."""
    assert (_ROOT / ".gitattributes").exists(), (
        "no .gitattributes, so what a checkout does to these scripts is "
        "whatever the person cloning happens to have configured"
    )

    for name, expected in [(n, "lf") for n in _MUST_BE_LF] + [(n, "crlf") for n in _MUST_BE_CRLF]:
        result = _git("check-attr", "eol", "--", name)
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        line = result.stdout.decode("utf-8", "replace").strip()
        assert line.endswith(f"eol: {expected}"), line
