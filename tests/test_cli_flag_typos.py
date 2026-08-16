"""A mistyped command-line flag must not start the gallery anyway.

The flags are parsed with `parse_known_args`, which collects anything it
does not recognise into a list that was then dropped without a word. So
`--forcelogin`, or `--force_login` with an underscore, started a gallery
with no login at all while the operator believed it was shut. The same
silence applied to `--exhibition` and `--blind-rating`: the mode simply did
not happen, and nothing said so.

The check only applies when smartgallery.py is the program being run.
Imported -- by these tests, or by anything embedding the gallery -- argv
belongs to the host and its arguments are not ours to police. That matters
here: pytest's own argv would otherwise stop the suite at import.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(args, env_extra):
    """Run smartgallery.py as a program, only far enough to parse argv."""
    env = dict(os.environ, ENABLE_AI_DAM="false", AI_DAM_AUTO_PROVISION="false",
               **env_extra)
    # -c so nothing serves; argv[0] is what makes the check apply.
    script = (
        "import sys\n"
        f"sys.argv = ['smartgallery.py'] + {args!r}\n"
        "import smartgallery\n"
        "print('STARTED', smartgallery.FORCE_LOGIN, smartgallery.IS_EXHIBITION_MODE)\n"
    )
    return subprocess.run([sys.executable, "-c", script], cwd=_ROOT, env=env,
                          capture_output=True, text=True, timeout=300)


@pytest.fixture()
def gallery_env(tmp_path):
    gallery = tmp_path / "gallery"
    output = tmp_path / "output"
    gallery.mkdir()
    output.mkdir()
    return {"BASE_OUTPUT_PATH": str(output), "BASE_SMARTGALLERY_PATH": str(gallery)}


@pytest.mark.parametrize("typo", [
    "--forcelogin",
    "--force_login",
    "--exhibiton",
    "--blind_rating",
    "--enable-guest-logins",
])
def test_a_mistyped_flag_refuses_to_start(gallery_env, typo):
    """The regression: these started a gallery that was not what was asked
    for, and said nothing."""
    proc = _run([typo, "--admin-pass", "correct-horse-battery"], gallery_env)

    assert proc.returncode != 0, (
        f"{typo} started the gallery anyway:\n{proc.stdout}")
    assert "STARTED" not in proc.stdout
    assert typo in proc.stdout + proc.stderr, "the message does not name the option"


def test_the_message_suggests_the_real_flag(gallery_env):
    """A refusal that does not say what was meant just moves the problem."""
    proc = _run(["--forcelogin", "--admin-pass", "correct-horse-battery"], gallery_env)

    assert "--force-login" in proc.stdout + proc.stderr, (
        f"no suggestion offered:\n{proc.stdout}\n{proc.stderr}")


def test_the_real_flags_still_start(gallery_env):
    """The counterpart -- refusing everything would pass the tests above."""
    proc = _run(["--force-login", "--admin-pass", "correct-horse-battery",
                 "--blind-rating"], gallery_env)

    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "STARTED True False" in proc.stdout, proc.stdout


def test_no_flags_at_all_still_starts(gallery_env):
    proc = _run([], gallery_env)

    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "STARTED False False" in proc.stdout, proc.stdout


def test_importing_the_module_ignores_the_hosts_arguments(gallery_env):
    """pytest runs with its own argv, and anything embedding the gallery has
    its own too. The check must not fire for them -- this suite would not
    run at all if it did."""
    env = dict(os.environ, ENABLE_AI_DAM="false", AI_DAM_AUTO_PROVISION="false",
               **gallery_env)
    script = (
        "import sys\n"
        "sys.argv = ['some-other-program', '-q', '--not-our-flag']\n"
        "import smartgallery\n"
        "print('IMPORTED OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], cwd=_ROOT, env=env,
                          capture_output=True, text=True, timeout=300)

    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "IMPORTED OK" in proc.stdout
