"""The container entrypoint must not print the admin password.

docker_init.bash starts as one user, dumps the whole environment to a file,
switches to another, and reads it back, announcing each variable it sets.
It masked values whose NAME contained TOKEN, API or KEY -- so HF_TOKEN was
covered -- and printed everything else in full. The admin password is not
in a variable whose name admits it: the README's docker examples put it
inside CLI_ARGS. So every container start wrote

  ++ Setting environment variable CLI_ARGS [--port 8189 --admin-pass hunter2222 --force-login]

into the container log, where `docker logs` and anything collecting logs
can read it. smartgallery.py already masks --admin-pass out of its own
startup banner, so the intent was settled; the entrypoint just did not do
it.

Two fixes, because there are two shapes: the name list gained PASS, SECRET
and CRED, and the value itself is scrubbed of `--admin-pass <value>` before
anything is echoed.

The tests source the real functions out of the real file rather than
restating them, and check the property that actually matters twice over:
the secret is absent from the output, AND the variable still carries its
true value afterwards. A redaction that also changed what gets exported
would hand every Docker user a broken password.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_INIT = _REPO_ROOT / "docker_init.bash"

_PASSWORD = "hunter2222"
_PASSPHRASE = "s3cr3t-passphrase"
_DUMP = (f"CLI_ARGS=--port 8189 --admin-pass {_PASSWORD} --force-login\n"
         f"ADMIN_PASSWORD={_PASSPHRASE}\n"
         "HF_TOKEN=abcdef123\n"
         "BASE_OUTPUT_PATH=/mnt/output\n")


def _bash():
    found = shutil.which("bash")
    if found is None:
        pytest.skip("no bash on PATH; the entrypoint cannot be exercised here")
    return found


@pytest.fixture(scope="module")
def functions(tmp_path_factory):
    """The real definitions, cut from the real script at the point where
    its top-level logic begins."""
    text = _INIT.read_text(encoding="utf-8")
    marker = "# smartgallerytoo is a specfiic user"
    assert marker in text, "the entrypoint no longer has the expected shape"

    prefix = text[:text.index(marker)]
    # Only that the function under test is in the slice. Requiring the
    # helper that implements the fix would make these tests ERROR on an
    # unfixed entrypoint instead of reporting the leak they exist to catch.
    assert "load_env()" in prefix, prefix[-400:]

    path = tmp_path_factory.mktemp("entrypoint") / "functions.bash"
    path.write_text(prefix, encoding="utf-8", newline="\n")
    return path


def _run(functions, tmp_path, script):
    dump = tmp_path / "envdump.txt"
    dump.write_text(_DUMP, encoding="utf-8", newline="\n")
    # A deliberately bare environment: the suite's own conftest exports
    # BASE_OUTPUT_PATH, and inheriting it sends load_env down its
    # "overwriting" branch instead of its "setting" one.
    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR")
           if key in os.environ}
    done = subprocess.run(
        [_bash(), "-c", f'source "{functions.as_posix()}"; '
                        f'load_env "{dump.as_posix()}" true; {script}'],
        capture_output=True, text=True, timeout=300, env=env)
    return done.stdout + done.stderr


def test_the_harness_reaches_the_real_function(functions, tmp_path):
    """Control. If sourcing failed or load_env never ran, every absence
    below would be an absence of output rather than of secrets."""
    output = _run(functions, tmp_path, "true")

    assert "Loading environment variables from" in output, output
    assert "BASE_OUTPUT_PATH [/mnt/output]" in output, (
        "a value that is not secret should still be printed in full:\n" + output)


def test_the_password_is_not_printed(functions, tmp_path):
    """The leak, in the shape the README's docker examples produce."""
    output = _run(functions, tmp_path, "true")

    assert _PASSWORD not in output, output
    assert "--admin-pass ********" in output, output


def test_a_password_shaped_variable_is_not_printed(functions, tmp_path):
    """The other shape: compose now passes ADMIN_PASSWORD in its own
    variable, whose name the obfuscation list has to recognise."""
    output = _run(functions, tmp_path, "true")

    assert _PASSPHRASE not in output, output
    assert "ADMIN_PASSWORD [**OBFUSCATED**]" in output, output


def test_the_values_that_reach_the_app_are_unchanged(functions, tmp_path):
    """The dangerous way to pass the tests above: redact the value itself.
    Every Docker user would then get a password of literal asterisks."""
    output = _run(
        functions, tmp_path,
        'echo "EXPORTED_CLI=[$CLI_ARGS]"; echo "EXPORTED_PASS=[$ADMIN_PASSWORD]"')

    assert f"EXPORTED_CLI=[--port 8189 --admin-pass {_PASSWORD} --force-login]" in output, output
    assert f"EXPORTED_PASS=[{_PASSPHRASE}]" in output, output


def test_the_entrypoint_is_valid_bash():
    """`bash -n` on the whole file, not just the part sourced above."""
    done = subprocess.run([_bash(), "-n", str(_INIT)],
                          capture_output=True, text=True, timeout=300)

    assert done.returncode == 0, done.stderr
