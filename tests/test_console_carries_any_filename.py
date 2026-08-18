"""Printing the name of a file must not be able to stop the gallery.

The gallery reports damaged files, unreachable folders and offline mounts
by name, in 36 places, and those names are often not written in English --
which is the whole point of the unicode work elsewhere. Whether such a
name can be printed at all depends on where the output goes. Attached to a
console, Python writes wide characters and anything prints. Redirected to
a file or a pipe -- a launcher keeping a log, ComfyUI starting the gallery
itself, anything reading its output -- the encoding becomes the machine's
code page, which on most Windows installs is cp1252.

Printing a Japanese filename to cp1252 raises UnicodeEncodeError, and it
raises inside the message rather than inside the work. The scan is built
so that one damaged file costs that file and nothing else; the line saying
so then killed the process. Measured on a library of one good picture and
one damaged one, both named in CJK, output redirected:

    STARTUP DIED: 'charmap' codec can't encode characters in position
    38-42: character maps to <undefined>

This was invisible on the machine it was written on, where PYTHONUTF8=1
was set in the environment. The checks below therefore clear it and pass
-X utf8=0, and test_the_condition_is_real confirms the child really is on
a code page that cannot hold the name -- without it, every check here
would pass on a UTF-8 box while the bug sat untouched.
"""

from __future__ import annotations

import ast
import io
import os
import pathlib
import subprocess
import sys

import pytest

import smartgallery

pytestmark = pytest.mark.spawns  # every check here runs another program

# Chinese, Japanese, Korean, Cyrillic, and a German name with an umlaut --
# none of which cp1252 can hold, except the last, which it can.
_NAMES = ["测试图片.png", "壊れた画像.png", "사진.png", "рисунок.png"]

_REPO = str(pathlib.Path(smartgallery.__file__).parent)


def _run(body):
    """Run a program in a child whose output is a pipe, not a console, on a
    machine that has not opted into UTF-8."""
    environment = dict(os.environ, PYTHONPATH=_REPO)
    environment.pop("PYTHONUTF8", None)
    environment.pop("PYTHONIOENCODING", None)
    return subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", body], env=environment, capture_output=True, timeout=600
    )


def test_the_condition_is_real():
    """Control, and the one that keeps this file honest.

    Everything below asserts that something does NOT fail. That is worth
    nothing unless the same thing fails without the gallery's doing, so
    this prints the same names with no import at all and requires it to
    break."""
    finished = _run("print('\\u6d4b\\u8bd5\\u56fe\\u7247.png')")

    assert finished.returncode != 0, (
        "a Chinese filename printed to a redirected stream succeeded with "
        "no help, so this machine is not reproducing the condition and none "
        "of the checks below mean anything"
    )
    assert b"UnicodeEncodeError" in finished.stderr, finished.stderr[-400:]


def test_every_name_survives_a_real_redirected_pipe():
    """The symptom, through the real import, on a real pipe.

    One child rather than seven. Each name used to get its own interpreter
    -- three seconds of process start and gallery loading apiece -- to
    prove the same mechanism over again. The mechanism itself is checked in
    this process by test_a_code_page_stream_stops_raising below; what only
    a child can show is that it holds for a genuinely redirected stream on
    a machine that has not opted into UTF-8, and one child shows that for
    every name at once.

    Three claims ride together because they share the child: the process
    survives each name, the names arrive intact rather than as question
    marks, and ordinary English is untouched.
    """
    printed = "".join("print('%s')\n" % name.encode("unicode_escape").decode() for name in _NAMES)
    finished = _run(
        "import sys; sys.argv = ['smartgallery.py']\n"
        "import smartgallery\n" + printed + "print('INFO: Starting full file scan...')\n"
        "print('still running')\n"
    )

    assert finished.returncode == 0, (
        "printing the names ended the process:\n" + finished.stderr.decode("utf-8", "replace")[-1500:]
    )
    assert b"still running" in finished.stdout, finished.stdout[-400:]

    out = finished.stdout.decode("utf-8", "replace")
    for name in _NAMES:
        # Not raising is the requirement; arriving intact is the point. A
        # stream that swallowed every such name into question marks would
        # satisfy the return code and tell nobody which file was damaged.
        assert name in out, f"{name} came back as something else:\n{out!r}"

    # Over-reach guard: the great majority of what this prints is plain
    # English, and it has to be exactly as it was.
    assert "INFO: Starting full file scan..." in out, out


def test_a_stream_that_cannot_be_reconfigured_is_left_alone():
    """Over-reach guard. Output is not always a file: under a test runner,
    a service wrapper, or an embedding host it can be an object with no
    reconfigure at all, and reaching for one must not be what breaks."""

    class Plain:
        def __init__(self):
            self.written = []

        def write(self, text):
            self.written.append(text)

    stream = Plain()

    smartgallery.make_output_carry_any_filename([stream])

    stream.write("still usable")
    assert stream.written == ["still usable"]


def test_a_code_page_stream_stops_raising():
    """The mechanism on its own, without a subprocess: a stream that could
    not hold the name before can be written to afterwards."""

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", newline="")

    with pytest.raises(UnicodeEncodeError):
        stream.write("测试图片.png")
        stream.flush()

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", newline="")
    smartgallery.make_output_carry_any_filename([stream])
    stream.write("测试图片.png")
    stream.flush()

    assert raw.getvalue().decode("utf-8") == "测试图片.png"


def test_it_runs_before_anything_can_print(gallery_tree):
    """Placement is the whole of it. A call made after the first print is
    a call made after the first chance to die."""

    tree = gallery_tree

    called_at = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "make_output_carry_any_filename"
    ]
    assert called_at, "nothing calls make_output_carry_any_filename"

    printed_at = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
    ]
    assert printed_at, "no print() calls found; this check is stale"

    assert min(called_at) < min(printed_at), (
        f"the first print() is at line {min(printed_at)} and the streams are not set up until line {min(called_at)}"
    )
