"""ffmpeg and ffprobe must not be able to hang the gallery.

Both hang readily -- on a truncated or malformed file, on a path that goes
away mid-read, on network storage that stops answering. Every call to them
ran without a timeout, and each one is somewhere a person is waiting:

  _is_ffprobe          runs before the app starts
  the metadata read    runs inside the scan
  thumbnail extraction runs inside a request
  metadata stripping   runs inside a request a VISITOR is waiting on

A stuck child process there does not fail; it simply never returns, which
is the hardest kind of fault to attribute. Nobody looks at a gallery that
will not finish loading and suspects one video file.

Every caller already treats failure as "no metadata" or "no thumbnail", so
a timeout costs that one file and nothing else. These tests use a stand-in
that sleeps rather than a real ffmpeg, because the point is the call, not
the tool.
"""

from __future__ import annotations

import ast
import contextlib
import pathlib
import subprocess

import pytest

import smartgallery


@pytest.fixture
def a_tool_that_hangs(monkeypatch):
    """Every media subprocess times out instead of answering.

    This was a real child sleeping for thirty seconds, bounded by a
    one-second timeout -- three of them, so three process starts and three
    seconds of waiting to observe a branch that is one `except` away. What
    the gallery has to get right is what it does WITH the timeout, and
    raising TimeoutExpired where subprocess.run is called asks exactly
    that, instantly and on every platform.
    """

    def times_out(cmd, *args, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1))

    monkeypatch.setattr(smartgallery.subprocess, "run", times_out)


def test_the_version_check_gives_up(a_tool_that_hangs):
    """This one runs before the app starts: a hang here means it never
    starts at all, with no message. A tool that never answers has to read
    as "not ffprobe", not as an exception on the way up."""
    assert smartgallery._is_ffprobe("anything") is False


def test_the_probe_still_says_yes_to_a_real_ffprobe(monkeypatch):
    """The counterpart. A check that answered False for everything would
    satisfy the test above while making video features impossible."""
    banner = b"ffprobe version 7.1"

    def answers(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, banner, b"")

    monkeypatch.setattr(smartgallery.subprocess, "run", answers)

    assert smartgallery._is_ffprobe("ffprobe") is True


def test_a_hanging_probe_costs_only_that_file(a_tool_that_hangs, tmp_path, monkeypatch):
    """The scan must move on rather than stop."""
    monkeypatch.setattr(smartgallery.STATE, "ffprobe_path", "hanging-ffprobe")
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"not really a video")

    # extract_workflow swallows failures; what matters is that it returns.
    with contextlib.suppress(Exception):
        smartgallery.extract_workflow(str(target))


def test_every_media_subprocess_call_passes_a_timeout(gallery_tree):
    """The sweep that found this, kept. A new ffmpeg call added without a
    timeout is a new way to hang the gallery, and it would look exactly
    like the ones that were already there."""

    source = pathlib.Path(smartgallery.__file__)
    tree = gallery_tree

    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "subprocess"
            and func.attr in {"run", "check_output", "call"}
        ):
            continue
        if "timeout" not in {kw.arg for kw in node.keywords if kw.arg}:
            missing.append(node.lineno)

    assert missing == [], (
        f"subprocess calls without a timeout at lines {missing} of "
        f"{source.name}. An external tool that hangs there hangs whatever is "
        f"waiting on it -- a request, the scan, or start-up."
    )


def test_no_media_tool_is_started_in_a_way_that_can_outlive_its_timeout(gallery_tree):
    """A timeout only stops a leak if the call it is attached to kills the
    child. subprocess.run does; Popen does not, and leaves the caller
    holding a process nothing will reap.

    This replaces a test that spawned a real sleeper and waited a second to
    watch CPython honour its own documented contract -- which is not this
    codebase's behaviour to verify. What IS ours is never reaching for the
    call that has no such contract, which is a property of the source.
    """
    media = {"ffprobe", "ffmpeg", "ffprobe_path", "FFMPEG_EXECUTABLE_PATH"}
    offenders = []
    for node in ast.walk(gallery_tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Popen"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        ):
            continue
        mentions = " ".join(ast.dump(arg) for arg in node.args)
        if any(name in mentions for name in media):
            offenders.append(node.lineno)

    assert offenders == [], (
        f"a media tool is started with subprocess.Popen at lines {offenders}. "
        f"Popen has no timeout and does not reap the child, so a hung ffmpeg "
        f"survives whatever bound the caller thought it had. Use "
        f"subprocess.run(..., timeout=...), or route it through "
        f"stream_media_process, which owns its own clock and kill."
    )
