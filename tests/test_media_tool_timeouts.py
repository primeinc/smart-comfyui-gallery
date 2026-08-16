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

import subprocess
import sys
import time

import pytest

import smartgallery


@pytest.fixture()
def sleeper(tmp_path):
    """A program that ignores its arguments and sleeps."""
    script = tmp_path / "sleeper.py"
    script.write_text("import time, sys\ntime.sleep(30)\n", encoding="utf-8")
    return [sys.executable, str(script)]


def test_the_version_check_gives_up(sleeper, monkeypatch):
    """This one runs before the app starts: a hang here means it never
    starts at all, with no message."""
    monkeypatch.setattr(smartgallery, "FFPROBE_TIMEOUT", 1)

    started = time.monotonic()
    result = smartgallery._is_ffprobe(sleeper[1])  # the script path alone
    elapsed = time.monotonic() - started

    assert result is False
    assert elapsed < 15, f"the version check took {elapsed:.1f}s"


def test_a_hanging_probe_costs_only_that_file(tmp_path, monkeypatch, sleeper):
    """The scan must move on rather than stop."""
    monkeypatch.setattr(smartgallery, "FFPROBE_TIMEOUT", 1)
    monkeypatch.setattr(smartgallery, "FFPROBE_EXECUTABLE_PATH", sleeper[1])

    target = tmp_path / "clip.mp4"
    target.write_bytes(b"not really a video")

    started = time.monotonic()
    # analyze_media_metadata swallows failures; what matters is that it returns.
    try:
        smartgallery.extract_workflow(str(target))
    except Exception:
        pass
    elapsed = time.monotonic() - started

    assert elapsed < 20, f"metadata extraction took {elapsed:.1f}s on a hanging probe"


@pytest.mark.parametrize("name,expected", [
    ("FFPROBE_TIMEOUT", 30),
    ("FFMPEG_TIMEOUT", 300),
])
def test_the_ceilings_are_set(name, expected):
    """A regression here would be someone removing the constant rather than
    the keyword, so the value is pinned as well as its use."""
    assert getattr(smartgallery, name) == expected


def test_every_media_subprocess_call_passes_a_timeout():
    """The sweep that found this, kept. A new ffmpeg call added without a
    timeout is a new way to hang the gallery, and it would look exactly
    like the ones that were already there."""
    import ast
    import io
    import pathlib

    source = pathlib.Path(smartgallery.__file__)
    tree = ast.parse(io.open(source, encoding="utf-8").read())

    missing = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "subprocess"
                and func.attr in {"run", "check_output", "call"}):
            continue
        if "timeout" not in {kw.arg for kw in node.keywords if kw.arg}:
            missing.append(node.lineno)

    assert missing == [], (
        f"subprocess calls without a timeout at lines {missing} of "
        f"{source.name}. An external tool that hangs there hangs whatever is "
        f"waiting on it -- a request, the scan, or start-up.")


def test_a_timeout_leaves_no_child_behind(sleeper, monkeypatch):
    """subprocess.run kills the child when the timeout expires. If that ever
    stopped being true, every timeout would leak a process."""
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(sleeper, capture_output=True, timeout=1)
    assert time.monotonic() - started < 10
