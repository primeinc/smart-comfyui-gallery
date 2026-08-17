"""Say which ffprobe is running, not just which one was asked for.

The configuration table prints FFPROBE_MANUAL_PATH -- the path that was
requested. When that file is absent the app quietly falls back to whatever
`ffprobe` is on PATH, so the table names a tool that is not the one doing
the work.

Nothing said so. Resolution reported its failures and stayed silent on
success, which leaves someone whose video thumbnails are missing reading a
configuration table that looks correct and concluding the problem lies
elsewhere. Half the video features degrade quietly without ffmpeg: duration
and dimensions, video thumbnails, waveforms, and -- since the fail-closed
change -- metadata stripping refuses rather than serving originals.

It now names the tool it resolved, once per run, and the not-found warning
lists what stops working.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _startup(env_extra, tmp_path, body="print(sg.find_ffprobe_path())\n"):
    gallery = tmp_path / "gallery"
    output = tmp_path / "output"
    gallery.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    env = dict(os.environ, ENABLE_AI_DAM="false", AI_DAM_AUTO_PROVISION="false",
               BASE_OUTPUT_PATH=str(output), BASE_SMARTGALLERY_PATH=str(gallery),
               **env_extra)
    script = ("import sys\nsys.argv = ['smartgallery.py']\n"
              "import smartgallery as sg\n" + body)
    return subprocess.run([sys.executable, "-c", script], cwd=_ROOT, env=env,
                          capture_output=True, text=True, timeout=300)


@pytest.fixture()
def no_ffprobe_anywhere(tmp_path):
    """An environment with nothing named ffprobe on PATH."""
    return {"PATH": str(tmp_path / "empty-path"),
            "FFPROBE_MANUAL_PATH": str(tmp_path / "nowhere" / "ffprobe.exe")}


def test_a_missing_ffprobe_says_what_stops_working(no_ffprobe_anywhere, tmp_path):
    """The regression: 'ffprobe not found. Video metadata analysis will be
    disabled.' understated it -- thumbnails, waveforms and the metadata
    stripping go too."""
    proc = _startup(no_ffprobe_anywhere, tmp_path)

    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "ffprobe not found" in out, out
    for consequence in ("thumbnail", "waveform", "stripping"):
        assert consequence in out.lower(), (
            f"the warning does not mention {consequence}:\n{out}")
    assert "FFPROBE_MANUAL_PATH" in out, "it does not say how to fix it"


def test_resolution_is_announced_once(tmp_path):
    """Whatever it resolves, it says so -- and does not repeat itself on
    every later call, of which the scan makes many."""
    body = ("sg.find_ffprobe_path()\n"
            "sg.find_ffprobe_path()\n"
            "sg.find_ffprobe_path()\n"
            "print('CALLS DONE')\n")
    proc = _startup({}, tmp_path, body)

    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    announcements = [line for line in lines if line.startswith("INFO: ffprobe:")]
    warnings = [line for line in lines if "ffprobe not found" in line]

    assert len(announcements) <= 1, f"announced more than once: {announcements}"
    assert announcements or warnings, (
        f"resolution said nothing at all:\n{proc.stdout}")


def test_a_manual_path_with_no_ffprobe_anywhere_near_it_still_warns(tmp_path):
    """Pointing FFPROBE_MANUAL_PATH somewhere with no ffprobe in it must
    keep saying so, rather than falling silent."""
    decoy = tmp_path / "ffmpeg.exe"
    decoy.write_text("not a real binary", encoding="utf-8")

    proc = _startup({"FFPROBE_MANUAL_PATH": str(decoy)}, tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "FFPROBE_MANUAL_PATH" in proc.stdout, proc.stdout
    assert "falling back to PATH" in proc.stdout, proc.stdout


@pytest.mark.parametrize("point_at", ["the ffmpeg beside it", "the folder"])
def test_a_manual_path_aimed_at_the_install_finds_ffprobe(tmp_path, point_at):
    """Nobody installs "ffprobe" -- they install ffmpeg. Aiming the
    setting at the ffmpeg program, or at the folder both live in, used to
    be refused with "does not point at ffprobe" even though the install
    was perfectly good; on a machine with nothing on PATH that meant no
    video features at all.

    Uses the real ffprobe from this machine, because the check that
    decides runs the program and reads its banner."""
    real = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
    if not real:
        pytest.skip("no ffprobe on PATH to copy")

    bin_dir = tmp_path / "ffmpeg_install" / "bin"
    bin_dir.mkdir(parents=True)
    copied = bin_dir / os.path.basename(real)
    shutil.copy2(real, copied)
    beside = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    beside.write_text("not a real binary", encoding="utf-8")

    setting = str(beside) if point_at == "the ffmpeg beside it" else str(bin_dir)
    proc = _startup({"FFPROBE_MANUAL_PATH": setting}, tmp_path)

    assert proc.returncode == 0, proc.stderr
    assert "falling back to PATH" not in proc.stdout, proc.stdout
    assert "INFO: ffprobe:" in proc.stdout, proc.stdout
    assert str(bin_dir).replace(os.sep, "/").lower() in \
        proc.stdout.replace(os.sep, "/").lower(), (
        f"resolved to something outside the install it was pointed at:\n"
        f"{proc.stdout}")
