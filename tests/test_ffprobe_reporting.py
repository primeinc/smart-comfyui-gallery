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

Each case used to start a fresh interpreter, three seconds of process start
and module loading, to call find_ffprobe_path once and read stdout.
Everything it reads is a module attribute -- FFPROBE_MANUAL_PATH,
FFMPEG_AUTO_DOWNLOAD, and the _FFPROBE_ANNOUNCED memo that makes "once per
run" mean anything -- so monkeypatch sets them and capsys reads the output
back. The memo is reset per case, which a fresh process used to do for
free and is now the fixture's job.
"""

from __future__ import annotations

import os
import shutil

import pytest


@pytest.fixture
def resolving(smartgallery_app, monkeypatch):
    """A gallery whose ffprobe resolution starts from nothing said yet.

    STATE.ffprobe_announced is process state that a fresh interpreter reset for
    free. Without this every case after the first would see silence and
    read it as "announced nothing".
    """
    monkeypatch.setattr(smartgallery_app.STATE, "ffprobe_announced", False)
    # Never fetch a ~170MB build from a test.
    monkeypatch.setattr(smartgallery_app, "FFMPEG_AUTO_DOWNLOAD", False)
    return smartgallery_app


@pytest.fixture
def no_ffprobe_anywhere(resolving, monkeypatch, tmp_path):
    """An environment with nothing named ffprobe on PATH."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    monkeypatch.setattr(resolving, "FFPROBE_MANUAL_PATH", str(tmp_path / "nowhere" / "ffprobe.exe"))
    return resolving


def test_a_missing_ffprobe_says_what_stops_working(no_ffprobe_anywhere, capsys):
    """The regression: 'ffprobe not found. Video metadata analysis will be
    disabled.' understated it -- thumbnails, waveforms and the metadata
    stripping go too."""
    no_ffprobe_anywhere.find_ffprobe_path()

    out = capsys.readouterr().out
    assert "ffprobe not found" in out, out
    for consequence in ("thumbnail", "waveform", "stripping"):
        assert consequence in out.lower(), f"the warning does not mention {consequence}:\n{out}"
    assert "FFPROBE_MANUAL_PATH" in out, "it does not say how to fix it"


def test_resolution_is_announced_once(resolving, capsys):
    """Whatever it resolves, it says so -- and does not repeat itself on
    every later call, of which the scan makes many."""
    resolving.find_ffprobe_path()
    resolving.find_ffprobe_path()
    resolving.find_ffprobe_path()

    lines = capsys.readouterr().out.splitlines()
    announcements = [line for line in lines if line.startswith("INFO: ffprobe:")]
    warnings = [line for line in lines if "ffprobe not found" in line]

    assert len(announcements) <= 1, f"announced more than once: {announcements}"
    assert announcements or warnings, "resolution said nothing at all"


def test_a_manual_path_with_no_ffprobe_anywhere_near_it_still_warns(resolving, monkeypatch, capsys, tmp_path):
    """Pointing FFPROBE_MANUAL_PATH somewhere with no ffprobe in it must
    keep saying so, rather than falling silent."""
    decoy = tmp_path / "ffmpeg.exe"
    decoy.write_text("not a real binary", encoding="utf-8")
    monkeypatch.setattr(resolving, "FFPROBE_MANUAL_PATH", str(decoy))

    resolving.find_ffprobe_path()

    out = capsys.readouterr().out
    assert "FFPROBE_MANUAL_PATH" in out, out
    assert "falling back to PATH" in out, out


@pytest.mark.parametrize("point_at", ["the ffmpeg beside it", "the folder"])
def test_a_manual_path_aimed_at_the_install_finds_ffprobe(resolving, monkeypatch, capsys, tmp_path, point_at):
    """Nobody installs "ffprobe" -- they install ffmpeg. Aiming the setting
    at the ffmpeg program, or at the folder both live in, used to be
    refused with "does not point at ffprobe" even though the install was
    perfectly good; on a machine with nothing on PATH that meant no video
    features at all.

    Uses the real ffprobe from this machine, because the check that decides
    runs the program and reads its banner."""
    real = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
    if not real:
        pytest.skip("no ffprobe on PATH to copy")

    bin_dir = tmp_path / "ffmpeg_install" / "bin"
    bin_dir.mkdir(parents=True)
    shutil.copy2(real, bin_dir / os.path.basename(real))
    beside = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    beside.write_text("not a real binary", encoding="utf-8")

    setting = str(beside) if point_at == "the ffmpeg beside it" else str(bin_dir)
    monkeypatch.setattr(resolving, "FFPROBE_MANUAL_PATH", setting)

    resolving.find_ffprobe_path()

    out = capsys.readouterr().out
    assert "falling back to PATH" not in out, out
    assert "INFO: ffprobe:" in out, out
    assert str(bin_dir).replace(os.sep, "/").lower() in out.replace(os.sep, "/").lower(), (
        f"resolved to something outside the install it was pointed at:\n{out}"
    )
