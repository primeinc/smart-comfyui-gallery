"""A blank environment variable must read as "unset", not as "".

`set "BASE_OUTPUT_PATH="` in a .bat file, `export BASE_OUTPUT_PATH=` in a
shell, and an empty field in a Docker/Unraid template all DEFINE the
variable with an empty value. `os.environ.get(name, default)` then returns
"" -- the default never applies -- so clearing a path in the shipped
launcher template (the obvious way to say "just use the default") pointed
the gallery root at "" and scattered cache directories, the database, and
multi-GB model weights into whatever the working directory happened to be.
"""

from __future__ import annotations

import importlib
import os

import pytest


@pytest.mark.parametrize("name,expected_default", [
    ("BASE_OUTPUT_PATH", "C:/ComfyUI/output"),
    ("BASE_INPUT_PATH", "C:/ComfyUI/input"),
    ("FFPROBE_MANUAL_PATH", "C:/ffmpeg/bin/ffprobe.exe"),
])
def test_env_or_treats_blank_as_unset(smartgallery_app, monkeypatch, name, expected_default):
    monkeypatch.setenv(name, "")
    assert smartgallery_app.env_or(name, expected_default) == expected_default
    monkeypatch.setenv(name, "   ")          # whitespace-only is still blank
    assert smartgallery_app.env_or(name, expected_default) == expected_default
    monkeypatch.setenv(name, " D:/real/path ")   # a real value wins, trimmed
    assert smartgallery_app.env_or(name, expected_default) == "D:/real/path"
    monkeypatch.delenv(name)
    assert smartgallery_app.env_or(name, expected_default) == expected_default


def test_plain_environ_get_would_have_returned_blank(monkeypatch):
    """The bug this guards, stated as a control: without the helper the
    default is silently skipped."""
    monkeypatch.setenv("SG_BLANK_PROBE", "")
    assert os.environ.get("SG_BLANK_PROBE", "fallback") == ""


def test_ai_config_blank_dirs_fall_back_under_base_path(monkeypatch, tmp_path):
    """Blank AI_DAM_* directories must land under the gallery root, not in
    the process working directory -- these hold the vector cache and
    multi-GB model weights."""
    from smartgallery_ai import AIConfig

    monkeypatch.setenv("AI_DAM_MODELS_DIR", "")
    monkeypatch.setenv("AI_DAM_CACHE_DIR", "   ")
    cfg = AIConfig.from_env(str(tmp_path), str(tmp_path / "db.sqlite"))
    assert cfg.models_dir == os.path.join(str(tmp_path), ".AImodels")
    assert cfg.cache_dir == os.path.join(str(tmp_path), ".ai_cache")

    monkeypatch.setenv("AI_DAM_MODELS_DIR", str(tmp_path / "elsewhere"))
    cfg = AIConfig.from_env(str(tmp_path), str(tmp_path / "db.sqlite"))
    assert cfg.models_dir == str(tmp_path / "elsewhere")


def test_launcher_template_has_no_personal_paths():
    """run_smartgallery.bat ships to everyone: it must not carry one
    developer's machine paths, and its placeholders must be values a user
    replaces rather than blanks that used to break startup."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "run_smartgallery.bat")
    if not os.path.exists(path):
        pytest.skip("no launcher template in this checkout")
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    lowered = text.lower()
    for marker in ("users/will", "users\\will", "!comfy-output"):
        assert marker not in lowered, (
            f"launcher template contains a personal path ({marker!r})")


# --- ffprobe identity -----------------------------------------------------
# Exit status cannot distinguish ffprobe from ffmpeg: both answer
# `-version` successfully. The launcher template shipped
# FFPROBE_MANUAL_PATH aimed at ffmpeg.exe, which passed the old check and
# then failed every metadata call, because ffmpeg rejects ffprobe's args.

def _fake_run(banner: bytes):
    from types import SimpleNamespace

    def run(cmd, **kwargs):
        return SimpleNamespace(stdout=banner, stderr=b"", returncode=0)
    return run


def test_ffprobe_probe_rejects_an_ffmpeg_binary(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app.subprocess, "run",
                        _fake_run(b"ffmpeg version 2025-07-21 Copyright (c) 2000"))
    assert smartgallery_app._is_ffprobe("C:/ffmpeg/bin/ffmpeg.exe") is False


def test_ffprobe_probe_accepts_a_real_ffprobe(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app.subprocess, "run",
                        _fake_run(b"ffprobe version 2025-07-21 Copyright (c) 2007"))
    assert smartgallery_app._is_ffprobe("C:/ffmpeg/bin/ffprobe.exe") is True


def test_ffprobe_probe_survives_a_missing_binary(smartgallery_app, monkeypatch):
    def boom(*_a, **_k):
        raise FileNotFoundError("nope")
    monkeypatch.setattr(smartgallery_app.subprocess, "run", boom)
    assert smartgallery_app._is_ffprobe("nothing-here") is False


def test_launcher_template_points_at_ffprobe():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "run_smartgallery.bat")
    if not os.path.exists(path):
        pytest.skip("no launcher template in this checkout")
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "FFPROBE_MANUAL_PATH" in line and "=" in line:
                value = line.split("=", 1)[1].strip().strip('"')
                if value:
                    assert "ffprobe" in value.lower(), (
                        f"FFPROBE_MANUAL_PATH points at {value!r}, not ffprobe")
