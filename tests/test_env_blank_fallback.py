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


# --- numeric knobs --------------------------------------------------------
# These are read at module scope, so int('') used to raise ValueError during
# import: the gallery refused to start, with a traceback that never named
# the offending variable.

@pytest.mark.parametrize("attr,name,default", [
    ("THUMBNAIL_WIDTH", "THUMBNAIL_WIDTH", 300),
    ("PAGE_SIZE", "PAGE_SIZE", 100),
    ("BATCH_SIZE", "BATCH_SIZE", 500),
    ("STREAM_THRESHOLD_MB", "STREAM_THRESHOLD_MB", 20),
    ("SERVER_PORT", "SERVER_PORT", 8189),
])
def test_numeric_knobs_survive_blank_and_garbage(smartgallery_app, monkeypatch,
                                                  attr, name, default):
    del attr  # the constant is already bound; env_num is what we exercise
    for bad in ("", "   ", "not-a-number", "12abc"):
        monkeypatch.setenv(name, bad)
        assert smartgallery_app.env_num(name, default) == default
    monkeypatch.setenv(name, " 42 ")
    assert smartgallery_app.env_num(name, default) == 42
    monkeypatch.delenv(name)
    assert smartgallery_app.env_num(name, default) == default


def test_env_num_supports_floats(smartgallery_app, monkeypatch):
    monkeypatch.setenv("WEBP_ANIMATED_FPS", "")
    assert smartgallery_app.env_num("WEBP_ANIMATED_FPS", 16.0, float) == 16.0
    monkeypatch.setenv("WEBP_ANIMATED_FPS", "23.5")
    assert smartgallery_app.env_num("WEBP_ANIMATED_FPS", 16.0, float) == 23.5


def test_ai_config_numeric_knobs_survive_blanks(monkeypatch, tmp_path):
    from smartgallery_ai import AIConfig

    for name in ("AI_DAM_NEAR_DUP_DISTANCE", "AI_DAM_FACE_MIN_PX",
                 "AI_DAM_SIMILAR_K", "AI_DAM_FACE_DETECT_MAX_SIDE",
                 "AI_DAM_FACE_CLUSTER_THRESHOLD"):
        monkeypatch.setenv(name, "")
    cfg = AIConfig.from_env(str(tmp_path), str(tmp_path / "db.sqlite"))
    assert (cfg.near_dup_max_distance, cfg.face_min_px, cfg.similar_default_k,
            cfg.face_detect_max_side) == (8, 24, 24, 1600)
    # None means "use the embedder's own default" -- a blank must mean that
    # too, not float('').
    assert cfg.face_cluster_threshold is None


def test_ai_config_blank_backend_selector_stays_auto(monkeypatch, tmp_path):
    """A blank selector must not become "", which resolves to no backend."""
    from smartgallery_ai import AIConfig

    monkeypatch.setenv("AI_DAM_SEMANTIC_BACKEND", "")
    monkeypatch.setenv("AI_DAM_CRITIC_BACKEND", "   ")
    cfg = AIConfig.from_env(str(tmp_path), str(tmp_path / "db.sqlite"))
    assert cfg.semantic_backend == "auto" and cfg.critic_backend == "auto"


def test_enable_ai_dam_blank_does_not_silently_disable_the_layer(monkeypatch, tmp_path):
    from smartgallery_ai import AIConfig

    monkeypatch.setenv("ENABLE_AI_DAM", "")
    assert AIConfig.from_env(str(tmp_path), str(tmp_path / "db.sqlite")).enabled is True
    monkeypatch.setenv("ENABLE_AI_DAM", "false")
    assert AIConfig.from_env(str(tmp_path), str(tmp_path / "db.sqlite")).enabled is False


# --- yes/no flags ---------------------------------------------------------
# `os.environ.get(name, "true").lower() == "true"` made a BLANK read as
# False, so clearing GENERATE_THUMBNAILS in the launcher silently turned
# thumbnails off instead of restoring the documented default.

@pytest.mark.parametrize("default", [True, False])
def test_env_flag_blank_keeps_the_default(smartgallery_app, monkeypatch, default):
    for blank in ("", "   "):
        monkeypatch.setenv("SG_FLAG_PROBE", blank)
        assert smartgallery_app.env_flag("SG_FLAG_PROBE", default) is default
    monkeypatch.delenv("SG_FLAG_PROBE")
    assert smartgallery_app.env_flag("SG_FLAG_PROBE", default) is default


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("Yes", True), ("on", True),
    ("0", False), ("false", False), ("No", False), ("off", False),
    (" true ", True),
])
def test_env_flag_recognized_words(smartgallery_app, monkeypatch, value, expected):
    monkeypatch.setenv("SG_FLAG_PROBE", value)
    assert smartgallery_app.env_flag("SG_FLAG_PROBE", not expected) is expected


def test_env_flag_unrecognized_word_keeps_default(smartgallery_app, monkeypatch):
    """A typo must not silently mean False."""
    monkeypatch.setenv("SG_FLAG_PROBE", "maybe")
    assert smartgallery_app.env_flag("SG_FLAG_PROBE", True) is True


def test_env_num_none_default_for_optional_settings(smartgallery_app, monkeypatch):
    """MAX_PARALLEL_WORKERS documents "leave empty for all cores"; blank and
    garbage must both reach the auto path rather than raising."""
    for value in ("", "   ", "lots"):
        monkeypatch.setenv("MAX_PARALLEL_WORKERS", value)
        assert smartgallery_app.env_num("MAX_PARALLEL_WORKERS", None) is None
    monkeypatch.setenv("MAX_PARALLEL_WORKERS", "4")
    assert smartgallery_app.env_num("MAX_PARALLEL_WORKERS", None) == 4


def test_configuration_doc_covers_the_user_facing_env_vars():
    """docs/CONFIGURATION.md is the reference; a setting users can set must
    appear in it. Derived from the source so a new knob is caught here."""
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc = open(os.path.join(root, "docs", "CONFIGURATION.md"),
               encoding="utf-8").read()
    src = open(os.path.join(root, "smartgallery.py"), encoding="utf-8").read()
    pattern = re.compile(
        r"""\b(?:env_or|env_num|env_flag)\s*\(\s*['"]([A-Z][A-Z0-9_]{2,})['"]""")
    # PATH/DISPLAY-style environment probes are not settings; everything the
    # app reads through its own config helpers is.
    referenced = set(pattern.findall(src))
    missing = sorted(v for v in referenced if v not in doc)
    assert not missing, f"undocumented settings in docs/CONFIGURATION.md: {missing}"


# --- DELETE_TO (trash instead of permanent deletion) ----------------------
# The whole feature crashed on first use: its validation block prints
# coloured diagnostics at import time, but `class Colors` was defined
# further down the file, so creating the trash folder -- the ordinary first
# run -- raised NameError before the app could start. Only an install whose
# <DELETE_TO>/SmartGallery folder already existed ever booted.

def test_colors_is_defined_before_the_configuration_block_uses_it():
    """Ordering guard: the config block runs at import, so anything it
    references must already exist above it."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "smartgallery.py"), encoding="utf-8").read()
    definition = src.index("class Colors:")
    first_use = src.index("Colors.RED")
    assert definition < first_use, (
        "class Colors is defined after its first use; the DELETE_TO "
        "validation paths will raise NameError at import")


def test_delete_to_moves_files_to_trash_without_overwriting(tmp_path):
    """safe_delete_file must relocate rather than remove, and two files of
    the same name deleted in the same second must both survive.

    Run in a subprocess: DELETE_TO is resolved at module scope, so this
    needs a fresh interpreter with that environment -- and reimporting
    smartgallery inside the test session would hand other tests a
    different module object than the session fixture holds.
    """
    import subprocess
    import sys

    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    gallery = tmp_path / "gallery"
    gallery.mkdir()

    script = """
import os, sys
sys.argv = ['smartgallery.py']
import smartgallery as sg
assert sg.TRASH_FOLDER and os.path.isdir(sg.TRASH_FOLDER), 'trash folder missing'
victim = os.path.join(sg.BASE_OUTPUT_PATH, 'gone.png')
open(victim, 'wb').write(b'first')
sg.safe_delete_file(victim)
assert not os.path.exists(victim), 'file was not removed from the gallery'
assert len(os.listdir(sg.TRASH_FOLDER)) == 1, 'file did not arrive in the trash'
open(victim, 'wb').write(b'second')
sg.safe_delete_file(victim)
n = len(os.listdir(sg.TRASH_FOLDER))
assert n == 2, f'a same-name delete overwrote the first ({n} file(s) in trash)'
print('OK')
"""
    env = dict(os.environ,
               DELETE_TO=str(trash_root),
               BASE_OUTPUT_PATH=str(gallery),
               BASE_SMARTGALLERY_PATH=str(gallery),
               ENABLE_AI_DAM="false",
               AI_DAM_AUTO_PROVISION="false")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, (
        f"DELETE_TO first-run path failed:\n{proc.stdout}\n{proc.stderr}")
    assert "OK" in proc.stdout


def test_delete_to_covers_folder_deletion_too(tmp_path):
    """Deleting a FOLDER used to call shutil.rmtree unconditionally, so an
    install configured for recoverable deletes still lost a whole directory
    of media permanently. A link is still only unlinked -- that destroys
    nothing and must not relocate its target."""
    import subprocess
    import sys

    trash_root = tmp_path / "trash"
    trash_root.mkdir()
    gallery = tmp_path / "gallery"
    gallery.mkdir()

    script = """
import os, sys
sys.argv = ['smartgallery.py']
import smartgallery as sg

album = os.path.join(sg.BASE_OUTPUT_PATH, 'album')
os.makedirs(album)
for name in ('a.png', 'b.png'):
    open(os.path.join(album, name), 'wb').write(b'x')

sg.safe_delete_tree(album)
assert not os.path.exists(album), 'folder was not removed from the gallery'

entries = os.listdir(sg.TRASH_FOLDER)
assert len(entries) == 1, f'expected one trashed folder, found {entries}'
recovered = os.path.join(sg.TRASH_FOLDER, entries[0])
assert sorted(os.listdir(recovered)) == ['a.png', 'b.png'], 'contents were lost'

# Without DELETE_TO the old behaviour stands: gone for good.
sg.DELETE_TO = None
sg.TRASH_FOLDER = None
album2 = os.path.join(sg.BASE_OUTPUT_PATH, 'album2')
os.makedirs(album2)
open(os.path.join(album2, 'c.png'), 'wb').write(b'x')
sg.safe_delete_tree(album2)
assert not os.path.exists(album2)
assert len(os.listdir(os.path.dirname(recovered))) == 1, 'unexpected new trash entry'
print('OK')
"""
    env = dict(os.environ,
               DELETE_TO=str(trash_root),
               BASE_OUTPUT_PATH=str(gallery),
               BASE_SMARTGALLERY_PATH=str(gallery),
               ENABLE_AI_DAM="false",
               AI_DAM_AUTO_PROVISION="false")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run([sys.executable, "-c", script], cwd=root, env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "OK" in proc.stdout


# --- test-suite safety ----------------------------------------------------

def test_suite_paths_are_confined_to_a_temp_directory(smartgallery_app):
    """The suite creates files in the gallery root, scans it end to end,
    and deletes rows. conftest used to `setdefault` these paths, so anyone
    with BASE_OUTPUT_PATH exported -- which is everyone who runs the
    gallery, since run_smartgallery.bat sets it -- had the tests operate on
    their real library. They are forced to a temp directory now."""
    import tempfile

    tmp_root = os.path.realpath(tempfile.gettempdir())
    for attr in ("BASE_OUTPUT_PATH", "BASE_SMARTGALLERY_PATH", "BASE_INPUT_PATH"):
        resolved = os.path.realpath(getattr(smartgallery_app, attr))
        assert resolved.startswith(tmp_root), (
            f"{attr} points outside the temp directory ({resolved}); a test "
            f"run could modify a real collection")


def test_suite_does_not_inherit_a_real_trash_folder(smartgallery_app):
    """An inherited DELETE_TO would scatter deleted test files through the
    developer's trash instead of the temp tree."""
    configured = smartgallery_app.DELETE_TO
    if configured:
        import tempfile
        assert os.path.realpath(configured).startswith(
            os.path.realpath(tempfile.gettempdir()))
