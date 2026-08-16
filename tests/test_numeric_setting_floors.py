"""A number that parses can still be nonsense, and zero is the easy slip.

CONFIGURATION.md promises that an unparseable number warns and falls back.
Zero parses. It was accepted and then broke something a long way from the
setting, with a traceback naming Python rather than the variable:

    BATCH_SIZE=0            range() arg 3 must not be zero -- the scan dies
    MAX_PARALLEL_WORKERS=0  max_workers must be greater than 0 -- same
    THUMBNAIL_WIDTH=0       ZeroDivisionError on every thumbnail

Each is a plausible thing to type when you mean "unlimited" or "default",
which is exactly what makes it worth catching: the person who typed it has
no reason to connect a `range()` error to a batch size.

Values below the floor now warn and fall back, like an unparseable one.

The AI layer's numbers are deliberately NOT floored: several are documented
as meaningful at zero or below -- AI_DAM_FACE_DETECT_MAX_SIDE=0 disables
the cap, AI_DAM_GPU_LAYERS=-1 means all layers -- and the worker already
wraps its own batch in max(1, ...). A blanket floor would have broken
documented behaviour.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _import_with(env_extra, expression):
    env = dict(os.environ, ENABLE_AI_DAM="false", AI_DAM_AUTO_PROVISION="false",
               **env_extra)
    script = ("import sys\n"
              "sys.argv = ['smartgallery.py']\n"
              "import smartgallery\n"
              f"print('VALUE=', {expression})\n")
    return subprocess.run([sys.executable, "-c", script], cwd=_ROOT, env=env,
                          capture_output=True, text=True, timeout=300)


@pytest.fixture()
def gallery_env(tmp_path):
    gallery = tmp_path / "gallery"
    output = tmp_path / "output"
    gallery.mkdir()
    output.mkdir()
    return {"BASE_OUTPUT_PATH": str(output), "BASE_SMARTGALLERY_PATH": str(gallery)}


@pytest.mark.parametrize("name,expression,bad,expected", [
    ("BATCH_SIZE", "smartgallery.BATCH_SIZE", "0", "500"),
    ("BATCH_SIZE", "smartgallery.BATCH_SIZE", "-5", "500"),
    ("MAX_PARALLEL_WORKERS", "smartgallery.MAX_PARALLEL_WORKERS", "0", "None"),
    ("THUMBNAIL_WIDTH", "smartgallery.THUMBNAIL_WIDTH", "0", "300"),
    ("PAGE_SIZE", "smartgallery.PAGE_SIZE", "0", "100"),
    ("SERVER_PORT", "smartgallery.SERVER_PORT", "0", "8189"),
    ("WEBP_ANIMATED_FPS", "smartgallery.WEBP_ANIMATED_FPS", "0", "16.0"),
])
def test_a_value_below_the_floor_falls_back(gallery_env, name, expression,
                                            bad, expected):
    """The regression: these were taken at face value."""
    proc = _import_with(dict(gallery_env, **{name: bad}), expression)

    assert proc.returncode == 0, f"{name}={bad} stopped the import:\n{proc.stderr}"
    assert f"VALUE= {expected}" in proc.stdout, (
        f"{name}={bad} gave {proc.stdout.strip()}, expected the default {expected}")
    assert name in proc.stdout, "the warning does not name the setting"
    assert "minimum" in proc.stdout, "the warning does not say what was wrong"


@pytest.mark.parametrize("name,expression,good", [
    ("BATCH_SIZE", "smartgallery.BATCH_SIZE", "250"),
    ("MAX_PARALLEL_WORKERS", "smartgallery.MAX_PARALLEL_WORKERS", "4"),
    ("THUMBNAIL_WIDTH", "smartgallery.THUMBNAIL_WIDTH", "512"),
    ("PAGE_SIZE", "smartgallery.PAGE_SIZE", "50"),
    # Zero is legitimate here: stream everything, whatever its size.
    ("STREAM_THRESHOLD_MB", "smartgallery.STREAM_THRESHOLD_MB", "0"),
])
def test_a_real_value_is_still_honoured(gallery_env, name, expression, good):
    """The counterpart -- a floor that rejected everything would satisfy
    every test above."""
    proc = _import_with(dict(gallery_env, **{name: good}), expression)

    assert proc.returncode == 0, proc.stderr
    assert f"VALUE= {good}" in proc.stdout, proc.stdout


def test_a_scan_still_runs_with_a_zeroed_batch_size(gallery_env):
    """End to end: BATCH_SIZE=0 used to raise inside the scan, which is
    where the setting is actually used, not where it is read."""
    env = dict(os.environ, ENABLE_AI_DAM="false", AI_DAM_AUTO_PROVISION="false",
               BATCH_SIZE="0", **gallery_env)
    script = (
        "import sys, os\n"
        "sys.argv = ['smartgallery.py']\n"
        "import smartgallery as sg\n"
        "from PIL import Image\n"
        "os.makedirs(sg.SQLITE_CACHE_DIR, exist_ok=True)\n"
        "sg.init_db()\n"
        "Image.new('RGB', (16, 16)).save(os.path.join(sg.BASE_OUTPUT_PATH, 'floor.png'))\n"
        "conn = sg.get_db_connection()\n"
        "sg.full_sync_database(conn)\n"
        "print('ROWS=', conn.execute(\"SELECT COUNT(*) FROM files\").fetchone()[0])\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], cwd=_ROOT, env=env,
                          capture_output=True, text=True, timeout=600)

    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "ROWS= 1" in proc.stdout, proc.stdout
