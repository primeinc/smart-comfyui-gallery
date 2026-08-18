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

Every case here ran in its own interpreter, importing the whole module to
read one constant back: three seconds of process start and import per
parameter. Nothing about the claim needed that. It splits in two, and both
halves run in this process:

  * what a bad value DOES is env_num's behaviour, and env_num reads the
    environment when it is called -- so monkeypatch.setenv drives it and
    capsys reads the warning back
    (pytest doc/en/how-to/monkeypatch.rst:267-339, capture-stdout-stderr.rst:112-142)
  * that each setting is WIRED to env_num with the right floor is a
    property of the source, read off the shared parse of smartgallery.py

The end-to-end case at the bottom keeps its scan, because "the setting
survives where it is used, not where it is read" is the half that a unit
call cannot show.
"""

from __future__ import annotations

import ast

import pytest

import smartgallery

# name -> (default, floor). The floors this file exists to defend.
_FLOORED = {
    "SERVER_PORT": (8189, 1),
    "THUMBNAIL_WIDTH": (300, 1),
    "PAGE_SIZE": (100, 1),
    "BATCH_SIZE": (500, 1),
    "MAX_PARALLEL_WORKERS": (None, 1),
    "WEBP_ANIMATED_FPS": (16.0, 0.1),
}


def _env_num_call(tree, setting):
    """The `env_num(...)` call that defines `setting` at module scope."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if setting not in names:
            continue
        call = node.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "env_num":
            return call
    return None


@pytest.mark.parametrize("setting", sorted(_FLOORED))
def test_every_floored_setting_is_read_through_env_num(gallery_tree, setting):
    """The wiring. A floor that is not passed cannot be enforced, and this
    is what the old subprocess was really proving by reading the constant
    back out of a fresh import."""
    call = _env_num_call(gallery_tree, setting)

    assert call is not None, (
        f"{setting} is no longer assigned from env_num(...) at module scope, so nothing applies a floor to it"
    )
    floors = [kw for kw in call.keywords if kw.arg == "minimum"]
    assert floors, f"{setting} is read by env_num with no minimum="
    assert floors[0].value.value == _FLOORED[setting][1], f"{setting}'s floor moved to {floors[0].value.value}"


@pytest.mark.parametrize(
    ("name", "bad"),
    [
        ("BATCH_SIZE", "0"),
        ("BATCH_SIZE", "-5"),
        ("MAX_PARALLEL_WORKERS", "0"),
        ("THUMBNAIL_WIDTH", "0"),
        ("PAGE_SIZE", "0"),
        ("SERVER_PORT", "0"),
        ("WEBP_ANIMATED_FPS", "0"),
    ],
)
def test_a_value_below_the_floor_falls_back(monkeypatch, capsys, name, bad):
    """The regression: these were taken at face value."""
    default, floor = _FLOORED[name]
    cast = float if isinstance(default, float) else int
    monkeypatch.setenv(name, bad)

    value = smartgallery.env_num(name, default, cast, minimum=floor)
    printed = capsys.readouterr().out

    assert value == default, f"{name}={bad} gave {value!r}, expected the default {default!r}"
    assert name in printed, "the warning does not name the setting"
    assert "minimum" in printed, "the warning does not say what was wrong"


@pytest.mark.parametrize(
    ("name", "good"),
    [
        ("BATCH_SIZE", 250),
        ("MAX_PARALLEL_WORKERS", 4),
        ("THUMBNAIL_WIDTH", 512),
        ("PAGE_SIZE", 50),
    ],
)
def test_a_real_value_is_still_honoured(monkeypatch, capsys, name, good):
    """The counterpart -- a floor that rejected everything would satisfy
    every test above."""
    default, floor = _FLOORED[name]
    monkeypatch.setenv(name, str(good))

    value = smartgallery.env_num(name, default, minimum=floor)

    assert value == good, f"{name}={good} was not honoured, got {value!r}"
    assert capsys.readouterr().out == "", "a valid value should warn about nothing"


def test_zero_is_still_allowed_where_it_means_something(monkeypatch, capsys):
    """STREAM_THRESHOLD_MB=0 means stream everything, whatever its size. A
    blanket floor would have taken that away."""
    monkeypatch.setenv("STREAM_THRESHOLD_MB", "0")

    value = smartgallery.env_num("STREAM_THRESHOLD_MB", 20)

    assert value == 0, "zero no longer reaches the setting that documents it"
    assert capsys.readouterr().out == ""


def test_a_scan_still_runs_with_a_zeroed_batch_size(smartgallery_app, tmp_path, monkeypatch):
    """End to end: BATCH_SIZE=0 used to raise inside the scan, which is
    where the setting is actually used, not where it is read.

    Runs against the imported gallery with BATCH_SIZE patched to what
    env_num would have produced, so the scan sees the same value a bad
    setting leaves behind."""
    monkeypatch.setattr(smartgallery_app, "BATCH_SIZE", smartgallery_app.env_num("BATCH_SIZE", 500, minimum=1))

    from PIL import Image

    picture = tmp_path / "floor.png"
    Image.new("RGB", (16, 16)).save(picture)
    monkeypatch.setattr(smartgallery_app, "BASE_OUTPUT_PATH", str(tmp_path))

    # The session database is shared, so this counts only what THIS scan
    # indexed and removes it again. Counting by name alone saw the previous
    # run's row too and reported 2.
    here = str(tmp_path).replace("\\", "/")
    conn = smartgallery_app.get_db_connection()
    try:
        smartgallery_app.full_sync_database(conn)
        rows = conn.execute(
            "SELECT COUNT(*) FROM files WHERE name = ? AND REPLACE(path, '\\', '/') LIKE ?", ("floor.png", here + "%")
        ).fetchone()[0]
        conn.execute("DELETE FROM files WHERE REPLACE(path, '\\', '/') LIKE ?", (here + "%",))
        conn.commit()
    finally:
        conn.close()

    assert rows == 1, "the scan did not index the file with a zeroed batch size"
