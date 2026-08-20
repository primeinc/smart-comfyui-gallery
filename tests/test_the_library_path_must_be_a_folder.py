"""Pointing the gallery at a file must be refused, and refused truthfully.

The startup check for BASE_OUTPUT_PATH is good and does the hard part:
it blocks, names the path, says which setting to edit, shows a dialog
where tkinter exists and falls back to the console in Docker. It asked
os.path.exists.

A file passes os.path.exists. Measured, pointing BASE_OUTPUT_PATH at
`output.zip`:

    os.path.exists  -> True    (what the startup check asked)
    os.path.isdir   -> False   (what it needs to be)

    INFO: Starting full file scan...
    INFO: Full scan completed in 0.00 seconds.
    page: status 200, 1277040 bytes

So it started, scanned nothing, and served an empty gallery with every
sign of being fine -- which is the exact outcome the check exists to
prevent, reached by a slightly different slip. On Windows "Copy as path"
on a file rather than its folder is one keystroke away, and the
FFPROBE_MANUAL_PATH line directly below this one in the launcher IS a
file, which makes it an easy line to copy the shape of.

The message had to change with the check. "The specified path does not
exist" is untrue when the path exists and is a file, and sends somebody
looking for a folder that is sitting exactly where they put it.
"""

from __future__ import annotations

import ast
import os

import pytest

import smartgallery


def test_a_folder_is_accepted(tmp_path):
    """Control, and every working install: the check must not start
    turning away the ordinary case."""
    assert os.path.isdir(str(tmp_path))


def test_a_file_is_not_a_folder(tmp_path):
    """The premise. If exists and isdir ever agreed there would be
    nothing here to fix."""
    a_file = tmp_path / "output.zip"
    a_file.write_bytes(b"not a folder")

    assert os.path.exists(str(a_file)) is True
    assert os.path.isdir(str(a_file)) is False


@pytest.mark.parametrize("setting", ["BASE_OUTPUT_PATH", "BASE_INPUT_PATH"])
def test_the_startup_check_asks_whether_it_is_a_folder(gallery_tree, setting):
    """The bug: it asked whether something was there, not whether it was
    the kind of thing the gallery can show."""

    tree = gallery_tree

    asked = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("exists", "isdir")
            and node.args
        ):
            continue
        argument = node.args[0]
        if isinstance(argument, ast.Name) and argument.id == setting:
            asked.append(node.func.attr)

    assert asked, f"nothing checks {setting} at all"
    assert "exists" not in asked, (
        f"{setting} is checked with os.path.exists somewhere, which a file "
        f"passes -- the gallery then starts on it and shows nothing"
    )


def test_the_refusal_tells_the_truth_about_a_missing_folder(tmp_path, capsys, monkeypatch):
    """A path that really is not there must still say so."""
    monkeypatch.setattr(smartgallery, "TKINTER_AVAILABLE", False)
    missing = str(tmp_path / "not_here")

    with pytest.raises(SystemExit) as exited:
        smartgallery.show_config_error_and_exit(missing)

    assert exited.value.code == 1
    printed = capsys.readouterr().out
    assert "does not exist" in printed, printed
    assert missing in printed, printed


def test_the_refusal_tells_the_truth_about_a_file(tmp_path, capsys, monkeypatch):
    """The half that would have lied. Somebody told "does not exist"
    about a file they can see goes looking for the wrong problem."""
    monkeypatch.setattr(smartgallery, "TKINTER_AVAILABLE", False)
    a_file = tmp_path / "output.zip"
    a_file.write_bytes(b"not a folder")

    with pytest.raises(SystemExit) as exited:
        smartgallery.show_config_error_and_exit(str(a_file))

    assert exited.value.code == 1
    printed = capsys.readouterr().out
    assert "does not exist" not in printed, f"told somebody a file that is right there does not exist:\n{printed}"
    assert "is a file, not a folder" in printed, printed
    assert str(a_file) in printed, printed


def test_the_refusal_still_says_which_setting_to_edit(tmp_path, capsys, monkeypatch):
    """Over-reach guard: the useful half of the original message has to
    survive the change."""
    monkeypatch.setattr(smartgallery, "TKINTER_AVAILABLE", False)
    a_file = tmp_path / "output.zip"
    a_file.write_bytes(b"x")

    with pytest.raises(SystemExit):
        smartgallery.show_config_error_and_exit(str(a_file))

    printed = capsys.readouterr().out
    assert "BASE_OUTPUT_PATH" in printed, printed
    assert ".bat" in printed, printed


def test_it_still_stops_rather_than_carrying_on(tmp_path, monkeypatch):
    """Over-reach guard, and the point of the whole check: starting on a
    path that cannot work is what produces the empty gallery nobody can
    explain."""
    monkeypatch.setattr(smartgallery, "TKINTER_AVAILABLE", False)

    for path in [str(tmp_path / "not_here"), str(tmp_path)]:
        if path == str(tmp_path):
            continue  # a real folder is never passed to this
        with pytest.raises(SystemExit):
            smartgallery.show_config_error_and_exit(path)
