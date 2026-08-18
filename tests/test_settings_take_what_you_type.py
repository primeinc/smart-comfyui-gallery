"""A setting should accept what people actually type.

Two rules the gallery used to push onto the person configuring it, both
now handled in the code instead.

PATHS. The docs said "Use forward slashes even on Windows", which is the
wrong way round: a Windows user copying a path out of Explorer gets
backslashes, and Explorer's "Copy as path" wraps the whole thing in
quotes. Every spelling of the same folder now normalises to one form.

The form matters more than it looks. A file is identified by its path, so
a canonical form that differed by one character from the one the scan
already computes would re-identify every file in the library and take its
ratings, comments and album membership with it. Measured before changing
anything -- what the folder scan canonicalises to, against what the new
normalisation produces:

    C:/ComfyUI/output          C:/ComfyUI/output    same
    C:\\ComfyUI\\output          C:/ComfyUI/output    same
    C:/ComfyUI/output/         C:/ComfyUI/output    same
    C:\\ComfyUI\\output\\         C:/ComfyUI/output    same
    C:/ComfyUI//output         C:/ComfyUI/output    same
    C:/ComfyUI/./output        C:/ComfyUI/output    same
    C:/ComfyUI/sub/../output   C:/ComfyUI/output    same
    C:/ComfyUI/OUTPUT          C:/ComfyUI/OUTPUT    same

Case is deliberately not folded: two spellings open the same folder on
Windows, but the rows were written with whichever was configured, so
folding it would re-identify the library of anyone whose setting does not
match the disk. looks_like_a_renamed_root covers that instead.

FFMPEG. Nobody installs "ffprobe" -- they install ffmpeg, and ffprobe is
one of the programs in it. The setting is called FFPROBE_MANUAL_PATH, so
people point it at ffmpeg.exe, or at the bin folder, and the gallery used
to refuse and fall back to PATH:

    WARNING: FFPROBE_MANUAL_PATH does not point at ffprobe
    (C:/ffmpeg/bin/ffmpeg.exe); falling back to PATH.

That install was perfectly good. On a machine with nothing on PATH it
meant no video features at all. The old failure message even told people
to point the setting at "the folder holding ffprobe", which the check
then rejected for not being a file.
"""

from __future__ import annotations

import ast
import os

import pytest

import smartgallery

_BACKSLASH = chr(92)


def _canonical_as_the_scan_does(raw):
    """What get_dynamic_folder_config computes for a folder, which is the
    string every file id is built on top of."""
    return os.path.normpath(raw).replace(_BACKSLASH, "/")


def test_the_scan_already_treated_these_spellings_as_one_folder():
    """Control, and the reason normalising at config time is safe rather
    than a migration.

    It uses nothing but os.path, so it holds against the build before this
    change as well as after it. What it establishes: the folder scan
    already collapsed these spellings to a single string, so moving that
    collapse earlier cannot change any file's identity. If this ever stops
    being true, normalising the setting starts re-identifying libraries
    and the tests below are no longer safe."""
    spellings = [
        "C:/ComfyUI/output",
        "C:" + _BACKSLASH + "ComfyUI" + _BACKSLASH + "output",
        "C:/ComfyUI/output/",
        "C:/ComfyUI//output",
        "C:/ComfyUI/./output",
        "C:/ComfyUI/sub/../output",
    ]

    canonical = {_canonical_as_the_scan_does(s) for s in spellings}

    assert canonical == {"C:/ComfyUI/output"}, canonical


@pytest.mark.parametrize(
    "written",
    [
        "C:/ComfyUI/output",
        "C:" + _BACKSLASH + "ComfyUI" + _BACKSLASH + "output",
        "C:/ComfyUI/output/",
        "C:" + _BACKSLASH + "ComfyUI" + _BACKSLASH + "output" + _BACKSLASH,
        "C:/ComfyUI//output",
        "C:/ComfyUI/./output",
        "C:/ComfyUI/sub/../output",
        '"C:' + _BACKSLASH + "ComfyUI" + _BACKSLASH + 'output"',
        "  C:/ComfyUI/output  ",
    ],
)
def test_every_way_of_writing_one_folder_means_one_folder(written):
    """The point of the change: stop making people spell it our way."""
    assert smartgallery.normalize_configured_path(written) == "C:/ComfyUI/output"


@pytest.mark.parametrize(
    "written",
    [
        "C:/ComfyUI/output",
        "C:" + _BACKSLASH + "ComfyUI" + _BACKSLASH + "output",
        "C:/ComfyUI/output/",
        "C:/ComfyUI//output",
        "C:/ComfyUI/sub/../output",
        "C:/ComfyUI/OUTPUT",
    ],
)
def test_normalising_does_not_re_identify_anybody_s_library(written):
    """The check that made this safe to do at all. The normalised form has
    to be the exact string the scan already builds file ids from -- one
    character of difference and every rating in every library moves to a
    file that does not exist."""
    assert smartgallery.normalize_configured_path(written) == _canonical_as_the_scan_does(written)


def test_case_is_left_alone():
    """Deliberate. Folding case would re-identify the library of anyone
    whose setting does not match the disk, which is the damage this is
    supposed to avoid."""
    lower = smartgallery.normalize_configured_path("C:/comfyui/output")
    upper = smartgallery.normalize_configured_path("C:/ComfyUI/OUTPUT")

    assert lower != upper


def test_a_home_relative_path_is_expanded():
    """`~/ComfyUI/output` is the natural thing to write on Linux and a
    macOS, and it used to be taken as a folder literally called ~."""
    expanded = smartgallery.normalize_configured_path("~/ComfyUI/output")

    assert "~" not in expanded, expanded
    assert expanded.endswith("/ComfyUI/output"), expanded


@pytest.mark.parametrize("blank", ["", "   ", None, '""', "''"])
def test_nothing_stays_nothing(blank):
    """normpath('') is '.', which would quietly make the gallery root the
    working directory. A blank has to stay blank so env_or's default
    applies."""
    assert smartgallery.normalize_configured_path(blank) == ""


def test_the_settings_actually_go_through_it(gallery_tree):
    """The helper is only worth having if the settings use it."""

    tree = gallery_tree

    read_with = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        ):
            continue
        if node.value.func.id not in ("env_or", "env_path"):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                read_with[target.id] = node.value.func.id

    for name in (
        "BASE_OUTPUT_PATH",
        "BASE_INPUT_PATH",
        "BASE_SMARTGALLERY_PATH",
        "BASE_MODELS_PATH",
        "LORAS_PATH",
        "CHECKPOINTS_PATH",
        "UNET_PATH",
        "FFPROBE_MANUAL_PATH",
    ):
        assert read_with.get(name) == "env_path", (
            f"{name} is read with {read_with.get(name)}, so whatever the person typed is used as-is"
        )


# --- ffmpeg ---------------------------------------------------------------


def _fake_install(tmp_path, names):
    """An ffmpeg-shaped folder holding programs that answer -version."""
    bin_dir = tmp_path / "ffmpeg" / "bin"
    bin_dir.mkdir(parents=True)
    made = {}
    for name in names:
        target = bin_dir / name
        target.write_bytes(b"not a real program")
        made[name] = str(target)
    return str(tmp_path / "ffmpeg"), str(bin_dir), made


@pytest.mark.parametrize("point_at", ["ffprobe", "ffmpeg", "bin", "install"])
def test_ffprobe_is_found_from_anything_in_the_install(tmp_path, monkeypatch, point_at):
    """The bug: only the exact ffprobe path was accepted."""
    exe = ".exe" if os.name == "nt" else ""
    root, bin_dir, made = _fake_install(tmp_path, [f"ffprobe{exe}", f"ffmpeg{exe}"])

    # Everything in the fake install answers as ffprobe except ffmpeg,
    # so the resolver has to pick the right neighbour rather than the
    # first file it is handed.
    monkeypatch.setattr(smartgallery, "_is_ffprobe", lambda path: os.path.basename(path).lower().startswith("ffprobe"))

    setting = {"ffprobe": made[f"ffprobe{exe}"], "ffmpeg": made[f"ffmpeg{exe}"], "bin": bin_dir, "install": root}[
        point_at
    ]

    found = smartgallery.resolve_ffprobe_from(setting)

    assert found is not None, f"nothing found from {point_at}"
    assert os.path.basename(found).lower().startswith("ffprobe"), found


def test_pointing_straight_at_ffprobe_runs_nothing_else(tmp_path, monkeypatch):
    """Order matters: the named file is tried first, so the common case
    does not start a subprocess per neighbour."""
    exe = ".exe" if os.name == "nt" else ""
    _root, _bin, made = _fake_install(tmp_path, [f"ffprobe{exe}", f"ffmpeg{exe}"])

    tried = []

    def _record(path):
        tried.append(os.path.basename(path).lower())
        return tried[-1].startswith("ffprobe")

    monkeypatch.setattr(smartgallery, "_is_ffprobe", _record)
    smartgallery.resolve_ffprobe_from(made[f"ffprobe{exe}"])

    assert len(tried) == 1, tried


def test_a_folder_with_no_ffmpeg_finds_nothing(tmp_path, monkeypatch):
    """Over-reach guard. Answering with something from a folder that has
    no ffmpeg in it would be worse than answering nothing."""
    monkeypatch.setattr(smartgallery, "_is_ffprobe", lambda path: True)
    empty = tmp_path / "somewhere"
    empty.mkdir()

    assert smartgallery.resolve_ffprobe_from(str(empty)) is None


def test_a_blank_setting_finds_nothing(monkeypatch):
    """Over-reach guard: unset means unset, not 'search the whole disk'."""
    monkeypatch.setattr(smartgallery, "_is_ffprobe", lambda path: True)

    assert smartgallery.resolve_ffprobe_from("") is None
    assert smartgallery.resolve_ffprobe_from(None) is None


def test_ffmpeg_is_not_accepted_as_ffprobe(tmp_path, monkeypatch):
    """The check that has to survive all this. ffmpeg answers -version
    happily and then rejects every ffprobe argument, so an install with
    only ffmpeg in it must come back empty rather than half-working."""
    exe = ".exe" if os.name == "nt" else ""
    _root, bin_dir, made = _fake_install(tmp_path, [f"ffmpeg{exe}"])

    monkeypatch.setattr(smartgallery, "_is_ffprobe", lambda path: os.path.basename(path).lower().startswith("ffprobe"))

    assert smartgallery.resolve_ffprobe_from(made[f"ffmpeg{exe}"]) is None
    assert smartgallery.resolve_ffprobe_from(bin_dir) is None
