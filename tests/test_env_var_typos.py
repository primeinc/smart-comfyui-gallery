"""A misspelt setting NAME is the quietest way to configure nothing.

A mistyped value at least reaches a parser. A mistyped name never reaches
anything: the variable is not read, the default applies, and there is no
error to notice. `BASE_OUTPUT_PATH` with a letter missing shows the default
folder rather than the library, which reads as "the gallery cannot see my
files" -- and nothing on screen connects that to the typo.

The gallery cannot know what someone meant to set, but it can notice a name
that is one letter away from a real one and say so. Only close matches are
reported: warning on anything that merely starts with a familiar prefix
would fire on other programs' variables, ComfyUI's own among them, and a
warning that cries wolf is worth less than none.

Never fatal. CONFIGURATION.md states that nothing in the environment stops
the app from starting, and that stays true.
"""

from __future__ import annotations

import os
import pathlib
import re

import pytest

import smartgallery


@pytest.mark.parametrize(
    ("typo", "expected"),
    [
        ("BASE_OUTPUT_PAT", "BASE_OUTPUT_PATH"),
        ("BASE_OUTPUT_PATHH", "BASE_OUTPUT_PATH"),
        ("BASE_OUPUT_PATH", "BASE_OUTPUT_PATH"),
        ("ENABLE_AI_DM", "ENABLE_AI_DAM"),
        ("ENABLE_AI_DAMM", "ENABLE_AI_DAM"),
        ("GENERATE_THUMBNAIL", "GENERATE_THUMBNAILS"),
        ("DELETE_T0", "DELETE_TO"),
        ("SERVER_PORTS", "SERVER_PORT"),
        ("ADMIN_PASSWORDS", "ADMIN_PASSWORD"),
    ],
)
def test_a_near_miss_is_noticed(typo, expected):
    """The regression: these were read by nothing and mentioned by nobody."""
    found = smartgallery.find_misspelt_env_vars({typo: "value"})

    assert found == [(typo, expected)], f"{typo} was not recognised as a misspelling of {expected}: {found}"


def test_a_correct_name_is_not_flagged():
    """Control: the real names must never be reported, or every start-up
    would carry warnings and nobody would read them."""
    every_real_name = dict.fromkeys(smartgallery.KNOWN_ENV_VARS, "x")

    assert smartgallery.find_misspelt_env_vars(every_real_name) == []


@pytest.mark.parametrize(
    "unrelated",
    [
        "PATH",
        "HOME",
        "USERPROFILE",
        "TEMP",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "COMFYUI_PORT",  # ComfyUI's own, not ours
        "CUDA_VISIBLE_DEVICES",
        "PROCESSOR_ARCHITECTURE",
        "NUMBER_OF_PROCESSORS",
        # Found set on a real machine during this work, and reported as a
        # misspelling of AI_DAM_MODELS_DIR at the cutoff first chosen. It is
        # why the cutoff is measured rather than picked.
        "PAI_MODEL_DIR",
        "AI_MODELS_DIR",
    ],
)
def test_other_programs_variables_are_left_alone(unrelated):
    """The cost of a false positive is a warning about someone else's
    setting, which teaches people to ignore warnings."""
    assert smartgallery.find_misspelt_env_vars({unrelated: "x"}) == []


def test_the_known_list_matches_what_the_code_reads():
    """The list is only useful while it is complete: a setting added without
    being listed here would be reported as a misspelling of something else."""

    root = pathlib.Path(smartgallery.__file__).resolve().parent
    pattern = re.compile(
        # \s* after the paren on purpose. Without it a call whose name sat on
        # the next line was invisible, and AI_DAM_FACE_CLUSTER_THRESHOLD sat
        # that way -- read, documented, and absent from KNOWN_ENV_VARS, so
        # anyone setting it correctly was told it was a misspelling. Whether
        # a setting is detected must not depend on where the line wraps.
        r"""(?:os\.environ\.get\(|os\.getenv\(|env_or\(|env_num\(|env_flag\(
            |env_path\(
            |_env_str\(|_env_num\(|_env_bool\(|ENV_MODEL\w*\s*=\s*)
            \s*["']([A-Z][A-Z0-9_]{2,})["']""",
        re.VERBOSE,
    )
    skip = {"PATH", "DISPLAY", "HOME", "TEMP", "TMP", "USERPROFILE"}

    # Pruned during the walk rather than filtered afterwards: rglob descends
    # into .venv and .AImodels in full before anything is discarded, which is
    # most of a second to read forty files.
    not_the_app = {".venv", "tests", "benchmarks", "probes", "vendor", ".git", "__pycache__", ".AImodels"}

    read = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in not_the_app]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            text = (pathlib.Path(dirpath) / name).read_text(encoding="utf-8")
            for match in pattern.finditer(text):
                if match.group(1) not in skip:
                    read.add(match.group(1))

    listed = set(smartgallery.KNOWN_ENV_VARS)

    assert read - listed == set(), (
        f"settings the code reads but KNOWN_ENV_VARS omits: {sorted(read - listed)}. "
        f"Add them, or a user setting one correctly will be told it is a typo."
    )
    assert listed - read == set(), f"listed in KNOWN_ENV_VARS but read by nothing: {sorted(listed - read)}"


def test_the_warning_is_printed_and_not_fatal(capsys):
    """It reports and returns; the documented contract is that nothing in
    the environment stops the app from starting."""

    os.environ["BASE_OUTPUT_PAT"] = "C:/somewhere"
    try:
        smartgallery.warn_about_misspelt_env_vars()
    finally:
        del os.environ["BASE_OUTPUT_PAT"]

    out = capsys.readouterr().out
    assert "BASE_OUTPUT_PAT" in out, out
    assert "BASE_OUTPUT_PATH" in out, out
    assert "ignored" in out
