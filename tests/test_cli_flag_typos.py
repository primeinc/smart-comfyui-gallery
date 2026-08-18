"""A mistyped command-line flag must not start the gallery anyway.

The flags are parsed with `parse_known_args`, which collects anything it
does not recognise into a list that was then dropped without a word. So
`--forcelogin`, or `--force_login` with an underscore, started a gallery
with no login at all while the operator believed it was shut. The same
silence applied to `--exhibition` and `--blind-rating`: the mode simply did
not happen, and nothing said so.

The check only applies when smartgallery.py is the program being run.
Imported -- by these tests, or by anything embedding the gallery -- argv
belongs to the host and its arguments are not ours to police. That matters
here: pytest's own argv would otherwise stop the suite at import.

Every case used to run smartgallery.py as a fresh program, three seconds of
interpreter start and module import each, to read one refusal back off
stdout. The refusal is now `refuse_unrecognised_flags`, which takes the
leftovers and argv[0] and can simply be called: SystemExit is the refusal,
capsys is the message (pytest doc/en/how-to/capture-stdout-stderr.rst:112-142).
"""

from __future__ import annotations

import ast

import pytest

import smartgallery

_PROGRAM = "smartgallery.py"


def _leftovers(args):
    """What argparse would hand the check for this command line."""
    _known, unknown = smartgallery._parser.parse_known_args(args)
    return unknown


@pytest.mark.parametrize(
    "typo",
    [
        "--forcelogin",
        "--force_login",
        "--exhibiton",
        "--blind_rating",
        "--enable-guest-logins",
    ],
)
def test_a_mistyped_flag_refuses_to_start(capsys, typo):
    """The regression: these started a gallery that was not what was asked
    for, and said nothing."""
    unknown = _leftovers([typo, "--admin-pass", "correct-horse-battery"])

    with pytest.raises(SystemExit) as refused:
        smartgallery.refuse_unrecognised_flags(unknown, _PROGRAM)

    assert refused.value.code == 2, "a refusal has to be a failing exit"
    assert typo in capsys.readouterr().out, "the message does not name the option"


def test_the_message_suggests_the_real_flag(capsys):
    """A refusal that does not say what was meant just moves the problem."""
    unknown = _leftovers(["--forcelogin", "--admin-pass", "correct-horse-battery"])

    with pytest.raises(SystemExit):
        smartgallery.refuse_unrecognised_flags(unknown, _PROGRAM)

    assert "--force-login" in capsys.readouterr().out, "no suggestion offered"


@pytest.mark.parametrize(
    "args",
    [
        ["--force-login", "--admin-pass", "correct-horse-battery", "--blind-rating"],
        ["--exhibition"],
        ["--enable-guest-login"],
        [],
    ],
)
def test_the_real_flags_still_start(capsys, args):
    """The counterpart -- refusing everything would pass the tests above."""
    unknown = _leftovers(args)

    refused = smartgallery.refuse_unrecognised_flags(unknown, _PROGRAM)

    assert refused == [], f"{args} was refused: {capsys.readouterr().out}"
    assert capsys.readouterr().out == "", "a valid command line complained"


def test_the_real_flags_reach_the_settings_they_name():
    """The other half of "still start": parsing has to produce the modes,
    not merely decline to refuse them."""
    parsed, _unknown = smartgallery._parser.parse_known_args(["--force-login", "--blind-rating"])

    assert parsed.force_login is True
    assert parsed.blind_rating is True
    assert parsed.exhibition is False


def test_importing_the_module_ignores_the_hosts_arguments(capsys):
    """pytest runs with its own argv, and anything embedding the gallery has
    its own too. The check must not fire for them -- this suite would not
    run at all if it did, which is itself the standing proof."""
    unknown = _leftovers(["-q", "--not-our-flag"])
    assert unknown, "nothing stray to ignore; this test is not measuring it"

    refused = smartgallery.refuse_unrecognised_flags(unknown, "some-other-program")

    assert refused == [], "the gallery policed a host's arguments"
    assert capsys.readouterr().out == ""


def test_the_check_runs_at_import(gallery_tree):
    """The refusal is only worth anything if startup still calls it. It was
    an inline block once and could be again, or the call could be dropped
    while the function stayed."""
    called = {
        node.func.id
        for node in ast.walk(gallery_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "refuse_unrecognised_flags" in called, (
        "nothing calls refuse_unrecognised_flags at startup, so a misspelt flag is silently dropped again"
    )
