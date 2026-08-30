"""Every way a recipe step's failure is lost, and proof each one is caught.

Under errexit a failing command fails the recipe. The ways a status escapes
that are a finite set, so this checks the set rather than a list of spellings
somebody remembered:

    cmd || true         forced to 0
    if ! cmd; then $?   `$?` belongs to the negation, always 0
    a | b               a's status dropped without pipefail (SG017)
    set +e              everything after it can fail silently

`a && b` and `if cmd; then` are deliberately NOT in it: under errexit a
failing left operand still fails the line, and a condition's failure is the
whole point of a condition.

All four were live in this repository at once. `if ! just compat "$lane"; then
code=$?; fi` recorded every failing lane as a pass, and underneath it the
`pins` lane had been dying on ModuleNotFoundError with nobody able to see it.
"""

from __future__ import annotations

import pathlib

from sglint import rules


def _just(where: pathlib.Path, name: str, body: str) -> pathlib.Path:
    target = where / name
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(body)
    return target


def _codes(found: list) -> list[tuple[str, int]]:
    return [(one.code, one.line) for one in found]


def test_or_true_is_caught(tmp_path):
    """`cmd || true` forces the status to 0."""
    _just(tmp_path, "bad.just", "x:\n    just compat cases || true\n    just compat pins || :\n")
    assert _codes(rules.rule_recipe_exit_codes(tmp_path)) == [("SG015", 2), ("SG015", 3)]


def test_or_true_inside_a_trap_is_still_caught(tmp_path):
    """A trap is not an exemption: cleanup that fails leaks and says nothing."""
    _just(tmp_path, "bad.just", "x:\n    trap 'kill \"$pid\" || true' EXIT\n")
    assert _codes(rules.rule_recipe_exit_codes(tmp_path)) == [("SG015", 2)]


def test_unsetting_errexit_is_caught(tmp_path):
    """`set +e` makes every later step in the recipe silent."""
    _just(tmp_path, "bad.just", "x:\n    set +e\n    just compat cases\n")
    assert _codes(rules.rule_recipe_exit_codes(tmp_path)) == [("SG015", 2)]


def test_a_script_file_without_pipefail_is_caught(tmp_path):
    """just defaults a [script] recipe to `sh -eu`, which has no pipefail."""
    _just(tmp_path, "bad.just", "[script]\nx:\n    git log | head -1\n")
    assert _codes(rules.rule_script_recipes_fail_loudly(tmp_path)) == [("SG017", 1)]


def test_an_interpreter_without_pipefail_is_caught(tmp_path):
    """Declaring one is not enough; it has to carry -e and pipefail."""
    _just(tmp_path, "bad.just", "set script-interpreter := ['bash', '-cu']\n\n[script]\nx:\n    git log | head -1\n")
    assert _codes(rules.rule_script_recipes_fail_loudly(tmp_path)) == [("SG017", 1)]


def test_the_handled_forms_are_not_caught(tmp_path):
    """Every shape that keeps or reports the status stays silent.

    `|| code=$?` records it, `|| exit` and `|| { ...; exit 1; }` fail on it,
    `|| echo >&2` reports it, `[ x ] || [ y ]` is a test rather than a
    command, and `if cmd; then` is a condition doing its job.

    `cmd && other` is NOT here. Measured under `sh`: `false && echo right`
    does not fire errexit and the script runs on, so a failing left
    operand is lost exactly the way `|| true` loses one.
    """
    _just(
        tmp_path,
        "good.just",
        "set script-interpreter := ['bash', '-euo', 'pipefail']\n"
        "\n"
        "[script]\n"
        "x:\n"
        "    code=0\n"
        "    just compat cases || code=$?\n"
        "    just compat pins || exit 1\n"
        '    grep -q needle file || { echo "absent" >&2; exit 1; }\n'
        '    rm -rf "$scratch" || echo "warning: $scratch was left behind" >&2\n'
        '    [ -n "$a" ] || [ -n "$b" ]\n'
        "    if just compat hf; then echo ok; fi\n",
    )
    assert rules.rule_recipe_exit_codes(tmp_path) == []
    assert rules.rule_script_recipes_fail_loudly(tmp_path) == []


def test_a_failing_left_operand_of_and_is_caught(tmp_path):
    """`cmd && other` loses the left operand's failure.

    Measured in isolated processes under `sh`, which is just's default script
    interpreter: `set -eu; false && echo right; echo REACHED` prints REACHED
    and exits 0. errexit does not fire on the left of `&&`, so the recipe
    carries on past a step that failed -- the same loss as `|| true`, and the
    reason `&&` belongs in the enumerated set rather than excluded from it.
    """
    _just(tmp_path, "bad.just", "x:\n    just compat cases && echo chained\n")
    assert _codes(rules.rule_recipe_exit_codes(tmp_path)) == [("SG015", 2)]


def test_a_test_builtin_before_and_is_not_caught(tmp_path):
    """`[ -n "$a" ] && cmd` tests a condition; there is no status to lose."""
    _just(tmp_path, "good.just", 'x:\n    [ -n "$a" ] && echo present\n    test -f x && echo here\n')
    assert rules.rule_recipe_exit_codes(tmp_path) == []


def test_a_declaring_builtin_hiding_a_substitution_is_caught(tmp_path):
    """`export x=$(cmd)` reports the builtin's status, never the command's.

    Measured under `sh`, which is just's default script interpreter:
    `x=$(false)` fires errexit at rc=1, `export x=$(false)` exits 0 and
    carries on. The same holds for `local` inside a function.
    """
    _just(
        tmp_path,
        "bad.just",
        "x:\n"
        "    export a=$(git rev-parse HEAD)\n"
        "    local b=$(git rev-parse HEAD)\n"
        "    readonly c=`git rev-parse HEAD`\n",
    )
    assert _codes(rules.rule_recipe_exit_codes(tmp_path)) == [("SG015", 2), ("SG015", 3), ("SG015", 4)]


def test_declaring_then_assigning_separately_is_not_caught(tmp_path):
    """The fix: declare on one line, assign on the next, where errexit sees it."""
    _just(tmp_path, "good.just", "x:\n    local a\n    a=$(git rev-parse HEAD)\n    export a\n")
    assert rules.rule_recipe_exit_codes(tmp_path) == []
