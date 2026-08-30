"""SG015 and SG016: two ways a recipe reports a pass it did not earn.

Both were live in `compat.just` at the same time, and the first hid the second.

`if ! just compat "$lane"; then code=$?; fi` captures the status of the
NEGATION, which is 0 whenever the branch is taken, so every failing lane was
written into `lanes.json` as a pass -- and the ledger reads that file to decide
which cells are BLOCKED.

Underneath it, `pins` ran `python compat/harness/provenance.py` by path, which
puts `compat/harness` on `sys.path` instead of the repository root, so
`import proc` raised ModuleNotFoundError. The lane had been dying for hours and
the swallowed exit code is why nobody saw it.

A rule that has never gone red is not known to discriminate, so each is shown
firing on the defect and silent on the correct form.
"""

from __future__ import annotations

import pathlib

from sglint import rules


def _just(where: pathlib.Path, name: str, body: str) -> pathlib.Path:
    target = where / name
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(body)
    return target


def test_a_negated_if_capturing_the_exit_code_is_caught(tmp_path):
    """`$?` inside `if ! ...` is the negation's status, and always 0."""
    _just(
        tmp_path,
        "bad.just",
        'run:\n    for lane in a b; do\n        code=0\n        if ! just compat "$lane"; then\n'
        '            code=$?\n        fi\n        echo "$code"\n    done\n',
    )
    found = rules.rule_recipe_exit_codes(tmp_path)
    assert [(one.code, one.path.name) for one in found] == [("SG015", "bad.just")], found
    assert "always 0" in found[0].message


def test_the_or_form_that_keeps_the_real_code_is_not_caught(tmp_path):
    """`cmd || code=$?` keeps the status and is exempt from errexit."""
    _just(
        tmp_path,
        "good.just",
        'run:\n    code=0\n    just compat "$lane" || code=$?\n    echo "$code"\n',
    )
    assert rules.rule_recipe_exit_codes(tmp_path) == []


def test_a_script_run_by_path_is_caught(tmp_path):
    """`python pkg/mod/thing.py` drops the repository root from sys.path."""
    _just(tmp_path, "bad.just", "pins:\n    {{ python }} compat/harness/provenance.py\n")
    found = rules.rule_recipe_module_imports(tmp_path)
    assert [(one.code, one.path.name) for one in found] == [("SG016", "bad.just")], found
    assert "provenance.py" in found[0].message


def test_the_module_form_and_a_root_level_script_are_not_caught(tmp_path):
    """`-m` puts the root on sys.path, and a root-level file needs no change."""
    _just(
        tmp_path,
        "good.just",
        "pins:\n    {{ python }} -m compat.harness.provenance\n\nattack:\n    {{ python }} proc_attack.py\n",
    )
    assert rules.rule_recipe_module_imports(tmp_path) == []
