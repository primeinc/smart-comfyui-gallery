"""The linter can fail -- the controls the structural rules used to
carry as tests, now aimed at sglint itself.

Every rule is an absence ("nothing does X"), and a sweep that
understood nothing would report the same absence. So each rule is fed
the shape it exists to catch and must flag it, and each sweep is shown
to reach the tree it claims to read. These are tests of the linter;
the rules over the application run as `python -m sglint`.
"""

from __future__ import annotations

import ast
import pathlib

from sglint import policy, rules


def test_the_tree_is_clean_under_the_rules():
    found = rules.run()
    assert found == [], "\n".join(f.spelled() for f in found)


def test_the_spawn_sweep_reaches_the_tree_and_catches_each_shape():
    total = sum(len(rules.spawn_calls(rules.parsed(s))) for s in rules.every_source())
    assert total >= 2, f"only {total} subprocess calls found; the sweep is not reaching them"
    bad = ast.parse(
        "import subprocess\nsubprocess.run('ffprobe ' + path, shell=True)\nsubprocess.run([tool, path])\n"
        "subprocess.Popen([tool], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)\n"
        "subprocess.Popen([tool], stdout=PIPE)\n"
    )
    calls = rules.spawn_calls(bad)
    assert len(calls) == 4
    assert rules.through_a_shell(calls[0])
    assert isinstance(calls[0].args[0], ast.BinOp)
    assert rules.keyword(calls[1], "timeout") is None
    assert rules.keyword(calls[1], "check") is None
    assert rules.pipes_output(calls[2]), "the undrained-pipe check does not see the shape it exists for"
    assert not rules.pipes_output(calls[1])
    assert rules.pipes_output(calls[3]), "a directly-imported PIPE slips the predicate its own sweep sees"


def test_the_spawn_rule_reports_in_linter_shape(tmp_path):
    bad = tmp_path / "spawner.py"
    bad.write_text("import subprocess\nsubprocess.run('x', shell=True)\n", encoding="utf-8")
    codes = sorted(f.code for f in rules.rule_spawns([bad]))
    assert codes == ["SG001", "SG002", "SG003", "SG005"]
    assert (
        rules.rule_spawns([bad])[0]
        .spelled()
        .endswith("SG001 starts a program through a shell; pass a list of arguments")
    )


def test_the_sql_sweep_reaches_the_tree_and_the_list_is_exact():
    slots: dict[str, int] = {}
    for source in rules.shipped():
        for slot, _line, _col in rules.sql_interpolations(rules.parsed(source)):
            slots[slot] = slots.get(slot, 0) + 1
    assert len(slots) >= 5, sorted(slots)
    assert "marks" in slots
    assert "runnable" in slots
    assert set(policy.SQL_STRUCTURE) == set(slots), {
        "listed but no longer written anywhere": sorted(set(policy.SQL_STRUCTURE) - set(slots)),
        "written but not listed": sorted(set(slots) - set(policy.SQL_STRUCTURE)),
    }


def test_a_value_written_into_a_statement_would_be_caught(tmp_path):
    smuggled = ast.parse("q = f\"SELECT id FROM files WHERE name = '{user_input}'\"")
    assert [slot for slot, _l, _c in rules.sql_interpolations(smuggled)] == ["user_input"]
    bad = tmp_path / "smuggle.py"
    bad.write_text("q = f\"SELECT id FROM files WHERE name = '{user_input}'\"\n", encoding="utf-8")
    assert [f.code for f in rules.rule_sql_structure([bad])] == ["SG101"]


def test_the_connect_sweep_sees_the_decided_files_and_flags_a_stray(tmp_path):
    decided = [s for s in rules.new_pairing_sources() if s.name in policy.RAW_CONNECT_DECIDED]
    assert len(decided) == len(policy.RAW_CONNECT_DECIDED), decided
    assert any(rules.raw_connects(rules.parsed(s)) for s in decided)
    stray = tmp_path / "stray.py"
    stray.write_text("import sqlite3\nc = sqlite3.connect('x.db')\n", encoding="utf-8")
    assert [f.code for f in rules.rule_one_connect([stray])] == ["SG201"]


def test_the_import_sweep_reads_the_tree_and_an_eager_heavy_import_would_be_caught(tmp_path):
    seen: dict[str, tuple[int, int]] = {}
    for source in rules.shipped():
        seen.update(rules.import_time_modules(rules.parsed(source)))
    assert "litestar" in seen
    assert "sqlite3" in seen
    assert len(seen) > 20
    eager = rules.import_time_modules(ast.parse("import os\nimport torch\n"))
    lazy = rules.import_time_modules(ast.parse("import os\n\n\ndef go():\n    import torch\n    return torch\n"))
    assert "torch" in eager
    assert "torch" not in lazy, "an import inside a function is the fix, not the defect"
    assert "os" in lazy
    bad = tmp_path / "eager.py"
    bad.write_text("import torch\n", encoding="utf-8")
    assert [f.code for f in rules.rule_lazy_heavy([bad])] == ["SG301"]


def test_the_producer_gate_can_see_an_unwired_table():
    assert rules.unwired({"derived_nothing_writes_this"}) != {}
    schema = (rules.REPO_ROOT / "db" / "schema.sql").read_text(encoding="utf-8")
    tables = rules.derived_tables(schema)
    assert tables, "the schema lost its derived namespace?"
    assert set(rules.unwired(tables)) == set(policy.DERIVED_RESERVED)


def test_the_adapter_rules_can_fail(tmp_path, monkeypatch):
    """A copy of the tree with one adapter running its own statement."""
    root = tmp_path / "repo"
    for relative in (
        *policy.ADAPTER_DB_VOCABULARY,
        *policy.MUST_CALL_QUALIFIED,
        *policy.MUST_NOT_CALL_QUALIFIED,
        *policy.MUST_IMPORT,
        *policy.MUST_NOT_CONTAIN,
        *policy.MUST_CONTAIN,
        *policy.ONE_TO_MANY_MODULES,
        *policy.LITERAL_STATEMENTS_ONLY,
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((rules.REPO_ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8")
    assert rules.rule_adapters(root) == []
    gallery = root / "sg_web" / "gallery.py"
    gallery.write_text(
        gallery.read_text(encoding="utf-8") + "\n\ndef _leak(conn):\n    return conn.execute('SELECT 1')\n",
        encoding="utf-8",
    )
    rules.parsed.cache_clear()
    codes = {f.code for f in rules.rule_adapters(root)}
    assert "SG401" in codes
    rules.parsed.cache_clear()


def test_the_surface_rule_can_fail(tmp_path):
    root = tmp_path / "repo"
    (root / "sg_web" / "templates").mkdir(parents=True)
    (root / "sg_web" / "static").mkdir(parents=True)
    for i in range(policy.SURFACE_MINIMUM):
        (root / "sg_web" / "templates" / f"t{i}.html").write_text("<p>fine</p>", encoding="utf-8")
    (root / "sg_web" / "static" / "bad.js").write_text("fetch('/search?q=')", encoding="utf-8")
    assert [f.code for f in rules.rule_surfaces(root)] == ["SG501"]
    assert rules.rule_surfaces(pathlib.Path(rules.REPO_ROOT)) == []
