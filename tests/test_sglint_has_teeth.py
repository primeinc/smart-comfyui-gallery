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

import pytest

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
    assert codes == ["SG002", "SG003"], "shell=True and check= are Ruff's own (S602, PLW1510)"
    spelled = rules.rule_spawns([bad])[0].spelled()
    assert spelled.endswith("SG002 passes a bare command string instead of a list")


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


# --- the second batch: source-text pins and the schema contract ---------------------------


def _copy_of_tree(tmp_path):
    """The files the text rules read, copied so one can be bent."""
    here = tmp_path / "repo"
    wanted = {
        *policy.ADAPTER_DB_VOCABULARY,
        *policy.MUST_CALL_QUALIFIED,
        *policy.MUST_NOT_CALL_QUALIFIED,
        *policy.MUST_IMPORT,
        *policy.MUST_NOT_CONTAIN,
        *policy.MUST_CONTAIN,
        *policy.MUST_NOT_CONTAIN_BEFORE,
        *policy.MUST_NOT_CONTAIN_AFTER_DOCSTRING,
        *policy.NO_PARAMETER_NAMED,
        *policy.ONE_TO_MANY_MODULES,
        *policy.LITERAL_STATEMENTS_ONLY,
        "db/pages.py",
    }
    for relative in wanted:
        target = here / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((rules.REPO_ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8")
    for package in policy.PACKAGE_FORBIDDEN_PATTERNS:
        for source in (rules.REPO_ROOT / package).glob("*.py"):
            target = here / package / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return here


def _bend(path: pathlib.Path, addition: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + addition, encoding="utf-8")
    rules.parsed.cache_clear()


@pytest.mark.parametrize(
    ("relative", "addition", "code"),
    [
        ("sg_web/static/evolution.js", "\nfetch('/x');\n", "SG406"),
        ("db/evolution.py", "\n# generation_prompt\n", "SG406"),
        ("db/stories.py", "\n# UPDATE story_snapshot\n", "SG406"),
        ("sg_web/story_view.py", "\n# derived_\n", "SG406"),
        ("story_renderers/formatting.py", "\nPOLICY_VERSION = 9\n", "SG406"),
        ("sg_web/templates/story.html", "\n{{ x|safe }}\n", "SG406"),
        ("story_renderers/claims.py", "\nPOLICY_VERSION = 9\n", "SG407"),
        ("db/param_writer.py", "Q = 'INSERT OR REPLACE INTO file_param(a) VALUES(1)'\n", "SG410"),
    ],
)
def test_each_text_pin_fires_on_the_shape_it_exists_for(tmp_path, relative, addition, code):
    here = _copy_of_tree(tmp_path)
    assert rules.rule_adapters(here) == [], "the copy starts clean"
    target = here / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text("", encoding="utf-8")
    _bend(target, addition)
    codes = {f.code for f in rules.rule_adapters(here)}
    rules.parsed.cache_clear()
    assert code in codes, codes


def test_the_narrator_pins_fire_before_the_marker_and_on_the_signature(tmp_path):
    here = _copy_of_tree(tmp_path)
    rendering = here / "db" / "rendering.py"
    head, tail = rendering.read_text(encoding="utf-8").split("# --- persistence", 1)
    rendering.write_text(head + "\n_x = 'FROM '\n# --- persistence" + tail, encoding="utf-8")
    rules.parsed.cache_clear()
    assert "SG407" in {f.code for f in rules.rule_adapters(here)}
    rendering.write_text(
        (rules.REPO_ROOT / "db" / "rendering.py")
        .read_text(encoding="utf-8")
        .replace(
            "    def render(self, snapshot: dict, plan: dict,", "    def render(self, conn, snapshot: dict, plan: dict,"
        ),
        encoding="utf-8",
    )
    rules.parsed.cache_clear()
    assert "SG408" in {f.code for f in rules.rule_adapters(here)}
    rules.parsed.cache_clear()


def test_a_page_query_restated_elsewhere_is_seen(tmp_path):
    here = _copy_of_tree(tmp_path)
    pages = here / "db" / "pages.py"
    pages.write_text('NEWEST = "SELECT id FROM file"\nONE = "SELECT 1"\n', encoding="utf-8")
    rules.parsed.cache_clear()
    assert "SG411" in {f.code for f in rules.rule_adapters(here)}
    rules.parsed.cache_clear()


def test_the_schema_contract_holds_and_each_rule_can_fail():
    from sglint import schema_rules

    ddl = schema_rules.SCHEMA.read_text(encoding="utf-8")
    assert schema_rules.rule_schema(ddl) == []
    person = "person_id     INTEGER NOT NULL REFERENCES person(id) ON DELETE CASCADE"
    assert ddl.count(person) == 1, "the control's handle moved"
    assert {f.code for f in schema_rules.rule_schema(ddl.replace(person, "person_id     INTEGER NOT NULL"))} >= {
        "SG704",
        "SG705",
    }, "a removed foreign key is seen, and the load-bearing reference it carried"
    repointed = ddl.replace(person, person.replace("person(id)", "artifact(id)"))
    assert "SG705" in {f.code for f in schema_rules.rule_schema(repointed)}
    assert "SG703" in {
        f.code for f in schema_rules.rule_schema(ddl.replace(person, person.replace("person(id)", "nobody(id)")))
    }
    loose = ddl.replace("CREATE TABLE rating (", "CREATE TABLE rating_loose (a INTEGER);\nCREATE TABLE rating (", 1)
    assert "SG701" in {f.code for f in schema_rules.rule_schema(loose)}
    no_action = ddl.replace(person, person.replace(" ON DELETE CASCADE", ""), 1)
    assert "SG706" in {f.code for f in schema_rules.rule_schema(no_action)}
    assert schema_rules.has_rowid(schema_rules.built(ddl), "file_param"), "the long tail keeps its rowid"


def test_the_migration_ledger_rule_reads_the_tree_and_can_fail(tmp_path):
    from sglint import schema_rules

    assert schema_rules.rule_migrations() == []
    here = tmp_path / "repo"
    (here / "db").mkdir(parents=True)
    (here / "db" / "connect.py").write_text("USER_VERSION = 3\n", encoding="utf-8")
    (here / "db" / "migrate.py").write_text(
        "def step(n):\n    return lambda f: f\n\n\n"
        "@step(1)\ndef a(conn):\n    pass\n\n\n@step(3)\ndef c(conn):\n    pass\n",
        encoding="utf-8",
    )
    codes = {f.code for f in schema_rules.rule_migrations(here)}
    assert codes == {"SG707", "SG708"}, codes


def test_a_test_that_spawns_is_a_lint_error(tmp_path):
    """A test never starts a program: the same spawn that is merely held
    to its arguments elsewhere is SG006 under tests/."""
    tests = tmp_path / "tests"
    tests.mkdir()
    bad = tests / "test_spawny.py"
    bad.write_text("import subprocess\nsubprocess.run(['git', 'status'], timeout=5, check=True)\n", encoding="utf-8")
    assert [f.code for f in rules.rule_spawns([bad], tests=tests)] == ["SG006"]
    elsewhere = tmp_path / "tool.py"
    elsewhere.write_text(bad.read_text(encoding="utf-8"), encoding="utf-8")
    assert rules.rule_spawns([elsewhere], tests=tests) == [], "outside tests/ the same spawn is fine"


class _FakeGit:
    """A git that answers from a table, so the repo rules can be bent
    without starting a program."""

    def __init__(self, answers: dict[tuple[str, ...], str], rc: dict[tuple[str, ...], int] | None = None):
        self.answers = answers
        self.rc = rc or {}

    def __call__(self, *args, cwd=None):
        import subprocess

        return subprocess.CompletedProcess(args, self.rc.get(args, 0), self.answers.get(args, ""), "")


def test_the_repo_rules_fire_on_a_fake_index(tmp_path):
    from sglint import repo_rules

    tracked = "\n".join(f"f{i}.py" for i in range(150)) + "\ndb/schema.sql\nrun_smartgallery.bat\n"
    git = _FakeGit(
        {
            ("ls-files",): tracked,
            ("ls-files", "-i", "-o", "--exclude-standard"): "scratch.tmp\n",
            ("ls-files", "-i", "-c", "--exclude-standard"): "secrets.env\n",
        },
        rc={("check-ignore", "--no-index", "-q", "run_exhibition.sh"): 1},
    )
    codes = sorted(f.code for f in repo_rules.rule_index(git, tmp_path))
    assert codes == ["SG801", "SG802", "SG803"], codes
    eol = "\n".join(f"i/lf w/lf attr/text\tf{i}.py" for i in range(120)) + "\ni/crlf w/crlf attr/\tbad.txt\n"
    git = _FakeGit({("ls-files", "--eol"): eol})
    codes = sorted(f.code for f in repo_rules.rule_line_endings(git, tmp_path, checkouts=False))
    assert codes == ["SG804", "SG805"], codes
    (tmp_path / "pytest.ini").write_text("[pytest]\ntestpaths = tests\n", encoding="utf-8")
    assert [f.code for f in repo_rules.rule_pytest_path(tmp_path)] == ["SG809"]
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["numpy>=2", "pillow"]\n[dependency-groups]\nai = ["torch"]\n',
        encoding="utf-8",
    )
    (tmp_path / "requirements.txt").write_text("numpy>=1\n", encoding="utf-8")
    codes = sorted(f.code for f in repo_rules.rule_requirements(tmp_path))
    assert codes == ["SG807", "SG807", "SG808"], codes
