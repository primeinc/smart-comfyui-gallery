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


@pytest.mark.slow
def test_the_tree_is_clean_under_the_rules():
    found = rules.run()
    assert found == [], "\n".join(f.spelled() for f in found)


@pytest.mark.slow
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


def test_the_adapter_rules_can_fail(tree):
    """The tree with one adapter running its own statement."""
    _bend(tree / "sg_web" / "gallery.py", "\n\ndef _leak(conn):\n    return conn.execute('SELECT 1')\n")
    assert "SG401" in {f.code for f in rules.rule_adapters(tree)}


def _surfaces(root: pathlib.Path, templates: int, scripts: int) -> None:
    """A tree with `templates` clean templates and `scripts` clean scripts."""
    (root / "sg_web" / "templates").mkdir(parents=True, exist_ok=True)
    (root / "frontend" / "src").mkdir(parents=True, exist_ok=True)
    for i in range(templates):
        (root / "sg_web" / "templates" / f"t{i}.html").write_text("<p>fine</p>", encoding="utf-8")
    for i in range(scripts):
        (root / "frontend" / "src" / f"s{i}.ts").write_text("export {};\n", encoding="utf-8")


def test_the_surface_rule_can_fail(tmp_path):
    root = tmp_path / "repo"
    _surfaces(root, policy.TEMPLATE_MINIMUM, policy.SCRIPT_MINIMUM - 1)
    (root / "frontend" / "src" / "bad.ts").write_text("fetch('/search?q=')", encoding="utf-8")
    assert [f.code for f in rules.rule_surfaces(root)] == ["SG501"]
    assert rules.rule_surfaces(pathlib.Path(rules.REPO_ROOT)) == []


def test_the_surface_sweep_notices_either_half_vanishing(tmp_path):
    """The sweep counts templates and scripts separately. One shared
    minimum could not see every script leave for frontend/src, because
    the templates alone cleared it: a sweep with no scripts at all
    reported nothing. Each half must fail on its own."""
    scriptless = tmp_path / "scriptless"
    _surfaces(scriptless, policy.TEMPLATE_MINIMUM, 0)
    assert [f.code for f in rules.rule_surfaces(scriptless)] == ["SG500"], "no scripts is not a pass"

    templateless = tmp_path / "templateless"
    _surfaces(templateless, 0, policy.SCRIPT_MINIMUM)
    assert [f.code for f in rules.rule_surfaces(templateless)] == ["SG500"], "no templates is not a pass"


# --- the second batch: source-text pins and the schema contract ---------------------------


BENT: list[pathlib.Path] = []


@pytest.fixture(scope="module")
def _tree(tmp_path_factory):
    """The files the text rules read, copied once so one can be bent."""
    here = _copy_of_tree(tmp_path_factory.mktemp("pins"))
    assert rules.rule_adapters(here) == [], "the copy starts clean"
    return here


@pytest.fixture
def tree(_tree):
    """The copy with every bend of the previous test undone -- the bent
    files alone are rewritten from the repository, not the whole tree."""
    for bent in BENT:
        relative = bent.relative_to(_tree)
        source = rules.REPO_ROOT / relative
        if source.exists():
            bent.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            bent.unlink(missing_ok=True)
    BENT.clear()
    return _tree


def _copy_of_tree(tmp_path):
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
    BENT.append(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    held = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(held + addition, encoding="utf-8")


def _rewrite(path: pathlib.Path, text: str) -> None:
    BENT.append(path)
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize(
    ("relative", "addition", "code"),
    [
        ("frontend/src/evolution.ts", "\nfetch('/x');\n", "SG406"),
        ("db/evolution.py", "\n# generation_prompt\n", "SG406"),
        ("db/stories.py", "\n# UPDATE story_snapshot\n", "SG406"),
        ("sg_web/story_view.py", "\n# derived_\n", "SG406"),
        ("story_renderers/formatting.py", "\nPOLICY_VERSION = 9\n", "SG406"),
        ("sg_web/templates/story.html", "\n{{ x|safe }}\n", "SG406"),
        ("story_renderers/claims.py", "\nPOLICY_VERSION = 9\n", "SG407"),
        ("db/param_writer.py", "Q = 'INSERT OR REPLACE INTO file_param(a) VALUES(1)'\n", "SG410"),
    ],
)
def test_each_text_pin_fires_on_the_shape_it_exists_for(tree, relative, addition, code):
    _bend(tree / relative, addition)
    codes = {f.code for f in rules.rule_adapters(tree)}
    assert code in codes, codes


def test_the_narrator_pins_fire_before_the_marker_and_on_the_signature(tree):
    rendering = tree / "db" / "rendering.py"
    head, tail = rendering.read_text(encoding="utf-8").split("# --- persistence", 1)
    _rewrite(rendering, head + "\n_x = 'FROM '\n# --- persistence" + tail)
    assert "SG407" in {f.code for f in rules.rule_adapters(tree)}
    _rewrite(
        rendering,
        (rules.REPO_ROOT / "db" / "rendering.py")
        .read_text(encoding="utf-8")
        .replace(
            "    def render(self, snapshot: dict, plan: dict,", "    def render(self, conn, snapshot: dict, plan: dict,"
        ),
    )
    assert "SG408" in {f.code for f in rules.rule_adapters(tree)}


def test_a_page_query_restated_elsewhere_is_seen(tree):
    _rewrite(tree / "db" / "pages.py", 'NEWEST = "SELECT id FROM file"\nONE = "SELECT 1"\n')
    assert "SG411" in {f.code for f in rules.rule_adapters(tree)}


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


def test_the_requirements_rule_sees_an_extra_drift(tmp_path):
    """`litestar[pydantic]` and `litestar` install different packages. The
    rule compared specifiers and markers only, so a file carrying the extra
    and a file without it agreed on nothing and reported it."""
    from sglint import repo_rules

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\ndependencies = ["litestar[pydantic]"]\n', encoding="utf-8"
    )
    (tmp_path / "requirements.txt").write_text("litestar\n", encoding="utf-8")
    assert [f.code for f in repo_rules.rule_requirements(tmp_path)] == ["SG807"]

    (tmp_path / "requirements.txt").write_text("litestar[pydantic]\n", encoding="utf-8")
    assert repo_rules.rule_requirements(tmp_path) == []


def test_the_wire_vocabulary_rule_can_fail(tmp_path):
    """SG709 compares a Literal in the wire against a CHECK in the DDL.

    Four controls, because the rule has four outcomes and three of them are
    failures that must not be mistaken for agreement: a vocabulary the DDL
    does not constrain, an assignment that is not a Literal, sets that
    disagree, and sets that match.
    """
    from sglint import schema_rules

    listed = "'queued','running','done'"
    ddl = f"CREATE TABLE job (\n    state TEXT NOT NULL CHECK (state IN\n      ({listed}))\n) STRICT;\n"

    root = tmp_path / "repo"
    (root / "sg_web").mkdir(parents=True)
    where = root / "sg_web" / "app.py"
    only = {"sg_web/app.py": {"JobState": ("job", "state")}}

    def check(source: str):
        where.write_text(source, encoding="utf-8")
        return [f.code for f in schema_rules.rule_wire_vocabularies(root, ddl, only)]

    agreed = 'JobState = Literal["queued", "running", "done"]\n'
    assert check(agreed) == [], "the sets are equal"

    assert check('JobState = Literal["queued", "running"]\n') == ["SG709"], "a member the database allows is missing"
    assert check(agreed.replace('"done"', '"done", "melted"')) == ["SG709"], "a member no row can hold"
    assert check("JobState = str\n") == ["SG709"], "not a Literal at all is unreadable, not agreement"

    unconstrained = "CREATE TABLE job (\n    state TEXT NOT NULL\n) STRICT;\n"
    where.write_text(agreed, encoding="utf-8")
    assert [f.code for f in schema_rules.rule_wire_vocabularies(root, unconstrained, only)] == ["SG709"], (
        "a DDL with no CHECK cannot be agreed with"
    )


#: The contracts and the near-miss the route controls below are written
#: against. A dataclass is the near-miss on purpose: it names its fields,
#: which is most of what a body needs and none of what the policy says.
_CONTRACTS = (
    "import dataclasses\n"
    "from sg_web.wire import Wire\n"
    "\n"
    "class Named(Wire):\n"
    "    name: str\n"
    "\n"
    "class Narrower(Named):\n"
    "    more: str\n"
    "\n"
    "@dataclasses.dataclass\n"
    "class Loose:\n"
    "    name: str\n"
)


def _flagged(found) -> set[str]:
    """The handlers a rule reported, by name. Every control below lives in
    ONE module read in ONE pass, so a clean case is proved clean while the
    broken ones are present -- which is what the repository actually looks
    like, and stronger than asking about each alone."""
    return {one.message.split()[0] for one in found}


def test_the_request_contract_rule_can_fail():
    """SG412 holds every route's JSON body to a Wire contract.

    A body that obeys the policy, one that narrows another contract, three
    that do not obey it, a form (a different contract, not a broken one), a
    GET that carries no body, and a handler the ledger excuses.
    """
    module = rules.from_text(
        "sg_web/routes.py",
        _CONTRACTS + "\n"
        "@post('/a')\ndef contracted(data: Named) -> None: ...\n"
        "@post('/b')\ndef narrowed(data: Narrower) -> None: ...\n"
        "@post('/c')\ndef optional(data: Named | None = None) -> None: ...\n"
        "@post('/d')\ndef formed(data: URLEncodedBody[Loose]) -> None: ...\n"
        "@get('/e')\ndef reading(data: dict) -> None: ...\n"
        "@post('/f')\ndef unnamed(data: dict) -> None: ...\n"
        "@post('/g')\ndef dataclassed(data: Loose) -> None: ...\n"
        "@patch('/h')\ndef patched(data: dict) -> None: ...\n",
    )

    assert _flagged(rules.rule_request_contracts([module], frozenset())) == {"unnamed", "dataclassed", "patched"}

    excused = frozenset({"sg_web/routes.py:unnamed", "sg_web/routes.py:dataclassed"})
    assert _flagged(rules.rule_request_contracts([module], excused)) == {"patched"}, (
        "the ledger is what excuses a body, and it is read"
    )

    stale = excused | {"sg_web/routes.py:contracted"}
    assert _flagged(rules.rule_request_contracts([module], stale)) == {"patched", "contracted"}, (
        "and it reports its own stale line, so the ledger can only shrink"
    )


def test_the_response_contract_rule_can_fail():
    """SG413 holds every route's JSON answer to a wire model.

    The negotiated routes are why this rule reads `responses=` as well as
    the return type: a handler that answers a page to a person and JSON to
    a machine cannot say the JSON half in its signature, and OpenAPI reads
    the declaration instead.
    """
    module = rules.from_text(
        "sg_web/routes.py",
        _CONTRACTS + "\n"
        "@get('/a')\ndef contracted() -> Named: ...\n"
        "@get('/b')\ndef narrowed() -> Narrower: ...\n"
        "@get('/c')\ndef listed() -> list[Named]: ...\n"
        "@get('/d')\ndef redirecting() -> Response[Named] | Redirect: ...\n"
        "@get('/e')\ndef rendered() -> Template: ...\n"
        "@get('/f')\ndef bytes_out() -> Response[bytes]: ...\n"
        "@get('/g', responses={200: ResponseSpec(data_container=list[Named])})\n"
        "def declared() -> Template | Response: ...\n"
        "@get('/h')\ndef unnamed() -> dict: ...\n"
        "@get('/i')\ndef unnamed_list() -> list[dict]: ...\n"
        "@get('/j')\ndef unparameterized() -> Response: ...\n"
        "@get('/k')\ndef negotiating() -> Template | Response: ...\n"
        "@get('/l', responses={200: ResponseSpec(data_container=list[dict])})\n"
        "def declared_vague() -> Template: ...\n",
    )
    broken = {"unnamed", "unnamed_list", "unparameterized", "negotiating", "declared_vague"}

    assert _flagged(rules.rule_response_contracts([module], frozenset())) == broken

    held = frozenset({f"sg_web/routes.py:{one}" for one in broken})
    assert _flagged(rules.rule_response_contracts([module], held)) == set(), (
        "the ledger excuses what is not converted yet"
    )

    stale = held | {"sg_web/routes.py:contracted"}
    assert _flagged(rules.rule_response_contracts([module], stale)) == {"contracted"}, (
        "and it reports its own stale line, so the ledger can only shrink"
    )


def test_the_connection_lifetime_rule_can_fail():
    """SG103 catches a database opened and then dropped.

    The escapes are the point, and their narrowness is the rest of it.
    Returning transfers ownership structurally -- alone, in the tuple a
    fixture hands back beside the ids it minted, or in a dict. STORING one
    does not: it shows the connection left the function and nothing about
    anyone closing it, so a leak cannot be silenced by putting it in an
    object. A connection that really is kept for the life of the process is
    named in the ledger, and the ledger reports its own stale lines.
    """
    module = rules.from_text(
        "tests/opener.py",
        "from db import connect\n\n"
        "_KEPT = {}\n\n"
        "def closes():\n    conn = connect.memory()\n    try:\n        pass\n"
        "    finally:\n        connect.close(conn)\n\n"
        "def returns():\n    conn = connect.memory()\n    return conn\n\n"
        "def returns_a_tuple():\n    conn = connect.memory()\n    return conn, 1\n\n"
        "def returns_a_dict():\n    conn = connect.memory()\n    return {'conn': conn}\n\n"
        "def yields():\n    conn = connect.memory()\n    yield conn\n\n"
        "def drops():\n    conn = connect.memory()\n    conn.execute('SELECT 1')\n\n"
        "def drops_a_file(p):\n    conn = connect.connect(p)\n    conn.execute('SELECT 1')\n\n"
        "def stores(k):\n    conn = connect.connect(k)\n    _KEPT[k] = conn\n\n"
        "def returns_what_it_read(k):\n    conn = connect.connect(k)\n"
        "    return f\"v{conn.execute('PRAGMA data_version').fetchone()[0]}\"\n\n"
        "class A:\n    def hangs_it_off_self(self):\n        self.conn = connect.memory()\n",
    )
    dropped = {"drops", "drops_a_file", "stores", "returns_what_it_read", "hangs_it_off_self"}

    assert _flagged(rules.rule_connection_lifetime([module], frozenset())) == dropped, (
        "storing a connection is not a transfer -- not in a module dict, not on self -- because it shows"
        " only that the connection left, never that anyone closes it; and neither is RETURNING what it"
        " read, which is db/resultset.py's monitor exactly and the shape that first escaped this rule"
    )

    ledger = frozenset({"tests/opener.py:stores"})
    assert _flagged(rules.rule_connection_lifetime([module], ledger)) == dropped - {"stores"}, (
        "a declared long-lived owner is excused, by name"
    )

    stale = ledger | {"tests/opener.py:closes"}
    assert _flagged(rules.rule_connection_lifetime([module], stale)) == (dropped - {"stores"}) | {"closes"}, (
        "and a declaration that no longer describes the function is reported, so the ledger can only shrink"
    )
