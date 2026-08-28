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
def test_every_capability_is_reachable_or_recorded():
    """SG010: the compare tray shipped fully built, styled, tested, and
    openable only by pressing a letter nothing on screen mentioned. Nothing
    caught it because nothing compared what the application can do against
    what a person can find.

    Each of the four ways that can go wrong is fed to the rule. A rule that
    only ever passes proves nothing, and this one is the record of which
    gaps were decided on rather than missed -- so a stale record has to
    fail too, or the record rots into fiction.
    """
    assert rules.rule_capability_has_a_way_in() == []
    served = (rules.REPO_ROOT / "sg_web").glob("*.py")
    assert sum(1 for _ in served) > 5, "the sweep is not reaching the application"

    # Every recorded address is one the application actually serves.
    addresses = "\n".join(p.read_text(encoding="utf-8") for p in (rules.REPO_ROOT / "sg_web").glob("*.py"))
    for path in rules.UNSURFACED:
        assert f'"{path}' in addresses, f"{path} is recorded as unsurfaced but nothing serves it"

    # And every recorded kind is one the schema still declares.
    for kind in rules.UNIMPLEMENTED_KINDS:
        assert kind in rules._JOB_KINDS, f"{kind} is recorded as unimplemented but is no longer declared"


@pytest.mark.slow
def test_the_template_sweep_reaches_the_tree_and_catches_a_broken_one(tmp_path):
    """SG008: `just check` reads no .html, so a template that 500s on every
    request to its page passed the whole gate. Twice.

    The sweep is shown to reach the real templates, then fed each shape
    that actually shipped: a comment an extraction sliced in half, and a
    `{# #}` written inside a `{{ }}`.
    """
    assert rules.rule_templates_parse() == []
    reached = list((rules.REPO_ROOT / "sg_web" / "templates").rglob("*.html"))
    assert len(reached) > 20, f"only {len(reached)} templates found; the sweep is not reaching them"

    def wrote(body: str) -> pathlib.Path:
        held = tmp_path / "one.html"
        with held.open("w", encoding="utf-8", newline="") as sink:
            sink.write(body)
        return held

    for broken in (
        "{# a comment nobody closed\n<p>after</p>",
        '<p>{{ {# why #} "x" }}</p>',
        "{% for one in many %}<li>{{ one }}</li>",
    ):
        wrote(broken)
        assert [f.code for f in rules.rule_templates_parse(tmp_path)] == ["SG008"], broken

    wrote("{# fine #}<p>{{ one }}</p>")
    assert rules.rule_templates_parse(tmp_path) == []


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
    """A tree with `templates` clean templates and `scripts` clean scripts.

    The templates are written as fragments: SG502 holds pages to the shell,
    and these exist to exercise SG500 and SG501, not that one.
    """
    (root / "sg_web" / "templates").mkdir(parents=True, exist_ok=True)
    (root / "frontend" / "src").mkdir(parents=True, exist_ok=True)
    for i in range(templates):
        (root / "sg_web" / "templates" / f"_t{i}.html").write_text("<p>fine</p>", encoding="utf-8")
    for i in range(scripts):
        (root / "frontend" / "src" / f"s{i}.ts").write_text("export {};\n", encoding="utf-8")


def test_the_surface_rule_can_fail(tmp_path):
    root = tmp_path / "repo"
    _surfaces(root, policy.TEMPLATE_MINIMUM, policy.SCRIPT_MINIMUM - 1)
    (root / "frontend" / "src" / "bad.ts").write_text("fetch('/search?q=')", encoding="utf-8")
    assert [f.code for f in rules.rule_surfaces(root)] == ["SG501"]
    assert rules.rule_surfaces(pathlib.Path(rules.REPO_ROOT)) == []


@pytest.mark.parametrize(
    "claim",
    [
        'document.addEventListener("keydown", (e) => e);',
        'window.addEventListener("keydown", (e) => e);',
        'onDocument("keydown", (e) => e);',
    ],
)
def test_a_second_module_claiming_the_keyboard_can_fail(tmp_path, claim):
    """SG503: one module owns keystroke dispatch, whatever the spelling.

    The defect it exists for: the viewer and the authored strip each had a
    document listener, so F was focus AND favorite, 1 was actual-pixels AND
    one star, 0 was fit AND clear-rating -- and every one of them fired
    both. Two listeners cannot agree about a key by being careful.
    """
    root = tmp_path / "repo"
    _surfaces(root, policy.TEMPLATE_MINIMUM, policy.SCRIPT_MINIMUM - 1)
    (root / "frontend" / "src" / "greedy.ts").write_text(claim, encoding="utf-8")
    assert [f.code for f in rules.rule_surfaces(root)] == ["SG503"]


def test_the_keyboard_router_and_a_widget_may_both_listen(tmp_path):
    """The negative control, twice over: the router itself is the point of
    the rule, and an ELEMENT-scoped listener is a widget minding its own
    keys, not a claim on the surface. A rule that flagged either would be
    unusable, and a rule that flagged neither would have no teeth."""
    root = tmp_path / "repo"
    _surfaces(root, policy.TEMPLATE_MINIMUM, policy.SCRIPT_MINIMUM - 2)
    (root / "frontend" / "src" / rules.KEY_ROUTER).write_text(
        'document.addEventListener("keydown", (e) => e);', encoding="utf-8"
    )
    (root / "frontend" / "src" / "widget.ts").write_text(
        'swap.addEventListener("keydown", (e) => e);', encoding="utf-8"
    )
    assert [f.code for f in rules.rule_surfaces(root)] == []


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
        ("db/evolution.py", '\n_x = "SELECT 1 FROM generation_prompt"\n', "SG406"),
        ("db/stories.py", '\n_x = "UPDATE story_snapshot SET a = 1"\n', "SG406"),
        ("sg_web/story_view.py", '\n_x = "SELECT 1 FROM derived_annotation"\n', "SG406"),
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


def test_a_ban_reads_what_a_module_does_not_what_it_says(tree):
    """A ban says what a module must not DO. A comment is prose ABOUT
    what it does, and a module agreeing with its own rule -- "this one
    never DELETEs" -- was being failed for saying so.

    Not hypothetical: the failure reads as an architecture violation
    rather than a word, and `sg_web/media_view.py` lost a line to it,
    DELETED rather than satisfied.

    A string literal still fires, and must: these bans are about
    reaching for SQL, and SQL lives in a string. What stopped being
    read is prose, not code.
    """
    view = tree / "sg_web" / "story_view.py"
    whole = view.read_text(encoding="utf-8")

    _rewrite(view, whole + "\n# this module never touches derived_ tables\n")
    assert "SG406" not in {f.code for f in rules.rule_adapters(tree)}, "a comment failed the build"

    _rewrite(view, whole + '\ndef _f():\n    """It reads no derived_ table."""\n')
    assert "SG406" not in {f.code for f in rules.rule_adapters(tree)}, "a docstring failed the build"

    _rewrite(view, whole + '\n_x = "SELECT 1 FROM derived_annotation"\n')
    assert "SG406" in {f.code for f in rules.rule_adapters(tree)}, "a real reach for SQL stopped failing"


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
    assert check(agreed.replace("JobState = Literal", "JobState = typing.Literal")) == [], (
        "a vocabulary spelled through the module is the same vocabulary"
    )

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

    `redirecting` is here as the case that used to pass and should not.
    Litestar builds one response schema from the whole annotation, so a
    union that mixes a JSON arm with a page or a redirect reaches the
    document as `application/json: {schema: {}}` -- measured on v2.24.0
    for `Template | Response[X]` and `Response[X] | Redirect` alike, where
    `Response[X]` on its own reaches it as a $ref. A precisely written arm
    inside such a union is a contract nobody is ever given.

    `picture` is the other side of it: bytes are not JSON, so a union that
    only ever answers a byte stream has nothing to declare.
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
        "def declared_vague() -> Template: ...\n"
        "@get('/m')\ndef picture() -> Response[bytes] | Redirect: ...\n"
        "@get('/n', responses={200: ResponseSpec(data_container=Named)})\n"
        "def declared_negotiating() -> Template | Response[Named]: ...\n",
    )
    broken = {"unnamed", "unnamed_list", "unparameterized", "negotiating", "declared_vague", "redirecting"}

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

    # The stale line is still found in a module that has stopped using
    # `connect` ALTOGETHER, which is the likeliest way for one to go
    # stale and the case the rule's own source-level skip could swallow:
    # it passes over a file with no `connect` in it, and must not pass
    # over one an excusal names.
    moved_on = rules.from_text("tests/opener.py", "def stores(k):\n    return k\n")
    assert _flagged(rules.rule_connection_lifetime([moved_on], ledger)) == {"stores"}, (
        "a ledger line outliving the last connection in its file is the one that most needs reporting"
    )


def test_the_structural_schema_sweeps_hold_and_each_can_fail():
    """SG710, SG711, SG712: three sweeps that used to be pytest.

    Each is decidable from the DDL alone -- build it in memory, ask the
    PRAGMAs -- so each moved to the linter, and each is bent here to prove
    the new gate has teeth rather than being a green sweep nobody can fail.
    """
    from sglint import schema_rules

    ddl = schema_rules.SCHEMA.read_text(encoding="utf-8")
    assert schema_rules.rule_foreign_key_indexes(ddl) == []
    assert schema_rules.rule_closed_columns(ddl) == []
    assert schema_rules.rule_index_prefixes(ddl) == []

    # SG710: drop the index a child's foreign key leads, and deleting the
    # parent starts scanning that child
    index = "CREATE INDEX job_event_job ON job_event(job_id, id);"
    assert ddl.count(index) == 1, "the control's handle moved"
    unindexed = schema_rules.rule_foreign_key_indexes(ddl.replace(index, ""))
    assert {f.code for f in unindexed} == {"SG710"}
    assert any("job_event.job_id" in f.message for f in unindexed)

    # SG711: a TEXT column named for a vocabulary, carrying no CHECK
    loose = ddl.replace(
        "CREATE TABLE rating (",
        "CREATE TABLE mood (id INTEGER PRIMARY KEY, state TEXT) STRICT;\nCREATE TABLE rating (",
        1,
    )
    told = schema_rules.rule_closed_columns(loose)
    assert {f.code for f in told} == {"SG711"}
    assert any("mood.state" in f.message for f in told)
    # and the exemption is read: the same column, named free text
    excused = schema_rules.rule_closed_columns(loose, free_text=schema_rules.policy.FREE_TEXT | {"state"})
    assert excused == [], "a column named in FREE_TEXT is a decision, not a finding"

    # SG712: an index whose columns are a prefix of another's, same predicate
    doubled = ddl.replace(index, index + "\nCREATE INDEX job_event_job_only ON job_event(job_id);", 1)
    prefixed = schema_rules.rule_index_prefixes(doubled)
    assert {f.code for f in prefixed} == {"SG712"}
    assert any("job_event_job_only is a prefix of job_event_job" in f.message for f in prefixed)
    # a partial index over different rows is NOT replaced by the wider one
    partial = ddl.replace(
        index, index + "\nCREATE INDEX job_event_job_open ON job_event(job_id) WHERE item_id IS NULL;", 1
    )
    assert schema_rules.rule_index_prefixes(partial) == [], "a different predicate answers for different rows"


def test_the_vocabulary_and_handler_rules_hold_and_can_fail(tmp_path):
    """SG713 and SG415: two parities that used to be pytest.

    SG713 compared `console.RENDERINGS` with `ledger.EventType` -- two
    module-level literals. SG415 asked Python for a handler's SOURCE and
    substring-searched it, which is a linter wearing a pytest nametag.
    """
    from sglint import schema_rules

    assert schema_rules.rule_vocabulary_handlers() == []
    assert schema_rules.rule_handlers_report() == []

    here = tmp_path / "repo"
    (here / "words").mkdir(parents=True)
    (here / "run").mkdir(parents=True)
    (here / "words" / "vocab.py").write_text('import typing\n\nKind = typing.Literal["a", "b"]\n', encoding="utf-8")
    handlers = {"words/say.py": {"SAYS": ("words/vocab.py", "Kind")}}

    # covered exactly
    (here / "words" / "say.py").write_text("SAYS = {'a': 1, 'b': 2}\n", encoding="utf-8")
    assert schema_rules.rule_vocabulary_handlers(here, handlers) == []
    # a member nothing covers, and words for a member that cannot be written
    (here / "words" / "say.py").write_text("SAYS = {'a': 1, 'c': 3}\n", encoding="utf-8")
    told = schema_rules.rule_vocabulary_handlers(here, handlers)
    assert {f.code for f in told} == {"SG713"}
    assert "missing ['b']" in told[0].message
    assert "unknown ['c']" in told[0].message

    # SG415: a handler that never reports, and a dispatcher whose modes do
    (here / "run" / "worker.py").write_text(
        "def _quiet(conn, item):\n    return 1\n\n"
        "def _loud(conn, item):\n    report().phase('x')\n    return 2\n\n"
        "HANDLERS = {'quiet': _quiet, 'loud': _loud}\n",
        encoding="utf-8",
    )
    silent = schema_rules.rule_handlers_report(here, {"run/worker.py": "HANDLERS"})
    assert {f.code for f in silent} == {"SG415"}
    assert len(silent) == 1
    assert "_quiet handles 'quiet'" in silent[0].message


def test_the_page_shape_rule_can_fail(tmp_path):
    """SG502: template inheritance and document shape, which used to be a
    pytest that walked the template directory reading files."""
    where = tmp_path / "templates"
    where.mkdir()
    shell = where / policy.SHELL_TEMPLATE
    shell.write_text("<!doctype html><html><body>{% block body %}{% endblock %}</body></html>", encoding="utf-8")
    page = where / "good.html"
    page.write_text(policy.EXTENDS_SHELL + "\n{% block body %}<p>fine</p>{% endblock %}\n", encoding="utf-8")
    fragment = where / "_good.html"
    fragment.write_text("<p>a piece</p>\n", encoding="utf-8")
    clean = [shell, page, fragment]
    assert rules._page_shapes(clean) == [], "the shell itself is not a page, and these two are right"

    orphan = where / "orphan.html"
    orphan.write_text("<p>no shell</p>\n", encoding="utf-8")
    told = rules._page_shapes([*clean, orphan])
    assert {f.code for f in told} == {"SG502"}
    assert "does not open with" in told[0].message

    whole = where / "whole.html"
    whole.write_text(policy.EXTENDS_SHELL + "\n<!DOCTYPE html>\n", encoding="utf-8")
    assert "carrying its own document" in rules._page_shapes([*clean, whole])[0].message

    swollen = where / "_swollen.html"
    swollen.write_text("<html><body>a fragment that grew</body></html>\n", encoding="utf-8")
    assert "is a page" in rules._page_shapes([*clean, swollen])[0].message


def test_the_before_marker_sweep_excuses_a_named_function_and_nothing_else(tmp_path):
    """SG407's cut-out: the planner's `engine_for` is allowed a connection
    before the marker, and nothing else is.

    Widening the word list instead would excuse `(conn` everywhere in the
    module, which is the opposite of what the pin is for.
    """
    here = tmp_path / "repo"
    (here / "db").mkdir(parents=True)
    module = here / "db" / "planner.py"
    marker = "# --- persistence"
    # annotated because `dict` is invariant in its value: a
    # `tuple[str, str]` nested in one is not a `tuple[str, ...]`,
    # even though the tuple alone would be.
    wanted: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
        "db/planner.py": (marker, ("(conn", "execute("), ("engine_for",))
    }
    clean = (
        "def engine_for(conn, name):\n    return conn.execute('SELECT 1')\n\n\n"
        "def plan(document):\n    return document\n\n\n" + marker + "\n\ndef save(conn):\n    pass\n"
    )
    module.write_text(clean, encoding="utf-8")
    assert rules._before_marker(here, wanted) == [], "the excused function is allowed both"

    # the same words, one function further along, are a finding
    module.write_text(
        clean.replace("def plan(document):\n    return document", "def plan(conn):\n    return conn"), encoding="utf-8"
    )
    told = rules._before_marker(here, wanted)
    assert {f.code for f in told} == {"SG407"}

    # and an excused name that is not there is itself a finding: a pin
    # cannot outlive the thing it excuses
    module.write_text(clean.replace("def engine_for(conn, name):", "def engine_of(conn, name):"), encoding="utf-8")
    assert any("nothing named engine_for" in f.message for f in rules._before_marker(here, wanted))


def test_the_written_column_sweep_reads_statements_not_prose():
    """SG714's control, moved with the rule it belongs to.

    Without it the sweep passes on a column that is only ever MENTIONED,
    which is how `file.width` and `file.height` read as produced while
    nothing had ever written either.
    """
    from sglint import schema_rules

    conn = schema_rules.built(schema_rules.SCHEMA.read_text(encoding="utf-8"))
    try:
        written, everything = schema_rules.written_columns(rules.REPO_ROOT, conn)
    finally:
        conn.close()
    assert "file" not in everything, "every INSERT into file names its columns"
    assert {"folder_id", "name", "content_sha256"} <= written["file"], (
        "the sweep cannot see the columns apply_scan plainly writes"
    )
    # a word this repo says constantly, and never as a column of `file`
    assert "parsed_by" not in written["file"]
    # filled entirely by a trigger, which is the half a Python-only sweep
    # would call dead
    assert "occurrences" in written["param_key"]


def test_the_written_column_rule_can_fail():
    """SG714: add a column nothing writes, and the DDL must say so."""
    from sglint import schema_rules

    ddl = schema_rules.SCHEMA.read_text(encoding="utf-8")
    assert schema_rules.rule_written_columns(ddl=ddl) == []
    handle = "CREATE TABLE rating ("
    assert ddl.count(handle) == 1, "the control's handle moved"
    silent = ddl.replace(handle, "CREATE TABLE weather (id INTEGER PRIMARY KEY, mood TEXT) STRICT;\n" + handle, 1)
    told = schema_rules.rule_written_columns(ddl=silent)
    assert {f.code for f in told} == {"SG714"}
    assert any("weather.mood" in f.message for f in told)
    # the admission in the comment block above it is the way out
    admitted = ddl.replace(
        handle,
        f"CREATE TABLE weather (\n    id INTEGER PRIMARY KEY,\n    -- {schema_rules.UNWRITTEN_ADMISSION}\n"
        "    mood TEXT\n) STRICT;\n" + handle,
        1,
    )
    assert schema_rules.rule_written_columns(ddl=admitted) == [], "an admitted column is a decision, not a finding"


def test_the_layer_boundary_holds_and_can_fail(tmp_path):
    """SG007: the rule the layers rest on.

    If an assertion can be decided from source, AST, schema structure,
    generated contracts or types without exercising behaviour, it is not a
    pytest. Every such sweep in this repository has moved to sglint; this
    is what stops the next one arriving.

    The predicate is QUALIFIED on purpose. A bare `parse` matched
    `facets.parse` and `resultset.parse` sixty-two times -- the
    application's own functions being called, which is the opposite of
    what this looks for.
    """
    here = tmp_path / "tests"
    here.mkdir()
    (here / "test_behaviour.py").write_text(
        "import facets\n\n\ndef test_a_facet_round_trips():\n    assert facets.parse(facets.spell(one)) == one\n",
        encoding="utf-8",
    )
    assert rules.rule_tests_run_things(here) == [], "calling the application's own parse is not source inspection"

    (here / "test_source.py").write_text(
        "import ast\nimport inspect\nimport pathlib\n\nfrom db import when\n\n\n"
        "def test_the_judge_ignores_the_folder():\n"
        "    assert 'folder_id' not in inspect.getsource(when)\n"
        "    ast.parse(pathlib.Path('db/when.py').read_text())\n"
        "    for one in pathlib.Path('db').rglob('*.py'):\n        assert one\n",
        encoding="utf-8",
    )
    told = rules.rule_tests_run_things(here)
    assert {f.code for f in told} == {"SG007"}
    assert len(told) == 3, [f.message for f in told]
    assert any("asks Python for a function's source" in f.message for f in told)
    assert any("parses production source" in f.message for f in told)
    assert any("sweeps the tree for Python source" in f.message for f in told)

    # the linter's own tests are the one place handing a rule source IS
    # the point, and they are excused by name
    assert rules.rule_tests_run_things(here, excused=frozenset({"test_source.py"})) == []
