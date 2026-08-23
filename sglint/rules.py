"""The structural rules this repository holds itself to, as a linter.

Each rule reads the tree -- one parsed copy, discovered rather than
listed -- and yields Findings in the shape every linter speaks:
`path:line:col: CODE message`. Nothing here imports the application;
a rule that understood nothing would report a clean tree, so
tests/test_sglint_has_teeth.py feeds each rule the shape it exists to
catch.

What Ruff can say, Ruff says: shell=True is S602, a missing check= is
PLW1510, a raw sqlite3.connect and a bare Image.open are TID251, a
module-level torch is TID253 (pyproject.toml). These are the rest.

Families:
    SG0xx  programs are started safely (subprocess); never from a test
    SG1xx  SQL is built from structure only
    SG4xx  the web adapters own no semantics
    SG5xx  templates and scripts carry no query logic
    SG6xx  every derived table has a producer something calls
    SG7xx  the schema contract (sglint/schema_rules.py)
"""

from __future__ import annotations

import ast
import dataclasses
import functools
import os
import pathlib
import re
import typing

from . import policy

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


@dataclasses.dataclass(frozen=True)
class Finding:
    path: pathlib.Path
    line: int
    col: int
    code: str
    message: str

    def spelled(self) -> str:
        rel = self.path.relative_to(REPO_ROOT) if self.path.is_relative_to(REPO_ROOT) else self.path
        return f"{rel.as_posix()}:{self.line}:{self.col}: {self.code} {self.message}"


# --- the tree ---------------------------------------------------------------------------------


@functools.cache
def _parsed_as_of(source: pathlib.Path, _stamp: tuple[int, int]) -> ast.Module:
    return ast.parse(source.read_text(encoding="utf-8"))


def parsed(source: pathlib.Path) -> ast.Module:
    """The file's tree, parsed once per (mtime, size): a file rewritten
    under the linter re-parses itself and nothing else."""
    held = source.stat()
    return _parsed_as_of(source, (held.st_mtime_ns, held.st_size))


@functools.cache
def every_source() -> tuple[pathlib.Path, ...]:
    """Every .py file this repository owns, discovered rather than listed."""
    found: list[pathlib.Path] = []
    for current, subdirs, names in os.walk(REPO_ROOT):
        subdirs[:] = sorted(d for d in subdirs if d not in policy.NOT_OURS)
        found.extend(pathlib.Path(current) / name for name in sorted(names) if name.endswith(".py"))
    return tuple(found)


@functools.cache
def shipped() -> tuple[pathlib.Path, ...]:
    """The application as a user receives it: every source outside tooling."""
    return tuple(p for p in every_source() if p.relative_to(REPO_ROOT).parts[0] not in policy.TOOLING)


# --- SG0xx: programs are started safely ------------------------------------------------------

_SPAWNERS = frozenset({"run", "call", "check_call", "check_output", "Popen"})


def spawn_calls(tree: ast.AST) -> list[ast.Call]:
    """Every subprocess.<spawner>(...) call node in `tree`."""
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _SPAWNERS:
            root = node.func.value
            if isinstance(root, ast.Name) and root.id == "subprocess":
                calls.append(node)
    return calls


def spawner_name(call: ast.Call) -> str:
    """The subprocess.<name> a spawn_calls() hit names."""
    if not isinstance(call.func, ast.Attribute):
        raise TypeError("not a spawn call")
    return call.func.attr


def keyword(call: ast.Call, name: str):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def bare_command_string(call: ast.Call) -> bool:
    return bool(call.args) and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)


def is_pipe(given) -> bool:
    if isinstance(given, ast.Name) and given.id == "PIPE":
        return True
    return (
        isinstance(given, ast.Attribute)
        and given.attr == "PIPE"
        and isinstance(given.value, ast.Name)
        and given.value.id == "subprocess"
    )


def pipes_output(call: ast.Call) -> bool:
    """Popen handed a PIPE this repo has nobody reading."""
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "Popen":
        return False
    return any(is_pipe(keyword(call, stream)) for stream in ("stdout", "stderr"))


def rule_spawns(
    sources: typing.Iterable[pathlib.Path] | None = None, tests: pathlib.Path = REPO_ROOT / "tests"
) -> list[Finding]:
    """SG002-SG004 over every source; SG006 for any spawn under tests/: a
    test never starts a program -- what needs one is a lint rule
    (`python -m sglint --repo`) or a just recipe. A shell (S602) and a
    missing check= (PLW1510) are Ruff's own."""
    found: list[Finding] = []
    for source in sources if sources is not None else every_source():
        for call in spawn_calls(parsed(source)):
            attr = spawner_name(call)
            at = (source, call.lineno, call.col_offset)
            if source.is_relative_to(tests):
                found.append(
                    Finding(*at, "SG006", "a test starts a program; move the check to sglint --repo or a just recipe")
                )
            if bare_command_string(call):
                found.append(Finding(*at, "SG002", "passes a bare command string instead of a list"))
            if attr != "Popen" and keyword(call, "timeout") is None:
                found.append(Finding(*at, "SG003", "starts a program with no timeout"))
            if pipes_output(call):
                found.append(Finding(*at, "SG004", "hands a child a pipe nobody drains; sink to a file instead"))
    return found


# --- SG1xx: SQL is built from structure only --------------------------------------------------

SQL_SHAPED = re.compile(
    r"\b(select\s+.+\s+from\b|insert\s+into\b|update\s+\w+\s+set\b|delete\s+from\b|where\b|order\s+by\b)",
    re.IGNORECASE | re.DOTALL,
)


def sql_interpolations(tree: ast.AST) -> list[tuple[str, int, int]]:
    """[(slot, line, col)] for every interpolation in a SQL-shaped f-string."""
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        literal = "".join(
            part.value for part in node.values if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
        if not SQL_SHAPED.search(literal):
            continue
        found.extend(
            (ast.unparse(part.value), node.lineno, node.col_offset)
            for part in node.values
            if isinstance(part, ast.FormattedValue)
        )
    return found


def rule_sql_structure(sources: typing.Iterable[pathlib.Path] | None = None) -> list[Finding]:
    """SG101: a value written into a statement. SG102: a structure name
    listed but no longer written anywhere -- the list is the tree's truth."""
    found: list[Finding] = []
    written: set[str] = set()
    for source in sources if sources is not None else shipped():
        for slot, line, col in sql_interpolations(parsed(source)):
            written.add(slot)
            if slot not in policy.SQL_STRUCTURE:
                found.append(
                    Finding(
                        source,
                        line,
                        col,
                        "SG101",
                        f"{slot!r} is written into a SQL statement; bind it as ? or, if it is structure this"
                        " codebase wrote, add it to sglint/policy.py SQL_STRUCTURE and say which kind",
                    )
                )
    if sources is None:
        found.extend(
            Finding(
                REPO_ROOT / "sglint" / "policy.py",
                1,
                0,
                "SG102",
                f"SQL_STRUCTURE lists {stale!r} but nothing writes it any more; remove the line",
            )
            for stale in sorted(set(policy.SQL_STRUCTURE) - written)
        )
    return found


# --- SG4xx: the web adapters own no semantics ------------------------------------------------


def _called_attrs(tree: ast.AST) -> dict[str, tuple[int, int]]:
    found: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            found.setdefault(node.func.attr, (node.lineno, node.col_offset))
    return found


def _called_qualified(tree: ast.AST) -> dict[str, tuple[int, int]]:
    found: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        ):
            found.setdefault(f"{node.func.value.id}.{node.func.attr}", (node.lineno, node.col_offset))
    return found


def _db_vocabulary(tree: ast.AST) -> dict[str, tuple[int, int]]:
    found: dict[str, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "db":
            for alias in node.names:
                found.setdefault(alias.name, (node.lineno, node.col_offset))
    return found


def rule_adapters(root: pathlib.Path = REPO_ROOT) -> list[Finding]:
    """SG401 an adapter ran its own statement; SG402 it imported a query
    module; SG403 it stopped delegating; SG404 a one-item adapter no
    longer shares the _many implementation; SG405 a non-literal statement
    where only literals may run; SG406 a forbidden word, or a required
    word missing; SG407 a word before a marker or after the docstring;
    SG408 a parameter a signature may not take; SG410 a forbidden
    pattern in a package; SG411 the page queries no longer ship here."""
    found: list[Finding] = []
    for relative, vocabulary in policy.ADAPTER_DB_VOCABULARY.items():
        path = root / relative
        tree = parsed(path)
        for attr, (line, col) in _called_attrs(tree).items():
            if attr in policy.STATEMENT_METHODS:
                found.append(Finding(path, line, col, "SG401", f"ran its own statement ({attr}); queries live in db/"))
        for name, (line, col) in _db_vocabulary(tree).items():
            if name not in vocabulary:
                found.append(
                    Finding(path, line, col, "SG402", f"imports db.{name}; this adapter may speak {sorted(vocabulary)}")
                )
        required = policy.ADAPTER_MUST_CALL.get(relative, frozenset())
        found.extend(
            Finding(path, 1, 0, "SG403", f"stopped delegating: no call to {missing}()")
            for missing in sorted(required - set(_called_attrs(tree)))
        )
    for relative, required_q in policy.MUST_CALL_QUALIFIED.items():
        path = root / relative
        seen = _called_qualified(parsed(path))
        found.extend(
            Finding(path, 1, 0, "SG403", f"stopped consuming the seam: no call to {missing}()")
            for missing in sorted(required_q - set(seen))
        )
    for relative, forbidden_q in policy.MUST_NOT_CALL_QUALIFIED.items():
        path = root / relative
        for name, (line, col) in _called_qualified(parsed(path)).items():
            if name in forbidden_q:
                found.append(Finding(path, line, col, "SG403", f"calls {name}(), which has no business on this path"))
    for relative, (module, name) in policy.MUST_IMPORT.items():
        path = root / relative
        imported = {
            alias.name
            for node in ast.walk(parsed(path))
            if isinstance(node, ast.ImportFrom) and node.module == module
            for alias in node.names
        }
        if name not in imported:
            found.append(Finding(path, 1, 0, "SG403", f"stopped importing {name} from {module}"))
    bodies: dict[str, str] = {}
    for relative in policy.ONE_TO_MANY_MODULES:
        tree = parsed(root / relative)
        bodies |= {node.name: ast.unparse(node) for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    for one, many in policy.ONE_DELEGATES_TO_MANY:
        where = root / policy.ONE_TO_MANY_MODULES[0]
        if one not in bodies or many not in bodies[one]:
            found.append(Finding(where, 1, 0, "SG404", f"{one} stopped delegating to {many}"))
        if many in bodies and "executemany" not in bodies[many]:
            found.append(Finding(where, 1, 0, "SG404", f"{many} no longer writes with executemany"))
    for relative in policy.LITERAL_STATEMENTS_ONLY:
        path = root / relative
        for node in ast.walk(parsed(path)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
                statement = node.args[0] if node.args else None
                if not isinstance(statement, ast.Constant):
                    found.append(
                        Finding(
                            path,
                            node.lineno,
                            node.col_offset,
                            "SG405",
                            "a formatted statement is a road from stored text to execution; every statement here"
                            " must be a literal",
                        )
                    )
    for relative, words in policy.MUST_NOT_CONTAIN.items():
        held = (root / relative).read_text(encoding="utf-8")
        for word in words:
            if word in held:
                line = held[: held.index(word)].count("\n") + 1
                found.append(
                    Finding(root / relative, line, 0, "SG406", f"carries {word!r}, which was deleted on purpose")
                )
    for relative, words in policy.MUST_CONTAIN.items():
        held = (root / relative).read_text(encoding="utf-8")
        found.extend(
            Finding(root / relative, 1, 0, "SG406", f"no longer carries {word!r}") for word in words if word not in held
        )
    for relative, (marker, words) in policy.MUST_NOT_CONTAIN_BEFORE.items():
        held = (root / relative).read_text(encoding="utf-8")
        head = held.split(marker, 1)[0]
        for word in words:
            if word in head:
                line = head[: head.index(word)].count("\n") + 1
                found.append(Finding(root / relative, line, 0, "SG407", f"carries {word!r} before {marker!r}"))
    for relative, words in policy.MUST_NOT_CONTAIN_AFTER_DOCSTRING.items():
        held = (root / relative).read_text(encoding="utf-8")
        body = held.split('"""', 2)[2] if held.count('"""') >= 2 else held
        for word in words:
            if word in body:
                line = held[: len(held) - len(body) + body.index(word)].count("\n") + 1
                found.append(Finding(root / relative, line, 0, "SG407", f"carries {word!r} outside its docstring"))
    for relative, pairs in policy.NO_PARAMETER_NAMED.items():
        tree = parsed(root / relative)
        for dotted, param in pairs:
            owner, method = dotted.split(".")
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == owner:
                    for fn in node.body:
                        if not (isinstance(fn, ast.FunctionDef) and fn.name == method):
                            continue
                        if any(a.arg == param for a in fn.args.args + fn.args.kwonlyargs):
                            found.append(
                                Finding(root / relative, fn.lineno, fn.col_offset, "SG408", f"{dotted} takes {param!r}")
                            )
    for package, patterns in policy.PACKAGE_FORBIDDEN_PATTERNS.items():
        for source in sorted((root / package).rglob("*.py")):
            held = source.read_text(encoding="utf-8")
            for pattern, why in patterns:
                for match in re.finditer(pattern, held, re.IGNORECASE):
                    line = held[: match.start()].count("\n") + 1
                    found.append(Finding(source, line, 0, "SG410", why))
    pages = parsed(root / "db" / "pages.py")
    shipped_queries = [
        node
        for node in pages.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets)
        and "SELECT" in ast.unparse(node.value)
    ]
    if len(shipped_queries) < policy.PAGE_QUERIES_MINIMUM:
        found.append(
            Finding(
                root / "db" / "pages.py",
                1,
                0,
                "SG411",
                f"only {len(shipped_queries)} page queries ship here; a page restated its query elsewhere",
            )
        )
    return found


# --- SG5xx: templates and scripts carry no query logic --------------------------------------


def rule_surfaces(root: pathlib.Path = REPO_ROOT) -> list[Finding]:
    """SG501: a template or script carries query logic or a /search side channel."""
    web = root / "sg_web"
    authored = root / "frontend" / "src"
    # The browser source is authored TypeScript under frontend/src; what
    # sg_web/static holds is vendored htmx and esbuild's generated bundles,
    # neither of which this sweep judges.
    templates = sorted((web / "templates").glob("*.html"))
    # frontend/src/generated is the browser's view of the application's own
    # OpenAPI document. It is not authored, and it names every route the
    # application serves -- including /search -- so sweeping it for the words
    # a hand-written surface must not contain would fail on the contract
    # itself rather than on anybody's code.
    scripts = sorted(p for p in authored.rglob("*.ts") if "generated" not in p.parts)
    found: list[Finding] = []
    for what, where, sources, minimum in (
        ("templates", web / "templates", templates, policy.TEMPLATE_MINIMUM),
        ("scripts", authored, scripts, policy.SCRIPT_MINIMUM),
    ):
        if len(sources) < minimum:
            found.append(Finding(where, 1, 0, "SG500", f"the sweep lost its {what}: {len(sources)} of {minimum}"))
    surfaces = [*templates, *scripts]
    for source in surfaces:
        held = source.read_text(encoding="utf-8")
        for word in policy.SURFACE_FORBIDDEN_WORDS:
            if word in held:
                line = held[: held.index(word)].count("\n") + 1
                found.append(Finding(source, line, 0, "SG501", f"carries query logic: {word!r}"))
    return found


# --- SG6xx: every derived table has a producer something calls ------------------------------


def derived_tables(schema_sql: str) -> set[str]:
    return set(re.findall(r"^CREATE TABLE (derived_\w+) \(", schema_sql, re.MULTILINE))


def unwired(tables: set[str], root: pathlib.Path = REPO_ROOT) -> dict[str, str]:
    """{table: why} for derived tables whose INSERT nobody outside
    db/derived.py can reach."""
    derived_source = (root / "db" / "derived.py").read_text(encoding="utf-8")
    module = ast.parse(derived_source)
    functions = [node for node in ast.walk(module) if isinstance(node, ast.FunctionDef)]
    named = {node.name for node in functions}
    writers: dict[str, set[str]] = {}
    callers: dict[str, set[str]] = {}
    for node in functions:
        body = ast.get_source_segment(derived_source, node) or ""
        for table in tables:
            if f"INTO {table}" in body:
                writers.setdefault(table, set()).add(node.name)
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in named:
                callers.setdefault(call.func.id, set()).add(node.name)

    def reachers(direct: set[str]) -> set[str]:
        reached = set(direct)
        grew = True
        while grew:
            grew = False
            for callee in tuple(reached):
                for caller in callers.get(callee, ()):
                    if caller not in reached:
                        reached.add(caller)
                        grew = True
        return reached

    elsewhere = "\n".join(
        source.read_text(encoding="utf-8")
        for package in policy.DERIVED_PRODUCER_PACKAGES
        for source in sorted((root / package).glob("*.py"))
        if source.name != "derived.py"
    )
    out: dict[str, str] = {}
    for table in sorted(tables):
        direct = writers.get(table)
        if direct is None:
            if f"INTO {table}" not in elsewhere:
                out[table] = f"no INSERT anywhere in {', '.join(policy.DERIVED_PRODUCER_PACKAGES)}"
        elif not any(f".{writer}(" in elsewhere for writer in reachers(direct)):
            out[table] = f"writer {sorted(direct)} called by nothing outside db/derived.py"
    return out


def rule_producers(root: pathlib.Path = REPO_ROOT) -> list[Finding]:
    """SG601 a derived table nothing produces; SG602 a reserved table that
    got wired (remove its DERIVED_RESERVED line)."""
    schema = (root / "db" / "schema.sql").read_text(encoding="utf-8")
    tables = derived_tables(schema)
    found: list[Finding] = []
    if not tables:
        found.append(Finding(root / "db" / "schema.sql", 1, 0, "SG600", "the schema lost its derived namespace"))
    missing = unwired(tables, root)
    for table, why in missing.items():
        if table not in policy.DERIVED_RESERVED:
            found.append(
                Finding(root / "db" / "schema.sql", 1, 0, "SG601", f"{table}: {why} -- a producer came unwired")
            )
    found.extend(
        Finding(root / "sglint" / "policy.py", 1, 0, "SG602", f"{table} got wired; remove its DERIVED_RESERVED line")
        for table in policy.DERIVED_RESERVED
        if table in tables and table not in missing
    )
    return found


# --- SG4xx: the request contracts ----------------------------------------------------------------


#: The decorators that make a function a route with a request body.
_BODY_ROUTES = frozenset({"post", "put", "patch", "delete", "route"})


def _wire_contracts(root: pathlib.Path) -> set[str]:
    """Every class in sg_web that inherits Wire, by name."""
    named: set[str] = set()
    for source in sorted((root / "sg_web").glob("*.py")):
        for node in ast.walk(parsed(source)):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(base, ast.Name) and base.id == "Wire" for base in node.bases
            ):
                named.add(node.name)
    return named


def _annotation_name(node: ast.expr | None) -> str | None:
    """The contract an annotation names, seeing through `X | None`."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        for side in (node.left, node.right):
            if isinstance(side, ast.Constant) and side.value is None:
                continue
            return _annotation_name(side)
    if isinstance(node, ast.Subscript):
        return _annotation_name(node.value)
    return None


def rule_request_contracts(root: pathlib.Path = REPO_ROOT, reserved: frozenset[str] | None = None) -> list[Finding]:
    """SG412: a route's JSON body is not a Wire contract.

    sg_web/wire.py states one policy for every JSON shape crossing the
    seam -- name every field, refuse the rest, translate rather than
    coerce. A body annotated `dict` obeys none of it and a dataclass obeys
    only some, and either way the OpenAPI document describes nothing, so
    the browser's generated types cannot describe the request either.

    Forms are exempt by shape, not by name: URLEncodedBody carries a form,
    which is a different contract with different rules.
    """
    contracts = _wire_contracts(root)
    excused = policy.REQUEST_CONTRACT_RESERVED if reserved is None else reserved
    found: list[Finding] = []
    for source in sorted((root / "sg_web").glob("*.py")):
        for node in ast.walk(parsed(source)):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            routed = any(
                isinstance(one, ast.Call) and isinstance(one.func, ast.Name) and one.func.id in _BODY_ROUTES
                for one in node.decorator_list
            )
            if not routed:
                continue
            for argument in (*node.args.args, *node.args.kwonlyargs):
                if argument.arg != "data":
                    continue
                named = _annotation_name(argument.annotation)
                if named == "URLEncodedBody":
                    continue
                relative = source.relative_to(root).as_posix()
                if f"{relative}:{node.name}" in excused:
                    continue
                if named not in contracts:
                    found.append(
                        Finding(
                            source,
                            argument.lineno,
                            argument.col_offset,
                            "SG412",
                            f"{node.name} takes a JSON body typed {named or 'nothing'}, which is not a Wire contract",
                        )
                    )
    return found


# --- all of it ----------------------------------------------------------------------------------

RULES = (
    rule_spawns,
    rule_sql_structure,
    rule_adapters,
    rule_surfaces,
    rule_producers,
    rule_request_contracts,
)


def run() -> list[Finding]:
    from . import schema_rules

    found: list[Finding] = []
    for rule in (
        *RULES,
        schema_rules.rule_schema,
        schema_rules.rule_migrations,
        schema_rules.rule_wire_vocabularies,
    ):
        found.extend(rule())
    return sorted(found, key=lambda f: (str(f.path), f.line, f.col, f.code))
