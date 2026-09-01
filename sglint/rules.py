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
import tokenize
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


# A file's TEXT is not cached on (mtime, size): the stamp cannot separate two
# same-length rewrites inside one clock tick, which would hand a rule source it
# never saw. `walked` below is keyed on the tree object instead.

#: Files that would not parse, and why. Written by `parsed` and reported by
#: `rule_sources_parse`, so every other rule still runs over the files that
#: can be read.
_UNPARSEABLE: dict[pathlib.Path, str] = {}

#: What a caller gets for a file that will not parse. Empty, so a rule walking
#: it finds nothing rather than crashing, and never silently -- SG011 reports
#: the file, so "no findings here" can only mean the file was read.
_NOTHING = ast.Module(body=[], type_ignores=[])


@functools.cache
def _parsed_as_of(source: pathlib.Path, _stamp: tuple[int, int]) -> ast.Module:
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as why:
        _UNPARSEABLE[source] = f"cannot be read: {why}"
        return _NOTHING
    try:
        return ast.parse(text)
    except SyntaxError as why:
        where = f" at line {why.lineno}" if why.lineno else ""
        _UNPARSEABLE[source] = f"{why.msg}{where}"
        return _NOTHING
    except ValueError as why:
        # A null byte, or a source too deeply nested for the compiler.
        # `ast.parse` raises these outside SyntaxError.
        _UNPARSEABLE[source] = str(why)
        return _NOTHING


def parsed(source: pathlib.Path) -> ast.Module:
    """The file's tree, parsed once per (mtime, size): a file rewritten
    under the linter re-parses itself and nothing else.

    NEVER RAISES. A file that will not parse comes back empty and is
    recorded in `_UNPARSEABLE` for SG011 to report, because one bad file
    must not be able to stop every rule from running over every good one.
    """
    try:
        held = source.stat()
    except OSError as why:
        _UNPARSEABLE[source] = f"cannot be read: {why}"
        return _NOTHING
    return _parsed_as_of(source, (held.st_mtime_ns, held.st_size))


def rule_sources_parse(root: pathlib.Path = REPO_ROOT) -> list[Finding]:
    """SG011: every Python file this repository owns parses.

    Runs first, and reads everything, so the answer to "why did the
    linter say nothing about that file" is always in the output rather
    than in a traceback. A file that cannot be parsed cannot be checked,
    and a check that was never made must not look like a check that
    passed.
    """
    for source in every_source():
        parsed(source)
    return [
        Finding(source, 1, 0, "SG011", f"does not parse, so no rule can read it -- {why}")
        for source, why in sorted(_UNPARSEABLE.items())
        if source.is_relative_to(root)
    ]


#: Walked trees, keyed on the tree itself: `ast.AST` hashes by identity, so the
#: dict holds each tree alive for as long as it holds its walk. Keying on `id()`
#: would not, a collected tree's id being reusable by the next tree along.
_WALKED: dict[ast.AST, tuple[ast.AST, ...]] = {}


def walked(tree: ast.AST) -> tuple[ast.AST, ...]:
    """Every node of `tree`, walked once and kept.

    Thirty-odd rules read the same two hundred and fifty modules, and
    each was walking every one of them from the top. `ast.walk` is a
    breadth-first generator over a deque, so that is a full traversal
    and a fresh deque per rule per file; this is one traversal per file,
    iterated as a tuple thereafter.
    """
    held = _WALKED.get(tree)
    if held is None:
        held = tuple(ast.walk(tree))
        _WALKED[tree] = held
    return held


#: The Call nodes of each tree, filtered once. Same keying, same reason.
_CALLS: dict[ast.AST, tuple[ast.Call, ...]] = {}


def calls(tree: ast.AST) -> tuple[ast.Call, ...]:
    """Every call in `tree`, filtered once and kept.

    Eleven places ask a tree for its calls -- the spawn sweep, the SQL
    sweep, the statement rules, the route rules -- and each was walking
    every node of the file to find them. A module is mostly not calls, so
    that is a thousand `isinstance` checks each to reach a few dozen
    nodes, once per rule that asks.

    In document order, because it is a filter of `walked` and rules
    report the FIRST offence they meet.
    """
    held = _CALLS.get(tree)
    if held is None:
        held = tuple(node for node in walked(tree) if isinstance(node, ast.Call))
        _CALLS[tree] = held
    return held


#: The functions of each tree, filtered once. Same keying, same reason.
_FUNCTIONS: dict[ast.AST, tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]] = {}


def functions(tree: ast.AST) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...]:
    """Every function in `tree`, async ones included, in document order.

    Six rules ask a tree for its functions, and the connection-lifetime
    sweep asks it of EVERY source in the repository -- each time walking
    the whole file to find the handful of nodes that are functions.
    """
    held = _FUNCTIONS.get(tree)
    if held is None:
        held = tuple(node for node in walked(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef))
        _FUNCTIONS[tree] = held
    return held


def _code_only(source: pathlib.Path, held: str) -> str:
    """`held` with COMMENTS and DOCSTRINGS blanked, offsets preserved.

    A pure function of the text, so it is cached on the text: several
    rules read the same module, and blanking it is a tokenize pass plus
    a rewrite of every character in the file.

    A ban says what a module must not DO, and a comment is prose ABOUT
    what it does. `db/evolution.py` may not DELETE, and a comment saying
    "this module never DELETEs" is the module agreeing with the rule and
    being failed for it -- and the failure then reads as an architecture
    violation rather than a word, which is how `media_view.py:
    "neighbour"` came to be DELETED rather than satisfied.

    String literals are deliberately KEPT. `sg_web/story_view.py` may
    not reach for SQL and SQL lives in a string; blanking those would
    turn a real ban into a decoration. What goes is prose, not code.

    Only Python: several of these files are Jinja templates, where there
    is no such thing as a docstring and `{# #}` is not a Python comment.
    Their text is returned as it stands.

    Blanked with SPACES rather than removed, so a finding's line number
    still points at the line it came from.
    """
    return _blanked(source.suffix, held)


@functools.cache
def _blanked(suffix: str, held: str) -> str:
    if suffix != ".py":
        return held
    import io as _io
    import tokenize

    starts = [0]
    for line in held.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))

    def at(row: int, col: int) -> int:
        return starts[row - 1] + col

    out = list(held)

    def blank(begin: int, end: int) -> None:
        for i in range(begin, min(end, len(out))):
            if out[i] != chr(10):
                out[i] = " "

    try:
        for token in tokenize.generate_tokens(_io.StringIO(held).readline):
            if token.type == tokenize.COMMENT:
                blank(at(*token.start), at(*token.end))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # A file the tokenizer will not read is one every other rule here
        # already fails on. The ban falls back to the whole text rather
        # than silently checking nothing.
        return held
    # The text's own tree, not `parsed(source)`: this function answers for
    # the text it was handed, and a file rewritten inside one clock tick at
    # the same length would hand back the previous one.
    for node in ast.walk(ast.parse(held)):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and first.end_lineno is not None
        ):
            blank(at(first.lineno, first.col_offset), at(first.end_lineno, first.end_col_offset or 0))
    return "".join(out)


@dataclasses.dataclass(frozen=True)
class Source:
    """One module a rule reads: what to call it, and its tree.

    A rule is text in, findings out. Taking paths made that untrue: a
    control had to write a file to ask a question, which on Windows costs
    more than the rule does, and `parsed` caches on (mtime, size) -- so
    two rewrites inside one clock tick at the same length hand back the
    FIRST tree and the control silently checks source it did not write.
    """

    relative: str
    tree: ast.Module

    @property
    def path(self) -> pathlib.Path:
        return REPO_ROOT / self.relative


def on_disk(path: pathlib.Path, root: pathlib.Path = REPO_ROOT) -> Source:
    relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
    return Source(relative, parsed(path))


def from_text(relative: str, text: str) -> Source:
    """A module written as a string. What the controls use."""
    return Source(relative, ast.parse(text))


def web_sources() -> list[Source]:
    """Every module that carries the HTTP seam."""
    return [on_disk(one) for one in sorted((REPO_ROOT / "sg_web").glob("*.py"))]


def shipped_sources() -> list[Source]:
    """Every module outside tooling, as trees.

    SG018's scope, stated where it can be read: the application a user
    receives AND the proof harness, which is where a swallowed error
    turns a measurement into a silently degraded one. `tests`,
    `benchmarks` and `sglint` itself author failure shapes deliberately
    -- handing them to the rule would be judging the controls by the
    thing they exist to catch.
    """
    return [on_disk(one) for one in shipped()]


@functools.cache
def every_source() -> tuple[pathlib.Path, ...]:
    """Every .py file this repository owns, discovered rather than listed."""
    found: list[pathlib.Path] = []
    for current, subdirs, names in os.walk(REPO_ROOT):
        here = pathlib.Path(current)
        # A virtualenv is not ours whatever it is CALLED: a second environment
        # beside `.venv` walked as repository source can stop every rule on one
        # third-party file. `pyvenv.cfg` makes a directory an environment (PEP 405).
        at_root = here == REPO_ROOT
        subdirs[:] = sorted(
            d
            for d in subdirs
            if d not in policy.NOT_OURS
            and not (at_root and d in policy.NOT_OURS_AT_ROOT)
            and not (here / d / "pyvenv.cfg").is_file()
        )
        found.extend(here / name for name in sorted(names) if name.endswith(".py"))
    return tuple(found)


@functools.cache
def shipped() -> tuple[pathlib.Path, ...]:
    """The application as a user receives it: every source outside tooling."""
    return tuple(p for p in every_source() if p.relative_to(REPO_ROOT).parts[0] not in policy.TOOLING)


# --- SG0xx: programs are started safely ------------------------------------------------------

_SPAWNERS = frozenset({"run", "call", "check_call", "check_output", "Popen"})


def spawn_calls(tree: ast.AST) -> list[ast.Call]:
    """Every subprocess.<spawner>(...) call node in `tree`."""
    found = []
    for node in calls(tree):
        if isinstance(node.func, ast.Attribute) and node.func.attr in _SPAWNERS:
            root = node.func.value
            if isinstance(root, ast.Name) and root.id == "subprocess":
                found.append(node)
    return found


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
    """A child handed a pipe, by any of the spellings that hand one.

    `capture_output=True` IS two pipes: it is documented as shorthand for
    `stdout=PIPE, stderr=PIPE`. `check_output` pipes stdout by construction.
    A rule reading only `Popen` passes both, and 22 call sites carrying the
    exact defect it describes went through the gate.

    The cost was not theoretical. `subprocess.run(argv, capture_output=True,
    timeout=N)` kills the child when the timeout fires and then calls
    `communicate()` a SECOND time with no timeout to drain those pipes; that
    call waits for every handle on the write end to close, and a grandchild
    inherited them. `just compat hf` stalled over forty minutes with its own
    timeout already fired.
    """
    if not isinstance(call.func, ast.Attribute):
        return False
    if call.func.attr == "check_output":
        # Pipes stdout by construction: there is no spelling of it that does not.
        return True
    captured = keyword(call, "capture_output")
    if isinstance(captured, ast.Constant) and captured.value is True:
        return True
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


#: How a test inspects production SOURCE rather than exercising it, as (module,
#: function): each is a linter wearing a pytest nametag, able to fail without
#: running the thing whose behaviour it claims.

#: Qualified, and that matters: `parse` alone also matches `facets.parse` and
#: `resultset.parse`, the application's own functions being CALLED.
_SOURCE_INSPECTION = {
    ("inspect", "getsource"): "asks Python for a function's source and searches it",
    ("inspect", "getsourcelines"): "asks Python for a function's source and searches it",
    ("ast", "parse"): "parses production source",
}


def rule_tests_run_things(
    tests: pathlib.Path = REPO_ROOT / "tests", excused: frozenset[str] | None = None
) -> list[Finding]:
    """SG007: a test that inspects source instead of running the thing.

    The rule the layers are built on: if an assertion can be decided from
    source, AST, schema structure, generated contracts or types WITHOUT
    exercising behaviour, it is not a pytest. Every such sweep in this
    repository has moved to sglint, and this is what stops the next one
    arriving -- `inspect.getsource`, `ast.parse`, and a `*.py` glob over
    the tree are the three shapes they all took.

    The linter's own tests are exempt by name: proving a rule fires means
    handing it source, which is the one place that is the point.
    """
    skip = policy.SOURCE_INSPECTION_EXCUSED if excused is None else excused
    found: list[Finding] = []
    for source in sorted(tests.rglob("*.py")):
        if source.name in skip:
            continue
        for node in calls(parsed(source)):
            if not isinstance(node.func, ast.Attribute):
                continue
            named = node.func.attr
            owner = node.func.value.id if isinstance(node.func.value, ast.Name) else ""
            why = _SOURCE_INSPECTION.get((owner, named))
            if why is not None:
                found.append(
                    Finding(source, node.lineno, node.col_offset, "SG007", f"a test {why}; sglint can prove it cheaper")
                )
            if named in ("glob", "rglob") and node.args:
                held = node.args[0]
                if isinstance(held, ast.Constant) and isinstance(held.value, str) and held.value.endswith(".py"):
                    found.append(
                        Finding(
                            source,
                            node.lineno,
                            node.col_offset,
                            "SG007",
                            "a test sweeps the tree for Python source; sglint can prove it cheaper",
                        )
                    )
    return found


# --- SG1xx: SQL is built from structure only --------------------------------------------------

SQL_SHAPED = re.compile(
    r"\b(select\s+.+\s+from\b|insert\s+into\b|update\s+\w+\s+set\b|delete\s+from\b|where\b|order\s+by\b)",
    re.IGNORECASE | re.DOTALL,
)


def sql_interpolations(tree: ast.AST) -> list[tuple[str, int, int]]:
    """[(slot, line, col)] for every interpolation in a SQL-shaped f-string."""
    found = []
    for node in walked(tree):
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


# --- SG1xx: a connection is closed by whoever opened it ----------------------------------------

#: How this application opens a database. sqlite3.connect is banned
#: everywhere else (pyproject.toml TID251), so these two are all of them.
_OPENS = frozenset({"connect", "memory"})


def _opened_here(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[str | None, ast.Call]]:
    """[(name bound, call)] for every connection this function opens."""
    held: list[tuple[str | None, ast.Call]] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            call, targets = node.value, node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            call, targets = node.value, [node.target]
        else:
            continue
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "connect"
            and call.func.attr in _OPENS
        ):
            continue
        named = next((one.id for one in targets if isinstance(one, ast.Name)), None)
        held.append((named, call))
    return held


def _handed_onward(fn: ast.FunctionDef | ast.AsyncFunctionDef, name: str | None) -> bool:
    """Whether the function structurally gives the connection away.

    Returned or yielded, and nothing else. A fixture usually returns it
    inside the tuple or dict it hands back beside the ids it minted, so the
    whole returned expression is searched rather than only a bare
    `return conn`.

    Storing it -- `self.conn = conn`, `registry[key] = conn` -- is NOT a
    transfer. It shows the connection left this function; it shows nothing
    about anyone closing it, and accepting it would let any leak be
    silenced by putting it in an object. The two places this repository
    really does keep one for the life of the process are named in
    policy.CONNECTION_KEPT, each with its reason.
    """
    if name is None:
        # Bound to an attribute or a subscript and to no name at all:
        # stored, which is the shape this rule refuses to read as a
        # transfer.
        return False
    for node in ast.walk(fn):
        if isinstance(node, ast.Return | ast.Yield) and node.value is not None and _carries(node.value, name):
            return True
    return False


def _carries(held: ast.expr, name: str) -> bool:
    """Whether this returned expression hands over the object itself.

    Structurally, not by mention. `return conn` and the collection
    literals a fixture builds around it hand it over; `return f"v{conn...}"`
    returns a string that merely READ the connection, and db/resultset.py's
    monitor is exactly that -- taking it as a transfer excused the one
    connection in this repository that most needed declaring.
    """
    if isinstance(held, ast.Name):
        return held.id == name
    if isinstance(held, ast.Tuple | ast.List | ast.Set):
        return any(_carries(one, name) for one in held.elts)
    if isinstance(held, ast.Dict):
        return any(one is not None and _carries(one, name) for one in held.values)
    return False


def _closed_here(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether the function closes a connection.

    `connect.close(...)`, `conn.close()`, or a `with closing(...)`. NOT any
    with-block: the first draft took every `with <call>:` as proof, which
    meant one `with resultset.snapshot(conn):` excused everything the
    function opened -- and `facts()` in sg_web/collection_view.py has
    exactly that shape.
    """
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        named = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if named in ("close", "closing"):
            return True
    return False


def rule_connection_lifetime(
    sources: typing.Iterable[Source] | None = None, kept: frozenset[str] | None = None
) -> list[Finding]:
    """SG103: a connection is acquired and neither closed here nor given away.

    The invariant, exactly: a function that opens a canonical connection
    closes it locally or returns it. That is narrower than "every
    connection is closed" and the difference matters. Once one is handed
    to a caller, the caller's own body is what this rule reads; one stored
    in an object leaves its sight entirely. What it proves is that nobody
    drops a connection on the floor of the function that made it.

    The floor is where the damage was. An unclosed Connection sits in a
    reference cycle until a later collection sweeps it, and CPython's
    sqlite3 raises ResourceWarning from the finalizer -- an unraisable,
    attributed to whatever happened to be running at that moment. Under
    `filterwarnings = error` that fails an unrelated test at an unrelated
    time, which reads as a flake and gets carried as one.

    db/connect.py is outside this rule by construction: it is the one file
    where raw sqlite3.connect is allowed, so the handle between
    sqlite3.connect and _prepared is never bound to a name this rule can
    follow. That path is held by connect._prepare_or_close and by
    test_a_connection_that_cannot_be_prepared_closes_its_handle.
    """
    excused = policy.CONNECTION_KEPT if kept is None else kept
    held = [on_disk(one) for one in every_source()] if sources is None else list(sources)
    found: list[Finding] = []
    #: Which files an excusal names. A module with no `connect` in it is skipped
    #: below, and that skip must not take the stale-excusal report with it: an
    #: entry naming a function that stopped opening anything is what it finds.
    excused_in = {one.rsplit(":", 1)[0] for one in excused}
    for source in held:
        # Nothing here binds the name `connect`, so nothing here can open one:
        # `_opened_here` matches `connect.<attr>(...)`, an ast.Name node. Read
        # off the module's own walk, which `walked` has already built.
        if source.relative not in excused_in and not any(
            isinstance(node, ast.Name) and node.id == "connect" for node in walked(source.tree)
        ):
            continue
        for fn in functions(source.tree):
            dropped = [] if _closed_here(fn) else [c for n, c in _opened_here(fn) if not _handed_onward(fn, n)]
            if f"{source.relative}:{fn.name}" in excused:
                if not dropped:
                    found.append(
                        Finding(
                            source.path,
                            fn.lineno,
                            fn.col_offset,
                            "SG103",
                            f"{fn.name} keeps no connection now; remove its CONNECTION_KEPT line",
                        )
                    )
                continue
            found.extend(
                Finding(
                    source.path,
                    call.lineno,
                    call.col_offset,
                    "SG103",
                    f"{fn.name} opens a database and neither closes it nor hands it on; close it in"
                    " try/finally, return it, or name it in sglint/policy.py CONNECTION_KEPT with"
                    " the reason it outlives the call",
                )
                for call in dropped
            )
    return found


# --- SG4xx: the web adapters own no semantics ------------------------------------------------


def _called_attrs(tree: ast.AST) -> dict[str, tuple[int, int]]:
    found: dict[str, tuple[int, int]] = {}
    for node in calls(tree):
        if isinstance(node.func, ast.Attribute):
            found.setdefault(node.func.attr, (node.lineno, node.col_offset))
    return found


def _called_qualified(tree: ast.AST) -> dict[str, tuple[int, int]]:
    found: dict[str, tuple[int, int]] = {}
    for node in calls(tree):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            found.setdefault(f"{node.func.value.id}.{node.func.attr}", (node.lineno, node.col_offset))
    return found


def _db_vocabulary(tree: ast.AST) -> dict[str, tuple[int, int]]:
    found: dict[str, tuple[int, int]] = {}
    for node in walked(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "db":
            for alias in node.names:
                found.setdefault(alias.name, (node.lineno, node.col_offset))
    return found


def _before_marker(
    root: pathlib.Path, pins: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] | None = None
) -> list[Finding]:
    """SG407: a word before a marker that the module's first half may not
    carry -- the narrator above its persistence section, the planner above
    its orchestration.

    A named function is cut out first. The planner's `engine_for` resolves
    WHICH provider is configured and is the one function before the marker
    allowed a connection; cutting it by name keeps the rule over the rest
    exact, where widening the word list would excuse the same word
    everywhere in the module.
    """
    declared = policy.MUST_NOT_CONTAIN_BEFORE if pins is None else pins
    found: list[Finding] = []
    for relative, (marker, words, excused) in declared.items():
        held = (root / relative).read_text(encoding="utf-8")
        head = held.split(marker, 1)[0]
        for name in excused:
            opened = head.find(f"def {name}(")
            if opened == -1:
                found.append(Finding(root / relative, 1, 0, "SG407", f"nothing named {name} to excuse"))
                continue
            closed = head.find("\n\n\ndef ", opened)
            head = head[:opened] + (head[closed:] if closed != -1 else "")
        for word in words:
            if word in head:
                line = head[: head.index(word)].count("\n") + 1
                found.append(Finding(root / relative, line, 0, "SG407", f"carries {word!r} before {marker!r}"))
    return found


def rule_adapters(
    root: pathlib.Path = REPO_ROOT,
    must_not_contain_before: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] | None = None,
) -> list[Finding]:
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
            for node in walked(parsed(path))
            if isinstance(node, ast.ImportFrom) and node.module == module
            for alias in node.names
        }
        if name not in imported:
            found.append(Finding(path, 1, 0, "SG403", f"stopped importing {name} from {module}"))
    bodies: dict[str, str] = {}
    for relative in policy.ONE_TO_MANY_MODULES:
        tree = parsed(root / relative)
        bodies |= {node.name: ast.unparse(node) for node in walked(tree) if isinstance(node, ast.FunctionDef)}
    for one, many in policy.ONE_DELEGATES_TO_MANY:
        where = root / policy.ONE_TO_MANY_MODULES[0]
        if one not in bodies or many not in bodies[one]:
            found.append(Finding(where, 1, 0, "SG404", f"{one} stopped delegating to {many}"))
        if many in bodies and "executemany" not in bodies[many]:
            found.append(Finding(where, 1, 0, "SG404", f"{many} no longer writes with executemany"))
    for relative in policy.LITERAL_STATEMENTS_ONLY:
        path = root / relative
        for node in calls(parsed(path)):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "execute":
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
        held = _code_only(root / relative, (root / relative).read_text(encoding="utf-8"))
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
    found.extend(_before_marker(root, must_not_contain_before))
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
            # "Class.method" names one method; a bare name is a function
            # of the module, which is how a judge or a planner is spelled
            owner, _, method = dotted.rpartition(".")
            scopes: list[list[ast.stmt]] = (
                [tree.body]
                if not owner
                else [node.body for node in walked(tree) if isinstance(node, ast.ClassDef) and node.name == owner]
            )
            wanted = [
                fn
                for scope in scopes
                for fn in scope
                if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef) and fn.name == method
            ]
            if not wanted:
                found.append(Finding(root / relative, 1, 0, "SG408", f"{dotted} is not there to check"))
            found.extend(
                Finding(root / relative, fn.lineno, fn.col_offset, "SG408", f"{dotted} takes {param!r}")
                for fn in wanted
                if any(a.arg == param for a in fn.args.args + fn.args.kwonlyargs)
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
    # OpenAPI document. It is not authored, and it names every route served --
    # including /search -- so sweeping it would fail on the contract itself.
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
    found.extend(_page_shapes(templates))
    found.extend(_one_keyboard(scripts))
    return found


#: The one module allowed to listen to the document for keystrokes.
KEY_ROUTER = "keys.ts"
#: Claiming keys for a whole surface, however it is spelled: the direct
#: listener, or one through a module's own helper. An element-scoped listener is
#: deliberately not matched -- a key inside one widget is that widget's business.
_DOCUMENT_KEYDOWN = re.compile(r'(?:(?:document|window)\.addEventListener|onDocument)\(\s*"keydown"')


def _one_keyboard(scripts: typing.Iterable[pathlib.Path]) -> list[Finding]:
    """SG503: a browser module listens to the document for keystrokes.

    Two modules cannot agree about a key by being careful. The viewer and
    the authored strip ship in the same bundle on the same surfaces, and
    each had grown its own document listener: `F` was focus AND favorite,
    `1` was actual-pixels AND one star, `0` was fit AND clear-rating, and
    every one of them fired both -- somebody looking closely at a
    photograph was silently rating it.

    So the dispatch lives in ONE place (frontend/src/keys.ts) and modules
    register what they answer to, where a second claim on a live key is
    refused by name. That only holds while nothing else listens, which is
    what this rule is. An element-scoped listener is untouched: a key
    pressed inside one widget is that widget's business.
    """
    found: list[Finding] = []
    for source in scripts:
        if source.name == KEY_ROUTER:
            continue
        held = source.read_text(encoding="utf-8")
        claim = _DOCUMENT_KEYDOWN.search(held)
        if claim is None:
            continue
        line = held[: claim.start()].count("\n") + 1
        found.append(
            Finding(
                source,
                line,
                0,
                "SG503",
                f"listens to the document for keystrokes; register them with {KEY_ROUTER} instead",
            )
        )
    return found


def _page_shapes(templates: typing.Iterable[pathlib.Path]) -> list[Finding]:
    """SG502: a page that is not a child of the shell, or a fragment that
    is a whole document.

    A full page extends base.html and owns no document of its own, so the
    navigation, the notice and the activity surface cannot be missing from
    one page and present on the rest. A fragment is mounted into a page
    that already has a document; one carrying `<html>` or a doctype is a
    page, not a fragment.
    """
    found: list[Finding] = []
    for source in templates:
        held = source.read_text(encoding="utf-8")
        lowered = held.lower()
        if source.name.startswith("_"):
            for word in ("<html", "<!doctype"):
                if word in lowered:
                    line = lowered[: lowered.index(word)].count("\n") + 1
                    found.append(Finding(source, line, 0, "SG502", f"a fragment carrying {word} is a page"))
            continue
        if source.name == policy.SHELL_TEMPLATE or source.name in policy.OWN_DOCUMENT:
            continue
        # Past any leading Jinja comment: the rule is that a page EXTENDS the
        # shell, not that its first bytes are the tag. A page opening with what
        # it is and why would otherwise read as a page that extends nothing.
        opens = held.lstrip()
        while opens.startswith("{#"):
            shut = opens.find("#}")
            if shut == -1:
                break
            opens = opens[shut + 2 :].lstrip()
        if not opens.startswith(policy.EXTENDS_SHELL):
            found.append(Finding(source, 1, 0, "SG502", f"a page that does not open with {policy.EXTENDS_SHELL}"))
        if "<!doctype" in lowered:
            line = lowered[: lowered.index("<!doctype")].count("\n") + 1
            found.append(Finding(source, line, 0, "SG502", "a page carrying its own document; the shell owns it"))

    # A recorded decision about a page that is gone names something that does
    # not exist, which is worse than no note at all.

    # Against THIS REPOSITORY's templates, not the set the caller passed:
    # `_page_shapes` is handed arbitrary template lists, sglint's own tests
    # among them, and a rule that fires on its own fixtures says nothing.
    here = REPO_ROOT / "sg_web" / "templates"
    if here.is_dir():
        names = {source.name for source in here.glob("*.html")}
        found.extend(
            Finding(
                REPO_ROOT / "sglint" / "policy.py",
                1,
                0,
                "SG502",
                f"{held} is recorded as owning its document but is not a template",
            )
            for held in policy.OWN_DOCUMENT
            if held not in names
        )
    return found


# --- SG6xx: every derived table has a producer something calls ------------------------------


def derived_tables(schema_sql: str) -> set[str]:
    return set(re.findall(r"^CREATE TABLE (derived_\w+) \(", schema_sql, re.MULTILINE))


def unwired(tables: set[str], root: pathlib.Path = REPO_ROOT) -> dict[str, str]:
    """{table: why} for derived tables whose INSERT nobody outside
    db/derived.py can reach."""
    derived_source = (root / "db" / "derived.py").read_text(encoding="utf-8")
    module = ast.parse(derived_source)
    functions = [node for node in walked(module) if isinstance(node, ast.FunctionDef)]
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

#: Every decorator that makes a function a route.
_ROUTES = _BODY_ROUTES | {"get"}


def _wire_contracts(sources: typing.Iterable[Source]) -> set[str]:
    """Every class in sg_web that inherits Wire, by name.

    Closed over inheritance rather than read one base deep: a contract that
    narrows another (JobSnapshot over JobListed) obeys the same policy, and
    a rule that could not see that would report the honest half of the tree
    as the broken half.
    """
    bases: dict[str, set[str]] = {}
    #: `X = A | B | C` at module level, by the names it joins.
    aliases: dict[str, set[str]] = {}
    for source in sources:
        for node in walked(source.tree):
            if isinstance(node, ast.ClassDef):
                held = {base.id for base in node.bases if isinstance(base, ast.Name)}
                # `class X(RootModel[Annotated[A | B, ...]])` is how a discriminated
                # body is spelled: litestar takes a body only when the annotation is
                # a model CLASS, so the union travels inside one.
                for base in node.bases:
                    if isinstance(base, ast.Subscript) and _annotation_name(base.value) == "RootModel":
                        held |= {one.id for one in ast.walk(base.slice) if isinstance(one, ast.Name)}
                bases[node.name] = held
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp):
                joined = _union_members(node.value)
                if joined:
                    for one in node.targets:
                        if isinstance(one, ast.Name):
                            aliases[one.id] = joined
    named = {"Wire"}
    growing = True
    while growing:
        found = {name for name, held in bases.items() if held & named}
        # An alias every member of which is a contract is one too: it is
        # how a discriminated document is spelled, and a rule that could
        # not read it would send every route serving one to the ledger.
        found |= {name for name, held in aliases.items() if held and held <= (named | found)}
        growing = not found <= named
        named |= found
    return named - {"Wire"}


def _union_members(node: ast.expr) -> set[str]:
    """The names a `A | B | C` expression joins, or nothing if any arm is
    not a bare name."""
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left, right = _union_members(node.left), _union_members(node.right)
        return left | right if left and right else set()
    return set()


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


def rule_request_contracts(
    sources: typing.Iterable[Source] | None = None, reserved: frozenset[str] | None = None
) -> list[Finding]:
    """SG412: a route's JSON body is not a Wire contract.

    sg_web/wire.py states one policy for every JSON shape crossing the
    seam -- name every field, refuse the rest, translate rather than
    coerce. A body annotated `dict` obeys none of it and a dataclass obeys
    only some, and either way the OpenAPI document describes nothing, so
    the browser's generated types cannot describe the request either.

    Forms are exempt by shape, not by name: URLEncodedBody carries a form,
    which is a different contract with different rules.
    """
    held = list(web_sources() if sources is None else sources)
    contracts = _wire_contracts(held)
    excused = policy.REQUEST_CONTRACT_RESERVED if reserved is None else reserved
    found: list[Finding] = []
    for source in held:
        for node in functions(source.tree):
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
                vague = named not in contracts
                if f"{source.relative}:{node.name}" in excused:
                    if not vague:
                        found.append(
                            Finding(
                                source.path,
                                argument.lineno,
                                argument.col_offset,
                                "SG412",
                                f"{node.name} names its body now; remove its REQUEST_CONTRACT_RESERVED line",
                            )
                        )
                    continue
                if vague:
                    found.append(
                        Finding(
                            source.path,
                            argument.lineno,
                            argument.col_offset,
                            "SG412",
                            f"{node.name} takes a JSON body typed {named or 'nothing'}, which is not a Wire contract",
                        )
                    )
    return found


#: Return types that carry no JSON: a rendered page, a redirect, a byte
#: stream. A handler that only ever answers with one of these has no wire
#: contract to state.
_NOT_JSON = frozenset({"Template", "Redirect", "Stream", "File", "ASGIResponse"})

#: JSON values that are genuinely a primitive and need no model.
_PRIMITIVE = frozenset({"str", "int", "float", "bool", "bytes", "None"})


def _carries_json(node: ast.expr) -> bool:
    """Whether one alternative of a return annotation carries JSON at all.

    Bytes are not: `Response[bytes]` is a picture or a download, and the
    browser addresses it with a URL rather than a generated type.
    """
    named = _annotation_name(node)
    if named in _NOT_JSON:
        return False
    if isinstance(node, ast.Subscript) and named == "Response":
        return _annotation_name(node.slice) != "bytes"
    return named != "bytes"


def _json_parts(node: ast.expr | None) -> list[ast.expr]:
    """The alternatives of a return annotation that carry JSON."""
    if node is None:
        return []
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _json_parts(node.left) + _json_parts(node.right)
    return [node] if _carries_json(node) else []


def _muddled(node: ast.expr | None) -> bool:
    """Whether a union puts a JSON answer beside one that is not JSON.

    Litestar builds the response schema from the whole annotation, and a
    union it cannot render as one media type collapses to the empty
    schema: measured on v2.24.0, `Template | Response[Held]` AND
    `Response[Held] | Redirect` both reach the document as
    `application/json: {schema: {}}`, while `Response[Held]` alone reaches
    it as a $ref. So a route that negotiates cannot state its contract in
    the return type however precisely it writes it -- it has to say it in
    `responses=`, which is where OpenAPI reads it.
    """
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.BitOr):
        return False
    arms: list[ast.expr] = []

    def walk(one: ast.expr) -> None:
        if isinstance(one, ast.BinOp) and isinstance(one.op, ast.BitOr):
            walk(one.left)
            walk(one.right)
        else:
            arms.append(one)

    walk(node)
    return any(_annotation_name(one) in _NOT_JSON for one in arms) and any(_carries_json(one) for one in arms)


def _precise(node: ast.expr, contracts: set[str]) -> bool:
    """Whether this alternative names a shape the browser can be given.

    A Wire contract, a list of one, a primitive -- or `Response[X]` where X
    is any of those. A bare `Response`, a `dict`, a `list[dict]` and an
    `Any` all describe nothing, and OpenAPI writes down exactly that.
    """
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    if isinstance(node, ast.Name):
        return node.id in contracts or node.id in _PRIMITIVE
    if isinstance(node, ast.Attribute):
        # `collection_view.CollectionWriteAnswer`: a contract another
        # module owns is the same contract.
        return node.attr in contracts
    if isinstance(node, ast.Subscript):
        held = _annotation_name(node.value)
        if held in ("Response", "list", "Sequence"):
            return _precise(node.slice, contracts)
        return held in contracts
    return False


def _declared_containers(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.expr] | None:
    """The data_container of every ResponseSpec in the route's `responses=`,
    or None when the route declares none."""
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        for word in decorator.keywords:
            if word.arg != "responses" or not isinstance(word.value, ast.Dict):
                continue
            held: list[ast.expr] = []
            for spec in word.value.values:
                if not isinstance(spec, ast.Call):
                    return []
                held.extend(k.value for k in spec.keywords if k.arg == "data_container")
            return held
    return None


def rule_response_contracts(
    sources: typing.Iterable[Source] | None = None, reserved: frozenset[str] | None = None
) -> list[Finding]:
    """SG413: a route answers JSON the contract does not describe.

    `dict`, `list[dict]` and a bare `Response` all reach OpenAPI as "an
    object", so the browser's generated types say nothing and every reader
    of that JSON is back to guessing which keys are there. A route that
    negotiates -- a page to a person, JSON to a machine -- cannot say it in
    the return type, and says it in `responses=` instead; that is the same
    contract, written where OpenAPI reads it -- and it has to, because a
    union that mixes a page with a JSON answer reaches the document as the
    empty schema however precisely each arm is written (_muddled).
    """
    held = list(web_sources() if sources is None else sources)
    contracts = _wire_contracts(held)
    excused = policy.RESPONSE_CONTRACT_RESERVED if reserved is None else reserved
    found: list[Finding] = []
    for source in held:
        for node in functions(source.tree):
            if not any(
                isinstance(one, ast.Call) and isinstance(one.func, ast.Name) and one.func.id in _ROUTES
                for one in node.decorator_list
            ):
                continue
            declared = _declared_containers(node)
            carried = _json_parts(node.returns) if declared is None else declared
            # a negotiating union states nothing whatever its arms say, so
            # every JSON arm of one is vague until `responses=` names it
            muddled = declared is None and _muddled(node.returns)
            vague = carried if muddled else [one for one in carried if not _precise(one, contracts)]
            if f"{source.relative}:{node.name}" in excused:
                if not vague:
                    found.append(
                        Finding(
                            source.path,
                            node.lineno,
                            node.col_offset,
                            "SG413",
                            f"{node.name} names its answer now; remove its RESPONSE_CONTRACT_RESERVED line",
                        )
                    )
                continue
            if not vague:
                continue
            spelled = ", ".join(sorted({ast.unparse(one) for one in vague})) or "nothing"
            if muddled:
                said = (
                    f"{node.name} returns {spelled} beside a page or a redirect, and OpenAPI writes the whole"
                    " union down as the empty schema; declare the JSON answer in responses="
                )
            else:
                where = "responses=" if declared is not None else "returns"
                said = f"{node.name} {where} {spelled}, which describes no shape the browser can be typed against"
            found.append(Finding(source.path, node.lineno, node.col_offset, "SG413", said))
    return found


def rule_templates_parse(templates: pathlib.Path = REPO_ROOT / "sg_web" / "templates") -> list[Finding]:
    """SG008: a template Jinja cannot parse.

    `just check` runs ruff, ty, pyrefly, biome and tsc, and not one of
    them reads a `.html`. A template with an unclosed `{#` comment, or a
    `{# #}` inside a `{{ }}`, passes the whole gate and then 500s on
    every request to the page that includes it. Both of those shipped
    green.

    Jinja's own parser, not a pattern: nesting and quoting are what a
    regex gets wrong here, and the parser is what the application will
    use at render.

    Parsing only. It cannot know whether a name the template reads was
    supplied -- that needs a render, and the browser tests do it.
    """
    from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError

    if not templates.is_dir():
        return []
    # The application's loader, so `{% extends %}` and `{% include %}`
    # resolve the way they will at render (sg_web/app.py _template_engine).
    env = Environment(loader=FileSystemLoader(str(templates)), autoescape=True)
    found: list[Finding] = []
    for source in sorted(templates.rglob("*.html")):
        try:
            env.parse(source.read_text(encoding="utf-8"), name=source.name, filename=str(source))
        except TemplateSyntaxError as broken:
            found.append(Finding(source, broken.lineno or 1, 0, "SG008", f"Jinja cannot parse this: {broken.message}"))
    return found


#: One concern, one module. The file that owns it, and what nothing else
#: may reach for directly.
SOLE_OWNERS = (
    ("localStorage", "workspace.ts", "the workspace (frontend/src/workspace.ts `remember`)"),
    ('document.addEventListener("key', "keys.ts", "the key registry (frontend/src/keys.ts `register`)"),
)


def rule_one_owner(source: pathlib.Path = REPO_ROOT / "frontend" / "src") -> list[Finding]:
    """SG009: a browser concern reached for outside the module that owns it.

    workspace.ts exists because "every surface that wanted to remember
    something had started inventing its own key in localStorage" -- its
    own words -- and two surfaces were still doing it: the install prompt
    kept `sg-install-dismissed` and the timeline kept `timeline.row`.
    Three keys under a module that says there is one.

    keys.ts says "there is one listener here" and means it, so a second
    `document.addEventListener("keydown")` anywhere else is a second
    keyboard nothing can reconcile -- which is the bug that module was
    written to end.

    Ownership that is only a convention is ownership until somebody is in
    a hurry. The owner is exempt; nothing else is. A module with a real
    reason may say so at the call and take the exemption by name.
    """
    found: list[Finding] = []
    if not source.is_dir():
        return []
    for path in sorted(source.rglob("*.ts")):
        if path.name.endswith(".test.ts"):
            continue
        for token, owner, instead in SOLE_OWNERS:
            if path.name == owner:
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if token in line and not line.lstrip().startswith(("//", "*", "/*")):
                    found.append(
                        Finding(
                            path, n, line.index(token), "SG009", f"{token} belongs to {owner}; go through {instead}"
                        )
                    )
    return found


# --- SG010: every capability has a way in ---------------------------------------------------------


def _reachable_text(root: pathlib.Path) -> str:
    """Everything a person could be led by: the templates and the authored
    browser source. Generated types are not a way in."""
    parts: list[str] = []
    for where, pattern in ((root / "sg_web" / "templates", "*.html"), (root / "frontend" / "src", "*.ts")):
        if not where.is_dir():
            continue
        parts.extend(p.read_text(encoding="utf-8") for p in where.rglob(pattern) if "generated" not in str(p))
    return "\n".join(parts)


def _queued_by(root: pathlib.Path) -> dict[str, set[str]]:
    """Which job kind each `submit_*` puts on the queue.

    Read from the source rather than declared, because the console's
    vocabulary and the worker's differ ON PURPOSE: the console offers
    "faces" and "thumbs", the worker runs `detect_faces` and `hash`.
    Comparing those two name sets directly would assert a correspondence
    this application deliberately does not have.

    Both modules that submit are read. `db/prompts.py` has its own
    `submit_embed`, and reading only `db/runner.py` makes `embed_prompts`
    look unstartable when its button is right there beside the others.
    """
    found: dict[str, set[str]] = {}
    for name in ("runner", "prompts"):
        source = root / "db" / f"{name}.py"
        if not source.is_file():
            continue
        tree = parsed(source)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("submit_"):
                continue
            body = ast.unparse(node)
            kinds = set(re.findall(r"""jobs\.submit\(\s*conn,\s*['"]([a-z_]+)['"]""", body))
            kinds |= set(re.findall(r"""kind\s*=\s*['"]([a-z_]+)['"]""", body))
            found[f"{name}.{node.name}"] = kinds
    return found


def rule_job_kind_has_a_way_in(root: pathlib.Path = REPO_ROOT) -> list[Finding]:
    """SG012: a job the worker can run that nobody can start.

    A job kind is a capability in the fullest sense -- a handler, a
    queue, a ledger and tests -- and a kind with no way to start one is
    as unshipped as a button nobody drew.

    Three things, because a register is only worth what its weakest half
    is worth: every kind is startable or recorded; every recorded kind
    still exists; and the thing each record POINTS AT still exists. The
    third is what stops the register decaying into prose -- "the session
    card starts this" stays readable and reassuring long after the
    session card stops doing it.

    This lived in a test that read `inspect.getsource`, which SG007
    rightly refuses: a check over the source belongs here, where it runs
    on every commit rather than only when somebody runs the slow suite.
    """
    runner = root / "db" / "runner.py"
    console = root / "sg_web" / "operations.py"
    if not runner.is_file() or not console.is_file():
        return []
    at = (console, 1, 0)
    found: list[Finding] = []

    handlers = set(re.findall(r"""^\s+['"]([a-z_]+)['"]:""", _handlers_block(runner), re.MULTILINE))
    if not handlers:
        # POSITIVE CONTROL. "No handlers found" is a fact about this
        # rule's reading, never about the application, and the two must
        # never be confused.
        return [Finding(runner, 1, 0, "SG012", "HANDLERS could not be read, so no gap found here means nothing")]

    queued = _queued_by(root)
    if not any(queued.values()):
        return [
            Finding(runner, 1, 0, "SG012", "no submit_* appears to queue a kind; this rule is misreading the source")
        ]

    text = console.read_text(encoding="utf-8")
    reachable: set[str] = set()
    for module, called in re.findall(r"\b(runner|prompts)\.(submit_\w+|catch_up)\b", text):
        if called == "catch_up":
            # The ordered run of all of them: it reaches whatever every
            # submit reaches.
            reachable |= {kind for kinds in queued.values() for kind in kinds}
        else:
            reachable |= queued.get(f"{module}.{called}", set())

    found.extend(
        Finding(
            runner, 1, 0, "SG012", f"`{kind}` is a job the worker runs and nothing starts; add a launcher or record it"
        )
        for kind in sorted(handlers - reachable - set(policy.STARTED_ELSEWHERE))
    )
    found.extend(
        Finding(runner, 1, 0, "SG012", f"`{kind}` is recorded as started elsewhere but is not a job kind any more")
        for kind in sorted(set(policy.STARTED_ELSEWHERE) - handlers)
    )

    # What each record points at, still there.
    markup = _reachable_text(root)
    if "story_plan" in policy.STARTED_ELSEWHERE and "/stories/sessions/" not in markup:
        found.append(
            Finding(
                *at,
                "SG012",
                "story_plan is recorded as started by opening a sitting, but nothing builds /stories/sessions/",
            )
        )
    if "walk" in policy.STARTED_ELSEWHERE and '"catch_up"' not in text and "'catch_up'" not in text:
        found.append(
            Finding(*at, "SG012", "walk is recorded as catch_up's first step, but there is no catch_up launcher")
        )
    return found


def _handlers_block(runner: pathlib.Path) -> str:
    """The text of `db/runner.py HANDLERS`, and nothing else. Read as
    text rather than imported: importing the worker to ask what it can do
    loads torch."""
    text = runner.read_text(encoding="utf-8")
    at = text.find("HANDLERS")
    if at == -1:
        return ""
    shut = text.find("\n}", at)
    return text[at:shut] if shut != -1 else ""


def rule_capability_has_a_way_in(root: pathlib.Path = REPO_ROOT) -> list[Finding]:
    """SG010: something the application can do that no surface reaches.

    The source of truth is the addresses served, not anybody's memory: an
    address whose static prefix appears in no template and no authored
    module is not reachable.

    The prefix rather than the whole path, because a parameterised address
    is built and never written out: `/f/{slug}` is reached by `/f/`.

    EVERY VERB, not only GET. This read `@get` alone, which left every
    POST outside the rule -- and twelve of them were unreached: the sweep
    endpoints in app.py, whose work a person starts from the console at
    /operations/jobs/{kind} instead. A capability you can only start by
    writing your own request is the same defect as one you can only start
    by pressing an undocumented letter.

    The router prefix, too. A decorator path is not the address:
    operations.py ends in `Router(path="/operations", ...)`, so its
    `@post("/jobs/{kind}")` answers at /operations/jobs/{kind}. Reading
    the decorator alone files it under the wrong name.

    Capabilities that are not addresses cannot be found by reading routes.
    Those are recorded by hand in policy.UNSURFACED_BEYOND_ROUTES.
    """
    web = root / "sg_web"
    if not web.is_dir():
        return []
    found: list[Finding] = []
    reachable = _reachable_text(root)

    served: list[tuple[str, pathlib.Path, int]] = []
    verb = re.compile(r"@(?:get|post|put|patch|delete)\(\"([^\"]+)\"")
    mounted = re.compile(r"Router\(\s*path=\"([^\"]+)\"")
    for source in sorted(web.glob("*.py")):
        text = source.read_text(encoding="utf-8")
        at = mounted.search(text)
        prefix = at.group(1).rstrip("/") if at else ""
        for n, line in enumerate(text.splitlines(), 1):
            hit = verb.search(line)
            if hit:
                served.append((f"{prefix}{hit.group(1)}", source, n))

    for path, source, line in served:
        cut = path.find("{")
        prefix = (path[:cut] if cut != -1 else path).rstrip("/")
        # Matched on the prefix, because a recorded address is written the
        # way a person says it and the route carries its parameter types.
        if path == "/" or path in policy.UNSURFACED or prefix in policy.UNSURFACED:
            continue
        if prefix and prefix not in reachable:
            found.append(
                Finding(source, line, 0, "SG010", f"{path} is served and nothing reaches it; surface it or record it")
            )

    # A stale exemption says a decision was taken about something that no
    # longer exists, which is worse than no note at all.
    addresses = {path for path, _, _ in served}
    found.extend(
        Finding(web / "app.py", 1, 0, "SG010", f"{held} is recorded as unsurfaced but is not served")
        for held in policy.UNSURFACED
        if not any(path.startswith(held) for path in addresses)
    )

    return found


def rule_comment_blocks(root: pathlib.Path = REPO_ROOT) -> list[Finding]:
    """SG013: a comment block outside a docstring runs past three lines.

    CONTRIBUTING.md holds a non-docstring comment block to two sentences or
    three physical lines. Vale enforces the sentence half and cannot enforce
    this one: it joins a block into one string with the newlines stripped, and
    `occurrence` puts word boundaries around its token, so a whitespace or
    newline token matches nothing there while a word token matches. Counting
    physical lines needs the file, which is why this lives here.

    Only standalone `#` comments count. A trailing comment is one line by
    construction, and a docstring is exempt by the standard.
    """
    found: list[Finding] = []
    for source in every_source():
        if not source.is_relative_to(root):
            continue
        try:
            with tokenize.open(source) as handle:
                comments = [
                    tok
                    for tok in tokenize.generate_tokens(handle.readline)
                    if tok.type == tokenize.COMMENT and not tok.line[: tok.start[1]].strip()
                ]
        except (OSError, SyntaxError, tokenize.TokenError, UnicodeDecodeError):
            continue
        run: list[tokenize.TokenInfo] = []
        for tok in [*comments, None]:
            if run and (tok is None or tok.start[0] != run[-1].start[0] + 1):
                if len(run) > 3:
                    found.append(
                        Finding(
                            source,
                            run[0].start[0],
                            run[0].start[1],
                            "SG013",
                            f"comment block runs to {len(run)} lines; the limit is three",
                        )
                    )
                run = []
            if tok is not None:
                run.append(tok)
    found += _just_comment_blocks(root)
    return found


def every_just(root: pathlib.Path = REPO_ROOT) -> list[pathlib.Path]:
    """Every justfile this repository owns."""
    found = [*root.rglob("*.just"), *root.rglob("justfile")]
    return sorted(
        one
        for one in found
        if one.is_file() and not {"vendor", "node_modules", ".git"} & set(one.relative_to(root).parts)
    )


def _just_comment_blocks(root: pathlib.Path) -> list[Finding]:
    """SG013 over justfiles, which tokenize cannot read.

    The standard is about COMMENTS, not about Python: CONTRIBUTING.md says a
    comment block outside a docstring holds to two sentences or three physical
    lines, and names no language. `every_source` walks `.py` only, so
    `compat.just` -- the file that runs the whole compatibility suite -- was
    the one place in the repository where a comment could say anything at any
    length, and it accumulated twenty blocks over the limit.
    """
    found: list[Finding] = []
    for source in every_just(root):
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        run: list[int] = []
        for number, line in enumerate([*lines, ""], start=1):
            # A standalone comment only. A trailing `#` is one line by
            # construction, and a shebang is not a comment.
            standalone = line.lstrip().startswith("#") and not line.lstrip().startswith("#!")
            if run and not (standalone and number == run[-1] + 1):
                if len(run) > 3:
                    found.append(
                        Finding(
                            source, run[0], 0, "SG013", f"comment block runs to {len(run)} lines; the limit is three"
                        )
                    )
                run = []
            if standalone:
                run.append(number)
    return found


#: The one module allowed to reach `subprocess`. Everything else goes through
#: it, so a timeout kills the whole process tree and no stream is ever a pipe.
THE_RUNNER = "proc.py"

#: Names that spawn a process without going through the runner.
_DIRECT_SPAWN = frozenset({"run", "Popen", "call", "check_call", "check_output", "getoutput", "getstatusoutput"})
_SHELL_OUT = frozenset({"system", "popen"})


def rule_one_runner(root: pathlib.Path = REPO_ROOT) -> list[Finding]:
    """SG005: a module reaches for `subprocess` instead of `proc`.

    SG003 and SG004 catch a spawn that is unbounded or piped, one call site at
    a time. This makes the boundary structural: `proc.py` owns every spawn, so
    the timeout semantics are decided once and cannot be re-derived wrongly at
    the next call site.

    `vendor/` is excluded because this repository does not own that code.

    The SPAWNERS are named, not the module: `subprocess.CompletedProcess` is a
    return type and `subprocess.PIPE` a constant, and a module that only reads
    those is not spawning anything. Reaching a spawner means naming it, as an
    attribute or an import, and either spelling is caught -- including a bound
    alias, because `held = subprocess.run` is that attribute.
    """
    found: list[Finding] = []
    runner = (root / THE_RUNNER).resolve()
    for source in every_source():
        if not source.is_relative_to(root) or source.resolve() == runner:
            continue
        if "vendor" in source.relative_to(root).parts:
            continue
        try:
            tree = parsed(source)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            # `ast.walk` is typed as yielding bare AST, which carries no
            # position. Every node matched below is an ImportFrom or an
            # Attribute, and both do.
            if not isinstance(node, ast.ImportFrom | ast.Attribute):
                continue
            named = ""
            if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "subprocess":
                named = next((f"subprocess.{one.name}" for one in node.names if one.name in _DIRECT_SPAWN), "")
            elif isinstance(node, ast.ImportFrom) and (node.module or "") == "os":
                named = next((f"os.{one.name}" for one in node.names if one.name in _SHELL_OUT), "")
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "subprocess" and node.attr in _DIRECT_SPAWN:
                    named = f"subprocess.{node.attr}"
                elif node.value.id == "os" and node.attr in _SHELL_OUT:
                    named = f"os.{node.attr}"
            if named:
                found.append(
                    Finding(
                        source,
                        node.lineno,
                        node.col_offset,
                        "SG005",
                        f"{named} spawns outside {THE_RUNNER}; call proc.run/proc.text instead",
                    )
                )
    return found


#: A verdict for a case that did not run, in any spelling. A proof that did not
#: happen fails; it does not get a word of its own.
_DID_NOT_RUN = "UNSUPPORTED"


def rule_no_case_skips(root: pathlib.Path = REPO_ROOT) -> list[Finding]:
    """SG014: compat names a did-not-run verdict, or declares a boundary underivable.

    An absent weight, an unwritten derivation and a detector that missed are
    each a proof that did not happen, and each one fails. A runner raising
    `NotImplementedError` on a case path says the same thing in another word.

    Read from the AST, so prose describing the ban does not trip it.
    """
    found: list[Finding] = []
    compat = root / "compat"
    for source in every_source():
        if not source.is_relative_to(compat):
            continue
        try:
            tree = parsed(source)
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        consumer = source.parent.name == "consumers"
        for node in ast.walk(tree):
            # `ast.walk` yields the Module too, and it carries no position.
            if not isinstance(node, (ast.Attribute, ast.Constant, ast.Raise)):
                continue
            at = (source, node.lineno, node.col_offset)
            if isinstance(node, ast.Attribute) and node.attr == _DID_NOT_RUN:
                found.append(Finding(*at, "SG014", f"names Verdict.{_DID_NOT_RUN}; a case that did not run fails"))
            elif isinstance(node, ast.Constant) and node.value == _DID_NOT_RUN:
                found.append(Finding(*at, "SG014", f"carries the string {_DID_NOT_RUN!r}"))
            elif consumer and isinstance(node, ast.Raise) and node.exc is not None:
                called = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
                if isinstance(called, ast.Name) and called.id == "NotImplementedError":
                    found.append(Finding(*at, "SG014", "raises NotImplementedError; derive the boundary or drop it"))
    return found


# --- all of it ----------------------------------------------------------------------------------

#: `$?` inside a negated `if`. The status belongs to the negation, so it is
#: always 0 and the real code is lost.
_NEGATED_IF = re.compile(r"^\s*if\s+!\s")
_CAPTURE = re.compile(r"=\s*\$\?")

#: A recipe running a Python file by PATH. `sys.path[0]` becomes the script's
#: own directory, so a repo-root import fails.
_BY_PATH = re.compile(r"(?:python(?:\.exe)?|\{\{ *python *\}\})\s+(?!-)(\S+\.py)\b")


#: `cmd || true` discards a failure outright, in a trap as much as anywhere:
#: cleanup that fails leaks a worktree or a process and says nothing. A step
#: that may fail says so on stderr instead.
_OR_TRUE = re.compile(r"\|\|\s*(?:true|:)\s*(?:;|'|$)")


#: Turning errexit off. Everything after it in that recipe can fail silently.
_UNSET_ERREXIT = re.compile(r"^\s*set\s+\+(?:e\b|o\s+errexit\b)")

#: `export x=$(cmd)` and friends: the status is the declaring builtin's, so a
#: failing command is assigned and forgotten. Measured under `sh`:
#: `x=$(false)` fires errexit at rc=1, `export x=$(false)` exits 0.
_DECLARED_SUBSTITUTION = re.compile(r"^\s*(export|local|declare|readonly|typeset)\s+[A-Za-z_]\w*=(?:\$\(|`)")

#: `cmd && other`. Measured under `sh`: `set -eu; false && echo right` runs
#: on and exits 0, losing the left operand exactly as `|| true` does. A `[`
#: or `test` on the left is a condition, and has no status to lose.
_AND_CHAIN = re.compile(r"^\s*(?!(?:\[|!|test\b|if\b|while\b|until\b))\S.*?(?<![&|])&&(?!&)")


def rule_recipe_exit_codes(root: pathlib.Path = REPO_ROOT) -> list[Finding]:
    """SG015: a recipe throws away an exit code.

    The ways a command's failure is LOST under errexit are enumerable, and
    this covers the set rather than chasing spellings:

        cmd || true         the status is forced to 0
        cmd && other        the left operand's failure is exempt from errexit
        if ! cmd; then $?   `$?` there belongs to the negation, always 0
        a | b               a's status is dropped -- SG017, a shell option
        set +e              everything after it can fail silently
        export x=$(cmd)     the status is the builtin's, never the command's

    `if cmd`, `while cmd` and `until cmd` are NOT in the set: measured under
    `sh`, all three are exempt, and a condition's failure is the point of a
    condition. Their BODIES are checked, which is where a lost status matters.

    `if ! cmd; then code=$?; fi` captures the status of the NEGATION, which is
    always 0 when the branch is taken. A runner recording lane results that
    way writes every failure down as a pass. `cmd || code=$?` keeps the real
    code and is equally exempt from errexit.

    `cmd || true` discards it outright, and a trap is no exception: cleanup
    that fails leaks a worktree or a process and reports nothing. A step
    allowed to fail says so on stderr.
    """
    found: list[Finding] = []
    for source in every_just(root):
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        guarded = 0
        for number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _OR_TRUE.search(line):
                found.append(Finding(source, number, 0, "SG015", "discards an exit code with `|| true`"))
            if _AND_CHAIN.match(line):
                found.append(
                    Finding(source, number, 0, "SG015", "`cmd && other` loses the left operand's failure under errexit")
                )
            if _UNSET_ERREXIT.match(line):
                found.append(Finding(source, number, 0, "SG015", "turns errexit off; later steps fail silently"))
            declaring = _DECLARED_SUBSTITUTION.match(line)
            if declaring:
                found.append(
                    Finding(
                        source,
                        number,
                        0,
                        "SG015",
                        f"`{declaring.group(1)} x=$(cmd)` takes its status from the builtin; declare, then assign",
                    )
                )
            if _NEGATED_IF.match(line):
                guarded = number
            elif stripped in {"fi", "done"} or (stripped and not line.startswith((" ", "\t"))):
                guarded = 0
            elif guarded and _CAPTURE.search(line):
                found.append(
                    Finding(source, number, 0, "SG015", "reads `$?` inside a negated `if`; it is always 0 there")
                )
                guarded = 0
    return found


def rule_script_recipes_fail_loudly(root: pathlib.Path = REPO_ROOT) -> list[Finding]:
    """SG017: a `[script]` recipe runs under a shell that hides failures.

    Enumerating spellings does not close this: a step escapes `errexit`
    through `||`, through a negated `if`, and through any pipeline whose last
    command succeeds. The last one is not a spelling to ban, it is a shell
    option, and without `pipefail` `a | b` reports only b.

    just defaults a `[script]` recipe to `sh -eu` -- no pipefail, not bash --
    so a file with script recipes states its own interpreter or inherits a
    shell that reports the wrong status for every pipeline in it.
    """
    found: list[Finding] = []
    for source in every_just(root):
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "[script]" not in text:
            continue
        declared = [one for one in text.splitlines() if one.startswith("set script-interpreter")]
        if not declared:
            found.append(
                Finding(source, 1, 0, "SG017", "[script] recipes and no interpreter set; sh -eu has no pipefail")
            )
            continue
        if "pipefail" not in declared[0] or "-e" not in declared[0]:
            found.append(Finding(source, 1, 0, "SG017", f"script interpreter lacks -e or pipefail: {declared[0]}"))
    return found


def rule_recipe_module_imports(root: pathlib.Path = REPO_ROOT) -> list[Finding]:
    """SG016: a recipe runs a Python file by path, losing the repo root.

    `python pkg/mod/thing.py` puts `pkg/mod` on `sys.path` and NOT the
    repository root, so an import of a root-level module raises
    ModuleNotFoundError. `python -m pkg.mod.thing` puts the root there.

    A file that already sits at the root is exempt: for it the two are the
    same path.
    """
    found: list[Finding] = []
    for source in every_just(root):
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(lines, start=1):
            if line.strip().startswith("#"):
                continue
            for match in _BY_PATH.finditer(line):
                named = match.group(1)
                if "/" not in named and "\\" not in named:
                    continue
                found.append(
                    Finding(
                        source,
                        number,
                        0,
                        "SG016",
                        f"runs {named} by path, so the repository root is not on sys.path; use `-m`",
                    )
                )
    return found


# --- SG018: no error is swallowed into silence -------------------------------------------------


def _always_raises(body: list[ast.stmt]) -> bool:
    """Whether control can leave `body` without an exception.

    Conservative on purpose: a handler that raises in a shape this cannot
    read is REPORTED, and gets rewritten or waived by name. The opposite
    slant would be a check that passes on a body it did not understand.
    """
    if not body:
        return False
    last = body[-1]
    if isinstance(last, ast.Raise):
        return True
    if isinstance(last, ast.If):
        return _always_raises(last.body) and _always_raises(last.orelse)
    if isinstance(last, (ast.With, ast.AsyncWith)):
        return _always_raises(last.body)
    if isinstance(last, ast.Try):
        # The `finally` runs on every path, so a raise there is total.
        if _always_raises(last.finalbody):
            return True
        return _always_raises(last.body) and all(_always_raises(one.body) for one in last.handlers)
    return False


def _terminal_name(call: ast.Call) -> str:
    """What a call is CALLED, ignoring what it was reached through:
    `report`, `self._missed`, `ledger.report` all answer their last name."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return ""


def _records(node: ast.AST, vocabulary: frozenset[str]) -> bool:
    return any(_terminal_name(one) in vocabulary for one in calls(node))


def _hands_the_error_on(node: ast.ExceptHandler) -> bool:
    """The caught error carried OUT of the handler, not dropped in it.

    Two shapes no call vocabulary can see, because neither is a call.
    Written onto something that outlives the handler --
    `held.reason = f"{type(problem).__name__}: {problem}"`, eight vendor
    acceptance handlers -- or RETURNED, which hands the caller the same
    fact through a different door.

    The discriminator in both is THE BOUND NAME IN THE VALUE. A handler
    that binds the error and then writes or returns something else has
    dropped it, and is still an offence; requiring an attribute or
    subscript target rather than any assignment is what separates a record
    that leaves the handler from a local nobody reads.
    """
    if node.name is None:
        return False
    for child in walked(node):
        if isinstance(child, ast.Assign) and any(
            isinstance(one, (ast.Attribute, ast.Subscript)) for one in child.targets
        ):
            carried: ast.expr | None = child.value
        elif isinstance(child, ast.Return):
            carried = child.value
        else:
            continue
        if carried is not None and any(isinstance(one, ast.Name) and one.id == node.name for one in ast.walk(carried)):
            return True
    return False


def _suppression(node: ast.With | ast.AsyncWith) -> ast.Call | None:
    """The `contextlib.suppress(...)` a `with` is holding, if it is.

    `suppress` is an except clause wearing a different syntax, and it has
    no handler body at all -- there is nowhere for it to record and
    nothing for it to re-raise, so it can only ever be the silent form.
    A rule reading `except` alone would stop one step short of the
    mechanism and report a tree with fifteen live suppressions as clean.
    """
    for item in node.items:
        held = item.context_expr
        if isinstance(held, ast.Call) and _terminal_name(held) == "suppress":
            return held
    return None


#: How long a list of caught types may be before a key shortens it to the
#: first name and a count of the rest. The count still moves when a catch
#: widens, so a shortened key keeps discriminating.
_SPELLING_LIMIT = 40


def _shortened(names: list[str]) -> str:
    joined = ",".join(names)
    return joined if len(joined) <= _SPELLING_LIMIT or len(names) < 2 else f"{names[0]},+{len(names) - 1}"


def _spelled_type(node: ast.expr | None) -> str:
    #: A bare `except:` catches BaseException, and says so in the key.
    if node is None:
        return "BaseException"
    if isinstance(node, ast.Tuple):
        return f"({_shortened([ast.unparse(one) for one in node.elts])})"
    return ast.unparse(node)


#: What a guard can be. All three carry lineno and col_offset; bare
#: `ast.AST` does not, and a Finding needs both.
Guard = ast.ExceptHandler | ast.With | ast.AsyncWith


def _guards(tree: ast.AST) -> list[tuple[str, Guard, str]]:
    """Every place a module catches an exception: (where, node, spelling).

    `where` is the dotted path of the enclosing definitions, so two
    methods that share a name cannot share a waiver line.
    """
    found: list[tuple[str, Guard, str]] = []

    def walk(node: ast.AST, where: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = where
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = child.name if where == "<module>" else f"{where}.{child.name}"
            if isinstance(child, ast.ExceptHandler):
                found.append((where, child, _spelled_type(child.type)))
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                held = _suppression(child)
                if held is not None:
                    args = _shortened([ast.unparse(one) for one in held.args])
                    found.append((where, child, f"suppress({args})"))
            walk(child, inner)

    walk(tree, "<module>")
    return found


def rule_no_silent_except(
    sources: typing.Iterable[Source] | None = None,
    waived: dict[str, str] | None = None,
    vocabulary: frozenset[str] | None = None,
    inherited: frozenset[str] | None = None,
    ceiling: int | None = None,
) -> list[Finding]:
    """SG018: an error caught and hidden.

    The ruling, verbatim in intent: nothing may swallow an error into
    silence. Every handler does one of three things -- it PROPAGATES, it
    CONVERTS to a typed refusal the caller must handle, or it TOLERATES a
    narrow expected condition WITH THE OCCURRENCE RECORDED somewhere
    something reads. A cache may miss on a corrupt file; the miss is
    counted and surfaced, never invisible. Tolerating an error and hiding
    it are independent choices and only the first is ever permitted.

    Written as an ALLOWLIST, which is the polarity the campaign learned
    twice: a denylist of bad handler shapes admits every shape nobody
    imagined. Raising is read from the body's structure and recording
    from a declared vocabulary, so a handler doing neither is an offence
    by default rather than an unclassified pass.

    The escape is a WAIVER, declared in one list and cross-checked, never
    a comment beside the handler: an inline suppression is invisible to
    everything except the reader who is already there. Both directions
    are reported, so a list line outliving the handler it names is a
    finding and the lists can only shrink.

    Two lists, kept apart on purpose. SILENT_EXCEPT_WAIVED holds the
    handlers somebody RULED may be silent, each with its reason.
    SILENT_EXCEPT_INHERITED holds what was already there when this rule
    landed -- a debt, not a decision -- under a ceiling pinned in policy
    where the list cannot move it. Merging them would let "0 unwaived"
    stand for "162 waived in one act", which is the reporting failure S6
    was written against.

    Both lists get an exactness sweep and RECORDS_THE_MISS deliberately
    does not. Those two are claims about the tree, so a line outliving
    its handler is a stale fact; the vocabulary is a claim about what
    recording IS, and an unused level is not stale. Sweeping it would red
    the gate the moment somebody converted a handler by logging at a
    level no other handler had reached yet -- a check whose only failure
    mode is punishing the fix. It would not have caught the abuse it
    looks like it guards either: a name added to silence a finding has a
    caller by construction.
    """
    held = list(shipped_sources() if sources is None else sources)
    excused = dict(policy.SILENT_EXCEPT_WAIVED if waived is None else waived)
    owed = policy.SILENT_EXCEPT_INHERITED if inherited is None else inherited
    spoken = policy.RECORDS_THE_MISS if vocabulary is None else vocabulary
    policy_file = REPO_ROOT / "sglint" / "policy.py"

    found: list[Finding] = []
    caught: list[tuple[str, Guard, pathlib.Path]] = []
    for source in held:
        for where, node, spelling in _guards(source.tree):
            #: Only a HANDLER can answer for what it caught: a `suppress` body
            #: is the protected code, and reading it would let guarded work
            #: vouch for its own guard.
            if isinstance(node, ast.ExceptHandler) and (
                _always_raises(node.body) or _records(node, spoken) or _hands_the_error_on(node)
            ):
                continue
            caught.append((f"{source.relative}:{where}:{spelling}", node, source.path))

    #: One key, one handler: eleven groups shared a name at SG018's landing,
    #: and a line answering for three handlers excuses two nobody read. The
    #: ordinal is positional, so a renumbering vacates a key, reported below.
    seen: dict[str, int] = {}
    silent: dict[str, Finding] = {}
    for stem, node, path in caught:
        seen[stem] = seen.get(stem, 0) + 1
        key = stem if seen[stem] == 1 else f"{stem}#{seen[stem]}"
        said = (
            f"catches {stem.rsplit(':', 1)[1]} and neither raises nor records it"
            f" -- convert it, or name it in SILENT_EXCEPT_WAIVED as {key!r}"
        )
        silent[key] = Finding(path, node.lineno, node.col_offset, "SG018", said)

    for key, offence in sorted(silent.items()):
        if key in excused and key in owed:
            found.append(
                dataclasses.replace(offence, message=f"{key!r} is both waived and inherited; one list answers for it")
            )
        elif key not in excused and key not in owed:
            found.append(offence)
        elif key in excused and not excused[key].strip():
            found.append(
                dataclasses.replace(offence, message=f"is waived by {key!r} with no reason; say why or convert it")
            )

    found.extend(
        Finding(policy_file, 1, 0, "SG018", f"{key!r} waives no silent handler; remove its SILENT_EXCEPT_WAIVED line")
        for key in sorted(excused)
        if key not in silent
    )
    found.extend(
        Finding(policy_file, 1, 0, "SG018", f"{key!r} names no silent handler; remove its SILENT_EXCEPT_INHERITED line")
        for key in sorted(owed)
        if key not in silent
    )
    pinned = policy.SILENT_EXCEPT_CEILING if ceiling is None else ceiling
    if len(owed) > pinned:
        found.append(
            Finding(
                policy_file,
                1,
                0,
                "SG018",
                f"{len(owed)} inherited swallows against a ceiling of {pinned};"
                " the debt may only shrink, and raising the pin is a recorded ruling",
            )
        )
    return found


RULES = (
    rule_sources_parse,
    rule_spawns,
    rule_recipe_exit_codes,
    rule_recipe_module_imports,
    rule_script_recipes_fail_loudly,
    rule_one_runner,
    rule_no_case_skips,
    rule_templates_parse,
    rule_one_owner,
    rule_capability_has_a_way_in,
    rule_job_kind_has_a_way_in,
    rule_tests_run_things,
    rule_sql_structure,
    rule_connection_lifetime,
    rule_adapters,
    rule_surfaces,
    rule_producers,
    rule_request_contracts,
    rule_response_contracts,
    rule_comment_blocks,
    rule_no_silent_except,
)


def run() -> list[Finding]:
    from . import schema_rules

    found: list[Finding] = []
    for rule in (
        *RULES,
        schema_rules.rule_schema,
        schema_rules.rule_migrations,
        schema_rules.rule_wire_vocabularies,
        schema_rules.rule_foreign_key_indexes,
        schema_rules.rule_closed_columns,
        schema_rules.rule_index_prefixes,
        schema_rules.rule_vocabulary_handlers,
        schema_rules.rule_handlers_report,
        schema_rules.rule_written_columns,
    ):
        found.extend(rule())
    return sorted(found, key=lambda f: (str(f.path), f.line, f.col, f.code))
