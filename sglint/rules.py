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


#: A file's TEXT is deliberately not cached on (mtime, size) the way its
#: tree is, and the reads through this module stay uncached.
#:
#: The stamp is not enough to tell two versions apart. A control writes a
#: file, runs a rule, rewrites it and runs the rule again -- and
#: `test_the_vocabulary_and_handler_rules_hold_and_can_fail` does exactly
#: that with `SAYS = {'a': 1, 'b': 2}` and `SAYS = {'a': 1, 'c': 3}`,
#: which are the same LENGTH inside one clock tick. A stamped cache hands
#: back the first, the rule finds nothing wrong with source it never saw,
#: and the control that exists to prove the rule can fail passes while
#: proving nothing. Measured: it was tried, and that test caught it.
#:
#: The saving was 0.08s. `walked` below is where the real one is, and it
#: is keyed on the tree OBJECT, so it cannot go stale.
@functools.cache
def _parsed_as_of(source: pathlib.Path, _stamp: tuple[int, int]) -> ast.Module:
    return ast.parse(source.read_text(encoding="utf-8"))


def parsed(source: pathlib.Path) -> ast.Module:
    """The file's tree, parsed once per (mtime, size): a file rewritten
    under the linter re-parses itself and nothing else."""
    held = source.stat()
    return _parsed_as_of(source, (held.st_mtime_ns, held.st_size))


#: Walked trees, keyed on the tree itself.
#:
#: `ast.AST` hashes by identity, so the dict holds each tree alive for as
#: long as it holds its walk -- which is what makes this sound. Keying on
#: `id()` alone would not be: a control's tree is built from text and can
#: be collected, and the next tree along can be handed the same id.
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


#: How a test inspects production SOURCE rather than exercising it, as
#: (module, function): each is a linter wearing a pytest nametag -- it can
#: fail without running the thing whose behaviour it claims.
#:
#: Qualified, and that matters: `parse` alone matched `facets.parse` and
#: `resultset.parse` sixty-two times, which are the application's own
#: functions being CALLED, the opposite of what this looks for.
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
    #: Which files an excusal names. A module with no `connect` in it is
    #: skipped below, and that skip must not take the stale-excusal
    #: report with it: an entry naming a function that stopped opening
    #: anything is exactly what that report exists to find.
    excused_in = {one.rsplit(":", 1)[0] for one in excused}
    for source in held:
        # Nothing here binds the name `connect`, so nothing here can open
        # one: `_opened_here` matches `connect.<attr>(...)`, which is an
        # ast.Name node. Read off the module's own walk, which `walked`
        # has already built for the other rules -- against `_closed_here`
        # and `_opened_here` each walking every function in the file.
        # This rule was 0.587s of the 1.09s every rule costs together.
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
    found.extend(_page_shapes(templates))
    found.extend(_one_keyboard(scripts))
    return found


#: The one module allowed to listen to the document for keystrokes.
KEY_ROUTER = "keys.ts"
#: Claiming keys for a whole surface, however it is spelled -- the direct
#: listener, one through a module's own helper. An element-scoped listener
#: (`swap.addEventListener("keydown", ...)`) is deliberately not matched:
#: a key pressed inside one widget is that widget's business.
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
    page somebody will eventually serve whole.
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
        if source.name == policy.SHELL_TEMPLATE:
            continue
        if not held.lstrip().startswith(policy.EXTENDS_SHELL):
            found.append(Finding(source, 1, 0, "SG502", f"a page that does not open with {policy.EXTENDS_SHELL}"))
        if "<!doctype" in lowered:
            line = lowered[: lowered.index("<!doctype")].count("\n") + 1
            found.append(Finding(source, line, 0, "SG502", "a page carrying its own document; the shell owns it"))
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
                # `class X(RootModel[Annotated[A | B, ...]])` is how a
                # discriminated body is spelled: litestar takes a body only
                # when the annotation is a model CLASS, so the union travels
                # inside one. It is a contract when its arms are.
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


# --- all of it ----------------------------------------------------------------------------------

RULES = (
    rule_spawns,
    rule_tests_run_things,
    rule_sql_structure,
    rule_connection_lifetime,
    rule_adapters,
    rule_surfaces,
    rule_producers,
    rule_request_contracts,
    rule_response_contracts,
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
