"""SG7xx: the schema contract, held over schema.sql itself.

The DDL is built in memory and inspected through sqlite_master and the
PRAGMAs -- behavioural where the text would lie (a table has a rowid or
it does not; matching the phrase matched it inside a comment). Every
rule takes the DDL as an argument so the controls can bend one line and
watch the rule fire.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sqlite3
import typing

from db import connect

from . import policy
from .rules import REPO_ROOT, Finding

SCHEMA = REPO_ROOT / "db" / "schema.sql"


def built(ddl: str) -> sqlite3.Connection:
    conn = connect.memory()
    conn.executescript(ddl)
    return conn


def virtual_table_names(conn: sqlite3.Connection) -> set[str]:
    """FTS5 tables and the shadow tables they own: neither can be STRICT,
    neither declares foreign keys."""
    virt = [
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE 'CREATE VIRTUAL TABLE%'")
    ]
    return {
        name
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        if any(name == v or name.startswith(v + "_") for v in virt)
    }


def _tables(conn: sqlite3.Connection) -> list[str]:
    virt = virtual_table_names(conn)
    return [
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        if r[0] not in virt
    ]


def has_rowid(conn: sqlite3.Connection, table: str) -> bool:
    try:
        conn.execute(f"SELECT rowid FROM {table} LIMIT 0")
    except sqlite3.OperationalError:
        return False
    return True


def unconstrained_reference_columns(conn: sqlite3.Connection) -> list[str]:
    """Columns ending in _id that declare no foreign key and are not named
    in policy.NOT_A_REFERENCE. A primary-key component is not excused."""
    out = []
    for table in _tables(conn):
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info({table})")]
        fks = {r[3] for r in conn.execute(f"PRAGMA foreign_key_list({table})")}
        out.extend(
            f"{table}.{col}"
            for col in cols
            if col.endswith("_id") and col not in fks and (table, col) not in policy.NOT_A_REFERENCE
        )
    return out


def rule_schema(ddl: str | None = None) -> list[Finding]:
    """SG701 a table that is not STRICT; SG702 a join table paying for a
    rowid, or the long tail without one; SG703 a foreign key at a missing
    table; SG704 a reference column with no foreign key; SG705 a
    load-bearing reference at the wrong table; SG706 a foreign key with no
    stated delete action."""
    text = ddl if ddl is not None else SCHEMA.read_text(encoding="utf-8")
    conn = built(text)
    try:
        return _rule_schema(conn, text)
    finally:
        conn.close()


def _rule_schema(conn: sqlite3.Connection, text: str) -> list[Finding]:
    found: list[Finding] = []
    at = SCHEMA

    def where(table: str) -> int:
        match = re.search(rf"^CREATE (?:VIRTUAL )?TABLE {re.escape(table)} ?\(", text, re.MULTILINE)
        return text[: match.start()].count("\n") + 1 if match else 1

    virt = virtual_table_names(conn)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for name, sql in conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ):
        if name not in virt and "STRICT" not in (sql or "").upper():
            found.append(
                Finding(at, where(name), 0, "SG701", f"{name} is not STRICT, so its column types are advisory")
            )
    found.extend(
        Finding(at, where(table), 0, "SG702", f"{table} has a composite key and still pays for a rowid")
        for table in policy.WITHOUT_ROWID
        if table in names and has_rowid(conn, table)
    )
    found.extend(
        Finding(at, where(table), 0, "SG702", f"{table} holds the long tail and must keep its rowid")
        for table in policy.KEEPS_ROWID
        if table in names and not has_rowid(conn, table)
    )
    actual: dict[tuple[str, str], str] = {}
    actions: list[tuple[str, str]] = []
    for table in sorted(names - virt):
        for row in conn.execute(f"PRAGMA foreign_key_list({table})"):
            actual[(table, row[3])] = row[2]
            actions.append((table, row[6]))
            if row[2] not in names:
                found.append(
                    Finding(at, where(table), 0, "SG703", f"{table}.{row[3]} references {row[2]}, which does not exist")
                )
    for column in unconstrained_reference_columns(conn):
        table = column.split(".", 1)[0]
        found.append(
            Finding(
                at,
                where(table),
                0,
                "SG704",
                f"{column} names a row in another table but declares no foreign key; add REFERENCES with an"
                " explicit ON DELETE, or name it in sglint/policy.py NOT_A_REFERENCE with the reason",
            )
        )
    for (table, column), target in policy.LOAD_BEARING_REFERENCES.items():
        if actual.get((table, column)) != target:
            found.append(
                Finding(
                    at,
                    where(table),
                    0,
                    "SG705",
                    f"{table}.{column} should reference {target}, not {actual.get((table, column))}",
                )
            )
    for table, action in actions:
        if action not in ("CASCADE", "SET NULL", "RESTRICT"):
            found.append(
                Finding(at, where(table), 0, "SG706", f"a foreign key on {table} states no delete action ({action})")
            )
    if not actions:
        found.append(Finding(at, 1, 0, "SG700", "no foreign keys found; the sweep is broken"))
    return found


def rule_migrations(root: pathlib.Path = REPO_ROOT) -> list[Finding]:
    """SG707: USER_VERSION moved and no step leaves a version behind it;
    SG708: a step already leaves the current version (USER_VERSION forgot
    to move with it). Read statically from db/connect.py and db/migrate.py."""
    connect = ast.parse((root / "db" / "connect.py").read_text(encoding="utf-8"))
    version = next(
        node.value.value
        for node in ast.walk(connect)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "USER_VERSION" for t in node.targets)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, int)
    )
    migrate = ast.parse((root / "db" / "migrate.py").read_text(encoding="utf-8"))
    steps = {
        dec.args[0].value
        for node in ast.walk(migrate)
        if isinstance(node, ast.FunctionDef)
        for dec in node.decorator_list
        if isinstance(dec, ast.Call)
        and isinstance(dec.func, ast.Name)
        and dec.func.id == "step"
        and dec.args
        and isinstance(dec.args[0], ast.Constant)
    }
    found: list[Finding] = []
    missing = [v for v in range(1, version) if v not in steps]
    if missing:
        found.append(
            Finding(
                root / "db" / "migrate.py",
                1,
                0,
                "SG707",
                f"USER_VERSION is {version} but no migration leaves v{missing}; a database there cannot be opened",
            )
        )
    if version in steps:
        found.append(
            Finding(
                root / "db" / "connect.py",
                1,
                0,
                "SG708",
                f"a step already leaves v{version}; did USER_VERSION forget to move with it?",
            )
        )
    return found


def _checked_members(ddl: str, table: str, column: str) -> frozenset[str] | None:
    """The values a `CHECK (<column> IN (...))` constraint admits, or None
    when the table's DDL carries no such constraint for that column."""
    declaration = re.search(rf"CREATE TABLE {table} \((.*?)\n\) STRICT;", ddl, re.DOTALL)
    if declaration is None:
        return None
    found = re.search(rf"CHECK \({column} IN\s*\(([^)]*)\)\)", declaration.group(1), re.DOTALL)
    if found is None:
        return None
    return frozenset(re.findall(r"'([^']*)'", found.group(1)))


def _literal_members(module: ast.Module, name: str) -> frozenset[str] | None:
    """The members of a module-level `X = Literal["a", "b"]`, or None when
    no such assignment is there to read."""
    for node in module.body:
        targets: typing.Sequence[ast.expr]
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not any(isinstance(one, ast.Name) and one.id == name for one in targets):
            continue
        if not isinstance(value, ast.Subscript):
            return None
        # `Literal[...]` and `typing.Literal[...]` are the same type, and a
        # vocabulary written the second way is not a vocabulary this rule
        # may refuse to read.
        held = value.value
        named = held.id if isinstance(held, ast.Name) else held.attr if isinstance(held, ast.Attribute) else None
        if named != "Literal":
            return None
        held = value.slice.elts if isinstance(value.slice, ast.Tuple) else [value.slice]
        spelled = [one.value for one in held if isinstance(one, ast.Constant) and isinstance(one.value, str)]
        return frozenset(spelled) if len(spelled) == len(held) else None
    return None


def rule_wire_vocabularies(
    root: pathlib.Path = REPO_ROOT,
    ddl: str | None = None,
    wanted: dict[str, dict[str, tuple[str, str]]] | None = None,
) -> list[Finding]:
    """SG709: a closed vocabulary the wire restates does not match the
    schema's CHECK constraint, or could not be read from either side.

    A vocabulary the browser is given as a union has to be the one the
    database will accept. Both halves are text, so this compares them
    without building a database or serving a request -- and it reports a
    half it could not read rather than passing, because an unreadable
    constraint and an equal one are not the same result.
    """
    text = ddl if ddl is not None else SCHEMA.read_text(encoding="utf-8")
    declared = policy.WIRE_VOCABULARIES if wanted is None else wanted
    found: list[Finding] = []
    for relative, names in declared.items():
        source = root / relative
        module = ast.parse(source.read_text(encoding="utf-8"))
        for name, (table, column) in sorted(names.items()):
            stated = _literal_members(module, name)
            constrained = _checked_members(text, table, column)
            if stated is None:
                found.append(Finding(source, 1, 0, "SG709", f"{name} is not a module-level Literal of strings"))
                continue
            if constrained is None:
                found.append(Finding(SCHEMA, 1, 0, "SG709", f"no CHECK ({column} IN ...) on {table} for {name}"))
                continue
            if stated != constrained:
                missing = sorted(constrained - stated)
                extra = sorted(stated - constrained)
                found.append(
                    Finding(
                        source,
                        1,
                        0,
                        "SG709",
                        f"{name} disagrees with {table}.{column}: missing {missing}, unknown {extra}",
                    )
                )
    return found


def _real_tables(conn: sqlite3.Connection) -> list[str]:
    """Every table the structural sweeps judge: FTS5 tables and the shadow
    tables they own are neither STRICT nor indexed by us."""
    virt = virtual_table_names(conn)
    return [
        name
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        if name not in virt
    ]


def rule_foreign_key_indexes(ddl: str | None = None) -> list[Finding]:
    """SG710: a foreign key column nothing can be looked up by.

    SQLite's own `.lint fkey-indexes`, as a gate. Deleting a parent row
    makes SQLite run `SELECT 1 FROM child WHERE child_key = ?` against
    every child table (sqlite/sqlite src/shell.c.in:5981-6014). Unindexed
    that is a full scan per delete, so removing one file walks every
    derivation, every annotation and every piece of feedback in the
    library -- work that is invisible until the library is large.

    A column leading an index or belonging to the primary key is already
    reachable; nothing else is.
    """
    text = ddl if ddl is not None else SCHEMA.read_text(encoding="utf-8")
    conn = built(text)
    try:
        found: list[Finding] = []
        for table in _real_tables(conn):
            leading = set()
            for (index,) in conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?", (table,)):
                columns = list(conn.execute(f"PRAGMA index_info({index})"))
                if columns:
                    leading.add(columns[0][2])
            primary = {row[1] for row in conn.execute(f"PRAGMA table_info({table})") if row[5]}
            for row in conn.execute(f"PRAGMA foreign_key_list({table})"):
                column = row[3]
                if column not in leading and column not in primary:
                    found.append(
                        Finding(
                            SCHEMA,
                            1,
                            0,
                            "SG710",
                            f"deleting a {row[2]} row scans {table}: {table}.{column} leads no index",
                        )
                    )
        return found
    finally:
        conn.close()


def rule_closed_columns(ddl: str | None = None, free_text: frozenset[str] | None = None) -> list[Finding]:
    """SG711: a column naming a fixed set that accepts anything.

    An unconstrained vocabulary accepts every typo, and every typo is a
    row that never matches the filter meant to find it. `param_key.source`
    and `file_param.source` name the same set, and `slug_history.kind` and
    `entity.kind` name the same set: a pair enforced in one place and not
    the other drifts on the first direct write.

    A TEXT column whose name ends in one of policy.VOCABULARY_ENDINGS is
    read as a vocabulary unless it is named in policy.FREE_TEXT.
    """
    text = ddl if ddl is not None else SCHEMA.read_text(encoding="utf-8")
    excused = policy.FREE_TEXT if free_text is None else free_text
    conn = built(text)
    try:
        found: list[Finding] = []
        for table in _real_tables(conn):
            sql = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0] or ""
            for row in conn.execute(f"PRAGMA table_info({table})"):
                column, kind = row[1], row[2]
                if kind != "TEXT" or column in excused:
                    continue
                if not column.endswith(policy.VOCABULARY_ENDINGS):
                    continue
                if not re.search(rf"\b{column}\b[^,]*CHECK|CHECK\s*\(\s*{column}\b", sql):
                    found.append(
                        Finding(
                            SCHEMA,
                            1,
                            0,
                            "SG711",
                            f"{table}.{column} names a fixed set and accepts anything; add a CHECK, or name it"
                            " in sglint/policy.py FREE_TEXT with the reason",
                        )
                    )
        return found
    finally:
        conn.close()


def rule_index_prefixes(ddl: str | None = None) -> list[Finding]:
    """SG712: an index whose columns are a prefix of another's.

    Under the same partial predicate that is write cost on every insert
    for a read the wider one already serves. Stated as a prefix rule
    rather than measured by dropping and replanning: a planner probe on
    the leading column alone condemns any composite index whose first
    column is also covered by a narrower one, which flagged
    `file_param_key_num` -- it exists for `key = ? AND value_num BETWEEN
    ?` and is not redundant at all.

    The predicates have to match too: an index over all rows is not
    replaced by a partial one, which cannot serve the rows it
    excludes.
    """
    text = ddl if ddl is not None else SCHEMA.read_text(encoding="utf-8")
    conn = built(text)
    try:

        def shape(index: str) -> tuple[list[str], str | None]:
            columns = [row[2] for row in conn.execute(f"PRAGMA index_xinfo({index})") if row[5]]
            sql = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (index,)).fetchone()[0] or ""
            where = sql.upper().split(" WHERE ", 1)
            return columns, (where[1].strip() if len(where) > 1 else None)

        found: list[Finding] = []
        for table in _real_tables(conn):
            indexes = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL", (table,)
                )
            ]
            for candidate in indexes:
                columns, predicate = shape(candidate)
                for other in indexes:
                    if other == candidate:
                        continue
                    wider, other_predicate = shape(other)
                    if len(wider) > len(columns) and wider[: len(columns)] == columns and other_predicate == predicate:
                        found.append(
                            Finding(SCHEMA, 1, 0, "SG712", f"{candidate} is a prefix of {other} and earns nothing")
                        )
        return found
    finally:
        conn.close()


def _dict_keys(module: ast.Module, name: str) -> frozenset[str] | None:
    """The string keys of a module-level `X = {...}`, or None when no such
    assignment is there to read."""
    for node in module.body:
        targets: typing.Sequence[ast.expr]
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not any(isinstance(one, ast.Name) and one.id == name for one in targets):
            continue
        if not isinstance(value, ast.Dict):
            return None
        spelled = [k.value for k in value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
        return frozenset(spelled) if len(spelled) == len(value.keys) else None
    return None


def rule_vocabulary_handlers(
    root: pathlib.Path = REPO_ROOT, wanted: dict[str, dict[str, tuple[str, str]]] | None = None
) -> list[Finding]:
    """SG713: a dispatch table that does not cover its vocabulary.

    An event the ledger can write and the console cannot say is a failing
    build, and so is words for an event that cannot be written. Both
    halves are module-level literals -- a `Literal` and a dict's keys --
    so this is text against text, and needs no database and no rendering.

    What each renderer actually SAYS is behaviour and stays a test; that
    every member has one is this.
    """
    declared = policy.VOCABULARY_HANDLERS if wanted is None else wanted
    found: list[Finding] = []
    for relative, tables in declared.items():
        source = root / relative
        module = ast.parse(source.read_text(encoding="utf-8"))
        for table, (vocabulary_at, vocabulary) in sorted(tables.items()):
            covered = _dict_keys(module, table)
            spoken = _literal_members(ast.parse((root / vocabulary_at).read_text(encoding="utf-8")), vocabulary)
            if covered is None:
                found.append(Finding(source, 1, 0, "SG713", f"{table} is not a module-level dict of string keys"))
                continue
            if spoken is None:
                found.append(
                    Finding(
                        root / vocabulary_at, 1, 0, "SG713", f"{vocabulary} is not a module-level Literal of strings"
                    )
                )
                continue
            if covered != spoken:
                missing = sorted(spoken - covered)
                extra = sorted(covered - spoken)
                found.append(
                    Finding(
                        source,
                        1,
                        0,
                        "SG713",
                        f"{table} does not cover {vocabulary}: missing {missing}, unknown {extra}",
                    )
                )
    return found


def _calls(node: ast.AST, name: str) -> bool:
    """Whether anything inside `node` calls the bare function `name`."""
    return any(
        isinstance(one, ast.Call) and isinstance(one.func, ast.Name) and one.func.id == name for one in ast.walk(node)
    )


def rule_handlers_report(root: pathlib.Path = REPO_ROOT, wanted: dict[str, str] | None = None) -> list[Finding]:
    """SG415: a shipped job handler that never reaches the reporting seam.

    A long handler that says nothing between item.started and item.done is
    a frozen progress bar, and the page has no way to tell it from a hung
    worker. Every handler the runner ships must call `report()` somewhere.

    A kind whose handler only dispatches is named in
    policy.HANDLER_DISPATCH, and its modes are what must report -- the
    router itself says nothing and should not.
    """
    declared = policy.HANDLER_REGISTRIES if wanted is None else wanted
    found: list[Finding] = []
    for relative, registry in declared.items():
        source = root / relative
        module = ast.parse(source.read_text(encoding="utf-8"))
        defined = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)}
        held = next(
            (
                node.value
                for node in module.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == registry for t in node.targets)
                and isinstance(node.value, ast.Dict)
            ),
            None,
        )
        if not isinstance(held, ast.Dict):
            found.append(Finding(source, 1, 0, "SG415", f"{registry} is not a module-level dict of handlers"))
            continue
        for key, value in zip(held.keys, held.values, strict=True):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            names = policy.HANDLER_DISPATCH.get(key.value)
            if names is None:
                names = (value.id,) if isinstance(value, ast.Name) else ()
            for name in names:
                node = defined.get(name)
                if node is None:
                    found.append(Finding(source, 1, 0, "SG415", f"{registry}[{key.value!r}] names no function {name}"))
                elif not _calls(node, policy.REPORTING_CALL):
                    found.append(
                        Finding(
                            source,
                            node.lineno,
                            node.col_offset,
                            "SG415",
                            f"{name} handles {key.value!r} and never calls {policy.REPORTING_CALL}();"
                            " a handler that says nothing between item.started and item.done is a frozen bar",
                        )
                    )
    return found


def _statements(root: pathlib.Path, conn: sqlite3.Connection) -> str:
    """Every statement this repository can run, flattened.

    The Python is read as text because SQL here is written as adjacent
    string literals, so a statement only reads as one after the delimiters
    between its halves are gone. Triggers are producers too -- param_key
    is filled entirely by one -- so the DDL's own triggers are read beside
    the Python.
    """
    source = "".join(path.read_text(encoding="utf-8") for path in sorted((root / "db").rglob("*.py")))
    source += "".join(row[0] or "" for row in conn.execute("SELECT sql FROM sqlite_master WHERE type='trigger'"))
    return " ".join(re.sub(r"""["']""", " ", source).split())


def written_columns(root: pathlib.Path, conn: sqlite3.Connection) -> tuple[dict[str, set[str]], set[str]]:
    """Each table mapped to the columns some INSERT or UPDATE names on it,
    and the tables written without a column list at all.

    Parsed from the statements, not matched against the text. This used to
    be `re.search(rf"\b{column}\b", source)` over every db/*.py file
    concatenated -- comments and docstrings included -- so a column counted
    as produced when its name appeared anywhere: in prose, as a local, as
    an attribute of an unrelated object, or as a column of another table.
    `file.width` and `file.height` passed that for years' worth of
    `typed.width` and `raw.width` in db/ingest.py while nothing has ever
    written either one.
    """
    flat = _statements(root, conn)
    written: dict[str, set[str]] = {}
    everything: set[str] = set()
    for match in re.finditer(
        r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\(([^()]*)\))?", flat, re.IGNORECASE
    ):
        table = match.group(1)
        if match.group(3) is None:
            everything.add(table)  # no column list: every column is written
            continue
        written.setdefault(table, set()).update(name.strip() for name in match.group(3).split(","))
    for match in re.finditer(
        r"UPDATE\s+([A-Za-z_][A-Za-z0-9_]*)\s+SET\s+(.*?)(?:\s+WHERE\s|\s+RETURNING\s|;)", flat, re.IGNORECASE
    ):
        written.setdefault(match.group(1), set()).update(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", match.group(2)))
    # `DO UPDATE SET` needs no pass of its own: an upsert can only set
    # columns its INSERT already named, and those are collected above.
    return written, everything


#: What a column with no producer must say about itself, in the comment
#: block above it.
UNWRITTEN_ADMISSION = "NOTHING WRITES THIS YET"


def rule_written_columns(root: pathlib.Path = REPO_ROOT, ddl: str | None = None) -> list[Finding]:
    """SG714: a column no producer fills, and no admission in the DDL.

    Such a column reads as a feature -- a facet built on it returns an
    empty library, and nothing distinguishes "no video has a duration"
    from "nothing has ever measured one". Whichever it is, the DDL has to
    say so.
    """
    text = ddl if ddl is not None else SCHEMA.read_text(encoding="utf-8")
    conn = built(text)
    try:
        written, everything = written_columns(root, conn)
        found: list[Finding] = []
        for table in _real_tables(conn):
            if table in everything:
                continue
            declaration = conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0] or ""
            for row in conn.execute(f"PRAGMA table_info({table})"):
                column, is_pk = row[1], row[5]
                if is_pk or column in written.get(table, ()):
                    continue
                # the admission sits in the comment block ABOVE the column,
                # so the whole declaration is read rather than forward from
                # the name
                if UNWRITTEN_ADMISSION in declaration.upper() and re.search(
                    rf"{UNWRITTEN_ADMISSION}.{{0,400}}\b{re.escape(column)}\b",
                    declaration,
                    re.IGNORECASE | re.DOTALL,
                ):
                    continue
                found.append(
                    Finding(
                        SCHEMA,
                        1,
                        0,
                        "SG714",
                        f"no producer writes {table}.{column}, and the DDL does not admit it",
                    )
                )
        return found
    finally:
        conn.close()
