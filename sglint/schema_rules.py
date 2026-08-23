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
