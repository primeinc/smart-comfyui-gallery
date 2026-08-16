"""The ONE sandboxed gate for executing raw SELECT statements against the
gallery database. Both callers -- the Advanced manual-SQL endpoint and the
nl2sql model's generated queries -- go through this exact function; there
is no second, weaker path.

Defense layers (all three required, none sufficient alone):

1. Prefix check: after stripping comments, the statement must begin with
   SELECT.
2. True read-only connection: SQLite URI `mode=ro` -- writes fail at the
   VFS level even if everything else is bypassed.
3. C-engine authorizer: only SQLITE_SELECT / SQLITE_READ / SQLITE_FUNCTION
   are permitted, which blocks PRAGMA, ATTACH, and every write opcode
   inside the engine itself, including smuggled multi-statement text.

The result contract is intentionally narrow: the first column of every
row, de-duplicated in order, as strings -- "a list of file ids" is the
only thing OmniQuery ever asks SQL for.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from typing import List, Optional

_COMMENT_RE = re.compile(r"(/\*.*?\*/)|(--.*?(\n|$))", re.DOTALL)
_SELECT_RE = re.compile(r"^SELECT\b", re.IGNORECASE)


@dataclass(frozen=True)
class SqlExecResult:
    """Outcome of one sandboxed execution: ids on success, else `error`."""

    ids: Optional[List[str]]  # first-column values, de-duplicated; None on failure
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.ids is not None


def run_readonly_select(db_path: str, sql: str, max_rows: int = 5000) -> SqlExecResult:
    """Execute `sql` under the full sandbox and return first-column ids.

    Never raises: every rejection and engine error comes back as
    SqlExecResult(error=...). `max_rows` bounds the fetch so a
    pathological cross join cannot balloon memory.
    """
    sql = (sql or "").strip()
    if not sql:
        return SqlExecResult(ids=None, error="empty SQL")

    clean = _COMMENT_RE.sub("", sql).strip()
    if not _SELECT_RE.match(clean):
        return SqlExecResult(ids=None, error="only SELECT statements are allowed")

    def _authorizer(action, _arg1, _arg2, _dbname, _source):
        # 21 = SQLITE_SELECT, 20 = SQLITE_READ, 31 = SQLITE_FUNCTION
        if action in (21, 20, 31):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    db_uri = f"file:{os.path.abspath(db_path)}?mode=ro"
    try:
        with sqlite3.connect(db_uri, uri=True) as conn:
            conn.set_authorizer(_authorizer)
            cursor = conn.execute(sql)
            rows = cursor.fetchmany(max_rows)
    except Exception as exc:
        return SqlExecResult(ids=None, error=f"SQL execution error: {exc}")

    ids = [str(row[0]) for row in rows if row and row[0] is not None]
    return SqlExecResult(ids=list(dict.fromkeys(ids)))
