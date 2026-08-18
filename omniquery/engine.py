"""OmniQuery execution engine: parse -> validate -> resolve AI -> compile ->
execute on a read-only, authorizer-locked SQLite connection.

The engine is the only place in this package allowed to read the wall
clock (compiler.py must stay deterministic given an injected now_epoch) and
the only place that opens a database connection. It never writes to the
database: every connection is opened ``mode=ro`` and additionally locked
down with a SQLite authorizer that permits only SELECT/READ/FUNCTION.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from omniquery import fields
from omniquery.ast import ASTError, Query, iter_conditions, parse_query
from omniquery.compiler import CompileError, CompileParams, resolution_key
from omniquery.compiler import compile as compile_query
from omniquery.validation import AuthContext, ValidationError, validate

AiResolver = Callable[[Any], list[str]]  # validated file_ref value -> matching file ids

# 21 = SQLITE_SELECT, 20 = SQLITE_READ, 31 = SQLITE_FUNCTION. Everything else
# (INSERT/UPDATE/DELETE/ATTACH/PRAGMA/...) is denied at the C-engine level,
# matching smartgallery.py's execute_omniquery authorizer pattern.
_ALLOWED_AUTHORIZER_ACTIONS = frozenset(
    {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}
)


def _authorizer(action: int, _arg1, _arg2, _dbname, _source) -> int:
    """SQLite authorizer callback: permit SELECT/READ/FUNCTION, deny every
    other action code."""
    if action in _ALLOWED_AUTHORIZER_ACTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


@dataclass(frozen=True)
class QueryOutcome:
    """Result envelope for one run(): ok=True with a payload, or ok=False
    with an error message -- run() never raises for input-level failures."""

    ok: bool
    kind: str | None = None          # "ids" | "count"
    ids: list[str] | None = None     # file ids as strings; set when kind == "ids"
    count: int | None = None         # set when kind == "count"
    sql: str | None = None           # compiled statement, for logging/diagnostics
    params: tuple | None = None      # its bind values
    error: str | None = None         # human-readable failure reason when ok is False


class OmniQueryEngine:
    """Runs the full pipeline against one gallery database. Holds only static
    configuration; every execution opens its own short-lived read-only
    connection."""

    def __init__(self, db_path: str, base_path: str,
                 ai_resolvers: dict[str, AiResolver] | None = None):
        """ai_resolvers maps file_ref field names to resolver callables; a
        file_ref condition whose field has no resolver fails the query with
        an 'AI feature unavailable' error."""
        self.db_path = db_path
        self.base_path = base_path
        self.ai_resolvers = ai_resolvers or {}

    def run(self, ast_dict_or_query: dict | str | Query, ctx: AuthContext,
            now_epoch: float | None = None) -> QueryOutcome:
        """Execute one query end to end (parse, validate, resolve file_refs,
        compile, run). Every input-level failure comes back as an error
        QueryOutcome rather than an exception. now_epoch overrides the wall
        clock, making relative-date queries reproducible."""
        try:
            query = (ast_dict_or_query if isinstance(ast_dict_or_query, Query)
                      else parse_query(ast_dict_or_query))
        except ASTError as exc:
            return QueryOutcome(ok=False, error=f"invalid query: {exc}")

        try:
            vq = validate(query, ctx)
        except ValidationError as exc:
            return QueryOutcome(ok=False, error=str(exc))

        try:
            ai_resolutions = self._resolve_ai_predicates(query)
        except ValidationError as exc:
            return QueryOutcome(ok=False, error=str(exc))

        effective_now = now_epoch if now_epoch is not None else time.time()
        params = CompileParams(now_epoch=effective_now, base_path=self.base_path,
                                client_uuid=ctx.client_uuid, ai_resolutions=ai_resolutions)
        try:
            compiled = compile_query(vq, params)
        except CompileError as exc:
            return QueryOutcome(ok=False, error=str(exc))

        try:
            rows = self._execute(compiled.sql, compiled.params)
        except sqlite3.Error as exc:
            return QueryOutcome(ok=False, error=f"SQL execution error: {exc}")

        if query.result == "count":
            return QueryOutcome(ok=True, kind="count", count=int(rows[0][0]) if rows else 0,
                                 sql=compiled.sql, params=compiled.params)

        ids = [str(row[0]) for row in rows]
        return QueryOutcome(ok=True, kind="ids", ids=ids,
                             sql=compiled.sql, params=compiled.params)

    def _resolve_ai_predicates(self, query: Query) -> dict[Any, list[str]]:
        """Resolve every file_ref Cond's value to a concrete id list *before*
        compilation, so the compiler never has to call out of SQL land."""
        resolutions: dict[Any, list[str]] = {}
        for cond in iter_conditions(query.where):
            spec = fields.get_field(cond.field)
            if spec is None or spec.kind != fields.Kind.FILE_REF:
                continue
            key = resolution_key(cond.field, cond.value)
            if key in resolutions:
                continue
            resolver = self.ai_resolvers.get(cond.field)
            if resolver is None:
                raise ValidationError(f"AI feature unavailable: '{cond.field}' has no resolver")
            try:
                resolved = resolver(cond.value)
            except Exception as exc:  # resolver failure is not a validation bug
                raise ValidationError(
                    f"AI feature unavailable: '{cond.field}' resolver failed: {exc}"
                ) from exc
            resolutions[key] = [str(x) for x in resolved]
        return resolutions

    def _execute(self, sql: str, params: tuple) -> list:
        """Open a locked-down read-only connection and run exactly one
        statement. Exposed (not name-mangled) so tests can prove the
        authorizer blocks writes even if compile() is bypassed entirely."""
        uri = f"file:{os.path.abspath(self.db_path)}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            conn.set_authorizer(_authorizer)
            cursor = conn.execute(sql, params)
            return cursor.fetchall()
        finally:
            conn.close()
