"""Deterministic compiler: ValidatedQuery -> parameterized read-only SELECT.

This is the only module that assembles SQL text. It accepts nothing but a
:class:`~omniquery.validation.ValidatedQuery` (proof that validation ran)
plus :class:`CompileParams` (the injected, deterministic runtime context: a
clock reading, the gallery base path, the caller's client_uuid, and any
pre-resolved AI predicate results). Every literal value -- including LIKE
patterns and our own constant filters -- is bound as a ``?`` parameter;
nothing from a Cond, base_path, or client_uuid is ever interpolated into
the SQL text itself. Given the same inputs it always produces byte-identical
SQL and parameters: it never reads the clock or touches the filesystem.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from omniquery import fields
from omniquery.ast import Cond, Group, Node, Not
from omniquery.validation import ValidatedQuery

# SQLite comparison operators for the six scalar comparison ops.
_CMP_SQL = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}


class CompileError(ValueError):
    """Raised for internal compile-time inconsistencies (never user input)."""


@dataclass(frozen=True)
class CompileParams:
    """Injected runtime context: everything compile() needs beyond the query
    itself, supplied by the engine so compilation stays a pure function."""

    now_epoch: float  # 'now' in epoch seconds; anchor for relative-date values
    base_path: str    # gallery root; folder predicates only match beneath it
    client_uuid: Optional[str] = None  # keys the caller-specific 'my_rating' subquery
    # (field_name, json.dumps(cond.value, sort_keys=True)) -> resolved file ids.
    # Populated by the engine's AI pre-resolution pass before compiling any
    # query that contains a file_ref condition.
    ai_resolutions: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class CompiledQuery:
    """A ready-to-execute statement: SQL containing only '?' placeholders,
    plus its bind values."""

    sql: str
    params: tuple  # positional bind values, in placeholder order
    effective_limit: int  # row cap; bound as the LIMIT of "ids" queries


def compile(vq: ValidatedQuery, params: CompileParams) -> CompiledQuery:
    """Compile a validated query into a single SELECT over DISTINCT file ids:
    'ids' queries carry ORDER BY and a bound LIMIT; 'count' queries wrap the
    id set in COUNT(*) and take neither."""
    assert isinstance(vq, ValidatedQuery), "compile() requires a validated query"
    query = vq.query

    where_sql = ""
    where_params: List[Any] = []
    if query.where is not None:
        where_sql, where_params = _compile_node(query.where, params)

    if query.result == "count":
        sql = "SELECT COUNT(*) FROM (SELECT DISTINCT f.id FROM files f"
        if where_sql:
            sql += f" WHERE {where_sql}"
        sql += ")"
        return CompiledQuery(sql=sql, params=tuple(where_params),
                              effective_limit=vq.effective_limit)

    sql = "SELECT DISTINCT f.id FROM files f"
    if where_sql:
        sql += f" WHERE {where_sql}"
    sql += " ORDER BY " + _order_by_clause(query.order_by)
    sql += " LIMIT ?"
    all_params = where_params + [vq.effective_limit]
    return CompiledQuery(sql=sql, params=tuple(all_params),
                          effective_limit=vq.effective_limit)


def _order_by_clause(order_by: tuple) -> str:
    """Comma-joined ORDER BY expressions for the given specs; always ends
    with an f.id tiebreaker."""
    parts = []
    for ospec in order_by:
        spec = fields.get_field(ospec.field)
        if spec is None or not spec.orderable:  # validated already
            raise CompileError(f"field '{ospec.field}' is not a valid order_by field")
        direction = "ASC" if ospec.direction == "asc" else "DESC"
        parts.append(f"{spec.order_expr} {direction}")
    # A stable tiebreaker keeps LIMIT/pagination deterministic even when the
    # user-chosen order columns tie.
    parts.append("f.id ASC")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Node compilation (AND/OR/NOT/Cond -> SQL text + positional params)
# ---------------------------------------------------------------------------

def _compile_node(node: Node, params: CompileParams) -> Tuple[str, List[Any]]:
    """Recursively compile a where node to (SQL fragment, bind values);
    groups and negations parenthesize themselves, so nesting never depends
    on operator precedence."""
    if isinstance(node, Cond):
        return _compile_cond(node, params)
    if isinstance(node, Not):
        inner_sql, inner_params = _compile_node(node.child, params)
        return f"NOT ({inner_sql})", inner_params
    if isinstance(node, Group):
        parts: List[str] = []
        all_params: List[Any] = []
        for child in node.children:
            csql, cparams = _compile_node(child, params)
            parts.append(csql)
            all_params.extend(cparams)
        joiner = " AND " if node.op == "and" else " OR "
        return "(" + joiner.join(parts) + ")", all_params
    raise CompileError(f"unhandled node type {type(node).__name__}")


def _compile_cond(cond: Cond, params: CompileParams) -> Tuple[str, List[Any]]:
    """Dispatch a leaf condition to the builder for its field's kind."""
    spec = fields.get_field(cond.field)
    if spec is None:  # validated already
        raise CompileError(f"unknown field '{cond.field}'")

    if spec.kind == fields.Kind.TEXT:
        return _build_text(spec, cond.op, cond.value, params)
    if spec.kind == fields.Kind.NUMBER:
        return _build_number(spec, cond.op, cond.value, params)
    if spec.kind == fields.Kind.BOOL:
        return _build_bool(spec, cond.op, cond.value)
    if spec.kind == fields.Kind.ENUM:
        return _build_enum(spec, cond.op, cond.value)
    if spec.kind == fields.Kind.DATETIME:
        return _build_datetime(spec, cond.op, cond.value, params)
    if spec.kind == fields.Kind.FILE_REF:
        return _build_file_ref(spec, cond.value, params)
    raise CompileError(f"unhandled kind {spec.kind!r}")


# ---------------------------------------------------------------------------
# LIKE pattern helpers (case-insensitive by SQLite default for ASCII)
# ---------------------------------------------------------------------------

def _escape_like(value: str) -> str:
    """Escape LIKE wildcards ('%', '_') and the escape character itself so a
    bound value matches only literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _like_clause(column: str, op: str, value: str) -> Tuple[str, List[Any]]:
    """One LIKE comparison with ESCAPE '\\'. eq/ne also go through LIKE, so
    text equality is case-insensitive (for ASCII) like every other text op."""
    pattern = _escape_like(value)
    if op == "eq":
        return f"{column} LIKE ? ESCAPE '\\'", [pattern]
    if op == "ne":
        return f"{column} NOT LIKE ? ESCAPE '\\'", [pattern]
    if op == "contains":
        return f"{column} LIKE ? ESCAPE '\\'", ["%" + pattern + "%"]
    if op == "prefix":
        return f"{column} LIKE ? ESCAPE '\\'", [pattern + "%"]
    if op == "suffix":
        return f"{column} LIKE ? ESCAPE '\\'", ["%" + pattern]
    raise CompileError(f"unhandled text op {op!r}")


# ---------------------------------------------------------------------------
# TEXT
# ---------------------------------------------------------------------------

def _build_text(spec: fields.FieldSpec, op: str, value: Any,
                 params: CompileParams) -> Tuple[str, List[Any]]:
    """Text predicate per strategy: direct column LIKE, folder path match, or
    an EXISTS probe into a per-file child table."""
    if op in ("is_null", "not_null"):
        return (f"{spec.column} IS NULL" if op == "is_null"
                else f"{spec.column} IS NOT NULL"), []

    if spec.strategy == fields.Strategy.COLUMN:
        return _like_clause(spec.column, op, value)

    if spec.strategy == fields.Strategy.FOLDER:
        return _build_folder(op, value, params)

    if spec.strategy == fields.Strategy.EXISTS_TEXT:
        inner_sql, inner_params = _like_clause(spec.inner_column, op, value)
        core = (f"SELECT 1 FROM {spec.table} {spec.alias} "
                f"WHERE {spec.alias}.file_id = f.id AND {inner_sql}")
        return f"EXISTS ({core})", inner_params

    if spec.strategy == fields.Strategy.EXISTS_NAMED:
        name_sql, name_params = _like_clause("c.name", op, value)
        core = ("SELECT 1 FROM collection_files cf "
                "JOIN collections c ON c.id = cf.collection_id "
                f"WHERE cf.file_id = f.id AND c.type = ? AND {name_sql}")
        return f"EXISTS ({core})", [spec.type_filter] + name_params

    if spec.strategy == fields.Strategy.EXISTS_USER:
        name_sql, name_params = _like_clause("u.username", op, value)
        core = (f"SELECT 1 FROM {spec.table} {spec.alias} "
                f"JOIN users u ON {spec.alias}.client_uuid = CAST(u.user_id AS TEXT) "
                f"WHERE {spec.alias}.file_id = f.id AND {name_sql}")
        return f"EXISTS ({core})", name_params

    raise CompileError(f"unhandled text strategy {spec.strategy!r}")


def _build_folder(op: str, value: str, params: CompileParams) -> Tuple[str, List[Any]]:
    """Folder predicate over '/'-normalized paths, anchored beneath base_path."""
    # Stored paths carry whichever separator the scanning host used, so a
    # Windows-scanned library stores 'C:\gallery\output\foo.png'. Normalize
    # both the column and every compared value to '/' so folder predicates
    # match regardless of the separator convention on disk.
    column = "REPLACE(f.path, '\\', '/')"
    base = params.base_path.replace("\\", "/").rstrip("/")
    folder = value.replace("\\", "/").strip("/")
    if op == "eq":
        pattern = _escape_like(f"{base}/{folder}/") + "%"
        return f"{column} LIKE ? ESCAPE '\\'", [pattern]
    # contains: file lives somewhere under base_path AND the folder value
    # appears anywhere in its path.
    base_pattern = _escape_like(base) + "/%"
    folder_pattern = "%" + _escape_like(folder) + "%"
    return (f"({column} LIKE ? ESCAPE '\\' AND {column} LIKE ? ESCAPE '\\')",
            [base_pattern, folder_pattern])


# ---------------------------------------------------------------------------
# NUMBER
# ---------------------------------------------------------------------------

def _build_number(spec: fields.FieldSpec, op: str, value: Any,
                   params: CompileParams) -> Tuple[str, List[Any]]:
    """Numeric comparison against a column, computed expression, or scalar
    subquery; my_rating additionally binds the caller's client_uuid, and
    face_cluster compiles to an EXISTS membership probe."""
    if spec.strategy == fields.Strategy.MY_RATING:
        sub = "(SELECT r.rating FROM file_ratings r WHERE r.file_id = f.id AND r.client_uuid = ?)"
        if op == "between":
            lo, hi = value
            return f"{sub} BETWEEN ? AND ?", [params.client_uuid, lo, hi]
        return f"{sub} {_CMP_SQL[op]} ?", [params.client_uuid, value]

    if spec.strategy == fields.Strategy.EXISTS_NUMERIC:
        core = "SELECT 1 FROM ai_face_instances fa WHERE fa.file_id = f.id AND fa.cluster_id = ?"
        return f"EXISTS ({core})", [value]

    if spec.strategy in (fields.Strategy.COLUMN, fields.Strategy.EXPR,
                          fields.Strategy.SUBQUERY_SCALAR):
        expr = spec.column if spec.strategy == fields.Strategy.COLUMN else spec.expr
        if op == "between":
            lo, hi = value
            return f"{expr} BETWEEN ? AND ?", [lo, hi]
        return f"{expr} {_CMP_SQL[op]} ?", [value]

    raise CompileError(f"unhandled number strategy {spec.strategy!r}")


# ---------------------------------------------------------------------------
# BOOL
# ---------------------------------------------------------------------------

def _build_bool(spec: fields.FieldSpec, _op: str, value: bool) -> Tuple[str, List[Any]]:
    """Boolean predicate: 0/1 column comparison, or EXISTS / NOT EXISTS for
    presence-style fields."""
    if spec.strategy == fields.Strategy.COLUMN:
        return f"{spec.column} = ?", [1 if value else 0]
    if spec.strategy == fields.Strategy.EXISTS_BOOL:
        core = "SELECT 1 FROM ai_face_instances fa WHERE fa.file_id = f.id"
        return (f"EXISTS ({core})" if value else f"NOT EXISTS ({core})"), []
    raise CompileError(f"unhandled bool strategy {spec.strategy!r}")


# ---------------------------------------------------------------------------
# ENUM
# ---------------------------------------------------------------------------

def _inner_match(column: str, op: str, value: Any) -> Tuple[str, List[Any]]:
    """eq/ne/in match against a single column; 'ne' is handled by the caller
    wrapping the whole EXISTS in NOT EXISTS (inner match stays an eq/in)."""
    if op == "in":
        placeholders = ",".join(["?"] * len(value))
        return f"{column} IN ({placeholders})", list(value)
    return f"{column} = ?", [value]


def _build_enum(spec: fields.FieldSpec, op: str, value: Any) -> Tuple[str, List[Any]]:
    """Enum predicate; for EXISTS strategies 'ne' means "no matching row",
    compiled as NOT EXISTS around an inner equality (see _inner_match)."""
    if spec.strategy == fields.Strategy.COLUMN:
        if op == "in":
            placeholders = ",".join(["?"] * len(value))
            return f"{spec.column} IN ({placeholders})", list(value)
        return f"{spec.column} {_CMP_SQL[op]} ?", [value]

    if spec.strategy == fields.Strategy.EXISTS_NAMED:
        inner_sql, inner_params = _inner_match("c.name", op, value)
        core = ("SELECT 1 FROM collection_files cf "
                "JOIN collections c ON c.id = cf.collection_id "
                f"WHERE cf.file_id = f.id AND c.type = ? AND {inner_sql}")
        all_params = [spec.type_filter] + inner_params
        return (f"NOT EXISTS ({core})" if op == "ne" else f"EXISTS ({core})"), all_params

    if spec.strategy == fields.Strategy.EXISTS_ENUM:
        inner_sql, inner_params = _inner_match("rf.type", op, value)
        core = f"SELECT 1 FROM ai_review_findings rf WHERE rf.file_id = f.id AND {inner_sql}"
        return (f"NOT EXISTS ({core})" if op == "ne" else f"EXISTS ({core})"), inner_params

    raise CompileError(f"unhandled enum strategy {spec.strategy!r}")


# ---------------------------------------------------------------------------
# DATETIME
# ---------------------------------------------------------------------------

def _local_epoch(dt: datetime) -> float:
    """Naive datetime -> epoch seconds in the machine's local timezone."""
    # Naive datetimes mean local wall-clock time (a bare date is 00:00
    # local). mktime is a pure conversion; the only clock input to
    # compilation is the injected now_epoch.
    return time.mktime(dt.timetuple())


def _resolve_datetime(value: Any, now_epoch: float) -> Tuple[float, bool]:
    """Returns (epoch_seconds, is_bare_date). is_bare_date is True only for
    a plain 'YYYY-MM-DD' string, used by 'between' to cover whole days."""
    if isinstance(value, dict):
        if "days_ago" in value:
            return now_epoch - float(value["days_ago"]) * 86400.0, False
        return now_epoch - float(value["hours_ago"]) * 3600.0, False
    if len(value) == 10:
        return _local_epoch(datetime.strptime(value, "%Y-%m-%d")), True
    return _local_epoch(datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")), False


def _build_datetime(spec: fields.FieldSpec, op: str, value: Any,
                     params: CompileParams) -> Tuple[str, List[Any]]:
    """Datetime comparison in epoch seconds; 'between' is half-open
    [lo, hi) and widens a bare end date to cover that entire local day."""
    column = spec.column
    if op == "between":
        lo_raw, hi_raw = value
        lo_epoch, _ = _resolve_datetime(lo_raw, params.now_epoch)
        hi_epoch, hi_is_bare_date = _resolve_datetime(hi_raw, params.now_epoch)
        if hi_is_bare_date:
            # Whole day: up to (exclusive) the next local calendar midnight,
            # constructed as a date rather than midnight + 86400s -- on DST
            # transition days the local day is 23 or 25 hours long, so a
            # fixed offset would land an hour off the boundary.
            next_day = datetime.strptime(hi_raw, "%Y-%m-%d") + timedelta(days=1)
            hi_epoch = _local_epoch(next_day)
        return f"{column} >= ? AND {column} < ?", [lo_epoch, hi_epoch]
    epoch, _ = _resolve_datetime(value, params.now_epoch)
    return f"{column} {_CMP_SQL[op]} ?", [epoch]


# ---------------------------------------------------------------------------
# FILE_REF (resolved outside SQL by the engine's AI resolvers)
# ---------------------------------------------------------------------------

def resolution_key(field_name: str, value: Any) -> Tuple[str, str]:
    """Deterministic lookup key for CompileParams.ai_resolutions. Exposed so
    the engine can populate the mapping with keys the compiler will find."""
    return field_name, json.dumps(value, sort_keys=True, separators=(",", ":"))


def _build_file_ref(spec: fields.FieldSpec, value: Any,
                     params: CompileParams) -> Tuple[str, List[Any]]:
    """Membership test against the pre-resolved id list; an empty resolution
    compiles to a constant-false predicate."""
    key = resolution_key(spec.name, value)
    resolved_ids = params.ai_resolutions.get(key)
    if resolved_ids is None:
        raise CompileError(
            f"field '{spec.name}': no AI resolution supplied for value {value!r} "
            "(the engine must resolve file_ref predicates before compiling)"
        )
    if not resolved_ids:
        return "0=1", []
    placeholders = ",".join(["?"] * len(resolved_ids))
    return f"f.id IN ({placeholders})", list(resolved_ids)
