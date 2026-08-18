"""Schema, semantic, authorization, and complexity validation for OmniQuery.

This runs entirely outside any model: it checks a parsed :class:`Query`
(ast.py) against the field registry (fields.py) and an :class:`AuthContext`,
and produces a :class:`ValidatedQuery` -- the only input compiler.py accepts.

Nothing here builds SQL. Nothing here trusts anything but the AST's own
structural guarantees (ast.py already caps nesting depth, condition count,
list/dict sizes, and string length).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from omniquery import fields
from omniquery.ast import Query, iter_conditions

DEFAULT_LIMIT = 500  # row cap applied when the query specifies no limit
MAX_LIMIT = 2000     # hard ceiling on any requested limit
MAX_CORRELATED_FIELDS = 8  # distinct join/subquery-backed fields allowed per query

PRIVILEGED_ROLES = {"ADMIN", "MANAGER", "STAFF"}  # roles allowed to query privileged fields


class ValidationError(ValueError):
    """Raised for any schema/semantic/authorization/complexity violation."""


@dataclass(frozen=True)
class AuthContext:
    """Caller identity and entitlements, established by the host application
    outside any model. Validation trusts this object, never the query."""

    role: str  # caller's role name; gates privileged fields via PRIVILEGED_ROLES
    user_id: str | None  # authenticated user id, when known
    client_uuid: str | None  # per-device identity; keys 'my_rating' lookups
    ai_enabled: bool  # whether AI-derived fields may be queried at all


# ---------------------------------------------------------------------------
# ValidatedQuery: an immutable capability token. Only validate() (via the
# module-private _new_validated_query factory below) may construct one;
# compiler.py asserts isinstance(vq, ValidatedQuery) before compiling.
# ---------------------------------------------------------------------------

_CONSTRUCTOR_SENTINEL = object()  # module-private token; possession marks construction via validate()


class ValidatedQuery:
    """Immutable capability token: holding one is proof the wrapped Query
    passed validate() under the wrapped AuthContext."""

    __slots__ = ("_ctx", "_effective_limit", "_query")

    def __init__(self, query: Query, effective_limit: int, ctx: AuthContext,
                 *, _sentinel: Any = None):
        """Blocked outside this module: without the private sentinel the call
        fails, so validate() stays the only construction path."""
        if _sentinel is not _CONSTRUCTOR_SENTINEL:
            raise TypeError(
                "ValidatedQuery cannot be constructed directly; "
                "call omniquery.validation.validate() instead"
            )
        object.__setattr__(self, "_query", query)
        object.__setattr__(self, "_effective_limit", effective_limit)
        object.__setattr__(self, "_ctx", ctx)

    def __setattr__(self, key: str, value: Any) -> None:
        """Reject all mutation; state is set once in __init__ via object.__setattr__."""
        raise AttributeError("ValidatedQuery is immutable")

    @property
    def query(self) -> Query:
        """The validated Query, exactly as validate() received it."""
        return self._query

    @property
    def effective_limit(self) -> int:
        """Row cap the compiler must bind: the query's limit, or DEFAULT_LIMIT when unset."""
        return self._effective_limit

    @property
    def ctx(self) -> AuthContext:
        """The AuthContext the authorization decisions were made under."""
        return self._ctx


def _new_validated_query(query: Query, effective_limit: int,
                          ctx: AuthContext) -> ValidatedQuery:
    """Module-private factory: the one sanctioned ValidatedQuery constructor."""
    return ValidatedQuery(query, effective_limit, ctx, _sentinel=_CONSTRUCTOR_SENTINEL)


# ---------------------------------------------------------------------------
# Value validation
# ---------------------------------------------------------------------------

# Accepted absolute datetime literal shapes (interpreted as local time by the compiler).
_ISO_DATE_FMT = "%Y-%m-%d"
_ISO_DATETIME_FMT = "%Y-%m-%dT%H:%M:%S"


def _check_number(value: Any, field_name: str) -> None:
    """Require an int/float; bool is excluded despite being an int subclass."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"field '{field_name}': expected a numeric value, got {value!r}")


def _check_enum_member(spec: fields.FieldSpec, value: Any) -> None:
    """Require a string drawn from the spec's enum_values."""
    if not isinstance(value, str) or value not in (spec.enum_values or frozenset()):
        raise ValidationError(f"field '{spec.name}': invalid enum value {value!r}")


def _check_date_string(value: str, field_name: str) -> None:
    """Require 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS'."""
    for fmt in (_ISO_DATE_FMT, _ISO_DATETIME_FMT):
        try:
            datetime.strptime(value, fmt)
            return
        except ValueError:
            continue
    raise ValidationError(
        f"field '{field_name}': invalid date string {value!r} "
        "(expected 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SS')"
    )


def _check_datetime_value(spec: fields.FieldSpec, value: Any) -> None:
    """Accept an absolute date string, or a single-key relative object
    {'days_ago': N} / {'hours_ago': N} with N a non-negative number."""
    if isinstance(value, str):
        _check_date_string(value, spec.name)
        return
    if isinstance(value, dict):
        keys = set(value.keys())
        if keys == {"days_ago"} or keys == {"hours_ago"}:
            key = next(iter(keys))
            n = value[key]
            if isinstance(n, bool) or not isinstance(n, (int, float)) or n < 0:
                raise ValidationError(
                    f"field '{spec.name}': '{key}' must be a non-negative number"
                )
            return
        raise ValidationError(
            f"field '{spec.name}': relative date must be {{'days_ago': N}} "
            f"or {{'hours_ago': N}}, got keys {sorted(keys)}"
        )
    raise ValidationError(
        f"field '{spec.name}': expected a date string or relative-date object, got {value!r}"
    )


def _check_file_ref_value(spec: fields.FieldSpec, value: Any) -> None:
    """Accept a non-empty file id string; similarity fields may instead take
    {'file_id': str, 'k': 1..200} (k = neighbor count), while 'near_dup_of'
    is restricted to the plain string form."""
    if isinstance(value, str):
        if not value:
            raise ValidationError(f"field '{spec.name}': file id string must not be empty")
        return
    if isinstance(value, dict):
        if spec.name == "near_dup_of":
            raise ValidationError(
                f"field '{spec.name}': requires a plain file id string, not an object"
            )
        allowed_keys = {"file_id", "k"}
        keys = set(value.keys())
        if not keys <= allowed_keys or "file_id" not in keys:
            raise ValidationError(
                f"field '{spec.name}': object value must have 'file_id' and optional 'k'"
            )
        file_id = value["file_id"]
        if not isinstance(file_id, str) or not file_id:
            raise ValidationError(f"field '{spec.name}': 'file_id' must be a non-empty string")
        if "k" in value:
            k = value["k"]
            if isinstance(k, bool) or not isinstance(k, int) or not (1 <= k <= 200):
                raise ValidationError(f"field '{spec.name}': 'k' must be an integer in [1, 200]")
        return
    raise ValidationError(
        f"field '{spec.name}': expected a file id string or {{'file_id', 'k'}} object, "
        f"got {value!r}"
    )


def _validate_value(spec: fields.FieldSpec, op: str, value: Any) -> None:
    """Type-check a condition value against the field's kind and operator:
    list arities for 'between'/'in', enum membership, date/file-ref shapes."""
    if op in ("is_null", "not_null"):
        if value is not None:
            raise ValidationError(f"field '{spec.name}': op '{op}' takes no value")
        return

    if spec.kind == fields.Kind.TEXT:
        if not isinstance(value, str) or isinstance(value, bool):
            raise ValidationError(f"field '{spec.name}': op '{op}' requires a string value")

    elif spec.kind == fields.Kind.NUMBER:
        if op == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise ValidationError(
                    f"field '{spec.name}': op 'between' requires a 2-item list"
                )
            for v in value:
                _check_number(v, spec.name)
        else:
            _check_number(value, spec.name)

    elif spec.kind == fields.Kind.BOOL:
        if not isinstance(value, bool):
            raise ValidationError(f"field '{spec.name}': expected a boolean value, got {value!r}")

    elif spec.kind == fields.Kind.ENUM:
        if op == "in":
            if not isinstance(value, list) or not value:
                raise ValidationError(
                    f"field '{spec.name}': op 'in' requires a non-empty list"
                )
            for v in value:
                _check_enum_member(spec, v)
        else:
            _check_enum_member(spec, value)

    elif spec.kind == fields.Kind.DATETIME:
        if op == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise ValidationError(
                    f"field '{spec.name}': op 'between' requires a 2-item list"
                )
            for v in value:
                _check_datetime_value(spec, v)
        else:
            _check_datetime_value(spec, value)

    elif spec.kind == fields.Kind.FILE_REF:
        _check_file_ref_value(spec, value)

    else:  # exhaustive over Kind
        raise ValidationError(f"field '{spec.name}': unhandled kind {spec.kind!r}")


# ---------------------------------------------------------------------------
# Top-level validation
# ---------------------------------------------------------------------------

def _check_field_authorization(spec: fields.FieldSpec, ctx: AuthContext) -> None:
    """Enforce role and AI-layer entitlements for one field."""
    if spec.privileged and ctx.role not in PRIVILEGED_ROLES:
        raise ValidationError(
            f"field '{spec.name}': requires a privileged role ({sorted(PRIVILEGED_ROLES)}), "
            f"got {ctx.role!r}"
        )
    if spec.requires_ai and not ctx.ai_enabled:
        raise ValidationError(f"field '{spec.name}': requires the AI layer to be enabled")


def validate(query: Query, ctx: AuthContext) -> ValidatedQuery:
    """Run every semantic, authorization, and complexity check on a
    structurally parsed Query and mint the ValidatedQuery that compiler.py
    requires. Raises ValidationError on the first violation."""
    conditions = list(iter_conditions(query.where))
    if len(conditions) > 32:  # ast.py already enforces this; defensive re-check
        raise ValidationError("query has more than 32 conditions")

    correlated_fields: set = set()
    for cond in conditions:
        spec = fields.get_field(cond.field)
        if spec is None:
            raise ValidationError(f"unknown field {cond.field!r}")
        if cond.op not in spec.ops:
            raise ValidationError(
                f"field '{spec.name}': op {cond.op!r} not supported "
                f"(allowed: {sorted(spec.ops)})"
            )
        _validate_value(spec, cond.op, cond.value)
        _check_field_authorization(spec, ctx)
        if spec.needs_client_uuid and not ctx.client_uuid:
            raise ValidationError(f"field '{spec.name}': requires a client_uuid")
        if spec.correlated:
            correlated_fields.add(spec.name)

    if len(correlated_fields) > MAX_CORRELATED_FIELDS:
        raise ValidationError(
            f"query touches {len(correlated_fields)} distinct EXISTS-style fields "
            f"(max {MAX_CORRELATED_FIELDS}): {sorted(correlated_fields)}"
        )

    for ospec in query.order_by:
        spec = fields.get_field(ospec.field)
        if spec is None:
            raise ValidationError(f"order_by: unknown field {ospec.field!r}")
        if not spec.orderable:
            raise ValidationError(f"order_by: field '{spec.name}' is not orderable")
        _check_field_authorization(spec, ctx)

    limit = query.limit if query.limit is not None else DEFAULT_LIMIT
    if limit > MAX_LIMIT:
        raise ValidationError(f"limit {limit} exceeds the maximum of {MAX_LIMIT}")

    return _new_validated_query(query, limit, ctx)
