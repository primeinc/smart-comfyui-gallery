"""Declarative field registry for OmniQuery v2.

Each :class:`FieldSpec` describes one queryable field: its value kind, the
operators it accepts, whether it may be used in ``ORDER BY``, whether it
requires a privileged role or the AI layer to be enabled, and which SQL
generation *strategy* the compiler should use for it.

FieldSpec instances hold only static, trusted metadata -- column names,
table names, SQL expression templates authored by us. They never contain
user-supplied values. Turning that metadata plus a validated ``Cond`` into
parameterized SQL text is exclusively compiler.py's job; this module never
builds a SQL string that includes a bound value.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, List, Optional


class Kind(str, Enum):
    TEXT = "text"
    NUMBER = "number"
    BOOL = "bool"
    ENUM = "enum"
    DATETIME = "datetime"
    FILE_REF = "file_ref"


class Strategy(str, Enum):
    """Which SQL shape compiler.py builds for a field's conditions."""

    COLUMN = "column"                    # direct comparison on a files column
    EXPR = "expr"                        # comparison on a computed SQL expression
    FOLDER = "folder"                    # path-prefix match relative to base_path
    SUBQUERY_SCALAR = "subquery_scalar"  # correlated scalar subquery
    MY_RATING = "my_rating"              # scalar subquery keyed by client_uuid
    EXISTS_TEXT = "exists_text"          # EXISTS ... inner text column LIKE/eq match
    EXISTS_NAMED = "exists_named"        # EXISTS via collection_files/collections
    EXISTS_USER = "exists_user"          # EXISTS ... JOIN users on client_uuid
    EXISTS_BOOL = "exists_bool"          # EXISTS with no inner value comparison
    EXISTS_NUMERIC = "exists_numeric"    # EXISTS ... inner numeric column match
    EXISTS_ENUM = "exists_enum"          # EXISTS ... inner enum/text column match
    FILE_REF = "file_ref"                # resolved externally to an id list


_TEXT_FULL = frozenset({"eq", "ne", "contains", "prefix", "suffix"})
_TEXT_CONTAINS = frozenset({"contains"})
_NUMBER_CMP = frozenset({"eq", "ne", "lt", "le", "gt", "ge", "between"})
_ENUM_CMP = frozenset({"eq", "ne", "in"})
_DATETIME_CMP = frozenset({"eq", "ne", "lt", "le", "gt", "ge", "between"})

# --- trusted SQL expression templates (static; never carry user data) ------
#
# duration is stored as 'MM:SS' text, or 'H:MM:SS' for media over an hour
# (core format_duration emits both); guarded against empty/no-colon values.
# Colon count distinguishes the two forms.
DURATION_SECONDS_EXPR = (
    "(CASE "
    "WHEN f.duration IS NULL OR f.duration = '' "
    "OR instr(f.duration, ':') = 0 THEN NULL "
    "WHEN length(f.duration) - length(replace(f.duration, ':', '')) = 2 THEN "
    "CAST(substr(f.duration, 1, instr(f.duration, ':') - 1) AS INTEGER) * 3600 "
    "+ CAST(substr(substr(f.duration, instr(f.duration, ':') + 1), 1, "
    "instr(substr(f.duration, instr(f.duration, ':') + 1), ':') - 1) AS INTEGER) * 60 "
    "+ CAST(substr(substr(f.duration, instr(f.duration, ':') + 1), "
    "instr(substr(f.duration, instr(f.duration, ':') + 1), ':') + 1) AS INTEGER) "
    "ELSE CAST(substr(f.duration, 1, instr(f.duration, ':') - 1) AS INTEGER) * 60 "
    "+ CAST(substr(f.duration, instr(f.duration, ':') + 1) AS INTEGER) END)"
)
# dimensions are stored as 'WxH' text; guarded against empty/no-'x' values.
WIDTH_EXPR = (
    "(CASE WHEN f.dimensions IS NULL OR f.dimensions = '' "
    "OR instr(f.dimensions, 'x') = 0 THEN NULL "
    "ELSE CAST(substr(f.dimensions, 1, instr(f.dimensions, 'x') - 1) AS INTEGER) END)"
)
HEIGHT_EXPR = (
    "(CASE WHEN f.dimensions IS NULL OR f.dimensions = '' "
    "OR instr(f.dimensions, 'x') = 0 THEN NULL "
    "ELSE CAST(substr(f.dimensions, instr(f.dimensions, 'x') + 1) AS INTEGER) END)"
)
# NULL propagates through arithmetic, so no extra guard is needed here.
MEGAPIXELS_EXPR = f"(({WIDTH_EXPR}) * ({HEIGHT_EXPR}) / 1000000.0)"
# Binary megabytes (1024*1024 bytes), matching common file-manager display.
SIZE_MB_EXPR = "(f.size / 1048576.0)"

RATING_AVG_EXPR = "(SELECT AVG(r.rating) FROM file_ratings r WHERE r.file_id = f.id)"
RATING_COUNT_EXPR = "(SELECT COUNT(*) FROM file_ratings r WHERE r.file_id = f.id)"
COMMENT_COUNT_EXPR = "(SELECT COUNT(*) FROM file_comments c WHERE c.file_id = f.id)"
FACE_COUNT_EXPR = "(SELECT COUNT(*) FROM ai_face_instances fa WHERE fa.file_id = f.id)"
# "latest" review = most recent by computed_at, per (file_id, rubric, model).
REVIEW_QUALITY_EXPR = (
    "(SELECT rv.quality_score FROM ai_reviews rv WHERE rv.file_id = f.id "
    "ORDER BY rv.computed_at DESC LIMIT 1)"
)
REVIEW_ALIGNMENT_EXPR = (
    "(SELECT rv.prompt_alignment_score FROM ai_reviews rv WHERE rv.file_id = f.id "
    "ORDER BY rv.computed_at DESC LIMIT 1)"
)

STATUS_FLAG_VALUES: FrozenSet[str] = frozenset(
    {"Approved", "Review", "To Edit", "Rejected", "Select"}
)
FILE_TYPE_VALUES: FrozenSet[str] = frozenset(
    {"image", "video", "animated_image", "audio", "document"}
)
REVIEW_ISSUE_VALUES: FrozenSet[str] = frozenset({
    "anatomy", "artifact", "composition", "lighting", "text_render",
    "prompt_mismatch", "style", "detail_loss", "other",
})


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: Kind
    ops: FrozenSet[str]
    strategy: Strategy
    orderable: bool = False
    privileged: bool = False
    requires_ai: bool = False
    # Counts toward validation's "distinct EXISTS-style fields" complexity cap:
    # true for every field whose SQL strategy adds a join or (correlated)
    # subquery rather than a plain column/expression comparison.
    correlated: bool = False
    needs_client_uuid: bool = False
    enum_values: Optional[FrozenSet[str]] = None
    column: Optional[str] = None        # Strategy.COLUMN
    expr: Optional[str] = None          # Strategy.EXPR / SUBQUERY_SCALAR
    order_expr: Optional[str] = None    # only set when orderable
    table: Optional[str] = None         # EXISTS_TEXT / EXISTS_USER inner table
    alias: Optional[str] = None         # inner table alias
    inner_column: Optional[str] = None  # EXISTS_TEXT inner column
    type_filter: Optional[str] = None   # EXISTS_NAMED collections.type constant


def _f(**kwargs) -> FieldSpec:
    return FieldSpec(**kwargs)


_SPECS: List[FieldSpec] = [
    _f(name="name", kind=Kind.TEXT, ops=_TEXT_FULL, strategy=Strategy.COLUMN,
       column="f.name", orderable=True, order_expr="f.name"),
    _f(name="path", kind=Kind.TEXT, ops=_TEXT_FULL, strategy=Strategy.COLUMN,
       column="f.path"),
    _f(name="folder", kind=Kind.TEXT, ops=frozenset({"eq", "contains"}),
       strategy=Strategy.FOLDER),
    _f(name="type", kind=Kind.ENUM, ops=_ENUM_CMP, strategy=Strategy.COLUMN,
       column="f.type", enum_values=FILE_TYPE_VALUES),
    _f(name="size_bytes", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.COLUMN,
       column="f.size", orderable=True, order_expr="f.size"),
    _f(name="size_mb", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.EXPR,
       expr=SIZE_MB_EXPR),
    _f(name="mtime", kind=Kind.DATETIME, ops=_DATETIME_CMP, strategy=Strategy.COLUMN,
       column="f.mtime", orderable=True, order_expr="f.mtime"),
    _f(name="duration_seconds", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.EXPR,
       expr=DURATION_SECONDS_EXPR, orderable=True, order_expr=DURATION_SECONDS_EXPR),
    _f(name="width", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.EXPR, expr=WIDTH_EXPR),
    _f(name="height", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.EXPR, expr=HEIGHT_EXPR),
    _f(name="megapixels", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.EXPR,
       expr=MEGAPIXELS_EXPR),
    _f(name="is_favorite", kind=Kind.BOOL, ops=frozenset({"eq"}), strategy=Strategy.COLUMN,
       column="f.is_favorite"),
    _f(name="has_workflow", kind=Kind.BOOL, ops=frozenset({"eq"}), strategy=Strategy.COLUMN,
       column="f.has_workflow"),
    _f(name="workflow_prompt", kind=Kind.TEXT, ops=_TEXT_CONTAINS, strategy=Strategy.COLUMN,
       column="f.workflow_prompt"),
    _f(name="workflow_files", kind=Kind.TEXT, ops=_TEXT_CONTAINS, strategy=Strategy.COLUMN,
       column="f.workflow_files"),
    _f(name="ai_caption", kind=Kind.TEXT, ops=frozenset({"contains", "is_null", "not_null"}),
       strategy=Strategy.COLUMN, column="f.ai_caption"),
    _f(name="rating_avg", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.SUBQUERY_SCALAR,
       expr=RATING_AVG_EXPR, orderable=True, order_expr=RATING_AVG_EXPR, correlated=True),
    _f(name="rating_count", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.SUBQUERY_SCALAR,
       expr=RATING_COUNT_EXPR, correlated=True),
    _f(name="my_rating", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.MY_RATING,
       correlated=True, needs_client_uuid=True),
    _f(name="comment_count", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.SUBQUERY_SCALAR,
       expr=COMMENT_COUNT_EXPR, correlated=True),
    _f(name="comment_contains", kind=Kind.TEXT, ops=_TEXT_CONTAINS, strategy=Strategy.EXISTS_TEXT,
       table="file_comments", alias="c", inner_column="c.comment_text", correlated=True),
    _f(name="collection", kind=Kind.TEXT, ops=frozenset({"eq", "contains"}),
       strategy=Strategy.EXISTS_NAMED, type_filter="user_album", correlated=True),
    _f(name="status_flag", kind=Kind.ENUM, ops=_ENUM_CMP, strategy=Strategy.EXISTS_NAMED,
       type_filter="system_flag", enum_values=STATUS_FLAG_VALUES, correlated=True),
    _f(name="rated_by_user", kind=Kind.TEXT, ops=frozenset({"eq"}), strategy=Strategy.EXISTS_USER,
       table="file_ratings", alias="r", privileged=True, correlated=True),
    _f(name="commented_by_user", kind=Kind.TEXT, ops=frozenset({"eq"}), strategy=Strategy.EXISTS_USER,
       table="file_comments", alias="c", privileged=True, correlated=True),
    _f(name="has_faces", kind=Kind.BOOL, ops=frozenset({"eq"}), strategy=Strategy.EXISTS_BOOL,
       requires_ai=True, correlated=True),
    _f(name="face_count", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.SUBQUERY_SCALAR,
       expr=FACE_COUNT_EXPR, requires_ai=True, correlated=True),
    _f(name="face_cluster", kind=Kind.NUMBER, ops=frozenset({"eq"}), strategy=Strategy.EXISTS_NUMERIC,
       requires_ai=True, correlated=True),
    _f(name="review_quality", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.SUBQUERY_SCALAR,
       expr=REVIEW_QUALITY_EXPR, orderable=True, order_expr=REVIEW_QUALITY_EXPR,
       requires_ai=True, correlated=True),
    _f(name="review_alignment", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.SUBQUERY_SCALAR,
       expr=REVIEW_ALIGNMENT_EXPR, requires_ai=True, correlated=True),
    _f(name="review_issue", kind=Kind.ENUM, ops=_ENUM_CMP, strategy=Strategy.EXISTS_ENUM,
       enum_values=REVIEW_ISSUE_VALUES, requires_ai=True, correlated=True),
    _f(name="near_dup_of", kind=Kind.FILE_REF, ops=frozenset({"eq"}), strategy=Strategy.FILE_REF,
       requires_ai=True, correlated=True),
    _f(name="similar_to_semantic", kind=Kind.FILE_REF, ops=frozenset({"eq"}),
       strategy=Strategy.FILE_REF, requires_ai=True, correlated=True),
    _f(name="similar_to_visual", kind=Kind.FILE_REF, ops=frozenset({"eq"}),
       strategy=Strategy.FILE_REF, requires_ai=True, correlated=True),
]

FIELDS: Dict[str, FieldSpec] = {spec.name: spec for spec in _SPECS}

ORDERABLE_FIELDS: FrozenSet[str] = frozenset(n for n, s in FIELDS.items() if s.orderable)
PRIVILEGED_FIELDS: FrozenSet[str] = frozenset(n for n, s in FIELDS.items() if s.privileged)
AI_FIELDS: FrozenSet[str] = frozenset(n for n, s in FIELDS.items() if s.requires_ai)
CORRELATED_FIELDS: FrozenSet[str] = frozenset(n for n, s in FIELDS.items() if s.correlated)


def get_field(name: str) -> Optional[FieldSpec]:
    return FIELDS.get(name)


def field_names() -> List[str]:
    return sorted(FIELDS)


def all_ops() -> List[str]:
    """Every operator name used by any field (for json_schema embedding)."""
    ops: set = set()
    for spec in FIELDS.values():
        ops |= spec.ops
    return sorted(ops)
