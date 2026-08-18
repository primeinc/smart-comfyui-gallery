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


class Kind(str, Enum):
    """Value domain of a field; selects which value checks and SQL builders apply."""

    TEXT = "text"
    NUMBER = "number"
    BOOL = "bool"
    ENUM = "enum"          # string drawn from the spec's enum_values
    DATETIME = "datetime"  # epoch-seconds column; literals are ISO strings or relative offsets
    FILE_REF = "file_ref"  # value names another file; resolved to an id list outside SQL


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
    ANY_TEXT = "any_text"                # OR over every text-bearing surface


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

# Correlated scalar subqueries over per-file child tables. AVG and the
# review-score forms yield NULL for a file with no matching rows (so such
# files fail every comparison); the COUNT forms yield 0.
RATING_AVG_EXPR = "(SELECT AVG(r.rating) FROM file_ratings r WHERE r.file_id = f.id)"
RATING_COUNT_EXPR = "(SELECT COUNT(*) FROM file_ratings r WHERE r.file_id = f.id)"
COMMENT_COUNT_EXPR = "(SELECT COUNT(*) FROM file_comments c WHERE c.file_id = f.id)"
FACE_COUNT_EXPR = "(SELECT COUNT(*) FROM ai_face_instances fa WHERE fa.file_id = f.id)"
# "latest" review = the file's single most recent ai_reviews row by computed_at.
REVIEW_QUALITY_EXPR = (
    "(SELECT rv.quality_score FROM ai_reviews rv WHERE rv.file_id = f.id "
    "ORDER BY rv.computed_at DESC LIMIT 1)"
)
REVIEW_ALIGNMENT_EXPR = (
    "(SELECT rv.prompt_alignment_score FROM ai_reviews rv WHERE rv.file_id = f.id "
    "ORDER BY rv.computed_at DESC LIMIT 1)"
)

# collections.name values for the built-in type='system_flag' collections.
STATUS_FLAG_VALUES: frozenset[str] = frozenset(
    {"Approved", "Review", "To Edit", "Rejected", "Select"}
)
# Accepted values of the files.type column.
FILE_TYPE_VALUES: frozenset[str] = frozenset(
    {"image", "video", "animated_image", "audio", "document"}
)
# Accepted values of the ai_review_findings.type column.
REVIEW_ISSUE_VALUES: frozenset[str] = frozenset({
    "anatomy", "artifact", "composition", "lighting", "text_render",
    "prompt_mismatch", "style", "detail_loss", "other",
})


@dataclass(frozen=True)
class FieldSpec:
    """Static, trusted description of one queryable field: what values and
    operators it accepts, who may use it, and how the compiler renders it."""

    name: str            # field name as parsers emit it
    kind: Kind
    ops: frozenset[str]  # operator names validation accepts for this field
    strategy: Strategy
    orderable: bool = False   # usable in ORDER BY (order_expr must then be set)
    privileged: bool = False  # restricted to validation.PRIVILEGED_ROLES
    requires_ai: bool = False  # only valid when the AI layer is enabled
    # Counts toward validation's "distinct EXISTS-style fields" complexity cap:
    # true for every field whose SQL strategy adds a join or (correlated)
    # subquery rather than a plain column/expression comparison.
    correlated: bool = False
    needs_client_uuid: bool = False  # rejected unless the AuthContext carries a client_uuid
    enum_values: frozenset[str] | None = None  # allowed literals for Kind.ENUM fields
    column: str | None = None        # Strategy.COLUMN
    expr: str | None = None          # Strategy.EXPR / SUBQUERY_SCALAR
    order_expr: str | None = None    # only set when orderable
    table: str | None = None         # EXISTS_TEXT / EXISTS_USER inner table
    alias: str | None = None         # inner table alias
    inner_column: str | None = None  # EXISTS_TEXT inner column
    type_filter: str | None = None   # EXISTS_NAMED collections.type constant


def _f(**kwargs) -> FieldSpec:
    """Keyword shorthand keeping the _SPECS table compact."""
    return FieldSpec(**kwargs)


# Text surfaces the universal `text` field fans out over: direct files
# columns, plus generation_params columns probed through one EXISTS.
# Deliberately excludes gen_negative_prompt: a term someone searches FOR
# must not match files that were generated explicitly WITHOUT it.
ANY_TEXT_COLUMNS: tuple = ("f.name", "f.path", "f.workflow_prompt", "f.ai_caption")
ANY_TEXT_GP_COLUMNS: tuple = ("gp.positive_prompt", "gp.model", "gp.loras")

# Registry source of truth: one spec per queryable field.
_SPECS: list[FieldSpec] = [
    # Universal free-text search: one term matched (contains) against every
    # text surface at once -- filename, path, workflow prompt, AI caption,
    # generation prompt, model name, LoRA names. This is the field bare
    # search-box terms compile to; a query like "girlnextdoor" must always
    # find the files whose prompt/LoRA carries it.
    _f(name="text", kind=Kind.TEXT, ops=_TEXT_CONTAINS, strategy=Strategy.ANY_TEXT,
       correlated=True),
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
    # Per-face scalar analytics (typed columns on ai_face_instances,
    # written by the insightface pipeline's genderage / 3D-pose heads).
    _f(name="face_sex", kind=Kind.TEXT, ops=frozenset({"eq"}),
       strategy=Strategy.EXISTS_TEXT, table="ai_face_instances", alias="fx",
       inner_column="fx.sex", requires_ai=True, correlated=True),
    _f(name="face_age_min", kind=Kind.NUMBER, ops=_NUMBER_CMP,
       strategy=Strategy.SUBQUERY_SCALAR,
       expr="(SELECT MIN(fa.age) FROM ai_face_instances fa WHERE fa.file_id = f.id)",
       requires_ai=True, correlated=True),
    _f(name="face_age_max", kind=Kind.NUMBER, ops=_NUMBER_CMP,
       strategy=Strategy.SUBQUERY_SCALAR,
       expr="(SELECT MAX(fa.age) FROM ai_face_instances fa WHERE fa.file_id = f.id)",
       requires_ai=True, correlated=True),
    _f(name="face_yaw_abs_max", kind=Kind.NUMBER, ops=_NUMBER_CMP,
       strategy=Strategy.SUBQUERY_SCALAR,
       expr="(SELECT MAX(ABS(fa.pose_yaw)) FROM ai_face_instances fa WHERE fa.file_id = f.id)",
       requires_ai=True, correlated=True),
    _f(name="review_quality", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.SUBQUERY_SCALAR,
       expr=REVIEW_QUALITY_EXPR, orderable=True, order_expr=REVIEW_QUALITY_EXPR,
       requires_ai=True, correlated=True),
    _f(name="review_alignment", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.SUBQUERY_SCALAR,
       expr=REVIEW_ALIGNMENT_EXPR, requires_ai=True, correlated=True),
    _f(name="review_issue", kind=Kind.ENUM, ops=_ENUM_CMP, strategy=Strategy.EXISTS_ENUM,
       enum_values=REVIEW_ISSUE_VALUES, requires_ai=True, correlated=True),
    # First-class typed generation parameters (generation_params table,
    # one row per file, written by the indexer from metaparse.typed).
    _f(name="gen_tool", kind=Kind.TEXT, ops=frozenset({"eq", "contains"}),
       strategy=Strategy.EXISTS_TEXT, table="generation_params", alias="gp",
       inner_column="gp.tool", correlated=True),
    _f(name="gen_model", kind=Kind.TEXT, ops=frozenset({"eq", "contains"}),
       strategy=Strategy.EXISTS_TEXT, table="generation_params", alias="gp",
       inner_column="gp.model", correlated=True),
    _f(name="gen_sampler", kind=Kind.TEXT, ops=frozenset({"eq", "contains"}),
       strategy=Strategy.EXISTS_TEXT, table="generation_params", alias="gp",
       inner_column="gp.sampler", correlated=True),
    _f(name="gen_scheduler", kind=Kind.TEXT, ops=frozenset({"eq", "contains"}),
       strategy=Strategy.EXISTS_TEXT, table="generation_params", alias="gp",
       inner_column="gp.scheduler", correlated=True),
    _f(name="gen_lora", kind=Kind.TEXT, ops=_TEXT_CONTAINS,
       strategy=Strategy.EXISTS_TEXT, table="generation_params", alias="gp",
       inner_column="gp.loras", correlated=True),
    _f(name="gen_positive_prompt", kind=Kind.TEXT, ops=_TEXT_CONTAINS,
       strategy=Strategy.EXISTS_TEXT, table="generation_params", alias="gp",
       inner_column="gp.positive_prompt", correlated=True),
    _f(name="gen_negative_prompt", kind=Kind.TEXT, ops=_TEXT_CONTAINS,
       strategy=Strategy.EXISTS_TEXT, table="generation_params", alias="gp",
       inner_column="gp.negative_prompt", correlated=True),
    _f(name="gen_seed", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.SUBQUERY_SCALAR,
       expr="(SELECT gp.seed FROM generation_params gp WHERE gp.file_id = f.id)",
       correlated=True),
    _f(name="gen_steps", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.SUBQUERY_SCALAR,
       expr="(SELECT gp.steps FROM generation_params gp WHERE gp.file_id = f.id)",
       correlated=True),
    _f(name="gen_cfg", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.SUBQUERY_SCALAR,
       expr="(SELECT gp.cfg FROM generation_params gp WHERE gp.file_id = f.id)",
       correlated=True),
    _f(name="gen_denoise", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.SUBQUERY_SCALAR,
       expr="(SELECT gp.denoise FROM generation_params gp WHERE gp.file_id = f.id)",
       correlated=True),
    _f(name="gen_clip_skip", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.SUBQUERY_SCALAR,
       expr="(SELECT gp.clip_skip FROM generation_params gp WHERE gp.file_id = f.id)",
       correlated=True),
    _f(name="gen_width", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.SUBQUERY_SCALAR,
       expr="(SELECT gp.width FROM generation_params gp WHERE gp.file_id = f.id)",
       correlated=True),
    _f(name="gen_height", kind=Kind.NUMBER, ops=_NUMBER_CMP, strategy=Strategy.SUBQUERY_SCALAR,
       expr="(SELECT gp.height FROM generation_params gp WHERE gp.file_id = f.id)",
       correlated=True),
    _f(name="near_dup_of", kind=Kind.FILE_REF, ops=frozenset({"eq"}), strategy=Strategy.FILE_REF,
       requires_ai=True, correlated=True),
    _f(name="similar_to_semantic", kind=Kind.FILE_REF, ops=frozenset({"eq"}),
       strategy=Strategy.FILE_REF, requires_ai=True, correlated=True),
    _f(name="similar_to_visual", kind=Kind.FILE_REF, ops=frozenset({"eq"}),
       strategy=Strategy.FILE_REF, requires_ai=True, correlated=True),
]

FIELDS: dict[str, FieldSpec] = {spec.name: spec for spec in _SPECS}  # name -> spec

ORDERABLE_FIELDS: frozenset[str] = frozenset(n for n, s in FIELDS.items() if s.orderable)
PRIVILEGED_FIELDS: frozenset[str] = frozenset(n for n, s in FIELDS.items() if s.privileged)
AI_FIELDS: frozenset[str] = frozenset(n for n, s in FIELDS.items() if s.requires_ai)
CORRELATED_FIELDS: frozenset[str] = frozenset(n for n, s in FIELDS.items() if s.correlated)


def get_field(name: str) -> FieldSpec | None:
    """Spec for a field name, or None (not an exception) for unknown names,
    so callers can phrase their own error."""
    return FIELDS.get(name)


def field_names() -> list[str]:
    """Sorted field vocabulary, for schema embedding and error messages."""
    return sorted(FIELDS)


def all_ops() -> list[str]:
    """Every operator name used by any field (for json_schema embedding)."""
    ops: set = set()
    for spec in FIELDS.values():
        ops |= spec.ops
    return sorted(ops)
