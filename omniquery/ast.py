"""Typed SmartGallery query AST.

This module is the contract between the NL parsers, the validator, and the
SQL compiler. Parsers (rules-based or model-backed) produce a JSON object; it
is parsed here into typed nodes with strict structural checks. Field-level
semantics (does the field exist, which operators/values it accepts, who may
query it) are enforced by omniquery.validation — not here.

Structure:

    Query
      target:   "files"                      (only supported target)
      result:   "ids" | "count"
      where:    Node | None
      order_by: [OrderSpec]                  (field + "asc"/"desc")
      limit:    int | None

    Node = Group(op="and"|"or", children=[Node, ...])
         | Not(child=Node)
         | Cond(field=str, op=str, value=JSON scalar/list/dict)

Cond values are JSON-native only (str, int, float, bool, None, list of
scalars, or a small dict for structured values such as relative dates).
Anything else is a structural error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Union

AST_VERSION = 1  # wire-format version; parse_query rejects any other

MAX_DEPTH = 6           # maximum nesting of Group/Not nodes
MAX_CONDITIONS = 32     # maximum number of Cond leaves in one query
MAX_GROUP_CHILDREN = 16  # maximum children in a single and/or group
MAX_IN_VALUES = 64      # maximum number of items in an "in" list
MAX_STR_LEN = 512       # maximum length of a string literal


class ASTError(ValueError):
    """Structural AST parsing/serialization error."""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Cond:
    """Leaf predicate: one field/operator/value comparison. Only structure is
    guaranteed here; whether the field exists and accepts the op/value is
    omniquery.validation's job."""

    field: str  # field name; resolved against the registry by validation
    op: str     # operator name; allowed set depends on the field
    value: Any = None  # JSON scalar, flat scalar list, or small dict; None for value-less ops

    def to_dict(self) -> dict:
        """JSON form parse_query accepts; the value key is omitted when None."""
        d: dict = {"field": self.field, "op": self.op}
        if self.value is not None:
            d["value"] = self.value
        return d


@dataclass(frozen=True)
class Not:
    """Logical negation of a single child node."""

    child: Node

    def to_dict(self) -> dict:
        """JSON form parse_query accepts."""
        return {"op": "not", "child": self.child.to_dict()}


@dataclass(frozen=True)
class Group:
    """N-ary conjunction or disjunction; always has at least one child."""

    op: str  # "and" | "or"
    children: tuple  # tuple[Node, ...], non-empty

    def to_dict(self) -> dict:
        """JSON form parse_query accepts."""
        return {"op": self.op, "children": [c.to_dict() for c in self.children]}


Node = Union[Cond, Not, Group]


@dataclass(frozen=True)
class OrderSpec:
    """One ORDER BY entry; the field must be registered as orderable, which
    validation enforces."""

    field: str
    direction: str = "desc"  # "asc" | "desc"

    def to_dict(self) -> dict:
        """JSON form parse_query accepts (direction serializes as 'dir')."""
        return {"field": self.field, "dir": self.direction}


@dataclass(frozen=True)
class Query:
    """AST root: result shape, optional where tree, ordering, and row limit."""

    target: str = "files"  # only "files" is supported
    result: str = "ids"  # "ids" | "count"
    where: Node | None = None  # None means "match every file"
    order_by: tuple = field(default_factory=tuple)  # tuple[OrderSpec, ...]
    limit: int | None = None  # None defers to validation's DEFAULT_LIMIT
    version: int = AST_VERSION

    def to_dict(self) -> dict:
        """JSON form parse_query accepts; unset optional parts are omitted."""
        d: dict = {"version": self.version, "target": self.target,
                   "result": self.result}
        if self.where is not None:
            d["where"] = self.where.to_dict()
        if self.order_by:
            d["order_by"] = [o.to_dict() for o in self.order_by]
        if self.limit is not None:
            d["limit"] = self.limit
        return d

    def to_json(self) -> str:
        """Compact serialization with sorted keys, so equal queries yield
        byte-identical strings (usable as a cache/comparison key)."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Strict parsing from JSON-compatible dicts
# ---------------------------------------------------------------------------

_SCALAR_TYPES = (str, int, float, bool)  # JSON-native scalar types a Cond value may contain


def _check_scalar(value: Any, ctx: str) -> None:
    """Reject non-JSON literal types and over-long strings; None and all
    numeric/bool scalars pass. ctx names the location for error messages."""
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        if len(value) > MAX_STR_LEN:
            raise ASTError(f"{ctx}: string literal too long (>{MAX_STR_LEN})")
        return
    if isinstance(value, (int, float)):
        return
    raise ASTError(f"{ctx}: unsupported literal type {type(value).__name__}")


def _parse_value(value: Any, ctx: str) -> Any:
    """Accept JSON scalars, flat lists of scalars, or small string-keyed
    dicts of scalars (used for structured values like relative dates)."""
    if value is None or isinstance(value, _SCALAR_TYPES):
        _check_scalar(value, ctx)
        return value
    if isinstance(value, list):
        if len(value) > MAX_IN_VALUES:
            raise ASTError(f"{ctx}: list has more than {MAX_IN_VALUES} items")
        for i, item in enumerate(value):
            if not (item is None or isinstance(item, _SCALAR_TYPES)):
                raise ASTError(f"{ctx}[{i}]: lists may only contain scalars")
            _check_scalar(item, f"{ctx}[{i}]")
        return list(value)
    if isinstance(value, dict):
        if len(value) > 8:
            raise ASTError(f"{ctx}: dict value has too many keys")
        for k, v in value.items():
            if not isinstance(k, str):
                raise ASTError(f"{ctx}: dict keys must be strings")
            if not (v is None or isinstance(v, _SCALAR_TYPES)):
                raise ASTError(f"{ctx}.{k}: dict values must be scalars")
            _check_scalar(v, f"{ctx}.{k}")
        return dict(value)
    raise ASTError(f"{ctx}: unsupported value type {type(value).__name__}")


def _parse_node(obj: Any, depth: int, counter: dict, ctx: str) -> Node:
    """Recursively parse one where-tree node, enforcing the depth, group-size,
    and whole-query condition caps. counter["conds"] is the Cond-leaf tally
    shared across the entire recursive parse."""
    if depth > MAX_DEPTH:
        raise ASTError(f"{ctx}: nesting deeper than {MAX_DEPTH}")
    if not isinstance(obj, dict):
        raise ASTError(f"{ctx}: node must be an object")

    op = obj.get("op")

    if op in ("and", "or"):
        _require_keys(obj, {"op", "children"}, ctx)
        children = obj.get("children")
        if not isinstance(children, list) or not children:
            raise ASTError(f"{ctx}: '{op}' requires a non-empty children list")
        if len(children) > MAX_GROUP_CHILDREN:
            raise ASTError(f"{ctx}: more than {MAX_GROUP_CHILDREN} children")
        parsed = tuple(
            _parse_node(c, depth + 1, counter, f"{ctx}.children[{i}]")
            for i, c in enumerate(children)
        )
        return Group(op=op, children=parsed)

    if op == "not":
        _require_keys(obj, {"op", "child"}, ctx)
        if "child" not in obj:
            raise ASTError(f"{ctx}: 'not' requires a child")
        return Not(child=_parse_node(obj["child"], depth + 1, counter, f"{ctx}.child"))

    # Condition leaf
    if "field" not in obj:
        raise ASTError(f"{ctx}: expected a condition with 'field' or a group op")
    _require_keys(obj, {"field", "op", "value"}, ctx)
    fname = obj["field"]
    if not isinstance(fname, str) or not fname:
        raise ASTError(f"{ctx}: 'field' must be a non-empty string")
    if not isinstance(op, str) or not op:
        raise ASTError(f"{ctx}: 'op' must be a non-empty string")
    counter["conds"] += 1
    if counter["conds"] > MAX_CONDITIONS:
        raise ASTError(f"query has more than {MAX_CONDITIONS} conditions")
    value = _parse_value(obj.get("value"), f"{ctx}.value")
    return Cond(field=fname, op=op, value=value)


def _require_keys(obj: dict, allowed: set, ctx: str) -> None:
    """Reject any key outside the allowed set: the schema is closed, so
    parser extensions cannot smuggle extra data past validation."""
    unknown = set(obj.keys()) - allowed
    if unknown:
        raise ASTError(f"{ctx}: unknown keys {sorted(unknown)}")


def parse_query(obj: Any) -> Query:
    """Parse a JSON-compatible dict into a Query. Raises ASTError on any
    structural problem. Unknown keys are rejected everywhere: parsers must
    emit exactly this schema."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except json.JSONDecodeError as exc:
            raise ASTError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ASTError("query must be a JSON object")

    _require_keys(obj, {"version", "target", "result", "where", "order_by",
                        "limit"}, "query")

    version = obj.get("version", AST_VERSION)
    if version != AST_VERSION:
        raise ASTError(f"unsupported AST version {version!r}")

    target = obj.get("target", "files")
    if target != "files":
        raise ASTError(f"unsupported target {target!r}")

    result = obj.get("result", "ids")
    if result not in ("ids", "count"):
        raise ASTError(f"unsupported result kind {result!r}")

    where_obj = obj.get("where")
    counter = {"conds": 0}  # Cond-leaf tally, shared across the recursive parse
    where = None
    if where_obj is not None:
        where = _parse_node(where_obj, 1, counter, "where")

    order_by: list[OrderSpec] = []
    ob = obj.get("order_by", [])
    if ob is not None:
        if not isinstance(ob, list) or len(ob) > 3:
            raise ASTError("order_by must be a list of at most 3 entries")
        for i, spec in enumerate(ob):
            if not isinstance(spec, dict):
                raise ASTError(f"order_by[{i}] must be an object")
            _require_keys(spec, {"field", "dir"}, f"order_by[{i}]")
            f_name = spec.get("field")
            if not isinstance(f_name, str) or not f_name:
                raise ASTError(f"order_by[{i}].field must be a string")
            direction = spec.get("dir", "desc")
            if direction not in ("asc", "desc"):
                raise ASTError(f"order_by[{i}].dir must be 'asc' or 'desc'")
            order_by.append(OrderSpec(field=f_name, direction=direction))

    limit = obj.get("limit")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ASTError("limit must be an integer")
        if limit < 1:
            raise ASTError("limit must be >= 1")

    return Query(target=target, result=result, where=where,
                 order_by=tuple(order_by), limit=limit, version=version)


def iter_conditions(node: Node | None):
    """Yield every Cond leaf under a node (depth-first)."""
    if node is None:
        return
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, Cond):
            yield cur
        elif isinstance(cur, Not):
            stack.append(cur.child)
        elif isinstance(cur, Group):
            stack.extend(cur.children)


def canonicalize(query: Query) -> dict:
    """Stable, comparison-friendly form used by the benchmark harness:
    and/or children are sorted by their serialized form so semantically
    identical queries with reordered conjuncts compare equal."""

    def canon_node(node: Node) -> dict:
        if isinstance(node, Cond):
            return node.to_dict()
        if isinstance(node, Not):
            return {"op": "not", "child": canon_node(node.child)}
        children = sorted(
            (canon_node(c) for c in node.children),
            key=lambda d: json.dumps(d, sort_keys=True),
        )
        return {"op": node.op, "children": children}

    d = query.to_dict()
    if query.where is not None:
        d["where"] = canon_node(query.where)
    return d


# ---------------------------------------------------------------------------
# JSON Schema (for constrained decoding / tool-call definitions)
# ---------------------------------------------------------------------------

def json_schema(field_names: list[str] | None = None,
                operator_names: list[str] | None = None) -> dict:
    """JSON Schema for the AST. When field/operator vocabularies are given
    (from omniquery.fields), they are embedded as enums so constrained
    decoders can only emit valid vocabulary.

    `value` mirrors omniquery.validation's accepted shapes: scalars, arrays
    of scalars, {days_ago}/{hours_ago} relative dates, and {file_id, k}
    similarity refs. Anything looser lets grammar-constrained decoders emit
    object-wrapped scalars ({"number": 42}) that validation then rejects."""
    cond: dict = {
        "type": "object",
        "properties": {
            "field": {"type": "string"},
            "op": {"type": "string"},
            "value": {"$ref": "#/$defs/value"},
        },
        "required": ["field", "op"],
        "additionalProperties": False,
    }
    if field_names:
        cond["properties"]["field"]["enum"] = sorted(field_names)
    if operator_names:
        cond["properties"]["op"]["enum"] = sorted(operator_names)

    node = {
        "anyOf": [
            {"$ref": "#/$defs/group"},
            {"$ref": "#/$defs/not"},
            {"$ref": "#/$defs/cond"},
        ]
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "version": {"const": AST_VERSION},
            "target": {"const": "files"},
            "result": {"enum": ["ids", "count"]},
            "where": node,
            "order_by": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "dir": {"enum": ["asc", "desc"]},
                    },
                    "required": ["field"],
                    "additionalProperties": False,
                },
            },
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["target"],
        "additionalProperties": False,
        "$defs": {
            "scalar": {"type": ["string", "number", "boolean"]},
            "value": {
                "anyOf": [
                    {"$ref": "#/$defs/scalar"},
                    {
                        "type": "array",
                        "minItems": 1,
                        "items": {"$ref": "#/$defs/scalar"},
                    },
                    {
                        "type": "object",
                        "properties": {"days_ago": {"type": "number", "minimum": 0}},
                        "required": ["days_ago"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"hours_ago": {"type": "number", "minimum": 0}},
                        "required": ["hours_ago"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "file_id": {"type": "string"},
                            "k": {"type": "integer", "minimum": 1},
                        },
                        "required": ["file_id"],
                        "additionalProperties": False,
                    },
                ]
            },
            "cond": cond,
            "not": {
                "type": "object",
                "properties": {"op": {"const": "not"}, "child": node},
                "required": ["op", "child"],
                "additionalProperties": False,
            },
            "group": {
                "type": "object",
                "properties": {
                    "op": {"enum": ["and", "or"]},
                    "children": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_GROUP_CHILDREN,
                        "items": node,
                    },
                },
                "required": ["op", "children"],
                "additionalProperties": False,
            },
        },
    }
