"""Tests for the OmniQuery v2 AST contract (omniquery/ast.py).

ast.py is a read-only contract file: these tests exercise it, they don't
change it. The point is to pin down the guarantees fields.py, validation.py,
compiler.py, and engine.py all build on.
"""

from __future__ import annotations

import pytest

from omniquery import fields
from omniquery.ast import (
    ASTError, Cond, Group, Not, OrderSpec, canonicalize,
    iter_conditions, json_schema, parse_query,
)


# ---------------------------------------------------------------------------
# Round-trips
# ---------------------------------------------------------------------------

def test_round_trip_simple_cond():
    obj = {"target": "files", "result": "ids",
           "where": {"field": "type", "op": "eq", "value": "image"}}
    q = parse_query(obj)
    assert q.where == Cond(field="type", op="eq", value="image")
    assert parse_query(q.to_dict()).to_dict() == q.to_dict()


def test_round_trip_nested_group_and_not():
    obj = {
        "target": "files", "result": "ids",
        "where": {
            "op": "and",
            "children": [
                {"field": "is_favorite", "op": "eq", "value": True},
                {"op": "not", "child": {"field": "type", "op": "eq", "value": "document"}},
                {"op": "or", "children": [
                    {"field": "name", "op": "contains", "value": "sunset"},
                    {"field": "name", "op": "contains", "value": "dawn"},
                ]},
            ],
        },
        "order_by": [{"field": "mtime", "dir": "desc"}],
        "limit": 50,
    }
    q = parse_query(obj)
    assert isinstance(q.where, Group)
    assert q.where.op == "and"
    assert len(q.where.children) == 3
    assert isinstance(q.where.children[1], Not)
    assert q.order_by == (OrderSpec(field="mtime", direction="desc"),)
    assert q.limit == 50
    # Round-trip through to_dict/parse_query is lossless.
    assert parse_query(q.to_dict()).to_dict() == obj | {"version": 1}


def test_round_trip_via_json_string():
    obj = {"target": "files", "result": "count"}
    q1 = parse_query(obj)
    q2 = parse_query(q1.to_json())
    assert q1.to_dict() == q2.to_dict()


def test_defaults_when_optional_keys_absent():
    q = parse_query({})
    assert q.target == "files"
    assert q.result == "ids"
    assert q.where is None
    assert q.order_by == ()
    assert q.limit is None


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("obj", [
    {"target": "albums"},
    {"result": "rows"},
    {"version": 2},
    {"bogus_key": 1},
    {"where": {"field": "type", "op": "eq", "value": "image", "extra": 1}},
    {"where": {"op": "and", "children": []}},
    {"where": {"op": "and"}},
    {"where": {"op": "not"}},
    {"where": 5},
    {"order_by": [{"field": "mtime", "dir": "sideways"}]},
    {"order_by": "mtime"},
    {"limit": 0},
    {"limit": -1},
    {"limit": 1.5},
    {"limit": True},
])
def test_rejects_structurally_invalid_query(obj):
    with pytest.raises(ASTError):
        parse_query(obj)


def test_rejects_condition_missing_field():
    with pytest.raises(ASTError):
        parse_query({"where": {"op": "eq", "value": "image"}})


def test_rejects_too_many_conditions():
    children = [{"field": "name", "op": "eq", "value": str(i)} for i in range(40)]
    # 16 is MAX_GROUP_CHILDREN; nest multiple groups to exceed MAX_CONDITIONS
    # while staying within per-group / depth limits.
    nested = {"op": "and", "children": [
        {"op": "or", "children": children[:16]},
        {"op": "or", "children": children[16:32]},
        {"field": "name", "op": "eq", "value": "one-too-many-x1"},
        {"field": "name", "op": "eq", "value": "one-too-many-x2"},
    ]}
    with pytest.raises(ASTError, match="more than 32 conditions"):
        parse_query({"where": nested})


def test_rejects_too_many_group_children():
    children = [{"field": "name", "op": "eq", "value": str(i)} for i in range(17)]
    with pytest.raises(ASTError):
        parse_query({"where": {"op": "or", "children": children}})


def test_rejects_excess_nesting_depth():
    node = {"field": "name", "op": "eq", "value": "x"}
    for _ in range(8):
        node = {"op": "not", "child": node}
    with pytest.raises(ASTError):
        parse_query({"where": node})


def test_rejects_string_literal_too_long():
    with pytest.raises(ASTError):
        parse_query({"where": {"field": "name", "op": "eq", "value": "x" * 513}})


def test_rejects_unsupported_value_type():
    with pytest.raises(ASTError):
        parse_query({"where": {"field": "name", "op": "eq", "value": {"nested": {"a": 1}}}})


def test_rejects_list_value_too_long():
    with pytest.raises(ASTError):
        parse_query({"where": {"field": "type", "op": "in", "value": [str(i) for i in range(65)]}})


def test_rejects_invalid_json_string():
    with pytest.raises(ASTError):
        parse_query("{not json")


def test_rejects_non_object_top_level():
    with pytest.raises(ASTError):
        parse_query([1, 2, 3])


# ---------------------------------------------------------------------------
# iter_conditions
# ---------------------------------------------------------------------------

def test_iter_conditions_visits_every_leaf():
    q = parse_query({"where": {"op": "and", "children": [
        {"field": "a", "op": "eq", "value": 1},
        {"op": "not", "child": {"field": "b", "op": "eq", "value": 2}},
        {"op": "or", "children": [
            {"field": "c", "op": "eq", "value": 3},
            {"field": "d", "op": "eq", "value": 4},
        ]},
    ]}})
    found = {c.field for c in iter_conditions(q.where)}
    assert found == {"a", "b", "c", "d"}


def test_iter_conditions_empty_for_none():
    assert list(iter_conditions(None)) == []


# ---------------------------------------------------------------------------
# canonicalize
# ---------------------------------------------------------------------------

def test_canonicalize_equivalent_for_reordered_and_children():
    q1 = parse_query({"where": {"op": "and", "children": [
        {"field": "type", "op": "eq", "value": "image"},
        {"field": "is_favorite", "op": "eq", "value": True},
    ]}})
    q2 = parse_query({"where": {"op": "and", "children": [
        {"field": "is_favorite", "op": "eq", "value": True},
        {"field": "type", "op": "eq", "value": "image"},
    ]}})
    assert q1.to_dict() != q2.to_dict()  # different child order pre-canonicalization
    assert canonicalize(q1) == canonicalize(q2)


def test_canonicalize_distinguishes_semantically_different_queries():
    q1 = parse_query({"where": {"op": "and", "children": [
        {"field": "type", "op": "eq", "value": "image"},
        {"field": "type", "op": "eq", "value": "video"},
    ]}})
    q2 = parse_query({"where": {"op": "or", "children": [
        {"field": "type", "op": "eq", "value": "image"},
        {"field": "type", "op": "eq", "value": "video"},
    ]}})
    assert canonicalize(q1) != canonicalize(q2)


def test_canonicalize_recurses_into_nested_groups():
    q1 = parse_query({"where": {"op": "and", "children": [
        {"op": "or", "children": [
            {"field": "a", "op": "eq", "value": 1},
            {"field": "b", "op": "eq", "value": 2},
        ]},
        {"field": "c", "op": "eq", "value": 3},
    ]}})
    q2 = parse_query({"where": {"op": "and", "children": [
        {"field": "c", "op": "eq", "value": 3},
        {"op": "or", "children": [
            {"field": "b", "op": "eq", "value": 2},
            {"field": "a", "op": "eq", "value": 1},
        ]},
    ]}})
    assert canonicalize(q1) == canonicalize(q2)


# ---------------------------------------------------------------------------
# json_schema
# ---------------------------------------------------------------------------

def test_json_schema_embeds_field_and_operator_enums_from_fields_registry():
    schema = json_schema(field_names=fields.field_names(), operator_names=fields.all_ops())
    cond_schema = schema["$defs"]["cond"]
    assert cond_schema["properties"]["field"]["enum"] == sorted(fields.field_names())
    assert cond_schema["properties"]["op"]["enum"] == sorted(fields.all_ops())
    # Sanity: a couple of concrete fields/ops actually show up.
    assert "status_flag" in cond_schema["properties"]["field"]["enum"]
    assert "between" in cond_schema["properties"]["op"]["enum"]


def test_json_schema_without_vocabularies_has_no_enum_constraint():
    schema = json_schema()
    cond_schema = schema["$defs"]["cond"]
    assert "enum" not in cond_schema["properties"]["field"]
    assert "enum" not in cond_schema["properties"]["op"]


def test_json_schema_is_self_consistent_dict_shape():
    schema = json_schema(field_names=["type"], operator_names=["eq"])
    assert schema["type"] == "object"
    assert schema["properties"]["result"]["enum"] == ["ids", "count"]
    assert schema["properties"]["target"]["const"] == "files"
