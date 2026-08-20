"""Tests for omniquery/validation.py."""

from __future__ import annotations

import pytest

from omniquery import fields
from omniquery.ast import parse_query
from omniquery.validation import (
    DEFAULT_LIMIT,
    MAX_CORRELATED_FIELDS,
    MAX_LIMIT,
    PRIVILEGED_ROLES,
    AuthContext,
    ValidatedQuery,
    ValidationError,
    _validate_value,
    validate,
)

GUEST = AuthContext(role="GUEST", user_id=None, client_uuid=None, ai_enabled=False)
STAFF = AuthContext(role="STAFF", user_id="3", client_uuid="client-3", ai_enabled=True)
ADMIN_NO_AI = AuthContext(role="ADMIN", user_id="5", client_uuid="client-5", ai_enabled=False)


def _q(where=None, **kw):
    obj = {"target": "files", "result": "ids"}
    if where is not None:
        obj["where"] = where
    obj.update(kw)
    return parse_query(obj)


def _cond(field, op, value=None):
    d = {"field": field, "op": op}
    if value is not None:
        d["value"] = value
    return d


# ---------------------------------------------------------------------------
# Field / op / value checks
# ---------------------------------------------------------------------------


def test_unknown_field_rejected():
    q = _q(where=_cond("not_a_real_field", "eq", "x"))
    with pytest.raises(ValidationError, match="unknown field"):
        validate(q, GUEST)


def test_unsupported_op_for_field_rejected():
    q = _q(where=_cond("is_favorite", "contains", "x"))
    with pytest.raises(ValidationError, match="not supported"):
        validate(q, GUEST)


def test_bad_enum_value_rejected():
    q = _q(where=_cond("type", "eq", "not-a-type"))
    with pytest.raises(ValidationError, match="invalid enum value"):
        validate(q, GUEST)


def test_valid_enum_value_accepted():
    q = _q(where=_cond("type", "eq", "image"))
    vq = validate(q, GUEST)
    assert isinstance(vq, ValidatedQuery)


def test_enum_in_with_empty_list_rejected():
    q = _q(where=_cond("type", "in", []))
    with pytest.raises(ValidationError, match="non-empty list"):
        validate(q, GUEST)


def test_enum_in_with_invalid_member_rejected():
    q = _q(where=_cond("type", "in", ["image", "bogus"]))
    with pytest.raises(ValidationError, match="invalid enum value"):
        validate(q, GUEST)


def test_number_field_rejects_non_numeric_value():
    q = _q(where=_cond("size_bytes", "gt", "not-a-number"))
    with pytest.raises(ValidationError, match="numeric"):
        validate(q, GUEST)


def test_number_field_rejects_bool_as_numeric():
    q = _q(where=_cond("size_bytes", "gt", True))
    with pytest.raises(ValidationError, match="numeric"):
        validate(q, GUEST)


def test_number_between_requires_two_item_list():
    q = _q(where=_cond("size_bytes", "between", [1]))
    with pytest.raises(ValidationError, match="2-item list"):
        validate(q, GUEST)


def test_bool_field_requires_bool_value():
    q = _q(where=_cond("is_favorite", "eq", "yes"))
    with pytest.raises(ValidationError, match="boolean"):
        validate(q, GUEST)


def test_text_field_requires_string_value():
    q = _q(where=_cond("name", "eq", 42))
    with pytest.raises(ValidationError, match="string"):
        validate(q, GUEST)


def test_is_null_op_rejects_a_value():
    q = _q(where=_cond("ai_caption", "is_null", "x"))
    with pytest.raises(ValidationError, match="takes no value"):
        validate(q, GUEST)


def test_is_null_op_without_value_accepted():
    q = _q(where=_cond("ai_caption", "is_null"))
    vq = validate(q, GUEST)
    assert isinstance(vq, ValidatedQuery)


@pytest.mark.parametrize("value", ["2025-01-01", "2025-01-01T12:30:00"])
def test_datetime_field_accepts_iso_strings(value):
    q = _q(where=_cond("mtime", "ge", value))
    validate(q, GUEST)


@pytest.mark.parametrize("value", ["01/01/2025", "2025-13-40", "not-a-date", "2025-01-01 12:30"])
def test_datetime_field_rejects_non_iso_strings(value):
    q = _q(where=_cond("mtime", "ge", value))
    with pytest.raises(ValidationError, match="invalid date string"):
        validate(q, GUEST)


def test_datetime_field_accepts_relative_dict():
    q = _q(where=_cond("mtime", "ge", {"days_ago": 7}))
    validate(q, GUEST)


def test_datetime_field_rejects_negative_relative_value():
    q = _q(where=_cond("mtime", "ge", {"days_ago": -1}))
    with pytest.raises(ValidationError, match="non-negative"):
        validate(q, GUEST)


def test_datetime_field_rejects_unknown_relative_key():
    q = _q(where=_cond("mtime", "ge", {"weeks_ago": 1}))
    with pytest.raises(ValidationError, match="relative date"):
        validate(q, GUEST)


def test_file_ref_near_dup_rejects_dict_value():
    q = _q(where=_cond("near_dup_of", "eq", {"file_id": "f001"}))
    with pytest.raises(ValidationError, match="plain file id string"):
        validate(q, STAFF)


def test_file_ref_similar_to_accepts_plain_string_or_dict():
    validate(_q(where=_cond("similar_to_semantic", "eq", "f001")), STAFF)
    validate(_q(where=_cond("similar_to_semantic", "eq", {"file_id": "f001", "k": 10})), STAFF)


def test_file_ref_rejects_k_over_200():
    q = _q(where=_cond("similar_to_semantic", "eq", {"file_id": "f001", "k": 201}))
    with pytest.raises(ValidationError, match="'k'"):
        validate(q, STAFF)


def test_file_ref_rejects_k_zero():
    q = _q(where=_cond("similar_to_semantic", "eq", {"file_id": "f001", "k": 0}))
    with pytest.raises(ValidationError, match="'k'"):
        validate(q, STAFF)


def test_file_ref_rejects_missing_file_id_key():
    q = _q(where=_cond("similar_to_semantic", "eq", {"k": 10}))
    with pytest.raises(ValidationError, match="file_id"):
        validate(q, STAFF)


# ---------------------------------------------------------------------------
# Authorization: privileged roles
# ---------------------------------------------------------------------------


def test_privileged_field_denied_for_guest():
    q = _q(where=_cond("rated_by_user", "eq", "carol"))
    with pytest.raises(ValidationError, match="privileged role"):
        validate(q, GUEST)


@pytest.mark.parametrize("role", sorted(PRIVILEGED_ROLES))
def test_privileged_field_allowed_for_privileged_roles(role):
    ctx = AuthContext(role=role, user_id="1", client_uuid="c1", ai_enabled=False)
    q = _q(where=_cond("rated_by_user", "eq", "carol"))
    validate(q, ctx)  # must not raise


def test_privileged_field_denied_for_non_privileged_named_role():
    ctx = AuthContext(role="CUSTOMER", user_id="1", client_uuid="c1", ai_enabled=False)
    q = _q(where=_cond("commented_by_user", "eq", "dave"))
    with pytest.raises(ValidationError, match="privileged role"):
        validate(q, ctx)


# ---------------------------------------------------------------------------
# Authorization: AI gating
# ---------------------------------------------------------------------------


def test_requires_ai_field_denied_when_ai_disabled():
    q = _q(where=_cond("has_faces", "eq", True))
    with pytest.raises(ValidationError, match="AI layer"):
        validate(q, ADMIN_NO_AI)


def test_requires_ai_field_allowed_when_ai_enabled():
    q = _q(where=_cond("has_faces", "eq", True))
    validate(q, STAFF)  # STAFF has ai_enabled=True


def test_requires_ai_order_by_denied_when_ai_disabled():
    q = _q(order_by=[{"field": "review_quality", "dir": "desc"}])
    with pytest.raises(ValidationError, match="AI layer"):
        validate(q, ADMIN_NO_AI)


# ---------------------------------------------------------------------------
# my_rating needs client_uuid regardless of role
# ---------------------------------------------------------------------------


def test_my_rating_without_client_uuid_rejected():
    ctx = AuthContext(role="ADMIN", user_id="5", client_uuid=None, ai_enabled=False)
    q = _q(where=_cond("my_rating", "ge", 4))
    with pytest.raises(ValidationError, match="client_uuid"):
        validate(q, ctx)


def test_my_rating_with_client_uuid_accepted():
    ctx = AuthContext(role="GUEST", user_id=None, client_uuid="anon-123", ai_enabled=False)
    q = _q(where=_cond("my_rating", "ge", 4))
    validate(q, ctx)  # must not raise; my_rating itself is not privileged/AI-gated


# ---------------------------------------------------------------------------
# limit handling
# ---------------------------------------------------------------------------


def test_default_limit_applied_when_absent():
    vq = validate(_q(), GUEST)
    assert vq.effective_limit == DEFAULT_LIMIT


def test_explicit_limit_within_cap_preserved():
    vq = validate(_q(limit=100), GUEST)
    assert vq.effective_limit == 100


def test_limit_at_cap_accepted():
    vq = validate(_q(limit=MAX_LIMIT), GUEST)
    assert vq.effective_limit == MAX_LIMIT


def test_limit_over_cap_rejected():
    with pytest.raises(ValidationError, match="exceeds the maximum"):
        validate(_q(limit=MAX_LIMIT + 1), GUEST)


# ---------------------------------------------------------------------------
# order_by validation
# ---------------------------------------------------------------------------


def test_order_by_unknown_field_rejected():
    q = _q(order_by=[{"field": "not_a_field", "dir": "asc"}])
    with pytest.raises(ValidationError, match="unknown field"):
        validate(q, GUEST)


def test_order_by_non_orderable_field_rejected():
    q = _q(order_by=[{"field": "path", "dir": "asc"}])
    with pytest.raises(ValidationError, match="not orderable"):
        validate(q, GUEST)


def test_order_by_orderable_field_accepted():
    q = _q(order_by=[{"field": "name", "dir": "asc"}])
    validate(q, GUEST)


def test_random_is_not_a_registered_orderable_field():
    assert "random" not in fields.ORDERABLE_FIELDS
    assert "random" not in fields.FIELDS
    q = _q(order_by=[{"field": "random", "dir": "asc"}])
    with pytest.raises(ValidationError):
        validate(q, GUEST)


# ---------------------------------------------------------------------------
# Complexity: distinct EXISTS-style field cap
# ---------------------------------------------------------------------------

_CORRELATED_SAMPLE = [
    ("rating_avg", "ge", 1),
    ("rating_count", "ge", 0),
    ("comment_count", "ge", 0),
    ("comment_contains", "contains", "hi"),
    ("collection", "eq", "Portfolio"),
    ("status_flag", "eq", "Approved"),
    ("rated_by_user", "eq", "carol"),
    ("commented_by_user", "eq", "dave"),
]


def test_exists_style_field_count_at_cap_accepted():
    assert len(_CORRELATED_SAMPLE) == MAX_CORRELATED_FIELDS
    children = [_cond(f, op, v) for f, op, v in _CORRELATED_SAMPLE]
    q = _q(where={"op": "and", "children": children})
    validate(q, STAFF)  # STAFF is privileged, so rated_by_user/commented_by_user pass too


def test_exists_style_field_count_over_cap_rejected():
    children = [_cond(f, op, v) for f, op, v in _CORRELATED_SAMPLE]
    children.append(_cond("face_count", "ge", 1))
    q = _q(where={"op": "and", "children": children})
    ctx = AuthContext(role="STAFF", user_id="3", client_uuid="c3", ai_enabled=True)
    with pytest.raises(ValidationError, match="EXISTS-style"):
        validate(q, ctx)


def test_repeated_use_of_same_correlated_field_does_not_count_twice():
    children = [
        _cond("rating_avg", "ge", 1),
        _cond("rating_avg", "le", 5),
    ]
    q = _q(where={"op": "and", "children": children})
    validate(q, GUEST)  # only 1 distinct correlated field, well under the cap


# ---------------------------------------------------------------------------
# ValidatedQuery construction guard
# ---------------------------------------------------------------------------


def test_validated_query_cannot_be_constructed_directly():
    q = _q()
    with pytest.raises(TypeError, match="cannot be constructed directly"):
        ValidatedQuery(q, 500, GUEST)


def test_validated_query_rejects_arbitrary_sentinel():
    q = _q()
    with pytest.raises(TypeError):
        ValidatedQuery(q, 500, GUEST, _sentinel=object())


def test_validated_query_is_immutable():
    vq = validate(_q(), GUEST)
    with pytest.raises(AttributeError):
        vq.effective_limit = 1


def test_validate_returns_validated_query_wrapping_original_query():
    q = _q(where=_cond("type", "eq", "image"))
    vq = validate(q, GUEST)
    assert vq.query is q
    assert vq.ctx is GUEST


# ---------------------------------------------------------------------------
# Defensive dispatch: _validate_value's kind dispatch is exhaustive over the
# real registry, so this is reached only by constructing a spec whose kind
# isn't one of the real Kind members (validate() never builds one).
# ---------------------------------------------------------------------------


def test_validate_value_rejects_unhandled_kind():
    bogus_spec = fields.FieldSpec(
        name="x", kind="not_a_real_kind", ops=frozenset({"eq"}), strategy=fields.Strategy.COLUMN
    )
    with pytest.raises(ValidationError, match="unhandled kind"):
        _validate_value(bogus_spec, "eq", "v")
