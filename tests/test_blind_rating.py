"""`--blind-rating`: the flag exists so raters are not anchored.

The point of the flag is that a rater cannot see the crowd's average
before forming their own opinion. That guarantee has exactly one gate,
`is_effectively_blind()`, and it is worth pinning because the failure mode
is silent: nothing errors when blindness stops applying, the averages
simply reappear and every rating collected afterwards is quietly worth
less.

The property that matters most is that a RATER cannot switch it off. An
operator may -- they configured it and may need totals to do their job --
but a guest setting the same session key must be ignored, and the endpoint
that sets it must refuse them outright.

Testing style follows the Flask docs: `test_request_context` in a `with`
block to unit-test the single context-reading gate (docs/appcontext.rst),
and `client.session_transaction()` to seed a session before a request
(docs/testing.rst).
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def client(smartgallery_app):
    return smartgallery_app.app.test_client()


@pytest.fixture()
def blind_server(smartgallery_app, monkeypatch):
    """A server started with --blind-rating, with logins in play so guests
    and staff are distinguishable (without FORCE_LOGIN every caller is the
    local admin and there is no rater to protect)."""
    monkeypatch.setattr(smartgallery_app, "BLIND_RATING", True)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    return smartgallery_app


def _blind_with(smartgallery_app, **session_values):
    """is_effectively_blind() evaluated against a given session."""
    from flask import session

    with smartgallery_app.app.test_request_context("/galleryout/"):
        session.update(session_values)
        return smartgallery_app.is_effectively_blind()


def test_guest_stays_blind_even_when_claiming_the_override(blind_server):
    """The one that matters: a rater must not unblind themselves by
    setting the same session key the operator's toggle uses."""
    assert _blind_with(blind_server, role="GUEST") is True
    assert _blind_with(blind_server, role="GUEST", override_blind=True) is True, (
        "a guest disabled blind rating by setting override_blind")
    assert _blind_with(blind_server, role="CUSTOMER", override_blind=True) is True


@pytest.mark.parametrize("role", ["ADMIN", "MANAGER", "STAFF"])
def test_staff_can_lift_it_for_themselves(blind_server, role):
    assert _blind_with(blind_server, role=role) is True, (
        f"{role} was not blind before opting out")
    assert _blind_with(blind_server, role=role, override_blind=True) is False, (
        f"{role} could not lift blind rating")


def test_without_the_flag_blindness_is_opt_in_per_session(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "BLIND_RATING", False)
    assert _blind_with(smartgallery_app, role="GUEST") is False
    assert _blind_with(smartgallery_app, role="GUEST", my_ratings_only=True) is True


def test_the_toggle_endpoint_refuses_a_guest(blind_server, client):
    with client.session_transaction() as session:
        session["role"] = "GUEST"
        session["user_id"] = 7

    resp = client.post("/galleryout/api/exhibition/toggle_blind")

    assert resp.status_code == 403, "a guest was allowed to toggle blind rating"
    with client.session_transaction() as session:
        assert not session.get("override_blind"), (
            "the refused request still set the override")


def test_the_toggle_endpoint_flips_for_staff(blind_server, client):
    with client.session_transaction() as session:
        session["role"] = "ADMIN"
        session["user_id"] = 1

    assert client.post("/galleryout/api/exhibition/toggle_blind").status_code == 200
    with client.session_transaction() as session:
        assert session.get("override_blind") is True

    assert client.post("/galleryout/api/exhibition/toggle_blind").status_code == 200
    with client.session_transaction() as session:
        assert session.get("override_blind") is False


def test_opting_into_blindness_needs_no_privilege(smartgallery_app, client):
    """my_ratings_only only ever hides more, so anyone may set it."""
    with client.session_transaction() as session:
        session["role"] = "GUEST"

    assert client.post("/galleryout/api/exhibition/toggle_my_ratings").status_code == 200
    with client.session_transaction() as session:
        assert session.get("my_ratings_only") is True


def test_per_rater_detail_stays_behind_the_management_gate(smartgallery_app, client):
    """rating_details returns every individual rating with names. That is a
    moderation view, and a rater reaching it would defeat the flag by
    reading the crowd directly."""
    with client.session_transaction() as session:
        session["role"] = "GUEST"
        session["user_id"] = 7

    resp = client.get("/galleryout/api/exhibition/rating_details?file_id=whatever")

    assert resp.status_code == 403, "a guest could read per-rater detail"
