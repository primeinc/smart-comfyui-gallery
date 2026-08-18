"""The AI status report is a cross-file report, so it needs the same gate.

Every other route in the AI blueprint is deliberate about this: per-file
routes call the host's file-access check, and anything that reads across
files -- cluster listings, the reviews list, the duplicates overview --
carries the management guard, with a comment saying why.

`/status` was registered bare. It reports counts for the whole library and
the worker's recent errors, and those error messages quote the path of the
file that failed:

    self._note_error(f"faces:{file_id}", f"faces: failed for {path}: {exc}")

So an unidentified caller on a login-protected gallery could read the
server's own filesystem paths out of it, which is the same disclosure that
closed on `ai_indexing/status` one commit ago, through a blueprint the
route sweep never looked at.

It keeps reporting while the AI layer is switched off -- that is how the
panel knows to say so -- because the guard is applied without the
enabled-check the other routes use.
"""

from __future__ import annotations

import pytest

_STATUS = "/galleryout/api/aidam/status"


@pytest.fixture
def locked(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    return smartgallery_app


def test_an_anonymous_caller_is_refused(locked):
    """The regression: this answered with library counts and worker errors."""
    resp = locked.app.test_client().get(_STATUS)

    assert resp.status_code in (401, 403), (
        f"the AI status answered {resp.status_code} to a caller with no session")


def test_a_customer_is_refused(locked):
    client = locked.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 9
        session["role"] = "CUSTOMER"

    assert client.get(_STATUS).status_code == 403


def test_staff_still_get_it(locked):
    """The counterpart: the AI panel polls this constantly."""
    client = locked.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "ADMIN"

    resp = client.get(_STATUS)

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
    assert "enabled" in (resp.get_json() or {}), resp.get_json()


def test_the_default_local_install_still_gets_it(smartgallery_app, monkeypatch):
    """No login configured: one person, and the panel is theirs."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    resp = smartgallery_app.app.test_client().get(_STATUS)

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
    assert "enabled" in (resp.get_json() or {}), resp.get_json()


def test_it_still_reports_while_the_ai_layer_is_disabled(smartgallery_app, monkeypatch):
    """The property the bare registration was protecting: status answers
    even when the layer is off, so the panel can say it is off. Guarding it
    must not have turned that into an error."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    monkeypatch.setattr(smartgallery_app.AI_CONFIG, "enabled", False)

    resp = smartgallery_app.app.test_client().get(_STATUS)

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
    assert (resp.get_json() or {}).get("enabled") is False, resp.get_json()
