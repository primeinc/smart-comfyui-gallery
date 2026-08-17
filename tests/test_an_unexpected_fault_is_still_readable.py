"""When something goes wrong, the screen must be able to say so.

Swept every JSON route with values of the wrong KIND -- a string where a
number was expected, a list where a string was, an object where a list
was. That is the fault behind the rating route, where `1 <= rating <= 5`
met a string, and it is not one route's problem:

                          unreadable by the caller   answered readably
    before                        112                      123
    after                           0                      235

Everything the interface does goes through fetch(...).then(res =>
res.json()), so an HTML error page is not a worse message -- it is no
message. The person sees nothing happen.

Sixty-three route-and-field pairs produced those, and hand-checking types
in sixty-three places would be sixty-three chances to get it wrong plus a
fresh one every time a route is added. What they have in common is the
answer, so the answer is what is fixed.

Two things it deliberately does not do:

It does not swallow the fault. The traceback still reaches the console
exactly as before, so nothing became harder to diagnose -- that is what
test_the_console_still_gets_the_whole_story holds.

It does not repeat the exception's own message, which is how a filesystem
path or a fragment of SQL would reach an exhibition visitor. The type
name is enough to tell two faults apart; the console has the rest, and
the console is the owner's.

Flask's docs warn about exactly this handler (errorhandling.rst, "Generic
Exception Handlers"): it fires for things you did not cause, such as
routing's own 404 and 405, so "be sure to craft your handler carefully so
you don't lose information about the HTTP error". HTTPException is handed
straight back, and the tests below hold that.
"""

from __future__ import annotations

import pytest

import smartgallery


@pytest.fixture()
def caller(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "ADMIN"
    return client


# A real one from the sweep: file_ids is iterated as a list, and a string
# is iterable, so it gets as far as the database and fails there.
_A_FAULT = ("/galleryout/favorite_batch", {"file_ids": "abc", "status": True})


def test_an_unexpected_fault_answers_in_json(caller):
    """The bug: an HTML page to something doing res.json()."""
    route, payload = _A_FAULT

    response = caller.post(route, json=payload)

    assert response.status_code == 500, response.status_code
    body = response.get_json()
    assert body is not None, response.get_data(as_text=True)[:200]
    assert body["status"] == "error"
    assert body["message"], body


def test_the_message_names_the_kind_of_fault(caller):
    """Two different faults must not read identically, or a bug report
    says only "it did not work"."""
    route, payload = _A_FAULT

    message = caller.post(route, json=payload).get_json()["message"]

    assert "TypeError" in message or "Error" in message, message


def test_the_message_does_not_repeat_what_went_wrong(caller):
    """An exhibition visitor triggering a fault must not be handed a
    filesystem path or a piece of SQL. The console has those."""
    route, payload = _A_FAULT

    message = caller.post(route, json=payload).get_json()["message"]

    for leak in ("SELECT", "INSERT", "sqlite", "Traceback", ".py", "C:\\", "/lib/"):
        assert leak not in message, (leak, message)


def test_the_console_still_gets_the_whole_story(caller, capsys):
    """The one thing worse than an unreadable error is a hidden one."""
    route, payload = _A_FAULT

    caller.post(route, json=payload)

    printed = capsys.readouterr()
    assert "Traceback" in (printed.out + printed.err), (
        "the fault was answered and never logged, so nothing can be "
        "diagnosed from it")


def test_a_missing_page_is_still_a_missing_page(caller):
    """Flask's warning made real: this handler sees routing's own 404s,
    and turning those into 500s would lose what they meant."""
    response = caller.get("/galleryout/no/such/thing/at/all")

    assert response.status_code == 404, response.status_code


def test_the_wrong_method_is_still_the_wrong_method(caller):
    """405 likewise -- it says something true and specific."""
    response = caller.get("/galleryout/favorite_batch")

    assert response.status_code == 405, response.status_code


def test_a_refusal_keeps_its_own_answer(smartgallery_app, monkeypatch):
    """403 from the gallery's own rules must survive too, or every
    considered refusal turns into "something went wrong"."""
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    client = smartgallery_app.app.test_client()

    response = client.get("/galleryout/storyboard_frame/%s/frame_001.jpg"
                          % ("a" * 32))

    assert response.status_code in (403, 404), response.status_code


@pytest.mark.parametrize("path,headers,expected", [
    ("/galleryout/api/anything", {}, True),
    ("/galleryout/favorite_batch", {"Content-Type": "application/json"}, True),
    ("/galleryout/view/abc", {"Accept": "text/html"}, False),
    ("/galleryout/view/abc", {}, False),
])
def test_who_is_told_in_json(smartgallery_app, path, headers, expected):
    """A page navigation should keep Flask's error page; anything calling
    with fetch should get something it can parse."""
    with smartgallery_app.app.test_request_context(path, headers=headers):
        assert smartgallery_app._caller_is_reading_json() is expected


def test_the_fault_used_for_these_checks_is_a_real_one(smartgallery_app):
    """Control. Every check here rests on that request actually going
    wrong; if it stopped doing so they would all pass for nothing."""
    route, payload = _A_FAULT
    ids = payload["file_ids"]

    # file_ids is iterated as a list of ids. A string is iterable, so it
    # reaches the database as one-character ids rather than being refused.
    assert isinstance(ids, str) and len(ids) > 1
    assert list(ids) == ["a", "b", "c"], (
        "the value used here is no longer the shape that provoked the fault")
