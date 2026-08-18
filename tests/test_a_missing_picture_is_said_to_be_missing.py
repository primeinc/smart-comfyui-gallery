"""Asking for a picture that is not there must say so.

`get_file_info_from_db` calls `abort(404)`. Several routes wrap their
whole body in `except Exception`, which catches that abort like any other
exception, and answered with something else entirely.

Measured across the 24 addresses that take a file id and nothing else,
asking each for an id that is not in the library:

    twelve   404   said it plainly
    three    500   reported a server fault
    one      200   said the request had succeeded, with the 404's own
                   text pasted into the message

The three were /galleryout/storyboard/, /galleryout/api/remix/info/ and
/galleryout/api/remix/companion/; the fourth was /galleryout/node_summary/.

The difference is the whole meaning of the answer. A stale tab, a
bookmarked picture since deleted, a second browser open on a library
being tidied -- all ordinary -- were reported to the person as the
gallery having broken. And a real fault looked exactly the same as that,
so neither could be believed. The 200 is worse in its own way: the page
was told it had worked.

Flask's docs say it outright (errorhandling.rst, "Generic Exception
Handlers"): a handler that catches everything must be written so that it
does not lose the HTTP error. The three routes now let their own refusal
through, in JSON, because every caller of these addresses reads JSON.

This is written as a sweep rather than four checks, so a route added
later that swallows its own 404 fails here without anyone remembering to
come back.
"""

from __future__ import annotations

import json
import struct
import zlib

import pytest
from werkzeug.exceptions import Forbidden, NotFound

import smartgallery

MISSING = "deadbeef" * 4  # the right shape, no such picture


def a_real_png():
    def chunk(kind, payload):
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\xff"))
        + chunk(b"IEND", b"")
    )


def file_id_routes(app):
    """Every address whose only path argument names a file."""
    found = []
    for rule in app.url_map.iter_rules():
        if set(rule.arguments or ()) != {"file_id"}:
            continue
        for method in sorted((rule.methods or set()) - {"HEAD", "OPTIONS"}):
            found.append((method, str(rule)))
    return sorted(found)


@pytest.fixture
def a_gallery_with_one_picture(smartgallery_app, tmp_path, monkeypatch):
    sg = smartgallery_app
    root = tmp_path / "one_picture"
    root.mkdir()
    monkeypatch.setattr(sg, "BASE_OUTPUT_PATH", str(root))

    path = root / "real.png"
    path.write_bytes(a_real_png())

    client = sg.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "ADMIN"
    return sg, client


def ask(client, method, template, file_id):
    return client.open(
        template.replace("<file_id>", file_id).replace("<string:file_id>", file_id),
        method=method,
        json={} if method in ("POST", "PUT", "PATCH") else None,
        headers={"Sec-Fetch-Site": "same-origin"},
    )


def test_there_are_routes_to_sweep(a_gallery_with_one_picture):
    """Control. If the sweep ever finds nothing it must fail rather than
    pass by measuring an empty list."""
    sg, _client = a_gallery_with_one_picture

    routes = file_id_routes(sg.app)
    assert len(routes) >= 20, f"only found {len(routes)} routes taking a file id"


def test_no_route_calls_a_missing_picture_a_server_fault(a_gallery_with_one_picture):
    """The defect. Three of them answered 500."""
    sg, client = a_gallery_with_one_picture

    faulted = []
    for method, template in file_id_routes(sg.app):
        answer = ask(client, method, template, MISSING)
        if answer.status_code >= 500:
            faulted.append(f"{method} {template} -> {answer.status_code}")

    assert not faulted, (
        "asking for a picture that is not in the library was reported as "
        "the gallery having broken:\n  " + "\n  ".join(faulted)
    )


def test_no_route_calls_a_missing_picture_a_success(a_gallery_with_one_picture):
    """The other half: one answered 200 while saying, in its own body,
    that it had failed."""
    sg, client = a_gallery_with_one_picture

    lying = []
    for method, template in file_id_routes(sg.app):
        answer = ask(client, method, template, MISSING)
        if answer.status_code != 200:
            continue
        if not answer.is_json:
            continue
        body = answer.get_json()
        if isinstance(body, dict) and body.get("status") == "error":
            lying.append("{} {} -> 200 but {!r}".format(method, template, body.get("message")))

    assert not lying, "answered 200 while its own body reported a failure:\n  " + "\n  ".join(lying)


def test_the_refusal_does_not_quote_the_machine(a_gallery_with_one_picture):
    """Over-reach guard against fixing this by pasting the exception in.

    A message handed to a visitor must not carry a path from the disk, a
    fragment of SQL, or a source file name -- the same rule the general
    fault handler follows."""
    sg, client = a_gallery_with_one_picture

    leaked = []
    for method, template in file_id_routes(sg.app):
        answer = ask(client, method, template, MISSING)
        if not answer.is_json:
            continue
        text = json.dumps(answer.get_json())
        for giveaway in ("Traceback", "SELECT ", "sqlite", ".py", "C:\\", "/home/", "smartgallery."):
            if giveaway in text:
                leaked.append(f"{method} {template} leaked {giveaway!r}: {text[:120]}")

    assert not leaked, "\n  ".join(leaked)


def test_a_refusal_says_what_is_wrong_in_words(a_gallery_with_one_picture):
    """The default text for a 404 is "The requested URL was not found on
    the server", which is untrue here: the address exists, the picture
    does not."""
    sg, client = a_gallery_with_one_picture

    # the route that raises it, rather than one that answers for itself
    answer = ask(client, "GET", "/galleryout/storyboard/<string:file_id>", MISSING)
    assert answer.status_code == 404
    assert "not in the gallery" in answer.get_data(as_text=True), answer.get_data(as_text=True)[:200]

    # and nowhere may still claim the address itself was wrong
    untrue = []
    for method, template in file_id_routes(sg.app):
        said = ask(client, method, template, MISSING).get_data(as_text=True)
        if "requested URL was not found" in said:
            untrue.append(f"{method} {template}")
    assert not untrue, (
        "told somebody the address was wrong when the address was right and the picture was gone: " + ", ".join(untrue)
    )


def test_the_picture_that_is_there_still_answers(a_gallery_with_one_picture):
    """Over-reach guard, and the point of all of it: none of this may
    start refusing pictures that exist."""
    sg, client = a_gallery_with_one_picture

    with sg.get_db_connection() as conn:
        conn.execute("DELETE FROM files")
        conn.commit()
        sg.full_sync_database(conn)
        row = conn.execute("SELECT id FROM files WHERE name = 'real.png'").fetchone()
    assert row, "the scan did not record the picture this check depends on"

    # read-only addresses only: the sweep above must not be able to delete
    # the very file it is checking for
    answer = ask(client, "GET", "/galleryout/file/<string:file_id>", row["id"])
    assert answer.status_code == 200, answer.status_code

    details = ask(client, "GET", "/galleryout/api/file_full_details/<string:file_id>", row["id"])
    assert details.status_code == 200, details.status_code

    with smartgallery.get_db_connection() as conn:
        conn.execute("DELETE FROM files")
        conn.commit()


class TestTheHelperItself:
    def test_it_keeps_the_status(self):

        with smartgallery.app.test_request_context("/"):
            _body, code = smartgallery.answer_an_abort_readably(NotFound())
            assert code == 404
            _body, code = smartgallery.answer_an_abort_readably(Forbidden())
            assert code == 403

    def test_it_says_something_when_there_is_no_description(self):

        with smartgallery.app.test_request_context("/"):
            body, _code = smartgallery.answer_an_abort_readably(Forbidden())
            said = body.get_json()
            assert said["status"] == "error"
            assert said["message"], "handed back an empty message"
