"""Every route that answers with data has to say who may ask.

This came out of a sweep over all 94 routes rather than another single
find. Nine had no gate of any kind -- no decorator, no session check, no
per-file check. Two of those are meant to be open (the redirect and the
login form itself); the rest answered anyone who could reach the port,
including on a server started with --force-login:

  ai_indexing/status   named the file being indexed and the next ten
                       queued, so an anonymous caller could watch the
                       library go by, filenames and all
  ai_queue             accepted a search, wrote to the database and
                       scheduled model work, unauthenticated
  ai_check             status of any search session
  check_rescan_status  status of a scan job
  check_zip_status     the name of a prepared zip
  serve_zip            the zip itself: original files, prompts included

None of them is used by the exhibition template -- all six belong to the
management interface -- so requiring management rights breaks nothing a
visitor does.
"""

from __future__ import annotations

import pytest

# (method, path) for each route that was open. GETs and one POST.
_ROUTES = [
    ("GET", "/galleryout/ai_indexing/status"),
    ("POST", "/galleryout/ai_queue"),
    ("GET", "/galleryout/ai_check/whatever-session"),
    ("GET", "/galleryout/check_rescan_status/whatever-job"),
    ("GET", "/galleryout/check_zip_status/whatever-job"),
    ("GET", "/galleryout/serve_zip/smartgallery_whatever.zip"),
    # Not a status report: this one walks the folder, writes what changed to
    # the database, and names each file in the stream as it goes.
    ("GET", "/galleryout/sync_status/_root_"),
    # Listed every album regardless of who asked, private ones included, and
    # resolved shared_users into the full names of the people it was shared
    # with. Asking only for a session was not enough.
    ("GET", "/galleryout/api/sidebar_state"),
    # A probe, sometimes a transcode, and eleven ffmpeg frame extractions per
    # call. Only the management page asks for it, but any visitor who could
    # see a video could set it going, for every video they could see.
    ("GET", "/galleryout/storyboard/whatever-file"),
]


def _call(client, method, path):
    if method == "POST":
        return client.post(path, json={"query": "anything"})
    return client.get(path)


@pytest.fixture()
def locked(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", True)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    return smartgallery_app


@pytest.mark.parametrize("method,path", _ROUTES)
def test_an_anonymous_caller_is_refused(locked, method, path):
    """The regression: each of these answered with data."""
    resp = _call(locked.app.test_client(), method, path)

    assert resp.status_code in (401, 403), (
        f"{path} answered {resp.status_code} to a caller with no session")


@pytest.mark.parametrize("method,path", _ROUTES)
def test_a_logged_in_customer_is_refused(locked, method, path):
    """Authentication is not authorisation: these belong to the management
    interface, which a CUSTOMER may not use."""
    client = locked.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 9
        session["role"] = "CUSTOMER"

    resp = _call(client, method, path)

    assert resp.status_code == 403, (
        f"{path} answered a customer with {resp.status_code}")


@pytest.mark.parametrize("method,path", _ROUTES)
def test_staff_still_reach_them(locked, method, path):
    """The counterpart. These are polled constantly by the management page,
    so a blanket denial would break the interface it belongs to -- and
    would satisfy both tests above on its own."""
    client = locked.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "ADMIN"

    resp = _call(client, method, path)

    assert resp.status_code not in (401, 403), (
        f"{path} refused staff with {resp.status_code}")


def test_an_anonymous_caller_cannot_set_a_scan_going(locked):
    """Status codes are the symptom; this is the property. The sync route
    walks the folder and writes what it finds, so a refusal has to mean the
    work never happened -- not merely that the reply looked like a refusal."""
    import os

    from PIL import Image

    base = locked.BASE_OUTPUT_PATH
    path = os.path.join(base, "gating_unindexed.png")
    Image.new("RGB", (16, 16), (7, 7, 7)).save(path)

    conn = locked.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE name = 'gating_unindexed.png'")
        conn.commit()
        before = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    finally:
        conn.close()

    try:
        locked.app.test_client().get("/galleryout/sync_status/_root_")

        conn = locked.get_db_connection()
        try:
            after = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            indexed = conn.execute(
                "SELECT COUNT(*) FROM files WHERE name = 'gating_unindexed.png'"
            ).fetchone()[0]
        finally:
            conn.close()

        assert indexed == 0, "an anonymous request indexed a file"
        assert after == before, f"an anonymous request wrote to the database ({before} -> {after})"
    finally:
        os.remove(path)


@pytest.mark.parametrize("method,path", _ROUTES)
def test_the_default_local_install_reaches_them(smartgallery_app, monkeypatch,
                                                method, path):
    """No login configured: one person, and the gallery is theirs."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    resp = _call(smartgallery_app.app.test_client(), method, path)

    assert resp.status_code not in (401, 403), (
        f"{path} refused the local administrator with {resp.status_code}")
