"""A prepared download must be cleared, and must not be offered once it is.

Preparing a download writes a zip holding a second copy of every file that
went into it, into the gallery folder -- often the same disk as the
library, often a synced one. It is kept for a day.

Two things were wrong with that, and they are the same mistake from two
sides: the only sweep ran at the END of building another zip.

Prepare one download and never prepare another, and nothing ever removed
it. Measured: a zip aged to 25 hours old was still on disk, because
nothing had happened since to look at it.

And the job entry was never removed with the file. After a later download
did trigger the sweep, the entry still said ready:

    after a second download was prepared:
      yesterday's file on disk: False

    check_zip_status("yesterday") -> 200
      {'download_url': '/galleryout/serve_zip/smartgallery_yesterday.zip',
       'filename': 'smartgallery_yesterday.zip', 'status': 'ready'}

    following the download_url it handed out:
      status 404, content-type text/html; charset=utf-8

So the gallery offered a link to a file it had deleted itself, and the
link answered with an HTML error page.

The sweep now also runs at startup, which is the one moment every gallery
reaches, and it forgets the jobs along with the files. The status route
asks the disk rather than the note, and says so in words the download
panel already knows how to show.
"""

from __future__ import annotations

import ast
import os
import time

import pytest

import smartgallery


@pytest.fixture
def zip_cache(smartgallery_app, tmp_path, monkeypatch):
    cache = tmp_path / "zips"
    cache.mkdir()
    monkeypatch.setattr(smartgallery_app, "ZIP_CACHE_DIR", str(cache))
    before = dict(smartgallery_app.zip_jobs)
    smartgallery_app.zip_jobs.clear()
    yield cache
    smartgallery_app.zip_jobs.clear()
    smartgallery_app.zip_jobs.update(before)


@pytest.fixture
def staff(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "ADMIN"
    return client


def _write(cache, name, age_seconds=0):
    path = cache / name
    path.write_bytes(b"PK\x03\x04 pretend zip")
    if age_seconds:
        when = time.time() - age_seconds
        os.utime(str(path), (when, when))
    return path


def test_a_download_nobody_came_back_for_is_cleared(zip_cache):
    """The one that needed a second download to happen before it would."""
    old = _write(zip_cache, "smartgallery_old.zip", age_seconds=25 * 3600)

    removed, _forgotten = smartgallery.prune_zip_cache()

    assert removed == 1
    assert not old.exists()


def test_a_download_from_this_morning_is_left_alone(zip_cache):
    """Over-reach guard. Clearing everything would satisfy the check above
    and take away downloads people are still using."""
    fresh = _write(zip_cache, "smartgallery_fresh.zip", age_seconds=60)

    smartgallery.prune_zip_cache()

    assert fresh.exists(), "a download prepared a minute ago was deleted"


def test_the_job_is_forgotten_with_its_file(zip_cache):
    smartgallery.zip_jobs["gone"] = {"status": "ready",
                                     "filename": "smartgallery_gone.zip",
                                     "created": time.time()}

    _removed, forgotten = smartgallery.prune_zip_cache()

    assert forgotten == 1
    assert "gone" not in smartgallery.zip_jobs


def test_a_job_still_being_built_is_left_alone(zip_cache):
    """Over-reach guard: a zip in progress has no file yet, and forgetting
    it would lose the download somebody is waiting on."""
    smartgallery.zip_jobs["busy"] = {"status": "processing",
                                     "created": time.time()}

    smartgallery.prune_zip_cache()

    assert "busy" in smartgallery.zip_jobs


def test_a_failure_is_forgotten_once_it_is_old(zip_cache):
    """A failed job has no file to go with it, so nothing would ever
    remove it -- which is how the registry grew for the life of the
    process."""
    smartgallery.zip_jobs["failed_now"] = {"status": "error",
                                           "message": "no",
                                           "created": time.time()}
    smartgallery.zip_jobs["failed_old"] = {"status": "error",
                                           "message": "no",
                                           "created": time.time() - 25 * 3600}

    smartgallery.prune_zip_cache()

    assert "failed_now" in smartgallery.zip_jobs, "a fresh failure was hidden"
    assert "failed_old" not in smartgallery.zip_jobs


def test_a_cleared_download_is_not_offered(zip_cache, staff):
    """The bug, through the route the download panel actually polls: the
    entry says ready, the file is gone, and it handed out a link anyway."""
    smartgallery.zip_jobs["stale"] = {"status": "ready",
                                      "filename": "smartgallery_stale.zip",
                                      "created": time.time()}

    response = staff.get("/galleryout/check_zip_status/stale")
    body = response.get_json()

    assert body is not None, response.get_data(as_text=True)[:200]
    assert body["status"] != "ready", (
        f"offered a download whose file is gone: {body}")
    assert "download_url" not in body, body
    assert "prepare it again" in body["message"].lower(), body["message"]


def test_a_download_that_is_there_is_still_offered(zip_cache, staff):
    """Over-reach guard, and the ordinary case: checking the disk must not
    take away downloads that exist."""
    _write(zip_cache, "smartgallery_here.zip")
    smartgallery.zip_jobs["here"] = {"status": "ready",
                                     "filename": "smartgallery_here.zip",
                                     "created": time.time()}

    response = staff.get("/galleryout/check_zip_status/here")
    body = response.get_json()

    assert response.status_code == 200, body
    assert body["status"] == "ready", body
    assert body["download_url"].endswith("smartgallery_here.zip"), body

    fetched = staff.get(body["download_url"])
    assert fetched.status_code == 200, (
        "the link it handed out does not serve the file")
    assert fetched.get_data().startswith(b"PK")


def test_the_answer_is_always_something_the_panel_can_read(zip_cache, staff):
    """Control for the shape: the panel does res.json() and reads .message,
    so every one of these has to be JSON whatever went wrong."""
    smartgallery.zip_jobs["stale"] = {"status": "ready",
                                      "filename": "smartgallery_missing.zip",
                                      "created": time.time()}

    for job_id in ("stale", "never_existed"):
        response = staff.get(f"/galleryout/check_zip_status/{job_id}")
        body = response.get_json()
        assert body is not None, (
            f"{job_id} answered {response.status_code} with "
            f"{response.get_data(as_text=True)[:120]}")
        assert body.get("message"), body


def test_the_link_would_have_answered_with_a_page(zip_cache, staff):
    """Control. The checks above are about not offering a dead link; that
    matters because following one is not a readable failure."""
    dead = staff.get("/galleryout/serve_zip/smartgallery_not_there.zip")

    assert dead.status_code == 404
    assert dead.get_json() is None, (
        "a missing zip now answers in JSON, so handing out a dead link "
        "would no longer be the failure these checks are about")


def test_startup_clears_what_it_finds(gallery_tree):
    """The sweep has to run somewhere other than the end of building a zip,
    or a gallery that prepared one download and stopped keeps it for good.
    Startup is the one moment every gallery reaches."""

    tree = gallery_tree

    starters = [node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
                and node.name.startswith("initialize_gallery")]
    assert starters, "no initialize_gallery function found; this check is stale"

    for start in starters:
        called = {node.func.id for node in ast.walk(start)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name)}
        assert "prune_zip_cache" in called, (
            f"{start.name} does not clear old prepared downloads, so a "
            f"gallery that never prepares another one keeps them")
