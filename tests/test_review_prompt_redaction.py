"""A review must not quote the prompt to someone who may not read prompts.

The review's alignment elements are, by the code's own definition, "always
a verbatim slice of the user's own positive prompt" -- the model judges
elements, it never invents them. The endpoint gated on whether the caller
may see the FILE, which is a different permission from whether they may
see how it was made. A visitor in exhibition mode could read the prompt
back one element at a time for every picture they were allowed to look at.

The scores and the findings stay: they describe the picture rather than
quote the request, which is what a viewer-facing review would want anyway.
The summary goes, because the critic writes it as free prose and quotes
the prompt when explaining a mismatch.
"""

from __future__ import annotations

import pytest

_PROMPT_SLICE = "a brass diving helmet"
_SUMMARY = f"the image does not show {_PROMPT_SLICE}"


@pytest.fixture()
def reviewed_file(smartgallery_app, monkeypatch):
    """One file, in a public album, with a stored review that quotes the
    prompt in its alignment elements."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", True)
    # The suite runs with the AI layer off, and every one of these routes
    # answers the disabled body when it is. Without this the redaction test
    # passes because there is no review at all -- which is why the two
    # "still visible" tests exist alongside it.
    monkeypatch.setattr(smartgallery_app.AI_CONFIG, "enabled", True)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("INSERT OR REPLACE INTO files (id, path, mtime, name, type, size) "
                     "VALUES ('rev1', '/x/rev.png', 1.0, 'rev.png', 'image', 1)")
        conn.execute("INSERT INTO collections (name, type, is_public, created_at) "
                     "VALUES ('revalbum', 'user_album', 1, 1.0)")
        coll_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("INSERT INTO collection_files (collection_id, file_id) "
                     "VALUES (?, 'rev1')", (coll_id,))
        conn.execute(
            "INSERT INTO ai_reviews (file_id, rubric_version, model_id, model_version, "
            "quality_score, prompt_alignment_score, summary, source_mtime, computed_at) "
            "VALUES ('rev1', 'v1', 'critic', '1', 7.5, 0.5, ?, 1.0, 1.0)", (_SUMMARY,))
        review_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO ai_review_alignment (review_id, file_id, ordinal, text, "
            "satisfied, confidence) VALUES (?, 'rev1', 0, ?, 0, 0.9)",
            (review_id, _PROMPT_SLICE))
        conn.execute(
            "INSERT INTO ai_review_findings (review_id, file_id, type, severity, "
            "confidence, localizable, description) "
            "VALUES (?, 'rev1', 'anatomy', 'low', 0.8, 0, 'an extra finger')",
            (review_id,))
        conn.commit()
    finally:
        conn.close()

    yield "rev1"

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute("DELETE FROM files WHERE id = 'rev1'")
        conn.execute("DELETE FROM collections WHERE name = 'revalbum'")
        conn.commit()
    finally:
        conn.close()


def _as(smartgallery_app, role):
    client = smartgallery_app.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 9
        session["role"] = role
    return client


def test_staff_see_the_whole_review(smartgallery_app, reviewed_file):
    """Control: the fixture's review is reachable and complete, so an
    absent prompt below is redaction rather than an empty database."""
    resp = _as(smartgallery_app, "ADMIN").get(f"/galleryout/api/aidam/review/{reviewed_file}")

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
    body = resp.get_json()
    assert body["review"]["alignment"], "the review has no alignment elements at all"
    assert body["review"]["alignment"][0]["text"] == _PROMPT_SLICE
    assert body["review"]["summary"] == _SUMMARY


def test_a_visitor_does_not_get_the_prompt_back(smartgallery_app, reviewed_file):
    """The regression: the elements are slices of the prompt."""
    resp = _as(smartgallery_app, "CUSTOMER").get(
        f"/galleryout/api/aidam/review/{reviewed_file}")

    assert resp.status_code == 200, resp.get_data(as_text=True)[:200]
    body = resp.get_data(as_text=True)
    assert _PROMPT_SLICE not in body, "the prompt was returned to a visitor"
    assert _SUMMARY not in body, "the summary quoted the prompt to a visitor"


def test_a_visitor_still_gets_the_verdict(smartgallery_app, reviewed_file):
    """Withholding the quotes must not empty the review: the scores and the
    findings describe the picture, not the request."""
    body = _as(smartgallery_app, "CUSTOMER").get(
        f"/galleryout/api/aidam/review/{reviewed_file}").get_json()

    assert body["review"]["scores"]["quality"] == 7.5
    assert body["findings"], "the findings went too"
    assert body["findings"][0]["description"] == "an extra finger"


def test_the_default_local_install_sees_everything(smartgallery_app, reviewed_file,
                                                    monkeypatch):
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)

    body = smartgallery_app.app.test_client().get(
        f"/galleryout/api/aidam/review/{reviewed_file}").get_json()

    assert body["review"]["alignment"][0]["text"] == _PROMPT_SLICE
    assert body["review"]["summary"] == _SUMMARY
