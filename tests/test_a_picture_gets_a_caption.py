"""The annotate job says one sentence about every picture and keeps it.

A caption is a derived annotation: per model, per kind, replaced by the
same model and kept beside a different one. The job reads the
`caption_model` setting once at submit, provisions the weights itself
(the one caller that may), and the media page shows what was said.
Nothing here touches the network: the captioner is replaced at the
seam and what reaches it is asserted.
"""

from __future__ import annotations

import pathlib
import time

import pytest
from litestar.testing import TestClient
from PIL import Image

from db import connect, derived, naming, runner, settings
from sg_web.app import build_app
from sglint import policy
from tests.staging import fresh_schema
from vision import captions, weights


class FakeCaptioner:
    model_id = "fake/captioner"
    model_version = "v1"

    def __init__(self, say="a red square on a table"):
        self.say = say
        self.seen: list = []

    def describe(self, image):
        self.seen.append(image.size)
        return self.say


def _one_picture(conn, root: pathlib.Path) -> int:
    from db import library, scan

    root.mkdir()
    Image.new("RGB", (16, 16), (200, 90, 40)).save(root / "one.png")
    root_id = library.add_root(conn, str(root), "library", 0.0)
    scan.scan(conn, root_id, str(root), 0.0)
    return conn.execute("SELECT id FROM file").fetchone()[0]


def test_submit_reads_the_caption_model_setting_into_the_payload(tmp_path):
    import json

    conn = fresh_schema()
    file_id = _one_picture(conn, tmp_path / "lib")
    settings.put(conn, "caption_model", "someone/other-captioner")

    job_id = runner.submit_annotate(conn, 0.0, models_dir="M")

    kind, payload = conn.execute("SELECT kind, payload FROM job WHERE id = ?", (job_id,)).fetchone()
    assert kind == "annotate"
    assert json.loads(payload) == {"models_dir": "M", "model": "someone/other-captioner", "kind": "caption"}
    assert [r[0] for r in conn.execute("SELECT item_id FROM job_item WHERE job_id = ?", (job_id,))] == [file_id]


def test_submit_skips_pictures_the_model_already_captioned_for_these_bytes(tmp_path):
    """A sweep is for what is missing: a file holding a caption from the
    configured model for its current bytes is not an item again; new
    bytes, another model, or `everything` put it back."""
    conn = fresh_schema()
    file_id = _one_picture(conn, tmp_path / "lib")
    sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()[0]
    model = settings.value(conn, "caption_model")
    assert runner.submit_annotate(conn, 0.0, models_dir="M") is not None

    derived.annotate(conn, file_id, "caption", "a square", model, "abc", sha, 1.0)
    assert runner.submit_annotate(conn, 2.0, models_dir="M") is None, "nothing left to caption"
    assert runner.submit_annotate(conn, 2.0, models_dir="M", everything=True) is not None

    settings.put(conn, "caption_model", "someone/other-captioner")
    assert runner.submit_annotate(conn, 3.0, models_dir="M") is not None, "another model has not spoken"
    settings.put(conn, "caption_model", model)
    conn.execute("UPDATE file SET content_sha256 = 'f' * 64 WHERE id = ?", (file_id,))
    assert runner.submit_annotate(conn, 4.0, models_dir="M") is not None, "new bytes, no caption for them"


def test_a_caption_model_that_is_no_repository_is_refused_at_submit(tmp_path):
    conn = fresh_schema()
    _one_picture(conn, tmp_path / "lib")
    settings.put(conn, "caption_model", "blip")
    with pytest.raises(ValueError, match="repository id"):
        runner.submit_annotate(conn, 0.0, models_dir="M")
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        held = connect.connect(client.app.state.db_path)
        try:
            settings.put(held, "caption_model", "blip")
            held.commit()
        finally:
            connect.close(held)
        refused = client.post("/jobs/annotate", json={})
        assert refused.status_code == 400, refused.text
        assert "repository id" in refused.text


def test_the_job_item_provisions_captions_and_records_what_was_said(tmp_path, monkeypatch):
    conn = fresh_schema()
    file_id = _one_picture(conn, tmp_path / "lib")
    asked: list = []
    fake = FakeCaptioner()

    def chosen(models_dir, model, *, provision):
        asked.append((models_dir, model, provision))
        return fake

    monkeypatch.setattr(captions, "captioner_for", chosen)
    monkeypatch.setattr(runner, "_CAPTIONERS", {})
    payload = {"models_dir": "M", "model": "fake/captioner", "kind": "caption"}

    runner._annotate_item(conn, file_id, payload, 5.0)
    runner._annotate_item(conn, file_id, payload, 6.0)

    assert asked == [("M", "fake/captioner", True)], "one captioner per payload, and only the job provisions"
    assert fake.seen == [(16, 16), (16, 16)]
    rows = conn.execute(
        "SELECT kind, text, model_id, model_version, computed_at FROM derived_annotation WHERE file_id = ?", (file_id,)
    ).fetchall()
    assert rows == [("caption", "a red square on a table", "fake/captioner", "v1", 6.0)], "the same model replaces"
    found = conn.execute(
        "SELECT file_id FROM annotation_fts JOIN derived_annotation a ON a.id = annotation_fts.rowid"
        " WHERE annotation_fts MATCH ?",
        ("square",),
    ).fetchall()
    assert found == [(file_id,)], "a caption is searchable by word"


def test_a_captioner_that_says_nothing_is_an_item_failure(tmp_path, monkeypatch):
    conn = fresh_schema()
    file_id = _one_picture(conn, tmp_path / "lib")
    monkeypatch.setattr(captions, "captioner_for", lambda *a, **k: FakeCaptioner(say="   "))
    monkeypatch.setattr(runner, "_CAPTIONERS", {})

    with pytest.raises(ValueError, match="said nothing"):
        runner._annotate_item(conn, file_id, {"models_dir": "M", "model": "fake/captioner"}, 0.0)
    assert conn.execute("SELECT count(*) FROM derived_annotation").fetchone()[0] == 0


def test_a_refused_captioner_is_held_so_every_item_fails_by_name(tmp_path, monkeypatch):
    conn = fresh_schema()
    file_id = _one_picture(conn, tmp_path / "lib")
    tried: list = []

    def refuses(*a, **k):
        tried.append(1)
        raise weights.Unprovisioned("no weights")

    monkeypatch.setattr(captions, "captioner_for", refuses)
    monkeypatch.setattr(runner, "_CAPTIONERS", {})
    payload = {"models_dir": "M", "model": "fake/captioner"}

    for _ in range(2):
        with pytest.raises(weights.Unprovisioned):
            runner._annotate_item(conn, file_id, payload, 0.0)
    assert len(tried) == 1


def test_a_repository_that_is_not_there_is_held_like_any_refusal(tmp_path, monkeypatch):
    """from_pretrained raises OSError for a missing or gated repository
    (transformers utils/hub.py); the job holds that too, so 22k items
    fail by one sentence instead of attempting 22k downloads."""
    conn = fresh_schema()
    file_id = _one_picture(conn, tmp_path / "lib")
    tried: list = []

    def gone(*a, **k):
        tried.append(1)
        raise OSError("Salesforce/nope is not a local folder and is not a valid model identifier")

    monkeypatch.setattr(captions, "captioner_for", gone)
    monkeypatch.setattr(runner, "_CAPTIONERS", {})
    payload = {"models_dir": "M", "model": "Salesforce/nope"}
    for _ in range(3):
        with pytest.raises(OSError, match="not a valid model identifier"):
            runner._annotate_item(conn, file_id, payload, 0.0)
    assert len(tried) == 1


def test_serving_never_downloads_the_captioner(monkeypatch):
    monkeypatch.setattr(weights, "hub_cached", lambda *a, **k: None)
    with pytest.raises(weights.Unprovisioned, match="run /jobs/annotate"):
        captions.BlipCaptioner("M", provision=False)
    with pytest.raises(ValueError, match="repository id"):
        captions.captioner_for("M", "not-a-repo")


def test_a_snapshot_counts_only_when_complete(monkeypatch, tmp_path):
    held = {"model.safetensors", "config.json", "preprocessor_config.json", "tokenizer_config.json"}

    def cached(repo, name, models_dir, revision=None):
        return str(tmp_path / "snap" / name) if name in held else None

    monkeypatch.setattr(captions, "hub_cached", cached)
    assert captions._cached_snapshot("M", captions.MODEL) is None, "no vocab, no load"
    held.add("vocab.txt")
    assert captions._cached_snapshot("M", captions.MODEL) == str(tmp_path / "snap")


def test_the_annotation_table_is_no_longer_reserved():
    assert "derived_annotation" not in policy.DERIVED_RESERVED


def test_the_caption_reaches_the_media_page_through_the_app(tmp_path, monkeypatch):
    root = tmp_path / "lib"
    root.mkdir()
    Image.new("RGB", (20, 12), (10, 200, 40)).save(root / "green.png")
    monkeypatch.setattr(captions, "captioner_for", lambda *a, **k: FakeCaptioner(say="a green field"))
    monkeypatch.setattr(runner, "_CAPTIONERS", {})
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        conn = connect.connect(client.app.state.db_path)
        try:
            settings.put(conn, "caption_model", "fake/captioner")  # the model the fake answers as
            conn.commit()
        finally:
            connect.close(conn)
        asked = client.post("/jobs/annotate", json={})
        assert asked.status_code in (200, 201, 202), asked.text
        conn = connect.connect(client.app.state.db_path)
        try:
            while runner.run_next(conn, "test-worker", time.time()) is not None:
                conn.commit()
            conn.commit()
            file_id = conn.execute("SELECT id FROM file").fetchone()[0]
            slug = naming.entity_slug(conn, file_id)[1]
            state = conn.execute("SELECT state FROM job WHERE kind = 'annotate'").fetchone()[0]
        finally:
            connect.close(conn)
        assert state == "done"
        page = client.get(f"/i/{slug}", headers={"accept": "text/html"})
        assert page.status_code == 200, page.text[:300]
        assert 'data-said-kind="caption"' in page.text
        assert "a green field" in page.text
        assert "fake/captioner" in page.text
        told = client.get(f"/i/{slug}", headers={"accept": "application/json"}).json()
        assert [a["text"] for a in told["said"]] == ["a green field"]
        console = client.get("/operations", headers={"accept": "text/html"}).text
        assert 'hx-post="/operations/jobs/annotate"' in console
        assert client.post("/jobs/annotate", json={}).status_code == 204, "everything is captioned"
        assert client.post("/jobs/annotate", json={"everything": True}).status_code in (200, 201, 202)
        # the grid says it on hover, and the machine answer carries it
        grid = client.get("/g", headers={"accept": "text/html"}).text
        assert 'title="a green field" data-said' in grid
        conn = connect.connect(client.app.state.db_path, read_only=True)
        try:
            assert derived.said_first(conn, [file_id]) == {file_id: "a green field"}
            assert derived.said_first(conn, []) == {}
        finally:
            connect.close(conn)


def _clip(path: pathlib.Path, seconds: int = 5, rate: int = 5) -> None:
    import av
    import numpy as np

    with av.open(str(path), "w") as container:
        stream = container.add_stream("h264", rate=rate)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        for n in range(seconds * rate):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), (n * 9) % 256, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_a_clip_is_captioned_whole_and_at_its_sampled_moments(tmp_path, monkeypatch):
    """A video gets the poster's caption on the file and one caption per
    sampled moment (db/sample.py), each saying which second it describes;
    the grid's one line is the file's, the page lists the moments."""
    from db import library, scan

    root = tmp_path / "lib"
    root.mkdir()
    _clip(root / "walk.mp4")
    conn = fresh_schema()
    root_id = library.add_root(conn, str(root), "library", 0.0)
    scan.scan(conn, root_id, str(root), 0.0)
    file_id = conn.execute("SELECT id FROM file WHERE kind = 'video'").fetchone()[0]

    class Moments(FakeCaptioner):
        def describe(self, image):
            self.seen.append(image.size)
            return f"frame {len(self.seen)}"

    fake = Moments()
    monkeypatch.setattr(captions, "captioner_for", lambda *a, **k: fake)
    monkeypatch.setattr(runner, "_CAPTIONERS", {})
    runner._annotate_item(conn, file_id, {"models_dir": "M", "model": "fake/captioner", "kind": "caption"}, 1.0)

    said = derived.said_about(conn, file_id)
    whole = [one for one in said if one["sample_id"] is None]
    moments = [one for one in said if one["sample_id"] is not None]
    assert len(whole) == 1, "the poster's caption is the file's"
    assert len(moments) >= 2, f"a five-second clip is sampled at more than one moment: {moments}"
    assert [one["offset_ms"] for one in moments] == sorted(one["offset_ms"] for one in moments)
    assert len({one["text"] for one in moments}) == len(moments), "every moment is its own caption"
    assert derived.said_first(conn, [file_id]) == {file_id: whole[0]["text"]}, (
        "the grid says the file's, not a moment's"
    )
    # the same item again replaces its own moments, never doubles them
    runner._annotate_item(conn, file_id, {"models_dir": "M", "model": "fake/captioner", "kind": "caption"}, 2.0)
    assert len(derived.said_about(conn, file_id)) == 1 + len(moments)


def test_a_moments_caption_is_a_door_into_the_clip(tmp_path, monkeypatch):
    """On the page a moment's caption carries the second it describes,
    and the page's script plays the clip from there."""
    root = tmp_path / "lib"
    root.mkdir()
    _clip(root / "walk.mp4")
    monkeypatch.setattr(captions, "captioner_for", lambda *a, **k: FakeCaptioner(say="a grey walk"))
    monkeypatch.setattr(runner, "_CAPTIONERS", {})
    with TestClient(app=build_app(str(tmp_path / "run"), worker=False)) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")
        conn = connect.connect(client.app.state.db_path)
        try:
            settings.put(conn, "caption_model", "fake/captioner")
            file_id = conn.execute("SELECT id FROM file WHERE kind = 'video'").fetchone()[0]
            runner._annotate_item(conn, file_id, {"models_dir": "M", "model": "fake/captioner", "kind": "caption"}, 1.0)
            conn.commit()
            slug = naming.entity_slug(conn, file_id)[1]
        finally:
            connect.close(conn)
        page = client.get(f"/i/{slug}", headers={"accept": "text/html"}).text
        assert 'data-said-seek="0"' in page, "the first sampled moment is the clip's start"
        assert "<video" in page
        assert page.count("data-said-seek=") >= 2
