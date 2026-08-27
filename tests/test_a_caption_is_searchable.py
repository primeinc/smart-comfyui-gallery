"""A caption is text somebody can find: the words a model said about a
picture enter the same rank fusion semantic search runs, as one more
ranking named `captions`. A library nothing has captioned lists no such
participant; one that has, says so even when no caption matches.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from PIL import Image

from db import connect, derived, retrieval, scan, settings
from tests.staging import NOW
from vision import semantic

SCHEMA = pathlib.Path(__file__).resolve().parents[1] / "db" / "schema.sql"
CLIP = ("openclip", "ViT-B-32", "laion2b_s34b_b79k")


class Asks:
    def encode_query(self, phrase):
        probe = np.zeros(16, dtype=np.float32)
        probe[0] = 1.0
        return probe


def _shelf(tmp_path, cosines: dict[str, float]):
    """Three pictures in one 16-d space, ranked by the cosine each is
    given; the ids by name."""
    root = tmp_path / "pics"
    root.mkdir()
    for i, name in enumerate(cosines):
        Image.new("RGB", (8, 8), (20 * i, 40, 60)).save(root / f"{name}.png")
    conn = connect.memory()
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,?,'library',0)", (str(root),))
    scan.scan(conn, 1, root, NOW)
    ids = {name: file_id for file_id, name in conn.execute("SELECT id, replace(name, '.png', '') FROM file")}
    shas = dict(conn.execute("SELECT id, content_sha256 FROM file"))
    space = semantic.space(*CLIP, 16)
    for axis, (name, cosine) in enumerate(cosines.items(), start=1):
        v = np.zeros(16, dtype=np.float32)
        v[0] = cosine
        v[axis] = np.sqrt(1.0 - cosine * cosine)
        derived.record_embedding(conn, ids[name], space, v, shas[ids[name]], NOW)
    settings.put(conn, "semantic_model", "ViT-B-32/laion2b_s34b_b79k")
    conn.commit()
    return conn, ids, space.key


@pytest.fixture
def asks(monkeypatch):
    monkeypatch.setattr(semantic, "encoder", lambda *args, **kwargs: Asks())


def test_a_library_nothing_captioned_lists_no_captions_participant(tmp_path, asks):
    conn, ids, clip = _shelf(tmp_path, {"a": 0.9, "b": 0.8, "c": 0.7})
    found = retrieval.query(conn, str(tmp_path), "bicycle", 3, NOW)
    assert found["participants"] == [clip]
    assert found["contributors"] == [clip]
    assert found["missing"] == {}
    assert [row["file_id"] for row in found["results"]] == [ids["a"], ids["b"], ids["c"]]


def test_a_caption_lifts_the_file_it_describes(tmp_path, asks):
    conn, ids, clip = _shelf(tmp_path, {"a": 0.9, "b": 0.8, "c": 0.7})
    sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (ids["b"],)).fetchone()[0]
    derived.annotate(conn, ids["b"], "caption", "a red bicycle leaning on a wall", "m", "1", sha, NOW)

    found = retrieval.query(conn, str(tmp_path), "bicycle", 3, NOW)

    assert found["participants"] == [clip, "captions"]
    assert found["contributors"] == [clip, "captions"]
    assert found["missing"] == {}
    assert [row["file_id"] for row in found["results"]] == [ids["b"], ids["a"], ids["c"]], (
        "two rankings agree on b; the space alone ranked it second"
    )
    assert set(found["results"][0]["sources"]) == {clip, "captions"}
    assert found["results"][0]["sources"]["captions"]["rank"] == 1


def test_a_phrase_no_caption_mentions_is_unmatched_not_a_contributor(tmp_path, asks):
    conn, ids, clip = _shelf(tmp_path, {"a": 0.9, "b": 0.8})
    sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (ids["b"],)).fetchone()[0]
    derived.annotate(conn, ids["b"], "caption", "a red bicycle", "m", "1", sha, NOW)

    found = retrieval.query(conn, str(tmp_path), "helicopter", 3, NOW)

    assert found["participants"] == [clip, "captions"]
    assert found["contributors"] == [clip]
    assert found["missing"] == {}, "a word match that matched nothing is not a space that could not answer"
    assert found["unmatched"] == {"captions": "no caption mentions a word of the phrase in this scope"}
    # and a scope the captioned file sits outside of is the same verdict
    scoped = retrieval.query(conn, str(tmp_path), "bicycle", 3, NOW, allowed={ids["a"]})
    assert scoped["contributors"] == [clip]
    assert "captions" in scoped["unmatched"]
    assert [row["file_id"] for row in scoped["results"]] == [ids["a"]]


def test_captions_alone_answer_when_no_space_is_provisioned(tmp_path, monkeypatch):
    """Degraded is an answer: with the encoder unprovisioned, the
    caption ranking still answers and the space is named as missing."""
    conn, ids, clip = _shelf(tmp_path, {"a": 0.9, "b": 0.8})
    sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (ids["a"],)).fetchone()[0]
    derived.annotate(conn, ids["a"], "caption", "a brass trumpet", "m", "1", sha, NOW)

    def refuses(*args, **kwargs):
        raise LookupError("ViT-B-32/laion2b_s34b_b79k is not provisioned; run /jobs/embed")

    monkeypatch.setattr(semantic, "encoder", refuses)
    found = retrieval.query(conn, str(tmp_path), "trumpet", 3, NOW)
    assert found["contributors"] == ["captions"]
    assert clip in found["missing"]
    assert [row["file_id"] for row in found["results"]] == [ids["a"]]


def test_ranking_by_annotation_is_one_row_per_present_file_best_first(tmp_path, asks):
    conn, ids, _ = _shelf(tmp_path, {"a": 0.9, "b": 0.8, "c": 0.7})
    shas = dict(conn.execute("SELECT id, content_sha256 FROM file"))
    derived.annotate(conn, ids["a"], "caption", "a dog on a beach", "m", "1", shas[ids["a"]], NOW)
    derived.annotate(conn, ids["a"], "tag", "dog", "m", "1", shas[ids["a"]], NOW)
    derived.annotate(conn, ids["b"], "caption", "a cat on a sofa", "m", "1", shas[ids["b"]], NOW)
    derived.annotate(conn, ids["c"], "caption", "a dog", "m", "1", shas[ids["c"]], NOW)
    conn.execute("UPDATE file SET missing_since = ? WHERE id = ?", (NOW, ids["c"]))

    ranked = derived.rank_by_annotation(conn, "dog beach", 10)
    assert [file_id for file_id, _ in ranked] == [ids["a"]], "one row for a, none for the missing c, none for the cat"
    assert derived.rank_by_annotation(conn, "   ", 10) == []
    quoted = derived.rank_by_annotation(conn, 'dog "quoted', 10)
    assert [file_id for file_id, _ in quoted] == [ids["a"]], "quotes in a phrase do not break the match"


def test_the_configured_models_caption_is_the_one_a_cell_says(tmp_path, asks):
    """Two models captioned the same picture: the grid's one line is the
    configured model's, whatever its name sorts to; with no setting match
    the first by name stands."""
    conn, ids, _ = _shelf(tmp_path, {"a": 0.9})
    sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (ids["a"],)).fetchone()[0]
    derived.annotate(conn, ids["a"], "caption", "first by name", "aaa/early", "1", sha, NOW)
    derived.annotate(conn, ids["a"], "caption", "the configured one", "zzz/blip", "1", sha, NOW)
    assert derived.said_first(conn, [ids["a"]]) == {ids["a"]: "first by name"}
    assert derived.said_first(conn, [ids["a"]], prefer="zzz/blip") == {ids["a"]: "the configured one"}
    assert derived.said_first(conn, [ids["a"]], prefer="nobody/spoke") == {ids["a"]: "first by name"}


def test_a_caption_of_older_bytes_ranks_nothing(tmp_path, asks):
    """The staleness contract the sweep keeps, retrieval keeps too: a
    file replaced on disk does not answer for the picture it used to be."""
    conn, ids, clip = _shelf(tmp_path, {"a": 0.9, "b": 0.8})
    sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (ids["b"],)).fetchone()[0]
    derived.annotate(conn, ids["b"], "caption", "a red bicycle", "m", "1", sha, NOW)
    assert [f for f, _ in derived.rank_by_annotation(conn, "bicycle", 10)] == [ids["b"]]
    conn.execute("UPDATE file SET content_sha256 = ? WHERE id = ?", ("e" * 64, ids["b"]))
    assert derived.rank_by_annotation(conn, "bicycle", 10) == []
    found = retrieval.query(conn, str(tmp_path), "bicycle", 3, NOW)
    assert found["contributors"] == [clip], "the caption is of bytes that are gone; it entered no fusion"
