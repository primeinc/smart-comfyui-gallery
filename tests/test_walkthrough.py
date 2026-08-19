"""The per-file pipeline walkthrough (AAA).

One file, every stage, and for each stage that did not run, the reason. The
contract the page depends on:
  - a stage that has not run is PENDING only when nothing prevents it;
  - a stage whose backend could not load is BLOCKED and carries the loader's
    OWN message, so the operator learns which weights file is missing rather
    than the word "unavailable";
  - a stage that will never run for this file is N/A, not pending -- an
    audio file is not waiting for a face detector;
  - "scanned, found nothing" is DONE, never PENDING.

Model-free: stub backends and real tiny PNGs only.
"""

from __future__ import annotations

import sqlite3

import pytest
from flask import Flask
from PIL import Image

from smartgallery_ai import RUBRIC_VERSION, AIConfig, backends, walkthrough
from smartgallery_ai.embedders import BackendUnavailable
from smartgallery_ai.schema import init_schema
from smartgallery_ai.service import _index_one_file, create_ai_blueprint, set_worker

_PREFIX = "/aidam"


def _make_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE files (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            mtime REAL NOT NULL,
            name TEXT NOT NULL,
            type TEXT,
            size INTEGER DEFAULT 0,
            workflow_prompt TEXT DEFAULT ''
        )
        """
    )
    init_schema(conn)
    return conn


def _cfg(tmp_path, **overrides) -> AIConfig:
    base = {
        "enabled": True,
        "base_path": str(tmp_path),
        "db_path": str(tmp_path / "gallery.sqlite"),
        "models_dir": str(tmp_path / "models"),
        "cache_dir": str(tmp_path / "cache"),
        "ephemeral_index": True,
        "semantic_backend": "stub",
        "visual_backend": "stub",
        "face_backend": "none",
        "critic_backend": "none",
        "segmenter_backend": "none",
    }
    base.update(overrides)
    return AIConfig(**base)


def _add_image(conn, tmp_path, file_id: str, prompt: str = "", mtime: float = 1000.0) -> None:
    path = str(tmp_path / f"{file_id}.png")
    Image.new("RGB", (16, 16), (40, 90, 200)).save(path)
    conn.execute(
        "INSERT INTO files (id, path, mtime, name, type, size, workflow_prompt) VALUES (?, ?, ?, ?, 'image', 64, ?)",
        (file_id, path, mtime, file_id, prompt),
    )
    conn.commit()


def _by_key(result) -> dict:
    return {stage["key"]: stage for stage in result["stages"]}


# --- states -------------------------------------------------------------------


def test_an_unindexed_file_reads_as_pending_not_blocked(tmp_path):
    """Nothing has run, but nothing prevents it either: every runnable stage
    is PENDING and names the action that would run it."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    _add_image(conn, tmp_path, "fresh")

    stages = _by_key(walkthrough.walk(conn, cfg, "fresh"))
    conn.close()

    assert stages["indexed"]["state"] == walkthrough.DONE
    for key in ("hashes", "semantic", "visual"):
        assert stages[key]["state"] == walkthrough.PENDING, key
        assert stages[key]["action"] == "index", key


def test_indexed_stages_report_what_they_stored(tmp_path):
    """After the inline index path runs, the same stages are DONE and carry
    the model identity that produced them."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    _add_image(conn, tmp_path, "done_file")
    row = conn.execute("SELECT * FROM files WHERE id = 'done_file'").fetchone()
    _index_one_file(conn, cfg, row, force=False)

    stages = _by_key(walkthrough.walk(conn, cfg, "done_file"))
    conn.close()

    assert stages["hashes"]["state"] == walkthrough.DONE
    assert stages["semantic"]["state"] == walkthrough.DONE
    assert stages["semantic"]["evidence"]["model_id"] == "stub-semantic"
    assert stages["visual"]["evidence"]["dim"] == 64
    # Derived capabilities follow from the rows the stages wrote.
    assert stages["near_dup"]["state"] == walkthrough.DONE
    assert stages["similar"]["state"] == walkthrough.DONE


def test_a_type_no_stage_can_render_is_not_pending(tmp_path):
    """An audio file is not waiting for an embedder; it is N/A, and saying
    'pending' would promise a result that can never arrive."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    conn.execute(
        "INSERT INTO files (id, path, mtime, name, type) VALUES ('sound', '/g/sound.wav', 1000.0, 'sound', 'audio')"
    )
    conn.commit()

    stages = _by_key(walkthrough.walk(conn, cfg, "sound"))
    conn.close()

    assert stages["semantic"]["state"] == walkthrough.NA
    assert "audio" in stages["semantic"]["detail"]
    assert stages["faces"]["state"] == walkthrough.NA


def test_a_blocked_stage_carries_the_loaders_own_message(tmp_path, monkeypatch):
    """'unavailable' is not actionable. The row repeats what the backend
    said -- which file it could not find -- and the command that fixes it."""
    cfg = _cfg(tmp_path, visual_backend="auto")

    def _missing(_config):
        raise BackendUnavailable("dinov2 weights not found at /models/dinov2-small")

    monkeypatch.setitem(backends._KINDS, "visual", backends._KINDS["visual"]._replace(resolve=_missing))
    backends.reset()

    conn = _make_db(cfg.db_path)
    _add_image(conn, tmp_path, "blocked_file")
    stages = _by_key(walkthrough.walk(conn, cfg, "blocked_file"))
    conn.close()

    visual = stages["visual"]
    assert visual["state"] == walkthrough.BLOCKED
    assert visual["blocked_reason"] == "dinov2 weights not found at /models/dinov2-small"
    assert visual["fix"] == "python -m smartgallery_ai provision visual"
    # The stage that is fine is unaffected.
    assert stages["semantic"]["state"] == walkthrough.PENDING


def test_scanned_with_no_faces_is_done_not_pending(tmp_path):
    """The distinction ai_scan_log exists for: a detector looked and found
    nothing, which is a result, not the absence of one."""
    cfg = _cfg(tmp_path, face_backend="stub")
    conn = _make_db(cfg.db_path)
    _add_image(conn, tmp_path, "no_faces")
    row = conn.execute("SELECT * FROM files WHERE id = 'no_faces'").fetchone()
    _index_one_file(conn, cfg, row, force=False)

    stages = _by_key(walkthrough.walk(conn, cfg, "no_faces"))
    conn.close()

    assert stages["faces"]["state"] == walkthrough.DONE
    assert stages["faces"]["detail"] == "scanned, no faces found"
    assert stages["clustering"]["state"] == walkthrough.NA


def test_a_null_alignment_on_a_prompted_file_is_blocked_not_absent(tmp_path):
    """A file WITH a prompt whose alignment is null is stuck, not exempt --
    the panel renders that as 'no prompt to compare with', which is false."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    _add_image(conn, tmp_path, "stuck", prompt="a red car at night")
    conn.execute(
        "INSERT INTO ai_reviews (file_id, rubric_version, model_id, model_version, quality_score, "
        "prompt_alignment_score, source_mtime, computed_at) VALUES (?, ?, 'critic', 'v1', 0.8, NULL, 1000.0, 1.0)",
        ("stuck", RUBRIC_VERSION),
    )
    conn.commit()

    stages = _by_key(walkthrough.walk(conn, cfg, "stuck"))
    conn.close()

    alignment = stages["alignment"]
    assert alignment["state"] == walkthrough.BLOCKED
    assert "did not parse" in alignment["blocked_reason"]
    assert alignment["action"] == "review"


def test_no_prompt_makes_alignment_not_applicable(tmp_path):
    """The same null score on a file with no prompt is correct, not stuck."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    _add_image(conn, tmp_path, "promptless")

    stages = _by_key(walkthrough.walk(conn, cfg, "promptless"))
    conn.close()

    assert stages["alignment"]["state"] == walkthrough.NA
    assert stages["metadata"]["state"] == walkthrough.NA


def test_a_recorded_scan_with_no_review_is_reported_as_stuck(tmp_path):
    """A scan-log row with nothing stored tells the worker the file is
    current, so it never retries. That is a blocked stage, not a pending one."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    _add_image(conn, tmp_path, "ghost")
    conn.execute(
        "INSERT INTO ai_scan_log (file_id, kind, model_id, model_version, source_mtime, scanned_at, result_count) "
        "VALUES ('ghost', 'review', 'critic', 'v1', 1000.0, 1.0, 0)"
    )
    conn.commit()

    stages = _by_key(walkthrough.walk(conn, cfg, "ghost"))
    conn.close()

    assert stages["review"]["state"] == walkthrough.BLOCKED
    assert "will not retry" in stages["review"]["blocked_reason"]


def test_the_layer_being_off_is_said_once_per_stage(tmp_path):
    """Every model-backed stage reports the same actionable cause rather
    than each inventing its own explanation."""
    cfg = _cfg(tmp_path, enabled=False)
    conn = _make_db(cfg.db_path)
    _add_image(conn, tmp_path, "off_file")

    stages = _by_key(walkthrough.walk(conn, cfg, "off_file"))
    conn.close()

    for key in ("hashes", "semantic", "visual", "faces"):
        assert stages[key]["state"] == walkthrough.BLOCKED, key
        assert stages[key]["fix"] == "set ENABLE_AI_DAM=true and restart", key


def test_every_stage_says_what_it_actually_computes(tmp_path):
    """A state without a mechanism is not an explanation. Each row carries
    the model, dimension, metric, or algorithm behind it -- named concretely
    enough to tell you whether its answer is the one you wanted."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    _add_image(conn, tmp_path, "explained", prompt="a red car at night")
    stages = _by_key(walkthrough.walk(conn, cfg, "explained"))
    conn.close()

    for key, stage in stages.items():
        assert stage.get("does"), f"{key} reports a state with no mechanism behind it"

    # The two embedding spaces must not read as interchangeable: the reason
    # both exist is that they answer different questions.
    assert "512-d" in stages["semantic"]["does"]
    assert "text tower" in stages["semantic"]["does"]
    assert "384-d" in stages["visual"]["does"]
    assert "No text side" in stages["visual"]["does"]
    # Index type AND metric, since "similar" is meaningless without them.
    assert "IndexFlatIP" in stages["similar"]["does"]
    assert "cosine" in stages["similar"]["does"]
    assert "IndexBinaryFlat" in stages["near_dup"]["does"]
    assert "Hamming" in stages["near_dup"]["does"]
    assert "Chinese Whispers" in stages["clustering"]["does"]


def test_an_unknown_file_has_no_walkthrough(tmp_path):
    """None, so the route can answer 404 rather than inventing stages."""
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    assert walkthrough.walk(conn, cfg, "nope") is None
    conn.close()


# --- route --------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    cfg = _cfg(tmp_path)
    conn = _make_db(cfg.db_path)
    _add_image(conn, tmp_path, "routed")
    conn.close()
    set_worker(None)
    app = Flask(__name__)
    app.register_blueprint(create_ai_blueprint(cfg), url_prefix=_PREFIX)
    return app.test_client()


def test_the_route_answers_every_stage_for_a_known_file(client):
    """The page's whole payload comes from one request."""
    res = client.get(f"{_PREFIX}/walkthrough/routed")
    assert res.status_code == 200
    body = res.get_json()
    assert body["file"]["file_id"] == "routed"
    assert body["worker"]["running"] is False
    keys = [stage["key"] for stage in body["stages"]]
    assert keys[:3] == ["indexed", "metadata", "hashes"]
    # counts summarise exactly the list the page renders
    assert sum(body["counts"].values()) == len(body["stages"])


def test_the_route_404s_an_unknown_file(client):
    """Indistinguishable from a policy-hidden file, like every per-file route."""
    assert client.get(f"{_PREFIX}/walkthrough/missing").status_code == 404
