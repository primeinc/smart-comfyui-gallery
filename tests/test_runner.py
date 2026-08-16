"""Tests for smartgallery_ai.runner: step selection, event ordering, the
partial-run contract (a run that does not store must not touch the scan
log), busy rejection, live progress forwarding, and per-step error
isolation. Only stub backends -- never real weights."""

from __future__ import annotations

import sqlite3
import threading

import pytest
from PIL import Image

from smartgallery_ai import AIConfig, RUBRIC_VERSION, runner
from smartgallery_ai.review import CriticBackend, StubCritic
from smartgallery_ai.schema import init_schema


def _make_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE files (id TEXT PRIMARY KEY, path TEXT NOT NULL UNIQUE, "
        "mtime REAL NOT NULL, name TEXT NOT NULL, type TEXT, "
        "workflow_prompt TEXT DEFAULT '')")
    init_schema(conn)
    conn.commit()
    conn.close()


def _add_file(db_path: str, file_id: str, path: str, prompt: str = "") -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO files (id, path, mtime, name, type, workflow_prompt) "
        "VALUES (?, ?, 1000.0, ?, 'image', ?)", (file_id, path, file_id, prompt))
    conn.commit()
    conn.close()


@pytest.fixture()
def env(tmp_path):
    db_path = str(tmp_path / "g.sqlite")
    _make_db(db_path)
    img_path = str(tmp_path / "img.png")
    img = Image.new("RGB", (64, 64), (200, 200, 200))
    for x in range(16, 48):
        for y in range(16, 48):
            img.putpixel((x, y), (255, 0, 0))  # StubCritic's red artifact
    img.save(img_path)
    _add_file(db_path, "f1", img_path, prompt="a red cube, a blue sphere")
    config = AIConfig(enabled=True, base_path=str(tmp_path), db_path=db_path,
                      cache_dir=str(tmp_path / "cache"), ephemeral_index=True)
    return config, db_path


def _drain(config, **kwargs):
    return list(runner.run_review(config, "f1", critic=StubCritic(), **kwargs))


def _steps_of(events):
    return [e["step"] for e in events]


def _by(events, step, status):
    return next(e for e in events if e["step"] == step and e["status"] == status)


# --- step selection ----------------------------------------------------------


def test_parse_steps_defaults_to_the_whole_pipeline():
    assert runner.parse_steps(None) == tuple(n for n, _ in runner.STEPS)
    assert runner.parse_steps("") == tuple(n for n, _ in runner.STEPS)


def test_parse_steps_preserves_pipeline_order_not_request_order():
    """'store' before 'validate' is not a legal pipeline; asking for it in
    that order must not produce it."""
    assert runner.parse_steps("store,validate,resolve") == ("resolve", "validate", "store")


def test_parse_steps_drops_unknown_names_rather_than_raising():
    """The spec arrives from a URL query parameter; a typo should run less,
    never 500."""
    assert runner.parse_steps("resolve,not_a_step") == ("resolve",)


# --- full run ----------------------------------------------------------------


def test_full_run_emits_every_step_in_order_and_stores(env):
    config, db_path = env
    events = _drain(config)
    assert events[0]["step"] == "run" and events[0]["status"] == "start"
    assert events[-1]["step"] == "run" and events[-1]["status"] == "done"

    ok_steps = [e["step"] for e in events if e["status"] == "ok"]
    assert ok_steps == ["resolve", "load", "critic", "validate", "store", "masks", "log"]

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM ai_reviews WHERE file_id='f1'").fetchone()[0] == 1
        scan = conn.execute(
            "SELECT input_key FROM ai_scan_log "
            "WHERE file_id='f1' AND kind='review'").fetchone()
        assert scan is not None
        assert scan[0] != "", "the scan must be keyed on the inputs it ran against"
    finally:
        conn.close()


def test_resolve_reports_the_prompt_it_will_score_against(env):
    config, _ = env
    detail = _by(_drain(config), "resolve", "ok")["detail"]
    assert detail["has_prompt"] is True
    assert "a red cube" in detail["prompt_preview"]
    assert detail["input_key"]


def test_validate_reports_scores_and_alignment_elements(env):
    config, _ = env
    detail = _by(_drain(config), "validate", "ok")["detail"]
    assert detail["quality"] == pytest.approx(8.0)  # one finding
    assert detail["prompt_alignment"] == pytest.approx(0.5)  # StubCritic: every other
    assert [e["text"] for e in detail["alignment"]] == ["a red cube", "a blue sphere"]


# --- partial runs ------------------------------------------------------------


def test_dry_run_stops_before_storing_and_leaves_the_db_untouched(env):
    config, db_path = env
    events = _drain(config, steps="resolve,load,critic,validate")
    assert [e["step"] for e in events if e["status"] == "ok"] == [
        "resolve", "load", "critic", "validate"]

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM ai_reviews").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM ai_scan_log").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_run_that_stored_nothing_never_writes_a_scan_log_row(env):
    """The scan log means 'this file is current'. Writing it for a run that
    produced no review would tell the worker to skip a file that has none --
    the exact lie this codebase already paid for."""
    config, db_path = env
    events = _drain(config, steps="resolve,log")
    assert _by(events, "log", "ok")["detail"]["skipped"]

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM ai_scan_log").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_step_missing_its_input_errors_and_halts_the_run(env):
    """Running 'critic' with no 'load' must name the failing step, not raise
    out of the generator."""
    config, _ = env
    events = _drain(config, steps="critic")
    err = _by(events, "critic", "error")
    assert "load did not run" in err["detail"]["error"]
    assert events[-1]["step"] == "run"


# --- live progress -----------------------------------------------------------


class _EmittingCritic(CriticBackend):
    """Reports protocol stages the way QwenVlCritic does."""

    model_id = "emitting-critic"
    model_version = "emit-v1"

    def review(self, img, prompt_text, rubric_version, negative_text=None):
        del img, prompt_text, rubric_version, negative_text
        self._emit("describe")
        self._emit("assess", grounding_margin=0.42)
        return {"quality_score": 5.0, "prompt_alignment_score": None,
                "summary": "s", "findings": [], "alignment": []}


def test_critic_protocol_stages_are_forwarded_as_events(env):
    config, _ = env
    events = list(runner.run_review(config, "f1", steps="resolve,load,critic",
                                    critic=_EmittingCritic()))
    assert "critic:describe" in _steps_of(events)
    assert _by(events, "critic:assess", "info")["detail"]["grounding_margin"] == 0.42
    # Stages must stream WHILE the step runs, i.e. before it completes --
    # that is the whole point. (The step's own 'start' event legitimately
    # precedes them, so compare against its 'ok'.)
    done_at = events.index(_by(events, "critic", "ok"))
    assert _steps_of(events).index("critic:describe") < done_at
    assert _steps_of(events).index("critic:assess") < done_at


def test_the_progress_sink_is_detached_after_the_run(env):
    """A sink left installed would push events into a dead queue for every
    later background review."""
    config, _ = env
    critic = _EmittingCritic()
    list(runner.run_review(config, "f1", steps="resolve,load,critic", critic=critic))
    assert critic.progress is None


def test_a_failing_progress_sink_cannot_break_the_review():
    """Observation must not be able to break the thing observed."""
    critic = _EmittingCritic()
    critic.progress = lambda *_a: 1 / 0
    payload = critic.review(None, None, RUBRIC_VERSION)
    assert payload["quality_score"] == 5.0


# --- concurrency -------------------------------------------------------------


def test_a_second_concurrent_run_is_refused_not_queued(env):
    config, _ = env
    started = threading.Event()
    release = threading.Event()

    class _BlockingCritic(CriticBackend):
        model_id = "blocking"
        model_version = "v1"

        def review(self, img, prompt_text, rubric_version, negative_text=None):
            del img, prompt_text, rubric_version, negative_text
            started.set()
            release.wait(timeout=5)
            return {"quality_score": 1.0, "prompt_alignment_score": None,
                    "summary": "s", "findings": []}

    first = runner.run_review(config, "f1", steps="resolve,load,critic",
                              critic=_BlockingCritic())
    consumed = []
    thread = threading.Thread(target=lambda: consumed.extend(first), daemon=True)
    thread.start()
    try:
        assert started.wait(timeout=5), "first run never reached the critic"
        with pytest.raises(runner.RunnerBusy):
            list(runner.run_review(config, "f1", critic=StubCritic()))
    finally:
        release.set()
        thread.join(timeout=5)

    # and the lock is released afterwards
    assert _drain(config)[-1]["status"] == "done"


def test_closing_the_generator_early_releases_the_lock(env):
    """A client that hangs up mid-stream must not wedge the runner: without
    release-on-close every later run would be refused until restart."""
    config, _ = env
    gen = runner.run_review(config, "f1", critic=StubCritic())
    next(gen)          # take the 'run start' event, leaving the lock held
    gen.close()
    assert _drain(config)[-1]["status"] == "done"
