"""Interactive review runner: one file, one review, reported live.

Background indexing is deliberately opaque -- it processes a backlog and
tells you a count. That is the wrong shape when you are looking at ONE image
and asking "what does the critic actually think, and why is it taking so
long". A ~200s review that reports nothing until it finishes is
indistinguishable from a hung one.

This module runs the same pipeline as the worker's review stage over a
single file, as an ordered list of NAMED, INDIVIDUALLY SELECTABLE steps,
emitting an event per step boundary plus one per VLM protocol stage. It is
the same code path, not a parallel implementation: the critic backend,
`validate_review_payload` and `store_review` are exactly the ones the worker
uses, so what you watch here is what the worker would have done.

Modularity is the point:
  - `STEPS` is data. Run all of them, or a prefix, or one.
  - Steps hand state to each other through `RunContext`, so a partial run
    is a legal run -- `resolve,load,critic` inspects a payload without
    touching the database.
  - Every step reports start/ok/error/skip, so a failure names the step it
    failed in rather than surfacing as one opaque traceback.

Concurrency: a module-level lock admits ONE run at a time. The critic owns
a multi-GB model; two concurrent interactive runs would contend for it and
make both slower than either alone, and interleave with the worker's own
review. Callers get `RunnerBusy` rather than a queue -- an interactive
action that silently waits behind a 200s job is a worse answer than "busy".
"""

from __future__ import annotations

import contextlib
import json
import queue
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from smartgallery_ai import RUBRIC_VERSION, AIConfig, schema
from smartgallery_ai import review as review_mod
from smartgallery_ai.worker import load_source_image, record_scan, stage_input_key
import logging

_logger = logging.getLogger(__name__)

__all__ = ["STEPS", "RunContext", "RunnerBusy", "parse_steps", "run_review"]

# Ordered pipeline. Each entry is (name, human label); the implementation is
# `_step_<name>`. Order is a dependency order -- 'store' cannot precede
# 'validate' -- and `parse_steps` preserves it regardless of request order.
STEPS = (
    ("resolve", "Resolve file and prompts"),
    ("load", "Decode source frame"),
    ("critic", "Run critic protocol"),
    ("validate", "Validate payload"),
    ("store", "Store review"),
    ("masks", "Segment findings and matches"),
    ("log", "Record scan"),
)

_STEP_NAMES = tuple(name for name, _ in STEPS)

# One interactive run at a time (see module docstring).
_RUN_LOCK = threading.Lock()

# How long a queue drain waits before emitting nothing, so a wedged critic
# still produces a heartbeat rather than a dead connection.
_DRAIN_TIMEOUT = 1.0


class RunnerBusy(RuntimeError):
    """Another interactive run holds the critic."""


@dataclass
class RunContext:
    """State threaded through the steps. Everything a later step needs from
    an earlier one lives here, so running a subset is well-defined: a step
    whose inputs are absent reports 'skip' instead of raising."""

    config: AIConfig
    file_id: str
    conn: sqlite3.Connection
    critic: object = None
    segmenter: object = None

    path: str | None = None
    file_type: str | None = None
    mtime: float | None = None
    prompt_text: str | None = None
    negative_text: str | None = None
    input_key: str = ""
    img: object = None
    payload: dict | None = None
    result: object = None
    review_id: int | None = None
    masks: int = 0
    findings: list = field(default_factory=list)


def parse_steps(spec: str | None) -> tuple:
    """Resolve a comma-separated step spec into pipeline order.

    Unknown names are dropped rather than raising: the caller is a URL query
    parameter, and a typo should run less, never 500. An empty or absent
    spec means the whole pipeline.
    """
    if not spec:
        return _STEP_NAMES
    wanted = {part.strip() for part in spec.split(",") if part.strip()}
    return tuple(name for name in _STEP_NAMES if name in wanted)


def _event(step: str, status: str, **detail) -> dict:
    return {"step": step, "status": status, "detail": detail, "at": time.time()}


# -- steps -------------------------------------------------------------------
# Each returns an iterable of extra detail dicts to merge into its 'ok'
# event, or raises. Raising is fine: run_review turns it into an 'error'
# event naming this step.


def _step_resolve(ctx: RunContext) -> dict:
    row = ctx.conn.execute("SELECT path, type, mtime FROM files WHERE id = ?", (ctx.file_id,)).fetchone()
    if row is None:
        raise LookupError(f"no such file: {ctx.file_id}")
    ctx.path, ctx.file_type, ctx.mtime = row[0], row[1], row[2]
    ctx.prompt_text, ctx.negative_text = review_mod.resolve_prompt_texts(ctx.conn, ctx.file_id)
    ctx.input_key = stage_input_key(ctx.prompt_text, ctx.negative_text, RUBRIC_VERSION)
    return {
        "path": ctx.path,
        "type": ctx.file_type,
        "has_prompt": ctx.prompt_text is not None,
        "prompt_preview": (ctx.prompt_text or "")[:160],
        "has_negative": ctx.negative_text is not None,
        "input_key": ctx.input_key,
    }


def _step_load(ctx: RunContext) -> dict:
    if ctx.path is None:
        raise LookupError("resolve did not run; nothing to load")
    ctx.img = load_source_image(ctx.path, ctx.file_type)
    if ctx.img is None:
        raise ValueError(f"no renderable frame in {ctx.path}")
    return {"size": list(ctx.img.size)}


def _step_critic(ctx: RunContext) -> dict:
    if ctx.critic is None:
        raise RuntimeError("no critic backend is available")
    if ctx.img is None:
        raise LookupError("load did not run; nothing to review")
    ctx.payload = ctx.critic.review(ctx.img, ctx.prompt_text, RUBRIC_VERSION, negative_text=ctx.negative_text)
    return {"raw_keys": sorted(ctx.payload or {})}


def _step_validate(ctx: RunContext) -> dict:
    if ctx.payload is None:
        raise LookupError("critic did not run; nothing to validate")
    ctx.result = review_mod.validate_review_payload(ctx.payload)
    ctx.findings = [
        {
            "type": f.type,
            "severity": f.severity,
            "confidence": f.confidence,
            "localizable": f.localizable,
            "description": f.description,
        }
        for f in ctx.result.findings
    ]
    return {
        "quality": ctx.result.quality_score,
        "prompt_alignment": ctx.result.prompt_alignment_score,
        "summary": ctx.result.summary,
        "findings": ctx.findings,
        "alignment": [
            {"ordinal": e.ordinal, "text": e.text, "satisfied": e.satisfied, "confidence": e.confidence}
            for e in ctx.result.alignment
        ],
    }


def _step_store(ctx: RunContext) -> dict:
    if ctx.result is None:
        raise LookupError("validate did not run; nothing to store")
    ctx.review_id = review_mod.store_review(
        ctx.conn,
        ctx.file_id,
        ctx.result,
        ctx.critic.model_id,
        ctx.critic.model_version,
        RUBRIC_VERSION,
        json.dumps(ctx.payload),
        ctx.mtime,
        time.time(),
    )
    return {"review_id": ctx.review_id}


def _step_masks(ctx: RunContext) -> dict:
    if ctx.review_id is None:
        raise LookupError("store did not run; nothing to segment")
    if ctx.segmenter is None:
        return {"skipped": "no segmenter backend"}
    generated = 0
    failures = []
    finding_ids = [
        r[0]
        for r in ctx.conn.execute(
            "SELECT finding_id FROM ai_review_findings WHERE review_id = ? AND localizable = 1 AND mask_path IS NULL",
            (ctx.review_id,),
        )
    ]
    for finding_id in finding_ids:
        try:
            review_mod.generate_finding_mask(
                ctx.conn, ctx.config.cache_dir, ctx.img, ctx.file_id, finding_id, ctx.segmenter
            )
            generated += 1
        except Exception as exc:
            _logger.debug("handled a failure in _step_masks", exc_info=True)
            failures.append(f"finding {finding_id}: {exc}")
    element_ids = [
        r[0]
        for r in ctx.conn.execute(
            "SELECT element_id FROM ai_review_alignment WHERE review_id = ? "
            "AND satisfied = 1 AND bbox_x IS NOT NULL AND mask_path IS NULL",
            (ctx.review_id,),
        )
    ]
    for element_id in element_ids:
        try:
            review_mod.generate_alignment_mask(
                ctx.conn, ctx.config.cache_dir, ctx.img, ctx.file_id, element_id, ctx.segmenter
            )
            generated += 1
        except Exception as exc:
            _logger.debug("handled a failure in _step_masks", exc_info=True)
            failures.append(f"element {element_id}: {exc}")
    ctx.masks = generated
    return {"generated": generated, "failures": failures}


def _step_log(ctx: RunContext) -> dict:
    """Mark the scan so the worker does not immediately redo this work.

    Only when the review actually stored: logging a scan for a run that
    stopped at 'critic' would tell the worker a file is current when no
    review exists for it, which is precisely the lie this codebase already
    paid for once."""
    if ctx.review_id is None:
        return {"skipped": "nothing stored; scan log untouched"}
    record_scan(ctx.conn, ctx.file_id, "review", ctx.critic, ctx.mtime, time.time(), len(ctx.findings), ctx.input_key)
    ctx.conn.commit()
    return {"input_key": ctx.input_key, "result_count": len(ctx.findings)}


_IMPL = {
    "resolve": _step_resolve,
    "load": _step_load,
    "critic": _step_critic,
    "validate": _step_validate,
    "store": _step_store,
    "masks": _step_masks,
    "log": _step_log,
}


def run_review(
    config: AIConfig,
    file_id: str,
    steps: str | None = None,
    critic=None,
    segmenter=None,
    connect: Callable | None = None,
) -> Iterator[dict]:
    """Run the review pipeline over one file, yielding JSON-ready events.

    Raises `RunnerBusy` immediately if another run holds the critic. The
    generator owns the lock for its lifetime, so a caller that abandons it
    must close it (a `for` loop over an SSE response does this).

    `critic`/`segmenter` may be injected; otherwise they are resolved from
    `config` the same way the worker resolves them.
    """
    if not _RUN_LOCK.acquire(blocking=False):
        raise RunnerBusy("another review run is in progress")
    try:
        yield from _run_locked(config, file_id, steps, critic, segmenter, connect)
    finally:
        _RUN_LOCK.release()


def _run_locked(config, file_id, steps, critic, segmenter, connect):
    selected = parse_steps(steps)
    yield _event("run", "start", file_id=file_id, steps=list(selected))

    opener = connect or (lambda: schema.connect(config.db_path))
    conn = opener()
    # The worker migrates on every cycle; this runner reaches a database
    # through its own connection and may be the FIRST thing to touch it
    # after an upgrade. Without this, storing dies on a table the running
    # process has never created -- observed live as
    # "no such table: ai_review_alignment" after a ~90s review had already
    # been computed and was then thrown away.
    schema.init_schema(conn)
    try:
        if critic is None:
            critic = review_mod.get_reviewer(config)
        if segmenter is None:
            segmenter = review_mod.get_segmenter_backend(config)
    except Exception as exc:
        # An unavailable backend is a legitimate outcome, reported as data.
        _logger.debug("handled a failure in _run_locked", exc_info=True)
        yield _event("run", "error", error=f"backend resolution failed: {exc}")
        conn.close()
        return

    ctx = RunContext(config=config, file_id=file_id, conn=conn, critic=critic, segmenter=segmenter)

    # The critic reports its protocol stages from inside a blocking call, so
    # it runs on a thread and this generator drains its events. Without that
    # the whole point -- watching a 200s review progress -- is lost.
    sink: queue.Queue = queue.Queue()
    if critic is not None:
        critic.progress = lambda stage, detail: sink.put(_event(f"critic:{stage}", "info", **detail))

    try:
        for name in selected:
            label = next(lbl for n, lbl in STEPS if n == name)
            yield _event(name, "start", label=label)
            started = time.monotonic()

            if name == "critic":
                # Only this step is slow enough to be worth threading.
                yield from _run_threaded(ctx, name, sink, started)
                continue

            try:
                detail = _IMPL[name](ctx) or {}
            except Exception as exc:
                _logger.debug("handled a failure in _run_locked", exc_info=True)
                yield _event(name, "error", error=str(exc), seconds=round(time.monotonic() - started, 3))
                break
            yield _event(name, "ok", seconds=round(time.monotonic() - started, 3), **detail)
    finally:
        if critic is not None:
            critic.progress = None
        if ctx.img is not None:
            with contextlib.suppress(Exception):
                ctx.img.close()
        conn.close()

    yield _event("run", "done", review_id=ctx.review_id, masks=ctx.masks)


def _run_threaded(ctx: RunContext, name: str, sink: queue.Queue, started: float):
    """Run one step on a thread, forwarding whatever it emits meanwhile."""
    box: dict = {}

    def work():
        try:
            box["detail"] = _IMPL[name](ctx) or {}
        except Exception as exc:  # carried out, re-reported as an event
            _logger.debug("handled a failure in work", exc_info=True)
            box["error"] = str(exc)
        finally:
            sink.put(None)  # sentinel: the step is finished

    thread = threading.Thread(target=work, name=f"review-{name}", daemon=True)
    thread.start()

    while True:
        try:
            item = sink.get(timeout=_DRAIN_TIMEOUT)
        except queue.Empty:
            # Heartbeat: proves to the client that a long stage is alive
            # rather than a stalled connection.
            yield _event(name, "waiting", seconds=round(time.monotonic() - started, 1))
            continue
        if item is None:
            break
        yield item

    thread.join()
    seconds = round(time.monotonic() - started, 3)
    if "error" in box:
        yield _event(name, "error", error=box["error"], seconds=seconds)
        return
    yield _event(name, "ok", seconds=seconds, **box.get("detail", {}))
