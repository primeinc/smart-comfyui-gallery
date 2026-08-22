"""How the Operations Console says each ledger event in words.

A presentation Adapter over db/ledger.py rows: one renderer per event
type, held to the vocabulary by a contract test -- an event the ledger
can write and this module cannot say is a failing build, so "all
events" on the console means all of them. The raw row always rides
beside the words (`envelope`); the words never replace it.

The one judgement made here that the row does not spell: an item
failure and a worker defect are different conditions and are labelled
as such (`condition`). The runner already split them (db/runner.py
ITEM_FAILURES); the console must not fold them back together.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from db import ledger


def _seconds(value) -> str:
    if value is None:
        return "?"
    value = float(value)
    if value < 60:
        return f"{value:.1f}s"
    if value < 3600:
        return f"{int(value // 60)}m {int(value % 60):02d}s"
    return f"{int(value // 3600)}h {int((value % 3600) // 60):02d}m"


def _item(event: Mapping) -> str:
    return f"item {event['item_id']}" if event.get("item_id") is not None else "item"


def _submitted(e: Mapping) -> str:
    d = e.get("data") or {}
    total = d.get("total")
    return f"queued {d.get('kind', '')}" + (f" with {total:,} items" if total is not None else " (no enumerable items)")


def _claimed(e: Mapping) -> str:
    d = e.get("data") or {}
    return (
        f"{d.get('owner', 'a worker')} claimed the job · attempt {d.get('attempt', '?')} · fence {d.get('fence', '?')}"
    )


def _reclaimed(e: Mapping) -> str:
    d = e.get("data") or {}
    return (
        f"{d.get('owner', 'a worker')} reclaimed the job after its lease lapsed · attempt {d.get('attempt', '?')}"
        f" · fence {d.get('fence', '?')}"
    )


def _paused(e: Mapping) -> str:
    d = e.get("data") or {}
    why = {"budget": "the turn's budget", "stop": "a shutdown"}.get(str(d.get("why") or ""), "the worker")
    return f"paused by {why} after {d.get('did', '?')} items; the lease is released for the next turn"


def _cancel_requested(e: Mapping) -> str:
    return "cancel asked for; the runner stops at the next item boundary"


def _cancelled(e: Mapping) -> str:
    d = e.get("data") or {}
    return f"stopped cooperatively at an item boundary · {d.get('did', '?')} items this turn · cancelled"


def _done(e: Mapping) -> str:
    d = e.get("data") or {}
    return f"done · {d.get('did', '?')} items this turn · {d.get('failed', 0)} failed · {_seconds(d.get('seconds'))}"


def _failed(e: Mapping) -> str:
    d = e.get("data") or {}
    return f"the job failed: {d.get('error') or e.get('message') or 'no reason recorded'}"


def _item_started(e: Mapping) -> str:
    return f"{_item(e)} started"


def _item_done(e: Mapping) -> str:
    return f"{_item(e)} done"


def _item_failed(e: Mapping) -> str:
    d = e.get("data") or {}
    return f"{_item(e)} failed: {d.get('error') or e.get('message')} · the job continues"


def _item_observed(e: Mapping) -> str:
    d = dict(e.get("data") or {})
    name = d.pop("name", e.get("message") or "observation")
    facts = ", ".join(f"{k} {v}" for k, v in d.items())
    return f"{_item(e)} · {name}" + (f": {facts}" if facts else "")


def _phase_started(e: Mapping) -> str:
    d = e.get("data") or {}
    facts = ", ".join(f"{k} {v}" for k, v in d.items())
    return f"{_item(e)} · phase {e.get('phase')} started" + (f" ({facts})" if facts else "")


def _phase_progress(e: Mapping) -> str:
    d = e.get("data") or {}
    total = d.get("total")
    spelled = f"{d.get('done', '?')}" + (f" / {total}" if total is not None else "") + f" {d.get('unit', '')}"
    return f"{_item(e)} · {e.get('phase') or 'phase'} · {spelled.strip()}"


def _phase_finished(e: Mapping) -> str:
    return f"{_item(e)} · phase {e.get('phase')} finished"


def _checkpoint(e: Mapping) -> str:
    d = e.get("data") or {}
    return f"checkpoint moved to {d.get('checkpoint')!r}" + (
        f" · done {d['done']}" if d.get("done") is not None else ""
    )


def _turn_failed(e: Mapping) -> str:
    d = e.get("data") or {}
    lease = d.get("lease_until")
    return (
        f"WORKER TURN CRASHED on {_item(e)}: {d.get('exception', 'Exception')}: {d.get('error', '')}"
        f" · attempt {d.get('attempt', '?')} lost · fence {d.get('fence', '?')}"
        + (" · the lease will lapse and the job is reclaimable" if lease is not None else "")
    )


#: type -> the words. Held equal to db/ledger.py TYPES by contract.
RENDERINGS: dict[str, Callable[[Mapping], str]] = {
    "job.submitted": _submitted,
    "job.claimed": _claimed,
    "job.reclaimed": _reclaimed,
    "job.paused": _paused,
    "job.cancel_requested": _cancel_requested,
    "job.cancelled": _cancelled,
    "job.done": _done,
    "job.failed": _failed,
    "item.started": _item_started,
    "item.done": _item_done,
    "item.failed": _item_failed,
    "item.observed": _item_observed,
    "phase.started": _phase_started,
    "phase.progress": _phase_progress,
    "phase.finished": _phase_finished,
    "checkpoint.changed": _checkpoint,
    "worker.turn_failed": _turn_failed,
}

#: The condition an event announces, for the two the console must keep
#: apart: an expected per-item failure and a defect in the worker.
CONDITIONS: dict[str, str] = {
    "item.failed": "item-failure",
    "worker.turn_failed": "worker-defect",
    "job.failed": "job-failure",
    "job.reclaimed": "reclaim",
    "job.cancel_requested": "cancel",
    "job.cancelled": "cancel",
}


#: The kinds a handler reports from INSIDE an item: a committed row of
#: any other kind settles the item, and the live report with it.
INSIDE_ITEM = frozenset({"phase.started", "phase.progress", "phase.finished", "item.observed"})


def inside_item(type_: str) -> bool:
    return type_ in INSIDE_ITEM


#: What each schema job kind does, in words, shown beside the raw kind
#: (db/schema.sql job.kind CHECK; the contract test holds the two equal).
KINDS: dict[str, str] = {
    "scan": "read every file's metadata",
    "hash": "verify every file's bytes",
    "embed": "embed every picture for search",
    "detect_faces": "detect faces",
    "cluster_faces": "cluster faces into people",
    "sample_frames": "sample frames from every video",
    "annotate": "caption every picture",
    "remix": "remix pictures",
    "zip": "pack files for download",
    "context": "interpret every file's time and place",
    "events": "propose events",
    "story_plan": "plan a story",
    "embed_prompts": "embed every prompt",
}

#: The 'hash' kind's modes, told apart by the payload's `derive`
#: (db/runner.py _hash_item).
HASH_MODES: dict[str | None, str] = {
    None: KINDS["hash"],
    "perceptual": "fingerprint every picture",
    "thumbs": "render every missing thumbnail",
    "groups": "group perceptual copies",
}


def describe_kind(kind: str, derive: str | None = None) -> str:
    """The human line for a job: its kind's words, or its hash mode's."""
    if kind == "hash":
        return HASH_MODES.get(derive, f"hash, mode {derive}")
    return KINDS.get(kind, kind.replace("_", " "))


def describe(event: Mapping) -> str:
    """The words for one event. An unknown type is a defect in this
    module, not a quiet blank -- it raises."""
    try:
        render = RENDERINGS[event["type"]]
    except KeyError as unknown:
        raise ValueError(f"no console rendering for event type {event.get('type')!r}") from unknown
    return render(event)


def envelope(event: Mapping) -> dict:
    """The event as the feed and the pages carry it: the row, its words,
    its condition. `pending` survives when the runner marked it so."""
    told = {**event, "text": describe(event), "condition": CONDITIONS.get(event["type"])}
    if event.get("data") is not None:
        told["data"] = ledger.redacted(event["data"])
    return told
