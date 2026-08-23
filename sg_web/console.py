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
from typing import Literal

from litestar import get

from db import ledger
from sg_web.wire import Wire


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

#: What a ledger event can be, per db/schema.sql job_event.type.
EventType = Literal[
    "job.submitted",
    "job.claimed",
    "job.reclaimed",
    "job.paused",
    "job.cancel_requested",
    "job.cancelled",
    "job.done",
    "job.failed",
    "item.started",
    "item.done",
    "item.failed",
    "item.observed",
    "phase.started",
    "phase.progress",
    "phase.finished",
    "checkpoint.changed",
    "worker.turn_failed",
]

#: How loudly, per db/schema.sql job_event.severity.
Severity = Literal["info", "warning", "error"]


class Reported(Wire):
    """What a handler said, whether or not it became a row.

    The event's own columns plus what this module adds: `text` is the
    event said in words, and `condition` is the handful the console must
    keep apart -- an expected per-item failure reads differently from a
    defect in the worker. Both are derived here, so neither is a column.

    Everything except the id, because a report from inside an item has
    none until the item settles.
    """

    job_id: int
    at: float
    type: EventType
    item_id: int | None
    phase: str | None
    severity: Severity
    message: str | None
    #: whatever the event carried, with every secret-named key replaced
    #: (db/ledger.py redacted)
    data: dict[str, object] | None
    text: str
    #: None for the events that announce nothing the console reacts to
    condition: str | None


class Event(Reported):
    """A committed ledger row.

    The same shape crosses the HTTP pages and the /ws/events feed, because
    it is the same event; a second spelling for the socket would be the
    same contract written twice.
    """

    id: int


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


class EventFrame(Event):
    """A committed ledger row, live."""

    frame: Literal["event"]


class PendingFrame(Reported):
    """A report from INSIDE the item a handler is on.

    Reported, not Event, and that is the whole point of the split: there
    is no id, because it is not a row yet and may never be one --
    db/runner.py Report lands at the item boundary. A browser narrowed to
    this arm cannot reach for an id that was never sent. The console holds
    the newest per job and drops it when any other kind of event settles
    the item.
    """

    frame: Literal["pending"]


class BacklogFrame(Wire):
    """Everything the client missed, read from the rows on connect.

    Sent before the live subscription is drained, so a row committed
    during the read is queued behind it rather than lost, and a row that
    lands in both arrives as the same id twice -- the client keeps one.
    """

    frame: Literal["backlog"]
    events: list[Event]
    after: int
    last_id: int


#: What arrives on /ws/events. Discriminated on `frame`, so a browser
#: narrows to the arm it is handling and cannot read `events` off a
#: single event or an id off a pending report.
#:
#: Not an OpenAPI path -- a socket has none -- so this is carried into the
#: contract by sg_web/wire.py socket_frames(), which puts the union in the
#: document's components. The browser's type is generated from that, never
#: written twice.
Frame = EventFrame | PendingFrame | BacklogFrame


@get("/ws/events/frames", sync_to_thread=False)
def socket_frames() -> Frame:
    """Every frame /ws/events sends, carried into the document.

    A socket has no path OpenAPI can describe, so its frames would never
    reach the generated types and the browser would go back to
    hand-written interfaces beside `JSON.parse` -- the exact duplication
    this contract exists to remove. Declaring the union as one route's
    answer puts all three arms in components, and openapi-typescript
    generates the browser's union from them.

    It answers the empty backlog rather than raising: a route that exists
    only to be read by a generator is still a route somebody can request,
    and one that 500s when they do is a trap. Nothing needs to call it --
    the frames arrive on the socket.
    """
    return BacklogFrame(frame="backlog", events=[], after=0, last_id=0)


def _said(event: Mapping) -> dict:
    """Everything a row and a report share, said in words."""
    held = event.get("data")
    scrubbed = None if held is None else ledger.redacted(held)
    # The column is any valid JSON, so a rebuild rather than a cast: an
    # event whose data is a list or a scalar carries no named facts, and
    # saying so is more honest than asserting it is an object.
    data: dict[str, object] | None = None
    if isinstance(scrubbed, dict):
        data = {str(k): v for k, v in scrubbed.items()}
    return {
        "job_id": event["job_id"],
        "at": event["at"],
        "type": event["type"],
        "item_id": event.get("item_id"),
        "phase": event.get("phase"),
        "severity": event.get("severity", "info"),
        "message": event.get("message"),
        "data": data,
        "text": describe(event),
        "condition": CONDITIONS.get(event["type"]),
    }


def envelope(event: Mapping) -> Event:
    """A committed row as the feed and the pages carry it."""
    return Event(id=event["id"], **_said(event))


def event_frame(event: Mapping) -> EventFrame:
    """A committed row, on its way to the socket."""
    return EventFrame(frame="event", id=event["id"], **_said(event))


def pending_frame(event: Mapping) -> PendingFrame:
    """A report from inside an item, on its way to the socket."""
    return PendingFrame(frame="pending", **_said(event))
