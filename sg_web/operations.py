"""The runtime's own surface: the console, roots, sweeps, the worker,
clustering.

Everything here already exists as a machine route in sg_web/app.py --
POST /roots, POST /roots/{id}/scan, POST /jobs/*, POST /settings/{key},
POST /clusterings/choose. Those keep their JSON shape for machines. This
module is the BROWSER's way in: one page, forms that post url-encoded
(what an htmx form sends without help), fragments back in place. It is
registered as one Litestar Router under /operations, so every operational
page and form shares that prefix and whatever policy the layer grows
(litestar-org/litestar@v2.24.0 docs/usage/routing/overview.rst "Routers";
litestar/router.py Router.__init__ for the layered kwargs).

The console is the expert depth (db/inspecting.py): the health strip,
the job matrix, one job's inspector, the ledger as a tape. It reads the
operations read model and the ledger, never `jobs.active` widened -- the
shell's list stays tiny. Live delivery is /ws/events (sg_web/app.py);
the routes here are what a cold load and a gap-fill read.

Nothing operational is offered anywhere else: the gallery header asks
questions about media, this page runs the library.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import sqlite3
import time
import typing
from collections.abc import Callable

from litestar import MediaType, Request, Router, get, post
from litestar.datastructures import State
from litestar.exceptions import ClientException, HTTPException, NotFoundException
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import FromPath, FromQuery, URLEncodedBody
from litestar.response import Response, Template
from litestar.status_codes import HTTP_500_INTERNAL_SERVER_ERROR, HTTP_503_SERVICE_UNAVAILABLE

from db import (
    authored,
    connect,
    derived,
    inspecting,
    jobs,
    ledger,
    library,
    pages,
    prompts,
    runner,
    scan,
    scheduling,
    settings,
    verdicts,
)
from sg_web import console, home
from sg_web.presenting import VARIES, wants_json
from sg_web.submitting import submitted
from sg_web.wire import Wire

_logger = logging.getLogger(__name__)

Launcher = Callable[[State, object], list[int]]


def _weights(state: State, conn) -> str:
    return str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))


def _behind(missing: dict[str, int]) -> dict[str, int]:
    """The sweeps with work outstanding, in the order the chain runs them.

    Named sweeps rather than a total, because there is no honest total: a
    file missing both a reading and a caption is ONE file and TWO sweeps,
    and adding the counts would say two.

    Anything not in `LAUNCHERS` still comes through, at the end. A sweep
    added tomorrow must show as work nobody has done, not vanish from the
    one place that says what is left.
    """
    ordered = [kind for kind in LAUNCHERS if missing.get(kind)]
    ordered += [kind for kind, n in missing.items() if n and kind not in LAUNCHERS]
    return {kind: missing[kind] for kind in ordered}


def _ingest(state: State, conn) -> list[int]:
    job_id = runner.submit_ingest(conn, time.time())
    return [] if job_id is None else [job_id]


def _verify(state: State, conn) -> list[int]:
    return [runner.submit_verify(conn, time.time())]


def _phash(state: State, conn) -> list[int]:
    job_id = runner.submit_phash(conn, time.time())
    return [] if job_id is None else [job_id]


def _dupes(state: State, conn) -> list[int]:
    return [runner.submit_dupes(conn, time.time())]


def _thumbs(state: State, conn) -> list[int]:
    job_id = runner.submit_thumbs(conn, time.time(), thumbs_dir=str(home.thumbs_dir(pathlib.Path(state.home))))
    return [] if job_id is None else [job_id]


def _embed(state: State, conn) -> list[int]:
    return runner.submit_embed(conn, time.time(), models_dir=_weights(state, conn))


def _embed_prompts(state: State, conn) -> list[int]:
    return prompts.submit_embed(conn, time.time(), models_dir=_weights(state, conn))


def _faces(state: State, conn) -> list[int]:
    cache = str(home.thumbs_dir(pathlib.Path(state.home))) if settings.flag(conn, "thumbnail_precache") else None
    job_id = runner.submit_faces(conn, time.time(), models_dir=_weights(state, conn), thumbs_dir=cache)
    return [] if job_id is None else [job_id]


def _annotate(state: State, conn) -> list[int]:
    job_id = runner.submit_annotate(conn, time.time(), models_dir=_weights(state, conn))
    return [] if job_id is None else [job_id]


def _cluster(state: State, conn) -> list[int]:
    return [runner.submit_cluster(conn, time.time())]


def _context(state: State, conn) -> list[int]:
    job_id = runner.submit_context(conn, time.time())
    return [] if job_id is None else [job_id]


def _events(state: State, conn) -> list[int]:
    return [runner.submit_events(conn, time.time())]


#: What the page can start, in the order a library is usually built:
#: find files, read them, fingerprint, thumbnail, group copies, embed, detect faces,
#: cluster, interpret time, group events. Each launcher returns the job
#: ids it queued -- the same db/runner.py entry points the JSON routes use.
def _catch_up(state: State, conn) -> list[int]:
    """Every step of the derivation chain, gated in order.

    The twelve buttons beside this one are each honest and each require
    knowing which to press and when: `cluster` over an unembedded
    library settles `done` having clustered nothing, so pressing them
    out of order does not look like a mistake. This is the order, and
    the runner holds it (db/runner.py `catch_up`).
    """
    cache = str(home.thumbs_dir(pathlib.Path(state.home))) if settings.flag(conn, "thumbnail_precache") else None
    return runner.catch_up(conn, time.time(), models_dir=_weights(state, conn), thumbs_dir=cache)


#: What starts a sweep, in the order the buttons appear. `catch_up`
#: first, and it is not one sweep among thirteen: it is the answer to
#: the question the other twelve force a person to answer themselves --
#: which of these, and in what order. The rest stay because knowing
#: exactly what you want is a real thing to want.
LAUNCHERS: dict[str, tuple[str, Launcher]] = {
    "catch_up": ("bring the library up to date, in order", _catch_up),
    "ingest": ("read the metadata of every file not yet read", _ingest),
    "verify": ("verify every file's bytes", _verify),
    "phash": ("fingerprint every picture not yet fingerprinted", _phash),
    "thumbs": ("render every missing thumbnail", _thumbs),
    "dupes": ("group perceptual copies", _dupes),
    "embed": ("embed every picture not yet embedded", _embed),
    "embed_prompts": ("embed every prompt", _embed_prompts),
    "faces": ("detect faces in every picture not yet looked at", _faces),
    "cluster": ("cluster faces into people", _cluster),
    "annotate": ("caption every picture not yet captioned", _annotate),
    "context": ("interpret every file not yet interpreted", _context),
    "events": ("propose events", _events),
}


def _one(job_id: int | None) -> list[int]:
    """A submit's answer as the launcher's list: None is "nothing to do"."""
    return [] if job_id is None else [job_id]


def _phash_again(state: State, conn) -> list[int]:
    return _one(runner.submit_phash(conn, time.time(), everything=True))


def _faces_again(state: State, conn) -> list[int]:
    cache = str(home.thumbs_dir(pathlib.Path(state.home))) if settings.flag(conn, "thumbnail_precache") else None
    return _one(
        runner.submit_faces(conn, time.time(), models_dir=_weights(state, conn), thumbs_dir=cache, everything=True)
    )


def _embed_again(state: State, conn) -> list[int]:
    return runner.submit_embed(conn, time.time(), models_dir=_weights(state, conn), everything=True)


def _annotate_again(state: State, conn) -> list[int]:
    return _one(runner.submit_annotate(conn, time.time(), models_dir=_weights(state, conn), everything=True))


def _context_again(state: State, conn) -> list[int]:
    return _one(runner.submit_context(conn, time.time(), everything=True))


#: The sweeps that are for what is missing, each with its "all of it
#: again" -- the second button beside the first, never a hidden flag.
def _ingest_again(state: State, conn) -> list[int]:
    return _one(runner.submit_ingest(conn, time.time(), everything=True))


AGAIN: dict[str, Launcher] = {
    "ingest": _ingest_again,
    "phash": _phash_again,
    "faces": _faces_again,
    "embed": _embed_again,
    "annotate": _annotate_again,
    "context": _context_again,
}


def _roots(conn) -> list[dict]:
    """Every root with its live reachability -- the probe only, no write
    (db/library.py probe_roots); the JSON /roots route is what records."""
    return [{"id": root_id, "path": path, "online": online} for root_id, path, online in library.probe_roots(conn)]


def _page_context(state: State) -> dict:
    """What the page renders from cold.

    The ledger is NOT serialized in here. The page states one number --
    `last_event_id`, the head it read -- and the browser reads ids at or
    below it from /operations/events/before and asks the socket to resume
    above it, so the two halves meet exactly once and no event can fall
    between them. The head is taken from the overview already assembled
    rather than read again, so the number the page hands the browser and
    the number the health strip shows are the same read.
    """
    now = time.time()
    conn = connect.connect(state.db_path, read_only=True)
    try:
        console_state = _state_of(state, conn, now)
        runs = pages.standings(conn)
        return {
            "roots": _roots(conn),
            "settings": settings.snapshot(conn),
            "clusterings": runs,
            "primary": _primary(runs),
            "schedules": [one.model_dump(mode="json") for one in _schedules(conn)],
            "runnable": list(scheduling.RUNNABLE),
            "launchers": [
                {"kind": kind, "label": label, "again": kind in AGAIN} for kind, (label, _) in LAUNCHERS.items()
            ],
            "notice": None,
            "behind": _behind(console_state.overview.coverage.missing),
            "measured": [one.model_dump(mode="json") for one in _recorded()],
            **console_state.model_dump(mode="json"),
            "last_event_id": console_state.overview.ledger.last_id,
            "now": now,
        }
    finally:
        connect.close(conn)


@get("/", sync_to_thread=True)
def operations_page(state: State) -> Template:
    return Template(template_name="operations.html", context=_page_context(state), headers=VARIES)


@get("/overview", sync_to_thread=True)
def overview(state: State) -> OperationsState:
    """The health strip and the matrix, from the rows: what the console
    re-reads after a reconnect, and what a machine asks."""
    now = time.time()
    conn = connect.connect(state.db_path, read_only=True)
    try:
        return _state_of(state, conn, now)
    finally:
        connect.close(conn)


def _state_of(state: State, conn, now: float) -> OperationsState:
    """The console's whole read, assembled once.

    The cold page and the JSON route are the same facts, so they are the
    same assembly: the page used to build a parallel dict and mutate it
    afterwards, which is how the two drifted into disagreeing about what
    `worker` carries.
    """
    held = inspecting.overview(conn, now, models_dir=_weights(state, conn))
    thread = getattr(state, "worker_thread", None)
    worker = WorkerHealth(
        **held["worker"],
        thread_alive=bool(thread is not None and thread.is_alive()),
        thread=getattr(thread, "name", None),
    )
    rows = [_matrix_row(state, row) for row in inspecting.matrix(conn, now)]
    return OperationsState(
        overview=Overview(
            now=held["now"],
            coverage=Coverage(**held["coverage"]),
            worker=worker,
            queue=QueueHealth(**held["queue"]),
            ledger=LedgerHealth(**held["ledger"]),
        ),
        matrix=rows,
        collections=_collapsed(rows),
        judged=_judged(conn),
    )


def _judged(conn) -> WhatTheThumbsSay:
    """The verdicts, added up (db/verdicts.py)."""
    return WhatTheThumbsSay(
        producers=[
            ProducerJudged(
                model_id=one.model_id,
                model_version=one.model_version,
                kind=one.kind,
                right=one.right,
                wrong=one.wrong,
                unsure=one.unsure,
                judged=one.judged,
                wrong_share=one.wrong_share,
                needs=one.needs,
            )
            for one in verdicts.by_producer(conn)
        ],
        contests=[
            ProducerContest(
                kind=one.kind,
                shared=one.shared,
                enough=one.enough,
                wrong={f"{model_id}@{model_version}": n for (model_id, model_version), n in one.wrong.items()},
            )
            for one in verdicts.contests(conn)
        ],
        corrected=[
            ProducerCorrected(
                model_id=one.model_id,
                model_version=one.model_version,
                corrections=one.corrections,
                people=one.people,
            )
            for one in verdicts.corrections(conn)
        ],
        floor=verdicts.ENOUGH,
    )


#: Columns whose stored form is not the fact the wire states, so they are
#: named beside the row's straight copy rather than swept in with it.
_TRANSLATED = frozenset({"cancel_requested"})


def _matrix_row(state: State, row: dict) -> MatrixRow:
    """One job row as the console is told it.

    The live phase is beside the row, not in it: what a worker is doing
    inside the item it is on lives in process memory until the item
    settles (sg_web/app.py live_reports), so a row read from the database
    cannot carry it and a job running in another process has none.
    """
    live = getattr(state, "live_reports", {})
    held = live.get(row["id"]) if row["state"] == "running" else None
    return MatrixRow(
        **{one: row[one] for one in MatrixRow.model_fields if one in row and one not in _TRANSLATED},
        cancel_requested=bool(row["cancel_requested"]),
        what=console.describe_kind(row["kind"], row.get("derive"), row.get("total"), row.get("path")),
        live=None
        if not held
        else LiveReport(phase=held.get("phase"), type=held["type"], text=held["text"], item_id=held.get("item_id")),
    )


class Coverage(Wire):
    """Is the library done: present files, and what each missing-only
    sweep still has to do.

    `missing` is keyed by sweep name and `embed_spaces` by space key, so
    both are open by nature -- a space added tomorrow is a key nobody
    edits a model for. The VALUES are counts, which is the fact.
    """

    files: int
    missing: dict[str, int]
    embed_spaces: dict[str, int] | None = None


class WorkerHealth(Wire):
    """Whether anything is actually turning the crank."""

    enabled: bool
    owners: list[str]
    #: a job is running AND its heartbeat is inside the lease. Running
    #: with a stale heartbeat is a worker that died holding one.
    working: bool
    last_heartbeat: float | None
    heartbeat_age: float | None
    lease_seconds: float
    #: whether THIS process is turning the crank. The rows cannot say it:
    #: a job may be running under another owner entirely, and a thread
    #: that died still leaves its row running until the lease lapses.
    thread_alive: bool
    thread: str | None


class QueueHealth(Wire):
    """What is waiting, what is moving, and what settled in a day."""

    queued: int
    running: int
    oldest_queued_age: float | None
    oldest_running_age: float | None
    #: job state -> how many finished in that state in the last 24h; the
    #: keys are the job-state vocabulary, counted only where non-zero
    settled_24h: dict[str, int]


class LedgerHealth(Wire):
    """Where the event ledger stands."""

    last_id: int
    events: int


class Overview(Wire):
    """The health strip: the console's answer to "is anything wrong"."""

    now: float
    coverage: Coverage
    worker: WorkerHealth
    queue: QueueHealth
    ledger: LedgerHealth


class Lifecycle(Wire):
    """What a job's numbers mean, derived rather than stored.

    Every field here is computed from the row and the clock, so none of it
    can disagree with the row -- which is why the console reads these
    instead of doing the arithmetic itself.
    """

    elapsed: float | None
    queue_wait: float
    fraction: float | None
    pending: int | None
    succeeded: int
    rate: float | None
    eta: float | None
    cancellation: typing.Literal["cancelled", "requested", "not_requested"]
    heartbeat_age: float | None
    lease_remaining: float | None
    lease_expired: bool


class LiveReport(Wire):
    """What only the running process knows: the phase inside the item a
    worker is on right now. The row cannot hold it until the item
    settles, so it lives in process memory (sg_web/app.py live_reports)
    and is null for every job that is not running here."""

    phase: str | None
    type: str
    text: str
    item_id: int | None


class MatrixRow(Wire):
    """One job as the matrix shows it."""

    id: int
    kind: jobs.JobKind
    state: jobs.JobState
    #: has somebody asked this job to stop. SQLite stores the answer as
    #: 0 or 1 because it has no boolean; that is storage, not the fact,
    #: and the wire says the fact (_matrix_row does the translating).
    cancel_requested: bool
    total: int | None
    done_count: int
    failed_count: int
    attempt: int
    owner: str | None
    fence: int | None
    heartbeat_at: float | None
    lease_until: float | None
    created_at: float
    started_at: float | None
    finished_at: float | None
    error: str | None
    #: which derivation a sweep of this kind is doing, when its kind
    #: covers several
    derive: str | None
    derived: Lifecycle
    settled: bool
    #: the sweep said in words
    what: str
    live: LiveReport | None
    #: what this job is a step OF, when it is one. None for a job that
    #: stands alone, which is every job submitted on its own.
    collection: str | None = None
    #: the step it waits on. A queued job with this set is not stalled --
    #: it is waiting its turn, which is a different thing and the console
    #: must not show them alike.
    after_id: int | None = None


class FaceVector(Wire):
    """One detected face, and enough about it to be checked."""

    sha256: str
    captured_at: float | None
    det_score: float | None
    dim: int | None
    region: Box
    embedding: list[float]


class FaceSpace(Wire):
    """One immutable space, its centroid, and the faces in it.

    Everything above `faces` is what stops the numbers being opaque: the
    producer that made them, the preprocessing that fed it, the metric
    they are compared under and how many of them there are. A bare
    vector without these recreates exactly the dependency an export is
    meant to escape.
    """

    space: str
    representation: str
    dimensions: int
    metric: str
    producer: str
    producer_version: str
    preprocess: str
    preprocess_version: str
    spec_hash: str
    centroid: list[float]
    faces: list[FaceVector]


class FaceExport(Wire):
    """One person's face representation, by space."""

    person: str
    name: str | None
    spaces: list[FaceSpace]


#: The benchmarks that measure THROUGHPUT, and the job each one is
#: about. Three of twenty-three, and the number is the point: the rest
#: of `benchmarks/results/` are calibration and evidence documents with
#: nothing in common -- no key is shared by all of them -- and four
#: carry real filesystem paths, which is not something a page should
#: put on screen. These three agree on a shape because one script
#: (benchmarks/job_phases.py) writes them.
MEASURED = {"job_scan.json": "scan", "embed_job.json": "embed", "job_annotate.json": "annotate"}

#: Where those files land. Absent in an installed copy that ships no
#: benchmarks, and then the panel simply does not appear.
RESULTS = pathlib.Path(__file__).resolve().parent.parent / "benchmarks" / "results"


class Phase(Wire):
    """Where one job's time went."""

    phase: str
    ms: float
    share: float


class Measured(Wire):
    """A RECORDING of one job's throughput, not a live reading.

    `recorded_at` is the file's own mtime and is not decoration: these
    files carry no timestamp of their own, and a rate shown without one
    is a claim about code that may have changed since. The surface says
    when, and says that it was measured rather than observed.
    """

    kind: str
    items_per_second: float
    items: int | None
    wall_ms: float | None
    cores_busy: float | None
    cpu_count: int | None
    recorded_at: float
    phases: list[Phase]


def _recorded(root: pathlib.Path = RESULTS) -> list[Measured]:
    """The last recorded throughput of each measured job."""
    told: list[Measured] = []
    for name, kind in MEASURED.items():
        one = root / name
        if not one.is_file():
            continue
        try:
            held = json.loads(one.read_text(encoding="utf-8"))
        except (OSError, ValueError) as unreadable:
            # A half-written or hand-edited result is not worth taking
            # the console down for; it is simply not shown.
            _logger.info("%s could not be read as a benchmark result (%s)", one, unreadable)
            continue
        rate = held.get("items_per_second")
        if rate is None:
            continue
        phases = sorted(
            ((k, float(v.get("total_ms", 0.0))) for k, v in (held.get("by_phase") or {}).items()),
            key=lambda pair: -pair[1],
        )
        whole = sum(ms for _, ms in phases) or 1.0
        told.append(
            Measured(
                kind=kind,
                items_per_second=float(rate),
                items=held.get("pictures") or held.get("files"),
                wall_ms=held.get("wall_ms"),
                cores_busy=held.get("cores_busy"),
                cpu_count=held.get("cpu_count"),
                recorded_at=one.stat().st_mtime,
                phases=[Phase(phase=k, ms=ms, share=ms / whole) for k, ms in phases[:4]],
            )
        )
    return sorted(told, key=lambda one: one.kind)


class Named(Wire):
    """A person, by the address their pictures name them at."""

    slug: str
    name: str


class Shelved(Wire):
    """One collection, and the collection it sits under -- both by slug,
    so a shelf rebuilds from this alone."""

    slug: str
    name: str
    kind: str
    parent: str | None


class Box(Wire):
    """Where in the picture, in fractions of it."""

    x: float
    y: float
    w: float
    h: float


class Appears(Wire):
    """A person in a picture, or expressly NOT in it.

    `stance` rides along because a negative is a claim here rather than
    the absence of one: "not her" survives a rebuild and constrains the
    next clustering, and an export that dropped it would hand back a
    library that starts making the same mistake again.
    """

    person: str
    stance: str
    region: Box | None


class Keyword(Wire):
    """One word on a picture: the identity a filter is built from, and
    the spelling somebody typed. Both, because a reader that kept only
    the label would have to re-fold it and might fold it differently."""

    tag: str
    label: str


class Picture(Wire):
    """What somebody said about one photograph, named by its bytes."""

    sha256: str
    name: str
    rating: int | None
    favorite: bool
    place: str | None
    collections: list[str]
    tags: list[Keyword]
    people: list[Appears]


class Authored(Wire):
    """Everything a person told this library about their own pictures."""

    people: list[Named]
    collections: list[Shelved]
    pictures: list[Picture]


class VerdictRow(Wire):
    """One judgement, as an eval set carries it.

    A producer identity, what kind of claim it was, the verdict, the
    bytes it was about and when. No path, no file name, no folder, no
    person's name, no embedding -- so "this model got these 41 wrong"
    leaves the machine with none of the media leaving with it.

    `sha256` is null where the picture is gone: `feedback`'s pointers
    are ON DELETE SET NULL on purpose, so a judgement outlives the
    derived thing it judged and still says a model was wrong.

    `note` is free text somebody typed and is null unless the export was
    asked for it by name. The KEY is always here because a route's
    answer has to describe itself; the VALUE is the part withheld.
    """

    judged: str
    verdict: str
    model_id: str | None
    model_version: str | None
    annotation_kind: str | None
    sha256: str | None
    other_sha256: str | None
    at: float
    note: str | None


class ProducerJudged(Wire):
    """What one producer has been told about its own claims.

    `wrong_share` is null below the floor, and `needs` says how many
    more judgements it would take. Null, never zero: a zero would read
    as "never wrong" for a model nobody has judged yet.
    """

    model_id: str
    model_version: str
    kind: str
    right: int
    wrong: int
    unsure: int
    judged: int
    wrong_share: float | None
    needs: int


class ProducerContest(Wire):
    """Two producers over the files BOTH were judged on.

    The only comparison a biased verdict set supports. People judge what
    they happen to look at and reach for `wrong` far sooner than for
    `right`, so a raw error rate is a statement about which pictures got
    opened; restricted to the shared files, the person, the day and the
    pictures are the same on both sides.
    """

    kind: str
    shared: int
    enough: bool
    #: "model_id@model_version" -> how many of the shared files it got
    #: wrong. A mapping rather than two fields, because which producers
    #: exist is a fact about the library and not about this contract.
    wrong: dict[str, int]


class ProducerCorrected(Wire):
    """How many of a face producer's attributions people took back.

    A COUNT, and there is deliberately no share beside it. A correction
    is recorded only when somebody says a person is NOT in a picture, so
    these verdicts are 100% `wrong` by construction -- nobody stops to
    confirm a face that is simply right. There is no denominator, and a
    percentage here would be the most confidently wrong number in the
    application.

    It costs the person nothing: they were correcting the picture
    anyway, and the correction is the same click.
    """

    model_id: str
    model_version: str
    corrections: int
    #: how many distinct people those corrections were about
    people: int


class WhatTheThumbsSay(Wire):
    """The verdicts, added up -- and what they refuse to say.

    Deliberately no headline verdict on any model. An error rate over a
    sample nobody drew at random is not a measurement, and printing one
    beside a name is how a number gets used to make a decision it cannot
    support.
    """

    producers: list[ProducerJudged]
    contests: list[ProducerContest]
    #: Face producers whose attributions were corrected by hand. Counted
    #: rather than rated, and kept apart from `producers` for that
    #: reason: the two cannot go in one table without the reader
    #: comparing a rate against a tally.
    corrected: list[ProducerCorrected]
    #: how many verdicts a producer needs before a rate is shown at all
    floor: int


class Collapsed(Wire):
    """A collection as one row: its steps, added up.

    The console showed ten rows where a person had asked for one thing.
    Each was honest and none was the answer -- somebody watching a
    catch-up wants to know whether the catch-up is going well, and the
    eight rows made that a sum they had to do themselves while the rows
    moved.

    So the steps stop being rows and become a bar. They are still THERE,
    under it: the row that failed is the one somebody needs, and
    collapsing must not be the same as hiding.
    """

    name: str
    #: every step, in the order each gates the next
    steps: list[int]
    #: units done and units expected across the steps that state a total.
    #: A step with no total (a walk, which finds its work by doing it)
    #: contributes to neither, so the bar measures what can be measured
    #: rather than inventing a denominator.
    done: int
    total: int | None
    #: how many steps have settled, and how many there are
    settled: int
    #: what the collection as a whole is doing. `failed` if any step
    #: failed -- one failed step means the collection did not do what it
    #: was asked, whatever the others managed.
    state: str
    #: the step running right now, when one is
    running: int | None = None
    #: the first step that failed, which is the row somebody wants
    failed: int | None = None


def _collapsed(rows: list[MatrixRow]) -> list[Collapsed]:
    """Fold every collection's steps into one row each, oldest first.

    Order by the first step, not by the newest: a collection is one act
    and its place in the list is where that act began.
    """
    held: dict[str, list[MatrixRow]] = {}
    for row in rows:
        if row.collection is not None:
            held.setdefault(row.collection, []).append(row)
    made = []
    for name, steps in held.items():
        steps.sort(key=lambda one: one.id)
        totals = [one.total for one in steps if one.total is not None]
        failed = next((one.id for one in steps if one.state == "failed"), None)
        running = next((one.id for one in steps if one.state == "running"), None)
        settled = sum(1 for one in steps if one.settled)
        made.append(
            Collapsed(
                name=name,
                steps=[one.id for one in steps],
                done=sum(one.done_count for one in steps),
                total=sum(totals) if totals else None,
                settled=settled,
                state=_collection_state(steps, failed, running, settled),
                running=running,
                failed=failed,
            )
        )
    made.sort(key=lambda one: one.steps[0])
    return made


def _collection_state(steps: list[MatrixRow], failed: int | None, running: int | None, settled: int) -> str:
    """One word for what a collection is doing.

    `failed` wins over everything, including steps still to come: a
    catch-up whose embed step died did not do what it was asked, and
    saying "running" while the rest of the chain drains would be the
    console agreeing with the queue instead of with the person.
    """
    if failed is not None:
        return "failed"
    if running is not None:
        return "running"
    if settled == len(steps):
        return "cancelled" if all(one.state == "cancelled" for one in steps) else "done"
    return "queued"


class OperationsState(Wire):
    """What the console re-reads after a reconnect, and what a machine
    asks: the health strip and every job worth showing."""

    overview: Overview
    matrix: list[MatrixRow]
    #: The collections among those rows, each as one row. The steps stay
    #: in `matrix` -- this is a fold over them, not a replacement, and a
    #: client that wants the step that failed still has it.
    collections: list[Collapsed]
    #: What this library has been told about its own models. Empty until
    #: somebody judges something, which is honest: no verdicts is not a
    #: model with nothing wrong with it.
    judged: WhatTheThumbsSay


class NamedItem(Wire):
    """One unit of a job, named and addressed when the kind's units are
    files and the row still exists (db/inspecting.py _named_item); the
    bare id otherwise."""

    id: int
    name: str | None
    href: str | None


class InspectedItem(NamedItem):
    """A unit of a job with where it stands."""

    state: jobs.ItemState
    error: str | None


class ItemPage(Wire):
    """A page of one job's units, in unit order.

    `next_after` is the cursor for the page after this one and null when
    this page reached the end, so a reader knows whether it has them all
    without asking again. Never folded into the detail: a 22,000-item job
    is read a page at a time.
    """

    items: list[InspectedItem]
    next_after: int | None


class FailedItem(NamedItem):
    """A unit the job could not do. The job continued past it -- an item
    failure and a defect in the worker are different conditions."""

    error: str | None


class JobTarget(Wire):
    """The entity a job is about, when it is about one rather than the
    whole library. `kind` and `slug` are null when the row it named is
    gone."""

    id: int
    kind: str | None
    slug: str | None


class Attempt(Wire):
    """One claim, reclaim or pause, as the ledger recorded it.

    `data` is the event's own payload, nested rather than spread up to
    this level. What the runner records about a turn is historical and
    open by design -- owner, attempt, fence, lease_until and resumed for a
    claim; did, failed and why for a pause -- and a row written before a
    key existed does not carry it. Naming those as fields here would put
    an old job's inspector one missing key away from a 500, and adding a
    fact about a turn would be a contract change.
    """

    at: float
    type: ledger.EventType
    data: dict[str, object] | None


class Defect(Wire):
    """A worker turn that crashed on a unit.

    The same open `data` as Attempt, carrying the exception, its
    traceback, and the attempt and fence the turn lost.
    """

    at: float
    id: int
    item: NamedItem | None
    data: dict[str, object] | None


class SettledPhase(Wire):
    """The last phase the last SETTLED unit reached, from the rows."""

    phase: str | None
    type: ledger.EventType
    message: str | None
    at: float
    data: dict[str, object] | None


class LivePhase(Wire):
    """What the running process is doing inside the unit it is on.

    Not from the rows: it lives in process memory until the unit settles
    (sg_web/app.py live_reports), so it is null for a job running under
    another process, and `live` says which of the two this is.
    """

    phase: str | None
    type: ledger.EventType
    message: str | None
    text: str
    at: float
    data: dict[str, object] | None
    live: typing.Literal[True]


class CurrentWork(Wire):
    """What the job is on right now, and how far inside it."""

    item: NamedItem | None
    since: float | None
    phase: LivePhase | None
    last_settled_phase: SettledPhase | None


class JobDetail(Wire):
    """One job, whole (db/inspecting.py job_detail).

    Every column of the row is here -- `payload` redacted -- because an
    operator asking why a job sits under an expiring lease needs `fence`,
    `attempt`, `lease_until` and the traceback, not a progress bar. The
    one deliberate omission is the unit list, which is paged through
    /operations/job/{id}/items.
    """

    id: int
    kind: jobs.JobKind
    state: jobs.JobState
    #: SQLite stores this as 0 or 1 because it has no boolean; the fact is
    #: yes or no, and _job_detail does the translating
    cancel_requested: bool
    #: what the launcher asked for, with every secret-named key replaced
    payload: dict[str, object] | None
    #: where to resume work with no enumerable units: whatever the handler
    #: passed to db/jobs.py checkpoint, so any JSON value
    checkpoint: object | None
    total: int | None
    done_count: int
    attempt: int
    owner: str | None
    fence: int | None
    lease_until: float | None
    heartbeat_at: float | None
    error: str | None
    created_at: float
    started_at: float | None
    finished_at: float | None
    target: JobTarget | None
    failed_count: int
    pending_count: int
    succeeded_count: int
    item_count: int
    derived: Lifecycle
    settled: bool
    #: at most the first 200; the rest are paged from the items route
    failures: list[FailedItem]
    event_count: int
    last_event_id: int
    attempts: list[Attempt]
    defects: list[Defect]
    current: CurrentWork
    #: the newest few, whole; the tape holds the rest
    recent_events: list[console.Event]
    #: the sweep said in words
    what: str


def _named(held: dict | None) -> NamedItem | None:
    return None if held is None else NamedItem(id=held["id"], name=held.get("name"), href=held.get("href"))


def _live_phase(state: State, told: dict) -> LivePhase | None:
    """What the running process is doing inside the unit it is on, when
    that process is this one and the job is still running."""
    live = getattr(state, "live_reports", {}).get(told["id"])
    if live is None or told["state"] != "running":
        return None
    return LivePhase(
        phase=live.get("phase"),
        type=live["type"],
        message=live.get("message"),
        text=live["text"],
        at=live["at"],
        data=live.get("data"),
        live=True,
    )


def _job_detail(state: State, told: dict) -> JobDetail:
    """The read model as the contract states it."""
    payload = told["payload"]
    settled_phase = told["current"]["last_settled_phase"]
    return JobDetail(
        id=told["id"],
        kind=told["kind"],
        state=told["state"],
        cancel_requested=bool(told["cancel_requested"]),
        payload=payload,
        checkpoint=told["checkpoint"],
        total=told["total"],
        done_count=told["done_count"],
        attempt=told["attempt"],
        owner=told["owner"],
        fence=told["fence"],
        lease_until=told["lease_until"],
        heartbeat_at=told["heartbeat_at"],
        error=told["error"],
        created_at=told["created_at"],
        started_at=told["started_at"],
        finished_at=told["finished_at"],
        target=None
        if told["target"] is None
        else JobTarget(id=told["target"]["id"], kind=told["target"]["kind"], slug=told["target"]["slug"]),
        failed_count=told["failed_count"],
        pending_count=told["pending_count"],
        succeeded_count=told["succeeded_count"],
        item_count=told["item_count"],
        derived=Lifecycle(**told["derived"]),
        settled=told["settled"],
        failures=[
            FailedItem(id=one["id"], name=one.get("name"), href=one.get("href"), error=one["error"])
            for one in told["failures"]
        ],
        event_count=told["event_count"],
        last_event_id=told["last_event_id"],
        attempts=[
            Attempt(at=one["at"], type=one["type"], data={k: v for k, v in one.items() if k not in ("at", "type")})
            for one in told["attempts"]
        ],
        defects=[
            Defect(
                at=one["at"],
                id=one["id"],
                item=_named(one["item"]),
                data={k: v for k, v in one.items() if k not in ("at", "id", "item")},
            )
            for one in told["defects"]
        ],
        current=CurrentWork(
            item=_named(told["current"]["item"]),
            since=told["current"]["since"],
            phase=_live_phase(state, told),
            last_settled_phase=None
            if settled_phase is None
            else SettledPhase(
                phase=settled_phase["phase"],
                type=settled_phase["type"],
                message=settled_phase["message"],
                at=settled_phase["at"],
                data=settled_phase["data"],
            ),
        ),
        recent_events=[console.envelope(event) for event in told["recent_events"]],
        what=console.describe_kind(
            told["kind"], (payload or {}).get("derive"), told.get("total"), (payload or {}).get("path")
        ),
    )


@get(
    "/job/{job_id:int}",
    # The route negotiates, and a union that mixes a fragment with a JSON
    # answer reaches OpenAPI as the empty schema however precisely the
    # arms are written (measured on litestar v2.24.0). The JSON answer is
    # declared here, where the document reads it.
    responses={
        200: ResponseSpec(
            data_container=JobDetail,
            description="One job, whole: its row, its numbers, its turns and the newest of its events",
            media_type=MediaType.JSON,
            generate_examples=False,
        )
    },
    sync_to_thread=True,
)
def job_inspector(state: State, request: Request, job_id: FromPath[int]) -> Template | Response[JobDetail]:
    """One job, whole (db/inspecting.py job_detail): JSON to a machine,
    the inspector fragment to the console. Every column of the row is in
    it; the payload is redacted at this seam."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        try:
            told = inspecting.job_detail(conn, job_id, time.time())
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
    finally:
        connect.close(conn)
    detail = _job_detail(state, told)
    if wants_json(request):
        return Response(detail, headers=VARIES)
    return Template(
        template_name="_operations_job.html",
        context={"job": detail.model_dump(mode="json")},
        headers=VARIES,
    )


@get(
    "/job/{job_id:int}/items",
    responses={
        200: ResponseSpec(
            data_container=ItemPage,
            description="A page of one job's units, in unit order, with the cursor for the next page",
            media_type=MediaType.JSON,
            generate_examples=False,
        )
    },
    sync_to_thread=True,
)
def job_items(
    state: State,
    request: Request,
    job_id: FromPath[int],
    state_filter: FromQuery[str | None] = None,
    after: FromQuery[int] = 0,
    limit: FromQuery[int] = 200,
) -> Response[ItemPage] | Template:
    """A page of one job's items by state: `?state_filter=failed&after=N`
    -- JSON to a machine, a fragment the inspector swaps in otherwise.
    Paged, never folded into the detail: a 22,000-item job is read a
    page at a time."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        try:
            told = inspecting.items(conn, job_id, state=state_filter, after=after, limit=limit)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
    finally:
        connect.close(conn)
    page = ItemPage(
        items=[
            InspectedItem(
                id=one["id"], name=one.get("name"), href=one.get("href"), state=one["state"], error=one["error"]
            )
            for one in told["items"]
        ],
        next_after=told["next_after"],
    )
    if wants_json(request):
        return Response(page, headers=VARIES)
    return Template(
        template_name="_operations_items.html",
        context={
            "items": page.model_dump(mode="json")["items"],
            "next_after": page.next_after,
            "job_id": job_id,
            "state_filter": state_filter,
        },
        headers=VARIES,
    )


class EventPage(Wire):
    """A page of the ledger, ascending by id.

    `next_after` is the cursor for the page after this one, and null when
    this page reached the head -- a reader that gets null has caught up,
    where a reader that gets a number has not. `last_id` is where the
    ledger stands, so a client can tell how far behind it is without
    asking again.
    """

    events: list[console.Event]
    after: int
    next_after: int | None
    last_id: int


@get("/events", sync_to_thread=True)
def events(
    state: State, after: FromQuery[int] = 0, job: FromQuery[int | None] = None, limit: FromQuery[int] = 500
) -> EventPage:
    """A page of the ledger, ascending from `after`, the whole ledger or
    one job's: the gap-fill and the "earlier" read. Every row, never a
    sample; `next_after` pages the rest."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        told = inspecting.events(conn, job_id=job, after=after, limit=limit)
    finally:
        connect.close(conn)
    return EventPage(
        events=[console.envelope(event) for event in told["events"]],
        after=told["after"],
        next_after=told["next_after"],
        last_id=told["last_id"],
    )


class EarlierEvents(Wire):
    """The page above the one a reader holds, and which one that was."""

    events: list[console.Event]
    before: int


@get("/events/before", sync_to_thread=True)
def events_before(
    state: State, before: FromQuery[int] = 0, job: FromQuery[int | None] = None, limit: FromQuery[int] = 500
) -> EarlierEvents:
    """The `limit` events with id < `before`, ascending: the tape's
    "earlier" button. Bounded; walks the index backwards and stops.
    No `before` is nothing earlier than the beginning: an empty page."""
    limit = max(1, min(int(limit), ledger.PAGE_MOST))
    conn = connect.connect(state.db_path, read_only=True)
    try:
        page = inspecting.events_before(conn, before, job_id=job, limit=limit)
    finally:
        connect.close(conn)
    return EarlierEvents(events=[console.envelope(event) for event in page], before=before)


@post("/jobs/{kind:str}", sync_to_thread=True)
def launch(state: State, kind: FromPath[str], everything: FromQuery[bool] = False) -> Template:
    """Start one sweep from its button -- `?everything=true` from the
    "again" button of a sweep that is otherwise for what is missing. The
    answer is the notice fragment; the job itself arrives on the
    activity surface through the feed, as every job does, so the page
    never grows a second list of jobs."""
    found = LAUNCHERS.get(kind)
    if found is None:
        raise NotFoundException(f"/operations/jobs/{kind}: nothing to start by that name")
    label, launcher = found
    if everything:
        again = AGAIN.get(kind)
        if again is None:
            raise NotFoundException(f"/operations/jobs/{kind}: this sweep has no 'again'; it already does all of it")
        label, launcher = f"{label}, all of it again", again
    conn = connect.connect(state.db_path)
    try:
        try:
            job_ids = launcher(state, conn)
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        told = [submitted(state, conn, job_id) for job_id in job_ids]
    finally:
        connect.close(conn)
    queued = ", ".join(f"#{job['id']}" for job in told)
    notice = f"{label}: queued {queued}" if told else f"{label}: nothing to do"
    return Template(template_name="_operations_notice.html", context={"notice": notice, "error": None}, headers=VARIES)


@dataclasses.dataclass
class RootForm:
    path: str
    kind: str = "library"


@post("/roots", sync_to_thread=True)
def add_root(state: State, data: URLEncodedBody[RootForm]) -> Template:
    """Register a media directory from the form; the roots section comes
    back re-read. Registering reads nothing -- scanning is its own button."""
    cleaned = data.path.strip()
    if not cleaned:
        raise ClientException("a root needs a path")
    conn = connect.connect(state.db_path)
    try:
        library.add_root(conn, cleaned, data.kind, time.time())
        conn.commit()
        roots = _roots(conn)
        behind = _behind(inspecting.coverage(conn, _weights(state, conn))["missing"])
    finally:
        connect.close(conn)
    return Template(
        template_name="_operations_roots.html",
        context={"roots": roots, "notice": f"registered {cleaned}", "behind": behind},
        headers=VARIES,
    )


@post("/roots/{root_id:int}/scan", sync_to_thread=True)
def scan_root(state: State, root_id: FromPath[int]) -> Template:
    """Walk one root now. Synchronous like the JSON route: a walk is
    cheap, and its counts are the answer the person pressed for."""
    conn = connect.connect(state.db_path)
    try:
        path = library.root_path(conn, root_id)
        if path is None:
            raise NotFoundException(f"no root {root_id}")
        result = scan.scan(conn, root_id, path, time.time())
        conn.commit()
        cache = str(home.thumbs_dir(pathlib.Path(state.home)))
        precache = runner.precache_after_scan(conn, time.time(), result, thumbs_dir=cache)
        if precache is not None:
            conn.commit()
            submitted(state, conn, precache)
        roots = _roots(conn)
        # A scan LISTS files. Nothing here reads, fingerprints or captions
        # one, so the person who just registered a library is looking at a
        # gallery with no metadata, no search and no people -- and the
        # button that fixes it is in a different section further down.
        behind = _behind(inspecting.coverage(conn, _weights(state, conn))["missing"])
    finally:
        connect.close(conn)
    notice = (
        f"scanned {path}: {result.added} added, {result.matched} matched, {result.replaced} replaced,"
        f" {result.missing} missing, {result.ambiguous} ambiguous"
        + (f"; thumbnails queued as job #{precache}" if precache is not None else "")
    )
    return Template(
        template_name="_operations_roots.html",
        context={"roots": roots, "notice": notice, "behind": behind},
        headers=VARIES,
    )


@dataclasses.dataclass
class SettingForm:
    value: str


@post("/settings/{key:str}", sync_to_thread=True)
def change_setting(state: State, key: FromPath[str], data: URLEncodedBody[SettingForm]) -> Template:
    """One setting, changed live, the whole vocabulary re-read. Refusals
    are the registry's (db/settings.py put)."""
    conn = connect.connect(state.db_path)
    try:
        try:
            settings.put(conn, key, data.value)
        except (KeyError, ValueError) as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        rows = settings.snapshot(conn)
    finally:
        connect.close(conn)
    return Template(
        template_name="_operations_settings.html",
        context={"settings": rows, "notice": f"{key} = {data.value}"},
        headers=VARIES,
    )


@post("/clusterings/choose", sync_to_thread=True)
def choose_primary(state: State) -> Template:
    conn = connect.connect(state.db_path)
    try:
        chosen = derived.choose_primary(conn)
        conn.commit()
        runs = pages.standings(conn)
    finally:
        connect.close(conn)
    return Template(
        template_name="_operations_clusterings.html",
        context={"clusterings": runs, "primary": _primary(runs), "notice": f"primary run: {chosen}"},
        headers=VARIES,
    )


class Scheduled(Wire):
    """One thing that runs without being asked."""

    collection: str
    every_hours: float
    enabled: bool
    #: when it last STARTED its collection, never when that finished:
    #: the next run is measured from the start, so a three-hour catch-up
    #: on a nightly schedule still runs once a night rather than
    #: drifting three hours later every day
    last_started_at: float | None
    #: when it is next due, null when it never runs. 0 means "now",
    #: which is what a schedule somebody just turned on should be --
    #: waiting a full interval to prove it works is not proof.
    next_due: float | None
    #: is its collection going right now. A due schedule whose collection
    #: is already running does not start a second one.
    running: bool


@dataclasses.dataclass
class ScheduleForm:
    """The body of POST /operations/schedules/{collection}.

    Form-encoded like the settings form beside it, not JSON: htmx posts
    a form as a form, and reaching for an extension to make one console
    control behave differently from the others is a second convention
    for no gain.

    `enabled` is a string because an unchecked checkbox sends NOTHING --
    the absence is the answer, and a bool with a default would read a
    missing field as True.
    """

    every_hours: str
    enabled: str = ""


def _schedules(conn) -> list[Scheduled]:
    return [
        Scheduled(
            collection=row["collection"],
            every_hours=row["every_hours"],
            enabled=bool(row["enabled"]),
            last_started_at=row["last_started_at"],
            next_due=scheduling.next_due(row),
            running=scheduling.running(conn, row["collection"]),
        )
        for row in scheduling.all_of(conn)
    ]


@post("/collections/{collection:str}/stop", sync_to_thread=True)
def stop_collection(state: State, collection: FromPath[str]) -> Template:
    """Stop a whole collection, in one press.

    Eight rows was already tedious to stop by hand when a person had
    queued them. Since a schedule can start one at 3am, it is not an
    option at all: the thing to stop may be something nobody started.

    A running step is ASKED and stops at its next item boundary -- work
    in flight is the runner's to end. Everything still queued is settled
    here, because nothing holds it and a cancel no worker will ever
    claim is a cancel nobody honours.
    """
    conn = connect.connect(state.db_path)
    try:
        asked, stopped = jobs.stop_collection(conn, collection, time.time())
        conn.commit()
    finally:
        connect.close(conn)
    if not asked:
        notice = f"{collection}: nothing running"
    else:
        running = len(asked) - len(stopped)
        notice = f"{collection}: stopped {len(stopped)} queued" + (
            f", asked {running} running to stop" if running else ""
        )
    return Template(template_name="_operations_notice.html", context={"notice": notice, "error": None}, headers=VARIES)


@post("/schedules/{collection:str}", sync_to_thread=True)
def set_schedule(state: State, collection: FromPath[str], data: URLEncodedBody[ScheduleForm]) -> Template:
    """Set what runs without being asked, and how often.

    Hours rather than a cron expression. A cron string is a small
    language, and a small language wants a parser, a validator and a way
    to say what it will do next; hours answer the question somebody
    actually has -- nightly, twice a day -- and can be shown as a time
    without interpreting anything.
    """
    on = data.enabled.strip().lower() in ("true", "on", "1", "yes")
    conn = connect.connect(state.db_path)
    try:
        try:
            hours = float(data.every_hours)
        except ValueError as bad:
            raise ClientException(f"every-how-many-hours must be a number, not {data.every_hours!r}") from bad
        try:
            scheduling.put(conn, collection, hours, time.time(), enabled=on)
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        rows = _schedules(conn)
    finally:
        connect.close(conn)
    said = f"every {hours:g}h" if on else "off"
    return Template(
        template_name="_operations_schedules.html",
        context={"schedules": rows, "runnable": scheduling.RUNNABLE, "notice": f"{collection}: {said}"},
        headers=VARIES,
    )


def _primary(runs: list[dict]) -> int | None:
    """The run the site is showing, or None before anything is chosen.

    Which is the only run worth comparing AGAINST: the question somebody
    has while looking at a list of thresholds is always "is this one
    better than what I am living with".
    """
    return next((run["id"] for run in runs if run["is_primary"]), None)


@get("/clusterings/{left:int}/against/{right:int}", sync_to_thread=True)
def compare_clusterings(state: State, left: FromPath[int], right: FromPath[int]) -> Template:
    """The pictures two clusterings describe differently.

    The other half of a reachable threshold. Changing it writes a new run
    beside the old (schema.sql derived_face_run_identity), which makes
    trying one safe -- and left somebody with two runs, two numbers, and
    no way to see what actually moved between them.

    The disagreement is the answer, not the counts: one threshold welds
    strangers together and another splits one person into four, and both
    read as "more groups" or "fewer groups" until you put the two
    answers on the same photograph.

    Bounded, and it says by how much. `db/pages.py disagreements` returns
    the total alongside the page, because "these are the only twelve" and
    "the first fifty of nine thousand" are opposite answers to the
    question being asked.
    """
    conn = connect.connect(state.db_path)
    try:
        runs = {run["id"]: run for run in pages.standings(conn)}
        for one in (left, right):
            if one not in runs:
                raise NotFoundException(f"no clustering run #{one}")
        held = pages.disagreements(conn, left, right)
    finally:
        connect.close(conn)
    return Template(
        template_name="_operations_compare.html",
        context={"left": runs[left], "right": runs[right], "held": held},
        headers=VARIES,
    )


@get("/export/faces/{name:str}", sync_to_thread=True)
def export_faces(
    state: State,
    name: FromPath[str],
    since: FromQuery[float | None] = None,
    until: FromQuery[float | None] = None,
) -> Response[FaceExport]:
    """One person's face vectors, with the provenance that makes them
    mean something.

    The most expensive thing this application learns about a library and
    the least portable: a naked 512-float vector recreates exactly the
    opaque dependency an export is supposed to escape. So each group
    carries the producer, its version, the preprocessing, the metric and
    the dimensions -- `similarity_space`, which is immutable by trigger
    -- and the centroid the clustering computed.

    Grouped BY SPACE because a vector is comparable only to another from
    the same one. A library re-detected under a new model holds two
    representations of one person, and flattening them would invite a
    comparison that means nothing.

    These are BIOMETRIC TEMPLATES. This is your own library answering on
    your own machine, which is the only place face vectors are allowed
    to be here -- the surface offering it says so, because somebody
    should know what they are about to put in a file.
    """
    # The extension rides INSIDE the captured value, the way
    # `/thumbs/{shard}/{name}` takes `<sha>.webp`: a path parameter
    # occupies a whole segment, so `{slug:str}.json` matches nothing at
    # all -- checked, it 404s before the handler is reached.
    slug = name.removesuffix(".json")
    conn = connect.connect(state.db_path)
    try:
        told = derived.person_faces(conn, slug, since=since, until=until)
    finally:
        connect.close(conn)
    if told is None:
        raise NotFoundException(f"no person at {slug}")
    return Response(
        content=FaceExport(
            person=told["person"],
            name=told["name"],
            spaces=[
                FaceSpace(
                    **one | {"faces": [FaceVector(**who | {"region": Box(**who["region"])}) for who in one["faces"]]}
                )
                for one in told["spaces"]
            ],
        ),
        headers={"content-disposition": f'attachment; filename="faces-{slug}.json"', **VARIES},
    )


@get("/export/authored.json", sync_to_thread=True)
def export_authored(state: State) -> Response[Authored]:
    """Everything you told this library about your own pictures.

    The opposite shape from the verdict export beside it, and
    deliberately. That one is for SHARING, so it carries no name and no
    path. This one is for CUSTODY: it is yours, it is about your
    pictures, and an export that withheld the names would be withholding
    them from their owner. An application about keeping your own data
    must not be the only place your understanding can exist.

    Keyed by `content_sha256` and never by a row id, which is the
    difference between an export and a dump: an id belongs to one
    database file, a hash names the same photograph in any library that
    holds it. Only pictures carrying something authored -- a library is
    mostly pictures nobody has said anything about.
    """
    conn = connect.connect(state.db_path)
    try:
        told = authored.exported(conn)
    finally:
        connect.close(conn)
    return Response(
        content=Authored(
            people=[Named(**one) for one in told["people"]],
            collections=[Shelved(**one) for one in told["collections"]],
            pictures=[
                Picture(
                    **one
                    | {
                        "tags": [Keyword(**word) for word in one["tags"]],
                        "people": [Appears(**who | {"region": _boxed(who["region"])}) for who in one["people"]],
                    }
                )
                for one in told["pictures"]
            ],
        ),
        headers={"content-disposition": 'attachment; filename="authored.json"', **VARIES},
    )


def _boxed(region: dict | None) -> Box | None:
    return None if region is None else Box(**region)


@get("/export/verdicts.json", sync_to_thread=True)
def export_verdicts(state: State, include: FromQuery[str] = "") -> Response[list[VerdictRow]]:
    """Every verdict, as an eval set that carries no pictures.

    The cheapest valuable thing this library accumulates and the easiest
    to share safely: forty judgements over YOUR pictures are worth more
    than a leaderboard somebody else ran over theirs, and they can leave
    the machine with none of the media leaving with them.

    Under `/export/` because it is BYTES somebody saves and not a page
    they land on -- the same distinction `/thumb/` and `/media/` are
    under (tests/test_the_shell_mounts_every_surface.py `_links`).
    """
    wanted = tuple(one.strip() for one in include.split(",") if one.strip())
    conn = connect.connect(state.db_path)
    try:
        try:
            told = [VerdictRow(**row) for row in verdicts.exported(conn, include=wanted)]
        except ValueError as refusal:
            raise ClientException(str(refusal)) from refusal
    finally:
        connect.close(conn)
    return Response(
        content=told,
        headers={"content-disposition": 'attachment; filename="verdicts.json"', **VARIES},
    )


def refused(request: Request, exc: HTTPException) -> Template:
    """A refusal, rendered where the person is looking: the shell notice,
    carrying the refusal's own status. htmx swaps 4xx/5xx into
    #shell-notice by the shell's response-handling config
    (templates/base.html), so the reason lands on screen instead of in
    the console. The status is the exception's -- a 400 stays a 400
    (litestar-org/litestar@v2.24.0 litestar/router.py:96,135
    `exception_handlers`; docs/usage/exceptions.rst:95
    per_exception_handlers.py: a Router-level handler overrides the
    app's JSON one for this layer only)."""
    return Template(
        template_name="_operations_notice.html",
        context={"error": exc.detail, "notice": None},
        status_code=exc.status_code,
        headers=VARIES,
    )


def busy(request: Request, exc: sqlite3.OperationalError) -> Template:
    """A console click while another writer holds the lane is
    backpressure, not a crash.

    SQLite has ONE writer lane and a long write -- a scan committing a
    new root -- holds it past busy_timeout. The worker already answers
    this shape with "no turn this pass" (db/runner.py); a form deserves
    the same honesty: a 503 saying try again, never a 500 traceback.
    Recognised by result NAME (runner.BUSY), the way the worker does;
    any other OperationalError is the internal error it is. Nothing
    partial happened -- the statement that hit the lock rolled back
    with its transaction."""
    if getattr(exc, "sqlite_errorname", "") not in runner.BUSY:
        return failed(request, exc)
    return Template(
        template_name="_operations_notice.html",
        context={"error": "the library is busy writing; nothing was started -- try again in a moment", "notice": None},
        status_code=HTTP_503_SERVICE_UNAVAILABLE,
        headers=VARIES,
    )


def failed(request: Request, exc: Exception) -> Template:
    """An unexpected error on an operations form: logged whole, shown as
    the 500 it is, never swallowed into a blank notice."""
    _logger.error("operations request %s %s failed", request.method, request.url.path, exc_info=exc)
    return Template(
        template_name="_operations_notice.html",
        context={"error": f"internal error: {exc}", "notice": None},
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        headers=VARIES,
    )


router = Router(
    path="/operations",
    route_handlers=[
        operations_page,
        overview,
        job_inspector,
        job_items,
        events,
        events_before,
        launch,
        add_root,
        scan_root,
        change_setting,
        choose_primary,
        compare_clusterings,
        export_verdicts,
        export_authored,
        export_faces,
        set_schedule,
        stop_collection,
    ],
    exception_handlers={HTTPException: refused, sqlite3.OperationalError: busy, Exception: failed},
)
