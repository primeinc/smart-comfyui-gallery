"""The story snapshot adapters: freeze one current event, read one
frozen document.

Freezing is synchronous on purpose -- "freeze the event I am looking
at right now" cannot be queued, because the event could change before
a worker reached it; it is database work measured in milliseconds and
proves its own currentness (db/stories.py). Model work never happens
here: planning and writing are later durable jobs over the frozen
input. Reading a snapshot consults history only.
"""

from __future__ import annotations

import json
import pathlib
import time
import typing
import urllib.parse
from typing import Literal

from litestar import MediaType, Request, get, post
from litestar.datastructures import State
from litestar.exceptions import ClientException, NotFoundException
from litestar.openapi.datastructures import ResponseSpec
from litestar.params import FromPath, FromQuery
from litestar.response import Redirect, Response, Template

from db import connect, derived, evolution, facets, naming, pages, planning, rendering, settings, stories
from sg_web import home, submitting
from sg_web.presenting import VARIES, presented_page, wants_json
from sg_web.wire import Wire


def _window(subject: dict) -> str | None:
    """The timeline's hour window around the frozen session, in the
    domain the evidence claims it in; None when the subject holds no
    interval."""
    when = subject.get("time") or {}
    held = when.get("local") or when.get("instant")
    if not held:
        return None
    start = int(held[0] // 3600) * 3600
    end = int(held[-1] // 3600) * 3600 + 3600
    return "/timeline?" + urllib.parse.urlencode({"bin": "hour", "start": start, "end": end})


@get("/stories", sync_to_thread=True)
def stories_index(state: State, request: Request, kind: FromQuery[str | None] = None) -> Template | Response:
    """Every story told, newest first -- the shelf the timeline's
    buttons fill -- or, with `?kind=`, those of one session kind. Each
    entry is its title and dek as rendered, its heroes, its subject
    kind, its profile, and links to the render and the plan's
    evolution."""
    if kind is not None and kind not in stories.EVENT_KINDS:
        raise ClientException(f"kind is one of {', '.join(stories.EVENT_KINDS)}, not {kind!r}")
    conn = connect.connect(state.db_path, read_only=True)
    try:
        kinds = pages.story_kinds(conn)
        told = []
        for row in pages.stories(conn, kind=kind):
            render_id, plan_id, profile, created_at, document, planner, event_kind, snapshot_id, frozen = row
            words = json.loads(document)
            told.append(
                {
                    "id": render_id,
                    "plan_id": plan_id,
                    "snapshot_id": snapshot_id,
                    "profile": profile,
                    "planner": planner,
                    "kind": event_kind,
                    "told_at": created_at,
                    "title": words.get("title", ""),
                    "dek": words.get("dek", ""),
                    "members": len((words.get("support") or {}).get("member_refs") or []),
                    "heroes": rendering.heroes(conn, words, json.loads(frozen)),
                    "href": f"/stories/renders/{render_id}",
                    "evolution": f"/stories/plans/{plan_id}/evolution",
                }
            )
    finally:
        connect.close(conn)
    return presented_page(request, told, page="stories.html", context={"stories": told, "kind": kind, "kinds": kinds})


class FreezeRequest(Wire):
    """The body of POST /stories/snapshots: which current event."""

    event_id: int


class MadeOrFound(Wire):
    """A content-addressed row, and whether this request is what made it.

    A snapshot of an event and a render of a plan are both identified by
    the sha of their content, so asking twice is one row. `reused` says
    which happened -- the same thing the status code says (200 found, 201
    made), in the body, so a client reading only JSON still knows.
    """

    id: int
    sha256: str
    reused: bool


@post("/stories/snapshots", sync_to_thread=True)
def freeze_snapshot(state: State, data: FreezeRequest) -> Response[MadeOrFound]:
    conn = connect.connect(state.db_path)
    try:
        try:
            made = stories.snapshot_event(conn, data.event_id, time.time())
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
    finally:
        connect.close(conn)
    return Response(
        MadeOrFound(id=made.id, sha256=made.sha256, reused=made.reused),
        status_code=200 if made.reused else 201,
    )


@get("/stories/snapshots/{snapshot_id:int}", sync_to_thread=True)
def snapshot_document(state: State, snapshot_id: FromPath[int]) -> dict:
    conn = connect.connect(state.db_path, read_only=True)
    try:
        try:
            return stories.load_snapshot(conn, snapshot_id)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as corrupt:
            raise ClientException(str(corrupt), status_code=409) from corrupt
    finally:
        connect.close(conn)


#: How a plan reads a snapshot, per db/schema.sql story_plan.planner.
Planner = Literal["generation_history", "capture_history", "file_history"]

#: Which voice a render speaks in, per db/schema.sql story_render.profile.
RenderProfile = Literal["memory", "technical", "compact"]


class PlanRequest(Wire):
    """The body of POST /stories/plans: which frozen snapshot, under
    which planner, read by which similarity engine -- named EXACTLY
    (`lexical`, or a configured semantic provider such as `openclip` or
    `qwen`; never a default that might mean something else)."""

    snapshot_id: int
    planner: Planner = "generation_history"
    similarity: str = "lexical"
    settings: dict[str, object] | None = None


@post("/stories/plans", sync_to_thread=True)
def plan_snapshot(state: State, data: PlanRequest) -> Response:
    """Ask for a plan. The service records the request's identity,
    reuses a finished plan or a live job for the same request, or
    queues durable work -- no weights load on this thread. 200 with a
    plan id when the answer already exists, 202 with the job otherwise."""
    conn = connect.connect(state.db_path)
    try:
        try:
            weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
            engine = planning.engine_for(conn, data.similarity, weights)
            asked = planning.request_plan(conn, data.snapshot_id, data.planner, engine, data.settings, time.time())
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except stories.Corrupt as corrupt:
            raise ClientException(str(corrupt), status_code=409) from corrupt
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
        told = {"request_sha256": asked.request_sha256, "plan_id": asked.plan_id, "job": None}
        if asked.job_id is not None:
            # Announced and woken like every other submit: a plan job is
            # a job on the feed from the moment it is queued.
            told["job"] = submitting.submitted(state, conn, asked.job_id)
    finally:
        connect.close(conn)
    return Response(told, status_code=200 if asked.plan_id is not None else 202)


@get("/stories/plans/{plan_id:int}", sync_to_thread=True)
def plan_document(state: State, request: Request, plan_id: FromPath[int]) -> dict | Redirect:
    """The plan document to a machine; a browser is sent to the plan's
    page, which is its evolution view -- a person never lands on JSON."""
    if not wants_json(request):
        return Redirect(path=f"/stories/plans/{plan_id}/evolution", status_code=302)
    conn = connect.connect(state.db_path, read_only=True)
    try:
        try:
            return planning.load_plan(conn, plan_id)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as corrupt:
            raise ClientException(str(corrupt), status_code=409) from corrupt
    finally:
        connect.close(conn)


class RenderRequest(Wire):
    """The body of POST /stories/renders: which plan, under which
    profile and locale. Rendering is pure code and synchronous."""

    plan_id: int
    profile: RenderProfile = "memory"
    locale: str = "en"


@post("/stories/renders", sync_to_thread=True)
def render_plan(state: State, data: RenderRequest) -> Response[MadeOrFound]:
    conn = connect.connect(state.db_path)
    try:
        try:
            narrator = rendering.TemplateStoryRenderer(data.profile, data.locale)
            made = rendering.render_plan(conn, data.plan_id, narrator, time.time())
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except stories.Corrupt as corrupt:
            raise ClientException(str(corrupt), status_code=409) from corrupt
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        conn.commit()
    finally:
        connect.close(conn)
    return Response(
        MadeOrFound(id=made.id, sha256=made.sha256, reused=made.reused),
        status_code=200 if made.reused else 201,
    )


@get("/stories/sessions/{event_id:int}", sync_to_thread=True)
def session_story(state: State, event_id: FromPath[int], back: FromQuery[str | None] = None) -> Redirect:
    """Opening a session tells its story: the session is frozen, its plan
    asked for, and the story rendered and shown -- the three story steps
    as one link. When the plan is durable work still running, the person
    is sent back where they came from (`back`, a path on this site) and
    the timeline refreshes itself when the job settles; the next opening
    lands on the story."""
    conn = connect.connect(state.db_path)
    try:
        try:
            # each step commits before the next: freezing, planning and
            # rendering each own their transaction (db/planning.py
            # request_plan begins its own), exactly as the three routes do
            frozen = stories.snapshot_event(conn, event_id, time.time())
            conn.commit()
            weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
            engine = planning.engine_for(conn, "lexical", weights)
            kind = str((stories.load_snapshot(conn, frozen.id).get("subject") or {}).get("event_kind") or "")
            planner = _PLANNER_FOR_KIND.get(kind, "file_history")
            asked = planning.request_plan(conn, frozen.id, planner, engine, None, time.time())
            if asked.job_id is not None:
                submitting.submitted(state, conn, asked.job_id)
            conn.commit()
            if asked.plan_id is None:
                where = back if back and back.startswith("/") and not back.startswith("//") else "/timeline"
                return Redirect(path=f"{where}#session-{event_id}", status_code=303)
            made = rendering.render_plan(
                conn, asked.plan_id, rendering.TemplateStoryRenderer("memory", "en"), time.time()
            )
            conn.commit()
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except stories.Corrupt as corrupt:
            raise ClientException(str(corrupt), status_code=409) from corrupt
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
    finally:
        connect.close(conn)
    return Redirect(path=f"/stories/renders/{made.id}", status_code=303)


#: Which planner tells which kind of session's story.
_PLANNER_FOR_KIND = {
    "generation_session": "generation_history",
    "capture_session": "capture_history",
    "file_session": "file_history",
}


@get("/stories/renders/{render_id:int}", sync_to_thread=True)
def render_document(state: State, render_id: FromPath[int], request: Request) -> Response | Template:
    """The verified render, as JSON or laid out as HTML by Accept. The
    page interprets no claim, joins no table, calls no model: every
    hero and every member is the FROZEN name, linked to its picture only
    through address resolution -- a member whose file has since left the
    library keeps its name and loses its link."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        try:
            story, members, plan_id, subject = rendering.load_render_with_members(conn, render_id)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except ValueError as corrupt:
            raise ClientException(str(corrupt), status_code=409) from corrupt
        if wants_json(request):
            return Response(story, headers=VARIES)
        addressed = {}
        ids: dict[str, int] = {}
        for ref, member in members.items():
            held = naming.by_uuid(conn, member["file_uuid"])
            slug = held[1] if held and held[0] == "file" else None
            addressed[ref] = {
                "name": member["name"],
                "slug": slug,
                "page": f"/i/{slug}" if slug else None,
                "thumbnail": f"/thumb/{slug}" if slug else None,
                "kind": member.get("media_kind"),
                "said": None,
            }
            found = naming.resolve(conn, "file", slug) if slug else None
            if found is not None:
                ids[ref] = found[0]
        # What a model says about a hero TODAY -- live, by address, never
        # part of the frozen story: the evidence the snapshot froze is the
        # story's; this is a courtesy line the page labels as its own.
        said = derived.said_first(conn, ids.values(), prefer=settings.value(conn, "caption_model"))
        for ref, file_id in ids.items():
            addressed[ref]["said"] = said.get(file_id)
    finally:
        connect.close(conn)
    # Rendered by the application's one engine (sg_web/app.py
    # _template_engine: StrictUndefined, so a missing frozen field explodes
    # instead of rendering "You introduced ."; autoescape, because frozen
    # evidence is evidence, not trusted markup).
    return Template(
        template_name="story.html",
        context={
            "story": story,
            "members": addressed,
            "render_id": render_id,
            "plan_id": plan_id,
            "profiles": rendering.PROFILES,
            "session_window": _window(subject),
        },
        headers=VARIES,
    )


# --- the Evolution Explorer's contract ---------------------------------------
#
# db/evolution.py measures; these say what a reader is given. Two kinds of
# thing share the document and are modelled differently on purpose.
#
# MEASURED HERE, NOW -- the cosines, the deltas, the space they were taken
# in. One function produces them over a closed set of facts, so every field
# is named exactly.
#
# FROZEN THEN -- a StoryPlan at ITS OWN format version (there have been
# seven) and a StorySnapshot as it was written. A claim's `facts` have to
# fit its kind under the plan's vocabulary version, and db/planning.py
# validates that seven different ways; restating the union here would be an
# eighth spelling free to disagree with all of them. So `facts` stays open,
# and `plan.format` says which vocabulary wrote it.
#
# The same reasoning keeps the frozen strings as strings. `media.kind` came
# out of the library at freeze time; holding it to today's CHECK would make
# a historical document fail for having been true.


class EvolutionUnsupported(Wire):
    """Something the planner would not claim, and why it would not."""

    kind: str
    reason: str
    member_refs: list[str] | None = None


class EvolutionPlan(Wire):
    """The plan this view is of, and its own format version."""

    id: int
    sha256: str
    format: int
    #: whether the evidence establishes an order; without one there are no
    #: transitions to measure
    sequenced: bool
    unsupported: list[EvolutionUnsupported]
    label: str


class EvolutionTime(Wire):
    """The session's interval, in each domain the evidence claims it in:
    start and end, or null where the evidence has no such clock."""

    local: list[float] | None
    instant: list[float] | None


class EvolutionSnapshot(Wire):
    """The frozen evidence the plan was built over."""

    sha256: str
    time: EvolutionTime
    members: int
    subject: str


class EvolutionSemantic(Wire):
    """The one space and query policy every metric here was taken in.

    `unavailable` is the reason there are no numbers rather than a zero
    dressed as one: no provider configured, several configured and none
    named, or nothing embedded under this space yet.
    """

    provider: str | None
    space_id: int | None
    space: str | None
    prompt_policy_hash: str | None
    unavailable: str | None


class EvolutionClaim(Wire):
    """One thing the plan asserts about a phase.

    `facts` is open, and deliberately: what belongs in it is decided by
    `kind` under the plan's own format version (db/planning.py
    _facts_valid_v1..v7). A reader takes `kind` and `plan.format` first.
    """

    id: str
    kind: str
    facts: dict[str, object]


class EvolutionPhase(Wire):
    """A run of members the plan drew a boundary around."""

    id: str
    label: str
    member_refs: list[str]
    representative_refs: list[str]
    claims: list[EvolutionClaim]


class EvolutionMedia(Wire):
    """The member's frozen file identity, and where it lives now.

    `uuid` and `content_sha256` are what was frozen; `slug` is the address
    that identity has today, null when the file is gone -- so `thumbnail`
    and `page` are null with it. Everything below `slug` is this module's
    doing: db/evolution.py returns identities and owns no URL.
    """

    uuid: str
    name: str
    kind: str
    content_sha256: str
    slug: str | None
    thumbnail: str | None
    page: str | None


class EvolutionOccurrence(Wire):
    """When the frozen evidence puts this member, and how sure it is.

    `certainty` is an ordinal's fixed spelling, not a probability:
    corroborated .9, claimed .6, contested .4 (db/when.py Verdict, whose
    `supports` and `conflicts` name the readings that agreed and
    disagreed) -- so a contested time says what contests it.

    Every field is always here, and a frozen document that does not carry
    one reads as null or empty -- which says the same thing: this document
    does not name it. The projection defaults rather than demands, because
    a snapshot written before a field existed is still a snapshot the
    library must serve.
    """

    kind: str
    basis: str
    certainty: float | None
    precision: str
    local_at: float | None
    instant_at: float | None
    tz_offset_min: int | None
    supports: list[str]
    conflicts: list[str]
    finished_at: float | None
    estimated_at: float | None
    source_order: int | None
    act_key: str | None


class EvolutionPrompt(Wire):
    """One role's prompt as the snapshot froze it.

    `main` is the section the planner reads, and `main_hash` identifies it;
    `prompt_id` is the live row still holding that text, null when none
    does -- an address for "prompts like this", never evidence.
    """

    text: str
    hash: str
    main: str
    main_hash: str
    prompt_id: int | None


class EvolutionPrompts(Wire):
    """The two roles this view measures: what was written, and what ran.
    Either is null when the snapshot froze none."""

    effective: EvolutionPrompt | None
    original: EvolutionPrompt | None


class EvolutionGeneration(Wire):
    """The recipe, as frozen. Every field null for a member with no
    generation evidence at all."""

    seed: int | None
    steps: int | None
    cfg: float | None
    denoise: float | None
    clip_skip: int | None
    sampler: str | None
    scheduler: str | None
    width: int | None
    height: int | None
    tool: str | None
    #: the checkpoint's frozen name
    model: str | None
    loras: list[str]
    #: the same LoRAs by frozen identity -- two files sharing a name are
    #: two LoRAs, one file renamed is one
    lora_uuids: list[str]


class EvolutionMetrics(Wire):
    """One member's own cosines. A null number carries the reason beside
    it: an unavailable metric says why rather than reading as zero."""

    original_effective_cosine: float | None
    original_effective_cosine_unavailable: str | None = None
    text_image_cosine: float | None
    text_image_cosine_unavailable: str | None = None


class EvolutionMember(Wire):
    """One member of the frozen session, measured."""

    ref: str
    phase_ref: str | None
    media: EvolutionMedia
    occurrence: EvolutionOccurrence | None
    prompt: EvolutionPrompts
    generation: EvolutionGeneration
    metrics: EvolutionMetrics


class EvolutionParameterChange(Wire):
    """One recipe fact that differed across a transition."""

    name: evolution.ChangedParameter
    before: int | float | str | None
    after: int | float | str | None


class EvolutionChanges(Wire):
    """Exactly what differed between two consecutive members.

    `parameters` is a LIST rather than an object with a field per
    parameter: the module reports only what actually changed, and a field
    per parameter would make the wire say "the seed did not change" where
    the measurement says nothing at all. Membership carries the sparseness
    that key-presence used to.
    """

    parameters: list[EvolutionParameterChange]
    loras_added: list[str]
    loras_removed: list[str]
    lora_uuids_added: list[str]
    lora_uuids_removed: list[str]


class EvolutionTransition(Wire):
    """One consecutive pair, measured. Sequenced plans only.

    `phase_boundary` is where the PLAN put a boundary, never where a
    cosine dipped.
    """

    before: str
    after: str
    phase_boundary: bool
    prompt_cosine: float | None
    prompt_cosine_unavailable: str | None = None
    visual_cosine: float | None
    visual_cosine_unavailable: str | None = None
    changes: EvolutionChanges


class EvolutionEdge(Wire):
    """One frozen lineage edge. An end inside the session is a member ref;
    an end outside it is that file's frozen uuid."""

    parent: str
    child: str
    kind: str


class EvolutionLinks(Wire):
    """Addresses this module builds from the identities db/evolution.py
    returned. `neighbours` is a template: a prompt id goes in `{id}`."""

    story: str | None
    gallery_day: str | None
    search: str
    neighbours: str


class EvolutionView(Wire):
    """The Generation Evolution Explorer's whole document."""

    v: int
    plan: EvolutionPlan
    snapshot: EvolutionSnapshot
    semantic: EvolutionSemantic
    phases: list[EvolutionPhase]
    members: list[EvolutionMember]
    transitions: list[EvolutionTransition]
    lineage: list[EvolutionEdge]
    links: EvolutionLinks


def _prompt_of(held: dict | None) -> EvolutionPrompt | None:
    if held is None:
        return None
    return EvolutionPrompt(
        text=held["text"],
        hash=held["hash"],
        main=held["main"],
        main_hash=held["main_hash"],
        prompt_id=held["prompt_id"],
    )


def _occurrence_of(held: dict | None) -> EvolutionOccurrence | None:
    if held is None:
        return None
    return EvolutionOccurrence(
        kind=held["kind"],
        basis=held["basis"],
        certainty=held.get("certainty"),
        precision=held["precision"],
        local_at=held.get("local_at"),
        instant_at=held.get("instant_at"),
        tz_offset_min=held.get("tz_offset_min"),
        supports=list(held.get("supports") or []),
        conflicts=list(held.get("conflicts") or []),
        finished_at=held.get("finished_at"),
        estimated_at=held.get("estimated_at"),
        source_order=held.get("source_order"),
        act_key=held.get("act_key"),
    )


def _changes_of(held: dict) -> EvolutionChanges:
    """The module's sparse delta as the contract states it: a key present
    only because it changed becomes a member of `parameters`."""
    return EvolutionChanges(
        parameters=[
            EvolutionParameterChange(name=name, before=held[name]["from"], after=held[name]["to"])
            for name in typing.get_args(evolution.ChangedParameter)
            if name in held
        ],
        loras_added=list(held["loras_added"]),
        loras_removed=list(held["loras_removed"]),
        lora_uuids_added=list(held["lora_uuids_added"]),
        lora_uuids_removed=list(held["lora_uuids_removed"]),
    )


def evolution_document(view: dict, *, render_id: int | None) -> EvolutionView:
    """The measured view as the contract states it.

    Identities into addresses happens HERE and nowhere else: a member's
    slug becomes its thumbnail and its page, the session's local day the
    gallery's day-facet link. db/evolution.py owns no URL, and building
    this rather than mutating what it returned is what keeps that true.
    """
    day = view["identities"]["local_day"]
    return EvolutionView(
        v=view["v"],
        plan=EvolutionPlan(
            id=view["plan"]["id"],
            sha256=view["plan"]["sha256"],
            format=view["plan"]["format"],
            sequenced=view["plan"]["sequenced"],
            unsupported=[
                EvolutionUnsupported(kind=one["kind"], reason=one["reason"], member_refs=one.get("member_refs"))
                for one in view["plan"]["unsupported"]
            ],
            label=view["plan"]["label"],
        ),
        snapshot=EvolutionSnapshot(
            sha256=view["snapshot"]["sha256"],
            time=EvolutionTime(
                local=view["snapshot"]["time"]["local"],
                instant=view["snapshot"]["time"]["instant"],
            ),
            members=view["snapshot"]["members"],
            subject=view["snapshot"]["subject"],
        ),
        semantic=EvolutionSemantic(
            provider=view["semantic"]["provider"],
            space_id=view["semantic"]["space_id"],
            space=view["semantic"]["space"],
            prompt_policy_hash=view["semantic"]["prompt_policy_hash"],
            unavailable=view["semantic"]["unavailable"],
        ),
        phases=[
            EvolutionPhase(
                id=phase["id"],
                label=phase["label"],
                member_refs=list(phase["member_refs"]),
                representative_refs=list(phase["representative_refs"]),
                claims=[EvolutionClaim(id=one["id"], kind=one["kind"], facts=one["facts"]) for one in phase["claims"]],
            )
            for phase in view["phases"]
        ],
        members=[
            EvolutionMember(
                ref=member["ref"],
                phase_ref=member["phase_ref"],
                media=EvolutionMedia(
                    uuid=member["media"]["uuid"],
                    name=member["media"]["name"],
                    kind=member["media"]["kind"],
                    content_sha256=member["media"]["content_sha256"],
                    slug=member["media"]["slug"],
                    thumbnail=f"/thumb/{member['media']['slug']}" if member["media"]["slug"] else None,
                    page=f"/i/{member['media']['slug']}" if member["media"]["slug"] else None,
                ),
                occurrence=_occurrence_of(member["occurrence"]),
                prompt=EvolutionPrompts(
                    effective=_prompt_of(member["prompt"]["effective"]),
                    original=_prompt_of(member["prompt"]["original"]),
                ),
                generation=EvolutionGeneration(
                    seed=member["generation"]["seed"],
                    steps=member["generation"]["steps"],
                    cfg=member["generation"]["cfg"],
                    denoise=member["generation"]["denoise"],
                    clip_skip=member["generation"]["clip_skip"],
                    sampler=member["generation"]["sampler"],
                    scheduler=member["generation"]["scheduler"],
                    width=member["generation"]["width"],
                    height=member["generation"]["height"],
                    tool=member["generation"]["tool"],
                    model=member["generation"]["model"],
                    loras=list(member["generation"]["loras"]),
                    lora_uuids=list(member["generation"]["lora_uuids"]),
                ),
                metrics=EvolutionMetrics(
                    original_effective_cosine=member["metrics"]["original_effective_cosine"],
                    original_effective_cosine_unavailable=member["metrics"].get(
                        "original_effective_cosine_unavailable"
                    ),
                    text_image_cosine=member["metrics"]["text_image_cosine"],
                    text_image_cosine_unavailable=member["metrics"].get("text_image_cosine_unavailable"),
                ),
            )
            for member in view["members"]
        ],
        transitions=[
            EvolutionTransition(
                before=one["from"],
                after=one["to"],
                phase_boundary=one["phase_boundary"],
                prompt_cosine=one["prompt_cosine"],
                prompt_cosine_unavailable=one.get("prompt_cosine_unavailable"),
                visual_cosine=one["visual_cosine"],
                visual_cosine_unavailable=one.get("visual_cosine_unavailable"),
                changes=_changes_of(one["changes"]),
            )
            for one in view["transitions"]
        ],
        lineage=[
            EvolutionEdge(parent=edge["parent"], child=edge["child"], kind=edge["kind"]) for edge in view["lineage"]
        ],
        links=EvolutionLinks(
            story=f"/stories/renders/{render_id}" if render_id is not None else None,
            gallery_day=(
                "/g?" + urllib.parse.urlencode([("f", facets.spell(facets.facet("context.local_day", "eq", day)))])
                if day
                else None
            ),
            search="/search?q=",
            neighbours="/prompts/{id}/neighbours",
        ),
    )


@get(
    "/stories/plans/{plan_id:int}/evolution",
    # The route negotiates, and a union that mixes a page with a JSON
    # answer reaches OpenAPI as the empty schema however precisely the arms
    # are written (litestar v2.24.0). The JSON answer is declared here.
    responses={
        200: ResponseSpec(
            data_container=EvolutionView,
            description="One story plan measured over its frozen evidence, in one semantic space",
            media_type=MediaType.JSON,
            generate_examples=False,
        )
    },
    sync_to_thread=True,
)
def plan_evolution(
    state: State, plan_id: FromPath[int], request: Request, space: FromQuery[str | None] = None
) -> Response[EvolutionView] | Template:
    """The Generation Evolution Explorer: a read-only view of one plan
    (db/evolution.py) -- JSON, or the page by Accept. `space` names
    the provider whose joint space and query policy every metric is
    measured in; the first configured provider otherwise. No writes,
    no model loads, no embedding computation."""
    conn = connect.connect(state.db_path, read_only=True)
    try:
        weights = str(home.models_dir(pathlib.Path(state.home), settings.value(conn, "models_dir")))
        try:
            view = evolution.load(conn, plan_id, provider=space, models_dir=weights)
        except LookupError as missing:
            raise NotFoundException(str(missing)) from missing
        except stories.Corrupt as corrupt:
            raise ClientException(str(corrupt), status_code=409) from corrupt
        except ValueError as refused:
            raise ClientException(str(refused)) from refused
        render_id = rendering.latest_render_id(conn, plan_id)
    finally:
        connect.close(conn)
    document = evolution_document(view, render_id=render_id)
    if wants_json(request):
        return Response(document, headers=VARIES)
    # The page renders its shell from the same document, as HTML. What it
    # does NOT do is serialize the document into the HTML for the browser
    # to parse back out: the explorer asks this route for it, by Accept,
    # and is given the one contract OpenAPI describes.
    return Template(
        template_name="evolution.html",
        context={"view": document.model_dump(mode="json"), "plan_id": plan_id},
        headers=VARIES,
    )
