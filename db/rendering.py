"""StoryRender: a story written from frozen evidence and an evidence-
backed plan -- structure, not HTML; the deterministic narrator that
every later narrator is measured against.

TemplateStoryRenderer receives exactly two immutable, VERIFIED things:
a StorySnapshot (db/stories.py) and a StoryPlan (db/planning.py). It
decides nothing: no phases, no similarity, no database, no model. It
selects deterministic wording for each Claim through a CLOSED registry
of Adapters (story_renderers/claims.py), spells values through one
formatting Module, and assembles a StoryRender document: title, dek,
summary, sections that keep their claim_refs and member/hero refs, and
notes that say what the evidence does NOT support. Jinja, in the web
adapter, lays that document out; it understands none of it.

The chain a reader can follow is the whole point: a sentence cites a
Claim, the Claim cites evidence refs, the refs resolve to frozen
snapshot facts. `violations()` proves that chain for every render
before it is persisted and again on every read -- and enforces the
rule that when the plan declares chronology unsupported, the render
contains no sequencing language at all. The deterministic narrator is
the oracle the LLM narrator will be tested against on exactly these
invariants: cite only plan Claims, invent nothing, keep unsupported
unsupported, name no member outside the snapshot, touch no database.

Profiles change emphasis, never truth; the two identities are the
planner's: a REQUEST identity known before any work (format, plan sha,
snapshot sha, renderer kind/version, profile, locale, wording policy)
and a DOCUMENT identity (canonical sha, rendered_at excluded).
Rendering is milliseconds of pure code and happens synchronously;
story_render rows are insert-only and re-verified on read.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re

from story_renderers import claims as wording
from story_renderers import formatting

from . import planning, stories
from .stories import canonical, digest

FORMAT_VERSION = 1
PROFILES = ("memory", "technical", "compact")
LOCALES = ("en",)

#: Words that assert an order in time. A render of a plan whose
#: chronology is unsupported may contain none of them -- the rule is
#: enforced, not admired.
_SEQUENCING = re.compile(
    r"\b(first|firstly|then|later|finally|next|earlier|before|after|afterwards|subsequently|previous|previously"
    r"|followed|following|began|ended|started|last)\b",
    re.IGNORECASE,
)


@dataclasses.dataclass(frozen=True)
class RenderRef:
    id: int
    sha256: str
    reused: bool


class TemplateStoryRenderer:
    """The deterministic narrator. `render(snapshot, plan)` is a pure
    function of its two documents and the profile."""

    kind = "template"
    version = 1

    def __init__(self, profile: str = "memory", locale: str = "en"):
        if profile not in PROFILES:
            raise ValueError(f"no render profile named {profile!r}; one of {', '.join(PROFILES)}")
        if locale not in LOCALES:
            raise ValueError(f"no locale named {locale!r}; one of {', '.join(LOCALES)}")
        self.profile = profile
        self.locale = locale

    @property
    def wording_policy(self) -> int:
        return wording.POLICY_VERSION

    def render(self, snapshot: dict, plan: dict, snapshot_sha256: str, plan_sha256: str) -> dict:
        if snapshot.get("v") != 1 or plan.get("v") != planning.FORMAT_VERSION:
            raise ValueError("this renderer reads StorySnapshot v1 and StoryPlan v1")
        if plan["snapshot_sha256"] != snapshot_sha256:
            raise ValueError("the plan was not made from this snapshot")
        members = {planning._member_ref(one["ordinal"]): one for one in snapshot["members"]}
        total = len(members)
        sequenced = bool(plan["subject"]["sequenced"])
        ctx = wording.Context(snapshot=snapshot, plan=plan, profile=self.profile, sequenced=sequenced)
        claims_by_id = {claim["id"]: claim for claim in plan["claims"]}

        day = formatting.day_label(_day_anchor(snapshot))
        tool = next(((m.get("generation") or {}).get("tool") for m in snapshot["members"] if m.get("generation")), None)
        what = f"{total} {tool} images" if tool else f"{formatting.count(total, 'generated image')}"
        title = f"{what} from {day}" if day else what
        groups = "phases" if sequenced else "prompt families"
        count_groups = formatting.count(len(plan["phases"]), groups.removesuffix("s"), groups)
        dek = f"{count_groups} across {formatting.count(total, 'generated image')}"
        these = f"These {formatting.count(total, 'generated image')}"
        summary = f"{these} fall into {count_groups}."
        if day:
            summary = f"{these} were generated on {day} and fall into {count_groups}."

        sections = []
        if self.profile != "compact":
            for number, phase in enumerate(plan["phases"], start=1):
                paragraphs = [f"{formatting.count(len(phase['member_refs']), 'image')}."]
                cited = []
                for claim_id in phase["claim_refs"]:
                    sentence = wording.word(claims_by_id[claim_id], phase, ctx)
                    if sentence is not None:
                        paragraphs.append(sentence)
                        cited.append(claim_id)
                sections.append(
                    {
                        "id": f"section-{number:03d}",
                        "title": phase["label_hint"],
                        "paragraphs": paragraphs,
                        "hero_refs": list(phase["representative_refs"]),
                        "member_refs": list(phase["member_refs"]),
                        "claim_refs": cited,
                    }
                )

        notes = []
        for told in plan["unsupported"]:
            if told["kind"] == "chronology":
                where = f" within {day}" if day else ""
                text = f"Available evidence does not establish the order of these images{where}."
                notes.append({"kind": "chronology", "text": text})
            elif told["kind"] == "prompt_evidence":
                n = len(told.get("member_refs", []))
                notes.append(
                    {
                        "kind": "prompt_evidence",
                        "text": f"Prompt text is not available for {formatting.count(n, 'image')}.",
                    }
                )

        return {
            "v": FORMAT_VERSION,
            "snapshot_sha256": snapshot_sha256,
            "plan_sha256": plan_sha256,
            "renderer": {
                "kind": self.kind,
                "version": self.version,
                "profile": self.profile,
                "locale": self.locale,
                "wording_policy": self.wording_policy,
            },
            "title": title,
            "dek": dek,
            "summary": summary,
            "sections": sections,
            "notes": notes,
        }


def _day_anchor(snapshot: dict) -> float | None:
    """The day the story is about: the event's wall-clock start when it
    has one, its instant start otherwise, nothing when neither."""
    when = snapshot["subject"]["time"]
    if when.get("local"):
        return when["local"][0]
    if when.get("instant"):
        return when["instant"][0]
    return None


# --- the exact grammar -------------------------------------------------------


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _strings(value) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def _keys(node, exact: set[str], optional: set[str], where: str, bad: list[str]) -> bool:
    if not isinstance(node, dict):
        bad.append(f"{where} is not an object")
        return False
    held = set(node)
    if held - exact - optional or exact - held:
        bad.append(f"{where} keys are {sorted(held)}, not {sorted(exact)} (+ optional {sorted(optional)})")
        return False
    return True


_SHA = re.compile(r"^[0-9a-f]{64}$")
_SECTION_ID = re.compile(r"^section-[0-9]{3}$")
_NOTE_KINDS = {"chronology", "prompt_evidence"}


def validate_story_render_v1(render) -> list[str]:
    """The exact grammar of a StoryRender v1: shape, types, vocabularies.
    Controlled reasons for any bytes, never an exception."""
    bad: list[str] = []
    top = {"v", "snapshot_sha256", "plan_sha256", "renderer", "title", "dek", "summary", "sections", "notes"}
    if not _keys(render, top, {"rendered_at"}, "render", bad):
        return bad
    if render["v"] != FORMAT_VERSION:
        bad.append(f"format v{render['v']!r}, not v{FORMAT_VERSION}")
    bad.extend(
        f"{key} is not a sha256"
        for key in ("snapshot_sha256", "plan_sha256")
        if not (isinstance(render[key], str) and _SHA.match(render[key]))
    )
    if "rendered_at" in render and not _number(render["rendered_at"]):
        bad.append("rendered_at is not a number")
    who = render["renderer"]
    if _keys(who, {"kind", "version", "profile", "locale", "wording_policy"}, set(), "renderer", bad):
        if who["kind"] != "template":
            bad.append(f"unknown renderer {who['kind']!r}")
        bad.extend(
            f"renderer.{key} is not an integer"
            for key in ("version", "wording_policy")
            if not isinstance(who[key], int) or isinstance(who[key], bool)
        )
        if who["profile"] not in PROFILES:
            bad.append(f"unknown profile {who['profile']!r}")
        if who["locale"] not in LOCALES:
            bad.append(f"unknown locale {who['locale']!r}")
    bad.extend(
        f"{key} is not a non-empty string"
        for key in ("title", "dek", "summary")
        if not isinstance(render[key], str) or not render[key].strip()
    )
    if not isinstance(render["sections"], list) or not isinstance(render["notes"], list):
        bad.append("sections and notes are lists")
        return bad
    for i, section in enumerate(render["sections"]):
        where = f"sections[{i}]"
        exact = {"id", "title", "paragraphs", "hero_refs", "member_refs", "claim_refs"}
        if not _keys(section, exact, set(), where, bad):
            continue
        if not (isinstance(section["id"], str) and _SECTION_ID.match(section["id"])):
            bad.append(f"{where}.id is not a section id")
        if not isinstance(section["title"], str) or not section["title"].strip():
            bad.append(f"{where}.title is not a non-empty string")
        if not _strings(section["paragraphs"]) or not section["paragraphs"]:
            bad.append(f"{where}.paragraphs is not a non-empty list of strings")
        bad.extend(
            f"{where}.{key} is not a list of refs"
            for key in ("hero_refs", "member_refs", "claim_refs")
            if not _strings(section[key])
        )
        if _strings(section["member_refs"]) and not section["member_refs"]:
            bad.append(f"{where}.member_refs is empty: a section must be about some member")
    for i, note in enumerate(render["notes"]):
        where = f"notes[{i}]"
        if not _keys(note, {"kind", "text"}, set(), where, bad):
            continue
        if note["kind"] not in _NOTE_KINDS:
            bad.append(f"{where}.kind {note['kind']!r} is not a known note kind")
        if not isinstance(note["text"], str) or not note["text"].strip():
            bad.append(f"{where}.text is not a non-empty string")
    return bad


def violations(render: dict, plan: dict, snapshot: dict, snapshot_sha256: str, plan_sha256: str) -> list[str]:
    """Every way a render can be wrong about its plan and snapshot --
    the chain sentence -> Claim -> evidence -> frozen fact, and the
    unsupported-chronology rule. Empty is the only acceptable answer."""
    bad = validate_story_render_v1(render)
    if bad:
        return bad
    if render["snapshot_sha256"] != snapshot_sha256:
        bad.append("the render names a different snapshot")
    if render["plan_sha256"] != plan_sha256:
        bad.append("the render names a different plan")
    claims = {claim["id"] for claim in plan["claims"]}
    members = {planning._member_ref(one["ordinal"]) for one in snapshot["members"]}
    section_ids = [section["id"] for section in render["sections"]]
    if len(set(section_ids)) != len(section_ids):
        bad.append("section ids are not unique")
    for section in render["sections"]:
        bad.extend(f"{section['id']} cites unknown claim {ref}" for ref in section["claim_refs"] if ref not in claims)
        for key in ("hero_refs", "member_refs"):
            bad.extend(f"{section['id']} names unknown member {ref}" for ref in section[key] if ref not in members)
        bad.extend(
            f"{section['id']} hero {ref} is not one of its members"
            for ref in section["hero_refs"]
            if ref not in section["member_refs"]
        )
    unsupported = {told["kind"] for told in plan["unsupported"]}
    if "chronology" in unsupported:
        if not any(note["kind"] == "chronology" for note in render["notes"]):
            bad.append("the plan declares chronology unsupported and the render does not say so")
        prose = [render["title"], render["dek"], render["summary"]]
        prose += [text for section in render["sections"] for text in (section["title"], *section["paragraphs"])]
        for text in prose:
            hit = _SEQUENCING.search(text)
            if hit:
                bad.append(f"chronology is unsupported but the render sequences: {hit.group(0)!r} in {text!r}")
    return bad


# --- identities --------------------------------------------------------------


def identity(render: dict) -> tuple[str, str]:
    spelled = canonical({key: value for key, value in render.items() if key != "rendered_at"})
    return spelled, digest(spelled)


def request_identity(
    plan_sha256: str, snapshot_sha256: str, kind: str, version: int, profile: str, locale: str, wording_policy: int
) -> str:
    return digest(
        canonical(
            {
                "format": FORMAT_VERSION,
                "plan_sha256": plan_sha256,
                "snapshot_sha256": snapshot_sha256,
                "renderer": {"kind": kind, "version": version},
                "profile": profile,
                "locale": locale,
                "wording_policy": wording_policy,
            }
        )
    )


# --- persistence ---------------------------------------------------------------


def _verified_inputs(conn, plan_id: int) -> tuple[dict, str, dict, str, int]:
    """The plan, re-verified (db/planning.py load_plan re-hashes it and
    re-validates it against its re-verified snapshot), and the snapshot
    by value with both identities."""
    row = conn.execute("SELECT snapshot_id, document_sha256 FROM story_plan WHERE id = ?", (plan_id,)).fetchone()
    if row is None:
        raise LookupError(f"no story plan {plan_id}")
    plan = planning.load_plan(conn, plan_id)
    snapshot = stories.load_snapshot(conn, int(row[0]))
    return plan, row[1], snapshot, plan["snapshot_sha256"], int(row[0])


def render_plan(conn, plan_id: int, renderer: TemplateStoryRenderer, now: float) -> RenderRef:
    """Render one verified plan under one profile and persist -- or
    return the existing row for the same request. Pure code, so it is
    synchronous; the transaction is left open for the caller."""
    plan, plan_sha, snapshot, snapshot_sha, snapshot_id = _verified_inputs(conn, plan_id)
    who = renderer
    request = request_identity(
        plan_sha, snapshot_sha, who.kind, who.version, who.profile, who.locale, who.wording_policy
    )
    held = conn.execute("SELECT id, document_sha256 FROM story_render WHERE request_sha256 = ?", (request,)).fetchone()
    if held:
        return RenderRef(int(held[0]), held[1], True)
    render = renderer.render(snapshot, plan, snapshot_sha, plan_sha)
    wrong = violations(render, plan, snapshot, snapshot_sha, plan_sha)
    if wrong:
        raise AssertionError(f"the renderer produced an invalid render: {wrong}")
    sha = identity(render)[1]
    held = conn.execute("SELECT id FROM story_render WHERE document_sha256 = ?", (sha,)).fetchone()
    if held:
        return RenderRef(int(held[0]), sha, True)
    render["rendered_at"] = now
    render_id = int(
        conn.execute(
            "INSERT INTO story_render(plan_id, snapshot_id, format_version, renderer, renderer_version, profile,"
            " locale, wording_policy, request_sha256, document_json, document_sha256, created_at)"
            " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                plan_id,
                snapshot_id,
                FORMAT_VERSION,
                renderer.kind,
                renderer.version,
                renderer.profile,
                renderer.locale,
                renderer.wording_policy,
                request,
                canonical(render),
                sha,
                now,
            ),
        ).lastrowid
        or 0
    )
    return RenderRef(render_id, sha, False)


def load_render_with_members(conn, render_id: int) -> tuple[dict, dict]:
    """The verified render and its snapshot's members by ref -- what a
    page needs to spell a hero's FROZEN name. The page asks here; it
    never joins a table itself."""
    render = load_render(conn, render_id)
    snapshot_id = conn.execute("SELECT snapshot_id FROM story_render WHERE id = ?", (render_id,)).fetchone()[0]
    members = {
        planning._member_ref(one["ordinal"]): one for one in stories.load_snapshot(conn, int(snapshot_id))["members"]
    }
    return render, members


def load_render(conn, render_id: int) -> dict:
    """The stored render, RE-VERIFIED on read: it must hash to its stored
    identity and still pass every violation check against its
    re-verified plan and snapshot. A page lays out only what passes."""
    row = conn.execute(
        "SELECT plan_id, document_json, document_sha256 FROM story_render WHERE id = ?", (render_id,)
    ).fetchone()
    if row is None:
        raise LookupError(f"no story render {render_id}")
    render = json.loads(row[1])
    if identity(render)[1] != row[2]:
        raise ValueError(f"story render {render_id} no longer hashes to its identity; refusing to serve it")
    plan, plan_sha, snapshot, snapshot_sha, _ = _verified_inputs(conn, int(row[0]))
    wrong = violations(render, plan, snapshot, snapshot_sha, plan_sha)
    if wrong:
        raise ValueError(f"story render {render_id} is no longer valid against its plan: {wrong}")
    return render
