"""StoryRender: a story written from frozen evidence and an evidence-
backed plan -- structure, not HTML; the deterministic narrator that
every later narrator is measured against.

TemplateStoryRenderer receives exactly two immutable, VERIFIED things:
a StorySnapshot (db/stories.py) and a StoryPlan (db/planning.py). It
decides nothing: no phases, no similarity, no database, no model. It
selects deterministic wording for each Claim through a CLOSED registry
of Adapters (story_renderers/claims.py), spells values through one
formatting Module, and assembles a StoryRender document: a lede (title,
dek, summary) that declares its structural support, one section per
plan phase made of BLOCKS -- each block is either `structure` (a count,
supported by member refs) or `claim` (a sentence, supported by claim
refs) -- and notes that say what the evidence does NOT support. Jinja,
in the web adapter, lays that document out; it understands none of it.

The chain a reader can follow is the whole point, and `violations()`
proves it mechanically for every render before persistence and again on
every read: every section names its plan phase and carries exactly that
phase's members; every hero is one of that phase's representatives;
every claim block cites only Claims the plan attached to THAT phase;
every structure block names members of its own section; no block is
without support. When the plan declares chronology unsupported the
render carries the note and cites no directional Claim -- the plan
cannot contain one (db/planning.py validate_plan), so the structural
rule holds by construction, and a lexical scan for sequencing words
stays as defense in depth, exempting frozen evidence names.

The v1 grammar is FROZEN in `STORY_RENDER_V1`: a v1 row written today
parses as v1 after a v2 exists. The renderer declares which input
versions it reads as literals, and the document records which it read.

What `violations()` proves is ATTRIBUTION -- every block names the
Claim or members it rests on, and those belong to the block's own
phase. It does not prove ENTAILMENT: a block citing a prompt-similarity
Claim with the text "the user hated the lighting" is a provenance
closure over a false sentence. The deterministic narrator is truthful
because a closed wording registry writes its sentences; a narrator that
composes prose needs a separate faithfulness check, and this validator
must never be mistaken for one.

Two identities, the planner's: a REQUEST identity known before any work
(render format, plan sha, snapshot sha, renderer kind/version, profile,
locale, and the render POLICY -- one token covering every
output-affecting behaviour: wording, formatting, lede, profile
membership) and a DOCUMENT identity (canonical sha, rendered_at
excluded). Rendering is milliseconds of pure code and happens
synchronously; story_render rows are insert-only and re-verified on
read.
"""

from __future__ import annotations

import dataclasses
import math
import re

import story_renderers
from story_renderers import claims as wording
from story_renderers import formatting

from . import planning, stories
from .stories import canonical, digest

FORMAT_VERSION = 1
PROFILES = ("memory", "technical", "compact")
LOCALES = ("en",)

#: Words that assert an order in time -- defense in depth behind the
#: structural rule (no directional Claim without chronology). A hit
#: inside a frozen evidence name (an artifact called "AfterDetail") is
#: evidence, not narration, and is exempt.
_SEQUENCING = re.compile(
    r"\b(first|firstly|then|later|finally|next|earlier|before|after|afterwards|subsequently|previous|previously"
    r"|followed|following|began|ended|started|last|new|initial|initially|eventually|subsequent)\b",
    re.IGNORECASE,
)


@dataclasses.dataclass(frozen=True)
class RenderRef:
    id: int
    sha256: str
    reused: bool


#: What each renderer VERSION reads -- FROZEN, per version, forever.
#: Changing the inputs a renderer accepts is an Interface change and is
#: a new version; a stored render names its version, and `violations()`
#: holds the inputs it says it read to this map, so a later version can
#: never reinterpret what an earlier one produced. Template v1 read
#: Plan v1 and v2; v2 stopped reading Plan v1, whose producer wrote
#: directed claims for unsequenced families, and learned Plan v3.
COMPATIBILITY: dict[tuple[str, int], dict[str, frozenset[int]]] = {
    ("template", 1): {"snapshot": frozenset({1}), "plan": frozenset({1, 2})},
    ("template", 2): {"snapshot": frozenset({1}), "plan": frozenset({2, 3})},
}


class TemplateStoryRenderer:
    """The deterministic narrator. `render(snapshot, plan)` is a pure
    function of its two documents and the profile."""

    kind = "template"
    version = 2

    @property
    def reads(self) -> dict[str, frozenset[int]]:
        return COMPATIBILITY[(self.kind, self.version)]

    def __init__(self, profile: str = "memory", locale: str = "en"):
        if profile not in PROFILES:
            raise ValueError(f"no render profile named {profile!r}; one of {', '.join(PROFILES)}")
        if locale not in LOCALES:
            raise ValueError(f"no locale named {locale!r}; one of {', '.join(LOCALES)}")
        self.profile = profile
        self.locale = locale

    @property
    def policy(self) -> int:
        return story_renderers.POLICY_VERSION

    def render(self, snapshot: dict, plan: dict, snapshot_sha256: str, plan_sha256: str) -> dict:
        if snapshot.get("v") not in self.reads["snapshot"] or plan.get("v") not in self.reads["plan"]:
            raise ValueError(
                f"renderer {self.kind} v{self.version} reads StorySnapshot {sorted(self.reads['snapshot'])}"
                f" and StoryPlan {sorted(self.reads['plan'])}, not snapshot v{snapshot.get('v')!r}"
                f" with plan v{plan.get('v')!r}"
            )
        if plan["snapshot_sha256"] != snapshot_sha256:
            raise ValueError("the plan was not made from this snapshot")
        members = {planning._member_ref(one["ordinal"]): one for one in snapshot["members"]}
        total = len(members)
        sequenced = bool(plan["subject"]["sequenced"])
        ctx = wording.Context(snapshot=snapshot, plan=plan, profile=self.profile, sequenced=sequenced)
        claims_by_id = {claim["id"]: claim for claim in plan["claims"]}

        day, one_day = _day_label(snapshot)
        # a session is grouped by time and may mix tools: a tool is named
        # only when every member agrees on it
        tools = {(m.get("generation") or {}).get("tool") for m in snapshot["members"] if m.get("generation")}
        tool = tools.pop() if len(tools) == 1 else None
        what = f"{total} {tool} images" if tool else f"{formatting.count(total, 'generated image')}"
        title = f"{what} from {day}" if day else what
        groups = "phases" if sequenced else "prompt families"
        count_groups = formatting.count(len(plan["phases"]), groups.removesuffix("s"), groups)
        dek = f"{count_groups} across {formatting.count(total, 'generated image')}"
        these = f"These {formatting.count(total, 'generated image')}"
        summary = f"{these} fall into {count_groups}."
        if day:
            when = f"on {day}" if one_day else f"over {day}"
            summary = f"{these} were generated {when} and fall into {count_groups}."

        sections = []
        if self.profile != "compact":
            for number, phase in enumerate(plan["phases"], start=1):
                blocks = [
                    {
                        "kind": "structure",
                        "text": f"{formatting.count(len(phase['member_refs']), 'image')}.",
                        "member_refs": list(phase["member_refs"]),
                    }
                ]
                cited = []
                for claim_id in phase["claim_refs"]:
                    sentence = wording.word(claims_by_id[claim_id], phase, ctx)
                    if sentence is not None:
                        blocks.append({"kind": "claim", "text": sentence, "claim_refs": [claim_id]})
                        cited.append(claim_id)
                sections.append(
                    {
                        "id": f"section-{number:03d}",
                        "phase_ref": phase["id"],
                        "title": phase["label_hint"],
                        "blocks": blocks,
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
                "policy": self.policy,
                "reads": {"snapshot": snapshot["v"], "plan": plan["v"]},
            },
            "title": title,
            "dek": dek,
            "summary": summary,
            "support": {"member_refs": sorted(members), "phase_refs": [phase["id"] for phase in plan["phases"]]},
            "sections": sections,
            "notes": notes,
        }


def _day_label(snapshot: dict) -> tuple[str | None, bool]:
    """The days the story spans, in the domain the evidence claims them
    in: the event's wall-clock interval spells the human's own calendar
    days; an instant interval with no wall clock is UTC and says so --
    the two domains the snapshot keeps apart are never fused here. A
    session that crosses midnight spans two days and is narrated as a
    range, never as "on" either. Returns (label, single day?)."""
    when = snapshot["subject"]["time"]
    for domain, utc in (("local", False), ("instant", True)):
        held = when.get(domain)
        if held:
            start, end = held[0], held[1] if len(held) > 1 else None
            label = formatting.day_range(start, end, utc=utc)
            return label, label == formatting.day_label(start, utc=utc)
    return None, True


# --- the exact grammar -------------------------------------------------------

#: The durable vocabulary of a StoryRender v1 -- FROZEN. Describes what
#: a v1 document may contain and never tracks the running code: a v1
#: row written today must still parse as v1 after a v2 exists. Adding a
#: renderer, a profile, a block kind, or a key is a v2.
STORY_RENDER_V1 = {
    "version": 1,
    "renderers": frozenset({"template"}),
    "profiles": frozenset({"memory", "technical", "compact"}),
    "locales": frozenset({"en"}),
    "blocks": frozenset({"structure", "claim"}),
    "notes": frozenset({"chronology", "prompt_evidence"}),
}


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _in(value, vocabulary) -> bool:
    """Membership that cannot raise: an unhashable value is simply not
    in a vocabulary of strings."""
    return isinstance(value, str) and value in vocabulary


def _strings(value) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


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
_SECTION_ID = re.compile(r"^section-[0-9]{3,}$")
_PHASE_ID = re.compile(r"^phase-[0-9]{3,}$")


def validate_story_render_v1(render) -> list[str]:
    """The exact grammar of a StoryRender v1 against the FROZEN v1
    vocabulary, with no reference to any plan, snapshot or running
    code: shape, types, vocabularies, cardinalities. Controlled reasons
    for any bytes -- unhashable kinds included -- never an exception."""
    bad: list[str] = []
    top = {"v", "snapshot_sha256", "plan_sha256", "renderer", "title", "dek", "summary", "support", "sections", "notes"}
    if not _keys(render, top, {"rendered_at"}, "render", bad):
        return bad
    if render["v"] != STORY_RENDER_V1["version"]:
        bad.append(f"format v{render['v']!r}, not v1")
    bad.extend(
        f"{key} is not a sha256"
        for key in ("snapshot_sha256", "plan_sha256")
        if not (isinstance(render[key], str) and _SHA.match(render[key]))
    )
    if "rendered_at" in render and not _number(render["rendered_at"]):
        bad.append("rendered_at is not a number")
    who = render["renderer"]
    if _keys(who, {"kind", "version", "profile", "locale", "policy", "reads"}, set(), "renderer", bad):
        if not _in(who["kind"], STORY_RENDER_V1["renderers"]):
            bad.append(f"unknown renderer {who['kind']!r}")
        bad.extend(
            f"renderer.{key} is not a positive integer" for key in ("version", "policy") if not _integer(who[key], 1)
        )
        if not _in(who["profile"], STORY_RENDER_V1["profiles"]):
            bad.append(f"unknown profile {who['profile']!r}")
        if not _in(who["locale"], STORY_RENDER_V1["locales"]):
            bad.append(f"unknown locale {who['locale']!r}")
        if _keys(who["reads"], {"snapshot", "plan"}, set(), "renderer.reads", bad):
            bad.extend(
                f"renderer.reads.{key} is not a positive integer"
                for key in ("snapshot", "plan")
                if not _integer(who["reads"][key], 1)
            )
    bad.extend(f"{key} is not a non-empty string" for key in ("title", "dek", "summary") if not _text(render[key]))
    if _keys(render["support"], {"member_refs", "phase_refs"}, set(), "support", bad):
        bad.extend(
            f"support.{key} is not a non-empty list of refs"
            for key in ("member_refs", "phase_refs")
            if not _strings(render["support"][key]) or not render["support"][key]
        )
    if not isinstance(render["sections"], list) or not isinstance(render["notes"], list):
        bad.append("sections and notes are lists")
        return bad
    for i, section in enumerate(render["sections"]):
        where = f"sections[{i}]"
        exact = {"id", "phase_ref", "title", "blocks", "hero_refs", "member_refs", "claim_refs"}
        if not _keys(section, exact, set(), where, bad):
            continue
        if not (isinstance(section["id"], str) and _SECTION_ID.match(section["id"])):
            bad.append(f"{where}.id is not a section id")
        if not (isinstance(section["phase_ref"], str) and _PHASE_ID.match(section["phase_ref"])):
            bad.append(f"{where}.phase_ref is not a phase id")
        if not _text(section["title"]):
            bad.append(f"{where}.title is not a non-empty string")
        bad.extend(
            f"{where}.{key} is not a list of refs"
            for key in ("hero_refs", "member_refs", "claim_refs")
            if not _strings(section[key])
        )
        if _strings(section["member_refs"]) and not section["member_refs"]:
            bad.append(f"{where}.member_refs is empty: a section must be about some member")
        if not isinstance(section["blocks"], list) or not section["blocks"]:
            bad.append(f"{where}.blocks is not a non-empty list")
            continue
        for j, block in enumerate(section["blocks"]):
            at = f"{where}.blocks[{j}]"
            if not isinstance(block, dict) or not _in(block.get("kind"), STORY_RENDER_V1["blocks"]):
                bad.append(f"{at} is not a block of a known kind")
                continue
            support = "member_refs" if block["kind"] == "structure" else "claim_refs"
            if not _keys(block, {"kind", "text", support}, set(), at, bad):
                continue
            if not _text(block["text"]):
                bad.append(f"{at}.text is not a non-empty string")
            if not _strings(block[support]) or not block[support]:
                bad.append(f"{at}.{support} is not a non-empty list of refs: a block without support is prose")
    for i, note in enumerate(render["notes"]):
        where = f"notes[{i}]"
        if not _keys(note, {"kind", "text"}, set(), where, bad):
            continue
        if not _in(note["kind"], STORY_RENDER_V1["notes"]):
            bad.append(f"{where}.kind {note['kind']!r} is not a known note kind")
        if not _text(note["text"]):
            bad.append(f"{where}.text is not a non-empty string")
    return bad


_GRAMMARS = {1: validate_story_render_v1}


def validate_story_render(render) -> list[str]:
    """Dispatch on the document's OWN version: a v1 row is judged by the
    frozen v1 grammar however many versions exist later."""
    version = render.get("v") if isinstance(render, dict) else None
    grammar = _GRAMMARS.get(version) if isinstance(version, int) and not isinstance(version, bool) else None
    if grammar is None:
        return [f"no StoryRender grammar for version {version!r}; known: {sorted(_GRAMMARS)}"]
    return grammar(render)


def _frozen_names(snapshot: dict) -> list[str]:
    names = {member["name"] for member in snapshot["members"] if isinstance(member.get("name"), str)}
    for member in snapshot["members"]:
        for artifact in (member.get("generation") or {}).get("artifacts") or []:
            if isinstance(artifact.get("name"), str):
                names.add(artifact["name"])
    return sorted(names, key=len, reverse=True)


def _sequencing_words(text: str, names: list[str]) -> list[str]:
    """Sequencing words in `text` that are NOT inside an occurrence of
    a frozen evidence name."""
    covered: list[tuple[int, int]] = []
    for name in names:
        if _SEQUENCING.search(name):
            start = text.find(name)
            while start >= 0:
                covered.append((start, start + len(name)))
                start = text.find(name, start + 1)
    return [
        hit.group(0)
        for hit in _SEQUENCING.finditer(text)
        if not any(s <= hit.start() and hit.end() <= e for s, e in covered)
    ]


def violations(render: dict, plan: dict, snapshot: dict, snapshot_sha256: str, plan_sha256: str) -> list[str]:
    """Every way a render can be wrong about its plan and snapshot -- the
    chain block -> Claim -> phase -> frozen members, and the unsupported-
    chronology rule. Empty is the only acceptable answer."""
    bad = validate_story_render(render)
    if bad:
        return bad
    if render["snapshot_sha256"] != snapshot_sha256:
        bad.append("the render names a different snapshot")
    if render["plan_sha256"] != plan_sha256:
        bad.append("the render names a different plan")
    read = render["renderer"]["reads"]
    if read != {"snapshot": snapshot.get("v"), "plan": plan.get("v")}:
        bad.append(f"the render says it read snapshot v{read['snapshot']} and plan v{read['plan']}; it did not")
    who = (render["renderer"]["kind"], render["renderer"]["version"])
    able = COMPATIBILITY.get(who)
    if able is None:
        bad.append(f"no frozen compatibility is recorded for renderer {who[0]} v{who[1]}")
    elif read["snapshot"] not in able["snapshot"] or read["plan"] not in able["plan"]:
        bad.append(
            f"renderer {who[0]} v{who[1]} reads snapshot {sorted(able['snapshot'])} and plan {sorted(able['plan'])},"
            f" not snapshot v{read['snapshot']} with plan v{read['plan']}"
        )
    phases = {phase["id"]: phase for phase in plan["phases"]}
    claims = {claim["id"]: claim for claim in plan["claims"]}
    members = {planning._member_ref(one["ordinal"]) for one in snapshot["members"]}
    if render["support"]["member_refs"] != sorted(members):
        bad.append("the lede's support is not every member of the snapshot")
    if render["support"]["phase_refs"] != [phase["id"] for phase in plan["phases"]]:
        bad.append("the lede's support is not every phase of the plan, in order")
    section_ids = [section["id"] for section in render["sections"]]
    if len(set(section_ids)) != len(section_ids):
        bad.append("section ids are not unique")
    told = [section["phase_ref"] for section in render["sections"]]
    if told and told != [phase["id"] for phase in plan["phases"]]:
        bad.append("sections do not cover the plan's phases exactly once, in the plan's order")
    for section in render["sections"]:
        phase = phases.get(section["phase_ref"])
        if phase is None:
            bad.append(f"{section['id']} names unknown phase {section['phase_ref']}")
            continue
        if section["member_refs"] != phase["member_refs"]:
            bad.append(f"{section['id']} does not carry exactly the members of {phase['id']}")
        bad.extend(
            f"{section['id']} hero {ref} is not a representative of {phase['id']}"
            for ref in section["hero_refs"]
            if ref not in phase["representative_refs"]
        )
        cited: list[str] = []
        for block in section["blocks"]:
            if block["kind"] == "structure":
                bad.extend(
                    f"{section['id']} structure block names {ref}, not one of its members"
                    for ref in block["member_refs"]
                    if ref not in section["member_refs"]
                )
            else:
                for ref in block["claim_refs"]:
                    if ref not in claims:
                        bad.append(f"{section['id']} cites unknown claim {ref}")
                    elif ref not in phase["claim_refs"]:
                        bad.append(f"{section['id']} cites {ref}, which the plan did not attach to {phase['id']}")
                    if ref not in cited:
                        cited.append(ref)
        if section["claim_refs"] != cited:
            bad.append(f"{section['id']} claim_refs are not exactly the claims its blocks cite")
    unsupported = {told["kind"] for told in plan["unsupported"]}
    if "chronology" in unsupported:
        if not any(note["kind"] == "chronology" for note in render["notes"]):
            bad.append("the plan declares chronology unsupported and the render does not say so")
        for section in render["sections"]:
            for block in section["blocks"]:
                for ref in block.get("claim_refs", []):
                    kind = claims.get(ref, {}).get("kind")
                    if kind in planning._DIRECTIONAL:
                        bad.append(f"chronology is unsupported but {section['id']} cites the directional claim {ref}")
        names = _frozen_names(snapshot)
        prose = [render["title"], render["dek"], render["summary"]]
        prose += [section["title"] for section in render["sections"]]
        prose += [block["text"] for section in render["sections"] for block in section["blocks"]]
        for text in prose:
            bad.extend(
                f"chronology is unsupported but the render sequences: {word!r} in {text!r}"
                for word in _sequencing_words(text, names)
            )
    return bad


# --- identities --------------------------------------------------------------


def identity(render: dict) -> tuple[str, str]:
    spelled = canonical({key: value for key, value in render.items() if key != "rendered_at"})
    return spelled, digest(spelled)


def request_identity(
    plan_sha256: str, snapshot_sha256: str, kind: str, version: int, profile: str, locale: str, policy: int
) -> str:
    """The REQUEST identity, known before any work: every policy input
    that can change one byte of the output, the render format included."""
    return digest(
        canonical(
            {
                "format": FORMAT_VERSION,
                "plan_sha256": plan_sha256,
                "snapshot_sha256": snapshot_sha256,
                "renderer": {"kind": kind, "version": version},
                "profile": profile,
                "locale": locale,
                "policy": policy,
            }
        )
    )


# --- persistence ---------------------------------------------------------------


def _verified_inputs(conn, plan_id: int) -> tuple[dict, str, dict, str]:
    """The plan, re-verified (db/planning.py load_plan re-hashes it and
    re-validates it against its re-verified snapshot), and the snapshot
    by value with both identities. The snapshot is the PLAN's: a render
    has no snapshot of its own to disagree with."""
    row = conn.execute("SELECT snapshot_id, document_sha256 FROM story_plan WHERE id = ?", (plan_id,)).fetchone()
    if row is None:
        raise LookupError(f"no story plan {plan_id}")
    plan = planning.load_plan(conn, plan_id)
    snapshot = stories.load_snapshot(conn, int(row[0]))
    return plan, row[1], snapshot, plan["snapshot_sha256"]


def render_plan(conn, plan_id: int, renderer: TemplateStoryRenderer, now: float) -> RenderRef:
    """Render one verified plan under one profile and persist -- or
    return the existing row for the same request, RE-VERIFIED: reuse
    means a row that still hashes and still passes, never "the index
    said something exists". The look-then-insert runs under ONE writer
    lane (BEGIN IMMEDIATE), so two identical requests cannot both see
    nothing and race into the UNIQUE constraint. Pure code, so it is
    synchronous; the transaction is left open for the caller."""
    plan, plan_sha, snapshot, snapshot_sha = _verified_inputs(conn, plan_id)
    who = renderer
    request = request_identity(plan_sha, snapshot_sha, who.kind, who.version, who.profile, who.locale, who.policy)
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    held = conn.execute("SELECT id FROM story_render WHERE request_sha256 = ?", (request,)).fetchone()
    if held:
        return RenderRef(int(held[0]), identity(_load(conn, int(held[0]))[0])[1], True)
    render = renderer.render(snapshot, plan, snapshot_sha, plan_sha)
    wrong = violations(render, plan, snapshot, snapshot_sha, plan_sha)
    if wrong:
        raise AssertionError(f"the renderer produced an invalid render: {wrong}")
    sha = identity(render)[1]
    held = conn.execute("SELECT id FROM story_render WHERE document_sha256 = ?", (sha,)).fetchone()
    if held:
        return RenderRef(int(held[0]), identity(_load(conn, int(held[0]))[0])[1], True)
    render["rendered_at"] = now
    render_id = int(
        conn.execute(
            "INSERT INTO story_render(plan_id, format_version, renderer, renderer_version, profile,"
            " locale, render_policy, request_sha256, document_json, document_sha256, created_at)"
            " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                plan_id,
                FORMAT_VERSION,
                renderer.kind,
                renderer.version,
                renderer.profile,
                renderer.locale,
                renderer.policy,
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
    """The verified render and its plan's snapshot members by ref --
    what a page needs to spell a hero's FROZEN name. The page asks here;
    it never joins a table itself."""
    render, snapshot = _load(conn, render_id)
    members = {planning._member_ref(one["ordinal"]): one for one in snapshot["members"]}
    return render, members


def load_render(conn, render_id: int) -> dict:
    """The stored render, RE-VERIFIED on read: it must hash to its stored
    identity and still pass every violation check against its
    re-verified plan and snapshot. A page lays out only what passes."""
    return _load(conn, render_id)[0]


def _load(conn, render_id: int) -> tuple[dict, dict]:
    row = conn.execute(
        "SELECT plan_id, document_json, document_sha256 FROM story_render WHERE id = ?", (render_id,)
    ).fetchone()
    if row is None:
        raise LookupError(f"no story render {render_id}")
    render = stories.parsed(row[1], f"story render {render_id}")
    if identity(render)[1] != row[2]:
        raise stories.Corrupt(f"story render {render_id} no longer hashes to its identity; refusing to serve it")
    plan, plan_sha, snapshot, snapshot_sha = _verified_inputs(conn, int(row[0]))
    wrong = violations(render, plan, snapshot, snapshot_sha, plan_sha)
    if wrong:
        raise stories.Corrupt(f"story render {render_id} is no longer valid against its plan: {wrong}")
    return render, snapshot
