"""A StoryRender words what the plan claims and nothing else.

The deterministic narrator receives two immutable, verified documents
and decides nothing: every block declares its support -- members for
structure, Claims of ITS OWN phase for sentences -- every hero is a
representative of its phase, unsupported chronology stays unsupported
structurally (no directional Claim without a chronology) with a lexical
scan as defense in depth that exempts frozen evidence names. The v1
grammar is frozen and dispatched by the document's own version; the
renderer declares which input versions it reads as literals. One
policy token covers every output-affecting behaviour. The document is
structure, never HTML; the page lays it out strictly and escapes frozen
evidence. Identities are the planner's: request and document,
insert-only, re-verified on read, corruption of any shape refused.
"""

from __future__ import annotations

import copy
import datetime
import json
import sqlite3

import pytest
from litestar.testing import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import story_renderers
from db import connect, ingest, planning, rendering, runner, stories
from sg_web.app import build_app
from story_renderers import claims as wording
from story_renderers import formatting

NOW = 1_700_000_000.0
HOUR = 3600.0
MIN = 60.0
JULY_18 = 1_784_332_800.0  # 2026-07-18 00:00 as a wall clock


def _spelled(moment: float) -> str:
    return datetime.datetime.fromtimestamp(moment, datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")


LIGHTHOUSE = [
    "a tin lighthouse on a cliff, stormy sky, oil painting",
    "a brass lighthouse on a cliff, stormy sky, oil painting",
    "a copper lighthouse on a cliff, stormy sky, oil painting",
]
HELMET = ["a brass diving helmet in a museum case, studio light", "a copper diving helmet, studio light"]


def _member(ordinal, prompt, *, precision="second", artifacts=(), seed=100, name=None, lora_names=None):
    return {
        "ordinal": ordinal,
        "file_uuid": f"{ordinal:032x}",
        "content_sha256": f"{ordinal:064x}",
        "media_kind": "image",
        "name": name or f"gen_{ordinal}.png",
        "occurrence": {
            "kind": "generation",
            "basis": "embedded",
            "local_at": JULY_18 + ordinal * 3 * MIN,
            "instant_at": None,
            "tz_offset_min": None,
            "precision": precision,
            "certainty": 0.6,
        },
        "generation": {
            "tool": "Qwen Image Edit",
            "detection": "graph",
            "seed": seed,
            "steps": 20,
            "cfg": 7.0,
            "denoise": None,
            "clip_skip": None,
            "sampler": "Euler a",
            "scheduler": None,
            "width": 512,
            "height": 512,
            "prompt": prompt,
            "negative_prompt": "blur",
            "workflow_uuid": None,
            "artifacts": [
                {
                    "ordinal": i,
                    "role": "lora",
                    "uuid": uuid,
                    "kind": "lora",
                    "name": (lora_names or {}).get(uuid, f"lora-{uuid[:4]}"),
                    "model_weight": 0.8,
                    "clip_weight": None,
                }
                for i, uuid in enumerate(artifacts)
            ],
            "params": {},
        },
        "capture": None,
        "people": None,
        "place": None,
        "lineage": None,
        "annotations": None,
    }


def _snapshot(members, *, local=True, instant=None) -> tuple[dict, str]:
    document = {
        "v": 1,
        "frozen_at": NOW,
        "subject": {
            "kind": "event",
            "event_kind": "generation_session",
            "grouper": "generation_session",
            "grouper_version": "4",
            "settings_hash": "abc",
            "claim": "generation",
            "member_hash": "deadbeef",
            "context_generation": 7,
            "context_policy_version": 4,
            "time": {"local": [JULY_18, JULY_18 + HOUR] if local else None, "instant": instant},
            "place": None,
            "confidence": None,
            "observed_event_id": 1,
        },
        "members": members,
    }
    return document, stories._identity(document)[1]


def _planned(members, **kwargs):
    document, sha = _snapshot(members, **kwargs)
    plan = planning.GenerationHistoryPlanner(planning.LexicalPromptSimilarity()).plan(document, sha)
    return document, sha, plan, planning.identity(plan)[1]


LORA_A, LORA_B = "a" * 32, "b" * 32


def _world():
    return _planned(
        [
            _member(0, LIGHTHOUSE[0], artifacts=(LORA_A,), seed=1),
            _member(1, LIGHTHOUSE[1], artifacts=(LORA_A,), seed=2),
            _member(2, LIGHTHOUSE[2], artifacts=(LORA_A,), seed=3),
            _member(3, HELMET[0], artifacts=(LORA_B,), seed=4),
            _member(4, HELMET[1], artifacts=(LORA_B,), seed=5),
        ]
    )


def _rendered(profile="memory", world=None):
    snapshot, snapshot_sha, plan, plan_sha = world or _world()
    render = rendering.TemplateStoryRenderer(profile).render(snapshot, plan, snapshot_sha, plan_sha)
    return snapshot, snapshot_sha, plan, plan_sha, render


def test_the_golden_render_for_a_sequenced_session():
    """The exact document the deterministic narrator produces for a
    frozen world -- pinned whole. Every block below declares what
    supports it; the renderer invented none of it."""
    snapshot, snapshot_sha, plan, plan_sha, render = _rendered()
    assert rendering.violations(render, plan, snapshot, snapshot_sha, plan_sha) == []
    assert render == {
        "v": 1,
        "snapshot_sha256": snapshot_sha,
        "plan_sha256": plan_sha,
        "renderer": {
            "kind": "template",
            "version": 2,
            "profile": "memory",
            "locale": "en",
            "policy": 2,
            "reads": {"snapshot": 1, "plan": 3},
        },
        "title": "5 Qwen Image Edit images from July 18, 2026",
        "dek": "2 phases across 5 generated images",
        "summary": "These 5 generated images were generated on July 18, 2026 and fall into 2 phases.",
        "support": {
            "member_refs": ["member-001", "member-002", "member-003", "member-004", "member-005"],
            "phase_refs": ["phase-001", "phase-002"],
        },
        "sections": [
            {
                "id": "section-001",
                "phase_ref": "phase-001",
                "title": "Phase 1",
                "blocks": [
                    {
                        "kind": "structure",
                        "text": "3 images.",
                        "member_refs": ["member-001", "member-002", "member-003"],
                    },
                    {
                        "kind": "claim",
                        "text": "The 3 images in this phase share closely related prompt wording.",
                        "claim_refs": ["claim-001"],
                    },
                ],
                "hero_refs": ["member-001"],
                "member_refs": ["member-001", "member-002", "member-003"],
                "claim_refs": ["claim-001"],
            },
            {
                "id": "section-002",
                "phase_ref": "phase-002",
                "title": "Phase 2 · new artifacts",
                "blocks": [
                    {"kind": "structure", "text": "2 images.", "member_refs": ["member-004", "member-005"]},
                    {
                        "kind": "claim",
                        "text": "The prompt wording changes here compared with the previous phase.",
                        "claim_refs": ["claim-003"],
                    },
                    {
                        "kind": "claim",
                        "text": "The 2 images in this phase share closely related prompt wording.",
                        "claim_refs": ["claim-004"],
                    },
                    {
                        "kind": "claim",
                        "text": "Compared with the previous phase, lora-bbbb appears in this group;"
                        " lora-aaaa is not used.",
                        "claim_refs": ["claim-006"],
                    },
                ],
                "hero_refs": ["member-004"],
                "member_refs": ["member-004", "member-005"],
                "claim_refs": ["claim-003", "claim-004", "claim-006"],
            },
        ],
        "notes": [],
    }
    technical = _rendered("technical")[-1]
    assert any(b["text"] == "2 distinct seeds were used." for b in technical["sections"][1]["blocks"])
    assert any("Minimum pairwise prompt similarity" in b["text"] for b in technical["sections"][0]["blocks"])
    assert set(technical["sections"][1]["claim_refs"]) > set(render["sections"][1]["claim_refs"]), (
        "a profile surfaces more or fewer claims; it never changes which claims exist"
    )
    compact = _rendered("compact")[-1]
    assert compact["sections"] == []
    assert compact["title"] == render["title"]


def test_unsupported_chronology_is_said_structurally_and_lexically():
    """Day-precision evidence: the plan declares chronology unsupported;
    the render carries the note, cites no directional Claim, and says
    no sequencing word. Each rule has teeth -- and the lexical rule
    exempts a frozen artifact named "AfterDetail", which is evidence,
    not narration."""
    names = {LORA_A: "AfterDetail.safetensors", LORA_B: "lora-bbbb"}
    world = _planned(
        [
            _member(0, LIGHTHOUSE[0], precision="day", artifacts=(LORA_A,), lora_names=names),
            _member(1, LIGHTHOUSE[1], precision="day", artifacts=(LORA_A,), lora_names=names),
            _member(2, HELMET[0], precision="day", artifacts=(LORA_B,), lora_names=names),
            _member(3, HELMET[1], precision="day", artifacts=(LORA_B,), lora_names=names),
        ]
    )
    snapshot, snapshot_sha, plan, plan_sha, render = _rendered(world=world)
    assert [one["kind"] for one in plan["unsupported"]] == ["chronology"]
    assert rendering.violations(render, plan, snapshot, snapshot_sha, plan_sha) == []
    assert render["notes"] == [
        {
            "kind": "chronology",
            "text": "Available evidence does not establish the order of these images within July 18, 2026.",
        }
    ]
    assert all(section["title"].startswith("Prompt family") for section in render["sections"])
    assert render["sections"][1]["title"] == "Prompt family 2 · different artifacts"
    told = [b["text"] for b in render["sections"][1]["blocks"] if b["kind"] == "claim"]
    assert (
        "Compared with Prompt family 1, lora-bbbb is used only here; AfterDetail.safetensors is used only there."
        in told
    )
    assert "prompt families" in render["dek"]

    sequenced = copy.deepcopy(render)
    sequenced["sections"][1]["blocks"][0]["text"] = "Then the helmet appeared."
    told = rendering.violations(sequenced, plan, snapshot, snapshot_sha, plan_sha)
    assert any("sequences: 'Then'" in why for why in told)
    assert not any("'After'" in why for why in told), "a frozen name is exempt from the lexical scan"

    silent = copy.deepcopy(render)
    silent["notes"] = []
    assert "the plan declares chronology unsupported and the render does not say so" in rendering.violations(
        silent, plan, snapshot, snapshot_sha, plan_sha
    )

    # the structural rule: a directional Claim smuggled into an
    # unsequenced plan is refused by the PLAN's validator, and a render
    # citing it is refused by the render's
    directed = copy.deepcopy(plan)
    directed["claims"].append(
        {
            "id": "claim-099",
            "kind": "prompt_shift",
            "confidence": 1.0,
            "evidence_refs": ["member-001:generation.prompt", "member-003:generation.prompt"],
            "facts": {"cosine": 0.1, "threshold": 0.5},
        }
    )
    directed["phases"][1]["claim_refs"].append("claim-099")
    assert any(
        "directional claim prompt_shift" in why for why in planning.validate_plan(directed, snapshot, snapshot_sha)
    )
    cites = copy.deepcopy(render)
    cites["sections"][1]["blocks"].append({"kind": "claim", "text": "Different.", "claim_refs": ["claim-099"]})
    cites["sections"][1]["claim_refs"].append("claim-099")
    told = rendering.violations(cites, directed, snapshot, snapshot_sha, planning.identity(directed)[1])
    assert any("cites the directional claim claim-099" in why for why in told)


def test_every_block_declares_support_from_its_own_phase():
    """The Renderer Seam an LLM must satisfy: a section names its phase
    and carries exactly its members; heroes are that phase's
    representatives; a claim block cites only Claims the plan attached
    to THAT phase; a structure block names only its own members; a
    block without support is prose and is refused; section claim_refs
    are exactly what its blocks cite; the lede's support is the whole
    subject."""
    snapshot, snapshot_sha, plan, plan_sha, render = _rendered()

    def told(mutate):
        bent = copy.deepcopy(render)
        mutate(bent)
        reasons = rendering.violations(bent, plan, snapshot, snapshot_sha, plan_sha)
        assert reasons, "violations() accepted a render that says more than its plan"
        return reasons

    assert "section-001 cites unknown claim claim-099" in told(
        lambda r: (
            r["sections"][0]["blocks"].append({"kind": "claim", "text": "x", "claim_refs": ["claim-099"]}),
            r["sections"][0]["claim_refs"].append("claim-099"),
        )
    )
    assert "section-001 cites claim-003, which the plan did not attach to phase-001" in told(
        lambda r: (
            r["sections"][0]["blocks"].append({"kind": "claim", "text": "x", "claim_refs": ["claim-003"]}),
            r["sections"][0]["claim_refs"].append("claim-003"),
        )
    )
    assert "section-001 does not carry exactly the members of phase-001" in told(
        lambda r: r["sections"][0]["member_refs"].append("member-004")
    )
    assert "section-001 hero member-002 is not a representative of phase-001" in told(
        lambda r: r["sections"][0].__setitem__("hero_refs", ["member-002"])
    )
    assert "section-001 structure block names member-004, not one of its members" in told(
        lambda r: r["sections"][0]["blocks"][0]["member_refs"].append("member-004")
    )
    assert "section-001 claim_refs are not exactly the claims its blocks cite" in told(
        lambda r: r["sections"][0]["claim_refs"].append("claim-003")
    )
    assert any(
        "a block without support is prose" in why
        for why in told(
            lambda r: r["sections"][0]["blocks"].append(
                {"kind": "claim", "text": "The user hated every image.", "claim_refs": []}
            )
        )
    )
    assert any(
        "is not a block of a known kind" in why
        for why in told(
            lambda r: r["sections"][0]["blocks"].append({"kind": "prose", "text": "They changed direction."})
        )
    )
    assert "the lede's support is not every member of the snapshot" in told(lambda r: r["support"]["member_refs"].pop())
    assert "sections do not cover the plan's phases exactly once, in the plan's order" in told(
        lambda r: r["sections"].reverse()
    )
    assert "section-001 names unknown phase phase-009" in told(
        lambda r: r["sections"][0].__setitem__("phase_ref", "phase-009")
    )
    assert "the render names a different plan" in told(lambda r: r.__setitem__("plan_sha256", "f" * 64))
    assert any("it did not" in why for why in told(lambda r: r["renderer"]["reads"].__setitem__("plan", 1)))


def test_the_v1_grammar_is_frozen_exact_and_exception_proof(monkeypatch):
    """A v1 render parses as v1 after a render v2 and a plan v3 exist:
    the grammar reads frozen constants and the renderer's input
    compatibility is a literal. Any bytes -- unhashable kinds included
    -- yield controlled reasons, never an exception."""
    snapshot, snapshot_sha, plan, plan_sha, render = _rendered()
    assert rendering.validate_story_render_v1(render) == []
    profiles = rendering.PROFILES
    monkeypatch.setattr(rendering, "FORMAT_VERSION", 2)
    monkeypatch.setattr(rendering, "PROFILES", ())
    monkeypatch.setattr(planning, "FORMAT_VERSION", 4)
    assert rendering.validate_story_render_v1(render) == [], "v1 is judged by v1's frozen vocabulary"
    assert rendering.validate_story_render(render) == []
    assert rendering.violations(render, plan, snapshot, snapshot_sha, plan_sha) == []
    monkeypatch.setattr(rendering, "PROFILES", profiles)
    assert rendering.TemplateStoryRenderer().reads == {"snapshot": {1}, "plan": {2, 3}}, "plan v1 is history, not input"
    assert rendering.COMPATIBILITY[("template", 1)] == {"snapshot": {1}, "plan": {1, 2}}, "v1's inputs stay frozen"
    # a stored render must agree with ITS version's frozen map: a v1 render
    # of a v3 plan is impossible, and a v2 render cannot claim a v1 plan
    bent = copy.deepcopy(render)
    bent["renderer"]["version"] = 1
    assert any(
        "renderer template v1 reads" in why
        for why in rendering.violations(bent, plan, snapshot, snapshot_sha, plan_sha)
    )
    bent["renderer"]["version"] = 7
    assert any(
        "no frozen compatibility" in why for why in rendering.violations(bent, plan, snapshot, snapshot_sha, plan_sha)
    )
    again = rendering.TemplateStoryRenderer("memory").render(snapshot, plan, snapshot_sha, plan_sha)
    assert again["renderer"]["reads"] == {"snapshot": 1, "plan": 3}
    with pytest.raises(ValueError, match="reads StorySnapshot \\[1\\] and StoryPlan \\[2, 3\\]"):
        rendering.TemplateStoryRenderer("memory").render(snapshot, {**plan, "v": 4}, snapshot_sha, plan_sha)
    with pytest.raises(ValueError, match="reads StorySnapshot"):
        rendering.TemplateStoryRenderer("memory").render(snapshot, {**plan, "v": 1}, snapshot_sha, plan_sha)
    assert rendering.validate_story_render({**render, "v": 7})
    assert rendering.validate_story_render({**render, "v": True})

    for mutate in (
        lambda r: r.__setitem__("html", "<b>no</b>"),
        lambda r: r["sections"][0].__setitem__("mood", "wistful"),
        lambda r: r["renderer"].__setitem__("kind", "llm"),
        lambda r: r["renderer"].__setitem__("kind", []),
        lambda r: r["renderer"].__setitem__("profile", "poetic"),
        lambda r: r["renderer"].__setitem__("profile", {}),
        lambda r: r["renderer"].__setitem__("policy", 0),
        lambda r: r["renderer"]["reads"].__setitem__("plan", "2"),
        lambda r: r["sections"][0].__setitem__("blocks", []),
        lambda r: r["sections"][0]["blocks"].__setitem__(0, {"kind": [], "text": "x"}),
        lambda r: r["sections"][0]["blocks"].__setitem__(0, {"kind": "structure", "text": "x", "member_refs": []}),
        lambda r: r["sections"][0].__setitem__("member_refs", []),
        lambda r: r["sections"][0].__setitem__("phase_ref", "section-001"),
        lambda r: r["notes"].append({"kind": "weather", "text": "rain"}),
        lambda r: r["notes"].append({"kind": [], "text": "rain"}),
        lambda r: r["support"].__setitem__("phase_refs", []),
        lambda r: r.__setitem__("title", ""),
    ):
        bent = copy.deepcopy(render)
        mutate(bent)
        assert rendering.validate_story_render_v1(bent), "the grammar accepted a malformed render"
    for garbage in (None, [], "story", {"v": 1}, {"v": 1, "sections": "many"}):
        assert rendering.validate_story_render_v1(garbage)
        assert rendering.validate_story_render(garbage)

    ctx = wording.Context(snapshot=snapshot, plan=plan, profile="memory", sequenced=True)
    with pytest.raises(ValueError, match="no wording is registered"):
        wording.word(
            {"id": "claim-x", "kind": "vibe_shift", "confidence": 1.0, "evidence_refs": [], "facts": {}},
            plan["phases"][0],
            ctx,
        )
    with pytest.raises(ValueError, match="no render profile"):
        rendering.TemplateStoryRenderer("poetic")
    with pytest.raises(ValueError, match="no locale"):
        rendering.TemplateStoryRenderer("memory", "tlh")


def test_a_day_is_spelled_in_the_domain_the_evidence_claims_it_in():
    """A wall-clock start is the human's calendar day; an instant with no
    wall clock is a UTC day and says so; neither is fused into the
    other."""
    assert formatting.day_label(JULY_18) == "July 18, 2026"
    assert formatting.day_label(JULY_18, utc=True) == "July 18, 2026 UTC"
    assert formatting.day_label(None) is None
    assert formatting.count(1, "image") == "1 image"
    assert formatting.count(55, "image") == "55 images"
    assert formatting.percent(0.92314) == "92%"
    assert formatting.join_names(["foo"]) == "foo"
    assert formatting.join_names(["foo", "bar"]) == "foo and bar"
    assert formatting.join_names(["foo", "bar", "baz"]) == "foo, bar and baz"

    members = [_member(i, text, precision="day") for i, text in enumerate(LIGHTHOUSE)]
    world = _planned(members, local=False, instant=[JULY_18, JULY_18 + HOUR])
    render = _rendered(world=world)[-1]
    assert render["title"] == "3 Qwen Image Edit images from July 18, 2026 UTC"
    assert render["notes"][0]["text"].endswith("within July 18, 2026 UTC.")
    render = _rendered(world=_planned(members, local=False))[-1]
    assert render["title"] == "3 Qwen Image Edit images"
    assert render["notes"][0]["text"].endswith("order of these images.")


def test_the_narrator_names_a_tool_and_a_day_only_when_the_evidence_agrees():
    """A session is grouped by time and may mix tools: the title names a
    tool only when every member used it. A session that crosses
    midnight spans two days: "generated over July 18 to 19", never "on"
    either -- and an instant-only interval keeps saying UTC."""
    mixed = [
        _member(0, LIGHTHOUSE[0]),
        _member(1, LIGHTHOUSE[1]),
        dict(_member(2, HELMET[0]), generation={**_member(2, HELMET[0])["generation"], "tool": "Flux"}),
    ]
    render = _rendered(world=_planned(mixed))[-1]
    assert render["title"] == "3 generated images from July 18, 2026"
    assert _planned(mixed)[2]["subject"]["label_hint"].startswith("generation session")
    assert _rendered()[-1]["title"].startswith("5 Qwen Image Edit images")

    members = [_member(i, text) for i, text in enumerate(LIGHTHOUSE)]
    document, sha = _snapshot(members)
    document["subject"]["time"] = {
        "local": [JULY_18 + 23 * HOUR + 55 * MIN, JULY_18 + 24 * HOUR + 5 * MIN],
        "instant": None,
    }
    sha = stories._identity(document)[1]
    plan = planning.GenerationHistoryPlanner(planning.LexicalPromptSimilarity()).plan(document, sha)
    render = rendering.TemplateStoryRenderer("memory").render(document, plan, sha, planning.identity(plan)[1])
    assert render["title"] == "3 Qwen Image Edit images from July 18" + formatting.EN_DASH + "19, 2026"
    assert "were generated over July 18" + formatting.EN_DASH + "19, 2026 and" in render["summary"]
    document["subject"]["time"] = {
        "local": None,
        "instant": [JULY_18 + 23 * HOUR + 55 * MIN, JULY_18 + 24 * HOUR + 5 * MIN],
    }
    sha = stories._identity(document)[1]
    plan = planning.GenerationHistoryPlanner(planning.LexicalPromptSimilarity()).plan(document, sha)
    render = rendering.TemplateStoryRenderer("memory").render(document, plan, sha, planning.identity(plan)[1])
    assert render["title"] == "3 Qwen Image Edit images from July 18" + formatting.EN_DASH + "19, 2026 UTC"
    assert (
        formatting.day_range(JULY_18, JULY_18 + 14 * 24 * HOUR) == "July 18 " + formatting.EN_DASH + " August 1, 2026"
    )
    assert (
        formatting.day_range(JULY_18 + 166 * 24 * HOUR, JULY_18 + 167 * 24 * HOUR)
        == "December 31, 2026 " + formatting.EN_DASH + " January 1, 2027"
    )
    assert formatting.day_range(JULY_18, JULY_18 + HOUR, utc=True) == "July 18, 2026 UTC"


def test_a_thousand_members_are_a_plan_and_a_render_not_a_grammar_error():
    members = [_member(i, LIGHTHOUSE[i % 3]) for i in range(1000)]
    document, sha = _snapshot(members)
    plan = planning.GenerationHistoryPlanner(planning.LexicalPromptSimilarity()).plan(document, sha)
    assert "member-1000" in plan["phases"][-1]["member_refs"]
    assert planning.validate_plan(plan, document, sha) == []
    render = rendering.TemplateStoryRenderer("memory").render(document, plan, sha, planning.identity(plan)[1])
    assert rendering.violations(render, plan, document, sha, planning.identity(plan)[1]) == []


def test_identities_same_request_one_document_policy_coexists(monkeypatch):
    _snapshot, snapshot_sha, _plan, plan_sha, one = _rendered()
    two = _rendered()[-1]
    assert rendering.identity(one) == rendering.identity(two)
    other = _rendered("technical")[-1]
    assert rendering.identity(other)[1] != rendering.identity(one)[1]
    request = rendering.request_identity(plan_sha, snapshot_sha, "template", 2, "memory", "en", 1)
    assert request != rendering.request_identity(plan_sha, snapshot_sha, "template", 2, "memory", "en", 2), (
        "a policy bump is a different request"
    )
    assert request != rendering.request_identity(plan_sha, snapshot_sha, "template", 2, "technical", "en", 1)
    assert request != rendering.request_identity(plan_sha, snapshot_sha, "template", 1, "memory", "en", 1), (
        "a renderer version is a different request"
    )
    monkeypatch.setattr(rendering, "FORMAT_VERSION", 2)
    assert request != rendering.request_identity(plan_sha, snapshot_sha, "template", 2, "memory", "en", 1)
    # ONE token covers wording AND formatting: the renderer's policy is
    # the package's, and a formatting change has nowhere else to go
    monkeypatch.setattr(story_renderers, "POLICY_VERSION", 3)
    assert rendering.TemplateStoryRenderer("memory").policy == 3
    assert _rendered()[-1]["renderer"]["policy"] == 3
    source = (__import__("pathlib").Path(formatting.__file__)).read_text(encoding="utf-8")
    assert "POLICY_VERSION" not in source, "formatting carries no version of its own; the package token is it"
    assert (
        "POLICY_VERSION"
        not in (__import__("pathlib").Path(wording.__file__)).read_text(encoding="utf-8").split('"""', 2)[2]
    ), "claims carries no version of its own; the package token is it"


def test_the_renderer_owns_no_connection_no_model_and_jinja_lays_out_only():
    import inspect
    import pathlib

    here = pathlib.Path(__file__).resolve().parent.parent
    source = (here / "db" / "rendering.py").read_text(encoding="utf-8")
    head = source.split("# --- persistence", 1)[0]
    for banned in ("execute(", "FROM ", "JOIN ", "sqlite3", "(conn", "conn,", "conn)"):
        assert banned not in head, f"the narrator reached for the database: {banned!r}"
    for banned in ("import openai", "anthropic", "import requests", "import httpx", "torch", "jinja"):
        assert banned not in source
    assert "conn" not in inspect.signature(rendering.TemplateStoryRenderer.render).parameters
    adapters = (here / "story_renderers" / "claims.py").read_text(encoding="utf-8")
    for banned in ("execute(", "sqlite3", "jinja", "import db"):
        assert banned not in adapters
    page = (here / "sg_web" / "templates" / "story.html").read_text(encoding="utf-8")
    for banned in ("|safe", "{% set", "cosine", "similarity", "claim.kind", "execute"):
        assert banned not in page, f"the template reasons: {banned!r}"


# --- persistence and the page, against a real frozen world ------------------


def _library(tmp) -> tuple:
    root = tmp / "lib"
    root.mkdir()
    # the hero is the phase's medoid (ties to the earliest), so the
    # escaping probe rides the FIRST name; NTFS forbids < >, & and ' probe
    names = ["gen_0 & 'friends'.png", "gen_1.png", "gen_2.png"]
    for i, (text, name) in enumerate(zip(LIGHTHOUSE, names, strict=True)):
        info = PngInfo()
        info.add_text(
            "parameters",
            f"{text}\nNegative prompt: blur\n"
            f"Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: {100 + i}, Size: 512x512, Model: alpha",
        )
        Image.new("RGB", (12, 12), (40 + i * 30, 90, 140)).save(root / name, pnginfo=info)
    return tmp / "run", root


def _drain(client) -> None:
    conn = connect.connect(client.app.state.db_path)
    try:
        while runner.run_next(conn, "test-worker", NOW + 24 * HOUR) is not None:
            conn.commit()
        conn.commit()
    finally:
        connect.close(conn)


@pytest.fixture
def planned(tmp_path):
    burrow, root = _library(tmp_path)
    with TestClient(app=build_app(str(burrow), worker=False)) as client:
        client.post("/roots", json={"path": str(root)})
        client.post("/roots/1/scan")
        conn = connect.connect(client.app.state.db_path)
        try:
            names = dict(conn.execute("SELECT name, id FROM file").fetchall())
            for name, file_id in names.items():
                ingest.one(conn, file_id, root / name, NOW)
            conn.executemany(
                "INSERT OR REPLACE INTO file_param(file_id, source, key, value_text)"
                " VALUES(?, 'generation', 'date', ?)",
                [(file_id, _spelled(NOW + i * 4 * MIN)) for i, file_id in enumerate(names.values())],
            )
            conn.commit()
        finally:
            connect.close(conn)
        client.post("/jobs/context")
        client.post("/jobs/events")
        _drain(client)
        conn = connect.connect(client.app.state.db_path)
        try:
            event_id = conn.execute("SELECT id FROM derived_event WHERE kind = 'generation_session'").fetchone()[0]
            snap = stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
            planner = planning.GenerationHistoryPlanner(planning.LexicalPromptSimilarity())
            made = planning.plan_snapshot(conn, snap.id, planner, NOW + 31 * HOUR)
            conn.commit()
        finally:
            connect.close(conn)
        yield client, snap, made


def test_a_persisted_render_is_immutable_reused_and_reverified(planned):
    client, snap, plan_ref = planned
    conn = connect.connect(client.app.state.db_path)
    try:
        first = rendering.render_plan(conn, plan_ref.id, rendering.TemplateStoryRenderer("memory"), NOW + 32 * HOUR)
        conn.commit()
        assert first.reused is False
        again = rendering.render_plan(conn, plan_ref.id, rendering.TemplateStoryRenderer("memory"), NOW + 33 * HOUR)
        assert (again.id, again.sha256, again.reused) == (first.id, first.sha256, True)
        other = rendering.render_plan(conn, plan_ref.id, rendering.TemplateStoryRenderer("technical"), NOW + 33 * HOUR)
        conn.commit()
        assert other.id != first.id
        told = rendering.load_render(conn, first.id)
        assert told["snapshot_sha256"] == snap.sha256
        assert told["plan_sha256"] == plan_ref.sha256
        columns = {row[1] for row in conn.execute("PRAGMA table_info(story_render)")}
        assert "snapshot_id" not in columns, "the plan's snapshot is the only snapshot a render has"
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE story_render SET document_json = '{}' WHERE id = ?", (first.id,))
        conn.rollback()
        with pytest.raises(LookupError, match="no story plan"):
            rendering.render_plan(conn, 99_999, rendering.TemplateStoryRenderer("memory"), NOW + 34 * HOUR)

        conn.execute("DROP TRIGGER story_render_is_immutable")
        good = conn.execute("SELECT document_json FROM story_render WHERE id = ?", (first.id,)).fetchone()[0]
        bent = json.loads(good)
        bent["summary"] = "A sentence the plan never claimed."
        conn.execute("UPDATE story_render SET document_json = ? WHERE id = ?", (json.dumps(bent), first.id))
        conn.commit()
        with pytest.raises(ValueError, match="no longer hashes"):
            rendering.load_render(conn, first.id)
        assert client.get(f"/stories/renders/{first.id}").status_code == 409
        # reuse is VERIFIED reuse: the same request against a corrupted row
        # is a refusal, never "200 reused"
        told = client.post("/stories/renders", json={"plan_id": plan_ref.id, "profile": "memory"})
        assert told.status_code == 409, told.text
        with pytest.raises(ValueError, match="no longer hashes"):
            rendering.render_plan(conn, plan_ref.id, rendering.TemplateStoryRenderer("memory"), NOW + 35 * HOUR)
        conn.rollback()
        # corruption of ANY json shape is a controlled refusal, never a 500
        for corrupt in ("[]", "null", '"story"', "7"):
            conn.execute("UPDATE story_render SET document_json = ? WHERE id = ?", (corrupt, first.id))
            conn.commit()
            with pytest.raises(ValueError, match="not a document"):
                rendering.load_render(conn, first.id)
            assert client.get(f"/stories/renders/{first.id}").status_code == 409, corrupt
    finally:
        connect.close(conn)


def test_two_simultaneous_identical_requests_make_one_row(planned):
    """The look-then-insert runs under one writer lane: two identical
    POSTs at once yield one row and two successes, never a UNIQUE
    violation surfacing as a 500."""
    import threading

    client, _snap, plan_ref = planned
    answers: list[int] = []

    def ask():
        answers.append(client.post("/stories/renders", json={"plan_id": plan_ref.id, "profile": "memory"}).status_code)

    threads = [threading.Thread(target=ask) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(answers) in ([200, 200, 200, 201], [200, 200, 201, 201]) or all(a in (200, 201) for a in answers)
    assert all(a in (200, 201) for a in answers), answers
    conn = connect.connect(client.app.state.db_path)
    try:
        assert conn.execute("SELECT count(*) FROM story_render").fetchone() == (1,)
    finally:
        connect.close(conn)


def test_the_page_lays_out_the_verified_render_and_escapes_evidence(planned):
    """The HTML Adapter: title, sections, the note, hero links -- and a
    file name that is markup stays text. The page carries every block
    the JSON carries: deleting story.html would lose nothing the JSON
    does not already say."""
    from db import resultset

    client, _snap, plan_ref = planned
    made = client.post("/stories/renders", json={"plan_id": plan_ref.id, "profile": "memory"})
    assert made.status_code == 201, made.text
    render_id = made.json()["id"]
    assert client.post("/stories/renders", json={"plan_id": plan_ref.id, "profile": "memory"}).status_code == 200
    assert client.post("/stories/renders", json={"plan_id": plan_ref.id, "profile": "poetic"}).status_code == 400
    assert client.post("/stories/renders", json={"plan_id": 99_999}).status_code == 404
    assert client.post("/stories/renders", json={}).status_code == 400

    conn = connect.connect(client.app.state.db_path)
    try:
        before = resultset.currency(conn)
    finally:
        connect.close(conn)
    told = client.get(f"/stories/renders/{render_id}", headers={"accept": "application/json"})
    assert told.status_code == 200
    story = told.json()
    page = client.get(f"/stories/renders/{render_id}", headers={"accept": "text/html"})
    assert page.status_code == 200
    html = page.text
    assert f"<title>{story['title']}</title>" in html
    for section in story["sections"]:
        assert f'data-story-section="{section["id"]}"' in html
        assert f'data-story-phase="{section["phase_ref"]}"' in html
        for block in section["blocks"]:
            assert block["text"] in html, "the page drops nothing the JSON says"
    assert "gen_0 & 'friends'.png" not in html, "frozen evidence is text, never markup"
    assert "gen_0 &amp; &#39;friends&#39;.png" in html
    assert html.count('href="/m/') >= 1, "a hero links through address resolution"
    assert client.get("/stories/renders/424242").status_code == 404
    conn = connect.connect(client.app.state.db_path)
    try:
        assert resultset.currency(conn) == before, "a GET of a story wrote something"
    finally:
        connect.close(conn)
