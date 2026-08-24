"""A StoryPlan is structure from frozen evidence -- never prose, never a
database connection.

The planner receives only the snapshot document and a versioned
similarity engine; it cannot see the live library, so nothing that
happens to the library after the snapshot can change the plan. Every
structure is a Claim -- a boundary is a prompt_shift, a family is a
prompt_family -- and every reference resolves inside the snapshot; a
plan is an exact partition of its snapshot, proven before persistence
and again on every read. Settings and similarity output fail closed.
Prompt identity alone never splits a phase; day-precision evidence
never becomes sub-day chronology. Two identities: the REQUEST's, known
before any model work, so an identical request reuses the plan or the
live job; and the DOCUMENT's, so the same evidence under the same
policy is one plan and a new policy coexists with the old one.
Production planning is durable work, off the request thread.

The render's persistence and page (tests/test_a_story_render_words_only_what_the_plan_claims.py
holds the narrator's grammar) and the snapshot that freezes evidence:

A StorySnapshot freezes evidence, not prose.

It is an immutable, self-contained record of exactly what the
application knew about ONE current event at ONE instant: file identity
AND the bytes actually observed, the occurrence that placed each member
in this event, the source facts by value -- never a foreign key into
today's rebuildable hypotheses. Identity is the canonical document's
hash, so identical evidence is one row; freezing proves currentness
the way regrouping does; and after the library, the policy and the
event have all moved on, the old snapshot loads byte-identical while a
fresh one is visibly different.
"""

from __future__ import annotations

import copy
import datetime
import json
import math
import os
import pathlib
import re
import sqlite3
import typing

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import connect, context, events, ingest, planning, rendering, runner, scan, stories
from tests.staging import Stage, staged

NOW = 1_700_000_000.0
HOUR = 3600.0
MIN = 60.0


def _spelled(moment: float) -> str:
    return datetime.datetime.fromtimestamp(moment, datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")


# --- a synthetic snapshot: the planner's whole world ------------------------

LIGHTHOUSE = [
    "a tin lighthouse on a cliff, stormy sky, oil painting",
    "a brass lighthouse on a cliff, stormy sky, oil painting",
    "a copper lighthouse on a cliff, stormy sky, oil painting",
    "a tin lighthouse on a cliff, clear sky, oil painting",
]
HELMET = [
    "a brass diving helmet in a museum case, studio light",
    "a copper diving helmet in a museum case, studio light",
]


def _member(ordinal, prompt, *, seed=100, precision="second", artifacts=(), sampler="Euler a", steps=20):
    return {
        "ordinal": ordinal,
        "file_uuid": f"{ordinal:032x}",
        "content_sha256": f"{ordinal:064x}",
        "media_kind": "image",
        "name": f"gen_{ordinal}.png",
        "occurrence": {
            "kind": "generation",
            "basis": "embedded",
            "local_at": NOW + ordinal * 3 * MIN,
            "instant_at": None,
            "tz_offset_min": None,
            "precision": precision,
            "certainty": 0.6,
        },
        "generation": {
            "tool": "ComfyUI",
            "detection": "graph",
            "seed": seed,
            "steps": steps,
            "cfg": 7.0,
            "denoise": None,
            "clip_skip": None,
            "sampler": sampler,
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
                    "name": uuid[:6],
                    "model_weight": 0.8,
                    "clip_weight": None,
                }
                for i, uuid in enumerate(artifacts)
            ],
            "params": {"original_prompt": "a __material__ lighthouse"},
        },
        "capture": None,
        "people": None,
        "place": None,
        "lineage": None,
        "annotations": None,
    }


def _snapshot(members) -> tuple[dict, str]:
    document = {
        "v": 1,
        "frozen_at": NOW + 30 * HOUR,
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
            "time": {"local": [NOW, NOW + 30 * MIN], "instant": None},
            "place": None,
            "confidence": None,
            "observed_event_id": 1,
        },
        "members": members,
    }
    return document, stories._identity(document)[1]


def _planner(**settings):
    return planning.GenerationHistoryPlanner(planning.LexicalPromptSimilarity(), settings or None)


def _claims_of(plan, phase_index):
    by_id = {claim["id"]: claim for claim in plan["claims"]}
    return {by_id[ref]["kind"]: by_id[ref] for ref in plan["phases"][phase_index]["claim_refs"]}


def test_wildcard_expansions_are_one_phase_and_a_boundary_is_a_claim():
    """The 55-file lesson: prompts that differ only by a wildcard
    expansion are one creative thread. Identity is never consulted --
    only similarity -- so the four lighthouse variants stay one phase,
    the diving helmet opens the next, and the boundary itself is a
    prompt_shift claim with evidence on BOTH sides."""
    members = [_member(i, text, seed=100 + i) for i, text in enumerate(LIGHTHOUSE + HELMET)]
    document, sha = _snapshot(members)
    plan = _planner().plan(document, sha)
    assert plan["subject"]["sequenced"] is True
    assert [phase["member_refs"] for phase in plan["phases"]] == [
        ["member-001", "member-002", "member-003", "member-004"],
        ["member-005", "member-006"],
    ]
    assert set(_claims_of(plan, 0)) == {"prompt_similarity", "seed_variation"}
    shift = _claims_of(plan, 1)["prompt_shift"]
    assert shift["evidence_refs"] == ["member-001:generation.prompt", "member-005:generation.prompt"]
    assert shift["facts"]["cosine"] < shift["facts"]["threshold"] == 0.5
    assert plan["phases"][0]["representative_refs"] == ["member-001"], "the medoid of the family, ties to the earliest"
    assert "parameter_change" not in _claims_of(plan, 1), "nothing but the prompt changed"
    assert plan["unsupported"] == []
    assert planning.validate_plan(plan, document, sha) == []


def test_day_precision_evidence_yields_families_backed_by_claims():
    """Files that only claim a DAY have no order among them. The planner
    finds prompt families, lists members in event order, backs every
    family with a prompt_family claim, and says plainly that chronology
    is unsupported -- it never calls a family a phase in time."""
    members = [_member(i, text, precision="day") for i, text in enumerate(LIGHTHOUSE + HELMET)]
    document, sha = _snapshot(members)
    plan = _planner().plan(document, sha)
    assert plan["subject"]["sequenced"] is False
    assert [one["kind"] for one in plan["unsupported"]] == ["chronology"]
    assert [phase["member_refs"] for phase in plan["phases"]] == [
        ["member-001", "member-002", "member-003", "member-004"],
        ["member-005", "member-006"],
    ]
    for i, phase in enumerate(plan["phases"]):
        family = _claims_of(plan, i)["prompt_family"]
        assert family["facts"]["size"] == len(phase["member_refs"])
        assert family["evidence_refs"] == [f"{ref}:generation.prompt" for ref in phase["member_refs"]]
        assert phase["label_hint"].startswith("Prompt family")
    assert "prompt_shift" not in {claim["kind"] for claim in plan["claims"]}, "no sequence, no shift"

    # interleaved in event order, the same families: without chronology,
    # adjacency means nothing
    shuffled = [LIGHTHOUSE[0], HELMET[0], LIGHTHOUSE[1], HELMET[1], LIGHTHOUSE[2], LIGHTHOUSE[3]]
    document, sha = _snapshot([_member(i, text, precision="day") for i, text in enumerate(shuffled)])
    plan = _planner().plan(document, sha)
    assert sorted(len(phase["member_refs"]) for phase in plan["phases"]) == [2, 4]
    assert planning.validate_plan(plan, document, sha) == []


def test_artifact_and_parameter_changes_are_claims_about_a_boundary_not_boundaries():
    lora_a, lora_b = "a" * 32, "b" * 32
    members = [
        _member(0, LIGHTHOUSE[0], artifacts=(lora_a,)),
        _member(1, LIGHTHOUSE[1], artifacts=(lora_b,)),  # swapped inside the family
        _member(2, HELMET[0], artifacts=(lora_b,), sampler="DPM++ 2M", steps=30),
        _member(3, HELMET[1], artifacts=(lora_b,), sampler="DPM++ 2M", steps=30),
    ]
    document, sha = _snapshot(members)
    plan = _planner().plan(document, sha)
    assert len(plan["phases"]) == 2, "the LoRA swap did not split the lighthouse family"
    second = _claims_of(plan, 1)
    assert second["artifact_change"]["facts"] == {"added": [], "removed": [lora_a]}
    assert second["parameter_change"]["facts"]["changed"]["sampler"] == {"from": ["Euler a"], "to": ["DPM++ 2M"]}
    assert any(ref.startswith("member-001:") for ref in second["artifact_change"]["evidence_refs"])
    assert any(ref.startswith("member-003:") for ref in second["artifact_change"]["evidence_refs"])
    assert plan["phases"][1]["label_hint"].endswith("new artifacts")


def test_without_chronology_families_differ_and_nothing_is_added_or_previous():
    """Day-precision evidence: two families that differ in LoRA and
    sampler yield SYMMETRIC difference claims -- what is used only
    here, what only there -- never added/removed, from/to, or a label
    that says "new". A sequenced plan carrying a symmetric claim, or an
    unsequenced plan carrying a directed one, is refused by the
    validator: direction needs a chronology."""
    lora_a, lora_b = "a" * 32, "b" * 32
    members = [
        _member(0, LIGHTHOUSE[0], artifacts=(lora_a,), precision="day"),
        _member(1, LIGHTHOUSE[1], artifacts=(lora_a,), precision="day"),
        _member(2, HELMET[0], artifacts=(lora_b,), precision="day", sampler="DPM++ 2M"),
        _member(3, HELMET[1], artifacts=(lora_b,), precision="day", sampler="DPM++ 2M"),
    ]
    document, sha = _snapshot(members)
    plan = _planner().plan(document, sha)
    assert plan["v"] == planning.FORMAT_VERSION
    assert plan["subject"]["sequenced"] is False
    second = _claims_of(plan, 1)
    assert second["artifact_difference"]["facts"] == {"only_here": [lora_b], "only_other": [lora_a]}
    assert second["parameter_difference"]["facts"] == {
        "differs": {"sampler": {"here": ["DPM++ 2M"], "other": ["Euler a"]}}
    }
    kinds = {claim["kind"] for claim in plan["claims"]}
    assert not kinds & {"artifact_change", "parameter_change", "prompt_shift"}
    assert plan["phases"][1]["label_hint"] == "Prompt family 2 · different artifacts"
    spelled = json.dumps(plan)
    for word in ("added", "removed", '"from"', '"to"', "new", "previous"):
        assert word not in spelled, f"an unsequenced plan said {word!r}"
    assert planning.validate_plan(plan, document, sha) == []

    # the validator refuses direction without chronology, and vice versa
    directed = copy.deepcopy(plan)
    directed["claims"][-1]["kind"] = "artifact_change"
    directed["claims"][-1]["facts"] = {"added": [lora_b], "removed": [lora_a]}
    assert any("directional claim artifact_change" in why for why in planning.validate_plan(directed, document, sha))
    members = [
        _member(i, text, artifacts=(lora_a,) if i < 2 else (lora_b,)) for i, text in enumerate(LIGHTHOUSE[:2] + HELMET)
    ]
    document, sha = _snapshot(members)
    plan = _planner().plan(document, sha)
    assert plan["subject"]["sequenced"] is True
    assert "artifact_change" in _claims_of(plan, 1)
    symmetric = copy.deepcopy(plan)
    for claim in symmetric["claims"]:
        if claim["kind"] == "artifact_change":
            claim["kind"] = "artifact_difference"
            claim["facts"] = {"only_here": claim["facts"]["added"], "only_other": claim["facts"]["removed"]}
    assert any("symmetric claim artifact_difference" in why for why in planning.validate_plan(symmetric, document, sha))


def test_a_plan_is_an_exact_partition_and_the_validator_has_teeth():
    """validate_plan refuses a plan that omits a member, places one
    twice, names a representative outside its phase, repeats an id, or
    points anywhere outside the snapshot."""
    members = [_member(i, text) for i, text in enumerate(LIGHTHOUSE + HELMET)]
    document, sha = _snapshot(members)
    plan = _planner().plan(document, sha)
    assert planning.validate_plan(plan, document, sha) == []

    omitted = copy.deepcopy(plan)
    omitted["phases"][1]["member_refs"].remove("member-006")
    assert any("missing ['member-006']" in why for why in planning.validate_plan(omitted, document, sha))

    doubled = copy.deepcopy(plan)
    doubled["phases"][1]["member_refs"].append("member-001")
    assert any("repeated ['member-001']" in why for why in planning.validate_plan(doubled, document, sha))

    stray = copy.deepcopy(plan)
    stray["phases"][1]["representative_refs"] = ["member-001"]
    reasons = planning.validate_plan(stray, document, sha)
    assert any("representative member-001 is not one of its members" in why for why in reasons)

    twice = copy.deepcopy(plan)
    twice["phases"][1]["id"] = "phase-001"
    assert "phase ids are not unique" in planning.validate_plan(twice, document, sha)

    outward = copy.deepcopy(plan)
    outward["claims"][0]["evidence_refs"].append("member-001:generation.nonsense")
    outward["phases"][0]["claim_refs"].append("claim-999")
    assert planning.unresolved(outward, document) == ["claim-999", "member-001:generation.nonsense"]

    other = copy.deepcopy(plan)
    other["snapshot_sha256"] = "f" * 64
    assert "the plan names a different snapshot" in planning.validate_plan(other, document, sha)


def test_settings_fail_closed():
    """V1 means exactly phase_threshold: finite, numeric, not bool, in
    [0, 1]. Anything else is refused, never merged into identity."""
    for bad in (
        {"moon_phase": "waning"},
        {"phase_threshold": 0.5, "moon_phase": "waning"},
        {"phase_threshold": True},
        {"phase_threshold": "0.5"},
        {"phase_threshold": math.nan},
        {"phase_threshold": math.inf},
        {"phase_threshold": -1},
        {"phase_threshold": 1.2},
    ):
        with pytest.raises(ValueError, match=r"unknown planner setting|finite number|lies in"):
            _planner(**bad)
    assert _planner(phase_threshold=1).settings == {"phase_threshold": 1.0}, "an int in range is a number"
    assert _planner().settings == {"phase_threshold": 0.5}


def test_the_similarity_producer_contract_is_enforced():
    """N texts -> exactly N vectors of one dimension, all finite. A
    producer that returns fewer, ragged, or NaN vectors is broken, and
    the planner refuses rather than padding it into coherence."""

    class Broken:
        name, version = "broken", "0"

        def __init__(self, rows):
            self.rows = rows

        def embed(self, texts):
            return self.rows

    members = [_member(i, text) for i, text in enumerate(LIGHTHOUSE[:3])]
    document, sha = _snapshot(members)
    for rows, why in (
        ([[1.0, 0.0], [0.0, 1.0]], "2 vectors for 3 texts"),
        ([[1.0, 0.0], [0.0, 1.0], [1.0]], "mixed dimensions"),
        ([[1.0, 0.0], [0.0, math.nan], [1.0, 1.0]], "non-finite"),
        ([[1.0, 0.0], [0.0, "1"], [1.0, 1.0]], "non-numeric"),
        ([[], [], []], "zero-dimensional"),
    ):
        with pytest.raises(ValueError, match=why):
            planning.GenerationHistoryPlanner(Broken(rows)).plan(document, sha)


def test_missing_evidence_is_a_gap_never_positive_shift_evidence():
    """A member with no frozen prompt is placed by chronology and asserts
    nothing about prompts: in a sequenced plan it joins the running
    phase (phases stay contiguous -- the phase list IS the chronology),
    the phase carries a prompt_evidence_missing claim naming it, and it
    is never one side of a prompt_shift. Its OTHER facts survive: its
    LoRA counts at the next boundary, its seed in seed_variation."""
    lora_a, lora_b = "a" * 32, "b" * 32
    members = [
        _member(0, LIGHTHOUSE[0], artifacts=(lora_a,), seed=1),
        _member(1, "", artifacts=(lora_b,), seed=2),  # no prompt; a LoRA change
        _member(2, LIGHTHOUSE[1], artifacts=(lora_a,), seed=3),
        _member(3, HELMET[0], artifacts=(lora_b,), seed=4),
    ]
    document, sha = _snapshot(members)
    plan = _planner().plan(document, sha)
    told = next(one for one in plan["unsupported"] if one["kind"] == "prompt_evidence")
    assert told["member_refs"] == ["member-002"]
    assert [phase["member_refs"] for phase in plan["phases"]] == [
        ["member-001", "member-002", "member-003"],
        ["member-004"],
    ], "the gap joins the running phase; phases are contiguous"
    first = _claims_of(plan, 0)
    assert first["prompt_evidence_missing"]["evidence_refs"] == ["member-002"]
    assert first["prompt_evidence_missing"]["facts"] == {"members": 1}
    assert first["prompt_similarity"]["evidence_refs"] == [
        "member-001:generation.prompt",
        "member-003:generation.prompt",
    ]
    assert first["seed_variation"]["facts"] == {"distinct_seeds": 3}, "the gap member's seed is a fact"
    second = _claims_of(plan, 1)
    assert second["prompt_shift"]["evidence_refs"] == ["member-001:generation.prompt", "member-004:generation.prompt"]
    assert second["artifact_change"]["facts"] == {"added": [], "removed": [lora_a]}, (
        "the gap member's LoRA counted in the phase before the boundary: nothing was added"
    )
    assert plan["phases"][0]["representative_refs"] == ["member-001"], (
        "a representative has a prompt when any member does"
    )
    assert planning.validate_plan(plan, document, sha) == []

    # without chronology the gap is its own family, grouped with nothing
    trio = [LIGHTHOUSE[0], "", LIGHTHOUSE[1]]
    document, sha = _snapshot([_member(i, text, precision="day") for i, text in enumerate(trio)])
    plan = _planner().plan(document, sha)
    assert [phase["member_refs"] for phase in plan["phases"]] == [["member-001", "member-003"], ["member-002"]]
    assert plan["phases"][1]["label_hint"] == "Prompt evidence gap"
    assert planning.validate_plan(plan, document, sha) == []


def test_a_historical_v1_plan_stays_readable_as_history_and_is_planned_again_as_v3(frozen):
    """Planner v4 wrote directed artifact_change claims for UNSEQUENCED
    families -- a real v1 plan, not a v3 plan with "v": 1 stamped on.
    load_plan reads it byte-identically under its frozen grammar and
    the stored-structure rules; the CURRENT producer contract refuses
    it; the renderer declines to read v1 at all; and the same snapshot
    planned today yields a separate v3 plan with artifact_difference."""
    from db import rendering

    client, snap = frozen
    conn = connect.connect(client.app.state.db_path)
    try:
        snapshot = stories.load_snapshot(conn, snap.id)
        # the snapshot's members carry LoRA A on two and LoRA B on one;
        # write what planner v4 wrote for day-precision evidence
        refs = [planning._member_ref(one["ordinal"]) for one in sorted(snapshot["members"], key=lambda m: m["ordinal"])]
        historical = {
            "v": 1,
            "snapshot_sha256": snap.sha256,
            "planner": {
                "kind": "generation_history",
                "version": 4,
                "settings": {"phase_threshold": 0.5},
                "similarity": {"name": "lexical-bow", "version": "1"},
            },
            "subject": {
                "kind": "generation_session",
                "sequenced": False,
                "label_hint": "3 outputs · 2 prompt families",
            },
            "phases": [
                {
                    "id": "phase-001",
                    "member_refs": refs[:2],
                    "representative_refs": [refs[0]],
                    "label_hint": "Prompt family 1",
                    "claim_refs": ["claim-001"],
                },
                {
                    "id": "phase-002",
                    "member_refs": refs[2:],
                    "representative_refs": [refs[2]],
                    "label_hint": "Prompt family 2 · new artifacts",
                    "claim_refs": ["claim-002", "claim-003"],
                },
            ],
            "claims": [
                {
                    "id": "claim-001",
                    "kind": "prompt_family",
                    "confidence": 1.0,
                    "evidence_refs": [f"{r}:generation.prompt" for r in refs[:2]],
                    "facts": {"size": 2, "threshold": 0.5, "min_pairwise_cosine": 0.9},
                },
                {
                    "id": "claim-002",
                    "kind": "prompt_family",
                    "confidence": 1.0,
                    "evidence_refs": [f"{refs[2]}:generation.prompt"],
                    "facts": {"size": 1, "threshold": 0.5, "min_pairwise_cosine": None},
                },
                {
                    "id": "claim-003",
                    "kind": "artifact_change",
                    "confidence": 1.0,
                    "evidence_refs": [f"{r}:generation.artifacts" for r in refs],
                    "facts": {"added": ["f" * 32], "removed": []},
                },
            ],
            "unsupported": [{"kind": "chronology", "reason": "day precision"}],
            "planned_at": NOW,
        }
        _spelled, sha = planning.identity(historical)
        conn.execute(
            "INSERT INTO story_plan(snapshot_id, format_version, planner, planner_version, similarity,"
            " similarity_version, settings_hash, request_sha256, document_json, document_sha256, created_at)"
            " VALUES(?, 1, 'generation_history', 4, 'lexical-bow', '1', 'x', ?, ?, ?, ?)",
            (snap.id, "1" * 64, planning.canonical(historical), sha, NOW),
        )
        conn.commit()
        plan_id = conn.execute("SELECT id FROM story_plan WHERE document_sha256 = ?", (sha,)).fetchone()[0]
        assert planning.load_plan(conn, plan_id) == historical, "history loads byte-identically"
        assert planning.validate_stored_plan(historical, snapshot, snap.sha256) == []
        assert any(
            "directional claim artifact_change" in why
            for why in planning.validate_current_plan(historical, snapshot, snap.sha256)
        )
        assert client.get(f"/stories/plans/{plan_id}").status_code == 200
        with pytest.raises(ValueError, match="reads StorySnapshot"):
            rendering.TemplateStoryRenderer("memory").render(snapshot, historical, snap.sha256, sha)
        assert client.post("/stories/renders", json={"plan_id": plan_id, "profile": "memory"}).status_code == 400

        today = planning.plan_snapshot(conn, snap.id, _planner(), NOW + 40 * HOUR)
        conn.commit()
        assert today.reused is False
        assert today.id != plan_id
        fresh = planning.load_plan(conn, today.id)
        assert fresh["v"] == planning.FORMAT_VERSION
        assert fresh["planner"]["version"] == planning.GenerationHistoryPlanner.version
        assert planning.validate_current_plan(fresh, snapshot, snap.sha256) == []
        assert planning.load_plan(conn, plan_id) == historical, "and history is untouched beside it"
    finally:
        connect.close(conn)


def test_a_planner_v3_gap_plan_from_9575cc6_is_still_history(frozen):
    """The real shape planner v3 wrote (9575cc6): a SEQUENCED v1 plan
    whose promptless member sat in a phase of its own -- phases
    [member-001, member-003], [member-002], [member-004] -- blessed by
    that commit's own tests. Contiguity is the CURRENT producer's rule;
    history loads byte-identically and serves 200."""
    client, snap = frozen
    conn = connect.connect(client.app.state.db_path)
    try:
        snapshot = stories.load_snapshot(conn, snap.id)
        refs = [planning._member_ref(one["ordinal"]) for one in sorted(snapshot["members"], key=lambda m: m["ordinal"])]
        assert len(refs) >= 3
        historical = {
            "v": 1,
            "snapshot_sha256": snap.sha256,
            "planner": {
                "kind": "generation_history",
                "version": 3,
                "settings": {"phase_threshold": 0.5},
                "similarity": {"name": "lexical-bow", "version": "1"},
            },
            "subject": {"kind": "generation_session", "sequenced": True, "label_hint": "3 outputs · 2 phases"},
            "phases": [
                {
                    "id": "phase-001",
                    "member_refs": [refs[0], refs[2]],
                    "representative_refs": [refs[0]],
                    "label_hint": "Phase 1",
                    "claim_refs": ["claim-001"],
                },
                {
                    "id": "phase-002",
                    "member_refs": [refs[1]],
                    "representative_refs": [refs[1]],
                    "label_hint": "Phase 2",
                    "claim_refs": [],
                },
            ],
            "claims": [
                {
                    "id": "claim-001",
                    "kind": "prompt_similarity",
                    "confidence": 0.9,
                    "evidence_refs": [f"{refs[0]}:generation.prompt", f"{refs[2]}:generation.prompt"],
                    "facts": {"relationship": "same_prompt_family", "min_pairwise_cosine": 0.9},
                }
            ],
            "unsupported": [{"kind": "prompt_evidence", "reason": "no frozen prompt", "member_refs": [refs[1]]}],
            "planned_at": NOW,
        }
        _spelled, sha = planning.identity(historical)
        conn.execute(
            "INSERT INTO story_plan(snapshot_id, format_version, planner, planner_version, similarity,"
            " similarity_version, settings_hash, request_sha256, document_json, document_sha256, created_at)"
            " VALUES(?, 1, 'generation_history', 3, 'lexical-bow', '1', 'x', ?, ?, ?, ?)",
            (snap.id, "3" * 64, planning.canonical(historical), sha, NOW),
        )
        conn.commit()
        plan_id = conn.execute("SELECT id FROM story_plan WHERE document_sha256 = ?", (sha,)).fetchone()[0]
        assert planning.validate_stored_plan(historical, snapshot, snap.sha256) == []
        assert any("interleave" in why for why in planning.validate_current_plan(historical, snapshot, snap.sha256))
        assert planning.load_plan(conn, plan_id) == historical
        assert client.get(f"/stories/plans/{plan_id}").status_code == 200
    finally:
        connect.close(conn)


def test_a_blank_prompt_is_never_embedded():
    """Only known prompts reach the engine: an empty string is not a
    prompt, and a vector for nothing is a lie waiting for a consumer."""

    class Spy(planning.LexicalPromptSimilarity):
        seen: typing.ClassVar[list] = []

        def embed(self, texts):
            self.seen.append(list(texts))
            return super().embed(texts)

    members = [_member(0, LIGHTHOUSE[0]), _member(1, ""), _member(2, LIGHTHOUSE[1])]
    document, sha = _snapshot(members)
    planning.GenerationHistoryPlanner(Spy()).plan(document, sha)
    assert Spy.seen == [[LIGHTHOUSE[0], LIGHTHOUSE[1]]]
    assert "" not in Spy.seen[0]


def test_a_sequenced_plan_is_a_chronology_and_the_validator_refuses_interleaving():
    members = [_member(i, text) for i, text in enumerate(LIGHTHOUSE + HELMET)]
    document, sha = _snapshot(members)
    plan = _planner().plan(document, sha)
    order = [int(ref.split("-")[1]) for phase in plan["phases"] for ref in phase["member_refs"]]
    assert order == sorted(order)
    bent = copy.deepcopy(plan)
    bent["phases"][0]["member_refs"], bent["phases"][1]["member_refs"] = (
        ["member-001", "member-002", "member-003", "member-006"],
        ["member-004", "member-005"],
    )
    assert any("interleave" in why for why in planning.validate_plan(bent, document, sha))


def test_the_same_request_is_one_identity_and_policies_coexist():
    members = [_member(i, text) for i, text in enumerate(LIGHTHOUSE + HELMET)]
    document, sha = _snapshot(members)
    engine = planning.LexicalPromptSimilarity()
    assert engine.embed(LIGHTHOUSE) == engine.embed(LIGHTHOUSE)
    one = _planner().plan(document, sha)
    two = _planner().plan(document, sha)
    assert planning.identity(one) == planning.identity(two)
    strict = _planner(phase_threshold=0.99).plan(document, sha)
    assert planning.identity(strict)[1] != planning.identity(one)[1]
    assert len(strict["phases"]) > len(one["phases"]), "a stricter threshold splits the near-variants"
    loose = {"phase_threshold": 0.5}
    request = planning.request_identity(sha, "generation_history", 7, "lexical-bow", "1", loose)
    assert request == planning.request_identity(sha, "generation_history", 7, "lexical-bow", "1", loose)
    strict_settings = {"phase_threshold": 0.99}
    assert request != planning.request_identity(sha, "generation_history", 7, "lexical-bow", "1", strict_settings)
    assert request != planning.request_identity(sha, "generation_history", 7, "lexical-bow", "2", loose)


def test_the_document_format_is_part_of_the_request_identity(monkeypatch):
    """A format change must never hand back yesterday's shape under an
    unchanged planner version: FORMAT_VERSION rides the request hash."""
    sha = "a" * 64
    before = planning.request_identity(sha, "generation_history", 7, "lexical-bow", "1", {"phase_threshold": 0.5})
    monkeypatch.setattr(planning, "FORMAT_VERSION", planning.FORMAT_VERSION + 1)
    after = planning.request_identity(sha, "generation_history", 7, "lexical-bow", "1", {"phase_threshold": 0.5})
    assert before != after


def test_the_threshold_is_pinned_against_the_engine():
    """The lexical oracle's numbers are known: two ten-token prompts
    differing in one word cosine to 11/12; lighthouse vs helmet share
    'a' and little else. The default threshold sits between."""
    engine = planning.LexicalPromptSimilarity()
    cosine = planning.pairwise_cosine(engine.embed([LIGHTHOUSE[0], LIGHTHOUSE[1], HELMET[0]]))
    assert cosine[0][1] == pytest.approx(11 / 12, abs=1e-3)
    assert cosine[0][2] < 0.5 < cosine[0][1]
    assert planning.GenerationHistoryPlanner.defaults["phase_threshold"] == 0.5


# --- persistence, the service and the job, against a real frozen snapshot ----


def _library(root: pathlib.Path) -> None:
    # the hero is the phase's medoid (ties to the earliest), so the
    # escaping probe rides the FIRST name; NTFS forbids < >, & and ' probe
    names = ["gen_0 & 'friends'.png", "gen_1.png", "gen_2.png"]
    for i, (text, name) in enumerate(zip(LIGHTHOUSE[:3], names, strict=True)):
        info = PngInfo()
        info.add_text(
            "parameters",
            f"{text}\nNegative prompt: blur\n"
            f"Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: {100 + i}, Size: 512x512, Model: alpha",
        )
        Image.new("RGB", (12, 12), (40 + i * 30, 90, 140)).save(root / name, pnginfo=info)


def _drain(client) -> None:
    conn = connect.connect(client.app.state.db_path)
    try:
        while runner.run_next(conn, "test-worker", NOW + 24 * HOUR) is not None:
            conn.commit()
        conn.commit()
    finally:
        connect.close(conn)


def _prepare(stage: Stage) -> None:
    client, root = stage.client, stage.root
    conn = connect.connect(client.app.state.db_path)
    try:
        names = dict(conn.execute("SELECT name, id FROM file").fetchall())
        for name, file_id in names.items():
            ingest.one(conn, file_id, root / name, NOW)
        conn.executemany(
            "INSERT OR REPLACE INTO file_param(file_id, source, key, value_text) VALUES(?, 'generation', 'date', ?)",
            [(names[name], _spelled(NOW + i * 4 * MIN)) for i, name in enumerate(sorted(names))],
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
        made = stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
        conn.commit()
    finally:
        connect.close(conn)
    stage.held["made"] = made


@pytest.fixture(scope="module")
def _stage(tmp_path_factory):
    with staged(tmp_path_factory, "test_a_story_plan_is_structure_from_frozen_evidence", _library, _prepare) as stage:
        yield stage


@pytest.fixture
def frozen(_stage):
    _stage.restore()
    return _stage.client, _stage.held["made"]


@pytest.fixture
def planned(frozen):
    """The frozen world with its plan made -- per test, on the restored
    snapshot, so the plan tests above still see an unplanned snapshot."""
    client, snap = frozen
    conn = connect.connect(client.app.state.db_path)
    try:
        planner = planning.GenerationHistoryPlanner(planning.LexicalPromptSimilarity())
        made = planning.plan_snapshot(conn, snap.id, planner, NOW + 31 * HOUR)
        conn.commit()
    finally:
        connect.close(conn)
    return client, snap, made


def test_a_persisted_plan_is_immune_to_everything_that_happens_after_the_snapshot(frozen):
    client, snap = frozen
    conn = connect.connect(client.app.state.db_path)
    try:
        first = planning.plan_snapshot(conn, snap.id, _planner(), NOW + 31 * HOUR)
        conn.commit()
        assert first.reused is False
        member = conn.execute("SELECT id FROM file WHERE name = 'gen_1.png'").fetchone()[0]
        conn.execute("UPDATE generation SET seed = 4242 WHERE file_id = ?", (member,))
        context.stale(conn, member)
        conn.commit()
    finally:
        connect.close(conn)
    client.post("/jobs/context")
    client.post("/jobs/events")
    _drain(client)
    conn = connect.connect(client.app.state.db_path)
    try:
        again = planning.plan_snapshot(conn, snap.id, _planner(), NOW + 40 * HOUR)
        conn.commit()
        assert (again.id, again.sha256, again.reused) == (first.id, first.sha256, True)
        told = planning.load_plan(conn, first.id)
        assert told["snapshot_sha256"] == snap.sha256
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE story_plan SET document_json = '{}' WHERE id = ?", (first.id,))
        conn.rollback()
        other = planning.plan_snapshot(conn, snap.id, _planner(phase_threshold=0.99), NOW + 41 * HOUR)
        conn.commit()
        assert other.id != first.id, "a new policy coexists with the old plan"
        assert conn.execute("SELECT count(*) FROM story_plan WHERE snapshot_id = ?", (snap.id,)).fetchone()[0] == 2
        with pytest.raises(LookupError, match="no story snapshot"):
            planning.plan_snapshot(conn, 99_999, _planner(), NOW + 42 * HOUR)
    finally:
        connect.close(conn)


def test_a_tampered_snapshot_or_plan_is_refused_not_served(frozen):
    """The stored bytes must hash to the stored identity -- on planning
    (the input) and on reading (the output). The immutability triggers
    are bypassed here the only way they can be: by dropping them, which
    is what a corrupted file amounts to."""
    client, snap = frozen
    conn = connect.connect(client.app.state.db_path)
    try:
        made = planning.plan_snapshot(conn, snap.id, _planner(), NOW + 31 * HOUR)
        conn.commit()
        conn.execute("DROP TRIGGER story_plan_is_immutable")
        conn.execute("DROP TRIGGER story_snapshot_is_immutable")
        good_plan = conn.execute("SELECT document_json FROM story_plan WHERE id = ?", (made.id,)).fetchone()[0]
        tampered = json.loads(good_plan)
        tampered["phases"][0]["label_hint"] = "Something the evidence never said"
        conn.execute("UPDATE story_plan SET document_json = ? WHERE id = ?", (json.dumps(tampered), made.id))
        conn.commit()
        with pytest.raises(ValueError, match="no longer hashes"):
            planning.load_plan(conn, made.id)
        assert client.get(f"/stories/plans/{made.id}").status_code == 409
        conn.execute("UPDATE story_plan SET document_json = ? WHERE id = ?", (good_plan, made.id))
        conn.commit()
        assert client.get(f"/stories/plans/{made.id}").status_code == 200, "restored bytes serve again"

        good_snap = conn.execute("SELECT document_json FROM story_snapshot WHERE id = ?", (snap.id,)).fetchone()[0]
        bent = json.loads(good_snap)
        bent["members"][0]["generation"]["prompt"] = "a prompt nobody wrote"
        conn.execute("UPDATE story_snapshot SET document_json = ? WHERE id = ?", (json.dumps(bent), snap.id))
        conn.commit()
        with pytest.raises(ValueError, match="no longer hashes"):
            planning.plan_snapshot(conn, snap.id, _planner(phase_threshold=0.7), NOW + 32 * HOUR)
        conn.rollback()
        with pytest.raises(ValueError, match="no longer hashes"):
            stories.load_snapshot(conn, snap.id)
        with pytest.raises(ValueError, match="no longer hashes"):
            planning.load_plan(conn, made.id)  # a good plan over a corrupt snapshot is not servable either
    finally:
        connect.close(conn)


def test_planning_is_durable_work_and_an_identical_request_reuses_it(frozen):
    """POST records the request and queues a story_plan job (202); the
    same request again while queued reuses the JOB; once the worker has
    planned, the same request is 200 with the plan id and no second
    job, no second embedding. Unknown engine or settings are 400 before
    anything is queued."""
    client, snap = frozen
    asked = client.post("/stories/plans", json={"snapshot_id": snap.id, "similarity": "lexical"})
    assert asked.status_code == 202, asked.text
    body = asked.json()
    assert body["plan_id"] is None
    assert body["job"]["kind"] == "story_plan"
    again = client.post("/stories/plans", json={"snapshot_id": snap.id, "similarity": "lexical"})
    assert again.status_code == 202
    assert again.json()["job"]["id"] == body["job"]["id"], "the queued job is reused, not duplicated"
    assert again.json()["request_sha256"] == body["request_sha256"]

    assert client.post("/stories/plans", json={"snapshot_id": snap.id, "similarity": "astrology"}).status_code == 400
    assert client.post("/stories/plans", json={"snapshot_id": snap.id, "planner": "vibes"}).status_code == 400
    assert client.post("/stories/plans", json={"snapshot_id": snap.id, "settings": {"moon": 1}}).status_code == 400
    assert client.post("/stories/plans", json={"snapshot_id": 99_999, "similarity": "lexical"}).status_code == 404
    assert client.post("/stories/plans", json={}).status_code == 400

    _drain(client)
    settled = client.post("/stories/plans", json={"snapshot_id": snap.id, "similarity": "lexical"})
    assert settled.status_code == 200
    plan_id = settled.json()["plan_id"]
    assert plan_id is not None
    assert settled.json()["job"] is None
    conn = connect.connect(client.app.state.db_path)
    try:
        assert conn.execute("SELECT count(*) FROM job WHERE kind = 'story_plan'").fetchone()[0] == 1
        assert conn.execute("SELECT state FROM job WHERE kind = 'story_plan'").fetchone()[0] == "done"
        request = conn.execute("SELECT request_sha256 FROM story_plan WHERE id = ?", (plan_id,)).fetchone()[0]
        assert request == body["request_sha256"]
    finally:
        connect.close(conn)
    read = client.get(f"/stories/plans/{plan_id}")
    assert read.status_code == 200
    assert read.json()["planner"]["similarity"] == {"name": "lexical-bow", "version": "1"}
    assert read.json()["phases"][0]["member_refs"] == ["member-001", "member-002", "member-003"]
    assert client.get("/stories/plans/424242").status_code == 404


def test_the_engine_selector_means_exactly_what_it_says(frozen):
    """`openclip` is OpenCLIP, `qwen` is Qwen; a provider that is not
    among the configured semantic spaces is refused, never substituted
    by whichever provider happened to be first. The semantic engine's
    identity names the pinned checkpoint with no weights loaded."""
    client, _ = frozen
    conn = connect.connect(client.app.state.db_path)
    try:
        assert isinstance(planning.engine_for(conn, "lexical", "unused"), planning.LexicalEngine)
        with pytest.raises(ValueError, match="no similarity engine named"):
            planning.engine_for(conn, "astrology", "unused")
        with pytest.raises(ValueError, match="no configured semantic space matches"):
            planning.engine_for(conn, "qwen", "unused")
        engine = planning.engine_for(conn, "openclip", "unused")
        assert isinstance(engine, planning.SemanticEngine)
        assert engine.provider == "openclip"
        name, version = engine.identity()
        assert engine.model in name
        assert version.startswith("q"), "the version is a digest of the query policy"
    finally:
        connect.close(conn)


def test_the_story_plan_grammar_is_exact_and_fails_closed():
    """A self-consistent, correctly hashed document is still invalid when
    it carries an unknown key, a claim kind nobody defined, facts of the
    wrong shape, a second representative, or a boolean confidence -- and
    the answer is a controlled reason, never an exception, whatever the
    bytes say."""
    members = [_member(i, text) for i, text in enumerate(LIGHTHOUSE[:2] + HELMET[:1])]
    document, sha = _snapshot(members)
    plan = _planner().plan(document, sha)
    assert planning.validate_story_plan(plan) == []

    def broken(mutate):
        bent = copy.deepcopy(plan)
        mutate(bent)
        reasons = planning.validate_story_plan(bent)
        assert reasons, "the grammar accepted a malformed document"

    broken(lambda p: p.__setitem__("narrative", "a story"))
    broken(lambda p: p["phases"][0].__setitem__("mood", "wistful"))
    broken(lambda p: p["claims"][0].__setitem__("kind", "vibe_shift"))
    broken(lambda p: p["claims"][0].__setitem__("facts", {"cosine": "high"}))
    broken(lambda p: p["claims"][0].__setitem__("confidence", True))
    broken(lambda p: p["claims"][0].__setitem__("evidence_refs", []))
    broken(lambda p: p["phases"][0].__setitem__("representative_refs", ["member-001", "member-002"]))
    broken(lambda p: p["phases"][0].__setitem__("id", "phase-1"))
    broken(lambda p: p["planner"]["settings"].__setitem__("moon", 1))
    broken(lambda p: p["subject"].__setitem__("sequenced", "yes"))
    broken(lambda p: p["unsupported"].append({"kind": "weather", "reason": "rain"}))
    broken(lambda p: p.__setitem__("snapshot_sha256", "nope"))
    for garbage in (None, [], "plan", {"v": 1}, {"v": 1, "phases": "many"}, {"v": 2}):
        assert planning.validate_story_plan(garbage), f"{garbage!r} read as a plan"


def test_query_policy_is_the_engine_identity_not_the_stored_space(monkeypatch):
    """Qwen's stored-media policy deliberately omits QUERY_INSTRUCTION;
    the planner's vectors are QUERY vectors, so the instruction must be
    part of the engine identity -- change it and the request identity
    changes. OpenCLIP's identity names its pinned checkpoint."""
    from vision import semantic
    from vision.semantic import openclip, qwen_vl

    qwen = planning.SemanticEngine("qwen", "Qwen/Qwen3-VL-Embedding-2B", "c" * 40, "unused")
    before = qwen.identity()
    monkeypatch.setattr(qwen_vl, "QUERY_INSTRUCTION", "Find the vibe.")
    after = qwen.identity()
    assert before[0] == after[0], "the space is the same space"
    assert before[1] != after[1], "but the QUERY policy is a different engine"
    clip = planning.SemanticEngine("openclip", openclip.MODEL, openclip.CHECKPOINT, "unused")
    name, version = clip.identity()
    assert openclip.CHECKPOINT in name
    assert version.startswith("q")
    assert semantic.query_policy("openclip", openclip.MODEL, openclip.CHECKPOINT)["checkpoint"] == openclip.CHECKPOINT


def test_a_mutable_checkpoint_is_never_queued_and_a_moved_one_is_never_loaded(frozen, monkeypatch):
    """Qwen configured at `main` with nothing provisioned cannot be pinned,
    so no plan may be queued under provenance that could move. And a
    worker that loads weights whose identity differs from the queued
    engine refuses rather than planning under a lie."""
    from vision import semantic

    client, _ = frozen
    conn = connect.connect(client.app.state.db_path)
    try:
        monkeypatch.setattr("db.retrieval.choices", lambda _conn: [("qwen", "Qwen/Qwen3-VL-Embedding-2B", "main")])
        with pytest.raises(ValueError, match="mutable revision"):
            planning.engine_for(conn, "qwen", "unused")
    finally:
        connect.close(conn)

    class Weights:
        model_id = "fake"
        dimensions = 2

        def space(self):
            return semantic.space("qwen", "Qwen/Qwen3-VL-Embedding-2B", "d" * 40, 1)

        def encode_query(self, text):
            return [1.0, 0.0]

    queued = planning.SemanticEngine("qwen", "Qwen/Qwen3-VL-Embedding-2B", "c" * 40, "unused")
    with pytest.raises(ValueError, match="not the engine the request was queued under"):
        planning.SemanticPromptSimilarity(Weights(), queued)
    same = planning.SemanticEngine("qwen", "Qwen/Qwen3-VL-Embedding-2B", "d" * 40, "unused")
    assert planning.SemanticPromptSimilarity(Weights(), same).version == same.identity()[1]


def test_concurrent_identical_requests_create_one_job(frozen):
    """Two connections asking for the same plan at once: the second
    waits on the writer lane the first holds, then finds the job the
    first queued. One job row, one model run."""
    import threading

    client, snap = frozen
    engine = planning.LexicalEngine()
    first = connect.connect(client.app.state.db_path)
    outcome: dict = {}
    try:
        held = planning.request_plan(first, snap.id, "generation_history", engine, None, NOW + 31 * HOUR)
        assert held.job_id is not None

        def race():
            second = connect.connect(client.app.state.db_path)  # its own thread, its own connection
            try:
                outcome["second"] = planning.request_plan(
                    second, snap.id, "generation_history", engine, None, NOW + 31 * HOUR
                )
                second.commit()
            except (ValueError, LookupError, sqlite3.Error) as why:
                outcome["error"] = why
            finally:
                connect.close(second)

        waiter = threading.Thread(target=race)
        waiter.start()
        waiter.join(timeout=0.5)
        assert waiter.is_alive(), f"the second request must wait on the lane, not race past it: {outcome}"
        first.commit()
        waiter.join(timeout=10)
        assert not waiter.is_alive()
        assert "error" not in outcome, outcome.get("error")
        assert outcome["second"].job_id == held.job_id, "exactly one live job for one request"
        assert first.execute("SELECT count(*) FROM job WHERE kind = 'story_plan'").fetchone()[0] == 1
    finally:
        connect.close(first)


def test_a_corrupt_snapshot_is_a_409_on_the_wire(frozen):
    client, snap = frozen
    conn = connect.connect(client.app.state.db_path)
    try:
        conn.execute("DROP TRIGGER story_snapshot_is_immutable")
        conn.execute("UPDATE story_snapshot SET document_json = ? WHERE id = ?", ('{"v": 1}', snap.id))
        conn.commit()
    finally:
        connect.close(conn)
    assert client.get(f"/stories/snapshots/{snap.id}").status_code == 409


def test_the_v1_grammar_is_frozen_and_exception_proof(monkeypatch):
    """A v1 row must still parse as v1 after later versions exist: the
    grammar reads frozen constants, never the running FORMAT_VERSION or
    registry. A v1 document is a v2 document without the symmetric
    claims, so one is made here by stamping v1 on a sequenced plan. And
    any bytes -- unhashable kinds included -- yield controlled reasons,
    never an exception."""
    members = [_member(i, text) for i, text in enumerate(LIGHTHOUSE[:2] + HELMET[:1])]
    document, sha = _snapshot(members)
    plan = {**_planner().plan(document, sha), "v": 1}
    assert planning.validate_story_plan_v1(plan) == []
    monkeypatch.setattr(planning, "FORMAT_VERSION", 4)
    monkeypatch.setattr(planning, "PLANNERS", {})
    assert planning.validate_story_plan_v1(plan) == [], "v1 is judged by v1's frozen vocabulary"
    assert planning.validate_story_plan(plan) == [], "the dispatcher routes a v1 document to the v1 grammar"
    assert planning.validate_story_plan({**plan, "v": 2}) == [], "v1's vocabulary is inside v2's"
    assert planning.validate_story_plan({**plan, "v": 3}) == [], "and inside v3's"
    assert planning.validate_story_plan({**plan, "v": 8}), "an undefined version is invalid, not a crash"
    assert planning.validate_story_plan({**plan, "v": True})
    v1_only = {
        **plan,
        "claims": [
            *plan["claims"],
            {
                "id": "claim-099",
                "kind": "artifact_difference",
                "confidence": 1.0,
                "evidence_refs": ["member-001"],
                "facts": {"only_here": ["x"], "only_other": []},
            },
        ],
    }
    assert any("not a v1 claim kind" in why for why in planning.validate_story_plan_v1(v1_only))
    assert planning.validate_story_plan_v2({**v1_only, "v": 2}) == []

    def bent(mutate):
        held = copy.deepcopy(plan)
        mutate(held)
        reasons = planning.validate_story_plan_v1(held)
        assert reasons, "the grammar accepted a malformed document"
        return reasons

    bent(lambda p: p["planner"].__setitem__("kind", []))
    bent(lambda p: p["claims"][0].__setitem__("kind", {}))
    bent(lambda p: p["unsupported"].append({"kind": [], "reason": "x"}))
    bent(lambda p: p["subject"].__setitem__("kind", {"a": 1}))
    bent(
        lambda p: p["claims"].append(
            {
                "id": "claim-099",
                "kind": "seed_variation",
                "confidence": 1.0,
                "evidence_refs": ["member-001:generation.seed"],
                "facts": {"distinct_seeds": True},
            }
        )
    )
    bent(
        lambda p: p["claims"].append(
            {
                "id": "claim-099",
                "kind": "seed_variation",
                "confidence": 1.0,
                "evidence_refs": ["member-001:generation.seed"],
                "facts": {"distinct_seeds": 1},
            }
        )
    )
    bent(
        lambda p: p["claims"].append(
            {
                "id": "claim-099",
                "kind": "seed_variation",
                "confidence": 1.0,
                "evidence_refs": ["member-001:generation.seed"],
                "facts": {"distinct_seeds": 0},
            }
        )
    )
    bent(
        lambda p: p["claims"].append(
            {
                "id": "claim-099",
                "kind": "prompt_family",
                "confidence": 1.0,
                "evidence_refs": ["member-001:generation.prompt"],
                "facts": {"size": 0, "threshold": 0.5, "min_pairwise_cosine": None},
            }
        )
    )
    bent(
        lambda p: p["claims"].append(
            {
                "id": "claim-099",
                "kind": "prompt_shift",
                "confidence": 1.0,
                "evidence_refs": ["member-001:generation.prompt"],
                "facts": {"cosine": 7, "threshold": 0.5},
            }
        )
    )
    bent(
        lambda p: p["claims"].append(
            {
                "id": "claim-099",
                "kind": "prompt_shift",
                "confidence": 1.5,
                "evidence_refs": ["member-001:generation.prompt"],
                "facts": {"cosine": 0.1, "threshold": 0.5},
            }
        )
    )


def test_a_queued_request_is_re_proven_before_any_weights_load(frozen, monkeypatch):
    """A deploy between queue and run that changes the plan format, the
    planner or an engine's query policy makes the queued job a stale
    ask: the worker recomputes the request identity from what it would
    actually do and refuses on mismatch -- before loading anything --
    and no plan lands under a new identity."""
    client, snap = frozen
    asked = client.post("/stories/plans", json={"snapshot_id": snap.id, "similarity": "lexical"})
    assert asked.status_code == 202
    job_id = asked.json()["job"]["id"]

    monkeypatch.setattr(planning, "FORMAT_VERSION", planning.FORMAT_VERSION + 1)
    _drain(client)
    conn = connect.connect(client.app.state.db_path)
    try:
        settled = client.get(f"/jobs/{job_id}").json()
        assert settled["state"] == "done", settled
        assert settled["failed_count"] == 1, settled
        assert conn.execute("SELECT count(*) FROM story_plan").fetchone()[0] == 0, "no plan under a new identity"
        why = conn.execute("SELECT error FROM job_item WHERE job_id = ?", (job_id,)).fetchone()[0]
        assert "no longer means what it meant" in why
    finally:
        connect.close(conn)

    # the same under a planner-version change
    monkeypatch.undo()
    asked = client.post("/stories/plans", json={"snapshot_id": snap.id, "similarity": "lexical"})
    assert asked.status_code == 202, "a fresh ask under the restored policy queues a new job"
    monkeypatch.setattr(planning.GenerationHistoryPlanner, "version", planning.GenerationHistoryPlanner.version + 1)
    _drain(client)
    conn = connect.connect(client.app.state.db_path)
    try:
        assert conn.execute("SELECT count(*) FROM story_plan").fetchone()[0] == 0
    finally:
        connect.close(conn)

    # and a loaded engine whose identity drifted is refused the same way
    loaded = planning.LexicalEngine()
    monkeypatch.undo()
    same = planning.request_identity(
        snap.sha256,
        "generation_history",
        planning.GenerationHistoryPlanner.version,
        *loaded.identity(),
        {"phase_threshold": 0.5},
    )
    conn = connect.connect(client.app.state.db_path)
    try:
        with pytest.raises(ValueError, match="no longer means"):
            planning.plan_item(
                conn,
                0,
                {
                    "request_sha256": "0" * 64,
                    "snapshot_id": snap.id,
                    "planner": "generation_history",
                    "settings": {"phase_threshold": 0.5},
                    "engine": {"selector": "lexical"},
                },
                NOW + 40 * HOUR,
            )
        conn.rollback()
        planning.plan_item(
            conn,
            0,
            {
                "request_sha256": same,
                "snapshot_id": snap.id,
                "planner": "generation_history",
                "settings": {"phase_threshold": 0.5},
                "engine": {"selector": "lexical"},
            },
            NOW + 41 * HOUR,
        )
        conn.commit()
        assert conn.execute("SELECT count(*) FROM story_plan").fetchone()[0] == 1, "the matching request plans"
    finally:
        connect.close(conn)


# --- a story render is persisted, reused and laid out -----------------------


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
    assert html.count('href="/i/') >= len(story["sections"]), "a hero links to its picture through address resolution"
    assert html.count('src="/thumb/') >= len(story["support"]["member_refs"]), "every member is shown, not only named"
    assert "data-story-evolution" in html
    assert "/evolution" in html
    assert "data-story-profile-here" in html, "the page names the profile it was told under"
    for other in rendering.PROFILES:
        if other != story["renderer"]["profile"]:
            assert f'data-story-profile-ask="{other}"' in html, "and offers the others"
    assert re.search(r'src="/static/build/story\.js\?v=\d+"', html), "the script carries the static version stamp"
    assert client.get("/stories/renders/424242").status_code == 404
    conn = connect.connect(client.app.state.db_path)
    try:
        assert resultset.currency(conn) == before, "a GET of a story wrote something"
    finally:
        connect.close(conn)


# --- a story snapshot freezes evidence -------------------------------------


def _storied_library(root: pathlib.Path) -> None:
    for i in range(3):
        info = PngInfo()
        info.add_text(
            "parameters",
            f"a tin lighthouse, variant {i}\nNegative prompt: blur\n"
            f"Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: {100 + i}, Size: 512x512, Model: alpha",
        )
        Image.new("RGB", (12, 12), (40 + i * 30, 90, 140)).save(root / f"gen_{i}.png", pnginfo=info)


def _storied_prepare(stage: Stage) -> None:
    """Three generated stills forming ONE current generation session,
    with the generator's own parameter bag present to be frozen."""
    client, root = stage.client, stage.root
    conn = connect.connect(client.app.state.db_path)
    try:
        names = dict(conn.execute("SELECT name, id FROM file").fetchall())
        for name, file_id in names.items():
            ingest.one(conn, file_id, root / name, NOW)
        conn.executemany(
            "INSERT OR REPLACE INTO file_param(file_id, source, key, value_text) VALUES(?, 'generation', ?, ?)",
            [
                *[(names[f"gen_{i}.png"], "date", _spelled(NOW + i * 4 * MIN)) for i in range(3)],
                *[(names[f"gen_{i}.png"], "original_prompt", "a __material__ lighthouse") for i in range(3)],
            ],
        )
        conn.commit()
    finally:
        connect.close(conn)
    client.post("/jobs/context")
    client.post("/jobs/events")
    _drain(client)


@pytest.fixture(scope="module")
def _storied_stage(tmp_path_factory):
    with staged(tmp_path_factory, "storied", _storied_library, _storied_prepare) as stage:
        yield stage


@pytest.fixture
def storied(_storied_stage):
    _storied_stage.restore()
    return _storied_stage.client, _storied_stage.root


def _event(conn) -> int:
    return conn.execute(
        "SELECT e.id FROM derived_event e JOIN derived_event_run r ON r.id = e.run_id"
        " WHERE e.kind = 'generation_session'"
    ).fetchone()[0]


def test_a_snapshot_freezes_identity_bytes_occurrence_and_facts(storied):
    """The document carries what a later story needs and nothing a later
    story could re-derive: uuids AND observed content hashes, the claim
    that placed each member here, prompts, the generator's own bag."""
    client, _ = storied
    conn = connect.connect(client.app.state.db_path)
    try:
        made = stories.snapshot_event(conn, _event(conn), NOW + 30 * HOUR)
        conn.commit()
        assert made.reused is False
        told = stories.load_snapshot(conn, made.id)
        assert told["v"] == stories.FORMAT_VERSION
        subject = told["subject"]
        assert (subject["event_kind"], subject["claim"], subject["grouper"]) == (
            "generation_session",
            "generation",
            "generation_session",
        )
        assert (subject["context_generation"], subject["context_policy_version"]) == context.state(conn)
        assert subject["time"]["local"] == [NOW, NOW + 8 * MIN], "the interval in the domain the event knows"
        assert subject["time"]["instant"] is None, "and no invented instant"
        members = told["members"]
        assert [one["ordinal"] for one in members] == [0, 1, 2]
        shas = dict(conn.execute("SELECT name, content_sha256 FROM file").fetchall())
        uuids = {
            name: uuid.hex()
            for name, uuid in conn.execute("SELECT f.name, e.uuid FROM file f JOIN entity e ON e.id = f.id")
        }
        for one in members:
            assert one["content_sha256"] == shas[one["name"]], "the bytes actually observed, frozen"
            assert one["file_uuid"] == uuids[one["name"]]
            assert one["occurrence"]["kind"] == "generation"
            assert (one["occurrence"]["basis"], one["occurrence"]["precision"]) == ("embedded", "second")
            assert one["generation"]["prompt"].startswith("a tin lighthouse, variant")
            assert one["generation"]["params"]["original_prompt"] == "a __material__ lighthouse"
            assert one["generation"]["seed"] in (100, 101, 102)
            assert one["capture"] is None, "a generated still makes no camera claim"
            assert one["people"] is None
            assert one["lineage"] is None
            assert one["annotations"] is None
        assert stories.verify(told, made.sha256), "the stored document hashes to the identity it was stored under"
    finally:
        connect.close(conn)


def test_identical_evidence_is_one_snapshot(storied):
    """The identity is the evidence's hash: freezing the same current
    event twice -- at different wall-clock instants -- is one row."""
    client, _ = storied
    conn = connect.connect(client.app.state.db_path)
    try:
        first = stories.snapshot_event(conn, _event(conn), NOW + 30 * HOUR)
        conn.commit()
        again = stories.snapshot_event(conn, _event(conn), NOW + 31 * HOUR)
        conn.commit()
        assert (again.id, again.sha256, again.reused) == (first.id, first.sha256, True)
        assert conn.execute("SELECT count(*) FROM story_snapshot").fetchone()[0] == 1
    finally:
        connect.close(conn)


def test_the_snapshot_outlives_everything_it_was_made_from(storied, monkeypatch):
    """The hostile acceptance test, modelled on every mistake already
    paid for. After the snapshot: a member is renamed, a member's bytes
    are replaced in place under the same address, a member's prompt and
    seed change, the interpretation policy moves, the library regroups
    so the original event disappears and its run is deleted. The old
    snapshot loads byte-identical -- original uuids, original content
    hashes, original prompt, original occurrence times, original
    ordering and member hash, original provenance -- and a fresh
    snapshot of the new current world has a different identity."""
    client, root = storied
    conn = connect.connect(client.app.state.db_path)
    try:
        event_id = _event(conn)
        frozen = stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
        conn.commit()
        before = stories.load_snapshot(conn, frozen.id)
        original_run = conn.execute("SELECT run_id FROM derived_event WHERE id = ?", (event_id,)).fetchone()[0]
        ids = dict(conn.execute("SELECT name, id FROM file").fetchall())
    finally:
        connect.close(conn)

    # rename A on disk; replace B's bytes in place; change C's prompt and seed
    os.rename(root / "gen_0.png", root / "renamed_a.png")
    info = PngInfo()
    info.add_text("parameters", "something else entirely\nSteps: 5, Seed: 7")
    Image.new("RGB", (12, 12), (250, 250, 250)).save(root / "gen_1.png", pnginfo=info)
    client.post("/roots/1/scan")
    conn = connect.connect(client.app.state.db_path)
    try:
        prompt_id = scan.mint(conn, "prompt", "a brass diving helmet")
        conn.execute(
            "INSERT INTO prompt(id, text, text_hash, created_at) VALUES(?, 'a brass diving helmet', 'h-helmet', ?)",
            (prompt_id, NOW),
        )
        conn.execute("UPDATE generation SET seed = 999 WHERE file_id = ?", (ids["gen_2.png"],))
        conn.execute(
            "UPDATE generation_prompt SET prompt_id = ? WHERE file_id = ? AND role = 'effective'",
            (prompt_id, ids["gen_2.png"]),
        )
        conn.execute(
            "UPDATE file_param SET value_text = ? WHERE file_id = ? AND source = 'generation' AND key = 'date'",
            (_spelled(NOW + 3 * HOUR), ids["gen_2.png"]),
        )
        context.stale(conn, ids["gen_2.png"])
        conn.commit()
    finally:
        connect.close(conn)

    # the interpretation policy moves on (a real upgrade: the running
    # constant changes and the jobs re-interpret at the new meaning),
    # and the library regroups
    monkeypatch.setattr(context, "POLICY_VERSION", context.POLICY_VERSION + 1)
    client.post("/jobs/context")
    client.post("/jobs/events")
    _drain(client)

    conn = connect.connect(client.app.state.db_path)
    try:
        gone = conn.execute("SELECT count(*) FROM derived_event_run WHERE id = ?", (original_run,)).fetchone()[0]
        assert gone == 0, "the original hypothesis is gone; regrouping keeps no museum"
        # SQLite reuses the rowid: whatever now sits at the old event id is
        # a DIFFERENT hypothesis -- which is why the snapshot keeps the id
        # as provenance only, never as the thing it survives by.
        reused = conn.execute("SELECT run_id FROM derived_event WHERE id = ?", (event_id,)).fetchone()
        assert reused is None or reused[0] != original_run

        after = stories.load_snapshot(conn, frozen.id)
        assert after == before, "the snapshot is BYTE-IDENTICAL after everything it was made from changed"
        assert stories.verify(after, frozen.sha256)
        names = [one["name"] for one in after["members"]]
        assert names == ["gen_0.png", "gen_1.png", "gen_2.png"], "original names and ordering"
        live_sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (ids["gen_1.png"],)).fetchone()[0]
        frozen_sha = next(one["content_sha256"] for one in after["members"] if one["name"] == "gen_1.png")
        assert frozen_sha != live_sha, "the entity survived a replacement; the snapshot kept yesterday's bytes"
        frozen_c = next(one for one in after["members"] if one["name"] == "gen_2.png")
        assert frozen_c["generation"]["prompt"] == "a tin lighthouse, variant 2"
        assert frozen_c["generation"]["seed"] == 102
        assert frozen_c["occurrence"]["local_at"] == NOW + 8 * MIN, "the original occurrence time"
        assert after["subject"]["observed_event_id"] == event_id, "provenance by value, not by survival"

        # the new current world is a DIFFERENT snapshot
        fresh_event = conn.execute(
            "SELECT e.id FROM derived_event e WHERE e.kind = 'generation_session' ORDER BY e.id DESC LIMIT 1"
        ).fetchone()
        assert fresh_event is not None, "two of the three still sit within the gap"
        fresh = stories.snapshot_event(conn, fresh_event[0], NOW + 40 * HOUR)
        conn.commit()
        assert fresh.sha256 != frozen.sha256
        assert fresh.id != frozen.id
        told = stories.load_snapshot(conn, fresh.id)
        assert told["subject"]["member_hash"] != before["subject"]["member_hash"]
        assert told["subject"]["context_policy_version"] == before["subject"]["context_policy_version"] + 1, (
            "the fresh snapshot says which policy it was frozen under"
        )
        assert conn.execute("SELECT count(*) FROM story_snapshot").fetchone()[0] == 2
    finally:
        connect.close(conn)


def test_a_snapshot_is_insert_only(storied):
    """Immutability is enforced by the schema, not admired in a docstring."""
    client, _ = storied
    conn = connect.connect(client.app.state.db_path)
    try:
        made = stories.snapshot_event(conn, _event(conn), NOW + 30 * HOUR)
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE story_snapshot SET document_json = '{}' WHERE id = ?", (made.id,))
        conn.rollback()
        assert stories.verify(stories.load_snapshot(conn, made.id), made.sha256)
    finally:
        connect.close(conn)


def test_freezing_refuses_anything_but_a_current_complete_event(storied):
    """A stale hypothesis is not a subject. An unknown id, a run whose
    generation moved on, an incomplete interpretation: each refuses
    with nothing written."""
    client, _ = storied
    conn = connect.connect(client.app.state.db_path)
    try:
        with pytest.raises(LookupError, match="no event"):
            stories.snapshot_event(conn, 99_999, NOW + 30 * HOUR)
        conn.rollback()
        event_id = _event(conn)
        context.repopulated(conn)  # the population moved; the run is no longer current
        conn.commit()
        with pytest.raises(ValueError, match="not current"):
            stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
        conn.rollback()
        member = conn.execute("SELECT id FROM file WHERE name = 'gen_2.png'").fetchone()[0]
        context.stale(conn, member)  # a member's claim changes: the hypothesis itself is deleted
        conn.commit()
        with pytest.raises(LookupError, match="no event"):
            stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
        conn.rollback()
        assert conn.execute("SELECT count(*) FROM story_snapshot").fetchone()[0] == 0
    finally:
        connect.close(conn)


def test_freezing_cannot_race_the_library(storied, monkeypatch):
    """The WI-45 shape at the story seam: evidence collected over one
    generation must never land as a snapshot of another. A mutation in
    the handoff triggers ONE recollect; a persistent race refuses."""
    client, _ = storied
    conn = connect.connect(client.app.state.db_path)
    try:
        event_id = _event(conn)
        real = stories._document
        raced: list[int] = []

        def racing(conn_, subject, now):
            raced.append(1)
            document = real(conn_, subject, now)
            if len(raced) == 1:
                context._advance(conn_)
                conn_.commit()
                # the run is no longer current after that advance: prove it
                # by re-proving the world (the events job would) so the
                # retry finds a CURRENT subject again
                events.regroup(conn_, now)
            return document

        monkeypatch.setattr(stories, "_document", racing)
        with pytest.raises((ValueError, LookupError)):
            # the original event id died with its run in the recompute;
            # the honest outcome of a race that changed the subject is a
            # refusal, never a snapshot of a different world
            stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
        conn.rollback()
        assert conn.execute("SELECT count(*) FROM story_snapshot").fetchone()[0] == 0
        assert len(raced) >= 1
    finally:
        connect.close(conn)


def test_the_http_adapters_freeze_and_read_only(storied):
    """POST freezes synchronously and says whether it reused; GET reads
    history and writes nothing."""
    from db import resultset

    client, _ = storied
    conn = connect.connect(client.app.state.db_path)
    try:
        event_id = _event(conn)
    finally:
        connect.close(conn)
    made = client.post("/stories/snapshots", json={"event_id": event_id})
    assert made.status_code == 201, made.text
    body = made.json()
    assert body["reused"] is False
    assert len(body["sha256"]) == 64
    again = client.post("/stories/snapshots", json={"event_id": event_id})
    assert (again.status_code, again.json()["reused"], again.json()["id"]) == (200, True, body["id"])
    assert client.post("/stories/snapshots", json={"event_id": 99_999}).status_code == 404
    assert client.post("/stories/snapshots", json={}).status_code == 400, "a request without an event is a 400"

    conn = connect.connect(client.app.state.db_path)
    try:
        before = resultset.currency(conn)
    finally:
        connect.close(conn)
    read = client.get(f"/stories/snapshots/{body['id']}")
    assert read.status_code == 200
    assert read.json()["subject"]["member_hash"]
    assert client.get("/stories/snapshots/424242").status_code == 404
    conn = connect.connect(client.app.state.db_path)
    try:
        assert resultset.currency(conn) == before, "a GET of history wrote something"
        assert conn.execute("SELECT count(*) FROM story_snapshot").fetchone()[0] == 1
    finally:
        connect.close(conn)


def test_a_captioned_member_makes_the_story_say_what_a_model_saw():
    """v6: a member whose frozen annotations hold a caption yields a
    `seen` claim citing those annotations, with how many members were
    described and by which models; the renderer quotes one sentence.
    No caption, no claim -- the vocabulary is frozen, not the story."""
    from db import rendering

    members = [_member(i, text) for i, text in enumerate(LIGHTHOUSE[:3])]
    members[1]["annotations"] = [
        {"kind": "caption", "text": "a lighthouse on a rocky shore", "confidence": None, "model": ["blip/base", "82a3"]}
    ]
    members[2]["annotations"] = [{"kind": "tag", "text": "sea", "confidence": 0.9, "model": ["tagger", "1"]}]
    document, sha = _snapshot(members)
    plan = _planner().plan(document, sha)
    assert plan["v"] == 7
    assert planning.validate_story_plan(plan) == []
    seen = [claim for claim in plan["claims"] if claim["kind"] == "seen"]
    assert len(seen) == 1
    assert seen[0]["facts"] == {"members": 1, "models": ["blip/base"]}
    assert seen[0]["evidence_refs"] == ["member-002:annotations"], "a tag is not a caption"
    told = rendering.TemplateStoryRenderer("memory").render(document, plan, sha, planning.identity(plan)[1])
    assert rendering.violations(told, plan, document, sha, planning.identity(plan)[1]) == []
    words = " ".join(block["text"] for section in told["sections"] for block in section["blocks"])
    assert 'of one it said "a lighthouse on a rocky shore"' in words
    assert "described 1 file here" in words

    plain = _planner().plan(*_snapshot([_member(i, text) for i, text in enumerate(LIGHTHOUSE[:3])]))
    assert not [claim for claim in plain["claims"] if claim["kind"] == "seen"]
    assert planning.validate_story_plan({**plan, "v": 5}), "v5 does not know `seen`; the claim is a v6 claim"


def test_a_placed_member_makes_the_story_say_where():
    """v7: a member whose frozen place names where it happened yields a
    `located` claim citing those places; the renderer says it in words.
    No place, no claim."""
    from db import rendering

    members = [_member(i, text) for i, text in enumerate(LIGHTHOUSE[:3])]
    lisbon = {
        "uuid": "a" * 32,
        "chain": [
            {"uuid": "a" * 32, "kind": "city", "name": "Lisbon"},
            {"uuid": "b" * 32, "kind": "country", "name": "Portugal"},
        ],
    }
    members[0]["place"] = lisbon
    members[2]["place"] = lisbon
    document, sha = _snapshot(members)
    plan = _planner().plan(document, sha)
    assert plan["v"] == 7
    assert planning.validate_story_plan(plan) == []
    located = [claim for claim in plan["claims"] if claim["kind"] == "located"]
    assert len(located) == 1
    assert located[0]["facts"] == {"members": 2, "places": ["Lisbon"]}
    assert located[0]["evidence_refs"] == ["member-001:place", "member-003:place"]
    told = rendering.TemplateStoryRenderer("memory").render(document, plan, sha, planning.identity(plan)[1])
    assert rendering.violations(told, plan, document, sha, planning.identity(plan)[1]) == []
    words = " ".join(block["text"] for section in told["sections"] for block in section["blocks"])
    assert "In Lisbon (Portugal), by a person's word, for 2 files here." in words, "the frozen chain is spelled"
    assert planning.validate_story_plan({**plan, "v": 6}), "v6 does not know `located`"


def test_two_places_of_one_name_are_two_places_in_a_story():
    """Springfield, Illinois and Springfield, Oregon are two entities: the
    claim counts two places and spells them apart by what they are
    within."""
    from db import rendering

    members = [_member(i, text) for i, text in enumerate(LIGHTHOUSE[:2])]
    members[0]["place"] = {
        "uuid": "a" * 32,
        "chain": [
            {"uuid": "a" * 32, "kind": "city", "name": "Springfield"},
            {"uuid": "c" * 32, "kind": "region", "name": "Illinois"},
        ],
    }
    members[1]["place"] = {
        "uuid": "b" * 32,
        "chain": [
            {"uuid": "b" * 32, "kind": "city", "name": "Springfield"},
            {"uuid": "d" * 32, "kind": "region", "name": "Oregon"},
        ],
    }
    document, sha = _snapshot(members)
    plan = _planner().plan(document, sha)
    located = [claim for claim in plan["claims"] if claim["kind"] == "located"]
    assert located[0]["facts"] == {"members": 2, "places": ["Springfield (Illinois)", "Springfield (Oregon)"]}
    told = rendering.TemplateStoryRenderer("memory").render(document, plan, sha, planning.identity(plan)[1])
    words = " ".join(block["text"] for section in told["sections"] for block in section["blocks"])
    assert "Springfield (Illinois) and Springfield (Oregon)" in words
    assert "(Illinois) (Illinois)" not in words
