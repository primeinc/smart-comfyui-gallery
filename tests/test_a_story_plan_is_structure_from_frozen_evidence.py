"""A StoryPlan is structure from frozen evidence -- never prose, never a
database connection.

The planner receives only the snapshot document and a versioned
similarity engine; it cannot see the live library, so nothing that
happens to the library after the snapshot can change the plan. Every
reference resolves inside the snapshot; every conclusion is a Claim;
prompt identity alone never splits a phase; day-precision evidence
never becomes sub-day chronology -- the planner says `unsupported`
rather than inventing an order. Identity is content-addressed: the
same snapshot under the same policy is one plan, and a new policy
coexists with the old plan instead of overwriting it.
"""

from __future__ import annotations

import copy
import datetime
import sqlite3

import pytest
from litestar.testing import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import connect, context, ingest, planning, runner, stories
from sg_web.app import build_app

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


def test_wildcard_expansions_are_one_phase_and_a_new_subject_is_another():
    """The 55-file lesson: prompts that differ only by a wildcard
    expansion are one creative thread. Identity is never consulted --
    only similarity -- so the four lighthouse variants stay one phase
    and the diving helmet opens the next."""
    members = [_member(i, text, seed=100 + i) for i, text in enumerate(LIGHTHOUSE + HELMET)]
    document, sha = _snapshot(members)
    plan = _planner().plan(document, sha)
    assert plan["subject"]["sequenced"] is True
    assert [phase["member_refs"] for phase in plan["phases"]] == [
        ["member-001", "member-002", "member-003", "member-004"],
        ["member-005", "member-006"],
    ]
    first, second = plan["phases"]
    kinds = {claim["id"]: claim["kind"] for claim in plan["claims"]}
    assert {kinds[ref] for ref in first["claim_refs"]} == {"prompt_similarity", "seed_variation"}
    assert first["representative_refs"] == ["member-001"], "the medoid of the family, ties to the earliest"
    assert "parameter_change" not in {kinds[ref] for ref in second["claim_refs"]}, "nothing but the prompt changed"
    assert plan["unsupported"] == []


def test_day_precision_evidence_yields_families_and_an_unsupported_chronology():
    """Fifty-five files that only claim a DAY have no order among them.
    The planner still finds prompt families, lists members in event
    order, and says plainly that chronology is unsupported -- it never
    calls a family a 'phase' in time."""
    members = [_member(i, text, precision="day") for i, text in enumerate(LIGHTHOUSE + HELMET)]
    document, sha = _snapshot(members)
    plan = _planner().plan(document, sha)
    assert plan["subject"]["sequenced"] is False
    assert [one["kind"] for one in plan["unsupported"]] == ["chronology"]
    assert [phase["member_refs"] for phase in plan["phases"]] == [
        ["member-001", "member-002", "member-003", "member-004"],
        ["member-005", "member-006"],
    ]
    assert all(phase["label_hint"].startswith("Prompt family") for phase in plan["phases"])
    assert "prompt families" in plan["subject"]["label_hint"]

    # the same families, interleaved in event order, are still the same
    # families: without chronology, adjacency means nothing
    shuffled = [LIGHTHOUSE[0], HELMET[0], LIGHTHOUSE[1], HELMET[1], LIGHTHOUSE[2], LIGHTHOUSE[3]]
    document, sha = _snapshot([_member(i, text, precision="day") for i, text in enumerate(shuffled)])
    plan = _planner().plan(document, sha)
    assert sorted(len(phase["member_refs"]) for phase in plan["phases"]) == [2, 4]


def test_artifact_and_parameter_changes_are_claims_about_a_boundary_not_boundaries():
    """A LoRA swap inside one prompt family does NOT split the phase; a
    LoRA swap across a prompt boundary is a claim on the new phase, with
    evidence pointing at both sides."""
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
    claims = {claim["id"]: claim for claim in plan["claims"]}
    second = [claims[ref] for ref in plan["phases"][1]["claim_refs"]]
    by_kind = {claim["kind"]: claim for claim in second}
    assert by_kind["artifact_change"]["facts"] == {"added": [], "removed": [lora_a]}
    assert by_kind["parameter_change"]["facts"]["changed"]["sampler"] == {"from": ["Euler a"], "to": ["DPM++ 2M"]}
    assert by_kind["parameter_change"]["facts"]["changed"]["steps"] == {"from": ["20"], "to": ["30"]}
    assert any(ref.startswith("member-001:") for ref in by_kind["artifact_change"]["evidence_refs"])
    assert any(ref.startswith("member-003:") for ref in by_kind["artifact_change"]["evidence_refs"])
    assert plan["phases"][1]["label_hint"].endswith("new artifacts")


def test_every_reference_resolves_inside_the_snapshot_and_nothing_else():
    """A plan is closed over its evidence: every member_ref,
    representative_ref and evidence_ref names a member path that exists
    in the snapshot, every claim_ref names a claim. The resolver proves
    its own teeth on a tampered plan."""
    members = [_member(i, text) for i, text in enumerate(LIGHTHOUSE + HELMET)]
    document, sha = _snapshot(members)
    plan = _planner().plan(document, sha)
    assert planning.unresolved(plan, document) == []
    tampered = copy.deepcopy(plan)
    tampered["phases"][0]["member_refs"].append("member-099")
    tampered["claims"][0]["evidence_refs"].append("member-001:generation.nonsense")
    tampered["phases"][0]["claim_refs"].append("claim-999")
    assert planning.unresolved(tampered, document) == ["member-099", "claim-999", "member-001:generation.nonsense"]


def test_missing_evidence_is_declared_never_explained():
    """A member with no frozen prompt is named in `unsupported`; the
    planner does not guess a family for it from anything else."""
    members = [_member(0, LIGHTHOUSE[0]), _member(1, LIGHTHOUSE[1]), _member(2, "")]
    document, sha = _snapshot(members)
    plan = _planner().plan(document, sha)
    told = next(one for one in plan["unsupported"] if one["kind"] == "prompt_evidence")
    assert told["member_refs"] == ["member-003"]
    assert [phase["member_refs"] for phase in plan["phases"]] == [["member-001", "member-002"], ["member-003"]]
    assert plan["phases"][1]["claim_refs"] == [], "a singleton with no prompt supports no claim"


def test_the_same_snapshot_under_the_same_policy_is_one_plan_and_policies_coexist():
    """Determinism and identity: planning twice is byte-identical; a
    different threshold is a different plan with its own identity, and
    the engine's vectors are stable across calls."""
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
    assert strict["planner"]["settings"] == {"phase_threshold": 0.99}
    assert one["snapshot_sha256"] == sha
    assert one["planner"]["similarity"] == {"name": "lexical-bow", "version": "1"}


def test_the_threshold_is_pinned_against_the_engine():
    """The lexical oracle's numbers are known: two ten-token prompts
    differing in one word cosine to 11/12; lighthouse vs helmet share
    'a' and little else. The default threshold sits between."""
    cosine = planning.pairwise_cosine(
        planning.LexicalPromptSimilarity().embed([LIGHTHOUSE[0], LIGHTHOUSE[1], HELMET[0]])
    )
    assert cosine[0][1] == pytest.approx(11 / 12, abs=1e-3)
    assert cosine[0][2] < 0.5 < cosine[0][1]
    assert planning.GenerationHistoryPlanner.defaults["phase_threshold"] == 0.5


def test_the_planner_owns_no_connection_no_sql_and_no_model():
    import inspect
    import pathlib

    source = (pathlib.Path(__file__).resolve().parent.parent / "db" / "planning.py").read_text(encoding="utf-8")
    head = source.split("def plan_snapshot(", 1)[0]
    for banned in ("execute(", "FROM ", "JOIN ", "sqlite3", "(conn", "conn,", "conn)"):
        assert banned not in head, f"the planner reached for the database: {banned!r}"
    for banned in ("import openai", "anthropic", "import requests", "import httpx", "torch"):
        assert banned not in source
    assert "conn" not in inspect.signature(planning.GenerationHistoryPlanner.plan).parameters


# --- persistence, against a real frozen snapshot -----------------------------


def _library(tmp) -> tuple:
    root = tmp / "lib"
    root.mkdir()
    for i, text in enumerate(LIGHTHOUSE[:3]):
        info = PngInfo()
        info.add_text(
            "parameters",
            f"{text}\nNegative prompt: blur\n"
            f"Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: {100 + i}, Size: 512x512, Model: alpha",
        )
        Image.new("RGB", (12, 12), (40 + i * 30, 90, 140)).save(root / f"gen_{i}.png", pnginfo=info)
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
def frozen(tmp_path):
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
                [(names[f"gen_{i}.png"], _spelled(NOW + i * 4 * MIN)) for i in range(3)],
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
        yield client, made


def test_a_persisted_plan_is_immune_to_everything_that_happens_after_the_snapshot(frozen):
    """Plan, then mutate and regroup the live library, then plan the
    SAME snapshot again: the row is reused, byte for byte -- the planner
    never saw the library, so the library cannot reach it."""
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
        assert planning.unresolved(told, stories.load_snapshot(conn, snap.id)) == []
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


def test_the_http_adapters_plan_and_read_only(frozen):
    from db import resultset

    client, snap = frozen
    made = client.post("/stories/plans", json={"snapshot_id": snap.id, "similarity": "lexical"})
    assert made.status_code == 201, made.text
    body = made.json()
    again = client.post("/stories/plans", json={"snapshot_id": snap.id, "similarity": "lexical"})
    assert (again.status_code, again.json()["id"], again.json()["reused"]) == (200, body["id"], True)
    assert client.post("/stories/plans", json={"snapshot_id": 99_999, "similarity": "lexical"}).status_code == 404
    assert client.post("/stories/plans", json={"snapshot_id": snap.id, "similarity": "astrology"}).status_code == 400
    assert client.post("/stories/plans", json={"snapshot_id": snap.id, "planner": "vibes"}).status_code == 400
    assert client.post("/stories/plans", json={}).status_code == 400
    conn = connect.connect(client.app.state.db_path)
    try:
        before = resultset.currency(conn)
    finally:
        connect.close(conn)
    read = client.get(f"/stories/plans/{body['id']}")
    assert read.status_code == 200
    assert read.json()["planner"]["kind"] == "generation_history"
    assert read.json()["phases"][0]["member_refs"] == ["member-001", "member-002", "member-003"]
    assert client.get("/stories/plans/424242").status_code == 404
    conn = connect.connect(client.app.state.db_path)
    try:
        assert resultset.currency(conn) == before
    finally:
        connect.close(conn)
