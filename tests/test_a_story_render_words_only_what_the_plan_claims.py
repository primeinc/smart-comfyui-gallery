"""A StoryRender words what the plan claims and nothing else.

The deterministic narrator receives two immutable, verified documents
and decides nothing: every sentence maps to a Claim or an unsupported
entry, every hero and member resolves in the snapshot, unsupported
chronology stays unsupported -- and these are enforced by a validator,
not a convention. The document is structure, never HTML; the page lays
it out strictly and escapes frozen evidence. Identities are the
planner's: request and document, insert-only, re-verified on read.
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


def _member(ordinal, prompt, *, precision="second", artifacts=(), seed=100, name=None):
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
                    "name": f"lora-{uuid[:4]}",
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


def _snapshot(members, *, local=True) -> tuple[dict, str]:
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
            "time": {"local": [JULY_18, JULY_18 + HOUR] if local else None, "instant": None},
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


def test_the_golden_render_for_a_sequenced_session():
    """The exact document the deterministic narrator produces for a
    frozen world -- pinned whole. Every sentence below traces to a
    Claim; the renderer invented none of it."""
    snapshot, snapshot_sha, plan, plan_sha = _world()
    render = rendering.TemplateStoryRenderer("memory").render(snapshot, plan, snapshot_sha, plan_sha)
    assert rendering.violations(render, plan, snapshot, snapshot_sha, plan_sha) == []
    assert render == {
        "v": 1,
        "snapshot_sha256": snapshot_sha,
        "plan_sha256": plan_sha,
        "renderer": {"kind": "template", "version": 1, "profile": "memory", "locale": "en", "wording_policy": 1},
        "title": "5 Qwen Image Edit images from July 18, 2026",
        "dek": "2 phases across 5 generated images",
        "summary": "These 5 generated images were generated on July 18, 2026 and fall into 2 phases.",
        "sections": [
            {
                "id": "section-001",
                "title": "Phase 1",
                "paragraphs": ["3 images.", "The 3 images in this phase share closely related prompt wording."],
                "hero_refs": ["member-001"],
                "member_refs": ["member-001", "member-002", "member-003"],
                "claim_refs": ["claim-001"],
            },
            {
                "id": "section-002",
                "title": "Phase 2 · new artifacts",
                "paragraphs": [
                    "2 images.",
                    "The prompt wording changes here compared with the previous phase.",
                    "The 2 images in this phase share closely related prompt wording.",
                    "Compared with the previous phase, lora-bbbb appears in this group; lora-aaaa is not used.",
                ],
                "hero_refs": ["member-004"],
                "member_refs": ["member-004", "member-005"],
                "claim_refs": ["claim-003", "claim-004", "claim-006"],
            },
        ],
        "notes": [],
    }
    technical = rendering.TemplateStoryRenderer("technical").render(snapshot, plan, snapshot_sha, plan_sha)
    assert "2 distinct seeds were used." in technical["sections"][1]["paragraphs"]
    assert any("Minimum pairwise prompt similarity" in p for p in technical["sections"][0]["paragraphs"])
    assert set(technical["sections"][1]["claim_refs"]) > set(render["sections"][1]["claim_refs"]), (
        "a profile surfaces more or fewer claims; it never changes which claims exist"
    )
    compact = rendering.TemplateStoryRenderer("compact").render(snapshot, plan, snapshot_sha, plan_sha)
    assert compact["sections"] == []
    assert compact["title"] == render["title"]


def test_unsupported_chronology_is_said_and_never_sequenced():
    """Day-precision evidence: the plan declares chronology unsupported;
    the render MUST carry the note and MUST NOT sequence. The rule has
    teeth: a render that slips in 'then' is a violation, and a render
    that drops the note is a violation."""
    snapshot, snapshot_sha, plan, plan_sha = _planned(
        [_member(i, text, precision="day") for i, text in enumerate(LIGHTHOUSE + HELMET)]
    )
    assert [one["kind"] for one in plan["unsupported"]] == ["chronology"]
    render = rendering.TemplateStoryRenderer("memory").render(snapshot, plan, snapshot_sha, plan_sha)
    assert rendering.violations(render, plan, snapshot, snapshot_sha, plan_sha) == []
    assert render["notes"] == [
        {
            "kind": "chronology",
            "text": "Available evidence does not establish the order of these images within July 18, 2026.",
        }
    ]
    assert all(section["title"].startswith("Prompt family") for section in render["sections"])
    assert "prompt families" in render["dek"]

    sequenced = copy.deepcopy(render)
    sequenced["sections"][1]["paragraphs"].append("Then the helmet appeared.")
    told = rendering.violations(sequenced, plan, snapshot, snapshot_sha, plan_sha)
    assert any("sequences: 'Then'" in why for why in told)

    silent = copy.deepcopy(render)
    silent["notes"] = []
    assert "the plan declares chronology unsupported and the render does not say so" in rendering.violations(
        silent, plan, snapshot, snapshot_sha, plan_sha
    )


def test_a_render_cites_only_plan_claims_and_names_only_snapshot_members():
    snapshot, snapshot_sha, plan, plan_sha = _world()
    render = rendering.TemplateStoryRenderer("memory").render(snapshot, plan, snapshot_sha, plan_sha)

    invented = copy.deepcopy(render)
    invented["sections"][0]["claim_refs"].append("claim-099")
    assert "section-001 cites unknown claim claim-099" in rendering.violations(
        invented, plan, snapshot, snapshot_sha, plan_sha
    )
    stranger = copy.deepcopy(render)
    stranger["sections"][0]["hero_refs"] = ["member-077"]
    told = rendering.violations(stranger, plan, snapshot, snapshot_sha, plan_sha)
    assert "section-001 names unknown member member-077" in told
    elsewhere = copy.deepcopy(render)
    elsewhere["sections"][0]["hero_refs"] = ["member-004"]
    assert "section-001 hero member-004 is not one of its members" in rendering.violations(
        elsewhere, plan, snapshot, snapshot_sha, plan_sha
    )
    other = copy.deepcopy(render)
    other["plan_sha256"] = "f" * 64
    assert "the render names a different plan" in rendering.violations(other, plan, snapshot, snapshot_sha, plan_sha)


def test_the_grammar_is_exact_and_the_wording_registry_is_closed():
    snapshot, snapshot_sha, plan, plan_sha = _world()
    render = rendering.TemplateStoryRenderer("memory").render(snapshot, plan, snapshot_sha, plan_sha)
    assert rendering.validate_story_render_v1(render) == []
    for mutate in (
        lambda r: r.__setitem__("html", "<b>no</b>"),
        lambda r: r["sections"][0].__setitem__("mood", "wistful"),
        lambda r: r["renderer"].__setitem__("kind", "llm"),
        lambda r: r["renderer"].__setitem__("profile", "poetic"),
        lambda r: r["sections"][0].__setitem__("paragraphs", []),
        lambda r: r["sections"][0].__setitem__("member_refs", []),
        lambda r: r["notes"].append({"kind": "weather", "text": "rain"}),
        lambda r: r.__setitem__("title", ""),
    ):
        bent = copy.deepcopy(render)
        mutate(bent)
        assert rendering.validate_story_render_v1(bent), "the grammar accepted a malformed render"
    for garbage in (None, [], "story", {"v": 1}):
        assert rendering.validate_story_render_v1(garbage)

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


def test_formatting_is_decided_once():
    assert formatting.day_label(JULY_18) == "July 18, 2026"
    assert formatting.day_label(None) is None
    assert formatting.count(1, "image") == "1 image"
    assert formatting.count(55, "image") == "55 images"
    assert formatting.percent(0.92314) == "92%"
    assert formatting.join_names(["foo"]) == "foo"
    assert formatting.join_names(["foo", "bar"]) == "foo and bar"
    assert formatting.join_names(["foo", "bar", "baz"]) == "foo, bar and baz"


def test_identities_same_request_one_document_policy_coexists(monkeypatch):
    snapshot, snapshot_sha, plan, plan_sha = _world()
    one = rendering.TemplateStoryRenderer("memory").render(snapshot, plan, snapshot_sha, plan_sha)
    two = rendering.TemplateStoryRenderer("memory").render(snapshot, plan, snapshot_sha, plan_sha)
    assert rendering.identity(one) == rendering.identity(two)
    other = rendering.TemplateStoryRenderer("technical").render(snapshot, plan, snapshot_sha, plan_sha)
    assert rendering.identity(other)[1] != rendering.identity(one)[1]
    request = rendering.request_identity(plan_sha, snapshot_sha, "template", 1, "memory", "en", 1)
    assert request != rendering.request_identity(plan_sha, snapshot_sha, "template", 1, "memory", "en", 2), (
        "a wording-policy bump is a different request"
    )
    assert request != rendering.request_identity(plan_sha, snapshot_sha, "template", 1, "technical", "en", 1)
    monkeypatch.setattr(rendering, "FORMAT_VERSION", 2)
    assert request != rendering.request_identity(plan_sha, snapshot_sha, "template", 1, "memory", "en", 1)


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
            made = planning.plan_snapshot(
                conn, snap.id, planning.GenerationHistoryPlanner(planning.LexicalPromptSimilarity()), NOW + 31 * HOUR
            )
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
    finally:
        connect.close(conn)


def test_the_page_lays_out_the_verified_render_and_escapes_evidence(planned):
    """The HTML Adapter: title, sections, the note, hero links -- and a
    file name that is a script tag stays text. The page carries every
    paragraph the JSON carries: deleting story.html would lose nothing
    the JSON does not already say."""
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
        for paragraph in section["paragraphs"]:
            assert paragraph in html, "the page drops nothing the JSON says"
    assert "gen_0 & 'friends'.png" not in html, "frozen evidence is text, never markup"
    assert "gen_0 &amp; &#39;friends&#39;.png" in html
    assert html.count('href="/m/') >= 1, "a hero links through address resolution"
    assert client.get("/stories/renders/424242").status_code == 404
    conn = connect.connect(client.app.state.db_path)
    try:
        assert resultset.currency(conn) == before, "a GET of a story wrote something"
    finally:
        connect.close(conn)
