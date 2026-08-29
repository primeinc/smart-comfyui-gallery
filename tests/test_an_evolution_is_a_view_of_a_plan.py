"""The Generation Evolution Explorer is a read-only view of a StoryPlan.

It decides nothing: phases, families and chronology come from the plan,
facts from the snapshot; it MEASURES over frozen facts with vectors
looked up by frozen text hash under one (space, policy) and by frozen
file bytes -- never today's relation, never a replacement file -- and
says why when it cannot. No writes, no model loads, O(n) transitions,
deterministic JSON.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import sys
import types

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import connect, evolution, ingest, planning, prompts, resultset, runner, settings, stories
from tests.staging import HOUR, NOW, Stage, staged
from vision import semantic
from vision.faiss_index import SpaceSpec

MIN = 60.0


def _spelled(moment: float) -> str:
    return datetime.datetime.fromtimestamp(moment, datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")


def _unit(text: str, dims: int = 4) -> np.ndarray:
    raw = np.frombuffer(hashlib.sha256(text.encode("utf-8")).digest()[: dims * 4], dtype=np.uint32).astype(np.float32)
    return (raw / np.linalg.norm(raw)).astype(np.float32)


class _Encoder:
    provider = "fake"

    def __init__(self, model, checkpoint):
        self.model, self.checkpoint, self.dimensions = model, checkpoint, 4
        self.calls: list[str] = []

    def space(self):
        return _fake.space(self.model, self.checkpoint, self.dimensions)

    def encode_query(self, text):
        self.calls.append(text)
        return _unit(text)

    def encode_media(self, media):
        self.calls.append(f"media:{media.path}")
        return _unit(media.path)


_ENCODERS: dict[tuple, _Encoder] = {}


class _FakeProvider(types.ModuleType):
    """A semantic provider module, as vision/semantic looks one up."""

    @staticmethod
    def parse(reference):
        return tuple(reference.split("/", 1))

    @staticmethod
    def immutable(checkpoint):
        return True

    @staticmethod
    def query_policy(model, checkpoint):
        return {"provider": "fake", "model": model, "checkpoint": checkpoint}

    @staticmethod
    def space(model, checkpoint, dims):
        return SpaceSpec(
            key=f"semantic.fake.{model}.{checkpoint}",
            representation="float32",
            dimensions=int(dims),
            metric="cosine",
            producer=f"fake:{model}",
            producer_version=checkpoint,
            preprocess="fake.media",
            preprocess_version="v1",
        )

    @staticmethod
    def encoder(models_dir, model, checkpoint, *, offline=False):
        return _ENCODERS.setdefault((model, checkpoint), _Encoder(model, checkpoint))


_fake = _FakeProvider("tests._fake_semantic_evolution")

WRITTEN = "a __material__ lighthouse on a cliff"
PROMPTS = [
    "a tin lighthouse on a cliff",
    "a brass lighthouse on a cliff",
    "a brass diving helmet in a museum case <segment:face> fixed face",
]


def _swarm(path, prompt, *, original=None, seed=1, lora=None):
    payload = {
        "sui_image_params": {
            "prompt": prompt,
            "negativeprompt": "blur",
            "model": "flux-dev",
            "seed": seed,
            "steps": 20,
            "cfgscale": 7,
            "width": 512,
            "height": 512,
            **({"loras": [lora]} if lora else {}),
        },
        "sui_extra_data": {"original_prompt": original} if original is not None else {},
    }
    info = PngInfo()
    info.add_text("parameters", json.dumps(payload))
    Image.new("RGB", (12, 12), (40 + seed * 20, 90, 140)).save(path, pnginfo=info)


def _drain(client) -> None:
    conn = connect.connect(client.app.state.db_path)
    try:
        while runner.run_next(conn, "test-worker", NOW + 24 * HOUR) is not None:
            conn.commit()
        conn.commit()
    finally:
        connect.close(conn)


def _library(root: pathlib.Path) -> None:
    """Three Swarm stills in one sequenced session (two lighthouses, then
    a helmet with a LoRA)."""
    _swarm(root / "gen_0.png", PROMPTS[0], original=WRITTEN, seed=1)
    _swarm(root / "gen_1.png", PROMPTS[1], original=WRITTEN, seed=2)
    _swarm(root / "gen_2.png", PROMPTS[2], seed=3, lora="detail")


def _planned(stage: Stage) -> None:
    """Embedded, frozen and planned -- once."""
    client, root = stage.client, stage.root
    conn = stage.conn()
    try:
        names = dict(conn.execute("SELECT name, id FROM file").fetchall())
        for name, file_id in names.items():
            ingest.one(conn, file_id, root / name, NOW)
        conn.executemany(
            "INSERT OR REPLACE INTO file_param(file_id, source, key, value_text) VALUES(?, 'generation', 'date', ?)",
            [(names[f"gen_{i}.png"], _spelled(NOW + i * 4 * MIN)) for i in range(3)],
        )
        settings.put(conn, "semantic_model", "fake:toy/v1")
        conn.commit()
    finally:
        connect.close(conn)
    client.post("/jobs/embed")
    client.post("/jobs/embed_prompts")
    client.post("/jobs/context")
    client.post("/jobs/events")
    _drain(client)
    conn = stage.conn()
    try:
        event_id = conn.execute("SELECT id FROM derived_event WHERE kind = 'generation_session'").fetchone()[0]
        snap = stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
        planner = planning.GenerationHistoryPlanner(planning.LexicalPromptSimilarity())
        made = planning.plan_snapshot(conn, snap.id, planner, NOW + 31 * HOUR)
        conn.commit()
    finally:
        connect.close(conn)
    stage.held.update(names=names, snap=snap, made=made)


@pytest.fixture(scope="module")
def _stage(tmp_path_factory):
    sys.modules["tests._fake_semantic_evolution"] = _fake
    with pytest.MonkeyPatch.context() as held:
        held.setitem(semantic.PROVIDERS, "fake", "tests._fake_semantic_evolution")
        _ENCODERS.clear()
        with staged(tmp_path_factory, "evolution", _library, _planned) as stage:
            yield stage
    _ENCODERS.clear()


@pytest.fixture
def planned(_stage, monkeypatch):
    monkeypatch.setitem(semantic.PROVIDERS, "fake", "tests._fake_semantic_evolution")
    _stage.restore()
    _ENCODERS[("toy", "v1")].calls.clear()
    return _stage.client, _stage.root, _stage.held["names"], _stage.held["snap"], _stage.held["made"]


def test_the_view_is_the_plans_structure_measured_without_writing_or_loading(planned):
    client, _root, _names, snap, made = planned
    conn = connect.connect(client.app.state.db_path)
    try:
        before = resultset.currency(conn)
    finally:
        connect.close(conn)
    told = client.get(f"/stories/plans/{made.id}/evolution", headers={"accept": "application/json"})
    assert told.status_code == 200, told.text
    view = told.json()
    assert view["v"] == 1
    assert view["plan"] == {
        "id": made.id,
        "sha256": made.sha256,
        "format": planning.FORMAT_VERSION,
        "sequenced": True,
        "unsupported": [],
        "label": view["plan"]["label"],
    }
    assert view["snapshot"]["sha256"] == snap.sha256
    assert view["semantic"]["space"] == "semantic.fake.toy.v1"
    assert view["semantic"]["prompt_policy_hash"] == semantic.policy_hash("fake", "toy", "v1")
    assert view["semantic"]["unavailable"] is None
    # phases are the PLAN's, exactly
    conn = connect.connect(client.app.state.db_path)
    try:
        plan = planning.load_plan(conn, made.id)
        after = resultset.currency(conn)
    finally:
        connect.close(conn)
    assert [p["id"] for p in view["phases"]] == [p["id"] for p in plan["phases"]]
    assert [p["member_refs"] for p in view["phases"]] == [p["member_refs"] for p in plan["phases"]]
    assert [m["phase_ref"] for m in view["members"]] == ["phase-001", "phase-001", "phase-002"]
    # every member: frozen media identity, roles, facts, metrics in one space
    first = view["members"][0]
    assert first["media"]["name"] == "gen_0.png"
    # Content-addressed, because these files are hashed. The `/thumb/<slug>`
    # route is the fallback for bytes ingest has not reached, and pointing
    # at it for a hashed file costs a slug lookup per picture.
    assert first["media"]["thumbnail"].startswith("/thumbs/"), first["media"]["thumbnail"]
    assert first["media"]["thumbnail"].endswith(".webp")
    assert first["prompt"]["effective"]["main"] == PROMPTS[0]
    assert first["prompt"]["original"]["text"] == WRITTEN
    assert first["prompt"]["effective"]["prompt_id"] is not None
    assert first["generation"]["model"] == "flux-dev"
    assert first["metrics"]["original_effective_cosine"] == pytest.approx(
        float(_unit(WRITTEN) @ _unit(PROMPTS[0])), abs=1e-3
    )
    assert first["metrics"]["text_image_cosine"] is not None
    third = view["members"][2]
    assert third["prompt"]["effective"]["main"] == "a brass diving helmet in a museum case", "the MAIN section"
    assert third["prompt"]["original"] is None
    assert third["metrics"]["original_effective_cosine"] is None
    assert third["metrics"]["original_effective_cosine_unavailable"] == "no frozen original prompt"
    assert third["generation"]["loras"] == ["detail"]
    # transitions: O(n) consecutive pairs, boundary from the plan, exact deltas
    assert [(t["before"], t["after"], t["phase_boundary"]) for t in view["transitions"]] == [
        ("member-001", "member-002", False),
        ("member-002", "member-003", True),
    ]
    assert view["transitions"][0]["prompt_cosine"] == pytest.approx(
        float(_unit(PROMPTS[0]) @ _unit(PROMPTS[1])), abs=1e-3
    )
    # a parameter is in `parameters` because it changed; an unchanged fact
    # is absent from the list rather than present and null
    assert view["transitions"][0]["changes"] == {
        "parameters": [{"name": "seed", "before": 1, "after": 2}],
        "loras_added": [],
        "loras_removed": [],
        "lora_uuids_added": [],
        "lora_uuids_removed": [],
    }
    assert view["transitions"][1]["changes"]["loras_added"] == ["detail"]
    changed = {one["name"]: one for one in view["transitions"][1]["changes"]["parameters"]}
    assert changed["seed"] == {"name": "seed", "before": 2, "after": 3}
    assert "model" not in changed, "an unchanged fact is not a change"
    # zero writes, zero model work
    assert after == before, "a GET wrote something"
    assert _ENCODERS[("toy", "v1")].calls == [], "a GET loaded or ran a model"
    # deterministic
    assert client.get(f"/stories/plans/{made.id}/evolution", headers={"accept": "application/json"}).json() == view
    assert client.get("/stories/plans/424242/evolution").status_code == 404
    assert client.get(f"/stories/plans/{made.id}/evolution", params={"space": "openclip"}).status_code == 400


def test_the_page_draws_the_view_and_only_the_view(planned):
    client, _root, _names, _snap, made = planned
    page = client.get(f"/stories/plans/{made.id}/evolution", headers={"accept": "text/html"})
    assert page.status_code == 200
    html = page.text
    assert "/static/build/app.js" in html
    assert 'data-tab="sequence"' in html
    assert 'data-tab="drift"' in html
    assert f'data-plan="{made.id}"' in html, "the shell names the plan; the explorer asks the route for the document"
    assert "application/json" not in html, "the document is not serialized into the page to be parsed back out"


def _synthetic(n: int, precision: str) -> tuple[dict, str]:
    """A frozen generation session of n Swarm stills, seven prompt
    variants cycling, at the given time precision."""
    members = []
    for i in range(n):
        text = f"a lighthouse number {i % 7}"
        members.append(
            {
                "ordinal": i,
                "file_uuid": f"{i:032x}",
                "content_sha256": f"{i:064x}",
                "media_kind": "image",
                "name": f"gen_{i}.png",
                "occurrence": {
                    "kind": "generation",
                    "basis": "embedded",
                    "local_at": NOW + i * 60,
                    "instant_at": None,
                    "tz_offset_min": None,
                    "precision": precision,
                    "certainty": 0.6,
                },
                "generation": {
                    "tool": "SwarmUI",
                    "detection": "marker",
                    "seed": i,
                    "steps": 20,
                    "cfg": 7.0,
                    "denoise": None,
                    "clip_skip": None,
                    "sampler": "Euler a",
                    "scheduler": None,
                    "width": 512,
                    "height": 512,
                    "prompt": text,
                    "negative_prompt": None,
                    "prompts": [
                        {"role": "effective", "uuid": f"{i:032x}", "text": text, "text_hash": prompts.text_hash(text)}
                    ],
                    "workflow_uuid": None,
                    "artifacts": [],
                    "params": {},
                },
                "capture": None,
                "people": None,
                "place": None,
                "lineage": None,
                "annotations": None,
            }
        )
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
            "time": {"local": [NOW, NOW + HOUR], "instant": None},
            "place": None,
            "confidence": None,
            "observed_event_id": 1,
        },
        "members": members,
    }
    return document, stories._identity(document)[1]


def _store(conn, document: dict, sha: str) -> int:
    """The synthetic snapshot and its lexical plan, as rows; returns the plan id."""
    plan = planning.GenerationHistoryPlanner(planning.LexicalPromptSimilarity()).plan(document, sha)
    snapshot_id = conn.execute(
        "INSERT INTO story_snapshot(format_version, source_kind, event_kind, grouper, context_generation,"
        " context_policy_version, member_hash, document_json, document_sha256, created_at)"
        " VALUES(1, 'event', 'generation_session', 'generation_session', 7, 4, 'deadbeef', ?, ?, ?)",
        (stories.canonical(document), sha, NOW),
    ).lastrowid
    plan["planned_at"] = NOW
    return int(
        conn.execute(
            "INSERT INTO story_plan(snapshot_id, format_version, planner, planner_version, similarity,"
            " similarity_version, settings_hash, request_sha256, document_json, document_sha256, created_at)"
            " VALUES(?, ?, 'generation_history', ?, 'lexical-bow', '1', 'x', ?, ?, ?, ?)",
            (
                snapshot_id,
                planning.FORMAT_VERSION,
                planning.GenerationHistoryPlanner.version,
                hashlib.sha256(sha.encode()).hexdigest(),
                planning.canonical(plan),
                planning.identity(plan)[1],
                NOW,
            ),
        ).lastrowid
        or 0
    )


def test_unsequenced_plans_have_no_transitions_and_lead_with_families(planned):
    """Day-precision evidence: the plan declares chronology unsupported;
    the view has no transitions and the page offers no Sequence or
    Drift tab -- families lead, with no arrow anywhere."""
    client, _root, _names, _snap, _made = planned
    document, sha = _synthetic(6, "day")
    conn = connect.connect(client.app.state.db_path)
    try:
        plan_id = _store(conn, document, sha)
        conn.commit()
    finally:
        connect.close(conn)
    view = client.get(f"/stories/plans/{plan_id}/evolution", headers={"accept": "application/json"}).json()
    assert view["plan"]["sequenced"] is False
    assert view["transitions"] == []
    assert [one["kind"] for one in view["plan"]["unsupported"]] == ["chronology"]
    assert [p["label"] for p in view["phases"]] == ["Prompt family 1"], "the plan's families, verbatim"
    html = client.get(f"/stories/plans/{plan_id}/evolution", headers={"accept": "text/html"}).text
    assert 'data-tab="sequence"' not in html
    assert 'data-tab="drift"' not in html
    assert 'data-tab="families" class="on"' in html


def test_no_space_yields_reasons_not_numbers(planned):
    client, _root, _names, _snap, made = planned
    conn = connect.connect(client.app.state.db_path)
    try:
        settings.put(conn, "semantic_model", "fake:other/v9")
        conn.commit()
    finally:
        connect.close(conn)
    view = client.get(f"/stories/plans/{made.id}/evolution", headers={"accept": "application/json"}).json()
    assert view["semantic"]["space_id"] is None
    assert "no vectors recorded" in view["semantic"]["unavailable"]
    assert all(m["metrics"]["text_image_cosine"] is None for m in view["members"])
    assert all("no vectors recorded" in t["prompt_cosine_unavailable"] for t in view["transitions"])
    assert view["transitions"][1]["changes"]["loras_added"] == ["detail"], "facts need no vectors"


@pytest.mark.slow
def test_a_thousand_members_are_o_n_work():
    """No matrix: the transition count is n-1 and the module asks the
    database a bounded number of statements however large the session.

    Slow by construction and marked as such: a thousand synthetic members
    are built, stored and read back. What it proves is a bound, so it is
    worth a second in `just test-slow`; it is not worth a second on every
    `just test`, and the marker's own definition is a call phase over one
    second."""

    from db import build

    n = 1000
    document, sha = _synthetic(n, "second")
    conn = connect.memory()
    try:
        conn.executescript(build.schema_sql())
        plan_id = _store(conn, document, sha)
        statements: list[str] = []
        conn.set_trace_callback(statements.append)
        view = evolution.load(conn, plan_id, models_dir="unused")
        conn.set_trace_callback(None)
        assert len(view["members"]) == n
        assert len(view["transitions"]) == n - 1
        assert len(view["phases"]) == len(planning.load_plan(conn, plan_id)["phases"])
        selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
        assert len(selects) < 40, f"{len(selects)} statements for {n} members: not bounded"
    finally:
        connect.close(conn)


def test_the_view_reads_the_mains_the_snapshot_froze_not_the_running_parser(monkeypatch):
    """An old plan under a newer prompt parser: the Explorer's prompt
    inputs are the MAIN texts the snapshot froze, so the metrics do not
    move when `prompt_sections.VERSION` does. The running parser is
    never consulted for a member that carries a frozen main."""

    from db import build, prompt_sections

    document, sha = _synthetic(4, "second")
    for member in document["members"]:
        held = member["generation"]["prompts"][0]
        held["main"] = f"frozen main {member['ordinal']}"
        held["main_hash"] = prompts.text_hash(held["main"])
        held["grammar"] = "swarm"
        held["parser"] = prompt_sections.VERSION
    sha = stories._identity(document)[1]
    conn = connect.memory()
    try:
        conn.executescript(build.schema_sql())
        plan_id = _store(conn, document, sha)
        before = evolution.load(conn, plan_id, models_dir="unused")
        assert [m["prompt"]["effective"]["main"] for m in before["members"]] == [f"frozen main {i}" for i in range(4)]

        def never(*_a, **_k):
            raise AssertionError("the running parser was consulted for a frozen main")

        # `monkeypatch` rather than a try/finally: it restores on a
        # failing assertion too, and rebinding a module's `def` and its
        # VERSION literal is not an assignment a type checker will take.
        monkeypatch.setattr(prompt_sections, "VERSION", prompt_sections.VERSION + 1)
        monkeypatch.setattr(prompt_sections, "main", never)
        after = evolution.load(conn, plan_id, models_dir="unused")
        monkeypatch.undo()
        assert [m["prompt"] for m in after["members"]] == [m["prompt"] for m in before["members"]]
        assert [t["changes"] for t in after["transitions"]] == [t["changes"] for t in before["transitions"]]
        assert [m["metrics"] for m in after["members"]] == [m["metrics"] for m in before["members"]]
    finally:
        connect.close(conn)


def test_artifact_deltas_are_by_frozen_identity_and_spelled_by_frozen_name():
    """Two different LoRA files that share a display name CHANGED; one
    file renamed between members did NOT."""

    from db import build

    document, sha = _synthetic(3, "second")
    loras = [
        {"role": "lora", "uuid": "a" * 32, "name": "detail", "weight": 1.0},
        {"role": "lora", "uuid": "b" * 32, "name": "detail", "weight": 1.0},
        {"role": "lora", "uuid": "b" * 32, "name": "detail-renamed", "weight": 1.0},
    ]
    for member, lora in zip(document["members"], loras, strict=True):
        member["generation"]["artifacts"] = [lora]
    sha = stories._identity(document)[1]
    conn = connect.memory()
    try:
        conn.executescript(build.schema_sql())
        plan_id = _store(conn, document, sha)
        view = evolution.load(conn, plan_id, models_dir="unused")
        first, second = (t["changes"] for t in view["transitions"])
        assert (first["lora_uuids_added"], first["lora_uuids_removed"]) == (["b" * 32], ["a" * 32])
        assert (first["loras_added"], first["loras_removed"]) == (["detail"], ["detail"]), "same name, different file"
        assert (second["lora_uuids_added"], second["lora_uuids_removed"], second["loras_added"]) == ([], [], []), (
            "a rename is not a change"
        )
    finally:
        connect.close(conn)


def test_the_module_returns_identities_and_the_route_addresses_them(planned):
    """`db/evolution.py` owns no URL: members carry slug and the session
    its local day; the web Adapter turns those into thumbnail, page
    and links."""
    client, _root, _names, _snap, made = planned
    conn = connect.connect(client.app.state.db_path)
    try:
        raw = evolution.load(conn, made.id, models_dir="unused")
    finally:
        connect.close(conn)
    assert "thumbnail" not in raw["members"][0]["media"]
    assert "links" not in raw
    assert raw["identities"]["local_day"] is not None
    view = client.get(f"/stories/plans/{made.id}/evolution", headers={"accept": "application/json"}).json()
    assert view["members"][0]["media"]["thumbnail"].startswith("/thumbs/")
    assert view["members"][0]["media"]["page"] == f"/i/{view['members'][0]['media']['slug']}"
    assert view["links"]["gallery_day"].startswith("/g?f=context.local_day%3Aeq%3A"), "spelled by the Facet Interface"
    assert view["links"]["search"].startswith("/search")


def test_the_view_is_immune_to_the_live_library_moving_on(planned, request):
    """The relation changes, a file is replaced, a file is gone: the view
    over the frozen plan reads frozen hashes and frozen bytes only --
    the replacement's vector is never substituted, and what cannot be
    measured says why.

    The only test here that changes the library on disk -- it replaces one
    file's bytes and deletes another -- and `Stage.restore` compares the
    library's (size, mtime) listing, so a mismatch sends the NEXT test down
    the rebuild path. It puts the library back rather than relying on running
    last, which no ordering guarantees; both mutations are recoverable, because
    `_listing` keys on (size, mtime), the bytes are known, and `os.utime`
    restores a stamp.
    """
    client, root, _names, _snap, made = planned
    held = {at: ((root / at).read_bytes(), (root / at).stat()) for at in ("gen_1.png", "gen_2.png")}

    def put_back() -> None:
        for at, (was, stamped) in held.items():
            (root / at).write_bytes(was)
            os.utime(root / at, ns=(stamped.st_atime_ns, stamped.st_mtime_ns))

    request.addfinalizer(put_back)
    before = client.get(f"/stories/plans/{made.id}/evolution", headers={"accept": "application/json"}).json()
    conn = connect.connect(client.app.state.db_path)
    try:
        other = ingest.prompt(conn, "something else entirely", NOW)
        conn.execute("UPDATE generation_prompt SET prompt_id = ? WHERE role = 'effective'", (other,))
        conn.commit()
    finally:
        connect.close(conn)
    assert client.get(f"/stories/plans/{made.id}/evolution", headers={"accept": "application/json"}).json() == before

    # replace gen_1's bytes and re-embed: the frozen member's vector is gone, not swapped
    info = PngInfo()
    info.add_text("parameters", json.dumps({"sui_image_params": {"prompt": "a replacement"}}))
    Image.new("RGB", (12, 12), (250, 250, 250)).save(root / "gen_1.png", pnginfo=info)
    client.post("/roots/1/scan")
    client.post("/jobs/embed")
    _drain(client)
    view = client.get(f"/stories/plans/{made.id}/evolution", headers={"accept": "application/json"}).json()
    second = view["members"][1]
    assert second["media"]["content_sha256"] == before["members"][1]["media"]["content_sha256"], "frozen bytes named"
    assert second["metrics"]["text_image_cosine"] is None
    assert "frozen bytes" in second["metrics"]["text_image_cosine_unavailable"]
    assert view["transitions"][0]["visual_cosine"] is None
    assert view["transitions"][0]["prompt_cosine"] == before["transitions"][0]["prompt_cosine"], (
        "the prompt vector is by frozen text hash and untouched"
    )
    assert view["members"][0]["metrics"] == before["members"][0]["metrics"]

    # the file is deleted: the member keeps its frozen identity, loses its link
    (root / "gen_2.png").unlink()
    client.post("/roots/1/scan")
    view = client.get(f"/stories/plans/{made.id}/evolution", headers={"accept": "application/json"}).json()
    assert view["members"][2]["media"]["name"] == "gen_2.png"
