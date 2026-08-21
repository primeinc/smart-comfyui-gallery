"""A prompt is a reusable semantic substrate, not a StoryPlanner-private
string.

One prompt entity per distinct text; one typed ROLE relation per
generation, with `original_*` rows only where the generator recorded
one (silence is silence); a tool-neutral SECTION parse per (prompt,
grammar) whose section texts are ordinary prompt rows; one vector per
(text, joint space, query policy), produced by explicit idempotent
work, living in the provider's media space (comparable) and its own
resident index (separate corpus); consumers -- the planner's
similarity Seam, prompt neighbours -- read vectors by text hash under
(space, policy) and never by file, generation, section or role, and a
role filter constrains candidates before ranking. The migration carries
prompt ids and FTS integrity across.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import sqlite3
import sys
import types

import numpy as np
import pytest
from litestar.testing import TestClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import (
    connect,
    context,
    derived,
    ingest,
    planning,
    prompt_sections,
    prompts,
    runner,
    settings,
    similarity,
    stories,
)
from sg_web.app import build_app
from vision import semantic
from vision.faiss_index import SpaceSpec

NOW = 1_700_000_000.0
HOUR = 3600.0
MIN = 60.0


def _spelled(moment: float) -> str:
    return datetime.datetime.fromtimestamp(moment, datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")


# --- a fake semantic provider: a text tower with no weights ---------------------


def _unit(text: str, dims: int = 4) -> np.ndarray:
    raw = np.frombuffer(hashlib.sha256(text.encode("utf-8")).digest()[: dims * 4], dtype=np.uint32).astype(np.float32)
    return (raw / np.linalg.norm(raw)).astype(np.float32)


class _Encoder:
    provider = "fake"
    calls: list[str]

    def __init__(self, model, checkpoint):
        self.model, self.checkpoint, self.dimensions = model, checkpoint, 4
        self.calls = []

    def space(self):
        return _fake.space(self.model, self.checkpoint, self.dimensions)

    def encode_query(self, text):
        self.calls.append(text)
        return _unit(text)

    def encode_media(self, media):
        return _unit(media.path)


_fake = types.ModuleType("tests._fake_semantic")
_fake.INSTRUCTION = "describe"
_fake.parse = lambda reference: tuple(reference.split("/", 1))
_fake.immutable = lambda checkpoint: True
_fake.query_policy = lambda model, checkpoint: {
    "provider": "fake",
    "model": model,
    "checkpoint": checkpoint,
    "instruction": _fake.INSTRUCTION,
}
_fake.space = lambda model, checkpoint, dims: SpaceSpec(
    key=f"semantic.fake.{model}.{checkpoint}",
    representation="float32",
    dimensions=int(dims),
    metric="cosine",
    producer=f"fake:{model}",
    producer_version=checkpoint,
    preprocess="fake.media",
    preprocess_version="v1",
)
_ENCODERS: dict[tuple, _Encoder] = {}


def _encoder(models_dir, model, checkpoint, *, offline=False):
    return _ENCODERS.setdefault((model, checkpoint), _Encoder(model, checkpoint))


_fake.encoder = _encoder


@pytest.fixture
def fake_provider(monkeypatch):
    sys.modules["tests._fake_semantic"] = _fake
    monkeypatch.setitem(semantic.PROVIDERS, "fake", "tests._fake_semantic")
    monkeypatch.setattr(_fake, "INSTRUCTION", "describe")
    _ENCODERS.clear()
    yield _fake
    _ENCODERS.clear()


# --- a library: SwarmUI files that carry the prompt as written --------------------

WRITTEN = "a __material__ lighthouse on a cliff"
RAN = [
    "a tin lighthouse on a cliff",
    "a brass lighthouse on a cliff",
    "a copper lighthouse on a cliff",
]
TAGGED = "a copper lighthouse on a cliff <segment:face,0.6,0.5//cid=11> weathered brass <refiner> crisp detail"


def _swarm(path, prompt, *, original=None, negative="blur", seed=1):
    payload = {
        "sui_image_params": {
            "prompt": prompt,
            "negativeprompt": negative,
            "model": "flux-dev",
            "seed": seed,
            "steps": 20,
            "cfgscale": 7,
            "width": 512,
            "height": 512,
        },
        "sui_extra_data": {"original_prompt": original} if original is not None else {},
    }
    info = PngInfo()
    info.add_text("parameters", json.dumps(payload))
    Image.new("RGB", (12, 12), (40 + seed * 20, 90, 140)).save(path, pnginfo=info)


def _library(tmp):
    root = tmp / "lib"
    root.mkdir()
    for i, text in enumerate(RAN[:2]):
        _swarm(root / f"gen_{i}.png", text, original=WRITTEN, seed=i + 1)
    _swarm(root / "gen_2.png", TAGGED, original=WRITTEN, seed=3)
    _swarm(root / "plain.png", "a plain lighthouse", negative="", seed=9)  # no original, no negative
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
def library(tmp_path, fake_provider):
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
            settings.put(conn, "semantic_model", "fake:toy/v1")
            conn.commit()
        finally:
            connect.close(conn)
        yield client, root, names


def _roles(conn, file_id):
    return dict(
        conn.execute(
            "SELECT gp.role, p.text FROM generation_prompt gp JOIN prompt p ON p.id = gp.prompt_id"
            " WHERE gp.file_id = ? ORDER BY gp.role",
            (file_id,),
        ).fetchall()
    )


def test_roles_are_one_relation_over_one_identity_and_silence_is_silence(library):
    """Effective, original and negative are rows of ONE relation over
    the SAME prompt identity -- three files written from one original
    share one prompt row -- `generation` carries no prompt column, the
    generator's parameter stays as evidence, and a file whose metadata
    recorded no original or negative has no row for it: nothing is
    inferred from absence. A re-parse replaces."""
    client, root, names = library
    conn = connect.connect(client.app.state.db_path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(generation)")}
        assert not columns & {"prompt_id", "negative_id"}, "the relation is the one source of truth"
        assert _roles(conn, names["gen_0.png"]) == {"effective": RAN[0], "negative": "blur", "original": WRITTEN}
        assert _roles(conn, names["plain.png"]) == {"effective": "a plain lighthouse"}, "absence creates nothing"
        originals = conn.execute(
            "SELECT count(DISTINCT prompt_id), count(*) FROM generation_prompt WHERE role = 'original'"
        ).fetchone()
        assert originals == (1, 3), "one written prompt, interned once, playing its role three times"
        assert conn.execute(
            "SELECT value_text FROM file_param WHERE file_id = ? AND key = 'original_prompt'", (names["gen_0.png"],)
        ).fetchone() == (WRITTEN,), "the generator's own claim stays as evidence"
        assert conn.execute("SELECT count(*) FROM prompt WHERE text = ''").fetchone()[0] == 0
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO generation_prompt(file_id, role, prompt_id) VALUES(?, 'wildcard', 1)",
                (names["gen_0.png"],),
            )
        conn.rollback()
        # the written text may equal the run text: two roles, one prompt id
        same = ingest.prompt(conn, RAN[0], NOW)
        prompts.assign(conn, names["gen_0.png"], "original", same)
        held = dict(
            conn.execute("SELECT role, prompt_id FROM generation_prompt WHERE file_id = ?", (names["gen_0.png"],))
        )
        assert held["original"] == held["effective"]
        conn.rollback()
        # a re-parse is a replacement: the rows are taken back and rewritten, never accumulated
        ingest.one(conn, names["gen_0.png"], root / "gen_0.png", NOW + 1)
        assert conn.execute(
            "SELECT count(*) FROM generation_prompt WHERE file_id = ?", (names["gen_0.png"],)
        ).fetchone() == (3,)
    finally:
        connect.close(conn)


def test_sections_are_one_tool_neutral_ir_fed_by_the_tools_grammar():
    """The Swarm adapter reads Swarm's grammar; the plain grammar reads
    everything else as one main section. Section texts are the words
    the generator conditioned on: tags resolved, references and
    comments gone, expansions left verbatim, `//cid=N` stripped."""
    held = prompt_sections.parse(TAGGED, "swarm")
    assert [(s.ordinal, s.kind, s.spec, s.text) for s in held] == [
        (0, "main", None, "a copper lighthouse on a cliff"),
        (1, "segment", "face,0.6,0.5", "weathered brass"),
        (2, "refiner", None, "crisp detail"),
    ]
    assert prompt_sections.parse(TAGGED, "plain") == [prompt_sections.Section(0, "main", None, TAGGED)]
    rich = (
        "a <random:red|blue> cat <lora:detail:0.8> <comment:hi> <break> on a mat"
        " <region:0,0,0.5,1,0.8> a dog <object:0,0.5,1,0.5,0.1,0.5> a fish <video> the cat walks"
        " <extend:81> the cat runs <base> base only <videoswap> swapped <pixeldecoder> decoded"
    )
    kinds = [(s.kind, s.spec, s.text) for s in prompt_sections.parse(rich, "swarm")]
    assert kinds == [
        ("main", None, "a <random:red|blue> cat on a mat"),
        ("region", "0,0,0.5,1,0.8", "a dog"),
        ("object", "0,0.5,1,0.5,0.1,0.5", "a fish"),
        ("video", None, "the cat walks"),
        ("extend", "81", "the cat runs"),
        ("base", None, "base only"),
        ("videoswap", None, "swapped"),
        ("pixeldecoder", None, "decoded"),
    ]
    assert prompt_sections.parse("<segment:yolo-face_yolov8m-seg_60.pt-1,0.8,0.25> a face", "swarm")[0].text == ""
    nested = prompt_sections.parse("a photo of a <setvar[animal]:<wildcard:animals>> playing", "swarm")
    assert nested[0].text == "a photo of a <setvar[animal]:<wildcard:animals>> playing"
    assert prompt_sections.parse("unclosed <segment:face", "swarm")[0].text == "unclosed <segment:face"
    assert prompt_sections.main("a (cat:1.2) BREAK blue sky", "plain") == "a (cat:1.2) BREAK blue sky"
    with pytest.raises(ValueError, match="no prompt grammar"):
        prompt_sections.parse("x", "comfy")
    assert prompts.grammar_for("SwarmUI") == "swarm"
    assert prompts.grammar_for("A1111 / Forge") == "plain"
    assert prompts.grammar_for(None) == "plain"


def test_section_texts_are_prompt_rows_and_a_reparse_never_reembeds(library):
    """Each section's text is an ordinary interned prompt row -- the
    main section of a tag-free prompt IS its prompt -- so "a red fox" as
    a main prompt, a region, or a negative is one identity with one
    vector. A parser upgrade re-parses boundaries and leaves every text
    vector where it was."""
    client, _root, names = library
    conn = connect.connect(client.app.state.db_path)
    try:
        tagged = conn.execute(
            "SELECT prompt_id FROM generation_prompt WHERE file_id = ? AND role = 'effective'", (names["gen_2.png"],)
        ).fetchone()[0]
        held = prompts.sections(conn, tagged, "swarm", NOW)
        assert [(s.kind, s.text) for s, _ in held] == [
            ("main", RAN[2]),
            ("segment", "weathered brass"),
            ("refiner", "crisp detail"),
        ]
        main_text_id = held[0][1]
        plain_effective = conn.execute(
            "SELECT prompt_id FROM generation_prompt WHERE file_id = ? AND role = 'effective'", (names["gen_1.png"],)
        ).fetchone()[0]
        assert main_text_id != plain_effective, "a different text is a different row"
        assert conn.execute("SELECT text FROM prompt WHERE id = ?", (main_text_id,)).fetchone() == (RAN[2],)
        whole = prompts.sections(conn, plain_effective, "swarm", NOW)
        assert whole[0][1] == plain_effective, "a tag-free prompt's main section is the prompt itself"
        again = prompts.sections(conn, tagged, "swarm", NOW + 1)
        assert again == held, "a current parse is read, not redone"
        blur = conn.execute("SELECT prompt_id FROM generation_prompt WHERE role = 'negative' LIMIT 1").fetchone()[0]
        assert ingest.prompt(conn, "blur", NOW) == blur, "the same text in another role is the same row"
        conn.rollback()
    finally:
        connect.close(conn)


def _joint(dims=4):
    return SpaceSpec(
        key="semantic.fake.toy.v1",
        representation="float32",
        dimensions=dims,
        metric="cosine",
        producer="fake:toy",
        producer_version="v1",
        preprocess="fake.media",
        preprocess_version="v1",
    )


def test_one_vector_per_text_space_and_policy_in_the_joint_space(library):
    """A prompt vector lives in the provider's JOINT space -- the same
    space_id as that provider's media vectors, so cosine(prompt, media)
    is permitted -- keyed by the query policy that produced it: a
    changed instruction is a new row that coexists; its identity is
    immutable (a replacement is a new id); its length is held to the
    space; currency is the text hash; and the two corpora never share
    a resident index."""
    client, _root, names = library
    conn = connect.connect(client.app.state.db_path)
    try:
        prompt_id, digest = conn.execute(
            "SELECT p.id, p.text_hash FROM generation_prompt gp JOIN prompt p ON p.id = gp.prompt_id"
            " WHERE gp.file_id = ? AND gp.role = 'effective'",
            (names["gen_0.png"],),
        ).fetchone()
        media = derived.record_embedding(conn, names["gen_0.png"], _joint(), _unit("pixels"), "abc", NOW)
        first = derived.record_prompt_embedding(conn, prompt_id, _joint(), "qA", _unit("one"), digest, NOW)
        again = derived.record_prompt_embedding(conn, prompt_id, _joint(), "qA", _unit("two"), digest, NOW + 1)
        assert again > first, "a replacement is a NEW immutable id"
        sid = similarity.space_id(conn, _joint(), NOW)
        spaces = conn.execute(
            "SELECT (SELECT space_id FROM derived_embedding WHERE id = ?),"
            " (SELECT space_id FROM derived_prompt_embedding WHERE id = ?)",
            (media, again),
        ).fetchone()
        assert spaces == (sid, sid), "one coordinate system: prompt and media vectors are comparable"
        other = derived.record_prompt_embedding(conn, prompt_id, _joint(), "qB", _unit("three"), digest, NOW + 2)
        assert other != again
        rows = conn.execute(
            "SELECT policy_hash FROM derived_prompt_embedding WHERE prompt_id = ? ORDER BY id", (prompt_id,)
        ).fetchall()
        assert rows == [("qA",), ("qB",)], "a changed policy coexists; nothing is relabeled"
        assert prompts.current_vectors(conn, sid, "qA", [digest])[digest] == pytest.approx(_unit("two").tolist())
        assert prompts.current_vectors(conn, sid, "qB", [digest])[digest] == pytest.approx(_unit("three").tolist())
        with pytest.raises(sqlite3.IntegrityError, match="disagrees with its space"):
            conn.execute(
                "INSERT INTO derived_prompt_embedding(prompt_id, space_id, policy_hash, vector, source_text_hash,"
                " computed_at) VALUES(?, ?, 'qC', ?, ?, ?)",
                (prompt_id, sid, _unit("x", 3).tobytes(), digest, NOW),
            )
        conn.execute("UPDATE prompt SET text = text || '!', text_hash = 'moved' WHERE id = ?", (prompt_id,))
        assert prompts.current_vectors(conn, sid, "qA", [digest]) == {}, "the text moved; the vector no longer vouches"
        keys = {key for key, _, _ in similarity._PENDING[id(conn)]}
        assert keys == {
            f"semantic.fake.toy.v1@{sid}",
            f"semantic.fake.toy.v1@{sid}+prompts+qA",
            f"semantic.fake.toy.v1@{sid}+prompts+qB",
        }, "media and prompt rows are noted into different resident indexes"
        conn.rollback()
        similarity.discard_pending(conn)
    finally:
        connect.close(conn)


def test_embedding_is_explicit_durable_and_idempotent_work(library):
    """`/jobs/embed_prompts` parses sections and queues every TEXT --
    role prompts and section texts -- without a current vector under
    (space, policy), and only those: a second ask queues nothing, a
    retried item recomputes nothing, and an instruction change queues
    everything again under a new policy while the old rows stay."""
    client, _root, _names = library
    made = client.post("/jobs/embed_prompts")
    assert made.status_code in (200, 201), made.text
    assert [job["kind"] for job in made.json()] == ["embed_prompts"]
    # 3 effective (one tagged) + written + blur + plain + the tagged prompt's 3 section texts
    assert made.json()[0]["total"] == 9
    _drain(client)
    conn = connect.connect(client.app.state.db_path)
    try:
        assert conn.execute("SELECT state, done_count FROM job WHERE kind = 'embed_prompts'").fetchone() == ("done", 9)
        assert conn.execute("SELECT count(*) FROM derived_prompt_embedding").fetchone() == (9,)
        assert conn.execute("SELECT count(*) FROM similarity_space WHERE key LIKE 'text.%'").fetchone() == (0,)
        shape = conn.execute(
            "SELECT count(DISTINCT space_id), count(DISTINCT policy_hash) FROM derived_prompt_embedding"
        )
        assert shape.fetchone() == (1, 1)
        encoder = _ENCODERS[("toy", "v1")]
        assert len(encoder.calls) == 9
        again = client.post("/jobs/embed_prompts")
        assert again.json()[0]["total"] == 0, "nothing current is re-queued"
        one = conn.execute("SELECT prompt_id FROM derived_prompt_embedding ORDER BY id LIMIT 1").fetchone()[0]
        queued = json.loads(conn.execute("SELECT payload FROM job WHERE kind = 'embed_prompts' LIMIT 1").fetchone()[0])
        prompts.embed_item(conn, one, queued, NOW + 5)
        assert len(encoder.calls) == 9, "a retried item recomputes nothing"
    finally:
        connect.close(conn)
    _fake.INSTRUCTION = "represent for retrieval"
    moved = client.post("/jobs/embed_prompts")
    assert moved.json()[0]["total"] == 9, "a new query policy is new work for every text"
    _drain(client)
    conn = connect.connect(client.app.state.db_path)
    try:
        shape = conn.execute("SELECT count(DISTINCT policy_hash), count(*) FROM derived_prompt_embedding")
        assert shape.fetchone() == (2, 18), "old and new policy rows coexist"
    finally:
        connect.close(conn)


def _request(snap, engine, threshold):
    return planning.request_identity(
        snap.sha256,
        "generation_history",
        planning.GenerationHistoryPlanner.version,
        *engine.identity(),
        {"phase_threshold": threshold},
    )


def test_the_planner_reads_vectors_by_frozen_text_hash_only(library):
    """The planner's Seam stays value-in/vector-out; orchestration puts
    the durable cache behind it KEYED BY TEXT HASH under (space,
    policy). Once a frozen text has a vector, planning embeds nothing --
    even after the file that carried the prompt is gone and the role
    relation has changed -- and a text the cache has never seen is
    computed once and remembered. The planner compares MAIN sections."""
    client, _root, names = library
    client.post("/jobs/context")
    client.post("/jobs/events")
    _drain(client)
    conn = connect.connect(client.app.state.db_path)
    try:
        event_id = conn.execute("SELECT id FROM derived_event WHERE kind = 'generation_session'").fetchone()[0]
        snap = stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
        conn.commit()
        members = stories.load_snapshot(conn, snap.id)["members"]
        frozen = [one["generation"]["prompts"] for one in members]
        assert all(
            {"role", "uuid", "text", "text_hash"} == set(p) and p["text_hash"] == prompts.text_hash(p["text"])
            for held in frozen
            for p in held
        ), "the snapshot freezes role, stable identity, exact text and the hash of those bytes"

        engine = planning.engine_for(conn, "fake", "unused")
        assert isinstance(engine, planning.SemanticEngine)
        assert engine.identity()[1] == semantic.policy_hash("fake", "toy", "v1"), "one policy token everywhere"
        encoder = _ENCODERS.setdefault(("toy", "v1"), _Encoder("toy", "v1"))
        payload = {
            "request_sha256": _request(snap, engine, 0.5),
            "snapshot_id": snap.id,
            "planner": "generation_history",
            "settings": {"phase_threshold": 0.5},
            "engine": {"selector": "fake", **engine.payload()},
        }
        planning.plan_item(conn, 0, payload, NOW + 31 * HOUR)
        conn.commit()
        mains = sorted(
            {
                prompt_sections.main(p["text"], "swarm")
                for held in frozen
                for p in held
                if p["role"] in ("effective", "original")
            }
        )
        assert sorted(set(encoder.calls)) == mains, "every frozen MAIN text embedded exactly once; no tag tails"
        assert TAGGED not in encoder.calls
        marks = ",".join("?" for _ in mains)
        rows = conn.execute(f"SELECT count(*) FROM prompt WHERE text IN ({marks})", mains).fetchone()[0]
        assert rows == len(mains), "every main-section text is a prompt row from ingest on"
        assert conn.execute("SELECT count(*) FROM derived_prompt_embedding").fetchone() == (rows,), (
            "the vectors were remembered under the prompt rows holding those texts"
        )

        # the library moves on: a file goes, a role relation changes --
        # the frozen snapshot planned under a new request still embeds NOTHING
        conn.execute("DELETE FROM file WHERE id = ?", (names["gen_1.png"],))
        conn.execute("DELETE FROM generation_prompt WHERE role = 'original'")
        conn.commit()
        encoder.calls.clear()
        payload["settings"] = {"phase_threshold": 0.6}
        payload["request_sha256"] = _request(snap, engine, 0.6)
        planning.plan_item(conn, 0, payload, NOW + 32 * HOUR)
        conn.commit()
        assert encoder.calls == [], "served by frozen text hash; no file, no generation, no role was consulted"
        assert conn.execute("SELECT count(*) FROM story_plan").fetchone() == (2,)
    finally:
        connect.close(conn)

    spelled = " ".join(c for c in prompts.current_vectors.__code__.co_consts if isinstance(c, str) and "SELECT" in c)
    for banned in ("file", "generation", "role", "section"):
        assert banned not in spelled, f"the cache lookup knows {banned!r}"


def test_the_cache_adapter_is_exact_and_computes_each_text_once():
    inner = planning.LexicalPromptSimilarity()
    seen: list[list[str]] = []

    class Counting(planning.LexicalPromptSimilarity):
        def embed(self, texts):
            seen.append(list(texts))
            return super().embed(texts)

    store: dict[str, list[float]] = {}
    cached = planning.CachedPromptSimilarity(
        Counting(), lambda hashes: {h: store[h] for h in hashes if h in store}, store.__setitem__
    )
    assert (cached.name, cached.version) == (inner.name, inner.version)
    texts = ["a tin lighthouse", "a brass helmet", "a tin lighthouse"]
    first = cached.embed(texts)
    assert seen == [["a tin lighthouse", "a brass helmet"]], "duplicates in a batch embed once"
    assert first[0] == first[2]
    assert len(store) == 2
    second = cached.embed(["a brass helmet", "a copper helmet"])
    assert seen[-1] == ["a copper helmet"], "only the unseen text reaches the engine"
    assert second[0] == first[1]


def _member(ordinal, ran, *, written=None, precision="second", tool="SwarmUI"):
    held = [{"role": "effective", "uuid": f"{ordinal:032x}", "text": ran, "text_hash": prompts.text_hash(ran)}]
    if written is not None:
        held.append(
            {
                "role": "original",
                "uuid": f"{99 - ordinal:032x}",
                "text": written,
                "text_hash": prompts.text_hash(written),
            }
        )
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
            "tool": tool,
            "detection": "marker",
            "seed": ordinal,
            "steps": 20,
            "cfg": 7.0,
            "denoise": None,
            "clip_skip": None,
            "sampler": "Euler a",
            "scheduler": None,
            "width": 512,
            "height": 512,
            "prompt": ran,
            "negative_prompt": None,
            "prompts": held,
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


def _snapshot(members):
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


def test_written_versus_run_is_a_claim_about_a_member_never_a_boundary():
    """A member carrying both the prompt as written and the prompt the
    generator ran is compared with itself on the MAIN section: a
    substantial rewrite is a `prompt_rewrite` claim naming that member;
    wildcard-level drift (same hash, or above threshold) is nothing; a
    missing original is nothing; a tag tail is not a rewrite. Phases
    are exactly what they would be without originals. The claim is v3
    grammar; v2 refuses it; the renderer words it without a temporal
    word."""
    from db import rendering

    rewritten = "__subject__ in the __style__ style"  # no token survives expansion
    members = [
        _member(0, RAN[0], written=rewritten),
        _member(1, RAN[1], written=RAN[1]),  # same text: no pair
        _member(2, RAN[2]),  # no original: absence
        _member(3, "a brass diving helmet in a museum case", written="a __metal__ diving helmet in a museum case"),
        _member(4, RAN[2] + " <segment:face> fixed face", written=RAN[2]),  # a tag tail is not a rewrite
    ]
    document, sha = _snapshot(members)
    planner = planning.GenerationHistoryPlanner(planning.LexicalPromptSimilarity())
    plan = planner.plan(document, sha)
    assert plan["v"] == 3
    claims = {c["kind"]: c for c in plan["claims"]}
    assert claims["prompt_rewrite"]["evidence_refs"] == ["member-001:generation.prompts"]
    assert claims["prompt_rewrite"]["facts"]["members"] == 1
    assert claims["prompt_rewrite"]["facts"]["min_cosine"] < 0.5
    assert "member-004" not in json.dumps(claims["prompt_rewrite"]), "wildcard-level drift is not a rewrite"
    bare = planner.plan(*_snapshot([_member(i, m["generation"]["prompt"]) for i, m in enumerate(members)]))
    assert [p["member_refs"] for p in plan["phases"]] == [p["member_refs"] for p in bare["phases"]], (
        "originals never move a boundary"
    )
    assert [p["member_refs"] for p in plan["phases"]] == [
        ["member-001", "member-002", "member-003"],
        ["member-004"],
        ["member-005"],
    ], "sequenced phases are contiguous; the tagged member follows the helmet in time"
    assert planning.validate_plan(plan, document, sha) == []
    assert any("not a v2 claim kind" in why for why in planning.validate_story_plan_v2({**plan, "v": 2}))
    plan_sha = planning.identity(plan)[1]
    render = rendering.TemplateStoryRenderer("technical").render(document, plan, sha, plan_sha)
    assert rendering.violations(render, plan, document, sha, plan_sha) == []
    wanted = [claims["prompt_rewrite"]["id"]]
    told = [b["text"] for s in render["sections"] for b in s["blocks"] if b.get("claim_refs") == wanted]
    assert told == [
        (
            "For one image here, the prompt the generator ran differs substantially from the prompt as written."
            " Minimum similarity between written and run prompt is 0%."
        )
    ]
    daily = [dict(m, occurrence={**m["occurrence"], "precision": "day"}) for m in members]
    document, sha = _snapshot(daily)
    plan = planner.plan(document, sha)
    assert "prompt_rewrite" in {c["kind"] for c in plan["claims"]}
    plan_sha = planning.identity(plan)[1]
    render = rendering.TemplateStoryRenderer("memory").render(document, plan, sha, plan_sha)
    assert rendering.violations(render, plan, document, sha, plan_sha) == []


def test_neighbours_answer_inside_one_space_and_policy_with_role_before_rank(library):
    """Neighbours come from ONE (space, policy) -- another policy's rows
    are history, not candidates -- and a role filter constrains the
    candidates before ranking at full depth: the only original-role
    prompt is returned whatever its global rank."""
    client, _root, names = library
    client.post("/jobs/embed_prompts")
    _drain(client)
    conn = connect.connect(client.app.state.db_path)
    try:
        mine = conn.execute(
            "SELECT prompt_id FROM generation_prompt WHERE file_id = ? AND role = 'effective'", (names["gen_0.png"],)
        ).fetchone()[0]
        written = conn.execute("SELECT DISTINCT prompt_id FROM generation_prompt WHERE role = 'original'").fetchone()[0]
        for (prompt_id,) in conn.execute("SELECT id FROM prompt").fetchall():
            derived.record_prompt_embedding(conn, prompt_id, _joint(), "qOld", _unit("same for all"), "stale", NOW)
        conn.commit()
        similarity.discard_pending(conn)
        policy = semantic.policy_hash("fake", "toy", "v1")
    finally:
        connect.close(conn)
    told = client.get(f"/prompts/{mine}/neighbours", params={"space": "fake", "k": 2})
    assert told.status_code == 200, told.text
    body = told.json()
    assert (body["space"], body["policy"]) == ("semantic.fake.toy.v1", policy)
    assert len(body["results"]) == 2
    assert mine not in [r["prompt_id"] for r in body["results"]]
    assert all({"prompt_id", "uuid", "slug", "text", "score"} == set(r) for r in body["results"])
    assert body["results"][0]["score"] >= body["results"][1]["score"]
    everything = client.get(f"/prompts/{mine}/neighbours", params={"space": "fake", "k": 100}).json()
    assert len(everything["results"]) == 8, "the current policy's corpus (every text but this one), and only that"
    rank = [r["prompt_id"] for r in everything["results"]].index(written)
    filtered = client.get(f"/prompts/{mine}/neighbours", params={"space": "fake", "k": 1, "role": "original"}).json()
    assert [r["prompt_id"] for r in filtered["results"]] == [written], f"rank {rank} globally; first among originals"
    assert client.get(f"/prompts/{mine}/neighbours", params={"space": "fake", "role": "vibe"}).status_code == 400
    assert client.get(f"/prompts/{mine}/neighbours", params={"space": "openclip"}).status_code == 400, (
        "a space that is not configured is refused, never substituted"
    )
    assert client.get("/prompts/424242/neighbours", params={"space": "fake"}).status_code == 404


def test_the_migration_carries_prompt_ids_roles_and_fts_integrity(tmp_path):
    """A v17 database with prompts on `generation` and the generator's
    `original_*` parameters comes across with every prompt id intact,
    the roles filled, the originals interned, the FTS index whole, and
    the job vocabulary widened."""
    from db import build, migrate, scan

    path = tmp_path / "old.db"
    build.build(path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("DROP TABLE derived_prompt_embedding")
    conn.execute("DROP TABLE derived_prompt_section")
    conn.execute("DROP TABLE generation_prompt")
    conn.execute("DROP TABLE generation")
    conn.execute(
        "CREATE TABLE generation (file_id INTEGER PRIMARY KEY REFERENCES file(id) ON DELETE CASCADE,"
        " tool TEXT NOT NULL, detection TEXT NOT NULL CHECK (detection IN ('graph','marker','heuristic','stealth')),"
        " workflow_id INTEGER REFERENCES artifact(id) ON DELETE SET NULL,"
        " prompt_id INTEGER REFERENCES prompt(id) ON DELETE SET NULL,"
        " negative_id INTEGER REFERENCES prompt(id) ON DELETE SET NULL,"
        " seed INTEGER, steps INTEGER, cfg REAL, denoise REAL, clip_skip INTEGER, sampler TEXT, scheduler TEXT,"
        " width INTEGER, height INTEGER, parser TEXT NOT NULL, parsed_at REAL NOT NULL) STRICT"
    )
    for index in (
        "generation_workflow ON generation(workflow_id)",
        "generation_prompt ON generation(prompt_id)",
        "generation_negative ON generation(negative_id)",
        "generation_seed ON generation(seed)",
    ):
        conn.execute(f"CREATE INDEX {index}")
    conn.execute("PRAGMA user_version = 17")
    conn.commit()
    conn.close()
    conn = connect.connect(str(path))
    try:
        root_id = conn.execute("INSERT INTO root(path, kind, created_at) VALUES('C:/x', 'library', 0)").lastrowid
        folder = scan.mint(conn, "folder", "x")
        conn.execute(
            "INSERT INTO folder(id, root_id, parent_id, name, depth) VALUES(?, ?, NULL, 'x', 0)", (folder, root_id)
        )
        file_id = scan.mint(conn, "file", "g.png")
        conn.execute(
            "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256, first_seen_at, last_seen_at)"
            " VALUES(?, ?, 'g.png', 'image', 1, 0, 'abc', 0, 0)",
            (file_id, folder),
        )
        ran = ingest.prompt(conn, "a tin lighthouse", NOW)
        neg = ingest.prompt(conn, "blur", NOW)
        conn.execute(
            "INSERT INTO generation(file_id, tool, detection, prompt_id, negative_id, parser, parsed_at)"
            " VALUES(?, 'SwarmUI', 'marker', ?, ?, 'metaparse/1', 0)",
            (file_id, ran, neg),
        )
        conn.executemany(
            "INSERT INTO file_param(file_id, source, key, value_text) VALUES(?, 'generation', ?, ?)",
            [(file_id, "original_prompt", WRITTEN), (file_id, "original_negativeprompt", "__bad__")],
        )
        conn.commit()
    finally:
        connect.close(conn)
    assert migrate.migrate(path) == [18]
    conn = connect.connect(str(path))
    try:
        held = dict(
            conn.execute("SELECT role, prompt_id FROM generation_prompt WHERE file_id = ?", (file_id,)).fetchall()
        )
        assert (held["effective"], held["negative"]) == (ran, neg), "prompt ids are carried, not re-minted"
        assert conn.execute("SELECT text FROM prompt WHERE id = ?", (held["original"],)).fetchone() == (WRITTEN,)
        assert conn.execute("SELECT text FROM prompt WHERE id = ?", (held["original_negative"],)).fetchone() == (
            "__bad__",
        )
        conn.execute("INSERT INTO prompt_fts(prompt_fts) VALUES('integrity-check')")
        found = conn.execute("SELECT rowid FROM prompt_fts WHERE prompt_fts MATCH 'lighthouse' ORDER BY rowid")
        assert found.fetchall() == [(ran,), (held["original"],)]
        conn.execute("INSERT INTO job(kind, state, created_at) VALUES('embed_prompts', 'queued', 0)")
        assert conn.execute("SELECT count(*) FROM derived_prompt_embedding").fetchone() == (0,)
    finally:
        connect.close(conn)


def test_nothing_in_the_snapshot_depends_on_todays_relation(library):
    """A frozen snapshot is immune to the live relation changing after
    it: reassigning a role today does not change what was frozen, and
    the context doctrine governs the present -- a staled member takes
    its hypothesis with it, so there is no current event to freeze
    again until the events job runs."""
    client, _root, names = library
    client.post("/jobs/context")
    client.post("/jobs/events")
    _drain(client)
    conn = connect.connect(client.app.state.db_path)
    try:
        event_id = conn.execute("SELECT id FROM derived_event WHERE kind = 'generation_session'").fetchone()[0]
        snap = stories.snapshot_event(conn, event_id, NOW + 30 * HOUR)
        conn.commit()
        before = stories.load_snapshot(conn, snap.id)
        other = ingest.prompt(conn, "something else entirely", NOW)
        conn.execute(
            "UPDATE generation_prompt SET prompt_id = ? WHERE file_id = ? AND role = 'effective'",
            (other, names["gen_0.png"]),
        )
        context.stale(conn, names["gen_0.png"])
        conn.commit()
        assert stories.load_snapshot(conn, snap.id) == before
        with pytest.raises(LookupError, match="no event"):
            stories.snapshot_event(conn, event_id, NOW + 31 * HOUR)
    finally:
        connect.close(conn)


# --- freshness, frozen jobs, corpus, exact selection (review of 4d1fa31) ---------


def _provisionable(monkeypatch):
    """The fake provider learns hub-style mutability: `main` is a
    pointer, pinned to a commit only once something is cached."""
    monkeypatch.setattr(_fake, "PINNED", {}, raising=False)
    monkeypatch.setattr(_fake, "immutable", lambda checkpoint: checkpoint != "main")
    monkeypatch.setattr(
        _fake, "pin", lambda models_dir, model, checkpoint: _fake.PINNED.get(checkpoint, checkpoint), raising=False
    )

    def encoder(models_dir, model, checkpoint, *, offline=False):
        # loading downloads: the mutable pointer resolves to a commit
        resolved = _fake.PINNED.setdefault(checkpoint, "a" * 40) if checkpoint == "main" else checkpoint
        return _ENCODERS.setdefault((model, resolved), _Encoder(model, resolved))

    monkeypatch.setattr(_fake, "encoder", encoder)


def test_a_fresh_mutable_checkpoint_is_refused_then_pinned_and_every_row_names_the_commit(library, monkeypatch):
    client, _root, _names = library
    _provisionable(monkeypatch)
    conn = connect.connect(client.app.state.db_path)
    try:
        settings.put(conn, "semantic_model", "fake:toy/main")
        conn.commit()
    finally:
        connect.close(conn)
    refused = client.post("/jobs/embed_prompts")
    assert refused.status_code == 400, refused.text
    assert "mutable revision 'main'" in refused.json()["detail"]
    _fake.PINNED["main"] = "b" * 40  # /jobs/embed provisioned and pinned
    made = client.post("/jobs/embed_prompts")
    assert made.status_code == 201, made.text
    _drain(client)
    conn = connect.connect(client.app.state.db_path)
    try:
        payload = json.loads(
            conn.execute("SELECT payload FROM job WHERE id = ?", (made.json()[0]["id"],)).fetchone()[0]
        )
        assert payload["choice"] == ["fake", "toy", "b" * 40]
        assert payload["space"] == "semantic.fake.toy." + "b" * 40
        assert payload["policy_hash"] == semantic.policy_hash("fake", "toy", "b" * 40)
        spaces = conn.execute(
            "SELECT producer_version FROM similarity_space WHERE key LIKE 'semantic.fake.%'"
        ).fetchall()
        assert spaces == [("b" * 40,)], "no durable identity names the pointer"
        landed = conn.execute(
            "SELECT count(*) FROM derived_prompt_embedding WHERE policy_hash = ?", (payload["policy_hash"],)
        ).fetchone()[0]
        assert landed == 9
        assert "main" not in json.dumps(conn.execute("SELECT key, producer_version FROM similarity_space").fetchall())
    finally:
        connect.close(conn)


def test_a_queued_embed_job_is_re_proven_and_never_duplicated(library, monkeypatch):
    client, _root, _names = library
    first = client.post("/jobs/embed_prompts").json()[0]["id"]
    for _ in range(3):
        assert client.post("/jobs/embed_prompts").json()[0]["id"] == first, "one live computation per (space, policy)"
    monkeypatch.setattr(_fake, "INSTRUCTION", "a new instruction")  # the policy moved between queue and run
    _drain(client)
    conn = connect.connect(client.app.state.db_path)
    try:
        state = conn.execute("SELECT state FROM job WHERE id = ?", (first,)).fetchone()[0]
        failed = conn.execute(
            "SELECT count(*) FROM job_item WHERE job_id = ? AND state = 'failed'", (first,)
        ).fetchone()
        assert (state, failed) == ("done", (9,)), "every item refused: the job no longer meant what it meant"
        assert conn.execute("SELECT count(*) FROM derived_prompt_embedding").fetchone() == (0,), (
            "nothing landed under a name it was not queued under"
        )
        why = conn.execute("SELECT error FROM job_item WHERE job_id = ? LIMIT 1", (first,)).fetchone()[0]
        assert "no longer means" in why
    finally:
        connect.close(conn)
    again = client.post("/jobs/embed_prompts").json()[0]
    assert again["id"] != first
    assert again["total"] == 9, "a fresh ask under the new policy queues everything again"


def test_a_text_a_parser_no_longer_produces_leaves_the_corpus_but_keeps_its_vector(library, monkeypatch):
    client, _root, names = library
    client.post("/jobs/embed_prompts")
    _drain(client)
    conn = connect.connect(client.app.state.db_path)
    try:
        mine = conn.execute(
            "SELECT prompt_id FROM generation_prompt WHERE file_id = ? AND role = 'effective'", (names["gen_0.png"],)
        ).fetchone()[0]
        section_text = conn.execute("SELECT id FROM prompt WHERE text = 'weathered brass'").fetchone()[0]
        before = conn.execute("SELECT count(*) FROM derived_prompt_embedding").fetchone()[0]
    finally:
        connect.close(conn)
    everything = client.get(f"/prompts/{mine}/neighbours", params={"space": "fake", "k": 100}).json()
    assert section_text in [r["prompt_id"] for r in everything["results"]]
    # the parser moves on and, under the new grammar, the tagged prompt is read as one main section
    monkeypatch.setattr(prompt_sections, "VERSION", prompt_sections.VERSION + 1)
    monkeypatch.setattr(
        prompt_sections, "parse", lambda text, grammar: [prompt_sections.Section(0, "main", None, text)]
    )
    assert client.post("/jobs/embed_prompts").json()[0]["total"] == 0, (
        "under the new grammar the tagged prompt IS its main section, and that text already has a vector"
    )
    _drain(client)
    everything = client.get(f"/prompts/{mine}/neighbours", params={"space": "fake", "k": 100}).json()
    assert section_text not in [r["prompt_id"] for r in everything["results"]], "out of the corpus"
    conn = connect.connect(client.app.state.db_path)
    try:
        kept = conn.execute("SELECT count(*) FROM derived_prompt_embedding WHERE prompt_id = ?", (section_text,))
        assert kept.fetchone() == (1,), "its vector stays as history"
        assert conn.execute("SELECT count(*) FROM derived_prompt_embedding").fetchone()[0] == before
    finally:
        connect.close(conn)


def test_a_space_selector_is_exact_and_ambiguity_is_refused(library):
    client, _root, names = library
    conn = connect.connect(client.app.state.db_path)
    try:
        settings.put(conn, "semantic_model", "fake:toy/v1,fake:toy2/v1")
        conn.commit()
        mine = conn.execute(
            "SELECT prompt_id FROM generation_prompt WHERE file_id = ? AND role = 'effective'", (names["gen_0.png"],)
        ).fetchone()[0]
    finally:
        connect.close(conn)
    client.post("/jobs/embed_prompts")
    _drain(client)
    vague = client.get(f"/prompts/{mine}/neighbours", params={"space": "fake"})
    assert vague.status_code == 400
    assert "names 2 configured spaces" in vague.json()["detail"]
    exact = client.get(f"/prompts/{mine}/neighbours", params={"space": "fake:toy2@v1", "k": 2})
    assert exact.status_code == 200, exact.text
    assert exact.json()["space"] == "semantic.fake.toy2.v1"
    assert client.get(f"/prompts/{mine}/neighbours", params={"space": "fake:toy", "k": 2}).status_code == 200
    conn = connect.connect(client.app.state.db_path)
    try:
        with pytest.raises(ValueError, match="names 2 configured spaces"):
            planning.engine_for(conn, "fake", "unused")
        assert planning.engine_for(conn, "fake:toy2", "unused").model == "toy2"
    finally:
        connect.close(conn)


def test_the_migration_carries_the_unsampler_prompt(tmp_path):
    from db import build, migrate, scan

    path = tmp_path / "old.db"
    build.build(path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=OFF")
    for table in ("derived_prompt_embedding", "derived_prompt_section", "generation_prompt", "generation"):
        conn.execute(f"DROP TABLE {table}")
    conn.execute(
        "CREATE TABLE generation (file_id INTEGER PRIMARY KEY REFERENCES file(id) ON DELETE CASCADE,"
        " tool TEXT NOT NULL, detection TEXT NOT NULL CHECK (detection IN ('graph','marker','heuristic','stealth')),"
        " workflow_id INTEGER REFERENCES artifact(id) ON DELETE SET NULL,"
        " prompt_id INTEGER REFERENCES prompt(id) ON DELETE SET NULL,"
        " negative_id INTEGER REFERENCES prompt(id) ON DELETE SET NULL,"
        " seed INTEGER, steps INTEGER, cfg REAL, denoise REAL, clip_skip INTEGER, sampler TEXT, scheduler TEXT,"
        " width INTEGER, height INTEGER, parser TEXT NOT NULL, parsed_at REAL NOT NULL) STRICT"
    )
    for index in (
        "generation_workflow ON generation(workflow_id)",
        "generation_prompt ON generation(prompt_id)",
        "generation_negative ON generation(negative_id)",
        "generation_seed ON generation(seed)",
    ):
        conn.execute(f"CREATE INDEX {index}")
    conn.execute("PRAGMA user_version = 17")
    conn.commit()
    conn.close()
    conn = connect.connect(str(path))
    try:
        root_id = conn.execute("INSERT INTO root(path, kind, created_at) VALUES('C:/x', 'library', 0)").lastrowid
        folder = scan.mint(conn, "folder", "x")
        conn.execute(
            "INSERT INTO folder(id, root_id, parent_id, name, depth) VALUES(?, ?, NULL, 'x', 0)", (folder, root_id)
        )
        file_id = scan.mint(conn, "file", "g.png")
        conn.execute(
            "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256, first_seen_at, last_seen_at)"
            " VALUES(?, ?, 'g.png', 'image', 1, 0, 'abc', 0, 0)",
            (file_id, folder),
        )
        ran = ingest.prompt(conn, "a tin lighthouse", NOW)
        conn.execute(
            "INSERT INTO generation(file_id, tool, detection, prompt_id, parser, parsed_at)"
            " VALUES(?, 'SwarmUI', 'marker', ?, 'metaparse/1', 0)",
            (file_id, ran),
        )
        conn.execute(
            "INSERT INTO file_param(file_id, source, key, value_text) VALUES(?, 'generation', 'unsamplerprompt', ?)",
            (file_id, "a man wearing a black hat"),
        )
        conn.commit()
    finally:
        connect.close(conn)
    assert migrate.migrate(path) == [18]
    conn = connect.connect(str(path))
    try:
        held = dict(
            conn.execute("SELECT role, prompt_id FROM generation_prompt WHERE file_id = ?", (file_id,)).fetchall()
        )
        told = conn.execute("SELECT text FROM prompt WHERE id = ?", (held["unsampler"],)).fetchone()
        assert told == ("a man wearing a black hat",)
        assert conn.execute("SELECT count(*) FROM file_param WHERE key = 'unsamplerprompt'").fetchone() == (1,), (
            "raw evidence stays"
        )
    finally:
        connect.close(conn)
