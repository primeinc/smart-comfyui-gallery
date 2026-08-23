"""An artifact describes a media predicate; ResultSet owns the answer.

The artifact page, /g?artifact=, MediaView walking, bulk selection and
saved smart views all consume the ONE materialized projection -- the
same ordered ids under the same answer identity -- and no surface owns
a private artifact media list. Whether membership means file_artifact
or generation.workflow_id is the ResultSet's private knowledge; a
twice-stacked LoRA is one media member; counts count pictures. These
tests pin the Interface -- two surfaces asking one question get one
answer -- never today's helper mechanics.
"""

from __future__ import annotations

import os
import pathlib

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import collection_rules, collections, connect, ingest, naming, resultset
from tests.staging import Stage, staged

NOW = 1_700_000_000.0


def _library(root: pathlib.Path) -> None:
    """Six recipe-carrying stills: four on checkpoint alpha (two with the
    filmGrain LoRA), two on beta."""
    for i in range(6):
        model = "alpha" if i < 4 else "beta"
        lora = "<lora:filmGrain:0.4> " if i in (0, 1) else ""
        info = PngInfo()
        info.add_text(
            "parameters",
            f"a tin lighthouse {lora}\nNegative prompt: blur\n"
            f"Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: {i}, "
            f"Size: 512x512, Model: {model}",
        )
        path = root / f"pic_{i}.png"
        Image.new("RGB", (12, 12), (30 + i * 30, 80, 120)).save(path, pnginfo=info)
        os.utime(path, (NOW + i * 60, NOW + i * 60))


def _prepare(stage: Stage) -> None:
    client, root = stage.client, stage.root
    conn = connect.connect(client.app.state.db_path)
    try:
        for file_id, name in conn.execute("SELECT id, name FROM file ORDER BY id").fetchall():
            ingest.one(conn, file_id, root / name, NOW)
        # A workflow artifact, attached the way workflows attach:
        # through generation, never file_artifact.
        flow = ingest.artifact(conn, "workflow", "tinUpscale", NOW)
        conn.execute(
            "UPDATE generation SET workflow_id = ? WHERE file_id IN"
            " (SELECT id FROM file WHERE name IN ('pic_2.png', 'pic_3.png'))",
            (flow,),
        )
        # The same LoRA stacked TWICE in one file: a second ordinal,
        # exactly what the schema permits and counts must not double.
        lora_id = conn.execute("SELECT id FROM artifact WHERE kind = 'lora'").fetchone()[0]
        first = conn.execute(
            "SELECT file_id FROM file_artifact WHERE role = 'lora' ORDER BY file_id LIMIT 1"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO file_artifact(file_id, ordinal, artifact_id, role) VALUES(?, 1, ?, 'lora')",
            (first, lora_id),
        )
        conn.commit()
    finally:
        connect.close(conn)
    for slug, stars in (("pic-1", 5), ("pic-3", 4), ("pic-4", 2)):
        client.post(f"/i/{slug}/rating", json={"value": stars})


@pytest.fixture(scope="module")
def _stage(tmp_path_factory):
    with staged(tmp_path_factory, "test_an_artifact_is_a_resultset_facet", _library, _prepare) as stage:
        yield stage


@pytest.fixture
def recipes(_stage):
    _stage.restore()
    return _stage.client


def _asked(client, **params) -> dict:
    conn = connect.connect(client.app.state.db_path)
    try:
        return resultset.page(conn, "", resultset.parse(**params), 1, NOW, actor_id=client.app.state.actor_id)
    finally:
        connect.close(conn)


# --- one answer, every surface ---------------------------------------------


def test_the_artifact_page_and_the_gallery_answer_one_question(recipes):
    """/l/{slug} and /g?artifact={slug}: same ordered ids, same answer
    identity, and the shelf's aggregate count agrees -- no surface owns
    rival arithmetic."""
    page = recipes.get("/l/lora-filmgrain").json()
    direct = _asked(recipes, artifact="lora-filmgrain")
    assert [row["slug"] for row in page["gallery"]["items"]] == [row["slug"] for row in direct["items"]]
    assert page["gallery"]["answer"] == direct["answer"], "two surfaces, one answer identity"
    assert page["count"] == direct["total"]
    shelf = {row["slug"]: row["pictures"] for row in recipes.get("/loras").json()}
    assert shelf["lora-filmgrain"] == page["count"], "the shelf count IS the ResultSet total"

    # The workflow relation is invisible above ResultSet: same contract,
    # different physical join, nobody up here can tell.
    flow = recipes.get("/workflows").json()[0]
    flow_page = recipes.get(f"/w/{flow['slug']}").json()
    flow_direct = _asked(recipes, artifact=flow["slug"])
    assert [row["slug"] for row in flow_page["gallery"]["items"]] == [row["slug"] for row in flow_direct["items"]]
    assert sorted(row["slug"] for row in flow_page["gallery"]["items"]) == ["pic-2", "pic-3"]
    assert flow["pictures"] == flow_page["count"] == 2


def test_a_twice_stacked_lora_is_one_media_member(recipes):
    """file_artifact legally holds one artifact at two ordinals in one
    file; every count and every membership says one picture."""
    page = recipes.get("/l/lora-filmgrain").json()
    slugs = [row["slug"] for row in page["gallery"]["items"]]
    assert sorted(slugs) == ["pic-0", "pic-1"], "two files use the LoRA, one of them twice"
    assert page["count"] == 2
    assert {row["slug"]: row["pictures"] for row in recipes.get("/loras").json()}["lora-filmgrain"] == 2
    together = {row["slug"]: row["together"] for row in page["used_with"]}
    assert together == {"checkpoint-alpha": 2}, "synergy counts media co-occurrence, not relation rows"


def test_the_artifact_facet_is_a_conjunction(recipes):
    """artifact composes with kind, folder and the authored facets like
    any other predicate -- one real question, not a mode."""
    told = _asked(recipes, artifact="checkpoint-alpha")
    assert [row["slug"] for row in told["items"]] == ["pic-3", "pic-2", "pic-1", "pic-0"]
    rated = _asked(recipes, artifact="checkpoint-alpha", rating_min=4)
    assert sorted(row["slug"] for row in rated["items"]) == ["pic-1", "pic-3"]
    scoped = _asked(recipes, artifact="checkpoint-alpha", folder="lib", kind="image")
    assert scoped["total"] == 4
    crossed = _asked(recipes, artifact="lora-filmgrain", rating_min=4)
    assert [row["slug"] for row in crossed["items"]] == ["pic-1"]


def test_a_semantic_artifact_question_constrains_before_fusion(recipes, monkeypatch):
    """The artifact's members reach retrieval as the ALLOWED set before
    per-space ranking and RRF -- and the projection preserves the
    constrained fusion's own order, so a global-RRF-then-filter
    implementation (whose order the global ranks would dictate) fails
    this pin."""
    from db import retrieval

    conn = connect.connect(recipes.app.state.db_path)
    try:
        members = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT fa.file_id FROM file_artifact fa"
                " JOIN artifact a ON a.id = fa.artifact_id WHERE a.kind = 'lora'"
            )
        }
        witnessed: dict = {}

        def fused(conn_, models_dir, phrase, k, now, *, offline=True, allowed=None):
            witnessed["allowed"] = None if allowed is None else set(allowed)
            held = sorted(allowed or (), reverse=True)  # NOT the global order: the pin below notices reordering
            return {
                "results": [{"file_id": i, "score": 1.0, "sources": {}} for i in held],
                "participants": ["fake"],
                "contributors": ["fake"],
                "missing": {},
            }

        monkeypatch.setattr(retrieval, "query", fused)
        told = resultset.page(conn, "", resultset.parse(artifact="lora-filmgrain", text="grain"), 1, NOW)
        assert witnessed["allowed"] == members, "eligibility must reach retrieval BEFORE fusion, as the allowed set"
        assert [row["id"] for row in told["items"]] == sorted(members, reverse=True), (
            "the projection is the constrained fusion's own order, untouched by any post-filter"
        )
    finally:
        connect.close(conn)


# --- identity is not spelling ----------------------------------------------


def test_a_retired_artifact_spelling_heals_to_the_canonical_address(recipes):
    before = _asked(recipes, artifact="lora-filmgrain")
    conn = connect.connect(recipes.app.state.db_path)
    try:
        found = naming.resolve(conn, "artifact", "lora-filmgrain")
        assert found is not None
        naming.rename(conn, found[0], "film grain xl", NOW)
        conn.commit()
    finally:
        connect.close(conn)
    after = _asked(recipes, artifact="lora-filmgrain")  # the retired spelling still binds
    assert after["answer"] == before["answer"], "identity is the entity, not the spelling"
    assert "artifact=film-grain-xl" in after["qs"], "the canonical spelling heals to the live slug"
    moved = recipes.get("/m/lora-filmgrain", follow_redirects=False)
    assert (moved.status_code, moved.headers["location"]) == (301, "/l/film-grain-xl"), (
        "wrong shelf + retired slug is ONE 301 to the canonical address"
    )


def test_an_unknown_artifact_refuses_loudly(recipes):
    assert recipes.get("/g", params={"artifact": "never-was"}).status_code == 404
    assert recipes.get("/l/never-was").status_code == 404
    conn = connect.connect(recipes.app.state.db_path)
    try:
        with pytest.raises(LookupError, match="artifact"):
            resultset.describe(conn, "", resultset.parse(artifact="never-was"), NOW)
    finally:
        connect.close(conn)


# --- saved views ------------------------------------------------------------


def test_a_saved_artifact_view_is_a_rule_by_uuid(recipes):
    made = recipes.post("/albums/smart", json={"name": "Grainy", "artifact": "lora-filmgrain"})
    assert made.status_code == 201, made.text
    assert [row["slug"] for row in made.json()["gallery"]["items"]] == ["pic-1", "pic-0"]

    conn = connect.connect(recipes.app.state.db_path)
    try:
        version, told = conn.execute(
            "SELECT r.rule_version, r.rule_json FROM collection_rule r"
            " JOIN collection c ON c.id = r.collection_id WHERE c.name = 'Grainy'"
        ).fetchone()
        lora_uuid = conn.execute(
            "SELECT e.uuid FROM entity e JOIN artifact a ON a.id = e.id WHERE a.kind = 'lora'"
        ).fetchone()[0]
        assert version == 3, "this build authors v3 rules; the artifact reference arrived in v2 and stays"
        import json as json_module

        assert json_module.loads(told)["where"]["artifact"] == lora_uuid.hex(), "the rule holds the UUID, never a slug"
        # Renaming the artifact moves its address, never the membership.
        found = naming.resolve(conn, "artifact", "lora-filmgrain")
        assert found is not None
        naming.rename(conn, found[0], "film grain xl", NOW)
        conn.commit()
    finally:
        connect.close(conn)
    told = recipes.get("/t/grainy", headers={"accept": "application/json"}).json()
    assert told["state"] == "evaluated"
    assert [row["slug"] for row in told["gallery"]["items"]] == ["pic-1", "pic-0"]

    # A deleted artifact makes the rule BROKEN -- never an empty album.
    conn = connect.connect(recipes.app.state.db_path)
    try:
        conn.execute(
            "DELETE FROM entity WHERE id = (SELECT id FROM artifact WHERE kind = 'lora')",
        )
        conn.commit()
    finally:
        connect.close(conn)
    told = recipes.get("/t/grainy", headers={"accept": "application/json"}).json()
    assert told["state"] == "broken"
    assert told["gallery"] is None
    assert recipes.get("/g", params={"album": "grainy"}).status_code == 400


def test_a_v1_rule_still_means_exactly_what_it_meant(recipes):
    """The versioned format actually versions: a stored v1 row decodes
    through the v1 shape -- no artifact field, no reinterpretation."""
    import json as json_module

    conn = connect.connect(recipes.app.state.db_path)
    try:
        smart = collections.collection(conn, "Old Faithful", NOW, kind="smart")
        spelled = json_module.dumps(
            {
                "v": 1,
                "where": {"folder": None, "person": None, "kind": "image", "favorite": None, "rating_min": None},
                "select": {"sort": None, "text": None, "take": None},
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        conn.execute(
            "INSERT INTO collection_rule(collection_id, rule_version, rule_json, created_at, updated_at)"
            " VALUES(?, 1, ?, ?, ?)",
            (smart, spelled, NOW, NOW),
        )
        conn.commit()
        held = collection_rules.load(conn, smart)
        assert held is not None
        assert (held.version, held.artifact_uuid, held.kind) == (1, None, "image")
    finally:
        connect.close(conn)
    told = recipes.get("/t/old-faithful", headers={"accept": "application/json"}).json()
    assert told["state"] == "evaluated"
    assert told["count"] == 6


def test_the_durable_shape_refuses_what_this_build_cannot_mean(recipes):
    """Versioned means versioned: exact key sets per version, exact
    32-hex spellings before the whitespace-forgiving decoder -- every
    deviation is BROKEN, never an evaluation that quietly dropped the
    part this build did not understand."""
    import json as json_module

    valid_where = {"folder": None, "person": None, "artifact": None, "kind": None, "favorite": None, "rating_min": None}
    valid_select = {"sort": None, "text": None, "take": None}
    spaced = "aabbccdd eeff0011 2233445566778899"[:32]  # 32 chars, spaces inside, decodes to <16 bytes
    hostile = [
        (2, {"v": 2, "where": {**valid_where, "artifact": spaced}, "select": valid_select}),
        (1, {"v": 1, "where": valid_where, "select": valid_select}),  # v1 carrying v2's artifact key
        (
            1,
            {
                "v": 1,
                "where": dict.fromkeys(("folder", "person", "kind", "favorite", "rating_min", "moon_phase")),
                "select": valid_select,
            },
        ),
        (2, {"v": 2, "where": {k: v for k, v in valid_where.items() if k != "artifact"}, "select": valid_select}),
        (2, {"v": 2, "where": valid_where, "select": valid_select, "future": True}),
        (2, {"v": 2, "where": valid_where, "select": {**valid_select, "limit": 5}}),
    ]
    conn = connect.connect(recipes.app.state.db_path)
    try:
        smart = collections.collection(conn, "Fragile", NOW, kind="smart")
        conn.commit()
    finally:
        connect.close(conn)
    for version, payload in hostile:
        conn = connect.connect(recipes.app.state.db_path)
        try:
            conn.execute(
                "INSERT INTO collection_rule(collection_id, rule_version, rule_json, created_at, updated_at)"
                " VALUES(?, ?, ?, ?, ?)"
                " ON CONFLICT(collection_id) DO UPDATE SET rule_version = excluded.rule_version,"
                " rule_json = excluded.rule_json, updated_at = excluded.updated_at",
                (smart, version, json_module.dumps(payload, sort_keys=True, separators=(",", ":")), NOW, NOW),
            )
            conn.commit()
        finally:
            connect.close(conn)
        told = recipes.get("/t/fragile", headers={"accept": "application/json"}).json()
        assert told["state"] == "broken", f"{payload} was evaluated instead of refused"
        assert told["gallery"] is None
        assert recipes.get("/g", params={"album": "fragile"}).status_code == 400


def test_a_healed_question_saves_the_same_identity(recipes):
    """The healing crosses the save Seam: a retired artifact spelling
    whose ResultSet answer is on screen saves -- and replaces -- as the
    SAME entity uuid the live spelling would, because durable meaning is
    the identity, never the words the URL arrived with."""
    import json as json_module

    conn = connect.connect(recipes.app.state.db_path)
    try:
        found = naming.resolve(conn, "artifact", "lora-filmgrain")
        assert found is not None
        naming.rename(conn, found[0], "film grain xl", NOW)
        conn.commit()
        lora_uuid = conn.execute("SELECT uuid FROM entity WHERE id = ?", (found[0],)).fetchone()[0]
    finally:
        connect.close(conn)

    made = recipes.post("/albums/smart", json={"name": "Grainy", "artifact": "lora-filmgrain"})
    assert made.status_code == 201, made.text

    def stored() -> str:
        conn = connect.connect(recipes.app.state.db_path)
        try:
            told = conn.execute(
                "SELECT r.rule_json FROM collection_rule r JOIN collection c ON c.id = r.collection_id"
                " WHERE c.name = 'Grainy'"
            ).fetchone()[0]
        finally:
            connect.close(conn)
        return json_module.loads(told)["where"]["artifact"]

    assert stored() == lora_uuid.hex(), "the retired spelling saved a different identity than the answer on screen"

    replaced = recipes.put("/t/grainy/rule", json={"artifact": "lora-filmgrain", "rating_min": 4, "expected_rev": 1})
    assert replaced.status_code == 200, replaced.text
    assert stored() == lora_uuid.hex(), "replacing through the retired spelling forgot the healing"
    assert [row["slug"] for row in replaced.json()["gallery"]["items"]] == ["pic-1"]


# --- reuse, not implementation ----------------------------------------------


def test_selection_and_the_walk_ride_the_artifact_answer(recipes):
    """Bulk curation and MediaView walking work on an artifact answer
    with zero artifact-specific code -- the payoff for the facet being
    a facet."""
    direct = _asked(recipes, artifact="checkpoint-alpha")
    keys = [row["uuid"] for row in direct["items"][:2]]
    told = recipes.post(
        "/g/selection/favorite",
        params={"artifact": "checkpoint-alpha"},
        json={"answer": direct["answer"], "items": keys, "value": True},
    )
    assert told.status_code < 300, told.text
    assert told.json()["targets"] == 2

    walked = recipes.get("/g/locate/pic-2", params={"artifact": "checkpoint-alpha"}).json()
    assert walked["in_answer"] is True
    assert (walked["previous"], walked["next"]) == ("pic-3", "pic-1"), "the arrows walk the artifact answer"
    assert "artifact=checkpoint-alpha" in walked["qs"]
