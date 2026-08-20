"""metaparse.typed: strict type coercion per the first-party contracts
(SwarmUI's metadata doc mandates consumer-side type forcing; A1111
infotext is stringly by format; JSON tools round-trip losslessly), the
verbatim-extra guarantee for unmappable values, DB row shape, and the
OmniQuery generation_params fields compiling to parameterized SQL."""

import json
import sqlite3

from metaparse.model import ParsedMetadata
from metaparse.typed import ROW_COLUMNS, GenerationParams, split_size, to_float, to_int
from omniquery.ast import parse_query
from omniquery.compiler import CompileParams, compile_query
from omniquery.validation import AuthContext, validate
from sqlbind import with_id_placeholders


def test_to_int_strict():
    assert to_int(5) == 5
    assert to_int("5") == 5
    assert to_int("5.0") == 5
    assert to_int(5.0) == 5
    assert to_int(5.5) is None
    assert to_int("euler") is None
    assert to_int(True) is None  # bools are not seeds
    assert to_int(None) is None


def test_to_float_strict():
    assert to_float("7.5") == 7.5
    assert to_float(7) == 7.0
    assert to_float("cfg") is None
    assert to_float(None) is None


def test_split_size():
    assert split_size("832x1216") == (832, 1216)
    assert split_size("832 X 1216") == (832, 1216)
    assert split_size("portrait") == (None, None)
    assert split_size(None) == (None, None)


def test_from_parsed_types_swarmui_stringified_values():
    """SwarmUI doc: numbers may arrive stringified; they must land typed."""
    parsed = ParsedMetadata(tool="SwarmUI", positive="a cat", negative="dog")
    parsed.params.update(
        {
            "model": "OfficialStableDiffusion/sd_xl_base_1.0",
            "seed": "123456789",
            "steps": "20",
            "cfg": "7.0",
            "size": "1024x1024",
        }
    )
    parsed.extra["automaticvae"] = "true"
    gp = GenerationParams.from_parsed(parsed)
    assert gp.seed == 123456789
    assert isinstance(gp.seed, int)
    assert gp.steps == 20
    assert gp.cfg == 7.0
    assert isinstance(gp.cfg, float)
    assert (gp.width, gp.height) == (1024, 1024)
    assert gp.negative_prompt == "dog"
    assert gp.extra["automaticvae"] == "true"  # verbatim, never dropped


def test_from_parsed_unmappable_value_stays_verbatim_in_extra():
    parsed = ParsedMetadata(tool="A1111 / Forge", positive="x")
    parsed.params.update({"seed": "12, 13", "cfg": "high", "size": "tall"})
    gp = GenerationParams.from_parsed(parsed)
    assert gp.seed is None
    assert gp.extra["seed"] == "12, 13"
    assert gp.cfg is None
    assert gp.extra["cfg"] == "high"
    assert gp.width is None
    assert gp.extra["size"] == "tall"


def test_from_comfy_typed_graph_values():
    gp = GenerationParams.from_comfy(
        {
            "seed": 42,
            "steps": 30,
            "cfg": 4.5,
            "sampler": "euler",
            "scheduler": "karras",
            "model": "flux1-dev.safetensors",
            "positive_prompt": "a red cube",
            "negative_prompt": "blurry",
            "width": 1216,
            "height": 832,
            "loras": [{"name": "detail", "weight": 0.8}],
            "positive_prompt_clean": "a red cube",
        }
    )
    assert gp.tool == "ComfyUI"
    assert gp.detection == "graph"
    assert gp.seed == 42
    assert gp.cfg == 4.5
    assert gp.loras == [{"name": "detail", "weight": 0.8}]
    assert "positive_prompt_clean" not in gp.extra


def test_to_row_round_trips_through_sqlite_with_real_types():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE generation_params (" + ", ".join(ROW_COLUMNS) + ")")
    gp = GenerationParams.from_comfy(
        {
            "seed": 7,
            "steps": 25,
            "cfg": 6.5,
            "width": 512,
            "height": 768,
            "positive_prompt": "p",
            "negative_prompt": "n",
            "model": "m",
        }
    )
    row = gp.to_row("file-1", 1000.0)
    assert len(row) == len(ROW_COLUMNS)
    conn.execute(with_id_placeholders("INSERT INTO generation_params VALUES ({ids})", ROW_COLUMNS), row)
    got = conn.execute(
        "SELECT seed, steps, cfg, width, negative_prompt FROM generation_params WHERE seed = 7"
    ).fetchone()
    assert got == (7, 25, 6.5, 512, "n")
    assert isinstance(got[0], int)
    assert isinstance(got[2], float)


def test_extra_json_serializes():
    parsed = ParsedMetadata(tool="Fooocus", positive="x")
    parsed.extra["styles"] = "['Fooocus V2']"
    gp = GenerationParams.from_parsed(parsed)
    row = gp.to_row("f", 1.0)
    extra = json.loads(row[ROW_COLUMNS.index("extra")])
    assert extra["styles"] == "['Fooocus V2']"


def test_omniquery_gen_fields_compile():
    """gen_* fields ride the existing strategies into parameterized SQL."""

    ctx = AuthContext(role="STAFF", user_id="1", client_uuid="c", ai_enabled=False)
    vq = validate(
        parse_query(
            {
                "where": {
                    "op": "and",
                    "children": [
                        {"field": "gen_seed", "op": "eq", "value": 42},
                        {"field": "gen_model", "op": "contains", "value": "flux"},
                        {"field": "gen_negative_prompt", "op": "contains", "value": "blurry"},
                    ],
                }
            }
        ),
        ctx,
    )
    cq = compile_query(
        vq, CompileParams(now_epoch=1000.0, base_path="/g", client_uuid=ctx.client_uuid, ai_resolutions={})
    )
    assert "generation_params" in cq.sql
    assert 42 in cq.params
    assert "%flux%" in cq.params
    assert "flux" not in cq.sql  # values are bound, never spliced
