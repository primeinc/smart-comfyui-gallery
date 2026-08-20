"""The parser's real output has to fit the schema, for every tool.

The schema contract tests check constraints against data written to satisfy
them. That proves the rules hold; it does not prove the rules describe what
actually arrives. This module runs the real metaparse adapters over the
payload shapes each tool really writes -- the same fixtures as
tests/test_metaparse.py, which mirror the format references cited in
metaparse/adapters.py -- and stores everything they produce.

A field with nowhere to go fails here. That is the point: "properly parsed
and indexed" is a claim about the seam between parser and schema, and the
seam is the only place it can be checked.

PngInfo.add_text(key, value) writes a tEXt chunk
(refs/python-pillow/Pillow/src/PIL/PngImagePlugin.py:339).
"""

import gzip
import hashlib
import io
import json
import pathlib
import sqlite3

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import metaparse
from metaparse.typed import GenerationParams

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"

# --- what each tool writes -------------------------------------------------

A1111_INFOTEXT = (
    "a castle on a hill <lora:castleLora:0.8> <lora:filmGrain:0.35>\n"
    "Negative prompt: blurry, ugly\n"
    "Steps: 20, Sampler: Euler a, Schedule type: Karras, CFG scale: 7, "
    "Seed: 12345, Size: 512x768, Model hash: abc123def, Model: dreamshaper_8, "
    'Denoising strength: 0.4, Clip skip: 2, Lora hashes: "castleLora: deadbeef", '
    "Version: v1.10.1"
)

SWARM_PARAMS = json.dumps(
    {
        "sui_image_params": {
            "prompt": "a photo of a cat",
            "negativeprompt": "dog",
            "model": "OfficialStableDiffusion/sd_xl_base_1.0",
            "seed": 1,
            "steps": 20,
            "cfgscale": 7.0,
            "width": 1024,
            "height": 1024,
            "comfyuisampler": "euler",
            "swarm_version": "0.9.3.1",
        },
        "sui_extra_data": {"date": "2025-01-25", "generation_time": "4.84 sec"},
        "sui_models": [{"name": "sd_xl_base_1.0.safetensors", "param": "model", "hash": "0xd7a9"}],
    }
)

FOOOCUS_JSON = json.dumps(
    {
        "prompt": "an astronaut riding a horse",
        "negative_prompt": "low quality",
        "base_model": "juggernautXL",
        "sampler": "dpmpp_2m_sde_gpu",
        "scheduler": "karras",
        "seed": 42,
        "steps": 30,
        "guidance_scale": 4.0,
        "width": 1152,
        "height": 896,
    }
)

INVOKEAI_METADATA = json.dumps(
    {
        "positive_prompt": "a lighthouse at dawn",
        "negative_prompt": "fog",
        "model": {"model_name": "stable-diffusion-xl", "base_model": "sdxl"},
        "scheduler": "euler",
        "seed": 777,
        "steps": 25,
        "cfg_scale": 7.5,
        "width": 1024,
        "height": 1024,
        "app_version": "4.2.0",
    }
)

NOVELAI_COMMENT = json.dumps(
    {"uc": "lowres, bad anatomy", "steps": 28, "sampler": "k_euler_ancestral", "seed": 999, "scale": 11.0}
)

COMFY_PROMPT_GRAPH = json.dumps(
    {"3": {"class_type": "KSampler", "inputs": {"seed": 5}}, "9": {"class_type": "SaveImage", "inputs": {}}}
)
COMFY_WORKFLOW = json.dumps({"nodes": [{"id": 1}], "links": [], "version": 0.4})

# Easy Diffusion writes one PNG chunk per field rather than a single payload.
EASY_DIFFUSION_CHUNKS = {
    "prompt": "a bowl of ramen",
    "negative_prompt": "text",
    "seed": "8",
    "use_stable_diffusion_model": "C:\\models\\sd-v1-5.ckpt",
    "sampler_name": "euler_a",
    "width": "512",
    "height": "512",
    "num_inference_steps": "25",
    "guidance_scale": "7.5",
}


def _png(path, chunks, mode="RGB"):
    info = PngInfo()
    for key, value in chunks.items():
        info.add_text(key, value)
    colour = (200, 100, 50) if mode == "RGB" else (200, 100, 50, 255)
    Image.new(mode, (16, 16), colour).save(path, pnginfo=info)
    return str(path)


def _stealth_png(path, text):
    """Writer port of Forge modules/stealth_infotext.py, alpha mode."""
    payload = gzip.compress(text.encode("utf-8"))
    bits = "".join(format(byte, "08b") for byte in b"stealth_pngcomp")
    data = "".join(format(byte, "08b") for byte in payload)
    bits += format(len(data), "032b") + data
    side = 1
    while side * side < len(bits):
        side += 1
    # Only the alpha channel carries payload, so the colour is a constant and
    # this builds the alpha plane alone. Forge walks columns first (x outer,
    # y inner); putdata is row-major, so the index converts rather than the
    # loop. The ground alpha is 255, hence `254 | bit`.
    alpha = [255] * (side * side)
    for index, bit in enumerate(bits):
        x, y = divmod(index, side)
        alpha[y * side + x] = 254 | int(bit)
    image = Image.new("RGBA", (side, side), (200, 100, 50, 255))
    image.putdata([(200, 100, 50, a) for a in alpha])
    image.save(path)
    return str(path)


# Every writer the adapters claim to read, one file each, with the parse
# options that writer needs. Stealth is opt-in because reading it costs a
# full pixel pass (adapters.parse_file's allow_stealth).
WRITERS = {
    "a1111": (lambda d: _png(d / "a1111.png", {"parameters": A1111_INFOTEXT}), {}),
    "swarmui": (lambda d: _png(d / "swarm.png", {"parameters": SWARM_PARAMS}), {}),
    "fooocus": (
        lambda d: _png(d / "fooocus.png", {"parameters": FOOOCUS_JSON, "fooocus_scheme": "fooocus"}),
        {},
    ),
    "invokeai": (lambda d: _png(d / "invoke.png", {"invokeai_metadata": INVOKEAI_METADATA}), {}),
    "novelai": (
        lambda d: _png(d / "novel.png", {"Comment": NOVELAI_COMMENT, "Software": "NovelAI"}),
        {},
    ),
    "comfyui": (
        lambda d: _png(d / "comfy.png", {"prompt": COMFY_PROMPT_GRAPH, "workflow": COMFY_WORKFLOW}),
        {},
    ),
    "easydiffusion": (lambda d: _png(d / "easy.png", EASY_DIFFUSION_CHUNKS), {}),
    "stealth": (lambda d: _stealth_png(d / "stealth.png", A1111_INFOTEXT), {"allow_stealth": True}),
}


# --- the schema, and one ingestion of one file into it ---------------------


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(io.open(SCHEMA, "r", encoding="utf-8", newline="").read())
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'C:/out','library',0)")
    conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(1,X'00000000000000000000000000000001','folder','out')")
    conn.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(1,1,NULL,'out',0)")
    yield conn
    conn.close()


def _name_key(text):
    return "".join(c for c in str(text).lower() if c.isalnum())


class Ingest:
    """One file, parser to rows. Reports anything that found no home.

    Deliberately not a general ingestion adapter -- that ships in db/ when it
    is written. This is the narrowest thing that can answer "does the output
    fit", so a gap here is the schema's, not an adapter's.
    """

    def __init__(self, conn):
        self.conn = conn
        self.next_id = 100
        self.artifacts = {}
        self.prompts = {}

    def _entity(self, kind, slug):
        self.next_id += 1
        i = self.next_id
        self.conn.execute(
            "INSERT INTO entity(id,uuid,kind,slug) VALUES(?,?,?,?)",
            (i, i.to_bytes(16, "big"), kind, slug),
        )
        return i

    def _artifact(self, kind, name, quoted=None):
        key = (kind, _name_key(name))
        if key not in self.artifacts:
            i = self._entity("artifact", f"{kind}-{_name_key(name)}"[:60])
            self.conn.execute(
                "INSERT INTO artifact(id,kind,name,name_key,quoted_hash,first_seen_at)"
                " VALUES(?,?,?,?,?,0)",
                (i, kind, str(name), _name_key(name), quoted),
            )
            self.artifacts[key] = i
        return self.artifacts[key]

    def _prompt(self, text):
        if not text:
            return None
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest not in self.prompts:
            i = self._entity("prompt", f"prompt-{digest[:8]}")
            self.conn.execute(
                "INSERT INTO prompt(id,text,text_hash,created_at) VALUES(?,?,?,0)",
                (i, text, digest),
            )
            self.prompts[digest] = i
        return self.prompts[digest]

    def __call__(self, written):
        path, options = written
        parsed = metaparse.parse_file(path, **options)
        assert parsed is not None, f"metaparse read nothing from {path}"
        gen = GenerationParams.from_parsed(parsed)

        file_id = self._entity("file", f"file-{self.next_id + 1}")
        self.conn.execute(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,first_seen_at,last_seen_at)"
            " VALUES(?,1,?,'image',1,1,0,0)",
            (file_id, pathlib.Path(path).name),
        )

        # The carrier is kept whether or not anything understood it, so an
        # adapter improved later re-parses the database instead of the disk.
        raw = parsed.raw or ""
        digest = hashlib.sha256(raw.encode()).hexdigest()
        self.conn.execute(
            "INSERT OR IGNORE INTO blob(hash,payload,byte_len) VALUES(?,?,?)",
            (digest, raw, len(raw.encode())),
        )
        self.conn.execute(
            "INSERT INTO file_blob(file_id,carrier,slot,blob_hash,parsed_by,seen_at)"
            " VALUES(?,'png_text','parameters',?,?,0)",
            (file_id, digest, f"metaparse/{gen.tool}"),
        )

        if gen.model:
            self.conn.execute(
                "INSERT INTO file_artifact(file_id,artifact_id,role) VALUES(?,?,'checkpoint')",
                (file_id, self._artifact("checkpoint", gen.model, gen.model_hash)),
            )
        for ordinal, lora in enumerate(gen.loras):
            self.conn.execute(
                "INSERT INTO file_artifact(file_id,ordinal,artifact_id,role,model_weight)"
                " VALUES(?,?,?,'lora',?)",
                (file_id, ordinal, self._artifact("lora", lora.get("name", "?")), lora.get("weight")),
            )

        self.conn.execute(
            "INSERT INTO generation(file_id,tool,detection,prompt_id,negative_id,seed,steps,cfg,"
            "denoise,clip_skip,sampler,scheduler,width,height,parser,parsed_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,'metaparse/1',0)",
            (
                file_id,
                gen.tool,
                gen.detection,
                self._prompt(gen.positive_prompt),
                self._prompt(gen.negative_prompt),
                gen.seed,
                gen.steps,
                gen.cfg,
                gen.denoise,
                gen.clip_skip,
                gen.sampler,
                gen.scheduler,
                gen.width,
                gen.height,
            ),
        )

        # `version` names the tool, not the picture, so it rides the long tail
        # with every other key rather than earning a column.
        tail = dict(gen.extra)
        if gen.version:
            tail["version"] = gen.version

        homeless = []
        for key, value in tail.items():
            if isinstance(value, (dict, list)):
                homeless.append((key, type(value).__name__))
                continue
            text = None if value is None else str(value)
            number = None
            if text is not None:
                try:
                    number = float(text)
                except ValueError:
                    number = None
            self.conn.execute(
                "INSERT INTO file_param(file_id,source,key,value_text,value_num)"
                " VALUES(?,'generation',?,?,?)",
                (file_id, key, text, number),
            )
        return {"file_id": file_id, "parsed": parsed, "gen": gen, "homeless": homeless}


@pytest.fixture
def ingest(db):
    return Ingest(db)


@pytest.fixture
def written(tmp_path):
    """Every writer's file, paired with the parse options it needs."""
    return {name: (build(tmp_path), options) for name, (build, options) in WRITERS.items()}


# --- the contracts ---------------------------------------------------------


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_every_writer_reaches_the_schema(db, ingest, written, writer):
    """Parse to rows, for each tool, without a constraint rejecting anything."""
    result = ingest(written[writer])
    stored = db.execute(
        "SELECT tool, detection FROM generation WHERE file_id=?", (result["file_id"],)
    ).fetchone()
    assert stored is not None, f"{writer} parsed but stored no generation row"
    assert stored[0] == result["gen"].tool
    assert stored[1] == result["gen"].detection


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_no_parsed_field_is_dropped(db, ingest, written, writer):
    """Every key the parser produced is queryable afterwards.

    The old app had one `extra` JSON column and nothing could search it.
    A key that survives parsing and then vanishes into a blob is the same
    loss with extra steps.
    """
    result = ingest(written[writer])
    assert not result["homeless"], (
        f"{writer}: {result['homeless']} has no scalar home; file_param stores "
        f"text and numbers, so a structured value needs flattening or its own table"
    )
    stored = {
        key for (key,) in db.execute(
            "SELECT key FROM file_param WHERE file_id=? AND source='generation'",
            (result["file_id"],),
        )
    }
    expected = {k for k, v in result["gen"].extra.items() if not isinstance(v, (dict, list))}
    if result["gen"].version:
        expected.add("version")
    assert expected <= stored, f"{writer} lost {expected - stored}"


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_the_carrier_survives_being_understood(db, ingest, written, writer):
    """The bytes stay, so improving an adapter re-parses the DB not the disk."""
    result = ingest(written[writer])
    row = db.execute(
        "SELECT b.byte_len, fb.parsed_by FROM file_blob fb JOIN blob b ON b.hash=fb.blob_hash"
        " WHERE fb.file_id=?",
        (result["file_id"],),
    ).fetchone()
    assert row is not None, f"{writer} kept no carrier"
    assert row[0] > 0, f"{writer} stored an empty carrier"
    assert row[1], f"{writer} recorded no parser, so the backlog query cannot tell it apart"


def test_the_checkpoint_becomes_an_artifact_not_a_string(db, ingest, written):
    """A model named by two tools is one row, joinable, not two substrings."""
    for writer in ("a1111", "stealth"):
        ingest(written[writer])
    rows = db.execute(
        "SELECT a.name, count(*) FROM file_artifact fa JOIN artifact a ON a.id=fa.artifact_id"
        " WHERE fa.role='checkpoint' GROUP BY a.id"
    ).fetchall()
    assert rows == [("dreamshaper_8", 2)], rows


def test_loras_named_in_the_prompt_become_artifacts(db, ingest, written):
    """`<lora:name:weight>` is how every A1111-family tool names a LoRA.

    If it stays inside the prompt string, /loras and LoRA synergy are back
    to matching substrings -- which is the defect the schema exists to fix.
    """
    result = ingest(written["a1111"])
    stored = db.execute(
        "SELECT a.name, fa.model_weight FROM file_artifact fa JOIN artifact a ON a.id=fa.artifact_id"
        " WHERE fa.file_id=? AND fa.role='lora' ORDER BY fa.ordinal",
        (result["file_id"],),
    ).fetchall()
    assert stored == [("castleLora", 0.8), ("filmGrain", 0.35)], (
        f"the prompt names two LoRAs and the database holds {stored}"
    )


def test_the_long_tail_registers_itself_across_tools(db, ingest, written):
    """param_key learns what the library actually contains, unprompted."""
    for writer in sorted(WRITERS):
        ingest(written[writer])
    keys = db.execute(
        "SELECT key, occurrences FROM param_key WHERE source='generation' ORDER BY key"
    ).fetchall()
    assert keys, "eight tools wrote metadata and the registry learned nothing"
    counted = dict(keys)
    actual = dict(
        db.execute(
            "SELECT key, count(*) FROM file_param WHERE source='generation' GROUP BY key"
        ).fetchall()
    )
    assert counted == actual, "the registry disagrees with the rows it is counting"
