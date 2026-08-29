"""metaparse: marker-first detection, per-tool field mapping, stealth decode.

Fixture payloads mirror what each tool actually writes (see the format
references in metaparse/adapters.py).
"""

import gzip
import json

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import metaparse
from metaparse.containers import decode_user_comment
from metaparse.model import ParsedMetadata


def parse_file(path, **kwargs):
    """metaparse.parse_file with "it parsed at all" asserted once, so every
    test reads fields instead of re-proving Optional-ness."""
    parsed = metaparse.parse_file(path, **kwargs)
    assert parsed is not None, f"{path} did not parse"
    return parsed


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

A1111_INFOTEXT = (
    "a castle on a hill <lora:castleLora:0.8>\n"
    "Negative prompt: blurry, ugly\n"
    "Steps: 20, Sampler: Euler a, Schedule type: Karras, CFG scale: 7, "
    "Seed: 12345, Size: 512x768, Model hash: abc123def, Model: dreamshaper_8, "
    'Denoising strength: 0.4, Clip skip: 2, Lora hashes: "castleLora: deadbeef", '
    "Version: v1.10.1"
)

COMFY_PROMPT_GRAPH = json.dumps(
    {
        "3": {"class_type": "KSampler", "inputs": {"seed": 5}},
        "9": {"class_type": "SaveImage", "inputs": {}},
    }
)
COMFY_WORKFLOW = json.dumps({"nodes": [{"id": 1}], "links": [], "version": 0.4})

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
    {
        "uc": "lowres, bad anatomy",
        "steps": 28,
        "sampler": "k_euler_ancestral",
        "seed": 999,
        "scale": 11.0,
    }
)

DRAWTHINGS_XMP = (
    '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
    '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF '
    'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    '<rdf:Description xmlns:exif="http://ns.adobe.com/exif/1.0/">'
    "<exif:UserComment><rdf:Alt><rdf:li>"
    + json.dumps(
        {
            "c": "a watercolor fox",
            "uc": "photo",
            "model": "dreamshaper_v8",
            "sampler": "DPM++ 2M Karras",
            "seed": 31337,
            "scale": 6.5,
            "steps": 22,
            "size": "512x512",
        }
    )
    + "</rdf:li></rdf:Alt></exif:UserComment></rdf:Description></rdf:RDF></x:xmpmeta>"
    '<?xpacket end="w"?>'
)


def make_png(path, chunks: dict, size=(16, 16), mode="RGB", itxt=()):
    info = PngInfo()
    for key, value in chunks.items():
        if key in itxt:
            info.add_itxt(key, value)
        else:
            info.add_text(key, value)
    Image.new(mode, size, (200, 100, 50) if mode == "RGB" else (200, 100, 50, 255)).save(path, pnginfo=info)
    return str(path)


def make_webp_exif(path, tags: dict, size=(16, 16)):
    img = Image.new("RGB", size, (10, 20, 30))
    exif = Image.Exif()
    for tag, value in tags.items():
        exif[tag] = value
    img.save(path, format="WEBP", exif=exif)
    return str(path)


def embed_stealth_alpha(img: Image.Image, text: str, compressed=True) -> Image.Image:
    """Writer port of Forge modules/stealth_infotext.py (alpha mode)."""
    sig = "stealth_pngcomp" if compressed else "stealth_pnginfo"
    payload = gzip.compress(text.encode("utf-8")) if compressed else text.encode("utf-8")
    bits = "".join(format(b, "08b") for b in sig.encode("utf-8"))
    data_bits = "".join(format(b, "08b") for b in payload)
    bits += format(len(data_bits), "032b") + data_bits
    img = img.convert("RGBA")
    pixels = img.load()
    assert pixels is not None
    width, height = img.size
    assert len(bits) <= width * height, "test image too small for payload"
    index = 0
    for x in range(width):
        for y in range(height):
            if index >= len(bits):
                return img
            pixel = pixels[x, y]
            assert isinstance(pixel, tuple)
            r, g, b, a = pixel
            pixels[x, y] = (r, g, b, (a & ~1) | int(bits[index]))
            index += 1
    return img


# ---------------------------------------------------------------------------
# adapter detection + field mapping
# ---------------------------------------------------------------------------


def test_a1111_png_marker(tmp_path):
    path = make_png(tmp_path / "a.png", {"parameters": A1111_INFOTEXT})
    parsed = parse_file(path)
    assert parsed.tool == "A1111 / Forge"
    assert parsed.detection == "marker"
    assert parsed.positive == "a castle on a hill <lora:castleLora:0.8>"
    assert parsed.negative == "blurry, ugly"
    assert parsed.params["seed"] == "12345"
    assert parsed.params["cfg"] == "7"
    assert parsed.params["size"] == "512x768"
    assert parsed.params["model"] == "dreamshaper_8"
    assert parsed.params["model_hash"] == "abc123def"
    assert parsed.params["scheduler"] == "Karras"
    assert parsed.params["denoise"] == "0.4"
    assert parsed.params["clip_skip"] == "2"
    assert parsed.params["version"] == "v1.10.1"
    # quoted value survives the comma inside it
    assert parsed.extra["Lora hashes"] == "castleLora: deadbeef"


def test_swarmui_png_marker(tmp_path):
    path = make_png(tmp_path / "s.png", {"parameters": SWARM_PARAMS})
    parsed = parse_file(path)
    assert parsed.tool == "SwarmUI"
    assert parsed.positive == "a photo of a cat"
    assert parsed.negative == "dog"
    assert parsed.params["cfg"] == "7.0"
    assert parsed.params["size"] == "1024x1024"
    assert parsed.params["sampler"] == "euler"
    assert parsed.params["version"] == "0.9.3.1"
    assert parsed.extra["generation_time"] == "4.84 sec"
    # `sui_models` is a manifest, not a sentence: joining it into one
    # comma-separated string throws away the role, which tells a checkpoint
    # from a LoRA, and the hash, which is what matches a weight file on disk.
    assert parsed.artifacts == [
        {
            "name": "sd_xl_base_1.0.safetensors",
            "role": "checkpoint",
            "hash": "0xd7a9",
        }
    ]


def test_swarmui_beats_a1111_on_shared_chunk_name(tmp_path):
    # Both tools write a chunk literally named `parameters`; the sui marker wins.
    path = make_png(tmp_path / "s2.png", {"parameters": SWARM_PARAMS})
    assert parse_file(path).tool == "SwarmUI"


def test_fooocus_a1111_scheme(tmp_path):
    path = make_png(
        tmp_path / "f.png",
        {"parameters": A1111_INFOTEXT, "fooocus_scheme": "a1111"},
    )
    parsed = parse_file(path)
    assert parsed.tool == "Fooocus"
    assert parsed.params["seed"] == "12345"


def test_fooocus_json_scheme(tmp_path):
    path = make_png(
        tmp_path / "fj.png",
        {"parameters": FOOOCUS_JSON, "fooocus_scheme": "fooocus"},
    )
    parsed = parse_file(path)
    assert parsed.tool == "Fooocus"
    assert parsed.positive == "an astronaut riding a horse"
    assert parsed.params["model"] == "juggernautXL"
    assert parsed.params["cfg"] == "4.0"
    assert parsed.params["size"] == "1152x896"


def test_fooocus_legacy_comment_chunk(tmp_path):
    path = make_png(tmp_path / "fl.png", {"Comment": FOOOCUS_JSON})
    parsed = parse_file(path)
    assert parsed.tool == "Fooocus"
    assert parsed.negative == "low quality"


def test_invokeai_v3(tmp_path):
    path = make_png(tmp_path / "i.png", {"invokeai_metadata": INVOKEAI_METADATA})
    parsed = parse_file(path)
    assert parsed.tool == "InvokeAI"
    assert parsed.positive == "a lighthouse at dawn"
    assert parsed.params["model"] == "stable-diffusion-xl"
    assert parsed.params["sampler"] == "euler"
    assert parsed.params["size"] == "1024x1024"
    assert parsed.extra["app_version"] == "4.2.0"


def test_invokeai_dream_legacy(tmp_path):
    dream = '"a boat [storm]" -s 50 -S 1234 -W 512 -H 512 -C 7.5 -A k_lms'
    path = make_png(tmp_path / "d.png", {"Dream": dream})
    parsed = parse_file(path)
    assert parsed.tool == "InvokeAI"
    assert parsed.positive == "a boat"
    assert parsed.negative == "storm"
    assert parsed.params["steps"] == "50"
    assert parsed.params["sampler"] == "k_lms"
    assert parsed.params["size"] == "512x512"


def test_novelai_legacy(tmp_path):
    path = make_png(
        tmp_path / "n.png",
        {"Software": "NovelAI", "Description": "1girl, best quality", "Comment": NOVELAI_COMMENT},
        size=(64, 48),
    )
    parsed = parse_file(path)
    assert parsed.tool == "NovelAI"
    assert parsed.positive == "1girl, best quality"
    assert parsed.negative == "lowres, bad anatomy"
    assert parsed.params["cfg"] == "11.0"
    assert parsed.params["size"] == "64x48"


def test_easydiffusion_chunks(tmp_path):
    path = make_png(
        tmp_path / "e.png",
        {
            "prompt": "a desert oasis",
            "negative_prompt": "cactus",
            "seed": "2024",
            "use_stable_diffusion_model": "C:\\models\\sd-v1-5.ckpt",
            "sampler_name": "euler_a",
            "width": "512",
            "height": "512",
            "num_inference_steps": "25",
            "guidance_scale": "7.5",
        },
    )
    parsed = parse_file(path)
    assert parsed.tool == "Easy Diffusion"
    assert parsed.positive == "a desert oasis"
    assert parsed.params["model"] == "sd-v1-5.ckpt"
    assert parsed.params["steps"] == "25"


def test_drawthings_xmp(tmp_path):
    path = make_png(
        tmp_path / "dt.png",
        {"XML:com.adobe.xmp": DRAWTHINGS_XMP},
        itxt=("XML:com.adobe.xmp",),
    )
    parsed = parse_file(path)
    assert parsed.tool == "Draw Things"
    assert parsed.positive == "a watercolor fox"
    assert parsed.params["cfg"] == "6.5"
    assert parsed.params["size"] == "512x512"


def test_comfyui_graph_detected_not_rendered(tmp_path):
    path = make_png(tmp_path / "c.png", {"prompt": COMFY_PROMPT_GRAPH, "workflow": COMFY_WORKFLOW})
    parsed = parse_file(path)
    assert parsed.tool == "ComfyUI"
    assert metaparse.render_report(parsed) is None  # workflow panel owns this


def test_comfyui_a1111_compatible(tmp_path):
    path = make_png(
        tmp_path / "ca.png",
        {"prompt": COMFY_PROMPT_GRAPH, "parameters": A1111_INFOTEXT},
    )
    parsed = parse_file(path)
    assert parsed.tool == "ComfyUI (A1111-compatible)"
    assert parsed.params["seed"] == "12345"
    assert metaparse.render_report(parsed)


def test_comfyui_webp_exif_tags(tmp_path):
    path = make_webp_exif(
        tmp_path / "c.webp",
        {0x0110: "prompt:" + COMFY_PROMPT_GRAPH, 0x010F: "workflow:" + COMFY_WORKFLOW},
    )
    parsed = parse_file(path)
    assert parsed.tool == "ComfyUI"


def test_swarmui_legacy_webp_model_tag(tmp_path):
    path = make_webp_exif(tmp_path / "s.webp", {0x0110: SWARM_PARAMS})
    parsed = parse_file(path)
    assert parsed.tool == "SwarmUI"
    assert parsed.positive == "a photo of a cat"


def test_no_metadata_returns_none(tmp_path):
    path = make_png(tmp_path / "plain.png", {})
    assert metaparse.parse_file(path, allow_stealth=True) is None


# ---------------------------------------------------------------------------
# stealth
# ---------------------------------------------------------------------------


def test_stealth_forge_infotext(tmp_path):
    img = embed_stealth_alpha(Image.new("RGB", (96, 96), (120, 60, 30)), A1111_INFOTEXT)
    path = str(tmp_path / "st.png")
    img.save(path)
    assert metaparse.parse_file(path, allow_stealth=False) is None
    parsed = parse_file(path, allow_stealth=True)
    assert parsed.tool == "A1111 / Forge (stealth)"
    assert parsed.detection == "stealth"
    assert parsed.params["seed"] == "12345"


def test_stealth_novelai_json(tmp_path):
    payload = json.dumps(
        {
            "Description": "1girl",
            "Software": "NovelAI",
            "Comment": NOVELAI_COMMENT,
        }
    )
    img = embed_stealth_alpha(Image.new("RGB", (96, 96), (5, 5, 5)), payload)
    path = str(tmp_path / "stn.png")
    img.save(path)
    parsed = parse_file(path, allow_stealth=True)
    assert parsed is not None
    assert parsed.tool == "NovelAI (stealth)"
    assert parsed.negative == "lowres, bad anatomy"


def test_stealth_uncompressed_variant(tmp_path):
    img = embed_stealth_alpha(Image.new("RGB", (128, 128), (9, 9, 9)), A1111_INFOTEXT, compressed=False)
    path = str(tmp_path / "stu.png")
    img.save(path)
    parsed = parse_file(path, allow_stealth=True)
    assert parsed is not None
    assert parsed.params["seed"] == "12345"


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------


def test_decode_user_comment_variants():
    text = "hello, Steps: 20"
    assert decode_user_comment(b"UNICODE\x00" + text.encode("utf-16-be")) == text
    assert decode_user_comment(b"UNICODE\x00" + text.encode("utf-16-le")) == text
    assert decode_user_comment(b"ASCII\x00\x00\x00" + text.encode("ascii")) == text
    assert decode_user_comment(text.encode("utf-8")) == text
    assert decode_user_comment(None) is None
    assert decode_user_comment(b"") is None


def test_render_report_sections():
    parsed = metaparse.parse_infotext(A1111_INFOTEXT, "A1111 / Forge")
    report = metaparse.render_report(parsed)
    assert report is not None
    assert "=== A1111 / Forge Generation Parameters ===" in report
    assert "castleLora (Strength: 0.8)" in report
    assert "Resolution: 512x768" in report
    assert "Checkpoint: dreamshaper_8" in report


def test_render_report_empty_parse_is_none():
    assert metaparse.render_report(ParsedMetadata(tool="ComfyUI")) is None


# ---------------------------------------------------------------------------
# gallery wiring: non-ComfyUI prompts reach the indexed prompt column
# ---------------------------------------------------------------------------
