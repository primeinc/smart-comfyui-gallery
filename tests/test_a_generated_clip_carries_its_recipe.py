"""A generated clip has a recipe, and nothing could read it.

metaparse was Pillow-only. Every AI-generated video in every library
carried its workflow inside the file and no row anywhere recorded it, so
"show me the generated videos" was a question the query vocabulary could
express, the filter surface could offer, the analysis could break down --
and that always, silently, answered nothing. Not because any of those
were wrong. Because ingest had never written the row.

The fixtures here are written the way ComfyUI writes them, from its own
source: `workflow`, `prompt` and `extra_pnginfo` go in as CONTAINER
METADATA TAGS, JSON-encoded, with `movflags=use_metadata_tags` so an mp4
keeps custom tags at all
(refs/Comfy-Org/ComfyUI/comfy_api/latest/_input_impl/video_types.py:41-44,
:93-100). They are the same payloads it writes into a PNG's text chunks,
which is the whole reason this is a reader and not a second parser: the
adapters that already read a generated picture read a generated clip
without knowing a container changed.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import connect, ingest, resultset, scan
from metaparse import containers
from tests.staging import NOW

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"

#: A ComfyUI API graph, in the shape its `prompt` tag carries: node id ->
#: {class_type, inputs}. Small, and enough for the adapter to recognise.
GRAPH = {
    "3": {
        "class_type": "KSampler",
        "inputs": {"seed": 987654, "steps": 24, "cfg": 6.5, "sampler_name": "euler", "scheduler": "simple"},
    },
    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "wan2_1_t2v.safetensors"}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a paper boat going over a waterfall"}},
    "9": {"class_type": "SaveAnimatedWEBP", "inputs": {}},
}
WORKFLOW = {"nodes": [{"id": 3, "type": "KSampler"}], "links": [], "version": 0.4}


def _clip(path: pathlib.Path, metadata: dict[str, str] | None = None) -> None:
    """A real, playable mp4 -- with ComfyUI's tags when asked for."""
    import av

    # `movflags=use_metadata_tags` is not optional and not decoration:
    # without it the mp4 muxer DROPS every tag it does not recognise, so
    # `workflow` and `prompt` never reach the file. ComfyUI passes exactly
    # this, for exactly this reason
    # (refs/Comfy-Org/ComfyUI/comfy_api/latest/_input_impl/video_types.py:41).
    # A fixture that omitted it would be testing a clip ComfyUI never
    # writes, and would have proved the reader broken when it was not.
    with av.open(str(path), "w", options={"movflags": "use_metadata_tags"}) as container:
        if metadata:
            for key, value in metadata.items():
                container.metadata[key] = value
        stream = container.add_stream("h264", rate=6)
        stream.width, stream.height = 320, 180
        stream.pix_fmt = "yuv420p"
        for i in range(6):
            frame = av.VideoFrame.from_ndarray(
                np.full((180, 320, 3), (10 * i, 40, 200), dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


# --- the reader -------------------------------------------------------------


def test_the_container_reader_finds_what_comfyui_wrote(tmp_path):
    """The narrow claim: the tags go in, and they come back out in the
    same shape a PNG's text chunks arrive in."""
    made = tmp_path / "boat.mp4"
    _clip(made, {"prompt": json.dumps(GRAPH), "workflow": json.dumps(WORKFLOW)})

    raw = containers.load_raw_video(str(made))
    assert raw is not None, "the container would not open"
    assert raw.text.get("prompt"), "ComfyUI's API graph is where it put it"
    assert json.loads(raw.text["prompt"])["3"]["class_type"] == "KSampler"
    assert json.loads(raw.text["workflow"])["version"] == 0.4
    assert (raw.width, raw.height) == (320, 180)
    # A container carries no EXIF at all. That is a fact about the format
    # and not a read that failed, and the difference is what lets ingest
    # skip re-opening the file.
    assert raw.exif_state == "absent"


def test_a_clip_with_nothing_in_it_says_nothing(tmp_path):
    """An honest empty. A plain clip is not a damaged one."""
    made = tmp_path / "plain.mp4"
    _clip(made)
    raw = containers.load_raw_video(str(made))
    assert raw is not None
    assert not raw.text.get("prompt")
    assert not raw.text.get("workflow")


def test_a_file_that_is_not_a_container_is_refused_quietly(tmp_path):
    """Nothing here may raise into ingest: a scan crosses whatever is on
    the disk, and half of it lies about what it is."""
    liar = tmp_path / "notreally.mp4"
    liar.write_bytes(b"this is not a video")
    assert containers.load_raw_video(str(liar)) is None


# --- through ingest, into the query vocabulary ------------------------------


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "pics"
    root.mkdir()
    _clip(root / "boat.mp4", {"prompt": json.dumps(GRAPH), "workflow": json.dumps(WORKFLOW)})
    _clip(root / "handheld.mp4")
    info = PngInfo()
    info.add_text("prompt", json.dumps(GRAPH))
    Image.new("RGB", (32, 24), (80, 20, 20)).save(root / "still.png", pnginfo=info)

    conn = connect.memory()
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,?,'library',0)", (str(root),))
    scan.scan(conn, 1, root, NOW)
    for file_id, name in conn.execute("SELECT id, name FROM file").fetchall():
        ingest.one(conn, file_id, root / name, NOW)
    conn.commit()
    yield conn
    conn.close()


def _total(conn, **kwargs) -> int:
    return resultset.describe(conn, "", resultset.parse(**kwargs), NOW)["total"]


def test_a_generated_clip_gets_a_generation_row(library):
    """The row that was never written."""
    row = library.execute(
        "SELECT g.tool, g.seed, g.steps FROM generation g JOIN file f ON f.id = g.file_id WHERE f.name = 'boat.mp4'"
    ).fetchone()
    assert row is not None, "the clip carried its whole recipe and ingest recorded none of it"
    tool, seed, steps = row
    assert tool, "something read it, and says what"
    assert (seed, steps) == (987654, 24), (seed, steps)


def test_the_question_that_used_to_answer_nothing(library):
    """The headline: every video that was generated.

    Before the reader this was 0 for every library that ever existed --
    the filter was right, the vocabulary was right, and the row was
    missing.
    """
    assert _total(library, kind="video") == 2
    assert _total(library, kind="video", facets=["has.generation:eq:1"]) == 1
    assert _total(library, kind="video", facets=["has.generation:eq:0"]) == 1, "the handheld one, honestly"
    # and it did not start claiming stills are clips, or vice versa
    assert _total(library, kind="image", facets=["has.generation:eq:1"]) == 1


def test_a_clip_and_a_still_of_one_graph_are_read_identically(library):
    """The claim that matters, and the reason this is a reader rather
    than a second parser.

    The same ComfyUI graph in a PNG text chunk and in an mp4 metadata
    tag must produce the same rows, because the adapter never learns
    which container it came out of. Asserted as PARITY rather than
    against a fixed expectation: what a given graph yields is the
    adapter's business and may improve, but the two must move together
    or a clip is a second-class file again.
    """
    rows = dict(
        library.execute(
            "SELECT f.name, g.tool || '|' || COALESCE(g.seed,'') || '|' || COALESCE(g.steps,'')"
            "  || '|' || COALESCE(g.cfg,'') || '|' || COALESCE(g.sampler,'')"
            " FROM generation g JOIN file f ON f.id = g.file_id"
        ).fetchall()
    )
    assert rows.get("boat.mp4") == rows.get("still.png") is not None, rows

    # and the same for what each hung off itself: prompts by role, and
    # artifacts by role. This library's graph carries no links, so the
    # adapter resolves no prompt ROLE from it -- for either file. That is
    # a fact about the graph, and the point here is that it is the SAME
    # fact about both.
    def held(name: str) -> tuple:
        prompts = library.execute(
            "SELECT gp.role, p.text FROM generation_prompt gp JOIN prompt p ON p.id = gp.prompt_id"
            " JOIN file f ON f.id = gp.file_id WHERE f.name = ? ORDER BY gp.role",
            (name,),
        ).fetchall()
        artifacts = library.execute(
            "SELECT fa.role, a.name FROM file_artifact fa JOIN artifact a ON a.id = fa.artifact_id"
            " JOIN file f ON f.id = fa.file_id WHERE f.name = ? ORDER BY fa.role, a.name",
            (name,),
        ).fetchall()
        return (prompts, artifacts)

    assert held("boat.mp4") == held("still.png")
    assert held("boat.mp4")[1], "the graph named a checkpoint and both files recorded it"


def test_the_checkpoint_is_an_artifact_a_clip_shares_with_a_picture(library):
    """The whole point of one vocabulary: a model used for a video and a
    model used for a still are one artifact, so a question about it
    answers across both."""
    held = library.execute(
        "SELECT a.name, COUNT(DISTINCT fa.file_id) FROM artifact a"
        " JOIN file_artifact fa ON fa.artifact_id = a.id AND fa.role = 'checkpoint'"
        " GROUP BY a.id"
    ).fetchall()
    assert held, "no checkpoint was recorded for either the clip or the still"
    names = {name for name, _ in held}
    assert any("wan" in name.lower() for name in names), names
    # the clip and the still name the same checkpoint, so it is ONE row
    assert [count for name, count in held if "wan" in name.lower()] == [2]
