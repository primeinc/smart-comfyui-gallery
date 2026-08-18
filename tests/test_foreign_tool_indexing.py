"""A picture made by something other than ComfyUI still has to work.

Plenty of people run A1111 or Forge beside ComfyUI, or moved from one to
the other and kept the old output. Those files carry their generation
settings as an infotext block in the `parameters` chunk, not as a ComfyUI
graph, and the gallery reads them through a different path entirely:
metaparse's adapters rather than the workflow parser.

Every piece of that path is tested on its own -- the adapters in
test_metaparse, the typed parameters in test_metaparse_typed, the search
operators in test_prompt_search against hand-written rows. Nothing joined
them up, so nothing would notice if the seam came apart: a file that
indexes but whose parameters never reach the database looks perfectly
normal in the grid, and only stops working when someone searches
`seed:12345` and gets nothing.

This walks the seam: a real A1111 infotext, written the way A1111 writes
it, scanned, stored, and then found by each typed operator.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os

import pytest
from PIL import Image, PngImagePlugin

_PREFIX = "foreign_"
_INFOTEXT = (
    f"a {_PREFIX}photo of a cat, highly detailed\n"
    "Negative prompt: blurry, watermark\n"
    "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 12345, Size: 512x512, "
    "Model hash: abc123, Model: v1-5-pruned"
)


class _InlineExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:
            future.set_exception(exc)
        return future


@pytest.fixture
def a1111_file(smartgallery_app, monkeypatch):
    monkeypatch.setattr(smartgallery_app.concurrent.futures,
                        "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    base = smartgallery_app.BASE_OUTPUT_PATH
    path = os.path.join(base, f"{_PREFIX}pic.png")
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", _INFOTEXT)  # exactly how A1111 writes it
    Image.new("RGB", (32, 32), (5, 90, 120)).save(path, pnginfo=info)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        row = conn.execute("SELECT id FROM files WHERE name = ?",
                           (f"{_PREFIX}pic.png",)).fetchone()
    finally:
        conn.close()

    assert row, "the A1111 file was not indexed at all"
    yield row[0]

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.commit()
    finally:
        conn.close()
    with contextlib.suppress(OSError):
        os.remove(path)


def test_the_prompt_is_read_out_of_the_infotext(smartgallery_app, a1111_file):
    """Control for everything below: without this the searches could pass
    against a file that carries no metadata at all."""
    conn = smartgallery_app.get_db_connection()
    try:
        prompt = conn.execute("SELECT workflow_prompt FROM files WHERE id = ?",
                              (a1111_file,)).fetchone()[0]
    finally:
        conn.close()

    assert prompt and "photo of a cat" in prompt, repr(prompt)


def test_the_typed_parameters_reach_the_database(smartgallery_app, a1111_file):
    """The seam: metaparse parsed it, and the scan has to store it."""
    conn = smartgallery_app.get_db_connection()
    try:
        row = conn.execute(
            "SELECT tool, model, sampler, seed, steps, cfg, positive_prompt, "
            "negative_prompt FROM generation_params WHERE file_id = ?",
            (a1111_file,)).fetchone()
    finally:
        conn.close()

    assert row, "no typed parameters were stored for an A1111 file"
    tool, model, sampler, seed, steps, cfg, positive, negative = row
    assert "A1111" in tool, tool
    assert model == "v1-5-pruned"
    assert sampler == "Euler a"
    assert seed == 12345
    assert steps == 20
    assert float(cfg) == 7.0
    assert "photo of a cat" in positive
    assert negative == "blurry, watermark"


@pytest.mark.parametrize("term", ["seed:12345", "model:v1-5", "sampler:Euler",
                                  "steps:20", "cfg:7"])
def test_each_typed_operator_finds_it(smartgallery_app, a1111_file, term):
    """What a person actually does with this: search by what made it."""
    page = smartgallery_app.app.test_client().get(
        f"/galleryout/view/_root_?workflow_prompt={term}")

    assert page.status_code == 200, page.get_data(as_text=True)[:200]
    assert a1111_file in page.get_data(as_text=True), (
        f"searching {term!r} did not find the A1111 file")


def test_a_search_that_should_miss_still_misses(smartgallery_app, a1111_file):
    """Folding every operator into 'match anything' would satisfy the five
    above."""
    page = smartgallery_app.app.test_client().get(
        "/galleryout/view/_root_?workflow_prompt=seed:99999")

    assert page.status_code == 200
    assert a1111_file not in page.get_data(as_text=True)
