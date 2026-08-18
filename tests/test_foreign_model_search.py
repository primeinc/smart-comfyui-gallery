"""Searching by model has to find pictures other tools made, too.

`workflow_files` is what the Models search box reads, and what the "models
used" list on a picture shows. Only the ComfyUI branch of the scan ever
filled it, so an A1111 or Forge picture was searchable by its prompt --
that branch already backfills workflow_prompt from the parsed metadata --
and invisible to a search for the checkpoint that made it.

The information was there the whole time: `model:dreamshaper` typed into
the prompt box found it through generation_params, while typing
`dreamshaper` into the box labelled Models found nothing. One idea, two
fields, two answers, and no reason for anyone to guess which surface knew.

Clustering never had this problem -- the foreign hashes are computed from
the parsed metadata rather than from this field -- which is also why
filling it in cannot move anybody's existing clusters.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import os

import pytest
from PIL import Image, PngImagePlugin

_PREFIX = "fmodel_"


def _infotext(model, extra_prompt=""):
    return (
        f"a {_PREFIX}photo of a cat{extra_prompt}\n"
        "Negative prompt: blurry\n"
        "Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 512x512, "
        f"Model hash: hash_{model}, Model: {model}"
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
def library(smartgallery_app, monkeypatch):
    """Two foreign pictures from different checkpoints, one with a LoRA."""
    monkeypatch.setattr(smartgallery_app.concurrent.futures, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    base = smartgallery_app.BASE_OUTPUT_PATH
    made = []
    for model, extra in (("dreamshaper_v8", " <lora:filmgrain_v2:0.6>"), ("juggernaut_xl", "")):
        name = f"{_PREFIX}{model}.png"
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", _infotext(model, extra))
        Image.new("RGB", (24, 24), (3, 3, 3)).save(os.path.join(base, name), pnginfo=info)
        made.append(name)

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.commit()
        smartgallery_app.full_sync_database(conn)
        ids = {r[0]: r[1] for r in conn.execute(f"SELECT name, id FROM files WHERE name LIKE '{_PREFIX}%'").fetchall()}
    finally:
        conn.close()

    yield ids

    conn = smartgallery_app.get_db_connection()
    try:
        conn.execute(f"DELETE FROM files WHERE name LIKE '{_PREFIX}%'")
        conn.commit()
    finally:
        conn.close()
    for name in made:
        with contextlib.suppress(OSError):
            os.remove(os.path.join(base, name))


def _search(smartgallery_app, ids, query):
    html = smartgallery_app.app.test_client().get(f"/galleryout/view/_root_?{query}").get_data(as_text=True)
    return sorted(name for name, fid in ids.items() if fid and fid in html)


def test_the_model_is_recorded_where_the_search_looks(smartgallery_app, library):
    """The regression, at its source: this field was empty."""
    conn = smartgallery_app.get_db_connection()
    try:
        rows = {
            r[0]: r[1]
            for r in conn.execute(f"SELECT name, workflow_files FROM files WHERE name LIKE '{_PREFIX}%'").fetchall()
        }
    finally:
        conn.close()

    assert "dreamshaper_v8" in rows[f"{_PREFIX}dreamshaper_v8.png"], rows
    assert "juggernaut_xl" in rows[f"{_PREFIX}juggernaut_xl.png"], rows


def test_a_lora_is_recorded_too(smartgallery_app, library):
    """A1111 writes LoRAs into the prompt itself; the ComfyUI branch records
    them as files, so the foreign branch should agree."""
    conn = smartgallery_app.get_db_connection()
    try:
        value = conn.execute(
            "SELECT workflow_files FROM files WHERE name = ?", (f"{_PREFIX}dreamshaper_v8.png",)
        ).fetchone()[0]
    finally:
        conn.close()

    assert "filmgrain_v2" in value, value


def test_the_models_box_finds_the_right_picture(smartgallery_app, library):
    """What a person does: type a checkpoint name into the Models box."""
    found = _search(smartgallery_app, library, "workflow_files=dreamshaper")

    assert found == [f"{_PREFIX}dreamshaper_v8.png"], found


def test_the_models_box_still_excludes_the_others(smartgallery_app, library):
    """Recording the model must not make every search match everything."""
    found = _search(smartgallery_app, library, "workflow_files=juggernaut")

    assert found == [f"{_PREFIX}juggernaut_xl.png"], found


def test_a_model_nobody_used_finds_nothing(smartgallery_app, library):
    assert _search(smartgallery_app, library, "workflow_files=no_such_model") == []


def test_the_typed_operator_still_works(smartgallery_app, library):
    """The surface that already worked has to keep working: both now agree
    rather than one having been swapped for the other."""
    found = _search(smartgallery_app, library, "workflow_prompt=model:juggernaut")

    assert found == [f"{_PREFIX}juggernaut_xl.png"], found


def test_clustering_is_untouched(smartgallery_app, library):
    """The foreign hashes come from the parsed metadata, not from this
    field, so filling it in must not move anybody's clusters: two different
    checkpoints still hash differently, and neither hash is empty."""
    conn = smartgallery_app.get_db_connection()
    try:
        hashes = {
            r[0]: r[1]
            for r in conn.execute(f"SELECT name, models_hash FROM files WHERE name LIKE '{_PREFIX}%'").fetchall()
        }
    finally:
        conn.close()

    values = list(hashes.values())
    assert all(values), f"a foreign file lost its models_hash: {hashes}"
    assert len(set(values)) == 2, f"two checkpoints collided: {hashes}"
