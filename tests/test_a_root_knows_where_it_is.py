"""A root is stored absolute, because everything else is built from it.

`detect.path_of` composes a file's location as the root's recorded path
plus every folder name below it plus the file's own name. So the root's
path is not one string among many: it is the head of every path in the
library, and a relative one makes every file's location depend on which
directory the reading process happens to be in.

That is not hypothetical. The request that registers a root and the
background worker that later reads the bytes are different processes with
no reason to share a working directory, and the failure it produces is
the confusing kind: FileNotFoundError on a file the library does not
think is missing, naming a path that looks like half of one.

Those three spellings were already one root -- the marker written inside
the directory says which root it is -- so normalising at the door does
not add that, it just lets the path column answer without the file read.
"""

from __future__ import annotations

import os
import pathlib

import pytest
from PIL import Image

from db import detect, library, scan
from tests.staging import fresh_schema

pytestmark = pytest.mark.slow


@pytest.fixture
def db():
    conn = fresh_schema()
    yield conn
    conn.close()


@pytest.fixture
def library_on_disk(tmp_path):
    root = tmp_path / "lib" / "A2F" / "Bob's Shoots" / "wa lk"
    root.mkdir(parents=True)
    Image.new("RGB", (48, 36), (180, 140, 120)).save(root / "one'two.png")
    return tmp_path / "lib"


def test_a_relative_path_is_stored_absolute(db, library_on_disk, monkeypatch):
    """The defect, at the one seam that can prevent it."""
    monkeypatch.chdir(library_on_disk.parent)
    root_id = library.add_root(db, "lib", "library", 1_700_000_000.0)
    held = library.root_path(db, root_id)
    assert held is not None
    assert os.path.isabs(held), f"a root was stored as {held!r}"
    assert pathlib.Path(held) == library_on_disk


def test_a_file_is_found_from_a_directory_the_registrar_never_stood_in(db, library_on_disk, tmp_path, monkeypatch):
    """What the absolute path is FOR. The scan runs where the person
    registered the root; the job that reads the bytes runs somewhere
    else, and must still find them."""
    monkeypatch.chdir(library_on_disk.parent)
    root_id = library.add_root(db, "lib", "library", 1_700_000_000.0)
    scan.scan(db, root_id, library.root_path(db, root_id), 1_700_000_000.0)
    db.commit()

    elsewhere = tmp_path / "some" / "other" / "place"
    elsewhere.mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    for (file_id,) in db.execute("SELECT id FROM file ORDER BY id").fetchall():
        where = detect.path_of(db, file_id)
        assert os.path.isabs(where), where
        assert os.path.exists(where), f"{where} is not there, and the library does not think it is missing"


def test_three_spellings_of_one_directory_are_one_root(db, library_on_disk, monkeypatch):
    """Held by the MARKER before this change and by the path column after
    it, which is why this one passes either way. It is here because the
    normalisation must not break what the marker was already doing."""
    monkeypatch.chdir(library_on_disk.parent)
    first = library.add_root(db, "lib", "library", 1_700_000_000.0)
    assert library.add_root(db, "./lib", "library", 1_700_000_001.0) == first
    assert library.add_root(db, str(library_on_disk), "library", 1_700_000_002.0) == first
    assert db.execute("SELECT count(*) FROM root").fetchone()[0] == 1


def test_relocating_stores_an_absolute_path_too(db, library_on_disk, tmp_path, monkeypatch):
    """The other door into `root.path`, and it had the same hole."""
    monkeypatch.chdir(library_on_disk.parent)
    root_id = library.add_root(db, str(library_on_disk), "library", 1_700_000_000.0)
    moved = tmp_path / "moved"
    library_on_disk.rename(moved)
    monkeypatch.chdir(tmp_path)
    library.relocate(db, root_id, "moved")
    held = library.root_path(db, root_id)
    assert held is not None
    assert os.path.isabs(held), f"a relocated root was stored as {held!r}"
    assert pathlib.Path(held) == moved
