"""Downloading a selection has to hand back everything that was selected.

Each file went into the zip under its bare filename. Two entries sharing
a name is not an error a zip reports -- both are written, and whoever
opens it keeps whichever came last. ComfyUI numbers its output per
folder, so ComfyUI_00001_.png exists once in every folder anybody has,
and selecting across folders is the ordinary way to use a gallery.

Measured before the fix, on three folders each holding that name:

    warnings raised while building the zip: 2
       UserWarning: Duplicate name: 'ComfyUI_00001_.png'
       UserWarning: Duplicate name: 'ComfyUI_00001_.png'
    job says: {'status': 'ready', 'filename': 'smartgallery_probe.zip'}
    entries inside the zip : 3
    files after extracting : 1
    selected 3, received 1

Nothing was reported. The job said ready, the download worked, and two of
the three pictures were not in it.

A name nothing else in the selection shares is untouched, so a download
from one folder is byte for byte what it was. A name that repeats is
qualified on every copy, including the first: qualifying only the later
ones leaves one arbitrary file holding the bare name, and no way to tell
which folder that one came from.
"""

from __future__ import annotations

import os
import zipfile


import smartgallery


def _names(pairs):
    return smartgallery.zip_entry_names(pairs)


def test_a_single_folder_download_is_unchanged():
    """Over-reach guard, and the case almost every download is. These
    names must come back exactly as they went in."""
    rows = [("/lib/a/one.png", "one.png"),
            ("/lib/a/two.png", "two.png"),
            ("/lib/a/three.mp4", "three.mp4")]

    assert _names(rows) == ["one.png", "two.png", "three.mp4"]


def test_every_copy_of_a_repeated_name_says_where_it_came_from():
    """The bug. All three are ComfyUI_00001_.png."""
    rows = [("/lib/portraits/ComfyUI_00001_.png", "ComfyUI_00001_.png"),
            ("/lib/landscapes/ComfyUI_00001_.png", "ComfyUI_00001_.png"),
            ("/lib/drafts/ComfyUI_00001_.png", "ComfyUI_00001_.png")]

    names = _names(rows)

    assert len(set(names)) == 3, f"names still collide: {names}"
    assert names == ["ComfyUI_00001_ (portraits).png",
                     "ComfyUI_00001_ (landscapes).png",
                     "ComfyUI_00001_ (drafts).png"]


def test_nothing_keeps_a_bare_name_while_its_twin_is_qualified():
    """Qualifying from the second occurrence onwards would pass the
    uniqueness check above and still leave one file unattributable."""
    rows = [("/lib/portraits/shot.png", "shot.png"),
            ("/lib/drafts/shot.png", "shot.png")]

    names = _names(rows)

    assert "shot.png" not in names, (
        f"one copy kept the bare name, so which folder it came from is "
        f"anyone's guess: {names}")


def test_two_files_from_one_folder_with_one_name_still_both_arrive():
    """The folder cannot tell these apart -- the same name twice in one
    place, which a rename or a mount overlap can produce. Uniqueness is
    the requirement; the qualifier is a nicety."""
    rows = [("/lib/a/dup.png", "dup.png"),
            ("/lib/a/dup.png", "dup.png"),
            ("/lib/a/dup.png", "dup.png")]

    names = _names(rows)

    assert len(set(names)) == 3, names


def test_the_count_out_always_equals_the_count_in():
    """The property, over a mixed selection: whatever goes in, that many
    distinct names come out, in the same order."""
    rows = [("/lib/a/x.png", "x.png"),
            ("/lib/b/x.png", "x.png"),
            ("/lib/a/y.png", "y.png"),
            ("/lib/c/x.png", "x.png"),
            ("/lib/b/y.png", "y.png"),
            ("/lib/a/z.mov", "z.mov")]

    names = _names(rows)

    assert len(names) == len(rows)
    assert len(set(names)) == len(rows), names
    assert names[-1] == "z.mov", "an unrepeated name was changed anyway"


def test_extensions_survive_being_qualified():
    """The suffix goes before the extension, or the file stops opening."""
    rows = [("/lib/a/clip.tar.gz", "clip.tar.gz"),
            ("/lib/b/clip.tar.gz", "clip.tar.gz")]

    for name in _names(rows):
        assert name.endswith(".gz"), name


def test_the_zip_that_gets_built_holds_them_all(smartgallery_app, tmp_path,
                                                monkeypatch):
    """End to end through the real job, because the naming is only half of
    it -- the writing has to use it."""
    monkeypatch.setattr(smartgallery, "ZIP_CACHE_DIR", str(tmp_path / "zips"))

    made = []
    for folder in ("portraits", "landscapes", "drafts"):
        directory = tmp_path / folder
        directory.mkdir()
        target = directory / "ComfyUI_00001_.png"
        target.write_bytes(b"png-" + folder.encode())
        made.append(str(target))

    conn = smartgallery.get_db_connection()
    ids = [f"zipdup{i:026d}" for i in range(len(made))]
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO files (id, path, mtime, name, type) "
            "VALUES (?,?,?,?,?)",
            [(ids[i], made[i], 1700000000.0, "ComfyUI_00001_.png", "image")
             for i in range(len(made))])
        conn.commit()
    finally:
        conn.close()

    try:
        smartgallery.background_zip_task("testjob", ids)

        status = smartgallery.zip_jobs.get("testjob")
        assert status and status.get("status") == "ready", status

        archive = os.path.join(smartgallery.ZIP_CACHE_DIR, status["filename"])
        with zipfile.ZipFile(archive) as zf:
            entries = zf.namelist()
            zf.extractall(str(tmp_path / "out"))

        assert len(entries) == 3, entries
        assert len(set(entries)) == 3, f"the zip holds a repeated name: {entries}"

        extracted = sorted(os.listdir(str(tmp_path / "out")))
        assert len(extracted) == 3, (
            f"selected 3, extracted {len(extracted)}: {extracted}")

        contents = set()
        for name in extracted:
            with open(os.path.join(str(tmp_path / "out"), name), "rb") as f:
                contents.add(f.read())
        assert contents == {b"png-portraits", b"png-landscapes", b"png-drafts"}, (
            "the files arrived but some are copies of each other")
    finally:
        smartgallery.zip_jobs.pop("testjob", None)
        conn = smartgallery.get_db_connection()
        try:
            conn.executemany("DELETE FROM files WHERE id = ?",
                             [(i,) for i in ids])
            conn.commit()
        finally:
            conn.close()


def test_a_single_folder_zip_still_has_the_original_names(smartgallery_app,
                                                          tmp_path, monkeypatch):
    """Over-reach guard through the real job, so it holds against the
    build before this change as well as after it. Names that do not clash
    must arrive exactly as they were -- a fix that renamed everything
    would satisfy every uniqueness check in this file."""
    monkeypatch.setattr(smartgallery, "ZIP_CACHE_DIR", str(tmp_path / "zips"))

    directory = tmp_path / "one_folder"
    directory.mkdir()
    wanted = ["alpha.png", "beta.png", "gamma.mp4"]
    for name in wanted:
        (directory / name).write_bytes(b"data-" + name.encode())

    ids = [f"zipone{i:026d}" for i in range(len(wanted))]
    conn = smartgallery.get_db_connection()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO files (id, path, mtime, name, type) "
            "VALUES (?,?,?,?,?)",
            [(ids[i], str(directory / wanted[i]), 1700000000.0, wanted[i],
              "image") for i in range(len(wanted))])
        conn.commit()
    finally:
        conn.close()

    try:
        smartgallery.background_zip_task("testjob_one", ids)
        status = smartgallery.zip_jobs.get("testjob_one")
        assert status and status.get("status") == "ready", status

        archive = os.path.join(smartgallery.ZIP_CACHE_DIR, status["filename"])
        with zipfile.ZipFile(archive) as zf:
            assert sorted(zf.namelist()) == sorted(wanted), zf.namelist()
    finally:
        smartgallery.zip_jobs.pop("testjob_one", None)
        conn = smartgallery.get_db_connection()
        try:
            conn.executemany("DELETE FROM files WHERE id = ?",
                             [(i,) for i in ids])
            conn.commit()
        finally:
            conn.close()


def test_a_duplicate_name_would_have_been_lost(tmp_path):
    """Control. Everything above asserts that files survive; that means
    nothing unless writing them the old way loses them."""
    source = tmp_path / "src"
    source.mkdir()
    files = []
    for folder in ("one", "two"):
        directory = source / folder
        directory.mkdir()
        target = directory / "same.png"
        target.write_bytes(b"from-" + folder.encode())
        files.append(str(target))

    archive = tmp_path / "old.zip"
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with zipfile.ZipFile(archive, "w") as zf:
            for path in files:
                zf.write(path, os.path.basename(path))  # the old way

    assert any("Duplicate name" in str(w.message) for w in caught), (
        "zipfile no longer warns about duplicate names")

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(str(tmp_path / "old_out"))
    assert len(os.listdir(str(tmp_path / "old_out"))) == 1, (
        "two entries with one name no longer collapse on extraction, so the "
        "checks above are guarding something that cannot happen")
