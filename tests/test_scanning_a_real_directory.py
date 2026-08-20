"""What a scan does to authored state when the disk changes behind it.

Every case here is performed on a real directory with real files, because
the failures this guards against are filesystem behaviours -- case
insensitivity, mtime granularity, a name reused by different bytes -- and a
fake filesystem is exactly where those stop being observable.

The old app lost data here: a file moved by Explorer rather than by the UI
was seen as a delete plus an add, and the cascade took its ratings, comments
and album membership with it (smartgallery.py:5121-5129). Each test below
changes something on disk, rescans, and asks whether the human's work is
still attached to the same picture.
"""

import io
import pathlib
import sqlite3

import pytest

from db import scan

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"
NOW = 1_700_000_000.0


@pytest.fixture
def library(tmp_path):
    """An empty database and an empty directory, wired to each other."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(io.open(SCHEMA, "r", encoding="utf-8", newline="").read())
    conn.execute("PRAGMA foreign_keys=ON")
    root = tmp_path / "library"
    root.mkdir()
    conn.execute(
        "INSERT INTO root(id, path, kind, created_at) VALUES(1, ?, 'library', 0)",
        (str(root),),
    )
    conn.execute(
        "INSERT INTO user(id, username, password_hash, role, created_at)"
        " VALUES(1, 'will', 'x', 'ADMIN', 0)"
    )
    yield conn, root
    conn.close()


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)
    return path


def rescan(conn, root, now=NOW):
    return scan.scan(conn, 1, root, now)


def file_row(conn, sha):
    return conn.execute(
        "SELECT f.id, f.name, fo.name, f.missing_since FROM file f"
        " JOIN folder fo ON fo.id = f.folder_id WHERE f.content_sha256 = ?",
        (sha,),
    ).fetchone()


def authored_state(conn, file_id):
    """The 4-tuple a human built by hand, and the thing that must survive."""
    rating = conn.execute("SELECT rating FROM rating WHERE file_id = ?", (file_id,)).fetchone()
    comments = conn.execute(
        "SELECT count(*) FROM comment WHERE file_id = ?", (file_id,)
    ).fetchone()[0]
    favorite = conn.execute(
        "SELECT count(*) FROM favorite WHERE file_id = ?", (file_id,)
    ).fetchone()[0]
    albums = conn.execute(
        "SELECT count(*) FROM collection_file WHERE file_id = ?", (file_id,)
    ).fetchone()[0]
    return (rating[0] if rating else None, comments, favorite, albums)


def author(conn, file_id):
    """Rate, comment, favourite and file the picture, as a person would."""
    conn.execute(
        "INSERT INTO rating(file_id, user_id, rating, created_at) VALUES(?, 1, 5, 0)",
        (file_id,),
    )
    conn.execute(
        "INSERT INTO comment(file_id, user_id, body, created_at) VALUES(?, 1, 'keep', 0)",
        (file_id,),
    )
    conn.execute(
        "INSERT INTO favorite(file_id, user_id, created_at) VALUES(?, 1, 0)", (file_id,)
    )
    album = conn.execute("SELECT id FROM collection WHERE name = 'Keepers'").fetchone()
    if album is None:
        album_id = scan.mint(conn, "collection", "Keepers")
        conn.execute(
            "INSERT INTO collection(id, name, kind, created_at) VALUES(?, 'Keepers', 'album', 0)",
            (album_id,),
        )
    else:
        album_id = album[0]
    conn.execute(
        "INSERT INTO collection_file(collection_id, file_id, added_at) VALUES(?, ?, 0)",
        (album_id, file_id),
    )


# --- the walk itself -------------------------------------------------------


def test_a_first_scan_finds_the_tree(library):
    conn, root = library
    write(root / "portraits" / "2026" / "a.png", "alpha")
    write(root / "portraits" / "b.jpg", "bravo")
    write(root / "notes.txt", "not media")
    result = rescan(conn, root)
    assert (result.added, result.matched, result.missing) == (2, 0, 0)
    names = [n for (n,) in conn.execute("SELECT name FROM file ORDER BY name")]
    assert names == ["a.png", "b.jpg"], "a .txt is not media and must not be indexed"
    depths = dict(conn.execute("SELECT name, depth FROM folder ORDER BY depth"))
    assert depths["portraits"] == 1 and depths["2026"] == 2, depths


def test_rescanning_an_unchanged_library_reads_no_bytes(library):
    conn, root = library
    for i in range(5):
        write(root / f"{i}.png", f"content {i}")
    assert rescan(conn, root).hashed == 5
    again = rescan(conn, root)
    assert again.hashed == 0, "an unchanged file must not be re-read"
    assert (again.matched, again.added, again.missing) == (5, 0, 0)


# --- physical change, authored state must not move -------------------------


def test_renaming_a_file_keeps_its_identity_and_its_ratings(library):
    conn, root = library
    write(root / "a.png", "alpha")
    rescan(conn, root)
    file_id, *_ = file_row(conn, scan.sha256_of(root / "a.png"))
    author(conn, file_id)
    before = authored_state(conn, file_id)

    (root / "a.png").rename(root / "renamed.png")
    result = rescan(conn, root)

    assert (result.matched, result.added, result.missing) == (1, 0, 0)
    row = conn.execute("SELECT id, name FROM file").fetchall()
    assert row == [(file_id, "renamed.png")], "a rename must not mint a new identity"
    assert authored_state(conn, file_id) == before


def test_moving_a_file_between_folders_keeps_everything(library):
    conn, root = library
    write(root / "inbox" / "a.png", "alpha")
    rescan(conn, root)
    file_id, *_ = file_row(conn, scan.sha256_of(root / "inbox" / "a.png"))
    author(conn, file_id)
    before = authored_state(conn, file_id)

    (root / "keep").mkdir()
    (root / "inbox" / "a.png").rename(root / "keep" / "a.png")
    result = rescan(conn, root)

    assert (result.matched, result.added, result.missing) == (1, 0, 0)
    stored_id, name, folder, missing = file_row(conn, scan.sha256_of(root / "keep" / "a.png"))
    assert (stored_id, folder, missing) == (file_id, "keep", None)
    assert authored_state(conn, file_id) == before


def test_renaming_a_folder_touches_one_row(library):
    conn, root = library
    for i in range(4):
        write(root / "portraits" / f"{i}.png", f"content {i}")
    rescan(conn, root)
    files_before = conn.execute(
        "SELECT id, folder_id, name FROM file ORDER BY id"
    ).fetchall()
    folder_id = conn.execute("SELECT id FROM folder WHERE name = 'portraits'").fetchone()[0]

    (root / "portraits").rename(root / "people")
    rescan(conn, root)

    assert conn.execute("SELECT name FROM folder WHERE id = ?", (folder_id,)).fetchone() is not None
    assert (
        conn.execute("SELECT id, folder_id, name FROM file ORDER BY id").fetchall()
        == files_before
    ), "renaming a folder must not re-identify anything inside it"


def test_two_files_swapping_names_follow_their_bytes(library):
    """The case that makes a path hit provisional.

    Both lookups succeed after a swap, so a scanner that trusts
    `(folder_id, name)` leaves every rating on the path and silently attaches
    it to the other picture.
    """
    conn, root = library
    write(root / "a.png", "alpha")
    write(root / "b.png", "bravo")
    rescan(conn, root)
    alpha_id, *_ = file_row(conn, scan.sha256_of(root / "a.png"))
    bravo_id, *_ = file_row(conn, scan.sha256_of(root / "b.png"))
    author(conn, alpha_id)

    (root / "a.png").rename(root / "tmp.png")
    (root / "b.png").rename(root / "a.png")
    (root / "tmp.png").rename(root / "b.png")
    rescan(conn, root)

    assert conn.execute("SELECT name FROM file WHERE id = ?", (alpha_id,)).fetchone()[0] == "b.png"
    assert conn.execute("SELECT name FROM file WHERE id = ?", (bravo_id,)).fetchone()[0] == "a.png"
    assert authored_state(conn, alpha_id) == (5, 1, 1, 1)
    assert authored_state(conn, bravo_id) == (None, 0, 0, 0)


def test_a_deleted_file_is_marked_missing_never_deleted(library):
    conn, root = library
    write(root / "a.png", "alpha")
    rescan(conn, root)
    file_id, *_ = file_row(conn, scan.sha256_of(root / "a.png"))
    author(conn, file_id)

    (root / "a.png").unlink()
    result = rescan(conn, root)

    assert result.missing == 1
    missing_since = conn.execute(
        "SELECT missing_since FROM file WHERE id = ?", (file_id,)
    ).fetchone()[0]
    assert missing_since == NOW, "gone from disk is a state, not a deletion"
    assert authored_state(conn, file_id) == (5, 1, 1, 1)


def test_a_file_that_comes_back_stops_being_missing(library):
    conn, root = library
    write(root / "a.png", "alpha")
    rescan(conn, root)
    file_id, *_ = file_row(conn, scan.sha256_of(root / "a.png"))
    (root / "a.png").unlink()
    rescan(conn, root)

    write(root / "restored" / "a.png", "alpha")
    rescan(conn, root, now=NOW + 60)

    row = conn.execute(
        "SELECT f.id, fo.name, f.missing_since FROM file f JOIN folder fo ON fo.id = f.folder_id"
    ).fetchall()
    assert row == [(file_id, "restored", None)], row


def test_replacing_a_file_in_place_keeps_the_address(library):
    """Same path, different bytes.

    The entity continues, so the URL survives; the rating is kept, because
    silently dropping it is worse than keeping a stale one, and the
    alternative breaks the address. Pinned so a future reader sees a
    decision rather than an accident.
    """
    conn, root = library
    write(root / "a.png", "alpha")
    rescan(conn, root)
    file_id, *_ = file_row(conn, scan.sha256_of(root / "a.png"))
    author(conn, file_id)

    write(root / "a.png", "completely different bytes")
    result = rescan(conn, root)

    assert (result.replaced, result.matched, result.added, result.missing) == (1, 0, 0, 0)
    stored = conn.execute(
        "SELECT id, content_sha256 FROM file WHERE id = ?", (file_id,)
    ).fetchone()
    assert stored[1] == scan.sha256_of(root / "a.png")
    assert authored_state(conn, file_id) == (5, 1, 1, 1)


def test_two_identical_files_are_never_guessed_between(library):
    """sha256 proves byte equality, not object continuity.

    With two identical copies a delete-and-add pair cannot be attributed, so
    the scan must decline rather than move one file's ratings onto another.
    """
    conn, root = library
    write(root / "a.png", "same bytes")
    write(root / "b.png", "same bytes")
    rescan(conn, root)
    assert conn.execute("SELECT count(*) FROM file").fetchone()[0] == 2

    (root / "a.png").unlink()
    (root / "b.png").unlink()
    write(root / "c.png", "same bytes")
    result = rescan(conn, root)

    assert result.ambiguous == 1, "one candidate of two must not be chosen"
    assert result.missing == 2
    assert conn.execute("SELECT count(*) FROM file").fetchone()[0] == 3


def test_changing_only_the_case_of_a_name_is_not_a_new_file(library):
    """NTFS is case-insensitive, so `A.PNG` and `a.png` are one file."""
    conn, root = library
    write(root / "a.png", "alpha")
    rescan(conn, root)
    file_id, *_ = file_row(conn, scan.sha256_of(root / "a.png"))
    author(conn, file_id)

    (root / "a.png").rename(root / "A.PNG")
    result = rescan(conn, root)

    assert (result.added, result.missing) == (0, 0), "case is not identity"
    assert conn.execute("SELECT count(*) FROM file").fetchone()[0] == 1
    assert authored_state(conn, file_id) == (5, 1, 1, 1)


def test_a_folder_renamed_only_by_case_is_not_a_second_folder(library):
    conn, root = library
    write(root / "portraits" / "a.png", "alpha")
    rescan(conn, root)
    (root / "portraits").rename(root / "Portraits")
    rescan(conn, root)
    assert conn.execute("SELECT count(*) FROM folder").fetchone()[0] == 2, (
        "the library root and one subfolder, not a duplicate of the subfolder"
    )
