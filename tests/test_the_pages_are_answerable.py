"""The questions the product asks, answered against a real library.

The schema has been checked against its own constraints, its producers, a real
folder, real EXIF and 100k rows. What it had not been checked against is the
thing it exists for: a page. Every table can be correct and every write linear
while the query a page needs is still awkward, wrong, or a scan.

Each test here is one page. It builds a library through the real producers --
the scanner, the metadata readers, the authored-state writers -- asks the
question that page asks, and checks two things: the answer is right, and the
plan is not a scan of a table that grows with the library.

The plan assertion is what makes this useful at 12 files. A page that reads
every row is fine on a fixture and unusable at 100k, and the difference is
visible in the plan long before it is visible in the clock.
"""

import io
import pathlib
import sqlite3

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import authored, ingest, lineage, scan

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"
NOW = 1_700_000_000.0

A1111 = (
    "a brass diving helmet at dusk <lora:filmGrain:0.35>\n"
    "Negative prompt: blurry\n"
    "Steps: 28, Sampler: Euler a, CFG scale: 7, Seed: 4242, Size: 832x1216, "
    "Model: dreamshaper_8"
)

#: Tables whose row count grows with the library. A page that scans one of
#: these does not have a performance problem, it has a design problem.
GROWS = {"file", "entity", "file_param", "capture", "file_artifact",
         "derived_file_person", "file_blob", "blob", "generation"}


@pytest.fixture
def library(tmp_path):
    """A small library built the way the application builds one."""
    root = tmp_path / "pics"
    (root / "portraits").mkdir(parents=True)
    (root / "landscape").mkdir()

    for folder, count in (("portraits", 7), ("landscape", 5)):
        for i in range(count):
            info = PngInfo()
            if folder == "portraits":
                info.add_text("parameters", A1111)
            path = root / folder / f"{folder}_{i:02d}.png"
            Image.new("RGB", (16, 16), (20 + i * 7, 60, 90 + i)).save(path, pnginfo=info)

    conn = sqlite3.connect(":memory:")
    conn.executescript(io.open(SCHEMA, "r", encoding="utf-8", newline="").read())
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,?,'library',0)", (str(root),))

    scan.scan(conn, 1, root, NOW)
    for file_id, name, folder_id in conn.execute(
        "SELECT f.id, f.name, f.folder_id FROM file f"
    ).fetchall():
        parts = []
        walk = folder_id
        while walk:
            row = conn.execute("SELECT name,parent_id FROM folder WHERE id=?", (walk,)).fetchone()
            parts.append(row[0])
            walk = row[1]
        path = root.parent / "/".join(reversed(parts)) / name
        ingest.one(conn, file_id, path, NOW)

    user = authored.add_user(conn, "will", "hash", "ADMIN", NOW)
    ilse = authored.person(conn, "Ilse", NOW)
    album = authored.collection(conn, "Keepers", NOW)
    first = conn.execute("SELECT id FROM file ORDER BY id LIMIT 1").fetchone()[0]
    authored.rate(conn, first, user, 5, NOW)
    authored.comment(conn, first, user, "the good one", NOW)
    authored.favourite(conn, first, user, NOW)
    authored.add_to_collection(conn, album, first, NOW)
    authored.assert_person(conn, ilse, first, user, NOW)
    conn.execute(
        "INSERT INTO derived_file_person(file_id,person_id,model_id,model_version)"
        " VALUES(?,?,'insightface','v1')",
        (first, ilse),
    )
    conn.commit()
    return {"conn": conn, "root": root, "user": user, "person": ilse,
            "album": album, "first": first}


def plan(conn, sql, args=()):
    return [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql, args)]


def assert_no_growing_scan(conn, sql, args=()):
    """Fail if the page reads a table that grows with the library.

    "SCAN <t> USING INDEX <i>" is an ordered walk of an index and is how a
    "newest first" page is supposed to work, so it is not what this catches.
    A bare `SCAN t` is.
    """
    offenders = [
        step
        for step in plan(conn, sql, args)
        if step.startswith("SCAN ")
        and "USING" not in step
        and any(step.startswith(f"SCAN {t}") or f"SCAN {t} " in step for t in GROWS)
    ]
    assert offenders == [], f"this page reads the whole library: {offenders}"


# --- the front page --------------------------------------------------------


def test_the_front_page_shows_the_newest_first(library):
    conn = library["conn"]
    sql = (
        "SELECT e.slug, f.name FROM file f JOIN entity e ON e.id = f.id"
        " WHERE f.missing_since IS NULL ORDER BY f.mtime DESC LIMIT 60"
    )
    rows = conn.execute(sql).fetchall()
    assert len(rows) == 12
    mtimes = [
        conn.execute("SELECT mtime FROM file WHERE name=?", (name,)).fetchone()[0]
        for _, name in rows
    ]
    assert mtimes == sorted(mtimes, reverse=True)
    assert "file_recent" in " ".join(plan(conn, sql)), "the partial index is not being used"


def test_a_missing_file_leaves_the_front_page(library):
    """`missing_since` is a state, so the grid stops showing it without the
    row, the rating or the album membership going anywhere."""
    conn, first = library["conn"], library["first"]
    conn.execute("UPDATE file SET missing_since=? WHERE id=?", (NOW, first))
    slugs = [
        r[0] for r in conn.execute(
            "SELECT e.slug FROM file f JOIN entity e ON e.id=f.id"
            " WHERE f.missing_since IS NULL"
        )
    ]
    assert len(slugs) == 11
    assert conn.execute("SELECT rating FROM rating WHERE file_id=?", (first,)).fetchone()[0] == 5
    assert conn.execute(
        "SELECT count(*) FROM collection_file WHERE file_id=?", (first,)
    ).fetchone()[0] == 1


# --- one picture -----------------------------------------------------------


def test_the_image_page_answers_in_one_query(library):
    """Everything the page shows about one picture, by its address."""
    conn = library["conn"]
    slug = conn.execute(
        "SELECT e.slug FROM entity e JOIN file f ON f.id=e.id"
        " JOIN folder fo ON fo.id=f.folder_id WHERE fo.name='portraits' LIMIT 1"
    ).fetchone()[0]
    sql = """
        SELECT f.name, fo.name AS folder,
          (SELECT a.name FROM file_artifact fa JOIN artifact a ON a.id=fa.artifact_id
            WHERE fa.file_id=f.id AND fa.role='checkpoint') AS checkpoint,
          (SELECT group_concat(a.name) FROM file_artifact fa
             JOIN artifact a ON a.id=fa.artifact_id
            WHERE fa.file_id=f.id AND fa.role='lora') AS loras,
          (SELECT p.text FROM generation g JOIN prompt p ON p.id=g.prompt_id
            WHERE g.file_id=f.id) AS prompt,
          (SELECT g.seed FROM generation g WHERE g.file_id=f.id) AS seed,
          (SELECT count(*) FROM file_param WHERE file_id=f.id) AS fields
        FROM file f JOIN entity e ON e.id=f.id JOIN folder fo ON fo.id=f.folder_id
        WHERE e.kind='file' AND e.slug=?
    """
    row = conn.execute(sql, (slug,)).fetchone()
    assert row is not None
    name, folder, checkpoint, loras, prompt, seed, fields = row
    assert folder == "portraits"
    assert checkpoint == "dreamshaper_8"
    assert loras == "filmGrain"
    assert prompt.startswith("a brass diving helmet")
    assert seed == 4242
    assert fields > 0, "the parsed long tail is not reachable from the page"
    assert_no_growing_scan(conn, sql, (slug,))


def test_the_image_page_lists_every_parsed_field(library):
    conn, first = library["conn"], library["first"]
    sql = (
        "SELECT source, key, value_text FROM file_param WHERE file_id = ?"
        " ORDER BY source, key"
    )
    fields = conn.execute(sql, (first,)).fetchall()
    assert fields, "no field is shown for a file the parser read"
    assert {f[0] for f in fields} <= {"container", "generation", "exif", "sidecar"}
    assert_no_growing_scan(conn, sql, (first,))


def test_the_next_and_previous_picture_are_reachable(library):
    """A lightbox needs neighbours, and getting them by reading the folder is
    how a page becomes O(folder)."""
    conn = library["conn"]
    folder_id, mtime, slug = conn.execute(
        "SELECT f.folder_id, f.mtime, e.slug FROM file f JOIN entity e ON e.id=f.id"
        " ORDER BY f.mtime LIMIT 1 OFFSET 1"
    ).fetchone()
    sql = (
        "SELECT e.slug FROM file f JOIN entity e ON e.id=f.id"
        " WHERE f.folder_id=? AND f.missing_since IS NULL AND (f.mtime, e.slug) < (?, ?)"
        " ORDER BY f.mtime DESC, e.slug DESC LIMIT 1"
    )
    assert conn.execute(sql, (folder_id, mtime, slug)).fetchone() is not None
    assert_no_growing_scan(conn, sql, (folder_id, mtime, slug))


# --- folders ---------------------------------------------------------------


def test_a_folder_page_lists_its_own_files(library):
    conn = library["conn"]
    folder_slug = conn.execute(
        "SELECT e.slug FROM folder fo JOIN entity e ON e.id=fo.id WHERE fo.name='portraits'"
    ).fetchone()[0]
    sql = (
        "SELECT e.slug, f.name FROM file f JOIN entity e ON e.id=f.id"
        " JOIN folder fo ON fo.id=f.folder_id JOIN entity fe ON fe.id=fo.id"
        " WHERE fe.slug=? AND f.missing_since IS NULL ORDER BY f.name LIMIT 120"
    )
    rows = conn.execute(sql, (folder_slug,)).fetchall()
    assert len(rows) == 7
    assert_no_growing_scan(conn, sql, (folder_slug,))


def test_a_breadcrumb_walks_up_without_a_path(library):
    conn = library["conn"]
    folder_id = conn.execute("SELECT id FROM folder WHERE name='portraits'").fetchone()[0]
    crumbs = conn.execute(
        """WITH RECURSIVE up(id,parent_id,name,depth) AS (
               SELECT id,parent_id,name,depth FROM folder WHERE id=?
               UNION SELECT f.id,f.parent_id,f.name,f.depth
                 FROM folder f JOIN up ON f.id=up.parent_id)
           SELECT name FROM up ORDER BY depth""",
        (folder_id,),
    ).fetchall()
    assert [c[0] for c in crumbs] == ["pics", "portraits"]


# --- equipment and recipe --------------------------------------------------


def test_the_models_page_counts_pictures_not_mentions(library):
    """A checkpoint named by seven files is one row used seven times."""
    conn = library["conn"]
    sql = (
        "SELECT a.name, e.slug, count(*) AS used FROM artifact a"
        " JOIN entity e ON e.id=a.id"
        " JOIN file_artifact fa ON fa.artifact_id=a.id AND fa.role='checkpoint'"
        " JOIN file f ON f.id=fa.file_id AND f.missing_since IS NULL"
        " GROUP BY a.id ORDER BY used DESC"
    )
    rows = conn.execute(sql).fetchall()
    assert rows == [("dreamshaper_8", rows[0][1], 7)]


def test_a_model_page_lists_what_it_made(library):
    conn = library["conn"]
    slug = conn.execute(
        "SELECT e.slug FROM artifact a JOIN entity e ON e.id=a.id WHERE a.kind='checkpoint'"
    ).fetchone()[0]
    sql = (
        "SELECT fe.slug, f.name FROM entity e JOIN artifact a ON a.id=e.id"
        " JOIN file_artifact fa ON fa.artifact_id=a.id"
        " JOIN file f ON f.id=fa.file_id AND f.missing_since IS NULL"
        " JOIN entity fe ON fe.id=f.id WHERE e.slug=? ORDER BY f.name"
    )
    assert len(conn.execute(sql, (slug,)).fetchall()) == 7
    assert_no_growing_scan(conn, sql, (slug,))


def test_the_cross_axis_view_is_one_query(library):
    """The payoff for the join tables: where a model's output actually sits."""
    conn = library["conn"]
    rows = conn.execute(
        "SELECT a.name, count(DISTINCT f.folder_id) AS folders, count(*) AS pictures"
        " FROM artifact a JOIN file_artifact fa ON fa.artifact_id=a.id AND fa.role='checkpoint'"
        " JOIN file f ON f.id=fa.file_id GROUP BY a.id"
    ).fetchall()
    assert rows == [("dreamshaper_8", 1, 7)], rows


# --- people ----------------------------------------------------------------


def test_the_people_page_is_sorted_by_most(library):
    """The question this whole rewrite started from."""
    conn = library["conn"]
    sql = (
        "SELECT COALESCE(p.name,'(unnamed)') AS name, e.slug,"
        " count(DISTINCT fp.file_id) AS images"
        " FROM person p JOIN entity e ON e.id=p.id"
        " JOIN derived_file_person fp ON fp.person_id=p.id"
        " JOIN file f ON f.id=fp.file_id AND f.missing_since IS NULL"
        " GROUP BY p.id ORDER BY images DESC, name"
    )
    rows = conn.execute(sql).fetchall()
    assert rows and rows[0][0] == "Ilse" and rows[0][2] == 1


def test_a_person_keeps_their_page_after_being_renamed(library):
    """The address is the entity's, so naming does not break the link."""
    conn, person = library["conn"], library["person"]
    from db import naming

    old = naming.entity_slug(conn, person)[1]
    authored.name_person(conn, person, "Marguerite", NOW + 1)
    assert naming.resolve(conn, "person", old) == (person, False)
    assert naming.resolve(conn, "person", "marguerite") == (person, True)


# --- search ----------------------------------------------------------------


def test_search_finds_a_picture_by_its_prompt(library):
    conn = library["conn"]
    rows = conn.execute(
        "SELECT count(*) FROM prompt_fts JOIN prompt p ON p.id = prompt_fts.rowid"
        " WHERE prompt_fts MATCH ?",
        ("brass AND dusk",),
    ).fetchone()[0]
    assert rows == 1


def test_search_finds_a_model_by_part_of_its_name(library):
    conn = library["conn"]
    hit = conn.execute(
        "SELECT a.name FROM name_fts n JOIN artifact a ON a.id = n.rowid"
        " WHERE name_fts MATCH ?",
        ('"eamshap"',),
    ).fetchall()
    assert [h[0] for h in hit] == ["dreamshaper_8"]


def test_search_finds_a_picture_by_a_scraped_field(library):
    """The long tail is searchable without anyone writing a facet for it."""
    conn = library["conn"]
    value = conn.execute(
        "SELECT value_text FROM file_param WHERE source='container' AND key='Format' LIMIT 1"
    ).fetchone()[0]
    found = conn.execute(
        "SELECT count(DISTINCT p.file_id) FROM param_fts f"
        " JOIN file_param p ON p.rowid = f.rowid WHERE param_fts MATCH ?",
        (f'"{value}"',),
    ).fetchone()[0]
    assert found == 12


# --- what can be searched --------------------------------------------------


def test_the_ways_page_is_generated_from_the_library(library):
    """`/ways` is not a hand-written list: param_key learns every field on
    ingest, so a tag nobody predicted is offered the day it appears."""
    conn = library["conn"]
    rows = conn.execute(
        "SELECT source, key, value_kind, occurrences FROM param_key ORDER BY occurrences DESC"
    ).fetchall()
    assert rows, "the library taught the facet list nothing"
    counted = dict(
        conn.execute("SELECT key, count(*) FROM file_param GROUP BY key").fetchall()
    )
    assert {r[1]: r[3] for r in rows} == counted, "the facet counts disagree with the rows"


# --- lineage ---------------------------------------------------------------


def test_a_remixed_picture_names_its_parent(library):
    """The edge is written when it is asked for; the page reads it back."""
    conn = library["conn"]
    parent, child = [r[0] for r in conn.execute("SELECT id FROM file ORDER BY id LIMIT 2")]
    lineage.intend(conn, parent, "remix", "comfy-7f2", NOW)
    lineage.resolve(conn, "comfy-7f2", child, NOW + 30)

    sql = (
        "SELECT pe.slug, p.name, d.kind FROM file_derivation d"
        " JOIN file p ON p.id=d.parent_id JOIN entity pe ON pe.id=p.id"
        " WHERE d.child_id=?"
    )
    row = conn.execute(sql, (child,)).fetchone()
    assert row is not None and row[2] == "remix"
    assert_no_growing_scan(conn, sql, (child,))


# --- authored state --------------------------------------------------------


def test_the_album_page_lists_its_members(library):
    conn, album = library["conn"], library["album"]
    sql = (
        "SELECT fe.slug, f.name FROM collection_file cf"
        " JOIN file f ON f.id=cf.file_id AND f.missing_since IS NULL"
        " JOIN entity fe ON fe.id=f.id WHERE cf.collection_id=? ORDER BY f.name"
    )
    assert len(conn.execute(sql, (album,)).fetchall()) == 1
    assert_no_growing_scan(conn, sql, (album,))


def pages(conn):
    """What every page would show, as one comparable value."""
    return {
        "front": conn.execute(
            "SELECT count(*) FROM file WHERE missing_since IS NULL"
        ).fetchone()[0],
        "rating": conn.execute("SELECT rating FROM rating").fetchone()[0],
        "comments": conn.execute("SELECT body FROM comment").fetchall(),
        "album": conn.execute("SELECT count(*) FROM collection_file").fetchone()[0],
        "people": conn.execute("SELECT count(*) FROM derived_file_person").fetchone()[0],
        "assertions": conn.execute("SELECT count(*) FROM person_assertion").fetchone()[0],
        # every FILE address, which is the thing a link points at
        "file slugs": conn.execute(
            "SELECT e.slug FROM entity e JOIN file f ON f.id=e.id ORDER BY e.id"
        ).fetchall(),
        "checkpoint": conn.execute(
            "SELECT count(*) FROM file_artifact WHERE role='checkpoint'"
        ).fetchone()[0],
    }


def test_moving_the_files_behind_the_apps_back_disturbs_no_page(library):
    """The contract, tripped the way a person trips it: drag the folder
    somewhere else in Explorer and open the gallery again.

    Written first asserting that EVERY entity slug was unchanged, which
    failed -- correctly. The destination is a directory that did not exist
    before, so a folder entity for it is new and should be. What must not
    change is any address a link points at, and any row a person authored.
    """
    conn, root = library["conn"], library["root"]
    before = pages(conn)

    (root / "moved").mkdir()
    for path in sorted((root / "portraits").iterdir()):
        path.rename(root / "moved" / path.name)
    (root / "portraits").rmdir()

    result = scan.scan(conn, 1, root, NOW + 100)
    assert result.matched == 12, f"files were not recognised after the move: {result}"
    assert result.added == 0, "moving files created new ones"
    assert result.missing == 0, "moving files lost them"

    assert pages(conn) == before, "a rescan changed what the pages show"
    assert conn.execute(
        "SELECT fo.name FROM file f JOIN folder fo ON fo.id=f.folder_id"
        " WHERE f.name='portraits_00.png'"
    ).fetchone()[0] == "moved", "the file did not follow its bytes"


def test_identical_files_are_never_guessed_between_on_a_move(library):
    """Two byte-identical pictures cannot be told apart, so a move of both
    is unattributable. The scan declines rather than picking one, because
    picking one moves somebody's rating onto a different photograph.
    """
    conn, root = library["conn"], library["root"]
    (root / "twins").mkdir()
    for name in ("a.png", "b.png"):
        Image.new("RGB", (16, 16), (7, 7, 7)).save(root / "twins" / name)
    scan.scan(conn, 1, root, NOW + 100)
    assert conn.execute(
        "SELECT count(*) FROM file WHERE folder_id="
        "(SELECT id FROM folder WHERE name='twins')"
    ).fetchone()[0] == 2

    (root / "twins2").mkdir()
    for name in ("a.png", "b.png"):
        (root / "twins" / name).rename(root / "twins2" / name)
    result = scan.scan(conn, 1, root, NOW + 200)

    assert result.ambiguous == 2, f"a coin was tossed instead of declining: {result}"
    assert result.missing == 2, "the originals were not left as missing"
    assert conn.execute("SELECT count(*) FROM file WHERE missing_since IS NULL").fetchone()[0] == 14
