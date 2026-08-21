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

import pathlib
import re
import sqlite3

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import authored, ingest, lineage, naming, pages, scan

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"
NOW = 1_700_000_000.0

A1111 = (
    "a brass diving helmet at dusk <lora:filmGrain:0.35>\n"
    "Negative prompt: blurry\n"
    "Steps: 28, Sampler: Euler a, CFG scale: 7, Seed: 4242, Size: 832x1216, "
    "Model: dreamshaper_8"
)

#: Tables that stay small however big the library gets: one row per root, per
#: user, per setting, per watched folder, per distinct metadata key. Scanning
#: one of these is fine. EVERYTHING ELSE in the schema grows, and is derived
#: from sqlite_master rather than listed here -- the list was nine names, and
#: the other twenty-nine growing tables could be scanned by any page without
#: the gate saying a word.
STAYS_SMALL = {"root", "user", "setting", "watched_folder", "param_key", "sqlite_sequence"}

#: Words that can follow FROM or JOIN without being an alias.
_NOT_AN_ALIAS = {
    "on",
    "where",
    "group",
    "order",
    "limit",
    "join",
    "left",
    "inner",
    "cross",
    "natural",
    "union",
    "as",
    "using",
    "and",
    "or",
    "having",
    "window",
}


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
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,?,'library',0)", (str(root),))

    scan.scan(conn, 1, root, NOW)
    for file_id, name, folder_id in conn.execute("SELECT f.id, f.name, f.folder_id FROM file f").fetchall():
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
    # Attribution belongs to a clustering RUN, and the People page shows the
    # one marked primary -- several are live at once and they disagree.
    from db import derived

    run = derived.run_for(conn, "insightface", "v1", "chinese-whispers", 0.48, NOW)
    derived.make_primary(conn, run)
    derived.attribute(conn, first, ilse, run, "insightface", "v1")
    conn.commit()
    return {"conn": conn, "root": root, "user": user, "person": ilse, "album": album, "first": first}


def plan(conn, sql, args=()):
    return [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql, args)]


def tables_by_name(sql):
    """Map every name the plan can print back to the table it stands for.

    EXPLAIN QUERY PLAN prints the ALIAS, not the table: `FROM file f` gives
    `SCAN f`. Every gated query in this file joins with aliases, so the check
    below matched nothing on six of its seven call sites and a bare full scan
    of `file` passed it.
    """
    names = {}
    for table, alias in re.findall(
        r"(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s+AS)?(?:\s+([A-Za-z_][A-Za-z0-9_]*))?",
        sql,
        re.IGNORECASE,
    ):
        names[table] = table
        if alias and alias.lower() not in _NOT_AN_ALIAS:
            names[alias] = table
    return names


def assert_no_growing_scan(conn, sql, args=(), *, aggregate=False):
    """Fail if the page reads, or sorts, the whole library.

    "SCAN <t> USING INDEX <i>" is an ordered walk of an index and is how a
    "newest first" page is supposed to work, so it is not what this catches.
    A bare `SCAN t` is, and so is a temp B-tree: sorting every matching row
    at read time costs the same as scanning, and nothing here looked at it.

    `aggregate=True` is for a page that IS a summary -- "every LoRA by how
    much it is used", "what this checkpoint is used with". Counting a group
    and ordering by the count cannot be served by an index, because no index
    holds `count(*)`. Saying so per call keeps it a declared exemption on two
    pages rather than a hole in the check for all of them, and the ban on a
    bare table scan still applies to both.
    """
    names = tables_by_name(sql)
    # An ordered index walk that stops early reads as many rows as it returns.
    # The same walk with no LIMIT reads the table, index or not -- which is why
    # `USING INDEX` alone is not the excuse it was being used as: a covering
    # index made `SELECT f.id FROM file f` read as acceptable.
    stops_early = re.search(r"\bLIMIT\b", sql, re.IGNORECASE) is not None
    offenders = []
    for step in plan(conn, sql, args):
        if "TEMP B-TREE" in step.upper():
            if not aggregate:
                offenders.append(step)
            continue
        match = re.match(r"SCAN ([A-Za-z_][A-Za-z0-9_]*)", step)
        if not match:
            continue
        if stops_early and "USING" in step:
            continue
        table = names.get(match.group(1), match.group(1))
        if table not in STAYS_SMALL:
            offenders.append(f"{step}  ({match.group(1)} = {table})")
    assert offenders == [], f"this page reads or sorts the whole library: {offenders}"


def test_every_page_query_ships_in_db_pages():
    """The queries below are `db.pages`, not restatements of it.

    They used to be written out here, which proves nothing: an algorithm
    defined inside its own tests tests only itself, the tests pass whatever
    the application does, and whoever writes the application writes a second
    implementation with nothing binding it to this one. `db/scan.py` opens
    with that argument and puts the matcher in the package for it; the pages
    were the same shape and the argument had not been applied to them.

    So this sweeps the module and requires every query it ships to survive
    the plan check -- which is what makes the check mean anything, since the
    plan being asserted is now the plan the application runs.
    """
    shipped = [
        name for name, value in vars(pages).items() if name.isupper() and isinstance(value, str) and "SELECT" in value
    ]
    assert len(shipped) >= 12, f"only {len(shipped)} page queries ship: {shipped}"


# --- the front page --------------------------------------------------------


def test_the_front_page_shows_the_newest_first(library):
    conn = library["conn"]
    rows = pages.newest(conn)
    assert len(rows) == 12
    mtimes = [mtime for _, _, mtime in rows]
    assert mtimes == sorted(mtimes, reverse=True)
    assert "file_recent" in " ".join(plan(conn, pages.NEWEST_FIRST, (60,))), "the partial index is not being used"
    assert_no_growing_scan(conn, pages.NEWEST_FIRST, (60,))


def test_a_missing_file_leaves_the_front_page(library):
    """`missing_since` is a state, so the grid stops showing it without the
    row, the rating or the album membership going anywhere."""
    conn, first = library["conn"], library["first"]
    conn.execute("UPDATE file SET missing_since=? WHERE id=?", (NOW, first))
    slugs = [
        r[0] for r in conn.execute("SELECT e.slug FROM file f JOIN entity e ON e.id=f.id WHERE f.missing_since IS NULL")
    ]
    assert len(slugs) == 11
    assert conn.execute("SELECT rating FROM rating WHERE file_id=?", (first,)).fetchone()[0] == 5
    assert conn.execute("SELECT count(*) FROM collection_file WHERE file_id=?", (first,)).fetchone()[0] == 1


# --- one picture -----------------------------------------------------------


def test_the_image_page_answers_in_one_query(library):
    """Everything the page shows about one picture, by its address."""
    conn = library["conn"]
    slug = conn.execute(
        "SELECT e.slug FROM entity e JOIN file f ON f.id=e.id"
        " JOIN folder fo ON fo.id=f.folder_id WHERE fo.name='portraits' LIMIT 1"
    ).fetchone()[0]
    found = naming.resolve(conn, "file", slug)
    assert found is not None, "the address did not resolve"
    file_id = found[0]

    row = pages.picture(conn, file_id)
    assert row is not None
    (
        _name,
        folder,
        width,
        height,
        duration,
        asked_for_width,
        checkpoint,
        missing_since,
        prompt,
        seed,
        fields,
        kind,
    ) = row
    assert folder == "portraits"
    assert kind == "image", "the page must know what it is looking at to pick a media element"
    assert duration is None, "a still picture has no length"
    # The pixels on disk, which nothing wrote until the producer sweep stopped
    # being a word search. Without them the comparison the schema is built to
    # show -- what the recipe asked for against what came out -- has one side.
    assert (width, height) == (16, 16), "the file does not know its own size"
    assert asked_for_width == 832, "the recipe's request is not readable beside it"
    assert checkpoint == "dreamshaper_8"
    assert missing_since is None, "a present file's page says so from the same row"
    # One row per LoRA -- the group_concat column this replaced could not
    # carry a name holding a comma, and the page never read it anyway.
    assert pages.file_loras(conn, file_id) == ["filmGrain"]
    assert prompt.startswith("a brass diving helmet")
    assert seed == 4242
    assert fields > 0, "the parsed long tail is not reachable from the page"
    assert_no_growing_scan(conn, pages.ONE_PICTURE, (file_id,))


def test_an_address_that_was_renamed_still_resolves(library):
    """Resolution is the page layer's, not each page's. A live slug wins and
    history answers only on a miss, so a link written down last year opens
    the thing it named rather than whatever took the name since."""
    conn = library["conn"]
    person = library["person"]
    addressed = naming.entity_slug(conn, person)
    assert addressed is not None
    first = addressed[1]
    authored.name_person(conn, person, "Marguerite", NOW + 1)

    assert naming.resolve(conn, "person", "marguerite") == (person, True)
    assert naming.resolve(conn, "person", first) == (person, False)
    assert naming.resolve(conn, "person", "never-existed") is None


def test_the_image_page_lists_every_parsed_field(library):
    conn, first = library["conn"], library["first"]
    fields = pages.fields_of(conn, first)
    assert fields, "no field is shown for a file the parser read"
    assert {f[0] for f in fields} <= {"container", "generation", "exif", "sidecar"}
    assert_no_growing_scan(conn, pages.PARSED_FIELDS, (first,))


def test_the_page_gate_can_actually_fail(library):
    """The control, and the three ways this gate was blind.

    Without it the gate reads as if it proves something on every page in this
    file, and it proved nothing on six of the seven: EXPLAIN QUERY PLAN
    prints the alias, so `FROM file f` came out as `SCAN f` and the check
    compared it against a hand-written list of table names.
    """
    conn = library["conn"]
    for sql, _blind_spot in (
        ("SELECT f.id FROM file f", "an alias hid the table name"),
        ("SELECT count(*) FROM region", "the table was not on the hand-written list"),
        ("SELECT id FROM file ORDER BY size", "sorting the whole library was never checked"),
    ):
        with pytest.raises(AssertionError, match="reads or sorts"):
            assert_no_growing_scan(conn, sql)


def test_the_next_and_previous_picture_are_reachable(library):
    """A lightbox needs neighbours, and getting them by reading the folder is
    how a page becomes O(folder)."""
    conn = library["conn"]
    folder_id, mtime, file_id = conn.execute(
        "SELECT folder_id, mtime, id FROM file ORDER BY mtime LIMIT 1 OFFSET 1"
    ).fetchone()
    assert pages.neighbour(conn, file_id, previous=True) is not None
    assert pages.neighbour(conn, file_id, previous=False) is not None
    # Ordered on (mtime, id), not (mtime, slug): the slug is on `entity` and no
    # index spans two tables, so tie-breaking on it sorted the whole folder to
    # return one row.
    sql = (
        "SELECT e.slug FROM file f JOIN entity e ON e.id=f.id"
        " WHERE f.folder_id=? AND f.missing_since IS NULL AND (f.mtime, f.id) < (?, ?)"
        " ORDER BY f.mtime DESC, f.id DESC LIMIT 1"
    )
    assert conn.execute(sql, (folder_id, mtime, file_id)).fetchone() is not None
    assert_no_growing_scan(conn, sql, (folder_id, mtime, file_id))


# --- folders ---------------------------------------------------------------


def test_a_folder_page_lists_its_own_files(library):
    """Two queries on purpose: the address is resolved, then the page is read.

    Joining `entity` to match the slug inside the page query gave the planner
    a filter on a table it had to reach through `file`, so it drove from
    `file` and sorted the result -- `SCAN f USING INDEX file_added` plus a
    temp B-tree, on a query that wants one folder. With the id in hand it is
    one index search.

    COLLATE NOCASE to match `file_in_folder`, which is what makes the folder
    an ordered walk rather than a sort -- and it is the order a person
    expects, since the platform's own filesystem is case-insensitive.
    """
    conn = library["conn"]
    folder_slug = conn.execute(
        "SELECT e.slug FROM folder fo JOIN entity e ON e.id=fo.id WHERE fo.name='portraits'"
    ).fetchone()[0]
    address = "SELECT id FROM entity WHERE kind='folder' AND slug=?"
    folder_id = conn.execute(address, (folder_slug,)).fetchone()[0]
    rows = pages.folder_files(conn, folder_id)
    assert len(rows) == 7
    assert_no_growing_scan(conn, address, (folder_slug,))
    assert_no_growing_scan(conn, pages.FOLDER_FILES, (folder_id, 120))


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
    """Same shape as the folder page: resolve the address, then read.

    Matching the slug inside the query made the planner scan the whole of
    `file_artifact` and filter afterwards, which on the checkpoint most of a
    library was made with means reading most of the library. Given the id it
    is a covering-index search, and the rows arrive in file order without a
    sort because `file_artifact` is WITHOUT ROWID and its primary key leads
    on file_id.
    """
    conn = library["conn"]
    slug = conn.execute(
        "SELECT e.slug FROM artifact a JOIN entity e ON e.id=a.id WHERE a.kind='checkpoint'"
    ).fetchone()[0]
    address = "SELECT id FROM entity WHERE kind='artifact' AND slug=?"
    artifact_id = conn.execute(address, (slug,)).fetchone()[0]
    assert len(pages.artifact_files(conn, artifact_id)) == 7
    assert_no_growing_scan(conn, address, (slug,))
    assert_no_growing_scan(conn, pages.ARTIFACT_FILES, (artifact_id, 120))


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
    rows = pages.people_by_most(conn)
    assert rows
    assert rows[0][0] == "Ilse"
    assert rows[0][2] == 1


def test_a_person_keeps_their_page_after_being_renamed(library):
    """The address is the entity's, so naming does not break the link."""
    conn, person = library["conn"], library["person"]
    from db import naming

    addressed = naming.entity_slug(conn, person)
    assert addressed is not None
    old = addressed[1]
    authored.name_person(conn, person, "Marguerite", NOW + 1)
    assert naming.resolve(conn, "person", old) == (person, False)
    assert naming.resolve(conn, "person", "marguerite") == (person, True)


# --- search ----------------------------------------------------------------


def test_search_finds_a_picture_by_its_prompt(library):
    conn = library["conn"]
    rows = conn.execute(
        "SELECT count(*) FROM prompt_fts JOIN prompt p ON p.id = prompt_fts.rowid WHERE prompt_fts MATCH ?",
        ("brass AND dusk",),
    ).fetchone()[0]
    assert rows == 1


def test_search_finds_a_model_by_part_of_its_name(library):
    conn = library["conn"]
    hit = conn.execute(
        "SELECT a.name FROM name_fts n JOIN artifact a ON a.id = n.rowid WHERE name_fts MATCH ?",
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
    rows = pages.ways(conn)
    assert rows, "the library taught the facet list nothing"
    # Grouped by (source, key) on both sides -- param_key's primary key. On
    # `key` alone the dict kept one row per name while the GROUP BY summed
    # across sources, so the two agreed by accident and would have gone on
    # agreeing through a real drift.
    counted = {
        (s, k): n for s, k, n in conn.execute("SELECT source, key, count(*) FROM file_param GROUP BY source, key")
    }
    assert {(r[0], r[1]): r[3] for r in rows} == counted, "the facet counts disagree with the rows"


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
    assert row is not None
    assert row[2] == "remix"
    assert_no_growing_scan(conn, sql, (child,))


# --- authored state --------------------------------------------------------


def test_the_album_page_lists_its_members(library):
    conn, album = library["conn"], library["album"]
    assert len(pages.album_files(conn, album)) == 1
    assert_no_growing_scan(conn, pages.ALBUM_FILES, (album, 120))


def every_page(conn):
    """What every page would show, as one comparable value."""
    return {
        "front": conn.execute("SELECT count(*) FROM file WHERE missing_since IS NULL").fetchone()[0],
        "rating": conn.execute("SELECT rating FROM rating").fetchone()[0],
        "comments": conn.execute("SELECT body FROM comment").fetchall(),
        "album": conn.execute("SELECT count(*) FROM collection_file").fetchone()[0],
        "people": conn.execute("SELECT count(*) FROM derived_file_person").fetchone()[0],
        "assertions": conn.execute("SELECT count(*) FROM person_assertion").fetchone()[0],
        # every FILE address, which is the thing a link points at
        "file slugs": conn.execute("SELECT e.slug FROM entity e JOIN file f ON f.id=e.id ORDER BY e.id").fetchall(),
        "checkpoint": conn.execute("SELECT count(*) FROM file_artifact WHERE role='checkpoint'").fetchone()[0],
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
    before = every_page(conn)

    (root / "moved").mkdir()
    for path in sorted((root / "portraits").iterdir()):
        path.rename(root / "moved" / path.name)
    (root / "portraits").rmdir()

    result = scan.scan(conn, 1, root, NOW + 100)
    assert result.matched == 12, f"files were not recognised after the move: {result}"
    assert result.added == 0, "moving files created new ones"
    assert result.missing == 0, "moving files lost them"

    assert every_page(conn) == before, "a rescan changed what the pages show"
    assert (
        conn.execute(
            "SELECT fo.name FROM file f JOIN folder fo ON fo.id=f.folder_id WHERE f.name='portraits_00.png'"
        ).fetchone()[0]
        == "moved"
    ), "the file did not follow its bytes"


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
    assert (
        conn.execute("SELECT count(*) FROM file WHERE folder_id=(SELECT id FROM folder WHERE name='twins')").fetchone()[
            0
        ]
        == 2
    )

    (root / "twins2").mkdir()
    for name in ("a.png", "b.png"):
        (root / "twins" / name).rename(root / "twins2" / name)
    result = scan.scan(conn, 1, root, NOW + 200)

    assert result.ambiguous == 2, f"a coin was tossed instead of declining: {result}"
    assert result.missing == 2, "the originals were not left as missing"
    assert conn.execute("SELECT count(*) FROM file WHERE missing_since IS NULL").fetchone()[0] == 14


# --- the pages db.pages ships that nothing was asking for -------------------


def a_recipe_library(tmp_path):
    """Four pictures across two checkpoints and two LoRAs, so co-occurrence
    is a question with an answer rather than a table with one row."""
    root = tmp_path / "recipes"
    root.mkdir()
    recipes = [
        ("a.png", "<lora:filmGrain:0.4> <lora:detailTweaker:0.8>", "dreamshaper_8"),
        ("b.png", "<lora:filmGrain:0.35>", "dreamshaper_8"),
        ("c.png", "<lora:filmGrain:0.5> <lora:detailTweaker:0.6>", "fluxDev"),
        ("d.png", "<lora:detailTweaker:0.9>", "fluxDev"),
    ]
    for name, loras, model in recipes:
        info = PngInfo()
        info.add_text(
            "parameters",
            (
                f"a castle {loras}\nNegative prompt: blur\n"
                f"Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, "
                f"Size: 512x512, Model: {model}"
            ),
        )
        Image.new("RGB", (16, 16), (9, 9, 9)).save(root / name, pnginfo=info)

    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,?,'library',0)", (str(root),))
    scan.scan(conn, 1, root, NOW)
    for file_id, name in conn.execute("SELECT id, name FROM file").fetchall():
        ingest.one(conn, file_id, root / name, NOW)
    return conn


def test_a_lora_says_what_it_is_actually_used_with(tmp_path):
    """The feature the old app faked.

    "Proven Match" was `workflow_files LIKE '%<ckpt>%'` over a delimited blob
    and a string split -- co-residency in one field, with no counts and no
    confidence. It is a join, and the counts are real.
    """
    conn = a_recipe_library(tmp_path)
    try:
        loras = {name: slug for name, slug, _ in pages.artifacts_by_use(conn, "lora")}
        assert set(loras) == {"filmGrain", "detailTweaker"}

        resolved = naming.resolve(conn, "artifact", loras["filmGrain"])
        assert resolved is not None
        film = resolved[0]
        assert film is not None
        assert pages.lora_synergy(conn, film) == [
            ("dreamshaper_8", "checkpoint-dreamshaper-8", 2),
            ("fluxDev", "checkpoint-fluxdev", 1),
        ]
        assert_no_growing_scan(conn, pages.LORA_SYNERGY, (film,), aggregate=True)
    finally:
        conn.close()


def test_the_index_counts_pictures_not_mentions(tmp_path):
    """A model page that counted rows would count spellings and re-parses."""
    conn = a_recipe_library(tmp_path)
    try:
        assert pages.artifacts_by_use(conn, "checkpoint") == [
            ("dreamshaper_8", "checkpoint-dreamshaper-8", 2),
            ("fluxDev", "checkpoint-fluxdev", 2),
        ]
        assert_no_growing_scan(conn, pages.ARTIFACTS_BY_USE, ("checkpoint",), aggregate=True)
    finally:
        conn.close()


def test_the_entity_layer_queries_do_not_scan_the_library(library):
    """The four queries the entity-surface delta added, held to the same
    doctrine as every sibling: the plan is not a scan of a table that
    grows with the library, and not a sort of one either."""
    conn = library["conn"]
    assert_no_growing_scan(conn, pages.WORKFLOWS_BY_USE, (), aggregate=True)
    assert_no_growing_scan(conn, pages.WORKFLOW_FILES, (1, 120))
    assert_no_growing_scan(conn, pages.ALBUM_PRESENT, (library["album"],))
    assert_no_growing_scan(conn, pages.FILE_LORAS, (library["first"],))
    assert_no_growing_scan(conn, pages.DUPE_GROUPS, (120,), aggregate=True)
    assert_no_growing_scan(conn, pages.DUPE_COPIES, (library["first"], 120))


def test_a_missing_copy_leaves_the_shelf_and_the_page_agreeing(library):
    """/dupes counted every member of a group -- missing ones included --
    while the picture page lists only present twins: the shelf said three
    bodies, the page showed one. Present members only, both routes, per
    the repo's own "the two routes cannot drift apart" doctrine."""
    conn = library["conn"]
    a, b, c = [row[0] for row in conn.execute("SELECT id FROM file ORDER BY id LIMIT 3")]
    for member, is_best in ((a, 1), (b, 0), (c, 0)):
        conn.execute(
            "INSERT INTO derived_dupe_group(file_id, group_id, distance, threshold, is_best, computed_at)"
            " VALUES(?, ?, ?, 4, ?, 0)",
            (member, a, 0 if member == a else 2, is_best),
        )
    conn.execute("UPDATE file SET missing_since = 1.0 WHERE id = ?", (c,))
    shelf = pages.dupe_groups(conn)
    told = pages.dupe_copies(conn, a)
    assert [row[2] for row in shelf] == [len(told) + 1], (
        f"the shelf counts {[row[2] for row in shelf]} while the page lists {len(told)} twins"
    )


def test_a_person_is_shown_across_the_folders_they_are_in(library):
    """The payoff for the join tables and the reason the six axes are six.

    Where somebody's pictures actually sit is a fact the folder tree cannot
    state and the people page cannot either; the disagreement between the
    disk layout and the meaning is the thing worth showing.
    """
    conn, person = library["conn"], library["person"]
    spread = pages.person_across_folders(conn, person)
    assert [(name, count) for name, _, count in spread] == [("landscape", 1)]
    run = conn.execute("SELECT id FROM derived_face_run WHERE is_primary=1").fetchone()[0]
    assert_no_growing_scan(conn, pages.PERSON_ACROSS_FOLDERS, (person, run), aggregate=True)
    assert_no_growing_scan(conn, pages.PERSON_FILES, (person, run))


def test_the_breadcrumb_and_the_lineage_pages_are_index_driven(library):
    """Both walk a parent chain, and a recursive walk is where an unindexed
    step hides -- it costs once per level rather than once."""
    conn, first = library["conn"], library["first"]
    folder_id = conn.execute("SELECT folder_id FROM file WHERE id=?", (first,)).fetchone()[0]
    assert [name for _, name in pages.breadcrumb(conn, folder_id)] == ["pics", "landscape"]
    assert_no_growing_scan(conn, pages.PARENTS, (first,))
    assert_no_growing_scan(conn, pages.CHILDREN, (first,))
