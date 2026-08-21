"""Every constraint db/schema.sql claims must actually reject something.

The schema this replaces derived a file's identity from its path
(`content_digest(filepath)`, smartgallery.py:3994), so renaming a folder
re-identified every file beneath it, and `_reassign_file_ids` had to walk nine
tables to carry the user's ratings across. Five `file_id` columns had no
foreign key at all; `collections.parent_id` had none and no cycle guard while
four recursive CTEs walked it.

This file exists so none of that can come back quietly. Each test pairs a
positive control that must succeed with a negative that must be rejected -- a
constraint that never rejects anything is not a constraint, and a sweep with
no failing control is not a gate.

The relationship sweep uses `PRAGMA foreign_key_list`, not `table_info`:
table_info can see that a column is named `person_id`, but only
foreign_key_list can prove it points at `person`.
"""

from __future__ import annotations

import pathlib
import re
import sqlite3

import pytest

from db.scan import Outcome, resolve_scan

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"

# Columns ending in _id that are deliberately not references to another row.
_NOT_A_REFERENCE = {
    ("entity", "id"),
    ("root", "id"),
    ("user", "id"),
    ("job", "id"),
    ("derived_media_sample", "id"),
    ("comment", "id"),
    ("feedback", "id"),
    ("derivation_intent", "id"),
    ("file_derivation", "id"),
    ("derived_face_cluster", "id"),
    ("derived_face_instance", "id"),
    ("derived_annotation", "id"),
    ("region", "id"),
    # backend identity strings ("insightface", "qwen-vl"), not rows
    ("derived_embedding", "model_id"),
    ("derived_face_cluster", "model_id"),
    ("derived_face_instance", "model_id"),
    ("derived_annotation", "model_id"),
    ("derived_file_person", "model_id"),
    ("derived_face_run", "model_id"),
    ("job_item", "item_id"),
}

# Guards that fire on INSERT and deliberately not on UPDATE. Each needs a
# reason, because the default reading of an INSERT-only rule is that somebody
# forgot the other half.
_INSERT_ONLY_ON_PURPOSE = {
    # FK cascade actions DO fire UPDATE triggers (verified). Guarding this on
    # UPDATE would abort the ON DELETE SET NULL that detaches a judged target,
    # and losing the human judgement is worse than holding a nulled pointer.
    "feedback: feedback must name what it judges",
}

# TEXT columns whose name looks like an enum but whose values are genuinely
# open. Each is a decision on the record, not an oversight.
_FREE_TEXT = {
    # the location of a root, which is the one place a path IS the fact
    "path",
    # a person's own words, and the name of a thing as its metadata spelled it
    "name",
    "note",
    "summary",
    "description",
    "body",
    "text",
}


@pytest.fixture(scope="module")
def ddl():
    return SCHEMA.read_text(encoding="utf-8")


@pytest.fixture
def db(ddl):
    conn = sqlite3.connect(":memory:")
    conn.executescript(ddl)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def entity(conn, eid, kind, slug):
    conn.execute(
        "INSERT INTO entity(id,uuid,kind,slug) VALUES(?,?,?,?)",
        (eid, eid.to_bytes(16, "big"), kind, slug),
    )


def tree(conn):
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'/lib','library',0)")
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(2,'/ext','mount',0)")
    entity(conn, 1, "folder", "portraits")
    conn.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(1,1,NULL,'portraits',0)")
    entity(conn, 2, "folder", "y2026")
    conn.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(2,1,1,'2026',1)")
    # Two clustering runs, because attribution belongs to a run and the whole
    # point of the shape is that a second one can exist beside the first.
    conn.execute(
        "INSERT INTO derived_face_run(id,model_id,model_version,method,threshold,"
        "is_primary,computed_at) VALUES(1,'insightface','v1','chinese-whispers',0.48,1,0)"
    )
    conn.execute(
        "INSERT INTO derived_face_run(id,model_id,model_version,method,threshold,"
        "computed_at) VALUES(2,'insightface','v2','chinese-whispers',0.48,0)"
    )


def a_space(conn) -> int:
    """A minted similarity_space row -- hash rows carry the identity of what
    computed them, so a bare derived_file_hash insert needs one."""
    return conn.execute(
        "INSERT INTO similarity_space(key,representation,dimensions,metric,"
        "producer,producer_version,preprocess,preprocess_version,spec_hash,created_at)"
        " VALUES('perceptual.test','binary',64,'hamming','p','1','pp','1',hex(randomblob(8)),0)"
    ).lastrowid


def a_file(conn, eid, folder_id, name, sha=None):
    entity(conn, eid, "file", f"file-{eid}")
    conn.execute(
        "INSERT INTO file(id,folder_id,name,kind,size,mtime,content_sha256,"
        "first_seen_at,last_seen_at) VALUES(?,?,?,'image',10,0,?,0,0)",
        (eid, folder_id, name, sha),
    )


def unconstrained_reference_columns(conn):
    out = []
    virt = virtual_table_names(conn)
    tables = [
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        if r[0] not in virt
    ]
    for table in tables:
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        fks = {r[3] for r in conn.execute(f"PRAGMA foreign_key_list({table})")}
        # Being part of a primary key does NOT excuse a column from declaring
        # its reference: file_person.person_id is both. Surrogate keys are
        # named in _NOT_A_REFERENCE instead, one line each, so the exemption
        # is a decision on the record rather than a rule that quietly widens.
        out.extend(
            f"{table}.{col}"
            for col in (c[1] for c in cols)
            if col.endswith("_id") and col not in fks and (table, col) not in _NOT_A_REFERENCE
        )
    return out


# ------------------------------------------------------------------- shape


def test_the_schema_builds(db):
    """Control: if the DDL stops loading, everything below passes vacuously."""
    n = db.execute("SELECT count(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    assert n >= 30, f"only {n} tables built; the schema did not load"


def virtual_table_names(conn):
    """FTS5 tables and the shadow tables they own. Neither can be STRICT, and
    neither declares foreign keys, so both sweeps must skip them."""
    virt = [
        r[0]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE 'CREATE VIRTUAL TABLE%'")
    ]
    return {
        name
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        if any(name == v or name.startswith(v + "_") for v in virt)
    }


def test_every_table_is_strict(db):
    virt = virtual_table_names(db)
    loose = [
        name
        for name, sql in db.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        if name not in virt and "STRICT" not in (sql or "").upper()
    ]
    assert loose == [], f"tables without STRICT, so column types are advisory: {loose}"


def has_rowid(conn, table):
    """Behavioural, not textual. Matching the phrase in `sql` was matching it
    inside a comment -- file_param says "deliberately NOT WITHOUT ROWID" and
    that made the substring check pass for exactly the wrong reason."""
    try:
        conn.execute(f"SELECT rowid FROM {table} LIMIT 0")
    except sqlite3.OperationalError:
        return False
    else:
        return True


def test_join_tables_carry_no_rowid(db):
    """sqlite.org/withoutrowid.html names composite PKs with small rows as the
    case this optimization exists for. A rowid on these is pure overhead."""
    want = ["file_artifact", "derived_file_person", "collection_file", "rating", "favorite"]
    still = [t for t in want if has_rowid(db, t)]
    assert still == [], f"composite-PK tables still paying for a rowid: {still}"


def test_the_long_tail_keeps_its_rowid(db):
    """The counterpart, and the reason the check above must be behavioural:
    file_param absorbs values that run to multiple KB, which is precisely what
    the optimization is not for. It must NOT be in the list above."""
    assert has_rowid(db, "file_param"), "file_param is WITHOUT ROWID, but it holds the long tail's multi-KB values"


def test_no_foreign_key_points_at_a_missing_table(db):
    names = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    broken = [
        f"{t} -> {row[2]}"
        for t in sorted(names - virtual_table_names(db))
        for row in db.execute(f"PRAGMA foreign_key_list({t})")
        if row[2] not in names
    ]
    assert broken == [], f"foreign keys naming tables that do not exist: {broken}"


def test_every_reference_column_is_constrained(db):
    unconstrained = unconstrained_reference_columns(db)
    assert unconstrained == [], (
        "these columns name a row in another table but declare no foreign key: "
        f"{unconstrained}.\nAdd REFERENCES with an explicit ON DELETE, or add the "
        "column to _NOT_A_REFERENCE in this file with the reason."
    )


def test_the_load_bearing_references_point_where_they_claim(db):
    """A declared foreign key is not enough; it has to name the right table.

    The sweep above proves a reference exists. This proves it goes somewhere
    sensible, for the relations the product's correctness rests on.
    """
    expected = {
        ("file", "folder_id"): "folder",
        ("file", "id"): "entity",
        ("folder", "parent_id"): "folder",
        ("folder", "root_id"): "root",
        ("derived_file_person", "person_id"): "person",
        ("derived_file_person", "file_id"): "file",
        ("file_artifact", "artifact_id"): "artifact",
        ("collection_file", "collection_id"): "collection",
        ("capture", "file_id"): "file",
        ("file_param", "file_id"): "file",
        ("file_relation", "related_id"): "file",
        ("slug_history", "entity_id"): "entity",
        ("derivation_intent", "parent_id"): "file",
        ("file_derivation", "child_id"): "file",
        ("generation", "workflow_id"): "artifact",
        ("generation", "prompt_id"): "prompt",
        ("collection", "parent_id"): "collection",
        ("derived_face_instance", "sample_id"): "derived_media_sample",
        ("derived_face_cluster", "person_id"): "person",
        ("job", "target_id"): "entity",
    }
    actual = {
        (table, row[3]): row[2]
        for table in (
            {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")} - virtual_table_names(db)
        )
        for row in db.execute(f"PRAGMA foreign_key_list({table})")
    }
    wrong = {k: (v, actual.get(k)) for k, v in expected.items() if actual.get(k) != v}
    assert wrong == {}, f"references pointing at the wrong table (expected, actual): {wrong}"


def test_the_target_check_can_actually_fail(ddl):
    """Control: repoint one reference at a real but wrong table."""
    broken = sqlite3.connect(":memory:")
    broken.executescript(
        ddl.replace(
            "person_id     INTEGER NOT NULL REFERENCES person(id) ON DELETE CASCADE",
            "person_id     INTEGER NOT NULL REFERENCES artifact(id) ON DELETE CASCADE",
        )
    )
    actual = {
        (t, row[3]): row[2]
        for t in {r[0] for r in broken.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for row in broken.execute(f"PRAGMA foreign_key_list({t})")
    }
    # the control must exercise the gate, not merely prove the edit landed
    assert actual.get(("derived_file_person", "person_id")) != "person", "control failed: the repoint did not take"


def test_the_reference_sweep_can_actually_fail(ddl):
    """Control for the sweep above. Remove one foreign key; it must be seen.

    Without this, a sweep that silently matched nothing would pass forever --
    the same fault the old route-classification test was written to catch.
    """
    broken = sqlite3.connect(":memory:")
    broken.executescript(
        ddl.replace(
            "person_id     INTEGER NOT NULL REFERENCES person(id) ON DELETE CASCADE",
            "person_id     INTEGER NOT NULL",
        )
    )
    assert "derived_file_person.person_id" in unconstrained_reference_columns(broken)


# ----------------------------------------------------------------- folders


def test_two_root_folders_cannot_share_a_name(db):
    tree(db)
    entity(db, 90, "folder", "portraits-2")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(90,1,NULL,'portraits',0)")


def test_one_name_may_repeat_under_different_parents(db):
    tree(db)
    entity(db, 91, "folder", "y2026-b")
    db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(91,1,2,'2026',2)")


def test_one_parent_cannot_hold_two_children_of_a_name(db):
    tree(db)
    entity(db, 92, "folder", "dupe")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(92,1,1,'2026',1)")


def test_a_folder_cannot_claim_a_root_its_parent_does_not(db):
    tree(db)
    entity(db, 93, "folder", "crossroot")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(93,2,1,'x',1)")


def test_a_folder_cannot_become_its_own_ancestor(db):
    tree(db)
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE folder SET parent_id=2 WHERE id=1")


def test_a_collection_cannot_become_its_own_ancestor(db):
    entity(db, 10, "collection", "a")
    entity(db, 11, "collection", "b")
    db.execute("INSERT INTO collection(id,parent_id,name,kind,created_at) VALUES(10,NULL,'A','album',0)")
    db.execute("INSERT INTO collection(id,parent_id,name,kind,created_at) VALUES(11,10,'B','album',0)")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE collection SET parent_id=11 WHERE id=10")


# --------------------------------------------------------------- addressing


def test_a_slug_is_unique_within_its_kind(db):
    entity(db, 20, "person", "ilse")
    entity(db, 21, "artifact", "ilse")  # an artifact may share a person's slug
    with pytest.raises(sqlite3.IntegrityError):
        entity(db, 22, "person", "ilse")


def test_a_uuid_is_sixteen_bytes(db):
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(99,x'0011','person','x')")


def test_unnamed_people_are_still_addressable(db):
    """The People page leads with unnamed groups, so they need URLs on day one.
    Naming is a product goal, never an addressing prerequisite."""
    entity(db, 30, "person", "person-8a41f2")
    entity(db, 31, "person", "person-b7c093")
    db.execute("INSERT INTO person(id,name,created_at) VALUES(30,NULL,0)")
    db.execute("INSERT INTO person(id,name,created_at) VALUES(31,NULL,0)")
    n = db.execute("SELECT count(DISTINCT slug) FROM entity WHERE kind='person'").fetchone()[0]
    assert n == 2, "two unnamed people collapsed to one address"


def test_two_people_may_share_a_display_name(db):
    entity(db, 40, "person", "ilse")
    entity(db, 41, "person", "ilse-2")
    db.execute("INSERT INTO person(id,name,created_at) VALUES(40,'Ilse',0)")
    db.execute("INSERT INTO person(id,name,created_at) VALUES(41,'Ilse',0)")
    n = db.execute("SELECT count(DISTINCT slug) FROM entity WHERE kind='person'").fetchone()[0]
    assert n == 2, "identical display names produced one address"


def test_a_retired_slug_still_resolves(db):
    entity(db, 50, "person", "person-8a41f2")
    db.execute("INSERT INTO person(id,name,created_at) VALUES(50,NULL,0)")
    db.execute("INSERT INTO slug_history(kind,slug,entity_id,retired_at) VALUES('person','person-8a41f2',50,1.0)")
    db.execute("UPDATE entity SET slug='ilse' WHERE id=50")
    row = db.execute("SELECT entity_id FROM slug_history WHERE kind='person' AND slug='person-8a41f2'").fetchone()
    assert row, "the old URL stopped pointing at the thing it named"
    assert row[0] == 50, "the old URL stopped pointing at the thing it named"


def test_moving_a_file_does_not_change_its_address(db):
    """The whole point. The old schema keyed identity on the path, so a move
    produced a different file and every URL to it died."""
    tree(db)
    a_file(db, 60, 1, "dusk.png")
    before = db.execute("SELECT slug FROM entity WHERE id=60").fetchone()[0]
    db.execute("UPDATE file SET folder_id=2 WHERE id=60")
    db.execute("UPDATE folder SET name='renamed' WHERE id=1")
    after = db.execute("SELECT slug FROM entity WHERE id=60").fetchone()[0]
    assert before == after, "moving the file, or renaming a folder above it, changed its URL"


def test_renaming_a_folder_touches_one_row(db):
    tree(db)
    for i in range(600, 610):
        a_file(db, i, 2, f"f{i}.png")
    cur = db.execute("UPDATE folder SET name='renamed' WHERE id=1")
    assert cur.rowcount == 1, f"a folder rename rewrote {cur.rowcount} rows"


# ---------------------------------------------------------------- relations


def test_one_folder_cannot_hold_two_files_of_a_name(db):
    tree(db)
    a_file(db, 70, 1, "a.png")
    entity(db, 71, "file", "file-71")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,first_seen_at,last_seen_at) "
            "VALUES(71,1,'a.png','image',1,0,0,0)"
        )


def name_key(name):
    """The one normalization rule. Mirrors what the search box must use: if
    ingest dedupes on a different rule than search matches on, a model becomes
    unfindable by its own name -- the failure
    test_a_model_is_found_by_its_own_name was written for."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def an_artifact(conn, eid, kind, name, sha=None, quoted=None):
    entity(conn, eid, "artifact", f"{kind}-{name}")
    conn.execute(
        "INSERT INTO artifact(id,kind,name,name_key,content_sha256,quoted_hash,first_seen_at) VALUES(?,?,?,?,?,?,0)",
        (eid, kind, name, name_key(name), sha, quoted),
    )


def test_a_lora_stack_keeps_its_order(db):
    """Two workflows with the same LoRAs in a different order are different
    recipes. Without an ordinal they collapse to the same relational set."""
    tree(db)
    a_file(db, 80, 1, "x.png")
    an_artifact(db, 100, "lora", "detail")
    an_artifact(db, 101, "lora", "film")
    db.execute("INSERT INTO file_artifact(file_id,ordinal,artifact_id,role,model_weight) VALUES(80,0,100,'lora',0.8)")
    db.execute("INSERT INTO file_artifact(file_id,ordinal,artifact_id,role,model_weight) VALUES(80,1,101,'lora',0.4)")
    order = [
        r[0]
        for r in db.execute("SELECT artifact_id FROM file_artifact WHERE file_id=80 AND role='lora' ORDER BY ordinal")
    ]
    assert order == [100, 101], f"stack order lost: {order}"
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO file_artifact(file_id,ordinal,artifact_id,role) VALUES(80,0,101,'lora')")


def test_one_join_carries_every_artifact_kind(db):
    """A checkpoint, a LoRA and a camera body are the same shape. Twelve kinds
    should not have meant twelve tables -- adding one is a CHECK entry."""
    tree(db)
    a_file(db, 90, 1, "shot.jpg")
    an_artifact(db, 130, "checkpoint", "flux-dev")
    an_artifact(db, 131, "lora", "detail")
    an_artifact(db, 132, "camera", "X-T5")
    an_artifact(db, 133, "lens", "XF 35mm F1.4")
    an_artifact(db, 134, "vae", "ae.sft")
    for aid, role in ((130, "checkpoint"), (131, "lora"), (132, "captured_with"), (133, "mounted_lens"), (134, "vae")):
        db.execute("INSERT INTO file_artifact(file_id,artifact_id,role) VALUES(?,?,?)", (90, aid, role))
    kinds = {
        r[0]
        for r in db.execute(
            "SELECT a.kind FROM file_artifact fa JOIN artifact a ON a.id = fa.artifact_id WHERE fa.file_id=90"
        )
    }
    assert kinds == {"checkpoint", "lora", "camera", "lens", "vae"}


def test_a_camera_is_a_page_not_a_string(db):
    """The old plan wrote off Camera and Places as facets with no backing data.
    A photograph in the library disproves that the moment EXIF is read."""
    tree(db)
    a_file(db, 91, 1, "trip.jpg")
    an_artifact(db, 140, "camera", "X-T5")
    db.execute("INSERT INTO file_artifact(file_id,artifact_id,role) VALUES(91,140,'captured_with')")
    db.execute(
        "INSERT INTO capture(file_id,captured_at,iso,f_number,exposure_time,focal_length,"
        "gps_lat,gps_lon,parsed_at) VALUES(91,1700000000.0,400,1.4,0.004,35,51.5,-0.12,0)"
    )
    slug, iso = db.execute(
        "SELECT e.slug, c.iso FROM file f "
        "JOIN file_artifact fa ON fa.file_id = f.id AND fa.role='captured_with' "
        "JOIN entity e ON e.id = fa.artifact_id "
        "JOIN capture c ON c.file_id = f.id WHERE f.id=91"
    ).fetchone()
    assert slug == "camera-X-T5"
    assert iso == 400
    located = db.execute("SELECT count(*) FROM capture WHERE gps_lat IS NOT NULL").fetchone()[0]
    assert located == 1, "a geotagged photo is not findable by place"


def test_capture_time_is_not_file_mtime(db):
    """When the shutter opened, when the bytes were last written, and when the
    file was created are three different facts. Conflating them is why a copied
    photo sorts as if it were taken today."""
    tree(db)
    a_file(db, 92, 1, "old.jpg")
    db.execute("UPDATE file SET mtime=1900000000.0, btime=1800000000.0 WHERE id=92")
    db.execute("INSERT INTO capture(file_id,captured_at,parsed_at) VALUES(92,1000000000.0,0)")
    mtime, btime, shot = db.execute(
        "SELECT f.mtime, f.btime, c.captured_at FROM file f JOIN capture c ON c.file_id=f.id WHERE f.id=92"
    ).fetchone()
    assert shot < btime < mtime, "the three timestamps collapsed into one"


def test_any_tail_field_is_queryable_without_a_migration(db):
    """The answer to 'there are thousands of metadata fields'. A JSON blob
    holds them; only an indexed key makes them findable."""
    tree(db)
    a_file(db, 93, 1, "a.jpg")
    a_file(db, 94, 1, "b.jpg")
    db.execute(
        "INSERT INTO file_param(file_id,source,key,value_text,value_num) VALUES(93,'iptc','Credit','Reuters',NULL)"
    )
    db.execute(
        "INSERT INTO file_param(file_id,source,key,value_text,value_num) "
        "VALUES(93,'exif','LensSerialNumber','44A1',NULL)"
    )
    db.execute(
        "INSERT INTO file_param(file_id,source,key,value_text,value_num) VALUES(94,'container','BitDepth','16',16)"
    )
    assert db.execute("SELECT file_id FROM file_param WHERE key='Credit' AND value_text='Reuters'").fetchone()[0] == 93
    assert db.execute("SELECT count(*) FROM file_param WHERE key='BitDepth' AND value_num >= 16").fetchone()[0] == 1
    # the same key may arrive from two origins and must stay distinguishable
    db.execute("INSERT INTO file_param(file_id,source,key,value_text) VALUES(93,'xmp','Credit','AP')")
    assert db.execute("SELECT count(*) FROM file_param WHERE file_id=93 AND key='Credit'").fetchone()[0] == 2
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO file_param(file_id,source,key,value_text) VALUES(93,'iptc','Credit','dup')")


def test_person_membership_records_which_backend_said_so(db):
    """Two face backends may disagree. Neither may overwrite the other, and
    neither may be laundered into unqualified fact."""
    tree(db)
    a_file(db, 85, 1, "p.png")
    entity(db, 110, "person", "ilse")
    db.execute("INSERT INTO person(id,name,created_at) VALUES(110,'Ilse',0)")
    db.execute(
        "INSERT INTO derived_file_person(file_id,person_id,run_id,model_id,model_version) "
        "VALUES(85,110,1,'insightface','v1')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO derived_file_person(file_id,person_id,run_id,model_id,model_version) "
            "VALUES(85,110,1,'insightface','v1')"
        )
    # A second RUN may say the same thing about the same picture. That is the
    # point: two clusterings coexist, and their agreeing is as informative as
    # their disagreeing.
    db.execute(
        "INSERT INTO derived_file_person(file_id,person_id,run_id,model_id,model_version) "
        "VALUES(85,110,2,'insightface','v2')"
    )


def test_people_can_be_ordered_by_image_count(db):
    """The query the old schema could not express: /faces/clusters ordered by
    cluster_id, and its size column counted detections rather than images."""
    tree(db)
    entity(db, 120, "person", "ilse")
    entity(db, 121, "person", "rook")
    db.execute("INSERT INTO person(id,name,created_at) VALUES(120,'Ilse',0)")
    db.execute("INSERT INTO person(id,name,created_at) VALUES(121,'Rook',0)")
    for i, pid in enumerate([120, 120, 120, 121]):
        fid = 200 + i
        a_file(db, fid, 1, f"n{fid}.png")
        db.execute(
            "INSERT INTO derived_file_person(file_id,person_id,run_id,model_id,model_version) "
            "VALUES(?,?,1,'insightface','v1')",
            (fid, pid),
        )
    rows = db.execute(
        "SELECT e.slug, COUNT(DISTINCT fp.file_id) AS image_count "
        "FROM person p JOIN entity e ON e.id = p.id "
        "JOIN derived_file_person fp ON fp.person_id = p.id "
        "GROUP BY p.id ORDER BY image_count DESC, e.slug"
    ).fetchall()
    assert rows[0] == ("ilse", 3), f"wrong order or wrong count: {rows}"


def test_a_companion_file_is_not_a_derivation(db):
    """The app already looks for a sidecar PNG carrying the graph a video
    cannot hold. That is an association, not descent, and conflating the two
    would put a parent-child edge where no generation happened."""
    tree(db)
    a_file(db, 160, 1, "clip.mp4")
    a_file(db, 161, 1, "clip.png")
    db.execute("INSERT INTO file_relation(file_id,related_id,kind,created_at) VALUES(160,161,'companion',0)")
    assert db.execute("SELECT count(*) FROM file_derivation").fetchone()[0] == 0
    found = db.execute(
        "SELECT f.name FROM file_relation r JOIN file f ON f.id = r.related_id "
        "WHERE r.file_id=160 AND r.kind='companion'"
    ).fetchone()[0]
    assert found == "clip.png"
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO file_relation(file_id,related_id,kind,created_at) VALUES(162,162,'sidecar',0)")


def test_trash_is_a_place_not_a_state(db):
    """A trashed file's bytes still exist; restore is a move. Excluding it by
    root ancestry beats matching paths against a configured string, which is
    how the old app had to do it."""
    db.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'/lib','library',0)")
    db.execute("INSERT INTO root(id,path,kind,created_at) VALUES(9,'/lib/.trash','trash',0)")
    entity(db, 1, "folder", "portraits")
    db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(1,1,NULL,'portraits',0)")
    entity(db, 9, "folder", "trashroot")
    db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(9,9,NULL,'.trash',0)")
    a_file(db, 170, 1, "keep.png")
    a_file(db, 171, 9, "gone.png")
    live = db.execute(
        "SELECT count(*) FROM file f JOIN folder fo ON fo.id=f.folder_id "
        "JOIN root r ON r.id=fo.root_id WHERE r.kind <> 'trash'"
    ).fetchone()[0]
    assert live == 1, "the trashed file was still counted as part of the library"
    assert db.execute("SELECT missing_since FROM file WHERE id=171").fetchone()[0] is None, (
        "a trashed file was marked missing; its bytes are present and restorable"
    )


def test_an_absent_prompt_gets_no_identity(db):
    """The old app synthesized a hash for files with no prompt and grouped
    files that shared nothing; clear_synthetic_prompt_hashes undid it. An
    empty prompt must be impossible to store, not merely discouraged."""
    entity(db, 180, "prompt", "p-real")
    db.execute("INSERT INTO prompt(id,text,text_hash,created_at) VALUES(180,'a brass helmet','h1',0)")
    entity(db, 181, "prompt", "p-empty")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO prompt(id,text,text_hash,created_at) VALUES(181,'','h2',0)")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO prompt(id,text,text_hash,created_at) VALUES(181,'   ','h3',0)")


def test_one_prompt_serves_positive_and_negative(db):
    """The role lives on the reference, not on the value: the same text used
    as a negative somewhere else is the same prompt."""
    tree(db)
    a_file(db, 185, 1, "g.png")
    entity(db, 186, "prompt", "p-pos")
    entity(db, 187, "prompt", "p-neg")
    db.execute("INSERT INTO prompt(id,text,text_hash,created_at) VALUES(186,'a brass helmet','hp',0)")
    db.execute("INSERT INTO prompt(id,text,text_hash,created_at) VALUES(187,'blurry, watermark','hn',0)")
    db.execute(
        "INSERT INTO generation(file_id,tool,detection,prompt_id,negative_id,parser,parsed_at) "
        "VALUES(185,'ComfyUI','graph',186,187,'comfy/3',0)"
    )
    pos, neg = db.execute(
        "SELECT p.text, n.text FROM generation g "
        "JOIN prompt p ON p.id = g.prompt_id JOIN prompt n ON n.id = g.negative_id "
        "WHERE g.file_id=185"
    ).fetchone()
    assert pos == "a brass helmet"
    assert neg == "blurry, watermark"


def test_one_name_makes_one_artifact_however_many_files_mention_it(db):
    """Almost all generation metadata is a string scraped from a PNG chunk, so
    content_hash is null and the *name* is what dedupes. Without this, a
    thousand files naming one checkpoint would make a thousand rows."""
    an_artifact(db, 190, "checkpoint", "Flux-Dev.safetensors")
    entity(db, 191, "artifact", "checkpoint-dupe")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO artifact(id,kind,name,name_key,first_seen_at) VALUES(?,?,?,?,0)",
            (191, "checkpoint", "flux-dev.safetensors", name_key("flux-dev.safetensors")),
        )
    # a different kind with the same name is a different thing
    an_artifact(db, 192, "lora", "Flux-Dev.safetensors")


def test_every_key_encountered_registers_itself(db):
    """A field nobody knows about is not searchable. The registry is kept by
    the database, not by remembering to update it."""
    tree(db)
    a_file(db, 195, 1, "a.jpg")
    a_file(db, 196, 1, "b.jpg")
    db.execute("INSERT INTO file_param(file_id,source,key,value_text) VALUES(195,'exif','LensMake','Fujifilm')")
    db.execute("INSERT INTO file_param(file_id,source,key,value_text,value_num) VALUES(196,'exif','LensMake','7',7)")
    kind, seen = db.execute(
        "SELECT value_kind, occurrences FROM param_key WHERE source='exif' AND key='LensMake'"
    ).fetchone()
    assert seen == 2, f"the registry lost count: {seen}"
    assert kind == "mixed", f"a key seen as both text and number should be mixed, got {kind}"
    db.execute("DELETE FROM file_param WHERE file_id=196")
    assert db.execute("SELECT occurrences FROM param_key WHERE key='LensMake'").fetchone()[0] == 1


def test_text_is_searchable_by_word_and_by_substring(db):
    """Two tokenizers because there are two questions: 'which images mention
    dusk' and 'which checkpoint has afeten in its filename'. The second is
    what the per-row fuzzykey UDFs were emulating."""
    tree(db)
    entity(db, 200, "prompt", "p1")
    db.execute("INSERT INTO prompt(id,text,text_hash,created_at) VALUES(200,'a brass diving helmet at dusk','h1',0)")
    hit = db.execute(
        "SELECT p.id FROM prompt_fts f JOIN prompt p ON p.id = f.rowid WHERE prompt_fts MATCH 'brass AND dusk'"
    ).fetchone()
    assert hit, "word search over prompts found nothing"
    assert hit[0] == 200, "word search over prompts found nothing"

    an_artifact(db, 201, "checkpoint", "flux-dev.safetensors")
    # the rowid IS the entity id, which is what makes a delete a lookup
    sub = db.execute("SELECT rowid FROM name_fts WHERE name_fts MATCH 'afeten'").fetchall()
    assert [r[0] for r in sub] == [201], f"substring search over names failed: {sub}"

    a_file(db, 202, 1, "z.jpg")
    db.execute("INSERT INTO file_param(file_id,source,key,value_text) VALUES(202,'iptc','Credit','Magnum Photos')")
    # param_fts is external content over file_param, so the file it belongs to
    # is read by joining on the rowid rather than from a copy in the index.
    tail = db.execute(
        "SELECT p.file_id FROM param_fts f JOIN file_param p ON p.rowid = f.rowid WHERE param_fts MATCH 'agnum'"
    ).fetchall()
    assert [r[0] for r in tail] == [202], "a scraped field was not searchable the day it appeared"


def test_deleting_a_prompt_removes_it_from_the_index(db):
    """An external-content FTS table that is not maintained returns rows for
    things that no longer exist."""
    entity(db, 205, "prompt", "p2")
    db.execute("INSERT INTO prompt(id,text,text_hash,created_at) VALUES(205,'a copper kettle','h2',0)")
    assert db.execute("SELECT count(*) FROM prompt_fts WHERE prompt_fts MATCH 'copper'").fetchone()[0] == 1
    db.execute("DELETE FROM prompt WHERE id=205")
    assert db.execute("SELECT count(*) FROM prompt_fts WHERE prompt_fts MATCH 'copper'").fetchone()[0] == 0


def test_a_folder_cannot_be_inserted_as_its_own_parent(db):
    """The guard was BEFORE UPDATE only, so a row naming itself was accepted:
    the foreign key is satisfied because the row exists by the time it is
    checked. Worse, once any cycle existed the guard's UNION ALL walk never
    terminated -- in a single-writer WAL database that is a write lock held
    forever, not an error."""
    tree(db)
    entity(db, 50, "folder", "selfparent")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(50,1,50,'self',0)")


def test_a_collection_cannot_be_inserted_as_its_own_parent(db):
    entity(db, 60, "collection", "selfcoll")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO collection(id,parent_id,name,kind,created_at) VALUES(60,60,'A','album',0)")


def test_a_folder_cannot_be_moved_across_roots(db):
    """The root guard was INSERT-only, so an UPDATE could straddle a subtree
    across two roots -- which silently breaks every 'exclude the trash subtree
    by ancestry' query."""
    tree(db)
    entity(db, 70, "folder", "otherroot")
    db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(70,2,NULL,'ext',0)")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE folder SET root_id=2 WHERE id=2")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE folder SET parent_id=70 WHERE id=2")


def test_feedback_outlives_the_thing_it_judged(db):
    """Restored after a regression: the foreign keys were flipped to CASCADE
    and the test asserting survival was deleted in the same edit, so nothing
    reported that deleting a file destroyed the human judgement about it."""
    tree(db)
    a_file(db, 900, 1, "j.png")
    db.execute(
        "INSERT INTO feedback(id,target_kind,file_id,annotation_kind,verdict,created_at) "
        "VALUES(1,'annotation',900,'caption','wrong',0)"
    )
    db.execute("DELETE FROM entity WHERE id=900")
    row = db.execute("SELECT file_id, verdict FROM feedback WHERE id=1").fetchone()
    assert row is not None, "deleting the judged file destroyed the judgement"
    assert row[0] is None, row
    assert row[1] == "wrong", row


def test_feedback_must_name_what_it_judges(db):
    """Enforced at write rather than as a row invariant, so ON DELETE SET NULL
    can detach a target without making the surviving row illegal."""
    tree(db)
    a_file(db, 901, 1, "k.png")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO feedback(id,target_kind,verdict,created_at) VALUES(2,'review','x',0)")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO feedback(id,target_kind,file_id,verdict,created_at) VALUES(3,'duplicate',901,'x',0)")


# ------------------------------------------------- the rescan matcher contract


def test_two_files_that_swap_names_keep_their_own_history(db):
    """The failure the naive matcher produces: both paths still exist, both
    lookups hit, and every rating ends up attached to a location."""
    tree(db)
    a_file(db, 800, 1, "a.png", sha="AAA")
    a_file(db, 801, 1, "b.png", sha="BBB")
    observed: dict[tuple[int, str], str | None] = {(1, "a.png"): "BBB", (1, "b.png"): "AAA"}
    result, missing = resolve_scan(db, observed)
    assert missing == [], f"a swap invented a missing file: {missing}"
    assert result[(1, "a.png")] == (Outcome.UNIQUE_MATCH, 801), result
    assert result[(1, "b.png")] == (Outcome.UNIQUE_MATCH, 800), result


def test_a_three_way_rotation_keeps_every_identity(db):
    tree(db)
    a_file(db, 810, 1, "x.png", sha="X")
    a_file(db, 811, 1, "y.png", sha="Y")
    a_file(db, 812, 1, "z.png", sha="Z")
    observed: dict[tuple[int, str], str | None] = {(1, "x.png"): "Z", (1, "y.png"): "X", (1, "z.png"): "Y"}
    result, missing = resolve_scan(db, observed)
    assert missing == []
    assert result[(1, "x.png")][1] == 812
    assert result[(1, "y.png")][1] == 810
    assert result[(1, "z.png")][1] == 811


def test_an_untouched_library_settles_without_reconciling(db):
    """Control: the cheap path must still be the cheap path."""
    tree(db)
    a_file(db, 820, 1, "p.png", sha="P")
    a_file(db, 821, 1, "q.png", sha="Q")
    result, missing = resolve_scan(db, {(1, "p.png"): "P", (1, "q.png"): "Q"})
    assert missing == []
    assert {k: v.outcome for k, v in result.items()} == {
        (1, "p.png"): Outcome.UNIQUE_MATCH,
        (1, "q.png"): Outcome.UNIQUE_MATCH,
    }


def test_identical_bytes_stay_ambiguous_rather_than_guessing(db):
    """Two byte-identical files, one moved out of band. sha256 proves equality,
    not continuity, so nothing may be attributed."""
    tree(db)
    a_file(db, 830, 1, "dup-a.png", sha="SAME")
    a_file(db, 831, 1, "dup-b.png", sha="SAME")
    result, missing = resolve_scan(db, {(2, "moved.png"): "SAME"})
    assert result[(2, "moved.png")].outcome is Outcome.AMBIGUOUS, result
    assert sorted(missing) == [830, 831], missing


def test_a_move_out_of_band_is_recognised_as_a_move(db):
    tree(db)
    a_file(db, 840, 1, "orig.png", sha="ONLY")
    result, missing = resolve_scan(db, {(2, "orig.png"): "ONLY"})
    assert result[(2, "orig.png")] == (Outcome.UNIQUE_MATCH, 840)
    assert missing == []


# ------------------------------------------------------------------ lineage


def test_a_remix_records_its_parent_before_the_child_exists(db):
    """Remix submits to ComfyUI and gets a job id back; the output file does
    not exist yet. The edge cannot wait for it, and a retry must not double."""
    tree(db)
    a_file(db, 300, 1, "parent.png")
    db.execute("INSERT INTO job(id,kind,state,created_at) VALUES(1,'remix','running',0)")
    db.execute(
        "INSERT INTO derivation_intent(id,parent_id,kind,external_ref,job_id,created_at) "
        "VALUES(1,300,'remix','comfy-prompt-abc',1,0)"
    )
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO derivation_intent(id,parent_id,kind,external_ref,created_at) "
            "VALUES(2,300,'remix','comfy-prompt-abc',0)"
        )
    a_file(db, 301, 1, "child.png")
    db.execute(
        "INSERT INTO file_derivation(id,intent_id,parent_id,child_id,kind,created_at) VALUES(1,1,300,301,'remix',0)"
    )
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO file_derivation(id,intent_id,parent_id,child_id,kind,created_at) VALUES(2,1,300,301,'remix',0)"
        )


# -------------------------------------------------------- presence & staleness


def test_a_missing_file_keeps_everything_authored(db):
    """Unreachable is not deleted. The old app defended this with three
    separate scan guards because a row's absence was the only way to say
    'gone'; here it is a column, so deletion stays a deliberate act."""
    tree(db)
    a_file(db, 700, 1, "offline.png", sha="aaa")
    db.execute("INSERT INTO user(id,username,password_hash,role,created_at) VALUES(1,'u','h','ADMIN',0)")
    db.execute("INSERT INTO rating(file_id,user_id,rating,created_at) VALUES(700,1,5,0)")
    db.execute("UPDATE file SET missing_since=123.0 WHERE id=700")
    row = db.execute("SELECT missing_since FROM file WHERE id=700").fetchone()
    assert row[0] == 123.0, "a file could not be marked missing"
    assert db.execute("SELECT rating FROM rating WHERE file_id=700").fetchone()[0] == 5, (
        "marking a file missing destroyed the rating attached to it"
    )
    db.execute("UPDATE file SET missing_since=NULL WHERE id=700")
    assert db.execute("SELECT missing_since FROM file WHERE id=700").fetchone()[0] is None


def test_an_ambiguous_content_match_moves_nothing(db):
    """Two files with identical bytes. One disappears from disk. sha256 proves
    byte equality, not object continuity, so nothing authored may follow it."""
    tree(db)
    a_file(db, 710, 1, "copy-a.png", sha="dup")
    a_file(db, 711, 2, "copy-b.png", sha="dup")
    db.execute("INSERT INTO user(id,username,password_hash,role,created_at) VALUES(1,'u','h','ADMIN',0)")
    db.execute("INSERT INTO rating(file_id,user_id,rating,created_at) VALUES(710,1,5,0)")
    candidates = db.execute("SELECT id FROM file WHERE content_sha256='dup'").fetchall()
    assert len(candidates) == 2, "the fixture no longer produces an ambiguous match"
    db.execute("UPDATE file SET missing_since=1.0 WHERE id=710")
    assert db.execute("SELECT count(*) FROM rating WHERE file_id=711").fetchone()[0] == 0, (
        "authored state crossed to a byte-identical but different file"
    )
    assert db.execute("SELECT rating FROM rating WHERE file_id=710").fetchone()[0] == 5


def test_derived_rows_go_stale_on_content_not_on_mtime(db):
    """mtime lies -- a backup restore or a sync client changes it while the
    bytes are identical, and the old app has a documented incident where
    exactly that wiped user state. Staleness is a content question."""
    tree(db)
    a_file(db, 720, 1, "e.png", sha="v1")
    db.execute(
        "INSERT INTO derived_file_hash(file_id,space_id,source_sha256,computed_at) VALUES(720,?,'v1',0)", (a_space(db),)
    )
    stale = (
        "SELECT count(*) FROM derived_file_hash d JOIN file f ON f.id = d.file_id "
        "WHERE d.source_sha256 IS NOT f.content_sha256"
    )
    db.execute("UPDATE file SET mtime=999999 WHERE id=720")
    assert db.execute(stale).fetchone()[0] == 0, "a touched file invalidated derived work for nothing"
    db.execute("UPDATE file SET content_sha256='v2' WHERE id=720")
    assert db.execute(stale).fetchone()[0] == 1, "replaced bytes left stale derived work looking current"


# ---------------------------------------------------------------- integrity


def test_a_file_cannot_live_in_a_folder_that_does_not_exist(db):
    tree(db)
    entity(db, 400, "file", "file-400")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,first_seen_at,last_seen_at) "
            "VALUES(400,999,'x.png','image',1,0,0,0)"
        )


def test_a_text_size_is_refused(db):
    tree(db)
    entity(db, 410, "file", "file-410")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,first_seen_at,last_seen_at) "
            "VALUES(410,1,'x.png','image','not-a-number',0,0,0)"
        )


def test_a_rating_is_between_one_and_five(db):
    tree(db)
    a_file(db, 420, 1, "r.png")
    db.execute("INSERT INTO user(id,username,password_hash,role,created_at) VALUES(1,'u','h','ADMIN',0)")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO rating(file_id,user_id,rating,created_at) VALUES(420,1,9,0)")


def test_a_claim_about_the_whole_frame_carries_no_coordinates(db):
    """A claim either points at a region or it does not.

    A `localizable` flag and a five-clause CHECK used to approximate this,
    and left a third state where a claim half-pointed at somewhere. It is
    structural now: `region_id` is present or NULL, and the combination the
    CHECK existed to reject cannot be written.
    """
    tree(db)
    a_file(db, 421, 1, "q.png")
    db.execute(
        "INSERT INTO derived_annotation(id,file_id,kind,text,model_id,model_version,"
        "source_sha256,computed_at) VALUES(1,421,'caption','a helmet','m','1','abc',0)"
    )
    assert db.execute("SELECT region_id FROM derived_annotation WHERE id=1").fetchone()[0] is None
    loose = [
        column
        for column in (row[1] for row in db.execute("PRAGMA table_info(derived_annotation)"))
        if column.startswith("bbox") or column in ("localizable", "mask_path")
    ]
    assert loose == [], f"geometry is back inline on the row: {loose}"


def test_deleting_an_entity_takes_its_subtype_and_its_ratings(db):
    tree(db)
    a_file(db, 430, 1, "c.png")
    db.execute("INSERT INTO user(id,username,password_hash,role,created_at) VALUES(1,'u','h','ADMIN',0)")
    db.execute("INSERT INTO rating(file_id,user_id,rating,created_at) VALUES(430,1,5,0)")
    db.execute("DELETE FROM entity WHERE id=430")
    assert db.execute("SELECT count(*) FROM file WHERE id=430").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM rating WHERE file_id=430").fetchone()[0] == 0


def test_feedback_names_a_durable_target(db):
    """Feedback survives every rebuild, so it must never point at a row a
    rebuild destroys. An earlier version carried an unconstrained target_ref --
    the exact polymorphic reference the entity registry exists to remove."""
    tree(db)
    a_file(db, 440, 1, "f.png")
    db.execute(
        "INSERT INTO feedback(id,target_kind,file_id,annotation_kind,verdict,created_at) "
        "VALUES(1,'annotation',440,'caption','wrong',0)"
    )
    # every target must name a durable thing, never a disposable row id
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO feedback(id,target_kind,verdict,created_at) VALUES(2,'annotation','wrong',0)")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO feedback(id,target_kind,file_id,verdict,created_at) VALUES(3,'duplicate',440,'wrong',0)"
        )
    # "the model was wrong about this file" is not actionable when the model
    # said four different things about it
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO feedback(id,target_kind,file_id,verdict,created_at) VALUES(4,'annotation',440,'wrong',0)"
        )
    # a judgement re-attaches to a re-run model by file plus what was judged
    db.execute(
        "INSERT INTO derived_annotation(id,file_id,kind,text,model_id,model_version,"
        "source_sha256,computed_at) VALUES(9,440,'caption','a helmet','m','2','abc',0)"
    )
    again = db.execute(
        "SELECT a.id FROM feedback f JOIN derived_annotation a "
        "ON a.file_id = f.file_id AND a.kind = f.annotation_kind WHERE f.id=1"
    ).fetchone()
    assert again, "feedback could not re-attach to a re-run model"
    assert again[0] == 9, "feedback could not re-attach to a re-run model"


def test_dropping_the_derived_namespace_keeps_everything_authored(db):
    """The rebuild contract: derived state is disposable by construction, and
    'drop derived' is mechanical rather than a hand-kept list."""
    tree(db)
    a_file(db, 450, 1, "d.png")
    db.execute("INSERT INTO user(id,username,password_hash,role,created_at) VALUES(1,'u','h','ADMIN',0)")
    db.execute("INSERT INTO rating(file_id,user_id,rating,created_at) VALUES(450,1,4,0)")
    db.execute(
        "INSERT INTO derived_file_hash(file_id,space_id,source_sha256,computed_at) VALUES(450,?,'abc',0)",
        (a_space(db),),
    )
    derived = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'derived_%'")]
    assert len(derived) >= 6, f"derived namespace too small to be the whole of it: {derived}"
    for table in derived:
        db.execute(f"DROP TABLE {table}")
    assert db.execute("SELECT count(*) FROM rating").fetchone()[0] == 1, (
        "dropping derived state destroyed authored state"
    )
    assert db.execute("SELECT count(*) FROM file").fetchone()[0] == 1
    assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_unresolved_artifacts_may_share_an_unknown_hash(db):
    """Metadata often names a checkpoint without a hash. Those are observed
    references, not resolved artifacts, and must not collide with each other."""
    an_artifact(db, 500, "checkpoint", "a")
    an_artifact(db, 501, "checkpoint", "b")
    assert db.execute("SELECT count(*) FROM artifact WHERE content_sha256 IS NULL").fetchone()[0] == 2
    an_artifact(db, 502, "checkpoint", "c", sha="deadbeef")
    with pytest.raises(sqlite3.IntegrityError):
        an_artifact(db, 503, "checkpoint", "d", sha="deadbeef")


# =========================== the maintenance layer ===========================
# Every test below covers a defect found by adversarial review. Each one passed
# the suite before it existed, which is the point of writing it down.


def test_replace_on_prompt_cannot_orphan_the_index(db):
    """DELETE triggers do not fire for rows removed by REPLACE conflict
    resolution -- recursive_triggers is off by default -- so INSERT OR REPLACE
    on the dedupe key orphaned an FTS entry pointing at a dead rowid, which
    once the id was reused attributed one image's prompt to another.

    The integrity-check MUST pass the rank argument: for an external-content
    table the content comparison only runs when it is non-zero, so the
    zero-argument form is a false green on exactly this corruption.
    """
    entity(db, 1, "prompt", "p1")
    db.execute("INSERT INTO prompt(id,text,text_hash,created_at) VALUES(1,'a copper kettle','h',0)")
    entity(db, 2, "prompt", "p2")
    db.execute("INSERT OR REPLACE INTO prompt(id,text,text_hash,created_at) VALUES(2,'a brass helmet','h',0)")
    assert db.execute("SELECT count(*) FROM prompt").fetchone()[0] == 1
    db.execute("INSERT INTO prompt_fts(prompt_fts, rank) VALUES('integrity-check', 1)")


def test_the_registry_survives_a_reparse(db):
    """Re-parsing is the normal case -- improving a parser is a re-parse of
    the database -- so writing the same field four times must leave the
    registry saying one, and the search index holding one entry."""
    tree(db)
    a_file(db, 9, 1, "a.jpg")
    for attempt in ("alpha", "beta", "gamma", "delta"):
        db.execute(
            "INSERT INTO file_param(file_id,source,key,value_text) "
            "VALUES(9,'exif','Lens',?) "
            "ON CONFLICT(file_id,source,key) DO UPDATE SET value_text = excluded.value_text",
            (attempt,),
        )
    occ = db.execute("SELECT occurrences FROM param_key WHERE key='Lens'").fetchone()[0]
    assert occ == 1, f"four re-parses of one row counted as {occ}"
    assert db.execute("SELECT count(*) FROM param_fts WHERE param_fts MATCH 'delta'").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM param_fts WHERE param_fts MATCH 'alpha'").fetchone()[0] == 0, (
        "the superseded value is still findable"
    )
    db.execute("INSERT INTO param_fts(param_fts, rank) VALUES('integrity-check', 1)")


def test_nothing_writes_file_param_with_replace():
    """The rule the counter and the search index both rest on.

    REPLACE fires no DELETE trigger and gives the replacement a new rowid, so
    `occurrences` drifts up forever and the FTS entry keyed on the old rowid
    is stranded. Absorbing that by recomputing from scratch is what cost a
    full scan per insert -- 1.5 ms per row at 8k rows and still doubling,
    against a flat 34 us/row once the writes are honest.

    It cannot be a trigger: SQLite runs BEFORE INSERT triggers before conflict
    resolution, so a guard there rejects `ON CONFLICT DO UPDATE` too -- the
    exact statement it exists to steer callers towards. Written as a trigger
    first, and that is how this test came to exist.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for path in sorted((root / "db").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for match in re.finditer(r"INSERT\s+OR\s+REPLACE\s+INTO\s+file_param", source, re.IGNORECASE):
            line = source[: match.start()].count("\n") + 1
            offenders.append(f"{path.name}:{line}")
    assert offenders == [], (
        f"these strand an FTS entry and inflate param_key: {offenders}. "
        f"Use ON CONFLICT(file_id, source, key) DO UPDATE."
    )


def test_an_upsert_is_not_mistaken_for_a_replace(db):
    """The control for the rule above: the supported statement must work, and
    must leave both the counter and the index saying one."""
    tree(db)
    a_file(db, 9, 1, "a.jpg")
    for value in ("a", "b", "c"):
        db.execute(
            "INSERT INTO file_param(file_id,source,key,value_text) VALUES(9,'exif','Lens',?)"
            " ON CONFLICT(file_id,source,key) DO UPDATE SET value_text = excluded.value_text",
            (value,),
        )
    assert db.execute("SELECT occurrences FROM param_key WHERE key='Lens'").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM file_param").fetchone()[0] == 1
    db.execute("INSERT INTO param_fts(param_fts, rank) VALUES('integrity-check', 1)")


def test_the_registry_counts_down_as_well_as_up(db):
    """An arithmetic counter is only right if both directions are wired: a
    key nobody uses any more is not a field the library contains."""
    tree(db)
    for file_id in (9, 10, 11):
        a_file(db, file_id, 1, f"{file_id}.jpg")
        db.execute(
            "INSERT INTO file_param(file_id,source,key,value_text) VALUES(?,'exif','Lens','x')",
            (file_id,),
        )
    assert db.execute("SELECT occurrences FROM param_key WHERE key='Lens'").fetchone()[0] == 3
    db.execute("DELETE FROM file_param WHERE file_id=9")
    assert db.execute("SELECT occurrences FROM param_key WHERE key='Lens'").fetchone()[0] == 2
    db.execute("DELETE FROM file_param")
    assert db.execute("SELECT count(*) FROM param_key WHERE key='Lens'").fetchone()[0] == 0


def test_a_key_learns_that_it_holds_both_kinds(db):
    """value_kind is a lattice that only widens, so it never needs the whole
    history re-read to answer a question with three possible values."""
    tree(db)
    a_file(db, 9, 1, "a.jpg")
    a_file(db, 10, 1, "b.jpg")
    db.execute("INSERT INTO file_param(file_id,source,key,value_text,value_num) VALUES(9,'exif','ISO','400',400)")
    assert db.execute("SELECT value_kind FROM param_key WHERE key='ISO'").fetchone()[0] == "number"
    db.execute("INSERT INTO file_param(file_id,source,key,value_text) VALUES(10,'exif','ISO','auto')")
    assert db.execute("SELECT value_kind FROM param_key WHERE key='ISO'").fetchone()[0] == "mixed"


def test_the_registry_follows_an_update(db):
    tree(db)
    a_file(db, 9, 1, "a.jpg")
    db.execute("INSERT INTO file_param(file_id,source,key,value_text) VALUES(9,'exif','Lens','alpha')")
    db.execute("UPDATE file_param SET key='NewLens', value_text='bravo' WHERE file_id=9")
    assert [r[0] for r in db.execute("SELECT key FROM param_key")] == ["NewLens"]
    assert db.execute("SELECT count(*) FROM param_fts WHERE param_fts MATCH 'alph'").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM param_fts WHERE param_fts MATCH 'brav'").fetchone()[0] == 1


def test_deleting_one_source_leaves_the_other_indexed(db):
    """The file_param key is (file_id, source, key); a delete predicate without
    `source` wiped the XMP row's index entry when the IPTC row went."""
    tree(db)
    a_file(db, 9, 1, "a.jpg")
    db.execute("INSERT INTO file_param(file_id,source,key,value_text) VALUES(9,'iptc','Credit','Reuters')")
    db.execute("INSERT INTO file_param(file_id,source,key,value_text) VALUES(9,'xmp','Credit','Associated')")
    db.execute("DELETE FROM file_param WHERE source='iptc'")
    assert db.execute("SELECT count(*) FROM file_param").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM param_fts WHERE param_fts MATCH 'ssociat'").fetchone()[0] == 1


def test_a_key_nobody_uses_leaves_the_registry(db):
    tree(db)
    a_file(db, 9, 1, "a.jpg")
    db.execute("INSERT INTO file_param(file_id,source,key,value_text) VALUES(9,'exif','Ghost','g')")
    db.execute("DELETE FROM file_param WHERE key='Ghost'")
    assert db.execute("SELECT count(*) FROM param_key WHERE key='Ghost'").fetchone()[0] == 0


def test_the_registry_records_when_it_learned_a_key(db):
    tree(db)
    a_file(db, 9, 1, "a.jpg")
    db.execute("INSERT INTO file_param(file_id,source,key,value_text) VALUES(9,'exif','T','x')")
    first, last = db.execute("SELECT first_seen_at, last_seen_at FROM param_key WHERE key='T'").fetchone()
    assert first > 0, f"timestamps were never written: {(first, last)}"
    assert last > 0, f"timestamps were never written: {(first, last)}"


def test_a_subtype_cannot_disagree_with_its_entity(db):
    """The foreign key proved the entity existed; nothing said the entity's kind
    matched the table actually holding the row."""
    tree(db)
    entity(db, 300, "person", "notafolder")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(300,1,NULL,'x',0)")


def test_deleting_a_subtype_takes_its_entity(db):
    """An entity with no subtype is an address that resolves to nothing, and it
    squats its slug forever."""
    tree(db)
    entity(db, 301, "folder", "temp")
    db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(301,1,NULL,'temp',0)")
    db.execute("DELETE FROM folder WHERE id=301")
    assert db.execute("SELECT count(*) FROM entity WHERE id=301").fetchone()[0] == 0


def test_the_name_index_follows_a_rename(db):
    tree(db)
    a_file(db, 9, 1, "dusk.png")
    db.execute("UPDATE file SET name='dawnlight.png' WHERE id=9")
    stale = db.execute("SELECT count(*) FROM name_fts WHERE name_fts MATCH ?", ('"dusk"',)).fetchone()[0]
    fresh = db.execute("SELECT count(*) FROM name_fts WHERE name_fts MATCH ?", ('"dawnlight"',)).fetchone()[0]
    assert stale == 0, "the index still answers with the old name"
    assert fresh == 1


def test_every_named_kind_is_indexed(db):
    """Two of six kinds were indexed while the comment claimed the index covered
    'anything addressable'."""
    tree(db)
    a_file(db, 9, 1, "picture.png")
    entity(db, 310, "person", "ilse")
    db.execute("INSERT INTO person(id,name,created_at) VALUES(310,'Ilse Bergman',0)")
    entity(db, 311, "collection", "cw")
    db.execute("INSERT INTO collection(id,name,kind,created_at) VALUES(311,'Client Work','album',0)")
    an_artifact(db, 312, "checkpoint", "flux-dev")
    for needle in ('"ergman"', '"ient Wo"', '"lux-de"', '"ictur"', '"ortrait"'):
        n = db.execute("SELECT count(*) FROM name_fts WHERE name_fts MATCH ?", (needle,)).fetchone()[0]
        assert n >= 1, f"nothing indexed for {needle}"


def test_recomputing_a_derived_row_is_not_an_append(db):
    """An interrupted job re-run must not triple every face and every review."""
    tree(db)
    a_file(db, 9, 1, "a.png")
    sid = db.execute(
        "INSERT INTO similarity_space(key,representation,dimensions,metric,"
        "producer,producer_version,preprocess,preprocess_version,spec_hash,created_at)"
        " VALUES('semantic.test','float32',1,'cosine','p','1','pp','1',hex(randomblob(8)),0)"
    ).lastrowid
    for _ in range(3):
        db.execute(
            "INSERT OR REPLACE INTO derived_embedding(file_id,space_id,vector,source_sha256,computed_at)"
            " VALUES(9,?,x'00000000','s',0)",
            (sid,),
        )
    assert db.execute("SELECT count(*) FROM derived_embedding").fetchone()[0] == 1


def test_whitespace_is_not_a_prompt(db):
    """trim() with one argument strips the space character only, so a tab, a
    newline, U+00A0 and U+3000 each bought a manufactured identity."""
    for i, blank in enumerate(["", "   ", "\n", "\t", "\u00a0", "\u3000"]):
        entity(db, 400 + i, "prompt", f"blank{i}")
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO prompt(id,text,text_hash,created_at) VALUES(?,?,?,0)",
                (400 + i, blank, f"h{i}"),
            )


def test_a_file_cannot_derive_from_itself(db):
    tree(db)
    a_file(db, 9, 1, "a.png")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO file_derivation(id,parent_id,child_id,kind,created_at) VALUES(1,9,9,'remix',0)")
    a_file(db, 10, 1, "b.png")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO file_derivation(id,parent_id,child_id,kind,created_at) VALUES(2,9,10,'not-a-real-kind',0)"
        )


def test_a_slug_can_be_retired_more_than_once(db):
    """Rename A away, let C take the freed slug, rename C away: the second
    retirement must be recordable or that redirect is simply lost."""
    entity(db, 500, "person", "temp-a")
    entity(db, 501, "person", "temp-b")
    db.execute("INSERT INTO slug_history(kind,slug,entity_id,retired_at) VALUES('person','ilse',500,1.0)")
    db.execute("INSERT INTO slug_history(kind,slug,entity_id,retired_at) VALUES('person','ilse',501,2.0)")
    latest = db.execute(
        "SELECT entity_id FROM slug_history WHERE kind='person' AND slug='ilse' ORDER BY retired_at DESC LIMIT 1"
    ).fetchone()[0]
    assert latest == 501


def test_a_case_only_rename_is_not_a_second_file(db):
    """On the stated platform 'A.png' and 'a.png' are one file, so
    case-sensitive uniqueness produced two rows, one permanently missing."""
    tree(db)
    a_file(db, 9, 1, "Dusk.PNG")
    entity(db, 10, "file", "file-10")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,first_seen_at,last_seen_at) "
            "VALUES(10,1,'dusk.png','image',1,0,0,0)"
        )


def test_a_missing_file_does_not_hold_its_path_against_a_live_one(db):
    """A path is exclusive only while the bytes are there.

    The missing row keeps its last known path so the app can say where the
    file used to be. Enforcing uniqueness over that stale path meant a
    directory whose contents had been swapped out failed to scan at all: the
    departed row still owned the name the arriving file needed.
    """
    tree(db)
    a_file(db, 9, 1, "dusk.png", sha="aa")
    db.execute("UPDATE file SET missing_since = 100 WHERE id = 9")
    entity(db, 10, "file", "file-10")
    db.execute(
        "INSERT INTO file(id,folder_id,name,kind,size,mtime,content_sha256,"
        "first_seen_at,last_seen_at) VALUES(10,1,'dusk.png','image',10,0,'bb',0,0)"
    )
    live = db.execute("SELECT id FROM file WHERE name='dusk.png' AND missing_since IS NULL").fetchall()
    assert live == [(10,)], "exactly one file may be live at a path"

    # and the exclusion still bites for two live rows -- otherwise this test
    # would pass just as well against an index that was dropped entirely
    entity(db, 11, "file", "file-11")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,content_sha256,"
            "first_seen_at,last_seen_at) VALUES(11,1,'dusk.png','image',10,0,'cc',0,0)"
        )


def test_booleans_are_zero_or_one(db):
    """STRICT constrains storage class, not domain: localizable = 2 silently
    took the not-localizable branch of its own CHECK."""
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO root(id,path,kind,online,created_at) VALUES(9,'/x','library',7,0)")


def test_a_region_cannot_describe_an_impossible_rectangle(db):
    """Four coordinates that only mean something together were four columns
    nothing checked, in four tables. One table now, so one place says what a
    rectangle is -- and says it in fractions of the frame, because a box in
    pixels points somewhere else on a thumbnail."""
    tree(db)
    for x, y, w, h in (
        (0.9, 0.1, 0.5, 0.1),  # runs off the right edge
        (0.1, 0.9, 0.1, 0.5),  # runs off the bottom
        (-0.1, 0.1, 0.2, 0.2),  # starts outside the frame
        (0.1, 0.1, 0.0, 0.2),  # zero width locates nothing
        (0.1, 0.1, 0.2, -0.2),  # negative height
    ):
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO region(x,y,w,h) VALUES(?,?,?,?)", (x, y, w, h))
    # the control: a rectangle actually inside the frame is accepted
    db.execute("INSERT INTO region(x,y,w,h) VALUES(0.25,0.25,0.5,0.5)")


def test_an_unreferenced_blob_is_reclaimed(db):
    """Deduplicated payloads never became collectable, so the table grew for the
    life of the library."""
    tree(db)
    a_file(db, 9, 1, "a.png")
    db.execute("INSERT INTO blob(hash,payload,byte_len) VALUES('h','{}',2)")
    db.execute("INSERT INTO file_blob(file_id,carrier,slot,blob_hash,seen_at) VALUES(9,'png_text','workflow','h',0)")
    db.execute("DELETE FROM entity WHERE id=9")
    assert db.execute("SELECT count(*) FROM blob").fetchone()[0] == 0


def test_the_database_states_its_version(db):
    """The DDL's stamp is the version, and it must be the one this build opens.

    `>= 1` was the assertion here, and under it schema.sql stamped v1 while
    connect.USER_VERSION said v3 -- every database built by anything other
    than db.build (which re-stamped afterwards) was refused by check_version
    with no step registered to move it forward.
    """
    from db.connect import APPLICATION_ID, USER_VERSION

    assert db.execute("PRAGMA user_version").fetchone()[0] == USER_VERSION
    assert db.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID


def test_every_version_left_behind_has_a_step_off_it():
    """A bump with no step is a database this build cannot open.

    The registry is keyed on where a step starts, so every version from the
    first to the one before current needs one. At v1 this passes with nothing
    registered, and it fails the moment USER_VERSION moves without a step.
    """
    from db.connect import USER_VERSION
    from db.migrate import STEPS

    missing = [v for v in range(1, USER_VERSION) if v not in STEPS]
    assert not missing, (
        f"USER_VERSION is {USER_VERSION} but no migration leaves v{missing}. "
        f"A database at that version cannot be opened or upgraded."
    )

    # The control. At v1 the loop above is `range(1, 1)` -- it asserts over an
    # empty list and cannot fail, which is exactly the state that would let a
    # bump ship with no step and nothing say a word. This asks the same
    # question of the version after this one and requires the answer to be no.
    ahead = [v for v in range(1, USER_VERSION + 1) if v not in STEPS]
    assert ahead, (
        f"a step off v{USER_VERSION} is already registered, so this check has "
        f"nothing left to catch -- did USER_VERSION forget to move with it?"
    )


def test_the_front_page_query_uses_an_index(db):
    """'Newest first' is the default view of a gallery. Without an index every
    page load sorts the whole table in a temp B-tree."""
    plan = " ".join(
        r[3]
        for r in db.execute(
            "EXPLAIN QUERY PLAN SELECT id FROM file WHERE missing_since IS NULL ORDER BY mtime DESC LIMIT 60"
        )
    )
    assert "TEMP B-TREE" not in plan.upper(), f"front page still sorts the whole table: {plan}"
    assert "file_recent" in plan, plan


def test_every_foreign_key_states_its_delete_action(db):
    """A census, because the delete-action split drifted mid-review with nothing
    noticing. The point is not the totals; it is that none is unspecified."""
    virt = virtual_table_names(db)
    names = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")} - virt
    actions = [row[6] for t in names for row in db.execute(f"PRAGMA foreign_key_list({t})")]
    assert actions, "no foreign keys found; the sweep is broken"
    unspecified = sorted({a for a in actions if a not in ("CASCADE", "SET NULL", "RESTRICT")})
    assert unspecified == [], f"foreign keys with no stated delete action: {unspecified}"


def test_every_closed_vocabulary_rejects_a_stranger(db):
    """Nine CHECK...IN lists had no negative test at all, so emptying them went
    undetected across seventy mutations."""
    tree(db)
    a_file(db, 9, 1, "a.png")
    db.execute("INSERT INTO blob(hash,payload,byte_len) VALUES('bh','{}',2)")
    cases = [
        "INSERT INTO entity(id,uuid,kind,slug) VALUES(90,x'0000000000000000000000000000005a','nope','s')",
        "INSERT INTO root(id,path,kind,created_at) VALUES(90,'/z','nope',0)",
        "INSERT INTO job(id,kind,state,created_at) VALUES(90,'scan','nope',0)",
        "INSERT INTO file_param(file_id,source,key,value_text) VALUES(9,'nope','k','v')",
        # the blob must exist, or this raises on the foreign key and passes for
        # the wrong reason -- an over-determined negative proves nothing
        "INSERT INTO file_blob(file_id,carrier,slot,blob_hash,seen_at) VALUES(9,'nope','s','bh',0)",
        "INSERT INTO derived_media_sample(id,file_id,kind,policy) VALUES(90,9,'nope','p')",
        "INSERT INTO user(id,username,password_hash,role,created_at) VALUES(90,'u','h','nope',0)",
    ]
    for sql in cases:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(sql)


def test_an_unhashed_file_does_not_read_as_current(db):
    """The staleness comparison was `<>`, which is NULL-blind -- and NULL is the
    normal state, because hashing is deferred until the cheap lookup misses.
    Every derived row attached to an unhashed file read as current forever."""
    tree(db)
    a_file(db, 9, 1, "e.png", sha=None)
    db.execute(
        "INSERT INTO derived_file_hash(file_id,space_id,source_sha256,computed_at) VALUES(9,?,'v1',0)", (a_space(db),)
    )
    stale = (
        "SELECT count(*) FROM derived_file_hash d JOIN file f ON f.id = d.file_id "
        "WHERE d.source_sha256 IS NOT f.content_sha256"
    )
    assert db.execute(stale).fetchone()[0] == 1, "a derived row on a file with no content hash reported itself current"


def test_the_built_database_matches_the_ddl(tmp_path):
    """The built file drifted a whole generation behind schema.sql and nothing
    reported it: the suite loads the DDL into :memory:, so it never reads the
    file it claims to describe.

    Both files are checked. The developer's `db/gallery.db` is the one that
    drifts, and it is gitignored -- so on CI and on a fresh clone this test
    skipped, and the only check that reads a built file never ran anywhere it
    mattered. Building one here means the build path itself is always
    exercised.
    """
    from db.build import DEFAULT, build, drift

    made = tmp_path / "gallery.db"
    build(made)
    assert drift(made) == [], "the build path does not produce the DDL's database"

    if not DEFAULT.exists():
        pytest.skip("no local build to compare; the built-from-scratch check above ran")
    assert drift(DEFAULT) == []


def test_the_drift_check_can_actually_fail(tmp_path):
    """Control: a database built from a mutated DDL must be reported as drifted."""
    import sqlite3 as _s

    from db.build import drift
    from db.connect import schema_sql

    path = tmp_path / "stale.db"
    conn = _s.connect(str(path))
    conn.executescript(schema_sql().replace("CREATE INDEX file_kind ON file(kind);", ""))
    conn.commit()
    conn.close()
    assert drift(path) != [], "the drift check cannot see a missing index"


def test_the_drift_check_sees_inside_string_literals(tmp_path):
    """Control for the comparator's literal-awareness: a trigger message
    whose spacing changed is a different message and must read as drift,
    while spacing between TOKENS stays fold-away noise."""
    import sqlite3 as _s

    from db.build import drift
    from db.connect import schema_sql

    reworded = tmp_path / "reworded.db"
    conn = _s.connect(str(reworded))
    conn.executescript(schema_sql().replace("nothing is filed into it", "nothing  is filed into it"))
    conn.commit()
    conn.close()
    assert drift(reworded) != [], "a changed literal was folded into equality"

    respaced = tmp_path / "respaced.db"
    conn = _s.connect(str(respaced))
    conn.executescript(
        schema_sql().replace(
            "CREATE TRIGGER collection_file_not_into_smart", "CREATE  TRIGGER  collection_file_not_into_smart"
        )
    )
    conn.commit()
    conn.close()
    assert drift(respaced) == [], "token spacing is not drift"


def test_the_squeeze_survives_an_apostrophe_in_a_comment():
    """A DDL comment's apostrophe ("the group's seed") flipped the old
    quote-parity split, so every literal downstream of it was misread as
    plain SQL and its internal spacing folded -- the exact false negative
    the literal-aware comparator exists to prevent, alive again for 14
    shipped literals that sit after odd-apostrophe comment regions."""
    from db.build import _squeezed

    said = "CREATE TABLE t ( -- the group's seed\n  x TEXT CHECK (x = 'a  b'))"
    assert _squeezed(said) != _squeezed(said.replace("'a  b'", "'a b'")), (
        "a literal changed behind a commented apostrophe was folded into equality"
    )
    assert _squeezed(said) == _squeezed(said.replace("  x TEXT", " x  TEXT")), "token spacing is not drift"


def test_the_squeeze_keeps_comment_ends_and_quoted_names():
    """A line comment ends at its newline: folding that newline away would
    read "-- note\\n+ 2" and "-- note + 2" -- different SQL -- as equal.
    And a double-quoted name's spacing is content, same as a literal's."""
    from db.build import _squeezed

    assert _squeezed("SELECT 1 -- note\n+ 2") != _squeezed("SELECT 1 -- note + 2")
    assert _squeezed('CREATE TABLE "a  b" (x)') != _squeezed('CREATE TABLE "a b" (x)')


def test_the_drift_check_sees_a_wrong_stamp(tmp_path):
    """Control for the half it could not see. `objects()` reads sqlite_master
    only, so a file carrying the wrong version -- the case the stamps exist
    for -- was reported as in sync with the DDL that does not stamp it."""
    import sqlite3 as _s

    from db.build import drift
    from db.connect import schema_sql

    path = tmp_path / "misstamped.db"
    conn = _s.connect(str(path))
    conn.executescript(schema_sql())
    conn.execute("PRAGMA user_version = 99")
    conn.execute("PRAGMA application_id = 0")
    conn.commit()
    conn.close()
    assert any("stamped" in line for line in drift(path)), drift(path)


def test_deleting_a_param_removes_it_from_the_index(db):
    """Found by mutation: dropping param_fts_delete went undetected, because
    every existing test only ever added rows."""
    tree(db)
    a_file(db, 9, 1, "a.jpg")
    db.execute("INSERT INTO file_param(file_id,source,key,value_text) VALUES(9,'exif','Lens','alphabet')")
    assert db.execute("SELECT count(*) FROM param_fts WHERE param_fts MATCH 'phabe'").fetchone()[0] == 1
    db.execute("DELETE FROM file_param WHERE file_id=9")
    assert db.execute("SELECT count(*) FROM param_fts WHERE param_fts MATCH 'phabe'").fetchone()[0] == 0


def test_folder_depth_is_maintained_by_the_database(db):
    """Found by mutation: both depth triggers could be deleted with nothing
    failing, because a fixture merely named the column in an INSERT."""
    tree(db)
    entity(db, 320, "folder", "deep")
    # a deliberately wrong depth on the way in must be corrected
    db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(320,1,2,'deep',99)")
    assert db.execute("SELECT depth FROM folder WHERE id=320").fetchone()[0] == 2
    db.execute("UPDATE folder SET parent_id=NULL WHERE id=320")
    assert db.execute("SELECT depth FROM folder WHERE id=320").fetchone()[0] == 0, (
        "a reparented folder kept its old depth"
    )

    # Reparenting moves a subtree. This test used to reparent a leaf, so the
    # descendants stayed one level wrong and nothing said so.
    entity(db, 321, "folder", "deeper")
    entity(db, 322, "folder", "deepest")
    db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(321,1,320,'deeper',0)")
    db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(322,1,321,'deepest',0)")
    assert [r[0] for r in db.execute("SELECT depth FROM folder WHERE id IN (320,321,322)")] == [
        0,
        1,
        2,
    ]

    db.execute("UPDATE folder SET parent_id=2 WHERE id=320")
    assert [r[0] for r in db.execute("SELECT depth FROM folder WHERE id IN (320,321,322)")] == [
        2,
        3,
        4,
    ], "the subtree under a reparented folder kept its old depth"


def test_a_file_cannot_disagree_with_its_entity(db):
    """Found by mutation: only the folder subtype's kind guard was covered, so
    file_kind_agrees could be deleted unnoticed. Every subtype needs its own."""
    tree(db)
    entity(db, 330, "collection", "notafile")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,first_seen_at,last_seen_at) "
            "VALUES(330,1,'x.png','image',1,0,0,0)"
        )


def test_two_prompts_cannot_share_a_text_hash(db):
    """Found by mutation: the dedupe key could lose UNIQUE with nothing failing.
    The dedupe trigger silently swallows the second insert, so the constraint
    itself needs a direct negative -- an UPDATE, which the trigger does not see."""
    entity(db, 340, "prompt", "p-a")
    entity(db, 341, "prompt", "p-b")
    db.execute("INSERT INTO prompt(id,text,text_hash,created_at) VALUES(340,'first','h1',0)")
    db.execute("INSERT INTO prompt(id,text,text_hash,created_at) VALUES(341,'second','h2',0)")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE prompt SET text_hash='h1' WHERE id=341")


def test_a_reference_must_agree_with_what_it_points_at(db):
    """A foreign key proves the row exists and says nothing about whether it is
    the right kind of thing. All four of these were accepted: a camera as a
    checkpoint, a lens as a workflow, a face citing another file's frame, and a
    finding sitting on a file its own review never looked at."""
    tree(db)
    a_file(db, 9, 1, "a.png")
    a_file(db, 10, 1, "b.png")
    an_artifact(db, 600, "camera", "X-T5")
    an_artifact(db, 601, "lens", "XF 35mm")

    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO file_artifact(file_id,artifact_id,role) VALUES(9,600,'checkpoint')")
    # the same camera in its own role is fine
    db.execute("INSERT INTO file_artifact(file_id,artifact_id,role) VALUES(9,600,'captured_with')")

    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO generation(file_id,tool,detection,workflow_id,parser,parsed_at) "
            "VALUES(9,'ComfyUI','graph',601,'p',0)"
        )

    db.execute("INSERT INTO derived_media_sample(id,file_id,kind,policy) VALUES(900,10,'frame','p')")
    db.execute("INSERT INTO region(id,x,y,w,h) VALUES(700,0,0,1,1)")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO derived_face_instance(id,file_id,sample_id,region_id,"
            "model_id,model_version,source_sha256,computed_at) VALUES(1,9,900,700,'m','1','s',0)"
        )

    # a caption may cite a frame, but only a frame of the file it describes
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO derived_annotation(id,file_id,sample_id,kind,text,model_id,"
            "model_version,source_sha256,computed_at) "
            "VALUES(1,9,900,'caption','wrong film','m','1','s',0)"
        )
    db.execute(
        "INSERT INTO derived_annotation(id,file_id,sample_id,kind,text,model_id,"
        "model_version,source_sha256,computed_at) "
        "VALUES(2,10,900,'caption','right film','m','1','s',0)"
    )


def test_a_name_survives_dropping_everything_derived(db):
    """The rebuild contract, end to end. The naming a human did is authored and
    must outlive the evidence that suggested it -- otherwise 'drop derived and
    re-index' quietly discards the only irreplaceable thing in the face
    pipeline. Sampling policies and inferred membership are derived and go."""
    tree(db)
    a_file(db, 9, 1, "a.png")
    entity(db, 700, "person", "ilse")
    db.execute("INSERT INTO person(id,name,created_at) VALUES(700,'Ilse',0)")
    # a human said so
    db.execute("INSERT INTO person_assertion(person_id,file_id,user_id,created_at) VALUES(700,9,NULL,0)")
    # a backend inferred it
    db.execute(
        "INSERT INTO derived_file_person(file_id,person_id,run_id,model_id,model_version) "
        "VALUES(9,700,1,'insightface','v1')"
    )
    derived = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'derived_%'")]
    for t in derived:
        db.execute(f"DROP TABLE {t}")

    assert db.execute("SELECT name FROM person WHERE id=700").fetchone()[0] == "Ilse"
    assert db.execute("SELECT count(*) FROM person_assertion WHERE person_id=700").fetchone()[0] == 1
    assert db.execute("PRAGMA foreign_key_check").fetchall() == []
    # and the People page still answers, from assertions alone
    rows = db.execute(
        "SELECT p.name, COUNT(DISTINCT pa.file_id) FROM person p "
        "JOIN person_assertion pa ON pa.person_id = p.id GROUP BY p.id"
    ).fetchall()
    assert rows == [("Ilse", 1)]


def test_no_index_is_a_prefix_of_another(db):
    """An index whose columns are a prefix of another index's, under the same
    partial predicate, is write cost on every insert for a read the other one
    already serves.

    Stated as a prefix rule rather than measured by dropping and replanning.
    A planner probe on the leading column alone condemns any composite index
    whose first column is also covered by a narrower one -- it flagged
    `file_param_key_num`, which exists for `key = ? AND value_num BETWEEN ?`
    and is not redundant at all.

    The predicates have to match too: an index over all rows is not replaced
    by a partial one, because the partial one cannot answer for the rows it
    excludes.
    """

    def shape(index):
        columns = [row[2] for row in db.execute(f"PRAGMA index_xinfo({index})") if row[5]]
        sql = db.execute("SELECT sql FROM sqlite_master WHERE name=?", (index,)).fetchone()[0]
        where = sql.upper().split(" WHERE ", 1)
        return columns, (where[1].strip() if len(where) > 1 else None)

    redundant = []
    for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        if table in virtual_table_names(db):
            continue
        indexes = [
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL",
                (table,),
            )
        ]
        for candidate in indexes:
            columns, predicate = shape(candidate)
            for other in indexes:
                if other == candidate:
                    continue
                wider, other_predicate = shape(other)
                if len(wider) > len(columns) and wider[: len(columns)] == columns and other_predicate == predicate:
                    redundant.append(f"{candidate} is a prefix of {other}")
    assert redundant == [], f"these indexes earn nothing: {redundant}"


def test_a_guard_that_only_fires_on_insert_is_declared_as_such(db):
    """A rule enforced on INSERT and not on UPDATE is one statement from
    useless: write the row correctly, then change it.

    Four were bypassable this way -- a camera's role updated to 'checkpoint',
    a workflow reference repointed at a LoRA, and two derived rows updated to
    cite a frame from a different file.

    `feedback_names_a_target` is the deliberate exception and says so in the
    schema: FK cascade actions DO fire UPDATE triggers, so guarding it on
    UPDATE would abort the ON DELETE SET NULL that detaches a judged target,
    and losing the human judgement is worse than holding a nulled pointer.
    """
    insert_only = []
    guards = {}
    for name, sql in db.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='trigger' AND sql LIKE '%RAISE(ABORT%'"
    ):
        match = re.search(r"(?:BEFORE|AFTER)\s+(\w+)[^;]*?\sON\s+(\w+)", sql, re.DOTALL)
        if not match:
            continue
        event, table = match.group(1).upper(), match.group(2)
        message = re.search(r"RAISE\(ABORT,\s*'([^']+)'", sql)
        guards.setdefault((table, message.group(1) if message else name), set()).add(event)
    for (table, message), events in sorted(guards.items()):
        if "UPDATE" not in events and f"{table}: {message}" not in _INSERT_ONLY_ON_PURPOSE:
            insert_only.append(f"{table}: {message}")

    assert insert_only == [], (
        f"these rules are bypassed by an UPDATE: {insert_only}. Add the UPDATE "
        f"counterpart, or add the guard to _INSERT_ONLY_ON_PURPOSE with the reason."
    )


def test_a_subtype_row_cannot_be_repointed_at_another_entity(db):
    """Written expecting the foreign key to cover this. It does not.

    The FK only proves the target entity exists, and entity 1 does -- it is
    just a folder. The rejection observed while reasoning about this came
    from a row referencing the file, not from the supertype, so a file with
    nothing pointing at it moved onto a folder's entity and stood.
    """
    tree(db)
    a_file(db, 9, 1, "a.png")
    with pytest.raises(sqlite3.IntegrityError, match="does not match file"):
        db.execute("UPDATE file SET id=1 WHERE id=9")  # 1 is a folder's entity


def test_an_entity_cannot_change_what_it_is(db):
    """Six triggers check a subtype row sits on an entity of the matching
    kind, all on INSERT. Guarding the supertype closes all six: without it
    `UPDATE entity SET kind` left the file row and its entity disagreeing,
    and nothing reported it."""
    tree(db)
    a_file(db, 9, 1, "a.png")
    with pytest.raises(sqlite3.IntegrityError, match="cannot change kind"):
        db.execute("UPDATE entity SET kind='folder' WHERE id=9")
    # the control: renaming is what entities DO change, and must still work
    db.execute("UPDATE entity SET slug='renamed' WHERE id=9")
    assert db.execute("SELECT slug FROM entity WHERE id=9").fetchone()[0] == "renamed"


def test_a_role_cannot_be_updated_into_a_lie(db):
    """The insert-side rule with the statement that used to undo it."""
    tree(db)
    a_file(db, 9, 1, "a.png")
    an_artifact(db, 600, "camera", "X-T5")
    db.execute("INSERT INTO file_artifact(file_id,artifact_id,role) VALUES(9,600,'captured_with')")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("UPDATE file_artifact SET role='checkpoint' WHERE file_id=9")


def test_every_foreign_key_column_can_be_looked_up_by(db):
    """SQLite's own `.lint fkey-indexes`, as a gate.

    Deleting a parent row makes SQLite run `SELECT 1 FROM child WHERE
    child_key = ?` against every child table (src/shell.c.in:5981-6014).
    Unindexed that is a full scan per delete, so removing one file walks
    every derivation, every annotation and every piece of feedback in the
    library -- work that is invisible until the library is large.
    """
    unindexed = []
    for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        if table in virtual_table_names(db):
            continue
        leading = set()
        for (index,) in db.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=?", (table,)):
            columns = list(db.execute(f"PRAGMA index_info({index})"))
            if columns:
                leading.add(columns[0][2])
        primary = {r[1] for r in db.execute(f"PRAGMA table_info({table})") if r[5]}
        for row in db.execute(f"PRAGMA foreign_key_list({table})"):
            column = row[3]
            if column not in leading and column not in primary:
                unindexed.append(f"{table}.{column} -> {row[2]}")
    assert unindexed == [], f"deleting a parent row scans these child tables: {unindexed}"


def test_a_column_naming_a_fixed_set_is_constrained_to_it(db):
    """An unconstrained enum accepts every typo, and every typo is a row that
    never matches the filter that was meant to find it.

    `param_key.source` and `file_param.source` name the same set, and
    `slug_history.kind` and `entity.kind` name the same set; the registry and
    the history were the two that carried no CHECK, so the pair enforced in
    one place and not the other could drift on the first direct write.
    """
    unconstrained = []
    for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        if table in virtual_table_names(db):
            continue
        sql = db.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0]
        for row in db.execute(f"PRAGMA table_info({table})"):
            column, kind = row[1], row[2]
            if kind != "TEXT" or column in _FREE_TEXT:
                continue
            if not re.search(
                r"(kind|state|role|verdict|space|carrier|source|severity|sex|policy)$",
                column,
            ):
                continue
            if not re.search(rf"\b{column}\b[^,]*CHECK|CHECK\s*\(\s*{column}\b", sql):
                unconstrained.append(f"{table}.{column}")
    assert unconstrained == [], (
        f"these name a fixed set but accept anything: {unconstrained}. "
        f"Add a CHECK, or add the column to _FREE_TEXT with the reason."
    )


def columns_actually_written(db):
    """Map each table to the columns some INSERT or UPDATE names on it.

    Parsed from the statements, not matched against the text. The check here
    used to be `re.search(rf"\\b{column}\\b", source)` over every db/*.py file
    concatenated -- comments and docstrings included -- so a column counted as
    produced when its name appeared anywhere at all: in prose, as a local
    variable, as an attribute of an unrelated object, or as a column of a
    different table. `file.width` and `file.height` passed it for years'
    worth of `typed.width` and `raw.width` in db/ingest.py while nothing has
    ever written either one.
    """
    source = "".join(
        path.read_text(encoding="utf-8")
        for path in pathlib.Path(__file__).resolve().parent.parent.joinpath("db").rglob("*.py")
    )
    # Triggers are producers too: param_key is filled entirely by one, and a
    # sweep that only reads Python would call the whole registry dead.
    source += "".join(row[0] or "" for row in db.execute("SELECT sql FROM sqlite_master WHERE type='trigger'"))
    # Quotes out, whitespace flattened: SQL in this repo is written as adjacent
    # Python string literals, so a statement only reads as one after the
    # delimiters between its halves are gone.
    flat = " ".join(re.sub(r"""["']""", " ", source).split())

    written: dict[str, set[str]] = {}
    everything: set[str] = set()
    for match in re.finditer(
        r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\(([^()]*)\))?",
        flat,
        re.IGNORECASE,
    ):
        table = match.group(1)
        if match.group(3) is None:
            # No column list: every column is being written.
            everything.add(table)
            continue
        written.setdefault(table, set()).update(name.strip() for name in match.group(3).split(","))
    for match in re.finditer(
        r"UPDATE\s+([A-Za-z_][A-Za-z0-9_]*)\s+SET\s+(.*?)(?:\s+WHERE\s|\s+RETURNING\s|;)",
        flat,
        re.IGNORECASE,
    ):
        written.setdefault(match.group(1), set()).update(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=", match.group(2)))
    # `DO UPDATE SET` needs no pass of its own: an upsert can only set columns
    # its INSERT already named, and those are collected above.
    return written, everything


def test_a_column_nothing_writes_says_so(db):
    """A column no producer fills is unfinished, not neutral.

    It reads as a feature -- a facet built on it returns an empty library,
    and nothing distinguishes "no video has a duration" from "nothing has
    ever measured one". Whichever it is, the DDL has to say.
    """
    written, everything = columns_actually_written(db)
    silent = []
    for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        if table in virtual_table_names(db) or table in everything:
            continue
        declaration = db.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()[0]
        for row in db.execute(f"PRAGMA table_info({table})"):
            column, is_pk = row[1], row[5]
            if is_pk or column in written.get(table, ()):
                continue
            # The admission sits in the comment block above the column, so
            # look at the whole declaration rather than forward from the name.
            if "NOTHING WRITES THIS YET" in declaration.upper() and re.search(
                rf"NOTHING WRITES THIS YET.{{0,400}}\b{re.escape(column)}\b",
                declaration,
                re.IGNORECASE | re.DOTALL,
            ):
                continue
            silent.append(f"{table}.{column}")
    assert silent == [], f"no producer writes these, and the DDL does not admit it: {silent}"


def test_the_producer_sweep_reads_statements_not_prose(db):
    """The control. Without it the sweep passes on a column that is only ever
    mentioned, which is how `file.width` and `file.height` read as produced
    while nothing had ever written either."""
    written, everything = columns_actually_written(db)

    assert "file" not in everything, "every INSERT into file names its columns"
    assert {"folder_id", "name", "content_sha256"} <= written["file"], (
        "the sweep cannot see the columns apply_scan plainly writes"
    )
    # A word this repo says constantly, and never as a column of `file`.
    assert "parsed_by" not in written["file"]
    # Filled entirely by a trigger, which is the half a Python-only sweep
    # would call dead.
    assert "occurrences" in written["param_key"]


def test_the_build_control_counts_real_tables(db):
    """The old threshold was `>= 30` over every table, and the three FTS indexes
    contribute seventeen shadow tables between them -- so a third of the schema
    could be missing and the control still passed."""
    virt = virtual_table_names(db)
    real = [
        r[0]
        for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        if r[0] not in virt
    ]
    assert len(real) == 43, f"expected 43 real tables, found {len(real)}: {sorted(real)}"
