"""What each page asks the database, written once, here.

These queries lived in their own test file. An algorithm defined inside its
own tests tests only itself: the tests pass regardless of what the
application does, and whoever writes the application writes a second
implementation with nothing binding it to this one. `db/scan.py` says exactly
that at the top and puts the matcher here for the reason; the pages were the
same shape and the same argument had not been applied to them.

So this is where a page's question lives, and the tests call these rather
than restating them. That is also what makes the plan check mean anything:
the plan being asserted is the plan the application will run.

Three rules hold across all of them.

**An address is resolved before the page is read, never inside it.** Joining
`entity` to match a slug inside the page query gives the planner a filter it
must reach through the file table, so it drives from the wrong end and sorts
the result -- measured, `SCAN f USING INDEX file_added` plus a temp B-tree on
a query that wants one folder. `db/naming.py` resolve does the lookup; the
queries here take ids.

**Ordering follows an index or it follows the join key.** A page that sorts
its whole result set costs the same as one that scans, and on the checkpoint
most of a library was made with, "its files sorted by name" is most of the
library sorted by name.

**An index page may aggregate; a listing may not.** A shelf that counts a
whole library is a summary, and its GROUP BY may build a TEMP B-TREE -- those
queries pass `aggregate=True` to the plan gate. A bare scan of a growing
table is never allowed, exemption or not.
"""

from __future__ import annotations

#: How many rows a grid asks for at once.
PAGE = 60


# --- the grid --------------------------------------------------------------

NEWEST_FIRST = (
    "SELECT e.slug, f.name, f.mtime FROM file f JOIN entity e ON e.id = f.id"
    " WHERE f.missing_since IS NULL ORDER BY f.mtime DESC LIMIT ?"
)


def newest(conn, limit: int = PAGE):
    """The front page. Walks `file_recent` in order rather than sorting."""
    return conn.execute(NEWEST_FIRST, (limit,)).fetchall()


# --- one picture -----------------------------------------------------------

ONE_PICTURE = """
    SELECT f.name, fo.name AS folder, f.width, f.height, f.duration,
      (SELECT g.width FROM generation g WHERE g.file_id = f.id) AS asked_for_width,
      (SELECT a.name FROM file_artifact fa JOIN artifact a ON a.id = fa.artifact_id
        WHERE fa.file_id = f.id AND fa.role = 'checkpoint') AS checkpoint,
      f.missing_since,
      (SELECT p.text FROM generation g JOIN prompt p ON p.id = g.prompt_id
        WHERE g.file_id = f.id) AS prompt,
      (SELECT g.seed FROM generation g WHERE g.file_id = f.id) AS seed,
      (SELECT count(*) FROM file_param WHERE file_id = f.id) AS fields,
      f.kind
    FROM file f JOIN folder fo ON fo.id = f.folder_id
   WHERE f.id = ?
"""

PARSED_FIELDS = "SELECT source, key, value_text FROM file_param WHERE file_id = ? ORDER BY source, key"

#: One row per LoRA, never a group_concat: SQLite's one-argument
#: group_concat separator is a bare comma, and a LoRA name may hold one.
#: Ordered by ordinal alone: within (file_id, role) the ordinal is the
#: primary key's own tail, so the order is index-served -- and unique, so
#: a name tiebreak was unreachable anyway and only bought a sort.
FILE_LORAS = (
    "SELECT a.name FROM file_artifact fa JOIN artifact a ON a.id = fa.artifact_id"
    " WHERE fa.file_id = ? AND fa.role = 'lora' ORDER BY fa.ordinal"
)

NEIGHBOUR = (
    "SELECT e.slug FROM file f JOIN entity e ON e.id = f.id"
    " WHERE f.folder_id = ? AND f.missing_since IS NULL AND (f.mtime, f.id) {way} (?, ?)"
    " ORDER BY f.mtime {order}, f.id {order} LIMIT 1"
)


#: Presence for the byte-serving guard, asked here so routes carry no
#: SQL of their own.
FILE_PRESENT = "SELECT missing_since IS NULL FROM file WHERE id = ?"


def picture(conn, file_id: int):
    return conn.execute(ONE_PICTURE, (file_id,)).fetchone()


def file_present(conn, file_id: int) -> bool | None:
    """True = present, False = marked missing, None = no such row."""
    row = conn.execute(FILE_PRESENT, (file_id,)).fetchone()
    return None if row is None else bool(row[0])


def fields_of(conn, file_id: int):
    return conn.execute(PARSED_FIELDS, (file_id,)).fetchall()


def file_loras(conn, file_id: int) -> list[str]:
    return [row[0] for row in conn.execute(FILE_LORAS, (file_id,))]


def neighbour(conn, file_id: int, *, previous: bool = True):
    """The picture before or after this one in its folder.

    Ordered on (mtime, id), not (mtime, slug): the slug lives on `entity` and
    no index spans two tables, so tie-breaking on it sorted the whole folder
    to return one row -- 50,007 rows sorted per arrow-key press on the
    largest folder in a real library.
    """
    row = conn.execute("SELECT folder_id, mtime FROM file WHERE id = ?", (file_id,)).fetchone()
    if row is None:
        return None
    sql = NEIGHBOUR.format(**({"way": "<", "order": "DESC"} if previous else {"way": ">", "order": "ASC"}))
    found = conn.execute(sql, (row[0], row[1], file_id)).fetchone()
    return found[0] if found else None


# --- folders ---------------------------------------------------------------

FOLDER_FILES = (
    "SELECT e.slug, f.name FROM file f JOIN entity e ON e.id = f.id"
    " WHERE f.folder_id = ? AND f.missing_since IS NULL"
    " ORDER BY f.name COLLATE NOCASE LIMIT ?"
)

BREADCRUMB = """
    WITH RECURSIVE up(id, name, parent_id, depth) AS (
      SELECT id, name, parent_id, depth FROM folder WHERE id = ?
      UNION ALL
      SELECT f.id, f.name, f.parent_id, f.depth
        FROM folder f JOIN up ON f.id = up.parent_id
    )
    SELECT id, name FROM up ORDER BY depth
"""


def folder_files(conn, folder_id: int, limit: int = 120):
    """COLLATE NOCASE to match `file_in_folder`, which is what makes this an
    ordered walk rather than a sort -- and the order a person expects, since
    the platform's own filesystem is case-insensitive."""
    return conn.execute(FOLDER_FILES, (folder_id, limit)).fetchall()


def breadcrumb(conn, folder_id: int):
    """Walked by parent, never by splitting a path. A path is presentation."""
    return conn.execute(BREADCRUMB, (folder_id,)).fetchall()


#: The root path rides along for the reachability probe ONLY -- it is
#: server-side state and never part of any rendered answer; a folder's
#: durable identity is its slug and its parent chain.
FOLDER_CARD = (
    "SELECT f.name, f.parent_id, f.missing_since, r.path  FROM folder f JOIN root r ON r.id = f.root_id WHERE f.id = ?"
)

FOLDER_CHILDREN = (
    "SELECT e.slug, f.name,"
    " (SELECT count(*) FROM file WHERE folder_id = f.id AND missing_since IS NULL) AS pictures"
    "  FROM folder f JOIN entity e ON e.id = f.id"
    " WHERE f.parent_id = ? AND f.missing_since IS NULL"
    " ORDER BY f.name COLLATE NOCASE"
)


def folder_card(conn, folder_id: int):
    return conn.execute(FOLDER_CARD, (folder_id,)).fetchone()


def folder_children(conn, folder_id: int):
    """The immediate child directories, each with its own DIRECT media
    count -- `folder=` means the folder itself, never its subtree, so a
    child is something to navigate into, not part of the answer."""
    return conn.execute(FOLDER_CHILDREN, (folder_id,)).fetchall()


#: The NAVIGABLE roots only: trash is a real storage location whose
#: subtree views exclude, never a shelf to browse. Reachability comes
#: from db/library.py probe_roots, which verifies the marker rather
#: than trusting that A directory exists at the recorded path.
ROOT_SHELF = "SELECT id, kind FROM root WHERE kind IN ('library', 'mount') ORDER BY id"

FOLDER_TOPS = (
    "SELECT e.slug, f.name,"
    " (SELECT count(*) FROM file WHERE folder_id = f.id AND missing_since IS NULL) AS pictures"
    "  FROM folder f JOIN entity e ON e.id = f.id"
    " WHERE f.root_id = ? AND f.parent_id IS NULL AND f.missing_since IS NULL"
    " ORDER BY f.name COLLATE NOCASE"
)


def roots_shelf(conn):
    return conn.execute(ROOT_SHELF).fetchall()


def folder_tops(conn, root_id: int):
    """One root's depth-0 folder entities -- where physical navigation
    enters. Slugs and names only: the root's path is operational state
    and never part of the browsing surface."""
    return conn.execute(FOLDER_TOPS, (root_id,)).fetchall()


# --- the recipe axis -------------------------------------------------------

#: count(DISTINCT f.id): "pictures" counts pictures. file_artifact
#: legally holds one artifact at several ordinals in one file -- a LoRA
#: stacked twice -- and a row count would name relation instances after
#: media. The shelf number must equal the ResultSet total.
ARTIFACTS_BY_USE = (
    "SELECT a.name, e.slug, count(DISTINCT f.id) AS pictures FROM artifact a"
    "  JOIN entity e ON e.id = a.id"
    "  JOIN file_artifact fa ON fa.artifact_id = a.id"
    "  JOIN file f ON f.id = fa.file_id AND f.missing_since IS NULL"
    " WHERE a.kind = ? GROUP BY a.id ORDER BY pictures DESC, a.name"
)

#: What this LoRA is actually used with, and how often. The old app answered
#: this by matching one filename against a delimited blob holding another and
#: calling co-residency a match; it is a join with real counts.
LORA_SYNERGY = (
    "SELECT ckpt.name, e.slug, count(DISTINCT fl.file_id) AS together FROM file_artifact fl"
    "  JOIN file_artifact fc ON fc.file_id = fl.file_id AND fc.role = 'checkpoint'"
    "  JOIN artifact ckpt ON ckpt.id = fc.artifact_id"
    "  JOIN entity e ON e.id = ckpt.id"
    " WHERE fl.artifact_id = ? AND fl.role = 'lora'"
    " GROUP BY ckpt.id ORDER BY together DESC, ckpt.name"
)


#: Workflows attach through `generation.workflow_id`, never `file_artifact`
#: -- a workflow is how the picture was made, not a weight it loaded -- so
#: the workflow shelf has its own joins.
WORKFLOWS_BY_USE = (
    "SELECT a.name, e.slug, count(*) AS pictures FROM artifact a"
    "  JOIN entity e ON e.id = a.id"
    "  JOIN generation g ON g.workflow_id = a.id"
    "  JOIN file f ON f.id = g.file_id AND f.missing_since IS NULL"
    " WHERE a.kind = 'workflow'"
    " GROUP BY a.id ORDER BY pictures DESC, a.name"
)


def artifacts_by_use(conn, kind: str):
    """The models or LoRAs index -- counted by pictures, not by mentions,
    which is why it joins `file` rather than counting rows."""
    return conn.execute(ARTIFACTS_BY_USE, (kind,)).fetchall()


def workflows_by_use(conn):
    return conn.execute(WORKFLOWS_BY_USE).fetchall()


#: The artifact's own facts -- never its media membership, which is the
#: ResultSet's answer to artifact={slug}.
ARTIFACT_CARD = "SELECT name, kind, architecture, content_sha256, quoted_hash, first_seen_at FROM artifact WHERE id = ?"


def artifact_card(conn, artifact_id: int):
    return conn.execute(ARTIFACT_CARD, (artifact_id,)).fetchone()


def lora_synergy(conn, lora_id: int):
    return conn.execute(LORA_SYNERGY, (lora_id,)).fetchall()


# --- people ----------------------------------------------------------------

#: Counted per (file, person), so two faces of one person in one photograph
#: are one picture. The old schema counted detections and needed a warning.
PEOPLE_BY_MOST = (
    "SELECT COALESCE(p.name, '(unnamed)') AS name, e.slug,"
    " count(DISTINCT fp.file_id) AS pictures"
    "  FROM person p JOIN entity e ON e.id = p.id"
    "  JOIN derived_file_person fp ON fp.person_id = p.id AND fp.run_id = ?"
    "  JOIN file f ON f.id = fp.file_id AND f.missing_since IS NULL"
    " GROUP BY p.id ORDER BY pictures DESC, name"
)

#: No DISTINCT: the primary key is (file_id, person_id, run_id) and this
#: filters on person and run, so a file can appear once. Adding one bought
#: nothing but a temp B-tree over the result.
PERSON_FILES = (
    "SELECT fe.slug, f.name FROM derived_file_person fp"
    "  JOIN file f ON f.id = fp.file_id AND f.missing_since IS NULL"
    "  JOIN entity fe ON fe.id = f.id WHERE fp.person_id = ? AND fp.run_id = ?"
)

#: Where a person's pictures actually sit. The disagreement between the disk
#: layout and the meaning is the thing the six-axis design exists to show.
PERSON_ACROSS_FOLDERS = (
    "SELECT fo.name, e.slug, count(DISTINCT f.id) AS pictures"
    "  FROM derived_file_person fp"
    "  JOIN file f ON f.id = fp.file_id AND f.missing_since IS NULL"
    "  JOIN folder fo ON fo.id = f.folder_id"
    "  JOIN entity e ON e.id = fo.id"
    " WHERE fp.person_id = ? AND fp.run_id = ?"
    " GROUP BY fo.id ORDER BY pictures DESC, fo.name"
)


PERSON_NAME = "SELECT name FROM person WHERE id = ?"

#: How many durable naming assertions anchor this person -- the count the
#: naming route refuses on, because a name with no face to assert it
#: against is lost by the next re-cluster.
PERSON_ASSERTIONS = "SELECT count(*) FROM person_assertion WHERE person_id = ?"


def person_name(conn, person_id: int) -> str | None:
    """The person's given name, or None while they are still unnamed."""
    row = conn.execute(PERSON_NAME, (person_id,)).fetchone()
    return row[0] if row else None


def person_assertions(conn, person_id: int) -> int:
    return conn.execute(PERSON_ASSERTIONS, (person_id,)).fetchone()[0]


def people_by_most(conn, run_id: int | None = None):
    """The People page, for one clustering run.

    `run_id` defaults to the one marked primary. Several runs are live at
    once and they disagree -- that is what they are for -- so a page that
    does not say which one it is showing is showing whichever wrote last.
    """
    if run_id is None:
        row = conn.execute("SELECT id FROM derived_face_run WHERE is_primary = 1").fetchone()
        if row is None:
            return []
        run_id = row[0]
    return conn.execute(PEOPLE_BY_MOST, (run_id,)).fetchall()


# --- comparing two clusterings ---------------------------------------------

RUNS = (
    "SELECT r.id, r.model_id, r.model_version, r.method, r.threshold,"
    " r.is_primary, r.faces, r.clusters, r.backend,"
    " (SELECT count(*) FROM derived_face_cluster c"
    "   WHERE c.run_id = r.id AND c.person_id IS NOT NULL) AS named"
    "  FROM derived_face_run r ORDER BY r.is_primary DESC, r.computed_at DESC"
)

#: Where two runs disagree about one picture: the people each says are in it.
#: The disagreement is the point -- one threshold welds strangers together and
#: another splits one person in four, and the only way to see which is
#: happening is to put the two answers side by side on the same photograph.
RUNS_DISAGREE = """
    SELECT f.name,
           group_concat(DISTINCT CASE WHEN fp.run_id = ? THEN p.name END) AS left_says,
           group_concat(DISTINCT CASE WHEN fp.run_id = ? THEN p.name END) AS right_says
      FROM derived_file_person fp
      JOIN person p ON p.id = fp.person_id
      JOIN file f ON f.id = fp.file_id
     WHERE fp.run_id IN (?, ?)
     GROUP BY f.id
    HAVING IFNULL(left_says, '') <> IFNULL(right_says, '')
     ORDER BY f.name
"""

#: How the same face was grouped by each run. A face one run puts with
#: fifty others and another puts alone is the single most useful row here.
FACE_ACROSS_RUNS = """
    SELECT fi.id,
           (SELECT count(*) FROM derived_face_membership m2
             JOIN derived_face_cluster c2 ON c2.id = m2.cluster_id
             WHERE c2.run_id = ? AND m2.cluster_id IN (
               SELECT cluster_id FROM derived_face_membership m3
                JOIN derived_face_cluster c3 ON c3.id = m3.cluster_id
               WHERE m3.face_id = fi.id AND c3.run_id = ?)) AS in_left,
           (SELECT count(*) FROM derived_face_membership m2
             JOIN derived_face_cluster c2 ON c2.id = m2.cluster_id
             WHERE c2.run_id = ? AND m2.cluster_id IN (
               SELECT cluster_id FROM derived_face_membership m3
                JOIN derived_face_cluster c3 ON c3.id = m3.cluster_id
               WHERE m3.face_id = fi.id AND c3.run_id = ?)) AS in_right
      FROM derived_face_instance fi
     WHERE fi.embedding IS NOT NULL
     ORDER BY abs(in_left - in_right) DESC, fi.id
     LIMIT ?
"""


def clusterings(conn):
    """Every clustering the library holds, side by side."""
    cursor = conn.execute(RUNS)
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor]


def disagreements(conn, left: int, right: int):
    """The pictures two runs describe differently."""
    return conn.execute(RUNS_DISAGREE, (left, right, left, right)).fetchall()


def face_across_runs(conn, left: int, right: int, limit: int = 20):
    """Per face, how big a group each run put it in. The faces where those
    two numbers differ most are where the two clusterings actually differ."""
    return conn.execute(FACE_ACROSS_RUNS, (left, left, right, right, limit)).fetchall()


def person_files(conn, person_id: int, run_id: int | None = None):
    if run_id is None:
        row = conn.execute("SELECT id FROM derived_face_run WHERE is_primary = 1").fetchone()
        if row is None:
            return []
        run_id = row[0]
    return conn.execute(PERSON_FILES, (person_id, run_id)).fetchall()


def person_across_folders(conn, person_id: int, run_id: int | None = None):
    if run_id is None:
        row = conn.execute("SELECT id FROM derived_face_run WHERE is_primary = 1").fetchone()
        if row is None:
            return []
        run_id = row[0]
    return conn.execute(PERSON_ACROSS_FOLDERS, (person_id, run_id)).fetchall()


# --- copies of copies ------------------------------------------------------

#: Every group of perceptual copies, its best face forward and its count.
#: An index page over a summary -- the aggregate exemption, like the
#: shelves. Present members only, on both sides: a group whose best went
#: missing is not shown, and a missing member is not counted -- the same
#: convention DUPE_COPIES holds, so the shelf and the page agree.
DUPE_GROUPS = (
    "SELECT e.slug, f.name, count(*) AS copies FROM derived_dupe_group best"
    "  JOIN derived_dupe_group member ON member.group_id = best.group_id"
    "  JOIN file mf ON mf.id = member.file_id AND mf.missing_since IS NULL"
    "  JOIN file f ON f.id = best.file_id AND f.missing_since IS NULL"
    "  JOIN entity e ON e.id = best.file_id"
    " WHERE best.is_best = 1"
    " GROUP BY best.group_id ORDER BY copies DESC, best.group_id LIMIT ?"
)

#: One picture's other bodies: everything sharing its group, itself
#: excluded. Ordered by the group index's own tail -- sorting by
#: `distance` bought a TEMP B-TREE for presentation the caller can do
#: over one page of rows; the distance rides along as data.
DUPE_COPIES = (
    "SELECT e.slug, f.name, twin.distance, twin.is_best FROM derived_dupe_group mine"
    "  JOIN derived_dupe_group twin ON twin.group_id = mine.group_id AND twin.file_id <> mine.file_id"
    "  JOIN file f ON f.id = twin.file_id AND f.missing_since IS NULL"
    "  JOIN entity e ON e.id = twin.file_id"
    " WHERE mine.file_id = ? ORDER BY twin.file_id LIMIT ?"
)


def dupe_groups(conn, limit: int = 120):
    return conn.execute(DUPE_GROUPS, (limit,)).fetchall()


def dupe_copies(conn, file_id: int, limit: int = 120):
    return conn.execute(DUPE_COPIES, (file_id, limit)).fetchall()


# --- albums ----------------------------------------------------------------

#: Every ACTIVE collection with how many present pictures it holds.
#: LEFT JOINs so an album somebody just made lists at zero instead of
#: vanishing; archived collections keep their rows and their addresses
#: but leave the shelves.
ALBUMS = (
    "SELECT c.name, e.slug, c.kind, count(f.id) AS pictures"
    "  FROM collection c JOIN entity e ON e.id = c.id"
    "  LEFT JOIN collection_file cf ON cf.collection_id = c.id"
    "  LEFT JOIN file f ON f.id = cf.file_id AND f.missing_since IS NULL"
    " WHERE c.archived_at IS NULL"
    " GROUP BY c.id ORDER BY c.name COLLATE NOCASE"
)

#: The management shelf: what was retired, flat -- hierarchy is an
#: active-tree presentation and an archived row keeps only its own state.
ARCHIVED_ALBUMS = (
    "SELECT c.name, e.slug, c.kind, count(f.id) AS pictures"
    "  FROM collection c JOIN entity e ON e.id = c.id"
    "  LEFT JOIN collection_file cf ON cf.collection_id = c.id"
    "  LEFT JOIN file f ON f.id = cf.file_id AND f.missing_since IS NULL"
    " WHERE c.archived_at IS NOT NULL"
    " GROUP BY c.id ORDER BY c.name COLLATE NOCASE"
)

ARCHIVED_COUNT = "SELECT count(*) FROM collection WHERE archived_at IS NOT NULL"


def albums(conn):
    return conn.execute(ALBUMS).fetchall()


def archived_albums(conn):
    return conn.execute(ARCHIVED_ALBUMS).fetchall()


def archived_count(conn) -> int:
    return conn.execute(ARCHIVED_COUNT).fetchone()[0]


ALBUM_FILES = (
    "SELECT fe.slug, f.name FROM collection_file cf"
    "  JOIN file f ON f.id = cf.file_id AND f.missing_since IS NULL"
    "  JOIN entity fe ON fe.id = f.id WHERE cf.collection_id = ?"
    " ORDER BY cf.file_id LIMIT ?"
)

#: How many of an album's members are actually present -- the number every
#: route answers with, so the shelf and the membership writes cannot drift.
ALBUM_PRESENT = (
    "SELECT count(*) FROM collection_file cf"
    "  JOIN file f ON f.id = cf.file_id AND f.missing_since IS NULL"
    " WHERE cf.collection_id = ?"
)


def album_files(conn, collection_id: int, limit: int = 120):
    return conn.execute(ALBUM_FILES, (collection_id, limit)).fetchall()


def album_present(conn, collection_id: int) -> int:
    return conn.execute(ALBUM_PRESENT, (collection_id,)).fetchone()[0]


#: The whole definition plus lifecycle, updated_by resolved to the
#: username -- an id is not a presentation fact.
COLLECTION_CARD = (
    "SELECT c.name, c.kind, c.color, c.description, c.parent_id,"
    " c.archived_at, c.definition_rev, c.updated_at, u.username"
    "  FROM collection c LEFT JOIN user u ON u.id = c.updated_by WHERE c.id = ?"
)

#: `IS ?` rather than `= ?` so one statement serves both levels -- NULL
#: names the top of the hierarchy -- and both ride collection_parent
#: (probed: SEARCH under either argument, ordering included).
COLLECTION_CHILDREN = (
    "SELECT c.id, e.slug, c.name, c.kind,"
    " (SELECT count(*) FROM collection_file cf JOIN file f ON f.id = cf.file_id"
    "   AND f.missing_since IS NULL WHERE cf.collection_id = c.id) AS pictures"
    "  FROM collection c JOIN entity e ON e.id = c.id"
    " WHERE c.parent_id IS ? AND c.archived_at IS NULL ORDER BY c.name COLLATE NOCASE"
)


def collection_card(conn, collection_id: int):
    return conn.execute(COLLECTION_CARD, (collection_id,)).fetchone()


def collection_children(conn, collection_id: int | None):
    """One level of the authored hierarchy: `None` asks for the top --
    the entity page's own children, one index search."""
    return conn.execute(COLLECTION_CHILDREN, (collection_id,)).fetchall()


#: The whole shelf in ONE statement -- /albums promises every collection,
#: so reading O(N) rows is the page's own meaning; what stays forbidden
#: is a read-time sort, and ORDER BY (parent_id, name) IS the
#: collection_parent index's order, so the plan is one ordered walk.
#: One statement is also what makes the rendered tree one snapshot: a
#: reparent committed mid-render cannot show a collection twice or not
#: at all, where a query per node could.
COLLECTION_SHELF = (
    "SELECT c.id, c.parent_id, e.slug, c.name, c.kind,"
    " (SELECT count(*) FROM collection_file cf JOIN file f ON f.id = cf.file_id"
    "   AND f.missing_since IS NULL WHERE cf.collection_id = c.id) AS pictures"
    "  FROM collection c JOIN entity e ON e.id = c.id"
    " WHERE c.archived_at IS NULL"
    " ORDER BY c.parent_id, c.name COLLATE NOCASE"
)


def collection_shelf(conn):
    return conn.execute(COLLECTION_SHELF).fetchall()


#: The album picker's choices: every LISTED collection -- smart ones are
#: rule-derived and have no membership to offer -- with whether this
#: file is filed in each. Ordered as the shelf is, riding
#: collection_parent; a whole-index page, because a chooser that hides
#: albums is a chooser that loses photographs.
COLLECTION_CHOICES = (
    "SELECT e.slug, c.name, c.kind,"
    " EXISTS(SELECT 1 FROM collection_file cf WHERE cf.collection_id = c.id AND cf.file_id = ?) AS filed"
    "  FROM collection c JOIN entity e ON e.id = c.id"
    " WHERE c.kind IN ('album', 'flag') AND c.archived_at IS NULL"
    " ORDER BY c.parent_id, c.name COLLATE NOCASE"
)


def collection_choices(conn, file_id: int):
    return conn.execute(COLLECTION_CHOICES, (file_id,)).fetchall()


def collections_named(conn, ids) -> list[tuple[str, str, int]]:
    """`(slug, name, archived)` for an id set the caller already
    resolved -- the parent picker's labels, name-ordered like every
    shelf. Archived rides along so the one archived offer (the current
    parent) can say what it is."""
    if not ids:
        return []
    marks = ",".join("?" for _ in ids)
    return conn.execute(
        f"SELECT e.slug, c.name, c.archived_at IS NOT NULL"
        f"  FROM collection c JOIN entity e ON e.id = c.id"
        f" WHERE c.id IN ({marks}) ORDER BY c.name COLLATE NOCASE",
        list(ids),
    ).fetchall()


# --- what can be searched --------------------------------------------------

WAYS = "SELECT source, key, value_kind, occurrences FROM param_key ORDER BY occurrences DESC, key"


def ways(conn):
    """`/ways` is not a hand-written list: `param_key` learns every field on
    ingest, so a tag nobody predicted is offered the day it first appears."""
    return conn.execute(WAYS).fetchall()


# --- lineage ---------------------------------------------------------------

PARENTS = (
    "SELECT e.slug, f.name, d.kind FROM file_derivation d"
    "  JOIN file f ON f.id = d.parent_id JOIN entity e ON e.id = f.id"
    " WHERE d.child_id = ?"
)

CHILDREN = (
    "SELECT e.slug, f.name, d.kind FROM file_derivation d"
    "  JOIN file f ON f.id = d.child_id JOIN entity e ON e.id = f.id"
    " WHERE d.parent_id = ?"
)


def parents(conn, file_id: int):
    return conn.execute(PARENTS, (file_id,)).fetchall()


def children(conn, file_id: int):
    return conn.execute(CHILDREN, (file_id,)).fetchall()
