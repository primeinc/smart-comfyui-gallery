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

from . import context
from .context import HUMAN_MOMENT

#: How many rows a grid asks for at once.
PAGE = 60


# --- the grid --------------------------------------------------------------

NEWEST_FIRST = (
    "SELECT e.slug, f.name, f.mtime FROM file f JOIN entity e ON e.id = f.id"
    " WHERE f.missing_since IS NULL ORDER BY f.mtime DESC LIMIT ?"
)

#: The machine front door: what the library HOLDS, in one statement --
#: a summary, not a media answer (the gallery is the ResultSet's).
LIBRARY_SUMMARY = (
    "SELECT"
    " (SELECT count(*) FROM file WHERE missing_since IS NULL) AS files,"
    " (SELECT count(*) FROM folder WHERE missing_since IS NULL) AS folders,"
    " (SELECT count(*) FROM person) AS people,"
    " (SELECT count(*) FROM collection WHERE archived_at IS NULL) AS collections,"
    " (SELECT count(*) FROM artifact) AS artifacts"
)


def newest(conn, limit: int = PAGE):
    """The newest strip. Walks `file_recent` in order rather than sorting."""
    return conn.execute(NEWEST_FIRST, (limit,)).fetchall()


def library_summary(conn):
    return conn.execute(LIBRARY_SUMMARY).fetchone()


# --- one picture -----------------------------------------------------------

ONE_PICTURE = """
    SELECT f.name, fo.name AS folder, f.width, f.height, f.duration,
      (SELECT g.width FROM generation g WHERE g.file_id = f.id) AS asked_for_width,
      (SELECT a.name FROM file_artifact fa JOIN artifact a ON a.id = fa.artifact_id
        WHERE fa.file_id = f.id AND fa.role = 'checkpoint') AS checkpoint,
      f.missing_since,
      (SELECT p.text FROM generation_prompt gp JOIN prompt p ON p.id = gp.prompt_id
        WHERE gp.file_id = f.id AND gp.role = 'effective') AS prompt,
      (SELECT g.seed FROM generation g WHERE g.file_id = f.id) AS seed,
      (SELECT count(*) FROM file_param WHERE file_id = f.id) AS fields,
      f.kind,
      CASE WHEN f.ingested_sha256 IS NULL THEN 'never'
           WHEN f.ingested_sha256 = f.content_sha256 THEN 'current'
           ELSE 'stale' END AS read_state
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

#: Two counts per folder, both shown: `pictures` is the folder's OWN
#: media (what `folder=` answers), `below` the whole subtree's -- a
#: library whose media all live two folders down is not "0 pictures".
FOLDER_CHILDREN = (
    "SELECT e.slug, f.name,"
    " (SELECT count(*) FROM file WHERE folder_id = f.id AND missing_since IS NULL) AS pictures,"
    " (WITH RECURSIVE sub(id) AS (SELECT f.id UNION ALL SELECT c.id FROM folder c JOIN sub ON c.parent_id = sub.id"
    "   WHERE c.missing_since IS NULL)"
    "  SELECT count(*) FROM file WHERE folder_id IN (SELECT id FROM sub) AND missing_since IS NULL) AS below"
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
    " (SELECT count(*) FROM file WHERE folder_id = f.id AND missing_since IS NULL) AS pictures,"
    " (WITH RECURSIVE sub(id) AS (SELECT f.id UNION ALL SELECT c.id FROM folder c JOIN sub ON c.parent_id = sub.id"
    "   WHERE c.missing_since IS NULL)"
    "  SELECT count(*) FROM file WHERE folder_id IN (SELECT id FROM sub) AND missing_since IS NULL) AS below"
    "  FROM folder f JOIN entity e ON e.id = f.id"
    " WHERE f.root_id = ? AND f.parent_id IS NULL AND f.missing_since IS NULL"
    " ORDER BY f.name COLLATE NOCASE"
)


#: When each top folder's pictures are from, descendants included: the
#: earliest and latest human moment over every present file below it
#: that the context job has interpreted, keyed by the folder's slug. A
#: folder nothing has interpreted yet has no row.
FOLDER_TOP_SPANS = (
    "WITH RECURSIVE sub(top, id) AS ("
    "  SELECT f.id, f.id FROM folder f WHERE f.root_id = ? AND f.parent_id IS NULL AND f.missing_since IS NULL"
    "  UNION ALL SELECT sub.top, c.id FROM folder c JOIN sub ON c.parent_id = sub.id WHERE c.missing_since IS NULL)"
    " SELECT e.slug, min(" + HUMAN_MOMENT + "), max(" + HUMAN_MOMENT + ")"
    "  FROM sub JOIN entity e ON e.id = sub.top"
    "  JOIN file fi ON fi.folder_id = sub.id AND fi.missing_since IS NULL"
    "  JOIN derived_media_context mc ON mc.file_id = fi.id AND mc.policy_version = ?"
    " GROUP BY sub.top"
)

#: One folder's subtree: when its pictures are from, and where they
#: happened (the places its interpreted members were said to be in,
#: most first, bounded).
_SUBTREE = (
    "WITH RECURSIVE sub(id) AS (SELECT ? UNION ALL SELECT c.id FROM folder c JOIN sub ON c.parent_id = sub.id"
    "  WHERE c.missing_since IS NULL)"
)
FOLDER_SPAN = (
    _SUBTREE + " SELECT min(" + HUMAN_MOMENT + "), max(" + HUMAN_MOMENT + ")"
    "  FROM sub JOIN file fi ON fi.folder_id = sub.id AND fi.missing_since IS NULL"
    "  JOIN derived_media_context mc ON mc.file_id = fi.id AND mc.policy_version = ?"
)
FOLDER_PLACES = (
    _SUBTREE + " SELECT p.id, e.slug, p.name, p.kind, count(*) AS pictures"
    "  FROM sub JOIN file fi ON fi.folder_id = sub.id AND fi.missing_since IS NULL"
    "  JOIN derived_media_context mc ON mc.file_id = fi.id AND mc.policy_version = ?"
    "  JOIN place p ON p.id = mc.place_id JOIN entity e ON e.id = p.id"
    " GROUP BY p.id ORDER BY pictures DESC, p.name COLLATE NOCASE LIMIT ?"
)
#: The same for one collection over its present members.
COLLECTION_PLACES = (
    "SELECT p.id, e.slug, p.name, p.kind, count(*) AS pictures"
    "  FROM collection_file cf JOIN file f ON f.id = cf.file_id AND f.missing_since IS NULL"
    "  JOIN derived_media_context mc ON mc.file_id = f.id AND mc.policy_version = ?"
    "  JOIN place p ON p.id = mc.place_id JOIN entity e ON e.id = p.id"
    " WHERE cf.collection_id = ? GROUP BY p.id ORDER BY pictures DESC, p.name COLLATE NOCASE LIMIT ?"
)
PLACES_ON_A_PAGE = 8


def folder_span(conn, folder_id: int) -> tuple:
    row = conn.execute(FOLDER_SPAN, (folder_id, context.POLICY_VERSION)).fetchone()
    return (row[0], row[1]) if row else (None, None)


def folder_places(conn, folder_id: int, limit: int = PLACES_ON_A_PAGE):
    return conn.execute(FOLDER_PLACES, (folder_id, context.POLICY_VERSION, limit)).fetchall()


def collection_places(conn, collection_id: int, limit: int = PLACES_ON_A_PAGE):
    return conn.execute(COLLECTION_PLACES, (context.POLICY_VERSION, collection_id, limit)).fetchall()


#: The same for every active collection over its present members.
COLLECTION_SPANS = (
    "SELECT e.slug, min(" + HUMAN_MOMENT + "), max(" + HUMAN_MOMENT + ")"
    "  FROM collection c JOIN entity e ON e.id = c.id"
    "  JOIN collection_file cf ON cf.collection_id = c.id"
    "  JOIN file f ON f.id = cf.file_id AND f.missing_since IS NULL"
    "  JOIN derived_media_context mc ON mc.file_id = f.id AND mc.policy_version = ?"
    " WHERE c.archived_at IS NULL GROUP BY c.id"
)


def roots_shelf(conn):
    return conn.execute(ROOT_SHELF).fetchall()


def folder_top_spans(conn, root_id: int) -> dict[str, tuple[float, float]]:
    return {
        slug: (first, last) for slug, first, last in conn.execute(FOLDER_TOP_SPANS, (root_id, context.POLICY_VERSION))
    }


def collection_spans(conn) -> dict[str, tuple[float, float]]:
    return {slug: (first, last) for slug, first, last in conn.execute(COLLECTION_SPANS, (context.POLICY_VERSION,))}


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


#: When each person was first and last seen, on the human timeline,
#: for one clustering run: the People index's answer beside the count.
#: An aggregate over the run's attributions -- the index page's own
#: summary, like the shelves.
PEOPLE_SPANS = (
    "SELECT fp.person_id, min(" + HUMAN_MOMENT + "), max(" + HUMAN_MOMENT + ")"
    "  FROM derived_file_person fp"
    "  JOIN derived_media_context mc ON mc.file_id = fp.file_id AND mc.policy_version = ?"
    "  JOIN file f ON f.id = fp.file_id AND f.missing_since IS NULL"
    " WHERE fp.run_id = ? GROUP BY fp.person_id"
)

#: When a person was seen: every CURRENT session holding one of their
#: pictures, with how many of its members are theirs and the story told
#: of it -- the timeline's answer for one face, newest first.
PERSON_SESSIONS = (
    "SELECT ev.id, ev.kind, COALESCE(ev.local_start, ev.instant_start), COALESCE(ev.local_end, ev.instant_end),"
    " count(DISTINCT ef.file_id) AS theirs,"
    " (SELECT count(*) FROM derived_event_file x WHERE x.event_id = ev.id) AS pictures,"
    " (SELECT sr.id FROM story_snapshot s JOIN story_plan sp ON sp.snapshot_id = s.id"
    "   JOIN story_render sr ON sr.plan_id = sp.id WHERE s.member_hash = ev.member_hash"
    "   AND s.event_kind = ev.kind AND s.grouper = r.grouper ORDER BY sr.id DESC LIMIT 1)"
    "  FROM derived_file_person fp"
    "  JOIN derived_event_file ef ON ef.file_id = fp.file_id"
    "  JOIN derived_event ev ON ev.id = ef.event_id"
    "  JOIN derived_event_run r ON r.id = ev.run_id"
    " WHERE fp.person_id = ? AND fp.run_id = ?"
    "   AND r.context_generation = (SELECT generation FROM derived_context_state)"
    "   AND r.context_policy_version = ?"
    " GROUP BY ev.id ORDER BY COALESCE(ev.local_start, ev.instant_start) DESC"
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


def standings(conn) -> list[dict]:
    """Every run with where it stands against the People page -- the
    default, or the sentence saying why not (db/derived.py standing)."""
    from . import derived

    return [
        {
            "id": run["id"],
            "model_id": run["model_id"],
            "faces": run["faces"],
            "clusters": run["clusters"],
            "standing": derived.standing(conn, run["id"], run["model_id"], run["threshold"]),
        }
        for run in clusterings(conn)
    ]


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


EVENT_DOMAIN = "SELECT local_start, instant_start FROM derived_event WHERE id = ?"


def event_domain(conn, event_id: int):
    """`(local_start, instant_start)`: which clock domain a session knows."""
    return conn.execute(EVENT_DOMAIN, (event_id,)).fetchone()


PEOPLE_IDS = "SELECT p.id, e.slug FROM person p JOIN entity e ON e.id = p.id"


def people_ids(conn):
    """Every person's id beside their live slug -- the join the index
    page needs to lay a span next to a card."""
    return conn.execute(PEOPLE_IDS).fetchall()


def people_spans(conn, run_id: int | None = None) -> dict[int, tuple[float, float]]:
    """`{person_id: (first, last)}` on the human timeline for one run
    (the primary by default); a person with no interpreted picture is
    absent, and the page says nothing rather than a guess."""
    if run_id is None:
        row = conn.execute("SELECT id FROM derived_face_run WHERE is_primary = 1").fetchone()
        if row is None:
            return {}
        run_id = row[0]
    return {
        int(pid): (first, last) for pid, first, last in conn.execute(PEOPLE_SPANS, (context.POLICY_VERSION, run_id))
    }


def person_sessions(conn, person_id: int, run_id: int | None = None):
    """The current sessions one person appears in, for one clustering
    run (the primary by default), newest first."""
    if run_id is None:
        row = conn.execute("SELECT id FROM derived_face_run WHERE is_primary = 1").fetchone()
        if row is None:
            return []
        run_id = row[0]
    return conn.execute(PERSON_SESSIONS, (person_id, run_id, context.POLICY_VERSION)).fetchall()


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


# --- the timeline ----------------------------------------------------------

#: The human timeline's one axis: context.HUMAN_MOMENT, the same
#: fragment the day facet filters by, so the shelf and the door into
#: the gallery cannot disagree. Only rows of the RUNNING code's
#: interpretation policy answer -- bound at call time, never the
#: version the database happens to remember, so an upgraded build shows
#: honest absence until the context job re-interprets.
TIMELINE_MONTHS = (
    "SELECT strftime('%Y-%m', " + HUMAN_MOMENT + ", 'unixepoch') AS month,"
    " count(*) AS pictures"
    "  FROM derived_media_context mc"
    "  JOIN file f ON f.id = mc.file_id AND f.missing_since IS NULL"
    " WHERE mc.policy_version = ?"
    " GROUP BY month ORDER BY month DESC"
)

#: Days are PRESENTATION grouping, read straight off the contexts --
#: deliberately never a persisted event kind.
TIMELINE_DAYS = (
    "SELECT strftime('%Y-%m-%d', " + HUMAN_MOMENT + ", 'unixepoch') AS day,"
    " count(*) AS pictures"
    "  FROM derived_media_context mc"
    "  JOIN file f ON f.id = mc.file_id AND f.missing_since IS NULL"
    " WHERE mc.policy_version = ?"
    " GROUP BY day ORDER BY day DESC LIMIT ?"
)

#: Event runs answer only while they can PROVE they were computed over
#: the current interpretation: generation and policy both match, or the
#: hypothesis is stale -- whoever its members are.
TIMELINE_EVENTS = (
    "SELECT e.id, r.grouper, e.kind, e.local_start, e.local_end,"
    " e.instant_start, e.instant_end, e.confidence, e.member_hash,"
    " (SELECT count(*) FROM derived_event_file ef WHERE ef.event_id = e.id) AS pictures"
    "  FROM derived_event e JOIN derived_event_run r ON r.id = e.run_id"
    " WHERE r.context_generation = (SELECT generation FROM derived_context_state)"
    "   AND r.context_policy_version = ?"
    " ORDER BY e.instant_start DESC LIMIT ?"
)


#: The surface: pictures per bin of the human moment, ONE statement
#: for any zoom. Only rows FINE enough for the bin are counted in it
#: (a day-fine claim does not fall into an hour); the coarse rest are
#: returned as spans by TIMELINE_SPANS so the page draws them across
#: the bins they cover -- shown at the width the signal has. Each bin also says
#: how many of its pictures spoke on the wall clock and how many only
#: as instants -- the two domains the axis coalesces for the door.
#: Bins are anchored: `CAST((m - anchor) / w) * w + anchor`, so a week
#: starts on a Monday (the epoch's day 0 is a Thursday; 345,600s later
#: is Monday 1970-01-05) and every other bin starts where the epoch
#: does. Each bin also says how many of its pictures were captured,
#: generated, both, or merely imported.
TIMELINE_DENSITY = (
    "SELECT CAST((" + HUMAN_MOMENT + " - ?) / ? AS INTEGER) * ? + ? AS bin, count(*) AS pictures,"
    " sum(mc.local_at IS NOT NULL) AS wall, sum(mc.local_at IS NULL) AS instant,"
    " sum(mc.origin = 'captured') AS captured, sum(mc.origin = 'generated') AS generated,"
    " sum(mc.origin = 'mixed') AS mixed, sum(mc.origin = 'imported') AS imported"
    "  FROM derived_media_context mc"
    "  JOIN file f ON f.id = mc.file_id AND f.missing_since IS NULL"
    " WHERE mc.policy_version = ?"
    "   AND " + HUMAN_MOMENT + " >= ? AND " + HUMAN_MOMENT + " < ?"
    "   AND mc.time_precision IN (SELECT value FROM json_each(?))"
    " GROUP BY bin ORDER BY bin"
)

#: The first few pictures of each bin, in moment order -- the strip of
#: thumbnails under the bars. Window-numbered so one statement serves
#: every bin; asked only when the page draws few enough bins to show
#: them (db/pages.py timeline_samples).
TIMELINE_BIN_SAMPLES = (
    "SELECT bin, slug FROM ("
    "  SELECT CAST((" + HUMAN_MOMENT + " - ?) / ? AS INTEGER) * ? + ? AS bin, e.slug,"
    "   row_number() OVER (PARTITION BY CAST((" + HUMAN_MOMENT + " - ?) / ? AS INTEGER)"
    "     ORDER BY " + HUMAN_MOMENT + ", mc.file_id) AS rn"
    "    FROM derived_media_context mc"
    "    JOIN file f ON f.id = mc.file_id AND f.missing_since IS NULL"
    "    JOIN entity e ON e.id = mc.file_id"
    "   WHERE mc.policy_version = ?"
    "     AND " + HUMAN_MOMENT + " >= ? AND " + HUMAN_MOMENT + " < ?"
    "     AND mc.time_precision IN (SELECT value FROM json_each(?)))"
    " WHERE rn <= ? ORDER BY bin, rn"
)

#: A session's first members, by the grouper's own ordinal.
SESSION_SAMPLES = (
    "SELECT ef.event_id, e.slug FROM derived_event_file ef JOIN entity e ON e.id = ef.file_id"
    " WHERE ef.event_id = ? AND ef.ordinal < ? ORDER BY ef.ordinal"
)

#: How much of the library the timeline can show, and how much of that
#: the sources disputed -- the honesty line above the surface.
TIMELINE_COVERAGE = (
    "SELECT count(mc.file_id), count(*), sum(mc.time_conflicts IS NOT NULL) FROM file f"
    " LEFT JOIN derived_media_context mc ON mc.file_id = f.id AND mc.policy_version = ?"
    " WHERE f.missing_since IS NULL"
)

#: One picture's place on the human timeline, with its evidence: the
#: media page's "when" block reads this and nothing else.
MEDIA_WHEN = (
    "SELECT local_at, instant_at, tz_offset_min, time_basis, time_certainty, time_supports,"
    " time_conflicts, time_precision, origin, " + HUMAN_MOMENT + " AS moment,"
    " strftime('%Y-%m-%d', " + HUMAN_MOMENT + ", 'unixepoch') AS local_day"
    "  FROM derived_media_context mc WHERE mc.file_id = ? AND mc.policy_version = ?"
)

#: The CURRENT sessions one picture belongs to, each with its story
#: render when one was told of exactly that subject.
MEDIA_SESSIONS = (
    "SELECT ev.id, ev.kind, COALESCE(ev.local_start, ev.instant_start), COALESCE(ev.local_end, ev.instant_end),"
    " (SELECT count(*) FROM derived_event_file x WHERE x.event_id = ev.id),"
    " (SELECT sr.id FROM story_snapshot s JOIN story_plan sp ON sp.snapshot_id = s.id"
    "   JOIN story_render sr ON sr.plan_id = sp.id WHERE s.member_hash = ev.member_hash"
    "   AND s.event_kind = ev.kind AND s.grouper = r.grouper ORDER BY sr.id DESC LIMIT 1)"
    "  FROM derived_event_file ef JOIN derived_event ev ON ev.id = ef.event_id"
    "  JOIN derived_event_run r ON r.id = ev.run_id"
    " WHERE ef.file_id = ? AND r.context_generation = (SELECT generation FROM derived_context_state)"
    "   AND r.context_policy_version = ? ORDER BY ev.id"
)

#: The claims too coarse for the bin, as spans: each precision's
#: window start and width, with how many pictures claim it.
TIMELINE_SPANS = (
    "SELECT " + HUMAN_MOMENT + " AS start, mc.time_precision, count(*) AS pictures"
    "  FROM derived_media_context mc"
    "  JOIN file f ON f.id = mc.file_id AND f.missing_since IS NULL"
    " WHERE mc.policy_version = ?"
    "   AND " + HUMAN_MOMENT + " >= ? AND " + HUMAN_MOMENT + " < ?"
    "   AND mc.time_precision NOT IN (SELECT value FROM json_each(?))"
    " GROUP BY start, mc.time_precision ORDER BY start"
)

#: The extent of the interpreted library on the human axis.
TIMELINE_EXTENT = (
    "SELECT min(" + HUMAN_MOMENT + "), max(" + HUMAN_MOMENT + "), count(*)"
    "  FROM derived_media_context mc"
    "  JOIN file f ON f.id = mc.file_id AND f.missing_since IS NULL"
    " WHERE mc.policy_version = ?"
)

#: Sessions whose interval touches a range, in THEIR domain (the wall
#: pair when the session knows it, the instant pair otherwise -- the
#: same coalesce as the axis), with the latest story told of exactly
#: this SUBJECT, when one exists: the same event kind, the same grouper,
#: the same ordered membership. `member_hash` alone is a checksum of
#: the files -- a capture session and a generation session over the
#: same mixed files share it -- never an identity on its own.
TIMELINE_SESSIONS = (
    "SELECT e.id, e.kind, e.local_start, e.local_end, e.instant_start, e.instant_end,"
    " (SELECT count(*) FROM derived_event_file ef WHERE ef.event_id = e.id) AS pictures,"
    " (SELECT s.id FROM story_snapshot s WHERE s.member_hash = e.member_hash"
    "   AND s.event_kind = e.kind AND s.grouper = r.grouper ORDER BY s.id DESC LIMIT 1),"
    " (SELECT sr.id FROM story_snapshot s JOIN story_plan sp ON sp.snapshot_id = s.id"
    "   JOIN story_render sr ON sr.plan_id = sp.id WHERE s.member_hash = e.member_hash"
    "   AND s.event_kind = e.kind AND s.grouper = r.grouper"
    "   ORDER BY sr.id DESC LIMIT 1),"
    " e.place_id, p.name, pe.slug"
    "  FROM derived_event e JOIN derived_event_run r ON r.id = e.run_id"
    "  LEFT JOIN place p ON p.id = e.place_id LEFT JOIN entity pe ON pe.id = e.place_id"
    " WHERE r.context_generation = (SELECT generation FROM derived_context_state)"
    "   AND r.context_policy_version = ?"
    "   AND COALESCE(e.local_start, e.instant_start) < ?"
    "   AND COALESCE(e.local_end, e.instant_end) >= ?"
    " ORDER BY COALESCE(e.local_start, e.instant_start)"
)

#: Bin widths a zoom may ask for, by name. Anything else is refused.
BINS = {"week": 604_800, "day": 86_400, "hour": 3_600, "quarter": 900, "minute": 60}
#: Where each bin's grid starts: Monday for the week, the epoch otherwise.
MONDAY = 345_600
_ANCHOR = {"week": MONDAY}
#: Bins few enough to carry a thumbnail strip.
SAMPLED_BINS_MOST = 120
SAMPLES_PER_BIN = 3
SAMPLES_PER_SESSION = 6
#: Which precisions are fine enough for each bin: a claim enters a bin
#: only when its own granule fits inside it.
_FINE_ENOUGH = {
    "week": ["day", "hour", "minute", "second", "subsecond"],
    "day": ["day", "hour", "minute", "second", "subsecond"],
    "hour": ["hour", "minute", "second", "subsecond"],
    "quarter": ["minute", "second", "subsecond"],
    "minute": ["minute", "second", "subsecond"],
}
#: The most bins one answer carries. Wider asks are refused with the
#: remedy, never served as a 30,000-bar page.
MAX_BINS = 4_000


def timeline_extent(conn):
    return conn.execute(TIMELINE_EXTENT, (context.POLICY_VERSION,)).fetchone()


def timeline_density(conn, bin_name: str, lo: float, hi: float):
    """Bins of `bin_name` over [lo, hi) plus the spans too coarse for
    them. Refuses an unknown bin or an ask wider than MAX_BINS."""
    import json as _json

    if bin_name not in BINS:
        raise ValueError(f"no bin named {bin_name!r}; one of {', '.join(BINS)}")
    width = BINS[bin_name]
    if hi <= lo:
        raise ValueError("the range is empty")
    if (hi - lo) / width > MAX_BINS:
        raise ValueError(f"{int((hi - lo) / width)} bins of a {bin_name} is more than {MAX_BINS}; narrow the range")
    fine = _json.dumps(_FINE_ENOUGH[bin_name])
    anchor = _ANCHOR.get(bin_name, 0)
    bins = conn.execute(
        TIMELINE_DENSITY, (anchor, width, width, anchor, context.POLICY_VERSION, lo, hi, fine)
    ).fetchall()
    spans = conn.execute(TIMELINE_SPANS, (context.POLICY_VERSION, lo, hi, fine)).fetchall()
    return width, bins, spans


def timeline_samples(conn, bin_name: str, lo: float, hi: float, bins: int) -> dict[int, list[str]]:
    """`{bin: [slug, ...]}` -- the first SAMPLES_PER_BIN pictures of each
    bin, when the answer carries SAMPLED_BINS_MOST bins or fewer; an
    empty map otherwise, and the page says so rather than drawing a
    strip of 4,000 thumbnails."""
    import json as _json

    if bins > SAMPLED_BINS_MOST:
        return {}
    width = BINS[bin_name]
    anchor = _ANCHOR.get(bin_name, 0)
    fine = _json.dumps(_FINE_ENOUGH[bin_name])
    held: dict[int, list[str]] = {}
    for at, slug in conn.execute(
        TIMELINE_BIN_SAMPLES,
        (anchor, width, width, anchor, anchor, width, context.POLICY_VERSION, lo, hi, fine, SAMPLES_PER_BIN),
    ):
        held.setdefault(int(at), []).append(slug)
    return held


def session_samples(conn, event_id: int) -> list[str]:
    return [row[1] for row in conn.execute(SESSION_SAMPLES, (event_id, SAMPLES_PER_SESSION))]


def timeline_coverage(conn) -> tuple[int, int, int]:
    """(interpreted present files, present files, contested contexts)."""
    row = conn.execute(TIMELINE_COVERAGE, (context.POLICY_VERSION,)).fetchone()
    return (int(row[0] or 0), int(row[1] or 0), int(row[2] or 0))


def media_when(conn, file_id: int):
    return conn.execute(MEDIA_WHEN, (file_id, context.POLICY_VERSION)).fetchone()


def media_sessions(conn, file_id: int):
    return conn.execute(MEDIA_SESSIONS, (file_id, context.POLICY_VERSION)).fetchall()


def timeline_sessions(conn, lo: float, hi: float):
    return conn.execute(TIMELINE_SESSIONS, (context.POLICY_VERSION, hi, lo)).fetchall()


def timeline_months(conn):
    return conn.execute(TIMELINE_MONTHS, (context.POLICY_VERSION,)).fetchall()


def timeline_days(conn, limit: int = 400):
    return conn.execute(TIMELINE_DAYS, (context.POLICY_VERSION, limit)).fetchall()


def timeline_events(conn, limit: int = 200):
    return conn.execute(TIMELINE_EVENTS, (context.POLICY_VERSION, limit)).fetchall()


#: Who is in one session, by the primary clustering: each person's
#: address, name (NULL until named) and how many of the session's
#: pictures hold them, most first. Bounded: a card names a few.
SESSION_PEOPLE = (
    "SELECT e.slug, p.name, count(DISTINCT fp.file_id) AS pictures"
    "  FROM derived_event_file ef"
    "  JOIN derived_file_person fp ON fp.file_id = ef.file_id"
    "  JOIN derived_face_run r ON r.id = fp.run_id AND r.is_primary = 1"
    "  JOIN person p ON p.id = fp.person_id JOIN entity e ON e.id = p.id"
    " WHERE ef.event_id = ? GROUP BY p.id ORDER BY pictures DESC, e.slug LIMIT ?"
)

#: People a session card names before saying "and others".
SESSION_PEOPLE_MOST = 4


def session_people(conn, event_id: int, limit: int = SESSION_PEOPLE_MOST):
    return conn.execute(SESSION_PEOPLE, (event_id, limit)).fetchall()


#: Where one picture happened, as the current interpretation holds it:
#: the place entity and on what basis. No row while uninterpreted or
#: placeless.
MEDIA_PLACE = (
    "SELECT p.id, e.slug, p.name, p.kind, mc.location_basis FROM derived_media_context mc"
    "  JOIN place p ON p.id = mc.place_id JOIN entity e ON e.id = p.id"
    " WHERE mc.file_id = ? AND mc.policy_version = ?"
)

#: Where one person was seen, by the primary clustering: each place and
#: how many of their pictures are there, most first.
PERSON_PLACES = (
    "SELECT p.id, e.slug, p.name, p.kind, count(DISTINCT fp.file_id) AS pictures"
    "  FROM derived_file_person fp JOIN derived_face_run r ON r.id = fp.run_id AND r.is_primary = 1"
    "  JOIN derived_media_context mc ON mc.file_id = fp.file_id AND mc.policy_version = ?"
    "  JOIN place p ON p.id = mc.place_id JOIN entity e ON e.id = p.id"
    " WHERE fp.person_id = ? GROUP BY p.id ORDER BY pictures DESC, p.name COLLATE NOCASE"
)


def person_places(conn, person_id: int):
    return conn.execute(PERSON_PLACES, (context.POLICY_VERSION, person_id)).fetchall()


#: Every place named, with how many present pictures the current
#: interpretation puts there, most first.
PLACES_SHELF = (
    "SELECT p.id, e.slug, p.name, p.kind, count(f.id) AS pictures,"
    " min(" + HUMAN_MOMENT + "), max(" + HUMAN_MOMENT + ")"
    "  FROM place p JOIN entity e ON e.id = p.id"
    "  LEFT JOIN derived_media_context mc ON mc.place_id = p.id AND mc.policy_version = ?"
    "  LEFT JOIN file f ON f.id = mc.file_id AND f.missing_since IS NULL"
    " GROUP BY p.id ORDER BY pictures DESC, p.name COLLATE NOCASE"
)


def places_shelf(conn):
    return conn.execute(PLACES_SHELF, (context.POLICY_VERSION,)).fetchall()


#: Every place anyone has named, for the picker. Bounded.
PLACES_NAMED = "SELECT p.name, p.kind FROM place p ORDER BY p.name COLLATE NOCASE LIMIT ?"


def media_place(conn, file_id: int):
    return conn.execute(MEDIA_PLACE, (file_id, context.POLICY_VERSION)).fetchone()


def places_named(conn, limit: int = 200):
    return conn.execute(PLACES_NAMED, (limit,)).fetchall()


# --- one picture's faces ----------------------------------------------------

#: Who the primary clustering says is in one file: each person's address,
#: name (NULL until a human names them) and how many of the file's faces
#: are theirs.
MEDIA_PEOPLE = (
    "SELECT e.slug, p.name, fp.face_count FROM derived_file_person fp"
    "  JOIN derived_face_run r ON r.id = fp.run_id AND r.is_primary = 1"
    "  JOIN person p ON p.id = fp.person_id JOIN entity e ON e.id = p.id"
    " WHERE fp.file_id = ? ORDER BY fp.face_count DESC, e.slug"
)

#: Every detector's pass over this file's CURRENT bytes: who looked, when,
#: how many faces it found. No row means nobody has looked at these bytes.
MEDIA_FACE_SCANS = (
    "SELECT s.model_id, s.model_version, s.faces, s.computed_at FROM derived_face_scan s"
    "  JOIN file f ON f.id = s.file_id AND f.content_sha256 = s.source_sha256"
    " WHERE s.file_id = ? ORDER BY s.computed_at DESC"
)


def media_people(conn, file_id: int):
    return conn.execute(MEDIA_PEOPLE, (file_id,)).fetchall()


def media_face_scans(conn, file_id: int):
    return conn.execute(MEDIA_FACE_SCANS, (file_id,)).fetchall()


# --- stories ---------------------------------------------------------------

#: Every story told, newest first: the render with its profile, the plan
#: it words and the frozen snapshot it was told of. The render's own
#: document carries the title, dek and hero refs; the snapshot's carries
#: the members those refs name. Walks story_render backwards by id and
#: stops.
STORIES = (
    "SELECT sr.id, sr.plan_id, sr.profile, sr.created_at, sr.document_json,"
    " sp.planner, s.event_kind, s.id, s.document_json"
    "  FROM story_render sr JOIN story_plan sp ON sp.id = sr.plan_id"
    "  JOIN story_snapshot s ON s.id = sp.snapshot_id"
)
_STORIES_OF_KIND = " WHERE s.event_kind = ?"
_STORIES_NEWEST = " ORDER BY sr.id DESC LIMIT ?"

#: How many stories each session kind has, for the shelf's filter.
STORY_KINDS = (
    "SELECT s.event_kind, count(*) FROM story_render sr JOIN story_plan sp ON sp.id = sr.plan_id"
    "  JOIN story_snapshot s ON s.id = sp.snapshot_id GROUP BY s.event_kind ORDER BY s.event_kind"
)


def stories(conn, limit: int = 60, kind: str | None = None):
    if kind is None:
        return conn.execute(STORIES + _STORIES_NEWEST, (limit,)).fetchall()
    return conn.execute(STORIES + _STORIES_OF_KIND + _STORIES_NEWEST, (kind, limit)).fetchall()


def story_kinds(conn) -> list[tuple[str, int]]:
    return [(kind, int(n)) for kind, n in conn.execute(STORY_KINDS)]


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
