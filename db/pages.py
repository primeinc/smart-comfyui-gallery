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

Two rules hold across all of them.

**An address is resolved before the page is read, never inside it.** Joining
`entity` to match a slug inside the page query gives the planner a filter it
must reach through the file table, so it drives from the wrong end and sorts
the result -- measured, `SCAN f USING INDEX file_added` plus a temp B-tree on
a query that wants one folder. `resolve` does the lookup; the rest take ids.

**Ordering follows an index or it follows the join key.** A page that sorts
its whole result set costs the same as one that scans, and on the checkpoint
most of a library was made with, "its files sorted by name" is most of the
library sorted by name.
"""

from __future__ import annotations

#: How many rows a grid asks for at once.
PAGE = 60


def resolve(conn, kind: str, slug: str) -> int | None:
    """The entity an address names, following one retirement if it must.

    A slug that was renamed still answers, which is the whole point of
    keeping history: a live `entity.slug` wins, and history answers only on a
    miss, most recent retirement first.
    """
    row = conn.execute(
        "SELECT id FROM entity WHERE kind = ? AND slug = ?", (kind, slug)
    ).fetchone()
    if row:
        return row[0]
    row = conn.execute(
        "SELECT entity_id FROM slug_history WHERE kind = ? AND slug = ?"
        " ORDER BY retired_at DESC LIMIT 1",
        (kind, slug),
    ).fetchone()
    return row[0] if row else None


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
      (SELECT group_concat(a.name) FROM file_artifact fa
         JOIN artifact a ON a.id = fa.artifact_id
        WHERE fa.file_id = f.id AND fa.role = 'lora') AS loras,
      (SELECT p.text FROM generation g JOIN prompt p ON p.id = g.prompt_id
        WHERE g.file_id = f.id) AS prompt,
      (SELECT g.seed FROM generation g WHERE g.file_id = f.id) AS seed,
      (SELECT count(*) FROM file_param WHERE file_id = f.id) AS fields
    FROM file f JOIN folder fo ON fo.id = f.folder_id
   WHERE f.id = ?
"""

PARSED_FIELDS = (
    "SELECT source, key, value_text FROM file_param WHERE file_id = ?"
    " ORDER BY source, key"
)

NEIGHBOUR = (
    "SELECT e.slug FROM file f JOIN entity e ON e.id = f.id"
    " WHERE f.folder_id = ? AND f.missing_since IS NULL AND (f.mtime, f.id) {way} (?, ?)"
    " ORDER BY f.mtime {order}, f.id {order} LIMIT 1"
)


def picture(conn, file_id: int):
    return conn.execute(ONE_PICTURE, (file_id,)).fetchone()


def fields_of(conn, file_id: int):
    return conn.execute(PARSED_FIELDS, (file_id,)).fetchall()


def neighbour(conn, file_id: int, *, previous: bool = True):
    """The picture before or after this one in its folder.

    Ordered on (mtime, id), not (mtime, slug): the slug lives on `entity` and
    no index spans two tables, so tie-breaking on it sorted the whole folder
    to return one row -- 50,007 rows sorted per arrow-key press on the
    largest folder in a real library.
    """
    row = conn.execute(
        "SELECT folder_id, mtime FROM file WHERE id = ?", (file_id,)
    ).fetchone()
    if row is None:
        return None
    sql = NEIGHBOUR.format(**({"way": "<", "order": "DESC"} if previous
                              else {"way": ">", "order": "ASC"}))
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


# --- the recipe axis -------------------------------------------------------

ARTIFACT_FILES = (
    "SELECT fe.slug, f.name FROM file_artifact fa"
    " JOIN file f ON f.id = fa.file_id AND f.missing_since IS NULL"
    " JOIN entity fe ON fe.id = f.id WHERE fa.artifact_id = ?"
)

ARTIFACTS_BY_USE = (
    "SELECT a.name, e.slug, count(*) AS pictures FROM artifact a"
    "  JOIN entity e ON e.id = a.id"
    "  JOIN file_artifact fa ON fa.artifact_id = a.id"
    "  JOIN file f ON f.id = fa.file_id AND f.missing_since IS NULL"
    " WHERE a.kind = ? GROUP BY a.id ORDER BY pictures DESC, a.name"
)

#: What this LoRA is actually used with, and how often. The old app answered
#: this by matching one filename against a delimited blob holding another and
#: calling co-residency a match; it is a join with real counts.
LORA_SYNERGY = (
    "SELECT ckpt.name, e.slug, count(*) AS together FROM file_artifact fl"
    "  JOIN file_artifact fc ON fc.file_id = fl.file_id AND fc.role = 'checkpoint'"
    "  JOIN artifact ckpt ON ckpt.id = fc.artifact_id"
    "  JOIN entity e ON e.id = ckpt.id"
    " WHERE fl.artifact_id = ? AND fl.role = 'lora'"
    " GROUP BY ckpt.id ORDER BY together DESC, ckpt.name"
)


def artifacts_by_use(conn, kind: str):
    """The models, LoRAs or workflows index -- counted by pictures, not by
    mentions, which is why it joins `file` rather than counting rows."""
    return conn.execute(ARTIFACTS_BY_USE, (kind,)).fetchall()


def artifact_files(conn, artifact_id: int):
    return conn.execute(ARTIFACT_FILES, (artifact_id,)).fetchall()


def lora_synergy(conn, lora_id: int):
    return conn.execute(LORA_SYNERGY, (lora_id,)).fetchall()


# --- people ----------------------------------------------------------------

#: Counted per (file, person), so two faces of one person in one photograph
#: are one picture. The old schema counted detections and needed a warning.
PEOPLE_BY_MOST = (
    "SELECT COALESCE(p.name, '(unnamed)') AS name, e.slug,"
    " count(DISTINCT fp.file_id) AS pictures"
    "  FROM person p JOIN entity e ON e.id = p.id"
    "  JOIN derived_file_person fp ON fp.person_id = p.id"
    "  JOIN file f ON f.id = fp.file_id AND f.missing_since IS NULL"
    " GROUP BY p.id ORDER BY pictures DESC, name"
)

PERSON_FILES = (
    "SELECT fe.slug, f.name FROM derived_file_person fp"
    "  JOIN file f ON f.id = fp.file_id AND f.missing_since IS NULL"
    "  JOIN entity fe ON fe.id = f.id WHERE fp.person_id = ?"
)

#: Where a person's pictures actually sit. The disagreement between the disk
#: layout and the meaning is the thing the six-axis design exists to show.
PERSON_ACROSS_FOLDERS = (
    "SELECT fo.name, e.slug, count(*) AS pictures FROM derived_file_person fp"
    "  JOIN file f ON f.id = fp.file_id AND f.missing_since IS NULL"
    "  JOIN folder fo ON fo.id = f.folder_id"
    "  JOIN entity e ON e.id = fo.id"
    " WHERE fp.person_id = ? GROUP BY fo.id ORDER BY pictures DESC, fo.name"
)


def people_by_most(conn):
    return conn.execute(PEOPLE_BY_MOST).fetchall()


def person_files(conn, person_id: int):
    return conn.execute(PERSON_FILES, (person_id,)).fetchall()


def person_across_folders(conn, person_id: int):
    return conn.execute(PERSON_ACROSS_FOLDERS, (person_id,)).fetchall()


# --- albums ----------------------------------------------------------------

ALBUM_FILES = (
    "SELECT fe.slug, f.name FROM collection_file cf"
    "  JOIN file f ON f.id = cf.file_id AND f.missing_since IS NULL"
    "  JOIN entity fe ON fe.id = f.id WHERE cf.collection_id = ?"
    " ORDER BY cf.file_id"
)


def album_files(conn, collection_id: int):
    return conn.execute(ALBUM_FILES, (collection_id,)).fetchall()


# --- what can be searched --------------------------------------------------

WAYS = (
    "SELECT source, key, value_kind, occurrences FROM param_key"
    " ORDER BY occurrences DESC, key"
)


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
