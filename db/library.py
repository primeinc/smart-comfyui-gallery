"""The roots a library is made of, and the settings that describe it.

`root` is where bytes live. A library root holds the pictures; a mount is
removable and may be offline without that meaning anything was deleted; trash
is a real place a deleted file's bytes go, so restoring is a move and views
exclude the subtree by ancestry rather than by matching paths against a
configured string.

`online` is the flag the whole deletion doctrine rests on. An unplugged drive
and an emptied folder look identical from a directory listing, so a root that
cannot be read is marked offline and its files are left alone -- unreachable
and deleted are different, and only one of them is recoverable.
"""

from __future__ import annotations

import json
import logging
import os
import uuid

_logger = logging.getLogger(__name__)

#: Written inside a root so the directory can say which root it is. Dotted, so
#: `observe_tree` skips it along with everything else the app leaves lying in
#: a library.
MARKER = ".smartgallery-root"


def _marker(path) -> bytes | None:
    """The identity the directory claims, if it claims one."""
    try:
        with open(os.path.join(str(path), MARKER), encoding="ascii") as handle:
            return uuid.UUID(handle.read().strip()).bytes
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as why:
        _logger.warning("%s: marker unreadable: %s: %s", path, type(why).__name__, why)
        return None


def _write_marker(path, identity: bytes) -> None:
    try:
        with open(os.path.join(str(path), MARKER), "w", encoding="ascii", newline="") as handle:
            handle.write(str(uuid.UUID(bytes=identity)))
    except OSError as why:
        # A read-only mount can still be a root. It simply cannot prove which
        # one it is after a move, and falls back to matching on its path.
        _logger.warning("%s: marker not written, the root matches by path: %s: %s", path, type(why).__name__, why)


def where(path) -> str:
    """A root's path as it will be STORED: absolute and normalised.

    Every path in this library is composed from a root's -- `path_of`
    joins the root's recorded path to the folder names below it -- so a
    root stored relative makes every file's location depend on which
    directory the reading process happens to be in. The registering
    request and the background worker have no reason to share one, and
    the failure it produces is the confusing kind: FileNotFoundError on a
    file that is not missing, naming half a path.

    `abspath`, not `realpath`: a library deliberately reached through a
    symlink is a library at that path, and resolving the link would
    silently re-register it somewhere its owner did not name.
    """
    return os.path.abspath(os.path.normpath(str(path)))


def add_root(conn, path, kind: str, now: float) -> int:
    """Register a place bytes live.

    Idempotent on the root's identity, not on the string it was registered
    under. Registering a moved library used to mint a second root and leave
    the whole library behind the first one, which then read as offline; the
    marker inside the directory is what makes that a relocation instead.

    Normalising first makes the CHEAP half of that idempotence work as
    well: `lib`, `./lib` and the absolute spelling now match on the path
    column rather than falling through to the marker read.
    """
    path = where(path)
    row = conn.execute("SELECT id FROM root WHERE path = ?", (path,)).fetchone()
    if row:
        return row[0]

    claimed = _marker(path)
    if claimed is not None:
        known = conn.execute("SELECT id FROM root WHERE uuid = ?", (claimed,)).fetchone()
        if known:
            relocate(conn, known[0], path)
            return known[0]

    identity = claimed or uuid.uuid4().bytes
    cursor = conn.execute(
        "INSERT INTO root(uuid, path, kind, online, created_at) VALUES(?, ?, ?, 1, ?)",
        (identity, path, kind, now),
    )
    _write_marker(path, identity)
    return int(cursor.lastrowid or 0)


def root_path(conn, root_id: int) -> str | None:
    """Where one root lives, or None when no such root."""
    row = conn.execute("SELECT path FROM root WHERE id = ?", (root_id,)).fetchone()
    return str(row[0]) if row else None


def relocate(conn, root_id: int, path) -> None:
    """Point a root at where it now is.

    The library moves with it: nothing under the root is re-identified,
    because a folder's identity is its row and a file's is its bytes. This is
    the operation that did not exist, which is why a moved library had no way
    back short of a rebuild.
    """
    path = where(path)
    row = conn.execute("SELECT uuid FROM root WHERE id = ?", (root_id,)).fetchone()
    if row is None:
        raise LookupError(f"no root {root_id}")
    conn.execute(
        "UPDATE root SET path = ?, online = ? WHERE id = ?",
        (path, 1 if os.path.isdir(path) else 0, root_id),
    )
    if os.path.isdir(path):
        _write_marker(path, row[0])


def set_online(conn, root_id: int, online: bool) -> None:
    conn.execute("UPDATE root SET online = ? WHERE id = ?", (1 if online else 0, root_id))


def _reachable(path: str, identity: bytes) -> bool:
    """Whether this directory can be read AND is the root it claims to be.

    Reachable is not enough: a directory can exist at the recorded path
    and be a different one, which is what happens when a library is
    moved and something else takes its place. A marker that names
    another root means this is not it."""
    reachable = os.path.isdir(path)
    if reachable:
        claimed = _marker(path)
        if claimed is not None and claimed != identity:
            reachable = False
    return reachable


def probe_roots(conn, kinds: tuple[str, ...] | None = None) -> list[tuple[int, str, bool]]:
    """Observe each root's reachability WITHOUT recording it: filesystem
    and marker inspection only, no SQL writes, so a browsing GET can ask
    on a read-only connection without taking the writer lane. `kinds`
    bounds the probe -- navigation has no reason to stat a trash root.

    Returns `(root_id, path, reachable)`."""
    sql = "SELECT id, path, uuid FROM root"
    args: tuple = ()
    if kinds:
        sql += f" WHERE kind IN ({','.join('?' * len(kinds))})"
        args = kinds
    return [(root_id, path, _reachable(path, identity)) for root_id, path, identity in conn.execute(sql, args)]


def check_roots(conn) -> list[tuple[int, str, bool]]:
    """Look at each root and RECORD whether it can currently be read --
    the operational half; the caller owns the commit. Callers use this
    before a scan: a scan of an offline root would observe nothing and,
    without this flag, conclude that everything in it had been deleted.

    Returns `(root_id, path, online)`.
    """
    seen = probe_roots(conn)
    for root_id, _, reachable in seen:
        set_online(conn, root_id, reachable)
    return seen


#: What hangs off a file, as (name, table, its file column). Counted so
#: a person removing "just a directory" is told what else goes with it.
_ATTACHED = (
    ("ratings", "rating", "file_id"),
    ("favorites", "favorite", "file_id"),
    ("comments", "comment", "file_id"),
    ("people_named", "person_assertion", "file_id"),
    ("places", "file_place", "file_id"),
    ("keywords", "file_tag", "file_id"),
    ("in_collections", "collection_file", "file_id"),
)


def removal_cost(conn, root_id: int) -> dict:
    """Everything that goes if this root is removed.

    Counted BEFORE anything is touched, because the answer is the point:
    `folder.root_id` cascades to folders, folders cascade to files, and
    files cascade to every rating, comment, favourite, name and place
    somebody attached to them. A person deleting "just a directory"
    would otherwise find that out afterwards.

    Nothing on disk is counted because nothing on disk is touched. This
    removes rows.
    """
    told = {
        "root": root_id,
        "path": root_path(conn, root_id),
        "folders": conn.execute("SELECT count(*) FROM folder WHERE root_id = ?", (root_id,)).fetchone()[0],
        "files": conn.execute(
            "SELECT count(*) FROM file f JOIN folder d ON d.id = f.folder_id WHERE d.root_id = ?", (root_id,)
        ).fetchone()[0],
    }
    for name, table, column in _ATTACHED:
        told[name] = conn.execute(
            f"SELECT count(*) FROM {table} t JOIN file f ON f.id = t.{column}"
            "  JOIN folder d ON d.id = f.folder_id WHERE d.root_id = ?",
            (root_id,),
        ).fetchone()[0]
    return told


def forget_root(conn, root_id: int) -> dict:
    """Stop indexing a directory, and drop what was indexed from it.

    NOT a deletion of anything on disk -- the bytes are exactly where
    they were, and re-adding the directory finds them again. What goes
    is this library's knowledge of them, which is the part that cannot
    be recomputed: the ratings, the names, the places, the memberships.

    Returns what it removed, counted first, so a caller can say what
    happened rather than "done".
    """
    cost = removal_cost(conn, root_id)
    if cost["path"] is None:
        raise LookupError(f"no root {root_id}")
    # ON DELETE CASCADE does the rest: root -> folder -> file -> every
    # row that hangs off a file. Said here so the one line below is not
    # mistaken for a small one.
    conn.execute("DELETE FROM root WHERE id = ?", (root_id,))
    return cost


def roots(conn, *, kind=None) -> list[tuple]:
    if kind:
        return conn.execute("SELECT id, path, kind, online FROM root WHERE kind = ? ORDER BY path", (kind,)).fetchall()
    return conn.execute("SELECT id, path, kind, online FROM root ORDER BY path").fetchall()


# --- settings --------------------------------------------------------------


def put(conn, key: str, value) -> None:
    """One setting. Stored as JSON text so a type survives the round trip."""
    conn.execute(
        "INSERT INTO setting(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, json.dumps(value)),
    )


def get(conn, key: str, default=None):
    row = conn.execute("SELECT value FROM setting WHERE key = ?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row[0])
    except ValueError:
        return row[0]


def settings(conn) -> dict:
    return {key: get(conn, key) for (key,) in conn.execute("SELECT key FROM setting")}
