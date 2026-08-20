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
import os
import uuid


#: Written inside a root so the directory can say which root it is. Dotted, so
#: `observe_tree` skips it along with everything else the app leaves lying in
#: a library.
MARKER = ".smartgallery-root"


def _marker(path) -> bytes | None:
    """The identity the directory claims, if it claims one."""
    try:
        with open(os.path.join(str(path), MARKER), "r", encoding="ascii") as handle:
            return uuid.UUID(handle.read().strip()).bytes
    except (OSError, ValueError):
        return None


def _write_marker(path, identity: bytes) -> None:
    try:
        with open(os.path.join(str(path), MARKER), "w", encoding="ascii", newline="") as handle:
            handle.write(str(uuid.UUID(bytes=identity)))
    except OSError:
        # A read-only mount can still be a root. It simply cannot prove which
        # one it is after a move, and falls back to matching on its path.
        pass


def add_root(conn, path, kind: str, now: float) -> int:
    """Register a place bytes live.

    Idempotent on the root's identity, not on the string it was registered
    under. Registering a moved library used to mint a second root and leave
    the whole library behind the first one, which then read as offline; the
    marker inside the directory is what makes that a relocation instead.
    """
    path = str(path)
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


def relocate(conn, root_id: int, path) -> None:
    """Point a root at where it now is.

    The library moves with it: nothing under the root is re-identified,
    because a folder's identity is its row and a file's is its bytes. This is
    the operation that did not exist, which is why a moved library had no way
    back short of a rebuild.
    """
    path = str(path)
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


def check_roots(conn) -> list[tuple[int, str, bool]]:
    """Look at each root and record whether it can currently be read.

    Returns `(root_id, path, online)`. Callers use this before a scan: a
    scan of an offline root would observe nothing and, without this flag,
    conclude that everything in it had been deleted.
    """
    seen = []
    for root_id, path, identity in conn.execute("SELECT id, path, uuid FROM root").fetchall():
        reachable = os.path.isdir(path)
        # Reachable is not enough: a directory can exist at the recorded path
        # and be a different one, which is what happens when a library is
        # moved and something else takes its place. A marker that names
        # another root means this is not it.
        if reachable:
            claimed = _marker(path)
            if claimed is not None and claimed != identity:
                reachable = False
        set_online(conn, root_id, reachable)
        seen.append((root_id, path, reachable))
    return seen


def roots(conn, *, kind=None) -> list[tuple]:
    if kind:
        return conn.execute(
            "SELECT id, path, kind, online FROM root WHERE kind = ? ORDER BY path", (kind,)
        ).fetchall()
    return conn.execute("SELECT id, path, kind, online FROM root ORDER BY path").fetchall()


# --- settings --------------------------------------------------------------


def put(conn, key: str, value) -> None:
    """One setting. Stored as JSON text so a type survives the round trip."""
    conn.execute(
        "INSERT INTO setting(key, value) VALUES(?, ?)"
        " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
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
