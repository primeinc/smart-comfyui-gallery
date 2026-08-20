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


def add_root(conn, path, kind: str, now: float) -> int:
    """Register a place bytes live. Idempotent on the path."""
    path = str(path)
    row = conn.execute("SELECT id FROM root WHERE path = ?", (path,)).fetchone()
    if row:
        return row[0]
    cursor = conn.execute(
        "INSERT INTO root(path, kind, online, created_at) VALUES(?, ?, 1, ?)",
        (path, kind, now),
    )
    return int(cursor.lastrowid or 0)


def set_online(conn, root_id: int, online: bool) -> None:
    conn.execute("UPDATE root SET online = ? WHERE id = ?", (1 if online else 0, root_id))


def check_roots(conn) -> list[tuple[int, str, bool]]:
    """Look at each root and record whether it can currently be read.

    Returns `(root_id, path, online)`. Callers use this before a scan: a
    scan of an offline root would observe nothing and, without this flag,
    conclude that everything in it had been deleted.
    """
    seen = []
    for root_id, path in conn.execute("SELECT id, path FROM root"):
        reachable = os.path.isdir(path)
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
