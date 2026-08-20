"""Deciding what a file found on disk refers to.

This is the scanner's identity matcher. It lives here rather than in the test
file because an algorithm defined inside its own tests tests only itself: the
tests pass regardless of what the real scanner does, and whoever writes the
scanner writes a second implementation with nothing binding it to this one.

The obvious implementation is wrong in a way that silently corrupts identity.
Treating a `(folder_id, name)` hit as a match means that when two files
exchange names, both lookups succeed and every rating stays with the *path*
instead of following the bytes -- the exact defect the schema exists to remove,
reintroduced one layer up.

So a path hit is provisional. It is confirmed only when the content also
matches; everything else joins a changed set reconciled by content first. The
candidate pool includes rows still sitting at a path that now holds different
bytes, not only rows that vanished -- without that, a swap looks like two
unrelated edits.
"""

from __future__ import annotations

import enum
import hashlib
import os
import re
import unicodedata
import uuid
from typing import NamedTuple


class Outcome(enum.Enum):
    """What a scanned path turned out to be.

    AMBIGUOUS exists because sha256 proves byte equality, not object
    continuity: with two identical copies a delete/add pair cannot be
    attributed, and guessing moves one file's ratings onto another.
    """

    UNIQUE_MATCH = "unique_match"
    REPLACED = "replaced"
    AMBIGUOUS = "ambiguous"
    NEW = "new"


class Resolution(NamedTuple):
    outcome: Outcome
    file_id: int | None


def resolve_scan(conn, observed: dict[tuple[int, str], str | None]):
    """Map each observed ``(folder_id, name) -> content_sha256`` to a decision.

    Returns ``(resolutions, missing)`` where *resolutions* is keyed by the same
    tuples and *missing* lists file ids whose content was found nowhere. A
    missing file is never deleted here; the caller records ``missing_since``,
    because unreachable and deleted are different things.
    """
    rows = {
        (r[1], r[2]): (r[0], r[3])
        for r in conn.execute("SELECT id, folder_id, name, content_sha256 FROM file")
    }
    result: dict[tuple[int, str], Resolution] = {}
    settled: set[int] = set()

    # Pass 1 -- same place, same bytes. Nothing to reconcile, no hashing beyond
    # what the caller already had to do to fill `observed`.
    for key, sha in observed.items():
        row = rows.get(key)
        if row and sha is not None and row[1] == sha:
            result[key] = Resolution(Outcome.UNIQUE_MATCH, row[0])
            settled.add(row[0])

    # Pass 2 -- reconcile by content across everything still unsettled.
    candidates: dict[str, list[int]] = {}
    for file_id, sha in rows.values():
        if file_id not in settled and sha is not None:
            candidates.setdefault(sha, []).append(file_id)

    for key, sha in observed.items():
        if key in result:
            continue
        pool = [f for f in candidates.get(sha, []) if f not in settled] if sha else []
        if len(pool) == 1:
            result[key] = Resolution(Outcome.UNIQUE_MATCH, pool[0])
            settled.add(pool[0])
        elif len(pool) > 1:
            result[key] = Resolution(Outcome.AMBIGUOUS, None)

    # Pass 3 -- in-place replacement. Only now, with every content match
    # already claimed, does a row still sitting unclaimed at this exact path
    # mean the bytes there were overwritten. Running this before pass 2 would
    # be path-trust again: after a swap both paths still resolve, and each
    # would be called a replacement while the bytes went to the wrong row.
    #
    # The entity continues, so the address survives. Derived work is
    # invalidated by the changed hash; authored state is kept, because
    # silently dropping somebody's rating is worse than keeping a stale one
    # and the alternative breaks the URL.
    for key, sha in observed.items():
        if key in result:
            continue
        row = rows.get(key)
        if row and row[0] not in settled:
            result[key] = Resolution(Outcome.REPLACED, row[0])
            settled.add(row[0])
        else:
            result[key] = Resolution(Outcome.NEW, None)

    missing = [file_id for file_id, _ in rows.values() if file_id not in settled]
    return result, missing


# --- walking a real directory ---------------------------------------------

#: Suffix to `file.kind`. A suffix that is not here is not media and is
#: skipped: the library indexes pictures, not the .txt sitting beside them.
KIND_BY_SUFFIX = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
    ".bmp": "image", ".tif": "image", ".tiff": "image", ".avif": "image",
    ".jxl": "image", ".heic": "image",
    ".gif": "animated_image", ".apng": "animated_image",
    ".mp4": "video", ".mkv": "video", ".webm": "video", ".mov": "video",
    ".avi": "video", ".m4v": "video",
    ".mp3": "audio", ".wav": "audio", ".flac": "audio", ".m4a": "audio",
    ".ogg": "audio", ".opus": "audio",
    ".pdf": "document",
}

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


class Found(NamedTuple):
    """One media file as the filesystem currently reports it."""

    sha: str | None
    size: int
    mtime: float
    btime: float | None
    inode: int | None
    kind: str


def sha256_of(path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def slugify(text: str) -> str:
    """A URL-safe seed for a slug.

    Never the address by itself: `mint` owns collisions, because two files
    may legitimately share a name.
    """
    folded = unicodedata.normalize("NFKD", str(text))
    ascii_only = folded.encode("ascii", "ignore").decode("ascii").lower()
    return _SLUG_STRIP.sub("-", ascii_only).strip("-")


def mint(conn, kind: str, seed: str) -> int:
    """Create an entity and return its id.

    The slug is seeded from a name and then belongs to the entity: renaming
    the file on disk does not change it. An entity with nothing readable to
    seed from gets `<kind>-<short id>` rather than no address at all.
    """
    base = slugify(seed) or None
    row = conn.execute("SELECT max(id) FROM entity").fetchone()
    next_id = (row[0] or 0) + 1
    slug = base or f"{kind}-{next_id:x}"
    suffix = 1
    while conn.execute(
        "SELECT 1 FROM entity WHERE kind = ? AND slug = ?", (kind, slug)
    ).fetchone():
        suffix += 1
        slug = f"{base or kind}-{suffix}"
    conn.execute(
        "INSERT INTO entity(id, uuid, kind, slug) VALUES(?, ?, ?, ?)",
        (next_id, uuid.uuid4().bytes, kind, slug),
    )
    return next_id


def ensure_folder(
    conn, root_id: int, parent_id: int | None, name: str, inode: int | None = None,
    *, now: float | None = None,
) -> int:
    """The folder row for one path segment, created once.

    A directory has no bytes, so it cannot prove continuity the way a file
    does. The filesystem's own id is the substitute: it survives a rename and
    a move within the volume, so a renamed folder updates one row instead of
    minting a second and orphaning the first along with its URL.

    Name matching still has to work by itself -- the id is absent on some
    filesystems and different after a copy or a restore -- and is
    case-insensitive because the stated platform is, so `Portraits` and
    `portraits` are one folder rather than half a library each.
    """
    if inode is not None:
        row = conn.execute(
            "SELECT id, parent_id, name FROM folder WHERE root_id = ? AND inode = ?",
            (root_id, inode),
        ).fetchone()
        if row:
            if (row[1], row[2]) != (parent_id, name):
                conn.execute(
                    "UPDATE folder SET parent_id = ?, name = ? WHERE id = ?",
                    (parent_id, name, row[0]),
                )
            conn.execute(
                "UPDATE folder SET missing_since = NULL"
                " WHERE id = ? AND missing_since IS NOT NULL",
                (row[0],),
            )
            return row[0]

    if parent_id is None:
        row = conn.execute(
            "SELECT id, inode FROM folder WHERE root_id = ? AND parent_id IS NULL"
            " AND name = ? COLLATE NOCASE AND missing_since IS NULL",
            (root_id, name),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id, inode FROM folder WHERE parent_id = ? AND name = ? COLLATE NOCASE"
            " AND missing_since IS NULL",
            (parent_id, name),
        ).fetchone()
    taken_over = False
    if row and inode is not None and row[1] is not None and row[1] != inode:
        # The name matches and the filesystem says these are different
        # directories. Rename `Archive` to `Zoo` and create a fresh
        # `Archive`, and os.walk hands us the new one first (it sorts), so
        # adopting the name match here handed the new directory the old
        # one's entity, its slug and its watched-folder row -- while the real
        # one, met later, minted a second entity and lost its address.
        #
        # So the old row stands aside rather than being overwritten. It is
        # marked missing, which frees the name; if it is met further along
        # under its new name, the inode branch above claims it back and
        # clears the mark.
        conn.execute(
            "UPDATE folder SET missing_since = COALESCE(?, unixepoch()) WHERE id = ?",
            (now, row[0]),
        )
        row, taken_over = None, True

    if row is None and not taken_over:
        # A directory that went away and came back. Nothing is competing for
        # the name, so reclaiming the row is what keeps its address alive --
        # and it is only safe here, where no live row wanted it. The
        # take-over case above must never reach this, or it would hand the
        # new directory the row it just stood aside.
        if parent_id is None:
            row = conn.execute(
                "SELECT id, inode FROM folder WHERE root_id = ? AND parent_id IS NULL"
                " AND name = ? COLLATE NOCASE AND missing_since IS NOT NULL",
                (root_id, name),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id, inode FROM folder WHERE parent_id = ? AND name = ? COLLATE NOCASE"
                " AND missing_since IS NOT NULL",
                (parent_id, name),
            ).fetchone()
        if row:
            conn.execute("UPDATE folder SET missing_since = NULL WHERE id = ?", (row[0],))

    if row:
        if inode is not None:
            conn.execute("UPDATE folder SET inode = ? WHERE id = ?", (inode, row[0]))
        return row[0]

    folder_id = mint(conn, "folder", name)
    # depth is set by the folder_depth_ins trigger, so it is never computed
    # by two callers that could disagree.
    conn.execute(
        "INSERT INTO folder(id, root_id, parent_id, name, depth, inode)"
        " VALUES(?, ?, ?, ?, 0, ?)",
        (folder_id, root_id, parent_id, name, inode),
    )
    return folder_id


def _inode_of(path) -> int | None:
    """The filesystem's id for a directory, or None where there isn't one.

    Zero is treated as absent: filesystems that do not implement the concept
    report it for every entry, and one shared value would collapse every
    folder in the library onto a single row.
    """
    try:
        return os.stat(path).st_ino or None
    except OSError:
        return None


def observe_tree(conn, root_id: int, root_path, now: float | None = None) -> tuple[dict, int]:
    """Walk `root_path`, ensure its folders exist, and report its media.

    Returns ``(observed, hashed)`` where *observed* maps
    ``(folder_id, name)`` to :class:`Found`, and *hashed* counts the files
    whose bytes were actually read.

    Hashing reads every byte, so a file whose size, mtime *and* filesystem id
    all match the row already stored at that path keeps its recorded hash.
    That is what makes a rescan of an unchanged library cheap.

    The filesystem id is in that test because size and mtime are not enough:
    renaming two same-sized files onto each other's names changes neither, so
    the shortcut reported both as unchanged, handed the matcher the hashes it
    already had, and left the ratings on the paths. Content reconciliation
    cannot catch that -- it never sees the real bytes.
    """
    stored = {
        (folder_id, name): (size, mtime, inode, sha)
        for folder_id, name, size, mtime, inode, sha in conn.execute(
            "SELECT folder_id, name, size, mtime, inode, content_sha256 FROM file"
        )
    }
    observed: dict[tuple[int, str], Found] = {}
    hashed = 0
    root_path = os.fspath(root_path)
    root_folder = ensure_folder(
        conn, root_id, None, os.path.basename(root_path) or root_path, _inode_of(root_path),
        now=now,
    )
    folder_ids = {os.path.normcase(root_path): root_folder}

    for current, subdirs, names in os.walk(root_path):
        # A leading dot means "not the user's content", and the app puts its
        # own state directly inside the library root: caches, downloaded
        # weights, the root marker. Measured against a real library, 5998 of
        # the 11775 media files under it live in dot-directories -- and 5992
        # of those are the thumbnail cache, which would have entered the
        # gallery as photographs and outnumbered the real ones.
        subdirs[:] = sorted(d for d in subdirs if not d.startswith("."))
        names = sorted(n for n in names if not n.startswith("."))
        folder_id = folder_ids.get(os.path.normcase(current))
        if folder_id is None:
            continue
        for name in subdirs:
            child = os.path.join(current, name)
            folder_ids[os.path.normcase(child)] = ensure_folder(
                conn, root_id, folder_id, name, _inode_of(child), now=now
            )
        for name in names:
            kind = KIND_BY_SUFFIX.get(os.path.splitext(name)[1].lower())
            if kind is None:
                continue
            path = os.path.join(current, name)
            try:
                info = os.stat(path)
            except OSError:
                # Vanished or unreadable between walk and stat. Reporting it
                # as absent would make a transient lock look like a deletion,
                # so it is simply not observed on this pass.
                continue
            inode = info.st_ino or None
            previous = stored.get((folder_id, name))
            if previous is not None and previous[:3] == (info.st_size, info.st_mtime, inode):
                sha = previous[3]
            else:
                sha = sha256_of(path)
                hashed += 1
            observed[(folder_id, name)] = Found(
                sha=sha,
                size=info.st_size,
                mtime=info.st_mtime,
                btime=getattr(info, "st_birthtime", None),
                inode=inode,
                kind=kind,
            )

    # Directories that were there last time and are not there now. Marked,
    # never deleted -- deleting cascades to every file beneath and takes the
    # ratings with it, and a folder that has gone missing is exactly as
    # ambiguous as a file that has. `scan` refuses to run at all against a
    # root it cannot read, which is what keeps an unplugged drive from
    # arriving here as an empty tree.
    standing = set(folder_ids.values())
    for (folder_id,) in conn.execute(
        "SELECT id FROM folder WHERE root_id = ? AND missing_since IS NULL", (root_id,)
    ).fetchall():
        if folder_id not in standing:
            conn.execute(
                "UPDATE folder SET missing_since = COALESCE(?, unixepoch()) WHERE id = ?",
                (now, folder_id),
            )
    return observed, hashed


class ScanResult(NamedTuple):
    matched: int
    replaced: int
    added: int
    ambiguous: int
    missing: int
    hashed: int


def apply_scan(conn, observed: dict, now: float) -> ScanResult:
    """Write what the matcher decided.

    Nothing here deletes. A row whose bytes are gone gets `missing_since`,
    because a scan cannot tell an unplugged drive from a deletion, and the
    destructive reading of that ambiguity is the one that loses data.

    The write order is load-bearing, and each step exists because leaving it
    out raises a uniqueness error on a perfectly ordinary change.
    """
    resolutions, missing = resolve_scan(conn, {k: v.sha for k, v in observed.items()})
    counts = dict.fromkeys(Outcome, 0)

    # 1. Missing first. A path is exclusive only while the bytes are there, so
    #    marking the departed rows is what frees their names for whatever now
    #    stands in the same place.
    for file_id in missing:
        conn.execute(
            "UPDATE file SET missing_since = ? WHERE id = ? AND missing_since IS NULL",
            (now, file_id),
        )

    moves = []
    for key, resolution in resolutions.items():
        counts[resolution.outcome] += 1
        if resolution.outcome in (Outcome.UNIQUE_MATCH, Outcome.REPLACED):
            moves.append((key, resolution.file_id))

    # 2. Park the names of everything that moved. Two files that exchange
    #    names are each other's obstacle: whichever is written first collides
    #    with the one still holding the target name. The parked name uses '?',
    #    which Windows forbids in a filename, so it cannot collide with a real
    #    row -- and every row is parked under its own id, so not with each
    #    other either.
    parked = [
        (key, file_id)
        for key, file_id in moves
        if conn.execute(
            "SELECT 1 FROM file WHERE id = ? AND (folder_id <> ? OR name <> ?)",
            (file_id, key[0], key[1]),
        ).fetchone()
    ]
    for _, file_id in parked:
        conn.execute("UPDATE file SET name = ? WHERE id = ?", (f"?parked-{file_id}", file_id))

    # 3. The real update. Identity is untouched: only where the file sits, what
    #    it now weighs, and what it now hashes to.
    for (folder_id, name), file_id in moves:
        found = observed[(folder_id, name)]
        conn.execute(
            "UPDATE file SET folder_id = ?, name = ?, size = ?, mtime = ?,"
            " btime = ?, inode = ?, content_sha256 = ?, last_seen_at = ?,"
            " missing_since = NULL WHERE id = ?",
            (
                folder_id, name, found.size, found.mtime, found.btime,
                found.inode, found.sha, now, file_id,
            ),
        )

    # 4. New rows last, once every name they might want has been vacated.
    #    AMBIGUOUS lands here too: it becomes a new file rather than being
    #    resolved onto one of the candidates, because guessing would move
    #    somebody else's ratings onto this picture.
    for (folder_id, name), resolution in resolutions.items():
        if resolution.outcome in (Outcome.UNIQUE_MATCH, Outcome.REPLACED):
            continue
        found = observed[(folder_id, name)]
        file_id = mint(conn, "file", os.path.splitext(name)[0])
        conn.execute(
            "INSERT INTO file(id, folder_id, name, kind, size, mtime, btime,"
            " inode, content_sha256, first_seen_at, last_seen_at)"
            " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                file_id, folder_id, name, found.kind, found.size, found.mtime,
                found.btime, found.inode, found.sha, now, now,
            ),
        )

    return ScanResult(
        matched=counts[Outcome.UNIQUE_MATCH],
        replaced=counts[Outcome.REPLACED],
        added=counts[Outcome.NEW],
        ambiguous=counts[Outcome.AMBIGUOUS],
        missing=len(missing),
        hashed=0,
    )


class RootOffline(Exception):
    """The root could not be read, so nothing can be concluded about it."""


def scan(conn, root_id: int, root_path, now: float) -> ScanResult:
    """One pass: walk the root, decide, write.

    The veto comes first. `os.walk` on a path that is not there yields
    nothing and raises nothing, so a scan of an unplugged drive observes an
    empty library and every row in it -- files and now folders -- is marked
    missing. `library.check_roots` records the flag; this refuses to act on
    the reading, which is the half that actually protects anything.
    """
    if not os.path.isdir(root_path):
        conn.execute("UPDATE root SET online = 0 WHERE id = ?", (root_id,))
        raise RootOffline(
            f"{root_path} cannot be read. Nothing was marked missing: an "
            f"unreachable root and an emptied one look the same from here."
        )
    conn.execute("UPDATE root SET online = 1 WHERE id = ?", (root_id,))
    observed, hashed = observe_tree(conn, root_id, root_path, now)
    return apply_scan(conn, observed, now)._replace(hashed=hashed)
