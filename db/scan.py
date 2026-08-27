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

import contextlib
import enum
import hashlib
import os
import re
import unicodedata
import uuid
from typing import NamedTuple

from vision.decode import RAW_SUFFIXES as _RAW_SUFFIXES


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


def resolve_scan(conn, observed: dict[tuple[int, str], str | None], *, roots=None):
    """Map each observed ``(folder_id, name) -> content_sha256`` to a decision.

    Returns ``(resolutions, missing)`` where *resolutions* is keyed by the same
    tuples and *missing* lists file ids whose content was found nowhere. A
    missing file is never deleted here; the caller records ``missing_since``,
    because unreachable and deleted are different things.

    The candidate pool is deliberately global, unlike `observe_tree`. A file
    dragged from one drive to another is a move, and scoping the pool to the
    root being walked would make it a delete on one scan and an unrelated
    arrival on the next -- the data-loss shape this whole module exists to
    remove. `scan_all` is how a multi-root library gets that in one pass
    whatever order the roots come in.

    `roots` is what stops that generosity from becoming its own defect. The
    pool has to span the library; the MISSING verdict must not. `observed`
    covers only the tree the caller walked, so with the two of them global
    together, scanning one root reported every file in every other root as
    missing -- and in a two-root library each scan flagged the other half,
    alternately, for ever. A row nobody looked for is unexamined, not gone.
    Passing None means "I walked everything", which is only true of
    `scan_all`.
    """
    rows = {(r[1], r[2]): (r[0], r[3]) for r in conn.execute("SELECT id, folder_id, name, content_sha256 FROM file")}
    result: dict[tuple[int, str], Resolution] = {}
    settled: set[int] = set()

    # Pass 1 -- same place, same bytes. Nothing to reconcile, no hashing beyond
    # what the caller already had to do to fill `observed`.
    #
    # A stored hash of NULL means "never hashed", not "different bytes". It
    # never equals anything, so the row fell past pass 2 (which skips NULL
    # candidates) into pass 3 and was reported REPLACED -- on every scan,
    # for the life of the row, because the hash was never acquired either.
    # REPLACED is what says the bytes were overwritten, so the whole library
    # re-invalidated its derived work every pass.
    for key, sha in observed.items():
        row = rows.get(key)
        if row and (row[1] is None or row[1] == sha):
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
    for key in observed:
        if key in result:
            continue
        row = rows.get(key)
        if row and row[0] not in settled:
            result[key] = Resolution(Outcome.REPLACED, row[0])
            settled.add(row[0])
        else:
            result[key] = Resolution(Outcome.NEW, None)

    examined = None
    if roots is not None:
        marks = ",".join("?" * len(roots))
        examined = {
            row[0]
            for row in conn.execute(
                f"SELECT f.id FROM file f JOIN folder d ON d.id = f.folder_id WHERE d.root_id IN ({marks})",
                tuple(roots),
            )
        }
    missing = [
        file_id for file_id, _ in rows.values() if file_id not in settled and (examined is None or file_id in examined)
    ]
    return result, missing


# --- walking a real directory ---------------------------------------------

#: Suffix to `file.kind`. A suffix that is not here is not media and is
#: skipped: the library indexes pictures, not the .txt sitting beside them.
#: Every suffix here is DECODABLE by this install -- the rule vision/decode.py
#: states, held by tests/test_every_claimed_suffix_is_supported.py. Stills
#: through Pillow and its shipped plugins, RAW through LibRaw, moving
#: pictures through PyAV. `animated_image` for .gif/.apng is provisional by
#: suffix; ingest refines it from the decoded frame count, because an
#: animated WebP/AVIF/PNG wears the same suffix as its still sibling.
KIND_BY_SUFFIX = {
    # stills Pillow decodes natively or via registered plugins
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".jpe": "image",
    ".jfif": "image",
    ".jif": "image",
    ".webp": "image",
    ".bmp": "image",
    ".dib": "image",
    ".tif": "image",
    ".tiff": "image",
    ".avif": "image",
    ".jxl": "image",
    ".heic": "image",
    ".heif": "image",
    ".hif": "image",
    ".heics": "image",
    ".heifs": "image",
    ".jp2": "image",
    ".j2k": "image",
    ".jpf": "image",
    ".jpx": "image",
    ".mpo": "image",
    ".psd": "image",
    ".gif": "animated_image",
    ".apng": "animated_image",
    # the RAW family, via LibRaw
    **dict.fromkeys(_RAW_SUFFIXES, "image"),
    # video containers, via PyAV -- every one mux/demux/decode proven
    ".mp4": "video",
    ".m4v": "video",
    ".mov": "video",
    ".qt": "video",
    ".mkv": "video",
    ".webm": "video",
    ".avi": "video",
    ".divx": "video",
    ".mpg": "video",
    ".mpeg": "video",
    ".mpe": "video",
    ".m2v": "video",
    ".mjpeg": "video",
    ".mjpg": "video",
    ".ogv": "video",
    ".ogm": "video",
    ".vob": "video",
    ".ts": "video",
    ".mts": "video",
    ".m2ts": "video",
    ".m2t": "video",
    ".3gp": "video",
    ".3gpp": "video",
    ".wmv": "video",
    ".asf": "video",
    ".flv": "video",
    ".f4v": "video",
    ".mxf": "video",
    ".rm": "video",
    ".rmvb": "video",
    # audio, via PyAV
    ".mp3": "audio",
    ".mp2": "audio",
    ".wav": "audio",
    ".flac": "audio",
    ".m4a": "audio",
    ".ogg": "audio",
    ".oga": "audio",
    ".mka": "audio",
    ".weba": "audio",
    ".caf": "audio",
    ".au": "audio",
    ".opus": "audio",
    ".aac": "audio",
    ".wma": "audio",
    ".aiff": "audio",
    ".aif": "audio",
    # documents, via pypdf
    ".pdf": "document",
}

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


class Found(NamedTuple):
    """One media file as the filesystem currently reports it."""

    sha: str | None
    size: int
    mtime: float
    btime: float | None
    fs_id: str | None
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


def _taken(conn, kind: str, slug: str) -> bool:
    return conn.execute("SELECT 1 FROM entity WHERE kind = ? AND slug = ?", (kind, slug)).fetchone() is not None


def _free_slug(conn, kind: str, stem: str) -> str:
    """`stem-N` for the first N nobody has used, found in ONE read.

    Probing `stem-2`, `stem-3`, ... one SELECT at a time is quadratic in
    the number of files sharing a name, and files sharing a name is how
    libraries are ORGANISED: a cover.jpg in every album, a card of
    photographs numbered from DSC00001 beside another card numbered the
    same. Measured before this: 4,000 files all called cover.png held
    SQLite's one write lane for 16.8 seconds against 0.6 for 4,000
    distinct names, quadrupling with every doubling -- so twenty
    thousand albums was minutes of held lane, and every write from a
    route in that window fails rather than waits.

    Found by doubling until a free suffix appears and then bisecting
    back, so it costs O(log n) reads instead of n. Every read is a point
    lookup on `UNIQUE (kind, slug)`, which is the one thing this table
    is certainly indexed for. Asking SQLite for the highest suffix
    instead -- `max(CAST(substr(slug, ...)))` over a range -- reads no
    better, because a max over an EXPRESSION cannot use the index and
    scans every `stem-` row: measured, that was still quadratic, 2.4
    seconds at 4,000 where this is 0.4.

    The number it lands on is the same one the old probe found, because
    with no gaps the first free suffix is the only free suffix. Where
    there ARE gaps it may return a later one, which is not a defect:
    `slug_history` answers a retired address on a miss, so handing a
    retired slug to a new entity would make somebody's saved link
    resolve to a different picture -- the same reason this module
    refuses to compute `max(id) + 1`.
    """
    prefix = f"{stem}-"

    def taken(suffix: int) -> bool:
        return _taken(conn, kind, f"{prefix}{suffix}")

    if not taken(2):
        return f"{prefix}2"
    # `low` is taken and `high` is free, held across both loops.
    low, high = 2, 4
    while taken(high):
        low, high = high, high * 2
    while low + 1 < high:
        middle = (low + high) // 2
        if taken(middle):
            low = middle
        else:
            high = middle
    return f"{prefix}{high}"


def mint(conn, kind: str, seed: str) -> int:
    """Create an entity and return its id.

    The slug is seeded from a name and then belongs to the entity: renaming
    the file on disk does not change it. An entity with nothing readable to
    seed from gets `<kind>-<short id>` rather than no address at all, taken
    from its uuid.

    The id comes from SQLite, never from `max(id) + 1`. Computing it here
    reused the id of a deleted entity -- with a different uuid, so an id held
    anywhere outside this database resolved afterwards to a different picture
    -- and, being a second statement with no transaction around it, handed
    two concurrent writers the same number. `entity.id` is AUTOINCREMENT and
    this reads what the insert allocated.
    """
    base = slugify(seed) or None
    identity = uuid.uuid4()
    slug = base or f"{kind}-{identity.hex[:6]}"
    if _taken(conn, kind, slug):
        slug = _free_slug(conn, kind, base or kind)
    cursor = conn.execute(
        "INSERT INTO entity(uuid, kind, slug) VALUES(?, ?, ?)",
        (identity.bytes, kind, slug),
    )
    return int(cursor.lastrowid or 0)


def ensure_folder(
    conn,
    root_id: int,
    parent_id: int | None,
    name: str,
    fs_id: str | None = None,
    *,
    now: float | None = None,
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
    if fs_id is not None:
        row = conn.execute(
            "SELECT id, parent_id, name FROM folder WHERE root_id = ? AND fs_id = ?",
            (root_id, fs_id),
        ).fetchone()
        if row:
            if (row[1], row[2]) != (parent_id, name):
                conn.execute(
                    "UPDATE folder SET parent_id = ?, name = ? WHERE id = ?",
                    (parent_id, name, row[0]),
                )
            conn.execute(
                "UPDATE folder SET missing_since = NULL WHERE id = ? AND missing_since IS NOT NULL",
                (row[0],),
            )
            return row[0]

    if parent_id is None:
        row = conn.execute(
            "SELECT id, fs_id FROM folder WHERE root_id = ? AND parent_id IS NULL"
            " AND name = ? COLLATE NOCASE AND missing_since IS NULL",
            (root_id, name),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id, fs_id FROM folder WHERE parent_id = ? AND name = ? COLLATE NOCASE AND missing_since IS NULL",
            (parent_id, name),
        ).fetchone()
    taken_over = False
    if row and fs_id is not None and row[1] is not None and row[1] != fs_id:
        # The name matches and the filesystem says these are different
        # directories. Rename `Archive` to `Zoo` and create a fresh
        # `Archive`, and os.walk hands us the new one first (it sorts), so
        # adopting the name match here handed the new directory the old
        # one's entity, its slug and its watched-folder row -- while the real
        # one, met later, minted a second entity and lost its address.
        #
        # So the old row stands aside rather than being overwritten. It is
        # marked missing, which frees the name; if it is met further along
        # under its new name, the fs_id branch above claims it back and
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
                "SELECT id, fs_id FROM folder WHERE root_id = ? AND parent_id IS NULL"
                " AND name = ? COLLATE NOCASE AND missing_since IS NOT NULL",
                (root_id, name),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id, fs_id FROM folder WHERE parent_id = ? AND name = ? COLLATE NOCASE"
                " AND missing_since IS NOT NULL",
                (parent_id, name),
            ).fetchone()
        if row:
            conn.execute("UPDATE folder SET missing_since = NULL WHERE id = ?", (row[0],))

    if row:
        if fs_id is not None:
            conn.execute("UPDATE folder SET fs_id = ? WHERE id = ?", (fs_id, row[0]))
        return row[0]

    folder_id = mint(conn, "folder", name)
    # depth is set by the folder_depth_ins trigger, so it is never computed
    # by two callers that could disagree.
    conn.execute(
        "INSERT INTO folder(id, root_id, parent_id, name, depth, fs_id) VALUES(?, ?, ?, ?, 0, ?)",
        (folder_id, root_id, parent_id, name, fs_id),
    )
    return folder_id


def fs_id(value: int) -> str | None:
    """The filesystem's own identifier for a file or directory, as stored.

    Opaque on purpose. Nothing does arithmetic on this -- it is compared
    for equality and nothing else -- so the honest representation is the
    exact one, and the exact one is text.

    Since Python 3.12 Windows reports `st_ino` "up to 128 bits, depending
    on the file system" (cpython Doc/library/os.rst); SQLite's INTEGER is
    signed 64-bit (sqlite/sqlite src/sqliteInt.h:1031 LARGEST_INT64). A
    larger value bound to the old INTEGER column raised

        OverflowError: Python int too large to convert to SQLite INTEGER

    and ended the scan on the first ReFS directory it walked. Folding or
    masking it to fit would have turned an exact identifier into a
    lossy one inside the code whose whole job is to avoid assigning the
    wrong identity, so the column changed instead.

    Zero is absent: filesystems that do not implement the concept report
    it for every entry, and one shared value would collapse every folder
    in the library onto a single row.
    """
    return str(value) if value else None


def _fs_id_of(path) -> str | None:
    """The filesystem's id for a directory, or None where there isn't one."""
    try:
        return fs_id(os.stat(path).st_ino)
    except OSError:
        return None


class Survey(NamedTuple):
    """What a walk saw, before anything is written.

    Keyed by PATH rather than by folder id, because a folder id is a row
    and rows are what the write phase makes. Producing this costs the
    whole walk and a hash of every changed file; consuming it costs a
    handful of statements.
    """

    #: every directory met, parents first (os.walk is top-down), as
    #: (path, parent path or None, name, filesystem id)
    dirs: list[tuple[str, str | None, str, str | None]]
    #: (directory path, file name) -> what was found there
    files: dict[tuple[str, str], Found]
    #: how many files had to be read to hash them
    hashed: int


def _folders_by_path(conn, root_id: int, root_path: str) -> dict[str, int]:
    """The folder rows this root already has, by absolute path.

    Best effort ON PURPOSE. It exists so the walk can ask "have I seen
    this file before" without writing anything, and a path it fails to
    resolve simply means the file gets hashed -- correct, just slower.
    Nothing durable is decided from it: the write phase asks
    `ensure_folder`, which is authoritative.
    """
    kids: dict[int | None, list[tuple[int, str]]] = {}
    for folder_id, parent_id, name in conn.execute(
        "SELECT id, parent_id, name FROM folder WHERE root_id = ? AND missing_since IS NULL",
        (root_id,),
    ):
        kids.setdefault(parent_id, []).append((folder_id, name))
    found: dict[str, int] = {}
    for folder_id, _ in kids.get(None, []):
        stack = [(folder_id, root_path)]
        while stack:
            here, path = stack.pop()
            found[os.path.normcase(path)] = here
            for child, child_name in kids.get(here, []):
                stack.append((child, os.path.join(path, child_name)))
    return found


def survey(conn, root_id: int, root_path, now: float | None = None, watch=None) -> Survey:
    """Walk the tree and hash what changed, WITHOUT writing anything.

    This is the long half of a scan -- `sha256_of` reads every changed
    file off the disk -- and it used to run inside the same savepoint as
    the writes. SQLite has one write lane per database, so a scan of a
    new root (where every directory is created, opening the transaction
    on the first one) held that lane for as long as the hashing took.
    Everything else that writes was dead meanwhile: the background
    worker could not even claim a job, and each attempt raised
    `database is locked`.

    Nothing here writes, so nothing here holds the lane. The reads are
    two statements taken up front.

    `watch(folders, files, hashed)` is called as the walk goes, if given.
    This is the long half and it used to be silent: a person scanning a
    large root watched a request hang with nothing to look at, while
    every cheaper sweep that followed reported itself. The callback is
    how the walk says what it is doing without this module knowing what
    a job is.
    """
    del now  # nothing here is dated; the write phase carries the clock
    root_path = os.fspath(root_path)
    known = _folders_by_path(conn, root_id, root_path)
    # `stored` re-keyed by PATH, so the "unchanged, reuse its hash"
    # shortcut works without a folder id having been minted first.
    at_path = {folder_id: path for path, folder_id in known.items()}
    stored: dict[tuple[str, str], tuple] = {}
    for folder_id, name, size, mtime, identifier, sha in conn.execute(
        "SELECT f.folder_id, f.name, f.size, f.mtime, f.fs_id, f.content_sha256"
        "  FROM file f JOIN folder d ON d.id = f.folder_id WHERE d.root_id = ?",
        (root_id,),
    ):
        where = at_path.get(folder_id)
        if where is not None:
            stored[(where, name)] = (size, mtime, identifier, sha)

    dirs: list[tuple[str, str | None, str, str | None]] = [
        (root_path, None, os.path.basename(root_path) or root_path, _fs_id_of(root_path))
    ]
    seen = {os.path.normcase(root_path)}
    files: dict[tuple[str, str], Found] = {}
    hashed = 0

    for current, subdirs, names in os.walk(root_path):
        # A leading dot means "not the user's content", and the app puts its
        # own state directly inside the library root: caches, downloaded
        # weights, the root marker. Measured against a real library, 5998 of
        # the 11775 media files under it live in dot-directories -- and 5992
        # of those are the thumbnail cache, which would have entered the
        # gallery as photographs and outnumbered the real ones.
        subdirs[:] = sorted(d for d in subdirs if not d.startswith("."))
        kept = sorted(n for n in names if not n.startswith("."))
        if os.path.normcase(current) not in seen:
            continue
        for name in subdirs:
            child = os.path.join(current, name)
            dirs.append((child, current, name, _fs_id_of(child)))
            seen.add(os.path.normcase(child))
        for name in kept:
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
            # Same hazard as a directory's: a FILE on ReFS or a network
            # share reports a 128-bit identifier too, and this one reaches
            # an INSERT rather than a SELECT.
            held = fs_id(info.st_ino)
            previous = stored.get((os.path.normcase(current), name))
            # `previous[3] is not None` is part of the test: a row that has
            # never been hashed has nothing to reuse, and returning its NULL
            # here is what made the missing hash permanent -- the shortcut
            # kept handing back NULL for as long as size, mtime and fs_id
            # held still, which for an untouched file is forever.
            if previous is not None and previous[3] is not None and previous[:3] == (info.st_size, info.st_mtime, held):
                sha = previous[3]
            else:
                try:
                    sha = sha256_of(path)
                except OSError:
                    # Locked or vanished between stat and read. Not observed
                    # on this pass rather than observed as unreadable.
                    continue
                hashed += 1
            files[(current, name)] = Found(
                sha=sha,
                size=info.st_size,
                mtime=info.st_mtime,
                btime=getattr(info, "st_birthtime", None),
                fs_id=held,
                kind=kind,
            )
            # Per FILE, and the CALLER decides how often that is worth
            # acting on. Reporting per directory instead looks thriftier
            # and says nothing: a library is often one flat folder, and
            # 900 files in one of them produced exactly one report, at
            # the end, which is the silence this exists to break.
            if watch is not None:
                watch(len(dirs), len(files), hashed)
    return Survey(dirs=dirs, files=files, hashed=hashed)


def record(conn, root_id: int, held: Survey, now: float | None = None) -> tuple[dict, int]:
    """Turn a survey into folder rows and an observation keyed by them.

    The short half, and the only half that writes. Every statement here
    is a row: no filesystem is touched and nothing is hashed, so the
    write lane is held for about as long as the database takes.
    """
    folder_ids: dict[str, int] = {}
    for path, parent, name, identifier in held.dirs:
        parent_id = None if parent is None else folder_ids.get(os.path.normcase(parent))
        if parent is not None and parent_id is None:
            continue  # its parent was skipped, so this subtree is not ours
        folder_ids[os.path.normcase(path)] = ensure_folder(conn, root_id, parent_id, name, identifier, now=now)

    observed: dict[tuple[int, str], Found] = {}
    for (where, name), found in held.files.items():
        folder_id = folder_ids.get(os.path.normcase(where))
        if folder_id is not None:
            observed[(folder_id, name)] = found

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
    return observed, held.hashed


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
    held = survey(conn, root_id, root_path, now)
    return record(conn, root_id, held, now)


class ScanResult(NamedTuple):
    matched: int
    replaced: int
    added: int
    ambiguous: int
    missing: int
    hashed: int


def apply_scan(conn, observed: dict, now: float, *, hashed: int = 0, roots=None) -> ScanResult:
    """Write what the matcher decided.

    `roots` names what the caller actually walked, so a row in a root nobody
    looked at is left alone rather than reported gone. See `resolve_scan`.

    Nothing here deletes. A row whose bytes are gone gets `missing_since`,
    because a scan cannot tell an unplugged drive from a deletion, and the
    destructive reading of that ambiguity is the one that loses data.

    The write order is load-bearing, and each step exists because leaving it
    out raises a uniqueness error on a perfectly ordinary change. It is also
    the reason for the transaction: step 2 parks every moved row under a name
    like `?parked-12` and step 3 gives them their real ones, so a failure in
    between leaves a library of files literally called `?parked-12` unless
    the whole sequence is one write. Calling this "load-bearing" and leaving
    atomicity to whoever remembers is not a contract.
    """
    with _one_write(conn, "apply_scan"):
        return _apply(conn, observed, now, hashed, roots)


@contextlib.contextmanager
def _one_write(conn, name: str):
    """The enclosed writes, all or none.

    A SAVEPOINT rather than BEGIN/COMMIT, because this has to hold whichever
    way the connection is configured. Under Python's legacy default an
    ordinary INSERT has already opened a transaction, so `BEGIN` raises and a
    guard that skips itself when one is open would be inert on exactly the
    path it exists for. A savepoint nests inside a caller's transaction when
    there is one and starts its own when there is not, and `ROLLBACK TO`
    undoes precisely these writes either way.
    """
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield
    except BaseException:
        conn.execute(f"ROLLBACK TO {name}")
        conn.execute(f"RELEASE {name}")
        raise
    conn.execute(f"RELEASE {name}")


def _apply(conn, observed: dict, now: float, hashed: int, roots) -> ScanResult:
    from . import context as context_module

    resolutions, missing = resolve_scan(conn, {k: v.sha for k, v in observed.items()}, roots=roots)
    counts = dict.fromkeys(Outcome, 0)

    # 1. Missing first. A path is exclusive only while the bytes are there, so
    #    marking the departed rows is what frees their names for whatever now
    #    stands in the same place. Departing IS a population change: the
    #    file's interpretation goes stale with it -- deleting any event that
    #    claimed the picture and advancing the currentness generation, so a
    #    hypothesis over the old population stops being current in the same
    #    transaction that shrank it.
    for file_id in missing:
        told = conn.execute(
            "UPDATE file SET missing_since = ? WHERE id = ? AND missing_since IS NULL",
            (now, file_id),
        )
        if told.rowcount:
            context_module.stale(conn, file_id)

    # What the rows already say, so a scan can tell a file that changed from a
    # file it merely looked at again. Without this every matched row was
    # rewritten on every pass: at 80,000 files an unchanged rescan spent 3.3
    # seconds issuing 80,000 UPDATEs that set each column to the value it
    # already held, against 154 ms of actually deciding anything.
    was = {
        row[0]: row[1:]
        for row in conn.execute(
            "SELECT id, folder_id, name, size, mtime, btime, fs_id, content_sha256, missing_since FROM file"
        )
    }

    moves, changed = [], []
    for key, resolution in resolutions.items():
        counts[resolution.outcome] += 1
        if resolution.outcome not in (Outcome.UNIQUE_MATCH, Outcome.REPLACED):
            continue
        moves.append((key, resolution.file_id))
        found = observed[key]
        before = was.get(resolution.file_id)
        after = (
            key[0],
            key[1],
            found.size,
            found.mtime,
            found.btime,
            found.fs_id,
            found.sha,
            None,
        )
        if before != after:
            changed.append((key, resolution.file_id))

    # 2. Park the names of everything that moved. Two files that exchange
    #    names are each other's obstacle: whichever is written first collides
    #    with the one still holding the target name. The parked name uses '?',
    #    which Windows forbids in a filename, so it cannot collide with a real
    #    row -- and every row is parked under its own id, so not with each
    #    other either.
    #    Only rows that are actually going somewhere need parking, so this
    #    reads the loaded state rather than asking the database once per file.
    for key, file_id in changed:
        if was.get(file_id, (None, None))[:2] != key:
            conn.execute("UPDATE file SET name = ? WHERE id = ?", (f"?parked-{file_id}", file_id))

    # 3. The real update, for the rows that really differ. Identity is
    #    untouched: only where the file sits, what it now weighs, and what it
    #    now hashes to.
    for (folder_id, name), file_id in changed:
        found = observed[(folder_id, name)]
        # The filesystem's time claims are moving under this file: its
        # derived interpretation goes stale with them (db/context.py).
        context_module.stale(conn, file_id)
        conn.execute(
            "UPDATE file SET folder_id = ?, name = ?, size = ?, mtime = ?,"
            " btime = ?, fs_id = ?, content_sha256 = ?, last_seen_at = ?,"
            " missing_since = NULL WHERE id = ?",
            (
                folder_id,
                name,
                found.size,
                found.mtime,
                found.btime,
                found.fs_id,
                found.sha,
                now,
                file_id,
            ),
        )

    # 3b. Everything else that was seen is still where it was, and only
    #     `last_seen_at` has moved on. One statement over the ids rather than
    #     one statement each: this is the difference between a rescan of an
    #     unchanged library costing 3.3 seconds and costing 50 milliseconds.
    rewritten = {file_id for _, file_id in changed}
    untouched = [file_id for _, file_id in moves if file_id not in rewritten]
    if untouched:
        from . import connect

        for start in range(0, len(untouched), connect.PARAM_BATCH):
            batch = untouched[start : start + connect.PARAM_BATCH]
            conn.execute(
                f"UPDATE file SET last_seen_at = ? WHERE id IN ({','.join('?' * len(batch))})",
                (now, *batch),
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
            " fs_id, content_sha256, first_seen_at, last_seen_at)"
            " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                file_id,
                folder_id,
                name,
                found.kind,
                found.size,
                found.mtime,
                found.btime,
                found.fs_id,
                found.sha,
                now,
                now,
            ),
        )

    # A minted row grew the present population: no context of its own to
    # stale, but every event hypothesis is now a statement about a library
    # that no longer exists. Once per scan, not once per file.
    if counts[Outcome.NEW] or counts[Outcome.AMBIGUOUS]:
        context_module.repopulated(conn)

    return ScanResult(
        matched=counts[Outcome.UNIQUE_MATCH],
        replaced=counts[Outcome.REPLACED],
        added=counts[Outcome.NEW],
        ambiguous=counts[Outcome.AMBIGUOUS],
        missing=len(missing),
        hashed=hashed,
    )


class RootOffline(Exception):
    """The root could not be read, so nothing can be concluded about it."""


#: How often a walk says where it is, in files. Every directory would
#: be a write per folder, and a library of ten-thousand-file folders
#: would report once an aeon; this is a number of files, so the cadence
#: follows the work rather than the shape of the tree.
#:
#: Here rather than beside either caller, because there are two now --
#: the request that walks inline and the worker that claims a queued one
#: -- and a person watching should not be told about their library at
#: two different rates depending on which asked.
WALK_EVERY = 250


def scan(conn, root_id: int, root_path, now: float, watch=None) -> ScanResult:
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
    # The folder writes belong inside the same savepoint as the file writes:
    # observe_tree creates, renames and marks folders missing, and a failure
    # during apply_scan would otherwise leave those standing.
    # The walk and every hash happen OUTSIDE the savepoint. SQLite has one
    # write lane per database file, and this used to hold it for the whole
    # survey -- `sha256_of` reading every changed file off the disk -- so a
    # scan of a new root made every other writer wait minutes. The
    # background worker could not even claim a job; it got `database is
    # locked` and reported it as a crash. Reads do not take the lane, so
    # only the second half of this function does.
    held = survey(conn, root_id, root_path, now, watch)
    with _one_write(conn, "scan"):
        observed, hashed = record(conn, root_id, held, now)
        return apply_scan(conn, observed, now, hashed=hashed, roots={root_id})


def scan_all(conn, now: float) -> ScanResult:
    """Every online root, observed first and reconciled once.

    Scanning roots one at a time cannot see a file dragged from one drive to
    another: whichever root is walked first either loses the row to
    `missing_since` or mints a second one, and the authored state stays with
    whichever half the ordering picked. Observing everything before resolving
    anything makes that one move, in either order.

    A root that cannot be read is skipped and marked offline. It is not
    observed as empty, so nothing under it is concluded to be gone.
    """
    # Every root surveyed first, with no lane held, for the same reason
    # `scan` does it: the hashing is the long part and it writes nothing.
    # Which roots are reachable is decided here too, so the write half
    # never stats a disk.
    surveyed: list[tuple[int, Survey | None]] = []
    for root_id, path in conn.execute("SELECT id, path FROM root").fetchall():
        surveyed.append((root_id, survey(conn, root_id, path, now) if os.path.isdir(path) else None))

    with _one_write(conn, "scan_all"):
        observed: dict = {}
        walked: set[int] = set()
        hashed = 0
        for root_id, held in surveyed:
            if held is None:
                conn.execute("UPDATE root SET online = 0 WHERE id = ?", (root_id,))
                continue
            conn.execute("UPDATE root SET online = 1 WHERE id = ?", (root_id,))
            found, read = record(conn, root_id, held, now)
            observed.update(found)
            walked.add(root_id)
            hashed += read
        # The roots actually walked, not all of them: an offline drive was
        # skipped, and reporting what is on it as missing is exactly the
        # reading `online` exists to prevent.
        return apply_scan(conn, observed, now, hashed=hashed, roots=walked)
