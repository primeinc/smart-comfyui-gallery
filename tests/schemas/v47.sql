PRAGMA foreign_keys = ON;

-- ============ addressable entity supertype ============
-- id is the rowid: the on-disk key, used by every FK and join in this schema.
-- uuid is portable identity for export / merge / import across databases,
-- stored as a 16-byte BLOB rather than 36-char text. It is never a join key
-- and never appears in a URL (that is what entity.slug is for).
CREATE TABLE entity (
    -- ================= CONVENTIONS FOR THE WHOLE SCHEMA =================
    -- Deliberately inside a CREATE statement. SQLite keeps only the comments
    -- that sit within one; everything written above a table is discarded, so
    -- a rule stated there is invisible to anyone reading the built database
    -- and survives only in the source file. This is the first table, so this
    -- is the first thing `.schema` prints.
    --
    -- TIME   Every *_at column is UNIX EPOCH SECONDS IN UTC, as a REAL --
    --        EXCEPT the columns that say they hold a human WALL CLOCK:
    --        capture.captured_at (an instant only when capture.tz_offset_min
    --        is present) and every local_at / local_start / local_end, which
    --        are the epoch-shaped spelling of what a clock on the wall read
    --        and are never instants. The two kinds are never compared or
    --        converted into each other by convention -- only by a recorded
    --        offset on the specific row.
    -- SIZE   Bytes.
    -- SCORES det_score and confidence are 0..1, never percentages.
    -- ANGLES Degrees.
    -- BOXES  Fractions of the frame, 0..1. See `region`.
    -- ====================================================================
    -- AUTOINCREMENT, which on a rowid table means "never hand out an id this
    -- table has ever used". Plain `INTEGER PRIMARY KEY` reuses the largest
    -- free rowid, and the minter compounded it by computing `max(id) + 1`
    -- itself: delete the newest entity and the next one created took its id
    -- with a different uuid, so anything holding an id outside this database
    -- -- a thumbnail cache key, an export, a bookmarked address -- silently
    -- resolved to a different picture. SQLite keeps the maximum ever used in
    -- `sqlite_sequence` (refs/sqlite/sqlite/src/insert.c:385-391).
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid BLOB NOT NULL UNIQUE CHECK (length(uuid) = 16),
    kind TEXT NOT NULL CHECK (kind IN
           ('file','folder','person','artifact','prompt','collection','place')),
    slug TEXT NOT NULL,
    UNIQUE (kind, slug)
) STRICT;

-- An entity is what it is for life. Six triggers check that a subtype row
-- sits on an entity of the matching kind, and every one of them fires on
-- INSERT only -- so `UPDATE entity SET kind='folder'` on a file's entity was
-- accepted, and afterwards the file row and its entity disagreed with nothing
-- reporting it. Guarding the supertype closes all six at the source.
CREATE TRIGGER entity_kind_is_permanent BEFORE UPDATE OF kind ON entity
WHEN NEW.kind <> OLD.kind BEGIN
  SELECT RAISE(ABORT,'an entity cannot change kind');
END;

-- composite PK, tiny rows: WITHOUT ROWID per sqlite.org/withoutrowid.html
CREATE TABLE slug_history (
    -- The same list `entity.kind` is held to. Unconstrained, a retirement
    -- could name a kind no entity can ever be, and that address then
    -- resolves to nothing for the rest of the library's life.
    kind       TEXT    NOT NULL CHECK (kind IN
                 ('file','folder','person','artifact','prompt','collection','place')),
    slug       TEXT    NOT NULL,
    entity_id  INTEGER NOT NULL REFERENCES entity(id) ON DELETE CASCADE,
    retired_at REAL    NOT NULL,
    -- retired_at is in the key: a slug released, reissued and released again
    -- must be recordable twice. Resolution order is fixed: a live
    -- entity.slug always wins, history answers only on a miss, most recent
    -- retirement first.
    PRIMARY KEY (kind, slug, retired_at)
) STRICT, WITHOUT ROWID;
CREATE INDEX slug_history_entity ON slug_history(entity_id);

-- ============ physical ============
CREATE TABLE root (
    id            INTEGER PRIMARY KEY,
    -- What the root IS, as against where it currently sits. Written into a
    -- marker file inside the directory, so a library that moved is recognised
    -- as the same library rather than registered a second time.
    --
    -- Without it `path` was a root's only identity: re-registering a moved
    -- library minted a second root while every folder and file stayed under
    -- the first, which `check_roots` then marked offline -- the whole library
    -- stranded behind a root nobody could reach, and no operation anywhere
    -- that could move it back. The one place in this schema where a path was
    -- still identity.
    -- Defaulted by the database, so a root registered by any route has an
    -- identity whether or not the caller thought to mint one.
    uuid          BLOB    NOT NULL UNIQUE DEFAULT (randomblob(16))
                          CHECK (length(uuid) = 16),
    -- Where it is now. Still UNIQUE -- two roots cannot occupy one directory
    -- -- but no longer what the root is.
    path          TEXT    NOT NULL UNIQUE,
    -- 'trash' is a real location, not a state: a deleted file's bytes are
    -- still somewhere, restore is a move, and views exclude the subtree by
    -- ancestry rather than by matching paths against a configured string.
    --
    -- There was a 'mount' beside 'library' and nothing anywhere branched
    -- on the difference -- every read that cared spelled
    -- `kind IN ('library','mount')`. The distinction it reached for,
    -- "this one is not always attached", is `online` below: per-root,
    -- set by probing, and what the whole deletion doctrine rests on. So
    -- it was a choice offered on the add-a-folder form that changed
    -- nothing, made by somebody who had no way to know that.
    kind          TEXT    NOT NULL CHECK (kind IN ('library','trash')),
    -- `online` is the flag the whole deletion doctrine rests on: an unplugged
    -- drive and an emptied folder look identical from a directory listing, so
    -- an unreadable root is marked offline and its files are left alone.
    online        INTEGER NOT NULL DEFAULT 1 CHECK (online IN (0,1)),
    created_at    REAL    NOT NULL
) STRICT;

CREATE TABLE folder (
    id        INTEGER PRIMARY KEY REFERENCES entity(id) ON DELETE CASCADE,
    root_id   INTEGER NOT NULL    REFERENCES root(id)   ON DELETE CASCADE,
    parent_id INTEGER             REFERENCES folder(id) ON DELETE CASCADE,
    name      TEXT    NOT NULL,
    -- Distance from the root, which is itself 0. Maintained by trigger, so
    -- no caller computes it and no two callers can disagree about the base.
    depth     INTEGER NOT NULL,
    -- The filesystem's own id for the directory (NTFS FileID via st_ino),
    -- which survives a rename and a move within the volume while a copy gets
    -- a fresh one. A file proves continuity by its bytes; a directory has no
    -- bytes, so without this a folder rename mints a new folder, the old
    -- entity is orphaned and its URL rots.
    --
    -- A HINT, never identity: it is volume-scoped, lost on copy or restore,
    -- and absent on filesystems that do not report one. Matched only when
    -- present and unique, and name matching still has to work on its own.
    --
    -- TEXT, and the decimal spelling of the identifier, because the value is
    -- OPAQUE: nothing does arithmetic on it, only equality. Since Python 3.12
    -- Windows reports `st_ino` "up to 128 bits, depending on the file system"
    -- (cpython Doc/library/os.rst), SQLite's INTEGER is signed 64-bit
    -- (sqlite/sqlite src/sqliteInt.h:1031 LARGEST_INT64), and binding a
    -- larger one raised `OverflowError: Python int too large to convert to
    -- SQLite INTEGER` -- killing the scan on the first ReFS directory. TEXT
    -- rather than an INTEGER column holding a string: affinity is not a
    -- constraint, and an INTEGER-affinity column silently converts
    -- '340282366920938463463374607431768211455' to the REAL
    -- 3.402823669209385e+38, which is the wrong identity rather than a
    -- refusal. Named `fs_id` and not `inode` because on the stated platform
    -- it is not one.
    fs_id     TEXT,
    -- Set when the directory was not found where it was last seen. Presence
    -- is a state here for the same reason it is one on `file`: without it,
    -- "gone" and "the drive is unplugged" are the same row, and the only way
    -- to make room for a new directory taking an old one's name is to delete
    -- the old one and everything hanging off it.
    missing_since REAL
) STRICT;
-- Partial, as on `file`: a name is exclusive only while the directory is
-- there. Rename `Archive` to `Zoo` and make a new `Archive`, and both rows
-- have to exist at once -- with a total index the scanner had to choose
-- between failing and giving the new directory the old one's identity.
-- NOCASE, as on `file`: the scanner matches directory names the way the
-- stated platform does (db/scan.py ensure_folder), and a binary index
-- permitted live siblings 'Vacation' and 'vacation' the scanner treats as
-- one directory. NOCASE also makes these indexes serve the child listing's
-- name ordering instead of a temp B-tree.
CREATE UNIQUE INDEX folder_root_unique  ON folder(root_id, name COLLATE NOCASE)
    WHERE parent_id IS NULL AND missing_since IS NULL;
CREATE UNIQUE INDEX folder_child_unique ON folder(parent_id, name COLLATE NOCASE)
    WHERE parent_id IS NOT NULL AND missing_since IS NULL;
-- No index on parent_id alone. folder_child_unique leads on it, and although
-- that index is partial the planner does use it for `parent_id = ?`, which is
-- the shape the foreign key runs on every delete -- checked, not assumed.
CREATE UNIQUE INDEX folder_fs_id ON folder(root_id, fs_id) WHERE fs_id IS NOT NULL;

-- Both guards cover INSERT and UPDATE. INSERT-only was bypassable: a row
-- naming itself as its own parent satisfies the foreign key, because the row
-- exists by the time the constraint is checked.
--
-- UNION, never UNION ALL. With UNION ALL a tree that already contains a cycle
-- makes the walk non-terminating -- and in a single-writer WAL database that
-- is not an error, it is a write lock held forever.
CREATE TRIGGER folder_root_consistent_ins BEFORE INSERT ON folder
WHEN NEW.parent_id IS NOT NULL BEGIN
  SELECT RAISE(ABORT,'folder root mismatch')
  WHERE (SELECT root_id FROM folder WHERE id = NEW.parent_id) <> NEW.root_id;
END;

CREATE TRIGGER folder_root_consistent_upd BEFORE UPDATE OF root_id, parent_id ON folder
WHEN NEW.parent_id IS NOT NULL BEGIN
  SELECT RAISE(ABORT,'folder root mismatch')
  WHERE (SELECT root_id FROM folder WHERE id = NEW.parent_id) <> NEW.root_id;
END;

CREATE TRIGGER folder_no_self_parent BEFORE INSERT ON folder
WHEN NEW.parent_id IS NOT NULL AND NEW.parent_id = NEW.id BEGIN
  SELECT RAISE(ABORT,'folder parent cycle');
END;

CREATE TRIGGER folder_no_cycle BEFORE UPDATE OF parent_id ON folder
WHEN NEW.parent_id IS NOT NULL BEGIN
  SELECT RAISE(ABORT,'folder parent cycle') WHERE NEW.id IN (
    WITH RECURSIVE up(id) AS (
      SELECT NEW.parent_id
      UNION SELECT f.parent_id FROM folder f JOIN up ON f.id = up.id
        WHERE f.parent_id IS NOT NULL)
    SELECT id FROM up);
END;

CREATE TABLE file (
    id             INTEGER PRIMARY KEY REFERENCES entity(id) ON DELETE CASCADE,
    folder_id      INTEGER NOT NULL REFERENCES folder(id) ON DELETE CASCADE,
    name           TEXT    NOT NULL,
    kind           TEXT    NOT NULL CHECK (kind IN
                     ('image','animated_image','video','audio','document')),
    size           INTEGER NOT NULL,
    mtime          REAL    NOT NULL,
    -- filesystem birth time where the platform reports it. Distinct from mtime,
    -- which a copy or a sync client rewrites, and from capture.captured_at,
    -- which is when the shutter actually opened.
    btime          REAL,
    -- The filesystem's own id for the file, kept only so a scan can tell
    -- "this path still holds the same file" from "this path now holds a
    -- different one". Size and mtime alone cannot: renaming two same-sized
    -- files onto each other's names changes neither, and the scan then
    -- skipped hashing and left every rating on the path instead of on the
    -- bytes -- the defect this schema exists to remove.
    --
    -- A HINT for change detection, never identity and never a matcher:
    -- content is what proves continuity. Absent where the filesystem
    -- reports none, and different after a copy or a restore.
    --
    -- TEXT for the reason `folder.fs_id` is: the value is opaque, only ever
    -- compared for equality, and on Windows can exceed what an INTEGER holds.
    fs_id          TEXT,
    content_sha256 TEXT,
    -- The pixels actually on disk, not what any recipe asked for; see
    -- `generation.width`, which is the request and may differ. Written by
    -- ingest from the container it already opens to read the metadata, so
    -- they cost nothing extra -- and NULL on a video, whose dimensions need
    -- the same probe `duration` is waiting on.
    --
    -- Both were NULL for everything until the sweep that was supposed to
    -- catch that stopped being a word search: `width` and `height` appear all
    -- over db/ingest.py as attributes of the parsed recipe, so the column
    -- read as produced while the disagreement this schema exists to expose
    -- was unobservable in one direction.
    width          INTEGER,
    height         INTEGER,
    -- Seconds, from the container. NULL on a still picture, which has no
    -- length, and on a video whose container does not state one.
    duration       REAL,
    first_seen_at  REAL    NOT NULL,
    last_seen_at   REAL    NOT NULL,
    -- NULL while the bytes are present. Set when a scan cannot find them, or
    -- when a content match was ambiguous. Deletion is then a deliberate act
    -- rather than a scan side effect: unreachable is not the same as deleted.
    missing_since  REAL
, ingested_sha256 TEXT, ingested_by TEXT) STRICT;
-- file.ingested_by: WHICH READER took it, so improving one re-reads.
--
-- `ingested_sha256` says which BYTES were read, which makes a file stale
-- when its bytes change and never when the READER changes. That is half a
-- freshness rule, and the missing half cost a person a folder of broken
-- pictures: the sniffer called every .m4a a video, ingest wrote the wrong
-- `kind`, and fixing the sniffer could not fix the rows -- nothing recorded
-- that they had been read by the version that was wrong. The only repair
-- was re-reading the entire library by hand, which is a bug turned into a
-- chore.
--
-- So the reader states its own identity (db/ingest.py READER) and bumps it
-- whenever a change alters what it would WRITE for the same bytes. Every
-- file read by an older one is then stale by the ordinary rule, and the
-- ordinary sweep -- the one for what is missing, the one a worker already
-- runs -- repairs the library with nobody asked to do anything.
--
-- NULL means "read before this column existed". Treated as stale, because
-- it is exactly the population that may carry what an old reader decided.
--
-- file.ingested_sha256: the bytes the last metadata read was taken from
-- (db/ingest.py one). NULL until ingest reads the file; unequal to
-- content_sha256 once a scan records new bytes. The ingest sweep queues
-- only files where it is NULL or stale -- the record of the read, so
-- "never read" and "read, found no metadata" are different rows. Spelled
-- exactly as SQLite stores an ALTER TABLE ADD COLUMN on the v26 text --
-- a newline, then `, ingested_sha256 TEXT)` -- so a migrated file's DDL
-- reads equal to a fresh build's (db/build.py drift).
-- NOCASE: the stated platform is Windows, where 'A.png' and 'a.png' are one
-- file. Case-sensitive uniqueness let a case-only rename create a second row,
-- one of which is permanently missing.
--
-- Partial on missing_since, because a path is exclusive only while the bytes
-- are there. A missing row keeps its last known path so the app can say where
-- the file used to be; without the WHERE clause that stale path blocked a new
-- file from taking the name, and a scan of a directory whose contents had been
-- replaced failed outright.
CREATE UNIQUE INDEX file_in_folder ON file(folder_id, name COLLATE NOCASE)
    WHERE missing_since IS NULL;
-- "newest first" is the default view; without this every page load sorts the
-- whole table in a temp B-tree. Partial, because the default view is live files.
-- Carries the id tiebreak in the SAME direction, because the ResultSet's
-- ordering contract is (mtime DESC, id DESC) / reversed -- matching
-- file_in_folder_by_time's -- and the index implements the contract, never
-- the other way around: on (mtime DESC) alone, equal keys sit rowid ASC and
-- the walk fell back to a whole-membership sort.
CREATE INDEX file_recent ON file(mtime DESC, id DESC) WHERE missing_since IS NULL;
-- The lightbox's next and previous. Without it "the picture before this one in
-- this folder" sorted the whole folder to return one row, which on the largest
-- folder in a real library is 50,007 rows sorted per arrow-key press. The
-- tiebreak is `id` rather than the slug because the slug lives on `entity` and
-- an index cannot span two tables.
CREATE INDEX file_in_folder_by_time ON file(folder_id, mtime, id)
    WHERE missing_since IS NULL;
CREATE INDEX file_added  ON file(first_seen_at DESC) WHERE missing_since IS NULL;
CREATE INDEX file_sha  ON file(content_sha256);
CREATE INDEX file_kind ON file(kind);

-- ============ entity subtypes (PK is entity.id) ============
CREATE TABLE person (
    id         INTEGER PRIMARY KEY REFERENCES entity(id) ON DELETE CASCADE,
    name       TEXT,
    created_at REAL NOT NULL
, exemplar_file_id INTEGER REFERENCES file(id) ON DELETE SET NULL) STRICT;
-- person.exemplar_file_id: which picture their face is taken FROM.
--
-- Spelled as ALTER TABLE leaves it, the convention this file holds for
-- every added column (see file.ingested_sha256): SQLite stores the
-- literal statement text and `build.drift` compares it.
--
-- A FILE and not a face, which is the whole point. `/avatar/<slug>`
-- crops the highest-confidence detection in the primary run, and that
-- is usually right and sometimes a blurred profile in a crowd. But a
-- face is a DERIVED row: `derived.drop_all` deletes every
-- `derived_face_instance` and a rebuild mints new ones, so a remembered
-- face id would be a pointer at something the next re-detect destroys --
-- exactly the mistake `person_assertion` exists to avoid by naming a
-- file and a region rather than a cluster.
--
-- Naming the picture survives all of it. After any rebuild the avatar
-- takes whichever face of theirs is in that picture, and if they are no
-- longer found there it falls back to the confident one rather than
-- showing nothing.
--
-- SET NULL, because deleting the picture is not a statement about the
-- person: they simply go back to the automatic choice.
--
-- And indexed, for the reason `job_target` is: SQLite runs
-- `SELECT 1 FROM child WHERE child_key = ?` against every child table
-- when a parent row goes, so without this every file deletion scans
-- every person.
CREATE INDEX person_exemplar ON person(exemplar_file_id);

-- One table, not one per kind. A checkpoint, a LoRA, a VAE, a ControlNet and a
-- camera body are the same shape: a named thing that exists independently of
-- any image, that many files reference, and that deserves its own page.
-- Adding a kind costs a CHECK entry, not a table and not a migration.
CREATE TABLE artifact (
    id            INTEGER PRIMARY KEY REFERENCES entity(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL CHECK (kind IN (
                    -- weights that participate in generation
                    'checkpoint','lora','vae','controlnet','upscaler','workflow',
                    'embedding','hypernetwork','ip_adapter','text_encoder','unet',
                    -- physical capture equipment, from EXIF
                    'camera','lens',
                    -- the software that produced or processed the file
                    'node_pack','application')),
    name          TEXT NOT NULL,             -- as the metadata spelled it
    -- The working identity. Almost all generation metadata is a string
    -- scraped from a PNG tEXt chunk or an EXIF UserComment, so content_hash is
    -- NULL for most rows and the *name* is what actually dedupes. name_key is
    -- that name put through one normalization rule -- the same rule the search
    -- box uses, because a model found by its own name and a model deduped on
    -- ingest must agree or the library grows a row per mention.
    name_key      TEXT NOT NULL,
    -- SD1.5 / SDXL / Flux, or NULL for anything that is not weights.
    -- NOTHING WRITES THIS YET: no adapter reports an architecture, so every
    -- row is NULL and a facet built on it would return an empty library.
    -- Reading it from safetensors headers is the work that makes it real.
    architecture  TEXT,
    -- Two different facts, deliberately not one column.
    -- content_sha256: computed here, from a file actually in hand. Identity.
    -- quoted_hash:    what the metadata claimed (A1111 writes "Model hash:",
    --                 AutoV1/AutoV2, not sha256). Evidence, never proof, and
    --                 never unique -- two tools may quote colliding short
    --                 hashes and neither is verification.
    content_sha256 TEXT,
    quoted_hash    TEXT,
    first_seen_at  REAL NOT NULL,
    -- A camera body has no bytes. NULL there means "not applicable" and must
    -- never be confused with "not yet known", which is what NULL means for a
    -- checkpoint whose file is not on this machine.
    CHECK (kind NOT IN ('camera','lens','application')
           OR (content_sha256 IS NULL AND quoted_hash IS NULL))
) STRICT;
-- Identity is (kind, name) *while the content is unknown*, and (kind, name,
-- content) once it is known. Two different checkpoints that happen to be
-- called model.safetensors are two artifacts and both get rows and addresses;
-- what stays impossible is two rows for one name with nothing to tell them
-- apart, which is the duplicate-per-mention failure.
CREATE UNIQUE INDEX artifact_ident
    ON artifact(kind, name_key, IFNULL(content_sha256, ''));
-- Identity once verified: content wins, and two artifacts cannot share it.
-- Discovering that two name-rows hash the same is a merge, and a merge is a
-- deliberate operation -- never a silent overwrite.
CREATE UNIQUE INDEX artifact_sha ON artifact(content_sha256) WHERE content_sha256 IS NOT NULL;
-- Deliberately NOT unique: a quoted short hash is a lead for resolution, not
-- an identity. Indexed so "which artifacts did some tool call a1b2c3d4" is
-- answerable.
CREATE INDEX artifact_quoted ON artifact(quoted_hash) WHERE quoted_hash IS NOT NULL;
-- No index on kind alone: artifact_ident already leads on it, so a second one
-- is write cost for a read the first already serves.

-- A prompt earns a table because it recurs across files *by design*: reusing
-- one prompt across checkpoints is a documented workflow, which is why the old
-- app already carried a prompt_hash. A caption does not recur and does not
-- belong here.
--
-- The CHECK is a scar. The old app synthesized md5(workflow_hash + '_prompt')
-- for files with no prompt, which grouped files that shared nothing, and
-- clear_synthetic_prompt_hashes exists to undo it. An absent prompt must have
-- no identity at all rather than a manufactured one.
CREATE TABLE prompt (
    id         INTEGER PRIMARY KEY REFERENCES entity(id) ON DELETE CASCADE,
    -- trim() with one argument strips the space character only, so newline,
    -- tab, U+00A0 and U+3000 all slipped through as "non-empty" -- a
    -- manufactured identity for an absent prompt, which is what this CHECK
    -- exists to prevent.
    text       TEXT NOT NULL
                 CHECK (length(trim(text, char(32,9,10,13,11,12,160,12288))) > 0),
    text_hash  TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL
) STRICT;

-- A static album lists its members; a smart one derives them from a rule.
-- Same shape, one kind apart -- so one table, not two.
CREATE TABLE collection (
    id          INTEGER PRIMARY KEY REFERENCES entity(id) ON DELETE CASCADE,
    -- RESTRICT: authored children have independent addresses, and deleting
    -- an organizer must never silently take a subtree with it.
    parent_id   INTEGER REFERENCES collection(id) ON DELETE RESTRICT,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('album','flag','smart')),
    color       TEXT,
    description TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    created_by  INTEGER REFERENCES user(id) ON DELETE SET NULL,
    updated_by  INTEGER REFERENCES user(id) ON DELETE SET NULL,
    -- Lifecycle, never deletion: archiving keeps the address, the members,
    -- the children and the rule; it changes discoverability only. NULL
    -- means active.
    archived_at REAL,
    -- Optimistic concurrency over the DEFINITION -- name, kind, color,
    -- description, parent, archive state, rule. Membership never bumps it:
    -- filing a picture does not invalidate an open description editor.
    definition_rev INTEGER NOT NULL DEFAULT 1 CHECK (definition_rev > 0)
) STRICT;

-- How a SMART collection's membership is derived. The collection row says
-- what the entity is; this row says how its members are decided -- a typed,
-- versioned rule the ResultSet evaluates. Nothing here is ever executable:
-- `source_text` and `legacy_sql_text` are provenance a human reads, and a
-- row whose rule_json is NULL is an UNEVALUATED collection, never an empty
-- one. Entity references inside rule_json are entity.uuid -- an address
-- spelling can be renamed and eventually reused; a saved rule must mean the
-- entity that was selected.
CREATE TABLE collection_rule (
    collection_id INTEGER PRIMARY KEY REFERENCES collection(id) ON DELETE CASCADE,
    rule_version  INTEGER,
    rule_json     TEXT CHECK (rule_json IS NULL OR json_valid(rule_json)),
    -- WHOSE judgement the rule's authored facets (favorite, rating_min)
    -- mean -- pinned at creation, never the viewer. RESTRICT: nulling it
    -- would silently change what the rule answers.
    actor_id      INTEGER REFERENCES user(id) ON DELETE RESTRICT,
    source_text     TEXT,
    legacy_sql_text TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    CHECK ((rule_json IS NULL AND rule_version IS NULL) OR (rule_json IS NOT NULL AND rule_version IS NOT NULL))
) STRICT;
CREATE INDEX collection_rule_actor ON collection_rule(actor_id);
-- Leads on parent_id for the foreign key's delete shape; carries the name
-- NOCASE so a collection's child listing is an index walk, not a sort.
CREATE INDEX collection_parent ON collection(parent_id, name COLLATE NOCASE);
-- Authorship leads for the user FK's delete shape: without them,
-- removing a user scans every collection row, twice.
CREATE INDEX collection_created_by ON collection(created_by);
CREATE INDEX collection_updated_by ON collection(updated_by);

CREATE TRIGGER collection_no_self_parent BEFORE INSERT ON collection
WHEN NEW.parent_id IS NOT NULL AND NEW.parent_id = NEW.id BEGIN
  SELECT RAISE(ABORT,'collection parent cycle');
END;

CREATE TRIGGER collection_no_cycle BEFORE UPDATE OF parent_id ON collection
WHEN NEW.parent_id IS NOT NULL BEGIN
  SELECT RAISE(ABORT,'collection parent cycle') WHERE NEW.id IN (
    WITH RECURSIVE up(id) AS (
      SELECT NEW.parent_id
      UNION SELECT a.parent_id FROM collection a JOIN up ON a.id = up.id
        WHERE a.parent_id IS NOT NULL)
    SELECT id FROM up);
END;

CREATE TABLE user (
    id            INTEGER PRIMARY KEY,
    username      TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('ADMIN','USER')),
    created_at    REAL NOT NULL
) STRICT;

-- ============ where in the picture ============
-- One place a rectangle is defined, because four tables were each carrying
-- their own bbox_x/y/w/h and one of them also carried a `localizable` flag
-- and a five-clause CHECK whose only job was policing those columns.
--
-- NORMALIZED, 0..1, never pixels. A box in pixels is a box against one
-- particular rendering: draw it on a thumbnail, a rotated frame or a
-- re-encoded proxy and it lands somewhere else. The old geometry columns
-- declared no unit at all, so every producer picked its own.
--
-- A mask is bytes, and bytes live in `blob`. It was a filesystem path --
-- identity derived from location, in the schema written to delete exactly
-- that, where a moved cache directory silently voided every mask in it.
CREATE TABLE region (
    id        INTEGER PRIMARY KEY,
    -- FRACTIONS OF THE FRAME, 0..1. Never pixels: a box in pixels is a box
    -- against one particular rendering, and the same numbers on a thumbnail,
    -- a rotated frame or a re-encoded proxy point somewhere else.
    x         REAL NOT NULL,
    y         REAL NOT NULL,
    w         REAL NOT NULL,
    h         REAL NOT NULL,
    -- A mask is bytes, in the blob store. It was a filesystem path, which is
    -- identity derived from location in the schema written to delete that.
    mask_hash TEXT REFERENCES blob(hash) ON DELETE SET NULL,
    CHECK (w > 0 AND h > 0),
    -- a hair over 1 absorbs float error from a pixel->fraction conversion
    -- without admitting a box that is genuinely outside the frame
    CHECK (x >= 0 AND y >= 0 AND x + w <= 1.001 AND y + h <= 1.001)
) STRICT;
CREATE INDEX region_mask ON region(mask_hash) WHERE mask_hash IS NOT NULL;

-- ============ near-duplicate groups (perceptual) ============
-- The same PICTURE, whatever became of its bytes: files whose phash64
-- agrees within the run's threshold, grouped, with one member marked the
-- group's best face forward. Wholesale replaced by every dupes job --
-- rebuilt from derived_file_hash alone -- like every derived answer.
CREATE TABLE derived_dupe_group (
    file_id     INTEGER PRIMARY KEY REFERENCES file(id) ON DELETE CASCADE,
    -- the group's seed: its lowest member id. Deleting the seed file
    -- cascades the whole group away; the next job rebuilds what remains.
    group_id    INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    -- hamming bits from this member's phash64 to the BEST member's -- the
    -- canonical image every member is a duplicate OF. Never to an arbitrary
    -- neighbour: A~B and B~C does not make A a duplicate of C, and a chain
    -- is exactly how a "duplicate group" collects two pictures its own
    -- verifier says are different.
    distance    INTEGER NOT NULL CHECK (distance BETWEEN 0 AND 64),
    threshold   INTEGER NOT NULL CHECK (threshold BETWEEN 0 AND 64),
    is_best     INTEGER NOT NULL DEFAULT 0 CHECK (is_best IN (0, 1)),
    -- 1: this member's dHash agreed with the best member's -- two
    -- independent fingerprints both said duplicate. 0: pHash alone said so
    -- (a dHash was missing, or verification was off). A verified duplicate
    -- and an unverified candidate are different claims, and a page that
    -- cannot tell them apart flattens the difference into false confidence.
    verified    INTEGER NOT NULL DEFAULT 0 CHECK (verified IN (0, 1)),
    computed_at REAL NOT NULL
) STRICT;
CREATE INDEX derived_dupe_group_group ON derived_dupe_group(group_id);
CREATE UNIQUE INDEX derived_dupe_group_best ON derived_dupe_group(group_id) WHERE is_best = 1;

-- ============ evidence locator (video/document faces) ============
-- Named derived_: a sampling policy produced these rows, so "drop the derived
-- namespace and re-index" takes them -- EXCEPT the rows a person_assertion
-- pins, which drop_all keeps the way it keeps asserted regions: the human's
-- claim names a moment, and deleting the row erases which one. Instances
-- citing dead policies still go, because the instances themselves go; the
-- seeder compares moments by (file, kind, offset, page), never by row id, so a
-- rebuild under a new policy token re-attaches the same frames.
CREATE TABLE derived_media_sample (
    id         INTEGER PRIMARY KEY,
    file_id    INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    -- 'frame' comes from db/sample.py, at a cadence, off the probed duration.
    -- 'page' comes from db/probe.py, one per page of a document. Both exist
    -- so a claim can say which moment or which page it was looking at: a
    -- caption of a video is a caption of a frame, and OCR from page nine is
    -- not OCR of the file.
    kind       TEXT NOT NULL CHECK (kind IN ('still','frame','page')),
    offset_ms  INTEGER,
    page_index INTEGER,
    -- How this sample was chosen, as a canonical token ('every-2s',
    -- 'keyframes', 'scene-cuts'). Deliberately not a CHECK, because a
    -- sampler added later must not require a schema change -- but it has to
    -- be a token and not a sentence, or the same policy spelled three ways
    -- cannot be grouped and a re-run cannot tell it already did this.
    policy     TEXT NOT NULL CHECK (policy = lower(policy) AND policy NOT LIKE '% %')
) STRICT;
CREATE UNIQUE INDEX derived_media_sample_pos
    ON derived_media_sample(file_id, kind, IFNULL(offset_ms,-1), IFNULL(page_index,-1), policy);

-- ============ relations ============
-- One join for every artifact kind. `ordinal` preserves stack order, which
-- LoRA recipes need and a checkpoint simply leaves at 0. The two weights cover
-- the model/clip pair LoRAs carry; other kinds leave them null.
CREATE TABLE file_artifact (
    file_id      INTEGER NOT NULL REFERENCES file(id)     ON DELETE CASCADE,
    ordinal      INTEGER NOT NULL DEFAULT 0,
    artifact_id  INTEGER NOT NULL REFERENCES artifact(id) ON DELETE CASCADE,
    role         TEXT NOT NULL CHECK (role IN
                   ('checkpoint','refiner','lora','vae','controlnet','upscaler',
                    'embedding','hypernetwork','ip_adapter','text_encoder','unet',
                    'captured_with','mounted_lens','produced_by')),
    -- Multipliers, not percentages, and not normalised to anything: 1.0 is
    -- full strength and values above it are legal and common. `model_weight`
    -- is the unet multiplier, `clip_weight` the text-encoder one; a tag that
    -- gives only one number sets both to it.
    model_weight REAL,
    clip_weight  REAL,
    PRIMARY KEY (file_id, role, ordinal)
) STRICT, WITHOUT ROWID;
CREATE INDEX file_artifact_artifact ON file_artifact(artifact_id);
CREATE INDEX file_artifact_role     ON file_artifact(role);

-- "backend X infers this person appears in this file". Model-versioned, so
-- it is derived by construction and belongs in the namespace a rebuild drops.
-- Keeping it on the durable side meant a rebuild left rows keyed on model
-- versions that no longer produced anything.
-- Keyed on the RUN, not on the embedder. Who is in a picture is the output
-- of one clustering, and two clusterings disagree -- that disagreement is
-- the whole reason for running both. Keyed on (model, version) alone the
-- second run overwrote the first's attributions, so the cluster tables could
-- hold two answers while this table could hold one, and the People page read
-- whichever wrote last.
CREATE TABLE derived_file_person (
    file_id       INTEGER NOT NULL REFERENCES file(id)   ON DELETE CASCADE,
    person_id     INTEGER NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    run_id        INTEGER NOT NULL REFERENCES derived_face_run(id) ON DELETE CASCADE,
    model_id      TEXT NOT NULL,
    model_version TEXT NOT NULL,
    face_count    INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (file_id, person_id, run_id)
) STRICT, WITHOUT ROWID;
CREATE INDEX derived_file_person_person ON derived_file_person(person_id);
CREATE INDEX derived_file_person_run    ON derived_file_person(run_id);

CREATE TABLE collection_file (
    collection_id INTEGER NOT NULL REFERENCES collection(id) ON DELETE CASCADE,
    file_id  INTEGER NOT NULL REFERENCES file(id)  ON DELETE CASCADE,
    added_at REAL NOT NULL,
    PRIMARY KEY (collection_id, file_id)
) STRICT, WITHOUT ROWID;
CREATE INDEX collection_file_file ON collection_file(file_id);

-- ============ jobs ============
CREATE TABLE job (
    id               INTEGER PRIMARY KEY,
    -- Constrained like every other `kind` here. A typo is otherwise a job
    -- that queues successfully and no worker ever claims, because claim()
    -- filters on the kinds it knows -- so it waits forever and looks fine.
    -- 'walk' is the directory walk itself, which for a long time was the
    -- one expensive thing in this application that was NOT a job: the
    -- scan route did it inline, so a person who asked to scan 80,000
    -- files watched a request hang for a minute with nothing to look at,
    -- while every cheaper sweep after it reported progress. 'scan' was
    -- already taken -- it is the metadata read (db/runner.py _ingest_item)
    -- -- so the walk gets its own word rather than borrowing one.
    kind             TEXT NOT NULL CHECK (kind IN
                       ('walk','scan','hash','embed','detect_faces','cluster_faces',
                        'sample_frames','annotate','remix','zip','context','events',
                        'story_plan','embed_prompts')),
    target_id        INTEGER REFERENCES entity(id) ON DELETE SET NULL,
    state            TEXT NOT NULL CHECK (state IN
                       ('queued','running','done','failed','cancelled')),
    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0,1)),
    payload          TEXT,
    total            INTEGER,
    done_count       INTEGER NOT NULL DEFAULT 0,
    checkpoint       TEXT,
    attempt          INTEGER NOT NULL DEFAULT 0,
    -- a lease nobody owns cannot fence anyone: the reclaiming worker must be
    -- able to prove it holds the job, and the evicted one must be rejected.
    owner            TEXT,
    fence            INTEGER NOT NULL DEFAULT 0,
    lease_until      REAL,
    heartbeat_at     REAL,
    error            TEXT,
    -- No external_ref here. `derivation_intent` already carries the
    -- generator's own id, UNIQUE, and having it on both meant two rows could
    -- claim the same external job and disagree about which one owned it.
    created_at       REAL NOT NULL,
    started_at       REAL,
    finished_at      REAL
, collection TEXT, after_id INTEGER REFERENCES job(id) ON DELETE SET NULL) STRICT;
-- job.collection / job.after_id: a job that is a STEP of something.
--
-- Spelled as ALTER TABLE leaves them, which is the convention this file
-- holds for every added column (see file.ingested_sha256 above): SQLite
-- stores the literal statement text, `build.drift` compares it, and a
-- migrated database must be indistinguishable from a fresh build down to
-- the words. The documentation goes here rather than inside the parens
-- because ALTER cannot put it there.
--
-- Adding a root meant pressing eight buttons in an order only the
-- application knew: scan, ingest, context, events, embed, detect_faces,
-- cluster_faces, annotate. The order is REAL -- cluster_faces over an
-- unembedded library is a job that honestly settles `done` having
-- clustered nothing -- and the application knew it and made a person
-- re-derive it every time.
--
-- `after_id` is that order, recorded where the claim can read it:
-- db/jobs.py `claim` will not take a job whose predecessor has not
-- settled `done`. SET NULL rather than CASCADE, because an aged-out
-- predecessor must not delete the step that ran after it, and a NULL
-- edge reads as "nothing gates this" -- true, once it is gone.
--
-- `collection` is the name the steps share. A free name rather than a
-- table: a collection has no identity of its own to keep, it IS the
-- jobs, and a row per group would need deleting when the last one aged
-- out. The console groups on it, and a schedule NAMES it -- "every
-- night, catch up" points at a collection, where naming individual
-- kinds would mean re-deriving the order at 3am.
CREATE INDEX job_state ON job(state);
-- The claim reads it on every attempt, and a cascade-cancel walks it
-- backwards from a failed step to everything that depended on it.
CREATE INDEX job_after ON job(after_id);
CREATE INDEX job_collection ON job(collection) WHERE collection IS NOT NULL;
-- Not for a query anyone writes: SQLite runs `SELECT 1 FROM child WHERE
-- child_key = ?` against every child table when a parent row is deleted, and
-- without an index that is a full scan per delete. Its own shell ships
-- `.lint fkey-indexes` to find these (src/shell.c.in:5981-6014).
CREATE INDEX job_target ON job(target_id);

CREATE TABLE job_item (
    job_id  INTEGER NOT NULL REFERENCES job(id) ON DELETE CASCADE,
    item_id INTEGER NOT NULL,
    state   TEXT NOT NULL CHECK (state IN ('pending','done','failed')),
    error   TEXT,
    PRIMARY KEY (job_id, item_id)
) STRICT, WITHOUT ROWID;

-- ============ the operational ledger ============
-- The job row is current truth; this is historical observation; the
-- channel is transport. One row per operationally meaningful transition,
-- typed, append-only. AUTOINCREMENT: the id is the ORDER a subscriber
-- resumes from and a gap in the ids a client holds is a gap it can name,
-- so an id is never reused. Never sampled, never compacted -- a 22,000-file
-- sweep leaves 44,000 rows and the console pages them.
CREATE TABLE job_event (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id   INTEGER NOT NULL REFERENCES job(id) ON DELETE CASCADE,
    at       REAL NOT NULL,
    -- the vocabulary db/ledger.py TYPES spells; a typo is a refused insert,
    -- never an event no renderer knows
    type     TEXT NOT NULL CHECK (type IN
               ('job.submitted','job.claimed','job.reclaimed','job.paused',
                'job.cancel_requested','job.cancelled','job.done','job.failed',
                'item.started','item.done','item.failed','item.observed',
                'phase.started','phase.progress','phase.finished',
                'checkpoint.changed','worker.turn_failed')),
    item_id  INTEGER,
    phase    TEXT,
    severity TEXT NOT NULL DEFAULT 'info' CHECK (severity IN ('info','warning','error')),
    message  TEXT,
    data     TEXT CHECK (data IS NULL OR json_valid(data))
) STRICT;
CREATE INDEX job_event_job ON job_event(job_id, id);

-- ============ lineage: intent then realized edge ============
CREATE TABLE derivation_intent (
    id           INTEGER PRIMARY KEY,
    parent_id    INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    kind         TEXT NOT NULL CHECK (kind IN
                   ('remix','upscale','inpaint','img2img','video')),
    external_ref TEXT NOT NULL UNIQUE,
    job_id       INTEGER REFERENCES job(id) ON DELETE SET NULL,
    created_at   REAL NOT NULL
) STRICT;

CREATE TABLE file_derivation (
    id         INTEGER PRIMARY KEY,
    intent_id  INTEGER REFERENCES derivation_intent(id) ON DELETE SET NULL,
    parent_id  INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    child_id   INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL CHECK (kind IN
                 ('remix','upscale','inpaint','img2img','video')),
    created_at REAL NOT NULL,
    UNIQUE (parent_id, child_id, kind),
    CHECK (parent_id <> child_id)
) STRICT;
CREATE INDEX derivation_child ON file_derivation(child_id);
CREATE INDEX derivation_of_intent ON file_derivation(intent_id);
CREATE INDEX intent_parent ON derivation_intent(parent_id);
CREATE INDEX intent_job    ON derivation_intent(job_id);

-- ============ files that accompany other files ============
-- Distinct from file_derivation: a companion PNG is not *derived from* the
-- video it sits beside, it *accompanies* it and carries metadata the video
-- format cannot hold. The app already relies on this
-- (/api/remix/companion/<file_id>) with nowhere to record it. RAW+JPEG pairs
-- and .xmp sidecars are the same shape.
CREATE TABLE file_relation (
    file_id      INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    related_id   INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    kind         TEXT NOT NULL CHECK (kind IN
                   ('companion','sidecar','raw_pair','proxy','subtitle','note')),
    created_at   REAL NOT NULL,
    PRIMARY KEY (file_id, related_id, kind),
    CHECK (file_id <> related_id)
) STRICT, WITHOUT ROWID;
CREATE INDEX file_relation_related ON file_relation(related_id);

-- ============ generation ============
CREATE TABLE generation (
    file_id     INTEGER PRIMARY KEY REFERENCES file(id) ON DELETE CASCADE,
    tool        TEXT NOT NULL,
    detection   TEXT NOT NULL CHECK (detection IN
                  ('graph','marker','heuristic','stealth')),
    -- the workflow is an artifact like any other weights file: nameable,
    -- content-hashed, and referenced by many images
    workflow_id INTEGER REFERENCES artifact(id) ON DELETE SET NULL,
    seed INTEGER, steps INTEGER, cfg REAL, denoise REAL, clip_skip INTEGER,
    sampler TEXT, scheduler TEXT,
    -- What the recipe ASKED FOR, which is not what the file is: `file.width`
    -- is the pixels on disk. An upscale, a crop or a re-encode makes them
    -- differ, and that disagreement is the interesting part -- but only if
    -- something says which column means which, so neither is read as the
    -- other's stale copy.
    width INTEGER, height INTEGER,
    -- Which adapter and version produced this row, so improving a parser is a
    -- re-parse of the database rather than a re-read of every file on disk.
    parser        TEXT NOT NULL,
    parsed_at     REAL NOT NULL
    -- Deliberately NO `extra` JSON column. The old app dumped every
    -- unrecognised key into one, where nothing could query it. Every key a
    -- tool emits is parsed into file_param, registered in param_key, and
    -- indexed. A field that exists only inside a blob is not captured.
) STRICT;
CREATE INDEX generation_workflow ON generation(workflow_id);
CREATE INDEX generation_seed     ON generation(seed);

-- Which prompt played which ROLE for one generation: `effective` is the
-- text the generator ran, `negative` the negative prompt, `original`
-- and `original_negative` the texts as written before the tool
-- expanded wildcards (SwarmUI records `original_<param>` only when
-- processing changed the text -- src/Text2Image/T2IParamInput.cs:592 --
-- and an absent one is recorded here as NOTHING, never as "same"),
-- `unsampler` Swarm's third prompt field. One relation, one source of
-- truth -- the prompt columns that used to sit on `generation` were a
-- second copy of two of these roles with no room for the rest. The
-- generator's own parameter stays in file_param as the evidence a role
-- was read from.
CREATE TABLE generation_prompt (
    file_id   INTEGER NOT NULL REFERENCES generation(file_id) ON DELETE CASCADE,
    role      TEXT NOT NULL CHECK (role IN
                ('effective','original','negative','original_negative','unsampler')),
    prompt_id INTEGER NOT NULL REFERENCES prompt(id) ON DELETE CASCADE,
    PRIMARY KEY (file_id, role)
) STRICT, WITHOUT ROWID;
CREATE INDEX generation_prompt_prompt ON generation_prompt(prompt_id, role);

-- ============ capture: EXIF, for files a camera made ============
-- A photograph is not "generated". It has its own origin story, and the app
-- currently reads EXIF only as a smuggling channel for workflow JSON
-- (metaparse/containers.py:17), so none of this was recoverable before.
-- Camera and lens are artifacts, not strings, so /camera/<slug> is a page.
CREATE TABLE capture (
    file_id       INTEGER PRIMARY KEY REFERENCES file(id) ON DELETE CASCADE,
    captured_at   REAL,          -- EXIF DateTimeOriginal; NOT file mtime
    tz_offset_min INTEGER,       -- OffsetTimeOriginal, so "the viewer's day" is answerable
    -- 0 is "not recorded" (a clip's thumbnail writes it), never a sensitivity
    iso           INTEGER CHECK (iso IS NULL OR iso > 0),
    f_number      REAL,
    exposure_time REAL,          -- seconds
    focal_length  REAL,          -- mm
    focal_35mm    REAL,
    orientation   INTEGER,
    gps_lat       REAL,
    gps_lon       REAL,
    gps_alt       REAL,
    -- the camera's finer clock and its own identity: SubSecTimeOriginal as
    -- milliseconds, BodySerialNumber, and the clock's zone from the maker
    -- note when OffsetTimeOriginal is absent (Canon TimeInfo)
    subsec_ms           INTEGER CHECK (subsec_ms IS NULL OR subsec_ms BETWEEN 0 AND 999),
    body_serial         TEXT,
    maker_tz_offset_min INTEGER,
    parsed_at     REAL NOT NULL
) STRICT;
CREATE INDEX capture_when ON capture(captured_at);
CREATE INDEX capture_body ON capture(body_serial) WHERE body_serial IS NOT NULL;
CREATE INDEX capture_where ON capture(gps_lat, gps_lon) WHERE gps_lat IS NOT NULL;

-- ============ provenance for re-parsing -- NOT the storage of record ========
-- Every field is parsed out into file_param, registered in param_key and
-- indexed. This table exists only so a *better* parser can re-read what the
-- file actually said without touching the disk again; it is never where a
-- value lives. `parsed_by IS NULL` is the backlog of carriers no adapter
-- understood yet -- the one case where a payload is held and not yet parsed.
--
-- Deduplicated by content hash because a workflow graph recurs across files
-- exactly as a prompt does: thirty seed variations of one pipeline carry one
-- identical 100 KB JSON blob between them.
CREATE TABLE blob (
    hash        TEXT PRIMARY KEY,      -- sha256 of the payload as found
    payload     TEXT,                  -- text carriers
    payload_bin BLOB,                  -- carriers that are not text
    byte_len    INTEGER NOT NULL,
    CHECK (payload IS NOT NULL OR payload_bin IS NOT NULL)
) STRICT;

CREATE TABLE file_blob (
    file_id   INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    carrier   TEXT NOT NULL CHECK (carrier IN (
                'png_text','png_ztxt','png_itxt','exif','xmp','iptc',
                'jpeg_comment','stealth_lsb','id3','container','sidecar')),
    slot      TEXT NOT NULL,           -- chunk keyword, EXIF tag, sidecar name
    blob_hash TEXT NOT NULL REFERENCES blob(hash) ON DELETE RESTRICT,
    -- The adapter and version that consumed this payload. NULL means nothing
    -- understood it yet, which makes unparsed metadata a queryable backlog
    -- rather than a silent loss.
    parsed_by TEXT,
    seen_at   REAL NOT NULL,
    PRIMARY KEY (file_id, carrier, slot)
) STRICT, WITHOUT ROWID;
CREATE INDEX file_blob_hash     ON file_blob(blob_hash);
CREATE INDEX file_blob_unparsed ON file_blob(carrier) WHERE parsed_by IS NULL;

-- ============ the long tail, queryable ============
-- Every key any tool, camera, container or filesystem emits that is not worth
-- a typed column. Indexed on key, so a new field becomes searchable without a
-- migration -- which is the difference between this and a JSON blob nobody can
-- query. Promote a key to a real column only when it earns a facet.
CREATE TABLE file_param (
    file_id    INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    source     TEXT NOT NULL CHECK (source IN
                 ('exif','xmp','iptc','id3','container','filesystem',
                  'generation','sidecar')),
    key        TEXT NOT NULL,
    value_text TEXT,
    value_num  REAL,             -- populated when the value parses as a number
    PRIMARY KEY (file_id, source, key)
-- deliberately NOT WITHOUT ROWID: this table absorbs the long tail, whose
-- values run to multiple KB, and that optimization is for small rows.
) STRICT;
CREATE INDEX file_param_key     ON file_param(key);
CREATE INDEX file_param_key_num ON file_param(key, value_num) WHERE value_num IS NOT NULL;

-- Every key ever encountered, so the tail is discoverable rather than merely
-- present. Without this, "what fields does my library contain?" is a full
-- scan, and /ways has nothing to render.
CREATE TABLE param_key (
    -- The same list `file_param.source` is held to. The registry is fed by a
    -- trigger today, so the two cannot diverge in practice -- but one list
    -- enforced and one not is a difference waiting for the first direct
    -- write, and the registry is what the facet UI is generated from.
    source        TEXT NOT NULL CHECK (source IN
                    ('exif','xmp','iptc','id3','container','filesystem',
                     'generation','sidecar')),
    key           TEXT NOT NULL,
    value_kind    TEXT NOT NULL DEFAULT 'text'
                    CHECK (value_kind IN ('text','number','mixed')),
    occurrences   INTEGER NOT NULL DEFAULT 0,
    first_seen_at REAL NOT NULL,
    last_seen_at  REAL NOT NULL,
    PRIMARY KEY (source, key)
) STRICT, WITHOUT ROWID;
CREATE INDEX param_key_key ON param_key(key);

-- Maintained by the database, not by remembering -- and by RECOMPUTING rather
-- than incrementing. INSERT OR REPLACE deletes the old row without firing a
-- DELETE trigger (recursive_triggers is off by default), so a counter that is
-- only ever incremented drifts on every re-parse. Re-parsing is the normal
-- case here: improving a parser is a re-parse of the database.
-- NOTHING MAY WRITE THIS TABLE WITH INSERT OR REPLACE. Use
-- `ON CONFLICT(file_id, source, key) DO UPDATE`.
--
-- REPLACE fires no DELETE trigger and gives the replacement a new rowid, so
-- the counter below would drift up forever and the FTS entry keyed on the old
-- rowid would be stranded. Recomputing both from scratch absorbed that, and
-- cost a full aggregate scan of every row sharing the key ON EVERY INSERT --
-- 1.5 ms per row at 8k rows and still doubling, against a flat 34 us/row once
-- the writes are honest and the counters are arithmetic.
--
-- This cannot be a trigger. SQLite runs BEFORE INSERT triggers before conflict
-- resolution, so a guard there cannot tell REPLACE from the upsert it is
-- steering callers towards -- it rejects both. The rule is enforced by a test
-- over the source instead: tests/test_schema_contract.py.
CREATE TRIGGER param_key_learn AFTER INSERT ON file_param BEGIN
  INSERT INTO param_key(source,key,value_kind,occurrences,first_seen_at,last_seen_at)
  VALUES (NEW.source, NEW.key,
          CASE WHEN NEW.value_num IS NULL THEN 'text' ELSE 'number' END,
          1, unixepoch(), unixepoch())
  ON CONFLICT(source,key) DO UPDATE SET
    occurrences  = occurrences + 1,
    -- A three-state lattice that only ever widens: once a key has been seen
    -- both ways it is mixed and stays mixed. Deciding it by aggregate meant
    -- reading the whole key's history to answer a question with three
    -- possible values.
    value_kind   = CASE
                     WHEN value_kind = 'mixed' THEN 'mixed'
                     WHEN value_kind =
                       CASE WHEN NEW.value_num IS NULL THEN 'text' ELSE 'number' END
                       THEN value_kind
                     ELSE 'mixed' END,
    last_seen_at = unixepoch();
END;

-- An update can move a row between keys, so the old key loses one and the new
-- key gains one. Both are arithmetic, and the WHEN keeps the common case --
-- rewriting a value in place -- from touching the registry at all.
CREATE TRIGGER param_key_relearn AFTER UPDATE ON file_param BEGIN
  UPDATE param_key SET occurrences = occurrences - 1
   WHERE (source, key) = (OLD.source, OLD.key)
     AND (OLD.source, OLD.key) <> (NEW.source, NEW.key);
  INSERT INTO param_key(source,key,value_kind,occurrences,first_seen_at,last_seen_at)
  VALUES (NEW.source, NEW.key,
          CASE WHEN NEW.value_num IS NULL THEN 'text' ELSE 'number' END,
          CASE WHEN (OLD.source, OLD.key) = (NEW.source, NEW.key) THEN 1 ELSE 1 END,
          unixepoch(), unixepoch())
  ON CONFLICT(source,key) DO UPDATE SET
    occurrences = occurrences
      + CASE WHEN (OLD.source, OLD.key) = (NEW.source, NEW.key) THEN 0 ELSE 1 END,
    value_kind  = CASE
                    WHEN value_kind = 'mixed' THEN 'mixed'
                    WHEN value_kind =
                      CASE WHEN NEW.value_num IS NULL THEN 'text' ELSE 'number' END
                      THEN value_kind
                    ELSE 'mixed' END,
    last_seen_at = unixepoch();
  DELETE FROM param_key WHERE occurrences <= 0;
END;

CREATE TRIGGER param_key_forget AFTER DELETE ON file_param BEGIN
  UPDATE param_key SET occurrences = occurrences - 1
   WHERE source = OLD.source AND key = OLD.key;
  -- a key nobody uses is not a field the library contains
  DELETE FROM param_key WHERE source = OLD.source AND key = OLD.key
     AND occurrences <= 0;
END;

-- ============ search: every text surface is indexed ============
-- unicode61 for prose (prompts, captions): word matching, diacritic folded.
-- trigram for identifiers (model filenames, param values): real indexed
-- substring search, which is what the per-row fuzzykey/wordkey UDFs were
-- emulating at a Python call per row.

-- External content: the prompt text is stored once, in `prompt`.
CREATE VIRTUAL TABLE prompt_fts USING fts5(
    text,
    content='prompt',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
-- Dedupe is enforced here rather than left to the writer's choice of verb.
-- INSERT OR REPLACE on text_hash would delete the displaced row WITHOUT firing
-- the DELETE trigger, orphaning an index entry that then resolves to a dead or
-- reused rowid -- silently attributing one image's prompt to another. RAISE
-- IGNORE turns that statement into a no-op instead.
CREATE TRIGGER prompt_dedupe BEFORE INSERT ON prompt
WHEN EXISTS (SELECT 1 FROM prompt WHERE text_hash = NEW.text_hash) BEGIN
  SELECT RAISE(IGNORE);
END;

CREATE TRIGGER prompt_fts_insert AFTER INSERT ON prompt BEGIN
  INSERT INTO prompt_fts(rowid, text) VALUES (NEW.id, NEW.text);
END;
CREATE TRIGGER prompt_fts_delete AFTER DELETE ON prompt BEGIN
  INSERT INTO prompt_fts(prompt_fts, rowid, text) VALUES('delete', OLD.id, OLD.text);
END;
CREATE TRIGGER prompt_fts_update AFTER UPDATE ON prompt BEGIN
  INSERT INTO prompt_fts(prompt_fts, rowid, text) VALUES('delete', OLD.id, OLD.text);
  INSERT INTO prompt_fts(rowid, text) VALUES (NEW.id, NEW.text);
END;

-- Names of anything addressable, searchable by substring.
-- Names of anything addressable, searchable by substring. Every named kind is
-- indexed, and every one follows a rename -- an index that only learns names
-- at insert answers with the old one forever.
-- Note: the trigram tokenizer emits nothing below three characters
-- (fts5_tokenize.c), so short names such as "XL" are not substring-searchable
-- and callers must fall back to an equality match on name_key.
-- THE ROWID IS THE ENTITY ID. Five tables feed one index, so this cannot
-- be an external-content table -- but it can key on the entity, and that
-- is what makes removing an entry a B-tree lookup instead of a scan.
--
-- It carried `entity_id UNINDEXED` and every update and delete trigger
-- matched on it, so renaming or deleting one row scanned the whole index.
-- Invisible on a small library and quadratic on a real one: the scanner
-- renames and moves constantly, and a folder rename touches every name
-- under it.
CREATE VIRTUAL TABLE name_fts USING fts5(name, tokenize='trigram');

CREATE TRIGGER name_fts_artifact_ins AFTER INSERT ON artifact
WHEN NEW.name IS NOT NULL BEGIN
  INSERT INTO name_fts(rowid, name) VALUES (NEW.id, NEW.name);
END;
CREATE TRIGGER name_fts_artifact_upd AFTER UPDATE OF name ON artifact BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
  INSERT INTO name_fts(rowid, name)
    SELECT NEW.id, NEW.name WHERE NEW.name IS NOT NULL;
END;
CREATE TRIGGER name_fts_artifact_del AFTER DELETE ON artifact BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
END;

CREATE TRIGGER name_fts_file_ins AFTER INSERT ON file
WHEN NEW.name IS NOT NULL BEGIN
  INSERT INTO name_fts(rowid, name) VALUES (NEW.id, NEW.name);
END;
CREATE TRIGGER name_fts_file_upd AFTER UPDATE OF name ON file BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
  INSERT INTO name_fts(rowid, name)
    SELECT NEW.id, NEW.name WHERE NEW.name IS NOT NULL;
END;
CREATE TRIGGER name_fts_file_del AFTER DELETE ON file BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
END;

CREATE TRIGGER name_fts_folder_ins AFTER INSERT ON folder
WHEN NEW.name IS NOT NULL BEGIN
  INSERT INTO name_fts(rowid, name) VALUES (NEW.id, NEW.name);
END;
CREATE TRIGGER name_fts_folder_upd AFTER UPDATE OF name ON folder BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
  INSERT INTO name_fts(rowid, name)
    SELECT NEW.id, NEW.name WHERE NEW.name IS NOT NULL;
END;
CREATE TRIGGER name_fts_folder_del AFTER DELETE ON folder BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
END;

CREATE TRIGGER name_fts_person_ins AFTER INSERT ON person
WHEN NEW.name IS NOT NULL BEGIN
  INSERT INTO name_fts(rowid, name) VALUES (NEW.id, NEW.name);
END;
CREATE TRIGGER name_fts_person_upd AFTER UPDATE OF name ON person BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
  INSERT INTO name_fts(rowid, name)
    SELECT NEW.id, NEW.name WHERE NEW.name IS NOT NULL;
END;
CREATE TRIGGER name_fts_person_del AFTER DELETE ON person BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
END;

CREATE TRIGGER name_fts_collection_ins AFTER INSERT ON collection
WHEN NEW.name IS NOT NULL BEGIN
  INSERT INTO name_fts(rowid, name) VALUES (NEW.id, NEW.name);
END;
CREATE TRIGGER name_fts_collection_upd AFTER UPDATE OF name ON collection BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
  INSERT INTO name_fts(rowid, name)
    SELECT NEW.id, NEW.name WHERE NEW.name IS NOT NULL;
END;
CREATE TRIGGER name_fts_collection_del AFTER DELETE ON collection BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
END;

-- The long tail's own values, so a scraped field is searchable the day it
-- first appears rather than when someone writes a facet for it.
-- `source` is carried here because the file_param key is (file_id, source,
-- key): without it the delete predicate wipes the XMP row's entry when the
-- IPTC row of the same name is removed.
-- EXTERNAL CONTENT, keyed on file_param's rowid. This was a standalone table
-- carrying its own copies of file_id/key/source as UNINDEXED columns, and the
-- delete half of each trigger matched on them -- so every insert SCANNED THE
-- WHOLE INDEX to find the row it was replacing. Measured at 8k rows: 1.5 ms
-- per row and still doubling with size, against a flat 2 us/row for FTS5
-- itself at any size. The quadratic term was the scan, never the tokenizer.
--
-- Keyed on the rowid the delete is a B-tree lookup, and the columns those
-- scans existed to filter on are read by joining back to file_param, which a
-- search has to do anyway to render a result.
CREATE VIRTUAL TABLE param_fts USING fts5(
    -- The column MUST be named as file_param names it: for external content
    -- FTS5 builds 'SELECT T."<col>" FROM <content>' from the FTS column names
    -- verbatim (refs/sqlite/sqlite/ext/fts5/fts5_config.c:530), so a mismatch
    -- is not a rename, it is a query against a column that does not exist.
    value_text, content='file_param', content_rowid='rowid', tokenize='trigram');

-- Correct only because nothing writes file_param with INSERT OR REPLACE:
-- REPLACE deletes the conflicting row WITHOUT firing a DELETE trigger and the
-- replacement gets a NEW rowid, stranding the old index entry forever.
-- `file_param_no_replace` below is what keeps that true.
CREATE TRIGGER param_fts_insert AFTER INSERT ON file_param
WHEN NEW.value_text IS NOT NULL BEGIN
  INSERT INTO param_fts(rowid, value_text) VALUES (NEW.rowid, NEW.value_text);
END;

CREATE TRIGGER param_fts_update AFTER UPDATE ON file_param BEGIN
  INSERT INTO param_fts(param_fts, rowid, value_text)
    SELECT 'delete', OLD.rowid, OLD.value_text WHERE OLD.value_text IS NOT NULL;
  INSERT INTO param_fts(rowid, value_text)
    SELECT NEW.rowid, NEW.value_text WHERE NEW.value_text IS NOT NULL;
END;

CREATE TRIGGER param_fts_delete AFTER DELETE ON file_param
WHEN OLD.value_text IS NOT NULL BEGIN
  INSERT INTO param_fts(param_fts, rowid, value_text)
    VALUES('delete', OLD.rowid, OLD.value_text);
END;

-- ============ derived_*: drop this namespace, re-index, reproduced ============
-- Recomputing must replace, never append: an interrupted job re-run must not
-- triple every face. Two shapes do that, and which one applies is a property
-- of the answer rather than of the table.
--
-- Where one row IS the answer for a subject -- a hash, an embedding, a caption
-- from one model -- the table carries a natural key and the producer upserts.
--
-- Where the answer is a SET whose size can change, there is no such key: a
-- face is located by a `region`, and a re-run mints a new region, so keying on
-- one would append forever, and a detector that finds two faces where the last
-- version found three must be able to say so. Those producers delete their own
-- scope and rewrite it -- see derived.record_faces and derived.recluster. A
-- row-at-a-time insert into these two tables is a defect, which is why the
-- functions that do it are private.
-- One fingerprint per (file, space): pHash rows live under the
-- perceptual.phash64 space, dHash rows under perceptual.dhash64, and the
-- next algorithm is a new space row rather than a new column -- two
-- values sharing one row shared one provenance, and only one of them was
-- telling the truth about who computed it.
CREATE TABLE derived_file_hash (
    file_id       INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    -- Which immutable space produced these bits. Part of the key: after a
    -- producer or preprocess upgrade the same file gets a NEW row under
    -- the new space, and the old row keeps saying -- forever -- which
    -- implementation actually computed it. Without this column an
    -- upgrade relabeled old hashes as new ones by doing nothing at all.
    space_id      INTEGER NOT NULL REFERENCES similarity_space(id) ON DELETE RESTRICT,
    -- 64 bits of fingerprint, and SQLite INTEGER is SIGNED 64-bit: any
    -- value with the top bit set is stored negative. Compare them bitwise,
    -- never with < or > -- ordering these numerically is meaningless, and
    -- Hamming distance is the only comparison that means anything.
    value         INTEGER,
    source_sha256 TEXT NOT NULL, computed_at REAL NOT NULL,
    PRIMARY KEY (file_id, space_id)
) STRICT;
CREATE INDEX derived_file_hash_space ON derived_file_hash(space_id);

-- The IMMUTABLE identity of one similarity representation: what its
-- vectors are (representation, dimensions, metric), what computed them
-- (producer, producer_version) and what fed the computation
-- (preprocess, preprocess_version -- the orient/poster policy is as
-- much a part of a phash's meaning as the hash algorithm). Rows are
-- minted once, keyed by spec_hash, and never change: a change in any
-- meaning-bearing field IS a different space and mints a new id, so a
-- representation row pointing here can never be relabeled as something
-- newer than what actually produced it. AUTOINCREMENT for the same
-- reason as `entity`: these ids are identities held outside this table.
CREATE TABLE similarity_space (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    key                TEXT NOT NULL,
    representation     TEXT NOT NULL CHECK (representation IN ('binary','float32')),
    dimensions         INTEGER NOT NULL CHECK (dimensions > 0),
    metric             TEXT NOT NULL CHECK (metric IN ('hamming','cosine')),
    producer           TEXT NOT NULL,
    producer_version   TEXT NOT NULL,
    preprocess         TEXT NOT NULL,
    preprocess_version TEXT NOT NULL,
    spec_hash          TEXT NOT NULL UNIQUE,
    created_at         REAL NOT NULL
) STRICT;

CREATE TRIGGER similarity_space_is_immutable
BEFORE UPDATE ON similarity_space
BEGIN
    SELECT RAISE(ABORT, 'similarity_space rows are immutable: a changed meaning is a new space');
END;

-- One whole-file embedding per (file, space): the joint image/text space
-- semantic search lives in. Which model, which weights, and which
-- preprocessing produced a vector is the space row's identity -- a CLIP
-- image vector is only comparable to text encoded by the SAME
-- checkpoint, and rows keyed by immutable space id cannot be answered
-- by the wrong encoder after an upgrade.
CREATE TABLE derived_embedding (
    -- AUTOINCREMENT for the same reason as derived_face_instance: this id
    -- is what the resident index stores, and index alignment treats an id
    -- as an IMMUTABLE identity. A file's embedding legitimately changes
    -- (re-embed after an in-place byte replacement, a recompute), so the
    -- file id cannot be the vector's identity -- a crash between commit
    -- and index sync would leave a snapshot holding the OLD vector under
    -- an id alignment has no reason to doubt. A replacement row is a NEW
    -- id; the old id disappears; alignment sees exactly that.
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id       INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    space_id      INTEGER NOT NULL REFERENCES similarity_space(id) ON DELETE RESTRICT,
    -- Packed float32, as the encoder emits it. No dim column: the space
    -- row owns the dimensions, and the triggers below hold every vector
    -- to them -- a second copy of the same fact is a place for the two
    -- to disagree.
    vector        BLOB NOT NULL,
    source_sha256 TEXT NOT NULL, computed_at REAL NOT NULL,
    UNIQUE (file_id, space_id)
) STRICT;
CREATE INDEX derived_embedding_space ON derived_embedding(space_id);

-- A vector whose byte length disagrees with its space's dimensions
-- unpacks into noise, and a search over mixed lengths groups strangers.
CREATE TRIGGER derived_embedding_fits_its_space
BEFORE INSERT ON derived_embedding
WHEN EXISTS (
    SELECT 1 FROM similarity_space s WHERE s.id = NEW.space_id
      AND s.dimensions <> length(NEW.vector) / 4
)
BEGIN
    SELECT RAISE(ABORT, 'embedding length disagrees with its space''s dimensions');
END;
CREATE TRIGGER derived_embedding_fits_its_space_update
BEFORE UPDATE ON derived_embedding
WHEN EXISTS (
    SELECT 1 FROM similarity_space s WHERE s.id = NEW.space_id
      AND s.dimensions <> length(NEW.vector) / 4
)
BEGIN
    SELECT RAISE(ABORT, 'embedding length disagrees with its space''s dimensions');
END;

-- A prompt text is a document of SECTIONS (db/prompt_sections.py): the
-- main prompt, then each alternate prompt a tool routes to a stage or an
-- area (SwarmUI's <segment:> <region:> <refiner> <video> ...). One parse
-- per (prompt, grammar), the grammar chosen by the generation's tool.
-- Each section's text is an ordinary interned prompt row: "a red fox"
-- as a main prompt, inside a region, or as a negative is ONE text
-- identity, so its vector is computed once and a parser upgrade --
-- which re-parses boundaries -- never re-embeds identical bytes.
CREATE TABLE derived_prompt_section (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id        INTEGER NOT NULL REFERENCES prompt(id) ON DELETE CASCADE,
    grammar          TEXT NOT NULL CHECK (grammar IN ('plain','swarm')),
    ordinal          INTEGER NOT NULL,
    kind             TEXT NOT NULL CHECK (kind IN
                       ('main','base','pixeldecoder','refiner','video','videoswap',
                        'extend','object','region','segment')),
    spec             TEXT,
    text             TEXT NOT NULL,
    text_prompt_id   INTEGER REFERENCES prompt(id) ON DELETE CASCADE,
    source_text_hash TEXT NOT NULL, parser_version INTEGER NOT NULL, computed_at REAL NOT NULL,
    UNIQUE (prompt_id, grammar, ordinal)
) STRICT;
CREATE INDEX derived_prompt_section_text ON derived_prompt_section(text_prompt_id);
CREATE INDEX derived_prompt_section_kind ON derived_prompt_section(kind, grammar);

-- One vector per (prompt TEXT, space, query policy). `space_id` is the
-- provider's JOINT space -- the coordinate system its media vectors
-- live in, because encode_query produces vectors there and retrieval
-- already searches media with them -- so a prompt vector may be
-- compared with a media vector of the same space. `policy_hash` is the
-- QUERY policy that produced it (instruction, tokenizer, package):
-- provenance and currentness, so a changed instruction is a NEW row
-- that coexists with the old one. Prompt rows never enter the media
-- resident index: their own index per (space, policy) holds them --
-- same coordinates, different corpus, no id collisions.
-- `source_text_hash` is the exact text the vector was computed from; a
-- row is current only while it equals the prompt's own text_hash, and
-- a consumer looks vectors up by THAT hash, never by a file or a
-- generation. AUTOINCREMENT for the same reason as derived_embedding.
CREATE TABLE derived_prompt_embedding (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id        INTEGER NOT NULL REFERENCES prompt(id) ON DELETE CASCADE,
    space_id         INTEGER NOT NULL REFERENCES similarity_space(id) ON DELETE RESTRICT,
    policy_hash      TEXT NOT NULL,
    vector           BLOB NOT NULL,
    source_text_hash TEXT NOT NULL, computed_at REAL NOT NULL,
    UNIQUE (prompt_id, space_id, policy_hash)
) STRICT;
CREATE INDEX derived_prompt_embedding_space ON derived_prompt_embedding(space_id, policy_hash);
CREATE INDEX derived_prompt_embedding_hash  ON derived_prompt_embedding(source_text_hash, space_id, policy_hash);
CREATE TRIGGER derived_prompt_embedding_fits_its_space
BEFORE INSERT ON derived_prompt_embedding
WHEN EXISTS (
    SELECT 1 FROM similarity_space s WHERE s.id = NEW.space_id
      AND s.dimensions <> length(NEW.vector) / 4
)
BEGIN
    SELECT RAISE(ABORT, 'prompt embedding length disagrees with its space''s dimensions');
END;
CREATE TRIGGER derived_prompt_embedding_fits_its_space_update
BEFORE UPDATE ON derived_prompt_embedding
WHEN EXISTS (
    SELECT 1 FROM similarity_space s WHERE s.id = NEW.space_id
      AND s.dimensions <> length(NEW.vector) / 4
)
BEGIN
    SELECT RAISE(ABORT, 'prompt embedding length disagrees with its space''s dimensions');
END;

-- One grouping of the library's faces, by one algorithm at one threshold
-- over one embedder's output. All four decide who ends up in a cluster, so
-- all four are the run's identity.
--
-- A first version keyed clusters on the embedder alone, and that quietly
-- made "which clustering is right" unanswerable: a second method or a second
-- threshold replaced the first, so the only way to try one was to destroy
-- the other. There is no settled answer to compare against either -- this
-- repo's own measurements put ArcFace at 0.48 and SFace at 0.55, and being
-- wrong by a tenth puts 96% of a library into a single person
-- (docs/FACE_CLUSTERING.md, Chaining). Comparing runs is the normal activity.
--
-- Several runs are live at once. `primary_run` says which one the People
-- page shows when nobody asked for a particular one; the rest are there to
-- be looked at, argued with, and promoted.
CREATE TABLE derived_face_run (
    id            INTEGER PRIMARY KEY,
    model_id      TEXT NOT NULL, model_version TEXT NOT NULL,
    -- A token, so runs group and count: 'chinese-whispers',
    -- 'connected-components', 'hdbscan'. Not a CHECK list -- a method added
    -- later must not need a schema change -- but not a sentence either.
    method        TEXT NOT NULL CHECK (method = lower(method) AND method NOT LIKE '% %'),
    -- What it grouped at. NULL only for a method that has no threshold.
    threshold     REAL,
    -- Set on the one run the site shows by default. At most one, enforced
    -- below: two defaults is a People page that depends on which row was
    -- read first.
    is_primary    INTEGER NOT NULL DEFAULT 0 CHECK (is_primary IN (0,1)),
    faces         INTEGER NOT NULL DEFAULT 0,
    clusters      INTEGER NOT NULL DEFAULT 0,
    -- Which similarity backend found the pairs: 'faiss-gpu', 'faiss-cpu',
    -- 'numpy'. They compute the same edges, so this does not change who is
    -- in a cluster -- it is here because a timing nobody can attribute to a
    -- machine is not a measurement, and because "faiss is GPU here" and
    -- "faiss is a CPU wheel here" are both true of different environments on
    -- the same box.
    backend       TEXT,
    computed_at   REAL NOT NULL
) STRICT;
-- IFNULL, because a NULL threshold is distinct from every other NULL in a
-- plain UNIQUE, and two runs of a threshold-less method would both be
-- allowed to exist.
CREATE UNIQUE INDEX derived_face_run_identity
    ON derived_face_run(model_id, model_version, method, IFNULL(threshold, -1));
CREATE UNIQUE INDEX derived_face_run_primary
    ON derived_face_run(is_primary) WHERE is_primary = 1;

CREATE TABLE derived_face_cluster (
    id            INTEGER PRIMARY KEY,
    run_id        INTEGER NOT NULL REFERENCES derived_face_run(id) ON DELETE CASCADE,
    person_id     INTEGER REFERENCES person(id) ON DELETE SET NULL,
    centroid      BLOB, dim INTEGER,
    model_id      TEXT NOT NULL, model_version TEXT NOT NULL,
    updated_at    REAL NOT NULL
) STRICT;
CREATE INDEX derived_face_cluster_person ON derived_face_cluster(person_id);
CREATE INDEX derived_face_cluster_run    ON derived_face_cluster(run_id);

-- Which faces a cluster holds. A join table rather than a column on the face,
-- because a face belongs to one cluster PER RUN and several runs coexist:
-- with `derived_face_instance.cluster_id` the second run overwrote the
-- first's answer for every face, which is why nothing could be compared.
CREATE TABLE derived_face_membership (
    cluster_id INTEGER NOT NULL REFERENCES derived_face_cluster(id) ON DELETE CASCADE,
    face_id    INTEGER NOT NULL REFERENCES derived_face_instance(id) ON DELETE CASCADE,
    PRIMARY KEY (cluster_id, face_id)
) STRICT, WITHOUT ROWID;
CREATE INDEX derived_face_membership_face ON derived_face_membership(face_id);

-- One detector's pass over one file's current bytes: that detection
-- HAPPENED, and found `faces` of them -- zero included. The instance
-- rows cannot say that (no faces, no rows), and a sweep that cannot
-- tell "never looked" from "looked, found nobody" re-looks at every
-- face-free picture forever. Keyed per model so two backends' passes
-- coexist; the sweep asks whether ANY pass covers the current bytes.
CREATE TABLE derived_face_scan (
    file_id       INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    model_id      TEXT NOT NULL,
    model_version TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    faces         INTEGER NOT NULL CHECK (faces >= 0),
    computed_at   REAL NOT NULL,
    PRIMARY KEY (file_id, model_id, model_version)
) STRICT, WITHOUT ROWID;

CREATE TABLE derived_face_instance (
    -- AUTOINCREMENT for the same reason as `entity`: these ids feed the
    -- resident face index (vision/faiss_index.py), and index alignment
    -- treats an id as a stable identity. Plain INTEGER PRIMARY KEY
    -- reuses the largest free rowid, so a re-detected file could delete
    -- face 42 and mint a DIFFERENT face as 42 -- same id, new embedding,
    -- and an aligned index would keep serving the old vector.
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id       INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    sample_id     INTEGER REFERENCES derived_media_sample(id) ON DELETE CASCADE,
    -- No cluster_id. Which cluster a face is in is a fact about a RUN, and
    -- several runs coexist, so it lives in `derived_face_membership`.
    -- NOT NULL: a face detection with no location cannot be shown, checked,
    -- cropped, or asserted against. RESTRICT, because deleting the region
    -- would leave exactly that.
    region_id     INTEGER NOT NULL REFERENCES region(id) ON DELETE RESTRICT,
    -- Point pairs, consumed whole by the aligner and never filtered on --
    -- unlike generation fields, which is why those are rows and this is not.
    -- BLOB, not TEXT: it is packed floats, and storing it as JSON meant
    -- parsing a string on every crop to get numbers back.
    landmarks BLOB,
    -- The face's own vector, and the whole reason clustering can happen.
    -- `derived_embedding` cannot hold it: that table is keyed per FILE, and
    -- a photograph of eight people has eight face vectors. Without this
    -- column the People page has no input -- `dim` sat here on its own,
    -- describing a vector nothing could store, and every test in the suite
    -- hid it by assigning cluster_id by hand instead of computing it.
    --
    -- Packed float32, as the model emits it, not JSON: a 512-d ArcFace
    -- vector is 2 KB of bytes or 9 KB of decimal text that has to be parsed
    -- back into floats on every comparison, and a clustering pass compares
    -- every pair.
    embedding BLOB,
    det_score REAL CHECK (det_score IS NULL OR det_score BETWEEN 0 AND 1),
    -- How many floats are in `embedding`, and it has to agree: a length that
    -- disagrees with the bytes is a vector that unpacks into noise, and a
    -- clustering pass comparing a 512-d vector against a mislabelled 128-d
    -- one produces groups of strangers. float32, so four bytes each.
    dim INTEGER,
    age INTEGER,
    -- What the model reported, not what anyone is. A free-text column here
    -- collected 'M', 'male', 'Male' and 'F' from one backend, and a facet
    -- over that matches a quarter of what it should.
    sex TEXT CHECK (sex IS NULL OR sex IN ('male','female','unknown')),
    -- Degrees, not radians. Backends differ, and a library that mixed the
    -- two would answer "faces looking away" with a set that is mostly not.
    pose_yaw REAL, pose_pitch REAL, pose_roll REAL,
    model_id TEXT NOT NULL, model_version TEXT NOT NULL,
    -- Which immutable space the embedding belongs to -- the producer AND
    -- the preprocessing that fed it (orientation policy, frame choice).
    -- Paired with `embedding`: a face with no vector is in no space, and
    -- a vector whose space is unknown cannot be compared with anything.
    -- The model_id/model_version columns above stay as the query keys the
    -- pipeline filters on; the space row is the identity that survives
    -- upgrades without relabeling.
    space_id INTEGER REFERENCES similarity_space(id) ON DELETE RESTRICT,
    source_sha256 TEXT NOT NULL, computed_at REAL NOT NULL, native BLOB,
    -- `dim` describes `embedding`, so it has to agree with it. A length that
    -- disagrees with the bytes unpacks into noise, and a clustering pass
    -- comparing a 512-d vector against a mislabelled 128-d one groups
    -- strangers together. float32, so four bytes each.
    CHECK ((embedding IS NULL AND dim IS NULL)
           OR (embedding IS NOT NULL AND dim = length(embedding) / 4)),
    CHECK ((embedding IS NULL) = (space_id IS NULL))
) STRICT;
-- derived_face_instance.pose_yaw / pose_pitch / pose_roll: yaw-first, and
-- the producer's array is not. InsightFace emits [pitch, yaw, roll]
-- (deepinsight/insightface model_zoo/landmark.py:111), so a positional copy
-- lands pitch in `pose_yaw` and nothing downstream can tell: three REAL
-- columns holding plausible degrees either way, no CHECK that can fire, no
-- value out of range. `db/derived.py _insert_face` therefore takes a mapping
-- keyed yaw/pitch/roll and REFUSES a bare triple by name -- the shape that
-- carries no axis names is the shape that carries the bug.
--
-- derived_face_instance.native: the producer's COMPLETE output for this
-- face, one `vision/facestore.py` envelope -- every key the producer's
-- record held, captured by iteration with dtype, shape, byte order and
-- container structure intact. This is the canonical record; every other
-- face column here is a projection out of it for the values a facet
-- filters on or a page renders. Replay and export (`db/faces_native.py`)
-- read THIS column and nothing else, so a value nobody thought to name a
-- column for still comes back bit-exact, without rerunning the pass and
-- without the original file.
--
-- The pass is expensive and the bytes are not: antelopev2 loads a 143 MB
-- 1k3d68 session per worker, and its whole record is ~3.8 KB per face --
-- against re-reading and re-inferring every file in the library, which is
-- only possible at all while the originals are still on disk.
--
-- NULL when the backend handed over no native record -- the test stub is
-- the one shipped backend that may not. A replay of such a row refuses by
-- name; nothing guesses.
--
-- LAST in the column list, on the same line as `computed_at`, and OUTSIDE
-- the table like feedback.model_id below: `ALTER TABLE ADD COLUMN` appends
-- at SQLite's addColOffset -- the end of the final column, ahead of the
-- table constraints -- and it writes no comment. A migrated file therefore
-- reads `computed_at REAL NOT NULL, native BLOB,` with nothing between,
-- so a comment placed there exists only in a fresh build and
-- `test_the_built_database_matches_the_ddl` reports drift on the table.
-- Every future column added to this table by ALTER goes the same way.
CREATE INDEX derived_face_file       ON derived_face_instance(file_id);
-- Clustering reads every vector this model produced, once per re-cluster.
-- Without this it is a full table scan of every face from every model.
CREATE INDEX derived_face_by_model   ON derived_face_instance(model_id, model_version)
    WHERE embedding IS NOT NULL;
CREATE INDEX derived_face_sample     ON derived_face_instance(sample_id);
CREATE INDEX derived_face_region     ON derived_face_instance(region_id);
CREATE INDEX derived_face_space      ON derived_face_instance(space_id);

-- The duplicated model columns are query conveniences; the space row is
-- the identity. This guard keeps the convenience honest: a face row
-- cannot claim one model while pointing at a space another produced.
CREATE TRIGGER derived_face_space_agrees
BEFORE INSERT ON derived_face_instance
WHEN NEW.space_id IS NOT NULL AND EXISTS (
    SELECT 1 FROM similarity_space s WHERE s.id = NEW.space_id
      AND (s.producer IS NOT NEW.model_id OR s.producer_version IS NOT NEW.model_version)
)
BEGIN
    SELECT RAISE(ABORT, 'face row names one model while its space was produced by another');
END;
CREATE TRIGGER derived_face_space_agrees_update
BEFORE UPDATE ON derived_face_instance
WHEN NEW.space_id IS NOT NULL AND EXISTS (
    SELECT 1 FROM similarity_space s WHERE s.id = NEW.space_id
      AND (s.producer IS NOT NEW.model_id OR s.producer_version IS NOT NEW.model_version)
)
BEGIN
    SELECT RAISE(ABORT, 'face row names one model while its space was produced by another');
END;

-- What a model says about a picture in words: a caption, a longer
-- description, a tag, text it read out of the image. One table, because they
-- differ only in `kind` -- same subject, same provenance, same lifetime --
-- and because a caption is a thing you search for, not a document you open.
--
-- `region_id` is how an annotation points at part of the picture: OCR text
-- sits somewhere, a tag may be about one object. NULL means it is about the
-- whole frame.
--
-- `sample_id` is how it points at a moment: a caption for a video is a
-- caption of a frame, and a description that cannot say which frame is not
-- checkable.
CREATE TABLE derived_annotation (
    id            INTEGER PRIMARY KEY,
    file_id       INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    sample_id     INTEGER REFERENCES derived_media_sample(id) ON DELETE CASCADE,
    region_id     INTEGER REFERENCES region(id) ON DELETE SET NULL,
    kind          TEXT NOT NULL CHECK (kind IN
                    ('caption','description','alt_text','tag','ocr','title')),
    text          TEXT NOT NULL,
    confidence    REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    -- The same picture may carry a caption from two models on purpose: they
    -- are compared, not merged. Uniqueness is per model, per kind, per region.
    model_id      TEXT NOT NULL,
    model_version TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    computed_at   REAL NOT NULL,
    CHECK (length(text) > 0)
) STRICT;
-- No index on file_id alone: derived_annotation_one leads on it.
CREATE INDEX derived_annotation_kind   ON derived_annotation(kind, file_id);
CREATE INDEX derived_annotation_region ON derived_annotation(region_id);
CREATE INDEX derived_annotation_sample ON derived_annotation(sample_id);
CREATE UNIQUE INDEX derived_annotation_one
    ON derived_annotation(file_id, kind, model_id, model_version,
                          IFNULL(region_id, 0), IFNULL(sample_id, 0));

-- Captions are prose, and prose is searched by word. Without this a caption
-- is text nobody can find, which is the same as not having captioned.
CREATE VIRTUAL TABLE annotation_fts USING fts5(
    text, content='derived_annotation', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER annotation_fts_insert AFTER INSERT ON derived_annotation BEGIN
  INSERT INTO annotation_fts(rowid, text) VALUES (NEW.id, NEW.text);
END;
CREATE TRIGGER annotation_fts_delete AFTER DELETE ON derived_annotation BEGIN
  INSERT INTO annotation_fts(annotation_fts, rowid, text)
    VALUES('delete', OLD.id, OLD.text);
END;
CREATE TRIGGER annotation_fts_update AFTER UPDATE OF text ON derived_annotation BEGIN
  INSERT INTO annotation_fts(annotation_fts, rowid, text)
    VALUES('delete', OLD.id, OLD.text);
  INSERT INTO annotation_fts(rowid, text) VALUES (NEW.id, NEW.text);
END;

-- ============ authored ============
CREATE TABLE rating (
    file_id INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES user(id) ON DELETE CASCADE,
    rating  INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    created_at REAL NOT NULL,
    PRIMARY KEY (file_id, user_id)
) STRICT, WITHOUT ROWID;
CREATE INDEX rating_user ON rating(user_id);

CREATE TABLE comment (
    id INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES user(id) ON DELETE CASCADE,
    body TEXT NOT NULL, created_at REAL NOT NULL, edited_at REAL
) STRICT;
CREATE INDEX comment_file ON comment(file_id);
CREATE INDEX comment_user ON comment(user_id);

-- Where a picture happened, said by a person: the one authored location
-- claim. Survives every rebuild; the context ladder reads it as the
-- 'authored' basis above GPS. One place per file -- a picture happened
-- in one place; a change of mind replaces the row.
CREATE TABLE file_place (
    file_id     INTEGER PRIMARY KEY REFERENCES file(id) ON DELETE CASCADE,
    place_id    INTEGER NOT NULL REFERENCES place(id) ON DELETE RESTRICT,
    user_id     INTEGER NOT NULL REFERENCES user(id) ON DELETE CASCADE,
    asserted_at REAL NOT NULL
) STRICT;
CREATE INDEX file_place_place ON file_place(place_id);
CREATE INDEX file_place_user  ON file_place(user_id);

CREATE TABLE favorite (
    file_id INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES user(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    PRIMARY KEY (file_id, user_id)
) STRICT, WITHOUT ROWID;
CREATE INDEX favorite_user ON favorite(user_id);

-- A keyword. The oldest idea in digital asset management and the last
-- one this schema grew, so the reasons it looks like this are worth
-- stating.
--
-- NOT an entity, unlike `person`, `place` and `collection`. `entity.kind`
-- is a CHECK constraint, so admitting one more kind means rebuilding the
-- most-referenced table in the file -- and a tag would get nothing for
-- it. An entity exists to have an address, a history of spellings and a
-- page; a keyword has a name, and renaming one is this UPDATE rather
-- than a retired slug somebody may still hold a bookmark to.
--
-- NOT `derived_annotation` with kind='tag' either, which is where this
-- nearly went. That namespace requires a model_id and a source_sha256
-- and is deleted wholesale by derived.drop_all -- so a keyword a person
-- typed would be a durable claim living in the disposable half, gone at
-- the next re-annotate with nothing reporting it. Authored facts sit
-- here beside `rating` and `file_place`: they survive every rebuild.
--
-- Two columns for one name because case is not identity. `tag` is the
-- normalised form (casefolded, whitespace collapsed -- db/authored.py
-- `normalised`) and carries the UNIQUE; `label` is what somebody
-- actually typed, so "New York" is displayed the way they wrote it and
-- "new york" typed later lands on the same keyword rather than
-- splitting the library in two. Folded in Python and not by COLLATE
-- NOCASE, which folds ASCII only and would leave CAFE and Cafe apart.
CREATE TABLE tag (
    id         INTEGER PRIMARY KEY,
    tag        TEXT NOT NULL UNIQUE,
    label      TEXT NOT NULL,
    created_at REAL NOT NULL
) STRICT;

-- Shared rather than per-actor, which is the difference from `rating`
-- and `favorite` beside it: those are one person's opinion and two
-- people may hold different ones, while a keyword is a fact about the
-- picture that everybody reads. So the key is (file, tag) and `user_id`
-- records who first said it rather than whose it is.
CREATE TABLE file_tag (
    file_id    INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    tag_id     INTEGER NOT NULL REFERENCES tag(id)  ON DELETE CASCADE,
    user_id    INTEGER NOT NULL REFERENCES user(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    PRIMARY KEY (file_id, tag_id)
) STRICT, WITHOUT ROWID;
-- Both for the reason `job_target` is indexed: SQLite checks every child
-- table when a parent row goes, and an unindexed FK makes that a full
-- scan of `file_tag` per deleted file.
CREATE INDEX file_tag_tag  ON file_tag(tag_id);
CREATE INDEX file_tag_user ON file_tag(user_id);

-- Feedback is authored and survives every rebuild, so it must not point at a
-- row that a rebuild destroys. An earlier version carried
-- (target_kind, target_ref) with no constraint -- reintroducing the exact
-- polymorphic reference `entity` exists to eliminate, and aiming durable rows
-- at disposable ones.
--
-- Every target is now a durable entity, and the judgement re-attaches after
-- the derived layer is rebuilt:
--   annotation      -> the file it described, plus the kind of description
--   similarity/dup  -> the pair of files
--   person          -> the person, which is authored identity
CREATE TABLE feedback (
    id             INTEGER PRIMARY KEY,
    target_kind    TEXT NOT NULL CHECK (target_kind IN
                     ('annotation','similarity','duplicate','person')),
    file_id        INTEGER REFERENCES file(id)   ON DELETE SET NULL,
    other_file_id  INTEGER REFERENCES file(id)   ON DELETE SET NULL,
    person_id      INTEGER REFERENCES person(id) ON DELETE SET NULL,
    -- Which description was judged. The annotation row itself is derived and
    -- will be deleted by the next rebuild; this survives it, so "the caption
    -- for this file was wrong" still means something afterwards.
    annotation_kind TEXT CHECK (annotation_kind IS NULL OR annotation_kind IN
                     ('caption','description','alt_text','tag','ocr','title')),
    verdict        TEXT NOT NULL CHECK (verdict IN ('right','wrong','unsure')),
    note           TEXT,
    user_id        INTEGER REFERENCES user(id) ON DELETE SET NULL,
    created_at     REAL NOT NULL, model_id TEXT, model_version TEXT,
    CHECK (other_file_id IS NULL OR other_file_id <> file_id)
) STRICT;
-- feedback.model_id / model_version: WHICH producer was judged, copied at
-- judgement time. Not a foreign key, and deliberately not the annotation's
-- row -- the derived layer is disposable and a judgement has to outlive
-- being rebuilt, which is why `annotation_kind` is a column here rather
-- than an id. Without these two the table could say a caption was wrong and
-- never which model wrote it, so "this model gets 12% of my library wrong"
-- -- the reason to collect verdicts at all -- was unanswerable the moment
-- the layer was rebuilt. Null where a verdict is not about a model: a
-- person judgement names a person.
--
-- Spelled exactly as SQLite stores an ALTER TABLE ADD COLUMN on the v33
-- text -- inside the parenthesis, after the table constraint, no comment --
-- so a migrated file's DDL reads equal to a fresh build's (db/build.py
-- drift). The same reason file.ingested_sha256 is written that way.
-- Enforced at write, not as a row invariant: ON DELETE SET NULL must be able
-- to detach a judged target without the row becoming illegal. Losing the
-- pointer is acceptable; losing the human judgement is not.
CREATE INDEX feedback_producer ON feedback(model_id, model_version, annotation_kind);
CREATE TRIGGER feedback_names_a_target BEFORE INSERT ON feedback BEGIN
  SELECT RAISE(ABORT,'feedback must name what it judges')
  WHERE NOT (
      (NEW.target_kind = 'annotation'                AND NEW.file_id IS NOT NULL
                                                     AND NEW.annotation_kind IS NOT NULL)
   OR (NEW.target_kind IN ('similarity','duplicate') AND NEW.file_id IS NOT NULL
                                                     AND NEW.other_file_id IS NOT NULL)
   OR (NEW.target_kind = 'person'                     AND NEW.person_id IS NOT NULL));
END;

CREATE INDEX feedback_file   ON feedback(file_id);
CREATE INDEX feedback_other  ON feedback(other_file_id);
CREATE INDEX feedback_person ON feedback(person_id);
CREATE INDEX feedback_user   ON feedback(user_id);

-- "a human says this person appears in this file". The one durable fact in
-- the face pipeline, and the seed a rebuild re-attributes from -- so naming is
-- re-applied from a record rather than re-guessed by centroid similarity, as
-- _match_preserved_labels had to.
CREATE TABLE person_assertion (
    person_id  INTEGER NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    file_id    INTEGER NOT NULL REFERENCES file(id)   ON DELETE CASCADE,
    sample_id  INTEGER REFERENCES derived_media_sample(id) ON DELETE SET NULL,
    -- Optional and SET NULL: the claim is "this person is in this file", and
    -- it stands whether or not anybody drew a box around them.
    region_id  INTEGER REFERENCES region(id) ON DELETE SET NULL,
    user_id    INTEGER REFERENCES user(id) ON DELETE SET NULL,
    created_at REAL NOT NULL, stance     TEXT NOT NULL DEFAULT 'is' CHECK (stance IN ('is','is_not')),
    PRIMARY KEY (person_id, file_id)
) STRICT, WITHOUT ROWID;
-- person_assertion.stance: whether the claim is that this person IS in this
-- file or is NOT.
--
-- A negative is a CLAIM, and that is the whole reason it is a row rather
-- than the absence of one. Retracting an assertion deletes it, which means
-- "I take that back" -- and the next clustering run is then free to decide
-- the same thing again, because nothing recorded that it was wrong. "Not
-- her" has to survive the rebuild and constrain it, exactly as a positive
-- does, which is what makes a correction permanent instead of a chore
-- somebody repeats after every re-run.
--
-- One row per (person, file) either way: a person cannot both be and not be
-- in one picture, and the upsert makes saying the second withdraw the first.
--
-- Spelled the way ALTER TABLE ADD COLUMN leaves it in v35 -- inside the
-- parenthesis, before the PRIMARY KEY, no comment -- so a migrated file and
-- a fresh build hold the same DDL text (db/build.py drift). The same reason
-- file.ingested_sha256 and feedback.model_id are written that way.
CREATE INDEX person_assertion_file   ON person_assertion(file_id);
CREATE INDEX person_assertion_user   ON person_assertion(user_id);
CREATE INDEX person_assertion_region ON person_assertion(region_id);
CREATE INDEX person_assertion_sample ON person_assertion(sample_id);

CREATE TABLE watched_folder (
    folder_id INTEGER PRIMARY KEY REFERENCES folder(id) ON DELETE CASCADE,
    recursive INTEGER NOT NULL DEFAULT 1 CHECK (recursive IN (0,1)),
    added_at  REAL NOT NULL
) STRICT, WITHOUT ROWID;

-- A question worth asking again.
--
-- The third thing people mean, and the one that had nowhere to go.
-- An ALBUM is what somebody deliberately put together; a SMART
-- COLLECTION is a dynamic grouping that behaves like one -- it has
-- members, an address, a place on the shelf, things filed under it. A
-- SAVED VIEW is none of that: it is "that was a useful question,
-- remember it", and making one a collection put five things that are
-- not albums into somebody's album list.
--
-- They share a GalleryQuery underneath without being one product
-- object, which is why this stores the canonical SPELLING
-- (db/resultset.py `canonical`) rather than a rule: the spelling is
-- entity-aware and heals a retired slug to the live one as it is
-- navigated, so a view saved before a rename still answers.
CREATE TABLE saved_view (
    id           INTEGER PRIMARY KEY,
    -- NOCASE, like every other name a person types here: "Portraits"
    -- and "portraits" are the same question asked twice, and the second
    -- should replace the first rather than sit beside it.
    name         TEXT NOT NULL UNIQUE COLLATE NOCASE,
    -- The canonical query string, without a page: a remembered question
    -- opens at its beginning, never at page 7 of an answer that has
    -- since changed length.
    qs           TEXT NOT NULL,
    created_at   REAL NOT NULL,
    -- So the list can put what somebody actually uses at the top. NULL
    -- until it is opened once.
    last_used_at REAL
) STRICT;

CREATE TABLE setting (
    key TEXT PRIMARY KEY, value TEXT NOT NULL
) STRICT, WITHOUT ROWID;

-- What runs without being asked, and how often.
--
-- A schedule points at a COLLECTION, never at a kind. Naming kinds would
-- mean re-deriving the order at 3am -- which of scan, ingest, embed,
-- detect_faces, cluster_faces, and in which sequence -- and that order is
-- the thing job.after_id exists so nobody has to carry.
--
-- One row per collection, by the UNIQUE: two schedules for one act would
-- disagree about when it last ran, and the guard against queueing a
-- second catch-up over a draining first one is per-collection.
--
-- An interval in hours rather than a cron expression. A cron string is a
-- small language, and a small language wants a parser, a validator and a
-- way to say what it will do next; hours answer the question somebody
-- actually has ("nightly", "twice a day") and can be shown as a time
-- without interpreting anything.
CREATE TABLE schedule (
    id              INTEGER PRIMARY KEY,
    collection      TEXT NOT NULL UNIQUE,
    every_hours     REAL NOT NULL CHECK (every_hours > 0),
    enabled         INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0,1)),
    -- When this schedule last STARTED the collection, not when that
    -- collection finished. The next run is measured from the start, so a
    -- catch-up that takes three hours on a nightly schedule still runs
    -- once a night rather than drifting later every day.
    --
    -- NULL means it has never run, which is due now: a schedule somebody
    -- just turned on should not wait a full interval to prove it works.
    last_started_at REAL,
    created_at      REAL NOT NULL
) STRICT;

-- A deduplicated payload with no remaining referrer is garbage; without this it
-- accumulates for the life of the library. RESTRICT protects referenced blobs,
-- and says nothing about unreferenced ones.
--
-- Two referrers, so both are checked: a carrier holds a payload, and a region
-- holds a segmentation mask. Collecting on the carrier alone deleted masks
-- that were still pointed at; collecting on neither leaked every mask in the
-- library on every rebuild, because `derived.drop_all` removes the regions and
-- nothing was watching that side.
CREATE TRIGGER blob_reclaim AFTER DELETE ON file_blob BEGIN
  DELETE FROM blob WHERE hash = OLD.blob_hash
     AND NOT EXISTS (SELECT 1 FROM file_blob WHERE blob_hash = OLD.blob_hash)
     AND NOT EXISTS (SELECT 1 FROM region WHERE mask_hash = OLD.blob_hash);
END;

-- A re-parse whose payload changed. The upsert that replaced INSERT OR REPLACE
-- here updates the row rather than deleting it, so without this the old
-- payload -- a whole workflow graph -- is stranded in `blob` for good.
CREATE TRIGGER blob_reclaim_replaced AFTER UPDATE OF blob_hash ON file_blob
WHEN NEW.blob_hash <> OLD.blob_hash BEGIN
  DELETE FROM blob WHERE hash = OLD.blob_hash
     AND NOT EXISTS (SELECT 1 FROM file_blob WHERE blob_hash = OLD.blob_hash)
     AND NOT EXISTS (SELECT 1 FROM region WHERE mask_hash = OLD.blob_hash);
END;

CREATE TRIGGER blob_reclaim_mask AFTER DELETE ON region
WHEN OLD.mask_hash IS NOT NULL BEGIN
  DELETE FROM blob WHERE hash = OLD.mask_hash
     AND NOT EXISTS (SELECT 1 FROM file_blob WHERE blob_hash = OLD.mask_hash)
     AND NOT EXISTS (SELECT 1 FROM region WHERE mask_hash = OLD.mask_hash);
END;

-- #16: nothing distinguished a database built from this DDL from one built by an
-- earlier generation of it, which is how a stale build went unnoticed.
-- ============ places: where media happened, as identity ============
-- "Hawaii", "HI" and "Hawai'i" as strings are three unrelated spellings;
-- a place is an entity with an address and a hierarchy, so a query for
-- the island naturally includes the beach. Rows are minted by explicit
-- enrichment or authoring -- never by a GET, and never automatically
-- from raw GPS: coordinates without a resolver stay coordinates.
CREATE TABLE place (
    id           INTEGER PRIMARY KEY REFERENCES entity(id) ON DELETE CASCADE,
    -- RESTRICT: places nest, and deleting a region must never silently
    -- take its cities' identities with it.
    parent_id    INTEGER REFERENCES place(id) ON DELETE RESTRICT,
    kind         TEXT NOT NULL CHECK (kind IN
                   ('country','region','island','county','city','locality','neighborhood','poi')),
    name         TEXT NOT NULL,
    centroid_lat REAL,
    centroid_lon REAL,
    country_code TEXT,
    -- Which enrichment provider claimed this place, and its key there --
    -- provenance for refresh, never identity.
    provider     TEXT,
    provider_key TEXT,
    created_at   REAL NOT NULL
) STRICT;
CREATE INDEX place_parent ON place(parent_id);
-- One place per (kind, name, parent): "never two Lisbons" is the
-- database's word, not only the minter's.
CREATE UNIQUE INDEX place_identity ON place(kind, name COLLATE NOCASE, IFNULL(parent_id, 0));

CREATE TRIGGER place_kind_agrees BEFORE INSERT ON place BEGIN
  SELECT RAISE(ABORT,'entity kind does not match place')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'place';
END;
CREATE TRIGGER place_kind_keeps_agreeing BEFORE UPDATE OF id ON place BEGIN
  SELECT RAISE(ABORT,'entity kind does not match place')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'place';
END;
CREATE TRIGGER place_takes_its_entity AFTER DELETE ON place BEGIN
  DELETE FROM entity WHERE id = OLD.id;
END;

CREATE TRIGGER place_no_self_parent BEFORE INSERT ON place
WHEN NEW.parent_id IS NOT NULL AND NEW.parent_id = NEW.id BEGIN
  SELECT RAISE(ABORT,'place parent cycle');
END;
CREATE TRIGGER place_no_cycle BEFORE UPDATE OF parent_id ON place
WHEN NEW.parent_id IS NOT NULL BEGIN
  SELECT RAISE(ABORT,'place parent cycle') WHERE NEW.id IN (
    WITH RECURSIVE up(id) AS (
      SELECT NEW.parent_id
      UNION SELECT a.parent_id FROM place a JOIN up ON a.id = up.id
        WHERE a.parent_id IS NOT NULL)
    SELECT id FROM up);
END;

CREATE TRIGGER name_fts_place_ins AFTER INSERT ON place
WHEN NEW.name IS NOT NULL BEGIN
  INSERT INTO name_fts(rowid, name) VALUES (NEW.id, NEW.name);
END;
CREATE TRIGGER name_fts_place_upd AFTER UPDATE OF name ON place BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
  INSERT INTO name_fts(rowid, name)
    SELECT NEW.id, NEW.name WHERE NEW.name IS NOT NULL;
END;
CREATE TRIGGER name_fts_place_del AFTER DELETE ON place BEGIN
  DELETE FROM name_fts WHERE rowid = OLD.id;
END;

-- ============ media context: the ONE interpretation ============
-- Derived and rebuildable: raw evidence (blob/file_blob) and source
-- facts (capture, generation, file, file_param) are never replaced by
-- this projection -- it is the application's best current understanding
-- of when, where and how each media item happened, with its BASIS
-- recorded so no date is ever unexplained. Invalidation lives at the
-- source-fact writer seams (db/context.py stale), never as triggers
-- here or on the sources: a source-table trigger referencing a derived
-- table would break the drop-derived-and-reindex contract.
CREATE TABLE derived_media_context (
    file_id             INTEGER PRIMARY KEY REFERENCES file(id) ON DELETE CASCADE,
    -- Coexistence is FACT, never precedence: a photograph that was also
    -- run through a generator has both claims, and `origin` is fully
    -- determined from them by CHECK -- a classification that could
    -- silently erase one fact is the lie this shape forbids.
    has_capture         INTEGER NOT NULL CHECK (has_capture IN (0, 1)),
    has_generation      INTEGER NOT NULL CHECK (has_generation IN (0, 1)),
    origin              TEXT NOT NULL CHECK (origin IN
                          ('captured','generated','mixed','imported')),
    -- TWO time concepts, never one column doing both jobs: `local_at`
    -- is what the human clock said (the wall time a camera or a
    -- generator claimed); `instant_at` is the actual UTC instant,
    -- present ONLY when knowable. An unzoned claim keeps its wall time
    -- and has no instant -- a known human clock is never replaced by a
    -- filesystem time to make a column easier to sort.
    local_at            REAL,
    instant_at          REAL,
    tz_offset_min       INTEGER,
    -- the source that supplied the value; the sources that supported
    -- it and the ones that conflicted, named (db/when.py): a date is
    -- never unexplained and a conflict is never silently resolved.
    -- time_certainty is an ORDINAL's fixed spelling (corroborated .9,
    -- claimed .6, contested .4), not a probability.
    -- `first_seen` was declared here and nothing could ever write it:
    -- `judge_file`'s no-claim branch returns None when mtime and btime
    -- are both absent, so the one case it named produces no row at all,
    -- and derived_media_occurrence.basis below never listed it.
    time_basis          TEXT CHECK (time_basis IN
                          ('capture','embedded','filename','folder','btime','mtime')),
    time_certainty      REAL CHECK (time_certainty BETWEEN 0 AND 1),
    time_supports       TEXT,
    time_conflicts      TEXT,
    -- How FINE the claim is -- orthogonal to certainty: a day-resolution
    -- generator date can be almost certainly the right DAY while saying
    -- nothing about minutes, and a distrusted btime is subsecond-fine.
    -- A coarse claim is REFINED by every consistent signal the file
    -- carries (the finish-implied second inside a claimed minute, the
    -- write inside a claimed day) and the refinement is the moment
    -- shown: a signal not exposed is wasted. Only an estimate that
    -- contradicts its claim is held back, as a named conflict.
    -- The coarse half is not decoration. A scanned photograph's only date
    -- is often the folder it sits in -- `1998/`, `2003-07/`, `1970s/` --
    -- and with nowhere to put that claim the file fell through to mtime,
    -- which dates a 1964 photograph by when somebody last copied it. Six
    -- of the corpus's Commons photographs are pre-1990.
    -- No `hour`. Every precision this column can hold is one `db/when.py`
    -- constructs a Verdict with: a stamped name gives day, second or
    -- subsecond, a folder gives day, month, year or decade, a generator
    -- date gives second, day or minute, and the filesystem fallback gives
    -- subsecond. Nothing reads an hour without also reading its minutes.
    -- The hour remains a real unit in `when.SPAN` and a real timeline bin;
    -- it was never a precision a file could claim.
    time_precision      TEXT CHECK (time_precision IN
                          ('decade','year','month','day','minute','second','subsecond')),
    gps_lat             REAL,
    gps_lon             REAL,
    place_id            INTEGER REFERENCES place(id) ON DELETE SET NULL,
    -- Two, not four. `db/context.py` assigns `authored` when a person has
    -- said where a picture happened and `gps` when the camera wrote a
    -- fix; `sidecar` and `inferred` were vocabulary for readers that do
    -- not exist -- sidecar ingest carries generation parameters and never
    -- a location, and nothing infers one.
    location_basis      TEXT CHECK (location_basis IN
                          ('gps','authored')),
    location_certainty  REAL CHECK (location_certainty BETWEEN 0 AND 1),
    -- WHICH MEANING produced this row: the interpretation ladder's own
    -- version, so a better ladder tomorrow visibly obsoletes today's
    -- rows instead of impersonating them.
    policy_version      INTEGER NOT NULL,
    rebuilt_at          REAL NOT NULL,
    -- a time without a recorded basis is an unexplained date
    CHECK ((time_basis IS NULL) = (local_at IS NULL AND instant_at IS NULL)),
    -- and one without a precision is an unexplained kind of date
    CHECK ((time_basis IS NULL) = (time_precision IS NULL)),
    -- an offset explains a wall clock; without one it explains nothing
    CHECK (tz_offset_min IS NULL OR local_at IS NOT NULL),
    -- origin is DETERMINED, never asserted
    CHECK (origin = CASE
             WHEN has_generation = 1 AND has_capture = 1 THEN 'mixed'
             WHEN has_generation = 1 THEN 'generated'
             WHEN has_capture = 1 THEN 'captured'
             ELSE 'imported' END)
) STRICT;
CREATE INDEX media_context_when ON derived_media_context(instant_at);
CREATE INDEX media_context_local ON derived_media_context(local_at);
CREATE INDEX media_context_place ON derived_media_context(place_id);
CREATE INDEX media_context_origin_when ON derived_media_context(origin, instant_at);

-- One temporal CLAIM of one KIND about one media item. The capture
-- occurrence and the generation occurrence of the SAME file are two
-- different historical acts with two different times: a photograph run
-- through a generator years later tells the capture story at capture
-- time and the generation story at generation time. Groupers consume
-- the occurrence of their own claim -- the context keeps the ONE
-- primary human-timeline interpretation. Derived, rebuilt with the
-- context, stamped with the same policy.
CREATE TABLE derived_media_occurrence (
    file_id        INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    kind           TEXT NOT NULL CHECK (kind IN ('capture','generation','file')),
    -- the same two-domain doctrine as the context: a wall clock when
    -- claimed, an instant only when knowable, never fused
    local_at       REAL,
    instant_at     REAL,
    tz_offset_min  INTEGER,
    -- the CLAIM's source, the sources that supported it and the ones
    -- that conflicted, named (db/when.py). `certainty` is an ordinal's
    -- fixed spelling (corroborated .9, claimed .6, contested .4).
    basis          TEXT NOT NULL CHECK (basis IN
                     ('capture','embedded','filename','folder','mtime','btime')),
    certainty      REAL NOT NULL CHECK (certainty BETWEEN 0 AND 1),
    supports       TEXT,
    conflicts      TEXT,
    -- the filesystem's FINISH instant and the request ESTIMATED from it
    -- (finish minus generation time, a wall-clock reading) -- beside the
    -- claim, never in its place: a grouper sequences by the claim, a
    -- page may show the estimate as inferred
    finished_at    REAL,
    estimated_at   REAL,
    -- the generator's own order inside the claimed bucket (SwarmUI's
    -- per-minute request counter): ordering evidence, never seconds
    source_order   INTEGER,
    -- ONE ACT, several files: a RAW and its JPEG are two renditions of one
    -- shutter press. The key is derived from the body, the capture clock to
    -- the millisecond and the camera's frame name, so renditions share it
    -- wherever they were copied; a grouper counts acts, not files
    act_key        TEXT,
    -- The SAME vocabulary as derived_media_context above, and it has to
    -- be: an occurrence is where a claim is recorded and the context is
    -- what is concluded from it, so a precision the context can hold and
    -- the occurrence cannot is a claim with nowhere to be written. Exactly
    -- that happened -- the coarse rungs were added to one CHECK and not
    -- the other, and every context job item died on `CHECK constraint
    -- failed: time_precision IN` while the context table would have taken
    -- the row.
    time_precision TEXT NOT NULL CHECK (time_precision IN
                     ('decade','year','month','day','minute','second','subsecond')),
    policy_version INTEGER NOT NULL,
    PRIMARY KEY (file_id, kind),
    -- an occurrence with no time is not an occurrence
    CHECK (local_at IS NOT NULL OR instant_at IS NOT NULL),
    CHECK (tz_offset_min IS NULL OR local_at IS NOT NULL)
) STRICT, WITHOUT ROWID;
CREATE INDEX media_occurrence_kind_instant ON derived_media_occurrence(kind, instant_at);
CREATE INDEX media_occurrence_kind_local ON derived_media_occurrence(kind, local_at);
CREATE INDEX media_occurrence_act ON derived_media_occurrence(kind, act_key) WHERE act_key IS NOT NULL;

-- One row: the interpretation's identity. `generation` advances on
-- EVERY context add, change or delete, so anything computed over the
-- contexts can prove it was computed over THESE contexts; the policy
-- says which ladder meaning is current. Derived like its subject: drop
-- the namespace and the first rebuild re-mints both.
CREATE TABLE derived_context_state (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    generation     INTEGER NOT NULL,
    policy_version INTEGER NOT NULL
) STRICT;

-- ============ events: grouping hypotheses over contexts ============
-- A trip or a generation session is a HYPOTHESIS over a set of files,
-- never a property stamped onto them. Runs are rebuildable; membership
-- is hashed so a changed membership is visibly a different event; and
-- every run names the context generation and policy it was computed
-- over -- a run whose generation is no longer current is a stale
-- hypothesis, whoever its members are. A calendar day is deliberately
-- NOT an event kind: days are presentation, read off the contexts.
CREATE TABLE derived_event_run (
    id                     INTEGER PRIMARY KEY,
    grouper                TEXT NOT NULL,
    grouper_version        TEXT NOT NULL,
    settings_hash          TEXT NOT NULL,
    context_generation     INTEGER NOT NULL,
    context_policy_version INTEGER NOT NULL,
    created_at             REAL NOT NULL
) STRICT;
CREATE INDEX event_run_grouper ON derived_event_run(grouper, created_at);

CREATE TABLE derived_event (
    id            INTEGER PRIMARY KEY,
    run_id        INTEGER NOT NULL REFERENCES derived_event_run(id) ON DELETE CASCADE,
    parent_id     INTEGER REFERENCES derived_event(id) ON DELETE CASCADE,
    kind          TEXT NOT NULL CHECK (kind IN ('generation_session','capture_session','file_session')),
    -- The interval carries its TEMPORAL DOMAIN: a wall-clock pair, an
    -- instant pair, or both when every member makes both knowable --
    -- never one ambiguous pair that is secretly sometimes each. Unlike
    -- domains are never subtracted from each other.
    local_start   REAL,
    local_end     REAL,
    instant_start REAL,
    instant_end   REAL,
    place_id      INTEGER REFERENCES place(id) ON DELETE SET NULL,
    confidence    REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    member_hash   TEXT NOT NULL,
    CHECK ((local_start IS NULL) = (local_end IS NULL)),
    CHECK ((instant_start IS NULL) = (instant_end IS NULL)),
    CHECK (local_start IS NULL OR local_start <= local_end),
    CHECK (instant_start IS NULL OR instant_start <= instant_end),
    CHECK (local_start IS NOT NULL OR instant_start IS NOT NULL)
) STRICT;
CREATE INDEX event_run ON derived_event(run_id);
CREATE INDEX event_parent ON derived_event(parent_id);
CREATE INDEX event_when_instant ON derived_event(instant_start);
CREATE INDEX event_when_local ON derived_event(local_start);
CREATE INDEX event_place ON derived_event(place_id);

CREATE TABLE derived_event_file (
    event_id INTEGER NOT NULL REFERENCES derived_event(id) ON DELETE CASCADE,
    file_id  INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    ordinal  INTEGER NOT NULL,
    score    REAL,
    PRIMARY KEY (event_id, file_id)
) STRICT, WITHOUT ROWID;
CREATE INDEX event_file_file ON derived_event_file(file_id);

-- ============ stories: frozen evidence, never prose ============
-- A StorySnapshot is an immutable, self-contained record of exactly
-- what the application knew about ONE current event at ONE instant. It
-- freezes EVIDENCE, not prose: file uuids AND the content hashes it
-- actually observed, the occurrence that placed each member in this
-- event, prompts, artifacts, capture facts, people, place hierarchy,
-- lineage, annotations -- by VALUE. It is deliberately not a foreign
-- key to derived_event/derived_event_run: those are rebuildable
-- hypotheses replaced on every regroup, and a story written from
-- yesterday's evidence must not vanish because today regrouped. Its
-- identity is the canonical document's SHA-256, so identical evidence
-- is one row and retries are idempotent. Insert-only: a change in the
-- library, the policy or the evidence producers means a NEW snapshot.
-- Historical, not derived: dropping the derived namespace leaves it.
CREATE TABLE story_snapshot (
    id                     INTEGER PRIMARY KEY,
    format_version         INTEGER NOT NULL,
    source_kind            TEXT NOT NULL CHECK (source_kind = 'event'),
    event_kind             TEXT NOT NULL CHECK (event_kind IN
                             ('generation_session','capture_session','file_session')),
    grouper                TEXT NOT NULL,
    context_generation     INTEGER NOT NULL,
    context_policy_version INTEGER NOT NULL,
    member_hash            TEXT NOT NULL,
    document_json          TEXT NOT NULL,
    document_sha256        TEXT NOT NULL UNIQUE CHECK (length(document_sha256) = 64),
    created_at             REAL NOT NULL
) STRICT;
CREATE INDEX story_snapshot_member ON story_snapshot(member_hash, created_at);
CREATE TRIGGER story_snapshot_is_immutable BEFORE UPDATE ON story_snapshot
BEGIN
  SELECT RAISE(ABORT,'a story snapshot is immutable; freeze a new one');
END;

-- A StoryPlan is what ONE planner policy made of ONE frozen snapshot:
-- phases, representatives, first-class Claims with evidence references
-- that resolve inside the snapshot, and what the evidence does NOT
-- support. Structure, never prose -- label_hint is the only human-facing
-- field. Identity is the canonical document's SHA-256, which embeds the
-- snapshot's identity, the planner kind/version, the similarity
-- engine/version and the settings, so the same evidence under the same
-- policy is one row and a new policy coexists with the old plan instead
-- of overwriting it. Insert-only; a snapshot's plans go with it.
CREATE TABLE story_plan (
    id                 INTEGER PRIMARY KEY,
    snapshot_id        INTEGER NOT NULL REFERENCES story_snapshot(id) ON DELETE CASCADE,
    format_version     INTEGER NOT NULL,
    planner            TEXT NOT NULL CHECK (planner IN ('generation_history','capture_history','file_history')),
    planner_version    INTEGER NOT NULL,
    similarity         TEXT NOT NULL,
    similarity_version TEXT NOT NULL,
    settings_hash      TEXT NOT NULL,
    -- The REQUEST's identity, known before any model work: snapshot sha,
    -- planner kind/version, engine name/version, settings. Deterministic
    -- planning makes request -> document one-to-one, so the same request
    -- asked twice reuses the row -- and the queued job -- without
    -- embedding anything again. document_sha256 stays the OUTPUT identity.
    request_sha256     TEXT NOT NULL UNIQUE CHECK (length(request_sha256) = 64),
    document_json      TEXT NOT NULL,
    document_sha256    TEXT NOT NULL UNIQUE CHECK (length(document_sha256) = 64),
    created_at         REAL NOT NULL
) STRICT;
CREATE INDEX story_plan_snapshot ON story_plan(snapshot_id, created_at);
CREATE TRIGGER story_plan_is_immutable BEFORE UPDATE ON story_plan
BEGIN
  SELECT RAISE(ABORT,'a story plan is immutable; plan again under a new policy');
END;

-- A StoryRender is what ONE renderer policy said about ONE plan:
-- structured narration (a lede with its support, one section per plan
-- phase made of support-bearing blocks, notes for what the evidence
-- does not support) -- never HTML or Markdown, which are presentation
-- encodings a later page lays out. request_sha256 is the identity of
-- the ask (format, plan sha, snapshot sha, renderer kind/version,
-- profile, locale, render policy), known before any work;
-- document_sha256 is the canonical output. The snapshot is the plan's:
-- a second snapshot column here was a second source for one fact with
-- nothing holding the two equal. Insert-only; a plan's renders go with it.
CREATE TABLE story_render (
    id               INTEGER PRIMARY KEY,
    plan_id          INTEGER NOT NULL REFERENCES story_plan(id) ON DELETE CASCADE,
    format_version   INTEGER NOT NULL,
    renderer         TEXT NOT NULL CHECK (renderer IN ('template')),
    renderer_version INTEGER NOT NULL,
    profile          TEXT NOT NULL CHECK (profile IN ('memory','technical','compact')),
    locale           TEXT NOT NULL CHECK (locale IN ('en')),
    render_policy    INTEGER NOT NULL,
    request_sha256   TEXT NOT NULL UNIQUE CHECK (length(request_sha256) = 64),
    document_json    TEXT NOT NULL,
    document_sha256  TEXT NOT NULL UNIQUE CHECK (length(document_sha256) = 64),
    created_at       REAL NOT NULL
) STRICT;
CREATE INDEX story_render_plan ON story_render(plan_id, created_at);
CREATE TRIGGER story_render_is_immutable BEFORE UPDATE ON story_render
BEGIN
  SELECT RAISE(ABORT,'a story render is immutable; render again under a new policy');
END;

-- The generation of everything an ANSWER can be computed from.
--
-- db/resultset.py caches the whole ordered answer and pages it by
-- slicing, valid for one (question, library state) pair. That state was
-- `PRAGMA data_version`, which bumps when any connection commits
-- anything -- the same mechanism FTS5 uses for its own structure cache
-- (sqlite/sqlite ext/fts5/fts5_index.c fts5IndexDataVersion). FTS5 can
-- afford it because the thing it re-reads is one small record. Ours is
-- the whole library, so at 80,000 files a page cost 0.18 ms at rest and
-- 38 ms while anything wrote -- 214x -- and jobs commit per item.
--
-- Traced, the job that runs for hours writes ONLY the ledger: a thumbs
-- pass over 12 files made 180 writes, all of them `job`, `job_item` and
-- `job_event`, none able to change any answer.
--
-- So this counter moves for every table EXCEPT those three. Stated that
-- way round on purpose: a table wrongly included costs a little speed,
-- a table wrongly excluded serves a stale answer, which is the one
-- failure the currency contract exists to prevent. The FTS tables are
-- absent because a virtual table cannot carry a trigger and its rows
-- only change when `file` or `folder` do, which are here.
--
-- tests/test_an_answer_knows_when_it_is_stale.py holds the coverage, so
-- a table added later without its triggers fails the gate rather than
-- quietly stopping invalidation. Measured cost on the ingest path:
-- 1.02x over 20,000 inserts.
CREATE TABLE answer_generation (
    id    INTEGER PRIMARY KEY CHECK (id = 1),
    value INTEGER NOT NULL
) STRICT;
INSERT INTO answer_generation(id, value) VALUES(1, 0);

CREATE TRIGGER answer_moved_artifact_ins AFTER INSERT ON artifact BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_artifact_upd AFTER UPDATE ON artifact BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_artifact_del AFTER DELETE ON artifact BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_blob_ins AFTER INSERT ON blob BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_blob_upd AFTER UPDATE ON blob BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_blob_del AFTER DELETE ON blob BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_capture_ins AFTER INSERT ON capture BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_capture_upd AFTER UPDATE ON capture BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_capture_del AFTER DELETE ON capture BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_collection_ins AFTER INSERT ON collection BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_collection_upd AFTER UPDATE ON collection BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_collection_del AFTER DELETE ON collection BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_collection_file_ins AFTER INSERT ON collection_file BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_collection_file_upd AFTER UPDATE ON collection_file BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_collection_file_del AFTER DELETE ON collection_file BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_collection_rule_ins AFTER INSERT ON collection_rule BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_collection_rule_upd AFTER UPDATE ON collection_rule BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_collection_rule_del AFTER DELETE ON collection_rule BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_comment_ins AFTER INSERT ON comment BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_comment_upd AFTER UPDATE ON comment BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_comment_del AFTER DELETE ON comment BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derivation_intent_ins AFTER INSERT ON derivation_intent BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derivation_intent_upd AFTER UPDATE ON derivation_intent BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derivation_intent_del AFTER DELETE ON derivation_intent BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_annotation_ins AFTER INSERT ON derived_annotation BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_annotation_upd AFTER UPDATE ON derived_annotation BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_annotation_del AFTER DELETE ON derived_annotation BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_context_state_ins AFTER INSERT ON derived_context_state BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_context_state_upd AFTER UPDATE ON derived_context_state BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_context_state_del AFTER DELETE ON derived_context_state BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_dupe_group_ins AFTER INSERT ON derived_dupe_group BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_dupe_group_upd AFTER UPDATE ON derived_dupe_group BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_dupe_group_del AFTER DELETE ON derived_dupe_group BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_embedding_ins AFTER INSERT ON derived_embedding BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_embedding_upd AFTER UPDATE ON derived_embedding BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_embedding_del AFTER DELETE ON derived_embedding BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_event_ins AFTER INSERT ON derived_event BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_event_upd AFTER UPDATE ON derived_event BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_event_del AFTER DELETE ON derived_event BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_event_file_ins AFTER INSERT ON derived_event_file BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_event_file_upd AFTER UPDATE ON derived_event_file BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_event_file_del AFTER DELETE ON derived_event_file BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_event_run_ins AFTER INSERT ON derived_event_run BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_event_run_upd AFTER UPDATE ON derived_event_run BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_event_run_del AFTER DELETE ON derived_event_run BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_face_cluster_ins AFTER INSERT ON derived_face_cluster BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_face_cluster_upd AFTER UPDATE ON derived_face_cluster BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_face_cluster_del AFTER DELETE ON derived_face_cluster BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_face_instance_ins AFTER INSERT ON derived_face_instance BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_face_instance_upd AFTER UPDATE ON derived_face_instance BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_face_instance_del AFTER DELETE ON derived_face_instance BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_face_membership_ins AFTER INSERT ON derived_face_membership BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_face_membership_upd AFTER UPDATE ON derived_face_membership BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_face_membership_del AFTER DELETE ON derived_face_membership BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_face_run_ins AFTER INSERT ON derived_face_run BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_face_run_upd AFTER UPDATE ON derived_face_run BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_face_run_del AFTER DELETE ON derived_face_run BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_face_scan_ins AFTER INSERT ON derived_face_scan BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_face_scan_upd AFTER UPDATE ON derived_face_scan BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_face_scan_del AFTER DELETE ON derived_face_scan BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_file_hash_ins AFTER INSERT ON derived_file_hash BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_file_hash_upd AFTER UPDATE ON derived_file_hash BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_file_hash_del AFTER DELETE ON derived_file_hash BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_file_person_ins AFTER INSERT ON derived_file_person BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_file_person_upd AFTER UPDATE ON derived_file_person BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_file_person_del AFTER DELETE ON derived_file_person BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_media_context_ins AFTER INSERT ON derived_media_context BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_media_context_upd AFTER UPDATE ON derived_media_context BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_media_context_del AFTER DELETE ON derived_media_context BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_media_occurrence_ins AFTER INSERT ON derived_media_occurrence BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_media_occurrence_upd AFTER UPDATE ON derived_media_occurrence BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_media_occurrence_del AFTER DELETE ON derived_media_occurrence BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_media_sample_ins AFTER INSERT ON derived_media_sample BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_media_sample_upd AFTER UPDATE ON derived_media_sample BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_media_sample_del AFTER DELETE ON derived_media_sample BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_prompt_embedding_ins AFTER INSERT ON derived_prompt_embedding BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_prompt_embedding_upd AFTER UPDATE ON derived_prompt_embedding BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_prompt_embedding_del AFTER DELETE ON derived_prompt_embedding BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_prompt_section_ins AFTER INSERT ON derived_prompt_section BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_prompt_section_upd AFTER UPDATE ON derived_prompt_section BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_derived_prompt_section_del AFTER DELETE ON derived_prompt_section BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_entity_ins AFTER INSERT ON entity BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_entity_upd AFTER UPDATE ON entity BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_entity_del AFTER DELETE ON entity BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_favorite_ins AFTER INSERT ON favorite BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_favorite_upd AFTER UPDATE ON favorite BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_favorite_del AFTER DELETE ON favorite BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_feedback_ins AFTER INSERT ON feedback BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_feedback_upd AFTER UPDATE ON feedback BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_feedback_del AFTER DELETE ON feedback BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_ins AFTER INSERT ON file BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_upd AFTER UPDATE ON file BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_del AFTER DELETE ON file BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_artifact_ins AFTER INSERT ON file_artifact BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_artifact_upd AFTER UPDATE ON file_artifact BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_artifact_del AFTER DELETE ON file_artifact BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_blob_ins AFTER INSERT ON file_blob BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_blob_upd AFTER UPDATE ON file_blob BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_blob_del AFTER DELETE ON file_blob BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_derivation_ins AFTER INSERT ON file_derivation BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_derivation_upd AFTER UPDATE ON file_derivation BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_derivation_del AFTER DELETE ON file_derivation BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_param_ins AFTER INSERT ON file_param BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_param_upd AFTER UPDATE ON file_param BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_param_del AFTER DELETE ON file_param BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_place_ins AFTER INSERT ON file_place BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_place_upd AFTER UPDATE ON file_place BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_place_del AFTER DELETE ON file_place BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_relation_ins AFTER INSERT ON file_relation BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_relation_upd AFTER UPDATE ON file_relation BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_relation_del AFTER DELETE ON file_relation BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_tag_ins AFTER INSERT ON file_tag BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_tag_upd AFTER UPDATE ON file_tag BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_file_tag_del AFTER DELETE ON file_tag BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_folder_ins AFTER INSERT ON folder BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_folder_upd AFTER UPDATE ON folder BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_folder_del AFTER DELETE ON folder BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_generation_ins AFTER INSERT ON generation BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_generation_upd AFTER UPDATE ON generation BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_generation_del AFTER DELETE ON generation BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_generation_prompt_ins AFTER INSERT ON generation_prompt BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_generation_prompt_upd AFTER UPDATE ON generation_prompt BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_generation_prompt_del AFTER DELETE ON generation_prompt BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_param_key_ins AFTER INSERT ON param_key BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_param_key_upd AFTER UPDATE ON param_key BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_param_key_del AFTER DELETE ON param_key BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_person_ins AFTER INSERT ON person BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_person_upd AFTER UPDATE ON person BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_person_del AFTER DELETE ON person BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_person_assertion_ins AFTER INSERT ON person_assertion BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_person_assertion_upd AFTER UPDATE ON person_assertion BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_person_assertion_del AFTER DELETE ON person_assertion BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_place_ins AFTER INSERT ON place BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_place_upd AFTER UPDATE ON place BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_place_del AFTER DELETE ON place BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_prompt_ins AFTER INSERT ON prompt BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_prompt_upd AFTER UPDATE ON prompt BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_prompt_del AFTER DELETE ON prompt BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_rating_ins AFTER INSERT ON rating BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_rating_upd AFTER UPDATE ON rating BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_rating_del AFTER DELETE ON rating BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_region_ins AFTER INSERT ON region BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_region_upd AFTER UPDATE ON region BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_region_del AFTER DELETE ON region BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_root_ins AFTER INSERT ON root BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_root_upd AFTER UPDATE ON root BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_root_del AFTER DELETE ON root BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_setting_ins AFTER INSERT ON setting BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_setting_upd AFTER UPDATE ON setting BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_setting_del AFTER DELETE ON setting BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_similarity_space_ins AFTER INSERT ON similarity_space BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_similarity_space_upd AFTER UPDATE ON similarity_space BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_similarity_space_del AFTER DELETE ON similarity_space BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_slug_history_ins AFTER INSERT ON slug_history BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_slug_history_upd AFTER UPDATE ON slug_history BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_slug_history_del AFTER DELETE ON slug_history BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_story_plan_ins AFTER INSERT ON story_plan BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_story_plan_upd AFTER UPDATE ON story_plan BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_story_plan_del AFTER DELETE ON story_plan BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_story_render_ins AFTER INSERT ON story_render BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_story_render_upd AFTER UPDATE ON story_render BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_story_render_del AFTER DELETE ON story_render BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_story_snapshot_ins AFTER INSERT ON story_snapshot BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_story_snapshot_upd AFTER UPDATE ON story_snapshot BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_story_snapshot_del AFTER DELETE ON story_snapshot BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_tag_ins AFTER INSERT ON tag BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_tag_upd AFTER UPDATE ON tag BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_tag_del AFTER DELETE ON tag BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_user_ins AFTER INSERT ON user BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_user_upd AFTER UPDATE ON user BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_user_del AFTER DELETE ON user BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_watched_folder_ins AFTER INSERT ON watched_folder BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_watched_folder_upd AFTER UPDATE ON watched_folder BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;
CREATE TRIGGER answer_moved_watched_folder_del AFTER DELETE ON watched_folder BEGIN UPDATE answer_generation SET value = value + 1 WHERE id = 1; END;


PRAGMA application_id = 0x53474C59;
PRAGMA user_version   = 47;

-- ============ the entity registry must agree with its subtypes ============
-- The foreign key proves the entity row exists; nothing tied entity.kind to the
-- table actually holding the row. A file could be created against an entity
-- declaring kind='person', one id could sit in two subtype tables, and deleting
-- a subtype row left the entity behind -- an address that resolves to nothing
-- and permanently squats its slug.

CREATE TRIGGER file_kind_agrees BEFORE INSERT ON file BEGIN
  SELECT RAISE(ABORT,'entity kind does not match file')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'file';
END;
-- The same rule on UPDATE. Moving a subtype row onto a different entity
-- was accepted whenever nothing referenced it, which left a file row
-- sitting on an entity of another kind and nothing reporting it.
CREATE TRIGGER file_kind_keeps_agreeing BEFORE UPDATE OF id ON file BEGIN
  SELECT RAISE(ABORT,'entity kind does not match file')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'file';
END;
CREATE TRIGGER file_takes_its_entity AFTER DELETE ON file BEGIN
  DELETE FROM entity WHERE id = OLD.id;
END;

CREATE TRIGGER folder_kind_agrees BEFORE INSERT ON folder BEGIN
  SELECT RAISE(ABORT,'entity kind does not match folder')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'folder';
END;
-- The same rule on UPDATE. Moving a subtype row onto a different entity
-- was accepted whenever nothing referenced it, which left a folder row
-- sitting on an entity of another kind and nothing reporting it.
CREATE TRIGGER folder_kind_keeps_agreeing BEFORE UPDATE OF id ON folder BEGIN
  SELECT RAISE(ABORT,'entity kind does not match folder')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'folder';
END;
CREATE TRIGGER folder_takes_its_entity AFTER DELETE ON folder BEGIN
  DELETE FROM entity WHERE id = OLD.id;
END;

CREATE TRIGGER person_kind_agrees BEFORE INSERT ON person BEGIN
  SELECT RAISE(ABORT,'entity kind does not match person')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'person';
END;
-- The same rule on UPDATE. Moving a subtype row onto a different entity
-- was accepted whenever nothing referenced it, which left a person row
-- sitting on an entity of another kind and nothing reporting it.
CREATE TRIGGER person_kind_keeps_agreeing BEFORE UPDATE OF id ON person BEGIN
  SELECT RAISE(ABORT,'entity kind does not match person')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'person';
END;
CREATE TRIGGER person_takes_its_entity AFTER DELETE ON person BEGIN
  DELETE FROM entity WHERE id = OLD.id;
END;

CREATE TRIGGER artifact_kind_agrees BEFORE INSERT ON artifact BEGIN
  SELECT RAISE(ABORT,'entity kind does not match artifact')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'artifact';
END;
-- The same rule on UPDATE. Moving a subtype row onto a different entity
-- was accepted whenever nothing referenced it, which left a artifact row
-- sitting on an entity of another kind and nothing reporting it.
CREATE TRIGGER artifact_kind_keeps_agreeing BEFORE UPDATE OF id ON artifact BEGIN
  SELECT RAISE(ABORT,'entity kind does not match artifact')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'artifact';
END;
CREATE TRIGGER artifact_takes_its_entity AFTER DELETE ON artifact BEGIN
  DELETE FROM entity WHERE id = OLD.id;
END;

CREATE TRIGGER prompt_kind_agrees BEFORE INSERT ON prompt BEGIN
  SELECT RAISE(ABORT,'entity kind does not match prompt')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'prompt';
END;
-- The same rule on UPDATE. Moving a subtype row onto a different entity
-- was accepted whenever nothing referenced it, which left a prompt row
-- sitting on an entity of another kind and nothing reporting it.
CREATE TRIGGER prompt_kind_keeps_agreeing BEFORE UPDATE OF id ON prompt BEGIN
  SELECT RAISE(ABORT,'entity kind does not match prompt')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'prompt';
END;
CREATE TRIGGER prompt_takes_its_entity AFTER DELETE ON prompt BEGIN
  DELETE FROM entity WHERE id = OLD.id;
END;

CREATE TRIGGER collection_kind_agrees BEFORE INSERT ON collection BEGIN
  SELECT RAISE(ABORT,'entity kind does not match collection')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'collection';
END;
-- The same rule on UPDATE. Moving a subtype row onto a different entity
-- was accepted whenever nothing referenced it, which left a collection row
-- sitting on an entity of another kind and nothing reporting it.
CREATE TRIGGER collection_kind_keeps_agreeing BEFORE UPDATE OF id ON collection BEGIN
  SELECT RAISE(ABORT,'entity kind does not match collection')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'collection';
END;
CREATE TRIGGER collection_takes_its_entity AFTER DELETE ON collection BEGIN
  DELETE FROM entity WHERE id = OLD.id;
END;

-- A smart collection's members are what its rule says, freshly, every
-- time. A stored member row would give it a second, disagreeing answer --
-- refused on the way in, refused as a destination, and a filled
-- collection cannot quietly BECOME smart for the same reason.
CREATE TRIGGER collection_file_not_into_smart BEFORE INSERT ON collection_file
WHEN (SELECT kind FROM collection WHERE id = NEW.collection_id) = 'smart' BEGIN
  SELECT RAISE(ABORT,'a smart collection derives its members from its rule; nothing is filed into it');
END;
CREATE TRIGGER collection_file_not_moved_into_smart BEFORE UPDATE OF collection_id ON collection_file
WHEN (SELECT kind FROM collection WHERE id = NEW.collection_id) = 'smart' BEGIN
  SELECT RAISE(ABORT,'a smart collection derives its members from its rule; nothing is filed into it');
END;
CREATE TRIGGER collection_with_members_stays_listed BEFORE UPDATE OF kind ON collection
WHEN NEW.kind = 'smart' AND OLD.kind <> 'smart'
 AND EXISTS (SELECT 1 FROM collection_file WHERE collection_id = NEW.id) BEGIN
  SELECT RAISE(ABORT,'this collection holds filed members; empty it before making it smart');
END;

-- The mirror-image guards: exactly ONE membership definition per
-- collection. A rule belongs only to a smart collection -- an album with
-- filed rows AND a dormant rule is two authored answers waiting to
-- disagree -- and a rule-carrying smart collection cannot quietly become
-- listed with its rule still attached. Deleting the rule first is the
-- deliberate act that makes the transition honest.
CREATE TRIGGER collection_rule_only_on_smart BEFORE INSERT ON collection_rule
WHEN (SELECT kind FROM collection WHERE id = NEW.collection_id) <> 'smart' BEGIN
  SELECT RAISE(ABORT,'only a smart collection carries a rule; a listed collection''s membership is its filed rows');
END;
CREATE TRIGGER collection_rule_stays_on_smart BEFORE UPDATE OF collection_id ON collection_rule
WHEN (SELECT kind FROM collection WHERE id = NEW.collection_id) <> 'smart' BEGIN
  SELECT RAISE(ABORT,'only a smart collection carries a rule; a listed collection''s membership is its filed rows');
END;
CREATE TRIGGER collection_with_rule_stays_smart BEFORE UPDATE OF kind ON collection
WHEN OLD.kind = 'smart' AND NEW.kind <> 'smart'
 AND EXISTS (SELECT 1 FROM collection_rule WHERE collection_id = NEW.id) BEGIN
  SELECT RAISE(ABORT,'this collection is rule-defined; delete its rule before making it listed');
END;

-- ============ a reference must agree with what it points at ============
-- A declared foreign key proves the row exists; it says nothing about whether
-- the row is the right *kind* of thing. Without these, a camera could be
-- recorded as a file's checkpoint, a lens could be its workflow, a face could
-- cite a frame sampled from a different file, and a finding could sit on a
-- file its own review never looked at. All four were accepted.

CREATE TRIGGER file_artifact_role_matches_kind BEFORE INSERT ON file_artifact BEGIN
  SELECT RAISE(ABORT,'artifact kind does not match the role it is used in')
  WHERE (SELECT kind FROM artifact WHERE id = NEW.artifact_id) <> CASE NEW.role
      WHEN 'checkpoint'    THEN 'checkpoint'
      WHEN 'refiner'       THEN 'checkpoint'
      WHEN 'lora'          THEN 'lora'
      WHEN 'vae'           THEN 'vae'
      WHEN 'controlnet'    THEN 'controlnet'
      WHEN 'upscaler'      THEN 'upscaler'
      WHEN 'embedding'     THEN 'embedding'
      WHEN 'hypernetwork'  THEN 'hypernetwork'
      WHEN 'ip_adapter'    THEN 'ip_adapter'
      WHEN 'text_encoder'  THEN 'text_encoder'
      WHEN 'unet'          THEN 'unet'
      WHEN 'captured_with' THEN 'camera'
      WHEN 'mounted_lens'  THEN 'lens'
      WHEN 'produced_by'   THEN 'application'
    END;
END;

-- The same rule on UPDATE. Without it the check was one statement away from
-- useless: insert a camera as 'captured_with', then UPDATE the role to
-- 'checkpoint' and it stands.
CREATE TRIGGER file_artifact_role_keeps_matching
BEFORE UPDATE OF role, artifact_id ON file_artifact BEGIN
  SELECT RAISE(ABORT,'artifact kind does not match the role it is used in')
  WHERE (SELECT kind FROM artifact WHERE id = NEW.artifact_id) <> CASE NEW.role
      WHEN 'checkpoint'    THEN 'checkpoint'
      WHEN 'refiner'       THEN 'checkpoint'
      WHEN 'lora'          THEN 'lora'
      WHEN 'vae'           THEN 'vae'
      WHEN 'controlnet'    THEN 'controlnet'
      WHEN 'upscaler'      THEN 'upscaler'
      WHEN 'embedding'     THEN 'embedding'
      WHEN 'hypernetwork'  THEN 'hypernetwork'
      WHEN 'ip_adapter'    THEN 'ip_adapter'
      WHEN 'text_encoder'  THEN 'text_encoder'
      WHEN 'unet'          THEN 'unet'
      WHEN 'captured_with' THEN 'camera'
      WHEN 'mounted_lens'  THEN 'lens'
      WHEN 'produced_by'   THEN 'application'
    END;
END;

CREATE TRIGGER generation_workflow_is_a_workflow BEFORE INSERT ON generation
WHEN NEW.workflow_id IS NOT NULL BEGIN
  SELECT RAISE(ABORT,'generation.workflow_id must name an artifact of kind workflow')
  WHERE (SELECT kind FROM artifact WHERE id = NEW.workflow_id) <> 'workflow';
END;

CREATE TRIGGER generation_workflow_stays_a_workflow
BEFORE UPDATE OF workflow_id ON generation
WHEN NEW.workflow_id IS NOT NULL BEGIN
  SELECT RAISE(ABORT,'generation.workflow_id must name an artifact of kind workflow')
  WHERE (SELECT kind FROM artifact WHERE id = NEW.workflow_id) <> 'workflow';
END;

CREATE TRIGGER face_sample_belongs_to_its_file BEFORE INSERT ON derived_face_instance
WHEN NEW.sample_id IS NOT NULL BEGIN
  SELECT RAISE(ABORT,'face cites a sample from a different file')
  WHERE (SELECT file_id FROM derived_media_sample WHERE id = NEW.sample_id) <> NEW.file_id;
END;

CREATE TRIGGER face_sample_stays_with_its_file
BEFORE UPDATE OF sample_id, file_id ON derived_face_instance
WHEN NEW.sample_id IS NOT NULL BEGIN
  SELECT RAISE(ABORT,'face cites a sample from a different file')
  WHERE (SELECT file_id FROM derived_media_sample WHERE id = NEW.sample_id) <> NEW.file_id;
END;

-- An annotation may cite a frame, and that frame has to be a frame of the
-- file being annotated. Without this a caption can quote a moment from a
-- different video, and the evidence link reads as sound.
CREATE TRIGGER annotation_sample_belongs_to_its_file
BEFORE INSERT ON derived_annotation WHEN NEW.sample_id IS NOT NULL BEGIN
  SELECT RAISE(ABORT,'annotation cites a sample from another file')
  WHERE (SELECT file_id FROM derived_media_sample WHERE id = NEW.sample_id)
        <> NEW.file_id;
END;

CREATE TRIGGER annotation_sample_stays_with_its_file
BEFORE UPDATE OF sample_id, file_id ON derived_annotation
WHEN NEW.sample_id IS NOT NULL BEGIN
  SELECT RAISE(ABORT,'annotation cites a sample from another file')
  WHERE (SELECT file_id FROM derived_media_sample WHERE id = NEW.sample_id)
        <> NEW.file_id;
END;

-- ============ folder.depth is derived, so the database derives it ============
-- It was a denormalised column maintained by nothing: a reparented folder kept
-- its old depth, and the only reason a test noticed its removal was that a
-- fixture named it in an INSERT.
CREATE TRIGGER folder_depth_ins AFTER INSERT ON folder BEGIN
  UPDATE folder
     SET depth = COALESCE((SELECT depth + 1 FROM folder WHERE id = NEW.parent_id), 0)
   WHERE id = NEW.id;
END;

-- Reparenting moves a subtree, not a folder. Fixing only the folder that moved
-- left every descendant one level wrong -- root/mid/leaf, move `mid` one level
-- down, and `mid` became 2 while `leaf` stayed 2. The trigger cannot recurse
-- into itself: it fires on UPDATE OF parent_id and writes depth.
--
-- So it walks the subtree once. `parent_id` is not what this statement writes,
-- so the CTE reads a stable tree, and the new parent is outside the subtree --
-- folder_no_cycle refuses the alternative before this ever runs.
CREATE TRIGGER folder_depth_upd AFTER UPDATE OF parent_id ON folder BEGIN
  UPDATE folder
     SET depth = COALESCE((SELECT p.depth + 1 FROM folder p WHERE p.id = NEW.parent_id), 0)
               + (WITH RECURSIVE below(id, distance) AS (
                    SELECT NEW.id, 0
                    UNION ALL
                    SELECT f.id, below.distance + 1
                      FROM folder f JOIN below ON f.parent_id = below.id)
                  SELECT distance FROM below WHERE below.id = folder.id)
   WHERE id IN (WITH RECURSIVE below(id) AS (
                  SELECT NEW.id
                  UNION ALL
                  SELECT f.id FROM folder f JOIN below ON f.parent_id = below.id)
                SELECT id FROM below);
END;
