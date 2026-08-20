PRAGMA foreign_keys = ON;

-- ============ addressable entity supertype ============
-- id is the rowid: the on-disk key, used by every FK and join in this schema.
-- uuid is portable identity for export / merge / import across databases,
-- stored as a 16-byte BLOB rather than 36-char text. It is never a join key
-- and never appears in a URL (that is what entity.slug is for).
CREATE TABLE entity (
    id   INTEGER PRIMARY KEY,
    uuid BLOB NOT NULL UNIQUE CHECK (length(uuid) = 16),
    kind TEXT NOT NULL CHECK (kind IN
           ('file','folder','person','artifact','prompt','collection')),
    slug TEXT NOT NULL,
    UNIQUE (kind, slug)
) STRICT;

-- composite PK, tiny rows: WITHOUT ROWID per sqlite.org/withoutrowid.html
CREATE TABLE slug_history (
    -- The same list `entity.kind` is held to. Unconstrained, a retirement
    -- could name a kind no entity can ever be, and that address then
    -- resolves to nothing for the rest of the library's life.
    kind       TEXT    NOT NULL CHECK (kind IN
                 ('file','folder','person','artifact','prompt','collection')),
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
    path          TEXT    NOT NULL UNIQUE,
    -- 'trash' is a real location, not a state: a deleted file's bytes are
    -- still somewhere, restore is a move, and views exclude the subtree by
    -- ancestry rather than by matching paths against a configured string.
    kind          TEXT    NOT NULL CHECK (kind IN ('library','mount','trash')),
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
    inode     INTEGER
) STRICT;
CREATE UNIQUE INDEX folder_root_unique  ON folder(root_id, name)   WHERE parent_id IS NULL;
CREATE UNIQUE INDEX folder_child_unique ON folder(parent_id, name) WHERE parent_id IS NOT NULL;
CREATE INDEX folder_parent ON folder(parent_id);
CREATE UNIQUE INDEX folder_inode ON folder(root_id, inode) WHERE inode IS NOT NULL;

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
    inode          INTEGER,
    content_sha256 TEXT,
    -- The pixels actually on disk, not what any recipe asked for; see
    -- `generation.width`, which is the request and may differ.
    width          INTEGER,
    height         INTEGER,
    -- Seconds. NOTHING WRITES THIS YET: it needs a container probe, so every
    -- video currently reads as having no length.
    duration       REAL,
    first_seen_at  REAL    NOT NULL,
    last_seen_at   REAL    NOT NULL,
    -- NULL while the bytes are present. Set when a scan cannot find them, or
    -- when a content match was ambiguous. Deletion is then a deliberate act
    -- rather than a scan side effect: unreachable is not the same as deleted.
    missing_since  REAL
) STRICT;
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
CREATE INDEX file_recent ON file(mtime DESC) WHERE missing_since IS NULL;
CREATE INDEX file_added  ON file(first_seen_at DESC) WHERE missing_since IS NULL;
CREATE INDEX file_sha  ON file(content_sha256);
CREATE INDEX file_kind ON file(kind);

-- ============ entity subtypes (PK is entity.id) ============
CREATE TABLE person (
    id         INTEGER PRIMARY KEY REFERENCES entity(id) ON DELETE CASCADE,
    name       TEXT,
    created_at REAL NOT NULL
) STRICT;

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
CREATE INDEX artifact_kind ON artifact(kind);

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
    parent_id   INTEGER REFERENCES collection(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('album','flag','smart')),
    color       TEXT,
    description TEXT,
    sql_text    TEXT,   -- smart only
    nl_text     TEXT,   -- smart only
    created_at  REAL NOT NULL,
    CHECK (kind = 'smart' OR (sql_text IS NULL AND nl_text IS NULL))
) STRICT;
CREATE INDEX collection_parent ON collection(parent_id);

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
    x         REAL NOT NULL,
    y         REAL NOT NULL,
    w         REAL NOT NULL,
    h         REAL NOT NULL,
    mask_hash TEXT REFERENCES blob(hash) ON DELETE SET NULL,
    CHECK (w > 0 AND h > 0),
    -- a hair over 1 absorbs float error from a pixel->fraction conversion
    -- without admitting a box that is genuinely outside the frame
    CHECK (x >= 0 AND y >= 0 AND x + w <= 1.001 AND y + h <= 1.001)
) STRICT;
CREATE INDEX region_mask ON region(mask_hash) WHERE mask_hash IS NOT NULL;

-- ============ evidence locator (video/document faces) ============
-- Named derived_: a sampling policy produced these rows, so "drop the derived
-- namespace and re-index" must take them. Leaving them behind left face
-- instances citing frames whose policy no longer existed.
CREATE TABLE derived_media_sample (
    id         INTEGER PRIMARY KEY,
    file_id    INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
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
CREATE TABLE derived_file_person (
    file_id       INTEGER NOT NULL REFERENCES file(id)   ON DELETE CASCADE,
    person_id     INTEGER NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    model_id      TEXT NOT NULL,
    model_version TEXT NOT NULL,
    face_count    INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (file_id, person_id, model_id, model_version)
) STRICT, WITHOUT ROWID;
CREATE INDEX derived_file_person_person ON derived_file_person(person_id);

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
    kind             TEXT NOT NULL CHECK (kind IN
                       ('scan','hash','embed','detect_faces','cluster_faces',
                        'sample_frames','annotate','remix','zip')),
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
) STRICT;
CREATE INDEX job_state ON job(state);

CREATE TABLE job_item (
    job_id  INTEGER NOT NULL REFERENCES job(id) ON DELETE CASCADE,
    item_id INTEGER NOT NULL,
    state   TEXT NOT NULL CHECK (state IN ('pending','done','failed')),
    error   TEXT,
    PRIMARY KEY (job_id, item_id)
) STRICT, WITHOUT ROWID;

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
    prompt_id   INTEGER REFERENCES prompt(id)   ON DELETE SET NULL,
    negative_id INTEGER REFERENCES prompt(id)   ON DELETE SET NULL,
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
CREATE INDEX generation_prompt   ON generation(prompt_id);
CREATE INDEX generation_seed     ON generation(seed);

-- ============ capture: EXIF, for files a camera made ============
-- A photograph is not "generated". It has its own origin story, and the app
-- currently reads EXIF only as a smuggling channel for workflow JSON
-- (metaparse/containers.py:17), so none of this was recoverable before.
-- Camera and lens are artifacts, not strings, so /camera/<slug> is a page.
CREATE TABLE capture (
    file_id       INTEGER PRIMARY KEY REFERENCES file(id) ON DELETE CASCADE,
    captured_at   REAL,          -- EXIF DateTimeOriginal; NOT file mtime
    tz_offset_min INTEGER,       -- OffsetTimeOriginal, so "the viewer's day" is answerable
    iso           INTEGER,
    f_number      REAL,
    exposure_time REAL,          -- seconds
    focal_length  REAL,          -- mm
    focal_35mm    REAL,
    orientation   INTEGER,
    gps_lat       REAL,
    gps_lon       REAL,
    gps_alt       REAL,
    parsed_at     REAL NOT NULL
) STRICT;
CREATE INDEX capture_when ON capture(captured_at);
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
CREATE TRIGGER param_key_learn AFTER INSERT ON file_param BEGIN
  INSERT INTO param_key(source,key,value_kind,occurrences,first_seen_at,last_seen_at)
  VALUES (NEW.source, NEW.key, 'text', 0, unixepoch(), unixepoch())
  ON CONFLICT(source,key) DO NOTHING;
  UPDATE param_key SET
    occurrences  = (SELECT count(*) FROM file_param
                     WHERE source = NEW.source AND key = NEW.key),
    value_kind   = (SELECT CASE
                      WHEN count(*) FILTER (WHERE value_num IS NULL) = 0 THEN 'number'
                      WHEN count(*) FILTER (WHERE value_num IS NOT NULL) = 0 THEN 'text'
                      ELSE 'mixed' END
                    FROM file_param WHERE source = NEW.source AND key = NEW.key),
    last_seen_at = unixepoch()
  WHERE source = NEW.source AND key = NEW.key;
END;

CREATE TRIGGER param_key_relearn AFTER UPDATE ON file_param BEGIN
  INSERT INTO param_key(source,key,value_kind,occurrences,first_seen_at,last_seen_at)
  VALUES (NEW.source, NEW.key, 'text', 0, unixepoch(), unixepoch())
  ON CONFLICT(source,key) DO NOTHING;
  UPDATE param_key SET
    occurrences = (SELECT count(*) FROM file_param
                    WHERE source = param_key.source AND key = param_key.key),
    value_kind  = (SELECT CASE
                     WHEN count(*) FILTER (WHERE value_num IS NULL) = 0 THEN 'number'
                     WHEN count(*) FILTER (WHERE value_num IS NOT NULL) = 0 THEN 'text'
                     ELSE 'mixed' END
                   FROM file_param WHERE source = param_key.source AND key = param_key.key),
    last_seen_at = unixepoch()
  WHERE (source, key) IN ((OLD.source, OLD.key), (NEW.source, NEW.key));
  DELETE FROM param_key WHERE occurrences = 0;
END;

CREATE TRIGGER param_key_forget AFTER DELETE ON file_param BEGIN
  UPDATE param_key SET
    occurrences = (SELECT count(*) FROM file_param
                    WHERE source = OLD.source AND key = OLD.key)
  WHERE source = OLD.source AND key = OLD.key;
  -- a key nobody uses is not a field the library contains
  DELETE FROM param_key WHERE source = OLD.source AND key = OLD.key AND occurrences = 0;
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
CREATE VIRTUAL TABLE name_fts USING fts5(name, entity_id UNINDEXED, tokenize='trigram');

CREATE TRIGGER name_fts_artifact_ins AFTER INSERT ON artifact
WHEN NEW.name IS NOT NULL BEGIN
  INSERT INTO name_fts(name, entity_id) VALUES (NEW.name, NEW.id);
END;
CREATE TRIGGER name_fts_artifact_upd AFTER UPDATE OF name ON artifact BEGIN
  DELETE FROM name_fts WHERE entity_id = OLD.id;
  INSERT INTO name_fts(name, entity_id)
    SELECT NEW.name, NEW.id WHERE NEW.name IS NOT NULL;
END;
CREATE TRIGGER name_fts_artifact_del AFTER DELETE ON artifact BEGIN
  DELETE FROM name_fts WHERE entity_id = OLD.id;
END;

CREATE TRIGGER name_fts_file_ins AFTER INSERT ON file
WHEN NEW.name IS NOT NULL BEGIN
  INSERT INTO name_fts(name, entity_id) VALUES (NEW.name, NEW.id);
END;
CREATE TRIGGER name_fts_file_upd AFTER UPDATE OF name ON file BEGIN
  DELETE FROM name_fts WHERE entity_id = OLD.id;
  INSERT INTO name_fts(name, entity_id)
    SELECT NEW.name, NEW.id WHERE NEW.name IS NOT NULL;
END;
CREATE TRIGGER name_fts_file_del AFTER DELETE ON file BEGIN
  DELETE FROM name_fts WHERE entity_id = OLD.id;
END;

CREATE TRIGGER name_fts_folder_ins AFTER INSERT ON folder
WHEN NEW.name IS NOT NULL BEGIN
  INSERT INTO name_fts(name, entity_id) VALUES (NEW.name, NEW.id);
END;
CREATE TRIGGER name_fts_folder_upd AFTER UPDATE OF name ON folder BEGIN
  DELETE FROM name_fts WHERE entity_id = OLD.id;
  INSERT INTO name_fts(name, entity_id)
    SELECT NEW.name, NEW.id WHERE NEW.name IS NOT NULL;
END;
CREATE TRIGGER name_fts_folder_del AFTER DELETE ON folder BEGIN
  DELETE FROM name_fts WHERE entity_id = OLD.id;
END;

CREATE TRIGGER name_fts_person_ins AFTER INSERT ON person
WHEN NEW.name IS NOT NULL BEGIN
  INSERT INTO name_fts(name, entity_id) VALUES (NEW.name, NEW.id);
END;
CREATE TRIGGER name_fts_person_upd AFTER UPDATE OF name ON person BEGIN
  DELETE FROM name_fts WHERE entity_id = OLD.id;
  INSERT INTO name_fts(name, entity_id)
    SELECT NEW.name, NEW.id WHERE NEW.name IS NOT NULL;
END;
CREATE TRIGGER name_fts_person_del AFTER DELETE ON person BEGIN
  DELETE FROM name_fts WHERE entity_id = OLD.id;
END;

CREATE TRIGGER name_fts_collection_ins AFTER INSERT ON collection
WHEN NEW.name IS NOT NULL BEGIN
  INSERT INTO name_fts(name, entity_id) VALUES (NEW.name, NEW.id);
END;
CREATE TRIGGER name_fts_collection_upd AFTER UPDATE OF name ON collection BEGIN
  DELETE FROM name_fts WHERE entity_id = OLD.id;
  INSERT INTO name_fts(name, entity_id)
    SELECT NEW.name, NEW.id WHERE NEW.name IS NOT NULL;
END;
CREATE TRIGGER name_fts_collection_del AFTER DELETE ON collection BEGIN
  DELETE FROM name_fts WHERE entity_id = OLD.id;
END;

-- The long tail's own values, so a scraped field is searchable the day it
-- first appears rather than when someone writes a facet for it.
-- `source` is carried here because the file_param key is (file_id, source,
-- key): without it the delete predicate wipes the XMP row's entry when the
-- IPTC row of the same name is removed.
CREATE VIRTUAL TABLE param_fts USING fts5(
    value, file_id UNINDEXED, key UNINDEXED, source UNINDEXED, tokenize='trigram');

-- Each of these deletes first, so INSERT OR REPLACE -- which fires no DELETE
-- trigger -- cannot accumulate stale index rows.
CREATE TRIGGER param_fts_insert AFTER INSERT ON file_param BEGIN
  DELETE FROM param_fts
   WHERE file_id = NEW.file_id AND key = NEW.key AND source = NEW.source;
  INSERT INTO param_fts(value, file_id, key, source)
    SELECT NEW.value_text, NEW.file_id, NEW.key, NEW.source
     WHERE NEW.value_text IS NOT NULL;
END;

CREATE TRIGGER param_fts_update AFTER UPDATE ON file_param BEGIN
  DELETE FROM param_fts
   WHERE file_id = OLD.file_id AND key = OLD.key AND source = OLD.source;
  INSERT INTO param_fts(value, file_id, key, source)
    SELECT NEW.value_text, NEW.file_id, NEW.key, NEW.source
     WHERE NEW.value_text IS NOT NULL;
END;

CREATE TRIGGER param_fts_delete AFTER DELETE ON file_param BEGIN
  DELETE FROM param_fts
   WHERE file_id = OLD.file_id AND key = OLD.key AND source = OLD.source;
END;

-- ============ derived_*: drop this namespace, re-index, reproduced ============
-- Every table here carries a natural key, so recomputation is an upsert rather
-- than an append: an interrupted job re-run must not triple every face.
CREATE TABLE derived_file_hash (
    file_id       INTEGER PRIMARY KEY REFERENCES file(id) ON DELETE CASCADE,
    phash64 INTEGER, dhash64 INTEGER,
    source_sha256 TEXT NOT NULL, computed_at REAL NOT NULL
) STRICT;
CREATE INDEX derived_file_hash_phash ON derived_file_hash(phash64);

CREATE TABLE derived_embedding (
    file_id       INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    space         TEXT NOT NULL CHECK (space IN ('semantic','visual')),
    vector        BLOB NOT NULL, dim INTEGER NOT NULL,
    model_id      TEXT NOT NULL, model_version TEXT NOT NULL,
    source_sha256 TEXT NOT NULL, computed_at REAL NOT NULL,
    PRIMARY KEY (file_id, space, model_id, model_version)
) STRICT;

CREATE TABLE derived_face_cluster (
    id            INTEGER PRIMARY KEY,
    person_id     INTEGER REFERENCES person(id) ON DELETE SET NULL,
    centroid      BLOB, dim INTEGER,
    model_id      TEXT NOT NULL, model_version TEXT NOT NULL,
    updated_at    REAL NOT NULL
) STRICT;
CREATE INDEX derived_face_cluster_person ON derived_face_cluster(person_id);

CREATE TABLE derived_face_instance (
    id            INTEGER PRIMARY KEY,
    file_id       INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    sample_id     INTEGER REFERENCES derived_media_sample(id) ON DELETE CASCADE,
    cluster_id    INTEGER REFERENCES derived_face_cluster(id) ON DELETE SET NULL,
    -- NOT NULL: a face detection with no location cannot be shown, checked,
    -- cropped, or asserted against. RESTRICT, because deleting the region
    -- would leave exactly that.
    region_id     INTEGER NOT NULL REFERENCES region(id) ON DELETE RESTRICT,
    -- Point pairs, consumed whole by the aligner and never filtered on --
    -- unlike generation fields, which is why those are rows and this is not.
    -- BLOB, not TEXT: it is packed floats, and storing it as JSON meant
    -- parsing a string on every crop to get numbers back.
    landmarks BLOB, det_score REAL, dim INTEGER,
    age INTEGER,
    -- What the model reported, not what anyone is. A free-text column here
    -- collected 'M', 'male', 'Male' and 'F' from one backend, and a facet
    -- over that matches a quarter of what it should.
    sex TEXT CHECK (sex IS NULL OR sex IN ('male','female','unknown')),
    pose_yaw REAL, pose_pitch REAL, pose_roll REAL,
    model_id TEXT NOT NULL, model_version TEXT NOT NULL,
    source_sha256 TEXT NOT NULL, computed_at REAL NOT NULL
) STRICT;
CREATE INDEX derived_face_file       ON derived_face_instance(file_id);
CREATE INDEX derived_face_cluster_ix ON derived_face_instance(cluster_id);
CREATE INDEX derived_face_sample     ON derived_face_instance(sample_id);

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
    confidence    REAL,
    -- The same picture may carry a caption from two models on purpose: they
    -- are compared, not merged. Uniqueness is per model, per kind, per region.
    model_id      TEXT NOT NULL,
    model_version TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    computed_at   REAL NOT NULL,
    CHECK (length(text) > 0)
) STRICT;
CREATE INDEX derived_annotation_file ON derived_annotation(file_id);
CREATE INDEX derived_annotation_kind ON derived_annotation(kind, file_id);
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

CREATE TABLE favorite (
    file_id INTEGER NOT NULL REFERENCES file(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES user(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    PRIMARY KEY (file_id, user_id)
) STRICT, WITHOUT ROWID;
CREATE INDEX favorite_user ON favorite(user_id);

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
    created_at     REAL NOT NULL,
    CHECK (other_file_id IS NULL OR other_file_id <> file_id)
) STRICT;
-- Enforced at write, not as a row invariant: ON DELETE SET NULL must be able
-- to detach a judged target without the row becoming illegal. Losing the
-- pointer is acceptable; losing the human judgement is not.
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
    created_at REAL NOT NULL,
    PRIMARY KEY (person_id, file_id)
) STRICT, WITHOUT ROWID;
CREATE INDEX person_assertion_file ON person_assertion(file_id);

CREATE TABLE watched_folder (
    folder_id INTEGER PRIMARY KEY REFERENCES folder(id) ON DELETE CASCADE,
    recursive INTEGER NOT NULL DEFAULT 1 CHECK (recursive IN (0,1)),
    added_at  REAL NOT NULL
) STRICT, WITHOUT ROWID;

CREATE TABLE setting (
    key TEXT PRIMARY KEY, value TEXT NOT NULL
) STRICT, WITHOUT ROWID;

-- A deduplicated payload with no remaining referrer is garbage; without this it
-- accumulates for the life of the library. RESTRICT protects referenced blobs,
-- and says nothing about unreferenced ones.
CREATE TRIGGER blob_reclaim AFTER DELETE ON file_blob BEGIN
  DELETE FROM blob WHERE hash = OLD.blob_hash
     AND NOT EXISTS (SELECT 1 FROM file_blob WHERE blob_hash = OLD.blob_hash);
END;

-- #16: nothing distinguished a database built from this DDL from one built by an
-- earlier generation of it, which is how a stale build went unnoticed.
PRAGMA application_id = 0x53474C59;
PRAGMA user_version   = 1;

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
CREATE TRIGGER file_takes_its_entity AFTER DELETE ON file BEGIN
  DELETE FROM entity WHERE id = OLD.id;
END;

CREATE TRIGGER folder_kind_agrees BEFORE INSERT ON folder BEGIN
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
CREATE TRIGGER person_takes_its_entity AFTER DELETE ON person BEGIN
  DELETE FROM entity WHERE id = OLD.id;
END;

CREATE TRIGGER artifact_kind_agrees BEFORE INSERT ON artifact BEGIN
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
CREATE TRIGGER prompt_takes_its_entity AFTER DELETE ON prompt BEGIN
  DELETE FROM entity WHERE id = OLD.id;
END;

CREATE TRIGGER collection_kind_agrees BEFORE INSERT ON collection BEGIN
  SELECT RAISE(ABORT,'entity kind does not match collection')
  WHERE (SELECT kind FROM entity WHERE id = NEW.id) <> 'collection';
END;
CREATE TRIGGER collection_takes_its_entity AFTER DELETE ON collection BEGIN
  DELETE FROM entity WHERE id = OLD.id;
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

CREATE TRIGGER generation_workflow_is_a_workflow BEFORE INSERT ON generation
WHEN NEW.workflow_id IS NOT NULL BEGIN
  SELECT RAISE(ABORT,'generation.workflow_id must name an artifact of kind workflow')
  WHERE (SELECT kind FROM artifact WHERE id = NEW.workflow_id) <> 'workflow';
END;

CREATE TRIGGER face_sample_belongs_to_its_file BEFORE INSERT ON derived_face_instance
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

-- ============ folder.depth is derived, so the database derives it ============
-- It was a denormalised column maintained by nothing: a reparented folder kept
-- its old depth, and the only reason a test noticed its removal was that a
-- fixture named it in an INSERT.
CREATE TRIGGER folder_depth_ins AFTER INSERT ON folder BEGIN
  UPDATE folder
     SET depth = COALESCE((SELECT depth + 1 FROM folder WHERE id = NEW.parent_id), 0)
   WHERE id = NEW.id;
END;

CREATE TRIGGER folder_depth_upd AFTER UPDATE OF parent_id ON folder BEGIN
  UPDATE folder
     SET depth = COALESCE((SELECT depth + 1 FROM folder WHERE id = NEW.parent_id), 0)
   WHERE id = NEW.id;
END;
