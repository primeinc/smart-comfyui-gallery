# Backlog

Work that is known, agreed and not done. One line per item: what, and why
it matters. Delete an entry when it ships — this file is the pending list,
not a history.

## Scope

The product is local cross-media provenance: captured and generated media
searched, explained, analysed and curated through one authoritative answer.
Immich covers self-hosted photo libraries; Infinite Image Browsing covers
generation output. Neither covers both under one answer.

The design rule the items below are measured against: no destructive action
requires faith. A file has identity apart from its path, so duplicates need
no cleanup to work. Authored claims outlive recomputation, so a name
survives reclustering. Renames keep address history. Capture evidence and
generation evidence coexist. Bulk writes record which answer they were made
against. The timeline separates evidence from interpretation.

Out of scope, deliberately: portable catalogs, workspaces federating
several of them, reversible repartitioning, PDFs, email, and anything
priced.

## Derived data classification

`derived_*` is one namespace and holds three kinds of row. The code does
not distinguish them.

| kind | examples | policy |
|---|---|---|
| cache | thumbnails, query result projections | disposable by design |
| derived state | FAISS index, current clustering | replaced on rebuild |
| derived observation | embeddings, detections, captions, scores | retained |

An observation costs GPU time proportional to the library and is not
recoverable at lower cost. Recomputing is recovery or schema migration.
A newer model supersedes an older observation for the current projection;
it does not delete it. Retaining generations answers questions a single
generation cannot: which faces the 2026 model confused, what successive
captioners said about one picture, how a person's embedding moved between
ages 30 and 40.

Face embeddings are biometric templates. Retaining generations of them
makes encryption-at-rest and no-accidental-network storage rules.

Open work:

- **`derived.drop_all` deletes all three kinds.** It is the only bulk
  delete and it does not distinguish cache from observation.
- **Nothing records supersession.** A newer producer's rows sit beside an
  older producer's with no relation saying one replaces the other for the
  current projection.

## Hedges to remove

Each of these is a workaround for treating derived rows as disposable.
They are debt, not design.

- **`feedback` cannot name what it judged.** Every pointer is
  `ON DELETE SET NULL` and the producer is stored as copied
  `model_id`/`model_version` text rather than a reference, because the
  judged row could be deleted by a re-run. With observations retained, the
  verdict can reference the row it judged.
- **`feedback` points at a file, not a face.** No `region_id` column, so
  "the face in the corner of this picture is not Sarah" is unspellable. A
  picture with two faces cannot say which one is the mistake.
- **`annotation_kind` is a column rather than an annotation id**, for the
  same reason.
- **`verdicts.by_producer` cannot join `derived_annotation`.** It reads
  `feedback` alone, so it reports what was judged and never the text that
  was judged.

## Performance

- **A page rebuilds the whole answer during an ingest.** `answer_generation`
  moves for every table except `job`, `job_item` and `job_event`, so a
  ledger-only job no longer discards answers. Measured
  (`just bench answer-currency`) at 80,000 files:

      at rest              0.179 ms
      a ledger commit      0.233 ms      1.3x
      an answer commit    37.930 ms    211.8x

  Ingest, embed and context write tables answers are built from, so they
  still cost 38 ms per page while they run. Two options:

  - **Per-answer dependencies.** A timed gallery page does not read
    `derived_embedding`, so an embed job should not discard it.
    `db/vocabulary.py` already records which dimensions read what.
  - **Best effort on the read path.** Mutating against a stale ordering is
    already guarded independently: `resultset.AnswerChanged`
    (resultset.py:1166, curating.py:106) refuses a selection made against a
    generation the answer no longer has, with a 409. The read path pays
    full price to protect display only. Serving the cached answer and
    revalidating behind the request removes the 211x for every job.

    Undecided: whether a page should be right, or quick and unable to
    cause data loss.

- **RAW takes whichever preview LibRaw calls the default.** A raw file can
  hold several embedded previews. LibRaw exposes them through
  `imgdata.thumbs_list` and `unpack_thumb_ex(i)`, with each preview's size
  and its own `tflip`, which can differ from the main image's orientation.
  rawpy 0.27.0 exposes `unpack_thumb`, `extract_thumb` and
  `dcraw_make_mem_thumb` and nothing else, so `extract_thumb()` takes the
  default — on the sample 5D Mark III files a 5760x3840 JPEG where 1440 was
  wanted.

  That preview loads at scale 1/4 and raw decode is 41.7 ms of a 137 ms
  file, so the saving is part of 41.7 ms and costs either ctypes into
  LibRaw or a contribution to rawpy. The per-preview `tflip` needs fixtures
  before it is trusted.

- **One raster serves every other consumer.** Thumbnails ask for what they
  need (`decode.open_bounded`, `oriented.for_derivatives`), but
  `oriented.for_model()` returns a full-resolution frame to the perceptual
  hash, face detection, OpenCLIP, BLIP and Qwen. Their input contracts
  differ by orders of magnitude — the perceptual hash reduces to 32x32,
  YuNet caps at 1600 — so most of that decode is discarded.

  Changing the pixels a model sees changes its output, so any such change
  becomes part of the recorded producer identity for embeddings, hashes and
  captions. Otherwise the store mixes vectors from two pipelines.

- **The embed job runs at 29 items/sec and the encoder is not the reason.**
  Measured end to end (`just bench embed-job`, 400 pictures across every
  root), per item:

  | | ms/item | share |
  |---|---|---|
  | decode and orient | 14.4 | 42% |
  | encode at batch 64 | 7.3 | 21% |
  | per-item commit, ledger, similarity | 12.5 | 37% |

  p50 14.9 ms, p95 91.2 ms, max 343.2 ms. The spread is the corpus:
  0.04 MP portraits beside 22 MP raws.

  The encoder alone reaches 594 img/sec on already-decoded pictures
  (`just bench clip-batch`). In the job the GPU is idle 76% of the time and
  the process keeps 7.27 of 16 cores busy, so neither the device nor spare
  cores are the constraint. A free encoder would take the job to about 47
  items/sec.

  By phase, from the job's own records:

  | phase | ms/400 items | share |
  |---|---|---|
  | decoding | 5844 | 39% |
  | inference | 4306 | 29% |
  | preprocess | 2462 | 17% |
  | recording | 854 | 6% |
  | from-device | 791 | 5% |
  | to-device | 122 | 1% |

  Ranked by removable wall time and by semantic risk:

  1. **Batch and thread the encoder.** Reaches 51% of the job and changes
     no pixels: threaded preprocess is bit-identical and batching costs 3
     of 800 nearest-neighbour answers. Batching also amortises the per-item
     copy back.
  2. **Bounded raster for the model path.** Reaches decoding, 39%, and
     changes what the model sees, so it needs its own retrieval gate
     (`just bench clip-retrieval`) first.
  3. **Per-item bookkeeping**, 6% for recording plus runner time outside
     any phase.

- **Batching changes about half a percent of nearest-neighbour answers.**
  Text search is identical at top-1, top-5 and top-20 over 800 distinct
  pictures. Image similarity is not: 3 of 800 best matches change at batch
  64, 5 of 800 in a mixed old/new index, maximum rank move 2 places. The
  two candidates are within 2.3e-03 of cosine. Decide before re-embedding a
  library.

## Correctness

- **A v1 or v2 library reaches today with the wrong derived-face schema.**
  Measured 2026-08-25 by seeding from the shipped schemas
  (`tests/schemas/v01.sql`, `v02.sql`) rather than by inverting today's:
  after all 35 steps the file has no `derived_face_space`, neither
  `derived_face_space_agrees` trigger, no `derived_file_hash_space`, and
  still carries `derived_file_hash_phash`. `derived_face_instance` and
  `derived_file_hash` differ from a fresh build.

  Those objects entered `schema.sql` while it was stamped v3 and no step
  was written for them. `@step(2)` creates one table; `@step(3)` repairs
  drift for `similarity_space` and `derived_embedding` only.

  Pinned by `KNOWN_DRIFT` in `tests/test_a_database_survives_an_upgrade.py`
  so it cannot widen silently, and not repaired. The fix is a step that
  reconciles those objects on any database. The population it serves is
  libraries that started at v1 or v2, which may be nobody. Decide whether
  that population exists before writing it.

- **32 substring bans should be reclassified.** `MUST_NOT_CONTAIN` reads
  code only now (`rules._code_only`), so the false-positive class is
  closed. The 32 bans that name something used five or more times
  elsewhere would each be better as `MUST_NOT_CALL_QUALIFIED` (a call, by
  AST), `MUST_NOT_CONTAIN_BEFORE` (not above this marker) or
  `PACKAGE_FORBIDDEN_PATTERNS` (a regex with a stated reason). The
  mechanisms exist.

- **A big first scan crosses `busy_timeout`.** Measured 2026-08-25 on
  synthesised libraries, with a control write proving the symptom:

  | scan | lane held, 60,000 files | per 1000 | crosses 5 s at |
  | --- | --- | --- | --- |
  | first scan | 9,498 ms | 158 ms | ~32,000 files |
  | rescan, nothing changed | 510 ms | 8.5 ms | ~590,000 files |
  | rescan, one new file | 503 ms | 8.4 ms | ~590,000 files |

  At 60,000 files a competing write waited 5,575 ms and got `database is
  locked`; the same write with no scan running took 0.1 ms. At 40,000 it
  waited 5,273 ms and succeeded. The wait is whatever remains from when
  somebody presses, so above the crossing a growing fraction of presses
  fail rather than all of them.

  A rescan is nineteen times cheaper per file, so this is a first-import
  problem. The exposed writes are console actions during the first read —
  changing a setting, adding a second root — not rating a picture, since
  nothing is in the gallery yet.

  Two routes, and the second is measured:

  - **Split the write half.** Steps 1-3 of `_apply` (mark missing, park
    names, move rows) are interdependent and nearly free on a first scan.
    Step 4, minting new rows, is where the 9.5 seconds goes and is
    independent per row. A half-applied step 4 leaves a scan that added
    some files and not others, which an interrupted scan already leaves and
    scanning again heals; no row is stranded under a `?parked-` name,
    because parking finishes inside the atomic part. `scan()` wraps
    `record` + `apply_scan` in one savepoint, so the split happens there
    too, and its comment states that folder writes belong inside the file
    writes' savepoint.
  - **Defer the FTS population.** Measured 2026-08-25, 15,000 new rows,
    two runs per arm:

        as shipped                     88.8 us/file
        without the name_fts trigger   51.2
        without the generation counter 83.6
        without the kind check         87.8
        the bare INSERT, all dropped    9.1

    90% of what a first scan holds the lane for is index maintenance and
    the filename FTS index is nearly all of it; keeping only that one costs
    83.2. The savings do not add up because SQLite pays most of the trigger
    cost for having any trigger on the statement.

    This trades "filename search is behind during a scan" for "a scan is
    one atomic reconciliation".

  `mint` is not the constraint either way: 14.8 us of the 108, and its
  collision probe is 4.7 of that.

- **18 unsound assignments from sqlite rows.** ty's `unsound-assignment`
  reports 18 places where an Any or Unknown out of sqlite lands in a
  narrower declared type. They are real. Fixing them means changing how
  database rows are typed. The rule is off in `[tool.ty.rules]` with that
  count beside it.

## Modalities the schema allows and nothing produces

Each has a slot cut for it and nothing writing to it.

- **OCR.** `db/schema.sql:1469` permits `said.kind = 'ocr'` and
  `sg_web/media_view.py:475` types it. Nothing writes one. Text inside a
  screenshot, receipt or document is a whole modality missing. Immich
  searches it as a first-class field.

- **Auto-tagging: a model that proposes a keyword.** The authored half
  ships: `tag` and `file_tag` (schema v42), `f=tag:eq:<word>` with AND and
  OR, the media inspector, and `/keywords` for folding spellings and
  removing a word. `derived_annotation.kind = 'tag'` is permitted and never
  written.

  The work is a job that writes `derived_annotation` tags from a classifier
  and a surface that offers them for acceptance; accepting writes a
  `file_tag`. Evidence suggests, a person authors, and the two never share
  a table — the same shape reverse geocoding needs below.

  - **A smart album cannot filter by keyword.** `collection_rule`'s stored
    format is versioned (`RULE_VERSION`), so a `tag` field there is a
    durable-format change with its own migration, not a registry line.

- **Reading an authored export back in.**
  `GET /operations/export/authored.json` writes names, ratings, favourites,
  places, keywords, albums with nesting, and who is and is not in what,
  keyed by `content_sha256`. Reading one back has its own conflicts to
  resolve: a hash matching two files, a name already taken, an album that
  exists with different members.

- **XMP sidecars.** The only thing that lets another application see what
  was decided here. LibrePhotos round-trips face regions and names as
  MWG-RS; digiKam edits EXIF/IPTC/XMP in place. JSON solves custody; XMP
  solves interop.

- **Reverse geocoding, as a suggestion.** `db/places.py:8` names a future
  reverse-geocoding job cached by geographic cell. The rule that GPS never
  mints a human place stands. Missing is the middle step: GPS evidence
  produces a derived suggestion, a person accepts or corrects it, and the
  acceptance is the authored claim.

## Workflows the data supports and the product does not

- **A video with nobody in it is the most expensive video to process.**
  `harvest_video` samples a cadence and, because it found nothing, bisects
  the widest gaps for up to `REFINE_MOST = 32` extra decoded frames,
  stopping only when faces appear (db/detect.py). The refinement is
  correct: a fixed interval can land every sample on the establishing shot
  of a clip that is otherwise all people. But a landscape timelapse pays
  the cadence, then thirty-two more decodes, then records that it found
  nothing.

  A cheap gate before the expensive certainty: `derived_embedding` holds a
  semantic vector per file after the embed job, and the cosine against an
  encoded phrase like "a person" is arithmetic on a vector already stored.
  `db/retrieval.py` already asks a space a question and reports where a
  score stands.

  Three constraints:

  - **Not a silent skip.** A false negative hides a person permanently and
    reads as a broken face pipeline. A gated file records that it was
    gated, with the threshold and the model that gated it.
    `derived_face_scan` says a file was looked at and what was found;
    "looked at cheaply and declined" is a third state, not an absence. A
    later run at a lower threshold then revisits exactly those files.
  - **Not a new hardcoded number.** The threshold belongs with the settings
    entry below, not as a constant in db/detect.py.
  - **Not assumed to pay.** The gate is worth building only if decode is
    where the time goes. The embed benchmark says decoding is 39% of that
    job; the equivalent split for `detect_faces` over a video-heavy corpus
    does not exist yet, nor does the recall cost at each candidate
    threshold — how many face-carrying videos a given cut would have
    skipped, checked against the ones already scanned. That measurement is
    most of the work.

  A cheaper variant gates the refinement rather than the file: the cadence
  frames are already decoded and already found nothing, so the question is
  only whether to spend the extra thirty-two.

- **A verdict on a similarity has no surface.** The annotation arm ships
  (`POST /i/{slug}/said/verdict`), the person arm is written by correcting
  a face, and the duplicate arm by saying two pictures are not one
  (`/dupes`, `db/authored.py reject_duplicate`). "These two are not alike",
  against a semantic neighbour, has nothing.

- **Nothing reads a verdict to change anything.** `db/verdicts.py
  by_producer` and `contests` read the annotation arm; corrections read as
  a count. No threshold moves, no model is deselected, no re-run is
  suggested.

  What the data supports once there is enough of it:

  - **Which producer to prefer.** Two caption models over the same files
    with verdicts on both is a direct comparison on the corpus that
    matters. Same for two face backends, whose embedding spaces never mix.
  - **Where a model fails.** Verdicts join to kind, capture time, folder,
    camera and whether a file is generated. "Wrong 8% overall, 40% on video
    frames" is actionable where "captions are bad" is not.
  - **Which knob a rejection is about.** A rejected merge has a face pair
    with a cosine distance. A run of rejections just above the operating
    point is the argument for moving it: "17 of your 20 rejected merges
    scored between 0.48 and 0.52".

  Three rules for that surface:

  - **It is a biased sample.** People judge what they look at and click
    `wrong` more readily than `right`, so a raw error rate is not
    publishable. A comparison between producers over the same judged set
    shares the bias.
  - **State n, and show nothing below a floor.** A model is not worse than
    another on four verdicts. Below the floor, say how many more judgements
    are needed.
  - **Never present a correlation as a cause.** "Wrong more often on video"
    is an observation, with a link to the files it came from.

- **A negative person claim does not constrain clustering.**
  `person_assertion.stance` records "not her" and `db/derived.py` re-applies
  assertions after reclustering, but re-attachment happens after the fact.
  `db/grouping.py METHODS` takes a graph and vectors and no constraints.
  Feeding must-link / cannot-link edges into the graph makes each
  correction permanent and local instead of trading it against a global
  threshold. A false merge is one cannot-link edge from being fixed on
  every future run.

- **Detection knobs are constants in a module.** The clustering operating
  point is the `face_cluster_threshold` setting, validated at submit,
  pinned into the payload and shown per run. Unreachable without an edit
  and a restart: `_LABEL_MATCH_THRESHOLD = 0.9`, `min_det_score=0.5`,
  `min_face_px=24` (vision/faces.py), `FLOOR = 0.7` (db/detect.py).

  These are detection rather than grouping: changing one changes which
  faces exist at all, and a face nothing detected cannot be recovered by
  re-running the clustering. Exposing them needs the re-detect path to be
  as cheap to undo as re-clustering, and it is not.

- **`pages.face_across_runs` has no caller.** Per face, how big a group
  each run put it in. A face one run puts with fifty others and another
  puts alone is where two clusterings differ, and the per-picture view
  (`pages.disagreements`, on the console) does not surface it.

- **A captioner does not know who is in the picture.** `person_assertion`
  says this person appears in this file, optionally with region and frame,
  signed by a user. The `Captioner` protocol is `describe(image) -> str`
  with no prompt parameter (vision/captions.py). BLIP takes a conditional
  prefix; Qwen-VL runs under a system instruction here
  (`MEDIA_INSTRUCTION`, vision/semantic/qwen_vl.py) and can be told who is
  present and where.

  Four rules:

  1. **Only a claim a human signed.** A name enters a prompt only where
     `person_assertion.user_id IS NOT NULL`. A caption is read, believed
     and searchable; a derived cluster match is not sufficient grounds to
     write a name into one.
  2. **The prompt is part of the producer identity.** `derived_annotation`
     is unique on `(file_id, kind, model_id, model_version, region_id,
     sample_id)` and the prompt is not in it, so the same model captioning
     the same picture before and after a person was named would collide.
     The name-set fed in has to be recorded and participate in that
     uniqueness.
  3. **Do not write an authored claim into a derived row.** Baking "Sarah"
     into derived text freezes it: rename the person and every caption is
     wrong until something re-infers them. Caption "a woman in a red coat"
     and substitute the current name at render time from the assertion.
  4. **Local only.** A name plus a face crop leaving the machine is what
     the biometric doctrine forbids.

  The remaining reason to build the prompt seam is a better sentence from a
  model that knows two people are present and where. Search already ships:
  typing "Sarah at the beach" offers the split `person=sarah` plus
  `q="at the beach"`, offered and never applied.

- **Duplicate consolidation.** `/dupes` shows every group read-only: each
  copy's folder, the collections it is filed under, and whether copies are
  byte-identical or merely alike. `pages.dupe_placements` names every
  collection a member is filed under and how whole each is.

  The operation is missing, and the naive version is the one to avoid.
  Byte identity and organisational identity are different: three copies of
  one file in `Iowa 2019`, `Family` and `Old Backup` are one content and
  three placements, and a deduper reporting "1 duplicate removed" has
  turned a complete collection into an incomplete one. Build *consolidate
  redundant storage while preserving every logical placement*, shown as a
  post-state preview before anything is touched:

      3 exact copies - SHA-256 identical
      Used by:  Iowa 2019  428/428 present
                Family     113/113 present
      After:    3 placements, 1 stored payload, all collections complete

  Hydrus is the reference for duplicate/alternate relationships; Immich for
  the review-and-keep flow.

- **The incumbents have not been run on this library.** Until they are,
  "ours is different" is an assertion. Install Immich (as a read-only
  external library), Infinite Image Browsing, digiKam 9.2 and LibrePhotos,
  and run the same questions through each: all generated videos with this
  LoRA; what prompts dominate this answer; find this person in August;
  which files contain this screenshot text; compare three outputs and their
  recipes; fix a wrong face; resolve these duplicates; export my names so
  another DAM sees them; why does it think this happened in 2023.

  Verdict per app per question: better than ours / good enough / painful
  but possible / impossible. Where an incumbent wins and the slice is not
  structurally required by our model, stop maintaining the inferior
  reinvention — integrate, copy the pattern, or delete ours.

  Licence: IIB and LibrePhotos are MIT and can be borrowed from with
  attribution. Immich is AGPL-3.0 — readable, consequential to copy into an
  MIT tree.

- **SwarmUI-Quarry's mechanism is worth taking; its database is not.**
  Quarry (`jtreminio/SwarmUI-Quarry`, MIT) promotes a small core of fields,
  keeps every other `sui_image_params` / `sui_extra_data` property
  generically, and asks the index which keys the corpus contains — the
  field catalog, independently arrived at. Not worth taking: its parallel
  path-keyed history index, where `file_param` + `param_key` + one
  authoritative ResultSet is the better shape. Its metadata-extraction and
  filter-builder tests are a regression corpus worth porting with the
  licence notice.

## Corpus and test instruments

- **The coverage instrument counts a crash as coverage.** `tests/reach.py`
  `watch()` suppresses every exception, so lines executed before a raise
  still count. That is correct for measuring reach and wrong as the only
  lens: a `.jxl` crash improved the score. Nothing separates "reached this
  line" from "died on this line", so the corpus cannot report a regression.

- **The 8 generator dialects are fake.** `tests/corpus.py` writes three
  payloads invented from memory (A1111, ComfyUI, SwarmUI) and never touches
  NovelAI, Fooocus, InvokeAI, Easy Diffusion or Draw Things. The ComfyUI
  graph is structurally invalid — node "9" references a node "8" that does
  not exist — so `db/graph.py`'s back-walk never runs and only its fallback
  is exercised, which its own docstring calls worse than reporting nothing.
  Six of the eight writers are cloned under `../refs/`; the two closed ones
  have format specs in
  `../refs/receyuki/stable-diffusion-prompt-reader/sd_prompt_reader/format/`.

- **`tests/reach.py` excludes `db/scan.py` and the exclusion followed a bad
  number.** The first run was `509/2162 = 23.5%`; dropping `scan.py` made
  it `502/1858 = 27.0%`. The stated reason — most of it takes a connection,
  not a path — is true and was applied only to the module dragging the
  average. `db/probe.py` also has connection-taking functions and was kept.
  Apply it to both or to neither.

- **The corpus mutation bucket buys nothing.** Measured 35.1% with and
  without. ExifTool's corpus is already 134 truncated files, so the
  readers' failure arms were reached before anything was broken on purpose.
  Keep it only if the specimen set becomes all-valid; otherwise delete it.

- **The corpus is not a published dataset.** It has a README, a per-part
  licence table and three lockfiles carrying per-file provenance
  (`docs/CORPUS_SOURCES.md`, `../sg-corpus/README.md`). Missing before
  publication: a version, a citation, an intended-use statement, and a
  decision about which parts ship. ExifTool's images are GPL-3 with mixed
  per-image provenance and carry real coordinates (`Apple.jpg` 53.38N,
  `Google.jpg` 40.40N); they are referenced by checksum and never vendored.

- **`.cap` and `.k25` are claimed and unread.** The corpus reads 97 of the
  99 suffixes the application claims. Both are real LibRaw formats with no
  sample found in three archives; they are BLOCKED_EXTERNALLY in
  `tests/needs.py BLOCKED` with that search evidence, `tests/rawsamples.py`
  retries them on every fetch, and the gate fails the day a sample lands
  while the excuse still stands.

- **Refused, each outside the tree's reach:** rotated video and QuickTime
  dual timestamps (need a handset); the other 14 RAW suffixes
  (`darktable-org/rawspeed`, raw.pixls.us); AVIF; the audio tag matrix
  (`quodlibet/mutagen` test data); hostile PDFs (`mozilla/pdf.js`
  `test/pdfs/`); round-trip write-preservation, which is where
  `db/authored.py` will break; fuzzing with the corpus as seed.

- **The generated contract carries types, not values.** Five server
  constants reach the browser as transcriptions held by eye:
  `db/jobs.py TERMINAL` (timeline.ts SETTLED), `db/ledger.py PAGE_MOST`
  (operations.ts TAPE_PAGE), `sg_web/timeline_view.py NARROWEST` and `_W`,
  `db/resultset.py MAX_PAGE_SIZE` (timeline.ts TILES_MOST). Each copy names
  its source (MAGIC-FUCKUPS.json, the five `partial` rows). The fix is one
  value channel: emit them into the OpenAPI document or a constants payload
  the shell already ships, so a changed server value is a type error rather
  than a stale clamp.

## Flakes

- **`test_writes_stay_linear.py` fails near its own tolerance and is
  unreproduced.** The ratio gate is `< 2.0` and reported failures sat on
  it: three of four runs, on three different cases (`writing a parsed
  field` 2.0019, `rescanning an unchanged library` 2.1, `renaming a file`
  2.4).

  Probed 2026-08-25 under both suspected causes and neither reproduced.
  Against a 600,000-object retained heap, gc-on vs gc-off: every write case
  0.93-1.06, no arm distinguishable. Against 16 of 16 cores contended by
  confirmed-running spinner processes: 0.89-1.11, 0/6 over the gate. A full
  single-process `pytest tests/ -m slow` passed clean. The cause is neither
  collector pressure nor CPU contention.

  **Do not implement the repeated-timings-with-a-median fix.** It was
  measured and it is worse. `timeit`'s documentation
  (`../refs/python/cpython/Doc/library/timeit.rst`, `Timer.repeat`) argues
  for `min` over mean because interference only adds time — but that
  reasoning covers one measurement, and this gate divides two. Taking the
  min of each side independently lets a lucky-fast small swing the
  quotient: under contention min-of-3 widened the spread to 0.67 on
  `writing a parsed field` and 0.59 on `deleting a file`, where single-shot
  ratios stayed inside 0.93-1.09.

  Options: a wider SMALL/LARGE gap so the ratio is not measuring noise, or
  timing the pair inside one measurement rather than as two. A gate that
  fails one run in three teaches people to re-run it, so leaving it alone
  is not an option.

## Delivery

- **Nothing is served by anything but Litestar.** Measured 2026-08-25 by
  driving the ASGI app directly (no httpx, no server), best of three runs
  of 300:

      /thumbs/<sha>.webp   File, no database    1480 us
      /settings            JSON, one database   5033 us
      reading the file, nothing else              80 us

  An asset costs about 1.5 ms of application CPU and a 60-cell page about
  89 ms, on requests that touch no database. The cost is fixed: 810 bytes
  and 101 KB both cost ~1400 us above the file read, so it is dispatch, not
  streaming.

  That is server CPU, not what a person waits — a browser fetches sixty
  cells in parallel, and a real server adds cost this in-process
  measurement does not have. It is still 89 ms per page view that a cache
  in front would not spend.

  **`create_static_files_router` over the thumbs directory would be a
  regression.** A miss on that path renders (`sg_web/app.py asset_bytes`),
  which is why a fresh library gets a slow grid instead of broken pictures;
  a static router 404s. The viable shape is a front server that tries the
  file and falls back to the application on a miss (Caddy/nginx
  `try_files`). The router's directory is also relative to the working
  directory, and the thumbs cache is an absolute path under the run home.
