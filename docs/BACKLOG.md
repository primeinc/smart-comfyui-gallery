# Backlog

Work that is known, agreed and not done. One line per item: what, and why
it matters. Delete an entry when it ships — this file is the pending list,
not a history.

## What this is for

Not a backlog item; the thing the items are measured against, written
down because it decides which of them matter.

The commodity product is media management, and the incumbents are good
at it. If the goal were a self-hosted photo library, the answer is
Immich. If it were a generation-output browser, the answer is Infinite
Image Browsing. The only defensible reason this exists is narrower:

> **Local cross-media provenance. Captured and generated media searched,
> explained, analysed and curated through one authoritative answer.**

Underneath that is the thing that makes it worth finishing:

> **Make the data legible enough to stop being afraid to touch it.**

That is what the pedantic parts are for, and why they are not
architecture for its own sake. A file has identity apart from its path,
so duplicates need no cleanup to work. Authored claims outlive derived
rebuilds, so a name survives reclustering. Renames keep address history.
Capture evidence and generation evidence coexist rather than one
winning. Bulk writes prove which answer they were made against. The
timeline separates evidence from interpretation. Every one of those is
the same sentence: **no destructive action should require faith.**

Two consequences that are not yet true in the code:

- **Reproducible does not mean disposable.** "Derived" is currently a
  synonym for "cache", and it should be three things: *cache*
  (thumbnails, query results — disposable by design), *derived state*
  (the FAISS index, the current clustering — rebuildable and
  replaceable), and *derived observations* (embeddings, detections,
  captions, scores — reproducible, and potentially worth keeping).
  Inference roughly doubles in quality every few months, so a newer
  model should SUPERSEDE an older observation for the current
  projection without deleting it. Ten years of a stable corpus then
  answers questions nothing else can: which faces did the 2026 model
  confuse, what did successive captioners say about this picture, how
  did one person's embedding move from 30 to 40. None of that survives
  an upgrade that vacuums the old outputs because they were labelled
  derived. Face embeddings are biometric templates, so retaining
  generations of them makes encryption-at-rest and no-accidental-network
  a storage rule, not a preference.

- **Anything expensive the program learns should be exportable without
  the program.** The application can always rebuild an index; the
  accumulated understanding is the valuable part. Concretely: export a
  person's face representation over a date range as embeddings WITH
  their provenance — model identity and version, dimensions,
  normalisation, preprocessing, source content hash, occurrence,
  capture time, face region — plus a centroid. Not a naked 512-float
  vector, which recreates exactly the opaque dependency this is
  supposed to escape.

Deferred on purpose, and deliberately not listed below: portable
catalogs, workspaces federating several of them, reversible
repartitioning, PDFs, email, and anything with a price on it. They are
consequences the architecture should keep room for, not work.

## Performance

- **A page still rebuilds the whole answer during an INGEST.** Half of
  this shipped: `answer_generation` (db/schema.sql) moves for every
  table except `job`, `job_item` and `job_event`, so the job that runs
  for hours -- a precache, which writes nothing but the ledger -- no
  longer discards anything. Measured (`just bench answer-currency`) at
  80,000 files:

      at rest              0.179 ms
      a ledger commit      0.233 ms      1.3x
      an answer commit    37.930 ms    211.8x

  The second number was the whole problem and is gone. The third is
  unchanged ON PURPOSE -- an answer that can have changed must be
  rebuilt -- but it means ingest, embed and context still give a person
  38 ms pages for as long as they run, because those jobs really do
  write tables answers are built from.

  Two ways on from here, and the second is more interesting:

  - **Per-answer dependencies.** A timed gallery page does not read
    `derived_embedding`, so an embed job should not discard it.
    db/vocabulary.py already knows which dimensions read what. More
    counters, more precision, same absolute-prevention posture.

  - **Best effort on the READ path.** Worth asking whether absolute
    prevention is buying anything here. Serving a slightly stale
    ordering costs a person a picture appearing one page late.
    Mutating against a stale ordering costs them data -- and that is
    already guarded independently: `resultset.AnswerChanged`
    (resultset.py:1166, curating.py:106) refuses a selection made
    against a generation the answer no longer has, with a 409. So the
    destructive path has its own hard gate, and the read path is
    paying full price to protect a DISPLAY.

    If reads served the cached answer and revalidated behind the
    request, the 211x would go too, for every job rather than only the
    ledger-only ones. What needs deciding first is what a person is
    owed: a page that is right, or a page that is quick and cannot
    hurt them. The machinery for the second already exists.

- **Thumbnails are still serial.** 6.76 files/sec over the real library
  (`just bench thumbs-phases`), up from 1.80, but one worker renders one
  file at a time. `run_next()` loops over every pending item, so a 20,000
  file precache owns the only background worker for an hour. Decode,
  resize and encode are all per-file work with nothing shared, so this is
  the next win: claim a bounded batch, render a few concurrently, keep
  the database worker as the thing that commits. Benchmark 1/2/4/8 in
  flight rather than picking a number.

  Interactive work should also outrank precache — a browser waiting on
  `/thumb/x` is real work and a speculative queue is not — and two
  requests for the same missing key currently render it twice.

- **RAW takes whichever preview LibRaw calls the default.** A raw file
  can hold several embedded previews; LibRaw knows them all through
  `imgdata.thumbs_list` and `unpack_thumb_ex(i)`, and exposes each one's
  size and its own `tflip`, which it warns can differ from the main
  image's orientation. rawpy 0.27.0 exposes neither — checked: it has
  `unpack_thumb`, `extract_thumb` and `dcraw_make_mem_thumb` and nothing
  else — so `extract_thumb()` takes the default, which on the sample
  5D Mark III files is the full 5760x3840 JPEG when 1440 was wanted.

  Smaller than it sounds now that the decoder is asked in the right
  aspect: that preview loads at scale 1/4, and raw decode is 41.7 ms of a
  137 ms file. Picking a smaller preview saves part of that 41.7 and
  costs either ctypes into LibRaw or a contribution to rawpy. Worth doing
  when the pipeline is parallel and decode is the remaining cost, not
  before, and the per-preview `tflip` needs fixtures before it is trusted.

- **One raster serves every OTHER consumer.** Thumbnails now ask for what
  they need (`decode.open_bounded`, `oriented.for_derivatives`), but
  `oriented.for_model()` still returns a full-resolution frame to the
  perceptual hash, face detection, OpenCLIP, BLIP and Qwen. Their real
  input contracts differ by orders of magnitude — the perceptual hash
  reduces to 32x32, YuNet caps at 1600 — so most of that decode is
  discarded. Each should ask for the size it needs and let the source
  adapter find the cheapest correct way to produce it, the way the
  thumbnailer now does.

  Changing the pixels a model sees changes its output, so any such change
  has to become part of the recorded producer identity for embeddings,
  hashes and captions. Otherwise the store quietly mixes vectors from two
  different pipelines.

- **The embed job runs at 29 items/sec, and the encoder is not the
  reason.** Measured end to end (`just bench embed-job`, 400 pictures
  across every root), per item:

  | | ms/item | share |
  |---|---|---|
  | decode and orient | 14.4 | 42% |
  | encode at batch 64 | 7.3 | 21% |
  | everything else: per-item commit, ledger, similarity | 12.5 | 37% |

  p50 14.9 ms, p95 91.2 ms, max 343.2 ms — the spread is the corpus,
  0.04 MP portraits beside 22 MP raws.

  The encoder alone reaches 594 img/sec with pictures already decoded
  (`just bench clip-batch`): thread the preprocess 4.45×, batch the GPU
  6×, overlap them 1.55×, all bit-identically or near enough. But in the
  job the GPU is **idle 76% of the time** and the process already keeps
  **7.27 of 16 cores** busy, so neither the device nor the spare cores
  are what it is waiting for. An encoder that cost nothing would take
  the job from 29 to about 47 items/sec.

  The job records its own phases now, so the split is a query rather
  than a benchmark:

  | phase | ms/400 items | share |
  |---|---|---|
  | decoding | 5844 | 39% |
  | inference | 4306 | 29% |
  | preprocess | 2462 | 17% |
  | recording | 854 | 6% |
  | from-device | 791 | 5% |
  | to-device | 122 | 1% |

  Ranked by removable wall time **and** by semantic risk:

  1. **Batch and thread the encoder.** Reaches inference, preprocess and
     the copy back — 51% of the job — and neither changes the pixels:
     threaded preprocess is bit-identical, and batching costs 3 of 800
     nearest-neighbour answers. Batching also amortises the per-item
     copy back and part of the bookkeeping, which is the argument for
     touching the runner; the GPU sitting idle is not.
  2. **Bounded raster for the model path.** Reaches decoding, 39%, and
     changes what the model sees, so it needs its own retrieval gate
     (`just bench clip-retrieval`) before it can ship.
  3. **Per-item bookkeeping**, 6% for recording plus whatever the runner
     spends outside any phase.

- **Batching changes about half a percent of nearest-neighbour answers.**
  Text search is identical at top-1, top-5 and top-20 over 800 distinct
  pictures. Image similarity is not: 3 of 800 best matches change at
  batch 64, 5 of 800 in a mixed old/new index, maximum rank move 2
  places. A product decision, not a defect — the two candidates are
  within 2.3e-03 of cosine and either is defensible — but it is not zero
  and should be decided before a library is re-embedded.

## Surfaces

- **`library` and `mount` are the same thing.** `root.kind` admits
  `library`, `mount` and `trash`, and nothing anywhere branches on the
  first two: `db/pages.py:276` selects `kind IN ('library','mount')`
  and `sg_web/folder_view.py:55` probes `kinds=("library","mount")`.
  Only `trash` is load-bearing, and the schema says why -- it is a real
  location, not a state, so views exclude that subtree by ancestry.

  The distinction `mount` was reaching for -- "this one is not always
  attached" -- is already carried by `root.online`, which is per-root,
  set by probing, and is what the whole deletion doctrine rests on. So
  either give `mount` a behaviour it alone has, or collapse it: one
  media kind and `trash`. Two names for one thing is a decision
  everybody reading the schema has to make again.

- **Benchmarks in the UI.** `just bench thumbs-phases` and the other
  benchmark recipes write JSON that only a terminal ever sees. The
  operations console is where a run's own facts already live; these
  belong there too, so throughput is something you can look at rather
  than something you have to run.

## Correctness

- **Thirty-two substring bans cannot tell a statement from a sentence.**
  `sglint/policy.py` `MUST_NOT_CONTAIN` holds 59 banned tokens, and 32
  of them name something this tree uses five or more times elsewhere:
  `sg_web/story_view.py` bans `"FROM "` (731 uses elsewhere) and
  `"execute("` (912); `db/evolution.py` bans `"DELETE"` (305) and
  `"INSERT"` (212); `db/rendering.py` bans `"torch"` (37);
  `sg_web/templates/story.html` bans `"similarity"` (231).

  Each guards a real constraint -- evolution.py genuinely must not
  write, story_view.py genuinely must not reach for SQL -- but by a
  mechanism that reads prose. A comment saying "this module never
  DELETEs" trips it, and the failure looks like an architecture
  violation rather than a word. That already happened once, to
  `media_view.py: "neighbour"`, which was deleted rather than satisfied.

  The 27 remaining bans are fine and should stay: `ARTIFACT_FILES`,
  `workers=`, `import openai` and the rest appear nowhere else, so they
  cannot fire innocently.

  The file already carries the better mechanisms, so this is
  reclassification rather than invention: `MUST_NOT_CALL_QUALIFIED` for
  a call (AST), `MUST_NOT_CONTAIN_BEFORE` for "not above this marker",
  `PACKAGE_FORBIDDEN_PATTERNS` for a regex with a stated reason. Each
  prose-fragile ban should move to whichever states what it means, and
  the ones that cannot should at least stop scanning comments.

- **A big enough scan still crosses `busy_timeout`.** The walk no longer
  holds SQLite's write lane (db/scan.py `survey`/`record`), and over the
  sample roots the hold fell from 3,116 ms per 1000 files to 147 ms
  (`just bench scan-lock`). 147 ms per 1000 crosses the 5000 ms
  `busy_timeout` at roughly 34,000 files, and libraries are bigger than
  that -- so on a large first scan a writing ROUTE can still wait five
  seconds and then 500. The worker is fine (a busy claim is now no turn
  rather than a crash), but a person rating a picture mid-scan is not.

  The fix is to stop making the write half one transaction: `record` +
  `apply_scan` could commit in bounded batches, which trades "a scan is
  one atomic reconciliation" for "nothing waits more than a moment".
  That trade needs deciding rather than assuming -- a half-applied scan
  is a state the module has never had to describe.

- **`neighborhood`'s test matrix is short two cases** the filmstrip
  claims to support: the neighbourhood under a FILTERED ResultSet, and
  under a SEMANTIC one. `sort=oldest` covers the non-default timed
  order; nothing covers those two.

- **No cold acceptance lane.** Nothing runs the documented bootstrap from
  a checkout with no `.venv`, no `node_modules` and no
  `sg_web/static/build`. `test_the_documented_launch_serves_a_whole_application.py`
  tests the launcher's refusals in-process and says so; it is not that
  lane.

## The gate's remaining hole

- **The ten-second gate has no Python type check.** `just check` is held
  to ten seconds (`just budget` proves it) and pyright cannot fit: 137.5s
  over 170 files, `--threads` makes it 181s. The cost is one import --
  `vision/semantic/openclip.py`, `vision/semantic/qwen_vl.py` and
  `vision/captions.py` each take ~90s ALONE and share `torch`. torch and
  transformers both ship `py.typed`, so pyright reads their inline
  annotations from source and `useLibraryCodeForTypes = false` does not
  skip them; no setting keeps torch's types and avoids parsing torch. So
  pyright moved to `just check-deep` and the fast gate lost cross-module
  Python inference entirely.

  The lead: **ty 0.0.74 checks the same tree in 4.6s** -- 30x -- and
  `pyproject.toml` has carried `[tool.ty.src]` and `[tool.ty.analysis]`
  since before this. It reported **74 diagnostics** on a tree pyright
  calls clean (real ones among them: `db/capture.py:101` `_CLAIMED |=
  _GPS_CLAIMED` between `set[Base]` and `set[GPS | IFD]`;
  `db/planning.py:1367` `|` on `int | frozenset[str]`). Triage those and
  ty restores type checking to the fast gate. It was installed, measured
  and removed rather than left in the tree unwired.

## The query workspace, as far as it got

Built: the query vocabulary (db/vocabulary.py), filter discovery
(db/discovery.py), answer analysis (db/analysis.py), the filter drawer,
Gallery/Table/Analyze, the compare tray, endless browsing, reading
generation metadata out of video containers, Any/All multi-select on
every dimension where both readings mean something, value lists on
`folder`/`album`/`people.person`/`place.id`, a door to the long tail
(`param.has`, `param.is`), and a cut on the semantic ranking so a
search answers with a set. What is NOT built:

- **The field catalog: one searchable Add-filter over every fact.**
  This is the largest remaining gap and the door we built is the proof
  of it. `param.is` is a text box whose placeholder is `key=value`
  (`frontend/src/filters.ts:334`), so it can only be used by somebody
  who already knows the internal spelling -- which is the one thing the
  application is supposed to remember for you. The requirement is not a
  nicer box; it is that the application TEACHES its own vocabulary:

      Add filter…    [ edit                          ]
      ─────────────────────────────────────────────
      Used local editor                     SwarmUI
      Edit reference megapixels          Generation

  You type what you half-remember, it tells you what it knows. Then
  the chosen field decides its own operators and offers its own values
  (`is / above / below / between` for a number, `is / any of` for an
  enumeration, `contains` for text). The catalog must match on friendly
  label, alias, raw source key and source application, so `edit`,
  `editor` and the ugly serialised key all arrive at the same fact.

  `param.has` already discovers keys with counts and should stop being
  a filter somebody uses by hand and become the thing that POWERS this.
  The curated/discovered split stays real internally -- one has
  semantics we understand, the other is a recorded fact -- and stops
  being visible to the person.

  Three things make this harder than it looks, all measured against
  the real 3,748-file library (108 distinct `file_param` keys):

  1. **`_param()` flattens lists into INDEXED keys.** `used_wildcards.0`
     through `used_wildcards.6` are one concept wearing seven names, 55
     files each; likewise `loras.0`/`loras.1` and
     `unused_parameters.0..2`. "Did this use a wildcard" is currently
     seven separate questions. The catalog has to collapse an indexed
     family into one dimension whose repeats OR -- which is exactly the
     `multi="any"` machinery already built.
  2. **Everything is TEXT.** `automaticvae` is the string `'True'` with
     `value_num` NULL on all 155 files, so no operator can be derived
     from storage. The observed type has to be inferred from the values.
  3. **About 40 of the 108 keys are EXIF plumbing** -- `StripOffsets`,
     `YCbCrPositioning`, `FocalPlaneXResolution`. A picker built
     straight off `param_key` is a haystack. Rank by how much a key
     discriminates within the answer, and let the ugly ones be findable
     without being offered first.

  The acceptance test is a browser interaction, not a unit test for
  `param.is`: open the app knowing nothing of the schema, type `edit`,
  discover the field exists, choose `is -> yes`, get the right media,
  save the question, reload, get the same answer.

- **A saved view is not a first-class thing.** Everything a question can
  become is a collection today. People distinguish three: an **album**
  (things I deliberately put together), a **smart collection** (a
  dynamic grouping that behaves like a collection), and a **saved
  view** ("that was a useful question, remember it"). They can share
  one `GalleryQuery` underneath without being one product object. The
  tell that this is missing: having composed a good question twice, the
  only offer is "save view", which makes a collection.

- **The analysis has no prompt-term view.** Exact prompt identity is
  built and counted. Recurring TERMS across an answer -- which is a
  different claim with a different error mode -- is not, and is
  deliberately absent rather than quietly mixed into the exact counts.

- **The table cannot be sorted by its columns.** Order is the
  ResultSet's, which is correct; clicking a column heading would have to
  become a `sort` the ResultSet understands, not a client-side reorder of
  one page.

- **The compare tray has no two-up A/B mode.** It shows everything kept,
  side by side, in tray order. Two is the common case and works; naming
  one A and one B and flipping between them is not built. digiKam's
  Light Table is the reference, including synchronised pan and zoom.

## Modalities the schema allows and nothing produces

Each of these has a slot already cut for it. The slot being empty is
not a design decision anybody made; it is work nobody did.

- **OCR.** `db/schema.sql:1469` permits `said.kind = 'ocr'` and
  `sg_web/media_view.py:475` types it. Nothing writes one. For an
  application whose thesis is "search what you have", text sitting
  inside a screenshot, a receipt or a document is a whole modality
  missing. Immich searches it as a first-class field.

- **Tags and keywords.** `said.kind = 'tag'` is likewise permitted and
  likewise never written, and `db/vocabulary.py` has no tag dimension
  at all -- 41 dimensions and not one of them is the oldest idea in
  digital asset management. Every serious DAM has hierarchical
  keywords; this has people, places, albums and generation entities and
  no free-form vocabulary at all.

- **Metadata portability, in and out.** There is no export route in
  `sg_web/app.py`. Names, ratings, places, tags, collections and
  provenance live only in `gallery.db`, so deleting the application
  deletes the understanding. This one contradicts the product thesis
  directly -- an application about custody of your own data must not be
  the only place your knowledge can exist. XMP sidecars are the boring
  standard: LibrePhotos round-trips face regions and names as MWG-RS,
  digiKam edits EXIF/IPTC/XMP in place. Read AND write, so another DAM
  can see what you decided here.

- **Reverse geocoding, as a suggestion.** `db/places.py:8` already
  names "a future reverse-geocoding job (cached by geographic cell)".
  The current rule -- GPS never mints a human place, a person authors
  it -- is right and must survive. What is missing is the middle step:
  GPS evidence produces a derived SUGGESTION, a person accepts or
  corrects it, and the acceptance is the authored claim. "We have your
  coordinates but refuse to mention they are in Detroit" is not the
  same virtue as "we did not silently decide for you".

## Human workflows we have the data for and not the product

- **Face correction.** The persistence model is arguably better than
  anyone's -- an accepted name survives reclustering, which is the
  whole point of separating authored claims from derived observations
  -- and there is no workflow around it. Merging two clusters,
  rejecting a bad match, reviewing unknown faces, choosing an exemplar:
  Immich, PhotoPrism and LibrePhotos all expose these. A durable model
  with no correction UI means the durability protects whatever mistake
  was made first.

- **A model that describes a picture does not know who is in it.**
  The library knows: `person_assertion` says this person appears in
  this file, optionally with the face region and the frame, signed by a
  user. The captioner is handed pixels and nothing else -- the
  `Captioner` protocol is `describe(image) -> str` with no prompt
  parameter at all (vision/captions.py) -- so a picture whose subject
  has a name for years is still captioned "a woman standing on a
  beach".

  The mechanism exists at both ends. BLIP takes a conditional prefix
  and continues it; Qwen-VL already runs under a system instruction
  here (`MEDIA_INSTRUCTION`, vision/semantic/qwen_vl.py) and can be
  told who is present and where their boxes are. What is missing is
  the seam between the two, and the rules about what may cross it.

  Four of those rules, because getting them wrong is worse than not
  building this:

  1. **Only a claim a human signed.** A name may enter a prompt only
     where `person_assertion.user_id IS NOT NULL`. A caption is prose
     somebody reads and believes, and it is searchable; a caption
     naming the wrong person is worse than an unnamed one, and a
     derived cluster match is not a good enough reason to write one.
  2. **The prompt is part of the producer identity.**
     `derived_annotation` is unique on `(file_id, kind, model_id,
     model_version, region_id, sample_id)` -- the prompt is not in it.
     So the same model captioning the same picture before and after a
     person was named would collide and one would overwrite the other,
     when they are two different observations of the kind this store
     exists to keep both of. The name-set fed in has to be recorded and
     has to participate in that uniqueness.
  3. **Do not launder an authored claim into a derived one.** If the
     model writes "Sarah at the beach", the word "Sarah" came from the
     authored name, not from the model recognising anybody. Baking it
     into derived text also freezes it: rename the person and every
     caption is quietly wrong until something re-infers them. The
     cheaper and more honest shape for DISPLAY is the opposite --
     caption "a woman in a red coat", and substitute the current name
     at render time from the assertion, so the name is always right and
     never needs re-inference.
  4. **Local only.** Face embeddings are biometric templates and this
     tree already says so; a name plus a face crop leaving the machine
     is the precise thing that doctrine forbids.

  Which leaves the question of what feeding names to a model is
  actually FOR, and it is worth answering before building it, because
  the two answers want different work:

  - **Better description.** Knowing two people are present and where
    they are changes what a vision-language model attends to, which
    improves the sentence whether or not it says a name. This is the
    case for the prompt seam, and rule 3 above says the sentence can
    still come back nameless.
  - **Search.** "Sarah at the beach" is the thing somebody types, and
    it fails today because the CLIP text encoder has never heard of
    Sarah and never will -- no amount of captioning fixes that, since
    the ranking is over image embeddings. The answer there is not the
    caption model at all: it is the query splitting into
    `person=sarah` (a filter the vocabulary already has) plus `q="at
    the beach"` (the phrase). That belongs with the field catalog, and
    it is probably the more valuable half.

- **Duplicate review.** `/jobs/dupes` and `/dupes` detect groups.
  Nothing resolves them, and the naive resolution is the one to avoid:
  byte identity and organisational identity are different things. Three
  copies of one file in `Iowa 2019`, `Family` and `Old Backup` are one
  content and three placements, and a deduper that celebrates "1
  duplicate removed" has silently turned a complete collection into an
  incomplete one. The operation worth building is not *delete
  duplicates* but *consolidate redundant storage while preserving every
  logical placement*, shown as a preview of the post-state before
  anything is touched:

      3 exact copies - SHA-256 identical
      Used by:  Iowa 2019  428/428 present
                Family     113/113 present
      After:    3 placements, 1 stored payload, all collections complete

  Hydrus is the deep reference for duplicate/alternate relationships;
  Immich for the review-and-keep flow.

- **The replacement gauntlet.** Nobody has put the incumbents on this
  library and lost to them on purpose. Until that happens, "ours is
  different" is an assertion. Install current **Immich** (as a
  read-only external library), **Infinite Image Browsing** (MIT, and
  alarmingly close to the generated-media half of this), **digiKam
  9.2** and **LibrePhotos** (MIT), and run real questions through each:
  all generated videos with this LoRA; what prompts dominate this
  answer; find this person in August; which files contain this
  screenshot text; compare three outputs and their recipes; fix a wrong
  face; resolve these duplicates; export my names so another DAM sees
  them; why does it think this happened in 2023. Verdict per app per
  question: **better than ours / good enough / painful but possible /
  impossible**. Then the rule is brutal -- where an incumbent wins and
  the slice is not structurally required by our model, stop maintaining
  the inferior reinvention. Integrate, copy the pattern, or delete ours.

  Licence note, because it changes what reuse means: IIB and
  LibrePhotos are MIT and can be read from and borrowed with
  attribution. Immich is AGPL-3.0 -- excellent to study, consequential
  to copy into an MIT tree.

- **Steal SwarmUI-Quarry's mechanism, not its database.** Quarry
  (`jtreminio/SwarmUI-Quarry`, MIT) is where Swarm's Image Search tab
  comes from, and its useful idea is not a list of fields: it promotes
  a small core, keeps every other `sui_image_params` / `sui_extra_data`
  property generically, and asks the index which keys the corpus
  actually contains. That is the field catalog above, independently
  arrived at. What NOT to take is its parallel path-keyed history
  index; `file_param` + `param_key` + one authoritative ResultSet is
  the better shape, and its recursive flattening is better than a JSON
  blob. Its metadata-extraction and filter-builder tests are a boring
  regression corpus worth porting with the licence notice.

  Its own rough edges are worth not copying either: creating a rule
  opens a native browser prompt, and it refuses large result counts.
  We had both of those defects until this week.

## Three flakes in the suite, characterised and not fixed

- **`test_writes_stay_linear.py` fails near its own tolerance.** The
  ratio gate is `< 2.0` and the measurements sit on it: across four runs
  it failed three times on three DIFFERENT cases (`writing a parsed
  field` 2.0019, `rescanning an unchanged library` 2.1, `renaming a file`
  2.4) and passed twice clean. Different case each time is the signature
  of machine-load sensitivity, not of one path that regressed. It needs
  either a bigger gap between SMALL and LARGE so the ratio is not
  measuring noise, or repeated timings with a median. Ignoring it is
  worse than either: a gate that fails one run in three teaches people
  to re-run it.

- **`test_browsing_does_not_stop_at_sixty.py` fails about one run in
  four, on a different case each time.** Measured at both revisions
  before it was believed: 4 runs at HEAD gave 1 failure, 3 runs with an
  unrelated change gave 1 failure -- the same rate, so it is the suite's
  and not any change's. Always the same shape: `_grew_to` times out at
  15s waiting for the next page to append, and the case that loses
  varies (`scrolling_to_the_end`, `it_keeps_going`,
  `dropping_from_the_top`, `scrolling_back_up`). Alone, every one of
  them passes. The trigger is a scroll-driven fetch racing the harness's
  own scroll, so the fix is in the test: wait for the loader's own
  signal rather than for a count to move.

- **`test_the_bytes_are_served.py` fails whenever something earlier left
  a running event loop.** Not xdist, as this was previously recorded:
  it reproduces on a plain single-process `pytest tests/ -m slow`.
  `test_a_raw_latin1_range_octet_is_answered_not_crashed` raises
  `asyncio.run() cannot be called from a running event loop`, and the
  unawaited coroutine it leaves behind surfaces as a
  RuntimeWarning inside whichever test the garbage collector reaches
  next — most recently
  `test_the_harness_owns_its_connections.py::test_a_kept_connection_is_held_rather_than_its_address`,
  which has nothing to do with either. `filterwarnings = error` turns
  that into a second failure some distance from its cause.

  Isolation proves the ordering: both files pass alone, with the
  pytest-playwright plugin loaded AND with `-p no:playwright`. Only the
  full run fails. The fix is in the test, not the application — it
  drives the ASGI app with `asyncio.run` and must instead use the loop
  it is already inside.

## Thumbnail delivery: done for the gallery, not for every surface

The gallery grid, the table, the rail preview and the compare tray now
point at `/thumbs/<shard>/<sha>.webp` -- content-addressed, immutable,
served with no database at all. Measured over one 60-cell page
(`just bench thumbs-delivery`, benchmarks/results/thumb_delivery.json):

    SQLite connections per page view   63  ->  3
    ...for the sixty thumbnails        60  ->  0
    cacheable responses              0/60  -> 60/60
    fan-out                        285.8ms -> 178.4ms

These surfaces still spell `/thumb/<slug>` and still pay a connection
per picture. Each needs the content hash carried on the row its page
already reads, then the same `thumbs.asset_url` call:

- `person.html` and `_person_drawer.html`
- `folder.html`, `album.html`, `artifact.html`, `artifacts.html`
- `_timeline_surface.html` -- the densest of them all: session strips,
  scrubber segments, month and day cells are dozens of thumbnails per
  page, every one a connection
- `frontend/src/timeline.ts` (two places)

Also not done:

- **Nothing is served by anything but Litestar.** The point of a
  content-addressed immutable path is that a static server or a cache
  in front can answer it without the application running at all;
  `create_static_files_router` over the thumbs directory, or a Caddy
  rule, would take even the ASGI dispatch out. Worth measuring before
  assuming it matters at this scale.
- **No sprite sheet or atlas, deliberately.** The measurement says the
  static fan-out is no longer material: sixty requests that touch no
  database and cache for a year are not the problem nine
  database-backed ones were.
