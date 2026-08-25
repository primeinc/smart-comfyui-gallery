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

- **Adding a root means pressing seven buttons in an order only the
  application knows.** scan, then ingest, then context, then events,
  then embed, then detect_faces, then cluster_faces, then annotate. The
  order is real -- `cluster_faces` over an unembedded library is a job
  that honestly settles `done` having clustered nothing -- and the
  application knows it and makes a person re-derive it. Two places
  already chain by hand (`precache_after_scan`, and `submit_ingest`
  queueing a `scan`), which is the evidence that the need is real and
  currently hand-rolled.

  The EDGE shipped: `job.after_id` gates the claim on its predecessor
  settling `done`, `job.collection` is the name the steps share, and
  `POST /jobs/catch-up` queues the derivation chain in order. A failed
  step cancels exactly what depended on it, transitively, and leaves
  every unrelated step in the collection running -- the product question
  this entry named, decided that way because a partial catch-up is
  normal and one unreadable file must not abandon four thousand others.

  The console collapses: a collection is one row with a bar and its
  steps nested under it, open while it is running or has failed.

  What is left, in order of what it earns:

  - ~~**A chain is only as correct as its least lazy step.**~~ Done:
    steps count their units when a worker claims them
    (`runner.COUNTERS`, `jobs.count_now`), the walk leads the chain, and
    a catch-up reads the files its own walk found. Per-file items are
    kept -- only WHEN the list is made moved. Pressing a sweep on its own
    still counts at submit, so "nothing to do" is still said then.
  - **Cancelling a collection** cancels its unstarted steps only when a
    step FAILS. Cancelling a running step by hand does the same thing by
    the same path, but nothing cancels a collection as a unit.
  - **Scheduling.** Now possible AND now worth it: there is a name to
    point at, and a catch-up finally notices files arriving, which is
    the whole difference between a nightly job and a nightly no-op.

  The original design note, for the parts not yet done:

  - **The console collapses.** Ten rows become one with a bar, and the
    little jobs stop mattering because they are steps rather than rows.
  - **The order stops being knowledge a person carries.** The claim
    gains a "every predecessor has settled" clause -- a WHERE, not a new
    engine, since `jobs.claim` already filters on state.
  - **Scheduling becomes possible at all.** A cron needs something to
    NAME. "Every night, catch up" points at a collection; scheduling
    individual kinds would mean re-deriving the order at 3am, which is
    the same defect with a timer on it.

  Grouping ALONE is not worth building. If a collection is only a label
  over rows, the console gets quieter and a person still does not know
  what to press. The dependency edge is the part that earns it.

  What has to be decided before it is coded, because it is a product
  question and not an implementation detail: **what a failed step does
  to its collection.** A partial catch-up is normal and useful -- one
  unreadable file must not abandon the other four thousand -- so the
  likely answer is "report it, stop only the steps that depended on it,
  let the rest finish". But that is a decision, and "the collection
  failed" is the other defensible one.

  Two smaller ones that follow: cancelling a collection has to cancel
  its unstarted steps, and a collection must not become a second
  scheduler -- the runner stays the only thing that runs jobs.

- **Benchmarks in the UI.** `just bench thumbs-phases` and the other
  benchmark recipes write JSON that only a terminal ever sees. The
  operations console is where a run's own facts already live; these
  belong there too, so throughput is something you can look at rather
  than something you have to run.

## Correctness

- **A v1 or v2 library reaches today with the wrong derived-face
  schema.** Measured 2026-08-25 by seeding from the schema that shipped
  (`tests/schemas/v01.sql`, `v02.sql`) rather than by inverting today's:
  after all 35 steps the file has no `derived_face_space`, neither
  `derived_face_space_agrees` trigger, no `derived_file_hash_space`, and
  still carries `derived_file_hash_phash`; `derived_face_instance` and
  `derived_file_hash` differ from a fresh build.

  Those objects entered `schema.sql` while it was stamped v3 and no step
  was written for them. `@step(2)` says "purely additive for real this
  time" and creates one table; `@step(3)` says "version 3 drifted during
  development" and repairs that drift for `similarity_space` and
  `derived_embedding` only.

  Pinned by `KNOWN_DRIFT` in
  `tests/test_a_database_survives_an_upgrade.py` so it cannot widen
  silently, and NOT repaired: the fix is a step that reconciles those
  objects on any database, and the population it would serve is
  libraries that started at v1 or v2, which may be nobody. Decide
  whether that population exists before writing it.

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

- **The field catalog is built; two pieces of it are not.** One
  searchable list over the curated vocabulary and every discovered
  metadata key now answers the Add-filter box (db/catalog.py,
  `/g/fields`, frontend/src/filters.ts `mountFind`). Indexed families
  collapse, the observed type comes from `param_key.value_kind`, and
  ranking is by what would cut THIS answer, with a camera's plumbing
  ranked down rather than hidden.

  What is not built:

  - **A field does not choose its own operators yet.** The catalog
    sends `value_kind` and `ops` per field and the surface ignores both
    for discovered keys, which always get `is`. A number-kinded key --
    `param_key.value_kind` says which, and `Steps` and `CFG scale` are
    the obvious ones -- should offer above / below / between, which is
    the machinery `drawRange` already has.

    This one is not only a surface change, which is why it was not done
    with the value list. `param.is` compares `fp.value_text` and admits
    `eq` and `any` only (db/facets.py). A range wants `fp.value_num`,
    which the schema populates when a value parses as a number and
    already indexes for exactly this (`file_param_key_num ON
    file_param(key, value_num) WHERE value_num IS NOT NULL`). So the
    work is a second spec beside `param.is` -- call it `param.atleast`
    / `param.atmost` -- rather than widening one whose SQL compares the
    wrong column.

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

- **The table sorts by five of its columns, not by the joined ones.**
  Name, kind, size, pixels and length are sorts the ResultSet
  understands (`resultset.COLUMN_ORDERS`), each with its reverse, each
  total on `(column, f.id)`. What is not sortable is every column that
  is a LEFT JOIN -- checkpoint, sampler, steps, cfg, seed, camera, iso,
  aperture, focal length, and the authored rating.

  Not a copy-paste of the five. Each needs the join in the ordering
  statement, and `rating` needs the ACTOR bound into it, which means an
  argument in the ORDER BY as well as the WHERE. And the NULL question
  gets sharper: a photograph has no sampler, so "sort by sampler" is
  mostly a list of files that have none -- the position is honest but
  probably not what was wanted, and whether such a sort should also
  narrow to the files that HAVE one is a product decision, not an
  implementation detail.

- **The comparison has no zoom, so it cannot have a synchronised one.**
  Flipping is built -- one at a time in the same pixels, lettered, Space
  and the arrows, every column decoded so the flip is a repaint -- which
  was the half that answers "did this change". What is still missing is
  the half digiKam's Light Table is actually the reference for: zooming
  into a detail on one and having the other follow, so two 4k
  generations can be compared at the grain rather than at the thumbnail.

  The viewer already owns zoom and pan over one picture
  (frontend/src/viewer.ts: fit / fill / actual / free, with a tether so
  a zoomed picture cannot be flung off screen). The comparison shows
  `object-fit: contain` and nothing else. So the work is not new
  arithmetic, it is deciding what a shared transform MEANS across
  pictures of different shapes -- same scale, or same fraction of each
  frame? -- and those give different answers for a 3:2 beside a square,
  which is exactly the pair somebody is comparing.

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

- **A video with nobody in it is the most expensive video to look at.**
  Not a guess about the code -- it is what the code is for.
  `harvest_video` samples a cadence, and *because* it found nothing it
  then bisects the widest gaps, up to `REFINE_MOST = 32` extra decoded
  frames, and only stops when faces appear (db/detect.py). That
  refinement is right: a fixed interval can land every moment on the
  establishing shot of a clip that is otherwise all people, and without
  it those people are simply missed. But it means a landscape timelapse
  pays the cadence, then pays thirty-two more decodes, then records that
  it found nothing -- and a library of screen recordings and B-roll
  spends most of its face budget proving negatives.

  What is missing is a cheap gate BEFORE the expensive certainty, and
  the library already computes the signal for it: `derived_embedding`
  holds a semantic vector per file after the embed job, and the cosine
  against an encoded phrase like "a person" is arithmetic on a vector
  already in the database. A zero-shot person detector for free, on a
  file that has been embedded. `db/retrieval.py` already knows how to
  ask a space a question and where a score stands relative to the rest.

  Three things it must not become:

  - **Not a silent skip.** A false negative hides a person for ever and
    reads as a broken face pipeline, which is the worst failure this
    subsystem has. A gated file must record that it was gated, with the
    threshold and the model that gated it -- `derived_face_scan` already
    exists to say a file was looked at and what was found; "looked at
    cheaply and declined" is a third state, not an absence. Then a later
    run at a lower threshold, or a person saying "look properly", can
    revisit exactly the right files.
  - **Not a new hardcoded number.** The threshold is another knob, and
    belongs with the entry below rather than as a constant in
    db/detect.py.
  - **Not assumed to pay.** The gate is only worth building if decode is
    where the time goes, and that is measurable rather than obvious: the
    embed benchmark already says decoding is 39% of that job. The
    equivalent split for `detect_faces` over a video-heavy corpus should
    exist before the gate does, along with the recall cost at each
    candidate threshold -- how many face-carrying videos a given cut
    would have skipped, checked against the ones already scanned. That
    measurement is most of the work and all of the confidence.

  A cheaper variant needs no embedding and could gate the REFINEMENT
  rather than the whole file: the cadence is already decoded and already
  found nothing, so the question is only whether to spend the extra
  thirty-two. Deciding that from what the cadence frames looked like is
  a smaller claim than deciding a video has nobody in it, and it targets
  exactly the frames that are pure cost today.

- **A verdict on a similarity or a duplicate still cannot be given.**
  (Was: "this is not that person" cannot be said -- that half shipped.)
  The positive claim is built and the doctrine around it is
  the best thing in this schema: `person_assertion` is a human saying
  "she is in this picture", and `db/derived.py` re-applies it after
  every reclustering rather than re-guessing by centroid similarity.
  The negative claim has nothing.

  Three separate gaps, and they compound:

  1. ~~**No way to spell it.**~~ Shipped: `person_assertion.stance`,
     `deny_person`, the route, and the control on the media inspector
     and over every thumbnail on the person's own page.
  2. **The similarity and duplicate arms.** The annotation arm ships
     (`POST /i/{slug}/said/verdict`) and the person arm is now written
     by correcting a face rather than by a separate gesture
     (`db/authored.py deny_person`). A verdict on a SIMILARITY or a
     DUPLICATE still has no surface, and shipping a general endpoint
     whose remaining arms nothing exercises would be two contracts
     nobody has tested.
  3. **Only the annotation arm is rated.** `db/verdicts.py by_producer`
     and `contests` read the annotation arm; corrections read as a
     count (`corrections`), which is all a denial-only sample can
     support. Nothing yet reads a verdict to CHANGE anything -- no
     threshold moves, no model is deselected, no re-run is suggested.

  Also worth fixing while it is open: `feedback` points at a FILE, not
  at a face. There is no `region_id` on it, so "the face in the corner
  of this picture is not Sarah" is unspellable even in the table --
  only "the person judgement about this file was wrong". A picture with
  two faces cannot say which one is the mistake, which is the ordinary
  case.

  The reason this matters more than a correction UI: a negative claim
  is the highest-value input clustering can take. A false merge is one
  cannot-link edge away from being fixed for ever, on every future run,
  where a threshold is a global compromise that trades somebody else's
  correct grouping for this one. See the entry below.

- **A verdict on anything a model said has no surface, no aggregate and
  no way out.** The table is general and was designed for this:
  `feedback` judges an `annotation`, a `similarity`, a `duplicate` or a
  `person`, with `verdict IN ('right','wrong','unsure')` and a free
  note, and its pointers are ON DELETE SET NULL on purpose so dropping
  the whole derived namespace leaves the judgement standing. It is the
  one authored table whose subject is disposable. Nothing writes it but
  a test.

  The first of three is BUILT for captions: a thumb beside the sentence
  in the inspector, one click, retract by clicking the lit one, and
  `feedback` carries the producer it judged (v34) so the aggregate below
  is possible at all. What is left:

  1. **The other three things it can judge.** Only the annotation arm
     has a route. A face chip on a picture, a duplicate group and a
     similar-to row each render a derived claim with nowhere to say it
     is wrong -- and the person arm is the one that matters most,
     because "not her" has to constrain clustering rather than only
     being counted (see above).
  2. **An aggregate that says what to change.** "BLIP base: 340
     captions, 41 judged wrong" is the number that tells a person to
     try another model; "arcface at 0.48: 12 merges rejected" tells
     them to move a knob. This is the honest reason to collect verdicts
     at all, and it is what makes the two entries below actionable
     rather than a settings page nobody knows how to set. Nothing reads
     the verdicts yet.
  3. **Exportable without the pictures.** Verdicts are the cheapest
     valuable thing this application accumulates and the easiest to
     share safely: a row is a producer identity, a kind, a verdict, a
     content hash and a timestamp. No pixels, no names, no paths, no
     embeddings -- so an eval set of "this model got these 41 wrong"
     leaves the machine without any of the media leaving with it. That
     is the privacy-forward shape, and it should be the DEFAULT export,
     with anything richer an explicit opt-in per field.

  And then, once there is enough of it, **infer** rather than count.
  The yes/no data is the only ground truth this application will ever
  have about its own models on THIS library, and a count is the weakest
  thing to do with it. What it can reasonably support:

  - **Which producer to prefer.** Two caption models over the same
    files with verdicts on both is a direct comparison on the corpus
    that matters, not a benchmark somebody else ran. Same for two face
    backends, whose embedding spaces never mix anyway.
  - **Where a model fails.** Verdicts join to everything the library
    already knows -- kind, capture time, folder, camera, whether it is
    generated. "Wrong 8% overall, 40% on video frames" or "on this
    folder" is a fact worth showing, and is the difference between
    "captions are bad" and something a person can act on.
  - **Which knob a rejection is about.** A rejected merge has a face
    pair with a cosine distance. A run of rejections clustered just
    above the operating point IS the argument for moving it, and it can
    be stated as one: "17 of your 20 rejected merges scored between
    0.48 and 0.52".

  Three rules, because this is the kind of surface that lies
  confidently:

  - **It is a biased sample.** People judge what they look at, and they
    click `wrong` more readily than `right`. So a raw error rate is
    unpublishable; what is honest is a comparison BETWEEN producers
    over the same judged set, where the bias is shared.
  - **Say the n, and say nothing under it.** A model is not worse than
    another on four verdicts. Below a floor, the surface should say how
    many more judgements it would take, not show a number.
  - **Never present a correlation as a cause.** "Wrong more often on
    video" is an observation and should read as one, with the door to
    the files it came from so somebody can go and look.

  The negative person claim above is a special case of this and should
  not be built as one: "not her" needs to constrain clustering, which a
  `wrong` verdict on a file cannot do. Same gesture, two destinations.

- **The DETECTION knobs are still constants in a module.** The
  clustering operating point is now the `face_cluster_threshold` setting
  ("auto" for the measured per-embedder point), validated at submit,
  pinned into the payload, and shown per run on the operations console
  beside what it produced. The two guards it needed were already there
  or are now: the threshold is part of `derived_face_run`'s identity, so
  a new one writes a new run beside the old rather than over it; and the
  console says plainly that it changes what the NEXT run computes.

  Still unreachable, each an edit and a restart:
  `_LABEL_MATCH_THRESHOLD = 0.9` (a recomputed cluster inheriting an old
  label), `min_det_score=0.5`, `min_face_px=24` (both vision/faces.py),
  and `FLOOR = 0.7` in db/detect.py. These are DETECTION rather than
  grouping, so they have a property the threshold does not: changing one
  changes which faces exist at all, and a face nothing detected cannot
  be recovered by re-running the clustering. Exposing them needs the
  re-detect path to be as cheap to undo as re-clustering is, and it is
  not.

  `pages.disagreements` is now the console's "what changed against the
  primary", bounded and saying how many more there are.
  `pages.face_across_runs` -- per FACE, how big a group each run put it
  in -- is still called by nothing. It answers a different question from
  the per-picture view and a sharper one: a face one run puts with fifty
  others and another puts alone is where two clusterings actually
  differ, and no picture-level diff surfaces it.

  And the harder half, which is why (a) is not simply better: a global
  threshold cannot express "these two are the same and those two are
  not". `db/grouping.py METHODS` takes a graph and vectors and no
  constraints, so every authored claim is applied AFTER the fact, by
  re-attachment. Feeding must-link / cannot-link edges into the graph
  itself is the version worth wanting -- it makes each correction
  permanent and local instead of trading it against a global compromise
  -- and it needs the entry above to exist first, because there are no
  cannot-link claims to feed it.

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

- **Duplicate CONSOLIDATION.** The review shipped: `/dupes` is a page,
  linked from the shell, showing every group, each copy's folder and the
  collections it is filed under, and whether the copies are byte-
  identical or merely alike -- with the post-state sentence ("3
  placements, 1 payload, every collection still complete") offered only
  where the bytes actually match. It is read-only and removes nothing.

  What is left is the operation itself, and the naive version is still
  the one to avoid:
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

  The page states the last line already. The missing half of the preview
  is the collection-completeness count ("428/428 present"), which needs
  a query nothing has: how many of a collection's members would still
  resolve after a given consolidation.

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

The person, folder, album and artifact pages joined it:
`thumbs.address` points a page of an answer at its assets from the
`sha` and `kind` the ResultSet already read, and a folder page's
thumbnails now open zero connections.

What is left is the TIMELINE, and it is the densest of them all --
session strips, scrubber segments, month and day cells, frames and bins
are dozens of thumbnails per page, every one a connection. It is not a
fifth copy of the same edit, which is why it is still here:

- Its pictures do not come from a ResultSet page. A dozen separate
  statements in `sg_web/timeline_view.py` hand out bare SLUGS -- a
  session's samples, a bin's sample, a day's hero, a month's hero, a
  segment's strip, a scrubber face -- and none of them selects
  `content_sha256`. So the work is a dozen statements widened, not one
  call added.
- Several are lists of slugs with no kind beside them, and
  `asset_url` needs the kind to answer None for audio and documents.
  A timeline of a mixed library would otherwise draw broken images
  where the grid already knows not to.
- `frontend/src/timeline.ts` builds two of these addresses in the
  browser, so the shape it is handed has to change with the templates.

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
