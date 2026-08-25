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

- ~~**Anything expensive the program learns should be exportable
  without the program.**~~ Shipped, and it is now three exports rather
  than one, each with a different audience:

      /operations/export/verdicts.json       what you judged, to SHARE
      /operations/export/authored.json       what you said, for CUSTODY
      /operations/export/faces/<slug>.json   what it learned, with proof

  The third is this entry: a person's face vectors, grouped by the
  immutable `similarity_space` that gives them meaning -- producer,
  version, preprocessing, metric, dimensions, spec hash -- with the
  cluster's own centroid, each face named by the `content_sha256` of
  the picture it was found in, its region, its detection score and its
  capture time. Offered on the person's own page, not a console, and
  the link says these are BIOMETRIC TEMPLATES before somebody puts them
  in a file.

  Grouped BY SPACE rather than flattened: a vector is comparable only
  to another from the same one, and a library re-detected under a new
  model holds two representations of one person.

  A date range is over CAPTURE time, so a picture whose camera never
  said when excludes itself the moment a range is given -- the honest
  reading of "their faces from 2019", stated rather than silent. With
  no range the undated sort LAST, because SQLite sorts NULL first and
  would otherwise have led the file with the least locatable pictures.

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

- ~~**Thumbnails are still serial.**~~ Fixed. An item now renders its
  own thumbnails and the next few of its job's pending items beside
  them, so those are on disk when their turn comes and take the
  `already-cached` return. The runner still works ONE item at a time --
  started, committed, worked and settled on its own -- which is what
  resumability, cancellation at a boundary and per-item failure all rest
  on. Only WHEN the pixels are computed moved.

  The same bargain `_Ahead` makes for vectors, and a cheaper one: a
  vector must be held in memory because a row written ahead would not be
  safe, while a thumbnail's result IS a file in a content-addressed
  cache. Rendering one early is exactly what the job would have
  produced, so nothing is held and a cancel undoes nothing.

  Benchmarked 1/2/4/8 as this entry asked, end to end through
  `run_next`, 32 pictures at 4000x3000:

      1 in flight     4.64 files/sec
      2               9.55            2.1x
      4              16.95            3.7x
      8              23.55            5.1x

  The renderer alone measured the same shape and found the knee: 8 gives
  6.0x, 12 gives 5.9x and 16 gives 5.4x, because past it the two thread
  pools oversubscribe. So the number is half the cores, capped at eight,
  never below two.

  Worth recording that the obvious reading was wrong: libvips already
  uses every core to calculate ONE image
  (`../refs/libvips/libvips/doc/using-threads.md`), which sounds like an
  argument that a second file in flight can only fight the first. One
  thumbnail is not enough work to fill sixteen cores; the win is across
  files, and libvips is documented thread-safe for exactly that -- its
  own example runs fifty threads resizing at once. Only the drawing
  operators and Regions are not.

  Video stays on the single path, as it does for embedding: a seek and
  a decode of a different shape does not belong inside a group whose
  size is bounded for somebody waiting on a cancel.

  ~~Interactive work should also outrank precache.~~ Done, and the
  reason it became urgent is that rendering several at a time made it
  worse: the queue now fills every core while GUESSING at what will be
  wanted later. Measured on a grid of 30 misses served while a precache
  ran, 4000x3000 throughout:

      precache 1 in flight              1.25s   42 ms/cell
      precache 8, not standing aside    1.81s   60 ms/cell
      precache 8, standing aside        1.53s   51 ms/cell

  So a browser blocked on a cell marks itself (`derive.waited_on`) and
  the speculative side does not START another picture until nobody is
  (`derive.stand_aside`). Half the regression, and the shortfall is
  honest rather than a bug: a render already running cannot be
  interrupted, so a person arriving mid-batch pays the tail of what is
  in flight and no more. Bounded by `PATIENCE`, so a marker that leaks
  costs the queue a pause and never the queue.

  An Event rather than a lock or a semaphore, because the two sides want
  different things: the person must NEVER wait -- they are the work
  being prioritised -- and the queue only has to hold off starting.

  ~~Two requests for the same missing key render it twice.~~ Fixed and
  measured: four concurrent requests for one missing thumbnail rendered
  it four times, started within 1 ms of each other, each decoding,
  resizing and encoding the same picture on its own database
  connection. Now one render and four answers (`app._rendered_once`).
  Per FILE rather than one lock over all rendering, which would have
  serialised a fresh library's whole grid behind one picture at a time
  -- pinned by a test where two different pictures must meet inside the
  renderer at a `threading.Barrier`, which they cannot do if one is
  waiting for the other.

  Per process, deliberately: the bytes land through a staging name and
  `os.replace`, so racing writers produce identical bytes and a replace
  that changes nothing. This was never a correctness bug, only waste.

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

- ~~**Adding a root means pressing seven buttons in an order only the
  application knows.**~~ Closed: the headline and all three sub-items
  below are done, and the section with it. scan, then ingest, then context, then events,
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

  Stopping one is one press: queued steps end, a running one is asked
  and stops at its next item boundary.

  What is left, in order of what it earns:

  - ~~**A chain is only as correct as its least lazy step.**~~ Done:
    steps count their units when a worker claims them
    (`runner.COUNTERS`, `jobs.count_now`), the walk leads the chain, and
    a catch-up reads the files its own walk found. Per-file items are
    kept -- only WHEN the list is made moved. Pressing a sweep on its own
    still counts at submit, so "nothing to do" is still said then.
  - ~~**Scheduling.**~~ Done: `schedule` is one row per collection, an
    interval in hours, set on the operations console, started by the
    runner on the worker's own turn. A collection already going is never
    started twice, and the clock runs from the START so a long catch-up
    on a nightly schedule does not drift later every day.
  - ~~**A scan says what it left undone.**~~ Done, and it was the
    headline surviving at a smaller scale. A scan LISTS files and
    derives nothing, so somebody who had just registered a library was
    looking at pictures with no metadata, no search and no people --
    while the button that fixes it sat in a different section further
    down the page. The roots panel now names the sweeps still
    outstanding with their counts and offers the chain in place
    (`operations._behind`, `_operations_roots.html`). Sweep names and
    never a total: a file missing both a reading and a caption is one
    file and two sweeps.

- ~~**Benchmarks in the UI.**~~ Shipped, narrower than this entry
  assumed, and the files are why.

  There are TWENTY-THREE documents under `benchmarks/results/` and **no
  key is shared by all of them** -- calibration sweeps, recall tables,
  backend equivalence evidence. Rendering "the benchmark JSON" would
  have been a JSON viewer. Exactly THREE measure throughput and agree on
  a shape, because one script writes them (`benchmarks/job_phases.py`):
  scan, embed and annotate. Those are on the console now, with their
  rate, the item count, the wall time, the cores they used and where
  the time went by phase.

  Two things the entry did not name and the files made necessary:

  - **Four of the twenty-three carry real filesystem paths.** Not the
    three shown -- but that is a fact about today, so it is asserted
    against the FILES as well as the rendered page, and a later
    benchmark that starts recording paths fails the test rather than
    quietly reaching a screen.
  - **None of them carries a timestamp.** A rate on screen with no date
    is a claim about whatever the tree looked like when somebody last
    ran the recipe, and this tree moves -- thumbnails went 4.64 to 23.55
    files a second in one afternoon. The file's mtime is shown and the
    panel says the numbers were RECORDED, not observed.

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

- **Substring bans could not tell a statement from a sentence.** The
  false-positive class is closed; the reclassification is not.

  `MUST_NOT_CONTAIN` searched the WHOLE file text, so a comment saying
  "this module never DELETEs" was the module agreeing with its own rule
  and being failed for saying so -- and the failure read as an
  architecture violation rather than a word, which is how
  `media_view.py: "neighbour"` came to be DELETED rather than
  satisfied.

  It reads code now (`rules._code_only`): comments and docstrings are
  blanked, with spaces so line numbers still point where they came
  from. **String literals are deliberately kept** -- `story_view.py`
  may not reach for SQL and SQL lives in a string, so blanking those
  would have turned a real ban into a decoration. Non-Python files are
  left alone; a Jinja template has no docstrings and `{# #}` is not a
  Python comment.

  Three cases in `test_sglint_has_teeth.py` had been injecting the
  banned word as a COMMENT, which fired only because the rule read
  prose. They inject SQL in a string now, which is the shape the test's
  own name says it is about.

  Still worth doing, and now optional rather than urgent: the 32 bans
  that name something used five or more times elsewhere would each be
  better as `MUST_NOT_CALL_QUALIFIED` (a call, by AST),
  `MUST_NOT_CONTAIN_BEFORE` (not above this marker) or
  `PACKAGE_FORBIDDEN_PATTERNS` (a regex with a stated reason). The
  mechanisms exist; what is gone is the failure mode that made it
  urgent.

- ~~**A shared filename made the first scan quadratic.**~~ Fixed, and
  found while measuring the entry below. `mint` picked a free slug by
  probing `cover-2`, `cover-3`, ... one SELECT at a time, so the nth
  file sharing a name cost n reads and the library cost n^2 -- and
  files sharing a name is how libraries are ORGANISED: a cover.jpg per
  album, two camera cards both numbering from DSC00001, a folder.jpg
  per artist.

  Measured: 4,000 files all called cover.png held the write lane for
  16,811 ms against 630 ms for 4,000 distinct names, quadrupling with
  every doubling -- twenty thousand albums extrapolates to minutes of
  held lane, during which every write from a route FAILS rather than
  waits. Now 1,657 ms, ten times faster, and no longer quadratic.

  Doubling-then-bisecting over the `UNIQUE (kind, slug)` index, which
  is O(log n) point lookups. Asking SQLite for the highest suffix
  instead was tried and measured first: `max(CAST(substr(slug, ...)))`
  is a max over an EXPRESSION, cannot use the index, scans every
  `cover-` row, and was still quadratic at 2,421 ms.

  It also stopped handing out RETIRED slugs. The old probe took the
  lowest free suffix, so deleting `cover-3` gave the next picture that
  address -- and `slug_history` answers a retired address on a miss, so
  somebody's saved link resolved to a different picture. Pinned by
  `tests/test_a_shared_name_does_not_make_a_scan_quadratic.py`, which
  counts statements rather than timing them, so it says the same thing
  on any machine.

- **A big enough FIRST scan still crosses `busy_timeout`.** Measured
  2026-08-25 on synthesised libraries, with a control write proving the
  symptom rather than extrapolating to it:

  | scan | lane held, 60,000 files | per 1000 | crosses 5 s at |
  | --- | --- | --- | --- |
  | first scan | 9,498 ms | 158 ms | ~32,000 files |
  | rescan, nothing changed | 510 ms | 8.5 ms | ~590,000 files |
  | rescan, one new file | 503 ms | 8.4 ms | ~590,000 files |

  At 60,000 files a competing write waited 5,575 ms and got `database
  is locked`; the same write with no scan running took 0.1 ms. At
  40,000 it waited 5,273 ms and squeaked through, because the wait is
  not the whole hold -- it is whatever REMAINS from when somebody
  presses, so above the crossing a growing fraction of presses fail
  rather than all of them.

  The earlier reading of this entry was wrong about which scan hurts. A
  rescan is nineteen times cheaper per file (the `was` comparison skips
  rows that did not change), so this is a FIRST-IMPORT problem, and the
  writes exposed to it are whatever a person does on the console while
  their new library is being read -- changing a setting, adding a
  second root -- not rating a picture, since nothing is in the gallery
  to rate yet.

  The fix is still to stop making the write half one transaction, and
  the seam is narrower than "commit in bounded batches": steps 1-3 of
  `_apply` (mark missing, park names, move rows) are interdependent and
  are nearly free on a first scan, and step 4 -- minting new rows -- is
  where the 9.5 seconds goes and is INDEPENDENT PER ROW, because it
  runs only once every name it might want has been vacated. So the
  atomic part can stay atomic and only step 4 needs splitting. What a
  half-applied step 4 leaves is a scan that added some new files and
  not others, which is what an interrupted scan already leaves and is
  healed by scanning again -- no row is stranded under a `?parked-`
  name, because parking finishes inside the atomic part.

  Note that `scan()` wraps `record` + `apply_scan` in one savepoint of
  its own, so the split has to happen there too, and its comment says
  the folder writes belong inside the file writes' savepoint. That is
  the invariant to argue with before touching this.

  **And there is a second route, which this entry did not have.**
  Measured 2026-08-25, 15,000 new rows, two runs per arm: the write
  half is not the write, it is the TRIGGERS.

      as shipped                     88.8 us/file
      without the name_fts trigger   51.2
      without the generation counter 83.6
      without the kind check         87.8
      the bare INSERT, all dropped    9.1

  So 90% of what a first scan holds the lane for is index maintenance,
  and the filename FTS index is nearly all of it -- keeping only that
  one costs 83.2. (The savings do not add up because SQLite pays most
  of the trigger cost for having ANY trigger on the statement; drop one
  of three and the machinery still runs.)

  That makes deferring the FTS population -- once per scan rather than
  once per row -- an alternative to splitting the transaction, and a
  different trade: it costs "filename search is behind during a scan"
  instead of "a scan is one atomic reconciliation". Which is the better
  price is a real question, and it is now a question with numbers.

  Whichever is chosen, `mint` is not the problem: 14.8 us of the 108,
  and its collision probe is 4.7 of that.

- ~~**`neighborhood`'s test matrix is short two cases.**~~ Both written,
  and the filmstrip was right under both -- no bug, and now it cannot
  regress unnoticed.

  Worth recording how the FILTERED one nearly did not test anything. It
  first narrowed by FOLDER, which reads like the obvious filter and
  proves nothing here: the fixture stamps one folder entirely after the
  other, so a folder's pictures are one contiguous run of the timed
  order, and a strip that walked the whole LIBRARY around one of them
  returned the identical seven. Checked before believing it. The filter
  is every-other-picture now, so the excluded ones interleave: the same
  library-walk leaks four pictures the person's answer does not
  contain, which is what the assertion is for.

  The SEMANTIC one ranks through the `retrieving.answered` double over a
  ROTATED order no SQL sort produces, and asserts retrieval ran once --
  so the strip is proved to read the materialized ordering the grid
  read rather than re-running the fusion.

- ~~**No cold acceptance lane.**~~ `just acceptance-cold`.

  `git checkout-index` writes what a CLONE would get -- no `.venv`, no
  `node_modules`, no build output that is not committed -- and then the
  two lines under "Run" in README.md are the only thing that happens: a
  real process, a real socket, a real page.

  What it catches and nothing else does: the README says "No Node, no
  npm -- the browser bundles are committed", and every other lane here
  builds them first (`web::build` is a dependency of test, check and
  smoke). A bundle rebuilt and never committed is therefore invisible
  everywhere except in a clone, where the symptom is a page that renders
  with scripts that 404 -- the pictures arrive and nothing about them
  works. So every `/static/` asset a rendered page names is fetched and
  its status checked; four today, `gallery.js` among them.

  Outside pytest deliberately, which is what
  `test_the_documented_launch_serves_a_whole_application.py` already
  said in its docstring -- it costs an install, and pytest is not where
  that belongs. Run once end to end before this was written down: 200
  on the page and on all four assets.

  `grep`/`curl`/`seq` rather than `rg`: a lane that proves a cold
  machine can run this must not need a tool a cold machine lacks.

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
  since before this.

  Triaged 2026-08-25, and the count in this entry was the wrong unit.
  ty reported 89 diagnostics, but many are one call site reported once
  per overload: `test_the_resultset_is_authoritative.py:276` is eleven
  of them. **48 distinct sites** was the real number, now **18** (21
  diagnostics).

  (An earlier revision of this line said 16, which was a miscount --
  the count command was run against a tree that had not been re-checked.
  The number here is what `uvx ty@0.0.74 check` reports today.)

  Two were the real bugs this entry named, and one of them is fixed:

  - `db/capture.py:101` -- FIXED. `_CLAIMED` was inferred `set[Base]`
    and the very next statement unioned `set[GPS | IFD]` into it. Right
    at runtime, since these are all IntEnum and are tested against the
    raw integer keys of an EXIF dict; wrong as a claim. Both membership
    sets say `set[int]` now, which is what they hold.
  - `db/planning.py:1367` and `:1375` -- NOT fixed, and it is not the
    quick one it looks like. `STORY_PLAN_V3["claims"] | frozenset(...)`
    fails because the plan dicts are heterogeneous, so every key infers
    as `int | frozenset[str]`. The code is correct; the type is
    invisible. Fixing it means a TypedDict over a structure whose own
    comments say FROZEN, and `respect-type-ignore-comments = false`
    means this tree does not get to silence it instead.

  **Almost all of the rest is one pattern**, not 32 separate defects: a
  dict literal holding an int, a str, a None and a list infers every key
  as the union of those, so `held["pictures"].append(...)` is `.append`
  on `int | None | ...`. The fix is a TypedDict per shape, and it is
  worth doing for its own sake -- the checker found two annotations that
  were simply wrong while this was being written, `_drawn` returning
  thumbnail ADDRESSES where the shape said picture rows.

  `sg_web/timeline_view.py` is done as the worked example: 12 sites to
  0, four shapes spelled out (`_Segment`, `_Cell`, `_Bin`, `_Group`).
  Annotated dict literals rather than constructor calls, so the readable
  form and the checkable one are the same thing.

  Also fixed, all the same pattern:

  - `db/pages.py:1403` -- the second real one. The signature and the
    docstring both said `{bin: [(slug, sha, kind), ...]}` and the local
    said `dict[int, list[str]]`. The annotation was the lie.
  - `db/stories.py:67` -- one site, NINE diagnostics: `**_CANONICAL`
    into `json.dumps` offered `bool | tuple[str, str]` to each keyword.
  - `metaparse/adapters.py:214`, `sg_web/collection_view.py:619` --
    `.rsplit` and `.append` on the union a heterogeneous literal infers.

  The test bulk is gone, and it was three annotations for 27
  diagnostics. All the same shape -- a TABLE of request-shaped arguments
  spread with `**` into a function whose parameters are differently
  typed. One thing learned doing it: **a declaration on the loop
  variable does not reach a for-target**, whose type comes from the
  iterable's elements. `refused: dict[str, Any]` above
  `for refused, why in (...)` had been written and did nothing;
  annotating the TUPLE is what works.

  `db/planning.py`'s FROZEN plan dicts are done, 4 sites to 2, and the
  shape is why one TypedDict was not enough. `settings` is a flat set of
  keys through v3 and one set PER PLANNER from v4 on -- it changed
  exactly once. A single dict with
  `frozenset[str] | dict[str, frozenset[str]]` merely moved the
  complaint: it says v3 might carry a dict and v5 might carry a set, so
  every construction in the chain has to be read as if it could, and the
  count came back to 4. Two names -- `StoryPlanFlat` and `StoryPlan` --
  say what actually happened and cost nothing, because a frozen version
  is frozen and both shapes are live for ever.

  Six more went as one pattern: an attribute PATCHED with a
  differently-typed function -- `backend._embed = record`,
  `manager.upsert = broken`, `connect.connect = counting`. They are
  `monkeypatch.setattr` now, which the checker does not object to and
  which is better besides: it puts the real one back when an assertion
  inside the patched region fails, where a hand-written try/finally
  only does if somebody wrote it, and two of these had none.

  **THREE errors left.** Five more went, and two of them were real
  annotation defects rather than checker noise:

  - `vision/faces.py` narrowed on `callable()`, which narrows to "some
    callable" and loses the signature, so the call could not be checked
    at all. It asks `isinstance(..., Mapping)` now -- the half with a
    runtime-checkable ABC -- and both branches have a type.
  - `plan_snapshot` was annotated `planner: GenerationHistoryPlanner`
    and is handed any of THREE. A `Planner` union now sits beside the
    `PLANNERS` registry, because the two lists are the same list and
    one drifting from the other is how that happened.

  The three that remain are each blocked on something outside this
  tree, and none of them blocks the gate more than the others:

      db/planning.py:2098   `maker(None, ...)` is safe because
                            `if maker.uses_similarity` guards it, and a
                            checker cannot see a guard on a class
                            attribute. Annotating it `| None` would be a
                            LIE: that planner reads `.similarity.name`.
      qwen_vl.py:358        transformers types `apply_chat_template`
                            more narrowly than it accepts.
      thumb_delivery.py:86  a genuine RUFF/TY standoff: ty rejects the
                            assignment, and ruff's B010 forbids the
                            `setattr` the tests use through monkeypatch.
                            No form satisfies both.

  So the fast gate is three sites away, and all three want a decision
  about SUPPRESSION rather than more typing -- which the overrides
  mechanism above can now express, per file and with a reason.

  **The count that matters is 17 ERRORS, not 21 diagnostics.** ty exits
  0 when only warning-level violations remain
  (`../refs/astral-sh/ty/docs/rules.md:16`), so the three
  `redundant-cast` warnings do not block the gate -- which is just as
  well, because they are a genuine two-checker conflict: those casts
  exist FOR pyright, whose `Module.to` sees the wrapped descriptor
  transformers decorates it with. Removing them to please ty would
  break `just check-deep`.

  And a correction to what this entry said last: **a suppression
  mechanism does exist.** `[[tool.ty.overrides]]` takes a glob and
  per-rule severities
  (`../refs/astral-sh/ty/docs/reference/configuration.md:551`), which is
  not what `respect-type-ignore-comments = false` forbids -- that bans a
  COMMENT silencing an error on the author's say so, where an override
  is a named file, a named rule and a reason in the file everyone reads,
  one grep from a list of every exception granted. Exactly the shape
  `sglint/policy.py` already uses.

  It was not needed for the jinja line, and that is worth keeping: an
  override is per FILE, so ignoring `invalid-assignment` across a
  2000-line composition root to silence one line would stop checking
  everything else in it. A local `cast` says the same thing in one place
  -- and the tree already casts for exactly this reason in
  `vision/captions.py`, around the same missing-annotation problem.

## The query workspace, as far as it got

Built: the query vocabulary (db/vocabulary.py), filter discovery
(db/discovery.py), answer analysis (db/analysis.py), the filter drawer,
Gallery/Table/Analyze, the compare tray, endless browsing, reading
generation metadata out of video containers, Any/All multi-select on
every dimension where both readings mean something, value lists on
`folder`/`album`/`people.person`/`place.id`, a door to the long tail
(`param.has`, `param.is`, `param.num`), a cut on the semantic ranking so
a search answers with a set, sorting by every column the table draws,
saved views as their own object, a synchronised glass over the compare
tray, and recurring prompt TERMS beside the exact counts -- as a
separate panel that says what it assumes, because a reading of a comma
convention does not get to borrow an identity's certainty.

This section has nothing left in it.

## Modalities the schema allows and nothing produces

Each of these has a slot already cut for it. The slot being empty is
not a design decision anybody made; it is work nobody did.

- **OCR.** `db/schema.sql:1469` permits `said.kind = 'ocr'` and
  `sg_web/media_view.py:475` types it. Nothing writes one. For an
  application whose thesis is "search what you have", text sitting
  inside a screenshot, a receipt or a document is a whole modality
  missing. Immich searches it as a first-class field.

- **Auto-tagging: a model that proposes a keyword.**

  The human half shipped: `tag` and `file_tag` are authored tables
  (schema v42), `f=tag:eq:<word>` is a gallery dimension with AND and
  OR, and the media inspector types one in. What is still missing is the
  other reading of the same word.

  `derived_annotation.kind = 'tag'` is permitted and never written, and
  that is the MODEL's tag -- it requires a `model_id`, a `model_version`
  and a `source_sha256`, and `derived.drop_all` deletes it wholesale.
  Which is why the human keyword did not go there: an authored claim in
  the disposable namespace would be gone at the next rebuild with
  nothing recording that it had ever been said.

  So the work is a job that writes `derived_annotation` tags from a
  classifier, and a surface that offers them for a person to ACCEPT --
  at which point the acceptance writes a `file_tag` and becomes theirs.
  The same shape reverse geocoding needs below: evidence suggests, a
  person authors, and the two never share a table.

  Two smaller things the human half left open, both deliberate:

  - **A smart album cannot filter by keyword.** `collection_rule`'s
    stored format is versioned (`RULE_VERSION`), so a `tag` field there
    is a durable-format change with a migration of its own, not a line
    in the registry. Every other surface got the dimension for free.
  - **No keyword management page.** `authored.rename_tag` folds one word
    into another and is covered, but nothing calls it: renaming and
    merging are `/albums`-shaped work with no surface yet.

- **Metadata portability: OUT is done, IN and interop are not.**

  `GET /operations/export/authored.json` carries everything a person
  told this library about their own pictures -- names, ratings,
  favourites, places, keywords, albums with their nesting, and who is in
  what including who expressly is NOT. Keyed by `content_sha256` and never
  by a row id, which is the difference between an export and a dump: an
  id belongs to one database file, a hash names the same photograph in
  any library that holds it. Only pictures carrying something authored,
  because a library is mostly pictures nobody has said anything about.

  The opposite shape from the verdict export beside it, deliberately.
  That one is for SHARING and carries no name or path; this one is for
  CUSTODY, and withholding a person's own names from them would be the
  defect rather than the feature.

  What is left is the harder half and the one this entry was really
  about:

  - **Reading one back in.** A different problem with its own conflicts
    to resolve -- a hash that matches two files, a name already taken,
    an album that exists with different members.
  - **XMP sidecars.** The boring standard, and the only thing that lets
    ANOTHER application see what you decided here. LibrePhotos
    round-trips face regions and names as MWG-RS; digiKam edits
    EXIF/IPTC/XMP in place. JSON solves custody; only XMP solves
    interop.

- **Reverse geocoding, as a suggestion.** `db/places.py:8` already
  names "a future reverse-geocoding job (cached by geographic cell)".
  The current rule -- GPS never mints a human place, a person authors
  it -- is right and must survive. What is missing is the middle step:
  GPS evidence produces a derived SUGGESTION, a person accepts or
  corrects it, and the acceptance is the authored claim. "We have your
  coordinates but refuse to mention they are in Detroit" is not the
  same virtue as "we did not silently decide for you".

## Human workflows we have the data for and not the product

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

- **A verdict on a SIMILARITY still cannot be given.**
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
  2. **The similarity arm.** The annotation arm ships
     (`POST /i/{slug}/said/verdict`), the person arm is written by
     correcting a face, and the duplicate arm by saying two pictures are
     not one (`/dupes`, `db/authored.py reject_duplicate`). A verdict on
     a SIMILARITY -- "these two are not alike" against a semantic
     neighbour -- still has no surface, and shipping the last arm with
     nothing exercising it would be a contract nobody has tested.
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
  3. ~~**Exportable without the pictures.**~~ Shipped:
     `GET /operations/export/verdicts.json`, offered under the panel
     that adds the verdicts up, because an export nobody can find is
     nearly one that does not exist. A row is a producer identity, what
     kind of claim, the verdict, the bytes it was about and when.

     The opt-in landed one step narrower than this entry imagined, and
     the architecture is why. `SG413` requires a route's answer to
     describe a shape the browser can be typed against, so the KEY set
     is fixed and what `?include=note` decides is whether the note
     carries its VALUE or a null -- the content is the part that could
     hold anything, and it is the part withheld. Anything other than
     `note` is refused with a 400 naming what it will add, because a
     field quietly ignored hands somebody a file they believe holds
     something it does not.

     Under `/export/` because it is BYTES somebody saves rather than a
     page they land on, which is the distinction `/thumb/` and `/media/`
     already sit under; and inside the operations router because
     `SG402` says `sg_web/app.py` may not speak `db.verdicts` and
     `operations.py` is where verdicts already live.

     Most of the tests are about ABSENCE, checked against the whole
     serialised body rather than key by key: a column added to
     `feedback` next year would ride along in a `SELECT *`, and a test
     naming only today's keys would not notice.

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
  - ~~**Search.**~~ Shipped, and it was the more valuable half: typing
    "Sarah at the beach" now offers the split -- `person=sarah` plus
    `q="at the beach"` -- OFFERED and never applied, because rewriting a
    typed question silently is how somebody stops trusting the box.

    Which leaves only the first case above as the reason to build the
    prompt seam at all, and it is the weaker one: a better SENTENCE,
    from a model that knows two people are present and where. Rule 3
    still holds -- the sentence comes back nameless and the name is
    substituted at render time.

- **Duplicate CONSOLIDATION.** The review shipped: `/dupes` is a page,
  linked from the shell, showing every group, each copy's folder and the
  collections it is filed under, and whether the copies are byte-
  identical or merely alike -- with the post-state sentence ("3
  placements, 1 payload, every collection still complete") offered only
  where the bytes actually match. It is read-only and removes nothing.

  The operation itself is the part still missing, and the naive version
  is the one to avoid:
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

  ~~The missing half of the preview is the collection-completeness
  count.~~ Shipped: `pages.dupe_placements` names every collection a
  member of the group is filed under and how whole each one is right
  now, rendered ABOVE the verdict rather than below it -- the evidence
  before the claim it supports, because "every collection still
  complete" is a sentence about somebody's own albums.

  An album already short of its own members says so (`1/2`) instead of
  rounding up. A file whose bytes are gone keeps its placement --
  `missing_since`, never a delete -- so that number can be short before
  anybody consolidates anything, and the whole use of showing it is
  that somebody can watch whether the operation MOVES it.

  Two scalar subqueries, not a join from the group's members to the
  collection's: the join multiplies the two. Measured on this tree --
  four copies filed in one four-picture album counted 16 -- and that is
  the case one of the tests is.

  What is left is the operation itself, still not built, and the naive
  version is still the one to avoid.

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

## Three flakes in the suite: two closed, one unreproduced

- **`test_writes_stay_linear.py` fails near its own tolerance, and the
  suggested fix is the wrong one.** The ratio gate is `< 2.0` and the
  reported failures sat on it: three of four runs, on three DIFFERENT
  cases (`writing a parsed field` 2.0019, `rescanning an unchanged
  library` 2.1, `renaming a file` 2.4).

  Probed 2026-08-25, and it did not reproduce under either suspected
  cause. Against a 600,000-object retained heap, gc-on vs gc-off: every
  write case 0.93-1.06, no arm distinguishable. Against 16 of 16 cores
  contended by confirmed-running spinner processes: 0.89-1.11, 0/6 over
  the gate in every case. A full single-process `pytest tests/ -m slow`
  passed clean. So the cause is neither collector pressure nor CPU
  contention, and it is still unreproduced -- the earlier reading of
  "machine-load sensitivity" is not supported by a measurement.

  **Do not implement the repeated-timings-with-a-median fix.** It was
  measured and it is worse. `timeit`'s own documentation
  (`../refs/python/cpython/Doc/library/timeit.rst`, `Timer.repeat`)
  argues for `min` over mean because interference only ever ADDS time --
  but that reasoning is about ONE measurement, and this gate divides
  two. Taking the min of each side independently lets a lucky-fast SMALL
  swing the quotient: under contention min-of-3 widened the spread, to
  0.67 on `writing a parsed field` and 0.59 on `deleting a file`, where
  the single-shot ratios stayed inside 0.93-1.09.

  What is left is to reproduce it at all. Until something does, the
  honest options are a wider SMALL/LARGE gap so the ratio is not
  measuring noise, or timing the pair inside one measurement rather than
  as two. A gate that fails one run in three teaches people to re-run
  it, so leaving it alone is the one thing that is not an option.

- ~~**`test_browsing_does_not_stop_at_sixty.py` fails about one run in
  four.**~~ Fixed, and it was a PRODUCT bug rather than a test one: a
  reader already at the bottom could not restart a loader that had
  stopped. Every wake-up was an edge -- the observer fires when
  intersection CHANGES and a sentinel already on screen never changes;
  a scroll event needs the page to move and somebody at the bottom
  cannot move it. Two fixes, both measured as load-bearing by removing
  each and re-running: the pump re-arms when it finishes with the end
  still in reach, and a scroll asks where the window IS rather than
  waiting for an edge. Eight runs green, from one-in-three failing.

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

The timeline joined it too, and it was the densest: session strips,
scrubber segments, month and day cells, frames and bins.

Two surfaces were missed by all of that and found by a check that reads
every template at once rather than one page at a time -- the artifacts
shelf and the person drawer both went on spelling `/thumb/<slug>` into
their own markup long after "the artifact pages joined it" was written
down here. That check now fails the build, which is the only reason to
believe the list is complete this time.

Also not done:

- **Nothing is served by anything but Litestar.** Measured 2026-08-25,
  which this entry asked for before assuming it matters. It does, and
  the mechanism it proposed is the wrong one.

  Driving the ASGI app directly (no httpx, no server), best of three
  runs of 300:

      /thumbs/<sha>.webp   File, no database    1480 us
      /settings            JSON, one database   5033 us
      reading the file, nothing else              80 us

  So an asset costs about **1.5 ms of application CPU** and a 60-cell
  page about 89 ms of it -- on requests that touch no database at all.
  And it is FIXED: 810 bytes and 101 KB both cost ~1400 us above the
  file read, so it is dispatch, not streaming.

  Read honestly: that is server CPU, not what a person waits. A browser
  fetches sixty cells in parallel, and a real server adds its own cost
  this in-process measurement does not have. But it is 89 ms per page
  view that a cache in front would not spend at all.

  **`create_static_files_router` over the thumbs directory would be a
  regression, not the fix.** A miss on that path RENDERS
  (`sg_web/app.py` asset_bytes), which is the only reason a fresh
  library gets a slow grid instead of broken pictures -- a static
  router 404s. The viable shape is a front server that tries the file
  and FALLS BACK to the application on a miss (Caddy/nginx `try_files`),
  which keeps the render. The docs also warn that the router's
  directory is relative to the working directory, and the thumbs cache
  is an absolute path under the run home.
- **No sprite sheet or atlas, deliberately.** The measurement says the
  static fan-out is no longer material: sixty requests that touch no
  database and cache for a year are not the problem nine
  database-backed ones were.
