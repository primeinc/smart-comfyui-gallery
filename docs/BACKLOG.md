# Backlog

Work that is known, agreed and not done. One line per item: what, and why
it matters. Delete an entry when it ships — this file is the pending list,
not a history.

## Performance

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

- **Benchmarks in the UI.** `just bench thumbs-phases` and the other
  benchmark recipes write JSON that only a terminal ever sees. The
  operations console is where a run's own facts already live; these
  belong there too, so throughput is something you can look at rather
  than something you have to run.

## Correctness

- **Filmstrip requests thumbnails that cannot exist.** `media_view.py`
  gives every neighbour `/thumb/{slug}` regardless of kind, and
  `_media_viewer.html` renders it unconditionally, but `/thumb` returns
  404 for audio and documents (`app.py:803`). A walk containing either
  produces first-party 404s. Needs kind-aware cells and a mixed-media
  browser fixture — the current filmstrip corpus is entirely PNGs, which
  is why it passes.

- **A new bundle can escape the committed-bundle gate.** `web::fresh`
  rebuilds and runs `git diff --quiet -- sg_web/static/build`, which
  catches a modified or deleted tracked bundle but not a newly generated
  untracked one. Add an entry point, forget to `git add` its output, and
  local tests pass while a clean checkout 404s.

- **`static_v` does not version the bundles.** `_static_version()` reads
  only the immediate files in `sg_web/static`, but templates stamp that
  value onto `/static/build/*.js`. Changing only TypeScript leaves the
  cache-buster unchanged.

- **`unbroken` is opt-in.** The fixture that fails a browser test on
  first-party HTTP failures and console errors has to be requested by
  name, so a test author who forgets it gets no such check. It should be
  the default for browser tests, with an explicit opt-out for tests that
  exercise failure deliberately.

- **Two build contracts.** The README documents `npm run build-web`,
  which does not clear stale output; `just web build` removes
  `sg_web/static/build` first because esbuild leaves obsolete files
  behind. One canonical command should own the clean.

- **`neighborhood` clamps a count it says it refuses.** The bound is
  documented and tested as a refusal, and implemented as
  `min(max(1, count), NEIGHBOURHOOD_MOST)`. Pick one and prove it at 0,
  negative, 1, the maximum and the maximum plus one.

- **No cold acceptance lane.** Nothing runs the documented bootstrap from
  a checkout with no `.venv`, no `node_modules` and no
  `sg_web/static/build`. `test_the_documented_launch_serves_a_whole_application.py`
  tests the launcher's refusals in-process and says so; it is not that
  lane.

## Layering

- **`viewer.ts` imports `isPlainClick` from `overlay.ts`.** The viewer
  core is meant to be container-independent and the overlay is the
  container. A generic click predicate belongs in a neutral module.

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
Gallery/Table/Analyze, the compare tray, endless browsing, and reading
generation metadata out of video containers. What is NOT built:

- **Within a dimension, multi-select is AND, never OR.** Repeated facets
  conjoin, which is right for "this checkpoint with that LoRA" and wrong
  for "image or video". Every dimension whose values are mutually
  exclusive (kind, sampler, checkpoint) therefore offers a list where
  choosing a second value gives zero. The UI does not pretend otherwise
  -- the count says 0 before you click -- but the honest fix is an `in`
  operator in db/facets.py taking a list, and an Any/All control on the
  dimensions where both readings are meaningful (people, LoRAs).

- **`context.origin` is not offered as a multi-select either**, for the
  same reason, which is why `has.generation` exists beside it.

- **No `folder`/`album`/`person` value lists.** Those dimensions are
  `slug`-carried scopes with no `discover` statement, so the drawer shows
  them with no options. They need a name-to-slug listing each; the
  vocabulary has the field (`discover`) and they have no value for it.

- **Advanced metadata has no door.** `file_param` holds every key any
  tool emitted and `param_key` registers them; the plan was a section
  that lets somebody ask `generation.foo >= 17` by key. Nothing is built:
  the vocabulary is deliberately curated and the long tail is
  unreachable from any surface.

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
  one A and one B and flipping between them is not built.

## Two flakes in the suite, characterised and not fixed

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

- **`test_the_bytes_are_served.py` fails only under xdist.** `rc=0` in
  isolation; under `-n 4` one case raises `asyncio.run() cannot be called
  from a running event loop` and the next errors in setup. The trigger is
  an unawaited-coroutine RuntimeWarning surfacing inside another test's
  setup — `filterwarnings = error` turns it into a failure wherever the
  garbage collector happens to run it. The test drives the ASGI app with
  `asyncio.run` inside a worker that already has a loop.
