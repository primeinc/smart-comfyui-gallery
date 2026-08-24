# Backlog

Work that is known, agreed and not done. One line per item: what, and why
it matters. Delete an entry when it ships — this file is the pending list,
not a history.

## Performance

- **Thumbnails are still serial.** 5.00 files/sec over the real library
  (`just bench thumbs-phases`), up from 1.76, but one worker renders one
  file at a time. `run_next()` loops over every pending item, so a 20,000
  file precache owns the only background worker for an hour. Decode,
  resize and encode are all per-file work with nothing shared, so this is
  the next win: claim a bounded batch, render a few concurrently, keep
  the database worker as the thing that commits. Benchmark 1/2/4/8 in
  flight rather than picking a number.

  Interactive work should also outrank precache — a browser waiting on
  `/thumb/x` is real work and a speculative queue is not — and two
  requests for the same missing key currently render it twice.

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

- **Batch size 1 on the GPU.** `run_next()` processes job items one at a
  time, so every accelerator-backed adapter encodes a single image per
  pass.

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
