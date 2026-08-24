# Backlog

Work that is known, agreed and not done. One line per item: what, and why
it matters. Delete an entry when it ships — this file is the pending list,
not a history.

## Performance

- **Thumbnail pipeline.** Measured at 1.73 files/sec serial over the real
  library (`just bench thumbs-phases`). The costs, per phase, are in
  `benchmarks/results/thumbnail_phases.json`. Known waste: sources are
  upscaled to 1440 and encoded at that size even when the original is
  0.03 MP; the 512 thumb is resized from the original instead of from the
  1440 preview; RAW is fully demosaiced rather than using its embedded
  preview; JPEG is fully decoded because `load()` runs before anything can
  call `draft()`; orientation 1 still pays a full-resolution copy.

- **One raster serves every consumer.** `oriented.for_model()` returns a
  full-resolution frame to the thumbnailer, the perceptual hash, face
  detection, OpenCLIP, BLIP and Qwen. Their real input contracts differ by
  orders of magnitude — the perceptual hash reduces to 32x32, YuNet caps
  at 1600 — so most of that decode is discarded. A raster request should
  state the size it needs and let the source adapter find the cheapest
  correct way to produce it.

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
