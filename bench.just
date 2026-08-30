# Repeatable benchmarks through the PRODUCTION pipeline: the db package,
# the vision backends, and real sample datasets.

set export
set positional-arguments
set script-interpreter := ['bash', '-euo', 'pipefail']

# venv interpreter path differs by OS: Scripts/ on Windows, bin/ elsewhere
python := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

# bare `just bench` lists instead of executing
default:
    @just --list bench

# both sample datasets through the production db pipeline: every clustering method judged
faces-validate *args:
    @"$python" benchmarks/face_pipeline_validation.py "$@"

# The app over the real sample datasets, photographed by Chromium:
# benchmarks/results/browser/report.html
browser-report *args:
    @"$python" benchmarks/browser_report.py "$@"

# The query workspace at a wide desktop, a laptop and a phone, over a real
# mixed library the application scanned itself:
# benchmarks/results/workspace/contact-sheet.html

# The half browser tests cannot see: a panel overlapping a heading, a column
# off the screen, a section that is a wall of empty rows.
workspace *args:
    @"$python" benchmarks/workspace_shots.py "$@"

# What one gallery page costs to DELIVER, per thumbnail: requests, SQLite
# connections opened, bytes, and whether a second view can cost nothing.
# `--label before` / `--label after` keep both in the same results file.
thumbs-delivery *args:
    @"$python" benchmarks/thumb_delivery.py "$@"

# Captioning throughput against batch size, precision and decode size, each
# variant's captions compared word for word with today's: a caption that got
# faster by becoming a different caption is not the same feature.
captions *args:
    @"$python" benchmarks/caption_batch.py "$@"

# Where thumbnail time goes, per phase and media kind, over a run's own
# library: benchmarks/results/thumbnail_phases.json
thumbs-phases *args:
    @"$python" benchmarks/thumbnail_phases.py "$@"

# OpenCLIP throughput against batch size, with vector equivalence against
# batch 1. `--under swarm-mixed` picks a population; source megapixels change
# the answer.
clip-batch *args:
    @"$python" benchmarks/openclip_batch.py "$@"

# What the host-to-device copy of a prepared CLIP batch costs, and which
# stream/memory configuration overlaps it. `--trace <path>` writes Chrome
# traces, so the overlap is read rather than inferred from a total.
clip-transfer *args:
    @"$python" benchmarks/openclip_transfer.py "$@"

# Whether batching changes what a SEARCH returns, over distinct pictures
# and real queries: benchmarks/results/openclip_retrieval.json. Vector
# drift is not the question; ranking is.
clip-retrieval *args:
    @"$python" benchmarks/openclip_retrieval.py "$@"

# What bounding OpenCLIP's input costs and buys, per population:
# benchmarks/results/openclip_raster[_<under>].json. Run the populations
# APART -- a bound does nothing to a PNG and changes a raw's rendering.
clip-raster *args:
    @"$python" benchmarks/openclip_raster.py "$@"

# What a JOB achieves end to end and where its time goes, read off the
# runner's own phase durations. `--job scan|embed|hash|context|annotate`,
# `--under <root>` to pick a population. Scans into a TEMPORARY database.
job-phases *args:
    @"$python" benchmarks/job_phases.py "$@"

# How long a scan holds SQLite's ONE write lane, split into the half that
# hashes (no lock) and the half that writes rows (locked). The locked figure
# says whether a large library crosses `busy_timeout`. TEMPORARY database.
scan-lock *args:
    @"$python" benchmarks/scan_lock.py "$@"

# What a page costs while anything else is writing. The ordered answer is
# cached per (question, data currency), and currency is PRAGMA data_version,
# which any commit bumps -- including to a table the answer never reads.

# Sweeps library sizes, synthesising rows in a TEMPORARY database.
answer-currency *args:
    @"$python" benchmarks/answer_currency.py "$@"
