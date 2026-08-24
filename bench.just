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

# Where thumbnail time goes, per phase and media kind, over a run's own
# library: benchmarks/results/thumbnail_phases.json
thumbs-phases *args:
    @"$python" benchmarks/thumbnail_phases.py "$@"

# OpenCLIP throughput against batch size, with vector equivalence against
# batch 1: benchmarks/results/openclip_batch.json. `--under swarm-mixed`
# picks a population; source megapixels change the answer.
clip-batch *args:
    @"$python" benchmarks/openclip_batch.py "$@"

# What the host-to-device copy of a prepared CLIP batch costs, and which
# stream/memory configuration overlaps it: benchmarks/results/openclip_transfer.json.
# `--trace <path>` also writes Chrome traces, so the overlap is read
# rather than inferred from a total.
clip-transfer *args:
    @"$python" benchmarks/openclip_transfer.py "$@"

# Whether batching changes what a SEARCH returns, over distinct pictures
# and real queries: benchmarks/results/openclip_retrieval.json. Vector
# drift is not the question; ranking is.
clip-retrieval *args:
    @"$python" benchmarks/openclip_retrieval.py "$@"

# What the embed job actually achieves end to end -- decode, database,
# ledger and all -- against the encoder's ceiling over the same files:
# benchmarks/results/embed_job.json. Scans into a TEMPORARY database.
embed-job *args:
    @"$python" benchmarks/embed_job.py "$@"
