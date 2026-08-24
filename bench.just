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
