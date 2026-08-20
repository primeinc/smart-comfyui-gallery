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
