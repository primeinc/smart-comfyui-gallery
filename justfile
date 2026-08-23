# smart-comfyui-gallery task runner

set positional-arguments
set windows-shell := ["bash", "-cu"]

# venv interpreter path differs by OS: Scripts/ on Windows, bin/ elsewhere
python := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

# The fast lane: every test not marked slow, spread over the cores
# (pytest-xdist, one module per worker: module-scoped stages assume the
# file's own order; 73s -> 20s on 16 cores). pytest.ini already carries -q;
# a second one would hide the pass/fail summary.
test:
    {{ python }} -m pytest tests/ -m "not slow" -n auto --dist loadfile

# The slow lane: the tests marked slow (real sample libraries, real
# browsers) -- a few seconds each, four at a time
test-slow:
    {{ python }} -m pytest tests/ -m slow -n 4 --dist loadfile

# Ruff over the whole tree, then this repository's own structural rules (sglint)
lint:
    {{ python }} -m ruff check .
    {{ python }} -m sglint

# Ruff format in report-only mode; never rewrites
fmt-check:
    {{ python }} -m ruff format --check .

# Pyright: cross-module type inference, the half ruff cannot do; part of the gate
types:
    {{ python }} -m pyright

# Repository hygiene: the git index, line endings, the requirements
# file, the test command, the evidence stamp -- sglint's SG8xx, which ask
# git; no test ever does
repo-check:
    {{ python }} -m sglint --repo

# The gate: lint, format, repo hygiene. No tests -- `just test` is its own step
check: lint fmt-check types repo-check

# The gate plus both test lanes
check-all: check test test-slow

# The repo-wide structural gates, on their own and in seconds: sglint's
# rules (discovered scope: a package created tomorrow is swept the day it
# is born) and the tooling checks. `just test` runs the tests too.
[doc('Structural gates alone: sglint code rules, sglint --repo hygiene, and the linter self-tests')]
audit: repo-check
    {{ python }} -m sglint
    {{ python }} -m pytest -q tests/test_sglint_has_teeth.py

# The application, served. Flags pass through: `just serve --port 9000`,
# `just serve --home D:/runs/two` -- everything else is a settings row
# changed in the running app.
[doc('Serve the gallery (python -m sg_web)')]
serve *ARGS:
    {{ python }} -m sg_web "$@"

# Benchmarks through the production pipeline (bench.just)
mod bench

# Which faiss the app selects at runtime: the vendored GPU build
# (vendor/faiss-gpu-win64, CUDA DLLs from the nvidia wheels) on
# Windows+NVIDIA, else the installed faiss-cpu. The faiss_gpu setting
# row forces the fallback.
[doc('Print which faiss build the app loads, and how many GPUs it sees')]
faiss-verify:
    {{ python }} -c "from vision.faiss_runtime import import_faiss; f = import_faiss(); print(f.__file__); print('faiss GPUs:', f.get_num_gpus())"
