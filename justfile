# smart-comfyui-gallery task runner

set positional-arguments
set windows-shell := ["bash", "-cu"]

# venv interpreter path differs by OS: Scripts/ on Windows, bin/ elsewhere
python := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

# The fast lane: every test not marked slow, spread over the cores
# (pytest-xdist, one module per worker: module-scoped stages assume the
# file's own order; 73s -> 20s on 16 cores). pytest.ini already carries -q;
# a second one would hide the pass/fail summary.
test: web::build
    {{ python }} -m pytest tests/ -m "not slow" -n auto --dist loadfile

# The slow lane: the tests marked slow (real sample libraries, real
# browsers) -- a few seconds each, four at a time
test-slow: web::build
    {{ python }} -m pytest tests/ -m slow -n 4 --dist loadfile

# Ruff over the Python, Biome over the browser source (biome.json: the
# first-party JS and CSS, never the vendored htmx), then this repository's
# own structural rules (sglint)
lint:
    {{ python }} -m ruff check .
    npm run --silent lint
    {{ python }} -m sglint

# Ruff and Biome format in report-only mode; never rewrites
fmt-check:
    {{ python }} -m ruff format --check .
    npm run --silent format-check

# Pyright over the Python and tsc over the browser source: the cross-module
# inference neither ruff nor esbuild can do. Part of the gate.
#
# Both halves always run. As a dependency with a body, the body was skipped
# whenever the dependency failed, so a red browser source meant no Python was
# type checked at all -- which during a migration is every single run.
[parallel]
types: web::types types-python

[private]
types-python:
    {{ python }} -m pyright

# Repository hygiene: the git index, line endings, the requirements
# file, the test command, the evidence stamp -- sglint's SG8xx, which ask
# git; no test ever does
repo-check:
    {{ python }} -m sglint --repo

# The gate: lint, format, types, repo hygiene, and the real database's
# version held against this build -- a schema bump with no step from the
# version in the home directory fails here, in under a second, before any
# commit. No tests -- `just test` is its own step
check: web::fresh api::check lint fmt-check types web::unit repo-check db-check

# The gate, both test lanes, and the real run walked
check-all: check test test-slow smoke

# The home directory's database against this build: every version between
# the file's and USER_VERSION must have a migration step; a newer file
# is refused. Reads the version only -- nothing is migrated here.
[doc('Refuse a build that cannot open ~/.smartgallery (no migration step from its version)')]
db-check:
    {{ python }} -c "from sg_web import home; from db import migrate; p = home.db_path(home.home(None)); print(p, 'pending', migrate.pending(p) if p.exists() else 'no database yet')"

# Every surface and five real pictures over the database in the home
# directory -- the check a lane over fresh databases cannot make.
# `just smoke --home D:/runs/two` for another run.
[doc('Walk the real run: every surface and real pictures, over ~/.smartgallery')]
smoke *ARGS: web::build
    {{ python }} -m sg_web.smoke "$@"

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
serve *ARGS: web::build
    {{ python }} -m sg_web "$@"

# Benchmarks through the production pipeline (bench.just)
mod bench

# The browser source: type check, bundle, watch (web.just)
mod web

# The Python/browser JSON contract: OpenAPI out, TypeScript in (api.just)
mod api

# Which faiss the app selects at runtime: the vendored GPU build
# (vendor/faiss-gpu-win64, CUDA DLLs from the nvidia wheels) on
# Windows+NVIDIA, else the installed faiss-cpu. The faiss_gpu setting
# row forces the fallback.
[doc('Print which faiss build the app loads, and how many GPUs it sees')]
faiss-verify:
    {{ python }} -c "from vision.faiss_runtime import import_faiss; f = import_faiss(); print(f.__file__); print('faiss GPUs:', f.get_num_gpus())"
