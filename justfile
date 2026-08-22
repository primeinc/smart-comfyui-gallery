# smart-comfyui-gallery task runner

set positional-arguments
set windows-shell := ["bash", "-cu"]

# venv interpreter path differs by OS: Scripts/ on Windows, bin/ elsewhere
python := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

# Full test suite in the dev venv
test:
    {{ python }} -m pytest tests/ -q

# Ruff over the whole tree, then this repository's own structural rules (sglint)
lint:
    {{ python }} -m ruff check .
    {{ python }} -m sglint

# Ruff format in report-only mode; never rewrites
fmt-check:
    {{ python }} -m ruff format --check .

# Pyright type check
types:
    {{ python }} -m pyright

# Repository hygiene: the git index, line endings, the generated
# requirements file, the documented test command -- not gallery
# behaviour. Two of them spawn git, so the spawn gate is opened here.
repo-check:
    RUN_TOOL_TESTS=1 {{ python }} -m pytest -q -p no:xdist -m repo -o addopts="" tests/

# Everything: lint, format, types, repo hygiene, tests
check: lint fmt-check types repo-check test

# The repo-wide structural gates, on their own and in seconds: sglint's
# rules (discovered scope: a package created tomorrow is swept the day it
# is born) and the tooling checks. `just test` runs the tests too.
[doc('Structural gates alone: subprocess safety, SQL hygiene, lazy imports, tracked files, runnability')]
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
