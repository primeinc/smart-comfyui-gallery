# smart-comfyui-gallery task runner

set positional-arguments
set windows-shell := ["bash", "-cu"]

# venv interpreter path differs by OS: Scripts/ on Windows, bin/ elsewhere
python := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

# Full test suite in the dev venv
test:
    {{ python }} -m pytest tests/ -q

# Ruff lint over the whole tree
lint:
    {{ python }} -m ruff check .

# Ruff format in report-only mode; never rewrites
fmt-check:
    {{ python }} -m ruff format --check .

# Pyright type check
types:
    {{ python }} -m pyright

# Everything: lint, format, types, tests
check: lint fmt-check types test

# The repo-wide structural gates, on their own and in seconds. Discovered
# scope (tests/source_tree.py): a package created tomorrow is swept the day
# it is born. `just test` runs these too; this is for running them alone.
[doc('Structural gates alone: subprocess safety, SQL hygiene, lazy imports, tracked files, runnability')]
audit:
    {{ python }} -m pytest -q         tests/test_suite_is_runnable.py         tests/test_tracked_files.py         tests/test_line_endings_survive_the_clone.py         tests/test_requirements_sync.py         tests/test_programs_are_started_safely.py         tests/test_sql_is_built_from_structure_only.py         tests/test_the_heavy_layer_stays_lazy.py

# Benchmarks through the production pipeline with live load context (bench.just)
mod bench

# Hardware matrix, decode canaries, search probes, acceptance benchmarks.
# Only the last line of this reaches --list, so it carries the summary:
# AI/ML debug surfaces (ai.just)
mod ai

# Which faiss the app selects at runtime: the vendored GPU build
# (vendor/faiss-gpu-win64, CUDA DLLs from the nvidia wheels) on
# Windows+NVIDIA, else the installed faiss-cpu. AI_DAM_FAISS_GPU=0
# forces the fallback.
[doc('Print which faiss build the app loads, and how many GPUs it sees')]
faiss-verify:
    {{ python }} -c "from smartgallery_ai.faiss_runtime import import_faiss; f = import_faiss(); print(f.__file__); print('faiss GPUs:', f.get_num_gpus())"
