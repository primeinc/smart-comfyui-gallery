# smart-comfyui-gallery task runner

set positional-arguments
set windows-shell := ["bash", "-cu"]

# venv interpreter path differs by OS: Scripts/ on Windows, bin/ elsewhere
python := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

# Full test suite in the dev venv
test:
    {{ python }} -m pytest tests/ -q

# The structural checks, on their own and in a few seconds. Each fails when
# a class of bug comes back rather than when one instance of it is noticed:
# a route added without deciding who may call it, a prompt reaching a
# visitor, an id change that leaves its ratings behind, a documented setting
# nothing reads, a subprocess that can hang, a misspelt setting name.
# `just test` runs these too; this is for running them alone.
#
# --list shows only a comment's LAST line, which turns an explanation into a
# fragment, so the summary is stated explicitly.
[doc('Structural checks alone: route gating, prompt leaks, id changes, settings, timeouts')]
audit:
    {{ python }} -m pytest -q \
        tests/test_every_route_is_classified.py \
        tests/test_exhibition_leak_sweep.py \
        tests/test_file_id_changes_carry_their_data.py \
        tests/test_configuration_doc_matches_code.py \
        tests/test_media_tool_timeouts.py \
        tests/test_env_var_typos.py \
        tests/test_legacy_ai_search.py

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
