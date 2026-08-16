# smart-comfyui-gallery task runner

set positional-arguments
set windows-shell := ["bash", "-cu"]

# venv interpreter path differs by OS: Scripts/ on Windows, bin/ elsewhere
python := if os_family() == 'windows' { './.venv/Scripts/python.exe' } else { './.venv/bin/python' }

# Full test suite in the dev venv
test:
    {{ python }} -m pytest tests/ -q

# Benchmarks through the production pipeline with live load context (bench.just)
mod bench

# AI/ML debug surfaces: hardware matrix, decode canaries, search probes,
# acceptance benchmarks (ai.just)
mod ai

# Which faiss the app selects at runtime: the vendored GPU build
# (vendor/faiss-gpu-win64, CUDA DLLs from the nvidia wheels) on
# Windows+NVIDIA, else the installed faiss-cpu. AI_DAM_FAISS_GPU=0
# forces the fallback.
faiss-verify:
    {{ python }} -c "from smartgallery_ai.faiss_runtime import import_faiss; f = import_faiss(); print(f.__file__); print('faiss GPUs:', f.get_num_gpus())"
