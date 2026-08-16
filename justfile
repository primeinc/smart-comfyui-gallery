# smart-comfyui-gallery task runner

set positional-arguments
set windows-shell := ["bash", "-cu"]

# Full test suite in the dev venv
test:
    ./.venv/Scripts/python.exe -m pytest tests/ -q

# Benchmarks through the production pipeline with live load context (bench.just)
mod bench

# Which faiss the app selects at runtime: the vendored GPU build
# (vendor/faiss-gpu-win64, CUDA DLLs from the nvidia wheels) on
# Windows+NVIDIA, else the installed faiss-cpu. AI_DAM_FAISS_GPU=0
# forces the fallback.
faiss-verify:
    ./.venv/Scripts/python.exe -c "from smartgallery_ai.faiss_runtime import import_faiss; f = import_faiss(); print(f.__file__); print('faiss GPUs:', f.get_num_gpus())"
