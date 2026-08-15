# smart-comfyui-gallery task runner

set positional-arguments
set windows-shell := ["bash", "-cu"]

# Full test suite in the dev venv
test:
    ./.venv/Scripts/python.exe -m pytest tests/ -q

# Benchmarks through the production pipeline with live load context (bench.just)
mod bench

# Swap the venv's faiss-cpu wheel for the local Windows GPU faiss build
# (docs/FAISS_GPU_WINDOWS.md; package dir carries its DLLs). `uv sync`
# restores faiss-cpu; rerun this to restore the GPU build.
faiss-gpu-install src='C:/Users/will/dev/sg-lab/faiss-pkg/faiss':
    -uv pip uninstall faiss-cpu --python ./.venv/Scripts/python.exe
    rm -rf ./.venv/Lib/site-packages/faiss
    cp -r "$1" ./.venv/Lib/site-packages/faiss
    ./.venv/Scripts/python.exe -c "import faiss; print('faiss GPUs:', faiss.get_num_gpus())"
