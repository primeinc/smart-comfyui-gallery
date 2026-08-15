# smart-comfyui-gallery task runner

set positional-arguments
set windows-shell := ["bash", "-cu"]

# Full test suite in the dev venv
test:
    ./.venv/Scripts/python.exe -m pytest tests/ -q

# Benchmarks with idle-preflight and live load monitoring (bench.just)
mod bench
