# AI Layer Install (cold start)

The base gallery install ([installation.md](installation.md)) does not cover
the AI layer. This page does: what a fresh machine needs before the Faces /
Similar / Review tabs and OmniQuery natural-language search work.

Everything runs locally. No cloud inference, no telemetry.

## Requirements

- Python **3.10+** (release CUDA wheels for `llama-cpp-python` are
  `py3-none`, so no specific minor version is required).
- ~7 GB free disk for model weights (see the download list below).
- Optional NVIDIA GPU. The layer works CPU-only; a GPU makes the critic
  ~40x faster. Driver with CUDA 13.0+ gets the prebuilt CUDA
  `llama-cpp-python` wheel automatically.
- Windows, Linux, macOS. (macOS: CPU/MPS only; the CUDA paths below are
  skipped.)

## Zero-step install (default)

There is nothing to do. The AI layer is ON by default and provisions
itself: on first startup the background worker pip-installs the missing
runtime packages into the running environment and downloads the model
weights to `.AImodels/` (override: `AI_DAM_MODELS_DIR`). Progress streams
to the console and to the AI panel in the UI. Cycles are never blocked;
each capability lights up as its pieces land.

`ENABLE_AI_DAM=false` turns the whole layer off.
`AI_DAM_AUTO_PROVISION=false` keeps the layer on but forbids all egress
(for air-gapped hosts: pre-provision, below).

## What gets downloaded

| Group | Enables | Weights | Size |
|---|---|---|---|
| faces | Faces tab (detection, clustering, detector compare) | YuNet, SFace, insightface antelopev2 pack | ~444 MB |
| semantic | Similar tab (semantic space), critic grounding | OpenCLIP ViT-B-32 laion2b | 605 MB |
| visual | Similar tab (visual space) | DINOv2-small | 90 MB |
| segmenter | Defect masks for review findings | MobileSAM | 40 MB |
| critic | Review tab (scores + typed findings) | Qwen2.5-VL-7B Q4_K_M + mmproj | 5.5 GB |

Total ~6.7 GB. Licenses, exact repos, and SHA-256 digests:
[AI_MODELS.md](AI_MODELS.md). One flag: the antelopev2 pack is
**non-commercial research** licensed (insightface); every other weight is
MIT/Apache-2.0. `AI_DAM_FACE_BACKEND=opencv` keeps face detection on the
permissively-licensed YuNet+SFace stack only.

## Explicit / air-gapped provisioning

```bash
python -m smartgallery_ai provision --list      # show groups and sizes
python -m smartgallery_ai provision all         # or: faces semantic visual segmenter critic
```

Run it on a connected staging box, then ship the venv plus `.AImodels/`
to the air-gapped host with `AI_DAM_AUTO_PROVISION=false`.

## GPU notes

- **torch**: a CPU-build torch on CUDA hardware is swapped for CUDA wheels
  automatically, matched to the newest card's generation
  (`AI_DAM_CUDA_INDEX` overrides the index; `AI_DAM_DEVICE=cpu` opts out).
- **llama-cpp-python (the critic)**: the provisioner installs the official
  GitHub release wheel matching the driver's CUDA version (cu132 for
  driver CUDA >= 13.2, cu130 for >= 13.0). These wheels are `py3-none` and
  bundle their CUDA runtime — no toolkit install, no compiler, no source
  build. Machines whose driver predates CUDA 13.0 fall back to the
  `abetlen.github.io` cu124 index (needs CPython <= 3.12) plus the
  `nvidia-cuda-runtime-cu12`/`nvidia-cublas-cu12` DLL packages.
- **Blackwell (RTX 50-series, compute capability 12.x)**: release wheels
  v0.3.34 ship **no sm_120 kernels** — the critic crashes with a CUDA
  error if it offloads to such a card. On mixed rigs pin offloading to a
  pre-Blackwell card (`CUDA_VISIBLE_DEVICES=0`), or build
  `llama-cpp-python` from source with `CMAKE_CUDA_ARCHITECTURES=120`.
- **faiss GPU**: Windows CUDA builds of faiss are vendored in
  `vendor/faiss-gpu-win64/` and used automatically; see
  [FAISS_GPU_WINDOWS.md](FAISS_GPU_WINDOWS.md). `AI_DAM_FAISS_GPU=0`
  forces CPU faiss, `AI_DAM_VECTOR_GPU=0` keeps vector top-k on CPU.
- **onnxruntime (faces)**: on NVIDIA boxes the provisioner swaps in
  `onnxruntime-gpu`. Per-stage placement is measured-informed: detection
  runs on CPU (dynamic shapes), recognition on CUDA (4.4x faster).
  `AI_DAM_ORT_PROVIDERS=cpu` forces everything to CPU.
- **Vision flash attention**: `AI_DAM_VISION_FA` toggles llama.cpp flash
  attention for the critic. Leave it on (default): with FA off, large
  images can demand a multi-GB compute buffer and crash upstream
  llama.cpp (bug reproduced on master; fix pending upstream).

## OmniQuery fallback parser (optional, off by default)

Natural-language search works out of the box (deterministic heuristic +
needle2 router). The grammar-constrained GGUF fallback is opt-in:

```bash
OMNIQUERY_ENABLE_FALLBACK=true
OMNIQUERY_FALLBACK_GGUF=/path/to/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
OMNIQUERY_FALLBACK_GPU_LAYERS=-1   # full GPU offload when llama-cpp-python is a CUDA build
```

## Verify

- Console on boot: `[AI] <model> on device cuda` per loaded backend;
  `/galleryout/api/aidam/status` reports `devices` and per-group
  provisioning state.
- The AI dashboard (`/galleryout/aidam`) shows installed pipelines,
  cluster/dupe views, and the detector compare tool.
- With weights provisioned: `RUN_REAL_BACKEND_TESTS=1 python -m pytest
  tests/test_real_backends.py` re-proves the embedders and face stack
  live.

Full environment-knob reference (devices, thresholds, backends per
capability): [AI_MODELS.md](AI_MODELS.md) and
[AIDAM_ARCHITECTURE.md](AIDAM_ARCHITECTURE.md).
