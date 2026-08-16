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
| omniquery (opt-in) | Search palette nl2sql refinement | distil-qwen3-4b-text2sql GGUF | 2.5 GB |
| llama-cuda (auto on Blackwell) | Official llama.cpp CUDA binaries, every GPU arch | ggml-org b9976 win-cuda zips | 640 MB |

Total ~6.7 GB for the zero-step default set; `omniquery` is explicit
opt-in and `llama-cuda` joins automatically only on Windows machines with
a compute >= 12.0 GPU. Licenses, exact repos, and SHA-256 digests:
[AI_MODELS.md](AI_MODELS.md). One flag: the antelopev2 pack is
**non-commercial research** licensed (insightface); every other weight is
MIT/Apache-2.0. `AI_DAM_FACE_BACKEND=opencv` keeps face detection on the
permissively-licensed YuNet+SFace stack only.

## Explicit / air-gapped provisioning

```bash
python -m smartgallery_ai provision --list      # show groups and sizes
python -m smartgallery_ai provision all         # or any of: faces semantic visual segmenter omniquery critic llama-cuda
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
  v0.3.34 ship no sm_120 kernels, so on such GPUs provisioning
  automatically adds the `llama-cuda` group — the **official
  ggml-org/llama.cpp CUDA binaries** (b9976, the exact llama.cpp commit
  the wheel's bindings target, ~640 MB, covering every current GPU
  architecture). The runtime loads them via the binding's documented
  `LLAMA_CPP_LIB_PATH` override and registers their dynamic compute
  backends. No compiler, no source build; also installable explicitly
  with `python -m smartgallery_ai provision llama-cuda` (Windows CUDA
  machines only). Verified on this matrix: CPU, RTX 3070 Ti (sm_86), and
  RTX 5060 Ti (sm_120) all decode (`probes/hardware_matrix_probe.py`).
- **Model loads are canaried**: a GPU llama build that cannot actually
  decode (garbage logits, sampler crash) reloads CPU-only with a logged
  warning; a decode canary also catches silently corrupted GGUF files —
  if you see it fire on a freshly provisioned model, verify the file's
  sha256 and your disk health.
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

## Search palette (Ctrl/Cmd+P, Alt+P)

A search field over your living library: open it and the newest files are
already there as a masonry; type and it morphs live (sub-millisecond
rules path); pause and the AI answer swaps in. No SQL is ever shown.

Works out of the box with zero downloads via the deterministic rules
answerer. The AI answerer — the nl2sql model that reads your actual
database schema and writes read-only queries in an agentic
execute-and-refine loop — is one download:

```bash
python -m smartgallery_ai provision omniquery   # 2.5 GB, Apache-2.0
# or point at your own text2sql GGUF:
OMNIQUERY_NL2SQL_GGUF=/path/to/model.gguf
OMNIQUERY_FALLBACK_GPU_LAYERS=-1   # full GPU offload; 0 forces CPU decode
```

Model-generated SQL executes only through the same sandbox as the manual
Advanced endpoint (SELECT-only, read-only connection, engine-level
authorizer). Model loads are canaried: a GPU build that cannot actually
decode reloads CPU-only with a logged warning, and any model failure
falls back to the rules answer — search never breaks.

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
