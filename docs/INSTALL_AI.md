# AI Layer Install (cold start)

The base gallery install ([installation.md](installation.md)) does not cover
the AI layer. This page does: what a fresh machine needs before the Faces /
Similar / Review tabs and OmniQuery natural-language search work.

Everything runs locally. No cloud inference, no telemetry.

## Requirements

- Python **3.10+**.
- ~12 GB free disk for model weights (see the download list below).
- Optional NVIDIA GPU. The layer works CPU-only; a GPU makes reviews
  dramatically faster. Which torch you get is decided by `uv sync` and the
  index pinned in `pyproject.toml`, not by the app — see GPU notes below.
- Windows, Linux, macOS. (macOS: CPU/MPS only; the CUDA paths below are
  skipped.)

## Install

`uv sync` installs every runtime — the AI layer is core, and there are
no optional dependency groups: torch, torchvision, open_clip and
transformers land without a flag. Pip users install `requirements.txt`;
its header carries the one caveat (torch and torchvision must come from
the same index).

Nothing installs packages at runtime. The AI layer is ON by default and
provisions WEIGHTS only: on first startup the background worker downloads
the missing model files to `.AImodels/` (override: `AI_DAM_MODELS_DIR`).
Progress streams to the console and to the AI panel in the UI. Cycles are
never blocked; each capability lights up as its weights land. A backend
whose runtime package is missing stays unavailable until you install it —
`/status` and the per-file walkthrough both name which one.

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
| critic | Review tab (scores + typed findings) | Qwen3-VL-2B-Instruct | 4.4 GB |
| omniquery (opt-in) | Search palette nl2sql refinement | distil-qwen3-4b-text2sql | 8.1 GB |

Total ~5.6 GB for the zero-step default set; `omniquery` is explicit
opt-in. Licenses, exact repos, and SHA-256 digests:
[AI_MODELS.md](AI_MODELS.md). One flag: the antelopev2 pack is
**non-commercial research** licensed (insightface); every other weight is
MIT/Apache-2.0. `AI_DAM_FACE_BACKEND=opencv` keeps face detection on the
permissively-licensed YuNet+SFace stack only.

## Explicit / air-gapped provisioning

```bash
python -m smartgallery_ai provision --list      # show groups and sizes
python -m smartgallery_ai provision all         # or any of: faces semantic visual segmenter omniquery critic
```

Run it on a connected staging box, then ship the venv plus `.AImodels/`
to the air-gapped host with `AI_DAM_AUTO_PROVISION=false`.

Downloads are preflighted against free disk space (declared artifact
sizes plus 1 GB headroom for caches and the database). A fit that is
impossible fails up front with an actionable message — point
`AI_DAM_MODELS_DIR` at a roomier volume — instead of filling the drive
mid-download.

## GPU notes

- **torch**: uv installs it, from the CUDA index pinned in
  `pyproject.toml` (`[tool.uv.sources]`). `uv sync` is the whole story —
  nothing in the app installs or swaps packages. To move to a different
  CUDA line, change that pin, or install directly with
  `uv pip install --torch-backend=auto torch torchvision`, which picks the
  index from your driver.
- **Generative models (reviewer, nl2sql)**: every one loads through
  `smartgallery_ai/models.py` on torch — whatever torch can use, they use.
  Blackwell (RTX 50-series, sm_120) needs no special payload: a CUDA torch
  build covers it. `AI_DAM_DEVICE=cuda:N` pins a card;
  `AI_DAM_ATTN=kernels-community/flash-attn2` fetches a prebuilt
  FlashAttention kernel from the Hub, which is the only route to it on a
  machine with no compiler.
- **Decode is verified per device**: `probes/hardware_matrix_probe.py`
  exercises generation on CPU and on EACH GPU in a crash-contained
  subprocess and checks the output is not garbage — "it loaded" proves
  nothing.
- **faiss GPU**: Windows CUDA builds of faiss are vendored in
  `vendor/faiss-gpu-win64/` and used automatically; see
  [FAISS_GPU_WINDOWS.md](FAISS_GPU_WINDOWS.md). `AI_DAM_FAISS_GPU=0`
  forces CPU faiss, `AI_DAM_VECTOR_GPU=0` keeps vector top-k on CPU.
- **onnxruntime (faces)**: `onnxruntime-gpu` is installed by default on
  Linux and Windows and needs no CUDA wheels of its own — it finds the
  runtime in torch's lib directory. Per-stage placement is measured-informed: detection
  runs on CPU (dynamic shapes), recognition on CUDA (4.4x faster).
  `AI_DAM_ORT_PROVIDERS=cpu` forces everything to CPU.

## Search palette (Ctrl/Cmd+P, Alt+P)

A search field over your living library: open it and the newest files are
already there as a masonry; type and it morphs live (sub-millisecond
rules path); pause and the AI answer swaps in. No SQL is ever shown.

Works out of the box with zero downloads via the deterministic rules
answerer. The AI answerer — the nl2sql model that reads your actual
database schema and writes read-only queries in an agentic
execute-and-refine loop — is one download:

```bash
python -m smartgallery_ai provision omniquery   # 8.1 GB, Apache-2.0
# or point at your own text2sql checkpoint (a provisioned directory name
# or a Hugging Face repo id):
OMNIQUERY_NL2SQL_MODEL=my-org/my-text2sql
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
