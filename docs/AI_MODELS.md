# Model weights: what runs, where it lives, how it arrives

Every model **runtime** is a core dependency (`pyproject.toml`
`[project.dependencies]`; `uv sync` or `pip install -r requirements.txt`
installs all of it -- there are no optional dependency groups). Model
**weights** are data, not packages: they live under the run's models
directory and arrive only when a job that needs them runs. Nothing on a
serving path ever downloads -- `/search` resolves the exact local
artifact or refuses with the fix named.

## Where weights live

The `models_dir` setting (a database row, changeable while the
application runs) names the directory; empty means `<home>/models`.
Semantic weights use the Hugging Face cache layout inside it, so
`hf download --cache-dir <models_dir>` pre-provisions for air-gapped
machines. Face weights use insightface's own layout:
`<models_dir>/insightface/models/antelopev2/`.

## The models

| Purpose | Model | License | Arrives via |
|---|---|---|---|
| Semantic search (default space) | OpenCLIP `ViT-B-32` / `laion2b_s34b_b79k` | MIT (code); weights LAION | `POST /jobs/embed` |
| Semantic search (optional space) | `Qwen/Qwen3-VL-Embedding-2B` (2048-d, video-native) | Apache-2.0 | `POST /jobs/embed` with a `qwen:` entry in `semantic_model` |
| Face detection + embedding | insightface `antelopev2` pack: SCRFD-10GF + glintr100 (512-d) + genderage | **Non-commercial research** (insightface model zoo) | `POST /jobs/faces` |
| Face fallback (permissive stack) | YuNet 2023mar + SFace 2021dec via OpenCV | Apache-2.0 | same job, `face backend` choice |
| Perceptual identity | pHash64 + dHash64 (`imagehash`) | BSD | no weights |
| Similarity index | FAISS -- vendored CUDA build in `vendor/`, `faiss-cpu` wheel otherwise | MIT | no weights |

The `semantic_model` setting is a comma list of
`[provider:]<model>/<checkpoint>` entries; every entry is its own
immutable similarity space and rankings fuse by rank (db/retrieval.py).
Changing an entry never rewrites history: old spaces keep their
provenance, the embed job fills the new one fresh.

## Doctrine

- **Jobs provision, serving refuses.** The embed job may download; a GET
  never does. An unprovisioned model on the query path is a 400 naming
  `/jobs/embed`, structurally -- the serving code path only ever hands
  local file paths to the loaders.
- **Runtimes install, weights arrive.** Nothing installs packages at
  runtime; nothing bundles weights in the repository.
- **Device choice is a setting.** `faiss_gpu` (on/off) picks the index
  device at manager construction; `ort_providers` (`auto`/`cpu`/...)
  picks ONNX Runtime execution providers for the face stack; the
  semantic encoders use CUDA when torch sees it.
- **The insightface pack is research-licensed.** The default face
  backend's weights are non-commercial; the OpenCV stack is the
  permissive alternative and stays runnable.
