# AI Models, Licenses, and Parameters (WI-31)

Everything below runs **locally**. No cloud inference, no telemetry. Model
weights are provisioned into `.AImodels/` (env `AI_DAM_MODELS_DIR` /
`OMNIQUERY_FALLBACK_GGUF`) as a separate step; the application never
downloads at runtime. Licenses were verified against the primary sources on
2026-08-14/15 (links inline).

## OmniQuery v2

| Role | Exact identifier | License | Size / RAM | Notes |
|---|---|---|---|---|
| Primary intent parser | `Cactus-Compute/needle2` via PyPI **`cactus-needle==2.0.4`** (import `needle`) | Apache-2.0 ([HF model card](https://huggingface.co/Cactus-Compute/needle2)) | 14 MB engine, ~30–35 MB peak RAM (measured) | 45M-param tool-calling model, byte-level grammar-constrained decoding, calibrated-confidence head. Engine binary caches to `~/.cache/cactus-needle/<ver>/` on first use; fully offline afterwards. |
| Fallback (single, optional) | `Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF`, file `qwen2.5-coder-0.5b-instruct-q4_k_m.gguf` | Apache-2.0 ([HF](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)) | 469 MB file | Run via **llama-cpp-python** (MIT). Output is GBNF-grammar-constrained to the exact typed AST JSON schema — it cannot emit SQL or free text. |

Measured Needle2 behavior on this corpus (see `omniquery/benchmark/`):
small flat tool schemas parse well; wide schemas cause hallucinated filler
values and token-budget exhaustion; multi-constraint queries can silently
drop constraints while the confidence head reports high confidence
(observed 0.73 on a constraint-dropping parse vs 0.20 on a correct one).
Consequently confidence is **only one routing input**: a deterministic
term-coverage guard (numbers, quoted strings, and keyword classes must be
reflected in the AST) gates acceptance, and thresholds in
`omniquery/parsers/routing_defaults.json` are set from benchmark
measurements, not from the model's self-report.

Full-corpus results (83 entries, `benchmarks/results/`):

| Backend | Execution match | False-confident | Latency p50 | Peak RSS |
|---|---|---|---|---|
| heuristic (deterministic) | **67.1%** | 24% accepting all parses; ~6–7% at coverage ≥ 0.7 | 0.13 ms | 21 MB |
| needle2 standalone | 21.1% | 0% at confidence ≥ 0.7; 33–75% below | 504 ms | 56 MB |
| fallback (Qwen 0.5B, grammar) | 0% | n/a (no confidence signal) | 1.8 s | ~870 MB |
| **router (tuned)** | 64.5% | **7.5%** | 0.24 ms (p95 5.4 s on escalation) | — |

The tuned router trades ~2.6 points of raw accuracy vs. the
accept-everything heuristic for a 3× lower false-confident rate and 100%
unsupported-precision: out-of-scope queries get an explicit "unsupported"
with per-backend reasons instead of silently wrong results. The fallback
model is retained but **disabled by default** — the measured evidence does
not justify invoking it (`OMNIQUERY_ENABLE_FALLBACK=true` re-enables).

## Similarity / faces / review

| Role | Exact identifier | License | Dim | Notes |
|---|---|---|---|---|
| Exact duplicate | SHA-256 (stdlib `hashlib`) | — | — | No model. |
| Near-duplicate | In-repo pHash64 (32×32 numpy DCT-II, top-left 8×8, DC-excluded median) + dHash64 | — | 64-bit | No GPU, no external deps (the `ImageHash` package is deliberately **not** used — avoids scipy/PyWavelets). Hamming threshold default **≤ 8** (`AI_DAM_NEAR_DUP_DISTANCE`). |
| Face detection | OpenCV Zoo YuNet `face_detection_yunet_2023mar.onnx` | **MIT** ([opencv_zoo LICENSE](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)) | — | 232 KB; runs via `cv2.FaceDetectorYN` (no extra runtime). |
| Face embedding | OpenCV Zoo SFace `face_recognition_sface_2021dec.onnx` | **Apache-2.0** ([opencv_zoo LICENSE](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE)) | **128** (float32, verified at runtime on this machine) | Runs via `cv2.FaceRecognizerSF` with alignCrop. Cluster threshold default cosine ≥ **0.55** (`AI_DAM_FACE_CLUSTER_THRESHOLD`) — a starting point; calibrate on your corpus via the feedback loop. |
| Face stack (higher-accuracy alternative, not shipped) | `fal/AuraFace-v1` (SCRFD `scrfd_10g_bnkps.onnx` + `glintr100.onnx`, 512-d) | Apache-2.0 ([HF](https://huggingface.co/fal/AuraFace-v1)) | 512 | Drop-in candidate if SFace clustering quality is insufficient. |
| **Explicitly avoided** | insightface model zoo (`buffalo_l`, SCRFD/ArcFace pretrained weights) | **Non-commercial research only** ([insightface README](https://github.com/deepinsight/insightface/blob/master/python-package/README.md)) | — | WI-31 forbids silently shipping these; they are not referenced anywhere in the code. |
| Semantic embedding (provision to enable) | OpenCLIP (`mlfoundations/open_clip`, MIT) with `laion/CLIP-ViT-B-32-laion2B-s34b-b79k` | MIT (code + weights per model card) | 512 | Joint image/text space (`space='semantic'`). `auto` backend resolution never substitutes stubs. |
| Visual embedding (provision to enable) | `facebook/dinov2-small` | Apache-2.0 | 384 | Image-only self-supervised space (`space='visual'`). Kept strictly separate from the semantic space. |
| Review critic (interface ready; candidates, none shipped) | `HuggingFaceTB/SmolVLM2-2.2B-Instruct`; `vikhyatk/moondream2`; `Qwen/Qwen2.5-VL-7B-Instruct` | Apache-2.0 (all three) | — | Qwen2.5-VL GGUF vision currently needs the `HimariO/llama.cpp.qwen2.5vl` fork; mainline llama.cpp support pending. |
| Segmentation (interface ready; candidates, none shipped) | `facebookresearch/segment-anything-2` (SAM 2); `ChaoningZhang/MobileSAM`; `yunyangx/EfficientSAM` | Apache-2.0 (code + weights) | — | Box/point-prompted masks for **localizable** findings only. |

## Supporting libraries

| Library | Version | License | Used for |
|---|---|---|---|
| `argon2-cffi` | 25.1.0 | MIT | Argon2id password hashing (`sg_auth.py`) |
| `llama-cpp-python` | 0.3.x | MIT | Grammar-constrained fallback parser |
| `cactus-needle` | 2.0.4 | Apache-2.0 | Needle2 engine |
| `sqlite-vec` | — | MIT/Apache-2.0 | **Evaluated, not adopted**: pre-v1 API churn; numpy brute-force cosine is sufficient and dependency-free at personal-gallery scale (~10⁵ vectors). Decision can be revisited behind the `VectorStore` interface. |

## Index and invalidation parameters

- Vector index: one implementation (`smartgallery_ai/vectors.py`) hosting
  named spaces keyed `(space, model_id, model_version)`; L2-normalized
  float32 matrices; exact brute-force cosine top-k; per-space `.npz` disk
  cache with (row-count, max-computed-at) stamp; `AI_DAM_EPHEMERAL_INDEX=true`
  keeps indexes memory-only. SQLite (`ai_embeddings`) is the authoritative
  derived record; all indexes rebuild from it.
- Invalidation: a derived row is stale iff `source_mtime != files.mtime` or
  its version stamp differs from the active one
  (`HASH_ALGO_VERSION = sha256+phash64/dhash64-v1`,
  `RUBRIC_VERSION = review-rubric-v1`, embedder `model_version` strings).
  Deterministic; the worker re-queues stale rows automatically.
