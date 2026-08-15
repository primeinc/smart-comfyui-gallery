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
| Face detection (**runtime-verified**) | OpenCV Zoo YuNet `face_detection_yunet_2023mar.onnx` | **MIT** ([opencv_zoo LICENSE](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)) | — | 232 KB; runs via `cv2.FaceDetectorYN`. Measured: real-photo detection score 0.93 with 5 landmarks in ~80 ms (CPU, 512×512). |
| Face embedding (**runtime-verified**) | OpenCV Zoo SFace `face_recognition_sface_2021dec.onnx` | **Apache-2.0** ([opencv_zoo LICENSE](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE)) | **128** (float32, verified at runtime) | Runs via `cv2.FaceRecognizerSF` with alignCrop. Measured: same-identity cosine 0.987 across a brightness edit; the two instances clustered together at the default threshold cosine ≥ **0.55** (`AI_DAM_FACE_CLUSTER_THRESHOLD`) — calibrate on your corpus via the feedback loop. |
| Face stack (higher-accuracy alternative, not shipped) | `fal/AuraFace-v1` (SCRFD `scrfd_10g_bnkps.onnx` + `glintr100.onnx`, 512-d) | Apache-2.0 ([HF](https://huggingface.co/fal/AuraFace-v1)) | 512 | Drop-in candidate if SFace clustering quality is insufficient. |
| **Explicitly avoided** | insightface model zoo (`buffalo_l`, SCRFD/ArcFace pretrained weights) | **Non-commercial research only** ([insightface README](https://github.com/deepinsight/insightface/blob/master/python-package/README.md)) | — | WI-31 forbids silently shipping these; they are not referenced anywhere in the code. |
| Semantic embedding (**runtime-verified**) | OpenCLIP (`mlfoundations/open_clip`, MIT) with `laion/CLIP-ViT-B-32-laion2B-s34b-b79k`, file `open_clip/ViT-B-32_laion2b_s34b_b79k.bin` | MIT (code + weights per model card) | 512 | Joint image/text space (`space='semantic'`). Measured on CPU: load 5.6s, ~40 ms/image, ~24 ms/text; text→image retrieval verified (correct caption ranks its image above unrelated caption and above noise). `auto` never substitutes stubs. |
| Visual embedding (**runtime-verified**) | `facebook/dinov2-small`, snapshot dir `dinov2-small/` | Apache-2.0 | 384 | Image-only self-supervised space (`space='visual'`). Measured on CPU: load 8s, ~59 ms/image; same-image edit cosine 0.988 vs unrelated −0.03. Kept strictly separate from the semantic space. |
| Review critic (**runtime-verified, default via `auto`**) | `QwenVlCritic`: `Qwen/Qwen2.5-VL-7B-Instruct` (Apache-2.0, verified on model card), official `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF` Q4_K_M (4.68 GB) or Q8_0 + mmproj, via `llama-cpp-python` `Qwen25VLChatHandler` | Apache-2.0 | — | **Decomposed architecture** (`critic_qwen.py`): describe → CLIP grounding gate → grammar-constrained assess → grammar-constrained localize → deterministic assembly; prompt-alignment is CLIPScore on the proven OpenCLIP space (`min(10, 25·max(cos,0))`), computed outside the VLM. **Measured 2026-08-15 (CPU, ~200 s/review): 4/4 schema-valid image-grounded reviews** — clean image scored 8.0; planted red-square defect detected and localized to its quadrant; dark image still grounded; noise image with mismatched prompt correctly got quality 0.0 / alignment 3.0 (no fabricated content). Grounding gate verified on negative cases: fabricated-example description cos 0.130 → REJECT, unrelated 
−0.112 → REJECT vs grounded 0.291 → ACCEPT (threshold 0.20). Measured tendencies: emits up to the 3-finding cap with mostly low severities; some bboxes fall back to coarse region boxes. |
| Review critic (superseded attempts, opt-in only) | `SmolVlmCritic` for `HuggingFaceTB/SmolVLM2-2.2B-Instruct` / `SmolVLM2-500M-Video-Instruct` | Apache-2.0 | — | **Measured unfit: 0/7 image-grounded** (example parroting / truncation — the failure mode the grounding gate now catches). Reachable only via explicit `'smolvlm'`. |
| Defect segmentation (**runtime-verified**) | `MobileSamSegmenter`: `ChaoningZhang/MobileSAM` vit_t, weights `mobile_sam.pt` (40 MB) | Apache-2.0 | — | Box/point-prompted; **measured: IoU 0.998** vs ground truth on a planted defect with a loose box prompt, 3.6 s CPU. The worker segments every localizable finding of a fresh review; `generate_finding_mask` still forbids masks for global findings. |
| Segmentation (interface ready; candidates, none shipped) | `facebookresearch/segment-anything-2` (SAM 2); `ChaoningZhang/MobileSAM`; `yunyangx/EfficientSAM` | Apache-2.0 (code + weights) | — | Box/point-prompted masks for **localizable** findings only. |

## Supporting libraries

| Library | Version | License | Used for |
|---|---|---|---|
| `argon2-cffi` | 25.1.0 | MIT | Argon2id password hashing (`sg_auth.py`) |
| `llama-cpp-python` | 0.3.x | MIT | Grammar-constrained fallback parser |
| `cactus-needle` | 2.0.4 | Apache-2.0 | Needle2 engine |
| `sqlite-vec` | — | MIT/Apache-2.0 | **Evaluated, not adopted**: pre-v1 API churn; numpy brute-force cosine is sufficient and dependency-free at personal-gallery scale (~10⁵ vectors). Decision can be revisited behind the `VectorStore` interface. |

## Runtime verification record (2026-08-15, CPU-only sandbox)

Everything marked *runtime-verified* above was proven live, not statically:

- **Full pipeline**: with all real weights provisioned, `auto` backend
  resolution picked OpenCLIP + DINOv2 + YuNet/SFace; the background worker
  indexed a 3-image corpus (hashes, 6 embeddings across both spaces, faces)
  with 0 errors; `/galleryout/api/aidam/similar` returned the correct
  nearest neighbor per space (visual top: the edited copy at 0.988); the
  OmniQuery `similar_to_visual` resolver returned the same ranking.
- **Reproducible**: `RUN_REAL_BACKEND_TESTS=1 python -m pytest
  tests/test_real_backends.py` re-runs the DINOv2 / OpenCLIP / YuNet+SFace
  proofs whenever weights are provisioned (skipped otherwise).
- **Probes**: `probes/egress_probe.py` and `probes/media_readonly_probe.py`
  both PASS with the real-model stack installed.
- **Critic**: measured unfit at ≤2.2B (see table); the failure mode that
  matters is *schema-valid fabrication* (example parroting), which no
  validator can catch — hence the opt-in-only policy.

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
