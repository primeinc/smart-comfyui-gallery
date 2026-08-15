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
| Review critic (**runtime-verified, default via `auto`**) | `QwenVlCritic`: `Qwen/Qwen2.5-VL-7B-Instruct` (Apache-2.0, verified on model card), official `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF` Q4_K_M (4.68 GB) or Q8_0 + mmproj, via `llama-cpp-python` `Qwen25VLChatHandler` | Apache-2.0 | — | **Decomposed architecture** (`critic_qwen.py`): describe → CLIP grounding gate → grammar-constrained assess → grammar-constrained localize → deterministic assembly; prompt-alignment is CLIPScore on the proven OpenCLIP space (`min(10, 25·max(cos,0))`), computed outside the VLM. **Measured 2026-08-15 (CPU, ~200 s/review): 4/4 schema-valid reviews with description-level grounding** — clean image scored 8.0; the planted red-square defect (the one finding with ground truth) detected and localized; dark image still grounded; noise image with mismatched prompt correctly got quality 0.0 / alignment 3.0. **Gate v2 (post-oracle):** the v1 absolute-cosine gate (0.20) was shown by adversarial review to accept vacuous and example-parroting descriptions (27% FAR); v2 additionally requires a calibrated **contrastive margin ≥ 0.09** over a generic-baseline text (FAR 3.1% / FRR 25% on the committed calibration set, `probes/grounding_calibration.py` → `benchmarks/results/grounding_calibration.json`), and localizable findings must additionally pass **topical crop verification** (finding text beats baseline on its own bbox crop) or be dropped. Scope: these are layered deterministic filters over description- and region-level grounding — they verify that named content is visually present, not the truth of subjective quality judgments; ~11 of ~12 findings in the calibration run had no ground truth to check. Tendencies: emits up to the 3-finding cap, mostly low severities. |
| Review critic (superseded attempts, opt-in only) | `SmolVlmCritic` for `HuggingFaceTB/SmolVLM2-2.2B-Instruct` / `SmolVLM2-500M-Video-Instruct` | Apache-2.0 | — | **Measured unfit: 0/7 image-grounded** (example parroting / truncation — the failure mode the grounding gate now catches). Reachable only via explicit `'smolvlm'`. |
| Defect segmentation (**runtime-verified**) | `MobileSamSegmenter`: `ChaoningZhang/MobileSAM` vit_t, weights `mobile_sam.pt` (40 MB) | Apache-2.0 | — | Box/point-prompted; **measured: IoU 0.998** vs ground truth on a planted defect with a loose box prompt, 3.6 s CPU. The worker segments every localizable finding of a fresh review; `generate_finding_mask` still forbids masks for global findings. |
| Segmentation (interface ready; candidates, none shipped) | `facebookresearch/segment-anything-2` (SAM 2); `ChaoningZhang/MobileSAM`; `yunyangx/EfficientSAM` | Apache-2.0 (code + weights) | — | Box/point-prompted masks for **localizable** findings only. |

## Critic failure mechanism and alternatives (vendor-doc vetted)

The measured SmolVLM2 failure is the known sub-3B pathology of **visual
token neglect**: under few-shot structured-output prompting, small
language backbones over-attend the text tokens (including any example
payload) and bypass cross-attention to the visual embeddings — producing
schema-valid text that ignores the image. The shipped critic implements
the standard mitigations, which is why it measures 4/4 grounded:
grammar masks at sampling time instead of an in-context example payload
to parrot (llama.cpp GBNF via `response_format`), zero-shot structural
directives only, a describe-first stage with a deterministic CLIP
grounding gate, and decomposed atomic questions (the same VQA-
decomposition idea as TIFA/DSG applied to defect review).

Alternative critics evaluated against WI-31's licensing stop condition:

| Candidate | License (verified from source) | Verdict |
|---|---|---|
| `Qwen/Qwen2.5-VL-3B-Instruct` | **Qwen RESEARCH LICENSE — non-commercial only** (LICENSE file in repo; corrects an earlier research pass that wrongly reported all Qwen2.5-VL sizes Apache-2.0) | **Excluded** per the no-research-only-weights stop condition, despite its attractive ~2.5 GB Q4 footprint. |
| `Qwen/Qwen2.5-VL-7B-Instruct` | Apache-2.0 (model card) | **Shipped default** (measured 4/4 grounded). |
| `microsoft/Phi-3.5-vision-instruct` (+ official INT4 ONNX) | MIT | Viable candidate; different runtime surface (onnxruntime-genai); unmeasured here. |
| `OpenGVLab/InternVL2_5-2B/-4B` | Mixed (InternViT MIT; LM backbones carry InternLM/Qwen terms) | Requires per-model license verification before any adoption; unmeasured. |
| `THUDM/ImageReward`, `yuvalkirstain/PickScore_v1`, HPSv2 | Apache-2.0 / MIT-family scalar scorers | Attractive future *score enrichment* (human-preference-calibrated quality/alignment scalars alongside CLIPScore); not typed-finding critics, so they complement rather than replace the VLM stage. |

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
- **Critic**: the monolithic SmolVLM2 attempts measured unfit at ≤2.2B
  (0/7; schema-valid fabrication) and stay opt-in-only. The decomposed
  Qwen2.5-VL-7B critic superseded them and **measured 4/4 schema-valid
  image-grounded reviews** on the calibration suite (clean 8.0 / planted
  defect detected+localized / dark still grounded / mismatched-prompt
  noise 0.0 quality, 3.0 alignment), with the grounding gate verified to
  reject fabricated and unrelated descriptions on negative cases.
  `critic_backend='auto'` enablement is no longer a hand-set constant: at
  resolution time `smartgallery_ai.review._auto_critic_measurement_passed()`
  reads the **committed** calibration report
  `benchmarks/results/grounding_calibration.json` (written by
  `probes/grounding_calibration.py`, whose input population is pinned by a
  SHA-256 manifest over the committed `probes/data/calibration_portrait.png`
  plus deterministic generated images) and requires FAR ≤ 5% / FRR ≤ 30% at
  the shipped margin threshold; a missing or failing report disables
  `auto`. **Scope honesty:** the gate grounds the stage-1 *description*;
  per-finding
  grounding is only evidenced by the planted-defect case, the calibration
  suite is 4 images, and the 0.20 threshold was calibrated on a small
  description sample — it is one defense layer alongside decomposition
  and grammar constraints, not a proof of finding-level truth. Findings
  tend toward the 3-item cap with low severities.
- **Segmenter IoU scope:** the 0.998 IoU was measured on a solid-color
  rectangle over a smooth background — a best-case boundary. The repo
  test asserts IoU > 0.7; soft-boundary defects will score lower. The
  masks remain genuine model segmentations of model-claimed regions.

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
