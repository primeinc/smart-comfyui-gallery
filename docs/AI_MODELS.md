# AI Models, Licenses, and Parameters (WI-31)

Everything below runs **locally**. No cloud inference, no telemetry. Model
weights live in `.AImodels/` (env `AI_DAM_MODELS_DIR` /
`OMNIQUERY_FALLBACK_GGUF`) and arrive one of two ways — both driven by
`smartgallery_ai/provision.py`, which pins destinations to the exact paths
the backends load and verifies SHA-256 digests for single-file artifacts:

- **Auto (default):** the AI layer is ON by default (`ENABLE_AI_DAM=false`
  opts out) and the background worker makes itself runnable once,
  **asynchronously**, on startup (`AI_DAM_AUTO_PROVISION`, default
  `true`): it pip-installs any missing runtime packages into the current
  environment — torch **and torchvision** wheels chosen by hardware and
  always from the same index (mixed indexes break torchvision's compiled
  ops): CUDA-capable when an NVIDIA driver is present (`AI_DAM_DEVICE=cpu`
  forces CPU), CPU-index otherwise — then downloads any missing weights.
  A **CPU-build torch already installed on CUDA hardware is swapped for
  the CUDA wheels automatically** at the same point (static installers pin
  the CPU index because package resolution cannot see GPUs; the running
  app can). If torch was already imported when the swap ran, one restart
  finishes it — the console says so. Loaded backends log their device
  (`[AI] <model> on device cuda`), `/status` reports `devices`, and the
  panel shows "Compute: … on cuda" while indexing. The qwen-vl critic
  offloads all layers to GPU when the installed `llama-cpp-python` is a
  CUDA build (`AI_DAM_GPU_LAYERS` tunes partial offload; the default PyPI
  wheel is CPU-only and ignores the setting).
  **Multi-GPU**: the default device is the card with the most total VRAM
  (bare `cuda` would mean PCI enumeration order). `AI_DAM_DEVICE=cuda:1`
  pins everything to one card (the critic maps it to llama.cpp's
  `main_gpu`); per-backend pins `AI_DAM_SEMANTIC_DEVICE` /
  `AI_DAM_VISUAL_DEVICE` / `AI_DAM_SEGMENTER_DEVICE` spread the small
  models across cards. The critic — the only model big enough to be worth
  splitting — layer-splits across all visible GPUs natively (llama.cpp
  default); `AI_DAM_TENSOR_SPLIT=0.6,0.4` sets the proportions, and
  `CUDA_VISIBLE_DEVICES` remains the universal override. Cycles are never
  blocked; freshly landed runtimes/weights activate through the worker's
  backend re-probe without a restart, and torch backends place models on
  the best available device (CUDA > MPS > CPU). On an egress-denied host
  the attempt fails fast and the layer degrades exactly as if nothing had
  been provisioned.
- **Explicit:** `python -m smartgallery_ai provision [--list] [groups]`
  (groups: `faces semantic visual segmenter critic all`) — for
  pre-provisioning, air-gapped staging, or `AI_DAM_AUTO_PROVISION=false`
  strict no-egress deployments.

Backends and request handlers never download; inference is always local.
Licenses were verified against the primary sources on 2026-08-14/15
(links inline).

## OmniQuery search

The search field is an LLM. Two answerers, fused per query in the
endpoint (`smartgallery.py omniquery_nlq`), and **every query answers**:

- **`nlq`** (`omniquery/parsers/nlq.py`, dependency-free): deterministic
  rules consume recognized structure; every leftover term becomes a
  full-text condition on the universal `text` field (filename, path,
  workflow prompt, AI caption, generation prompt, model, LoRA names in
  one OR). It answers exactly when it consumes the whole query, and it is
  the only path live typing ever touches.
- **`SqlSearch`** (`omniquery/parsers/nl2sql.py`): the nl2sql model doing
  its documented text2sql job — the LIVE schema from `sqlite_master`
  (with data-driven value hints) plus the question, SQL out, prompt per
  the distil model card at temperature 0. It runs an agentic loop:
  generate, execute, READ the outcome — execution errors go back for
  repair, zero rows offer broaden-or-confirm (identical SQL asserts
  emptiness), rows are accepted. It answers queries carrying free
  language; any failure falls back to the nlq result.

Model SQL is data, never trusted code: every statement executes through
`omniquery/sqlexec.py`, the single sandboxed gate (SELECT-prefix check,
read-only URI connection, C-engine authorizer permitting only
SELECT/READ/FUNCTION) shared with the manual Advanced endpoint. Answers
classify into result cards — tiles / stat / spotlight / empty — one
vocabulary for both answerers; responses never carry SQL or any IR.

| Role | Exact identifier | License | Size | Measured |
|---|---|---|---|---|
| Rules answerer | `nlq` (in-repo) | — | 0 | 98-entry corpus, 2026-08-16: **100% execution match**, parse p50 0.157 ms; live endpoint round trip p50 1.37 ms / p95 2.26 ms. |
| **Fusion (the shipped path)** | endpoint policy over both | — | — | 98-entry corpus, GPU, 2026-08-16, reproducible via `just ai bench-fusion`: **94.9% execution match** (rules exact on the 81 fully-consumed queries; model correct on 11 of 17 free-language queries, several of which are vague/adversarial entries with debatable ground truth; model hard-failures fall back to the rules answer). |
| nl2sql answerer | `distil-labs/distil-qwen3-4b-text2sql-gguf-4bit`, `model.gguf` (provision group `omniquery`) | Apache-2.0 | 2.5 GB | Best of five GGUF candidates screened 2026-08-16 (43.4% vs Qwen3-1.7B 40.8%, 0.5B-class ≤21%, SS-350M 1.3%). Loop latency ~0.4–3 s/query on GPU. Decode canary at load: garbage logits or sampler crashes reload CPU-only, loudly — also catches silently corrupted GGUF files. |

Superseded (removed 2026-08-16): the needle2/cactus-needle intent parser,
the four-threshold router, and the NL→JSON-AST middle layer for the
model path — the router contract ("Couldn't confidently parse") failed
the search box's job on bare terms like "girlnextdoor", and the AST
grammar throttled a text2sql-tuned model into a foreign output format.
The typed AST/validation/compiler pipeline remains as the rules
answerer's execution path and the API's typed query surface.

## Similarity / faces / review

| Role | Exact identifier | License | Dim | Notes |
|---|---|---|---|---|
| Exact duplicate | SHA-256 (stdlib `hashlib`) | — | — | No model. |
| Near-duplicate | In-repo pHash64 (32×32 numpy DCT-II, top-left 8×8, DC-excluded median) + dHash64 | — | 64-bit | No GPU, no external deps (the `ImageHash` package is deliberately **not** used — avoids scipy/PyWavelets). Hamming threshold default **≤ 8** (`AI_DAM_NEAR_DUP_DISTANCE`). |
| Face pipeline (**default via `auto`**) | insightface `FaceAnalysis(name='antelopev2')`: SCRFD-10GF detection + glintr100 embedding + genderage/landmark heads, official `antelopev2.zip` release pack | **Non-commercial research** ([insightface README](https://github.com/deepinsight/insightface/blob/master/python-package/README.md)) — disclosed in [INSTALL_AI.md](INSTALL_AI.md); `AI_DAM_FACE_BACKEND=opencv` opts out to the permissive stack | **512** (float32) | Default cluster threshold cosine ≥ **0.40** (`AI_DAM_FACE_CLUSTER_THRESHOLD`). Attributes (age, sex, pose, 106/68-pt landmarks) persist per face instance. Per-stage ORT providers measured: detection CPU (dynamic shapes), recognition CUDA (4.4×). `model_version=scrfd10g+glintr100-v1`. |
| Face detection (opencv backend, **runtime-verified**) | OpenCV Zoo YuNet `face_detection_yunet_2023mar.onnx` | **MIT** ([opencv_zoo LICENSE](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)) | — | 232 KB; runs via `cv2.FaceDetectorYN`. Measured: real-photo detection score 0.93 with 5 landmarks in ~80 ms (CPU, 512×512). |
| Face embedding (opencv backend, **runtime-verified**) | ArcFace glintr100 via `cv2.dnn` (shared with the antelopev2 pack; `AI_DAM_FACE_EMBEDDER=arcface`, 512-d) or OpenCV Zoo SFace `face_recognition_sface_2021dec.onnx` (`=sface`, 128-d, **Apache-2.0** [opencv_zoo LICENSE](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE)) | see identifiers | 512 / 128 | SFace measured: same-identity cosine 0.987 across a brightness edit at threshold ≥ 0.55. A/B vs glintr100 in `benchmarks/results/face_embedder_ab.json` — glintr100 won and is the arcface default. |
| Detector compare | all installed pipelines above, side by side | — | — | `/galleryout/aidam` dashboard tool; per-pipeline detections, landmarks, and attributes on any picked file. |
| Semantic embedding (**runtime-verified**) | OpenCLIP (`mlfoundations/open_clip`, MIT) with `laion/CLIP-ViT-B-32-laion2B-s34b-b79k`, file `open_clip/ViT-B-32_laion2b_s34b_b79k.bin` | MIT (code + weights per model card) | 512 | Joint image/text space (`space='semantic'`). Measured on CPU: load 5.6s, ~40 ms/image, ~24 ms/text; text→image retrieval verified (correct caption ranks its image above unrelated caption and above noise). `auto` never substitutes stubs. |
| Visual embedding (**runtime-verified**) | `facebook/dinov2-small`, snapshot dir `dinov2-small/` | Apache-2.0 | 384 | Image-only self-supervised space (`space='visual'`). Measured on CPU: load 8s, ~59 ms/image; same-image edit cosine 0.988 vs unrelated −0.03. Kept strictly separate from the semantic space. |
| Review critic (**runtime-verified, default via `auto`**) | `QwenVlCritic`: `Qwen/Qwen2.5-VL-7B-Instruct` (Apache-2.0, verified on model card), official `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF` Q4_K_M (4.68 GB) or Q8_0 + mmproj, via `llama-cpp-python` `Qwen25VLChatHandler` | Apache-2.0 | — | **Decomposed architecture** (`critic_qwen.py`): describe → CLIP grounding gate → grammar-constrained assess → grammar-constrained prompt-element alignment (deterministic verbatim slices from `generation_params`, unbounded mismatch findings) → grammar-constrained localize → deterministic assembly; prompt-alignment score is CLIPScore on the proven OpenCLIP space (`min(10, 25·max(cos,0))`), computed outside the VLM. **Measured 2026-08-15 (CPU, ~200 s/review): 4/4 schema-valid reviews with description-level grounding** — clean image scored 8.0; the planted red-square defect (the one finding with ground truth) detected and localized; dark image still grounded; noise image with mismatched prompt correctly got quality 0.0 / alignment 3.0. **Gate v2 (post-oracle):** the v1 absolute-cosine gate (0.20) was shown by adversarial review to accept vacuous and example-parroting descriptions (27% FAR); v2 additionally requires a calibrated **contrastive margin ≥ 0.09** over a generic-baseline text (FAR 3.1% / FRR 25% on the committed calibration set, `probes/grounding_calibration.py` → `benchmarks/results/grounding_calibration.json`), and localizable findings must additionally pass **topical crop verification** (finding text beats baseline on its own bbox crop) or be dropped. Scope: these are layered deterministic filters over description- and region-level grounding — they verify that named content is visually present, not the truth of subjective quality judgments; ~11 of ~12 findings in the calibration run had no ground truth to check. Finding lists are de-anchored: the assess prompt asks for every concrete defect (schema cap 16, "may well be empty") instead of a numeric ask that begets exactly-N outputs. |
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
