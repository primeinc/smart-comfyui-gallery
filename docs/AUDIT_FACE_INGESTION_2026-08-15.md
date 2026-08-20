# Face-ingestion audit — 2026-08-15

Adversarial-oracle audit of the face pipeline against an external "2026
face-ingestion architecture" narrative. Self-review declared: the audited
branch's face-pipeline commits (`a9f7ed7`..`2e4a4b3`) are co-authored by the
same model lineage that ran this audit.

**Verdict: the repo already implements the narrative's proposed pipeline
box-for-box; divergence is confined to detection policy — no multi-scale/tiled
pass, no measured detection recall, CPU-only single-pass inference. The
narrative's central `det_size=640` criticism does not apply to this repo, and
its "YuNet 2026 refresh" is a weight-identical re-export, not a model
improvement.**

## Scope

- Full read: `smartgallery_ai/faces.py`, `smartgallery_ai/embedders.py`,
  `docs/FACE_CLUSTERING.md`
- Targeted read: `smartgallery_ai/worker.py` (`_load_image`,
  `_process_faces`), `smartgallery_ai/vectors.py` (topk), `pyproject.toml`,
  `smartgallery_ai/__init__.py` (AIConfig), `smartgallery_ai/review.py`
  (backend registries)
- Upstream (refs clones, pulled current at audit time):
  `opencv/opencv_zoo@47534e2` (2026-05-28),
  `deepinsight/insightface@7fadd42` (2026-07-27, read via `git show`)
- Excluded: `tests/`, `benchmarks/faiss_graph_evidence.py`, `bench.just`,
  `docs/FAISS_GPU_WINDOWS.md` (timing/build claims uncontested)

## Alignment map

| Narrative prescription | Repo | Status |
|---|---|---|
| detect → 5-pt landmarks → align → embed | YuNet `row[4:14]` → `alignCrop` → SFace (`faces.py:201-204`) | ALIGNED |
| threshold graph → Chinese Whispers | CSR cosine graph, deterministic CW (`faces.py:404-474`) | ALIGNED |
| persistent identity labels | centroid match > 0.9 (`faces.py:477-515`) | ALIGNED |
| faiss = retrieval, not identity creation | CW clusters; faiss is topk (`vectors.py:184-196`) + one equivalence-contracted graph backend | ALIGNED |
| evaluate downstream yield, not WIDER AP | `face_min_px=24` calibrated on cluster-purity impact (`FACE_CLUSTERING.md:120-127`) | ALIGNED |
| commercially-clean deployed detector | YuNet MIT | ALIGNED |
| SCRFD as accuracy/recall reference | absent | DIVERGED |
| multi-scale / tiled detector policy | absent — single native-res pass | DIVERGED |
| detection-recall measurement | absent — all metrics condition on detected faces | DIVERGED |
| GPU/quantized detection for ingestion | absent — OpenCV CPU DNN, per-image | DIVERGED |
| embedding-quality final gate | size+confidence proxy only | PARTIAL |

## Findings

### 1. Missing invariant gate — detection recall unmeasured. HIGH

Purity and top-cluster-share (`FACE_CLUSTERING.md:77-92`) are computed over
detected faces only; no ground-truth recall harness exists. YuNet's training
band is ~10–300px (`opencv_zoo` model card); at native resolution a
high-res close-up puts the face far past 300px. Docs acknowledge the band
(`FACE_CLUSTERING.md:22-24`); nothing measures how many faces exit it. A
missed face is invisible to every downstream metric.

Fix: labeled-recall fixture in the evidence harness (known face counts per
size band), reported per band by a `just bench` recipe. **This gates
finding 2 — do it first.**

### 2. Detector model and detector policy are one thing. MEDIUM

`worker.py:91-92` feeds full-res pixels to a single native-res CPU pass
(`faces.py:187`). Two-sided cost: large faces exceed the 300px band (recall
loss) and detection runs over every megapixel of large images. Upstream
abandoned single-resolution defaults: InsightFace 1.0 (2026-05-23) made
`prepare()` a joint 128+640 pass with unified NMS.

Fix: policy seam above `OpenCVFaceBackend.detect` (native pass + downscaled
pass, NMS-deduped). Adopt only if finding 1's recall data shows it pays.

### 3. Unpinned `opencv-python` against a static-shape ONNX model. MEDIUM

`pyproject.toml:21` has no version constraint. The OpenCV 5.x ONNX Runtime
engine requires dynamic dims for variable input; `2026may` exists precisely as
a symbolic-dims re-export of the repo's 2023mar weights. A future resolve to
opencv-python 5.x can break per-image `setInputSize` in production.

Fix: pin `opencv-python<5`, or provision the weight-identical `2026may`
export and bump `model_version` when moving to 5.x.

### 4. No per-crop quality signal at the embedding gate. LOW

Every detection surviving `det_score>=0.5` + `min_face_px>=24` is embedded
unconditionally (`faces.py:194-203`). Bounded: the 24px gate was empirically
calibrated against the junk-blob failure mode. Revisit after finding 1.

## Clean at scope

`embedders.py`; `vectors.py` topk path; `faces.py` graph/CW/label code;
`FACE_CLUSTERING.md` — every upstream constraint it quotes verified against
refs clones; it quotes no WIDER marketing numbers.

## Narrative claim diff (verified against refs clones)

| Claim | Verdict |
|---|---|
| InsightFace 1.0, 2026-05-23, `prepare()` Auto = SCRFD 128+640 + unified NMS | holds (`python-package/README.md` changelog) |
| InsightFace: MIT code, non-commercial pretrained models, commercial licensing offered | holds (License section; `docs/commercial_evaluation.md:124-128`) |
| July 2026 50M-vector GPU search server | holds (`server/README.md:5`) |
| YuNet May 2026: dynamic-shape ONNX, CPU/CUDA/FP16, MIT | holds — **but `2026may` is a re-export of 2023mar weights, identical accuracy; the repo's deployed weights ARE the "2026 model"** |
| YuNet README metric inconsistency (0.834/0.824/0.708 vs 0.884/0.866/0.750) | holds (`README.md:3` vs `:17-21`) |
| RetinaFace multi-scale vs SCRFD single-scale not comparable | holds (`model_zoo/README.md:131,143`) |
| "single det_size=640 treats all image types alike" | misdirected — repo has no fixed det_size; detects at native resolution |
| EResFD 80.4 WIDER-Hard, 37.7ms/VGA CPU | unverified (no refs clone) |
| MediaPipe short/full-range models | unverified (no refs clone); moot — unused |

## Post-audit correction

Initial phrasing "no swappable detector" overstated. Correct statement: the
detector (YuNet, `faces.py:168`) and embedder (SFace, `faces.py:171`) are
separate model objects with a clean handoff, but there is one swap seam, not
two — a single `face_backend` config knob and a single `detect()` interface
returning embeddings attached. Detector-only or embedder-only swap-by-config
requires the interface split. Component-by-component swap inventory:
`MINIMUM-REQ.md`.
