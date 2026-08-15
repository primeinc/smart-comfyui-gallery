# SmartGallery AI DAM + OmniQuery v2 — Architecture (WI-31)

This fork extends SmartGallery into a local-first generation-analysis DAM.
SmartGallery (the Flask monolith + SQLite) remains the UI and system of
record; everything added here is a **derived, rebuildable AI layer** plus a
**local typed query path**. No cloud inference, no telemetry, no mandatory
egress.

## Layout

```
smartgallery.py          # existing monolith (unchanged except: auth fix,
                         #   blueprint registration, omniquery v2 endpoint)
sg_auth.py               # one-way password hashing + legacy migration
smartgallery_ai/         # derived-AI layer (hashing, vectors, faces, review)
omniquery/               # NL -> typed AST -> validated -> compiled SELECT
tests/                   # pytest suite
probes/                  # runtime evidence scripts (egress, media read-only)
```

## Derived-AI layer (`smartgallery_ai`)

- **Authority**: all AI tables (`ai_file_hashes`, `ai_embeddings`,
  `ai_face_instances`, `ai_face_clusters`, `ai_reviews`,
  `ai_review_findings`, `ai_dam_state`) live in the main SQLite DB with
  `ON DELETE CASCADE` to `files(id)`, and are **derived state**: dropping
  them and re-indexing reproduces them. `ai_feedback` is the one
  human-authored table and is never dropped on rebuild.
- **Invalidation** (`invalidation.py`): a derived row is stale iff
  `source_mtime != files.mtime` or its `model_version`/`algo_version`/
  `rubric_version` differs from the active one. Deterministic, no
  heuristics; the worker re-queues stale rows.
- **Similarity is four distinct concepts, never merged**:
  1. *exact duplicate* — SHA-256 over bytes;
  2. *near-duplicate* — perceptual hashes (pHash/dHash, 64-bit, numpy DCT,
     no GPU) with Hamming distance;
  3. *semantic similarity* — joint image/text embedding space
     (`space='semantic'`);
  4. *visual similarity* — image-only self-supervised space
     (`space='visual'`).
  Face similarity is a fifth, per-face space (`space='face'`).
- **Vector index** (`vectors.py`): one implementation (numpy matrix +
  cosine top-k) hosts all named spaces; embeddings are stored in SQLite
  (authoritative derived record, model-versioned); the in-memory/on-disk
  index is a cache that can always be rebuilt, and `ephemeral_index=True`
  keeps it memory-only.
- **Model backends are optional plugins** (`embedders.py`, `faces.py`,
  `review.py`): a deterministic stub backend exists for tests; real
  backends lazy-import their runtime and self-report unavailable instead
  of raising. Model weights are provisioned separately into `.AImodels/`.
  Chosen models + licenses are recorded in `docs/AI_MODELS.md`; insightface
  pretrained weights are explicitly **not** used (non-commercial).
- **Faces**: every detection is its own `ai_face_instances` row (bbox,
  landmarks, det score, embedding, model version); clustering groups
  recurring generated identities by cosine threshold. Cluster labels are
  user nicknames, never real-world identity claims.
- **Review**: the critic emits JSON validated against a strict schema:
  quality score, prompt-alignment score, and typed findings with
  `type/severity/confidence/localizable`. Only `localizable=true` findings
  may carry bbox/point grounding and a segmentation mask (enforced by a
  SQL CHECK and by code); global findings are never given fake masks.
  Masks are PNGs in the derived cache — source media is never written.
- **Worker** (`worker.py`): background thread/process separate from request
  handling; consumes the indexing queue; the Flask UI never blocks on
  inference.

## OmniQuery v2 (`omniquery`)

The clipboard "paste SQL from an external LLM" flow is replaced as the
primary UX by a fully local path:

```
NL -> parser backend -> typed AST (ast.py, strict, versioned)
   -> validation (fields.py registry: schema, semantics, authorization,
      complexity caps — all outside any model)
   -> deterministic compiler (compiler.py): parameterized read-only SELECT
   -> execution on a mode=ro connection with a SQLite authorizer
```

- **No model emits SQL.** Parsers emit the AST (or a flat intent frame that
  expands deterministically into the AST). The compiler only accepts
  validator-approved ASTs and binds every literal as a parameter.
- **Backends**: `needle2` (primary; `cactus-needle` pip package,
  Apache-2.0, 14MB local engine — measured here to handle flat tool
  schemas well but not deeply recursive ones, hence the intent-frame
  design), `heuristic` (deterministic rule-based baseline, zero deps,
  always available), and at most one constrained fallback
  (Qwen2.5-Coder-0.5B-Instruct via grammar-constrained decoding) which
  must emit the same AST schema.
- **Routing**: parser confidence is only an input. Empirically the Needle2
  confidence head is not trustworthy a priori (a correct parse scored
  0.20 while degenerate empty parses scored 0.72–0.87 in initial probes),
  so thresholds come from `omniquery/benchmark/` measurements against the
  SmartGallery-specific corpus; the harness reports AST accuracy,
  invalid-call rate, false-confident rate, escalation rate, latency, and
  execution accuracy against a fixture DB.
- The legacy manual-SQL endpoint remains as a demoted "advanced" path with
  its existing authorizer; it is not the primary UX and no model feeds it.

## Status against WI-31 (honest accounting)

**19 of 19 acceptance criteria are met with runtime evidence** as of
2026-08-15. The two previously-unmet criteria were closed by re-deriving
the critic from first principles rather than by relabeling:

- **Generation review (AC6) — MET, by measurement.** The earlier
  monolithic SmolVLM2 critics measured 0/7 image-grounded (schema-valid
  fabrication) and remain opt-in-only. The shipped default is the
  **decomposed Qwen2.5-VL-7B critic** (`critic_qwen.py`): the model only
  answers small grammar-constrained questions (describe → assess →
  localize), deterministic code assembles the typed payload, prompt
  alignment is CLIPScore computed outside the VLM, and a deterministic
  CLIP grounding gate aborts any review whose description does not match
  the image — the exact measured fabrication failure mode, verified
  rejected on negative cases. Measured **4/4 schema-valid image-grounded
  reviews** on the calibration suite (planted defect detected and
  localized; mismatched-prompt noise correctly scored 0.0 quality / 3.0
  alignment). `critic_backend='auto'` resolves to it only when
  `_auto_critic_measurement_passed()` accepts the committed, hash-pinned
  calibration evidence (`benchmarks/results/grounding_calibration.json`);
  without that evidence in bounds, `auto` yields no critic.
- **Defect segmentation (AC7) — MET, by measurement.** Real MobileSAM
  backend (Apache-2.0) measured at IoU 0.998 on a planted defect; the
  worker segments every localizable finding of fresh reviews end to end
  (critic finding → box grounding → MobileSAM mask → API-served overlay),
  while the localizable-only gate and source immutability stay enforced.

## Security

- Passwords: Argon2id via `argon2-cffi` (`sg_auth.py`). No decrypt
  function, no plaintext-password API fields. Legacy Fernet ciphertexts are
  migrated in bulk at startup (decrypt once → hash → overwrite) and the
  Fernet key file is deleted; undecryptable rows get an unusable sentinel
  requiring an admin reset. See `docs/SECURITY_MIGRATION.md`.
- Runtime claims (no egress, source media untouched) are proven by
  `probes/`, not by static inspection.
