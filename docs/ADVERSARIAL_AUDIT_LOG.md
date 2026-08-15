# Adversarial Audit Log (WI-31)

Three adversarial rounds ran against this branch: a human external audit, a
2-skeptic Opus review of the auth/OmniQuery seams, and a 29-agent oracle
(2 attackers + 27 verifiers) against the AC6/AC7 closure. This log records
the **per-finding disposition of every oracle finding**, independently
re-verified against the final head — not just the confirmed subset. Verify
verdicts were advisory; each disposition below was checked directly in
code/tests by the implementer.

Legend — **FIXED**: code change + test/measurement; **PREVENTED**: made
structurally impossible; **DOCUMENTED**: real limitation, recorded with
measurements, accepted; **REFUTED**: claim contradicted by code/measurement.

| # | Finding (sev) | Disposition | Evidence at final head |
|---|---|---|---|
| 1 | `auto` critic ships with grounding gate silently disabled (critical) | **FIXED** | Fail-closed in `QwenVlCritic.__init__` AND the factory; `test_qwen_critic_requires_semantic_embedder`. |
| 2 | 0.20 threshold false-accepts 27% incl. parroted example (critical) | **FIXED** | Contrastive gate v2 (`check_grounding`: floor + margin ≥ 0.09 over baseline); calibration artifact `benchmarks/results/grounding_calibration.json` (FAR 3.1%/FRR 25%); vacuous+parroted classes in `test_real_grounding_gate_negative_cases`. |
| 3 | Gate never inspects findings; "0 fabrications" over ~11 unverified findings (high) | **FIXED** (mechanism) + **DOCUMENTED** (scope) | `verify_finding_region` crop check drops unverifiable localizable findings; every "image-grounded/0 fabrications" claim restated to description+region-level scope in `review.py` flag comment and `AI_MODELS.md`. |
| 4 | Grounding rejection → no scan-log → 200s infinite retry (high) | **FIXED** | Failed reviews log `result_count=-1`; retry only on mtime/model change. |
| 5 | AC6 measurement had no committed artifact (high) | **FIXED** | Opt-in chain regression `test_real_critic_to_mask_chain` (passing, 210 s) + calibration script/results committed. |
| 6 | Flag's cited authority section said the opposite (high) | **FIXED** | `AI_MODELS.md` "Runtime verification record" critic bullet rewritten; flag comment states precise scope. |
| 7 | Auto-resolution test asserted pre-flip policy for the wrong reason (high) | **FIXED** | `test_get_critic_backend_stub_explicit_only` comment now states the post-flip fail-closed rationale; the fail-closed invariant has its own test. |
| 8 | Assembly-time clamping defeats "reject, never clamp" (high) | **FIXED** | `_clamp` removed; out-of-range quality/confidence rejected by `validate_review_payload` (review errors, nothing stored). |
| 9/20 | `localizable` from region enum; global types become maskable (high) | **FIXED** | `_LOCALIZABLE_TYPES` gate: lighting/style/composition/prompt_mismatch can never be localizable; plus crop verification. |
| 10/17 | Region-box fallback = fabricated geometry laundered through MobileSAM (critical) | **FIXED** | Fallback removed entirely; failed localization → GLOBAL finding with region as text; `localizable` requires a valid model-emitted bbox + crop verification. |
| 11 | Gate-disabled reviews indistinguishable/never re-derived (medium) | **PREVENTED** | Gate-less critic construction is impossible (fail-closed); any historical rows from the vulnerable window are derived state (`python -m smartgallery_ai rebuild`). |
| 12/19 | 5 GB model constructed every poll cycle; /status loads in request thread (high) | **FIXED** | Worker `_backend()` memoizes per lifetime (None cached too); service probes backends once per process; `/index` never constructs the critic. |
| 13 | `prompt_alignment_score` saturates at cos 0.40; None without embedder (medium) | **DOCUMENTED** / **PREVENTED** | Standard CLIPScore w=2.5 scaling kept (it discriminated in practice: 7.6–8.6 aligned vs 3.0 mismatched — the "structurally unable to discriminate" claim is contradicted by measurement); None-without-embedder is impossible (embedder mandatory; None ⇔ no prompt). Saturation noted in `AI_MODELS.md`. |
| 14 | Grounding comparison fail-open on NaN (low) | **FIXED** | `not (x >= t)` comparisons in `check_grounding`. |
| 15 | Failed masks never retried; scan log marks file done (low) | **FIXED** | Masks are their own segmenter-keyed `ai_scan_log` unit; standalone worker stage retries; `test_worker_masks_generated_when_segmenter_arrives_late`. |
| 16 | POST /index destroys masks (critical) | **FIXED** | `/index` no longer runs the critic or `store_review`; force clears the review scan-log so the worker re-reviews asynchronously with mask regeneration. |
| 18 | Duplicate of #1 | **FIXED** | See #1. |
| 21 | IoU 0.998 is a best case (medium) | **DOCUMENTED** | `AI_MODELS.md` "Segmenter IoU scope": solid-boundary best-case; repo test asserts > 0.7. |
| 22 | SQL CHECK covers only bbox_x/mask_path (medium) | **FIXED** (new installs) + **DOCUMENTED** (old DBs) | CHECK widened to all geometry + points; pre-existing DBs keep the narrow CHECK until rebuild, with `validate_review_payload` enforcing the full invariant in code everywhere. |
| 23 | Mask endpoint containment too broad; forced PNG mimetype (medium) | **FIXED** / **REFUTED** (mimetype) | Containment tightened to `cache_dir/masks`; the writer only ever produces PNGs into that tree, so the fixed mimetype is correct by construction. |
| 24 | No worker→mask chain test (medium) | **FIXED** | See #5. |
| 25 | Zero-area / out-of-frame bboxes validate (low) | **FIXED** | `_validate_bbox` requires positive area and in-frame extent. |
| 26 | Duplicate of #6 | **FIXED** | See #6. |
| 27 | Mask PNGs leak on re-review and file deletion (low) | **FIXED** | Re-review: `store_review` unlinks superseded mask files. File deletion: worker `_sweep_orphaned_masks` removes mask dirs for vanished file ids each cycle (`test_worker_sweeps_orphaned_mask_dirs`). |

Prior rounds (for completeness): the auth/OmniQuery review produced 3
findings — non-ASCII admin-password 500/lockout (**FIXED** + test),
login-timing enumeration (**FIXED**, decoy verify + test), migration-gate
LIKE/GLOB divergence (**FIXED** as hardening). The human audits' findings
— completion-accounting error (**FIXED**: corrected then genuinely closed
by measurement), fail-open gate (**FIXED**, = #1), missing critic
regression test (**FIXED**, = #5), undeclared MobileSAM runtime dependency
(**FIXED**: declared in `requirements-ai.txt`).
