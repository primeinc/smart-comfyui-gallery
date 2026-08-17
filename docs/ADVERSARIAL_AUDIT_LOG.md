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
| 1 | `auto` critic ships with grounding gate silently disabled (critical) | **FIXED** | Fail-closed in `VlmCritic.__init__` AND the factory; `test_qwen_critic_requires_semantic_embedder`. |
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

## Round: Codex bot PR review (11 findings, head 63841a5)

| # | Finding (severity) | Disposition | Resolution / evidence |
|---|---|---|---|
| C1 | `_backend` resolver exception deadlocks worker via `_note_error` re-acquiring the non-reentrant lock (P1) | **FIXED** | Resolution moved outside the lock (cache check → unlocked resolve → `setdefault`); `test_backend_resolver_exception_cached_none_no_deadlock` proves a raising resolver returns None, caches, and never hangs. |
| C2 | `_fetch_candidates` loads the whole backlog into memory (P1) | **FIXED** | Chunked scanning (500/batch) with merge by `(-mtime, id)`, bounded by `limit`. |
| C3 | Mask scan-log row recorded even when all mask attempts failed → never retried (P1) | **FIXED** | `_log_masks_if_complete` records completion only when zero localizable findings remain unmasked; `test_failed_mask_generation_is_retried_not_logged_complete` covers the segmenter-recovers path. |
| C4 | Faces indexed but never clustered automatically (P1) | **FIXED** | Worker triggers `cluster_faces` after any cycle that indexed faces; `test_worker_clusters_faces_after_indexing`. |
| C5 | AI read routes bypass gallery access rules (P1) | **FIXED** | `create_ai_blueprint(file_access_check=...)`: per-file 404 on duplicates/similar/faces/review/mask; cluster listings now management-guarded; host wires `is_file_accessible`; `test_file_access_check_scopes_per_file_routes`. |
| C6 | Vector cache tmp-file collisions between concurrent writers (P2) | **FIXED** | Per-writer tmp name (pid + thread ident) with cleanup on failure. |
| C7 | `topk` compares a stored row's vector against the *active* model version during migration (P2) | **FIXED** | `topk(..., model_version=)` pins the matrix; both service call sites pass the row's own version; `test_topk_pins_model_version_during_migration`. |
| C8 | Duration SQL only parses MM:SS; H:MM:SS silently misfilters (P2) | **FIXED** | `DURATION_SECONDS_EXPR` branches on colon count; `test_duration_seconds_expr_handles_hms_and_ms`. |
| C9 | Bare-date `between` adds fixed 86400s → wrong on DST days (P2) | **FIXED** | Upper bound is the constructed next local calendar midnight; `test_between_bare_dates_dst_transition_days` (23h/25h days). |
| C10 | Folder predicates never match Windows-separator paths (P2) | **FIXED** | Column and values normalized to `/` (`REPLACE(f.path,'\','/')`); `test_folder_predicates_match_windows_separators`. |
| C11 | Modal "Show results in gallery" link routes collection-origin queries into `collection_view`, dropping the session (P2) | **FIXED** | Link always targets the physical root view (`_root_`), which renders OmniQuery sessions. |

## Round: owner adversarial review (VERDICT: FAIL at 63841a5, 8 findings)

| # | Finding | Disposition | Resolution / evidence |
|---|---|---|---|
| O1 | Critic overwrites the grounded `description` with per-defect text; summary pairs rejected text with a margin computed for a different string | **FIXED** | Separate `finding_description`; summary always quotes the step-1 description; `test_qwen_critic_summary_survives_rejected_last_finding` (last finding crop-rejected). |
| O2 | Term-coverage guard collapses media-type/status classes to one boolean, so "images or videos" with a dropped disjunct scores full coverage | **FIXED** | Per-normalized-value coverage units against the AST's condition values (incl. `in` lists); negative tests for both corpus cases. |
| O3 | Worker deadlock (same as C1) | **FIXED** | See C1. |
| O4 | `auto` critic enablement justified by evidence absent from the repo (gitignored report, env-dependent population) | **FIXED** | Constant replaced by `_auto_critic_measurement_passed()`, which reads the now-committed `benchmarks/results/grounding_calibration.json` and requires FAR ≤ 5% / FRR ≤ 30% at the shipped threshold, fail-closed. The probe emits a SHA-256 input manifest; the portrait input is committed (`probes/data/calibration_portrait.png`, public-domain NASA photo). `test_auto_critic_gate_reads_calibration_report`. |
| O5 | Mask files unlinked before the DB replacement commits → rollback restores rows pointing at deleted PNGs | **FIXED** | Unlink deferred until after commit; `test_store_review_failed_replacement_preserves_old_review_and_mask` injects a mid-transaction constraint failure. |
| O6 | AI route authorization bypass (same as C5) | **FIXED** | See C5. |
| O7 | Mask retry contract (same as C3) | **FIXED** | See C3. |
| O8 | "today/yesterday/this week/this month" implemented as rolling windows; benchmark certifies the bug | **FIXED** | Heuristic resolves calendar vocabulary to local calendar boundaries from the injected clock (bare-date values; ISO Monday weeks); corpus entry now uses date placeholders the harness resolves from the same clock; boundary tests incl. month transition. |

Verdict preamble ("438 automated tests claim has no independent run
attached"): **CLOSED — owner decision (2026-08-15).** A GitHub Actions
workflow was committed and triggered on every push/PR, but every run on
this fork failed at startup with the account's billing lock (no runner
assigned, zero steps executed). The repository owner has ruled out
enabling GitHub Actions for this repository, so the workflow was removed
— it produced only a failing check pair per push and could never provide
the independent run. All test counts in this record are local runs;
independent verification, if ever wanted, needs a different CI provider.

## Round: owner adversarial re-review of the repair delta (63841a5..79074db, 6 findings)

| # | Finding | Disposition | Resolution / evidence |
|---|---|---|---|
| D1 | [P1] Authorization checked only for the anchor file: `/duplicates` and `/similar` serialize neighbor ids without applying the visibility policy, leaking hidden file ids/relationships through visible relatives | **FIXED** | Every returned id passes `file_access_check`; hidden neighbors are dropped, not backfilled. `test_file_access_check_filters_returned_neighbors` (visible anchor, hidden exact/near/vector neighbors). |
| D2 | [P2] Unavailable backends cached `None` for the worker lifetime, so late provisioning never activates the standalone mask stage despite `_process_masks()` claiming that path | **FIXED** | Successful instances cached for the lifetime; unavailable results re-probed after a bounded retry window (`_backend_retry_seconds`, 300 s). `test_unavailable_backend_reprobed_after_retry_window` exercises `_backend()` itself. |
| D3 | [P2] "this week"/"this month" emitted only a lower bound, admitting future-dated files | **FIXED** | Both emit bounded `between` bare-date ranges (`[monday, sunday]`, `[first, last]`); `test_calendar_upper_boundaries_exclude_files_just_past_the_period` runs the compiled SQL against rows one second either side of each upper boundary. |
| D4 | [P2] Automatic face clustering had no retry state: face scans commit before clustering, so one clustering failure was never retried | **FIXED** | Persistent pending marker in `ai_dam_state` written before the attempt, cleared on success; the next cycle retries with zero face candidates. `test_face_clustering_retried_after_failure` (fails once, retries, then stops re-running). |
| D5 | [P2] The auto-enable gate accepted any JSON with in-bounds numbers at the shipped threshold — not bound to the evidence's identity (backend, baseline, input population) | **FIXED** | `_auto_critic_measurement_passed` now also requires the report's backend to equal `OpenClipSemanticEmbedder`'s model_id/version, `baseline_text` to equal the shipped constant, the committed portrait entry to be present, and every file-backed manifest hash to match this checkout. `test_auto_critic_gate_binds_to_evidence_identity` covers each mismatch, including the previously-accepted minimal synthetic report. |
| D6 | Verification claim false: both `pytest` checks at `79074db` failed before starting (account billing lock), so the workflow provides no independent run | **WITHDRAWN by reviewer** (CI/billing ruled outside the review's scope) | The factual correction stands regardless: the independent-run caveat remains recorded as OPEN/externally-blocked (above), and the PR body labels all test counts as local runs until a CI execution completes. |
