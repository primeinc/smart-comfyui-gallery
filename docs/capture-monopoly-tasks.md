# FROZEN MASTER TASK LIST — Capture Monopoly

Reconciled by team-lead from five Wave-1 artifacts (population 576 ln, architecture 968 ln,
proof 534 ln, proof↔architecture crosscheck+addendum 358 ln, adversary 1209 ln + 6 runnable
probes), cross-checked against the lead's R0 file:line evidence. Tree: design-system @ 66d11aa
+ session working set. Scope may not be phased; execution is. Nothing below disappears because
it lands in a later phase.

GOVERNING INVARIANT, VERBATIM: "If it is output by any producer, store it as whatever it
actually is, without semantic narrowing, and prove that using the stored result is the same as
doing it live, without cheating."

USER MANDATE: caching maintained or enhanced — an identity HIT skips producer execution
entirely; every replacement key is a strict superset of the key it replaces.

REVIEW LENS (applies to every task's acceptance): no check that cannot fail; no key that
destroys what it protects; no record produced but never read. Shape 1 has two flavors
needing different techniques: a check written wrong (found by reading what exists and asking
whether it can fail) and a chain that stops being specified one step early, which nobody
wrote at all (found only by walking each mechanism to its end and noticing where the
specification stops). Every R1/R2 reviewer walks mechanisms to their ends. Reviewer-
practice failures come in two kinds with different remedies: enumeration misses (never
looked — answered by deriving populations from code, G4's method constraint) and
generalization misses (looked, recorded correctly, under-classified — answered NOT by
prose but by G4's family column: every finding filed as a sub-defect carries the family
it was classified into, against a DECLARED-CLOSED family list (currently three:
unreachable member G6, empty population G4, inconsistent inputs G9); an
under-classification is then visible as a finding whose declared parent is its sibling,
and a fourth family arriving is an explicit recorded event, never a silent label reuse.
The prose-only version of this remedy was itself a check that cannot fail — caught by
the adversary applying the lens to this document). Third practice failure: trusting a
summary statistic over the thing it summarizes — A COUNT IS NOT AN ENUMERATION (two
independent reproductions in one session: `git status | wc -l` returned stale counts
while direct enumeration was correct, for both adversary and architecture; either trusted
number would have reported a successful cleanup as failed, and the same trust inverted is
how a green gate gets believed). Absence claims rest on classified searches with positive
controls or printed enumerations, never on counts.

Engineering states: NOT STARTED / IMPLEMENTING / IMPLEMENTED / PROVEN / CONTRADICTED / BLOCKED.
IMPLEMENTED ≠ PROVEN. A task is PROVEN when its standing attack exists, RUNS THROUGH A GATE
THAT CAN FAIL, and has been observed failing on the defect it guards (negative control).

---

## PHASE G — proof gates (PRECEDE or accompany R1; both reviewers + adversary concur)
Owner: proof. All nine (plus G8b) are small and mechanical; they are the difference between R1 being
provable and R1 being asserted.

- G1  Closure reads lane exits: `all lanes exited 0` as a closure condition naming the red
      ones; lanes.json present and identity-stamped. [E3; compat.just:16-25; closure.py]
- G2  Ledger stage cells each derive from a stage-specific recorded event (row count,
      re-read digest, thawed key set); a stage with no evidence source is BLOCKED.
      ACCEPTANCE: each stage cell traces to a distinct evidence field, proven by SIX
      per-stage negative controls (suppress the durable write → only durable_write red;
      suppress the read-back → only durable_read_back red; etc.). The no-two-cells-same-
      object assertion is a cheap tripwire underneath, NOT the proof — six equal-but-
      distinct cells from one boolean defeat it. [E1/B7; ledger.py:140-145; proof's
      shape-1-inside-the-fix catch]
- G3  Evidence identity covers db/schema.sql, *.sql, compat.just, justfile, tests/,
      pyproject.toml, uv.lock, metaparse/, conftest.py, compat/ root, ty/pyrefly configs.
      Keep identity.py:147's incompleteness self-check. [D3/B17]
- G4  THE EMPTY-POPULATION RULE, three-state conditions everywhere: no condition may
      report held on an empty population; empty is not-applicable; not-applicable is not
      green; every condition declares its non-empty predicate (weights: ≥1 weight row;
      population: ≥1 declared consumer; one-tree: artifacts present to compare). Closes as
      one defect: the closure.py:49-50 one-tree waiver [E4], the weight trio passing on
      weights==[] ("0 weights VERIFIED") [E9], and the bootstrap-vacuity class. Same pass
      unifies the state predicate closure and ledger read (closure whitelists three bad
      states, ledger blacklists ≠VERIFIED — E10, latent today because ledger's blacklist
      is a superset, live the moment the predicates are reconciled the wrong direction).
      Per-face negative controls. ACCEPTANCE additionally: an audit of ALL nine conditions
      against their input sources confirms no remaining condition reports held over an
      empty input — the found sites are two of a family; the audit covers the third nobody
      enumerated (flavor-2 shape 1). METHOD CONSTRAINT (proof, against itself): the audit
      is DERIVED FROM CODE — every field each condition reads, traced to its writer —
      never from a remembered list of conditions; two enumeration misses this session
      (C5's untraced lane, G8's unopened corpus persistence) prove a remembered list
      produces a worthless validated-empty. BOUNDARY RULE (proof, from its retracted G8
      verification): every field is traced to its writer AND to whatever validates the
      writer's output — trace-to-writer alone stops one screen above the guard, which is
      where the guarantee usually lives. A clean audit is recorded as a validated empty
      with its derivation shown. [proof + adversary, reconciled]
- G5  closure_attack drives `ledger.build()` on synthetic INPUTS, never hand-written
      ledger.json rows — the derivation is what gets attacked. [proof P4-5]
- G6  Ablation verdicts reach a gate: aggregate into case verdict or closure condition
      (`no ablation INCONCLUSIVE/CONTRADICTED` with declared shrinking allowance); delete
      the constant-true CONTRADICTED term (run.py:411) or make it read ablations; make
      unreachable Verdict members unrepresentable at CaseResult level. [E2; 497/1037
      INCONCLUSIVE ship green today]
- G7  The GREEN-with-red-attack demonstration (adversary attack_green.py: all nine
      conditions ok via ledger.build() itself with attack=1, selftest=1 in the same
      output) becomes the standing negative control for the whole G phase: after the other
      six land, re-running it must print RED. SCOPE HONESTY: it stamps digests and does
      not empty the weight set — strong for the green-with-red-lanes path it demonstrates,
      NOT a general vacuity detector; G4's empty-input audit covers what it cannot.
- G8  ONE envelope: compat/corpus/cache.py _store/_restore (:105-131) is a second
      producer-record codec — type-name strings, a four-entry builtin map with an
      array[()] fallback, no digest, no producer identity, no container. Its WRITE path
      self-checks (_same, :179-184) so same-codec round-trips are faithful; the hole is
      that the codec is ABSENT from CONTRIBUTORS (:35-39), so a codec edit invalidates
      nothing (namespace hash identical with a byte appended to cache.py), and the READ
      path (:148-165) verifies nothing — entries written under one codec are silently
      reinterpreted under the next (demonstrated: one-line _BUILTIN edit, same bytes,
      np.float32 → Python float, values comparing equal so equality cannot see it).
      Blast radius: our_face feeds gallery_storage, reactor_face_model,
      producer_derivations. Replace with facestore (or refuse-by-name); ADDITIONALLY the
      codec serving a kind must appear in CONTRIBUTORS for that kind, and the read path
      must verify. Attack: write an entry, edit the codec, re-read — must refuse or
      return bit-exact, never reinterpret. [population round 2, diagnosis corrected by
      adversary E12/E13, verified by population]
- G8b E13, a check written wrong: _parts is called at cache.py:171 OUTSIDE the try
      opening at :175; the except at :185 names exactly the exception _parts raises and
      sits where it cannot catch it — one nested structured field (insightface Face
      promotes nested mappings) turns every corpus load into a crash instead of a cache
      miss. Fix with G8. [adversary E13, verified by population]
- G9  Consumption-graph consistency: each generated artifact records the digests of the
      artifacts it CONSUMED, not only the tree digest; one-tree asserts the consumption
      graph agrees. (Demonstrated E11: build ledger.json against clean pins, regenerate
      provenance.json with a bad weight state — `just compat pins` alone after a full run
      does exactly this — closure prints nine-of-nine GREEN with the bad weight in the
      file it just read. Already live: shipped ledger.json embeds a lanes block naming a
      lane `answer` that no longer exists, while stamped current.) Third family of
      held-without-comparing: unreachable member (G6), empty population (G4), inconsistent
      inputs (G9). G4's audit checks input CONSISTENCY as well as non-emptiness. Second
      G7 scope limit: attack_green.py re-stamps digests — the very move E11 exploits.
      [adversary E11, RUN 2 reproduced]

## PHASE R1 — monopoly core (application). Owner: runtime (+ nonface for R1-N block)

### Schema (the canonical class — greenfield migration v47→v48, no legacy promotion)
- S1  `producer_invocation` / `producer_result` outside the derived_ lifecycle (NOT
      derived_-prefixed; drop_all cannot touch it), immutability + permanence triggers,
      envelope BLOB + envelope digest, captured_at. [B3/B23/F1]
- S2  Input-edge/DAG rows: composite producers preserve constituent invocations and their
      complete results (ladder rungs incl. empty answers, padded retry, per-face recognizer
      forwards). [B10/A4/A5/A6]
- S3  `producer_determinism` append-only + immutability trigger; determinism_id recorded on
      VERIFICATIONS, not results. [converged]
- S4  `producer_contradiction` with BOTH triggers + append-only
      `producer_contradiction_judgment` (re-judging is a recorded verdict, never a mutation);
      re-blessing check is a GATED closure condition. [A8 exchange]
- S5  `producer_variance` with same-runtime / changed-runtime discriminator; three-state
      justification (unjustified/justified/contradicted; unjustified ≠ pass; declared
      MIN_SAMPLES); bounded factor versioned append-only. [firehose fix + guardrails]
- S6  `producer_contradiction_waiver`: append-only, both triggers, waived_by + reason
      NOT NULL, composite key (contradiction_id, determinism_id) — a waiver names ONE
      contradiction under ONE declaration; a later loosening mints a new determinism_id the
      old waiver does not cover. Blanket waivers unrepresentable. Gate reporting
      distinguishes waivers accumulated individually from N rows sharing one author/
      timestamp/reason — "0 unwaived" must not hide "47 waived in one act". [closed by
      architecture, final round]
- S7  Projections (`derived_face_instance`, `derived_embedding`, `derived_annotation`) carry
      canonical_result_id FK (SQL-enforceable lineage); face table drops `native`.
- S8  drop_all refuses any table carrying canonical data, enforced by a DECLARED
      table_class drop_all asserts against — not a LIKE-pattern naming convention
      (standing attack: drop with a missing-file library must not destroy last copies).
      [F1; architecture round 5]
- S9  reproject check: because the FK proves parentage, not agreement, a standing check
      re-derives every projection from its canonical result and compares — zero producer
      cost. SCOPE (adversary): reproject proves AGREEMENT at check time, not DERIVATION —
      a write path still reading the live object passes wherever the two happen to agree,
      which is every row except the ones that matter. Lineage (FK) / agreement (S9) /
      derivation (C1's projections-built-from-thawed-record) are three properties; S9 is
      the detector, C1 is the fix, and S9 does NOT discharge C1. [architecture round 5 +
      adversary round 3]

### Capture door (faces) — adversary's best-ratio pair first
- C1  DERIVATION enforced by mechanism, not property-wording: the projection function's
      signature takes the thawed record and has NO parameter through which a live producer
      object could arrive (e.g. project_faces(conn, resolved, file_id, now)), plus the
      import-boundary check that no module outside the capture package imports a producer
      implementation. Lineage=FK (S7), agreement=reproject (S9, detector only),
      derivation=THIS — agreement looks like it proves derivation and does not (a
      live-reading write path passes reproject on every row where the two agree). The
      perturbed-live-vs-native attack stays as a divergence detector, not the provenance
      proof. [A3 — the 9.0/7.0 row; architecture round 6]
- C2  Score/size floors become query predicates over stored rows, not `continue` before the
      write; all six filter sites (faces.py:365,368,558,561,564; detect.py:93) move behind
      capture. Standing attack: graded ladder 0.0→0.99 + embedding-None + sub-pixel; stored
      producer records == emitted count. [A1/B1]
- C3  `derived_face_scan` records emitted AND promoted counts; a 3-emitted/1-promoted pass
      must read emitted=3. faces=0-suppression (runner.py:261-263) keyed to emitted. [A2]
- C4  Ladder: every rung's result captured (empty = a measurement); winning rung's det size
      in the envelope/provenance; a 448-hit must read 448 back. [A4/A6]
- C5  padded_recovery: DECIDED at checkpoint — capture it in the app path (both recoveries
      stored as the docstrings claim) or delete the docstring claims. Default proposal:
      capture both (UniPortrait-shaped consumers are served from the store). [A5]
- C6  det_thresh passed explicitly to prepare on every rung; configured AND effective
      thresholds in stored provenance; InsightFaceBackend(min_det_score=0.3) attack must
      see detector det_thresh 0.3. Fix the ctx_id=-1 per-rung provider reset. [A7/L-2]

### Envelope repairs (vision/facestore.py)
- V1  Aliasing: reference nodes or refuse-by-name (`Unpreservable` naming the path); attack:
      thawed["a"] is thawed["b"] or capture refused. [C3-adv]
- V2  Cycles: visited-set → `Unpreservable` with path, not RecursionError. [C4-adv]
- V3  Nested container identity: per-node container recording; tuple/dict SUBCLASSES refuse
      or preserve — never silently widen (facestore.py:151,154). [B12/C2-adv]
- V4  Root `container`: rebuild on thaw (structural-class adapter registry: insightface
      Face's __getattr__→None protocol + computed properties) or delete the docstring claim
      and the field's promise. Consumer-path attack: same consumer code against live and
      stored, including missing-key behavior. [C1-adv]
- V5  Non-contiguity: record honestly (values preserved; flags/strides normalized) — align
      docstring with behavior. [C5-adv]

### Identity & resolver
- I1  ResultIdentity preimage: source content digest, upstream result identities,
      preprocessing/adapter revision (incl. OpenCV 1600px downscale factor — recorded),
      producer implementation revision, weight FILE digests, invocation configuration
      (effective det_thresh, det sizes, floors chosen for execution), capture codec version,
      runtime/provider facts (execution provider, load-bearing package versions). [B4/B15/D1]
- I2  Resolver: HIT returns stored result, zero producer execution (CACHE MANDATE); MISS
      executes + commits; same-identity different-bytes = CONTRADICTED, never overwrite.
      [B24]
- I3  reverify(): sampled/scheduled forced recompute through store.put + HIT-path
      runtime_observed comparison enqueuing producer_reverify_candidate. The sampling
      rate is a STATED budget independent of library size (the number was never written;
      correctness and the caching headline both depend on it). Standing attack phrased to
      fail the dormant design. [A8; architecture round 5]
- I4  Face freshness (runner.py:261-263) gains model + weight-digest terms; weights-bytes
      swap attack must re-queue. [D1]
- I5  _Ahead/_Said memos keyed with the sha they were computed from; take() asserts match.
      [D2]
- I6  Determinism classes: fidelity bit-exact forever (thaw takes NO tolerance — signature-
      enforced; no helper with defaulted-exact tolerance); reproducibility per-contract,
      declared+digested+attacked, never defaulted. PREREQUISITE MEASUREMENT: re-run the
      openclip batch-width variance empirically (the 2.2e-03 is a docstring nobody re-ran —
      adversary ruling); until measured, openclip's class is `unjustified`, not assumed.

### Transactions & batches
- T1  THE RULE, verbatim (architecture, post-adversary): COMMIT AFTER EACH PRODUCER'S
      CANONICAL CAPTURE, NOT ONCE PER HANDLER. A handler running N producers has N
      canonical commit points; producer k's output is durable before producer k+1 starts.
      (Basis: detect.py:85-89 writes perceptual hashes — a genuine producer over pixels —
      before backend.detect at :92; BackendUnavailable ∈ ITEM_FAILURES → rollback at
      runner.py:1946 destroys a completed producer's output through a different producer's
      failure. A single per-handler commit point does not close this.) Projection failure
      still leaves capture durable (runner clears txn at handler start, runner.py:1937).
      Standing attack: handler whose second producer raises BackendUnavailable; first
      producer's canonical row survives. [B20/A8-adv]
- T2  Bisect budget: durable rows now decrement as documented (latent bug fixed) — add the
      chosen-consequence reset path + standing attack proving reset works. [exchange]
- T3  Batch = one invocation committed whole (all N outputs durable with item mappings +
      ordinals; members get 1-wide identities FK'd to the N-wide canonical); cancel-after-
      first-item attack: all N durable. Batch width recorded as observation, NOT identity.
      [B21]

### Non-face producers (owner: nonface — vision/captions.py, vision/semantic/*; runner.py
edits routed through runtime)
- N1  BLIP captions through capture: full return (not TEXT-narrowed; token/score structure
      per contract) + canonical FK on derived_annotation. [B22/B3-adv]
- N2  OpenCLIP + Qwen embeddings through capture: dtype-preserving envelope (float64-in →
      float64-back attack) + canonical FK on derived_embedding. [B22/B2-adv]
- N3  db/prompts.remember:211 silent vector drop: capture the computed vector regardless of
      prompt-row existence. [population leak #4]
- N4  Qwen video frame selection recorded (sample identity into the invocation). [leak #7]
- N5  Load-time probe passes + query-path vectors + imagehash: scope ruled at checkpoint
      (see DECISIONS). [leaks #2/#3 + imagehash]

### Boundary enforcement & attacks
- X1  sglint import-boundary rule: model backends importable only by the capture runtime;
      raw producer output exists only inside capture runtime + LIVE proof branch. This IS
      the enforcement of A6's monopoly — nothing in Python stops a resolver bypass, so
      the boundary check is the structural kill, not the convention. [architecture
      round-5 downgrade of "kills B14 structurally"]
- X2  Standing attacks for every bypass B1-B24 + every lettered finding above, each with a
      negative control, each running through a gate that can fail (Phase G prerequisite).
- X3  Migrate existing native tests (stored-face replay/export, OpenCV whole-lane,
      expensive-pass) to the ProducerResult API.
- X4  Gates green + full main suite green + prove-push cached.

## PHASE R2 — proof system rebuild. Owner: proof
- P1  case.py → LIVE/STORED ProducerResult proof cases; STORED branch process-isolated,
      receives only committed identity + legitimate consumer inputs (fresh interpreter where
      imports/caches could leak). [B5/B16 — A6 choreography makes live-replay
      unrepresentable]
- P2  Registry observes raw returns: no str(key), no asarray, no None-drop; registry's
      insightface producer runs the APP's ladder (first_hit_descending), not a parallel
      app.get. Emission deleted; compat routes through the store. [L-6/B12-compat]
- P3  Observation/case fixes: None-valued keys are first-class cases (E5); missing key =
      DIVERGED, no dtype stand-in fabrication (E6); keys carry type (E7); node-type tree
      compared, not just values (E8).
- P4  Population authority: mechanically discovered producer/consumer/variant population +
      dynamic observation as closure conditions; manifest annotates, never defines; census
      disagreement = closure RED. The census includes the proof lane's own tier: ten
      producer invocations in compat/consumers/* (aligned_crop:119, face_family:87/112/
      186/192 incl. app.get(padded), face_selection:100, masked_reference:123,
      reactor_face_model:102, producer_derivations:69, reference_sets:99) plus
      observe_attack:91 — LIVE-branch invocations are legitimate but must be enumerated
      and policed by the import boundary. EXPANDED (population round 2): the hole is three
      directories — compat/consumers (11 sites incl. face_family:127
      init_recognition_model call), compat/vendor/acceptance.py (10 sites incl. :669
      ReActorFaceAnalysis — a THIRD-PARTY producer class in no census — and N→1 narrowing
      wrappers _largest_face/_first_face at :470,:678,:808,:957,:1193), and
      compat/corpus/loaded.py:104 behind the persistent disk cache. Census sweep pattern
      must cover producer.analysis().get(...) shapes, not bare app.get(. [B9/L-8/adversary/
      population]
- P5  Status generated from ProofRuns; doctrine artifact re-stamped with proof IDs (the
      artifact's hand-stamped table currently violates its own B18 rule).
- P6  ExporterBinding: upstream-compatible claims only with pinned upstream
      exporter/loader contract; else explicitly application-native; refusal attack. [B19]
- P7  The 17 generated artifacts no closure condition reads: each becomes read-by-a-gate or
      explicitly documented as informational. [shape 3]
- P8  Tier accounting: the store-reading runner counts toward consumer coverage or a
      consumer-tier store case exists per consumer. [P-9]

## PHASE R3 — closure. Owner: adversary (attack) + lead (verdict)
- Z1  Rediscover population from the settled tree; exact agreement with registered
      contracts or explicit user-approved exclusions.
- Z2  Every standing bypass attack + every mandatory lane green on current-tree identity;
      cold STORED branches without leakage; deliberate verifier attack (one seeded defect
      must go RED).
- Z3  Adversary's hostile closure attack; lead independently verifies; single verdict:
      CLOSED / NOT CLOSED / BLOCKED.
- Z4  Legacy: MEASURED (adversary) — the repo's db/gallery.db holds 0 files/faces/
      embeddings/annotations and all 40 backups hold 0 file rows (migration fixtures, not
      snapshots); tree-migration cost is ZERO and there are no native=NULL rows to mark.
      All prior generated compat evidence still invalidated. The user's PRODUCTION library
      DB, if it lives outside the repo, is unlocated — its re-detect cost is stated when
      found, before any run.

---

## DECISIONS FOR THE USER (checkpoint)
0. THE BOUNDARY RULING both census agents jointly demand: the producer population is
   UNBOUNDED until "producer" is defined mechanically. Decisions 1 and 2 are the same
   question from two directions; your definition, recorded in the doctrine, is what makes
   every later census attack decidable. Lead's proposed definition: a producer is any
   boundary whose output derives from LEARNED WEIGHTS or an EXTERNAL MODEL RUNTIME
   (detectors, recognizers, captioners, embedders, hashers-by-design like imagehash);
   deterministic pixel/byte transforms (decode, resize, EXIF parse) are preprocessing
   whose identity (library version, params) joins ResultIdentity preimages, and their
   narrowing surfaces are tracked as fidelity items, not producer captures.
1. Under that definition — decoders (rawpy/LibRaw, libvips, thumbs): NOT producers
   (preprocessing, identity-bearing). If you rule otherwise, population grows by three.
2. EXIF — RECLASSIFIED (adversary, accepting population's distinction): NOT a
   definition-dependent question. db/capture.py:580 keeps a tag's name and type label and
   drops its VALUE; store() never persists homeless/unrecorded — narrowing under ANY
   definition (two lists built carefully, named precisely, never read: lens item 3 in the
   application). It is a FINDING, in scope, lowest tier — envelope the raw tag map after
   R1 core. Only the decoder question (decision 1) actually needs your definition.
3. padded_recovery (C5): capture both recoveries in the app path (recommended) or delete
   the docstring claims. Strengthening evidence (population round 2): the compat lane
   COMPUTES the padded recovery (loaded.py:113-117) but persists it nowhere — never
   face_put — and the corpus cache holds exactly ONE face per image (best_face collapses
   N→1 before put). Today neither recovery is stored anywhere, app or harness.
4. Probe passes / query vectors / imagehash (N5): recommendation — imagehash yes (cheap,
   real producer); load-time probes recorded as invocation facts, not results; query
   vectors captured (they are producer outputs; storage is small).
5. Wave-2 spend approval: G → R1 (runtime+nonface in parallel post-G) → R2 → R3, same team,
   TaskCompleted/TeammateIdle enforcement hooks armed for implementation tasks.
