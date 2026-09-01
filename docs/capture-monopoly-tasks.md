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
it was classified into, against a DECLARED-CLOSED family list (currently five:
unreachable member G6, empty population G4, inconsistent inputs G9, self-referential
control — recorded 2026-09-01 from the G1-G5 re-attack: a control derived from the
artifact it guards passes at every value of the thing it constrains; discriminator:
could the control fail if implementation and control were wrong in the same
direction?; present-but-unclassifiable — recorded 2026-09-01, F11: a denylist
condition admits every malformed shape it never enumerated, and the empty-only audit
vocabulary cannot catch it — conditions are allowlists, audits inject a malformed
member as their second degenerate shape); an under-classification is then visible as a
finding whose declared parent is its sibling, and a further family arriving is an
explicit recorded event, never a silent label reuse.
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
- G3  Evidence identity covers db/schema.sql, *.sql, ALL *.just modules (globbed from
      the filesystem, never a named list — the list this spec originally carried missed
      api/bench/corpus/schema/web.just, the enumeration failure inside the spec itself),
      justfile, pyproject.toml, uv.lock, metaparse/, conftest.py, tests/ (.py + .sql),
      compat/ root, ty/pyrefly configs. Keep identity.py:147's incompleteness self-check.
      LANDED 0658a03 with 14 controls separating COVERAGE (declared paths present, whole
      trees cannot drop silently) from SENSITIVITY (every part moves the digest — a part
      can be hashed and never fed to digest_of), plus a live filesystem probe and its
      negative half (writes to generated/ must NOT move the digest). Widening the
      identity stales all prior evidence — correct, not a regression. [D3/B17]
- G4  THE EMPTY-POPULATION RULE, three-state conditions everywhere: no condition may
      report held on an empty population; empty is not-applicable; not-applicable is not
      green; every condition declares its non-empty predicate (weights: ≥1 weight row;
      population: ≥1 declared consumer; one-tree: artifacts present to compare). Closes as
      one defect: the closure.py:49-50 one-tree waiver [E4], the weight trio passing on
      weights==[] ("0 weights VERIFIED") [E9], and the bootstrap-vacuity class. Same pass
      unifies the state predicate closure and ledger read (closure whitelists three bad
      states, ledger blacklists ≠VERIFIED — E10, latent today because ledger's blacklist
      is a superset, live the moment the predicates are reconciled the wrong direction).
      Per-face negative controls. ACCEPTANCE additionally: an audit of ALL closure
      conditions (a COUNT is not the spec: at G4's landing this is 8 — the original nine
      minus the weight trio unified into one, plus G1's lane condition, plus nothing
      dropped — and the audit enumerates from closure.conditions() itself, so the set is
      derived, never remembered)
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
      LANDED 1032760: CaseVerdict is PASS/FAIL-only and CaseResult.verdict is typed to
      it; the clean term reads ablation_tally() (emitted as ablation_verdicts by both
      evidence writers); closure condition `every ablation concluded` — CONTRADICTED
      zero-allowance, INCONCLUSIVE bounded by ABLATION_INCONCLUSIVE_ALLOWANCE=497,
      DECLARED A RATCHET (lower-only; raising it is a recorded ruling, never an edit).
      Controls: 15 closure_attack mutations zero missed incl. ablation_contradicted and
      ablations_over_allowance; condition_audit 9 conditions / 7 sources with an emptier
      for the new population.
- G7  The GREEN-with-red-attack demonstration (adversary attack_green.py: all nine
      conditions ok via ledger.build() itself with attack=1, selftest=1 in the same
      output) becomes the standing negative control for the whole G phase: after the other
      six land, re-running it must print RED. SCOPE HONESTY: it stamps digests and does
      not empty the weight set — strong for the green-with-red-lanes path it demonstrates,
      NOT a general vacuity detector; G4's empty-input audit covers what it cannot.
      AMENDED (adversary F9, reproduced attack_g7_control.py): "must print RED" is
      satisfiable by a stale artifact — the original probe predates G1's lanes.json
      shape change and now prints the SAME RED whether the attack lane failed or passed
      (missing identity stamp + unparseable lane block, not the guarded defect).
      Acceptance is the REASON, never the colour: RED whose failing condition is
      `every lane exited 0` with the attack lane named in its detail, probe written
      against the CURRENT lanes.json shape, never the pre-G1 artifact verbatim. Third
      direction of the self-referential-control family: a control frozen as an ARTIFACT
      rather than as a CLAIM drifts out from under the claim while the claim keeps
      passing. (Lead verified: proof's staged green_attack.py already uses the current
      shape at :43 and asserts the named condition at :45.)
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

### G repairs — from the adversary's G1-G5 re-attack (2026-09-01, all reproduced;
### report scratchpad/wave1/gates-g1-g5-reattack.md, probe scratchpad/adversary/
### attack_g1_g5.py). G2 and G5 re-attacked CLEAN. Phase G is not complete until these
### land with observed-failing controls.
- G1r Deleted lane invisible: the gate iterates lanes that RAN; the set that SHOULD run
      lives only in compat.just's shell loop, compared to nothing. Reproduced: absent
      attack lane -> GREEN "ok 2 lane(s), all 0". Fix: declare the expected lane set as
      data; closure fails on expected-minus-recorded. Control: remove a lane, observe
      RED. [family: self-referential control (neighbour); adversary re-attack F1]
      IN-TREE FIX VERIFIED WORKING by the adversary (29 lanes parsed from compat.just,
      per-lane "declared but never recorded") — its P1 attack hypothesis on G7 was
      refuted BECAUSE this repair works. TWO REQUIREMENTS before it commits:
      lanes.declared() must FAIL on a regex parse miss, never return () and fall back
      to judging the record by itself (the exact defect G1r fixes); and G7's `named`
      check should require '[failed]' beside the condition name, not a bare substring.
- G2r The G2 control artifact overstates itself (adversary F12; the G2 IMPLEMENTATION
      verdict stays CLEAN): three of six stages (durable.written_bytes,
      durable.read_back_sha256, durable.thawed_keys) are exercised only against data
      ledger_attack._case() fabricates — CaseResult has no `durable` field and 0 of 302
      shipped results carry one, which ledger_attack's own comment admits. The ledger
      correctly BLOCKS those stages on real evidence (agrees with B5: 12 of 13 runners
      replay from live memory); the defect is ledger_controls.json recording "6
      stage(s), 0 not independently derived" without naming that split. Fix: check
      real cases.json for which declared fields a writer emits and record "exercised
      against real evidence: 3/6", so a real durable write becomes a visible event,
      not a silent BLOCKED-to-VERIFIED flip. [family: self-referential control
      (fabricated-evidence direction)]
- G3r Gate identity digests by hand-written list; control copied from the same list
      (identity_attack.COVERED["gates"] == GATE_GLOBS), so both wrong in the same
      direction cannot fail. GATE_GLOBS (identity.py:34) misses lefthook.yml — the most
      gate-deciding file in the repo — plus pytest.ini, .vale.ini, biome.json. Fix:
      digest by EXCLUSION (every git-tracked root/gate file minus a declared ignore
      list); control must be derived independently of the implementation's list.
      [family: self-referential control; adversary re-attack F3]
- G7r Two defects in the landed G7 control (adversary at d124c32; G7's F9 compliance
      itself verified — reason asserted, scope honest, no lane pretense): F14, the
      waiver probe guards a STRING — `"where != GENERATED" not in source` catches
      exactly one spelling; four rephrasings and a behavioural re-add via _read's
      default all reinstate the waiver undetected. Fix is behavioural: fixture in a
      non-default directory with mismatched stamps must still fail `evidence from one
      tree`. F15, the probe type is two-state where Condition is three-state: absent
      shipped artifacts return held=True with an honest detail main() never reads, so
      G7 reports 3-of-3 having tested nothing — the empty-population defect inside the
      control guarding Phase G, F6's one-of-two-types shape again. Latent today.
      CROSS-CONFIRMATION: G7's own shipped probe-2 detail ("no ledger cell BLOCKED
      [failed]" on re-stamped REAL evidence) demonstrates F12/G2r independently.
      [families: self-referential control (artifact direction) + empty population]
- G-rp REPAIRS IN FLIGHT (proof, uncommitted): G1r lane population parsed from the
      compat.just recipe itself; G3r coverage probe decoupled from GATE_GLOBS
      (enumerates *.just from the filesystem — NOTE: G3r's full scope is digest-by-
      exclusion so lefthook.yml/pytest.ini/.vale.ini/biome.json are covered, not only
      an independent *.just enumeration); G6r ceiling derived from the INCONCLUSIVE
      count in shipped evidence — a number the constant cannot move — verified firing
      at an inflated allowance and silent at the real one. PLUS two self-found by
      applying the discriminator to its own probes: _sensitivity iterated PARTS while
      digest_of also uses PARTS (a dropped part tested by nobody) — now checked
      against identity()'s own keys; and set(CODEC) <= set(held) vacuously true for
      empty CODEC — emptying the constant passed the probe guarding it. Fixed.
- G4r Declared population not tied to the checked collection: _over takes population,
      size, offending independently; "no skipped input" and "no shard failed"
      (closure.py:195,202) declare cases.json:results while checking skipped /
      shards_failed, and condition_audit proves them by emptying results — a set they
      do not read. Fix: _over takes the collection and derives size from it; audit
      empties the collection actually read. [family: empty population + self-referential
      control; adversary re-attack F2]
- G5r closure_attack is not hermetic: ledger_attack.green_fixture() captures the tree
      digest ONCE (:38) while every _write's ledger.build() recomputes it LIVE
      (ledger.py:118) — a mid-run tree change blocks every later stage and reports the
      mutations "missed" when they correctly went red (reproduced: 9 of 15 false-missed
      after proof's concurrent commits). condition_audit.run() shares the structure and
      would produce spurious held-over-empty ACCUSATIONS. G5's derivation verdict stays
      CLEAN — this is reliability, and it matters because closure-attack runs as a lane:
      a flake that cries wolf on the alarm most needing belief. Fix: thread ONE captured
      digest through the fixture and build (or stamp the fixture with what build
      computes at write time), in both harnesses. Control: mutate the tree mid-run;
      the run must be order-independent. [adversary re-attack F5]
      COMPOUNDED (F10, modest): identity_attack._live_probe writes a real .just file
      into the repo (identity_attack.py:86, cleaned up correctly by its finally) — a
      second, self-inflicted source of mid-run digest change alongside concurrent
      commits; observed externally in a routine git status. _live_probe stays as-is
      (its end-to-end-against-the-filesystem value is deliberate); G5r's hermeticity
      removes the blast radius. Sequential `just compat run` unaffected.
- G6r The INCONCLUSIVE allowance ratchet is a comment: nothing enforces lower-only, and
      closure_attack.py:84 builds ALLOWANCE+1 rows so the control is red at EVERY value
      including 1037 (verified green over 1037 INCONCLUSIVE at allowance 1037). Fix:
      pin the bound outside closure.py (the attack asserts allowance <= the recorded
      497 as a literal it does not import) and make the over-allowance mutation use a
      fixed count above the pin, so raising the allowance makes the control fail.
      [family: self-referential control; adversary re-attack F4]
- G6r2 The totality technique was applied to one of the two verdict types:
      Ablation.verdict is still `Verdict | None` (case.py:227) while CaseResult.verdict
      went total in the same commit for the same defect class. A null verdict matches
      neither CONTRADICTED nor INCONCLUSIVE, so "every ablation concluded" holds over
      50 ablations that concluded nothing (reproduced), and ablation_tally
      (run.py:294-296) tallies them to all zeros. Fix: make Ablation.verdict total or
      count null as unconcluded in BOTH the condition and the tally; assert the free
      invariant sum(tally.values()) == len(ablations). [family: unreachable member;
      adversary G6 attack F6/F7]
      SUPERSEDED BY F11 in fix-shape: null is one instance of PRESENT BUT
      UNCLASSIFIABLE (verdict "WOBBLE"/absent/"" all pass too, reproduced). The fix is
      the ALLOWLIST form proof already wrote for weights (weight_is_verified — whose
      own comment warns of exactly this polarity): anything not a known-good verdict
      is an offence. The invariant sum(tally)==len stays.
- G4r2 The audit's degenerate vocabulary has one word: every emptier makes a
      population absent or empty (set []/{} / pop key / unlink); none makes it PRESENT
      BUT MALFORMED, so G4+G6 jointly certify a condition sound over unclassifiable
      members — by construction, not oversight. Fix: every population in SOURCES gains
      a second degenerate shape — inject a present, unclassifiable member; the
      condition must go red or the audit fails it. [family: present-but-unclassifiable;
      adversary F11]
      AUDIT ATTACK RESULTS (adversary F13 pass): self_control() HOLDS — it fails
      loudly when run() produces no findings, refuting the vacuous-self-control
      suspicion; orphan-source detection HOLDS. LATENT F13: run() keys conditions by
      name in a dict, so duplicate names collapse silently and the printed count
      cannot show the drop — the count-is-not-an-enumeration failure inside the audit
      that enforces enumeration. No duplicates exist today. One-line fix folded into
      G4r work: assert len(control) == len(asked(where)), failing with the duplicate
      named.
- G10 Three gates, one missing idea: every coverage mechanism measures the population
      it was HANDED; none measures whether the RIGHT members are in it. Lanes that ran
      vs lanes that should (G1r), files globbed vs files that decide gates (G3r), and
      now ablations aggregated over all consumers: gallery_storage — the ONLY runner
      doing a real application-store round trip, the thing the invariant is about — is
      silent in three independent mechanisms at once (no consumer coverage via
      Tier.PRIMITIVE, zero ablations, invisible in the aggregate; counted over shipped
      evidence, 28 consumers). Fix: ablation counts reported PER CONSUMER; store-
      touching consumers must declare at least one; the general rule — a recorded set
      is always checked against a DECLARED expected set. Ties to R2-P8. [adversary G6
      attack F8]

## PHASE R1 — monopoly core (application). Owner: runtime (+ nonface for R1-N block)

### Schema (the canonical class — greenfield migration v47→v48, no legacy promotion)
S1-S6 + S8 LANDED b0ae4f2 on capture-monopoly-r1 (architecture, full gate green, 815
passed; SUBJECT OVERCLAIMED "S1-S8" — architecture self-corrected by grepping the
schema: S7 projection FKs and S9 reproject are NOT in it; the only result_id is the
DAG edge): ten tables at v48, none derived_-prefixed, immutability + permanence triggers
on every canonical table including contradictions; db/producers.py the only writer;
migration replays what schema.sql builds (byte-identical sqlite_master verified by
drift from a v47 snapshot, not asserted); S8 negative control OBSERVED RED (drop_all
assertion disabled -> rename attack fails; restored -> passes; plus every declared
canonical table must exist). TWO SHARED-TEST GATE CHANGES flagged for adversary
review: the insert-only-guard rule narrowed to its stated meaning (with
test_the_update_bypass_rule_can_actually_fail holding three cases), and the
version-step test de-pinned from literal 47 to USER_VERSION with a
no-step-off-current discriminator. S9 (reproject detector) remains.
S-REPAIRS from the adversary's b0ae4f2 attack (executed mutations, not trigger-text
readings; six HOLDS incl. contradiction-table gut-rewrite BLOCKED and the de-pinned
version test discriminating both ways; owner: architecture):
- S1r (F18, WORST — acceptance rule's last clause): the row carrying identity AND
      preimage_json is neither immutable nor permanent — one UPDATE to preimage_json
      and the stored identity no longer matches sha256(preimage); a resolver hit then
      serves immutable bytes with fabricated provenance. Fix: guard the row, and/or
      enforce identity == sha256(preimage) by trigger; control mutates the preimage
      and must be blocked or detected.
- S2r (F17): producer_determinism blocks UPDATE but DELETE SUCCEEDED — the exact
      inversion producer_contradiction's own comment warns about ("they DELETE these
      rows"). Waived rows survive via ON DELETE RESTRICT; unwaived (the normal state)
      do not. Add the DELETE guard.
- S3r (F19): four of eight canonical tables unguarded in at least one direction;
      producer_invocation, producer_input, producer_variance carry NO RAISE(ABORT)
      trigger at all. STRUCTURAL: the shared-test audit builds its population from
      triggers that EXIST, so a zero-guard table is never classified — table_class is
      the declared expected set and is not joined to the guard check. Same family as
      G1r/G3r/G10: join the audit to table_class so every declared canonical table
      must show both guards.
- S4r (F20): drop_all's declaration closes RENAME both ways but not OMISSION —
      declared-minus-present asserted, present-minus-declared not; an undeclared
      producer table survives only by lacking the derived_ prefix, the convention this
      commit replaces. Assert both directions.
- S-open (target 4, unreached): is the v47 snapshot behind the migration-replay drift
      check DERIVED or hand-written? If retyped, the check compares the migration
      against a transcription of the thing it verifies. Adversary takes this next.
- Note (dormant): guard "cover" is presence-based — a scoped UPDATE guard would
      excuse a DELETE-only guard; live the first time someone writes UPDATE OF.
- Disclosure: producer_input's exposure rests on trigger census, not an executed
      mutation (adversary's own INSERT was malformed).
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
V1-V5 LANDED a55082f on capture-monopoly-envelope (envelope teammate, one commit, +602
lines incl. a 295-line shape test file): flat node table — one object one node, so
aliasing survives thaw (V1); in-flight object = cycle, Unpreservable naming the
closing key (V2); per-node container class, subclasses rebuilt via registered adapter
or refused by name, never widened (V3); root container rebuilt — insightface Face
protocol held by InsightFaceRecord WITHOUT importing insightface (V4); strided values
bit-exact with layout normalization stated honestly, F-contiguity restored because
consumers can observe it (V5); magic bumped to sgface3, version-2 blobs refused by
name. CONTROLS EVIDENCE RECEIVED (second commit d6c90dc; 16 attacks in
tests/test_the_envelope_gives_back_the_shape_it_was_handed.py): nine observed failing
against the real pre-fix artifact (cycle attacks died with RecursionError at
facestore.py:160/153 — the defect V2 names; V4 failed as AttributeError 'dict' has no
'age'/'nose_tip' — C1-adv verbatim), three observed failing against a55082f's
collections-only stage; V5's two (docstring defect, pre-fix green expected) got
seeded-defect controls each turning exactly its own tests red. V3 walked past
:151/:154 to every subclassable node kind (MaskedArray, IntEnum, tensors, strings,
bytes; numpy scalars and bool exempt with reasons). V4 verified against the INSTALLED
insightface (real Face round-trips; upstream __getattr__/__setattr__ transcribed from
the pinned commit; consumer-path attack runs ONE unchanged consumer against live and
stored incl. absent-key). sgface2 refusal justified: no sgface2 data exists (R-6 +
Z4); the corpus cache hashes facestore into every namespace. RESIDUAL (stated):
register_container is module-global — a reader without the writer's adapter refuses
loudly at thaw, never reinterprets. ROUTED: root-container declaration-vs-value-
derived is architecture's C-block decision (envelope supports both); torch.nn.
Parameter refuses without an adapter — population registers one before N2. THIRD
COMMIT 9ee5413 (envelope self-attack via the fourth-family discriminator): a memo
keyed by EQUALITY rather than identity satisfies every V1 aliasing attack while
folding equal-but-distinct values into one — reproduced (repr-keyed memo passes 16 of
17); complement control added, test_two_equal_but_separate_values_stay_separate, all
17 pass restored. ADVERSARY ATTACK AT HEAD (9ee5413): NO FINDINGS — the first landing
of the campaign to survive every attack, attacked by the author of the original
C1-C5. Aliasing held three-deep with aliased-containers-of-aliases; cycle message
names path AND closing container; all eight subclass shapes refused incl. nested
OrderedDict; the V4 one-consumer question REFUTED (one function, both sides, tuple
equality, covering exactly the four things a plain mapping answers differently, with
isinstance(stored, dict) blocking the cease-to-be-a-mapping dodge); sgface2 refused
with reason, not generic parse failure; two off-list surfaces clean
(register_container collision refusal; _declared trials the adapter at freeze).
NOT-TESTED recorded with reasoning: forged node table with recomputed digest — the
trailing sha256 is unkeyed, so that attacker already has DB write access. Report:
scratchpad/wave1/envelope-v1-v5-attack.md. Status: PROVEN pending merge only.
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
      LANDED 1f2f963 on capture-monopoly-r1: producers.resolve is the single doorway;
      HIT never calls execute; MISS runs once, freezes, and DROPS THE LIVE OBJECT — the
      caller receives bytes decoded from the store, so a projection cannot read what
      the envelope failed to preserve. capture injected, not imported (db/producers.py
      knows nothing of producer output shapes). 24 standing attacks; the instrument is
      a producer counting its own calls ("nothing reran" read off the producer, never
      inferred from time or rows — a resolver that ran and discarded would satisfy
      every symptom). OBSERVED FAILING: hit branch removed -> 4 red incl. both cache-
      mandate halves; identity positive half = one attack per field the old freshness
      never carried (implementation, weights, configuration, provider, codec);
      negative half = a resolve a day later from another job still hits. PROVEN
      pending adversary attack + merge.
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
      MEASURED 2026-09-01 (population, RTX 3070 Ti, 128 pictures, artifact
      openclip_batch_i6_rerun.json alongside the preserved Aug-24 baseline): max_abs
      6.44e-04 at widths 8-128 — the documented 2.2e-03 is 3.4x larger and does NOT
      reproduce; bit_identical False at every width>1 holds; preprocess_equivalence
      reproduces exactly. ROOT CAUSE deeper than staleness: the benchmark corpus is
      unpinned (newest-N by id DESC — every import changes the sample) and the artifact
      records a COUNT, never file ids or a corpus digest, so no two runs can be shown
      to measure the same thing. openclip stays `unjustified` until: corpus pinned
      (explicit id list or digest), that identity written into the artifact, artifact
      tracked (I7), docstring restated from the tracked artifact or deleted.
- I7  Cited evidence must be readable: five shipped constants are justified by
      benchmarks/results artifacts that are gitignored and untracked (openclip_batch,
      caption_batch, face_pipeline_validation, fingerprint_calibration, answer_currency
      — the last backs a test), machine-local only (population, verified: .gitignore:84
      blanket + hand allowlist that never grew with citations; blanket from 57dbdcd,
      allowlist from 2ca81c5). RULING (user-corrected): do NOT extend the allowlist —
      the allowlist is the defect. Structural fix: a wholesale-tracked evidence
      location (no ignore, no allowlist) holds every artifact cited by shipped code;
      citations repoint; the results/* allowlist is deleted once tracked evidence
      moves out; transient output stays blanket-ignored. Enforcement: a check derives
      the cited-path set from code (the population rg pattern) and fails on any cited
      path that is ignored, untracked, or absent — the declared set is DERIVED from
      citations, never named. A citation that cannot resolve is deleted and its
      constant declared unjustified. Same shape as G9 with the arrow reversed: a
      record cited that cannot be read. [population wave-2 finding; crosses owners]
      ENFORCEMENT LANDED in the scanner (population): third state IGNORED with the
      exact rule from `git check-ignore -v` — all five resolve to .gitignore:84, whose
      own comment above it states the correct policy the hand allowlist fails to keep.
      NEW: the allowlist is wrong in BOTH directions — five cited artifacts ignored,
      EIGHT allowlisted artifacts cited by nothing (incl. face_detection_recall_ms1600
      while its cited sibling _native resolves). The evidence move handles the five;
      the eight are a separate reviewed question (still-wanted or dead), never a guess.

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
      INSTRUMENT DELIVERED (population, 2026-09-01, compat/producers/census.py,
      uncommitted pending tree-green): AST-derived from git ls-files, NO directory
      argument, NO site list; loader vocabulary imported from
      compat.harness.population.LOADERS (one list, never two). Acceptance 9/9 — every
      site that hid from BOTH manual censuses rediscovered, including the three
      producer.analysis().get() CHAINED shapes no receiver-name pattern can see.
      Census: 77 LOAD+CONFIRMED+CHAINED sites over 64 files (compat/vendor measures 25,
      not the 10 found by hand). Three self-found defects fixed, each the machine form
      of a human miss; discriminates dict.get by SIGNATURE (arity), not receiver name.
      HONEST LIMIT: CANDIDATE=342 is too noisy for a gate today; suppressed counts
      recorded in the JSON, never filtered. RULING (lead): closure wires
      LOAD+CONFIRMED+CHAINED against registered contracts; CANDIDATE is a reviewed
      queue that must be empty-or-explained, never a raw count — wiring is P4 work in
      proof's files, sequenced with R2. Exit 1 on unresolved stands: unresolved is not
      absent.
- Z2  Every standing bypass attack + every mandatory lane green on current-tree identity;
      cold STORED branches without leakage; deliberate verifier attack (one seeded defect
      must go RED).
      PRECONDITION (proof, 2026-09-01, from a spurious G9 probe failure): the closure
      run requires a QUIET TREE — no concurrent writers. Any control that computes
      identity() twice in one run is racy under shared-tree concurrency (observed: a
      teammate wrote db/derived.py between two identity() calls); controls compare
      against a recorded stamp, identity() is memoized per run with explicit forget().
      R3's full run happens with every other agent stopped or in a separate worktree.
- Z3  Adversary's hostile closure attack; lead independently verifies; single verdict:
      CLOSED / NOT CLOSED / BLOCKED.
- Z4  Legacy: MEASURED (adversary) — the repo's db/gallery.db holds 0 files/faces/
      embeddings/annotations and all 40 backups hold 0 file rows (migration fixtures, not
      snapshots); tree-migration cost is ZERO and there are no native=NULL rows to mark.
      All prior generated compat evidence still invalidated. The user's PRODUCTION library
      DB, if it lives outside the repo, is unlocated — its re-detect cost is stated when
      found, before any run.
      AMENDED (population census + USER RULING, 2026-09-01): the populated DB at
      C:/Users/will/.smartgallery/gallery.db (v45, 3,514 faces, 72 person_assertions —
      full census in ruling R-6) is a TEST gallery per the user, not production. The
      census stands as measurement; the DB serves as the realistic S8/migration
      fixture; a production DB remains UNLOCATED and Z4's cost-statement requirement
      re-attaches to it if one is ever identified.

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
