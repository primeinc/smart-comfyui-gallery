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
- G9r (adversary F22 on the landed G-wave; three other vectors HOLD — the G6r pin
      closed twice over with the shipped-evidence ceiling as the enforceable ratchet;
      G8's namespace-change-IS-invalidation verified with unreachable-not-deleted as
      a disk residual; G3r's ignore-list is a visible decision, not an omission):
      ledger._read's `consumed` parameter is OPTIONAL defaulting to None and omission
      is SILENT — executed: without the argument the same file is opened and read and
      nothing is recorded — while consumption_attack's derivation control compares
      against the hand literal {"provenance.json","cases.json","lanes.json"}, so a
      fourth _read omitting `consumed` passes the very control whose comment claims
      derivation. The self-referential family inside the fix for the inconsistent-
      inputs family. Fix removes the CLASS: make `consumed` REQUIRED (omission =
      TypeError, unrepresentable), demote the literal to a tripwire — the CaseVerdict
      move. [owner: proof, fold with in-flight repairs]
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
      G4r2+F13 LANDED d894fa2: every source carries the second degenerate shape (a
      member no classifier claims; every condition over that population must stop
      holding — per-condition, after proof caught itself weakening the rule to "the
      verdict must cost" and the self-control caught the weakening);
      AXIS SCOPE (adversary F23, post-respawn — verified all eight malformers red on
      their own axis): "shape" as implemented means the VALUE axis only — a fully
      populated member with an unrecognised value. The MISSING-KEY axis is untested
      across all eight populations, and four hard subscripts (closure.py:320,
      ledger.py:161/190/233) turn a key-missing artifact into a traceback instead of
      a verdict — fail-loud, not false-green, and F23 IS the mutation evidence the
      ledger.py:189 widening was waiting on: fix the malformer axis, not the four
      .gets one at a time. FURTHER (same round): F24 closure.py:145 truncates
      unconcluded[:4] BEFORE _over counts (220 CONTRADICTED report as "4 of 220");
      F25 lanes.exits() still coerces where closure classifies (false/0.0 fabricate
      clean exit 0; WOBBLE/null kill ledger.build); F26 the ledger's own lanes block
      is read by nothing (29 fictional lanes moved zero conditions). ORDERING
      CONSTRAINT: F25 lands BEFORE F26 — giving the lanes block a reader while
      exits() still fabricates converts an inert fabrication into a load-bearing
      lie. HELD this round: all eight value-axis malformers, classify-don't-coerce
      in the judge, the empty-population rule, G4r's collection-derived _over.
      [owner: proof, folds with the post-loop staleness item] a THIRD denylist
      found immediately ("no ledger cell BLOCKED"/"FAILED" both passed a WOBBLE cell
      — now one allowlist `every ledger cell VERIFIED`, importing the ledger's own
      constant since two spellings of the good state is how E10 happened); F13
      duplicate names are a named finding; _malform creates absent fields (two
      "findings" had been the injector no-oping); dual-shape self-control or main()
      refuses to report. 9 conditions over 8 sources VALIDATED EMPTY.
      ATTACKED (adversary): projection-miss vector HOLDS (a condition judging a field
      no malformer writes is caught loudly); both mid-landing corrections VERIFY by
      pre-fix reimplementation. THREE REPAIRS, fold into proof's frozen candidate:
      (1) DISTINCTNESS — nothing asserts VERIFIED differs from FAILED/BLOCKED;
      aliasing VERIFIED onto BLOCKED grades 22 blocked cells green with self-control
      passing, and the ledger already writes the contradiction (green=22,
      with_blocked=22, declared=22) with nothing reading it — hardening rank, needs a
      deliberate edit; (2) the MERGE-HOLD one: condition_audit.py:263-265 still
      argues the withdrawn verdict-must-cost rule in the author's voice above the
      loop that replaced it — the reproduction proves adopting that comment breaks
      the self-control; (3) read the on-disk totals contradiction. F13 + malformer
      coverage otherwise hold.
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
      FIRST HALF IN PROOF'S CANDIDATE: `every consumer in evidence has a ledger row`
      — runner-named consumers vs ledger-built rows, two producers that cannot be
      wrong in the same direction. RED on shipped evidence: 6 of 28 undeclared
      (gallery_storage worst — 36 cases, 0 ablations, F8's triple-blindspot exactly);
      the old `every declared member accounted for` stays green because both its
      sides derive from the manifest. 17th mutation observed red; unclassifiable-
      consumer injection per F11. RULINGS: the RED stands — it is the gate working;
      closure goes green only legitimately (ledger covers 28 + lanes re-run, R2/R3).
      Store-touching half DELIBERATELY UNDONE: taxonomy DERIVES from what consumer
      modules actually read (scanner pattern family) in R2-P8 — never a hand list.
      First-party source-provenance RULED (architecture, REJECTING the lead's
      tree-identity lean with cause: tree identity IS the staleness fact the seven
      BLOCKED cells already carry — grading by it is two cells from one object, the
      G2 defect; and an auto-pass category is six checks that cannot fail, the exact
      door for reclassifying a third-party consumer to dodge the pin). RULING: a
      DECLARED first-party classification CROSS-CHECKED against the pin, four
      states — declared+unpinned HELD; declared+pinned FAILED (contradictory);
      pinned+undeclared graded as today; NEITHER = FAILED unclassified, so a new
      consumer fails until classified rather than defaulting into the excused
      bucket. Same shape as table_class: a declaration that can be wrong and is
      checked, never a category granting exemption. Proof implements.
      CANDIDATE DELTAS (proof): the VERIFIED-import alias hazard closed BOTH ways —
      closure refuses the cell condition if any two state constants spell one
      string, and a new condition recomputes the ledger's four totals from rows on
      disk requiring green disjoint from failed+blocked (disjointness survives an
      alias whatever the states spell — the green=22/with_blocked=22 contradiction
      is now READ). Proof attacked its own G10 condition: first version filtered
      out cases lacking consumer_id — the F11 hole inside the F8 fix — now an
      unattributable case is an offence; the new mutation exposed a REAL recorder
      bug (ledger.py:189 hard subscript killed build() with KeyError before closure
      could classify — a check that cannot fail because the recorder dies first),
      fixed to .get, deliberately NOT widened to the other subscripts without the
      same mutation evidence.

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
ALL FOUR LANDED 6355b2e on capture-monopoly-r1. S1r: producer_invocation immutable +
permanent; sha256 can't run in a trigger, so prevention (nothing moves post-insert) +
DETECTION via producers.identity_disagreements() recomputing each identity from the
stored preimage + ordered inputs — with the detector's own control DROPPING the
trigger, mutating a preimage, asserting the detector sees it ("prevention nobody can
observe is indistinguishable from prevention that quietly stopped working"). S2r:
DELETE guard added. S3r: three tables gain both guards; the check now STARTS FROM
table_class, and a companion refuses any canonical guard scoped UPDATE OF/WHEN
(closes the dormant scoped-cover hole early). S4r: completeness both directions with
a manufactured-undeclared-table control. OBSERVED FAILING: guards removed = the
b0ae4f2 state -> six checks red incl. the declared-set audit seeing a zero-guard
table. BONUS: architecture self-audited the migration-replay control — the old
snapshot was of a database it had built (partly self-referential, F9 family, its own
admission); now compares against db/schema.sql AS GIT HOLDS IT at 1032760, subject
and control independent, with an honest scope statement (proves replay fidelity, not
semantic rightness). Still owed from the target-4/F21 round: vendor v47 snapshot,
fix the tests/schemas.py docstring claim.
F21 CLOSED cec2860: tests/schemas/v47.sql vendored from 0bfe0f5c (the commit that
shipped v47) — @step(47)'s exact starting state now seeded; `just schema prove`
enumerated: ten seeds ok, v1/v2 red = the pre-existing KNOWN_DRIFT hole the docstring
itself describes (recipe exited 2 before and after — no contributed failure).
Docstring count REMOVED, pointing at `just schema versions` instead (a count frozen
beside the claim it describes; git holds v1-v48, v43 absent, learned by running it).
ACCURATE CONTROL STATEMENT (architecture's self-correction under the forwarded-claim
rule, which its own commit message had violated): ONE durable control stands behind
the migration-replay claim — the vendored v47 — and the 1032760 build was a one-off
interactive check leaving no artifact.
ADVERSARY ATTACK ON 6355b2e: NO FINDINGS — F17-F20 all closed. Vector 1 held beyond
its own tests: producer_input correctly has no INSERT guard (edges are writable at
creation), and the adversary's added-edge probe was CAUGHT by
identity_disagreements() (stored c15b185a…, recomputed 876cd80e…) — prevention covers
UPDATE/DELETE, detection covers INSERT, split complete. MISSING CONTROL flagged to
architecture: the detector's own control only manufactures the invocation-preimage
disagreement; the upstream-join arm is reached by no detection test — the adversary's
INSERT probe IS that control, adopt it. Trigger-drop control raises (not silent) on
rename. Scoped-guard companion filters to declared canonical tables. MECHANISM
DESCRIPTION CORRECTED (the lead's forwarding was wrong, second instance — see the
forwarded-claim rule): drift compares migrated-vs-fresh from the CURRENT tree; the
anchor against real history is the SEED (git-derived vNN.sql fixtures) — the right
place. _V48_OBJECTS (migrate.py:3928) is literal DDL whose comment claims "replayed"
— it IS retyped, made safe by the drift check the same comment names; fix the
comment's two disagreeing halves.
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
- S-open (target 4): RESOLVED — HOLDS, the strongest discriminator instance in the
      project. tests/schemas.py seeds from tests/schemas/vNN.sql vendored VERBATIM
      from `git show` at the shipping commit, never edited; the docstring records the
      inverted-fixture trap ("IT STARTS FROM THE ANSWER") and the real v1/v2 bug the
      inverted fixture hid for thirty-five versions — the self-referential family
      found and repaired by architecture before this review existed. Nine authentic
      seeds (v1..v31) traverse the new v48 step. F21 (minor, doc): the commit message
      says "v47 snapshot" — no v47 snapshot exists (inventory tops at v31), and the
      module docstring claims "v1 through v35" over eleven files; neither weakens the
      verification. OBSERVATION: no seed starts in v32-v47 — vendor a v47 snapshot
      (`just schema vendor 47`) with the S-repairs, which also makes the commit
      message true. [owner: architecture]
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
      TWO PRE-WRITE DESIGN FINDINGS (architecture, both the chain-stops-one-step-
      early flavour — nobody wrote the missing step, so walking mechanisms to their
      ends found them where reading checks could not): (1) ADDRESSABILITY — the FK
      proves parentage but producer_result.invocation_id is UNIQUE and the envelope
      holds the WHOLE output list, so N face rows point at ONE result and reproject
      cannot tell which element a row projects. S7 carries an ORDINAL beside the FK:
      (canonical_result_id, result_ordinal) written at projection time; an
      out-of-range ordinal is itself an offence. The best-agreement matcher is
      REJECTED as circular (agreement is the property under test — self-referential
      family inside the lineage fix). (2) FRAME SIZE — bbox/kps normalize against
      the ORIENTED frame's w,h (faces.py:551; EXIF 5-8 swap dimensions per
      oriented.py:53-59; videos have NULL file dims), which the store never writes:
      geometry is unre-derivable today and NOTHING in the tree can check a geometry
      column against its record. The capture frame's dimensions join the invocation
      CONFIGURATION (already an identity preimage field). Both fold into the
      S7+C1-C6 unit's DDL/config; neither moves its boundaries.
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
PREMISE VERIFIED (architecture, empirically against envelope's branch, pre-
implementation): freeze takes a live LIST OF Face RECORDS — the real capture shape.
No adapter -> loud refusal naming path and type; adapter -> 494 bytes, root back as
list, ELEMENT back as Face (not dict), attribute access works, arrays bit-exact,
BOTH detections present (C2's point: today six filter sites drop sub-threshold
detections before anything stores — verified still at their cited lines). The derive
ruling holds in practice: the value's own dotted name short-circuits _declared.
ENVELOPE MERGED INTO r1 at 1319191 (850 passed; capture door verified buildable — a
live dict subclass survives freeze/thaw with attribute access and missing-key None;
adversary's candidate files unmoved, verification command supplied). Architecture
SUSTAINED envelope's registration-signature objection (matching the lead ruling) and
committed attack D — every registered revision equals its adapter file's digest — as
a hard requirement: "if it's not in the C-block candidate, the candidate isn't
finished." Its own isinstance pre-flight was self-caught as trivially true (lambda
constructing the same class); the real stand-in case is False by construction and
does NOT bite C1 (the only isinstance in the projection path tests a mapping the
projection constructs itself).
SEAM CONSTRAINT ON C1 (envelope, landed 6684ba7 — a test reading ATTRIBUTES on the
real harvest lane): derive-the-declaration and UN-FLATTEN-THE-VALUE are one move,
not two — vision/faces.py:570 hands freeze a flattened plain dict, so deriving
yields builtins.dict and V4 regresses to the defect it closed; neither owner's
existing controls could see it (envelope's V4 attack pins the declared path;
subscripts can't distinguish a mapping from the record). The new test is red
whichever way the runtime goes. Envelope also self-refuted its first coarse control
and shipped the narrower claim with the refuting control named.
SECOND C1 CONSTRAINT (envelope): the thawed element is a structural stand-in —
isinstance(thawed, Face) is False, forced by cold replay (importing insightface is
what cold replay forbids). C1's projection function must DUCK-TYPE; an isinstance
check in the projection path would reject every stored record.
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

### Capture-adapter registry (C-block item, from envelope's proposal — architecture
### rules; full text in architecture's inbox)
- C7  RULED (architecture, reply sent to envelope): proposal accepted essentially as
      written, with the missing piece added — X1's import boundary is what closes the
      capture non-determinism structurally: refusal-at-capture requires consulting
      the registry, which made capture import-dependent; the fix is exactly one
      capture door, not import hygiene. Option (b) declined (adapters do NOT live in
      facestore); adapters live in the capture package, contents derived from
      declared contracts; identity contribution scoped to records that recorded a
      container name. Envelope's determinism attack extends to the absent-dependency
      case (cold-replay machine) — same non-determinism wearing a dependency failure.
      Implementation AFTER the merge (register_container does not exist on r1; no
      second registry gets written).
      AMENDED (envelope walked its own accepted point 3 to the end — it does not
      work): ResultIdentity is INPUT-derived and computed BEFORE the producer runs
      (I2's HIT skips execution), so "scoped to records that recorded a container" is
      uncomputable at key time; and adapter rebuild output NEVER reaches stored bytes
      (verified by enumerating every rebuild call site: _container records the name
      only; the one rebuild call is _declared's discarded trial) — a module digest in
      the key would force a re-detect guaranteed byte-identical, key-that-destroys at
      R-6 scale. CORRECTED SPLIT: the cache key carries the SET OF REGISTERED
      CONTAINER NAMES (input-derived, decides refuse-vs-record, strict superset per
      the mandate); adapter IMPLEMENTATIONS move to a read-side staleness check,
      where per-record scoping is computable. OPEN, architecture's decision: keep or
      drop _declared's capture-time trial (the one place an implementation touches
      capture; envelope leans keep; the adversary judged it clean — removal would be
      a recorded decision, never a quiet revert). Proposed-unbuilt: containers_of(blob)
      header-only accessor — deliberately not written while its consumer scheme is
      undecided (a record produced but never read).
      FINAL RULINGS (architecture, verified envelope's call-site enumeration against
      the branch rather than accepting it): cache key gets the SET OF REGISTERED
      CONTAINER NAMES only — adapter implementations never enter it (their rebuild
      output never reaches stored bytes; a digest in the key = third campaign
      instance of key-that-destroys, and the first architecture ACCEPTED rather than
      caught). Root container: DERIVE — capture passes _dotted(type(value)), the
      declaration read off the object cannot lie, and it short-circuits _declared's
      trial at :484 so adapter implementations never affect the capture door,
      dissolving the keep-or-drop dilemma without a revert; the trial stays for
      callers declaring a foreign container (compat flattening a Face), where G8
      routes it. CONSTRAINTS adopted: an adapter must not require its producer's
      runtime importable (cold_replay enforces); try/except ImportError around a
      registration is BANNED (degrades to an empty registry — one machine stores a
      plain dict, another refuses, both think they succeeded).
- C8  READ-SIDE STALENESS (PLACED by lead ruling — invented in the C7 exchange,
      previously in no checklist): a stored record that recorded container X thaws
      differently today if X's adapter changed — stored bytes unchanged,
      reconstruction drifted. Mechanism: the adapter-module digest is a WRITE-TIME
      fact (the output exists then, unlike at key time) — record it in the envelope
      header at freeze; a read-side check compares against the current adapter and
      FLAGS (never refuses; refusal would make an adapter bugfix brick the library).
      Wire the flag into I3's reverify/runtime_observed family, not a new gate.
      Needs envelope's containers_of(blob) header-only accessor, which now has its
      consumer and may be written. Owner: architecture (mechanism) + envelope
      (accessor). NOT in the cache key, per the C7 ruling.
      GRANULARITY RULING (lead, on architecture's refinement): PER-CONTAINER —
      editing adapter B must not flag records that used only A (the muted-channel
      hazard: a mostly-false queue is a queue nobody works). The honest-mechanism
      objection is dissolved structurally: the registry is ONE ADAPTER PER FILE, so
      the per-adapter digest IS a file digest — the only honest mechanism available,
      at the right granularity. Header carries {container_name: adapter_file_digest};
      containers_of returns the name→digest mapping. A future adapter that cannot
      live alone in a file is a RECORDED exception carrying module granularity and
      its false-positive scope stated. inspect.getsource (whitespace-fragile) and
      declared revision strings (the I6 defect) are both rejected. Envelope may
      object on codec grounds before writing.
      UPGRADED FROM CHOSEN TO FORCED (envelope, measured): digesting the rebuild
      CALLABLE (co_code+co_consts+co_names) is structurally blind — changing
      InsightFaceRecord.sex 'M'/'F' → 'male'/'female' moved every thaw output with
      the digest UNMOVED, because for a class-based adapter the consumer-visible
      behaviour lives in the CLASS, not the callable. The file is the smallest unit
      reliably containing the behaviour; several-adapters-per-file would reopen a
      digest nobody can compute soundly. ADDITION (accepted): absent is not empty —
      freeze ALWAYS writes the container-digest field (empty when no adapter used);
      containers_of REFUSES BY NAME on a record lacking it (the version-2 refusal
      shape); safe within sgface3, no second magic bump, because no durable sgface3
      record exists anywhere (measured: production has no native column, zero native
      blobs tree-wide, corpus namespace hashes facestore).
      C8 CANDIDATE DECLARED (envelope, supersedes its measurement-test declaration —
      3 files +251/-49: register_container with revision, refusing second
      registrations differing in either half; freeze records {container: revision}
      under "adapters" by BOTH routes; containers_of header-only, {} = used-none,
      absent REFUSES by name; five isolating seed controls each turning exactly one
      test red). ADDITIVE, NO MAGIC BUMP — thaw never reads the field, only
      containers_of does, which is what lets it land before the capture package
      exists. RATIFIED INTERIM TRADE: InsightFaceRecord (still in facestore) gets a
      facestore-self-digest revision — over-broad (false positives) chosen over a
      literal (false negatives = silently-unflagged records, the fatal direction the
      blindness measurement proved); a noisy flag queue is recoverable under
      flag-never-refuse where an unflagged record is not. Resolves to the exact file
      digest when architecture moves the adapter out. Recorded as a TRADE MADE, not
      discovered. Known weakness declared to the adversary: the degenerate-record
      test reimplements the wire layout (second-writer shape, unavoidable while
      freeze always emits the field).
      TRADE REJECTED BY USER — the self-digest interim ("facestore digests its own
      file at import") is VETOED outright; the earlier lead ratification is
      withdrawn. Envelope then MEASURED the literal-marker dissolution's premise
      FALSE: a record carries NO codec digest (header keys printed: adapters,
      container, nodes, producer, producer_version, root; magic is a format version,
      not content); the codec identity lives in the cache KEY — input-derived, moves
      nothing already stored — so a facestore edit re-runs FUTURE captures while
      stored blobs sit in the column with changed thaw output and nothing to flag
      them (the corpus cache escapes only because its namespace is in the PATH; the
      native column is read by id). For an in-codec adapter the adapters field is
      the ONLY thing in the record that moves when behaviour moves. RESOLUTION
      (superseded in part by envelope's SECOND self-correction): that measurement
      was true of the WRONG OBJECT — the record is the ROW, and under S1/I2 a
      stored result carries its write identity, so codec coverage DOES detect
      in-codec adapter drift at the row. VETO IMPLEMENTED as an _IN_CODEC literal
      marker, facestore doing NO filesystem read (classified search; its first
      positive control failed and was redone against a known-present token),
      detection delegated to I1's codec-coverage term — recorded as a HARD
      REQUIREMENT in the I1 entry itself, per envelope's ask. The InsightFaceRecord
      move to the capture package STAYS ORDERED (C7 destination; dissolves the
      delegation entirely) at NORMAL C-block priority — the forced-NOW urgency
      rested on the falsified premise. MOVE IS ATOMIC-PLUS (envelope's constraint):
      registration move AND compat's read/write routed through the capture package
      in ONE change — the first half alone makes the compat lane silently
      non-caching via cache.py's except (defused to LOUD once the
      UnregisteredContainer split lands, but atomic remains the order). Marker is
      `unmoved:vision.facestore.InsightFaceRecord` — blind but SAYS SO in its own
      value (approved: declared-incomplete over silently-blind); attack D reads
      `unmoved:` as a REPORTED exception, never a pass — the check that notices if
      the move never lands. Envelope's twice-in-one-exchange pattern is
      self-named: correct measurement, unsupported inference — to be stated
      explicitly when it is doing it. Controls re-run on the new artifact, never
      carried over. Process note stands: noticing a problem and resolving it are
      different steps; the four-minute check would have prevented three messages.
      SIGNATURE CORRECTION (envelope, retracting its own twice-made no-change claim):
      register_container gains `revision: str` — the digest is SUPPLIED at
      registration by the layer that owns the files; facestore records what it is
      given, never computes it (computing would make the codec INFER the one-adapter-
      per-file policy and claim more precision than the layout has on a future
      module-granularity exception day — the InsightFaceRecord-in-facestore shape
      through a different door). LOAD-BEARING residual, stated: the channel permits a
      hand-written string, so the registry attack MUST compare each registered
      revision against its adapter file's actual digest, or I6 returns dressed as
      compliance. Awaiting architecture's ruling; then containers_of + the signature
      change go to the adversary as one candidate with the signature declared the
      load-bearing part.
      The register_container residual has TWO halves and the unnamed one is worse:
      READ (record written with adapter unreadable without it — loud, costs a rerun)
      and CAPTURE NON-DETERMINISM — identical producer output is STORED as a subclass
      node in a process with the adapter registered and REFUSED in one without, so
      capture behaviour depends on the import graph executed, not the code. Strikes
      the invariant directly. Bounded today (persisted records declare only dict or
      the shipped Face); the door opens at the second adapter. Proposal: one adapter
      module inside the capture package imported by its __init__, CONTENTS DERIVED
      from declared producer contracts; InsightFaceRecord moves out of facestore
      (producer knowledge in a module claiming to know no field names); adapter
      digest joins capture identity ONLY for records that recorded a container
      (unconditional inclusion = key-that-destroys-what-it-protects); entry points
      ruled out (invisible to G3 identity), lazy thaw-import ruled out (breaks cold
      replay). Four attacks specified, incl. capture determinism across two import
      graphs and a population check derived from contracts never from _ADAPTERS.

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
scratchpad/wave1/envelope-v1-v5-attack.md. FOURTH COMMIT 6ab494f (post-attack,
tests-only): the v2 refusal held by an attack — deleting the _SUPERSEDED branch turns
exactly one test red ("refuses by the version it names, not as an unknown format").
Envelope also SELF-CORRECTED its evidence: the nothing-reads-sgface2 claim had been
answered by CITING R-6/Z4 — rulings locate evidence, they are not evidence — and is
now MEASURED read-only (native column added v46→v47 at migrate.py:3922; production
v45 has no column; repo db 0 blobs; all 43 *.db files hold no non-null native;
**/*.sgface VALIDATED_EMPTY with a same-tool positive control), and the refusal
OBSERVED against a genuine v2 blob rebuilt from 1032760's encoder (v2 reads its own
bytes back; v3 refuses naming version 2; cache.py:142's except turns it into a miss —
the G8b shape holding).
PROVENANCE CORRECTION (adversary caught the lead's forwarding error): the paragraph
above describes envelope's REPORT, not the landed commit. 6ab494f's artifact is 16
lines — one test, two assertions against SYNTHETIC bytes — verified by the adversary
as a real negative control (emptying _SUPERSEDED turns exactly the version-2 match
red). The 43-db sweep, the **/*.sgface VALIDATED_EMPTY, and the rebuilt-v2-encoder
observation are SESSION PROSE, uncommitted — reports locate evidence, they are not
evidence. The load-bearing merge-safety claim (no sgface2 data exists to lose) does
NOT rest on that prose: production-v45-has-no-native-column is independently verified
twice (population census R-6; architecture's migration run) and repo-db-0-rows was
verified by the lead directly. DISPOSITION (envelope's choice, accepted): PROSE — the
genuine-blob probe would be a LONGER control, not a stronger one (thaw's magic branch
is the first statement and decides on blob[:8], so genuine and synthetic v2 blobs
exercise one path with one decision; the reorder hazard is covered because the test
matches the MESSAGE "version 2", which a digest-first reorder would break). And the
machine-fact half ("no stored v2 blob exists anywhere") CANNOT be a test — it asserts
a fact about a machine, not a property of the tree; it stays a re-runnable
measurement, never a gate. STATUS: 6ab494f CLEARED by the respawned adversary
(attacked against the commit's own tree via git show; both assertions verified,
negative control precise) — V1-V5 = PROVEN PENDING MERGE ONLY, full branch. One
narrow finding F27 (with envelope): a v3 blob with only its magic flipped to sgface2
is misreported "version 2, re-run the producer" while its digest disagrees — magic
sits inside the digested body and thaw decides on blob[:8] before computing the
digest, so a CORRUPTED record reads as intact-but-old, prescribing an R-6-scale
re-detect instead of naming corruption. Invariant not violated; diagnosis sign error.
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
- I1  HARD REQUIREMENT (from the C8 veto chain, must be visible here where I1 gets
      built): capture-codec-version MUST cover vision/facestore.py — the in-codec
      adapter's revision is a literal that does NOT move when behaviour moves;
      detection is DELEGATED to this term at the ROW level (a stored result carries
      the identity it was written under, S1/I2). If I1 ships without this coverage,
      records using the in-codec adapter go silently unflagged. Dissolves for that
      adapter when InsightFaceRecord moves to the capture package (ordered,
      C-block).
      COMPLETE, gate-clean, held as candidate 2 (architecture; capture/__init__.py,
      capture/contracts.py, one 16-attack test file; 831 passed): ProducerContract +
      ResultIdentity preimage. EVERY REVISION DERIVED, NEVER DECLARED — an
      implementation revision IS the digest of its declared files (answers
      model_version reading "scrfd10g+glintr100-v1" through every change; names
      hashed beside bytes so renames move identity). CACHE-WARM SPLIT MECHANICAL:
      preimage_of returns (preimage, observed); only DECLARED bit-affecting runtime
      facts reach the hash; declared-but-unobserved REFUSES (an absent distinguishing
      term fails open with every call agreeing). Weight PATH is provenance, not
      hashed (same weights under another models_dir must hit). C7's container name
      set is a first-class preimage field, empty until adapters exist. Controls:
      each fired on exactly ONE test (digest→declared-string swap; hash-everything
      swap); contradiction/rename/config/codec each move the preimage; two contracts
      cannot claim one name; population read, not remembered. ATTACKED (adversary,
      sharpest of the round): digest_of_files used name+NUL+bytes+NUL and file bytes
      can contain NUL — two different declared file sets produced ONE digest,
      reproduced with identical sha256s (fix: length-prefixed framing or per-file
      hashing); and `containers` defaulted to (), a well-formed WRONG cache key on
      omission — the ledger._read shape again; make it required. Weight-set
      collision vector did not open. HONEST LIMITATION
      declared in-contract: over_invalidates — both face backends live in
      vision/faces.py, so editing one moves the other's identity; the split
      (one-producer-per-file, same shape as the C8 ruling) is C-block/X1 work.
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
      negative half = a resolve a day later from another job still hits.
      ADVERSARY ATTACK: two vectors HOLD — three independent observation channels
      (calls, was_hit, envelope; strongest: store b"first answer", producer would now
      answer differently, hit returns the stored bytes — proves the store served
      without the counter); CONTRADICTED is implemented AND attacked (second answer
      under one identity raises with offered_envelope stored; same-bytes-twice is not
      a contradiction; no delete-to-clean; re-blessing is a row; waiver cannot
      pre-authorize a loosening). MERGE-ORDERING NOTE, not a finding: r1 carries
      sgface2 facestore, so a self-referential producer return raises bare
      RecursionError there — I2's fail-at-capture docstring promise becomes fully
      true when V1-V5 merges. POST-MERGE REGRESSION REQUIRED: a self-referential
      return raises Unpreservable naming the path THROUGH resolve. PROVEN pending
      merge + that regression.
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
      PINNED RUN (population, nonface worktree): corpus digest d51e1a11ecfe2e93…,
      128 pictures, 0 unhashed members; torch 2.13.0+cu132 / open_clip 3.3.0 / cuda
      13.2 / cudnn 92000 recorded in the artifact. max_abs 6.440356e-04 at batch 8,
      flat 6.437-6.440e-04 through 128; preprocess bit-identical at all worker
      counts. REPRODUCES the unpinned re-run to every digit — and the second
      independent run (separate process, separate working tree, same pinned corpus)
      is BIT-IDENTICAL at every batch width: determinism evidence n=2 — now n=3
      (post-fix pinned run, --expect-digest accepted, numbers exactly equal). DIGEST
      FACT (population's correction of the lead's respawn brief): the fixed
      bytes-hashing code still emits d51e1a11… because every member's disk bytes
      equal the recorded content_sha256 on this corpus — the VALUE held, the MEANING
      inverted; proven non-vacuous by two-directional tamper controls (DB-column
      tamper moves only the old function; byte tamper moves only the new, the old
      being blind since it never opened the file). ALSO: I6's own restatement
      enumeration missed benchmarks/openclip_retrieval.py:4 citing the refuted
      2.2e-03 — caught and restated in the same candidate. What was never
      reproducible was the corpus, which is exactly what the pin fixes. Still 3.4x
      below the old docstring's 2.2e-03. Acceptance:
      2 of 4 conditions hold (pinned, identity recorded); artifact tracking waits on
      I7; docstring restatement ordered from the pinned artifact.
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
- P4a (from the nonface candidate round, 4237ff8): compat/producers/census.py
      --check rebuilds and compares SITES against the shipped artifact (which now
      carries the tree's evidence identity) — the adversary's four-site drift
      reproduction is its standing negative control, observed failing then passing.
      DETECTABLE BUT UNASKED: no lane runs --check; wire it into compat.just's lane
      set (proof's file, P4 scope) so census drift reds closure. Population
      explicitly did not claim N1 closed without this.
- P5  Status generated from ProofRuns; doctrine artifact re-stamped with proof IDs (the
      artifact's hand-stamped table currently violates its own B18 rule).
- P6  ExporterBinding: upstream-compatible claims only with pinned upstream
      exporter/loader contract; else explicitly application-native; refusal attack. [B19]
- P7  The 17 generated artifacts no closure condition reads: each becomes read-by-a-gate or
      explicitly documented as informational. [shape 3]
      COMPANION F28 (adversary, enumerated not counted — and explicitly NOT the same
      set as P7's 17): of 31 generated artifacts, TEN carry an identity stamp but
      closure._one_tree compares exactly FOUR (ledger, cases, provenance, lanes) —
      seven artifacts pay the stamping cost with no reader; twenty-one carry no
      stamp at all, closure.json and condition_audit.json included. MITIGATION
      stated: per-lane control staleness is guarded indirectly by `every lane exited
      0`, so not an open hole today — a mechanism built and not load-bearing.
      ALSO RECORDED (author-half forwarded-claim): ce89bff's message claims "a
      malformed record is a verdict rather than a traceback" — true of closure,
      FALSE of the run (lanes.exits() coerces; ledger crashes on WOBBLE; and ledger/
      closure sit OUTSIDE the lane loop with no || guard under -euo pipefail, so a
      ledger crash aborts run and closure never executes — the concrete reachable
      crash for the post-loop staleness item). Fix rides F25 + the staleness fold.
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
