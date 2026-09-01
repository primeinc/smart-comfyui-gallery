# The Capture Monopoly — governing scope

This document is the durable source of truth for the capture-monopoly refactor. The
session task list coordinates; THIS FILE binds. A task may be added here when discovery
finds a new surface. A task may not disappear from here without an explicit recorded
ruling in the Rulings section below, naming who ruled and why.

## Governing invariant, verbatim

> If it is output by any producer, store it as whatever it actually is, without semantic
> narrowing, and prove that using the stored result is the same as doing it live, without
> cheating.

## Acceptance rule

The refactor is not done while any production code, or the STORED branch of any proof,
can obtain, transform, filter, derive from, persist, replay, export, cache, or compare
producer output except through the canonical ProducerResult. Raw producer output may
exist only inside the capture runtime and the isolated LIVE-baseline branch of a proof
run, and it may not escape either boundary. Every declared producer invocation has
exactly one path out: complete raw return → canonical ProducerResult → durable commit.
Batch outputs are captured as whole invocations; projections carry canonical lineage;
failed projections cannot roll capture back; identical result identities cannot silently
disagree.

## Task state semantics

NOT STARTED / IMPLEMENTING / IMPLEMENTED / PROVEN / CONTRADICTED / BLOCKED.
IMPLEMENTED is not PROVEN. A task is PROVEN when its standing attack exists, runs
through a gate that can fail, and has been observed failing on the defect it guards
(negative control). A parent is not PROVEN while any required child is not PROVEN.
The shared session task list's pending/in_progress/completed is coordination state only.

## Review lens

Three defect shapes cover every defect found in five review rounds:

1. A check that cannot fail — two flavors needing different techniques: a check written
   wrong (found by reading what exists and asking whether it can fail) and a chain that
   stops being specified one step early, which nobody wrote at all (found only by walking
   each mechanism to its end).
2. A key that destroys what it protects.
3. A record produced but never read.

Three reviewer-practice failures, each with a forcing artifact rather than prose:
enumeration misses (populations derived from code, never remembered lists — the G4
method constraint, with the boundary rule: every field traced to its writer AND to
whatever validates the writer's output); generalization misses (the G4 family column
against a declared-closed family list — currently four: unreachable member, empty
population, inconsistent inputs, self-referential control); statistic trust (a count is
not an enumeration — absence claims rest on classified searches with positive controls
or printed enumerations, never counts).

Fourth family, recorded 2026-09-01 (adversary, G1–G5 re-attack): SELF-REFERENTIAL
CONTROL — the control is derived from the artifact it guards, so it passes at every
value of the thing it constrains. Discriminator: could the control fail if the
implementation and the control were wrong in the same direction? If the control is
written from the implementation's own list or constant, no. Instances at recording:
identity_attack.COVERED["gates"] copied from GATE_GLOBS (G3); closure_attack building
ALLOWANCE+1 rows from the allowance it guards (G6); neighbour: G1's recorded lane set
judged against itself. Third direction (F9): a control frozen as an ARTIFACT rather
than a CLAIM drifts out from under the claim while the claim keeps passing.

Fifth family, recorded 2026-09-01 (adversary F11): PRESENT BUT UNCLASSIFIABLE — a
denylist condition enumerates bad values and admits every malformed shape it never
imagined (fifty ablations carrying verdict "WOBBLE"/None/absent/"" pass "every
ablation concluded"), while the allowlist form beside it (weight_is_verified, == the
one good state) rejects them all — and the polarity lesson was already written down in
weight_is_verified's own comment, learned in one place and not carried to code written
beside it in the same phase. Structural blind spot: G4's audit vocabulary has one
degenerate word (empty/absent) and cannot say MALFORMED, so G4+G6 jointly certify a
condition as sound over unclassifiable members. Rules: conditions are ALLOWLISTS
(anything not the known-good state is an offence); every population audit injects a
present-but-unclassifiable member as its second degenerate shape.

Recorded 2026-09-01 (user correction): THE CURRENT TREE IS THE DEFENDANT, NOT THE
AUTHORITY — a fix is never patterned on the mechanism under indictment. The .gitignore
benchmark allowlist is the canonical instance: extending it would have repeated the
expected-set defect it embodies. Existing code may serve as a reference pattern only
after being independently verified to hold (e.g. provenance.weight_identity iterating
the DECLARED set, verified 13-of-13).

SINGLE-WRITER RULE (recorded 2026-09-01, after three tree-blocks in one session): the
commit gate walks the whole tree, so any agent's in-flight file reddens every commit.
Until Phase G's commits land, proof is the ONLY writer in the shared checkout;
architecture (capture-monopoly-r1), population (capture-monopoly-nonface), and envelope
(capture-monopoly-envelope) work in their own worktrees, merged by the lead at
checkpoints. Commits use explicit pathspec (`git commit -- <paths>`); a gate-rejected
commit leaves the index staged — unstage after any failure.

Shared machine-local state worktrees do not isolate (recorded 2026-09-01; AMENDED —
architecture's correction: db/build.py:22 makes DEFAULT module-relative, so worktrees
DO isolate gallery.db; the v48 contamination was its pre-move cwd, a one-time event,
not a standing leak): node_modules —
CAUSE FOUND for the openapi-fetch gate failure: architecture had junctioned the r1
worktree's node_modules to the main checkout's, and its `npm ci` followed the junction
and emptied main's (disclosed, repaired, byte-identical rebuild verified). RULE: never
junction/symlink a directory an installer owns between worktrees; each worktree gets
its own node_modules. The .venv junction stands (python embeds no paths). Retry-first
guidance stands for residue of this class — but the same failure twice is not a race:
a race repairs itself; a deterministic deletion only looks transient because someone
repaired it. Ask the junction question on the second occurrence, not the third.

Seams between two owners are where flavour-2 lives (recorded 2026-09-01, envelope's
finding at 6684ba7): a question can go unasked not because either owner's checks are
incomplete for their own scope but precisely BECAUSE each is complete for its own
scope — a different generator from the lens's others, which are about how a check is
written; this one is about who is looking. At every ownership boundary, ask what
property crosses it and whose test reads it.

Repair the selector, not the site (recorded 2026-09-01, adversary's generator,
self-refuted from "uncarried repair is general" to this bounded positive form by
testing it): a pointwise repair leaves the same defect at sibling sites even when its
own comment states the general lesson — six reproduced instances F11/F23/F25/F33/
F39/F41, all in one cluster of Phase-G modules. The structural repair changes the
MECHANISM that selects the sites: S2r starting the guard check FROM table_class
covers every declared member by construction (measured: eight of eight tables, both
directions, none scoped), and G3r's exclusion form for root files is the same move —
F33 is what happens when that author applies it to files but not directories.
Discriminator at review time, until the fix is structural: what else reads this
field, has this shape, or was written beside this in the same change? Fix-in-place
re-declarations carry the discriminator's answer as part of the claim.

Dilemmas dissolve before they trade (recorded 2026-09-01, architecture's observation
across three instances — X1 closing capture-determinism, derive dissolving the trial
question, one-adapter-per-file making the honest digest exist at the wanted
granularity): ask what would have to be true rather than which horn to take. And its
companion honesty: verifying the right thing without having specified it is not the
same as having got it right.

Approving a proposal is not reviewing it (recorded 2026-09-01, architecture, after
accepting an uncomputable scoping it had the facts to refute): the reviewer who
already agreed is the one least likely to look; walked-to-the-end review applies to
proposals you endorsed, doubly.

Verify with the instrument that gates (recorded 2026-09-01, proof, third instance in
one session — `just compat check` omits sglint, twice; ruff run from .venv-compat while
the gate runs .venv): a green check from a different tool is not evidence about the
gate. Companion to "a count is not an enumeration".

NO SILENT EXCEPT — USER RULING (recorded 2026-09-01, verbatim intent: "why is anyone
swallowing output and or errors at all"): no except clause anywhere may swallow an
error into silence. Every except either (a) propagates, (b) converts to a TYPED
refusal the caller must handle, or (c) tolerates a narrow expected condition WITH THE
OCCURRENCE RECORDED somewhere something reads — a cache may miss on a corrupt file,
but the miss-on-error is counted and surfaced, never invisible. Tolerating an error
and hiding it are independent choices; only the first is ever permitted. ENFORCEMENT
IS A GATE, not prose: a new sglint rule flags any except clause that neither
re-raises, nor raises a typed refusal, nor records — with the existing swallows
enumerated and each converted or explicitly waived by recorded ruling. The G8b
miss-preferable-to-crash decision stands ONLY in its recorded form. Known mirror
(envelope): the READ side — _restore raises ValueError for a record naming an
adapter this build lacks, and cache _read eats ValueError — covered by the same
sweep, both directions of the boundary.

Declaration-from-diff rule (recorded 2026-09-01, adversary, after THREE of four
candidate declarations in one round stated a scope the worktree did not match —
declaring intent while the tree holds everything in progress): under candidate-first
the DECLARATION is the artifact, and a verdict on a mismatched declaration authorizes
a commit the attacker never saw. Declare from `git diff --stat`, never memory, and
STAGE the candidate so the declared set and the committable set are one set —
staging is the stronger half (envelope): memory can be wrong about a diff, but the
index cannot be wrong about itself, and an EMPTY unstaged diff makes the gate's green
a statement about what will commit rather than about something adjacent.
Amendment (2026-09-01, adversary, live instance in r1): the diff commands cannot see
UNTRACKED files, so a brand-new module — most of this campaign's work — is invisible
to the prescribed command, and a pathspec commit drops it the same way. Declare from
`git diff --cached --stat` PLUS `git ls-files --others --exclude-standard`, and the
untracked list must be empty OR declared by name as non-candidate (next-work files
disjoint from the frozen set — the pre-build mechanic stays legal, it just stops
being invisible). Second amendment (2026-09-01, adversary, observed twice under the
two-reviewer split): a re-declaration after a finding goes to BOTH reviewers, not
only the one whose finding forced it — otherwise the other reviewer's held
declaration silently stops matching the commit, and "probably the other reviewer's
fix" replaces knowing, which is what the rule exists to remove.

Forwarded-claim rule (recorded 2026-09-01, adversary, after TWO instances of the
lead's forwarding recording a mechanism the commit does not contain — envelope's
"genuine v2 blob" and architecture's "git-held schema.sql at 1032760"; both times the
work was sound and the summary claimed more): a forwarded claim about HOW something
was verified is an assertion and is checked against the diff before it is recorded.
The summary channel is not exempt from the evidence standard the code is held to.
EXTENDED (architecture's own request, after repeating the shape inside cec2860's
commit message — "two independent controls" where only one is in the diff): the rule
binds the AUTHOR's commit messages and reports too, not only the lead's forwarding. A
one-off interactive verification that left no artifact is stated as exactly that.

Probe-narration rule (recorded 2026-09-01, adversary self-catch on the census attack —
its draft narration asserted conclusions its own output contradicted): a probe's
narration is an assertion and must be written after its output, never before.

Two operational rules, recorded 2026-09-01 (adversary F5): (1) a harness that captures
identity() once and then calls anything recomputing it live is order-dependent on
concurrent commits — hermetic harnesses thread ONE captured identity end to end; (2) an
alarm is not a diagnosis — a red number gets its cause established before it is
reported, exactly as a green one does (companion to "a count is not an enumeration";
this session's fourth number that meant something other than what it said, and the
first alarming one).

## User mandate

Caching maintained or enhanced: a ResultIdentity hit skips producer execution entirely;
every replacement cache key is a strict superset of the key it replaces.

## Bypass registry

Doctrine bypasses B1–B24, all confirmed against source in Wave 1 except where noted:

| id | bypass | status |
|----|--------|--------|
| B1 | filter-before-capture (six gates, two inside producer configuration) | confirmed |
| B2 | per-face rows, not per-invocation capture | confirmed |
| B3 | canonical welded to derived_face_instance | confirmed |
| B4 | freshness by name-strings — worse: face freshness has NO model term | confirmed |
| B5 | consumer proofs replay from live memory (12 of 13 runners) | confirmed |
| B6 | retained/rebuilt policy residue | REFUTED — clean |
| B7 | ledger stage laundering (one Cell aliased into six stages) | confirmed |
| B8 | failed-lane laundering (closure never reads lane exits) | demonstrated |
| B9 | population = manifest list | confirmed |
| B10 | wrapper boundary laundering (ladder; winning rung also lost) | confirmed |
| B11 | non-mapping envelope roots | CLOSED (any-root freeze/thaw, 672ef45) |
| B12 | native semantics (aliasing lost, nested subclasses widened, container unrebuilt, cycles raise wrong class) | partial |
| B13 | verifier normalization + tolerance defaults; None-keys invisible at capture | confirmed |
| B14 | live-object leakage (projections from live objects; demonstrated 9.0/7.0 row) | confirmed |
| B15 | incomplete input identity | confirmed |
| B16 | LIVE/STORED contamination | confirmed |
| B17 | evidence identity omits schema/gates/deps | confirmed |
| B18 | hand-authored status in-tree | REFUTED — clean (the doctrine artifact re-stamp remains R2 work) |
| B19 | export-format laundering | partially refuted — reactor's pinned contract is genuine and unique |
| B20 | capture rollback — sharpened: per-producer, one producer's failure destroys another's output | confirmed |
| B21 | batch-output evaporation (_Ahead/_Said) | confirmed |
| B22 | non-face producers bypass any canonical store | confirmed |
| B23 | canonical-as-derived deletion (drop_all) | confirmed |
| B24 | same-identity overwrite | confirmed |

Wave-1 additions (full detail in the adversarial and census reports; checklist items in
docs/capture-monopoly-tasks.md): scan-row laundering of drops into faces=0 (A2);
promoted-column/record divergence (A3); ladder rung and empty answers unrecorded
(A4/A6); padded recovery stored nowhere, app or harness (A5); det_thresh never passed —
docstring claims behavior the code does not implement (A7); batch sha misattribution
(D2); embedding/caption narrowing at the schema (B2/B3-adv); prompts.remember silent
vector drop; corpus-cache codec absent from its own cache key with an unverifying read
(E12) and a mispositioned except (E13); consumption-graph inconsistency reaching GREEN
(E11); weight conditions vacuous on empty sets (E9); ablation verdicts reaching no gate
(E2); EXIF homeless/unrecorded values dropped (narrowing under any definition); a
third-party producer (ReActorFaceAnalysis) in no census.

## Producer population

Application producers (rulings 0–4 below): InsightFace faces (SCRFD+glintr100 composite
ladder), OpenCV faces (YuNet + arcface composite), BLIP captions, OpenCLIP embeddings,
Qwen3-VL embeddings, perceptual hashes (imagehash — ruled in). Preprocessing
(identity-bearing, not producers): rawpy/LibRaw decode, libvips derive, thumbs, EXIF
parse (its dropped values are a tracked narrowing finding regardless). Proof-lane LIVE
invocations (legitimate, enumerated, boundary-policed): compat/producers/*,
compat/consumers/* (11 sites), compat/vendor/acceptance.py (10 sites incl.
ReActorFaceAnalysis), compat/corpus/loaded.py. The mechanically discovered population
is authoritative at closure; this list is its seed, not its bound.

## Rulings in force

Rulings are lead-proposed defaults presented at the Wave-1 checkpoint; the checkpoint
reviewer directed execution ("make the scope durable first, then let the team execute
from it") without objecting to them. Any ruling is revisable only by a recorded entry
here.

- R-0 Producer definition: a producer is any boundary whose output derives from learned
  weights or an external model runtime. Deterministic transforms are preprocessing whose
  identity (library version, params) joins ResultIdentity preimages; their narrowing
  surfaces are tracked as fidelity items.
- R-1 Decoders (rawpy/libvips/thumbs): not producers under R-0.
- R-2 EXIF: not a producer; its dropped-value lists are an unconditional narrowing
  finding, lowest tier, enveloped after R1 core.
- R-3 padded_recovery: capture both recoveries in the app path.
- R-4 imagehash: producer, in scope. Load-time probe passes: recorded as invocation
  facts. Query vectors: captured.
- R-5 Execution: Phase G (nine gates + G8b) precedes R1; then R1 (runtime + nonface
  owners), R2, R3 hostile closure. Enforcement hooks arm for implementation tasks.
- R-6 (closes the former OPEN item) Production library database LOCATED at
  C:/Users/will/.smartgallery/gallery.db — 184 MB, schema v45 (no `native` column),
  read-only census 2026-09-01 (population): 3,748 files (3,737 present images, 12.74 GB,
  14.5 GP), 3,514 face instances (3,088 opencv + 426 insightface), 11,031 embeddings
  (7,283 qwen + 3,748 openclip), 3,852 BLIP captions, and human-authored rows that MUST
  survive any migration: 72 person_assertions, 431 persons/clusters, 1,457 memberships,
  ratings/favourites/collections/stories. Migration cost is NOT zero: v45→v48 leaves all
  3,514 faces without canonical records (v46/v47 add columns with no backfill), and
  canonical recovery means re-detecting 3,737 images plus re-embedding 11,031 vectors and
  re-captioning 3,852 files if those gain canonical capture. Z4's zero-cost measurement
  was of the repo fixture and does not generalize.
  USER RULING 2026-09-01: C:/Users/will/.smartgallery/gallery.db is a TEST gallery,
  "just like the one in this cwd" — its contents (including the 72 person_assertions)
  are test data, not irreplaceable user data. The census above stands as a measurement
  of that test gallery; the S8 drop_all standing attack still uses it as the realistic
  fixture. A production library database remains UNLOCATED; Z4's state-the-cost-before-
  any-run requirement re-attaches to whatever production DB is ever identified, and the
  compute cost of any large re-detect run is still stated before starting one, as cost.

## CLOSED criteria

CLOSED requires all of: the mechanically discovered producer/consumer/variant population
agrees exactly with registered contracts or carries explicit user-approved exclusions;
every standing bypass attack and every mandatory proof lane green on the current-tree
identity; every ledger stage backed by its own stage-specific evidence; cold STORED
branches reproduce consumer artifacts without producer/live-object leakage; every proof
artifact bound to the exact git commit/tree, schema DDL, and gate files being shipped,
with consumption-graph consistency across artifacts; the verifier attacked once
deliberately and observed going RED; no condition held over an empty population; no
unwaived contradiction, with waivers append-only, attributed, and per-declaration.
Verdict vocabulary: CLOSED / NOT CLOSED / BLOCKED, issued once, by the lead, after the
adversary's hostile closure attack.

## Master checklist

docs/capture-monopoly-tasks.md — the frozen Wave-1 master task list (phases G, R1, R2,
R3 with per-item acceptance and standing attacks). Its DECISIONS section is superseded
by the Rulings above. Current engineering state of every item: NOT STARTED, except B11
(any-root envelope) IMPLEMENTED at 672ef45 — not PROVEN until its standing attacks run
through a failing-capable gate.

## Wave-1 provenance

Four independent teammate reports plus a cross-review (five rounds, sixteen
architecture-side defects closed, two proof counter-proposals withdrawn, three
adversary/census self-corrections) produced the checklist. The reports are session
artifacts; every load-bearing claim they made that this document relies on was verified
against source with file:line during reconciliation, and the checklist carries those
citations inline.
