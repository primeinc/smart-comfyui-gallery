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
against a declared-closed family list — currently: unreachable member, empty population,
inconsistent inputs); statistic trust (a count is not an enumeration — absence claims
rest on classified searches with positive controls or printed enumerations, never
counts).

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
- OPEN: location of the production library database (the repo's gallery.db and all 40
  backups are measured empty; re-detect cost is stated when the production DB is found,
  before any run).

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
