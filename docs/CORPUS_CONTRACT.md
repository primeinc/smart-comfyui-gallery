# Corpus execution contract

The mission for the production sample gallery corpus, and the rules that decide
when it is finished. Recovered from transcript, not paraphrased from memory.

## Provenance of the requirement

Session `d6935ca1-5499-487f-8e4d-8395ab950233`, 2026-08-25. Verbatim user
messages, in order:

| line | time (UTC) | message |
|---|---|---|
| L30780 | 14:05:40 | `can you recraft a specific corpus of dervived sample pictures that better distributes both gen and non generated of huge ranages with or without real pictures / media mixed in` |
| L31037 | 14:25:45 | `... get @"claude-skills:adversarial-oracle (agent)" to review our sample corpus and tell use its defects we need to feature freeze and you need to tell me what we are missing in terms of a valid well defined sample corpus. our sample corpus should be good enough to post as a dataset on HF` |
| L31386 | 15:18:18 | `the corpus should be made up of real coverage needed + real life metadata shapes that are dirty and changed over time` / `does fabricated camera bodies change anything for anyone in any codebase at any time ever?` |
| L31429 | 15:20:51 | `but we dont need to synth half of this use \`hf\` cli and get real datasets and be a discoverer of wealth` |
| L31524 | 15:24:30 | `in my mind its fair game unless it has a restrictive license -- ... im worried about api backoff that may or maynot be needed over such an unbounded useless search for licenses` |
| L31551 | 15:25:33 | `we in our readme can publish a link to a collection of the og datasets we used to create our testing sample corpus no?` |
| L31565 | 15:26:53 | `so go back to the fruit OG search for things we actually need regardless of easy to fake or not and make a list and test all assumption including downloading whatever you need -- try to stay 50 gb and if thats unreasaonable ill give you a scratch drive` |
| L31662 | 15:30:37 | `my quest is getting a full sample corpus dataset to test against` |
| L32193 | 18:42:26 | `i told you i wanted a complete fucking sample corpus those things are fucking media files where are they` |
| L32518 | 18:54:26 | `go do it then, download the actual files` |

## SOURCE INVARIANT

The mission is to produce the requested full sample-gallery corpus:

- actual media files on disk
- generated and non-generated media mixed together
- broad real-world media diversity
- real dirty metadata shapes, including historical/writer variation
- real public source datasets used wherever reasonably available
- candidate assumptions tested by downloading and inspecting actual bytes
- provenance retained to the original source datasets/files
- approximately <=50 GB unless coverage evidence justifies more
- resulting corpus coherent and documented enough to publish as a Hugging Face dataset
- corpus defects adversarially reviewed and closed before feature freeze

The mission is the physical corpus.
Everything else is subordinate evidence or tooling.

## GOAL-SUBSTITUTION BAN

None of the following constitutes completion:

- a design
- a strategy
- a taxonomy
- a manifest
- a fixture generator
- a test suite
- passing tests
- an adapter audit
- format research
- a source list
- downloaded archives that were never inspected
- synthetic stand-ins for obtainable real-world diversity
- documentation describing media that is not actually present

Useful subordinate work is allowed.
It never inherits the completion status of the corpus.

## SOURCE HIERARCHY

1. This corpus contract and the recovered user requirement.
2. Actual behavior/data surfaces consumed by smart-comfyui-gallery, used to
   discover coverage needs.
3. Real media and authoritative upstream writer/spec/test evidence.
4. Existing tests/documentation.

Lower levels may reveal new requirements or defects.
They may not narrow level 1.

## MONOTONIC COVERAGE RULE

Maintain a durable coverage ledger.

Once a coverage need has been evidenced, it may only transition through:

```text
DISCOVERED -> UNSATISFIED -> PARTIAL -> SATISFIED
```

A row may also become `BLOCKED_EXTERNALLY`, only when an external constraint is
evidenced: no obtainable source, restrictive redistribution/access terms,
unavailable hardware/input, genuinely unsupported external format dependency.

The following are NOT valid terminal states:

- DEFERRED
- TODO
- SKIPPED
- NOT WORTH IT
- TESTS PASS WITHOUT IT
- SYNTHESIZED INSTEAD
- PROJECT DECIDED NOT TO SUPPORT IT

No need may disappear from the ledger without an explicit trace showing why the
original requirement itself no longer applies.

## REALITY-PRESERVATION RULE

When a real file exposes an application failure, surprising metadata,
unsupported shape, parser error, or bad assumption:

- preserve the evidence
- preserve the coverage requirement
- record the application/corpus finding
- continue acquisition when possible

Do not respond by:

- deleting the requirement
- changing expected results
- weakening an invariant
- adding an allowlist
- reducing the claimed coverage surface
- replacing the evidence with an easier synthetic sample

Evidence that makes completion harder makes completion harder.
It does not authorize redefining completion.

## REAL / GENERATED / MUTATED MATERIAL

These are acquisition methods, not mission categories.

**Real sourced files** — use when real producer diversity, metadata, historical
variation, encoding behavior, or naturally occurring messiness is relevant.

**Generated files** — allowed when generated media itself is the thing under
test, or when authoritative writer/spec behavior can be reproduced
deterministically.

**Mutated files** — allowed for deliberate malformed/error-path coverage.

A generated or mutated artifact must be labeled as such.

Synthetic material cannot satisfy a requirement whose purpose is real-world
producer diversity merely because it parses successfully.

## PROVENANCE

Every included corpus file must be traceable through durable metadata
sufficient to answer:

- Where did this file come from?
- Is it original, derived, generated, or mutated?
- What source dataset/project produced it?
- What original file/revision can be identified?
- Why is it in this corpus?
- Which coverage need(s) does it satisfy?
- What is its SHA-256 and byte size?

The exact ledger format is implementation choice. The information is mandatory.

## CANDIDATE VS CORPUS

Discovery candidates are not automatically corpus members.

A candidate may be rejected for an evidenced reason: duplicate behavioral
coverage, unusable/restrictive licensing, corrupt source unrelated to desired
corruption coverage, excessive size with no additional signal, wrong content
after inspection.

Candidate rejection does NOT remove the underlying coverage need.

## APPLICATION CHANGES

Finding an application defect does not authorize silently changing application
behavior while assembling the corpus.

If application repair is already independently authorized, keep the two state
transitions separate:

```text
corpus evidence  -> application defect
application defect -> explicit repair + validation against unchanged corpus evidence
```

Never modify the corpus merely to make the repaired application pass.

## PROGRESS PROOF

A progress claim must be tied to changed observable state. Report:

- corpus root
- actual media-file count
- actual bytes
- newly acquired sources
- newly included files
- coverage SATISFIED / PARTIAL / UNSATISFIED / BLOCKED_EXTERNALLY
- unresolved findings
- provenance/hash ledger status

Research narration alone is not progress toward physical corpus completion.

## DONE

Do not claim completion until all of these hold:

1. A physical corpus root containing the actual selected media exists.
2. Every included file has identity/provenance sufficient to reproduce or trace it.
3. Every evidenced coverage need is SATISFIED by named actual corpus files, or
   BLOCKED_EXTERNALLY with direct evidence.
4. The corpus demonstrably contains the requested generated/non-generated and
   real-world mixture.
5. Dirty/historical producer metadata is represented by actual files rather than
   fabricated field values wherever real examples are reasonably obtainable.
6. Source attribution is sufficient to link/document the original corpora used.
7. Size is approximately <=50 GB, or exceeding that limit is justified by named
   coverage requirements.
8. The actual gallery ingestion/runtime path has been exercised against the
   assembled corpus. Application failures are recorded as failures; they do not
   invalidate or erase valid corpus evidence.
9. The resulting package has the metadata/documentation/provenance needed to be
   reasonably publishable as a Hugging Face dataset.
10. An adversarial corpus review has compared the PHYSICAL FILE SET against the
    coverage ledger and this contract.
11. That review has produced no unresolved UNSATISFIED or PARTIAL requirement.

Only then is

```text
PRODUCTION SAMPLE GALLERY CORPUS COMPLETE
```

a valid statement.

## FINAL SELF-CHECK

Before every completion attempt, answer mechanically:

- Did I create more representative media, or merely more words/code/tests?
- Does every satisfied requirement point to actual files?
- Did any requirement disappear because it was inconvenient?
- Did a synthetic example replace obtainable real-world evidence?
- Did I change the application/test contract instead of preserving the corpus evidence?
- Can every included file be traced to why and where it exists?
- Could a third party receive this corpus and understand/reproduce its provenance?
- Would the Hugging Face publication claim survive inspection of the actual directory?

Any "no" that violates the contract blocks completion.

---

> If you discover evidence that makes completion harder, the permitted response
> is to add evidence, files, or an unresolved gap. You are never permitted to
> make the requested world smaller so that completion becomes easier.
