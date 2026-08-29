# Contributing

## Comments

This repository keeps a lot of reasoning in comments. That is deliberate and
worth keeping, but it only works while every sentence in them is true. A
comment nobody can check is indistinguishable from one that is wrong, and the
cost is not theoretical: a docstring asserting "there is no test here that
fits" is what marked the entire suite `slow` and left `just test` selecting
zero tests behind a green exit code.

### Silence beats a wrong comment

If a claim cannot be checked, delete it. Do not annotate it, do not soften it,
do not leave the number with a note saying not to trust the number. A comment
that apologises for itself is still a comment a reader has to evaluate.

The only thing worse than no comment is a confident one that is false.

### Every number is readable off something

A figure in a comment must be one of:

- read off a committed artifact, cited by path — `benchmarks/results/openclip_batch.json`
- reproducible by a command written in the comment — `python -X importtime -c "from PIL import Image"`
- measurable from the tree — a file size, a byte count, a row count

If it is none of those, it is folklore. Delete it, or go and measure it and
cite what you measured.

Prefer citing the artifact over restating its contents. Numbers copied into
prose go stale silently; a path stays correct as the file is regenerated.

### Citations must resolve

    upstream     org/repo@sha path:line      evilmartians/lefthook@8d9cfec internal/run/controller/controller.go
    in-repo      path:line                   compat/harness/answer.py:100-106
    artifact     path                        benchmarks/results/face_pipeline_validation.json

Check the reference before you write it. A citation that does not resolve is
worse than no citation, because it buys unearned confidence: this repository
has shipped a comment citing `benchmarks/face_embedder_ab.py`, which does not
exist, and two citing a corpus commit that is not in the corpus.

### History belongs in git

Git already stores what changed, when, and who did it, and stores it better
than prose can. Commit messages are the right place for "this used to do X and
here is why it changed".

    in a comment       why this value, what invariant holds, how to check it
    in a commit        what changed, what was wrong before, what evidence moved

So: no `until 2026-08-29`, no `this used to be`, no dates marking when a
measurement was taken unless the measurement itself is dated evidence.

Rationale is not history. "2 trades size for throughput" explains a live
constant and stays. "This was 4 until we measured it" is `git blame` and goes.

### Plain, objective, standard nomenclature

Write what is true in the fewest words that stay accurate. No capitals for
emphasis, no rhetorical intensity, no verdict words. Use the standard name for
the thing — `execution provider`, `fixture closure`, `collection hook` — not a
coined phrase a reader has to decode.

Say what a reader must know to change the code safely, and stop.

### Comments and docs are not evidence

When you need to establish something, go to the source, the runtime, git, or
the pinned upstream. Never derive a claim from another comment, a docstring, or
a file under `docs/`. Those are written by the same fallible process as the
code and have been wrong here before, including in ways that survived for days
because they read as settled fact.

This applies to comments you are editing. Verify before you preserve.

### Prefer a generated count to an asserted one

A comment that says "eighteen corruptions" and a list that holds nineteen will
diverge, and nothing will notice. If a count matters, assert it in a test
against the thing being counted.

## Gates

`just check` and `just test` run on every commit through `lefthook.yml`;
`just prove-push` runs on push. `just budget` holds each lane to its own
measured clock. A check that exists but no hook reaches is not a check — if
you add one, wire it into the hook, and prove it can fail.
