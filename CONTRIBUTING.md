# Contributing

## Comments

Comments record facts the code cannot state by itself.

Every comment must be:

- objective: verifiable from code, a test, a committed artifact, a reproducible command, a specification, or a pinned upstream source
- current: describes the code as it is, not as it was
- necessary: removing it would hide a constraint, invariant, external requirement, or non-obvious behaviour
- minimal: no more text than the fact needs

A comment block outside a docstring must not exceed two sentences or three physical lines. Consecutive comment lines are one block. Longer explanations belong in a test, an ADR, a documentation artifact, or an upstream reference.

A comment must not contain:

- opinions, recommendations, speculation, or guesses
- narrative, background, or design essays
- change history, dates, former behaviour, or migration notes
- future intent, plans, TODOs, or promises
- emphasis, verdict language, or persuasion
- benchmark numbers, counts, performance claims, or compatibility claims with no cited source
- restatements of what the code already shows

These words mark text that belongs elsewhere: probably, likely, should, hopefully, temporary, for now, used to, we decided.

An external fact names its source precisely enough to check:

```
org/repo@commit path
specification section
committed/artifact/path
reproducible command
```

A comment, docstring, README, or prose document is not evidence for a claim about code. Use the code, the runtime, a test, the specification, or the pinned upstream.

A comment that cannot be proven is shortened to the provable fact, or deleted.

### Why citing beats reading

One measurement existed in four files in three wrong versions. At 80,000 files `benchmarks/results/answer_currency.json` records `quiet_ms 0.179`, `answer_commit_ms 37.93`, `answer_factor 211.8`; `db/resultset.py`, `db/migrate.py` and `tests/test_an_answer_knows_when_it_is_stale.py` each said `0.18`, `38.26` and `214`. That last file also carried the 40,000-file row as `105.9` where the artifact records `98.4`.

Every one was found by trying to source the number and failing. None was found by reading the prose and doubting it — each reads as confident and correct, and three of them agreed with each other. That is the argument for citing rather than for writing carefully: careful reading was never going to find them.

### Docstrings

A docstring states a public API contract: accepted inputs, return values, raised errors, side effects, invariants, and externally imposed constraints.

The rules above apply to docstrings. A docstring is not a place for implementation history, design essays, speculation, or commentary on earlier versions.

## Gates

`just check` and `just test` run on every commit through `lefthook.yml`. `just prove-push` runs on push. `just budget` holds each lane to the clock it was measured against.

A check no hook runs does not gate anything. Wire a new check into the hook, and prove it can fail.

### A test that passes alone and fails in parallel

That is evidence of a race, not evidence against a defect. Serial execution beating a race is what a race looks like from outside.

`test_every_picture_carries_the_correction` failed five pushes with `assert 1 == 3` and passed alone in 2.92s each time. It waited for `[data-person-pictures]`, the container, then counted `[data-person-picture]`, the shells that arrive inside it. The count read whatever had rendered. `expect(...).to_have_count()` retries, and was already on the next line for a different locator.

Wait for the thing being asserted, not for its container. Reading does not catch this one either: the prose is right, the code is right, and only the ordering is wrong.
