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

### Docstrings

A docstring states a public API contract: accepted inputs, return values, raised errors, side effects, invariants, and externally imposed constraints.

The rules above apply to docstrings. A docstring is not a place for implementation history, design essays, speculation, or commentary on earlier versions.

## Gates

`just check` and `just test` run on every commit through `lefthook.yml`. `just prove-push` runs on push. `just budget` holds each lane to the clock it was measured against.

A check no hook runs does not gate anything. Wire a new check into the hook, and prove it can fail.
