"""OmniQuery v2: local natural-language querying for SmartGallery.

Pipeline:

    natural language
        -> search parser (deterministic nlq, which ALWAYS answers: rules
           consume recognized structure, every leftover term becomes a
           universal full-text condition; a grammar-constrained local
           nl2sql model may refine structurally-ambiguous phrasing), all
           paths emitting the SAME typed AST
        -> schema + semantic + authorization + complexity validation
        -> deterministic parameterized read-only SQLite SELECT compiler
        -> execution on a read-only connection with a SQLite authorizer

Invariants:
  - No model ever emits SQL. Models emit the typed AST defined in ast.py and
    nothing else; the deterministic compiler is the only component that
    produces SQL, and it only accepts ASTs that passed validation.
  - Validation and authorization happen OUTSIDE the model, in plain code.
  - All literal values are bound as SQLite parameters, never interpolated.
  - A model-produced AST replaces the deterministic parse only when it
    passes coverage_guard at full coverage (no dropped literals/keywords),
    measured on the benchmark corpus (omniquery/benchmark/), not assumed.
"""

AST_VERSION = 1  # wire-format version parsers must emit; mirrored in ast.py, which rejects any other
