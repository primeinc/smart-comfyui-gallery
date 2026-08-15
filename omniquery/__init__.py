"""OmniQuery v2: local natural-language querying for SmartGallery.

Pipeline:

    natural language
        -> intent parser (Needle2 primary / heuristic baseline / one optional
           constrained fallback model), all emitting the SAME typed AST
        -> schema + semantic + authorization + complexity validation
        -> deterministic parameterized read-only SQLite SELECT compiler
        -> execution on a read-only connection with a SQLite authorizer

Invariants:
  - No model ever emits SQL. Models emit the typed AST defined in ast.py and
    nothing else; the deterministic compiler is the only component that
    produces SQL, and it only accepts ASTs that passed validation.
  - Validation and authorization happen OUTSIDE the model, in plain code.
  - All literal values are bound as SQLite parameters, never interpolated.
  - Parser confidence is a routing input only; thresholds are calibrated on
    the SmartGallery benchmark corpus (omniquery/benchmark/), not assumed.
"""

AST_VERSION = 1  # wire-format version parsers must emit; mirrored in ast.py, which rejects any other
