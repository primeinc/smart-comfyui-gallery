"""OmniQuery v2: local natural-language querying for SmartGallery.

Two answerers, fused per query by the app's search endpoint:

    RULES:  natural language
              -> nlq parser (deterministic, ALWAYS answers: rules consume
                 recognized structure, every leftover term becomes a
                 universal full-text condition) -> typed AST
              -> schema + semantic + authorization + complexity validation
              -> deterministic parameterized read-only SELECT compiler
              -> execution on a read-only connection with an authorizer
            Exact for fully-consumed queries; the only live-typing path.

    MODEL:  natural language + the LIVE schema (sqlite_master)
              -> local text2sql GGUF (parsers/nl2sql.SqlSearch) -> SQL
              -> sqlexec.run_readonly_select, the ONE sandboxed gate
                 (SELECT prefix + read-only URI + C-engine authorizer)
            Agentic: the model executes, READS the outcome, and repairs /
            broadens before answering. Handles free language; any failure
            falls back to the rules answer.

Invariants:
  - Model SQL is data, not trusted code: it executes exclusively through
    the sqlexec sandbox and can at worst return wrong rows.
  - The rules path never interpolates literals (bound parameters only)
    and validates/authorizes outside any model.
  - Behavior is measured, not assumed: `just ai bench-fusion` reproduces
    the acceptance number over omniquery/benchmark/corpus.jsonl.
"""

AST_VERSION = 1  # wire-format version parsers must emit; mirrored in ast.py, which rejects any other
