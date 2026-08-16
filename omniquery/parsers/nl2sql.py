"""Grammar-constrained local nl2sql refiner: a text2sql-tuned GGUF decoded
under the typed AST grammar (never raw SQL -- the grammar from
`omniquery.ast.json_schema` structurally forbids it, so "nl2sql" here means
the model FAMILY, not the output format).

Role in the stack: `omniquery.parsers.nlq` answers every query by itself;
this model is consulted only by `parse_search` when the deterministic parse
left structural-looking text unexplained ("videos shorter then 2 min" with
a typo, tangled comparative phrasing). Its output replaces the rule parse
only when it validates AND passes coverage_guard at full coverage --
otherwise the deterministic result stands. Measured on the 83-entry corpus
(2026-08-16, GPU): distil-t2s-4b 43.4% standalone execution match, the best
of five GGUF candidates.

All the loading/decoding machinery lives in fallback_qwen (module-scope
model cache, grammar built from the real AST schema, ParserOutcome
contract); this class only rebinds identity, model resolution, and the
system prompt's role framing.
"""

from __future__ import annotations

import os
from typing import Optional

from omniquery.parsers.fallback_qwen import DEFAULT_MODEL_PATH, FallbackQwenBackend

# Model file resolution precedence: constructor argument, then
# $OMNIQUERY_NL2SQL_GGUF, then $OMNIQUERY_FALLBACK_GGUF (shared install),
# then the packaged default location.
ENV_MODEL_PATH = "OMNIQUERY_NL2SQL_GGUF"


class Nl2SqlBackend(FallbackQwenBackend):
    """Text2sql-family GGUF under the AST grammar; see module docstring."""

    name = "nl2sql"

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        resolved = (model_path
                    or os.environ.get(ENV_MODEL_PATH)
                    or os.environ.get("OMNIQUERY_FALLBACK_GGUF")
                    or DEFAULT_MODEL_PATH)
        super().__init__(model_path=resolved, **kwargs)
