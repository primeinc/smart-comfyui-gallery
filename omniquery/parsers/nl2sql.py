"""The nl2sql search model: natural language in, SQL out -- the way the
text2sql model family is built and documented to work. No intermediate
representation: the model reads the LIVE database schema (pulled from
sqlite_master, so it is correct for every install and every schema
version) plus the user's question, and emits one SQLite SELECT.

Generated SQL is DATA, not trusted code: every statement runs exclusively
through omniquery.sqlexec.run_readonly_select -- the same
prefix-check + read-only-URI + C-engine-authorizer sandbox that guards
the manual Advanced-SQL endpoint. A malicious or hallucinated statement
can at worst return wrong rows; it cannot write, PRAGMA, or ATTACH.

Prompt contract and sampling follow the distil-labs model card verbatim
(system rules + 'Schema:\\n<DDL>\\n\\nQuestion: <q>', temperature 0),
with one added rule the result pipeline needs: the first selected column
must be files.id, because "a list of file ids" is the only thing the
gallery ever asks SQL for.

The checkpoint resolves like every other model here: constructor argument,
then $OMNIQUERY_NL2SQL_MODEL, then DEFAULT_MODEL -- and it loads through
smartgallery_ai.models, the same loader the reviewer uses. The agentic
loop below is ONE `Chat`, so the schema block (the bulk of the prompt, and
identical every round) is encoded once and every retry reuses its keys and
values instead of re-sending the whole conversation.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import time

from omniquery.sqlexec import run_readonly_select
from smartgallery_ai import models as ai_models
from sqlbind import with_id_placeholders

_logger = logging.getLogger(__name__)

#: The safetensors distil-labs text2sql checkpoint this module's prompt
#: contract and sampling follow the card of. Any causal-LM checkpoint
#: works -- choosing one is configuration, not code.
DEFAULT_MODEL = "distil-labs/distil-qwen3-4b-text2sql"
ENV_MODEL = "OMNIQUERY_NL2SQL_MODEL"

# Tables worth the model's attention; internal bookkeeping (schema
# versions, sessions, scan logs) only wastes prompt tokens and invites
# joins against noise.
_SCHEMA_TABLES = (
    "files",
    "generation_params",
    "file_ratings",
    "file_comments",
    "collections",
    "collection_files",
    "ai_face_instances",
    "ai_face_clusters",
    "ai_reviews",
    "ai_review_findings",
    "users",
)

# The distil model card's system prompt, with the SQL dialect rules kept
# verbatim and the gallery's one output-contract rule added.
_SYSTEM_PROMPT = (
    "You are given a database schema and a natural language question. "
    "Generate the SQL query that answers the question.\n\n"
    "Rules:\n"
    "- Use only tables and columns from the provided schema\n"
    "- Use uppercase SQL keywords (SELECT, FROM, WHERE, etc.)\n"
    "- Use SQLite-compatible syntax\n"
    "- The first selected column must be files.id (use DISTINCT); the one "
    "exception is a 'how many' question, which may SELECT COUNT(...)\n"
    "- This is a media gallery of generated images/videos. Subject or "
    "style words (what a picture is OF) live in the text columns: "
    "files.name, files.path, files.workflow_prompt, files.ai_caption, "
    "generation_params.positive_prompt, generation_params.model, "
    "generation_params.loras -- search ALL of them with OR for such terms\n"
    "- files.mtime is a unix epoch in seconds; strftime('%s','now') is "
    "the current time\n"
    "- Text matching should be case-insensitive LIKE with % wildcards\n"
    "- Output only the SQL query, no explanations"
)

_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# (db_path, schema mtime-ish key) -> rendered schema block
_SCHEMA_CACHE: dict[str, tuple[float, str]] = {}
_SCHEMA_TTL_SECONDS = 300.0


def schema_block(db_path: str) -> str:
    """The live schema as CREATE TABLE DDL straight from sqlite_master,
    filtered to the tables that matter and cached briefly. This is the
    exact input shape the text2sql family is trained on, and it can never
    drift from the running database because it IS the running database."""
    now = time.monotonic()
    cached = _SCHEMA_CACHE.get(db_path)
    if cached is not None and now - cached[0] < _SCHEMA_TTL_SECONDS:
        return cached[1]
    with sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            with_id_placeholders(
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name IN ({ids}) ORDER BY name",
                _SCHEMA_TABLES,
            ),
            _SCHEMA_TABLES,
        ).fetchall()
        # Live value hints for the enum-ish columns the DDL cannot show
        # (observed miss: the model guessed type='photo' where the data
        # says 'image'). Sourced from the data itself, so they are right
        # for every install.
        hints: list[str] = []
        try:
            types = [
                r[0]
                for r in conn.execute("SELECT DISTINCT type FROM files WHERE type IS NOT NULL ORDER BY type LIMIT 12")
            ]
            if types:
                hints.append("-- files.type values: " + ", ".join(types))
        except sqlite3.Error:
            pass
        try:
            flags = [
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT name FROM collections WHERE type = 'system_flag' ORDER BY name LIMIT 12"
                )
            ]
            if flags:
                hints.append("-- collections.name values where type='system_flag': " + ", ".join(flags))
        except sqlite3.Error:
            pass
    block = "\n\n".join(sql for _name, sql in rows if sql)
    if hints:
        block += "\n\n" + "\n".join(hints)
    _SCHEMA_CACHE[db_path] = (now, block)
    return block


# Chat-template detritus the model can free-run into after its answer:
# '<|im...', '<tool_call>', '<s>', '</s>', '[INST]', '[ERROR]', fences.
# '<' followed by '|', a letter, or '/' can never occur in valid SQLite
# (real comparisons are '< 5', '<=', '<>'), so cutting there is lossless.
_JUNK_RE = re.compile(r"<[|A-Za-z/]|\[INST\]|\[ERROR\]|```")


def _extract_sql(content: str) -> str:
    """The statement itself: fenced block if the model added one, else the
    raw text, cut at the first statement terminator or junk marker (see
    _JUNK_RE; observed live: '; [INST] ...' free-runs and a trailing
    '<tool_call><s>' that broke 27/98 bench queries)."""
    m = _SQL_FENCE_RE.search(content)
    sql = m.group(1) if m else content
    sql = sql.split(";", 1)[0]
    junk = _JUNK_RE.search(sql)
    if junk:
        sql = sql[: junk.start()]
    return sql.strip()


class SqlSearch:
    """NL -> SQL through the local text2sql model. search() never raises;
    it returns (ids, sql, None) or (None, sql, reason)."""

    def __init__(
        self, db_path: str, model_ref: str | None = None, models_dir: str | None = None, max_tokens: int = 256
    ):
        self.db_path = db_path
        self.model_ref = model_ref or os.environ.get(ENV_MODEL) or DEFAULT_MODEL
        self.models_dir = models_dir or os.environ.get("AI_DAM_MODELS_DIR", ".AImodels")
        self.max_tokens = max_tokens

    def available(self) -> bool:
        """True when the runtime imports and the checkpoint is provisioned;
        never triggers the (expensive) model load."""
        try:
            from smartgallery_ai import models as ai_models
        except Exception:
            _logger.debug("handled a failure in available", exc_info=True)
            return False
        return ai_models.is_provisioned(self.model_ref, self.models_dir)

    def _chat(self):
        """One conversation for a whole search.

        Decoding is greedy: the distil card's own example runs at
        temperature 0, and this fine-tune (unlike base qwen3) is documented
        greedy-safe. There are no stop strings -- `_extract_sql` already
        cuts the statement at the first terminator or template junk, and it
        is the tested cut."""

        return ai_models.Chat(self.model_ref, models_dir=self.models_dir, system=_SYSTEM_PROMPT)

    def search(self, question: str, max_rounds: int = 3) -> tuple[list[str] | None, str | None, str | None]:
        """The agentic loop: the model generates SQL, READS the sandboxed
        execution outcome, and acts on it before answering.

          - execution error -> the error goes back for a fix
          - 0 rows -> the model chooses: broaden the query (other text
            columns, looser matching), or return the SAME query to assert
            that empty is genuinely the answer
          - rows -> accepted

        At most `max_rounds` generations; every execution goes through
        run_readonly_select and nothing else. Returns (ids, sql, None) on
        success or (None, last_sql, reason). Never raises."""

        try:
            schema = schema_block(self.db_path)
        except Exception as exc:
            _logger.debug("handled a failure in search", exc_info=True)
            return None, None, f"schema read error: {exc}"
        try:
            chat = self._chat()
        except Exception as exc:
            _logger.debug("handled a failure in search", exc_info=True)
            return None, None, f"model load error: {exc}"
        # The schema block is the bulk of the prompt and never changes, so
        # it is encoded once on this first turn and every retry below
        # reuses it from the cache.
        turn = f"Schema:\n{schema}\n\nQuestion: {question}"

        last_sql: str | None = None
        last_reason: str | None = None
        for _round in range(max_rounds):
            try:
                content = chat.ask(turn, max_new_tokens=self.max_tokens)
            except Exception as exc:
                _logger.debug("handled a failure in search", exc_info=True)
                return None, last_sql, f"generation error: {exc}"
            sql = _extract_sql(content)
            if not sql:
                return None, last_sql, "empty generation"

            result = run_readonly_select(self.db_path, sql)
            if result.ok and result.ids:
                return result.ids, sql, None
            if result.ok and sql == last_sql:
                # The model saw "0 rows" and stands by its query: empty
                # IS the answer.
                return [], sql, None

            # The model's own reply is already in the chat's history, so
            # only the next instruction has to be composed.
            if not result.ok:
                last_reason = result.error
                turn = f"That query failed with: {result.error}\nFix it. Output only the corrected SQL query."
            else:
                last_reason = "0 rows"
                turn = (
                    "That query ran but returned 0 rows. If the question "
                    "could match differently (other text columns from the "
                    "rules, looser LIKE patterns, fewer constraints), "
                    "output a broadened SQL query. If 0 results is "
                    "genuinely the correct answer, output the exact same "
                    "query again."
                )
            last_sql = sql

        if last_sql is not None and last_reason == "0 rows":
            # Rounds exhausted while still broadening: empty stands.
            return [], last_sql, None
        return None, last_sql, last_reason or "no usable SQL produced"
