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

The model file resolves like every other optional GGUF: constructor arg,
then $OMNIQUERY_NL2SQL_GGUF, then the provisioned default (provision
group 'omniquery'). Loading shares fallback_qwen's canaried loader: a GPU
build that decodes garbage or crashes the sampler reloads CPU-only,
loudly, instead of poisoning results.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

from omniquery.parsers.fallback_qwen import (
    DEFAULT_MODEL_PATH, _prepare_dll_path, load_canaried_llama,
)

_logger = logging.getLogger(__name__)

ENV_MODEL_PATH = "OMNIQUERY_NL2SQL_GGUF"

# Tables worth the model's attention; internal bookkeeping (schema
# versions, sessions, scan logs) only wastes prompt tokens and invites
# joins against noise.
_SCHEMA_TABLES = (
    "files", "generation_params", "file_ratings", "file_comments",
    "collections", "collection_files", "ai_face_instances",
    "ai_face_clusters", "ai_reviews", "ai_review_findings", "users",
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
_SCHEMA_CACHE: Dict[str, Tuple[float, str]] = {}
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
    placeholders = ",".join("?" for _ in _SCHEMA_TABLES)
    with sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            f"SELECT name, sql FROM sqlite_master WHERE type='table' "
            f"AND name IN ({placeholders}) ORDER BY name",
            _SCHEMA_TABLES,
        ).fetchall()
        # Live value hints for the enum-ish columns the DDL cannot show
        # (observed miss: the model guessed type='photo' where the data
        # says 'image'). Sourced from the data itself, so they are right
        # for every install.
        hints: List[str] = []
        try:
            types = [r[0] for r in conn.execute(
                "SELECT DISTINCT type FROM files WHERE type IS NOT NULL "
                "ORDER BY type LIMIT 12")]
            if types:
                hints.append("-- files.type values: " + ", ".join(types))
        except sqlite3.Error:
            pass
        try:
            flags = [r[0] for r in conn.execute(
                "SELECT DISTINCT name FROM collections "
                "WHERE type = 'system_flag' ORDER BY name LIMIT 12")]
            if flags:
                hints.append("-- collections.name values where "
                             "type='system_flag': " + ", ".join(flags))
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
        sql = sql[:junk.start()]
    return sql.strip()


class SqlSearch:
    """NL -> SQL through the local text2sql GGUF. generate() never raises;
    it returns (sql, None) or (None, reason)."""

    def __init__(self, db_path: str, model_path: Optional[str] = None,
                 n_ctx: int = 4096, n_threads: int = 4, max_tokens: int = 256):
        self.db_path = db_path
        self.model_path = (model_path
                           or os.environ.get(ENV_MODEL_PATH)
                           or os.environ.get("OMNIQUERY_FALLBACK_GGUF")
                           or DEFAULT_MODEL_PATH)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.max_tokens = max_tokens

    def available(self) -> bool:
        """True when llama_cpp imports and the model file exists; never
        triggers the (expensive) model load."""
        try:
            _prepare_dll_path()
            import llama_cpp
        except Exception:
            return False
        return bool(llama_cpp) and os.path.isfile(self.model_path)

    def _complete(self, messages: List[Dict[str, str]]) -> str:
        llama = load_canaried_llama(self.model_path, self.n_ctx, self.n_threads)
        resp = llama.create_chat_completion(
            messages=messages,
            # The distil card's own example decodes at temperature 0 (the
            # fine-tune, unlike base qwen3, is documented greedy-safe).
            temperature=0.0,
            max_tokens=self.max_tokens,
            stop=["<|", ";", "[INST]", "\n\n"],
        )
        return resp["choices"][0]["message"]["content"]

    def search(self, question: str, max_rounds: int = 3
               ) -> Tuple[Optional[List[str]], Optional[str], Optional[str]]:
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
        from omniquery.sqlexec import run_readonly_select

        try:
            schema = schema_block(self.db_path)
        except Exception as exc:
            return None, None, f"schema read error: {exc}"
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
        ]

        last_sql: Optional[str] = None
        last_reason: Optional[str] = None
        for _round in range(max_rounds):
            try:
                content = self._complete(messages)
            except Exception as exc:
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

            messages.append({"role": "assistant", "content": sql})
            if not result.ok:
                last_reason = result.error
                messages.append({
                    "role": "user",
                    "content": f"That query failed with: {result.error}\n"
                               "Fix it. Output only the corrected SQL query."})
            else:
                last_reason = "0 rows"
                messages.append({
                    "role": "user",
                    "content": "That query ran but returned 0 rows. If the "
                               "question could match differently (other text "
                               "columns from the rules, looser LIKE patterns, "
                               "fewer constraints), output a broadened SQL "
                               "query. If 0 results is genuinely the correct "
                               "answer, output the exact same query again."})
            last_sql = sql

        if last_sql is not None and last_reason == "0 rows":
            # Rounds exhausted while still broadening: empty stands.
            return [], last_sql, None
        return None, last_sql, last_reason or "no usable SQL produced"
