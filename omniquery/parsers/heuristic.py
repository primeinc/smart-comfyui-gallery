"""Deterministic, rule-based NL -> AST parser. Zero external dependencies,
always available. This is the baseline every model-backed backend has to
beat: the router accepts it outright whenever it fully covers the query
(coverage == 1.0), because a correct deterministic parse is strictly better
than any model's guess.

Design: the input text has its quoted substrings pulled out into
placeholder tokens first (so later word-matching can't be confused by quote
contents, and literals like SQL-injection strings pass through verbatim).
A battery of regexes then runs over the remaining lowercased text in a
fixed priority order, each marking the character span it consumed in a
shared `consumed` array so no span is claimed twice. Coverage is the
fraction of *significant* (non-stopword) tokens whose span ended up fully
consumed; that fraction doubles as confidence, except numbers and quoted
strings are special-cased -- if any of them was left unconsumed, confidence
is hard-capped at 0.4 regardless of the overall fraction (see
`omniquery.parsers.coverage_guard` for the analogous model-output guard).

Negation ("not X" / "except X" / "without X" / "un-favorited") is handled
generically: a trigger word claims whatever single predicate immediately
follows it (favorite / media type / status flag) and wraps it in a 'not'
node -- this must run *before* the plain positive rules for those same
three fields, or the plain rule would grab the bare predicate first and the
negation would be lost. A single top-level "X or Y" is detected only after
every other rule has run (so idioms containing the literal word "or", like
"4 stars or better", get consumed as part of their own rule and never look
like a disjunction boundary).
"""

from __future__ import annotations

import calendar
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from omniquery.parsers import ParserBackend, ParserOutcome, try_validate

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

_TYPE_SYNONYMS: Dict[str, str] = {
    "animated images": "animated_image", "animated image": "animated_image",
    "photos": "image", "photo": "image", "pictures": "image", "picture": "image",
    "images": "image", "image": "image",
    "videos": "video", "video": "video", "clips": "video", "clip": "video",
    "movies": "video", "movie": "video",
    "gifs": "animated_image", "gif": "animated_image",
    "sounds": "audio", "sound": "audio", "music": "audio",
    "songs": "audio", "song": "audio", "audio": "audio",
    "documents": "document", "document": "document", "pdfs": "document", "pdf": "document",
}

_STATUS_SYNONYMS: Dict[str, str] = {
    "needs review": "Review", "in review": "Review", "review": "Review",
    "approved": "Approved", "rejected": "Rejected",
    "needs edit": "To Edit", "to edit": "To Edit",
    "selected": "Select", "select": "Select",
}

_ISSUE_SYNONYMS: Dict[str, str] = {
    "anatomy": "anatomy", "artifact": "artifact", "artifacts": "artifact",
    "composition": "composition", "lighting": "lighting",
    "text render": "text_render", "text rendering": "text_render",
    "prompt mismatch": "prompt_mismatch", "style": "style",
    "detail loss": "detail_loss",
}

_MONTHS: Dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "is", "are", "was", "were",
    "be", "been", "being", "with", "to", "for", "and", "or", "that", "this",
    "these", "those", "it", "its", "from", "as", "by", "than", "then", "me",
    "my", "i", "show", "find", "get", "give", "please", "all", "any", "some",
    "want", "need", "list", "search", "query", "which", "what", "files",
    "file", "items", "item", "everything", "things", "thing", "let", "lets",
    "there", "have", "has", "had", "do", "does", "did",
    "not", "except", "without", "excluding",
    "more", "less", "better", "above", "below", "over", "under",
})

_PLACEHOLDER_FRAGMENT = r"qzq(\d+)qzq"


def _build_alternation(mapping: Dict[str, str]) -> re.Pattern:
    keys = sorted(mapping.keys(), key=len, reverse=True)
    parts = [re.escape(k).replace(r"\ ", r"\s+") for k in keys]
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.I)


_TYPE_RE = _build_alternation(_TYPE_SYNONYMS)
_STATUS_RE = _build_alternation(_STATUS_SYNONYMS)
_FAVORITE_RE = re.compile(r"\bfavou?rite[sd]?\b", re.I)
_UNFAVORITED_RE = re.compile(r"\bun-?favou?rited\b", re.I)

_RATING_AT_LEAST_RE = re.compile(r"\brated?\s+at\s+least\s+(\d+)\b", re.I)
_RATING_PLUS_RE = re.compile(r"\b(\d+)\s*\+\s*stars?\b", re.I)
_RATING_OR_BETTER_RE = re.compile(r"\b(\d+)\s*stars?\s+or\s+(?:better|more|higher|above)\b", re.I)
_RATING_EXACT_RE = re.compile(r"\brated\s+(\d+)\b", re.I)

_SIZE_OP_MAP = {
    "over": "gt", "bigger than": "gt", "larger than": "gt", "more than": "gt",
    "under": "lt", "smaller than": "lt", "less than": "lt", "at least": "ge",
}
_SIZE_RE = re.compile(
    r"\b(over|under|bigger\s+than|larger\s+than|smaller\s+than|less\s+than|"
    r"more\s+than|at\s+least)\s+(\d+(?:\.\d+)?)\s*(mb|gb|megabytes?|gigabytes?)\b", re.I,
)

_DURATION_OP_MAP = {"longer than": "gt", "over": "gt", "shorter than": "lt",
                     "under": "lt", "at least": "ge"}
_DURATION_UNIT_SECONDS = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
}
_DURATION_RE = re.compile(
    r"\b(longer\s+than|shorter\s+than|over|under|at\s+least)\s+(\d+(?:\.\d+)?)\s*"
    r"(seconds?|secs?|minutes?|mins?|hours?|hrs?)\b", re.I,
)

_MP_OP_MAP = {"over": "gt", "above": "gt", "more than": "gt",
              "under": "lt", "below": "lt", "less than": "lt", "at least": "ge"}
_MEGAPIXELS_RE = re.compile(
    r"\b(?:(over|under|at\s+least|above|below|more\s+than|less\s+than)\s+)?"
    r"(\d+(?:\.\d+)?)\s*(?:megapixels?|mp)\b", re.I,
)

_QUALITY_OP_MAP = {"above": "gt", "over": "gt", "at least": "ge", "below": "lt", "under": "lt"}
_QUALITY_RE = re.compile(r"\bquality\s+(above|over|at\s+least|below|under)\s+(\d+(?:\.\d+)?)\b", re.I)

_FOLDER_IN_THE_RE = re.compile(r"\bin\s+the\s+([a-z0-9_\-/]+(?:\s[a-z0-9_\-/]+)*?)\s+folder\b", re.I)
_FOLDER_IN_FOLDER_RE = re.compile(r"\bin\s+folder\s+([a-z0-9_\-/]+)\b", re.I)
_FOLDER_UNDER_RE = re.compile(r"\bunder\s+([a-z0-9_\-]+(?:/[a-z0-9_\-]+)*)/", re.I)

_COLLECTION_RE = re.compile(r"\bin\s+the\s+([a-z0-9_\- ]+?)\s+(?:collection|album)\b", re.I)

_NAME_RE = re.compile(rf"\b(?:named|called)\s+{_PLACEHOLDER_FRAGMENT}\b", re.I)
_PROMPT_MENTIONS_RE = re.compile(rf"\bprompt\s+(?:mentions|contains)\s+{_PLACEHOLDER_FRAGMENT}\b", re.I)
_PROMPT_IN_RE = re.compile(rf"\bwith\s+{_PLACEHOLDER_FRAGMENT}\s+in\s+the\s+prompt\b", re.I)
_CAPTION_RE = re.compile(rf"\bcaption\s+mentions\s+{_PLACEHOLDER_FRAGMENT}\b", re.I)
_COMMENT_MENTIONS_RE = re.compile(rf"\bcomments?\s+mention(?:s|ing)?\s+{_PLACEHOLDER_FRAGMENT}\b", re.I)
_COMMENTED_RE = re.compile(r"\bcommented\b", re.I)

_WORKFLOW_WITH_RE = re.compile(r"\bwith\s+workflows?\b", re.I)
_WORKFLOW_WITHOUT_RE = re.compile(r"\b(?:without|no)\s+workflows?\b", re.I)
_FACES_WITH_RE = re.compile(r"\bwith\s+faces?\b", re.I)
_FACES_WITHOUT_RE = re.compile(r"\b(?:without|no)\s+faces?\b", re.I)

_ISSUE_RE = re.compile(r"\bwith\s+([a-z]+(?:\s[a-z]+)?)\s+issues?\b", re.I)

_COUNT_META_RE = re.compile(r"\bhow\s+many\b|\bcount\s+of\b|\bnumber\s+of\b", re.I)
_NEWEST_FIRST_RE = re.compile(r"\b(?:newest|latest)\s+first\b", re.I)
_OLDEST_RE = re.compile(r"\boldest\b", re.I)
_LARGEST_RE = re.compile(r"\blargest\b", re.I)
_BEST_RATED_RE = re.compile(r"\bbest\s+rated\b", re.I)
_TOP_N_RE = re.compile(r"\b(?:top|first)\s+(\d+)\b", re.I)

_LAST_N_RE = re.compile(r"\blast\s+(\d+)\s+(days?|weeks?|months?)\b", re.I)
_YESTERDAY_RE = re.compile(r"\byesterday\b", re.I)
_TODAY_RE = re.compile(r"\btoday\b", re.I)
_THIS_WEEK_RE = re.compile(r"\bthis\s+week\b", re.I)
_THIS_MONTH_RE = re.compile(r"\bthis\s+month\b", re.I)
_FROM_MONTH_YEAR_RE = re.compile(r"\bfrom\s+([a-z]+)\s+(\d{4})\b", re.I)
_IN_YEAR_RE = re.compile(r"\bin\s+(\d{4})\b", re.I)
_SINCE_AFTER_RE = re.compile(r"\b(?:since|after)\s+(\d{4}-\d{2}-\d{2})\b", re.I)
_BEFORE_RE = re.compile(r"\bbefore\s+(\d{4}-\d{2}-\d{2})\b", re.I)
_BETWEEN_DATES_RE = re.compile(r"\bbetween\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})\b", re.I)

_NEGATION_TRIGGER_RE = re.compile(r"\b(not|except|without|excluding)\b", re.I)
_NEGATION_BOUNDARY_RE = re.compile(r",|;|\band\b|\bor\b")
_OR_TOKEN_RE = re.compile(r"\bor\b", re.I)
_QUOTE_EXTRACT_RE = re.compile(r'["\']([^"\']+)["\']')
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_DIGIT_RUN_RE = re.compile(r"\d+")
_PLACEHOLDER_RE = re.compile(_PLACEHOLDER_FRAGMENT)


# ---------------------------------------------------------------------------
# Rule application
# ---------------------------------------------------------------------------

def _apply_rule(text: str, consumed: List[bool], pattern: re.Pattern,
                 builder) -> List[Tuple[int, int, Any]]:
    """Find every non-overlapping (with already-consumed spans) match of
    `pattern`, run `builder(match)`; a None result means "matched the shape
    but not a value we recognize" and is skipped without consuming."""
    hits: List[Tuple[int, int, Any]] = []
    for m in pattern.finditer(text):
        s, e = m.span()
        if any(consumed[s:e]):
            continue
        built = builder(m)
        if built is None:
            continue
        for i in range(s, e):
            consumed[i] = True
        hits.append((s, e, built))
    return hits


def _and_or_single(conds: List[dict]) -> Optional[dict]:
    # De-duplicate identical conditions -- two synonyms for the same value
    # in one query (e.g. "video clips") each produce their own match, and a
    # repeated AND-term is semantically inert but structurally ugly.
    deduped: List[dict] = []
    seen = set()
    for c in conds:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    if not deduped:
        return None
    if len(deduped) == 1:
        return deduped[0]
    return {"op": "and", "children": deduped}


class HeuristicBackend(ParserBackend):
    name = "heuristic"

    def parse(self, text: str, now_epoch: float) -> ParserOutcome:  # noqa: ARG002
        t0 = time.monotonic()
        placeheld, quotes = _extract_quotes(text)
        working = placeheld.lower()
        consumed = [False] * len(working)

        spanned_conds: List[Tuple[int, int, dict]] = []
        meta: Dict[str, Any] = {}

        def _quote_builder(field: str):
            def builder(m: re.Match) -> Optional[dict]:
                literal = quotes.get(int(m.group(1)))
                if literal is None:
                    return None
                return {"field": field, "op": "contains", "value": literal}
            return builder

        # -- has_workflow / has_faces: specific booleans take priority over
        # the generic negation trigger so "without workflow" isn't reread as
        # NOT(has_workflow=True) via the generic path. --------------------
        spanned_conds += _apply_rule(working, consumed, _WORKFLOW_WITH_RE,
                                      lambda m: {"field": "has_workflow", "op": "eq", "value": True})
        spanned_conds += _apply_rule(working, consumed, _WORKFLOW_WITHOUT_RE,
                                      lambda m: {"field": "has_workflow", "op": "eq", "value": False})
        spanned_conds += _apply_rule(working, consumed, _FACES_WITH_RE,
                                      lambda m: {"field": "has_faces", "op": "eq", "value": True})
        spanned_conds += _apply_rule(working, consumed, _FACES_WITHOUT_RE,
                                      lambda m: {"field": "has_faces", "op": "eq", "value": False})

        # -- un-favorited: single-word negation, own direct rule. ----------
        spanned_conds += _apply_rule(
            working, consumed, _UNFAVORITED_RE,
            lambda m: {"op": "not", "child": {"field": "is_favorite", "op": "eq", "value": True}},
        )

        # -- star ratings (also protects "N stars or better"'s "or" from
        # ever being mistaken for a disjunction boundary). -----------------
        spanned_conds += _apply_rule(working, consumed, _RATING_AT_LEAST_RE,
                                      lambda m: {"field": "rating_avg", "op": "ge", "value": int(m.group(1))})
        spanned_conds += _apply_rule(working, consumed, _RATING_OR_BETTER_RE,
                                      lambda m: {"field": "rating_avg", "op": "ge", "value": int(m.group(1))})
        spanned_conds += _apply_rule(working, consumed, _RATING_PLUS_RE,
                                      lambda m: {"field": "rating_avg", "op": "ge", "value": int(m.group(1))})
        spanned_conds += _apply_rule(working, consumed, _RATING_EXACT_RE,
                                      lambda m: {"field": "rating_avg", "op": "eq", "value": int(m.group(1))})

        # -- size / duration / megapixels / review quality ------------------
        def _size_builder(m: re.Match) -> dict:
            qualifier, num, unit = m.group(1), float(m.group(2)), m.group(3).lower()
            op = _SIZE_OP_MAP.get(re.sub(r"\s+", " ", qualifier.lower()), "gt")
            value = num * 1024 if unit.startswith("g") else num
            return {"field": "size_mb", "op": op, "value": value}

        spanned_conds += _apply_rule(working, consumed, _SIZE_RE, _size_builder)

        def _duration_builder(m: re.Match) -> dict:
            qualifier, num, unit = m.group(1), float(m.group(2)), m.group(3).lower()
            op = _DURATION_OP_MAP.get(re.sub(r"\s+", " ", qualifier.lower()), "gt")
            mult = _DURATION_UNIT_SECONDS.get(unit, 1)
            return {"field": "duration_seconds", "op": op, "value": num * mult}

        spanned_conds += _apply_rule(working, consumed, _DURATION_RE, _duration_builder)

        def _mp_builder(m: re.Match) -> dict:
            qualifier, value = m.group(1), float(m.group(2))
            op = "eq" if not qualifier else _MP_OP_MAP.get(re.sub(r"\s+", " ", qualifier.lower()), "ge")
            return {"field": "megapixels", "op": op, "value": value}

        spanned_conds += _apply_rule(working, consumed, _MEGAPIXELS_RE, _mp_builder)

        def _quality_builder(m: re.Match) -> dict:
            qualifier, num = re.sub(r"\s+", " ", m.group(1).lower()), float(m.group(2))
            return {"field": "review_quality", "op": _QUALITY_OP_MAP.get(qualifier, "gt"), "value": num}

        spanned_conds += _apply_rule(working, consumed, _QUALITY_RE, _quality_builder)

        # -- dates ------------------------------------------------------------
        def _last_n_builder(m: re.Match) -> dict:
            n, unit = int(m.group(1)), m.group(2).lower()
            mult = 7 if unit.startswith("week") else 30 if unit.startswith("month") else 1
            return {"field": "mtime", "op": "ge", "value": {"days_ago": n * mult}}

        spanned_conds += _apply_rule(working, consumed, _LAST_N_RE, _last_n_builder)
        # ast.py forbids dicts inside a list, so a relative-date "between"
        # isn't representable as op="between" with two {"days_ago": N}
        # elements -- express "yesterday" as an AND of two plain bounds
        # instead (structurally just a Group, valid anywhere a Cond is).
        spanned_conds += _apply_rule(
            working, consumed, _YESTERDAY_RE,
            lambda m: {"op": "and", "children": [
                {"field": "mtime", "op": "ge", "value": {"days_ago": 2}},
                {"field": "mtime", "op": "lt", "value": {"days_ago": 1}},
            ]},
        )
        spanned_conds += _apply_rule(working, consumed, _TODAY_RE,
                                      lambda m: {"field": "mtime", "op": "ge", "value": {"days_ago": 1}})
        spanned_conds += _apply_rule(working, consumed, _THIS_WEEK_RE,
                                      lambda m: {"field": "mtime", "op": "ge", "value": {"days_ago": 7}})
        spanned_conds += _apply_rule(working, consumed, _THIS_MONTH_RE,
                                      lambda m: {"field": "mtime", "op": "ge", "value": {"days_ago": 30}})

        def _from_month_year_builder(m: re.Match) -> Optional[dict]:
            month = _MONTHS.get(m.group(1).lower())
            if month is None:
                return None
            year = int(m.group(2))
            last_day = calendar.monthrange(year, month)[1]
            lo = f"{year:04d}-{month:02d}-01"
            hi = f"{year:04d}-{month:02d}-{last_day:02d}"
            return {"field": "mtime", "op": "between", "value": [lo, hi]}

        spanned_conds += _apply_rule(working, consumed, _FROM_MONTH_YEAR_RE, _from_month_year_builder)
        spanned_conds += _apply_rule(
            working, consumed, _IN_YEAR_RE,
            lambda m: {"field": "mtime", "op": "between",
                       "value": [f"{m.group(1)}-01-01", f"{m.group(1)}-12-31"]},
        )
        spanned_conds += _apply_rule(working, consumed, _SINCE_AFTER_RE,
                                      lambda m: {"field": "mtime", "op": "ge", "value": m.group(1)})
        spanned_conds += _apply_rule(working, consumed, _BEFORE_RE,
                                      lambda m: {"field": "mtime", "op": "lt", "value": m.group(1)})
        spanned_conds += _apply_rule(
            working, consumed, _BETWEEN_DATES_RE,
            lambda m: {"field": "mtime", "op": "between", "value": [m.group(1), m.group(2)]},
        )

        # -- folder / collection / quoted text searches ----------------------
        # captured groups are sliced out of `placeheld` (pre-lowercasing) so
        # free-text values like folder/collection names keep their original
        # casing; matching itself still runs against `working` since these
        # regexes are otherwise case-insensitive.
        def _orig_slice(m: re.Match, group: int = 1) -> str:
            return placeheld[m.start(group):m.end(group)].strip()

        spanned_conds += _apply_rule(working, consumed, _FOLDER_IN_THE_RE,
                                      lambda m: {"field": "folder", "op": "eq", "value": _orig_slice(m)})
        spanned_conds += _apply_rule(working, consumed, _FOLDER_IN_FOLDER_RE,
                                      lambda m: {"field": "folder", "op": "eq", "value": _orig_slice(m)})
        spanned_conds += _apply_rule(working, consumed, _FOLDER_UNDER_RE,
                                      lambda m: {"field": "folder", "op": "contains", "value": _orig_slice(m)})
        spanned_conds += _apply_rule(working, consumed, _COLLECTION_RE,
                                      lambda m: {"field": "collection", "op": "eq", "value": _orig_slice(m)})
        spanned_conds += _apply_rule(working, consumed, _NAME_RE, _quote_builder("name"))
        spanned_conds += _apply_rule(working, consumed, _PROMPT_MENTIONS_RE, _quote_builder("workflow_prompt"))
        spanned_conds += _apply_rule(working, consumed, _PROMPT_IN_RE, _quote_builder("workflow_prompt"))
        spanned_conds += _apply_rule(working, consumed, _CAPTION_RE, _quote_builder("ai_caption"))
        spanned_conds += _apply_rule(working, consumed, _COMMENT_MENTIONS_RE, _quote_builder("comment_contains"))
        spanned_conds += _apply_rule(working, consumed, _COMMENTED_RE,
                                      lambda m: {"field": "comment_count", "op": "gt", "value": 0})

        # -- review issues ------------------------------------------------
        def _issue_builder(m: re.Match) -> Optional[dict]:
            value = _ISSUE_SYNONYMS.get(re.sub(r"\s+", " ", m.group(1).strip().lower()))
            if value is None:
                return None
            return {"field": "review_issue", "op": "eq", "value": value}

        spanned_conds += _apply_rule(working, consumed, _ISSUE_RE, _issue_builder)

        # -- meta: counts and presentation -----------------------------------
        if _apply_rule(working, consumed, _COUNT_META_RE, lambda m: True):
            meta["result"] = "count"
        if _apply_rule(working, consumed, _NEWEST_FIRST_RE, lambda m: True):
            meta["order"] = ("mtime", "desc")
        elif _apply_rule(working, consumed, _OLDEST_RE, lambda m: True):
            meta["order"] = ("mtime", "asc")
        elif _apply_rule(working, consumed, _LARGEST_RE, lambda m: True):
            meta["order"] = ("size_bytes", "desc")
        elif _apply_rule(working, consumed, _BEST_RATED_RE, lambda m: True):
            meta["order"] = ("rating_avg", "desc")
        top_hits = _apply_rule(working, consumed, _TOP_N_RE, lambda m: int(m.group(1)))
        if top_hits:
            meta["limit"] = top_hits[0][2]
            meta.setdefault("order", ("mtime", "desc"))

        # -- generic negation (favorite / media type / status flag) must run
        # BEFORE the plain positive rules for those three fields. -----------
        spanned_conds += _try_negation(working, consumed)

        # -- plain positive favorite / media type / status flag -------------
        spanned_conds += _apply_rule(working, consumed, _FAVORITE_RE,
                                      lambda m: {"field": "is_favorite", "op": "eq", "value": True})

        def _type_builder(m: re.Match) -> Optional[dict]:
            key = re.sub(r"\s+", " ", m.group(1).lower())
            value = _TYPE_SYNONYMS.get(key)
            return None if value is None else {"field": "type", "op": "eq", "value": value}

        spanned_conds += _apply_rule(working, consumed, _TYPE_RE, _type_builder)

        def _status_builder(m: re.Match) -> Optional[dict]:
            key = re.sub(r"\s+", " ", m.group(1).lower())
            value = _STATUS_SYNONYMS.get(key)
            return None if value is None else {"field": "status_flag", "op": "eq", "value": value}

        spanned_conds += _apply_rule(working, consumed, _STATUS_RE, _status_builder)

        # -- simple top-level disjunction: only an "or" no rule above wanted
        # gets treated as a split point. -------------------------------------
        disjunction_failed = False
        where: Optional[dict]
        or_matches = [m for m in _OR_TOKEN_RE.finditer(working) if not any(consumed[m.start():m.end()])]
        if or_matches:
            or_m = or_matches[0]
            left = [c for (s, e, c) in spanned_conds if e <= or_m.start()]
            right = [c for (s, e, c) in spanned_conds if s >= or_m.end()]
            if left and right:
                for i in range(*or_m.span()):
                    consumed[i] = True
                where = {"op": "or", "children": [_and_or_single(left), _and_or_single(right)]}
            else:
                disjunction_failed = True
                where = None
        else:
            where = _and_or_single([c for (_, _, c) in spanned_conds])

        latency_ms = (time.monotonic() - t0) * 1000.0

        if disjunction_failed:
            return ParserOutcome(ast=None, confidence=None, backend=self.name, unsupported=True,
                                  reason="unparsed disjunct", latency_ms=latency_ms)

        matched_anything = bool(spanned_conds) or bool(meta)
        if not matched_anything:
            return ParserOutcome(ast=None, confidence=None, backend=self.name, unsupported=True,
                                  reason="no recognizable predicates", latency_ms=latency_ms)

        coverage, unconsumed_words = _token_coverage(working, consumed)
        confidence = coverage
        if _has_unconsumed_literal(working, consumed):
            confidence = min(confidence, 0.4)

        ast_dict: Dict[str, Any] = {"result": meta.get("result", "ids")}
        if where is not None:
            ast_dict["where"] = where
        if "order" in meta:
            field_name, direction = meta["order"]
            ast_dict["order_by"] = [{"field": field_name, "dir": direction}]
        if "limit" in meta:
            ast_dict["limit"] = meta["limit"]

        query, err = try_validate(ast_dict)
        if err is not None:
            return ParserOutcome(ast=None, confidence=None, backend=self.name, unsupported=True,
                                  reason=err, coverage=coverage, latency_ms=latency_ms)

        reason = None
        if unconsumed_words:
            reason = "unconsumed tokens: " + ", ".join(unconsumed_words)

        return ParserOutcome(ast=query.to_dict(), confidence=confidence, backend=self.name,
                              unsupported=False, reason=reason, coverage=coverage,
                              latency_ms=latency_ms)


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------

def _match_predicate_at_start(s: str) -> Optional[Tuple[dict, int]]:
    m = _FAVORITE_RE.match(s)
    if m:
        return {"field": "is_favorite", "op": "eq", "value": True}, m.end()
    m = _TYPE_RE.match(s)
    if m:
        key = re.sub(r"\s+", " ", m.group(1).lower())
        value = _TYPE_SYNONYMS.get(key)
        if value is not None:
            return {"field": "type", "op": "eq", "value": value}, m.end()
    m = _STATUS_RE.match(s)
    if m:
        key = re.sub(r"\s+", " ", m.group(1).lower())
        value = _STATUS_SYNONYMS.get(key)
        if value is not None:
            return {"field": "status_flag", "op": "eq", "value": value}, m.end()
    return None


def _try_negation(text: str, consumed: List[bool]) -> List[Tuple[int, int, dict]]:
    hits: List[Tuple[int, int, dict]] = []
    for m in _NEGATION_TRIGGER_RE.finditer(text):
        s, e = m.span()
        if any(consumed[s:e]):
            continue
        boundary_m = _NEGATION_BOUNDARY_RE.search(text, e)
        boundary = boundary_m.start() if boundary_m else len(text)
        window = text[e:boundary]
        stripped = window.lstrip()
        lstrip_offset = len(window) - len(stripped)
        result = _match_predicate_at_start(stripped)
        if result is None:
            continue
        cond, matched_len = result
        abs_start, abs_end = s, e + lstrip_offset + matched_len
        if any(consumed[abs_start:abs_end]):
            continue
        for i in range(abs_start, abs_end):
            consumed[i] = True
        hits.append((abs_start, abs_end, {"op": "not", "child": cond}))
    return hits


# ---------------------------------------------------------------------------
# Quote extraction, tokenization, coverage
# ---------------------------------------------------------------------------

def _extract_quotes(text: str) -> Tuple[str, Dict[int, str]]:
    quotes: Dict[int, str] = {}
    counter = [0]

    def _replace(m: re.Match) -> str:
        idx = counter[0]
        counter[0] += 1
        quotes[idx] = m.group(1)
        return f"qzq{idx}qzq"

    return _QUOTE_EXTRACT_RE.sub(_replace, text), quotes


def _token_coverage(text: str, consumed: List[bool]) -> Tuple[float, List[str]]:
    significant = [
        (m.group(0), m.start(), m.end())
        for m in _TOKEN_RE.finditer(text)
        if m.group(0) not in STOPWORDS
    ]
    if not significant:
        return 1.0, []
    consumed_count = 0
    unconsumed: List[str] = []
    for word, s, e in significant:
        if all(consumed[s:e]):
            consumed_count += 1
        else:
            unconsumed.append(word)
    return consumed_count / len(significant), unconsumed


def _has_unconsumed_literal(text: str, consumed: List[bool]) -> bool:
    for m in _DIGIT_RUN_RE.finditer(text):
        s, e = m.span()
        if not all(consumed[s:e]):
            return True
    for m in _PLACEHOLDER_RE.finditer(text):
        s, e = m.span()
        if not all(consumed[s:e]):
            return True
    return False
