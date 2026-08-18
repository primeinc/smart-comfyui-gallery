"""The search-box parser: every query ALWAYS produces a runnable AST.

Contract (this is the whole design): the input is tokenized, a battery of
deterministic rules consumes the spans it recognizes (types, dates, sizes,
ratings, folders, negation, ordering, counts, ...), and then EVERY leftover
significant token becomes a full-text phrase matched (contains) against the
universal `text` field -- filename, path, workflow prompt, AI caption,
generation prompt, model, LoRA names at once. There is no "couldn't
confidently parse": a bare term like "girlnextdoor" or "trees" is a text
search, exactly as a search box owes its user. The only failure mode left
is AST validation itself, which cannot fire for anything this module emits
short of a bug.

Quoted substrings are pulled out into placeholder tokens first so their
contents can never be mis-tokenized; an unconsumed quote becomes an exact
text phrase. Consecutive leftover words (only whitespace/stopwords between
them) join into one phrase, so "photos of trees" searches the phrase
"trees" alongside type=image rather than three unrelated fragments.

The outcome's raw payload carries `interpretation` -- humanized chips like
"type = image" / "text ~ trees" -- for UIs that must explain the parse
without ever showing SQL, and `text_terms`, which the app's fusion
endpoint uses to decide whether the local nl2sql model (parsers/nl2sql,
through the sqlexec sandbox) should answer instead of the rules.
"""

from __future__ import annotations

import calendar
import json
import re
import time
from datetime import date, timedelta
from typing import Any

from omniquery.parsers import ParserBackend, ParserOutcome, try_validate

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Surface phrase -> AST 'type' enum value.
_TYPE_SYNONYMS: dict[str, str] = {
    "animated images": "animated_image", "animated image": "animated_image",
    "photos": "image", "photo": "image", "pictures": "image", "picture": "image",
    "pics": "image", "pic": "image",
    "images": "image", "image": "image",
    "videos": "video", "video": "video", "clips": "video", "clip": "video",
    "movies": "video", "movie": "video", "vids": "video", "vid": "video",
    "gifs": "animated_image", "gif": "animated_image",
    "sounds": "audio", "sound": "audio", "music": "audio",
    "songs": "audio", "song": "audio", "audio": "audio",
    "documents": "document", "document": "document", "pdfs": "document", "pdf": "document",
}

# Surface word -> filename suffixes: "pngs" means files whose name ends
# .png. Families with two spellings on disk carry both suffixes. gif and
# pdf stay type synonyms above (the broader semantic).
_EXT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "pngs": (".png",), "png": (".png",),
    "jpgs": (".jpg", ".jpeg"), "jpg": (".jpg", ".jpeg"),
    "jpegs": (".jpg", ".jpeg"), "jpeg": (".jpg", ".jpeg"),
    "webps": (".webp",), "webp": (".webp",),
    "heics": (".heic",), "heic": (".heic",),
    "bmps": (".bmp",), "bmp": (".bmp",),
    "svgs": (".svg",), "svg": (".svg",),
    "tiffs": (".tif", ".tiff"), "tiff": (".tif", ".tiff"),
    "tifs": (".tif", ".tiff"), "tif": (".tif", ".tiff"),
    "mp4s": (".mp4",), "mp4": (".mp4",),
    "webms": (".webm",), "webm": (".webm",),
    "mkvs": (".mkv",), "mkv": (".mkv",),
    "movs": (".mov",), "mov": (".mov",),
    "avi": (".avi",),
    "mp3s": (".mp3",), "mp3": (".mp3",),
    "wavs": (".wav",), "wav": (".wav",),
    "flacs": (".flac",), "flac": (".flac",),
}


def _ext_condition(key: str) -> dict | None:
    """The AST condition for one extension word: a single name-suffix test,
    or an 'or' group when the family has two on-disk spellings."""
    suffixes = _EXT_SYNONYMS.get(key)
    if not suffixes:
        return None
    conds = [{"field": "name", "op": "suffix", "value": s} for s in suffixes]
    return conds[0] if len(conds) == 1 else {"op": "or", "children": conds}


# Surface phrase -> AST status_flag value (canonical capitalization).
_STATUS_SYNONYMS: dict[str, str] = {
    "needs review": "Review", "in review": "Review",
    "approved": "Approved", "rejected": "Rejected",
    "needs edit": "To Edit", "to edit": "To Edit",
    "selected": "Select",
}

# Surface phrase -> AST review_issue enum value.
_ISSUE_SYNONYMS: dict[str, str] = {
    "anatomy": "anatomy", "artifact": "artifact", "artifacts": "artifact",
    "composition": "composition", "lighting": "lighting",
    "text render": "text_render", "text rendering": "text_render",
    "prompt mismatch": "prompt_mismatch", "style": "style",
    "detail loss": "detail_loss",
}

_MONTHS: dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

# Query scaffolding tokens: never significant on their own, and never worth
# a text search. Anything NOT here that survives the rules becomes a text
# phrase -- so this list is deliberately conservative.
STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "is", "are", "was", "were",
    "be", "been", "being", "with", "to", "for", "and", "or", "that", "this",
    "these", "those", "it", "its", "from", "as", "by", "than", "then", "me",
    "my", "i", "show", "find", "get", "give", "please", "all", "any", "some",
    "want", "need", "list", "search", "query", "which", "what", "files",
    "file", "items", "item", "everything", "things", "thing", "let", "lets",
    "there", "have", "has", "had", "do", "does", "did", "containing",
    "contains", "about", "featuring", "whose", "media", "either", "first",
    "not", "but", "except", "without", "excluding",
    "more", "less", "better", "above", "below", "over", "under",
})

# Quoted substrings are replaced with qzq<N>qzq tokens: stable under
# lowercasing and never a substring of real query vocabulary.
_PLACEHOLDER_FRAGMENT = r"qzq(\d+)qzq"


def _build_alternation(mapping: dict[str, str]) -> re.Pattern:
    """Case-insensitive whole-word alternation over the mapping's keys,
    longest key first; a key's internal spaces match any whitespace run."""
    keys = sorted(mapping.keys(), key=len, reverse=True)
    parts = [re.escape(k).replace(r"\ ", r"\s+") for k in keys]
    return re.compile(r"\b(" + "|".join(parts) + r")\b", re.IGNORECASE)


_TYPE_RE = _build_alternation(_TYPE_SYNONYMS)
_EXT_RE = _build_alternation(_EXT_SYNONYMS)
_STATUS_RE = _build_alternation(_STATUS_SYNONYMS)
_FAVORITE_RE = re.compile(r"\bfavou?rite[sd]?\b|\bfaves?\b", re.IGNORECASE)
_UNFAVORITED_RE = re.compile(r"\bun-?favou?rited\b", re.IGNORECASE)

_MY_RATING_AT_LEAST_RE = re.compile(r"\bi\s+rated?\s+at\s+least\s+(\d+)(?:\s*stars?)?\b", re.IGNORECASE)
_RATING_COUNT_RE = re.compile(r"\brated\s+by\s+at\s+least\s+(\d+)\s+(?:people|users)\b", re.IGNORECASE)
_RATED_BY_USER_RE = re.compile(r"\brated\s+by\s+([a-z0-9_\-]+)\b", re.IGNORECASE)
_COMMENTED_BY_USER_RE = re.compile(r"\bcommented\s+(?:on\s+)?by\s+([a-z0-9_\-]+)\b", re.IGNORECASE)
_RATING_BETWEEN_RE = re.compile(r"\brated?\s+between\s+(\d+)\s+and\s+(\d+)\b", re.IGNORECASE)
_RATING_AT_LEAST_RE = re.compile(r"\brated?\s+at\s+least\s+(\d+)(?:\s*stars?)?\b", re.IGNORECASE)
_RATING_PLUS_RE = re.compile(r"\b(\d+)\s*\+\s*stars?\b", re.IGNORECASE)
_RATING_OR_BETTER_RE = re.compile(r"\b(\d+)\s*stars?\s+or\s+(?:better|more|higher|above)\b", re.IGNORECASE)
_RATING_STARS_RE = re.compile(r"\b(\d+)\s*stars?\b", re.IGNORECASE)
_RATING_EXACT_RE = re.compile(r"\brated\s+(\d+)(?:\s*stars?)?\b", re.IGNORECASE)

_SIZE_OP_MAP = {
    "over": "gt", "bigger than": "gt", "larger than": "gt", "more than": "gt",
    "under": "lt", "smaller than": "lt", "less than": "lt", "at least": "ge",
}
_SIZE_RE = re.compile(
    r"\b(over|under|bigger\s+than|larger\s+than|smaller\s+than|less\s+than|"
    r"more\s+than|at\s+least)\s+(\d+(?:\.\d+)?)\s*(mb|gb|megabytes?|gigabytes?)\b", re.IGNORECASE,
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
    r"(seconds?|secs?|minutes?|mins?|hours?|hrs?)\b", re.IGNORECASE,
)

_MP_OP_MAP = {"over": "gt", "above": "gt", "more than": "gt",
              "under": "lt", "below": "lt", "less than": "lt", "at least": "ge"}
_MEGAPIXELS_RE = re.compile(
    r"\b(?:(over|under|at\s+least|above|below|more\s+than|less\s+than)\s+)?"
    r"(\d+(?:\.\d+)?)\s*(?:megapixels?|mp)\b", re.IGNORECASE,
)

_QUALITY_OP_MAP = {"above": "gt", "over": "gt", "at least": "ge", "below": "lt", "under": "lt"}
_QUALITY_RE = re.compile(r"\bquality\s+(above|over|at\s+least|below|under)\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)

# wider/taller comparisons; "shorter than N" requires an explicit pixel
# unit so it can never shadow duration's "shorter than 2 minutes".
_WIDTH_RE = re.compile(r"\b(wider|narrower)\s+than\s+(\d+)(?:\s*(?:pixels?|px))?\b", re.IGNORECASE)
_HEIGHT_RE = re.compile(r"\btaller\s+than\s+(\d+)(?:\s*(?:pixels?|px))?\b"
                         r"|\bshorter\s+than\s+(\d+)\s*(?:pixels?|px)\b", re.IGNORECASE)
_PATH_RE = re.compile(r"\b(?:under|in)\s+the\s+([a-z0-9_\-/]+)\s+path\b", re.IGNORECASE)
_FACE_CLUSTER_RE = re.compile(r"\bface\s+cluster\s+(\d+)\b", re.IGNORECASE)
_NEAR_DUP_RE = re.compile(r"\bnear[\s-]?duplicates?\s+of\s+([a-z0-9_\-.]+)\b", re.IGNORECASE)
_VISUALLY_SIMILAR_RE = re.compile(r"\bvisually\s+similar\s+to\s+([a-z0-9_\-.]+)\b", re.IGNORECASE)
_SIMILAR_RE = re.compile(r"\bsimilar\s+to\s+([a-z0-9_\-.]+)\b", re.IGNORECASE)
_CAPTION_NULL_RE = re.compile(r"\b(?:without|no)\s+(?:a\s+)?captions?\b", re.IGNORECASE)
_CAPTION_NOT_NULL_RE = re.compile(r"\b(?:with|have|has)\s+a\s+caption\b", re.IGNORECASE)

_SEED_RE = re.compile(r"\bseed\s+(\d+)\b", re.IGNORECASE)
_STEPS_RE = re.compile(r"\b(\d+)\s+steps\b|\bsteps\s+(\d+)\b", re.IGNORECASE)
_CFG_RE = re.compile(r"\bcfg\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)
_MODEL_RE = re.compile(r"\b(?:model|checkpoint)\s+([a-z0-9][a-z0-9_.\-]*)", re.IGNORECASE)
_LORA_RE = re.compile(r"\blora\s+([a-z0-9][a-z0-9_.\-]*)", re.IGNORECASE)
_SAMPLER_RE = re.compile(r"\bsampler\s+([a-z0-9][a-z0-9_.\-]*)", re.IGNORECASE)

_FOLDER_IN_THE_RE = re.compile(r"\bin\s+the\s+([a-z0-9_\-/]+(?:\s[a-z0-9_\-/]+)*?)\s+folder\b", re.IGNORECASE)
_FOLDER_IN_FOLDER_RE = re.compile(r"\bin\s+folder\s+([a-z0-9_\-/]+)\b", re.IGNORECASE)
_FOLDER_UNDER_RE = re.compile(r"\bunder\s+([a-z0-9_\-]+(?:/[a-z0-9_\-]+)*)/", re.IGNORECASE)

_COLLECTION_RE = re.compile(r"\bin\s+the\s+([a-z0-9_\- ]+?)\s+(?:collection|album)\b", re.IGNORECASE)

_NAME_RE = re.compile(rf"\b(?:named|called)\s+{_PLACEHOLDER_FRAGMENT}\b", re.IGNORECASE)
# Unquoted tails: "named X..." with no (or unbalanced) quotes claims the
# rest of the text verbatim -- SQL-looking literals included; the compiler
# binds every value as a parameter, so they are inert data.
_NAME_TAIL_RE = re.compile(r"\b(?:named|called)\s+(.+)$", re.IGNORECASE)
_PROMPT_MENTIONS_RE = re.compile(rf"\bprompt\s+(?:mentions|contains)\s+{_PLACEHOLDER_FRAGMENT}\b", re.IGNORECASE)
_PROMPT_TAIL_RE = re.compile(r"\bprompt\s+(?:mentions|contains)\s+(.+)$", re.IGNORECASE)
_PROMPT_IN_RE = re.compile(rf"\bwith\s+{_PLACEHOLDER_FRAGMENT}\s+in\s+the\s+prompt\b", re.IGNORECASE)
_CAPTION_RE = re.compile(rf"\bcaption\s+mentions\s+{_PLACEHOLDER_FRAGMENT}\b", re.IGNORECASE)
_COMMENT_MENTIONS_RE = re.compile(rf"\bcomments?\s+mention(?:s|ing)?\s+{_PLACEHOLDER_FRAGMENT}\b", re.IGNORECASE)
_COMMENTED_RE = re.compile(r"\bcommented\b", re.IGNORECASE)

_WORKFLOW_WITH_RE = re.compile(r"\b(?:with|have|has)\s+(?:a\s+)?workflows?(?:\s+data)?\b", re.IGNORECASE)
_WORKFLOW_WITHOUT_RE = re.compile(r"\b(?:without|no)\s+workflows?(?:\s+data)?\b", re.IGNORECASE)
_FACES_WITH_RE = re.compile(r"\bwith\s+faces?\b", re.IGNORECASE)
_FACES_WITHOUT_RE = re.compile(r"\b(?:without|no)\s+faces?\b", re.IGNORECASE)

_ISSUE_RE = re.compile(r"\bwith\s+([a-z]+(?:\s[a-z]+)?)\s+issues?\b", re.IGNORECASE)

_COUNT_META_RE = re.compile(r"\bhow\s+many\b|\bcount\s+of\b|\bnumber\s+of\b", re.IGNORECASE)
_NEWEST_FIRST_RE = re.compile(r"\b(?:newest|latest)(?:\s+first)?\b", re.IGNORECASE)
_OLDEST_RE = re.compile(r"\boldest\b", re.IGNORECASE)
_LARGEST_RE = re.compile(r"\blargest\b", re.IGNORECASE)
_BEST_RATED_RE = re.compile(r"\bbest\s+rated\b", re.IGNORECASE)
_TOP_N_RE = re.compile(r"\b(?:top|first)\s+(\d+)\b", re.IGNORECASE)

_LAST_N_RE = re.compile(r"\b(?:last|past)\s+(\d+)\s+(days?|weeks?|months?)\b", re.IGNORECASE)
_LAST_UNIT_RE = re.compile(r"\b(?:last|past)\s+(day|week|month|year)\b", re.IGNORECASE)
_YESTERDAY_RE = re.compile(r"\byesterday\b", re.IGNORECASE)
_TODAY_RE = re.compile(r"\btoday\b", re.IGNORECASE)
_THIS_WEEK_RE = re.compile(r"\bthis\s+week\b", re.IGNORECASE)
_THIS_MONTH_RE = re.compile(r"\bthis\s+month\b", re.IGNORECASE)
_FROM_MONTH_YEAR_RE = re.compile(r"\bfrom\s+([a-z]+)\s+(\d{4})\b", re.IGNORECASE)
_IN_YEAR_RE = re.compile(r"\bin\s+(\d{4})\b", re.IGNORECASE)
_SINCE_AFTER_RE = re.compile(r"\b(?:since|after)\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
_BEFORE_RE = re.compile(r"\bbefore\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)
_BETWEEN_DATES_RE = re.compile(r"\bbetween\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})\b", re.IGNORECASE)

_NEGATION_TRIGGER_RE = re.compile(r"\b(not|except|without|excluding|anything\s+but)\b", re.IGNORECASE)
_NEGATION_BOUNDARY_RE = re.compile(r",|;|\band\b|\bor\b")
_OR_TOKEN_RE = re.compile(r"\bor\b", re.IGNORECASE)
# Paired quotes only: the opening char must be matched by the SAME char,
# so "'; DROP ..." (double-quoted with an internal apostrophe) extracts
# the full literal instead of stopping at the inner quote.
_QUOTE_EXTRACT_RE = re.compile(r'(["\'])((?:(?!\1).)+)\1')
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PLACEHOLDER_RE = re.compile(_PLACEHOLDER_FRAGMENT)

# Structural vocabulary that, when it survives inside a leftover text
# phrase, suggests the rules missed real structure ("videos shorter then
# 2 min" with a typo, "between last tuesday and friday", ...). The
# escalation layer offers such queries to the local nl2sql model.
_STRUCT_HINT_RE = re.compile(
    r"\d|\b(than|least|between|before|after|since|during|until|newer|older|"
    r"bigger|smaller|longer|shorter|mb|gb|kb|seconds?|minutes?|hours?|days?|"
    r"weeks?|months?|years?|stars?|rated|january|jan|february|feb|march|mar|"
    r"april|apr|may|june|jun|july|jul|august|aug|september|sep|sept|october|"
    r"oct|november|nov|december|dec)\b", re.IGNORECASE)

# Field -> short human label for interpretation chips (UI never sees SQL;
# these chips ARE the explanation of what will run).
_CHIP_LABELS: dict[str, str] = {
    "type": "type", "status_flag": "status", "is_favorite": "favorite",
    "has_workflow": "workflow", "has_faces": "faces", "rating_avg": "rating",
    "size_mb": "size MB", "size_bytes": "size", "duration_seconds": "duration s",
    "megapixels": "megapixels", "mtime": "date", "folder": "folder",
    "collection": "collection", "name": "name", "workflow_prompt": "prompt",
    "ai_caption": "caption", "comment_contains": "comment",
    "comment_count": "comments", "review_quality": "quality",
    "review_issue": "issue", "text": "text",
    "gen_seed": "seed", "gen_steps": "steps", "gen_cfg": "cfg",
    "gen_model": "model", "gen_lora": "lora", "gen_sampler": "sampler",
}
_CHIP_OPS: dict[str, str] = {
    "eq": "=", "ne": "≠", "lt": "<", "le": "≤", "gt": ">",
    "ge": "≥", "contains": "~", "between": "between", "prefix": "starts",
    "suffix": "ends",
}


# ---------------------------------------------------------------------------
# Rule application machinery
# ---------------------------------------------------------------------------

def _apply_rule(text: str, consumed: list[bool], pattern: re.Pattern,
                 builder) -> list[tuple[int, int, Any]]:
    """Every non-overlapping (with already-consumed spans) match of
    `pattern`, run through `builder(match)`; a None result means "matched
    the shape but not a value we recognize" and consumes nothing."""
    hits: list[tuple[int, int, Any]] = []
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


def _and_or_single(conds: list[dict]) -> dict | None:
    """Collapse a condition list to one where-node: None when empty, the
    lone condition itself, else an 'and' group -- duplicates dropped."""
    deduped: list[dict] = []
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


def _chip(cond: dict) -> dict | None:
    """One humanized interpretation chip for a condition (or 'not' node)."""
    if cond.get("op") == "not":
        inner = _chip(cond.get("child") or {})
        if inner is None:
            return None
        return {"label": "not " + inner["label"], "field": inner.get("field")}
    if cond.get("op") == "or":
        inners = [c for ch in (cond.get("children") or []) if (c := _chip(ch)) is not None]
        if not inners:
            return None
        return {"label": " or ".join(c["label"] for c in inners),
                "field": inners[0].get("field")}
    field = cond.get("field")
    label = _CHIP_LABELS.get(field, field)
    op = _CHIP_OPS.get(cond.get("op"), cond.get("op"))
    value = cond.get("value")
    if isinstance(value, dict):
        if "days_ago" in value:
            rendered = f"last {int(value['days_ago'])} days"
            return {"label": f"{label}: {rendered}", "field": field}
        rendered = json.dumps(value)
    elif isinstance(value, list):
        rendered = " .. ".join(str(v) for v in value)
    elif value is True:
        return {"label": label, "field": field}
    elif value is False:
        return {"label": f"not {label}", "field": field}
    else:
        rendered = str(value)
    return {"label": f"{label} {op} {rendered}", "field": field}


class NlqParser(ParserBackend):
    """The always-answering deterministic parser; see the module docstring
    for the consume-or-search contract."""

    name = "nlq"

    def parse(self, text: str, now_epoch: float) -> ParserOutcome:
        t0 = time.monotonic()
        placeheld, quotes = _extract_quotes(text)
        working = placeheld.lower()
        consumed = [False] * len(working)

        spanned_conds: list[tuple[int, int, dict]] = []
        meta: dict[str, Any] = {}

        def _quote_builder(field: str):
            def builder(m: re.Match) -> dict | None:
                entry = quotes.get(int(m.group(1)))
                if entry is None:
                    return None
                return {"field": field, "op": "contains", "value": entry[1]}
            return builder

        def _orig_slice(m: re.Match, group: int = 1) -> str:
            return placeheld[m.start(group):m.end(group)].strip()

        def _tail_builder(field: str):
            """Unquoted-tail capture: the original-case rest of the text,
            placeholders restored verbatim (quote chars included)."""
            def builder(m: re.Match) -> dict | None:
                value = _restore_placeholders(_orig_slice(m), quotes)
                return {"field": field, "op": "contains", "value": value} if value else None
            return builder

        # -- booleans with their own negative forms first ------------------
        spanned_conds += _apply_rule(working, consumed, _WORKFLOW_WITH_RE,
                                      lambda _m: {"field": "has_workflow", "op": "eq", "value": True})
        spanned_conds += _apply_rule(working, consumed, _WORKFLOW_WITHOUT_RE,
                                      lambda _m: {"field": "has_workflow", "op": "eq", "value": False})
        spanned_conds += _apply_rule(working, consumed, _FACES_WITH_RE,
                                      lambda _m: {"field": "has_faces", "op": "eq", "value": True})
        spanned_conds += _apply_rule(working, consumed, _FACES_WITHOUT_RE,
                                      lambda _m: {"field": "has_faces", "op": "eq", "value": False})
        spanned_conds += _apply_rule(
            working, consumed, _UNFAVORITED_RE,
            lambda _m: {"op": "not", "child": {"field": "is_favorite", "op": "eq", "value": True}},
        )
        spanned_conds += _apply_rule(
            working, consumed, _CAPTION_NULL_RE,
            lambda _m: {"field": "ai_caption", "op": "is_null"})
        spanned_conds += _apply_rule(
            working, consumed, _CAPTION_NOT_NULL_RE,
            lambda _m: {"field": "ai_caption", "op": "not_null"})

        # -- ratings by ME / by count / by user run BEFORE the generic star
        # rules, whose broader patterns would otherwise claim their spans --
        spanned_conds += _apply_rule(working, consumed, _MY_RATING_AT_LEAST_RE,
                                      lambda m: {"field": "my_rating", "op": "ge", "value": int(m.group(1))})
        spanned_conds += _apply_rule(working, consumed, _RATING_COUNT_RE,
                                      lambda m: {"field": "rating_count", "op": "ge", "value": int(m.group(1))})
        spanned_conds += _apply_rule(working, consumed, _RATED_BY_USER_RE,
                                      lambda m: {"field": "rated_by_user", "op": "eq", "value": _orig_slice(m)})
        spanned_conds += _apply_rule(working, consumed, _COMMENTED_BY_USER_RE,
                                      lambda m: {"field": "commented_by_user", "op": "eq", "value": _orig_slice(m)})
        spanned_conds += _apply_rule(
            working, consumed, _RATING_BETWEEN_RE,
            lambda m: {"field": "rating_avg", "op": "between",
                       "value": [int(m.group(1)), int(m.group(2))]})

        # -- star ratings (protects "N stars or better"'s "or") ------------
        spanned_conds += _apply_rule(working, consumed, _RATING_AT_LEAST_RE,
                                      lambda m: {"field": "rating_avg", "op": "ge", "value": int(m.group(1))})
        spanned_conds += _apply_rule(working, consumed, _RATING_OR_BETTER_RE,
                                      lambda m: {"field": "rating_avg", "op": "ge", "value": int(m.group(1))})
        spanned_conds += _apply_rule(working, consumed, _RATING_PLUS_RE,
                                      lambda m: {"field": "rating_avg", "op": "ge", "value": int(m.group(1))})
        spanned_conds += _apply_rule(working, consumed, _RATING_EXACT_RE,
                                      lambda m: {"field": "rating_avg", "op": "eq", "value": int(m.group(1))})
        spanned_conds += _apply_rule(working, consumed, _RATING_STARS_RE,
                                      lambda m: {"field": "rating_avg", "op": "ge", "value": int(m.group(1))})

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

        # -- pixel dimensions / path / face cluster / similarity refs ------
        spanned_conds += _apply_rule(
            working, consumed, _WIDTH_RE,
            lambda m: {"field": "width",
                       "op": "gt" if m.group(1).lower() == "wider" else "lt",
                       "value": int(m.group(2))})
        spanned_conds += _apply_rule(
            working, consumed, _HEIGHT_RE,
            lambda m: ({"field": "height", "op": "gt", "value": int(m.group(1))}
                       if m.group(1) is not None
                       else {"field": "height", "op": "lt", "value": int(m.group(2))}))
        spanned_conds += _apply_rule(working, consumed, _PATH_RE,
                                      lambda m: {"field": "path", "op": "contains", "value": _orig_slice(m)})
        spanned_conds += _apply_rule(working, consumed, _FACE_CLUSTER_RE,
                                      lambda m: {"field": "face_cluster", "op": "eq", "value": int(m.group(1))})
        spanned_conds += _apply_rule(working, consumed, _NEAR_DUP_RE,
                                      lambda m: {"field": "near_dup_of", "op": "eq", "value": _orig_slice(m)})
        spanned_conds += _apply_rule(working, consumed, _VISUALLY_SIMILAR_RE,
                                      lambda m: {"field": "similar_to_visual", "op": "eq", "value": _orig_slice(m)})
        spanned_conds += _apply_rule(
            working, consumed, _SIMILAR_RE,
            lambda m: {"field": "similar_to_semantic", "op": "eq",
                       "value": {"file_id": _orig_slice(m), "k": 10}})

        # -- typed generation parameters -----------------------------------
        spanned_conds += _apply_rule(working, consumed, _SEED_RE,
                                      lambda m: {"field": "gen_seed", "op": "eq", "value": int(m.group(1))})
        spanned_conds += _apply_rule(
            working, consumed, _STEPS_RE,
            lambda m: {"field": "gen_steps", "op": "eq",
                       "value": int(m.group(1) or m.group(2))})
        spanned_conds += _apply_rule(working, consumed, _CFG_RE,
                                      lambda m: {"field": "gen_cfg", "op": "eq", "value": float(m.group(1))})
        spanned_conds += _apply_rule(working, consumed, _MODEL_RE,
                                      lambda m: {"field": "gen_model", "op": "contains", "value": _orig_slice(m)})
        spanned_conds += _apply_rule(working, consumed, _LORA_RE,
                                      lambda m: {"field": "gen_lora", "op": "contains", "value": _orig_slice(m)})
        spanned_conds += _apply_rule(working, consumed, _SAMPLER_RE,
                                      lambda m: {"field": "gen_sampler", "op": "contains", "value": _orig_slice(m)})

        # -- dates ------------------------------------------------------------
        def _last_n_builder(m: re.Match) -> dict:
            n, unit = int(m.group(1)), m.group(2).lower()
            mult = 7 if unit.startswith("week") else 30 if unit.startswith("month") else 1
            return {"field": "mtime", "op": "ge", "value": {"days_ago": n * mult}}

        spanned_conds += _apply_rule(working, consumed, _LAST_N_RE, _last_n_builder)

        def _last_unit_builder(m: re.Match) -> dict:
            unit = m.group(1).lower()
            days = {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
            return {"field": "mtime", "op": "ge", "value": {"days_ago": days}}

        spanned_conds += _apply_rule(working, consumed, _LAST_UNIT_RE, _last_unit_builder)

        local_today = date.fromtimestamp(now_epoch)
        yesterday_iso = (local_today - timedelta(days=1)).isoformat()
        week_start = local_today - timedelta(days=local_today.weekday())
        week_start_iso = week_start.isoformat()
        week_end_iso = (week_start + timedelta(days=6)).isoformat()
        month_start_iso = local_today.replace(day=1).isoformat()
        month_end_iso = local_today.replace(
            day=calendar.monthrange(local_today.year, local_today.month)[1]).isoformat()
        spanned_conds += _apply_rule(
            working, consumed, _YESTERDAY_RE,
            lambda _m: {"field": "mtime", "op": "between",
                       "value": [yesterday_iso, yesterday_iso]})
        spanned_conds += _apply_rule(
            working, consumed, _TODAY_RE,
            lambda _m: {"field": "mtime", "op": "between",
                       "value": [local_today.isoformat(), local_today.isoformat()]})
        spanned_conds += _apply_rule(
            working, consumed, _THIS_WEEK_RE,
            lambda _m: {"field": "mtime", "op": "between",
                       "value": [week_start_iso, week_end_iso]})
        spanned_conds += _apply_rule(
            working, consumed, _THIS_MONTH_RE,
            lambda _m: {"field": "mtime", "op": "between",
                       "value": [month_start_iso, month_end_iso]})

        def _from_month_year_builder(m: re.Match) -> dict | None:
            month = _MONTHS.get(m.group(1).lower())
            if month is None:
                return None
            year = int(m.group(2))
            last_day = calendar.monthrange(year, month)[1]
            return {"field": "mtime", "op": "between",
                    "value": [f"{year:04d}-{month:02d}-01",
                              f"{year:04d}-{month:02d}-{last_day:02d}"]}

        spanned_conds += _apply_rule(working, consumed, _FROM_MONTH_YEAR_RE, _from_month_year_builder)
        spanned_conds += _apply_rule(
            working, consumed, _IN_YEAR_RE,
            lambda m: {"field": "mtime", "op": "between",
                       "value": [f"{m.group(1)}-01-01", f"{m.group(1)}-12-31"]})
        spanned_conds += _apply_rule(working, consumed, _SINCE_AFTER_RE,
                                      lambda m: {"field": "mtime", "op": "ge", "value": m.group(1)})
        spanned_conds += _apply_rule(working, consumed, _BEFORE_RE,
                                      lambda m: {"field": "mtime", "op": "lt", "value": m.group(1)})
        spanned_conds += _apply_rule(
            working, consumed, _BETWEEN_DATES_RE,
            lambda m: {"field": "mtime", "op": "between", "value": [m.group(1), m.group(2)]})

        # -- generic negation runs before every positive rule it can wrap:
        # favorite / type / status below, and collection membership here --
        spanned_conds += _try_negation(working, consumed)

        # -- folder / collection / quoted-text targets ------------------------
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
        # Unquoted/unbalanced tails after the quoted forms: whatever follows
        # "named"/"prompt contains" to end of text, restored verbatim.
        spanned_conds += _apply_rule(working, consumed, _NAME_TAIL_RE, _tail_builder("name"))
        spanned_conds += _apply_rule(working, consumed, _PROMPT_TAIL_RE, _tail_builder("workflow_prompt"))
        spanned_conds += _apply_rule(working, consumed, _COMMENTED_RE,
                                      lambda _m: {"field": "comment_count", "op": "gt", "value": 0})

        # -- review issues ------------------------------------------------
        def _issue_builder(m: re.Match) -> dict | None:
            value = _ISSUE_SYNONYMS.get(re.sub(r"\s+", " ", m.group(1).strip().lower()))
            if value is None:
                return None
            return {"field": "review_issue", "op": "eq", "value": value}

        spanned_conds += _apply_rule(working, consumed, _ISSUE_RE, _issue_builder)

        # -- meta: counts and presentation -----------------------------------
        if _apply_rule(working, consumed, _COUNT_META_RE, lambda _m: True):
            meta["result"] = "count"
        if _apply_rule(working, consumed, _NEWEST_FIRST_RE, lambda _m: True):
            meta["order"] = ("mtime", "desc")
        elif _apply_rule(working, consumed, _OLDEST_RE, lambda _m: True):
            meta["order"] = ("mtime", "asc")
        elif _apply_rule(working, consumed, _LARGEST_RE, lambda _m: True):
            meta["order"] = ("size_bytes", "desc")
        elif _apply_rule(working, consumed, _BEST_RATED_RE, lambda _m: True):
            meta["order"] = ("rating_avg", "desc")
        top_hits = _apply_rule(working, consumed, _TOP_N_RE, lambda m: int(m.group(1)))
        if top_hits:
            meta["limit"] = top_hits[0][2]
            meta.setdefault("order", ("mtime", "desc"))

        # -- plain positives (negated forms were claimed above) -------------
        spanned_conds += _apply_rule(working, consumed, _FAVORITE_RE,
                                      lambda _m: {"field": "is_favorite", "op": "eq", "value": True})

        def _type_builder(m: re.Match) -> dict | None:
            key = re.sub(r"\s+", " ", m.group(1).lower())
            value = _TYPE_SYNONYMS.get(key)
            return None if value is None else {"field": "type", "op": "eq", "value": value}

        spanned_conds += _apply_rule(working, consumed, _TYPE_RE, _type_builder)
        spanned_conds += _apply_rule(working, consumed, _EXT_RE,
                                      lambda m: _ext_condition(m.group(1).lower()))

        def _status_builder(m: re.Match) -> dict | None:
            key = re.sub(r"\s+", " ", m.group(1).lower())
            value = _STATUS_SYNONYMS.get(key)
            return None if value is None else {"field": "status_flag", "op": "eq", "value": value}

        spanned_conds += _apply_rule(working, consumed, _STATUS_RE, _status_builder)

        # -- THE CONTRACT: everything left becomes a text search -------------
        # Unconsumed quote placeholders first (exact phrases), then maximal
        # runs of adjacent leftover words joined into phrases.
        text_terms: list[str] = []
        for m in _PLACEHOLDER_RE.finditer(working):
            s, e = m.span()
            if any(consumed[s:e]):
                continue
            entry = quotes.get(int(m.group(1)))
            literal = entry[1] if entry else None
            if not literal:
                continue
            for i in range(s, e):
                consumed[i] = True
            spanned_conds.append((s, e, {"field": "text", "op": "contains", "value": literal}))
            text_terms.append(literal)

        for s, e, phrase in _leftover_phrases(placeheld, working, consumed):
            for i in range(s, e):
                consumed[i] = True
            spanned_conds.append((s, e, {"field": "text", "op": "contains", "value": phrase}))
            text_terms.append(phrase)

        # -- top-level disjunction: an "or" no rule wanted splits the query.
        # Two readings: a full split ("favorite images or approved videos"
        # names two alternative subjects) vs a LOCAL disjunction over the
        # two conditions flanking the "or" ("images that are approved or
        # rejected"). Heuristic: when the right side is a single condition
        # that is not a media-type (a type on the right signals a new
        # subject phrase) and the left has more than one condition, the
        # "or" binds locally to its nearest left neighbor; everything else
        # stays ANDed around it. --------------------------------------------
        or_matches = [m for m in _OR_TOKEN_RE.finditer(working) if not any(consumed[m.start():m.end()])]
        where: dict | None
        if or_matches:
            or_m = or_matches[0]
            left_spanned = sorted(
                [(cs, ce, c) for (cs, ce, c) in spanned_conds if ce <= or_m.start()],
                key=lambda t: t[0])
            left = [c for (_, _, c) in left_spanned]
            right = [c for (cs, ce, c) in spanned_conds if cs >= or_m.end()]
            if left and right:
                right_is_new_subject = any(
                    isinstance(c, dict) and c.get("field") == "type" for c in right)
                if len(right) == 1 and len(left) > 1 and not right_is_new_subject:
                    rest = left[:-1]
                    local = {"op": "or", "children": [left[-1], right[0]]}
                    where = _and_or_single([*rest, local])
                else:
                    where = {"op": "or", "children": [_and_or_single(left), _and_or_single(right)]}
            else:
                where = _and_or_single([c for (_, _, c) in spanned_conds])
        else:
            where = _and_or_single([c for (_, _, c) in spanned_conds])

        ast_dict: dict[str, Any] = {"result": meta.get("result", "ids")}
        if where is not None:
            ast_dict["where"] = where
        if "order" in meta:
            field_name, direction = meta["order"]
            ast_dict["order_by"] = [{"field": field_name, "dir": direction}]
        if "limit" in meta:
            ast_dict["limit"] = meta["limit"]

        latency_ms = (time.monotonic() - t0) * 1000.0
        query, err = try_validate(ast_dict)
        if err is not None:
            # Should be unreachable for anything this module builds; still,
            # never raise out of parse().
            return ParserOutcome(ast=None, confidence=None, backend=self.name, unsupported=True,
                                  reason=err, coverage=None, latency_ms=latency_ms)

        chips = [c for (_, _, cond) in spanned_conds if (c := _chip(cond)) is not None]
        if meta.get("result") == "count":
            chips.append({"label": "count", "field": None})
        if "order" in meta:
            chips.append({"label": f"sort: {meta['order'][0]} {meta['order'][1]}", "field": None})
        if "limit" in meta:
            chips.append({"label": f"limit {meta['limit']}", "field": None})

        model_hint = any(_STRUCT_HINT_RE.search(t) for t in text_terms)
        return ParserOutcome(
            ast=query.to_dict(), confidence=1.0, backend=self.name,
            unsupported=False, reason=None, coverage=1.0, latency_ms=latency_ms,
            raw={"interpretation": chips, "text_terms": text_terms,
                 "model_hint": model_hint},
        )


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------

_NEG_COLLECTION_RE = re.compile(r"in\s+the\s+([a-z0-9_\- ]+?)\s+(?:collection|album)\b", re.IGNORECASE)


def _match_predicate_at_start(s: str) -> tuple[dict, int] | None:
    """One negatable predicate (favorite / media type / status flag /
    collection membership) anchored at the start of `s`, or None."""
    m = _FAVORITE_RE.match(s)
    if m:
        return {"field": "is_favorite", "op": "eq", "value": True}, m.end()
    m = _TYPE_RE.match(s)
    if m:
        value = _TYPE_SYNONYMS.get(re.sub(r"\s+", " ", m.group(1).lower()))
        if value is not None:
            return {"field": "type", "op": "eq", "value": value}, m.end()
    m = _EXT_RE.match(s)
    if m:
        cond = _ext_condition(m.group(1).lower())
        if cond is not None:
            return cond, m.end()
    m = _STATUS_RE.match(s)
    if m:
        value = _STATUS_SYNONYMS.get(re.sub(r"\s+", " ", m.group(1).lower()))
        if value is not None:
            return {"field": "status_flag", "op": "eq", "value": value}, m.end()
    m = _NEG_COLLECTION_RE.match(s)
    if m:
        return {"field": "collection", "op": "eq", "value": m.group(1)}, m.end()
    return None


def _try_negation(text: str, consumed: list[bool]) -> list[tuple[int, int, dict]]:
    """Each unconsumed negation trigger claims the single predicate right
    after it (scope ends at the next comma/semicolon/'and'/'or') and wraps
    it in a 'not' node; unrecognized predicates consume nothing."""
    hits: list[tuple[int, int, dict]] = []
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
# Quote extraction and leftover-phrase assembly
# ---------------------------------------------------------------------------

def _extract_quotes(text: str) -> tuple[str, dict[int, tuple[str, str]]]:
    """Replace each (properly paired) quoted substring with a qzq<N>qzq
    placeholder; quotes[N] = (quote_char, literal) so literals keep their
    exact content, can never be mis-tokenized, and can be restored
    verbatim -- quote characters included -- by _restore_placeholders."""
    quotes: dict[int, tuple[str, str]] = {}
    counter = [0]

    def _replace(m: re.Match) -> str:
        idx = counter[0]
        counter[0] += 1
        quotes[idx] = (m.group(1), m.group(2))
        return f"qzq{idx}qzq"

    return _QUOTE_EXTRACT_RE.sub(_replace, text), quotes


def _restore_placeholders(s: str, quotes: dict[int, tuple[str, str]]) -> str:
    """Inverse of _extract_quotes over a text slice: each placeholder
    becomes its original quoted literal, quote characters and all."""
    def _put_back(m: re.Match) -> str:
        char, literal = quotes.get(int(m.group(1)), ("", ""))
        return f"{char}{literal}{char}"
    return _PLACEHOLDER_RE.sub(_put_back, s)


def _leftover_phrases(placeheld: str, working: str,
                      consumed: list[bool]) -> list[tuple[int, int, str]]:
    """Maximal runs of adjacent unconsumed non-stopword tokens, joined into
    original-case phrases: (start, end, phrase) spans covering exactly the
    tokens, so intervening consumed/stopword spans break the run."""
    tokens = [
        (m.start(), m.end())
        for m in _TOKEN_RE.finditer(working)
        if m.group(0) not in STOPWORDS
        and not any(consumed[m.start():m.end()])
        and not _PLACEHOLDER_RE.fullmatch(working[m.start():m.end()])
    ]
    phrases: list[tuple[int, int, str]] = []
    for s, e in tokens:
        if phrases and working[phrases[-1][1]:s].strip() == "" :
            ps, _, _ = phrases[-1]
            phrases[-1] = (ps, e, placeheld[ps:e].strip())
        else:
            phrases.append((s, e, placeheld[s:e].strip()))
    return [p for p in phrases if p[2]]
