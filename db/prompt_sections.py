"""A prompt text is a document of SECTIONS -- one tool-neutral IR, fed
by per-tool adapters.

    Section(ordinal, kind, spec, text)

`main` is the prompt proper; other kinds are alternate prompts the
generator routes to a stage or an area. Which grammar a text is read
with is decided by the generation's TOOL (db/prompts.py grammar_for),
never guessed from the text: a string is not Swarm syntax because it
contains angle brackets.

GRAMMARS

`plain` -- the whole text is one main section. A1111/Forge weighting,
`BREAK`, `[a|b]` alternation and `<lora:>` references pass through as
text; they are not sections.

`swarm` -- SwarmUI (refs/mcmonkeyprojects/SwarmUI docs/Features/Prompt
Syntax.md; src/Text2Image/T2IPromptHandling.cs:598-646, 734):

    named sections      <base> <refiner> <pixeldecoder> <video> <videoswap>
                        -- the text after the tag goes only to that stage
    confined sections   <segment:args> <object:args> <region:args> <extend:args>
                        -- the text after the tag is an alternate prompt for
                        a found area, a box, or extra video frames; each is
                        its own section, and Swarm appends `//cid=N` to the
                        tag in the EFFECTIVE prompt it stores
    chunk boundary      <break> -- a conditioning split, same section
    dropped             <comment:...>
    references          <lora:...> <embed:...> <embedding:...> <preset:...>
                        <trigger> -- weights and macros, not words; the
                        artifacts are already rows (db/ingest.py)
    expansions          <random:...> <wildcard:...> <setvar:...> <var:...>
                        ... -- left verbatim: in an effective prompt they are
                        already expanded, in an original they ARE the words

Parsing is pure and versioned: a section row records the grammar, the
parser version and the text hash it was read from, so a grammar
improvement is a re-parse of the database, never a relabeling. The
ComfyUI adapter, when it comes, emits the same IR from conditioning
nodes (db/graph.py), and the semantic layer never learns the difference.
"""

from __future__ import annotations

import dataclasses
import re

#: Bump when a grammar change alters which sections a text yields.
VERSION = 1

GRAMMARS = ("plain", "swarm")
NAMED = frozenset({"base", "refiner", "pixeldecoder", "video", "videoswap"})
CONFINED = frozenset({"segment", "object", "region", "extend"})
KINDS = ("main", *sorted(NAMED), *sorted(CONFINED))
_DROPPED = frozenset({"comment"})
_REFERENCES = frozenset({"lora", "embed", "embedding", "preset", "trigger"})
_CID = re.compile(r"//cid=\d+$")
_SPACE = re.compile(r"\s+")


@dataclasses.dataclass(frozen=True)
class Section:
    ordinal: int
    kind: str
    spec: str | None
    text: str


def parse(text: str, grammar: str) -> list[Section]:
    """The sections of one prompt text under one grammar, in order;
    always at least the main section (possibly empty when the prompt is
    all tags)."""
    if grammar == "plain":
        return [Section(0, "main", None, _clean(text))]
    if grammar == "swarm":
        return _swarm(text)
    raise ValueError(f"no prompt grammar named {grammar!r}; one of {', '.join(GRAMMARS)}")


def main(text: str, grammar: str) -> str:
    """The main section's text -- what a prompt-to-prompt comparison is
    about."""
    return parse(text, grammar)[0].text


# --- the Swarm adapter -----------------------------------------------------------


def _tag_end(text: str, start: int) -> int:
    """Index of the `>` closing the tag opened at `start`, honouring
    nesting (`<segment:<var:x>>`), or -1 when the tag never closes."""
    depth = 0
    for at in range(start, len(text)):
        if text[at] == "<":
            depth += 1
        elif text[at] == ">":
            depth -= 1
            if depth == 0:
                return at
    return -1


def _split_tag(inner: str) -> tuple[str, str | None]:
    """(prefix, data): the name before the first top-level colon --
    `[...]` predata and `//cid=N` removed -- and what follows it."""
    depth = 0
    colon = -1
    for at, character in enumerate(inner):
        if character == "<":
            depth += 1
        elif character == ">":
            depth -= 1
        elif character == ":" and depth == 0:
            colon = at
            break
    prefix, data = (inner, None) if colon == -1 else (inner[:colon], inner[colon + 1 :])
    prefix = _CID.sub("", prefix)
    if data is not None:
        data = _CID.sub("", data)
    bracket = prefix.find("[")
    if bracket != -1 and prefix.endswith("]"):
        prefix = prefix[:bracket]
    return prefix.strip().lower(), data


def _clean(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def _swarm(text: str) -> list[Section]:
    sections: list[Section] = []
    kind, spec, held = "main", None, []

    def close() -> None:
        nonlocal held
        body = _clean("".join(held))
        if kind == "main" or body:
            sections.append(Section(len(sections), kind, spec, body))
        held = []

    at = 0
    while at < len(text):
        if text[at] != "<":
            held.append(text[at])
            at += 1
            continue
        end = _tag_end(text, at)
        if end == -1:
            held.append(text[at:])
            break
        verbatim = text[at : end + 1]
        prefix, data = _split_tag(text[at + 1 : end])
        at = end + 1
        if prefix in NAMED or prefix in CONFINED:
            close()
            kind, spec = prefix, (data if prefix in CONFINED else None)
        elif prefix == "break" or prefix in _DROPPED or prefix in _REFERENCES:
            held.append(" ")
        else:
            held.append(verbatim)
    close()
    return sections
