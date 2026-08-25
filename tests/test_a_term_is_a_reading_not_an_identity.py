"""What recurs across an answer, and what that number is worth.

The analysis counts exact prompts, and that is a FACT: `prompt.text_hash`
is a real identity, two files carrying one prompt share one row, and
"twelve files used this prompt" is a count.

Recurring TERMS is a different claim with a different error mode, which
is why it was deliberately absent rather than quietly mixed into the
exact counts. It assumes commas separate terms -- a convention every
diffusion UI's prompt box follows and no grammar enforces -- so a prompt
written as a sentence splits into clauses and is counted as terms nobody
asked for.

Both panels, side by side, each saying what kind of thing it is. The
reading does not get to borrow the fact's certainty.
"""

from __future__ import annotations

import uuid

import pytest

from db import analysis, resultset
from tests.staging import fresh_schema

pytestmark = pytest.mark.slow

NOW = 1_700_000_000.0


def _library(prompts: list[str]):
    """One file per prompt, generated, with that prompt as its effective one."""
    conn = fresh_schema()
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'C:/x','library',0)")
    conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(1,?,'folder','x')", (uuid.uuid4().bytes,))
    conn.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(1,1,NULL,'x',0)")
    at = 2
    for said in prompts:
        conn.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(?,?,'file',?)", (at, uuid.uuid4().bytes, f"f{at}"))
        conn.execute(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,first_seen_at,last_seen_at)"
            " VALUES(?,1,?,'image',1,0,0,0)",
            (at, f"f{at}.png"),
        )
        conn.execute(
            "INSERT INTO generation(file_id,tool,detection,parser,parsed_at) VALUES(?, 'test', 'marker', 'test', 0)",
            (at,),
        )
        held = conn.execute("SELECT id FROM prompt WHERE text = ?", (said,)).fetchone()
        if held is None:
            conn.execute(
                "INSERT INTO entity(id,uuid,kind,slug) VALUES(?,?,'prompt',?)",
                (1000 + at, uuid.uuid4().bytes, f"p{at}"),
            )
            conn.execute(
                "INSERT INTO prompt(id,text,text_hash,created_at) VALUES(?,?,?,0)", (1000 + at, said, f"h{at}")
            )
            prompt_id = 1000 + at
        else:
            prompt_id = held[0]
        conn.execute("INSERT INTO generation_prompt(file_id,role,prompt_id) VALUES(?, 'effective', ?)", (at, prompt_id))
        at += 1
    conn.commit()
    return conn


def _counted(conn) -> dict[str, int]:
    held, _ = analysis.terms(conn, resultset.parse(), now=NOW)
    return {one.term: one.files for one in held}


# --- the reading, in one place so it can be disagreed with -------------------


def test_commas_separate_terms():
    conn = _library(["rim light, sunset, portrait", "sunset, wide angle"])
    try:
        assert _counted(conn) == {"sunset": 2, "portrait": 1, "rim light": 1, "wide angle": 1}
    finally:
        conn.close()


def test_a_weight_is_not_part_of_the_term():
    """`(rim light:1.3)` is the same thing asked for as `rim light`,
    more loudly."""
    conn = _library(["(rim light:1.3), sunset", "rim light, ((sunset))"])
    try:
        assert _counted(conn) == {"rim light": 2, "sunset": 2}
    finally:
        conn.close()


def test_an_artifact_reference_is_not_a_term():
    """`<lora:film:0.8>` names an artifact, and the LoRA panel already
    counts those WITH the strengths they were used at. Counting it here
    too would report one fact twice under two kinds of claim."""
    conn = _library(["sunset, <lora:filmGrain:0.35>", "sunset"])
    try:
        assert _counted(conn) == {"sunset": 2}
    finally:
        conn.close()


def test_case_is_not_meaning():
    conn = _library(["Rim Light, sunset", "rim light"])
    try:
        assert _counted(conn) == {"rim light": 2, "sunset": 1}
    finally:
        conn.close()


def test_a_term_repeated_in_one_prompt_is_one_file():
    """FILES, never uses. A term written three times in one prompt is one
    file that wanted it, and counting the repeats would make a habit of
    typing look like a habit of generating."""
    conn = _library(["sunset, sunset, sunset", "sunset"])
    try:
        assert _counted(conn) == {"sunset": 2}
    finally:
        conn.close()


def test_a_sentence_prompt_is_one_term_which_is_the_reading_being_wrong():
    """The error mode, pinned rather than hidden.

    A prompt written as prose has no commas, so the whole sentence is
    one "term". That is what the assumption costs, it is visible in the
    panel, and the panel says what it assumes -- which is the difference
    between a reading and a lie.
    """
    conn = _library(["a castle on a hill", "a castle on a hill"])
    try:
        assert _counted(conn) == {"a castle on a hill": 2}
    finally:
        conn.close()


def test_empty_fragments_are_not_terms():
    """Trailing commas are how everybody's prompt ends."""
    conn = _library(["sunset, , portrait,", "sunset"])
    try:
        assert _counted(conn) == {"sunset": 2, "portrait": 1}
    finally:
        conn.close()


def test_an_unbalanced_bracket_does_not_hang():
    """A prompt is somebody's text. The weight-stripping loop is bounded
    for exactly this."""
    conn = _library(["(((sunset, ((("])
    try:
        held = _counted(conn)
    finally:
        conn.close()
    assert held, "the reading gave up on a prompt somebody actually wrote"


# --- and it stays a different claim from the counts --------------------------


def test_the_analysis_carries_both_apart():
    """Not folded into `prompts`: an exact prompt count is a fact and a
    term count is a reading, and mixing them would let the reading borrow
    the fact's certainty."""
    conn = _library(["rim light, sunset", "rim light, sunset", "wide angle"])
    try:
        told = analysis.analyze(conn, resultset.parse(), 3, now=NOW)
    finally:
        conn.close()

    assert {one.text for one in told.prompts} == {"rim light, sunset", "wide angle"}
    assert {one.uses for one in told.prompts} == {2, 1}
    assert {one.term for one in told.terms} == {"rim light", "sunset", "wide angle"}
    # the exact prompt used twice makes both its terms two files
    assert {one.term: one.files for one in told.terms}["sunset"] == 2


def test_a_reference_glued_to_the_prose_is_still_not_a_term():
    """How A1111 actually writes one.

    The reference does not arrive on its own: it is stuck to the end of
    a comma-free sentence, so a rule that only refused a fragment which
    was ENTIRELY a reference left every one of them inside a term. Found
    by driving the panel over prompts a real parser had read, not by
    these hand-written ones.
    """
    conn = _library(["a castle on a hill <lora:filmGrain:0.35>", "a castle on a hill"])
    try:
        assert _counted(conn) == {"a castle on a hill": 2}
    finally:
        conn.close()
