"""One vocabulary, and one definition of what a question already holds.

Three claims, and they are the reason this module exists rather than a
sixth place that knows about filters.

**A registered filter is an offerable filter.** A key the ResultSet can
answer and no surface can name is an invisible feature. `unknown_facets`
is that check, and it fails on the key rather than on a page nobody
built.

**Options are counted with their own dimension REMOVED.** Disjunctive
faceting is the difference between a list a person can broaden their
question from and a list that can only ever narrow it. It is invisible
in a screenshot and obvious in a count, so it is asserted on the count.

**Counting goes through db/resultset.py.** Every number here is taken
through `scope_of`, so a filter surface and the gallery cannot come to
disagree about which media a question holds.

The library is deliberately MIXED: generated stills, a plain photograph
and a real video, because the surface this feeds is cross-media and a
proof built only from generated images would not notice a dimension
that silently assumes one.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import connect, discovery, facets, ingest, resultset, scan, vocabulary

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"
NOW = 1_700_000_000.0


def _recipe(checkpoint: str, lora: str, sampler: str, steps: int) -> str:
    return (
        f"a brass diving helmet at dusk <lora:{lora}:0.35>\n"
        "Negative prompt: blurry\n"
        f"Steps: {steps}, Sampler: {sampler}, CFG scale: 7, Seed: 4242, Size: 832x1216, "
        f"Model: {checkpoint}"
    )


#: Four of one recipe, two of another, so a count can be wrong in a way
#: a test notices -- with one of each, every wrong answer is still 1.
MADE = [
    *[("dreamshaper_8", "filmGrain", "Euler a", 28)] * 4,
    *[("juggernautXL", "detailTweaker", "DPM++ 2M", 20)] * 2,
]


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = tmp_path_factory.mktemp("vocab") / "pics"
    root.mkdir(parents=True)
    for i, (checkpoint, lora, sampler, steps) in enumerate(MADE):
        info = PngInfo()
        info.add_text("parameters", _recipe(checkpoint, lora, sampler, steps))
        Image.new("RGB", (64, 48), (20 + i * 7, 60, 90)).save(root / f"made_{i:02d}.png", pnginfo=info)
    # a photograph: no recipe at all, and the control for every
    # "generated" count below
    Image.new("RGB", (64, 48), (10, 120, 10)).save(root / "taken.png")

    import av

    with av.open(str(root / "clip.mp4"), "w") as container:
        stream = container.add_stream("h264", rate=5)
        stream.width, stream.height = 320, 180
        stream.pix_fmt = "yuv420p"
        for _ in range(5):
            frame = av.VideoFrame.from_ndarray(np.full((180, 320, 3), (0, 0, 255), dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    conn = connect.memory()
    conn.executescript(SCHEMA.read_text(encoding="utf-8"))
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,?,'library',0)", (str(root),))
    scan.scan(conn, 1, root, NOW)
    for file_id, name in conn.execute("SELECT id, name FROM file").fetchall():
        ingest.one(conn, file_id, root / name, NOW)
    conn.commit()
    yield conn
    conn.close()


def _query(**kwargs) -> resultset.GalleryQuery:
    return resultset.parse(**kwargs)


def _total(conn, query) -> int:
    return resultset.describe(conn, "", query, NOW)["total"]


def _by_label(held) -> dict[str, int]:
    return {one.label: one.count for one in held.options}


# --- the vocabulary describes everything the engine can answer --------------


def test_every_registered_filter_has_a_dimension_describing_it(library):
    """A key the ResultSet answers and no surface can name is not a
    secret feature, it is an invisible one."""
    assert vocabulary.unknown_facets() == (), (
        "these facets can be answered but no surface can offer them: " + ", ".join(vocabulary.unknown_facets())
    )


def test_a_dimension_agrees_with_the_facet_it_describes(library):
    """The vocabulary states a key's operators for the surface; the
    registry states them for the engine. Two statements of one fact are
    two chances to be wrong, so they are held against each other."""
    for one in vocabulary.DIMENSIONS:
        if one.carried != "facet":
            continue
        spec = facets.REGISTRY.get(one.key)
        assert spec is not None, f"{one.key} describes a filter the engine does not have"
        assert set(one.ops) <= set(spec.ops), f"{one.key}: the surface offers {one.ops}, the engine allows {spec.ops}"


def test_every_group_a_dimension_names_is_a_real_section(library):
    named = {name for name, _ in vocabulary.GROUPS}
    for one in vocabulary.DIMENSIONS:
        assert one.group in named, f"{one.key} is in group {one.group!r}, which the surface does not have"


def test_a_machine_link_is_labelled_but_never_listed(library):
    """A timeline bar opens a window of epoch seconds at a granule. That
    chip has to read as words, and no list of epoch seconds is a filter
    anybody picks from -- so it is described and not offered."""
    listed = {one.key for one in vocabulary.offered()}
    for key in ("context.moment", "context.granule", "event.id"):
        one = vocabulary.dimension(key)
        assert one is not None, f"{key} has no dimension, so its chip would print the raw key"
        assert key not in listed, f"{key} is a machine's arithmetic; no one picks it off a list"
    assert vocabulary.chip(vocabulary.BY_KEY["context.moment"], "gte", 1700000000) == "moment from 1700000000"

    for _name, _label, held in vocabulary.grouped():
        for one in held:
            assert one.offered, f"{one.key} reached the filter surface and should not have"


def test_the_sections_offered_follow_the_medium_being_asked_about(library):
    """`kind=audio` must not offer aperture and width. A section whose
    every dimension is inapplicable is not shown at all -- an empty
    heading is a promise the application cannot keep."""
    everything = {name for name, _, _ in vocabulary.grouped()}
    assert "camera" in everything
    assert "media" in everything

    for_sound = {name: [one.key for one in held] for name, _, held in vocabulary.grouped("audio")}
    assert "media.width" not in for_sound.get("media", []), "a sound has no pixels"
    assert "media.duration" in for_sound.get("media", []), "a sound has a length"

    for_stills = {name: [one.key for one in held] for name, _, held in vocabulary.grouped("image")}
    assert "media.width" in for_stills["media"]
    assert "media.duration" not in for_stills["media"], "a still picture has no length"


# --- "AI generated" is a fact, not the interpretation's verdict --------------


def test_ai_generated_asks_the_fact_and_not_the_origin(library):
    """`has.generation` exists because `context.origin` cannot answer
    this. Origin has a fourth value, `mixed`, for a file carrying both
    capture and generation evidence -- and because repeated facets are
    ANDed, `origin=generated` plus `origin=mixed` is not an OR either.

    It also answers before the context job has run, which origin cannot:
    the generation row is written by ingest. This library has never had
    a context pass, so the two are measurably different here.
    """
    generated = _query(facets=["has.generation:eq:1"])
    assert _total(library, generated) == len(MADE)

    plain = _query(facets=["has.generation:eq:0"])
    # the photograph and the clip: everything with no recipe
    assert _total(library, plain) == 2

    # and the whole library is the two together, with nothing lost between
    assert _total(library, _query()) == len(MADE) + 2

    # the control: origin answers NOTHING here, because no context job
    # has run -- which is exactly the case `has.generation` exists for
    assert _total(library, _query(facets=["context.origin:eq:generated"])) == 0


# --- the resources, by role, and they compose -------------------------------


def test_a_checkpoint_and_a_lora_compose_into_one_question(library):
    """The `artifact` scope holds exactly one, so "this checkpoint with
    that LoRA" was unaskable. As facets they conjoin."""
    checkpoints = discovery.options(library, _query(), "generation.checkpoint")
    assert _by_label(checkpoints) == {"dreamshaper_8": 4, "juggernautXL": 2}

    loras = discovery.options(library, _query(), "generation.lora")
    assert _by_label(loras) == {"filmGrain": 4, "detailTweaker": 2}

    chosen = next(one for one in checkpoints.options if one.label == "dreamshaper_8")
    with_lora = next(one for one in loras.options if one.label == "filmGrain")
    both = _query(facets=[f"generation.checkpoint:eq:{chosen.value}", f"generation.lora:eq:{with_lora.value}"])
    assert _total(library, both) == 4

    other = next(one for one in loras.options if one.label == "detailTweaker")
    crossed = _query(facets=[f"generation.checkpoint:eq:{chosen.value}", f"generation.lora:eq:{other.value}"])
    assert _total(library, crossed) == 0, "no picture used dreamshaper with the other LoRA"


# --- disjunctive faceting ---------------------------------------------------


def test_a_dimensions_own_options_are_counted_without_it(library):
    """The assertion the whole surface rests on.

    Counted against the WHOLE question, choosing one sampler collapses
    every other sampler's count to zero, and the list a person opens to
    change their mind can only ever agree with them. Counted against the
    question MINUS this dimension, the other values still say what they
    would give.
    """
    samplers = discovery.options(library, _query(), "generation.sampler")
    assert _by_label(samplers) == {"Euler a": 4, "DPM++ 2M": 2}

    picked = _query(facets=["generation.sampler:eq:Euler a"])
    assert _total(library, picked) == 4

    again = discovery.options(library, picked, "generation.sampler")
    assert _by_label(again) == {"Euler a": 4, "DPM++ 2M": 2}, (
        "the other samplers must still say what they would give; this is disjunctive faceting"
    )
    assert [one.chosen for one in again.options if one.label == "Euler a"] == [True]

    # and a DIFFERENT dimension IS narrowed by it, which is the other
    # half of the rule -- otherwise every count would just be the library
    checkpoints = discovery.options(library, picked, "generation.checkpoint")
    assert _by_label(checkpoints) == {"dreamshaper_8": 4}


def test_counts_narrow_from_another_dimension(library):
    """A count means "from the rest of this question", so a question
    that already excludes half the library says so."""
    everything = discovery.options(library, _query(), "generation.lora")
    assert _by_label(everything) == {"filmGrain": 4, "detailTweaker": 2}

    narrowed = discovery.options(library, _query(facets=["generation.steps:eq:20"]), "generation.lora")
    assert _by_label(narrowed) == {"detailTweaker": 2}


# --- cross-media ------------------------------------------------------------


def test_the_kind_dimension_counts_every_medium_the_library_holds(library):
    # `media.kind`, not the `kind` scope: the scope is what old links
    # carry and is still answered, but a scope holds one value and the
    # surface needs one that can be OR'd.
    held = discovery.options(library, _query(), "media.kind")
    assert _by_label(held) == {"image": len(MADE) + 1, "video": 1}


def test_which_dimensions_apply_follows_either_spelling_of_the_kind(library):
    """A sound has no aperture, and the question can say which medium in
    two places now. Both have to reach the surface's own decision."""
    by_scope = {
        one.key for _, _, held in vocabulary.grouped(discovery.asked_kind(_query(kind="image"))) for one in held
    }
    by_facet = {
        one.key
        for _, _, held in vocabulary.grouped(discovery.asked_kind(_query(facets=["media.kind:any:image"])))
        for one in held
    }
    assert by_scope == by_facet
    assert "media.duration" not in by_scope, "a still picture has no length, however the question spelled it"

    # two kinds OR'd is not one medium, so everything either of them
    # carries still applies
    both = discovery.asked_kind(_query(facets=["media.kind:any:image", "media.kind:any:video"]))
    assert both is None
    everything = {one.key for _, _, held in vocabulary.grouped(both) for one in held}
    assert "media.duration" in everything


def test_a_dimension_that_is_not_a_fact_about_a_medium_is_not_offered(library):
    """An audio file has no LoRA and no aperture. Offering those under
    `kind=audio` is offering a filter whose every answer is empty, which
    reads as a broken library rather than an inapplicable question."""
    length = vocabulary.dimension("media.duration")
    width = vocabulary.dimension("media.width")
    lora = vocabulary.dimension("generation.lora")
    assert length is not None
    assert width is not None
    assert lora is not None

    assert vocabulary.applies_to(length, "video")
    assert not vocabulary.applies_to(length, "image"), "a still picture has no length"
    assert vocabulary.applies_to(width, "image")
    assert not vocabulary.applies_to(width, "audio"), "a sound has no pixels"
    # a LoRA names no kinds, so it is a question about any of them: a
    # generated video is as real as a generated picture
    assert vocabulary.applies_to(lora, "video")
    assert vocabulary.applies_to(lora, "image")

    # and `kind=None` offers everything, because the answer may hold it
    assert vocabulary.applies_to(length, None)


def test_a_video_composes_with_every_other_dimension(library):
    """The gallery's own filters are cross-media or they are a feature
    for one file type. `kind` and a media dimension conjoin."""
    clips = _query(kind="video")
    assert _total(library, clips) == 1
    assert _total(library, _query(kind="video", facets=["media.duration:gte:0.5"])) == 1
    assert _total(library, _query(kind="video", facets=["media.duration:gte:600"])) == 0
    # a still picture is not a member of a duration question, and that
    # is the honest answer rather than treating its absent length as 0
    assert _total(library, _query(kind="image", facets=["media.duration:gte:0"])) == 0


# --- what a chip says -------------------------------------------------------


def test_a_chip_reads_as_a_person_says_it(library):
    lora = vocabulary.dimension("generation.lora")
    rating = vocabulary.dimension("rating_min")
    made = vocabulary.dimension("has.generation")
    assert lora is not None
    assert rating is not None
    assert made is not None

    assert vocabulary.chip(lora, "eq", 41, {41: "detailTweaker"}) == "LoRA detailTweaker"
    assert vocabulary.chip(rating, "gte", 4) == "rating from 4"
    assert vocabulary.chip(made, "eq", 1) == "AI generated yes"
    # an entity that has since gone says so rather than printing a number
    assert "gone" in vocabulary.chip(lora, "eq", 999, {})


def test_removing_one_dimension_leaves_the_rest_of_the_question(library):
    """A chip's remove link and a dimension's option counts must mean
    the same thing by "without this filter", so they are one function."""
    asked = _query(kind="image", facets=["has.generation:eq:1", "generation.sampler:eq:Euler a"])
    assert discovery.without(asked, "generation.sampler").facets == (facets.facet("has.generation", "eq", "1"),)
    assert discovery.without(asked, "kind").kind is None
    assert discovery.without(asked, "kind").facets == asked.facets


def test_the_badge_counts_the_clauses_the_question_carries(library):
    asked = _query(kind="image", rating_min=3, facets=["has.generation:eq:1", "generation.lora:eq:7"])
    assert discovery.counts(asked) == {
        "kind": 1,
        "rating_min": 1,
        "has.generation": 1,
        "generation.lora": 1,
    }
    assert discovery.counts(_query()) == {}


# --- saying more than one thing about one dimension -------------------------


def test_repeating_a_key_with_any_means_or(library):
    """The gap the whole multi-select feature exists to close.

    Repeated facets conjoin, which is right for "this checkpoint with
    that LoRA" and catastrophic for "image or video": a file cannot be
    two kinds at once, so the AND reading answers nothing, every time,
    for the most ordinary multi-select there is.
    """
    both = _query(facets=["media.kind:any:image", "media.kind:any:video"])
    assert _total(library, both) == _total(library, _query()), "every file is one kind or the other"

    stills = _query(facets=["media.kind:any:image"])
    assert _total(library, stills) == len(MADE) + 1

    # and the AND reading is still available, and still says nothing --
    # which is correct, not a bug: no file is two kinds
    impossible = _query(facets=["media.kind:eq:image", "media.kind:eq:video"])
    assert _total(library, impossible) == 0


def test_or_and_and_are_both_available_on_a_dimension_that_needs_both(library):
    """A picture carries several LoRAs at once, so "any of these" and
    "all of these" are different questions and both are real."""
    loras = discovery.options(library, _query(), "generation.lora")
    film = next(one for one in loras.options if one.label == "filmGrain")
    detail = next(one for one in loras.options if one.label == "detailTweaker")

    either = _query(facets=[f"generation.lora:any:{film.value}", f"generation.lora:any:{detail.value}"])
    assert _total(library, either) == 6, "four with one, two with the other"

    both = _query(facets=[f"generation.lora:eq:{film.value}", f"generation.lora:eq:{detail.value}"])
    assert _total(library, both) == 0, "no picture in this library used both"


def test_an_or_group_composes_with_everything_else_as_one_clause(library):
    """The group is ONE thing the question says, so it narrows with the
    rest rather than replacing it."""
    checkpoints = discovery.options(library, _query(), "generation.checkpoint")
    dreamshaper = next(one for one in checkpoints.options if one.label == "dreamshaper_8")

    asked = _query(
        facets=[
            "media.kind:any:image",
            "media.kind:any:video",
            f"generation.checkpoint:eq:{dreamshaper.value}",
        ]
    )
    assert _total(library, asked) == 4, "either kind, AND that one checkpoint"


def test_the_vocabulary_says_which_reading_a_dimension_takes(library):
    """Which is right is a fact about the dimension, not a preference:
    a file has one kind and several LoRAs."""
    kind = vocabulary.dimension("media.kind")
    lora = vocabulary.dimension("generation.lora")
    seed = vocabulary.dimension("generation.seed")
    assert kind is not None
    assert lora is not None
    assert seed is not None
    assert kind.multi == "any", "a file is one kind, so OR is the only sane reading"
    assert lora.multi == "both", "a picture carries several, so both readings are real"
    assert seed.multi == "", "one seed made one picture; there is nothing to multi-select"


def test_the_kinds_the_two_modules_know_are_the_same_kinds(library):
    """`db/facets.py` states them rather than importing `resultset`,
    which imports it. Two statements of one fact, held together."""
    assert facets.KINDS == resultset.KINDS


def test_the_planner_states_the_same_vocabularies_the_library_does(library):
    """`db/planning.py` restates two vocabularies and nothing held them.

    The planner imports no database module on purpose (its own docstring
    at `prompt_sections_grammar`), so it states the time bases and the
    media kinds itself. That is the same arrangement as `facets.KINDS`
    above -- but where `facets.KINDS` has the test above holding it,
    these two had nothing, and they drifted:

    `_BASES` held six where `db/context.py TIME_BASES` held seven. The
    seventh was `first_seen`, a rung no code could produce, and the
    planner was the structure that was RIGHT. Nobody noticed, because
    nothing compared them.

    Stating a fact twice is allowed here. Stating it twice unheld is what
    let a dead value sit in the schema for as long as it did.
    """
    from db import context, planning

    assert frozenset(context.TIME_BASES) == planning._BASES
    assert frozenset(facets.KINDS) == planning._MEDIA_KINDS
