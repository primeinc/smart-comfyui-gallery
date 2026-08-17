"""Typing a model's name into the model filter must find it.

The filter compared two reductions of the same text that were written
separately. The typed text kept only English letters and digits
(`re.sub(r'[^a-zA-Z0-9]', '', s).lower()`); the stored name kept
everything except ` . _ - / \\ ( ) [ ]`. Any character in the gap between
those two lists -- a plus, an apostrophe, an at sign, a hash, an
exclamation mark, the German sharp s -- was dropped from the needle and
kept in the haystack, so the name could not match itself.

Measured through the page, typing each model's own name:

    niji+animation.safetensors     a plus                 NOT FOUND
    artist's_style.safetensors     an apostrophe          NOT FOUND
    flux@dev.safetensors           an at sign             NOT FOUND
    90s_anime!.safetensors         an exclamation mark    NOT FOUND
    color#grade.safetensors        a hash                 NOT FOUND
    straße.safetensors             German sharp s         NOT FOUND

    14 models, typing the name found 8, missed 6

It is not a corner: the gallery offers these names itself. The box has an
autocomplete backed by workflow_files_suggestions, and choosing an entry
puts the file name into the box (applyWfSuggestion). Driving that whole
loop -- ask for suggestions, take what it offers, search for it -- six of
eleven models were suggested and then found nothing.

Two more things the same fix settles.

The quoted "exact word" form could not match a name with its extension,
because the column turned `.` and `_` into spaces while the typed text
kept them. `"detail_tweaker_xl"` appeared to work only because SQL LIKE
reads `_` as a single-character wildcard, so it was matching `detail
tweaker xl` by accident. Against 100,000 files, `"detail_7"` found 0 of
the 2,000 that use it.

And because the typed text went into LIKE unescaped, `_` and `%` in a
name were wildcards rather than characters.

The fix is one rule instead of two lists: both sides go through the same
Python function, registered on the connection as fuzzykey/wordkey, and
compared with INSTR so nothing in the typed text is read as a pattern.

It costs something, and the something is a Python call per row: at
100,000 files one model filter goes from 0.19s to 0.36s. A cheap SQL
prefilter would win it back and would be the same defect again -- a
second rule that can disagree with the first -- so it is not there.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

import smartgallery


# name as ComfyUI would record it, and what makes it interesting
MODELS = [
    ('detail_tweaker_xl.safetensors', 'underscores'),
    ('add-detail-xl.safetensors', 'hyphens'),
    ('anime style v2.safetensors', 'spaces'),
    ('epiCRealism.safetensors', 'mixed case'),
    ('loras/portrait/skin.safetensors', 'in a subfolder'),
    ('niji+animation.safetensors', 'a plus'),
    ("artist's_style.safetensors", 'an apostrophe'),
    ('flux@dev.safetensors', 'an at sign'),
    ('90s_anime!.safetensors', 'an exclamation mark'),
    ('color#grade.safetensors', 'a hash'),
    ('straße.safetensors', 'German sharp s'),
    ('写真_v1.safetensors', 'Japanese'),
    ('рисунок_v2.safetensors', 'Russian'),
]


@pytest.fixture
def a_library_of_models(smartgallery_app, tmp_path, monkeypatch):
    """A library of its own, one file per model.

    Its own root because the checks below count what a filter returns,
    and files another test left in the shared root would be counted too.
    """
    sg = smartgallery_app
    root = tmp_path / "models_root"
    root.mkdir()
    monkeypatch.setattr(sg, "BASE_OUTPUT_PATH", str(root))

    ids = {}
    with sg.get_db_connection() as conn:
        conn.execute("DELETE FROM files")
        for index, (model, _what) in enumerate(MODELS):
            # exactly what the scan writes: normalize_smart_path
            stored = sg.normalize_smart_path(model)
            path = str(root / ("m%02d.png" % index))
            with open(path, "wb") as handle:
                handle.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
            fid = hashlib.md5(path.encode("utf-8")).hexdigest()
            conn.execute(
                "INSERT OR REPLACE INTO files (id, path, mtime, name, type, "
                "has_workflow, size, workflow_files, last_scanned) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (fid, path, 1.0, "m%02d.png" % index, "image", 1, 64,
                 stored, 1.0))
            ids[model] = fid
        conn.commit()

    client = sg.app.test_client()
    with client.session_transaction() as session:
        session["user_id"] = 1
        session["role"] = "ADMIN"

    def shown_for(typed):
        page = client.get("/galleryout/view/_root_", query_string={
            "workflow_files": typed, "scope": "global", "recursive": "true"},
            follow_redirects=True)
        assert page.status_code == 200, page.status_code
        body = page.get_data(as_text=True)
        return {model for model, fid in ids.items() if fid in body}

    yield client, ids, shown_for

    with smartgallery.get_db_connection() as conn:
        conn.execute("DELETE FROM files")
        conn.commit()


def test_the_filter_is_actually_applied(a_library_of_models):
    """Control. A filter that is quietly ignored shows the whole library,
    and then every check below passes without measuring anything."""
    _client, ids, shown_for = a_library_of_models

    assert shown_for("") == set(ids), "no filter should show every file"
    assert shown_for("zzqqxxnothinghasthis") == set(), \
        "a word in no model should show nothing"


@pytest.mark.parametrize("model,what", MODELS)
def test_typing_a_models_name_finds_it(a_library_of_models, model, what):
    """The defect. Six of these found nothing before."""
    _client, _ids, shown_for = a_library_of_models

    assert model in shown_for(model), (
        f"typing the name of a model with {what} found nothing")


@pytest.mark.parametrize("model,what", MODELS)
def test_the_quoted_form_finds_the_whole_name(a_library_of_models, model, what):
    """The quoted form is documented as exact; an exact name is exact."""
    _client, _ids, shown_for = a_library_of_models

    assert model in shown_for(f'"{model}"'), (
        f'"{model}" ({what}) found nothing')


def test_the_gallery_can_find_what_it_suggests(a_library_of_models):
    """The whole click-through, using the gallery's own answers: ask for
    suggestions, take the file name the page would insert, search."""
    client, _ids, shown_for = a_library_of_models

    offered = client.get("/galleryout/api/workflow_files_suggestions",
                         query_string={"q": ""})
    assert offered.status_code == 200, offered.status_code
    suggestions = json.loads(offered.get_data(as_text=True))["suggestions"]
    assert suggestions, "the gallery suggested nothing at all"

    unfindable = []
    for item in suggestions:
        # applyWfSuggestion puts the file name, not the whole path
        basename = item.replace("\\", "/").rsplit("/", 1)[-1]
        if not shown_for(basename):
            unfindable.append(basename)

    assert not unfindable, (
        f"the gallery offered {len(unfindable)} name(s) and then found "
        f"nothing for them: {unfindable}")


def test_a_quoted_word_still_means_the_whole_word(a_library_of_models):
    """Over-reach guard, and the promise printed under the box: `man`
    finds `woman`, `"man"` finds only `man`."""
    _client, _ids, shown_for = a_library_of_models

    loose = shown_for("detail")
    assert "detail_tweaker_xl.safetensors" in loose
    assert "add-detail-xl.safetensors" in loose

    exact = shown_for('"detail"')
    assert "add-detail-xl.safetensors" in exact, \
        "add-detail-xl has detail as a whole word"
    assert "detail_tweaker_xl.safetensors" in exact, \
        "detail_tweaker_xl has detail as a whole word"
    assert '写真_v1.safetensors' not in exact


def test_a_quoted_word_does_not_match_inside_a_longer_one(a_library_of_models):
    """The other half of that promise, which is the half that can be lost
    by making matching looser."""
    _client, _ids, shown_for = a_library_of_models

    assert "epiCRealism.safetensors" in shown_for("realism")
    assert "epiCRealism.safetensors" not in shown_for('"realism"'), \
        "epicrealism is one word; a quoted realism must not match inside it"


def test_excluding_still_excludes(a_library_of_models):
    """Over-reach guard: `!` is documented and is how people narrow."""
    _client, ids, shown_for = a_library_of_models

    without = shown_for("!detail")
    assert "detail_tweaker_xl.safetensors" not in without
    assert "add-detail-xl.safetensors" not in without
    assert "flux@dev.safetensors" in without


def test_and_and_or_still_mean_what_they_say(a_library_of_models):
    """Over-reach guard for the two separators in the hint."""
    _client, _ids, shown_for = a_library_of_models

    either = shown_for("niji; flux")
    assert "niji+animation.safetensors" in either
    assert "flux@dev.safetensors" in either

    both = shown_for("flux, dev")
    assert both == {"flux@dev.safetensors"}, both


def test_a_file_made_without_any_model_answers_an_exclusion(smartgallery_app,
                                                            a_library_of_models):
    """Over-reach guard: a picture that used no LoRA at all is one of the
    pictures that is not using the one you are excluding."""
    sg = smartgallery_app
    _client, _ids, shown_for = a_library_of_models

    with sg.get_db_connection() as conn:
        path = os.path.join(sg.BASE_OUTPUT_PATH, "plain.png")
        with open(path, "wb") as handle:
            handle.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        fid = hashlib.md5(path.encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT OR REPLACE INTO files (id, path, mtime, name, type, "
            "has_workflow, size, workflow_files, last_scanned) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (fid, path, 1.0, "plain.png", "image", 1, 64, "", 1.0))
        conn.commit()

    page = _client.get("/galleryout/view/_root_", query_string={
        "workflow_files": "!detail", "scope": "global", "recursive": "true"},
        follow_redirects=True)
    assert fid in page.get_data(as_text=True), \
        "a picture with no model recorded was excluded by !detail"


def test_typed_text_is_not_read_as_a_pattern(a_library_of_models):
    """SQL LIKE reads `_` and `%` as wildcards. They were reaching it
    unescaped, so `a_b` matched `axb` and a name containing `%` could not
    be searched for at all."""
    _client, _ids, shown_for = a_library_of_models

    # Under LIKE the `_` matches any single character, so this needle
    # would find epicrealism by having `_` stand in for the `c`.
    assert "epiCRealism.safetensors" not in shown_for('"epi_realism"'), \
        "an underscore in the typed text matched a letter, as a wildcard"

    # And `%` stood for anything at all, so a quoted term containing one
    # matched every file in the library.
    assert not shown_for('"epi%"'), \
        "a percent sign in the typed text was treated as a wildcard"


class TestTheFoldingRuleItself:
    """The two functions, directly. These are the whole fix."""

    def test_it_keeps_letters_from_every_script(self):
        assert smartgallery._normalize_fuzzy_string("Рисунок") == "рисунок"
        assert smartgallery._normalize_fuzzy_string("Γαλάζιο") == "γαλάζιο"
        assert smartgallery._normalize_fuzzy_string("写真") == "写真"

    def test_it_drops_punctuation(self):
        assert smartgallery._normalize_fuzzy_string(
            "niji+animation.safetensors") == "nijianimationsafetensors"
        assert smartgallery._normalize_fuzzy_string(
            "artist's_style") == "artistsstyle"

    def test_sharp_s_meets_ss(self):
        """casefold, not lower: STRASSE and straße are the same word."""
        assert (smartgallery._normalize_fuzzy_string("straße")
                == smartgallery._normalize_fuzzy_string("STRASSE"))

    def test_a_word_key_pads_every_word(self):
        assert smartgallery._word_key("detail_tweaker_xl.safetensors") == \
            " detail tweaker xl safetensors "
        assert smartgallery._word_key("") == " "
        assert smartgallery._word_key(None) == " "

    def test_nothing_folds_to_nothing(self):
        assert smartgallery._normalize_fuzzy_string(None) == ""
        assert smartgallery._normalize_fuzzy_string("") == ""
        assert smartgallery._normalize_fuzzy_string("+++") == ""
