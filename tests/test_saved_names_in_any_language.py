"""Naming something is not naming a file, and the difference cost data.

Saved prompts, saved queries and workflow templates all take a name from
the person and turn it into a filename with secure_filename(). That
function transliterates to ASCII and drops whatever will not convert:

    测试        -> ''
    рисунок     -> ''
    イラスト      -> ''
    测试.txt    -> 'txt'

So every saved prompt with a Chinese, Japanese, Korean or Cyrillic name
landed in one file called `txt`. Reproduced against the shipped code:
three saves, three "Prompt saved." replies, one file on disk holding the
third. The first two were destroyed without a word, and none could be
loaded back.

Workflow templates were the same, collapsing to a hidden file called
`.json`.

safe_media_filename() already existed for exactly this -- it was written
when uploads had the same fault -- and keeps the name while removing what
is actually dangerous: any directory part, the characters Windows forbids,
control characters, trailing dots and spaces, and the reserved device
names. Using it here is the whole fix.

German is in the set on purpose: Ordner-Größe is Latin script and still
lost its ö and ß, so this was never only about non-Latin alphabets.
"""

from __future__ import annotations

import ast
import os

import pytest

_NAMES = {
    "测试": "chinese",
    "рисунок": "cyrillic",
    "イラスト": "japanese",
    "한글": "korean",
    "Ordner-Größe": "german",
    "plain": "ascii",
}


@pytest.fixture
def client(smartgallery_app, monkeypatch, tmp_path):
    """A gallery folder of this test's own: these routes write real files."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)
    monkeypatch.setattr(smartgallery_app, "BASE_SMARTGALLERY_PATH", str(tmp_path))
    monkeypatch.setattr(smartgallery_app, "IMPORTED_WORKFLOWS_DIR", str(tmp_path / ".imported_workflows"))
    return smartgallery_app.app.test_client()


def _prompts_dir(smartgallery_app):
    return os.path.join(smartgallery_app.BASE_SMARTGALLERY_PATH, ".omniquery", "saved_prompts")


def _queries_dir(smartgallery_app):
    return os.path.join(smartgallery_app.BASE_SMARTGALLERY_PATH, ".omniquery", "saved_queries")


def test_every_saved_prompt_keeps_its_own_name(smartgallery_app, client):
    """The bug: they all became one file called `txt`."""
    for name, body in _NAMES.items():
        answer = client.post("/galleryout/api/omniquery/prompts/save", json={"name": name, "text": body}).get_json()
        assert answer["status"] == "success", (name, answer)

    on_disk = sorted(os.listdir(_prompts_dir(smartgallery_app)))

    assert len(on_disk) == len(_NAMES), (
        f"{len(_NAMES)} prompts were saved and {len(on_disk)} files exist: "
        f"{on_disk}. Names that collapse to the same text overwrite each "
        f"other, and every save still reported success."
    )
    for name in _NAMES:
        assert f"{name}.txt" in on_disk, (name, on_disk)


def test_a_saved_prompt_comes_back(smartgallery_app, client):
    """Saving distinctly is only half of it; the round trip is the point."""
    for name, body in _NAMES.items():
        client.post("/galleryout/api/omniquery/prompts/save", json={"name": name, "text": body})

    for name, body in _NAMES.items():
        answer = client.post("/galleryout/api/omniquery/prompts/load", json={"name": f"{name}.txt"}).get_json()
        assert answer["status"] == "success", (name, answer)
        assert answer["text"] == body, (name, answer)


def test_every_saved_query_keeps_its_own_name(smartgallery_app, client):
    """The same pair of routes again, for the other saved thing."""
    for name in _NAMES:
        answer = client.post(
            "/galleryout/api/omniquery/queries/save", json={"name": name, "sql": f"SELECT '{name}'"}
        ).get_json()
        assert answer["status"] == "success", (name, answer)

    on_disk = sorted(os.listdir(_queries_dir(smartgallery_app)))

    assert len(on_disk) == len(_NAMES), on_disk


def test_the_list_offers_the_names_back(smartgallery_app, client):
    """What the person sees. Collapsed names showed up as an entry called
    `txt` rather than what they typed."""
    for name in _NAMES:
        client.post("/galleryout/api/omniquery/queries/save", json={"name": name, "sql": "SELECT 1"})

    listed = client.get("/galleryout/api/omniquery/queries/list").get_json()
    names = {entry.get("name", "") for entry in listed.get("queries", [])}

    for name in _NAMES:
        assert any(name in listed_name for listed_name in names), (name, names)


def test_a_name_cannot_reach_outside_its_folder(smartgallery_app, client):
    """The one thing secure_filename was genuinely providing. Keeping the
    name must not mean keeping a path."""
    client.post("/galleryout/api/omniquery/prompts/save", json={"name": "../../escaped", "text": "nope"})

    root = smartgallery_app.BASE_SMARTGALLERY_PATH
    assert not os.path.exists(os.path.join(root, "escaped.txt")), "a saved prompt was written outside its folder"
    assert not os.path.exists(os.path.join(root, "..", "escaped.txt"))
    assert "escaped.txt" in os.listdir(_prompts_dir(smartgallery_app))


@pytest.mark.parametrize("name", ["..", ".", "..."])
def test_a_name_of_nothing_much_still_makes_one_ordinary_file(smartgallery_app, client, name):
    """`.` and `..` name directories, not files.

    Asserting the resulting filename would be asserting a guess: the route
    appends .txt BEFORE the name is made safe, so ".." arrives as "...txt",
    which is an ordinary file and perfectly safe. What has to hold is the
    property, not the spelling -- one real file, inside the folder, and no
    directory made."""
    answer = client.post("/galleryout/api/omniquery/prompts/save", json={"name": name, "text": "nope"}).get_json()

    assert answer["status"] == "success", answer
    directory = _prompts_dir(smartgallery_app)
    entries = os.listdir(directory)

    assert len(entries) == 1, entries
    made = os.path.join(directory, entries[0])
    assert os.path.isfile(made), f"{entries[0]} is not a file"
    assert os.path.dirname(os.path.realpath(made)) == os.path.realpath(directory)


def test_no_user_supplied_name_goes_through_secure_filename(gallery_tree, smartgallery_app):
    """Seventeen call sites had this fault, across saved prompts, saved
    queries and workflow templates. Reaching each through its own endpoint
    would need a real workflow payload for the template ones, and a test
    that skips when the payload is wrong guards nothing at all -- the first
    version of this file had exactly that and passed while proving nothing.

    So the rule is checked flat: the monolith does not call
    secure_filename. Everywhere a name becomes a path it uses
    safe_media_filename, which keeps the name and removes what is
    dangerous, and there is one rule to check rather than a judgement per
    site about whether that name could be someone's."""
    tree = gallery_tree

    def _called(node):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return ""

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    # Read as calls rather than as text: the first version of this matched
    # the word inside safe_media_filename's own docstring, which explains
    # the very fault being checked for, and failed against the fixed file.
    offenders = sorted({node.lineno for node in calls if _called(node) == "secure_filename"})
    replacements = [node for node in calls if _called(node) == "safe_media_filename"]

    assert not offenders, (
        f"line(s) {offenders} still turn a name into a filename with "
        f"secure_filename, which drops every non-ASCII character"
    )
    assert len(replacements) > 10, (
        f"only {len(replacements)} calls to safe_media_filename; the "
        f"sanitising has gone rather than been replaced, and this check "
        f"would pass over a file that does none at all"
    )
