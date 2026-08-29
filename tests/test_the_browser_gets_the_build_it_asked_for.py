"""Three ways a browser was served yesterday's build.

The bundles change with every TypeScript edit. Each of these was a hole
nothing failed on:

  * `static_v`, the cache-buster stamped onto every script and stylesheet
    URL, read only the files sitting DIRECTLY in `static/`. The bundles
    live in `static/build/`, so editing TypeScript left the value exactly
    where it was and a browser holding the old `gallery.js` was told the
    URL had not changed.

  * the gate that keeps committed bundles matching the source compared
    the tree against the INDEX, which says nothing about a file git has
    never been told exists. A new entry point whose bundle was never
    `git add`ed passed the gate and 404'd in a clean checkout.

  * the clean lived in `just web build` and not in `npm run build-web`,
    so the command the README documents left stale output behind while
    the command the gate runs did not. Two contracts, one of them unsafe.

Only the first is a unit fact, so only the first is a test. The other
two are proved by running git and by running the bundler, and a test
that starts a program is exactly what SG006 forbids -- they live in
`sglint --repo` as SG811 and SG812, where the shape of each recipe is
checked AND the two git commands are asked about a planted bundle.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


# --- the cache-buster ------------------------------------------------------


def _version_of(static: pathlib.Path) -> str:
    """`_static_version`'s rule, over an arbitrary directory.

    The function reads `sg_web/static` by construction, so the rule is
    exercised here over a tree this test built. `test_the_real_rule_is_
    the_one_measured` below holds the two together.
    """
    newest = max((p.stat().st_mtime_ns for p in static.rglob("*") if p.is_file()), default=0)
    return str(newest // 1_000_000)


def _one_level(static: pathlib.Path) -> str:
    """The one-level read, kept as the control."""
    newest = max((p.stat().st_mtime_ns for p in static.iterdir() if p.is_file()), default=0)
    return str(newest // 1_000_000)


@pytest.fixture
def served(tmp_path):
    """A stylesheet beside a build directory, exactly as `static/` is."""
    static = tmp_path / "static"
    (static / "build").mkdir(parents=True)
    (static / "gallery.css").write_text(".cell {}", encoding="utf-8")
    (static / "build" / "gallery.js").write_text("// old", encoding="utf-8")
    old = 1_700_000_000
    for path in static.rglob("*"):
        if path.is_file():
            os.utime(path, (old, old))
    return static


def test_editing_a_bundle_moves_the_cache_buster(served):
    """The defect, stated: this is the file that changes most often."""
    was = _version_of(served)
    bundle = served / "build" / "gallery.js"
    bundle.write_text("// new", encoding="utf-8")
    os.utime(bundle, (1_700_000_500, 1_700_000_500))
    assert _version_of(served) != was


def test_the_old_rule_could_not_see_it(served):
    """The control. Without this the test above passes for any rule that
    happens to notice a file, and never proves the walk was the problem."""
    was = _one_level(served)
    bundle = served / "build" / "gallery.js"
    bundle.write_text("// new", encoding="utf-8")
    os.utime(bundle, (1_700_000_500, 1_700_000_500))
    assert _one_level(served) == was, "the one-level walk would have seen it after all"


def test_a_stylesheet_beside_it_still_moves_it(served):
    """The case that always worked keeps working."""
    was = _version_of(served)
    sheet = served / "gallery.css"
    sheet.write_text(".cell { color: red }", encoding="utf-8")
    os.utime(sheet, (1_700_000_500, 1_700_000_500))
    assert _version_of(served) != was


def test_the_real_rule_is_the_one_measured():
    """`_static_version` walks the whole tree, so the rule above is its
    rule and not a second opinion this file invented."""
    from sg_web import app

    source = app._static_version.__code__.co_consts
    assert any(isinstance(one, str) and one == "*" for one in source), (
        "_static_version no longer walks with rglob('*'); this module's copy of its rule is stale"
    )


def test_every_bundle_the_templates_load_is_under_the_walk():
    """A template that loaded a script from somewhere `static_v` does not
    walk would be un-bustable again, quietly."""
    import re

    stamped = set()
    for page in (REPO / "sg_web" / "templates").glob("*.html"):
        stamped |= set(re.findall(r'"(/static/[^"?]+)\?v=\{\{ static_v \}\}"', page.read_text(encoding="utf-8")))
    assert stamped, "no template stamps the cache-buster at all"
    static = REPO / "sg_web" / "static"
    for url in sorted(stamped):
        served_file = static / url[len("/static/") :]
        assert served_file.is_file(), f"{url} is stamped but no file is there"
        assert served_file.resolve().is_relative_to(static.resolve()), url


# --- the gate ---------------------------------------------------------------


def _recipe(name: str) -> str:
    """One recipe's RUNNABLE lines, with its comments dropped.

    Asserting over the whole text was this file's own first mistake: the
    comment beneath `fresh` explains why `git diff --quiet` is wrong, and
    a substring check read that explanation as the defect. A rule that
    cannot tell a command from a sentence about the command is the same
    rule this repository just deleted from sglint.
    """
    text = (REPO / "web.just").read_text(encoding="utf-8")
    start = text.index(f"\n{name}:")
    rest = text[start + 1 :]
    end = rest.find("\n\n")
    body = rest if end < 0 else rest[:end]
    return "\n".join(line for line in body.splitlines() if not line.strip().startswith("#"))


def test_the_freshness_gate_looks_for_untracked_bundles_too():
    """`git diff` compares the tree against the INDEX. A bundle git has
    never been told about is in neither, so it was invisible."""
    fresh = _recipe("fresh")
    assert "ls-files --others" in fresh, f"nothing here asks about an untracked bundle:\n{fresh}"
    # and `git diff` STAYS, for the other question: a bundle rebuilt and
    # not staged. Answering both with one `git status --porcelain` is
    # wrong, because that also reports STAGED output -- the normal state
    # between `just web build` and the commit, so a gate refusing it
    # could never be passed. This file made that mistake first.
    assert "git diff" in fresh, f"nothing here asks about an unstaged rebuild:\n{fresh}"


# --- one build contract -----------------------------------------------------


def test_the_recipe_no_longer_owns_a_clean_of_its_own():
    """Two commands, one contract. The recipe must DELEGATE, or the
    command the README documents is the unsafe one again."""
    build = _recipe("build")
    assert "npm run --silent build-web" in build, build
    assert "rm -rf" not in build, f"the recipe owns a clean the documented command does not:\n{build}"

    scripts = json.loads((REPO / "package.json").read_text(encoding="utf-8"))["scripts"]
    assert scripts["build-web"] == "node frontend/build.ts", scripts["build-web"]
