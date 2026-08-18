"""Every AI route must say who may call it, the same as every other route.

tests/test_every_route_is_classified.py walks smartgallery.py and sorts
each @app.route by how it decides who may call it. It excludes the AI
layer, saying those routes "are registered through add_url_rule with an
explicit guarded/per-file policy of its own". That sentence was true and
nothing checked it, which is the same shape as the bug that let /static/
serve the management interface unguarded: an exclusion written once, in
prose, and then trusted.

Twenty endpoints hang off this blueprint. Seven are registered without the
management guard, and those are meant to police themselves by calling
_check_file_access on the file they are about to talk about. Nothing made
them. One added without either would answer anybody who can reach the port,
would look exactly like its neighbours, and no test in the suite would
notice.

Audited at the time of writing, all twenty are accounted for: 13 guarded,
6 per-file -- including both mask routes, which take a finding id rather
than a file id and resolve it through _serve_mask before serving anything
-- and search_semantic, which returns ids from across the library and so
passes every one of them through _visible instead.

The policies, as the code expresses them:

  guarded   registered _wrap(view, guarded=True), so the host's guard runs
            (management_api_only) -- for anything that reads across files
  per-file  the body calls _check_file_access, or _serve_mask which does
  filtered  every id it returns passes _visible
  OPEN      none of the above
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_SERVICE = pathlib.Path(__file__).resolve().parent.parent / "smartgallery_ai" / "service.py"

_PER_FILE_MARKERS = {"_check_file_access", "_serve_mask"}
_FILTER_MARKERS = {"_visible"}


def _names_used(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _registrations(tree):
    """(endpoint, view_name, guarded) for every bp.add_url_rule call."""
    found = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_url_rule"
        ):
            continue
        if len(node.args) < 3:
            continue
        endpoint = node.args[1].value if isinstance(node.args[1], ast.Constant) else "?"
        view = node.args[2]

        # `guard(status) if guard is not None else status` -- guarded when
        # the host supplies a guard, which smartgallery.py always does.
        if isinstance(view, ast.IfExp):
            names = _names_used(view) - {"guard"}
            found.append((endpoint, next(iter(names), "?"), True))
            continue

        if isinstance(view, ast.Call) and isinstance(view.func, ast.Name) and view.func.id == "_wrap":
            guarded = any(
                kw.arg == "guarded" and isinstance(kw.value, ast.Constant) and kw.value.value is True
                for kw in view.keywords
            )
            name = view.args[0].id if isinstance(view.args[0], ast.Name) else "?"
            found.append((endpoint, name, guarded))
            continue

        found.append((endpoint, "?", False))
    return found


def _classify(source: str):
    tree = ast.parse(source)
    bodies = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}

    verdict = {}
    for endpoint, view_name, guarded in _registrations(tree):
        if guarded:
            verdict[endpoint] = "guarded"
            continue
        body = bodies.get(view_name)
        if body is None:
            verdict[endpoint] = "OPEN"
            continue
        used = _names_used(body)
        if used & _PER_FILE_MARKERS:
            verdict[endpoint] = "per-file"
        elif used & _FILTER_MARKERS:
            verdict[endpoint] = "filtered"
        else:
            verdict[endpoint] = "OPEN"
    return verdict


@pytest.fixture(scope="module")
def classified():
    return _classify(open(_SERVICE, encoding="utf-8").read())


def test_the_classifier_sees_the_whole_blueprint(classified):
    """Control. A parser that matched nothing would leave every assertion
    below passing over an empty dictionary."""
    assert len(classified) == 20, sorted(classified)
    assert {"status", "similar", "review_mask", "search_semantic"} <= set(classified)


def test_it_can_tell_the_policies_apart(classified):
    """Second control: if everything came back "guarded" the check would be
    satisfied without distinguishing anything."""
    kinds = set(classified.values())

    assert "guarded" in kinds, classified
    assert "per-file" in kinds, classified
    assert classified["status"] == "guarded"
    assert classified["faces_clusters"] == "guarded"
    assert classified["similar"] == "per-file"
    assert classified["review_mask"] == "per-file", (
        "the mask routes take a finding id, not a file id; _serve_mask is what resolves it and checks the file"
    )


def test_no_ai_route_answers_everybody(classified):
    open_routes = sorted(e for e, kind in classified.items() if kind == "OPEN")

    assert not open_routes, (
        f"{open_routes} are registered without the management guard and "
        f"never check the file they talk about. Either register them "
        f"_wrap(view, guarded=True), or call _check_file_access(file_id) "
        f"in the body."
    )


def test_the_classifier_catches_a_route_that_checks_nothing():
    """The control that matters. Without it the sweep above could be
    passing because nothing is ever classified OPEN, and the next
    unguarded route would sail through.

    This is the shape someone would actually add: a view registered like
    its per-file neighbours, but with no check in it."""
    source = (
        "def make():\n"
        "    def _check_file_access(fid):\n"
        "        pass\n"
        "    def _wrap(f, guarded=False):\n"
        "        return f\n"
        "    def good(file_id):\n"
        "        _check_file_access(file_id)\n"
        "        return 1\n"
        "    def careless(file_id):\n"
        "        return 1\n"
        "    bp.add_url_rule('/good/<file_id>', 'good', _wrap(good))\n"
        "    bp.add_url_rule('/careless/<file_id>', 'careless', _wrap(careless))\n"
        "    bp.add_url_rule('/safe', 'safe', _wrap(careless, guarded=True))\n"
    )

    verdict = _classify(source)

    assert verdict == {"good": "per-file", "careless": "OPEN", "safe": "guarded"}, verdict
