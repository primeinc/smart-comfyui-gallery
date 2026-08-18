"""Every route must say who may call it, or fail this test.

Ten or so of the fixes in this suite were the same fault: a route that
answered somebody it should not, found one at a time by reading. That is a
poor way to hold a line. Nothing about an ungated route looks wrong -- it
looks like the ones beside it, which are gated by a decorator sitting three
lines further up.

So this walks every route in smartgallery.py and sorts it by how it decides
who may call it:

  management  the management_api_only decorator
  per-file    calls is_file_accessible, so visibility is the file's
  session     refuses a caller with no session where logins are required
  role        branches on should_strip_metadata or an explicit role
  OPEN        none of the above

OPEN is not forbidden -- a login form has to be open -- but it has to be
listed here deliberately. A route added without a gate fails, and the
message says what the options are.

The AI blueprint is not covered here: its routes are registered through
add_url_rule with an explicit guarded/per-file policy of its own, stated in
create_ai_blueprint's docstring. That exclusion is enforced rather than
promised -- test_ai_routes_are_classified.py sorts those twenty endpoints
the same way and fails on an OPEN one. Endpoints Flask registers that no
source-level audit can see (its own /static/) are covered by
test_static_assets.py.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_SOURCE = pathlib.Path(__file__).resolve().parent.parent / "smartgallery.py"

# Routes that answer anyone, on purpose.
_INTENTIONALLY_OPEN = {
    # The login form itself, and the redirect that leads to it.
    "exhibition_login": "the login form cannot require a login",
    "gallery_redirect_base": "a redirect to the gallery root",
    "exhibition_logout": "ending your own session needs no privilege",
    # Sets a flag in the caller's OWN session that only ever hides more
    # (test_blind_rating covers that anyone may opt into blindness).
    "toggle_my_ratings": "sets a flag in the caller's own session",
}


def _classify(gallery_source, gallery_tree):
    lines = gallery_source.splitlines()
    tree = gallery_tree

    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        routes, decorators = [], []
        for dec in node.decorator_list:
            if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "route"):
                if dec.args and isinstance(dec.args[0], ast.Constant):
                    routes.append(dec.args[0].value)
            elif isinstance(dec, ast.Name):
                decorators.append(dec.id)
        if not routes:
            continue

        body = "\n".join(lines[node.lineno - 1: node.end_lineno])
        if "management_api_only" in decorators:
            kind = "management"
        elif "is_file_accessible" in body:
            kind = "per-file"
        elif ("Authentication required" in body or "'user_id' in session" in body
              or "session.get('user_id')" in body):
            kind = "session"
        elif "should_strip_metadata" in body or "['ADMIN'" in body:
            # The list literal, not the bare word: a route that merely
            # mentions ADMIN in a comment must not count as gated by it.
            kind = "role"
        else:
            kind = "OPEN"
        found[node.name] = (kind, routes[0])
    return found


def test_the_classifier_still_reads_the_file(gallery_source, gallery_tree):
    """Control: if the parsing breaks, every assertion below passes on an
    empty set. Two known routes are pinned by name and category."""
    found = _classify(gallery_source, gallery_tree)

    assert len(found) > 80, f"only {len(found)} routes found; the parser is broken"
    assert found["delete_folder"][0] == "management", found.get("delete_folder")
    assert found["serve_file"][0] == "per-file", found.get("serve_file")


def test_no_route_is_open_by_accident(gallery_source, gallery_tree):
    """The regression this file exists for."""
    found = _classify(gallery_source, gallery_tree)

    unclassified = sorted(
        f"{name} ({route})" for name, (kind, route) in found.items()
        if kind == "OPEN" and name not in _INTENTIONALLY_OPEN)

    assert unclassified == [], (
        "these routes answer anyone and are not listed as deliberately "
        f"public: {unclassified}.\n"
        "Add management_api_only, or an is_file_accessible check, or refuse "
        "a caller with no session where logins are required -- or, if it "
        "really should answer anyone, add it to _INTENTIONALLY_OPEN in this "
        "file with the reason.")


def test_the_open_list_has_not_gone_stale(gallery_source, gallery_tree):
    """A name left here after the route gained a gate, or after it was
    removed, would silently excuse a future route of the same name."""
    found = _classify(gallery_source, gallery_tree)

    missing = sorted(n for n in _INTENTIONALLY_OPEN if n not in found)
    assert missing == [], f"listed as open but no longer a route: {missing}"

    no_longer_open = sorted(n for n in _INTENTIONALLY_OPEN
                            if found[n][0] != "OPEN")
    assert no_longer_open == [], (
        f"listed as deliberately open but now gated: {no_longer_open}. "
        f"Remove them from _INTENTIONALLY_OPEN.")


@pytest.mark.parametrize("name", sorted(_INTENTIONALLY_OPEN))
def test_each_open_route_has_a_stated_reason(name):
    assert _INTENTIONALLY_OPEN[name].strip(), f"{name} is listed without a reason"
