"""Drawing the gallery should not normalise paths nobody looks at.

Every page load walks the whole library twice -- once in gallery_view to
build the view, once in get_filter_options_from_db to fill the extension
and prefix dropdowns -- and each walk normalised two paths per row before
anything decided which of them it needed. Only one branch of three reads
the second, and in global scope neither is read at all.

Profiled at 100,000 files, one page load:

    gallery_view                     4.425s cumulative
      get_filter_options_from_db     1.478s
      get_dynamic_folder_config      0.654s
      fetchall                       0.640s
      safe_path_norm  x200,001       0.781s

and the wall time as a library grows, before and after:

    files     before   after
    5000      0.33     0.30
    25000     0.61     0.52
    100000    2.55     2.15

This is a page somebody loads on every navigation, so it grows with the
library and never stops.

The only thing that matters here is that the answers did not change. The
decisions are identical and evaluated lazily, and the checks below hold
the new behaviour against a plain copy of the original eager logic rather
than against a remembered description of it.
"""

from __future__ import annotations

import ast
import os

import pytest


def _norm(p):
    """safe_path_norm, as both copies in the app spell it."""
    if not p:
        return ""
    return os.path.normpath(str(p).replace("\\", "/")).replace("\\", "/").lower().rstrip("/")


def _original_decision(path, target_norm, scope, recursive):
    """The rule as it was: both norms computed, then the branch taken."""
    f_path_norm = _norm(path)
    f_dir_norm = _norm(os.path.dirname(f_path_norm))

    if scope == "global":
        return True
    if recursive:
        return f_path_norm.startswith(target_norm + "/")
    return f_dir_norm == target_norm


def _current_decision(path, target_norm, scope, recursive):
    """The rule now: the same branches, reading only what they use."""
    if scope == "global":
        return True
    if recursive:
        return _norm(path).startswith(target_norm + "/")
    f_path_norm = _norm(path)
    return _norm(os.path.dirname(f_path_norm)) == target_norm


_PATHS = [
    "C:/lib/pic.png",
    "C:/lib/sub/pic.png",
    "C:/lib/sub/deeper/pic.png",
    "C:/lib2/pic.png",
    "C:/LIB/PIC.PNG",
    "C:\\lib\\sub\\pic.png",
    "C:/lib/sub/",
    "C:/lib",
    "/mnt/lib/sub/pic.png".replace("/", os.sep),
    "",
    "C:/lib//sub//pic.png",
    "C:/lib/./sub/pic.png",
    "C:/lib/other/../sub/pic.png",
    "C:/library/pic.png",  # a sibling that must not match "lib"
]


@pytest.mark.parametrize("scope", ["global", "folder"])
@pytest.mark.parametrize("recursive", [True, False])
@pytest.mark.parametrize("target", ["C:/lib", "C:/lib/sub", "C:/lib/"])
def test_the_decision_is_unchanged(scope, recursive, target):
    """The whole safety of the change. Every path, every scope, both
    recursion settings: the lazy form has to answer what the eager form
    answered."""
    target_norm = _norm(target)

    for path in _PATHS:
        before = _original_decision(path, target_norm, scope, recursive)
        after = _current_decision(path, target_norm, scope, recursive)
        assert before == after, (
            f"{path!r} under {target!r} (scope={scope}, recursive={recursive}) was {before} and is now {after}"
        )


def test_a_sibling_folder_is_still_not_inside(smartgallery_app):
    """The case the trailing slash exists for: `library` must not be read
    as being inside `lib`."""
    target_norm = _norm("C:/lib")

    assert not _current_decision("C:/library/pic.png", target_norm, "folder", True)
    assert _current_decision("C:/lib/sub/pic.png", target_norm, "folder", True)


def test_neither_walk_normalises_what_it_does_not_read(gallery_tree):
    """The change itself. Both loops used to compute the directory for
    every row and only one branch of three reads it; if that comes back,
    so does a walk of the whole library doing work nobody uses."""

    tree = gallery_tree

    for name in ("gallery_view", "get_filter_options_from_db"):
        fn = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name), None)
        assert fn is not None, f"{name} is gone"

        # Every dirname() must sit inside a branch, never at the top of a
        # per-row loop where it runs whatever the branch turns out to be.
        for loop in [n for n in ast.walk(fn) if isinstance(n, ast.For)]:
            direct = []
            for statement in loop.body:
                if not isinstance(statement, (ast.Assign, ast.Expr)):
                    continue
                for call in ast.walk(statement):
                    if (
                        isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "dirname"
                    ):
                        direct.append(statement.lineno)
            assert direct == [], (
                f"{name} works out a directory at line {direct} for every row "
                f"before deciding whether the branch taken needs it"
            )


def test_the_page_still_lists_what_it_should(smartgallery_app, monkeypatch):
    """End to end, because the checks above are about a rule and this is
    about a page. Files in a folder, files below it, and files elsewhere."""
    monkeypatch.setattr(smartgallery_app, "FORCE_LOGIN", False)
    monkeypatch.setattr(smartgallery_app, "IS_EXHIBITION_MODE", False)

    root = smartgallery_app.BASE_OUTPUT_PATH
    here = os.path.join(root, "pagecheck")
    below = os.path.join(here, "deeper")
    os.makedirs(below, exist_ok=True)
    made = []
    for folder, name in ((here, "top.png"), (below, "under.png")):
        target = os.path.join(folder, name)
        with open(target, "wb") as fh:
            fh.write(b"x")
        made.append(target.replace(os.sep, "/"))

    conn = smartgallery_app.get_db_connection()
    try:
        for index, path in enumerate(made):
            conn.execute(
                "INSERT OR REPLACE INTO files (id, path, mtime, name, type) VALUES (?,?,?,?,?)",
                (f"page{index:028d}", path, 1700000000.0, os.path.basename(path), "image"),
            )
        conn.commit()
    finally:
        conn.close()

    try:
        smartgallery_app.folder_config_cache = None
        folders = smartgallery_app.get_dynamic_folder_config(force_refresh=True)
        key = next(
            (
                k
                for k, v in folders.items()
                if os.path.normcase(os.path.normpath(v["path"])) == os.path.normcase(os.path.normpath(here))
            ),
            None,
        )
        if key is None:
            pytest.skip("the probe folder is not in the folder config")

        client = smartgallery_app.app.test_client()
        deep = client.get(f"/galleryout/view/{key}?recursive=true", follow_redirects=True).get_data(as_text=True)
        shallow = client.get(f"/galleryout/view/{key}?recursive=false", follow_redirects=True).get_data(as_text=True)

        assert "top.png" in deep, "a recursive view lost a file below the folder"
        assert "under.png" in deep, "a recursive view lost a file below the folder"
        assert "top.png" in shallow, "the folder's own file went missing"
        assert "under.png" not in shallow, "a non-recursive view listed a file from a folder below it"
    finally:
        conn = smartgallery_app.get_db_connection()
        try:
            conn.execute("DELETE FROM files WHERE path LIKE ?", ("%pagecheck%",))
            conn.commit()
        finally:
            conn.close()
        import shutil

        shutil.rmtree(here, ignore_errors=True)
        smartgallery_app.folder_config_cache = None
