"""Listing folders must not get more expensive per folder.

Every gallery page load rebuilds the folder tree from disk -- gallery_view
calls get_dynamic_folder_config(force_refresh=True) deliberately, which is
why a folder ComfyUI created a moment ago appears without a restart. That
freshness is worth having and it is paid for on every page.

Measured on a local SSD, the walk is linear and cheap enough:

     100 folders ->   34 ms
     500 folders ->  143 ms
    2000 folders ->  572 ms

The danger is not the walk. It is someone adding a database lookup, or a
stat of every file, inside the loop -- the classic way an O(n) page becomes
an O(n x m) one. That does not show up on a developer's tidy library and
does show up on a NAS holding four years of dated folders.

So this asserts the SHAPE rather than the clock: the number of database
connections a folder scan opens must not depend on how many folders there
are. A wall-clock threshold would be flaky on shared hardware and would
tell nobody what broke.
"""

from __future__ import annotations


def _count_connections_for(smartgallery_app, monkeypatch, root, folder_count):
    """Open `folder_count` folders under `root`, then count the database
    connections one folder scan makes."""
    root.mkdir(exist_ok=True)
    for i in range(folder_count):
        (root / f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}-{i}").mkdir(exist_ok=True)

    opened = {"count": 0}
    real_connect = smartgallery_app.get_db_connection

    def counting_connect(*args, **kwargs):
        opened["count"] += 1
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(smartgallery_app, "BASE_OUTPUT_PATH", str(root))
    monkeypatch.setattr(smartgallery_app, "folder_config_cache", None)
    monkeypatch.setattr(smartgallery_app, "get_db_connection", counting_connect)
    try:
        folders = smartgallery_app.get_dynamic_folder_config(force_refresh=True)
    finally:
        monkeypatch.undo()

    return opened["count"], len(folders)


def test_the_scan_really_sees_the_folders(smartgallery_app, tmp_path, monkeypatch):
    """Control: if the walk stopped finding anything, the comparison below
    would hold perfectly and mean nothing."""
    _connections, found = _count_connections_for(
        smartgallery_app, monkeypatch, tmp_path / "control", 40)

    assert found == 41, f"expected 40 folders plus the root, found {found}"


def test_the_cost_does_not_grow_with_the_number_of_folders(smartgallery_app,
                                                           tmp_path, monkeypatch):
    """The regression this guards: a lookup moved inside the loop."""
    small, small_found = _count_connections_for(
        smartgallery_app, monkeypatch, tmp_path / "small", 20)
    large, large_found = _count_connections_for(
        smartgallery_app, monkeypatch, tmp_path / "large", 400)

    assert small_found == 21 and large_found == 401, (small_found, large_found)
    assert large <= small, (
        f"a scan of 400 folders opened {large} database connections against "
        f"{small} for 20. Something in the folder loop is querying per folder; "
        f"on a large library that is the difference between a page that loads "
        f"and one that does not.")


def test_the_scan_opens_only_a_handful_of_connections(smartgallery_app,
                                                      tmp_path, monkeypatch):
    """A ceiling in absolute terms too, so a constant-but-large number of
    lookups is not mistaken for fine."""
    connections, _found = _count_connections_for(
        smartgallery_app, monkeypatch, tmp_path / "ceiling", 100)

    assert connections <= 4, (
        f"a single folder scan opened {connections} database connections")
