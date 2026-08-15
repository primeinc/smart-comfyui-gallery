"""Regression tests: subfolder inclusion is the DEFAULT everywhere a view
can span a subtree. A bare URL (no `recursive` param) must show the whole
subtree; only an explicit recursive=false narrows to the folder or
collection itself — and that narrowing counts as an active filter."""

import time as _time

import pytest


_PREFIX = "sfd:"


@pytest.fixture()
def sg(smartgallery_app):
    """Monolith module plus per-test cleanup of every sfd: row."""
    yield smartgallery_app
    with smartgallery_app.get_db_connection() as conn:
        conn.execute(
            "DELETE FROM collection_files WHERE collection_id IN (SELECT id FROM collections WHERE name LIKE 'sfd_%')"
        )
        conn.execute("DELETE FROM collections WHERE name LIKE 'sfd_%'")
        conn.execute("DELETE FROM files WHERE id LIKE ?", (_PREFIX + '%',))
        conn.commit()


def _insert_file(conn, suffix, path, **overrides):
    fid = _PREFIX + suffix
    row = {
        'id': fid,
        'path': path,
        'mtime': 1000.0,
        'name': f"{suffix}.png",
        'type': 'image',
        'has_workflow': 0,
        'workflow_hash': '',
        'prompt_hash': '',
        'hash_failed': 0,
    }
    row.update(overrides)
    conn.execute(
        """INSERT INTO files (id, path, mtime, name, type, has_workflow, workflow_hash, prompt_hash, hash_failed)
           VALUES (:id, :path, :mtime, :name, :type, :has_workflow, :workflow_hash, :prompt_hash, :hash_failed)""",
        row,
    )
    conn.commit()
    return fid


def _root_paths(sg):
    base = sg.BASE_OUTPUT_PATH.replace('\\', '/')
    return f"{base}/direct_sfd.png", f"{base}/subdir_sfd/nested_sfd.png"


# --- physical folder browsing -----------------------------------------------


def test_folder_view_includes_subfolders_by_default(sg):
    # Arrange: one file directly in the root, one in a subfolder.
    direct_path, nested_path = _root_paths(sg)
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'direct', direct_path)
        _insert_file(conn, 'nested', nested_path)
    client = sg.app.test_client()

    # Act: bare URL, no recursive param.
    html = client.get("/galleryout/view/_root_").get_data(as_text=True)

    # Assert: the subtree is visible and the default counts as no filter.
    assert _PREFIX + 'direct' in html
    assert _PREFIX + 'nested' in html
    assert "const activeFiltersCount = 0;" in html


def test_folder_view_explicit_opt_out_shows_folder_only(sg):
    # Arrange
    direct_path, nested_path = _root_paths(sg)
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'direct', direct_path)
        _insert_file(conn, 'nested', nested_path)
    client = sg.app.test_client()

    # Act
    html = client.get("/galleryout/view/_root_?recursive=false").get_data(as_text=True)

    # Assert: subfolder file hidden; the narrowing registers as a filter.
    assert _PREFIX + 'direct' in html
    assert _PREFIX + 'nested' not in html
    assert "const activeFiltersCount = 1;" in html


def test_folder_view_explicit_recursive_true_still_works(sg):
    # Arrange
    direct_path, nested_path = _root_paths(sg)
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'direct', direct_path)
        _insert_file(conn, 'nested', nested_path)
    client = sg.app.test_client()

    # Act: legacy bookmarks with recursive=true keep their meaning.
    html = client.get("/galleryout/view/_root_?recursive=true").get_data(as_text=True)

    # Assert
    assert _PREFIX + 'direct' in html
    assert _PREFIX + 'nested' in html


# --- clustering in Current Context ------------------------------------------


def test_current_context_clustering_spans_subfolders_by_default(sg):
    """The scenario that motivated the flip: cluster members living in a
    date-stamped subfolder must appear in a Current Context clusterize of
    the parent folder without any extra parameter."""
    # Arrange: two assets sharing one architecture, one of them in a subfolder.
    direct_path, nested_path = _root_paths(sg)
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'direct', direct_path, has_workflow=1, workflow_hash='sfdH')
        _insert_file(conn, 'nested', nested_path, has_workflow=1, workflow_hash='sfdH')
    client = sg.app.test_client()

    # Act: bare cluster URL — no recursive param anywhere.
    html = client.get(
        "/galleryout/view/_root_?cluster_mode=workflow&cluster_scope=current"
    ).get_data(as_text=True)

    # Assert: both cluster members render, subfolder included.
    assert _PREFIX + 'direct' in html
    assert _PREFIX + 'nested' in html


# --- collection browsing -----------------------------------------------------


def test_collection_view_includes_subcollections_by_default(sg):
    # Arrange: parent album with one file, child album with another.
    direct_path, nested_path = _root_paths(sg)
    with sg.get_db_connection() as conn:
        f_parent = _insert_file(conn, 'cparent', direct_path)
        f_child = _insert_file(conn, 'cchild', nested_path)
        cur = conn.execute(
            "INSERT INTO collections (name, type, color, is_public, created_at) VALUES ('sfd_parent', 'user_album', '#fff', 1, ?)",
            (_time.time(),),
        )
        parent_id = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO collections (name, type, color, is_public, parent_id, created_at) VALUES ('sfd_child', 'user_album', '#fff', 1, ?, ?)",
            (parent_id, _time.time()),
        )
        child_id = cur.lastrowid
        conn.execute("INSERT INTO collection_files (collection_id, file_id, added_at) VALUES (?, ?, ?)", (parent_id, f_parent, _time.time()))
        conn.execute("INSERT INTO collection_files (collection_id, file_id, added_at) VALUES (?, ?, ?)", (child_id, f_child, _time.time()))
        conn.commit()
    client = sg.app.test_client()

    # Act
    default_html = client.get(f"/galleryout/collection/{parent_id}").get_data(as_text=True)
    narrowed_html = client.get(f"/galleryout/collection/{parent_id}?recursive=false").get_data(as_text=True)

    # Assert: nested collection's file visible by default, hidden on opt-out.
    assert _PREFIX + 'cparent' in default_html
    assert _PREFIX + 'cchild' in default_html
    assert _PREFIX + 'cparent' in narrowed_html
    assert _PREFIX + 'cchild' not in narrowed_html
