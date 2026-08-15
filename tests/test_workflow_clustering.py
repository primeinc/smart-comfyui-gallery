"""Behavior tests for the Smart Asset Clustering backend seams:
process_clustering (scope/basis/sort semantics, stale-hash refresh),
backfill_unhashed_workflows (failure marking, no perpetual re-index),
compute_workflow_hashes (no synthetic prompt hashes),
clear_synthetic_prompt_hashes (data repair), and the ViewSnapshotStore
that replaced the process-wide gallery_view_cache global together with
its /load_more and /api/current_view_ids consumers."""

import hashlib
import re

import pytest


_PREFIX = "wfc:"


def _insert_file(conn, suffix, **overrides):
    """Insert a minimal files row usable by the clustering queries."""
    fid = _PREFIX + suffix
    row = {
        'id': fid,
        'path': f"C:/nonexistent/wfc/{suffix}.png",
        'mtime': 1000.0,
        'name': f"{suffix}.png",
        'type': 'image',
        'has_workflow': 1,
        'workflow_hash': '',
        'prompt_hash': '',
        'models_hash': '',
        'hash_failed': 0,
    }
    row.update(overrides)
    conn.execute(
        """INSERT INTO files (id, path, mtime, name, type, has_workflow, workflow_hash, prompt_hash, models_hash, hash_failed)
           VALUES (:id, :path, :mtime, :name, :type, :has_workflow, :workflow_hash, :prompt_hash, :models_hash, :hash_failed)""",
        row,
    )
    conn.commit()
    return row


def _view_dict(row):
    """The in-memory file dict shape gallery_view passes to process_clustering."""
    d = dict(row)
    d.setdefault('avg_rating', None)
    return d


@pytest.fixture()
def sg(smartgallery_app, monkeypatch):
    """The monolith module plus per-test cleanup of every wfc: row.

    The async cluster-backfill dispatcher is stubbed to a no-op so tests never
    spawn real background threads by accident; tests that exercise the real
    dispatcher use the stashed `_real_ensure` reference."""
    real_ensure = smartgallery_app.ensure_cluster_backfill_async
    monkeypatch.setattr(smartgallery_app, 'ensure_cluster_backfill_async',
                        lambda force_all=False: True)
    smartgallery_app._real_ensure = real_ensure
    yield smartgallery_app
    with smartgallery_app.get_db_connection() as conn:
        conn.execute("DELETE FROM files WHERE id LIKE ?", (_PREFIX + '%',))
        conn.commit()


# --- process_clustering: mode off ------------------------------------------


def test_process_clustering_without_mode_is_identity(sg):
    # Arrange
    files = [{'id': 'a'}, {'id': 'b'}]

    # Act
    with sg.app.test_request_context('/'):
        result = sg.process_clustering(files, None, 'date_desc', None, 'current')

    # Assert
    assert result is files


# --- process_clustering: current scope, basis semantics ---------------------


def test_workflow_mode_current_scope_groups_by_workflow_hash(sg):
    # Arrange: two files share architecture A, one is B, one has no workflow.
    with sg.get_db_connection() as conn:
        r1 = _insert_file(conn, 'a1', workflow_hash='hashA', prompt_hash='p1')
        r2 = _insert_file(conn, 'a2', workflow_hash='hashA', prompt_hash='p2', mtime=2000.0)
        r3 = _insert_file(conn, 'b1', workflow_hash='hashB')
        r4 = _insert_file(conn, 'nowf', has_workflow=0)
    current = [_view_dict(r) for r in (r1, r2, r3, r4)]

    # Act
    with sg.app.test_request_context('/'):
        result = sg.process_clustering(current, 'workflow', 'date_desc', None, 'current')

    # Assert: the workflow-less file is dropped; clusters are contiguous.
    ids = [f['id'] for f in result]
    assert _PREFIX + 'nowf' not in ids
    assert sorted(ids) == sorted([_PREFIX + 'a1', _PREFIX + 'a2', _PREFIX + 'b1'])
    hashes = [f['workflow_hash'] for f in result]
    assert hashes == sorted(hashes)  # grouped by hash key


def test_prompt_mode_keeps_files_whose_workflow_hash_is_empty(sg):
    """A file with a real prompt but an unhashable graph must still appear in
    prompt clusters (the old filter demanded workflow_hash in every mode)."""
    # Arrange
    with sg.get_db_connection() as conn:
        prompt_only = _insert_file(conn, 'ponly', workflow_hash='', prompt_hash='promptZ', hash_failed=1)
        both = _insert_file(conn, 'pboth', workflow_hash='wf1', prompt_hash='promptZ')
    current = [_view_dict(prompt_only), _view_dict(both)]

    # Act
    with sg.app.test_request_context('/'):
        result = sg.process_clustering(current, 'prompt', 'date_desc', None, 'current')

    # Assert: both members of the prompt cluster survive.
    assert sorted(f['id'] for f in result) == sorted([_PREFIX + 'ponly', _PREFIX + 'pboth'])


def test_prompt_mode_drops_files_without_prompt_hash(sg):
    # Arrange: hashed architecture but no prompt text.
    with sg.get_db_connection() as conn:
        no_prompt = _insert_file(conn, 'nop', workflow_hash='wf2', prompt_hash='')

    # Act
    with sg.app.test_request_context('/'):
        result = sg.process_clustering([_view_dict(no_prompt)], 'prompt', 'date_desc', None, 'current')

    # Assert
    assert result == []


def test_current_scope_refreshes_hashes_computed_after_the_page_query(sg):
    """gallery_view fetches rows before the in-request backfill runs; files
    hashed during this very request must not be dropped as unhashed."""
    # Arrange: DB row carries the fresh hash, the passed dict is stale-empty.
    with sg.get_db_connection() as conn:
        row = _insert_file(conn, 'fresh', workflow_hash='freshWF', prompt_hash='freshPR')
    stale = _view_dict(dict(row, workflow_hash='', prompt_hash=''))

    # Act
    with sg.app.test_request_context('/'):
        result = sg.process_clustering([stale], 'workflow', 'date_desc', None, 'current')

    # Assert: the file survives with its DB hashes.
    assert [f['id'] for f in result] == [_PREFIX + 'fresh']
    assert result[0]['workflow_hash'] == 'freshWF'
    assert result[0]['prompt_hash'] == 'freshPR'


def test_current_scope_target_filters_to_the_target_cluster(sg):
    # Arrange
    with sg.get_db_connection() as conn:
        t = _insert_file(conn, 'tgt', workflow_hash='hashT')
        same = _insert_file(conn, 'tgt2', workflow_hash='hashT')
        other = _insert_file(conn, 'oth', workflow_hash='hashU')
    current = [_view_dict(r) for r in (t, same, other)]

    # Act
    with sg.app.test_request_context('/'):
        result = sg.process_clustering(current, 'workflow', 'date_desc', _PREFIX + 'tgt', 'current')

    # Assert
    assert sorted(f['id'] for f in result) == sorted([_PREFIX + 'tgt', _PREFIX + 'tgt2'])


def test_current_scope_unknown_target_yields_empty(sg):
    # Arrange
    with sg.get_db_connection() as conn:
        row = _insert_file(conn, 'lone', workflow_hash='hashL')

    # Act
    with sg.app.test_request_context('/'):
        result = sg.process_clustering([_view_dict(row)], 'workflow', 'date_desc', 'wfc:no-such-id', 'current')

    # Assert
    assert result == []


# --- process_clustering: inner-cluster sorting ------------------------------


def test_inner_sort_orders_within_cluster(sg):
    # Arrange: one cluster, three members with distinct mtime and rating.
    with sg.get_db_connection() as conn:
        old = _insert_file(conn, 'old', workflow_hash='hashS', mtime=100.0)
        mid = _insert_file(conn, 'mid', workflow_hash='hashS', mtime=200.0)
        new = _insert_file(conn, 'new', workflow_hash='hashS', mtime=300.0)

    def with_rating(row, rating):
        d = _view_dict(row)
        d['avg_rating'] = rating
        return d

    current = [with_rating(old, 5.0), with_rating(mid, 1.0), with_rating(new, 3.0)]

    # Act
    with sg.app.test_request_context('/'):
        newest = sg.process_clustering(list(current), 'workflow', 'date_desc', None, 'current')
        oldest = sg.process_clustering(list(current), 'workflow', 'date_asc', None, 'current')
        rated = sg.process_clustering(list(current), 'workflow', 'rating_desc', None, 'current')

    # Assert
    assert [f['id'] for f in newest] == [_PREFIX + 'new', _PREFIX + 'mid', _PREFIX + 'old']
    assert [f['id'] for f in oldest] == [_PREFIX + 'old', _PREFIX + 'mid', _PREFIX + 'new']
    assert [f['id'] for f in rated] == [_PREFIX + 'old', _PREFIX + 'new', _PREFIX + 'mid']


# --- process_clustering: global scope ---------------------------------------


def test_global_prompt_mode_only_returns_prompt_hashed_files(sg):
    # Arrange
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'gp1', workflow_hash='gw1', prompt_hash='gp')
        _insert_file(conn, 'gp2', workflow_hash='gw2', prompt_hash='gp')
        _insert_file(conn, 'gnop', workflow_hash='gw3', prompt_hash='', hash_failed=1)

    # Act
    with sg.app.test_request_context('/'):
        result = sg.process_clustering([], 'prompt', 'date_desc', None, 'global')

    # Assert: promptless file excluded from the global prompt clustering.
    ids = {f['id'] for f in result if f['id'].startswith(_PREFIX)}
    assert ids == {_PREFIX + 'gp1', _PREFIX + 'gp2'}


def test_global_target_without_required_hash_yields_empty(sg):
    # Arrange: target claims a workflow but its graph never hashed.
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'ghole', workflow_hash='', prompt_hash='', hash_failed=1)

    # Act
    with sg.app.test_request_context('/'):
        result = sg.process_clustering([], 'workflow', 'date_desc', _PREFIX + 'ghole', 'global')

    # Assert
    assert result == []


# --- non-ComfyUI generators join clusters (WI-31 "211 assets" defect) --------
#
# Regression for: a SwarmUI-dominated gallery showed "Global Scope 211 Assets"
# for BOTH cluster bases, because every clustering gate required has_workflow=1
# (an embedded ComfyUI graph) and 42k SwarmUI images had none.


_SUI_PARAMS_TMPL = (
    '{"sui_image_params": {"prompt": "%s", "negativeprompt": "", '
    '"model": "%s", "seed": 7, "steps": 20, "cfgscale": 7.0, '
    '"width": 64, "height": 64, "swarm_version": "0.9.3.1"}}'
)


def _make_swarm_png(tmp_path, name, prompt, model):
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo()
    info.add_text("parameters", _SUI_PARAMS_TMPL % (prompt, model))
    path = tmp_path / name
    Image.new("RGB", (8, 8), (30, 60, 90)).save(path, pnginfo=info)
    return str(path).replace('\\', '/')


def _make_swarm_png_params(tmp_path, name, params):
    import json
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo()
    info.add_text("parameters", json.dumps({"sui_image_params": params}))
    path = tmp_path / name
    Image.new("RGB", (8, 8), (30, 60, 90)).save(path, pnginfo=info)
    return str(path).replace('\\', '/')


def test_compute_hashes_swarmui_file_without_graph(sg, tmp_path):
    # Arrange: two SwarmUI files, same prompt, different models.
    p1 = _make_swarm_png(tmp_path, "s1.png", "a red fox", "modelA")
    p2 = _make_swarm_png(tmp_path, "s2.png", "a red fox", "modelB")

    # Act
    wf1, pr1, md1 = sg.compute_workflow_hashes(p1)
    wf2, pr2, md2 = sg.compute_workflow_hashes(p2)

    # Assert: all three identities without any ComfyUI graph; prompt is
    # shared, architecture and model set follow the model.
    assert wf1 and pr1 and md1
    assert pr1 == pr2
    assert wf1 != wf2
    assert md1 != md2


_PIPE_BASE = {
    "prompt": "a fox", "model": "modelP", "seed": 1, "steps": 20,
    "cfgscale": 7.0, "width": 64, "height": 64,
}


def test_workflow_identity_follows_param_shape(sg, tmp_path):
    """SwarmUI's parameter set shapes the ComfyUI workflow it generates, so
    parameter presence is architecture while seeds/steps/CFG are not — and
    the model-set identity ignores the pipeline shape entirely."""
    # Arrange
    p1 = _make_swarm_png_params(tmp_path, "p1.png", dict(_PIPE_BASE))
    p2 = _make_swarm_png_params(tmp_path, "p2.png", dict(_PIPE_BASE, seed=999, steps=50, cfgscale=3.5))
    p3 = _make_swarm_png_params(tmp_path, "p3.png", dict(_PIPE_BASE, refinermodel="refinerX"))

    # Act
    wf1, _, md1 = sg.compute_workflow_hashes(p1)
    wf2, _, md2 = sg.compute_workflow_hashes(p2)
    wf3, _, md3 = sg.compute_workflow_hashes(p3)

    # Assert
    assert wf1 and wf1 == wf2  # per-image knobs are not architecture
    assert wf1 != wf3          # an extra pipeline stage is
    assert md1 == md2 == md3   # model set: same checkpoint, no lora change


def test_schema_bump_marker_written_only_on_completion(sg, monkeypatch):
    """Regression: the marker used to be written BEFORE the migration ran,
    so a killed migration was skipped forever (models_hash empty gallery-wide
    with the marker claiming done)."""
    # Arrange: a hashed row and a stale schema marker.
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'schemaf', has_workflow=0, workflow_hash='oldF', prompt_hash='oldP')
        conn.execute("INSERT OR REPLACE INTO ai_metadata (key, value, updated_at) VALUES ('cluster_hash_schema', 'stale', 0)")
        conn.commit()
        calls = []
        monkeypatch.setattr(sg, 'ensure_cluster_backfill_async',
                            lambda force_all=False, on_complete=None: calls.append((force_all, on_complete)) or True)

        # Act: marker mismatch -> forced re-hash requested; marker NOT yet earned.
        sg.check_and_update_workflow_hashes(conn)
        assert len(calls) == 1 and calls[0][0] is True
        assert conn.execute("SELECT value FROM ai_metadata WHERE key = 'cluster_hash_schema'").fetchone()[0] == 'stale'

        # The completion hook records it; the next check runs the normal backfill.
        calls[0][1]()
        sg.check_and_update_workflow_hashes(conn)
        assert conn.execute("SELECT value FROM ai_metadata WHERE key = 'cluster_hash_schema'").fetchone()[0] == sg.CLUSTER_HASH_SCHEMA
        assert calls[1] == (False, None)


def test_backfill_abort_suppresses_completion_hook(sg, monkeypatch):
    import time as _time

    fired = []

    def _wait_done():
        for _ in range(100):
            if not sg._CLUSTER_BACKFILL_STATE['running']:
                return
            _time.sleep(0.05)

    # An aborted run must not fire the completion hook...
    monkeypatch.setattr(sg, 'backfill_unhashed_workflows',
                        lambda conn=None, force_all=False: sg._CLUSTER_BACKFILL_STATE.update(aborted=True) or 0)
    assert sg._real_ensure(force_all=True, on_complete=lambda: fired.append('aborted-run')) is True
    _wait_done()
    assert fired == []

    # ...and a clean run must.
    monkeypatch.setattr(sg, 'backfill_unhashed_workflows',
                        lambda conn=None, force_all=False: sg._CLUSTER_BACKFILL_STATE.update(aborted=False) or 0)
    assert sg._real_ensure(on_complete=lambda: fired.append('clean-run')) is True
    _wait_done()
    assert fired == ['clean-run']


def test_clustering_trigger_is_async_not_inline(sg, monkeypatch):
    """A clustering request with unhashed rows must kick the background
    worker and return partial results immediately, never hash inline."""
    # Arrange: one pending (unhashed) row and one already-clustered row.
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'apend', has_workflow=0, workflow_hash='', prompt_hash='')
        _insert_file(conn, 'adone', has_workflow=0, workflow_hash='archAsync', prompt_hash='pA')
    kicked = []
    monkeypatch.setattr(sg, 'ensure_cluster_backfill_async',
                        lambda force_all=False: kicked.append(force_all) or True)

    def _forbidden(conn=None, force_all=False):
        raise AssertionError("backfill ran inline during a clustering request")
    monkeypatch.setattr(sg, 'backfill_unhashed_workflows', _forbidden)

    # Act
    with sg.app.test_request_context('/'):
        result = sg.process_clustering([], 'workflow', 'date_desc', None, 'global')

    # Assert: worker kicked once, partial results served.
    assert kicked == [False]
    ids = {f['id'] for f in result if f['id'].startswith(_PREFIX)}
    assert _PREFIX + 'adone' in ids
    assert _PREFIX + 'apend' not in ids


def test_ensure_backfill_collapses_concurrent_runs(sg, monkeypatch):
    import threading
    release = threading.Event()
    started = threading.Event()

    def _slow_backfill(conn=None, force_all=False):
        started.set()
        release.wait(timeout=5)
        return 0
    monkeypatch.setattr(sg, 'backfill_unhashed_workflows', _slow_backfill)

    # Act: first call starts a run, second collapses while it is in flight.
    # (The sg fixture stubs the dispatcher; this test targets the real one.)
    assert sg._real_ensure() is True
    assert started.wait(timeout=5)
    assert sg._real_ensure() is False
    release.set()

    # The state flag clears once the thread finishes (poll briefly).
    import time as _t
    for _ in range(100):
        if not sg._CLUSTER_BACKFILL_STATE['running']:
            break
        _t.sleep(0.05)
    assert sg._CLUSTER_BACKFILL_STATE['running'] is False


def test_cluster_hash_status_endpoint(sg):
    # Arrange: one pending row.
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'statp', has_workflow=0, workflow_hash='', prompt_hash='')

    # Act
    data = sg.app.test_client().get('/galleryout/api/cluster_hash_status').get_json()

    # Assert
    assert data['status'] == 'success'
    assert data['pending'] >= 1
    assert 'running' in data and 'done' in data and 'total' in data


def test_models_mode_global_clustering(sg):
    # Arrange: same model set across two different architectures.
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'md1', has_workflow=0, workflow_hash='archA', prompt_hash='p1', models_hash='setX')
        _insert_file(conn, 'md2', has_workflow=0, workflow_hash='archB', prompt_hash='p2', models_hash='setX')
        _insert_file(conn, 'md3', has_workflow=0, workflow_hash='archC', prompt_hash='p3', models_hash='setY')

    # Act
    with sg.app.test_request_context('/'):
        all_sets = sg.process_clustering([], 'models', 'date_desc', None, 'global')
        targeted = sg.process_clustering([], 'models', 'date_desc', _PREFIX + 'md1', 'global')

    # Assert
    ids_all = {f['id'] for f in all_sets if f['id'].startswith(_PREFIX)}
    ids_target = {f['id'] for f in targeted if f['id'].startswith(_PREFIX)}
    assert ids_all == {_PREFIX + 'md1', _PREFIX + 'md2', _PREFIX + 'md3'}
    assert ids_target == {_PREFIX + 'md1', _PREFIX + 'md2'}


def test_global_clustering_includes_foreign_files_and_bases_differ(sg):
    """The user-visible symptom: both bases returned the same (ComfyUI-only)
    population. Foreign rows (has_workflow=0) with hashes must be clustered,
    and the two bases must reflect their own hash columns."""
    # Arrange: two swarm files sharing a prompt but not an architecture,
    # plus one with an architecture twin and no prompt.
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'sw1', has_workflow=0, workflow_hash='archX', prompt_hash='promptQ')
        _insert_file(conn, 'sw2', has_workflow=0, workflow_hash='archY', prompt_hash='promptQ')
        _insert_file(conn, 'sw3', has_workflow=0, workflow_hash='archX', prompt_hash='')

    # Act
    with sg.app.test_request_context('/'):
        by_arch = sg.process_clustering([], 'workflow', 'date_desc', None, 'global')
        by_prompt = sg.process_clustering([], 'prompt', 'date_desc', None, 'global')

    # Assert
    arch_ids = {f['id'] for f in by_arch if f['id'].startswith(_PREFIX)}
    prompt_ids = {f['id'] for f in by_prompt if f['id'].startswith(_PREFIX)}
    assert arch_ids == {_PREFIX + 'sw1', _PREFIX + 'sw2', _PREFIX + 'sw3'}
    assert prompt_ids == {_PREFIX + 'sw1', _PREFIX + 'sw2'}
    assert arch_ids != prompt_ids  # the two bases must not mirror each other


def test_global_target_cluster_works_for_foreign_file(sg):
    # Arrange
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'ft', has_workflow=0, workflow_hash='archT', prompt_hash='pT')
        _insert_file(conn, 'ft2', has_workflow=0, workflow_hash='archT', prompt_hash='pOther')
        _insert_file(conn, 'ftno', has_workflow=0, workflow_hash='archElse', prompt_hash='pT2')

    # Act
    with sg.app.test_request_context('/'):
        result = sg.process_clustering([], 'workflow', 'date_desc', _PREFIX + 'ft', 'global')

    # Assert
    ids = {f['id'] for f in result if f['id'].startswith(_PREFIX)}
    assert ids == {_PREFIX + 'ft', _PREFIX + 'ft2'}


def test_backfill_selects_foreign_image_rows(sg, tmp_path, monkeypatch):
    # Arrange: an unhashed image row without a ComfyUI graph.
    target = tmp_path / "sw.png"
    target.write_bytes(b"png")
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'fimg', path=str(target).replace('\\', '/'), has_workflow=0)
        monkeypatch.setattr(sg, 'compute_workflow_hashes', lambda p: ('archF', 'prF', 'mdF'))

        # Act
        updated = sg.backfill_unhashed_workflows(conn)

        # Assert
        assert updated >= 1
        row = conn.execute(
            "SELECT workflow_hash, prompt_hash FROM files WHERE id = ?", (_PREFIX + 'fimg',)
        ).fetchone()
        assert (row['workflow_hash'], row['prompt_hash']) == ('archF', 'prF')


def test_backfill_hashes_and_prompts_real_swarm_file(sg, tmp_path):
    """End-to-end at the backfill seam: a real SwarmUI PNG (no ComfyUI graph)
    gets cluster hashes AND a searchable workflow_prompt in one pass."""
    # Arrange
    path = _make_swarm_png(tmp_path, "real_swarm.png", "a lighthouse at dusk", "modelReal")
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'realsw', path=path, has_workflow=0)

        # Act
        updated = sg.backfill_unhashed_workflows(conn)

        # Assert
        assert updated >= 1
        row = conn.execute(
            "SELECT workflow_hash, prompt_hash, models_hash, workflow_prompt, hash_failed FROM files WHERE id = ?",
            (_PREFIX + 'realsw',),
        ).fetchone()
        assert row['workflow_hash'] != ''
        assert row['prompt_hash'] != ''
        assert row['models_hash'] != ''
        assert row['workflow_prompt'] == "a lighthouse at dusk"
        assert row['hash_failed'] == 0


def test_backfill_process_pool_path_end_to_end(sg, tmp_path, monkeypatch):
    """Regression for the GIL freeze: large runs must hash in worker
    PROCESSES (in-process threads starved the web server). Force the process
    path at size 1 and prove the pickling/spawn/result plumbing works with
    the real hasher."""
    # Arrange
    monkeypatch.setattr(sg, '_BACKFILL_PROCESS_THRESHOLD', 1)
    path = _make_swarm_png(tmp_path, "proc_swarm.png", "a comet over the sea", "modelProc")
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'procsw', path=path, has_workflow=0)

        # Act
        updated = sg.backfill_unhashed_workflows(conn)

        # Assert: hashed by a spawned worker process.
        assert updated >= 1
        row = conn.execute(
            "SELECT workflow_hash, prompt_hash, models_hash, workflow_prompt FROM files WHERE id = ?",
            (_PREFIX + 'procsw',),
        ).fetchone()
        assert row['workflow_hash'] and row['prompt_hash'] and row['models_hash']
        assert row['workflow_prompt'] == "a comet over the sea"


def test_backfill_does_not_reselect_partially_hashed_rows(sg, tmp_path, monkeypatch):
    """A file that yielded only a prompt hash is complete, not pending —
    selecting on workflow_hash alone would re-scan it on every request."""
    # Arrange
    target = tmp_path / "ponly.png"
    target.write_bytes(b"png")
    calls = []
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'pdone', path=str(target).replace('\\', '/'),
                     has_workflow=0, workflow_hash='', prompt_hash='prOnly')
        monkeypatch.setattr(sg, 'compute_workflow_hashes',
                            lambda p: calls.append(p) or ('', '', ''))

        # Act
        sg.backfill_unhashed_workflows(conn)

        # Assert: the partially hashed row was never re-scanned.
        assert all(_PREFIX + 'pdone' not in c for c in calls)
        row = conn.execute(
            "SELECT prompt_hash, hash_failed FROM files WHERE id = ?", (_PREFIX + 'pdone',)
        ).fetchone()
        assert (row['prompt_hash'], row['hash_failed']) == ('prOnly', 0)


# --- backfill_unhashed_workflows: failure marking ---------------------------


def test_backfill_marks_missing_and_unparseable_files_failed_once(sg, tmp_path):
    # Arrange: one row whose file vanished, one whose file has no workflow.
    plain = tmp_path / "plain.png"
    plain.write_bytes(b"not really a workflow png")
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'gone', path="C:/nonexistent/wfc/gone.png")
        _insert_file(conn, 'noise', path=str(plain).replace('\\', '/'))

        # Act
        first = sg.backfill_unhashed_workflows(conn)

        # Assert: neither hashed; both marked failed; the clustering trigger
        # condition is now quiet, so no further backfills fire.
        assert first == 0
        flags = dict(conn.execute(
            "SELECT id, hash_failed FROM files WHERE id LIKE ?", (_PREFIX + '%',)
        ).fetchall())
        assert flags[_PREFIX + 'gone'] == 1
        assert flags[_PREFIX + 'noise'] == 1
        pending = conn.execute(
            "SELECT COUNT(*) FROM files WHERE has_workflow = 1 AND (workflow_hash IS NULL OR workflow_hash = '') AND hash_failed = 0 AND id LIKE ?",
            (_PREFIX + '%',),
        ).fetchone()[0]
        assert pending == 0

        # Act again: nothing selected, nothing re-scanned.
        assert sg.backfill_unhashed_workflows(conn) == 0


def test_backfill_force_all_retries_failed_rows(sg, tmp_path, monkeypatch):
    # Arrange: a previously-failed row whose file now hashes successfully.
    target = tmp_path / "ok.png"
    target.write_bytes(b"png")
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'retry', path=str(target).replace('\\', '/'), hash_failed=1)
        monkeypatch.setattr(sg, 'compute_workflow_hashes', lambda p: ('wfNEW', 'prNEW', 'mdNEW'))

        # Act
        updated = sg.backfill_unhashed_workflows(conn, force_all=True)

        # Assert: hashes written and the failure flag cleared.
        assert updated >= 1
        row = conn.execute(
            "SELECT workflow_hash, prompt_hash, models_hash, hash_failed FROM files WHERE id = ?",
            (_PREFIX + 'retry',),
        ).fetchone()
        assert (row['workflow_hash'], row['prompt_hash'], row['models_hash'], row['hash_failed']) == ('wfNEW', 'prNEW', 'mdNEW', 0)


# --- compute_workflow_hashes: no synthetic prompt hash ----------------------

_API_WORKFLOW = (
    '{"1": {"class_type": "KSampler", "inputs": {"model": ["2", 0]}}, '
    '"2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}}}'
)


def test_compute_hashes_promptless_workflow_has_empty_prompt_hash(sg, tmp_path, monkeypatch):
    # Arrange: a real file whose embedded workflow has no positive prompt.
    f = tmp_path / "wf.png"
    f.write_bytes(b"png")
    monkeypatch.setattr(sg, 'extract_workflow', lambda path, target_type=None: _API_WORKFLOW)
    monkeypatch.setattr(sg, 'extract_workflow_prompt_string', lambda wf_json: '')

    # Act
    wf_hash, prompt_hash, _ = sg.compute_workflow_hashes(str(f))

    # Assert: architecture hashed, prompt hash stays empty (no synthesis).
    assert wf_hash != ''
    assert prompt_hash == ''


def test_compute_hashes_prompt_hash_is_normalized_md5_of_prompt(sg, tmp_path, monkeypatch):
    # Arrange
    f = tmp_path / "wf.png"
    f.write_bytes(b"png")
    monkeypatch.setattr(sg, 'extract_workflow', lambda path, target_type=None: _API_WORKFLOW)
    monkeypatch.setattr(sg, 'extract_workflow_prompt_string', lambda wf_json: '  A Cyberpunk STREET  ')

    # Act
    _, prompt_hash, _ = sg.compute_workflow_hashes(str(f))

    # Assert: stripped + lowercased before hashing.
    assert prompt_hash == hashlib.md5(b"a cyberpunk street").hexdigest()


# --- clear_synthetic_prompt_hashes ------------------------------------------


def test_clear_synthetic_prompt_hashes_removes_only_synthetic_values(sg):
    # Arrange: one legacy synthetic row, one genuine prompt hash.
    synthetic_val = hashlib.md5(("wfLEG" + "_prompt").encode('utf-8')).hexdigest()
    with sg.get_db_connection() as conn:
        _insert_file(conn, 'syn', workflow_hash='wfLEG', prompt_hash=synthetic_val)
        _insert_file(conn, 'real', workflow_hash='wfLEG2', prompt_hash='genuinehash')

        # Act
        cleared = sg.clear_synthetic_prompt_hashes(conn)

        # Assert: synthetic cleared, genuine untouched, second run is a no-op.
        assert cleared == 1
        rows = dict(conn.execute(
            "SELECT id, prompt_hash FROM files WHERE id LIKE ?", (_PREFIX + '%',)
        ).fetchall())
        assert rows[_PREFIX + 'syn'] == ''
        assert rows[_PREFIX + 'real'] == 'genuinehash'
        assert sg.clear_synthetic_prompt_hashes(conn) == 0


# --- ViewSnapshotStore ------------------------------------------------------


def test_snapshot_store_concurrent_views_do_not_clobber(sg):
    # Arrange: two renders (two tabs) snapshot different views.
    store = sg.ViewSnapshotStore(capacity=8)
    files_a = [{'id': 'a'}]
    files_b = [{'id': 'b'}]

    # Act
    token_a = store.put('', files_a)
    token_b = store.put('', files_b)

    # Assert: each token still resolves to its own view.
    assert store.get(token_a, '') == files_a
    assert store.get(token_b, '') == files_b


def test_snapshot_store_owner_mismatch_and_unknown_token_return_none(sg):
    # Arrange
    store = sg.ViewSnapshotStore(capacity=8)
    token = store.put('user-1', [{'id': 'a'}])

    # Act / Assert: another user's session cannot read the snapshot.
    assert store.get(token, 'user-2') is None
    assert store.get('no-such-token', 'user-1') is None
    assert store.get(token, 'user-1') == [{'id': 'a'}]


def test_snapshot_store_evicts_least_recently_used(sg):
    # Arrange
    store = sg.ViewSnapshotStore(capacity=2)
    token_a = store.put('', ['a'])
    token_b = store.put('', ['b'])

    # Act: touching A makes B the eviction candidate for the third insert.
    store.get(token_a, '')
    token_c = store.put('', ['c'])

    # Assert
    assert store.get(token_a, '') == ['a']
    assert store.get(token_b, '') is None
    assert store.get(token_c, '') == ['c']


# --- /galleryout/load_more and /api/current_view_ids ------------------------


def test_load_more_serves_only_its_own_snapshot(sg):
    # Arrange
    client = sg.app.test_client()
    files = [{'id': f'f{i}'} for i in range(3)]
    token = sg.VIEW_SNAPSHOTS.put('', files)

    # Act
    page = client.get(f"/galleryout/load_more?view_token={token}&offset=1").get_json()
    beyond = client.get(f"/galleryout/load_more?view_token={token}&offset=99").get_json()

    # Assert
    assert [f['id'] for f in page['files']] == ['f1', 'f2']
    assert beyond['files'] == []
    assert 'stale' not in page


def test_load_more_with_unknown_token_signals_stale_instead_of_guessing(sg):
    # Arrange
    client = sg.app.test_client()

    # Act
    data = client.get("/galleryout/load_more?view_token=bogus&offset=0").get_json()

    # Assert: never another view's files; the client is told to re-render.
    assert data == {'files': [], 'stale': True}


def test_current_view_ids_requires_a_live_snapshot(sg):
    # Arrange
    client = sg.app.test_client()
    token = sg.VIEW_SNAPSHOTS.put('', [{'id': 'x1'}, {'id': 'x2'}])

    # Act
    ok = client.get(f"/galleryout/api/current_view_ids?view_token={token}")
    stale = client.get("/galleryout/api/current_view_ids?view_token=bogus")

    # Assert
    assert ok.get_json() == {'status': 'success', 'ids': ['x1', 'x2']}
    assert stale.status_code == 410
    assert stale.get_json()['stale'] is True


def test_gallery_view_embeds_a_working_view_token(sg):
    # Arrange
    client = sg.app.test_client()

    # Act
    resp = client.get("/galleryout/view/_root_")
    match = re.search(r"const viewToken = '([^']+)'", resp.get_data(as_text=True))

    # Assert: the rendered page carries a token load_more accepts.
    assert resp.status_code == 200
    assert match is not None
    follow_up = client.get(f"/galleryout/load_more?view_token={match.group(1)}&offset=0").get_json()
    assert 'stale' not in follow_up
