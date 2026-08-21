"""Jobs run, stop, survive death, and cannot lie -- against the real schema.

The plan's gate, verbatim: cancel at item N, restart the process, resume
without repeating completed items or losing results. Every test here drives
`db.runner.run_next` -- the loop the application uses -- never the job
table directly, because the semantics under test are the runner's.
"""

from __future__ import annotations

import pathlib
import sqlite3

import pytest

from db import jobs, runner, scan

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"


@pytest.fixture(scope="module")
def ddl():
    return SCHEMA.read_text(encoding="utf-8")


@pytest.fixture
def db(ddl):
    conn = sqlite3.connect(":memory:")
    conn.executescript(ddl)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class Counter:
    """A handler that remembers exactly which items it performed."""

    def __init__(self, dies_on: int | None = None, fails_on: int | None = None):
        self.performed: list[int] = []
        self.dies_on = dies_on
        self.fails_on = fails_on

    def __call__(self, conn, item_id, payload, now):
        if item_id == self.dies_on:
            raise KeyboardInterrupt(f"process killed at item {item_id}")
        self.performed.append(item_id)
        if item_id == self.fails_on:
            raise ValueError(f"item {item_id} is broken")


def test_a_job_runs_to_done_and_the_row_says_so(db):
    job_id = jobs.submit(db, "embed", 0.0, items=[1, 2, 3, 4, 5])
    handler = Counter()
    turn = runner.run_next(db, "w1", 1.0, handlers={"embed": handler})
    assert turn == {"job": job_id, "state": "done", "did": 5, "failed": 0}
    assert handler.performed == [1, 2, 3, 4, 5]
    snap = jobs.snapshot(db, job_id)
    assert (snap["state"], snap["done_count"], snap["total"]) == ("done", 5, 5)


def test_an_expected_failure_lands_on_the_item_not_the_job(db):
    job_id = jobs.submit(db, "embed", 0.0, items=[1, 2, 3])
    turn = runner.run_next(db, "w1", 1.0, handlers={"embed": Counter(fails_on=2)})
    assert turn == {"job": job_id, "state": "done", "did": 3, "failed": 1}
    assert db.execute(
        "SELECT item_id, error FROM job_item WHERE job_id = ? AND state = 'failed'",
        (job_id,),
    ).fetchall() == [(2, "item 2 is broken")]


def test_a_failed_items_partial_writes_are_rolled_back(db):
    """An item that dies halfway must not leave its half in the rows.

    The failure record commits -- and before this was pinned, that commit
    carried everything the dead handler wrote before raising, because all
    DML on the connection rides one implicit transaction until commit
    (python/cpython Doc/library/sqlite3.rst:2709-2717). A torn item is a
    fact about the item; its half-written rows are not."""
    job_id = jobs.submit(db, "embed", 0.0, items=[1, 2])

    def half_writes(conn, item_id, payload, now):
        conn.execute("INSERT INTO region(x, y, w, h) VALUES(0.1, 0.1, 0.2, 0.2)")
        if item_id == 1:
            raise ValueError("died after writing")

    turn = runner.run_next(db, "w1", 1.0, handlers={"embed": half_writes})
    assert turn == {"job": job_id, "state": "done", "did": 2, "failed": 1}
    assert db.execute("SELECT count(*) FROM region").fetchone()[0] == 1, (
        "the failed item's partial write survived into the committed rows"
    )


def test_a_paused_job_is_resumed_by_the_very_next_turn(db):
    """A budget stop is deliberate, so the next turn continues immediately
    instead of waiting out a liveness lease meant for crashes."""
    job_id = jobs.submit(db, "embed", 0.0, items=[1, 2, 3, 4, 5])
    handler = Counter()
    first = runner.run_next(db, "w1", 1.0, handlers={"embed": handler}, budget=2)
    assert first == {"job": job_id, "state": "running", "did": 2, "failed": 0}
    second = runner.run_next(db, "w2", 1.5, handlers={"embed": handler})
    assert second == {"job": job_id, "state": "done", "did": 3, "failed": 0}
    assert handler.performed == [1, 2, 3, 4, 5]


def test_cancel_stops_at_an_item_boundary_and_repeats_nothing(db):
    job_id = jobs.submit(db, "embed", 0.0, items=[1, 2, 3, 4, 5])
    handler = Counter()
    runner.run_next(db, "w1", 1.0, handlers={"embed": handler}, budget=2)
    jobs.cancel(db, job_id)
    turn = runner.run_next(db, "w1", 2.0, handlers={"embed": handler})
    assert turn == {"job": job_id, "state": "cancelled", "did": 0, "failed": 0}
    assert handler.performed == [1, 2]
    snap = jobs.snapshot(db, job_id)
    assert (snap["state"], snap["done_count"]) == ("cancelled", 2)
    assert jobs.pending(db, job_id) == [3, 4, 5]


def test_the_off_switch_is_honoured_by_the_claim_itself(db):
    """The gate rides inside the claim's single UPDATE: a switch read a
    moment before the off-commit lands can otherwise win a job submitted
    a moment after -- observed live once per-request connection setup
    widened that gap. An absent row falls back to the given default."""
    jobs.submit(db, "embed", 0.0, items=[1])
    db.execute("INSERT INTO setting(key, value) VALUES('worker', 'off')")
    assert jobs.claim(db, "w1", 1.0, gate=("worker", "on")) is None, "an off switch lost to the claim"
    db.execute("UPDATE setting SET value = 'on' WHERE key = 'worker'")
    assert jobs.claim(db, "w1", 2.0, gate=("worker", "on")) is not None

    jobs.submit(db, "embed", 3.0, items=[2])
    db.execute("DELETE FROM setting WHERE key = 'worker'")
    assert jobs.claim(db, "w2", 4.0, gate=("worker", "off")) is None
    assert jobs.claim(db, "w2", 5.0, gate=("worker", "on")) is not None


def test_a_reclaim_waits_for_the_switch_like_everything_else(db):
    """An expired lease is still a job, and off means off: the reclaim
    queues behind the switch with everything else -- jobs are rows and
    lose nothing by waiting. Pinned so 'reclaims are safety, exempt them'
    cannot ship as a silent edit."""
    jobs.submit(db, "embed", 0.0, items=[1, 2])
    first = jobs.claim(db, "w1", 1.0)
    assert first is not None
    db.execute("INSERT INTO setting(key, value) VALUES('worker', 'off')")
    assert jobs.claim(db, "w2", 100.0, gate=("worker", "on")) is None, "an expired lease outranked the off switch"
    db.execute("UPDATE setting SET value = 'on' WHERE key = 'worker'")
    reclaimed = jobs.claim(db, "w2", 100.0, gate=("worker", "on"))
    assert reclaimed is not None
    assert reclaimed[0] == first[0]
    assert reclaimed[1] == first[1] + 1, "a reclaim must move the fence"


def test_a_killed_worker_strands_nothing(db):
    """The process dies mid-job. Nothing settles, nothing cleans up -- and
    after the lease runs out the next worker resumes from the items, having
    repeated only the item that was in flight when the lights went out."""
    job_id = jobs.submit(db, "embed", 0.0, items=[1, 2, 3, 4, 5])
    dying = Counter(dies_on=3)
    with pytest.raises(KeyboardInterrupt):
        runner.run_next(db, "w1", 1.0, handlers={"embed": dying})
    assert jobs.snapshot(db, job_id)["state"] == "running"

    # Too early: the lease still protects the (dead) owner.
    assert runner.run_next(db, "w2", 2.0, handlers={"embed": Counter()}) is None

    survivor = Counter()
    turn = runner.run_next(db, "w2", 2.0 + jobs.LEASE_SECONDS + 1, handlers={"embed": survivor})
    assert turn is not None
    assert turn["state"] == "done"
    assert survivor.performed == [3, 4, 5]
    assert jobs.pending(db, job_id) == []
    settled = db.execute(
        "SELECT count(*) FROM job_item WHERE job_id = ? AND state = 'done'",
        (job_id,),
    ).fetchone()[0]
    assert settled == 5


def test_an_evicted_worker_cannot_write_over_its_successor(db):
    job_id = jobs.submit(db, "embed", 0.0, items=[1, 2, 3])
    claimed = jobs.claim(db, "w1", 1.0)
    assert claimed is not None
    _, old_fence = claimed
    reclaimed = jobs.claim(db, "w2", 1.0 + jobs.LEASE_SECONDS + 1)
    assert reclaimed == (job_id, old_fence + 1)
    with pytest.raises(jobs.LeaseLost):
        jobs.finish_item(db, job_id, old_fence, 1)


def test_done_is_refused_while_items_are_outstanding(db):
    job_id = jobs.submit(db, "embed", 0.0, items=[1, 2])
    claimed = jobs.claim(db, "w1", 1.0)
    assert claimed is not None
    _, fence = claimed
    with pytest.raises(ValueError, match="unfinished"):
        jobs.settle(db, job_id, fence, "done", 2.0)


def test_a_job_no_worker_understands_fails_instead_of_vanishing(db):
    """`remix` is a legal kind this runner carries no handler for."""
    job_id = jobs.submit(db, "remix", 0.0, items=[1])
    turn = runner.run_next(db, "w1", 1.0, handlers={})
    assert turn == {"job": job_id, "state": "failed", "did": 0}
    assert "remix" in jobs.snapshot(db, job_id)["error"]


def test_the_verify_job_finds_bytes_changed_behind_the_librarys_back(db, tmp_path):
    """The application job, on real disk: three files scanned, one rewritten
    out of band, and the sweep names exactly that one without touching it."""
    from db import library

    root = tmp_path / "lib"
    root.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        (root / name).write_bytes(b"\x89PNG-of-" + name.encode())
    root_id = library.add_root(db, str(root), "library", 0.0)
    scan.scan(db, root_id, str(root), 0.0)

    (root / "b.png").write_bytes(b"\x89PNG-REPLACED")

    job_id = runner.submit_verify(db, 1.0)
    turn = runner.run_next(db, "w1", 2.0)
    assert turn == {"job": job_id, "state": "done", "did": 3, "failed": 1}
    verdicts = db.execute(
        "SELECT f.name, i.state FROM job_item i JOIN file f ON f.id = i.item_id WHERE i.job_id = ? ORDER BY f.name",
        (job_id,),
    ).fetchall()
    assert verdicts == [("a.png", "done"), ("b.png", "failed"), ("c.png", "done")]
    said = db.execute("SELECT error FROM job_item WHERE job_id = ? AND state = 'failed'", (job_id,)).fetchone()[0]
    assert "bytes changed" in said
    # A finding, never a mutation: the recorded hash still says what was
    # recorded, so the person decides what happens to the file.
    assert (
        db.execute(
            "SELECT count(*) FROM file f JOIN job_item i ON i.item_id = f.id"
            " WHERE i.job_id = ? AND f.content_sha256 IS NULL",
            (job_id,),
        ).fetchone()[0]
        == 0
    )


def _scanned_file_without_a_hash(db, tmp_path) -> int:
    """One real scanned file whose row carries no content hash."""
    from db import library

    root = tmp_path / "lib"
    root.mkdir()
    (root / "ghost.png").write_bytes(b"\x89PNG-ghost")
    root_id = library.add_root(db, str(root), "library", 0.0)
    scan.scan(db, root_id, str(root), 0.0)
    db.execute("UPDATE file SET content_sha256 = NULL")
    return db.execute("SELECT id FROM file").fetchone()[0]


def test_an_unknown_derive_fails_the_item_by_name_never_runs_a_guess(db, tmp_path):
    """The 'hash' dispatch fell through: any unrecognized derive value ran
    the integrity sweep -- a DIFFERENT job -- over items that legitimately
    include files with no recorded sha, and the sweep's mismatch message
    sliced that NULL: TypeError, which is outside ITEM_FAILURES, so the
    worker turn died, the job stayed running, and the lease cycled the
    crash forever. A corrupted payload is an item finding, by name."""
    file_id = _scanned_file_without_a_hash(db, tmp_path)
    job_id = jobs.submit(db, "hash", 0.0, payload={"derive": "prceptual"}, items=[file_id])
    turn = runner.run_next(db, "w1", 1.0)
    assert turn == {"job": job_id, "state": "done", "did": 1, "failed": 1}
    (said,) = db.execute("SELECT error FROM job_item WHERE job_id = ? AND item_id = ?", (job_id, file_id)).fetchone()
    assert "prceptual" in said, f"the item must name the bad payload, not whatever job ran instead: {said}"


def test_the_integrity_sweep_names_a_file_with_no_recorded_hash(db, tmp_path):
    """A bare 'hash' job reaching a sha-less row is a finding about the
    row -- there is nothing to verify against -- never the TypeError crash
    loop the message slice used to raise."""
    file_id = _scanned_file_without_a_hash(db, tmp_path)
    job_id = jobs.submit(db, "hash", 0.0, items=[file_id])
    turn = runner.run_next(db, "w1", 1.0)
    assert turn == {"job": job_id, "state": "done", "did": 1, "failed": 1}
    (said,) = db.execute("SELECT error FROM job_item WHERE job_id = ? AND item_id = ?", (job_id, file_id)).fetchone()
    assert "no recorded hash" in said


def test_a_degenerate_dupe_radius_is_refused_at_submit(db):
    """Two random 64-bit hashes disagree on 32 bits on average, so radius
    32 admits the average unrelated pair: range_search materializes the
    O(n^2) all-pairs result, and MemoryError is not an item failure -- the
    job wedges into the same lease-cycling crash loop. The dial stops
    before the cliff, at submit, where a bad value is a refused request."""
    from db import settings

    settings.put(db, "dupe_threshold", "32")
    with pytest.raises(ValueError, match="31"):
        runner.submit_dupes(db, 0.0)
    settings.put(db, "dupe_threshold", "31")
    assert runner.submit_dupes(db, 0.0) > 0


def _two_files(db, tmp_path) -> list[int]:
    from db import library

    root = tmp_path / "lib"
    root.mkdir()
    for name in ("a.png", "b.png"):
        (root / name).write_bytes(b"\x89PNG-" + name.encode())
    root_id = library.add_root(db, str(root), "library", 0.0)
    scan.scan(db, root_id, str(root), 0.0)
    return [row[0] for row in db.execute("SELECT id FROM file ORDER BY id")]


def test_dhash_vetoes_a_phash_pair_whose_structure_disagrees(db, tmp_path):
    """The second opinion: pHash proposes (global composition), dHash
    verifies (local gradient structure). Identical pHash with wildly
    different dHash is similar composition over different content -- not
    a duplicate, and the veto is independent evidence, not a re-vote."""
    from db import derived, settings

    files = _two_files(db, tmp_path)
    derived.record_hash(db, files[0], "aa", 0.0, phash64=0b1, dhash64=0)
    derived.record_hash(db, files[1], "bb", 0.0, phash64=0b1, dhash64=-1)  # all 64 bits apart
    db.commit()

    runner.run_next(db, "w1", 1.0)  # drain any queued work before submitting
    job_id = runner.submit_dupes(db, 2.0)
    turn = runner.run_next(db, "w1", 3.0)
    assert turn == {"job": job_id, "state": "done", "did": 1, "failed": 0}
    assert db.execute("SELECT count(*) FROM derived_dupe_group").fetchone()[0] == 0, (
        "structurally different pictures grouped on composition alone"
    )

    settings.put(db, "dupe_dhash_verify", "off")
    job_id = runner.submit_dupes(db, 4.0)
    turn = runner.run_next(db, "w1", 5.0)
    assert turn == {"job": job_id, "state": "done", "did": 1, "failed": 0}
    assert db.execute("SELECT count(*) FROM derived_dupe_group").fetchone()[0] == 2, (
        "with verification off, pHash alone must decide"
    )


def test_a_bad_dhash_verify_setting_is_refused_at_submit(db):
    from db import settings

    settings.put(db, "dupe_dhash_verify", "64")
    with pytest.raises(ValueError, match="63"):
        runner.submit_dupes(db, 0.0)
    settings.put(db, "dupe_dhash_verify", "sideways")
    with pytest.raises(ValueError, match="sideways"):
        runner.submit_dupes(db, 0.0)
    settings.put(db, "dupe_dhash_verify", "off")
    assert runner.submit_dupes(db, 0.0) > 0


def _pictures(db, tmp_path, sizes: dict[str, int]) -> dict[str, int]:
    """Real decodable images whose byte sizes the test controls -- the
    dupe policy picks its best member by pixels, then bytes."""
    from PIL import Image

    from db import library

    root = tmp_path / "lib"
    root.mkdir()
    for name, pad in sizes.items():
        Image.effect_noise((32, 32), 40).convert("RGB").save(root / name)
        with (root / name).open("ab") as grow:
            grow.write(b"\x00" * pad)
    root_id = library.add_root(db, str(root), "library", 0.0)
    scan.scan(db, root_id, str(root), 0.0)
    return dict(db.execute("SELECT name, id FROM file"))


def test_a_duplicate_group_never_chains_past_its_best(db, tmp_path):
    """A~B and B~C within threshold with A and C far apart is a chain,
    not a duplicate group: every member is checked against the BEST, and
    a member the canonical check rejects is dropped -- related, maybe,
    but not a duplicate this pass can claim."""
    from db import derived

    files = _pictures(db, tmp_path, {"a.png": 900, "b.png": 500, "c.png": 100})
    hashes = {"a.png": 0, "b.png": 0b1111, "c.png": 0b11111111}
    for name, value in hashes.items():
        derived.record_hash(db, files[name], "aa", 0.0, phash64=value, dhash64=0)
    db.commit()

    job_id = runner.submit_dupes(db, 1.0)  # dupe_threshold default 4: A~B=4, B~C=4, A~C=8
    turn = runner.run_next(db, "w1", 2.0)
    assert turn == {"job": job_id, "state": "done", "did": 1, "failed": 0}
    told = {
        n: db.execute(
            "SELECT distance, is_best, verified FROM derived_dupe_group WHERE file_id = ?", (files[n],)
        ).fetchone()
        for n in files
    }
    assert told["c.png"] is None, "the chain smuggled C into A's duplicate group"
    assert told["a.png"] == (0, 1, 1)
    assert told["b.png"] == (4, 0, 1), "distance must be measured to the best, and dHash agreement recorded"


def test_the_embed_job_fills_the_joint_space_with_provenance(db, tmp_path, monkeypatch):
    """The 'embed' kind, wired: every present picture gets one vector in
    the joint image/text space, keyed by the immutable space row naming
    the exact model, checkpoint and preprocessing that computed it."""
    import numpy as np

    from db import similarity
    from vision import semantic

    class Fake:
        dimensions = 8

        def space(self):
            return similarity.semantic_space("ViT-B-32", "laion2b_s34b_b79k", 8)

        def encode_media(self, media):
            assert media.kind == "image"
            assert media.frame() is not None, "the seam must hand over decodable media"
            rng = np.random.default_rng(7)
            v = rng.normal(size=8).astype(np.float32)
            return v / np.linalg.norm(v)

    monkeypatch.setattr(semantic, "encoder", lambda *args, **kwargs: Fake())
    files = _pictures(db, tmp_path, {"a.png": 0, "b.png": 0})
    [job_id] = runner.submit_embed(db, 0.0, models_dir=str(tmp_path))
    turn = runner.run_next(db, "w1", 1.0)
    assert turn == {"job": job_id, "state": "done", "did": 2, "failed": 0}
    told = db.execute(
        "SELECT count(*), s.key, s.producer, s.producer_version, s.preprocess FROM derived_embedding e"
        " JOIN similarity_space s ON s.id = e.space_id GROUP BY s.id"
    ).fetchall()
    assert len(told) == 1
    count, key, producer, version, preprocess = told[0]
    assert count == len(files)
    assert key == "semantic.openclip.ViT-B-32.laion2b_s34b_b79k"
    assert (producer, version) == ("open_clip:ViT-B-32", "laion2b_s34b_b79k")
    assert preprocess == "open_clip.transforms"


def test_a_bad_semantic_model_setting_is_refused_at_submit(db):
    from db import settings

    settings.put(db, "semantic_model", "no-checkpoint-here")
    with pytest.raises(ValueError, match="no-checkpoint-here"):
        runner.submit_embed(db, 0.0, models_dir="x")
