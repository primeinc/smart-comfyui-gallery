"""Jobs run, stop, survive death, and cannot lie -- against the real schema.

The plan's gate, verbatim: cancel at item N, restart the process, resume
without repeating completed items or losing results. Every test here drives
`db.runner.run_next` -- the loop the application uses -- never the job
table directly, because the semantics under test are the runner's.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from db import connect, jobs, runner, scan

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"


@pytest.fixture(scope="module")
def ddl():
    return SCHEMA.read_text(encoding="utf-8")


@pytest.fixture
def db(ddl):
    conn = connect.memory()
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


@pytest.mark.slow
def test_the_ingest_sweep_queues_only_files_not_read_for_their_current_bytes(db, tmp_path):
    """A read is recorded against the bytes it was of: a second sweep has
    nothing to do, new bytes put the file back, `everything` puts them
    all back. A file with no metadata at all was still READ."""
    files = _pictures(db, tmp_path, {"a.png": 0, "b.png": 0})
    job_id = runner.submit_ingest(db, 0.0)
    assert job_id is not None
    assert runner.run_next(db, "w1", 1.0) == {"job": job_id, "state": "done", "did": 2, "failed": 0}
    rows = db.execute("SELECT ingested_sha256 IS NOT NULL, ingested_sha256 = content_sha256 FROM file").fetchall()
    assert rows == [(1, 1), (1, 1)], "plain pictures carry no recipe and were read all the same"

    assert runner.submit_ingest(db, 2.0) is None, "every file is read"
    db.execute("UPDATE file SET content_sha256 = ? WHERE id = ?", ("e" * 64, files["b.png"]))
    again = runner.submit_ingest(db, 3.0)
    assert [r[0] for r in db.execute("SELECT item_id FROM job_item WHERE job_id = ?", (again,))] == [files["b.png"]]
    whole = runner.submit_ingest(db, 4.0, everything=True)
    assert db.execute("SELECT count(*) FROM job_item WHERE job_id = ?", (whole,)).fetchone()[0] == 2


def test_the_phash_sweep_queues_only_pictures_without_a_current_fingerprint(db, tmp_path):
    """A second sweep has nothing to do; new bytes put a picture back;
    `everything` puts them all back. Detection's byproduct hashes count
    too: they sit in the same space under the same bytes."""
    files = _pictures(db, tmp_path, {"a.png": 0, "b.png": 0})
    job_id = runner.submit_phash(db, 0.0)
    assert job_id is not None
    assert runner.run_next(db, "w1", 1.0) == {"job": job_id, "state": "done", "did": 2, "failed": 0}

    assert runner.submit_phash(db, 2.0) is None, "every picture is fingerprinted"
    db.execute("UPDATE file SET content_sha256 = ? WHERE id = ?", ("e" * 64, files["b.png"]))
    again = runner.submit_phash(db, 3.0)
    assert [r[0] for r in db.execute("SELECT item_id FROM job_item WHERE job_id = ?", (again,))] == [files["b.png"]]
    whole = runner.submit_phash(db, 4.0, everything=True)
    assert db.execute("SELECT count(*) FROM job_item WHERE job_id = ?", (whole,)).fetchone()[0] == 2


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


def test_one_encoder_pass_covers_many_items_and_each_keeps_its_own_vector(db, tmp_path, monkeypatch):
    """An adapter that can encode a group is given one, and the items it
    covered commit exactly as if they had been encoded alone.

    The durable contract does not move: every item is still started,
    worked and settled on its own row, so the job stays resumable,
    cancellable at a boundary, and able to lose one picture without the
    rest. What changes is only WHEN the arithmetic happened.

    The vectors are checked per file, not merely counted. A group encode
    hands back a list, and a list zipped against the wrong items would
    give every picture a plausible vector belonging to its neighbour --
    a defect no count, and no assertion about dimensions, would catch.
    """
    from db import detect, oriented, similarity
    from vision import semantic

    calls: list[int] = []

    class Grouping:
        dimensions = 4

        def space(self):
            return similarity.semantic_space("ViT-B-32", "laion2b_s34b_b79k", 4)

        def _vector(self, frame):
            # The mean pixel is enough to tell these fixtures apart, and
            # it is a property of the PICTURE, so a misplaced vector is
            # visible rather than merely different.
            mean = float(np.asarray(frame.convert("RGB"), dtype=np.float64).mean())
            v = np.array([mean, 1.0, 0.0, 0.0], dtype=np.float32)
            return v / np.linalg.norm(v)

        def encode_media(self, media):
            calls.append(1)
            return self._vector(media.frame())

        def encode_many(self, framers):
            calls.append(len(framers))
            return [self._vector(framer()) for framer in framers]

    monkeypatch.setattr(semantic, "encoder", lambda *args, **kwargs: Grouping())
    files = _pictures(db, tmp_path, {f"p{i}.png": i for i in range(6)})
    [job_id] = runner.submit_embed(db, 0.0, models_dir=str(tmp_path))
    assert runner.run_next(db, "w1", 1.0) == {"job": job_id, "state": "done", "did": 6, "failed": 0}

    assert max(calls) > 1, "the adapter was never handed a group"
    assert sum(calls) == len(files), f"pictures were encoded more than once: {calls}"

    # every item settled on its own row, as before
    states = [row[0] for row in db.execute("SELECT state FROM job_item WHERE job_id = ?", (job_id,))]
    assert states == ["done"] * len(files)

    # and every file kept ITS OWN vector
    backend = Grouping()
    for name, file_id in files.items():
        blob = db.execute("SELECT vector FROM derived_embedding WHERE file_id = ?", (file_id,)).fetchone()
        assert blob is not None, f"{name} recorded nothing"
        stored = np.frombuffer(blob[0], dtype=np.float32)
        want = backend._vector(oriented.for_model(db, file_id, detect.path_of(db, file_id)))
        assert np.allclose(stored, want), f"{name} was given another picture's vector"


def test_two_jobs_over_the_same_files_never_read_each_other_vectors():
    """The held vectors are keyed by job, space AND file, all three.

    An earlier lookup searched every held job for a matching space and
    file, which made the real contract `space + file` while the storage
    key said otherwise. Two jobs over overlapping files in one space
    would have crossed -- and the reason it would usually have looked
    fine, that the same file in the same space encodes to nearly the same
    vector, is exactly what would have kept it hidden.
    """

    class Space:
        key = "semantic.openclip.ViT-B-32.laion2b_s34b_b79k"

    class Other:
        key = "semantic.qwen.something"

    other = Other
    held = runner._Ahead()
    held._held[(1, Space.key)] = {42: "job one's vector"}
    held._held[(2, Space.key)] = {42: "job two's vector"}
    held._held[(1, other.key)] = {42: "another space's vector"}

    assert held.take(1, Space(), 42) == "job one's vector"
    assert held.take(2, Space(), 42) == "job two's vector"
    assert held.take(1, other(), 42) == "another space's vector"
    assert held.take(3, Space(), 42) is None, "a job that held nothing must not read a neighbour's"
    assert held.take(1, Space(), 42) is None, "and a vector is taken once"

    held.forget(1)
    held._held[(2, Space.key)] = {7: "still job two's"}
    assert held.take(2, Space(), 7) == "still job two's", "forgetting one job leaves the others alone"


def test_a_batch_is_bounded_by_pixels_including_the_item_that_leads_it(db, tmp_path, monkeypatch):
    """`BATCH_MEGAPIXELS` bounds the WHOLE batch.

    The leader decodes like any other member and the item leading a batch
    pays for all of it, so leaving it out of the budget made the stated
    bound a bound on the followers only: a 100-megapixel leader plus 150
    of followers formed a 250-megapixel batch under a limit of 160.

    Cancellation is checked between items, so this bound is also the
    longest a stop request can wait.
    """
    from db import similarity
    from vision import semantic

    widest: list[int] = []

    class Grouping:
        dimensions = 4

        def space(self):
            return similarity.semantic_space("ViT-B-32", "laion2b_s34b_b79k", 4)

        def _v(self):
            v = np.ones(4, dtype=np.float32)
            return v / np.linalg.norm(v)

        def encode_media(self, media):
            widest.append(1)
            return self._v()

        def encode_many(self, framers):
            widest.append(len(framers))
            return [self._v() for _ in framers]

    monkeypatch.setattr(semantic, "encoder", lambda *args, **kwargs: Grouping())
    monkeypatch.setattr(runner, "BATCH_MEGAPIXELS", 100.0)
    files = _pictures(db, tmp_path, {f"p{i}.png": i for i in range(8)})
    # 40 megapixels each against a bound of 100. Charging the leader
    # leaves room for exactly one follower, so batches are of two.
    # WITHOUT charging it there is room for two followers and they would
    # be three -- which is what makes this test tell the two apart rather
    # than pass either way.
    db.execute("UPDATE file SET width = 5000, height = 8000")
    db.commit()

    [job_id] = runner.submit_embed(db, 0.0, models_dir=str(tmp_path))
    assert runner.run_next(db, "w1", 1.0) == {"job": job_id, "state": "done", "did": 8, "failed": 0}
    assert max(widest) == 2, f"batches of {max(widest)}: the leader's own pixels were not charged"
    assert sum(widest) == len(files), f"pictures were encoded more than once: {widest}"


def test_an_adapter_without_a_group_encoder_is_still_called_one_at_a_time(db, tmp_path, monkeypatch):
    """The fallback is not a special case, it is the older path intact.

    Qwen has no `encode_many`, and a provider added tomorrow will not
    have one either until somebody writes it. Nothing about the job may
    depend on it existing.
    """
    from db import similarity
    from vision import semantic

    seen: list[int] = []

    class Single:
        dimensions = 4

        def space(self):
            return similarity.semantic_space("ViT-B-32", "laion2b_s34b_b79k", 4)

        def encode_media(self, media):
            seen.append(1)
            v = np.ones(4, dtype=np.float32)
            return v / np.linalg.norm(v)

    monkeypatch.setattr(semantic, "encoder", lambda *args, **kwargs: Single())
    files = _pictures(db, tmp_path, {"a.png": 0, "b.png": 1, "c.png": 2})
    [job_id] = runner.submit_embed(db, 0.0, models_dir=str(tmp_path))
    assert runner.run_next(db, "w1", 1.0) == {"job": job_id, "state": "done", "did": 3, "failed": 0}
    assert len(seen) == len(files), "an adapter with no group encoder must be called once per picture"


def test_the_embed_sweep_queues_only_pictures_without_a_current_vector(db, tmp_path, monkeypatch):
    """A second sweep over an embedded library has nothing to do and
    queues no job; new bytes put a picture back; `everything` puts them
    all back. What counts as current is retrieval's own definition."""
    import numpy as np

    from db import similarity
    from vision import semantic

    class Fake:
        dimensions = 8

        def space(self):
            return similarity.semantic_space("ViT-B-32", "laion2b_s34b_b79k", 8)

        def encode_media(self, media):
            v = np.random.default_rng(3).normal(size=8).astype(np.float32)
            return v / np.linalg.norm(v)

    monkeypatch.setattr(semantic, "encoder", lambda *args, **kwargs: Fake())
    files = _pictures(db, tmp_path, {"a.png": 0, "b.png": 0})
    [job_id] = runner.submit_embed(db, 0.0, models_dir=str(tmp_path))
    assert runner.run_next(db, "w1", 1.0) == {"job": job_id, "state": "done", "did": 2, "failed": 0}

    assert runner.submit_embed(db, 2.0, models_dir=str(tmp_path)) == [], "every picture is current"
    db.execute("UPDATE file SET content_sha256 = ? WHERE id = ?", ("e" * 64, files["b.png"]))
    [again] = runner.submit_embed(db, 3.0, models_dir=str(tmp_path))
    assert [r[0] for r in db.execute("SELECT item_id FROM job_item WHERE job_id = ?", (again,))] == [files["b.png"]]
    [whole] = runner.submit_embed(db, 4.0, models_dir=str(tmp_path), everything=True)
    assert db.execute("SELECT count(*) FROM job_item WHERE job_id = ?", (whole,)).fetchone()[0] == 2


def test_a_bad_semantic_model_setting_is_refused_at_submit(db):
    from db import settings

    settings.put(db, "semantic_model", "no-checkpoint-here")
    with pytest.raises(ValueError, match="no-checkpoint-here"):
        runner.submit_embed(db, 0.0, models_dir="x")


def test_the_qwen_provider_is_a_named_space_and_a_parsed_choice(db):
    """The second provider exists end to end below the weights: the
    setting parses in the provider's OWN grammar (a Hugging Face repo
    id, not a fake model/checkpoint split), duplicates collapse instead
    of voting twice, and the space identity pins repo, revision and the
    whole preprocessing policy."""
    from db import retrieval, settings
    from vision import semantic
    from vision.semantic import qwen_vl

    settings.put(
        db,
        "semantic_model",
        "ViT-B-32/laion2b_s34b_b79k, qwen:Qwen/Qwen3-VL-Embedding-2B, ViT-B-32/laion2b_s34b_b79k",
    )
    assert retrieval.choices(db) == [
        ("openclip", "ViT-B-32", "laion2b_s34b_b79k"),
        ("qwen", "Qwen/Qwen3-VL-Embedding-2B", "main"),
    ], "a repeated entry must collapse, not weight its model twice"
    assert qwen_vl.parse("Qwen/Qwen3-VL-Embedding-8B@9f2f7e7") == ("Qwen/Qwen3-VL-Embedding-8B", "9f2f7e7")
    for malformed in ("just-a-name", "org/repo/extra", "org/repo@"):
        with pytest.raises(ValueError, match=r"repo id|revision"):
            qwen_vl.parse(malformed)

    # Space identity is exercised with a COMMIT, the only checkpoint the
    # real paths ever mint under: the backend pins a mutable ref to the
    # cached commit after weights land, and retrieval pins before it
    # probes the registry -- "main" never becomes producer_version.
    commit = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
    spec = semantic.space("qwen", "Qwen/Qwen3-VL-Embedding-2B", commit, 2048)
    assert spec.key == f"semantic.qwen.Qwen/Qwen3-VL-Embedding-2B.{commit}"
    assert (spec.producer, spec.producer_version) == ("qwen3vl:Qwen/Qwen3-VL-Embedding-2B", commit)
    assert (spec.preprocess, spec.dimensions, spec.metric) == ("qwen3vl.chat-template", 2048, "cosine")

    # The preprocess version is a PROPERTY of the policy, not a label:
    # it names the two packages whose code is the preprocessing, and any
    # edited knob or media instruction changes the digest, hence the
    # spec hash, hence the space. The QUERY instruction is deliberately
    # outside it -- rewording a query prompt must not force a re-embed
    # of a library whose stored vectors are all still valid.
    assert spec.preprocess_version.startswith("tf")
    assert "+qvu" in spec.preprocess_version
    assert spec.preprocess_version.endswith(
        qwen_vl.policy_digest(
            qwen_vl.MAX_LENGTH,
            qwen_vl.MIN_PIXELS,
            qwen_vl.MAX_PIXELS,
            qwen_vl.FPS,
            qwen_vl.MAX_FRAMES,
            qwen_vl.IMAGE_PATCH_SIZE,
            qwen_vl.DO_RESIZE,
            qwen_vl.POOLING,
            qwen_vl.NORMALIZATION,
            qwen_vl.MEDIA_INSTRUCTION,
        )
    )
    with_other_fps = qwen_vl.policy_digest(
        qwen_vl.MAX_LENGTH,
        qwen_vl.MIN_PIXELS,
        qwen_vl.MAX_PIXELS,
        2,
        qwen_vl.MAX_FRAMES,
        qwen_vl.IMAGE_PATCH_SIZE,
        qwen_vl.DO_RESIZE,
        qwen_vl.POOLING,
        qwen_vl.NORMALIZATION,
        qwen_vl.MEDIA_INSTRUCTION,
    )
    assert not spec.preprocess_version.endswith(with_other_fps), "a changed budget must change the space"

    settings.put(db, "semantic_model", "nobody:some-model/some-checkpoint")
    with pytest.raises(ValueError, match="nobody"):
        retrieval.choices(db)


def test_a_mutable_revision_pins_to_the_cached_commit(tmp_path):
    """`main` is a pointer, and a similarity space keyed by a pointer
    changes meaning the day upstream moves it. Against a cache in the
    hub layout, pin() resolves the branch to the snapshot commit; a
    commit passes through; an uncached branch has nothing to pin
    against and returns as given -- and only the embed path, which pins
    after weights land, ever mints a space."""
    from vision import semantic
    from vision.semantic import qwen_vl

    commit = "9f" * 20
    repo_dir = tmp_path / "models--Qwen--Qwen3-VL-Embedding-2B"
    snapshot = repo_dir / "snapshots" / commit
    snapshot.mkdir(parents=True)
    for name in ("model.safetensors", *qwen_vl._SNAPSHOT_FILES):
        (snapshot / name).write_text("{}", encoding="utf-8")
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text(commit, encoding="utf-8")

    assert qwen_vl.pin(str(tmp_path), "Qwen/Qwen3-VL-Embedding-2B", "main") == commit
    assert qwen_vl.pin(str(tmp_path), "Qwen/Qwen3-VL-Embedding-2B", commit) == commit
    assert qwen_vl.pin(str(tmp_path), "Qwen/Absent-Model", "main") == "main"
    assert semantic.pin("qwen", str(tmp_path), "Qwen/Qwen3-VL-Embedding-2B", "main") == commit
    assert semantic.pin("openclip", str(tmp_path), "ViT-B-32", "laion2b_s34b_b79k") == "laion2b_s34b_b79k", (
        "an open_clip tag already IS the identity and must pass through"
    )

    # A shard index whose named shards are not all present is NOT
    # provisioned -- 'complete' means every byte loading will touch.
    sharded = tmp_path / "models--Qwen--Qwen3-VL-Embedding-8B"
    shard_snap = sharded / "snapshots" / ("8b" * 20)
    shard_snap.mkdir(parents=True)
    (shard_snap / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}',
        encoding="utf-8",
    )
    (shard_snap / "model-00001-of-00002.safetensors").write_text("x", encoding="utf-8")
    for name in qwen_vl._SNAPSHOT_FILES:
        (shard_snap / name).write_text("{}", encoding="utf-8")
    (sharded / "refs").mkdir()
    (sharded / "refs" / "main").write_text("8b" * 20, encoding="utf-8")
    assert qwen_vl._cached_snapshot(str(tmp_path), "Qwen/Qwen3-VL-Embedding-8B", "main") is None, (
        "a missing shard must read as unprovisioned, not fail mid-inference"
    )
    (shard_snap / "model-00002-of-00002.safetensors").write_text("y", encoding="utf-8")
    assert qwen_vl._cached_snapshot(str(tmp_path), "Qwen/Qwen3-VL-Embedding-8B", "main") == str(shard_snap)


def test_one_failing_provider_costs_its_own_space_only(db, tmp_path, monkeypatch):
    """The reason embed is one job per space: a model that cannot load
    fails its own items, and the other provider's vectors commit
    untouched."""
    import numpy as np

    from db import settings, similarity
    from vision import semantic

    class Fake:
        dimensions = 8

        def space(self):
            return similarity.semantic_space("ViT-B-32", "laion2b_s34b_b79k", 8)

        def encode_media(self, media):
            rng = np.random.default_rng(3)
            v = rng.normal(size=8).astype(np.float32)
            return v / np.linalg.norm(v)

    def per_provider(provider, *args, **kwargs):
        if provider == "qwen":
            raise LookupError("Qwen3-VL-Embedding-2B/main is not provisioned")
        return Fake()

    monkeypatch.setattr(semantic, "encoder", per_provider)
    settings.put(db, "semantic_model", "ViT-B-32/laion2b_s34b_b79k, qwen:Qwen/Qwen3-VL-Embedding-2B")
    files = _pictures(db, tmp_path, {"a.png": 0, "b.png": 0})
    clip_job, qwen_job = runner.submit_embed(db, 0.0, models_dir=str(tmp_path))

    first = runner.run_next(db, "w1", 1.0)
    assert first == {"job": clip_job, "state": "done", "did": 2, "failed": 0}
    second = runner.run_next(db, "w1", 2.0)
    assert second == {"job": qwen_job, "state": "done", "did": 2, "failed": 2}

    kept = db.execute(
        "SELECT s.key, count(*) FROM derived_embedding e JOIN similarity_space s ON s.id = e.space_id GROUP BY s.id"
    ).fetchall()
    assert kept == [("semantic.openclip.ViT-B-32.laion2b_s34b_b79k", len(files))], (
        "the healthy provider's vectors must survive the broken one"
    )


def test_the_qwen_adapter_takes_the_native_link_for_video():
    """Video reaches the model as the FILE with sampling budgets -- the
    whole point of the media-aware seam. Calling the canonical poster
    frame instead would judge a clip by one picture and make the seam
    decoration."""
    from vision import semantic
    from vision.semantic import qwen_vl

    backend = object.__new__(qwen_vl.QwenBackend)
    seen: dict = {}

    def record(instruction: str, content: dict) -> np.ndarray:
        seen.update({"instruction": instruction, "content": content})
        return np.zeros(1, dtype=np.float32)

    backend._embed = record

    def never_the_frame():
        raise AssertionError("a video must go through its own path, not the poster frame")

    backend.encode_media(semantic.MediaRef(path="C:/pics/clip.mp4", kind="video", frame=never_the_frame))
    assert seen["instruction"] == qwen_vl.MEDIA_INSTRUCTION
    assert seen["content"]["type"] == "video"
    assert seen["content"]["video"] == "file://C:/pics/clip.mp4"
    assert (seen["content"]["fps"], seen["content"]["max_frames"]) == (qwen_vl.FPS, qwen_vl.MAX_FRAMES)


def test_a_failed_decode_fails_the_item_and_embeds_nothing():
    """The upstream wrapper swaps a failed decode for the literal text
    "NULL" to keep an evaluation batch alive; this adapter must NOT --
    a NULL vector in the space is a picture that answers queries about
    nothing. The failure propagates and no embedding call happens."""
    from vision import semantic
    from vision.semantic import qwen_vl

    backend = object.__new__(qwen_vl.QwenBackend)
    called: list = []

    def record(instruction: str, content: dict) -> np.ndarray:
        called.append(content)
        return np.zeros(1, dtype=np.float32)

    backend._embed = record

    def undecodable():
        raise ValueError("the bytes are not a picture")

    with pytest.raises(ValueError, match="not a picture"):
        backend.encode_media(semantic.MediaRef(path="C:/pics/x.png", kind="image", frame=undecodable))
    assert called == [], "a failed decode still reached the model"


def test_the_real_qwen_weights_answer_by_meaning():
    """Opt-in smoke test over the actual 2B checkpoint: proves the port
    is an embedding model and not a 2B-parameter random-number
    generator. Set RUN_QWEN_SMOKE to a models_dir holding
    Qwen/Qwen3-VL-Embedding-2B in HF cache layout."""
    import os

    models_dir = os.environ.get("RUN_QWEN_SMOKE", "")
    if not models_dir:
        pytest.skip("set RUN_QWEN_SMOKE=<models_dir> with the 2B weights cached to run")
    import numpy as np
    from PIL import Image, ImageDraw

    from vision import semantic

    encoder = semantic.encoder("qwen", models_dir, "Qwen/Qwen3-VL-Embedding-2B", "main", offline=True)
    assert encoder.dimensions == 2048
    picture = Image.new("RGB", (256, 256), (200, 60, 40))
    ImageDraw.Draw(picture).ellipse((60, 60, 196, 196), fill=(250, 220, 40))
    vector = encoder.encode_media(semantic.MediaRef(path="drawn.png", kind="image", frame=lambda: picture))
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 0.01
    match = float(np.dot(vector, encoder.encode_query("a bright yellow circle on a red background")))
    off = float(np.dot(vector, encoder.encode_query("a spreadsheet of quarterly financial figures")))
    assert match > off + 0.2, f"the joint space lost its meaning: match={match:.3f} off={off:.3f}"


# --- the runner says what it did ---------------------------------------------


def test_a_turn_logs_the_claim_and_the_settlement(db, caplog):
    import logging

    job_id = jobs.submit(db, "embed", 0.0, items=[1, 2, 3])

    with caplog.at_level(logging.INFO, logger="db.runner"):
        runner.run_next(db, "w1", 1.0, handlers={"embed": Counter()}, clock=lambda: 2.5)

    said = [record.getMessage() for record in caplog.records if record.name == "db.runner"]
    assert said[0] == f"job #{job_id} embed: claimed, 3 of 3 items pending"
    assert said[-1] == f"job #{job_id} embed: done, 3 items, 0 failed, 0.0s"


def test_a_failed_item_is_a_warning_naming_the_item_and_the_reason(db, caplog):
    import logging

    job_id = jobs.submit(db, "embed", 0.0, items=[1, 2, 3])

    with caplog.at_level(logging.INFO, logger="db.runner"):
        runner.run_next(db, "w1", 1.0, handlers={"embed": Counter(fails_on=2)})

    warned = [r for r in caplog.records if r.name == "db.runner" and r.levelno == logging.WARNING]
    assert [r.getMessage() for r in warned] == [f"job #{job_id} embed: item 2 failed: item 2 is broken"]


def test_the_ingest_sweep_can_be_bounded_to_one_folder(db, tmp_path):
    """Re-reading is how this application corrects itself -- improving a
    parser is a re-parse -- and "re-read all eighty thousand files" is a
    price nobody pays to fix one folder of album tracks. A correction
    too expensive to apply is not a correction.

    The SUBTREE, not the one folder: somebody pointing at `music` means
    the albums inside it, and a scope that stopped at the top level
    would silently do a fraction of what was asked.
    """
    db.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'C:/x','library',0)")
    named = {}
    for at, (name, parent) in enumerate([("lib", None), ("music", 1), ("album", 2), ("pictures", 1)], start=1):
        db.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(?,?,'folder',?)", (at, bytes([at]) * 16, name))
        db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(?,1,?,?,0)", (at, parent, name))
        named[name] = at
    for at, folder in enumerate([named["music"], named["album"], named["pictures"]], start=10):
        db.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(?,?,'file',?)", (at, bytes([at]) * 16, f"f{at}"))
        db.execute(
            "INSERT INTO file(id,folder_id,name,kind,size,mtime,first_seen_at,last_seen_at)"
            " VALUES(?,?,?,'audio',1,0,0,0)",
            (at, folder, f"t{at}.m4a"),
        )
    db.commit()

    whole = runner.submit_ingest(db, 1.0, everything=True)
    assert db.execute("SELECT count(*) FROM job_item WHERE job_id = ?", (whole,)).fetchone()[0] == 3

    bounded = runner.submit_ingest(db, 2.0, everything=True, folder_id=named["music"])
    held = [row[0] for row in db.execute("SELECT item_id FROM job_item WHERE job_id = ? ORDER BY item_id", (bounded,))]
    assert held == [10, 11], "the subtree, and nothing outside it"


def test_a_bounded_sweep_with_nothing_to_do_says_so(db):
    """None, not an empty job: "already read" is an answer, and a job
    with no items would sit in the console claiming work."""
    db.execute("INSERT INTO root(id,path,kind,created_at) VALUES(1,'C:/x','library',0)")
    db.execute("INSERT INTO entity(id,uuid,kind,slug) VALUES(1,?,'folder','lib')", (b"\x01" * 16,))
    db.execute("INSERT INTO folder(id,root_id,parent_id,name,depth) VALUES(1,1,NULL,'lib',0)")
    db.commit()
    assert runner.submit_ingest(db, 1.0, everything=True, folder_id=1) is None
