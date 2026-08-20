"""Every table is written by something, and the writes honour their contracts.

A schema whose tables nothing fills is a design document. These exercise the
producers against the promises the DDL makes but cannot enforce: that an
address survives a rename, that dropping every derived table and re-indexing
leaves the library intact, that a job cannot report success with work
outstanding, and that a worker which lost its lease cannot still write.
"""

import io
import json
import pathlib
import sqlite3

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from db import authored, derived, ingest, jobs, library, lineage, naming, scan

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "db" / "schema.sql"
NOW = 1_700_000_000.0


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(io.open(SCHEMA, "r", encoding="utf-8", newline="").read())
    conn.execute("PRAGMA foreign_keys=ON")
    yield conn
    conn.close()


@pytest.fixture
def a_library(db, tmp_path):
    """A root, a folder, and one file to hang everything on."""
    root = tmp_path / "lib"
    root.mkdir()
    root_id = library.add_root(db, root, "library", NOW)
    folder_id = scan.ensure_folder(db, root_id, None, "lib")
    file_id = scan.mint(db, "file", "dusk")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, content_sha256,"
        " first_seen_at, last_seen_at) VALUES(?, ?, 'dusk.png', 'image', 10, 0, 'aa', ?, ?)",
        (file_id, folder_id, NOW, NOW),
    )
    user_id = authored.add_user(db, "will", "hash", "ADMIN", NOW)
    return {"root": root_id, "path": root, "folder": folder_id, "file": file_id, "user": user_id}


# --- addressing ------------------------------------------------------------


def test_a_rename_keeps_the_old_address_working(db, a_library):
    person = authored.person(db, None, NOW)
    first = naming.entity_slug(db, person)[1]

    authored.name_person(db, person, "Ilse", NOW)
    current = naming.entity_slug(db, person)[1]
    assert current == "ilse"

    assert naming.resolve(db, "person", "ilse") == (person, True)
    assert naming.resolve(db, "person", first) == (person, False), (
        "the address someone wrote down last year must still resolve"
    )


def test_a_slug_can_be_retired_more_than_once(db, a_library):
    """The history key includes the retirement time for exactly this."""
    person = authored.person(db, "Ilse", NOW)
    authored.name_person(db, person, "Rook", NOW + 1)
    authored.name_person(db, person, "Ilse", NOW + 2)
    authored.name_person(db, person, "Marguerite", NOW + 3)
    retired = db.execute(
        "SELECT slug, retired_at FROM slug_history WHERE entity_id = ? ORDER BY retired_at",
        (person,),
    ).fetchall()
    assert [r[0] for r in retired] == ["ilse", "rook", "ilse"]


def test_a_live_slug_beats_a_retired_one(db, a_library):
    """Otherwise renaming A frees a slug that B takes, and the old link to A
    starts answering with B."""
    first = authored.person(db, "Ilse", NOW)
    authored.name_person(db, first, "Rook", NOW + 1)
    second = authored.person(db, "Ilse", NOW + 2)

    assert naming.resolve(db, "person", "ilse") == (second, True)


def test_renaming_to_a_taken_name_does_not_steal_the_address(db, a_library):
    first = authored.person(db, "Ilse", NOW)
    second = authored.person(db, "Rook", NOW)
    authored.name_person(db, second, "Ilse", NOW + 1)

    assert naming.entity_slug(db, first)[1] == "ilse"
    assert naming.entity_slug(db, second)[1] == "ilse-2"


def test_a_rename_that_changes_nothing_writes_no_history(db, a_library):
    """A slug in history that is also live makes resolution depend on which
    table is consulted first."""
    person = authored.person(db, "Ilse", NOW)
    authored.name_person(db, person, "Ilse", NOW + 1)
    assert db.execute("SELECT count(*) FROM slug_history").fetchone()[0] == 0


# --- the rebuild contract --------------------------------------------------


def test_dropping_every_derived_table_leaves_the_library_standing(db, a_library):
    """The whole reason the derived namespace is segregated by name.

    Name a person, let a model infer them into files, drop the entire
    derived namespace, re-index -- and both the name and the attribution
    come back, the second from what a person asserted rather than from a
    similarity heuristic.
    """
    file_id, user_id = a_library["file"], a_library["user"]
    person = authored.person(db, "Ilse", NOW)
    authored.rate(db, file_id, user_id, 5, NOW)
    authored.assert_person(db, person, file_id, user_id, NOW)

    cluster = derived.add_cluster(db, "insightface", "v1", NOW, person_id=person)
    box = derived.region(db, 0.3, 0.2, 0.2, 0.3)
    derived.add_face(db, file_id, box, "insightface", "v1", "aa", NOW, cluster_id=cluster)
    derived.attribute(db, file_id, person, "insightface", "v1")
    derived.annotate(db, file_id, "caption", "a brass diving helmet", "qwen-vl", "2.5", "aa", NOW)
    verdict = authored.feedback(
        db, "person", "right", NOW, file_id=file_id, person_id=person, user_id=user_id
    )

    dropped = derived.drop_all(db)
    # named, not counted: a count passes just as well when a table is missed
    assert set(dropped) == {
        "derived_annotation", "derived_embedding", "derived_face_cluster",
        "derived_face_instance", "derived_file_hash", "derived_file_person",
        "derived_media_sample",
    }, dropped
    assert db.execute("SELECT count(*) FROM annotation_fts").fetchone()[0] == 0, (
        "the caption index outlived the captions"
    )

    # the authored side is untouched
    assert db.execute("SELECT name FROM person WHERE id = ?", (person,)).fetchone()[0] == "Ilse"
    assert db.execute("SELECT rating FROM rating WHERE file_id = ?", (file_id,)).fetchone()[0] == 5
    assert db.execute(
        "SELECT count(*) FROM person_assertion WHERE person_id = ?", (person,)
    ).fetchone()[0] == 1
    assert db.execute("SELECT verdict FROM feedback WHERE id = ?", (verdict,)).fetchone()[0] == "right"

    # re-index with a newer model, and the naming re-attaches from the record
    rebuilt = derived.add_cluster(db, "insightface", "v2", NOW + 10)
    box_again = derived.region(db, 0.3, 0.2, 0.2, 0.3)
    derived.add_face(
        db, file_id, box_again, "insightface", "v2", "aa", NOW + 10, cluster_id=rebuilt
    )
    named = derived.seed_clusters_from_assertions(db, "insightface", "v2")

    assert named == 1
    assert db.execute(
        "SELECT person_id FROM derived_face_cluster WHERE id = ?", (rebuilt,)
    ).fetchone()[0] == person
    assert db.execute(
        "SELECT count(*) FROM derived_file_person WHERE person_id = ?", (person,)
    ).fetchone()[0] == 1


def test_a_region_is_a_fraction_of_the_frame_not_a_pixel_count(db, a_library):
    """A box in pixels is a box against one rendering: the same numbers on a
    thumbnail or a re-encoded proxy point somewhere else."""
    box = derived.region_from_pixels(db, (256, 128, 512, 384), 1024, 768)
    stored = db.execute("SELECT x, y, w, h FROM region WHERE id = ?", (box,)).fetchone()
    assert stored == (0.25, pytest.approx(1 / 6), 0.5, 0.5)

    with pytest.raises(sqlite3.IntegrityError):
        derived.region(db, 0.9, 0.1, 0.5, 0.1)  # runs off the right edge
    with pytest.raises(sqlite3.IntegrityError):
        derived.region(db, 0.1, 0.1, 0.0, 0.1)  # zero width locates nothing


def test_a_mask_is_bytes_not_a_path(db, a_library):
    """A path is identity derived from location, which is the defect this
    schema exists to delete. Moving a cache directory must not void a mask."""
    box = derived.region(db, 0.1, 0.1, 0.4, 0.4, mask=b"\x89PNG\r\n\x1a\n-mask-bytes")
    row = db.execute(
        "SELECT b.payload_bin, b.byte_len FROM region r JOIN blob b ON b.hash = r.mask_hash"
        " WHERE r.id = ?",
        (box,),
    ).fetchone()
    assert row[0] == b"\x89PNG\r\n\x1a\n-mask-bytes"
    assert row[1] == len(b"\x89PNG\r\n\x1a\n-mask-bytes")


def test_a_caption_is_found_by_its_words(db, a_library):
    """A description nobody can search for is the same as not having one."""
    file_id = a_library["file"]
    derived.annotate(
        db, file_id, "caption", "a brass diving helmet at dusk", "qwen-vl", "2.5", "aa", NOW
    )
    derived.annotate(
        db, file_id, "description",
        "A weathered brass helmet rests on a jetty as the light fails.",
        "qwen-vl", "2.5", "aa", NOW,
    )
    hits = derived.search_annotations(db, "brass")
    assert {hit["kind"] for hit in hits} == {"caption", "description"}
    assert derived.search_annotations(db, "helicopter") == []


def test_two_models_may_describe_one_picture(db, a_library):
    """They are compared, not merged -- which is the point of running both."""
    file_id = a_library["file"]
    derived.annotate(db, file_id, "caption", "a diving helmet", "qwen-vl", "2.5", "aa", NOW)
    derived.annotate(db, file_id, "caption", "an old brass hat", "florence", "2", "aa", NOW)
    captions = derived.said_about(db, file_id, kind="caption")
    assert {c["model_id"] for c in captions} == {"qwen-vl", "florence"}


def test_rerunning_one_model_replaces_its_own_answer(db, a_library):
    """Otherwise a re-parse accumulates versions of the same claim."""
    file_id = a_library["file"]
    derived.annotate(db, file_id, "caption", "first attempt", "qwen-vl", "2.5", "aa", NOW)
    derived.annotate(db, file_id, "caption", "better attempt", "qwen-vl", "2.5", "aa", NOW + 5)
    captions = derived.said_about(db, file_id, kind="caption")
    assert [c["text"] for c in captions] == ["better attempt"]
    assert db.execute("SELECT count(*) FROM annotation_fts").fetchone()[0] == 1, (
        "the search index kept the superseded text"
    )


def test_an_annotation_may_point_at_part_of_the_picture(db, a_library):
    """Text read out of an image sits somewhere; a tag may be about one
    object. NULL means it is about the whole frame."""
    file_id = a_library["file"]
    box = derived.region(db, 0.6, 0.1, 0.3, 0.1)
    derived.annotate(
        db, file_id, "ocr", "CLOSED", "paddle", "3", "aa", NOW, region_id=box
    )
    row = db.execute(
        "SELECT a.text, r.x, r.w FROM derived_annotation a JOIN region r ON r.id = a.region_id"
        " WHERE a.file_id = ?",
        (file_id,),
    ).fetchone()
    assert row == ("CLOSED", 0.6, 0.3)


def test_an_annotation_cannot_cite_another_files_frame(db, a_library):
    """Otherwise a caption quotes a moment from a different video and the
    evidence link still reads as sound."""
    other = scan.mint(db, "file", "clip")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'clip.mp4', 'video', 10, 0, ?, ?)",
        (other, a_library["folder"], NOW, NOW),
    )
    frame = derived.add_sample(db, other, "frame", "every-2s", offset_ms=4000)
    with pytest.raises(sqlite3.IntegrityError):
        derived.annotate(
            db, a_library["file"], "caption", "wrong film", "qwen-vl", "2.5", "aa", NOW,
            sample_id=frame,
        )


def test_a_verdict_on_a_caption_survives_the_caption(db, a_library):
    """The annotation is derived and the next rebuild deletes it; "the
    caption for this file was wrong" has to still mean something after."""
    file_id, user_id = a_library["file"], a_library["user"]
    derived.annotate(db, file_id, "caption", "a diving helmet", "qwen-vl", "2.5", "aa", NOW)
    verdict = authored.feedback(
        db, "annotation", "wrong", NOW, file_id=file_id,
        annotation_kind="caption", user_id=user_id,
    )
    derived.drop_all(db)
    assert db.execute(
        "SELECT target_kind, annotation_kind, verdict FROM feedback WHERE id = ?", (verdict,)
    ).fetchone() == ("annotation", "caption", "wrong")


def test_feedback_must_say_which_description_it_judged(db, a_library):
    """"The model was wrong about this file" is not actionable when the model
    said four different things about it."""
    with pytest.raises(sqlite3.IntegrityError):
        authored.feedback(db, "annotation", "wrong", NOW, file_id=a_library["file"])


def test_feedback_outlives_the_thing_it_judged(db, a_library):
    """Its pointers are SET NULL, not CASCADE: a verdict is authored, and
    dropping derived state must not delete it."""
    file_id, user_id = a_library["file"], a_library["user"]
    person = authored.person(db, "Ilse", NOW)
    verdict = authored.feedback(
        db, "person", "wrong", NOW, file_id=file_id, person_id=person, user_id=user_id
    )
    db.execute("DELETE FROM person WHERE id = ?", (person,))
    row = db.execute(
        "SELECT verdict, person_id FROM feedback WHERE id = ?", (verdict,)
    ).fetchone()
    assert row == ("wrong", None)


def test_staleness_follows_the_bytes_not_the_clock(db, a_library):
    """A restore or a sync client rewrites mtime without changing a pixel."""
    file_id = a_library["file"]
    derived.record_hash(db, file_id, "aa", NOW)
    db.execute("UPDATE file SET mtime = ? WHERE id = ?", (NOW + 9999, file_id))
    assert derived.stale(db, "derived_file_hash") == []

    db.execute("UPDATE file SET content_sha256 = 'bb' WHERE id = ?", (file_id,))
    assert derived.stale(db, "derived_file_hash") == [file_id]


def test_an_unhashed_file_never_reads_as_current(db, a_library):
    """`<>` is NULL-blind, and NULL is the normal state before hashing."""
    file_id = a_library["file"]
    derived.record_hash(db, file_id, "v1", NOW)
    db.execute("UPDATE file SET content_sha256 = NULL WHERE id = ?", (file_id,))
    assert derived.stale(db, "derived_file_hash") == [file_id]


# --- jobs ------------------------------------------------------------------


def test_a_job_reports_its_own_progress_from_the_row(db, a_library):
    job = jobs.submit(db, "scan", NOW, items=[1, 2, 3])
    claimed = jobs.claim(db, "worker-a", NOW)
    assert claimed is not None
    job_id, fence = claimed
    assert job_id == job

    jobs.finish_item(db, job_id, fence, 1)
    state = jobs.finish_item(db, job_id, fence, 2)
    assert (state.done, state.total, state.state) == (2, 3, "running")
    assert state.fraction == pytest.approx(2 / 3)

    # a client arriving now renders from the row, not from messages it missed
    assert jobs.snapshot(db, job_id)["done_count"] == 2


def test_a_job_may_not_report_success_with_work_outstanding(db, a_library):
    job = jobs.submit(db, "scan", NOW, items=[1, 2])
    job_id, fence = jobs.claim(db, "worker-a", NOW)
    jobs.finish_item(db, job_id, fence, 1)

    with pytest.raises(ValueError, match="unfinished"):
        jobs.settle(db, job_id, fence, "done", NOW + 1)

    jobs.finish_item(db, job_id, fence, 2)
    jobs.settle(db, job_id, fence, "done", NOW + 2)
    assert jobs.progress(db, job_id).state == "done"


def test_cancelling_asks_and_the_runner_answers(db, a_library):
    """Flipping the state from outside would mark work finished that is
    still running."""
    job = jobs.submit(db, "scan", NOW, items=[1, 2])
    job_id, fence = jobs.claim(db, "worker-a", NOW)
    jobs.cancel(db, job_id)

    assert jobs.cancelled(db, job_id)
    assert jobs.progress(db, job_id).state == "running", "a request is not a state"

    jobs.settle(db, job_id, fence, "cancelled", NOW + 1)
    assert jobs.progress(db, job_id).state == "cancelled"


def test_a_resumed_job_repeats_nothing(db, a_library):
    job = jobs.submit(db, "scan", NOW, items=[1, 2, 3, 4])
    job_id, fence = jobs.claim(db, "worker-a", NOW)
    jobs.finish_item(db, job_id, fence, 1)
    jobs.finish_item(db, job_id, fence, 2)

    # the worker dies; its lease expires and another takes over
    later = NOW + jobs.LEASE_SECONDS + 1
    resumed = jobs.claim(db, "worker-b", later)
    assert resumed is not None
    assert resumed[0] == job_id
    assert jobs.pending(db, job_id) == [3, 4]


def test_an_evicted_worker_cannot_still_write(db, a_library):
    """A lease nobody can prove is not a lease: the reclaiming worker must
    fence the one it replaced."""
    jobs.submit(db, "scan", NOW, items=[1, 2])
    job_id, first_fence = jobs.claim(db, "worker-a", NOW)

    later = NOW + jobs.LEASE_SECONDS + 1
    job_again, second_fence = jobs.claim(db, "worker-b", later)
    assert job_again == job_id and second_fence != first_fence

    with pytest.raises(jobs.LeaseLost):
        jobs.finish_item(db, job_id, first_fence, 1)
    with pytest.raises(jobs.LeaseLost):
        jobs.settle(db, job_id, first_fence, "done", later)

    jobs.finish_item(db, job_id, second_fence, 1)
    assert jobs.progress(db, job_id).done == 1


def test_a_live_lease_is_not_stolen(db, a_library):
    """The control: without it the test above would pass on a claim that
    always succeeds."""
    jobs.submit(db, "scan", NOW, items=[1])
    jobs.claim(db, "worker-a", NOW)
    assert jobs.claim(db, "worker-b", NOW + 1) is None


def test_a_heartbeat_holds_the_lease(db, a_library):
    jobs.submit(db, "scan", NOW, items=[1])
    job_id, fence = jobs.claim(db, "worker-a", NOW)
    jobs.heartbeat(db, job_id, fence, NOW + jobs.LEASE_SECONDS - 1)
    assert jobs.claim(db, "worker-b", NOW + jobs.LEASE_SECONDS + 1) is None, (
        "a worker that is still reporting must keep its work"
    )


def test_work_without_units_resumes_from_a_checkpoint(db, a_library):
    """A scan cannot enumerate its units up front -- it is discovering them."""
    job = jobs.submit(db, "scan", NOW)
    job_id, fence = jobs.claim(db, "worker-a", NOW)
    jobs.checkpoint(db, job_id, fence, {"after": "portraits/2026"}, done=140)
    stored = db.execute("SELECT checkpoint, done_count FROM job WHERE id = ?", (job_id,)).fetchone()
    assert json.loads(stored[0]) == {"after": "portraits/2026"}
    assert stored[1] == 140


def test_watching_a_folder_starts_nothing(db, a_library):
    """A watch says which folders a scan should cover. It is not a thread."""
    jobs.watch_folder(db, a_library["folder"], NOW)
    assert [row[0] for row in jobs.watched(db)] == [a_library["folder"]]
    assert jobs.active(db) == [], "recording a watch must not queue work"


# --- lineage ---------------------------------------------------------------


def test_a_derivation_is_recorded_when_it_is_asked_for(db, a_library):
    """The edge is knowable at submit and unrecoverable afterwards: the
    output arrives looking like any other new file."""
    parent = a_library["file"]
    child = scan.mint(db, "file", "child")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'child.png', 'image', 10, 0, ?, ?)",
        (child, a_library["folder"], NOW, NOW),
    )
    lineage.intend(db, parent, "remix", "comfy-job-9f2", NOW)
    assert len(lineage.open_intents(db)) == 1

    edge = lineage.resolve(db, "comfy-job-9f2", child, NOW + 30)
    assert edge is not None
    assert lineage.open_intents(db) == []
    assert db.execute(
        "SELECT parent_id, child_id, kind FROM file_derivation WHERE id = ?", (edge,)
    ).fetchone() == (parent, child, "remix")


def test_submitting_twice_does_not_make_two_intents(db, a_library):
    first = lineage.intend(db, a_library["file"], "remix", "comfy-job-9f2", NOW)
    second = lineage.intend(db, a_library["file"], "remix", "comfy-job-9f2", NOW + 1)
    assert first == second


def test_a_file_cannot_derive_from_itself(db, a_library):
    """Every lineage walk from a self-edge is a cycle."""
    file_id = a_library["file"]
    lineage.intend(db, file_id, "upscale", "comfy-job-self", NOW)
    assert lineage.resolve(db, "comfy-job-self", file_id, NOW) is None
    assert lineage.link(db, file_id, file_id, "remix", NOW) is None
    assert db.execute("SELECT count(*) FROM file_derivation").fetchone()[0] == 0


def test_a_companion_is_found_from_either_side(db, a_library):
    raw_file = scan.mint(db, "file", "raw")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'dusk.dng', 'image', 10, 0, ?, ?)",
        (raw_file, a_library["folder"], NOW, NOW),
    )
    lineage.relate(db, a_library["file"], raw_file, "raw_pair", NOW)
    for left, right in ((a_library["file"], raw_file), (raw_file, a_library["file"])):
        assert db.execute(
            "SELECT count(*) FROM file_relation WHERE file_id = ? AND related_id = ?",
            (left, right),
        ).fetchone()[0] == 1


# --- ingest ----------------------------------------------------------------


A1111 = (
    "a brass diving helmet at dusk <lora:filmGrain:0.35>\n"
    "Negative prompt: blurry\n"
    "Steps: 28, Sampler: Euler a, CFG scale: 7, Seed: 4242, Size: 832x1216, "
    "Model: dreamshaper_8, Version: v1.10.1"
)


@pytest.fixture
def a_generated_file(db, a_library, tmp_path):
    info = PngInfo()
    info.add_text("parameters", A1111)
    path = tmp_path / "gen.png"
    Image.new("RGB", (16, 16), (30, 40, 60)).save(path, pnginfo=info)
    return path


def test_a_generated_file_becomes_rows(db, a_library, a_generated_file):
    out = ingest.one(db, a_library["file"], a_generated_file, NOW)
    assert out.tool == "A1111 / Forge"

    row = db.execute(
        "SELECT g.seed, g.steps, g.cfg, g.sampler, g.width, g.height, p.text, n.text"
        "  FROM generation g LEFT JOIN prompt p ON p.id = g.prompt_id"
        "  LEFT JOIN prompt n ON n.id = g.negative_id WHERE g.file_id = ?",
        (a_library["file"],),
    ).fetchone()
    assert row[:6] == (4242, 28, 7.0, "Euler a", 832, 1216)
    assert row[6].startswith("a brass diving helmet")
    assert row[7] == "blurry"

    weights = db.execute(
        "SELECT a.kind, a.name FROM file_artifact fa JOIN artifact a ON a.id = fa.artifact_id"
        " WHERE fa.file_id = ? ORDER BY a.kind",
        (a_library["file"],),
    ).fetchall()
    assert weights == [("checkpoint", "dreamshaper_8"), ("lora", "filmGrain")]


def test_no_field_is_written_as_a_document(db, a_library, a_generated_file):
    """A structure stored as JSON is a field nothing can search."""
    ingest.one(db, a_library["file"], a_generated_file, NOW)
    values = [
        value
        for (value,) in db.execute(
            "SELECT value_text FROM file_param WHERE file_id = ? AND value_text IS NOT NULL",
            (a_library["file"],),
        )
    ]
    assert values, "a file full of metadata produced no fields"
    for value in values:
        stripped = value.strip()
        assert not (stripped.startswith("{") and stripped.endswith("}")), value
        assert not (stripped.startswith("[") and stripped.endswith("]")), value


def test_a_nested_value_becomes_one_field_per_leaf(db, a_library):
    """Flattened under dotted keys, so each leaf is its own facet."""
    ingest._param(
        db, a_library["file"], "sidecar", "capture",
        {"lens": {"model": "XF35mmF1.4", "serial": "12ab"}, "tags": ["dusk", "brass"]},
    )
    stored = dict(
        db.execute(
            "SELECT key, value_text FROM file_param WHERE file_id = ? AND source = 'sidecar'",
            (a_library["file"],),
        )
    )
    assert stored == {
        "capture.lens.model": "XF35mmF1.4",
        "capture.lens.serial": "12ab",
        "capture.tags.0": "dusk",
        "capture.tags.1": "brass",
    }


def test_the_carrier_is_kept_and_says_whether_it_was_understood(db, a_library, a_generated_file):
    ingest.one(db, a_library["file"], a_generated_file, NOW)
    rows = db.execute(
        "SELECT fb.slot, fb.parsed_by, b.byte_len FROM file_blob fb"
        " JOIN blob b ON b.hash = fb.blob_hash WHERE fb.file_id = ?",
        (a_library["file"],),
    ).fetchall()
    assert rows, "nothing kept the payload it parsed"
    assert all(length > 0 for _, _, length in rows)


def test_the_registry_learns_what_the_file_contained(db, a_library, a_generated_file):
    ingest.one(db, a_library["file"], a_generated_file, NOW)
    learned = dict(db.execute("SELECT key, occurrences FROM param_key"))
    counted = dict(db.execute("SELECT key, count(*) FROM file_param GROUP BY key"))
    assert learned == counted
    assert "Format" in learned, "container facts are metadata too"


def test_two_files_naming_one_model_share_its_row(db, a_library, a_generated_file, tmp_path):
    """Otherwise a model page counts spellings instead of pictures."""
    second = scan.mint(db, "file", "gen2")
    db.execute(
        "INSERT INTO file(id, folder_id, name, kind, size, mtime, first_seen_at, last_seen_at)"
        " VALUES(?, ?, 'gen2.png', 'image', 10, 0, ?, ?)",
        (second, a_library["folder"], NOW, NOW),
    )
    ingest.one(db, a_library["file"], a_generated_file, NOW)
    ingest.one(db, second, a_generated_file, NOW)

    assert db.execute(
        "SELECT count(*) FROM artifact WHERE kind = 'checkpoint'"
    ).fetchone()[0] == 1
    assert db.execute(
        "SELECT count(*) FROM file_artifact WHERE role = 'checkpoint'"
    ).fetchone()[0] == 2
    assert db.execute("SELECT count(*) FROM prompt").fetchone()[0] == 2


# --- roots -----------------------------------------------------------------


def test_an_unreachable_root_is_marked_offline_not_emptied(db, a_library, tmp_path):
    """Unplugged and emptied look identical from a listing, and only one of
    them is recoverable."""
    missing = library.add_root(db, tmp_path / "not-here", "mount", NOW)
    checked = dict((row[0], row[2]) for row in library.check_roots(db))
    assert checked[a_library["root"]] is True
    assert checked[missing] is False
    assert db.execute(
        "SELECT count(*) FROM file WHERE missing_since IS NOT NULL"
    ).fetchone()[0] == 0, "checking a root must never touch a file"


def test_a_setting_keeps_its_type(db):
    library.put(db, "thumbnail_size", 512)
    library.put(db, "watch", True)
    library.put(db, "roots", ["a", "b"])
    assert library.get(db, "thumbnail_size") == 512
    assert library.get(db, "watch") is True
    assert library.get(db, "roots") == ["a", "b"]
    assert library.get(db, "absent", "fallback") == "fallback"
