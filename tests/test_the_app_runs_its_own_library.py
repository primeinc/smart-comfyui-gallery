"""The application over a library it builds itself, one per test.

Each test here needs a library of its own shape -- duplicates, failures,
redirected homes -- so each builds one; the shared world lives in
test_the_app_serves_the_schema.py.
"""

from __future__ import annotations

import pytest
from litestar.testing import TestClient

from db import connect, library, naming, scan
from sg_web.app import build_app
from tests.staging import settled


def scene(sky, ground, feature):
    """Deterministic picture-like content: a gradient sky over a ground
    band with one shape and one glow. Structured like a photograph, so
    perceptual hashes survive lossy re-encodes -- pure noise does not:
    webp q75 of an `effect_noise` field lands up to 10 pHash bits from
    its own original (measured, 30 trials), which made the dupe tests
    a coin flip. Deterministic, so the distances below never drift."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (96, 96))
    draw = ImageDraw.Draw(img)
    for y in range(96):
        t = y / 95
        draw.line([(0, y), (95, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(sky, ground, strict=True)))
    draw.rectangle(feature["box"], fill=feature["fill"])
    draw.ellipse(feature["sun"], fill=feature["glow"])
    return img


CASTLE = (
    (40, 60, 120),
    (180, 170, 150),
    {"box": (30, 50, 66, 90), "fill": (90, 80, 70), "sun": (10, 8, 26, 24), "glow": (250, 240, 200)},
)


MEADOW = (
    (150, 200, 240),
    (60, 140, 60),
    {"box": (5, 70, 90, 95), "fill": (50, 120, 50), "sun": (66, 6, 88, 28), "glow": (255, 255, 230)},
)


#: Waiting happens on the ROW (tests/staging.settled), never on delta
#: frames: the worker legitimately pauses a job mid-drain under load
#: ("paused after 1 items; the next turn resumes it"), so a fixed
#: inter-frame timeout is a coin flip on a saturated runner. The one
#: test about being SPOKEN to on the feed keeps its own socket below.


def test_the_recipe_axis_is_produced_and_served(tmp_path):
    """Ingestion is a job the application offers, and what it reads
    becomes addressable: the model has a page counting pictures, the LoRA
    knows what it is used with, the picture page carries its whole
    recipe, and a model reached on the wrong shelf 301s to its own."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    root = tmp_path / "lib"
    root.mkdir()
    for name, seed in (("helm_1.png", 4242), ("helm_2.png", 77)):
        info = PngInfo()
        info.add_text("workflow", '{"9": {"class_type": "SaveImage", "inputs": {}}}')
        info.add_text(
            "parameters",
            f"a brass diving helmet at dusk <lora:filmGrain:0.35>\nNegative prompt: blurry\n"
            f"Steps: 28, Sampler: Euler a, CFG scale: 7, Seed: {seed}, Size: 16x16, "
            f"Model: dreamshaper_8, Version: v1.10.1",
        )
        Image.new("RGB", (16, 16), (30, 40, 60)).save(root / name, pnginfo=info)

    with TestClient(app=build_app(str(tmp_path / "run"))) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        assert client.post(f"/roots/{made['id']}/scan").json()["added"] == 2
        job = client.post("/jobs/ingest").json()
        assert (job["kind"], job["total"]) == ("scan", 2)
        assert settled(client, job["id"]) == "done"
        told = client.get(f"/jobs/{job['id']}").json()
        assert (told["state"], told["failed_count"]) == ("done", 0)

        # Artifact slugs carry their kind: every shelf shares one entity
        # kind, and a checkpoint and a LoRA may share a name.
        models = client.get("/models").json()
        assert models == [{"name": "dreamshaper_8", "slug": "checkpoint-dreamshaper-8", "pictures": 2}]
        shelf = client.get("/m/checkpoint-dreamshaper-8").json()
        assert (shelf["name"], shelf["kind"]) == ("dreamshaper_8", "checkpoint")
        # The media answer IS the ResultSet's: same items, same count as
        # the shelf's aggregate -- no second membership arithmetic.
        assert sorted(p["name"] for p in shelf["gallery"]["items"]) == ["helm_1.png", "helm_2.png"]
        assert shelf["count"] == models[0]["pictures"] == 2

        loras = client.get("/loras").json()
        assert loras == [{"name": "filmGrain", "slug": "lora-filmgrain", "pictures": 2}]
        lora_page = client.get("/l/lora-filmgrain").json()
        assert lora_page["used_with"] == [{"name": "dreamshaper_8", "slug": "checkpoint-dreamshaper-8", "together": 2}]

        moved = client.get("/m/lora-filmgrain", follow_redirects=False)
        assert moved.status_code == 301
        assert moved.headers["location"] == "/l/lora-filmgrain"
        assert client.get("/m/nobody").status_code == 404

        # The workflow shelf, with data: both files carry the same graph
        # chunk, so one workflow artifact holds both pictures.
        shelved = client.get("/workflows").json()
        assert [row["pictures"] for row in shelved] == [2]
        graph_page = client.get(f"/w/{shelved[0]['slug']}").json()
        assert graph_page["kind"] == "workflow"
        assert sorted(p["name"] for p in graph_page["gallery"]["items"]) == ["helm_1.png", "helm_2.png"]

        # A kind with rows but no shelf yet says so, on every shelf.
        conn = connect.connect(client.app.state.db_path)
        from db import ingest as ingest_module

        camera_id = ingest_module.artifact(conn, "camera", "Canon EOS R5", 9.0)
        conn.commit()
        camera_slug = naming.entity_slug(conn, camera_id)
        conn.close()
        assert camera_slug is not None
        assert client.get(f"/m/{camera_slug[1]}").status_code == 404

        pic = client.get("/i/helm-1").json()
        assert (pic["name"], pic["creation"]["checkpoint"], pic["creation"]["seed"]) == (
            "helm_1.png",
            "dreamshaper_8",
            4242,
        )
        assert pic["creation"]["prompt"].startswith("a brass diving helmet")
        # the weight, not just the name: the recipe panel copies
        # `<lora:filmGrain:0.35>` and a name alone does not reproduce it
        assert pic["creation"]["loras"] == [{"name": "filmGrain", "weight": 0.35}]
        assert pic["params"], "the parsed fields are rows, not a blob"
        assert {pic["context"]["previous"], pic["context"]["next"]} == {None, "helm-2"}, (
            "neighbours walk the answer: the default ResultSet, newest first"
        )
        assert pic["lineage"]["parents"] == [], "no lineage yet, said honestly"
        assert pic["lineage"]["children"] == []

        folder = client.get("/f/lib").json()
        assert sorted(f["name"] for f in folder["files"]) == ["helm_1.png", "helm_2.png"]
        assert folder["breadcrumb"][-1]["name"] == "lib"
        assert client.get("/f/nowhere").status_code == 404


def test_ingest_failures_land_on_items_not_on_the_library(tmp_path):
    """An unreadable file is a FAILED item, and the recipe it already
    contributed survives: ingest retracts before it re-reads, so a rotten
    re-read must roll back rather than commit the destruction and report
    success. One root offline must never silently erase its recipe axis."""
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    root = tmp_path / "lib"
    root.mkdir()
    info = PngInfo()
    info.add_text(
        "parameters",
        "a helmet\nSteps: 28, Sampler: Euler a, CFG scale: 7, Seed: 1, Size: 16x16, "
        "Model: dreamshaper_8, Version: v1.10.1",
    )
    Image.new("RGB", (16, 16), (1, 2, 3)).save(root / "good.png", pnginfo=info)
    (root / "gone.png").write_bytes(b"\x89PNG-pretend")

    with TestClient(app=build_app(str(tmp_path / "run"))) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        assert client.post(f"/roots/{made['id']}/scan").json()["added"] == 2

        def drained_ingest(*, everything: bool = False) -> dict:
            job_id = client.post("/jobs/ingest", params={"everything": str(everything).lower()}).json()["id"]
            settled(client, job_id)
            return client.get(f"/jobs/{job_id}").json()

        first = drained_ingest()
        assert first["failed_count"] == 1, "junk bytes must be a failed item, never quiet success"
        before = client.get("/i/good").json()
        assert before["params"], "the good file ingested"
        assert before["creation"]["checkpoint"] == "dreamshaper_8"

        # The library rots behind the app's back: one corrupted, one deleted.
        (root / "good.png").write_bytes(b"\x89PNG-now-junk")
        (root / "gone.png").unlink()
        # by the record only the junk file is unread: its read failed, and a
        # failed read is not a read; no scan has seen the other one rot
        assert drained_ingest()["total"] == 1
        second = drained_ingest(everything=True)  # the rot detector: read all of it again
        assert second["failed_count"] == 2, "both rotten files must land on their items"
        after = client.get("/i/good").json()
        assert after["params"] == before["params"], "a failed re-read destroyed the recipe it could not replace"
        assert after["creation"]["checkpoint"] == before["creation"]["checkpoint"]


def test_perceptual_hashing_is_a_job_the_application_offers(tmp_path):
    """The backfill for libraries that never ran detection: POST
    /jobs/phash computes phash64/dhash64 for every present picture, and
    a resized re-encode lands within a few bits of its original --
    perceptual identity produced by the application, over HTTP."""
    from vision import dupes

    root = tmp_path / "lib"
    root.mkdir()
    picture = scene(*CASTLE)
    picture.save(root / "castle.png")
    picture.resize((32, 32)).save(root / "castle_half.jpg", quality=80)

    with TestClient(app=build_app(str(tmp_path / "run"))) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        assert client.post(f"/roots/{made['id']}/scan").json()["added"] == 2
        job = client.post("/jobs/phash").json()
        assert (job["kind"], job["total"]) == ("hash", 2)
        assert settled(client, job["id"]) == "done"
        told = client.get(f"/jobs/{job['id']}").json()
        assert (told["state"], told["failed_count"]) == ("done", 0)

        conn = connect.connect(client.app.state.db_path)
        hashes: dict = {}
        for name, producer, value in conn.execute(
            "SELECT f.name, s.producer, h.value FROM derived_file_hash h"
            " JOIN file f ON f.id = h.file_id JOIN similarity_space s ON s.id = h.space_id"
        ):
            hashes.setdefault(name, {})[producer] = value
        conn.close()
        assert set(hashes) == {"castle.png", "castle_half.jpg"}
        assert all(set(told) == {"imagehash.phash", "imagehash.dhash"} for told in hashes.values())
        apart = dupes.hamming(hashes["castle.png"]["imagehash.phash"], hashes["castle_half.jpg"]["imagehash.phash"])
        assert apart <= 6, f"the same picture resized measured {apart} bits apart"


def test_a_scan_queues_the_thumbnails_of_what_it_found(tmp_path):
    """New pictures are thumbnailed by the worker the moment a walk
    finds them, so the rail's hover never pays a first decode of the
    original; a walk that finds nothing new, and a cache already full,
    queue nothing."""
    from vision import thumbs

    root = tmp_path / "lib"
    root.mkdir()
    scene(*CASTLE).save(root / "castle.png")
    scene(*MEADOW).save(root / "meadow.png")

    with TestClient(app=build_app(str(tmp_path / "run"))) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        swept = client.post(f"/roots/{made['id']}/scan").json()
        assert swept["added"] == 2
        job = client.get(f"/jobs/{swept['precache']}").json()
        assert (job["kind"], job["total"]) == ("hash", 2)
        assert settled(client, job["id"]) == "done"
        cache = tmp_path / "run" / "thumbs"
        conn = connect.connect(client.app.state.db_path)
        shas = [row[0] for row in conn.execute("SELECT content_sha256 FROM file")]
        conn.close()
        assert len(shas) == 2
        assert all(thumbs.path_for(cache, sha, kind).exists() for sha in shas for kind in thumbs.EDGES)
        assert client.post(f"/roots/{made['id']}/scan").json()["precache"] is None
        assert client.post("/jobs/thumbs").status_code == 204


def test_copies_of_copies_collapse_into_pictures(tmp_path):
    """The dedupe story end to end, over HTTP: hash the pixels, group
    them through the FAISS binary index, and the copies collapse -- one
    group per picture, the largest body picked as its best face, every
    copy listed from the picture page. A distinct picture stays alone."""
    root = tmp_path / "lib"
    root.mkdir()
    picture = scene(*CASTLE)
    picture.save(root / "castle.png")
    picture.resize((48, 48)).save(root / "castle_small.jpg", quality=80)
    picture.save(root / "castle_web.webp", quality=75)
    scene(*MEADOW).save(root / "meadow.png")

    with TestClient(app=build_app(str(tmp_path / "run"))) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        assert client.post(f"/roots/{made['id']}/scan").json()["added"] == 4

        def drained(route: str) -> None:
            job_id = client.post(route).json()["id"]
            settled(client, job_id)
            told = client.get(f"/jobs/{job_id}").json()
            assert (told["state"], told["failed_count"]) == ("done", 0), route

        drained("/jobs/phash")
        drained("/jobs/dupes")

        groups = client.get("/dupes").json()
        assert groups == [{"slug": "castle", "name": "castle.png", "copies": 3}], (
            f"three bodies of one picture must be one group with the original as its face: {groups}"
        )

        page = client.get("/i/castle-small").json()
        assert sorted(copy["name"] for copy in page["lineage"]["copies"]) == ["castle.png", "castle_web.webp"]
        best = [copy for copy in page["lineage"]["copies"] if copy["is_best"]]
        assert [copy["name"] for copy in best] == ["castle.png"], "the largest body is the picture's face"
        assert client.get("/i/meadow").json()["lineage"]["copies"] == [], "a distinct picture stays alone"

        # the threshold is a live setting, refused when meaningless
        assert client.post("/settings/dupe_threshold", json={"value": "6"}).status_code < 300
        assert client.post("/settings/dupe_threshold", json={"value": "banana"}).status_code < 300, (
            "free-text setting; the SUBMIT validates"
        )
        assert client.post("/jobs/dupes").status_code == 400
        client.post("/settings/dupe_threshold", json={"value": "4"})


def test_the_dupe_representative_does_not_depend_on_job_order(tmp_path):
    """The representative policy reads pixel dimensions, but only ingest
    persists them -- and /jobs/dupes explicitly runs from /jobs/phash
    alone. The canonical member of a duplicate group must be the same
    picture whether the metadata pass has happened or not."""
    root = tmp_path / "lib"
    root.mkdir()
    picture = scene(*CASTLE)
    picture.save(root / "castle.png")
    picture.resize((48, 48)).save(root / "castle_small.jpg", quality=80)

    with TestClient(app=build_app(str(tmp_path / "run"))) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")

        def drained(route: str) -> None:
            job_id = client.post(route).json()["id"]
            settled(client, job_id)
            told = client.get(f"/jobs/{job_id}").json()
            assert (told["state"], told["failed_count"]) == ("done", 0), route

        drained("/jobs/phash")
        drained("/jobs/dupes")
        before = client.get("/dupes").json()
        assert [group["name"] for group in before] == ["castle.png"], (
            "before ingest the header on disk must answer for the missing dimensions"
        )

        drained("/jobs/ingest")
        drained("/jobs/dupes")
        after = client.get("/dupes").json()
        assert after == before, "the representative changed because a metadata job ran"


def test_resolution_beats_bytes_for_the_dupe_representative(tmp_path):
    """Bytes measure compression, not fidelity: a heavily compressed
    high-resolution body must outrank a low-resolution one padded fat --
    byte size is a tiebreak WITHIN a resolution, never a substitute."""
    from PIL.PngImagePlugin import PngInfo

    root = tmp_path / "lib"
    root.mkdir()
    picture = scene(*CASTLE)
    picture.save(root / "castle_full.jpg", quality=30)
    padding = PngInfo()
    padding.add_text("padding", "x" * 40000)
    picture.resize((48, 48)).save(root / "castle_small.png", pnginfo=padding)
    assert (root / "castle_small.png").stat().st_size > (root / "castle_full.jpg").stat().st_size, (
        "the fixture only proves the rule if the low-resolution file really is the byte-heavier one"
    )

    with TestClient(app=build_app(str(tmp_path / "run"))) as client:
        made = client.post("/roots", json={"path": str(root)}).json()
        client.post(f"/roots/{made['id']}/scan")

        def drained(route: str) -> None:
            settled(client, client.post(route).json()["id"])

        drained("/jobs/phash")
        drained("/jobs/dupes")
        groups = client.get("/dupes").json()
        assert [group["name"] for group in groups] == ["castle_full.jpg"], (
            f"a padded low-resolution file outranked the high-resolution picture: {groups}"
        )


def test_a_whole_run_is_contained_in_one_redirectable_directory(tmp_path):
    """One --home argument moves everything a run owns -- database, models,
    caches. Nothing lands in OS application-data folders, and a first run
    needs nothing but the command that starts it."""
    from sg_web import home

    burrow = tmp_path / "elsewhere"
    assert home.home(burrow) == burrow
    assert home.db_path(burrow) == burrow / "gallery.db"
    assert home.models_dir(burrow) == burrow / "models"
    assert home.thumbs_dir(burrow) == burrow / "thumbs"

    shared = tmp_path / "shared-weights"
    assert home.models_dir(burrow, str(shared)) == shared, "a shared model dir is an option"

    with TestClient(app=build_app(str(burrow))) as client:
        assert client.get("/health").text == "ok"
        assert client.get("/people").json() == []
    assert (burrow / "gallery.db").exists(), "the run did not live in its home"


def test_settings_are_rows_changed_while_the_application_runs(tmp_path):
    """Configuration is settings rows, not environment variables: listed,
    changed and validated over requests, effective without a restart."""
    with TestClient(app=build_app(str(tmp_path / "run"))) as client:
        listed = {row["key"]: row for row in client.get("/settings").json()}
        assert listed["faiss_gpu"]["value"] == "on"
        assert "off" in listed["faiss_gpu"]["choices"]

        changed = client.post("/settings/faiss_gpu", json={"value": "off"}).json()
        assert changed == {"key": "faiss_gpu", "value": "off"}
        listed = {row["key"]: row for row in client.get("/settings").json()}
        assert listed["faiss_gpu"]["value"] == "off"

        assert client.post("/settings/faiss_gpu", json={"value": "cuda-magic"}).status_code == 400
        assert client.post("/settings/not_a_setting", json={"value": "x"}).status_code == 400


def test_a_bodyless_or_pathless_root_request_is_a_400_not_a_500(tmp_path):
    """The body is typed, so a missing or mistyped `path` is refused by
    the signature model with a 400 -- never a KeyError escaping as 500
    from a write route."""
    with TestClient(app=build_app(str(tmp_path / "run"))) as client:
        assert client.post("/roots", json={}).status_code == 400
        assert client.post("/roots", json={"kind": "library"}).status_code == 400
        assert client.post("/roots", json={"path": 123}).status_code == 400
        assert client.post("/roots").status_code == 400


@pytest.mark.slow
def test_shutdown_stops_the_worker_before_the_channel_it_publishes_to(tmp_path, caplog):
    """Lifespan managers exit in reverse, so the worker must be REGISTERED
    after the channel: stopped and joined while the channel it publishes
    to is still alive. Ordered wrongly, a shutdown mid-drain logged
    "Plugin not yet initialized" from the bridge -- and the docstring's
    "no thread mid-write" was a lie. The job it was draining stays rows,
    picked up by the next run."""
    import logging

    from litestar.channels import ChannelsPlugin

    root = tmp_path / "lib"
    root.mkdir()
    for n in range(300):
        (root / f"frame_{n:03}.png").write_bytes(b"\x89PNG-" + f"{n:03}".encode() * 64)
    burrow = tmp_path / "run"
    burrow.mkdir()
    conn = connect.connect(burrow / "gallery.db")
    conn.executescript(connect.schema_sql())
    conn.execute("PRAGMA foreign_keys=ON")
    root_id = library.add_root(conn, str(root), "library", 0.0)
    scan.scan(conn, root_id, str(root), 0.0)
    conn.commit()
    conn.close()

    with caplog.at_level(logging.ERROR), TestClient(app=build_app(str(burrow))) as client:
        managers = client.app._lifespan_managers
        channel_at = next(i for i, m in enumerate(managers) if isinstance(m, ChannelsPlugin))
        worker_at = next(i for i, m in enumerate(managers) if getattr(m, "__name__", "") == "working")
        assert channel_at < worker_at, "the channel would tear down under a live worker"
        job_id = client.post("/jobs/verify").json()["id"]
        # leave immediately: the worker is mid-drain as the app exits

    said = " ".join(record.getMessage() for record in caplog.records)
    assert "Plugin not yet initialized" not in said, "a publish landed on a torn-down channel"
    assert "worker turn died" not in said

    # The next run picks the interrupted job up: its own feed says so.
    # The worker starts with the app and may drain fast rows before this
    # socket finishes its handshake, so a job already settled is also
    # "picked back up" -- only one still in the snapshot owes the feed
    # its remaining deltas.
    with TestClient(app=build_app(str(burrow))) as client:
        with client.websocket_connect("/ws/jobs") as feed:
            first = feed.receive_json(timeout=10)
            assert first["type"] == "snapshot"
            if any(row["id"] == job_id for row in first["jobs"]):
                delta = feed.receive_json(timeout=30)
                while delta["state"] not in ("done", "failed", "cancelled"):
                    delta = feed.receive_json(timeout=30)
                assert (delta["job"], delta["state"]) == (job_id, "done")
        told = client.get(f"/jobs/{job_id}").json()
        assert told["state"] == "done", "the interrupted job was not picked back up"


def test_media_roots_are_rows_managed_through_the_application(tmp_path):
    """Any number of media directories, anywhere, registered and scanned
    over requests -- the pictures never live inside the run's home."""
    box_one = tmp_path / "comfy-output"
    box_two = tmp_path / "phone-camera"
    for box in (box_one, box_two):
        box.mkdir()
        (box / "shot.png").write_bytes(b"\x89PNG-" + box.name.encode())

    with TestClient(app=build_app(str(tmp_path / "run"))) as client:
        first = client.post("/roots", json={"path": str(box_one)}).json()
        second = client.post("/roots", json={"path": str(box_two)}).json()
        assert first["id"] != second["id"]

        listed = client.get("/roots").json()
        assert [r["online"] for r in listed] == [True, True]

        swept = client.post(f"/roots/{first['id']}/scan").json()
        assert swept["added"] == 1
        swept = client.post(f"/roots/{second['id']}/scan").json()
        assert swept["added"] == 1
        assert client.post("/roots/999/scan").status_code == 404
