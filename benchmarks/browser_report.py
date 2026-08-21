"""The app over the real samples, photographed by a real browser.

This is the presentation layer for the serving claims: it boots
`python -m sg_web` over the sample datasets (the i2i renders -- the
app's own output -- and the KYC photo set the face benchmarks use),
runs face detection as a real job over HTTP while a browser WebSocket
watches it drain, asks the application to cluster the faces into people
-- another job on the same feed -- names them over HTTP, and then
photographs what the app actually serves: the thumbnail grid, the
people index with face avatars, one person across the library, a
full-size preview. The screenshots become one self-contained page,
benchmarks/results/browser/report.html, images inlined as data URIs.

Nothing here is synthetic, staged, or written behind the app's back:
every mutation is a request the application offers, and every image is
a Chromium screenshot of a page whose <img> tags point at the running
server. People are named "Person N" -- cluster nicknames, never
real-world identity claims, per the project's face doctrine.

Run through `just bench browser-report`.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import html
import json
import pathlib
import shutil
import socket
import sqlite3
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EVIDENCE = REPO / "benchmarks" / "results" / "browser"
DATASETS = "C:/ComfyUI/output/sample-datasets"
MODELS = "C:/ComfyUI/output/.AImodels"

#: Which claim each capture presents, in the order the page tells it.
SECTIONS = (
    (
        "The library, as served",
        "Real files, scanned and thumbnailed by the running app; every image an <img> against /thumb",
        ("library-grid", "render-preview"),
    ),
    (
        "The recipe, from the files",
        (
            "Whatever the files carry, as the shelf routes serve it -- read by the ingest job,"
            " the album made through the application; an empty shelf says so"
        ),
        ("recipe-shelves",),
    ),
    (
        "One picture, however many bodies",
        (
            "Perceptual identity over the real samples: the phash and dupes jobs on the live feed,"
            " groups served best-face-first -- and similar-but-distinct renders staying apart"
        ),
        ("copy-shelf",),
    ),
    (
        "People, from faces",
        (
            "Detection, clustering, naming and avatar crops, all through the application's own routes;"
            " names are cluster nicknames"
        ),
        ("people-index", "person-page"),
    ),
    (
        "The work, live",
        "The detection job that produced it all, as a browser WebSocket watched it drain",
        ("job-live", "job-feed"),
    ),
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _answers(web) -> bool:
    import httpx

    try:
        return web.get("/health").text == "ok"
    except httpx.TransportError:
        return False


def _wait_healthy(web, server) -> None:
    """Bounded readiness gate: a child process offers no push signal
    before its socket answers, so this is a retry with a deadline, not
    pacing. A child that already exited is reported as the crash it is,
    not as thirty quiet seconds of 'never answered'."""
    deadline = time.time() + 30
    while not _answers(web):
        if server.poll() is not None:
            raise RuntimeError(f"the server exited {server.returncode} before answering /health; see server.log")
        if time.time() > deadline:
            raise TimeoutError("the server never answered /health")
        time.sleep(0.2)


def _ensure_copies(datasets: pathlib.Path) -> None:
    """The copies-of-copies dataset, created once and kept.

    Real libraries hold the same picture as many files -- the original,
    a resize, a JPEG saved from a JPEG, a WebP export, a crop -- and the
    shipped samples do not, so the grouping claim had nothing true to
    show. Six originals from the existing sets each spawn resized,
    re-encoded and cropped bodies, the way files actually rot. The crop
    trims 1/12 off every edge: a whole-frame DCT hash may or may not
    survive it, and the group counts in the capture report which.
    Deterministic and idempotent: existing files are left exactly as
    they are, missing bodies are added.
    """
    target = datasets / "copies-of-copies"
    from PIL import Image

    if target.exists():
        _ensure_crops(target)
        return

    i2i = sorted((datasets / "i2i-test-output").glob("*.png"))[:3]
    kyc = sorted(
        p
        for p in (datasets / "caucasian-people-kyc-photo-dataset" / "files").rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )[:3]
    target.mkdir(parents=True)
    for number, source in enumerate(i2i + kyc):
        with Image.open(source) as opened:
            image = opened.convert("RGB")
            image.save(target / f"pic{number}_original.png")
            w, h = image.size
            image.resize((max(8, w // 2), max(8, h // 2))).save(target / f"pic{number}_half.png")
            image.resize((max(8, w // 4), max(8, h // 4))).save(target / f"pic{number}_quarter.jpg", quality=80)
            image.save(target / f"pic{number}_gen1.jpg", quality=90)
        with Image.open(target / f"pic{number}_gen1.jpg") as second:
            second.convert("RGB").save(target / f"pic{number}_gen2.jpg", quality=60)
        (target / f"pic{number}_gen1.jpg").unlink()
        image.save(target / f"pic{number}_web.webp", quality=75)
    _ensure_crops(target)
    print(f"made {target.name}: {len(list(target.iterdir()))} files from {len(i2i + kyc)} pictures", flush=True)


def _ensure_crops(target: pathlib.Path) -> None:
    """A cropped body beside every original that lacks one."""
    from PIL import Image

    made = 0
    for original in sorted(target.glob("*_original.png")):
        cropped = target / original.name.replace("_original.png", "_crop.png")
        if cropped.exists():
            continue
        with Image.open(original) as opened:
            image = opened.convert("RGB")
            w, h = image.size
            image.crop((w // 12, h // 12, w - w // 12, h - h // 12)).save(cropped)
        made += 1
    if made:
        print(f"added {made} cropped bodies to {target.name}", flush=True)


def _commit_stamp(where: pathlib.Path = REPO) -> str:
    """The tree the evidence came from -- honest about a dirty one.

    Evidence stamped with a clean parent commit that cannot produce it is
    provenance pointing at the wrong code; `-dirty` says exactly that, and
    `-unverified` says the cleanliness could not be checked at all.
    """
    git = shutil.which("git")
    if git is None:
        return "unknown"
    head = subprocess.run(
        [git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False, timeout=10, cwd=where
    )
    if head.returncode != 0:
        return "unknown"
    changed = subprocess.run(
        [git, "status", "--porcelain"], capture_output=True, text=True, check=False, timeout=10, cwd=where
    )
    if changed.returncode != 0:
        return head.stdout.strip() + "-unverified"
    return head.stdout.strip() + ("-dirty" if changed.stdout.strip() else "")


def _grid(base: str, slugs: list[str], kind: str = "thumb") -> str:
    cells = "".join(
        f'<img src="{base}/{kind}/{slug}" style="width:160px;height:160px;object-fit:cover;'
        f'border-radius:6px;background:#222" loading="eager">'
        for slug in slugs
    )
    return (
        '<body style="margin:0;background:#14171a">'
        f'<div id="grid" style="display:grid;grid-template-columns:repeat(6,160px);gap:6px;'
        f'padding:10px;width:fit-content">{cells}</div>'
    )


def _all_loaded(page, scope: str) -> list[str]:
    """Wait until every image SETTLED (loaded or errored), then name the
    ones that did not decode. An errored <img> has complete=true and
    naturalWidth=0 -- waiting on naturalWidth alone turns one failed
    request into a silent timeout instead of a named URL.

    Returns the broken URLs rather than raising: one broken tile must not
    cost the other five captures, and the report SHOWS the breakage --
    the screenshot carries the broken-image glyph and the facts carry the
    URL, which is more honest than a run that refuses to say."""
    page.wait_for_function(
        "(scope) => { const all = document.querySelectorAll(scope + ' img');"
        " return all.length > 0 && [...all].every(i => i.complete); }",
        arg=scope,
        timeout=120_000,
    )
    broken = page.evaluate(
        "(scope) => [...document.querySelectorAll(scope + ' img')].filter(i => i.naturalWidth === 0).map(i => i.src)",
        scope,
    )
    for url in broken:
        print(f"  ! broken image: {url}", flush=True)
    return broken


def _moved_once_released(src: pathlib.Path, dst: pathlib.Path, patience: float = 10.0) -> None:
    """Move a staged file, waiting out the server's grip on its own log.

    `.venv/Scripts/python.exe` is a launcher shim: terminate()+wait() reap
    the shim while the interpreter it spawned -- the one holding the
    inherited server.log handle -- lets go a beat later. Windows refuses
    the move until it does, so this retries briefly and then raises the
    real error."""
    deadline = time.time() + patience
    while not _gave_way(src, dst):
        if time.time() > deadline:
            shutil.move(str(src), str(dst))  # out of patience: the real error surfaces
            return
        time.sleep(0.2)


def _gave_way(src: pathlib.Path, dst: pathlib.Path) -> bool:
    try:
        shutil.move(str(src), str(dst))
    except PermissionError:
        return False
    return True


def _publish(staging: pathlib.Path) -> None:
    """Swap the staged run into EVIDENCE, replacing the previous one.

    Only called on success, and the previous evidence is retired aside
    rather than deleted before its replacement lands: a failure inside
    this swap leaves the old run recoverable in the named directory
    instead of half-gone."""
    import tempfile

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    retired = pathlib.Path(tempfile.mkdtemp(prefix="browser-report-retired-"))
    print(f"previous evidence retired to {retired} until the swap lands", flush=True)
    for stale in EVIDENCE.iterdir():
        shutil.move(str(stale), str(retired / stale.name))
    for made in sorted(staging.iterdir()):
        _moved_once_released(made, EVIDENCE / made.name)
    shutil.rmtree(retired)


def capture(datasets: str, models_dir: str) -> list[dict]:
    """Drive the whole thing and return the manifest entries.

    Everything lands in a fresh staging directory and is swapped into
    EVIDENCE only when every capture succeeded."""
    import tempfile

    import httpx
    from playwright.sync_api import sync_playwright

    staging = pathlib.Path(tempfile.mkdtemp())
    taken: list[dict] = []

    def keep(name: str, shot: bytes, caption: str, **facts) -> None:
        (staging / f"{name}.png").write_bytes(shot)
        taken.append({"name": name, "file": f"{name}.png", "caption": caption, "facts": facts})

    home = pathlib.Path(tempfile.mkdtemp()) / "run"
    port = _free_port()
    # The child's stdout is its access log, one line per request; a pipe
    # nobody drains blocks the server at the OS buffer mid-capture. The
    # log lands next to the screenshots as part of the run's evidence.
    with (staging / "server.log").open("wb") as server_log:
        server = subprocess.Popen(
            [sys.executable, "-m", "sg_web", "--home", str(home), "--port", str(port)],
            stdout=server_log,
            stderr=subprocess.STDOUT,
            cwd=REPO,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            with httpx.Client(base_url=base, timeout=10.0) as web, sync_playwright() as p:
                _wait_healthy(web, server)
                _ensure_copies(pathlib.Path(datasets))
                scanned = 0
                for sample in ("i2i-test-output", "caucasian-people-kyc-photo-dataset/files", "copies-of-copies"):
                    made = web.post("/roots", json={"path": str(pathlib.Path(datasets) / sample)}).json()
                    scanned += web.post(f"/roots/{made['id']}/scan").json()["added"]
                print(f"scanned {scanned} sample files")

                browser = p.chromium.launch()

                def note_failure(response) -> None:
                    if response.status >= 400:
                        print(f"  ! {response.status} {response.url}", flush=True)

                def watched_page():
                    """Every capture page reports the requests that went wrong --
                    a failed <img> is a named URL and status, never a timeout."""
                    fresh = browser.new_page()
                    fresh.on("response", note_failure)
                    return fresh

                page = watched_page()

                # The feed first, so the job is watched from its first delta.
                page.evaluate(
                    "(feed) => { window.__got = [];"
                    " window.__ws = new WebSocket(feed);"
                    " window.__ws.onmessage = (event) => window.__got.push(JSON.parse(event.data)); }",
                    f"ws://127.0.0.1:{port}/ws/jobs",
                )
                page.wait_for_function("() => window.__ws.readyState === 1")

                # The recipe first: the ingest job reads every file's own
                # metadata into entities, watched on the same feed.
                reading = web.post("/jobs/ingest").json()
                print(f"ingest job {reading['id']}: {reading['total']} files")
                page.wait_for_function(
                    "(id) => window.__got.some(m => m.job === id && ['done','failed','cancelled'].includes(m.state))",
                    arg=reading["id"],
                    timeout=600_000,
                )
                digested = web.get(f"/jobs/{reading['id']}").json()
                if digested["state"] != "done":
                    raise RuntimeError(f"the ingest job settled {digested['state']}: {digested['error']}")

                job = web.post("/jobs/faces", json={"models_dir": models_dir}).json()
                print(f"faces job {job['id']}: {job['total']} files")
                if job["total"] <= 5:
                    raise RuntimeError(
                        f"a mid-drain photograph needs more than 5 files to be mid of (got {job['total']});"
                        f" is --datasets pointing at the sample sets?"
                    )

                page.wait_for_function(
                    "(id) => window.__got.filter(m => m.job === id && m.state === 'running' && m.done > 4).length > 0",
                    arg=job["id"],
                    timeout=120_000,
                )
                page.evaluate(
                    "() => { const list = document.createElement('pre');"
                    " list.id = 'feed';"
                    " list.style.cssText = 'font: 12px/1.7 monospace; padding: 14px; margin: 0;"
                    " background: #101418; color: #9fe8b1; width: fit-content;';"
                    " const tail = window.__got.slice(-9);"
                    " list.textContent = tail.map(m => JSON.stringify(m)).join(String.fromCharCode(10));"
                    " document.body.appendChild(list); }"
                )
                mid = page.evaluate("() => window.__got[window.__got.length - 1]")
                keep(
                    "job-live",
                    page.locator("#feed").screenshot(),
                    "Mid-drain, exactly as the browser's WebSocket received it: one delta per committed"
                    " item, nobody polling.",
                    job_total=job["total"],
                    done_at_capture=mid.get("done"),
                )

                page.wait_for_function(
                    "(id) => window.__got.some(m => m.job === id && ['done','failed','cancelled'].includes(m.state))",
                    arg=job["id"],
                    timeout=1_800_000,
                )
                got = page.evaluate("() => window.__got")
                page.evaluate(
                    "() => { const tail = window.__got.slice(-6);"
                    " document.getElementById('feed').textContent ="
                    " tail.map(m => JSON.stringify(m)).join(String.fromCharCode(10)); }"
                )
                keep(
                    "job-feed",
                    page.locator("#feed").screenshot(),
                    "The end of the same feed: the terminal delta arrives only after its row committed.",
                    messages=len(got),
                    final_state=got[-1].get("state"),
                )

                # People are produced by the application too: clustering is
                # a job the app offers, watched on the same feed, and the
                # nicknames go on over the naming route.
                grouping = web.post("/jobs/cluster").json()
                page.wait_for_function(
                    "(id) => window.__got.some(m => m.job === id && ['done','failed','cancelled'].includes(m.state))",
                    arg=grouping["id"],
                    timeout=120_000,
                )
                settled = web.get(f"/jobs/{grouping['id']}").json()
                if settled["state"] != "done":
                    raise RuntimeError(f"the cluster job settled {settled['state']}: {settled['error']}")

                # Perceptual identity over the same feed: fingerprints,
                # then groups. Detection already recorded most hashes as
                # byproduct; the phash job backfills the rest.
                for asked in ("/jobs/phash", "/jobs/dupes"):
                    ran = web.post(asked).json()
                    page.wait_for_function(
                        "(id) => window.__got.some(m => m.job === id"
                        " && ['done','failed','cancelled'].includes(m.state))",
                        arg=ran["id"],
                        timeout=600_000,
                    )
                    outcome = web.get(f"/jobs/{ran['id']}").json()
                    if outcome["state"] != "done":
                        raise RuntimeError(f"{asked} settled {outcome['state']}: {outcome['error']}")
                page.close()

                names = web.get("/people").json()
                if not names:
                    raise RuntimeError("no people emerged from clustering -- did detection find any faces?")
                for n, person in enumerate(names, start=1):
                    christened = web.post(f"/p/{person['slug']}/name", json={"name": f"Person {n}"})
                    if christened.status_code >= 300:
                        raise RuntimeError(
                            f"naming {person['slug']} answered {christened.status_code}: {christened.text}"
                        )
                names = web.get("/people").json()
                print(f"clustered into {len(names)} people")

                # --- the grid, over real files --------------------------------
                page = watched_page()
                rows = web.get("/roots").json()
                # Read-only by construction (mode=ro): picks WHICH files to
                # photograph (there is no listing route yet); every pixel
                # still arrives over HTTP. `closing`, because sqlite3's
                # context manager manages transactions, not the connection.
                library = f"file:{(home / 'gallery.db').as_posix()}?mode=ro"
                with contextlib.closing(sqlite3.connect(library, uri=True)) as conn:
                    slug_rows = [
                        slug
                        for (slug,) in conn.execute(
                            "SELECT e.slug FROM entity e JOIN file f ON f.id = e.id"
                            " WHERE f.missing_since IS NULL ORDER BY f.id"
                        )
                    ]
                stride = max(1, len(slug_rows) // 18)
                page.set_content(_grid(base, slug_rows[::stride][:18]), wait_until="domcontentloaded")
                # An album, made and filled through the application, so the
                # shelves capture shows authored state the routes produced.
                keepsake = web.post("/albums", json={"name": "Sample picks"}).json()
                for pick in slug_rows[:3]:
                    kept = web.post(f"/t/{keepsake['slug']}/add", json={"file": pick})
                    if kept.status_code >= 300:
                        raise RuntimeError(f"album add answered {kept.status_code}: {kept.text}")

                broken = _all_loaded(page, "#grid")
                keep(
                    "library-grid",
                    page.locator("#grid").screenshot(),
                    "Eighteen of the library's files, spread across both datasets, served as 512-edge"
                    " thumbnails from the content-keyed cache the detection job pre-filled.",
                    library_files=len(slug_rows),
                    roots=len(rows),
                    broken_tiles=len(broken),
                )
                page.close()

                # --- the recipe shelves, from the app's own routes ------------
                page = watched_page()
                shelves = {
                    "Models": web.get("/models").json(),
                    "LoRAs": web.get("/loras").json(),
                    "Workflows": web.get("/workflows").json(),
                    "Albums": web.get("/albums").json(),
                }
                columns = []
                for title, listed in shelves.items():
                    lines = "".join(
                        f"<li>{html.escape(str(row['name']))} <b>{row['pictures']}</b></li>" for row in listed[:8]
                    )
                    columns.append(
                        f'<div style="min-width:190px"><h3 style="margin:0 0 6px;font:600 14px system-ui;'
                        f'color:#dfe5e1">{title} ({len(listed)})</h3>'
                        f'<ul style="margin:0;padding-left:18px;font:12.5px/1.8 system-ui;color:#9fb0a6">'
                        f"{lines}</ul></div>"
                    )
                page.set_content(
                    '<body style="margin:0;background:#14171a">'
                    f'<div id="shelves" style="display:flex;gap:26px;padding:18px;width:fit-content">'
                    f"{''.join(columns)}</div>"
                )
                keep(
                    "recipe-shelves",
                    page.locator("#shelves").screenshot(),
                    "The recipe axis, counted by real joins: every list fetched from the app's own"
                    " shelf routes at capture time, the album made and filled through the"
                    " application a moment earlier.",
                    models=len(shelves["Models"]),
                    loras=len(shelves["LoRAs"]),
                    workflows=len(shelves["Workflows"]),
                    albums=len(shelves["Albums"]),
                )
                page.close()

                # --- the copy shelf, from the app's own routes ----------------
                page = watched_page()
                bodies = web.get("/dupes").json()
                if bodies:
                    tiles = "".join(
                        f'<div style="width:132px;text-align:center;font:12.5px system-ui;color:#dfe5e1">'
                        f'<img src="{base}/thumb/{group["slug"]}" style="width:120px;height:120px;'
                        f'object-fit:cover;border-radius:6px;background:#222">'
                        f'<div style="padding-top:5px">&times;{group["copies"]} bodies</div></div>'
                        for group in bodies[:8]
                    )
                    shelf_body = (
                        f'<div id="copies" style="display:flex;gap:14px;padding:16px;width:fit-content">{tiles}</div>'
                    )
                else:
                    shelf_body = (
                        '<div id="copies" style="padding:22px;font:14px system-ui;color:#9fb0a6;width:fit-content">'
                        "No perceptual duplicates in these samples at the configured threshold &mdash; "
                        f"{len(slug_rows)} files, every one its own picture. The similar-style renders stayed apart."
                        "</div>"
                    )
                page.set_content(f'<body style="margin:0;background:#14171a">{shelf_body}')
                broken = _all_loaded(page, "#copies") if bodies else []
                keep(
                    "copy-shelf",
                    page.locator("#copies").screenshot(),
                    "GET /dupes after the phash and dupes jobs: each group is one picture shown by its"
                    " best body, counted. An empty shelf is the honest answer when the library holds"
                    " no perceptual copies -- and proof the similar renders did not falsely merge.",
                    groups=len(bodies),
                    copies_each=[group["copies"] for group in bodies[:8]],
                    broken_tiles=len(broken),
                )
                page.close()

                # --- one render, full preview ---------------------------------
                page = watched_page()
                render_slug = slug_rows[0]
                page.set_content(
                    f'<body style="margin:0;background:#14171a;padding:10px">'
                    f'<img id="big" src="{base}/preview/{render_slug}" style="max-width:900px;display:block">'
                )
                broken = _all_loaded(page, "body")
                recipe = web.get(f"/i/{render_slug}").json()
                keep(
                    "render-preview",
                    page.locator("#big").screenshot(),
                    "One of the app's own renders at lightbox size (1440 edge), decoded by the browser"
                    " from the same cache -- with the recipe its own page serves for it.",
                    slug=render_slug,
                    checkpoint=recipe["checkpoint"],
                    seed=recipe["seed"],
                    parsed_fields=recipe["fields"],
                    broken_tiles=len(broken),
                )
                page.close()

                # --- people, with real avatars --------------------------------
                page = watched_page()
                cards = "".join(
                    f'<div style="width:132px;text-align:center;font:13px system-ui;color:#dfe5e1">'
                    f'<img src="{base}/avatar/{person["slug"]}" style="width:120px;height:120px;'
                    f'border-radius:50%;object-fit:cover;background:#222">'
                    f'<div style="padding-top:6px">{html.escape(person["name"])}</div>'
                    f'<div style="color:#8b968f">{person["pictures"]} pictures</div></div>'
                    for person in names
                )
                page.set_content(
                    '<body style="margin:0;background:#14171a">'
                    f'<div id="folk" style="display:flex;gap:14px;padding:16px;width:fit-content">{cards}</div>'
                )
                broken = _all_loaded(page, "#folk")
                keep(
                    "people-index",
                    page.locator("#folk").screenshot(),
                    "The people index: every avatar is the highest-confidence detected face of its"
                    " cluster, cropped with context by the app. Names are nicknames for clusters,"
                    " never identity claims.",
                    people=len(names),
                    pictures_each=[person["pictures"] for person in names],
                    broken_tiles=len(broken),
                )
                page.close()

                # --- one person, across the library ---------------------------
                page = watched_page()
                biggest = names[0]
                person_page = web.get(f"/p/{biggest['slug']}").json()
                their = [picture["slug"] for picture in person_page["pictures"]][:12]
                page.set_content(_grid(base, their), wait_until="domcontentloaded")
                broken = _all_loaded(page, "#grid")
                keep(
                    "person-page",
                    page.locator("#grid").screenshot(),
                    f"Everything the primary clustering attributes to {biggest['name']}, as their page"
                    " serves it -- the cross-axis view: one person, wherever their pictures live.",
                    person=biggest["name"],
                    pictures=len(person_page["pictures"]),
                    across_folders=[row["folder"] for row in person_page["across_folders"]],
                    broken_tiles=len(broken),
                )
                page.close()
                browser.close()
        finally:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=15)
    _publish(staging)
    return taken


STYLE = """
:root {
  --bg: #f6f8f6; --surface: #ffffff; --ink: #1b211d; --muted: #5c6660;
  --accent: #1e7d46; --line: #e2e7e3; --well: #eef1ee; --wire: #101418;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #12161a; --surface: #1a2025; --ink: #e8ece9; --muted: #93a09a;
    --accent: #4cc47a; --line: #2a3238; --well: #161b1f; --wire: #0b0e11;
  }
}
:root[data-theme="dark"] {
  --bg: #12161a; --surface: #1a2025; --ink: #e8ece9; --muted: #93a09a;
  --accent: #4cc47a; --line: #2a3238; --well: #161b1f; --wire: #0b0e11;
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--ink); margin: 0;
  font: 16px/1.6 "Source Sans 3", "Segoe UI", system-ui, sans-serif;
}
main { max-width: 960px; margin: 0 auto; padding: 40px 20px 80px; }
header { border-bottom: 2px solid var(--ink); padding-bottom: 20px; margin-bottom: 8px; }
h1 {
  font-family: Archivo, "Segoe UI", system-ui, sans-serif;
  font-size: clamp(28px, 5vw, 40px); font-weight: 700; margin: 0 0 6px; text-wrap: balance;
}
.stamp { color: var(--muted); font-size: 14px; }
.stamp code { font-family: "JetBrains Mono", ui-monospace, monospace; color: var(--accent); }
h2 {
  font-family: Archivo, "Segoe UI", system-ui, sans-serif;
  font-size: 20px; font-weight: 600; margin: 44px 0 2px;
}
.contract { color: var(--muted); font-size: 14px; margin: 0 0 16px; }
.card {
  background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
  margin: 0 0 18px; overflow: hidden;
}
.frame { background: var(--well); padding: 16px; display: flex; justify-content: center; overflow-x: auto; }
.frame img { max-width: 100%; height: auto; display: block; border-radius: 4px; }
.frame.wire { background: var(--wire); }
.card .about { padding: 14px 18px 16px; }
.card .about p { margin: 0 0 10px; }
.facts { display: flex; flex-wrap: wrap; gap: 6px 8px; }
.fact {
  font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 12.5px;
  background: var(--well); border: 1px solid var(--line); border-radius: 999px;
  padding: 2px 10px; color: var(--muted); white-space: nowrap;
}
.fact b { color: var(--accent); font-weight: 600; }
footer { color: var(--muted); font-size: 13px; margin-top: 40px; }
"""


def _fact_chips(facts: dict) -> str:
    chips = []
    for key, value in facts.items():
        shown = json.dumps(value) if isinstance(value, (list, dict)) else str(value)
        if len(shown) > 90:
            shown = shown[:87] + "..."
        chips.append(f'<span class="fact">{html.escape(key)} <b>{html.escape(shown)}</b></span>')
    return "".join(chips)


def _card(entry: dict) -> str:
    image = base64.b64encode((EVIDENCE / entry["file"]).read_bytes()).decode("ascii")
    wire = " wire" if entry["name"].startswith("job-") else ""
    return (
        f'<div class="card">'
        f'<div class="frame{wire}"><img alt="{html.escape(entry["name"])}"'
        f' src="data:image/png;base64,{image}"></div>'
        f'<div class="about"><p>{html.escape(entry["caption"])}</p>'
        f'<div class="facts">{_fact_chips(entry["facts"])}</div></div>'
        f"</div>"
    )


def build() -> pathlib.Path:
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    by_name = {entry["name"]: entry for entry in manifest["evidence"]}

    sections = []
    for title, contract, wanted in SECTIONS:
        cards = "".join(_card(by_name[name]) for name in wanted if name in by_name)
        if cards:
            sections.append(f"<h2>{html.escape(title)}</h2><p class='contract'>{html.escape(contract)}</p>{cards}")

    page = (
        "<title>The Browser Agrees</title>\n"
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        "family=Archivo:wght@600;700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400"
        '&display=swap">\n'
        f"<style>{STYLE}</style>\n"
        "<main><header>"
        "<h1>The Browser Agrees</h1>"
        f'<div class="stamp">Chromium photographing the live app over the sample datasets &middot; commit '
        f"<code>{html.escape(manifest['commit'])}</code> &middot; {html.escape(manifest['taken_at'])}"
        "</div></header>"
        + "".join(sections)
        + "<footer>Every image is a Chromium screenshot of the running application serving real sample"
        " files -- no mockups, no synthetic pixels, and every mutation a route the application offers."
        " Regenerate with <code>just bench browser-report</code>."
        "</footer></main>\n"
    )
    target = EVIDENCE / "report.html"
    target.write_text(page, encoding="utf-8", newline="\n")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Photograph the app over the sample datasets.")
    parser.add_argument("--datasets", default=DATASETS)
    parser.add_argument("--models-dir", default=MODELS)
    asked = parser.parse_args()

    taken = capture(asked.datasets, asked.models_dir)
    manifest = {
        "taken_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": _commit_stamp(),
        "evidence": taken,
    }
    (EVIDENCE / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    print(build())


if __name__ == "__main__":
    main()
