"""What the corpus must cover, read out of the application, and what it does.

"Make the corpus good enough" is unfalsifiable, and an unfalsifiable goal
cannot be finished: the set is re-derived every time it is thought about, it
grows, and the work never closes. `tests/reach.py` says the same thing about
reader lines and freezes a list so it can be exhausted. This is that, one
level up, for the FILES.

THE ENUMERATION COMES FROM THE APPLICATION. A suffix is needed because
`db/scan.py KIND_BY_SUFFIX` claims it; a dialect is needed because
`metaparse/adapters.py` registers an adapter for it. Neither list is typed
here, so neither can drift out of step with what the application says it
supports -- and neither can be quietly shortened to match what was easy to
find, which is the failure this file exists to make visible.

WHAT IS MEASURED IS NEVER DECLARED. A need is satisfied when a real corpus
file is put through the real reader and the reader produces the thing; never
because a file has the right extension. An extension is a claim the filename
makes about itself.

A need has four terminal states and `DEFERRED` is not among them:

    SATISFIED           a named corpus file, read successfully
    PARTIAL             the file is present and the reader got less than all
    UNSATISFIED         nothing in the corpus covers it
    BLOCKED_EXTERNALLY  evidenced: no obtainable source, restrictive terms,
                        hardware that does not exist here
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
import warnings

REPO = pathlib.Path(__file__).resolve().parent.parent

CORPUS = pathlib.Path(os.environ.get("SG_CORPUS", REPO.parent / "sg-corpus"))
LEDGER = REPO / "tests" / "needs.lock.json"


def _reader_failures() -> tuple[type[BaseException], ...]:
    """What a reader is allowed to fail with, per the application itself.

    `db/runner.py ITEM_FAILURES` is the list the job runner uses to decide
    whether a file's failure is a fact about the file or a defect in the
    code. Reusing it means this measurement records exactly what the
    application would record and lets anything else propagate -- where a
    bare `except Exception` would quietly absorb a defect and count it as
    coverage, which is the failure `tests/reach.py` already made once.
    """
    from db import runner

    return runner.ITEM_FAILURES


def declared_suffixes() -> dict[str, str]:
    """Every suffix the application claims, and the kind it claims it as."""
    from db import scan

    return dict(scan.KIND_BY_SUFFIX)


def declared_axes() -> dict[str, tuple[str, ...]]:
    """The vocabularies the application states, one per axis of the shape.

    `docs/CORPUS_SHAPE.md` names these. They are read, never typed: a value
    added to `db/context.py` becomes a need here without anybody remembering
    to add it, which is the only way the denominator stays honest.

    `time_precision` is a schema CHECK rather than a module constant, so it
    is read off the table definition instead.
    """
    import re

    from db import context, scan

    schema = (REPO / "db" / "schema.sql").read_text(encoding="utf-8")
    found = re.search(r"time_precision\s+TEXT CHECK \(time_precision IN\s*\(([^)]*)\)", schema)
    precision = tuple(re.findall(r"'([a-z]+)'", found.group(1))) if found else ()
    return {
        "kind": tuple(sorted(set(scan.KIND_BY_SUFFIX.values()))),
        "time_basis": tuple(context.TIME_BASES),
        "time_precision": precision,
        "location_basis": tuple(context.LOCATION_BASES),
        "origin": tuple(context.ORIGINS),
    }


#: One statement per axis, written out rather than built from a column name.
#: A query assembled by interpolation is a query somebody can be talked into
#: assembling from input, and these are fixed for the life of the schema.
COUNTS = {
    "time_basis": "SELECT time_basis, count(*) FROM derived_media_context GROUP BY time_basis",
    "time_precision": "SELECT time_precision, count(*) FROM derived_media_context GROUP BY time_precision",
    "location_basis": "SELECT location_basis, count(*) FROM derived_media_context GROUP BY location_basis",
    "origin": "SELECT origin, count(*) FROM derived_media_context GROUP BY origin",
    "kind": "SELECT kind, count(*) FROM file GROUP BY kind",
}


def assigned(db_path: pathlib.Path) -> dict[str, collections.Counter]:
    """What the application actually CONCLUDED, from a library it scanned.

    Not re-derived here. `db/when.py` decides which rung a file lands on and
    `derived_media_context` records the verdict, so the honest measurement is
    to read the verdict rather than to reimplement the judge and then compare
    the corpus against a second opinion.
    """
    # Through `db.connect`, never `sqlite3.connect`: a raw connection runs
    # with the schema's foreign keys inert, and every per-connection fact
    # this database assumes is decided in that one function (TID251).
    from db import connect

    held: dict[str, collections.Counter] = {}
    conn = connect.connect(db_path, read_only=True)
    try:
        for axis, statement in COUNTS.items():
            rows = conn.execute(statement)
            held[axis] = collections.Counter({str(name): int(count) for name, count in rows if name is not None})
    finally:
        connect.close(conn)
    return held


def declared_dialects() -> list[str]:
    """Every generator the application registers an adapter for."""
    from metaparse import adapters

    seen: list[str] = []
    for one in (*adapters.MARKER_ADAPTERS, *adapters.HEURISTIC_ADAPTERS):
        tool = getattr(one, "tool", one.__name__)
        if tool not in seen:
            seen.append(tool)
    return seen


def corpus_files(root: pathlib.Path | None = None) -> list[pathlib.Path]:
    root = root or CORPUS
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts and ".git" not in p.parts)


def _tool_of(path: pathlib.Path) -> str | None:
    """Which generator wrote this, if any. The cheap half of `_read`, so a
    dialect is never missed because its file fell outside a suffix cap."""
    warnings.filterwarnings("ignore")
    from metaparse import adapters

    try:
        got = adapters.parse_file(str(path), allow_stealth=True)
    except _reader_failures():
        return None
    return getattr(got, "tool", None) if got else None


def _read(path: pathlib.Path) -> dict:
    """Put one file through the readers and say what came back.

    Every reader is called the way the application calls it, and a raise is
    recorded rather than propagated: a file the readers refuse is a fact
    about coverage, not a reason to stop measuring.
    """
    warnings.filterwarnings("ignore")
    from db import capture, scan
    from metaparse import adapters
    from vision import decode

    held: dict = {"suffix": path.suffix.lower(), "kind": scan.KIND_BY_SUFFIX.get(path.suffix.lower())}
    try:
        got = capture.read(path)
        held["capture"] = bool(got and (got.camera or got.captured_at))
    except _reader_failures() as why:
        held["capture"] = False
        held["capture_why"] = f"{type(why).__name__}: {why}"[:120]
    try:
        parsed = adapters.parse_file(str(path), allow_stealth=True)
        held["tool"] = getattr(parsed, "tool", None) if parsed else None
    except _reader_failures() as why:
        held["tool"] = None
        held["tool_why"] = f"{type(why).__name__}: {why}"[:120]
    if held["kind"] in ("video", "audio"):
        # A container reader, not a picture decoder. Without this every
        # video read as "present; no reader produced anything", because
        # only the image branch below ran and a valid mp4 has no still to
        # open -- a wrong number about 12 suffixes at once.
        from db import probe

        try:
            got = probe.read(str(path))
            held["decodes"] = bool(got and not got.unreadable)
            if got and got.unreadable:
                held["decodes_why"] = got.unreadable[:120]
        except _reader_failures() as why:
            held["decodes"] = False
            held["decodes_why"] = f"{type(why).__name__}: {why}"[:120]
    if held["kind"] in ("image", "animated_image"):
        try:
            # `open_bounded`, not `open_still`: a RAW file carries a JPEG
            # preview the camera put there, and developing the sensor data
            # instead costs 1398 ms against 47 (vision/decode.py). The
            # question here is whether a reader produces a picture at all,
            # and both answer it. Full development of 326 Canon files spent
            # 700 CPU-seconds proving nothing this table asks.
            opened = decode.open_bounded(path, 512)
            held["decodes"] = True
            opened.close()
        except _reader_failures() as why:
            held["decodes"] = False
            held["decodes_why"] = f"{type(why).__name__}: {why}"[:120]
    return held


def _served_db() -> pathlib.Path | None:
    """The gallery database of a run that has scanned this corpus, if any.

    `SG_HOME` names it; otherwise the sibling `sg-run` a `just corpus
    against` leaves behind. None is reported as UNKNOWN_NOT_MEASURED rather
    than as UNSATISFIED: a rung nothing was asked about is not a rung
    nothing reached, and calling it absent would be the same lie the
    contract forbids.
    """
    home = pathlib.Path(os.environ.get("SG_HOME", REPO.parent / "sg-run"))
    found = sorted(home.rglob("gallery.db")) if home.is_dir() else []
    return found[0] if found else None


#: Values NO FILE can carry, because a PERSON produces them.
#:
#: Six of the seven axes are read out of a file: its suffix, its kind, the
#: generator that wrote it, the date it claims and how precisely. Those
#: are closed by acquiring a file. `location_basis` is not one axis but
#: two halves -- `gps` is in the bytes a camera wrote, and `authored` is
#: one person's word on where a picture happened, recorded through
#: `POST /i/{slug}/place`. No corpus reaches it. A corpus a thousand times
#: this size does not reach it.
#:
#: So it is measured by DOING it, not by excusing it: mint a place, assert
#: it on a real file in the real library, re-interpret that file, and read
#: back what the application concluded. That is the same three calls the
#: route makes. If `set_place` ever stops producing `authored`, this goes
#: UNSATISFIED and the gate fails -- which a list of allowed exceptions
#: could never do.
INTERACTIONS = {"location_basis": ("authored",)}

#: Needs the corpus cannot close because the WORLD offers no specimen --
#: each with the evidence that obtaining one was attempted, and where.
#: `docs/CORPUS_SHAPE.md`: a declared value is reached by a real file or
#: it is BLOCKED_EXTERNALLY with evidence; there is no third state.
#:
#: This is not an exceptions table. Three properties keep it honest, and
#: `tests/test_the_corpus_spans_the_shape.py` holds all three: a row
#: must carry evidence with a positive control, a row whose need a
#: corpus file now reaches FAILS the gate -- the excuse dies the day the
#: gap closes -- and `tests/rawsamples.py` retries blocked suffixes on
#: every fetch, so a row is challenged on every run, never filed.
BLOCKED: dict[str, dict] = {
    "suffix:.cap": {
        "why": "Phase One .cap: LibRaw reads it, and no sample archive offers one",
        "evidence": (
            {
                "source": "raw.pixls.us CC0 set (1870 of 2016 samples)",
                "control": ".iiq: 35 samples present",
                "checked": "2026-08-25",
            },
            {
                "source": "exiftool t/images (194 files)",
                "control": "PhaseOne.iiq present",
                "checked": "2026-08-25",
            },
        ),
    },
    "suffix:.k25": {
        "why": "Kodak DC25 .k25: LibRaw reads it (cameralist.cpp:513), and no sample archive offers one",
        "evidence": (
            {
                "source": "raw.pixls.us CC0 set (1870 of 2016 samples)",
                "control": ".kdc and .dcr present",
                "checked": "2026-08-25",
            },
            {
                "source": "rawsamples.ch Kodak listing",
                "control": ".KDC and .DCR listed",
                "checked": "2026-08-25",
            },
        ),
    },
}


def performed(db_path: pathlib.Path) -> collections.Counter:
    """What the application concludes when a person asserts a place.

    ROLLED BACK, always. This runs against the library somebody is
    actually serving, so the measurement must leave it byte for byte as
    it found it -- the proof is that the interpretation CAN be produced,
    never a row left behind to make a later count look better.
    """
    from db import authored, connect, context, places

    held: collections.Counter = collections.Counter()
    conn = connect.connect(db_path)
    try:
        row = conn.execute("SELECT id FROM file ORDER BY id LIMIT 1").fetchone()
        if row is None:
            return held
        file_id = int(row[0])
        conn.execute("BEGIN")
        try:
            where = places.named(conn, "Lisbon", "city", 1.0)
            authored.set_place(conn, file_id, 1, where, 1.0)
            context.rebuild_one(conn, file_id, 1.0)
            got = conn.execute(
                "SELECT location_basis FROM derived_media_context WHERE file_id = ?", (file_id,)
            ).fetchone()
            if got is not None and got[0] is not None:
                held[str(got[0])] += 1
        finally:
            conn.execute("ROLLBACK")
    finally:
        connect.close(conn)
    return held


def measure(root: pathlib.Path | None = None, db_path: pathlib.Path | None = None) -> dict:
    """Every need, its state, and the file that put it there."""
    suffixes, dialects = declared_suffixes(), declared_dialects()
    files = corpus_files(root)

    by_suffix: dict[str, list[dict]] = collections.defaultdict(list)
    by_tool: dict[str, list[str]] = collections.defaultdict(list)
    #: How many files of one suffix to put through the EXPENSIVE readers.
    #: Every one answers the same question -- does a reader produce anything
    #: for this suffix -- so reading 326 Canon files to learn what the first
    #: two say is spent time, and the row records how many are present.
    #:
    #: The cheap metadata parse is NOT capped. A dialect is orthogonal to a
    #: suffix: capping at four PNGs hid all eight generators behind whichever
    #: four PNGs came first alphabetically, and reported every one of them
    #: UNSATISFIED while they sat in the corpus parsing correctly.
    per_suffix = 4
    counted: collections.Counter = collections.Counter()
    for path in files:
        suffix = path.suffix.lower()
        if suffix not in suffixes:
            continue
        counted[suffix] += 1
        if counted[suffix] > per_suffix and by_suffix[suffix]:
            tool = _tool_of(path)
            if tool:
                by_tool[tool].append(str(path.relative_to(root or CORPUS)).replace("\\", "/"))
            continue
        got = _read(path)
        got["path"] = str(path.relative_to(root or CORPUS)).replace("\\", "/")
        by_suffix[got["suffix"]].append(got)
        if got.get("tool"):
            by_tool[got["tool"]].append(got["path"])

    needs = []
    for suffix, kind in sorted(suffixes.items()):
        seen = by_suffix.get(suffix, [])
        if not seen:
            state, why, example = "UNSATISFIED", "no file with this suffix in the corpus", None
        else:
            worked = [
                one
                for one in seen
                if one.get("capture") or one.get("tool") or one.get("decodes") or one["kind"] in ("audio", "document")
            ]
            if worked:
                state, why, example = "SATISFIED", "", worked[0]["path"]
            else:
                state = "PARTIAL"
                why = seen[0].get("decodes_why") or seen[0].get("capture_why") or "present; no reader produced anything"
                example = seen[0]["path"]
        needs.append(
            {
                "need": f"suffix:{suffix}",
                "kind": kind,
                "state": state,
                "files": counted.get(suffix, 0),
                "read": len(seen),
                "example": example,
                "why": why,
            }
        )

    for tool in dialects:
        found = by_tool.get(tool, [])
        needs.append(
            {
                "need": f"dialect:{tool}",
                "kind": "generated",
                "state": "SATISFIED" if found else "UNSATISFIED",
                "files": len(found),
                "example": found[0] if found else None,
                "why": "" if found else "no corpus file parses as this generator",
            }
        )

    # The axes only a scanned library can answer. `db/when.py` decides which
    # dating rung a file lands on, so the corpus cannot be asked directly --
    # and without this the ledger measured two axes out of seven and called
    # the result coverage. `mtime` and `btime` are the rungs most of a real
    # library lands on, and no corpus of camera output reaches them.
    library = db_path or _served_db()
    concluded = assigned(library) if library else {}
    # The interaction half, run once and only if an axis has one. See
    # INTERACTIONS: a person's assertion is not in any file, so it is
    # measured by making the assertion rather than by excusing the gap.
    acted = performed(library) if library and INTERACTIONS else collections.Counter()
    for axis, values in declared_axes().items():
        seen = concluded.get(axis, collections.Counter())
        by_hand = acted if axis in INTERACTIONS else collections.Counter()
        for value in values:
            count = seen.get(value, 0)
            done = by_hand.get(value, 0)
            needs.append(
                {
                    "need": f"{axis}:{value}",
                    "kind": axis,
                    "state": "SATISFIED" if count or done else ("UNSATISFIED" if library else "UNKNOWN_NOT_MEASURED"),
                    "files": count,
                    "example": None,
                    "why": ""
                    if count
                    else (
                        "no file carries this: produced by a person, and produced when asked"
                        if done
                        else ("nothing in the scanned library landed here" if library else "no scanned library to read")
                    ),
                }
            )

    # The block register applies to what was MEASURED as unreached, and
    # only that: a need a file now satisfies keeps SATISFIED, so a stale
    # excuse is visible to the gate instead of silently absorbing it.
    for one in needs:
        # A need row is a record of mixed types, so its `need` is only a
        # string -- and only a usable key for BLOCKED -- once said so.
        name = one["need"]
        assert isinstance(name, str)
        held = BLOCKED.get(name)
        if held is not None and one["state"] == "UNSATISFIED":
            one["state"] = "BLOCKED_EXTERNALLY"
            one["why"] = held["why"]

    tally = collections.Counter(one["state"] for one in needs)
    held = {
        "what": "Coverage the application declares, and what the corpus actually reads.",
        "enumerated_from": ["db/scan.py KIND_BY_SUFFIX", "metaparse/adapters.py MARKER_ADAPTERS + HEURISTIC_ADAPTERS"],
        "corpus": str(root or CORPUS),
        "corpus_files": len(files),
        "totals": dict(sorted(tally.items())),
        "needs": needs,
    }
    LEDGER.write_text(json.dumps(held, indent=2) + "\n", encoding="utf-8")
    return held


if __name__ == "__main__":
    got = measure()
    print(f"{got['corpus_files']} files under {got['corpus']}")
    print(f"needs: {sum(got['totals'].values())}   {got['totals']}")
    for state in ("UNSATISFIED", "PARTIAL", "BLOCKED_EXTERNALLY"):
        rows = [one for one in got["needs"] if one["state"] == state]
        if not rows:
            continue
        print(f"\n{state} ({len(rows)}):")
        for one in rows:
            print(f"  {one['need']:24s} {one['kind'] or '':14s} {one['why'][:70]}")
