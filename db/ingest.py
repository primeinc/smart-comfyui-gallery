"""Everything one file says about itself, written as rows.

`scan.py` decides what a file on disk *is*. This decides what it *says*, and
puts each statement where it can be queried: the recipe in `generation`, the
weights and bodies in `artifact`, the prompt text in `prompt`, every other
field in `file_param`, and the carriers themselves in `blob` so nothing is
lost to a parser that does not understand it yet.

Two rules the layout depends on.

Nothing lands in a JSON column. A field inside a blob cannot be searched,
filtered, or counted, so an unrecognised key goes to `file_param` under its
own name and `param_key` learns it. The registry is what makes a field
nobody predicted available as a facet the day it first appears.

The carrier is stored whether or not it was understood. `file_blob.parsed_by`
is NULL until something claims it, which turns unparsed metadata into a
queryable backlog and makes improving an adapter a re-parse of the database
rather than a re-read of the disk.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field

import metaparse
from metaparse.containers import load_raw
from metaparse.typed import GenerationParams

from . import capture as capture_module
from .scan import mint

#: PNG text chunks that carry a whole workflow graph rather than a value.
_GRAPH_SLOTS = ("workflow", "prompt")

#: What each parser owns, and therefore what it has to be able to take back.
#:
#: Without this a re-parse could only add. A file first read by an adapter
#: that found three LoRAs and then re-read by a corrected one that finds one
#: kept all three, because `INSERT OR REPLACE` touches only the keys the new
#: parse produced; a field that stopped being emitted stayed in `file_param`
#: with `param_key.occurrences` still counting it. The stale rows are
#: indistinguishable from real ones -- neither table carries which parser
#: wrote it -- so "improving a parser is a re-parse of the database rather
#: than a re-read of every file on disk" was not true of anything.
_OWNED = {
    "generation": {
        "sources": ("generation", "container"),
        "roles": (
            "checkpoint", "refiner", "lora", "vae", "controlnet", "upscaler",
            "embedding", "hypernetwork", "ip_adapter", "text_encoder", "unet",
        ),
        "carriers": ("png_text", "xmp"),
    },
    "camera": {
        "sources": ("exif",),
        "roles": ("captured_with", "mounted_lens"),
        "carriers": ("exif",),
    },
    "sidecar": {
        "sources": ("sidecar",),
        "roles": (),
        "carriers": ("sidecar",),
    },
}


def retract(conn, file_id: int, scope: str) -> None:
    """Take back everything one parser wrote about this file.

    Called before the parser writes again, in the same transaction, so a
    re-parse is a replacement rather than an accumulation. Deleting the
    carriers rather than replacing them also lets `blob_reclaim` collect the
    payloads that are now referenced by nothing -- `INSERT OR REPLACE` fires
    no DELETE trigger, so every re-parse whose bytes had changed used to
    strand a whole workflow graph in `blob` permanently.
    """
    owned = _OWNED[scope]
    for table, column, values in (
        ("file_param", "source", owned["sources"]),
        ("file_artifact", "role", owned["roles"]),
        ("file_blob", "carrier", owned["carriers"]),
    ):
        if not values:
            continue
        marks = ",".join("?" * len(values))
        conn.execute(
            f"DELETE FROM {table} WHERE file_id = ? AND {column} IN ({marks})",
            (file_id, *values),
        )

@dataclass
class Ingested:
    """What one file contributed, for a caller that wants to report it."""

    tool: str | None = None
    detection: str | None = None
    prompt_id: int | None = None
    artifacts: list[tuple[str, str]] = field(default_factory=list)
    params: int = 0
    carriers: int = 0
    unparsed: int = 0
    captured: bool = False
    #: Set when the bytes could not be opened as an image at all, so a caller
    #: can report the file rather than reading "no camera tags" off it.
    unreadable: str | None = None


def _name_key(text: str) -> str:
    """The one normalization a name is deduped and searched by.

    Ingest and the search box have to agree on this or the library grows a
    row per spelling of the same model.
    """
    return "".join(character for character in str(text).lower() if character.isalnum())


def artifact(conn, kind: str, name: str, now: float, *, quoted=None, sha=None) -> int:
    """The row for one named thing, created once.

    Deduped on (kind, name_key, content hash). Most generation metadata is a
    string scraped out of a chunk with no file in hand, so the name is what
    actually dedupes and the hash is usually NULL.
    """
    row = conn.execute(
        "SELECT id FROM artifact WHERE kind = ? AND name_key = ?"
        " AND IFNULL(content_sha256, '') = IFNULL(?, '')",
        (kind, _name_key(name), sha),
    ).fetchone()
    if row:
        return row[0]
    artifact_id = mint(conn, "artifact", f"{kind}-{name}")
    conn.execute(
        "INSERT INTO artifact(id, kind, name, name_key, content_sha256, quoted_hash,"
        " first_seen_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
        (artifact_id, kind, str(name), _name_key(name), sha, quoted, now),
    )
    return artifact_id


def prompt(conn, text: str, now: float) -> int | None:
    """One prompt, deduped by the hash of its text.

    A `prompt_dedupe` trigger turns a colliding insert into a no-op rather
    than a replace, because REPLACE would delete the existing row and orphan
    its FTS entry.
    """
    text = (text or "").strip()
    if not text:
        return None
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    row = conn.execute("SELECT id FROM prompt WHERE text_hash = ?", (digest,)).fetchone()
    if row:
        return row[0]
    prompt_id = mint(conn, "prompt", f"prompt-{digest[:10]}")
    conn.execute(
        "INSERT INTO prompt(id, text, text_hash, created_at) VALUES(?, ?, ?, ?)",
        (prompt_id, text, digest, now),
    )
    return prompt_id


def _as_number(text: str) -> float | None:
    """A scraped string as a number, or None where it is not one.

    `float()` alone is too generous in two directions and both reach
    `value_num`, which facets range over.

    It accepts Python's own literal grammar, which no metadata format shares:
    `1_000` became 1000.0, indistinguishable from a real thousand written by
    a tool that meant it.

    It accepts infinities and NaN. `capture._number` already refuses those and
    says why -- NaN compares false against everything including itself, so a
    range facet silently drops the row -- and the same column was filtered on
    the EXIF path and unfiltered on this one.
    """
    if "_" in text or not any(character.isdigit() for character in text):
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _param(conn, file_id: int, source: str, key: str, value) -> bool:
    """One field, as text and as a number where it is one."""
    if value is None:
        return False
    if isinstance(value, (dict, list)):
        # Flattened rather than dumped: a structure written as JSON is a
        # field nothing can search, which is the shape this module exists to
        # avoid. Each leaf becomes its own dotted key.
        wrote = False
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for leaf, inner in items:
            wrote |= _param(conn, file_id, source, f"{key}.{leaf}", inner)
        return wrote
    text = str(value).strip()
    if not text:
        return False
    number = _as_number(text)
    # Never INSERT OR REPLACE: it fires no DELETE trigger, so the FTS entry
    # keyed on the old rowid is stranded and the registry count drifts.
    # The schema refuses it outright.
    conn.execute(
        "INSERT INTO file_param(file_id, source, key, value_text, value_num)"
        " VALUES(?, ?, ?, ?, ?)"
        " ON CONFLICT(file_id, source, key) DO UPDATE SET"
        " value_text = excluded.value_text, value_num = excluded.value_num",
        (file_id, source, key, text, number),
    )
    return True


def _carrier(conn, file_id: int, carrier: str, slot: str, payload, now: float, parsed_by=None):
    """Keep the payload, and record whether anything understood it."""
    if payload is None:
        return
    binary = isinstance(payload, bytes)
    data = payload if binary else str(payload).encode("utf-8")
    if not data:
        return
    digest = hashlib.sha256(data).hexdigest()
    conn.execute(
        "INSERT OR IGNORE INTO blob(hash, payload, payload_bin, byte_len) VALUES(?, ?, ?, ?)",
        (digest, None if binary else str(payload), payload if binary else None, len(data)),
    )
    # Not INSERT OR REPLACE. REPLACE fires no DELETE trigger, so `blob_reclaim`
    # never sees the payload the row used to point at and it stays in `blob`
    # forever -- measured in whole workflow graphs, per re-parse, per file.
    conn.execute(
        "INSERT INTO file_blob(file_id, carrier, slot, blob_hash, parsed_by, seen_at)"
        " VALUES(?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(file_id, carrier, slot) DO UPDATE SET"
        " blob_hash = excluded.blob_hash, parsed_by = excluded.parsed_by,"
        " seen_at = excluded.seen_at",
        (file_id, carrier, slot, digest, parsed_by, now),
    )


def generation(conn, file_id: int, path, now: float, out: Ingested) -> None:
    """The recipe: tool, prompt, weights, sampler settings, and the long tail."""
    retract(conn, file_id, "generation")
    parsed = metaparse.parse_file(path)
    raw = load_raw(path)

    reader = f"metaparse/{parsed.tool}" if parsed is not None else None
    # The text the adapter actually read, so the claim is about this carrier
    # rather than about its name. Marking only `_GRAPH_SLOTS` said every
    # A1111 `parameters` chunk was un-understood -- while the whole recipe
    # had just been read out of it -- so the backlog "what does nothing
    # understand yet" listed the files that parsed best.
    consumed = (parsed.raw or "").strip() if parsed is not None else ""
    if raw is not None:
        for slot, value in raw.text.items():
            understood = slot in _GRAPH_SLOTS or (
                bool(consumed) and str(value).strip() == consumed
            )
            claimed = reader if understood else None
            _carrier(conn, file_id, "png_text", slot, value, now, parsed_by=claimed)
            out.carriers += 1
            if claimed is None:
                out.unparsed += 1
        if raw.xmp:
            _carrier(conn, file_id, "xmp", "packet", raw.xmp, now)
            out.carriers += 1
            out.unparsed += 1
        # Container facts are metadata too, and the only kind every file has.
        for key, value in (
            ("Format", raw.format), ("Mode", raw.mode),
            ("Width", raw.width), ("Height", raw.height),
        ):
            if _param(conn, file_id, "container", key, value):
                out.params += 1
        # ...and the two the file row carries in its own right, because "the
        # pixels on disk" is not a searchable string, it is what every layout
        # decision and every "the recipe asked for 832x1216 and got this"
        # comparison reads. The decode has already happened.
        if raw.width and raw.height:
            conn.execute(
                "UPDATE file SET width = ?, height = ? WHERE id = ?",
                (raw.width, raw.height, file_id),
            )

    if parsed is None:
        return
    typed = GenerationParams.from_parsed(parsed)
    out.tool, out.detection = typed.tool, typed.detection

    workflow_id = None
    if raw is not None and raw.text.get("workflow"):
        graph = raw.text["workflow"]
        workflow_id = artifact(
            conn, "workflow", f"graph-{hashlib.sha256(graph.encode()).hexdigest()[:12]}",
            now, sha=hashlib.sha256(graph.encode()).hexdigest(),
        )
        out.artifacts.append(("workflow", "graph"))

    if typed.model:
        checkpoint = artifact(conn, "checkpoint", typed.model, now, quoted=typed.model_hash)
        conn.execute(
            "INSERT OR REPLACE INTO file_artifact(file_id, ordinal, artifact_id, role)"
            " VALUES(?, 0, ?, 'checkpoint')",
            (file_id, checkpoint),
        )
        out.artifacts.append(("checkpoint", typed.model))

    for ordinal, lora in enumerate(typed.loras):
        name = lora.get("name")
        if not name:
            continue
        lora_id = artifact(conn, "lora", name, now)
        conn.execute(
            "INSERT OR REPLACE INTO file_artifact(file_id, ordinal, artifact_id, role,"
            " model_weight, clip_weight) VALUES(?, ?, ?, 'lora', ?, ?)",
            (file_id, ordinal, lora_id, lora.get("weight"), lora.get("clip_weight")),
        )
        out.artifacts.append(("lora", str(name)))

    positive = prompt(conn, typed.positive_prompt, now)
    negative = prompt(conn, typed.negative_prompt, now)
    out.prompt_id = positive
    conn.execute(
        "INSERT OR REPLACE INTO generation(file_id, tool, detection, workflow_id, prompt_id,"
        " negative_id, seed, steps, cfg, denoise, clip_skip, sampler, scheduler,"
        " width, height, parser, parsed_at)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            file_id, typed.tool, typed.detection, workflow_id, positive, negative,
            typed.seed, typed.steps, typed.cfg, typed.denoise, typed.clip_skip,
            typed.sampler, typed.scheduler, typed.width, typed.height,
            "metaparse/1", now,
        ),
    )

    tail = dict(typed.extra)
    if typed.version:
        tail["version"] = typed.version
    for key, value in tail.items():
        if _param(conn, file_id, "generation", key, value):
            out.params += 1


def one(conn, file_id: int, path, now: float) -> Ingested:
    """Read one file completely: what made it, and how it was taken."""
    out = Ingested()
    generation(conn, file_id, path, now, out)

    # Retracted here rather than inside `store`, which returns early on a file
    # with no camera tags: a picture whose bytes were replaced by ones
    # carrying no EXIF has to lose the old readings, not keep them.
    retract(conn, file_id, "camera")
    conn.execute("DELETE FROM capture WHERE file_id = ?", (file_id,))

    found = capture_module.read(path)
    out.unreadable = found.unreadable
    if not found.is_empty:
        capture_module.store(conn, file_id, found, now, mint)
        out.captured = True
        out.params += len(found.params)
        out.unparsed += len(found.binaries)
        for kind, name in (("camera", found.camera), ("lens", found.lens)):
            if name:
                out.artifacts.append((kind, name))
    return out


def sidecar(conn, file_id: int, path, now: float) -> int:
    """A `.json` written beside a file, as fields rather than as a document.

    The slot is the sidecar's own filename, never the path it was read from.
    `file_blob.slot` is part of that table's primary key, so an absolute path
    there was identity derived from location -- inside the table whose whole
    point is that the payload outlives where it was found. Moving the library
    duplicated every sidecar row, and the old ones never went away.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return 0
    if not isinstance(document, dict):
        return 0
    retract(conn, file_id, "sidecar")
    _carrier(conn, file_id, "sidecar", os.path.basename(str(path)), json.dumps(document), now)
    return sum(1 for key, value in document.items() if _param(conn, file_id, "sidecar", key, value))
