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
from dataclasses import dataclass, field

import metaparse
from metaparse.containers import load_raw
from metaparse.typed import GenerationParams

from . import capture as capture_module
from .scan import mint

#: PNG text chunks that carry a whole workflow graph rather than a value.
_GRAPH_SLOTS = ("workflow", "prompt")

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
    try:
        number = float(text)
    except ValueError:
        number = None
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
    conn.execute(
        "INSERT OR REPLACE INTO file_blob(file_id, carrier, slot, blob_hash, parsed_by, seen_at)"
        " VALUES(?, ?, ?, ?, ?, ?)",
        (file_id, carrier, slot, digest, parsed_by, now),
    )


def generation(conn, file_id: int, path, now: float, out: Ingested) -> None:
    """The recipe: tool, prompt, weights, sampler settings, and the long tail."""
    parsed = metaparse.parse_file(path)
    raw = load_raw(path)

    reader = f"metaparse/{parsed.tool}" if parsed is not None else None
    if raw is not None:
        for slot, value in raw.text.items():
            claimed = reader if slot in _GRAPH_SLOTS else None
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
    """A `.json` written beside a file, as fields rather than as a document."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return 0
    if not isinstance(document, dict):
        return 0
    _carrier(conn, file_id, "sidecar", str(path), json.dumps(document), now)
    return sum(1 for key, value in document.items() if _param(conn, file_id, "sidecar", key, value))
