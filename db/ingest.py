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
import logging
import math
import os
from dataclasses import dataclass, field

import metaparse
from metaparse.containers import load_raw
from metaparse.typed import GenerationParams

from . import capture as capture_module
from . import graph as graph_module
from . import probe as probe_module
from . import prompts as prompts_module
from .scan import mint

_logger = logging.getLogger(__name__)

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
            "checkpoint",
            "refiner",
            "lora",
            "vae",
            "controlnet",
            "upscaler",
            "embedding",
            "hypernetwork",
            "ip_adapter",
            "text_encoder",
            "unet",
        ),
        "carriers": ("png_text", "xmp"),
        "relations": ("generation_prompt",),
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
    for table in owned.get("relations", ()):
        conn.execute(f"DELETE FROM {table} WHERE file_id = ?", (file_id,))
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
    #: Set when the container answered: this file knows its own size, and its
    #: length if it has one.
    probed: bool = False
    #: Set when the bytes could not be opened at all, so a caller can report
    #: the file rather than reading "no metadata" off it.
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
        "SELECT id FROM artifact WHERE kind = ? AND name_key = ? AND IFNULL(content_sha256, '') = IFNULL(?, '')",
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
    # Seeded from the words, because a prompt has something to read and the
    # address is where that shows. `prompt-558a568843` is the hash wearing a
    # slash -- the shape the plan retires the `💬 #C89B1` badge for, put back
    # by the only entity that had no name to seed from and did have text.
    # Six words is enough to recognise one and short enough to be a URL; the
    # collision suffix in `mint` handles two prompts that open alike.
    prompt_id = mint(conn, "prompt", " ".join(text.split()[:6]) or f"prompt-{digest[:10]}")
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


def _fill_from_graph(typed, recipe) -> None:
    """Put the graph's readings into the blanks the text parser left.

    Blanks only. A workflow that also wrote an A1111-compatible `parameters`
    chunk has already been read from it, and that text is what the tool
    itself reported it did -- the graph is what it was asked to do. Where
    both speak, the report wins; the graph fills the silence.
    """
    for name in ("seed", "steps", "cfg", "denoise", "sampler", "scheduler", "width", "height"):
        if getattr(typed, name, None) is None:
            setattr(typed, name, getattr(recipe, name))
    if not typed.model and recipe.model:
        typed.model = recipe.model
    if not typed.positive_prompt:
        typed.positive_prompt = recipe.positive
    if not typed.negative_prompt:
        typed.negative_prompt = recipe.negative
    if not typed.loras and recipe.loras:
        typed.loras = [
            {"name": name, "weight": model_weight, "clip_weight": clip_weight}
            for name, model_weight, clip_weight in recipe.loras
        ]


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
            understood = slot in _GRAPH_SLOTS or (bool(consumed) and str(value).strip() == consumed)
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
            ("Format", raw.format),
            ("Mode", raw.mode),
            ("Width", raw.width),
            ("Height", raw.height),
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

    # ComfyUI writes its graphs as PNG text chunks -- Comfy-Org/ComfyUI@
    # a9ab2b6 nodes.py:1701-1706 add_text("prompt", ...) plus extra_pnginfo
    # ("workflow") -- and metaparse reads text, so its adapter only names
    # the tool. Every ComfyUI picture therefore arrived with a tool and
    # nothing else: no seed, steps, cfg, sampler, checkpoint or LoRA rows.
    # The graph is right there in the chunk; db/graph.py reads it.
    if raw is not None:
        recipe = graph_module.read(raw.text.get("prompt") or raw.text.get("workflow") or "")
        if recipe is not None:
            _fill_from_graph(typed, recipe)
            # 'graph' says where the recipe came from, which is the whole
            # point of recording detection: a value read out of a node graph
            # is not the same claim as one a tool wrote down about itself.
            typed.detection = out.detection = "graph"

    workflow_id = None
    if raw is not None and raw.text.get("workflow"):
        graph = raw.text["workflow"]
        workflow_id = artifact(
            conn,
            "workflow",
            f"graph-{hashlib.sha256(graph.encode()).hexdigest()[:12]}",
            now,
            sha=hashlib.sha256(graph.encode()).hexdigest(),
        )
        out.artifacts.append(("workflow", "graph"))

    # What the tool said it loaded, with the role it loaded it into and the
    # hash it recorded. SwarmUI writes a whole manifest -- a sha256 per weight
    # file -- and it used to be joined into one comma-separated string under
    # a `used models` parameter, which lost the role, lost every hash, and
    # made a checkpoint and its LoRA the same kind of nothing. A hash is what
    # lets a weight file in this library be the same weight file as the one on
    # disk, so throwing it away is throwing away the join.
    stated = {}
    for entry in getattr(typed, "artifacts", ()) or ():
        role = entry.get("role")
        name = str(entry.get("name") or "").strip()
        if not role or not name:
            continue
        digest = str(entry.get("hash") or "").strip() or None
        # The role says how the weights were USED; the artifact row says
        # what they ARE. A refiner is checkpoint weights in the refiner
        # role -- the same file used as base elsewhere is one artifact,
        # two roles -- and the role-match trigger states this mapping.
        artifact_id = artifact(conn, "checkpoint" if role == "refiner" else role, name, now, quoted=digest)
        ordinal = stated.get(role, 0)
        stated[role] = ordinal + 1
        weight = _as_number(str(entry.get("weight"))) if entry.get("weight") is not None else None
        conn.execute(
            "INSERT OR REPLACE INTO file_artifact(file_id, ordinal, artifact_id, role,"
            " model_weight, clip_weight) VALUES(?, ?, ?, ?, ?, ?)",
            (file_id, ordinal, artifact_id, role, weight, weight),
        )
        out.artifacts.append((role, name))

    if typed.model and "checkpoint" not in stated:
        checkpoint = artifact(conn, "checkpoint", typed.model, now, quoted=typed.model_hash)
        conn.execute(
            "INSERT OR REPLACE INTO file_artifact(file_id, ordinal, artifact_id, role) VALUES(?, 0, ?, 'checkpoint')",
            (file_id, checkpoint),
        )
        out.artifacts.append(("checkpoint", typed.model))

    for ordinal, lora in enumerate(typed.loras):
        name = lora.get("name")
        if not name or "lora" in stated:
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
        "INSERT OR REPLACE INTO generation(file_id, tool, detection, workflow_id,"
        " seed, steps, cfg, denoise, clip_skip, sampler, scheduler,"
        " width, height, parser, parsed_at)"
        " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            file_id,
            typed.tool,
            typed.detection,
            workflow_id,
            typed.seed,
            typed.steps,
            typed.cfg,
            typed.denoise,
            typed.clip_skip,
            typed.sampler,
            typed.scheduler,
            typed.width,
            typed.height,
            "metaparse/1",
            now,
        ),
    )
    # The roles. A prompt AS WRITTEN is the generator's own
    # `original_<param>` (db/prompts.py ORIGINAL_ROLES), interned through
    # the same dedupe as the prompt it ran; the parameter stays in
    # file_param below as the evidence. No parameter, no row -- silence
    # is not a claim that written == ran.
    prompts_module.assign(conn, file_id, "effective", positive)
    prompts_module.assign(conn, file_id, "negative", negative)
    prompts_module.assign(conn, file_id, "unsampler", prompt(conn, str(typed.extra.get("unsamplerprompt") or ""), now))
    for key, role in prompts_module.ORIGINAL_ROLES.items():
        prompts_module.assign(conn, file_id, role, prompt(conn, str(typed.extra.get(key) or ""), now))
    # The sections, read with this tool's grammar (pure, microseconds):
    # every main-section text is a prompt row from the moment the file
    # is read, so a planner or a search finds a vector to hang on it.
    for held in prompts_module.roles(conn, file_id).values():
        prompts_module.sections(conn, held["id"], prompts_module.grammar_for(typed.tool), now)

    tail = dict(typed.extra)
    if typed.version:
        tail["version"] = typed.version
    for key, value in tail.items():
        if _param(conn, file_id, "generation", key, value):
            out.params += 1


#: Kinds Pillow cannot open, so their size and length come from the container
#: rather than from a decode.
_PROBED = ("video", "animated_image", "audio")


#: Formats whose byte-identical suffix covers both still and moving files
#: (immich-app/immich@f88fb62 server/src/utils/mime-types.ts:72).
_POSSIBLY_ANIMATED = {
    ".png",
    ".apng",
    ".gif",
    ".webp",
    ".avif",
    ".heic",
    ".heif",
    ".heics",
    ".heifs",
    ".jxl",
}

#: What each pipeline kind is called by the sniffer's family word.
_FAMILY = {"image": "image", "animated_image": "image", "video": "video", "audio": "audio", "document": "document"}

#: Sniffs that cannot overrule the suffix's kind. An MJPEG stream opens
#: with a complete JPEG -- its first frame -- so a JPEG signature against a
#: claimed video proves nothing either way.
_INCONCLUSIVE = {("jpeg", "video")}


def _really_animated(path) -> bool | None:
    """Whether the decoded picture moves; None when it will not decode.

    Suffixes that cannot animate are not opened at all -- a JPEG never
    moves, and opening every one to learn that would put a decode on the
    scan hot path for nothing.
    """
    import pathlib

    if pathlib.Path(path).suffix.lower() not in _POSSIBLY_ANIMATED:
        return None
    from vision import decode

    try:
        with decode.open_still(path) as image:  # the handle closes with the image, not at GC
            return decode.is_animated(image)
    except (OSError, ValueError) as why:
        _logger.warning("%s: cannot tell whether it animates: %s: %s", path, type(why).__name__, why)
        return None


def one(conn, file_id: int, path, now: float, *, kind: str | None = None) -> Ingested:
    """Read one file completely: what made it, and how it was taken.

    `kind` says which reader applies. Without it every file went through the
    image path alone, so a video had no length and no dimensions at all --
    `file.duration` had no producer in the whole package and the DDL said so
    rather than fixing it.
    """
    # The Pillow plugins register process-wide, and metaparse's container
    # reader opens files with plain Image.open -- a HEIC that arrives before
    # any decode call would otherwise be unreadable to it.
    from vision import decode as _decode
    from vision import sniff as sniff_module

    from .runner import report

    # Imported here rather than at module scope: the runner imports this
    # module inside its handler, and a link back at import time would
    # close the loop. Outside a job `report()` is the silent one, so this
    # costs nothing and needs no branch.
    told = report()

    _decode.ensure_decoders()

    out = Ingested()
    if kind is None:
        kind = conn.execute("SELECT kind FROM file WHERE id = ?", (file_id,)).fetchone()[0]
    kind = str(kind)

    # The suffix proposed this kind; the bytes get the casting vote. A
    # library accumulates liars -- an MP4 exported as .jpg, a HEIC renamed
    # on share -- and routing them by suffix feeds the wrong reader, then
    # records "unreadable" about a perfectly good file. The sniff is a
    # 512-byte read (vision/sniff.py, patterns from whatwg/mimesniff@39aa535);
    # the reader that follows is the proof.
    suffix_claimed = None
    told.phase("sniffing")
    sniffed = sniff_module.sniff_path(path)
    if sniffed is not None:
        family, token = sniffed
        if (
            family != _FAMILY.get(kind, kind)
            and family in ("image", "video", "audio", "document")
            and (token, kind) not in _INCONCLUSIVE
        ):
            suffix_claimed = kind
            kind = "image" if family == "image" else family
            conn.execute("UPDATE file SET kind = ? WHERE id = ?", (kind, file_id))

    # Named on its own because it is the step that reads what MADE the
    # picture -- a generator's whole workflow, embedded in the file's own
    # text chunks -- and that is a parse whose cost tracks the size of the
    # recipe rather than the size of the image. Without the name it was
    # billed to `sniffing`, which is a 512-byte read and could not
    # honestly have cost 48 ms.
    told.phase("reading-generation", kind=kind)
    generation(conn, file_id, path, now, out)

    # Written AFTER generation(), which retracts the whole 'container'
    # source before re-parsing -- facts written before it were deleted by it.
    if sniffed is not None:
        _param(conn, file_id, "container", "SniffedFormat", sniffed[1])
    if suffix_claimed is not None:
        _param(conn, file_id, "container", "SuffixClaimed", suffix_claimed)

    told.phase("probing", kind=kind)
    if kind in _PROBED:
        container = probe_module.read(path)
        probe_module.store(conn, file_id, container, now)
        out.params += len(container.params)
        out.probed = container.duration is not None or container.width is not None
        out.unreadable = container.unreadable
    elif kind == "document":
        # A document says how long it is in pages, and gets one sample per
        # page so a caption or a piece of OCR has somewhere to point. It was
        # the last kind that could not state its own length.
        opened = probe_module.document(path)
        probe_module.store(conn, file_id, opened, now)
        out.params += len(opened.params)
        out.probed = not opened.is_empty
        out.unreadable = opened.unreadable
        probe_module.pages_of(conn, file_id, opened)

    # Retracted here rather than inside `store`, which returns early on a file
    # with no camera tags: a picture whose bytes were replaced by ones
    # carrying no EXIF has to lose the old readings, not keep them.
    retract(conn, file_id, "camera")
    conn.execute("DELETE FROM capture WHERE file_id = ?", (file_id,))

    # Only where a camera could have written any. Running it on a video or a
    # PDF meant `unreadable` reported "Pillow cannot open this" for every one
    # of them -- true, uninteresting, and it buried the message from the
    # reader that could.
    if kind in ("image", "animated_image"):
        # The suffix guessed; the decoded file answers. An animated WebP,
        # AVIF or PNG wears a still suffix, and a single-frame GIF wears an
        # animated one -- n_frames is the fact, so the row records it.
        moving = _really_animated(path)
        if moving is not None and moving != (kind == "animated_image"):
            kind = "animated_image" if moving else "image"
            conn.execute("UPDATE file SET kind = ? WHERE id = ?", (kind, file_id))
    if kind in ("image", "animated_image", "video"):
        found = capture_module.read(path) if kind != "video" else capture_module.read_video(path)
        # First complaint stands: an animated image was already probed
        # above, and the capture read's silence must not erase why the
        # container reader could not read it -- duration would stay NULL
        # with nothing saying why.
        out.unreadable = out.unreadable or found.unreadable
        if found.orientation in capture_module.TRANSPOSED:
            # The decode reports the stored frame; the tag says to turn it a
            # quarter. Storing the stored size files every portrait photograph
            # in the library as landscape -- the same defect the video probe
            # handles for a display matrix, and it needs the same answer here.
            # SQLite reads both sides from the row as it was, so this really
            # is a swap.
            conn.execute("UPDATE file SET width = height, height = width WHERE id = ?", (file_id,))
        if not found.is_empty:
            capture_module.store(conn, file_id, found, now, mint)
            out.captured = True
            out.params += len(found.params)
            out.unparsed += len(found.binaries)
            for maker, name in (("camera", found.camera), ("lens", found.lens)):
                if name:
                    out.artifacts.append((maker, name))
    # The source claims about this file just changed hands: its derived
    # interpretation is no longer knowably true (db/context.py stale).
    from . import context as context_module

    context_module.stale(conn, file_id)
    if out.unreadable is None:
        # the record of the read: these bytes were read whole, metadata or
        # none -- the caller raises on an unreadable file and the runner
        # rolls this back with the rest
        from . import scan as scan_module

        sha = conn.execute("SELECT content_sha256 FROM file WHERE id = ?", (file_id,)).fetchone()[0]
        if sha is None:
            sha = scan_module.sha256_of(path)
            conn.execute("UPDATE file SET content_sha256 = ? WHERE id = ?", (sha, file_id))
        conn.execute("UPDATE file SET ingested_sha256 = ? WHERE id = ?", (sha, file_id))
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
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError) as why:
        _logger.warning("%s: sidecar unreadable: %s: %s", path, type(why).__name__, why)
        return 0
    if not isinstance(document, dict):
        _logger.warning("%s: sidecar is a %s, not an object", path, type(document).__name__)
        return 0
    retract(conn, file_id, "sidecar")
    _carrier(conn, file_id, "sidecar", os.path.basename(str(path)), json.dumps(document), now)
    return sum(1 for key, value in document.items() if _param(conn, file_id, "sidecar", key, value))
