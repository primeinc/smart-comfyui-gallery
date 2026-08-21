"""What this run has chosen, changeable while it runs.

Configuration is rows in `setting`, not environment variables: a value is
read where it is used, written over HTTP, and travels with the database.
An environment toggle is invisible to the application that obeys it --
nothing can list it, nothing can change it without a restart, and two
shells disagree about what is set. A row can do all three.

The registry below is the whole vocabulary. An unknown key is refused on
write, so a typo is an error at the moment somebody makes it, never a
setting that sits in the table and configures nothing.
"""

from __future__ import annotations

#: Every setting this application has, with its default and, where the
#: value is an enumeration, the choices. A None choice set means free text.
REGISTRY: dict[str, tuple[str, tuple[str, ...] | None]] = {
    # Where model weights are read from. Empty means `<home>/models`.
    "models_dir": ("", None),
    # ONNX Runtime execution providers for the recognition session:
    # "auto" (CUDA when the installed build offers it), "cpu", or an
    # explicit comma list of provider names.
    "ort_providers": ("auto", None),
    # Whether the vendored GPU faiss build may be used at all. Consulted
    # at the first faiss import in a process; changing it applies from
    # the next start.
    "faiss_gpu": ("on", ("on", "off")),
    # Whether jobs that already hold decoded pixels write the thumbnail
    # as a byproduct. Off, thumbnails are made only when first requested.
    "thumbnail_precache": ("on", ("on", "off")),
    # Whether the in-process worker drains jobs. Off, jobs queue until it
    # is turned back on -- they are rows, so nothing is lost by waiting.
    "worker": ("on", ("on", "off")),
    # Hamming bits (0..31) within which two phash64 values are the same
    # picture. Conservative by default: re-encodes and resizes land under
    # it, edits usually do not. Free text validated at submit; 32+ is
    # refused there because random pairs average 32 bits apart.
    "dupe_threshold": ("4", None),
    # The joint image/text model semantic search runs on, as
    # "<model>/<pretrained-checkpoint>" from open_clip.list_pretrained().
    # Changing it is a new immutable space: existing embeddings keep
    # their producer and the embed job fills the new space fresh.
    "semantic_model": ("ViT-B-32/laion2b_s34b_b79k", None),
    # The second opinion on every pHash dupe candidate: a pair also has
    # to agree within this many dHash bits, or "off" to trust pHash
    # alone. Two independent 64-bit fingerprints agreeing is much
    # stronger evidence than one -- pHash sees global low-frequency
    # composition, dHash local gradient structure, so a pair that is
    # pHash-close but dHash-far is similar composition over different
    # content. Permissive by default: it exists to veto gross
    # mismatches, not to out-vote pHash. Validated at submit, 0..63.
    "dupe_dhash_verify": ("16", None),
}


def value(conn, key: str) -> str:
    """The current value of one setting, defaulted from the registry."""
    default, _ = _known(key)
    row = conn.execute("SELECT value FROM setting WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def flag(conn, key: str) -> bool:
    """An on/off setting, as the boolean it is."""
    return value(conn, key) == "on"


def put(conn, key: str, new: str) -> None:
    """Change one setting, validated against the registry."""
    _, choices = _known(key)
    if choices is not None and new not in choices:
        raise ValueError(f"{key} must be one of {', '.join(choices)}, not {new!r}")
    conn.execute(
        "INSERT INTO setting(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, new),
    )


def snapshot(conn) -> list[dict]:
    """Every setting with its current value and default, for the page."""
    stored = dict(conn.execute("SELECT key, value FROM setting").fetchall())
    return [
        {"key": key, "value": stored.get(key, default), "default": default, "choices": choices}
        for key, (default, choices) in REGISTRY.items()
    ]


def _known(key: str) -> tuple[str, tuple[str, ...] | None]:
    entry = REGISTRY.get(key)
    if entry is None:
        raise KeyError(f"{key!r} is not a setting; the registry in db/settings.py is the vocabulary")
    return entry
