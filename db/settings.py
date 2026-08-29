"""What this run has chosen, changeable while it runs.

Configuration is rows in `setting`, not environment variables: a value is
read where it is used, written over HTTP, and travels with the database.
An environment toggle is invisible to the application that obeys it --
nothing can list it, nothing can change it without a restart, and two
shells disagree about what is set. A row can do all three.

The registry below is the whole vocabulary. An unknown key is refused on
write, so a typo is an error at the moment somebody makes it, never a
setting that sits in the table and configures nothing.

Four entries carry more than a line of contract.

``WheelModifier`` names the held key that turns the viewer's wheel into a step
through the walk. `ctrl` is not offered because a trackpad pinch reaches the
page as a wheel event with `ctrlKey` set (MDN, Element: wheel event;
MouseEvent.ctrlKey), and `shift` is not offered because a browser reads
shift+wheel as horizontal scroll. The unchosen modifiers keep whatever the
browser does with them, because the viewer calls preventDefault only for the
gestures it acts on (frontend/src/viewer.ts). It is spelled as a Literal so the
browser is typed against the same closed set the registry validates writes with.

``semantic_model`` is a comma list of "[provider:]<reference>", each reference
in its provider's own grammar. A bare entry is openclip, a
"<model>/<pretrained-tag>" pair from open_clip.list_pretrained();
"qwen:<org>/<repo>[@revision]" names a retrieval-trained Qwen3-VL embedding
checkpoint by its Hugging Face repo id (Qwen/Qwen3-VL-Embedding-2B), never the
-Instruct chat family, which shares the backbone but not the training. Every
entry is its own immutable space, searched independently with rankings fused by
rank, so changing an entry is a new space: existing embeddings keep their
producer and the embed job fills the new space fresh.

``dupe_dhash_verify`` is the second opinion on every pHash dupe candidate: a
pair also has to agree within this many dHash bits, or "off" to trust pHash
alone. Two independent 64-bit fingerprints agreeing is much stronger evidence
than one -- pHash sees global low-frequency composition, dHash local gradient
structure -- so a pair that is pHash-close but dHash-far is similar composition
over different content. Over 2,141 labelled pairs from real libraries
(benchmarks/fingerprint_calibration.py ->
benchmarks/results/fingerprint_calibration.json) every true duplicate agrees
within 4 dHash bits, the verifier vetoed no positive at any cutoff, and 8 vetoes
144 of the 183 false positives pHash proposes at radius 16. Validated at submit,
0..63.

``face_cluster_threshold`` is the cosine similarity at which two faces are taken
to be the same person. "auto" is the measured per-embedder operating point
(db/derived.py SAME_PERSON); the spaces are not comparable and one number is
wrong for all but one of them, with docs/FACE_CLUSTERING.md:71-74 putting
SFace's 0.363 applied to ArcFace at a top-cluster share of 0.963. The threshold
is part of a run's identity (schema.sql derived_face_run_identity), so
clustering at a new one writes a new run beside the old rather than over it and
the other grouping can be made primary again. It is read at submit and pinned
into the payload, so changing it mid-job cannot give two embedding spaces two
different answers in one run.
"""

from __future__ import annotations

import typing

#: Whether a held Alt turns the viewer's wheel into a step through the walk
#: rather than a zoom; "none" leaves the wheel to zoom alone. The module
#: docstring states which modifiers are not offered, and why.
WheelModifier = typing.Literal["alt", "none"]
WHEEL_MODIFIERS: tuple[WheelModifier, ...] = typing.get_args(WheelModifier)

#: Every setting this application has, with its default and, where the
#: value is an enumeration, the choices. A None choice set means free text.
REGISTRY: dict[str, tuple[str, tuple[str, ...] | None]] = {
    # Where model weights are read from. Empty means `<home>/models`.
    "models_dir": ("", None),
    # Which face pipeline the faces job runs (vision/faces.py backend_for);
    # "auto" takes insightface's antelopev2 pack, falling back to the OpenCV
    # YuNet+SFace stack when absent. Switching starts a fresh embedding space.
    "face_backend": ("auto", ("auto", "insightface", "opencv")),
    # ONNX Runtime execution providers for the insightface recognition
    # session: "auto" (CUDA when the installed build offers it), "cpu",
    # or an explicit comma list of provider names. Read at job submit.
    "ort_providers": ("auto", None),
    # The BLIP captioning checkpoint the annotate job runs, as a Hugging
    # Face repository id (vision/captions.py). Read at job submit; each
    # model's captions are kept beside the others', never merged.
    "caption_model": ("Salesforce/blip-image-captioning-base", None),
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
    # Hamming bits (0..31) within which two phash64 values are the same picture;
    # re-encodes and resizes land under the default, edits do not. Free text
    # validated at submit, where 32+ is refused: random pairs average 32 apart.
    "dupe_threshold": ("4", None),
    # The joint image/text models semantic search runs on, a comma list of
    # "[provider:]<reference>"; the module docstring holds the grammar and
    # what changing an entry means.
    "semantic_model": ("ViT-B-32/laion2b_s34b_b79k", None),
    # The second opinion on every pHash dupe candidate, in dHash bits, or "off"
    # to trust pHash alone. The module docstring carries the calibration and
    # its artifact (benchmarks/results/fingerprint_calibration.json).
    "dupe_dhash_verify": ("8", None),
    # The cosine similarity at which two faces are the same person; "auto" is
    # the measured per-embedder point (db/derived.py SAME_PERSON). The module
    # docstring holds the run-identity and submit-pinning rules.
    "face_cluster_threshold": ("auto", None),
    # Which held key makes the viewer's wheel walk to the next picture instead
    # of zooming (WheelModifier above). Read when a media page or its overlay
    # fragment is rendered, so a change applies to the next picture opened.
    "viewer_wheel_modifier": ("alt", WHEEL_MODIFIERS),
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
