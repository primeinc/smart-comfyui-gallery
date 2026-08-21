"""Retrieval over every participating semantic space, ranks fused.

This module owns what the adapter seam deliberately does not: which
spaces participate (the `semantic_model` setting, plural), the top-K
query against each space's resident index, the fusion of their rankings,
and the provenance of who said what. Adapters own their models;
db/similarity.py owns space identity and index alignment; this layer
composes them into one answer.

Two rules hold the composition honest:

- Ranks fuse, scores do not. An OpenCLIP cosine of .31 and another
  model's .31 are unrelated quantities from unrelated distributions --
  the separate immutable spaces already make them structurally
  incomparable, and Reciprocal Rank Fusion consumes only positions:
  score(d) = sum over spaces of 1 / (K + rank). Agreement between
  models rises, either model can still surface what the other missed,
  and a third model changes nothing here. Raw per-space scores ride
  along as provenance, never as inputs to the merge.

- Only CURRENT representations retrieve. Every candidate row must
  satisfy `source_sha256 = file.content_sha256`: a file replaced on
  disk stops being findable by its old picture the moment the scan
  records the new bytes, not whenever re-embedding gets around to it.
  (A file with no recorded content hash cannot vouch for any embedding
  and is likewise excluded -- the staleness doctrine of db/derived.py.)
"""

from __future__ import annotations

#: The RRF damping constant -- the field's customary default. Small K
#: weights top ranks harder; the choice matters little at gallery scale
#: and is deliberately not a setting until measurement says otherwise.
RRF_K = 60


def choices(conn) -> list[tuple[str, str, str]]:
    """The participating spaces, as (provider, model, checkpoint).

    `semantic_model` is a comma-separated list; each entry is
    `provider:model/checkpoint`, with a bare `model/checkpoint` meaning
    openclip. A malformed entry or unknown provider is a refused
    configuration, loudly -- never a silent substitution.
    """
    from vision import semantic

    from . import settings

    told = []
    for raw in settings.value(conn, "semantic_model").split(","):
        entry = raw.strip()
        if not entry:
            continue
        provider, colon, rest = entry.partition(":")
        if not colon:
            provider, rest = "openclip", entry
        model, slash, checkpoint = rest.partition("/")
        if not slash or not model or not checkpoint:
            raise ValueError(f"semantic_model entry must be '[provider:]<model>/<checkpoint>', not {entry!r}")
        semantic.provider_module(provider)  # unknown provider refused here
        told.append((provider, model, checkpoint))
    if not told:
        raise ValueError("semantic_model names no spaces")
    return told


def current_rows(conn, sid: int) -> list[tuple[int, int]]:
    """(embedding id, file id) for every CURRENT row of one space:
    present file, bytes unchanged since the vector was computed."""
    return [
        (int(row[0]), int(row[1]))
        for row in conn.execute(
            "SELECT e.id, e.file_id FROM derived_embedding e"
            " JOIN file f ON f.id = e.file_id AND f.missing_since IS NULL"
            " WHERE e.space_id = ? AND e.source_sha256 = f.content_sha256"
            " ORDER BY e.id",
            (sid,),
        )
    ]


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion over id rankings: positions in, one fused
    score per id out."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for position, member in enumerate(ranking):
            fused[member] = fused.get(member, 0.0) + 1.0 / (k + position + 1)
    return fused


def query(conn, models_dir: str, phrase: str, k: int, now: float, *, offline: bool = True) -> list[dict]:
    """One phrase against every participating space, rankings fused.

    Each configured space is searched independently -- its own encoder,
    its own resident index, its own top-K by its own cosine -- and the
    per-space rankings merge by RRF. The result rows carry the fused
    score AND each space's rank and raw score, because knowing which
    model earned its electricity is how the next model choice gets made.

    `offline=True` (the serving default) refuses to download model
    weights on the query path; provisioning belongs to /jobs/embed.
    Spaces with no current rows contribute nothing and are said to have
    contributed nothing, never guessed for.
    """
    from vision import semantic

    from . import similarity

    manager = similarity.manager_for(conn)
    per_space: list[tuple[str, list[tuple[int, float]]]] = []
    to_file: dict[int, int] = {}
    for provider, model, checkpoint in choices(conn):
        found = _space_of(conn, provider, model, checkpoint)
        if found is None:
            continue
        sid, spec = found
        rows = current_rows(conn, sid)
        if not rows:
            continue
        ids = [embedding_id for embedding_id, _ in rows]
        to_file.update(dict(rows))
        key = similarity.align(conn, manager, spec, ids, lambda wanted: _vectors(conn, wanted), now)
        encoder = semantic.encoder(provider, models_dir, model, checkpoint, offline=offline)
        labels, scores = manager.search(key, [encoder.encode_query(phrase)], min(max(int(k), 1), len(ids)))
        ranked = [
            (int(label), float(score)) for label, score in zip(labels[0], scores[0], strict=True) if int(label) != -1
        ]
        per_space.append((spec.key, ranked))

    # Fusion is over FILES: a file's agreement across spaces must
    # accumulate, and its embedding ids differ per space by design.
    fused = rrf([[to_file[embedding_id] for embedding_id, _ in ranked] for _, ranked in per_space])
    told: dict[int, dict] = {}
    for space_key, ranked in per_space:
        for position, (embedding_id, score) in enumerate(ranked):
            file_id = to_file[embedding_id]
            row = told.setdefault(file_id, {"file_id": file_id, "score": fused[file_id], "sources": {}})
            row["sources"][space_key] = {"rank": position + 1, "score": score}
    return sorted(told.values(), key=lambda row: row["score"], reverse=True)[: max(int(k), 1)]


def _space_of(conn, provider: str, model: str, checkpoint: str):
    from vision import semantic

    from . import similarity

    found = similarity._current_space_of(conn, semantic.space(provider, model, checkpoint, 1))
    if found is None:
        return None
    sid, dimensions = found
    return sid, semantic.space(provider, model, checkpoint, dimensions)


def _vectors(conn, wanted):
    """Embedding blobs by their immutable embedding ids, in order."""
    import numpy as np

    held = {}
    batch = [int(v) for v in wanted]
    for start in range(0, len(batch), 500):
        piece = batch[start : start + 500]
        marks = ",".join("?" for _ in piece)
        for embedding_id, blob in conn.execute(
            f"SELECT id, vector FROM derived_embedding WHERE id IN ({marks})", piece
        ):
            held[embedding_id] = np.frombuffer(blob, dtype=np.float32)
    return np.vstack([held[v] for v in batch])
