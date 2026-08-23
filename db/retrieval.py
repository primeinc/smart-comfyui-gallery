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
    `provider:<reference>`, with a bare reference meaning openclip. What
    a reference MEANS belongs to the provider -- open_clip entries are
    `model/pretrained-tag`, qwen entries a Hugging Face repo id -- so
    the provider module parses its own. A malformed entry or unknown
    provider is a refused configuration, loudly -- never a silent
    substitution.
    """
    from vision import semantic

    from . import settings

    told = []
    for raw in settings.value(conn, "semantic_model").split(","):
        entry = raw.strip()
        if not entry:
            continue
        provider, colon, reference = entry.partition(":")
        if not colon:
            provider, reference = "openclip", entry
        model, checkpoint = semantic.provider_module(provider).parse(reference)  # unknown provider refused here
        if (provider, model, checkpoint) not in told:
            # A repeated entry is one space, not a double vote: RRF sums
            # per-ranking contributions, so listing a model twice would
            # silently weight it 2x.
            told.append((provider, model, checkpoint))
    if not told:
        raise ValueError("semantic_model names no spaces")
    return told


def spelled(provider: str, model: str, checkpoint: str) -> str:
    """The exact name of one configured space: `provider:model@checkpoint`."""
    return f"{provider}:{model}@{checkpoint}"


def choice_for(conn, selector: str) -> tuple[str, str, str]:
    """ONE configured space by name -- exactly. A selector names a
    provider (`openclip`), a provider and model (`openclip:ViT-B-32`),
    or the full spelling (`openclip:ViT-B-32@laion2b_s34b_b79k`); it
    resolves when exactly one configured space matches and is REFUSED
    when several do, naming them -- first-match-wins would let a second
    configured model silently change what a request meant."""
    held = choices(conn)
    matched = [one for one in held if selector in (one[0], f"{one[0]}:{one[1]}", spelled(*one))]
    if len(matched) == 1:
        return matched[0]
    if not matched:
        known = ", ".join(spelled(*one) for one in held)
        raise ValueError(f"no configured semantic space matches {selector!r}; configured: {known}")
    raise ValueError(
        f"{selector!r} names {len(matched)} configured spaces; say which: "
        + ", ".join(spelled(*one) for one in matched)
    )


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


def query(
    conn, models_dir: str, phrase: str, k: int, now: float, *, offline: bool = True, allowed: set[int] | None = None
) -> dict:
    """One phrase against every participating space, rankings fused.

    Each configured space is searched independently -- its own encoder,
    its own resident index, its own candidates by its own cosine -- and
    the per-space rankings merge by RRF. Every space OVERFETCHES:
    fusion rewards agreement below any single space's top-k, so each
    space contributes max(k*4, 100) candidates and the cut to k happens
    only after the merge -- truncating per-space at k would make rank
    k+1 in one space invisible to a file's agreement everywhere else.

    `allowed` constrains the candidate set: only these file ids may
    answer. The constraint applies to EACH SPACE'S RANKING BEFORE the
    fusion -- out-of-scope candidates are discarded and the survivors
    renumbered 1..N -- because RRF consumes rank positions, and
    filtering a fused answer afterwards keeps every survivor's GLOBAL
    rank: two spaces whose out-of-scope candidates sit at different
    depths would compress by different amounts and the fused order can
    flip. "Search inside this album" means each model ranks the album,
    not the library. A constrained space is also searched at FULL depth
    rather than the overfetch heuristic: a scope survivor may sit
    arbitrarily deep in the unconstrained ranking.

    The answer says who was asked and who answered. `participants` is
    every configured space; `contributors` the ones whose ranking
    entered the fusion; `missing` maps the rest to why; `unmatched` the
    rankings that answered "nothing matches" (captions) -- because a
    result fused from one space when three are configured is a
    different claim than one all three agreed on, and a page that
    cannot tell them apart flattens the difference into false
    confidence. One unprovisioned model degrades the answer, never
    denies it; only NOTHING to answer from raises.

    Captions are one more ranking. Once the annotate job has said
    something about any file, the phrase's words are matched against
    every annotation (bm25) and that ranking enters the same fusion
    under the name `captions` -- a file a model described as "a red
    bicycle" answers "bicycle" whether or not an embedding agrees, and
    its agreement with the spaces lifts it. A library nothing has
    captioned lists no such participant: the channel is configured by
    having run the job.

    `offline=True` (the serving default) refuses to download model
    weights on the query path; provisioning belongs to /jobs/embed.
    """
    from vision import semantic

    from . import derived, similarity

    manager = similarity.manager_for(conn)
    final_k = max(int(k), 1)
    #: (ranking name, [(file_id, score)]) per ranking that enters the fusion
    per_space: list[tuple[str, list[tuple[int, float]]]] = []
    to_file: dict[int, int] = {}
    participants: list[str] = []
    missing: dict[str, str] = {}
    #: rankings that were asked and answered "nothing here" -- a word
    #: match with no matching word is the ordinary outcome, not a space
    #: that could not answer, and the page must not call it degraded
    unmatched: dict[str, str] = {}
    for provider, model, configured in choices(conn):
        # A mutable checkpoint (a Hugging Face branch) resolves to the
        # cached immutable commit BEFORE anything is keyed by it: the
        # registry rows were minted by the embed path post-pin, so an
        # unpinned probe would look for a space that never existed.
        checkpoint = semantic.pin(provider, models_dir, model, configured)
        name = semantic.space(provider, model, checkpoint, 1).key
        participants.append(name)
        found = _space_of(conn, provider, model, checkpoint)
        if found is None:
            missing[name] = "no embeddings recorded; run /jobs/embed"
            continue
        sid, spec = found
        rows = current_rows(conn, sid)
        if not rows:
            missing[name] = "no current embeddings: every recorded vector is stale or its file is gone"
            continue
        try:
            encoder = semantic.encoder(provider, models_dir, model, checkpoint, offline=offline)
        except LookupError as unprovisioned:
            missing[name] = str(unprovisioned)
            continue
        declared = encoder.space() if hasattr(encoder, "space") else None
        answers = getattr(declared, "dimensions", None)
        if answers is not None and int(answers) != int(spec.dimensions):
            # Rows recorded under one width cannot be searched with a query
            # of another: the index would assert, not answer. The rows are
            # another build's; a re-embed under this encoder replaces them.
            missing[name] = (
                f"the recorded space is {spec.dimensions}-dimensional and this encoder answers in {answers};"
                " its vectors were made by a different build -- run /jobs/embed to replace them"
            )
            continue
        ids = [embedding_id for embedding_id, _ in rows]
        to_file.update(dict(rows))
        key = similarity.align(conn, manager, spec, ids, lambda wanted: _vectors(conn, wanted), now)
        candidate_k = len(ids) if allowed is not None else min(max(final_k * 4, 100), len(ids))
        labels, scores = manager.search(key, [encoder.encode_query(phrase)], candidate_k)
        ranked = [
            (int(label), float(score)) for label, score in zip(labels[0], scores[0], strict=True) if int(label) != -1
        ]
        if allowed is not None:
            # Discarding here, before enumeration, IS the renumbering:
            # rrf() reads positions off the filtered list.
            ranked = [(embedding_id, score) for embedding_id, score in ranked if to_file[embedding_id] in allowed]
        if not ranked:
            # An empty ranking entered no fusion: a space whose current
            # embeddings all sit outside the scope must not be reported
            # as having agreed with anything.
            missing[name] = "no current embeddings in this scope"
            continue
        per_space.append((spec.key, [(to_file[embedding_id], score) for embedding_id, score in ranked]))

    if derived.any_annotations(conn):
        participants.append(derived.CAPTIONS)
        depth = len(allowed) if allowed is not None else max(final_k * 4, 100)
        worded = derived.rank_by_annotation(conn, phrase, depth, allowed=allowed)
        if worded:
            per_space.append((derived.CAPTIONS, worded))
        else:
            unmatched[derived.CAPTIONS] = "no caption mentions a word of the phrase in this scope"

    if not per_space and any("not provisioned" in why or "-dimensional" in why for why in missing.values()):
        # Degraded is an answer; NOTHING to answer from is a refusal that
        # must name its fix, exactly as the single-space case always did.
        raise LookupError("; ".join(f"{name}: {why}" for name, why in sorted(missing.items())))

    # Fusion is over FILES: a file's agreement across rankings must
    # accumulate, and its embedding ids differ per space by design.
    fused = rrf([[file_id for file_id, _ in ranked] for _, ranked in per_space])
    told: dict[int, dict] = {}
    for space_key, ranked in per_space:
        for position, (file_id, score) in enumerate(ranked):
            row = told.setdefault(file_id, {"file_id": file_id, "score": fused[file_id], "sources": {}})
            row["sources"][space_key] = {"rank": position + 1, "score": score}
    results = sorted(told.values(), key=lambda row: row["score"], reverse=True)[:final_k]
    return {
        "results": results,
        "participants": participants,
        "contributors": [space_key for space_key, _ in per_space],
        "missing": missing,
        "unmatched": unmatched,
    }


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
