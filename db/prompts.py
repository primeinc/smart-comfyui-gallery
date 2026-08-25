"""Prompts as a reusable semantic substrate.

A prompt is an entity (db/schema.sql `prompt`): one row per distinct
non-empty text, FTS-indexed, addressable. This module owns what sits on
top of that identity:

- ROLES. `generation_prompt` records which prompt played which role for
  one generation. SwarmUI processes exactly two prompt-like params,
  `prompt` and `negativeprompt`, and records `original_<param>` when
  processing changed the text, removing it again when it did not
  (refs/mcmonkeyprojects/SwarmUI src/Text2Image/T2IParamInput.cs:376-382,
  :592). So the roles are `effective`, `negative`, `original`,
  `original_negative`, and `unsampler` for its third prompt field. An
  absent `original_*` is recorded as NOTHING: metadata is optional,
  partial, or from another version, and silence is not evidence that
  written == ran. The raw parameter stays in `file_param` as the
  evidence a role was read from.

- SECTIONS. A prompt text is a document of sections (db/prompt_sections.py,
  a tool-neutral IR): the main prompt, then each `<segment:>`,
  `<region:>`, `<refiner>`, `<video>` ... section. The grammar is chosen
  by the generation's TOOL. `derived_prompt_section` holds the parse
  per (prompt, grammar) and points each section's text at an ORDINARY
  interned prompt row -- "a red fox" as a main prompt, inside a region,
  or as a negative is one text identity, parsed into context many times.

- VECTORS. `derived_prompt_embedding` holds one vector per (prompt text,
  space, query policy). `space_id` is the provider's JOINT space -- the
  coordinate system its media vectors live in, because `encode_query`
  produces vectors in that same space (vision/semantic) and retrieval
  already searches media with them; a stored prompt vector may be
  compared with a media vector of the same space. `policy_hash` is the
  QUERY policy that produced it (instruction, tokenizer, package):
  provenance and currentness, so a changed instruction is a new row
  that coexists with the old. Prompt rows never enter the media index:
  they have their own resident index per (space, policy) -- same
  coordinates, different corpus, no id collisions, no mixing of
  vectors two instructions produced.

- CONSUMERS. The StoryPlanner's similarity Seam stays value-in/
  vector-out and connection-free; `cached(...)` wraps a loaded engine so
  its frozen texts are looked up BY TEXT HASH under (space, policy) --
  never by file id, generation id, or today's role relation -- and
  computed once where missing. `neighbours(...)` returns "prompts like
  this one" inside ONE (space, policy); a role or section-kind filter
  constrains the CANDIDATES before ranking, at full depth, never the
  top-k afterwards.
"""

from __future__ import annotations

import hashlib

from . import prompt_sections

ROLES = ("effective", "original", "negative", "original_negative", "unsampler")

#: Swarm parameter id -> role, and the `original_<param>` each may carry.
PARAM_ROLES = {"prompt": "effective", "negativeprompt": "negative", "unsamplerprompt": "unsampler"}
ORIGINAL_ROLES = {"original_prompt": "original", "original_negativeprompt": "original_negative"}


def text_hash(text: str) -> str:
    """The identity of a text -- the same hash `prompt.text_hash` carries
    (db/ingest.py prompt), over the exact string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def grammar_for(tool: str | None) -> str:
    """Which section grammar a generation's prompts are read with,
    decided by the tool that wrote them -- never by the text."""
    return "swarm" if (tool or "").lower().startswith("swarm") else "plain"


def lane(policy: str) -> str:
    """The resident-index lane for prompt vectors under one query
    policy, beside the media lane of the same space."""
    return f"prompts+{policy}"


def roles(conn, file_id: int) -> dict[str, dict]:
    """Role -> {id, uuid, text, text_hash} for one generation."""
    held = {}
    for row in conn.execute(
        "SELECT gp.role, p.id, e.uuid, p.text, p.text_hash FROM generation_prompt gp"
        " JOIN prompt p ON p.id = gp.prompt_id JOIN entity e ON e.id = p.id WHERE gp.file_id = ?",
        (file_id,),
    ):
        held[row[0]] = {"id": int(row[1]), "uuid": row[2].hex(), "text": row[3], "text_hash": row[4]}
    return held


def assign(conn, file_id: int, role: str, prompt_id: int | None) -> None:
    """One role for one generation: the row when there is a prompt,
    nothing when there is not."""
    if role not in ROLES:
        raise ValueError(f"no prompt role named {role!r}; one of {', '.join(ROLES)}")
    conn.execute("DELETE FROM generation_prompt WHERE file_id = ? AND role = ?", (file_id, role))
    if prompt_id is not None:
        conn.execute(
            "INSERT INTO generation_prompt(file_id, role, prompt_id) VALUES(?, ?, ?)", (file_id, role, prompt_id)
        )


# --- sections ---------------------------------------------------------------------


def sections(conn, prompt_id: int, grammar: str, now: float) -> list[tuple[prompt_sections.Section, int | None]]:
    """The CURRENT sections of one prompt under one grammar, each with
    the prompt row its text is interned as (None for an empty section)
    -- parsed now when missing, written by an older parser, or read
    from older text. A re-parse replaces the rows; the texts' vectors
    are untouched, because they belong to the texts."""
    from .ingest import prompt as intern

    row = conn.execute("SELECT text, text_hash FROM prompt WHERE id = ?", (prompt_id,)).fetchone()
    if row is None:
        raise LookupError(f"no prompt {prompt_id}")
    text, digest = row
    held = conn.execute(
        "SELECT ordinal, kind, spec, text, text_prompt_id FROM derived_prompt_section"
        " WHERE prompt_id = ? AND grammar = ? AND source_text_hash = ? AND parser_version = ? ORDER BY ordinal",
        (prompt_id, grammar, digest, prompt_sections.VERSION),
    ).fetchall()
    if held:
        return [(prompt_sections.Section(int(r[0]), r[1], r[2], r[3]), r[4]) for r in held]
    conn.execute("DELETE FROM derived_prompt_section WHERE prompt_id = ? AND grammar = ?", (prompt_id, grammar))
    made = []
    for section in prompt_sections.parse(text, grammar):
        # the section's text is a prompt like any other: the SAME row when
        # the whole prompt is its own main section
        text_id = prompt_id if section.text == text else intern(conn, section.text, now)
        conn.execute(
            "INSERT INTO derived_prompt_section(prompt_id, grammar, ordinal, kind, spec, text, text_prompt_id,"
            " source_text_hash, parser_version, computed_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                prompt_id,
                grammar,
                section.ordinal,
                section.kind,
                section.spec,
                section.text,
                text_id,
                digest,
                prompt_sections.VERSION,
                now,
            ),
        )
        made.append((section, text_id))
    return made


def grammars_of(conn, prompt_id: int) -> list[str]:
    """The grammars this prompt is read with: one per distinct tool of
    the generations it plays a role in."""
    tools = conn.execute(
        "SELECT DISTINCT g.tool FROM generation_prompt gp JOIN generation g ON g.file_id = gp.file_id"
        " WHERE gp.prompt_id = ?",
        (prompt_id,),
    ).fetchall()
    return sorted({grammar_for(row[0]) for row in tools}) or ["plain"]


# --- the space and the policy --------------------------------------------------------


def space_of(conn, provider: str, model: str, checkpoint: str):
    """The provider's CURRENT joint space from the registry -- (space id,
    full spec) -- or None when nothing has minted it yet."""
    from vision import semantic

    from . import similarity

    found = similarity._current_space_of(conn, semantic.space(provider, model, checkpoint, 1))
    if found is None:
        return None
    sid, dimensions = found
    return sid, semantic.space(provider, model, checkpoint, dimensions)


def current_vectors(conn, sid: int, policy: str, hashes: list[str]) -> dict[str, list[float]]:
    """Vectors keyed by the hash of the exact text they were computed
    from, current under one (space, policy): the row's source hash must
    still be its prompt's own. Nothing here knows a file, a generation,
    a section, or a role."""
    import numpy as np

    held: dict[str, list[float]] = {}
    wanted = sorted(set(hashes))
    for start in range(0, len(wanted), 500):
        piece = wanted[start : start + 500]
        marks = ",".join("?" for _ in piece)
        for digest, blob in conn.execute(
            "SELECT e.source_text_hash, e.vector FROM derived_prompt_embedding e"
            " JOIN prompt p ON p.id = e.prompt_id AND e.source_text_hash = p.text_hash"
            f" WHERE e.space_id = ? AND e.policy_hash = ? AND e.source_text_hash IN ({marks})",
            (sid, policy, *piece),
        ):
            held[digest] = [float(x) for x in np.frombuffer(blob, dtype=np.float32)]
    return held


def remember(conn, spec, policy: str, digest: str, vector, now: float) -> bool:
    """Persist a vector for the prompt row holding this text hash, if one
    exists. A frozen text no prompt holds has nothing to hang a vector
    on and is simply not remembered."""
    from . import derived

    row = conn.execute("SELECT id FROM prompt WHERE text_hash = ?", (digest,)).fetchone()
    if row is None:
        return False
    derived.record_prompt_embedding(conn, int(row[0]), spec, policy, vector, digest, now)
    return True


def cached(conn, engine, loaded, now: float):
    """The loaded engine behind the durable cache: lookups by text hash
    under the engine's joint space and query policy, misses computed
    once and remembered. The lexical engine has no space and is
    returned as is."""
    from vision import semantic

    from . import planning, similarity

    if not isinstance(engine, planning.SemanticEngine):
        return loaded
    spec = semantic.space(engine.provider, engine.model, engine.checkpoint, loaded.dimensions)
    policy = semantic.policy_hash(engine.provider, engine.model, engine.checkpoint)
    sid = similarity.space_id(conn, spec, now)

    def lookup(hashes):
        return current_vectors(conn, sid, policy, hashes)

    def store(digest, vector):
        remember(conn, spec, policy, digest, vector, now)

    return planning.CachedPromptSimilarity(loaded, lookup, store)


# --- the current corpus ---------------------------------------------------------------

#: A prompt text is IN THE CORPUS while it plays a role in some
#: generation or is the text of a CURRENT section (read from the
#: prompt's current text by the current parser). A text a parser once
#: produced and no longer does keeps its stored vector as cache and
#: history -- and leaves the corpus: nothing searches it, nothing
#: indexes it, nothing queues work for it. One predicate, used by the
#: job, the index and the neighbours alike. Binds: parser version.
CORPUS = (
    "(EXISTS (SELECT 1 FROM generation_prompt gp WHERE gp.prompt_id = p.id)"
    " OR EXISTS (SELECT 1 FROM derived_prompt_section s JOIN prompt owner ON owner.id = s.prompt_id"
    "   WHERE s.text_prompt_id = p.id AND s.source_text_hash = owner.text_hash AND s.parser_version = ?))"
)


# --- durable work -----------------------------------------------------------------------

#: Every corpus text without a current vector under (space, policy).
_WANTED = (
    "SELECT DISTINCT p.id FROM prompt p WHERE " + CORPUS + " AND NOT EXISTS ("
    "  SELECT 1 FROM derived_prompt_embedding e WHERE e.prompt_id = p.id"
    "   AND e.space_id = ? AND e.policy_hash = ? AND e.source_text_hash = p.text_hash) ORDER BY p.id"
)


def submit_embed(conn, now: float, *, models_dir: str) -> list[int]:
    """One `embed_prompts` job per participating space. Sections are
    parsed here (pure, cheap) so the items are every corpus TEXT still
    without a current vector under (space, policy). The job FREEZES
    what it means -- the immutable checkpoint, the space key and the
    query policy the items were chosen under -- and the worker re-proves
    it (embed_item). A mutable checkpoint (a hub branch nothing has
    pinned yet) is refused here, exactly as planning refuses it: run
    /jobs/embed first, which provisions and pins. A live job for the
    same (space, policy) is reused, never duplicated."""
    import json

    from vision import semantic

    from . import jobs, retrieval

    for (prompt_id,) in conn.execute("SELECT DISTINCT prompt_id FROM generation_prompt ORDER BY prompt_id").fetchall():
        for grammar in grammars_of(conn, int(prompt_id)):
            sections(conn, int(prompt_id), grammar, now)
    # ONE writer lane for look-then-enqueue: two requests that both saw
    # "no live job" would otherwise each queue the same model work. The
    # lane is claimed before the check; the caller commits.
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    made = []
    for provider, model, configured in retrieval.choices(conn):
        checkpoint = semantic.pin(provider, models_dir, model, configured)
        if not semantic.immutable(provider, checkpoint):
            raise ValueError(
                f"{provider} {model} is configured at the mutable revision {configured!r} and nothing is"
                " provisioned locally to pin it to; run /jobs/embed first -- prompt vectors cannot be recorded"
                " under provenance that may move before the worker loads it"
            )
        policy = semantic.policy_hash(provider, model, checkpoint)
        space = semantic.space(provider, model, checkpoint, 1).key
        live = next(
            (
                int(job_id)
                for job_id, raw in conn.execute(
                    "SELECT id, payload FROM job WHERE kind = 'embed_prompts' AND state IN ('queued', 'running')"
                )
                if raw and (json.loads(raw).get("space"), json.loads(raw).get("policy_hash")) == (space, policy)
            ),
            None,
        )
        if live is not None:
            made.append(live)
            continue
        found = space_of(conn, provider, model, checkpoint)
        items = [
            int(row[0]) for row in conn.execute(_WANTED, (prompt_sections.VERSION, found[0] if found else None, policy))
        ]
        payload = {
            "models_dir": models_dir,
            "choice": [provider, model, checkpoint],
            "space": space,
            "policy_hash": policy,
        }
        made.append(jobs.submit(conn, "embed_prompts", now, payload=payload, items=items))
    return made


def embed_item(conn, prompt_id: int, payload: dict, now: float) -> None:
    """One text under one (space, policy). The identity persisted is the
    LOADED encoder's -- its pinned checkpoint, its space, the query
    policy of that checkpoint -- and it must be the identity the job
    was queued under, or the job is a stale ask and is refused: a
    deploy that changed the query policy, or weights that resolved to
    another commit, must never land vectors under the queued name.
    Skips when a current vector exists."""
    from vision import semantic

    from . import derived

    row = conn.execute("SELECT text, text_hash FROM prompt WHERE id = ?", (prompt_id,)).fetchone()
    if row is None:
        raise LookupError(f"no prompt {prompt_id}")
    text, digest = row
    provider, model, checkpoint = payload["choice"]
    encoder = semantic.encoder(provider, payload["models_dir"], model, checkpoint)
    spec = encoder.space()
    actual = spec.producer_version
    policy = semantic.policy_hash(provider, model, actual)
    queued = (payload.get("space"), payload.get("policy_hash"))
    if queued != (spec.key, policy):
        raise ValueError(
            f"this prompt-embedding job no longer means what it meant when it was queued (queued {queued[0]}"
            f" / {queued[1]}, loaded {spec.key} / {policy}); make the request again"
        )
    if not semantic.immutable(provider, actual):
        raise ValueError(f"the loaded {provider} encoder names the mutable revision {actual!r}; nothing is pinned")
    found = space_of(conn, provider, model, actual)
    if found is not None and current_vectors(conn, found[0], policy, [digest]):
        return
    derived.record_prompt_embedding(conn, prompt_id, spec, policy, encoder.encode_query(text), digest, now)


# --- neighbours in one space ---------------------------------------------------------------


def _vectors(conn, wanted):
    import numpy as np

    held = {}
    batch = [int(v) for v in wanted]
    for start in range(0, len(batch), 500):
        piece = batch[start : start + 500]
        marks = ",".join("?" for _ in piece)
        for embedding_id, blob in conn.execute(
            f"SELECT id, vector FROM derived_prompt_embedding WHERE id IN ({marks})", piece
        ):
            held[embedding_id] = np.frombuffer(blob, dtype=np.float32)
    return np.vstack([held[v] for v in batch])


def neighbours(
    conn, prompt_id: int, provider: str, models_dir: str, k: int, now: float, *, role: str | None = None
) -> dict:
    """The `k` prompt texts nearest to this one, in ONE (space, policy),
    by that space's own cosine. `provider` is an EXACT space selector
    (db/retrieval.py choice_for): ambiguity is refused, never resolved
    by position. The query vector is the prompt's stored vector -- no
    model loads -- so both sides of every comparison were computed by
    the same space and policy. Candidates are the CURRENT corpus;
    `role` constrains them further (texts playing that role in some
    generation) before ranking, searched at full depth, so a rank-300
    original is still the best original. Scores are that space's and
    nobody else's."""
    import numpy as np

    from vision import semantic

    from . import retrieval, similarity

    if role is not None and role not in ROLES:
        raise ValueError(f"no prompt role named {role!r}; one of {', '.join(ROLES)}")
    provider, model, configured = retrieval.choice_for(conn, provider)
    checkpoint = semantic.pin(provider, models_dir, model, configured)
    policy = semantic.policy_hash(provider, model, checkpoint)
    found = space_of(conn, provider, model, checkpoint)
    if found is None:
        raise LookupError(f"no prompt embeddings recorded under {provider}; run /jobs/embed_prompts")
    sid, spec = found
    # the CURRENT corpus only: a text no role and no current section
    # holds keeps its vector as history and is not a candidate
    rows = conn.execute(
        "SELECT e.id, e.prompt_id, e.vector FROM derived_prompt_embedding e"
        " JOIN prompt p ON p.id = e.prompt_id AND e.source_text_hash = p.text_hash"
        " WHERE e.space_id = ? AND e.policy_hash = ? AND " + CORPUS + " ORDER BY e.id",
        (sid, policy, prompt_sections.VERSION),
    ).fetchall()
    own = next((row for row in rows if int(row[1]) == prompt_id), None)
    if own is None:
        raise LookupError(f"prompt {prompt_id} has no current vector under {spec.key}; run /jobs/embed_prompts")
    allowed = None
    if role is not None:
        allowed = {int(r[0]) for r in conn.execute("SELECT prompt_id FROM generation_prompt WHERE role = ?", (role,))}
    ids = [int(row[0]) for row in rows]
    to_prompt = {int(row[0]): int(row[1]) for row in rows}
    manager = similarity.manager_for(conn)
    key = similarity.align(conn, manager, spec, ids, lambda wanted: _vectors(conn, wanted), now, lane=lane(policy))
    query = np.frombuffer(own[2], dtype=np.float32)
    depth = len(ids) if allowed is not None else min(max(int(k), 1) + 1, len(ids))
    labels, scores = manager.search(key, [query], depth)
    ranked = [
        (to_prompt[int(label)], float(score))
        for label, score in zip(labels[0], scores[0], strict=True)
        if int(label) != -1 and to_prompt[int(label)] != prompt_id
    ]
    if allowed is not None:
        ranked = [(other, score) for other, score in ranked if other in allowed]
    results = []
    for other, score in ranked[: max(int(k), 1)]:
        row = conn.execute(
            "SELECT e.uuid, e.slug, p.text FROM prompt p JOIN entity e ON e.id = p.id WHERE p.id = ?", (other,)
        ).fetchone()
        results.append({"prompt_id": other, "uuid": row[0].hex(), "slug": row[1], "text": row[2], "score": score})
    mine = conn.execute(
        "SELECT e.slug, p.text FROM prompt p JOIN entity e ON e.id = p.id WHERE p.id = ?", (prompt_id,)
    ).fetchone()
    return {
        "prompt_id": prompt_id,
        "slug": mine[0] if mine else None,
        "text": mine[1] if mine else None,
        "space": spec.key,
        "policy": policy,
        "role": role,
        "results": results,
    }
