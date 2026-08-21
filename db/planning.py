"""StoryPlanner: structure from frozen evidence. Never prose, never a
database connection.

A planner receives ONE frozen StorySnapshot document (db/stories.py)
and explicitly versioned computation services, and returns a StoryPlan:
phases, representatives, first-class CLAIMS whose evidence references
resolve inside the snapshot, and what the evidence does NOT support.
It answers "what is interesting, what changed, what are the phases" --
not "how should a human be told". `label_hint` is the only human-facing
field, and it is a deterministic suggestion a later labeler may
replace; nothing here writes a sentence.

EVERY structure is a conclusion and every conclusion is a Claim: a
phase boundary is a `prompt_shift` claim with evidence on both sides,
a family is a `prompt_family` claim, so a renderer may word what is
here and can invent nothing. A plan is an exact PARTITION of its
snapshot -- every member exactly once, representatives inside their
phase, every reference resolving inward -- and `validate_plan` proves
that before persistence and again on every read.

The similarity service is a Seam with the same discipline as the
snapshot: `embed(texts) -> vectors`, the planner supplying the FROZEN
prompt strings, the output VALIDATED (exactly N vectors, one dimension,
finite) because a producer that returns N-1 vectors is broken, not
padded. The same engine that answers semantic retrieval sits behind
the production Adapter, but nothing here may look a vector up by file:
that would smuggle today's library back across the seam the snapshot
exists to close. A lexical bag-of-tokens Adapter gives tests a
deterministic oracle and any library a model-free fallback.

Two identities, two purposes. The REQUEST identity -- snapshot sha,
planner kind/version, engine name/version, exact settings -- is known
before any model work, so an identical request reuses the finished
plan or the queued job without embedding anything again. The DOCUMENT
identity is the canonical plan's sha with `planned_at` excluded; the
same evidence under the same policy is ONE plan, and a new policy
coexists with the old one instead of overwriting it.

Production planning is DURABLE WORK: the request is recorded, a
`story_plan` job loads the engine and computes off the request thread,
exactly as the snapshot's own doctrine says model work happens after
freezing. The lexical engine is pure code and may be called directly.

V1 is deterministic end to end. Prompt identity alone never splits a
phase -- fifty wildcard expansions of one prompt are one creative
thread -- and day-precision evidence never yields sub-day chronology:
the planner says `unsupported` rather than inventing an order.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import typing

from . import prompt_sections
from .stories import canonical, digest

FORMAT_VERSION = 3

#: Claim kinds that assert DIRECTION -- before/after, added/removed,
#: from/to. They exist only in a sequenced plan, where the phase list is
#: a chronology. An unsequenced plan states symmetric DIFFERENCES.
_DIRECTIONAL = frozenset({"prompt_shift", "artifact_change", "parameter_change"})
_SYMMETRIC = frozenset({"artifact_difference", "parameter_difference"})


def prompt_sections_grammar(tool: str | None) -> str:
    """The section grammar for a member's prompts, from its frozen tool
    name -- the same rule the live library applies (db/prompts.py
    grammar_for), repeated here so the planner imports no database
    module."""
    return "swarm" if (tool or "").lower().startswith("swarm") else "plain"


#: Precisions fine enough that event order is evidence of sequence.
_SEQUENCED = {"hour", "minute", "second", "subsecond"}


# --- settings: exact, fail-closed --------------------------------------------


def validated_settings(settings: dict | None, defaults: dict) -> dict:
    """V1 means exactly `phase_threshold`: a finite number in [0, 1],
    not a bool, not a string. An unknown key would ride the identity
    while meaning nothing; a string would compare as a string. The same
    exact-shape doctrine CollectionRule paid for."""
    held = dict(defaults)
    if settings is None:
        return held
    if not isinstance(settings, dict):
        raise TypeError("planner settings are an object")
    unknown = sorted(set(settings) - set(defaults))
    if unknown:
        raise ValueError(f"unknown planner setting(s) {unknown}; this planner knows {sorted(defaults)}")
    for key, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} is a finite number, not {value!r}")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{key} lies in [0, 1], not {value!r}")
        held[key] = float(value)
    return held


# --- the similarity Seam -----------------------------------------------------


class PromptSimilarity(typing.Protocol):
    name: str
    version: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def validated_vectors(vectors, expected: int) -> list[list[float]]:
    """The producer contract: exactly `expected` vectors, one shared
    dimension, every value a finite number. Anything else is producer
    failure and is refused -- never padded into coherence."""
    made = [list(row) for row in vectors]
    if len(made) != expected:
        raise ValueError(f"the similarity engine returned {len(made)} vectors for {expected} texts")
    if not made:
        return made
    width = len(made[0])
    if width == 0:
        raise ValueError("the similarity engine returned zero-dimensional vectors; that is not a space")
    for row in made:
        if len(row) != width:
            raise ValueError(f"the similarity engine returned mixed dimensions ({width} and {len(row)})")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"the similarity engine returned a non-finite or non-numeric value {value!r}")
    return [[float(value) for value in row] for row in made]


_TOKEN = re.compile(r"[a-z0-9]+")


class LexicalPromptSimilarity:
    """A deterministic bag-of-tokens embedder over the SUPPLIED texts:
    the vocabulary is the sorted token set of the batch, a vector is the
    token counts. Cosine of two such vectors is token overlap. Exists as
    the test oracle and the model-free fallback; it is an embedder, not
    a second similarity store."""

    name = "lexical-bow"
    version = "1"

    def embed(self, texts):
        bags = [list(_TOKEN.findall(text.lower())) for text in texts]
        vocabulary = sorted({token for bag in bags for token in bag})
        index = {token: i for i, token in enumerate(vocabulary)}
        made = []
        for bag in bags:
            row = [0.0] * max(1, len(vocabulary))
            for token in bag:
                row[index[token]] += 1.0
            made.append(row)
        return made


class SemanticPromptSimilarity:
    """The production Adapter: a loaded semantic text encoder
    (vision/semantic) embedding the frozen prompt strings. Its identity
    is the ACTUAL encoder's space -- provider, model, pinned checkpoint
    and package version -- read from the engine spec, never assumed
    from the selector the request used."""

    def __init__(self, encoder, spec: SemanticEngine):
        self._encoder = encoder
        # PROVE the loaded weights are the engine the request was queued
        # under: a pointer that moved between queue and run is not it.
        loaded = dataclasses.replace(spec, checkpoint=encoder.space().producer_version)
        if loaded.identity() != spec.identity():
            raise ValueError(
                f"the loaded engine ({loaded.checkpoint}) is not the engine the request was queued under"
                f" ({spec.checkpoint}); make the request again"
            )
        self.name, self.version = spec.identity()
        self.dimensions = int(encoder.dimensions)

    def embed(self, texts):
        return [[float(x) for x in self._encoder.encode_query(text)] for text in texts]


class CachedPromptSimilarity:
    """A durable cache in front of a loaded engine, keyed by the sha256 of
    the EXACT text -- the same hash `prompt.text_hash` carries -- under
    the engine's own immutable text space. `lookup(hashes)` and
    `store(hash, vector)` are the whole contract; nothing here knows a
    file, a generation, or a role, so the snapshot Seam stays closed.
    Identity is the inner engine's: a cached vector is that engine's
    vector, byte for byte, or it is not served."""

    def __init__(self, inner: PromptSimilarity, lookup, store):
        self._inner = inner
        self._lookup = lookup
        self._store = store
        self.name, self.version = inner.name, inner.version

    def embed(self, texts):
        hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        held = dict(self._lookup(sorted(set(hashes))))
        missing: list[str] = []
        for text, key in zip(texts, hashes, strict=True):
            if key not in held and text not in missing:
                missing.append(text)
        if missing:
            fresh = validated_vectors(self._inner.embed(missing), len(missing))
            for text, vector in zip(missing, fresh, strict=True):
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                held[digest] = vector
                self._store(digest, vector)
        return [list(held[digest]) for digest in hashes]


@dataclasses.dataclass(frozen=True)
class LexicalEngine:
    """Engine spec: pure code, no weights, loads instantly."""

    selector: typing.ClassVar[str] = "lexical"

    def identity(self) -> tuple[str, str]:
        return (LexicalPromptSimilarity.name, LexicalPromptSimilarity.version)

    def load(self) -> PromptSimilarity:
        return LexicalPromptSimilarity()

    def payload(self) -> dict:
        return {}


@dataclasses.dataclass(frozen=True)
class SemanticEngine:
    """Engine spec for one configured semantic provider: identity is
    known WITHOUT loading weights (the space the provider declares for
    this model and pinned checkpoint), so a request's identity can be
    computed on the request thread and the weights loaded by the job."""

    provider: str
    model: str
    checkpoint: str
    models_dir: str

    @property
    def selector(self) -> str:
        return self.provider

    def identity(self) -> tuple[str, str]:
        from vision import semantic

        # (space key, digest of the QUERY policy): everything that turns a
        # text into a vector for this provider/model/checkpoint -- Qwen's
        # query instruction included, which its stored-media policy omits.
        # The same token keys stored prompt vectors (db/prompts.py).
        space = semantic.space(self.provider, self.model, self.checkpoint, 1)
        return (space.key, semantic.policy_hash(self.provider, self.model, self.checkpoint))

    def load(self) -> PromptSimilarity:
        from vision import semantic

        encoder = semantic.encoder(self.provider, self.models_dir, self.model, self.checkpoint, offline=True)
        return SemanticPromptSimilarity(encoder, self)

    def payload(self) -> dict:
        return dataclasses.asdict(self)


def engine_for(conn, selector: str, models_dir: str):
    """The engine spec a selector names -- EXACTLY. `lexical` is the pure
    engine; a provider name (`openclip`, `qwen`) is that provider's
    configured semantic model, refused when that provider is not among
    the configured spaces rather than silently substituted."""
    if selector == LexicalEngine.selector:
        return LexicalEngine()
    from vision import semantic

    from . import retrieval

    if selector not in semantic.PROVIDERS:
        known = ", ".join(sorted(semantic.PROVIDERS))
        raise ValueError(f"no similarity engine named {selector!r}; one of lexical, {known}")
    for provider, model, configured in retrieval.choices(conn):
        if provider == selector:
            checkpoint = semantic.pin(provider, models_dir, model, configured)
            if not semantic.immutable(provider, checkpoint):
                raise ValueError(
                    f"{provider} {model} is configured at the mutable revision {configured!r} and nothing is"
                    " provisioned locally to pin it to; run /jobs/embed first -- a plan cannot be queued"
                    " under provenance that may move before the worker loads it"
                )
            return SemanticEngine(provider, model, checkpoint, models_dir)
    raise ValueError(f"the {selector!r} provider is not among the configured semantic spaces")


def pairwise_cosine(vectors: list[list[float]]) -> list[list[float]]:
    """Every pair's cosine, via the shared unit-normalisation (db/similarity
    normalise) and one matrix product, over VALIDATED vectors. An
    all-zero vector has cosine 0 with everything, including itself."""
    from .similarity import normalise

    if not vectors:
        return []
    unit = normalise(vectors)
    return (unit @ unit.T).astype(float).tolist()


# --- the plan document -------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PlanRef:
    id: int
    sha256: str
    reused: bool


def _member_ref(ordinal: int) -> str:
    return f"member-{ordinal + 1:03d}"


class GenerationHistoryPlanner:
    """Phases of one generation session from frozen evidence alone.

    Chronology: when every member's occurrence is finer than a day, the
    event order IS the sequence and phases are consecutive runs; when
    any member is day-precision the planner declares chronology
    unsupported and phases are prompt FAMILIES, members listed in event
    order without a claim that one came before another.

    Boundaries: the cosine between a member's prompt and the running
    phase's first prompt falls below `phase_threshold` -- and the
    boundary is itself a `prompt_shift` claim with both prompts as
    evidence. Prompt identity is never consulted, so wildcard expansions
    of one prompt stay one phase. Parameter and artifact changes are
    CLAIMS about a boundary, not boundaries themselves.
    """

    kind = "generation_history"
    #: Bumped with db/prompt_sections.VERSION: the main-section texts the
    #: planner compares are that parser's output.
    version = 7
    defaults: typing.ClassVar[dict] = {"phase_threshold": 0.5}

    def __init__(self, similarity: PromptSimilarity, settings: dict | None = None):
        self.similarity = similarity
        self.settings = validated_settings(settings, self.defaults)

    def plan(self, snapshot: dict, snapshot_sha256: str) -> dict:
        if snapshot.get("v") != 1:
            raise ValueError(f"this planner reads StorySnapshot v1, not v{snapshot.get('v')!r}")
        if snapshot["subject"]["event_kind"] != "generation_session":
            raise ValueError("GenerationHistoryPlanner plans generation sessions only")
        members = sorted(snapshot["members"], key=lambda one: one["ordinal"])
        refs = [_member_ref(one["ordinal"]) for one in members]
        threshold = self.settings["phase_threshold"]
        unsupported: list[dict] = []

        sequenced = all((one.get("occurrence") or {}).get("precision") in _SEQUENCED for one in members)
        if not sequenced:
            unsupported.append(
                {
                    "kind": "chronology",
                    "reason": "at least one member's occurrence is day-precision or absent;"
                    " event order is not evidence of sequence",
                }
            )

        # The MAIN section of each prompt, read with the grammar of the
        # tool that wrote it (db/prompt_sections.py): a <segment:face>
        # tail or a <refiner> stage is not part of what the images are of.
        prompts = [
            prompt_sections.main(
                (one.get("generation") or {}).get("prompt") or "",
                prompt_sections_grammar((one.get("generation") or {}).get("tool")),
            )
            for one in members
        ]
        known = [i for i, text in enumerate(prompts) if text.strip()]
        gaps = [i for i in range(len(members)) if i not in known]
        if gaps:
            unsupported.append(
                {"kind": "prompt_evidence", "reason": "no frozen prompt", "member_refs": [refs[i] for i in gaps]}
            )

        # Only KNOWN prompts reach the engine: an empty string is not a
        # prompt, and embedding it would manufacture a vector for nothing.
        vectors = validated_vectors(self.similarity.embed([prompts[i] for i in known]), len(known)) if known else []
        sparse = pairwise_cosine(vectors)
        slot = {i: k for k, i in enumerate(known)}

        def cosine(a: int, b: int) -> float:
            return sparse[slot[a]][slot[b]]

        # Written vs run: a member whose frozen evidence carries BOTH the
        # prompt as written (role original) and the prompt the generator
        # ran (role effective) is compared with ITSELF. Wildcard expansion
        # keeps the two close; a substantial rewrite is a claim about that
        # member -- never a boundary, never chronology. A missing original
        # is absence: no pair, no vector, no claim. Its own embed call, so
        # the batch-scoped lexical oracle's vocabulary for the effective
        # prompts is untouched.
        pairs: list[tuple[int, str, str]] = []
        for i, one in enumerate(members):
            by_role = {p["role"]: p for p in ((one.get("generation") or {}).get("prompts") or [])}
            written, ran = by_role.get("original"), by_role.get("effective")
            if not written or not ran or written["text_hash"] == ran["text_hash"]:
                continue
            grammar = prompt_sections_grammar((one.get("generation") or {}).get("tool"))
            wrote, did = prompt_sections.main(written["text"], grammar), prompt_sections.main(ran["text"], grammar)
            if wrote and did and wrote != did:
                pairs.append((i, wrote, did))
        rewritten: dict[int, float] = {}
        if pairs:
            both = validated_vectors(self.similarity.embed([t for _, w, r in pairs for t in (w, r)]), 2 * len(pairs))
            grid = pairwise_cosine(both)
            for j, (i, _, _) in enumerate(pairs):
                rewritten[i] = grid[2 * j][2 * j + 1]

        # A member with no prompt evidence is placed by chronology and
        # asserts nothing about prompts: in a sequenced plan it joins the
        # running phase (it is not a boundary), so phases stay contiguous
        # and the phase list IS the chronology; without chronology it is
        # its own family, grouped with nothing. Its OTHER facts (seed,
        # artifacts, parameters) stay in every comparison they belong to.
        groups = (
            _sequential_phases(cosine, threshold, known, len(members))
            if sequenced
            else _family_phases(cosine, threshold, known, gaps)
        )

        claims: list[dict] = []
        phases: list[dict] = []
        for number, indexes in enumerate(groups, start=1):
            member_refs = [refs[i] for i in indexes]
            spoken = [i for i in indexes if i in slot]
            silent = [i for i in indexes if i not in slot]
            claim_refs = []
            inside = [cosine(a, b) for a in spoken for b in spoken if a < b]
            if sequenced:
                if number > 1:
                    previous_spoken = [i for i in groups[number - 2] if i in slot]
                    if previous_spoken and spoken:
                        claim_refs.append(
                            _claim(
                                claims,
                                "prompt_shift",
                                confidence=1.0,
                                evidence=[
                                    f"{refs[previous_spoken[0]]}:generation.prompt",
                                    f"{refs[spoken[0]]}:generation.prompt",
                                ],
                                facts={
                                    "cosine": round(cosine(previous_spoken[0], spoken[0]), 4),
                                    "threshold": threshold,
                                },
                            )
                        )
                if inside:
                    claim_refs.append(
                        _claim(
                            claims,
                            "prompt_similarity",
                            confidence=max(0.0, min([1.0, *inside])),
                            evidence=[f"{refs[i]}:generation.prompt" for i in spoken],
                            facts={"relationship": "same_prompt_family", "min_pairwise_cosine": round(min(inside), 4)},
                        )
                    )
            elif spoken:
                claim_refs.append(
                    _claim(
                        claims,
                        "prompt_family",
                        confidence=max(0.0, min([1.0, *inside])) if inside else 1.0,
                        evidence=[f"{refs[i]}:generation.prompt" for i in spoken],
                        facts={
                            "size": len(spoken),
                            "threshold": threshold,
                            "min_pairwise_cosine": round(min(inside), 4) if inside else None,
                        },
                    )
                )
            if silent:
                claim_refs.append(
                    _claim(claims, "prompt_evidence_missing", 1.0, [refs[i] for i in silent], {"members": len(silent)})
                )
            moved = [i for i in indexes if i in rewritten and rewritten[i] < threshold]
            if moved:
                claim_refs.append(
                    _claim(
                        claims,
                        "prompt_rewrite",
                        1.0,
                        [f"{refs[i]}:generation.prompts" for i in moved],
                        {
                            "members": len(moved),
                            "min_cosine": round(min(rewritten[i] for i in moved), 4),
                            "threshold": threshold,
                        },
                    )
                )
            seeds = sorted(
                {
                    members[i]["generation"]["seed"]
                    for i in indexes
                    if (members[i].get("generation") or {}).get("seed") is not None
                }
            )
            if len(seeds) >= 2:
                claim_refs.append(
                    _claim(
                        claims,
                        "seed_variation",
                        confidence=1.0,
                        evidence=[f"{refs[i]}:generation.seed" for i in indexes],
                        facts={"distinct_seeds": len(seeds)},
                    )
                )
            if sequenced:
                label = f"Phase {number}"
            elif spoken:
                label = f"Prompt family {number}"
            else:
                label = "Prompt evidence gap"
            if number > 1:
                # Sequenced: the previous phase came BEFORE, so a change has
                # a direction. Unsequenced: the family listed before is
                # merely another family, and the only honest claim is that
                # the two DIFFER -- symmetric facts, no before/after.
                previous = groups[number - 2]
                changed = _artifact_change(members, previous, indexes, refs, sequenced)
                if changed is not None:
                    kind = "artifact_change" if sequenced else "artifact_difference"
                    claim_refs.append(_claim(claims, kind, 1.0, changed[0], changed[1]))
                    label += " · new artifacts" if sequenced else " · different artifacts"
                params = _parameter_change(members, previous, indexes, refs, sequenced)
                if params is not None:
                    kind = "parameter_change" if sequenced else "parameter_difference"
                    claim_refs.append(_claim(claims, kind, 1.0, params[0], params[1]))
            representative = refs[_medoid(spoken, cosine)] if spoken else member_refs[0]
            phases.append(
                {
                    "id": f"phase-{number:03d}",
                    "member_refs": member_refs,
                    "representative_refs": [representative],
                    "label_hint": label,
                    "claim_refs": claim_refs,
                }
            )

        tool = next(((one.get("generation") or {}).get("tool") for one in members if one.get("generation")), None)
        return {
            "v": FORMAT_VERSION,
            "snapshot_sha256": snapshot_sha256,
            "planner": {
                "kind": self.kind,
                "version": self.version,
                "settings": dict(sorted(self.settings.items())),
                "similarity": {"name": self.similarity.name, "version": self.similarity.version},
            },
            "subject": {
                "kind": snapshot["subject"]["event_kind"],
                "sequenced": sequenced,
                "label_hint": f"{tool or 'generation'} session · {len(members)} outputs · {len(phases)} "
                + ("phases" if sequenced else "prompt families"),
            },
            "phases": phases,
            "claims": claims,
            "unsupported": unsupported,
        }


def _claim(claims: list, kind: str, confidence: float, evidence: list[str], facts: dict) -> str:
    claim_id = f"claim-{len(claims) + 1:03d}"
    claims.append(
        {"id": claim_id, "kind": kind, "confidence": round(confidence, 4), "evidence_refs": evidence, "facts": facts}
    )
    return claim_id


def _sequential_phases(cosine, threshold: float, known: list[int], total: int) -> list[list[int]]:
    """Contiguous runs over EVERY member, in event order -- the phase
    list is the chronology. A member with prompt evidence joins the
    running phase while its prompt stays within `threshold` of the
    phase's first KNOWN prompt; a member without prompt evidence is
    neither a boundary nor evidence, so it joins the running phase."""
    known_set = set(known)
    groups: list[list[int]] = []
    anchor: int | None = None
    for i in range(total):
        if i not in known_set:
            if not groups:
                groups.append([])
            groups[-1].append(i)
            continue
        if groups and (anchor is None or cosine(anchor, i) >= threshold):
            groups[-1].append(i)
            anchor = i if anchor is None else anchor
        else:
            groups.append([i])
            anchor = i
    return groups


def _family_phases(cosine, threshold: float, known: list[int], gaps: list[int]) -> list[list[int]]:
    """Without chronology: connected components of the `>= threshold`
    graph over the members WITH prompt evidence, each listed in event
    order, families ordered by their first member; a member without
    prompt evidence is its own family -- a partition, not a sequence."""
    seen: set[int] = set()
    groups: list[list[int]] = []
    for start in known:
        if start in seen:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            i = stack.pop()
            component.append(i)
            for j in known:
                if j not in seen and cosine(i, j) >= threshold:
                    seen.add(j)
                    stack.append(j)
        groups.append(sorted(component))
    groups.extend([i] for i in gaps)
    return sorted(groups, key=lambda group: group[0])


def _medoid(indexes: list[int], cosine) -> int:
    """The member closest to all the others -- ties to the earliest."""
    best, best_score = indexes[0], -2.0
    for i in indexes:
        score = sum(cosine(i, j) for j in indexes if j != i) / max(1, len(indexes) - 1)
        if score > best_score + 1e-12:
            best, best_score = i, score
    return best


def _artifact_uuids(member: dict) -> set[str]:
    return {one["uuid"] for one in ((member.get("generation") or {}).get("artifacts") or [])}


def _artifact_change(members, other, here, refs, sequenced: bool):
    """The artifact sets of two phases, as a directed change (sequenced:
    `other` came before `here`) or as a symmetric difference."""
    there = set().union(*(_artifact_uuids(members[i]) for i in other))
    this = set().union(*(_artifact_uuids(members[i]) for i in here))
    if there == this:
        return None
    evidence = [f"{refs[i]}:generation.artifacts" for i in (*other, *here)]
    if sequenced:
        return evidence, {"added": sorted(this - there), "removed": sorted(there - this)}
    return evidence, {"only_here": sorted(this - there), "only_other": sorted(there - this)}


_PARAMS = ("sampler", "scheduler", "steps", "cfg", "denoise", "clip_skip", "width", "height")


def _parameter_change(members, other, here, refs, sequenced: bool):
    def settled(indexes):
        return {key: sorted({str((members[i].get("generation") or {}).get(key)) for i in indexes}) for key in _PARAMS}

    there, this = settled(other), settled(here)
    differing = [key for key in _PARAMS if there[key] != this[key]]
    if not differing:
        return None
    evidence = [f"{refs[i]}:generation.{key}" for key in differing for i in (*other, *here)]
    if sequenced:
        return evidence, {"changed": {key: {"from": there[key], "to": this[key]} for key in differing}}
    return evidence, {"differs": {key: {"here": this[key], "other": there[key]} for key in differing}}


# --- validation: a plan is an exact partition of its snapshot ----------------

_REF = re.compile(r"^(member-\d{3})(?::([a-z_]+(?:\.[a-z_]+)*))?$")


def unresolved(plan: dict, snapshot: dict) -> list[str]:
    """Every member_ref, representative_ref and evidence_ref that does
    NOT resolve inside the snapshot, plus every claim_ref that names no
    claim. Empty means the plan points nowhere but inward."""
    members = {_member_ref(one["ordinal"]): one for one in snapshot["members"]}
    claims = {one["id"] for one in plan["claims"]}
    bad: list[str] = []

    def check(ref: str) -> None:
        match = _REF.match(ref)
        if not match or match.group(1) not in members:
            bad.append(ref)
            return
        path = match.group(2)
        if path:
            node: typing.Any = members[match.group(1)]
            for step in path.split("."):
                if not isinstance(node, dict) or step not in node:
                    bad.append(ref)
                    return
                node = node[step]

    for phase in plan["phases"]:
        for ref in (*phase["member_refs"], *phase["representative_refs"]):
            check(ref)
        bad.extend(ref for ref in phase["claim_refs"] if ref not in claims)
    for claim in plan["claims"]:
        for ref in claim["evidence_refs"]:
            check(ref)
    return bad


#: The durable vocabulary of a StoryPlan v1 -- FROZEN. These constants
#: describe what a v1 document may contain and never track the running
#: code: a v1 row written today must still parse as v1 after a v2 exists,
#: or the "immutable historical artifact" is a row that rots. Adding a
#: claim kind, a planner, or a key is a v2.
STORY_PLAN_V1 = {
    "version": 1,
    "planners": frozenset({"generation_history"}),
    "claims": frozenset(
        {
            "prompt_shift",
            "prompt_similarity",
            "prompt_family",
            "prompt_evidence_missing",
            "seed_variation",
            "artifact_change",
            "parameter_change",
        }
    ),
    "unsupported": frozenset({"chronology", "prompt_evidence"}),
    "settings": frozenset({"phase_threshold"}),
    "subjects": frozenset({"generation_session"}),
}

#: StoryPlan v2 -- FROZEN. v1 plus the symmetric difference claims an
#: UNSEQUENCED plan states between families (v1 had only the directed
#: change claims, which assert a before and an after).
STORY_PLAN_V2 = {
    **STORY_PLAN_V1,
    "version": 2,
    "claims": STORY_PLAN_V1["claims"] | frozenset({"artifact_difference", "parameter_difference"}),
}

#: StoryPlan v3 -- FROZEN. v2 plus `prompt_rewrite`: the prompt the
#: generator ran differs substantially from the prompt as written.
STORY_PLAN_V3 = {
    **STORY_PLAN_V2,
    "version": 3,
    "claims": STORY_PLAN_V2["claims"] | frozenset({"prompt_rewrite"}),
}
_ID = re.compile(r"^(phase|claim)-[0-9]{3}$")
_SHA = re.compile(r"^[0-9a-f]{64}$")


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _unit(value) -> bool:
    return _number(value) and 0.0 <= value <= 1.0


def _cosine(value) -> bool:
    return _number(value) and -1.0 <= value <= 1.0


def _in(value, vocabulary) -> bool:
    """Membership that cannot raise: an unhashable value is simply not
    in a vocabulary of strings."""
    return isinstance(value, str) and value in vocabulary


def _keys(node, exact: set[str], optional: set[str], where: str, bad: list[str]) -> bool:
    if not isinstance(node, dict):
        bad.append(f"{where} is not an object")
        return False
    held = set(node)
    if held - exact - optional or exact - held:
        bad.append(f"{where} keys are {sorted(held)}, not {sorted(exact)} (+ optional {sorted(optional)})")
        return False
    return True


def _strings(value) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def _facts_valid_v1(kind: str, facts) -> bool:
    if not isinstance(facts, dict):
        return False
    keys = set(facts)
    if kind == "prompt_shift":
        return keys == {"cosine", "threshold"} and _cosine(facts["cosine"]) and _unit(facts["threshold"])
    if kind == "prompt_similarity":
        return (
            keys == {"relationship", "min_pairwise_cosine"}
            and facts["relationship"] == "same_prompt_family"
            and _cosine(facts["min_pairwise_cosine"])
        )
    if kind == "prompt_family":
        return (
            keys == {"size", "threshold", "min_pairwise_cosine"}
            and _integer(facts["size"], 1)
            and _unit(facts["threshold"])
            and (facts["min_pairwise_cosine"] is None or _cosine(facts["min_pairwise_cosine"]))
        )
    if kind == "prompt_evidence_missing":
        return keys == {"members"} and _integer(facts["members"], 1)
    if kind == "seed_variation":
        return keys == {"distinct_seeds"} and _integer(facts["distinct_seeds"], 2)
    if kind == "artifact_change":
        return (
            keys == {"added", "removed"}
            and all(_strings(facts[k]) for k in keys)
            and bool(facts["added"] or facts["removed"])
        )
    if kind == "parameter_change":
        return _parameter_facts(facts, "changed", ("from", "to"))
    return False


def _parameter_facts(facts: dict, key: str, sides: tuple[str, str]) -> bool:
    return (
        set(facts) == {key}
        and isinstance(facts[key], dict)
        and bool(facts[key])
        and all(
            isinstance(k, str) and isinstance(v, dict) and set(v) == set(sides) and all(_strings(v[s]) for s in v)
            for k, v in facts[key].items()
        )
    )


def _facts_valid_v2(kind: str, facts) -> bool:
    if not isinstance(facts, dict):
        return False
    if kind == "artifact_difference":
        return (
            set(facts) == {"only_here", "only_other"}
            and all(_strings(facts[k]) for k in facts)
            and bool(facts["only_here"] or facts["only_other"])
        )
    if kind == "parameter_difference":
        return _parameter_facts(facts, "differs", ("here", "other"))
    return _facts_valid_v1(kind, facts)


def _facts_valid_v3(kind: str, facts) -> bool:
    if not isinstance(facts, dict):
        return False
    if kind == "prompt_rewrite":
        return (
            set(facts) == {"members", "min_cosine", "threshold"}
            and _integer(facts["members"], 1)
            and _cosine(facts["min_cosine"])
            and _unit(facts["threshold"])
        )
    return _facts_valid_v2(kind, facts)


def validate_story_plan_v1(plan) -> list[str]:
    """The exact grammar of a StoryPlan v1 document against the FROZEN
    v1 vocabulary, with no reference to any snapshot or to the running
    code: shape, types, vocabularies, cardinalities, domains. Controlled
    reasons, never an exception, whatever the bytes say."""
    return _validate_story_plan(plan, STORY_PLAN_V1, _facts_valid_v1)


def validate_story_plan_v2(plan) -> list[str]:
    """The exact grammar of a StoryPlan v2 document against the FROZEN
    v2 vocabulary."""
    return _validate_story_plan(plan, STORY_PLAN_V2, _facts_valid_v2)


def validate_story_plan_v3(plan) -> list[str]:
    """The exact grammar of a StoryPlan v3 document against the FROZEN
    v3 vocabulary."""
    return _validate_story_plan(plan, STORY_PLAN_V3, _facts_valid_v3)


def _validate_story_plan(plan, vocabulary: dict, facts_valid) -> list[str]:
    bad: list[str] = []
    top = {"v", "snapshot_sha256", "planner", "subject", "phases", "claims", "unsupported"}
    if not _keys(plan, top, {"planned_at"}, "plan", bad):
        return bad
    if plan["v"] != vocabulary["version"]:
        bad.append(f"format v{plan['v']!r}, not v{vocabulary['version']}")
    if not (isinstance(plan["snapshot_sha256"], str) and _SHA.match(plan["snapshot_sha256"])):
        bad.append("snapshot_sha256 is not a sha256")
    if "planned_at" in plan and not _number(plan["planned_at"]):
        bad.append("planned_at is not a number")
    planner = plan["planner"]
    if _keys(planner, {"kind", "version", "settings", "similarity"}, set(), "planner", bad):
        if not _in(planner["kind"], vocabulary["planners"]):
            bad.append(f"unknown planner {planner['kind']!r}")
        if not _integer(planner["version"], 1):
            bad.append("planner.version is not a positive integer")
        settings_ok = _keys(planner["settings"], set(vocabulary["settings"]), set(), "planner.settings", bad)
        if settings_ok and not _unit(planner["settings"]["phase_threshold"]):
            bad.append("planner.settings.phase_threshold is not a number in [0, 1]")
        similarity_ok = _keys(planner["similarity"], {"name", "version"}, set(), "planner.similarity", bad)
        if similarity_ok and not all(isinstance(planner["similarity"][k], str) for k in ("name", "version")):
            bad.append("planner.similarity names are not strings")
    subject = plan["subject"]
    if _keys(subject, {"kind", "sequenced", "label_hint"}, set(), "subject", bad):
        if not _in(subject["kind"], vocabulary["subjects"]):
            bad.append(f"unknown subject kind {subject['kind']!r}")
        if not isinstance(subject["sequenced"], bool):
            bad.append("subject.sequenced is not a bool")
        if not isinstance(subject["label_hint"], str):
            bad.append("subject.label_hint is not a string")
    if not isinstance(plan["phases"], list) or not isinstance(plan["claims"], list):
        bad.append("phases and claims are lists")
        return bad
    for i, phase in enumerate(plan["phases"]):
        where = f"phases[{i}]"
        exact = {"id", "member_refs", "representative_refs", "label_hint", "claim_refs"}
        if not _keys(phase, exact, set(), where, bad):
            continue
        if not (isinstance(phase["id"], str) and _ID.match(phase["id"]) and phase["id"].startswith("phase-")):
            bad.append(f"{where}.id is not a phase id")
        if not _strings(phase["member_refs"]) or not phase["member_refs"]:
            bad.append(f"{where}.member_refs is not a non-empty list of refs")
        if not _strings(phase["representative_refs"]) or len(phase["representative_refs"]) != 1:
            bad.append(f"{where}.representative_refs is not exactly one ref")
        if not isinstance(phase["label_hint"], str):
            bad.append(f"{where}.label_hint is not a string")
        if not _strings(phase["claim_refs"]):
            bad.append(f"{where}.claim_refs is not a list of claim ids")
    for i, claim in enumerate(plan["claims"]):
        where = f"claims[{i}]"
        if not _keys(claim, {"id", "kind", "confidence", "evidence_refs", "facts"}, set(), where, bad):
            continue
        if not (isinstance(claim["id"], str) and _ID.match(claim["id"]) and claim["id"].startswith("claim-")):
            bad.append(f"{where}.id is not a claim id")
        if not _in(claim["kind"], vocabulary["claims"]):
            bad.append(f"{where}.kind {claim['kind']!r} is not a v{vocabulary['version']} claim kind")
        elif not facts_valid(claim["kind"], claim["facts"]):
            bad.append(f"{where}.facts do not fit {claim['kind']}")
        if not _unit(claim["confidence"]):
            bad.append(f"{where}.confidence is not in [0, 1]")
        if not _strings(claim["evidence_refs"]) or not claim["evidence_refs"]:
            bad.append(f"{where}.evidence_refs is not a non-empty list of refs")
    if not isinstance(plan["unsupported"], list):
        bad.append("unsupported is a list")
        return bad
    for i, told in enumerate(plan["unsupported"]):
        where = f"unsupported[{i}]"
        if not _keys(told, {"kind", "reason"}, {"member_refs"}, where, bad):
            continue
        if not _in(told["kind"], vocabulary["unsupported"]):
            bad.append(f"{where}.kind {told['kind']!r} is not a v{vocabulary['version']} unsupported kind")
        if not isinstance(told["reason"], str):
            bad.append(f"{where}.reason is not a string")
        if "member_refs" in told and not _strings(told["member_refs"]):
            bad.append(f"{where}.member_refs is not a list of refs")
    return bad


_GRAMMARS = {1: validate_story_plan_v1, 2: validate_story_plan_v2, 3: validate_story_plan_v3}


def validate_story_plan(plan) -> list[str]:
    """Dispatch on the document's OWN version: a v1 row is judged by the
    frozen v1 grammar however many versions exist later. A document
    with no parsable version, or a version nobody defined, is invalid."""
    version = plan.get("v") if isinstance(plan, dict) else None
    grammar = _GRAMMARS.get(version) if isinstance(version, int) and not isinstance(version, bool) else None
    if grammar is None:
        return [f"no StoryPlan grammar for version {version!r}; known: {sorted(_GRAMMARS)}"]
    return grammar(plan)


def validate_plan(plan: dict, snapshot: dict, snapshot_sha256: str) -> list[str]:
    """Every way a plan can be wrong, as a list of reasons -- empty is
    the only acceptable answer. Its own version's grammar first; then
    the snapshot it names; an exact partition (every member exactly
    once); representatives inside their phase; unique phase and claim
    ids; every reference inward."""
    bad = validate_story_plan(plan)
    if bad:
        return bad
    if plan["snapshot_sha256"] != snapshot_sha256:
        return ["the plan names a different snapshot"]
    expected = sorted(_member_ref(one["ordinal"]) for one in snapshot["members"])
    placed = [ref for phase in plan["phases"] for ref in phase["member_refs"]]
    if sorted(placed) != expected:
        missing = sorted(set(expected) - set(placed))
        extra = sorted(set(placed) - set(expected))
        repeated = sorted({ref for ref in placed if placed.count(ref) > 1})
        bad.append(f"not a partition of the snapshot: missing {missing}, extra {extra}, repeated {repeated}")
    phase_ids = [phase["id"] for phase in plan["phases"]]
    if len(set(phase_ids)) != len(phase_ids):
        bad.append("phase ids are not unique")
    claim_ids = [claim["id"] for claim in plan["claims"]]
    if len(set(claim_ids)) != len(claim_ids):
        bad.append("claim ids are not unique")
    for phase in plan["phases"]:
        bad.extend(
            f"{phase['id']} representative {ref} is not one of its members"
            for ref in phase["representative_refs"]
            if ref not in phase["member_refs"]
        )
    kinds = {claim["kind"] for claim in plan["claims"]}
    if plan["subject"]["sequenced"]:
        # the phase list IS the chronology: member ordinals must not
        # interleave across phases
        order = [int(ref.split("-")[1]) for phase in plan["phases"] for ref in phase["member_refs"]]
        if order != sorted(order):
            bad.append("a sequenced plan's phases interleave members; the phase list is not a chronology")
        bad.extend(f"a sequenced plan states the symmetric claim {kind}" for kind in sorted(kinds & _SYMMETRIC))
    else:
        # no chronology, no direction: nothing in the plan may say that
        # one family came before another or that something was added
        bad.extend(
            f"chronology is unsupported and the plan asserts the directional claim {kind}"
            for kind in sorted(kinds & _DIRECTIONAL)
        )
    dangling = unresolved(plan, snapshot)
    if dangling:
        bad.append(f"references outside the snapshot: {dangling[:5]}")
    return bad


# --- identities --------------------------------------------------------------


def identity(plan: dict) -> tuple[str, str]:
    """The DOCUMENT identity: the plan's canonical spelling -- which
    already names the snapshot sha, the planner, the engine and the
    settings -- with nothing about WHEN it was planned."""
    spelled = canonical({key: value for key, value in plan.items() if key != "planned_at"})
    return spelled, digest(spelled)


def request_identity(
    snapshot_sha256: str, planner: str, planner_version: int, engine: str, engine_version: str, settings: dict
) -> str:
    """The REQUEST identity, known before any model work -- every policy
    input that can change the output, the document FORMAT included, so
    a format change can never hand back yesterday's shape."""
    return digest(
        canonical(
            {
                "format": FORMAT_VERSION,
                "snapshot_sha256": snapshot_sha256,
                "planner": {"kind": planner, "version": planner_version},
                "engine": {"name": engine, "version": engine_version},
                "settings": dict(sorted(settings.items())),
            }
        )
    )


def settings_hash(settings: dict) -> str:
    return hashlib.sha256(canonical(dict(sorted(settings.items()))).encode("utf-8")).hexdigest()[:16]


PLANNERS = {GenerationHistoryPlanner.kind: GenerationHistoryPlanner}


# --- persistence and orchestration -------------------------------------------


def _verified_snapshot(conn, snapshot_id: int) -> tuple[dict, str]:
    """The snapshot document, PROVEN to hash to the identity it is stored
    under: a corrupted row must not produce a plan that claims hash X
    while having read different bytes."""
    from . import stories

    row = conn.execute(
        "SELECT document_json, document_sha256 FROM story_snapshot WHERE id = ?", (snapshot_id,)
    ).fetchone()
    if row is None:
        raise LookupError(f"no story snapshot {snapshot_id}")
    document = stories.parsed(row[0], f"story snapshot {snapshot_id}")
    if not stories.verify(document, row[1]):
        raise ValueError(f"story snapshot {snapshot_id} no longer hashes to its identity; refusing to plan from it")
    return document, row[1]


def plan_snapshot(conn, snapshot_id: int, planner: GenerationHistoryPlanner, now: float) -> PlanRef:
    """Plan one verified snapshot under one policy and persist the
    result -- or return the existing row when the request (or the
    canonical plan) already exists. The planner never sees the
    connection. The transaction is left open for the caller."""
    snapshot, snapshot_sha = _verified_snapshot(conn, snapshot_id)
    request = request_identity(
        snapshot_sha,
        planner.kind,
        planner.version,
        planner.similarity.name,
        planner.similarity.version,
        planner.settings,
    )
    held = conn.execute("SELECT id, document_sha256 FROM story_plan WHERE request_sha256 = ?", (request,)).fetchone()
    if held:
        return PlanRef(int(held[0]), held[1], True)
    plan = planner.plan(snapshot, snapshot_sha)
    wrong = validate_plan(plan, snapshot, snapshot_sha)
    if wrong:
        raise AssertionError(f"the planner produced an invalid plan: {wrong}")
    sha = identity(plan)[1]
    held = conn.execute("SELECT id FROM story_plan WHERE document_sha256 = ?", (sha,)).fetchone()
    if held:
        return PlanRef(int(held[0]), sha, True)
    plan["planned_at"] = now
    plan_id = int(
        conn.execute(
            "INSERT INTO story_plan(snapshot_id, format_version, planner, planner_version, similarity,"
            " similarity_version, settings_hash, request_sha256, document_json, document_sha256, created_at)"
            " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot_id,
                FORMAT_VERSION,
                planner.kind,
                planner.version,
                planner.similarity.name,
                planner.similarity.version,
                settings_hash(planner.settings),
                request,
                canonical(plan),
                sha,
                now,
            ),
        ).lastrowid
        or 0
    )
    return PlanRef(plan_id, sha, False)


def load_plan(conn, plan_id: int) -> dict:
    """The stored plan, RE-VERIFIED on read: it must still hash to its
    stored identity and still be a valid partition of its (equally
    re-verified) snapshot. A renderer consumes only what passes."""
    row = conn.execute(
        "SELECT snapshot_id, document_json, document_sha256 FROM story_plan WHERE id = ?", (plan_id,)
    ).fetchone()
    if row is None:
        raise LookupError(f"no story plan {plan_id}")
    from . import stories

    plan = stories.parsed(row[1], f"story plan {plan_id}")
    if identity(plan)[1] != row[2]:
        raise ValueError(f"story plan {plan_id} no longer hashes to its identity; refusing to serve it")
    snapshot, snapshot_sha = _verified_snapshot(conn, int(row[0]))
    wrong = validate_plan(plan, snapshot, snapshot_sha)
    if wrong:
        raise ValueError(f"story plan {plan_id} is no longer valid against its snapshot: {wrong}")
    return plan


@dataclasses.dataclass(frozen=True)
class PlanRequest:
    request_sha256: str
    plan_id: int | None
    job_id: int | None


def request_plan(conn, snapshot_id: int, planner_kind: str, engine, settings: dict | None, now: float) -> PlanRequest:
    """The planning service: validate the request, compute its identity
    WITHOUT loading weights, reuse a finished plan or a live job for the
    same request, otherwise queue durable work. Nothing expensive
    happens on the request thread."""
    from . import jobs

    maker = PLANNERS.get(planner_kind)
    if maker is None:
        raise ValueError(f"no planner named {planner_kind!r}; one of {', '.join(sorted(PLANNERS))}")
    exact = validated_settings(settings, maker.defaults)
    snapshot_sha = _verified_snapshot(conn, snapshot_id)[1]
    engine_name, engine_version = engine.identity()
    request = request_identity(snapshot_sha, maker.kind, maker.version, engine_name, engine_version, exact)
    # ONE writer lane for look-then-enqueue: two requests that both saw
    # "no plan, no job" would otherwise each queue the same model work.
    # The lane is claimed before the rechecks; the caller commits.
    conn.execute("BEGIN IMMEDIATE")
    held = conn.execute("SELECT id FROM story_plan WHERE request_sha256 = ?", (request,)).fetchone()
    if held:
        return PlanRequest(request, int(held[0]), None)
    for job_id, raw in conn.execute(
        "SELECT id, payload FROM job WHERE kind = 'story_plan' AND state IN ('queued', 'running')"
    ):
        if raw and json.loads(raw).get("request_sha256") == request:
            return PlanRequest(request, None, int(job_id))
    payload = {
        "request_sha256": request,
        "snapshot_id": snapshot_id,
        "planner": maker.kind,
        "settings": exact,
        "engine": {"selector": engine.selector, **engine.payload()},
    }
    job_id = jobs.submit(conn, "story_plan", now, payload=payload, items=[0])
    return PlanRequest(request, None, job_id)


def plan_item(conn, _item: int, payload: dict, now: float) -> None:
    """The job's one item. FIRST the request is re-proven: the engine
    spec the request was queued under, the planner and format the
    running code has, and the settings must still produce the SAME
    request identity as the payload names -- a deploy that changed the
    format, the planner, or a query policy between queue and run makes
    this job a stale ask, refused BEFORE any weights load. Then the
    engine loads (re-proving its own checkpoint), plans and persists;
    reuse by request identity makes a retried job idempotent."""
    engine_payload = dict(payload["engine"])
    selector = engine_payload.pop("selector")
    engine = LexicalEngine() if selector == LexicalEngine.selector else SemanticEngine(**engine_payload)
    maker = PLANNERS[payload["planner"]]
    snapshot_sha = _verified_snapshot(conn, int(payload["snapshot_id"]))[1]
    engine_name, engine_version = engine.identity()
    current = request_identity(
        snapshot_sha, maker.kind, maker.version, engine_name, engine_version, dict(payload["settings"])
    )
    if current != payload["request_sha256"]:
        raise ValueError(
            "this planning request no longer means what it meant when it was queued (the plan format,"
            " the planner or the engine's query policy changed); make the request again"
        )
    from . import prompts

    # The loaded engine behind the durable prompt-vector cache: frozen
    # texts are answered by text hash under the engine's own text space,
    # computed once where missing (db/prompts.py cached).
    planner = maker(prompts.cached(conn, engine, engine.load(), now), payload["settings"])
    plan_snapshot(conn, int(payload["snapshot_id"]), planner, now)
