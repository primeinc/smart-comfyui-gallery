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

FORMAT_VERSION = 7

#: Claim kinds that assert DIRECTION -- before/after, added/removed,
#: from/to. They exist only in a sequenced plan, where the phase list is
#: a chronology. An unsequenced plan states symmetric DIFFERENCES.
_DIRECTIONAL = frozenset({"prompt_shift", "artifact_change", "parameter_change", "pause", "lens_change"})
_SYMMETRIC = frozenset({"artifact_difference", "parameter_difference"})


def prompt_sections_grammar(tool: str | None) -> str:
    """The section grammar for a member's prompts, from its frozen tool
    name -- the same rule the live library applies (db/prompts.py
    grammar_for), repeated here so the planner imports no database
    module."""
    return "swarm" if (tool or "").lower().startswith("swarm") else "plain"


def frozen_main(member: dict, role: str) -> str:
    """The MAIN section of a member's prompt in one role, as the
    snapshot FROZE it (`main`, parsed when the evidence was frozen).
    A snapshot frozen before mains were frozen is read with the running
    parser -- the only reading it can have -- and says so by absence."""
    generation = member.get("generation") or {}
    by_role = {p["role"]: p for p in generation.get("prompts") or []}
    held = by_role.get(role)
    if held is None:
        text = generation.get("prompt") or "" if role == "effective" else ""
        return prompt_sections.main(text, prompt_sections_grammar(generation.get("tool"))) if text else ""
    if "main" in held:
        return held["main"]
    return prompt_sections.main(held["text"], prompt_sections_grammar(generation.get("tool")))


#: Precisions fine enough that event order is evidence of sequence.
_SEQUENCED = {"hour", "minute", "second", "subsecond"}


# --- settings: exact, fail-closed --------------------------------------------


#: Each setting's domain, by name: a threshold is a fraction, a pause is
#: a positive length of time, a burst gap a non-negative one.
_DOMAINS: dict[str, tuple[float, float, bool]] = {
    "phase_threshold": (0.0, 1.0, True),
    "pause_minutes": (0.0, float("inf"), False),
    "burst_seconds": (0.0, float("inf"), True),
}


def validated_settings(settings: dict | None, defaults: dict) -> dict:
    """Exactly the planner's own keys, each a finite number inside its
    domain -- not a bool, not a string. An unknown key would ride the
    identity while meaning nothing; a string would compare as a string.
    The same exact-shape doctrine CollectionRule paid for."""
    # defaults are spelled as floats exactly as a given value is: `30`
    # and `30.0` are one setting, and a request hashed under one must
    # find the plan its job persisted under the other
    held = {key: float(value) for key, value in defaults.items()}
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
        low, high, closed_low = _DOMAINS[key]
        inside = (low <= float(value) if closed_low else low < float(value)) and float(value) <= high
        if not inside:
            spelled = f"[{low:g}, {high:g}]" if closed_low else f"({low:g}, {high:g}]"
            raise ValueError(f"{key} lies in {spelled}, not {value!r}")
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

    if selector.split(":", 1)[0] not in semantic.PROVIDERS:
        known = ", ".join(sorted(semantic.PROVIDERS))
        raise ValueError(f"no similarity engine named {selector!r}; one of lexical, {known}")
    # EXACT: a provider with two configured models is an ambiguity to
    # refuse, never a first match to take (db/retrieval.py choice_for)
    provider, model, configured = retrieval.choice_for(conn, selector)
    checkpoint = semantic.pin(provider, models_dir, model, configured)
    if not semantic.immutable(provider, checkpoint):
        raise ValueError(
            f"{provider} {model} is configured at the mutable revision {configured!r} and nothing is"
            " provisioned locally to pin it to; run /jobs/embed first -- a plan cannot be queued"
            " under provenance that may move before the worker loads it"
        )
    return SemanticEngine(provider, model, checkpoint, models_dir)


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
    uses_similarity = True
    #: The main-section texts the planner compares are the SNAPSHOT's
    #: frozen reading (stories.py freezes `main` per role); a parser
    #: change reaches a plan only through a new snapshot.
    version = 10
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
        prompts = [frozen_main(one, "effective") for one in members]
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
            wrote, did = frozen_main(one, "original"), frozen_main(one, "effective")
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
            seen = [i for i in indexes if _captioned(members[i])]
            if seen:
                claim_refs.append(
                    _claim(
                        claims,
                        "seen",
                        1.0,
                        [f"{refs[i]}:annotations" for i in seen],
                        {"members": len(seen), "models": _caption_models([members[i] for i in seen])},
                    )
                )
            placed = [i for i in indexes if _placed(members[i])]
            if placed:
                claim_refs.append(
                    _claim(
                        claims,
                        "located",
                        1.0,
                        [f"{refs[i]}:place" for i in placed],
                        {"members": len(placed), "places": _place_names([members[i] for i in placed])},
                    )
                )
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

        # A session is grouped by TIME; its members may come from several
        # tools. The label names a tool only when every member agrees.
        tools = {(one.get("generation") or {}).get("tool") for one in members if one.get("generation")}
        tool = tools.pop() if len(tools) == 1 else None
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


class _NoSimilarity:
    """The capture planner compares clocks and optics, never prompts:
    its similarity identity is fixed so the plan does not depend on
    which engine the request happened to name."""

    name = "none"
    version = "1"

    def embed(self, texts):
        raise AssertionError("the capture planner embeds nothing")


def _act_groups(members: list[dict]) -> list[list[int]]:
    """Consecutive members sharing an act key are one act (the grouper
    orders renditions together); a member without one is its own act."""
    acts: list[list[int]] = []
    for i, one in enumerate(members):
        key = (one.get("occurrence") or {}).get("act_key")
        if acts and key is not None and (members[acts[-1][0]].get("occurrence") or {}).get("act_key") == key:
            acts[-1].append(i)
        else:
            acts.append([i])
    return acts


def _moment(member: dict) -> float | None:
    held = member.get("occurrence") or {}
    return held.get("instant_at") if held.get("instant_at") is not None else held.get("local_at")


def _equipment(member: dict, role: str) -> str | None:
    for one in (member.get("capture") or {}).get("equipment") or []:
        if one["role"] == role:
            return one["uuid"]
    return None


def _extent(values: list[float]) -> list[float] | None:
    return [min(values), max(values)] if values else None


class CaptureHistoryPlanner:
    """Phases of one capture session from frozen evidence alone: the
    camera's clock, optics and body, never a prompt.

    An ACT is one shutter press; its renditions (a RAW and its JPEG)
    are the members that share an act key, so every count here is in
    acts, never files. Chronology is the camera's subsecond clock, so a
    capture session is sequenced whenever every act's occurrence is
    finer than a day.

    Boundaries: the photographer put the camera down (a gap longer
    than `pause_minutes` -- a `pause` claim with the gap), or changed
    the lens (`lens_change`, the lens artifacts either side). Inside a
    phase: `burst` runs (three or more acts with every gap at most
    `burst_seconds`), the `exposure_range` the phase spans, which
    `equipment` was in hand, `renditions` (acts with more than one
    file) and `video_clip` members.
    """

    kind = "capture_history"
    version = 3
    uses_similarity = False
    defaults: typing.ClassVar[dict] = {"pause_minutes": 10, "burst_seconds": 2.0}

    def __init__(self, similarity=None, settings: dict | None = None):
        self.similarity = _NoSimilarity()
        self.settings = validated_settings(settings, self.defaults)

    def plan(self, snapshot: dict, snapshot_sha256: str) -> dict:
        if snapshot.get("v") != 1:
            raise ValueError(f"this planner reads StorySnapshot v1, not v{snapshot.get('v')!r}")
        if snapshot["subject"]["event_kind"] != "capture_session":
            raise ValueError("CaptureHistoryPlanner plans capture sessions only")
        members = sorted(snapshot["members"], key=lambda one: one["ordinal"])
        refs = [_member_ref(one["ordinal"]) for one in members]
        pause = float(self.settings["pause_minutes"]) * 60.0
        burst_gap = float(self.settings["burst_seconds"])
        unsupported: list[dict] = []
        sequenced = all((one.get("occurrence") or {}).get("precision") in _SEQUENCED for one in members) and all(
            _moment(one) is not None for one in members
        )
        if not sequenced:
            unsupported.append(
                {
                    "kind": "chronology",
                    "reason": "at least one member's occurrence is day-precision or absent;"
                    " event order is not evidence of sequence",
                }
            )
        acts = _act_groups(members)
        primary = [act[0] for act in acts]

        # phase boundaries: a pause, or a lens change, between consecutive acts
        groups: list[list[int]] = [[0]] if acts else []
        for k in range(1, len(acts)):
            previous, here = primary[k - 1], primary[k]
            gap = (_moment(members[here]) or 0.0) - (_moment(members[previous]) or 0.0) if sequenced else 0.0
            lens_before, lens_after = (
                _equipment(members[previous], "mounted_lens"),
                _equipment(members[here], "mounted_lens"),
            )
            changed = lens_before is not None and lens_after is not None and lens_before != lens_after
            if sequenced and (gap > pause or changed):
                groups.append([k])
            else:
                groups[-1].append(k)

        claims: list[dict] = []
        phases: list[dict] = []
        for number, act_indexes in enumerate(groups, start=1):
            member_indexes = [i for k in act_indexes for i in acts[k]]
            member_refs = [refs[i] for i in member_indexes]
            claim_refs: list[str] = []
            label = f"Phase {number}" if sequenced else "Capture session"
            if sequenced and number > 1:
                before, first = primary[groups[number - 2][-1]], primary[act_indexes[0]]
                gap = (_moment(members[first]) or 0.0) - (_moment(members[before]) or 0.0)
                if gap > pause:
                    claim_refs.append(
                        _claim(
                            claims,
                            "pause",
                            1.0,
                            [f"{refs[before]}:occurrence", f"{refs[first]}:occurrence"],
                            {"gap_seconds": round(gap, 3)},
                        )
                    )
                    label += " · after a pause"
                lens_before, lens_after = (
                    _equipment(members[before], "mounted_lens"),
                    _equipment(members[first], "mounted_lens"),
                )
                if lens_before is not None and lens_after is not None and lens_before != lens_after:
                    claim_refs.append(
                        _claim(
                            claims,
                            "lens_change",
                            1.0,
                            [f"{refs[before]}:capture.equipment", f"{refs[first]}:capture.equipment"],
                            {"added": [lens_after], "removed": [lens_before]},
                        )
                    )
                    label += " · new lens"
            if sequenced:
                run: list[int] = [act_indexes[0]]
                bursts: list[list[int]] = []
                for k in act_indexes[1:]:
                    gap = (_moment(members[primary[k]]) or 0.0) - (_moment(members[primary[run[-1]]]) or 0.0)
                    if gap <= burst_gap:
                        run.append(k)
                    else:
                        if len(run) >= 3:
                            bursts.append(run)
                        run = [k]
                if len(run) >= 3:
                    bursts.append(run)
                for run in bursts:
                    span = (_moment(members[primary[run[-1]]]) or 0.0) - (_moment(members[primary[run[0]]]) or 0.0)
                    claim_refs.append(
                        _claim(
                            claims,
                            "burst",
                            1.0,
                            [f"{refs[primary[k]]}:occurrence" for k in run],
                            {
                                "frames": len(run),
                                "span_seconds": round(span, 3),
                                "frames_per_second": round((len(run) - 1) / span, 2) if span > 0 else None,
                            },
                        )
                    )
                if bursts:
                    label += " · burst"
            with_capture = [primary[k] for k in act_indexes if members[primary[k]].get("capture")]
            facts: dict = {}
            for key in ("iso", "f_number", "exposure_time", "focal_length"):
                held = _extent(
                    [members[i]["capture"][key] for i in with_capture if members[i]["capture"].get(key) is not None]
                )
                if held is not None:
                    facts[key] = held
            if facts:
                claim_refs.append(
                    _claim(claims, "exposure_range", 1.0, [f"{refs[i]}:capture" for i in with_capture], facts)
                )
            cameras = sorted({uuid for i in with_capture if (uuid := _equipment(members[i], "captured_with"))})
            lenses = sorted({uuid for i in with_capture if (uuid := _equipment(members[i], "mounted_lens"))})
            if cameras or lenses:
                claim_refs.append(
                    _claim(
                        claims,
                        "equipment",
                        1.0,
                        [f"{refs[i]}:capture.equipment" for i in with_capture],
                        {"cameras": cameras, "lenses": lenses},
                    )
                )
            multi = [k for k in act_indexes if len(acts[k]) > 1]
            if multi:
                claim_refs.append(
                    _claim(
                        claims,
                        "renditions",
                        1.0,
                        [f"{refs[i]}:occurrence.act_key" for k in multi for i in acts[k]],
                        {"acts": len(multi), "files": sum(len(acts[k]) for k in multi)},
                    )
                )
            clips = [i for i in member_indexes if members[i].get("media_kind") == "video"]
            if clips:
                seconds = [members[i]["duration"] for i in clips if members[i].get("duration") is not None]
                claim_refs.append(
                    _claim(
                        claims,
                        "video_clip",
                        1.0,
                        [f"{refs[i]}:media_kind" for i in clips],
                        {"clips": len(clips), "seconds": round(sum(seconds), 3) if seconds else None},
                    )
                )
            seen = [i for i in member_indexes if _captioned(members[i])]
            if seen:
                claim_refs.append(
                    _claim(
                        claims,
                        "seen",
                        1.0,
                        [f"{refs[i]}:annotations" for i in seen],
                        {"members": len(seen), "models": _caption_models([members[i] for i in seen])},
                    )
                )
            placed = [i for i in member_indexes if _placed(members[i])]
            if placed:
                claim_refs.append(
                    _claim(
                        claims,
                        "located",
                        1.0,
                        [f"{refs[i]}:place" for i in placed],
                        {"members": len(placed), "places": _place_names([members[i] for i in placed])},
                    )
                )
            middle = act_indexes[len(act_indexes) // 2]
            phases.append(
                {
                    "id": f"phase-{number:03d}",
                    "member_refs": member_refs,
                    "representative_refs": [refs[primary[middle]]],
                    "label_hint": label,
                    "claim_refs": claim_refs,
                }
            )

        camera_names = {
            one["name"]
            for member in members
            for one in (member.get("capture") or {}).get("equipment") or []
            if one["role"] == "captured_with"
        }
        camera = camera_names.pop() if len(camera_names) == 1 else None
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
                "label_hint": f"{camera or 'camera'} session · {len(acts)} frames · {len(phases)} "
                + ("phases" if sequenced else "group"),
            },
            "phases": phases,
            "claims": claims,
            "unsupported": unsupported,
        }


class FileHistoryPlanner:
    """Phases of one file session from frozen evidence alone: what the
    files themselves said about when they were saved -- a stamped name,
    a dated folder, or only the filesystem -- never a camera, never a
    prompt.

    A screenshot run, a download batch, a scan folder. Chronology is
    the file's own claim at the precision it carries, so the session is
    sequenced whenever every member's occurrence is finer than a day.

    Boundaries: nothing was saved for longer than `pause_minutes` (a
    `pause` claim with the gap). Inside a phase: `burst` runs (three or
    more files saved with every gap at most `burst_seconds`), the
    `time_basis` the members' dates rest on, `disputed_time` when the
    filesystem contradicts what a name or folder claimed, `media_mix`
    when the phase holds more than one kind of media, and `video_clip`
    members.
    """

    kind = "file_history"
    version = 3
    uses_similarity = False
    defaults: typing.ClassVar[dict] = {"pause_minutes": 30, "burst_seconds": 5.0}

    def __init__(self, similarity=None, settings: dict | None = None):
        self.similarity = _NoSimilarity()
        self.settings = validated_settings(settings, self.defaults)

    def plan(self, snapshot: dict, snapshot_sha256: str) -> dict:
        if snapshot.get("v") != 1:
            raise ValueError(f"this planner reads StorySnapshot v1, not v{snapshot.get('v')!r}")
        if snapshot["subject"]["event_kind"] != "file_session":
            raise ValueError("FileHistoryPlanner plans file sessions only")
        members = sorted(snapshot["members"], key=lambda one: one["ordinal"])
        refs = [_member_ref(one["ordinal"]) for one in members]
        pause = float(self.settings["pause_minutes"]) * 60.0
        burst_gap = float(self.settings["burst_seconds"])
        unsupported: list[dict] = []
        sequenced = all((one.get("occurrence") or {}).get("precision") in _SEQUENCED for one in members) and all(
            _moment(one) is not None for one in members
        )
        if not sequenced:
            unsupported.append(
                {
                    "kind": "chronology",
                    "reason": "at least one member's occurrence is day-precision or absent;"
                    " event order is not evidence of sequence",
                }
            )

        groups: list[list[int]] = [[0]] if members else []
        for i in range(1, len(members)):
            gap = (_moment(members[i]) or 0.0) - (_moment(members[i - 1]) or 0.0) if sequenced else 0.0
            if sequenced and gap > pause:
                groups.append([i])
            else:
                groups[-1].append(i)

        claims: list[dict] = []
        phases: list[dict] = []
        for number, indexes in enumerate(groups, start=1):
            claim_refs: list[str] = []
            label = f"Phase {number}" if sequenced else "File session"
            if sequenced and number > 1:
                before, first = groups[number - 2][-1], indexes[0]
                gap = (_moment(members[first]) or 0.0) - (_moment(members[before]) or 0.0)
                claim_refs.append(
                    _claim(
                        claims,
                        "pause",
                        1.0,
                        [f"{refs[before]}:occurrence", f"{refs[first]}:occurrence"],
                        {"gap_seconds": round(gap, 3)},
                    )
                )
                label += " · after a pause"
            if sequenced:
                run: list[int] = [indexes[0]]
                bursts: list[list[int]] = []
                for i in indexes[1:]:
                    gap = (_moment(members[i]) or 0.0) - (_moment(members[run[-1]]) or 0.0)
                    if gap <= burst_gap:
                        run.append(i)
                    else:
                        if len(run) >= 3:
                            bursts.append(run)
                        run = [i]
                if len(run) >= 3:
                    bursts.append(run)
                for run in bursts:
                    span = (_moment(members[run[-1]]) or 0.0) - (_moment(members[run[0]]) or 0.0)
                    claim_refs.append(
                        _claim(
                            claims,
                            "burst",
                            1.0,
                            [f"{refs[i]}:occurrence" for i in run],
                            {
                                "frames": len(run),
                                "span_seconds": round(span, 3),
                                "frames_per_second": round((len(run) - 1) / span, 2) if span > 0 else None,
                            },
                        )
                    )
                if bursts:
                    label += " · burst"
            bases: dict[str, int] = {}
            with_occurrence = [i for i in indexes if members[i].get("occurrence")]
            for i in with_occurrence:
                basis = members[i]["occurrence"]["basis"]
                bases[basis] = bases.get(basis, 0) + 1
            if bases:
                claim_refs.append(
                    _claim(
                        claims,
                        "time_basis",
                        1.0,
                        [f"{refs[i]}:occurrence.basis" for i in with_occurrence],
                        dict(sorted(bases.items())),
                    )
                )
            disputed = [i for i in with_occurrence if members[i]["occurrence"].get("conflicts")]
            if disputed:
                claim_refs.append(
                    _claim(
                        claims,
                        "disputed_time",
                        1.0,
                        [f"{refs[i]}:occurrence.conflicts" for i in disputed],
                        {"members": len(disputed)},
                    )
                )
            kinds: dict[str, int] = {}
            for i in indexes:
                kind = members[i].get("media_kind") or "document"
                kinds[kind] = kinds.get(kind, 0) + 1
            if len(kinds) > 1:
                claim_refs.append(
                    _claim(
                        claims,
                        "media_mix",
                        1.0,
                        [f"{refs[i]}:media_kind" for i in indexes],
                        dict(sorted(kinds.items())),
                    )
                )
            clips = [i for i in indexes if members[i].get("media_kind") == "video"]
            if clips:
                seconds = [members[i]["duration"] for i in clips if members[i].get("duration") is not None]
                claim_refs.append(
                    _claim(
                        claims,
                        "video_clip",
                        1.0,
                        [f"{refs[i]}:media_kind" for i in clips],
                        {"clips": len(clips), "seconds": round(sum(seconds), 3) if seconds else None},
                    )
                )
            seen = [i for i in indexes if _captioned(members[i])]
            if seen:
                claim_refs.append(
                    _claim(
                        claims,
                        "seen",
                        1.0,
                        [f"{refs[i]}:annotations" for i in seen],
                        {"members": len(seen), "models": _caption_models([members[i] for i in seen])},
                    )
                )
            placed = [i for i in indexes if _placed(members[i])]
            if placed:
                claim_refs.append(
                    _claim(
                        claims,
                        "located",
                        1.0,
                        [f"{refs[i]}:place" for i in placed],
                        {"members": len(placed), "places": _place_names([members[i] for i in placed])},
                    )
                )
            middle = indexes[len(indexes) // 2]
            phases.append(
                {
                    "id": f"phase-{number:03d}",
                    "member_refs": [refs[i] for i in indexes],
                    "representative_refs": [refs[middle]],
                    "label_hint": label,
                    "claim_refs": claim_refs,
                }
            )

        overall: dict[str, int] = {}
        for one in members:
            basis = (one.get("occurrence") or {}).get("basis")
            if basis:
                overall[basis] = overall.get(basis, 0) + 1
        leading = max(overall, key=lambda k: (overall[k], k)) if overall else None
        word = _BASIS_WORDS.get(leading or "", "file")
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
                "label_hint": f"{word} session · {len(members)} files · {len(phases)} "
                + ("phases" if sequenced else "group"),
            },
            "phases": phases,
            "claims": claims,
            "unsupported": unsupported,
        }


#: How a session is named by the claim most of its members rest on.
_BASIS_WORDS = {
    "filename": "stamped-name",
    "folder": "dated-folder",
    "mtime": "filesystem",
    "btime": "filesystem",
    "embedded": "embedded-date",
    "capture": "camera",
}

#: The time bases a file occurrence may rest on (db/schema.sql
#: derived_media_occurrence.basis) and the media kinds a member may be.
_BASES = frozenset({"capture", "embedded", "filename", "folder", "mtime", "btime"})
_MEDIA_KINDS = frozenset({"image", "animated_image", "video", "audio", "document"})


def _captioned(member: dict) -> bool:
    return any(one.get("kind") == "caption" for one in member.get("annotations") or [])


def _placed(member: dict) -> bool:
    held = member.get("place")
    return isinstance(held, dict) and bool(held.get("chain"))


def _place_names(members: list[dict]) -> list[str]:
    """The leaf places the claim rests on, one entry per ENTITY, sorted:
    two Springfields are two places, spelled apart by what they are
    within when their names collide."""
    leaves: dict[str, list[dict]] = {}
    for member in members:
        chain = member["place"]["chain"]
        leaves.setdefault(chain[0]["uuid"], chain)
    names: dict[str, int] = {}
    for chain in leaves.values():
        names[chain[0]["name"]] = names.get(chain[0]["name"], 0) + 1
    spelled = []
    for chain in leaves.values():
        name = chain[0]["name"]
        if names[name] > 1 and len(chain) > 1:
            name = f"{name} ({chain[1]['name']})"
        spelled.append(name)
    return sorted(spelled)


def _caption_models(members: list[dict]) -> list[str]:
    """The models whose captions the claim rests on, sorted, once each."""
    return sorted(
        {one["model"][0] for member in members for one in member["annotations"] if one.get("kind") == "caption"}
    )


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

#: Three digits or more: the thousandth member is member-1000, and a
#: grammar that admitted only three digits would have refused it.
_REF = re.compile(r"^(member-\d{3,})(?::([a-z_]+(?:\.[a-z_]+)*))?$")


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


class StoryPlanFlat(typing.TypedDict):
    """A StoryPlan vocabulary through v3, whose `settings` is one flat
    set of keys.

    Spelled out because a bare dict holding an int beside five
    frozensets infers every value as their union, so
    `STORY_PLAN_V3["claims"] | frozenset(...)` -- the way each version
    is built from the one before -- reads as `|` between
    `int | frozenset` and a frozenset, and nothing downstream can be
    checked.
    """

    version: int
    planners: frozenset[str]
    claims: frozenset[str]
    unsupported: frozenset[str]
    settings: frozenset[str]
    subjects: frozenset[str]


class StoryPlan(typing.TypedDict):
    """A StoryPlan vocabulary from v4 on: `settings` is PER PLANNER.

    Two shapes rather than one with a union, because the union is a
    lie in both directions -- it says v3 might carry a dict and v5
    might carry a set, and then every construction below has to be
    read as if it could. The shape changed exactly once, at v4, which
    the comment there states; two names say that and one does not.

    A frozen version stays frozen: a v1 row written today still parses
    as v1, so both shapes are live for ever.
    """

    version: int
    planners: frozenset[str]
    claims: frozenset[str]
    unsupported: frozenset[str]
    settings: dict[str, frozenset[str]]
    subjects: frozenset[str]


#: The durable vocabulary of a StoryPlan v1 -- FROZEN. These constants
#: describe what a v1 document may contain and never track the running
#: code: a v1 row written today must still parse as v1 after a v2 exists,
#: or the "immutable historical artifact" is a row that rots. Adding a
#: claim kind, a planner, or a key is a v2.
STORY_PLAN_V1: StoryPlanFlat = {
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
STORY_PLAN_V2: StoryPlanFlat = {
    **STORY_PLAN_V1,
    "version": 2,
    "claims": STORY_PLAN_V1["claims"] | frozenset({"artifact_difference", "parameter_difference"}),
}

#: StoryPlan v3 -- FROZEN. v2 plus `prompt_rewrite`: the prompt the
#: generator ran differs substantially from the prompt as written.
STORY_PLAN_V3: StoryPlanFlat = {
    **STORY_PLAN_V2,
    "version": 3,
    "claims": STORY_PLAN_V2["claims"] | frozenset({"prompt_rewrite"}),
}

#: StoryPlan v4 -- FROZEN. v3 plus the capture vocabulary: the
#: `capture_history` planner over a `capture_session`, whose claims are
#: the camera's clock and optics -- a pause, a lens change, a burst, the
#: exposure range, the equipment in hand, renditions of one act, clips.
STORY_PLAN_V4: StoryPlan = {
    **STORY_PLAN_V3,
    "version": 4,
    "planners": STORY_PLAN_V3["planners"] | frozenset({"capture_history"}),
    "claims": STORY_PLAN_V3["claims"]
    | frozenset({"pause", "lens_change", "burst", "exposure_range", "equipment", "renditions", "video_clip"}),
    # settings are PER PLANNER from v4 on: a planner's own keys, exactly
    "settings": {
        "generation_history": STORY_PLAN_V3["settings"],
        "capture_history": frozenset({"pause_minutes", "burst_seconds"}),
    },
    "subjects": STORY_PLAN_V3["subjects"] | frozenset({"capture_session"}),
}
#: StoryPlan v5 -- FROZEN. v4 plus the file vocabulary: the
#: `file_history` planner over a `file_session`, whose claims are what
#: the files said about themselves -- the basis their dates rest on, a
#: filesystem that disputes a name, a mix of media kinds -- beside the
#: shared pause, burst and clip claims.
STORY_PLAN_V5: StoryPlan = {
    **STORY_PLAN_V4,
    "version": 5,
    "planners": STORY_PLAN_V4["planners"] | frozenset({"file_history"}),
    "claims": STORY_PLAN_V4["claims"] | frozenset({"time_basis", "disputed_time", "media_mix"}),
    "settings": {
        **STORY_PLAN_V4["settings"],
        "file_history": frozenset({"pause_minutes", "burst_seconds"}),
    },
    "subjects": STORY_PLAN_V4["subjects"] | frozenset({"file_session"}),
}
#: StoryPlan v6 -- FROZEN. v5 plus `seen`: what a captioning model said
#: about a phase's members, frozen in the snapshot's annotations. Every
#: planner may emit it; the facts are how many members carry a caption
#: and which models spoke.
STORY_PLAN_V6: StoryPlan = {
    **STORY_PLAN_V5,
    "version": 6,
    "claims": STORY_PLAN_V5["claims"] | frozenset({"seen"}),
}
#: StoryPlan v7 -- FROZEN. v6 plus `located`: where a phase's members
#: happened, from the place each member froze (a person's word; GPS
#: alone names no place). Facts are how many members carry a place and
#: the leaf names they carry.
STORY_PLAN_V7: StoryPlan = {
    **STORY_PLAN_V6,
    "version": 7,
    "claims": STORY_PLAN_V6["claims"] | frozenset({"located"}),
}
_ID = re.compile(r"^(phase|claim)-[0-9]{3,}$")
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


def _facts_valid_v4(kind: str, facts) -> bool:
    if not isinstance(facts, dict):
        return False
    keys = set(facts)
    if kind == "pause":
        return keys == {"gap_seconds"} and _number(facts["gap_seconds"]) and facts["gap_seconds"] > 0
    if kind == "lens_change":
        return keys == {"added", "removed"} and all(_strings(facts[k]) for k in keys) and bool(facts["added"])
    if kind == "burst":
        return (
            keys == {"frames", "span_seconds", "frames_per_second"}
            and _integer(facts["frames"], 3)
            and _number(facts["span_seconds"])
            and facts["span_seconds"] >= 0
            and (facts["frames_per_second"] is None or _number(facts["frames_per_second"]))
        )
    if kind == "exposure_range":
        return (
            bool(keys)
            and keys <= {"iso", "f_number", "exposure_time", "focal_length"}
            and all(
                isinstance(v, list) and len(v) == 2 and all(_number(x) for x in v) and v[0] <= v[1]
                for v in facts.values()
            )
        )
    if kind == "equipment":
        return (
            keys == {"cameras", "lenses"}
            and all(_strings(facts[k]) for k in keys)
            and bool(facts["cameras"] or facts["lenses"])
        )
    if kind == "renditions":
        return keys == {"acts", "files"} and _integer(facts["acts"], 1) and _integer(facts["files"], 2)
    if kind == "video_clip":
        return (
            keys == {"clips", "seconds"}
            and _integer(facts["clips"], 1)
            and (facts["seconds"] is None or (_number(facts["seconds"]) and facts["seconds"] >= 0))
        )
    return _facts_valid_v3(kind, facts)


def _counts(facts: dict, vocabulary: frozenset[str], minimum_keys: int) -> bool:
    return len(facts) >= minimum_keys and set(facts) <= vocabulary and all(_integer(v, 1) for v in facts.values())


def _facts_valid_v5(kind: str, facts) -> bool:
    if not isinstance(facts, dict):
        return False
    if kind == "time_basis":
        return _counts(facts, _BASES, 1)
    if kind == "disputed_time":
        return set(facts) == {"members"} and _integer(facts["members"], 1)
    if kind == "media_mix":
        return _counts(facts, _MEDIA_KINDS, 2)
    return _facts_valid_v4(kind, facts)


def _facts_valid_v6(kind: str, facts) -> bool:
    if not isinstance(facts, dict):
        return False
    if kind == "seen":
        return (
            set(facts) == {"members", "models"}
            and _integer(facts["members"], 1)
            and _strings(facts["models"])
            and bool(facts["models"])
        )
    return _facts_valid_v5(kind, facts)


def _facts_valid_v7(kind: str, facts) -> bool:
    if not isinstance(facts, dict):
        return False
    if kind == "located":
        return (
            set(facts) == {"members", "places"}
            and _integer(facts["members"], 1)
            and _strings(facts["places"])
            and bool(facts["places"])
        )
    return _facts_valid_v6(kind, facts)


def validate_story_plan_v7(plan) -> list[str]:
    """The exact grammar of a StoryPlan v7 document against the FROZEN
    v7 vocabulary."""
    return _validate_story_plan(plan, STORY_PLAN_V7, _facts_valid_v7)


def validate_story_plan_v6(plan) -> list[str]:
    """The exact grammar of a StoryPlan v6 document against the FROZEN
    v6 vocabulary."""
    return _validate_story_plan(plan, STORY_PLAN_V6, _facts_valid_v6)


def validate_story_plan_v5(plan) -> list[str]:
    """The exact grammar of a StoryPlan v5 document against the FROZEN
    v5 vocabulary."""
    return _validate_story_plan(plan, STORY_PLAN_V5, _facts_valid_v5)


def validate_story_plan_v4(plan) -> list[str]:
    """The exact grammar of a StoryPlan v4 document against the FROZEN
    v4 vocabulary."""
    return _validate_story_plan(plan, STORY_PLAN_V4, _facts_valid_v4)


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


def _validate_story_plan(plan, vocabulary: StoryPlanFlat | StoryPlan, facts_valid) -> list[str]:
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
        expected = vocabulary["settings"]
        if isinstance(expected, dict):
            expected = expected.get(planner["kind"], frozenset())
        settings_ok = _keys(planner["settings"], set(expected), set(), "planner.settings", bad)
        if settings_ok and "phase_threshold" in expected and not _unit(planner["settings"]["phase_threshold"]):
            bad.append("planner.settings.phase_threshold is not a number in [0, 1]")
        if settings_ok and "pause_minutes" in expected:
            held = planner["settings"]
            if not (_number(held["pause_minutes"]) and held["pause_minutes"] > 0):
                bad.append("planner.settings.pause_minutes is not a positive number")
            if not (_number(held["burst_seconds"]) and held["burst_seconds"] >= 0):
                bad.append("planner.settings.burst_seconds is not a non-negative number")
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


_GRAMMARS = {
    1: validate_story_plan_v1,
    2: validate_story_plan_v2,
    3: validate_story_plan_v3,
    4: validate_story_plan_v4,
    5: validate_story_plan_v5,
    6: validate_story_plan_v6,
    7: validate_story_plan_v7,
}


def validate_story_plan(plan) -> list[str]:
    """Dispatch on the document's OWN version: a v1 row is judged by the
    frozen v1 grammar however many versions exist later. A document
    with no parsable version, or a version nobody defined, is invalid."""
    version = plan.get("v") if isinstance(plan, dict) else None
    grammar = _GRAMMARS.get(version) if isinstance(version, int) and not isinstance(version, bool) else None
    if grammar is None:
        return [f"no StoryPlan grammar for version {version!r}; known: {sorted(_GRAMMARS)}"]
    return grammar(plan)


def validate_stored_plan(plan: dict, snapshot: dict, snapshot_sha256: str) -> list[str]:
    """What makes a STORED plan readable, whatever planner policy wrote
    it: its own version's frozen grammar; the snapshot it names; an
    exact partition (every member exactly once); representatives inside
    their phase; unique ids; every reference inward. Nothing here is a
    policy the producer may later tighten: a v1 plan planner v4 wrote
    stays readable after planner v5 decided it would no longer write
    such a thing -- and planner v3 wrote sequenced plans whose gap
    members sat in a phase of their own, interleaving ordinals, so
    contiguity is the CURRENT producer's rule, not history's."""
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
    dangling = unresolved(plan, snapshot)
    if dangling:
        bad.append(f"references outside the snapshot: {dangling[:5]}")
    return bad


def validate_current_plan(plan: dict, snapshot: dict, snapshot_sha256: str) -> list[str]:
    """What the CURRENT producer must satisfy before a plan is persisted:
    everything a stored plan must, plus today's invariants -- direction
    only with a chronology, symmetry only without one."""
    bad = validate_stored_plan(plan, snapshot, snapshot_sha256)
    if bad:
        return bad
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
    return bad


#: The producer's contract, by its shorter name.
validate_plan = validate_current_plan


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


PLANNERS = {
    GenerationHistoryPlanner.kind: GenerationHistoryPlanner,
    CaptureHistoryPlanner.kind: CaptureHistoryPlanner,
    FileHistoryPlanner.kind: FileHistoryPlanner,
}

#: What `plan_snapshot` is handed. Beside the registry deliberately: the
#: two lists are the same list, and one drifting from the other is how a
#: signature comes to name one planner while three reach it.
Planner = GenerationHistoryPlanner | CaptureHistoryPlanner | FileHistoryPlanner


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
        raise stories.Corrupt(
            f"story snapshot {snapshot_id} no longer hashes to its identity; refusing to plan from it"
        )
    return document, row[1]


def plan_snapshot(conn, snapshot_id: int, planner: Planner, now: float) -> PlanRef:
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
    wrong = validate_current_plan(plan, snapshot, snapshot_sha)
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
        raise stories.Corrupt(f"story plan {plan_id} no longer hashes to its identity; refusing to serve it")
    snapshot, snapshot_sha = _verified_snapshot(conn, int(row[0]))
    # STORED validity: the plan's own frozen grammar and structure, never
    # today's producer policy -- a historical plan is read as history
    wrong = validate_stored_plan(plan, snapshot, snapshot_sha)
    if wrong:
        raise stories.Corrupt(f"story plan {plan_id} is no longer valid against its snapshot: {wrong}")
    return plan


@dataclasses.dataclass(frozen=True)
class PlanRequest:
    request_sha256: str
    plan_id: int | None
    job_id: int | None


def _engine_identity(maker, engine) -> tuple[str, str]:
    """The similarity identity a request is hashed under: the engine's
    for a planner that compares prompts, the planner's own fixed
    `none` otherwise -- the SAME identity plan_snapshot persists under.
    Hashing a no-similarity request with the engine the caller happened
    to name made every such request miss its own plan and queue the
    job again, forever."""
    if maker.uses_similarity:
        return engine.identity()
    held = _NoSimilarity()
    return held.name, held.version


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
    engine_name, engine_version = _engine_identity(maker, engine)
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
    engine_name, engine_version = _engine_identity(maker, engine)
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
    # computed once where missing (db/prompts.py cached). A planner that
    # compares no prompts never loads weights.
    if maker.uses_similarity:
        planner = maker(prompts.cached(conn, engine, engine.load(), now), payload["settings"])
    else:
        planner = maker(None, payload["settings"])
    plan_snapshot(conn, int(payload["snapshot_id"]), planner, now)
