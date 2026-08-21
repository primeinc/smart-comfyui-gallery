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

The similarity service is a Seam with the same discipline as the
snapshot: `embed(texts) -> vectors`, the planner supplying the FROZEN
prompt strings. The same engine that answers semantic retrieval sits
behind the production Adapter, but nothing here may look a vector up by
file: that would smuggle today's library back across the seam the
snapshot exists to close. A lexical bag-of-tokens Adapter gives tests a
deterministic oracle and any library a model-free fallback.

Identity is content-addressed exactly like the snapshot: the plan
document carries the snapshot's sha, the planner kind and version, the
similarity engine and version and the settings, and the plan's sha is
over that canonical document with `planned_at` excluded. The same
evidence under the same policy is ONE plan; a new policy coexists with
the old one instead of overwriting it.

V1 is deterministic end to end. Prompt identity alone never splits a
phase -- fifty wildcard expansions of one prompt are one creative
thread -- and day-precision evidence never yields sub-day chronology:
the planner says `unsupported` rather than inventing an order.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import typing

from .stories import canonical, digest

FORMAT_VERSION = 1

#: Precisions fine enough that event order is evidence of sequence.
_SEQUENCED = {"hour", "minute", "second", "subsecond"}


# --- the similarity Seam -----------------------------------------------------


class PromptSimilarity(typing.Protocol):
    name: str
    version: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


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
            row = [0.0] * len(vocabulary)
            for token in bag:
                row[index[token]] += 1.0
            made.append(row)
        return made


class ClipPromptSimilarity:
    """The production Adapter: the loaded semantic text encoder
    (vision/semantic) embedding the frozen prompt strings. Its identity
    is the encoder's model and checkpoint plus the package version, so
    a plan says which engine read the prompts."""

    def __init__(self, encoder, provider: str, checkpoint: str, package_version: str):
        self._encoder = encoder
        self.name = f"{provider}:{encoder.model_id}:{checkpoint}"
        self.version = package_version

    def embed(self, texts):
        return [[float(x) for x in self._encoder.encode_query(text)] for text in texts]


def pairwise_cosine(vectors: list[list[float]]) -> list[list[float]]:
    """Every pair's cosine, via the shared unit-normalisation (db/similarity
    normalise) and one matrix product. An all-zero vector has cosine 0
    with everything, including itself."""
    from .similarity import normalise

    if not vectors:
        return []
    width = max(len(row) for row in vectors)
    if width == 0:
        return [[0.0] * len(vectors) for _ in vectors]
    unit = normalise([row + [0.0] * (width - len(row)) for row in vectors])
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
    phase's representative prompt falls below `phase_threshold`. Prompt
    identity is never consulted, so wildcard expansions of one prompt
    stay one phase. Parameter and artifact changes are CLAIMS about a
    boundary, not boundaries themselves.
    """

    kind = "generation_history"
    version = 1
    defaults: typing.ClassVar[dict] = {"phase_threshold": 0.5}

    def __init__(self, similarity: PromptSimilarity, settings: dict | None = None):
        self.similarity = similarity
        self.settings = {**self.defaults, **(settings or {})}

    def plan(self, snapshot: dict, snapshot_sha256: str) -> dict:
        if snapshot.get("v") != 1:
            raise ValueError(f"this planner reads StorySnapshot v1, not v{snapshot.get('v')!r}")
        if snapshot["subject"]["event_kind"] != "generation_session":
            raise ValueError("GenerationHistoryPlanner plans generation sessions only")
        members = sorted(snapshot["members"], key=lambda one: one["ordinal"])
        refs = [_member_ref(one["ordinal"]) for one in members]
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

        prompts = [((one.get("generation") or {}).get("prompt") or "") for one in members]
        promptless = [ref for ref, text in zip(refs, prompts, strict=True) if not text.strip()]
        if promptless:
            unsupported.append({"kind": "prompt_evidence", "reason": "no frozen prompt", "member_refs": promptless})

        cosine = pairwise_cosine(self.similarity.embed(prompts)) if members else []
        groups = (
            _sequential_phases(cosine, self.settings["phase_threshold"])
            if sequenced
            else _family_phases(cosine, self.settings["phase_threshold"])
        )

        claims: list[dict] = []
        phases: list[dict] = []
        for number, indexes in enumerate(groups, start=1):
            phase_id = f"phase-{number:03d}"
            member_refs = [refs[i] for i in indexes]
            representative = _medoid(indexes, cosine)
            claim_refs = []
            if len(indexes) >= 2:
                inside = [cosine[a][b] for a in indexes for b in indexes if a < b]
                claim_refs.append(
                    _claim(
                        claims,
                        "prompt_similarity",
                        confidence=max(0.0, min([1.0, *inside])),
                        evidence=[f"{ref}:generation.prompt" for ref in member_refs],
                        facts={"relationship": "same_prompt_family", "min_pairwise_cosine": round(min(inside), 4)},
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
            label = f"Prompt family {number}" if not sequenced else f"Phase {number}"
            if number > 1:
                previous = groups[number - 2]
                changed = _artifact_change(members, previous, indexes, refs)
                if changed is not None:
                    claim_refs.append(_claim(claims, "artifact_change", 1.0, changed[0], changed[1]))
                    label += " · new artifacts"
                params = _parameter_change(members, previous, indexes, refs)
                if params is not None:
                    claim_refs.append(_claim(claims, "parameter_change", 1.0, params[0], params[1]))
            phases.append(
                {
                    "id": phase_id,
                    "member_refs": member_refs,
                    "representative_refs": [refs[representative]],
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


def _sequential_phases(cosine, threshold: float) -> list[list[int]]:
    """Consecutive runs: a member joins the running phase while its
    prompt stays within `threshold` of the phase's first prompt."""
    groups: list[list[int]] = []
    for i in range(len(cosine)):
        if groups and cosine[groups[-1][0]][i] >= threshold:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _family_phases(cosine, threshold: float) -> list[list[int]]:
    """Without chronology: connected components of the `>= threshold`
    graph, each listed in event order, families ordered by their first
    member -- a partition, not a sequence."""
    n = len(cosine)
    seen: set[int] = set()
    groups: list[list[int]] = []
    for start in range(n):
        if start in seen:
            continue
        stack, component = [start], []
        seen.add(start)
        while stack:
            i = stack.pop()
            component.append(i)
            for j in range(n):
                if j not in seen and cosine[i][j] >= threshold:
                    seen.add(j)
                    stack.append(j)
        groups.append(sorted(component))
    return groups


def _medoid(indexes: list[int], cosine) -> int:
    """The member closest to all the others -- ties to the earliest."""
    best, best_score = indexes[0], -2.0
    for i in indexes:
        score = sum(cosine[i][j] for j in indexes if j != i) / max(1, len(indexes) - 1)
        if score > best_score + 1e-12:
            best, best_score = i, score
    return best


def _artifact_uuids(member: dict) -> set[str]:
    return {one["uuid"] for one in ((member.get("generation") or {}).get("artifacts") or [])}


def _artifact_change(members, previous, current, refs):
    before = set().union(*(_artifact_uuids(members[i]) for i in previous))
    after = set().union(*(_artifact_uuids(members[i]) for i in current))
    if before == after:
        return None
    evidence = [f"{refs[i]}:generation.artifacts" for i in (*previous, *current)]
    return evidence, {"added": sorted(after - before), "removed": sorted(before - after)}


_PARAMS = ("sampler", "scheduler", "steps", "cfg", "denoise", "clip_skip", "width", "height")


def _parameter_change(members, previous, current, refs):
    def settled(indexes):
        return {key: sorted({str((members[i].get("generation") or {}).get(key)) for i in indexes}) for key in _PARAMS}

    before, after = settled(previous), settled(current)
    changed = {key: {"from": before[key], "to": after[key]} for key in _PARAMS if before[key] != after[key]}
    if not changed:
        return None
    evidence = [f"{refs[i]}:generation.{key}" for key in changed for i in (*previous, *current)]
    return evidence, {"changed": changed}


# --- resolution: a plan may only point inside its snapshot -------------------

_REF = re.compile(r"^(member-\d{3})(?::([a-z_]+(?:\.[a-z_]+)*))?$")


def unresolved(plan: dict, snapshot: dict) -> list[str]:
    """Every member_ref, representative_ref and evidence_ref that does
    NOT resolve inside the snapshot, plus every claim_ref that names no
    claim. Empty means the plan is closed over its evidence."""
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


# --- identity and persistence ------------------------------------------------


def identity(plan: dict) -> tuple[str, str]:
    """The sha is over the plan's canonical spelling -- which already
    names the snapshot sha, the planner, the similarity engine and the
    settings -- with nothing about WHEN it was planned."""
    spelled = canonical({key: value for key, value in plan.items() if key != "planned_at"})
    return spelled, digest(spelled)


def settings_hash(settings: dict) -> str:
    return hashlib.sha256(canonical(dict(sorted(settings.items()))).encode("utf-8")).hexdigest()[:16]


PLANNERS = {GenerationHistoryPlanner.kind: GenerationHistoryPlanner}


def plan_snapshot(conn, snapshot_id: int, planner: GenerationHistoryPlanner, now: float) -> PlanRef:
    """Plan one frozen snapshot under one policy and persist the result
    -- or return the existing row when the canonical plan already
    exists. The planner never sees the connection: it receives the
    document the snapshot module loads. The transaction is left open
    for the caller to commit."""
    row = conn.execute(
        "SELECT document_json, document_sha256 FROM story_snapshot WHERE id = ?", (snapshot_id,)
    ).fetchone()
    if row is None:
        raise LookupError(f"no story snapshot {snapshot_id}")
    snapshot = json.loads(row[0])
    plan = planner.plan(snapshot, row[1])
    dangling = unresolved(plan, snapshot)
    if dangling:
        raise AssertionError(f"the planner pointed outside its snapshot: {dangling[:5]}")
    sha = identity(plan)[1]
    held = conn.execute("SELECT id FROM story_plan WHERE document_sha256 = ?", (sha,)).fetchone()
    if held:
        return PlanRef(int(held[0]), sha, True)
    plan["planned_at"] = now
    plan_id = int(
        conn.execute(
            "INSERT INTO story_plan(snapshot_id, format_version, planner, planner_version, similarity,"
            " similarity_version, settings_hash, document_json, document_sha256, created_at)"
            " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot_id,
                FORMAT_VERSION,
                planner.kind,
                planner.version,
                planner.similarity.name,
                planner.similarity.version,
                settings_hash(planner.settings),
                canonical(plan),
                sha,
                now,
            ),
        ).lastrowid
        or 0
    )
    return PlanRef(plan_id, sha, False)


def load_plan(conn, plan_id: int) -> dict:
    row = conn.execute("SELECT document_json FROM story_plan WHERE id = ?", (plan_id,)).fetchone()
    if row is None:
        raise LookupError(f"no story plan {plan_id}")
    return json.loads(row[0])
