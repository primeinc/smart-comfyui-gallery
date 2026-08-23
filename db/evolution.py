"""The Generation Evolution Explorer's one Module: a READ-ONLY view of
a StoryPlan.

`load(conn, plan_id, ...)` assembles everything the page shows from
things that already exist and decides nothing:

    StoryPlan        phases, families, chronology supported or not,
                     representatives, claims              -- AUTHORITY
    StorySnapshot    members, prompt texts by role, parameters,
                     artifacts, lineage, dates             -- AUTHORITY
    prompt vectors   db/prompts.py, by FROZEN main-section text hash
                     under one (space, query policy)
    media vectors    derived_embedding, by the member's frozen uuid AND
                     frozen content_sha256 -- a replaced file's vector
                     is never substituted
    lineage          the snapshot's frozen parent/child edges

Over those frozen facts it MEASURES: cosine between consecutive members'
effective prompts and images (sequenced plans only -- O(n) pairs, never
a matrix), cosine between each member's prompt as written and as run,
cosine between each effective prompt and its image, and exact parameter
deltas. Every metric names one space and one policy; nothing blends
spaces into a score; every unavailable metric says why. Phase
boundaries are drawn where the plan put them, never where a cosine
dipped. No writes, no model loads, no embedding computation: the same
plan over the same cached vectors yields byte-identical JSON.
"""

from __future__ import annotations

import datetime
import itertools
import typing

from . import planning, prompts, stories

FORMAT_VERSION = 1

#: The recipe facts a transition compares, one per column of db/schema.sql
#: `generation`. Closed: a fact this list does not name is not compared,
#: and one it names that the table does not hold reads as None everywhere.
Parameter = typing.Literal["seed", "steps", "cfg", "denoise", "clip_skip", "sampler", "scheduler", "width", "height"]

#: What a transition can report as changed: a recipe parameter, or the
#: checkpoint. LoRAs are not here -- they are identities that come and go,
#: reported as added and removed rather than as before and after.
ChangedParameter = typing.Literal[Parameter, "model"]

_PARAMS: tuple[Parameter, ...] = typing.get_args(Parameter)


def _cosine(a, b) -> float:
    from .similarity import normalise

    unit = normalise([a, b])
    return round(float(unit[0] @ unit[1]), 4)


def _space(conn, provider: str | None, models_dir: str) -> dict:
    """The one (space, policy) every metric is measured in: the named
    provider's joint space and current query policy, or the first
    configured provider's. Nothing loads; an unminted space is reported,
    not created."""
    from vision import semantic

    from . import retrieval

    held: dict[str, str | int | None] = {
        "provider": None,
        "space_id": None,
        "space": None,
        "prompt_policy_hash": None,
        "unavailable": None,
    }
    if provider is not None:
        name, model, configured = retrieval.choice_for(conn, provider)  # exact; ambiguity refused
    else:
        try:
            choices = retrieval.choices(conn)
        except ValueError as refused:
            held["unavailable"] = str(refused)
            return held
        if len(choices) != 1:
            held["unavailable"] = "several semantic spaces are configured; name one with ?space=" + ", ".join(
                retrieval.spelled(*one) for one in choices
            )
            return held
        name, model, configured = choices[0]
    checkpoint = semantic.pin(name, models_dir, model, configured)
    held["provider"] = name
    held["space"] = semantic.space(name, model, checkpoint, 1).key
    held["prompt_policy_hash"] = semantic.policy_hash(name, model, checkpoint)
    found = prompts.space_of(conn, name, model, checkpoint)
    if found is None:
        held["unavailable"] = f"no vectors recorded under {held['space']} yet; run /jobs/embed and /jobs/embed_prompts"
    else:
        held["space_id"] = found[0]
    return held


def _media_vectors(conn, sid: int | None, members: list[dict]) -> dict[str, list[float]]:
    """Media vectors by member ref -- only a row computed from the
    FROZEN bytes (source_sha256 = the snapshot's content_sha256) for
    the frozen file identity counts."""
    import numpy as np

    if sid is None or not members:
        return {}
    held: dict[str, list[float]] = {}
    by_uuid = {one["file_uuid"]: one for one in members}
    marks = ",".join("?" for _ in by_uuid)
    for uuid, sha, blob in conn.execute(
        "SELECT ent.uuid, e.source_sha256, e.vector FROM derived_embedding e"
        " JOIN entity ent ON ent.id = e.file_id"
        f" WHERE e.space_id = ? AND ent.uuid IN ({marks})",
        (sid, *[bytes.fromhex(u) for u in by_uuid]),
    ):
        member = by_uuid.get(uuid.hex())
        if member is not None and sha == member["content_sha256"]:
            held[planning._member_ref(member["ordinal"])] = [float(x) for x in np.frombuffer(blob, dtype=np.float32)]
    return held


def _slugs(conn, members: list[dict]) -> dict[str, str | None]:
    """Each member's CURRENT address by its frozen uuid -- presentation
    only; None when the file is gone. Batched: the statement count is
    bounded however large the session."""
    held: dict[str, str | None] = {planning._member_ref(one["ordinal"]): None for one in members}
    by_uuid = {one["file_uuid"]: planning._member_ref(one["ordinal"]) for one in members}
    uuids = sorted(by_uuid)
    for start in range(0, len(uuids), 500):
        piece = uuids[start : start + 500]
        marks = ",".join("?" for _ in piece)
        for uuid, kind, slug in conn.execute(
            f"SELECT uuid, kind, slug FROM entity WHERE uuid IN ({marks})", [bytes.fromhex(u) for u in piece]
        ):
            if kind == "file":
                held[by_uuid[uuid.hex()]] = slug
    return held


def _prompt_ids(conn, hashes: set[str]) -> dict[str, int]:
    """Prompt rows by frozen text hash -- the one live lookup the page
    needs for its "prompts like this" link; absent when no row holds
    the text any more."""
    if not hashes:
        return {}
    wanted = sorted(hashes)
    marks = ",".join("?" for _ in wanted)
    return {
        row[0]: int(row[1])
        for row in conn.execute(f"SELECT text_hash, id FROM prompt WHERE text_hash IN ({marks})", wanted)
    }


def _prompt_view(frozen: dict | None, prompt_ids: dict[str, int]) -> dict | None:
    """One role's frozen prompt as the page reads it, or None when the
    snapshot froze none."""
    if frozen is None:
        return None
    return {
        "text": frozen["text"],
        "hash": frozen["hash"],
        "main": frozen["main"],
        "main_hash": frozen["main_hash"],
        "prompt_id": prompt_ids.get(frozen["main_hash"]),
    }


def _generation_facts(generation: dict | None) -> dict:
    generation = generation or {}
    artifacts = generation.get("artifacts") or []
    return {
        **{key: generation.get(key) for key in _PARAMS},
        "tool": generation.get("tool"),
        "model": next((a["name"] for a in artifacts if a["role"] == "checkpoint"), None),
        "loras": sorted(a["name"] for a in artifacts if a["role"] == "lora"),
        "lora_uuids": sorted(a["uuid"] for a in artifacts if a["role"] == "lora"),
        "lora_names": {a["uuid"]: a["name"] for a in artifacts if a["role"] == "lora"},
    }


def _changes(before: dict, after: dict) -> dict:
    """Exact deltas. LoRAs differ by frozen IDENTITY (uuid), spelled by
    their frozen names: two different files that share a name changed,
    one file renamed did not."""
    changed = {key: {"from": before[key], "to": after[key]} for key in (*_PARAMS, "model") if before[key] != after[key]}
    named = {**before["lora_names"], **after["lora_names"]}
    added = sorted(set(after["lora_uuids"]) - set(before["lora_uuids"]))
    removed = sorted(set(before["lora_uuids"]) - set(after["lora_uuids"]))
    return {
        **changed,
        "loras_added": [named[uuid] for uuid in added],
        "loras_removed": [named[uuid] for uuid in removed],
        "lora_uuids_added": added,
        "lora_uuids_removed": removed,
    }


def local_day(snapshot: dict) -> str | None:
    """The session's LOCAL calendar day as the gallery's day facet spells
    it, when the event has a wall clock -- an identity; the web adapter
    turns it into an address."""
    when = snapshot["subject"]["time"].get("local")
    if not when:
        return None
    return datetime.datetime.fromtimestamp(when[0], datetime.UTC).strftime("%Y-%m-%d")


def load(conn, plan_id: int, *, provider: str | None = None, models_dir: str = "") -> dict:
    """The EvolutionView for one plan. Raises LookupError for an unknown
    plan and stories.Corrupt for one that no longer verifies."""
    row = conn.execute("SELECT snapshot_id, document_sha256 FROM story_plan WHERE id = ?", (plan_id,)).fetchone()
    if row is None:
        raise LookupError(f"no story plan {plan_id}")
    plan = planning.load_plan(conn, plan_id)
    snapshot = stories.load_snapshot(conn, int(row[0]))
    members = sorted(snapshot["members"], key=lambda one: one["ordinal"])
    refs = [planning._member_ref(one["ordinal"]) for one in members]
    sequenced = bool(plan["subject"]["sequenced"])
    phase_of = {ref: phase["id"] for phase in plan["phases"] for ref in phase["member_refs"]}
    semantic = _space(conn, provider, models_dir)
    sid, policy = semantic["space_id"], semantic["prompt_policy_hash"]

    # prompts as the planner reads them: the MAIN section under the tool's grammar
    # prompts as the snapshot FROZE them: the main section read when the
    # evidence was frozen, never re-read by today's parser
    texts: dict[str, dict[str, dict | None]] = {}
    for ref, one in zip(refs, members, strict=True):
        generation = one.get("generation") or {}
        by_role = {p["role"]: p for p in generation.get("prompts") or []}
        held: dict[str, dict | None] = {}
        for role in ("effective", "original"):
            frozen = by_role.get(role)
            if frozen is None:
                held[role] = None
                continue
            main = planning.frozen_main(one, role)
            held[role] = {
                "text": frozen["text"],
                "hash": frozen["text_hash"],
                "main": main,
                "main_hash": frozen.get("main_hash") or prompts.text_hash(main),
            }
        texts[ref] = held
    hashes = {one["main_hash"] for held in texts.values() for one in held.values() if one}
    prompt_vectors = prompts.current_vectors(conn, sid, policy, sorted(hashes)) if sid is not None else {}
    media_vectors = _media_vectors(conn, sid, members)
    slugs = _slugs(conn, members)
    prompt_ids = _prompt_ids(conn, hashes)
    no_space = semantic["unavailable"]

    def prompt_vector(ref: str, role: str):
        held = texts[ref][role]
        if held is None:
            return None, f"no frozen {role} prompt"
        if no_space:
            return None, no_space
        vector = prompt_vectors.get(held["main_hash"])
        if vector is None:
            return (
                None,
                "no current vector for the frozen prompt text under this space and policy; run /jobs/embed_prompts",
            )
        return vector, None

    def media_vector(ref: str):
        if no_space:
            return None, no_space
        vector = media_vectors.get(ref)
        if vector is None:
            return None, "no media vector computed from the frozen bytes under this space; run /jobs/embed"
        return vector, None

    def metric(name: str, a, why_a, b, why_b) -> dict:
        if a is None or b is None:
            return {name: None, f"{name}_unavailable": why_a or why_b}
        return {name: _cosine(a, b)}

    facts = {ref: _generation_facts(one.get("generation")) for ref, one in zip(refs, members, strict=True)}
    out_members = []
    for ref, one in zip(refs, members, strict=True):
        effective, why_e = prompt_vector(ref, "effective")
        original, why_o = prompt_vector(ref, "original")
        image, why_i = media_vector(ref)
        held = texts[ref]
        slug = slugs[ref]
        out_members.append(
            {
                "ref": ref,
                "phase_ref": phase_of.get(ref),
                "media": {
                    "uuid": one["file_uuid"],
                    "name": one["name"],
                    "kind": one["media_kind"],
                    "content_sha256": one["content_sha256"],
                    "slug": slug,
                },
                "occurrence": one.get("occurrence"),
                "prompt": {role: _prompt_view(held[role], prompt_ids) for role in ("effective", "original")},
                "generation": {key: value for key, value in facts[ref].items() if key != "lora_names"},
                "metrics": {
                    **metric("original_effective_cosine", original, why_o, effective, why_e),
                    **metric("text_image_cosine", effective, why_e, image, why_i),
                },
            }
        )

    transitions = []
    if sequenced:
        for before, after in itertools.pairwise(refs):
            a_prompt, why_ap = prompt_vector(before, "effective")
            b_prompt, why_bp = prompt_vector(after, "effective")
            a_image, why_ai = media_vector(before)
            b_image, why_bi = media_vector(after)
            transitions.append(
                {
                    "from": before,
                    "to": after,
                    "phase_boundary": phase_of.get(before) != phase_of.get(after),
                    **metric("prompt_cosine", a_prompt, why_ap, b_prompt, why_bp),
                    **metric("visual_cosine", a_image, why_ai, b_image, why_bi),
                    "changes": _changes(facts[before], facts[after]),
                }
            )

    by_uuid = {one["file_uuid"]: ref for ref, one in zip(refs, members, strict=True)}
    lineage = []
    seen = set()
    for ref, one in zip(refs, members, strict=True):
        for edge in (one.get("lineage") or {}).get("parents") or []:
            key = (edge["uuid"], ref, edge["kind"])
            if key not in seen:
                seen.add(key)
                lineage.append({"parent": by_uuid.get(edge["uuid"], edge["uuid"]), "child": ref, "kind": edge["kind"]})
        for edge in (one.get("lineage") or {}).get("children") or []:
            key = (ref, edge["uuid"], edge["kind"])
            if key not in seen and edge["uuid"] not in by_uuid:
                seen.add(key)
                lineage.append({"parent": ref, "child": edge["uuid"], "kind": edge["kind"]})

    claims = {claim["id"]: claim for claim in plan["claims"]}
    phases = [
        {
            "id": phase["id"],
            "label": phase["label_hint"],
            "member_refs": list(phase["member_refs"]),
            "representative_refs": list(phase["representative_refs"]),
            "claims": [
                {"id": cid, "kind": claims[cid]["kind"], "facts": claims[cid]["facts"]} for cid in phase["claim_refs"]
            ],
        }
        for phase in plan["phases"]
    ]
    return {
        "v": FORMAT_VERSION,
        "plan": {
            "id": plan_id,
            "sha256": row[1],
            "format": plan["v"],
            "sequenced": sequenced,
            "unsupported": plan["unsupported"],
            "label": plan["subject"]["label_hint"],
        },
        "snapshot": {
            "sha256": plan["snapshot_sha256"],
            "time": snapshot["subject"]["time"],
            "members": len(members),
            "subject": snapshot["subject"]["event_kind"],
        },
        "semantic": semantic,
        "phases": phases,
        "members": out_members,
        "transitions": transitions,
        "lineage": lineage,
        "identities": {"local_day": local_day(snapshot)},
    }
