"""One Claim kind, one wording Adapter -- a CLOSED registry.

Each Adapter receives one already-validated Claim from a StoryPlan and
a `Context` over the frozen snapshot (to spell an artifact's name from
its frozen uuid, never from today's database) and returns one sentence,
or None when the profile does not surface that kind. The registry is
the whole vocabulary: a kind absent here is refused, never resolved by
probing the filesystem for a template of that name.

Wording is deliberately conservative. The deterministic narrator's job
is to be CORRECT; a later LLM Adapter may rephrase what is said here
and may say nothing that is not. Every sentence maps mechanically to
one Claim's facts. Nothing here sequences: "changes here" and "compared
with the previous phase" describe the plan's own ordering of phases,
and are only emitted for sequenced plans -- a renderer that narrates
an unsequenced plan has no "previous".

Every sentence here is output bytes: a change to any of them, cosmetic
or semantic, bumps the package's POLICY_VERSION (story_renderers
__init__), and every render made under the old wording coexists with
the new one instead of impersonating it.
"""

from __future__ import annotations

import dataclasses
import typing

from . import formatting

#: Which claim kinds each profile surfaces. Profiles change emphasis,
#: never truth: a kind a profile omits is still in the plan, still
#: cited by the section's claim_refs, merely not worded.
PROFILES: dict[str, frozenset[str]] = {
    "memory": frozenset(
        {
            "prompt_similarity",
            "prompt_family",
            "prompt_shift",
            "artifact_change",
            "artifact_difference",
            "prompt_evidence_missing",
            "prompt_rewrite",
        }
    ),
    "technical": frozenset(
        {
            "prompt_similarity",
            "prompt_family",
            "prompt_shift",
            "artifact_change",
            "artifact_difference",
            "parameter_change",
            "parameter_difference",
            "seed_variation",
            "prompt_evidence_missing",
            "prompt_rewrite",
        }
    ),
    "compact": frozenset(),
}


@dataclasses.dataclass(frozen=True)
class Context:
    """What an Adapter may know: the frozen snapshot (by value), the
    plan, the profile. No connection, no clock, no model."""

    snapshot: dict
    plan: dict
    profile: str
    sequenced: bool

    def artifact_name(self, uuid: str) -> str:
        for member in self.snapshot["members"]:
            for artifact in (member.get("generation") or {}).get("artifacts") or []:
                if artifact["uuid"] == uuid:
                    return artifact["name"]
        return f"an artifact ({uuid[:8]})"

    def member_count(self, phase: dict) -> int:
        return len(phase["member_refs"])


def _prompt_similarity(claim: dict, phase: dict, ctx: Context) -> str:
    n = len(claim["evidence_refs"])  # the members whose prompts the claim is about
    told = f"The {formatting.count(n, 'image')} in this phase share closely related prompt wording."
    if ctx.profile == "technical":
        told += f" Minimum pairwise prompt similarity is {formatting.percent(claim['facts']['min_pairwise_cosine'])}."
    return told


def _prompt_family(claim: dict, phase: dict, ctx: Context) -> str:
    size = claim["facts"]["size"]
    if size == 1:
        return "This image's prompt matches no other prompt in the session."
    told = f"This family's {formatting.count(size, 'image')} share related prompt wording."
    if ctx.profile == "technical" and claim["facts"]["min_pairwise_cosine"] is not None:
        told += f" Minimum pairwise prompt similarity is {formatting.percent(claim['facts']['min_pairwise_cosine'])}."
    return told


def _prompt_shift(claim: dict, phase: dict, ctx: Context) -> str:
    told = "The prompt wording changes here compared with the previous phase."
    if ctx.profile == "technical":
        facts = claim["facts"]
        told += f" Similarity to the previous phase's prompt is {formatting.percent(facts['cosine'])}, below the"
        told += f" {formatting.percent(facts['threshold'])} threshold."
    return told


def _compared_with(phase: dict, ctx: Context) -> str:
    """The phase this one is compared against: in a sequenced plan the
    previous phase; without chronology, the family listed before it --
    named by its label, never by a word that asserts time."""
    phases = ctx.plan["phases"]
    position = next(i for i, one in enumerate(phases) if one["id"] == phase["id"])
    if ctx.sequenced:
        return "Compared with the previous phase, "
    return f"Compared with {phases[position - 1]['label_hint']}, "


def _artifact_change(claim: dict, phase: dict, ctx: Context) -> str:
    added = [ctx.artifact_name(uuid) for uuid in claim["facts"]["added"]]
    removed = [ctx.artifact_name(uuid) for uuid in claim["facts"]["removed"]]
    parts = []
    if added:
        verb = formatting.plural_verb(len(added), "appears", "appear")
        parts.append(f"{formatting.join_names(added)} {verb} in this group")
    if removed:
        verb = formatting.plural_verb(len(removed), "is", "are")
        parts.append(f"{formatting.join_names(removed)} {verb} not used")
    return _compared_with(phase, ctx) + "; ".join(parts) + "."


def _artifact_difference(claim: dict, phase: dict, ctx: Context) -> str:
    """Symmetric: which artifacts each family uses that the other does
    not. No direction is asserted, because the plan asserted none."""
    here = [ctx.artifact_name(uuid) for uuid in claim["facts"]["only_here"]]
    other = [ctx.artifact_name(uuid) for uuid in claim["facts"]["only_other"]]
    parts = []
    if here:
        parts.append(f"{formatting.join_names(here)} {formatting.plural_verb(len(here), 'is', 'are')} used only here")
    if other:
        verb = formatting.plural_verb(len(other), "is", "are")
        parts.append(f"{formatting.join_names(other)} {verb} used only there")
    return _compared_with(phase, ctx) + "; ".join(parts) + "."


def _parameter_difference(claim: dict, phase: dict, ctx: Context) -> str:
    differs = claim["facts"]["differs"]
    parts = []
    for key in sorted(differs):
        here = formatting.join_names(differs[key]["here"])
        other = formatting.join_names(differs[key]["other"])
        parts.append(f"{key} differs: {here} here, {other} there")
    return _compared_with(phase, ctx) + "; ".join(parts) + "."


def _parameter_change(claim: dict, phase: dict, ctx: Context) -> str:
    changed = claim["facts"]["changed"]
    parts = []
    for key in sorted(changed):
        was = formatting.join_names(changed[key]["from"])
        now = formatting.join_names(changed[key]["to"])
        parts.append(f"{key} differs: {was} there, {now} here")
    return _compared_with(phase, ctx) + "; ".join(parts) + "."


def _seed_variation(claim: dict, phase: dict, ctx: Context) -> str:
    return f"{formatting.count(claim['facts']['distinct_seeds'], 'distinct seed')} were used."


def _prompt_evidence_missing(claim: dict, phase: dict, ctx: Context) -> str:
    n = claim["facts"]["members"]
    return f"Prompt text is not available for {'one image' if n == 1 else formatting.count(n, 'image')} here."


def _prompt_rewrite(claim: dict, phase: dict, ctx: Context) -> str:
    """Written vs run, for the members the claim names. No direction in
    time: the original was the input to the generator's own expansion,
    a fact about each image, not an order among images."""
    n = claim["facts"]["members"]
    who = "one image" if n == 1 else formatting.count(n, "image")
    told = f"For {who} here, the prompt the generator ran differs substantially from the prompt as written."
    if ctx.profile == "technical":
        told += (
            f" Minimum similarity between written and run prompt is {formatting.percent(claim['facts']['min_cosine'])}."
        )
    return told


#: The whole vocabulary. Resolution is by this mapping and nothing else.
REGISTRY: dict[str, typing.Callable[[dict, dict, Context], str]] = {
    "prompt_similarity": _prompt_similarity,
    "prompt_family": _prompt_family,
    "prompt_shift": _prompt_shift,
    "artifact_change": _artifact_change,
    "artifact_difference": _artifact_difference,
    "parameter_change": _parameter_change,
    "parameter_difference": _parameter_difference,
    "seed_variation": _seed_variation,
    "prompt_evidence_missing": _prompt_evidence_missing,
    "prompt_rewrite": _prompt_rewrite,
}


def word(claim: dict, phase: dict, ctx: Context) -> str | None:
    """One sentence for one claim under one profile, or None when the
    profile does not surface the kind. An unknown kind is a refusal."""
    kind = claim["kind"]
    adapter = REGISTRY.get(kind)
    if adapter is None:
        raise ValueError(f"no wording is registered for claim kind {kind!r}; the vocabulary is {sorted(REGISTRY)}")
    if kind not in PROFILES[ctx.profile]:
        return None
    return adapter(claim, phase, ctx)
