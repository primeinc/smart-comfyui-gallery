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
            "pause",
            "lens_change",
            "burst",
            "equipment",
            "renditions",
            "video_clip",
            "time_basis",
            "disputed_time",
            "media_mix",
            "seen",
            "located",
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
            "pause",
            "lens_change",
            "burst",
            "exposure_range",
            "equipment",
            "renditions",
            "video_clip",
            "time_basis",
            "disputed_time",
            "media_mix",
            "seen",
            "located",
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
            for one in (member.get("capture") or {}).get("equipment") or []:
                if one["uuid"] == uuid:
                    return one["name"]
        return f"an artifact ({uuid[:8]})"

    def member_count(self, phase: dict) -> int:
        return len(phase["member_refs"])

    @property
    def filed(self) -> bool:
        """A file session: nobody held a camera, so the pause and the
        burst are spoken as saving, never as shooting."""
        return self.plan["subject"]["kind"] == "file_session"


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


def _pause(claim: dict, phase: dict, ctx: Context) -> str:
    gap = formatting.duration(claim["facts"]["gap_seconds"])
    if ctx.filed:
        return f"Nothing was saved for {gap} before this phase."
    return f"The camera was down for {gap} before this phase."


def _lens_change(claim: dict, phase: dict, ctx: Context) -> str:
    added = [ctx.artifact_name(uuid) for uuid in claim["facts"]["added"]]
    removed = [ctx.artifact_name(uuid) for uuid in claim["facts"]["removed"]]
    told = f"The lens changes here to {formatting.join_names(added)}"
    if removed:
        told += f", from {formatting.join_names(removed)}"
    return told + "."


def _burst(claim: dict, phase: dict, ctx: Context) -> str:
    facts = claim["facts"]
    if ctx.filed:
        told = f"{formatting.count(facts['frames'], 'file')} saved within {formatting.duration(facts['span_seconds'])}"
        if ctx.profile == "technical" and facts["frames_per_second"] is not None:
            told += f" ({facts['frames_per_second']:g} files per second)"
        return told + "."
    told = f"A burst of {formatting.count(facts['frames'], 'frame')} in {formatting.duration(facts['span_seconds'])}"
    if ctx.profile == "technical" and facts["frames_per_second"] is not None:
        told += f" ({facts['frames_per_second']:g} frames per second)"
    return told + "."


def _exposure_range(claim: dict, phase: dict, ctx: Context) -> str:
    facts = claim["facts"]
    parts = []
    if "iso" in facts:
        parts.append(f"ISO {formatting.span(facts['iso'], lambda v: f'{v:g}')}")
    if "f_number" in facts:
        parts.append(formatting.span(facts["f_number"], lambda v: f"f/{v:g}"))
    if "exposure_time" in facts:
        parts.append(formatting.span(facts["exposure_time"], formatting.shutter))
    if "focal_length" in facts:
        parts.append(formatting.span(facts["focal_length"], lambda v: f"{v:g} mm"))
    return "Exposure spans " + "; ".join(parts) + "."


def _equipment(claim: dict, phase: dict, ctx: Context) -> str:
    cameras = [ctx.artifact_name(uuid) for uuid in claim["facts"]["cameras"]]
    lenses = [ctx.artifact_name(uuid) for uuid in claim["facts"]["lenses"]]
    parts = []
    if cameras:
        parts.append(f"Shot with {formatting.join_names(cameras)}")
    if lenses:
        parts.append(("through " if cameras else "Shot through ") + formatting.join_names(lenses))
    return " ".join(parts) + "."


def _renditions(claim: dict, phase: dict, ctx: Context) -> str:
    facts = claim["facts"]
    return (
        f"{formatting.count(facts['acts'], 'photograph')} here {formatting.plural_verb(facts['acts'], 'is', 'are')}"
        f" kept as {formatting.count(facts['files'], 'file')}."
    )


def _video_clip(claim: dict, phase: dict, ctx: Context) -> str:
    facts = claim["facts"]
    verb = formatting.plural_verb(facts["clips"], "was", "were")
    told = f"{formatting.count(facts['clips'], 'video clip')} {verb} recorded here"
    if facts["seconds"] is not None:
        told += f", {formatting.duration(facts['seconds'])} in all"
    return told + "."


_BASIS_PHRASE = {
    "filename": "a stamp in the file name",
    "folder": "a dated folder",
    "mtime": "the filesystem's modified time",
    "btime": "the filesystem's created time",
    "embedded": "a date embedded in the file",
    "capture": "the camera's clock",
}


def _time_basis(claim: dict, phase: dict, ctx: Context) -> str:
    parts = [
        f"{formatting.count(n, 'file')} by {_BASIS_PHRASE.get(basis, basis)}"
        for basis, n in sorted(claim["facts"].items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return "Dated by " + formatting.join_names(parts) + "."


def _disputed_time(claim: dict, phase: dict, ctx: Context) -> str:
    n = claim["facts"]["members"]
    return f"The filesystem disputes the stated time of {formatting.count(n, 'file')} here."


_KIND_NOUNS = {
    "image": ("image", "images"),
    "animated_image": ("animation", "animations"),
    "video": ("video", "videos"),
    "audio": ("audio file", "audio files"),
    "document": ("document", "documents"),
}


def _media_mix(claim: dict, phase: dict, ctx: Context) -> str:
    parts = [
        formatting.count(n, *_KIND_NOUNS.get(kind, (kind, kind + "s")))
        for kind, n in sorted(claim["facts"].items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return "A mix of " + formatting.join_names(parts) + "."


def _member_ref(ordinal: int) -> str:
    return f"member-{ordinal + 1:03d}"  # db/planning.py _member_ref, the snapshot's own ref grammar


def _seen(claim: dict, phase: dict, ctx: Context) -> str:
    """What a model said, quoted from the FROZEN annotations of the
    claim's lowest member ref: one sentence, and how many members were
    described."""
    members = {_member_ref(one["ordinal"]): one for one in ctx.snapshot["members"]}
    spoken = None
    for ref in sorted(claim["evidence_refs"]):
        member = members.get(ref.split(":", 1)[0])
        captions = [one for one in (member or {}).get("annotations") or [] if one.get("kind") == "caption"]
        if captions:
            spoken = captions[0]["text"]
            break
    told = f"A model described {formatting.count(claim['facts']['members'], 'file')} here"
    if spoken:
        told += f'; of one it said "{spoken}"'
    told += "."
    if ctx.profile == "technical":
        told += " Captioned by " + formatting.join_names(claim["facts"]["models"]) + "."
    return told


def _located(claim: dict, phase: dict, ctx: Context) -> str:
    """Where the members happened, from the places they froze: one
    place, or several named. The frozen chain above a place -- "Lisbon,
    Portugal" -- is spelled from the snapshot's members, never from the
    live library, so the words match what the evidence held."""
    members = {_member_ref(one["ordinal"]): one for one in ctx.snapshot["members"]}
    above: dict[str, list[str]] = {}
    for ref in claim["evidence_refs"]:
        member = members.get(ref.split(":", 1)[0]) or {}
        chain = [one["name"] for one in ((member.get("place") or {}).get("chain") or [])]
        if chain:
            above.setdefault(chain[0], chain[1:])
    spelled = [
        name + (f" ({', '.join(above[name])})" if above.get(name) and " (" not in name else "")
        for name in claim["facts"]["places"]
    ]
    n = claim["facts"]["members"]
    if len(spelled) == 1:
        return f"In {spelled[0]}, by a person's word, for {formatting.count(n, 'file')} here."
    return f"In {formatting.join_names(spelled)}, by a person's word, across {formatting.count(n, 'file')} here."


#: The whole vocabulary. Resolution is by this mapping and nothing else.
REGISTRY: dict[str, typing.Callable[[dict, dict, Context], str]] = {
    "time_basis": _time_basis,
    "disputed_time": _disputed_time,
    "media_mix": _media_mix,
    "seen": _seen,
    "located": _located,
    "pause": _pause,
    "lens_change": _lens_change,
    "burst": _burst,
    "exposure_range": _exposure_range,
    "equipment": _equipment,
    "renditions": _renditions,
    "video_clip": _video_clip,
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
