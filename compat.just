# The consumer compatibility suite.
#
# Answers one question by execution: what is the minimum durable state this
# application must keep after an expensive face observation so that every
# supported downstream consumer can still be served without reopening the
# source file or re-running the model.
#
# Six lanes, and `run` chains them. Each writes machine-generated evidence
# under compat/generated/ and each is red until it earns green.
#
# ITS OWN ENVIRONMENT, and this is the part to read before changing anything.
# compat/ pulls facexlib, diffusers, torchaudio, consisid-eva-clip and about
# thirty transitive packages. Those must NOT live in .venv:
#
#   `uv sync` removes extraneous packages. With the compat group installed
#   into .venv, a plain `uv sync` uninstalls 36 of them and this suite stops
#   running -- while `uv sync --group compat` pushes 36 packages into an
#   application environment that never asked for them. Measured, not feared:
#   `uv sync --dry-run` reported exactly that. Two commands that destroy each
#   other's work is a trap, and this tree is worked on by more than one
#   process at a time.
#
# So everything here runs from .venv-compat, which `just compat sync` builds.
# `UV_PROJECT_ENVIRONMENT` with a relative path resolves against the workspace
# root, and uv creates the environment when it is absent
# (uv concepts/projects/config.md:162-169).
#
# The dev group is synced alongside, because `check` runs the project's own
# ruff, ty and pyrefly over compat/ and they must be the versions pyproject
# pins rather than whatever is nearest.
#
# `set windows-shell` is repeated here on purpose: each module uses its own
# settings (just README.md:3649-3650), so the root justfile's shell does not
# reach this file. Recipes here run from the directory containing this file,
# which is the repository root (just README.md:3687-3688), so every path
# below is repo-relative.

set windows-shell := ["bash", "-cu"]

export UV_PROJECT_ENVIRONMENT := ".venv-compat"

python := if os_family() == 'windows' { './.venv-compat/Scripts/python.exe' } else { './.venv-compat/bin/python' }

# Build or refresh compat's own environment, leaving .venv untouched
sync:
    uv sync --group compat --group dev

# Every lane in order. Stops at the first red
run: sync check pins citations vendor acceptance producer union cases matrices attack selftest staleness answer

# Three independent layers, and none excuses the others: a clone at the right
# commit says nothing about what the interpreter will import, and neither says
# anything about which model files computed the numbers.

# Clones, installed wheel bytes and weight files against what the manifest pins
pins:
    {{ python }} compat/harness/provenance.py

# Runs the producer over real corpus photographs and records every field it
# emits -- including the ones nothing currently reads, which are precisely the
# ones at risk of being dropped because no column names them.

# One real observation pass, inventoried by shape, dtype and byte cost
producer:
    {{ python }} -m compat.producers.inventory

# ONE producer is one pack's opinion of what a face observation is. The
# storage contract has to answer for the union of what every provider emits,
# or a schema shaped around antelopev2 scores perfectly and proves nothing.
# A producer that cannot run here is recorded with its reason, never dropped.

# Every producer's keys, unioned
union:
    {{ python }} -m compat.producers.union

# Red while any declared consumer is unexercised. That is deliberate: a
# consumer which disappears when nothing runs it is how a suite reports
# success it did not earn.

# Sixteen lanes hold six ONNX packs, torch, mediapipe and every memoised
# full-resolution frame. One process holding all of it died during model
# loading with exit 1 and no traceback. Shards split the lanes across
# processes and merge one evidence file, recomputing the population from the
# FROZEN manifest -- so a dead shard shows up as unexercised consumers and a
# red gate, never as a smaller green population.

# Eleven wrong values were found in this manifest by hand in one session --
# four recognition packs, two combine rules, two det-size ladders, a max side
# length, a preprocessing note and a mis-attributed loader. Every one was
# written from a careful read of the source that later contradicted it, so
# the citations are resolved by machine rather than read again more carefully.
#
# Placed BEFORE the expensive lanes on purpose: a manifest defect found after
# the case shards have run costs a second GPU pass, and that is how this gate
# came to exist.

# Every path:line in the manifest, resolved against the pinned commit
citations:
    {{ python }} -m compat.harness.citations

# Baseline, replay and ablation for every consumer that has a runner
cases:
    {{ python }} -m compat.harness.sharded

# Reads only what the lanes above wrote, so a hand-typed conclusion cannot
# enter. Red while the population is incomplete, for the same reason `cases`
# is: a matrix reporting green over part of a population is the failure this
# lane exists to prevent.

# Generated views over the evidence, never hand-maintained
matrices:
    {{ python }} -m compat.harness.matrices

# A gate that has never gone red is not known to work. This corrupts pins,
# blob digests, weight paths and the evidence itself and asserts the checks
# notice, then re-runs the whole executor and asserts what it writes is
# byte-identical. An attack the gate does NOT see is a red result: every green
# ever reported for that class was uninformative.

# Attack the suite and prove its gates can fail
attack:
    {{ python }} -m compat.harness.attack

# Each upstream's OWN example input, resolved from its pinned commit and
# hashed. A consumer with no first-party input is VENDOR_BASELINE_UNAVAILABLE
# and stays visibly unresolved; our photographs are never substituted.

# First-party sample inputs, per consumer, by sha256
vendor:
    {{ python }} -m compat.vendor.fixtures

# LAYER ONE. Our adapter matching a reference proves nothing until the
# reference itself is known to run. This executes a vendor's OWN entrypoint,
# on its OWN example input, with its OWN published checkpoints.
#
# Seven vendors run here: consisid, pulid_upstream, infiniteyou,
# ipadapter_upstream, uniportrait, photomaker_v2 and reactor. Each computes
# its conditioning entirely before the first sampling step, so no base model
# is needed for the ID half. What is still absent is the diffusion checkpoint
# the rest of the population conditions -- FLUX or SDXL -- which makes those
# consumers UNSUPPORTED, a fact about this box, never
# VENDOR_BASELINE_UNAVAILABLE, which would be a claim about upstream.
#
# PhotoMaker is split by that line rather than excluded by it: its
# `analyze_faces` and id_embeds stack run, its ID ENCODER does not, because
# that forward takes SDXL prompt_embeds. The row records both halves.

# The pinned upstream reproducing its own example
acceptance:
    {{ python }} -m compat.vendor.acceptance

# Twelve deliberate corruptions -- pin, weight, manifest, runner source,
# dropped primitive, selection rule, reference order, stale evidence, missing
# consumer, vendor fixture, faked acceptance, lost round-trip key -- plus a
# positive control that must report CLEAN. An attack the
# gate does not see is red: every green ever reported for it was
# uninformative.

# Corrupt the harness deliberately and assert it notices
selftest:
    {{ python }} -m compat.harness.selftest

# Evidence that does not name its inputs cannot be stale. This recomputes the
# identity over fixtures, pins, weights, runtime, manifest and OUR OWN runner
# source, and fails when generated/cases.json answers an older question.

# Fail if the generated evidence is not current
staleness:
    {{ python }} -m compat.harness.identity

# The whole point. Every other lane produces evidence; this reads it and
# states the minimum durable set, citing the case that proves each line.
# Nothing in it decides anything: a primitive is in because an ablation
# removed it and a replay broke, and out because one did not.

# The minimum retained state, derived from the evidence
answer:
    {{ python }} -m compat.harness.answer

# The licence is cc-by-nc-nd-4.0, so nothing here copies, resizes or rewrites
# a file. A digest rather than a path is what keeps the evidence honest: a
# file edited in place under the same name would otherwise invalidate every
# baseline recorded against it without changing one line of the index.

# Index the corpus by path and sha256, never vendored
corpus:
    {{ python }} -m compat.corpus.index

# Regenerate the deterministic fixtures for the primitive tier
fixtures:
    {{ python }} -m compat.primitives.build

# The same gates `just check` applies to the application, run over compat/
# from the environment that can resolve its imports.
#
# compat/ is deliberately absent from [tool.ty.src] and [tool.pyrefly]
# project-includes, so the project-wide run neither checks it nor fails on
# imports .venv cannot see. compat/ty.toml and compat/pyrefly.toml carry the
# root's rules at compat's scope and point both tools at .venv-compat.
#
# Both are invoked with an EXPLICIT config, never a bare path. Passing
# `compat/` to the root project fails silently and in the worst possible way:
# `src.include` filters command-line paths, so ty printed "No python files
# found under the given path(s)" and then "All checks passed!" -- a green over
# zero files. A planted type error survived it. That is the exact shape of
# false pass this suite exists to refuse, found in its own gate, and both
# checkers are now verified against a planted error rather than trusted.

# ruff, ty and pyrefly over compat/, from compat's own environment
check:
    {{ python }} -m ruff format --check compat/
    {{ python }} -m ruff check compat/
    {{ python }} -m ty check --config-file compat/ty.toml compat
    {{ python }} -m pyrefly check -c compat/pyrefly.toml
