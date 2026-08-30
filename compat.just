# The consumer compatibility suite. Answers one question by execution: the
# minimum durable state this application must keep after an expensive face
# observation so every supported consumer can still be served.

# Each lane writes machine-generated evidence under compat/generated/ and is
# red until it earns green. `run` chains them; `closure` is the only gate that
# reports green.

# ITS OWN ENVIRONMENT: compat/ pulls facexlib, diffusers, torchaudio and about
# thirty transitive packages, and `uv sync` removes whatever a group does not
# declare. Two syncs against one environment destroy each other's work.

# So everything here runs from .venv-compat. `UV_PROJECT_ENVIRONMENT` with a
# relative path resolves against the workspace root, and uv creates the
# environment when it is absent (uv concepts/projects/config.md:162-169).

# The dev group is synced alongside: `check` runs the project's own ruff, ty
# and pyrefly over compat/, at the versions pyproject pins.

# `set windows-shell` is repeated because each module uses its own settings
# (just README.md:3649-3650). Recipes run from this file's directory, the
# repository root (just README.md:3687-3688), so every path is repo-relative.

set windows-shell := ["bash", "-cu"]
set script-interpreter := ['bash', '-euo', 'pipefail']

export UV_PROJECT_ENVIRONMENT := ".venv-compat"

python := if os_family() == 'windows' { './.venv-compat/Scripts/python.exe' } else { './.venv-compat/bin/python' }

# Build or refresh compat's own environment, leaving .venv untouched
sync:
    uv sync --group compat --group dev

# Every lane, in order, and none of them stops the rest. A prerequisite may
# block the cases that depend on it and may not stand in for their result.

# Each lane's exit code is recorded in `lanes.json`, which the ledger reads to
# mark a cell BLOCKED with the lane that blocked it. This recipe exits on the
# closure gate, never on a lane.
[script]
run: sync
    cd "$(git rev-parse --show-toplevel)"
    printf '{\n' > compat/generated/lanes.json
    first=1
    for lane in check proc-attack weights weights-attack pins citations hf population population-attack vendor acceptance producer union cases observe-attack reconcile reconcile-attack matrices attack closure-attack selftest staleness answer missing; do
        echo "=== compat: $lane ==="
        # `|| code=$?`, never `if ! cmd; then code=$?`: inside that `then` the
        # status belongs to the negation and is always 0. The left of `||` is
        # exempt from errexit and keeps the real code.
        code=0
        just compat "$lane" || code=$?
        [ "$first" = 1 ] || printf ',\n' >> compat/generated/lanes.json
        first=0
        printf '  "%s": %s' "$lane" "$code" >> compat/generated/lanes.json
        [ "$code" = 0 ] || echo "!! lane $lane exited $code; continuing so the rest still produce evidence"
    done
    printf '\n}\n' >> compat/generated/lanes.json
    echo "=== compat: ledger ==="
    just compat ledger
    echo "=== compat: closure ==="
    just compat closure

# The nine-cell ledger, one row per declared consumer, from this run only
ledger:
    {{ python }} -m compat.harness.ledger

# The ONLY gate that reports green. Reads the ledger; every cell must be
# VERIFIED
closure:
    {{ python }} -m compat.harness.closure

# Every lane below spawns git through the one runner at `proc.py`, the single
# exemption from the ban on spawning. It runs first and under attack, against
# a child holding an inherited output handle in a grandchild.

# Attack the process runner every other lane depends on
proc-attack:
    {{ python }} proc_attack.py

# Nothing here is vendored, and an absent weight is work to do rather than a
# reason a case cannot run. Sources live in `[provisioned]` with the citation
# that establishes each.

# Download every required artifact that is not on this machine
weights:
    {{ python }} -m compat.harness.provision

# Everything downstream reads files this lane put on disk, so a provisioner
# that accepts the wrong bytes hands `pins` a digest to attest.

# Attack the provisioner
weights-attack:
    {{ python }} -m compat.harness.provision_attack

# Three independent layers, none excusing the others: a clone at the right
# commit says nothing about what the interpreter imports, and neither says
# which model files computed the numbers.

# Clones, installed wheel bytes and weight files against what the manifest pins
pins:
    {{ python }} -m compat.harness.provenance

# Records every field the producer emits, including the ones nothing reads --
# precisely the ones at risk of being dropped because no column names them.

# One real observation pass, inventoried by shape, dtype and byte cost
producer:
    {{ python }} -m compat.producers.inventory

# One producer is one pack's opinion of what a face observation is. The
# storage contract answers for the union of what every provider emits, or a
# schema shaped around one pack proves nothing.

# Every producer's keys, unioned
union:
    {{ python }} -m compat.producers.union

# Resolved by machine rather than read again more carefully: a manifest value
# is a citation into a pinned commit, and it is checked there.

# Placed before the expensive lanes: a manifest defect found after the case
# shards have run costs a second GPU pass.

# Every path:line in the manifest, resolved against the pinned commit
citations:
    {{ python }} -m compat.harness.citations

# Every weight citation, resolved against the Hugging Face revision it names.
# The lfs pointer at that revision carries the content digest, so this attests
# a 260 MB file without holding a byte of it.

# A citation naming no revision, a revision this clone lacks, or a survey that
# finds no citations at all, is red.
hf:
    {{ python }} -m compat.harness.hf_check

# Shards split the lanes across processes and merge one evidence file,
# recomputing the population from the frozen manifest -- so a dead shard shows
# up as unexercised consumers and a red gate, never a smaller green population.

# Baseline, replay and ablation for every consumer that has a runner
cases:
    {{ python }} -m compat.harness.sharded

# The population is derived from the pinned consumer source, never declared:
# every declaration available to read is smaller than the truth. An UNRESOLVED
# variant is red.

# Every reachable face-model variant, read out of the pinned source
population:
    {{ python }} -m compat.harness.population

# Controls over real committed fixtures, plus mutations that each reintroduce
# one defect. A control that has never failed is not known to discriminate.

# Attack static discovery before any count it produces is believed
population-attack:
    {{ python }} -m compat.harness.population_attack

# An observer that cannot see a load may not be quoted on absence, so its
# coverage is measured rather than assumed, against a native open, a child
# process, an aliased reference and a real inference session.

# Attack the dynamic observer; it is a completeness oracle or it is nothing
observe-attack:
    {{ python }} -m compat.harness.observe_attack

# The two views against each other. An artifact the run opened and discovery
# never named is a hole in discovery; one discovered and never observed is a
# variant nothing exercised.

# Static discovery against what the run actually opened
reconcile:
    {{ python }} -m compat.harness.reconcile

# The reconciler decides what an unobserved artifact is allowed to mean, so
# it is the one lane whose verdict can be wrong because its inputs disagree
# rather than because the tree does.

# Attack the reconciler
reconcile-attack:
    {{ python }} -m compat.harness.reconcile_attack

# Corruptions of the closure gate itself. The only lane that reports green has
# to be shown capable of reporting red.

# Corrupt the closure gate and assert it notices
closure-attack:
    {{ python }} -m compat.harness.closure_attack

# Derived from the ledger rather than typed. An audit that ends as a list of
# deferrals is not work delivered, so this is read beside `closure`.

# The missing-work report
missing:
    {{ python }} -m compat.harness.missing

# Reads only what the lanes above wrote, so a hand-typed conclusion cannot
# enter. A matrix reporting green over part of a population is the failure
# this lane exists to prevent.

# Generated views over the evidence, never hand-maintained
matrices:
    {{ python }} -m compat.harness.matrices

# A gate that has never gone red is not known to work. This corrupts pins,
# digests, weight paths and the evidence itself, then re-runs the executor and
# asserts what it writes is byte-identical.

# Attack the suite and prove its gates can fail
attack:
    {{ python }} -m compat.harness.attack

# Each upstream's own example input, resolved from its pinned commit and
# hashed. A consumer with no first-party input stays visibly unresolved; our
# photographs are never substituted.

# First-party sample inputs, per consumer, by sha256
vendor:
    {{ python }} -m compat.vendor.fixtures

# LAYER ONE. Our adapter matching a reference proves nothing until the
# reference is known to run. This executes a vendor's own entrypoint, on its
# own example input, with its own published checkpoints.

# Each vendor computes its conditioning before the first sampling step, so no
# base model is needed for the ID half. A consumer whose remaining half needs
# a diffusion checkpoint records both halves separately.

# The pinned upstream reproducing its own example
acceptance:
    {{ python }} -m compat.vendor.acceptance

# Deliberate corruptions across pin, weight, manifest, runner and application
# source, corpus, cache, vendor fixture, primitive, selection rule and stale
# evidence, plus a positive control that must report CLEAN.

# Corrupt the harness deliberately and assert it notices
selftest:
    {{ python }} -m compat.harness.selftest

# Evidence that does not name its inputs cannot be stale. This recomputes the
# identity over fixtures, pins, weights, runtime, manifest and our own runner
# source, and fails when the evidence answers an older question.

# Fail if the generated evidence is not current
staleness:
    {{ python }} -m compat.harness.identity

# Reads the evidence and states the minimum durable set, citing the case that
# proves each line. A primitive is in because an ablation removed it and a
# replay broke, and out because one did not.

# The minimum retained state, derived from the evidence
answer:
    {{ python }} -m compat.harness.answer

# The licence is cc-by-nc-nd-4.0, so nothing here copies or rewrites a file. A
# digest rather than a path keeps the evidence honest: a file edited in place
# under one name would otherwise invalidate every baseline against it.

# Index the corpus by path and sha256, never vendored
corpus:
    {{ python }} -m compat.corpus.index

# Regenerate the deterministic fixtures for the primitive tier
fixtures:
    {{ python }} -m compat.primitives.build

# compat/ is absent from [tool.ty.src] and [tool.pyrefly] project-includes, so
# the project-wide run neither checks it nor fails on imports .venv cannot
# see. compat/ty.toml and compat/pyrefly.toml carry the root's rules here.

# Both are invoked with an EXPLICIT config, never a bare path: `src.include`
# filters command-line paths, so a bare path reports "No python files found"
# and then "All checks passed!" -- a green over zero files.

# ruff, ty and pyrefly over compat/, from compat's own environment
check:
    {{ python }} -m ruff format --check compat/
    {{ python }} -m ruff check compat/
    {{ python }} -m ty check --config-file compat/ty.toml compat
    {{ python }} -m pyrefly check -c compat/pyrefly.toml
