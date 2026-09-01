set windows-shell := ["bash", "-cu"]
set script-interpreter := ['bash', '-euo', 'pipefail']

export UV_PROJECT_ENVIRONMENT := ".venv-compat"

python := if os_family() == 'windows' { './.venv-compat/Scripts/python.exe' } else { './.venv-compat/bin/python' }

sync:
    uv sync --group compat --group dev

[script]
run: sync
    cd "$(git rev-parse --show-toplevel)"
    mkdir -p compat/generated
    # The VERDICT goes first, before any lane runs. A crash anywhere after this
    # point must leave no closure.json rather than the last run's, which read as
    # this run's answer because it carried no identity to contradict it.
    rm -f compat/generated/lanes.json compat/generated/lanes.raw compat/generated/closure.json
    for lane in check proc-attack weights weights-attack pins citations hf population population-attack vendor acceptance producer union cases observe-attack reconcile reconcile-attack matrices attack closure-attack condition-audit ledger-attack green-attack cache-attack consumption-attack selftest staleness identity-attack missing; do
        echo "=== compat: $lane ==="
        code=0
        just compat "$lane" || code=$?
        printf '%s %s\n' "$lane" "$code" >> compat/generated/lanes.raw
        [ "$code" = 0 ] || echo "!! lane $lane exited $code; continuing so the rest still produce evidence"
    done
    # The three derivations, GUARDED like the lanes: unguarded under `-euo
    # pipefail` a ledger crash aborted the recipe and closure never ran, so the
    # run ended with no verdict at all. Guarded, closure still answers.
    post=0
    # No exit is appended to lanes.raw here: lanes.json is written by the first
    # of these three, so nothing downstream would ever read the other two.
    for step in lanes ledger closure; do
        echo "=== compat: $step ==="
        code=0
        just compat "$step" || code=$?
        [ "$code" = 0 ] || { post=$code; echo "!! $step exited $code"; }
    done
    # A RED closure exits non-zero and IS an answer; a derivation that died
    # leaves no closure.json at all, because the file was removed before the
    # lanes ran. Absence is the mark on the un-derived case.
    if [ ! -f compat/generated/closure.json ]; then
        echo "!! no closure.json: the verdict was never derived, so nothing here is this run's answer"
    fi
    exit "$post"

lanes:
    {{ python }} -m compat.harness.lanes

ledger:
    {{ python }} -m compat.harness.ledger

closure:
    {{ python }} -m compat.harness.closure

proc-attack:
    {{ python }} proc_attack.py

weights:
    {{ python }} -m compat.harness.provision

weights-attack:
    {{ python }} -m compat.harness.provision_attack

pins:
    {{ python }} -m compat.harness.provenance

producer:
    {{ python }} -m compat.producers.inventory

union:
    {{ python }} -m compat.producers.union

citations:
    {{ python }} -m compat.harness.citations

hf:
    {{ python }} -m compat.harness.hf_check

cases:
    {{ python }} -m compat.harness.sharded

population:
    {{ python }} -m compat.harness.population

population-attack:
    {{ python }} -m compat.harness.population_attack

observe-attack:
    {{ python }} -m compat.harness.observe_attack

reconcile:
    {{ python }} -m compat.harness.reconcile

reconcile-attack:
    {{ python }} -m compat.harness.reconcile_attack

closure-attack:
    {{ python }} -m compat.harness.closure_attack

ledger-attack:
    {{ python }} -m compat.harness.ledger_attack

green-attack:
    {{ python }} -m compat.harness.green_attack

cache-attack:
    {{ python }} -m compat.harness.cache_attack

consumption-attack:
    {{ python }} -m compat.harness.consumption_attack

condition-audit:
    {{ python }} -m compat.harness.condition_audit

missing:
    {{ python }} -m compat.harness.missing

matrices:
    {{ python }} -m compat.harness.matrices

attack:
    {{ python }} -m compat.harness.attack

vendor:
    {{ python }} -m compat.vendor.fixtures

acceptance:
    {{ python }} -m compat.vendor.acceptance

selftest:
    {{ python }} -m compat.harness.selftest

staleness:
    {{ python }} -m compat.harness.identity

identity-attack:
    {{ python }} -m compat.harness.identity_attack

corpus:
    {{ python }} -m compat.corpus.index

fixtures:
    {{ python }} -m compat.primitives.build

check:
    {{ python }} -m ruff format --check compat/
    {{ python }} -m ruff check compat/
    {{ python }} -m ty check --config-file compat/ty.toml compat
    {{ python }} -m pyrefly check -c compat/pyrefly.toml
