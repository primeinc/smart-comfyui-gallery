set windows-shell := ["bash", "-cu"]
set script-interpreter := ['bash', '-euo', 'pipefail']

export UV_PROJECT_ENVIRONMENT := ".venv-compat"

python := if os_family() == 'windows' { './.venv-compat/Scripts/python.exe' } else { './.venv-compat/bin/python' }

sync:
    uv sync --group compat --group dev

[script]
run: sync
    cd "$(git rev-parse --show-toplevel)"
    rm -f compat/generated/lanes.json
    rows=""
    for lane in check proc-attack weights weights-attack pins citations hf population population-attack vendor acceptance producer union cases observe-attack reconcile reconcile-attack matrices attack closure-attack selftest staleness missing; do
        echo "=== compat: $lane ==="
        code=0
        just compat "$lane" || code=$?
        [ -z "$rows" ] || rows="$rows,"$'\n'
        rows="$rows  \"$lane\": $code"
        [ "$code" = 0 ] || echo "!! lane $lane exited $code; continuing so the rest still produce evidence"
    done
    printf '{\n%s\n}\n' "$rows" > compat/generated/lanes.json.part
    mv -f compat/generated/lanes.json.part compat/generated/lanes.json
    echo "=== compat: ledger ==="
    just compat ledger
    echo "=== compat: closure ==="
    just compat closure

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

corpus:
    {{ python }} -m compat.corpus.index

fixtures:
    {{ python }} -m compat.primitives.build

check:
    {{ python }} -m ruff format --check compat/
    {{ python }} -m ruff check compat/
    {{ python }} -m ty check --config-file compat/ty.toml compat
    {{ python }} -m pyrefly check -c compat/pyrefly.toml
